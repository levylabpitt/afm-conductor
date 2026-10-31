// Boot module for the edit cockpit (docs/COCKPIT_MODULES.md). The command log
// (logLine/stamp/escapeHtml/num) lives in ec-log.js; the network + websocket
// surface (api/request/connectWs + the onWs registry) in ec-api.js. Everything
// else is still inline here, carved out one area at a time by later slices.
import { logLine, num, escapeHtml } from "./ec-log.js";
import {
  api, connectWs, onWs, onWsDefault, latestOnly,
  getDesign, postDesignSvg, postDesignOp,
  postDesignReorder, postDesignChiral,
  postLithoToolpath, getScanLast, getScanContinuous, getMode, getState,
  getPorts, postPortAdd, postPortUpdate, postPortDelete,
  getDrift, postDriftEnable, postDriftReference,
  getDesignBackgroundChannels, postDesignBackground,
} from "./ec-api.js";
import {
  fitTransform as geomFitTransform,
  invTransform as geomInvTransform,
  pxPerUm as geomPxPerUm,
  boundsOfObjects as geomBoundsOfObjects,
  percentileRange,
} from "./ec-geom.js";
import { state, emit } from "./ec-state.js";
import { isRectBezier } from "./ec-bezier.js";
import { chiralSampler } from "./ec-measure.js";
import {
  initEditTab, drawEditLayer,
  editPointerDown, editPointerMove, editPointerUp, editDblClick, editKeydown,
  editAbortDrafts, editResetOnSelect, editReloadOnSelect,
} from "./ec-tab-edit.js";
import { initViewNav } from "./ec-viewnav.js";
import {
  initWritten, onObjectStart, onRunDone, recordStroke, strokeShown, clearHistory,
} from "./ec-written.js";
import { initPanes, initSideScroll } from "./ec-panes.js";
import { initRenderPane, applyRenderTab, isHalfArrangement } from "./ec-renderpane.js";
import { scopeXMap, rangeOver, zScaleLabels, dragZRange } from "./ec-render-layout.js";
import { cssVar, cssVarA } from "./ec-theme.js";
import { initDesignFiles } from "./ec-design-files.js";
import {
  initInstrument, hydrateInstrument, hydrateScanSetpoint,
  refreshWriteEstimate, getAmpGain, markWriting,
  getVoltageWindowMax, refreshWriteGate, writeSetpointEmpty, showOtherRefusal,
  showFailure, guardNoticeMark, clearGuardNotice, getModelNumber, getXyRangeUm,
} from "./ec-instrument.js";

"use strict";
const $ = (id) => document.getElementById(id);

// Session counters kept here: ws frames seen, and live scan lines this frame.
let msgCount = 0, lineCount = 0;

// =========================================================================
//  VOLTAGE COLORMAP -- mirrors litho-debug.html voltageColor(). Symmetric
//  about 0: green for positive, red for negative, dark/olive at zero.
//    u = v / Vmax clamped to [-1,1]
//    R = 255*(1-u)/2 ; G = 255*(1+u)/2 ; B = 0
// =========================================================================
function voltageColor(v, vmax) {
  const u = vmax > 0 ? Math.max(-1, Math.min(1, v / vmax)) : 0;
  const R = Math.round(255 * (1 - u) / 2);
  const G = Math.round(255 * (1 + u) / 2);
  return [R, G, 0];
}
function rgb(c) { return `rgb(${c[0]},${c[1]},${c[2]})`; }
function rgba(c, a) { return `rgba(${c[0]},${c[1]},${c[2]},${a})`; }

// =========================================================================
//  MOCK DATA -- MATCHES the real backend design-object shape so a future
//  /api/design/svg import can populate this same array unchanged. Each object:
//    { name, kind, voltage, voltage_end, speed_um_s, points_um, bezier? }
//  kind is one of the real backend kinds; displayKind maps it to the UI label:
//    "path"      -> "path"      (a wire / polyline)
//    "electrode" -> "funnel"    (a recognized funnel outline)
//    "shape"     -> "rectangle" if its bezier is an as_rect rectangle, else
//                   "region"    (a generic closed region -- how the compiler
//                                actually fills a hand-drawn blob or an
//                                unrecognized funnel: a region fill, not a sweep)
// =========================================================================
function displayKind(o) {
  const k = (o && typeof o === "object") ? o.kind : o;   // tolerate a bare kind
  if (k === "electrode") return "funnel";
  if (k === "path") return "path";
  if (k === "shape") {
    const b = o && typeof o === "object" ? o.bezier : null;
    return (b && isRectBezier(b)) ? "rectangle" : "region";
  }
  return k;
}

// The shared model -- objects, selection, view, scan frame, ports, litho/drift
// live-state, active tab/tool, designImported, source label -- now lives in
// ec-state.js's `state` object (imported above) so tab controllers can reassign
// it across module boundaries. Referenced as state.X throughout this file.

// ---- CONNECT-TAB electrode-port drafts (see the "CONNECT / ELECTRODE PORTS"
// block far below). placingPort/portMoveDrag/portDirDrag/lastPortDown are
// Connect-only transients, gated on state.currentTab === "connect".
let placingPort = false;       // "place electrode" mode: next scan click drops a port
let portMoveDrag = null;       // dragging a pin  { id, orig:[x,y], moved }
let portDirDrag = null;        // dragging the arrow-tip rotate handle { id }
let lastPortDown = { id: null, t: 0 };  // double-click (=angle) vs single (=move) detect

// Vmax over all objects (drives the diverging color scale).
function patternVmax() {
  let vmax = 0;
  for (const o of state.objects) {
    const a = Math.abs(o.voltage); if (a > vmax) vmax = a;
    // a modulated wire swings to v0 +/- v_k -- let the colour scale span that
    // range too, so the modulation isn't clipped to a flat green/red.
    if (o.v_profile) for (const v of o.v_profile) {
      const b = Math.abs(v); if (b > vmax) vmax = b;
    }
  }
  // Floor the scale at the instrument's tip-voltage window (once the model is
  // loaded) so a voltage's colour is stable against the physically allowed range,
  // not just the current design's max -- otherwise the single highest-voltage
  // object always shows the same pure colour and a voltage edit on it is invisible
  // (docs/INSTRUMENT_MODEL_SPEC.md sec 4: a mutation must be visible in the list).
  const winMax = getVoltageWindowMax();
  if (winMax > vmax) vmax = winMax;
  return vmax || 1;
}

// =========================================================================
//  VIEW TRANSFORM -- a micron->pixel mapping fitted to a micron bounding box,
//  preserving aspect ratio and centering with a small margin. Both zoom
//  buttons are REAL client-side fits: they set `view` to a bbox and redraw.
// =========================================================================
const scanCv = $("scan"), scanCx = scanCv.getContext("2d");
const ovCv = $("overlay"), ovCx = ovCv.getContext("2d");
// view / scanFrame / scanImg / lastScanCfg now live on `state` (ec-state.js).
let lastScopeLine = null;   // most recent line's raw {height,retrace,error,error_retrace}
let scopeYRange = null;     // eased {key,lo,hi} so the scope doesn't jitter line to line
let scanImgRect = null;     // the scan frame's box on the scan canvas, px {l,t,r,b}

// Write one scan line into a channel grid at row si (NaN where absent/bad).
function fillScanRow(grid, arr, si, P){
  if (!grid) return;
  for (let i = 0; i < P; i++){
    const v = arr ? arr[i] : NaN;
    grid[si * P + i] = (v == null || Number.isNaN(v)) ? NaN : v;
  }
}

// Thin adapter: the µm bounding box of the current `objects` (the "zoom to all"
// fit). The math lives in ec-geom.js; this binds it to the live objects list.
function boundsOfObjects() { return geomBoundsOfObjects(state.objects); }

// Thin adapters over ec-geom's pure view-transform math: bind it to the live
// `view` (falling back to SCAN_FRAME) and the overlay canvas backing store.
// fitTransform takes the target pixel size; invTransform/pxPerUm read the overlay
// canvas. Geometry is authored in microns, so every edit tool converts the mouse
// through invTransform first.
function fitTransform(px, py) { return geomFitTransform(state.view || state.SCAN_FRAME, px, py); }
function invTransform() { return geomInvTransform(state.view || state.SCAN_FRAME, ovCv.width, ovCv.height); }
function pxPerUm() { return geomPxPerUm(state.view || state.SCAN_FRAME, ovCv.width, ovCv.height); }
// A pointer event -> [x_um, y_um]. clientX/Y is CSS space; the backing store may
// be a different size, so scale into backing-store pixels before inverting.
function evToUm(ev) {
  const r = ovCv.getBoundingClientRect();
  const sx = ovCv.width / (r.width || 1), sy = ovCv.height / (r.height || 1);
  const px = (ev.clientX - r.left) * sx, py = (ev.clientY - r.top) * sy;
  return invTransform()(px, py);
}

// Keep the two canvases' backing store sized to their CSS box (crisp lines).
function syncCanvasSize() {
  const r = scanCv.parentElement.getBoundingClientRect();
  for (const c of [scanCv, ovCv]) {
    if (c.width !== Math.round(r.width) || c.height !== Math.round(r.height)) {
      c.width = Math.round(r.width); c.height = Math.round(r.height);
    }
  }
}

// Build the offscreen grayscale image of a loaded scan frame. The frame's
// height is row-major with slow_idx = row 0 = BOTTOM (increasing µm-y); an
// offscreen image's row 0 is the TOP, so offscreen row r reads height row
// rows-1-r -- higher µm-y ends up at the image top. Greyscale maps a robust
// 2nd..98th percentile display range (see percentileRange): values <= p2 clamp
// to black, >= p98 to white. This is Gwyddion's robust-range behaviour, not raw
// min->max, so the error channel's edge spikes / dropouts stop compressing the
// real signal into one grey level (docs/TECH_DEBT.md #1). NaN/null pixels are
// black. Stored in `scanImg`.
// Which channel the scan image / scope currently shows. The selector value maps
// straight to a grid key on scanFrame ("height" = ZSensor topography, "error" =
// Deflection). A channel with no accumulated grid just renders empty.
function scanChannel() { const s = $("scanChan"); return s ? s.value : "height"; }
// The error channel is MODE-AWARE (matches hw_scan._err_chan / passes err_name):
// contact -> Deflection, AC/tapping -> Amplitude, current -> Current. The scan captures
// + displays whichever the mode dictates, so the label/legend must follow the mode.
function errChanName() {
  const m = ($("mode") && $("mode").value) || "contact";
  return { ac: "Amplitude", current: "Current" }[m] || "Deflection";
}
function scanChannelLabel() { return scanChannel() === "error" ? errChanName() : "ZSensor"; }
// Relabel the #scanChan "error" option to the current mode's channel so the dropdown
// reads "Amplitude (error)" in AC, "Deflection (error)" in contact, etc.
function syncScanChanLabels() {
  const sc = $("scanChan"); if (!sc) return;
  const opt = sc.querySelector('option[value="error"]');
  if (opt) opt.textContent = errChanName() + " (error)";
  const lbl = $("scopeLabel");
  if (lbl && scanChannel() === "error") lbl.textContent = "line scope · " + scanChannelLabel();
}
function scanChannelGrid(f) { return (f && f[scanChannel()]) || []; }

// `rows`, when given, marks the rows to draw (1); the others draw as no data.
function buildScanImg(f, rows = null) {
  const P = f.pixels, R = f.rows, grid = scanChannelGrid(f);
  const h = rows ? Float32Array.from(grid, (v, k) => (rows[Math.floor(k / P)] ? v : NaN)) : grid;
  // Robust 2..98 percentile display range over the finite samples. Height is
  // already tight so p2/p98 ~ its min/max (looks unchanged); the error channel's
  // outliers are clamped instead of blowing out the range. No finite samples ->
  // a harmless [0,1] (every pixel is NaN -> black anyway).
  const auto = percentileRange(h, 2, 98);
  // A grey range set by hand on the z bar holds while the same channel is shown;
  // showing another channel (or the error channel of another mode) returns to auto.
  if (state.zManual && state.zManual.chan !== scanChannelLabel()) state.zManual = null;
  const pr = state.zManual ? [state.zManual.lo, state.zManual.hi] : auto;
  const [lo, hi] = pr || [0, 1];
  state.scanRange = pr ? { lo, hi, chan: scanChannel(), manual: !!state.zManual } : null;
  const span = (hi - lo) || 1;
  const off = document.createElement("canvas");
  off.width = P; off.height = R;
  const ox = off.getContext("2d");
  const img = ox.createImageData(P, R);
  for (let r = 0; r < R; r++) {
    const srcRow = R - 1 - r;                 // flip: bottom row -> image bottom
    for (let i = 0; i < P; i++) {
      const v = h[srcRow * P + i];
      const g = (v == null || Number.isNaN(v))
        ? 0 : Math.max(0, Math.min(255, Math.round(((v - lo) / span) * 255)));
      const o = (r * P + i) * 4;
      img.data[o] = img.data[o + 1] = img.data[o + 2] = g; img.data[o + 3] = 255;
    }
  }
  ox.putImageData(img, 0, 0);
  state.scanImg = off;
}

// ---- scan view `live | last | both` (display only). "last" is the most recent
//      scanned grid in whatever state it ended: any amount of scanning is shown, so a
//      partial or aborted frame counts. The page copies the grid on
//      every scan_done, aborted or not, and at a new frame's start when the previous
//      frame got no scan_done (state.lastFrame). It marks the rows the current frame
//      has received (scanFrame.liveRows). live: those rows only; last: that copy;
//      both: the current grid, whose rows not yet rescanned still hold the previous
//      frame when the geometry is unchanged. ----
let frameEnded = false;    // scan_done arrived (and was copied): the next line starts a new frame
// The row being scanned (slow_idx), or null when no scan is running: set by each line,
// cleared by scan_done and by a websocket close (a lost scan_done must not leave it).
// Display only.
let scanRowNow = null;
function hasData(f) {
  return !!(f && f.height && Array.prototype.some.call(f.height, (v) => Number.isFinite(v)));
}
function snapshotFrame() {
  const f = state.scanFrame;
  if (!hasData(f)) return;
  state.lastFrame = { pixels: f.pixels, rows: f.rows, size_um: f.size_um,
    x_offset_um: f.x_offset_um, y_offset_um: f.y_offset_um, angle_deg: f.angle_deg || 0,
    height: Float32Array.from(f.height), error: Float32Array.from(f.error || []) };
}
// A new frame starts: keep the one before it unless its scan_done already did.
function frameStarts() { if (!frameEnded) snapshotFrame(); }
function startLiveRows() {
  state.scanFrame.liveRows = new Uint8Array(state.scanFrame.rows);
  frameEnded = false;
}
function shownFrame() { return state.scanView === "last" ? state.lastFrame : state.scanFrame; }
function rebuildScanImg() {
  const f = shownFrame();
  if (!f || !f.height) { state.scanImg = null; state.scanRange = null; return; }
  buildScanImg(f, state.scanView === "live" ? (f.liveRows || new Uint8Array(f.rows)) : null);
}
function setScanView(v) {
  state.scanView = v;
  for (const b of document.querySelectorAll("#scanViewSeg .seg-btn")) {
    const on = b.dataset.view === v;
    b.classList.toggle("active", on);
    b.setAttribute("aria-pressed", on ? "true" : "false");
  }
  rebuildScanImg();
  redraw();
}

// Trace/retrace line scope (Asylum's line view): the current line of the channel
// #scanChan selects (ZSensor -> height / retrace, error -> error / error_retrace).
// Its x axis is the image's: pixel i sits under image column i, so the plot follows
// zoom and pan and is clipped to the image area. The z axis autoscales to the part of
// the line in view, eased so the sweep does not jump line to line.
const SCOPE_AXIS_PX = 48;
function scopeShown() { const sp = $("scopePanel"); return !!sp && sp.style.display !== "none"; }
function drawScope(){
  const cv = $("scope"); if (!cv || !scopeShown()) return;
  const r = cv.getBoundingClientRect();
  const W = Math.max(1, Math.round(r.width)), H = Math.max(1, Math.round(r.height));
  if (cv.width !== W || cv.height !== H){ cv.width = W; cv.height = H; }
  const cx = cv.getContext("2d");
  const chan = scanChannel();
  const lbl = $("scopeLabel"); if (lbl) lbl.textContent = "line scope · " + scanChannelLabel();
  cx.clearRect(0, 0, W, H);
  cx.fillStyle = cssVar("--panel"); cx.fillRect(0, 0, W, H);
  // the image area (scan canvas) in scope coordinates: the plot's clip
  const sr = scanCv.getBoundingClientRect();
  const clipL = Math.max(0, sr.left - r.left), clipR = Math.min(W, sr.right - r.left);
  const line = lastScopeLine;
  const trace   = line ? (chan === "error" ? line.error : line.height) : null;
  const retrace = line ? (chan === "error" ? line.error_retrace : line.retrace) : null;
  const n = (trace && trace.length) || (retrace && retrace.length) || 0;
  // the scan frame's x extent on screen; without one the line spans the image area
  const imgL = scanImgRect ? sr.left - r.left + scanImgRect.l : clipL;
  const imgR = scanImgRect ? sr.left - r.left + scanImgRect.r : clipR;
  const map = scopeXMap(imgL, imgR, n, clipL, clipR);
  // the frame and grid span the scan image's on-screen x range inside the clip, so
  // they stop where the scan data stops, with or without a line to draw
  const plotL = Math.max(clipL, imgL), plotR = Math.min(clipR, imgR);
  const rng = $("scopeRange");
  const yr = map ? rangeOver([trace, retrace], map.i0, map.i1) : null;
  if (!yr) {
    drawScopeGrid(cx, plotL, plotR, H);
    if (rng) rng.textContent = "—";
    scopeYRange = null;
    return;
  }
  let [lo, hi] = yr;
  if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }   // a flat line still gets height
  // ease only while the channel and the visible pixels stay the same; a zoom or pan
  // rescales at once
  const key = `${chan}:${map.i0}:${map.i1}`;
  const prev = scopeYRange && scopeYRange.key === key ? scopeYRange : null;
  if (prev) {
    lo = Math.min(lo, prev.lo + (lo - prev.lo) * 0.3);
    hi = Math.max(hi, prev.hi + (hi - prev.hi) * 0.3);
  }
  scopeYRange = { key, lo, hi };
  const pad = (hi - lo) * 0.08, dlo = lo - pad, dhi = hi + pad;
  drawScopeGrid(cx, plotL, plotR, H);
  const unit = chan === "error" ? "V" : "nm";
  if (rng) rng.textContent = `${lo.toFixed(2)} … ${hi.toFixed(2)} ${unit}`;
  const span = (dhi - dlo) || 1;
  const yOf = (v) => H - ((v - dlo) / span) * (H - 2) - 1;
  cx.save();
  cx.beginPath(); cx.rect(plotL, 0, plotR - plotL, H); cx.clip();
  const plot = (arr, colour) => {
    if (!arr || arr.length < 2) return;
    cx.strokeStyle = colour; cx.lineWidth = 1.25; cx.beginPath();
    let started = false;
    // one pixel past each edge so the trace reaches the clip
    for (let i = Math.max(0, map.i0 - 1); i <= Math.min(arr.length - 1, map.i1 + 1); i++){
      const v = arr[i];
      if (v == null || Number.isNaN(v)) { started = false; continue; }
      const x = map.x(i), y = yOf(v);
      if (started) cx.lineTo(x, y); else { cx.moveTo(x, y); started = true; }
    }
    cx.stroke();
  };
  plot(retrace, cssVar("--fg"));   // retrace under trace
  plot(trace, cssVar("--accent"));
  cx.restore();
  // z axis: min, mid, max of the drawn range; left of the plot when there is room,
  // otherwise (and always at half screen, where the plot spans the width) just inside it
  const inside = isHalfArrangement() || plotL < SCOPE_AXIS_PX;
  cx.font = `10px ${cssVar("--mono")}`;
  cx.fillStyle = cssVar("--dim");
  cx.textAlign = inside ? "left" : "right";
  const ax = inside ? plotL + 3 : plotL - 4;
  for (const t of zScaleLabels(lo, hi)) {
    const v = lo + t.frac * (hi - lo);
    cx.textBaseline = t.frac === 1 ? "top" : t.frac === 0 ? "bottom" : "middle";
    cx.fillText(t.text, ax, Math.max(1, Math.min(H - 1, yOf(v))));
  }
  if (!inside) { cx.textBaseline = "top"; cx.fillText(unit, ax, H / 2 + 8); }
}
function drawScopeGrid(cx, x0, x1, H){
  const w = x1 - x0; if (!(w > 0)) return;
  cx.strokeStyle = cssVar("--edge"); cx.lineWidth = 1;
  cx.beginPath();
  for (let k = 0; k <= 4; k++){ const y = Math.round((k / 4) * (H - 1)) + 0.5; cx.moveTo(x0, y); cx.lineTo(x1, y); }
  for (let k = 0; k <= 6; k++){ const x = Math.round(x0 + (k / 6) * (w - 1)) + 0.5; cx.moveTo(x, 0); cx.lineTo(x, H); }
  cx.stroke();
}

// The z scale bar: the grey ramp the image is drawn with (black at the range's low
// end, white at its high end) and its min, mid and max. It is the colour mapping, so
// it does not follow zoom; it only sits level with the visible part of the image.
function drawZScale() {
  const cv = $("zScaleCv"); if (!cv) return;
  const r = cv.getBoundingClientRect();
  const W = Math.max(1, Math.round(r.width)), H = Math.max(1, Math.round(r.height));
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const cx = cv.getContext("2d");
  cx.clearRect(0, 0, W, H);
  const rg = state.scanFrame && state.scanImg ? state.scanRange : null;
  zBar = null;
  if (!rg) { if ($("zAuto")) $("zAuto").hidden = true; return; }
  const unit = rg.chan === "error" ? "V" : "nm";
  const sr = scanCv.getBoundingClientRect();
  if (isHalfArrangement()) { drawZScaleAcross(cx, W, H, r, sr, rg, unit); return; }
  let top = 0, bot = sr.height;
  if (scanImgRect) { top = Math.max(0, scanImgRect.t); bot = Math.min(sr.height, scanImgRect.b); }
  top += sr.top - r.top + 22; bot += sr.top - r.top - 6;
  if (bot - top < 40) { top = 22; bot = H - 6; }
  bot -= Z_AUTO_PX;                              // room for the auto button under the ramp
  const rampW = 12, rampX = 2, y0 = Math.round(top), h = Math.max(2, Math.round(bot - top));
  const img = cx.createImageData(rampW, h);
  for (let y = 0; y < h; y++) {
    const g = Math.round(255 * (1 - y / (h - 1)));
    for (let x = 0; x < rampW; x++) {
      const o = (y * rampW + x) * 4;
      img.data[o] = img.data[o + 1] = img.data[o + 2] = g; img.data[o + 3] = 255;
    }
  }
  cx.putImageData(img, rampX, y0);
  cx.strokeStyle = cssVar("--edge"); cx.lineWidth = 1;
  cx.strokeRect(rampX - 0.5, y0 - 0.5, rampW + 1, h + 1);
  cx.font = `10px ${cssVar("--mono")}`; cx.fillStyle = cssVar("--fg");
  cx.textAlign = "left"; cx.textBaseline = "middle";
  for (const t of zScaleLabels(rg.lo, rg.hi))
    cx.fillText(t.text, rampX + rampW + 5, y0 + (1 - t.frac) * (h - 1));
  cx.fillStyle = cssVar("--dim"); cx.textBaseline = "bottom";
  cx.fillText(`Z ${unit}`, rampX, y0 - 6);
  drawZGrips(cx, rg.manual, (a) => [rampX - 1, a, rampX + rampW + 3, a], y0 + h - 1, y0);
  setZBar(true, y0 + h - 1, y0);
  placeZAuto(rg.manual, rampX, y0 + h + 6);
}
// The z bar laid across the top of the image (half screen): low on the left, high on
// the right, level with the visible part of the image, labels under the ramp.
function drawZScaleAcross(cx, W, H, r, sr, rg, unit) {
  cx.font = `10px ${cssVar("--mono")}`;
  const capW = Math.ceil(cx.measureText(`Z ${unit}`).width) + 8;
  let left = 0, right = sr.width;
  if (scanImgRect) { left = Math.max(0, scanImgRect.l); right = Math.min(sr.width, scanImgRect.r); }
  left += sr.left - r.left; right += sr.left - r.left;
  const autoW = 44;                              // the auto button sits right of the ramp
  left = Math.max(left, capW); right = Math.min(right, W - 2 - autoW);
  if (right - left < 60) { left = capW; right = W - 2 - autoW; }
  const rampH = 10, y0 = 3, x0 = Math.round(left), w = Math.max(2, Math.round(right - left));
  const img = cx.createImageData(w, rampH);
  for (let x = 0; x < w; x++) {
    const g = Math.round(255 * (x / (w - 1)));
    for (let y = 0; y < rampH; y++) {
      const o = (y * w + x) * 4;
      img.data[o] = img.data[o + 1] = img.data[o + 2] = g; img.data[o + 3] = 255;
    }
  }
  cx.putImageData(img, x0, y0);
  cx.strokeStyle = cssVar("--edge"); cx.lineWidth = 1;
  cx.strokeRect(x0 - 0.5, y0 - 0.5, w + 1, rampH + 1);
  cx.fillStyle = cssVar("--fg"); cx.textBaseline = "top";
  for (const t of zScaleLabels(rg.lo, rg.hi)) {
    cx.textAlign = t.frac === 0 ? "left" : t.frac === 1 ? "right" : "center";
    cx.fillText(t.text, x0 + t.frac * w, y0 + rampH + 4);
  }
  cx.fillStyle = cssVar("--dim"); cx.textAlign = "left"; cx.textBaseline = "middle";
  cx.fillText(`Z ${unit}`, 0, y0 + rampH / 2);
  drawZGrips(cx, rg.manual, (a) => [a, y0 - 1, a, y0 + rampH + 1], x0, x0 + w - 1);
  setZBar(false, x0, x0 + w - 1);
  placeZAuto(rg.manual, x0 + w + 8, 0);
}

// ---- manual z range: drag either end of the z bar; "auto" goes back to the
//      2nd..98th percentile clip. Display only: nothing is sent to the service. ----
const Z_AUTO_PX = 24;       // height kept under the upright ramp for the auto button
const Z_GRIP_PX = 10;       // how close to a ramp end a press grabs it
let zBar = null;            // the drawn ramp: {vertical, lo, hi} ends in bar-canvas px
let zDrag = null;           // {end, start:{lo,hi}, at, len} while an end is dragged
// A grip across each end of the ramp: accent once the range is set by hand.
function drawZGrips(cx, manual, seg, aLo, aHi) {
  cx.strokeStyle = cssVar(manual ? "--accent" : "--dim"); cx.lineWidth = 2;
  cx.beginPath();
  for (const a of [aLo, aHi]) { const [x0, y0, x1, y1] = seg(a); cx.moveTo(x0, y0); cx.lineTo(x1, y1); }
  cx.stroke();
}
// The grabbable ends, also on the canvas's data attributes (data-axis, data-lo-px,
// data-hi-px), so the page can be driven by its geometry.
function setZBar(vertical, lo, hi) {
  zBar = { vertical, lo, hi };
  const cv = $("zScaleCv");
  cv.dataset.axis = vertical ? "y" : "x"; cv.dataset.loPx = String(lo); cv.dataset.hiPx = String(hi);
}
function placeZAuto(manual, left, top) {
  const b = $("zAuto"); if (!b) return;
  b.hidden = false;
  b.style.left = `${left}px`; b.style.top = `${top}px`;
  b.classList.toggle("active", !manual);
  b.setAttribute("aria-pressed", String(!manual));
}
function zEndAt(ev) {
  if (!zBar || !state.scanRange) return null;
  const r = $("zScaleCv").getBoundingClientRect();
  const a = zBar.vertical ? ev.clientY - r.top : ev.clientX - r.left;
  const dLo = Math.abs(a - zBar.lo), dHi = Math.abs(a - zBar.hi);
  if (Math.min(dLo, dHi) > Z_GRIP_PX) return null;
  return dHi <= dLo ? "hi" : "lo";
}
function zAxis(ev) { return zBar.vertical ? ev.clientY : ev.clientX; }
let zFrameQueued = false;
function setZManual(rg) {
  state.zManual = rg ? { lo: rg.lo, hi: rg.hi, chan: scanChannelLabel() } : null;
  if (zFrameQueued) return;
  zFrameQueued = true;
  requestAnimationFrame(() => {
    zFrameQueued = false;
    rebuildScanImg();
    redraw();
  });
}
function initZScaleDrag() {
  const cv = $("zScaleCv"); if (!cv) return;
  cv.addEventListener("pointerdown", (ev) => {
    const end = ev.button === 0 ? zEndAt(ev) : null;
    if (!end) return;
    ev.preventDefault();
    zDrag = { end, start: { lo: state.scanRange.lo, hi: state.scanRange.hi },
              at: zAxis(ev), len: Math.max(1, Math.abs(zBar.hi - zBar.lo)), vertical: zBar.vertical };
    try { cv.setPointerCapture(ev.pointerId); } catch (e) {}
  });
  cv.addEventListener("pointermove", (ev) => {
    if (!zDrag) {
      const end = zEndAt(ev);
      cv.style.cursor = end ? (zBar.vertical ? "ns-resize" : "ew-resize") : "";
      return;
    }
    // the high end is up on the upright bar and right on the horizontal one
    const d = (zAxis(ev) - zDrag.at) / zDrag.len;
    setZManual(dragZRange(zDrag.start, zDrag.end, zDrag.vertical ? -d : d));
  });
  const end = (ev) => {
    if (!zDrag) return;
    zDrag = null;
    try { cv.releasePointerCapture(ev.pointerId); } catch (e) {}
  };
  cv.addEventListener("pointerup", end);
  cv.addEventListener("pointercancel", end);
  const b = $("zAuto");
  if (b) b.addEventListener("click", () => { if (state.zManual) setZManual(null); });
}

// ec-safety.js publishes the fault latch on <html data-fault-latched> (see there).
function faultLatched() { return document.documentElement.dataset.faultLatched === "1"; }

// A thin bare line across the image at the row being scanned, while a scan runs. Not
// in the "last" view (the row is not on it) nor while the fault latch holds (a latched
// fault refuses scanning, so a marker would claim a scan that is not happening).
function drawScanLineMarker(f, l, t, r, b) {
  if (scanRowNow == null || f !== state.scanFrame || state.scanView === "last"
      || faultLatched()) return;
  const R = f.rows; if (!(R > 0) || scanRowNow < 0 || scanRowNow >= R) return;
  // slow_idx 0 is the bottom row; the image's top row is R-1
  const y = Math.round(t + (R - 1 - scanRowNow + 0.5) * (b - t) / R) + 0.5;
  scanCx.strokeStyle = cssVar("--scanline"); scanCx.lineWidth = 1;
  scanCx.beginPath(); scanCx.moveTo(l, y); scanCx.lineTo(r, y); scanCx.stroke();
  scanCv.dataset.scanRow = String(scanRowNow);
}

// Scan background. With a loaded frame: paint the grayscale scan image into the
// µm-box its offsets/size define, using the SAME fitTransform() the overlay uses
// so the design lands in register. Without one: the dark micron-grid placeholder.
function drawScan() {
  const W = scanCv.width, H = scanCv.height;
  const tf = fitTransform(W, H);
  const f = shownFrame();
  delete scanCv.dataset.scanRow;          // set again by drawScanLineMarker when drawn
  if (f && state.scanImg) {
    const x0 = f.x_offset_um - f.size_um / 2, y0 = f.y_offset_um - f.size_um / 2;
    const x1 = x0 + f.size_um, y1 = y0 + f.size_um;
    const [dxl, dyt] = tf(x0, y1);            // top-left in pixels (µm-y up)
    const [dxr, dyb] = tf(x1, y0);            // bottom-right in pixels
    scanCx.clearRect(0, 0, W, H);
    scanCx.fillStyle = cssVar("--surround");
    scanCx.fillRect(0, 0, W, H);
    scanCx.imageSmoothingEnabled = false;
    scanCx.drawImage(state.scanImg, dxl, dyt, dxr - dxl, dyb - dyt);
    scanImgRect = { l: dxl, t: dyt, r: dxr, b: dyb };
    drawScanLineMarker(f, dxl, dyt, dxr, dyb);
    return;
  }
  // -- placeholder: dark canvas with a faint 1 µm grid (no scan loaded) --
  scanImgRect = null;
  scanCx.clearRect(0, 0, W, H);
  scanCx.fillStyle = cssVar("--surround");
  scanCx.fillRect(0, 0, W, H);
  const v = state.view || state.SCAN_FRAME;
  scanCx.strokeStyle = cssVar("--edge");
  scanCx.lineWidth = 1;
  const step = 1; // 1 micron grid
  const gx0 = Math.ceil(v.x0), gx1 = Math.floor(v.x0 + v.w);
  const gy0 = Math.ceil(v.y0), gy1 = Math.floor(v.y0 + v.h);
  scanCx.beginPath();
  for (let gx = gx0; gx <= gx1; gx += step) {
    const [sx, sy0] = tf(gx, v.y0), [, sy1] = tf(gx, v.y0 + v.h);
    scanCx.moveTo(sx, sy0); scanCx.lineTo(sx, sy1);
  }
  for (let gy = gy0; gy <= gy1; gy += step) {
    const [sx0, sy] = tf(v.x0, gy), [sx1] = tf(v.x0 + v.w, gy);
    scanCx.moveTo(sx0, sy); scanCx.lineTo(sx1, sy);
  }
  scanCx.stroke();
}

// =========================================================================
//  TOOLPATH PREVIEW -- the ACTUAL compiled tip path (fill passes), not the
//  authored outline. POST /api/litho/toolpath runs the real compiler on the
//  CURRENT hub.pattern (the imported design) and returns
//    { segments: [ { path_um:[[x,y],...], ... }, ... ] } in design-frame µm,
//  so the fill passes map with the SAME fitTransform() the outline uses.
//  Only meaningful after an SVG import (mock objects have no hub.pattern), so
//  the fetch is gated on `designImported`; for the mock it draws nothing.
// =========================================================================
let showTipPath = true;    // default ON -- mirrors the main UI's active "Preview toolpath" button
let toolpathSegs = null;   // cached segments (design-µm) or null
// designImported now lives on `state` (ec-state.js).

// =========================================================================
//  LIVE LITHO STATE -- state.litho, driven by the litho_* ws messages during a
//  write (see ec-state.js for the field docs). Drawn ON TOP in drawOverlay so
//  "where the tip HAS moved" is always visible.
// =========================================================================

// Coalesced (latestOnly): the compile of a dense fill takes the service seconds, and
// a fetch per design change would stack up.
const refreshToolpath = latestOnly(fetchToolpath);
async function fetchToolpath() {
  // Nothing to compile until a design has been imported into hub.pattern, and
  // no point fetching while the toggle is off.
  if (!showTipPath || !state.designImported) { toolpathSegs = null; return; }
  const res = await postLithoToolpath();
  if (!res.ok || res.data.error) { toolpathSegs = null; return; }  // fall back to the outline only
  toolpathSegs = res.data.segments || [];
}

// Draw the compiled toolpath -- every fill pass the tip ACTUALLY runs (zig-zag
// rows, funnel sweeps, the interpolated wire), each pass SOLID and coloured by
// its write bias, exactly like the main UI's blue "Preview toolpath" (≋) button
// (app-litho.js drawDesign toolpath branch). This is "the actual path in green"
// the user pointed at -- shown INSTEAD of the authored outline, NOT as a faint
// dashed overlay under it.
function drawToolpath(tf) {
  if (!toolpathSegs || !toolpathSegs.length) return;
  const vmax = patternVmax();
  ovCx.save();
  ovCx.lineCap = "round";
  // A chiral wire's compiled pass is RDP-simplified (12.5 nm), coarse relative
  // to the modulation, so it facets. Draw such objects ANALYTICALLY from their
  // profile instead -- smooth at any zoom -- and skip their toolpath segments
  // below. The analytic sine IS the path that gets written, so this is faithful.
  const chiralNames = new Set();
  for (const o of state.objects) {
    if (o.profile && o.profile.kind === "chiral"
        && drawChiralWire(o, tf, vmax, o.enabled === false, 1))
      chiralNames.add(o.name);
  }
  ovCx.lineWidth = 1;
  ovCx.setLineDash([]);
  for (const s of toolpathSegs) {
    if (chiralNames.has(s.object)) continue;   // drawn analytically above
    const path = s.path_um;
    if (!path || path.length < 2) continue;
    // A modulated pass (chiral wire) ships per-point `volts`; colour each
    // segment by its LOCAL bias so the voltage modulation shows in the tip
    // path too, not just the authored wire. Flat passes keep one colour.
    if (s.volts && s.volts.length === path.length) {
      for (let i = 1; i < path.length; i++) {
        const vm = (s.volts[i - 1] + s.volts[i]) / 2;
        ovCx.strokeStyle = rgba(voltageColor(vm, vmax), 0.85);
        ovCx.beginPath();
        const [xa, ya] = tf(path[i - 1][0], path[i - 1][1]);
        const [xb, yb] = tf(path[i][0], path[i][1]);
        ovCx.moveTo(xa, ya);
        ovCx.lineTo(xb, yb);
        ovCx.stroke();
      }
      continue;
    }
    ovCx.strokeStyle = rgba(voltageColor(s.voltage ?? 0, vmax), 0.85);
    ovCx.beginPath();
    const [x0, y0] = tf(path[0][0], path[0][1]);
    ovCx.moveTo(x0, y0);
    for (let i = 1; i < path.length; i++) {
      const [x, y] = tf(path[i][0], path[i][1]);
      ovCx.lineTo(x, y);
    }
    ovCx.stroke();
  }
  ovCx.restore();
}

// Draw a chiral wire ANALYTICALLY from its profile, so it is smooth at ANY zoom
// -- the served points_um is only a coarse fallback polyline, which facets at
// the peaks. Geometry is axis + normal*amp*sin(theta); bias is
// v0 + v_k*sin(theta + phase_rad) -- BOTH from the same theta, so the colour is
// locked to the wiggle exactly as the physics defines it. That voltage<->position
// phase relationship is the whole point of seeing a chiral wire, so it is drawn
// at the resolution the current zoom needs and coloured per segment by the bias.
function drawChiralWire(o, tf, vmax, off, lineWidth) {
  const P = o.profile;
  const at = chiralSampler(P);
  if (!at) return false;                 // missing analytic data -> polyline fallback
  const periods = P.periods || 0;
  // sample density from the CURRENT zoom: aim for ~2 px per segment so the sine
  // never facets, capped so a deep zoom stays affordable.
  const v = state.view || state.SCAN_FRAME;
  const s = Math.min(ovCv.width / v.w, ovCv.height / v.h);   // px per µm (matches fitTransform)
  const periodPx = (P.lambda_um || 1) * s;
  const perPeriod = Math.min(120, Math.max(6, Math.ceil(periodPx / 2)));
  const N = Math.max(2, Math.ceil(perPeriod * (periods || 1)));
  ovCx.setLineDash(off ? [4, 3] : []);
  ovCx.lineWidth = lineWidth;             // match the plain wires in this view
  let prev = at(0);
  for (let i = 1; i <= N; i++) {
    const cur = at(i / N);
    const vm = (prev.v + cur.v) / 2;     // colour by the mid-segment bias
    ovCx.strokeStyle = rgba(voltageColor(vm, vmax), off ? 0.5 : 0.95);
    ovCx.beginPath();
    const [xa, ya] = tf(prev.x, prev.y);
    const [xb, yb] = tf(cur.x, cur.y);
    ovCx.moveTo(xa, ya); ovCx.lineTo(xb, yb);
    ovCx.stroke();
    prev = cur;
  }
  ovCx.setLineDash([]);
  return true;
}

// Render the overlay. Two mutually-exclusive modes, matching the main UI:
//  - Preview toolpath ON (and a compiled path exists): draw the compiled passes
//    coloured by bias INSTEAD of the authored outline (the ≋ button behaviour).
//  - otherwise: draw the AUTHORED objects by kind -- shape(rect)/electrode(funnel)
//    as voltage-washed filled regions, path as an open voltage-coloured wire
//    (a chiral wire draws analytically via drawChiralWire).
// =========================================================================
//  OVERLAY PAINTER REGISTRY -- the overlay is composited by a z-ordered list of
//  painter layers instead of drawOverlay hard-coding each one. Painters run back
//  to front (design/toolpath -> litho -> edit tools -> ports); a future tab
//  module registers its own layer here rather than editing drawOverlay. Each
//  painter takes the shared µm->px transform `tf`.
// =========================================================================
const overlayPainters = [];
function registerOverlayPainter(fn) { overlayPainters.push(fn); }

// Design/toolpath layer + the selection marker/labels (mutually-exclusive modes,
// matching the main UI): Preview-toolpath ON draws the compiled passes coloured
// by bias INSTEAD of the authored outline; otherwise the AUTHORED objects by kind
// (shape/electrode filled, path as an open voltage-coloured wire; a chiral wire
// draws analytically via drawChiralWire). The selected object always gets a halo
// + name label on top so select/edit stays visible.
function drawDesignLayer(tf) {
  const showPasses = showTipPath && toolpathSegs && toolpathSegs.length;
  if (showPasses) {
    drawToolpath(tf);                       // compiled passes, coloured by bias
  } else {
    const vmax = patternVmax();
    state.objects.forEach((o, i) => {
      const pts = o.points_um;
      if (!pts || pts.length < 2) return;
      const filled = o.kind === "shape" || o.kind === "electrode";
      const off = o.enabled === false;         // excluded from the compiled path
      // A chiral wire draws from its analytic profile -- smooth at any zoom and
      // coloured continuously by the bias -- rather than the coarse served
      // polyline (which facets). Falls through if the profile lacks the data.
      if (!filled && o.profile && o.profile.kind === "chiral"
          && drawChiralWire(o, tf, vmax, off, i === state.selected ? 4 : 2.4)) return;
      // A modulated wire (a chiral wire, or any object carrying a per-point
      // bias profile) is coloured PER SEGMENT by its local bias, so the
      // voltage modulation shows in the wire's colour -- green<->red along its
      // length, like the main panel. Only for open wires, never filled regions.
      if (!filled && o.v_profile && o.v_profile.length === pts.length) {
        ovCx.setLineDash(off ? [4, 3] : []);
        ovCx.lineWidth = i === state.selected ? 4 : 2.4;
        for (let k = 1; k < pts.length; k++) {
          const vm = (o.v_profile[k - 1] + o.v_profile[k]) / 2;
          ovCx.strokeStyle = rgba(voltageColor(vm, vmax), off ? 0.5 : 0.95);
          ovCx.beginPath();
          const [xa, ya] = tf(pts[k - 1][0], pts[k - 1][1]);
          const [xb, yb] = tf(pts[k][0], pts[k][1]);
          ovCx.moveTo(xa, ya);
          ovCx.lineTo(xb, yb);
          ovCx.stroke();
        }
        ovCx.setLineDash([]);
        return;
      }
      const col = voltageColor(o.voltage, vmax);
      // disabled objects still appear, but faded + dashed so they read as "off"
      ovCx.setLineDash(off ? [4, 3] : []);
      ovCx.beginPath();
      const [x0, y0] = tf(pts[0][0], pts[0][1]);
      ovCx.moveTo(x0, y0);
      for (let k = 1; k < pts.length; k++) {
        const [x, y] = tf(pts[k][0], pts[k][1]);
        ovCx.lineTo(x, y);
      }
      if (filled) {
        ovCx.closePath();
        ovCx.fillStyle = rgba(col, off ? 0.12 : 0.28);
        ovCx.fill();
        ovCx.strokeStyle = off ? rgba(col, 0.4) : rgba(col, 1);
        ovCx.lineWidth = i === state.selected ? 3 : 1.6;
      } else {
        ovCx.strokeStyle = off ? rgba(col, 0.4) : rgb(col);
        ovCx.lineWidth = i === state.selected ? 4 : 2.4;
      }
      ovCx.stroke();
      ovCx.setLineDash([]);
    });
  }
  // Selected-object marker (halo + label) -- drawn in BOTH modes so the list
  // selection is always visible, even over the compiled passes.
  if (state.selected >= 0 && state.objects[state.selected] && (state.objects[state.selected].points_um || []).length >= 2) {
    const o = state.objects[state.selected];
    const pts = o.points_um;
    ovCx.save();
    ovCx.beginPath();
    const [hx, hy] = tf(pts[0][0], pts[0][1]);
    ovCx.moveTo(hx, hy);
    for (let k = 1; k < pts.length; k++) {
      const [x, y] = tf(pts[k][0], pts[k][1]);
      ovCx.lineTo(x, y);
    }
    if (o.kind === "shape" || o.kind === "electrode") ovCx.closePath();
    ovCx.strokeStyle = cssVarA("--accent", 0.7);
    ovCx.lineWidth = 1.5;
    ovCx.setLineDash([3, 3]);
    ovCx.stroke();
    ovCx.restore();
    const [lx, ly] = tf(pts[0][0], pts[0][1]);
    ovCx.fillStyle = cssVarA("--fg", 0.9);
    ovCx.font = `10px ${cssVar("--mono")}`;
    ovCx.fillText(o.name, lx + 5, ly - 4);
  } else {
    // labels for all objects when nothing is selected and we're in outline mode
    if (!showPasses) {
      ovCx.fillStyle = cssVarA("--fg", 0.85);
      ovCx.font = `10px ${cssVar("--mono")}`;
      state.objects.forEach((o) => {
        const pts = o.points_um; if (!pts || pts.length < 2) return;
        const [lx, ly] = tf(pts[0][0], pts[0][1]);
        ovCx.fillText(o.name, lx + 5, ly - 4);
      });
    }
  }
}

// Render the overlay: clear, compute the shared transform, then run every
// registered painter in z-order. lineCap/lineJoin are set once here because the
// design layer relies on round joins (litho/edit/ports save/restore their own).
function drawOverlay() {
  const W = ovCv.width, H = ovCv.height;
  ovCx.clearRect(0, 0, W, H);
  const tf = fitTransform(W, H);
  ovCx.lineCap = "round";
  ovCx.lineJoin = "round";
  for (const paint of overlayPainters) paint(tf);
}
// Built-in layers in z-order: design/toolpath + selection marker, then the live
// litho trace ON TOP (NEED #2), then the Edit-tool overlays (rubber rect / pen /
// node handles -- no-op off the Edit tab), then the Connect-tab ports (no-op off
// the Connect tab). drawLitho/drawPortsLayer are hoisted decls defined below;
// drawEditLayer is imported from ec-tab-edit.js. Registration order IS the draw
// order (drawEditLayer keeps its slot between litho and ports).
registerOverlayPainter(drawDesignLayer);
registerOverlayPainter(drawLitho);
// Edit-tab READ-ONLY electrode reference markers (EDIT_TOOLS §4), UNDER the edit
// tools so the wire/handles being routed draw on top. No-op off the Edit tab and
// with the "Show electrodes" toggle off. drawEditElectrodesLayer is a hoisted decl.
registerOverlayPainter(drawEditElectrodesLayer);
registerOverlayPainter(drawEditLayer);
registerOverlayPainter(drawPortsLayer);

// Draw the live/accumulated litho over the current view. Everything maps with
// the same `tf` -- in the twin the litho/lab frame is aligned with the design
// frame; a rotated lab frame (lab != design) is a known TODO, like the
// scan-angle note in loadLastScan().
function drawLitho(tf) {
  // tipHidden is set only on the twin, where "show tip" hides the marker in this
  // page; on hardware the service stops the tip_pos stream instead.
  const tipTrail = state.litho.tipHidden ? [] : (state.litho.tipTrail || []);
  // the `written` checkbox hides past writes only: the stroke being written and the
  // tip marker (it has its own `tip` box) always show
  const written = state.litho.written.filter(strokeShown);
  const live = state.litho.live;
  if (!written.length && !live.length && !state.litho.tip && !tipTrail.length) return;
  const vmax = patternVmax();
  ovCx.save();
  ovCx.lineCap = "round";
  ovCx.lineJoin = "round";
  // -- accumulated WRITTEN paths: each completed object's decimated path, per
  //    segment coloured by its local v_written when the lengths line up (same
  //    approach the chiral / v_profile wire uses), else the single voltage. --
  for (const w of written) {
    const path = w.path_um;
    if (!path || path.length < 2) continue;
    const perSeg = w.v_written && w.v_written.length === path.length;
    ovCx.lineWidth = 2.4;
    // An abort-cut object (litho_object_partial) rides in this same array,
    // tagged `partial` -- draw it dashed and dimmer, the same "not fully
    // committed" language the disabled-object dashing elsewhere uses, so it
    // reads as distinct from a completed stroke without a second overlay.
    ovCx.setLineDash(w.partial ? [4, 3] : []);
    const alpha = w.partial ? 0.6 : 0.95;
    if (perSeg) {
      for (let i = 1; i < path.length; i++) {
        const vm = (w.v_written[i - 1] + w.v_written[i]) / 2;
        ovCx.strokeStyle = rgba(voltageColor(vm, vmax), alpha);
        ovCx.beginPath();
        const [xa, ya] = tf(path[i - 1][0], path[i - 1][1]);
        const [xb, yb] = tf(path[i][0], path[i][1]);
        ovCx.moveTo(xa, ya); ovCx.lineTo(xb, yb);
        ovCx.stroke();
      }
    } else {
      ovCx.strokeStyle = rgba(voltageColor(w.voltage ?? 0, vmax), alpha);
      ovCx.beginPath();
      const [x0, y0] = tf(path[0][0], path[0][1]);
      ovCx.moveTo(x0, y0);
      for (let i = 1; i < path.length; i++) {
        const [x, y] = tf(path[i][0], path[i][1]);
        ovCx.lineTo(x, y);
      }
      ovCx.stroke();
    }
    ovCx.setLineDash([]);
  }
  // -- live tip POINT CLOUD: the intermediate measured positions as they stream
  //    in during the write, drawn as unconnected POINTS. The tip is sampled
  //    sparsely (4-20 Hz), so joining the samples with a line would fake motion
  //    detail we never measured -- the tip moves smoothly between them. On
  //    litho_object_done this cloud is REPLACED by the full-resolution connected
  //    written trace (litho.written), which is the real path. --
  for (let i = 0; i < live.length; i++) {
    const [x, y] = tf(live[i][0], live[i][1]);
    ovCx.fillStyle = "rgba(255,255,255,0.55)";
    ovCx.beginPath();
    ovCx.arc(x, y, 1.6, 0, 2 * Math.PI);
    ovCx.fill();
  }
  // -- live TIP trail: the last ~6 s of the Show-Tip stream, drawn as discrete
  //    POINTS, not a connected line. The stream is only a ~20 Hz poll, so joining
  //    the samples with segments would draw a jagged path the tip never took --
  //    the tip moves smoothly between samples; we simply don't sample that detail.
  //    Older points fade so the freshest history reads brightest. --
  {
    const n = tipTrail.length;
    for (let i = 0; i < n; i++) {
      const [x, y] = tf(tipTrail[i][0], tipTrail[i][1]);
      const a = 0.2 + 0.45 * (i / Math.max(1, n - 1));   // older -> fainter
      ovCx.fillStyle = `rgba(120,220,255,${a.toFixed(3)})`;
      ovCx.beginPath();
      ovCx.arc(x, y, 1.6, 0, 2 * Math.PI);
      ovCx.fill();
    }
  }
  // -- live TIP dot: the current tip position (XSensor/YSensor -> µm), a bright
  //    filled dot with a subtle glow ring. --
  if (state.litho.tip && !state.litho.tipHidden) {
    const [tx, ty] = tf(state.litho.tip.x, state.litho.tip.y);
    const W = ovCv.width, H = ovCv.height, m = 9;
    if (tx < 0 || tx > W || ty < 0 || ty > H) {
      // The tip is outside the view (before any scan the view is a placeholder
      // frame the tip need not be in): an arrowhead on the nearest edge, pointing
      // out toward it, so the marker is never simply absent.
      const ex = Math.min(Math.max(tx, m), W - m), ey = Math.min(Math.max(ty, m), H - m);
      const a = Math.atan2(ty - ey, tx - ex);
      ovCx.beginPath();
      ovCx.moveTo(ex + 7 * Math.cos(a), ey + 7 * Math.sin(a));
      ovCx.lineTo(ex + 6 * Math.cos(a + 2.4), ey + 6 * Math.sin(a + 2.4));
      ovCx.lineTo(ex + 6 * Math.cos(a - 2.4), ey + 6 * Math.sin(a - 2.4));
      ovCx.closePath();
      ovCx.fillStyle = "rgba(255,255,255,0.95)";
      ovCx.fill();
    } else {
      ovCx.beginPath();
      ovCx.arc(tx, ty, 7, 0, 2 * Math.PI);
      ovCx.fillStyle = "rgba(255,255,255,0.15)";       // glow ring
      ovCx.fill();
      ovCx.beginPath();
      ovCx.arc(tx, ty, 3, 0, 2 * Math.PI);
      ovCx.fillStyle = "rgba(255,255,255,0.95)";       // bright tip
      ovCx.fill();
    }
  }
  ovCx.restore();
}

function redraw() {
  syncCanvasSize(); drawScan(); drawOverlay(); drawZScale(); drawScope(); syncViewCap();
}

// #viewCap describes the current view: by name when a zoom button or an import set it,
// otherwise by size and centre. Every view change reaches redraw(), so it stays current.
let capView = null;
function sameView(a, b) {
  return a && b && a.x0 === b.x0 && a.y0 === b.y0 && a.w === b.w && a.h === b.h;
}
function setViewCap(name) {
  capView = { ...state.view };
  $("viewCap").textContent = `view: ${name} (${num(state.view.w)}×${num(state.view.h)} µm)`;
}
function syncViewCap() {
  const v = state.view;
  if (!v || sameView(v, capView)) return;
  capView = { ...v };
  $("viewCap").textContent =
    `view: ${num(v.w)}×${num(v.h)} µm @ (${num(v.x0 + v.w / 2)}, ${num(v.y0 + v.h / 2)})`;
}

// =========================================================================
//  OBJECT LIST -- one row per object: swatch (voltage color), name, kind.
// =========================================================================
function renderList() {
  const vmax = patternVmax();
  $("objList").innerHTML = state.objects.map((o, i) => {
    const c = rgb(voltageColor(o.voltage, vmax));
    const sel = i === state.selected ? " sel" : "";
    const dis = o.enabled === false ? " dis" : "";   // disabled -> dimmed + struck
    return `<div class="objrow${sel}${dis}" data-i="${i}" draggable="true">`
      + `<span class="ord">${i + 1}</span>`          // 1-based write position (top writes first)
      + `<span class="swatch" style="background:${c}"></span>`
      + `<span class="nm">${escapeHtml(o.name)}</span>`
      + `<span class="kd">${escapeHtml(displayKind(o))}</span></div>`;
  }).join("");
}
$("objList").addEventListener("click", (ev) => {
  const row = ev.target.closest(".objrow");
  if (!row) return;
  select(Number(row.dataset.i));
});

// =========================================================================
//  DRAG-TO-REORDER -- the list order IS the write order. An HTML5 drag reorders
//  the LOCAL `objects` array (a permutation of the current indices) and, for an
//  imported design, persists that permutation to hub.pattern via
//  POST /api/design/reorder {order:[...]}, where order[newPos] = oldIndex.
//  Coexists with click-to-select: we never preventDefault a plain click.
// =========================================================================
let dragFrom = null;                              // OLD index of the dragged row
// Persist a full permutation of the current indices to the backend write order.
// `order` is the NEW arrangement of OLD indices (order[newPos] = oldIndex).
async function persistReorder(order) {            // order = new arrangement of OLD indices
  if (!state.designImported) return;                    // mock objects: local-only
  await postDesignReorder(order);   // request() logs tx/rx (incl. any error)
}
$("objList").addEventListener("dragstart", (ev) => {
  const row = ev.target.closest(".objrow");
  if (!row) return;
  dragFrom = Number(row.dataset.i);
  ev.dataTransfer.effectAllowed = "move";
});
$("objList").addEventListener("dragover", (ev) => {
  ev.preventDefault();                            // allow drop
  ev.dataTransfer.dropEffect = "move";
  const row = ev.target.closest(".objrow");
  for (const r of $("objList").querySelectorAll(".drop-target")) r.classList.remove("drop-target");
  if (row) row.classList.add("drop-target");
});
$("objList").addEventListener("drop", async (ev) => {
  ev.preventDefault();
  const row = ev.target.closest(".objrow");
  for (const r of $("objList").querySelectorAll(".drop-target")) r.classList.remove("drop-target");
  const dragTo = row ? Number(row.dataset.i) : null;
  if (dragFrom == null || dragTo == null || dragFrom === dragTo) { dragFrom = null; return; }
  // preserve the current selection across the reorder by object identity
  const selObj = state.selected >= 0 ? state.objects[state.selected] : null;
  // tag each object with its OLD index so we can read back the permutation
  state.objects.forEach((o, k) => { o._oi = k; });
  // move the dragged item so it lands BEFORE the target row (in both directions)
  const item = state.objects[dragFrom];
  state.objects.splice(dragFrom, 1);
  const insertAt = dragFrom < dragTo ? dragTo - 1 : dragTo;
  state.objects.splice(insertAt, 0, item);
  // build the permutation of OLD indices in their NEW order, then drop the tag
  const order = state.objects.map((o) => o._oi);
  state.objects.forEach((o) => { delete o._oi; });
  state.selected = selObj ? state.objects.indexOf(selObj) : -1;
  dragFrom = null;
  logLine(`reordered: ${item.name} → position ${insertAt + 1}`, "tx");
  await persistReorder(order);
  renderList(); renderSettings();
  await refreshToolpath();                        // write order changed -> recompile
  drawOverlay();
});
$("objList").addEventListener("dragend", () => {
  dragFrom = null;
  for (const r of $("objList").querySelectorAll(".drop-target")) r.classList.remove("drop-target");
});

// =========================================================================
//  SETTINGS PANEL -- editable Voltage + Speed for the selected object.
// =========================================================================
// The last chiral params the user asked for, per object name. The profile
// only stores the curvature-CLAMPED per-point amp_um, so the Amplitude input
// would otherwise show the written value, not the requested one; this cache
// lets it show what was asked. Keyed by name so it survives re-adopt.
const chiralParams = {};
const FILL_KINDS = new Set(["shape", "electrode", "rect", "taper"]);
function renderSettings() {
  const o = state.selected >= 0 ? state.objects[state.selected] : null;
  if (!o) {
    $("setEmpty").style.display = ""; $("setFields").style.display = "none";
    return;
  }
  $("setEmpty").style.display = "none"; $("setFields").style.display = "";
  const vmax = patternVmax();
  $("setSwatch").style.background = rgb(voltageColor(o.voltage, vmax));
  $("setNameText").textContent = `${o.name} (${displayKind(o)})`;
  $("fVoltage").value = o.voltage;
  $("fSpeed").value = o.speed_um_s;
  $("fEnabled").checked = (o.enabled !== false);   // missing enabled reads as true
  // Chiral is a path-only setting: show the toggle only for paths, and the
  // four parameter fields only when it is actually on.
  const isPath = o.kind === "path";
  // Fill spacing is shown for every kind the compiler rasters with a fill pitch:
  // shape (drawn rectangle, region, outline), electrode/funnel, rect and taper.
  const isFill = FILL_KINDS.has(o.kind);
  $("densityRow").style.display = isFill ? "" : "none";
  $("fDensity").value = (o.fill_spacing_um != null ? o.fill_spacing_um : 0.15);
  const on = !!(o.profile && o.profile.kind === "chiral");
  $("chiralRow").style.display = isPath ? "" : "none";
  $("fChiral").checked = on;
  $("chiralFields").style.display = (isPath && on) ? "" : "none";
  // Populate from the profile when present (converting phase_rad -> degrees),
  // else from the last requested params, else the kto-helix preset defaults.
  const cached = chiralParams[o.name] || {};
  $("fLambda").value = (o.profile ? (o.profile.lambda_set_um ?? o.profile.lambda_um)
                        : (cached.lambda_um ?? 0.025));
  $("fVk").value = (o.profile ? o.profile.v_k : (cached.v_k ?? 5));
  $("fPhase").value = (o.profile ? Math.round(o.profile.phase_rad * 180 / Math.PI)
                       : (cached.phase_deg ?? 90));
  // Amplitude: show the REQUESTED value (the profile only keeps the clamped
  // per-point amp_um), falling back to the effective max amplitude, then the
  // preset default.
  const ampFallback = (o.profile && o.profile.amp_um && o.profile.amp_um.length)
      ? Math.max(...o.profile.amp_um) : 0.055;
  $("fAmp").value = (cached.y_amp_um ?? ampFallback);
}
function select(i) {
  state.selected = (state.selected === i) ? -1 : i;
  if (state.selected >= 0) logLine(`selected ${state.objects[state.selected].name}`, "tx");
  else logLine("cleared selection", "dim");
  editResetOnSelect();   // selection changed; node tool reloads (ec-tab-edit)
  renderList(); renderSettings(); drawOverlay();
  // if the node tool is up, show the new selection's handles immediately
  editReloadOnSelect();
}
// Persist a property edit to the backend so the compiled tip path (and any
// write) reflect the change -- otherwise the local edit and hub.pattern
// diverge and refreshToolpath() recompiles stale values. Only meaningful once
// a design has been imported into hub.pattern; mock objects stay local-only.
async function persistProps(i, patch) {
  if (!state.designImported) return;                 // mock objects: local-only, as before
  const body = { op: "set_props", indices: [i], ...patch };
  const res = await postDesignOp(body);   // request() logs tx/rx (incl. any error)
  // A refused edit changed nothing on the service: show its values again.
  if (res.data && res.data.error && res.data.design) {
    const name = state.objects[i] ? state.objects[i].name : null;
    adoptDesign(res.data.design, state.sourceLabel);
    state.selected = state.objects.findIndex((o) => o.name === name);
    renderList(); renderSettings();
  }
  // A set_props can change the enabled set / speed / voltage that the write time
  // depends on (docs/INSTRUMENT_MODEL_SPEC.md sec 4: estimate refreshed after every
  // set_props). The design_changed broadcast this POST triggers also refreshes it
  // on other clients.
  refreshWriteEstimate();
}
$("fVoltage").addEventListener("change", async () => {
  if (state.selected < 0) return;
  const o = state.objects[state.selected];
  const old = o.voltage;
  const nv = Number($("fVoltage").value);
  o.voltage = nv;
  logLine(`${o.name} voltage ${num(old)} → ${num(nv)}`, "info");
  // voltage change -> recolor swatch + overlay live (vmax may shift, so redraw all)
  renderList(); renderSettings();
  // persist to hub.pattern BEFORE recompiling so the tip path reflects the edit
  await persistProps(state.selected, { voltage: nv });
  // a voltage change can alter the compiled path/timing -> recompile the tip path
  await refreshToolpath();
  drawOverlay();
});
$("fSpeed").addEventListener("change", async () => {
  if (state.selected < 0) return;
  const o = state.objects[state.selected];
  const old = o.speed_um_s;
  const nv = Number($("fSpeed").value);
  o.speed_um_s = nv;
  logLine(`${o.name} speed ${num(old)} → ${num(nv)} µm/s`, "info");
  // persist to hub.pattern BEFORE recompiling so the tip path reflects the edit
  await persistProps(state.selected, { speed_um_s: nv });
  // a speed change can alter the compiled path/timing -> recompile the tip path
  await refreshToolpath();
  drawOverlay();
});
$("fEnabled").onchange = async () => {
  if (state.selected < 0) return;
  const o = state.objects[state.selected];
  o.enabled = $("fEnabled").checked;
  logLine(`${o.name} ${o.enabled ? "enabled" : "disabled"}`, "info");
  await persistProps(state.selected, { enabled: o.enabled });
  renderList();                 // reflect the dimmed state in the list
  await refreshToolpath();      // disabled objects drop out of the compiled path
  drawOverlay();
};
// Fill spacing (density) is a per-object field the compiler already honours
// everywhere it places fill lines; persist it via set_props (like voltage/speed)
// then recompile so the raster line pitch updates on screen. Geometry (points_um)
// is untouched -- only the compiled toolpath changes.
$("fDensity").addEventListener("change", async () => {
  if (state.selected < 0) return;
  const o = state.objects[state.selected];
  const old = o.fill_spacing_um != null ? o.fill_spacing_um : 0.15;
  const nv = Number($("fDensity").value);
  if (!(nv > 0)) { $("fDensity").value = old; return; }   // reject non-positive
  o.fill_spacing_um = nv;
  logLine(`${o.name} fill spacing ${num(old)} → ${num(nv)} µm`, "info");
  await persistProps(state.selected, { fill_spacing_um: nv });
  await refreshToolpath();      // fill pitch is recomputed at compile time
  drawOverlay();
});

// =========================================================================
//  COPY / PASTE -- Ctrl+C copies the selected object, Ctrl+V pastes a copy made
//  by the service's duplicate op (a deep copy of the model object, so every
//  property the service holds travels, not just what the design GET carries).
// =========================================================================
// The clipboard holds source NAMES and the paste count: each paste steps a
// further +1/-1 um from the source. A source deleted or renamed since the copy
// cannot be pasted, because the service copies the live object.
const PASTE_STEP_UM = 1.0;
let objClipboard = null;
// the duplicate button's own source and count, apart from the Ctrl+C clipboard
let dupClipboard = null;
function typingInField(t) {
  return !!t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA"
                 || t.tagName === "SELECT" || t.isContentEditable);
}
async function pasteObjects(cb = objClipboard, verb = "pasted") {
  const indices = cb.names.map((n) => state.objects.findIndex((o) => o.name === n));
  const gone = cb.names.filter((n, k) => indices[k] < 0);
  if (gone.length) { logLine(`paste: ${gone.join(", ")} no longer in the design`, "warn"); return; }
  const n = cb.pastes + 1;
  const res = await postDesignOp({ op: "duplicate", indices,
                                   dx_um: n * PASTE_STEP_UM, dy_um: -n * PASTE_STEP_UM });
  const sel = res.ok && res.data && Array.isArray(res.data.selection) ? res.data.selection : [];
  if (!await reAdopt(res, sel.length ? sel[0] : -1)) return;
  cb.pastes = n;
  // the service appends the copies in the sources' pattern order
  const srcs = cb.names.map((nm, k) => [indices[k], nm]).sort((a, b) => a[0] - b[0]);
  sel.forEach((i, k) => {
    const c = state.objects[i];
    const src = srcs[k] ? srcs[k][1] : "?";
    logLine(`${verb} ${src} as ${c.name} (+${num(n * PASTE_STEP_UM)}/-${num(n * PASTE_STEP_UM)} µm), `
            + (c.enabled !== false ? "enabled: included in a write-all" : "disabled"), "info");
  });
  refreshWriteEstimate();
}
// The shortcuts and the Select options' buttons run these same three.
function copySelected() {
  objClipboard = { names: [state.objects[state.selected].name], pastes: 0 };
  logLine(`copied ${objClipboard.names.join(", ")}`, "dim");
}
// a paste changes the design the running write was compiled from
function pasteRefused() {
  if (!(state.litho.writing || $("perform").dataset.writing === "1")) return false;
  logLine("paste refused: a write is in progress", "warn");
  return true;
}
const clipboardActions = {
  copy() {
    if (state.selected < 0 || !state.designImported) { logLine("copy: no object selected", "warn"); return; }
    copySelected();
  },
  paste() {
    if (!objClipboard) { logLine("paste: nothing copied", "warn"); return; }
    if (!pasteRefused()) pasteObjects();
  },
  // A copy and paste in one step that leaves the clipboard as it was. Repeated on
  // the same source, each duplicate steps a further +1/-1 µm, as pastes do.
  duplicate() {
    if (state.selected < 0 || !state.designImported) { logLine("duplicate: no object selected", "warn"); return; }
    const name = state.objects[state.selected].name;
    if (!dupClipboard || dupClipboard.names[0] !== name) dupClipboard = { names: [name], pastes: 0 };
    if (!pasteRefused()) pasteObjects(dupClipboard, "duplicated");
  },
};
window.addEventListener("keydown", (ev) => {
  if (state.currentTab !== "edit" && state.currentTab !== "write") return;
  if (!ev.ctrlKey || ev.altKey || ev.shiftKey || ev.metaKey || typingInField(ev.target)) return;
  // the node tool leaves Ctrl+C/Ctrl+V to the browser; the clipboard is kept
  if (state.currentTab === "edit" && state.activeTool === "node") return;
  const k = ev.key.toLowerCase();
  if (k === "c") {
    if (state.selected < 0 || !state.designImported) return;
    // selected page text (the log, a readout) is the browser's copy, not ours
    if (String(window.getSelection ? window.getSelection() : "").length) return;
    ev.preventDefault();
    copySelected();
  } else if (k === "v") {
    if (!objClipboard) return;
    ev.preventDefault();
    if (!pasteRefused()) pasteObjects();
  }
});

// The shared tail every persist* helper ends with: re-adopt the design the POST
// returned, restore the selection adoptDesign() reset, then re-render the list /
// settings and re-flatten the compiled toolpath. `sel` says how to restore the
// selection -- a name string re-finds the object (the design may have re-indexed
// it), a number pins it to that saved index. `after(sel)` runs once the
// selection is set and before the re-render, for a caller's own module state
// (the edit tab's base-Bézier bookkeeping). Returns false (and does nothing)
// when the request failed -- request() has already logged it.
async function reAdopt(res, sel, after) {
  if (!res.ok || res.data.error) return false;   // request() logged it
  adoptDesign(res.data.design, state.sourceLabel);
  state.selected = (typeof sel === "number")
    ? sel
    : state.objects.findIndex((o) => o.name === sel);
  if (after) after(state.selected);
  renderList(); renderSettings();
  await refreshToolpath();
  drawOverlay();
  return true;
}

// Apply / re-parametrize / revert chiral on the selected path. The backend
// captures the base axis once and always regenerates from it, so re-editing a
// field cannot compound; here we just POST the requested params, re-adopt the
// returned design (which carries the modulated points_um + profile + bias),
// and restore the selection adoptDesign resets.
async function persistChiral() {
  if (state.selected < 0 || !state.designImported) return;
  const o = state.objects[state.selected];
  if (o.kind !== "path") return;
  const enabled = $("fChiral").checked;
  const body = {
    indices: [state.selected], enabled,
    lambda_um: Number($("fLambda").value),
    y_amp_um:  Number($("fAmp").value),
    phase_deg: Number($("fPhase").value),
    v_k:       Number($("fVk").value),
    v0:        Number($("fVoltage").value),
  };
  const res = await postDesignChiral(body);
  if (!res.ok || res.data.error) {         // request() already logged it
    // A refused edit changed nothing on the service: show its values again.
    if (res.data && res.data.design) {
      adoptDesign(res.data.design, state.sourceLabel);
      state.selected = state.objects.findIndex((x) => x.name === o.name);
      renderList(); renderSettings(); drawOverlay();
    }
    return;
  }
  // remember what was asked so the Amplitude input shows the requested value,
  // not the curvature-clamped per-point amp_um the profile stores
  chiralParams[o.name] = {
    lambda_um: body.lambda_um, y_amp_um: body.y_amp_um,
    phase_deg: body.phase_deg, v_k: body.v_k,
  };
  const j = res.data;
  if (j.clamp) {
    logLine(`amplitude clamped on ${Math.round(j.clamp.fraction * 100)}% `
            + `of the wire`, "warn");
  }
  const sel = state.selected;
  await reAdopt(res, sel);   // restore by saved index (adoptDesign resets to -1)
}
$("fChiral").onchange = persistChiral;
for (const id of ["fAmp", "fPhase", "fLambda", "fVk"]) {
  $(id).addEventListener("change", persistChiral);
}

// =========================================================================
//  ZOOM BUTTONS -- both are REAL client-side view fits.
// =========================================================================
$("zoomAll").onclick = () => {
  state.view = boundsOfObjects();
  setViewCap("all objects");
  logLine("zoom to all", "tx");
  redraw();
};
function zoomToScan() {
  state.view = { ...state.SCAN_FRAME };
  setViewCap("scan frame");
  logLine("zoom to scan", "tx");
  redraw();
}
$("zoomScan").onclick = zoomToScan;

// =========================================================================
//  SVG IMPORT (piece 2) -- mirrors litho-debug.html importSvg(): POST the raw
//  file bytes to /api/design/svg with a ?name= query and image/svg+xml content
//  type. Scale is fixed server-side at 1 unit = 1 nm -- there is no scale param.
//  On success the returned design REPLACES the mock objects; select/edit, the
//  colormap, list, overlay, and zoom-to-all all keep working unchanged because
//  the design serializes into the same object shape the mock uses
//  ({ name, kind, voltage, voltage_end, speed_um_s, points_um }).
// =========================================================================
// Adopt a backend design dict (same shape from /api/design and the SVG
// import) into local state: object list, colormap, overlay, framing. Shared
// by importSvg() and loadCurrentDesign() so the two paths cannot drift.
function adoptDesign(design, sourceLabel) {
  state.objects = (design.objects || []).map((o) => ({
    name: o.name, kind: o.kind, voltage: o.voltage,
    voltage_end: (o.voltage_end !== undefined ? o.voltage_end : null),
    speed_um_s: o.speed_um_s,
    // the raster pitch the Fill spacing field shows; without it every re-adopt
    // (design_changed, reload) shows the 0.15 fallback instead of the model value
    fill_spacing_um: (o.fill_spacing_um != null ? o.fill_spacing_um : null),
    enabled: (o.enabled !== undefined ? o.enabled : true),
    v_profile: (o.v_profile !== undefined ? o.v_profile : null),
    profile: (o.profile !== undefined ? o.profile : null),
    // the control-point bezier (when the backend serialized one) so displayKind
    // can label rectangle-vs-region and the node tool can refuse rectangles.
    bezier: (o.bezier !== undefined ? o.bezier : null),
    // the endpoint->electrode binding (EDIT_TOOLS §4 snap-to-bind), when the
    // backend serialized one, so the node tool sees the persisted binding.
    end_binding: (o.end_binding !== undefined ? o.end_binding : null),
    // inner outlines, when the service serializes them; Select refuses a skew
    // that would leave them behind
    holes_um: (o.holes_um !== undefined ? o.holes_um : null),
    points_um: o.points_um || [],
  }));
  state.selected = -1;                   // reset selection (indices just changed)
  state.designImported = true;           // hub.pattern now holds this design
  $("src").textContent = `source: ${sourceLabel}`;
  state.sourceLabel = sourceLabel;   // the label is now model state, not #src text
  renderList();
  renderSettings();                // back to empty state
}

async function importSvg(file) {
  const res = await postDesignSvg(file.name, file);
  const j = res.data;
  if (!res.ok || j.error || !j.ok || !j.design) return;   // request() logged it; keep current objects
  adoptDesign(j.design, j.design.name || file.name);
  logLine(`imported ${j.design.name || file.name}: ${state.objects.length} object(s)`, "rx");
  // frame the imported design ("zoom to all")
  state.view = boundsOfObjects();
  setViewCap("all objects");
  // compile the actual tip path of the freshly imported design (if toggle on)
  await refreshToolpath();
  redraw();
}

// =========================================================================
//  HYDRATE FROM BACKEND (boot) -- GET /api/design returns the current
//  hub.pattern as the same design dict the SVG import serves. If a real
//  design is already loaded server-side, adopt it so a reload shows the
//  actual objects (and the tip path lands over them) instead of the mock
//  placeholders. An empty design or a failed fetch keeps the mock objects.
// =========================================================================
async function loadCurrentDesign() {
  const res = await getDesign();
  const d = res.data;
  if (!res.ok || !d || !Array.isArray(d.objects) || d.objects.length === 0) {
    logLine("no backend design; showing mock objects", "dim");
    return;                        // keep the mock objects + boot behavior
  }
  adoptDesign(d, d.name || "backend design");
  // frame the hydrated design ("zoom to all")
  state.view = boundsOfObjects();
  setViewCap("all objects");
  logLine(`hydrated ${state.objects.length} object(s) from backend`, "rx");
  await refreshToolpath();
  redraw();
}

// =========================================================================
//  LAST SCAN -- GET /api/scan/last returns the newest cached scan frame (the
//  backend persists every completed scan to disk). On success we build the
//  grayscale image, adopt the scan's µm bounds as SCAN_FRAME, frame the view
//  over it so the design overlay lands in register, and redraw. Any non-OK /
//  empty payload keeps the placeholder (the page still works with no scan yet).
// =========================================================================
async function loadLastScan() {
  const res = await getScanLast();
  // A missing/empty cache must NEVER blank a good frame: if we just drew a live
  // scan (scanFrame populated), keep it. On initial load scanFrame is null, so the
  // placeholder still shows correctly.
  if (!res.ok) { logLine(`no cached scan (HTTP ${res.status}); keeping ${state.scanFrame ? "live frame" : "placeholder"}`, "warn"); if (!state.scanFrame) { state.scanImg = null; redraw(); } return; }
  const f = res.data;
  if (!f || !f.pixels || !f.height) { logLine(`scan payload empty; keeping ${state.scanFrame ? "live frame" : "placeholder"}`, "warn"); if (!state.scanFrame) { state.scanImg = null; redraw(); } return; }
  state.scanFrame = f; rebuildScanImg();
  state.SCAN_FRAME = { x0: f.x_offset_um - f.size_um / 2, y0: f.y_offset_um - f.size_um / 2, w: f.size_um, h: f.size_um };
  if (f.angle_deg) logLine(`note: scan angle ${num(f.angle_deg)}° drawn axis-aligned (rotation not yet transformed)`, "warn");
  logLine(`scan ${f.pixels}×${f.rows} ${num(f.size_um)}µm`, "rx");
  // frame the scan by default so the design lands over it
  state.view = { ...state.SCAN_FRAME };
  setViewCap("scan frame");
  redraw();
}

// =========================================================================
//  SCAN CONTROLS + LIVE WEBSOCKET FEED -- ported from debug.html. The scan
//  controls POST to the SAME endpoints; the websocket streams the frame line by
//  line and reuses the cockpit's scanFrame / buildScanImg / redraw so the live
//  image lands in register under the design overlay.
// =========================================================================

// The scan request body, exactly as debug.html builds it.
function scanBody(direction){return{size_um:+$("size").value,pixels:+$("pixels").value,lines:+$("lines").value,line_rate_hz:+$("rate").value,angle_deg:+$("angle").value,mode:$("mode").value,setpoint:+$("setpoint").value,direction,x_offset_um:0,y_offset_um:0};}

// api() (the POST-JSON helper) now lives in ec-api.js and is imported above.

// Small chip helper (the ws status chip already lives in the header).
function setChip(id, text, cls){
  const el = $(id); if (!el) return;
  el.textContent = text; el.className = "chip" + (cls ? " " + cls : "");
}

// Reflect the live write status in the #lithoStatus chip: "writing <name> — NN%"
// while a write runs, "litho: idle" otherwise. litho_progress carries only the
// fraction of the stroke being written (no object index or count), so the Write
// tab's bar and label show that stroke; after litho_done they say how the run ended.
let lastWriteEnd = null;   // null before any write, else "done" | "aborted"
function updateLithoStatus(){
  const writing = !!(state.litho.writing && state.litho.name);
  const frac = Math.min(1, Math.max(0, Number(state.litho.frac) || 0));
  const pct = Math.round(frac * 100);
  if (writing)
    setChip("lithoStatus", `writing ${state.litho.name} — ${pct}%`, "busy");
  else
    setChip("lithoStatus", "litho: idle", "");
  const bar = $("writeProgress"), txt = $("writeProgressText");
  const barPct = writing ? pct : (lastWriteEnd === "done" ? 100 : (lastWriteEnd ? pct : 0));
  if (bar) {
    bar.firstElementChild.style.width = `${barPct}%`;
    bar.setAttribute("aria-valuenow", String(barPct));
  }
  if (txt) {
    txt.textContent = writing ? `writing ${state.litho.name} · ${pct} % of this stroke`
                              : (lastWriteEnd || "idle");
  }
}

// Initialize (or re-initialize) the live scan grid from a line/scan_reset frame
// header: a NaN-filled Float32Array plus the µm bounds so the overlay registers.
function initScanFrame(m){
  frameStarts();
  state.scanFrame = { pixels:m.pixels, rows:m.rows, size_um:m.size_um,
    x_offset_um:(state.lastScanCfg?.x_offset_um)||0, y_offset_um:(state.lastScanCfg?.y_offset_um)||0,
    angle_deg:m.angle_deg||0, height:new Float32Array(m.pixels*m.rows).fill(NaN),
    // parallel grids so the channel selector can render any captured channel
    error:new Float32Array(m.pixels*m.rows).fill(NaN) };
  state.SCAN_FRAME = { x0:state.scanFrame.x_offset_um-state.scanFrame.size_um/2, y0:state.scanFrame.y_offset_um-state.scanFrame.size_um/2,
    w:state.scanFrame.size_um, h:state.scanFrame.size_um };
  state.view = { ...state.SCAN_FRAME };
  setViewCap("scan frame");
  lineCount = 0;
  startLiveRows();
}

// Reset the live grid straight from a scan request config (scanBody) so the
// canvas resizes the instant Frame Up/Down is clicked -- before the first line
// arrives -- instead of showing the old frame until scan_done. Mirrors
// initScanFrame but reads the config's field names (cfg.lines -> rows; falls
// back to cfg.pixels when lines is 0/falsy so a squared frame still fills).
function resetScanFrameFromCfg(cfg){
  const rows = cfg.lines || cfg.pixels;
  frameStarts();
  // Re-running a scan at the SAME geometry must NOT blank the previous frame:
  // keep the old image and let the incoming line handler overwrite each row as
  // the new sweep arrives, so the operator can SEE the drift between consecutive
  // scans. Only allocate a fresh NaN frame when there is no current frame, or the
  // geometry differs (a different resolution/size/angle can't overlay the old
  // grid line-for-line). The `line` handler applies the same geometry test, so a
  // kept frame is filled in place; a differing one is re-initialised there too.
  const keep = state.scanFrame && state.scanFrame.height
    && state.scanFrame.pixels === cfg.pixels && state.scanFrame.rows === rows
    && state.scanFrame.size_um === cfg.size_um
    && (state.scanFrame.angle_deg || 0) === (cfg.angle_deg || 0);
  if (keep) {
    // reuse the existing height/error grids in place -- just re-point the frame's
    // offsets to the new request so a moved frame still lands where it was asked.
    state.scanFrame.x_offset_um = cfg.x_offset_um || 0;
    state.scanFrame.y_offset_um = cfg.y_offset_um || 0;
  } else {
    state.scanFrame = { pixels:cfg.pixels, rows, size_um:cfg.size_um,
      x_offset_um:cfg.x_offset_um||0, y_offset_um:cfg.y_offset_um||0,
      angle_deg:cfg.angle_deg||0, height:new Float32Array(cfg.pixels*rows).fill(NaN),
      error:new Float32Array(cfg.pixels*rows).fill(NaN) };
  }
  state.SCAN_FRAME = { x0:state.scanFrame.x_offset_um-state.scanFrame.size_um/2, y0:state.scanFrame.y_offset_um-state.scanFrame.size_um/2,
    w:state.scanFrame.size_um, h:state.scanFrame.size_um };
  state.view = { ...state.SCAN_FRAME };
  setViewCap("scan frame");
  lineCount = 0;
  startLiveRows();
  rebuildScanImg();
  redraw();
}

// ---- websocket -----------------------------------------------------------
// Reconnect-on-close + NaN/Infinity-safe parse, straight from debug.html.
// ---- live websocket feed -------------------------------------------------
// The ws connection (reconnect + NaN-safe parse) now lives in ec-api.js; this
// page registers a handler per message type on its onWs registry instead of a
// central switch, and boot calls connectWs() with the lifecycle hooks. Behaviour
// is byte-for-byte the same as the old connect()/dispatch(): "line" renders the
// live scan by reusing scanFrame / buildScanImg / redraw; "scan_done" reconciles
// to the authoritative cached frame via loadLastScan().

onWs("line", (m) => {
  // First line of a frame (or geometry changed) -> (re)initialize the grid.
  // Re-init on ANY geometry change -- resolution OR µm size OR angle -- so a
  // new scan at a different size_um rebuilds the grid at the correct bounds
  // from row 0 instead of filling the old frame and snapping at scan_done.
  if (!state.scanFrame || state.scanFrame.pixels !== m.pixels || state.scanFrame.rows !== m.rows
      || state.scanFrame.size_um !== m.size_um || (state.scanFrame.angle_deg || 0) !== (m.angle_deg || 0)
      || m.new_frame)
    initScanFrame(m);
  else if (frameEnded || !state.scanFrame.liveRows) {
    // the first line after a scan_done (a continuous run's next frame, or a scan
    // started elsewhere): a new frame over the same grid
    frameStarts();
    startLiveRows();
  }
  // Fill this row into every channel grid the line carries. slow_idx 0 =
  // bottom row (buildScanImg flips rows-1-r). m.height/m.error are already in
  // their native units (nm / V) -- same as the cached frame, do NOT rescale.
  const si = m.slow_idx, P = m.pixels;
  fillScanRow(state.scanFrame.height, m.height, si, P);
  fillScanRow(state.scanFrame.error, m.error, si, P);
  if (si >= 0 && si < state.scanFrame.rows) state.scanFrame.liveRows[si] = 1;
  scanRowNow = si;
  // hand the raw trace/retrace of THIS line to the scope (per-channel arrays)
  lastScopeLine = { height: m.height, retrace: m.retrace,
                    error: m.error, error_retrace: m.error_retrace };
  lineCount++;
  rebuildScanImg();          // rebuild the offscreen grayscale from the grid
  redraw();                  // image, design on top, z bar and this line's scope trace
  logLine(`rx line #${lineCount} row=${m.row} slow_idx=${m.slow_idx} `
    + `pixels=${m.pixels} rows=${m.rows}`, "rx");
});
onWs("scan_reset", (m) => {
  initScanFrame(m);
  rebuildScanImg(); redraw();
  logLine(`rx scan_reset pixels=${m.pixels} rows=${m.rows} size=${m.size_um}µm`, "rx");
});
onWs("scan_done", (m) => {
  logLine("scan_done aborted=" + m.aborted, m.aborted ? "warn" : "rx");
  lineCount = 0;
  snapshotFrame();
  frameEnded = true;
  scanRowNow = null;
  redraw();
  // Keep the live frame the user just watched build; only fall back to the
  // disk cache if there is no live frame. Reloading on every scan_done blanked
  // a good scan whenever the cached frame came back partial.
  if (!state.scanFrame || !state.scanFrame.height) loadLastScan();
});
onWs("telemetry", (m) => {
  if ($("logTele").checked) logLine("telemetry " + JSON.stringify(m), "dim");
});
onWs("log", (m) => {
  logLine("LOG: " + m.msg, "info");
});
// ---- live litho feed (NEED #2) -- accumulate + draw the written path ----
onWs("litho_object_start", (m) => {
  lastWriteEnd = null;
  onObjectStart();
  state.litho.writing = true; state.litho.name = m.name; state.litho.frac = 0; state.litho.live = [];
  logLine(`writing ${m.name}`, "tx");
  updateLithoStatus();
  drawOverlay();
});
onWs("litho_progress", (m) => {
  // x_um/y_um is the LIVE tip position (XSensor/YSensor -> µm). The twin sends a
  // few frames per second and lists in trail_um every position it stepped through
  // since its last frame; hardware sends one frame per stroke, without it.
  // The marker (state.litho.tip) is NOT moved here: it comes from `tip_pos` alone,
  // because the two message types reach the page in no fixed order.
  state.litho.frac = m.frac;
  if (Array.isArray(m.trail_um) && m.trail_um.length) state.litho.live.push(...m.trail_um);
  else state.litho.live.push([m.x_um, m.y_um]);
  if (state.litho.live.length > 4000) state.litho.live.splice(0, state.litho.live.length - 4000);
  updateLithoStatus();
  drawOverlay();
});
// Since 2026-09-11 one entry is one whole STROKE, not one 8,192-point compiled
// segment: `path_um` can be tens of thousands of points (already [::4]-subsampled
// server-side), and on the HARDWARE path `litho_progress` now fires once per stroke
// rather than once per 8,192 points. Nothing here assumed a size -- drawing a 20k-point
// polyline is cheap -- and the live TIP marker comes from the 10 Hz TELEM feed
// (`tip_pos`), which is unchanged.
onWs("litho_object_done", (m) => {
  // the COMPLETED written path -- keep it as accumulated "written material"
  const entry = { path_um: m.path_um, v_written: m.v_written, voltage: m.voltage };
  state.litho.written.push(entry);
  recordStroke(entry, m);
  // REPLACE the intermediate point cloud with the full connected trace: clear
  // both the twin live cloud and the hardware tip-stream cloud for this object.
  state.litho.live = []; state.litho.tipTrail = [];
  dropTipUnlessTwin();
  logLine(`${m.name} written`, "rx");
  drawOverlay();
});
// An abort cut this object mid-stroke (service/passes.py LithoRun.run). It is
// pushed into the SAME `written` array as a completed object -- so it survives
// litho_done (which never clears `written`, see below) and the clearTrace
// button the same way -- but tagged `partial` so drawLitho dashes it instead
// of drawing it as a solid completed stroke.
onWs("litho_object_partial", (m) => {
  const entry = { path_um: m.path_um, v_written: m.v_written, voltage: m.voltage, partial: true };
  state.litho.written.push(entry);
  recordStroke(entry, m);
  state.litho.live = []; state.litho.tipTrail = [];
  dropTipUnlessTwin();
  logLine(`${m.name} PARTIAL (aborted mid-stroke, frac ${(m.frac1 ?? 0).toFixed(2)})`, "warn");
  drawOverlay();
});
// ---- live TIP marker (Show Tip): a low-res ~20 Hz real-time position, a
//      SEPARATE stream from the per-object written path. Reuses the litho.tip
//      dot and adds a short fading trail; does not touch litho.written/live. ----
// On the twin the marker outlives an object and a write: the twin tip stream never
// stops, so the position is always current. On hardware it is dropped, because a
// bridge that died mid-write would leave it showing a stale position.
function dropTipUnlessTwin() {
  if (!state.litho.twin) state.litho.tip = null;
}
onWs("tip_pos", (m) => {
  state.litho.tip = { x: m.x_um, y: m.y_um };
  (state.litho.tipTrail = state.litho.tipTrail || []).push([m.x_um, m.y_um]);
  if (state.litho.tipTrail.length > 120) state.litho.tipTrail.splice(0, state.litho.tipTrail.length - 120);
  drawOverlay();
});
onWs("litho_done", (m) => {
  lastWriteEnd = m.aborted ? "aborted" : "done";
  onRunDone();
  state.litho.writing = false; state.litho.name = null; state.litho.live = [];
  state.litho.tipTrail = [];
  dropTipUnlessTwin();
  logLine(`lithography ${m.aborted ? "aborted" : "complete"}`, m.aborted ? "warn" : "rx");
  markWriting(false);   // clear the write-duration disable
  updateLithoStatus();
  drawOverlay();
});
onWs("sample_changed", (m) => {
  // The loaded sample changed -> its electrode ports change too (they are
  // keyed per-sample). Re-hydrate so the Connect tab never shows stale ports.
  logLine("rx sample_changed — reloading electrode ports", "rx");
  loadPorts();
});
onWs("instrument_rebound", () => {
  // Hardware replaced the twin: the service discarded the twin's imaging mode.
  logLine("rx instrument_rebound — re-reading imaging mode from the service", "rx");
  loadMode();
  renderRelative({ last_free_air_v: null, last_effective_v: null });
});
onWs("drift", (m) => {
  // Drift estimate / state change (Connect Phase 2). Mirror the toggle +
  // total, refresh the readout, and -- when the backend moved the ports
  // (TRACK) -- re-hydrate the overlay so the marks follow their electrodes.
  // The scan_start compensation frame also arrives here (offset only).
  if ("enabled" in m) state.driftState.enabled = !!m.enabled;
  if (Array.isArray(m.total)) state.driftState.total = m.total;
  if (m.reference_set || m.rebased) state.driftState.has_reference = true;
  const latest = (typeof m.dx_um === "number") ? [m.dx_um, m.dy_um] : null;
  renderDrift(latest);
  if (m.rejected) logLine(`rx drift rejected (${m.reason || "?"})`, "warn");
  else if (latest) logLine(`rx drift dx=${driftNm(m.dx_um)} dy=${driftNm(m.dy_um)} nm`
          + `${m.ports_moved ? " · ports tracked" : ""}`, "rx");
  // Drift tracking moved the ports: refresh them on the Connect tab.
  if (m.ports_moved && state.currentTab === "connect") loadPorts();
});
onWs("rail_guard", (m) => {
  // Rail-dwell safety guard live readout (armed indicator + the consecutive-
  // railed-line counter climbing toward the trip, and the trip reason).
  renderRail(m);
  if (m.tripped)
    logLine(`SAFETY rail-dwell (Guard ${m.tripped.guard}): ${m.tripped.reason||''}`, "warn");
});
onWs("continuous", (m) => {
  // Backend-owned auto-repeat loop state: start / per-frame increment /
  // stop-with-reason. Sync the toggle + readout (this is also how an abort
  // reflects the loop turning off -- scan/abort disables it in the service).
  renderContinuous(m);
  if (m.relative) renderRelative(m.relative);
  if (m.stopped_reason)
    logLine(`continuous stopped: ${m.stopped_reason} (${m.count||0} frames)`, "warn");
});
onWs("relative_setpoint", (m) => {
  // Between-frame re-reference readout (relative mode) or a config echo.
  renderRelative(m);
  if (m.effective_v != null)
    logLine(`rx relative setpoint: free-air ${m.free_air_v} V -> ${m.effective_v} V`, "rx");
});
// design_changed (docs/INSTRUMENT_MODEL_SPEC.md sec 4): ANY server-side design
// mutation, from EITHER door, is broadcast as a bare {type:"design_changed"}. Refetch
// GET /api/design and re-adopt the authoritative design WHOLESALE (enabled flags,
// voltages, everything) -- "server returns truth, client re-adopts" (COCKPIT_MODULES
// §2), so a second client's edit lands here without a reload. Selection is preserved
// by name; the view is NOT reframed (that is a user gesture). The write estimate is
// refreshed too, so a stale on-screen enabled set shows up as a total only the
// service's set could explain.
let _designChangedSeq = 0;
async function refetchDesign() {
  // Guard against out-of-order fetches: two rapid design_changed frames each fetch
  // /api/design, and a slow earlier response must not overwrite a newer one with
  // stale objects. Only the latest fetch is applied.
  const seq = ++_designChangedSeq;
  const res = await getDesign();
  if (seq !== _designChangedSeq) return;   // superseded by a newer design_changed
  const d = res.data;
  if (!res.ok || !d || !Array.isArray(d.objects)) { emit("designRefetch", false); return; }
  const selName = state.selected >= 0 ? state.objects[state.selected].name : null;
  adoptDesign(d, d.name || state.sourceLabel || "backend design");
  emit("designRefetch", true);
  state.selected = selName ? state.objects.findIndex((o) => o.name === selName) : -1;
  renderList(); renderSettings();
  refreshWriteEstimate();     // independent of the toolpath compile below
  hydrateInstrument();        // re-hydrate #lithoSetpoint (section 15 item 2) on any design mutation
  await refreshToolpath();
  drawOverlay();
}
onWs("design_changed", refetchDesign);
initDesignFiles({
  refetchDesign, showRefusal: showOtherRefusal, showFailure,
  guardNoticeMark, clearGuardNotice,
});
onWsDefault((m) => {
  // aborted / sessions_changed / ... CAP the dump so no
  // single message (a big telemetry payload) can ever flood the panel again.
  let s = JSON.stringify(m);
  if (s.length > 240) s = s.slice(0, 240) + `… (+${s.length - 240} more chars)`;
  logLine("rx " + (m.type || "?") + " " + s, "rx");
});

// ---- scan-control wiring -------------------------------------------------
$("approach").onclick = () => api("approach", {setpoint:+$("setpoint").value, igain:+$("igain").value, mode:$("mode").value});
$("applyGain").onclick = () => api("zgain", {setpoint:+$("setpoint").value, igain:+$("igain").value, mode:$("mode").value});
$("withdraw").onclick = () => api("withdraw");
// Mode dropdown: actually SWITCH the instrument's imaging mode (contact <-> AC) so the
// operator never has to change it by hand in the Asylum software. CRITICAL: the browser
// mode and the software mode must never disagree -- an AC scan with the software in
// Contact writes to a dead crosspoint. So the dropdown reflects the APPLIED instrument
// mode, and a switch that did NOT happen (refused because engaged, or a failure) reverts
// the dropdown back rather than leaving it showing an unapplied mode.
//
// `instrumentMode` tracks the last APPLIED imaging mode (synced from GET /api/mode on
// load, updated only on a successful switch). Only contact/ac are switchable imaging
// modes here (current stays in the deflection/contact family -- the vendor mode-profile
// call covers Contact/AC Mode only); the approach still reads $("mode").value for its
// PIDS InputChannel, so the two compose.
let instrumentMode = $("mode").value;
async function loadMode(){
  const d = (await getMode()).data;              // request() logs the traffic
  if (d && d.mode){ instrumentMode = d.mode; $("mode").value = d.mode; }
  else { instrumentMode = $("mode").value; }     // not established yet: adopt the dropdown
  const known = !!(d && d.mode);                 // unread: flag the dropdown, do not blank it
  markModeKnown(known);
  syncScanChanLabels();                          // error-channel label follows the mode
}
// While the mode is unread the dropdown keeps its value and shows an amber "unread" tag.
function markModeKnown(known){
  $("mode").title = known ? "" : "imaging mode unknown until read from the instrument";
  const tag = $("modeUnread");
  if (tag) tag.hidden = known;
}
function revertMode(){ $("mode").value = instrumentMode; syncScanChanLabels(); }
async function apiSetMode(mode){
  const d = await api("mode", { mode });
  if (!d || d.ok !== true){
    // the switch did NOT apply -> keep browser == software by reverting the dropdown.
    if (d && d.reason === "engaged")
      logLine("Mode switch REFUSED — withdraw before changing mode (dropdown reverted).", "warn");
    else
      logLine(`Mode switch failed${d && d.error ? ": " + d.error : ""} (dropdown reverted).`, "warn");
    revertMode();
    return;
  }
  instrumentMode = d.mode;                       // applied: the dropdown now matches software
  markModeKnown(true);
  logLine(`Imaging mode → ${d.mode} (${d.backend})`, "rx");
}
$("mode").onchange = () => {
  const m = $("mode").value;
  syncScanChanLabels();                          // reflect the picked mode's error channel
  hydrateScanSetpoint();                          // scan setpoint is per-mode (last_scan_params)
  if (m !== "contact" && m !== "ac"){
    logLine(`Mode "${m}" is not a switchable imaging mode (contact/ac only); instrument mode unchanged.`, "warn");
    return;
  }
  if (m === instrumentMode) return;              // no change to apply
  apiSetMode(m);
};
// When Continuous is ARMED, Frame Up/Down START the backend loop in that
// direction (the supervisor then drives scan_start per frame -- so DON'T also
// fire a single scan/start here). Not armed: the ordinary single-scan path.
$("up").onclick    = () => { if (contArmed){ apiContinuousStart("up");   return; } state.lastScanCfg = scanBody("up");   resetScanFrameFromCfg(state.lastScanCfg); api("scan/start", state.lastScanCfg); };
$("down").onclick  = () => { if (contArmed){ apiContinuousStart("down"); return; } state.lastScanCfg = scanBody("down"); resetScanFrameFromCfg(state.lastScanCfg); api("scan/start", state.lastScanCfg); };
$("stop").onclick  = () => api("scan/abort", { withdraw:true });   // every Stop retracts

// ---- Continuous imaging (backend-owned auto-repeat loop) ------------------
// The toggle only ARMS/DISARMS the service supervisor; the loop itself lives in
// the backend, so this page can reload without stopping it. The readout is
// driven purely by the "continuous" ws broadcast + a boot hydrate -- NO
// setInterval poll loop here (the backend owns the cadence).
// Continuous is an ARMED MODE, not an immediate start: checking the box only
// arms this flag; the loop is actually started by Frame Up/Down (see below),
// which hands the supervisor the direction the operator pressed. `contArmed`
// tracks the box, and is re-synced to real backend state by renderContinuous.
let contArmed = false;
function renderContinuous(st){
  const ro = $("contReadout"); if (!ro) return;
  const cb = $("continuous"); if (cb) cb.checked = !!st.enabled;
  contArmed = !!st.enabled;   // real state wins: running => armed, stopped => not
  if (st.enabled){
    ro.style.color = "var(--accent)";
    ro.textContent = `frame ${st.count||0} · ${Math.round(st.elapsed_s||0)}s`;
  } else if (st.stopped_reason){
    ro.style.color = "var(--dim)";
    ro.textContent = `stopped: ${st.stopped_reason} (${st.count||0} frames)`;
  } else {
    ro.style.color = "var(--dim)";
    ro.textContent = "off";
  }
}
function renderArmed(){
  const ro = $("contReadout"); if (!ro) return;
  ro.style.color = "var(--warn)";
  ro.textContent = "armed — press Frame Up/Down to start";
}
async function apiContinuousStart(direction){
  // Started from Frame Up/Down while armed: the supervisor owns the scanning
  // from here (scan_start per frame), so the button must NOT also fire a single
  // scan/start. Seed the live-frame overlay the same way a single scan would.
  state.lastScanCfg = scanBody(direction); resetScanFrameFromCfg(state.lastScanCfg);
  // blank caps -> null (unlimited); minutes -> seconds for the backend.
  const num = (id) => { const v = $(id).value.trim(); return v === "" ? null : +v; };
  const mins = num("contMaxMin");
  const body = { enabled:true, config: state.lastScanCfg,
                 max_frames: num("contMaxFrames"),
                 max_seconds: (mins == null ? null : mins * 60),
                 interval_s: (num("contInterval") || 0) };
  const d = await api("scan/continuous", body);
  if (d && d.error){ $("continuous").checked = false; contArmed = false; renderContinuous({enabled:false}); return; }
  if (d) renderContinuous(d);
}
async function apiContinuousStop(){
  // Unchecking disarms; if a loop is running the service stops it, and if
  // nothing is running enabled:false is a harmless no-op (readout -> off).
  const d = await api("scan/continuous", { enabled:false });
  if (d) renderContinuous(d);
}
$("continuous").onchange = (e) => {
  if (e.target.checked){ contArmed = true; renderArmed(); }   // ARM only -- no start
  else { contArmed = false; apiContinuousStop(); }
};
async function loadContinuous(){
  const j = (await getScanContinuous()).data;    // request() logs the traffic
  renderContinuous(j);
  if (j && j.relative) renderRelative(j.relative);
  if (j && j.rail) renderRail(j.rail);
}

// ---- Z-drive rail-dwell safety guard readout (safety kernel entry #1) --------
// Driven by the "rail_guard" ws broadcast (per-line poll tick + on-trip) and a
// boot hydrate from the continuous GET (`rail`). Shows the armed indicator, the
// consecutive-railed-line counter (climbing toward the trip), the current frame's
// railed fraction, and -- on a trip -- which guard fired and why. Watchdog only;
// the page never drives it (the backend supervisor owns the poll + trip).
function renderRail(st){
  const ro = $("railReadout"); if (!ro || !st) return;
  if (st.tripped){
    ro.style.color = "var(--err)";
    ro.textContent = `TRIPPED — Guard ${st.tripped.guard}: `
      + (st.tripped.reason || "rail-dwell; continuous stopped, tip retracted");
    return;
  }
  if (st.pending){
    // hardware backend: the guard is gated off until group 2 captures Output.Z on
    // the scan stream (its poll would share the scan pass's bridge link). Show it
    // honestly as not-yet-protecting rather than "armed".
    ro.style.color = "var(--dim)";
    ro.textContent = "pending hardware Z-drive capture";
    return;
  }
  if (st.armed){
    // accent while clean, amber once the counter starts climbing toward the trip.
    ro.style.color = (st.consec > 0) ? "var(--warn)" : "var(--accent)";
    const ff = (st.frame_frac != null) ? Math.round(st.frame_frac * 100) : 0;
    ro.textContent = `armed · railed lines: ${st.consec||0} / ${st.consec_limit||5}`
      + ` · frame ${ff}% railed`;
  } else {
    ro.style.color = "var(--dim)";
    ro.textContent = "off";
  }
}

// ---- Relative (drift-referenced) setpoint ---------------------------------
// A CONTINUOUS-mode option: the backend supervisor, between frames, withdraws +
// settles + reads the free-air deflection + re-approaches at free_air + offset,
// so a drifting deflection baseline never rails the Z servo. This page only sets
// the config (POST /api/scan/relative) and mirrors the readout; the loop owns
// the re-reference. State hydrates from the continuous GET (`relative` field).
let relState = { enabled: false, settle_s: 4.0,
                 last_free_air_v: null, last_effective_v: null };
function renderRelative(st){
  if (st){
    if ("enabled" in st) relState.enabled = st.enabled;
    else if ("mode" in st) relState.enabled = st.mode === "relative";  // legacy echo
    if (st.settle_s != null) relState.settle_s = st.settle_s;
    if ("last_free_air_v" in st) relState.last_free_air_v = st.last_free_air_v;
    if ("last_effective_v" in st) relState.last_effective_v = st.last_effective_v;
  }
  const en = $("relEnable"); if (en) en.checked = relState.enabled;
  const set = $("relSettle"); if (set && document.activeElement !== set) set.value = relState.settle_s;
  const ro = $("relReadout");
  if (ro){
    const fa = relState.last_free_air_v == null ? "—" : `${relState.last_free_air_v>=0?"+":""}${relState.last_free_air_v} V`;
    const ef = relState.last_effective_v == null ? "—" : `${relState.last_effective_v>=0?"+":""}${relState.last_effective_v} V`;
    ro.textContent = `${relState.enabled ? "relative" : "off"} · free-air ${fa}, setpoint ${ef}`;
    ro.style.color = relState.enabled ? "var(--accent)" : "var(--dim)";
  }
}
async function apiRelativeSet(patch){
  logLine(`→ POST /api/scan/relative ${JSON.stringify(patch)}`, "tx");
  const d = await api("scan/relative", patch);
  if (d && d.error){ logLine(`← relative failed: ${d.error}`, "warn"); return; }
  if (d && d.relative) renderRelative(d.relative);
}
$("relEnable").onchange = (e) => apiRelativeSet({ enabled: e.target.checked });
$("relSettle").onchange = (e) => apiRelativeSet({ settle_s: +e.target.value });

// ---- toolbar wiring ------------------------------------------------------
$("loadSvg").onclick = () => $("svgInput").click();
$("svgInput").onchange = (ev) => {
  const f = ev.target.files && ev.target.files[0];
  if (f) importSvg(f);
  ev.target.value = "";              // allow re-selecting the same file
};
// Design view: a two-way outline | tip path switch over the one showTipPath flag.
async function toggleTipPath() {
  showTipPath = !showTipPath;
  for (const [id, on] of [["showToolpath", showTipPath], ["showOutline", !showTipPath]]) {
    $(id).classList.toggle("active", on);
    $(id).setAttribute("aria-pressed", on ? "true" : "false");
  }
  if (showTipPath && !state.designImported)
    logLine("tip path needs an imported design (mock objects have no compiled path)", "warn");
  await refreshToolpath();       // fetch on switch-on; clears the cache on switch-off
  drawOverlay();
}
$("showToolpath").onclick = () => { if (!showTipPath) toggleTipPath(); };
for (const b of document.querySelectorAll("#scanViewSeg .seg-btn"))
  b.addEventListener("click", () => setScanView(b.dataset.view));
$("showOutline").onclick = () => { if (showTipPath) toggleTipPath(); };
$("loadScan").onclick = () => loadLastScan();
// Perform litho: (re)write EVERY enabled object, every time -- {all:true}. Litho
// is not draw-once: written nanowires degrade and get rewritten, so ignoring the
// already-written set is the common case (the backend still skips disabled ones,
// and the compiler does too). amp_gain rides along so the hardware DAC drives
// V_tip / gain. On {error} api() already logged it; just re-enable the button.
// #perform is disabled for the write duration (re-enabled on litho_done).
$("perform").onclick = async () => {
  // N2: an empty setpoint must NOT coerce to 0 (a silent write at free air). Refuse
  // and point at the field instead. The button is also disabled while empty, so this
  // is the keyboard-activation belt-and-suspenders.
  if (writeSetpointEmpty()) {
    logLine("Write refused: set the write setpoint (V above free air) first.", "warn");
    refreshWriteGate();
    return;
  }
  markWriting(true);                          // disable #perform for the write duration
  onRunDone();                                // a run left open by a missed litho_done
  // The explicit id list of the objects shown ENABLED on screen -- never {all:true}
  // (the 2026-09-22 incident: all:true wrote objects a stale screen showed disabled).
  // docs/INSTRUMENT_MODEL_SPEC.md sec 4/10.
  const objects = state.objects
    .filter((o) => o.enabled !== false)
    .map((o) => o.name);
  // The setpoint box is RELATIVE to free air (sec 4/12): the service converts
  // setpoint = hub.free_air.value_v + setpoint_relative at write start. No absolute
  // setpoint is ever sent from the cockpit.
  const setpoint_relative = Number($("lithoSetpoint").value);
  // amp gain comes from the service-owned model, NEVER a typed box (there is no
  // editable gain box anymore). A missing value is sent as-is so the guard refuses
  // honestly (amp_gain_missing) rather than a silent default.
  const amp_gain = getAmpGain();
  const d = await api("litho/write", { objects, setpoint_relative, amp_gain });
  if (!d || d.error) { markWriting(false); return; }
  logLine(`writing ${d.writing} enabled object(s)`, "info");
};
// N2: keep the Write gate in step with the setpoint field as it is typed/cleared.
$("lithoSetpoint").addEventListener("input", () => refreshWriteGate());
// Clear the drawn AFM tip-path trace (display-only). `written` is otherwise
// only ever appended (litho_object_done) and nothing clears it, so the green
// written-material trace lingers forever. This wipes the same overlay fields
// litho_done resets PLUS `written` (which litho_done does NOT), then repaints.
$("clearTrace").onclick = () => {
  state.litho.written = [];
  clearHistory();
  state.litho.live = [];
  state.litho.tipTrail = [];
  state.litho.tip = null;
  logLine("cleared written trace", "tx");
  drawOverlay();
};
// ABORT (NEED #6, safety-critical): the UNIFIED fail-safe primitive. One button,
// one endpoint -- /api/safe_park = stop everything and get the tip to a safe spot
// (out-of-band AFM_SafePark on hardware: td_Stop + stop Z loop + ramp Output.Z->0,
// with a readback verify; stop + withdraw on the twin). The same primitive every
// auto-detected failure path calls, so abort == fail-safe. Always clickable, never
// gated. The ec-safety.js module shows the result banner. See
// docs/SAFETY_KERNEL_PLAN.md Sec 0.
$("abortBtn").onclick = async () => {
  logLine("ABORT: safe_park — stopping + retracting the tip", "warn");
  await api("safe_park", { reason: "operator abort" });
};
// Show tip: the display switch of the live tip marker. On hardware it asks the
// service to start/stop broadcasting tip_pos from the TELEM poll, and the
// marker/trail clear immediately when turned off. On the twin the service always
// streams and the box only shows or hides the marker in THIS page (nothing is sent).
$("showTip").onchange = async () => {
  const on = $("showTip").checked;
  if (state.litho.twin) { state.litho.tipHidden = !on; drawOverlay(); return; }
  await api("tip/show", { on });
  if (!on) { state.litho.tip = null; state.litho.tipTrail = []; drawOverlay(); }
};
// On the twin the marker is on by default, so the box starts checked. Only the
// exact backend "twin" counts: a hardware station reports "hardware" or, before
// its bridge connects, "disconnected".
async function initShowTip() {
  const st = await getState();
  if (!st.ok || !st.data || st.data.backend !== "twin") return;
  state.litho.twin = true;
  $("showTip").checked = true;
}
// Header badge: the service's LIVE backend label (hub.backend_label via /api/state).
const BACKEND_BADGE = {
  twin: ["twin", "badge-twin"],
  hardware: ["hardware", "badge-hardware"],
  disconnected: ["hardware (bridge down)", "badge-disconnected"],
};
export function paintBackendBadge(backend) {
  const el = $("backendBadge"); if (!el) return;
  const [text, cls] = BACKEND_BADGE[backend] || ["not connected", "badge-disconnected"];
  el.textContent = text; el.className = "chip " + cls;
}
async function refreshBackendBadge() {
  const st = await getState();
  paintBackendBadge(st.ok && st.data ? st.data.backend : null);
}
$("clearLog").onclick = () => { $("log").innerHTML = ""; msgCount = 0; lineCount = 0; $("counts").textContent = "0 msgs"; $("logLatest").textContent = ""; };

// =========================================================================
//  TABS -- Scan / Connect / Edit / Write / File. Switching is CSS-only: the
//  inactive bodies are display:none, never removed, so every $() lookup still
//  resolves and every handler still fires. The SHARED objects list (#objPanel)
//  is shown for Write and Edit and hidden for the others.
// =========================================================================
function setTab(name) {
  for (const b of document.querySelectorAll(".tab"))
    b.classList.toggle("active", b.dataset.tab === name);
  $("tabScan").style.display    = name === "scan"    ? "" : "none";
  $("tabWrite").style.display   = name === "write"   ? "" : "none";
  $("tabEdit").style.display    = name === "edit"    ? "" : "none";
  $("tabConnect").style.display = name === "connect" ? "" : "none";
  $("tabFile").style.display    = name === "file"    ? "" : "none";
  // the shared object list rides with Write/Edit; the other tabs hide it
  $("objPanel").style.display = (name === "write" || name === "edit") ? "" : "none";
  // the read-only amp gain rides with the Write setpoint, and every model field
  // lives in the #configDialog modal reached from the header -- nothing to toggle.
  // Edit gives the image the space: no scope, telemetry and log folded
  { const sp = $("scopePanel"); if (sp) sp.style.display = name === "edit" ? "none" : ""; }
  applyRenderTab(name);
  state.currentTab = name;
  editAbortDrafts();   // leaving/entering a tab abandons any edit draft (ec-tab-edit)
  // Connect-tab drafts/mode never bleed into other tabs.
  portMoveDrag = null; portDirDrag = null;
  if (name !== "connect" && placingPort) setPlacing(false);
  applyToolCursor();                // the overlay only shows a tool cursor in Edit/Connect
  logLine(`tab: ${name}`, "dim");
  // Entering Connect re-hydrates the current sample's ports (survives a reload,
  // and picks up a sample switch made while away from the tab). Edit re-hydrates
  // them too so its read-only electrode reference markers (EDIT_TOOLS §4) are
  // available even when Connect was never opened this session.
  if (name === "connect" || name === "edit") { loadPorts(); }
  // The drift toggle/reference/total is an imaging setting on the Scan tab;
  // re-hydrate it on Scan-tab entry so the panel opens showing real state.
  if (name === "scan") { loadDrift(); loadDesignBackgroundChannels(); }
  redraw();                         // resync canvas backing store to its box
}
for (const b of document.querySelectorAll(".tab"))
  b.addEventListener("click", () => setTab(b.dataset.tab));

// =========================================================================
//  EDIT TOOLS -- a first, Inkscape-flavoured tool set on the overlay canvas.
//  All authoring is in MICRONS (evToUm); each tool no-ops unless the Edit tab
//  is active and it is the picked tool, so Scan/Write are never affected.
//    select : click an object to select, drag it to MOVE (op:transform)
//    node   : drag a selected path's vertices     (POST /api/design/points)
//    rect   : click-drag a rubber rectangle -> a NEW shape (POST /api/design/add)
//    pen    : click to place nodes (drag to pull symmetric handles) -> a NEW
//             path; Enter/double-click finishes, Escape cancels
// =========================================================================
const CURSORS = { select: "default", node: "crosshair", rect: "crosshair", pen: "crosshair" };
function applyToolCursor() {
  if (state.currentTab === "edit") ovCv.style.cursor = CURSORS[state.activeTool] || "default";
  else if (state.currentTab === "connect") ovCv.style.cursor = placingPort ? "crosshair" : "default";
  else ovCv.style.cursor = "";
}

// -- The Edit-tab controller (tool palette, pointer handlers, pen close-path
//    gesture, the editable-Bézier node tool, and drawEditLayer) lives in
//    ec-tab-edit.js (Layer 3). ec-main injects the shared DOM/redraw helpers,
//    keeps the overlay-canvas + keydown listeners here (each exported handler
//    still gates on state.currentTab === "edit"), and registered drawEditLayer
//    with the painter registry above. The tool-CURSOR map + applyToolCursor stay
//    here because Connect shares applyToolCursor.
initEditTab({
  evToUm, pxPerUm, drawOverlay, renderList, renderSettings,
  adoptDesign, refreshToolpath, applyToolCursor, ovCv, ovCx, reAdopt,
  clipboard: clipboardActions,
  xyRangeUm: () => getModelNumber("xy_range_um"),
});
ovCv.addEventListener("pointerdown", editPointerDown);
ovCv.addEventListener("pointermove", editPointerMove);
ovCv.addEventListener("pointerup", editPointerUp);
ovCv.addEventListener("dblclick", editDblClick);
window.addEventListener("keydown", editKeydown);
// Zoom/pan gestures; registered on the canvas parent in the capture phase so a pan
// is taken before the Edit and Connect handlers on the canvas see the pointer.
initViewNav({ ovCv, redraw, resetToScan: zoomToScan, applyToolCursor, xyRangeUm: getXyRangeUm });
initZScaleDrag();
window.addEventListener("ec-latch", () => redraw());
initWritten({ redraw: () => drawOverlay() });
// Cursor readout: the pointer in the design's lab µm, through the same inverse
// transform the edit tools use. Listened on the canvas host in the capture phase
// so it keeps following the pointer while a pan holds the pointer capture.
function showCursor(xy) {
  $("cursorPos").textContent = xy
    ? `cursor x ${xy[0].toFixed(3)}  y ${xy[1].toFixed(3)} µm` : "cursor —";
}
ovCv.parentElement.addEventListener("pointermove", (ev) => showCursor(evToUm(ev)), true);
ovCv.parentElement.addEventListener("pointerleave", () => showCursor(null));

// Show-electrodes toggle (Edit tab, EDIT_TOOLS §4): flip the read-only reference
// markers on/off. The painter reads the checkbox each frame; this only forces a
// repaint (and loads the ports on first turn-on, in case Connect was never opened).
{ const se = $("editShowPorts");
  if (se) se.addEventListener("change", () => {
    logLine(`edit: show electrodes ${se.checked ? "on" : "off"}`, "dim");
    if (se.checked && !state.ports.length) loadPorts(); else drawOverlay();
  });
}

// The channel selector: rebuild the grayscale image from the newly-selected
// channel's grid and re-render the scope in that channel, both from the frame
// already in hand -- no re-scan needed.
{ const sc = $("scanChan");
  if (sc) sc.addEventListener("change", () => {
    logLine(`scan channel → ${scanChannelLabel()}`, "dim");
    rebuildScanImg();
    redraw();
    if (state.currentTab === "scan") drawScope();
  });
}

// =========================================================================
//  CONNECT / ELECTRODE PORTS (Phase 1) -- the operator MARKS each existing
//  electrode on the scan as a connection PORT: a point (position_um in the SAME
//  scan-µm frame as the design), a direction the connecting wire comes IN from
//  (its tangent at the pad), a pad width, and an editable label. Ports are
//  backed by /api/connect/ports* and persisted per-sample, so they survive a
//  reload. All interaction is gated on the Connect tab, so Scan/Edit/Write are
//  untouched. Ports live in microns and map with the SAME fitTransform() the
//  design uses, so they land in register with the scan and the objects.
// =========================================================================
const PORT_ARROW_PX = 30;               // fixed on-screen arrow length (incoming dir)
function portById(id) { return state.ports.find((p) => p.id === id) || null; }
function scanCenterUm() {
  return [state.SCAN_FRAME.x0 + state.SCAN_FRAME.w / 2, state.SCAN_FRAME.y0 + state.SCAN_FRAME.h / 2];
}

// -- overlay: a pin, a direction arrow (incoming wire), a perpendicular width
//    bar (length = width_um in scan scale), and the label. Selected = halo +
//    a rotate handle at the arrow tip. No-op off the Connect tab.
function drawArrowhead(cx, ax, ay, bx, by, color) {
  const ang = Math.atan2(by - ay, bx - ax), h = 6;
  cx.strokeStyle = color;
  cx.beginPath();
  cx.moveTo(bx, by); cx.lineTo(bx - h * Math.cos(ang - Math.PI / 6), by - h * Math.sin(ang - Math.PI / 6));
  cx.moveTo(bx, by); cx.lineTo(bx - h * Math.cos(ang + Math.PI / 6), by - h * Math.sin(ang + Math.PI / 6));
  cx.stroke();
}
function drawPortsLayer(tf) {
  if (state.currentTab !== "connect") return;
  const Lum = PORT_ARROW_PX / pxPerUm();     // arrow length in µm at this zoom
  // unselected markers in the text colours, the selected one in the accent
  const acc = cssVar("--accent"), fg = cssVar("--fg"), dim = cssVar("--dim");
  const panel = cssVar("--panel");
  ovCx.save();
  ovCx.lineCap = "round"; ovCx.lineJoin = "round";
  ovCx.font = `11px ${cssVar("--mono")}`;
  for (const p of state.ports) {
    const sel = p.id === state.selectedPort;
    const px = p.position_um[0], py = p.position_um[1];
    const [sx, sy] = tf(px, py);
    // sphere of influence: a faint circle whose DIAMETER = width. It IS the click
    // target for this electrode, and a visual footprint -- if two circles overlap
    // the electrodes are too close (would short), so shrink a width.
    const rpx = (p.width_um || 0) / 2 * pxPerUm();
    if (rpx > 1) {
      ovCx.beginPath(); ovCx.arc(sx, sy, rpx, 0, 2 * Math.PI);
      ovCx.setLineDash([4, 3]);
      ovCx.strokeStyle = sel ? cssVarA("--accent", 0.5) : cssVarA("--dim", 0.32);
      ovCx.lineWidth = 1; ovCx.stroke(); ovCx.setLineDash([]);
    }
    const rad = (p.direction_deg || 0) * Math.PI / 180;
    const ux = Math.cos(rad), uy = Math.sin(rad);       // µm-space (y up)
    // width bar: perpendicular to the direction, centred on the pin
    const half = (p.width_um || 0) / 2;
    const [wax, way] = tf(px + -uy * half, py + ux * half);
    const [wbx, wby] = tf(px - -uy * half, py - ux * half);
    ovCx.strokeStyle = sel ? acc : dim;
    ovCx.lineWidth = sel ? 3 : 2;
    ovCx.beginPath(); ovCx.moveTo(wax, way); ovCx.lineTo(wbx, wby); ovCx.stroke();
    // direction arrow: from the pin along direction_deg (incoming-wire tangent)
    const [tx, ty] = tf(px + ux * Lum, py + uy * Lum);
    ovCx.strokeStyle = sel ? acc : cssVar("--fg");
    ovCx.lineWidth = sel ? 2.5 : 1.8;
    ovCx.beginPath(); ovCx.moveTo(sx, sy); ovCx.lineTo(tx, ty); ovCx.stroke();
    drawArrowhead(ovCx, sx, sy, tx, ty, ovCx.strokeStyle);
    // selection halo
    if (sel) {
      ovCx.beginPath(); ovCx.arc(sx, sy, 9, 0, 2 * Math.PI);
      ovCx.strokeStyle = cssVarA("--accent", 0.55); ovCx.lineWidth = 2; ovCx.stroke();
    }
    // pin dot
    ovCx.beginPath(); ovCx.arc(sx, sy, 4.5, 0, 2 * Math.PI);
    ovCx.fillStyle = sel ? acc : fg; ovCx.fill();
    ovCx.strokeStyle = panel; ovCx.lineWidth = 1; ovCx.stroke();
    // rotate handle at the arrow tip (selected only -- drag it to spin direction)
    if (sel) {
      ovCx.beginPath(); ovCx.arc(tx, ty, 4, 0, 2 * Math.PI);
      ovCx.fillStyle = acc; ovCx.fill();
      ovCx.strokeStyle = panel; ovCx.stroke();
    }
    // label
    ovCx.fillStyle = sel ? cssVar("--fg") : cssVarA("--fg", 0.85);
    ovCx.fillText(p.label, sx + 7, sy - 6);
  }
  ovCx.restore();
}

// -- Edit tab: READ-ONLY electrode reference markers (EDIT_TOOLS §4). Gated on
//    the Edit tab AND the "Show electrodes" toggle. No selection, no hit-testing,
//    no place/move/rotate -- electrodes are authored on the Connect tab; here they
//    are just a routing reference (pin + inward direction arrow + width bar +
//    label). // dup of drawPortsLayer -> shared helper: a deliberately small
//    parallel draw (the Connect layer's markers carry selection/halo/rotate-handle
//    machinery that read-only refs don't need); extract a shared drawPortMarker if
//    a third caller appears.
function drawEditElectrodesLayer(tf) {
  if (state.currentTab !== "edit") return;
  const cb = $("editShowPorts");
  if (!cb || !cb.checked) return;
  const Lum = PORT_ARROW_PX / pxPerUm();     // arrow length in µm at this zoom
  const fg = cssVar("--fg"), panel = cssVar("--panel");
  ovCx.save();
  ovCx.lineCap = "round"; ovCx.lineJoin = "round";
  ovCx.font = `11px ${cssVar("--mono")}`;
  for (const p of state.ports) {
    const px = p.position_um[0], py = p.position_um[1];
    const [sx, sy] = tf(px, py);
    const rad = (p.direction_deg || 0) * Math.PI / 180;
    const ux = Math.cos(rad), uy = Math.sin(rad);        // µm-space (y up)
    // width bar: perpendicular to the direction, centred on the pin
    const half = (p.width_um || 0) / 2;
    const [wax, way] = tf(px + -uy * half, py + ux * half);
    const [wbx, wby] = tf(px - -uy * half, py - ux * half);
    ovCx.strokeStyle = cssVarA("--dim", 0.7); ovCx.lineWidth = 2;
    ovCx.beginPath(); ovCx.moveTo(wax, way); ovCx.lineTo(wbx, wby); ovCx.stroke();
    // direction arrow: from the pin along direction_deg (incoming-wire tangent)
    const [tx, ty] = tf(px + ux * Lum, py + uy * Lum);
    ovCx.strokeStyle = cssVarA("--fg", 0.7); ovCx.lineWidth = 1.6;
    ovCx.beginPath(); ovCx.moveTo(sx, sy); ovCx.lineTo(tx, ty); ovCx.stroke();
    drawArrowhead(ovCx, sx, sy, tx, ty, ovCx.strokeStyle);
    // pin dot
    ovCx.beginPath(); ovCx.arc(sx, sy, 4, 0, 2 * Math.PI);
    ovCx.fillStyle = fg; ovCx.fill();
    ovCx.strokeStyle = panel; ovCx.lineWidth = 1; ovCx.stroke();
    // label
    ovCx.fillStyle = cssVarA("--fg", 0.75);
    ovCx.fillText(p.label, sx + 7, sy - 6);
  }
  ovCx.restore();
}

// -- right panel: ports list (label + position, click to select) with a live
//    count, and per-port settings that persist on change.
function renderPortList() {
  $("portCount").textContent = `${state.ports.length} electrode${state.ports.length === 1 ? "" : "s"}`;
  $("portList").innerHTML = state.ports.map((p) => {
    const sel = p.id === state.selectedPort ? " sel" : "";
    return `<div class="objrow${sel}" data-id="${escapeHtml(p.id)}">`
      + `<span class="swatch" style="background:var(--fg)"></span>`
      + `<span class="nm">${escapeHtml(p.label)}</span>`
      + `<span class="kd">${num(p.position_um[0])}, ${num(p.position_um[1])} µm</span></div>`;
  }).join("");
}
$("portList").addEventListener("click", (ev) => {
  const row = ev.target.closest(".objrow");
  if (row) selectPort(row.dataset.id);
});
function renderPortSettings() {
  const p = state.selectedPort ? portById(state.selectedPort) : null;
  if (!p) { $("portEmpty").style.display = ""; $("portFields").style.display = "none"; return; }
  $("portEmpty").style.display = "none"; $("portFields").style.display = "";
  $("portName").textContent = `electrode ${p.label}`;
  // don't clobber the field the user is actively typing in
  const a = document.activeElement;
  const set = (id, v) => { if (!a || a.id !== id) $(id).value = v; };
  set("pLabel", p.label);
  set("pPosX", num(p.position_um[0]));
  set("pPosY", num(p.position_um[1]));
  set("pDir", num(p.direction_deg));
  set("pWidth", num(p.width_um));
}
function selectPort(id) {
  state.selectedPort = id;
  if (id) logLine(`selected electrode ${portById(id) ? portById(id).label : id}`, "tx");
  renderPortList(); renderPortSettings(); drawOverlay();
}

// -- backend calls: add / update / delete / list. Each replaces the local list
//    with the server's authoritative one and re-renders.
async function apiPortAdd(body) {
  const res = await postPortAdd(body);
  const j = res.data;
  if (!res.ok || j.error) return null;   // request() logged it
  state.ports = Array.isArray(j.ports) ? j.ports : [];
  logLine(`port ${j.id} added (${state.ports.length} total)`, "rx");
  return j.id;
}
async function apiPortUpdate(id, patch) {
  const res = await postPortUpdate({ id, ...patch });
  if (!res.ok || res.data.error) return;   // request() logged it
  state.ports = Array.isArray(res.data.ports) ? res.data.ports : [];
  if (state.selectedPort && !portById(state.selectedPort)) state.selectedPort = null;
  renderPortList(); renderPortSettings(); drawOverlay();
}
async function apiPortDelete(id) {
  const res = await postPortDelete({ id });
  if (!res.ok || res.data.error) return;   // request() logged it
  state.ports = Array.isArray(res.data.ports) ? res.data.ports : [];
  if (state.selectedPort === id) state.selectedPort = null;
  logLine(`port ${id} deleted (${state.ports.length} left)`, "rx");
  renderPortList(); renderPortSettings(); drawOverlay();
}
async function loadPorts() {
  const res = await getPorts();
  const j = res.data;
  if (!res.ok || j.error) return state.ports;   // request() logged it
  state.ports = Array.isArray(j.ports) ? j.ports : [];
  if (state.selectedPort && !portById(state.selectedPort)) state.selectedPort = null;
  logLine(`${state.ports.length} port(s)${j.sample ? ` · sample ${j.sample}` : ""}`, "rx");
  renderPortList(); renderPortSettings(); drawOverlay();
  return state.ports;
}

// =========================================================================
//  TRACK AND COMPENSATE DRIFT (Connect Phase 2)
//  A toggle (#driftEnable) + a Set-reference button (#driftSetRef) backed by
//  /api/drift*. The backend estimates each scan's drift vs the reference, tracks
//  the ports and re-centres the next scan when ON; we mirror its state here and
//  update the readout + port overlay on every "drift" broadcast. Gated on the
//  Connect tab (the panel only shows there); nothing here touches Scan/Edit/Write.
// =========================================================================
// driftState now lives on `state` (ec-state.js).
// nm from µm, signed, for the compact readout
function driftNm(um) { return `${um >= 0 ? "+" : ""}${Math.round(um * 1000)}`; }
function renderDrift(latest) {
  const en = $("driftEnable"); if (en) en.checked = !!state.driftState.enabled;
  const rs = $("driftRefState");
  if (rs) {
    rs.textContent = state.driftState.has_reference ? "reference set" : "no reference";
    rs.style.color = state.driftState.has_reference ? "var(--accent)" : "";
  }
  const ro = $("driftReadout");
  if (ro) {
    const t = state.driftState.total || [0, 0];
    const l = latest
      ? `${driftNm(latest[0])}, ${driftNm(latest[1])} nm`
      : "—";
    ro.textContent = `latest: ${l} · total: ${driftNm(t[0])}, ${driftNm(t[1])} nm`;
  }
}
async function loadDrift() {
  const res = await getDrift();
  const j = res.data;
  if (!res.ok || j.error) return;   // request() logged it
  state.driftState = { enabled: !!j.enabled, has_reference: !!j.has_reference,
                 total: Array.isArray(j.total) ? j.total : [0, 0] };
  const last = Array.isArray(j.track) && j.track.length
    ? [j.track[j.track.length - 1].dx_um, j.track[j.track.length - 1].dy_um] : null;
  logLine(`drift ${state.driftState.enabled ? "ON" : "off"}, ` +
          `${state.driftState.has_reference ? "reference set" : "no reference"}`, "rx");
  renderDrift(last);
}
async function apiDriftEnable(enabled) {
  const res = await postDriftEnable(enabled);
  const j = res.data;
  if (!res.ok || j.error) { state.driftState.enabled = false; renderDrift(); return; }
  state.driftState = { enabled: !!j.enabled, has_reference: !!j.has_reference,
                 total: Array.isArray(j.total) ? j.total : [0, 0] };
  logLine(`track+compensate drift ${state.driftState.enabled ? "ON" : "off"}`, "rx");
  renderDrift();
}
async function apiDriftSetRef() {
  const res = await postDriftReference();
  const j = res.data;
  if (!res.ok || j.error) return;   // request() logged it
  state.driftState = { enabled: !!j.enabled, has_reference: !!j.has_reference,
                 total: Array.isArray(j.total) ? j.total : [0, 0] };
  logLine("drift reference set", "rx");
  renderDrift();
}
{ const b = $("driftEnable"); if (b) b.addEventListener("change", () => apiDriftEnable(b.checked)); }
{ const b = $("driftSetRef"); if (b) b.addEventListener("click", apiDriftSetRef); }

// =========================================================================
//  DESIGN BACKGROUND SWAP (deprecated and hidden; the code stays until removed)
//  "Update design background" (#designBgUpdate) + a channel <select>
//  (#designBgChannel), backed by /api/design/background*: POST, adopt the
//  response, log + render. While #designBgPanel is shown, the select is
//  (re)populated from the newest scan's sidecar channel list on Scan-tab entry;
//  a request() failure or an empty list leaves the built-in
//  height/deflection/amplitude fallback options untouched. Hidden, nothing is sent.
// =========================================================================
async function loadDesignBackgroundChannels() {
  const sel = $("designBgChannel");
  const panel = $("designBgPanel");
  if (!sel || !panel || panel.hidden) return;
  const res = await getDesignBackgroundChannels();
  const j = res.data;
  if (!res.ok || !j || j.error || !Array.isArray(j.channels) || !j.channels.length) return;
  const prev = sel.value;
  sel.innerHTML = "";
  for (const ch of j.channels) {
    const opt = document.createElement("option");
    opt.value = ch; opt.textContent = ch;
    sel.appendChild(opt);
  }
  sel.value = j.channels.includes(prev) ? prev
            : (j.channels.includes("height") ? "height" : j.channels[0]);
}
async function apiDesignBackgroundUpdate() {
  const sel = $("designBgChannel");
  const status = $("designBgStatus");
  const channel = (sel && sel.value) || "height";
  const res = await postDesignBackground(channel);
  const j = res.data;
  if (!res.ok || !j || j.error) {
    if (status) status.textContent = `failed: ${(j && (j.error || j.detail)) || res.error || res.status}`;
    return;   // request() already logged the traffic
  }
  logLine(`design background updated (${j.channel}): ${j.path}`, "rx");
  if (status) status.textContent = j.path;
}
{ const b = $("designBgUpdate"); if (b) b.addEventListener("click", apiDesignBackgroundUpdate); }

// Add a port at a scan-µm point. direction_deg is computed EXPLICITLY to point
// from the point toward the SCAN_FRAME centre (electrodes usually face inward),
// 0 = +x / CCW positive; width defaults to 1 µm. The new port becomes selected.
async function addPortAt(x, y) {
  const [cx, cy] = scanCenterUm();
  const dir = Math.atan2(cy - y, cx - x) * 180 / Math.PI;   // toward centre
  const id = await apiPortAdd({ position_um: [x, y], direction_deg: dir, width_um: 1.0 });
  if (id) { state.selectedPort = id; renderPortList(); renderPortSettings(); drawOverlay(); }
}

// "Place electrode" mode toggle (a click on the scan drops a port while on).
function setPlacing(on) {
  placingPort = on;
  const b = $("placeElectrode");
  if (b) b.classList.toggle("active", on);
  applyToolCursor();
  logLine(on ? "place electrode: press to drop a port, drag to aim its direction" : "place electrode: off", "dim");
}
$("placeElectrode").onclick = () => setPlacing(!placingPort);
$("portDelete").onclick = () => { if (state.selectedPort) apiPortDelete(state.selectedPort); };

// per-port settings fields -> persist on change
$("pLabel").addEventListener("change", () => {
  if (state.selectedPort) apiPortUpdate(state.selectedPort, { label: $("pLabel").value });
});
function persistPortPos() {
  if (state.selectedPort) apiPortUpdate(state.selectedPort,
    { position_um: [Number($("pPosX").value), Number($("pPosY").value)] });
}
$("pPosX").addEventListener("change", persistPortPos);
$("pPosY").addEventListener("change", persistPortPos);
$("pDir").addEventListener("change", () => {
  if (state.selectedPort) apiPortUpdate(state.selectedPort, { direction_deg: Number($("pDir").value) });
});
$("pWidth").addEventListener("change", () => {
  if (state.selectedPort) apiPortUpdate(state.selectedPort, { width_um: Number($("pWidth").value) });
});

// -- pointer wiring on the overlay canvas (Connect tab only). Separate from the
//    Edit-tab handlers -- each set no-ops off its own tab, so they never collide.
ovCv.addEventListener("pointerdown", async (ev) => {
  if (state.currentTab !== "connect" || ev.button !== 0) return;
  const [x, y] = evToUm(ev);
  ovCv.setPointerCapture(ev.pointerId);
  if (placingPort) {
    // press sets the LOCATION; dragging out aims the connection DIRECTION (release
    // persists it). A plain click with no drag keeps the toward-centre default.
    await addPortAt(x, y);
    if (state.selectedPort) portDirDrag = { id: state.selectedPort };
    return;
  }
  const tolUm = 10 / pxPerUm();
  // a rotate handle on the current selection takes priority over a re-pick
  if (state.selectedPort) {
    const p = portById(state.selectedPort);
    if (p) {
      const Lum = PORT_ARROW_PX / pxPerUm();
      const rad = (p.direction_deg || 0) * Math.PI / 180;
      const tipx = p.position_um[0] + Math.cos(rad) * Lum;
      const tipy = p.position_um[1] + Math.sin(rad) * Lum;
      if (Math.hypot(tipx - x, tipy - y) <= tolUm) { portDirDrag = { id: p.id }; return; }
    }
  }
  // hit-test each electrode's CIRCLE OF INFLUENCE (radius = width/2). Nearest
  // containing circle wins; a click outside every circle is empty space.
  let hit = null, best = Infinity;
  for (const p of state.ports) {
    const r = (p.width_um || 0) / 2;
    const d = Math.hypot(p.position_um[0] - x, p.position_um[1] - y);
    if (d <= r && d < best) { best = d; hit = p; }
  }
  if (hit) {
    selectPort(hit.id);
    // single click (+drag) = MOVE the location; double click (+drag) = set the ANGLE.
    const now = Date.now();
    const dbl = lastPortDown.id === hit.id && (now - lastPortDown.t) < 350;
    lastPortDown = { id: hit.id, t: now };
    if (dbl) portDirDrag = { id: hit.id };
    else portMoveDrag = { id: hit.id, orig: [hit.position_um[0], hit.position_um[1]], moved: false };
  } else {
    selectPort(null);
  }
});
ovCv.addEventListener("pointermove", (ev) => {
  if (state.currentTab !== "connect") return;
  const [x, y] = evToUm(ev);
  if (portDirDrag) {
    const p = portById(portDirDrag.id); if (!p) return;
    p.direction_deg = Math.atan2(y - p.position_um[1], x - p.position_um[0]) * 180 / Math.PI;
    renderPortSettings(); drawOverlay();
  } else if (portMoveDrag) {
    const p = portById(portMoveDrag.id); if (!p) return;
    p.position_um = [x, y]; portMoveDrag.moved = true;
    renderPortList(); renderPortSettings(); drawOverlay();
  }
});
ovCv.addEventListener("pointerup", async (ev) => {
  if (state.currentTab !== "connect") return;
  try { ovCv.releasePointerCapture(ev.pointerId); } catch (e) {}
  if (portDirDrag) {
    const p = portById(portDirDrag.id); portDirDrag = null;
    if (p) await apiPortUpdate(p.id, { direction_deg: p.direction_deg });
  } else if (portMoveDrag) {
    const md = portMoveDrag; portMoveDrag = null;
    const p = portById(md.id);
    if (p && md.moved) await apiPortUpdate(p.id, { position_um: p.position_um });
  }
});
// Del/Backspace deletes the selected port; Escape leaves placement mode. Gated
// on the Connect tab and ignored while typing in a field.
window.addEventListener("keydown", (ev) => {
  if (state.currentTab !== "connect") return;
  const t = ev.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
  if ((ev.key === "Delete" || ev.key === "Backspace") && state.selectedPort) {
    ev.preventDefault(); apiPortDelete(state.selectedPort);
  } else if (ev.key === "Escape" && placingPort) {
    setPlacing(false);
  }
});

// ---- boot ----------------------------------------------------------------
// The canvases follow their box, which moves with the window and the pane divider.
function refit() { redraw(); }
window.addEventListener("resize", refit);
if (typeof ResizeObserver === "function") new ResizeObserver(refit).observe(scanCv.parentElement);
initPanes();
initRenderPane();
initSideScroll($("renderTools"));
window.addEventListener("DOMContentLoaded", () => {
  state.view = { ...state.SCAN_FRAME };
  setViewCap("scan frame");
  renderList();
  renderSettings();   // empty state
  setTab("scan");     // default active tab (hides the shared objects list)
  redraw();           // also paints the empty scope grid until lines arrive
  logLine(`edit cockpit loaded — ${state.objects.length} mock objects`, "dim");
  loadLastScan();     // paint the newest cached scan (falls back to placeholder)
  loadCurrentDesign();// hydrate from hub.pattern if the backend already has one
  loadPorts();        // hydrate the current sample's electrode ports (Connect tab)
  loadContinuous();   // hydrate the continuous-imaging toggle/readout (survives reload)
  refreshBackendBadge();
  setInterval(refreshBackendBadge, 5000);
  initShowTip();      // twin: "show tip" starts on
  // Instrument model (docs/INSTRUMENT_MODEL_SPEC.md sec 1/4/10): wire the panel
  // buttons + seed the backend flag, hydrate the read-only model + the two setpoint
  // boxes from the service (no HTML literal survives), then refresh the write
  // estimate. hydrateInstrument runs AFTER loadMode so the per-mode scan setpoint
  // hydrates against the ACTUAL instrument mode, not the dropdown default.
  initInstrument();
  // hydrate the model, then re-render so swatches pick up the window-floored colour
  // scale (getVoltageWindowMax) now that the model is known.
  loadMode().then(() => hydrateInstrument()).then(() => {
    renderList(); drawOverlay();
    refreshWriteGate();   // N2: reflect whether the (possibly un-hydrated) setpoint is empty
  });
  refreshWriteEstimate();
  // Open the live websocket feed (scan lines + telemetry). The ws lives in
  // ec-api.js now; these hooks reproduce the old connect() exactly, incl. the
  // status chip, the "websocket open/closed" log lines, the per-frame counter
  // (counted BEFORE the parse, so unparseable frames count too), and the
  // non-json warning.
  connectWs({
    onConnecting: () => setChip("ws", "ws: connecting", "busy"),
    onOpen: () => { setChip("ws", "ws: open", "on"); refreshBackendBadge(); logLine("websocket open", "dim"); },
    onClose: () => {
      setChip("ws", "ws: closed", "bad"); logLine("websocket closed; reconnecting in 1s", "warn");
      // frames lost with the link (scan_done, litho_done) must not leave a scan or a
      // write looking open
      onRunDone();
      if (scanRowNow != null) { scanRowNow = null; redraw(); }
    },
    onError: () => logLine("websocket error", "warn"),
    onFrame: () => { msgCount++; $("counts").textContent = `${msgCount} msgs`; },
    onParseFail: (data) => logLine("non-json ws frame: " + data.slice(0, 160), "warn"),
  });
});
