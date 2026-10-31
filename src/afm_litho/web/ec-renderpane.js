// Render-pane arrangement: where the telemetry readout sits (column beside the square
// image, thin folded tab, or grid under the image), the command log's one-line fold,
// and the Edit tab's override that gives the image the space. The operator's own
// fold choices are stored per browser; Edit overrides them only while it is shown.
//
// Import-safe: no DOM access until initRenderPane() runs.

import { halfImageSide, telemetryLayout } from "./ec-render-layout.js";
import { getModelNumber } from "./ec-instrument.js";
import { state } from "./ec-state.js";

const KEY_TELE = "ec.telemetryCollapsed";
const KEY_LOG = "ec.logCollapsed";

// Storage can be blocked (private window, site data off) and then throws; the page
// works the same, only the choice is not remembered.
function readFlag(key) {
  try { return window.localStorage.getItem(key) === "1"; } catch { return false; }
}
function writeFlag(key, on) {
  try { window.localStorage.setItem(key, on ? "1" : "0"); } catch { /* blocked */ }
}

const pref = { tele: false, log: false };   // the operator's stored choices
let editFold = null;                        // {tele, log} while the Edit tab is shown
// The log's fold while the half-screen arrangement is shown (null otherwise): folded
// on arrival, so the image and scope get the height; a toggle lasts for that stay.
let halfLog = null;
let wasHalf = false;
let lastFit = null;                         // the telemetry fit last chosen (hysteresis)
let tab = "scan";

const teleFolded = () => (editFold ? editFold.tele : pref.tele);
const logFolded = () => (editFold ? editFold.log : halfLog ?? pref.log);

// Telemetry bar ranges the page already knows: the controller's ±rail for deflection
// and SUM; the park level to the rail-dwell threshold for the Z drive (the model
// carries no hard Output.Z rail, so the threshold is the bar's labelled end); the
// running scan's line count for the row, else the Lines field.
function telemetryRange(id) {
  const rail = getModelNumber("dac_rail_v");
  if (id === "defl") return rail ? { lo: -rail, hi: rail } : null;
  if (id === "sum") return rail ? { lo: 0, hi: rail } : null;
  if (id === "zdrive") {
    const dwell = getModelNumber("z_rail_v");
    const park = getModelNumber("z_park_v") ?? 0;
    return dwell ? { lo: Math.min(0, park), hi: dwell, hiText: `rail-dwell ${dwell}` } : null;
  }
  if (id === "row") {
    const n = (state.scanFrame && state.scanFrame.rows)
      || Number(document.getElementById("lines")?.value);
    return n > 0 ? { lo: 0, hi: n } : null;
  }
  return null;
}

// The telemetry readout (ec-telemetry.js, its own page entry) by its window events.
function telemetryChannelCount() {
  return Number(document.getElementById("ecTelemetry")?.dataset.channels) || 0;
}
function sendTelemetry(name, detail) {
  window.dispatchEvent(new CustomEvent(name, { detail }));
}

// True when the render pane's container query chose the half-screen arrangement
// (edit-cockpit.css sets --arr on #renderBody).
export function isHalfArrangement() {
  const body = document.getElementById("renderBody");
  return !!body && getComputedStyle(body).getPropertyValue("--arr").trim() === "half";
}

// Half screen: the image row holds the z bar over a square image as wide as the
// column, and the scope under it takes the remaining height. Elsewhere, and while
// the scope is hidden (Edit), the image row takes the height as the CSS gives it.
function sizeImageRow(half, imgCol, slot, scope) {
  const row = imgCol.querySelector(".imgrow");
  const zbar = document.getElementById("zScale");
  if (!row) return;
  const scopeOn = !!scope && scope.style.display !== "none";
  if (!half || !scopeOn) { row.style.flex = ""; return; }
  const colGap = parseFloat(getComputedStyle(imgCol).rowGap) || 0;
  const rowGap = parseFloat(getComputedStyle(row).rowGap) || 0;
  const zh = zbar ? zbar.offsetHeight : 0;
  const slotH = slot.offsetHeight ? slot.offsetHeight + colGap : 0;
  const avail = imgCol.clientHeight - slotH - colGap - zh - rowGap;
  row.style.flex = `0 0 ${zh + rowGap + halfImageSide(imgCol.clientWidth, avail)}px`;
}

function layout() {
  const body = document.getElementById("renderBody");
  const tele = document.getElementById("ecTelemetry");
  const imgCol = document.getElementById("imgCol");
  const slot = document.getElementById("telemGridSlot");
  if (!body || !tele || !imgCol || !slot) return;
  const half = isHalfArrangement();
  if (half !== wasHalf) {
    wasHalf = half;
    halfLog = half ? true : null;
    applyLog();
  }
  const scope = document.getElementById("scopePanel");
  const zbar = document.getElementById("zScale");
  const scopeH = scope && scope.style.display !== "none" ? scope.offsetHeight + 8 : 0;
  const imgH = imgCol.clientHeight - scopeH;
  const spare = body.clientWidth - (zbar ? zbar.offsetWidth : 0) - imgH;
  const fit = telemetryLayout({
    spare, height: body.clientHeight, n: telemetryChannelCount(), collapsed: teleFolded(), half,
    prev: lastFit,
  });
  lastFit = fit;
  const home = fit.mode === "grid" ? slot : body;
  if (tele.parentElement !== home) home.insertBefore(tele, home.firstChild);
  tele.classList.toggle("tele-col", fit.mode === "column");
  tele.classList.toggle("tele-tab", fit.mode === "tab");
  tele.classList.toggle("tele-grid", fit.mode === "grid");
  tele.style.width = fit.mode === "grid" ? "" : `${fit.width}px`;
  const tog = document.getElementById("telemToggle");
  if (tog) tog.setAttribute("aria-expanded", String(fit.mode !== "tab"));
  sendTelemetry("ec-telemetry-fit", fit);
  sizeImageRow(half, imgCol, slot, scope);
}

function applyLog() {
  const panel = document.querySelector(".logpanel");
  const btn = document.getElementById("logToggle");
  const folded = logFolded();
  if (panel) panel.classList.toggle("collapsed", folded);
  if (btn) btn.setAttribute("aria-expanded", String(!folded));
  if (!folded) {
    const log = document.getElementById("log");
    const auto = document.getElementById("autoscroll");
    if (log && (!auto || auto.checked)) log.scrollTop = log.scrollHeight;
  }
}

function toggleTele() {
  if (editFold) editFold.tele = !editFold.tele;
  else { pref.tele = !pref.tele; writeFlag(KEY_TELE, pref.tele); }
  layout();
}
function toggleLog() {
  if (editFold) editFold.log = !editFold.log;
  else if (halfLog !== null) halfLog = !halfLog;
  else { pref.log = !pref.log; writeFlag(KEY_LOG, pref.log); }
  applyLog();
}

// The Edit tab folds telemetry and the log (the scope is hidden by setTab); any other
// tab shows the operator's own choices again. A fold toggled while in Edit lasts only
// for that visit.
export function applyRenderTab(name) {
  const was = tab;
  tab = name;
  if (name === "edit" && was !== "edit") editFold = { tele: true, log: true };
  else if (name !== "edit") editFold = null;
  applyLog();
  layout();
}

export function relayoutRenderPane() { layout(); }

export function initRenderPane() {
  pref.tele = readFlag(KEY_TELE);
  pref.log = readFlag(KEY_LOG);
  for (const id of ["telemToggle", "telemTab"]) {
    const b = document.getElementById(id);
    if (b) b.addEventListener("click", toggleTele);
  }
  const lb = document.getElementById("logToggle");
  if (lb) lb.addEventListener("click", toggleLog);
  window.addEventListener("ec-telemetry-channels", () => layout());
  window.addEventListener("ec-telemetry-ready", () => {
    sendTelemetry("ec-telemetry-ranges", telemetryRange);
    if (lastFit) sendTelemetry("ec-telemetry-fit", lastFit);
  });
  sendTelemetry("ec-telemetry-ranges", telemetryRange);
  const body = document.getElementById("renderBody");
  if (body && typeof ResizeObserver === "function") new ResizeObserver(layout).observe(body);
  window.addEventListener("resize", layout);
  applyLog();
  layout();
}
