// Live telemetry readout in the render pane, fed by the `hw_telemetry` block of the /ws
// `telemetry` frame (the hardware TELEM poll, or the twin's simulated channels). The
// channel set is whatever the frame carries, through TELEMETRY_CHANNELS. Output.Z is
// shown in volts: that is the domain the rail-dwell guard watches. Amplitude is read
// from the frame's top-level amp_v, which the service blanks outside AC mode.
//
// It is its own module entry with its own websocket and imports only pure helpers, so
// an import or boot error elsewhere in the cockpit cannot stop it. Nothing imports it:
// the service stamps the entry's URL with ?v=<mtime>, so an import by plain path would
// load a second instance with a second websocket. ec-renderpane.js talks to it by
// window events instead (TELEMETRY_EVENTS below): it places the readout and sends how
// much it shows and the bar ranges; without them every channel still shows its label
// and number.

import {
  TELEMETRY_CHANNELS, channelValues, visibleChannels, valueText, barFraction, pushHistory,
  isNum, telemetryStaleness,
} from "./ec-render-layout.js";
import { cssVar } from "./ec-theme.js";

const HISTORY_LEN = 120;
const history = new Map();       // channel id -> recent values
let shown = [];                  // channels currently drawn
let lastMsg = null;
let fit = { mode: "column", bars: false, spark: false };
// A stalled hardware poll keeps sending the same frame time, and a dropped link sends
// nothing: either way the values must not look live.
let linkOpen = false;
let lastT = null;          // hw_telemetry.t of the newest frame
let advancedAt = null;     // performance.now() when that time last changed
const STALE_CHECK_MS = 250;

// The row bar from the scan line count in the Scan fields; the full set of ranges
// comes from setRangeSource. A range is {lo, hi, hiText?}; null draws no bar.
function linesRange(id) {
  if (id !== "row") return null;
  const n = Number(document.getElementById("lines")?.value);
  return n > 0 ? { lo: 0, hi: n } : null;
}
let rangeFor = linesRange;

const fmtTick = (v) => (Math.abs(v) >= 100 || Number.isInteger(v) ? String(Math.round(v)) : v.toFixed(1));

function body() { return document.getElementById("telemBody"); }

function buildRows() {
  const b = body(); if (!b) return;
  b.innerHTML = "";
  for (const ch of shown) {
    const row = document.createElement("div");
    row.className = "tch"; row.dataset.ch = ch.id;
    row.innerHTML =
      `<div class="tlbl">${ch.label}</div>` +
      `<div class="tline"><span class="tnum">—</span><span class="tunit">${ch.unit}</span>` +
      `<canvas class="tspark"></canvas></div>` +
      `<div class="tbar"><i></i></div><div class="tbarlbl"><span></span><span></span></div>`;
    b.appendChild(row);
  }
}

function drawSpark(cv, buf) {
  const r = cv.getBoundingClientRect();
  const W = Math.max(1, Math.round(r.width)), H = Math.max(1, Math.round(r.height));
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const cx = cv.getContext("2d");
  cx.clearRect(0, 0, W, H);
  let lo = Infinity, hi = -Infinity;
  for (const v of buf) if (isNum(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (!Number.isFinite(lo)) return;
  if (hi - lo < 1e-12) { lo -= 1; hi += 1; }
  cx.strokeStyle = cssVar("--accent"); cx.lineWidth = 1; cx.beginPath();
  let started = false;
  buf.forEach((v, i) => {
    if (!isNum(v)) { started = false; return; }
    // newest value at the right edge; a short buffer fills from the right
    const x = (W - 1) - (buf.length - 1 - i) * ((W - 1) / (HISTORY_LEN - 1));
    const y = H - 1 - ((v - lo) / (hi - lo)) * (H - 2);
    if (started) cx.lineTo(x, y); else { cx.moveTo(x, y); started = true; }
  });
  cx.stroke();
}

function paint() {
  const b = body(); if (!b || !lastMsg) return;
  for (const ch of shown) {
    const row = b.querySelector(`.tch[data-ch="${ch.id}"]`); if (!row) continue;
    const vals = channelValues(ch, lastMsg);
    row.querySelector(".tnum").textContent = valueText(ch, vals);
    const single = vals.length === 1;
    const range = single ? rangeFor(ch.id) : null;
    const frac = fit.bars && range ? barFraction(vals[0], [range.lo, range.hi]) : null;
    const hasBar = fit.bars && range != null;
    row.classList.toggle("has-bar", hasBar);
    if (hasBar) {
      row.querySelector(".tbar > i").style.width = `${((frac ?? 0) * 100).toFixed(1)}%`;
      const [l, h] = row.querySelectorAll(".tbarlbl > span");
      l.textContent = fmtTick(range.lo); h.textContent = range.hiText || fmtTick(range.hi);
    }
    const spark = fit.spark && single && ch.id !== "row";
    row.classList.toggle("has-spark", spark);
    if (spark) drawSpark(row.querySelector(".tspark"), history.get(ch.id) || []);
  }
}

function onFrame(m) {
  lastMsg = m;
  const t = m.hw_telemetry.t;
  if (t !== lastT || advancedAt == null) { lastT = t; advancedAt = performance.now(); }
  for (const ch of TELEMETRY_CHANNELS) {
    const vals = channelValues(ch, m);
    if (vals.length === 1) {
      if (!history.has(ch.id)) history.set(ch.id, []);
      pushHistory(history.get(ch.id), vals[0], HISTORY_LEN);
    }
  }
  const next = visibleChannels(m);
  if (next.map((c) => c.id).join() !== shown.map((c) => c.id).join()) {
    shown = next;
    buildRows();
    const el = document.getElementById("ecTelemetry");
    if (el) {
      el.classList.toggle("has-data", shown.length > 0);
      el.dataset.channels = String(shown.length);
    }
    window.dispatchEvent(new Event("ec-telemetry-channels"));
  }
  paint();
  paintStale();
}

function paintStale() {
  const el = document.getElementById("ecTelemetry");
  const tag = document.getElementById("telemStale");
  if (!el || !tag) return;
  const age = advancedAt == null ? null : (performance.now() - advancedAt) / 1000;
  const s = telemetryStaleness(linkOpen, age);
  el.classList.toggle("stale", s.stale);
  tag.hidden = !s.stale;
  tag.textContent = s.text;
  tag.title = s.why ? `Telemetry is not live: ${s.why}` : "";
}


function connect() {
  let ws;
  try {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
  } catch { return; }
  ws.onopen = () => { linkOpen = true; paintStale(); };
  ws.onmessage = (ev) => {
    let m;
    // the service serializes with allow_nan=True, so bare NaN/Infinity tokens occur
    try { m = JSON.parse(ev.data); }
    catch {
      try { m = JSON.parse(ev.data.replace(/\bNaN\b/g, "null").replace(/-?\bInfinity\b/g, "null")); }
      catch { return; }
    }
    if (m && m.type === "telemetry" && m.hw_telemetry) onFrame(m);
  };
  // reconnect on drop so an overnight run keeps reporting
  ws.onclose = () => { linkOpen = false; paintStale(); setTimeout(connect, 3000); };
  ws.onerror = () => { try { ws.close(); } catch { /* already closed */ } };
}

// The window events between this readout and ec-renderpane.js:
//   ec-telemetry-fit      in   detail: how much it shows, {mode, bars, spark}
//   ec-telemetry-ranges   in   detail: fn(channelId) -> {lo, hi, hiText?} or null
//   ec-telemetry-channels out  the drawn channel set changed; the count is on
//                              #ecTelemetry's data-channels
//   ec-telemetry-ready    out  this readout is listening (the pane re-sends fit/ranges)
window.addEventListener("ec-telemetry-fit", (ev) => { fit = ev.detail; paint(); });
window.addEventListener("ec-telemetry-ranges", (ev) => { rangeFor = ev.detail; paint(); });
connect();
setInterval(paintStale, STALE_CHECK_MS);
window.dispatchEvent(new Event("ec-telemetry-ready"));
