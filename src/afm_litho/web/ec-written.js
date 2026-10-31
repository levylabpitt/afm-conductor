// The written-trace history behind the render toolbar's `written ▾` control. The
// service sends one litho_object_done (or _partial) per STROKE, and a filled shape is
// many strokes named "<object>#k"; each carries its design object in `object`. A
// "write" here is one design object in one Perform litho run: the strokes are
// grouped by that, client-side, as they arrive. A run starts at the first
// litho_object_start after a litho_done (or after the page loaded).
//
// The checkbox shows or hides the whole written overlay; the list shows or hides
// each write. Display only: nothing is sent to the service.
//
// Import-safe: no DOM access until initWritten() runs.

import { state } from "./ec-state.js";

let run = 0;              // the current run's number, 0 before the first
let runOpen = false;      // a litho_object_start arrived and no litho_done yet
let nextId = 1;
// The writes by id: strokeShown runs once per stroke on every overlay draw.
const byId = new Map();

// The write a stroke belongs to, made on its first stroke. Pure over `writes`.
export function addStroke(writes, runNo, m, now) {
  const object = m.object || m.name;
  let w = writes.find((g) => g.run === runNo && g.object === object);
  if (!w) {
    w = { id: nextId++, run: runNo, object, voltage: m.voltage, t: now,
          shown: true, partial: false, strokes: 0 };
    writes.push(w);
  }
  w.strokes++;
  if (m.partial) w.partial = true;
  return w;
}

// "HH:MM · +3.0 V" for a write's row.
export function writeMeta(w) {
  const d = new Date(w.t);
  const hm = `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  const v = Number(w.voltage);
  return Number.isFinite(v) ? `${hm} · ${v >= 0 ? "+" : ""}${v.toFixed(1)} V` : hm;
}

export function onObjectStart() {
  if (!runOpen) { run++; runOpen = true; }
}
// litho_done, or anything that means the run is over without one (a websocket close,
// the next Perform litho), so a later run never merges into this one.
export function onRunDone() { runOpen = false; }

// Tag a written-trace entry with its write, from the litho_object_done/_partial frame.
export function recordStroke(entry, m) {
  if (!run) run = 1;
  const w = addStroke(state.litho.writes, run, { ...m, partial: entry.partial }, Date.now());
  byId.set(w.id, w);
  entry.write = w.id;
  renderList();
}

// Whether a written-trace entry is drawn: the overlay is on and its write is shown.
export function strokeShown(entry) {
  if (!state.litho.showWritten) return false;
  const w = byId.get(entry.write);
  return !w || w.shown;
}

export function clearHistory() {
  state.litho.writes = [];
  byId.clear();
  renderList();
}

let redraw = () => {};
const $ = (id) => document.getElementById(id);

function escapeText(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderList() {
  const box = typeof document !== "undefined" ? $("writtenList") : null;
  if (!box) return;
  const writes = state.litho.writes;
  if (!writes.length) { box.innerHTML = `<div class="wempty">no writes yet</div>`; return; }
  // newest first, as the operator looks for the write just made
  box.innerHTML = [...writes].reverse().map((w) =>
    `<label class="wrow${w.partial ? " partial" : ""}" title="${escapeText(w.object)}: `
    + `${w.strokes} stroke${w.strokes === 1 ? "" : "s"}${w.partial ? ", cut by an abort" : ""}">`
    + `<input type="checkbox" data-write="${w.id}"${w.shown ? " checked" : ""}>`
    + `<span class="wname">${escapeText(w.object)}${w.partial ? " (partial)" : ""}</span>`
    + `<em>${escapeText(writeMeta(w))}</em></label>`).join("");
}

function setOpen(open) {
  const box = $("writtenList"), btn = $("writtenMenu");
  if (!box || !btn) return;
  if (open) {
    renderList();
    // under the arrow, inside the render toolbar's wrapper (the strip itself scrolls
    // sideways and would clip it)
    const wrap = box.offsetParent || box.parentElement;
    const br = btn.getBoundingClientRect(), wr = wrap.getBoundingClientRect();
    box.hidden = false;
    const left = Math.max(0, Math.min(br.left - wr.left, wr.width - box.offsetWidth));
    box.style.left = `${left}px`;
  } else {
    box.hidden = true;
  }
  btn.setAttribute("aria-expanded", String(open));
}

export function initWritten(deps) {
  ({ redraw } = deps);
  const cb = $("showWritten"), btn = $("writtenMenu"), box = $("writtenList");
  if (!cb || !btn || !box) return;
  cb.checked = state.litho.showWritten;
  cb.addEventListener("change", () => { state.litho.showWritten = cb.checked; redraw(); });
  btn.addEventListener("click", () => setOpen(box.hidden));
  box.addEventListener("change", (ev) => {
    const id = Number(ev.target && ev.target.dataset && ev.target.dataset.write);
    const w = byId.get(id);
    if (!w) return;
    w.shown = ev.target.checked;
    redraw();
  });
  // a press outside the list or its arrow, or Escape, closes it
  document.addEventListener("pointerdown", (ev) => {
    if (box.hidden || box.contains(ev.target) || btn.contains(ev.target)) return;
    setOpen(false);
  }, true);
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && !box.hidden) setOpen(false);
  });
  renderList();
}
