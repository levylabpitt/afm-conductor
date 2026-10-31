// Zoom and pan of the shared canvas view (docs/COCKPIT_ZOOM_PAN_SPEC.md). Every
// canvas layer draws through state.view, so these gestures only change that box
// and redraw; nothing in the design, the service or the instrument moves.
//
//   Ctrl+wheel / touchpad pinch     zoom about the pointer   (every canvas tab)
//   wheel / two-finger scroll       pan                      (every canvas tab)
//   right-drag, Space+left-drag     pan                      (every canvas tab)
//   left-drag                       pan                      (Scan tab only)
//   double-click                    back to the scan frame   (Scan tab only)

import { state } from "./ec-state.js";
import { num } from "./ec-log.js";
import {
  zoomViewAbout, clampZoomFactor, wheelZoomFactor, pxPerUm, invTransform, boundsOfObjects,
  unionBox, clampViewToBounds, FALLBACK_XY_RANGE_UM,
} from "./ec-geom.js";

// Smallest view (larger side, um).
export const VIEW_MIN_UM = 0.1;
// Margin around the navigable area, as a fraction of its larger side.
export const VIEW_BOUNDS_MARGIN = 0.1;
// Zoom per wheel pixel: a 100 px mouse notch is ~18 %, a pinch step a few percent.
const ZOOM_PER_WHEEL_PX = 0.002;

const $ = (id) => document.getElementById(id);

let ovCv = null, redraw = null, resetToScan = null, applyToolCursor = null;
let xyRangeUm = () => null;   // the instrument's [x, y] scan range in um, or null
let pan = null;            // {pointerId, view, bx, by, right}: the drag start, while a drag pans
let spaceHeld = false;
let overCanvas = false;
let frameQueued = false;
// Armed by a right-button pan. Windows fires contextmenu at mouseup on the element
// under the pointer, not on the canvas holding capture, so a pan released off the
// canvas would open the menu there; the next contextmenu is swallowed once.
let swallowMenu = false;
let swallowTimer = null;

function typingInField(t) {
  return !!t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA"
                 || t.tagName === "SELECT" || t.isContentEditable);
}

// The area the view may pan and zoom out over: the design, the instrument's scan
// range and the scan frame together, plus a margin, so the operator cannot get lost
// in empty space. Before the instrument model loads, the range is the widest any
// supported scanner has (FALLBACK_XY_RANGE_UM), so the bounds never stop an ordinary
// pan on a range the page has not read yet.
export function viewBounds() {
  const boxes = [state.SCAN_FRAME];
  const r = xyRangeUm() || [FALLBACK_XY_RANGE_UM, FALLBACK_XY_RANGE_UM];
  if (r) boxes.push({ x0: -r[0] / 2, y0: -r[1] / 2, w: r[0], h: r[1] });
  if (state.objects && state.objects.length) boxes.push(boundsOfObjects(state.objects));
  return unionBox(boxes, VIEW_BOUNDS_MARGIN);
}

// Largest view (larger side, um): the whole navigable area.
export function viewMaxUm() {
  const b = viewBounds();
  return Math.max(b.w, b.h);
}

// Backing-store pixel of a pointer event (the canvas CSS box may differ in size).
function evPx(ev) {
  const r = ovCv.getBoundingClientRect();
  return [(ev.clientX - r.left) * ovCv.width / (r.width || 1),
          (ev.clientY - r.top) * ovCv.height / (r.height || 1)];
}

function curView() { return state.view || state.SCAN_FRAME; }

function setView(v) {
  v = clampViewToBounds(v, viewBounds());
  state.view = v;
  $("viewCap").textContent =
    `view: ${num(v.w)}×${num(v.h)} µm @ (${num(v.x0 + v.w / 2)}, ${num(v.y0 + v.h / 2)})`;
  if (frameQueued) return;
  frameQueued = true;
  requestAnimationFrame(() => { frameQueued = false; redraw(); });
}

function zoomAt(ev, f) {
  const v = curView();
  f = clampZoomFactor(v, f, VIEW_MIN_UM, viewMaxUm());
  if (f === 1) return;
  const [px, py] = evPx(ev);
  const [ux, uy] = invTransform(v, ovCv.width, ovCv.height)(px, py);
  setView(zoomViewAbout(v, ux, uy, f));
}

// Scroll pans like a document: content follows the fingers, so a downward
// scroll (deltaY > 0) shows what lies below (lower micron y).
function panByPx(v, dxPx, dyPx) {
  const s = pxPerUm(v, ovCv.width, ovCv.height);
  return { x0: v.x0 + dxPx / s, y0: v.y0 - dyPx / s, w: v.w, h: v.h };
}

function onWheel(ev) {
  if (ev.target !== ovCv) return;
  ev.preventDefault();                 // no page scroll, no browser page zoom
  if (ev.ctrlKey || ev.metaKey) {
    zoomAt(ev, wheelZoomFactor(ev.deltaY, ev.deltaMode, ZOOM_PER_WHEEL_PX));
    return;
  }
  const k = ev.deltaMode === 1 ? 16 : ev.deltaMode === 2 ? ovCv.height : 1;
  const r = ovCv.getBoundingClientRect();
  const scale = ovCv.width / (r.width || 1);
  setView(panByPx(curView(), ev.deltaX * k * scale, ev.deltaY * k * scale));
}

function panStarts(ev) {
  if (ev.button === 2) return true;
  if (ev.button !== 0) return false;
  return spaceHeld || state.currentTab === "scan";
}

function setGrabCursor(c) { ovCv.style.cursor = c; }
function restoreCursor() { applyToolCursor(); }

// The listeners sit on the canvas's parent in the capture phase, so they run
// before the Edit and Connect handlers on the canvas itself and can keep a pan
// gesture from reaching them.
function onPointerDown(ev) {
  if (ev.target !== ovCv || pan || !panStarts(ev)) return;
  ev.stopPropagation();
  const [bx, by] = evPx(ev);
  pan = { pointerId: ev.pointerId, view: { ...curView() }, bx, by, right: ev.button === 2 };
  if (pan.right) { swallowMenu = true; clearTimeout(swallowTimer); }
  try { ovCv.setPointerCapture(ev.pointerId); } catch (e) {}
  setGrabCursor("grabbing");
}

function onPointerMove(ev) {
  if (!pan || ev.pointerId !== pan.pointerId) return;
  ev.stopPropagation();
  const [px, py] = evPx(ev);
  setView(panByPx(pan.view, pan.bx - px, pan.by - py));
}

function onPointerUp(ev) {
  if (!pan || ev.pointerId !== pan.pointerId) return;
  ev.stopPropagation();
  // disarm soon after release, so a later plain right-click still gets its menu
  if (pan.right) swallowTimer = setTimeout(() => { swallowMenu = false; }, 400);
  pan = null;
  try { ovCv.releasePointerCapture(ev.pointerId); } catch (e) {}
  if (spaceHeld && overCanvas) ovCv.style.cursor = "grab";
  else restoreCursor();
}

function onKeyDown(ev) {
  if (ev.key !== " " || ev.ctrlKey || ev.altKey || ev.metaKey || typingInField(ev.target)) return;
  spaceHeld = true;
  // over the canvas Space is ours: no page scroll, no activation of a focused button
  if (overCanvas) {
    ev.preventDefault();
    if (!pan) setGrabCursor("grab");
  }
}

function releaseSpace() {
  spaceHeld = false;
  if (!pan) restoreCursor();
}

export function initViewNav(deps) {
  ({ ovCv, redraw, resetToScan, applyToolCursor } = deps);
  if (deps.xyRangeUm) xyRangeUm = deps.xyRangeUm;
  const host = ovCv.parentElement;
  host.addEventListener("wheel", onWheel, { capture: true, passive: false });
  host.addEventListener("pointerdown", onPointerDown, true);
  host.addEventListener("pointermove", onPointerMove, true);
  host.addEventListener("pointerup", onPointerUp, true);
  host.addEventListener("pointercancel", onPointerUp, true);
  ovCv.addEventListener("contextmenu", (ev) => ev.preventDefault());
  window.addEventListener("contextmenu", (ev) => {
    if (!swallowMenu) return;
    swallowMenu = false;
    clearTimeout(swallowTimer);
    ev.preventDefault();
  }, true);
  ovCv.addEventListener("dblclick", (ev) => {
    if (state.currentTab === "scan") { ev.preventDefault(); resetToScan(); }
  });
  ovCv.addEventListener("pointerenter", () => {
    overCanvas = true;
    if (spaceHeld && !pan) setGrabCursor("grab");
  });
  ovCv.addEventListener("pointerleave", () => {
    overCanvas = false;
    if (!pan) restoreCursor();
  });
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", (ev) => { if (ev.key === " ") releaseSpace(); });
  window.addEventListener("blur", releaseSpace);
}
