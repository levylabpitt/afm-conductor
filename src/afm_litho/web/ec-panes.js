// Two-pane split of <main>: the render pane (left) and the tab pane (right) share the
// width through #paneDivider. The right pane's width is what is set and stored, so the
// render pane takes whatever the window gains or loses.
//
// Import-safe: no DOM access until initPanes() or initSideScroll() runs.

// One stored width per window class, so a split chosen at full screen does not undo
// the equal split at half screen, and the other way round. LEGACY_KEY held one width
// for both classes, saved at either; initPanes drops it so each class starts from
// its default.
export const STORAGE_KEY = "ec.rightPaneWidthPx.full";
export const STORAGE_KEY_HALF = "ec.rightPaneWidthPx.half";
const LEGACY_KEY = "ec.rightPaneWidthPx";
export const MIN_RENDER_PX = 360;
// <main> this wide or narrower is a half-screen window (948 px is half of the operator's
// 1896 px screen); there the panes split the width equally until the operator drags.
export const HALF_WINDOW_PX = 1200;
// The right pane's width on a wider window until the operator drags: Edit's three
// columns and Write's two fit with object names whole.
export const DEFAULT_RIGHT_PX = 600;
// The widest tab content (Write and Connect) needs 450 px; narrower and it runs past
// the window edge.
export const MIN_RIGHT_PX = 460;
const KEY_STEP_PX = 24;

// The right pane's width for a wanted width, inside `total` px of <main> that also
// holds a `divider` px divider. The right pane's minimum wins when both cannot fit.
export function clampRightWidth(want, total, divider = 0) {
  const max = total - divider - MIN_RENDER_PX;
  return Math.round(Math.max(MIN_RIGHT_PX, Math.min(want, max)));
}

export const isHalfWindow = (total) => total <= HALF_WINDOW_PX;

// The right pane's width when the operator has not chosen one for this window class.
export function defaultRightWidth(total, divider = 0) {
  return isHalfWindow(total) ? (total - divider) / 2 : DEFAULT_RIGHT_PX;
}

// Storage can be blocked (private window, site data off) and then throws; the page
// works the same without it, only the position is not remembered.
function readStored(key) {
  try {
    const raw = window.localStorage.getItem(key);
    const v = Number(raw);
    return raw !== null && Number.isFinite(v) && v > 0 ? v : null;
  } catch { return null; }
}
function writeStored(key, px) {
  try { window.localStorage.setItem(key, String(Math.round(px))); } catch { /* blocked */ }
}

export function initPanes() {
  const divider = document.getElementById("paneDivider");
  const right = document.querySelector(".rightpanel");
  if (!divider || !right) return;
  const main = divider.parentElement;
  try { window.localStorage.removeItem(LEGACY_KEY); } catch { /* blocked */ }

  const current = () => right.getBoundingClientRect().width;
  const key = () => (isHalfWindow(main.clientWidth) ? STORAGE_KEY_HALF : STORAGE_KEY);
  // The operator's chosen width per window class (null: the default); the shown width
  // is this clamped to the window, so shrinking and regrowing the window comes back to
  // the same split.
  const wanted = {
    [STORAGE_KEY]: readStored(STORAGE_KEY), [STORAGE_KEY_HALF]: readStored(STORAGE_KEY_HALF),
  };
  // aria value: the render pane's width, i.e. where the divider sits from the left.
  const apply = () => {
    const total = main.clientWidth, dw = divider.offsetWidth;
    const w = clampRightWidth(wanted[key()] ?? defaultRightWidth(total, dw), total, dw);
    right.style.width = `${w}px`;
    divider.setAttribute("aria-valuemin", String(MIN_RENDER_PX));
    divider.setAttribute("aria-valuemax", String(Math.max(MIN_RENDER_PX, total - dw - MIN_RIGHT_PX)));
    divider.setAttribute("aria-valuenow", String(Math.max(0, Math.round(total - dw - w))));
  };
  const setWidth = (w) => { const k = key(); wanted[k] = w; apply(); wanted[k] = current(); };
  const store = () => writeStored(key(), current());
  apply();

  let drag = null;
  divider.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0) return;
    ev.preventDefault();
    divider.setPointerCapture(ev.pointerId);
    drag = { x: ev.clientX, w: current() };
    document.body.classList.add("pane-dragging");
  });
  divider.addEventListener("pointermove", (ev) => {
    if (drag) setWidth(drag.w - (ev.clientX - drag.x));
  });
  const end = (ev) => {
    if (!drag) return;
    drag = null;
    if (divider.hasPointerCapture(ev.pointerId)) divider.releasePointerCapture(ev.pointerId);
    document.body.classList.remove("pane-dragging");
    store();
  };
  divider.addEventListener("pointerup", end);
  divider.addEventListener("pointercancel", end);

  divider.addEventListener("keydown", (ev) => {
    // Home puts the divider at the far left (widest right pane), End at the far right.
    const step = { ArrowLeft: KEY_STEP_PX, ArrowRight: -KEY_STEP_PX,
                   Home: Infinity, End: -Infinity }[ev.key];
    if (step === undefined) return;
    ev.preventDefault();
    setWidth(Number.isFinite(step) ? current() + step
             : step > 0 ? main.clientWidth : 0);
    store();
  });

  // A narrower window must not squeeze the render pane under its minimum.
  window.addEventListener("resize", apply);
}

// A one-line strip that scrolls sideways instead of wrapping, with its scroll bar
// hidden: a plain wheel over it, or a drag that starts on its empty space, scrolls it.
// Ctrl+wheel is left alone (it is the browser's page zoom off the canvas).
export function initSideScroll(el) {
  if (!el) return;
  el.addEventListener("wheel", (ev) => {
    if (ev.ctrlKey || el.scrollWidth <= el.clientWidth) return;
    const d = Math.abs(ev.deltaX) > Math.abs(ev.deltaY) ? ev.deltaX : ev.deltaY;
    if (!d) return;
    ev.preventDefault();
    el.scrollLeft += d;
  }, { passive: false });
  // Empty space is the strip itself or a group's caption span, never a control or a
  // label (a press on a label would toggle its checkbox).
  const empty = (t) => t === el || (t.parentElement === el && t.tagName === "SPAN");
  let drag = null;
  el.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0 || !empty(ev.target)) return;
    ev.preventDefault();
    el.setPointerCapture(ev.pointerId);
    drag = { x: ev.clientX, s: el.scrollLeft };
  });
  el.addEventListener("pointermove", (ev) => {
    if (drag) el.scrollLeft = drag.s - (ev.clientX - drag.x);
  });
  const end = (ev) => {
    drag = null;
    if (el.hasPointerCapture(ev.pointerId)) el.releasePointerCapture(ev.pointerId);
  };
  el.addEventListener("pointerup", end);
  el.addEventListener("pointercancel", end);

  // The parent shows a fade and a › at the right edge only while part of the strip
  // is scrolled out of view there (CSS: .more).
  const wrap = el.parentElement;
  const sync = () => {
    if (wrap) wrap.classList.toggle("more", el.scrollLeft + el.clientWidth < el.scrollWidth - 1);
  };
  el.addEventListener("scroll", sync, { passive: true });
  window.addEventListener("resize", sync);
  if (typeof ResizeObserver === "function") new ResizeObserver(sync).observe(el);
  // the vendored fonts change the strip's content width once they load
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(sync);
  sync();
}
