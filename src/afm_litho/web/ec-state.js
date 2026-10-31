// The edit cockpit's shared mutable model + a tiny event bus (Layer 1, DOM-free;
// imports under node, where the state defaults and the bus are unit-tested).
//
// Why a single `state` object rather than a `let` per global: ES-module imports
// are read-only live bindings, so an importer can never reassign an imported
// `selected`/`view`. The tab controllers (ec-tab-edit and the rest) share and
// REASSIGN this core model, so it must live behind a stable binding whose
// PROPERTIES are writable from any module. `const state` is that binding; every
// field below is read and written as `state.X` across the cockpit.
//
// Ownership (docs/COCKPIT_MODULES.md §8): once a value lives here it is reachable
// ONLY via `state.X` -- never duplicated in another module, never bridged through
// `window.*`. The per-tab transient drafts (pen/rubberRect/node-drag,
// port-drag, ...) are NOT here: each is touched by exactly one tab and travels
// with that tab's controller when it is carved out.

// The shared model. Defaults are the exact values the monolith declared inline.
export const state = {
  // -- object list + selection -------------------------------------------
  // MATCHES the real backend design-object shape so a /api/design/svg import can
  // populate this same array unchanged. Each object:
  //   { name, kind, voltage, voltage_end, speed_um_s, points_um }
  // kind is a real backend kind (path -> "path", shape -> "rectangle",
  // electrode -> "funnel"; displayKind maps it to the UI label).
  objects: [
    {
      name: "wire-1", kind: "path", voltage: 8, voltage_end: null, speed_um_s: 2.0,
      // a zig-zag wire
      points_um: [[1, 1], [3, 4], [5, 1.5], [7, 4.5], [9, 2]],
    },
    {
      name: "rect-1", kind: "shape", voltage: -6, voltage_end: null, speed_um_s: 1.5,
      // closed rectangle (5 corner points, last == first)
      points_um: [[2, 7], [6, 7], [6, 10], [2, 10], [2, 7]],
      // the 4-node closed rectangle bezier a drawn/imported rect carries, so the
      // list labels it "rectangle" (not "region") and the node tool refuses it.
      bezier: { closed: true, nodes: [
        { x: 2, y: 7, h_in: null, h_out: null, kind: "cusp" },
        { x: 6, y: 7, h_in: null, h_out: null, kind: "cusp" },
        { x: 6, y: 10, h_in: null, h_out: null, kind: "cusp" },
        { x: 2, y: 10, h_in: null, h_out: null, kind: "cusp" },
      ] },
    },
    {
      name: "funnel-1", kind: "electrode", voltage: 4, voltage_end: null, speed_um_s: 3.0,
      // tapered outline: wide mouth narrowing to a throat, back up, closed
      points_um: [[8, 7], [11, 7], [10, 9.2], [10, 11], [9, 11], [9, 9.2], [8, 7]],
    },
    {
      name: "wire-2", kind: "path", voltage: -3, voltage_end: null, speed_um_s: 2.5,
      points_um: [[1, 5], [4, 5.5], [6.5, 5], [9.5, 5.6]],
    },
    {
      name: "rect-2", kind: "shape", voltage: 2, voltage_end: null, speed_um_s: 1.0,
      points_um: [[11.5, 1], [14, 1], [14, 3.2], [11.5, 3.2], [11.5, 1]],
      bezier: { closed: true, nodes: [
        { x: 11.5, y: 1, h_in: null, h_out: null, kind: "cusp" },
        { x: 14, y: 1, h_in: null, h_out: null, kind: "cusp" },
        { x: 14, y: 3.2, h_in: null, h_out: null, kind: "cusp" },
        { x: 11.5, y: 3.2, h_in: null, h_out: null, kind: "cusp" },
      ] },
    },
  ],
  selected: -1,               // index into objects, or -1 for no selection

  // -- view + scan frame -------------------------------------------------
  // SCAN_FRAME (microns): the field "Zoom to scan" fits. Becomes the loaded
  // scan's µm bounds once a real frame arrives (hence mutable).
  SCAN_FRAME: { x0: 0, y0: 0, w: 15, h: 12 },
  view: null,                 // {x0,y0,w,h} in microns
  scanFrame: null,            // the loaded scan frame (wire dict) or null = placeholder
  scanImg: null,              // offscreen grayscale canvas (pixels x rows) or null
  lastScanCfg: null,          // last scan/start body -- the live frame reads its x/y offset
  scanView: "both",           // the render's scan view: live | last | both
  lastFrame: null,            // the page's copy of the previous frame (same shape as scanFrame)
  zManual: null,              // grey range set on the z bar {lo, hi, chan}, or null = auto

  // -- design provenance -------------------------------------------------
  designImported: false,      // becomes true once an SVG import / add sets hub.pattern
  sourceLabel: "mock",        // the current design source (mirrors the #src chip text)

  // -- active tab / tool -------------------------------------------------
  currentTab: "scan",
  activeTool: "select",       // select | node | rect | pen

  // -- electrode ports (Connect tab) -------------------------------------
  // Ports are {id,label,position_um:[x,y],direction_deg,width_um} keyed to the
  // current sample; selectedPort holds the selected port's stable id.
  ports: [],
  selectedPort: null,         // the SELECTED port's id (stable), or null

  // -- live litho state (driven by the litho_* ws messages during a write) --
  //   writing : a write is in flight; name : object being written; frac : 0..1;
  //   tip : {x,y} live tip position (µm) or null; live : growing covered path;
  //   written : completed paths; tipTrail : the last ~6 s of the Show-Tip stream.
  //   writes : the written trace grouped per object per run, and showWritten the
  //   overlay's on/off (ec-written.js).
  litho: { writing: false, name: null, frac: 0, tip: null, live: [], written: [], tipTrail: [],
           writes: [], showWritten: true },

  // -- drift track/compensate state (Connect Phase 2) --------------------
  driftState: { enabled: false, has_reference: false, total: [0, 0] },
};

// ---- tiny event bus ------------------------------------------------------
// subscribe(topic, fn) registers a listener; emit(topic, ...args) fires them.
// Multiple subscribers per topic (fan-out); subscribe returns an unsubscribe fn.
// Used only where a state change already triggered a redraw/relayout, so the
// bus never changes WHEN work happens -- it only decouples WHO asked for it.
const busHandlers = new Map();   // topic -> Set<fn>

export function subscribe(topic, fn) {
  let set = busHandlers.get(topic);
  if (!set) { set = new Set(); busHandlers.set(topic, set); }
  set.add(fn);
  return () => set.delete(fn);
}

export function emit(topic, ...args) {
  const set = busHandlers.get(topic);
  if (set) for (const fn of [...set]) fn(...args);
}
