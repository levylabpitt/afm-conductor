// Edit-tab controller (Layer 3, docs/COCKPIT_MODULES.md §3) -- the first,
// Inkscape-flavoured tool set on the overlay canvas. All authoring is in MICRONS
// (evToUm); each tool no-ops unless the Edit tab is active and it is the picked
// tool, so Scan/Write/Connect are never affected.
//   select : click an object to select, drag it to MOVE (op:transform)
//   node   : drag a selected path's vertices     (POST /api/design/bezier|points)
//   rect   : click-drag a rubber rectangle -> a NEW shape (POST /api/design/add)
//   pen    : click to place nodes (drag to pull symmetric handles) -> a NEW
//            path; Enter/double-click finishes, Escape finalizes; double-clicking
//            the START node CLOSES the outline (a funnel / region fill).
//
// This module OWNS the per-tab transient edit drafts (the node-tool base Bézier,
// the select-tool transform drag, the move/node drags, the rubber rectangle and
// the pen draft) -- each is touched by exactly one tab, so it lives here rather
// than in the shared ec-state model (docs/COCKPIT_MODULES.md §8). The shared
// model is read/written through the imported `state` object.
//
// Layer discipline: imports Layer 1 only (ec-state / ec-bezier / ec-geom /
// ec-api / ec-log / ec-theme). The DOM/redraw helpers it still needs (evToUm, pxPerUm,
// drawOverlay, the render/adopt/refresh calls, applyToolCursor, the overlay 2-D
// context) live in ec-main / not-yet-extracted Layer 2, so ec-main INJECTS them
// via initEditTab() rather than this module importing upward. ec-main keeps the
// overlay-canvas + keydown listeners and delegates to the exported handlers, and
// registers drawEditLayer with the overlay painter registry.

import { state } from "./ec-state.js";
import { logLine, num } from "./ec-log.js";
import {
  getDesignBezier, postDesignBezier, postDesignOp, postDesignAdd, postDesignPoints,
} from "./ec-api.js";
import {
  bezMoveNode, bezSetHandle, bezSegments, cubicAt, bezFlatten,
  normalizeBez, synthBezier, bezCurveDrag,
  penToPolyline, penToBezier, rectBezier,
  setNodeType, deleteNode, insertNode, joinNodes, splitNode, isRectBezier,
} from "./ec-bezier.js";
import {
  pointInPoly, distToSeg, distToPolyline, objBounds, portSnapHit, skewAbout, boxFieldOk,
} from "./ec-geom.js";
import { cssVar, cssVarA } from "./ec-theme.js";
import { bezLength, objectLength } from "./ec-measure.js";

// ---- injected shared helpers (owned by ec-main / not-yet-extracted Layer 2) ----
// Bound once by initEditTab(); same names as the ec-main originals so the moved
// function bodies reference them unchanged.
let evToUm, pxPerUm, drawOverlay, renderList, renderSettings,
    adoptDesign, refreshToolpath, applyToolCursor, ovCv, ovCx, reAdopt, clipboard,
    xyRangeUm;

const $ = (id) => document.getElementById(id);

// =========================================================================
//  MODULE-OWNED TRANSIENT EDIT STATE
// =========================================================================
// The object whose Bézier is currently being node-edited, and that Bézier
// (the collapse-to-base editable curve). editBezier is null until a path/shape
// is picked in the node tool; it is reloaded when the selection changes.
let editBezier = null, editBezierIndex = -1, editBezierModulated = false;
// The node the toolbar acts on (Type/Delete/Split): an index into
// editBezier.nodes, or -1 for none. Set when a node/handle is clicked; cleared
// when the edited object reloads. Purely a UI selection -- it never changes geometry.
let selectedNodeIndex = -1;
// select-tool transform-handle drag (Inkscape select-tool): a selected object
// shows an axis-aligned bounding box with 8 scale handles + a rotate handle.
const OPP = { nw: "se", ne: "sw", se: "nw", sw: "ne", n: "s", s: "n", e: "w", w: "e" };
let xformDrag = null;   // { type:'scale'|'rotate', ... , sx, sy, rotDeg }
let moveDrag = null;    // { orig:[[x,y]...], startUm:[x,y], moved:bool }
let nodeDrag = null;    // { type:'node'|'handle'|'curve', ... }
let rubberRect = null;  // { x0,y0,x1,y1 } in microns while dragging
// PEN draft: anchor nodes with optional symmetric Bezier handles. Click = cusp
// node; click-drag = smooth node (hOut = drag point, hIn = its mirror).
let pen = { nodes: [], cursor: null, dragging: false };
// SNAP-TO-BIND (EDIT_TOOLS §4): while a node-tool drag holds an OPEN path's END
// node near an electrode, this is the matched target { portId, node:"start"|"end",
// pos:[x,y] }; null when the drag is not snapping. Set in editPointerMove, drawn
// as a highlight in drawEditLayer, consumed (recorded as a binding) in editPointerUp.
let snapActive = null;

// =========================================================================
//  HIT-TESTING + NAMING
// =========================================================================
// Topmost object at (x,y) in microns, or -1. Later-drawn objects (higher index)
// win. Filled kinds hit-test by area OR edge; paths by distance to the polyline.
function hitTest(x, y) {
  const tolUm = 6 / pxPerUm();
  for (let i = state.objects.length - 1; i >= 0; i--) {
    const pts = state.objects[i].points_um;
    if (!pts || pts.length < 2) continue;
    const filled = state.objects[i].kind === "shape" || state.objects[i].kind === "electrode";
    if (filled) {
      if (pointInPoly(pts, x, y) || distToPolyline(pts, x, y) <= tolUm) return i;
    } else if (distToPolyline(pts, x, y) <= tolUm) return i;
  }
  return -1;
}
// A unique name for a NEW local (mock) object; the backend does its own naming.
function uniqueName(base) {
  const have = new Set(state.objects.map((o) => o.name));
  let n = base, k = 1;
  while (have.has(n)) { k++; n = `${base}-${k}`; }
  return n;
}

// =========================================================================
//  EDITABLE BÉZIER -- the node tool edits the SAME control-point model the
//  backend stores and writes (ec-bezier.js is the JS port of litho/bezier.py).
// =========================================================================
// A recognized rectangle (the as_rect signature carried on the object's bezier).
// Rectangles are NOT node-editable this round (docs/EDIT_TOOLS.md §3).
function isRectObject(o) {
  return !!(o && o.bezier && isRectBezier(o.bezier));
}
function isNodeEditable(o) {
  return !!o && (o.kind === "path" || o.kind === "shape" || o.kind === "electrode")
    && !isRectObject(o);
}
// Load object i's editable base Bézier. Imported designs fetch the real control
// points (collapse-to-base done server-side); mock objects synthesize cusp nodes.
async function loadEditBezier(i) {
  editBezierIndex = i; editBezier = null; editBezierModulated = false;
  selectedNodeIndex = -1;   // a different object: the toolbar has no node yet
  const got = await baseBezierOf(i);
  if (!got) return;
  editBezier = got.bez; editBezierModulated = got.modulated;
}
// Object i's editable base Bézier as { bez, modulated }, or null for no object.
async function baseBezierOf(i) {
  const o = state.objects[i];
  if (!o) return null;
  if (state.designImported) {
    const j = (await getDesignBezier(i)).data;   // request() logs the traffic
    if (j && j.bezier) return { bez: normalizeBez(j.bezier), modulated: !!j.modulated };
    logLine(`bezier load: ${j && j.error ? j.error : "no geometry"}`, "warn");
  }
  // mock / fallback: a pen path carries its own control points, so edit THOSE
  // (the drawn handles) rather than re-synthesizing cusp nodes from the polyline.
  if (o.bezier && o.bezier.nodes && o.bezier.nodes.length >= 2 && !o.profile)
    return { bez: normalizeBez(o.bezier), modulated: false };
  const base = (o.profile && o.profile.axis_um) ? o.profile.axis_um : o.points_um;
  return { bez: synthBezier(base, o.kind !== "path"), modulated: !!(o.profile) };
}
// Persist an edited Bézier. Imported: POST /api/design/bezier (stores the
// control points, re-flattens, re-applies any modulation on top of the new
// base); mock: keep the client-flattened polyline locally. `binding` carries the
// snap-to-bind endpoint->port reference (EDIT_TOOLS §4): undefined leaves it
// untouched (an ordinary drag), an object sets it, null clears it -- sent only
// when present so an ordinary node drag serializes exactly as before.
async function persistBezier(i, bez, binding) {
  if (!state.designImported) {
    // mock: apply the binding locally so it round-trips in-memory like a real one
    if (binding !== undefined && state.objects[i]) state.objects[i].end_binding = binding;
    // mock: mirror the server's kind-follows-closedness rule so a join FILLS the
    // object in-memory and a split traces it again (the backend has as_funnel to
    // pick electrode; the mock only distinguishes filled region vs. path, and
    // leaves an existing electrode/shape/path unchanged when closedness matches).
    const o = state.objects[i];
    if (o) {
      if (bez.closed && o.kind === "path") o.kind = "shape";
      else if (!bez.closed && (o.kind === "shape" || o.kind === "electrode")) o.kind = "path";
    }
    drawOverlay(); return true;
  }
  const name = state.objects[i] && state.objects[i].name;
  const body = { index: i, bezier: bez };
  if (binding !== undefined) body.end_binding = binding;   // obj sets, null clears
  const res = await postDesignBezier(body);
  // keep the local base Bézier as authoritative for the reselected object
  return reAdopt(res, name, (sel) => { editBezierIndex = sel; });
}
// The end of a node edit (a drag release, or a typed node position): re-flatten
// object i's polyline from editBezier and persist it with the endpoint binding.
async function finishNodeMove(i, ni, kind, snap) {
  if (i < 0 || !state.objects[i]) return;
  state.objects[i].points_um = bezFlatten(editBezier, 0.004);
  // Resolve the endpoint->port binding for this edit (EDIT_TOOLS §4):
  //   undefined -> leave the binding as-is (an ordinary node/handle/curve drag)
  //   {node,port_id} -> a fresh snap binds this end
  //   null -> the bound end was moved off every port, so unbind it
  let binding;   // undefined by default
  const role = endpointRole(editBezier, ni);
  const prev = state.objects[i].end_binding || null;
  if (snap) {
    binding = { node: snap.node, port_id: snap.portId };
    logLine(`${state.objects[i].name}: ${snap.node} end bound to ${snap.portId}`, "info");
  } else if (role && prev && prev.node === role) {
    binding = null;   // moved the bound end away -> clear the reference
    logLine(`${state.objects[i].name}: ${role} end unbound`, "info");
  } else {
    logLine(`${state.objects[i].name}: ${kind} edited`, "info");
  }
  await persistBezier(i, editBezier, binding);
}
// The x/y fields move the selected node exactly as dragging it there would:
// the same snap test, re-flatten and persist. A field still showing the
// position it was given (only rounded) is not an edit.
async function applyNodeField(input) {
  const ni = selectedNodeIndex;
  if (!editBezier || ni < 0 || !editBezier.nodes[ni] || state.selected !== editBezierIndex) return;
  if (input.value === input.dataset.shown) return;
  const n = editBezier.nodes[ni];
  const x = $("nodeX").value === $("nodeX").dataset.shown ? n.x : Number($("nodeX").value);
  const y = $("nodeY").value === $("nodeY").dataset.shown ? n.y : Number($("nodeY").value);
  if ($("nodeX").value === "" || $("nodeY").value === "" || !isFinite(x) || !isFinite(y)) {
    // put the node's value back at once, so the following change event is no edit
    const shown = (input.id === "nodeX" ? n.x : n.y).toFixed(3);
    input.value = shown; input.dataset.shown = shown;
    return;
  }
  // Enter and the change event both arrive for one edit: the second sees no edit
  $("nodeX").dataset.shown = $("nodeX").value; $("nodeY").dataset.shown = $("nodeY").value;
  const snap = snapCandidate(editBezier, ni, x, y);
  if (snap) bezMoveNode(n, snap.pos[0], snap.pos[1]);
  else bezMoveNode(n, x, y);
  drawOverlay();
  await finishNodeMove(state.selected, ni, "node", snap);
}
// Hit-test the editable Bézier in µm: a handle circle, then a node square, then
// a segment (for curve-drag). Returns a drag descriptor or null.
function hitEditBezier(x, y) {
  const bez = editBezier; if (!bez) return null;
  const hTol = 8 / pxPerUm();
  for (let i = 0; i < bez.nodes.length; i++) {
    const n = bez.nodes[i];
    if (n.h_out && Math.hypot(n.h_out[0] - x, n.h_out[1] - y) <= hTol)
      return { type: "handle", ni: i, which: "out" };
    if (n.h_in && Math.hypot(n.h_in[0] - x, n.h_in[1] - y) <= hTol)
      return { type: "handle", ni: i, which: "in" };
  }
  for (let i = 0; i < bez.nodes.length; i++) {
    const n = bez.nodes[i];
    if (Math.hypot(n.x - x, n.y - y) <= hTol) return { type: "node", ni: i };
  }
  const seg = nearestBezSegment(bez, x, y);
  if (seg && seg.dist <= 6 / pxPerUm())
    return { type: "curve", ni0: seg.ni0, ni1: seg.ni1, t: seg.t };
  return null;
}
function nearestBezSegment(bez, x, y) {
  let best = null;
  for (const [p0, c1, c2, p3, i0, i1] of bezSegments(bez)) {
    const S = 24;
    let pv = cubicAt(p0, c1, c2, p3, 0);
    for (let s = 1; s <= S; s++) {
      const t = s / S, cv = cubicAt(p0, c1, c2, p3, t);
      const d = distToSeg(x, y, pv[0], pv[1], cv[0], cv[1]);
      if (!best || d < best.dist)
        best = { dist: d, ni0: i0, ni1: i1, t: Math.min(0.92, Math.max(0.08, t - 0.5 / S)) };
      pv = cv;
    }
  }
  return best;
}

// =========================================================================
//  NODE TOOLBAR -- the Type (cusp/smooth/symmetric) + Modify (insert/delete/
//  join/split) buttons (docs/EDIT_TOOLS.md §3). Each op mutates the module-owned
//  editBezier via a pure ec-bezier function, then re-flattens the object's drawn
//  polyline and persists through the SAME persistBezier path a node drag uses, so
//  a toolbar edit and a drag edit are indistinguishable to the backend.
// =========================================================================
// Re-flatten the edited base to the object's polyline and persist it (imported:
// POST /api/design/bezier; mock: local redraw). Shared tail of every node op.
async function commitBezEdit(say) {
  const i = state.selected;
  if (i < 0 || !state.objects[i] || !editBezier) return;
  state.objects[i].points_um = bezFlatten(editBezier, 0.004);
  if (say) logLine(`${state.objects[i].name}: ${say}`, "info");
  syncNodeToolbar();
  await persistBezier(i, editBezier);
}
// Type button: set the selected node's continuity kind (handles re-derive).
async function applyNodeType(kind) {
  if (!editBezier || selectedNodeIndex < 0 || !editBezier.nodes[selectedNodeIndex]) return;
  setNodeType(editBezier.nodes[selectedNodeIndex], kind);
  await commitBezEdit(`node ${selectedNodeIndex} -> ${kind}`);
}
// Delete button: dumb-delete the selected node (neighbours reconnect).
async function deleteSelectedNode() {
  if (!editBezier || selectedNodeIndex < 0) { logLine("delete node: select a node first", "warn"); return; }
  if (editBezier.nodes.length <= 2) { logLine("delete node: a path needs at least two nodes", "warn"); return; }
  deleteNode(editBezier, selectedNodeIndex);
  const was = selectedNodeIndex;
  selectedNodeIndex = Math.min(was, editBezier.nodes.length - 1);
  await commitBezEdit(`deleted node ${was}`);
}
// Join button: close an OPEN path by merging its two endpoints into one node.
async function joinPathEnds() {
  if (!editBezier) return;
  if (editBezier.closed) { logLine("join: the path is already closed", "dim"); return; }
  if (editBezier.nodes.length < 2) { logLine("join: need two endpoints", "warn"); return; }
  joinNodes(editBezier, 0, editBezier.nodes.length - 1);
  selectedNodeIndex = 0;
  await commitBezEdit("joined the two path ends");
}
// Split button: open a CLOSED path at the selected node (inverse of join).
async function splitAtSelectedNode() {
  if (!editBezier) return;
  if (!editBezier.closed) { logLine("split: the path is already open", "dim"); return; }
  if (selectedNodeIndex < 0) { logLine("split: select a node to break at", "warn"); return; }
  splitNode(editBezier, selectedNodeIndex);
  selectedNodeIndex = 0;                  // the split node becomes the new start
  await commitBezEdit("split the path open");
}
// Insert: double-click a segment -> hit-test to (segment, t) -> de Casteljau split.
async function insertNodeAtDblClick(ev) {
  if (!editBezier) { logLine("insert: select a path or shape first", "dim"); return; }
  const [x, y] = evToUm(ev);
  const seg = nearestBezSegment(editBezier, x, y);
  if (!seg || seg.dist > 8 / pxPerUm()) { logLine("insert: double-click nearer a segment", "dim"); return; }
  ev.preventDefault();
  selectedNodeIndex = insertNode(editBezier, seg.ni0, seg.t);
  await commitBezEdit(`inserted a node on segment ${seg.ni0}`);
}
const TOOL_NAMES = { select: "select", node: "node", pen: "pen", rect: "rectangle" };
// Reflect the current node-tool state into the toolbar: shown only when the node
// tool is active; Type buttons enabled + the current kind highlighted only with a
// node selected; Delete/Join/Split enabled per what the current path allows.
function syncNodeToolbar() {
  const bar = $("nodeToolbar");
  if (!bar) return;
  const active = state.currentTab === "edit" && state.activeTool === "node";
  bar.style.display = active ? "" : "none";
  // the options column is headed by the active tool; only Node has options
  const hdr = $("toolOptsHdr"), none = $("toolOptsNone");
  if (hdr) hdr.textContent = `${TOOL_NAMES[state.activeTool] || state.activeTool} options`;
  const hasOpts = state.activeTool === "node" || state.activeTool === "select";
  if (none) none.style.display = hasOpts ? "none" : "";
  const selOpts = $("selectOpts");
  if (selOpts)
    selOpts.style.display = state.currentTab === "edit" && state.activeTool === "select" ? "" : "none";
  syncEditReadouts();
  if (!active) return;
  const node = (editBezier && selectedNodeIndex >= 0) ? editBezier.nodes[selectedNodeIndex] : null;
  const kind = node ? node.kind : null;
  for (const b of bar.querySelectorAll("button.ntype")) {
    b.disabled = !node;
    b.classList.toggle("active", !!node && b.dataset.kind === kind);
  }
  const nNodes = editBezier ? editBezier.nodes.length : 0;
  const setEnabled = (id, on) => { const el = $(id); if (el) el.disabled = !on; };
  setEnabled("nInsert", !!editBezier);
  setEnabled("nDelete", !!node && nNodes > 2);
  setEnabled("nJoin", !!editBezier && !editBezier.closed && nNodes >= 2);
  setEnabled("nSplit", !!editBezier && editBezier.closed && selectedNodeIndex >= 0);
  const hint = $("nodeHint");
  if (hint) {
    hint.textContent = editBezier
      ? (node ? `node ${selectedNodeIndex} (${kind}) selected`
              : "click a node to select it; double-click a segment to insert")
      : "select a path or shape to edit its nodes";
  }
}

// The options' readouts (lengths, node count). drawEditLayer calls this on every
// repaint, so a drag, an adopt or an undo all show at once.
function syncEditReadouts() {
  const set = (id, text) => { const el = $(id); if (el) el.textContent = text; };
  const um = (L) => `${L.toFixed(3)} µm`;
  const o = state.selected >= 0 ? state.objects[state.selected] : null;
  set("selLength", o ? um(objectLength(o)) : "—");
  if (editBezier && editBezierIndex === state.selected && o) {
    // a modulated wire's editBezier is its base; the drawn wire is the modulation
    set("nodeLength", um(editBezierModulated ? objectLength(o) : bezLength(editBezier)));
    set("nodeCount", String(editBezier.nodes.length));
    set("nodeClosed", editBezier.closed ? "yes" : "no");
  } else {
    set("nodeLength", o ? um(objectLength(o)) : "—");
    set("nodeCount", "—"); set("nodeClosed", "—");
  }
  const b = o ? objBounds(o) : null;
  const box = b ? [b.minx, b.miny, b.maxx - b.minx, b.maxy - b.miny] : null;
  SEL_BOX.forEach((id, k) => showField(id, box ? box[k].toFixed(3) : ""));
  const noSkew = skewRefusal(o);
  for (const id of SEL_AMOUNT) {
    const el = $(id);
    if (!el) continue;
    const refuse = id !== "selRot" && !!noSkew;
    el.disabled = !b || refuse;
    if (!el.dataset.title) el.dataset.title = el.title;
    el.title = refuse ? noSkew : el.dataset.title;
  }
  const node = editBezier && editBezierIndex === state.selected && selectedNodeIndex >= 0
    ? editBezier.nodes[selectedNodeIndex] : null;
  showField("nodeX", node ? node.x.toFixed(3) : "");
  showField("nodeY", node ? node.y.toFixed(3) : "");
}
// A position field shows `shown` ("" disables it); dataset.shown remembers it so
// a field left as shown (only rounded) is not taken for an edit.
function showField(id, shown) {
  const el = $(id);
  if (!el) return;
  el.disabled = shown === "";
  // a field being typed in keeps what the operator typed
  if (document.activeElement === el && shown !== "") return;
  if (el.dataset.shown !== shown || el.value !== shown) { el.value = shown; el.dataset.shown = shown; }
}

// =========================================================================
//  SELECT-TOOL TRANSFORM HANDLES (Inkscape select-tool)
// =========================================================================
function handlePoints(b) {
  const cx = (b.minx + b.maxx) / 2, cy = (b.miny + b.maxy) / 2;
  const H = {
    nw: [b.minx, b.maxy], n: [cx, b.maxy], ne: [b.maxx, b.maxy],
    e: [b.maxx, cy], se: [b.maxx, b.miny], s: [cx, b.miny],
    sw: [b.minx, b.miny], w: [b.minx, cy],
  };
  H.rot = [cx, b.maxy + 18 / pxPerUm()];   // stem above the top edge (µm-up)
  return H;
}
function hitTransformHandle(x, y) {
  const o = state.objects[state.selected]; if (!o) return null;
  const b = objBounds(o); if (!b) return null;
  const H = handlePoints(b);
  const tol = 9 / pxPerUm();
  for (const k of ["rot", "nw", "n", "ne", "e", "se", "s", "sw", "w"])
    if (Math.hypot(H[k][0] - x, H[k][1] - y) <= tol) return k;
  return null;
}
function startXform(key, x, y) {
  const o = state.objects[state.selected];
  const b = objBounds(o); const H = handlePoints(b);
  const cx = (b.minx + b.maxx) / 2, cy = (b.miny + b.maxy) / 2;
  const orig = o.points_um.map((p) => [p[0], p[1]]);
  if (key === "rot")
    xformDrag = { type: "rotate", cx, cy, orig, a0: Math.atan2(y - cy, x - cx), rotDeg: 0 };
  else
    xformDrag = { type: "scale", key, anchor: H[OPP[key]], mv0: H[key], orig, sx: 1, sy: 1 };
}
function applyScale(x, y, uniform) {
  const xd = xformDrag, a = xd.anchor, mv0 = xd.mv0, key = xd.key;
  const controlsX = (key !== "n" && key !== "s");
  const controlsY = (key !== "e" && key !== "w");
  let sx = 1, sy = 1;
  if (controlsX && Math.abs(mv0[0] - a[0]) > 1e-9) sx = (x - a[0]) / (mv0[0] - a[0]);
  if (controlsY && Math.abs(mv0[1] - a[1]) > 1e-9) sy = (y - a[1]) / (mv0[1] - a[1]);
  if (uniform && controlsX && controlsY) {
    const s = Math.max(Math.abs(sx), Math.abs(sy));
    sx = (sx < 0 ? -1 : 1) * s; sy = (sy < 0 ? -1 : 1) * s;
  }
  if (Math.abs(sx) < 1e-3) sx = sx < 0 ? -1e-3 : 1e-3;
  if (Math.abs(sy) < 1e-3) sy = sy < 0 ? -1e-3 : 1e-3;
  xd.sx = sx; xd.sy = sy;
  state.objects[state.selected].points_um = scalePts(xd.orig, a, sx, sy);
}
function scalePts(pts, a, sx, sy) {
  return pts.map((p) => [a[0] + (p[0] - a[0]) * sx, a[1] + (p[1] - a[1]) * sy]);
}
function rotatePts(pts, cx, cy, dth) {
  const ct = Math.cos(dth), st = Math.sin(dth);
  return pts.map((p) => {
    const qx = p[0] - cx, qy = p[1] - cy;
    return [cx + qx * ct - qy * st, cy + qx * st + qy * ct];
  });
}
function applyRotate(x, y, snap) {
  const xd = xformDrag, cx = xd.cx, cy = xd.cy;
  let dth = Math.atan2(y - cy, x - cx) - xd.a0;
  if (snap) { const step = Math.PI / 12; dth = Math.round(dth / step) * step; }
  xd.rotDeg = dth * 180 / Math.PI;
  state.objects[state.selected].points_um = rotatePts(xd.orig, cx, cy, dth);
}
// Persist a scale/rotate as the NET transform via the existing op:transform
// (it moves points_um AND the bezier / axis / generator server-side).
async function persistTransform(patch, say) {
  if (!state.designImported || state.selected < 0) { drawOverlay(); return true; }
  const name = state.objects[state.selected].name;
  const res = await postDesignOp({ op: "transform", indices: [state.selected], ...patch, say });
  // geometry changed; drop the cached base Bézier so it reloads on demand
  return reAdopt(res, name, () => { editBezierIndex = -1; editBezier = null; });
}

// =========================================================================
//  SELECT-TOOL FIELDS -- typed position, size, rotation and skew. Each is the
//  edit a handle drag makes: the points change locally, then the same persist
//  (op:transform for move/scale/rotate). The service has no skew, so skew is an
//  affine on the base Bézier persisted through the node tool's path.
// =========================================================================
const SEL_BOX = ["selX", "selY", "selW", "selH"];
const SEL_AMOUNT = ["selRot", "selSkewX", "selSkewY"];
// The skew goes through the node tool's Bézier route, which carries the outer
// outline only, so holes would stay behind. The design the page reads does not
// carry holes_um, so a filled kind may have holes the page cannot see: skew is
// for open paths only, and never for an object known to have holes.
function hasHoles(o) {
  return !!o && Array.isArray(o.holes_um) && o.holes_um.length > 0;
}
const SKEW_FILLED_REASON = "skew is for open paths: a filled outline may have holes it would leave behind";
function skewRefusal(o) {
  if (!o) return null;
  if (o.kind !== "path") return SKEW_FILLED_REASON;
  if (hasHoles(o)) return "skew refused: it would leave the object's holes behind";
  return null;
}
// The value a Select field shows for the selection: the box for x/y/w/h, 0 for
// an amount still to apply.
function selFieldText(id, o) {
  if (SEL_AMOUNT.includes(id)) return "0";
  const b = o ? objBounds(o) : null;
  if (!b) return "";
  return [b.minx, b.miny, b.maxx - b.minx, b.maxy - b.miny][SEL_BOX.indexOf(id)].toFixed(3);
}
async function applySelectField(input) {
  const i = state.selected, o = state.objects[i];
  if (!o || input.value === input.dataset.shown) return;
  const b = objBounds(o);
  const v = Number(input.value);
  const id = input.id;
  // A refused value puts the real one back at once, even while the field has
  // focus, so the change event that follows Enter sees no edit and does not warn
  // a second time.
  const reset = (why) => {
    if (why) logLine(`${o.name}: ${why}`, "warn");
    const shown = selFieldText(id, o);
    input.value = shown; input.dataset.shown = shown;
  };
  if (input.value === "" || !isFinite(v) || !b) { reset(); return; }
  const field = { selX: "x", selY: "y", selW: "w", selH: "h" }[id];
  const range = xyRangeUm ? xyRangeUm() : null;
  if (field && !boxFieldOk(field, v, range)) {
    reset(field === "x" || field === "y"
      ? `${field} must be within ±${num(range || 0) || "the scanner range"} µm`
      : `${field === "w" ? "width" : "height"} must be above 0 and within the scanner range`);
    return;
  }
  input.dataset.shown = input.value;
  const name = o.name;
  const orig = o.points_um.map((p) => [p[0], p[1]]);
  // a refused edit leaves the canvas showing the geometry the service still has
  const restore = (ok) => {
    if (ok || state.objects[i] !== o) return;
    o.points_um = orig;
    reset();
    drawOverlay();
  };
  if (id === "selX" || id === "selY") {
    const dx = id === "selX" ? v - b.minx : 0, dy = id === "selY" ? v - b.miny : 0;
    o.points_um = orig.map((p) => [p[0] + dx, p[1] + dy]);
    drawOverlay();
    logLine(`${name} moved (${num(dx)}, ${num(dy)}) µm`, "info");
    restore(await persistMove(dx, dy));
  } else if (id === "selW" || id === "selH") {
    const bw = b.maxx - b.minx, bh = b.maxy - b.miny;
    const cur = id === "selW" ? bw : bh;
    if (cur < 1e-9) { reset(`${id === "selW" ? "width" : "height"} of a flat object cannot be set`); return; }
    const f = v / cur, lock = $("selLock").checked;
    const sx = id === "selW" || lock ? f : 1, sy = id === "selH" || lock ? f : 1;
    const anchor = [b.minx, b.miny];
    o.points_um = scalePts(orig, anchor, sx, sy);
    drawOverlay();
    logLine(`${name}: scaled (${num(sx)}×, ${num(sy)}×)`, "info");
    restore(await persistTransform({ sx, sy, dx_um: 0, dy_um: 0, about_um: anchor },
                                   `Scaled ${name}`));
  } else if (id === "selRot") {
    const cx = (b.minx + b.maxx) / 2, cy = (b.miny + b.maxy) / 2;
    o.points_um = rotatePts(orig, cx, cy, v * Math.PI / 180);
    drawOverlay();
    logLine(`${name}: rotated ${num(v)}°`, "info");
    restore(await persistTransform({ rot_deg: v, about_um: [cx, cy] }, `Rotated ${name}`));
  } else {
    const refused = skewRefusal(o);
    if (refused) { reset(refused); return; }
    if (Math.abs(v) >= 89) { reset("skew must be under 89°"); return; }
    const ax = id === "selSkewX" ? v : 0, ay = id === "selSkewY" ? v : 0;
    restore(await applySkew(i, b, ax, ay));
  }
  if (SEL_AMOUNT.includes(id)) { input.value = "0"; input.dataset.shown = "0"; }
}
// Skew object i about its box centre; false when the service refused it.
async function applySkew(i, b, ax, ay) {
  const o = state.objects[i];
  const c = [(b.minx + b.maxx) / 2, (b.miny + b.maxy) / 2];
  const sk = (p) => skewAbout(p, c, ax, ay);
  const got = await baseBezierOf(i);
  if (!got || state.objects[i] !== o) return true;
  const bez = got.bez;
  for (const n of bez.nodes) {
    [n.x, n.y] = sk([n.x, n.y]);
    if (n.h_in) n.h_in = sk(n.h_in);
    if (n.h_out) n.h_out = sk(n.h_out);
  }
  o.points_um = got.modulated ? o.points_um.map(sk) : bezFlatten(bez, 0.004);
  if (!state.designImported && o.bezier) o.bezier = bez;
  logLine(`${o.name}: skewed (${num(ax)}°, ${num(ay)}°)`, "info");
  const ok = await persistBezier(i, bez);
  editBezier = null; editBezierIndex = -1;   // the node tool reloads the new base
  return ok;
}

// =========================================================================
//  NEW-OBJECT PERSISTENCE
// =========================================================================
// Append a new object (rect/pen). This ALWAYS hits the backend POST
// /api/design/add so the object lands in hub.pattern (writable, compiled,
// reorderable) and gets a real toolpath. The FIRST draw in a fresh/mock
// session therefore ESTABLISHES the backend design: the placeholder demo
// objects give way to the real drawn object and designImported flips true,
// which is what wires up tip-path, chiral modulation, and per-object
// persistence (persistMove/persistProps/persistChiral) on drawn objects.
async function addObject(kind, points_um, opts = {}) {
  const voltage = opts.voltage != null ? opts.voltage : 8;
  const speed = opts.speed_um_s != null ? opts.speed_um_s : 2;
  const addBody = { kind, points_um, voltage, speed_um_s: speed };
  if (opts.bezier) addBody.bezier = opts.bezier;   // pen keeps its control points
  const res = await postDesignAdd(addBody);
  const j = res.data;
  if (!res.ok || j.error || !j.design) return;   // request() logged it
  adoptDesign(j.design, state.sourceLabel);
  state.designImported = true;            // a real backend design now exists
  state.selected = (typeof j.index === "number") ? j.index : state.objects.length - 1;
  renderList(); renderSettings();
  await refreshToolpath();          // a new object changes the compiled path
  drawOverlay();
}

// =========================================================================
//  SELECT + MOVE
// =========================================================================
// Translate the selected object's geometry in microns. For an imported design
// the final move persists via the existing op:transform (dx/dy in microns);
// mock objects keep the locally-applied translation.
async function persistMove(dxUm, dyUm) {
  if (!state.designImported || state.selected < 0) return true;
  const name = state.objects[state.selected] && state.objects[state.selected].name;
  const res = await postDesignOp({ op: "transform", indices: [state.selected],
    dx_um: dxUm, dy_um: dyUm, say: `Moved ${name}` });
  return reAdopt(res, name);
}

// =========================================================================
//  NODE editing -- persistPoints (raw polyline) for callers that replace
//  vertices wholesale (the node tool itself persists via the Bézier).
// =========================================================================
async function persistPoints(i, points_um) {
  if (!state.designImported) return;      // mock: local only, like the other edits
  const name = state.objects[i] && state.objects[i].name;
  const res = await postDesignPoints({ index: i, points_um });
  await reAdopt(res, name);
}

// =========================================================================
//  PEN -- penToPolyline / penToBezier / rectBezier live in ec-bezier.js;
//  penToPolyline takes the current pixels-per-µm (the adaptive segment count),
//  so callers pass pxPerUm().
// =========================================================================
function cancelPen() { pen = { nodes: [], cursor: null, dragging: false }; }
async function finishPen() {
  const nodes = pen.nodes.slice();
  // A double-click leaves a trailing node on top of the previous one; drop it.
  while (nodes.length >= 2) {
    const a = nodes[nodes.length - 1], b = nodes[nodes.length - 2];
    if (Math.hypot(a.x - b.x, a.y - b.y) < 0.01) nodes.pop(); else break;
  }
  cancelPen();
  if (nodes.length < 2) { logLine("pen cancelled (need 2+ nodes)", "dim"); drawOverlay(); return; }
  // Carry the Bézier so the finished path is node-editable exactly as drawn; the
  // polyline is still sent so a mock (non-imported) design has a drawable fallback.
  await addObject("path", penToPolyline(nodes, pxPerUm()), { bezier: penToBezier(nodes) });
}
// Commit the pen draft as a CLOSED outline -- reached when the finishing
// double-click lands on the START node. Sets closed=true with NO duplicate
// trailing point, and commits as a `shape`: the SAME object an imported closed
// outline becomes, so it reaches the existing fill machinery in expand() (a
// base+two-curves funnel sweeps between its sides; any other blob is a region
// fill). The server re-derives the kind (electrode for a funnel) exactly like
// SVG import, so a hand-drawn funnel === an imported one.
async function closePen() {
  const nodes = pen.nodes.slice();
  const start = nodes[0];
  const tol = start ? 8 / pxPerUm() : 0;   // the node hit-radius (hitEditBezier)
  // Drop the closing double-click's own clicks that landed on/near the start
  // node -- its two pointerdowns each pushed a node there. Keep >= 3 (the
  // minimum closed outline: a triangle / base + two sides).
  while (nodes.length > 3 &&
         Math.hypot(nodes[nodes.length - 1].x - start.x,
                    nodes[nodes.length - 1].y - start.y) <= tol) nodes.pop();
  cancelPen();
  if (nodes.length < 3) { logLine("pen close needs 3+ nodes", "dim"); drawOverlay(); return; }
  // The drawn polyline is closed (last == first) as the shape fallback; when the
  // Bezier is present the server re-flattens it, so this is a mock-only backstop.
  const outline = penToPolyline(nodes, pxPerUm());
  outline.push([nodes[0].x, nodes[0].y]);
  await addObject("shape", outline, { bezier: penToBezier(nodes, true) });
}

// =========================================================================
//  SNAP-TO-BIND (EDIT_TOOLS §4 / docs/CONNECTIONS.md steps 1-2). Dragging the END
//  node of an OPEN path near a marked electrode, approaching ALONG the port's
//  direction arrow, snaps the endpoint onto the port centre and records a
//  BINDING (the port id) on that endpoint -- a reference the later connections
//  generator + drift resolve through the port, NOT a coordinate copy. This slice
//  only shows electrodes, snaps, and records the id (no connection object / funnel
//  generator / regenerate -- that is the later connections deliverable).
// =========================================================================
// Snap interaction radius in SCREEN PIXELS. At normal zoom a 1 µm pad's old
// 0.5 µm radius was a few px -- impossible to land on by hand -- so the radius
// is sized to ~14 px on screen (converted to µm via pxPerUm at the call site),
// which snaps reliably at any zoom. A wide pad still uses its half-width when
// that is larger.
const SNAP_PX = 14;
// Which end of an OPEN path a node index is, or null (interior node / closed path
// / too few nodes -- only the two ends of an open path can bind).
function endpointRole(bez, ni) {
  if (!bez || bez.closed || bez.nodes.length < 2) return null;
  if (ni === 0) return "start";
  if (ni === bez.nodes.length - 1) return "end";
  return null;
}
// The port an endpoint at (x, y) should snap+bind to, or null. Gated on the Show
// electrodes toggle (you snap to what you can see). PURE PROXIMITY -- like the
// close-path node snap (Slice 2): within the snap radius of a port centre and it
// binds, regardless of approach angle (no directional gate). The radius is a
// SCREEN-pixel target (max(width/2, SNAP_PX px in µm)) so it is reachable by
// hand at any zoom.
function snapCandidate(bez, ni, x, y) {
  const role = endpointRole(bez, ni);
  if (!role) return null;
  const cb = $("editShowPorts");
  if (!cb || !cb.checked) return null;                 // electrodes hidden -> no snap
  const ports = state.ports || [];
  if (!ports.length) return null;
  const pxScale = pxPerUm();            // µm per screen px is 1/this
  for (const p of ports) {
    // Screen-pixel snap radius: ~SNAP_PX px on screen (µm = px / pxPerUm), or
    // the pad's half-width when that is larger. Computed HERE (has the zoom);
    // the pure ec-geom helper only applies the proximity test.
    const radiusUm = Math.max((p.width_um || 0) / 2, SNAP_PX / pxScale);
    if (portSnapHit(p, x, y, radiusUm))
      return { portId: p.id, node: role, pos: [p.position_um[0], p.position_um[1]] };
  }
  return null;
}

// =========================================================================
//  POINTER + KEYBOARD HANDLERS (exported; ec-main attaches them to the overlay
//  canvas / window and this module owns their logic). One set of handlers; each
//  branches on activeTool. All gated on the Edit tab.
// =========================================================================
export async function editPointerDown(ev) {
  if (state.currentTab !== "edit" || ev.button !== 0) return;
  const [x, y] = evToUm(ev);
  ovCv.setPointerCapture(ev.pointerId);
  if (state.activeTool === "select") {
    // a transform handle on the current selection takes priority over a re-pick
    if (state.selected >= 0 && state.objects[state.selected]) {
      const k = hitTransformHandle(x, y);
      if (k) { startXform(k, x, y); drawOverlay(); return; }
    }
    const i = hitTest(x, y);
    if (i >= 0) {
      state.selected = i;
      moveDrag = { orig: state.objects[i].points_um.map((p) => [p[0], p[1]]), startUm: [x, y], moved: false };
      logLine(`selected ${state.objects[i].name}`, "tx");
    } else { state.selected = -1; }
    renderList(); renderSettings(); drawOverlay();
  } else if (state.activeTool === "node") {
    // pick / keep a node-editable object (path, shape or funnel outline)
    if (state.selected < 0 || !isNodeEditable(state.objects[state.selected])) {
      const i = hitTest(x, y);
      if (i >= 0 && isRectObject(state.objects[i])) {
        // rectangles are not node-editable this round: refuse with a note
        state.selected = i; renderList(); renderSettings();
        editBezier = null; selectedNodeIndex = -1; syncNodeToolbar(); drawOverlay();
        logLine("node tool: rectangles aren't node-editable (edit with Select)", "dim");
        return;
      }
      state.selected = (i >= 0 && isNodeEditable(state.objects[i])) ? i : -1;
      renderList(); renderSettings();
      if (state.selected < 0) { editBezier = null; selectedNodeIndex = -1; syncNodeToolbar(); drawOverlay(); logLine("node tool: select a path or shape first", "dim"); return; }
    }
    if (editBezierIndex !== state.selected || !editBezier) await loadEditBezier(state.selected);
    if (!editBezier) { syncNodeToolbar(); drawOverlay(); return; }
    nodeDrag = hitEditBezier(x, y);
    if (nodeDrag && (nodeDrag.type === "node" || nodeDrag.type === "handle")) {
      selectedNodeIndex = nodeDrag.ni;          // clicking a node/handle selects it
    } else if (!nodeDrag) {
      // empty click: let it re-pick another node-editable object under the cursor
      const i = hitTest(x, y);
      if (i >= 0 && isRectObject(state.objects[i])) {
        state.selected = i; renderList(); renderSettings();
        editBezier = null; selectedNodeIndex = -1; syncNodeToolbar(); drawOverlay();
        logLine("node tool: rectangles aren't node-editable (edit with Select)", "dim");
        return;
      }
      if (i >= 0 && isNodeEditable(state.objects[i]) && i !== state.selected) {
        state.selected = i; await loadEditBezier(i); renderList(); renderSettings();
      } else {
        selectedNodeIndex = -1;                 // clicked empty space: deselect the node
      }
    }
    syncNodeToolbar();
    drawOverlay();
  } else if (state.activeTool === "rect") {
    rubberRect = { x0: x, y0: y, x1: x, y1: y };
  } else if (state.activeTool === "pen") {
    pen.nodes.push({ x, y, hIn: null, hOut: null });
    pen.dragging = true; pen.cursor = [x, y];
    drawOverlay();
  }
}
export function editPointerMove(ev) {
  if (state.currentTab !== "edit") return;
  const [x, y] = evToUm(ev);
  if (state.activeTool === "select" && xformDrag) {
    if (xformDrag.type === "scale") applyScale(x, y, ev.shiftKey);
    else applyRotate(x, y, ev.shiftKey);
    drawOverlay();
  } else if (state.activeTool === "select" && moveDrag) {
    const dx = x - moveDrag.startUm[0], dy = y - moveDrag.startUm[1];
    if (dx || dy) moveDrag.moved = true;
    state.objects[state.selected].points_um = moveDrag.orig.map((p) => [p[0] + dx, p[1] + dy]);
    drawOverlay();
  } else if (state.activeTool === "node" && nodeDrag && editBezier) {
    const nodes = editBezier.nodes;
    if (nodeDrag.type === "node") {
      // snap-to-bind: an open path's END node dragged onto a port (along its
      // arrow) sticks to the port centre; otherwise it follows the cursor.
      snapActive = snapCandidate(editBezier, nodeDrag.ni, x, y);
      if (snapActive) bezMoveNode(nodes[nodeDrag.ni], snapActive.pos[0], snapActive.pos[1]);
      else bezMoveNode(nodes[nodeDrag.ni], x, y);
    }
    else if (nodeDrag.type === "handle") bezSetHandle(nodes[nodeDrag.ni], nodeDrag.which, x, y);
    else if (nodeDrag.type === "curve") bezCurveDrag(editBezier, nodeDrag, x, y);
    // live preview: re-flatten the edited base to the object's drawn polyline
    if (state.selected >= 0 && state.objects[state.selected]) state.objects[state.selected].points_um = bezFlatten(editBezier, 0.004);
    drawOverlay();
  } else if (state.activeTool === "rect" && rubberRect) {
    rubberRect.x1 = x; rubberRect.y1 = y; drawOverlay();
  } else if (state.activeTool === "pen" && pen.nodes.length) {
    pen.cursor = [x, y];
    if (pen.dragging) {
      const n = pen.nodes[pen.nodes.length - 1];
      n.hOut = { x, y };
      n.hIn = { x: 2 * n.x - x, y: 2 * n.y - y };   // symmetric
    }
    drawOverlay();
  }
}
export async function editPointerUp(ev) {
  if (state.currentTab !== "edit") return;
  try { ovCv.releasePointerCapture(ev.pointerId); } catch (e) {}
  if (state.activeTool === "select" && xformDrag) {
    const xd = xformDrag; xformDrag = null;
    const name = state.objects[state.selected] && state.objects[state.selected].name;
    if (xd.type === "scale") {
      logLine(`${name}: scaled (${num(xd.sx)}×, ${num(xd.sy)}×)`, "info");
      await persistTransform({ sx: xd.sx, sy: xd.sy, dx_um: 0, dy_um: 0,
        about_um: xd.anchor }, `Scaled ${name}`);
    } else {
      logLine(`${name}: rotated ${num(xd.rotDeg)}°`, "info");
      await persistTransform({ rot_deg: xd.rotDeg, about_um: [xd.cx, xd.cy] },
        `Rotated ${name}`);
    }
  } else if (state.activeTool === "select" && moveDrag) {
    const md = moveDrag; moveDrag = null;
    if (md.moved) {
      const dx = state.objects[state.selected].points_um[0][0] - md.orig[0][0];
      const dy = state.objects[state.selected].points_um[0][1] - md.orig[0][1];
      logLine(`${state.objects[state.selected].name} moved (${num(dx)}, ${num(dy)}) µm`, "info");
      await persistMove(dx, dy);
    }
  } else if (state.activeTool === "node" && nodeDrag && editBezier) {
    const i = state.selected; const kind = nodeDrag.type; const ni = nodeDrag.ni;
    nodeDrag = null;
    const snap = snapActive; snapActive = null;
    await finishNodeMove(i, ni, kind, snap);
  } else if (state.activeTool === "rect" && rubberRect) {
    const rr = rubberRect; rubberRect = null;
    const x0 = Math.min(rr.x0, rr.x1), x1 = Math.max(rr.x0, rr.x1);
    const y0 = Math.min(rr.y0, rr.y1), y1 = Math.max(rr.y0, rr.y1);
    if ((x1 - x0) < 0.05 || (y1 - y0) < 0.05) { logLine("rectangle too small", "dim"); drawOverlay(); return; }
    // a CLOSED 5-point outline (last == first), the shape representation the
    // page's mock rectangles expect, PLUS a 4-node closed rectangle bezier so the
    // backend routes it to fill_zigzag (as_rect) exactly like an imported rect --
    // NOT the fill_polygons fallback a bezier-less shape gets.
    await addObject("shape", [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]],
                    { bezier: rectBezier(x0, y0, x1, y1) });
  } else if (state.activeTool === "pen") {
    if (pen.dragging) {
      // a click with no meaningful drag stays a cusp node (no handles)
      const n = pen.nodes[pen.nodes.length - 1];
      if (n && n.hOut && Math.hypot(n.hOut.x - n.x, n.hOut.y - n.y) < 3 / pxPerUm()) {
        n.hIn = null; n.hOut = null;
      }
      pen.dragging = false;
      drawOverlay();
    }
  }
}
// Double-click ends the pen path (Inkscape's end-path gesture). On the START
// node it CLOSES the path (SVG `z`) -> a filled region/funnel; elsewhere it
// finishes an OPEN traced line.
export function editDblClick(ev) {
  if (state.currentTab !== "edit") return;
  // Node tool: double-clicking a segment inserts a node there (de Casteljau split).
  if (state.activeTool === "node") { insertNodeAtDblClick(ev); return; }
  if (state.activeTool !== "pen") return;
  ev.preventDefault();
  const start = pen.nodes[0];
  if (start && pen.nodes.length >= 3) {
    const [x, y] = evToUm(ev);
    const tol = 8 / pxPerUm();   // the node hit-radius (hitEditBezier)
    if (Math.hypot(start.x - x, start.y - y) <= tol) { closePen(); return; }
  }
  finishPen();
}
// Enter finishes an OPEN pen path. Escape now FINALIZES the draft with what is
// already down (finishPen) rather than abandoning it -- users kept adding points
// trying to stop the tool; a draft with <2 nodes just cancels (finishPen does).
export function editKeydown(ev) {
  if (state.currentTab !== "edit") return;
  const t = ev.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
  if (ev.key === "Enter" && state.activeTool === "pen" && pen.nodes.length) { ev.preventDefault(); finishPen(); }
  else if (ev.key === "Escape") {
    if (state.activeTool === "pen" && pen.nodes.length) { ev.preventDefault(); finishPen(); }
    if (rubberRect) { rubberRect = null; drawOverlay(); }
  }
}

// =========================================================================
//  OVERLAY PAINTER (registered by ec-main via the painter registry). Draws the
//  edit-tool overlays (rubber rectangle, pen draft + handles, node handles,
//  select-tool transform box). No-op off the Edit tab so Scan/Write render
//  exactly as before.
// =========================================================================
export function drawEditLayer(tf) {
  if (state.currentTab !== "edit") return;
  syncEditReadouts();
  // Handles and boxes: panel fill with a text-coloured stroke; the active or
  // selected one in the accent. No green or red here: on the canvas they mean bias.
  const acc = cssVar("--accent"), fg = cssVar("--fg"), panel = cssVar("--panel");
  const dim = cssVar("--dim");
  ovCx.save();
  // rubber rectangle preview
  if (rubberRect) {
    const [ax, ay] = tf(rubberRect.x0, rubberRect.y0);
    const [bx, by] = tf(rubberRect.x1, rubberRect.y1);
    ovCx.strokeStyle = acc;
    ovCx.fillStyle = cssVarA("--accent", 0.12);
    ovCx.lineWidth = 1.5; ovCx.setLineDash([5, 3]);
    ovCx.beginPath();
    ovCx.rect(Math.min(ax, bx), Math.min(ay, by), Math.abs(bx - ax), Math.abs(by - ay));
    ovCx.fill(); ovCx.stroke();
    ovCx.setLineDash([]);
  }
  // pen draft: the committed segments, the rubber segment to the cursor, then
  // the anchor nodes and their symmetric handles
  if (pen.nodes.length) {
    const poly = penToPolyline(pen.nodes, pxPerUm());
    ovCx.strokeStyle = acc;
    ovCx.lineWidth = 2; ovCx.setLineDash([]);
    ovCx.beginPath();
    let [px, py] = tf(poly[0][0], poly[0][1]); ovCx.moveTo(px, py);
    for (let i = 1; i < poly.length; i++) { const [x, y] = tf(poly[i][0], poly[i][1]); ovCx.lineTo(x, y); }
    ovCx.stroke();
    // rubber segment from the last node to the cursor
    if (pen.cursor && !pen.dragging) {
      const last = pen.nodes[pen.nodes.length - 1];
      const [lx, ly] = tf(last.x, last.y), [cx, cy] = tf(pen.cursor[0], pen.cursor[1]);
      ovCx.strokeStyle = cssVarA("--accent", 0.5); ovCx.setLineDash([4, 4]);
      ovCx.beginPath(); ovCx.moveTo(lx, ly); ovCx.lineTo(cx, cy); ovCx.stroke();
      ovCx.setLineDash([]);
    }
    for (const n of pen.nodes) {
      for (const h of [n.hIn, n.hOut]) {
        if (!h) continue;
        const [nx, ny] = tf(n.x, n.y), [hx, hy] = tf(h.x, h.y);
        ovCx.strokeStyle = dim; ovCx.lineWidth = 1;
        ovCx.beginPath(); ovCx.moveTo(nx, ny); ovCx.lineTo(hx, hy); ovCx.stroke();
        ovCx.fillStyle = panel; ovCx.strokeStyle = fg;
        ovCx.beginPath(); ovCx.arc(hx, hy, 3, 0, 2 * Math.PI); ovCx.fill(); ovCx.stroke();
      }
      const [nx, ny] = tf(n.x, n.y);
      ovCx.fillStyle = panel; ovCx.strokeStyle = fg; ovCx.lineWidth = 1;
      ovCx.fillRect(nx - 3, ny - 3, 6, 6); ovCx.strokeRect(nx - 3, ny - 3, 6, 6);
    }
    // close affordance: hovering the start node (with 3+ nodes down) shows a
    // "○" so the double-click-to-close gesture is discoverable.
    if (pen.nodes.length >= 3 && pen.cursor) {
      const s = pen.nodes[0];
      if (Math.hypot(s.x - pen.cursor[0], s.y - pen.cursor[1]) <= 8 / pxPerUm()) {
        const [sx, sy] = tf(s.x, s.y);
        ovCx.strokeStyle = acc; ovCx.lineWidth = 2; ovCx.setLineDash([]);
        ovCx.beginPath(); ovCx.arc(sx, sy, 7, 0, 2 * Math.PI); ovCx.stroke();
      }
    }
  }
  // node tool: the editable Bézier -- handle lines + circles, then node squares.
  // Inkscape style: the flattened curve is already drawn (as the object's
  // polyline); here we overlay the control points on top of it.
  if (state.activeTool === "node" && editBezier && state.selected === editBezierIndex && state.selected >= 0) {
    ovCx.setLineDash([]);
    for (let i = 0; i < editBezier.nodes.length; i++) {
      const n = editBezier.nodes[i];
      const [nx, ny] = tf(n.x, n.y);
      for (const [h, which] of [[n.h_in, "in"], [n.h_out, "out"]]) {
        if (!h) continue;
        const [hx, hy] = tf(h[0], h[1]);
        ovCx.strokeStyle = dim; ovCx.lineWidth = 1;
        ovCx.beginPath(); ovCx.moveTo(nx, ny); ovCx.lineTo(hx, hy); ovCx.stroke();
        const on = nodeDrag && nodeDrag.type === "handle" && nodeDrag.ni === i && nodeDrag.which === which;
        ovCx.fillStyle = on ? acc : panel;
        ovCx.beginPath(); ovCx.arc(hx, hy, 3.2, 0, 2 * Math.PI); ovCx.fill();
        ovCx.strokeStyle = fg; ovCx.stroke();
      }
    }
    for (let i = 0; i < editBezier.nodes.length; i++) {
      const n = editBezier.nodes[i];
      const [x, y] = tf(n.x, n.y);
      const on = nodeDrag && nodeDrag.type === "node" && nodeDrag.ni === i;
      const sel = i === selectedNodeIndex;   // the toolbar's target node
      // the selected node reads bigger and in the accent so the toolbar's target is clear
      const r = sel ? 4.5 : 3.5;
      ovCx.fillStyle = on || sel ? acc : panel;
      ovCx.fillRect(x - r, y - r, 2 * r, 2 * r);
      ovCx.strokeStyle = fg;
      ovCx.lineWidth = sel ? 1.5 : 1;
      ovCx.strokeRect(x - r, y - r, 2 * r, 2 * r);
    }
    if (editBezierModulated) {
      ovCx.fillStyle = cssVar("--warn");
      ovCx.font = `10px ${cssVar("--mono")}`;
      const n0 = editBezier.nodes[0];
      if (n0) { const [lx, ly] = tf(n0.x, n0.y); ovCx.fillText("editing base (modulation re-applies)", lx + 6, ly + 14); }
    }
  }
  // select tool: the transform-handle set -- bounding box, 8 scale handles,
  // and a rotate handle on a stem above the top edge (Inkscape select-tool).
  if (state.activeTool === "select" && state.selected >= 0 && state.objects[state.selected]) {
    const b = objBounds(state.objects[state.selected]);
    if (b) {
      const H = handlePoints(b);
      const [x0, y0] = tf(b.minx, b.miny), [x1, y1] = tf(b.maxx, b.maxy);
      ovCx.strokeStyle = acc; ovCx.lineWidth = 1; ovCx.setLineDash([4, 3]);
      ovCx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
      ovCx.setLineDash([]);
      const [nx, ny] = tf(H.n[0], H.n[1]), [rx, ry] = tf(H.rot[0], H.rot[1]);
      ovCx.strokeStyle = cssVarA("--accent", 0.7); ovCx.lineWidth = 1;
      ovCx.beginPath(); ovCx.moveTo(nx, ny); ovCx.lineTo(rx, ry); ovCx.stroke();
      for (const k of ["nw", "n", "ne", "e", "se", "s", "sw", "w"]) {
        const [hx, hy] = tf(H[k][0], H[k][1]);
        ovCx.fillStyle = panel; ovCx.fillRect(hx - 4, hy - 4, 8, 8);
        ovCx.strokeStyle = fg; ovCx.strokeRect(hx - 4, hy - 4, 8, 8);
      }
      ovCx.fillStyle = acc;
      ovCx.beginPath(); ovCx.arc(rx, ry, 4.5, 0, 2 * Math.PI); ovCx.fill();
      ovCx.strokeStyle = fg; ovCx.stroke();
    }
  }
  // snap-to-bind (EDIT_TOOLS §4): while a drag is snapping, a solid accent ring on
  // the target port; and a persistent dashed ring on the selected object's
  // already-bound endpoint so the recorded binding stays visible after the drag.
  if (snapActive) {
    const [sx, sy] = tf(snapActive.pos[0], snapActive.pos[1]);
    ovCx.strokeStyle = acc; ovCx.lineWidth = 2.5; ovCx.setLineDash([]);
    ovCx.beginPath(); ovCx.arc(sx, sy, 9, 0, 2 * Math.PI); ovCx.stroke();
  }
  if (state.activeTool === "node" && editBezier && !editBezier.closed
      && state.selected === editBezierIndex && state.selected >= 0) {
    const o = state.objects[state.selected];
    const b = o && o.end_binding;
    if (b && (b.node === "start" || b.node === "end")) {
      const n = editBezier.nodes[b.node === "start" ? 0 : editBezier.nodes.length - 1];
      if (n) {
        const [bx, by] = tf(n.x, n.y);
        ovCx.strokeStyle = cssVarA("--accent", 0.7); ovCx.lineWidth = 1.8; ovCx.setLineDash([2, 2]);
        ovCx.beginPath(); ovCx.arc(bx, by, 7, 0, 2 * Math.PI); ovCx.stroke();
        ovCx.setLineDash([]);
      }
    }
  }
  ovCx.restore();
}

// =========================================================================
//  SHARED HOOKS -- ec-main's setTab / select() call these so the tab's transient
//  drafts and node-tool base Bézier stay owned here (never reachable in two
//  places, docs/COCKPIT_MODULES.md §8).
// =========================================================================
// setTab: leaving/entering a tab abandons any edit draft.
export function editAbortDrafts() {
  cancelPen(); rubberRect = null;
  nodeDrag = null; xformDrag = null; moveDrag = null; snapActive = null;
  syncNodeToolbar();   // hide the toolbar when leaving the Edit tab
}
// select(): the selection changed, so the node tool must reload its base Bézier.
export function editResetOnSelect() {
  editBezier = null; editBezierIndex = -1; selectedNodeIndex = -1;
  syncNodeToolbar();
}
// select(): if the node tool is up, show the new selection's handles immediately.
export function editReloadOnSelect() {
  if (state.currentTab === "edit" && state.activeTool === "node" && state.selected >= 0
      && isNodeEditable(state.objects[state.selected]))
    loadEditBezier(state.selected).then(() => { syncNodeToolbar(); drawOverlay(); });
  else
    syncNodeToolbar();
}

// =========================================================================
//  INIT -- ec-main injects the shared DOM/redraw helpers, then this wires the
//  tool palette + the delete button. Called once at boot, before the first
//  drawOverlay (so drawEditLayer's injected ovCx/pxPerUm are set by then).
// =========================================================================
export function initEditTab(deps) {
  evToUm = deps.evToUm;
  pxPerUm = deps.pxPerUm;
  drawOverlay = deps.drawOverlay;
  renderList = deps.renderList;
  renderSettings = deps.renderSettings;
  adoptDesign = deps.adoptDesign;
  refreshToolpath = deps.refreshToolpath;
  applyToolCursor = deps.applyToolCursor;
  ovCv = deps.ovCv;
  ovCx = deps.ovCx;
  reAdopt = deps.reAdopt;
  clipboard = deps.clipboard;
  xyRangeUm = deps.xyRangeUm;

  // Palette buttons pick the active tool (and clear any in-progress draft).
  for (const b of document.querySelectorAll(".tool"))
    b.addEventListener("click", async () => {
      for (const x of document.querySelectorAll(".tool")) x.classList.toggle("active", x === b);
      state.activeTool = b.dataset.tool;
      cancelPen(); rubberRect = null;
      nodeDrag = null; xformDrag = null; moveDrag = null; snapActive = null;
      editBezier = null; editBezierIndex = -1;   // reloaded on demand by the node tool
      selectedNodeIndex = -1;
      applyToolCursor();
      logLine(`tool: ${state.activeTool}`, "dim");
      // switching TO the node tool with a node-editable object already selected:
      // load its base Bézier so the handles appear without a first click
      if (state.activeTool === "node" && state.selected >= 0 && isNodeEditable(state.objects[state.selected]))
        await loadEditBezier(state.selected);
      syncNodeToolbar();
      drawOverlay();
    });

  // Node toolbar: Type buttons set the selected node's continuity; Modify buttons
  // run the ec-bezier op (Insert is the double-click-a-segment gesture, so its
  // button is a discoverable hint). All persist through commitBezEdit.
  for (const b of document.querySelectorAll("#nodeToolbar button.ntype"))
    b.addEventListener("click", () => applyNodeType(b.dataset.kind));
  const wire = (id, fn) => { const el = $(id); if (el) el.addEventListener("click", fn); };
  wire("nInsert", () => logLine("insert: double-click a segment to add a node there", "dim"));
  wire("nDelete", deleteSelectedNode);
  wire("nJoin", joinPathEnds);
  wire("nSplit", splitAtSelectedNode);
  for (const id of [...SEL_BOX, ...SEL_AMOUNT]) {
    const el = $(id);
    if (!el) continue;
    el.dataset.shown = el.value;
    el.addEventListener("change", () => applySelectField(el));
    el.addEventListener("keydown", (ev) => { if (ev.key === "Enter") applySelectField(el); });
  }
  wire("selCopy", () => clipboard.copy());
  wire("selPaste", () => clipboard.paste());
  wire("selDup", () => clipboard.duplicate());
  for (const id of ["nodeX", "nodeY"]) {
    const el = $(id);
    if (!el) continue;
    el.addEventListener("change", () => applyNodeField(el));
    el.addEventListener("keydown", (ev) => { if (ev.key === "Enter") applyNodeField(el); });
  }
  syncNodeToolbar();   // hidden until the node tool is picked

  // Delete the currently-selected object. For an imported design this uses the
  // EXISTING object-delete mechanism (POST /api/design/op {op:"delete"}), then
  // re-adopts the returned design; for mock objects it removes locally, matching
  // the local-only behaviour of the other edits on this page.
  $("editDelete").onclick = async () => {
    if (state.selected < 0) { logLine("delete: no object selected", "warn"); return; }
    const o = state.objects[state.selected];
    if (!state.designImported) {
      logLine(`deleting ${o.name} (local mock)`, "tx");
      state.objects.splice(state.selected, 1);
      state.selected = -1;
      renderList(); renderSettings(); drawOverlay();
      return;
    }
    const res = await postDesignOp({ op: "delete", indices: [state.selected] });
    if (!res.ok || res.data.error) return;   // request() logged it
    adoptDesign(res.data.design, state.sourceLabel);
    await refreshToolpath();        // the compiled path changed
    drawOverlay();
  };
}
