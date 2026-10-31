// The JS port of litho/bezier.py's Node + BezierPath, so the node tool edits the
// SAME control-point model the backend stores and writes. A node is
// { x, y, h_in:[x,y]|null, h_out:[x,y]|null, kind }; a path is { nodes:[...],
// closed }. h_in governs the segment ARRIVING, h_out the one LEAVING; null means
// "coincident with the anchor" (that side is straight).
//
// DOM-free (Layer 1): every function is a pure transform of its arguments, so
// this module imports under node, where it is contract-tested against the Python
// original (tests/ec_modules.mjs). Anything needing the current zoom (adaptive
// segment counts) takes pixels-per-µm as an explicit argument rather than reading
// the canvas, so nothing here touches the DOM.

export const CUSP = "cusp", SMOOTH = "smooth", SYMMETRIC = "symmetric";

export function bezHandleIn(n)  { return n.h_in  || [n.x, n.y]; }
export function bezHandleOut(n) { return n.h_out || [n.x, n.y]; }

// Drag the anchor; the handles travel with it (Node.move_to).
export function bezMoveNode(n, x, y) {
  const dx = x - n.x, dy = y - n.y;
  if (n.h_in)  n.h_in  = [n.h_in[0]  + dx, n.h_in[1]  + dy];
  if (n.h_out) n.h_out = [n.h_out[0] + dx, n.h_out[1] + dy];
  n.x = x; n.y = y;
}
// Move one handle, enforcing this node's continuity rule (Node.set_handle):
// smooth keeps the opposite handle collinear (its length), symmetric mirrors it,
// cusp lets the two move independently (a corner).
export function bezSetHandle(n, which, x, y) {
  const vx = x - n.x, vy = y - n.y;
  let other, keep;
  if (which === "out") { n.h_out = [x, y]; other = "h_in";  keep = bezHandleIn(n); }
  else                 { n.h_in  = [x, y]; other = "h_out"; keep = bezHandleOut(n); }
  if (n.kind === CUSP) return;
  const nrm = Math.hypot(vx, vy);
  if (nrm < 1e-12) return;
  const ux = -vx / nrm, uy = -vy / nrm;
  let length = Math.hypot(keep[0] - n.x, keep[1] - n.y);
  if (n.kind === SYMMETRIC || length < 1e-12) length = nrm;
  n[other] = [n.x + ux * length, n.y + uy * length];
}
// Yield [p0, c1, c2, p3, i0, i1] per segment, closing if closed (BezierPath.segments).
export function bezSegments(bez) {
  const nodes = bez.nodes, n = nodes.length, out = [];
  if (n < 2) return out;
  const last = bez.closed ? n : n - 1;
  for (let i = 0; i < last; i++) {
    const a = nodes[i], b = nodes[(i + 1) % n];
    out.push([[a.x, a.y], bezHandleOut(a), bezHandleIn(b), [b.x, b.y], i, (i + 1) % n]);
  }
  return out;
}
export function cubicAt(p0, c1, c2, p3, t) {
  const u = 1 - t;
  return [u * u * u * p0[0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t * t * t * p3[0],
          u * u * u * p0[1] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t * t * t * p3[1]];
}
// Adaptive subdivision to tol µm -- the same construction as bezier.flatten_cubic,
// so the client preview matches what the backend re-flattens on persist.
export function bezFlatten(bez, tol) {
  tol = tol || 0.004;
  const nodes = bez.nodes;
  if (!nodes.length) return [];
  const pts = [[nodes[0].x, nodes[0].y]];
  const sub = (p0, c1, c2, p3, d) => {
    const chx = p3[0] - p0[0], chy = p3[1] - p0[1];
    const L = Math.hypot(chx, chy);
    let flat;
    if (L < 1e-12) flat = Math.max(Math.hypot(c1[0] - p0[0], c1[1] - p0[1]),
                                   Math.hypot(c2[0] - p0[0], c2[1] - p0[1]));
    else {
      const nx = -chy / L, ny = chx / L;
      flat = Math.max(Math.abs((c1[0] - p0[0]) * nx + (c1[1] - p0[1]) * ny),
                      Math.abs((c2[0] - p0[0]) * nx + (c2[1] - p0[1]) * ny));
    }
    if (d >= 18 || flat <= tol) { pts.push(p3); return; }
    const a = [(p0[0] + c1[0]) / 2, (p0[1] + c1[1]) / 2];
    const b = [(c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2];
    const cc = [(c2[0] + p3[0]) / 2, (c2[1] + p3[1]) / 2];
    const dd = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    const e = [(b[0] + cc[0]) / 2, (b[1] + cc[1]) / 2];
    const m = [(dd[0] + e[0]) / 2, (dd[1] + e[1]) / 2];
    sub(p0, a, dd, m, d + 1); sub(m, e, cc, p3, d + 1);
  };
  for (const [p0, c1, c2, p3] of bezSegments(bez)) sub(p0, c1, c2, p3, 0);
  return pts;
}
// A backend/mock bezier dict -> a clean editable copy (arrays, kinds present).
export function normalizeBez(b) {
  return {
    closed: !!(b && b.closed),
    nodes: ((b && b.nodes) || []).map((n) => ({
      x: +n.x, y: +n.y,
      h_in:  n.h_in  ? [+n.h_in[0],  +n.h_in[1]]  : null,
      h_out: n.h_out ? [+n.h_out[0], +n.h_out[1]] : null,
      kind: n.kind || CUSP,
    })),
  };
}
// Synthesize a cusp-node bezier from a polyline (mock / no-control-point path).
export function synthBezier(pts, closed) {
  let p = (pts || []).map((q) => [+q[0], +q[1]]);
  const isClosed = closed || (p.length >= 4 &&
    Math.abs(p[0][0] - p[p.length - 1][0]) < 1e-9 &&
    Math.abs(p[0][1] - p[p.length - 1][1]) < 1e-9);
  if (isClosed && p.length >= 2 &&
      Math.abs(p[0][0] - p[p.length - 1][0]) < 1e-9 &&
      Math.abs(p[0][1] - p[p.length - 1][1]) < 1e-9) p = p.slice(0, -1);
  return { closed: !!isClosed,
    nodes: p.map(([x, y]) => ({ x, y, h_in: null, h_out: null, kind: CUSP })) };
}
// Curve-drag: bend a segment by adjusting its two bounding nodes' handles, the
// "Magic Bézier Drag Equations" from Inkscape's curve-drag-point.cpp. A straight
// segment first seeds handles at 1/3 (grabbed()), turning the line into a curve.
export function bezCurveDrag(bez, drag, mx, my) {
  const a = bez.nodes[drag.ni0], b = bez.nodes[drag.ni1];
  const t = drag.t;
  if (!a.h_out && !b.h_in) {
    a.h_out = [a.x + (b.x - a.x) / 3, a.y + (b.y - a.y) / 3];
    b.h_in  = [b.x - (b.x - a.x) / 3, b.y - (b.y - a.y) / 3];
  }
  const c1 = bezHandleOut(a), c2 = bezHandleIn(b);
  const pos = cubicAt([a.x, a.y], c1, c2, [b.x, b.y], t);
  const dx = mx - pos[0], dy = my - pos[1];
  let weight;
  if (t <= 1 / 6) weight = 0;
  else if (t <= 0.5) weight = Math.pow((6 * t - 1) / 2, 3) / 2;
  else if (t <= 5 / 6) weight = (1 - Math.pow((6 * (1 - t) - 1) / 2, 3)) / 2 + 0.5;
  else weight = 1;
  const o0 = (1 - weight) / (3 * t * (1 - t) * (1 - t));
  const o1 = weight / (3 * t * t * (1 - t));
  bezSetHandle(a, "out", c1[0] + o0 * dx, c1[1] + o0 * dy);
  bezSetHandle(b, "in",  c2[0] + o1 * dx, c2[1] + o1 * dy);
}
// Flatten the pen's node list to a polyline (microns): straight where a segment
// has no handles, a cubic Bezier where either endpoint carries one. `p2u` is the
// current pixels-per-µm (passed in so this stays DOM-free): it sets the adaptive
// segment count so a curve never facets at the current zoom.
export function penToPolyline(nodes, p2u) {
  if (nodes.length < 2) return nodes.map((n) => [n.x, n.y]);
  const out = [[nodes[0].x, nodes[0].y]];
  for (let i = 1; i < nodes.length; i++) {
    const a = nodes[i - 1], b = nodes[i];
    if (!a.hOut && !b.hIn) { out.push([b.x, b.y]); continue; }
    const c1 = a.hOut || { x: a.x, y: a.y }, c2 = b.hIn || { x: b.x, y: b.y };
    const len = Math.hypot(b.x - a.x, b.y - a.y);
    const N = Math.max(8, Math.min(64, Math.ceil(len * p2u / 6)));
    for (let s = 1; s <= N; s++) {
      const t = s / N, u = 1 - t;
      const x = u * u * u * a.x + 3 * u * u * t * c1.x + 3 * u * t * t * c2.x + t * t * t * b.x;
      const y = u * u * u * a.y + 3 * u * u * t * c1.y + 3 * u * t * t * c2.y + t * t * t * b.y;
      out.push([x, y]);
    }
  }
  return out;
}
// The pen's nodes, as a BézierPath.to_dict() (the same {x,y,h_in,h_out,kind}
// model the backend stores and the node tool edits). A node dragged out has
// symmetric handles (see pointermove), so it is a SMOOTH/SYMMETRIC node; a plain
// click is a CUSP corner. Kept on finish so the path is immediately node-editable
// with its ORIGINAL control handles — not re-fitted from a flattened polyline.
export function penToBezier(nodes, closed = false) {
  return {
    closed: !!closed,
    nodes: nodes.map((n) => ({
      x: n.x, y: n.y,
      h_in:  n.hIn  ? [n.hIn[0]  !== undefined ? n.hIn[0]  : n.hIn.x,
                       n.hIn[1]  !== undefined ? n.hIn[1]  : n.hIn.y] : null,
      h_out: n.hOut ? [n.hOut[0] !== undefined ? n.hOut[0] : n.hOut.x,
                       n.hOut[1] !== undefined ? n.hOut[1] : n.hOut.y] : null,
      kind: (n.hIn || n.hOut) ? SYMMETRIC : CUSP,
    })),
  };
}
// A drawn rectangle, as a BézierPath.to_dict() the backend recognizes via
// BezierPath.as_rect() -> the tested fill_zigzag unidirectional-return raster.
// WITHOUT this bezier the object has bezier=None, misses the as_rect branch in
// expand(), and falls to fill_polygons (the wrong raster). The four corners are
// CUSP (no handles) and ordered axis-aligned so as_rect (exactly 4 closed nodes,
// right angles) recognizes them -> a drawn rect === an imported rect.
export function rectBezier(x0, y0, x1, y1) {
  const corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  return {
    closed: true,
    nodes: corners.map(([x, y]) => ({ x, y, h_in: null, h_out: null, kind: CUSP })),
  };
}

// =========================================================================
//  NODE OPERATIONS (the node-toolbar ops, docs/EDIT_TOOLS.md §3). Pure
//  transforms of the bezier dict / a node -- no DOM, so they import and are
//  unit-tested under node (tests/ec_modules.mjs) alongside the flatten contract.
// =========================================================================

// Change a node's continuity kind and RE-DERIVE its handles to obey it (Type
// buttons). cusp leaves the two handles independent; smooth/symmetric re-align
// the opposite handle off whichever handle currently exists (h_out preferred),
// reusing bezSetHandle's continuity rule so a re-typed node behaves exactly like
// one dragged with that kind. A handle-less node just records the kind (there is
// nothing to align until a handle is pulled).
export function setNodeType(node, kind) {
  node.kind = kind;
  if (kind === CUSP) return node;
  if (node.h_out) bezSetHandle(node, "out", node.h_out[0], node.h_out[1]);
  else if (node.h_in) bezSetHandle(node, "in", node.h_in[0], node.h_in[1]);
  return node;
}

// DUMB delete (Delete button): drop node i; its neighbours reconnect directly
// with whatever handles they still carry (the segment between them becomes a
// plain cubic). No shape-preserving re-fit. Guarded so a path never drops below
// two nodes. Returns the (mutated) bezier.
export function deleteNode(bez, i) {
  if (!bez || !bez.nodes || bez.nodes.length <= 2) return bez;
  if (i < 0 || i >= bez.nodes.length) return bez;
  bez.nodes.splice(i, 1);
  return bez;
}

// Insert a node on segment `segIndex` at parameter t (double-click a segment).
// Splits the cubic at t by de Casteljau, so the two new sub-curves reproduce the
// original curve EXACTLY: the flanking nodes' facing handles shrink to the split
// controls and the new (smooth) node carries the interior controls. A handle that
// collapses onto its anchor is stored as null (a straight side stays straight).
// Mutates bez; returns the index of the inserted node.
export function insertNode(bez, segIndex, t) {
  const nodes = bez.nodes, n = nodes.length;
  const A = nodes[segIndex], B = nodes[(segIndex + 1) % n];
  const p0 = [A.x, A.y], p3 = [B.x, B.y];
  const c1 = bezHandleOut(A), c2 = bezHandleIn(B);
  const L = (u, v) => [u[0] + (v[0] - u[0]) * t, u[1] + (v[1] - u[1]) * t];
  const p01 = L(p0, c1), p12 = L(c1, c2), p23 = L(c2, p3);
  const p012 = L(p01, p12), p123 = L(p12, p23);
  const mid = L(p012, p123);
  const nz = (h, anchor) =>
    (Math.hypot(h[0] - anchor[0], h[1] - anchor[1]) < 1e-9) ? null : [h[0], h[1]];
  A.h_out = nz(p01, p0);
  B.h_in = nz(p23, p3);
  const nn = { x: mid[0], y: mid[1], h_in: nz(p012, mid), h_out: nz(p123, mid), kind: SMOOTH };
  bez.nodes.splice(segIndex + 1, 0, nn);
  return segIndex + 1;
}

// Join two endpoint nodes into one continuous path (Join button): merge the two
// ends of an OPEN path -- average their positions onto the lower-indexed survivor,
// carry the other end's arriving (h_in) handle onto it, drop the other node, and
// close the path. The inverse of splitNode. Mutates bez; returns it.
export function joinNodes(bez, i, j) {
  if (!bez || i === j) return bez;
  const nodes = bez.nodes;
  const lo = Math.min(i, j), hi = Math.max(i, j);
  const a = nodes[lo], b = nodes[hi];
  a.x = (a.x + b.x) / 2; a.y = (a.y + b.y) / 2;
  a.h_in = b.h_in ? [b.h_in[0], b.h_in[1]] : null;
  nodes.splice(hi, 1);
  bez.closed = true;
  return bez;
}

// Break a CLOSED path open at node i (Split button): rotate the ring so it starts
// at i, then duplicate that node as a coincident closing endpoint -- two ends
// where there was one continuous seam. The inverse of joinNodes. A path that is
// already open is returned unchanged. Mutates bez; returns it.
export function splitNode(bez, i) {
  if (!bez || !bez.closed || !bez.nodes.length) return bez;
  const n = bez.nodes.length;
  if (i < 0 || i >= n) return bez;
  const rot = bez.nodes.slice(i).concat(bez.nodes.slice(0, i));
  const head = rot[0];
  const dup = {
    x: head.x, y: head.y,
    h_in: head.h_in ? [head.h_in[0], head.h_in[1]] : null,
    h_out: head.h_out ? [head.h_out[0], head.h_out[1]] : null,
    kind: head.kind,
  };
  head.h_in = null;   // the opening start: nothing arrives at it now
  dup.h_out = null;   // the closing end: nothing leaves it now
  bez.nodes = rot.concat([dup]);
  bez.closed = false;
  return bez;
}

// True if this bezier is a recognized rectangle -- the JS mirror of
// litho/bezier.py BezierPath.as_rect (exactly 4 closed nodes, 4 straight sides
// meeting at right angles). The node tool refuses rectangles this round
// (docs/EDIT_TOOLS.md §3), and the cockpit labels them "rectangle" (vs a generic
// "region"); both ask this. Node kind is not tested -- a rectangle's corners are
// cusps by construction, but as_rect judges straightness from the control points.
export function isRectBezier(bez) {
  if (!bez || !bez.closed || !bez.nodes || bez.nodes.length !== 4) return false;
  const segs = bezSegments(bez);
  if (segs.length !== 4) return false;
  const cr = (a, b) => Math.abs(a[0] * b[1] - a[1] * b[0]);
  const straight = ([p0, c1, c2, p3]) => {
    const chx = p3[0] - p0[0], chy = p3[1] - p0[1];
    const chord = Math.hypot(chx, chy);
    if (chord < 1e-9) return false;
    const s = cr([chx, chy], [c1[0] - p0[0], c1[1] - p0[1]])
            + cr([chx, chy], [c2[0] - p0[0], c2[1] - p0[1]]);
    return s <= 0.05 * chord * chord;
  };
  if (!segs.every(straight)) return false;
  const pts = bez.nodes.map((nd) => [nd.x, nd.y]);
  const unit = (vx, vy) => { const m = Math.hypot(vx, vy); return m > 1e-9 ? [vx / m, vy / m] : [vx, vy]; };
  const edges = [];
  for (let i = 0; i < 4; i++) edges.push([pts[(i + 1) % 4][0] - pts[i][0], pts[(i + 1) % 4][1] - pts[i][1]]);
  for (let i = 0; i < 4; i++) {
    const a = unit(edges[i][0], edges[i][1]);
    const b = unit(edges[(i + 1) % 4][0], edges[(i + 1) % 4][1]);
    if (Math.abs(a[0] * b[0] + a[1] * b[1]) > 0.08) return false;
  }
  return true;
}
