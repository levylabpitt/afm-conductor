// View-transform math and 2-D hit-test cores for the edit cockpit (Layer 1,
// DOM-free). Geometry is authored in MICRONS; one fit/inv pair maps µm <-> canvas
// pixels, and every layer draws through it so the scan image, design overlay,
// litho trace, and ports all land in register.
//
// Everything here is a pure function of its arguments -- the current `view` box
// and the canvas pixel size are passed IN rather than read off the DOM -- so this
// module imports under node, where the round-trip (invTransform(fitTransform(p))
// == p) and the hit-test tolerances are unit-tested. ec-main.js keeps thin
// adapters that bind these to the live `view`/`SCAN_FRAME`/canvas.

// Fit a micron `view` box {x0,y0,w,h} into a px*py pixel canvas, preserving
// aspect and centering with the leftover as margin. Returns a µm -> pixel mapping
// (x,y) -> [sx,sy]; y flips so micron-up is screen-up.
export function fitTransform(view, px, py) {
  const v = view;
  const sx = px / v.w, sy = py / v.h;
  const s = Math.min(sx, sy);
  const offx = (px - s * v.w) / 2;
  const offy = (py - s * v.h) / 2;
  return (x, y) => [offx + (x - v.x0) * s, py - (offy + (y - v.y0) * s)];
}

// The exact INVERSE of fitTransform for the same view + canvas size: a pixel
// (backing-store) coordinate -> micron. Same s/offx/offy so a pointer maps back
// to the micron the overlay drew at.
export function invTransform(view, px, py) {
  const v = view;
  const s = Math.min(px / v.w, py / v.h);
  const offx = (px - s * v.w) / 2;
  const offy = (py - s * v.h) / 2;
  return (sxp, syp) => [v.x0 + (sxp - offx) / s, v.y0 + (py - syp - offy) / s];
}

// Pixels-per-micron of the current view (matches fitTransform's scale). Turns a
// hit/handle tolerance in px into microns.
export function pxPerUm(view, px, py) {
  const v = view;
  return Math.min(px / v.w, py / v.h);
}

// Even-odd point-in-polygon (microns). pts is [[x,y],...].
export function pointInPoly(pts, x, y) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const xi = pts[i][0], yi = pts[i][1], xj = pts[j][0], yj = pts[j][1];
    if (((yi > y) !== (yj > y)) &&
        (x < (xj - xi) * (y - yi) / ((yj - yi) || 1e-12) + xi)) inside = !inside;
  }
  return inside;
}

// Distance from a point to a segment [a,b] (microns).
export function distToSeg(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const l2 = dx * dx + dy * dy;
  let t = l2 ? ((px - ax) * dx + (py - ay) * dy) / l2 : 0;
  t = Math.max(0, Math.min(1, t));
  const cx = ax + t * dx, cy = ay + t * dy;
  return Math.hypot(px - cx, py - cy);
}

// Nearest distance from a point to a polyline (microns).
export function distToPolyline(pts, x, y) {
  let d = Infinity;
  for (let i = 1; i < pts.length; i++)
    d = Math.min(d, distToSeg(x, y, pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1]));
  return d;
}

// Axis-aligned bounds of one object's points_um, or null if it has no extent
// (empty, or a single degenerate point). Used by the select-tool transform box.
export function objBounds(o) {
  let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
  for (const [x, y] of (o.points_um || [])) {
    if (x < minx) minx = x; if (y < miny) miny = y;
    if (x > maxx) maxx = x; if (y > maxy) maxy = y;
  }
  if (!isFinite(minx) || maxx - minx < 1e-9 && maxy - miny < 1e-9) return null;
  return { minx, miny, maxx, maxy };
}

// Skew point p about c: x shifts by tan(ax) per unit of y, y by tan(ay) per unit
// of x (angles in degrees), both from p's original position.
export function skewAbout(p, c, axDeg, ayDeg) {
  const tx = Math.tan(axDeg * Math.PI / 180), ty = Math.tan(ayDeg * Math.PI / 180);
  const qx = p[0] - c[0], qy = p[1] - c[1];
  return [c[0] + qx + tx * qy, c[1] + qy + ty * qx];
}

// The widest XY range any supported scanner has (MFP-3D 90 µm, Cypher 30 µm), with
// margin: the bound a typed position or size uses before the instrument model loads.
export const FALLBACK_XY_RANGE_UM = 1000;
// Whether a typed Select box value is one to send: finite, a position within
// +/- the scanner range, a size above zero and no larger than the range.
export function boxFieldOk(field, v, rangeUm) {
  const r = rangeUm > 0 ? rangeUm : FALLBACK_XY_RANGE_UM;
  if (!Number.isFinite(v)) return false;
  if (field === "x" || field === "y") return Math.abs(v) <= r;
  return v > 0 && v <= r;
}

// Snap-to-bind DEFAULT radius (µm), DOM-free: half the pad width, floored at
// 0.5 µm (EDIT_TOOLS §4). The live call site does NOT use this -- it sizes the
// radius to a screen-pixel target (see ec-tab-edit `snapCandidate`) so snapping
// works at any zoom -- but portSnapHit keeps it as a safety fallback when a
// caller omits the radius, and it is still unit-tested here.
export function snapRadiusUm(widthUm) {
  return Math.max((widthUm || 0) / 2, 0.5);
}

// Snap-to-bind decision (EDIT_TOOLS §4), pure geometry so it unit-tests under
// node. PURE PROXIMITY -- matching the close-path node snap (Slice 2): a wire
// endpoint at (ex,ey) binds to a port whenever it lands within the interaction
// radius of the port centre, regardless of approach angle. There is NO
// directional gate -- the orientation check was found to keep rejecting a natural
// drag toward the port, so it was dropped in favour of the intuitive
// get-close-and-it-snaps behaviour.
//   radius: |endpoint - port.position_um| <= radiusUm. The radius is passed IN
//     (µm) by the caller, which sizes it to ~14 screen px so the endpoint snaps
//     when it lands near the port at ANY zoom; when omitted it falls back to
//     snapRadiusUm(port.width_um). This function stays DOM-free either way.
// Returns true/false; the caller records the binding.
export function portSnapHit(port, ex, ey, radiusUm) {
  const pos = port && port.position_um;
  if (!pos) return false;
  const r = (radiusUm == null) ? snapRadiusUm(port.width_um) : radiusUm;
  return Math.hypot(pos[0] - ex, pos[1] - ey) <= r;
}

// Robust display range [lo, hi] spanning the loPct..hiPct percentiles of the
// FINITE values in `values` (NaN / null / +-Infinity ignored -- a scan frame has
// NaN gaps). Used to map a scan channel to greyscale from a percentile clip
// instead of raw min->max, so the error channel's edge spikes / dropouts don't
// blow the range out (docs/TECH_DEBT.md #1). loPct/hiPct are PERCENTS (e.g. 2,
// 98). Percentile method is NEAREST-RANK over the ascending finite values: index
// = clamp(round(p/100 * (n-1)), 0, n-1). Returns null when there are no finite
// values (caller picks a safe fallback); when lo == hi (a flat frame) it widens
// by +-epsilon so a caller never divides by zero or renders all-black.
export function percentileRange(values, loPct, hiPct) {
  const fin = [];
  for (const v of values)
    if (v != null && Number.isFinite(v)) fin.push(v);
  if (fin.length === 0) return null;
  fin.sort((a, b) => a - b);
  const n = fin.length;
  const at = (pct) => fin[Math.max(0, Math.min(n - 1, Math.round((pct / 100) * (n - 1))))];
  let lo = at(loPct), hi = at(hiPct);
  if (!(hi > lo)) { const e = Math.abs(lo) * 1e-6 || 1e-9; lo -= e; hi += e; }
  return [lo, hi];
}

// A padded micron view box {x0,y0,w,h} enclosing every object's points_um -- the
// "zoom to all" fit. Falls back to a 10x10 box when there is nothing to frame.
export function boundsOfObjects(objects) {
  let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
  for (const o of objects)
    for (const [x, y] of o.points_um) {
      if (x < minx) minx = x; if (y < miny) miny = y;
      if (x > maxx) maxx = x; if (y > maxy) maxy = y;
    }
  if (!isFinite(minx)) return { x0: 0, y0: 0, w: 10, h: 10 };
  const padx = (maxx - minx) * 0.08 + 0.5, pady = (maxy - miny) * 0.08 + 0.5;
  return { x0: minx - padx, y0: miny - pady, w: (maxx - minx) + 2 * padx, h: (maxy - miny) + 2 * pady };
}

// Scale a view box by `f` about the micron point (ux, uy). fitTransform's scale is
// min(px/w, py/h) and its margin is (px - s*w)/2, so scaling w and h together
// leaves the pixel that (ux, uy) maps to unchanged: the point under the pointer
// stays put.
export function zoomViewAbout(view, ux, uy, f) {
  return { x0: ux - (ux - view.x0) * f, y0: uy - (uy - view.y0) * f,
           w: view.w * f, h: view.h * f };
}

// Limit a zoom factor so the view's larger side stays within [minUm, maxUm]. A
// view already outside the range may still move toward it but never further out.
export function clampZoomFactor(view, f, minUm, maxUm) {
  const size = Math.max(view.w, view.h);
  if (!(size > 0) || !(f > 0)) return 1;
  if (f < 1 && size * f < minUm) return Math.min(1, minUm / size);
  if (f > 1 && size * f > maxUm) return Math.max(1, maxUm / size);
  return f;
}

// The smallest box holding every given box {x0,y0,w,h}, grown on each side by
// marginFrac of its larger side. null when no box is given.
export function unionBox(boxes, marginFrac) {
  let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
  for (const b of boxes) {
    if (!b) continue;
    minx = Math.min(minx, b.x0); miny = Math.min(miny, b.y0);
    maxx = Math.max(maxx, b.x0 + b.w); maxy = Math.max(maxy, b.y0 + b.h);
  }
  if (!isFinite(minx)) return null;
  const m = Math.max(maxx - minx, maxy - miny) * marginFrac;
  return { x0: minx - m, y0: miny - m, w: maxx - minx + 2 * m, h: maxy - miny + 2 * m };
}

// Shift a view so it stays inside `bounds`: on an axis where the view is narrower
// than the bounds it may not cross their edges; where it is as wide or wider it is
// centred on them. Size is unchanged.
export function clampViewToBounds(view, bounds) {
  const axis = (v0, vw, b0, bw) =>
    (vw >= bw ? b0 + (bw - vw) / 2 : Math.min(Math.max(v0, b0), b0 + bw - vw));
  return { x0: axis(view.x0, view.w, bounds.x0, bounds.w),
           y0: axis(view.y0, view.h, bounds.y0, bounds.h), w: view.w, h: view.h };
}

// Wheel delta -> zoom factor, exponential in the delta so a touchpad pinch (many
// small deltas) and a mouse notch (one large delta) both zoom in proportion.
// deltaMode 1 (lines) and 2 (pages) are converted to pixels first.
export function wheelZoomFactor(deltaY, deltaMode, perPx) {
  const px = deltaY * (deltaMode === 1 ? 16 : deltaMode === 2 ? 800 : 1);
  return Math.exp(px * perPx);
}
