// Lengths of design objects, in microns (Layer 1, DOM-free, imported under node by
// tests/ec_modules.mjs). Each length is taken from the geometry the canvas draws:
// a Bezier flattened by ec-bezier's bezFlatten, a chiral wire from its analytic
// profile (ec-main drawChiralWire samples the same chiralSampler), else the
// served polyline.

import { bezFlatten } from "./ec-bezier.js";

// The flatten tolerance the node tool and the service both use (µm).
export const FLATTEN_TOL_UM = 0.004;
// Chiral samples per modulation period for a length: the canvas's own cap, so the
// length does not depend on the zoom the wire happens to be drawn at.
const CHIRAL_PER_PERIOD = 120;

// Sum of segment lengths. A closed outline always adds its closing edge, which
// is zero-length when the polyline already ends on its first point.
export function polylineLength(pts, closed) {
  if (!pts || pts.length < 2) return 0;
  let L = 0;
  for (let i = 1; i < pts.length; i++)
    L += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
  if (closed) {
    const a = pts[0], b = pts[pts.length - 1];
    L += Math.hypot(a[0] - b[0], a[1] - b[1]);
  }
  return L;
}

// Length of a Bezier {nodes, closed}; bezFlatten already ends a closed path on
// its first node, so no closing edge is added.
export function bezLength(bez, tol = FLATTEN_TOL_UM) {
  if (!bez || !bez.nodes || bez.nodes.length < 2) return 0;
  return polylineLength(bezFlatten(bez, tol), false);
}

// The analytic chiral wire as f in [0, 1] -> {x, y, v}: the smooth axis, normal
// and amplitude interpolated linearly, the sine evaluated exactly. Null when the
// profile lacks the analytic data (the canvas then draws the polyline).
export function chiralSampler(P) {
  if (!P) return null;
  const axis = P.axis_um, norm = P.normal, amp = P.amp_um;
  const m = axis && axis.length;
  if (!m || m < 2 || !norm || !amp || norm.length !== m || amp.length !== m) return null;
  const twoPiN = 2 * Math.PI * (P.periods || 0);
  return (f) => {
    const u = Math.min(1, Math.max(0, f)) * (m - 1);
    const k = Math.min(m - 2, Math.floor(u));
    const t = u - k;
    const ax = axis[k][0] + t * (axis[k + 1][0] - axis[k][0]);
    const ay = axis[k][1] + t * (axis[k + 1][1] - axis[k][1]);
    let nx = norm[k][0] + t * (norm[k + 1][0] - norm[k][0]);
    let ny = norm[k][1] + t * (norm[k + 1][1] - norm[k][1]);
    const nn = Math.hypot(nx, ny) || 1; nx /= nn; ny /= nn;
    const a = amp[k] + t * (amp[k + 1] - amp[k]);
    const th = twoPiN * f, sn = Math.sin(th);
    return { x: ax + nx * a * sn, y: ay + ny * a * sn,
             v: P.v0 + P.v_k * Math.sin(th + P.phase_rad) };
  };
}

export function chiralLength(P) {
  const at = chiralSampler(P);
  if (!at) return null;
  const N = Math.max(2, Math.ceil(CHIRAL_PER_PERIOD * (P.periods || 1)));
  let L = 0, prev = at(0);
  for (let i = 1; i <= N; i++) {
    const cur = at(i / N);
    L += Math.hypot(cur.x - prev.x, cur.y - prev.y);
    prev = cur;
  }
  return L;
}

// Filled kinds are closed outlines; a path is an open wire.
export function isClosedKind(o) {
  return !!o && (o.kind === "shape" || o.kind === "electrode");
}

// The drawn length of a cockpit object (µm). A modulated object's stored Bezier is
// its base, not what is drawn, so only an unmodulated one is measured from it.
export function objectLength(o) {
  if (!o) return 0;
  const closed = isClosedKind(o);
  if (!closed && o.profile && o.profile.kind === "chiral") {
    const L = chiralLength(o.profile);
    if (L != null) return L;
  }
  if (!o.profile && o.bezier && o.bezier.nodes && o.bezier.nodes.length >= 2)
    return bezLength(o.bezier);
  return polylineLength(o.points_um, closed);
}
