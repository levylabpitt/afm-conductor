// Pure layout and mapping rules for the render pane: how the telemetry readout steps
// down as space shrinks, how a scan line maps onto the image's on-screen x range, and
// the labels of the z scale bar. No DOM access, so node unit-tests them.

// One row per telemetry channel the `hw_telemetry` frame can carry, in display order.
// A channel is drawn whenever every key it reads is on the frame, as a dash while the
// value is null: on hardware a failed read arrives as null, and a safety channel that
// is broken must still be seen. `height_nm` has no row: ZSensor is not a polled
// channel, so that key is always null. `top` reads the value from the telemetry
// message itself instead of the block: the service blanks amplitude there outside AC.
export const TELEMETRY_CHANNELS = [
  { id: "xy", keys: ["x_um", "y_um"], label: "x, y", unit: "µm", digits: 3 },
  { id: "defl", keys: ["defl_v"], label: "deflection", unit: "V", digits: 4 },
  { id: "amp", keys: ["amp_v"], top: "amp_v", label: "amplitude", unit: "V", digits: 3 },
  { id: "zdrive", keys: ["zdrive_v"], label: "Z drive", unit: "V", digits: 2 },
  { id: "sum", keys: ["sum_v"], label: "SUM", unit: "V", digits: 3 },
  { id: "row", keys: ["row"], label: "row", unit: "", digits: 0 },
];

export const isNum = (v) => typeof v === "number" && Number.isFinite(v);

// The values a channel shows from one telemetry message `m` (its hw_telemetry block).
export function channelValues(ch, m) {
  const block = (m && m.hw_telemetry) || {};
  if (ch.top) return [m ? m[ch.top] : undefined];
  return ch.keys.map((k) => block[k]);
}

// A channel's number text: a dash unless every value it reads is a number.
export function valueText(ch, vals) {
  if (!vals.every(isNum)) return "—";
  return vals.map((v) => v.toFixed(ch.digits)).join(", ");
}

// The channels to draw: every table channel whose keys are all on the frame.
export function visibleChannels(m) {
  const block = (m && m.hw_telemetry) || {};
  return TELEMETRY_CHANNELS.filter((ch) => ch.keys.every((k) => k in block));
}

// Bar position 0..1 of `v` inside [lo, hi], or null when either is unknown.
export function barFraction(v, range) {
  if (!isNum(v) || !range || !isNum(range[0]) || !isNum(range[1]) || range[1] <= range[0])
    return null;
  return Math.max(0, Math.min(1, (v - range[0]) / (range[1] - range[0])));
}

// Append to a bounded history buffer, dropping the oldest value.
export function pushHistory(buf, v, max = 120) {
  buf.push(isNum(v) ? v : NaN);
  while (buf.length > max) buf.shift();
  return buf;
}

// Telemetry step-down. `spare` is the render body's width left after the square image
// and the z scale bar; `height` is the column's height; `n` the channel count.
// Returns the arrangement: "tab" (folded to a thin vertical tab), "column" beside the
// image with optional bars and history graphs, or "grid" (label + number only, under
// the image). A column never scrolls: bars appear only when every channel fits with one.
// `half` (the half-screen arrangement, where the image takes the full width) always
// gives the grid unless folded. The decision reads only the space given (the body's
// box and fixed per-row heights), never the readout's own content height. `prev`, the
// last result, adds hysteresis: bars and history graphs that are shown stay until the
// space is HYST_PX under their threshold, so a box at a boundary cannot toggle.
export const TELE = {
  TAB_PX: 26, COL_MIN_PX: 170, COL_MAX_PX: 320, SPARK_MIN_PX: 240,
  ROW_PX: 54, BAR_PX: 18, HEAD_PX: 30, HYST_PX: 12,
};
export function telemetryLayout({ spare, height, n, collapsed, half = false, prev = null }) {
  if (collapsed) return { mode: "tab", width: TELE.TAB_PX, bars: false, spark: false };
  if (half || spare < TELE.COL_MIN_PX || height < TELE.HEAD_PX + n * TELE.ROW_PX)
    return { mode: "grid", width: 0, bars: false, spark: false };
  const width = Math.round(Math.min(spare, TELE.COL_MAX_PX));
  const was = prev && prev.mode === "column" ? prev : { bars: false, spark: false };
  const at = (v, need, on) => v >= need - (on ? TELE.HYST_PX : 0);
  const bars = at(height, TELE.HEAD_PX + n * (TELE.ROW_PX + TELE.BAR_PX), was.bars);
  return { mode: "column", width, bars, spark: at(width, TELE.SPARK_MIN_PX, was.spark) };
}

// Half-screen image size: the side of the square image when it takes the full
// `width` and shares `avail` px of height with the scope under it. The scope keeps
// SCOPE_MIN_PX and takes the rest. The image outranks the scope and never drops
// under IMAGE_MIN_PX, so at very short heights the scope gets less than SCOPE_MIN_PX.
export const HALF = { SCOPE_MIN_PX: 120, IMAGE_MIN_PX: 160 };
export function halfImageSide(width, avail) {
  const side = Math.min(width, Math.max(HALF.IMAGE_MIN_PX, avail - HALF.SCOPE_MIN_PX));
  return Math.max(0, Math.round(side));
}

// Scope x mapping. The image's scan frame spans [imgLeft, imgRight] px in the scope's
// own coordinates; the plot is clipped to [clipLeft, clipRight] (the image area's
// visible width). Pixel i of an n-pixel line sits at the centre of its column, as it
// does in the image. Returns x(i) and the first/last pixel inside the clip, or null
// when no pixel is visible.
export function scopeXMap(imgLeft, imgRight, n, clipLeft, clipRight) {
  if (!(n > 0) || !(imgRight > imgLeft)) return null;
  const dx = (imgRight - imgLeft) / n;
  const x = (i) => imgLeft + (i + 0.5) * dx;
  const i0 = Math.max(0, Math.ceil((clipLeft - imgLeft) / dx - 0.5));
  const i1 = Math.min(n - 1, Math.floor((clipRight - imgLeft) / dx - 0.5));
  if (i1 < i0) return null;
  return { x, i0, i1, dx };
}

// [lo, hi] of the finite samples of `arrays` between indices i0..i1, or null.
export function rangeOver(arrays, i0, i1) {
  let lo = Infinity, hi = -Infinity;
  for (const a of arrays) {
    if (!a) continue;
    for (let i = Math.max(0, i0); i <= Math.min(a.length - 1, i1); i++) {
      const v = a[i];
      if (isNum(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
  }
  return Number.isFinite(lo) ? [lo, hi] : null;
}

// The z scale bar's three labels: min, mid and max of the image's grey range.
export function zScaleLabels(lo, hi) {
  if (!isNum(lo) || !isNum(hi)) return [];
  const d = Math.abs(hi - lo) < 10 ? 2 : 1;
  const f = (v) => `${v >= 0 ? "+" : ""}${v.toFixed(d)}`;
  return [{ frac: 0, text: f(lo) }, { frac: 0.5, text: f((lo + hi) / 2) },
          { frac: 1, text: f(hi) }];
}

// Telemetry older than this (s since hw_telemetry.t last advanced) is shown stale.
export const TELEMETRY_STALE_S = 5;
// Whether the readout is stale and its age text: stale when the link is closed or the
// frame time has not advanced for TELEMETRY_STALE_S. `ageS` is null before any frame.
export function telemetryStaleness(linkOpen, ageS) {
  const old = ageS != null && ageS > TELEMETRY_STALE_S;
  if (!linkOpen || old) {
    return { stale: true, text: ageS == null ? "stale" : `stale ${ageS.toFixed(1)} s`,
             why: linkOpen ? "no new telemetry from the service" : "websocket closed" };
  }
  return { stale: false, text: "", why: "" };
}

// A z bar end dragged by `frac` of the bar's length (positive toward the high end)
// moves that end by the same fraction of the range the drag started from. The ends
// never cross: the range keeps at least 1 % of its span. A flat image's range is a
// near-zero span, so the span used is at least Z_DRAG_MIN_SPAN_FRAC of the values'
// magnitude (and Z_DRAG_MIN_SPAN), so its ends can still be dragged apart.
export const Z_DRAG_MIN_SPAN_FRAC = 0.01;
export const Z_DRAG_MIN_SPAN = 1e-3;
export function dragZRange(start, end, frac) {
  const span = Math.max(start.hi - start.lo,
    Z_DRAG_MIN_SPAN_FRAC * Math.max(Math.abs(start.lo), Math.abs(start.hi)), Z_DRAG_MIN_SPAN);
  const gap = span * 0.01;
  if (end === "hi") return { lo: start.lo, hi: Math.max(start.lo + gap, start.hi + frac * span) };
  return { lo: Math.min(start.hi - gap, start.lo + frac * span), hi: start.hi };
}
