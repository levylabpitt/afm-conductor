"""Litho pattern compiler: shapes + properties -> toolpath waves.

Ports the LabVIEW tooling semantics at v0 scope: polyline paths and raster
rectangles, arc-length-equal interpolation, constant or linear-ramp voltage,
per-object speed. The output of compile_object is exactly what the executor
arms into the controller: X/Y waves in LVDT volts and a voltage wave, all the
same length, plus the tick interpolation factor floor(dist/speed * -CLOCK).
Every stroke is a multiple of 32 points and ends at 0 V at its last position
(`fired_points`), so the out-wave is as long as the capture that times it.

`compile_object` is the unit the transit planner and the canvas think in:
one object, one continuous tip path. `compile_pattern` is the unit the
INSTRUMENT thinks in: the whole pattern as one long wave per axis with NaN
marking pen-up. Both exist because both are true, and neither is a rewrite
of the other. See `nan_waves` below for the wire format.

WHY THE 8,192-POINT SPLIT IS GONE (2026-09-11). `compile_pattern` used to
mirror the vendor's `CheckLithoWaves`, cutting every stroke at 8,192 points
(`MAX_SEGMENT_POINTS`). That was wrong on this stack, for two reasons that
only showed up on the instrument:

  * it is not an ARC limit. The only documented out-wave cap is the DSP's
    per-wave one (`litho/chunker.MAX_OUTWAVE_POINTS`, 87,000), more than ten
    times larger;
  * the vendor does not LIFT at its own seams -- `Lithography.ipf`'s
    `LithoRamp` (~:1705-1778) moves between segments as a pure XY setpoint
    ramp with the Z feedback engaged, touching Z only `if (Snapping)`.

Our write path does lift: each segment gets its own lift / rate-limited move
/ set-down / arm / fire / lift. So copying the cut put a ~1.8 s
tip-off-surface, bias-zero gap in the middle of every line longer than 8,192
points -- the bug reported on 2026-09-11. The split is now opt-in
(`compile_pattern(max_segment_points=...)`); a stroke over the DSP cap is
chunked by the DRIVER instead, as a chain that keeps the tip down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .frame import LithoFrame

# `CheckLithoWaves` chops any segment longer than this, by inserting a pen-up NaN
# mid-stroke. NOT AN ARC LIMIT and NOT A LIFT:
#
#   * the only documented out-wave cap is the DSP's per-wave one
#     (`litho.chunker.MAX_OUTWAVE_POINTS`, 87,000), an order of magnitude above this;
#   * the vendor does NOT lift at its own seams -- `Lithography.ipf`'s `LithoRamp`
#     (~:1705-1778) moves between segments as a pure XY setpoint ramp with the Z
#     feedback still engaged, and touches Z only `if (Snapping)`.
#
# So mirroring the 8192 cut bought nothing and cost everything: OUR per-segment write
# cycle (lift / rate-limited move / set-down / arm / fire / lift) turned each cut into
# a ~1.8 s tip-off-surface, bias-zero seam in the middle of a line the user drew as
# one stroke (2026-09-11). `compile_pattern` therefore does NOT split by default;
# whole strokes go to the write path and only the DSP cap chunks them, through the
# seamless LITHOCHAIN continuation that keeps the tip down and the bias live.
# `split_long_segments` stays, opt-in, for anything that genuinely wants the vendor's
# segmentation.
MAX_SEGMENT_POINTS = 8192

# Default arc-length step a toolpath is resampled to (microns): the spacing the
# executor writes at, so it is a physical write-resolution knob, not a display
# detail. One source of truth for LithoObject's default and every pen that used
# to hardcode 0.02 (the chiral pen still derives its own from feature size).
DEFAULT_INTERP_UM = 0.02


@dataclass
class LithoObject:
    name: str
    kind: str                      # "path" | "rect" | "taper" | "shape"
    points_um: list[tuple[float, float]]   # path vertices; rect corners (2); taper axis (2)
    # "shape": a closed Bezier outline, filled with a serpentine. points_um
    # is its flattened outline; `holes_um` are inner outlines (even-odd, as
    # in Inkscape); `bezier` keeps the exact control points so the shape
    # stays editable and round-trips to SVG without ever being re-fitted.
    holes_um: list[list[tuple[float, float]]] = field(default_factory=list)
    bezier: dict | None = None
    # The base AXIS polyline a chiral (or other) modulation was built from.
    # When a path is made chiral, points_um becomes the modulated toolpath and
    # the plain axis it was drawn/imported as would otherwise be lost; this
    # keeps it so every re-parametrize regenerates from the untouched axis and
    # disabling reverts to it. Server-side only -- not part of to_dict. None
    # when the object carries no modulation. Travels through copy.deepcopy
    # (undo snapshots) and reorder because it is a field on the object.
    axis_um: list[tuple[float, float]] | None = None
    # The analytic description of a modulated wire (axis + wavelength +
    # amplitude + phase + bias), when the pen that made it has one. The
    # canvas renders from THIS; points_um stays the compiled toolpath the
    # executor writes. See pens.chiral_profile.
    profile: dict | None = None
    fill_angle_deg: float | None = None    # None = along the shape's long axis
    # How the region gets covered: "directional", "contour", "hilbert",
    # "radial-out", "radial-in". This is a PROPERTY of the object, resolved
    # into a toolpath by `expand` at compile time, and that is the whole
    # point of it being here: the region pen used to run the fill when the
    # stroke was drawn and store the serpentine, so the document held a
    # toolpath and not a region. Nothing downstream could then union two
    # regions, offset one, edit its nodes, or write it to OASIS as the
    # polygon it is -- because by then there was no polygon, only the
    # thousand lines that had been used to cover it.
    fill_kind: str = "directional"
    fill_centre_um: tuple[float, float] | None = None   # the radial fills
    inset_um: float = 0.0
    outline_pass: bool = False
    voltage: float = 8.0
    voltage_end: float | None = None       # linear ramp if set
    # arbitrary bias profile sampled uniformly over the object's arc length
    # (0..1). Superlattices, diode ratchets and chiral modulation are written
    # with this; it is also the handle an agent turns when exploring.
    v_profile: list[float] | None = None
    speed_um_s: float = 2.0
    interpolation_um: float = DEFAULT_INTERP_UM
    fill_spacing_um: float = 0.15          # rect/taper fills
    width0_um: float = 2.0                 # taper start width (at points[0])
    width1_um: float = 0.2                 # taper end width (at points[1])
    enabled: bool = True
    # An optional endpoint->electrode BINDING recorded when a wire end is snapped
    # onto a marked port (edit-tools snap-to-bind, docs/EDIT_TOOLS.md §4). Shape:
    # {"node": "start"|"end", "port_id": "<id>"} -- a REFERENCE to the port, never
    # a coord copy, so the later connections generator + drift resolve the endpoint
    # through the port's current position. None (absent from to_dict) by default so
    # existing serialization is unchanged. Travels through deepcopy (undo) + reorder.
    end_binding: dict | None = None


@dataclass
class Pattern:
    name: str
    objects: list[LithoObject] = field(default_factory=list)

    def to_dict(self, max_points: int | None = None) -> dict:
        """Serialize for the canvas.

        Geometry and bias are sampled TOGETHER, at the same stride, with the
        bias mapped onto the points by arc length exactly as compile_object
        does when it writes them.

        The bias used to be decimated on its own to a flat 160 samples
        whatever the object's length. For a fine helix that is far below
        Nyquist: a 3 um wire at a 25 nm pitch carries 120 modulation
        periods, and 160 samples over 120 periods (1.33 per period) aliased
        them into 44 apparent ones. The canvas then drew slow colour bands
        along a wire whose bias actually oscillates once per serpentine
        period, locked to the geometry. The wire was right; the picture of
        it was not.
        """
        def sampled(o):
            pts = np.asarray(o.points_um, dtype=np.float64)
            if len(pts) < 2:
                return [[float(a), float(b)] for a, b in pts], None
            # The budget has to come from the CONTENT, not from a round
            # number. A serpentine looks smooth when each period gets
            # enough samples and jagged when it does not, so the only thing
            # that matters is points PER PERIOD.
            #
            # Two earlier attempts got this wrong. A flat 3000 points gave
            # 1.24 samples per period on a long wire, which aliased into a
            # wandering beat. Replacing that with per-bin extremes fixed the
            # amplitude and the period count but joined them with straight
            # lines, turning a smooth sinusoid into a triangle wave: the
            # full geometry has no sharp corners at all, and that version
            # introduced 1547 of them. Faceting is exactly as wrong as
            # aliasing, just less obviously.
            if o.profile:
                # the canvas renders profiled wires analytically, so the
                # polyline is only a coarse fallback and bounds hint
                budget = max_points or MIN_DISPLAY_POINTS
            else:
                osc = count_oscillations(pts)
                need = osc * POINTS_PER_PERIOD if osc else MIN_DISPLAY_POINTS
                budget = max_points or int(np.clip(need, MIN_DISPLAY_POINTS,
                                                   MAX_DISPLAY_POINTS))
            stride = max(1, -(-len(pts) // budget))
            keep = pts[::stride]
            if stride > 1 and not np.array_equal(keep[-1], pts[-1]):
                keep = np.vstack([keep, pts[-1:]])   # never move an endpoint
            geom = [[round(float(a), 4), round(float(b), 4)] for a, b in keep]
            if not o.v_profile:
                return geom, None
            src = np.asarray(o.v_profile, dtype=np.float64)
            seg = np.hypot(*np.diff(pts, axis=0).T)
            s = np.concatenate([[0.0], np.cumsum(seg)])
            total = float(s[-1]) or 1.0
            # bias sampled at the kept points, by arc length, so geometry
            # and bias stay registered whichever selection was used
            idx = _nearest_index(pts, keep)
            v = np.interp(s[idx] / total,
                          np.linspace(0.0, 1.0, len(src)), src)
            return geom, [round(float(x), 3) for x in v]

        objects = []
        for o in self.objects:
            geom, v = sampled(o)
            entry = {
                "name": o.name, "kind": o.kind, "points_um": geom,
                "voltage": o.voltage, "voltage_end": o.voltage_end,
                "v_profile": v,
                "speed_um_s": o.speed_um_s, "enabled": o.enabled,
                "fill_spacing_um": o.fill_spacing_um,
            }
            if o.profile:
                entry["profile"] = o.profile
            # Carry the control-point bezier so the cockpit can tell a rectangle
            # from a generic region (label + node-tool rect exclusion, EDIT_TOOLS
            # §3) without a second round-trip. Absent when the object has none.
            if getattr(o, "bezier", None):
                entry["bezier"] = o.bezier
            # Carry an endpoint->electrode binding when present so the snap survives
            # a round-trip (EDIT_TOOLS §4). Absent when the object has none, so a
            # design with no bindings serializes exactly as before.
            if getattr(o, "end_binding", None):
                entry["end_binding"] = o.end_binding
            objects.append(entry)
        return {"name": self.name, "objects": objects}


# Samples per serpentine period the canvas needs before a curve stops
# looking like a polygon. Measured as chord error against the true curve,
# as a fraction of the amplitude: 6 gives 14%, 12 gives 4.2%, 24 gives
# 1.1%. Twenty-four is the first value that reads as a smooth line.
POINTS_PER_PERIOD = 24
MIN_DISPLAY_POINTS = 3000
MAX_DISPLAY_POINTS = 30000


def count_oscillations(pts: np.ndarray) -> int:
    """How many times the path swings from side to side.

    Counted from sign changes of the curvature, which is two per period of
    a serpentine and zero on a straight line. This is what decides how many
    points the display needs: a wire is smooth if each period gets enough
    samples, and the period count is a property of the geometry, not of how
    long the wire happens to be.
    """
    d = np.diff(pts, axis=0)
    if len(d) < 3:
        return 0
    cross = d[:-1, 0] * d[1:, 1] - d[:-1, 1] * d[1:, 0]
    sgn = np.sign(cross)
    sgn = sgn[sgn != 0]
    if len(sgn) < 2:
        return 0
    return int((np.diff(sgn) != 0).sum() // 2)


def _nearest_index(pts: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Index in `pts` of each kept point (they are drawn from it)."""
    from scipy.spatial import cKDTree
    return cKDTree(pts).query(keep)[1]


def _geom_points(pts: np.ndarray, spacing: float) -> int:
    """How many points the uniform arc-length rule gives a polyline at `spacing`:
    round(length / spacing) + 1, at least 2; 1 for a zero-length polyline."""
    seg = np.diff(pts, axis=0)
    total = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
    if total <= 0:
        return 1
    return max(2, int(round(total / spacing)) + 1)


def _sample_polyline(pts: np.ndarray, n: int) -> np.ndarray:
    """`n` points equally spaced by arc length along a polyline (um in, um out),
    first and last exact. A zero-length polyline gives `n` copies of its first point."""
    seg = np.diff(pts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(s[-1])
    if total <= 0:
        return np.repeat(pts[:1], int(n), axis=0)
    si = np.linspace(0.0, total, int(n))
    x = np.interp(si, s, pts[:, 0])
    y = np.interp(si, s, pts[:, 1])
    return np.column_stack([x, y])


def _interp_polyline(pts: np.ndarray, spacing: float) -> np.ndarray:
    """Points equally spaced by arc length along a polyline (um in, um out)."""
    return _sample_polyline(pts, _geom_points(pts, spacing))


def _taper_raster(p0, p1, w0: float, w1: float, spacing: float) -> np.ndarray:
    """Tapered lead fill: zigzag sweep between two converging edges.

    The taper runs from p0 (width w0, the contact side) to p1 (width w1, the
    device side): the standard lead-in geometry connecting an interface
    contact to a nanoscale wire.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    axis = p1 - p0
    length = float(np.hypot(*axis))
    if length <= 0:
        return np.array([p0])
    u = axis / length
    nvec = np.array([-u[1], u[0]])
    # strokes run ALONG the lead (the lab always writes the long direction);
    # each stroke spans wide-end offset -> narrow-end offset, so the tip end
    # closes in a natural zigzag with no square cap
    n_lines = max(2, int(round(w0 / spacing)) + 1)
    pts = []
    for i in range(n_lines):
        t = i / (n_lines - 1)          # 0..1 across the width
        off0 = (t - 0.5) * w0
        off1 = (t - 0.5) * w1
        a = p0 + nvec * off0
        b = p1 + nvec * off1
        pts += [tuple(a), tuple(b)] if i % 2 == 0 else [tuple(b), tuple(a)]
    return np.asarray(pts)


def _rect_raster(c0, c1, spacing: float) -> np.ndarray:
    """Boustrophedon fill of the rectangle with corners c0, c1 (um)."""
    x0, y0 = min(c0[0], c1[0]), min(c0[1], c1[1])
    x1, y1 = max(c0[0], c1[0]), max(c0[1], c1[1])
    n_lines = max(2, int(round((y1 - y0) / spacing)) + 1)
    ys = np.linspace(y0, y1, n_lines)
    pts = []
    for i, y in enumerate(ys):
        if i % 2 == 0:
            pts += [(x0, y), (x1, y)]
        else:
            pts += [(x1, y), (x0, y)]
    return np.asarray(pts)


def ar_get_deci(sample_rate: float, clock_hz: int = 50_000) -> int:
    """The instrument's decimation rule (ARGetDeci in HardWare.ipf).

    Integer ticks per sample against the 50 kHz master clock, which is why
    every rate on the machine is quantised to 50000/k.
    """
    if sample_rate <= 0:
        return 1
    return max(int(round(clock_hz / sample_rate)), 1)


def capture_points(n: int, block: int = 32) -> int:
    """Capture buffers are padded up to a multiple of 32 points.

    Lithography.ipf allocates `inputPoints + (32 - mod(inputPoints,32))` and
    remembers the padding as LithoExtra to trim afterwards.
    """
    n = max(int(n), 1)
    return n + (block - n % block) % block


def fired_points(n_geom: int) -> int:
    """Points a stroke is FIRED with, from the count the arc-length rule gives its
    geometry: N = 32 * ceil((n_geom + 1) / 32).

    The capture in-wave is a multiple of 32 (td_xSetInWavePair) and the controller
    holds the last out-wave sample until the capture fills (rig captures, Cypher
    2026-10-04), so the out-wave has the capture's length and its last sample is 0 V.
    The + 1 is that 0 V sample: N - 1 path points, then the last XY repeated.
    """
    return capture_points(max(int(n_geom), 1) + 1)


def trace_indices(path_points: int, stride: int = 4) -> np.ndarray:
    """Indices of a fired stroke's PATH points at `stride`, always ending at the last
    path point. The 0 V end sample is left out: a display trace of what was written
    ends at the last biased point, with its bias."""
    n = max(int(path_points), 0)
    idx = np.arange(0, n, max(int(stride), 1))
    if n and idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def _end_at_zero(path: np.ndarray, v: np.ndarray, n: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Extend path points and their bias to `n` fired points: the last XY repeated,
    at 0 V."""
    extra = int(n) - len(path)
    return (np.vstack([path, np.repeat(path[-1:], extra, axis=0)]),
            np.concatenate([v, np.zeros(extra)]))


def expand(obj: LithoObject) -> list[LithoObject]:
    """Resolve an object into things that compile to a single tip path.

    Only shapes need this. A filled outline can break into several
    disconnected regions (a shape with a hole, or two islands), and those
    must stay separate objects: joining them would drag the biased tip
    across sample that is not meant to be written. Keeping them separate
    also means the transit planner handles the gaps, with a lift or a
    zero-bias slew, like any other gap between strokes.

    An `electrode` is filled the same way: it is a closed outline whose
    Bezier is a funnel, imported from SVG or drawn closed with the pen,
    and it takes the same fill machinery below -- the funnel branch recognises
    its two side curves and sweeps between them.
    """
    if obj.kind not in ("shape", "electrode"):
        return [obj]
    from .fill import (fill_contour, fill_funnel_from_curves, fill_hilbert,
                       fill_polygons, fill_radial, fill_zigzag)
    strokes = None
    polys = [np.asarray(obj.points_um, dtype=np.float64)]
    polys += [np.asarray(h, dtype=np.float64) for h in (obj.holes_um or [])]
    angle = (None if obj.fill_angle_deg is None
             else math.radians(obj.fill_angle_deg))
    # Imported outlines carry their Bezier, so we recognise the two shapes the
    # archive is built from and fill each the way it was drawn:
    #   funnel    -- flat base + two curved sides to an apex -> the 2013 family
    #                (blend the two boundary curves' control points), one stroke;
    #   rectangle -- raster parallel to its OWN edges (not the unstable long
    #                axis of a near-square, which comes out diagonal), as one
    #                continuous WRITTEN zig zag: every row runs the same
    #                direction (the zig) and the tip keeps writing on the long
    #                diagonal back to the next row's start (the zag). Not a
    #                both-ways serpentine (rows do not reverse) and not a comb
    #                (the returns are written, bias stays on across the turn) --
    #                the lab's own raster. See docs r0.
    zigzag = False
    if strokes is None and getattr(obj, "bezier", None):
        from .bezier import BezierPath
        bp = BezierPath.from_dict(obj.bezier)
        funnel = bp.as_funnel()
        if funnel:
            strokes = fill_funnel_from_curves(funnel[0], funnel[1],
                                              obj.fill_spacing_um)
        elif angle is None:
            rect = bp.as_rect()
            if rect is not None:
                strokes = fill_zigzag(polys, obj.fill_spacing_um, rect,
                                      obj.inset_um)
                zigzag = True
    if strokes is None:
        # The same four fill strategies the region pen has always offered,
        # run here instead of when the stroke was drawn. Identical
        # parameters in, identical toolpath out: `test_region_object.py`
        # holds the two against each other point for point, because a
        # change of representation that also changed what gets written
        # would be a different change wearing this one's clothes.
        kind = obj.fill_kind or "directional"
        match kind:
            case "contour":
                strokes = fill_contour(polys, obj.fill_spacing_um,
                                       inset_um=obj.inset_um, outline=True)
            case "hilbert":
                strokes = fill_hilbert(polys, obj.fill_spacing_um,
                                       inset_um=obj.inset_um, outline=False)
            case "radial-out" | "radial-in":
                strokes = fill_radial(polys, obj.fill_spacing_um,
                                      centre=(tuple(obj.fill_centre_um)
                                              if obj.fill_centre_um else None),
                                      inward=(kind == "radial-in"))
            case _:                          # "directional" and any unknown kind
                strokes = fill_polygons(polys, obj.fill_spacing_um, angle,
                                        obj.inset_um, obj.outline_pass)
    if zigzag:
        # A zig zag, not a comb: every row writes the SAME direction (the zig),
        # and the tip keeps writing on the long diagonal back to the next row's
        # start (the zag) -- it goes back and forth with the bias on the whole
        # time, which is what the lab writes. Join the rows into ONE continuous
        # stroke so the returns are part of the written pass rather than
        # invisible teleports between separate lines.
        rows = [np.asarray(st, dtype=np.float64) for st in strokes
                if len(st) >= 2]
        strokes = [np.concatenate(rows)] if rows else []
    out = []
    for i, st in enumerate(strokes):
        st = np.asarray(st, dtype=np.float64)
        if len(st) < 2:
            continue
        out.append(LithoObject(
            name=obj.name if len(strokes) == 1 else f"{obj.name}#{i + 1}",
            kind="path",
            points_um=[(float(a), float(b)) for a, b in st],
            voltage=obj.voltage, voltage_end=obj.voltage_end,
            speed_um_s=obj.speed_um_s, interpolation_um=obj.interpolation_um,
            enabled=obj.enabled))
    return out


def time_segment(path: np.ndarray, speed_um_s: float,
                 clock_hz: int = 50_000, round_up: bool = False) -> dict:
    """The instrument's timing rule for one segment (Lithography.ipf).

    Not "spread the requested duration over the path" but "never exceed a
    velocity ceiling": the tick rate is set by the LARGEST single step in
    the segment, and the whole segment then runs at that one rate.

        TotalTime     = max(step) / MaxVelocity     seconds for one point
        Interpolation = ARGetDeci(1/TotalTime) = max(round(50000*TotalTime), 1)
        PPS           = 50000 / Interpolation

    Two consequences worth having in the twin, because they are visible on
    the real machine: the achieved speed is quantised (the interpolation
    factor is an integer, so you ask for one speed and get a nearby one, and
    the software writes the achieved value back into the panel), and a path
    containing one long step is slowed down everywhere, not just across that
    step.

    This lives on its own because the 8192-point split re-times each piece.
    The vendor splits first and computes timing per segment afterwards, so a
    piece that does not contain the worst step runs FASTER than the stroke
    it was cut from. Sharing the rule is the only way the two paths cannot
    drift apart.

    `round_up` is for lithography STROKES: the tick is the ceiling instead of
    ARGetDeci's nearest integer, so a stroke never runs faster than requested (a
    32-point stroke of 0.02 um at 10 um/s: 3.33 ticks -> 4, not 3). Transits and
    scan-side moves keep the vendor's rounding.
    """
    seg = np.diff(path, axis=0)
    step = np.hypot(seg[:, 0], seg[:, 1])
    n = len(path)
    max_step_um = float(step.max()) if len(step) else 0.0
    if max_step_um <= 0:
        ticks_per_point = 1
    elif round_up and speed_um_s > 0:
        # 1e-9 under: a value that is an integer up to float error is not bumped
        ticks_per_point = max(
            math.ceil(clock_hz * max_step_um / speed_um_s * (1.0 - 1e-9)), 1)
    else:
        ticks_per_point = ar_get_deci(speed_um_s / max_step_um, clock_hz)
    pps = clock_hz / ticks_per_point
    return {
        "ticks_per_point": ticks_per_point,
        "points_per_second": pps,
        "duration_s": (n - 1) / pps,
        "length_um": float(step.sum()),
        "speed_achieved_um_s": max_step_um * pps,
    }


def compile_object(obj: LithoObject, x_sens: float, y_sens: float,
                   x_offset: float = 0.0, y_offset: float = 0.0,
                   clock_hz: int = 50_000,
                   frame: LithoFrame | None = None) -> dict:
    """Compile one object to the waves that are FIRED.

    Returns {x_v, y_v, v, path_um, path_points, ticks_per_point, duration_s, ...}.

    The stroke has N = `fired_points(n_geom)` points, n_geom being the count the
    uniform arc-length rule gives the object at `interpolation_um`: N - 1 path points
    sampled by that same rule from the object's own vertices, carrying its bias, then
    the last XY repeated at 0 V.

    `frame` is the litho frame: the pattern's coordinates are mapped through
    it into scanner coordinates before anything else happens, so `path_um`
    and the waves are all in lab um. The default identity frame leaves a
    centred design exactly where it was.
    """
    pts = np.asarray(obj.points_um, dtype=np.float64)
    if obj.kind == "rect":
        pts = _rect_raster(pts[0], pts[1], obj.fill_spacing_um)
    elif obj.kind == "taper":
        pts = _taper_raster(pts[0], pts[1], obj.width0_um, obj.width1_um,
                            obj.fill_spacing_um)
    n = fired_points(_geom_points(pts, obj.interpolation_um))
    n_path = n - 1
    path = _sample_polyline(pts, n_path)
    if frame is not None:
        # rigid, so the interpolation spacing and therefore the timing are
        # untouched: only where the path sits changes
        path = frame.path_to_lab(path)

    if obj.v_profile:
        prof = np.asarray(obj.v_profile, dtype=np.float64)
        v = np.interp(np.linspace(0.0, 1.0, n_path),
                      np.linspace(0.0, 1.0, len(prof)), prof)
    else:
        v0 = obj.voltage
        v1 = obj.voltage if obj.voltage_end is None else obj.voltage_end
        v = np.linspace(v0, v1, n_path)
    path, v = _end_at_zero(path, v, n)

    t = time_segment(path, obj.speed_um_s, clock_hz, round_up=True)
    ticks_per_point = t["ticks_per_point"]
    pps = t["points_per_second"]
    duration_s = t["duration_s"]
    length_um = t["length_um"]
    speed_achieved_um_s = t["speed_achieved_um_s"]

    # LVDT convention (Asylum_Driver_XYMetersToVoltage.vi):
    #   Volts = Position / Sensitivity + Offset
    x_v = path[:, 0] * 1e-6 / x_sens + x_offset
    y_v = path[:, 1] * 1e-6 / y_sens + y_offset
    return {
        "x_v": x_v, "y_v": y_v, "v": v,
        "ticks_per_point": ticks_per_point,
        "duration_s": duration_s,
        "length_um": length_um,
        "path_um": path,
        # leading points that carry the object's bias; the last one is the 0 V end
        "path_points": n_path,
        # what the machine will actually do, after quantisation: the panel
        # field is a ceiling, and this is the speed it settles on
        "speed_achieved_um_s": speed_achieved_um_s,
        "points_per_second": pps,
        "capture_points": capture_points(n),
    }


# --------------------------------------------------------- the wire format --
#
# `XLitho` / `YLitho` are ONE wave each for the whole pattern, with NaN as
# the pen-up marker, and `LithoVolts` is the same length with NaN in the same
# places (the vendor writes `LithoVolts = setpoint*YLitho/YLitho`, letting the
# division propagate them). The executor walks it with `Find1Index(YLitho,
# "==", NaN, LithoIndex)`, so the format is not a serialisation choice: it is
# the instrument's addressing scheme, and the hardware driver has to speak it.
#
# The separators carry meaning:
#
#   seg1, NaN, seg2, NaN, seg3          one NaN between segments
#   seg1, NaN, NaN, seg2                a DOUBLE NaN is a group boundary
#
# which is why the index advances by 2 or 3: `if YLitho[Index+2] is NaN then
# LithoIndex = Index+3 else Index+2`. Groups come from the panel's MakeGroup,
# which appends a NaN to close the current group; the next segment then adds
# its own separator, and the pair is the boundary.
#
# `LithoTotal`, the denominator the panel counts segments against, is
# literally `V_numNaNs` from `WaveStats`. That only equals the segment count
# when the pattern is group-terminated, so we terminate by default: a pattern
# handed to the machine is a closed group, and then the vendor's own count is
# right about it.


def split_long_segments(segments: list[np.ndarray],
                        limit: int = MAX_SEGMENT_POINTS) -> list[np.ndarray]:
    """Chop any segment longer than `limit` points, as CheckLithoWaves does.

    The vendor counts non-NaN points and inserts a NaN once the count
    reaches 8192, without duplicating a point, so the pieces concatenate
    back to exactly the original stroke. What is NOT preserved is
    continuity of execution: each piece becomes a segment in its own right,
    with its own move-to-start and its own timing.

    OPT-IN. `compile_pattern` no longer calls this (see MAX_SEGMENT_POINTS): on this
    stack a segment boundary is a full lift/move/set-down, so cutting a stroke here
    put a dose gap in the middle of a line. Pass `max_segment_points=` to
    `compile_pattern` if you want the vendor's segmentation back.
    """
    out: list[np.ndarray] = []
    for seg in segments:
        if len(seg) <= limit:
            out.append(seg)
            continue
        for start in range(0, len(seg), limit):
            out.append(seg[start:start + limit])
    return out


def nan_waves(segments: list[np.ndarray], values: list[np.ndarray] | None = None,
              groups: set[int] | None = None,
              terminate: bool = True) -> dict:
    """Assemble segments into the NaN-separated waves the machine takes.

    `segments` are (N, 2) arrays of already-converted wave values (volts,
    not um: this is the wire, not the design). `values` is the matching
    per-point bias, one array per segment. `groups` holds indices of
    segments AFTER which a group boundary falls, written as the double NaN.

    Returns {x, y, v, nan_count, segment_count}.
    """
    xs: list[float] = []
    ys: list[float] = []
    vs: list[float] = []
    groups = groups or set()
    last = len(segments) - 1
    for i, seg in enumerate(segments):
        seg = np.asarray(seg, dtype=np.float64)
        xs.extend(seg[:, 0].tolist())
        ys.extend(seg[:, 1].tolist())
        if values is not None:
            vs.extend(np.asarray(values[i], dtype=np.float64).tolist())
        else:
            vs.extend([0.0] * len(seg))
        if i < last or terminate:
            xs.append(math.nan)
            ys.append(math.nan)
            vs.append(math.nan)
        if i in groups and i != last:
            xs.append(math.nan)      # the second NaN of the pair: group break
            ys.append(math.nan)
            vs.append(math.nan)
    x = np.array(xs, dtype=np.float64)
    return {
        "x": x,
        "y": np.array(ys, dtype=np.float64),
        "v": np.array(vs, dtype=np.float64),
        # WaveStats V_numNaNs, which is what the panel calls LithoTotal
        "nan_count": int(np.isnan(x).sum()),
        "segment_count": len(segments),
    }


def split_segments(x: np.ndarray, y: np.ndarray,
                   v: np.ndarray | None = None) -> list[dict]:
    """Walk the NaN waves back into segments, exactly as the executor does.

    Mirrors the `LithoIndex` loop: find the next NaN, take everything before
    it, then advance by 3 if the separator is doubled and by 2 if it is not.
    Returns one dict per segment with `x`, `y`, `v`, `start`, `stop` and
    `group_break` (True when a group boundary followed it).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    out: list[dict] = []
    i = 0
    while i < n:
        if math.isnan(x[i]):        # a stray separator: skip it, no segment
            i += 1
            continue
        nxt = np.flatnonzero(np.isnan(x[i:]))
        stop = (i + int(nxt[0]) - 1) if len(nxt) else n - 1
        double = (n > stop + 2) and math.isnan(x[stop + 2])
        seg = {
            "x": x[i:stop + 1],
            "y": y[i:stop + 1],
            "v": None if v is None else np.asarray(v, dtype=np.float64)[i:stop + 1],
            "start": i,
            "stop": stop,
            "group_break": bool(double),
        }
        if len(seg["x"]):
            out.append(seg)
        i = stop + 3 if double else stop + 2
    return out


def _split_stroke(wave: np.ndarray, v: np.ndarray, n_path: int,
                  limit: int) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Cut one compiled stroke into pieces of at most `limit` fired points, each a
    stroke in its own right: a multiple of 32 points ending at 0 V.

    Only the stroke's PATH points are cut, none duplicated or dropped
    (CheckLithoWaves). `limit` is taken down to a multiple of 32 (at least 32); a
    piece holds up to that many points less one, plus its 0 V end.

    Returns (wave, bias, path_points) per piece.
    """
    block = max(32, (int(limit) // 32) * 32)
    out = []
    for a in range(0, n_path, block - 1):
        b = min(a + block - 1, n_path)
        piece, bias = _end_at_zero(wave[a:b], v[a:b], fired_points(b - a))
        out.append((piece, bias, b - a))
    return out


def compile_pattern(pattern: Pattern, x_sens: float, y_sens: float,
                    x_offset: float = 0.0, y_offset: float = 0.0,
                    clock_hz: int = 50_000,
                    frame: LithoFrame | None = None,
                    terminate: bool = True,
                    group_objects: bool = False,
                    max_segment_points: int | None = None) -> dict:
    """Compile a whole pattern into the instrument's wire format.

    Every enabled object is expanded (a filled shape can become several
    disconnected strokes), compiled, and concatenated into one NaN-separated
    wave per axis plus the matching bias wave.

    A STROKE IS NEVER CUT HERE. `max_segment_points` is None by default, so one
    expanded part is exactly one segment -- one out-wave, one arm, one fire, with the
    tip down from its first point to its last. Pass an integer (e.g.
    `MAX_SEGMENT_POINTS`) to get the vendor's CheckLithoWaves segmentation back; see
    that constant for why it is no longer the default; each piece is then a stroke
    of its own (`_split_stroke`). A stroke longer than the DSP's
    per-out-wave cap is NOT split here either -- the driver fires it as a seamless
    chunk chain (`litho/chunker.chunk_boundaries`, cutting at CHAIN_CHUNK_POINTS).

    `group_objects` writes a double NaN after each object, the vendor's
    group boundary. It is off by default because groups are an explicit
    panel action there too, and because with single separators and a
    terminator the NaN count equals the segment count, which is the rule the
    panel's own counter assumes. Turn it on and the counts diverge, which is
    why `LithoTotal` is only ever a display denominator.

    Returns the waves plus a `segments` list carrying each piece's own
    timing, because the machine arms and fires one segment at a time.
    """
    segs: list[np.ndarray] = []       # (N, 2) in wave volts
    vals: list[np.ndarray] = []
    meta: list[dict] = []
    groups: set[int] = set()
    for obj in pattern.objects:
        if not obj.enabled:
            continue
        before = len(segs)
        for part in expand(obj):
            comp = compile_object(part, x_sens, y_sens, x_offset, y_offset,
                                  clock_hz, frame=frame)
            wave = np.column_stack([comp["x_v"], comp["y_v"]])
            n_path = int(comp["path_points"])
            if max_segment_points and len(wave) > int(max_segment_points):
                pieces = _split_stroke(wave, comp["v"], n_path,
                                       int(max_segment_points))
            else:
                pieces = [(wave, comp["v"], n_path)]
            for k, (piece, bias, piece_path) in enumerate(pieces):
                # re-time each piece the way the vendor does, after the cut
                path = np.column_stack([
                    (piece[:, 0] - x_offset) * x_sens * 1e6,
                    (piece[:, 1] - y_offset) * y_sens * 1e6])
                t = time_segment(path, part.speed_um_s, clock_hz, round_up=True)
                segs.append(piece)
                vals.append(bias)
                meta.append({
                    "name": part.name if len(pieces) == 1
                            else f"{part.name}[{k + 1}/{len(pieces)}]",
                    "object": obj.name,
                    "voltage": part.voltage,
                    "path_um": path,
                    "path_points": piece_path,
                    "split": len(pieces) > 1,
                    **t,
                })
        if group_objects and len(segs) > before:
            groups.add(len(segs) - 1)     # one object drawn = one group
    waves = nan_waves(segs, vals, groups, terminate=terminate)
    for m, seg, val in zip(meta, segs, vals):
        m["x_v"] = seg[:, 0]
        m["y_v"] = seg[:, 1]
        m["v"] = val
        m["capture_points"] = capture_points(len(seg))
    waves["segments"] = meta
    waves["frame"] = (frame or LithoFrame()).to_dict()
    waves["duration_s"] = sum(m["duration_s"] for m in meta)
    return waves


def demo_pattern() -> Pattern:
    """A SketchFET-flavored demo: source wire, two gate fingers, erase notch."""
    return Pattern(name="sketchfet-demo", objects=[
        LithoObject("wire", "path", [(-8.0, 0.0), (8.0, 0.0)],
                    voltage=8.0, speed_um_s=2.0),
        LithoObject("gate-left", "path", [(-2.5, 5.0), (-2.5, 0.9)],
                    voltage=8.0, speed_um_s=2.0),
        LithoObject("gate-right", "path", [(2.5, 5.0), (2.5, 0.9)],
                    voltage=8.0, speed_um_s=2.0),
        LithoObject("erase-notch", "path", [(0.0, 1.6), (0.0, -1.6)],
                    voltage=-8.0, speed_um_s=1.5),
    ])
