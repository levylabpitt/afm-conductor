"""Pens: sketch strokes become lithography.

A pen turns a hand-drawn stroke (touch screen, stylus, mouse) into one or
more LithoObjects. Like OneNote pens, each has a palette entry and a set of
parameters; unlike OneNote, the parameters are physical (bias, speed, pitch,
handedness) and every one of them is exposed for an agent to sweep.

The motivating use case is chirality-induced spin selectivity (CISS): writing
families of chiral wires at room temperature, varying pitch, amplitude,
handedness and bias, and measuring which ones show the effect. Everything
here is built so that "explore this parameter" is a loop over pen params.

Interpretation modes decide how literally the stroke is taken:
    literal  - the stroke as drawn (resampled, lightly denoised)
    clean    - simplified and smoothed, kinks removed
    ideal    - fitted to intent: a near-straight stroke becomes a straight
               line, a near-circular one becomes an arc

Chirality: voltage is the third dimension.

    y(x)    = y0 + y_k sin(2*pi*x/lambda)          lateral tip position
    V_tip(x) = V0 + V_k sin(2*pi*x/lambda + phi)   tip voltage

Both modulations act on the electron wavefunction: the lateral one displaces
it in-plane, the voltage one displaces and squeezes it vertically by setting
the local carrier density. Two displacements in quadrature trace a helix, so
the curve (x, y, V) is a genuine three-dimensional chiral object even though
the tip never leaves the plane. Handedness is the sign of phi: +90 and -90
degrees are enantiomers, while phi = 0 or 180 leaves the curve planar, hence
achiral, which is the control experiment.

Reference: Engineered chirality of one-dimensional nanowires, Sci. Adv.
(2025), doi:10.1126/sciadv.adx4761. Device A there used y_k = 10 nm,
lambda = 10 nm, V0 = 10 V, V_k = 5 V, phi = pi/2, 34 periods, against a
straight control waveguide, in a device about 700 nm end to end.

The pen's DEFAULTS are not Device A. They are the KTO helix regime in use
now (lambda = 25 nm, y_k = 55 nm, amplitude well above pitch); Device A is
the "device-a" entry in CHIRAL_PRESETS, beside "achiral-control" at phi = 0.

A note on looking at one of these. At the full 90 um scanner range a 25 nm
period is a small fraction of one screen pixel, so the bias modulation, which
is the whole physics, cannot be drawn at all: the wire renders as a smear of
the mean colour and looks nothing like the figures. That is a display scale
problem, not a waveform problem. Fit the frame to the design (the panel's
"Fit to design", or G) and each period gets ~20 px.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .bezier import signed_area
from .compiler import DEFAULT_INTERP_UM, LithoObject

AREA_EPS = 1e-9

# --------------------------------------------------------------- geometry --


def resample(points_um, spacing_um: float = 0.05) -> np.ndarray:
    """Equal-arc-length resampling of a raw stroke."""
    p = np.asarray(points_um, dtype=np.float64)
    if len(p) < 2:
        return p
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 0:
        return p[:1]
    n = max(2, int(round(total / spacing_um)) + 1)
    si = np.linspace(0, total, n)
    return np.column_stack([np.interp(si, s, p[:, 0]), np.interp(si, s, p[:, 1])])


def _smooth(p: np.ndarray, win: int = 9) -> np.ndarray:
    """Moving average with a window that shrinks to nothing at the ends.

    Two earlier versions of this were wrong in opposite directions.
    Convolving in "same" mode pads with zeros, which drags the ends toward
    the origin and leaves a spike where the restored endpoints meet the
    contaminated interior: that spike made smoothing *lengthen* a jittery
    stroke. Edge replication fixed the spike but still pulled both ends
    inward by about half the window, so a stroke drawn 3.000 um long came
    back 2.889 um. That is not cosmetic: every pen derives its period count
    from the axis length, so a 3.7% short axis wrote 116 helix periods
    where 120 were asked for.

    Shrinking the window towards the ends (radius 0 at the endpoints) fixes
    both. Endpoints are preserved exactly, the interior is smoothed in
    full, and there is no discontinuity between the two.
    """
    n = len(p)
    if n < 3:
        return p
    win = min(win if win % 2 else win + 1, n if n % 2 else n - 1)
    if win < 3:
        return p
    h = win // 2
    cs = np.concatenate([np.zeros((1, 2)), np.cumsum(p, axis=0)])
    idx = np.arange(n)
    r = np.minimum(np.minimum(idx, n - 1 - idx), h)      # radius, 0 at the ends
    lo, hi = idx - r, idx + r + 1
    return (cs[hi] - cs[lo]) / (hi - lo)[:, None]


def _straightness(p: np.ndarray) -> float:
    """How line-like a stroke is, as 1 - (transverse spread / chord).

    Deliberately NOT chord-over-arc-length: hand jitter inflates arc length
    badly (a steady line drawn on a touch screen measures only ~0.9 that
    way), while the transverse deviation from the chord stays small, which
    is what "the user meant a line" actually means.
    """
    chord = p[-1] - p[0]
    L = float(np.hypot(*chord))
    if L <= 0:
        return 0.0
    nvec = np.array([-chord[1], chord[0]]) / L
    dev = np.abs((p - p[0]) @ nvec)
    return float(1.0 - min(np.percentile(dev, 95) / L, 1.0))


def interpret(points_um, mode: str = "clean", spacing_um: float = 0.05):
    """Turn a raw stroke into the path the user meant.

    Sampling and smoothing both scale with the stroke, which they did not
    used to. A fixed 0.05 um resample and a fixed 9-sample moving average
    are fine on a 5 um line and destroy a small loop: a circle drawn at
    100 nm radius came back as nine points with its radius collapsed from
    0.100 to 0.037 um, because the smoothing window spanned several radians
    of arc. A moving average over an arc of angle a shrinks a circle by
    sinc(a/2), so the window has to stay short compared to the curvature,
    and the only way to guarantee that without knowing the shape is to
    measure both against the stroke's own length.
    """
    raw = np.asarray(points_um, dtype=np.float64)
    if len(raw) >= 2:
        L_raw = float(np.hypot(*np.diff(raw, axis=0).T).sum())
        if L_raw > 0:
            spacing_um = min(spacing_um, L_raw / 400.0)
    p = resample(points_um, spacing_um)
    if len(p) < 2 or mode == "literal":
        return p
    # smooth, then resample again: the second pass measures arc length along
    # the smoothed curve. Without it, hand jitter inflates the path length
    # (40% on a steadily drawn line), which would distort every pitch and
    # period the pens compute from it.
    win = max(3, min(9, len(p) // 20))
    p = resample(_smooth(p, win=win), spacing_um)
    if mode == "clean":
        return p
    # ideal: straighten if nearly straight, else fit a circular arc
    if _straightness(p) > 0.985:
        n = max(2, len(p))
        return np.column_stack([np.linspace(p[0, 0], p[-1, 0], n),
                                np.linspace(p[0, 1], p[-1, 1], n)])
    arc = _fit_arc(p)
    return arc if arc is not None else p


def _fit_arc(p: np.ndarray):
    """Least-squares circle through the stroke; None if it is not arc-like."""
    x, y = p[:, 0], p[:, 1]
    A = np.column_stack([x, y, np.ones(len(p))])
    b = x ** 2 + y ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = sol[0] / 2, sol[1] / 2
    r = math.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0.0))
    if not (1e-3 < r < 1e4):
        return None
    resid = np.abs(np.hypot(x - cx, y - cy) - r)
    if np.median(resid) > 0.08 * r:
        return None
    a0 = math.atan2(y[0] - cy, x[0] - cx)
    a1 = math.atan2(y[-1] - cy, x[-1] - cx)
    # unwrap toward the direction the stroke actually went
    mid = math.atan2(y[len(p) // 2] - cy, x[len(p) // 2] - cx)
    if not _between(a0, mid, a1):
        a1 += 2 * math.pi if a1 < a0 else -2 * math.pi
    t = np.linspace(a0, a1, max(8, len(p)))
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def _between(a, m, b) -> bool:
    d = (b - a) % (2 * math.pi)
    dm = (m - a) % (2 * math.pi)
    return dm <= d


def _frames(p: np.ndarray, span: int = 9):
    """Arc length, unit tangents and unit normals along a path.

    Tangents are estimated over a window rather than between neighbouring
    points: with a jittery hand-drawn stroke, adjacent samples can double
    back, and a per-segment tangent then flips, which would send a chiral
    side branch out of the wrong side of the wire.
    """
    d = np.diff(p, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    s = np.concatenate([[0.0], np.cumsum(seg)])

    n = len(p)
    h = max(1, span // 2)
    idx = np.arange(n)
    lo = np.clip(idx - h, 0, n - 1)
    hi = np.clip(idx + h, 0, n - 1)
    t = p[hi] - p[lo]
    mag = np.hypot(t[:, 0], t[:, 1])
    bad = mag < 1e-12
    if bad.any():                       # degenerate window: fall back to chord
        chord = p[-1] - p[0]
        cm = np.hypot(*chord) or 1.0
        t[bad] = chord / cm
        mag[bad] = 1.0
    t = t / mag[:, None]
    nvec = np.column_stack([-t[:, 1], t[:, 0]])
    return s, t, nvec


# ------------------------------------------------------------------- pens --


@dataclass
class Pen:
    key: str
    label: str
    hint: str
    params: dict
    render: Callable[[np.ndarray, dict, str], list[LithoObject]] = field(repr=False)
    # Parameters that take one of a fixed set of values, as
    # {param: [(value, label), ...]}. The panel renders these as a menu and
    # everything else as a number field, which is why this has to travel with
    # the pen: without it a string parameter became a number input, the
    # panel's parseFloat turned the choice into NaN, and the region pen's two
    # most important controls -- which fill, and rectangle or as-drawn --
    # could not be reached at all. A pen that grows a choice tomorrow gets a
    # working menu without anyone touching the front end.
    choices: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return {"key": self.key, "label": self.label, "hint": self.hint,
                "params": self.params,
                "choices": {k: [list(c) for c in v]
                            for k, v in self.choices.items()}}


def _obj(name, pts, **kw) -> LithoObject:
    return LithoObject(name=name, kind="path",
                       points_um=[(float(a), float(b)) for a, b in pts], **kw)


def _render_write(path, p, name):
    return [_obj(f"{name}", path, voltage=p["voltage"],
                 speed_um_s=p["speed_um_s"], interpolation_um=DEFAULT_INTERP_UM)]


def _render_erase(path, p, name):
    return [_obj(f"{name}", path, voltage=-abs(p["voltage"]),
                 speed_um_s=p["speed_um_s"], interpolation_um=DEFAULT_INTERP_UM)]


def _render_region(path, p, name):
    """A closed region, written at one bias, filled one of three ways.

    User request, 2026-08-08: "I want to be able to scan rectangles or arbitrary
    shapes that I define with whatever voltage. The rastering is done either
    along a certain direction for convex shapes or it is done in a
    self-similar shape-filling manner."

    So: `directional` sweeps parallel lines at a chosen angle, serpentined so
    the tip does not fly back at the end of every one -- the right answer for
    a convex region where one fast axis matters. `contour` follows the
    region's own outline inward, which keeps the toolpath parallel to the
    edge everywhere, writes the boundary as one continuous pass rather than
    as a thousand line-ends, and handles concave shapes without long
    retraces.

    Neither of those is a space-filling CURVE, and the user (2026-08-08) noted
    they were still missing. `hilbert` is the third: one continuous pass over
    the whole region with no run longer than a pitch, so there is no
    preferred direction to imprint on the written material, no turnaround
    dwell along two edges, and no join inside the region. See fill_hilbert.

    `rect` is not a third strategy, it is a convenience: the stroke's
    bounding box, for when what you wanted was a rectangle and drawing four
    corners by hand is beneath the tool.

    **What comes back is the REGION, not its fill.** It used to be the fill:
    the strategy ran here, when the stroke was drawn, and the design then
    held a few thousand serpentine lines with no polygon anywhere in it. The
    fill is now a property of the object and `compiler.expand` runs the same
    function with the same parameters when the pattern is compiled, so the
    tip path is unchanged (`tests/test_region_object.py` holds the two
    against each other point for point) and the things that were impossible
    become ordinary: union two regions, outset one, edit its nodes, change
    the pitch without redrawing it, and write it to OASIS as the polygon it
    always was.
    """
    # Literal means the outline as given. See render_stroke.
    raw = p.get("_stroke_um")
    pts = np.asarray(raw if (p.get("_mode") == "literal" and raw is not None
                             and len(raw) >= 3) else path, float)
    if len(pts) < 3:
        return []

    if p.get("shape") == "rect":
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        poly = np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                         [hi[0], hi[1]], [lo[0], hi[1]]], float)
    else:
        poly = pts if np.allclose(pts[0], pts[-1]) else np.vstack([pts, pts[:1]])

    # A region with no area is not a region. The fill used to say so by
    # returning no strokes for it; now that the object IS the outline,
    # nothing downstream would notice, and a zero-area shape would sit in
    # the document being selectable, unionable and unwritable.
    if abs(signed_area(poly)) < AREA_EPS:
        return []

    kind = str(p.get("fill", "directional"))
    c = p.get("centre_um")
    return [LithoObject(
        name=name, kind="shape",
        points_um=[(float(x), float(y)) for x, y in poly],
        voltage=float(p["voltage"]),
        speed_um_s=float(p["speed_um_s"]),
        interpolation_um=DEFAULT_INTERP_UM,
        fill_spacing_um=max(float(p["pitch_um"]), 1e-3),
        fill_kind=kind,
        # A hand-drawn outline has no centre of its own and the radial fills
        # fall back to its centroid; a shape placed by the shape tools knows
        # exactly where its centre is and says so.
        fill_centre_um=(float(c[0]), float(c[1])) if c else None,
        # `directional` sweeps at the angle asked for; every other strategy
        # ignores it, and storing None for those keeps `fill_angle_deg`
        # meaning "along the long axis" everywhere else in the program.
        fill_angle_deg=(float(p.get("angle_deg", 0.0))
                        if kind == "directional" else None),
        outline_pass=True,
    )]


def _render_superlattice(path, p, name):
    """Periodic potential along the wire: bias alternates between two levels.

    One object, one continuous pass; the modulation lives in the bias
    profile, so the lattice constant is set by period_um and the mark/space
    ratio by duty.
    """
    s, _, _ = _frames(path)
    L = float(s[-1]) or 1.0
    period = max(p["period_um"], 1e-3)
    n = max(64, int(L / period * 40))
    u = np.linspace(0, L, n)
    phase = np.mod(u, period) / period
    v = np.where(phase < p["duty"], p["v_hi"], p["v_lo"])
    return [_obj(name, path, voltage=p["v_hi"], v_profile=[float(x) for x in v],
                 speed_um_s=p["speed_um_s"], interpolation_um=DEFAULT_INTERP_UM)]


def _render_diode(path, p, name):
    """Asymmetric (ratchet) bias sawtooth: slow rise, fast fall.

    Inversion symmetry along the wire is broken by the profile itself, so a
    forward carrier sees a different barrier sequence than a backward one.
    asymmetry = fraction of each period spent rising.
    """
    s, _, _ = _frames(path)
    L = float(s[-1]) or 1.0
    period = max(p["period_um"], 1e-3)
    n = max(64, int(L / period * 60))
    u = np.linspace(0, L, n)
    ph = np.mod(u, period) / period
    a = min(max(p["asymmetry"], 0.05), 0.95)
    rise = ph / a
    fall = 1.0 - (ph - a) / (1 - a)
    shape = np.where(ph < a, rise, fall)
    v = p["v_min"] + (p["v_max"] - p["v_min"]) * shape
    return [_obj(name, path, voltage=p["v_max"], v_profile=[float(x) for x in v],
                 speed_um_s=p["speed_um_s"], interpolation_um=DEFAULT_INTERP_UM)]


# An offset curve folds once the offset reaches the local radius of
# curvature. Staying at 60% of it leaves the inner side comfortably convex
# and the tip path single-valued.
CURVATURE_HEADROOM = 0.6


def _axis_spline(path: np.ndarray, closed: bool):
    """Smooth curve through the drawn axis, with analytic derivatives.

    Everything downstream needs three things to agree: the position, the
    direction, and the curvature. Interpolating position by chords while
    interpolating the normal separately makes them disagree, and the
    modulation stops being perpendicular to the path it is modulating
    (measured: 83 degrees on a circle, 76 at a corner, where it must be 90).
    One spline gives all three from the same object, so they cannot drift
    apart.

    A closed stroke gets a periodic spline, so the frame at the join is the
    same frame arriving and leaving.
    """
    from scipy.interpolate import CubicSpline, PchipInterpolator

    seg = np.hypot(*np.diff(path, axis=0).T)
    s_axis = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 1e-12])
    s_axis, q = s_axis[keep], path[keep]
    if closed and len(q) > 2:
        if not np.allclose(q[0], q[-1], atol=1e-12):
            q = np.vstack([q, q[:1]])
            s_axis = np.concatenate([s_axis, [s_axis[-1] + float(np.hypot(*(q[0] - q[-2])))]])
        try:
            return CubicSpline(s_axis, q, bc_type="periodic"), float(s_axis[-1])
        except ValueError:
            pass
    return PchipInterpolator(s_axis, q, axis=0), float(s_axis[-1])


def total_turning_deg(path: np.ndarray) -> float:
    """How far the stroke turned, in degrees. A loop turns about 360."""
    p = np.asarray(path, dtype=np.float64)
    d = np.diff(p, axis=0)
    n = np.hypot(*d.T)
    keep = n > 1e-15
    d = d[keep]
    if len(d) < 2:
        return 0.0
    ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return float(np.degrees(abs(ang[-1] - ang[0])))


def is_closed(path: np.ndarray, gap_frac: float = 0.45,
              turn_deg: float = 300.0) -> bool:
    """Did the stroke mean to come back to where it started?

    Gap alone is the wrong test, and using it alone is why rings kept not
    closing. Nobody draws a circle that meets itself: a stroke stopped only
    2% short of the full turn already leaves a gap of 6% of its own extent,
    and one stopped 10% short leaves 31%. A 2% tolerance therefore called
    every real hand-drawn circle open, which is why rings kept showing a
    notch with a stray part-period hanging off it.

    What actually distinguishes a loop from an arc is how far it TURNED. A
    stroke that has come most of the way round and ends near where it began
    is a loop, and is treated as one.
    """
    p = np.asarray(path, dtype=np.float64)
    if len(p) < 8:
        return False
    extent = float(max(np.ptp(p[:, 0]), np.ptp(p[:, 1])))
    if extent <= 0:
        return False
    gap = float(np.hypot(*(p[-1] - p[0]))) / extent
    return gap <= gap_frac and total_turning_deg(p) >= turn_deg


# The point at which chiral_profile REFUSES a wire (with a diagnostic) rather than
# allocate it (u_fine is 4*n, so ~5M here is ~160 MB -- the memory wall). It sits
# FAR above any well-formed wire: a fine sub-nm pitch with a PROPORTIONAL amplitude
# stays in the ~10^5-10^6 range, while the runaway that crashed (lambda at its floor
# left with a large amplitude, ratio ~550) wants ~1.5e9. It is a count/memory limit,
# NOT a pitch floor -- fine pitches pass, an extreme amplitude:pitch ratio does not.
_CHIRAL_MAX_POINTS = 5_000_000

# The most a chord may cut below a sine peak, A(1 - cos(k s / 2)), and the coarsest
# chiral spacing allowed (an ordinary wire's default), both in um.
PEAK_TOL_UM = 0.0005
MAX_SPACING_UM = 0.02


def chiral_spacing_um(lam: float, amp: float) -> float:
    """Arc-length point spacing for a chiral wire of wavelength `lam`, amplitude `amp`.

    The spacing that keeps every peak within PEAK_TOL_UM of the sine, capped at
    MAX_SPACING_UM and at lam / 8 so the bias modulation keeps 8 points per period.
    """
    limit = min(MAX_SPACING_UM, lam / 8.0)
    if amp > PEAK_TOL_UM:
        k = 2.0 * math.pi / lam
        limit = min(limit, (2.0 / k) * math.acos(1.0 - PEAK_TOL_UM / amp))
    return max(limit, 1e-4)


@dataclass(frozen=True)
class ChiralBuild:
    """Everything ONE `chiral_profile` call produced, RETURNED rather than stashed.

    It used to hand the profile, the clamp report and the axis/normal arrays back as
    ATTRIBUTES ON THE FUNCTION (`chiral_profile.last_profile`, read back through
    `getattr` a few lines later). That is hidden, non-reentrant state, and the race is
    not theoretical: two builds in flight at once -- the cockpit's
    `POST /api/design/chiral` while a FLEX `loadPattern` builds its own chiral spec,
    both on threads off the same event loop -- and the second overwrites the first's
    profile between its return and its read. Measured 2026-09-12: 285 of 300
    concurrent builds came back with a CROSS-CONTAMINATED profile.

    Not a cosmetic mix-up. The profile is PERSISTED on the object: the canvas renders
    the wire from it, and `routers/design._reapply_chiral` rebuilds the geometry and
    the bias profile from it after a node edit -- so a swapped profile is a wire
    silently re-written to ANOTHER wire's amplitude, pitch and phase. A stale
    `last_clamp` was the same hazard in reverse: `curvature_clamped: false` reported
    on a wire whose amplitude had in fact been clamped, which in a chirality
    experiment is a wrong measurement rather than a wrong label.

    `profile` and `clamp` are None only for the degenerate early returns (a
    zero-length axis), where nothing was built to describe.
    """

    points: np.ndarray
    volts: np.ndarray
    arclen: float
    profile: dict | None = None
    clamp: dict | None = None
    axis: np.ndarray | None = None       # full resolution, what the modulation used
    normal: np.ndarray | None = None


def chiral_profile(path, lambda_um, y_amp_um, v0, v_k, phase_deg, hand=1.0,
                   samples_per_period: int = 24, step_frac: float = 0.04
                   ) -> ChiralBuild:
    """The engineered-chirality construction, returned as a `ChiralBuild`.

        y(x)     = y0 + y_k sin(2 pi x / lambda)
        V_tip(x) = V0 + V_k sin(2 pi x / lambda + phi)

    Both are parameterized by x, the coordinate ALONG the wire axis, so the
    voltage is resampled onto the modulated path's own arc length before it
    is handed to the compiler (the wiggly path is longer than its axis, and
    ignoring that would slide the bias out of phase with the geometry: at a
    10 nm wavelength that is the whole effect).
    """
    closed = is_closed(path)
    spline, L = _axis_spline(np.asarray(path, dtype=np.float64), closed)
    if L <= 0:
        return ChiralBuild(path, np.full(len(path), v0), 0.0)
    lam = max(float(lambda_um), 1e-4)

    # Sample so the step ALONG THE PATH stays small, not the step along the
    # axis. When the amplitude exceeds the wavelength (the KTO regime:
    # A ~ 55 nm, lambda ~ 25 nm) the path is ~9x longer than its axis and
    # the local slope reaches 2*pi*A/lambda ~ 14, so axial sampling would
    # deliver the sinusoid as a coarse polygon with facets a quarter of the
    # amplitude.
    slope = 2 * np.pi * float(y_amp_um) / lam
    density = math.hypot(1.0, slope)
    per_period = max(samples_per_period, int(density / step_frac / 4) + 1)
    n = max(len(path), int(L / lam * per_period) + 2)
    # DETECT a runaway sample count instead of blindly allocating it (u_fine
    # below is 4*n, so a 6.7e9-point n is a 50 GiB alloc -> hard crash). The
    # count is n ~ L/lam * per_period ~ L*amp/lam^2, so it is driven by the
    # AMPLITUDE-TO-PITCH RATIO (amp/lam), NOT by lam being small on its own: a
    # fine pitch with a proportionally fine amplitude is cheap. So we do NOT
    # impose a minimum pitch (some tasks need very fine ones); we refuse only the
    # runaway and say exactly why, so the caller can fix the offending parameter.
    #
    # WHICH parameter to complain about depends on which one is out of scale, and
    # saying the wrong one is worse than saying nothing: a 1.5 mm axis at the pen's
    # own default 55 nm / 25 nm used to be refused as "amplitude very large relative
    # to the pitch (ratio 2:1)", which is the RECOMMENDED regime and left the caller
    # nothing to act on. n = L/lam * per_period has two factors, so name the one
    # actually responsible -- the ratio when it is extreme, the axis length otherwise.
    if n > _CHIRAL_MAX_POINTS:
        ratio = float(y_amp_um) / lam
        if ratio >= 4.0:
            raise ValueError(
                f"chiral wire needs {n:,} sample points (limit "
                f"{_CHIRAL_MAX_POINTS:,}): amplitude {float(y_amp_um) * 1000:g} nm is "
                f"very large relative to the pitch {lam * 1000:g} nm (ratio "
                f"{ratio:.0f}:1). A chiral wire has amplitude comparable to its pitch "
                f"-- reduce the amplitude or increase the pitch. (Fine pitches are "
                f"fine if the amplitude scales down with them.)")
        raise ValueError(
            f"chiral wire needs {n:,} sample points (limit {_CHIRAL_MAX_POINTS:,}): "
            f"the axis is {L:,.0f} um long, which is {L / lam:,.0f} periods at a "
            f"{lam * 1000:g} nm pitch, and each period is sampled {per_period} times. "
            f"The amplitude-to-pitch ratio ({ratio:.1f}:1) is fine -- it is the LENGTH "
            f"that does not fit. Shorten the axis or coarsen the pitch.")

    # A closed loop must carry a WHOLE number of periods or the modulation
    # arrives at the join out of phase with itself. A drawn circle came to
    # 249.27 periods: a 97 degree phase step and a 5.6 V bias step at the
    # seam, which is the notch you can see. Quantising the wavelength shifts
    # the pitch by well under a percent and closes it exactly.

    # Sample at uniform TRUE ARC LENGTH, not uniform spline parameter.
    # The spline is parameterised by the chord length of the drawn polyline,
    # and its speed |dC/ds| is not 1: around a hand-drawn circle it varies
    # by about 9%. Stepping uniformly in the parameter therefore stretched
    # and squeezed the pitch by that much around the loop, which is a real
    # error in the written wavelength, not a display artifact.
    u_fine = np.linspace(0.0, L, max(4 * n, 4096))
    sp_fine = np.hypot(*spline(u_fine, 1).T)
    s_true = np.concatenate([[0.0], np.cumsum(
        0.5 * (sp_fine[1:] + sp_fine[:-1]) * np.diff(u_fine))])
    arclen = float(s_true[-1])
    if arclen <= 0:
        return ChiralBuild(path, np.full(len(path), v0), 0.0)
    u = np.interp(np.linspace(0.0, arclen, n), s_true, u_fine)
    ax = spline(u)
    d1 = spline(u, 1)
    speed = np.maximum(np.hypot(d1[:, 0], d1[:, 1]), 1e-12)
    tang = d1 / speed[:, None]
    nn = np.column_stack([-tang[:, 1], tang[:, 0]])   # exactly perpendicular

    # Curvature analytically from the same spline, so the amplitude limit
    # below is measured on the curve actually being modulated.
    d2 = spline(u, 2)
    kappa = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / speed ** 3
    r_curv = np.where(kappa > 1e-12, 1.0 / np.maximum(kappa, 1e-12), np.inf)

    # The amplitude has to fit inside the local radius of curvature. An
    # offset curve folds when the offset reaches that radius: the inner side
    # turns itself inside out, the path reverses, and the tip is asked to
    # double back. On a 100 nm circle at 55 nm amplitude (A/r = 0.55) this
    # produced a genuine starburst, 25 spokes with eight reversals, and it
    # read as a rendering bug because every number was individually fine.
    #
    # Clamped POINTWISE, and reported rather than done quietly: for a
    # chirality experiment the amplitude IS the physics, so writing a
    # different one in silence would be worse than the fold.
    amp = np.minimum(np.full(n, float(y_amp_um)), CURVATURE_HEADROOM * r_curv)
    clamped = amp < float(y_amp_um) - 1e-15

    # phase advances with distance travelled, so the pitch is constant in
    # microns wherever you are on the curve
    n_periods = arclen / lam
    if closed:
        n_periods = max(1.0, round(n_periods))
        lam = arclen / n_periods
    theta = 2 * np.pi * n_periods * np.linspace(0.0, 1.0, n)
    phi = math.radians(float(phase_deg)) * hand
    pts = ax + nn * (amp * np.sin(theta))[:, None]
    volts = float(v0) + float(v_k) * np.sin(theta + phi)
    if closed:
        pts[-1] = pts[0]              # exact seam, no residual gap
    # The ANALYTIC description of this wire: the smooth axis plus the
    # modulation parameters. This, not a point cloud, is what the canvas
    # should render from: the axis has no fine structure (a few hundred
    # samples reproduce it to sub-nanometre), and the fine structure is a
    # sinusoid the renderer can evaluate exactly at whatever resolution the
    # zoom in use requires. Every aliasing and faceting bug in this file's
    # history came from sending sampled points instead of this.
    m_prof = min(n, 512)
    f_prof = np.linspace(0.0, 1.0, m_prof)
    f_full = np.linspace(0.0, 1.0, n)
    ax_p = np.column_stack([np.interp(f_prof, f_full, ax[:, 0]),
                            np.interp(f_prof, f_full, ax[:, 1])])
    nn_p = np.column_stack([np.interp(f_prof, f_full, nn[:, 0]),
                            np.interp(f_prof, f_full, nn[:, 1])])
    nn_p /= np.maximum(np.hypot(nn_p[:, 0], nn_p[:, 1]), 1e-12)[:, None]
    amp_p = np.interp(f_prof, f_full, amp)
    profile = {
        "kind": "chiral",
        "axis_um": [[round(float(x), 5), round(float(y), 5)] for x, y in ax_p],
        "normal": [[round(float(x), 6), round(float(y), 6)] for x, y in nn_p],
        "amp_um": [round(float(a), 6) for a in amp_p],
        "arclen_um": round(float(arclen), 6),
        "periods": float(n_periods),
        "lambda_um": float(lam),
        "v0": float(v0), "v_k": float(v_k),
        "phase_rad": float(phi),
        # the REQUESTED amplitude (amp_um above is clamped pointwise); kept so
        # the modulation can be regenerated from the profile alone after the
        # base axis is node-edited (design.py _reapply_chiral). hand is already
        # folded into phase_rad (phi = radians(phase_deg) * hand), so re-running
        # with phase_deg = degrees(phase_rad), hand = 1 reproduces phi exactly.
        "y_amp_um": float(y_amp_um),
        "closed": bool(closed),
    }
    clamp = {
        "fraction": float(clamped.mean()),
        "min_amp_um": float(amp.min()),
        "requested_um": float(y_amp_um),
        "tightest_radius_um": float(np.min(r_curv)),
        "closed": bool(closed),
        "periods": float(n_periods),
        "lambda_um": float(lam),
    }

    # voltage lives on the axial coordinate; the compiler spreads v_profile
    # uniformly over ARC LENGTH, so map it there explicitly
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s_path = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s_path[-1]) or 1.0
    v_uniform = np.interp(np.linspace(0.0, total, n), s_path, volts)
    # RETURNED, all of it, in one record. See ChiralBuild for what stashing it on the
    # function cost. `axis`/`normal` are the full-resolution arrays the modulation was
    # actually built on, so nothing downstream has to re-derive them and drift.
    return ChiralBuild(pts, v_uniform, total, profile=profile, clamp=clamp,
                       axis=ax, normal=nn)


def is_chiral(o) -> bool:
    return (getattr(o, "profile", None) or {}).get("kind") == "chiral"


def chiral_settings(o) -> dict | None:
    """The stored chiral settings of `o`, or None if it has no complete chiral profile.

    `hand` is folded into `phase_rad` by `chiral_profile`, so the readback is
    `phase_deg = degrees(phase_rad)` with `hand = 1`. v0 is not a setting: it is the
    object's `voltage`. The pitch is the one asked for (`lambda_set_um`), not the one
    a closed spine quantised it to (`lambda_um`), so repeated re-bakes cannot drift.
    """
    prof = getattr(o, "profile", None) or {}
    if prof.get("kind") != "chiral":
        return None
    try:
        amp = prof.get("y_amp_um")
        if amp is None:
            amp = max(prof.get("amp_um") or [0.0])
        lam = prof.get("lambda_set_um")
        if lam is None:
            lam = prof["lambda_um"]
        return {"lambda_um": float(lam), "y_amp_um": float(amp),
                "v_k": float(prof["v_k"]),
                "phase_deg": math.degrees(float(prof["phase_rad"])), "hand": 1.0}
    except (KeyError, TypeError, ValueError):
        return None


def chiral_spine(o) -> list | None:
    """The spine a chiral wire is baked from: `axis_um`, else the profile's own axis."""
    if not is_chiral(o):
        return None
    spine = getattr(o, "axis_um", None) or (o.profile or {}).get("axis_um")
    return list(spine) if spine and len(spine) >= 2 else None


def bake_chiral(o, lambda_um, y_amp_um, v_k, phase_deg, hand=1.0,
                spine=None) -> ChiralBuild:
    """Derive `o`'s toolpath, bias and profile from its spine and these settings.

    The one place a chiral wire is baked onto an object. v0 is `o.voltage`, never the
    other way round. The spine is `spine`, else `o.axis_um`, else `o.points_um` (a
    plain path being made chiral), and is kept as `o.axis_um`. The pitch asked for
    is kept as `profile["lambda_set_um"]`. Raises ValueError (a spine with no length,
    a `chiral_profile` refusal, or a result that is not finite) with `o` untouched.
    """
    lam = max(float(lambda_um), 1e-4)
    amp = float(y_amp_um)
    if spine is None:
        spine = getattr(o, "axis_um", None) or o.points_um
    spine = [(float(x), float(y)) for x, y in spine]
    arr = np.asarray(spine, dtype=float)
    if len(arr) < 2 or float(np.hypot(*np.diff(arr, axis=0).T).sum()) <= 0.0:
        raise ValueError("its spine has no length (every point is at the same "
                         "place), so there is no wire to modulate")
    v0 = float(o.voltage)
    too_big = ("the settings are too large to build a wire from (the result is not "
               "finite)")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            build = chiral_profile(arr, lam, amp, v0,
                                   float(v_k), float(phase_deg), float(hand))
    except OverflowError as exc:
        raise ValueError(too_big) from exc
    if not (np.isfinite(build.points).all() and np.isfinite(build.volts).all()):
        raise ValueError(too_big)
    o.axis_um = spine
    o.points_um = [(float(a), float(b)) for a, b in build.points]
    o.v_profile = [float(x) for x in build.volts]
    o.voltage = v0
    o.profile = build.profile and {**build.profile, "lambda_set_um": lam}
    o.interpolation_um = chiral_spacing_um(lam, amp)
    return build


def rebake_chiral(o) -> ChiralBuild | None:
    """Re-apply a chiral wire's stored settings to its spine at its current voltage.

    Called after anything changes the voltage or the spine. Returns None, with `o`
    untouched, when `o` has no chiral profile, no spine, or incomplete settings.
    """
    settings = chiral_settings(o)
    spine = chiral_spine(o)
    if settings is None or spine is None:
        return None
    return bake_chiral(o, spine=spine, **settings)


def _render_chiral(path, p, name):
    """A chiral wire in the sense of Sci. Adv. doi:10.1126/sciadv.adx4761.

    The tip stays in the plane, but the electron wavefunction is displaced
    laterally by the path and vertically by the tip voltage (which sets the
    local density), so the curve (x, y, V) is a helix. phi = +/-90 degrees
    are the two enantiomers; phi = 0 or 180 keeps the curve planar and is
    the achiral control.
    """
    hand = -1.0 if str(p.get("handedness", "R")).upper().startswith("L") else 1.0
    lam = max(float(p["lambda_um"]), 1e-4)
    amp = float(p["y_amp_um"])
    build = chiral_profile(
        path, lam, amp, p["v0"], p["v_k"], p["phase_deg"], hand)
    o = _obj(name, build.points,
             voltage=p["v0"],
             v_profile=[float(x) for x in build.volts],
             speed_um_s=p["speed_um_s"],
             interpolation_um=chiral_spacing_um(lam, amp))
    o.profile = build.profile
    # The clamp report rides ON THIS OBJECT rather than on the function, so a caller
    # that went through `render_stroke` (the sketch endpoint) still gets it and gets
    # only its OWN -- a pen that never touches chiral_profile leaves the attribute
    # absent, and two concurrent strokes cannot read each other's.
    o.chiral_clamp = build.clamp
    return [o]


def path_report(objs) -> dict:
    """Axial vs travelled length, and what that costs in write time.

    In the high-amplitude regime the tip travels many times the length of
    the wire it is drawing, so a nanowire that looks like 2 um of device is
    20 um of writing. Worth knowing before starting an overnight run.
    """
    from .compiler import expand
    axial = travelled = seconds = 0.0
    # A region's outline is a few microns and its fill is a few thousand, so
    # reporting the object would understate an overnight run by three orders
    # of magnitude. Report what the tip does, which means resolving the fill
    # exactly as the compiler will.
    for seg in [s for o in objs for s in expand(o)]:
        p = np.asarray(seg.points_um, dtype=float)
        if len(p) < 2:
            continue
        d = float(np.hypot(*np.diff(p, axis=0).T).sum())
        travelled += d
        axial += float(np.hypot(*(p[-1] - p[0])))
        seconds += d / max(seg.speed_um_s, 1e-6)
    return {"axial_um": axial, "path_um": travelled,
            "ratio": (travelled / axial) if axial else 1.0,
            "seconds": seconds}


PENS: dict[str, Pen] = {
    p.key: p for p in [
        Pen("write", "Write path", "plain conducting wire at constant bias",
            {"voltage": 8.0, "speed_um_s": 2.0}, _render_write),
        Pen("region", "Region fill",
            "a closed shape written at one bias; sweep it in a direction, "
            "follow its own outline inward, or cover it with a space-filling "
            "curve (one pass, no preferred direction)",
            # 10 nm line separation by default (2026-08-08), and it
            # is the separation exactly, on every fill kind here.
            {"voltage": 10.0, "speed_um_s": 1.0, "pitch_um": 0.01,
             "fill": "directional", "angle_deg": 0.0, "shape": "as-drawn"},
            _render_region,
            choices={
                "fill": [("directional", "along an axis"),
                         ("hilbert", "space-filling curve"),
                         ("radial-out", "radial: outward"),
                         ("radial-in", "radial: inward"),
                         ("contour", "contour (outline inward)")],
                "shape": [("as-drawn", "as drawn"), ("rect", "bounding rect")],
            }),
        Pen("erase", "Erase path", "negative bias, removes written material",
            {"voltage": 8.0, "speed_um_s": 1.5}, _render_erase),
        Pen("superlattice", "Superlattice",
            "periodic potential: bias alternates along the wire",
            {"period_um": 0.5, "duty": 0.5, "v_hi": 8.0, "v_lo": 2.0,
             "speed_um_s": 1.5}, _render_superlattice),
        Pen("diode", "Diode array",
            "asymmetric ratchet bias: forward and reverse differ",
            {"period_um": 0.8, "asymmetry": 0.75, "v_min": 1.0, "v_max": 9.0,
             "speed_um_s": 1.5}, _render_diode),
        Pen("chiral", "Chiral wire",
            "lateral + voltage modulation in quadrature: voltage is the "
            "third dimension (doi:10.1126/sciadv.adx4761)",
            # Sayanwita's current KTO Helix regime: tall, tight helices,
            # amplitude well above the pitch. The paper's Device A
            # (10 nm / 10 nm, V0 = 10, Vk = 5) is the other preset.
            {"handedness": "R", "lambda_um": 0.025, "y_amp_um": 0.055,
             "v0": 10.0, "v_k": 5.0, "phase_deg": 90.0, "speed_um_s": 1.5},
            _render_chiral,
            choices={"handedness": [("R", "R (right)"), ("L", "L (left)")]}),
    ]
}


# Named starting points for the chiral pen. Presets, not limits: every value
# stays editable, which is the point for a parameter study.
CHIRAL_PRESETS = {
    "kto-helix": {
        "label": "KTO Helix (current)",
        "hint": "tall tight helices, amplitude well above pitch",
        "params": {"lambda_um": 0.025, "y_amp_um": 0.055, "v0": 10.0,
                   "v_k": 5.0, "phase_deg": 90.0},
    },
    "device-a": {
        "label": "Device A (published)",
        "hint": "the geometry of doi:10.1126/sciadv.adx4761",
        "params": {"lambda_um": 0.010, "y_amp_um": 0.010, "v0": 10.0,
                   "v_k": 5.0, "phase_deg": 90.0},
    },
    "achiral-control": {
        "label": "Achiral control",
        "hint": "same geometry and mean bias, phi = 0: the null experiment",
        "params": {"lambda_um": 0.025, "y_amp_um": 0.055, "v0": 10.0,
                   "v_k": 5.0, "phase_deg": 0.0},
    },
}


def render_stroke(points_um, pen_key: str, params: dict | None = None,
                  mode: str = "clean", name: str | None = None
                  ) -> list[LithoObject]:
    """Stroke + pen -> lithography objects, ready to compile and write."""
    pen = PENS[pen_key]
    p = dict(pen.params)
    p.update(params or {})
    path = interpret(points_um, mode)
    if len(path) < 2:
        return []
    # The stroke as it arrived, and how it was interpreted. `interpret`
    # resamples even in literal mode -- deliberately, because raw pointer
    # samples are clumped and every pen computes a pitch or a period from
    # arc length. But a shape PLACED by the shape tools is exact already:
    # its four corners come back as four hundred points with the corners
    # cut by a fraction of a nanometre, and once the document holds the
    # outline rather than the fill, that is the outline you would then be
    # editing. Only the region pen reads these, and only in literal mode.
    p["_stroke_um"] = points_um
    p["_mode"] = mode
    return pen.render(path, p, name or pen_key)


def palette() -> list[dict]:
    return [p.describe() for p in PENS.values()]
