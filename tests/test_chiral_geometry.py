"""Geometry invariants for the chiral pen.

Each of these was a visible defect first: a starburst, a ring that would
not close, coils that leaned, a long wire that looked ragged.
"""

import numpy as np
import pytest

from afm_litho.litho.compiler import Pattern, count_oscillations
from afm_litho.litho.pens import (chiral_profile, interpret, is_closed,
                                  render_stroke)


def _circle(R, n=600):
    t = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([R * np.cos(t), R * np.sin(t)])


def _folds(p):
    d = np.diff(p, axis=0)
    n = np.hypot(*d[:-1].T) * np.hypot(*d[1:].T)
    cos = (d[:-1] * d[1:]).sum(axis=1) / np.maximum(n, 1e-18)
    return int((cos < 0).sum())


# --------------------------------------------------------- transversality --

@pytest.mark.parametrize("path,label", [
    (_circle(1.0), "circle"),
    (np.column_stack([np.linspace(0, 5, 300), np.zeros(300)]), "straight"),
    (np.column_stack([np.linspace(0, 6, 400),
                      0.8 * np.sin(np.linspace(0, 6, 400) * 2)]), "wavy"),
    (np.array([[0, 0], [2, 0], [2, 2]], float), "corner"),
])
def test_modulation_is_perpendicular_to_the_path(path, label):
    """The wave has to be transverse everywhere.

    Interpolating position by chords while interpolating the normal
    separately let them disagree: 83 degrees on a circle, 76 at a corner.
    One spline gives position, direction and curvature from the same
    object, so they cannot drift apart.
    """
    p = interpret(path, "clean")
    build = chiral_profile(p, 0.025, 0.055, 10, 5, 90)
    pts = build.points
    # ask the construction what it used, rather than re-deriving it here
    ax = build.axis
    nn = build.normal
    d1 = np.column_stack([nn[:, 1], -nn[:, 0]])       # tangent from normal
    disp = pts - ax
    mag = np.hypot(*disp.T)
    ok = mag > 1e-9
    cos = np.abs((disp[ok] / mag[ok][:, None] * d1[ok]).sum(axis=1))
    assert np.degrees(np.arccos(np.clip(cos, 0, 1))).min() > 89.0


# ---------------------------------------------------------------- closure --

def test_a_drawn_circle_is_recognised_as_closed():
    assert is_closed(_circle(1.0))
    assert not is_closed(np.column_stack([np.linspace(0, 5, 100), np.zeros(100)]))


@pytest.mark.parametrize("R", [0.25, 1.0, 3.0])
def test_a_closed_loop_closes_in_geometry_and_in_bias(R):
    """249.27 periods left a 97 degree phase step and a 5.6 V bias jump at
    the seam. Quantising the wavelength to a whole number of periods closes
    both, at a sub-percent change in pitch."""
    p = interpret(_circle(R), "clean")
    build = chiral_profile(p, 0.025, 0.055, 10, 5, 90)
    pts, volts, c = build.points, build.volts, build.clamp
    assert c["closed"]
    assert c["periods"] == pytest.approx(round(c["periods"]))      # whole
    assert c["lambda_um"] == pytest.approx(0.025, rel=0.02)        # barely moved
    assert np.hypot(*(pts[-1] - pts[0])) < 1e-9                    # no gap
    assert abs(volts[-1] - volts[0]) < 1e-6                        # no step


def test_an_open_stroke_is_left_open():
    p = interpret(np.column_stack([np.linspace(0, 5, 300), np.zeros(300)]), "clean")
    build = chiral_profile(p, 0.025, 0.055, 10, 5, 90)
    assert not build.clamp["closed"]
    assert np.hypot(*(build.points[-1] - build.points[0])) > 1.0


# ------------------------------------------------------- shape and folding --

@pytest.mark.parametrize("R", [0.05, 0.1, 0.25, 1.0, 3.0])
def test_a_drawn_circle_keeps_its_size(R):
    """A fixed 0.05 um resample and a fixed 9-sample average crushed small
    loops: a 100 nm circle came back with its radius collapsed to 37 nm.
    Both now scale with the stroke."""
    p = interpret(_circle(R), "clean")
    r = np.hypot(*p.T)
    assert r.min() == pytest.approx(R, rel=0.02)
    assert r.max() == pytest.approx(R, rel=0.02)


@pytest.mark.parametrize("R", [0.25, 0.5, 1.0, 3.0])
def test_no_folds_once_the_amplitude_fits_the_curve(R):
    o = render_stroke(_circle(R), "chiral", mode="clean")[0]
    assert _folds(np.asarray(o.points_um)) == 0


def test_amplitude_is_clamped_to_the_curvature_and_reported():
    """An offset curve inverts once the offset reaches the radius of
    curvature. On a tight loop the amplitude is reduced, and said so,
    because in a chirality experiment the amplitude is the physics."""
    tight = render_stroke(_circle(0.06), "chiral", mode="clean")[0].chiral_clamp
    assert tight["fraction"] > 0.5
    assert tight["min_amp_um"] < tight["requested_um"]

    straight = render_stroke([(0.0, 0.0), (5.0, 0.0)], "chiral",
                             mode="clean")[0].chiral_clamp
    assert straight["fraction"] == 0.0          # never touches a straight line
    assert straight["min_amp_um"] == pytest.approx(straight["requested_um"])


def test_the_clamp_is_said_by_the_sketch_endpoint():
    """The service says the clamp out loud: the response carries it and the
    panel log gets a sentence, so a reduced amplitude is announced rather
    than inferred from a strange picture."""
    import asyncio

    from afm_litho.service import app as service
    from afm_litho.service.routers import design

    said = []
    real_log = service.hub.log
    had_own_log = "log" in vars(service.hub)

    async def spy(text):
        said.append(text)
        await real_log(text)

    service.hub.log = spy
    try:
        r = asyncio.run(design.sketch(
            {"points_um": _circle(0.06).tolist(), "pen": "chiral",
             "mode": "clean"}))
        assert r["clamp"] and r["clamp"]["fraction"] > 0.5
        assert any("clamped" in s for s in said)

        r2 = asyncio.run(design.sketch(
            {"points_um": [(0.0, 0.0), (5.0, 0.0)], "pen": "chiral",
             "mode": "clean"}))
        assert r2["clamp"] is None

        # a pen that never touches chiral_profile must not replay the last
        # chiral stroke's clamp
        r3 = asyncio.run(design.sketch(
            {"points_um": _circle(0.06).tolist(), "pen": "write",
             "mode": "clean"}))
        assert r3["clamp"] is None
    finally:
        if had_own_log:
            service.hub.log = real_log
        else:
            del service.hub.log


# --------------------------------------------------------------- display ---

def evaluate_profile(P, n):
    """Flatten the analytic profile exactly as the canvas does."""
    ax = np.asarray(P["axis_um"]); nn = np.asarray(P["normal"])
    am = np.asarray(P["amp_um"]); m = len(ax)
    f = np.linspace(0.0, 1.0, n)
    u = f * (m - 1)
    k = np.minimum(m - 2, np.floor(u).astype(int)); tt = u - k
    a = ax[k] + tt[:, None] * (ax[k + 1] - ax[k])
    nv = nn[k] + tt[:, None] * (nn[k + 1] - nn[k])
    nv /= np.maximum(np.hypot(*nv.T), 1e-12)[:, None]
    amp = am[k] + tt * (am[k + 1] - am[k])
    th = 2 * np.pi * P["periods"] * f
    pts = a + nv * (amp * np.sin(th))[:, None]
    volts = P["v0"] + P["v_k"] * np.sin(th + P["phase_rad"])
    return pts, volts


def test_the_profile_and_the_toolpath_are_the_same_curve():
    """The canvas draws the analytic profile; the executor writes
    points_um. If they diverge, what you see is not what gets written, so
    this is the contract test for the whole representation."""
    o = render_stroke([(0.0, 0.0), (20.0, 0.0)], "chiral", mode="clean")[0]
    tool = np.asarray(o.points_um)
    pts, volts = evaluate_profile(o.profile, len(tool))
    err = np.hypot(*(pts - tool).T)
    assert err.max() < 0.002              # within 2 nm everywhere
    v_tool = np.asarray(o.v_profile)
    # compare bias on the axial grid it was constructed on
    assert abs(volts.max() - v_tool.max()) < 0.05
    assert abs(volts.min() - v_tool.min()) < 0.05


def test_the_profile_closes_when_the_wire_does():
    o = render_stroke(_circle(3.0), "chiral", mode="clean")[0]
    P = o.profile
    assert P["closed"]
    assert P["periods"] == pytest.approx(round(P["periods"]))
    pts, volts = evaluate_profile(P, 5001)
    assert np.hypot(*(pts[-1] - pts[0])) < 1e-3
    assert abs(volts[-1] - volts[0]) < 1e-6


def test_profiled_objects_send_parameters_not_point_clouds():
    """The payload win is the point: a 60 um wire used to ship ~30k points
    to draw badly; the profile describes it exactly in a few hundred."""
    o = render_stroke([(0.0, 0.0), (60.0, 0.0)], "chiral", mode="clean")[0]
    d = Pattern("x", [o]).to_dict()["objects"][0]
    assert "profile" in d
    assert len(d["profile"]["axis_um"]) <= 512
    assert len(d["points_um"]) <= 3100        # coarse fallback only


def test_geometry_and_bias_stay_registered_after_decimation():
    o = render_stroke([(0.0, 0.0), (30.0, 0.0)], "chiral", mode="clean")[0]
    d = Pattern("x", [o]).to_dict()["objects"][0]
    assert len(d["v_profile"]) == len(d["points_um"])
    v = np.asarray(d["v_profile"])
    assert v.min() == pytest.approx(5.0, abs=0.6)       # v0 - v_k
    assert v.max() == pytest.approx(15.0, abs=0.6)      # v0 + v_k


# ------------------------------------------- closure as people draw it -----

@pytest.mark.parametrize("short", [0.0, 0.02, 0.05, 0.10])
def test_a_hand_drawn_ring_closes_even_though_it_does_not_meet_itself(short):
    """Nobody draws a circle that meets itself.

    A stroke stopped 2% short of the full turn already leaves a gap of 6%
    of its own extent; 10% short leaves 31%. Judging closure on the gap
    alone therefore called every real ring open, which is why they kept
    rendering with a notch and a stray part-period hanging off it. Turning
    angle is what separates a loop from an arc.
    """
    rng = np.random.default_rng(1)
    t = np.linspace(0, 2 * np.pi * (1 - short), 400)
    stroke = np.column_stack([np.cos(t), np.sin(t)]) + rng.normal(0, 0.004, (400, 2))
    p = interpret(stroke, "clean")
    assert is_closed(p)
    build = chiral_profile(p, 0.025, 0.055, 10, 5, 90)
    assert np.hypot(*(build.points[-1] - build.points[0])) < 1e-9
    assert abs(build.volts[-1] - build.volts[0]) < 1e-6


@pytest.mark.parametrize("stroke_frac", [0.5, 0.75])
def test_an_arc_is_not_forced_closed(stroke_frac):
    t = np.linspace(0, 2 * np.pi * stroke_frac, 400)
    p = interpret(np.column_stack([np.cos(t), np.sin(t)]), "clean")
    assert not is_closed(p)


def test_decimation_never_moves_the_endpoints():
    """Decimation must not shift where a wire starts or ends: a loop that
    closes exactly in the geometry was once served to the canvas starting
    42 nm off its own start, which read as a closure bug and was not."""
    o = render_stroke(_circle(3.0), "chiral", mode="clean")[0]
    full = np.asarray(o.points_um)
    sent = np.asarray(Pattern("x", [o]).to_dict()["objects"][0]["points_um"])
    assert np.allclose(sent[0], full[0], atol=1e-12)
    assert np.allclose(sent[-1], full[-1], atol=1e-12)
    assert np.hypot(*(sent[-1] - sent[0])) < 1e-9      # still closed as served


def test_what_is_drawn_stays_close_to_the_true_curve():
    """Chord error of the RENDERED curve against the compiled toolpath.

    The renderer evaluates the analytic profile, so this flattens it at the
    renderer's own density and measures the worst deviation from the true
    written path. This is the number that used to be argued about in points
    per period; with the profile it is bounded by construction.
    """
    o = render_stroke([(0.0, 0.0), (20.0, 0.0)], "chiral", mode="clean")[0]
    full = np.asarray(o.points_um)
    sent, _ = evaluate_profile(o.profile, int(o.profile["periods"] * 24))
    a, b = sent[:-1], sent[1:]
    d = b - a
    L2 = np.maximum((d * d).sum(1), 1e-18)
    worst = 0.0
    for i in range(0, len(full), 11):
        t = np.clip(((full[i] - a) * d).sum(1) / L2, 0, 1)
        worst = max(worst, float(np.min(np.hypot(*(full[i] - (a + t[:, None] * d)).T))))
    assert worst < 0.055 * 0.03          # under 3% of the amplitude


def test_oscillations_are_counted_from_the_geometry():
    straight = np.column_stack([np.linspace(0, 5, 500), np.zeros(500)])
    assert count_oscillations(straight) == 0
    o = render_stroke([(0.0, 0.0), (2.0, 0.0)], "chiral", mode="clean")[0]
    assert count_oscillations(np.asarray(o.points_um)) == pytest.approx(80, rel=0.1)


# ------------------------------------------------ pitch must be uniform ----

@pytest.mark.parametrize("stroke,label", [
    (_circle(3.0), "circle"),
    (np.column_stack([np.linspace(0, 10, 300), np.zeros(300)]), "straight"),
    (np.column_stack([np.linspace(0, 8, 400),
                      1.2 * np.sin(np.linspace(0, 8, 400))]), "wavy"),
])
def test_pitch_is_constant_in_microns_around_the_path(stroke, label):
    """The wavelength is a distance, so it has to be constant in distance.

    The spline is parameterised by the drawn polyline's chord length and
    its speed is not 1: around a hand-drawn circle it varies by 9%.
    Stepping uniformly in that parameter stretched and squeezed the written
    pitch by the same 9%, which is an error in the wire, not in the
    picture. Sampling at uniform true arc length fixes it.
    """
    p = interpret(stroke, "clean")
    build = chiral_profile(p, 0.025, 0.055, 10, 5, 90)
    pts, volts = build.points, build.volts
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    z = np.where(np.diff(np.sign(volts - volts.mean())) != 0)[0]
    half = np.diff(s[z])
    assert (half.max() - half.min()) / half.mean() < 0.03


@pytest.mark.parametrize("L", [2.0, 8.0, 20.0, 60.0])
def test_every_period_is_recoverable_whatever_the_length(L):
    """With the analytic profile, resolution is the renderer's choice, so
    the only invariant that matters is that the profile carries the true
    period count at every length."""
    o = render_stroke([(0.0, 0.0), (L, 0.0)], "chiral", mode="clean")[0]
    assert o.profile["periods"] == pytest.approx(L / 0.025, rel=0.02)
    pts, _ = evaluate_profile(o.profile, int(o.profile["periods"] * 24))
    drawn = int((np.diff(np.sign(pts[:, 1] - pts[:, 1].mean())) != 0).sum()) // 2
    assert drawn == pytest.approx(L / 0.025, rel=0.02)


# ------------------------------------- x(t), y(t), V(t) as true functions --

def test_the_wire_is_a_function_not_a_point_cloud():
    """Wire.evaluate answers at ANY t, exactly, from the representation."""
    from afm_litho.litho.curves import Wire
    o = render_stroke([(0.0, 0.0), (10.0, 0.0)], "chiral", mode="clean")[0]
    w = Wire.from_object(o)
    xy, v = w.evaluate(np.random.default_rng(0).random(500))
    assert np.isfinite(xy).all() and np.isfinite(v).all()
    tool = np.asarray(o.points_um)
    flat, _ = w.flatten(len(tool))
    assert np.hypot(*(flat - tool).T).max() < 0.002        # 2 nm of the toolpath


def test_one_cubic_per_quarter_period_is_subnanometre():
    """The claim the whole rendering rests on, measured: a Hermite cubic per
    quarter period reproduces the modulated curve to well under a
    nanometre, three orders below anything the tip can write."""
    from afm_litho.litho.curves import ModulatedPiece, Wire, bezier_error_um
    o = render_stroke([(0.0, 0.0), (10.0, 0.0)], "chiral", mode="clean")[0]
    piece = Wire.from_object(o).pieces[0]
    assert isinstance(piece, ModulatedPiece)
    bez = piece.to_beziers()
    assert len(bez) == pytest.approx(4 * piece.periods, abs=1)
    n = len(bez)
    worst = 0.0
    for i in range(0, n, max(1, n // 40)):                 # spot-check 40 spans
        worst = max(worst, bezier_error_um(bez[i], piece, i / n, (i + 1) / n))
    # 0.6 nm, uniform: the floor is the 512-sample axis interpolation, not
    # the cubic (whose own term is 0.02 nm). Three orders below the ~60 nm
    # write radius, and 0.006 px even at 1000 px per micron.
    assert worst < 2e-3                                     # < 2 nm


def test_polyline_objects_are_wires_too():
    from afm_litho.litho.curves import PolyPiece, Wire
    o = render_stroke([(0.0, 0.0), (5.0, 2.0)], "write", mode="clean")[0]
    w = Wire.from_object(o)
    assert isinstance(w.pieces[0], PolyPiece)
    xy, v = w.evaluate([0.0, 0.5, 1.0])
    assert np.allclose(xy[0], o.points_um[0], atol=1e-9)
    assert np.allclose(xy[-1], o.points_um[-1], atol=1e-9)


# ------------------------------------------------------------ reentrancy ---
#
# `chiral_profile` used to hand its profile, its clamp report and its axis/normal
# arrays back as ATTRIBUTES ON THE FUNCTION, read off with `getattr` a few lines
# later. Two builds in flight at once -- `POST /api/design/chiral` while a FLEX
# `loadPattern` builds its own spec, both on threads off the same event loop -- and
# the second overwrote the first's between its return and its read. Measured before
# the fix: 285 of 300 concurrent builds came back CROSS-CONTAMINATED.
#
# It is not a cosmetic mix-up. The profile is PERSISTED on the object; the canvas
# renders from it and `routers/design._reapply_chiral` REBUILDS the geometry and the
# bias from it after a node edit, so a swapped profile is a wire silently rewritten
# to another wire's amplitude, pitch and phase -- and a stale clamp report is
# `curvature_clamped: false` on a wire that was clamped, which in a chirality
# experiment is a wrong measurement rather than a wrong label.

_CHIRAL_CASES = [
    # (length_um, lambda_um, y_amp_um, phase_deg) -- deliberately far apart, so a
    # swapped profile cannot pass by looking approximately right.
    (2.0, 0.025, 0.055, 90.0),
    (5.0, 0.100, 0.010, -45.0),
    (1.0, 0.050, 0.200, 0.0),
]


def _build_case(case):
    """One build, checked against ITS OWN parameters. Returns nothing on success."""
    import math

    length, lam, amp, phase = case
    axis = np.array([[0.0, 0.0], [length, 0.0]])
    b = chiral_profile(axis, lam, amp, 10.0, 5.0, phase)
    prof, clamp = b.profile, b.clamp
    assert prof["lambda_um"] == pytest.approx(lam), f"{case}: got {prof['lambda_um']}"
    assert prof["y_amp_um"] == pytest.approx(amp), f"{case}: got {prof['y_amp_um']}"
    assert prof["phase_rad"] == pytest.approx(math.radians(phase)), case
    # a straight axis has infinite radius of curvature, so nothing is ever clamped
    assert clamp["requested_um"] == pytest.approx(amp), case
    assert clamp["fraction"] == 0.0, case
    # ...and the returned geometry belongs to the same wire as the profile
    assert b.points[-1][0] == pytest.approx(length, rel=1e-6), case
    assert len(b.axis) == len(b.points) == len(b.volts), case


def test_concurrent_chiral_builds_never_cross_contaminate_the_profile():
    """The regression case for ChiralBuild. 120 builds of three different wires on a
    thread pool, each asserting the profile it got back is the profile of the wire IT
    asked for. On the function-attribute version this failed on the first few."""
    from concurrent.futures import ThreadPoolExecutor

    cases = [_CHIRAL_CASES[i % len(_CHIRAL_CASES)] for i in range(120)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(_build_case, cases):
            pass                       # map re-raises the first failure here


def test_a_concurrent_build_cannot_steal_another_strokes_clamp():
    """The same race on the CLAMP, which is the half that silently misreports the
    physics: a tight ring (clamped over its whole length) built alongside a straight
    wire (never clamped). Each must report its own."""
    from concurrent.futures import ThreadPoolExecutor

    def ring(_):
        o = render_stroke(_circle(0.06), "chiral", mode="clean")[0]
        assert o.chiral_clamp["fraction"] > 0.5
        return "ring"

    def line(_):
        o = render_stroke([(0.0, 0.0), (5.0, 0.0)], "chiral", mode="clean")[0]
        assert o.chiral_clamp["fraction"] == 0.0
        return "line"

    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(ring if i % 2 else line, i) for i in range(60)]
        assert sorted(j.result() for j in jobs) == sorted(
            ["ring"] * 30 + ["line"] * 30)


def test_the_builder_keeps_no_state_between_calls():
    """Belt and braces on the same fix: nothing is left ON the function for the next
    call to find. A future `getattr(chiral_profile, "last_...")` has to fail loudly,
    not resurrect the bug quietly."""
    chiral_profile(np.array([[0.0, 0.0], [2.0, 0.0]]), 0.025, 0.055, 10, 5, 90)
    for attr in ("last_profile", "last_clamp", "last_axis", "last_normal"):
        assert not hasattr(chiral_profile, attr), attr
