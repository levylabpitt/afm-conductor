"""Pens: sketch strokes into lithography, including the chiral family.

The CISS use case needs two things to be true: the pens must be fully
parametrized (so an agent can sweep them), and left- and right-handed
chiral wires must be genuine enantiomers rather than the same structure
relabelled.
"""

import math

import numpy as np
import pytest

from afm_litho.litho.compiler import LithoObject, compile_object
from afm_litho.litho.pens import (
    PEAK_TOL_UM,
    PENS,
    bake_chiral,
    chiral_spacing_um,
    interpret,
    palette,
    render_stroke,
)


def _wobbly_line(n=200, seed=0):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, n)
    return np.column_stack([-5 + 10 * t + 0.03 * rng.randn(n),
                            0.02 * rng.randn(n)])


def test_palette_is_parametrized():
    keys = {p["key"] for p in palette()}
    assert {"write", "erase", "superlattice", "diode", "chiral"} <= keys
    for p in palette():
        assert p["params"], f"{p['key']} exposes no parameters"


def test_idealize_straightens_a_shaky_line():
    p = interpret(_wobbly_line(), "ideal")
    # residual from the straight line through the endpoints
    d = p[-1] - p[0]
    n = np.array([-d[1], d[0]]) / np.hypot(*d)
    resid = np.abs((p - p[0]) @ n)
    assert resid.max() < 1e-6


def test_superlattice_and_diode_write_bias_profiles():
    s = _wobbly_line()
    sl = render_stroke(s, "superlattice", {"period_um": 0.5, "v_hi": 8, "v_lo": 2})[0]
    v = np.array(sl.v_profile)
    assert set(np.unique(np.round(v, 3))) == {2.0, 8.0}      # two levels

    di = render_stroke(s, "diode", {"period_um": 1.0, "asymmetry": 0.8,
                                    "v_min": 1, "v_max": 9})[0]
    w = np.array(di.v_profile)
    rising = np.sum(np.diff(w) > 0)
    falling = np.sum(np.diff(w) < 0)
    assert rising > 2 * falling      # slow rise, fast fall: a ratchet


def _curve3d(obj):
    """The physical object: lateral position and voltage along the wire.

    Voltage is the third dimension (doi:10.1126/sciadv.adx4761), so the
    structure to reason about is the curve (x, y, V), not the tip path.
    Built through the compiler, which is what the controller actually
    receives, so geometry and bias are paired exactly as they will be
    written.
    """
    c = compile_object(obj, 7e-6, 7e-6)
    # the path points: the stroke's last sample (the last position again, at 0 V)
    # ends the write and is not part of the written curve
    n = c["path_points"]
    p = c["path_um"][:n]
    return np.column_stack([p[:, 0], p[:, 1], c["v"][:n]])


def _planarity(c):
    """Smallest singular value / largest: 0 means the curve lies in a plane."""
    c = c - c.mean(axis=0)
    c = c / np.abs(c).max(axis=0)          # equalise units (um vs volts)
    sv = np.linalg.svd(c, compute_uv=False)
    return sv[2] / sv[0]


def _straight_stroke(n=400):
    return np.column_stack([np.linspace(-0.5, 0.5, n), np.zeros(n)])


def test_chirality_comes_from_the_voltage_phase():
    """phi = +-90 deg makes the (x, y, V) curve a helix; phi = 0 or 180
    leaves it planar, which is the achiral control."""
    s = _straight_stroke()
    chiral = _curve3d(render_stroke(s, "chiral", {"phase_deg": 90})[0])
    flat0 = _curve3d(render_stroke(s, "chiral", {"phase_deg": 0})[0])
    flat180 = _curve3d(render_stroke(s, "chiral", {"phase_deg": 180})[0])
    assert _planarity(chiral) > 0.1, "quadrature modulation must be non-planar"
    assert _planarity(flat0) < 0.02, "in-phase modulation must stay planar"
    assert _planarity(flat180) < 0.02, "anti-phase modulation must stay planar"


def test_hands_are_enantiomers_through_the_voltage_phase():
    """L and R are mirror images.

    Reflecting the structure through the plane that contains the wire axis
    and the voltage direction sends y -> -y; re-indexing by half a period
    restores the lateral term and inverts the bias modulation about V0. So
    the enantiomer of (y_k sin, V0 + V_k sin(.+phi)) is
    (y_k sin, V0 - V_k sin(.+phi)), which for phi = +-90 deg is exactly what
    flipping the sign of phi produces. That is how handedness is switched.
    """
    s = _straight_stroke()
    pen = next(p for p in palette() if p["key"] == "chiral")
    v0 = pen["params"]["v0"]
    R = _curve3d(render_stroke(s, "chiral", {"handedness": "R"})[0])
    L = _curve3d(render_stroke(s, "chiral", {"handedness": "L"})[0])

    assert np.allclose(R[:, 0], L[:, 0], atol=1e-6)      # same axis
    assert np.allclose(R[:, 1], L[:, 1], atol=1e-6)      # same lateral track
    assert np.allclose(L[:, 2], 2 * v0 - R[:, 2], atol=1e-6)  # bias mirrored
    assert np.abs(R[:, 2] - v0).max() > 1.0              # modulation present
    assert _planarity(R) > 0.1 and _planarity(L) > 0.1   # both chiral


def test_chiral_presets_cover_the_real_regimes():
    """Device A as published, the KTO Helix regime in the lab today, and the
    achiral control that differs only in phi."""
    from afm_litho.litho.pens import CHIRAL_PRESETS

    a = CHIRAL_PRESETS["device-a"]["params"]
    assert a["lambda_um"] == pytest.approx(0.010)   # 10 nm
    assert a["y_amp_um"] == pytest.approx(0.010)    # 10 nm
    assert a["v0"] == pytest.approx(10.0) and a["v_k"] == pytest.approx(5.0)

    k = CHIRAL_PRESETS["kto-helix"]["params"]
    assert k["lambda_um"] == pytest.approx(0.025)   # 25 nm pitch
    assert 0.050 <= k["y_amp_um"] <= 0.060          # 50-60 nm amplitude
    assert k["y_amp_um"] > k["lambda_um"]           # amplitude beats pitch

    c = CHIRAL_PRESETS["achiral-control"]["params"]
    assert c["phase_deg"] == 0                      # the only difference
    for key in ("lambda_um", "y_amp_um", "v0", "v_k"):
        assert c[key] == k[key], "control must match the chiral geometry"


def test_high_amplitude_regime_is_finely_sampled():
    """A ~ 55 nm with a 25 nm pitch makes the path ~9x its own axis and the
    slope ~14, so axial sampling would deliver a coarse polygon. The tip
    path must stay smooth in the units that matter: nanometres travelled."""
    from afm_litho.litho.pens import CHIRAL_PRESETS, path_report

    s = _straight_stroke()
    objs = render_stroke(s, "chiral", CHIRAL_PRESETS["kto-helix"]["params"])
    p = np.asarray(objs[0].points_um)
    facet_nm = np.hypot(*np.diff(p, axis=0).T).max() * 1000
    assert facet_nm < 6.0, f"tip path faceted at {facet_nm:.1f} nm"

    r = path_report(objs)
    assert r["ratio"] > 5.0        # the tip really does travel much further
    assert r["seconds"] > 0


def test_voltage_stays_in_phase_with_the_geometry():
    """The wiggly path is longer than its axis; if the bias were spread over
    arc length without correction it would drift out of phase, which at a
    10 nm wavelength destroys the effect."""
    s = _straight_stroke()
    o = render_stroke(s, "chiral", {"lambda_um": 0.02, "y_amp_um": 0.02})[0]
    c = _curve3d(o)
    y = c[:, 1] - c[:, 1].mean()
    v = c[:, 2] - c[:, 2].mean()
    # quadrature: correlation between y and V should be near zero, and the
    # correlation between y and V shifted by a quarter period near maximal
    corr = float(np.corrcoef(y, v)[0, 1])
    assert abs(corr) < 0.25, f"lateral and bias drifted into phase (r={corr:.2f})"


def test_chiral_bias_modulation_is_handed():
    s = _wobbly_line()
    kw = {"v_mod": 2.0, "voltage": 8.0, "pitch_um": 0.6}
    R = render_stroke(s, "chiral", {**kw, "handedness": "R"})[0]
    L = render_stroke(s, "chiral", {**kw, "handedness": "L"})[0]
    vr, vl = np.array(R.v_profile), np.array(L.v_profile)
    assert not np.allclose(vr, vl)                  # phase flips with the hand
    assert vr.max() > 9.0 and vr.min() < 7.0        # modulation present


def test_pen_objects_compile_to_waves():
    s = _wobbly_line()
    for key in PENS:
        for o in render_stroke(s, key):
            c = compile_object(o, 7e-6, 7e-6)
            assert len(c["x_v"]) == len(c["v"]) > 1
            assert np.all(np.isfinite(c["v"]))


@pytest.mark.parametrize("lam", [0.01, 0.02, 0.05])
def test_chiral_wavelength_sets_the_period_count(lam):
    """Parameter sweeps behave as written: an agent can trust them."""
    s = _straight_stroke()
    c = _curve3d(render_stroke(s, "chiral", {"lambda_um": lam})[0])
    v = c[:, 2] - c[:, 2].mean()
    # count zero crossings of the bias modulation: two per period
    crossings = np.sum(np.diff(np.sign(v)) != 0)
    assert crossings / 2 == pytest.approx(1.0 / lam, rel=0.02)


def _periods(a):
    """Half-cycle crossings of a modulation, in periods."""
    sign = np.sign(np.asarray(a) - np.mean(a))
    return float(np.sum(np.diff(sign) != 0)) / 2


@pytest.mark.parametrize("mode", ["literal", "clean", "ideal"])
def test_interpretation_preserves_the_drawn_endpoints(mode):
    """A wire drawn 3 um long must be 3 um long.

    Smoothing with edge replication pulled both ends inward by about half
    the window, so `clean` returned 2.889 um for a 3.000 um stroke. Every
    pen derives its period count from the axis, so the wire quietly came
    out 3.7% short of what was asked for.
    """
    from afm_litho.litho.pens import interpret
    raw = [(0.0, 0.0), (3.0, 0.0)]
    p = np.asarray(interpret(raw, mode))
    assert p[0] == pytest.approx(raw[0], abs=1e-9)
    assert p[-1] == pytest.approx(raw[-1], abs=1e-9)
    length = float(np.hypot(*np.diff(p, axis=0).T).sum())
    assert length == pytest.approx(3.0, rel=1e-6)


def test_modulation_is_commensurate_with_the_serpentine():
    """One bias period per geometric period, exactly: they are the same
    theta. If these ever disagree the wire is not the helix it claims."""
    o = render_stroke([(0.0, 0.0), (3.0, 0.0)], "chiral",
                      {"lambda_um": 0.025, "y_amp_um": 0.055})[0]
    pts = np.asarray(o.points_um)
    geom = _periods(pts[:, 1])
    volts = _periods(o.v_profile)
    assert geom == pytest.approx(3.0 / 0.025, rel=0.01)
    assert volts == pytest.approx(geom, rel=0.01)


def test_display_sampling_resolves_the_modulation():
    """What the canvas is sent must not alias.

    The bias was decimated to a flat 160 samples regardless of length: for
    120 periods that is 1.33 samples per period, so the canvas drew 44 slow
    colour bands across a wire modulated 120 times. Geometry and bias now
    share one sampling, and it stays above Nyquist by a wide margin.
    """
    from afm_litho.litho.compiler import Pattern
    o = render_stroke([(0.0, 0.0), (3.0, 0.0)], "chiral",
                      {"lambda_um": 0.025, "y_amp_um": 0.055})[0]
    d = Pattern("t", [o]).to_dict()["objects"][0]
    pts = np.asarray(d["points_um"])
    v = np.asarray(d["v_profile"])
    assert len(v) == len(pts)                      # one bias per point
    assert _periods(v) == pytest.approx(3.0 / 0.025, rel=0.02)
    assert len(v) / _periods(v) > 8                # comfortably above Nyquist


@pytest.mark.parametrize("lam, amp, nm", [
    (0.1, 0.1, 3.2), (0.05, 0.1, 1.6), (0.5, 0.48, 7.3), (0.5, 0.1, 16.0)])
def test_chiral_spacing_holds_the_peak_tolerance(lam, amp, nm):
    s = chiral_spacing_um(lam, amp)
    assert s * 1000 == pytest.approx(nm, abs=0.1)
    # a chord centred on a peak cuts it by A(1 - cos(k s / 2))
    assert amp * (1 - math.cos(math.pi * s / lam)) <= PEAK_TOL_UM * (1 + 1e-9)


@pytest.mark.parametrize("lam, want", [(0.1, 0.0125), (0.5, 0.02)])
def test_chiral_spacing_without_a_wiggle_keeps_8_points_per_period(lam, want):
    assert chiral_spacing_um(lam, 0.0) == pytest.approx(want)
    assert chiral_spacing_um(lam, PEAK_TOL_UM) == pytest.approx(want)


def test_a_sine_sampled_at_the_chiral_spacing_keeps_its_peaks():
    lam, amp = 0.1, 0.1
    x = np.linspace(0.0, 1.0, 200_001)
    y = amp * np.sin(2 * np.pi * x / lam)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    si = np.arange(0.0, s[-1], chiral_spacing_um(lam, amp))
    xs, ys = np.interp(si, s, x), np.interp(si, s, y)
    peaks = (np.arange(10) + 0.25) * lam
    cut = amp - np.abs(np.interp(peaks, xs, ys))
    assert cut.max() <= PEAK_TOL_UM


def test_a_compiled_chiral_wire_is_finer_and_still_fires_in_32s():
    lam, amp = 0.1, 0.1
    o = LithoObject("w", "path", [(0.0, 0.0), (1.0, 0.0)], voltage=5.0)
    bake_chiral(o, lam, amp, 1.0, 90.0)
    c = compile_object(o, 7e-6, 7e-6)
    assert len(c["v"]) % 32 == 0 and c["v"][-1] == 0.0
    n = c["path_points"]
    p = c["path_um"][:n]
    peaks = (np.arange(10) + 0.25) * lam
    cut = amp - np.abs(np.interp(peaks, p[:, 0], p[:, 1]))
    # the baked profile is itself a polyline (~2.5 nm vertices here, cutting each
    # peak by up to ~0.3 nm), and the fired points lie on its chords
    assert cut.max() <= 0.001
    o.interpolation_um = min(lam, amp) / 8.0
    assert n > compile_object(o, 7e-6, 7e-6)["path_points"]
