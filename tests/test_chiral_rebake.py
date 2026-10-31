"""A chiral wire is a spine plus chiral settings; the baked points, bias and profile
are derived from them with the object's voltage as v0.

Every door that changes the voltage or the spine has to re-apply the settings:
`set_props` voltage, a node edit, a transform, a duplicate, and the write that
follows. FLEX `setObjects` is covered in tests/test_flex_litho.py.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import pytest
from _im_fixtures_guard import GUARD_WRITE_KW, make_model, seed_free_air

from afm_litho.litho import pathops
from afm_litho.litho.compiler import LithoObject, Pattern
from afm_litho.litho.pens import bake_chiral, rebake_chiral

# `client` comes from tests/conftest.py.

LAM, AMP, V_K, PHASE = 0.025, 0.055, 5.0, 90.0
AXIS = [(5.0, 5.0), (6.0, 5.0)]
# What a fresh bake measures between the cockpit's analytic drawing and the baked
# points (profile axis rounded to 1e-5 um): 0.0049 nm.
DRAW_TOL_NM = 0.005


# ------------------------------------------------------------------ helpers --

@pytest.fixture(autouse=True)
def _own_undo_history():
    """The undo history is a process global other test files leave full; every test
    here starts from an empty one and puts the old one back."""
    from afm_litho.service.routers import design
    saved = list(design._design_history)
    design._design_history.clear()
    yield
    design._design_history[:] = saved


def _wire(axis=AXIS, v0=8.0, name="w", speed=2.0) -> LithoObject:
    o = LithoObject(name, "path", list(axis), voltage=v0, speed_um_s=speed)
    bake_chiral(o, LAM, AMP, V_K, PHASE)
    return o


def _seed(*objs):
    from afm_litho.service.hub import hub
    hub.pattern = Pattern(name="seed", objects=list(objs))
    return hub


def _plain(axis=AXIS, v0=8.0, name="w", speed=2.0) -> LithoObject:
    return LithoObject(name, "path", list(axis), voltage=v0, speed_um_s=speed)


def _chiral_on(client, v0=8.0, v_k=V_K, index=0):
    r = client.post("/api/design/chiral", json={
        "indices": [index], "enabled": True, "lambda_um": LAM, "y_amp_um": AMP,
        "v_k": v_k, "phase_deg": PHASE, "v0": v0}).json()
    assert r.get("ok"), r
    return r


def _op(client, **body) -> dict:
    return client.post("/api/design/op", json=body).json()


def _pitch_amp(points, a0, a1):
    """Pitch (nm), peak offset (nm) and upward zero crossings of `points` about the
    straight spine a0 -> a1."""
    p = np.asarray(points, dtype=float)
    a0 = np.asarray(a0, dtype=float)
    d = np.asarray(a1, dtype=float) - a0
    t = d / np.hypot(*d)
    n = np.array([-t[1], t[0]])
    s = (p - a0) @ t
    off = (p - a0) @ n
    up = np.where((off[:-1] < 0) & (off[1:] >= 0))[0]
    xs = [s[i] - off[i] * (s[i + 1] - s[i]) / (off[i + 1] - off[i]) for i in up]
    return 1000 * float(np.mean(np.diff(xs))), 1000 * float(np.abs(off).max()), len(xs)


def _drawn(o) -> np.ndarray:
    """What the cockpit's drawChiralWire draws: axis + normal * amp * sin(2 pi N f),
    from the profile alone, evaluated at the baked points' own f."""
    prof = o.profile
    ax = np.asarray(prof["axis_um"], dtype=float)
    nm = np.asarray(prof["normal"], dtype=float)
    am = np.asarray(prof["amp_um"], dtype=float)
    f = np.linspace(0.0, 1.0, len(o.points_um))
    m = len(ax)
    u = f * (m - 1)
    k = np.minimum(m - 2, np.floor(u).astype(int))
    t = (u - k)[:, None]
    a = ax[k] + t * (ax[k + 1] - ax[k])
    nn = nm[k] + t * (nm[k + 1] - nm[k])
    nn /= np.hypot(nn[:, 0], nn[:, 1])[:, None]
    amp = am[k] + t[:, 0] * (am[k + 1] - am[k])
    return a + nn * (amp * np.sin(2 * math.pi * prof["periods"] * f))[:, None]


def _draw_error_nm(o) -> float:
    return 1000 * float(np.abs(_drawn(o) - np.asarray(o.points_um)).max())


def _measured_phase_deg(o) -> float:
    """The phase by which the bias leads the lateral offset, measured on the baked
    wire in its own direction of travel (offset positive to the left). +90 and -90
    are the two enantiomers."""
    p = np.asarray(o.points_um, dtype=float)
    a0 = np.asarray(o.axis_um[0], dtype=float)
    d = np.asarray(o.axis_um[-1], dtype=float) - a0
    t = d / np.hypot(*d)
    if (p[-1] - p[0]) @ t < 0:
        t = -t
    n = np.array([-t[1], t[0]])
    s_axis = (p - p[0]) @ t
    off = (p - a0) @ n
    seg = np.hypot(*np.diff(p, axis=0).T)
    s_path = np.concatenate([[0.0], np.cumsum(seg)])
    v = np.interp(s_path / s_path[-1], np.linspace(0.0, 1.0, len(o.v_profile)),
                  np.asarray(o.v_profile, dtype=float))
    carrier = np.exp(-1j * 2 * math.pi * s_axis / o.profile["lambda_um"])
    lead = np.angle(np.sum((v - o.voltage) * carrier)) - np.angle(np.sum(off * carrier))
    return math.degrees((lead + math.pi) % (2 * math.pi) - math.pi)


def _consistent(o) -> None:
    """Voltage, bias and profile agree, and the profile describes the baked points."""
    assert o.profile["v0"] == pytest.approx(o.voltage)
    # 5 mV: the sample grid need not land on a bias peak.
    assert (min(o.v_profile) + max(o.v_profile)) / 2 == pytest.approx(o.voltage, abs=5e-3)
    assert max(o.v_profile) - min(o.v_profile) == pytest.approx(2 * V_K, abs=5e-3)
    assert _draw_error_nm(o) <= DRAW_TOL_NM


# ------------------------------------------------- bug 1: the voltage edit --

def test_a_voltage_edit_rebakes_the_wire_and_a_node_edit_keeps_it(client):
    hub = _seed(_plain())
    _chiral_on(client, v0=8.0)
    r = _op(client, op="set_props", indices=[0], voltage=6)
    assert r.get("ok"), r
    o = hub.pattern.objects[0]
    assert o.voltage == 6.0
    assert o.profile["v0"] == 6.0
    assert min(o.v_profile) == pytest.approx(1.0, abs=1e-3)
    assert max(o.v_profile) == pytest.approx(11.0, abs=1e-3)
    served = r["design"]["objects"][0]
    assert served["voltage"] == 6.0 and served["profile"]["v0"] == 6.0
    assert max(served["v_profile"]) == pytest.approx(11.0, abs=0.05)
    _consistent(o)

    bez = client.get("/api/design/bezier/0").json()["bezier"]
    r = client.post("/api/design/bezier", json={"index": 0, "bezier": bez}).json()
    assert r.get("ok") and r["reapplied"], r
    o = hub.pattern.objects[0]
    assert o.voltage == 6.0 and o.profile["v0"] == 6.0
    assert max(o.v_profile) == pytest.approx(11.0, abs=1e-3)
    _consistent(o)


def test_a_voltage_edit_keeps_the_geometry_and_the_settings(client):
    hub = _seed(_plain())
    _chiral_on(client, v0=8.0)
    before = copy.deepcopy(hub.pattern.objects[0])
    assert _op(client, op="set_props", indices=[0], voltage=6).get("ok")
    o = hub.pattern.objects[0]
    assert np.allclose(o.points_um, before.points_um, atol=1e-12)
    assert o.axis_um == before.axis_um
    for key in ("lambda_um", "y_amp_um", "v_k", "phase_rad", "periods"):
        assert o.profile[key] == pytest.approx(before.profile[key])


def test_a_twin_write_after_a_voltage_edit_applies_the_new_voltage(client, tmp_path,
                                                                 monkeypatch):
    from afm_litho.service.sessions import SessionStore
    hub = _seed(_plain(speed=400.0))
    monkeypatch.setattr(hub, "store", SessionStore(tmp_path / "s.jsonl"))
    hub.host.written_objects = set()
    _chiral_on(client, v0=8.0)
    assert _op(client, op="set_props", indices=[0], voltage=6).get("ok")

    n0 = len([s for s in hub.store.list() if s["kind"] == "litho"])
    seed_free_air(hub)
    j = client.post("/api/litho/write", json={
        "objects": ["w"], "setpoint_relative": 0.2, **GUARD_WRITE_KW}).json()
    assert j.get("ok") and j["writing"] == 1, j
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if len([s for s in hub.store.list() if s["kind"] == "litho"]) > n0:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("write never completed")
    client.post("/api/withdraw")

    done = [e for e in hub.host.written_accum if e["name"] == "w"]
    assert len(done) == 1, [e["name"] for e in hub.host.written_accum]
    v = np.asarray(done[0]["v_written"], dtype=float)
    assert done[0]["voltage"] == 6.0
    assert v.min() == pytest.approx(1.0, abs=0.05)
    assert v.max() == pytest.approx(11.0, abs=0.05)
    assert v.mean() == pytest.approx(6.0, abs=0.1)


@pytest.mark.parametrize("key,value", [("voltage_end", 2.0), ("v_profile", [1, 2, 3]),
                                       ("v_profile", None)])
def test_voltage_end_and_v_profile_are_refused_on_a_chiral_wire(client, key, value):
    hub = _seed(_plain(), _plain(axis=[(5.0, 7.0), (6.0, 7.0)], name="plain"))
    _chiral_on(client, v0=8.0)
    before = copy.deepcopy(hub.pattern.objects)
    r = _op(client, op="set_props", indices=[0, 1], **{key: value})
    assert not r.get("ok"), r
    assert "'w'" in r["error"] and key in r["error"] and "chiral" in r["error"], r
    assert r["reason"] == "chiral_bias_fixed"
    assert hub.pattern.objects == before


def test_clearing_voltage_end_and_setting_both_on_a_plain_path_still_work(client):
    hub = _seed(_plain(), _plain(axis=[(5.0, 7.0), (6.0, 7.0)], name="plain"))
    _chiral_on(client, v0=8.0)
    assert _op(client, op="set_props", indices=[0], voltage=6, voltage_end=None).get("ok")
    assert hub.pattern.objects[0].profile["v0"] == 6.0
    r = _op(client, op="set_props", indices=[1], voltage_end=2.0, v_profile=[1, 2, 3])
    assert r.get("ok"), r
    plain = hub.pattern.objects[1]
    assert plain.voltage_end == 2.0 and plain.v_profile == [1.0, 2.0, 3.0]


# --------------------------------------------------------------- the guard --

def test_the_guard_judges_the_rebuilt_profile(client):
    hub = _seed(_plain())
    saved = hub.host.instrument_model
    hub.host.instrument_model = make_model(tip_voltage_window_v=(-8.0, 30.0))
    try:
        # The chiral door has no window check, so this lands at 15..35 V.
        _chiral_on(client, v0=25.0, v_k=10.0)
        assert max(hub.pattern.objects[0].v_profile) == pytest.approx(35.0, abs=1e-3)

        r = _op(client, op="set_props", indices=[0], voltage=6)
        assert r.get("ok"), r
        o = hub.pattern.objects[0]
        assert o.voltage == 6.0 and o.profile["v0"] == 6.0
        assert min(o.v_profile) == pytest.approx(-4.0, abs=1e-3)
        assert max(o.v_profile) == pytest.approx(16.0, abs=1e-3)

        before = copy.deepcopy(hub.pattern.objects)
        r = _op(client, op="set_props", indices=[0], voltage=25)
        assert not r.get("ok"), r
        assert r["reason"] == "tip_voltage_out_of_window"
        assert "'w'" in r["error"] and "35" in r["error"], r
        assert hub.pattern.objects == before
        assert r["design"]["objects"][0]["voltage"] == 6.0
    finally:
        hub.host.instrument_model = saved


# ------------------------------------------------- bug 2: the transforms --

def test_a_fresh_bake_matches_its_settings_and_its_drawing():
    o = _wire()
    pitch, amp, n = _pitch_amp(o.points_um, *AXIS)
    assert pitch == pytest.approx(25.0, abs=1e-3)
    assert amp == pytest.approx(55.0, abs=1e-3)
    assert n == 39 and o.profile["periods"] == pytest.approx(40.0)
    assert _draw_error_nm(o) <= DRAW_TOL_NM


def test_a_uniform_scale_keeps_pitch_and_amplitude_and_adds_periods():
    o = _wire()
    x0, y0, x1, y1 = pathops.bbox_of([o])
    pathops.transform(o, sx=1.5, sy=1.5, about=((x0 + x1) / 2, (y0 + y1) / 2))
    a0, a1 = o.axis_um[0], o.axis_um[-1]
    assert math.hypot(a1[0] - a0[0], a1[1] - a0[1]) == pytest.approx(1.5)
    pitch, amp, n = _pitch_amp(o.points_um, a0, a1)
    assert pitch == pytest.approx(25.0, abs=1e-3)
    assert amp == pytest.approx(55.0, abs=1e-3)
    assert n == 59 and o.profile["periods"] == pytest.approx(60.0)
    assert o.profile["lambda_um"] == pytest.approx(LAM)
    assert o.profile["y_amp_um"] == pytest.approx(AMP)
    _consistent(o)


def test_a_non_uniform_scale_keeps_pitch_and_amplitude():
    o = _wire()
    pathops.transform(o, sx=1.5, sy=3.0, about=(5.5, 5.0))
    a0, a1 = o.axis_um[0], o.axis_um[-1]
    assert (a0, a1) == ((4.75, 5.0), (6.25, 5.0))
    pitch, amp, n = _pitch_amp(o.points_um, a0, a1)
    assert pitch == pytest.approx(25.0, abs=1e-3)
    assert amp == pytest.approx(55.0, abs=1e-3)
    assert n == 59
    _consistent(o)


def test_a_rotation_rotates_the_spine_the_points_and_the_normal():
    o = _wire()
    x0, y0, x1, y1 = pathops.bbox_of([o])
    pathops.transform(o, rot_deg=10.0, about=((x0 + x1) / 2, (y0 + y1) / 2))
    a0, a1 = np.asarray(o.axis_um[0]), np.asarray(o.axis_um[-1])
    t = (a1 - a0) / np.hypot(*(a1 - a0))
    assert math.degrees(math.atan2(t[1], t[0])) == pytest.approx(10.0)
    assert np.allclose(o.profile["normal"], [[-t[1], t[0]]] * len(o.profile["normal"]),
                       atol=2e-6)
    pitch, amp, n = _pitch_amp(o.points_um, a0, a1)
    assert pitch == pytest.approx(25.0, abs=1e-3)
    # 0.02 nm: the rotated spine's sample grid does not land on the peaks.
    assert amp == pytest.approx(55.0, abs=0.02)
    assert n == 39
    _consistent(o)


def test_a_move_is_the_same_wire_somewhere_else():
    o = _wire()
    before = copy.deepcopy(o)
    pathops.transform(o, dx_um=0.7, dy_um=-0.3)
    assert np.allclose(np.asarray(o.points_um) - np.asarray(before.points_um),
                       [0.7, -0.3], atol=1e-12)
    assert np.allclose(np.asarray(o.axis_um) - np.asarray(before.axis_um), [0.7, -0.3])
    assert o.v_profile == before.v_profile
    assert o.profile["normal"] == before.profile["normal"]
    fresh = _wire(axis=[(5.7, 4.7), (6.7, 4.7)])
    assert np.allclose(o.points_um, fresh.points_um, atol=1e-9)
    _consistent(o)


def test_a_closed_loop_spine_scales_with_whole_periods():
    ang = np.linspace(0.0, 2 * math.pi, 241)
    ring = [(5.0 + math.cos(a), 5.0 + math.sin(a)) for a in ang]
    ring[-1] = ring[0]
    o = _wire(axis=ring)
    assert o.profile["closed"]
    periods = o.profile["periods"]
    pathops.transform(o, sx=1.5, sy=1.5, about=(5.0, 5.0))
    prof = o.profile
    assert prof["closed"] and prof["periods"] == round(prof["periods"])
    assert prof["periods"] == pytest.approx(1.5 * periods, abs=1.0)
    assert prof["lambda_um"] == pytest.approx(LAM, rel=0.01)
    assert max(prof["amp_um"]) == pytest.approx(AMP, abs=1e-6)
    r = np.hypot(*(np.asarray(o.points_um) - [5.0, 5.0]).T)
    assert r.max() == pytest.approx(1.5 + AMP, abs=1e-3)
    assert r.min() == pytest.approx(1.5 - AMP, abs=1e-3)
    assert o.points_um[0] == o.points_um[-1]
    assert _draw_error_nm(o) <= 0.05


@pytest.mark.parametrize("flip", [{"sx": -1.0}, {"sy": -1.0}])
def test_a_flip_mirrors_the_spine_and_keeps_the_handedness_the_settings_state(flip):
    o = _wire()
    assert _measured_phase_deg(o) == pytest.approx(90.0, abs=0.5)
    pathops.transform(o, about=(5.5, 5.0), **flip)
    expected = [(6.0, 5.0), (5.0, 5.0)] if "sx" in flip else AXIS
    assert o.axis_um == expected
    assert _measured_phase_deg(o) == pytest.approx(90.0, abs=0.5)
    assert math.degrees(o.profile["phase_rad"]) == pytest.approx(90.0)
    _consistent(o)


def test_a_transform_the_builder_refuses_leaves_the_wire_alone():
    o = _wire()
    before = copy.deepcopy(o)
    with pytest.raises(pathops.PathOpError, match="sample points"):
        pathops.transform(o, sx=5000.0, sy=5000.0, about=(5.5, 5.0))
    assert o == before


def test_a_wire_drawn_with_the_chiral_pen_rebakes_from_its_profile_axis():
    from afm_litho.litho.pens import render_stroke
    stroke = [(5.0 + 0.01 * i, 5.0) for i in range(101)]
    o = render_stroke(stroke, "chiral", {"lambda_um": LAM, "y_amp_um": AMP, "v0": 8.0,
                                         "v_k": V_K, "phase_deg": PHASE}, name="pen")[0]
    assert o.axis_um is None and o.profile["kind"] == "chiral"
    o.voltage = 6.0
    assert rebake_chiral(o) is not None
    assert o.axis_um is not None
    pitch, amp, _n = _pitch_amp(o.points_um, o.axis_um[0], o.axis_um[-1])
    assert pitch == pytest.approx(25.0, abs=0.01)
    assert amp == pytest.approx(55.0, abs=0.01)
    assert o.profile["v0"] == 6.0


def test_the_rest_transform_rebakes_and_reports_a_consistent_design(client):
    hub = _seed(_plain())
    _chiral_on(client, v0=8.0)
    r = _op(client, op="transform", indices=[0], sx=1.5, sy=1.5)
    assert r.get("ok"), r
    o = hub.pattern.objects[0]
    pitch, amp, n = _pitch_amp(o.points_um, o.axis_um[0], o.axis_um[-1])
    assert (pitch, amp, n) == (pytest.approx(25.0, abs=1e-3),
                               pytest.approx(55.0, abs=1e-3), 59)
    assert r["design"]["objects"][0]["profile"]["periods"] == pytest.approx(60.0)
    _consistent(o)


# ---------------------------------------------------------------- duplicate --

def test_a_duplicate_after_an_edit_is_consistent(client):
    hub = _seed(_plain())
    _chiral_on(client, v0=8.0)
    assert _op(client, op="set_props", indices=[0], voltage=6).get("ok")
    assert _op(client, op="transform", indices=[0], rot_deg=10.0).get("ok")
    r = _op(client, op="duplicate", indices=[0], dx_um=1.0, dy_um=-1.0)
    assert r.get("ok"), r
    src, dup = hub.pattern.objects
    assert dup.name == "w-copy" and dup.voltage == 6.0
    _consistent(src)
    _consistent(dup)
    assert dup.v_profile == src.v_profile
    assert np.allclose(np.asarray(dup.points_um) - np.asarray(src.points_um),
                       [1.0, -1.0], atol=1e-12)
    assert np.allclose(np.asarray(dup.axis_um) - np.asarray(src.axis_um), [1.0, -1.0])

    assert _op(client, op="set_props", indices=[1], voltage=4).get("ok")
    src, dup = hub.pattern.objects
    assert (src.voltage, src.profile["v0"]) == (6.0, 6.0)
    assert (dup.voltage, dup.profile["v0"]) == (4.0, 4.0)
    _consistent(dup)


# ------------------------------------------------ the chiral route is atomic --

def _history() -> list:
    from afm_litho.service.routers import design
    return design._design_history


@pytest.mark.parametrize("patch,field", [
    ({"lambda_um": None, "y_amp_um": None, "v_k": None, "phase_deg": None}, "lambda_um"),
    ({"v_k": None}, "v_k"),
    ({"phase_deg": "ninety"}, "phase_deg"),
    ({"y_amp_um": True}, "y_amp_um"),
    ({"v0": "six"}, "v0"),
])
def test_a_bad_chiral_setting_is_refused_and_nothing_changes(client, patch, field):
    hub = _seed(_plain())
    _chiral_on(client, v0=8.0)
    before = copy.deepcopy(hub.pattern.objects)
    steps = len(_history())
    body = {"indices": [0], "enabled": True, "lambda_um": LAM, "y_amp_um": AMP,
            "v_k": V_K, "phase_deg": PHASE, "v0": 6, **patch}
    body = {k: v for k, v in body.items() if v is not None}
    resp = client.post("/api/design/chiral", json=body)
    assert resp.status_code == 200, resp.text
    r = resp.json()
    assert not r.get("ok") and r["reason"] == "chiral_bad_setting", r
    assert r["field"] == field and field in r["error"], r
    assert hub.pattern.objects == before
    assert len(_history()) == steps
    o = hub.pattern.objects[0]
    assert (o.voltage, o.profile["v0"], max(o.v_profile)) == (8.0, 8.0, pytest.approx(13.0))


def test_a_builder_refusal_leaves_the_whole_selection_alone(client):
    hub = _seed(_plain(), _plain(axis=[(5.0, 7.0), (5.0, 7.0)], name="dot"))
    _chiral_on(client, v0=8.0)
    before = copy.deepcopy(hub.pattern.objects)
    steps = len(_history())
    r = client.post("/api/design/chiral", json={
        "indices": [0, 1], "enabled": True, "lambda_um": LAM, "y_amp_um": AMP,
        "v_k": V_K, "phase_deg": PHASE, "v0": 6}).json()
    assert not r.get("ok") and r["reason"] == "chiral_build_refused", r
    assert "'dot'" in r["error"] and "no length" in r["error"], r
    assert "at least 2 elements" not in r["error"]
    assert hub.pattern.objects == before
    assert len(_history()) == steps


# ------------------------------------------- the pitch asked for is kept --

def _ring(radius=1.0, n=241):
    ang = np.linspace(0.0, 2 * math.pi, n)
    ring = [(5.0 + radius * math.cos(a), 5.0 + radius * math.sin(a)) for a in ang]
    ring[-1] = ring[0]
    return ring


def test_repeated_scaling_of_a_closed_loop_does_not_drift_the_pitch():
    from afm_litho.service.host.litho_cmds import _chiral_view
    o = _wire(axis=_ring())
    assert o.profile["lambda_set_um"] == LAM
    assert o.profile["lambda_um"] != LAM          # quantised to whole periods
    for _ in range(50):
        pathops.transform(o, sx=1.01, sy=1.01, about=(5.0, 5.0))
    fresh = _wire(axis=list(o.axis_um))
    assert o.profile["periods"] == fresh.profile["periods"]
    assert o.profile["lambda_um"] == fresh.profile["lambda_um"]
    assert o.profile["lambda_set_um"] == LAM
    assert o.points_um == fresh.points_um
    assert o.v_profile == fresh.v_profile
    assert _chiral_view(o)["lambda_um"] == LAM


def test_a_profile_without_the_asked_for_pitch_falls_back_to_the_built_one():
    from afm_litho.litho.pens import chiral_settings
    from afm_litho.service.host.litho_cmds import _chiral_view
    o = _wire(axis=_ring())
    built = o.profile["lambda_um"]
    del o.profile["lambda_set_um"]
    assert chiral_settings(o)["lambda_um"] == built
    assert _chiral_view(o)["lambda_um"] == built
    assert rebake_chiral(o) is not None


# ------------------------------------------------ v0 is the object's voltage --

def test_a_node_edit_takes_v0_from_the_voltage_not_the_stale_profile(client):
    hub = _seed(_wire())
    o = hub.pattern.objects[0]
    o.voltage = 6.0                                # profile.v0 is still 8
    assert o.profile["v0"] == 8.0
    bez = client.get("/api/design/bezier/0").json()["bezier"]
    r = client.post("/api/design/bezier", json={"index": 0, "bezier": bez}).json()
    assert r.get("ok") and r["reapplied"], r
    o = hub.pattern.objects[0]
    assert o.voltage == 6.0 and o.profile["v0"] == 6.0
    assert max(o.v_profile) == pytest.approx(11.0, abs=5e-3)


def _unrebakeable(no_spine: bool) -> LithoObject:
    o = _wire()
    if no_spine:
        o.axis_um = None
        del o.profile["axis_um"]
    else:
        del o.profile["lambda_um"]
        del o.profile["lambda_set_um"]
    return o


@pytest.mark.parametrize("no_spine", [True, False])
def test_a_voltage_edit_on_an_unrebakeable_chiral_is_refused(client, no_spine):
    hub = _seed(_unrebakeable(no_spine))
    before = copy.deepcopy(hub.pattern.objects)
    r = _op(client, op="set_props", indices=[0], voltage=6)
    assert not r.get("ok") and r["reason"] == "chiral_unrebakeable", r
    assert "'w'" in r["error"]
    assert hub.pattern.objects == before


@pytest.mark.parametrize("no_spine", [True, False])
def test_flex_voltage_on_an_unrebakeable_chiral_is_refused_with_a_reason(no_spine):
    from afm_litho.service.host import litho_cmds
    from afm_litho.service.commands.errors import CommandError
    o = _unrebakeable(no_spine)
    before = copy.deepcopy(o)
    with pytest.raises(CommandError) as ei:
        litho_cmds._apply_object_update(copy.deepcopy(o), "w", {"voltage": 6.0})
    assert ei.value.data["reason"] == "chiral_unrebakeable"
    assert ei.value.data["object_id"] == "w" and ei.value.data["field"] == "voltage"
    assert o == before


@pytest.mark.parametrize("no_spine", [True, False])
def test_a_transform_of_an_unrebakeable_chiral_is_refused_but_a_move_is_not(no_spine):
    o = _unrebakeable(no_spine)
    before = copy.deepcopy(o)
    with pytest.raises(pathops.ChiralRefused) as ei:
        pathops.transform(o, sx=1.5, sy=1.5, about=(5.5, 5.0))
    assert ei.value.reason == "chiral_unrebakeable" and "'w'" in str(ei.value)
    assert o == before
    pathops.transform(o, dx_um=1.0)
    assert o.points_um[0][0] == pytest.approx(before.points_um[0][0] + 1.0)


# ------------------------------------------------------- plain-words refusals --

def test_a_scale_by_zero_is_refused_in_plain_words(client):
    hub = _seed(_plain())
    _chiral_on(client, v0=8.0)
    before = copy.deepcopy(hub.pattern.objects)
    r = _op(client, op="transform", indices=[0], sx=0.0, sy=0.0)
    assert not r.get("ok") and r["reason"] == "chiral_build_refused", r
    assert "'w'" in r["error"] and "no length" in r["error"], r
    assert "at least 2 elements" not in r["error"]
    assert hub.pattern.objects == before


def test_a_voltage_edit_on_a_degenerate_spine_is_refused_in_plain_words(client):
    o = _wire()
    o.axis_um = [(5.0, 5.0), (5.0, 5.0)]
    hub = _seed(o)
    before = copy.deepcopy(hub.pattern.objects)
    r = _op(client, op="set_props", indices=[0], voltage=6)
    assert not r.get("ok") and r["reason"] == "chiral_build_refused", r
    assert "'w'" in r["error"] and "no length" in r["error"], r
    assert hub.pattern.objects == before


# ------------------------------------------------- a transform that does nothing --

def test_a_flip_across_the_wires_own_axis_records_no_undo_step_and_no_log(
        client, monkeypatch):
    hub = _seed(_plain())
    _chiral_on(client, v0=8.0)
    before = copy.deepcopy(hub.pattern.objects)
    steps = len(_history())
    lines: list[str] = []
    original = hub.log

    async def spy(text):
        lines.append(text)
        await original(text)

    monkeypatch.setattr(hub, "log", spy)
    r = _op(client, op="transform", indices=[0], sy=-1.0, say="Flipped 1 object(s)")
    assert r.get("ok") and r["message"] == "No change", r
    assert hub.pattern.objects == before
    assert len(_history()) == steps and r["undo_steps"] == steps
    assert lines == []

    r = _op(client, op="transform", indices=[0], sx=-1.0, say="Flipped 1 object(s)")
    assert r.get("ok") and r["message"] == "Flipped 1 object(s)", r
    assert len(_history()) == steps + 1 and lines == ["Flipped 1 object(s)"]


# --------------------------------------- the chiral route never answers 500 --

_CHIRAL_BODY = ('{"indices": [0], "enabled": true, "lambda_um": 0.025, '
                '"y_amp_um": 0.055, "v_k": 5, "phase_deg": 90, "v0": 6}')


def _post_raw_chiral(client, old: str, new: str):
    assert old in _CHIRAL_BODY
    return client.post("/api/design/chiral", content=_CHIRAL_BODY.replace(old, new),
                       headers={"content-type": "application/json"})


@pytest.mark.parametrize("old,new,reason,field", [
    ('"v_k": 5', '"v_k": NaN', "chiral_bad_setting", "v_k"),
    ('"lambda_um": 0.025', '"lambda_um": Infinity', "chiral_bad_setting", "lambda_um"),
    ('"v0": 6', '"v0": -Infinity', "chiral_bad_setting", "v0"),
    ('"lambda_um": 0.025', '"lambda_um": 1' + "0" * 400, "chiral_bad_setting",
     "lambda_um"),
    ('"lambda_um": 0.025', '"lambda_um": 0', "chiral_bad_setting", "lambda_um"),
    ('"lambda_um": 0.025', '"lambda_um": -0.025', "chiral_bad_setting", "lambda_um"),
    ('"indices": [0]', '"indices": ["x"]', "chiral_bad_setting", "indices"),
    ('"indices": [0]', '"indices": [[0]]', "chiral_bad_setting", "indices"),
    ('"v_k": 5', '"v_k": 1e308', "chiral_build_refused", "settings"),
    ('"y_amp_um": 0.055', '"y_amp_um": 1e308', "chiral_build_refused", "settings"),
])
def test_an_unusable_chiral_request_is_a_refusal_not_a_500(client, old, new, reason,
                                                         field):
    hub = _seed(_plain())
    _chiral_on(client, v0=8.0)
    before = copy.deepcopy(hub.pattern.objects)
    steps = len(_history())
    resp = _post_raw_chiral(client, old, new)
    assert resp.status_code == 200, resp.text
    r = resp.json()
    assert not r.get("ok") and (r["reason"], r["field"]) == (reason, field), r
    assert hub.pattern.objects == before
    assert len(_history()) == steps
    served = client.get("/api/design")
    assert served.status_code == 200
    assert served.json()["objects"][0]["voltage"] == 8.0


def test_a_bake_that_is_not_finite_is_refused_with_the_object_untouched():
    o = _wire()
    before = copy.deepcopy(o)
    with pytest.raises(ValueError, match="not finite"):
        bake_chiral(o, LAM, AMP, 1e308, PHASE)
    assert o == before


# -------------------------------------------------------- the FLEX chiral view --

def test_the_chiral_view_reports_the_asked_for_and_the_built_pitch():
    from afm_litho.service.host import litho_cmds
    o = _wire(axis=_ring())
    view = litho_cmds._chiral_view(o)
    assert view["lambda_um"] == LAM
    assert view["lambda_built_um"] == o.profile["lambda_um"] != LAM
    assert set(view) == {"lambda_um", "lambda_built_um", "y_amp_um", "v_k",
                         "phase_deg", "hand"}
    # A script may send the view straight back; the built pitch is ignored.
    litho_cmds._apply_object_update(o, "w", {"chiral": {**view, "phase_deg": 0.0}})
    assert o.profile["lambda_set_um"] == LAM
    assert o.profile["phase_rad"] == 0.0


def test_a_flex_chiral_block_the_builder_refuses_carries_a_reason_and_the_field():
    from afm_litho.service.host import litho_cmds
    from afm_litho.service.commands.errors import CommandError
    block = {"lambda_um": LAM, "y_amp_um": AMP, "v_k": V_K, "phase_deg": 0.0,
             "hand": 1.0}
    dot = _plain(axis=[(5.0, 5.0), (5.0, 5.0)], name="dot")
    with pytest.raises(CommandError) as ei:
        litho_cmds._apply_object_update(dot, "dot", {"chiral": block})
    assert ei.value.data["reason"] == "chiral_build_refused"
    assert ei.value.data["field"] == "chiral" and "no length" in str(ei.value)

    wire = _plain()
    with pytest.raises(CommandError) as ei:
        litho_cmds._apply_object_update(wire, "w", {"chiral": {**block,
                                                               "y_amp_um": 100.0}})
    assert ei.value.data["reason"] == "chiral_build_refused"
    assert ei.value.data["field"] == "y_amp_um"


# ------------------------------------------------ a full undo history is kept --

def _fill_history() -> list:
    from afm_litho.service.routers import design
    history = _history()
    history[:] = [[_plain(name=f"step{i}")] for i in range(design._DESIGN_HISTORY_MAX)]
    return list(history)


def _same_steps(before: list) -> bool:
    now = _history()
    return len(now) == len(before) and all(a is b for a, b in zip(now, before))


def test_a_no_change_op_keeps_a_full_undo_history(client):
    _seed(_plain())
    _chiral_on(client, v0=8.0)
    before = _fill_history()
    r = _op(client, op="transform", indices=[0], sy=-1.0)
    assert r.get("ok") and r["message"] == "No change", r
    assert r["undo_steps"] == len(before)
    assert _same_steps(before)


@pytest.mark.parametrize("body,reason", [
    ({"op": "set_props", "voltage_end": 2.0}, "chiral_bias_fixed"),
    ({"op": "set_props", "voltage": 300}, "tip_voltage_out_of_window"),
    ({"op": "transform", "sx": 0.0, "sy": 0.0}, "chiral_build_refused"),
])
def test_a_refused_op_keeps_a_full_undo_history(client, body, reason):
    hub = _seed(_plain())
    _chiral_on(client, v0=8.0)
    objects = copy.deepcopy(hub.pattern.objects)
    before = _fill_history()
    r = _op(client, indices=[0], **body)
    assert not r.get("ok") and r["reason"] == reason, r
    assert _same_steps(before)
    assert hub.pattern.objects == objects


def test_a_refused_chiral_request_keeps_a_full_undo_history(client):
    _seed(_plain())
    _chiral_on(client, v0=8.0)
    before = _fill_history()
    r = client.post("/api/design/chiral", json={
        "indices": [0], "enabled": True, "lambda_um": 0, "y_amp_um": AMP,
        "v_k": V_K, "phase_deg": PHASE}).json()
    assert r["reason"] == "chiral_bad_setting", r
    assert _same_steps(before)


def test_a_real_edit_on_a_full_history_still_drops_the_oldest_step(client):
    from afm_litho.service.routers import design
    _seed(_plain())
    _chiral_on(client, v0=8.0)
    before = _fill_history()
    assert _op(client, op="set_props", indices=[0], voltage=6).get("ok")
    now = _history()
    assert len(now) == design._DESIGN_HISTORY_MAX
    assert all(a is b for a, b in zip(now[:-1], before[1:]))
