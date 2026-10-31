"""Stroke end, core (docs/STROKE_END_CORE_SPEC.md).

Igor reports a stroke done when its capture fills, and the capture is a multiple of 32
points. Every stroke is therefore fired as a multiple of 32 points whose last sample is
0 V at the last position, and the run waits for DONE from that fired length. On the
Cypher (2026-10-04) a 5-point cut at 4.32 s per point filled its 32-point capture in
138 s; the run waited 32 s.
"""

from __future__ import annotations

import queue
from itertools import pairwise

import numpy as np
import pytest

from afm_litho.hardware import lithography_run
from afm_litho.hardware.igor_stub import FakeIgorBridge, values_from_note
from afm_litho.hardware.lithography_run import LithographyRun
from afm_litho.litho.chunker import CHAIN_CHUNK_POINTS, _penup_mask, chunk_boundaries
from afm_litho.litho.compiler import (
    LithoObject,
    Pattern,
    _geom_points,
    _rect_raster,
    _sample_polyline,
    _taper_raster,
    compile_object,
    compile_pattern,
    expand,
    fired_points,
    trace_indices,
)
from afm_litho.litho.pens import render_stroke

SENS = 7.4643e-6
REFERENCE_CUT = [(1.608341056658206, -1.0106456506259907),
            (1.6057224996574544, -1.0970081152745188)]


def _cut() -> LithoObject:
    return LithoObject("path-2", "path", REFERENCE_CUT, voltage=-8.0, speed_um_s=0.005)


def _geometry(obj: LithoObject) -> np.ndarray:
    """The polyline the compiler samples for a single-stroke object."""
    pts = np.asarray(obj.points_um, dtype=float)
    if obj.kind == "rect":
        return _rect_raster(pts[0], pts[1], obj.fill_spacing_um)
    if obj.kind == "taper":
        return _taper_raster(pts[0], pts[1], obj.width0_um, obj.width1_um,
                             obj.fill_spacing_um)
    return pts


def _ring(r, n=64):
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return [(r * np.cos(t), r * np.sin(t)) for t in th]


def _strokes() -> list[LithoObject]:
    """One of every kind the compiler turns into a single tip path."""
    chiral = render_stroke([(-3.0, 0.5), (3.0, 0.5)], "chiral", {"phase_deg": 90})[0]
    out = [
        LithoObject("line", "path", [(0, 0), (4.12, 0)], voltage=8.0, speed_um_s=0.1),
        LithoObject("corner", "path", [(0, 0), (10, 0), (10, 5), (3, 7)],
                    interpolation_um=0.1),
        _cut(),
        LithoObject("ramp", "path", [(0, 0), (10, 0)], voltage=2.0, voltage_end=8.0),
        LithoObject("profile", "path", [(0, 0), (3, 0), (3, 2)],
                    v_profile=[0.0, 4.0, -4.0, 8.0, 1.0]),
        LithoObject("dot", "path", [(1.0, 2.0), (1.0, 2.0)], voltage=5.0),
        LithoObject("rect", "rect", [(0, 0), (2, 1)], fill_spacing_um=0.25),
        LithoObject("taper", "taper", [(-5.0, -4.0), (2.0, -4.0)],
                    width0_um=2.0, width1_um=0.3),
        chiral,
    ]
    for shape in (LithoObject("disc", "shape", _ring(1.5), voltage=6.0),
                  LithoObject("annulus", "shape", _ring(3.0), holes_um=[_ring(1.5)])):
        out.extend(expand(shape))
    return out


# ------------------------------------------------------------------ compiler ----

@pytest.mark.parametrize("obj", _strokes(), ids=lambda o: o.name)
def test_every_stroke_is_a_multiple_of_32_ending_in_one_zero_volt_sample(obj):
    c = compile_object(obj, SENS, SENS)
    n = len(c["v"])
    geom = _geometry(obj)
    n_geom = _geom_points(geom, obj.interpolation_um)
    assert n == fired_points(n_geom) == 32 * -(-(n_geom + 1) // 32)
    assert n % 32 == 0 and len(c["x_v"]) == len(c["y_v"]) == len(c["path_um"]) == n
    assert c["capture_points"] == n                 # capture length == out-wave length
    assert c["path_points"] == n - 1
    # the last sample: 0 V, at the last position
    assert c["v"][-1] == 0.0
    assert np.array_equal(c["path_um"][-1], c["path_um"][-2])
    assert c["x_v"][-1] == c["x_v"][-2] and c["y_v"][-1] == c["y_v"][-2]
    # the path points: one uniform arc-length pass over the object's own vertices,
    # first and last exact
    path = c["path_um"][:n - 1]
    assert np.allclose(path, _sample_polyline(geom, n - 1), rtol=0, atol=1e-12)
    assert np.array_equal(path[0], geom[0]) and np.allclose(path[-1], geom[-1], atol=1e-12)
    # their bias: the object's own function of arc fraction, as before
    frac = np.linspace(0.0, 1.0, n - 1)
    if obj.v_profile:
        prof = np.asarray(obj.v_profile, dtype=float)
        want = np.interp(frac, np.linspace(0.0, 1.0, len(prof)), prof)
    else:
        v1 = obj.voltage if obj.voltage_end is None else obj.voltage_end
        want = obj.voltage + frac * (v1 - obj.voltage)
    assert np.allclose(c["v"][:n - 1], want, rtol=0, atol=1e-12)
    # timing: main's rule on the fired points (first sample to last)
    assert c["duration_s"] == pytest.approx((n - 1) * c["ticks_per_point"] / 50_000)


def test_the_point_count_grows_only_to_the_next_multiple_of_32():
    for n_geom in (1, 2, 5, 30, 31, 32, 63, 64, 207, 86_975):
        n = fired_points(n_geom)
        assert n % 32 == 0 and 1 <= n - n_geom <= 32


def test_the_reference_cut_is_32_points():
    c = compile_object(_cut(), SENS, SENS)
    assert _geom_points(np.asarray(REFERENCE_CUT), 0.02) == 5
    assert len(c["v"]) == 32
    assert c["v"][:31].tolist() == [-8.0] * 31 and c["v"][31] == 0.0
    tick_s = c["ticks_per_point"] / 50_000
    assert tick_s == pytest.approx(0.0864 / 30 / 0.005, rel=1e-3)      # 0.576 s


def test_a_stroke_over_the_chunk_size_chains_in_whole_multiples_of_32():
    """Geometry of 86,976 points -> 87,008 fired: chained, the bias on across the
    seam, 0 V only on the last sample."""
    obj = LithoObject("long", "path", [(0.0, 0.0), ((CHAIN_CHUNK_POINTS - 1) * 0.02, 0.0)],
                      interpolation_um=0.02, voltage=7.0, speed_um_s=50.0)
    c = compile_object(obj, SENS, SENS)
    n = len(c["v"])
    assert n == 87_008 > CHAIN_CHUNK_POINTS
    bounds = chunk_boundaries(n, _penup_mask(c["v"], c["x_v"], c["y_v"]))
    assert bounds == [0, CHAIN_CHUNK_POINTS, n]
    assert all((b - a) % 32 == 0 for a, b in pairwise(bounds))
    assert np.count_nonzero(c["v"] == 0.0) == 1 and c["v"][-1] == 0.0
    assert c["v"][CHAIN_CHUNK_POINTS - 1] == c["v"][CHAIN_CHUNK_POINTS] == 7.0


def test_the_pattern_reports_the_fired_counts():
    waves = compile_pattern(Pattern("p", [_cut()]), SENS, SENS)
    seg = waves["segments"][0]
    assert len(seg["v"]) == seg["capture_points"] == 32 and seg["path_points"] == 31
    assert seg["v"][-1] == 0.0 and seg["v"][-2] == -8.0


def test_trace_indices_end_at_the_last_path_point_never_the_zero_volt_sample():
    for n_fired in (32, 64, 224, 8192):
        idx = trace_indices(n_fired - 1)
        assert idx[0] == 0 and idx[-1] == n_fired - 2
        assert np.all(np.diff(idx) > 0) and np.all(np.diff(idx)[:-1] == 4)
    assert trace_indices(0).tolist() == []
    assert trace_indices(1).tolist() == [0]
    assert trace_indices(6).tolist() == [0, 4, 5]


# ---------------------------------------------------------------------- twin ----

def test_the_twin_fires_the_compiled_arrays_and_its_trace_keeps_the_written_end():
    from afm_litho.service.passes import LithoRun, TwinInstrument
    from afm_litho.twin.arc import SoftwareARC
    from afm_litho.twin.physics import Sample, World

    arc = SoftwareARC(World(sample=Sample(height_map=np.zeros((64, 64)), extent=30e-6)))
    pat = Pattern(name="t", objects=[
        LithoObject("wire", "path", [(-3, 0), (3, 0)], voltage=8.0, speed_um_s=6.0,
                    interpolation_um=0.05),
        LithoObject("ramp", "path", [(0, 2), (0, 1)], voltage=2.0, voltage_end=7.0,
                    speed_um_s=6.0, interpolation_um=0.05)])
    w = arc.world
    segs = compile_pattern(pat, w.x_lvdt_sens, w.y_lvdt_sens,
                           w.x_lvdt_offset, w.y_lvdt_offset)["segments"]
    armed = []
    out_wave = arc.td_xSetOutWave

    def rec_wave(bank, event, channel, wave, interp):
        if channel == "Output.A" and np.any(np.asarray(wave) != 0.0):
            armed.append(np.array(wave, dtype=float))
        return out_wave(bank, event, channel, wave, interp)

    arc.td_xSetOutWave = rec_wave
    q = queue.Queue()
    record: dict = {}
    LithoRun(TwinInstrument(arc), pat, q, cmd_capture=record).run()
    done = []
    while not q.empty():
        m = q.get()
        if m["type"] == "litho_object_done":
            done.append(m)

    assert len(armed) == len(segs) == len(done) == 2
    for seg, v, m in zip(segs, armed, done):
        n = len(seg["v"])
        assert n % 32 == 0 and np.array_equal(v, seg["v"]) and v[-1] == 0.0 != v[-2]
        idx = trace_indices(seg["path_points"])
        assert m["path_um"] == [[round(float(x), 3), round(float(y), 3)]
                                for x, y in seg["path_um"][idx]]
        assert m["v_written"] == [round(float(x), 3) for x in seg["v"][idx]]
    assert done[0]["v_written"][-1] == 8.0 and done[1]["v_written"][-1] == 7.0
    # the commanded-bias record has the fired length of every stroke
    assert len(record["output_a_cmd_v"]) == sum(len(s["v"]) for s in segs)


# ---------------------------------------------------------------- the run -------

def _bridge(**kw):
    note = {"XLVDTSens": 7.0e-6, "XLVDTOffset": -0.6, "YLVDTSens": 7.0e-6,
            "YLVDTOffset": 0.07, "ScanRate": 1.0}
    return FakeIgorBridge(values_from_note(note), **kw)


def _run(port, **cfg):
    return LithographyRun.connect(
        "127.0.0.1", port, settle_move_s=0.0, cmd_gap_s=0.0, xy_withdraw_max_s=0.0,
        xy_withdraw_settle_s=0.0, first_object_settle_s=0.0, **cfg)


def test_the_cut_as_fired_on_2026_10_04_could_not_complete_within_its_wait():
    """Arithmetic of the failure: 5 points at 4.32 s, a 32-sample capture, and the wait
    main took from the unpadded stroke."""
    tick_s = 0.0864 / 4 / 0.005                       # 4.32 s per point
    fill_s = 32 * tick_s                              # 138 s to DONE
    old_wait_s = max(30.0, (5 - 1) * tick_s + 15.0)   # 32.3 s
    assert old_wait_s < fill_s
    # the same stroke as it is fired now
    c = compile_object(_cut(), SENS, SENS)
    new_fill_s = len(c["v"]) * c["ticks_per_point"] / 50_000
    new_wait_s = lithography_run._done_wait_s(30.0, new_fill_s)
    assert new_fill_s == pytest.approx(18.43, abs=0.01)
    assert new_wait_s == pytest.approx(new_fill_s + 15.0)
    # and the old stroke, were it fired today, would be waited for as well
    assert lithography_run._done_wait_s(30.0, fill_s) > fill_s


def test_the_cut_completes_against_a_bridge_that_fills_the_capture():
    """The stub completes the stroke when its capture fills (points x tick, scaled to
    milliseconds); the run waits the unscaled fill plus the margin."""
    c = compile_object(_cut(), 7.0e-6, 7.0e-6, -0.6, 0.07)
    p = c["path_um"]
    stub = _bridge(fill_scale=0.002)
    port = stub.serve_in_thread()
    try:
        run = _run(port)
        run.start()
        res = run.write_object("path-2", p[:, 0], p[:, 1], c["v"], setpoint=0.4,
                               speed=0.005)
        run.get_value("XSensor")                      # barrier
    finally:
        stub.stop()
    assert res.points == res.cap_len == 32 and not res.aborted
    # capture length == out-wave length, and the bias wave ends at 0 V: nothing held
    assert stub.arms[0][2] == len(stub.waves["VLitho"]) == 32
    assert stub.waves["VLitho"][-1] == 0.0 and stub.waves["VLitho"][-2] == -8.0
    assert stub.waves["XLitho"][-1] == stub.waves["XLitho"][-2]
    assert stub.arms[0][3] == c["ticks_per_point"]
    run.close()


def test_a_stroke_that_never_completes_raises_a_timeout_that_says_so(monkeypatch):
    """The bridge fills the capture 100x slower than the stroke's own time, far beyond
    the wait (patched short): the error names the stroke, the wait and the fill."""
    monkeypatch.setattr(lithography_run, "_done_wait_s", lambda timeout_s, fill_s: 0.3)
    c = compile_object(_cut(), 7.0e-6, 7.0e-6, -0.6, 0.07)
    p = c["path_um"]
    stub = _bridge(fill_scale=100.0)
    port = stub.serve_in_thread()
    try:
        run = _run(port)
        run.start()
        with pytest.raises(TimeoutError) as caught:
            run.write_object("path-2", p[:, 0], p[:, 1], c["v"], setpoint=0.4,
                             speed=0.005)
    finally:
        stub.stop()
    text = str(caught.value)
    assert "path-2" in text and "no DONE within 0.3 s" in text
    assert "capture fill 18.4 s" in text and "32 points" in text
    run.close()


def test_arrays_not_from_the_compiler_are_fired_ending_at_zero_volts():
    logs = []
    stub = _bridge()
    port = stub.serve_in_thread()
    try:
        run = _run(port, log=logs.append)
        run.start()
        res = run.write_object("raw", [0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [5.0, 5.0, 5.0],
                               setpoint=0.4, speed=2.0)
        run.get_value("XSensor")
    finally:
        stub.stop()
    assert res.points == res.cap_len == 32
    assert stub.waves["VLitho"] == [5.0] * 3 + [0.0] * 29
    assert any("29 points at 0 V appended" in m for m in logs)
    run.close()


# ------------------------------------------------------ the hardware worker ----

HW_CAL = {"x_lvdt_sens": 7.46e-6, "x_lvdt_offset": -0.63,
          "y_lvdt_sens": 7.04e-6, "y_lvdt_offset": 0.07,
          "z_lvdt_sens": 5e-6, "z_lvdt_offset": 0.0}
CHANNELS = ("UserIn2", "Lateral", "ZSensor", "Deflection", "XSensor", "YSensor")


def _wait(cond, timeout=20.0, interval=0.02):
    import time
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return False


def test_the_hardware_worker_reports_the_written_end_and_records_the_capture_length(
        client, tmp_path):
    """One wire through the hardware worker over the fake bridge: the wave armed has
    the capture's length and ends at 0 V; the trace frame ends at the last path point
    with its bias; the record's commanded bias is as long as the capture."""
    import contextlib

    from _im_fixtures_guard import GUARD_WRITE_KW, seed_free_air

    from afm_litho.hardware.hardware_arc import HardwareARC
    from afm_litho.service.app import hub
    from afm_litho.service.routers.litho import _litho_frame
    from afm_litho.service.sessions import SessionStore

    pat = Pattern(name="stroke-end", objects=[
        LithoObject("wire", "path", [(-1.0, 0.0), (1.0, 0.0)], voltage=6.0,
                    speed_um_s=8.0, interpolation_um=0.1)])
    w = hub.host.world
    seg = compile_pattern(pat, w.x_lvdt_sens, w.y_lvdt_sens, w.x_lvdt_offset,
                          w.y_lvdt_offset, frame=_litho_frame())["segments"][0]
    n = len(seg["v"])
    assert n == 32 and seg["path_points"] == 31

    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    stub = FakeIgorBridge({"DeflectionValue": -0.35},
                          capture=lambda name, cap_len: {ch: [0.1] * cap_len
                                                         for ch in CHANNELS})
    port = stub.serve_in_thread()
    hub.host.hardware_arc = HardwareARC.connect("127.0.0.1", port)
    hub.host.hw_cal = dict(HW_CAL)
    hub.pattern = pat
    hub.host.written_objects = set()
    hub.host.z_loop = {"mode": "contact"}
    hub.host.imaging_mode = "contact"
    done = None
    try:
        seed_free_air(hub, value_v=-0.2)
        with client.websocket_connect("/ws") as ws:
            ws.send_text("ping")
            r = client.post("/api/litho/write", json={
                "objects": ["wire"], "setpoint_relative": 0.2, **GUARD_WRITE_KW})
            assert r.json().get("backend") == "hardware", r.text
            for _ in range(2000):
                m = ws.receive_json()
                if m.get("type") == "litho_done":
                    done = m
                    break
        assert _wait(lambda: not (hub.host.litho_thread and hub.host.litho_thread.is_alive()))
        accum = list(getattr(hub.host, "written_accum", None) or [])
    finally:
        hub.host.engaged = False
        with contextlib.suppress(Exception):
            hub.host.hardware_arc.close()
        hub.host.hardware_arc = None
        hub.host.hw_cal = {}
        hub.host.hw_litho_capture = {}
        hub.host.z_loop = {}
        hub.host.imaging_mode = None
        hub.fault = None
        stub.stop()
        _wait(lambda: not (hub.host.litho_thread and hub.host.litho_thread.is_alive()))

    assert done is not None and done["aborted"] is False
    gain = GUARD_WRITE_KW["amp_gain"]
    assert stub.arms[0][2] == n == len(stub.waves["VLitho"])
    assert stub.waves["VLitho"][-1] == 0.0
    assert stub.waves["VLitho"][-2] == pytest.approx(6.0 / gain)
    # the trace: every 4th PATH point and the last one, never the 0 V sample
    idx = trace_indices(seg["path_points"])
    assert idx.tolist() == [0, 4, 8, 12, 16, 20, 24, 28, 30]
    assert accum[0]["path_um"] == [[round(float(x), 3), round(float(y), 3)]
                                   for x, y in seg["path_um"][idx]]
    assert accum[0]["v_written"] == [6.0] * len(idx)
    # the record: the commanded bias has the capture's length and ends at 0 V
    assert _wait(lambda: list((tmp_path / "writes" / "npz").glob("*.npz")))
    with np.load(next((tmp_path / "writes" / "npz").glob("*.npz"))) as z:
        cmd, cap = z["output_a_cmd_v"], z["ZSensor"]
    assert len(cmd) == len(cap) == n
    assert cmd[-1] == 0.0 and cmd[-2] == pytest.approx(6.0 / gain)


# ------------------------------------------------- the stroke's tick rounds up ----

def test_a_strokes_tick_is_the_ceiling_and_a_transits_is_the_vendors():
    """A stroke never runs faster than requested: its ticks per point are rounded UP
    at every tick size. Transits keep ARGetDeci's round-to-nearest."""
    import math

    from afm_litho.hardware.xy_transit import _build_wavepair_v
    from afm_litho.litho.compiler import time_segment

    # 0.02 um at 10 um/s, 31 path points: 3.33 ticks -> 4 (nearest: 3, 11 % fast)
    tiny = compile_object(LithoObject("tiny", "path", [(0, 0), (0.02, 0)],
                                      speed_um_s=10.0), SENS, SENS)
    assert len(tiny["v"]) == 32 and tiny["ticks_per_point"] == 4
    # the slow cut: 28800.6 ticks -> 28801
    cut = compile_object(_cut(), SENS, SENS)
    step = float(np.hypot(*np.diff(cut["path_um"][:2], axis=0)[0]))
    assert cut["ticks_per_point"] == math.ceil(50_000 * step / 0.005) == 28_801
    # an exact integer is not bumped by float error: 0.3 um in 30 steps at 5 um/s
    exact = compile_object(LithoObject("e", "path", [(0, 0), (0.3, 0)],
                                       speed_um_s=5.0), SENS, SENS)
    assert exact["ticks_per_point"] == 100
    assert exact["speed_achieved_um_s"] == pytest.approx(5.0)
    # never above the request, and the timing follows the rounded-up tick
    for length in (0.004, 0.02, 0.05, 0.3, 1.7, 9.0):
        for speed in (0.005, 0.5, 3.0, 10.0, 37.0, 200.0):
            c = compile_object(LithoObject("s", "path", [(0, 0), (length, 0.3 * length)],
                                           speed_um_s=speed), SENS, SENS)
            assert c["speed_achieved_um_s"] <= speed * (1 + 1e-8)
            assert c["duration_s"] == pytest.approx(
                (len(c["v"]) - 1) * c["ticks_per_point"] / 50_000)
    seg = compile_pattern(Pattern("p", [_cut()]), SENS, SENS)["segments"][0]
    assert seg["ticks_per_point"] == 28_801
    # a transit: 1 um at 3 um/s in 0.02 um steps is 333.33 ticks -> 333, as on main
    cal = {"x_lvdt_sens": 7e-6, "x_lvdt_offset": -0.6,
           "y_lvdt_sens": 7e-6, "y_lvdt_offset": 0.07}
    _xs, _ys, interp, duration_s, n = _build_wavepair_v(0.0, 0.0, 1.0, 0.0, cal, 3.0)
    assert (n, interp) == (51, -333.0)
    assert duration_s == pytest.approx(50 * 333 / 50_000)
    path = np.column_stack([np.linspace(0.0, 1.0, 51), np.zeros(51)])
    assert time_segment(path, 3.0)["ticks_per_point"] == 333
    assert time_segment(path, 3.0, round_up=True)["ticks_per_point"] == 334
