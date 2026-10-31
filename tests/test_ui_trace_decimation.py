"""The pump broadcasts a decimated COPY of the written-trace frames
(docs/UI_TRACE_DECIMATION_SPEC.md): the panel gets a few hundred points per
stroke, while everything the pump records keeps the worker's full frame.
"""

from __future__ import annotations

import asyncio
import copy
import json
import queue as _queue
import threading
import time

import numpy as np
import pytest

from afm_litho.service.host import litho_worker as rl
from afm_litho.service.host.litho_worker import (
    _UI_TRACE_MAX_POINTS,
    _UI_TRACE_TOL_UM,
    _ui_trace_frame,
)


def _zigzag(n_rows: int, per_row: int, size_um: float = 20.0):
    """A rectangle fill: rows along x, alternating direction, stepping in y."""
    xs = np.linspace(0.0, size_um, per_row)
    pts = []
    for r in range(n_rows):
        y = size_um * r / max(n_rows - 1, 1)
        row = xs if r % 2 == 0 else xs[::-1]
        pts.extend([round(float(x), 3), round(float(y), 3)] for x in row)
    return pts


def _done(path, v=None, **extra):
    msg = {"type": "litho_object_done", "name": "rect#0", "object": "rect",
           "voltage": 10.0, "profile": None, "path_um": path}
    if v is not None:
        msg["v_written"] = v
    return msg | extra


def _dist_to_polyline(pt, poly) -> float:
    p = np.asarray(poly, dtype=float)
    a, b = p[:-1], p[1:]
    d = b - a
    L2 = np.maximum((d ** 2).sum(axis=1), 1e-18)
    t = np.clip(((pt - a) * d).sum(axis=1) / L2, 0.0, 1.0)
    return float(np.min(np.hypot(*(a + t[:, None] * d - pt).T)))


# ------------------------------------------------------------- the pure helper --

def test_100k_point_stroke_is_capped_with_equal_lengths_and_endpoints():
    path = _zigzag(400, 250)                                   # 100,000 points
    v = [10.0] * len(path)
    msg = _done(path, v)
    before = copy.deepcopy(msg)
    t0 = time.perf_counter()
    ui = _ui_trace_frame(msg)
    took = time.perf_counter() - t0
    assert msg == before, "the worker's frame was modified"
    assert ui is not msg and ui["path_um"] is not msg["path_um"]
    assert len(ui["path_um"]) <= _UI_TRACE_MAX_POINTS
    assert len(ui["path_um"]) == len(ui["v_written"])
    assert ui["path_um"][0] == path[0] and ui["path_um"][-1] == path[-1]
    assert ui["v_written"][0] == v[0] and ui["v_written"][-1] == v[-1]
    assert len(json.dumps(ui)) < len(json.dumps(msg)) / 20
    assert took < 2.0
    for k in ("type", "name", "object", "voltage", "profile"):
        assert ui[k] == msg[k]


def test_zigzag_keeps_every_row_turnaround():
    n_rows, per_row = 60, 200
    path = _zigzag(n_rows, per_row)
    ui = _ui_trace_frame(_done(path, [10.0] * len(path)))
    kept = ui["path_um"]
    assert len(kept) < len(path) / 10
    for r in range(n_rows):
        for corner in (path[r * per_row], path[(r + 1) * per_row - 1]):
            assert _dist_to_polyline(np.array(corner), kept) <= _UI_TRACE_TOL_UM
            assert corner in kept


def test_partial_frame_is_decimated_the_same_way():
    path = _zigzag(100, 250)
    msg = {"type": "litho_object_partial", "name": "rect#0", "object": "rect",
           "object_index": 0, "voltage": 10.0, "frac0": 0.0, "frac1": 0.4,
           "path_um": path, "v_written": [10.0] * len(path)}
    ui = _ui_trace_frame(msg)
    assert len(ui["path_um"]) == len(ui["v_written"]) <= _UI_TRACE_MAX_POINTS
    assert len(ui["path_um"]) < len(path) / 10
    assert ui["path_um"][0] == path[0] and ui["path_um"][-1] == path[-1]
    assert (ui["frac0"], ui["frac1"], ui["object_index"]) == (0.0, 0.4, 0)


def test_transit_without_v_written_decimates_the_path_only():
    path = [[round(i * 0.001, 3), 0.0] for i in range(5000)]
    msg = {"type": "litho_transit", "mode": "lift", "distance_um": 5.0,
           "seconds": 1.0, "peak_um_s": None, "path_um": path}
    ui = _ui_trace_frame(msg)
    assert ui["path_um"] == [path[0], path[-1]]
    assert "v_written" not in ui


def test_modulated_bias_on_a_straight_wire_keeps_its_changes():
    n = 20000
    path = [[round(i * 0.001, 3), 0.0] for i in range(n)]
    v = [round(10.0 + 2.0 * np.sin(2 * np.pi * i / 2000.0), 3) for i in range(n)]
    ui = _ui_trace_frame(_done(path, v))
    assert len(ui["path_um"]) == len(ui["v_written"]) <= _UI_TRACE_MAX_POINTS
    assert 20 < len(ui["path_um"]) < n / 10
    # the kept samples reproduce the full bias by linear interpolation along x
    kx = [p[0] for p in ui["path_um"]]
    rebuilt = np.interp([p[0] for p in path], kx, ui["v_written"])
    assert float(np.max(np.abs(rebuilt - np.asarray(v)))) < 0.15
    assert max(ui["v_written"]) > 11.9 and min(ui["v_written"]) < 8.1


def test_bias_step_keeps_both_sides_of_the_edge():
    n = 3000
    path = [[round(i * 0.001, 3), 0.0] for i in range(n)]
    v = [5.0] * 1500 + [10.0] * 1500
    ui = _ui_trace_frame(_done(path, v))
    assert path[1499] in ui["path_um"] and path[1500] in ui["path_um"]
    assert len(ui["path_um"]) == len(ui["v_written"]) <= 8


def test_cap_applies_when_the_shape_itself_needs_more_points():
    rng = np.random.default_rng(0)
    path = [[round(float(x), 3), round(float(y), 3)]
            for x, y in rng.uniform(0, 20, size=(30000, 2))]
    v = [10.0] * len(path)
    ui = _ui_trace_frame(_done(path, v))
    assert len(ui["path_um"]) == len(ui["v_written"]) == _UI_TRACE_MAX_POINTS
    assert ui["path_um"][0] == path[0] and ui["path_um"][-1] == path[-1]


def test_surface_samples_is_dropped_from_the_copy_only():
    ss = {"x_um": [0.0] * 8, "y_um": [0.0] * 8, "z_nm": [0.0] * 8}
    msg = _done([[0.0, 0.0], [1.0, 0.0]], [1.0, 1.0], surface_samples=ss)
    ui = _ui_trace_frame(msg)
    assert "surface_samples" not in ui
    assert msg["surface_samples"] is ss


@pytest.mark.parametrize("msg", [
    {"type": "litho_progress", "name": "a", "frac": 1.0, "x_um": 1.0, "y_um": 2.0},
    {"type": "tip_pos", "x_um": 1.0, "y_um": 2.0},
    {"type": "litho_done", "aborted": False},
    {"type": "litho_object_start", "name": "a", "length_um": 1.0, "voltage": 1.0,
     "duration_s": 1.0, "path_um": [[0.0, 0.0]] * 10},
    {"type": "log", "t": 0.0, "msg": "x", "surface_samples": {"x_um": []}},
])
def test_other_frame_types_are_the_same_object(msg):
    before = json.dumps(msg)
    assert _ui_trace_frame(msg) is msg
    assert json.dumps(msg) == before


@pytest.mark.parametrize("frame", [
    _done([]),
    _done([[0.0, 0.0]], [1.0]),
    _done([[0.0, 0.0], [1.0, 1.0]]),                       # no v_written
    _done(_zigzag(20, 100), None, v_written=None),
    _done(_zigzag(20, 100), [10.0] * 7),                   # v shorter
    _done(_zigzag(2, 5), [10.0] * 500),                    # v longer
    {"type": "litho_transit", "mode": "lift"},             # no path at all
])
def test_odd_frames_do_not_raise(frame):
    before = copy.deepcopy(frame)
    ui = _ui_trace_frame(frame)
    assert frame == before
    assert len(ui.get("path_um") or []) <= max(len(frame.get("path_um") or []), 0)
    json.dumps(ui)


def test_mismatched_lengths_stay_registered_by_index():
    path = _zigzag(20, 100)
    v = [float(i) for i in range(700)]                      # shorter, all distinct
    ui = _ui_trace_frame(_done(path, v))
    for p, val in zip(ui["path_um"], ui["v_written"]):
        assert path[int(val)] == p


def test_rotated_zigzag_keeps_every_row_turnaround_within_tolerance():
    n_rows, per_row = 60, 200
    th = np.deg2rad(17.0)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    path = np.round(np.asarray(_zigzag(n_rows, per_row)) @ rot.T, 3).tolist()
    kept = _ui_trace_frame(_done(path, [10.0] * len(path)))["path_um"]
    assert len(kept) < len(path) / 10
    for r in range(n_rows):
        for corner in (path[r * per_row], path[(r + 1) * per_row - 1]):
            assert _dist_to_polyline(np.array(corner), kept) <= _UI_TRACE_TOL_UM


def test_collinear_out_and_back_keeps_its_turnaround():
    path = ([[round(i * 0.01, 3), 0.0] for i in range(1000)]
            + [[round(10 - i * 0.01, 3), 0.0] for i in range(1000)])
    ui = _ui_trace_frame(_done(path, [10.0] * len(path)))
    assert [10.0, 0.0] in ui["path_um"]
    assert ui["path_um"][0] == path[0] and ui["path_um"][-1] == path[-1]
    assert len(ui["path_um"]) == len(ui["v_written"]) <= 6


def test_arc_stays_within_the_12_5_nm_tolerance():
    """A quarter circle of radius 5 um. At ten times the tolerance the kept chords
    would sag up to 125 nm, so the deviation bound below fails."""
    ang = np.linspace(0.0, np.pi / 2, 20001)
    path = np.round(np.column_stack((5 * np.cos(ang), 5 * np.sin(ang))), 6).tolist()
    kept = _ui_trace_frame(_done(path, [10.0] * len(path)))["path_um"]
    assert 12 <= len(kept) < 200
    worst = max(_dist_to_polyline(np.array(pt), kept) for pt in path[::20])
    assert worst <= _UI_TRACE_TOL_UM * 1.001


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_points_in_a_path_do_not_raise(bad):
    path = _zigzag(40, 100)
    path[1234] = [bad, 1.0]
    path[2500] = [3.0, bad]
    v = [10.0] * len(path)
    v[77] = bad
    msg = _done(path, v)
    ui = _ui_trace_frame(msg)
    assert msg["path_um"] is path and len(path) == 4000
    assert len(ui["path_um"]) == len(ui["v_written"]) <= _UI_TRACE_MAX_POINTS
    assert ui["path_um"][0] == path[0] and ui["path_um"][-1] == path[-1]


# --------------------------------------------------------------- time bounds --

class _FakeWs:
    def __init__(self):
        self.frames: list[str] = []

    async def send_text(self, data: str):
        self.frames.append(data)


def _np_zigzag(n_rows, per_row, size=20.0):
    xs = np.linspace(0.0, size, per_row)
    x = np.tile(xs, (n_rows, 1))
    x[1::2] = x[1::2, ::-1]
    y = np.repeat(np.linspace(0.0, size, n_rows), per_row)
    return np.round(np.column_stack((x.ravel(), y)), 3).tolist()


def _sine(n, cycles):
    return np.round(10.0 + 2.0 * np.sin(2 * np.pi * cycles * np.arange(n) / n),
                    3).tolist()


def _chiral(n, cycles):
    ph = 2 * np.pi * cycles * np.arange(n) / n
    path = np.round(np.column_stack((50.0 * np.arange(n) / n, 0.05 * np.sin(ph))),
                    3).tolist()
    return path, np.round(10.0 + 2.0 * np.cos(ph), 3).tolist()


def _worst_case(kind):
    if kind == "500k sine 5000 cycles":
        return _np_zigzag(1000, 500), _sine(500_000, 5000)
    if kind == "500k square period 100":
        v = np.where((np.arange(500_000) // 50) % 2 == 0, 5.0, 10.0).tolist()
        return _np_zigzag(1000, 500), v
    if kind == "500k rotated fill, sine":
        th = np.deg2rad(17.0)
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        path = np.round(np.asarray(_np_zigzag(1000, 500)) @ rot.T, 3).tolist()
        return path, _sine(500_000, 5000)
    if kind == "500k random walk":
        rng = np.random.default_rng(0)
        path = np.round(np.cumsum(rng.normal(0, 0.05, (500_000, 2)), axis=0),
                        3).tolist()
        return path, _sine(500_000, 5000)
    return _chiral(250_000, 25_000)


@pytest.mark.parametrize("kind", [
    "500k sine 5000 cycles", "500k square period 100", "500k rotated fill, sine",
    "500k random walk", "chiral 250k"])
def test_worst_case_frames_finish_within_the_cpu_budget(kind):
    path, v = _worst_case(kind)
    msg = _done(path, v)
    # This thread's CPU time, best of three: a loaded machine adds cache misses and
    # preemption to a single run, but a real 2x regression slows every run.
    runs = []
    for _ in range(3):
        c0 = time.thread_time()
        ui = _ui_trace_frame(msg)
        runs.append(time.thread_time() - c0)
    cpu = min(runs)
    assert cpu <= 1.0, f"{kind}: {cpu:.2f} s of CPU"
    assert len(ui["path_um"]) == len(ui["v_written"]) <= _UI_TRACE_MAX_POINTS
    assert ui["path_um"][0] == path[0] and ui["path_um"][-1] == path[-1]
    # above the ceiling the modulation is aliased, but its range is the true one
    assert min(ui["v_written"]) == min(v) and max(ui["v_written"]) == max(v)


def test_the_loop_keeps_ticking_while_a_500k_modulated_frame_is_decimated():
    from afm_litho.service.hub import hub

    path, v = _worst_case("500k sine 5000 cycles")
    msg = _done(path, v)
    ws = _FakeWs()

    async def go():
        gaps = []
        stop = asyncio.Event()

        async def ticker():
            last = time.perf_counter()
            while not stop.is_set():
                await asyncio.sleep(0.01)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        tick = asyncio.create_task(ticker())
        await asyncio.sleep(0.05)
        await rl._broadcast_ui(msg)
        await asyncio.sleep(0.05)
        stop.set()
        await tick
        return max(gaps)

    saved = set(hub.clients)
    hub.clients.clear()
    hub.clients.add(ws)
    try:
        worst = asyncio.run(go())
    finally:
        hub.clients.discard(ws)
        hub.clients.update(saved)
    assert worst <= 0.1, f"the loop stalled {worst * 1000:.0f} ms"
    (raw,) = ws.frames
    assert len(json.loads(raw)["path_um"]) <= _UI_TRACE_MAX_POINTS


def test_the_fallback_does_not_raise_when_the_failure_log_fails_too(monkeypatch):
    from afm_litho.service.hub import hub

    def boom(_msg):
        raise RuntimeError("decimator down")

    async def log_boom(_text):
        raise RuntimeError("log down")

    monkeypatch.setattr(rl, "_ui_trace_failure_logged", False)
    monkeypatch.setattr(rl, "_ui_trace_frame", boom)
    monkeypatch.setattr(hub, "log", log_boom)
    msg = _done([[0.0, 0.0], [1.0, 0.0]], [1.0, 1.0])
    ws = _FakeWs()
    saved = set(hub.clients)
    hub.clients.clear()
    hub.clients.add(ws)
    try:
        asyncio.run(rl._broadcast_ui(msg))
    finally:
        hub.clients.discard(ws)
        hub.clients.update(saved)
    assert ws.frames == [json.dumps(msg)]


# ------------------------------------------------------------------ the pump --

def _run_pump(hub, tmp_path, msgs):
    """Drive the real pump over `msgs`; returns (run_id, raw ws frames)."""
    from afm_litho.service.sessions import SessionStore
    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    q = _queue.Queue()
    for m in msgs:
        q.put(m)
    worker = threading.Thread(target=lambda: None, daemon=True)
    claim = hub.claim_run("write")
    run_id = hub.next_run_id("write")
    claim.hand_over(run_id, worker)
    worker.start()
    worker.join(timeout=5)
    ws = _FakeWs()
    saved = set(hub.clients)
    hub.clients.clear()
    hub.clients.add(ws)
    try:
        asyncio.run(asyncio.wait_for(rl._litho_pump(q, run_id, claim), timeout=60))
    finally:
        hub.clients.discard(ws)
        hub.clients.update(saved)
    return run_id, ws.frames


@pytest.fixture()
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


def test_pump_sends_a_small_frame_and_records_the_full_one(hub, tmp_path):
    from afm_litho.service import flex_dispatch as fd

    path = _zigzag(400, 250)
    v = [10.0] * len(path)
    done = _done(path, v, duration_s=12.0, length_um=8000.0)
    progress = {"type": "litho_progress", "name": "rect#0", "frac": 1.0,
                "x_um": 0.0, "y_um": 20.0}
    start = {"type": "litho_object_start", "name": "rect#0", "length_um": 8000.0,
             "voltage": 10.0, "duration_s": 12.0}
    transit = {"type": "litho_transit", "mode": "lift", "distance_um": 3.0,
               "seconds": 0.5, "peak_um_s": None,
               "path_um": [[round(i * 0.001, 3), 0.0] for i in range(3000)]}
    end = {"type": "litho_done", "aborted": False}
    originals = copy.deepcopy([transit, start, progress, done, end])

    run_id, frames = _run_pump(hub, tmp_path, [transit, start, progress, done, end])

    assert [transit, start, progress, done, end] == originals, "the pump mutated a frame"
    by_type = {}
    for raw in frames:
        by_type.setdefault(json.loads(raw)["type"], []).append(raw)

    (raw_done,) = by_type["litho_object_done"]
    ui = json.loads(raw_done)
    assert len(ui["path_um"]) == len(ui["v_written"]) <= _UI_TRACE_MAX_POINTS
    assert ui["path_um"][0] == path[0] and ui["path_um"][-1] == path[-1]
    assert len(raw_done) < len(json.dumps(done)) / 20
    (raw_transit,) = by_type["litho_transit"]
    assert json.loads(raw_transit)["path_um"] == [transit["path_um"][0],
                                                  transit["path_um"][-1]]

    # every other frame the worker queued reaches the socket byte for byte
    assert by_type["litho_object_start"] == [json.dumps(start)]
    assert by_type["litho_progress"] == [json.dumps(progress)]
    assert by_type["litho_done"] == [json.dumps(end)]

    # the bookkeeping kept the worker's full-resolution data
    (entry,) = hub.host.written_accum
    assert entry["path_um"] == path and entry["v_written"] == v
    written = fd._h_get_written(hub, {})
    objs = written["objects"] if isinstance(written, dict) else written
    assert objs[0]["points"] == len(path)
    recs = [json.loads(line) for line in
            (tmp_path / "sessions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert recs[-1]["kind"] == "litho"
    assert recs[-1]["written"][0]["path_um"] == path
    assert recs[-1]["written"][0]["v_written"] == v
    rec = hub.runs.get(run_id)
    assert rec.finished is not None and rec.aborted is False
    assert rec.stats["objects"] == 1


def test_pump_partial_frame_is_small_on_the_wire_and_full_in_the_slot(hub, tmp_path):
    path = _zigzag(200, 250)
    partial = {"type": "litho_object_partial", "name": "rect#0", "object": "rect",
               "object_index": 0, "voltage": 10.0, "frac0": 0.0, "frac1": 0.5,
               "path_um": path, "v_written": [10.0] * len(path)}
    seen = {}
    real_finish = rl._litho_finish

    async def spy(msg, *a, **kw):
        seen["partial"] = copy.deepcopy(hub.host._litho_partial)
        return await real_finish(msg, *a, **kw)

    rl._litho_finish = spy
    try:
        _run_id, frames = _run_pump(
            hub, tmp_path, [partial, {"type": "litho_done", "aborted": True}])
    finally:
        rl._litho_finish = real_finish
    (ui,) = [f for f in map(json.loads, frames) if f["type"] == "litho_object_partial"]
    assert len(ui["path_um"]) == len(ui["v_written"]) <= _UI_TRACE_MAX_POINTS
    assert len(ui["path_um"]) < len(path) / 10
    assert seen["partial"]["path_um"] == path
    assert len(seen["partial"]["v_written"]) == len(path)
    assert hub.host.written_accum == []


def test_pump_ingests_the_full_surface_samples_but_does_not_broadcast_them(
        hub, tmp_path, monkeypatch):
    ss = {"x_um": [0.1 * i for i in range(10)], "y_um": [0.0] * 10,
          "z_nm": [1.0] * 10}
    got = {}
    monkeypatch.setattr(hub.surface, "ingest_path",
                        lambda cid, x, y, z: got.update(x=x, y=y, z=z))
    done = _done([[0.0, 0.0], [1.0, 0.0]], [10.0, 10.0], surface_samples=ss)
    _run_id, frames = _run_pump(hub, tmp_path,
                                [done, {"type": "litho_done", "aborted": False}])
    (ui,) = [f for f in map(json.loads, frames) if f["type"] == "litho_object_done"]
    assert "surface_samples" not in ui
    assert got == {"x": ss["x_um"], "y": ss["y_um"], "z": ss["z_nm"]}


def test_a_frame_the_decimator_cannot_read_goes_out_whole_and_the_run_closes(
        hub, tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "_ui_trace_failure_logged", False)
    bad_path = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]     # not (n, 2)
    bad = _done(bad_path, ["x", "y", "z"])
    bad2 = _done(bad_path, [1.0, 1.0, 1.0], name="rect#1")
    run_id, frames = _run_pump(
        hub, tmp_path, [bad, bad2, {"type": "litho_done", "aborted": False}])
    sent = [f for f in map(json.loads, frames) if f["type"] == "litho_object_done"]
    assert [f["path_um"] for f in sent] == [bad_path, bad_path]
    logs = [f for f in map(json.loads, frames)
            if f["type"] == "log" and "decimation failed" in f["msg"]]
    assert len(logs) == 1, "the failure is reported once, not per frame"
    assert len(hub.host.written_accum) == 2
    assert hub.runs.get(run_id).finished is not None
    assert hub._run_claim is None
