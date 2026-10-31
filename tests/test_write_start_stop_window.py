"""A stop that arrives while a write is starting is never lost
(docs/TWIN_FRAME_RATE_SPEC.md, fourth round).

On hardware the write start reads the AC guard and, when withdrawn with no fresh
baseline, free air, and only then clears `_hw_litho_abort` (every park leaves it set).
An Abort, Withdraw, park or dead-man that landed during those reads was erased with
it: the write started and the hardware worker set the tip down. The start now notes
the stop counters when it begins and compares them at the last check before the
worker thread starts.

Each case lands the stop at one awaited read of the start, through both doors (the
REST/cockpit `/api/litho/write` and FLEX `startWrite`'s `start_write`), on a hub bound
to a FakeIgorBridge, and checks the refused start left the station as any refused
start does: no pass fired, no worker, no claim, interlocks disarmed, not `writing`,
and the next start runs.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from _im_fixtures_guard import GUARD_WRITE_KW, seed_free_air
from hub_fields import owner

from afm_litho.hardware.hardware_arc import HardwareARC
from afm_litho.hardware.igor_stub import FakeIgorBridge
from afm_litho.litho.compiler import LithoObject, Pattern
from afm_litho.service.host import litho_cmds
from afm_litho.service.commands.errors import REFUSED, CommandError
from afm_litho.service.host import engage
from afm_litho.service.host.safe_park import signal_run_abort

# `client` comes from tests/conftest.py.
GV = {"XLVDTSens": 4.9163e-6, "YLVDTSens": 4.59453e-6,
      "XLVDTOffset": -0.02, "YLVDTOffset": -0.38,
      "ZLVDTSens": 5e-6, "ZLVDTOffset": 0.0}
FREE_AIR_V = -0.2                        # `seed_free_air`'s default baseline
_PASS_VERBS = ("FIRE", "ARMFIRE", "LITHOCHAIN", "SCANCHAIN")
_KEPT = ("pattern", "written_objects", "z_loop", "imaging_mode", "free_air", "engaged")


def _wait(cond, timeout=30.0, interval=0.02) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return False


def _quiet(hub) -> bool:
    t = getattr(hub.host, "litho_thread", None)
    return (t is None or not t.is_alive()) and getattr(hub, "_run_claim", None) is None


def _design(hub) -> dict:
    saved = {k: getattr(owner(hub, k), k, None) for k in _KEPT}
    hub.pattern = Pattern(name="stop-window", objects=[
        LithoObject("wire", "path", [(-1.0, 0.0), (1.0, 0.0)], voltage=6.0,
                    speed_um_s=8.0, interpolation_um=0.1)])
    hub.host.written_objects = set()
    hub.host.z_loop = {"mode": "contact"}
    hub.host.imaging_mode = "contact"
    seed_free_air(hub)
    return saved


@pytest.fixture
def rig(client, tmp_path):
    """The service hub bound to a FakeIgorBridge, with a one-wire design loaded."""
    from afm_litho.service.hub import hub
    from afm_litho.service.sessions import SessionStore
    from afm_litho.service.xy_cal import note_xy_cal

    stub = FakeIgorBridge(dict(GV), done_delay=0.0)
    stub.values["DetectorSum"] = 5.0
    hw = HardwareARC.connect("127.0.0.1", stub.serve_in_thread())
    saved_hw = (hub.store, hub.host.hardware_arc, hub.host.hw_cal, dict(hub.host.xy_cal_status))
    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    hub.host.hardware_arc = hw
    hub.host.hw_cal = hw.start()
    note_xy_cal(hub, "connect")
    saved = _design(hub)
    try:
        yield hub, stub
    finally:
        _wait(lambda: _quiet(hub))
        for k, v in saved.items():
            setattr(owner(hub, k), k, v)
        hub.store, hub.host.hardware_arc, hub.host.hw_cal = saved_hw[:3]
        hub.host.xy_cal_status = saved_hw[3]
        hw.close()
        stub.stop()


@pytest.fixture
def twin(client, tmp_path):
    """The service hub on the twin, with the same design loaded."""
    from afm_litho.service.hub import hub
    from afm_litho.service.sessions import SessionStore
    prev_store = hub.store
    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    saved = _design(hub)
    try:
        yield hub
    finally:
        _wait(lambda: _quiet(hub), 60.0)
        for k, v in saved.items():
            setattr(owner(hub, k), k, v)
        hub.store = prev_store


def _start(door: str, client, hub) -> dict:
    """Start a write through one door. Returns {"ok": reply} or {"refused": reason}."""
    if door == "rest":
        body = client.post("/api/litho/write", json={
            "objects": ["wire"], "setpoint_relative": 0.2, **GUARD_WRITE_KW}).json()
        if body.get("ok"):
            return {"ok": body}
        return {"refused": body.get("reason"), "body": body}
    # FLEX speaks an absolute setpoint: 0.2 V above the baseline `seed_free_air` sets
    params = {"objects": "all", "setpoint": FREE_AIR_V + 0.2, "amp_gain": 25.0}
    fut = asyncio.run_coroutine_threadsafe(litho_cmds.start_write(hub, params), hub._loop)
    try:
        return {"ok": fut.result(30.0)}
    except CommandError as exc:
        assert exc.code == REFUSED, exc
        return {"refused": (exc.data or {}).get("reason"), "body": exc.message}


def _assert_refused_cleanly(hub, res, stub=None) -> None:
    assert res.get("refused") == "aborted_during_start", res
    if stub is not None:
        assert not any(v in _PASS_VERBS for v in stub.transcript), "a pass fired"
    t = getattr(hub.host, "litho_thread", None)
    assert t is None or not t.is_alive()
    assert getattr(hub, "_run_claim", None) is None
    assert hub._in_flight_run_id() is None
    assert hub.state_label() != "writing"
    wi = getattr(hub.host, "write_interlocks", None)
    assert wi is None or not wi.armed
    assert hub.host.written_objects == set()


def _assert_next_start_runs(door, client, hub, backend=None) -> None:
    res = _start(door, client, hub)
    assert "ok" in res, res
    if backend is not None:
        assert res["ok"].get("backend") == backend, res
    assert res["ok"].get("run_id")
    assert _wait(lambda: _quiet(hub), 60.0), "the next write never finished"


# ---- where, in the hardware start, the stop lands -------------------------------

def _stop_at_ac_guard(hub, monkeypatch, stop):
    real = litho_cmds.ac_mode_active

    def read_then_stop(h):
        out = real(h)
        stop(hub)
        return out

    monkeypatch.setattr(litho_cmds, "ac_mode_active", read_then_stop)


def _stop_at_free_air(hub, monkeypatch, stop):
    hub.host.free_air = None                  # withdrawn, no baseline: the start reads one
    hub.host.engaged = False

    async def read_then_stop(h, mode, force=False):
        seed_free_air(h, mode=mode)
        stop(hub)
        return h.host.free_air

    monkeypatch.setattr(engage, "establish_free_air_async", read_then_stop)


def _stop_at_xy_cal(hub, monkeypatch, stop):
    async def refresh_then_stop(source):
        stop(hub)

    monkeypatch.setattr(hub, "refresh_xy_cal", refresh_then_stop)


def _stop_at_sum_read(hub, monkeypatch, stop):
    real = engage.read_detector_sum

    def stop_then_read(h):
        stop(hub)
        return real(h)

    monkeypatch.setattr(engage, "read_detector_sum", stop_then_read)


def _scan_stop(hub):
    hub.host._scan_stop_seq += 1


WHERE = {"ac_guard": _stop_at_ac_guard, "free_air": _stop_at_free_air,
         "xy_cal": _stop_at_xy_cal, "sum_read": _stop_at_sum_read}


@pytest.mark.parametrize("door", ["rest", "flex"])
@pytest.mark.parametrize("where", list(WHERE))
def test_a_stop_during_a_hardware_start_read_refuses_the_write(
        rig, client, monkeypatch, where, door):
    hub, stub = rig
    with monkeypatch.context() as m:
        WHERE[where](hub, m, signal_run_abort)
        res = _start(door, client, hub)
    _assert_refused_cleanly(hub, res, stub)
    # the stop is over: the same start, asked again, runs on the instrument
    seed_free_air(hub)
    _assert_next_start_runs(door, client, hub, backend="hardware" if door == "rest"
                            else None)
    assert any(v in _PASS_VERBS for v in stub.transcript)


@pytest.mark.parametrize("door", ["rest", "flex"])
def test_a_scan_stop_during_a_hardware_start_read_refuses_the_write(
        rig, client, monkeypatch, door):
    hub, stub = rig
    with monkeypatch.context() as m:
        _stop_at_ac_guard(hub, m, _scan_stop)
        res = _start(door, client, hub)
    _assert_refused_cleanly(hub, res, stub)


@pytest.mark.parametrize("door", ["rest", "flex"])
def test_a_stop_before_the_start_began_does_not_refuse_it(rig, client, door):
    """Every park leaves the abort flags set, and a write after a park is the normal
    case: only a stop that arrives DURING the start refuses it."""
    hub, stub = rig
    signal_run_abort(hub)
    hub.host._scan_stop_seq += 1
    assert hub.host._hw_litho_abort.is_set()
    _assert_next_start_runs(door, client, hub)
    assert any(v in _PASS_VERBS for v in stub.transcript)


# ---- the twin start has the same rule --------------------------------------------

@pytest.mark.parametrize("door", ["rest", "flex"])
@pytest.mark.parametrize("stop", [signal_run_abort, _scan_stop],
                         ids=["abort", "scan_stop"])
def test_a_stop_during_a_twin_start_refuses_the_write(twin, client, monkeypatch,
                                                      door, stop):
    hub = twin
    real = litho_cmds._ensure_write_free_air

    async def read_then_stop(h):
        await real(h)
        stop(hub)

    with monkeypatch.context() as m:
        m.setattr(litho_cmds, "_ensure_write_free_air", read_then_stop)
        res = _start(door, client, hub)
    _assert_refused_cleanly(hub, res)
    _assert_next_start_runs(door, client, hub)


@pytest.mark.parametrize("door", ["rest", "flex"])
def test_a_stop_before_a_twin_start_does_not_refuse_it(twin, client, door):
    hub = twin
    signal_run_abort(hub)
    hub.host._scan_stop_seq += 1
    _assert_next_start_runs(door, client, hub)
