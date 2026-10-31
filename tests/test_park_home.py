"""Slice 2.7b: one home for the park.

Every park trigger calls `safe_park.request_park`; every stop flag is set by
`safe_park._signal_stop`; every retract recipe lives in `hardware/retract.py`. These
tests pin that routing. The Igor commands themselves are pinned by tests/golden.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

import afm_litho
import importlib

sp = importlib.import_module("afm_litho.service.host.safe_park")

SRC = Path(afm_litho.__file__).parent
FULL, WATCHDOG, ARC_ABORT, WITHDRAW, DEFERRED, DROP = (
    sp.PARK_FULL, sp.PARK_WATCHDOG, sp.PARK_ARC_ABORT, sp.PARK_WITHDRAW,
    sp.PARK_DEFERRED, sp.PARK_BRIDGE_DROP)


# ------------------------------------------------------------ no stray stop flags --

_STOP_EVENTS = {"_hw_abort", "_hw_litho_abort", "_scan_repeat_stop", "_safe_park_now"}


def _stray_stop_writes(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "set"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr in _STOP_EVENTS):
            found.append(f"{path.name}:{node.lineno} {node.func.value.attr}.set()")
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Attribute) and t.attr == "_hw_withdraw_after"
                        and getattr(node.value, "value", None) is True):
                    found.append(f"{path.name}:{node.lineno} _hw_withdraw_after =")
                if (isinstance(t, ast.Attribute) and t.attr == "abort"
                        and isinstance(t.value, ast.Attribute)
                        and t.value.attr == "litho_run"):
                    found.append(f"{path.name}:{node.lineno} litho_run.abort =")
    return found


def test_no_stop_flag_is_set_outside_signal_stop():
    stray = []
    for path in SRC.rglob("*.py"):
        if path.name == "safe_park.py" and path.parent.name == "host":
            continue
        stray += _stray_stop_writes(path)
    assert stray == []


# --------------------------------------------------------------- the spy ----------

class _Calls(list):
    pass


@pytest.fixture
def parks(monkeypatch):
    """Replace `request_park` in every module that calls it; record (reason, how)."""
    from afm_litho.service import hw_scan
    from afm_litho.service.host import continuous as cont
    from afm_litho.service.host import scan_pump
    from afm_litho.service.routers import scan as scan_mod
    from afm_litho.service.host import engage

    calls = _Calls()

    def spy(hub, reason, how=FULL, **kw):
        calls.append((reason, how))
        if how == FULL:
            async def _done():
                if hub is not None and isinstance(getattr(hub.host, "_safe_park", None), dict):
                    hub.host._safe_park["active"] = True
                return {"reason": reason, "backend": "twin", "transport": "twin",
                        "verified": True, "z_v": None, "critical": False}
            return _done()
        if how == WITHDRAW:
            async def _w():
                return {}
            return _w()
        return None

    for mod in (sp, scan_mod, scan_pump, cont, hw_scan, engage):
        monkeypatch.setattr(mod, "request_park", spy)
    return calls


# ------------------------------------------------------------------ the triggers --

def _hub():
    from afm_litho.service.app import hub
    return hub


def test_rest_safe_park_reaches_request_park(client, parks):
    r = client.post("/api/safe_park", json={"reason": "because"})
    assert r.status_code == 200
    assert parks == [("because", FULL)]


def test_scan_stop_without_a_pass_reaches_request_park(parks, monkeypatch):
    from afm_litho.service.host import scan_cmds
    from afm_litho.service.routers import scan as scan_mod
    hub = _hub()
    monkeypatch.setattr(scan_cmds, "_real_bridge", lambda h: object())
    monkeypatch.setattr(hub.host, "_scan_task", None)
    asyncio.run(scan_mod.scan_abort({}))
    hub.host._hw_abort.clear()
    assert parks == [("scan stopped", FULL)]


def test_scan_stop_with_a_pass_defers_through_request_park(parks, monkeypatch):
    from afm_litho.service.host import scan_cmds
    from afm_litho.service.routers import scan as scan_mod
    hub = _hub()
    monkeypatch.setattr(scan_cmds, "_real_bridge", lambda h: object())
    monkeypatch.setattr(hub.host, "_scan_task", SimpleNamespace(done=lambda: False))
    asyncio.run(scan_mod.scan_abort({}))
    hub.host._hw_abort.clear()
    assert parks == [("scan stopped", DEFERRED)]


def test_deferred_scan_stop_withdraw_reaches_request_park(parks):
    from afm_litho.service.host import scan_pump
    hub = _hub()
    hub.host._hw_withdraw_after = True
    hub.host._hw_withdraw_epoch = None
    asyncio.run(scan_pump._deferred_withdraw_if_current())
    assert parks == [("scan stopped", FULL)]


def test_plain_withdraw_reaches_request_park(parks):
    from afm_litho.service.host.deadman import _withdraw
    asyncio.run(_withdraw(_hub()))          # the dead-man and engaged-idle retract
    assert parks == [("withdraw", WITHDRAW)]


class _RunHere:
    """A FLEX ctx whose `on_loop` runs the coroutine to completion in this thread."""

    @staticmethod
    def on_loop(coro, method):
        return asyncio.run(coro)


def test_both_withdraw_doors_reach_request_park(client, parks):
    from afm_litho.service import flex_dispatch as fd
    assert client.post("/api/withdraw", json={}).status_code == 200
    assert fd._h_withdraw(_hub(), {}, _RunHere())["ok"] is True
    assert parks == [("withdraw", WITHDRAW), ("withdraw", WITHDRAW)]


def test_a_withdraw_park_that_raises_fails_both_doors(client, monkeypatch):
    from afm_litho.service import flex_dispatch as fd

    def broken(hub, reason, how=FULL, **kw):
        raise RuntimeError("the withdraw park was reached")

    monkeypatch.setattr(sp, "request_park", broken)
    try:
        status = client.post("/api/withdraw", json={}).status_code
    except RuntimeError:
        status = None
    assert status != 200
    with pytest.raises(RuntimeError, match="withdraw park was reached"):
        fd._h_withdraw(_hub(), {}, _RunHere())


def test_deadman_lapse_reaches_request_park(parks, monkeypatch):
    from afm_litho.service import control
    from afm_litho.service.control import _deadman_tick
    from afm_litho.service.host import deadman

    class _Lock:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    async def _noop(*a, **k):
        return None

    cur = SimpleNamespace(deadman_s=5.0, holder="x", token="t",
                          expired=lambda now: True, withdraw_on_deadman=False)
    hub = SimpleNamespace(control=cur, _control_lock=_Lock(), log=_noop,
                          _remember_revoked=lambda t: None, broadcast_control=_noop)
    hub.host = hub  # fake: host fields live on the fake itself
    monkeypatch.setattr(deadman, "pass_state", lambda h: "scanning")
    asyncio.run(_deadman_tick(hub, 0.0))
    assert parks == [("deadman", FULL)]
    assert control is not None


def test_approach_failure_reaches_request_park(monkeypatch):
    from afm_litho.service.host import scan_cmds

    calls = []

    async def _fake_request_park(hub, reason, how=FULL, **kw):
        calls.append((reason, how))

    async def _boom(hub, body):
        raise RuntimeError("no contact")

    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(sp, "request_park", _fake_request_park)
    monkeypatch.setattr(scan_cmds, "_approach_impl", _boom)
    hub = SimpleNamespace(finish_run=lambda *a, **k: None, log=_noop,
                          release_claim=lambda c: None, _approach_in_flight=1,
                          pass_ended=lambda *a: None)
    hub.host = hub  # fake: host fields live on the fake itself
    asyncio.run(scan_cmds._approach_run(hub, {}, "run-1", object()))
    assert calls == [("approach failed: RuntimeError: no contact", FULL)]


def _twin_spy():
    """A stand-in twin plugin on a stub hub: `aborted` says whether it was told to stop."""
    twin = SimpleNamespace(aborted=False)
    twin.abort = lambda reason: setattr(twin, "aborted", True)
    return twin


def test_litho_abort_reaches_request_park(monkeypatch):
    from afm_litho.service.host.litho_cmds import abort_write

    calls = []

    async def _fake_request_park(hub, reason, how=FULL, **kw):
        calls.append((reason, how))
        return {"backend": "twin"}

    monkeypatch.setattr(sp, "request_park", _fake_request_park)
    hub = SimpleNamespace(_twin_plugin=_twin_spy())
    hub.host = hub  # fake: host fields live on the fake itself
    asyncio.run(abort_write(hub))
    assert calls == [("litho abort", FULL)]
    assert hub.host._twin_plugin.aborted is True


def test_continuous_termination_reaches_request_park(client, tmp_path, parks):
    from test_safe_park import _stop_and_reset, _tmp_store, _TINY, _wait
    hub = _tmp_store(tmp_path)
    _stop_and_reset(client, hub)
    try:
        client.post("/api/scan/continuous",
                    json={"enabled": True, "config": _TINY, "max_frames": 2})
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False)
        assert any("max_frames" in r and how == FULL for r, how in parks), parks
    finally:
        _stop_and_reset(client, hub)


def test_reref_failure_reaches_request_park(client, tmp_path, parks, monkeypatch):
    import math

    from test_safe_park import _arm_relative, _stop_and_reset, _tmp_store, _TINY, _wait

    import afm_litho.service.host.continuous as cont

    async def _sum():
        return math.nan

    hub = _tmp_store(tmp_path)
    _stop_and_reset(client, hub)
    monkeypatch.setattr(cont, "_read_sum", _sum)
    try:
        _arm_relative(client, _TINY)
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False)
        assert any("SUM read failed" in r and how == FULL for r, how in parks), parks
    finally:
        _stop_and_reset(client, hub)


def test_rail_dwell_trip_reaches_request_park(parks, monkeypatch):
    from afm_litho.service.host import continuous as cont
    hub = _hub()

    class _Guard:
        def __init__(self):
            self.tripped = {"reason": "rail", "withdrawn": False}

        def readout(self):
            return {}

    monkeypatch.setattr(cont, "_real_bridge", lambda h: None)
    asyncio.run(cont._rail_withdraw(_Guard()))
    assert parks == [("rail", FULL)]

    parks.clear()
    monkeypatch.setattr(cont, "_real_bridge", lambda h: object())
    monkeypatch.setattr(hub.host, "_scan_task", SimpleNamespace(done=lambda: False))
    asyncio.run(cont._rail_withdraw(_Guard()))
    assert parks == [("rail-dwell trip", DEFERRED)]


def test_interlock_trip_and_hw_scan_abort_call_request_park():
    """These two sit inside long-running loops (the supervisor, the chained-scan
    reader); pin the call itself rather than drive hardware loops."""
    from afm_litho.service import hw_scan
    from afm_litho.service.host import continuous as cont
    assert 'request_park(hub, ilk["reason"], full_park=safe_park)' in inspect.getsource(cont)
    src = inspect.getsource(hw_scan)
    assert "PARK_ARC_ABORT, arc=arc)" in src and "arc.abort()" not in src


def test_write_interlock_trip_reaches_request_park(parks, monkeypatch):
    from afm_litho.service.host import engage
    hub = _hub()
    monkeypatch.setattr(engage, "real_bridge", lambda h: object())
    monkeypatch.setattr(engage, "read_detector_sum", lambda h: 1.0)
    before = getattr(hub.host, "write_interlocks", None)
    try:
        engage.arm_interlocks(hub)
        hub.host.write_interlocks._on_trip("sum lost")
    finally:
        hub.host.write_interlocks = before
    assert parks == [("sum lost", WATCHDOG)]


def test_litho_end_of_run_withdraw_failure_reaches_request_park(parks, monkeypatch):
    import queue

    from afm_litho.hardware import hw_approach
    from afm_litho.service.host import litho_worker as litho_router

    def _raise(*a, **kw):
        raise OSError("bridge gone")

    hub = _hub()
    monkeypatch.setattr(hw_approach, "hw_withdraw", _raise)
    monkeypatch.setattr(hub.host, "hardware_arc", object(), raising=False)
    saved_fault = hub.fault
    hub.fault = None
    try:
        litho_router._hw_litho_worker(queue.Queue(), hub.pattern, 0.2, 1.0)
    finally:
        hub.fault = saved_fault
        hub.host.engaged = False
    assert len(parks) == 1
    reason, how = parks[0]
    assert how == WATCHDOG and reason.startswith("litho end-of-run withdraw failed")


def test_bridge_drop_hands_the_park_to_the_watchdog_else_fires_directly(monkeypatch):
    logs = []
    taken = []
    wd = SimpleNamespace(alive=True, park_now=lambda reason: taken.append(reason))
    hub = SimpleNamespace(_failsafe=wd)
    hub.host = hub  # fake: host fields live on the fake itself
    sp.request_park(hub, "bridge_closed", DROP, log=logs.append)
    assert taken == ["bridge_closed"]

    sent = []
    import afm_litho.hardware.bridge_router as br
    monkeypatch.setattr(br, "send_safe_park", lambda link, com, log=None: sent.append(1))
    hub = SimpleNamespace(_failsafe=None, hardware_arc=SimpleNamespace())
    hub.host = hub  # fake: host fields live on the fake itself
    sp.request_park(hub, "bridge_closed", DROP, log=logs.append)
    assert sent == [1] and "NO fail-safe watchdog" in logs[0]


def test_hub_mark_bridge_closed_reaches_request_park(parks):
    import threading

    from afm_litho.service.hub import Hub
    stub = SimpleNamespace(bridge_closed=threading.Event(), run_in_flight=lambda: True,
                           _log_sync=lambda m: None)
    stub.host = stub  # fake: host fields live on the fake itself
    Hub.mark_bridge_closed(stub, "dropped")
    assert parks == [("bridge_closed", DROP)]


def test_a_failing_park_request_cannot_raise_on_the_reader_thread(monkeypatch):
    import threading

    from afm_litho.service.hub import Hub

    def _boom(*a, **k):
        raise RuntimeError("boom")

    logs = []
    monkeypatch.setattr(sp, "request_park", _boom)
    stub = SimpleNamespace(bridge_closed=threading.Event(), run_in_flight=lambda: True,
                           _log_sync=logs.append)
    stub.host = stub  # fake: host fields live on the fake itself
    Hub.mark_bridge_closed(stub, "dropped")
    assert "SAFETY CRITICAL" in logs[0]


def test_flex_abort_and_safe_park_reach_request_park(monkeypatch):
    from afm_litho.service import flex_dispatch as fd
    from afm_litho.service.host.api import HostApi
    src = inspect.getsource(fd._h_abort) + inspect.getsource(fd._h_safe_park)
    assert src.count("await HostApi(hub).park(reason, stop_id=stop_id)") == 2
    assert "safe_park_async" not in src
    assert "request_park(self._hub, why, stop_id=stop_id)" in inspect.getsource(HostApi.park)


# ------------------------------------------------------- the styles and the flags --

def test_request_park_styles_do_what_the_triggers_did():
    import threading
    hub = SimpleNamespace(_safe_park_now=threading.Event(), _scan_epoch=7)
    hub.host = hub  # fake: host fields live on the fake itself
    sp.request_park(hub, "why", WATCHDOG)
    assert hub.host._safe_park_reason == "why" and hub.host._safe_park_now.is_set()

    sp.request_park(hub, "why", DEFERRED)
    assert hub.host._hw_withdraw_after is True and hub.host._hw_withdraw_epoch == 7

    arc = SimpleNamespace(abort=lambda: "watchdog")
    assert sp.request_park(None, "why", ARC_ABORT, arc=arc) == "watchdog"

    with pytest.raises(ValueError):
        sp.request_park(hub, "why", "nonsense")


def test_signal_stop_narrow_modes_set_only_their_own_flag():
    import threading
    hub = SimpleNamespace(_hw_abort=threading.Event(),
                          _hw_litho_abort=threading.Event(),
                          _scan_repeat_stop=threading.Event(),
                          _twin_plugin=_twin_spy())
    hub.host = hub  # fake: host fields live on the fake itself
    sp._signal_stop(hub, sp.STOP_REPEAT_LOOP)
    assert hub.host._scan_repeat_stop.is_set() and hub.host._twin_plugin.aborted is False
    assert not hub.host._hw_abort.is_set()
    sp._signal_stop(hub, sp.STOP_LITHO_RUN)
    assert hub.host._twin_plugin.aborted is True and not hub.host._hw_litho_abort.is_set()
    assert not hub.host._hw_abort.is_set()
    hub.host._twin_plugin.aborted = False
    sp._signal_stop(hub, sp.STOP_SCAN_PASS)          # the scan pass: its flag and the twin
    assert hub.host._hw_abort.is_set() and hub.host._twin_plugin.aborted is True
    assert not hub.host._hw_litho_abort.is_set()


def test_retract_recipes_live_in_one_file():
    from afm_litho.hardware import retract
    assert {"twin_abort", "confirm_retract", "withdraw_rearm",
            "vendor_withdraw"} <= set(dir(retract))
