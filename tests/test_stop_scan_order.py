"""The operator Stop (`host.scan_cmds.stop_scan`): the stop flags and the stop count are set
before its first await, on every branch; the twin branch leaves the tip disengaged."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from afm_litho.service.host import safe_park, scan_cmds

BRANCHES = ("hardware_pass_running", "hardware_idle", "twin")


def _hub(branch: str):
    async def log(msg):
        hub.logs.append(msg)
        await asyncio.sleep(0)

    async def broadcast(frame):
        hub.frames.append(frame)
        await asyncio.sleep(0)

    hub = SimpleNamespace(
        scan_repeat={"enabled": True}, _scan_stop_seq=0, engaged=True, logs=[], frames=[],
        _scan_task=SimpleNamespace(done=lambda: branch != "hardware_pass_running"),
        inst=SimpleNamespace(withdraw=lambda: {"z_v": 0.0}),
        log=log, broadcast=broadcast)
    hub.host = hub  # fake: host fields live on the fake itself
    return hub


@pytest.fixture
def signals(monkeypatch):
    """Spy `_signal_stop` and `request_park`; each signal records the stop count it saw."""
    seen: list = []

    def signal(hub, which=safe_park.STOP_ALL):
        seen.append(("signal", which))

    def request_park(hub, why, how=safe_park.PARK_FULL, **kw):
        seen.append(("park", why, how))
        if how == safe_park.PARK_DEFERRED:
            return None

        async def parked():
            await asyncio.sleep(0)
            return {}
        return parked()

    monkeypatch.setattr(safe_park, "_signal_stop", signal)
    monkeypatch.setattr(safe_park, "request_park", request_park)
    return seen


@pytest.mark.parametrize("branch", BRANCHES)
def test_the_stop_flags_and_count_are_set_before_the_first_await(branch, signals,
                                                                  monkeypatch):
    bridge = None if branch == "twin" else object()
    monkeypatch.setattr(scan_cmds, "_real_bridge", lambda h: bridge)
    hub = _hub(branch)

    async def step_once():
        coro = scan_cmds.stop_scan(hub, {})
        try:
            coro.send(None)                  # runs up to the first suspension only
        except StopIteration:
            pytest.fail("stop_scan finished without suspending")
        state = (list(signals), hub.host._scan_stop_seq, dict(hub.host.scan_repeat))
        coro.close()
        return state

    seen, count, repeat = asyncio.run(step_once())
    assert ("signal", safe_park.STOP_REPEAT_LOOP) in seen
    assert ("signal", safe_park.STOP_SCAN_PASS) in seen
    assert count == 1
    assert repeat == {"enabled": False, "stopped_reason": "aborted"}


def test_the_twin_stop_leaves_the_tip_disengaged(signals, monkeypatch):
    monkeypatch.setattr(scan_cmds, "_real_bridge", lambda h: None)
    hub = _hub("twin")
    withdrawn = threading.Event()
    hub.host.inst.withdraw = lambda: withdrawn.set()
    reply = asyncio.run(scan_cmds.stop_scan(hub, {}))
    assert reply == {"ok": True, "withdraw": True}
    assert withdrawn.is_set()
    assert hub.host.engaged is False
    assert hub.frames == [{"type": "aborted", "withdraw": True}]
