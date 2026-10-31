"""The park and safety-stop steps must keep going when a side step raises.

Nothing may stop the AFM from reaching a safe state, so these swallowed errors
are deliberate (DECISIONS: safe_park). Narrowing any of these excepts to a
specific type must fail one of these tests.
"""

from __future__ import annotations

import asyncio
import importlib
import threading
from types import SimpleNamespace

import pytest

sp = importlib.import_module("afm_litho.service.host.safe_park")
cont = importlib.import_module("afm_litho.service.host.continuous")


class _BadDict(dict):
    def __setitem__(self, k, v):
        raise RuntimeError("boom")


class _BadEvent:
    def set(self):
        raise RuntimeError("boom")


class _Store:
    def record(self, *a, **k):
        pass


class _Hub:
    def __init__(self, bad_repeat=False, bad_event=False, bad_broadcast=False):
        self.scan_repeat = _BadDict() if bad_repeat else {}
        self._scan_repeat_stop = _BadEvent() if bad_event else threading.Event()
        self.store = _Store()
        self.bad_broadcast = bad_broadcast
        self.engaged = True

    async def broadcast(self, m):
        if self.bad_broadcast:
            raise RuntimeError("ws dead")

    async def log(self, m):
        pass

    @property
    def host(self):                # fake: host fields live on the fake itself
        return self


CASES = [
    dict(bad_repeat=True),
    dict(bad_event=True),
    dict(bad_broadcast=True),
    dict(bad_repeat=True, bad_event=True, bad_broadcast=True),
]


@pytest.mark.parametrize("kw", CASES)
def test_signal_stop_still_aborts_the_run(kw, monkeypatch):
    called = []
    monkeypatch.setattr(sp, "signal_run_abort", lambda hub: called.append(1))
    hub = _Hub(**kw)
    sp._signal_stop(hub)
    assert called == [1]
    if not kw.get("bad_event"):
        assert hub.host._scan_repeat_stop.is_set()


@pytest.mark.parametrize("kw", CASES)
def test_finish_still_returns_the_result(kw):
    hub = _Hub(**kw)
    result = {"backend": "twin", "transport": "x", "verified": True, "z_v": 0.0,
              "critical": False, "retried": False, "at": 0.0}
    out = asyncio.run(sp._finish(hub, "t", result))
    assert out is result


def test_rail_trip_continues_when_abort_scan_raises(monkeypatch):
    class Inst:
        def abort_scan(self):
            raise RuntimeError("boom")

    logs = []

    async def log(m):
        logs.append(m)

    async def noop(*a, **k):
        pass

    hub = SimpleNamespace(
        scan_repeat={}, _scan_repeat_stop=threading.Event(), inst=Inst(),
        store=_Store(), log=log)
    hub.host = hub  # fake: host fields live on the fake itself
    monkeypatch.setattr(cont, "hub", hub)
    monkeypatch.setattr(cont, "_real_bridge", lambda h: None)
    monkeypatch.setattr(cont, "_broadcast_rail", noop)
    guard = SimpleNamespace(
        tripped=None, consec=3, frame_frac=0.5,
        limits=SimpleNamespace(consec_lines=3, frame_frac=0.5, as_dict=dict))
    asyncio.run(cont._rail_trip(guard, {"guard": "B"}))
    assert hub.host._scan_repeat_stop.is_set()
    assert guard.tripped is not None
    assert any(m.startswith("SAFETY") for m in logs)


def test_stop_supervisor_completes_when_the_repeat_task_died(monkeypatch):
    async def run():
        async def dies():
            raise RuntimeError("boom")

        task = asyncio.ensure_future(dies())
        hub = SimpleNamespace(
            scan_repeat={}, _scan_repeat_stop=threading.Event(), _scan_repeat_task=task)
        hub.host = hub  # fake: host fields live on the fake itself
        monkeypatch.setattr(cont, "hub", hub)
        await cont._stop_supervisor("t")
        return hub

    hub = asyncio.run(run())
    assert hub.host._scan_repeat_stop.is_set()
    assert hub.host._scan_repeat_task is None
