"""A silent host reads `disconnected` after 2 s; a healthy idle one never does.

Time is the virtual clock (testing mode); nothing patches product timing.
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

from afm_litho import clock
from afm_litho.service import board, control, flex_dispatch
from afm_litho.service.control import control_tick
from afm_litho.service.hub import hub
from afm_litho.service.requests import table  # noqa: F401  (import order)
from afm_litho.service.requests import reads
from afm_litho.service.requests.context import using_hub


@pytest.fixture
def virtual():
    with clock.using(clock.TESTING):
        hub.host._last_heard = None
        yield
        hub.host._last_heard = None


def _answers() -> dict:
    with using_hub(hub):
        return {"state": reads._state(None), "status": reads._status(None),
                "web": asyncio.run(reads._web_state(None))}


def test_unheard_host_is_not_stale(virtual):
    clock.advance(60)
    assert board.state(hub) != "disconnected"


def test_silent_host_is_disconnected_after_two_seconds_and_recovers(virtual):
    hub.host.heard()
    clock.advance(1.9)
    assert _answers()["state"] != "disconnected"
    clock.advance(0.2)
    got = _answers()
    assert got["state"] == "disconnected"
    assert got["status"]["state"] == "disconnected"
    assert "silent" in got["status"]["disconnected_reason"]
    assert got["web"]["state"] == "disconnected"
    assert "disconnected_reason" in got["web"]
    asyncio.run(hub.host.emit({"type": "log"}))          # the host speaks again
    got = _answers()
    assert got["state"] != "disconnected"
    assert "disconnected_reason" not in got["status"]


def test_a_stale_parked_host_does_not_read_parked(virtual, monkeypatch):
    monkeypatch.setattr(hub.host, "_safe_park", {"active": True, "last_reason": "t"})
    monkeypatch.setattr(hub.host, "engaged", True)
    hub.host.heard()
    clock.advance(3)
    assert board.state(hub) == "disconnected"


def test_idle_healthy_service_never_disconnected(virtual):
    for _ in range(10):
        asyncio.run(control_tick(hub))          # the 1 s control tick
        clock.advance(1.0)
        assert board.state(hub) != "disconnected"


def test_flex_is_never_pushed(virtual):
    class Page:
        def __init__(self):
            self.sent = []

        async def send_text(self, data):
            self.sent.append(json.loads(data))

    page, before = Page(), set(hub.clients)
    hub.clients.add(page)
    try:
        asyncio.run(hub.broadcast({"type": "log", "msg": "x"}))
    finally:
        hub.clients.discard(page)
    assert page.sent == [{"type": "log", "msg": "x"}]       # a /ws page is pushed
    assert set(hub.clients) == before                        # FLEX never joined clients


def test_process_status_from_keeper_file(virtual, tmp_path, monkeypatch):
    monkeypatch.setenv("AFM_LITHO_RUNTIME", str(tmp_path))
    assert "processes" not in board.status(hub)
    (tmp_path / "keeper").mkdir()
    (tmp_path / "keeper" / "children.json").write_text(json.dumps({
        "keeper_pid": 11, "children": [
            {"name": "host", "pid": 22, "state": "running", "ready": {"pid": 33}},
            {"name": "base", "pid": 44, "state": "starting", "ready": None}]}))
    assert board.status(hub)["processes"] == {"keeper_pid": 11, "children": [
        {"name": "host", "pid": 33, "state": "running"},
        {"name": "base", "pid": 44, "state": "starting"}]}


def test_a_slow_control_tick_does_not_silence_the_host(monkeypatch):
    """A dead-man withdraw can hold control_tick for a minute; the heartbeat keeps going."""
    async def slow_deadman(hub_, now):
        await asyncio.sleep(2.6)      # real: the beat is real

    monkeypatch.setattr(control, "_deadman_tick", slow_deadman)

    async def go():
        hub.host._last_heard = None
        hub.host.start_beat()
        try:
            await control.control_tick(hub)           # awaits 2.6 s, over the 2 s limit
            return board.state(hub)
        finally:
            hub.host.stop_beat()

    assert asyncio.run(go()) != "disconnected"
    hub.host._last_heard = None


def test_frozen_loop_reads_disconnected_to_flex_on_its_own_thread(virtual):
    hub.host.stop_beat()
    hub.host.heard()                  # startup stamp; no loop ever ticks after it
    clock.advance(2.5)
    box = []
    t = threading.Thread(target=lambda: box.append(flex_dispatch._h_get_state(hub, None)))
    t.start()
    t.join()
    assert box == ["disconnected"]
