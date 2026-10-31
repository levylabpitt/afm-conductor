"""The state stream (slice 2.9): one publisher, events journalled, delivery as before."""

from __future__ import annotations

import asyncio
import json
import threading

from afm_litho.service import journal as journal_module
from afm_litho.service.hub import Hub
from afm_litho.service.journal import Journal
from afm_litho.service.requests import stream


class Page:
    def __init__(self, fail=False):
        self.fail = fail
        self.sent = []

    async def send_text(self, text):
        if self.fail:
            raise RuntimeError("socket closed")
        self.sent.append(text)


def journal_lines(journal):
    journal.stop()
    return [json.loads(x) for x in journal.path.read_text("utf-8").splitlines()]


def test_broadcast_sends_the_same_frame_to_every_page_and_drops_dead_ones():
    hub = Hub()
    good, dead = Page(), Page(fail=True)
    hub.clients.update({good, dead})
    asyncio.run(hub.broadcast({"type": "design_changed"}))
    assert good.sent == [json.dumps({"type": "design_changed"})]
    assert hub.clients == {good}


def test_events_are_journalled_with_src_stream_and_telemetry_is_not(tmp_path):
    journal = Journal(tmp_path)
    journal.start()
    hub = Hub()
    hub.clients.add(Page())
    asyncio.run(hub.broadcast({"type": "fault", "fault": {"code": 7}}))
    asyncio.run(hub.broadcast({"type": "log", "t": 1.0, "msg": "hi"}))
    stream.telemetry_frame(hub)  # building a frame writes nothing
    lines = journal_lines(journal)
    events = [x for x in lines if x["src"] == "stream"]
    assert [x["what"] for x in events] == ["fault", "log"]
    assert events[0]["body"] == {"msg": {"type": "fault", "fault": {"code": 7}}}
    assert not any(x["what"] == "telemetry" for x in lines)


def test_high_rate_events_are_not_journalled(tmp_path):
    journal = Journal(tmp_path)
    journal.start()
    hub = Hub()
    hub.clients.add(Page())
    for kind in ("tip_pos", "litho_progress", "line", "scan_done"):
        asyncio.run(hub.broadcast({"type": kind}))
    lines = journal_lines(journal)
    assert [x["what"] for x in lines if x["src"] == "stream"] == ["line", "scan_done"]
    assert {"tip_pos", "litho_progress"} <= stream.NOT_JOURNALLED


def test_a_non_string_key_still_reaches_the_page():
    hub, page = Hub(), Page()
    hub.clients.add(page)
    asyncio.run(hub.broadcast({"type": "log", 1: "x"}))
    assert len(page.sent) == 1


def test_a_raising_note_does_not_stop_the_send(monkeypatch):
    class Broken:
        def note(self, *a, **k):
            raise RuntimeError("journal broke")

    monkeypatch.setattr(journal_module, "_running", Broken())
    hub, page = Hub(), Page()
    hub.clients.add(page)
    asyncio.run(hub.broadcast({"type": "log"}))
    assert page.sent == [json.dumps({"type": "log"})]


def test_a_stalled_journal_sink_does_not_stop_the_send(tmp_path):
    release = threading.Event()
    journal = Journal(tmp_path, sink=lambda line: release.wait(5))
    journal.start()
    try:
        hub, page = Hub(), Page()
        hub.clients.add(page)
        asyncio.run(hub.broadcast({"type": "log"}))
        assert len(page.sent) == 1
    finally:
        release.set()
        journal.stop()


def test_telemetry_frame_keeps_its_type_and_scanning_fields():
    frame = stream.telemetry_frame(Hub())
    assert frame["type"] == "telemetry"
    assert frame["scanning"] is False
    assert stream.TELEMETRY_PERIOD_S == 0.25


def test_subscribe_and_unsubscribe():
    hub, page = Hub(), Page()
    stream.subscribe(hub, page)
    assert page in hub.clients
    stream.unsubscribe(hub, page)
    stream.unsubscribe(hub, page)
    assert page not in hub.clients
