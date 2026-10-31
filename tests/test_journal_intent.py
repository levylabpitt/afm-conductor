"""6b1: an intent (a request that starts a run) returns only once its entry is fsynced, with a
1 s cap; stops never wait for the journal."""

from __future__ import annotations

import logging
import threading

from afm_litho import clock
from afm_litho.service import journal as journal_module
from afm_litho.service.journal import INTENT_WAIT_S, Journal
from afm_litho.service.requests import dispatch
from afm_litho.service.requests.table import Row


class SyncingDisk:
    """A sink that records what was written and what was fsynced, in order."""

    def __init__(self):
        self.events: list[str] = []

    def __call__(self, line: str) -> None:
        self.events.append("write:" + line)

    def sync(self) -> None:
        self.events.append("sync")


class StalledDisk:
    def __init__(self):
        self.release = threading.Event()

    def __call__(self, line: str) -> None:
        self.release.wait(10)


def test_arrival_numbers_rise_and_intent_returns_its_number(tmp_path):
    disk = SyncingDisk()
    j = Journal(tmp_path, sink=disk).start()
    j.note("door", "request")
    assert j.note_intent("door", "request", row="startScan") == 2
    j.stop()


def test_intent_returns_only_after_fsync(tmp_path):
    disk = SyncingDisk()
    j = Journal(tmp_path, sink=disk).start()
    with clock.using(clock.TESTING_TIME_SENSITIVE):
        j.note_intent("door", "request", row="startScan")
        seen = list(disk.events)  # the moment the caller got control back
    j.stop()
    assert seen[-1] == "sync"
    assert '"startScan"' in seen[-2]


def test_intent_gives_up_after_one_second_and_goes_on(tmp_path, caplog):
    disk = StalledDisk()
    j = Journal(tmp_path, sink=disk).start()
    with clock.using(clock.TESTING) as _, caplog.at_level(logging.WARNING):
        n = j.note_intent("door", "request", row="startScan")
        waits = [w for w in clock.waits() if w.kind == "event"]
    disk.release.set()
    j.stop()
    assert n is not None
    assert [w.seconds for w in waits] == [INTENT_WAIT_S] == [1.0]
    assert "not on disk" in caplog.text


def test_a_dropped_intent_does_not_wait(tmp_path):
    disk = StalledDisk()
    j = Journal(tmp_path, queue_size=1, sink=disk).start()
    j.note("door", "request")  # fills the one-slot queue behind the header
    j.note("door", "request")
    with clock.using(clock.TESTING):
        j.note_intent("door", "request")
        assert not [w for w in clock.waits() if w.kind == "event"]
    disk.release.set()
    j.stop()


def test_stop_row_never_waits_on_the_journal(tmp_path, monkeypatch):
    disk = StalledDisk()
    j = Journal(tmp_path, sink=disk).start()
    called = []
    monkeypatch.setattr(j, "note_intent", lambda *a, **k: called.append(1))
    row = Row(name="abort", kind="command", control="stop", intent=True,
              handler=lambda request: "done")
    with clock.using(clock.TESTING):
        assert dispatch.dispatch(row, "test", None) == "done"
        assert not [w for w in clock.waits() if w.kind == "event"]
    assert called == []
    disk.release.set()
    j.stop()
    assert journal_module.running().note_intent() is None


def _intent_row():
    return Row(name="startScan", kind="command", control="tip", intent=True,
               handler=lambda request: "ok")


def test_dispatch_async_accepted_intent_uses_note_intent(tmp_path):
    import asyncio

    j = Journal(tmp_path, sink=SyncingDisk()).start()
    seen = []
    real = j.note_intent
    j.note_intent = lambda *a, **k: seen.append(k.get("row")) or real(*a, **k)
    with clock.using(clock.TESTING_TIME_SENSITIVE):
        assert asyncio.run(dispatch.dispatch_async(_intent_row(), "t", {})) == "ok"
    j.stop()
    assert seen == ["startScan"]


def test_refusals_use_plain_note(tmp_path, monkeypatch):
    j = Journal(tmp_path, sink=SyncingDisk()).start()
    monkeypatch.setattr(j, "note_intent", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    dispatch._note_request(_intent_row(), "t", 1, {})  # a refusal passes no start flag
    j.stop()


def test_only_start_scan_and_start_write_are_intents():
    from afm_litho.service.requests.table import ROWS
    assert sorted(r.name for r in ROWS if r.intent) == ["startScan", "startWrite"]


def test_real_stop_rows_use_plain_note(tmp_path, monkeypatch):
    from afm_litho.service.requests.table import ROWS
    j = Journal(tmp_path, sink=SyncingDisk()).start()
    monkeypatch.setattr(j, "note_intent", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    for row in (r for r in ROWS if r.control == "stop"):
        dispatch._note_request(row, "t", 1, {}, start=True)
    j.stop()


def test_dead_writer_still_honours_the_cap(tmp_path):
    class Die(BaseException):
        pass

    def dying(line):
        raise Die

    j = Journal(tmp_path, sink=dying).start()
    j._writer.join(2)
    assert not j._writer.is_alive()
    with clock.using(clock.TESTING):
        assert j.note_intent("door", "request") is not None
        assert [w.seconds for w in clock.waits() if w.kind == "event"] == [1.0]
    j.stop()


def test_intent_through_the_real_file_sink(tmp_path):
    j = Journal(tmp_path).start()
    with clock.using(clock.TESTING_TIME_SENSITIVE):
        j.note_intent("door", "request", row="startWrite")
        text = j.path.read_text("utf-8")  # read before stop: it is already on disk
    j.stop()
    assert '"startWrite"' in text


def test_flex_start_never_waits_on_the_disk(tmp_path, monkeypatch):
    j = Journal(tmp_path, sink=StalledDisk()).start()
    monkeypatch.setattr(j, "note_intent", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert dispatch.dispatch(_intent_row(), "t", {}, wait_for_disk=False) == "ok"
    j.sink_release = None
    j.stop(0.2)
