"""Slice 6r: raw captures are saved by one writer thread behind an unchanged `save_capture`."""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

from afm_litho import clock
from afm_litho.service import captures
from afm_litho.service.journal import Journal

CAP = {"Height": np.arange(40, dtype=float), "Output.Z": np.linspace(0, 1, 40)}


@pytest.fixture(autouse=True)
def fresh_writer(monkeypatch):
    monkeypatch.setattr(captures, "_writer", captures._RawWriter())


def _stuck_save(monkeypatch):
    """Make the writer thread's save wait until `release` is set; other threads save normally."""
    release = threading.Event()
    entered = threading.Event()
    real = captures.save_capture_now

    def stuck(*a, **kw):
        if threading.current_thread().name == "raw-writer":
            entered.set()
            release.wait(10)
        return real(*a, **kw)

    monkeypatch.setattr(captures, "save_capture_now", stuck)
    return release, entered


def test_npz_is_byte_identical_to_a_direct_save(tmp_path):
    ref = captures.save_capture(tmp_path / "a", "s1", CAP)
    captures.save_capture_now(tmp_path / "b", "s1", CAP)
    assert ref == "captures/s1.npz"
    assert (tmp_path / "a/captures/s1.npz").read_bytes() == \
        (tmp_path / "b/captures/s1.npz").read_bytes()


def test_empty_capture_writes_nothing(tmp_path):
    assert captures.save_capture(tmp_path, "x", {}) == ""


def test_full_queue_writes_a_gap_line_and_does_not_block(tmp_path, monkeypatch):
    jr = Journal(tmp_path / "j").start()
    release, entered = _stuck_save(monkeypatch)
    monkeypatch.setattr(captures, "QUEUE_SIZE", 1)
    monkeypatch.setattr(captures, "_writer", captures._RawWriter())
    first = threading.Thread(target=captures.save_capture, args=(tmp_path, "q0", CAP))
    first.start()
    assert entered.wait(5)                       # the writer is stuck inside the first save
    second = threading.Thread(target=captures.save_capture, args=(tmp_path, "q1", CAP))
    second.start()                               # fills the one queue slot
    deadline = time.monotonic() + 5
    while captures._writer._queue.qsize() < 1:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    t0 = time.monotonic()
    ref = captures.save_capture(tmp_path, "overflow", CAP)      # queue full: saved on this thread
    assert time.monotonic() - t0 < 2 and ref == "captures/overflow.npz"
    release.set()
    first.join(5)
    second.join(5)
    jr.stop()
    lines = [json.loads(x) for x in jr.path.read_text("utf-8").splitlines()]
    assert [x for x in lines if x["src"] == "capture" and x["what"] == "gap"]


def test_a_stop_during_a_slow_save_is_not_delayed(tmp_path, monkeypatch):
    release, entered = _stuck_save(monkeypatch)
    t = threading.Thread(target=captures.save_capture, args=(tmp_path, "slow", CAP))
    t.start()
    assert entered.wait(5)
    t0 = time.monotonic()                        # the stop path saves with save_capture_now
    captures.save_capture_now(tmp_path, "forensic", CAP)
    assert time.monotonic() - t0 < 2
    release.set()
    t.join(5)


def test_the_writer_survives_a_save_exception(tmp_path):
    with pytest.raises(ValueError):
        captures.save_capture(tmp_path, "bad", {"x": ["not", "numbers"]})
    assert captures.save_capture(tmp_path, "good", CAP) == "captures/good.npz"
    assert captures._writer._thread.is_alive()


def test_a_writer_that_never_answers_is_bypassed_and_then_skips_the_item(tmp_path, monkeypatch):
    jr = Journal(tmp_path / "j").start()
    release, entered = _stuck_save(monkeypatch)
    blocker = threading.Thread(target=captures.save_capture, args=(tmp_path, "first", CAP))
    blocker.start()
    assert entered.wait(5)                       # the writer is stuck inside the first save
    stuck_real = captures.save_capture_now
    writer_saved = []

    def spy(*a, **kw):
        if threading.current_thread().name == "raw-writer":
            writer_saved.append(a[1])
        return stuck_real(*a, **kw)

    monkeypatch.setattr(captures, "save_capture_now", spy)
    with clock.using(clock.TESTING):
        ref = captures.save_capture(tmp_path, "late", CAP)       # queued, waits, is bypassed
    assert ref == "captures/late.npz"
    release.set()
    blocker.join(5)
    deadline = time.monotonic() + 5
    while captures._writer._queue.qsize() or not captures._writer._queue.empty():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    time.sleep(0.1)
    assert "late" not in writer_saved            # cancelled: the writer never wrote it
    assert not list((tmp_path / "captures").glob(".*.tmp"))
    jr.stop()
    lines = [json.loads(x) for x in jr.path.read_text("utf-8").splitlines()]
    assert [x for x in lines if x["src"] == "capture" and x["what"] == "gap"]


def test_temp_names_differ_between_threads(tmp_path, monkeypatch):
    names = []
    real_replace = captures.os.replace
    monkeypatch.setattr(captures.os, "replace",
                        lambda a, b: (names.append(a.name), real_replace(a, b)))
    captures.save_capture_now(tmp_path, "t", CAP)
    t = threading.Thread(target=captures.save_capture_now, args=(tmp_path, "t", CAP))
    t.start()
    t.join(5)
    assert len(names) == 2 and names[0] != names[1]


def test_a_dead_writer_thread_saves_inline_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(captures._writer, "_ensure_started", lambda: None)
    assert captures.save_capture(tmp_path, "inline", CAP) == "captures/inline.npz"
