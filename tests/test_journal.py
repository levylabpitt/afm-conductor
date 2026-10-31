"""Journal file format v1 (slice J1): round trip, gap, never blocking, masking, summaries."""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

from afm_litho.goldens.expander import summarise
from afm_litho.service import journal as journal_module
from afm_litho.service.journal import FORMAT_VERSION, Journal


def read_lines(journal: Journal) -> list[dict]:
    journal.stop()
    return [json.loads(x) for x in journal.path.read_text("utf-8").splitlines()]


class StalledDisk:
    """A sink that holds every write until released, standing in for a hung disk."""

    def __init__(self):
        self.entered = threading.Event()  # set when the writer is stuck inside the sink
        self.release = threading.Event()
        self.lines: list[str] = []

    def __call__(self, line: str) -> None:
        self.entered.set()
        self.release.wait(10)
        self.lines.append(line)


def test_round_trip(tmp_path):
    j = Journal(tmp_path).start()
    j.note("door", "request", actor="operator", trace=7, run="r1", chunk=3, row="engage", z=1.5)
    j.note("wire", "send", text="AFM_SafePark()")
    header, first, second, closing = read_lines(j)
    assert header["what"] == "header" and header["body"]["journal"] == FORMAT_VERSION
    assert {"started", "platform", "commit"} <= header["body"].keys()
    assert (first["n"], second["n"]) == (1, 2)
    assert first["src"] == "door" and first["what"] == "request"
    assert (first["actor"], first["trace"], first["run"], first["chunk"]) == ("operator", 7, "r1", 3)
    assert first["body"] == {"row": "engage", "z": 1.5}
    assert "actor" not in second and second["body"] == {"text": "AFM_SafePark()"}
    assert {"t", "mono"} <= first.keys()
    assert closing["what"] == "closed"
    assert j.path.parent == tmp_path and j.path.suffix == ".jsonl"


def test_two_runs_make_two_files(tmp_path):
    a, b = Journal(tmp_path).start(), Journal(tmp_path).start()
    a.stop()
    b.stop()
    assert a.path != b.path and len(list(tmp_path.glob("*.jsonl"))) == 2


def test_note_before_start_and_after_stop_is_a_no_op(tmp_path):
    j = Journal(tmp_path)
    j.note("door", "early")
    assert j.path is None and not list(tmp_path.iterdir())
    j.start()
    j.note("door", "during")
    j.stop()
    j.note("door", "late")
    whats = [e["what"] for e in read_lines(j)]
    assert whats[1:] == ["during", "closed"]


def test_stop_drains_and_writes_closing_line(tmp_path):
    j = Journal(tmp_path).start()
    for i in range(500):
        j.note("wire", "send", i=i)
    lines = read_lines(j)
    assert lines[0]["body"]["journal"] == FORMAT_VERSION
    assert [e["body"]["i"] for e in lines[1:-1]] == list(range(500))
    assert lines[-1]["what"] == "closed"


def test_stop_is_bounded_when_the_writer_is_stalled(tmp_path):
    disk = StalledDisk()
    j = Journal(tmp_path, sink=disk).start()
    j.note("door", "request")
    began = time.perf_counter()
    j.stop(timeout=0.3)
    assert time.perf_counter() - began < 1.5
    disk.release.set()


def test_running_accessor(tmp_path):
    assert journal_module.running().note("door", "x") is None  # none started: harmless
    j = Journal(tmp_path).start()
    assert journal_module.running() is j
    journal_module.running().note("door", "via accessor")
    j.stop()
    assert journal_module.running() is not j
    assert any(e["what"] == "via accessor" for e in read_lines(j))


def test_full_queue_gives_a_gap_entry(tmp_path):
    disk = StalledDisk()
    j = Journal(tmp_path, queue_size=5, sink=disk).start()
    assert disk.entered.wait(5)  # the writer has the header and is stuck in the sink
    for i in range(20):
        j.note("door", "request", i=i)
    disk.release.set()
    j.stop()
    entries = [json.loads(x) for x in disk.lines]
    gaps = [e for e in entries if e["what"] == "gap"]
    kept = [e for e in entries if e.get("src") == "door"]
    assert len(gaps) >= 1
    assert sum(g["body"]["dropped"] for g in gaps) == 20 - len(kept) > 0


def test_stalled_writer_never_blocks_note(tmp_path):
    disk = StalledDisk()
    j = Journal(tmp_path, queue_size=5, sink=disk).start()
    assert disk.entered.wait(5)
    start = time.perf_counter()
    for i in range(1000):  # far past the queue size, writer held the whole time
        j.note("wire", "send", i=i)
    elapsed = time.perf_counter() - start
    disk.release.set()
    j.stop()
    assert elapsed < 0.5, f"1000 notes took {elapsed:.3f}s with the writer held"


def test_note_never_raises(tmp_path):
    j = Journal(tmp_path).start()
    j.note("door", "request", weird=object(), bad={1: float("nan")})
    j.note("door", "after-bad")
    lines = read_lines(j)
    assert [e["what"] for e in lines if e.get("src") == "door"] == ["request", "after-bad"]


def test_tokens_are_masked(tmp_path):
    j = Journal(tmp_path).start()
    j.note("door", "hello", actor="operator", token="abc123", extra={"Password": "pw", "ok": 1})
    entry = read_lines(j)[1]
    assert entry["body"]["token"] == "***"
    assert entry["body"]["extra"] == {"Password": "***", "ok": 1}
    assert entry["actor"] == "operator"
    assert "abc123" not in j.path.read_text("utf-8")


def test_arrays_are_summarised(tmp_path):
    j = Journal(tmp_path).start()
    data = np.arange(50_000, dtype=float)
    j.note("wire", "outwave", wave=data, short=[1, 2, 3], long=list(range(100)))
    entry = read_lines(j)[1]
    wave = entry["body"]["wave"]
    assert (wave["array"], wave["first"], wave["last"]) == (50_000, 0.0, 49_999.0)
    assert len(wave["sha256"]) == 16
    assert entry["body"]["short"] == [1, 2, 3]
    assert entry["body"]["long"]["array"] == 100
    assert len(j.path.read_text("utf-8")) < 2_000


def test_body_keys_named_src_what_self_do_not_raise(tmp_path):
    j = Journal(tmp_path).start()
    j.note("door", "request", src="x", what="y", self="z")
    entry = read_lines(j)[1]
    assert (entry["src"], entry["what"]) == ("door", "request")
    assert entry["body"] == {"src": "x", "what": "y", "self": "z"}


def test_secrets_are_masked_everywhere(tmp_path):
    j = Journal(tmp_path).start()
    many = [{"token": "tok-long-list"}] * 20
    j.note("door", "r", actor="operator", trace={"authorization": "Bearer aa"}, run=np.int64(4),
           chunk=np.int32(2), headers={"X-Api-Key": "k1", "my_cookie_jar": "c", "ok": 1},
           many=many)
    entry = read_lines(j)[1]
    text = j.path.read_text("utf-8")
    assert entry["trace"] == {"authorization": "***"}
    assert (entry["run"], entry["chunk"]) == (4, 2)
    assert entry["body"]["headers"] == {"X-Api-Key": "***", "my_cookie_jar": "***", "ok": 1}
    assert entry["body"]["many"]["first"] == {"token": "***"}
    for leaked in ("tok-long-list", "Bearer aa", "k1"):
        assert leaked not in text


def test_failed_writes_are_counted_and_logged(tmp_path, caplog):
    calls = []

    def every_third_fails(line):
        calls.append(line)
        if len(calls) % 3 == 0:
            raise OSError("disk")

    j = Journal(tmp_path, sink=every_third_fails).start()
    for i in range(10):
        j.note("wire", "send", i=i)
    with caplog.at_level("WARNING"):
        j.stop()
    assert j.write_failures >= 3
    closed = json.loads(calls[-1])
    assert closed["what"] == "closed" and closed["body"]["write_failures"] >= 3
    assert sum("journal write failed" in r.message for r in caplog.records) == 1


def test_a_sink_that_always_fails_still_stops(tmp_path):
    def broken(line):
        raise OSError("disk")

    j = Journal(tmp_path, sink=broken).start()
    j.note("wire", "send")
    j.stop()
    assert j.write_failures >= 2


def test_gap_sits_where_the_drops_were(tmp_path):
    disk = StalledDisk()
    j = Journal(tmp_path, queue_size=5, sink=disk).start()
    assert disk.entered.wait(5)
    for i in range(20):
        j.note("door", "request", i=i)
    disk.release.set()
    j.stop()
    entries = [json.loads(x) for x in disk.lines]
    arrivals = [e for e in entries if "n" in e]
    assert [e["n"] for e in arrivals] == sorted(e["n"] for e in arrivals)
    gap_at = next(k for k, e in enumerate(entries) if e["what"] == "gap")
    gap = entries[gap_at]["body"]
    assert entries[gap_at - 1]["n"] == gap["first_n"] - 1  # last kept line before the drops
    after = entries[gap_at + 1]
    assert after["what"] == "closed" or after["n"] > gap["last_n"]
    assert gap["dropped"] == gap["last_n"] - gap["first_n"] + 1


def test_array_summary_equals_the_golden_tap(tmp_path):
    data = np.sin(np.arange(1000.0))
    j = Journal(tmp_path).start()
    j.note("wire", "outwave", wave=data, ints=list(range(100)))
    body = read_lines(j)[1]["body"]
    tap = summarise("w", data)
    assert (body["wave"]["array"], body["wave"]["first"], body["wave"]["last"],
            body["wave"]["sha256"]) == (tap.n, tap.first, tap.last, tap.sha)
    assert body["ints"]["sha256"] == summarise("w", np.arange(100.0)).sha


def test_numpy_scalars_and_nonfinite_numbers(tmp_path):
    j = Journal(tmp_path).start()
    j.note("door", "r", a=np.float32(1.5), b=np.int64(3), c=float("nan"), d=float("inf"),
           e=[float("-inf"), 1], f=b"ab")
    body = read_lines(j)[1]["body"]
    assert body == {"a": 1.5, "b": 3, "c": None, "d": None, "e": [None, 1], "f": {"bytes": "6162"}}


def test_long_strings_and_lists_keep_note_cheap(tmp_path):
    j = Journal(tmp_path).start()
    big_text = "x" * 50_000_000
    big_list = list(range(1_000_000))
    began = time.perf_counter()
    j.note("wire", "send", text=big_text, values=big_list, blob=b"y" * 50_000_000)
    elapsed = time.perf_counter() - began
    body = read_lines(j)[1]["body"]
    assert elapsed < 1.0, f"note() took {elapsed:.3f}s"
    assert body["text"]["length"] == 50_000_000 and len(body["text"]["text"]) == 4096
    assert body["values"]["array"] == 1_000_000
    assert body["blob"]["bytes"] == 50_000_000


def test_start_twice_raises(tmp_path):
    j = Journal(tmp_path).start()
    j.stop()
    with pytest.raises(RuntimeError):
        j.start()


def test_every_line_has_the_common_fields(tmp_path):
    j = Journal(tmp_path, queue_size=2, sink=None).start()
    for line in read_lines(j):
        assert {"src", "what", "t", "mono"} <= line.keys()
