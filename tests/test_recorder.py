"""6b2: the journal writer runs in the recorder process; the service sends it lines over loopback
and spills to a file when it is down. Real sockets and real time (these tests check timing),
the recorder in-process unless a test needs a real process to kill."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from afm_litho import clock
from afm_litho.keeper.beatfile import read_lines
from afm_litho.keeper.cli import child_specs
from afm_litho.keeper.protocol import Ready, parse_line
from afm_litho.recorder_main import Recorder
from afm_litho.service.journal import Journal
from afm_litho.service.recorder_link import RecorderSink
from afm_litho.service.requests import dispatch
from afm_litho.service.requests.table import Row

DOOR_CAP_S = 1.1  # the intent wait is 1.0 s; nothing a door does may cost more than this


@pytest.fixture(autouse=True)
def real_time():
    with clock.using(clock.TESTING_TIME_SENSITIVE):
        yield


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def journal_to(folder: Path, port: int) -> Journal:
    return Journal(folder, sink_for=lambda path, gap: RecorderSink(port, path, gap)).start()


def lines_of(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text("utf-8").splitlines()]


def wait_until(cond, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def restart_recorder(port: int) -> Recorder:
    for _ in range(50):
        try:
            return Recorder(port)
        except OSError:
            time.sleep(0.1)
    raise AssertionError("could not rebind the recorder port")


def masked(entries: list[dict]) -> list[dict]:
    """The lines with the clocks and the run-specific header fields removed."""
    out = []
    for e in entries:
        e = {k: v for k, v in e.items() if k not in ("t", "mono")}
        if e["what"] == "header":
            e["body"] = {"journal": e["body"]["journal"]}
        out.append(e)
    return out


def some_notes(j: Journal) -> None:
    j.note("door", "request", actor="flex", row="getState")
    j.note("wire", "send", cmd="td_Foo", values=list(range(40)))
    j.note_intent("door", "request", row="startScan", trace=7)
    j.note("door", "reply", row="startScan", ok=True)


def test_a_healthy_run_writes_the_same_journal_as_the_in_process_writer(tmp_path):
    rec = Recorder(0)
    try:
        local = Journal(tmp_path / "local").start()
        some_notes(local)
        local.stop()
        remote = journal_to(tmp_path / "remote", rec.port)
        some_notes(remote)
        remote.stop()
    finally:
        rec.stop()
    assert masked(lines_of(local.path)) == masked(lines_of(remote.path))
    assert local.path.read_bytes().count(b"\n") == remote.path.read_bytes().count(b"\n")
    assert not list((tmp_path / "remote").glob("*.spill"))


def test_an_intent_returns_only_after_the_recorders_fsync_acknowledgement(tmp_path):
    rec = Recorder(0)
    j = journal_to(tmp_path, rec.port)
    try:
        j.note_intent("door", "request", row="startWrite")
        assert '"startWrite"' in j.path.read_text("utf-8")  # on disk the moment it returned
    finally:
        j.stop()
        rec.stop()


def test_an_intent_with_the_recorder_down_returns_within_the_cap_and_is_on_disk(tmp_path):
    j = journal_to(tmp_path, free_port())  # nothing is listening
    t0 = time.monotonic()
    n = j.note_intent("door", "request", row="startScan")
    assert time.monotonic() - t0 < DOOR_CAP_S
    assert n is not None
    (spill,) = tmp_path.glob("*.spill")
    assert '"startScan"' in spill.read_text("utf-8")  # fsynced spill: the intent counts as written
    j.stop()


def test_a_recorder_that_accepts_but_never_answers_costs_a_start_at_most_the_cap(tmp_path):
    with socket.socket() as dead_air:
        dead_air.bind(("127.0.0.1", 0))
        dead_air.listen(5)
        j = journal_to(tmp_path, dead_air.getsockname()[1])
        t0 = time.monotonic()
        j.note_intent("door", "request", row="startScan")
        assert time.monotonic() - t0 < DOOR_CAP_S
        j.note("door", "reply")  # does not block either
        j.stop()
    assert any('"startScan"' in p.read_text("utf-8") for p in tmp_path.glob("*.spill"))


def test_the_recorder_starting_after_the_service_gets_the_early_entries_as_late(tmp_path):
    port = free_port()
    j = journal_to(tmp_path, port)  # the service is up; the recorder is not
    j.note("door", "request", row="getState")
    j.note_intent("door", "request", row="startScan")
    rec = Recorder(port)
    try:
        wait_until(lambda: j.path.exists() and '"startScan"' in j.path.read_text("utf-8"))
        j.note("door", "reply", row="startScan")
        j.stop()
    finally:
        rec.stop()
    entries = lines_of(j.path)
    assert entries[0]["src"] == "recorder" and entries[0]["what"] == "gap"
    assert [e["what"] for e in entries if e.get("late")] == ["header", "request", "request"]
    assert [e["n"] for e in entries if "n" in e] == [1, 2, 3]
    assert entries[-1]["what"] == "closed"
    assert not list(tmp_path.glob("*.spill"))


def test_killing_the_recorder_leaves_a_gap_and_late_entries_in_order(tmp_path):
    rec = Recorder(0)
    port = rec.port
    j = journal_to(tmp_path, port)
    for i in range(5):
        j.note("wire", "send", i=i)
    j.note_intent("door", "request", row="startScan")
    rec.stop()  # killed mid-run
    slowest = 0.0
    for i in range(30):
        t0 = time.monotonic()
        j.note("wire", "send", i=100 + i)
        slowest = max(slowest, time.monotonic() - t0)
        time.sleep(0.01)
    t0 = time.monotonic()
    j.note_intent("door", "request", row="startScan")
    assert time.monotonic() - t0 < DOOR_CAP_S
    assert slowest < 0.1
    rec2 = restart_recorder(port)
    try:
        wait_until(lambda: any(e["what"] == "gap" and e["src"] == "recorder"
                               for e in lines_of(j.path)))
        j.note("wire", "send", i=999)
        j.stop()
    finally:
        rec2.stop()
    entries = lines_of(j.path)
    numbers = [e["n"] for e in entries if "n" in e]
    assert sorted(set(numbers)) == list(range(1, max(numbers) + 1))  # none lost
    assert numbers == sorted(numbers)  # file order is arrival order
    gap = next(e for e in entries if e["what"] == "gap" and e["src"] == "recorder")
    assert gap["body"]["entries"] >= 30
    late = [e for e in entries if e.get("late")]
    assert late and all(e["n"] > 6 or e["what"] == "header" for e in late)
    assert entries.index(gap) < entries.index(late[0])
    assert not list(tmp_path.glob("*.spill"))


def test_a_recorder_process_killed_and_restarted(tmp_path):
    port = free_port()
    env = {**os.environ, "AFM_LITHO_RECORDER_PORT": str(port), "PYTHONIOENCODING": "utf-8",
           "AFM_LITHO_RUNTIME": str(tmp_path),  # the recorder only opens files in runtime/journal
           "AFM_LITHO_BEAT_FILE": str(tmp_path / "beat-recorder.txt")}
    beats = tmp_path / "beat-recorder.txt"
    folder = tmp_path / "journal"

    def start() -> subprocess.Popen:
        beats.unlink(missing_ok=True)
        proc = subprocess.Popen([sys.executable, "-m", "afm_litho.recorder_main"], env=env)
        wait_until(lambda: (read_lines(beats) or [""])[0] != "", 30)
        msg = parse_line(read_lines(beats)[0])
        assert isinstance(msg, Ready)
        proc.recorder_pid = msg.pid  # under a venv on Windows Popen's pid is the launcher stub
        return proc

    def kill(proc: subprocess.Popen) -> None:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.recorder_pid)],
                           capture_output=True, check=False)
        else:
            os.kill(proc.recorder_pid, 9)
        proc.wait()

    proc = start()
    j = journal_to(folder, port)
    try:
        j.note_intent("door", "request", row="startScan")
        kill(proc)
        j.note("wire", "send", during_outage=True)
        t0 = time.monotonic()
        j.note_intent("door", "request", row="startScan")
        assert time.monotonic() - t0 < DOOR_CAP_S
        proc = start()
        wait_until(lambda: any(e.get("what") == "gap" for e in lines_of(j.path)), 15)
        j.stop()
    finally:
        kill(proc)
    entries = lines_of(j.path)
    assert [e["what"] for e in entries].count("gap") == 1
    assert [e["body"].get("during_outage") for e in entries if e.get("late")
            and e["what"] == "send"] == [True]


def test_stop_never_touches_the_recorder(tmp_path):
    j = journal_to(tmp_path, free_port())
    row = Row(name="abort", kind="command", control="stop", intent=True,
              handler=lambda request: "done")
    clock.use(clock.TESTING)
    try:
        t0 = time.monotonic()
        assert dispatch.dispatch(row, "test", None) == "done"
        assert time.monotonic() - t0 < 0.5
        assert not [w for w in clock.waits() if w.kind == "event"]
    finally:
        clock.use(clock.TESTING_TIME_SENSITIVE)
        j.stop()


def test_a_dead_recorder_does_not_hold_back_the_service_closing(tmp_path):
    j = journal_to(tmp_path, free_port())
    j.note("door", "request")
    t0 = time.monotonic()
    j.stop()
    assert time.monotonic() - t0 < 2.0
    assert lines_of(next(tmp_path.glob("*.spill")))[0]["what"] == "header"


def test_the_keeper_starts_the_recorder_first_and_does_not_wait_for_it():
    recorder, service = child_specs(7461, "127.0.0.1")
    assert (recorder.name, service.name) == ("recorder", "service")
    assert recorder.essential is False
    port = recorder.env["AFM_LITHO_RECORDER_PORT"]
    assert service.env["AFM_LITHO_RECORDER_PORT"] == port
    assert recorder.argv[-1] == "afm_litho.recorder_main"


def test_the_recorder_listens_on_loopback_only():
    rec = Recorder(0)
    try:
        assert rec._listener.getsockname()[0] == "127.0.0.1"
    finally:
        rec.stop()


def test_the_recorder_refuses_a_file_that_is_not_a_journal(tmp_path):
    rec = Recorder(0)
    try:
        with socket.create_connection(("127.0.0.1", rec.port), timeout=2) as s:
            s.sendall(json.dumps({"k": 1, "op": "open",
                                  "path": str(tmp_path / "x.txt")}).encode() + b"\n")
            assert s.recv(100) == b""  # connection closed, no acknowledgement
        assert not (tmp_path / "x.txt").exists()
    finally:
        rec.stop()


def test_spill_files_left_by_an_earlier_run_are_replayed_into_their_run_file(tmp_path):
    first = journal_to(tmp_path, free_port())  # no recorder at all
    for i in range(5):
        first.note("wire", "send", i=i)
    first.note_intent("door", "request", row="startScan")
    first.stop()
    assert len(list(tmp_path.glob("*.spill"))) == 1
    old_run = first.path
    assert old_run.read_text("utf-8") == ""

    rec = Recorder(0, tmp_path)
    second = journal_to(tmp_path, rec.port)  # the service restarted, and the recorder is up
    try:
        wait_until(lambda: not list(tmp_path.glob("*.spill")))
        second.stop()
    finally:
        rec.stop()
    entries = lines_of(old_run)
    assert entries[0]["what"] == "gap" and entries[0]["src"] == "recorder"
    late = [e for e in entries if e.get("late")]
    assert [e["what"] for e in late][:1] == ["header"]
    assert [e["n"] for e in late if "n" in e] == [1, 2, 3, 4, 5, 6]
    assert not any("late" in e for e in lines_of(second.path))


def test_a_spill_file_with_no_run_file_is_logged_and_kept(tmp_path, caplog):
    stray = tmp_path / "2020-01-01T00-00-00.jsonl.99999.spill"
    stray.write_text('{"n": 1}\n', "utf-8")
    j = journal_to(tmp_path, free_port())
    j.stop()
    assert "no run file" in caplog.text
    assert stray.exists()


def test_a_hung_recorder_never_makes_an_intent_non_durable(tmp_path):
    with socket.socket() as dead_air:
        dead_air.bind(("127.0.0.1", 0))
        dead_air.listen(5)
        j = journal_to(tmp_path, dead_air.getsockname()[1])
        for i in range(3000):
            j.note("wire", "send", i=i)
        for k in range(6):  # spans several reconnect attempts and back-offs
            t0 = time.monotonic()
            j.note_intent("door", "request", row=f"startScan{k}")
            assert time.monotonic() - t0 < DOOR_CAP_S
            (spill,) = tmp_path.glob("*.spill")
            assert f'"startScan{k}"' in spill.read_text("utf-8")
            time.sleep(0.6)
        j.stop()


def test_one_gap_line_per_outage_and_no_repeats_after_a_replay(tmp_path):
    rec = Recorder(0)
    port = rec.port
    j = journal_to(tmp_path, port)
    rec.stop()
    time.sleep(0.2)
    for i in range(50):
        j.note("wire", "send", i=i)
    rec2 = restart_recorder(port)
    try:
        wait_until(lambda: any(e["what"] == "gap" for e in lines_of(j.path)))
        j.stop()
    finally:
        rec2.stop()
    entries = lines_of(j.path)
    assert [e["what"] for e in entries].count("gap") == 1
    numbers = [e["n"] for e in entries if "n" in e]
    assert sorted(numbers) == sorted(set(numbers))


def test_the_recorder_only_opens_files_inside_the_journal_folder(tmp_path):
    rec = Recorder(0, tmp_path / "journal")
    try:
        with socket.create_connection(("127.0.0.1", rec.port), timeout=2) as s:
            s.sendall(json.dumps({"k": 1, "op": "open",
                                  "path": str(tmp_path / "elsewhere.jsonl")}).encode() + b"\n")
            assert s.recv(100) == b""
        assert not (tmp_path / "elsewhere.jsonl").exists()
    finally:
        rec.stop()


def test_the_reader_puts_late_lines_in_arrival_order_and_drops_repeats(tmp_path):
    from afm_litho.service import journal_read
    rows = [
        {"src": "journal", "what": "header"},
        {"n": 1, "src": "a", "what": "x"},
        {"n": 3, "src": "a", "what": "x"},
        {"src": "recorder", "what": "gap"},
        {"n": 2, "src": "a", "what": "x", "late": True},
        {"n": 3, "src": "a", "what": "x", "late": True},
        {"src": "journal", "what": "closed"},
    ]
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    out = journal_read.load(path)
    assert [e.get("n") for e in out] == [None, 1, 2, None, 3, None]
    assert "late" not in out[4]
    assert journal_read.is_closed(out)
