"""Slice J3: the three journal questions, answered from the journal file alone.

Each test drives a real scenario through the doors on the twin (the FLEX door in process, the
Igor stub behind it), then reads the file with `journal_read` and asserts what it says. No test
asks the hub: if the file cannot answer, the test fails.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
from golden.harness import FlexDoor, HttpDoor
from golden.scenarios import WIRE, WIRE_LONG, _acquire, _release, _start_write
from journal_rig import chunk_captured, command_sent, journalled_rig, wait_in_file

from afm_litho.service import journal_read as jr
from afm_litho.service.requests.table import ROWS

SCRIPT = "golden"                      # the client name `_acquire` registers


@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


def _stop_and_settle(rig, tok) -> None:
    rig.door.call("abortWrite", {})
    rig.door.wait_state("idle", "engaged", "parked")
    _release(rig.door, tok)


# --- 1. what was in flight when the run stopped -----------------------------


@pytest.fixture(scope="module")
def seam_stop(hub, tmp_path_factory):
    """One two-chunk write stopped at its seam (chunk 0 delivered, chunk 1 not), read from
    the file twice: while the service ran, and after the stop. The slow scenario runs once
    for questions 1 and 2."""
    with journalled_rig(hub, tmp_path_factory.mktemp("seam")) as (rig, journal):
        tok = _acquire(rig.door)
        rig.inst.hold_chain_after_chunk(0)
        run = _start_write(rig.door, tok, [WIRE_LONG])
        rig.inst.wait_held()
        live = wait_in_file(journal, chunk_captured("wire_long", 1))
        _stop_and_settle(rig, tok)
    return SimpleNamespace(run=run, live=live, stopped=jr.load(journal.path))


def test_q1_what_was_in_flight_when_the_run_stopped(seam_stop):
    # The service is still running (no closing line): the answer is "at the end of the file".
    assert not jr.is_closed(seam_stop.live)
    now = jr.in_flight(seam_stop.live)
    assert not now.stopped
    # The same question, asked at the stop line of the finished file.
    then = jr.in_flight(seam_stop.stopped)
    assert jr.is_closed(seam_stop.stopped) and then.stopped
    assert then.at_n > now.at_n

    for answer in (now, then):
        assert answer.run_id == seam_stop.run["run_id"]
        assert (answer.request.row, answer.request.actor) == ("startWrite", SCRIPT)
        assert answer.object == "wire_long"
        assert len(answer.pending) == 1 and answer.pending[0].startswith("LITHOCHAIN ")
        assert answer.last_commands[-1].startswith("LITHOCHAIN ")
    # The commands sent just before the stop are the same pass's waves and its launch.
    assert [c.split()[0] for c in then.last_commands][-3:] == ["WAVEAPP", "WAVE", "LITHOCHAIN"]


def test_q1_a_run_that_ended_has_nothing_in_flight(hub, tmp_path):
    with journalled_rig(hub, tmp_path) as (rig, journal):
        tok = _acquire(rig.door)
        _start_write(rig.door, tok, [WIRE])
        rig.door.wait_state("idle", "engaged", "parked")
        _release(rig.door, tok)
    done = jr.in_flight(jr.load(journal.path))
    assert done.run_id is None and done.object is None and done.pending == []
    assert not done.stopped


# --- 2. did Stop land before or after chunk n -------------------------------


def test_q2_stop_after_chunk_0_before_chunk_1(seam_stop):
    entries = seam_stop.stopped
    landing = jr.stop_landing(entries)
    assert (landing.run_id, landing.object) == (seam_stop.run["run_id"], "wire_long")
    assert landing.chunks_before == 1
    assert jr.stop_vs_chunk(entries, 0) == "after"
    assert jr.stop_vs_chunk(entries, 1) == "before"
    # The stop line is the first thing the host sent to retract, after the last chunk it read.
    marker = jr.stop_marker(entries)
    assert (marker["what"], jr.text_of(marker)) == ("com", "AFM_SafePark()")
    assert marker["n"] == landing.stop_n


def test_q2_stop_before_chunk_0(hub, tmp_path):
    with journalled_rig(hub, tmp_path) as (rig, journal):
        tok = _acquire(rig.door)
        rig.inst.hold_next("LITHOCHAIN", 1)
        _start_write(rig.door, tok, [WIRE_LONG])
        rig.inst.wait_held()
        wait_in_file(journal, command_sent("LITHOCHAIN"))
        _stop_and_settle(rig, tok)
    entries = jr.load(journal.path)

    assert jr.stop_landing(entries).chunks_before == 0
    assert jr.stop_vs_chunk(entries, 0) == "before"
    # The pass was launched (LITHOCHAIN sent) and never answered before the stop line.
    assert jr.in_flight(entries).pending[0].startswith("LITHOCHAIN ")


def test_q2_no_stop_is_not_a_guess(hub, tmp_path):
    with journalled_rig(hub, tmp_path) as (rig, journal):
        tok = _acquire(rig.door)
        _start_write(rig.door, tok, [WIRE])
        rig.door.wait_state("idle", "engaged", "parked")
        _release(rig.door, tok)
    assert jr.stop_vs_chunk(jr.load(journal.path), 0) == "no stop"


# --- 3. who commanded step k ------------------------------------------------


def _first(steps, verb):
    return next(e for e in steps if jr.verb_of(e) == verb or jr.text_of(e) == verb)


def _script_writes_while_operator_stops(hub, client, folder, operator_stop: str):
    """A script (client "golden") starts a write and keeps polling with its token; mid-stroke
    the operator presses `operator_stop` (an HTTP stop route). Returns the file's entries."""
    polling = threading.Event()
    polls = []

    def poll(door, token):
        while not polling.is_set():
            polls.append(door.call("getStatus", {"token": token}))
            polls.append(door.call("getState", {"token": token}))
            time.sleep(0.05)

    with journalled_rig(hub, folder) as (rig, journal):
        tok = _acquire(rig.door)
        poller = threading.Thread(target=poll, args=(FlexDoor(hub), tok), daemon=True)
        poller.start()
        try:
            rig.inst.hold_next("ARMFIRE", 1)
            _start_write(rig.door, tok, [WIRE])
            rig.inst.wait_held()
            wait_in_file(journal, command_sent("ARMFIRE"))
            HttpDoor(client).post(operator_stop, {})
            rig.door.wait_state("idle", "engaged", "parked")
        finally:
            polling.set()
            poller.join(10)
    assert polls, "the script never polled"
    return jr.load(journal.path)


def _who(entries, step):
    c = jr.commander(entries, step)
    return (c.actor, c.row, c.basis) if c else None


def test_q3_script_steps_and_the_operators_withdraw_are_told_apart(hub, client, tmp_path):
    entries = _script_writes_while_operator_stops(hub, client, tmp_path, "/api/withdraw")
    steps = jr.commands(entries)

    # The request's own pre-flight reads carry its trace, whatever else is running.
    assert _who(entries, _first(steps, "READXPT")) == (SCRIPT, "startWrite", "trace")
    # The write worker's steps belong to the run, so to whoever started it.
    assert _who(entries, _first(steps, "ARMFIRE")) == (SCRIPT, "startWrite", "run")
    assert _who(entries, _first(steps, "SERVOLOOP")) == (SCRIPT, "startWrite", "run")
    # The Withdraw handler sends its own steps on the request's thread: exact, the operator's.
    arm = _first(steps, "ARMFIRE")["n"]
    operator = ("operator", "withdraw", "trace")
    mine = [e for e in steps if e["n"] > arm and _who(entries, e) == operator]
    assert mine and jr.verb_of(mine[0]) == "SERVOLOOP"
    # Polling went on throughout, under the script's name, and none of it is taken for a step.
    assert any(r.row == "getStatus" and r.actor == SCRIPT for r in jr.requests(entries))
    assert not any(_who(entries, e) and _who(entries, e)[1] in ("getStatus", "getState")
                   for e in steps)


def test_q3_the_retract_after_an_operators_abort_is_the_operators(hub, client, tmp_path):
    entries = _script_writes_while_operator_stops(hub, client, tmp_path, "/api/litho/abort")
    steps = jr.commands(entries)
    stop = jr.stop_marker(entries)
    assert (stop["what"], jr.text_of(stop)) == ("com", "AFM_SafePark()")

    # The retract runs on the fail-safe thread, which carries no trace: the nearest stop
    # request, here the operator's abort, is the one that asked for it.
    operator = ("operator", "abortWrite", "stop")
    assert _who(entries, stop) == operator
    retract = [e for e in steps if e["n"] > stop["n"] and jr.verb_of(e) in ("ABORTCONFIRM",)]
    assert retract and {_who(entries, e) for e in retract} == {operator}
    # The write's own steps before it still belong to the script.
    assert _who(entries, _first(steps, "ARMFIRE")) == (SCRIPT, "startWrite", "run")


def test_stop_rows_listed_in_the_reader_match_the_request_table():
    stops = {row.name for row in ROWS if row.control == "stop"}
    assert jr.STOP_ROWS == stops
