"""Slice J3, second half: the awkward cases for the three questions.

A stop while waves are still loading, two runs and two stops in one file, a retract nobody
asked for, a Stop during a scan (whose park is sent by a task the scan started), and the reader's
own edge cases on hand-made files.
"""

from __future__ import annotations

import json
import time

import pytest
from golden.harness import HttpDoor
from golden.scenarios import WIRE, WIRE_LONG, _acquire, _release, _scan, _start_write
from journal_rig import journalled_rig, wait_in_file
from test_journal_questions import SCRIPT, _first, _who

from afm_litho.service import journal_read as jr


@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


def _waves_sent(count: int):
    return lambda entries: sum(1 for e in jr.commands(entries)
                               if jr.verb_of(e) == "WAVEAPP") >= count


# --- question 1 and 2: a stop while the waves are still loading --------------


def test_stop_during_wave_load_has_nothing_pending_and_no_chunk(hub, tmp_path):
    with journalled_rig(hub, tmp_path) as (rig, journal):
        tok = _acquire(rig.door)
        _start_write(rig.door, tok, [WIRE_LONG])
        wait_in_file(journal, _waves_sent(3))
        rig.door.call("abortWrite", {})
        rig.door.wait_state("idle", "engaged", "parked")
        _release(rig.door, tok)
    entries = jr.load(journal.path)

    state = jr.in_flight(entries)
    assert state.stopped and state.run_id == jr.runs(entries)[0].run_id
    assert state.object == "wire_long"
    # The pass was never launched, so nothing is waiting for an answer.
    assert state.pending == []
    assert state.last_commands[-1].startswith("WAVEAPP ")
    assert not any(jr.verb_of(e) == "LITHOCHAIN" and e["n"] < jr.stop_marker(entries)["n"]
                   for e in jr.commands(entries))
    assert jr.stop_vs_chunk(entries, 0) == "before"


# --- two runs and two stops in one file ------------------------------------


def test_second_run_and_second_stop_are_told_apart(hub, tmp_path):
    with journalled_rig(hub, tmp_path) as (rig, journal):
        for _ in range(2):
            tok = _acquire(rig.door)
            rig.inst.hold_next("ARMFIRE", 1)
            _start_write(rig.door, tok, [WIRE])
            rig.inst.wait_held()
            rig.door.call("abortWrite", {})
            rig.door.wait_state("idle", "engaged", "parked")
            _release(rig.door, tok)
    entries = jr.load(journal.path)

    first, second = jr.runs(entries)
    assert first.ended_n < second.request.n
    one, two = jr.stop_landing(entries), jr.stop_landing(entries, second.run_id)
    assert (one.run_id, two.run_id) == (first.run_id, second.run_id)
    assert first.request.n < one.stop_n < second.request.n < two.stop_n
    assert jr.in_flight(entries, run_id=second.run_id).at_n == two.stop_n
    # Each run's launch is pending at its own stop only.
    assert len(jr.in_flight(entries, run_id=first.run_id).pending) == 1
    assert len(jr.in_flight(entries, run_id=second.run_id).pending) == 1
    # A step of the second run is the second request's.
    arm2 = [e for e in jr.commands(entries) if jr.verb_of(e) == "ARMFIRE"][1]
    assert jr.commander(entries, arm2).trace == second.request.trace


# --- a retract nobody asked for ---------------------------------------------


def test_q3_a_dead_man_retract_is_the_services_own(hub, tmp_path):
    with journalled_rig(hub, tmp_path) as (rig, journal):
        tok = rig.door.call("acquireControl", {"client": SCRIPT, "deadman_s": 5.0,
                                               "withdraw_on_deadman": False})["token"]
        rig.inst.hold_next("ARMFIRE", 1)
        _start_write(rig.door, tok, [WIRE])
        while not rig.inst.is_held():          # the start outlasts the 5 s dead-man
            rig.door.call("heartbeat", {"token": tok})
            time.sleep(0.5)
        rig.inst.wait_held()
        rig.door.wait_state("parked", timeout=60.0)    # the script goes silent; the service parks
        _release(rig.door, tok)
    entries = jr.load(journal.path)

    assert not any(r.row in jr.STOP_ROWS for r in jr.requests(entries))
    stop = jr.stop_marker(entries)
    who = jr.commander(entries, stop)
    assert (who.actor, who.basis, who.trace) == ("service", "fault", None)
    assert who.row.startswith("safe_park")
    retract = [e for e in jr.commands(entries) if e["n"] > stop["n"]
               and jr.verb_of(e) in ("ABORTCONFIRM",)]
    assert retract and {jr.commander(entries, e).basis for e in retract} == {"fault"}


# --- a Stop during a scan ---------------------------------------------------


def test_q3_a_stop_during_a_scan_is_not_blamed_on_the_script_that_started_it(
        hub, client, tmp_path):
    with journalled_rig(hub, tmp_path) as (rig, journal):
        tok = _acquire(rig.door)
        rig.inst.hold_scan_after_line(1)
        _scan(rig.door, tok, mode="contact", lines=4)
        rig.inst.wait_held()
        rig.door.wait_scan_row(1)
        before = jr.load(journal.path)[-1]["n"]
        HttpDoor(client).post("/api/scan/abort", {})       # the operator's Stop
        rig.inst.release_hold()
        rig.door.wait_state("idle", "engaged", "parked")
        _release(rig.door, tok)
    entries = jr.load(journal.path)
    scan = jr.runs(entries)[0]
    assert scan.request.row == "startScan"

    after = [e for e in jr.commands(entries) if e["n"] > before and jr.verb_of(e) != "GETVALUE"]
    assert after, "no commands after the Stop"
    # No command after the Stop carries the scan request's trace...
    assert all(e.get("trace") != scan.request.trace for e in after)
    # ...and every retract command is the Stop's, or unattributed, never the script's.
    for e in after:
        who = jr.commander(entries, e)
        assert who is None or who.row != "startScan" or who.basis == "run", (e, who)
    retract = [e for e in after if jr.text_of(e) in jr.STOP_LINES
               or jr.verb_of(e) in ("ABORTCONFIRM",)]
    assert retract
    assert {_who(entries, e) for e in retract} == {("operator", "abortScan", "stop")}
    # The scan's own launch is sent from a pooled thread that was already in use, which the
    # run rule will not claim: it is the script's or unattributed, never someone else's.
    launch = _who(entries, _first(jr.commands(entries), "SCANCHAIN"))
    assert launch in (None, (SCRIPT, "startScan", "run"))


# --- the reader on hand-made files ------------------------------------------


def _entry(n, src, what, **kw):
    return {"n": n, "src": src, "what": what, "t": "x", "mono": 0.0, **kw}


def _wire(n, what, text, thread="T"):
    return _entry(n, "wire", what, body={"text": text, "thread": thread})


def test_captures_read_after_the_stop_line_are_not_counted():
    chans = ["Output.A", "ZSensor", "Deflection", "XSensor", "YSensor"]
    entries = [
        _entry(1, "door", "request", trace=1, actor="s", body={"row": "startWrite"}),
        _entry(2, "door", "reply", trace=1, actor="s",
               body={"row": "startWrite", "result": {"run_id": "write-1"}}),
        _entry(3, "stream", "litho_object_start", body={"msg": {"name": "o"}}),
        _wire(4, "send", "LITHOCHAIN 9 o X Y V B 1 1", "W"),
        *[_wire(5 + i, "read", f"CAPTURE 9 o {c}", "R") for i, c in enumerate(chans[:2])],
        _wire(7, "com", "AFM_SafePark()", "afm-failsafe"),   # the stop lands mid-chunk
        *[_wire(8 + i, "read", f"CAPTURE 9 o {c}", "R") for i, c in enumerate(chans[2:])],
    ]
    landing = jr.stop_landing(entries)
    assert landing.chunks_before == 0 and landing.stop_n == 7
    assert jr.stop_vs_chunk(entries, 0) == "before"
    assert jr.chunks_delivered(entries, "o", 99) == 1     # asked after the fact it is whole


def test_a_torn_last_line_is_dropped_but_a_damaged_middle_is_reported(tmp_path):
    good = [json.dumps({"n": i, "src": "x", "what": "y"}) for i in (1, 2)]
    torn = tmp_path / "torn.jsonl"
    torn.write_text("\n".join(good) + '\n{"n": 3, "sr', encoding="utf-8")
    entries = jr.load(torn)
    assert len(entries) == 2 and not jr.is_closed(entries)
    damaged = tmp_path / "damaged.jsonl"
    damaged.write_text(good[0] + "\nnot json\n" + good[1] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        jr.load(damaged)


def test_a_step_is_matched_to_the_run_whose_window_holds_it():
    def run(n, run_id, trace):
        return [_entry(n, "door", "request", trace=trace, actor=f"a{trace}",
                       body={"row": "startWrite"}),
                _entry(n + 1, "door", "reply", trace=trace, actor=f"a{trace}",
                       body={"row": "startWrite", "result": {"run_id": run_id}})]
    entries = [*run(1, "write-1", 1), _wire(3, "send", "ARMFIRE 1 a", "w1"),
               _entry(4, "stream", "litho_done"), *run(5, "write-2", 2),
               _wire(7, "send", "ARMFIRE 2 b", "w2"), _entry(8, "stream", "litho_done")]
    assert [r.run_id for r in jr.runs(entries)] == ["write-1", "write-2"]
    assert jr.commander(entries, entries[2]).trace == 1
    assert jr.commander(entries, entries[6]).trace == 2


def test_a_thread_already_sending_before_the_run_is_not_taken_for_the_runs():
    entries = [
        _wire(1, "send", "GETVALUE 1 XSensor", "MainThread"),
        _entry(2, "door", "request", trace=1, actor="a", body={"row": "startWrite"}),
        _entry(3, "door", "reply", trace=1, actor="a",
               body={"row": "startWrite", "result": {"run_id": "write-1"}}),
        _wire(4, "send", "GETVALUE 2 XSensor", "MainThread"),
        _wire(5, "send", "ARMFIRE 3 a", "worker"),
    ]
    assert jr.commander(entries, entries[3]) is None
    assert jr.commander(entries, entries[4]).basis == "run"


def _run_entries(n0, run_id="write-1", trace=1):
    return [_entry(n0, "door", "request", trace=trace, actor="s", body={"row": "startWrite"}),
            _entry(n0 + 1, "door", "reply", trace=trace, actor="s",
                   body={"row": "startWrite", "result": {"run_id": run_id}})]


def _park(n, reason):
    return _entry(n, "stream", "safe_park", body={"msg": {"reason": reason}})


def _stop_request(n, trace, what="reply", row="abortWrite"):
    return [_entry(n, "door", "request", trace=trace, actor="operator", body={"row": row}),
            _entry(n + 1, "door", what, trace=trace, actor="operator", body={"row": row})]


def test_a_stop_after_the_run_ended_is_not_the_runs():
    entries = [*_run_entries(1), _entry(3, "stream", "litho_done"),
               _wire(4, "com", "AFM_SafePark()", "afm-failsafe")]
    assert jr.stop_landing(entries, "write-1") is None
    assert jr.stop_vs_chunk(entries, 0, "write-1") == "no stop"
    assert jr.stop_landing(entries).run_id is None        # a retract with no run to blame


def test_a_dead_man_park_followed_by_a_later_stop_stays_a_fault():
    entries = [
        *_run_entries(1),
        _wire(3, "send", "STOPLOOP 2", "afm-failsafe"), _park(4, "dead-man"),
        # later, the operator stops (a write-style Stop: its own park, then its request)
        _wire(5, "send", "STOPLOOP 2", "afm-failsafe"), _park(6, "litho abort"),
        *_stop_request(7, 2),
    ]
    first, second = (jr.commander(entries, e) for e in (entries[2], entries[4]))
    assert (first.actor, first.basis) == ("service", "fault")
    assert (second.actor, second.row, second.basis) == ("operator", "abortWrite", "stop")


def test_a_dead_man_park_followed_by_a_scan_style_stop_stays_a_fault():
    entries = [
        *_run_entries(1),
        _wire(3, "send", "STOPLOOP 2", "afm-failsafe"), _park(4, "dead-man"),
        *_stop_request(5, 2, row="abortScan"),            # answered at once; parks afterwards
        _wire(7, "send", "STOPLOOP 2", "afm-failsafe"), _park(8, "scan stopped"),
    ]
    first, second = (jr.commander(entries, e) for e in (entries[2], entries[6]))
    assert first.basis == "fault"
    assert (second.row, second.basis) == ("abortScan", "stop")


def test_a_stop_that_errored_did_not_run_so_its_park_is_a_fault():
    entries = [*_run_entries(1), _wire(3, "send", "STOPLOOP 2", "afm-failsafe"),
               _park(4, "dead-man"), *_stop_request(5, 2, what="error")]
    assert jr.commander(entries, entries[2]).basis == "fault"
