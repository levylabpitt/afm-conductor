"""Slice 3.2: the write and the scan journal each step's outcome, cited by (execution id, step).

Driven on the golden rig (the fake Igor) through the FLEX door, with a journal on. The tests
read the journal file and the stored run only: every outcome's step must be a line of the stored
run's body of the matching kind, and the steps must come in the run's order.
"""

from __future__ import annotations

import inspect
import json
import threading

import pytest
from golden.harness import HERE, GoldenRig, record
from golden.scenarios import SCENARIOS
from journal_rig import journalled_rig

from afm_litho.run.model import (Check, End, Engage, LithoObject, Move, Read, Scan, SetMode,
                                 Settle, Withdraw)
from afm_litho.service import journal as journal_module
from afm_litho.service import journal_read as jr
from afm_litho.service import run_store
from afm_litho.service.journal import Journal
from afm_litho.service.run_store import RunStore
from afm_litho.service.step_notes import StepNotes, step_groups

# The class of body line each outcome kind is about.
BODY_CLASS = {"check": Check, "move": Move, "settle": Settle, "read": Read, "engage": Engage,
              "pass": (LithoObject, Scan), "withdraw": Withdraw, "end": (Withdraw, End),
              "setmode": SetMode}


@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


def play(rig, name: str) -> None:
    doors = {"door": rig.door, "inst": rig.inst}
    fn = SCENARIOS[name]
    fn(**{k: doors[k] for k in inspect.signature(fn).parameters})
    rig.quiesce()


def stored_run(hub, execution_id: str):
    runtime = hub.store.path.parent
    return RunStore(run_store.store_dir(runtime)).load(execution_id)


def outcomes(entries: list[dict]) -> list[dict]:
    return [e for e in entries if e.get("src") == "step"]


def execution_of(entries: list[dict]) -> str:
    ids = {e["run"] for e in outcomes(entries) if e.get("run")}
    assert len(ids) == 1, ids
    return ids.pop()


def kinds(entries: list[dict]) -> list[str]:
    return [e["what"] for e in outcomes(entries)]


def step_of(entry: dict) -> int | None:
    return (entry.get("body") or {}).get("step")


def assert_steps_follow_the_body(entries: list[dict], run) -> None:
    """Each cited step is a line of the right kind, and the steps never go backwards."""
    body = run.recipe.body
    last = -1
    for e in outcomes(entries):
        n = step_of(e)
        assert n is not None, e
        assert e["run"] == run.execution.execution_id
        if e["what"] != "abort":
            assert isinstance(body[n], BODY_CLASS[e["what"]]), (e["what"], body[n])
        assert n >= last, "steps go forward"
        last = n


def test_an_aborted_two_object_write_journals_its_steps_in_the_runs_order(hub, tmp_path):
    with journalled_rig(hub, tmp_path) as (rig, journal):
        play(rig, "abort_in_transit")
    entries = jr.load(journal.path)
    run = stored_run(hub, execution_of(entries))
    assert run.recipe.kind == "write" and len([e for e in run.recipe.body
                                                if isinstance(e, LithoObject)]) == 2
    assert_steps_follow_the_body(entries, run)
    # The first object ran whole; the second's move finished, and the abort stopped its read.
    assert kinds(entries)[:7] == ["check", "move", "settle", "read", "engage", "pass",
                                  "withdraw"]
    first_pass = next(n for n, e in enumerate(run.recipe.body) if isinstance(e, LithoObject))
    story = jr.step_outcomes(entries, run.execution.execution_id)
    assert [step_of(e) for e in story[:7]] == [0, 1, 2, 3, 4, first_pass, first_pass + 1]
    by_kind = {e["what"]: e for e in story}
    assert by_kind["move"]["body"]["how"] == "transit"
    assert by_kind["move"]["body"]["verified"] is True
    assert by_kind["move"]["body"]["retried"] is False
    assert len(by_kind["read"]["body"]["reads"]) == 3
    assert by_kind["engage"]["body"]["setpoint_v"] == by_kind["read"]["body"]["setpoint_v"]
    assert by_kind["pass"]["body"]["fire"] == "ARMFIRE"
    assert by_kind["pass"]["body"]["aborted"] is False
    assert by_kind["pass"]["body"]["points"] == run.recipe.body[first_pass].points
    (abort,) = [e for e in story if e["what"] == "abort"]
    assert step_of(abort) == first_pass + 3
    assert abort["body"]["during"] == "read"
    assert step_of(story[7]) == first_pass + 2 and story[7]["what"] == "move"
    # The closing step is the run's last line.
    assert story[-1]["what"] == "end"
    assert step_of(story[-1]) == len(run.recipe.body) - 1
    # "What happened at step n" returns only that step's outcomes.
    (only,) = jr.step_outcomes(entries, run.execution.execution_id, first_pass)
    assert only["what"] == "pass"


def test_the_wire_entries_of_a_write_name_the_step_in_progress(hub, tmp_path):
    with journalled_rig(hub, tmp_path) as (rig, journal):
        play(rig, "litho_single")
    entries = jr.load(journal.path)
    run = stored_run(hub, execution_of(entries))
    body = run.recipe.body
    fire = [e for e in jr.commands(entries) if jr.verb_of(e) == "ARMFIRE"]
    assert len(fire) == 1
    assert fire[0]["run"] == run.execution.execution_id
    assert isinstance(body[fire[0]["body"]["step"]], Engage | LithoObject)
    # The stamp stays within the run's steps.
    stamped = [e for e in entries if e.get("src") == "wire" and "step" in (e.get("body") or {})]
    assert stamped and all(0 <= e["body"]["step"] < len(body) for e in stamped)


def test_a_completed_scan_journals_its_steps_in_the_runs_order(hub, tmp_path):
    with journalled_rig(hub, tmp_path) as (rig, journal):
        play(rig, "scan_chained_contact")
    entries = jr.load(journal.path)
    run = stored_run(hub, execution_of(entries))
    assert run.recipe.kind == "scan"
    assert_steps_follow_the_body(entries, run)
    assert kinds(entries) == ["check", "setmode", "move", "pass", "end"]
    story = jr.step_outcomes(entries, run.execution.execution_id)
    assert [step_of(e) for e in story] == [0, 1, 2, 3, 4]
    assert isinstance(run.recipe.body[3], Scan)
    scan = story[3]["body"]
    assert scan["fire"] == "SCANCHAIN" and scan["aborted"] is False
    assert scan["lines_delivered"] == scan["lines_planned"] == run.recipe.body[3].lines
    assert story[1]["body"]["switched"] is False  # the rig already reads "contact"
    assert story[2]["body"]["how"] == "transit" and story[2]["body"]["verified"] is True


def test_a_scan_that_engages_journals_the_engage(hub, tmp_path):
    with journalled_rig(hub, tmp_path) as (rig, journal):
        play(rig, "scan_chained_ac")
    entries = jr.load(journal.path)
    run = stored_run(hub, execution_of(entries))
    assert_steps_follow_the_body(entries, run)
    assert "engage" in kinds(entries) or not any(isinstance(e, Engage) for e in run.recipe.body)
    assert kinds(entries)[-2:] == ["pass", "end"]


def test_a_write_with_no_run_still_journals_its_outcomes_without_a_citation(
        hub, tmp_path, monkeypatch):
    def broken(**_kw):
        raise RuntimeError("forced")
    monkeypatch.setattr("afm_litho.run.build_write.build_write_run", broken)
    with journalled_rig(hub, tmp_path) as (rig, journal):
        play(rig, "litho_single")
    entries = jr.load(journal.path)
    notes = outcomes(entries)
    assert [e["what"] for e in notes if e["what"] != "check"][:5] == [
        "move", "settle", "read", "engage", "pass"]
    assert all("run" not in e and "step" not in (e.get("body") or {}) for e in notes)
    assert not [e for e in entries if e.get("src") == "wire" and "run" in e]


def frozen_wire(name: str):
    return json.loads((HERE / f"{name}.golden.json").read_text())["wire"]["actuation"]


def test_a_note_that_fails_changes_nothing_the_instrument_is_told(hub, client, monkeypatch):
    def broken(self, *_a, **_kw):
        raise RuntimeError("note failed")
    monkeypatch.setattr(StepNotes, "_note", broken)
    for name in ("litho_single", "scan_chained_contact"):
        assert record(hub, name, client=client)["wire"]["actuation"] == frozen_wire(name)


def test_a_journal_that_cannot_write_does_not_hold_up_a_write(hub, tmp_path):
    """The writer is stuck on its first line and the queue holds one: notes are dropped, and
    the write finishes with its normal result."""
    release = threading.Event()
    journal = Journal(tmp_path, queue_size=1, sink=lambda _line: release.wait(30)).start()
    try:
        with GoldenRig(hub, "litho_single") as rig:
            play(rig, "litho_single")
            reply = rig.door.call("getWriteResult", {})
        assert reply["aborted"] is False
        assert journal.write_failures == 0
    finally:
        release.set()
        journal.stop()
        journal_module._running = None


def test_steps_are_grouped_by_pass_from_the_runs_own_body(hub, tmp_path):
    with journalled_rig(hub, tmp_path) as (rig, journal):
        play(rig, "litho_chain_two_chunks")
    entries = jr.load(journal.path)
    run = stored_run(hub, execution_of(entries))
    groups = step_groups(run)
    passes = [n for n, e in enumerate(run.recipe.body) if isinstance(e, LithoObject)]
    assert [g["pass"] for g in groups if "pass" in g] == passes
    assert groups[0]["check"] == 0
    assert groups[-1]["end"] == len(run.recipe.body) - 1
    assert all(g["withdraw"] == g["pass"] + 1 for g in groups if "pass" in g)
    (done,) = [e for e in outcomes(entries) if e["what"] == "pass"]
    assert done["body"]["fire"] == "LITHOCHAIN" and done["body"]["chunks"] == 2


def test_a_transit_that_needed_a_retry_says_so(hub, tmp_path):
    with journalled_rig(hub, tmp_path) as (rig, journal):
        play(rig, "transit_retry")
    entries = jr.load(journal.path)
    (move,) = [e for e in outcomes(entries) if e["what"] == "move"]
    assert move["body"]["retried"] is True and move["body"]["verified"] is True


def test_a_write_with_a_filled_region_has_no_run_and_its_outcomes_are_uncited(hub, tmp_path):
    """The run contract has one fire per object, so a filled pad has no run (3.1a)."""
    with journalled_rig(hub, tmp_path) as (rig, journal):
        play(rig, "litho_multi_object")
    notes = outcomes(jr.load(journal.path))
    assert notes and all("run" not in e for e in notes)
    assert {"move", "pass", "withdraw", "end"} <= {e["what"] for e in notes}
    assert any((e.get("body") or {}).get("how") == "hover" for e in notes)


def test_a_pass_that_igor_aborted_journals_the_abort_at_that_pass(hub, tmp_path):
    with journalled_rig(hub, tmp_path) as (rig, journal):
        play(rig, "abort_mid_stroke")
    entries = jr.load(journal.path)
    run = stored_run(hub, execution_of(entries))
    first_pass = next(n for n, e in enumerate(run.recipe.body) if isinstance(e, LithoObject))
    (done,) = [e for e in outcomes(entries) if e["what"] == "pass"]
    assert done["body"]["aborted"] is True and step_of(done) == first_pass
    (abort,) = [e for e in outcomes(entries) if e["what"] == "abort"]
    assert step_of(abort) == first_pass and abort["body"]["during"] == "pass"


def test_the_closing_withdraw_after_an_abort_is_stamped_with_the_end_step(hub, tmp_path):
    with journalled_rig(hub, tmp_path) as (rig, journal):
        play(rig, "abort_in_transit")
    entries = jr.load(journal.path)
    run = stored_run(hub, execution_of(entries))
    (abort,) = [e for e in outcomes(entries) if e["what"] == "abort"]
    after = [e for e in entries if e.get("n", 0) > abort["n"] and e.get("src") == "wire"
             and e.get("run")]
    assert after
    assert {e["body"]["step"] for e in after} == {len(run.recipe.body) - 1}


def test_a_skipped_withdraw_leaves_the_abort_at_the_withdraw(tmp_path):
    from golden.scenarios import WIRE
    from test_build_write_run import compile_scenario

    from afm_litho.hardware.lithography_run import WriteSetpoint
    from afm_litho.run import Placement
    objs, segments = compile_scenario([WIRE])
    stored = run_store.emit_write_run(
        RunStore(tmp_path / "runs"), tmp_path, design_doc={"objects": []}, compile_doc={},
        design_name="w", objects=objs, segments=segments,
        setpoint=WriteSetpoint(0.0, relative=False), amp_gain=25.0, platform="cypher",
        placement=Placement("test", 30, 0, (0, 0)), bias_window_v=(-8, 30))
    run = stored.run
    journal = Journal(tmp_path / "j").start()
    try:
        notes = StepNotes(run, "write")
        for kind in ("check", "move", "settle", "read", "engage", "pass"):
            notes.outcome(kind)
        notes.outcome("withdraw", skipped=True)
        notes.abort("abort flag set after the pass")
    finally:
        journal.stop()
        journal_module._running = None
    entries = jr.load(journal.path)
    (abort,) = [e for e in outcomes(entries) if e["what"] == "abort"]
    withdraw = next(n for n, e in enumerate(run.recipe.body) if isinstance(e, Withdraw))
    assert step_of(abort) == withdraw and abort["body"]["during"] == "withdraw"


def test_notes_that_cannot_be_built_leave_the_write_and_the_scan_as_they_were(
        hub, client, monkeypatch):
    def broken(_run):
        raise RuntimeError("no groups")
    monkeypatch.setattr("afm_litho.service.step_notes.step_groups", broken)
    for name in ("litho_single", "scan_chained_contact"):
        assert record(hub, name, client=client)["wire"]["actuation"] == frozen_wire(name)


def test_an_outcome_that_cannot_be_worked_out_leaves_the_write_as_it_was(
        hub, client, monkeypatch):
    from afm_litho.hardware.lithography_run import LithographyRun

    class Odd(dict):
        def values(self):
            raise RuntimeError("cannot size the capture")
    fire = LithographyRun._fire_and_wait

    def fire_with_odd_capture(self, *a, **kw):
        verb, uid, capture = fire(self, *a, **kw)
        return verb, uid, Odd(capture or {})
    monkeypatch.setattr(LithographyRun, "_fire_and_wait", fire_with_odd_capture)
    assert record(hub, "litho_single", client=client)["wire"]["actuation"] == frozen_wire(
        "litho_single")
