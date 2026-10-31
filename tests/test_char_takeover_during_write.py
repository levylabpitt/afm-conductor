"""Characterisation (slice 0.7): the browser takes control while a FLEX script is writing.

A two-wire hardware write on the Cypher stub; the second wire's stroke is held on the
instrument, the cockpit posts `/api/control/take`, and the tests pin what the script sees,
what the write does, and the control state. The write is then ended with the token-free
`abortWrite`, which also gives the half-written marks (wire done, wire_b cut short).
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from char_rig import WIRE, WIRE_B, CharRig, acquire, start_write

pytest_plugins = ["char_rig"]    # the module-scope `hub` fixture


@pytest.fixture(scope="module")
def takeover(hub, client):
    with CharRig(hub, safety_sim=False) as rig:
        door = rig.door
        tok = acquire(door)
        rig.inst.hold_next("ARMFIRE", 2)
        run = start_write(door, tok, [WIRE, WIRE_B])
        rig.inst.wait_held()
        marks_before = [o["written"] for o in door.call("getPattern")["objects"]]
        verbs_before = rig.verbs()
        taken = client.post("/api/control/take", json={"client": "browser"}).json()
        time.sleep(1.0)
        seen = SimpleNamespace(
            taken=taken, marks_before=marks_before,
            state=door.call("getState"),
            verbs_unchanged=rig.verbs() == verbs_before,
            heartbeat=door.call_refused("heartbeat", {"token": tok}),
            start_write=door.call_refused("startWrite", {"token": tok, "objects": "all"}),
            commander=door.call("getStatus")["commander"],
            control=client.get("/api/control").json(),
            released=door.call("releaseControl", {"token": tok}))
        door.call("abortWrite", {})
        door.wait_state("idle", "engaged", "parked")
        seen.result = door.call("getWriteResult", {"run_id": run["run_id"]})
        seen.marks_after = [o["written"] for o in door.call("getPattern")["objects"]]
        seen.written = door.call("getWritten")
        seen.final_commander = door.call("getStatus")["commander"]
        rig.quiesce()
        return seen


def test_the_take_reports_whom_it_preempted(takeover):
    assert takeover.taken["ok"] is True
    assert takeover.taken["preempted"] == "characterisation"
    assert takeover.taken["control"]["kind"] == "operator"
    assert takeover.taken["control"]["expires_s"] is None


def test_the_write_carries_on_untouched_after_the_take(takeover):
    """No stop, no park, no extra command reaches the instrument; the state stays writing."""
    assert takeover.state == "writing"
    assert takeover.verbs_unchanged is True


def test_the_script_finds_out_only_by_its_next_call(takeover):
    for err in (takeover.heartbeat, takeover.start_write):
        assert err["code"] == -32011
        assert err["data"]["reason"] == "revoked" and err["data"]["holder"] == "operator"


def test_the_control_state_is_the_operator_with_no_dead_man(takeover):
    for c in (takeover.commander, takeover.control["control"]):
        assert (c["holder"], c["kind"]) == ("operator", "operator")
    assert takeover.commander["expires_s"] is None
    assert takeover.control["engaged"] is True


def test_the_scripts_release_after_the_take_is_stale_and_not_an_error(takeover):
    assert takeover.released["stale"] is True


def test_the_operator_keeps_control_after_the_write_is_aborted(takeover):
    assert takeover.final_commander["holder"] == "operator"


def test_abort_after_the_take_records_the_run_as_aborted(takeover):
    r = takeover.result
    assert (r["completed"], r["aborted"], r["error"]) == (True, True, None)
    assert r["stats"]["objects"] == 1 and r["stats"]["objects_submitted"] == 2


def test_an_abort_mid_write_leaves_only_the_finished_wire_marked(takeover):
    assert takeover.marks_before == [True, False]
    assert takeover.marks_after == [True, False]
    assert [o["name"] for o in takeover.written["objects"]] == ["wire"]
    assert takeover.written["aborted"] is True
