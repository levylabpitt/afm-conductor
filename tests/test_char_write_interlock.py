"""Characterisation (slice 0.7): an interlock trips during a hardware write.

Audit B listed this as tested on the twin only. Here a two-wire write runs on the Cypher stub
with the always-on tip poll streaming; the first wire's stroke is held on the instrument, the
detector SUM collapses, and the rail-dwell interlock (5 s) trips. The tests pin what the
service does and records today. The golden `interlock_trip` freezes the same trip while engaged
with no write; the wire commands of this one are not frozen.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from char_rig import WIRE, WIRE_B, CharRig, acquire, start_write

pytest_plugins = ["char_rig"]    # the module-scope `hub` fixture


@pytest.fixture(scope="module")
def trip(hub):
    """Two wires; SUM collapses while the first wire's stroke is on the instrument."""
    with CharRig(hub, telem_hz=50.0) as rig:
        door = rig.door
        tok = acquire(door)
        rig.inst.hold_next("ARMFIRE", 1)
        run = start_write(door, tok, [WIRE, WIRE_B])
        rig.inst.wait_held()
        state_before = door.call("getState")
        rig.inst.set_value("DetectorSum", 0.1)
        door.wait_state("parked", "fault", "idle", timeout=60.0)
        rig.quiesce()
        sim = rig.stub.safety_sim
        yield SimpleNamespace(
            state_before=state_before,
            state=door.call("getState"),
            status=door.call("getStatus"),
            park=door.call("getLastPark"),
            result=door.call("getWriteResult", {"run_id": run["run_id"]}),
            written=door.call("getWritten"),
            marks=[o["written"] for o in door.call("getPattern")["objects"]],
            verbs=rig.verbs(),
            bias_v=sim.bias_volts, tip_off=sim.tip_off_surface, xy_moving=sim.xy_moving)


def test_the_write_was_running_before_the_trip(trip):
    assert trip.state_before == "writing"


def test_the_trip_stops_the_write_and_the_second_wire_is_never_armed(trip):
    assert trip.verbs.count("ARMFIRE") == 1
    assert trip.state == "parked"


def test_the_park_follows_the_trip_with_the_reason(trip):
    assert trip.park["reason"] == "sum_collapse"
    assert trip.park["verified"] is True and trip.park["critical"] is False
    assert trip.status["detail"] == "sum_collapse"
    assert trip.status["safe_park"]["active"] is True


def test_the_park_sequence_is_after_the_arm(trip):
    """The worker's end-of-run Z loop, then the park's Height confirm (ABORTCONFIRM). Since
    slice AB the park sends no STOPLOOP or RAMP of its own (SLICES F5 defect-1 fixed)."""
    after_arm = trip.verbs[trip.verbs.index("ARMFIRE"):]
    assert "ABORTCONFIRM" in after_arm
    assert "STOPLOOP" not in after_arm and "RAMP" not in after_arm


def test_the_instrument_ends_in_the_safe_state(trip):
    assert trip.bias_v == 0.0
    assert trip.tip_off is True
    assert trip.xy_moving is False


def test_the_run_is_recorded_as_aborted_and_finished_with_no_reason(trip):
    """Today the run record holds `completed: true, aborted: true`, no objects counted
    and no error: the reason lives only in the park and the status detail."""
    r = trip.result
    assert (r["completed"], r["aborted"], r["error"]) == (True, True, None)
    assert r["stats"]["objects"] == 0 and r["stats"]["objects_submitted"] == 2


def test_no_object_is_marked_written_or_listed(trip):
    assert trip.marks == [False, False]
    assert trip.written["objects"] == [] and trip.written["aborted"] is True


def test_the_script_keeps_the_token_after_a_trip(trip):
    assert trip.status["commander"]["holder"] == "characterisation"
