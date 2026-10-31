"""Characterisation (slice 0.7): the written-marks record and commanded versus delivered dose.

Two hardware writes on the Cypher stub, through FLEX: two separate wires, and one chained wire
over the chunk size. The goldens freeze the commands; these tests pin what the service
records about each object (marks, `getWritten`, the npz bias array, the sidecar) against what
the stub received (the bias wave and its timing).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from char_rig import WIRE, WIRE_B, WIRE_LONG, CharRig, acquire, sidecar, start_write, waves_sent

pytest_plugins = ["char_rig"]    # the module-scope `hub` fixture

AMP_GAIN = 25.0
TICK_S = 20e-6              # one DSP tick; an ARMFIRE interp of -500 is 500 ticks per point


def _marks(door) -> dict[str, bool]:
    return {o["name"]: o["written"] for o in door.call("getPattern")["objects"]}


def _write(hub, objects, *, edit=None) -> SimpleNamespace:
    with CharRig(hub, safety_sim=False) as rig:
        door = rig.door
        tok = acquire(door)
        run = start_write(door, tok, objects, amp_gain=AMP_GAIN)
        door.wait_state("idle", "engaged", "parked")
        result = door.call("getWriteResult", {"run_id": run["run_id"]})
        out = SimpleNamespace(
            result=result, written=door.call("getWritten"), marks=_marks(door),
            pattern=door.call("getPattern"), sidecar=sidecar(result),
            cmd_v=np.load(result["files"]["npz"])["output_a_cmd_v"],
            bias_waves=waves_sent(rig, "VLitho"),
            armfires=[e["args"] for e in rig.tap.actuation if e["verb"] == "ARMFIRE"],
            chains=[e["args"] for e in rig.tap.actuation if e["verb"] == "LITHOCHAIN"])
        if edit:
            door.call("setObjects", {"token": tok, "updates": edit})
            out.marks_after_edit = _marks(door)
        rig.quiesce()
        return out


@pytest.fixture(scope="module")
def two_wires(hub):
    return _write(hub, [WIRE, WIRE_B], edit={"wire_b": {"voltage": 4.0}})


@pytest.fixture(scope="module")
def chained(hub):
    return _write(hub, [WIRE_LONG])


# ------------------------------------------------------------------- written marks

def test_every_object_of_a_finished_write_is_marked_written(two_wires):
    assert two_wires.marks == {"wire": True, "wire_b": True}
    assert [o["name"] for o in two_wires.written["objects"]] == ["wire", "wire_b"]


def test_an_edit_of_one_object_loses_the_marks_of_all_of_them(two_wires):
    """Defect 12 as it is today: `setObjects` deep-copies the pattern, the marks are keyed
    by `id(obj)`, so no mark survives. The next default write re-doses both wires."""
    assert two_wires.marks_after_edit == {"wire": False, "wire_b": False}


@pytest.mark.xfail(strict=True, reason="defect 12 (D-migration-risks item 12): marks are "
                   "keyed by id(obj), so an edit of wire_b also unmarks the untouched wire")
def test_an_edit_keeps_the_marks_of_the_objects_it_did_not_touch(two_wires):
    assert two_wires.marks_after_edit["wire"] is True


# ------------------------------------------------------------- dose, a single stroke

def test_the_bias_delivered_per_wire_is_the_voltage_over_the_amp_gain(two_wires):
    """Each wire's bias wave starts at its object voltage / amp gain and ends on 0."""
    first, second = two_wires.bias_waves
    assert first["first"] == pytest.approx(5.0 / AMP_GAIN, abs=1e-6)
    assert second["first"] == pytest.approx(3.0 / AMP_GAIN, abs=1e-6)
    assert first["last"] == second["last"] == 0.0


def test_the_record_lists_the_object_voltage_and_not_the_amp_gain_scaled_bias(two_wires):
    assert [o["voltage"] for o in two_wires.written["objects"]] == [5.0, 3.0]


def test_the_npz_bias_array_is_what_was_sent_wire_after_wire(two_wires):
    sent = [w["n"] for w in two_wires.bias_waves]
    assert sent == [32, 32]
    v = two_wires.cmd_v
    assert len(v) == sum(sent) == two_wires.result["shape"][1]
    assert v[0] == pytest.approx(0.2) and v[31] == 0.0
    assert v[32] == pytest.approx(0.12) and v[-1] == 0.0


def test_the_delivered_stroke_time_matches_length_over_speed(two_wires):
    """0.3 um at 1 um/s: the ARMFIRE says 32 points at 500 ticks each."""
    for args in two_wires.armfires:
        interp, n_points = abs(int(args[2])), int(args[3])
        assert n_points * interp * TICK_S == pytest.approx(0.3, abs=0.03)


def test_the_record_counts_fewer_points_than_were_delivered(two_wires):
    """`getWritten.points` is the decimated path (9), not the 32 points the controller
    played."""
    assert [o["points"] for o in two_wires.written["objects"]] == [9, 9]


@pytest.mark.xfail(strict=True, reason="NEW 0.7-1: on hardware getWritten.length_um and "
                   "t_start are null, so the record holds no length or duration per object")
def test_the_record_holds_a_length_and_a_start_time_per_object(two_wires):
    entry = two_wires.written["objects"][0]
    assert entry["length_um"] == pytest.approx(0.3, abs=0.01)
    assert entry["t_start"] is not None


def test_the_sidecar_holds_setpoints_and_gain_but_no_per_object_dose(two_wires):
    sc = two_wires.sidecar
    assert sc["amp_gain"] == AMP_GAIN and sc["aborted"] is False
    assert [o["object"] for o in sc["object_setpoints"]] == ["wire", "wire_b"]
    assert set(sc["object_setpoints"][0]) == {"object", "free_air_v", "setpoint_v", "note"}


def test_get_write_result_does_not_list_the_sidecar(two_wires):
    assert set(two_wires.result["files"]) == {"npz", "gwy", "ibw"}


# ------------------------------------------------------------- dose, a chained stroke

def test_a_chained_wire_is_one_entry_with_two_segments(chained):
    (entry,) = chained.written["objects"]
    assert entry["segments"] == 2 and entry["voltage"] == 5.0
    assert chained.marks == {"wire_long": True}


def test_the_chain_delivers_every_point_in_one_lithochain(chained):
    """The bias wave is sent in pieces and ends at 100,032 points (a multiple of 32); the
    record's channel holds all of them."""
    assert len(chained.chains) == 1
    assert chained.bias_waves[-1]["n"] == 100032
    assert chained.cmd_v.shape == (100032,) == (chained.result["shape"][1],)


def test_the_bias_stays_on_across_the_seam_and_ends_on_one_zero(chained):
    v = chained.cmd_v
    assert v[0] == pytest.approx(0.2) and v[-1] == 0.0
    assert int((v == 0.0).sum()) == 1


def test_the_chain_record_counts_the_decimated_path(chained):
    (entry,) = chained.written["objects"]
    assert entry["points"] == 25009 < chained.cmd_v.size
