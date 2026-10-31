"""Wire tap on FakeIgorBridge: exact actuation transcripts and their determinism."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from afm_litho.goldens.wire_tap import TappedCom, WireTap
from afm_litho.hardware.hardware_arc import HardwareARC
from afm_litho.hardware.igor_stub import FakeIgorBridge
from afm_litho.hardware.lithography_run import LithographyRun
from afm_litho.hardware.hw_approach import hw_engage, hw_withdraw

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "SA40656B0002.ibw"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample .ibw not present")


def _compact(tap):
    return [[e["verb"], e["args"]] for e in tap.actuation]


def _cypher_engage_withdraw():
    stub = FakeIgorBridge({"Deflection": 1.5, "height": 20.0}, done_delay=0.0)
    port = stub.serve_in_thread()
    hw = HardwareARC.connect("127.0.0.1", port)
    hw.startup = {"microscope_id": 4}       # cMicroscopeCypher
    try:
        hw_engage(hw, setpoint=1.0, pgain=0.0, igain=10.0, settle_s=0.0)
        hw_withdraw(hw, max_seconds=0.0, settle_margin=0.0)
    finally:
        hw.close()
        stub.stop()
    return stub.tap


def _armfire_single_object():
    from afm_litho.io.ibw import read_ibw
    note = read_ibw(SAMPLE).note_dict
    stub = FakeIgorBridge.from_ibw(SAMPLE, done_delay=0.0)
    port = stub.serve_in_thread()
    stub.values["XSensor"] = float(note["XLVDTOffset"])
    stub.values["YSensor"] = float(note["YLVDTOffset"])
    try:
        run = LithographyRun.connect(
            "127.0.0.1", port, settle_move_s=0.0, cmd_gap_s=0.0,
            xy_withdraw_max_s=0.0, xy_withdraw_settle_s=0.0, first_object_settle_s=0.0,
            amp_gain=10.0, retract=-10.0, decimation=50)
        run.start()
        run.write_object("obj0", [0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [3.0, 3.0, 3.0],
                         setpoint=0.4, speed=5.0)
        run.get_value("XSensor")        # barrier: the post-DONE lift has been processed
    finally:
        stub.stop()
    return stub.tap


def test_cypher_engage_then_withdraw_exact_actuation():
    tap = _cypher_engage_withdraw()
    assert _compact(tap) == [
        ["SERVOLOOP", ["2", "Deflection", "1", "0", "1000", "Height", "-10", "150"]],
        ["SERVOLOOP", ["2", "Deflection", "-10", "0", "1000", "Height", "0", "150"]],
    ]
    # the engage/withdraw polls are reads, kept out of the actuation list
    assert {e["verb"] for e in tap.reads} == {"GETVALUE"}
    assert [e["verb"] for e in tap.combined if e["kind"] != "read"] == ["SERVOLOOP"] * 2


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample .ibw not present")
def test_armfire_single_object_exact_actuation():
    tap = _armfire_single_object()
    assert _compact(tap) == [
        ["SERVOLOOP", ["2", "Deflection", "-10", "0", "1000", "Height", "0", "150"]],
        ["SERVOLOOP", ["0", "XSensor", "-0.63", "0", "0", "Output.X", "-10", "150"]],
        ["SERVOLOOP", ["1", "YSensor", "0.07", "0", "0", "Output.Y", "-10", "150"]],
        ["WAVE", [["XYTransitA", 2, "-0.629999995", "-0.629999995", "33af8e3b9f37d169"]]],
        ["WAVE", [["XYTransitB", 2, "0.0700000003", "0.0700000003", "9a1682a4aed99e75"]]],
        ["OUTPAIR", ["0", "0", "PIDSLoop.0.Setpoint", "XYTransitA",
                     "PIDSLoop.1.Setpoint", "XYTransitB", "-1"]],
        ["INPAIR", ["0", "0", "XSensor", "YSensor", "32", "1"]],
        ["FIRE", ["a1", "XYTransit", "0"]],
        ["@FILL", []],
        ["SERVOLOOP", ["2", "Deflection", "0.4", "0", "1200", "Height"]],
        ["WAVE", [["XLitho", 32, "-0.629999995", "-0.362057924", "ee126089a8221466"]]],
        ["WAVE", [["YLitho", 32, "0.0700000003", "0.0700000003", "ac7791b36f57b328"]]],
        ["WAVE", [["VLitho", 32, "0.300000012", "0", "22c4834b878029ab"]]],
        ["ARMFIRE", ["a2", "obj0", "-10000", "32", "10000"]],
        ["@FILL", []],
        ["SERVOLOOP", ["2", "Deflection", "-10", "0", "1200", "Height"]],
    ]
    assert tap.reads[0]["verb"] == "STARTUP"
    assert {e["verb"] for e in tap.reads} == {"STARTUP", "GETVALUE"}


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample .ibw not present")
def test_actuation_bytes_are_identical_over_20_runs():
    cy = {_cypher_engage_withdraw().actuation_json() for _ in range(20)}
    assert len(cy) == 1
    af = {_armfire_single_object().actuation_json() for _ in range(20)}
    assert len(af) == 1


def test_wave_summary_matches_whole_wave_across_appends():
    vals = [0.1, -2.5, 3.0, float("nan"), 7.123456789012345]
    txt = [f"{v:.12g}" for v in vals]
    whole, split = WireTap(), WireTap()
    whole.record("WAVE w " + ";".join(txt))
    split.record("WAVE w " + ";".join(txt[:2]))
    split.record("WAVEAPP w " + ";".join(txt[2:]))
    f32 = np.array(vals, dtype="<f4")
    expect = ["w", 5, f"{f32[0]:.9g}", f"{f32[-1]:.9g}",
              hashlib.sha256(f32.tobytes()).hexdigest()[:16]]
    assert whole.actuation[0]["args"] == [expect]
    assert split.actuation[1]["args"] == [expect]
    assert split.actuation[0]["args"][0][1] == 2          # length so far
    # a fresh WAVE with the same name restarts the count
    split.record("WAVE w 1;2")
    assert split.actuation[2]["args"][0][1] == 2


def test_uids_are_labelled_per_list_and_reads_are_separate():
    t = WireTap()
    t.record("GETVALUE 7 XSensor")
    t.record("FIRE 8 m 0")
    t.record("GETVALUE 9 YSensor")
    t.record("ARMFIRE 10 o -1 32 1")
    t.record("SHOWTIP ON")
    assert [e["args"][0] for e in t.reads] == ["r1", "r2"]
    assert [e["args"][0] for e in t.actuation[:2]] == ["a1", "a2"]
    assert t.actuation[2]["verb"] == "SHOWTIP"
    assert [e["seq"] for e in t.combined] == [0, 1, 2, 3, 4]


def test_unknown_verb_counts_as_actuation():
    t = WireTap()
    t.record("BOGUS 1 2")
    assert t.actuation[0]["verb"] == "BOGUS" and not t.reads


def test_external_calls_are_recorded_through_tapped_com():
    class Com:
        def execute(self, cmd, check=False):
            return 0

    t = WireTap()
    TappedCom(Com(), t).execute("AFM_SafePark()", check=True)
    assert t.actuation == [{"seq": 0, "kind": "ext", "verb": "@SAFEPARK",
                            "args": [], "source": "com"}]


def test_to_json_is_sorted_compact_and_parses():
    t = WireTap()
    t.record("SETVALUE a 1")
    t.record("GETVALUE 1 a")
    b = t.to_json()
    d = json.loads(b)
    assert set(d) == {"format", "actuation", "reads", "combined"}
    assert b == json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
    assert "combined" not in json.loads(t.to_json(combined=False))
