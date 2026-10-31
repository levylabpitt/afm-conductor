"""Characterisation (slice 0.7): a chained scan that engages, as a Cypher.

`test_hw_scan_chained` fakes `is_cypher = False`; the goldens freeze the commands of a scan at
the current engagement. Here `startScan` with `engage: true` runs on the Cypher stub from idle,
in contact and in AC, and the tests pin the results, flags and records the goldens do not see.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from char_rig import CharRig, acquire

pytest_plugins = ["char_rig"]    # the module-scope `hub` fixture

AC_CROSSPOINT = {"InFast": "ACDefl"}


def _engaging_scan(hub, mode: str, crosspoint=None) -> SimpleNamespace:
    with CharRig(hub, safety_sim=False) as rig:
        door = rig.door
        if crosspoint:
            rig.inst.set_crosspoint(crosspoint)
        tok = acquire(door)
        state_before = door.call("getState")
        run = door.call("startScan", {
            "token": tok, "size_um": 1.0, "pixels": 32, "lines": 2, "line_rate_hz": 50.0,
            "x_offset_um": 0.0, "y_offset_um": 0.0, "angle_deg": 0.0, "direction": "up",
            "trace_retrace": True, "mode": mode, "engage": True, "setpoint": 0.4})
        state_after = door.wait_state("idle", "engaged", "parked")
        result = door.call("getScanResult", {"run_id": run["run_id"]})
        with open(result["files"]["json"], encoding="utf-8") as f:
            sidecar = json.load(f)
        arrays = np.load(result["files"]["npz"])
        rig.quiesce()
        return SimpleNamespace(
            run=run, result=result, sidecar=sidecar, channels=arrays.files,
            state_before=state_before, state_after=state_after,
            mode=door.call("getMode"), loops=list(rig.stub.loops),
            z_loop=dict(hub.host.z_loop), free_air=hub.host.free_air,
            scanchains=list(rig.stub.scanchains))


@pytest.fixture(scope="module")
def contact(hub):
    return _engaging_scan(hub, "contact")


@pytest.fixture(scope="module")
def ac(hub):
    return _engaging_scan(hub, "ac", AC_CROSSPOINT)


def test_the_run_starts_chained_and_engaging(contact):
    cfg = contact.run["config"]
    assert (cfg["chained"], cfg["engage"], cfg["setpoint"]) == (True, True, 0.4)
    assert contact.run["backend"] == "hardware"
    assert contact.state_before == "idle"


def test_contact_engages_on_height_with_the_deflection_loop(contact):
    """The Z loop armed for the scan (the one before the end-of-scan withdraw): Deflection,
    positive gain, output Height (the Cypher's Z drive; Output.Z on the MFP-3D)."""
    assert contact.loops[-2] == (2, "Deflection", 0.4, 0.0, 1200.0, "Height")


def test_ac_engages_on_amplitude_with_the_gain_negated(ac):
    assert ac.loops[-2] == (2, "Amplitude", 0.4, 0.0, -1200.0, "Height")


def test_before_engaging_the_scan_withdraws_on_the_deflection_loop_in_either_mode(
        contact, ac):
    """The scan's first loop is the unreachable-setpoint withdraw, on Deflection even in AC."""
    withdraw = (2, "Deflection", -10.0, 0.0, 1000.0, "Height")
    assert contact.loops[0] == ac.loops[0] == withdraw


def test_the_scan_ends_with_the_routine_withdraw(contact, ac):
    withdraw = (2, "Deflection", -10.0, 0.0, 1000.0, "Height")
    assert contact.loops[-1] == ac.loops[-1] == withdraw


def test_the_tip_is_withdrawn_after_the_scan(contact, ac):
    for s in (contact, ac):
        assert s.state_after == "idle"
        assert s.mode["engaged"] is False and s.mode["backend"] == "hardware"


def test_internal_pin_the_scan_engage_records_no_z_loop_or_free_air_baseline(contact):
    """INTERNAL PIN (0.7-5): no door exposes `hub.z_loop` or `hub.free_air`, so this reads
    private hub state. Today a scan's engage leaves both empty."""
    assert contact.z_loop == {}
    assert contact.free_air is None


def test_the_result_record_in_contact(contact):
    r = contact.result
    assert (r["completed"], r["aborted"], r["partial"]) == (True, False, False)
    assert r["shape"] == [2, 32] and r["error"] is None
    assert r["channels"] == ["ZSensor", "Deflection"]
    assert r["stats"] == {"lines": 2, "pixels": 32, "size_um": 1.0, "mode": "contact"}


def test_the_result_record_in_ac_swaps_the_error_channel(ac):
    r = ac.result
    assert (r["completed"], r["aborted"]) == (True, False)
    assert r["channels"] == ["ZSensor", "Amplitude"]
    assert r["stats"]["mode"] == "ac"


def test_the_sidecar_names_the_mode_and_channels(contact, ac):
    assert contact.sidecar["mode"] == "contact"
    assert contact.sidecar["channels"] == ["deflection", "height"]
    assert ac.sidecar["mode"] == "ac"
    assert ac.sidecar["channels"] == ["amplitude", "height"]
    assert contact.sidecar["rows"] == 2 and contact.sidecar["pixels"] == 32


def test_one_scanchain_is_sent_per_scan(contact, ac):
    assert len(contact.scanchains) == len(ac.scanchains) == 1
