"""Start verbs refused for two reasons at once: which one is named (H5.pre).

Each case makes two refusal conditions true together and pins the one the code names today.
A later slice that lifts a check out of the start body (H5c, H5d) must keep these answers, so
a lifted check cannot jump ahead of one left behind. Driven through the FLEX door on the Igor
stub; codes are the JSON-RPC error codes.
"""

from __future__ import annotations

import pytest
from golden.harness import GoldenRig
from golden.scenarios import AC_CROSSPOINT, WIRE, _acquire, _release

from afm_litho.service import keeper_link

SCAN = {"size_um": 1.0, "pixels": 32, "lines": 2, "line_rate_hz": 50.0, "x_offset_um": 0.0,
        "y_offset_um": 0.0, "angle_deg": 0.0, "direction": "up", "trace_retrace": True,
        "mode": "contact"}
WRITE = {"objects": "all", "setpoint": 0.0, "amp_gain": 25.0}
FAULT = {"reason": "double-refusal test"}
INVALID_PARAMS, REFUSED, FAULT_LATCHED = -32602, -32021, -32050


@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


def _named(err: dict) -> str:
    data = err.get("data") or {}
    return data.get("reason") or data.get("param") or ""


# (case, verb, params, conditions that are also true, (code, named reason or param))
CASES = [
    ("scan: setpoint and pgain inert", "startScan", {**SCAN, "setpoint": 0.3, "pgain": 1.0},
     (), (INVALID_PARAMS, "setpoint")),
    ("scan: pgain and igain inert", "startScan", {**SCAN, "pgain": 1.0, "igain": 2.0},
     (), (INVALID_PARAMS, "pgain")),
    ("scan: inert setpoint and a latched fault", "startScan", {**SCAN, "setpoint": 0.3},
     ("fault",), (INVALID_PARAMS, "setpoint")),
    ("scan: inert setpoint and a frame off the scanner", "startScan",
     {**SCAN, "size_um": 5000.0, "setpoint": 0.3}, (), (INVALID_PARAMS, "setpoint")),
    ("scan: a latched fault and a frame off the scanner", "startScan",
     {**SCAN, "size_um": 5000.0}, ("fault",), (FAULT_LATCHED, "fault_latched")),
    ("write: keeper lost and the instrument in AC", "startWrite", dict(WRITE),
     ("keeper", "ac"), (REFUSED, "keeper_lost")),
    ("write: the instrument in AC and a latched fault", "startWrite", dict(WRITE),
     ("ac", "fault"), (REFUSED, "ac_mode_active")),
    ("write: the instrument in AC and a wrong amp gain", "startWrite", {**WRITE, "amp_gain": 3.0},
     ("ac",), (REFUSED, "ac_mode_active")),
    ("write: the instrument in AC and an unknown object", "startWrite",
     {**WRITE, "objects": ["nope"]}, ("ac",), (INVALID_PARAMS, "objects")),
    ("write: a latched fault and a wrong amp gain", "startWrite", {**WRITE, "amp_gain": 3.0},
     ("fault",), (REFUSED, "aborted_during_start")),
    ("write: a latched fault and an unknown object", "startWrite",
     {**WRITE, "objects": ["nope"]}, ("fault",), (INVALID_PARAMS, "objects")),
    ("write: a wrong amp gain and an unknown object", "startWrite",
     {**WRITE, "amp_gain": 3.0, "objects": ["nope"]}, (), (INVALID_PARAMS, "objects")),
]


def test_the_refusal_named_when_two_conditions_hold(hub):
    got = {}
    with GoldenRig(hub, "double-refusals") as rig:
        door = rig.door
        tok = _acquire(door)
        door.call("loadPattern", {"token": tok, "objects": [dict(WIRE)]})
        try:
            for case, verb, params, conditions, _want in CASES:
                hub.fault = FAULT if "fault" in conditions else None
                keeper_link._keeper_lost = "keeper gone (test)" if "keeper" in conditions else None
                rig.inst.set_crosspoint(AC_CROSSPOINT if "ac" in conditions else {})
                err = door.call_refused(verb, {"token": tok, **params})
                got[case] = (err["code"], _named(err))
        finally:
            hub.fault = None
            keeper_link._keeper_lost = None
            rig.inst.set_crosspoint({})
            _release(door, tok)
    assert got == {case: want for case, _v, _p, _c, want in CASES}


def test_a_scan_already_running_is_named_before_a_second_condition(hub):
    """With a scan in flight, a second start that is also inert is named inert; one that also
    meets a latched fault is named busy (its reply carries no reason or param)."""
    got = {}
    with GoldenRig(hub, "double-refusals-busy") as rig:
        door, inst = rig.door, rig.inst
        tok = _acquire(door)
        inst.hold_scan_after_line(1)
        door.call("startScan", {"token": tok, **SCAN, "lines": 4})
        inst.wait_held()
        door.wait_scan_row(1)
        try:
            got["busy"] = _named(door.call_refused("startScan", {"token": tok, **SCAN}))
            got["busy+inert"] = _named(door.call_refused(
                "startScan", {"token": tok, **SCAN, "setpoint": 0.3}))
            hub.fault = FAULT
            got["busy+fault"] = _named(door.call_refused("startScan", {"token": tok, **SCAN}))
        finally:
            hub.fault = None
            door.call("abort", {"scope": "scan"})
            inst.release_hold()
            door.wait_state("idle", "engaged", "parked")
            _release(door, tok)
    assert got == {"busy": "", "busy+inert": "setpoint", "busy+fault": ""}, got
