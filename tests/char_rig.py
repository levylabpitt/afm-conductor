"""Helpers for the slice 0.7 characterisation tests (tests/test_char_*.py).

A Cypher on the Igor stub, driven through FLEX (and HTTP where FLEX has no verb), the same
doors the goldens use. The stub here runs the fake-Igor safety simulation, so bias, XY and Z
can be read back as the controller would leave them.
"""

from __future__ import annotations

import json
import time

import pytest
from golden.harness import GoldenRig, GoldenStub, Inst

CLIENT = "characterisation"


@pytest.fixture(scope="module")
def hub(client):
    """The service's hub, for tests that need the live instance behind the in-process client.
    Test modules import it from here (`from char_rig import hub`)."""
    from afm_litho.service.hub import hub as real_hub
    # The module's own scans (some in AC) run in module-scoped fixtures, outside the
    # per-test restore in conftest; leave the scan config as the module found it.
    last, repeat = real_hub.host.last_scan_config, real_hub.host.scan_repeat.get("config")
    yield real_hub
    real_hub.host.last_scan_config = last
    real_hub.host.scan_repeat["config"] = repeat

WIRE = {"kind": "chiral", "name": "wire", "points_um": [[0.0, 0.0], [0.3, 0.0]],
        "lambda_um": 0.1, "y_amp_um": 0.0, "v0": 5.0, "v_k": 0.0, "phase_deg": 0.0,
        "hand": 1.0, "speed_um_s": 1.0}
WIRE_B = {**WIRE, "name": "wire_b", "points_um": [[0.5, 0.2], [0.8, 0.2]], "v0": 3.0}
# About 100,000 points: over the chunk size, so it is fired as a two-chunk chain.
WIRE_LONG = {**WIRE, "name": "wire_long", "points_um": [[0.0, 0.0], [10.0, 0.0]],
             "lambda_um": 0.0008}


class CharRig(GoldenRig):
    """`GoldenRig` with the stub's safety simulation on, following the wall clock so the
    bias and Z the stub reports move while the service sleeps."""

    def __init__(self, hub, *, telem_hz: float = 0.0, safety_sim: bool = True):
        super().__init__(hub, "characterisation", telem_hz=telem_hz)
        crosspoint = self.stub.crosspoint
        self.stub = GoldenStub(self.values, microscope_id=self.microscope_id,
                               done_delay=0.0, telem_hz=telem_hz, safety_sim=safety_sim,
                               safety_sim_real_time=safety_sim)
        self.stub.crosspoint = crosspoint
        self.tap = self.stub.tap
        self.inst = Inst(self.stub)

    def verbs(self) -> list[str]:
        """The wire verbs the stub received, in order."""
        return list(self.stub.transcript)

    def count(self, verb: str) -> int:
        return sum(1 for v in self.stub.transcript if v == verb)


def acquire(door, **kw) -> str:
    return door.call("acquireControl", {"client": CLIENT, "deadman_s": 120, **kw})["token"]


def start_write(door, tok, objects, **kw) -> dict:
    door.call("loadPattern", {"token": tok, "objects": [dict(o) for o in objects]})
    body = {"token": tok, "objects": "all", "setpoint": 0.0, "amp_gain": 25.0, **kw}
    return door.call("startWrite", body)


def wait_until(cond, timeout: float = 30.0, what: str = "condition") -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


def sidecar(result: dict) -> dict:
    """The json sidecar of a finished write. `getWriteResult.files` does not list it (SLICES
    0.7 finding), so it is found beside the npz, in the sibling `json` folder."""
    path = result["files"]["npz"].replace("/npz/", "/json/").replace(".npz", ".json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def waves_sent(rig, name: str) -> list[dict]:
    """Every WAVE the stub received under `name`, in order, as {n, first, last, sha}."""
    out = []
    for entry in rig.tap.actuation:
        if entry["verb"] in ("WAVE", "WAVEAPP"):
            for wave in entry["args"]:
                if wave[0] == name:
                    out.append({"n": wave[1], "first": float(wave[2]), "last": float(wave[3]),
                                "sha": wave[4]})
    return out
