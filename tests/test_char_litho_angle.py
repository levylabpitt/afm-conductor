"""Characterisation (slice 0.7): SLICES F15, a hardware write ignores the hardware scan angle.

A chained hardware scan at 30 degrees, then a write of the golden `litho_single` wire. The
service reads the litho frame's angle from the twin's scan, which a hardware scan never sets,
so the wire is written unrotated. The reference is the frozen golden (0 degrees, same wire).
"""

from __future__ import annotations

import json
import re

import pytest
from char_rig import WIRE, CharRig, acquire, start_write, waves_sent
from afm_litho.litho.frame import rotate
from golden.harness import HERE

pytest_plugins = ["char_rig"]    # the module-scope `hub` fixture

SCAN_ANGLE_DEG = 30.0


def _golden_wave(name: str) -> dict:
    """The XY wave named `name` in the frozen litho_single golden (0 degrees): n, first, last,
    sha."""
    golden = json.loads((HERE / "litho_single.golden.json").read_text(encoding="utf-8"))
    for line in golden["wire"]["actuation"]:
        m = re.match(rf"WAVE wave\({name},n=(\d+),first=(\S+),last=(\S+),sha=(\w+)\)", line)
        if m:
            return {"n": int(m[1]), "first": float(m[2]), "last": float(m[3]), "sha": m[4]}
    raise AssertionError(f"no {name} wave in the golden")


def _golden_sha(name: str) -> str:
    return _golden_wave(name)["sha"]


@pytest.fixture(scope="module")
def after_rotated_scan(hub):
    with CharRig(hub, safety_sim=False) as rig:
        door = rig.door
        tok = acquire(door)
        scan = door.call("startScan", {
            "token": tok, "size_um": 1.0, "pixels": 32, "lines": 2, "line_rate_hz": 50.0,
            "x_offset_um": 0.0, "y_offset_um": 0.0, "angle_deg": SCAN_ANGLE_DEG,
            "direction": "up", "trace_retrace": True, "mode": "contact"})
        door.wait_state("idle", "engaged", "parked")
        angle_seen = scan["config"]["angle_deg"]
        start_write(door, tok, [WIRE])
        door.wait_state("idle", "engaged", "parked")
        rig.quiesce()
        return {"angle": angle_seen, "xw": waves_sent(rig, "XLitho")[0],
                "yw": waves_sent(rig, "YLitho")[0]}


def test_the_scan_was_taken_at_the_angle(after_rotated_scan):
    assert after_rotated_scan["angle"] == SCAN_ANGLE_DEG


def test_today_the_wire_is_written_unrotated_after_a_rotated_scan(after_rotated_scan):
    assert after_rotated_scan["xw"]["sha"] == _golden_sha("XLitho")
    assert after_rotated_scan["yw"]["sha"] == _golden_sha("YLitho")


@pytest.mark.xfail(strict=True, reason="SLICES F15: _litho_frame() takes the angle from the "
                   "twin's scan, so a hardware scan's angle never reaches the written path")
def test_the_wire_is_rotated_by_the_scan_angle(after_rotated_scan):
    # The wire's two ends, rotated by the scan angle with the litho frame's own rotate()
    # (rigid and linear, so the volts rotate like the microns they stand for).
    x0, y0 = _golden_wave("XLitho"), _golden_wave("YLitho")
    want_first = rotate(x0["first"], y0["first"], SCAN_ANGLE_DEG)
    want_last = rotate(x0["last"], y0["last"], SCAN_ANGLE_DEG)
    xw, yw = after_rotated_scan["xw"], after_rotated_scan["yw"]
    assert (xw["n"], yw["n"]) == (x0["n"], y0["n"])
    assert (xw["first"], yw["first"]) == pytest.approx(want_first, abs=1e-6)
    assert (xw["last"], yw["last"]) == pytest.approx(want_last, abs=1e-6)
