"""Slice 3.1a: the write's run, built from the write goldens' own inputs, predicts the wire.

The golden scenarios (tests/golden/scenarios.py) give the objects; the goldens record what the
write sent. For each, the run's fire verb, chunk count and cut, wave-line size and the order of
move / withdraw / engage steps are checked against the recorded wire. The step order is read off
the wire with the one-letter tokens below; the full (run, wire) check is tests/test_run_wire_check.py.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from datetime import datetime, timezone
from math import ceil

import numpy as np
import pytest
from golden.harness import HERE
from golden.scenarios import PAD, WIRE, WIRE_B, WIRE_LONG

from afm_litho.goldens.expander import summarise
from afm_litho.hardware.lithography_run import (
    WriteSetpoint,
    as_write_setpoint,
    dac_volts,
    fired_point_count,
    is_hover_move,
    lvdt_volts,
)
from afm_litho.litho.chunker import CHAIN_CHUNK_POINTS, is_chained
from afm_litho.litho.compiler import compile_pattern
from afm_litho.run import Design, Engage, LithoObject, Move, Placement, Withdraw
from afm_litho.run.build_write import (
    UnsupportedWrite,
    build_write_run,
    fired_points,
    points_fingerprint,
)
from afm_litho.service.host.litho_cmds import _objects_pattern
from wire_facts import golden_actuation, loop2_lines, wave_lines, wire_steps

STARTED = datetime(2026, 10, 24, 18, 2, tzinfo=timezone.utc)
HEX = "5e0a1b2c" * 8


def compile_scenario(objects):
    """What the write worker has in hand: the pattern's objects and its compiled strokes."""
    pattern, _ = _objects_pattern([dict(o) for o in objects])
    waves = compile_pattern(pattern, 1e-5, 1e-5, 0.0, 0.0)
    return pattern.objects, waves["segments"]


def build(objects, **kw):
    objs, segments = compile_scenario(objects)
    args = dict(
        segments=segments, objects=objs, setpoint=WriteSetpoint(0.0, relative=False),
        amp_gain=25.0, platform="cypher", started=STARTED,
        placement=Placement("scan 13:58", 30, 0, (0, 0)),
        design=Design("golden", 1, len(objs), HEX, HEX, "afm-litho test"),
        bias_window_v=(-8, 30))
    return build_write_run(**{**args, **kw})


def run_steps(run):
    """The same letters, predicted from the run's body."""
    steps = []
    for e in run.recipe.body:
        if isinstance(e, Move):
            steps += ["withdraw", "transit"] if e.withdraw_first == "verified" else ["hover"]
        elif isinstance(e, Engage):
            steps.append("engage")
        elif isinstance(e, LithoObject):
            steps.append(e.fire)
        elif isinstance(e, Withdraw):
            steps.append("back-off" if e.how == "back-off" else "withdraw")
    return steps


def passes(run):
    return [e for e in run.recipe.body if isinstance(e, LithoObject)]


def test_single_wire_run_predicts_the_golden():
    run = build([WIRE])
    (p,) = passes(run)
    assert wire_steps("litho_single") == run_steps(run) == [
        "withdraw", "transit", "engage", "ARMFIRE", "back-off", "withdraw"]
    assert p.fire == "ARMFIRE" and p.chunks == 1 and p.chunk_cuts == ()
    lines = wave_lines("litho_single", "XLitho")
    assert len(lines) == 1 and int(lines[0].group(2)) == p.points == 32
    assert p.wave_line_points == 8192


def test_two_chunk_chain_run_predicts_the_golden():
    run = build([WIRE_LONG])
    (p,) = passes(run)
    assert wire_steps("litho_chain_two_chunks") == run_steps(run) == [
        "withdraw", "transit", "engage", "LITHOCHAIN", "back-off", "withdraw"]
    lines = wave_lines("litho_chain_two_chunks", "XLitho")
    assert int(lines[-1].group(2)) == p.points == 100032
    assert int(lines[0].group(2)) == p.wave_line_points == 8192
    assert len(lines) == ceil(p.points / p.wave_line_points)
    (bounds,) = [line for line in golden_actuation("litho_chain_two_chunks")
                 if line.startswith("WAVE wave(LithoBounds,")]
    recorded = summarise("LithoBounds", np.array([0, *p.chunk_cuts, p.points], dtype="<f4"))
    assert f"n={p.chunks + 1}," in bounds and f"sha={recorded.sha}" in bounds
    assert p.chunks == 2 and p.chunk_cuts == (CHAIN_CHUNK_POINTS,)


def test_two_wires_move_by_transit_with_a_verified_withdraw_each():
    run = build([WIRE, WIRE_B])
    assert [p.index for p in passes(run)] == [0, 1]
    assert run_steps(run) == [
        "withdraw", "transit", "engage", "ARMFIRE", "back-off",
        "withdraw", "transit", "engage", "ARMFIRE", "back-off", "withdraw"]
    # The golden of the same shape, up to the abort, starts the same way.
    assert wire_steps("abort_in_transit")[:5] == run_steps(run)[:5]


def test_multi_object_golden_moves_follow_the_hover_rule():
    """The pad is a fill, which the contract cannot state (one fire per object): the builder
    refuses it. The rule it would use, `is_hover_move`, predicts the golden's moves."""
    objs, segments = compile_scenario([PAD, WIRE])
    with pytest.raises(UnsupportedWrite, match="several strokes"):
        build_write_run(segments=segments, objects=objs, setpoint=0.0, amp_gain=25.0,
                        platform="cypher", started=STARTED,
                        placement=Placement("scan", 30, 0, (0, 0)),
                        design=Design("g", 1, 2, HEX, HEX, "t"), bias_window_v=(-8, 30))
    moves, last = [], None
    for i, meta in enumerate(segments):
        moves.append("hover" if is_hover_move(i == 0, meta["object"], last) else "transit")
        last = meta["object"]
    recorded = [s for s in wire_steps("litho_multi_object") if s in ("transit", "hover")]
    assert moves == recorded == ["transit", "hover", "transit"]


def test_the_builder_applies_the_write_rule_functions():
    assert not is_chained(CHAIN_CHUNK_POINTS) and is_chained(CHAIN_CHUNK_POINTS + 1)
    assert is_hover_move(False, "pad", "pad") and not is_hover_move(True, "pad", "pad")
    assert not is_hover_move(False, "wire", "pad") and not is_hover_move(False, None, None)
    assert fired_point_count(32, 0.0) == 32
    assert fired_point_count(32, 5.0) == 64 and fired_point_count(33, 0.0) == 64
    assert as_write_setpoint(0.2) == WriteSetpoint(0.2, relative=False)
    rel = WriteSetpoint(0.05, relative=True)
    assert as_write_setpoint(rel) is rel


def test_relative_setpoint_and_platform_reach_every_engage():
    run = build([WIRE, WIRE_B], setpoint=WriteSetpoint(0.05, relative=True), platform="mfp3d")
    assert run.recipe.setpoint.reference == "relative" and run.recipe.setpoint.volts == 0.05
    engages = [e for e in run.recipe.body if isinstance(e, Engage)]
    assert len(engages) == 2 and {e.out for e in engages} == {"Height"}
    assert {e.limits_v for e in engages} == {(-10.0, 150.0)}


def test_header_and_execution():
    run = build([WIRE])
    r = run.recipe
    assert (r.kind, r.mode, r.script_gone) == ("write", "contact", "park")
    assert r.bias.amp_gain == 25.0 and r.capture.pair_bank == 2
    assert run.execution.execution_id == "w-2026-10-24-1802-01"
    assert build([WIRE], taken_ids=["w-2026-10-24-1802-01"]).execution.execution_id.endswith("-02")


def test_first_object_settle_is_one_step_and_zero_removes_it():
    from afm_litho.run import Settle
    settles = [e for e in build([WIRE, WIRE_B]).recipe.body if isinstance(e, Settle)]
    assert [s.seconds for s in settles] == [3.0]
    assert not [e for e in build([WIRE], first_object_settle_s=0).recipe.body
                if isinstance(e, Settle)]


def test_points_fingerprint_follows_the_points():
    a = points_fingerprint([0.0, 1.0], [0.0, 0.0], [5.0, 0.0])
    assert a == points_fingerprint([0.0, 1.0], [0.0, 0.0], [5.0, 0.0])
    assert a != points_fingerprint([0.0, 1.0], [0.0, 0.0], [5.0, 0.1])
    assert build([WIRE]).recipe.body == build([WIRE]).recipe.body


# --- the numbers the run states, against the lines the golden recorded ---------------------


@pytest.mark.parametrize("name,objects", [("litho_single", [WIRE]),
                                          ("litho_chain_two_chunks", [WIRE_LONG])])
def test_setpoints_and_holds_match_the_golden_lines(name, objects):
    run = build(objects)
    body = run.recipe.body
    (engage,) = [e for e in body if isinstance(e, Engage)]
    (back_off,) = [e for e in body if isinstance(e, Withdraw) and e.how == "back-off"]
    (verified,) = [e for e in body if isinstance(e, Withdraw) and e.how == "verified"]
    roles = [(role, lo, hi) for role, _sp, lo, hi in loop2_lines(name)]
    setpoints = {role: sp for role, sp, _lo, _hi in loop2_lines(name) if role != "withdraw"}
    assert roles == [("withdraw", 0.0, 150.0), ("engage", None, None), ("back-off", None, None),
                     ("withdraw", verified.hold_v, engage.limits_v[1])]
    assert setpoints == {"engage": engage.setpoint.volts, "back-off": back_off.setpoint.volts}
    assert back_off.hold_v == back_off.setpoint.volts == -10.0
    assert [move.hold_v for move in body if isinstance(move, Move)] == [verified.hold_v]


CAL = json.loads((HERE / "fixture" / "igor_values.json").read_text())["values"]


@pytest.mark.parametrize("name,objects", [("litho_single", [WIRE]),
                                          ("litho_chain_two_chunks", [WIRE_LONG])])
def test_fired_points_convert_to_the_golden_wave_hashes(name, objects):
    """The run's fingerprint covers the fired um points; through the write's own conversion
    with the golden's calibration and amp gain, those points are the recorded waves."""
    _, segments = compile_scenario(objects)
    (meta,) = segments
    x, y, v = fired_points(meta)
    waves = {"XLitho": lvdt_volts(x, CAL["XLVDTSens"], CAL["XLVDTOffset"]),
             "YLitho": lvdt_volts(y, CAL["YLVDTSens"], CAL["YLVDTOffset"]),
             "VLitho": dac_volts(v, 25.0)}
    for wave, values in waves.items():
        last = wave_lines(name, wave)[-1].string
        # The wire carries 12 significant digits; the stub reads them back as float32.
        sent = [float(f"{u:.12g}") for u in values]
        recorded = summarise(wave, np.asarray(sent, dtype="<f4"))
        assert f"n={recorded.n}," in last and f"sha={recorded.sha}" in last, (wave, last)
    assert passes(build(objects))[0].points_fingerprint == points_fingerprint(x, y, v)


def test_a_stroke_off_the_grid_is_padded_the_way_the_write_fires_it():
    n = 33
    meta = {"name": "odd", "object": "odd",
            "path_um": np.column_stack([np.linspace(0, 1, n), np.zeros(n)]),
            "v": np.full(n, 5.0)}
    obj = SimpleNamespace(name="odd", kind="path", speed_um_s=1.0)
    run = build_write_run(
        segments=[meta], objects=[obj], setpoint=0.0, amp_gain=25.0, platform="cypher",
        started=STARTED, placement=Placement("scan", 30, 0, (0, 0)),
        design=Design("g", 1, 1, HEX, HEX, "t"), bias_window_v=(-8, 30))
    (p,) = passes(run)
    x, y, v = fired_points(meta)
    assert p.points == len(x) == 64 and v[32] == 5.0 and not v[33:].any()
    assert p.points_fingerprint == points_fingerprint(x, y, v)
