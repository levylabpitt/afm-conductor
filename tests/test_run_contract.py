"""Run contract v1 (slice 3.0): types, JSON round trip, fingerprint, execution ids, text form.

The two runs built here are the agreed mockups in docs/migration/plan/phase-3-run.md (a
three-object write and an AC scan), by hand.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from afm_litho.run import (Bias, Capture, Check, Design, End, Engage, Execution, LithoObject,
                           Move, Param, Placement, Read, Recipe, Run, Scan, ScanFrame, SetMode,
                           Setpoint, Settle, StepFile, Withdraw, next_execution_id,
                           recipe_fingerprint, run_from_json, run_text, run_to_json)

HEX_A = "5e0a1b2c" * 8
HEX_B = "9d4f7a10" * 8
STARTED = "2026-10-24T14:02:11.250-04:00"  # 18:02 UTC, the minute in the execution ids
EDT = timezone(timedelta(hours=-4))


def write_recipe() -> Recipe:
    relative = Setpoint("relative", 0.05)
    engage = Engage(relative, loop=2, on="Deflection", out="Height", limits_v=(-10, 150),
                    settle_s=0.3)
    back_off = Withdraw("back-off", hold_v=-10, setpoint=Setpoint("absolute", -10), settle_s=0.3)
    free_air = Read(("Deflection",))
    return Recipe(
        kind="write", platform="cypher", mode="contact", setpoint=relative,
        placement=Placement("scan 13:58", 30, 0, (12.4, -3.1)),
        design=Design("Device_B", 18, 3, HEX_A, HEX_B, "afm-litho 0.0.1"),
        bias=Bias(25, (-8, 30)),
        capture=Capture(("Height", "Deflection", "XSensor", "YSensor"), pair=("UserIn2", "Lateral"),
                        pair_bank=2, pair_fallbacks=(("UserIn2",), ())),
        script_gone="park", script_gone_note="write in progress",
        body=(
            Check(("sum-below-ceiling", "xy-calibration-refreshed")),
            Read(("Deflection",), agreeing=3, z_drive_max_v=1),
            Move("lifted", "start of object 0"),
            Settle(3.0, "first object only"),
            free_air,
            engage,
            LithoObject(0, "lead_chiral", "chiral wire", points=16608, chunks=1, fire="ARMFIRE",
                        ends_at_v=0, wave_line_points=8192, points_fingerprint=HEX_A,
                        params=(Param("length", 2.0, "um"), Param("wavelength", 25, "nm"),
                                Param("amplitude", 55, "nm"), Param("spacing", 1.07, "nm")),
                        speed_um_s=1.5),
            back_off,
            Move("transit", "object 1", hold_v=0),
            free_air,
            engage,
            LithoObject(1, "funnel_left", "one continuous serpentine", points=41280, chunks=1,
                        fire="ARMFIRE", ends_at_v=0, wave_line_points=8192,
                        points_fingerprint=HEX_B),
            back_off,
            Move("transit", "object 2"),
            free_air,
            engage,
            LithoObject(2, "pad_right", "zig-zag", points=121536, chunks=2, fire="LITHOCHAIN",
                        ends_at_v=0, wave_line_points=8192, points_fingerprint=HEX_A,
                        chunk_cuts=(86976,)),
            back_off,
            Withdraw("verified", hold_v=0, why="end of run"),
        ))


def scan_recipe() -> Recipe:
    absolute = Setpoint("absolute", 0.65)
    return Recipe(
        kind="scan", platform="cypher", mode="ac", setpoint=absolute,
        frame=ScanFrame("request", 30, 256, 256, 0, (0, 0), drift_offset=True, direction="up",
                        line_rate_hz=1.0, trace_and_retrace=True),
        capture=Capture(("ZSensor", "Amplitude", "UserIn2")),
        script_gone="park",
        body=(
            Check(("no-latched-fault", "frame-inside-xy-range")),
            SetMode("ac", when="only if withdrawn and the mode differs", settle_s=1.0),
            Move("lifted", "the first point", withdraw_first="verified-if-engaged", speed_um_s=5,
                 settle_s=0.5),
            Engage(absolute, loop=2, on="Amplitude", out="Height", limits_v=(-10, 150), sign=-1,
                   when="only when engage is requested"),
            Scan(lines=256, points_per_line=512, interpolation=195, decimation=195,
                 error_channel="Amplitude", xy_loops=(0, 1),
                 xy_outputs=("Output.X", "Output.Y"), limits_v=(-10, 150), wave_line_points=8192),
            End(engaged=True),
        ))


def write_run(started: str = STARTED, seq: int = 1) -> Run:
    return Run(write_recipe(), Execution(
        f"w-2026-10-24-1802-{seq:02d}", started, design_snapshot="runs/w/design.json",
        compile_settings="runs/w/compile.json",
        commanded_xy=(StepFile(6, "obj0.xy.f32"), StepFile(11, "obj1.xy.f32"),
                      StepFile(16, "obj2.xy.f32"))))


def scan_run() -> Run:
    return Run(scan_recipe(), Execution("s-2026-10-24-1815-01", "2026-10-24T14:15:00-04:00"))


# The fingerprints in these texts are pinned on purpose: if the canonical JSON changes, every
# stored run's fingerprint changes, and that must be a deliberate version step.
WRITE_TEXT = """\
run        write | recipe afcfedd9 | execution w-2026-10-24-1802-01
design     "Device_B" rev 18 | 3 objects | snapshot 5e0a1b2c
compile    settings 9d4f7a10 | compiler afm-litho 0.0.1
platform   cypher | contact mode
placement  from scan 13:58 (30 um, 0 deg, centre 12.4, -3.1 um)
setpoint   relative +0.05 V above free air
bias       amp gain 25 | tip-voltage window -8 to 30 V
capture    every pass: Height, Deflection, XSensor, YSensor, UserIn2 + Lateral
           (pair on in-bank 2; falls back to UserIn2 alone, then none)
if script goes silent   park (write in progress)
 0  check     SUM below ceiling | XY calibration refreshed
 1  read      free air: Deflection, 3 agreeing reads, Z drive < 1 V  [run time]
 2  move      lifted to start of object 0, verified withdraw first
 3  settle    3 s (first object only)
 4  read      free air: Deflection  [run time]
 5  engage    --relative --setpoint=+0.05 | loop 2 on Deflection, Height out, limits -10 to 150
              | settle 0.3 s
 6  PASS      litho object 0 "lead_chiral": chiral wire, length 2 um, wavelength 25 nm,
              amplitude 55 nm, spacing 1.07 nm, 16,608 points, 1.5 um/s, 1 chunk (ARMFIRE), ends
              at 0 V, waves loaded in 8,192-point lines, points 5e0a1b2c
 7  withdraw  setpoint -10 V, holding at -10 V | settle 0.3 s
 8  move      transit to object 1, verified withdraw first (holding at 0 V)
 9  read      free air: Deflection  [run time]
10  engage    --relative --setpoint=+0.05 | loop 2 on Deflection, Height out, limits -10 to 150
              | settle 0.3 s
11  PASS      litho object 1 "funnel_left": one continuous serpentine, 41,280 points, 1 chunk
              (ARMFIRE), ends at 0 V, waves loaded in 8,192-point lines, points 9d4f7a10
12  withdraw  setpoint -10 V, holding at -10 V | settle 0.3 s
13  move      transit to object 2, verified withdraw first
14  read      free air: Deflection  [run time]
15  engage    --relative --setpoint=+0.05 | loop 2 on Deflection, Height out, limits -10 to 150
              | settle 0.3 s
16  PASS      litho object 2 "pad_right": zig-zag, 121,536 points, 2 chunks (LITHOCHAIN, cut at
              86,976), bias on across the seam, ends at 0 V, waves loaded in 8,192-point lines,
              points 5e0a1b2c
17  withdraw  setpoint -10 V, holding at -10 V | settle 0.3 s
18  withdraw  verified, holding at 0 V (end of run)
"""

SCAN_TEXT = """\
run        scan | recipe 0aab3fa1 | execution s-2026-10-24-1815-01
platform   cypher | ac mode
frame      from request: 30 um, 256 x 256, 0 deg, offset 0, 0 um (+ drift offset), up, 1 line/s,
           trace and retrace
setpoint   absolute 0.65 V
capture    every pass: ZSensor, Amplitude, UserIn2
if script goes silent   park
 0  check     no latched fault | frame inside the XY range
 1  setmode   ac (only if withdrawn and the mode differs) | settle 1 s
 2  move      lifted to the first point at 5 um/s, verified withdraw first if engaged | settle
              0.5 s
 3  engage    --absolute --setpoint=0.65 | loop 2 on Amplitude, sign -1, Height out, limits -10
              to 150 (only when engage is requested)
 4  PASS      scan: 256 lines x 512 points, interpolation 195, decimation 195, error channel
              Amplitude, XY loops 0/1 on Output.X/Output.Y, limits -10 to 150, waves loaded in
              8,192-point lines (16 per wave), one SCANCHAIN command
 5  end       stays engaged (no withdraw)
"""


def test_write_text_matches_mockup():
    assert run_text(write_run()) == WRITE_TEXT


def test_scan_text_matches_mockup():
    assert run_text(scan_run()) == SCAN_TEXT


def test_text_is_ascii():
    for run in (write_run(), scan_run()):
        assert run_text(run).isascii()
        assert run_to_json(run).isascii()


@pytest.mark.parametrize("make", [write_run, scan_run])
def test_json_round_trip(make):
    run = make()
    stored = run_to_json(run)
    again = run_from_json(stored)
    assert again == run
    assert run_to_json(again) == stored


def test_json_is_canonical():
    stored = run_to_json(write_run())
    plain = json.loads(stored)
    assert stored == json.dumps(plain, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert [e["n"] for e in plain["recipe"]["body"]] == list(range(19))
    assert plain["recipe"]["body"][5]["kind"] == "engage"
    assert plain["recipe"]["version"] == 1


def test_json_refuses_a_tampered_recipe():
    plain = json.loads(run_to_json(write_run()))
    plain["recipe"]["body"][6]["points"] = 16609
    with pytest.raises(ValueError, match="fingerprint"):
        run_from_json(json.dumps(plain))


def test_json_refuses_misnumbered_and_unknown_entries():
    plain = json.loads(run_to_json(scan_run()))
    plain["recipe"]["body"][2]["n"] = 7
    with pytest.raises(ValueError, match="numbered"):
        run_from_json(json.dumps(plain))
    plain = json.loads(run_to_json(scan_run()))
    plain["recipe"]["body"][2]["kind"] = "teleport"
    with pytest.raises(ValueError, match="unknown kind"):
        run_from_json(json.dumps(plain))


def test_fingerprint_is_stable_and_excludes_the_execution():
    a = write_run(started=STARTED, seq=1)
    b = write_run(started="2026-10-24T14:02:59-04:00", seq=2)
    b = replace(b, execution=replace(b.execution, design_snapshot="elsewhere/design.json",
                                     commanded_xy=()))
    assert a.execution != b.execution
    assert recipe_fingerprint(a.recipe) == recipe_fingerprint(b.recipe)
    assert recipe_fingerprint(write_recipe()) == recipe_fingerprint(write_recipe())


def test_fingerprint_does_not_depend_on_int_or_float():
    a = Setpoint("absolute", 1)
    b = Setpoint("absolute", 1.0)
    assert a == b and type(a.volts) is float
    ra = replace(scan_recipe(), frame=replace(scan_recipe().frame, size_um=30))
    rb = replace(scan_recipe(), frame=replace(scan_recipe().frame, size_um=30.0))
    assert recipe_fingerprint(ra) == recipe_fingerprint(rb)


def test_fingerprint_changes_with_the_recipe():
    base = write_recipe()
    other = replace(base, placement=replace(base.placement, angle_deg=0.5))
    assert recipe_fingerprint(base) != recipe_fingerprint(other)
    body = list(base.body)
    body[7] = replace(body[7], settle_s=0.31)
    assert recipe_fingerprint(base) != recipe_fingerprint(replace(base, body=tuple(body)))


def test_execution_ids_differ_for_identical_recipes():
    started = datetime(2026, 10, 24, 14, 2, 11, tzinfo=EDT)
    first = next_execution_id("write", started)
    second = next_execution_id("write", started, taken=[first])
    assert (first, second) == ("w-2026-10-24-1802-01", "w-2026-10-24-1802-02")
    runs = [Run(write_recipe(), Execution(i, STARTED)) for i in (first, second)]
    assert recipe_fingerprint(runs[0].recipe) == recipe_fingerprint(runs[1].recipe)
    assert runs[0].execution.execution_id != runs[1].execution.execution_id
    assert next_execution_id("scan", started) == "s-2026-10-24-1802-01"


def test_execution_ids_use_utc_and_need_an_aware_time():
    with pytest.raises(ValueError, match="aware"):
        next_execution_id("write", datetime(2026, 10, 24, 14, 2))
    # 01:30 EDT and 01:10 EST a day later sort in start order only in UTC.
    edt = next_execution_id("write", datetime(2026, 11, 1, 1, 30, tzinfo=EDT))
    est = next_execution_id("write", datetime(2026, 11, 1, 1, 10,
                                              tzinfo=timezone(timedelta(hours=-5))))
    assert edt < est


def test_execution_id_must_fit_the_kind_and_the_start():
    with pytest.raises(ValueError, match="letter"):
        Run(write_recipe(), Execution("s-2026-10-24-1802-01", STARTED))
    with pytest.raises(ValueError, match="execution_id"):
        Execution("write-3", STARTED)
    with pytest.raises(ValueError, match="NN from 01"):
        Execution("w-2026-10-24-1802-00", STARTED)
    with pytest.raises(ValueError, match="start minute"):
        Execution("w-2026-10-24-1402-01", STARTED)
    with pytest.raises(ValueError, match="UTC offset"):
        Execution("w-2026-10-24-1402-01", "2026-10-24T14:02:11")
    with pytest.raises(ValueError, match="ISO 8601"):
        Execution("w-2026-10-24-1802-01", "yesterday")


def test_every_engage_uses_the_header_setpoint():
    with pytest.raises(ValueError, match="engage at step 5"):
        replace(write_recipe(), setpoint=Setpoint("absolute", 0.65))


def test_withdraw_setpoint_is_marked():
    with pytest.raises(ValueError, match="expected Setpoint"):
        Withdraw("back-off", hold_v=-10, setpoint=-10)  # type: ignore[arg-type]


def test_malformed_json_raises_value_error():
    plain = json.loads(run_to_json(write_run()))
    plain["recipe"]["body"][2]["to"] = {"x": 1}
    with pytest.raises(ValueError):
        run_from_json(json.dumps(plain))
    plain = json.loads(run_to_json(write_run()))
    plain["recipe"]["body"][2] = ["move"]
    with pytest.raises(ValueError):
        run_from_json(json.dumps(plain))
    plain = json.loads(run_to_json(write_run()))
    plain["recipe"]["body"] = {"0": 1}
    with pytest.raises(ValueError):
        run_from_json(json.dumps(plain))
    with pytest.raises(ValueError):
        run_from_json("[1, 2]")


def test_unknown_fields_are_refused():
    plain = json.loads(run_to_json(write_run()))
    plain["recipe"]["body"][2]["colour"] = "red"
    with pytest.raises(ValueError, match="unknown fields"):
        run_from_json(json.dumps(plain))


def test_numbers_are_finite_and_not_bools():
    with pytest.raises(ValueError, match="not finite"):
        Param("wavelength", float("nan"), "nm")
    with pytest.raises(ValueError, match="not finite"):
        Settle(float("inf"))
    with pytest.raises(ValueError, match="expected a number"):
        Param("wavelength", True, "nm")
    with pytest.raises(ValueError, match="whole number"):
        StepFile(True, "x.f32")
    with pytest.raises(ValueError, match="not one of"):
        replace(scan_recipe().body[3], sign=True)
    with pytest.raises(ValueError, match="not one of"):
        replace(scan_recipe().body[3], sign=1.0)


def test_canonical_json_refuses_nan():
    from afm_litho.run.json_form import canonical
    with pytest.raises(ValueError):
        canonical({"x": float("nan")})


def test_minus_zero_fingerprints_as_zero():
    a = replace(scan_recipe(), frame=replace(scan_recipe().frame, offset_um=(-0.0, 0.0)))
    assert recipe_fingerprint(a) == recipe_fingerprint(scan_recipe())


def test_names_have_no_tabs_newlines_or_quotes():
    for bad in ("lead\tchiral", "lead\nchiral"):
        with pytest.raises(ValueError, match="ASCII"):
            Param(bad, 1, "nm")
    with pytest.raises(ValueError, match="double quote"):
        replace(write_recipe().body[6], name='lead"chiral')
    with pytest.raises(ValueError, match="double quote"):
        replace(write_recipe().design, name='Device"B')


def test_fingerprints_must_be_hex():
    with pytest.raises(ValueError, match="points_fingerprint"):
        replace(write_recipe().body[6], points_fingerprint="not-hex!")
    with pytest.raises(ValueError, match="snapshot_fingerprint"):
        replace(write_recipe().design, snapshot_fingerprint="XYZ12345")


def test_version_must_be_1():
    with pytest.raises(ValueError, match="version"):
        replace(write_recipe(), version=2)


def test_litho_objects_are_distinct_objects_of_the_design():
    r = write_recipe()
    body = list(r.body)
    body[11] = replace(body[11], index=0)
    with pytest.raises(ValueError, match="distinct"):
        replace(r, body=tuple(body))
    with pytest.raises(ValueError, match="distinct"):
        replace(r, design=replace(r.design, objects=2))


def test_scan_lines_match_the_frame():
    r = scan_recipe()
    with pytest.raises(ValueError, match="frame"):
        replace(r, frame=replace(r.frame, lines=128))


def test_capture_bank_needs_a_pair():
    with pytest.raises(ValueError, match="need a pair"):
        Capture(("Height",), pair_bank=2)


def test_setpoint_reference_is_required():
    with pytest.raises(TypeError):
        Setpoint(0.05)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="not one of"):
        Setpoint("above-free-air", 0.05)  # type: ignore[arg-type]
    plain = json.loads(run_to_json(write_run()))
    del plain["recipe"]["body"][5]["setpoint"]["reference"]
    with pytest.raises(ValueError, match="missing fields"):
        run_from_json(json.dumps(plain))


def test_names_must_be_typeable_ascii():
    with pytest.raises(ValueError, match="ASCII"):
        Param("λ", 25, "nm")
    with pytest.raises(ValueError, match="ASCII"):
        Param("wavelength", 25, "µm")


def test_litho_fire_matches_chunks():
    good = write_recipe().body[6]
    with pytest.raises(ValueError, match="LITHOCHAIN"):
        replace(good, chunks=2)
    with pytest.raises(ValueError, match="chunk_cuts"):
        replace(good, chunk_cuts=(86976,))
    chained = write_recipe().body[16]
    three = replace(chained, chunks=3, points=200000, chunk_cuts=(86976, 173952))
    assert three.chunk_cuts == (86976, 173952)
    with pytest.raises(ValueError, match="chunk_cuts"):
        replace(chained, chunks=3, points=200000, chunk_cuts=(86976,))
    with pytest.raises(ValueError, match="increase"):
        replace(chained, chunks=3, points=200000, chunk_cuts=(173952, 86976))
    with pytest.raises(ValueError, match="one chunk"):
        replace(good, chunks=0)


def test_capture_pair_bank_is_a_real_bank():
    with pytest.raises(ValueError, match="pair_bank"):
        Capture(("Height",), pair=("UserIn2", "Lateral"), pair_bank=3)


def test_header_must_fit_the_kind():
    with pytest.raises(ValueError, match="do not fit"):
        replace(scan_recipe(), kind="write")
    with pytest.raises(ValueError, match="no pass"):
        replace(scan_recipe(), body=(End(engaged=False),))


def test_cite_names_execution_and_step():
    run = write_run()
    assert run.cite(6) == {"run": "w-2026-10-24-1802-01", "step": 6}
    with pytest.raises(ValueError):
        run.cite(19)


def test_commanded_xy_files_cite_litho_passes():
    with pytest.raises(ValueError, match="commanded"):
        Run(write_recipe(), Execution("w-2026-10-24-1802-01", STARTED,
                                      commanded_xy=(StepFile(5, "x.f32"),)))
