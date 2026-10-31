"""Slice 3.1b: the run built for a scan predicts what the scan does on the wire.

Two oracles: the frozen scan goldens (the SCANCHAIN line, the wave size, the loops armed and the
steps before the pass) and the chained-scan stub (the engage loop, the SCANCHAIN arguments and the
wave lines of requests the goldens do not make).
"""

from __future__ import annotations

import math
import re

import pytest
from test_hw_scan_chained import CAL, _run
from wire_facts import golden_actuation, scanchain_args

from afm_litho.hardware import bridge_protocol as P
from afm_litho.hardware.hardware_arc import WAVE_CHUNK_POINTS
from afm_litho.run import Execution, run_from_json, run_text, run_to_json
from afm_litho.run.build_scan import build_scan_run
from afm_litho.run.model import Check, End, Engage, Move, Scan, SetMode, Withdraw
from afm_litho.service.host.scan_cmds import build_scan_config
from afm_litho.service.passes import ScanConfig

EXECUTION = Execution("s-2026-10-24-1415-01", "2026-10-24T14:15:00+00:00")

# The startScan bodies the golden scenarios send (tests/golden/scenarios.py `_scan`).
_BASE = {"size_um": 1.0, "pixels": 32, "lines": 2, "line_rate_hz": 50.0, "x_offset_um": 0.0,
         "y_offset_um": 0.0, "angle_deg": 0.0, "direction": "up", "trace_retrace": True,
         "mode": "contact"}
SCAN_GOLDENS = {
    "scan_chained_contact": _BASE,
    "scan_chained_ac": dict(_BASE, mode="ac"),
    "scan_abort_mid_line": dict(_BASE, lines=4),
    "scan_stop_deferred_withdraw": dict(_BASE, lines=4),
    "continuous_two_frames": _BASE,
    "multi_pass": _BASE,
}


def _build(cfg: dict, *, platform="cypher", imaging_mode="contact", engaged=False, **kw):
    return build_scan_run(build_scan_config(cfg), platform=platform, execution=EXECUTION,
                          imaging_mode=imaging_mode, engaged=engaged, **kw)


def _kinds(run) -> list[type]:
    return [type(e) for e in run.recipe.body]


def _scan_pass(run) -> Scan:
    (p,) = [e for e in run.recipe.body if isinstance(e, Scan)]
    return p


def _scanchain_args(p: Scan) -> tuple:
    """The SCANCHAIN arguments the run states: ppl, lines, interpolation, decimation, error."""
    return (p.points_per_line, p.lines, p.interpolation, p.decimation, p.error_channel)


@pytest.mark.parametrize("name", SCAN_GOLDENS)
def test_run_predicts_the_goldens_scanchain(name):
    cfg = SCAN_GOLDENS[name]
    run = _build(cfg, imaging_mode=cfg["mode"])
    p = _scan_pass(run)
    actuation = golden_actuation(name)
    chains = [a for a in actuation if a.startswith("SCANCHAIN")]
    assert chains
    xw, yw, ppl, lines, interp, decim, err = scanchain_args(chains[0])
    assert _scanchain_args(p) == (int(ppl), int(lines), int(interp), int(decim), err)
    assert run.recipe.capture.channels == ("ZSensor", err, "UserIn2")

    # the waves are the whole frame, played from 8,192-point lines
    index = actuation.index(chains[0])
    waves = {m.group(1): int(m.group(2)) for a in actuation[:index]
             if (m := re.match(r"WAVE wave\((\w+),n=(\d+),", a))}
    assert waves[xw] == waves[yw] == p.points_per_line * p.lines
    assert p.wave_line_points == WAVE_CHUNK_POINTS
    assert math.ceil(waves[xw] / p.wave_line_points) == 1


@pytest.mark.parametrize("name", SCAN_GOLDENS)
def test_run_predicts_the_goldens_loops_and_steps(name):
    cfg = SCAN_GOLDENS[name]
    run = _build(cfg, imaging_mode=cfg["mode"])
    p = _scan_pass(run)
    actuation = golden_actuation(name)
    index = next(i for i, a in enumerate(actuation) if a.startswith("SCANCHAIN"))
    # the scan's own loop arm sits just before its waves: loops 0 and 1, limits from the pass
    arm = [a for a in actuation[:index] if a.startswith("SERVOLOOP")][-2:]
    for loop, out, line in zip(p.xy_loops, p.xy_outputs, arm):
        fields = line.split()
        assert fields[1] == str(loop) and fields[-3] == out
        assert (float(fields[-2]), float(fields[-1])) == p.limits_v
    # no engage was requested and the tip was not engaged: no loop 2 is armed for the scan
    assert _kinds(run) == [Check, Move, Scan, Withdraw]
    transit_start = max(i for i, a in enumerate(actuation[:index]) if a.startswith("FIRE")) + 1
    assert not any(a.startswith("SERVOLOOP 2") for a in actuation[transit_start:index])
    assert run.recipe.body[-1].how == "verified"


def test_goldens_record_the_transit_the_run_lists():
    actuation = golden_actuation("scan_chained_contact")
    assert any(a.startswith("OUTPAIR") for a in actuation)
    assert any(a.startswith("FIRE") and "pass0" in a for a in actuation)
    move = _build(SCAN_GOLDENS["scan_chained_contact"]).recipe.body[1]
    assert (move.withdraw_first, move.speed_um_s) == ("verified-if-engaged", 5.0)
    assert "max(0.5 s, 0.25 x move time)" in move.how


def test_defaults_match_todays_scan_config():
    run = _build({})
    r, frame, p = run.recipe, run.recipe.frame, _scan_pass(run)
    assert (frame.size_um, frame.pixels, frame.lines, frame.line_rate_hz) == (30.0, 256, 256, 1.0)
    assert frame.trace_and_retrace and frame.direction == "up"
    assert (r.setpoint.reference, r.setpoint.volts) == ("absolute", 0.2)
    assert _scanchain_args(p) == (512, 256, -195, 195, "Deflection")
    assert p.xy_loops == (0, 1) and p.xy_outputs == ("Output.X", "Output.Y")
    assert p.limits_v == (-10.0, 150.0)
    assert 256 * 512 // p.wave_line_points == 16


# --- the request's edge cases, each pinned against the wire ---------------------------------


def test_zero_lines_means_one_line_per_pixel():
    # ScanConfig already turns 0 into pixels on construction; the scan code still guards a
    # config whose lines were zeroed afterwards, so the test zeroes them the same way.
    config = ScanConfig(size_um=5.0, pixels=32, lines=4, line_rate_hz=50.0)
    config.lines = 0
    run = build_scan_run(config, platform="cypher", execution=EXECUTION,
                         imaging_mode="contact", engaged=False)
    assert _scan_pass(run).lines == run.recipe.frame.lines == 32
    _msgs, _summary, stub = _run(config, CAL)
    assert stub.scanchains[0][3] == 32


def test_the_fastest_line_rate_plays_every_clock_tick():
    cfg = {"size_um": 5.0, "pixels": 256, "lines": 2, "line_rate_hz": 200.0}
    run = _build(cfg)
    assert _scan_pass(run).interpolation == -1
    _msgs, _summary, stub = _run(build_scan_config(cfg), CAL)
    _x, _y, ppl, lines, interp, decim = stub.scanchains[0]
    assert (ppl, lines, interp, decim) == _scanchain_args(_scan_pass(run))[:4]


def test_the_check_names_the_frame_range():
    (check,) = [e for e in _build({}).recipe.body if isinstance(e, Check)]
    assert check.items == ("no-latched-fault", "frame-inside-xy-range")


def test_a_large_frame_loads_in_more_than_one_wave_line(monkeypatch):
    appended = []
    real = P.cmd_waveapp
    monkeypatch.setattr(P, "cmd_waveapp", lambda name, vals: appended.append(name) or
                        real(name, vals))
    cfg = {"size_um": 5.0, "pixels": 128, "lines": 64, "line_rate_hz": 50.0}
    p = _scan_pass(_build(cfg))
    per_wave = math.ceil(p.lines * p.points_per_line / p.wave_line_points)
    assert per_wave == 2
    _run(build_scan_config(cfg), CAL)
    # the X and Y scan waves each add the lines after their first; the small transit waves add none
    assert len(appended) == 2 * (per_wave - 1)


# --- the mode gate and the tip state --------------------------------------------------------


def test_the_mockup_case_plans_the_setmode_and_an_unconditional_verified_withdraw():
    """AC scan with engage, software still in contact, tip withdrawn: the mode gate sees the
    tip up (setmode happens), the transit sees it down because the scan marked itself engaged."""
    run = _build({"mode": "ac", "engage": True, "setpoint": 0.65}, imaging_mode="contact")
    assert _kinds(run) == [Check, SetMode, Move, Engage, Scan, Withdraw]
    setmode, move = run.recipe.body[1], run.recipe.body[2]
    assert (setmode.mode, setmode.settle_s) == ("ac", 1.0)
    assert move.withdraw_first == "verified"
    assert run.recipe.body[-1].how == "verified"


def test_setmode_only_when_the_gate_is_open_and_the_mode_differs():
    ac = {"mode": "ac"}
    assert SetMode not in _kinds(_build(ac, imaging_mode="ac"))
    assert SetMode in _kinds(_build(ac, imaging_mode="contact"))
    assert SetMode not in _kinds(_build(ac, imaging_mode="contact", engaged=True))
    # continuous imaging, a running raster or a write close the gate with the tip not engaged
    assert SetMode not in _kinds(_build(ac, imaging_mode="contact", mode_gate_closed=True))
    # a closed gate stays closed even when engaged is False and the other way round
    assert SetMode in _kinds(_build(ac, imaging_mode="contact", engaged=True,
                                    mode_gate_closed=False))


def test_the_tracked_mode_is_read_as_the_scan_reads_it():
    assert SetMode not in _kinds(_build({"mode": "ac"}, imaging_mode="AC"))
    unknown = _build({"mode": "ac"}, imaging_mode="").recipe.body
    assert _kinds(_build({"mode": "ac"}, imaging_mode=None)) == _kinds(
        _build({"mode": "ac"}, imaging_mode=""))
    (setmode,) = [e for e in unknown if isinstance(e, SetMode)]
    assert "instrument's own mode read" in setmode.when
    # an unknown mode while the gate is closed only records the mode: no switch
    assert SetMode not in _kinds(_build({"mode": "ac"}, imaging_mode="", engaged=True))


@pytest.mark.parametrize("mode,platform,on,out,sign",
                         [("contact", "mfp3d", "Deflection", "Height", 1),
                          ("ac", "mfp3d", "Amplitude", "Height", -1),
                          ("contact", "cypher", "Deflection", "Height", 1),
                          ("ac", "cypher", "Amplitude", "Height", -1)])
def test_engage_step_matches_the_loop_the_scan_arms(mode, platform, on, out, sign):
    cfg = {"size_um": 5.0, "pixels": 32, "lines": 4, "mode": mode, "engage": True,
           "setpoint": 0.30}
    run = _build(cfg, platform=platform, imaging_mode=mode)
    assert _kinds(run) == [Check, Move, Engage, Scan, Withdraw]
    e = run.recipe.body[2]
    assert (e.loop, e.on, e.out, e.sign, e.limits_v) == (2, on, out, sign, (-10.0, 150.0))
    assert (e.setpoint.reference, e.setpoint.volts) == ("absolute", 0.30)
    assert run.recipe.body[-1].how == "verified"


@pytest.mark.parametrize("mode", ["contact", "ac"])
def test_run_predicts_the_stub_scanchain_and_engage_loop(mode):
    config = ScanConfig(size_um=5.0, pixels=32, lines=4, mode=mode, engage=True, setpoint=0.30)
    _msgs, _summary, stub = _run(config, CAL)
    run = build_scan_run(config, platform="mfp3d", execution=EXECUTION, imaging_mode=mode,
                         engaged=False)
    p, engage = _scan_pass(run), run.recipe.body[2]
    _x, _y, ppl, lines, interp, decim = stub.scanchains[0]
    assert _scanchain_args(p)[:4] == (ppl, lines, interp, decim)
    loops = {lp[0]: lp for lp in stub.loops}
    _, inp, setpoint, _pg, ig, outp = loops[2]
    assert (inp, outp, setpoint) == (engage.on, engage.out, engage.setpoint.volts)
    assert math.copysign(1, ig) == engage.sign
    assert {0, 1} <= set(loops)


@pytest.mark.parametrize("mode", ["contact", "ac"])
def test_a_scan_started_engaged_re_engages_through_the_transit(mode):
    """The tip was engaged and the scan does not engage: the transit withdraws, moves, and
    re-arms loop 2 itself at the scan's setpoint."""
    config = ScanConfig(size_um=5.0, pixels=32, lines=4, mode=mode, engage=False, setpoint=0.30)
    run = build_scan_run(config, platform="mfp3d", execution=EXECUTION, imaging_mode=mode,
                         engaged=True)
    assert _kinds(run) == [Check, Move, Engage, Scan, Withdraw]
    move, engage = run.recipe.body[1], run.recipe.body[2]
    assert move.withdraw_first == "verified" and run.recipe.body[-1].how == "verified"
    assert engage.when.startswith("the transit's own re-engage")

    _msgs, _summary, stub = _run(config, CAL, engaged=True)
    last_z = [lp for lp in stub.loops if lp[0] == 2][-1]
    _, inp, setpoint, _pg, ig, outp = last_z
    assert (inp, outp, setpoint) == (engage.on, engage.out, engage.setpoint.volts)
    assert math.copysign(1, ig) == engage.sign
    # the scan itself arms no loop 2: the transit arms it twice, the withdraw (-10 V) then the re-engage
    assert [lp[0] for lp in stub.loops].count(2) == 2


def test_the_built_run_round_trips_and_prints():
    run = _build({"mode": "ac", "engage": True, "setpoint": 0.65}, imaging_mode="contact")
    assert run_from_json(run_to_json(run)) == run
    text = run_text(run)
    assert "decimation" in text and "re-engage" not in text


def test_a_continuous_frame_ends_without_a_withdraw():
    run = _build(_BASE, ends_with_withdraw=False)
    assert _kinds(run) == [Check, Move, Scan, End]
