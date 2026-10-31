"""The optional engage check (hardware/engage_check.py, AFM_LITHO_CONFIRM_ENGAGE=1).

Rig, Cypher: the bias came on and XY moved before the tip had engaged. Switch off (the
default): the set-down is the fixed SETTLE_MOVE_S (1.0 s) and nothing is read. Switch on:
the run polls Deflection after the set-down until ENGAGE_READS reads in a row have gone
ENGAGE_FRACTION of the way from free air to the setpoint; a timeout or an abort while waiting
fires nothing. A setpoint within MIN_ENGAGE_STEP_V of free air, or no agreed free-air read,
refuses the write before the set-down. Every test but the switch-off one runs with it on.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path

import pytest

from afm_litho import clock
from afm_litho.hardware import bridge_protocol as P
from afm_litho.hardware import engage_check
from afm_litho.hardware.engage_check import (
    ENGAGE_FRACTION,
    ENGAGE_READS,
    MIN_ENGAGE_STEP_V,
    EngageNotConfirmed,
    engage_progress,
)
from afm_litho.hardware.igor_stub import FakeIgorBridge
from afm_litho.hardware.lithography_run import (
    SETTLE_MOVE_S,
    FreeAirRefused,
    LithographyRun,
    WriteAborted,
    WriteSetpoint,
)

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "SA40656B0002.ibw"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample .ibw not present")

SETPOINT_V = 0.4
FREE_AIR_V = 0.0          # the sample stub's DeflectionValue


@pytest.fixture(autouse=True)
def _testing_clock():
    with clock.using(clock.TESTING) as c:
        yield c


@pytest.fixture(autouse=True)
def _switch_on(request):
    if request.node.name.startswith("test_switch_off"):
        yield
        return
    with engage_check.use(True):
        yield


def _serve():
    stub = FakeIgorBridge.from_ibw(SAMPLE, done_delay=0.0)
    lines: list[str] = []
    orig = stub._dispatch

    def rec(line, link):
        lines.append(line)
        orig(line, link)

    stub._dispatch = rec
    return stub, stub.serve_in_thread(), lines


def _run(port, **cfg):
    kw = {"settle_move_s": 0.0, "cmd_gap_s": 0.0, "xy_withdraw_max_s": 0.0,
          "xy_withdraw_settle_s": 0.0, "first_object_settle_s": 0.0}
    kw.update(cfg)
    run = LithographyRun.connect("127.0.0.1", port, **kw)
    run.start()
    return run


def _write(run, name="A", source="A", setpoint=None):
    return run.write_object(name, [10.0, 12.0], [0.0, 0.0], [3.0, 3.0],
                            setpoint=setpoint or WriteSetpoint(SETPOINT_V, relative=False),
                            speed=5.0, source=source)


def _set_down_index(lines, start=0, setpoint=SETPOINT_V):
    """Index of the first loop-2 arm at the write setpoint at or after `start`."""
    for i in range(start, len(lines)):
        p = lines[i].split(" ")
        if p[0] == P.SERVOLOOP and p[1] == "2" and float(p[3]) == pytest.approx(setpoint):
            return i
    raise AssertionError("no set-down")


def _set_downs(lines):
    """Loop-2 arms above the -10 V retract: set-downs toward the surface."""
    return [ln for ln in lines if ln.split(" ")[0] == P.SERVOLOOP
            and ln.split(" ")[1] == "2" and float(ln.split(" ")[3]) > -5.0]


def _defl_reads(lines):
    return [i for i, ln in enumerate(lines)
            if ln.startswith(P.GETVALUE + " ") and ln.endswith(" Deflection")]


def _first(lines, prefix, start=0):
    return next(i for i in range(start, len(lines)) if lines[i].startswith(prefix))


def _script_deflection(stub, lines, answers, setpoint):
    """After the set-down, answer the Deflection reads from `answers` in order (then hold
    the last one)."""
    stub.deflection_follows_loop = False
    orig = stub._dispatch
    left = list(answers)

    def answer(line, link):
        down = any(ln.split(" ")[0] == P.SERVOLOOP and ln.split(" ")[1] == "2"
                   and float(ln.split(" ")[3]) == pytest.approx(setpoint) for ln in lines)
        if down and line.startswith(P.GETVALUE + " ") and line.endswith(" Deflection"):
            stub.values["Deflection"] = left.pop(0) if len(left) > 1 else left[0]
        orig(line, link)

    stub._dispatch = answer


def test_progress_is_one_sided_from_free_air_toward_the_setpoint():
    assert engage_progress(0.0, 0.0, 0.1) == 0.0
    assert engage_progress(0.1, 0.0, 0.1) == 1.0
    assert engage_progress(-0.05, 0.0, 0.1) < 0
    assert engage_progress(0.12, 0.0, 0.1) > 1          # past the setpoint still counts
    assert engage_progress(-1.9, -1.0, -2.0) == pytest.approx(0.9)
    assert math.isnan(engage_progress(math.nan, 0.0, 0.1))


def test_a_confirmed_engage_reads_deflection_between_the_set_down_and_the_fire():
    stub, port, lines = _serve()
    steps = []
    try:
        run = _run(port)
        run.on_outcome = lambda kind, make: steps.append((kind, make()))
        _write(run)
        run.get_value("XSensor")                      # barrier
    finally:
        stub.stop()
    down = _set_down_index(lines)
    reads = [i for i in _defl_reads(lines) if i > down]
    first_wave = _first(lines, P.WAVE + " XLitho", down)
    fire = _first(lines, P.ARMFIRE + " ", down)
    assert len([i for i in reads if i < first_wave]) == ENGAGE_READS
    assert down < reads[0] and reads[ENGAGE_READS - 1] < first_wave < fire
    # nothing but reads between the set-down and the first wave load
    assert all(ln.startswith(P.GETVALUE + " ") for ln in lines[down + 1:first_wave])
    engage = dict(steps)["engage"]
    assert engage["deflection_v"] == pytest.approx(SETPOINT_V)
    assert engage["free_air_v"] == FREE_AIR_V
    assert len(stub.arms) == 1
    run.close()


def test_every_set_down_is_confirmed_hover_and_transit_alike():
    stub, port, lines = _serve()
    try:
        run = _run(port)
        _write(run, "A0", "A")
        _write(run, "A1", "A")                         # hover move, same source
        _write(run, "B0", "B")                         # transit, a new object
        run.get_value("XSensor")
    finally:
        stub.stop()
    start = 0
    for _ in range(3):
        down = _set_down_index(lines, start)
        fire = _first(lines, P.ARMFIRE + " ", down)
        assert len([i for i in _defl_reads(lines) if down < i < fire]) == ENGAGE_READS
        start = fire + 1
    assert len(stub.arms) == 3
    run.close()


def test_the_review_repro_a_step_under_the_minimum_fires_nothing():
    """Relative 0.03 V over free air with the tip never moving: the free-air reads would sit
    in a symmetric band and 'confirm'. The write is refused before the set-down instead."""
    stub, port, lines = _serve()
    stub.deflection_follows_loop = False
    try:
        run = _run(port)
        with pytest.raises(FreeAirRefused, match="within 0.05 V of free air"):
            _write(run, setpoint=WriteSetpoint(0.03, relative=True))
        run.get_value("XSensor")
    finally:
        stub.stop()
    assert stub.arms == [] and stub.lithochains == []
    assert _set_downs(lines) == [] and "XLitho" not in stub.waves
    run.close()


def test_a_step_under_the_minimum_is_refused_before_anything_moves_absolute_too():
    stub, port, lines = _serve()
    try:
        run = _run(port)
        with pytest.raises(FreeAirRefused):
            _write(run, setpoint=WriteSetpoint(FREE_AIR_V + 0.9 * MIN_ENGAGE_STEP_V,
                                               relative=False))
        run.get_value("XSensor")
    finally:
        stub.stop()
    assert _set_downs(lines) == [] and stub.arms == [] and "VLitho" not in stub.waves
    run.close()


def test_no_agreed_free_air_read_refuses_the_write_before_the_set_down():
    stub, port, lines = _serve()
    reads = iter([0.0, 1.0, 2.0])
    orig = stub._dispatch

    def disagree(line, link):
        if line.startswith(P.GETVALUE + " ") and line.endswith(" DeflectionValue"):
            stub.values["DeflectionValue"] = next(reads, 0.0)
        orig(line, link)

    stub._dispatch = disagree
    try:
        run = _run(port)
        with pytest.raises(FreeAirRefused, match="no free-air reference for the engage check"):
            _write(run)
        run.get_value("XSensor")
    finally:
        stub.stop()
    assert _set_downs(lines) == [] and stub.arms == []
    run.close()


def test_the_usual_0_1_v_confirms_only_after_ninety_percent():
    stub, port, lines = _serve()
    sp = FREE_AIR_V + 0.1
    # 0.089 is under 90% of the step: it restarts the count; the fire needs 3 at >= 0.09.
    _script_deflection(stub, lines, [0.0, 0.05, 0.089, 0.095, 0.089, 0.091, 0.1, 0.098], sp)
    try:
        run = _run(port)
        _write(run, setpoint=WriteSetpoint(0.1, relative=True))
        run.get_value("XSensor")
    finally:
        stub.stop()
    down = _set_down_index(lines, setpoint=sp)
    fire = _first(lines, P.ARMFIRE + " ", down)
    assert len([i for i in _defl_reads(lines) if down < i < fire]) == 8
    assert len(stub.arms) == 1
    assert ENGAGE_FRACTION == 0.9
    run.close()


def test_a_deflection_stalled_at_half_the_step_times_out_and_fires_nothing():
    stub, port, lines = _serve()
    sp = FREE_AIR_V + 0.1
    _script_deflection(stub, lines, [FREE_AIR_V + 0.05], sp)
    try:
        run = _run(port)
        with pytest.raises(EngageNotConfirmed) as err:
            _write(run, setpoint=WriteSetpoint(0.1, relative=True))
        run.get_value("XSensor")
    finally:
        stub.stop()
    msg = str(err.value)
    assert "did not engage within 6 s" in msg and "deflection 0.0500 V" in msg
    assert "setpoint 0.1000 V" in msg and "free air 0.0000 V" in msg
    down = _set_down_index(lines, setpoint=sp)
    assert stub.arms == [] and stub.lithochains == []
    assert "XLitho" not in stub.waves and "VLitho" not in stub.waves
    # reads only after the set-down: the caller's failure path owns the withdraw
    assert all(ln.startswith(P.GETVALUE + " ") for ln in lines[down + 1:])
    assert run._engaged is True                         # the next move must withdraw first
    run.close()


def test_a_tip_that_never_leaves_free_air_fires_nothing_and_raises():
    stub, port, lines = _serve()
    stub.deflection_follows_loop = False
    stub.values["Deflection"] = FREE_AIR_V
    try:
        run = _run(port)
        with pytest.raises(EngageNotConfirmed, match="did not engage"):
            _write(run)
        run.get_value("XSensor")
    finally:
        stub.stop()
    assert stub.arms == [] and "XLitho" not in stub.waves
    run.close()


def test_one_short_read_restarts_the_count():
    stub, port, lines = _serve()
    _script_deflection(stub, lines, [SETPOINT_V, SETPOINT_V, 0.0, SETPOINT_V, SETPOINT_V,
                                     SETPOINT_V], SETPOINT_V)
    try:
        run = _run(port)
        _write(run)
        run.get_value("XSensor")
    finally:
        stub.stop()
    down = _set_down_index(lines)
    fire = _first(lines, P.ARMFIRE + " ", down)
    assert len([i for i in _defl_reads(lines) if down < i < fire]) == 6
    run.close()


def test_an_abort_while_waiting_for_the_engage_fires_nothing():
    stub, port, lines = _serve()
    stub.deflection_follows_loop = False
    ab = threading.Event()
    orig = stub._dispatch
    seen = []

    def abort_on_second_read(line, link):
        if line.endswith(" Deflection") and line.startswith(P.GETVALUE + " "):
            seen.append(line)
            if len(seen) == 2:
                ab.set()
        orig(line, link)

    stub._dispatch = abort_on_second_read
    try:
        run = _run(port, abort_event=ab)
        with pytest.raises(WriteAborted, match="while waiting for the engage"):
            _write(run)
        run.get_value("XSensor")
    finally:
        stub.stop()
    down = _set_down_index(lines)
    assert stub.arms == [] and stub.lithochains == [] and "XLitho" not in stub.waves
    # no Z command after the abort: the park owns Z
    assert all(ln.startswith(P.GETVALUE + " ") for ln in lines[down + 1:])
    assert len(seen) == 2
    run.close()


def test_the_switch_reads_one_env_var_and_is_off_by_default():
    assert engage_check.from_env({}) is False
    assert engage_check.from_env({engage_check.ENV: "0"}) is False
    assert engage_check.from_env({engage_check.ENV: "1"}) is True
    with pytest.raises(ValueError, match=engage_check.ENV):
        engage_check.from_env({engage_check.ENV: "yes"})


def test_switch_off_the_set_down_reads_nothing_and_settles_one_second(_testing_clock):
    """Today's set-down with the 1.0 s settle: no Deflection read between the set-down
    and the fire, and a free-air step under 0.05 V still writes."""
    assert engage_check.ENABLED is False and SETTLE_MOVE_S == 1.0
    stub, port, lines = _serve()
    stub.deflection_follows_loop = False                # the tip never moves: no matter
    try:
        run = _run(port, settle_move_s=SETTLE_MOVE_S)
        mark = len(_testing_clock.waits)
        _write(run, setpoint=WriteSetpoint(0.03, relative=True))
        run.get_value("XSensor")
    finally:
        stub.stop()
    down = _set_down_index(lines, setpoint=0.03)
    fire = _first(lines, P.ARMFIRE + " ", down)
    assert [i for i in _defl_reads(lines) if down < i < fire] == []
    assert len(stub.arms) == 1
    sleeps = [w.seconds for w in _testing_clock.waits[mark:] if w.kind == "sleep"]
    assert 1.0 in sleeps
    run.close()
