"""Constants the architecture migration must reproduce exactly, stated as literals.

Each assertion compares a literal here with the code's value (or its observable effect on
the Igor-bound wire), so changing a source value fails a test instead of passing silently.
Igor-side defaults (`.ipf`) are out of reach of Python tests and are not pinned here.
"""

import asyncio
import importlib
import inspect
import time

import numpy as np
import pytest

from afm_litho.hardware import bridge_protocol as P
from afm_litho.hardware import channels
from afm_litho.hardware import hardware_arc, lithography_run, xy_transit
from afm_litho.hardware.hardware_arc import HardwareARC
from afm_litho.hardware.igor_stub import FakeIgorBridge
from afm_litho.litho import chunker, compiler
from afm_litho.hardware import free_air, hw_approach
from afm_litho.service import hw_scan, instrument_model
from afm_litho.service.passes import ScanConfig
from afm_litho.service.host import continuous, telemetry
from afm_litho.service.safety.rail_dwell import RailDwellLimits
from afm_litho.service.safety.write_interlocks import InterlockConfig

safe_park = importlib.import_module("afm_litho.service.host.safe_park")


def _connect(values=None, cypher=False):
    stub = FakeIgorBridge(values or {"Deflection": 1.5, "Amplitude": 0.65,
                                     "height": 20.0, "XSensor": 0.0, "YSensor": 0.0},
                          done_delay=0.0)
    port = stub.serve_in_thread()
    hw = HardwareARC.connect("127.0.0.1", port)
    if cypher:
        hw.startup = {"microscope_id": 4}
    return stub, hw


def _z_loop(stub):
    return {lp[0]: lp for lp in stub.loops}[2]


# ---------------------------------------------------------------- sizes

def test_outwave_and_chunk_sizes():
    # The v16 reference says 87,380; the v19 vendor cMaxOutWave (87,000) is what we obey.
    assert chunker.MAX_OUTWAVE_POINTS == 87_000      # litho/chunker.py:21
    assert chunker.CHAIN_CHUNK_POINTS == 86_976      # litho/chunker.py:29 (2718 * 32)
    assert 86_976 == 2718 * 32
    assert lithography_run.OUTWAVE_MAX_POINTS == 87_000   # hardware/lithography_run.py:63
    assert compiler.MAX_SEGMENT_POINTS == 8192       # litho/compiler.py:63
    assert hardware_arc.WAVE_CHUNK_POINTS == 8192    # hardware/hardware_arc.py:38
    assert lithography_run.WAVE_CHUNK_POINTS == 8192  # hardware/lithography_run.py:66


def test_capture_length_is_padded_to_a_multiple_of_32():
    assert compiler.capture_points(1) == 32          # litho/compiler.py:352
    assert compiler.capture_points(32) == 32
    assert compiler.capture_points(33) == 64
    assert compiler.capture_points(86_976) == 86_976  # the chunk keeps capLen == m
    assert compiler.capture_points(87_000) == 87_008  # why the chunk is not 87,000


def test_wave_longer_than_8192_goes_as_wave_then_waveapp():
    stub, hw = _connect()
    try:
        hw._load_wave(np.arange(8192 + 5, dtype=float))  # hardware/hardware_arc.py:383
        hw.td_ReadValue("sync")
    finally:
        hw.close()
        stub.stop()
    sent = [t for t in stub.transcript if t in (P.WAVE, P.WAVEAPP)]
    assert sent == [P.WAVE, P.WAVEAPP]


# ---------------------------------------------------------------- gains

def test_panel_gain_scaling_and_signs():
    assert hw_approach._PSCALE == 0.1                # hardware/hw_approach.py:68
    assert hw_approach._ISCALE == 100.0              # hardware/hw_approach.py:67
    assert hw_approach._CONTACT_GAIN_SIGN == 1.0     # hardware/hw_approach.py:43
    assert hw_approach._AC_GAIN_SIGN == -1.0         # hardware/hw_approach.py:44
    for mode, sign, chan in (("contact", 1.0, "Deflection"), ("ac", -1.0, "Amplitude")):
        stub, hw = _connect()
        try:
            hw_approach.hw_engage(hw, setpoint=0.5, pgain=2.0, igain=15.0, settle_s=0.0,
                                  mode=mode)           # hardware/hw_approach.py:112
        finally:
            hw.close()
            stub.stop()
        _, inp, sp, pg, ig, outp = _z_loop(stub)
        assert inp == chan and sp == 0.5
        assert pg == pytest.approx(sign * 0.2) and ig == pytest.approx(sign * 1500.0)
        assert outp == "Height"
        limits = [lim for lp, lim in zip(stub.loops, stub.loop_limits) if lp[0] == 2]
        assert limits[-1] == (-10.0, 150.0)          # hardware/hw_approach.py:21


# ---------------------------------------------------------------- retract / withdraw

def test_cypher_retract_constants(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    assert hw_approach._Z_RETRACT_SETPOINT == -10.0  # hardware/hw_approach.py:32
    assert hw_approach._Z_RETRACT_IGAIN == 1000.0    # hardware/hw_approach.py:33
    assert hw_approach.Z_RETRACT_OUTMIN == 0.0      # hardware/hw_approach.py:34
    stub, hw = _connect(cypher=True)
    try:
        hw_approach.hw_withdraw(hw)                  # hardware/hw_approach.py:180
    finally:
        hw.close()
        stub.stop()
    _, inp, sp, pg, ig, outp = _z_loop(stub)
    assert (inp, sp, pg, ig, outp) == ("Deflection", -10.0, 0.0, 1000.0, "Height")
    limits = [lim for lp, lim in zip(stub.loops, stub.loop_limits) if lp[0] == 2]
    assert limits[-1] == (0.0, 150.0)                # out-min 0, out-max 150
    assert not stub.ramps and 2 not in stub.stopped_loops


def test_withdraw_wait_and_ramp_rate_defaults(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", slept.append)
    stub, hw = _connect()
    try:
        hw_approach.hw_withdraw(hw)                  # defaults: hardware/hw_approach.py:180
    finally:
        hw.close()
        stub.stop()
    assert stub.ramps == []                          # the withdraw re-arms loop 2, no ramp
    assert slept[0] == 2.5                           # max_seconds 2 + settle_margin 0.5
    for fn in (hardware_arc.HardwareARC.ramp_output, lithography_run.LithographyRun.ramp_output):
        params = inspect.signature(fn).parameters    # hardware_arc.py:331, lithography_run.py:628
        assert params["rate"].default == 80.0 and params["max_seconds"].default == 2.0
    assert xy_transit.DEFAULT_WITHDRAW_MAX_S == 2.0  # hardware/xy_transit.py:63
    assert inspect.signature(xy_transit.xy_transit).parameters["withdraw_rate"].default == 80.0


# ---------------------------------------------------------------- litho set-down / lift

def test_litho_set_down_and_lift_defaults():
    params = inspect.signature(lithography_run.LithographyRun.__init__).parameters
    assert params["z_igain"].default == 1200.0       # hardware/lithography_run.py:143
    assert params["retract"].default == -10.0        # hardware/lithography_run.py:143
    # Cypher set-down/lift is cmd_servoloop(2, Deflection, sp, P=0, I=z_igain, out) with
    # no limit tokens, so Igor applies its own -10/150 default.
    assert P.cmd_servoloop(2, "Deflection", -10.0, 0, 1200.0, "Height") == \
        "SERVOLOOP 2 Deflection -10 0 1200 Height"   # hardware/bridge_protocol.py:79
    assert lithography_run._Z_LOOP == 2 and lithography_run._Z_IN == "Deflection"
    assert hw_approach.Z_MIN == -10.0 and hw_approach.Z_MAX == 150.0


# ---------------------------------------------------------------- scan timing

def _scan_wire(pixels, rate_hz):
    from tests.test_hw_scan_chained import CAL
    cfg = ScanConfig(size_um=5.0, pixels=pixels, lines=2, line_rate_hz=rate_hz)
    stub, hw = _connect({"XSensor": 0.0, "YSensor": 0.0})

    async def bc(_m):
        return None

    try:
        asyncio.run(hw_scan.run_hw_scan_chained(hw, cfg, CAL, bc))  # service/hw_scan.py:381
    finally:
        hw.close()
        stub.stop()
    return stub.scanchains[0]


@pytest.mark.parametrize("pixels, rate, interp", [
    (64, 2.0, -391),       # round(50000 / (64 * 2))
    (96, 1.0, -521),       # round(50000 / 96)
    (256, 200.0, -1),      # floor at one tick
])
def test_scan_interp_formula_and_decim(pixels, rate, interp):
    chain = _scan_wire(pixels, rate)
    assert chain[4] == interp and chain[5] == abs(interp)   # (interp, decim)


def test_clock_rate_is_50khz_everywhere():
    assert lithography_run.CLOCK_HZ == 50_000        # hardware/lithography_run.py:78
    assert inspect.signature(compiler.time_segment).parameters["clock_hz"].default == 50_000
    assert inspect.signature(compiler.ar_get_deci).parameters["clock_hz"].default == 50_000
    # 1 um step at 10 um/s: round(50000 * 1/10) ticks per point -- compiler.py:508
    seg = compiler.time_segment(np.array([[0.0, 0.0], [1.0, 0.0]]), 10.0)
    assert seg["ticks_per_point"] == 5000 and seg["points_per_second"] == 10.0


# ---------------------------------------------------------------- free air

def test_free_air_thresholds():
    assert lithography_run.WITHDRAWN_Z_DRIVE_MAX_V == 1.0   # hardware/lithography_run.py:75
    params = inspect.signature(free_air.read_free_air).parameters
    assert params["frac"].default == 0.10            # hardware/free_air.py:78
    assert params["dwell_s"].default == 5.0
    assert free_air.agree_within([1.0, 1.1, 1.0], frac=0.10)       # spread 0.1 <= 0.10 * mean
    assert not free_air.agree_within([1.0, 1.3, 1.0], frac=0.10)
    assert instrument_model._DEFAULT_FREE_AIR_ABS_BAND_V == 0.02   # instrument_model.py:64
    assert continuous.FREE_AIR_JUMP_FRAC == 0.10     # service/host/continuous.py:61
    assert hw_approach._FREE_AIR_DEFL == "DeflectionValue"          # hw_approach.py:58


# ---------------------------------------------------------------- interlocks

def test_sum_and_rail_interlock_numbers():
    assert instrument_model._DEFAULT_SUM_FLOOR_V == 0.5      # instrument_model.py:65
    assert instrument_model._DEFAULT_SUM_CEILING_FRAC == 0.9  # instrument_model.py:66
    assert instrument_model._DEFAULT_Z_RAIL_V == 140.0       # instrument_model.py:67
    assert instrument_model._DEFAULT_INTERLOCK_S == 5.0      # instrument_model.py:68
    assert telemetry.SUM_ABS_FLOOR_V == 0.5                 # host/telemetry.py:46
    assert telemetry.SUM_BASELINE_FRAC == 0.5               # host/telemetry.py:49
    assert telemetry.INTERLOCK_DEBOUNCE_POINTS == 5         # host/telemetry.py:50
    assert continuous.SUM_LOSS_V == 0.1                      # host/continuous.py:62
    assert continuous.SUM_CLAMP_FRAC == 0.9                  # host/continuous.py:63
    cfg = InterlockConfig()                                  # safety/write_interlocks.py:34
    assert (cfg.rail_v, cfg.rail_s, cfg.sum_floor_v, cfg.sum_frac, cfg.sum_s, cfg.defl_s) == \
        (140.0, 5.0, 0.5, 0.5, 5.0, 5.0)
    lim = RailDwellLimits()                                  # safety/rail_dwell.py:46
    assert (lim.v_hi, lim.v_lo, lim.frame_frac, lim.line_frac, lim.consec_lines,
            lim.poll_hz) == (140.0, 0.0, 0.5, 0.95, 5, 10.0)
    assert safe_park._WITHDRAWN_HEIGHT_TOL_V == 0.5          # host/safe_park.py:31
    assert safe_park._RETRACT_MAX_S == 2.0                   # host/safe_park.py:39
    assert hw_approach.VERIFY_READ_TIMEOUT_S == 2.0          # hw_approach.py:74


def test_sum_ceiling_boundaries_come_from_the_pinned_numbers():
    from types import SimpleNamespace

    from afm_litho.service.command_guard import GuardRefused, check_sum_ceiling
    model = SimpleNamespace(sum_floor_v=0.5, sum_ceiling_frac=0.9)
    check_sum_ceiling(model, 0.45, 0.5)              # service/command_guard.py:331
    with pytest.raises(GuardRefused):
        check_sum_ceiling(model, 0.46, 0.5)
    with pytest.raises(GuardRefused):
        check_sum_ceiling(model, 0.1, 0.49)


# ---------------------------------------------------------------- channel strings

def test_channel_strings():
    assert hw_approach._DEFL == "Deflection"         # hardware/hw_approach.py:19
    assert hw_approach._AMPL == "Amplitude"          # hardware/hw_approach.py:42
    assert hw_approach.SUM_CHANNEL == "DetectorSum"  # hardware/hw_approach.py:25
    assert hw_approach._Z_OUT == "Height"            # hardware/hw_approach.py:20
    assert hw_scan._HEIGHT == "ZSensor"              # service/hw_scan.py:29
    assert hw_scan._ERR == "Deflection"              # service/hw_scan.py:30
    assert hw_scan._BIAS == "UserIn2"                # service/hw_scan.py:31
    assert hw_scan._Z_OUT == "Height"                # service/hw_scan.py:60
    assert channels.Z_OUT == "Height"                # hardware/channels.py
    assert instrument_model._DEFAULT_Z_VERIFY_CHANNELS == ("ZSensor", "Deflection")  # :43
    assert xy_transit._X_SETPOINT == "PIDSLoop.0.Setpoint"   # hardware/xy_transit.py:69
    assert xy_transit._Y_SETPOINT == "PIDSLoop.1.Setpoint"
    assert hardware_arc.CYPHER_MICROSCOPE_ID == 4    # hardware/hardware_arc.py:49
    # the Cypher engage output follows the platform, not a literal
    stub, hw = _connect(cypher=True)
    try:
        hw_approach.hw_engage(hw, setpoint=1.0, pgain=0.0, igain=1.0, settle_s=0.0)
    finally:
        hw.close()
        stub.stop()
    assert _z_loop(stub)[5] == "Height"


def test_channel_table_is_pinned_name_by_name():
    # Written out in full so a changed or moved name fails, not just a changed shape.
    assert channels.READ_ALIASES == {
        "height": "Height",
        "deflection": "Deflection",
        "deflection_value": "DeflectionValue",
        "z_sensor": "ZSensor",
        "x_sensor": "XSensor",
        "y_sensor": "YSensor",
        "detector_sum": "DetectorSum",
        "amplitude": "Amplitude",
        "amplitude_lowercase": "amplitude",
        "height_lowercase": "height",
        "output_a": "Output.A",
        "output_x": "Output.X",
        "output_y": "Output.Y",
        "user_in2": "UserIn2",
        "lateral": "Lateral",
        "input_a": "Input.A",
        "input_fast": "Input.Fast",
    }
    assert channels.Z_OUT == "Height"
