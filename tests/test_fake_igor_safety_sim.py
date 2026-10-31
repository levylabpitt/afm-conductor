"""The fake-Igor safety simulation (SLICES 0.5b): bias, XY and Z behave as the controller does.

Each test names the `.ipf` line or audit C section it relies on. Calls go to
`FakeIgorBridge._dispatch` with a recording link; time is the stub's simulated clock, so
nothing sleeps except where a completion thread must run.
"""

import threading
import time

import pytest

from afm_litho.hardware import bridge_protocol as P
from afm_litho.hardware.igor_stub import FakeIgorBridge

CYPHER = {"MicroscopeID": 4.0, "Height": 20.0}      # engaged: Z drive above the surface
MFP = {"MicroscopeID": 3.0, "Output.Z": 20.0}


class _Link:
    def __init__(self):
        self.sent: list[str] = []

    def send(self, line: str) -> None:
        self.sent.append(line)


def _stub(values=None, **kw):
    s = FakeIgorBridge(dict(values or {}), safety_sim=True, **kw)
    return s, _Link()


def _d(s, link, line):
    s._dispatch(line, link)


def _wait_for(link, prefix):
    for _ in range(100):
        if any(x.startswith(prefix) for x in link.sent):
            return
        time.sleep(0.01)
    raise AssertionError(f"no {prefix} frame")


def _wave(s, link, name, values):
    _d(s, link, P.cmd_wave(name, values))


# ------------------------------------------------------------------ off by default

def test_safety_sim_is_opt_in():
    assert FakeIgorBridge({}).safety_sim is None


# ------------------------------------------------------------------------- bias

def test_bias_follows_setvalue_and_nothing_else_zeroes_it():
    s, link = _stub(CYPHER)
    _d(s, link, P.cmd_setvalue("Output.A", 3.0))
    _d(s, link, P.cmd_servoloop(2, "Deflection", 0.5, 0, 1200, "Height", -10, 150))
    _d(s, link, P.cmd_stoploop(2))
    _d(s, link, P.cmd_ramp("PIDSLoop.0.Setpoint", 0.0, 1.0, 1.0))
    s.advance(5.0)
    assert s.safety_sim.bias_volts == 3.0
    assert [why for _, why, _ in s.safety_sim.bias_log] == ["write"]
    s.stop()


def test_bound_bias_wave_drives_output_a_while_it_plays_and_holds_its_end():
    s, link = _stub(MFP)
    _wave(s, link, "V", [1.0, 2.0, 3.0, 4.0])
    _d(s, link, P.cmd_outwave(2, "7", "Output.A", "V", -50))     # 50 ticks = 1 ms per point
    _d(s, link, P.cmd_fire("u", "w", "7"))
    assert s.safety_sim.bias_volts == 1.0
    s.advance(0.0025)
    assert s.safety_sim.bias_volts == 3.0
    s.advance(1.0)
    assert s.safety_sim.bias_volts == 4.0
    s.stop()


def test_bias_is_zeroed_when_a_general_fire_completes():
    # AFM_Done (afm_litho_run.ipf:512-535)
    s, link = _stub(MFP)
    _d(s, link, P.cmd_setvalue("Output.A", 2.0))
    _d(s, link, P.cmd_inwave(1, "7", "XSensor", 32, 1))
    _d(s, link, P.cmd_fire("u", "w", "7"))
    _wait_for(link, "DONE")
    assert s.safety_sim.bias_volts == 0.0
    assert s.safety_sim.bias_log[-1][1] == "done"
    s.stop()


def test_bias_is_zeroed_when_a_litho_write_completes():
    # AFM_LithoDone (:696-711)
    s, link = _stub(CYPHER, done_delay=0.2)
    for w in ("XLitho", "YLitho"):
        _wave(s, link, w, [0.0] * 32)
    _wave(s, link, "VLitho", [5.0] * 32)
    _d(s, link, P.cmd_armfire("u", "obj", -50, 32, 50))
    assert s.safety_sim.bias_volts == 5.0
    _wait_for(link, "DONE")
    assert s.safety_sim.bias_volts == 0.0
    assert [why for _, why, _ in s.safety_sim.bias_log] == ["litho done"]
    s.stop()


def test_chained_write_zeroes_the_bias_once_at_the_true_end_not_at_a_seam():
    # AFM_LithoAdvance (:1124-1153): the seam keeps the bias live
    s, link = _stub(CYPHER)
    for w in ("XLitho", "YLitho"):
        _wave(s, link, w, [0.0] * 128)
    _wave(s, link, "VLitho", [5.0] * 128)
    _wave(s, link, "Bounds", [0, 64, 128])
    _d(s, link, P.cmd_lithochain("u", "obj", -50, 50, "Bounds"))
    _wait_for(link, "DONE")
    assert [why for _, why, _ in s.safety_sim.bias_log] == ["litho chain end"]
    assert s.safety_sim.bias_volts == 0.0
    s.stop()


def test_abort_zeroes_the_bias():
    # AFM_Abort (:730)
    s, link = _stub(CYPHER)
    _d(s, link, P.cmd_setvalue("Output.A", 4.0))
    _d(s, link, P.cmd_abort())
    assert s.safety_sim.bias_volts == 0.0
    assert s.safety_sim.bias_log[-1][1] == "abort"
    s.stop()


def test_run_end_zeroes_the_bias():
    # AFM_LithoRunEnd (:135)
    s, link = _stub(CYPHER)
    _d(s, link, P.cmd_setvalue("Output.A", 4.0))
    s.stop()
    assert s.safety_sim.bias_volts == 0.0
    assert s.safety_sim.bias_log[-1][1] == "run end"


def test_abort_stops_a_playing_bias_wave():
    s, link = _stub(MFP)
    _wave(s, link, "V", [1.0, 2.0, 3.0, 4.0])
    _d(s, link, P.cmd_outwave(2, "7", "Output.A", "V", -50))
    _d(s, link, P.cmd_fire("u", "w", "7"))
    _d(s, link, P.cmd_abort())
    s.advance(1.0)
    assert s.safety_sim.bias_volts == 0.0
    s.stop()


# --------------------------------------------------------------------------- XY

def _play_xy(s, link, xs, ys):
    _wave(s, link, "X", xs)
    _wave(s, link, "Y", ys)
    _d(s, link, P.cmd_outpair(0, "7", "PIDSLoop.0.Setpoint", "X",
                              "PIDSLoop.1.Setpoint", "Y", -50))
    _d(s, link, P.cmd_fire("u", "w", "7"))


def test_xy_is_at_rest_until_something_moves_it():
    s, link = _stub(MFP)
    s.advance(10.0)
    assert not s.safety_sim.xy_moving
    s.stop()


def test_xy_out_waves_move_the_sensors_and_xy_moving_says_so():
    s, link = _stub(MFP)
    _play_xy(s, link, [0.0, 1.0, 2.0, 3.0], [0.0, -1.0, -2.0, -3.0])
    assert s.safety_sim.xy_moving
    s.advance(0.0015)
    assert (s.values["XSensor"], s.values["YSensor"]) == (1.0, -1.0)
    assert s.safety_sim.xy_moving
    s.advance(1.0)
    assert (s.values["XSensor"], s.values["YSensor"]) == (3.0, -3.0)
    assert not s.safety_sim.xy_moving
    s.stop()


def test_abort_stops_xy_where_it_is():
    s, link = _stub(MFP)
    _play_xy(s, link, [0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0])
    s.advance(0.0015)
    _d(s, link, P.cmd_abort())
    s.advance(1.0)
    assert s.values["XSensor"] == 1.0
    assert not s.safety_sim.xy_moving
    s.stop()


def test_xy_setpoint_ramp_moves_at_the_rate_then_stops():
    # IR_SetRamp on PIDSLoop.0.Setpoint, the hover move (audit C 1.F)
    s, link = _stub(MFP)
    _d(s, link, P.cmd_ramp("PIDSLoop.0.Setpoint", 2.0, 4.0, 10.0))
    assert s.safety_sim.xy_moving
    s.advance(0.25)
    assert s.values["XSensor"] == pytest.approx(1.0)
    s.advance(0.5)
    assert s.values["XSensor"] == pytest.approx(2.0)
    assert not s.safety_sim.xy_moving
    s.stop()


def test_xy_loop_rearmed_at_a_new_setpoint_slews_the_sensor_there():
    s, link = _stub(MFP)
    _d(s, link, P.cmd_servoloop(0, "XSensor", 1.0, 1, 1, "Output.X"))
    assert s.safety_sim.xy_moving
    s.advance(1.0)
    assert s.values["XSensor"] == 1.0
    assert not s.safety_sim.xy_moving
    s.stop()


def test_chained_scan_lines_move_xy_and_it_comes_to_rest():
    s, link = _stub(MFP)
    xs = [float(i) for i in range(64)]
    _wave(s, link, "SX", xs)
    _wave(s, link, "SY", [0.0] * 64)
    _d(s, link, P.cmd_scanchain("u", "SX", "SY", 32, 2, -50, 50))
    _wait_for(link, "SCANDONE")
    s.advance(10.0)
    assert s.values["XSensor"] == 63.0
    assert not s.safety_sim.xy_moving
    s.stop()


# ------------------------------------------------------------------- Z, Cypher

PARK = P.cmd_servoloop(2, "Deflection", -10, 0, 1000, "Height", 0, 150)


def test_cypher_rearm_with_output_min_zero_parks_height_at_zero_volts():
    # AFM_SafePark's Cypher branch (:796-805)
    s, link = _stub(CYPHER)
    _d(s, link, PARK)
    s.advance(0.05)
    assert 0.0 < s.values["Height"] < 20.0       # on its way
    assert not s.safety_sim.tip_off_surface
    s.advance(1.0)
    assert s.values["Height"] == 0.0
    assert s.safety_sim.tip_off_surface
    s.stop()


def test_cypher_stoploop_right_after_the_rearm_freezes_height_above_zero():
    # the park sequence: re-arm, then Python's STOPLOOP 2 one dispatch tick later
    s, link = _stub(CYPHER)
    _d(s, link, P.cmd_abort())      # AFM_SafePark: the re-arm
    _d(s, link, P.cmd_stoploop(2))
    s.advance(10.0)
    assert s.values["Height"] == pytest.approx(15.0)     # PARK_GAP_S x 100 V/s down from 20
    assert not s.safety_sim.tip_off_surface
    s.stop()


def test_park_gap_is_a_knob_and_zero_freezes_height_where_it_stood():
    s, link = _stub(CYPHER, park_gap_s=0.0)
    _d(s, link, P.cmd_abort())      # AFM_SafePark: the re-arm
    _d(s, link, P.cmd_stoploop(2))
    s.advance(10.0)
    assert s.values["Height"] == 20.0
    s.stop()


def test_height_ramp_after_the_stoploop_retracts_only_under_the_old_hypothesis():
    # the older hypothesis (a stopped loop frees Height); the rig 2026-10-27 says it does not
    s, link = _stub(CYPHER, height_ramp_after_stoploop=True)
    _d(s, link, P.cmd_abort())      # AFM_SafePark: the re-arm
    _d(s, link, P.cmd_stoploop(2))
    _d(s, link, P.cmd_ramp("Height", 0.0, 80.0, 2.0))
    s.advance(0.1)
    assert 0.0 < s.values["Height"] < 15.0
    s.advance(2.0)
    assert s.values["Height"] == 0.0
    assert s.safety_sim.tip_off_surface
    s.stop()


def test_cypher_height_ramp_while_loop_2_runs_is_ignored():
    # "does NOTHING" (:777-778): the running Height loop owns Height
    s, link = _stub(CYPHER)
    _d(s, link, P.cmd_servoloop(2, "Deflection", 0.5, 0, 1200, "Height"))
    _d(s, link, P.cmd_ramp("Height", 0.0, 80.0, 2.0))
    s.advance(5.0)
    assert s.values["Height"] == 20.0
    assert [r[0] for r in s.safety_sim.ignored_ramps] == ["Height"]
    s.stop()


def test_height_ramp_after_the_stoploop_is_inert_by_default():
    # rig 2026-10-27: with loop 2 stopped a Height ramp does not move Z; Z stays where the
    # re-arm had got to
    s, link = _stub(CYPHER)
    _d(s, link, P.cmd_abort())
    _d(s, link, P.cmd_stoploop(2))
    _d(s, link, P.cmd_ramp("Height", 0.0, 80.0, 2.0))
    s.advance(5.0)
    assert s.values["Height"] == pytest.approx(15.0)
    assert [r[0] for r in s.safety_sim.ignored_ramps] == ["Height"]
    s.stop()


def test_mfp_height_ramp_after_the_stoploop_is_inert_like_the_cypher():
    s, link = _stub(MFP)
    _d(s, link, P.cmd_abort())
    _d(s, link, P.cmd_stoploop(2))
    _d(s, link, P.cmd_ramp("Height", 0.0, 80.0, 2.0))
    s.advance(5.0)
    assert s.values["Output.Z"] == pytest.approx(15.0)
    assert [r[0] for r in s.safety_sim.ignored_ramps] == ["Height"]
    s.stop()


def test_cypher_output_z_ramp_does_nothing():
    s, link = _stub(CYPHER)
    _d(s, link, P.cmd_ramp("Output.Z", 0.0, 80.0, 2.0))
    s.advance(5.0)
    assert s.values["Height"] == 20.0
    assert [r[0] for r in s.safety_sim.ignored_ramps] == ["Output.Z"]
    s.stop()


def test_height_ramp_with_no_loop_running_moves_height_under_the_old_hypothesis():
    s, link = _stub(CYPHER, height_ramp_after_stoploop=True)
    _d(s, link, P.cmd_ramp("Height", 0.0, 80.0, 2.0))
    s.advance(0.1)
    assert s.values["Height"] == pytest.approx(12.0)
    s.stop()


def test_engaged_cypher_loop_holds_height():
    s, link = _stub(CYPHER)
    _d(s, link, P.cmd_servoloop(2, "Deflection", 0.5, 0, 1200, "Height"))
    s.advance(10.0)
    assert s.values["Height"] == 20.0
    s.stop()


def test_cypher_lift_rearm_drives_height_to_the_minus_ten_limit():
    # the lift after a write (audit C 1.F step 7): output limits default to -10/150
    s, link = _stub(CYPHER)
    _d(s, link, P.cmd_servoloop(2, "Deflection", -10, 0, 1200, "Height"))
    s.advance(10.0)
    assert s.values["Height"] == -10.0
    s.stop()


def test_cypher_abort_parks_height_at_zero_through_the_loop():
    s, link = _stub(CYPHER)
    _d(s, link, P.cmd_abort())
    assert not s.safety_sim.tip_off_surface
    s.advance(1.0)
    assert s.values["Height"] == 0.0 and s.values["ZSensor"] == 0.0
    assert s.safety_sim.tip_off_surface
    s.stop()


def test_td_stop_undefines_the_loops_and_z_holds_where_it_is():
    # Rig 2026-10-27 (Cypher): Cypher.PIDSLoop.2.Status 1 -> -1 after td_Stop().
    s, link = _stub(CYPHER)
    _d(s, link, P.cmd_servoloop(2, "Deflection", 0.5, 0, 1200, "Height"))
    s.safety_sim.td_stop()
    assert s.safety_sim.loop_state == {}
    s.advance(1.0)
    assert s.values["Height"] == 20.0
    s.stop()


# ---------------------------------------------------------------------- Z, MFP-3D

def test_mfp_stoploop_freezes_z():
    s, link = _stub(MFP)
    _d(s, link, P.cmd_servoloop(2, "Deflection", -10, 0, 1200, "Output.Z"))
    s.advance(0.05)
    _d(s, link, P.cmd_stoploop(2))
    frozen = s.values["Output.Z"]
    s.advance(10.0)
    assert s.values["Output.Z"] == frozen < 20.0
    s.stop()


def test_mfp_ramp_of_output_z_moves_at_the_rate():
    # AFM_RampOutput (:584-589)
    s, link = _stub(MFP)
    _d(s, link, P.cmd_ramp("Output.Z", 0.0, 80.0, 2.0))
    s.advance(0.125)
    assert s.values["Output.Z"] == pytest.approx(10.0)
    assert not s.safety_sim.tip_off_surface
    s.advance(0.2)
    assert s.values["Output.Z"] == 0.0 and s.safety_sim.tip_off_surface
    s.stop()


def test_mfp_ramp_gives_up_after_its_time_limit():
    s, link = _stub(MFP)
    _d(s, link, P.cmd_ramp("Output.Z", 0.0, 1.0, 2.0))
    s.advance(10.0)
    assert s.values["Output.Z"] == pytest.approx(18.0)
    s.stop()


def test_mfp_abort_rearms_the_loop_and_drives_z_to_zero():
    # AFM_SafePark re-arms loop 2 the same way on the MFP-3D (:826-832); Height is Output.Z there
    s, link = _stub(MFP)
    _d(s, link, P.cmd_servoloop(2, "Deflection", 0.5, 0, 1200, "Output.Z"))
    _d(s, link, P.cmd_abort())
    loop = s.safety_sim.loop_state[2]
    assert (loop["setpoint"], loop["out"], loop["out_min"]) == (-10, "Output.Z", 0)
    s.advance(1.0)
    assert s.values["Output.Z"] == 0.0 and s.safety_sim.tip_off_surface
    s.stop()


def test_mfp_set_down_poke_changes_the_loop_setpoint_without_rearming():
    s, link = _stub(MFP)
    _d(s, link, P.cmd_servoloop(2, "Deflection", 0.5, 0, 1200, "Output.Z", -10, 150))
    _d(s, link, P.cmd_setvalue("PIDSLoop.2.Setpoint", -10))
    s.advance(10.0)
    assert s.values["Output.Z"] == -10.0
    s.stop()


# ------------------------------------------------------------- real-time clock

def test_real_time_mode_follows_the_wall_clock_on_reads():
    s, link = _stub(MFP, safety_sim_real_time=True)
    _d(s, link, P.cmd_abort())                      # ramp 20 V at 80 V/s: 0.25 s
    time.sleep(0.4)
    _d(s, link, P.cmd_getvalue("u", "Output.Z"))
    assert link.sent[-1].endswith(" 0")
    s.stop()


# ------------------------------------------------- through the doors the host reads

def _read(s, link, name):
    _d(s, link, P.cmd_getvalue("u", name))
    return float(link.sent[-1].split(" ")[-1])


def test_advance_with_safety_sim_off_says_so():
    with pytest.raises(RuntimeError, match="safety_sim=True"):
        FakeIgorBridge({}).advance(1.0)


def test_a_stub_seeded_from_a_cypher_note_runs_cypher_safety_sim():
    s = FakeIgorBridge.from_note({"MicroscopeID": 4, "ScanSize": 1e-6}, safety_sim=True)
    assert s.safety_sim.is_cypher
    _d(s, _Link(), P.cmd_ramp("Output.Z", 0.0, 80.0, 2.0))
    assert s.safety_sim.ignored_ramps
    s.stop()


def test_telemetry_z_is_height_and_sees_the_cypher_park():
    s, _ = _stub(CYPHER, telem_hz=200.0)
    s.safety_sim.abort(stop=True)
    s.advance(1.0)
    telem = _Link()
    threading.Thread(target=s._telem_loop, args=(telem,), daemon=True).start()
    for _ in range(100):
        if telem.sent:
            break
        time.sleep(0.01)
    s._stop.set()
    assert float(telem.sent[0].split(" ")[3]) == 0.0


def test_safe_state_is_judged_through_getvalue_reads_alone():
    s, link = _stub(CYPHER)
    _play_xy(s, link, [0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0])
    _d(s, link, P.cmd_setvalue("Output.A", 2.0))
    _d(s, link, P.cmd_abort())
    assert _read(s, link, "Output.A") == 0.0         # bias zero
    assert _read(s, link, "Height") >= 1.0           # tip still on the surface
    x1 = _read(s, link, "XSensor")
    s.advance(0.5)
    x2 = _read(s, link, "XSensor")
    assert x1 == x2                                  # XY at rest between two reads
    s.advance(2.0)
    assert _read(s, link, "Height") < 1.0            # tip off the surface
    assert _read(s, link, "ZSensor") < 1.0
    s.stop()


def test_xy_motion_shows_as_two_differing_reads():
    s, link = _stub(MFP)
    _play_xy(s, link, [0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0])
    x1 = _read(s, link, "XSensor")
    s.advance(0.002)
    assert _read(s, link, "XSensor") != x1
    s.stop()


def test_zsensor_is_current_on_every_read_without_an_advance():
    s, link = _stub(MFP)
    s.values["Output.Z"] = 7.0
    assert _read(s, link, "ZSensor") == 7.0
    s.stop()


# ------------------------------------------- AFM_SafePark: flag, confirm, 5 s backup

def test_the_first_abort_stops_the_controller_and_a_repeat_does_not():
    s, link = _stub(CYPHER)
    _d(s, link, P.cmd_abort())
    s.advance(0.5)                                  # the re-arm is retracting
    loop = dict(s.safety_sim.loop_state[2])
    _d(s, link, P.cmd_abort())                      # a repeat abort mid-retract
    assert s.td_stops == 1 and s.abort_stopped
    assert s.safety_sim.loop_state[2] == loop       # the same -10 V re-arm, not undefined
    s.advance(2.0)
    assert s.values["Height"] == 0.0 and s.safety_sim.tip_off_surface
    s.stop()


def test_a_confirm_clears_the_flag_so_the_next_abort_stops_again():
    s, link = _stub(CYPHER)
    _d(s, link, P.cmd_abort())
    _d(s, link, P.cmd_abortconfirm())
    assert not s.abort_stopped and s._backup is None
    _d(s, link, P.cmd_abort())
    assert s.td_stops == 2
    s.stop()


def test_repeat_aborts_from_a_safe_state_are_harmless():
    s, link = _stub({"MicroscopeID": 4.0, "Height": 0.0})
    for _ in range(3):
        _d(s, link, P.cmd_abort())
    s.advance(1.0)
    assert s.td_stops == 1 and s.values["Height"] == 0.0 and s.values["Output.A"] == 0.0
    assert not s.safety_sim.xy_moving
    s.stop()


def test_igor_backup_clears_the_flag_when_height_is_down():
    from afm_litho import clock
    with clock.using(clock.TESTING):
        s, link = _stub(CYPHER)
        _d(s, link, P.cmd_abort())
        s.advance(2.0)                              # retracted, but no host confirm
        clock.sleep(5.0)
        assert s.backup_runs == ["confirmed"] and not s.abort_stopped
        assert s.vendor_withdraws == 0 and not s.abort_fault
        s.stop()


def test_igor_backup_runs_the_vendor_withdraw_and_sets_its_fault_when_height_is_up():
    from afm_litho import clock
    with clock.using(clock.TESTING):
        s, link = _stub(CYPHER)
        _d(s, link, P.cmd_abort())
        s.safety_sim.loop_state.clear()             # the retract loop never ran
        clock.sleep(5.0)
        assert s.backup_runs == ["fault"] and s.vendor_withdraws == 1
        assert s.abort_stopped                      # the flag stays until a confirm
        assert s.values["Height"] == 0.0            # the vendor Withdraw retracted
        _d(s, link, P.cmd_abortfault("u1"))
        _d(s, link, P.cmd_abortfault("u2"))         # read only: a lost reply loses nothing
        assert link.sent[-2:] == [P.msg_value("u1", 1.0), P.msg_value("u2", 1.0)]
        s.stop()


def test_a_fault_clear_lets_the_next_abort_stop_a_new_pass():
    """Review must-fix (2026-10-27): after a fault path the stopped flag stays set, so the
    operator's fault clear (CLEARFAULT) resets it and Igor's fault, and the abort of a new
    pass runs td_Stop again: its bias wave stops."""
    from afm_litho import clock
    with clock.using(clock.TESTING):
        s, link = _stub(CYPHER)
        _d(s, link, P.cmd_abort())
        s.safety_sim.loop_state.clear()             # the retract never confirms
        clock.sleep(5.0)
        assert s.abort_stopped and s.abort_fault    # the fault path
        _d(s, link, P.cmd_clearfault())
        assert not s.abort_stopped and not s.abort_fault and s.fault_clears == 1
        _d(s, link, P.cmd_abortfault("u3"))
        assert link.sent[-1] == P.msg_value("u3", 0.0)
        _wave(s, link, "V", [1.0, 2.0, 3.0, 4.0])   # a new pass plays its bias wave
        _d(s, link, P.cmd_outwave(2, "7", "Output.A", "V", -50))
        _d(s, link, P.cmd_fire("u", "w", "7"))
        _d(s, link, P.cmd_abort())
        assert s.td_stops == 2
        s.advance(1.0)
        assert s.safety_sim.bias_volts == 0.0
        _d(s, link, P.cmd_abortconfirm())
        s.stop()


def test_without_a_fault_clear_the_next_abort_skips_td_stop():
    """The hazard the fault clear removes: the flag left set by an unconfirmed abort."""
    s, link = _stub(CYPHER)
    _d(s, link, P.cmd_abort())
    _d(s, link, P.cmd_abort())
    assert s.td_stops == 1 and s.abort_stopped
    s.stop()


def test_a_confirmed_abort_leaves_no_backup_to_fire():
    from afm_litho import clock
    with clock.using(clock.TESTING):
        s, link = _stub(CYPHER)
        _d(s, link, P.cmd_abort())
        _d(s, link, P.cmd_abortconfirm())
        clock.sleep(10.0)
        assert s.backup_runs == [] and s.vendor_withdraws == 0
        s.stop()


def test_a_failed_string_read_answers_err_exception_not_the_error_text():
    """td_ReadString gives "*ERROR*" rather than raising; the .ipf answers ERR <uid> exception."""
    s, link = _stub(CYPHER)
    _d(s, link, P.cmd_getstr("u9", "PIDSLoop.5.OutputChannel"))
    assert link.sent[-1] == "ERR u9 exception: td_ReadString(PIDSLoop.5.OutputChannel) failed"
    s.stop()
