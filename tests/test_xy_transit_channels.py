"""Which channels `xy_transit` reads and arms, observed on the wire the stub records.

Wrong wiring here (a swapped input alias) is the 2026-09-25 bug class; the channel table
alone cannot catch it because every name in the table is valid."""

from __future__ import annotations

import pytest

from afm_litho.hardware.hardware_arc import HardwareARC
from afm_litho.hardware.igor_stub import FakeIgorBridge
from afm_litho.hardware.xy_transit import xy_transit

CAL = {"x_lvdt_sens": 7.46e-6, "x_lvdt_offset": -0.63,
       "y_lvdt_sens": 7.04e-6, "y_lvdt_offset": 0.07,
       "z_lvdt_sens": 5e-6, "z_lvdt_offset": 0.0}
_FAST = dict(withdraw_max_s=0.0, withdraw_settle_s=0.0, ramp_settle_s=0.0, cmd_gap_s=0.0)


def _run(mode: str, *, engaged: bool, reapproach: bool):
    stub = FakeIgorBridge({"XSensor": 2.0, "YSensor": 2.0}, done_delay=0.0)
    hw = HardwareARC.connect("127.0.0.1", stub.serve_in_thread())
    try:
        xy_transit(hw, CAL, 10.0, 10.0, engaged=engaged, reapproach=reapproach,
                   setpoint=0.3, mode=mode, **_FAST)
    finally:
        hw.close()
        stub.stop()
    return stub


def _getvalue_names(stub) -> list[str]:
    return [e["args"][1] for e in stub.tap.reads if e["verb"] == "GETVALUE"]


def test_the_position_reads_are_xsensor_and_ysensor():
    names = _getvalue_names(_run("contact", engaged=False, reapproach=False))
    assert names and set(names) == {"XSensor", "YSensor"}


def test_withdraw_verify_reads_deflection_and_the_height_alias():
    names = set(_getvalue_names(_run("contact", engaged=True, reapproach=False)))
    # Lowercase `height` is the withdraw verify's own alias, not a table name.
    assert names == {"XSensor", "YSensor", "Deflection", "height"}


@pytest.mark.parametrize("mode, feedback_input, sign", [
    ("contact", "Deflection", 1.0), ("ac", "Amplitude", -1.0)])
def test_the_reapproach_loop_is_armed_on_the_modes_feedback_input(mode, feedback_input, sign):
    stub = FakeIgorBridge({"XSensor": 0.0, "YSensor": 0.0}, done_delay=0.0)
    hw = HardwareARC.connect("127.0.0.1", stub.serve_in_thread())
    try:
        x_v = (10.0e-6 / CAL["x_lvdt_sens"] + CAL["x_lvdt_offset"])
        y_v = (10.0e-6 / CAL["y_lvdt_sens"] + CAL["y_lvdt_offset"])
        stub.values["XSensor"], stub.values["YSensor"] = x_v, y_v
        xy_transit(hw, CAL, 10.0, 10.0, engaged=False, reapproach=True, setpoint=0.3,
                   mode=mode, z_pgain=2.0, z_igain=3.0, **_FAST)
        # Commands are sent without an ack: one read round trip means the stub has seen them all.
        hw.td_ReadValue("XSensor")
    finally:
        hw.close()
        stub.stop()
    z_loops = [lp for lp in stub.loops if lp[0] == 2]
    assert len(z_loops) == 1
    loop, in_name, setpoint, p_gain, i_gain, out_name = z_loops[0]
    assert in_name == feedback_input
    assert (p_gain, i_gain) == (2.0 * sign, 3.0 * sign)
    assert out_name == "Height"
