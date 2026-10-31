"""Pin the Z-servo arming so factoring out the repeated ir_SetPISLoop call
cannot change it.

approach / withdraw / transit / write all configure PIS loop 2 with the SAME
fixed shape -- Fast%Input -> Z%Output, igain 1200, output bounds -10..150,
running -- differing only in the deflection setpoint. These tests pin that shape
at the two direct entry points (approach, withdraw); the litho and transit
suites cover the streaming sites (tip-down, transit lift/land) end to end. If a
refactor that factors the repeated call into a helper drifts on any field, these
fail.
"""

import numpy as np

from afm_litho.service.passes import TwinInstrument, arm_z_servo
from afm_litho.twin.arc import SoftwareARC
from afm_litho.twin.physics import Sample, World


def _inst():
    s = Sample(height_map=np.zeros((64, 64)), extent=30e-6)
    return TwinInstrument(SoftwareARC(World(sample=s)))


def _assert_z_servo(arc, setpoint):
    lp = arc.loops[2]
    assert lp.input == "Fast%Input"
    assert lp.output == "Z%Output"
    assert (lp.pgain, lp.igain, lp.sgain) == (0.0, 1200.0, 0.0)
    assert (lp.out_min, lp.out_max) == (-10.0, 150.0)
    assert lp.running is True
    assert arc.params["PIDSLoop.2.Setpoint"] == setpoint


def test_approach_arms_the_z_servo_to_its_setpoint():
    inst = _inst()
    inst.approach(target_defl_v=0.2)
    _assert_z_servo(inst.arc, 0.2)


def test_approach_setpoint_tracks_the_argument():
    inst = _inst()
    inst.approach(target_defl_v=0.35)
    assert inst.arc.params["PIDSLoop.2.Setpoint"] == 0.35


def test_withdraw_arms_the_z_servo_open():
    inst = _inst()
    inst.withdraw()
    _assert_z_servo(inst.arc, -10.0)


# --- mode-aware Z arming (twin vocabulary). The twin resolves loop inputs by
# name: "Fast%Input" -> deflection, "Lockin.*" -> amplitude. Unlike the hardware
# loop (which negates the gain for AC), the twin encodes the AC sign inversion in
# its error direction (amp - target), so the gain STAYS positive; the channel is
# what makes the loop mode-correct. Default mode is contact (byte-identical).


def test_arm_z_servo_contact_uses_deflection_channel_positive_gain():
    inst = _inst()
    arm_z_servo(inst.arc, 0.2, mode="contact")
    lp = inst.arc.loops[2]
    assert lp.input == "Fast%Input" and lp.output == "Z%Output"
    assert (lp.pgain, lp.igain) == (0.0, 1200.0)   # positive
    assert lp.running is True


def test_arm_z_servo_default_mode_is_contact():
    inst = _inst()
    arm_z_servo(inst.arc, 0.2)                       # no mode kwarg
    assert inst.arc.loops[2].input == "Fast%Input"
    assert inst.arc.loops[2].igain == 1200.0


def test_arm_z_servo_ac_uses_amplitude_channel_gain_stays_positive():
    inst = _inst()
    arm_z_servo(inst.arc, 0.65, mode="ac")
    lp = inst.arc.loops[2]
    # twin amplitude channel (SoftwareARC.step: startswith "Lockin" -> amp branch)
    assert lp.input == "Lockin.0.R" and lp.output == "Z%Output"
    # gain positive: the twin inverts the AC error direction, not the gain
    assert (lp.pgain, lp.igain) == (0.0, 1200.0)
    assert lp.running is True
