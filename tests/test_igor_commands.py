"""Round-trip: the twin-side Igor command EMITTER (igor_commands) produces
strings the twin-side RECEIVER (igor.IgorSession) accepts and dispatches into
SoftwareARC.

This proves the operational bring-up map -- approach / withdraw / scan / tune /
force, the parameter writes, and the mode switch -- is wired end to end before
any real Igor or hardware is involved. On the instrument the same strings go to
IgorPro.Application.Execute2 instead of IgorSession.
"""

import numpy as np
import pytest

from afm_litho.twin import igor_commands as ig
from afm_litho.twin.arc import SoftwareARC
from afm_litho.twin.igor import IgorSession
from afm_litho.twin.physics import Sample, World


@pytest.fixture()
def session():
    sample = Sample(height_map=np.zeros((64, 64)), extent=30e-6, name="flat")
    return IgorSession(SoftwareARC(World(sample=sample)))


def run(session, cmds):
    for c in ([cmds] if isinstance(cmds, str) else cmds):
        session.execute(c)


def test_parameter_writes_land_in_globals(session):
    run(session, [ig.set_scan_size(5e-6), ig.set_scan_rate(1.0),
                  ig.set_scan_points(256), ig.set_scan_lines(128),
                  ig.set_scan_angle(15.0)])
    arc = session.arc
    assert arc.GV("ScanSize") == pytest.approx(5e-6)
    assert arc.GV("ScanRate") == pytest.approx(1.0)
    assert arc.GV("ScanPoints") == 256
    assert arc.GV("ScanLines") == 128
    assert arc.GV("ScanAngle") == pytest.approx(15.0)


def test_setpoint_is_mode_specific(session):
    run(session, ig.set_setpoint(0.2, "ac"))
    run(session, ig.set_setpoint(1.0, "contact"))
    arc = session.arc
    assert arc.GV("AmplitudeSetpointVolts") == pytest.approx(0.2)
    assert arc.GV("DeflectionSetpointVolts") == pytest.approx(1.0)


def test_mode_switch_fires_proc_and_sets_index(session):
    run(session, ig.set_mode("ac"))
    arc = session.arc
    assert arc.GV("ImagingMode") == 1.0
    assert ("MainPopupFunc", "ImagingModePopup", 0.0, "AC Mode") in arc.control_actions
    run(session, ig.set_mode("contact"))
    assert arc.GV("ImagingMode") == 0.0


@pytest.mark.parametrize("emit,proc,control", [
    (ig.approach,   "SimpleEngageMe", "SimpleEngageButton_0"),
    (ig.withdraw,   "DoScanFunc",     "Withdraw"),
    (ig.frame_up,   "DoScanFunc",     "UpScan_0"),
    (ig.frame_down, "DoScanFunc",     "DownScan_0"),
    (ig.stop,       "DoScanFunc",     "StopScan_0"),
])
def test_action_buttons_dispatch(session, emit, proc, control):
    run(session, emit())
    assert (proc, control, 0.0, "") in session.arc.control_actions


def test_stop_and_withdraw_is_two_commands(session):
    cmds = ig.stop_and_withdraw()
    assert len(cmds) == 2
    run(session, cmds)
    acts = session.arc.control_actions
    assert ("DoScanFunc", "StopScan_0", 0.0, "") in acts
    assert ("DoScanFunc", "Withdraw", 0.0, "") in acts


def test_tune_and_force_via_execute_control2(session):
    run(session, ig.tune())
    run(session, ig.force_curve())
    acts = session.arc.control_actions
    assert ("CantTuneFunc", "DoTuneAuto", 0.0, "") in acts
    assert ("DoForceFunc", "SingleForce", 0.0, "") in acts


def test_configure_scan_sets_everything(session):
    run(session, ig.configure_scan(size_m=2e-6, rate_hz=0.5, points=512,
                                   lines=512, angle_deg=0.0,
                                   setpoint_v=0.3, mode="ac"))
    arc = session.arc
    assert arc.GV("ScanSize") == pytest.approx(2e-6)
    assert arc.GV("ScanPoints") == 512
    assert arc.GV("ImagingMode") == 1.0
    assert arc.GV("AmplitudeSetpointVolts") == pytest.approx(0.3)


def test_endpoints_map(session):
    run(session, ig.ENDPOINTS["/api/scan/start"](direction="down"))
    assert ("DoScanFunc", "DownScan_0", 0.0, "") in session.arc.control_actions


def test_current_mode_unsupported():
    with pytest.raises(ValueError):
        ig.set_mode("current")
    with pytest.raises(ValueError):
        ig.set_setpoint(0.2, "current")
