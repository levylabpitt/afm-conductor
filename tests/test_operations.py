"""The hardware operations driver emits the right command strings through the
bridge. Uses a fake Igor (records execute()/answers value()) so it runs with no
pywin32 and no instrument -- the strings themselves are proven against the twin
in test_igor_commands.
"""

import pytest

from afm_litho.hardware import operations as op


class FakeIgor:
    def __init__(self, values=None):
        self.commands = []
        self._values = values or {}

    def execute(self, cmd, check=True):
        self.commands.append(cmd)

    def value(self, expr):
        return self._values.get(expr, 0.0)


def test_set_parameters_emits_and_orders():
    ig = FakeIgor()
    n = op.set_parameters(ig, mode="ac", size_m=5e-6, points=256, setpoint_v=0.2)
    assert n == 4
    assert 'PV("ScanSize",5e-06)' in ig.commands
    assert 'PV("ScanPoints",256)' in ig.commands
    assert 'PV("AmplitudeSetpointVolts",0.2)' in ig.commands
    # mode is set before the setpoint (which depends on it)
    assert next(i for i, c in enumerate(ig.commands) if "ImagingModePopup" in c) \
        < next(i for i, c in enumerate(ig.commands) if "SetpointVolts" in c)


def test_setpoint_without_mode_raises():
    with pytest.raises(ValueError):
        op.set_parameters(FakeIgor(), setpoint_v=0.2)


@pytest.mark.parametrize("action,expected", [
    (op.approach, 'SimpleEngageMe("SimpleEngageButton_0")'),
    (op.withdraw, 'DoScanFunc("Withdraw")'),
    (op.frame_up, 'DoScanFunc("UpScan_0")'),
    (op.frame_down, 'DoScanFunc("DownScan_0")'),
    (op.stop, 'DoScanFunc("StopScan_0")'),
])
def test_actions_emit(action, expected):
    ig = FakeIgor()
    action(ig)
    assert expected in ig.commands


def test_read_parameters_and_selftest():
    ig = FakeIgor(values={'GV("ScanSize")': 5e-6, 'GV("ImagingMode")': 1.0})
    got = op.read_parameters(ig)
    assert got["ScanSize"] == 5e-6
    assert got["ImagingMode"] == 1.0
    assert op.selftest(ig) == 5e-6


def test_verify_names_flags_missing():
    # exists() == 6 means the proc resolves; anything else (default 0.0) is missing.
    ig = FakeIgor(values={f'exists("{p}")': 6 for p in op._PROCS
                          if p != "MainPopupFunc"})
    res = op.verify_names(ig)
    assert res["DoScanFunc"] is True
    assert res["MainPopupFunc"] is False        # not in the values -> not found
    assert set(res) == set(op._PROCS)
