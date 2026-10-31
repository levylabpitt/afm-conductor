"""The Igor COM bridge, exercised against a fake automation server.

No real Igor here: the bridge wraps anything exposing Execute2, so the suite
drives it with a stand-in that mimics pywin32's out-parameter return shape
(errorCode, errorMsg, history, results) -- the order shown in Igor's own
Utilities.bas example and read from IgorPro.tlb at runtime.
"""

from collections import deque

import pytest

from afm_litho.hardware.arc_errors import decode
from afm_litho.hardware.igor import ArcError, Igor, IgorError


class _FakeApp:
    """Mimics IgorPro.Application.Execute2 for testing.

    Records every command, and returns queued `results` strings so value() and
    string() can be checked without a live Igor.
    """

    def __init__(self):
        self.commands = []
        self.results = deque()
        self.error_code = 0
        self.error_msg = ""

    def push(self, results_text):
        self.results.append(results_text)

    def Execute2(self, allow_interrupt, binary, command):
        self.commands.append(command)
        results = self.results.popleft() if self.results else ""
        return (self.error_code, self.error_msg, "", results)


def test_execute_passes_the_command_through():
    app = _FakeApp()
    Igor(app).execute('td_WriteValue("PIDSLoop.2.Setpoint", 0.2)')
    assert app.commands == ['td_WriteValue("PIDSLoop.2.Setpoint", 0.2)']


def test_execute_raises_on_igor_error():
    app = _FakeApp()
    app.error_code = 1
    app.error_msg = "expected a number"
    with pytest.raises(IgorError) as exc:
        Igor(app).execute("bogus()")
    assert exc.value.code == 1
    assert "bogus()" in str(exc.value)


def test_execute_can_suppress_the_error_check():
    app = _FakeApp()
    app.error_code = 42
    r = Igor(app).execute("might_fail()", check=False)
    assert r.error_code == 42            # returned, not raised


def test_value_reads_back_through_fprintf_unit_0():
    app = _FakeApp()
    app.push("5.5e-08")
    v = Igor(app).value('GV("XLVDTSens")')
    # It must ask via fprintf to unit 0 -- the capture's readback channel.
    assert app.commands[0].startswith("fprintf 0")
    assert 'GV("XLVDTSens")' in app.commands[0]
    assert v == pytest.approx(5.5e-8)


def test_value_errors_when_igor_returns_nothing():
    with pytest.raises(IgorError):
        Igor(_FakeApp()).value('GV("Missing")')   # empty results queue -> ""


def test_string_returns_the_results_text():
    app = _FakeApp()
    app.push("MFP3D")
    assert Igor(app).string("root:Packages:MFP3D:Main:MicroscopeModel") == "MFP3D"


def test_not_connected_is_a_clear_error():
    with pytest.raises(RuntimeError, match="not connected"):
        Igor().execute("nop()")


# --- ARC error codes: the traceable, documented failures --------------------

def test_decode_maps_documented_codes():
    c = decode(10)
    assert (c.name, c.category) == ("ae_BadAlias", "ParmLib")
    assert "alias" in c.description.lower()
    assert decode(0).name == "ae_None"
    assert decode(1102).name == "ae_USB"          # DSP may have hung


def test_decode_flags_undocumented_and_app_defined():
    assert decode(99999).name == "ae_ApplicationDefined"   # >= 50000
    assert decode(77).name == "ae_Undocumented"            # nonzero, absent


def test_arc_call_reads_the_return_code_via_fprintf():
    app = _FakeApp()
    app.push("0")                                  # ae_None
    assert Igor(app).arc_call('td_WriteString("Event.0", "Once")') == 0
    assert app.commands[0].startswith('fprintf 0, "%d"')
    assert 'td_WriteString("Event.0", "Once")' in app.commands[0]


def test_arc_call_raises_a_decoded_traceable_error():
    app = _FakeApp()
    app.push("10")                                 # ae_BadAlias
    with pytest.raises(ArcError) as exc:
        Igor(app).arc_call('ir_SetPISLoop(1, ..., "Y%Output", ...)')
    assert exc.value.code == 10
    assert exc.value.arc.name == "ae_BadAlias"
    text = str(exc.value)
    assert "ae_BadAlias" in text and "Requested alias" in text
    assert "Y%Output" in text                      # the offending call is shown
    assert isinstance(exc.value, IgorError)        # catchable as the base error


def test_errors_carry_a_command_trace():
    app = _FakeApp()
    app.results.extend(["0", "0", "1117"])         # third call: ae_PIDSLoopNotConfigured
    ig = Igor(app)
    ig.arc_call("first_ok()")
    ig.arc_call("second_ok()")
    with pytest.raises(ArcError) as exc:
        ig.arc_call("third_bad()")
    text = str(exc.value)
    assert "ae_PIDSLoopNotConfigured" in text
    assert "recent commands" in text
    assert "first_ok()" in text and "third_bad()" in text
