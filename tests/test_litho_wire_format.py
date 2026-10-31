"""The one-object write sequence, held against the annotated capture.

A litho write drives the same td_/ir_ commands the lab's LabVIEW sends
(docs/reference/igor-commands-commented.txt). This records the command-
generating calls a single wire emits and pins the parts that must not drift
from the capture -- most sharply the servo output channels, where "Y%Output"
(a Z-style name misapplied to Y) once stood in for "Output.Y" and only a
comparison against the capture caught it.
"""

import queue

import numpy as np

from afm_litho.litho.compiler import LithoObject, Pattern
from afm_litho.service.passes import LithoRun, TwinInstrument
from afm_litho.twin.arc import SoftwareARC
from afm_litho.twin.physics import Sample, World

# The calls that correspond to real Igor commands (step() is the sim clock).
_CMDS = ("ir_SetPISLoop", "td_xSetOutWavePair", "td_xSetOutWave",
         "td_xSetInWavePair", "td_WriteString", "td_WriteValue", "td_Stop")


def _record_one_wire():
    """Run one 6 um wire and return the ordered (name, args) command log."""
    arc = SoftwareARC(World(sample=Sample(height_map=np.zeros((64, 64)),
                                          extent=30e-6)))
    log = []

    def wrap(name):
        real = getattr(arc, name)

        def f(*args):
            log.append((name, args))
            return real(*args)
        return f

    for n in _CMDS:
        setattr(arc, n, wrap(n))
    pattern = Pattern("golden", [LithoObject("wire", "path",
                                             [(-3.0, 0.0), (3.0, 0.0)], voltage=8.0)])
    LithoRun(TwinInstrument(arc), pattern, queue.Queue()).run()
    return log


def test_the_servo_outputs_are_the_output_group_dotted_names():
    """X/Y piezo drives are Output.X / Output.Y, as in the capture -- never the
    Z-style %-form. This is the regression guard for the Y%Output fix."""
    log = _record_one_wire()
    loops = [args for name, args in log if name == "ir_SetPISLoop"]
    x = next(a for a in loops if a[2] == "XSensor")
    y = next(a for a in loops if a[2] == "YSensor")
    assert x[7] == "Output.X"
    assert y[7] == "Output.Y"
    # and nothing anywhere reaches for the misapplied Z-style names on X/Y
    strings = [a for _, args in log for a in args if isinstance(a, str)]
    assert not any(s in ("Y%Output", "X%Output") for s in strings)


def test_the_path_is_driven_the_way_the_capture_drives_it():
    """XY setpoints on out-bank 0, bias on Output.A out-bank 1, fired on Event.0
    -- the core arm/fire the capture uses at its cmds 57-59."""
    log = _record_one_wire()
    pairs = [args for name, args in log if name == "td_xSetOutWavePair"]
    assert any(a[2] == "PIDSLoop.0.Setpoint" and a[4] == "PIDSLoop.1.Setpoint"
               for a in pairs)
    singles = [args for name, args in log if name == "td_xSetOutWave"]
    assert any(a[2] == "Output.A" for a in singles)
    fires = [args for name, args in log if name == "td_WriteString"]
    assert any(a == ("Event.0", "Once") for a in fires)


def test_the_write_records_current_on_input_A():
    """The pass IS a scan: unlike the capture, the new stack arms an input bank
    to record during the write (PLAN.md: record on every write)."""
    log = _record_one_wire()
    caps = [args for name, args in log if name == "td_xSetInWavePair"]
    assert any(a[2] == "Input.A" for a in caps)
