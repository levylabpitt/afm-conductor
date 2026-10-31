"""Tip-voltage capture channel (UserIn2) on the litho WRITE and the SCAN.

The MEASURED tip bias (UserIn2, the front-panel BNC In2 the operator tees the bias line into)
is captured as an extra channel alongside the
existing ones -- for VERIFICATION against the COMMANDED bias (v_written), which is
untouched. These are the OFF-INSTRUMENT (Python) parts; the Igor arming/streaming
is fail-safe and hardware-verified by the operator.

Covered here:
  1. LithographyRun._await_completion / WriteResult.capture carries an "UserIn2"
     channel when the (fake) bridge streams a CAPTURE ... UserIn2 ... frame.
  2. save_capture / load_capture round-trips an "UserIn2" array verbatim.
  3. hw_scan's capture list includes "UserIn2" and a projected line carries the
     extra `tipv` field (both the direct projection and the real-backend path).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from afm_litho.service.hw_scan import _rows_to_lines
from afm_litho.service.passes import ScanConfig

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "SA40656B0002.ibw"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample .ibw not present")

CAL = {"x_lvdt_sens": 7.46e-6, "x_lvdt_offset": -0.63,
       "y_lvdt_sens": 7.04e-6, "y_lvdt_offset": 0.07,
       "z_lvdt_sens": 5e-6, "z_lvdt_offset": 0.0}


# --- 1. WRITE: the measured tip bias rides back on the WriteResult.capture -----

@pytest.mark.skipif(not SAMPLE.exists(), reason="sample .ibw not present")
def test_write_object_captures_input_a():
    """When the bridge streams a CAPTURE frame on channel UserIn2 (the measured
    tip-bias readback armed on the write's 3rd bank), the demux folds it into
    WriteResult.capture with no Python change -- proving the plumbing is channel-
    agnostic. The commanded bias (VLitho) is recorded separately and untouched."""
    from afm_litho.hardware.igor_stub import FakeIgorBridge
    from afm_litho.hardware.lithography_run import LithographyRun

    # the fake Igor "measures" a tip-bias readback (a ramp) on UserIn2, exactly as
    # AFM_LithoDone streams AFM_SendCapture(..., "UserIn2", "root:VcapLitho").
    stub = FakeIgorBridge.from_ibw(
        SAMPLE, done_delay=0.0,
        capture=lambda name, cap_len: {"UserIn2": [0.3] * cap_len})
    port = stub.serve_in_thread()
    try:
        run = LithographyRun.connect("127.0.0.1", port, settle_move_s=0.0,
                                     cmd_gap_s=0.0)
        run.start()
        res = run.write_object("obj", [0.0, 1.0], [0.0, 0.0], [3.0, 3.0],
                               setpoint=0.4, speed=2.0)
    finally:
        stub.stop()
    assert "UserIn2" in res.capture
    assert len(res.capture["UserIn2"]) == res.cap_len
    assert res.capture["UserIn2"][:2] == [0.3, 0.3]
    assert not res.aborted
    run.close()


# --- 2. PERSIST: UserIn2 round-trips through the npz recorder verbatim ---------

def test_save_load_round_trips_input_a(tmp_path):
    from afm_litho.service.captures import load_capture, save_capture
    cap = {"ZSensor": np.arange(8.0), "Deflection": np.zeros(8),
           "UserIn2": np.full(8, 0.3)}
    ref = save_capture(tmp_path, "scan-tipv", cap)
    assert ref == "captures/scan-tipv.npz"
    back = load_capture(tmp_path, "scan-tipv")
    assert set(back) == {"ZSensor", "Deflection", "UserIn2"}
    assert np.allclose(back["UserIn2"], 0.3)      # dotted channel name kept verbatim


# --- 3. SCAN: the capture list carries UserIn2 and lines carry `tipv` ----------

def test_rows_to_lines_carries_tipv():
    """The projection emits the measured bias as `tipv` (trace) / `tipv_retrace`
    when a bias array is supplied, and omits it entirely when it is not (the fail-
    safe case where the channel was not ADC-routable)."""
    cfg = ScanConfig(size_um=10.0, pixels=4, lines=1)
    cal = {"z_lvdt_sens": 1.0, "z_lvdt_offset": 0.0}
    bias = np.array([0.3, 0.31, 0.29, 0.30])
    msgs = _rows_to_lines(np.zeros(4), np.zeros(4), 0, 1, cfg, cal, v=bias)
    assert msgs[0]["tipv"] == [0.3, 0.31, 0.29, 0.30]
    # absent bias -> no tipv key, existing channels intact
    plain = _rows_to_lines(np.zeros(4), np.zeros(4), 0, 1, cfg, cal)
    assert "tipv" not in plain[0]
    assert len(plain[0]["height"]) == 4


def test_rows_to_lines_tipv_unsnakes_odd_rows():
    """Odd rows are captured reversed; the bias is un-snaked in lockstep with the
    height/error, so tipv aligns spatially with them."""
    cfg = ScanConfig(size_um=10.0, pixels=4, lines=2)
    cal = {"z_lvdt_sens": 1.0, "z_lvdt_offset": 0.0}
    bias = np.array([0., 1., 2., 3., 4., 5., 6., 7.])
    msgs = _rows_to_lines(np.zeros(8), np.zeros(8), 0, 2, cfg, cal, v=bias)
    assert msgs[0]["tipv"] == [0.0, 1.0, 2.0, 3.0]      # even: as captured
    assert msgs[1]["tipv"] == [7.0, 6.0, 5.0, 4.0]      # odd: un-reversed
