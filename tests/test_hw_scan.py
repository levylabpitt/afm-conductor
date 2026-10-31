"""The hardware scan projection: captured rows become the {"type":"line"} stream the
browser renders. The chained driver is tested in test_hw_scan_chained.py."""


import numpy as np

from afm_litho.service.hw_scan import (
    _flatten1,
    _rows_to_lines,
)
from afm_litho.service.passes import ScanConfig

CAL = {"x_lvdt_sens": 7.46e-6, "x_lvdt_offset": -0.63,
       "y_lvdt_sens": 7.04e-6, "y_lvdt_offset": 0.07,
       "z_lvdt_sens": 5e-6, "z_lvdt_offset": 0.0}


def test_rows_to_lines_unsnakes_odd_rows():
    """Odd rows are captured reversed and un-snaked by parity BEFORE the display-only
    per-line flatten. Uses an asymmetric (non-linear) profile: a linear ramp would
    flatten to zero and hide the ordering, so the un-snake would be untestable. Row 1's
    captured reverse equals row 0, so after un-snaking + flatten they must match."""
    cfg = ScanConfig(size_um=10.0, pixels=4, lines=3)
    cal = {"z_lvdt_sens": 1.0, "z_lvdt_offset": 0.0}     # height_nm = -z * 1e9
    z = np.array([0., 0., 0., 3.,     # row 0 (even): as captured
                  3., 0., 0., 0.,     # row 1 (odd): captured reverse of row 0
                  0., 3., 0., 0.])    # row 2 (even)
    msgs = _rows_to_lines(z, np.zeros(12), 0, 3, cfg, cal)
    # display height = _flatten1(physical height) of the UN-SNAKED row
    exp = lambda vs: [round(float(x), 3) for x in _flatten1(-np.asarray(vs, float) * 1e9)]
    assert msgs[0]["height"] == exp([0, 0, 0, 3])        # even row: as captured
    assert msgs[1]["height"] == exp([0, 0, 0, 3])        # odd row un-reversed -> matches row 0
    assert msgs[2]["height"] == exp([0, 3, 0, 0])


def test_rows_to_lines_splits_trace_and_retrace():
    """trace_retrace: each captured line is forward (trace) then reverse (retrace).
    The projection returns the forward sweep as `height` and the reverse sweep flipped
    back to the same spatial order as `retrace`, so on a symmetric surface they match."""
    cfg = ScanConfig(size_um=10.0, pixels=3, lines=1, trace_retrace=True)
    cal = {"z_lvdt_sens": -1e-9, "z_lvdt_offset": 0.0}   # height_nm == z here
    # forward positions 0,1,2 -> [10,25,30]; reverse is captured at positions 2,1,0
    # -> [30,25,10], which flipped is [10,25,30] again. Non-linear (25 != midpoint) so
    # the display-only per-line flatten does not zero it -- trace & retrace still align.
    z = np.array([10., 25., 30., 30., 25., 10.])
    msgs = _rows_to_lines(z, np.zeros(6), 0, 1, cfg, cal, trace_retrace=True)
    exp = [round(float(x), 3) for x in _flatten1(np.array([10., 25., 30.]))]
    assert len(msgs) == 1
    assert msgs[0]["trace_retrace"] is True
    assert msgs[0]["height"] == exp                       # trace (forward), display-flattened
    assert msgs[0]["retrace"] == exp                      # retrace flipped, aligned, same flatten
    assert msgs[0]["pixels"] == 3                         # reported per-line pixels, not 2x


def test_save_and_load_capture_round_trip(tmp_path):
    from afm_litho.service.captures import load_capture, save_capture
    cap = {"Output.Z": np.arange(10.0), "Deflection": np.zeros(10)}
    ref = save_capture(tmp_path, "scan-123", cap)
    assert ref == "captures/scan-123.npz"
    back = load_capture(tmp_path, "scan-123")
    assert set(back) == {"Output.Z", "Deflection"}
    assert np.allclose(back["Output.Z"], np.arange(10.0))
    assert load_capture(tmp_path, "missing") == {}        # absent -> {}


def test_save_capture_empty_is_noop(tmp_path):
    from afm_litho.service.captures import load_capture, save_capture
    assert save_capture(tmp_path, "x", {}) == ""
    assert load_capture(tmp_path, "x") == {}
