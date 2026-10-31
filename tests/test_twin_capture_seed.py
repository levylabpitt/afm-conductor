"""The twin seeded from a hardware scan capture shows the capture's own trace/retrace."""

from pathlib import Path

import numpy as np
import pytest

from afm_litho import clock
from afm_litho.instrument import ScanRaster
from afm_litho.instrument.twin import TwinInstrument
from afm_litho.instrument.twin_capture import (Z_LVDT_SENS_M_PER_V, capture_to_sample_npz,
                                               load_capture, split_trace_retrace)
from afm_litho.run.model import ScanFrame, Setpoint

CAPTURE = Path(__file__).parent.parent / "data" / "samples" / "20261022_193212.npz"
IBW_SEED = Path(__file__).parent.parent / "data" / "samples" / "SA40656B0000.ibw"


@pytest.fixture(autouse=True)
def testing_clock():
    with clock.using("testing"):
        yield


def test_loader_shapes_and_alignment():
    seed = load_capture(CAPTURE)
    assert seed["height"].shape == seed["retrace"].shape == (256, 256)
    assert seed["size_um"] == 30.0
    with np.load(CAPTURE) as z:
        raw = z["ZSensor"].reshape(256, 2, 256)
    expected = -raw[:, 0, :] * Z_LVDT_SENS_M_PER_V * 1e9
    assert np.allclose(seed["height"], expected)
    assert np.allclose(seed["retrace"], -raw[:, 1, ::-1] * Z_LVDT_SENS_M_PER_V * 1e9)


def test_split_flips_the_retrace_back():
    tr, re = split_trace_retrace(np.array([1, 2, 3, 4, 5, 6, 7, 8.0]), 4)
    assert tr.tolist() == [[1, 2, 3, 4]] and re.tolist() == [[8, 7, 6, 5]]


def test_a_non_square_capture_is_refused(tmp_path):
    bad = tmp_path / "bad.npz"
    np.savez(bad, ZSensor=np.zeros(100), Amplitude=np.zeros(100))
    with pytest.raises(ValueError):
        load_capture(bad)


def test_twin_scan_seeded_from_the_capture_has_a_real_retrace(tmp_path):
    seed = capture_to_sample_npz(CAPTURE, tmp_path / "seed.npz")
    twin = TwinInstrument(sample=seed, speed_factor=0)
    try:
        frame = ScanFrame("scan", 30.0, 256, 256, 0, (0.0, 0.0), False, "up", 1.0, True)
        result = twin.scan(ScanRaster(frame, "contact", Setpoint("relative", 0.3), True,
                                      ("Height",), transit_speed_um_s=50.0))
    finally:
        twin.close()
    trace, retrace = (r.volts for r in result.recorded)
    assert not np.allclose(trace, retrace)
    assert np.corrcoef(trace.ravel(), retrace.ravel())[0, 1] < 0.999


@pytest.mark.skipif(not IBW_SEED.exists(), reason="no ibw seed")
def test_ibw_style_seed_has_no_retrace_and_copies_the_trace(tmp_path):
    path = tmp_path / "plain.npz"
    np.savez(path, height=np.random.default_rng(0).normal(size=(20, 20)), size_um=5.0,
             x_offset_um=0.0, y_offset_um=0.0, angle_deg=0.0)
    twin = TwinInstrument(sample=path, speed_factor=0)
    try:
        assert twin.sample.retrace is None
        frame = ScanFrame("scan", 5.0, 5, 5, 0, (0.0, 0.0), False, "up", 1.0, True)
        result = twin.scan(ScanRaster(frame, "contact", Setpoint("relative", 0.3), True,
                                      ("Height",), transit_speed_um_s=50.0))
    finally:
        twin.close()
    trace, retrace = (r.volts for r in result.recorded)
    assert np.array_equal(trace, retrace)
