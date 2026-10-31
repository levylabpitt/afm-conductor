"""While engaged the twin reports the Z drive and amplitude the instrument recorded."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from afm_litho import clock
from afm_litho.instrument import EngageAt
from afm_litho.instrument import twin as twin_module
from afm_litho.instrument.reports import TelemetryFrame
from afm_litho.instrument.twin import TwinInstrument
from afm_litho.run.model import Setpoint
from afm_litho.service import board
from afm_litho.service.twin_plugin import sample_to_npz
from afm_litho.twin.physics import Sample

IBW = Path(__file__).resolve().parent.parent / "data" / "samples" / "SA40656B0000.ibw"
RELATIVE = Setpoint("relative", 0.3)


@pytest.fixture(autouse=True)
def testing_clock():
    with clock.using("testing"):
        yield


def make_twin(tmp_path, maps=True):
    zdrive = np.add.outer(np.arange(10), np.arange(10)) + 70.0     # rows (Y) x pixels (X)
    kw = {"zdrive_v": zdrive, "amp_v": zdrive / 100} if maps else {}
    path = tmp_path / "s.npz"
    np.savez(path, height=np.zeros((10, 10)), size_um=10.0, x_offset_um=0.0, y_offset_um=0.0,
             angle_deg=0.0, **kw)
    return TwinInstrument(sample=path), zdrive


def test_ibw_maps_are_in_the_recorded_volt_ranges():
    sample = Sample.from_ibw(IBW)
    maps = sample.telemetry_maps
    assert maps["zdrive_v"].shape == maps["amp_v"].shape == sample.height_map.shape
    assert 73.0 <= np.nanmin(maps["zdrive_v"]) and np.nanmax(maps["zdrive_v"]) <= 84.0
    assert 0.2 <= np.nanmin(maps["amp_v"]) and np.nanmax(maps["amp_v"]) <= 0.65


def test_npz_carries_the_maps_transposed_like_height(tmp_path):
    sample = Sample.from_ibw(IBW)
    z = np.load(sample_to_npz(sample, tmp_path / "s.npz"))
    assert z["zdrive_v"].shape == z["height"].shape
    assert np.array_equal(z["zdrive_v"], sample.telemetry_maps["zdrive_v"].T)


def test_engaged_frame_reports_the_pixel_value(tmp_path):
    twin, zdrive = make_twin(tmp_path)
    try:
        assert twin.engage(EngageAt(RELATIVE, "ac")).done
        frame = twin.frame_at(clock.now())
        row, col, _ = twin.sample._index(0.0, 0.0)
        assert frame.z_drive_v == pytest.approx(zdrive[row, col])
        assert frame.amplitude_v == pytest.approx(zdrive[row, col] / 100)
    finally:
        twin.close()


def test_withdrawn_and_mapless_frames_are_as_before(tmp_path):
    twin, _ = make_twin(tmp_path)
    plain, _ = make_twin(tmp_path, maps=False)
    try:
        assert twin.frame_at(clock.now()).z_drive_v == twin_module.WITHDRAWN_Z_V
        assert plain.engage(EngageAt(RELATIVE, "contact")).done
        assert plain.frame_at(clock.now()).z_drive_v == twin_module.ENGAGED_Z_V
    finally:
        twin.close()
        plain.close()


def test_board_takes_the_plugin_frame():
    frame = TelemetryFrame(1.0, 2.0, 3.0, 78.5, 0.1, 10.0, 0.4)
    hub = SimpleNamespace(host=SimpleNamespace(
        _twin_plugin=SimpleNamespace(frame_at=lambda t: frame), _scan_row=4, arc=None))
    out = board._twin_hw_telemetry(hub, {"x_um": 9.0, "defl_v": 9.0, "amp_v": 9.0})
    assert (out["zdrive_v"], out["amp_v"], out["defl_v"], out["x_um"]) == (78.5, 0.4, 0.1, 2.0)
    assert out["sum_v"] == 10.0 and out["height_nm"] is None and out["row"] == 4


def test_a_twin_scan_line_carries_the_recorded_amplitude_as_its_error_row(tmp_path):
    from afm_litho.service.host.scan_pump import _twin_line_message
    twin, zdrive = make_twin(tmp_path)
    xs = np.array([-4.5, -3.5, 50.0])                  # two pixels on row 5, one off the image
    err = twin.sample.recorded_line(twin.sample.amp_v, xs, np.full(3, 0.5))
    assert err.tolist() == pytest.approx([zdrive[5, 0] / 100, zdrive[5, 1] / 100, 0.0])
    config = SimpleNamespace(mode="ac", direction="up", pixels=3, size_um=10.0,
                             angle_deg=0.0, lines=8, trace_retrace=True)
    msg = _twin_line_message(config, 0, np.zeros(3), None, err)
    assert msg["errName"] == "Amplitude" and msg["error"] == pytest.approx(err.tolist())
    assert _twin_line_message(config, 0, np.zeros(3), None)["error"] == [0.0, 0.0, 0.0]


def test_a_sample_without_an_amplitude_map_has_no_error_line(tmp_path):
    twin, _ = make_twin(tmp_path, maps=False)
    assert twin.sample.recorded_line(twin.sample.amp_v, np.zeros(2), np.zeros(2)) is None
