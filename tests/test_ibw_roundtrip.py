"""`write_ibw` must be the byte-exact inverse of `read_ibw`.

Covers the two shapes the file-saving stack emits: a 1-D IV wave (current on a
scaled bias axis, with a note + data units) and a multi-channel scan wave (Asylum
stacks channels in the 3rd dimension, listed in the note's `ChannelList`). Also
checks `export_capture_ibw` projects a RAW capture into one openable 3-D wave.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from afm_litho.io.ibw import export_capture_ibw, read_ibw, write_ibw


def test_write_read_1d_iv_wave_round_trips(tmp_path):
    bias = np.linspace(-0.5, 0.5, 64)
    current = 1e-9 * bias / 5.0e3                              # ohmic-ish nA current
    step = (bias[-1] - bias[0]) / (len(bias) - 1)
    note = "Device: KTO-QD\rTemperature: 0.023\r"
    p = tmp_path / "iv.ibw"
    write_ibw(p, current, name="IV", sfA=[step], sfB=[bias[0]],
              data_units="A", dim_units=["V"], note=note)

    w = read_ibw(p)
    assert w.name == "IV"
    assert w.data.shape == (64,)
    assert np.allclose(w.data, current)
    assert w.data_units == "A"
    assert w.dim_units[0] == "V"
    # scaled bias axis: value(i) = sfB + i*sfA reconstructs the original bias.
    assert np.isclose(w.sfA[0], step) and np.isclose(w.sfB[0], bias[0])
    recovered_bias = w.sfB[0] + np.arange(64) * w.sfA[0]
    assert np.allclose(recovered_bias, bias)
    assert w.note_dict["Device"] == "KTO-QD"
    assert w.note_dict["Temperature"] == "0.023"


def test_write_read_3d_scan_wave_round_trips(tmp_path):
    # (pixels, rows, n_channels) -- Asylum channel stacking in dim-3.
    px, rows, nch = 5, 3, 4
    rng = np.random.default_rng(0)
    data = rng.standard_normal((px, rows, nch)).astype(np.float64)
    names = ["ZSensor trace", "ZSensor retrace", "Deflection trace",
             "Deflection retrace"]
    p = tmp_path / "scan.ibw"
    write_ibw(p, data, name="scan0", sfA=[2e-6 / px, 2e-6 / rows, 1.0],
              sfB=[0.0, 0.0, 0.0], data_units="m", dim_units=["m", "m", ""],
              note="ChannelList: " + ";".join(names) + "\r")

    w = read_ibw(p)
    assert w.data.shape == (px, rows, nch)
    assert np.allclose(w.data, data)
    assert np.isclose(w.sfA[0], 2e-6 / px) and np.isclose(w.sfA[1], 2e-6 / rows)
    assert w.dim_units[:2] == ["m", "m"]
    assert w.channel_names == names                           # resolved from the note


def test_write_read_2d_wave_round_trips(tmp_path):
    data = np.arange(12, dtype=np.float64).reshape(4, 3)      # (dim0=4, dim1=3)
    p = tmp_path / "img.ibw"
    write_ibw(p, data, name="img")
    w = read_ibw(p)
    assert w.data.shape == (4, 3)
    assert np.allclose(w.data, data)                          # F-order preserved


def test_dtype_and_checksum(tmp_path):
    # float32 stays float32; the header checksum makes the 384-byte short-sum zero.
    import struct
    data = np.array([1, 2, 3, 4], dtype=np.float32)
    p = tmp_path / "f32.ibw"
    write_ibw(p, data, name="f32")
    w = read_ibw(p)
    assert w.data.dtype == np.float32 and np.allclose(w.data, data)
    header = p.read_bytes()[:384]
    assert sum(struct.unpack_from("<192h", header, 0)) & 0xFFFF == 0


def test_export_capture_ibw_stacks_channels(tmp_path):
    """A 2-D scan capture -> ONE 3-D .ibw with trace+retrace per channel listed in
    the note; ZSensor as metres, Deflection raw volts, base SI."""
    cfg = SimpleNamespace(pixels=3, lines=2, size_um=6.0,
                          trace_retrace=True, direction="up")
    cal = {"z_lvdt_sens": 1.0, "z_lvdt_offset": 0.0}
    z = [10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42]
    e = [1, 2, 3, 7, 8, 9, 4, 5, 6, 0, 0, 0]
    path = export_capture_ibw(tmp_path, "scan-1", {"ZSensor": z, "Deflection": e},
                              cfg, cal)
    assert path == tmp_path / "captures" / "scan-1.ibw"

    w = read_ibw(path)
    assert w.data.ndim == 3
    px, rows, nch = w.data.shape
    assert (px, rows, nch) == (3, 2, 4)                       # pixels, rows, channels
    assert w.channel_names == ["ZSensor trace", "ZSensor retrace",
                               "Deflection trace", "Deflection retrace"]
    # ZSensor trace (channel 0) is height = -v (cal 1.0), as (pixels, rows).
    ztrace = w.data[:, :, 0]
    assert np.allclose(ztrace, -np.array([[10, 11, 12], [30, 31, 32]]).T)
    # Deflection trace (channel 2) kept raw volts.
    assert np.allclose(w.data[:, :, 2], np.array([[1, 2, 3], [4, 5, 6]]).T)
    # XY scaling is the real field size (6 um over 3 px, 2 rows).
    assert np.isclose(w.sfA[0], 6e-6 / 3) and np.isclose(w.sfA[1], 6e-6 / 2)


def test_export_empty_capture_returns_none(tmp_path):
    cfg = SimpleNamespace(pixels=3, lines=2, size_um=6.0,
                          trace_retrace=True, direction="up")
    assert export_capture_ibw(tmp_path, "sess", {}, cfg, {}) is None
