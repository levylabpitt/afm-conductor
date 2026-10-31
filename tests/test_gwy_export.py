"""Native Gwyddion (.gwy) export of a RAW capture.

`write_gwy` must emit a spec-valid native file: the `GWYP` magic, one serialized
`GwyContainer` whose components are `/i/data` (a `GwyDataField` object) and
`/i/data/title` (a string) per channel, each field carrying xres/yres/xreal/yreal,
`si_unit_xy`/`si_unit_z` (`GwySIUnit` objects), and a row-major double `data` array.
`export_capture_gwy` must project a multi-channel capture into ONE `.gwy` -- ZSensor
trace+retrace in metres, other channels raw volts -- and litho readbacks as thin
`1 x N` fields. The round-trip here hand-parses the bytes back (no gwy dependency).
"""

from __future__ import annotations

import struct
from types import SimpleNamespace

import numpy as np

from afm_litho.io.gwy import export_capture_gwy, write_gwy


# --- a minimal hand parser that walks the exact serialization format ---------

def _parse_components(blob: bytes) -> dict:
    comps: dict = {}
    off = 0
    while off < len(blob):
        end = blob.index(b"\x00", off)
        name = blob[off:end].decode("ascii")
        off = end + 1
        tchar = chr(blob[off])
        off += 1
        if tchar == "i":
            (v,) = struct.unpack_from("<i", blob, off)
            off += 4
        elif tchar == "d":
            (v,) = struct.unpack_from("<d", blob, off)
            off += 8
        elif tchar == "b":
            v = bool(blob[off])
            off += 1
        elif tchar == "s":
            e = blob.index(b"\x00", off)
            v = blob[off:e].decode("utf-8")
            off = e + 1
        elif tchar == "o":
            v, off = _parse_object(blob, off)
        elif tchar == "D":
            (count,) = struct.unpack_from("<I", blob, off)
            off += 4
            v = np.frombuffer(blob[off:off + 8 * count], dtype="<f8").copy()
            off += 8 * count
        else:
            raise AssertionError(f"unknown type char {tchar!r}")
        comps[name] = v
    return comps


def _parse_object(buf: bytes, off: int):
    """Return ((type_name, components_dict), new_off)."""
    end = buf.index(b"\x00", off)
    type_name = buf[off:end].decode("ascii")
    off = end + 1
    (size,) = struct.unpack_from("<I", buf, off)
    off += 4
    comps = _parse_components(buf[off:off + size])
    return (type_name, comps), off + size


def _read_gwy(path) -> list[dict]:
    """Parse a .gwy file into a list of channel dicts, sorted by /i/ index."""
    raw = path.read_bytes()
    assert raw[:4] == b"GWYP"
    (tn, container), off = _parse_object(raw, 4)
    assert tn == "GwyContainer"
    assert off == len(raw)                       # nothing trailing
    n = sum(1 for k in container if k.endswith("/data"))
    chans = []
    for i in range(n):
        (ftn, field) = container[f"/{i}/data"]
        assert ftn == "GwyDataField"
        (xytn, xy) = field["si_unit_xy"]
        (ztn, z) = field["si_unit_z"]
        assert xytn == "GwySIUnit" and ztn == "GwySIUnit"
        chans.append({
            "title": container[f"/{i}/data/title"],
            "xres": field["xres"], "yres": field["yres"],
            "xreal": field["xreal"], "yreal": field["yreal"],
            "xy_units": xy["unitstr"], "z_units": z["unitstr"],
            "data": field["data"].reshape(field["yres"], field["xres"]),
        })
    return chans


# --- write + hand-parse round-trip -------------------------------------------

def test_write_gwy_round_trips_two_channel_container(tmp_path):
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])          # 2 rows x 3 cols
    b = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])     # 3 rows x 2 cols
    channels = [
        {"name": "ZSensor trace", "grid2d": a, "x_real_m": 10e-6,
         "y_real_m": 20e-6, "z_units": "m"},
        {"name": "Deflection", "grid2d": b, "x_real_m": 5e-6,
         "y_real_m": 6e-6, "z_units": "V"},
    ]
    p = tmp_path / "sess.gwy"
    write_gwy(p, channels)

    got = _read_gwy(p)
    assert [c["title"] for c in got] == ["ZSensor trace", "Deflection"]

    z = got[0]
    assert z["xres"] == 3 and z["yres"] == 2
    assert z["xreal"] == 10e-6 and z["yreal"] == 20e-6
    assert z["xy_units"] == "m" and z["z_units"] == "m"
    assert np.allclose(z["data"], a)                          # row-major, first row top

    d = got[1]
    assert d["xres"] == 2 and d["yres"] == 3 and d["z_units"] == "V"
    assert np.allclose(d["data"], b)


def test_write_gwy_sanitizes_nan_and_inf(tmp_path):
    grid = np.array([[1.0, np.nan], [np.inf, -np.inf]])
    p = tmp_path / "nan.gwy"
    write_gwy(p, [{"name": "Z", "grid2d": grid, "x_real_m": 1e-6,
                   "y_real_m": 1e-6, "z_units": "m"}])
    data = _read_gwy(p)[0]["data"]
    assert np.isfinite(data).all()                            # spec forbids NaN/inf
    assert np.allclose(data, [[1.0, 0.0], [0.0, 0.0]])


# --- capture projection -> multi-channel .gwy --------------------------------

def test_export_scan_capture_writes_trace_and_retrace_per_channel(tmp_path):
    """A 2-D scan capture (rows*2*pixels per channel) -> ONE .gwy with trace + retrace
    for EVERY channel: ZSensor trace/retrace (metres) and Deflection trace/retrace (volts)."""
    cfg = SimpleNamespace(pixels=3, lines=2, size_um=6.0,
                          trace_retrace=True, direction="up")
    cal = {"z_lvdt_sens": 1.0, "z_lvdt_offset": 0.0}          # height = -v
    z = [10, 11, 12, 20, 21, 22,     # row0: trace 10,11,12  retrace 22,21,20 (flipped)
         30, 31, 32, 40, 41, 42]     # row1: trace 30,31,32  retrace 42,41,40
    e = [1, 2, 3, 7, 8, 9,           # row0: trace 1,2,3  retrace 9,8,7 (flipped)
         4, 5, 6, 0, 0, 0]           # row1: trace 4,5,6  retrace 0,0,0
    path = export_capture_gwy(tmp_path, "scan-1", {"ZSensor": z, "Deflection": e},
                              cfg, cal)
    assert path == tmp_path / "captures" / "scan-1.gwy"

    chans = _read_gwy(path)
    titles = [c["title"] for c in chans]
    assert titles == ["ZSensor trace", "ZSensor retrace",
                      "Deflection trace", "Deflection retrace"]
    for c in chans:
        assert c["xres"] == 3 and c["yres"] == 2                 # (rows, pixels)
        assert c["xy_units"] == "m"
        assert c["xreal"] == 6e-6 and c["yreal"] == 6e-6         # size_m square

    tr, re, err_tr, err_re = chans
    assert tr["z_units"] == "m"
    assert np.allclose(tr["data"], -np.array([[10, 11, 12], [30, 31, 32]]))
    assert re["z_units"] == "m"
    assert np.allclose(re["data"], -np.array([[22, 21, 20], [42, 41, 40]]))  # flipped
    assert err_tr["z_units"] == "V"
    assert np.allclose(err_tr["data"], [[1, 2, 3], [4, 5, 6]])   # raw volts, trace
    assert err_re["z_units"] == "V"
    assert np.allclose(err_re["data"], [[9, 8, 7], [0, 0, 0]])   # raw volts, retrace (flipped)


def test_export_litho_capture_writes_thin_1xN_fields(tmp_path):
    """A litho 1-D readback (length != rows*2*pixels) -> one 1 x N field per channel
    in the same .gwy; ZSensor is metres, Output.A raw volts."""
    n = 5
    cfg = SimpleNamespace(pixels=n, lines=1, size_um=float(n),
                          trace_retrace=False, direction="up")
    cap = {"ZSensor": [0.0, 0.1, 0.2, 0.3, 0.4],
           "Output.A": [-1.0, -1.0, 0.0, 1.0, 1.0]}
    path = export_capture_gwy(tmp_path, "litho-1", cap,
                              cfg, {"z_lvdt_sens": 1.0, "z_lvdt_offset": 0.0})
    assert path == tmp_path / "captures" / "litho-1.gwy"

    chans = {c["title"]: c for c in _read_gwy(path)}
    assert set(chans) == {"ZSensor", "Output.A"}
    a = chans["Output.A"]
    assert a["xres"] == 5 and a["yres"] == 1                    # thin 1 x N profile
    assert a["z_units"] == "V"
    assert np.allclose(a["data"].ravel(), [-1.0, -1.0, 0.0, 1.0, 1.0])
    zt = chans["ZSensor"]
    assert zt["yres"] == 1 and zt["z_units"] == "m"
    assert np.allclose(zt["data"].ravel(), -np.array([0.0, 0.1, 0.2, 0.3, 0.4]))


def test_export_empty_capture_writes_nothing(tmp_path):
    assert export_capture_gwy(tmp_path, "sess-3", {}, None, {}) is None
