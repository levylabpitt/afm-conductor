"""Native Gwyddion (.gwy) export of a RAW measured capture.

A scan captures MANY channels at once (ZSensor trace/retrace, Deflection, sometimes
Output.A). Gwyddion's single-field `.gsf` needed one file per channel; the native
`.gwy` container holds them all in one file as titled channels, so a session is a
single openable artifact.

`write_gwy(path, channels)` serializes a `GwyContainer` holding one `GwyDataField`
per channel; `export_capture_gwy` projects a RAW capture into those channels (via
`capture_channels_2d`) and writes `<session_id>.gwy`.

FORMAT (Gwyddion native, little-endian throughout; per the user-guide spec):
  * file magic: the 4 ASCII bytes ``GWYP`` (no NUL, no length),
  * then ONE serialized top-level object, a ``GwyContainer``.
  * OBJECT = type name as a NUL-terminated ASCII string, then a uint32 = the byte
    length of the components blob that follows (blob only -- excludes the name and
    this size field), then the components blob.
  * COMPONENT = name as a NUL-terminated ASCII string, then 1 ASCII type char, then
    the value: ``i`` int32, ``d`` double(8), ``b`` 1-byte bool, ``s`` NUL-terminated
    UTF-8 string, ``o`` a nested serialized OBJECT, ``D`` double array (uint32 count
    then that many doubles). Array type chars are uppercase.
  * the GwyContainer's components are the channel keys: for channel i (0-based),
    ``/i/data`` (``o`` = a GwyDataField) and ``/i/data/title`` (``s`` = the title).
  * a GwyDataField has ``xres`` ``i``, ``yres`` ``i``, ``xreal`` ``d``, ``yreal`` ``d``,
    ``si_unit_xy`` ``o`` (a GwySIUnit whose single ``unitstr`` ``s`` = ``m``),
    ``si_unit_z`` ``o`` (a GwySIUnit whose ``unitstr`` = the channel's z unit), and
    ``data`` ``D`` = xres*yres doubles ROW-MAJOR (first row = top), in base SI.

Doubles must be finite (the spec forbids inf/NaN), so every ``data`` array is
sanitized (NaN/inf -> 0.0) before it is written.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .capture_grid import capture_channels_2d

_MAGIC = b"GWYP"


def _ser_value(tchar: str, value) -> bytes:
    if tchar == "i":
        return struct.pack("<i", int(value))
    if tchar == "d":
        return struct.pack("<d", float(value))
    if tchar == "b":
        return struct.pack("<B", 1 if value else 0)
    if tchar == "s":
        return str(value).encode("utf-8") + b"\x00"
    if tchar == "o":
        return bytes(value)                       # already a serialized object
    if tchar == "D":
        # spec forbids inf/NaN in doubles -- sanitize padding/missing samples to 0.0.
        arr = np.nan_to_num(np.asarray(value, dtype="<f8"), nan=0.0,
                            posinf=0.0, neginf=0.0)
        arr = np.ascontiguousarray(arr.ravel())
        return struct.pack("<I", int(arr.size)) + arr.tobytes()
    raise ValueError(f"unknown component type char {tchar!r}")


def _ser_component(name: str, tchar: str, value) -> bytes:
    return name.encode("ascii") + b"\x00" + tchar.encode("ascii") + _ser_value(tchar, value)


def _ser_object(type_name: str, components: list[tuple[str, str, object]]) -> bytes:
    """Serialize an object: NUL-terminated type name, uint32 length of the components
    blob, then the blob. Nested objects are serialized first so the parent's length
    includes the fully-serialized child."""
    blob = b"".join(_ser_component(n, t, v) for n, t, v in components)
    return type_name.encode("ascii") + b"\x00" + struct.pack("<I", len(blob)) + blob


def _si_unit(unitstr: str) -> bytes:
    return _ser_object("GwySIUnit", [("unitstr", "s", unitstr)])


def _data_field(xres: int, yres: int, xreal: float, yreal: float,
                z_units: str, data) -> bytes:
    return _ser_object("GwyDataField", [
        ("xres", "i", int(xres)),
        ("yres", "i", int(yres)),
        ("xreal", "d", float(xreal)),
        ("yreal", "d", float(yreal)),
        ("si_unit_xy", "o", _si_unit("m")),
        ("si_unit_z", "o", _si_unit(z_units)),
        ("data", "D", data),
    ])


def write_gwy(path, channels: list[dict]) -> Path:
    """Write `channels` to `path` as a spec-valid native Gwyddion `.gwy` file.

    Each channel is a dict `{"name": title, "grid2d": (rows, cols) array (first row =
    top), "x_real_m": float, "y_real_m": float, "z_units": "m"|"V"}` (the shape of a
    `capture_channels_2d` entry). Values are stored verbatim in base SI (height in
    metres, voltages in volts); NaN/inf are sanitized to 0.0."""
    comps: list[tuple[str, str, object]] = []
    for i, ch in enumerate(channels):
        grid = np.asarray(ch["grid2d"], dtype=np.float64)
        if grid.ndim != 2:
            raise ValueError(f"channel {ch.get('name')!r} grid2d must be 2-D, "
                             f"got shape {grid.shape}")
        yres, xres = grid.shape
        field = _data_field(xres, yres, ch["x_real_m"], ch["y_real_m"],
                            ch["z_units"], grid)
        comps.append((f"/{i}/data", "o", field))
        comps.append((f"/{i}/data/title", "s", ch["name"]))
    blob = _MAGIC + _ser_object("GwyContainer", comps)
    path = Path(path)
    path.write_bytes(blob)
    return path


def export_capture_gwy(runtime_dir, session_id: str, cap: dict, config,
                       cal: dict, *, out_dir=None) -> Path | None:
    """Project a RAW capture into its channels and write ONE
    `<session_id>.gwy` next to the session's npz (`captures/`, or `out_dir` if given).
    `config`/`cal` are handed straight to `capture_channels_2d`. RAW/unflattened, base
    SI. Returns the path written, or `None` for an empty capture."""
    channels = capture_channels_2d(cap, config, cal)
    if not channels:
        return None
    out_dir = Path(out_dir) if out_dir is not None else Path(runtime_dir) / "captures"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{session_id}.gwy"
    return write_gwy(path, channels)
