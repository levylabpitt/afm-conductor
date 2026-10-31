"""A stdlib-only 8-bit greyscale PNG encoder for scan/channel frames.

No PNG/JPG writer exists in this codebase and Pillow/matplotlib are not
dependencies (docs/IMAGE_SAVING_NOTES.md) -- the cockpit paints frames straight
from the raw arrays it already has, and a stdlib PNG (zlib + struct) is ~30
lines. One PNG per captured channel (service/host/scan_pump.py): the height
display frame becomes `scans/png/<stem>_height.png`, and every other 2-D
channel in the raw capture becomes `scans/png/<stem>_<channel>.png`, each
auto-scaled on its own.

ORIENTATION: the cockpit (`web/ec-main.js` `buildScanImg`) draws grid row
`R-1` at the top of the canvas (`srcRow = R - 1 - r`). `flip_rows=True` (the
default) writes the array's LAST row first, so the PNG's top scanline is grid
row `R-1` -- pixel-identical to what the cockpit paints. The caller records
this as the sidecar's `"row0": "bottom"`; this module has no opinion on the
sidecar.
"""

from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path

import numpy as np

_SIG = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data +
            struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def write_grey8_png(path, grid2d, *, flip_rows: bool = True) -> Path:
    """Write `grid2d` (rows, pixels) as an 8-bit greyscale PNG at `path`.

    Auto-scaled to the finite min..max of the array (a flat or all-non-finite
    frame falls back to a fixed 0..1 range so the write never divides by
    zero); NaN and +/-inf map to 0 (black), matching the cockpit's
    `null -> 0` convention. `flip_rows=True` (default) writes row `R-1`
    first -- see the module docstring. Atomic (temp + `os.replace`), the same
    discipline as every other writer in this package. Returns the path
    written.
    """
    arr = np.asarray(grid2d, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"write_grey8_png needs a 2-D grid, got shape {arr.shape}")
    rows, cols = arr.shape
    finite = arr[np.isfinite(arr)]
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
    else:
        lo, hi = 0.0, 1.0
    span = (hi - lo) or 1.0
    scaled_f = np.clip(np.round((arr - lo) / span * 255.0), 0, 255)
    scaled = np.where(np.isfinite(arr), scaled_f, 0.0).astype(np.uint8)
    if flip_rows:
        scaled = scaled[::-1]

    raw = bytearray()
    for r in range(rows):
        raw.append(0)                      # PNG filter type 0 (none) per scanline
        raw.extend(scaled[r].tobytes())
    idat = zlib.compress(bytes(raw), level=6)

    # colour type 0 (greyscale), bit depth 8, no interlace.
    ihdr = struct.pack(">IIBBBBB", cols, rows, 8, 0, 0, 0, 0)
    png = (_SIG + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) +
           _chunk(b"IEND", b""))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(png)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    return path
