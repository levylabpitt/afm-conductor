"""`io/png.py:write_grey8_png` -- the stdlib-only 8-bit greyscale PNG encoder
for scan/channel frames. Decodes with a small hand-rolled parser (stdlib
struct + zlib only, matching the encoder's own dependencies) so this test
verifies the actual bytes on disk, not just that numpy round-trips through
itself.

ORIENTATION is the load-bearing behaviour: the cockpit (`web/ec-main.js`
`buildScanImg`) draws grid row R-1 at the top, so `flip_rows=True` (the
default and the only mode any caller uses) must write the array's LAST row
first.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from afm_litho.io.png import write_grey8_png


def _decode_grey8_png(path) -> np.ndarray:
    """A minimal PNG reader for exactly what write_grey8_png produces: 8-bit
    greyscale, no interlace, filter type 0 (none) on every scanline. Returns
    a (rows, cols) uint8 array. Raises if the file isn't shaped that way --
    a real chunk-parse round trip, not a numpy-only check."""
    data = Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos = 8
    chunks = {}
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        crc_stored = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])[0]
        assert zlib.crc32(tag + payload) & 0xFFFFFFFF == crc_stored, f"bad CRC on {tag!r}"
        if tag == b"IDAT":
            chunks.setdefault(tag, bytearray()).extend(payload)
        else:
            chunks[tag] = payload
        pos += 12 + length
        if tag == b"IEND":
            break
    assert b"IHDR" in chunks and b"IDAT" in chunks and b"IEND" in chunks
    width, height, bitdepth, colortype, comp, filt, interlace = struct.unpack(
        ">IIBBBBB", chunks[b"IHDR"])
    assert bitdepth == 8 and colortype == 0 and comp == 0 and filt == 0 and interlace == 0
    raw = zlib.decompress(bytes(chunks[b"IDAT"]))
    stride = width + 1                                    # +1 filter-type byte
    assert len(raw) == stride * height
    out = np.zeros((height, width), dtype=np.uint8)
    for r in range(height):
        row = raw[r * stride:(r + 1) * stride]
        assert row[0] == 0, "only filter type 0 (none) is written"
        out[r, :] = np.frombuffer(row[1:], dtype=np.uint8)
    return out


def test_basic_round_trip_shape_and_range(tmp_path):
    grid = np.linspace(0.0, 10.0, 4 * 6).reshape(4, 6)
    path = write_grey8_png(tmp_path / "a.png", grid)
    assert path == tmp_path / "a.png" and path.is_file()
    decoded = _decode_grey8_png(path)
    assert decoded.shape == (4, 6)
    assert decoded.min() == 0 and decoded.max() == 255    # min..max auto-scale


def test_orientation_row_r_minus_1_is_written_first(tmp_path):
    """flip_rows=True (default): the array's LAST row becomes the PNG's TOP
    scanline -- matching buildScanImg's `srcRow = R - 1 - r`."""
    rows, cols = 5, 3
    grid = np.zeros((rows, cols))
    for r in range(rows):
        grid[r, :] = r * 10.0                             # row r is constant value 10r

    decoded = _decode_grey8_png(write_grey8_png(tmp_path / "b.png", grid))
    lo, hi = 0.0, 40.0

    def expect_row(v):
        return round((v - lo) / (hi - lo) * 255)

    assert decoded[0, 0] == expect_row(40.0)               # PNG top = grid row 4 (R-1)
    assert decoded[-1, 0] == expect_row(0.0)                # PNG bottom = grid row 0


def test_flip_rows_false_keeps_grid_row_0_on_top(tmp_path):
    rows, cols = 3, 2
    grid = np.zeros((rows, cols))
    for r in range(rows):
        grid[r, :] = r
    decoded = _decode_grey8_png(write_grey8_png(tmp_path / "c.png", grid, flip_rows=False))
    assert decoded[0, 0] == 0                              # grid row 0 stayed on top
    assert decoded[-1, 0] == 255                           # grid row 2 (max) on bottom


def test_nan_and_inf_map_to_black(tmp_path):
    grid = np.array([[0.0, np.nan, 10.0], [np.inf, -np.inf, 5.0]])
    decoded = _decode_grey8_png(write_grey8_png(tmp_path / "d.png", grid, flip_rows=False))
    assert decoded[0, 1] == 0                              # NaN -> 0
    assert decoded[1, 0] == 0 and decoded[1, 1] == 0        # +/-inf -> 0
    assert decoded[0, 0] == 0                               # finite min (0.0) also scales to 0
    assert decoded[0, 2] == 255                             # finite max (10.0) scales to 255


def test_flat_array_falls_back_to_a_fixed_range_without_dividing_by_zero(tmp_path):
    grid = np.full((3, 3), 5.0)
    decoded = _decode_grey8_png(write_grey8_png(tmp_path / "e.png", grid))
    assert (decoded == 0).all()                            # span falls back to 1.0, (5-5)/1=0


def test_all_nonfinite_array_does_not_crash(tmp_path):
    grid = np.full((2, 2), np.nan)
    decoded = _decode_grey8_png(write_grey8_png(tmp_path / "f.png", grid))
    assert (decoded == 0).all()


def test_non_2d_grid_raises():
    with pytest.raises(ValueError):
        write_grey8_png("unused.png", np.zeros(4))


def test_write_is_atomic_no_leftover_tmp_file(tmp_path):
    grid = np.arange(6.0).reshape(2, 3)
    path = write_grey8_png(tmp_path / "g.png", grid)
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".") and p != path]
    assert leftovers == []
    assert path.is_file()
