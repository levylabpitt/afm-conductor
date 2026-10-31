"""The out-wave chunker: split a too-long trajectory into DSP-loadable pieces.

The DSP caps a single out-wave at 87,000 points (the v19 vendor cMaxOutWave; the
v16 reference's 87,380 is too high for the rig). A longer trajectory has to be
cut into contiguous sub-waves and run back-to-back. These tests protect the
invariants that make that safe:

  - the pieces TILE the original range exactly -- contiguous, no gaps, no
    overlaps, in order -- so nothing is skipped and nothing is written twice;
  - every piece fits under the cap;
  - a raster's seams land on whole-line boundaries, so the pause while the next
    piece loads happens at the end of a line, not mid-stroke;
  - a general path's seams land on block boundaries;
  - slicing the parallel arrays and re-concatenating them reproduces the input
    exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from afm_litho.litho.chunker import (
    MAX_OUTWAVE_POINTS,
    chunk_arrays,
    chunk_indices,
)


def _assert_tiles(slices, n):
    """Every shared invariant: contiguous, in order, covering [0, n) exactly."""
    if n == 0:
        assert slices == []
        return
    assert slices[0][0] == 0
    assert slices[-1][1] == n
    prev = 0
    for start, stop in slices:
        assert start == prev, f"gap/overlap: {start} != {prev}"
        assert stop > start, "empty or reversed slice"
        prev = stop
    assert prev == n


def test_single_chunk_small_n():
    slices = chunk_indices(1000)
    assert slices == [(0, 1000)]


def test_at_the_cap_is_one_chunk():
    slices = chunk_indices(MAX_OUTWAVE_POINTS)
    assert slices == [(0, MAX_OUTWAVE_POINTS)]


def test_empty():
    assert chunk_indices(0) == []


def test_exact_multiple_raster():
    # 512 x 512 raster, one line = 512 points.
    n = 512 * 512  # 262144
    line_len = 512
    slices = chunk_indices(n, line_len=line_len)
    _assert_tiles(slices, n)

    lines_per_chunk = MAX_OUTWAVE_POINTS // line_len  # 170
    for start, stop in slices:
        length = stop - start
        assert length <= MAX_OUTWAVE_POINTS
        assert length >= 1
    # every non-final chunk is a whole number of lines, exactly the cap-fit count
    for start, stop in slices[:-1]:
        length = stop - start
        assert length % line_len == 0
        assert length == lines_per_chunk * line_len
    # seams land on line boundaries (including the final one, n is a multiple)
    for start, stop in slices:
        assert start % line_len == 0


def test_remainder_raster():
    # A raster whose total is NOT a whole multiple of the chunk length, and
    # whose final line is partial too, so the last chunk is a short remainder.
    line_len = 500
    n = 200 * line_len + 137  # 100137, last "line" partial on purpose
    slices = chunk_indices(n, line_len=line_len)
    _assert_tiles(slices, n)

    lines_per_chunk = MAX_OUTWAVE_POINTS // line_len  # 174
    for start, stop in slices[:-1]:
        length = stop - start
        assert length == lines_per_chunk * line_len
        assert length <= MAX_OUTWAVE_POINTS
    for start, stop in slices:
        assert stop - start <= MAX_OUTWAVE_POINTS
    # non-final seams still fall on line boundaries
    for start, _ in slices[1:]:
        # start of every non-first chunk is a multiple of the chunk length,
        # hence a multiple of line_len
        pass
    for start, stop in slices[:-1]:
        assert stop % line_len == 0


def test_general_block_aligned_split():
    block = 32
    n = 3 * MAX_OUTWAVE_POINTS + 777
    slices = chunk_indices(n, block=block)
    _assert_tiles(slices, n)

    step = (MAX_OUTWAVE_POINTS // block) * block
    for start, stop in slices[:-1]:
        length = stop - start
        assert length == step
        assert length % block == 0
        assert length <= MAX_OUTWAVE_POINTS
    # only the last may be a non-multiple remainder
    assert slices[-1][1] - slices[-1][0] <= MAX_OUTWAVE_POINTS


def test_line_len_nonpositive_is_general():
    n = 2 * MAX_OUTWAVE_POINTS
    assert chunk_indices(n, line_len=0) == chunk_indices(n)
    assert chunk_indices(n, line_len=-5) == chunk_indices(n)


def test_reassembly_round_trip():
    n = 250000
    line_len = 640
    x = np.arange(n, dtype=float)
    y = np.sin(np.arange(n) / 100.0)
    v = np.arange(n, dtype=np.int64) * 2

    chunks = chunk_arrays([x, y, v], line_len=line_len)

    rebuilt = [np.concatenate([c[k] for c in chunks]) for k in range(3)]
    assert np.array_equal(rebuilt[0], x)
    assert np.array_equal(rebuilt[1], y)
    assert np.array_equal(rebuilt[2], v)


def test_chunk_arrays_length_mismatch_raises():
    a = np.arange(10)
    b = np.arange(11)
    with pytest.raises(ValueError):
        chunk_arrays([a, b])


def test_chunk_arrays_empty_list():
    assert chunk_arrays([]) == []


def test_chunk_arrays_matches_indices():
    n = 200000
    a = np.arange(n)
    chunks = chunk_arrays([a], block=32)
    slices = chunk_indices(n, block=32)
    assert len(chunks) == len(slices)
    for (start, stop), chunk in zip(slices, chunks):
        assert np.array_equal(chunk[0], a[start:stop])


def test_oversized_single_line_fallback():
    # A single line is wider than the cap: honour tiling, fall back to
    # block-aligned sub-splitting within the line.
    block = 32
    line_len = MAX_OUTWAVE_POINTS + 5000  # one line > cap
    n = 3 * line_len
    slices = chunk_indices(n, line_len=line_len, block=block)
    _assert_tiles(slices, n)

    for start, stop in slices:
        assert stop - start <= MAX_OUTWAVE_POINTS
    # more slices than lines, since each oversized line was sub-split
    assert len(slices) > 3
    # within-line sub-splits are block-aligned except where a line boundary or
    # the end truncates them
    step = (MAX_OUTWAVE_POINTS // block) * block
    # the first sub-slice of the first line is a full block-aligned step
    assert slices[0] == (0, step)
