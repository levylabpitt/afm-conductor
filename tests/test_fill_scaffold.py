"""Golden-master the parallel-scanline fills before extracting their shared setup.

fill_polygons and fill_zigzag begin with the same block: coerce the polys, rotate
so the fill axis is horizontal, inset the bounds, place the scan lines, and build
the inverse rotation. These pin exact outputs (captured from the current code) for
axis-aligned, rotated, and inset cases, plus the two empty-return branches, so
factoring that block into one helper can be shown to change nothing.
"""

import numpy as np

from afm_litho.litho.fill import fill_polygons, fill_zigzag

RECT = [(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0)]
SQ = [(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)]


def _same(out, expected):
    assert len(out) == len(expected)
    for s, e in zip(out, expected):
        assert np.allclose(np.asarray(s), np.asarray(e), atol=1e-6)


def test_fill_polygons_axis_aligned():
    out = fill_polygons([RECT], 2.0, angle_rad=0.0, inset_um=0.0)
    _same(out, [[[0.0, 0.0], [10.0, 0.0], [10.0, 2.0], [0.0, 2.0]]])


def test_fill_polygons_rotated_and_inset():
    out = fill_polygons([RECT], 1.0, angle_rad=np.pi / 4, inset_um=0.5)
    _same(out, [[[9.292893, 0.0], [10.0, 0.707107], [10.0, 2.105527],
                 [7.894473, 0.0], [6.496052, 0.0], [9.894473, 3.398421],
                 [9.097631, 4.0], [5.097631, 0.0], [3.69921, 0.0],
                 [7.69921, 4.0], [6.30079, 4.0], [2.30079, 0.0],
                 [0.902369, 0.0], [4.902369, 4.0], [3.503948, 4.0],
                 [0.105527, 0.601579], [0.0, 1.894473], [2.105527, 4.0],
                 [0.707107, 4.0], [0.0, 3.292893]]])


def test_fill_zigzag_axis_aligned():
    out = fill_zigzag([RECT], 2.0, angle_rad=0.0, inset_um=0.0)
    _same(out, [[[0.0, 0.0], [10.0, 0.0]], [[0.0, 2.0], [10.0, 2.0]]])


def test_fill_zigzag_rotated():
    out = fill_zigzag([SQ], 1.5, angle_rad=np.pi / 6, inset_um=0.0)
    _same(out, [
        [[4.877656, 0.0], [8.0, 1.802686]],
        [[1.755312, 0.0], [8.0, 3.605372]],
        [[0.0, 0.789256], [8.0, 5.408058]],
        [[0.0, 2.591942], [8.0, 7.210744]],
        [[0.0, 4.394628], [6.244688, 8.0]],
        [[0.0, 6.197314], [3.122344, 8.0]],
    ])


def test_empty_when_inset_swallows_the_shape():
    assert fill_polygons([RECT], 1.0, angle_rad=0.0, inset_um=5.0) == []
    assert fill_zigzag([RECT], 1.0, angle_rad=0.0, inset_um=5.0) == []


def test_empty_when_no_valid_polys():
    assert fill_polygons([[(0.0, 0.0), (1.0, 1.0)]], 1.0) == []   # < 3 points
    assert fill_zigzag([], 1.0) == []
