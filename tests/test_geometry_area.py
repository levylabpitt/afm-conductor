"""Pin polygon-area behaviour before folding the shoelace spellings into one.

fill._signed_area, BezierPath.area, and the zero-area guard in the region pen all
compute the same polygon area three different ways. These tests pin the two
OBSERVABLE surfaces -- the area a BezierPath reports (value and sign) and whether
the region pen keeps or drops an outline -- so consolidating the internals onto a
single signed_area() can be shown to change nothing. (_centroid keeps its own
per-vertex cross product; it is not part of this.)
"""

import pytest

from afm_litho.litho.bezier import parse_svg_path
from afm_litho.litho.pens import _render_region


def _sq():                       # 2x2 square, area 4
    return parse_svg_path("M 0,0 L 2,0 L 2,2 L 0,2 Z")[0]


def _tri():                      # right triangle, area 6
    return parse_svg_path("M 0,0 L 4,0 L 0,3 Z")[0]


def test_bezier_area_magnitude_of_known_shapes():
    assert abs(_sq().area()) == pytest.approx(4.0, abs=1e-9)
    assert abs(_tri().area()) == pytest.approx(6.0, abs=1e-9)


def test_bezier_area_sign_flips_with_winding():
    fwd = _sq().area()
    rev = parse_svg_path("M 0,0 L 0,2 L 2,2 L 2,0 Z")[0].area()   # reversed winding
    assert fwd == pytest.approx(-rev, abs=1e-9)
    assert abs(fwd) == pytest.approx(4.0, abs=1e-9)


def _region_params():
    return {"voltage": 8.0, "speed_um_s": 2.0, "pitch_um": 0.5, "fill": "directional"}


def test_region_pen_keeps_a_real_region():
    out = _render_region([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
                         _region_params(), "sq")
    assert len(out) == 1 and out[0].kind == "shape"


def test_region_pen_rejects_a_zero_area_outline():
    collinear = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]   # no area
    assert _render_region(collinear, _region_params(), "deg") == []
