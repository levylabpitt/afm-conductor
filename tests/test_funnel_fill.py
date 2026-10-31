"""Funnel outlines fill by the 2013 family, not a generic raster.

A funnel is a flat base and two curved sides meeting at an apex -- the shape
the SVG funnels have. Its fill is the interpolated family of the two boundary
curves' control points, cx = x*A + (1-x)*B, every sweep ending at the apex.
Caught rendering Mock Device.svg, where the funnels came out as a straight-line
raster instead.
"""

import numpy as np

from afm_litho.litho.bezier import parse_svg_path
from afm_litho.litho.compiler import LithoObject, expand
from afm_litho.litho.fill import fill_funnel_from_curves

# base line (0,0)->(4,0), then two curved sides up to the apex (2,6)
FUNNEL_D = "M 0,0 L 4,0 C 3.5,3 2.5,5 2,6 C 1.5,5 0.5,3 0,0 Z"


def _funnel():
    return parse_svg_path(FUNNEL_D)[0]


def test_as_funnel_finds_the_base_and_two_sides():
    f = _funnel().as_funnel()
    assert f is not None
    curve_a, curve_b, apex = f
    assert np.allclose(curve_a[3], apex)          # both sides run corner -> apex
    assert np.allclose(curve_b[3], apex)


def test_a_rectangle_is_not_a_funnel():
    rect = parse_svg_path("M 0,0 L 4,0 L 4,3 L 0,3 Z")[0]
    assert rect.as_funnel() is None               # four sides, no single base


def test_the_fill_is_one_continuous_serpentine():
    curve_a, curve_b, apex = _funnel().as_funnel()
    strokes = fill_funnel_from_curves(curve_a, curve_b, 0.25)
    assert len(strokes) == 1                       # one connected write -- no lifts
    path = strokes[0]
    b0, b1 = np.asarray(curve_b[0]), np.asarray(curve_a[0])
    assert np.allclose(path[0], b0, atol=1e-6)     # starts at one edge's base corner
    assert np.isfinite(path).all()
    apex_hits = int(np.sum(np.hypot(*(path - apex).T) < 1e-6))
    assert apex_hits >= 2                           # turns at the apex, up then down
    # both boundary curves are written: both base corners appear on the path
    assert float(np.min(np.hypot(*(path - b0).T))) < 1e-6
    assert float(np.min(np.hypot(*(path - b1).T))) < 1e-6


def test_it_returns_to_the_base_between_sweeps():
    """Serpentine: the path keeps coming back down to the base (the base steps),
    not fanning out once from a single base start."""
    curve_a, curve_b, _ = _funnel().as_funnel()
    path = fill_funnel_from_curves(curve_a, curve_b, 0.25)[0]
    b0, b1 = np.asarray(curve_b[0]), np.asarray(curve_a[0])
    seg = b1 - b0
    t = (path - b0) @ seg / (seg @ seg)
    proj = b0 + np.outer(np.clip(t, 0.0, 1.0), seg)
    on_base = np.hypot(*(path - proj).T) < 1e-6         # points sitting on the base
    assert int(on_base.sum()) >= 4                      # corners + interior base steps


def test_a_funnel_shape_compiles_to_that_family():
    bp = _funnel()
    obj = LithoObject("f", "shape",
                      [(float(x), float(y)) for x, y in bp.flatten(0.01)],
                      bezier=bp.to_dict(), voltage=8.0, fill_spacing_um=0.25)
    parts = expand(obj)
    assert len(parts) == 1                          # one continuous serpentine write
    assert len(parts[0].points_um) > 50             # ... and it is the whole fill
