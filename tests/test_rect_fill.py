"""Rectangles raster parallel to their own edges, as a unidirectional zig zag.

A near-square rectangle has no stable principal axis, so the generic long-axis
fill comes out diagonal. And lithography writes direction-dependently, so every
row must go the SAME way (the AFM's trace/retrace) -- not a serpentine that
writes both ways. Caught on Mock Device.svg r0.
"""

import math

import numpy as np

from afm_litho.litho.bezier import BezierPath, parse_svg_path
from afm_litho.litho.compiler import LithoObject, expand
from afm_litho.litho.fill import fill_zigzag


def _js_rect_bezier(x0, y0, x1, y1):
    """The exact dict the cockpit's rect tool now attaches on commit (ec-main.js
    `rectBezier`): a closed 4-node CUSP rectangle, corners axis-aligned so
    `BezierPath.as_rect()` recognizes it. This is the JS<->Python contract -- if
    the drawn shape matches, a drawn rect === an imported rect."""
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return {"closed": True,
            "nodes": [{"x": x, "y": y, "h_in": None, "h_out": None, "kind": "cusp"}
                      for x, y in corners]}


def _rect(w=8.0, h=6.0, rot_deg=0.0):
    c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    pts = [(0, 0), (w, 0), (w, h), (0, h)]
    r = [(c * x - s * y, s * x + c * y) for x, y in pts]
    return parse_svg_path("M " + " L ".join(f"{x},{y}" for x, y in r) + " Z")[0]


def _angmod(a):
    return a % math.pi


def test_as_rect_finds_the_longer_edge_angle():
    assert abs(_rect(8, 6, 0).as_rect()) < 1e-6                    # width 8 -> horizontal
    assert abs(_angmod(_rect(8, 6, 30).as_rect()) - _angmod(math.radians(30))) < 1e-6


def test_a_funnel_is_not_a_rectangle():
    funnel = parse_svg_path("M 0,0 L 4,0 C 3.5,3 2.5,5 2,6 C 1.5,5 0.5,3 0,0 Z")[0]
    assert funnel.as_rect() is None


def test_zigzag_rows_all_run_the_same_direction():
    rows = fill_zigzag([_rect(8, 6, 0).flatten(0.01)], 0.5, 0.0)
    assert len(rows) >= 5
    dirs = [(r[-1] - r[0]) / np.linalg.norm(r[-1] - r[0]) for r in rows]
    assert all(float(np.dot(d, dirs[0])) > 0.999 for d in dirs)   # never reversed


def test_a_rectangle_shape_compiles_to_one_continuous_written_zigzag():
    """One stroke, written throughout: the rows are the zig (all one direction),
    the long diagonal returns are the zag -- and they are written too (green),
    so the tip goes back and forth with the bias on the whole time."""
    bp = _rect(8, 6, 0)
    obj = LithoObject("r", "shape",
                      [(float(x), float(y)) for x, y in bp.flatten(0.01)],
                      bezier=bp.to_dict(), voltage=8.0, fill_spacing_um=0.5)
    parts = expand(obj)
    assert len(parts) == 1                           # one continuous stroke, not a comb
    assert parts[0].voltage == 8.0                   # written throughout: zags are green too
    p = np.asarray(parts[0].points_um, float)
    dx = np.diff(p[:, 0])
    assert (dx > 1e-6).any() and (dx < -1e-6).any()  # it goes back AND forth


def test_the_writes_never_reverse_only_the_returns_do():
    """The horizontal write rows all run the same way; only the diagonal returns
    (which step in y) go the other way in x. That is what makes it a zig zag and
    not a row-reversing serpentine."""
    bp = _rect(8, 6, 0)
    obj = LithoObject("r", "shape",
                      [(float(x), float(y)) for x, y in bp.flatten(0.01)],
                      bezier=bp.to_dict(), voltage=8.0, fill_spacing_um=0.5)
    p = np.asarray(expand(obj)[0].points_um, float)
    seg = np.diff(p, axis=0)
    rows = seg[np.abs(seg[:, 1]) < 0.25]             # spacing 0.5 -> rows are ~horizontal
    assert len(rows) >= 5
    assert (rows[:, 0] > 0).all()                    # every write row runs +x


def test_drawn_rect_bezier_is_recognized_as_a_rectangle():
    """The cockpit's rect tool attaches a closed 4-node CUSP bezier; parsed back
    it must satisfy as_rect (drawn === imported) so it reaches fill_zigzag."""
    bp = BezierPath.from_dict(_js_rect_bezier(0, 0, 8, 6))
    assert bp.closed and len(bp.nodes) == 4
    assert abs(bp.as_rect()) < 1e-6                   # width 8 -> horizontal long edge


def test_drawn_rect_compiles_to_the_same_unidirectional_raster_as_imported():
    """A shape carrying the JS rect bezier compiles through fill_zigzag exactly as
    an imported rect: ONE continuous stroke, every write row the same direction."""
    bez = _js_rect_bezier(0, 0, 8, 6)
    pts = [(float(x), float(y)) for x, y in BezierPath.from_dict(bez).flatten(0.01)]
    obj = LithoObject("drawn", "shape", pts,
                      bezier=bez, voltage=8.0, fill_spacing_um=0.5)
    parts = expand(obj)
    assert len(parts) == 1                            # one stroke, not a fill_polygons comb
    p = np.asarray(parts[0].points_um, float)
    seg = np.diff(p, axis=0)
    rows = seg[np.abs(seg[:, 1]) < 0.25]              # spacing 0.5 -> rows ~horizontal
    assert len(rows) >= 5
    assert (rows[:, 0] > 0).all()                     # every write row runs +x (no serpentine)


def test_fill_spacing_um_sets_the_raster_pitch():
    """The density knob writes fill_spacing_um; a smaller spacing yields more rows
    (denser fill). This is the field the Write per-object control persists."""
    bez = _js_rect_bezier(0, 0, 8, 6)
    pts = [(float(x), float(y)) for x, y in BezierPath.from_dict(bez).flatten(0.01)]

    def rows_for(spacing):
        obj = LithoObject("d", "shape", list(pts),
                          bezier=bez, voltage=8.0, fill_spacing_um=spacing)
        p = np.asarray(expand(obj)[0].points_um, float)
        seg = np.diff(p, axis=0)
        return int((np.abs(seg[:, 1]) < spacing * 0.5).sum())

    assert rows_for(0.25) > rows_for(0.75)            # denser spacing -> more rows
