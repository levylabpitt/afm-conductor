"""Bezier paths, SVG round-trip, and filling closed shapes.

The lab drew virtual electrodes in Inkscape for years. These tests pin the
parts of that workflow that have to behave identically here: curves that
mean what they meant in the file, and a fill that sweeps the shape the way
the lab writes it.
"""

import numpy as np
import pytest

from afm_litho.litho.bezier import (CUSP, SMOOTH, SYMMETRIC, BezierPath, Node,
                                    load_svg, parse_svg_path, to_svg)
from afm_litho.litho.compiler import expand
from afm_litho.litho.fill import fill_polygons, long_axis_angle
from afm_litho.litho.pens import render_stroke


def _inside(poly, pts):
    x, y = pts[:, 0], pts[:, 1]
    c = np.zeros(len(pts), bool)
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        cond = (ay > y) != (by > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (bx - ax) * (y - ay) / np.where(by != ay, by - ay, 1) + ax
        c ^= cond & (x < xint)
    return c


def _dist_to_poly(poly, pts):
    a, b = poly, np.roll(poly, -1, axis=0)
    best = np.full(len(pts), 1e9)
    for i in range(len(a)):
        p0, d = a[i], b[i] - a[i]
        L2 = max(float(d @ d), 1e-18)
        t = np.clip(((pts - p0) @ d) / L2, 0, 1)
        best = np.minimum(best, np.hypot(*(pts - (p0 + t[:, None] * d)).T))
    return best


# ------------------------------------------------------------------ curves --

@pytest.mark.parametrize("tol", [0.05, 0.005, 0.0005])
def test_flattening_honours_its_tolerance(tol):
    """The tolerance is a lithography parameter: how far the written path
    may stray from the designed curve. It has to mean what it says."""
    p = BezierPath([Node(0, 0, h_out=(0, 10)), Node(10, 0, h_in=(10, 10))])
    f = p.flatten(tol)
    t = np.linspace(0, 1, 1500)[:, None]
    exact = ((1 - t) ** 3 * np.array([0, 0]) + 3 * (1 - t) ** 2 * t * np.array([0, 10])
             + 3 * (1 - t) * t ** 2 * np.array([10, 10]) + t ** 3 * np.array([10, 0]))
    assert _dist_to_poly(f, exact).max() <= tol * 1.05


def test_svg_path_round_trips():
    d = "M 0,0 C 1,2 3,2 4,0 L 4,-1 Z"
    p = parse_svg_path(d)[0]
    assert p.closed
    again = parse_svg_path(p.to_svg_d())[0]
    assert np.allclose(again.flatten(0.001), p.flatten(0.001), atol=1e-6)


def test_shorthand_and_quadratic_match_their_long_forms():
    """S/T reflect the previous control point and Q is an exact cubic. A
    parser that fumbles these silently changes the shape of an archive
    file rather than failing."""
    a = parse_svg_path("M 0,0 C 1,1 2,1 3,0 S 5,-1 6,0")[0].flatten(0.001)
    b = parse_svg_path("M 0,0 C 1,1 2,1 3,0 C 4,-1 5,-1 6,0")[0].flatten(0.001)
    assert np.allclose(a, b, atol=1e-9)
    q = parse_svg_path("M 0,0 Q 2,4 4,0")[0].flatten(0.001)
    c = parse_svg_path("M 0,0 C 1.3333333,2.6666667 2.6666667,2.6666667 4,0")[0].flatten(0.001)
    assert np.allclose(q, c, atol=1e-4)


def test_arcs_become_curves_rather_than_being_dropped():
    """Inkscape emits A for circles and rounded corners."""
    p = parse_svg_path("M 0,0 A 5,5 0 1 0 10,0 A 5,5 0 1 0 0,0 Z")[0]
    f = p.flatten(0.01)
    r = np.hypot(*(f - np.array([5.0, 0.0])).T)
    assert len(f) > 12
    assert r.max() == pytest.approx(5.0, rel=0.02)


def test_node_kinds_come_back_from_the_file():
    """A smooth node in Inkscape must still be smooth here, or dragging a
    handle behaves differently than it did in the drawing."""
    smooth = parse_svg_path("M 0,0 C 1,1 2,2 3,3 C 4,4 5,3 6,0")[0]
    assert smooth.nodes[1].kind in (SMOOTH, SYMMETRIC)
    corner = parse_svg_path("M 0,0 C 1,1 2,2 3,3 C 4,0 5,0 6,0")[0]
    assert corner.nodes[1].kind == CUSP


def test_smooth_node_keeps_its_handles_collinear():
    n = Node(0, 0, h_in=(-1, 0), h_out=(2, 0), kind=SMOOTH)
    n.set_handle("out", 0, 3)                 # swing the out handle up
    a = np.asarray(n.h_in)
    assert np.allclose(a, [0, -1], atol=1e-9)  # opposite side follows, length kept


def test_symmetric_node_mirrors_exactly():
    n = Node(0, 0, h_in=(-1, 0), h_out=(2, 0), kind=SYMMETRIC)
    n.set_handle("out", 0, 3)
    assert np.allclose(n.h_in, [0, -3], atol=1e-9)


def test_cusp_node_leaves_the_other_handle_alone():
    n = Node(0, 0, h_in=(-1, 0), h_out=(2, 0), kind=CUSP)
    n.set_handle("out", 0, 3)
    assert np.allclose(n.h_in, [-1, 0])


def test_svg_document_scales_to_microns_and_flips_y():
    """A 100 um wide page with a 10-unit viewBox is 10 um per unit, and SVG
    counts down the page while the sample counts up."""
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="0.1mm" height="0.1mm" '
           'viewBox="0 0 10 10"><path d="M 0,0 L 10,0"/></svg>')
    p = load_svg(svg, center=False)[0]
    f = p.flatten(0.01)
    assert f[:, 0].max() == pytest.approx(100.0, rel=1e-6)   # 0.1 mm = 100 um
    assert f[:, 1].max() == pytest.approx(100.0, rel=1e-6)   # y flipped to the top


def test_group_transforms_are_applied():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
           '<g transform="translate(10,0)"><path d="M 0,0 L 10,0"/></g></svg>')
    p = load_svg(svg, um_per_unit=1.0, center=False)[0]
    assert p.flatten(0.01)[:, 0].min() == pytest.approx(10.0, abs=1e-6)


def test_svg_export_reimports_unchanged():
    src = parse_svg_path("M 0,0 C 2,3 6,3 8,0 L 8,-2 Z")
    back = load_svg(to_svg(src), center=False)
    a = src[0].flatten(0.002)
    b = back[0].flatten(0.002)
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    assert np.allclose(a, b, atol=1e-3)


# -------------------------------------------------------------------- fill --

def test_fill_runs_along_the_long_axis():
    """The lab writes the long direction: a lead swept the short way is
    thousands of tiny strokes and thousands of turnarounds."""
    funnel = parse_svg_path("M 0,-1.5 C 4,-1.4 6,-0.2 10,-0.15 "
                            "L 10,0.15 C 6,0.2 4,1.4 0,1.5 Z")[0]
    poly = funnel.flatten(0.005)
    assert abs(np.sin(long_axis_angle(poly))) < 0.05      # long axis is x
    st = fill_polygons([poly], spacing_um=0.25)[0]
    d = np.abs(np.diff(st, axis=0))
    assert d[:, 0].mean() > 5 * d[:, 1].mean()


def test_holes_are_not_written_and_split_the_stroke():
    outer = np.array([[-5, -5], [5, -5], [5, 5], [-5, 5]], float)
    hole = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2]], float)
    strokes = fill_polygons([outer, hole], spacing_um=0.5, angle_rad=0.0)
    pts = np.vstack(strokes)
    assert not ((np.abs(pts[:, 0]) < 1.9) & (np.abs(pts[:, 1]) < 1.9)).any()
    assert len(strokes) == 2        # above and below: never dragged across


def test_non_shape_objects_pass_through_expand():
    o = render_stroke([(0.0, 0.0), (5.0, 0.0)], "write", mode="clean")[0]
    assert expand(o) == [o]
