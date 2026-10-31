"""Writing a region: sweep it, or follow its own outline inward.

User request, 2026-08-08: "I want to be able to scan rectangles or arbitrary shapes
that I define with whatever voltage. The rastering is done either along a
certain direction for convex shapes or it is done in a self-similar
shape-filling manner."

Two strategies, because they fail in different places:

  directional -- parallel sweeps at a chosen angle, serpentined. The right
                 answer for a convex region where one fast axis matters.
                 On a concave region it needs long retraces to get from one
                 span to the next.

  contour     -- rings following the boundary inward. The toolpath stays
                 parallel to the edge everywhere, the boundary is written as
                 ONE continuous pass rather than as a thousand line-ends, and
                 a region that necks down splits into separate rings, which
                 is correct and is why it is done by eroding a mask rather
                 than offsetting an outline analytically.
"""

from __future__ import annotations

import numpy as np
import pytest

from afm_litho.litho.compiler import expand
from afm_litho.litho.fill import fill_contour
from afm_litho.litho.pens import PENS, render_stroke

SQUARE = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2]], float)
# a dumbbell: two lobes joined by a thin waist
DUMBBELL = np.array([[-3, -2], [-1, -2], [-1, -0.25], [1, -0.25], [1, -2],
                     [3, -2], [3, 2], [1, 2], [1, 0.25], [-1, 0.25],
                     [-1, 2], [-3, 2]], float)


def _inside(poly, pts, tol=0.02):
    """Inside, counting the boundary itself as inside.

    An even-odd test says a point exactly ON an edge is outside, and the
    outline ring of a contour fill lies exactly on the edge by construction.
    Testing against the raw polygon therefore fails the very pass that is
    most obviously correct. The polygon is grown by `tol` about its centroid
    so that "on the edge" counts, and anything genuinely escaping the shape
    still does not.
    """
    from afm_litho.litho.fill import _points_in_poly
    poly = np.asarray(poly, float)
    c = poly.mean(axis=0)
    r = np.linalg.norm(poly - c, axis=1).max()
    grown = c + (poly - c) * (1.0 + tol / max(r, 1e-9))
    p = np.asarray(pts, float)
    return _points_in_poly(grown, p[:, 0], p[:, 1])


def written(points, params, mode="clean"):
    """The strokes the tip will actually write for this region.

    The pen returns the REGION now, with the fill as a property of it, and
    the fill is resolved by `compiler.expand` when the pattern is compiled
    (see test_region_object.py, which holds the toolpath to the one the pen
    used to return, point for point). Every assertion below is about the
    FILL, so every one of them expands first. Nothing about what these tests
    claim has changed; only where in the program the claim is checked.
    """
    objs = render_stroke(points, "region", params, mode, "region")
    return [s for o in objs for s in expand(o)]


# ------------------------------------------------------------- the pen ----

def test_the_pen_exists_and_offers_both_strategies():
    pen = PENS["region"]
    assert pen.params["fill"] == "directional"
    assert "voltage" in pen.params and "pitch_um" in pen.params
    assert "angle_deg" in pen.params


@pytest.mark.parametrize("kind", ["directional", "contour"])
def test_a_region_is_written_at_the_voltage_asked_for(kind):
    """'with whatever voltage' -- one bias for the whole region."""
    objs = written(SQUARE, {"fill": kind, "pitch_um": 0.4, "voltage": 7.5})
    assert objs, f"{kind} produced nothing"
    assert all(o.voltage == 7.5 for o in objs)


@pytest.mark.parametrize("kind", ["directional", "contour"])
def test_the_fill_stays_inside_the_region(kind):
    objs = written(SQUARE, {"fill": kind, "pitch_um": 0.4})
    pts = np.array([p for o in objs for p in o.points_um])
    frac = _inside(SQUARE, pts).mean()
    assert frac > 0.9, f"{kind}: only {frac:.0%} of the path is inside the shape"


@pytest.mark.parametrize("kind", ["directional", "contour"])
def test_a_finer_pitch_writes_more(kind):
    coarse = written(SQUARE, {"fill": kind, "pitch_um": 0.8})
    fine = written(SQUARE, {"fill": kind, "pitch_um": 0.3})
    n = lambda o: sum(len(x.points_um) for x in o)
    assert n(fine) > n(coarse)


def test_a_stroke_can_be_squared_off_into_its_bounding_box():
    """Drawing four corners by hand to get a rectangle is beneath the tool."""
    rough = [(0, 0), (3, 0.2), (3.1, 2), (0.1, 1.9)]
    objs = written(rough, {"shape": "rect", "pitch_um": 0.4})
    pts = np.array([p for o in objs for p in o.points_um])
    assert pts[:, 0].min() >= -0.05 and pts[:, 0].max() <= 3.15
    assert pts[:, 1].min() >= -0.05 and pts[:, 1].max() <= 2.05


# --------------------------------------------------------- directional ----

def test_the_sweep_angle_is_obeyed():
    """A convex region swept at 0 and at 90 degrees must not come out the
    same, or the angle is decoration."""
    a = written(SQUARE, {"fill": "directional", "angle_deg": 0, "pitch_um": 0.4})
    b = written(SQUARE, {"fill": "directional", "angle_deg": 90, "pitch_um": 0.4})

    def spread(objs):
        pts = np.array([p for o in objs for p in o.points_um])
        # the fill direction is the axis along which single runs are long
        d = np.diff(pts, axis=0)
        return abs(d[:, 0]).sum(), abs(d[:, 1]).sum()

    ax, ay = spread(a)
    bx, by = spread(b)
    assert ax > ay, "a 0 degree sweep should travel mostly in x"
    assert by > bx, "a 90 degree sweep should travel mostly in y"


# ------------------------------------------------------------- contour ----

def test_contour_rings_follow_the_outline_inward():
    rings = fill_contour([SQUARE], 0.4)
    assert len(rings) >= 3, "a 4 um square at 0.4 um pitch should nest rings"
    # each successive ring encloses less area than the one before it
    def area(r):
        x, y = r[:, 0], r[:, 1]
        return abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    areas = [area(r) for r in rings]
    assert areas[0] > areas[-1], "rings should shrink inward"


def test_every_contour_ring_is_closed():
    """It is a loop around the region; leaving it open would put a seam in
    the written edge."""
    for r in fill_contour([SQUARE], 0.4):
        assert np.allclose(r[0], r[-1]), "ring is not closed"


def test_a_necking_region_splits_into_separate_rings():
    """The reason this erodes a mask instead of offsetting an outline: past
    the waist a dumbbell becomes TWO loops, and a planner that cannot
    represent that will write a line straight across the gap."""
    rings = fill_contour([DUMBBELL], 0.3)
    assert len(rings) > 2

    # no ring may cross the waist: the neck is the strip |y| < 0.25 between
    # the lobes, and a ring that spans both lobes has points either side of
    # x = 0 at |y| > 1, with nothing bridging them inside the neck.
    for r in rings[1:]:                       # skip the outline itself
        in_neck = np.abs(r[:, 1]) < 0.2
        crossing = in_neck & (np.abs(r[:, 0]) < 0.9)
        assert not crossing.any(), "a ring crossed the waist, which is solid"


def test_contour_writes_the_boundary_as_one_pass():
    """The point of the strategy: the edge is a single continuous run, not
    the ends of a thousand parallel lines."""
    rings = fill_contour([SQUARE], 0.4, outline=True)
    outer = rings[0]
    assert len(outer) >= 4 and np.allclose(outer[0], outer[-1])


def test_nothing_is_produced_for_a_degenerate_shape():
    assert fill_contour([np.array([[0, 0], [1, 1]], float)], 0.2) == []
    assert written([(0, 0), (1, 1)], {}) == []


def test_the_ring_count_is_bounded_by_the_geometry():
    """A tiny pitch in a big region must terminate, not run for ever."""
    rings = fill_contour([SQUARE], 0.15)
    assert 5 < len(rings) < 400
