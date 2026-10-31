"""A region is an object, and its fill is a property of it.

The region pen used to run the fill the moment the stroke was drawn and
store the serpentine. The design then held a few thousand lines and no
polygon, which is why nothing in the program could union two regions, offset
one, edit its nodes, change its pitch without redrawing it, or write it to
OASIS as the polygon it always was. Adopting Inkscape's model needs the
document to hold shapes; this is that change.

**The whole point is that it changes representation and nothing else.** The
same fill function runs with the same parameters, at compile time instead of
at draw time, so the tip path must be identical -- and that is what the
first test here checks, point for point, for every fill strategy the pen
offers. A change of representation that also quietly changed what gets
written would be a different change wearing this one's clothes.
"""

from __future__ import annotations

import numpy as np
import pytest

from afm_litho.litho.compiler import expand
from afm_litho.litho.fill import (fill_contour, fill_hilbert, fill_polygons,
                                  fill_radial)
from afm_litho.litho.pens import PENS, render_stroke

SQUARE = [(0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0), (0.0, 0.0)]
KINDS = ["directional", "contour", "hilbert", "radial-out", "radial-in"]


def _poly(pts):
    p = np.asarray(pts, float)
    return p if np.allclose(p[0], p[-1]) else np.vstack([p, p[:1]])


def _the_old_way(kind: str, pitch: float, angle_deg: float = 0.0,
                 centre=None) -> list[np.ndarray]:
    """What `_render_region` did before this change, spelled out here.

    Copied deliberately rather than imported: the point of the test is that
    the new code produces this, and a test that called the new code twice
    would prove nothing.
    """
    poly = _poly(SQUARE)
    if kind == "contour":
        return fill_contour([poly], pitch, outline=True)
    if kind == "hilbert":
        return fill_hilbert([poly], pitch, outline=False)
    if kind in ("radial-out", "radial-in"):
        return fill_radial([poly], pitch, centre=centre,
                           inward=(kind == "radial-in"))
    return fill_polygons([poly], pitch, angle_rad=np.deg2rad(angle_deg),
                         outline=True)


@pytest.mark.parametrize("kind", KINDS)
def test_the_tip_path_is_unchanged(kind: str):
    """The written geometry is the same as it was. Point for point."""
    pitch = 0.35
    objs = render_stroke(SQUARE[:-1], "region",
                         {"fill": kind, "pitch_um": pitch, "voltage": 9.0,
                          "speed_um_s": 1.0, "angle_deg": 0.0},
                         "literal", "r")
    assert len(objs) == 1 and objs[0].kind == "shape", (
        "the pen must return the region, not its fill")
    now = [np.asarray(o.points_um, float) for o in expand(objs[0])]
    before = [np.asarray(r, float) for r in _the_old_way(kind, pitch)]
    assert len(now) == len(before), (
        f"{kind}: {len(before)} stroke(s) before, {len(now)} now")
    for a, b in zip(now, before):
        assert a.shape == b.shape, f"{kind}: {b.shape} before, {a.shape} now"
        assert np.allclose(a, b), f"{kind}: the toolpath moved"


def test_the_document_holds_a_polygon():
    """The thing the whole change is for: there IS a region to operate on."""
    o = render_stroke(SQUARE[:-1], "region", {"pitch_um": 0.3}, "literal", "r")[0]
    assert o.kind == "shape"
    assert len(o.points_um) == len(SQUARE), "the outline is the outline drawn"
    assert o.fill_kind == "directional"


def test_the_fill_can_be_changed_without_redrawing():
    """Editing the pitch is now editing a number, not drawing again."""
    o = render_stroke(SQUARE[:-1], "region", {"pitch_um": 0.6}, "literal", "r")[0]
    coarse = sum(len(s.points_um) for s in expand(o))
    o.fill_spacing_um = 0.2
    fine = sum(len(s.points_um) for s in expand(o))
    assert fine > coarse * 2, (coarse, fine)


def test_the_bounding_rect_convenience_still_squares_the_stroke():
    rough = [(0.1, 0.0), (5.9, 0.3), (6.0, 3.8), (0.0, 4.0)]
    o = render_stroke(rough, "region",
                      {"shape": "rect", "pitch_um": 0.4}, "literal", "r")[0]
    p = np.asarray(o.points_um, float)
    assert np.allclose(p.min(axis=0), [0.0, 0.0])
    assert np.allclose(p.max(axis=0), [6.0, 4.0])


def test_the_sweep_angle_only_survives_where_it_means_something():
    """`fill_angle_deg` means "along the long axis" when it is None.

    Storing the directional fill's angle on a contour or radial fill would
    make that sentence false for every other strategy, and the next thing to
    read the field would believe an angle that nothing uses.
    """
    d = render_stroke(SQUARE[:-1], "region",
                      {"fill": "directional", "angle_deg": 30.0,
                       "pitch_um": 0.4}, "literal", "r")[0]
    assert d.fill_angle_deg == 30.0
    c = render_stroke(SQUARE[:-1], "region",
                      {"fill": "contour", "angle_deg": 30.0,
                       "pitch_um": 0.4}, "literal", "r")[0]
    assert c.fill_angle_deg is None


def test_a_placed_shape_keeps_its_own_centre():
    """The radial fills spiral from a point, and the point is the shape's."""
    o = render_stroke(SQUARE[:-1], "region",
                      {"fill": "radial-out", "pitch_um": 0.4,
                       "centre_um": [1.0, 1.0]}, "literal", "r")[0]
    assert o.fill_centre_um == (1.0, 1.0)


def test_nothing_is_produced_for_a_degenerate_shape():
    assert render_stroke([(0, 0), (1, 1)], "region", {}, "literal", "r") == []


def test_every_fill_the_pen_offers_is_one_expand_knows():
    """The menu and the compiler cannot drift apart.

    A fill offered in the palette that `expand` does not know would fall
    through to the directional sweep and write the wrong toolpath in
    silence.
    """
    offered = [k for k, _ in PENS["region"].choices["fill"]]
    for kind in offered:
        o = render_stroke(SQUARE[:-1], "region",
                          {"fill": kind, "pitch_um": 0.5}, "literal", "r")[0]
        strokes = expand(o)
        assert strokes, f"{kind} expanded to nothing"
        if kind != "directional":
            plain = expand(render_stroke(
                SQUARE[:-1], "region",
                {"fill": "directional", "pitch_um": 0.5}, "literal", "r")[0])
            same = (len(plain) == len(strokes) and
                    all(np.array_equal(np.asarray(a.points_um),
                                       np.asarray(b.points_um))
                        for a, b in zip(plain, strokes)))
            assert not same, (
                f"{kind} produced the directional fill: expand does not know it")
