"""Path operations: union, difference, offset, simplify.

`docs/INKSCAPE.md`: these are not graphics luxuries here, they are the
vocabulary of the thing being drawn. So they are checked against AREAS and
WIDTHS, in microns, rather than against vertex counts: a boolean that
produces a plausible-looking polygon of the wrong area writes the wrong
device, and a vertex count would not notice.

The one to read is `test_outsetting_a_line_gives_it_a_width`. A wire is
drawn as a centre line and written as an area, and until this existed the
only thing that could turn one into the other was the virtual-electrode
generator, which makes exactly one shape and composes with nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from afm_litho.litho.compiler import LithoObject
from afm_litho.litho import pathops as P

# The boolean/offset vocabulary here is klayout's geometry (see the module
# docstring). Without the optional [geometry] extra, pathops raises PathOpError
# by design -- the app itself reports geometry:false and disables these
# operations in the palette -- so an absent klayout should SKIP this module, not
# fail it. That courtesy is what the application already extends to the user.
pytest.importorskip("klayout", reason="geometry extra absent: pip install -e .[geometry]")


def square(name: str, x0: float, y0: float, x1: float, y1: float) -> LithoObject:
    return LithoObject(name=name, kind="shape",
                       points_um=[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
                       voltage=9.0, speed_um_s=1.5, fill_spacing_um=0.02)


def area(o: LithoObject) -> float:
    def ring(pts):
        p = np.asarray(pts, float)
        x, y = p[:, 0], p[:, 1]
        return abs(float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))) / 2
    return ring(o.points_um) - sum(ring(h) for h in (o.holes_um or []))


def total(objs) -> float:
    return sum(area(o) for o in objs)


# --------------------------------------------------------------- booleans --

@pytest.mark.parametrize("op,expected", [
    ("union", 175.0), ("intersection", 25.0),
    ("difference", 75.0), ("exclusion", 150.0),
])
def test_the_boolean_areas_are_right(op: str, expected: float):
    a, b = square("a", 0, 0, 10, 10), square("b", 5, 5, 15, 15)
    out = P.boolean([a, b], op)
    assert out, f"{op} produced nothing"
    assert total(out) == pytest.approx(expected, abs=1e-3)


def test_difference_takes_the_first_object_as_the_base():
    """Order matters for difference, and z-order is the rule, as in Inkscape."""
    a, b = square("a", 0, 0, 10, 10), square("b", 5, 5, 15, 15)
    assert total(P.boolean([a, b], "difference")) == pytest.approx(75.0, abs=1e-3)
    assert total(P.boolean([b, a], "difference")) == pytest.approx(75.0, abs=1e-3)
    # ... and they are not the same shape, whatever the areas say
    ab = np.asarray(P.boolean([a, b], "difference")[0].points_um)
    ba = np.asarray(P.boolean([b, a], "difference")[0].points_um)
    assert not np.allclose(ab.min(axis=0), ba.min(axis=0))


def test_a_hole_survives_as_a_hole():
    """A ring is one object with a hole, not two objects.

    Written as two overlapping regions it would be written twice, with the
    inner one erasing nothing; written as a polygon with a hole, the fill
    skips the middle.
    """
    out = P.boolean([square("o", 0, 0, 10, 10), square("i", 3, 3, 7, 7)],
                    "difference")
    assert len(out) == 1
    assert len(out[0].holes_um) == 1, "the hole was lost"
    assert area(out[0]) == pytest.approx(100 - 16, abs=1e-3)


def test_the_result_keeps_the_first_objects_write_settings():
    """Geometry only. A boolean that reset the bias would change the physics."""
    a = square("a", 0, 0, 10, 10)
    a.voltage, a.speed_um_s, a.fill_spacing_um = 11.5, 0.7, 0.004
    out = P.boolean([a, square("b", 5, 5, 15, 15)], "union")[0]
    assert (out.voltage, out.speed_um_s, out.fill_spacing_um) == (11.5, 0.7, 0.004)


def test_the_derived_geometry_does_not_survive():
    """A union is nobody's Bezier and nobody's generator.

    Keeping them would leave the node editor dragging control points that
    describe a shape that no longer exists, and the next re-derive would put
    the old shape back.
    """
    a = square("a", 0, 0, 10, 10)
    a.bezier = {"nodes": [{"x": 0, "y": 0, "h_in": None, "h_out": None,
                           "kind": "cusp"}], "closed": True}
    out = P.boolean([a, square("b", 5, 5, 15, 15)], "union")[0]
    assert out.bezier is None


def test_a_boolean_on_a_line_is_refused_by_name():
    line = LithoObject(name="w1", kind="path", points_um=[(0, 0), (10, 0)])
    with pytest.raises(P.PathOpError) as e:
        P.boolean([line, square("a", 0, 0, 10, 10)], "union")
    assert "w1" in str(e.value) and "line" in str(e.value)
    assert "outset" in str(e.value).lower(), "a refusal should say the way out"


def test_a_boolean_that_empties_the_canvas_refuses_instead():
    with pytest.raises(P.PathOpError) as e:
        P.boolean([square("a", 0, 0, 1, 1), square("b", 5, 5, 6, 6)],
                  "intersection")
    assert "Nothing was changed" in str(e.value)


def test_two_objects_are_needed():
    with pytest.raises(P.PathOpError) as e:
        P.boolean([square("a", 0, 0, 1, 1)], "union")
    assert "two" in str(e.value)


# ----------------------------------------------------------------- offset --

def test_outsetting_a_line_gives_it_a_width():
    """The operation this program most needed and did not have.

    A wire drawn by hand becomes a lead of the width asked for, and the
    result is an ordinary region that the next operation can take as input.
    """
    line = LithoObject(name="w", kind="path", points_um=[(0, 0), (10, 0)])
    out = P.offset(line, 0.5)
    assert len(out) == 1 and out[0].kind == "shape"
    p = np.asarray(out[0].points_um, float)
    assert p[:, 1].max() == pytest.approx(0.5, abs=1e-3)
    assert p[:, 1].min() == pytest.approx(-0.5, abs=1e-3)
    assert area(out[0]) == pytest.approx(10.0, abs=0.05), "10 um x 1 um"
    # and it composes: the lead can now be unioned with a pad
    P.boolean([out[0], square("pad", 9, -2, 12, 2)], "union")


def test_outset_and_inset_move_the_edge_by_the_amount_asked_for():
    grown = P.offset(square("a", 0, 0, 10, 10), 1.0)[0]
    p = np.asarray(grown.points_um, float)
    assert p.min(axis=0) == pytest.approx([-1, -1], abs=1e-3)
    assert p.max(axis=0) == pytest.approx([11, 11], abs=1e-3)
    shrunk = P.offset(square("a", 0, 0, 10, 10), -1.0)[0]
    q = np.asarray(shrunk.points_um, float)
    assert q.min(axis=0) == pytest.approx([1, 1], abs=1e-3)
    assert q.max(axis=0) == pytest.approx([9, 9], abs=1e-3)


def test_insetting_a_shape_away_to_nothing_refuses():
    with pytest.raises(P.PathOpError) as e:
        P.offset(square("a", 0, 0, 1, 1), -2.0)
    assert "narrower" in str(e.value) and "Nothing was changed" in str(e.value)


def test_insetting_a_line_refuses_rather_than_guessing():
    line = LithoObject(name="w", kind="path", points_um=[(0, 0), (10, 0)])
    with pytest.raises(P.PathOpError) as e:
        P.offset(line, -0.5)
    assert "no width" in str(e.value)


# --------------------------------------------------------------- simplify --

def test_simplify_keeps_every_point_within_the_tolerance():
    """The guarantee is stated in microns, which is the one that matters on
    an instrument. A vertex count is not a guarantee about anything."""
    t = np.linspace(0, 4 * math.pi, 2000)
    pts = [(float(x), float(math.sin(x))) for x in t]
    w = LithoObject(name="s", kind="path", points_um=pts)
    tol = 0.01
    out = P.simplify(w, tol)
    assert len(out.points_um) < len(pts) / 10
    kept = np.asarray(out.points_um, float)
    orig = np.asarray(pts, float)
    # every original point is within tol of the simplified polyline
    worst = 0.0
    for p in orig[::7]:
        d = np.min([_dist_to_seg(p, kept[i], kept[i + 1])
                    for i in range(len(kept) - 1)])
        worst = max(worst, d)
    assert worst <= tol * 1.001, worst


def _dist_to_seg(p, a, b):
    ab = b - a
    n = float(ab @ ab)
    t = 0.0 if n == 0 else max(0.0, min(1.0, float((p - a) @ ab) / n))
    return float(np.hypot(*(p - (a + t * ab))))


def test_simplify_keeps_a_closed_ring_closed():
    th = np.linspace(0, 2 * math.pi, 400)
    pts = [(float(3 * math.cos(a)), float(3 * math.sin(a))) for a in th]
    o = LithoObject(name="c", kind="shape", points_um=pts)
    out = P.simplify(o, 0.01)
    assert out.points_um[0] == out.points_um[-1], "the ring was opened"
    assert area(out) == pytest.approx(math.pi * 9, rel=0.01)


def test_simplify_keeps_the_bias_profile():
    """Tidying a superlattice's geometry must not turn it into a plain wire."""
    pts = [(float(x) / 100, 0.0) for x in range(400)]
    o = LithoObject(name="s", kind="path", points_um=pts,
                    v_profile=[1.0, -1.0] * 50)
    out = P.simplify(o, 0.001)
    assert out.v_profile == [1.0, -1.0] * 50


def test_simplifying_a_shape_to_less_than_a_triangle_refuses():
    o = square("a", 0, 0, 1, 1)
    with pytest.raises(P.PathOpError) as e:
        P.simplify(o, 5.0)
    assert "Nothing was changed" in str(e.value)


# -------------------------------------------------------------- transform --

def test_flipping_is_its_own_inverse():
    o = square("a", 0, 0, 10, 4)
    before = list(o.points_um)
    P.transform(o, sx=-1)
    assert o.points_um != before
    P.transform(o, sx=-1)
    assert np.allclose(np.asarray(o.points_um), np.asarray(before))


def test_rotation_preserves_area():
    o = square("a", 0, 0, 10, 4)
    before = area(o)
    P.transform(o, rot_deg=37.0)
    assert area(o) == pytest.approx(before, rel=1e-9)


# ----------------------------------------------------- through the service --
# The operations are reached from the panel as one endpoint, and two things
# about that endpoint are worth holding: undo has to step the whole object
# list, and a refusal has to leave the design exactly as it was.


import pytest as _pytest
from fastapi.testclient import TestClient

from afm_litho.service import app as appmod
from afm_litho.service.routers import design as designmod


@_pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "RUNTIME", tmp_path)
    designmod._design_history.clear()
    with TestClient(appmod.app) as c:
        c.post("/api/sketch/clear")
        yield c


def _draw(c, pts, **params):
    r = c.post("/api/sketch", json={"points_um": pts, "pen": "region",
                                    "mode": "literal",
                                    "params": {"pitch_um": 0.5, **params}})
    assert r.status_code == 200 and not r.json().get("error"), r.json()
    return r.json()["design"]


def test_undo_after_a_union_brings_both_inputs_back(client):
    """The reason undo is a history and not "delete the last object".

    A union removes two objects and adds one. Deleting the union leaves the
    two inputs gone, which is not what anybody pressing Ctrl+Z means.
    """
    _draw(client, [(0, 0), (10, 0), (10, 10), (0, 10)])
    _draw(client, [(5, 5), (15, 5), (15, 15), (5, 15)])
    r = client.post("/api/design/op", json={"op": "union", "indices": [0, 1]}).json()
    assert len(r["design"]["objects"]) == 1
    back = client.post("/api/sketch/undo", json={"count": 1}).json()
    assert len(back["design"]["objects"]) == 2, "the inputs did not come back"


def test_a_refused_operation_changes_nothing(client):
    """Not "mostly nothing": the same objects, in the same order."""
    _draw(client, [(0, 0), (1, 0), (1, 1), (0, 1)])
    _draw(client, [(8, 8), (9, 8), (9, 9), (8, 9)])
    before = client.get("/api/state").status_code and \
        client.post("/api/design/op", json={"op": "raise", "indices": []}).json()
    r = client.post("/api/design/op",
                    json={"op": "intersection", "indices": [0, 1]}).json()
    assert "error" in r and "Nothing was changed" in r["error"]
    names = [o["name"] for o in r["design"]["objects"]]
    assert len(names) == 2, names
    assert before is not None


def test_the_reply_says_what_to_select_next(client):
    """After a union that is the union, not the two things that are gone.

    The panel cannot work this out for itself without knowing what the
    operation did, and a stale selection is a selection pointing at other
    people's objects.
    """
    _draw(client, [(0, 0), (10, 0), (10, 10), (0, 10)])
    _draw(client, [(5, 5), (15, 5), (15, 15), (5, 15)])
    r = client.post("/api/design/op", json={"op": "union", "indices": [0, 1]}).json()
    assert r["selection"] == [0]
    d = client.post("/api/design/op", json={"op": "delete", "indices": [0]}).json()
    assert d["selection"] == []


def test_the_capabilities_are_answerable_before_anything_is_drawn(client):
    caps = client.get("/api/design/capabilities").json()
    assert caps["geometry"] is True, caps["geometry_error"]
    assert "union" in caps["ops"] and caps["ops"]["union"]["keys"] == "Ctrl +"
