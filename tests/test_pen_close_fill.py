"""A closed pen path reaches the SAME fill machinery an imported outline does.

Slice 2 lets the pen close a path (double-click the start node). The drawn
closed outline must be indistinguishable from an imported one:

  * a funnel (flat base + two curved sides to an apex) commits as an
    `electrode` -- exactly the kind the SVG import picks -- and compiles to the
    2013 funnel family (one continuous sweep between its two side curves);
  * any other closed blob stays a `shape` and compiles as a generic region.

The client sends `kind:"shape"` for every closed pen path; the add-handler
re-derives `electrode` for a funnel (mirroring routers/design.py SVG import), so
a HAND-DRAWN funnel === an IMPORTED funnel. No new fill code -- this only makes
the drawn closed path REACH `expand()`.
"""


from fastapi.testclient import TestClient

from afm_litho.litho.bezier import BezierPath
from afm_litho.litho.compiler import LithoObject, expand

# A funnel, in the {nodes, closed} shape the cockpit's penToBezier emits:
# base (0,0)->(4,0) straight, two curved sides up to the apex (2,6) and back.
FUNNEL_BEZIER = {
    "closed": True,
    "nodes": [
        {"x": 0.0, "y": 0.0, "h_in": [0.5, 3.0], "h_out": None, "kind": "cusp"},
        {"x": 4.0, "y": 0.0, "h_in": None, "h_out": [3.5, 3.0], "kind": "cusp"},
        {"x": 2.0, "y": 6.0, "h_in": [2.5, 5.0], "h_out": [1.5, 5.0], "kind": "cusp"},
    ],
}

# A generic closed blob (a pentagon): 5 cusp nodes, no handles. Neither a funnel
# (as_funnel needs exactly 3 nodes) nor a rectangle (as_rect needs exactly 4).
BLOB_BEZIER = {
    "closed": True,
    "nodes": [
        {"x": 0.0, "y": 0.0, "h_in": None, "h_out": None, "kind": "cusp"},
        {"x": 4.0, "y": 0.0, "h_in": None, "h_out": None, "kind": "cusp"},
        {"x": 5.0, "y": 3.0, "h_in": None, "h_out": None, "kind": "cusp"},
        {"x": 2.0, "y": 5.0, "h_in": None, "h_out": None, "kind": "cusp"},
        {"x": -1.0, "y": 3.0, "h_in": None, "h_out": None, "kind": "cusp"},
    ],
}


def _client():
    from afm_litho.service.app import app
    return TestClient(app)


def _add(c, bezier):
    corners = [[n["x"], n["y"]] for n in bezier["nodes"]]
    r = c.post("/api/design/add", json={"kind": "shape", "points_um": corners,
                                        "bezier": bezier})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("ok"), body
    return body["design"]["objects"][body["index"]]


def test_the_funnel_bezier_is_a_funnel_to_the_recogniser():
    """Guard the geometry itself: the dict penToBezier emits is a funnel."""
    bp = BezierPath.from_dict(FUNNEL_BEZIER)
    assert bp.closed
    assert bp.as_funnel() is not None          # base + two sides -> apex
    assert bp.as_rect() is None
    blob = BezierPath.from_dict(BLOB_BEZIER)
    assert blob.as_funnel() is None and blob.as_rect() is None


def test_drawn_closed_funnel_commits_as_electrode():
    with _client() as c:
        o = _add(c, FUNNEL_BEZIER)
        # drawn === imported: the SVG import labels a funnel `electrode`, so must
        # the drawn one (routers/design.py add-handler re-derivation).
        assert o["kind"] == "electrode", o


def test_drawn_closed_blob_stays_shape():
    with _client() as c:
        o = _add(c, BLOB_BEZIER)
        assert o["kind"] == "shape", o


def test_drawn_closed_funnel_compiles_to_one_sweep():
    """The funnel fills by the 2013 family: one continuous stroke, the whole
    fill (mirrors test_funnel_fill's compile assertion, from the drawn dict)."""
    bp = BezierPath.from_dict(FUNNEL_BEZIER)
    obj = LithoObject("f", "electrode",
                      [(float(x), float(y)) for x, y in bp.flatten(0.01)],
                      bezier=bp.to_dict(), voltage=8.0, fill_spacing_um=0.25)
    parts = expand(obj)
    assert len(parts) == 1                      # one connected write -- no lifts
    assert len(parts[0].points_um) > 50         # ... and it is the whole fill


def test_drawn_closed_blob_compiles_as_a_region():
    """A non-funnel, non-rect closed outline fills as a generic region."""
    bp = BezierPath.from_dict(BLOB_BEZIER)
    obj = LithoObject("blob", "shape",
                      [(float(x), float(y)) for x, y in bp.flatten(0.01)],
                      bezier=bp.to_dict(), voltage=8.0, fill_spacing_um=0.25)
    parts = expand(obj)
    assert parts                                # it compiled to fill strokes
    assert all(len(p.points_um) >= 2 for p in parts)


def test_drawn_closed_funnel_toolpath_end_to_end():
    """Full path: add via the endpoint, then the compiled toolpath carries the
    funnel object -- drawn outline -> server -> expand()."""
    with _client() as c:
        o = _add(c, FUNNEL_BEZIER)
        name = o["name"]
        tp = c.post("/api/litho/toolpath")
        assert tp.status_code == 200, tp.text
        data = tp.json()
        assert "error" not in data, data
        segs = [s for s in data["segments"] if s["object"] == name]
        assert segs, [s["object"] for s in data["segments"]]
        assert all(len(s["path_um"]) >= 2 for s in segs)
