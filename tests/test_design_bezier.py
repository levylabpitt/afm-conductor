"""The node tool edits the Bezier control points, not the flat polyline.

`GET /api/design/bezier/{index}` serves the editable curve; `POST
/api/design/bezier` stores edited control points, re-flattens to `points_um`, and
-- the load-bearing part -- COLLAPSES TO BASE: a modulated wire's editable curve
is its base axis, and editing it re-applies the modulation on top of the reshaped
base, so the sine never edits a baked toolpath. Geometry is checked against the
full-resolution `hub.pattern` object, not the coarse to_dict polyline.
"""

import math

from fastapi.testclient import TestClient


def _client():
    from afm_litho.service.app import app
    return TestClient(app)


AXIS = [(0.0, 0.0), (3.0, 0.0)]


def _seed_path():
    from afm_litho.litho.compiler import LithoObject, Pattern
    from afm_litho.service.hub import hub
    o = LithoObject("wire", "path", list(AXIS), voltage=8.0, speed_um_s=4.0)
    hub.pattern = Pattern(name="seed", objects=[o])
    return hub


def _obj():
    from afm_litho.service.hub import hub
    return hub.pattern.objects[0]


def test_get_synthesizes_a_bezier_for_a_plain_polyline():
    """A hand-drawn wire never came from control points; the node tool still
    needs one, so a plain polyline is served as cusp nodes with no handles."""
    _seed_path()
    with _client() as c:
        r = c.get("/api/design/bezier/0")
        assert r.status_code == 200, r.text
        j = r.json()
        assert not j.get("modulated")
        nodes = j["bezier"]["nodes"]
        assert len(nodes) == 2
        assert all(n["h_in"] is None and n["h_out"] is None for n in nodes)
        assert nodes[0]["kind"] == "cusp"


def test_post_stores_bezier_and_reflattens_points():
    """Editing the control points must keep BOTH the bezier and a re-flattened
    points_um, so the object still compiles/writes. A curved segment flattens to
    more than its two endpoints."""
    _seed_path()
    with _client() as c:
        # bend the single segment into a curve by pulling handles off the axis
        bez = {"closed": False, "nodes": [
            {"x": 0.0, "y": 0.0, "h_in": None, "h_out": [1.0, 1.0], "kind": "cusp"},
            {"x": 3.0, "y": 0.0, "h_in": [2.0, 1.0], "h_out": None, "kind": "cusp"},
        ]}
        r = c.post("/api/design/bezier", json={"index": 0, "bezier": bez})
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["ok"] and not j["reapplied"]
    o = _obj()
    assert o.bezier is not None and len(o.bezier["nodes"]) == 2
    assert len(o.points_um) > 2                       # the curve flattened
    # the flattened curve bulges off the straight axis (y != 0 in the middle)
    assert max(abs(y) for _, y in o.points_um) > 0.1


def test_modulated_wire_collapses_to_base_and_reapplies():
    """A chiral wire node-edits its BASE axis; the modulation re-applies on top.

    After making the wire chiral, editing the bezier (the base) must (a) keep the
    edited curve as the base axis and (b) regenerate the modulated toolpath from
    it -- so points_um is the wiggly toolpath again, not the plain base, and the
    per-point bias profile is present."""
    _seed_path()
    with _client() as c:
        # make it chiral -- straight axis, so no curvature clamp
        r = c.post("/api/design/chiral", json={
            "indices": [0], "enabled": True,
            "lambda_um": 0.05, "y_amp_um": 0.02, "phase_deg": 90.0,
            "v_k": 5.0, "v0": 8.0, "handedness": "R"})
        assert r.status_code == 200, r.text
        # the editable curve is now the BASE axis (2 straight cusp nodes)
        g = c.get("/api/design/bezier/0").json()
        assert g["modulated"] is True
        assert len(g["bezier"]["nodes"]) == 2
        # extend the base axis to x=5 and re-store it
        bez = g["bezier"]
        bez["nodes"][1]["x"] = 5.0
        r = c.post("/api/design/bezier", json={"index": 0, "bezier": bez})
        assert r.status_code == 200, r.text
        assert r.json()["reapplied"] is True
    o = _obj()
    # base axis followed the edit
    assert abs(o.axis_um[-1][0] - 5.0) < 1e-6
    # the toolpath is the modulated wiggle (many points, off-axis), not the base
    assert len(o.points_um) > 10
    assert max(abs(y) for _, y in o.points_um) > 0.005
    assert o.v_profile is not None and len(o.v_profile) > 1
    assert o.profile and o.profile["kind"] == "chiral"


def test_reapply_reconstructs_phase_from_the_profile():
    """The re-applied modulation matches the requested phase: hand is folded into
    phase_rad, so regenerating from the profile alone reproduces it."""
    _seed_path()
    with _client() as c:
        c.post("/api/design/chiral", json={
            "indices": [0], "enabled": True,
            "lambda_um": 0.05, "y_amp_um": 0.02, "phase_deg": 90.0,
            "v_k": 5.0, "v0": 8.0, "handedness": "R"})
        phase0 = _obj().profile["phase_rad"]
        g = c.get("/api/design/bezier/0").json()
        c.post("/api/design/bezier", json={"index": 0, "bezier": g["bezier"]})
    assert math.isclose(_obj().profile["phase_rad"], phase0, abs_tol=1e-6)


def test_add_carries_the_bezier_and_flattens_it_to_points():
    """The Pen finishes by CREATING an object that carries its bezier (like
    Inkscape), not a flattened polyline: POST /api/design/add with a `bezier`
    stores the control points on the new object AND sets points_um to their
    flatten, so it is immediately node-editable with the SAME handles and still
    compiles/writes. A curved segment flattens to more than its two endpoints."""
    from afm_litho.litho.compiler import Pattern
    from afm_litho.service.hub import hub
    hub.pattern = Pattern(name="seed", objects=[])
    with _client() as c:
        bez = {"closed": False, "nodes": [
            {"x": 0.0, "y": 0.0, "h_in": None, "h_out": [1.0, 1.0],
             "kind": "symmetric"},
            {"x": 3.0, "y": 0.0, "h_in": [2.0, 1.0], "h_out": None,
             "kind": "symmetric"},
        ]}
        # a stale polyline is also sent (the pen's fallback); the bezier wins
        r = c.post("/api/design/add", json={
            "kind": "path", "points_um": [[0.0, 0.0], [3.0, 0.0]],
            "bezier": bez})
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["ok"] and isinstance(j["index"], int)
        o = hub.pattern.objects[j["index"]]
        # the bezier rode onto the object verbatim (same node count + handle)
        assert o.bezier is not None and len(o.bezier["nodes"]) == 2
        assert o.bezier["nodes"][0]["h_out"] is not None
        # points_um is the FLATTEN of the curve, not the 2-point polyline sent
        assert len(o.points_um) > 2
        assert max(abs(y) for _, y in o.points_um) > 0.1
        # and the node tool gets those exact control points straight back
        g = c.get(f"/api/design/bezier/{j['index']}").json()
        assert not g.get("modulated")
        assert len(g["bezier"]["nodes"]) == 2
        assert g["bezier"]["nodes"][0]["h_out"] is not None


def test_add_without_bezier_still_uses_points():
    """Back-compat: a plain add (rect/no-bezier) keeps using points_um and stores
    no bezier."""
    from afm_litho.litho.compiler import Pattern
    from afm_litho.service.hub import hub
    hub.pattern = Pattern(name="seed", objects=[])
    with _client() as c:
        r = c.post("/api/design/add", json={
            "kind": "path", "points_um": [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]]})
        assert r.status_code == 200, r.text
        o = hub.pattern.objects[r.json()["index"]]
        assert o.bezier is None
        assert len(o.points_um) == 3


def test_add_with_a_degenerate_bezier_is_reported_not_raised():
    """A one-node bezier flattens to < 2 points -> refused cleanly."""
    from afm_litho.litho.compiler import Pattern
    from afm_litho.service.hub import hub
    hub.pattern = Pattern(name="seed", objects=[])
    with _client() as c:
        r = c.post("/api/design/add", json={
            "kind": "path", "points_um": [[0.0, 0.0], [1.0, 0.0]],
            "bezier": {"closed": False,
                       "nodes": [{"x": 0, "y": 0, "kind": "cusp"}]}})
        assert "error" in r.json()


def test_end_binding_persists_through_add_and_bezier_roundtrip():
    """Snap-to-bind (EDIT_TOOLS §4): an endpoint->electrode binding rides through
    the add + bezier handlers, serializes in to_dict, and round-trips via
    /api/design -- optional, so a design with no binding is unchanged. The binding
    is a REFERENCE (a port id), never a coord copy."""
    from afm_litho.litho.compiler import Pattern
    from afm_litho.service.hub import hub
    hub.pattern = Pattern(name="seed", objects=[])
    with _client() as c:
        # add a plain wire WITHOUT a binding -> absent from the serialization
        r = c.post("/api/design/add", json={
            "kind": "path", "points_um": [[0.0, 0.0], [3.0, 0.0]]})
        assert r.status_code == 200, r.text
        idx = r.json()["index"]
        assert "end_binding" not in r.json()["design"]["objects"][idx]
        assert hub.pattern.objects[idx].end_binding is None

        # node-edit the curve AND bind the END node to a port -> stored + serialized
        bez = {"closed": False, "nodes": [
            {"x": 0.0, "y": 0.0, "h_in": None, "h_out": None, "kind": "cusp"},
            {"x": 3.0, "y": 0.0, "h_in": None, "h_out": None, "kind": "cusp"},
        ]}
        r = c.post("/api/design/bezier", json={
            "index": idx, "bezier": bez,
            "end_binding": {"node": "end", "port_id": "e3"}})
        assert r.status_code == 200, r.text
        assert hub.pattern.objects[idx].end_binding == {"node": "end", "port_id": "e3"}
        ser = r.json()["design"]["objects"][idx]["end_binding"]
        assert ser == {"node": "end", "port_id": "e3"}

        # it survives a fresh GET (round-trip through to_dict again)
        got = c.get("/api/design").json()["objects"][idx]
        assert got["end_binding"] == {"node": "end", "port_id": "e3"}

        # an ordinary node drag (no end_binding key) leaves the binding untouched
        bez["nodes"][1]["x"] = 4.0
        c.post("/api/design/bezier", json={"index": idx, "bezier": bez})
        assert hub.pattern.objects[idx].end_binding == {"node": "end", "port_id": "e3"}

        # dragging the bound end off every port sends null -> the reference clears
        r = c.post("/api/design/bezier", json={
            "index": idx, "bezier": bez, "end_binding": None})
        assert hub.pattern.objects[idx].end_binding is None
        assert "end_binding" not in r.json()["design"]["objects"][idx]

        # a malformed binding (bad node / non-string id) normalizes to None (clear)
        c.post("/api/design/bezier", json={
            "index": idx, "bezier": bez,
            "end_binding": {"node": "middle", "port_id": "e3"}})
        assert hub.pattern.objects[idx].end_binding is None


def test_end_binding_can_arrive_on_add_and_via_points():
    """The binding also persists through the add handler (a pre-bound object) and
    the raw-points handler (a wholesale vertex replace that also rebinds)."""
    from afm_litho.litho.compiler import Pattern
    from afm_litho.service.hub import hub
    hub.pattern = Pattern(name="seed", objects=[])
    with _client() as c:
        r = c.post("/api/design/add", json={
            "kind": "path", "points_um": [[0.0, 0.0], [3.0, 0.0]],
            "end_binding": {"node": "start", "port_id": "e7"}})
        idx = r.json()["index"]
        assert hub.pattern.objects[idx].end_binding == {"node": "start", "port_id": "e7"}
        # /api/design/points carries a binding update too
        c.post("/api/design/points", json={
            "index": idx, "points_um": [[0.0, 0.0], [2.0, 1.0]],
            "end_binding": {"node": "end", "port_id": "e9"}})
        assert hub.pattern.objects[idx].end_binding == {"node": "end", "port_id": "e9"}


def test_bezier_kind_follows_closedness_join_fills_split_traces():
    """Joining a path's two ends should make it a FILLED region, and splitting a
    closed shape open should trace it again (EDIT_TOOLS). The /api/design/bezier
    handler re-derives kind from bezier.closed exactly like /api/design/add: a
    closed funnel -> electrode, any other closed outline -> shape, open -> path."""
    _seed_path()                                   # a 2-node open "path"
    assert _obj().kind == "path"
    with _client() as c:
        # close it into a generic (non-funnel) region -> filled `shape`
        square = {"closed": True, "nodes": [
            {"x": 0.0, "y": 0.0, "h_in": None, "h_out": None, "kind": "cusp"},
            {"x": 2.0, "y": 0.0, "h_in": None, "h_out": None, "kind": "cusp"},
            {"x": 2.0, "y": 2.0, "h_in": None, "h_out": None, "kind": "cusp"},
            {"x": 0.0, "y": 2.0, "h_in": None, "h_out": None, "kind": "cusp"},
        ]}
        r = c.post("/api/design/bezier", json={"index": 0, "bezier": square})
        assert r.status_code == 200, r.text
        assert _obj().kind == "shape"
        assert r.json()["design"]["objects"][0]["kind"] == "shape"

        # split it back open -> traced `path`
        openp = {"closed": False, "nodes": square["nodes"]}
        r = c.post("/api/design/bezier", json={"index": 0, "bezier": openp})
        assert r.status_code == 200, r.text
        assert _obj().kind == "path"

        # a closed FUNNEL (flat base + two curved sides to an apex) -> electrode
        funnel = {"closed": True, "nodes": [
            {"x": 0.0, "y": 0.0, "h_in": [0.2, 1.2], "h_out": None, "kind": "cusp"},
            {"x": 2.0, "y": 0.0, "h_in": None, "h_out": [1.8, 1.2], "kind": "cusp"},
            {"x": 1.0, "y": 2.0, "h_in": [1.6, 1.6], "h_out": [0.4, 1.6], "kind": "cusp"},
        ]}
        r = c.post("/api/design/bezier", json={"index": 0, "bezier": funnel})
        assert r.status_code == 200, r.text
        assert _obj().kind == "electrode"
        assert r.json()["design"]["objects"][0]["kind"] == "electrode"


def test_bad_index_and_bad_bezier_are_reported_not_raised():
    _seed_path()
    with _client() as c:
        assert "error" in c.get("/api/design/bezier/9").json()
        assert "error" in c.post(
            "/api/design/bezier", json={"index": 9, "bezier": {}}).json()
        # a one-node bezier flattens to < 2 points -> refused, not crashed
        assert "error" in c.post("/api/design/bezier", json={
            "index": 0,
            "bezier": {"closed": False,
                       "nodes": [{"x": 0, "y": 0, "kind": "cusp"}]}}).json()
