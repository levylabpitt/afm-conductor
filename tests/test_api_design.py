"""HTTP-layer tests for the design-editing and shapes
endpoints -- the half of app.py the original test_api.py never drove.

Same harness as test_api.py: the module-scoped `client` fixture (tests/conftest.py)
is a FastAPI TestClient. The `hub` is a process
global, so every test here loads or sketches its own objects and resets the state
it touches; nothing depends on another test having run first.
"""

from __future__ import annotations

# `client` comes from tests/conftest.py (a TestClient).


# --------------------------------------------------------------- helpers --

# A design-origin import needs the scan size the design was drawn over. The
# cockpit writes it into the SVG as a `topo_bgd` background image (width =
# size_um * 1000); these helpers embed one so the import resolves a size the way
# a real drawing does, rather than depending on a scan frame being loaded first.
# Extent- and offset-based assertions here are invariant under the resulting
# translate + y-flip, so only the presence of the image is new.
def _bg(span_units):
    return (f'<image id="topo_bgd" x="0" y="0" '
            f'width="{span_units}" height="{span_units}"/>')


def _svg(rect):
    x, y, w, h = rect
    span = max(x + w, y + h)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'{_bg(span)}'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="black"/></svg>')


def _svg_two(a, b):
    """Two rects in one file -> two overlapping (or separate) closed shapes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    span = max(ax + aw, bx + bw, ay + ah, by + bh)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{span}" height="{span}" viewBox="0 0 {span} {span}">'
            f'{_bg(span)}'
            f'<rect x="{ax}" y="{ay}" width="{aw}" height="{ah}" fill="black"/>'
            f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="black"/></svg>')


def _post_svg(client, svg, name="t.svg"):
    return client.post(f"/api/design/svg?name={name}",
                       content=svg, headers={"content-type": "image/svg+xml"})


def _sketch(client, pts, pen="write", mode="literal"):
    return client.post("/api/sketch",
                       json={"pen": pen, "points_um": pts, "mode": mode})


def _objects(design):
    return design["objects"]


def _load_two_shapes(client, overlap=True):
    """Import a design of two closed rect shapes, returning the design dict."""
    if overlap:
        svg = _svg_two((0, 0, 12000, 12000), (6000, 6000, 12000, 12000))
    else:
        svg = _svg_two((0, 0, 12000, 12000), (16000, 16000, 12000, 12000))
    j = _post_svg(client, svg, name="two.svg").json()
    assert j.get("ok"), j
    assert len(_objects(j["design"])) == 2
    assert all(o["kind"] == "shape" for o in _objects(j["design"]))
    return j["design"]


def _tmp_store(tmp_path):
    """Redirect the session archive to a temp file so tests do not append to the
    real runtime/sessions.jsonl, and return the live hub for assertions."""
    from afm_litho.service.app import hub
    from afm_litho.service.sessions import SessionStore
    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    return hub


# =============================================================== sketch ==

def test_sketch_appends_an_object(client):
    client.post("/api/sketch/clear")
    r = _sketch(client, [[0, 0], [5, 0], [5, 5]])
    assert r.status_code == 200
    j = r.json()
    assert j.get("ok") is True
    assert j["added"] >= 1
    assert len(_objects(j["design"])) == j["added"]
    assert "report" in j


def test_sketch_rejects_a_stroke_too_short(client):
    r = _sketch(client, [[0, 0]])          # one point is not a stroke
    assert r.json() == {"error": "stroke too short"}


def test_sketch_undo_removes_the_last_object(client):
    client.post("/api/sketch/clear")
    _sketch(client, [[0, 0], [4, 0], [4, 4]])
    n_after_first = len(_objects(_sketch(client, [[1, 1], [6, 1], [6, 6]]).json()["design"]))
    r = client.post("/api/sketch/undo").json()
    assert r["ok"] is True
    assert len(_objects(r["design"])) == n_after_first - 1


def test_sketch_undo_reports_nothing_to_undo_on_empty_history(client):
    # Empty the pattern and its history, then undo has nothing to step back to. A stroke
    # first, so the pattern exists whatever test ran before this one.
    _sketch(client, [[0, 0], [5, 0], [5, 5]])
    client.post("/api/sketch/clear")
    from afm_litho.service.routers.design import _design_history
    _design_history.clear()
    r = client.post("/api/sketch/undo").json()
    assert r["ok"] is True
    # nothing removed: an already-empty canvas stays empty
    assert _objects(r["design"]) == []


def test_sketch_clear_empties_the_pattern(client):
    _sketch(client, [[0, 0], [5, 0], [5, 5]])
    r = client.post("/api/sketch/clear").json()
    assert r == {"ok": True}
    # the pattern is now empty; a toolpath export confirms nothing is left
    exp = client.get("/api/design/svg").json()
    assert exp.get("error") == "no Bezier shapes in the current design"


# =========================================================== design/op ==

def test_design_op_delete_removes_the_selection(client):
    _load_two_shapes(client)
    r = client.post("/api/design/op", json={"op": "delete", "indices": [0]}).json()
    assert r["ok"] is True
    assert len(_objects(r["design"])) == 1
    assert r["selection"] == []            # nothing survives a delete


def test_design_op_union_merges_overlapping_regions(client):
    _load_two_shapes(client, overlap=True)
    r = client.post("/api/design/op", json={"op": "union", "indices": [0, 1]}).json()
    assert r["ok"] is True
    assert len(_objects(r["design"])) == 1    # two overlapping regions -> one
    assert "Union" in r["message"]


def test_design_op_difference_subtracts_one_region(client):
    _load_two_shapes(client, overlap=True)
    r = client.post("/api/design/op", json={"op": "difference", "indices": [0, 1]}).json()
    assert r["ok"] is True
    assert len(_objects(r["design"])) >= 1
    assert "Difference" in r["message"]


def test_design_op_duplicate_adds_a_copy(client):
    _load_two_shapes(client)
    r = client.post("/api/design/op",
                    json={"op": "duplicate", "indices": [0], "dx_um": 1.0}).json()
    assert r["ok"] is True
    assert len(_objects(r["design"])) == 3       # 2 + 1 copy
    # the copy is what the panel should now have selected
    assert len(r["selection"]) == 1


def _dup(client, name, **kw):
    i = [o["name"] for o in _objects(client.get("/api/design").json())].index(name)
    r = client.post("/api/design/op",
                    json={"op": "duplicate", "indices": [i], **kw}).json()
    return _objects(r["design"])[r["selection"][0]]


def test_duplicate_names_copies_stem_copy_k_never_a_drawn_id(client):
    _load_two_shapes(client)
    names = [o["name"] for o in _objects(client.get("/api/design").json())]
    assert names == ["two-1", "two-2"]
    # the stem drops a trailing -k and an existing -copy[-k]
    copies = [_dup(client, "two-1")["name"] for _ in range(3)]
    assert copies == ["two-copy", "two-copy-2", "two-copy-3"]
    assert _dup(client, "two-2")["name"] == "two-copy-4"
    assert _dup(client, "two-copy-2")["name"] == "two-copy-5"


def test_duplicate_after_a_delete_never_takes_the_next_drawn_id(client):
    """/add shape, /add shape (shape-2), delete shape-2, duplicate shape: the copy
    must not become "shape-2", the id the next drawn object and scripts use."""
    sq = [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]
    for _ in range(2):
        assert client.post("/api/design/add", json={"kind": "shape",
                                                     "points_um": sq}).json()["ok"]
    names = [o["name"] for o in _objects(client.get("/api/design").json())]
    assert names[-2:] == ["shape", "shape-2"]
    r = client.post("/api/design/op", json={"op": "delete",
                                            "indices": [len(names) - 1]}).json()
    assert r["ok"]
    assert _dup(client, "shape")["name"] == "shape-copy"
    # /add naming for a plain name is unchanged
    r = client.post("/api/design/add", json={"kind": "shape", "points_um": sq}).json()
    assert r["name"] == "shape-2"


def test_duplicate_clears_the_port_binding(client):
    r = client.post("/api/design/add", json={
        "kind": "path", "name": "wire", "points_um": [[0, 0], [3, 0]],
        "end_binding": {"node": "end", "port_id": "p1"}}).json()
    assert r["ok"], r
    src = next(o for o in _objects(r["design"]) if o["name"] == "wire")
    assert src.get("end_binding") == {"node": "end", "port_id": "p1"}
    cp = _dup(client, "wire", dx_um=1.0, dy_um=-1.0)
    assert cp["name"] == "wire-copy"
    assert cp.get("end_binding") is None
    src = next(o for o in _objects(client.get("/api/design").json()) if o["name"] == "wire")
    assert src.get("end_binding") == {"node": "end", "port_id": "p1"}


def test_transform_moves_the_radial_fill_centre_with_the_region():
    from afm_litho.litho import pathops
    from afm_litho.litho.compiler import LithoObject
    o = LithoObject("r", "shape", [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)],
                    fill_kind="radial-out", fill_centre_um=(1.0, 1.0))
    pathops.transform(o, dx_um=1.0, dy_um=-1.0)
    assert o.fill_centre_um == (2.0, 0.0)


def test_design_op_outset_and_inset_change_the_geometry(client):
    _load_two_shapes(client)
    out = client.post("/api/design/op",
                      json={"op": "outset", "indices": [0], "delta_um": 0.5}).json()
    assert out["ok"] is True
    assert "Outset" in out["message"]
    ins = client.post("/api/design/op",
                      json={"op": "inset", "indices": [0], "delta_um": 0.3}).json()
    assert ins["ok"] is True
    assert "Inset" in ins["message"]


def test_design_op_simplify_reports_vertex_counts(client):
    _load_two_shapes(client)
    r = client.post("/api/design/op",
                    json={"op": "simplify", "indices": [0], "tol_um": 0.05}).json()
    assert r["ok"] is True
    assert "Simplify" in r["message"]
    # object count is preserved; simplify only thins vertices
    assert len(_objects(r["design"])) == 2


def test_design_op_transform_translates_the_object(client):
    d = _load_two_shapes(client)
    p0 = _objects(d)[0]["points_um"][0]
    r = client.post("/api/design/op",
                    json={"op": "transform", "indices": [0], "dx_um": 3.0, "dy_um": 0.0}).json()
    assert r["ok"] is True
    p1 = _objects(r["design"])[0]["points_um"][0]
    assert round(p1[0] - p0[0], 3) == 3.0        # moved +3 um in x
    assert round(p1[1] - p0[1], 3) == 0.0


def test_design_op_zorder_raise_swaps_neighbours(client):
    d = _load_two_shapes(client, overlap=False)
    names0 = [o["name"] for o in _objects(d)]
    r = client.post("/api/design/op", json={"op": "raise", "indices": [0]}).json()
    assert r["ok"] is True
    names1 = [o["name"] for o in _objects(r["design"])]
    assert names1 == list(reversed(names0))      # index 0 rose above index 1


def test_design_op_zorder_top_moves_to_end(client):
    d = _load_two_shapes(client, overlap=False)
    first = _objects(d)[0]["name"]
    r = client.post("/api/design/op", json={"op": "top", "indices": [0]}).json()
    assert r["ok"] is True
    assert _objects(r["design"])[-1]["name"] == first   # raised to the top of the stack


def test_design_op_rejects_an_unknown_op(client):
    _load_two_shapes(client)
    r = client.post("/api/design/op", json={"op": "frobnicate", "indices": [0]}).json()
    assert r["error"] == "no such operation: frobnicate"


def test_design_op_rejects_a_bad_index(client):
    _load_two_shapes(client)
    r = client.post("/api/design/op", json={"op": "delete", "indices": [99]}).json()
    assert "no object at index 99" in r["error"]


# =========================================================== design/svg ==

def test_design_svg_export_returns_svg_bytes(client):
    _post_svg(client, _svg((0, 0, 20000, 10000)))
    r = client.get("/api/design/svg")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/svg+xml")
    assert b"<svg" in r.content


def test_design_svg_export_reports_when_there_is_nothing_to_export(client):
    client.post("/api/sketch/clear")
    r = client.get("/api/design/svg").json()
    assert r == {"error": "no Bezier shapes in the current design"}


# =============================================================== shapes ==

def test_shape_toolpath_returns_the_fill_strokes(client):
    _post_svg(client, _svg((0, 0, 20000, 10000)))
    r = client.get("/api/shapes/0/toolpath").json()
    assert r["strokes"] >= 1
    assert r["points"] >= 2
    assert len(r["paths"]) == r["strokes"]
    assert len(r["paths"][0][0]) == 2            # each vertex is an [x, y] pair


def test_shape_update_edits_the_outline(client):
    _post_svg(client, _svg((0, 0, 20000, 10000)))
    bez = {"nodes": [
        {"x": 0.0, "y": 0.0, "h_in": None, "h_out": None, "kind": "cusp"},
        {"x": 12.0, "y": 0.0, "h_in": None, "h_out": None, "kind": "cusp"},
        {"x": 12.0, "y": 7.0, "h_in": None, "h_out": None, "kind": "cusp"},
        {"x": 0.0, "y": 7.0, "h_in": None, "h_out": None, "kind": "cusp"}],
        "closed": True}
    r = client.post("/api/shapes/0", json={"bezier": bez}).json()
    assert r["ok"] is True
    pts = _objects(r["design"])[0]["points_um"]
    # the re-flattened outline follows the new 12x7 box, not the old 20x10
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    assert round(max(xs) - min(xs), 2) == 12.0
    assert round(max(ys) - min(ys), 2) == 7.0


def test_shape_update_rejects_a_missing_object(client):
    _post_svg(client, _svg((0, 0, 20000, 10000)))
    r = client.post("/api/shapes/99", json={"bezier": {"nodes": [], "closed": True}}).json()
    assert r == {"error": "no such object"}


def test_shape_update_rejects_a_non_shape(client):
    # A bare sketched stroke is a `path`, not an editable shape.
    client.post("/api/sketch/clear")
    _sketch(client, [[0, 0], [5, 0]])
    r = client.post("/api/shapes/0", json={"bezier": {"nodes": [], "closed": True}}).json()
    assert "not an editable shape" in r["error"]


# ============================================================= contacts ==

def test_contacts_round_trip(client):
    pills = [{"x_um": 1.0, "y_um": 2.0, "w_um": 1.0, "h_um": 0.5, "angle_deg": 0.0}]
    r = client.post("/api/contacts", json={"contacts": pills}).json()
    assert r["ok"] is True and r["n"] == 1
    g = client.get("/api/contacts").json()
    assert g["defined"] is True
    assert len(g["contacts"]) == 1
    assert g["contacts"][0]["x_um"] == 1.0
    # reset so the erased-state does not leak into other tests
    client.post("/api/contacts", json={"contacts": []})


# ============================================================= sessions ==

def test_sessions_recall_returns_a_recorded_session(client, tmp_path):
    hub = _tmp_store(tmp_path)
    rec = hub.store.record("scan", {"label": "recall me"})
    r = client.get("/api/sessions/recall", params={"id": rec["id"]}).json()
    assert r["id"] == rec["id"]
    assert r["label"] == "recall me"


def test_sessions_recall_reports_an_unknown_id(client, tmp_path):
    _tmp_store(tmp_path)
    r = client.get("/api/sessions/recall", params={"id": "no-such-id"}).json()
    assert r == {"error": "unknown session"}
