"""The `set_props` design op: edit an object's electrical properties in place.

This is the backend half of the cockpit's object-property panel (docs/
LITHO_AUTHORING.md lever C: "edit the selected object's voltage like a style").
It mirrors the `transform` op -- same `indices` selection, same undo snapshot,
same `{design: ...}` return -- but mutates the object's bias/speed/kind fields
rather than its geometry. Only fields present in the body are written, so a
panel that changes one field cannot silently reset the others.

Same harness as test_api_design.py: the module-scoped `client` fixture stands up
a TestClient; every test loads its own objects.
"""

from __future__ import annotations


def _svg(rect):
    # topo_bgd background image (width = size_um * 1000) so a design-origin import
    # can resolve the scan size the design was drawn over.
    x, y, w, h = rect
    span = max(x + w, y + h)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<image id="topo_bgd" x="0" y="0" width="{span}" height="{span}"/>'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="black"/></svg>')


def _load_one(client):
    """Import a single closed rect shape and return the design dict."""
    r = client.post("/api/design/svg?name=one.svg",
                    content=_svg((0, 0, 20000, 10000)),
                    headers={"content-type": "image/svg+xml"})
    j = r.json()
    assert j.get("ok"), j
    assert len(j["design"]["objects"]) == 1
    return j["design"]


def test_set_props_changes_voltage_and_speed(client):
    _load_one(client)
    r = client.post("/api/design/op",
                    json={"op": "set_props", "indices": [0],
                          "voltage": 3.5, "speed_um_s": 1.25}).json()
    assert r["ok"] is True
    o = r["design"]["objects"][0]
    assert o["voltage"] == 3.5
    assert o["speed_um_s"] == 1.25
    # the selection is preserved so the panel keeps the object selected
    assert r["selection"] == [0]


def test_set_props_only_touches_fields_that_are_sent(client):
    d = _load_one(client)
    before_speed = d["objects"][0]["speed_um_s"]
    # send voltage alone; speed must survive untouched
    r = client.post("/api/design/op",
                    json={"op": "set_props", "indices": [0], "voltage": 9.0}).json()
    o = r["design"]["objects"][0]
    assert o["voltage"] == 9.0
    assert o["speed_um_s"] == before_speed


def test_set_props_sets_voltage_end_ramp_and_clears_it(client):
    _load_one(client)
    ramp = client.post("/api/design/op",
                       json={"op": "set_props", "indices": [0],
                             "voltage": 2.0, "voltage_end": 6.0}).json()
    assert ramp["design"]["objects"][0]["voltage_end"] == 6.0
    # an explicit null clears the ramp back to a flat bias
    flat = client.post("/api/design/op",
                      json={"op": "set_props", "indices": [0],
                            "voltage_end": None}).json()
    assert flat["design"]["objects"][0]["voltage_end"] is None


def test_set_props_accepts_pen_as_kind_alias(client):
    _load_one(client)
    r = client.post("/api/design/op",
                    json={"op": "set_props", "indices": [0], "pen": "path"}).json()
    assert r["ok"] is True
    assert r["design"]["objects"][0]["kind"] == "path"


def test_set_props_edits_only_the_selected_object(client):
    # two objects; change one, the other is untouched
    from afm_litho.service.app import hub
    client.post("/api/sketch/clear")
    client.post("/api/sketch", json={"pen": "write", "mode": "literal",
                                     "points_um": [[0, 0], [5, 0]]})
    client.post("/api/sketch", json={"pen": "write", "mode": "literal",
                                     "points_um": [[0, 1], [5, 1]]})
    v0_before = hub.pattern.objects[0].voltage
    r = client.post("/api/design/op",
                    json={"op": "set_props", "indices": [1], "voltage": -7.0}).json()
    objs = r["design"]["objects"]
    assert objs[1]["voltage"] == -7.0
    assert objs[0]["voltage"] == v0_before        # object 0 unchanged


def test_set_props_is_undoable(client):
    _load_one(client)
    client.post("/api/design/op",
                json={"op": "set_props", "indices": [0], "voltage": 1.0})
    r = client.post("/api/design/op",
                    json={"op": "set_props", "indices": [0], "voltage": 4.0}).json()
    assert r["design"]["objects"][0]["voltage"] == 4.0
    undone = client.post("/api/sketch/undo").json()
    # one undo steps back to the 1.0 write (set_props took a snapshot)
    assert undone["design"]["objects"][0]["voltage"] == 1.0
