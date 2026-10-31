"""The properties panel toggles `enabled`, and the object list can be reordered.

Two backend affordances the litho cockpit needs. `enabled` already exists on
`LithoObject` and the compiler already SKIPS disabled objects, so `set_props`
just has to accept the flag without clobbering the electrical fields. And
`pattern.objects` list order IS the write order, so `/api/design/reorder`
rewrites that order from a caller-supplied permutation (drag to top -> writes
first) -- and must refuse anything that is not a clean permutation.
"""


from fastapi.testclient import TestClient


def _client():
    from afm_litho.service.app import app
    return TestClient(app)


def _seed(names):
    """Put a fresh pattern of named path objects on the hub, return the hub."""
    from afm_litho.litho.compiler import LithoObject, Pattern
    from afm_litho.service.hub import hub
    objs = [LithoObject(nm, "path", [(0.0, float(i)), (1.0, float(i))],
                        voltage=8.0, speed_um_s=4.0)
            for i, nm in enumerate(names)]
    hub.pattern = Pattern(name="seed", objects=objs)
    return hub


def test_set_props_toggles_enabled_without_touching_other_fields():
    with _client() as c:
        _seed(["a", "b"])
        r = c.post("/api/design/op",
                   json={"op": "set_props", "indices": [0], "enabled": False})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("ok"), body
        o0 = body["design"]["objects"][0]
        assert o0["enabled"] is False, o0
        # geometry/electrical fields left alone
        assert o0["voltage"] == 8.0, o0
        assert o0["speed_um_s"] == 4.0, o0

        # and back to enabled
        r = c.post("/api/design/op",
                   json={"op": "set_props", "indices": [0], "enabled": True})
        assert r.status_code == 200, r.text
        assert r.json()["design"]["objects"][0]["enabled"] is True


def test_reorder_moves_last_object_to_the_front():
    with _client() as c:
        _seed(["a", "b", "c"])
        # move the last object (c) to the front: [2, 0, 1]
        r = c.post("/api/design/reorder", json={"order": [2, 0, 1]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert "error" not in body, body
        names = [o["name"] for o in body["design"]["objects"]]
        assert names == ["c", "a", "b"], names
        # same objects, none lost or duplicated
        assert sorted(names) == ["a", "b", "c"], names


def test_reorder_rejects_a_non_permutation_and_does_not_mutate():
    with _client() as c:
        _seed(["a", "b", "c"])
        # [0, 0, 1] is the right length but not a permutation of 0..2
        r = c.post("/api/design/reorder", json={"order": [0, 0, 1]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert "error" in body, body
        names = [o["name"] for o in body["design"]["objects"]]
        assert names == ["a", "b", "c"], names  # unchanged

        # a wrong-length order is rejected too
        r = c.post("/api/design/reorder", json={"order": [0, 1]})
        body = r.json()
        assert "error" in body, body
        assert [o["name"] for o in body["design"]["objects"]] == ["a", "b", "c"]
