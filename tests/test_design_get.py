"""GET /api/design serves the current design (hub.pattern) so a client -- the
litho cockpit -- can hydrate from the backend on load instead of booting with
placeholder objects. It must return an empty objects list when nothing is
loaded, and the real objects (with points, voltage, enabled, ...) once a design
is in hand. The shape is exactly the SVG import's `design`, so the frontend can
consume the two identically.
"""


from fastapi.testclient import TestClient

# A funnel path + a rectangle -- enough to give hub.pattern real objects via the
# existing import route, without hand-building a Pattern.
FUNNEL_D = ("M 0,0 L 1000,0 C 1100,500 800,1200 500,1500 "
            "C 200,1200 -100,500 0,0 Z")
SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="3200" height="1600" '
    'viewBox="-200 0 3200 1600">'
    # topo_bgd background image: a design-origin import reads the scan size from
    # it (width = size_um * 1000), so the design is self-sufficient here.
    '<image id="topo_bgd" x="-200" y="0" width="3200" height="3200"/>'
    f'<path d="{FUNNEL_D}"/>'
    '<rect x="1800" y="200" width="900" height="500"/>'
    '</svg>'
)


def _client():
    from afm_litho.service.app import app
    return TestClient(app)


def test_design_get_empty_when_nothing_loaded():
    from afm_litho.service.hub import hub
    hub.pattern = None                       # nothing loaded
    with _client() as c:
        r = c.get("/api/design")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["objects"] == [], body
        # name may be None when empty
        assert body.get("name") is None, body


def test_design_get_returns_the_loaded_design():
    with _client() as c:
        imp = c.post("/api/design/svg?name=funnels.svg", content=SVG)
        assert imp.status_code == 200, imp.text
        assert imp.json().get("ok"), imp.text

        r = c.get("/api/design")
        assert r.status_code == 200, r.text
        body = r.json()
        objs = body["objects"]
        assert len(objs) >= 1, body
        names = {o["name"] for o in objs}
        assert "funnels-1" in names, names
        # every object carries the fields the cockpit hydrates from
        for o in objs:
            assert "points_um" in o and len(o["points_um"]) >= 2, o
            assert "voltage" in o, o
            assert "enabled" in o, o
        # and it matches the import response's design exactly (same shape)
        assert body == imp.json()["design"], (body, imp.json()["design"])
