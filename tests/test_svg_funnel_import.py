"""SVG import must recognise FUNNELS, not label every closed outline a shape.

The archive designs are mostly funnels -- a flat base and two curved sides
meeting at an apex. They used to import as generic `shape` objects, which the
cockpit then mislabels "rectangle". Import now runs `BezierPath.as_funnel()` on
each closed outline: a funnel becomes an `electrode` object (the cockpit's
"funnel"), a plain rectangle stays a `shape`, and both must still COMPILE --
the funnel by filling between its two side curves.
"""


from fastapi.testclient import TestClient

# A funnel: straight base (0,0)->(1000,0), two curved sides up to the apex
# (500,1500) and back. Exactly the 3-node shape as_funnel accepts.
FUNNEL_D = ("M 0,0 L 1000,0 C 1100,500 800,1200 500,1500 "
            "C 200,1200 -100,500 0,0 Z")

# The `topo_bgd` background image (width = size_um * 1000) is what the cockpit
# writes and what a design-origin import reads the scan size from, so the design
# lands with its page top-left on the scan top-left. Embedding one makes this
# test self-sufficient rather than depending on a scan frame being loaded first.
SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="3200" height="1600" '
    'viewBox="-200 0 3200 1600">'
    '<image id="topo_bgd" x="-200" y="0" width="3200" height="3200"/>'
    f'<path d="{FUNNEL_D}"/>'
    '<rect x="1800" y="200" width="900" height="500"/>'
    '</svg>'
)


def _client():
    from afm_litho.service.app import app
    return TestClient(app)


def test_funnel_imports_as_electrode_rectangle_stays_shape():
    with _client() as c:
        r = c.post("/api/design/svg?name=funnels.svg", content=SVG)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("ok"), body
        objs = body["design"]["objects"]
        kinds = {o["name"]: o["kind"] for o in objs}
        # document order: the <path> funnel first, the <rect> second
        assert kinds["funnels-1"] == "electrode", kinds
        assert kinds["funnels-2"] == "shape", kinds


def test_as_funnel_accepts_the_imported_outline():
    """Guard the geometry directly: the outline we import really is a funnel
    to the recogniser (not just labelled one by a lucky branch)."""
    from afm_litho.litho.bezier import load_svg

    paths = load_svg(SVG, 1e-3)
    # the funnel path is the closed 3-node one
    funnels = [p for p in paths if p.closed and p.as_funnel() is not None]
    assert len(funnels) == 1


def test_imported_funnel_compiles_to_funnel_sweeps():
    with _client() as c:
        r = c.post("/api/design/svg?name=funnels.svg", content=SVG)
        assert r.status_code == 200, r.text

        tp = c.post("/api/litho/toolpath")
        assert tp.status_code == 200, tp.text
        data = tp.json()
        assert "error" not in data, data
        segs = data["segments"]
        # the funnel object must produce at least one compiled fill stroke
        funnel_segs = [s for s in segs if s["object"] == "funnels-1"]
        assert funnel_segs, [s["object"] for s in segs]
        assert all(len(s["path_um"]) >= 2 for s in funnel_segs)
