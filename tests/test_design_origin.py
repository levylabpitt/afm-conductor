"""The design-origin rule: the top-left of the SVG page is the top-left of the scan.

`docs/DESIGN_ORIGIN_SPEC.md`. An SVG design imported through either door
(`POST /api/design/svg`, FLEX `loadPattern`) is placed with its page origin on
the scan origin: `load_svg(..., preserve_origin=True)` keeps page-origin microns
(x right, y DOWN), and `litho.frame.svg_to_lab` maps them to centred lab microns
once, `x_lab = x_um - size_um/2`, `y_lab = size_um/2 - y_um`. `size_um` is the
scan the design was drawn over -- the design's own `topo_bgd` background image
when present, else the current scan frame, else the import is refused.
"""

from __future__ import annotations

import contextlib
import types

import pytest

from afm_litho.litho.bezier import load_svg
from afm_litho.litho.frame import LithoFrame, bezier_paths_to_lab, svg_to_lab

# `client` comes from tests/conftest.py (a TestClient).


def _stub_hub(size_um=None, last=None):
    """A minimal hub for the resolver: an optional current scan and last-scan."""
    scan = None
    if size_um is not None:
        scan = types.SimpleNamespace(config=types.SimpleNamespace(size_um=size_um))
    hub = types.SimpleNamespace(inst=types.SimpleNamespace(scan=scan),
                                last_scan_config=last)
    hub.host = hub  # fake: host fields live on the fake itself
    return hub


# A 1000 x 1000-unit rect at the page origin, drawn over a 30 um scan (the
# topo_bgd image is 30 um * 1000 = 30000 units wide). Exactly the plan's fixture.
RECT_30UM = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30000 30000">'
    '<image id="topo_bgd" x="0" y="0" width="30000" height="30000"/>'
    '<rect x="0" y="0" width="1000" height="1000"/>'
    '</svg>'
)


# ------------------------------------------------------------ pure mapping ----

def test_svg_to_lab_translates_and_flips_y_without_scale():
    # page top-left, one unit right, one unit down, and the page centre.
    pts = svg_to_lab([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (15.0, 15.0)], 30.0)
    assert pts[0] == (-15.0, 15.0)      # page top-left -> lab top-left
    assert pts[1] == (-14.0, 15.0)      # +1 um right  -> +1 lab x
    assert pts[2] == (-15.0, 14.0)      # +1 um DOWN   -> -1 lab y
    assert pts[3] == (0.0, 0.0)         # page centre  -> lab origin


def test_load_svg_preserve_origin_keeps_the_page_origin_y_down():
    # a unit rect at the page origin: x right, y measured DOWN from the top, no
    # viewBox-bottom flip and no bbox centring.
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">'
           '<rect x="0" y="0" width="1000" height="1000"/></svg>')
    p = load_svg(svg, 1e-3, preserve_origin=True)[0]
    f = p.flatten(0.001)
    assert f[:, 0].min() == pytest.approx(0.0)
    assert f[:, 0].max() == pytest.approx(1.0)      # 1000 units * 1e-3 = 1 um
    assert f[:, 1].min() == pytest.approx(0.0)      # top of the page is y=0
    assert f[:, 1].max() == pytest.approx(1.0)      # y grows DOWNWARD


def test_bezier_paths_to_lab_maps_the_rect_corners():
    p = load_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">'
                 '<rect x="0" y="0" width="1000" height="1000"/></svg>',
                 1e-3, preserve_origin=True)
    bezier_paths_to_lab(p, 30.0)
    corners = {(round(n.x, 6), round(n.y, 6)) for n in p[0].nodes}
    # (0,0),(1,0),(1,1),(0,1) page microns -> the four lab corners
    assert corners == {(-15.0, 15.0), (-14.0, 15.0), (-14.0, 14.0), (-15.0, 14.0)}


# --------------------------------------------------------- size resolution ----

def test_resolve_size_prefers_the_scan_frame_over_the_image():
    # The current scan the operator is looking at wins over the embedded image.
    from afm_litho.service.design_import import resolve_import_size_um
    size, source = resolve_import_size_um(RECT_30UM, _stub_hub(size_um=12.0))
    assert (size, source) == (12.0, "frame")     # frame wins over the 30 um image


def test_resolve_size_stale_picture_frame_wins():
    # Mock_Device.svg case: a 40 um page around a stale 20 um picture, but the UI
    # is on a 30 um scan -> the live frame is authoritative.
    from afm_litho.service.design_import import resolve_import_size_um
    stale = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40000 40000">'
             '<image id="topo_bgd" x="0" y="0" width="20000" height="20000"/>'
             '<rect x="0" y="0" width="1000" height="1000"/></svg>')
    size, source = resolve_import_size_um(stale, _stub_hub(size_um=30.0))
    assert (size, source) == (30.0, "frame")


def test_resolve_size_uses_the_image_when_no_scan_frame():
    # Only when there is no scan frame at all does the background image size it.
    from afm_litho.service.design_import import resolve_import_size_um
    size, source = resolve_import_size_um(RECT_30UM, _stub_hub())
    assert (size, source) == (30.0, "image")


def test_resolve_size_uses_last_scan_config_when_no_live_scan():
    from afm_litho.service.design_import import resolve_import_size_um
    no_bg = ('<svg xmlns="http://www.w3.org/2000/svg">'
             '<path d="M 0 0 L 1000 0"/></svg>')
    size, source = resolve_import_size_um(no_bg, _stub_hub(last={"size_um": 40.0}))
    assert (size, source) == (40.0, "frame")


def test_resolve_size_refuses_when_neither_present():
    from afm_litho.service.design_import import DesignSizeError, resolve_import_size_um
    no_bg = ('<svg xmlns="http://www.w3.org/2000/svg">'
             '<path d="M 0 0 L 1000 0"/></svg>')
    with pytest.raises(DesignSizeError):
        resolve_import_size_um(no_bg, _stub_hub())


# --------------------------------------------------------------- REST door ----

def _objects(design):
    return design["objects"]


@contextlib.contextmanager
def _scan_frame(size_um):
    """Force the process-global hub's scan frame to `size_um` (or clear it with
    None), restoring whatever was there. Frame now wins over the image, so an
    image-source test has to guarantee NO frame, and a frame test has to set one."""
    from afm_litho.service.hub import hub
    saved_scan = getattr(hub.host.inst, "scan", None)
    saved_last = getattr(hub.host, "last_scan_config", None)
    hub.host.inst.scan = None
    hub.host.last_scan_config = None if size_um is None else {"size_um": size_um}
    try:
        yield
    finally:
        hub.host.inst.scan = saved_scan
        hub.host.last_scan_config = saved_last


def test_rest_import_places_the_rect_at_the_page_corner(client):
    # No scan frame -> the 30 um background image sizes the import.
    with _scan_frame(None):
        r = client.post("/api/design/svg?name=corner.svg", content=RECT_30UM,
                        headers={"content-type": "image/svg+xml"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["origin"] == {"size_um": 30.0, "source": "image",
                              "outside_frame": False}
    pts = _objects(body["design"])[0]["points_um"]
    corners = {(round(x, 4), round(y, 4)) for x, y in pts}
    assert {(-15.0, 15.0), (-14.0, 15.0), (-14.0, 14.0), (-15.0, 14.0)} <= corners


def test_rest_import_refuses_without_a_size(client):
    # Genuinely "no frame, no image" -> a 4xx refusal (design-origin rule).
    with _scan_frame(None):
        no_bg = ('<svg xmlns="http://www.w3.org/2000/svg">'
                 '<path d="M 0 0 L 1000 0"/></svg>')
        r = client.post("/api/design/svg?name=nobg.svg", content=no_bg,
                        headers={"content-type": "image/svg+xml"})
    assert r.status_code == 400, r.text
    assert "error" in r.json()


# -------------------------------------------------- both doors agree ----------

def test_both_doors_give_identical_coordinates(client):
    """The REST build loop and the FLEX `_svg_pattern` loop must land the same
    geometry for the same SVG (a wire, so no funnel/fill difference between them)."""
    from afm_litho.service.host.litho_cmds import _svg_pattern
    wire = ('<svg xmlns="http://www.w3.org/2000/svg">'
            '<image id="topo_bgd" x="0" y="0" width="1000" height="1000"/>'
            '<path d="M 100 200 L 900 800"/></svg>')
    # FLEX door, called directly with no scan frame -> the 1 um image sizes it.
    flex_pat, flex_origin = _svg_pattern(wire, name="wire.svg", hub=_stub_hub())
    flex_pts = flex_pat.to_dict()["objects"][0]["points_um"]
    assert flex_origin == {"size_um": 1.0, "source": "image"}

    # REST door, through the TestClient, with the scan frame cleared so the same
    # image sizes it. `outside_frame` (added by the door) is the only extra key.
    with _scan_frame(None):
        body = client.post("/api/design/svg?name=wire.svg", content=wire,
                           headers={"content-type": "image/svg+xml"}).json()
    assert body["origin"] == {"size_um": 1.0, "source": "image",
                              "outside_frame": False}
    rest_pts = body["design"]["objects"][0]["points_um"]
    assert len(rest_pts) == len(flex_pts)
    for (rx, ry), (fx, fy) in zip(rest_pts, flex_pts):
        assert rx == pytest.approx(fx) and ry == pytest.approx(fy)


# -------------------------------------------------- outside-frame flag --------

def _spy_log(monkeypatch):
    """Capture hub.log messages during a REST import."""
    from afm_litho.service.hub import hub
    msgs = []
    orig = hub.log

    async def _log(msg, *a, **k):
        msgs.append(str(msg))
        return await orig(msg, *a, **k)

    monkeypatch.setattr(hub, "log", _log)
    return msgs


def test_out_of_frame_design_is_flagged_and_warned(client, monkeypatch):
    # A 40 um-wide wire placed over a 10 um scan frame runs well past +/- 5 um.
    msgs = _spy_log(monkeypatch)
    wire = ('<svg xmlns="http://www.w3.org/2000/svg">'
            '<path d="M 0 0 L 40000 0"/></svg>')
    with _scan_frame(10.0):
        body = client.post("/api/design/svg?name=big.svg", content=wire,
                           headers={"content-type": "image/svg+xml"}).json()
    assert body["origin"]["size_um"] == 10.0
    assert body["origin"]["source"] == "frame"
    assert body["origin"]["outside_frame"] is True          # placed, not refused
    assert any("outside the 10 um frame" in m for m in msgs), msgs


def test_in_frame_design_is_not_flagged(client, monkeypatch):
    msgs = _spy_log(monkeypatch)
    wire = ('<svg xmlns="http://www.w3.org/2000/svg">'
            '<path d="M 0 0 L 8000 0"/></svg>')       # 8 um across a 30 um frame
    with _scan_frame(30.0):
        body = client.post("/api/design/svg?name=small.svg", content=wire,
                           headers={"content-type": "image/svg+xml"}).json()
    assert body["origin"]["outside_frame"] is False
    assert not any("outside the" in m for m in msgs), msgs


# -------------------------------------------------- twin end-to-end -----------

def test_twin_compile_writes_the_wire_at_the_frame_top_left(client):
    """The cockpit picture and the tip agree: a wire drawn from the page top-left
    imports to lab top-left, and compiling it (identity litho frame, the twin's
    default) puts the FIRST out-wave point there too."""
    from afm_litho.litho.compiler import compile_pattern
    from afm_litho.service.hub import hub
    size = 30.0
    wire = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30000 30000">'
            '<image id="topo_bgd" x="0" y="0" width="30000" height="30000"/>'
            '<path d="M 0 0 L 15000 15000"/></svg>')
    with _scan_frame(None):     # no frame -> the 30 um image sizes it
        body = client.post("/api/design/svg?name=twin.svg", content=wire,
                           headers={"content-type": "image/svg+xml"}).json()
    assert body["origin"] == {"size_um": size, "source": "image",
                              "outside_frame": False}
    first_vertex = _objects(body["design"])[0]["points_um"][0]
    assert first_vertex[0] == pytest.approx(-size / 2)   # lab top-left x
    assert first_vertex[1] == pytest.approx(size / 2)    # lab top-left y

    # Compile through the identity frame the twin uses; the first out-wave point
    # (lab um) is the wire's start, i.e. the frame's top-left corner.
    waves = compile_pattern(hub.pattern, 1.0, 1.0, 0.0, 0.0, frame=LithoFrame())
    first_point = waves["segments"][0]["path_um"][0]
    assert first_point[0] == pytest.approx(-size / 2)
    assert first_point[1] == pytest.approx(size / 2)
