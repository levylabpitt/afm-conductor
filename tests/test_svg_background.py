"""SVG background swap (docs/IMAGE_SAVING_NOTES.md 2026-09-17):

  - `swap_background` (service/svg_background.py): pure XML transform. Must
    round-trip an Inkscape file's namespaces with no `ns0:` prefixes, pick
    the BOTTOM-MOST `<image>` (first in document order) that is bigger than
    100x100 user units, skip small icons, insert one when none qualifies,
    LINK the PNG (never embed), and stamp `id="topo_bgd"`.
  - `POST /api/design/background` (routers/design.py): the Scan-tab button's
    route. Finds the newest scan PNG for a channel by globbing the Slice-A
    contract path (`scans/png/<stem>_<channel>.png` + sidecar
    `scans/json/<stem>.json`), writes `scans/svg/<stem>_<channel>_<name>.svg`,
    and 400s on each of: no design loaded, no PNG for that channel, no
    sidecar (size unknown).

The route tests build the fixture PNG/sidecar files by hand under a temp
runtime dir (retargeting `hub.store`, exactly like test_scan_last.py) --
independent of however Slice A's writers actually produce them.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET

from fastapi.testclient import TestClient

from afm_litho.service.svg_background import swap_background

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
# The REAL Inkscape sodipodi URI (ends .dtd, not .0 -- see svg_background.py).
# A previous version of this fixture shared the .0 typo with the buggy
# constant, so the namespace test passed without ever exercising the real
# registration; a real Inkscape file caught it after this branch merged.
SODIPODI_NS = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
CC_NS = "http://creativecommons.org/ns#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


def _client():
    from afm_litho.service.app import app
    return TestClient(app)


# ------------------------------------------------------- swap_background --

INKSCAPE_SVG = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns:dc="http://purl.org/dc/elements/1.1/"
     xmlns:cc="http://creativecommons.org/ns#"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:svg="http://www.w3.org/2000/svg"
     xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     width="1000" height="1000" viewBox="0 0 1000 1000">
  <metadata id="metadata1">
    <rdf:RDF>
      <cc:Work rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/>
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <sodipodi:namedview id="nv" pagecolor="#ffffff" inkscape:zoom="2.5"/>
  <defs id="defs1"/>
  <g inkscape:label="Layer 1" inkscape:groupmode="layer" id="layer1">
    <image id="icon" x="5" y="5" width="20" height="20"
           xlink:href="icon.png"/>
    <image id="bg" x="10" y="20" width="500" height="500"
           transform="rotate(3)" xlink:href="old.png" href="old.png"/>
    <rect id="wire1" width="10" height="10" style="fill:#000"/>
  </g>
</svg>"""


def test_namespaces_survive_with_no_ns0_prefix():
    out = swap_background(INKSCAPE_SVG, "../png/x_height.png", 5.0)
    # No invented prefixes ANYWHERE (ns0:, ns1:, ns2:, ...) -- every namespace
    # actually present in the file (svg default, xlink, inkscape, sodipodi,
    # cc, rdf) must be registered under its real name.
    assert not re.search(r"\bns\d+:", out), out
    assert "xlink:href" in out
    assert "inkscape:label" in out
    assert "sodipodi:namedview" in out
    assert "xmlns:cc=" in out
    assert "xmlns:rdf=" in out
    root = ET.fromstring(out)
    # sodipodi/inkscape elements and attributes are still there, correctly
    # namespaced, not dropped or mangled.
    nv = root.find(f"{{{SODIPODI_NS}}}namedview")
    assert nv is not None
    layer = root.find(f"{{{SVG_NS}}}g")
    assert layer.get(f"{{{INKSCAPE_NS}}}label") == "Layer 1"
    work = root.find(f"{{{SVG_NS}}}metadata/{{{RDF_NS}}}RDF/{{{CC_NS}}}Work")
    assert work is not None


def test_bottom_most_image_is_selected_and_edited_in_place():
    out = swap_background(INKSCAPE_SVG, "../png/scan_height.png", 5.0)
    root = ET.fromstring(out)
    images = list(root.iter(f"{{{SVG_NS}}}image"))
    assert len(images) == 2                      # nothing added, nothing removed
    icon, bg = images
    # the small icon (both dims <= 100) is untouched
    assert icon.get("id") == "icon"
    assert icon.get("width") == "20"
    assert icon.get(f"{{{XLINK_NS}}}href") == "icon.png"
    # the >100x100 background image is the one that got swapped
    assert bg.get("id") == "topo_bgd"
    assert bg.get(f"{{{XLINK_NS}}}href") == "../png/scan_height.png"
    assert bg.get("href") == "../png/scan_height.png"      # SVG2 href kept in sync
    # position and transform on the EXISTING element are untouched
    assert bg.get("x") == "10" and bg.get("y") == "20"
    assert bg.get("transform") == "rotate(3)"


def test_size_filter_skips_small_icon_only_image():
    """When the ONLY <image> present is small (<=100 in either dimension), it
    must not be treated as the background -- a new one is inserted instead."""
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" '
           'xmlns:xlink="http://www.w3.org/1999/xlink" width="200" height="200">'
           '<image id="icon" x="5" y="5" width="40" height="40" '
           'xlink:href="icon.png"/></svg>')
    out = swap_background(svg, "../png/x_height.png", 3.0)
    root = ET.fromstring(out)
    images = list(root.iter(f"{{{SVG_NS}}}image"))
    assert len(images) == 2                      # icon kept, new bg inserted
    assert images[0].get("id") == "topo_bgd"      # inserted as FIRST child
    assert images[0].get(f"{{{XLINK_NS}}}href") == "../png/x_height.png"
    assert images[1].get("id") == "icon"          # icon still there, unmodified
    assert images[1].get("width") == "40"


def test_insert_when_no_image_present():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200"><rect/></svg>'
    out = swap_background(svg, "../png/x_height.png", 12.5)
    root = ET.fromstring(out)
    children = list(root)
    assert children[0].tag == f"{{{SVG_NS}}}image"     # first child of the root
    img = children[0]
    assert img.get("id") == "topo_bgd"
    assert img.get("x") == "0" and img.get("y") == "0"
    assert img.get("width") == "12500" and img.get("height") == "12500"
    assert img.get(f"{{{XLINK_NS}}}href") == "../png/x_height.png"


def test_link_not_embed_and_id_set():
    out = swap_background(INKSCAPE_SVG, "../png/scan_deflection.png", 7.0)
    assert "base64" not in out
    assert "data:image" not in out
    root = ET.fromstring(out)
    bg = next(el for el in root.iter(f"{{{SVG_NS}}}image") if el.get("id") == "topo_bgd")
    assert bg.get("width") == "7000" and bg.get("height") == "7000"
    assert bg.get(f"{{{XLINK_NS}}}href") == "../png/scan_deflection.png"


# --------------------------------------------------------------- the route --

SIMPLE_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">'
              '<rect width="10" height="10"/></svg>')


def _seed_runtime(tmp_path, stem="20260917_120000", channels=("height", "deflection"),
                   size_um=5.0, write_sidecar=True):
    """A hand-built fixture: PNG files + sidecar under a temp runtime dir --
    exactly the Slice A contract, built by hand rather than through the scan
    pump (out of scope here)."""
    png_dir = tmp_path / "scans" / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    for ch in channels:
        (png_dir / f"{stem}_{ch}.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    if write_sidecar:
        json_dir = tmp_path / "scans" / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        (json_dir / f"{stem}.json").write_text(
            json.dumps({"size_um": size_um, "channels": list(channels)}),
            encoding="utf-8")
    return png_dir


def _retarget_store(tmp_path):
    from afm_litho.service.sessions import SessionStore
    return SessionStore(tmp_path / "sessions.jsonl")


def test_background_route_success_writes_svg_with_relative_href(tmp_path):
    from afm_litho.service.hub import hub

    prev_store, prev_src = hub.store, hub.design_source
    hub.store = _retarget_store(tmp_path)
    hub.design_source = {"name": "chip.svg", "text": SIMPLE_SVG}
    _seed_runtime(tmp_path, stem="20260917_120000", size_um=5.0)
    try:
        with _client() as c:
            r = c.post("/api/design/background", json={"channel": "height"})
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["ok"] is True
            assert body["channel"] == "height"
            path = body["path"]
            assert "\\" not in path                      # forward-slashed
            assert path.endswith(
                "scans/svg/20260917_120000_height_chip.svg")
            written = (tmp_path / "scans" / "svg"
                       / "20260917_120000_height_chip.svg").read_text(encoding="utf-8")
            assert 'xlink:href="../png/20260917_120000_height.png"' in written
            assert "topo_bgd" in written
            assert "5000" in written                      # 5 um -> 5000 units
    finally:
        hub.store, hub.design_source = prev_store, prev_src


def test_background_route_400_when_no_design_loaded(tmp_path):
    from afm_litho.service.hub import hub

    prev_store, prev_src = hub.store, hub.design_source
    hub.store = _retarget_store(tmp_path)
    hub.design_source = None
    _seed_runtime(tmp_path)
    try:
        with _client() as c:
            r = c.post("/api/design/background", json={"channel": "height"})
            assert r.status_code == 400, r.text
            assert "design" in r.json()["detail"].lower()
    finally:
        hub.store, hub.design_source = prev_store, prev_src


def test_failed_reupload_clears_stale_design_source(tmp_path, monkeypatch):
    """A design upload that FAILS to parse must not leave the previous
    upload's source in place -- otherwise the background-swap route would
    silently act on a design that is no longer the one shown/loaded (review
    finding on this branch). Upload a good SVG, then an image-only one (no
    drawable geometry -> the route's own error), then the background route
    must 400 as if nothing were loaded at all."""
    from afm_litho.service.hub import hub

    prev_store, prev_src = hub.store, hub.design_source
    hub.store = _retarget_store(tmp_path)
    hub.design_source = None
    # The import is placed against a scan frame; give the test its own.
    monkeypatch.setattr(hub.host, "last_scan_config", {"size_um": 5.0}, raising=False)
    _seed_runtime(tmp_path, stem="20260917_120000", size_um=5.0)
    image_only_svg = ('<svg xmlns="http://www.w3.org/2000/svg" '
                       'xmlns:xlink="http://www.w3.org/1999/xlink" '
                       'width="200" height="200">'
                       '<image x="0" y="0" width="150" height="150" '
                       'xlink:href="bg.png"/></svg>')
    try:
        with _client() as c:
            good = c.post("/api/design/svg?name=good.svg", content=SIMPLE_SVG)
            assert good.status_code == 200, good.text
            assert good.json().get("ok"), good.text

            bad = c.post("/api/design/svg?name=bad.svg", content=image_only_svg)
            assert bad.status_code == 200, bad.text          # this route 200s on {"error":...}
            assert "error" in bad.json(), bad.json()
            assert "no drawable geometry" in bad.json()["error"].lower()

            r = c.post("/api/design/background", json={"channel": "height"})
            assert r.status_code == 400, r.text
            assert "no design loaded" in r.json()["detail"].lower()
    finally:
        hub.store, hub.design_source = prev_store, prev_src


def test_background_route_400_when_channel_png_missing(tmp_path):
    from afm_litho.service.hub import hub

    prev_store, prev_src = hub.store, hub.design_source
    hub.store = _retarget_store(tmp_path)
    hub.design_source = {"name": "chip.svg", "text": SIMPLE_SVG}
    _seed_runtime(tmp_path, channels=("height", "deflection"))
    try:
        with _client() as c:
            r = c.post("/api/design/background", json={"channel": "amplitude"})
            assert r.status_code == 400, r.text
            detail = r.json()["detail"].lower()
            assert "amplitude" in detail
            # lists the channels that DO exist for the latest scan
            assert "height" in detail and "deflection" in detail
    finally:
        hub.store, hub.design_source = prev_store, prev_src


def test_background_route_400_when_sidecar_missing(tmp_path):
    from afm_litho.service.hub import hub

    prev_store, prev_src = hub.store, hub.design_source
    hub.store = _retarget_store(tmp_path)
    hub.design_source = {"name": "chip.svg", "text": SIMPLE_SVG}
    _seed_runtime(tmp_path, stem="20260917_120000", write_sidecar=False)
    try:
        with _client() as c:
            r = c.post("/api/design/background", json={"channel": "height"})
            assert r.status_code == 400, r.text
            assert "sidecar" in r.json()["detail"].lower()
    finally:
        hub.store, hub.design_source = prev_store, prev_src


def test_background_route_default_channel_is_height(tmp_path):
    """No body / no `channel` key defaults to height."""
    from afm_litho.service.hub import hub

    prev_store, prev_src = hub.store, hub.design_source
    hub.store = _retarget_store(tmp_path)
    hub.design_source = {"name": "chip.svg", "text": SIMPLE_SVG}
    _seed_runtime(tmp_path, stem="20260917_120000", size_um=5.0)
    try:
        with _client() as c:
            r = c.post("/api/design/background", json={})
            assert r.status_code == 200, r.text
            assert r.json()["channel"] == "height"
    finally:
        hub.store, hub.design_source = prev_store, prev_src


def test_background_channels_route_lists_newest_stem_channels(tmp_path):
    from afm_litho.service.hub import hub

    prev_store = hub.store
    hub.store = _retarget_store(tmp_path)
    _seed_runtime(tmp_path, stem="20260917_090000", channels=("height",))
    _seed_runtime(tmp_path, stem="20260917_120000", channels=("height", "amplitude"),
                  write_sidecar=False)
    try:
        with _client() as c:
            r = c.get("/api/design/background/channels")
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["stem"] == "20260917_120000"
            assert sorted(body["channels"]) == ["amplitude", "height"]
    finally:
        hub.store = prev_store


def test_background_channels_route_empty_when_no_scans(tmp_path):
    from afm_litho.service.hub import hub

    prev_store = hub.store
    hub.store = _retarget_store(tmp_path)
    try:
        with _client() as c:
            r = c.get("/api/design/background/channels")
            assert r.status_code == 200, r.text
            assert r.json() == {"channels": [], "stem": None}
    finally:
        hub.store = prev_store
