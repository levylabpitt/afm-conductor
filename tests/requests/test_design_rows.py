"""The design edits (slice 2.4): HTTP-only rows, path and query values reach the intake, the
400 on import survives, and each request is journalled under its row name."""

from __future__ import annotations

import json
import pathlib

import pytest

from afm_litho.service import journal
from afm_litho.service.requests import table
from afm_litho.service.requests.table import HttpCall

NAMES = ["designReorder", "designAdd", "designPoints", "designBezierGet", "designBezier",
         "designOp", "designChiral", "designSvgImport", "designBackground", "designSave",
         "designOpen"]
SVG = ('<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1000" '
       'viewBox="0 0 1000 1000"><image id="topo_bgd" x="0" y="0" width="20000" '
       'height="20000"/><rect x="0" y="0" width="1000" height="1000"/></svg>')


class Recorder:
    def __init__(self):
        self.events = []

    def note(self, src, what, /, **fields):
        self.events.append((what, fields))


@pytest.fixture
def events(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(journal, "_running", rec)
    return rec.events


def test_design_rows_are_http_only_edits_with_no_tip_control():
    rows = {r.name: r for r in table.ROWS if r.name in NAMES}
    assert sorted(rows) == sorted(NAMES)
    for row in rows.values():
        assert row.flex is None and row.http.gate == "none" and row.control == "none"


def test_an_intake_receives_path_query_and_body_in_one_call():
    by_name = {r.name: r for r in table.ROWS}
    assert by_name["designBezierGet"].http.intake(HttpCall(path={"index": "3"})) == 3
    got = by_name["designSvgImport"].http.intake(
        HttpCall(body="<svg/>", query={"name": "a.svg", "voltage": "5"}))
    assert got == {"text": "<svg/>", "name": "a.svg", "voltage": 5.0,
                   "speed_um_s": 4.0, "fill_spacing_um": 0.25}


def test_path_param_reaches_the_handler_and_is_journalled(client, events):
    client.post("/api/sketch/clear")
    assert client.get("/api/design/bezier/0").json() == {"error": "no such object"}
    request = [f for w, f in events if w == "request" and f["row"] == "designBezierGet"]
    assert request and request[0]["params"] == 0 and request[0]["actor"] == "operator"
    assert client.get("/api/design/bezier/x").status_code == 422


def test_svg_import_keeps_query_values_and_the_400_status(client, events, monkeypatch):
    # The 400 is "no scan frame and no background image to size the design against", so the
    # test must start with no frame; an earlier test in the worker may have left one.
    from afm_litho.service.hub import hub
    monkeypatch.setattr(hub.host, "last_scan_config", None, raising=False)
    scan = getattr(hub.host.inst, "scan", None)
    if scan is not None:
        monkeypatch.setattr(scan, "config", None, raising=False)
    ok = client.post("/api/design/svg?name=r.svg&voltage=5", content=SVG)
    assert ok.status_code == 200 and ok.json()["ok"] is True
    assert ok.json()["design"]["objects"][0]["voltage"] == 5.0
    bad = client.post("/api/design/svg?name=n.svg",
                      content='<svg xmlns="http://www.w3.org/2000/svg"><rect width="5" '
                              'height="5"/></svg>')
    assert bad.status_code == 400 and "error" in bad.json()
    assert [f["row"] for w, f in events if w == "request"] == ["designSvgImport", "designSvgImport"]


def test_empty_body_on_save_is_an_empty_request_not_a_crash(client):
    assert client.post("/api/design/save").status_code == 200
# --- malformed requests answer as the old routes did ------------------------------------

JSON = {"content-type": "application/json"}
BODY_ROUTES = ["reorder", "add", "points", "bezier", "op", "chiral", "save", "open",
               "background"]
BAD_BODIES = [("array", b"[1]", JSON), ("string", b'"s"', JSON), ("number", b"5", JSON),
              ("broken", b"{", JSON), ("blank", b"   ", JSON),
              ("text_plain", b'{"a": 1}', {"content-type": "text/plain"}),
              ("no_header", b'{"a": 1}', {}), ("empty", b"", JSON)]
CASES = [(f"{r}-{n}", "POST", f"/api/design/{r}", {"content": b, "headers": h})
         for r in BODY_ROUTES for n, b, h in BAD_BODIES]
CASES += [(f"svg-{q}", "POST", f"/api/design/svg?{q}", {"content": "<svg/>"})
          for q in ("voltage=abc", "voltage=", "speed_um_s=x", "fill_spacing_um=1e")]
CASES += [("bezier-x", "GET", "/api/design/bezier/x", {}),
          ("bezier-1.5", "GET", "/api/design/bezier/1.5", {})]


def _detail(reply):
    if reply.status_code != 422:
        return None
    return [(d["type"], tuple(d["loc"]), d["msg"]) for d in reply.json()["detail"]]


@pytest.fixture(scope="module")
def client_():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from afm_litho.service.requests import http_door
    app = FastAPI()
    http_door.install(app)
    return TestClient(app, raise_server_exceptions=False)


EXPECTED = json.loads(
    (pathlib.Path(__file__).parent / "malformed_design.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("label,method,url,kw", CASES, ids=[c[0] for c in CASES])
def test_a_malformed_request_is_answered_as_the_old_route_answered_it(
        client_, events, label, method, url, kw):
    """EXPECTED holds what the old route answered (recorded from it before it was retired)."""
    want = EXPECTED[label]
    got = client_.request(method, url, **kw)
    assert got.status_code == want["status"]
    assert _detail(got) == (None if want["detail"] is None
                            else [(t, tuple(loc), m) for t, loc, m in want["detail"]])
    if want["status"] != 422:
        assert got.text == want["text"]
    else:
        assert [w for w, _ in events][-2:] == ["request", "error"]
