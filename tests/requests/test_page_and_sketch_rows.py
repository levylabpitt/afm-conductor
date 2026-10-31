"""Slice 7a: the two pages, the simulation speed, the optics basemap, the pens, the sketch
and the shapes are rows. Each keeps its path, body and reply."""

from __future__ import annotations

import pytest

from afm_litho.service import journal
from afm_litho.service.requests import table

NAMES = {
    "indexPage": ("GET", "/"), "editCockpitPage": ("GET", "/edit-cockpit"),
    "optics": ("GET", "/api/optics"), "simSpeed": ("POST", "/api/speed/{factor}"),
    "pens": ("GET", "/api/pens"), "sketch": ("POST", "/api/sketch"),
    "sketchUndo": ("POST", "/api/sketch/undo"), "sketchClear": ("POST", "/api/sketch/clear"),
    "shapes": ("GET", "/api/shapes"), "shapeUpdate": ("POST", "/api/shapes/{index}"),
    "shapeToolpath": ("GET", "/api/shapes/{index}/toolpath"),
}
STROKE = {"points_um": [[0, 0], [1, 0], [2, 1], [3, 1]], "pen": "write", "mode": "literal"}


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


def test_each_route_is_an_http_only_row_with_no_tip_control():
    rows = {r.name: r for r in table.ROWS if r.name in NAMES}
    assert sorted(rows) == sorted(NAMES)
    for name, row in rows.items():
        assert (row.http.method, row.http.path) == NAMES[name]
        assert row.flex is None and row.control == "none" and row.http.gate == "none"


def test_the_pages_are_html_and_nothing_is_journalled(client, events):
    root = client.get("/", follow_redirects=False)
    assert root.status_code == 307 and root.headers["location"] == "/edit-cockpit"
    assert not [f for w, f in events if f["row"] == "indexPage"]
    reply = client.get("/edit-cockpit")
    assert reply.status_code == 200 and reply.headers["content-type"].startswith("text/html")
    assert "?v=" in reply.text
    assert not [f for w, f in events if f["row"] == "editCockpitPage"]


def test_speed_is_clamped_and_a_bad_factor_is_a_422(client):
    try:
        assert client.post("/api/speed/1000").json() == {"speed": 64.0}
        assert client.post("/api/speed/0").json() == {"speed": 0.25}
        assert client.post("/api/speed/x").status_code == 422
    finally:
        client.post("/api/speed/8")


def test_optics_is_the_basemap(client):
    j = client.get("/api/optics").json()
    assert j["n"] == 160 and len(j["gray"]) == 160 * 160


def test_a_sketch_stroke_is_added_and_undone(client, events):
    client.post("/api/sketch/clear")
    j = client.post("/api/sketch", json=STROKE).json()
    assert j["ok"] and j["added"] >= 1
    assert client.post("/api/sketch", json={"points_um": [[0, 0]]}).json() == {
        "error": "stroke too short"}
    assert client.post("/api/sketch").status_code == 422
    assert client.post("/api/sketch/undo").status_code == 200
    assert not [f for w, f in events if f.get("row") in NAMES]


def test_shape_routes_answer_and_report_path_and_body_faults_together(client):
    client.post("/api/sketch/clear")
    assert client.get("/api/shapes").json() == {"shapes": []}
    assert client.get("/api/shapes/0/toolpath").json() == {"error": "no such object"}
    assert client.post("/api/shapes/0", json={}).json() == {"error": "no such object"}
    reply = client.post("/api/shapes/x", content=b"[]",
                        headers={"content-type": "application/json"})
    assert reply.status_code == 422
    assert {tuple(e["loc"]) for e in reply.json()["detail"]} == {("path", "index"), ("body",)}


def test_the_pens_come_with_their_modes(client):
    j = client.get("/api/pens").json()
    assert {m["key"] for m in j["modes"]} == {"literal", "clean", "ideal"} and j["pens"]


def test_an_ordinary_row_is_still_journalled(client, events):
    client.get("/api/design")
    assert [w for w, f in events if f["row"] == "design"] == ["request", "reply"]
