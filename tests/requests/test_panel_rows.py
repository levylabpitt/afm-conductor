"""Slice 7b: the litho, scan, sample and session routes are rows. Each keeps its path,
method, body and reply; a bad query value is the 422 FastAPI gave."""

from __future__ import annotations

from afm_litho.service.requests import table

NAMES = {
    "transitGet": ("GET", "/api/transit"), "transitSet": ("POST", "/api/transit"),
    "lithoFrameGet": ("GET", "/api/litho/frame"), "lithoFrameSet": ("POST", "/api/litho/frame"),
    "lithoDemo": ("GET", "/api/litho/demo"), "lithoWritten": ("GET", "/api/litho/written"),
    "scanParams": ("POST", "/api/scan/params"), "forceCurve": ("POST", "/api/force_curve"),
    "tuneThermal": ("POST", "/api/tune/thermal"), "tuneSweep": ("POST", "/api/tune/sweep"),
    "samples": ("GET", "/api/samples"), "sampleLoad": ("POST", "/api/sample/load"),
    "sampleUpload": ("POST", "/api/sample/upload"), "contactsGet": ("GET", "/api/contacts"),
    "contactsSet": ("POST", "/api/contacts"), "sessions": ("GET", "/api/sessions"),
    "sessionRecall": ("GET", "/api/sessions/recall"),
}


def test_each_route_is_an_http_only_row_with_no_tip_control():
    rows = {r.name: r for r in table.ROWS if r.name in NAMES}
    assert sorted(rows) == sorted(NAMES)
    for name, row in rows.items():
        assert (row.http.method, row.http.path) == NAMES[name]
        assert row.flex is None and row.control == "none" and row.http.gate == "none"
        assert row.journalled is False


def test_written_map_takes_its_query_values_and_refuses_bad_ones(client):
    assert client.get("/api/litho/written?n=8").json()["n"] == 8
    bad = client.get("/api/litho/written?n=x")
    assert bad.status_code == 422 and bad.json()["detail"][0]["loc"] == ["query", "n"]
    assert client.get("/api/litho/written?size_um=x").status_code == 422


def test_recall_needs_its_id(client):
    reply = client.get("/api/sessions/recall")
    assert reply.status_code == 422 and reply.json()["detail"][0]["type"] == "missing"


def test_a_body_is_required_where_the_route_took_one(client):
    for path in ("/api/transit", "/api/litho/frame", "/api/sample/load", "/api/contacts"):
        reply = client.post(path)
        assert reply.status_code == 422, path
        assert reply.json()["detail"][0]["loc"] == ["body"]


def test_upload_reaches_the_handler_as_raw_bytes(client, monkeypatch):
    from afm_litho.service.routers import sample
    seen = {}

    async def fake(data, name="scan.ibw"):
        seen.update(data=data, name=name)
        return {"ok": True}

    monkeypatch.setattr(sample, "sample_upload", fake)
    payload = bytes(range(256))
    assert client.post("/api/sample/upload?name=a.ibw", content=payload).json() == {"ok": True}
    assert seen == {"data": payload, "name": "a.ibw"}
    client.post("/api/sample/upload", content=b"x")
    assert seen["name"] == "scan.ibw"


def test_a_whole_float_reads_as_an_int_and_a_fractional_one_is_refused(client):
    assert client.get("/api/litho/written?n=8.0").json()["n"] == 8
    for bad in ("8.5", "nan", "inf"):
        reply = client.get(f"/api/litho/written?n={bad}")
        assert reply.status_code == 422, bad
        assert reply.json()["detail"][0]["loc"] == ["query", "n"]


def test_upload_refuses_an_empty_body_and_a_wrong_extension(client):
    assert client.post("/api/sample/upload?name=a.ibw").json() == {"error": "empty file"}
    wrong = client.post("/api/sample/upload?name=a.txt", content=b"x")
    assert wrong.json() == {"error": "expected an Igor binary wave (.ibw)"}
