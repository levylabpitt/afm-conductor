"""HTTP-layer integration tests through a FastAPI TestClient.

The suite has deep unit tests of the twin, fills and contacts, but almost nothing
goes THROUGH the endpoints -- app.py sat at 43% coverage. These drive the
service the way the panel does.
"""

from __future__ import annotations

import time

import pytest
from _im_fixtures_guard import GUARD_WRITE_KW, seed_free_air
from hub_fields import owner

# `client` comes from tests/conftest.py (a TestClient).


# --------------------------------------------------------------- read smoke --

@pytest.mark.parametrize("path", [
    "/api/state", "/api/samples", "/api/pens", "/api/optics",
    "/api/litho/frame", "/api/transit", "/api/shapes", "/api/sessions",
    "/api/contacts", "/api/design/capabilities",
])
def test_read_endpoints_respond(client, path):
    r = client.get(path)
    assert r.status_code == 200, (path, r.text[:200])
    assert isinstance(r.json(), (dict, list))


# ---------------------------------------------------------------- design/svg --

def _svg(rect):
    # Embed the `topo_bgd` background image the cockpit writes, so a design-origin
    # import can resolve the scan size it was drawn over (design-origin rule).
    x, y, w, h = rect
    span = max(x + w, y + h)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<image id="topo_bgd" x="0" y="0" width="{span}" height="{span}"/>'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="black"/></svg>')


def _post_svg(client, svg, name="t.svg"):
    return client.post(f"/api/design/svg?name={name}",
                       content=svg, headers={"content-type": "image/svg+xml"})


def test_design_svg_imports_a_shape(client):
    r = _post_svg(client, _svg((2000, 2000, 20000, 10000)))   # 20x10 um at 1nm/unit
    assert r.status_code == 200
    j = r.json()
    assert j.get("ok") is True
    assert len(j["design"]["objects"]) >= 1


def test_design_svg_rejects_a_mis_scaled_design(client):
    # 3,000,000 units at the fixed 1 nm/unit is 3 mm -- past MAX_DESIGN_UM. The
    # span guard must refuse it (else it flattens to millions of points and OOMs).
    r = _post_svg(client, _svg((0, 0, 3_000_000, 3_000_000)), name="huge.svg")
    j = r.json()
    assert "error" in j
    assert "scanner" in j["error"].lower() or "not importing" in j["error"].lower()


def test_design_svg_rejects_non_svg(client):
    r = client.post("/api/design/svg", content=b"this is not an svg",
                    headers={"content-type": "text/plain"})
    assert r.json().get("error") == "not an SVG"


# -------------------------------------------------------------- litho/toolpath --

def test_litho_toolpath_compiles_the_loaded_design(client):
    # load a fresh design, then ask for its compiled toolpath
    assert _post_svg(client, _svg((0, 0, 20000, 10000))).json().get("ok")
    r = client.post("/api/litho/toolpath")
    assert r.status_code == 200
    segs = r.json()["segments"]
    assert len(segs) >= 1
    s = segs[0]
    assert "path_um" in s and "voltage" in s and len(s["path_um"]) >= 2


# --------------------------------------------------------------- sample/load --

@pytest.fixture
def restore_twin():
    """Loading a sample swaps the process hub's world, instrument and design state; put them
    back so a later test in the same worker still meets the default twin."""
    from afm_litho.service.app import hub
    names = ("instrument", "world", "arc", "inst", "surface", "written_objects",
             "written_accum", "sample_name", "pattern", "ports", "ports_seq")
    saved = {n: getattr(owner(hub, n), n) for n in names}
    yield
    for n, v in saved.items():
        setattr(owner(hub, n), n, v)


def test_sample_load_switches_the_twin(client, restore_twin):
    files = client.get("/api/samples").json()["files"]
    assert files, "no sample .ibw files to load"
    r = client.post("/api/sample/load", json={"file": files[0]})
    assert r.status_code == 200
    j = r.json()
    assert j.get("ok") is True
    assert j.get("name") and "instrument" in j


def test_sample_load_reports_a_missing_file(client):
    r = client.post("/api/sample/load", json={"file": "does-not-exist.ibw"})
    assert "not found" in r.json().get("error", "")


# -------------------------------------------------- sessions.py (persistence) --

def test_session_store_round_trips(tmp_path):
    from afm_litho.service.sessions import SessionStore
    s = SessionStore(tmp_path / "sessions.jsonl")
    rec = s.record("scan", {"label": "first scan"})
    assert rec["kind"] == "scan" and rec["id"].startswith("scan-")
    lst = s.list()
    assert len(lst) == 1
    assert lst[0]["id"] == rec["id"] and lst[0]["label"] == "first scan"
    got = s.get(rec["id"])
    assert got is not None and got["kind"] == "scan"
    assert s.get("no-such-id") is None


def test_session_store_lists_newest_first(tmp_path):
    from afm_litho.service.sessions import SessionStore
    s = SessionStore(tmp_path / "s.jsonl")
    for i in range(5):
        s.record("litho", {"label": f"n{i}"})
    lst = s.list(limit=3)
    assert len(lst) == 3
    assert lst[0]["label"] == "n4"          # most recent first


def test_session_store_skips_corrupt_lines(tmp_path):
    from afm_litho.service.sessions import SessionStore
    p = tmp_path / "s.jsonl"
    s = SessionStore(p)
    s.record("scan", {"label": "good"})
    p.write_text(p.read_text() + "{ not json at all\n", encoding="utf-8")
    assert len(s.list()) == 1               # the bad line is skipped, not fatal


def test_llab_adapter_is_unavailable_without_a_database():
    from afm_litho.service.sessions import Llab079Adapter
    a = Llab079Adapter(dsn="host=127.0.0.1 port=1 dbname=nope connect_timeout=1")
    assert a.available() is False           # unreachable/absent driver -> graceful False


def test_llab_adapter_is_off_with_no_dsn_and_never_connects(monkeypatch):
    """Shipped blank: no AFM_LITHO_SESSIONS_DSN means no database attempt."""
    import sys
    import types
    from afm_litho.service.sessions import Llab079Adapter
    monkeypatch.delenv("AFM_LITHO_SESSIONS_DSN", raising=False)

    def _boom(*a, **k):
        raise AssertionError("must not connect without a DSN")
    monkeypatch.setitem(sys.modules, "psycopg2", types.SimpleNamespace(connect=_boom))
    assert Llab079Adapter().available() is False


def test_llab_adapter_reads_its_dsn_from_the_environment(monkeypatch):
    from afm_litho.service.sessions import Llab079Adapter
    monkeypatch.setenv("AFM_LITHO_SESSIONS_DSN", "host=db.example.org dbname=x")
    assert Llab079Adapter().dsn == "host=db.example.org dbname=x"


# ------------------------------------------- streaming: write + scan pumps --
# These drive the two async pump loops end to end -- the write worker thread and
# _litho_pump, and the scan loop _scan_pump -- which were entirely uncovered.

def _wait(cond, timeout=30.0, interval=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return False


def _tmp_store(tmp_path):
    """Redirect the session archive to a temp file so tests do not append to the
    real runtime/sessions.jsonl, and return the live hub for assertions."""
    from afm_litho.service.app import hub
    from afm_litho.service.sessions import SessionStore
    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    return hub


def test_litho_write_runs_to_completion_and_records_a_session(client, tmp_path):
    hub = _tmp_store(tmp_path)
    # one coarse shape so the write is quick; treat it as unwritten
    r = client.post("/api/design/svg?name=w.svg&fill_spacing_um=2.0",
                    content=_svg((0, 0, 8000, 4000)),
                    headers={"content-type": "image/svg+xml"})
    assert r.json().get("ok")
    hub.host.written_objects = set()

    # Command guard (INSTRUMENT_MODEL_SPEC sections 2/10, 13): the REST write door
    # needs `amp_gain` and a fresh withdrawn free-air baseline; the default (no
    # `objects` key) still writes everything unwritten, which is this test's intent.
    seed_free_air(hub)
    j = client.post("/api/litho/write",
                    json={"setpoint_relative": 0.2, **GUARD_WRITE_KW}).json()
    assert j.get("ok") is True and j["writing"] >= 1

    assert _wait(lambda: any(s["kind"] == "litho" for s in hub.store.list())), \
        "the write never recorded a litho session (pump did not reach litho_done)"
    rec_id = next(s["id"] for s in hub.store.list() if s["kind"] == "litho")
    rec = hub.store.get(rec_id)
    assert rec["written"], "litho session recorded but no objects were written"


def test_scan_streams_and_records_a_session(client, tmp_path):
    hub = _tmp_store(tmp_path)
    r = client.post("/api/scan/start", json={
        "size_um": 5.0, "pixels": 16, "lines": 8,
        "line_rate_hz": 200.0, "mode": "contact"})
    assert r.json().get("ok") is True

    assert _wait(lambda: any(s["kind"] == "scan" for s in hub.store.list())), \
        "the scan never recorded a session (pump did not finish)"
    rec_id = next(s["id"] for s in hub.store.list() if s["kind"] == "scan")
    assert "scan" in hub.store.get(rec_id)["label"]


# -------------------------------------------------------------- websocket /ws --

def test_ws_streams_telemetry(client):
    """The /ws heartbeat: accept (auth stepped aside), then telemetry snapshots.
    A stray broadcast from an earlier test can arrive first, so look for the
    telemetry frame rather than assuming it is the first."""
    with client.websocket_connect("/ws") as ws:
        for _ in range(20):
            msg = ws.receive_json()
            if msg["type"] == "telemetry":
                assert "scanning" in msg
                break
        else:
            raise AssertionError("no telemetry frame received over /ws")


def test_ws_receives_a_broadcast(client):
    """Hub.broadcast's send-to-client loop only runs with a client connected.
    Trigger a broadcast (abort logs one) and confirm it reaches the socket."""
    with client.websocket_connect("/ws") as ws:
        client.post("/api/litho/abort")           # broadcasts {"type": "log", ...}
        seen = []
        for _ in range(20):
            m = ws.receive_json()
            seen.append(m["type"])
            if m["type"] == "log":
                break
        assert "log" in seen, f"no log broadcast reached the socket; got {seen}"
