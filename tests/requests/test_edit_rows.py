"""The other edits (slice 2.5): rows with today's gates, malformed requests answered as the old
routes answered them, loadPattern keeps its carrier difference and its 2.0 s bound."""

from __future__ import annotations

import json
import pathlib

import pytest

from afm_litho.service import flex_dispatch as fd
from afm_litho.service import journal
from afm_litho.service.control import TIP_MUTATING
from afm_litho.service.requests import table
from afm_litho.service.requests.edits import PatternSource
from afm_litho.service.requests.table import HttpCall

HTTP_NAMES = {
    "instrumentOverride": "/api/instrument/override",
    "instrumentReload": "/api/instrument/reload",
    "instrumentOpenConfig": "/api/instrument/open_config",
    "faultClear": "/api/fault/clear",
    "keeperContact": "/api/keeper/contact",
    "connectPortsAdd": "/api/connect/ports/add",
    "connectPortsUpdate": "/api/connect/ports/update",
    "connectPortsDelete": "/api/connect/ports/delete",
    "connectPortsReplace": "/api/connect/ports/replace",
    "driftReference": "/api/drift/reference",
    "driftEnable": "/api/drift/enable",
    "scanRelative": "/api/scan/relative",
    "lithoToolpath": "/api/litho/toolpath",
    "tipShow": "/api/tip/show",
    "loadPattern": "/api/litho/load",
}


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


def test_every_edit_row_has_today_s_gate_none_and_none_is_a_tip_route():
    by_name = {r.name: r for r in table.ROWS}
    for name, path in HTTP_NAMES.items():
        row = by_name[name]
        assert row.http.method == "POST" and row.http.path == path
        assert row.http.gate == "none" and row.control == "none"
        assert path not in TIP_MUTATING
    for verb in ("loadPattern", "setObjects"):
        flex = by_name[verb].flex
        assert flex.socket == "command" and flex.gate == "none"
        assert flex.bound_s == fd.REGISTRY[verb].bound_s
    assert by_name["loadPattern"].flex.bound_s == 2.0
    assert by_name["setObjects"].http is None


def test_the_two_carriers_of_loadPattern_differ_only_in_their_intake():
    row = {r.name: r for r in table.ROWS}["loadPattern"]
    assert row.http.intake(HttpCall(body=None)) == PatternSource("auto", False)
    assert row.http.intake(HttpCall(body={"source": "../x.svg"})) == \
        PatternSource("../x.svg", False)
    params = {"token": "t", "source": "D:/designs/a.oas"}
    assert row.flex.intake(params) is params       # the handler opens paths (allow_paths=True)


def test_fault_clear_without_confirm_keeps_the_fault_and_is_journalled(client, events):
    from afm_litho.service.hub import hub
    hub.fault = {"reason": "x"}
    try:
        assert client.post("/api/fault/clear").json() == {
            "ok": True, "cleared": False, "fault": {"reason": "x"}}
        assert hub.fault == {"reason": "x"}
        reply = client.post("/api/fault/clear", json={"confirm": True}).json()
        # the twin has no Igor bridge, so no CLEARFAULT is sent and the reply says so
        assert reply["ok"] is True and reply["cleared"] is True
        assert reply["igor_cleared"] is None
        assert hub.fault is None
    finally:
        hub.fault = None
    rows = [f["row"] for w, f in events if w == "request"]
    assert rows == ["faultClear", "faultClear"]


# --- malformed requests answer as the old routes did ------------------------------------

JSON = {"content-type": "application/json"}
BAD_BODIES = [("array", b"[1]", JSON), ("string", b'"s"', JSON), ("number", b"5", JSON),
              ("broken", b"{", JSON), ("blank", b"   ", JSON),
              ("text_plain", b'{"a": 1}', {"content-type": "text/plain"}),
              ("no_header", b'{"a": 1}', {}), ("empty", b"", JSON)]
BODY_ROUTES = ["/api/instrument/override", "/api/fault/clear", "/api/connect/ports/add",
               "/api/connect/ports/update", "/api/connect/ports/delete",
               "/api/connect/ports/replace", "/api/drift/reference", "/api/drift/enable",
               "/api/scan/relative", "/api/tip/show", "/api/litho/load"]
CASES = [(f"{u.rsplit('/', 1)[-1]}-{n}", "POST", u, {"content": b, "headers": h})
         for u in BODY_ROUTES for n, b, h in BAD_BODIES]
# These routes declare no body: whatever is sent is ignored, a broken body included.
CASES += [(f"{u.rsplit('/', 1)[-1]}-{n}", "POST", u, {"content": b, "headers": h})
          for u in ("/api/instrument/reload", "/api/litho/toolpath")
          for n, b, h in BAD_BODIES]
# Bodies that are objects but wrong inside; the handler answers them, never a 422.
CASES += [("override-nofield", "POST", "/api/instrument/override", {"json": {}}),
          ("ports-add-nopos", "POST", "/api/connect/ports/add", {"json": {}}),
          ("ports-update-noid", "POST", "/api/connect/ports/update", {"json": {}}),
          ("ports-delete-noid", "POST", "/api/connect/ports/delete", {"json": {}}),
          ("ports-replace-noports", "POST", "/api/connect/ports/replace", {"json": {}}),
          ("relative-bad-settle", "POST", "/api/scan/relative", {"json": {"settle_s": "x"}})]


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
    (pathlib.Path(__file__).parent / "malformed_edits.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("label,method,url,kw", CASES, ids=[c[0] for c in CASES])
def test_a_malformed_request_is_answered_as_the_old_route_answered_it(
        client_, events, label, method, url, kw):
    """EXPECTED holds what the old route answered (recorded from it before it was retired)."""
    want = EXPECTED[label]
    if label.startswith("toolpath-"):          # the recorded answer is the auto design's path
        assert client_.post("/api/litho/load", json={}).status_code == 200
    got = client_.request(method, url, **kw)
    assert got.status_code == want["status"]
    assert _detail(got) == (None if want["detail"] is None
                            else [(t, tuple(loc), m) for t, loc, m in want["detail"]])
    if label == "relative-empty":        # the live continuous-scan state, which other tests move
        assert json.loads(got.text).keys() == json.loads(want["text"]).keys()
    elif want["status"] != 422:
        assert got.text == want["text"]
    else:
        assert [w for w, _ in events][-2:] == ["request", "error"]


class ThreadRecorder(Recorder):
    """Notes which thread each journal entry was written on (the event loop's)."""

    def note(self, src, what, /, **fields):
        import threading
        super().note(src, what, **fields)
        if what == "request":
            self.loop_thread = threading.get_ident()


# (path, module, function the row runs in a worker thread, body)
PLAIN_DEF_ROWS = [("/api/instrument/override", "host.instrument_cmds", "set_override", {}),
                  ("/api/instrument/reload", "routers.instrument", "post_instrument_reload",
                   None),
                  ("/api/instrument/open_config", "routers.instrument",
                   "post_instrument_open_config", None),
                  ("/api/fault/clear", "host.instrument_cmds", "clear_fault", {})]


@pytest.mark.parametrize("path,module,function,body", PLAIN_DEF_ROWS,
                         ids=[p[2] for p in PLAIN_DEF_ROWS])
def test_a_plain_def_route_still_runs_off_the_event_loop_thread(
        client, monkeypatch, path, module, function, body):
    import importlib
    import threading

    target = importlib.import_module(f"afm_litho.service.{module}")
    rec = ThreadRecorder()
    monkeypatch.setattr(journal, "_running", rec)
    ran_on = []

    def probe(*_args):
        ran_on.append(threading.get_ident())
        return {"ok": True}

    monkeypatch.setattr(target, function, probe)
    assert client.post(path, **({} if body is None else {"json": body})).status_code == 200
    assert ran_on and ran_on[0] != rec.loop_thread


@pytest.mark.parametrize("verb", ["loadPattern", "setObjects"])
def test_flex_dead_man_is_refreshed_once_per_command_call_with_a_string_token(
        monkeypatch, verb):
    from afm_litho.service.hub import hub
    from afm_litho.service.requests import flex_door
    touched = []
    monkeypatch.setattr(hub, "touch_control", lambda token: touched.append(token))

    def call(params, role="command"):
        touched.clear()
        with pytest.raises(fd.RpcError):
            flex_door.handle(hub, verb, params, role=role, ctx=None)
        return list(touched)

    assert call({"token": "t1"}) == ["t1"]
    assert call({}) == []
    assert call({"token": 5}) == []
    assert call({"token": "t1"}, role="read") == []


@pytest.mark.parametrize("verb", ["loadPattern", "setObjects", "heartbeat"])
def test_a_command_verb_asked_of_the_read_socket_is_refused_and_journalled(events, verb):
    from afm_litho.service.hub import hub
    from afm_litho.service.requests import flex_door
    with pytest.raises(fd.RpcError) as got:
        flex_door.handle(hub, verb, {"token": "t"}, role="read", ctx=None)
    assert got.value.code == fd.METHOD_NOT_FOUND
    assert got.value.message == f"{verb} is served by the COMMAND socket only"
    assert [(w, f["row"]) for w, f in events] == [("request", verb), ("error", verb)]


def test_loadPattern_over_flex_is_refused_without_a_token_and_journalled(events):
    from afm_litho.service.hub import hub
    from afm_litho.service.requests import flex_door
    for params in (None, {}, {"source": 5, "token": "x"}):
        events.clear()
        with pytest.raises(fd.RpcError) as got:
            flex_door.handle(hub, "loadPattern", params, role="command", ctx=None)
        assert got.value.code == -32030
        assert got.value.message == ("control_required: this call needs the commander token; "
                                     "call acquireControl first")
        assert [(w, f["row"]) for w, f in events] == [("request", "loadPattern"),
                                                      ("error", "loadPattern")]
