"""`dispatch` and the two adapters (slice 2.1): journalling, actor labels, fall-through, and the
rule that a stop row runs before anything is noted. The table is empty in the product; these
tests put a test-only row in it."""

from __future__ import annotations

import ast
import asyncio
import json
import threading
from pathlib import Path

import pytest

from afm_litho.service import flex_dispatch as fd
from afm_litho.service import flex_provider, journal
from afm_litho.service.commands.errors import CommandError
from afm_litho.service.flex_dispatch import RpcError
from afm_litho.service.hub import hub
from afm_litho.service.requests import flex_door, http_door, table
from afm_litho.service.requests.dispatch import dispatch, dispatch_async
from afm_litho.service.requests.table import Flex, Http, Row


class Recorder:
    """Stands in for the running journal and keeps the entries in the order they were noted."""

    def __init__(self, events: list):
        self.events = events

    def note(self, src, what, /, **fields):
        self.events.append((what, fields))


@pytest.fixture
def events(monkeypatch):
    out: list = []
    monkeypatch.setattr(journal, "_running", Recorder(out))
    return out


@pytest.fixture
def rows(monkeypatch):
    monkeypatch.setattr(table, "ROWS", [])
    return table.ROWS


def _row(handler, control="none", **kw):
    return Row(name="probe", kind="command", control=control, handler=handler, **kw)


READS = ["ACK", "IDN", "getCapabilities", "getInstrument", "getLastPark", "getEstimate",
         "getMode", "HELP", "getState", "getStatus", "getTelemetry", "getPattern",
         "getWritten", "getScanResult", "getWriteResult",
         "getControl", "stateSnapshot", "lastScan", "drift", "connectPorts",
         "scanContinuous", "design", "designCapabilities", "designSvg",
         "designBackgroundChannels", "designFiles"]
COMMANDS = ["acquireControl", "releaseControl", "heartbeat", "takeControl", "loadPattern",
            "setObjects"]
STOPS = ["abort", "safePark", "withdraw", "abortWrite", "abortScan"]
EDITS = ["designReorder", "designAdd", "designPoints", "designBezier", "designOp",
         "designChiral", "designSvgImport", "designBackground", "designSave", "designOpen"]
OTHER_EDITS = ["instrumentOverride", "instrumentReload", "instrumentOpenConfig", "faultClear",
               "keeperContact", "connectPortsAdd", "connectPortsUpdate", "connectPortsDelete",
               "connectPortsReplace", "driftReference", "driftEnable", "scanRelative",
               "lithoToolpath", "tipShow"]
PAGE_AND_DESIGN_EDITS = ["sketch", "sketchUndo", "sketchClear", "shapeUpdate", "simSpeed"]
PAGE_AND_DESIGN_READS = ["pens", "shapes", "shapeToolpath", "indexPage", "editCockpitPage",
                         "optics"]
PANEL_EDITS = ["transitSet", "lithoFrameSet", "scanParams", "forceCurve", "tuneThermal",
               "tuneSweep", "sampleLoad", "sampleUpload", "contactsSet"]
PANEL_READS = ["transitGet", "lithoFrameGet", "lithoDemo", "lithoWritten", "samples",
               "contactsGet", "sessions", "sessionRecall"]
KINDS = {**dict.fromkeys(READS, "read"), **dict.fromkeys(COMMANDS, "command"),
         **dict.fromkeys(STOPS, "command"),
         **dict.fromkeys(EDITS + OTHER_EDITS + PAGE_AND_DESIGN_EDITS + PANEL_EDITS, "edit"),
         **dict.fromkeys(PAGE_AND_DESIGN_READS + PANEL_READS, "read"), "designBezierGet": "read",
         "setMode": "command", "setZGain": "command", "startScan": "command",
         "setContinuous": "command", "startApproach": "command", "startWrite": "command"}


def test_the_product_table_holds_the_reads_control_design_and_stop_rows():
    assert {r.name: r.kind for r in table.ROWS} == KINDS
    tips = {"setMode", "setZGain", "startScan", "setContinuous", "startApproach",
            "startWrite"}
    assert {r.name for r in table.ROWS if r.control == "stop"} == set(STOPS)
    assert {r.name for r in table.ROWS if r.control not in ("none", "stop")} == tips
    assert all(r.control == "none" for r in table.ROWS
               if r.name not in STOPS and r.name not in tips)


def test_a_request_and_its_reply_are_noted_with_actor_and_trace(events):
    out = dispatch(_row(lambda req: {"echo": req}), "operator", {"a": 1})
    assert out == {"echo": {"a": 1}}
    (what1, f1), (what2, f2) = events
    assert (what1, what2) == ("request", "reply")
    assert f1["actor"] == f2["actor"] == "operator"
    assert f1["trace"] == f2["trace"]
    assert f1["row"] == "probe" and f1["params"] == {"a": 1}
    assert f2["result"] == {"echo": {"a": 1}}


def test_each_request_gets_its_own_trace(events):
    row = _row(lambda req: None)
    dispatch(row, "a", None)
    dispatch(row, "a", None)
    assert events[0][1]["trace"] != events[2][1]["trace"]


def test_an_error_is_noted_and_raised_unchanged(events):
    boom = RpcError(-32000, "boom")

    def handler(req):
        raise boom

    with pytest.raises(RpcError) as caught:
        dispatch(_row(handler), "operator", {})
    assert caught.value is boom
    assert [w for w, _ in events] == ["request", "error"]
    assert events[1][1]["code"] == -32000 and "boom" in events[1][1]["error"]


def test_a_stop_row_runs_its_handler_before_anything_is_noted(events):
    seen_at_start: list = []

    def handler(req):
        seen_at_start.append(len(events))
        return "stopped"

    dispatch(_row(handler, control="stop"), "operator", {})
    assert seen_at_start == [0]
    assert [w for w, _ in events] == ["request", "reply"]


def test_a_stop_row_that_fails_still_notes_request_then_error(events):
    def handler(req):
        raise RuntimeError("no")

    with pytest.raises(RuntimeError):
        dispatch(_row(handler, control="stop"), "operator", {})
    assert [w for w, _ in events] == ["request", "error"]


def test_an_ordinary_row_is_noted_before_its_handler_runs(events):
    seen_at_start: list = []
    dispatch(_row(lambda req: seen_at_start.append(len(events))), "operator", {})
    assert seen_at_start == [1]


def test_a_stop_row_does_not_wait_for_a_stalled_journal(tmp_path):
    release = threading.Event()
    j = journal.Journal(tmp_path, sink=lambda line: release.wait(10))
    j.start()
    try:
        done = []
        t = threading.Thread(target=lambda: done.append(
            dispatch(_row(lambda req: "ok", control="stop"), "operator", {})))
        t.start()
        t.join(2.0)
        assert done == ["ok"]
    finally:
        release.set()
        j.stop()


def test_async_stop_row_awaits_only_its_own_handler(events):
    order: list = []

    async def handler(req):
        order.append(("handler", len(events)))
        return "ok"

    asyncio.run(dispatch_async(_row(handler, control="stop"), "operator", {}))
    assert order == [("handler", 0)]
    assert [w for w, _ in events] == ["request", "reply"]


def test_async_dispatch_runs_a_plain_function_too(events):
    assert asyncio.run(dispatch_async(_row(lambda req: 3), "operator", {})) == 3


# --- the FLEX door ---------------------------------------------------------------------


def _call(method, params=None, role="command"):
    raw = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    return json.loads(flex_provider.handle_message(hub, raw, role=role))


def test_an_unknown_verb_still_answers_method_not_found(events, rows):
    reply = _call("noSuchVerb")
    assert reply["error"]["code"] == -32601
    assert [w for w, _ in events] == ["request", "error"]


def test_a_request_with_the_current_token_is_labelled_with_the_script_name(events, rows):
    held = hub.acquire_control("lab-script")
    try:
        _call("getState", {"token": held.token})
    finally:
        hub.release_control(held.token)
    assert events[0][1]["actor"] == "lab-script"


def test_a_stale_token_gives_the_unnamed_label_and_refuses_nothing(events):
    reply = _call("getState", {"token": "not-a-token"})
    assert "result" in reply
    assert events[0][1]["actor"] == flex_door.UNNAMED_ACTOR


def test_stop_verbs_are_rows_noted_after_the_handler(events, monkeypatch):
    ran: list = []
    monkeypatch.setattr(fd, "_h_abort_write",
                        lambda *a: ran.append(len(events)) or {"ok": True})
    _call("abortWrite", {})
    assert ran == [0]
    assert events[0][0] == "request" and events[0][1]["row"] == "abortWrite"


def test_a_row_is_found_by_its_verb_and_its_intake_is_applied(events, rows):
    rows.append(_row(lambda req: {"got": req}, flex=Flex(
        verb="probeVerb", socket="read", bound_s=1.0, intake=lambda p: ["in", p], gate="none")))
    reply = _call("probeVerb", {"x": 1}, role="read")
    assert reply["result"] == {"got": ["in", {"x": 1}]}
    assert events[0][1]["row"] == "probe"


def test_a_command_socket_row_is_refused_on_the_read_socket(events, rows):
    rows.append(_row(lambda req: "no", flex=Flex(
        verb="probeCmd", socket="command", bound_s=1.0, intake=lambda p: p, gate="token")))
    reply = _call("probeCmd", {}, role="read")
    assert reply["error"]["code"] == -32601
    assert "COMMAND socket only" in reply["error"]["message"]
    assert [w for w, _ in events] == ["request", "error"]


def test_a_command_error_from_a_row_becomes_its_rpc_error(events, rows):
    def handler(req):
        raise CommandError(-32602, "bad thing")

    rows.append(_row(handler, flex=Flex(
        verb="probeBad", socket="read", bound_s=1.0, intake=lambda p: p, gate="none")))
    reply = _call("probeBad", {}, role="read")
    assert reply["error"]["code"] == -32602 and "bad thing" in reply["error"]["message"]
    assert [w for w, _ in events] == ["request", "error"]


# --- the HTTP door ---------------------------------------------------------------------


def test_an_empty_table_adds_no_routes(rows):
    from fastapi import FastAPI
    app = FastAPI()
    before = len(app.routes)
    http_door.install(app)
    assert len(app.routes) == before


def test_a_row_with_an_http_binding_gets_a_route_labelled_operator(events, rows):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    rows.append(_row(lambda req: {"seen": req}, http=Http(
        method="POST", path="/api/probe", intake=lambda call: call.body, gate="none")))
    app = FastAPI()
    http_door.install(app)
    reply = TestClient(app).post("/api/probe", json={"k": 2})
    assert reply.json() == {"seen": {"k": 2}}
    assert events[0][1]["actor"] == "operator"
    assert [w for w, _ in events] == ["request", "reply"]


# --- no carrier branches ---------------------------------------------------------------

CARRIER_NAMES = {"flex", "http", "https", "zmq", "rest", "browser", "ws", "websocket"}
PACKAGE = Path(__file__).resolve().parents[2] / "src" / "afm_litho" / "service" / "requests"


def _mentions_a_door(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in {"door", "carrier"}:
            return True
        if isinstance(sub, ast.Attribute) and sub.attr in {"door", "carrier"}:
            return True
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str) \
                and sub.value.lower() in CARRIER_NAMES:
            return True
    return False


@pytest.mark.parametrize("path", sorted(PACKAGE.glob("*.py")), ids=lambda p: p.name)
def test_nothing_in_requests_branches_on_the_carrier(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = [n.lineno for n in ast.walk(tree)
                 if isinstance(n, ast.Compare | ast.Match) and _mentions_a_door(n)]
    assert offenders == [], f"{path.name} compares on a door or carrier at lines {offenders}"


def test_the_carrier_check_catches_a_door_comparison():
    tree = ast.parse("if door == 'flex':\n    pass\nif x == 'http':\n    pass")
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.Compare) and _mentions_a_door(n)]
    assert len(hits) == 2


def test_a_journal_that_raises_after_a_stop_cannot_fail_the_stop(monkeypatch):
    class Broken:
        def note(self, *a, **k):
            raise OSError("disk gone")

    monkeypatch.setattr(journal, "_running", Broken())
    assert dispatch(_row(lambda req: "stopped", control="stop"), "operator", {}) == "stopped"
    out = asyncio.run(dispatch_async(_row(lambda req: "stopped", control="stop"), "operator", {}))
    assert out == "stopped"


def test_a_stop_verb_looks_up_its_actor_only_after_the_handler(monkeypatch, events):
    order = []
    monkeypatch.setattr(fd, "_h_abort", lambda *a, **k: order.append("handler") or "ok")
    monkeypatch.setattr(flex_door, "actor_for",
                        lambda *a: order.append("actor") or "operator")
    flex_door.handle(hub, "abort", {}, role="command", ctx=None)
    assert order[0] == "handler"
