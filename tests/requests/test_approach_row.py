"""The startApproach row (slice 2.8c): FLEX startApproach (token gate, then backend check, then
the busy claim) and POST /api/approach (operator take), answered as the old route answered."""

from __future__ import annotations

import asyncio
import threading

import pytest
from afm_litho.service import access
from afm_litho.service import control as ctl
from afm_litho.service import journal
from afm_litho.service.flex_dispatch import RpcError
from afm_litho.service.hub import hub
from afm_litho.service.requests import flex_door, table

JSON = {"content-type": "application/json"}


class Recorder:
    def __init__(self):
        self.events = []

    def note(self, src, what, /, **fields):
        self.events.append((src, what, fields))


@pytest.fixture
def events(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(journal, "_running", rec)
    return rec.events


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    thread = threading.Thread(target=lp.run_forever, daemon=True)
    started = threading.Event()
    lp.call_soon(started.set)
    thread.start()
    # run_on_loop refuses a loop that is not yet running; wait for the first tick.
    assert started.wait(10), "the test loop never started"
    prev = getattr(hub, "_loop", None)
    hub._loop = lp
    yield lp
    hub._loop = prev
    lp.call_soon_threadsafe(lp.stop)
    thread.join(5)


@pytest.fixture(autouse=True)
def free_instrument():
    """Each test starts with no operator, no run claim and a disengaged tip, and leaves
    them so: the takes-once tests engage for real or hold the claim, and must not depend
    on (or leak into) whatever another file left on the process-wide hub."""
    hub._run_claim, hub.host.engaged, hub.control = None, False, None
    yield
    hub._run_claim, hub.host.engaged, hub.control = None, False, None


def _flex(params):
    return flex_door.handle(hub, "startApproach", params, role="command", ctx=None)


def test_start_approach_is_one_row_with_both_carriers_and_their_gates():
    row = table.flex_row("startApproach")
    assert row.control == "tip" and row.flex.gate == "token"
    assert (row.http.method, row.http.path, row.http.gate) == ("POST", "/api/approach",
                                                               "operator_tip")


def test_flex_start_approach_without_the_token_is_refused(loop, events):
    with pytest.raises(RpcError) as bad:
        _flex({"setpoint": 0.2})
    assert bad.value.code == ctl.CONTROL_REQUIRED
    assert [(s, w) for s, w, _ in events] == [("door", "request"), ("door", "error")]


def test_flex_start_approach_given_non_object_params_answers_as_the_registry_did(loop):
    for params in ([1], "s"):
        with pytest.raises(RpcError) as bad:
            _flex(params)
        assert (bad.value.code, bad.value.data) == (-32602, {"method": "startApproach"})


def test_flex_start_approach_with_the_token_starts_a_run(loop, events):
    token = flex_door.handle(hub, "acquireControl", {"client": "s"}, role="command",
                             ctx=None)["token"]
    events.clear()
    out = _flex({"token": token})
    assert "run_id" in out and "est_s" in out
    assert events[0][2]["row"] == "startApproach" and events[-1][1] == "reply"


def test_http_approach_takes_access_once(monkeypatch, client):
    calls = []
    real = type(hub).preempt_for_operator

    async def counting(self, reason):
        calls.append(reason)
        return await real(self, reason)

    monkeypatch.setattr(type(hub), "preempt_for_operator", counting)
    access.script_requests(hub, "script")
    reply = client.post("/api/approach", json={})
    assert reply.status_code == 200
    assert calls == ["/api/approach"]


NOT_A_DICT = [("dict_type", ("body",), "Input should be a valid dictionary")]
CASES = [("array", b"[1]", JSON, 422, NOT_A_DICT),
         ("string", b'"s"', JSON, 422, NOT_A_DICT),
         ("number", b"5", JSON, 422, NOT_A_DICT),
         ("broken", b"{", JSON, 422, [("json_invalid", ("body", 1), "JSON decode error")]),
         ("blank", b"   ", JSON, 422, [("json_invalid", ("body", 3), "JSON decode error")]),
         ("text_plain", b"{}", {"content-type": "text/plain"}, 422, NOT_A_DICT),
         ("no_header", b"{}", {}, 422, NOT_A_DICT)]


@pytest.mark.parametrize("label,body,headers,status,expected", CASES, ids=[c[0] for c in CASES])
def test_a_malformed_approach_request_is_answered_as_before(client, label, body, headers,
                                                            status, expected):
    reply = client.post("/api/approach", content=body, headers=headers)
    assert reply.status_code == status
    got = [(d["type"], tuple(d["loc"]), d["msg"]) for d in reply.json()["detail"]]
    assert got == expected


# --- the operator take fires exactly once, whatever the request then turns out to be ----------

@pytest.fixture
def takes(monkeypatch):
    calls = []
    real = type(hub).preempt_for_operator

    async def counting(self, reason):
        calls.append(reason)
        return await real(self, reason)

    monkeypatch.setattr(type(hub), "preempt_for_operator", counting)
    access.script_requests(hub, "script")
    return calls


def test_a_malformed_body_still_takes_once(client, takes):
    reply = client.post("/api/approach", content=b"[1]", headers=JSON)
    assert reply.status_code == 422
    assert takes == ["/api/approach"]


def test_a_null_body_takes_once_and_engages(client, takes, litho_ready_hub):
    reply = client.post("/api/approach", content=b"null", headers=JSON)
    assert reply.status_code == 200 and "error" not in reply.json()
    assert takes == ["/api/approach"]


def test_a_busy_refusal_takes_once(client, takes):
    claim = hub.claim_run("scan")
    try:
        reply = client.post("/api/approach", json={})
    finally:
        hub.release_claim(claim)
    assert reply.status_code == 200 and reply.json()["error"] == "busy"
    assert takes == ["/api/approach"]


def test_a_refusal_while_engaged_takes_once(client, takes, monkeypatch):
    monkeypatch.setattr(hub.host, "engaged", True)
    claim = hub.claim_run("scan")
    try:
        reply = client.post("/api/approach", json={})
    finally:
        hub.release_claim(claim)
    assert reply.status_code == 200 and "error" in reply.json()
    assert takes == ["/api/approach"]


def test_a_backend_mismatch_refusal_takes_once(client, takes, monkeypatch):
    monkeypatch.setattr(hub, "require_live_backend", lambda: "bridge is not live")
    reply = client.post("/api/approach", json={})
    assert reply.status_code == 200 and reply.json()["reason"] == "backend_not_live"
    assert takes == ["/api/approach"]


def test_start_approach_on_the_read_socket_is_method_not_found(loop):
    with pytest.raises(RpcError) as bad:
        flex_door.handle(hub, "startApproach", {"token": "t"}, role="read", ctx=None)
    assert bad.value.code == -32601
