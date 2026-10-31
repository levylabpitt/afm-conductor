"""The tip rows (slice 2.8a): setMode over FLEX (token gate) and POST /api/mode (operator take),
the refusal while engaged unchanged, and malformed bodies answered as the old route answered."""

from __future__ import annotations

import asyncio
import threading

import pytest
from afm_litho.service import access
from afm_litho.service import control as ctl
from afm_litho.service import journal
from afm_litho.service import flex_dispatch as fd
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
    saved = (hub.host.imaging_mode, hub.host.free_air)
    hub.control = None
    yield
    hub.control = None
    hub.host.imaging_mode, hub.host.free_air = saved


def _flex(params):
    return flex_door.handle(hub, "setMode", params, role="command", ctx=None)


def test_set_mode_is_one_row_with_both_carriers_and_their_gates():
    row = table.flex_row("setMode")
    assert row.control == "tip" and row.flex.gate == "token"
    assert (row.http.method, row.http.path, row.http.gate) == ("POST", "/api/mode",
                                                               "operator_tip")


def test_flex_set_mode_without_the_token_is_refused_before_the_handler(loop, events):
    with pytest.raises(RpcError) as bad:
        _flex({"mode": "contact"})
    assert bad.value.code == ctl.CONTROL_REQUIRED
    assert [(s, w) for s, w, _ in events] == [("door", "request"), ("door", "error")]


def test_the_flex_gate_refuses_before_the_handler_runs(loop, monkeypatch):
    entered = []
    monkeypatch.setattr(fd, "_h_set_mode", lambda *a: entered.append(a))
    with pytest.raises(RpcError):
        _flex({"mode": "contact"})
    assert entered == []


@pytest.mark.parametrize("verb", ["setMode", "heartbeat"])
def test_a_token_gated_verb_given_non_object_params_answers_as_the_registry_did(loop, verb):
    for params in ([1], "s"):
        with pytest.raises(RpcError) as bad:
            flex_door.handle(hub, verb, params, role="command", ctx=None)
        if verb == "setMode":
            assert (bad.value.code, bad.value.data) == (-32602, {"method": "setMode"})
            assert bad.value.message == ("setMode takes a params OBJECT with named fields "
                                         f"(it carries a `token`), got {type(params).__name__}")
        else:
            assert bad.value.code == ctl.CONTROL_REQUIRED


def test_a_positional_heartbeat_token_still_passes_the_gate(loop):
    token = flex_door.handle(hub, "acquireControl", {"client": "s"}, role="command",
                             ctx=None)["token"]
    out = flex_door.handle(hub, "heartbeat", [token], role="command", ctx=None)
    assert out["valid"] is True


def test_flex_set_mode_with_the_token_switches_and_is_journalled(loop, events):
    token = flex_door.handle(hub, "acquireControl", {"client": "s"}, role="command",
                             ctx=None)["token"]
    events.clear()
    out = _flex({"token": token, "mode": "ac"})
    assert out["ok"] is True and out["mode"] == "ac" and hub.host.imaging_mode == "ac"
    assert events[0][2]["row"] == "setMode" and events[-1][1] == "reply"


def test_flex_bad_mode_keeps_its_invalid_params_error(loop):
    token = flex_door.handle(hub, "acquireControl", {"client": "s"}, role="command",
                             ctx=None)["token"]
    with pytest.raises(RpcError) as bad:
        _flex({"token": token, "mode": "tapping"})
    assert bad.value.data == {"param": "mode"}


def test_http_set_mode_switches_and_takes_access_once(monkeypatch, client):
    calls = []
    real = type(hub).preempt_for_operator

    async def counting(self, reason):
        calls.append(reason)
        return await real(self, reason)

    monkeypatch.setattr(type(hub), "preempt_for_operator", counting)
    access.script_requests(hub, "script")
    reply = client.post("/api/mode", json={"mode": "contact"})
    assert reply.status_code == 200 and reply.json()["mode"] == "contact"
    assert calls == ["/api/mode"]


def test_http_set_mode_is_refused_while_engaged_and_still_takes(monkeypatch, client):
    monkeypatch.setattr(hub.host, "engaged", True)
    access.script_requests(hub, "script")
    reply = client.post("/api/mode", json={"mode": "ac"})
    assert reply.status_code == 200
    assert reply.json()["reason"] == "engaged" and "withdraw first" in reply.json()["error"]
    assert hub.control is None or hub.control.client != "script"


# --- malformed requests: statuses and bodies as the old route answered them (pinned) --------

BAD_MODE = {"error": "mode must be 'contact' or 'ac'", "reason": "bad_mode"}
NOT_A_DICT = [("dict_type", ("body",), "Input should be a valid dictionary")]
CASES = [("array", b"[1]", JSON, 422, NOT_A_DICT),
         ("string", b'"s"', JSON, 422, NOT_A_DICT),
         ("number", b"5", JSON, 422, NOT_A_DICT),
         ("broken", b"{", JSON, 422, [("json_invalid", ("body", 1), "JSON decode error")]),
         ("blank", b"   ", JSON, 422, [("json_invalid", ("body", 3), "JSON decode error")]),
         ("text_plain", b'{"mode": "ac"}', {"content-type": "text/plain"}, 422, NOT_A_DICT),
         ("no_header", b'{"mode": "ac"}', {}, 422, NOT_A_DICT),
         ("empty", b"", JSON, 200, BAD_MODE),
         ("no_mode", b"{}", JSON, 200, BAD_MODE),
         ("bad_mode", b'{"mode": "x"}', JSON, 200, BAD_MODE),
         ("null_mode", b'{"mode": null}', JSON, 200, BAD_MODE)]


@pytest.mark.parametrize("label,body,headers,status,expected", CASES, ids=[c[0] for c in CASES])
def test_a_malformed_mode_request_is_answered_as_before(client, label, body, headers, status,
                                                        expected):
    reply = client.post("/api/mode", content=body, headers=headers)
    assert reply.status_code == status
    if status == 422:
        got = [(d["type"], tuple(d["loc"]), d["msg"]) for d in reply.json()["detail"]]
    else:
        got = reply.json()
    assert got == expected


@pytest.mark.parametrize("verb", ["setMode", "startApproach", "setZGain", "startScan",
                                  "setContinuous", "startWrite"])
def test_a_flex_tip_verb_with_non_object_params_is_refused_and_journalled(events, verb):
    """The intake's refusal goes through `dispatch`: who tried what is in the journal."""
    with pytest.raises(RpcError) as err:
        flex_door.handle(hub, verb, [1, 2], role="command", ctx=None)
    assert err.value.code == fd.INVALID_PARAMS
    assert [(s, w, f["row"]) for s, w, f in events] == [("door", "request", verb),
                                                        ("door", "error", verb)]
    assert events[0][2]["params"] == [1, 2]
    assert events[1][2]["code"] == fd.INVALID_PARAMS
