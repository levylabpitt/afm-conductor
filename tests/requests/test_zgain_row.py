"""The setZGain row (slice 2.8b): FLEX token gate, HTTP with no take and no token (C2), the twin's
refusal unchanged, and malformed bodies answered as the old /api/zgain route answered them."""

from __future__ import annotations

import asyncio
import threading

import pytest
from afm_litho.service import access
from afm_litho.service import control as ctl
from afm_litho.service import flex_dispatch as fd
from afm_litho.service.flex_dispatch import RpcError
from afm_litho.service.hub import hub
from afm_litho.service.requests import flex_door, table

JSON = {"content-type": "application/json"}


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
    hub.control = None
    yield
    hub.control = None


def _flex(params):
    return flex_door.handle(hub, "setZGain", params, role="command", ctx=None)


def _token():
    return flex_door.handle(hub, "acquireControl", {"client": "s"}, role="command",
                            ctx=None)["token"]


def test_z_gain_is_a_tip_row_that_takes_no_control_over_http():
    row = table.flex_row("setZGain")
    assert row.control == "tip" and row.flex.gate == "token"
    assert (row.http.method, row.http.path, row.http.gate) == ("POST", "/api/zgain", "none")
    assert "/api/zgain" not in ctl.TIP_MUTATING


def test_flex_without_the_token_is_refused_before_the_handler(loop, monkeypatch):
    entered = []
    monkeypatch.setattr(fd, "_h_set_zgain", lambda *a: entered.append(a))
    with pytest.raises(RpcError) as bad:
        _flex({"igain": 5})
    assert bad.value.code == ctl.CONTROL_REQUIRED and entered == []


@pytest.mark.parametrize("params", [[1], "s"])
def test_non_object_params_are_refused_before_the_token_gate(loop, params):
    with pytest.raises(RpcError) as bad:
        _flex(params)
    assert (bad.value.code, bad.value.data) == (-32602, {"method": "setZGain"})
    assert bad.value.message == ("setZGain takes a params OBJECT with named fields "
                                 f"(it carries a `token`), got {type(params).__name__}")


def test_the_twin_refuses_a_gain_change_over_flex(loop):
    with pytest.raises(RpcError) as bad:
        _flex({"token": _token(), "igain": 5})
    assert bad.value.code == -32010


def test_http_on_the_twin_is_refused_with_the_body_and_takes_nothing(monkeypatch, client):
    calls = []

    async def counting(self, reason):
        calls.append(reason)

    monkeypatch.setattr(type(hub), "preempt_for_operator", counting)
    access.script_requests(hub, "script")
    reply = client.post("/api/zgain", json={"igain": 5})
    assert reply.status_code == 200
    assert reply.json() == {"error": "Z gain adjust is hardware-only; no bridge connected."}
    assert calls == [] and hub.control is not None and hub.control.client == "script"


NOT_A_DICT = [("dict_type", ("body",), "Input should be a valid dictionary")]
NO_BRIDGE = {"error": "Z gain adjust is hardware-only; no bridge connected."}
CASES = [("array", b"[1]", JSON, 422, NOT_A_DICT),
         ("string", b'"s"', JSON, 422, NOT_A_DICT),
         ("number", b"5", JSON, 422, NOT_A_DICT),
         ("broken", b"{", JSON, 422, [("json_invalid", ("body", 1), "JSON decode error")]),
         ("blank", b"   ", JSON, 422, [("json_invalid", ("body", 3), "JSON decode error")]),
         ("text_plain", b'{"igain": 5}', {"content-type": "text/plain"}, 422, NOT_A_DICT),
         ("no_header", b'{"igain": 5}', {}, 422, NOT_A_DICT),
         ("empty", b"", JSON, 200, NO_BRIDGE),
         ("empty_object", b"{}", JSON, 200, NO_BRIDGE),
         ("good_gains", b'{"igain": 5, "pgain": 1}', JSON, 200, NO_BRIDGE),
         ("bad_number", b'{"igain": "x"}', JSON, 200, None)]


@pytest.mark.parametrize("label,body,headers,status,expected", CASES, ids=[c[0] for c in CASES])
def test_a_malformed_z_gain_request_is_answered_as_before(client, label, body, headers, status,
                                                          expected):
    reply = client.post("/api/zgain", content=body, headers=headers)
    assert reply.status_code == status
    if status == 422:
        got = [(d["type"], tuple(d["loc"]), d["msg"]) for d in reply.json()["detail"]]
        assert got == expected
    elif expected is not None:
        assert reply.json() == expected
    else:
        assert "error" in reply.json()
