"""The scan rows (slice 2.8d): startScan and setContinuous over FLEX (token gate) and over HTTP
(the operator take, once, before the body is read), the REST gaps of C6 unchanged, the chained-false
refusal unchanged, and malformed bodies answered as the old routes answered them."""

from __future__ import annotations

import asyncio
import inspect
import threading

import pytest
from afm_litho.service import access
from afm_litho.service import control as ctl
from afm_litho.service import flex_dispatch as fd
from afm_litho.service.host.api import HostApi
from afm_litho.service.flex_dispatch import RpcError
from afm_litho.service.hub import hub
from afm_litho.service.requests import flex_door, table, tips

JSON = {"content-type": "application/json"}
VERBS = [("startScan", "/api/scan/start", "_h_start_scan"),
         ("setContinuous", "/api/scan/continuous", "_h_set_continuous")]


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


@pytest.fixture
def takes(monkeypatch):
    calls = []
    real = type(hub).preempt_for_operator

    async def counting(self, reason):
        calls.append(reason)
        return await real(self, reason)

    monkeypatch.setattr(type(hub), "preempt_for_operator", counting)
    return calls


def _flex(verb, params):
    return flex_door.handle(hub, verb, params, role="command", ctx=None)


def _token():
    return _flex("acquireControl", {"client": "s"})["token"]


@pytest.mark.parametrize("verb,path,_h", VERBS)
def test_each_is_a_tip_row_gated_by_token_over_flex_and_the_take_over_http(verb, path, _h):
    row = table.flex_row(verb)
    assert row.control == "tip" and row.flex.gate == "token" and row.flex.socket == "command"
    assert (row.http.method, row.http.path, row.http.gate) == ("POST", path, "operator_tip")
    assert path in ctl.TIP_MUTATING


@pytest.mark.parametrize("verb,path,handler", VERBS)
def test_flex_without_the_token_is_refused_before_the_handler(loop, monkeypatch, verb, path,
                                                              handler):
    entered = []
    monkeypatch.setattr(fd, handler, lambda *a: entered.append(a))
    with pytest.raises(RpcError) as bad:
        _flex(verb, {"enabled": False})
    assert bad.value.code == ctl.CONTROL_REQUIRED and entered == []


@pytest.mark.parametrize("verb,path,_h", VERBS)
@pytest.mark.parametrize("params", [[1], "s"])
def test_non_object_params_are_refused_before_the_token_gate(loop, verb, path, _h, params):
    with pytest.raises(RpcError) as bad:
        _flex(verb, params)
    assert (bad.value.code, bad.value.data) == (-32602, {"method": verb})
    assert bad.value.message == (f"{verb} takes a params OBJECT with named fields "
                                 f"(it carries a `token`), got {type(params).__name__}")


def test_flex_start_scan_refuses_the_retired_unchained_scan(loop):
    with pytest.raises(RpcError) as bad:
        _flex("startScan", {"token": _token(), "chained": False})
    assert bad.value.code == -32602 and bad.value.data == {"param": "chained"}
    assert bad.value.message == "startScan: unchained scan was retired; omit 'chained'"


def test_http_start_scan_refuses_the_retired_unchained_scan_and_still_takes(client, takes):
    access.script_requests(hub, "script")
    reply = client.post("/api/scan/start", json={"chained": "off"})
    assert reply.status_code == 200
    assert reply.json() == {"error": "startScan: unchained scan was retired; omit 'chained'",
                            "reason": "bad_params"}
    assert takes == ["/api/scan/start"]
    assert hub.control is None or hub.control.client != "script"


def test_flex_set_continuous_off_with_the_token_answers(loop):
    out = _flex("setContinuous", {"token": _token(), "enabled": False})
    assert out["ok"] is True and out["enabled"] is False


def test_http_set_continuous_off_takes_access_once(client, takes):
    access.script_requests(hub, "script")
    reply = client.post("/api/scan/continuous", json={"enabled": False})
    assert reply.status_code == 200 and reply.json()["enabled"] is False
    assert takes == ["/api/scan/continuous"]


def test_the_two_carriers_keep_their_own_handlers_so_the_rest_gaps_of_c6_stay():
    """FLEX reaches the wire form (silent-twin gate, busy check); HTTP the plain one."""
    assert "HostApi(hub).set_continuous" in inspect.getsource(fd._h_set_continuous)
    assert "set_continuous_wire" in inspect.getsource(HostApi.set_continuous)
    http_side = inspect.getsource(tips._cockpit_set_continuous)
    assert "continuous_set" in http_side and "set_continuous_wire" not in http_side
    assert "_h_set_continuous" in inspect.getsource(tips._script_set_continuous)


def test_a_refused_scan_body_still_takes_before_the_body_is_read(client, takes):
    access.script_requests(hub, "script")
    reply = client.post("/api/scan/start", content=b"[1]", headers=JSON)
    assert reply.status_code == 422
    assert takes == ["/api/scan/start"]


# --- malformed requests: statuses and bodies as the old routes answered them (pinned) -------

NOT_A_DICT = [("dict_type", ("body",), "Input should be a valid dictionary")]
MISSING = [("missing", ("body",), "Field required")]
BROKEN = [("json_invalid", ("body", 1), "JSON decode error")]
BLANK = [("json_invalid", ("body", 3), "JSON decode error")]
UNCHAINED = {"error": "startScan: unchained scan was retired; omit 'chained'",
             "reason": "bad_params"}
SCAN_CASES = [("array", b"[1]", JSON, 422, NOT_A_DICT),
              ("string", b'"s"', JSON, 422, NOT_A_DICT),
              ("number", b"5", JSON, 422, NOT_A_DICT),
              ("broken", b"{", JSON, 422, BROKEN),
              ("blank", b"   ", JSON, 422, BLANK),
              ("text_plain", b'{"chained": 0}', {"content-type": "text/plain"}, 422, NOT_A_DICT),
              ("no_header", b'{"chained": 0}', {}, 422, NOT_A_DICT),
              ("empty", b"", JSON, 422, MISSING),
              ("null", b"null", JSON, 422, NOT_A_DICT),
              ("unchained", b'{"chained": false}', JSON, 200, UNCHAINED)]
CONT_CASES = [("array", b"[1]", JSON, 422, NOT_A_DICT),
              ("string", b'"s"', JSON, 422, NOT_A_DICT),
              ("broken", b"{", JSON, 422, BROKEN),
              ("blank", b"   ", JSON, 422, BLANK),
              ("text_plain", b'{"enabled": false}', {"content-type": "text/plain"}, 422,
               NOT_A_DICT),
              ("no_header", b'{"enabled": false}', {}, 422, NOT_A_DICT),
              ("off", b'{"enabled": false}', JSON, 200, None),
              ("empty_no_config", b"", JSON, 200, "no_config"),
              ("null_json", b"null", JSON, 200, "no_config")]


def _check(reply, status, expected):
    assert reply.status_code == status
    if status == 422:
        got = [(d["type"], tuple(d["loc"]), d["msg"]) for d in reply.json()["detail"]]
        assert got == expected
    elif expected == "no_config":
        assert "no scan config" in reply.json()["error"]
    elif expected is not None:
        assert reply.json() == expected


@pytest.mark.parametrize("label,body,headers,status,expected", SCAN_CASES,
                         ids=[c[0] for c in SCAN_CASES])
def test_a_malformed_scan_start_request_is_answered_as_before(client, label, body, headers,
                                                              status, expected):
    _check(client.post("/api/scan/start", content=body, headers=headers), status, expected)


@pytest.mark.parametrize("label,body,headers,status,expected", CONT_CASES,
                         ids=[c[0] for c in CONT_CASES])
def test_a_malformed_continuous_request_is_answered_as_before(client, label, body, headers,
                                                              status, expected):
    hub.host.last_scan_config = None
    _check(client.post("/api/scan/continuous", content=body, headers=headers), status, expected)
