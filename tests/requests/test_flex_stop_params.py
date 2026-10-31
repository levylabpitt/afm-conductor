"""FLEX abort, safePark, withdraw and abortWrite run for any params shape (slice 2.7c).

The verb identifies the stop. A field that is malformed is ignored; the stop runs and the
reply has the normal success shape. The journal notes the request after the handler.
"""

from __future__ import annotations

import asyncio
import importlib
import threading

import pytest

from afm_litho.service import journal
from afm_litho.service.host import litho_cmds, scan_cmds
from afm_litho.service.hub import hub
from afm_litho.service.requests import flex_door

SHAPES = [[1], "s", 5, None, {"reason": 5}, {"reason": "x", "junk": [1]},
          {"token": "stale-token", "scope": 7}, [["a"], 2], {"token": 5}]
VERBS = {"abort": "park", "safePark": "park", "abortWrite": "abortWrite",
         "withdraw": "withdraw"}


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


@pytest.fixture
def spies(monkeypatch, loop):
    hub.control = None
    ran: list = []
    notes: list = []

    class Journal:
        def note(self, src, what, /, **fields):
            notes.append((what, list(ran)))

    monkeypatch.setattr(journal, "_running", Journal())

    async def parked(h, reason, *a, **k):
        ran.append("park")
        return {"reason": reason, "backend": "twin", "verified": True, "z_v": 0.0}

    sp = importlib.import_module("afm_litho.service.host.safe_park")

    def request_park(h, reason, how=sp.PARK_FULL, **k):
        # PARK_WITHDRAW is the plain withdraw, as in request_park itself.
        if how == sp.PARK_WITHDRAW:
            return scan_cmds.withdraw(h, k.get("body"))
        return parked(h, reason)

    monkeypatch.setattr(sp, "request_park", request_park)

    async def abort_write(h):
        ran.append("abortWrite")
        return {"backend": "twin", "safe_park": {}}

    monkeypatch.setattr(litho_cmds, "abort_write", abort_write)

    async def withdraw(h, body):
        ran.append("withdraw")
        return {"z_v": 0.0}

    monkeypatch.setattr(scan_cmds, "withdraw", withdraw)
    yield ran, notes
    hub.control = None


@pytest.mark.parametrize("params", SHAPES, ids=repr)
@pytest.mark.parametrize("verb", VERBS)
def test_a_flex_stop_runs_for_any_params(spies, verb, params):
    ran, notes = spies
    reply = flex_door.handle(hub, verb, params, role="command", ctx=None)
    assert reply["ok"] is True
    assert ran == [VERBS[verb]]
    assert ("request", [VERBS[verb]]) in notes      # noted after the handler ran


def test_a_valid_reason_is_still_used(spies):
    reply = flex_door.handle(hub, "safePark", {"reason": "because", "junk": [1]},
                             role="command", ctx=None)
    assert reply["reason"] == "because"
    reply = flex_door.handle(hub, "abort", {"scope": "write", "token": 5},
                             role="command", ctx=None)
    assert reply["reason"] == "flex abort (write)"


def _send(verb, params):
    import json

    from afm_litho.service.flex_provider import handle_message
    return json.loads(handle_message(
        hub, json.dumps({"id": "1", "method": verb, "params": params}), role="command"))


@pytest.mark.parametrize("params", [5, "s", True], ids=repr)
@pytest.mark.parametrize("verb", VERBS)
def test_the_provider_runs_a_stop_whose_params_are_not_an_object_or_array(spies, verb, params):
    ran, _ = spies
    reply = _send(verb, params)
    assert "error" not in reply and reply["result"]["ok"] is True
    assert ran == [VERBS[verb]]


def test_the_provider_still_refuses_such_params_for_other_verbs(spies):
    assert _send("getState", 5)["error"]["code"] == -32600


def _send_raw(envelope: dict):
    import json

    from afm_litho.service.flex_provider import handle_message
    return json.loads(handle_message(hub, json.dumps(envelope), role="command"))


@pytest.mark.parametrize("params", [[1], {"reason": 5}, None, "absent"], ids=repr)
@pytest.mark.parametrize("verb", VERBS)
def test_the_provider_runs_a_stop_for_list_dict_null_and_missing_params(spies, verb, params):
    ran, _ = spies
    envelope = {"id": "1", "method": verb}
    if params != "absent":
        envelope["params"] = params
    reply = _send_raw(envelope)
    assert reply["result"]["ok"] is True and ran == [VERBS[verb]]


@pytest.mark.parametrize("msg_id", [7, None, [1], {"a": 1}, 1.5, True])
@pytest.mark.parametrize("verb", VERBS)
def test_the_provider_runs_a_stop_whatever_the_id_is_and_echoes_it(spies, verb, msg_id):
    ran, _ = spies
    reply = _send_raw({"jsonrpc": "2.0", "id": msg_id, "method": verb, "params": 5})
    assert reply["id"] == msg_id and reply["result"]["ok"] is True
    assert ran == [VERBS[verb]]


@pytest.mark.parametrize("method", [5, None, ["abort"], {"m": "abort"}, "", True])
def test_a_method_that_is_not_a_string_names_no_stop_and_is_refused_in_form(spies, method):
    ran, _ = spies
    reply = _send_raw({"id": "1", "method": method, "params": {}})
    assert reply["error"]["code"] == -32600 and ran == []


def test_the_provider_stop_set_is_the_stop_rows():
    from afm_litho.service.flex_provider import _STOP_VERBS
    assert _STOP_VERBS == frozenset(VERBS)
