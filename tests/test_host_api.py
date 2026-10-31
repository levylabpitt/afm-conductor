"""HostApi v1: the surface is pinned, and each command verb reaches the function it wraps."""

from __future__ import annotations

import asyncio
import importlib
import inspect

import pytest

from afm_litho.service.host import api
from afm_litho.service.requests import table  # noqa: F401  (loads the rows before tips/edits)
from afm_litho.service.host.api import (COMMAND_VERBS, HOST_API_VERSION, STOP_VERBS, VERBS,
                                        HostApi)

V1_VERBS = (
    "start_scan", "start_write", "set_continuous", "set_mode", "set_zgain", "approach",
    "start_approach", "clear_fault", "set_override",
    "park", "abort_write", "withdraw", "stop_scan",
    "commander_lapsed", "snapshot",
)

# verb -> (module that holds the function, its name, parameters after self)
WRAPPED = {
    "start_scan": ("afm_litho.service.host.scan_cmds", "start_scan", ["cfg"]),
    "start_write": ("afm_litho.service.host.litho_cmds", "start_write", ["body"]),
    "set_continuous": ("afm_litho.service.host.continuous_cmds", "set_continuous_wire",
                       ["body"]),
    "set_mode": ("afm_litho.service.host.scan_cmds", "set_mode", ["body"]),
    "set_zgain": ("afm_litho.service.host.scan_cmds", "set_zgain", ["body"]),
    "approach": ("afm_litho.service.host.scan_cmds", "approach", ["body"]),
    "start_approach": ("afm_litho.service.host.scan_cmds", "start_approach", ["body"]),
    "clear_fault": ("afm_litho.service.host.instrument_cmds", "clear_fault", ["body"]),
    "set_override": ("afm_litho.service.host.instrument_cmds", "set_override", ["body"]),
}


def test_version_and_verbs_are_exactly_v1():
    assert HOST_API_VERSION == 1
    assert tuple(name for name, _ in VERBS) == V1_VERBS
    assert COMMAND_VERBS == V1_VERBS[:9]


def test_the_class_has_exactly_the_command_and_stop_verbs():
    methods = {n for n, _ in inspect.getmembers(HostApi, inspect.isfunction)
               if not n.startswith("_")}
    assert STOP_VERBS == ("park", "abort_write", "withdraw", "stop_scan")
    assert methods == set(COMMAND_VERBS) | set(STOP_VERBS) | {"commander_lapsed"}
    assert len(methods) == 14


def test_commander_lapsed_reads_the_state_in_the_call_and_returns_the_act(monkeypatch):
    """Plain, like the stop verbs: the state is read and the verdict made when the base
    calls it, and the returned awaitable only acts."""
    deadman = importlib.import_module("afm_litho.service.host.deadman")
    assert not inspect.iscoroutinefunction(HostApi.commander_lapsed)
    assert list(inspect.signature(HostApi.commander_lapsed).parameters)[1:] == [
        "holder", "withdraw_on_deadman"]
    reads, acts = [], []

    async def act(hub, holder, state, action):
        acts.append((holder, state, action))
        return {"action": action}

    monkeypatch.setattr(deadman, "pass_state", lambda hub: reads.append(hub) or "engaged")
    monkeypatch.setattr(deadman, "_act_on_lapse", act)
    hub = object()
    pending = HostApi(hub).commander_lapsed("s1", True)
    assert reads == [hub] and acts == []
    assert asyncio.run(pending) == {"action": "withdraw"}
    assert acts == [("s1", "engaged", "withdraw")]


@pytest.mark.parametrize("verb", COMMAND_VERBS)
def test_parameter_names_are_pinned(verb):
    params = list(inspect.signature(getattr(HostApi, verb)).parameters)[1:]
    assert params == WRAPPED[verb][2]


def test_every_verb_has_a_docstring():
    assert api.__doc__ and all(getattr(HostApi, v).__doc__
                               for v in COMMAND_VERBS + STOP_VERBS)


@pytest.mark.parametrize("verb", COMMAND_VERBS)
def test_each_verb_calls_its_function_and_returns_its_reply(verb, monkeypatch):
    module, name, _params = WRAPPED[verb]
    target = importlib.import_module(module)
    calls = []
    is_async = inspect.iscoroutinefunction(getattr(HostApi, verb))

    if is_async:
        async def stub(*args):
            calls.append(args)
            return {"reply": verb}
    else:
        def stub(*args):
            calls.append(args)
            return {"reply": verb}

    monkeypatch.setattr(target, name, stub)
    hub, arg = object(), {"x": 1}
    call = getattr(HostApi(hub), verb)(arg)
    reply = asyncio.run(call) if is_async else call
    assert reply == {"reply": verb}
    assert calls == [(hub, arg)]


# verb -> the row handler (module, name) that must reach it through HostApi
ROW_TO_VERB = {
    "set_mode": ("afm_litho.service.requests.tips", "_operator_set_mode"),
    "approach": ("afm_litho.service.requests.tips", "_operator_approach"),
    "set_zgain": ("afm_litho.service.requests.tips", "_cockpit_set_zgain"),
    "set_override": ("afm_litho.service.requests.edits", "_instrument_override"),
    "clear_fault": ("afm_litho.service.requests.edits", "_fault_clear"),
}


@pytest.mark.parametrize("verb", ROW_TO_VERB)
def test_a_command_row_reaches_the_host_through_its_verb(verb, monkeypatch):
    module, name = ROW_TO_VERB[verb]
    mod = importlib.import_module(module)
    seen = []

    async def async_stub(self, body):
        seen.append((verb, body))
        return {"reply": verb}

    def sync_stub(self, body=None):
        seen.append((verb, body))
        return {"reply": verb}

    stub = async_stub if inspect.iscoroutinefunction(getattr(HostApi, verb)) else sync_stub
    monkeypatch.setattr(HostApi, verb, stub)
    monkeypatch.setattr(mod, "current_hub", lambda: object(), raising=False)
    assert asyncio.run(getattr(mod, name)({"b": 1})) == {"reply": verb}
    assert seen == [(verb, {"b": 1})]


@pytest.mark.parametrize("verb,handler", [
    ("set_mode", "_h_set_mode"), ("start_approach", "_h_start_approach"),
    ("set_zgain", "_h_set_zgain"), ("start_scan", "_h_start_scan"),
    ("set_continuous", "_h_set_continuous"), ("start_write", "_h_start_write")])
def test_a_flex_handler_reaches_the_host_through_its_verb(verb, handler):
    from afm_litho.service import flex_dispatch as fd

    assert f"HostApi(hub).{verb}(body)" in inspect.getsource(getattr(fd, handler))


# --- the stop verbs --------------------------------------------------------------------------

# verb -> (module that holds today's function, its name, the verb's parameters after self)
STOP_WRAPPED = {
    "park": ("afm_litho.service.host.safe_park", "request_park", ["why", "stop_id"]),
    "abort_write": ("afm_litho.service.host.litho_cmds", "abort_write", []),
    "withdraw": ("afm_litho.service.host.safe_park", "request_park", ["body"]),
    "stop_scan": ("afm_litho.service.host.scan_cmds", "stop_scan", ["body"]),
}
STOP_ARGS = {"park": ("why", "sid"), "abort_write": (), "withdraw": ({"b": 1},),
             "stop_scan": ({"b": 1},)}


@pytest.mark.parametrize("verb", STOP_VERBS)
def test_stop_verb_parameter_names_are_pinned(verb):
    params = list(inspect.signature(getattr(HostApi, verb)).parameters)[1:]
    assert params == STOP_WRAPPED[verb][2]


@pytest.mark.parametrize("verb", STOP_VERBS)
@pytest.mark.host_api_unserialized
def test_a_stop_verb_is_a_plain_method_returning_todays_awaitable(verb, monkeypatch):
    # Plain, not async: the verb adds no coroutine of its own, so the caller awaits exactly
    # the object today's function returned.
    assert not inspect.iscoroutinefunction(getattr(HostApi, verb))
    module, name, _ = STOP_WRAPPED[verb]
    calls, sentinel = [], object()

    def stub(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(importlib.import_module(module), name, stub)
    hub = object()
    assert getattr(HostApi(hub), verb)(*STOP_ARGS[verb]) is sentinel
    expected = {
        "park": ((hub, "why"), {"stop_id": "sid"}),     # today's call; the style defaults to FULL
        "abort_write": ((hub,), {}),
        "withdraw": ((hub, "withdraw", "withdraw"), {"body": {"b": 1}}),   # PARK_WITHDRAW
        "stop_scan": ((hub, {"b": 1}), {}),
    }[verb]
    assert calls == [expected]


def test_park_with_no_stop_id_is_todays_unpaired_full_park(monkeypatch):
    sp = importlib.import_module("afm_litho.service.host.safe_park")
    seen = []

    async def full_park(hub, reason, **kw):
        seen.append((reason, kw))
        return {"verified": True}

    monkeypatch.setattr(sp, "safe_park_async", full_park)
    assert asyncio.run(HostApi(object()).park("why")) == {"verified": True}
    assert seen == [("why", {})]


@pytest.fixture
def stop_spies(monkeypatch):
    """Each stop verb replaced by a spy; returns the (verb, args, kwargs) it was called with."""
    seen = []

    def spy(verb, reply):
        def method(self, *args, **kwargs):
            seen.append((verb, args, kwargs))

            async def done():
                return reply
            return done()
        return method

    monkeypatch.setattr(HostApi, "park", spy("park", {"verified": True, "z_v": 0.0}))
    monkeypatch.setattr(HostApi, "abort_write",
                        spy("abort_write", {"ok": True, "backend": "twin", "safe_park": {}}))
    monkeypatch.setattr(HostApi, "withdraw", spy("withdraw", {"z_v": 0.0}))
    monkeypatch.setattr(HostApi, "stop_scan", spy("stop_scan", {"ok": True, "withdraw": True}))
    return seen


class _Ctx:
    """A FLEX ctx whose `on_loop` runs the coroutine to completion in this thread."""

    @staticmethod
    def on_loop(coro, method):
        return asyncio.run(coro)


class _StubHub:
    @staticmethod
    def backend_label():
        return "twin"


@pytest.mark.parametrize("row,verb,args", [
    ("abort", "park", ("flex abort",)),
    ("safePark", "park", ("flex safePark",)),
    ("withdraw", "withdraw", ({},)),
    ("abortWrite", "abort_write", ()),
])
def test_each_flex_stop_row_reaches_its_verb(stop_spies, monkeypatch, row, verb, args):
    from afm_litho.service import flex_dispatch as fd
    from afm_litho.service.requests import stops
    from afm_litho.service.requests.context import using_hub

    sent = []
    monkeypatch.setattr(stops, "tell_keeper", lambda who, why, sid: sent.append(sid))
    monkeypatch.setattr(fd, "get_state", lambda hub: "idle")
    handler = next(r.handler for r in stops.ROWS if r.name == row)
    with using_hub(_StubHub(), _Ctx()):
        assert handler({})["ok"] is True
    assert [(v, a) for v, a, _ in stop_spies] == [(verb, args)]
    if verb == "park":                    # the keeper's copy and the door's park share one id
        assert stop_spies[0][2] == {"stop_id": sent[0]} and sent[0]


@pytest.mark.parametrize("row,verb,args", [
    ("safePark", "park", ("operator abort",)),
    ("withdraw", "withdraw", ({"b": 1},)),
    ("abortWrite", "abort_write", ()),
    ("abortScan", "stop_scan", ({"b": 1},)),
])
def test_each_browser_stop_row_reaches_its_verb(stop_spies, monkeypatch, row, verb, args):
    from afm_litho.service.requests import stops
    from afm_litho.service.requests.context import using_hub

    monkeypatch.setattr(stops, "tell_keeper", lambda *a: None)
    found = next(r for r in stops.ROWS if r.name == row)
    handler = found.http.handler or found.handler       # abortScan's one handler serves both
    with using_hub(_StubHub()):
        assert asyncio.run(handler(None if row == "abortWrite" else {"b": 1}))
    assert [(v, a) for v, a, _ in stop_spies] == [(verb, args)]


@pytest.mark.parametrize("row", ["abort", "safePark"])
def test_with_the_loop_blocked_a_flex_stop_still_hands_the_keeper_its_stop(monkeypatch, row):
    """K1 for the rows: the keeper's copy, which parks through the stop port with no event
    loop, is sent before the row's verb is handed to the loop."""
    import threading

    from afm_litho.service.requests import stops
    from afm_litho.service.requests.context import using_hub

    told = threading.Event()
    monkeypatch.setattr(stops, "tell_keeper", lambda *a: told.set())
    asked, release = threading.Event(), threading.Event()

    class BlockedLoop:
        @staticmethod
        def on_loop(coro, method):
            coro.close()
            asked.set()
            release.wait(10)               # the loop is blocked: the park never runs here
            return {}

    handler = next(r.handler for r in stops.ROWS if r.name == row)

    def go():
        with using_hub(_StubHub(), BlockedLoop()):
            handler({})

    t = threading.Thread(target=go, daemon=True)
    t.start()
    try:
        assert asked.wait(5)
        assert told.is_set(), "the keeper was not told before the loop was asked"
    finally:
        release.set()
        t.join(5)


# --- H5d: the moved bodies run on the process hub, from every caller --------------------------

@pytest.mark.parametrize("caller,module,name", [
    ("afm_litho.service.requests.edits._fault_clear",
     "afm_litho.service.host.instrument_cmds", "clear_fault"),
    ("afm_litho.service.requests.edits._instrument_override",
     "afm_litho.service.host.instrument_cmds", "set_override"),
    ("afm_litho.service.routers.instrument.post_fault_clear",
     "afm_litho.service.host.instrument_cmds", "clear_fault"),
    ("afm_litho.service.routers.instrument.post_instrument_override",
     "afm_litho.service.host.instrument_cmds", "set_override"),
    ("afm_litho.service.requests.stops._browser_scan_abort",
     "afm_litho.service.host.scan_cmds", "stop_scan"),
    ("afm_litho.service.routers.scan.scan_abort",
     "afm_litho.service.host.scan_cmds", "stop_scan"),
])
def test_every_caller_of_a_moved_body_passes_the_process_hub(caller, module, name, monkeypatch):
    from afm_litho.service.hub import hub as process_hub

    seen = []

    def stub(h, body=None):
        seen.append((h, body))
        if name == "stop_scan":
            async def done():
                return {"ok": True}
            return done()
        return {"ok": True}

    monkeypatch.setattr(importlib.import_module(module), name, stub)
    mod_name, fn = caller.rsplit(".", 1)
    call = getattr(importlib.import_module(mod_name), fn)({"b": 1})
    reply = asyncio.run(call) if inspect.isawaitable(call) else call
    assert reply == {"ok": True}
    assert len(seen) == 1 and seen[0][0] is process_hub and seen[0][1] == {"b": 1}
