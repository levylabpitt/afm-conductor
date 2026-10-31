"""The write row (slice 2.8e): startWrite over FLEX (token gate) and POST /api/litho/write over HTTP
(the operator take, once, before the body is read), with the two carriers' setpoint intakes.

The parity tables below pin what the old handlers answered. They hold on base too; only the row
shape and intake tests (marked) need the new names.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from afm_litho.hardware.lithography_run import WriteSetpoint
from afm_litho.litho.compiler import demo_pattern
from afm_litho.service import access
from afm_litho.service import control as ctl
from afm_litho.service import flex_dispatch as fd
from afm_litho.service.host import litho_cmds
from afm_litho.service.commands.errors import CommandError
from afm_litho.service.flex_dispatch import RpcError
from afm_litho.service.hub import hub
from afm_litho.service.requests import flex_door, table

JSON = {"content-type": "application/json"}
PLAIN = {"content-type": "text/plain"}


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    thread = threading.Thread(target=lp.run_forever, daemon=True)
    thread.start()
    prev = getattr(hub, "_loop", None)
    hub._loop = lp
    yield lp
    hub._loop = prev
    lp.call_soon_threadsafe(lp.stop)
    thread.join(5)


@pytest.fixture(autouse=True)
def free_instrument():
    hub.control = None
    prev = getattr(hub, "pattern", None)
    hub.pattern = demo_pattern()
    prev_written = getattr(hub.host, "written_objects", set())
    hub.host.written_objects = set()          # an earlier test's marks would answer already_written
    yield
    hub.control = None
    hub.pattern = prev
    hub.host.written_objects = prev_written


@pytest.fixture
def takes(monkeypatch):
    calls = []
    real = type(hub).preempt_for_operator

    async def counting(self, reason):
        calls.append(reason)
        return await real(self, reason)

    monkeypatch.setattr(type(hub), "preempt_for_operator", counting)
    return calls


def _flex(params):
    return flex_door.handle(hub, "startWrite", params, role="command", ctx=None)


def _token():
    return flex_door.handle(hub, "acquireControl", {"client": "s"}, role="command",
                            ctx=None)["token"]


# --- the row (new names) ---------------------------------------------------------------------

def test_start_write_is_a_tip_row_gated_by_token_over_flex_and_the_take_over_http():
    row = table.flex_row("startWrite")
    assert row.control == "tip" and row.flex.gate == "token" and row.flex.socket == "command"
    assert (row.http.method, row.http.path, row.http.gate) == ("POST", "/api/litho/write",
                                                              "operator_tip")
    assert "/api/litho/write" in ctl.TIP_MUTATING


def test_flex_without_the_token_is_refused_before_the_handler(loop, monkeypatch):
    entered = []
    monkeypatch.setattr(fd, "_h_start_write", lambda *a: entered.append(a))
    with pytest.raises(RpcError) as bad:
        _flex({"objects": "all"})
    assert bad.value.code == ctl.CONTROL_REQUIRED and entered == []


# --- the intakes: the one place the carriers differ (new names) ------------------------------

def test_the_flex_intake_needs_a_setpoint_and_reads_its_reference():
    intake = litho_cmds.flex_intake
    assert intake.setpoint({"setpoint": "0.35"}) == WriteSetpoint(0.35, relative=False)
    free_air = intake.setpoint({"setpoint": 0.35, "setpoint_reference": "free_air"})
    assert free_air == WriteSetpoint(0.35, relative=True)
    absolute = intake.setpoint({"setpoint": 0.35, "setpoint_reference": "absolute"})
    assert absolute == WriteSetpoint(0.35, relative=False)
    assert intake.verb == "startWrite"
    assert intake.refuse_selection({"all": True}) is None


def test_the_http_intake_needs_a_setpoint_and_reads_its_reference():
    intake = litho_cmds.http_intake
    assert intake.setpoint({"setpoint": 0.2}) == WriteSetpoint(0.2, relative=False)
    absolute = intake.setpoint({"setpoint": 0.2, "setpoint_reference": "absolute"})
    assert absolute == WriteSetpoint(0.2, relative=False)
    free_air = intake.setpoint({"setpoint": 0.2, "setpoint_reference": "free_air"})
    assert free_air == WriteSetpoint(0.2, relative=True)
    # the cockpit's own key: the same thing as a free-air setpoint
    assert intake.setpoint({"setpoint_relative": 0.1}) == WriteSetpoint(0.1, relative=True)
    assert intake.verb == "litho/write"
    with pytest.raises(CommandError) as bad:
        intake.refuse_selection({"all": True})
    assert bad.value.http_body()["reason"] == "unknown_object"


@pytest.mark.parametrize("intake", ["flex_intake", "http_intake"])
@pytest.mark.parametrize("body", [{}, {"setpoint": None}, {"setpoint_reference": "absolute"}])
def test_a_write_with_no_setpoint_is_refused_on_both_doors(intake, body):
    with pytest.raises(CommandError) as bad:
        getattr(litho_cmds, intake).setpoint(body)
    assert bad.value.http_body()["reason"] == "setpoint_missing"
    assert "required" in bad.value.http_body()["error"]


@pytest.mark.parametrize("intake", ["flex_intake", "http_intake"])
def test_a_setpoint_reference_other_than_free_air_or_absolute_is_refused(intake):
    body = {"setpoint": 0.2, "setpoint_reference": "relative"}
    with pytest.raises(CommandError) as bad:
        getattr(litho_cmds, intake).setpoint(body)
    assert "setpoint_reference" in bad.value.http_body()["error"]


def test_the_cockpit_key_with_setpoint_or_with_absolute_is_refused():
    intake = litho_cmds.http_intake
    for body in ({"setpoint": 0.2, "setpoint_relative": 0.1},
                 {"setpoint_relative": 0.1, "setpoint_reference": "absolute"}):
        with pytest.raises(CommandError) as bad:
            intake.setpoint(body)
        assert bad.value.http_body()["reason"] == "invalid_params"


def test_flex_refuses_the_cockpit_key_with_or_without_a_setpoint():
    for body in ({"setpoint": 0.2, "setpoint_relative": 0.1}, {"setpoint_relative": 0.1}):
        with pytest.raises(CommandError) as bad:
            litho_cmds.flex_intake.setpoint(body)
        assert bad.value.http_body()["reason"] == "invalid_params"


def test_start_write_without_a_setpoint_is_refused_before_anything_is_claimed(loop):
    tok = _token()
    with pytest.raises(RpcError) as bad:
        _flex({"token": tok, "objects": "all", "amp_gain": 25.0})
    assert "setpoint" in str(bad.value)
    assert hub._in_flight_run_id() is None


class _FreeAir:
    value_v = 0.5


def test_a_relative_setpoint_is_added_to_the_free_air_and_stands_alone_without_one():
    class _Hub:
        free_air = _FreeAir()

    class _NoAir:
        free_air = None

    _Hub.host, _NoAir.host = _Hub, _NoAir  # fake: host fields live on the fake itself

    relative = WriteSetpoint(0.1, relative=True)
    assert litho_cmds._absolute_setpoint(_Hub, relative) == pytest.approx(0.6)
    assert litho_cmds._absolute_setpoint(_NoAir, relative) == 0.1
    assert litho_cmds._absolute_setpoint(_Hub, WriteSetpoint(0.2, relative=False)) == 0.2


def test_litho_write_has_no_door_argument():
    import inspect
    assert "door" not in inspect.signature(litho_cmds.litho_write).parameters


# --- the take ---------------------------------------------------------------------------------

@pytest.mark.parametrize("body,headers", [(b"[1]", JSON), (b'{"amp_gain": 25, "setpoint": 0.2}',
                                                            JSON)])
def test_a_refused_write_request_still_takes_once_before_the_body_is_read(client, takes, body,
                                                                         headers):
    access.script_requests(hub, "script")
    client.post("/api/litho/write", content=body, headers=headers)
    assert takes == ["/api/litho/write"]


# --- malformed HTTP bodies: answers pinned (hold on base) -------------------------------------

NOT_A_DICT = (422, [("dict_type", ("body",), "Input should be a valid dictionary")])
NO_GAIN = (200, {"error": "amp_gain is required and was not given", "reason": "amp_gain_missing"})
NO_SETPOINT = (200, {"error": "litho/write: `setpoint` (volts) is required, with "
                    '`setpoint_reference` "absolute" (default) or "free_air"',
                    "reason": "setpoint_missing"})
ALL = (200, {"error": "litho/write: the REST door takes an explicit list of object ids; the "
             "`all` select-all key is a FLEX startWrite convenience, not a valid object id",
             "reason": "unknown_object"})


def _bad(what, _got=None):
    return (200, {"error": f"litho/write: {what}", "reason": "bad_params"})


HTTP_CASES = [
    ("null", b"null", JSON, NO_SETPOINT),
    ("empty", b"", JSON, NO_SETPOINT),
    ("no_setpoint", b"{}", JSON, NO_SETPOINT),
    ("no_gain", b'{"setpoint": 0.1}', JSON, NO_GAIN),
    ("no_gain_cockpit_key", b'{"setpoint_relative": 0.1}', JSON, NO_GAIN),
    ("array", b"[1]", JSON, NOT_A_DICT),
    ("string", b'"s"', JSON, NOT_A_DICT),
    ("text_plain", b'{"amp_gain": 25}', PLAIN, NOT_A_DICT),
    ("setpoint_string", b'{"amp_gain": 25, "setpoint": "x"}', JSON,
     _bad("setpoint must be a number, got str 'x'", None)),
    ("bad_reference", b'{"amp_gain": 25, "setpoint": 0.1, "setpoint_reference": "rel"}', JSON,
     _bad('setpoint_reference must be "free_air" or "absolute", got \'rel\'', None)),
    ("absolute_past_the_sum_ceiling",
     b'{"amp_gain": 25, "setpoint": 50, "setpoint_reference": "absolute"}', JSON,
     (200, {"error": "|setpoint| 50 V exceeds 0.9*SUM (4.5 V; SUM=5 V)",
            "reason": "setpoint_exceeds_sum"})),
    ("free_air_past_the_sum_ceiling",
     b'{"amp_gain": 25, "setpoint": 50, "setpoint_reference": "free_air"}', JSON,
     (200, {"error": "|setpoint| 50 V exceeds 0.9*SUM (4.5 V; SUM=5 V)",
            "reason": "setpoint_exceeds_sum"})),
    ("absolute_null_then_bad_relative",
     b'{"amp_gain": 25, "setpoint": null, "setpoint_relative": "x"}', JSON,
     _bad("setpoint_relative must be a number, got str 'x'", None)),
    ("relative_string", b'{"amp_gain": 25, "setpoint_relative": "loud"}', JSON,
     _bad("setpoint_relative must be a number, got str 'loud'", None)),
    ("relative_bool", b'{"amp_gain": 25, "setpoint_relative": true}', JSON,
     _bad("setpoint_relative must be a number, got bool True", None)),
    ("relative_list", b'{"amp_gain": 25, "setpoint_relative": [1]}', JSON,
     _bad("setpoint_relative must be a number, got list [1]", None)),
    ("all_with_gain", b'{"all": true, "amp_gain": 25}', JSON, ALL),
    ("all_without_gain", b'{"all": true}', JSON, ALL),
    ("gain_string", b'{"amp_gain": "x", "setpoint_relative": 0.1}', JSON,
     _bad("amp_gain must be a number, got str 'x'", None)),
    ("gain_bool", b'{"amp_gain": true, "setpoint_relative": 0.1}', JSON,
     _bad("amp_gain must be a number, got bool True", None)),
    ("twin_speed_string", b'{"amp_gain": 25, "setpoint_relative": 0.1, "twin_speed": "fast"}',
     JSON, _bad("twin_speed must be a number, got str 'fast'", None)),
    ("no_such_object", b'{"amp_gain": 25, "setpoint_relative": 0.1, "objects": [99]}', JSON,
     (200, {"error": "startWrite: object index 99 is out of range -- the loaded design has 4 "
            "object(s)", "reason": "bad_params"})),
    ("relative_past_the_sum_ceiling", b'{"amp_gain": 25, "setpoint_relative": 50}', JSON,
     (200, {"error": "|setpoint| 50 V exceeds 0.9*SUM (4.5 V; SUM=5 V)",
            "reason": "setpoint_exceeds_sum"})),
    ("gain_not_the_models", b'{"amp_gain": 1e9, "setpoint_relative": 0.1}', JSON,
     (200, {"error": "amp_gain 1000000000.0 does not match the model's 25.0",
            "reason": "amp_gain_mismatch"})),
]


@pytest.mark.parametrize("label,body,headers,expected", HTTP_CASES,
                         ids=[c[0] for c in HTTP_CASES])
def test_a_malformed_write_request_is_answered_as_before(client, label, body, headers, expected):
    reply = client.post("/api/litho/write", content=body, headers=headers)
    status, want = expected
    assert reply.status_code == status
    if status == 422:
        got = [(d["type"], tuple(d["loc"]), d["msg"]) for d in reply.json()["detail"]]
        assert got == want
    else:
        assert reply.json() == want


# --- malformed FLEX params: answers pinned (hold on base) -------------------------------------

def _refusal(params):
    with pytest.raises(RpcError) as bad:
        _flex(params)
    return bad.value.code, bad.value.message, bad.value.data


def test_flex_non_object_params_are_refused_before_the_token_gate(loop):
    for params, kind in (([1], "list"), ("s", "str")):
        assert _refusal(params) == (
            -32602, f"startWrite takes a params OBJECT with named fields (it carries a "
                    f"`token`), got {kind}", {"method": "startWrite"})


def test_flex_without_a_token_or_with_a_stale_one_is_refused(loop):
    _token()
    want = (-32030, "control_required: this call needs the commander token; call "
                    "acquireControl first", {"holder": "s", "valid": False})
    assert _refusal({"objects": "all"}) == want
    assert _refusal({"token": "zzz", "objects": "all"}) == want


@pytest.mark.parametrize("extra,message,param", [
    ({"setpoint": 0.1, "objects": "x"}, "startWrite: objects must be \"all\" or a list of object indices or "
                       "ids, got 'x'", "objects"),
    ({"setpoint": 0.1, "objects": 5}, "startWrite: objects must be \"all\" or a list of object indices or "
                     "ids, got int", "objects"),
    ({"setpoint": "loud"}, "startWrite: setpoint must be a number, got str 'loud'", "setpoint"),
    ({"setpoint": True}, "startWrite: setpoint must be a number, got bool True", "setpoint"),
    ({"setpoint": 0.1, "amp_gain": "q"}, "startWrite: amp_gain must be a number, got str 'q'", "amp_gain"),
    ({"setpoint": 0.1, "twin_speed": "warp"}, "startWrite: twin_speed must be a number, got str 'warp'",
     "twin_speed"),
    ({"setpoint": 0.1, "deadman_s": "abc"}, "startWrite: deadman_s must be a number of seconds, got str",
     "deadman_s"),
    ({"setpoint": 0.1, "deadman_s": True}, "startWrite: deadman_s must be a number of seconds, got bool",
     "deadman_s"),
])
def test_flex_bad_parameters_are_refused_naming_the_parameter(loop, extra, message, param):
    assert _refusal({"token": _token(), **extra}) == (-32602, message, {"param": param})


def test_flex_with_the_operator_holding_is_revoked_even_with_a_deadman_s(loop):
    token = _token()
    asyncio.run_coroutine_threadsafe(hub.preempt_for_operator("test"), loop).result(10)
    assert _refusal({"token": token, "objects": "all", "deadman_s": 30}) == (
        -32011, "control_revoked: control was revoked (preempted or expired)",
        {"holder": None, "reason": "revoked", "valid": False})
