"""The control rows (slice 2.6b): FLEX acquire/release/heartbeat, the browser's take, release and
state, the gate moved into `dispatch`, and that a moved route takes access exactly once."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from afm_litho.service import access
from afm_litho.service import control as ctl
from afm_litho.service import journal
from afm_litho.service.flex_dispatch import RpcError
from afm_litho.service.hub import hub
from afm_litho.service.requests import flex_door, http_door, table
from afm_litho.service.requests.context import using_hub
from afm_litho.service.requests.dispatch import dispatch, dispatch_async
from afm_litho.service.requests.table import Http, Row


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
    """A running service loop for the FLEX handlers, which marshal to it."""
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


def _flex(verb, params=None, role="command"):
    return flex_door.handle(hub, verb, params, role=role, ctx=None)


def test_acquire_control_is_a_row_and_is_journalled(loop, events):
    out = _flex("acquireControl", {"client": "script-client", "deadman_s": 20})
    assert out["deadman_s"] == 20 and hub.control.token == out["token"]
    kinds = [(s, w) for s, w, _ in events]
    assert kinds[0] == ("door", "request") and ("access", "grant") in kinds
    assert kinds[-1] == ("door", "reply")
    request = events[0][2]
    assert request["row"] == "acquireControl" and request["actor"] == "flex (unnamed)"
    assert table.flex_row("acquireControl").flex.gate == "none"


def test_heartbeat_gate_refusal_is_journalled_as_request_then_error(loop, events):
    with pytest.raises(RpcError) as bad:
        _flex("heartbeat", {"token": "never-issued"})
    assert bad.value.code == ctl.CONTROL_REQUIRED
    assert [(s, w) for s, w, _ in events] == [("door", "request"), ("door", "error")]
    assert events[1][2]["code"] == ctl.CONTROL_REQUIRED and events[1][2]["row"] == "heartbeat"
    token = _flex("acquireControl", {"client": "a"})["token"]
    assert _flex("heartbeat", {"token": token})["valid"] is True
    _flex("releaseControl", {"token": token})
    with pytest.raises(RpcError) as revoked:
        _flex("heartbeat", {"token": token})
    assert revoked.value.code == ctl.CONTROL_REVOKED


def test_flex_release_without_a_token_is_stale_not_an_error(events):
    assert _flex("releaseControl", {}) == {"ok": True, "stale": True}
    assert _flex("releaseControl", None) == {"ok": True, "stale": True}


def test_control_verbs_are_command_socket_only():
    with pytest.raises(RpcError):
        _flex("acquireControl", {"client": "a"}, role="read")


def test_http_take_release_and_state(loop, client, events):
    token = _flex("acquireControl", {"client": "script"})["token"]
    state = client.get("/api/control").json()
    assert state["control"]["holder"] == "script" and token not in str(state)
    took = client.post("/api/control/take", json={"client": "operator", "reason": "hands on"})
    assert took.status_code == 200 and took.json()["preempted"] == "script"
    assert hub.control.kind == "operator"
    ours = [f for s, w, f in events if s == "door" and w == "request"
            and f["row"] == "takeControl"]
    assert ours and ours[0]["actor"] == "operator"
    assert [f["actor"] for s, w, f in events if (s, w) == ("access", "take")] == ["operator"]
    assert client.post("/api/control/release", json={}).json()["stale"] is False
    assert hub.control is None


def test_http_release_without_a_token_releases_a_script_holder(loop, client):
    _flex("acquireControl", {"client": "script"})
    out = client.post("/api/control/release", json={}).json()
    assert out["ok"] is True and out["stale"] is False and hub.control is None
    assert client.post("/api/control/release", json={}).json()["stale"] is True


def test_stop_rows_never_touch_access(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("a stop row called the access holder")

    monkeypatch.setattr(access, "check", forbidden)
    row = Row(name="t", kind="command", control="stop", handler=lambda r: "stopped")
    assert dispatch(row, "x", {}, gate="token", token="bogus") == "stopped"
    assert asyncio.run(dispatch_async(row, "x", {}, gate="operator_tip",
                                      route="/api/withdraw")) == "stopped"


def test_token_gate_runs_before_the_handler(events):
    ran = []
    row = Row(name="t", kind="command", control="none", handler=lambda r: ran.append(1))
    with using_hub(hub), pytest.raises(ctl.ControlRequired):
        dispatch(row, "x", {}, gate="token", token="bogus")
    assert ran == []
    assert [(s, w) for s, w, _ in events] == [("door", "request"), ("door", "error")]


def test_a_moved_tip_route_takes_access_once(monkeypatch, events):
    """The row's gate takes access, once."""
    row = Row(name="t", kind="command", control="tip", handler=lambda r: {"ok": True},
              http=Http(method="POST", path="/api/withdraw", intake=lambda call: call.body,
                        gate="operator_tip"))
    monkeypatch.setattr(table, "ROWS", [row])
    calls = []
    real = type(hub).preempt_for_operator

    async def counting(self, reason):
        calls.append(reason)
        return await real(self, reason)

    monkeypatch.setattr(type(hub), "preempt_for_operator", counting)
    app = FastAPI()
    http_door.install(app)
    with TestClient(app) as c:
        token = access.script_requests(hub, "script").token
        assert c.post("/api/withdraw", json={}).json() == {"ok": True}
        assert calls == ["/api/withdraw"]
        assert hub.control is None and token in hub._revoked_tokens
        assert [f["route"] for s, w, f in events if (s, w) == ("access", "auto_take")] == [
            "/api/withdraw"]


def test_sync_dispatch_refuses_an_operator_tip_gate():
    row = Row(name="t", kind="command", control="tip", handler=lambda r: "no")
    with pytest.raises(ValueError):
        dispatch(row, "x", {}, gate="operator_tip")


def _probe_apps(monkeypatch, gate="none", path="/api/probe"):
    """A door app with one POST row, and an app with the same route as FastAPI parsed it."""
    row = Row(name="probe", kind="command", control="tip" if gate != "none" else "none",
              handler=lambda r: {"got": r},
              http=Http(method="POST", path=path, intake=INTAKE, gate=gate))
    monkeypatch.setattr(table, "ROWS", [row])
    ours = FastAPI()
    http_door.install(ours)
    base = FastAPI()

    @base.post(path)
    async def probe(body: dict | None = None):
        return {"got": body or {}}

    return ours, base


INTAKE = table.flex_row("releaseControl").http.intake
JSON = {"content-type": "application/json"}


@pytest.mark.parametrize("raw,headers", [
    (b"{bad", JSON), (b'{"a":', JSON), (b"[]", JSON), (b'"s"', JSON), (b"5", JSON),
    (b'{"a": 1}', {}), (b"", JSON), (b'{"client": "x"}', JSON)])
def test_body_errors_match_what_fastapi_returned_before(monkeypatch, raw, headers):
    ours, base = _probe_apps(monkeypatch)
    with TestClient(ours) as a, TestClient(base) as b:
        mine = a.post("/api/probe", content=raw, headers=headers)
        theirs = b.post("/api/probe", content=raw, headers=headers)
    assert (mine.status_code, mine.json()) == (theirs.status_code, theirs.json())


def test_an_operator_tip_row_takes_before_it_reads_a_malformed_body(monkeypatch, events):
    ours, _ = _probe_apps(monkeypatch, gate="operator_tip", path="/api/withdraw")
    with TestClient(ours) as c:
        access.script_requests(hub, "script")
        assert c.post("/api/withdraw", content=b"{bad", headers=JSON).status_code == 422
        assert hub.control is None
        assert [w for s, w, _ in events if s == "access"][-1] == "auto_take"


@pytest.mark.parametrize("path", sorted(ctl.TIP_MUTATING))
def test_each_tip_route_is_taken_exactly_once(monkeypatch, client, path):
    # Every route announces the revocation once. The Withdraw (a stop row, F46) announces in
    # a background task after it has run, so the count is waited for.
    calls = []
    real = type(hub).announce_revoked

    async def counting(self, prev, route):
        calls.append(route)
        return await real(self, prev, route)

    monkeypatch.setattr(type(hub), "announce_revoked", counting)
    access.script_requests(hub, "script")
    client.post(path, content=b"[]", headers=JSON)
    for _ in range(100):
        if calls:
            break
        time.sleep(0.02)
    time.sleep(0.1)
    assert calls == [path]


@pytest.mark.parametrize("path", ["/api/mode", "/api/scan/continuous"])
def test_a_malformed_body_on_a_real_operator_tip_row_is_refused_after_one_take(
        monkeypatch, client, path):
    calls = []
    real = type(hub).preempt_for_operator

    async def counting(self, reason):
        calls.append(reason)
        return await real(self, reason)

    monkeypatch.setattr(type(hub), "preempt_for_operator", counting)
    token = access.script_requests(hub, "script").token
    assert client.post(path, content=b"{bad", headers=JSON).status_code == 422
    assert calls == [path]
    assert hub.control is None and token in hub._revoked_tokens


def test_a_malformed_body_on_the_real_withdraw_row_is_not_refused_and_takes_once(
        monkeypatch, client):
    calls = []
    real = type(hub).announce_revoked

    async def counting(self, prev, route):
        calls.append(route)
        return await real(self, prev, route)

    monkeypatch.setattr(type(hub), "announce_revoked", counting)
    token = access.script_requests(hub, "script").token
    assert client.post("/api/withdraw", content=b"{bad", headers=JSON).status_code == 200
    for _ in range(100):
        if calls:
            break
        time.sleep(0.02)
    time.sleep(0.1)
    assert calls == ["/api/withdraw"]
    assert hub.control is None and token in hub._revoked_tokens


def test_a_tip_route_with_no_post_row_is_refused(monkeypatch):
    monkeypatch.setattr(table, "ROWS", [])
    with pytest.raises(RuntimeError):
        table._check_every_tip_route_has_a_row()


def test_a_table_row_on_a_tip_route_must_take_access(monkeypatch):
    row = Row(name="t", kind="command", control="tip", handler=lambda r: 1,
              http=Http(method="POST", path="/api/withdraw", intake=lambda call: call.body, gate="none"))
    monkeypatch.setattr(table, "ROWS", [row])
    with pytest.raises(RuntimeError):
        table._check_tip_routes_are_gated()
