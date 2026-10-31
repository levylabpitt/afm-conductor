"""The stop rows (slice 2.7): abort, safePark, withdraw, abortWrite, scan/abort.

A stop never waits on the access holder or the journal, a held control lock is not in its way,
and the browser Withdraw (F46) takes control in memory, runs, and only then tells the browser.
"""

from __future__ import annotations

import asyncio
import importlib
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from afm_litho.service import access
from afm_litho.service import flex_dispatch as fd
from afm_litho.service import journal
from afm_litho.service.host import litho_cmds, scan_cmds
from afm_litho.service.hub import hub
from afm_litho.service.requests import flex_door, http_door, table
from afm_litho.service.requests.table import Http, Row
from afm_litho.service.routers import scan as scan_router

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


class BlockedJournal:
    """A journal whose note never returns until released; it remembers what had run by then."""

    def __init__(self, ran: list):
        self.ran = ran
        self.at_first_note: list | None = None
        self.release = threading.Event()

    def note(self, src, what, /, **fields):
        if self.at_first_note is None:
            self.at_first_note = list(self.ran)
        self.release.wait(10)


@pytest.fixture
def stopped_world(monkeypatch):
    """The access holder and the journal both blocked, and the control lock held elsewhere.
    Returns the list the stand-in stop actions append to, and the journal."""
    ran: list = []
    journal_ = BlockedJournal(ran)
    monkeypatch.setattr(journal, "_running", journal_)

    def forbidden(*a, **k):
        raise AssertionError("a stop touched the access holder")

    for name in ("check", "refresh", "operator_tip_action", "operator_takes",
                 "script_requests", "release"):
        monkeypatch.setattr(access, name, forbidden)
    monkeypatch.setattr(hub, "touch_control", forbidden)

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
    monkeypatch.setattr(scan_router, "request_park", parked)    # for routers.scan.do_withdraw

    async def abort_write(h):
        ran.append("abortWrite")
        return {"backend": "twin", "safe_park": {}}

    monkeypatch.setattr(litho_cmds, "abort_write", abort_write)

    async def withdraw(h, body):
        ran.append("withdraw")
        return {"z_v": 0.0}

    monkeypatch.setattr(scan_cmds, "withdraw", withdraw)
    holder_done = threading.Event()
    locked = threading.Event()

    def hold_lock():
        with hub._control_lock:
            locked.set()
            holder_done.wait(30)

    t = threading.Thread(target=hold_lock, daemon=True)
    t.start()
    assert locked.wait(5)
    yield ran, journal_
    journal_.release.set()
    holder_done.set()
    t.join(5)


def _run_blocked(call, ran, journal_, expect):
    """Run `call` in a thread; the stop's action must finish while the journal is stuck."""
    done = threading.Event()
    out: list = []

    def go():
        out.append(call())
        done.set()

    threading.Thread(target=go, daemon=True).start()
    deadline = time.time() + 10
    while expect not in ran and time.time() < deadline:
        time.sleep(0.01)
    assert expect in ran, "the stop did not run while the journal and access were blocked"
    journal_.release.set()
    assert done.wait(10)
    assert expect in journal_.at_first_note, "the journal was noted before the stop ran"
    return out[0]


@pytest.mark.parametrize("verb,params,expect", [
    ("abort", {}, "park"), ("safePark", {}, "park"),
    ("abortWrite", {}, "abortWrite"), ("withdraw", {}, "withdraw")])
def test_a_flex_stop_runs_with_access_journal_and_lock_blocked(loop, stopped_world, verb,
                                                               params, expect):
    ran, journal_ = stopped_world
    reply = _run_blocked(lambda: flex_door.handle(hub, verb, params, role="command", ctx=None),
                         ran, journal_, expect)
    assert reply["ok"] is True


def _app():
    app = FastAPI()
    http_door.install(app)
    return app


@pytest.mark.parametrize("path,expect", [
    ("/api/safe_park", "park"), ("/api/litho/abort", "abortWrite"),
    ("/api/scan/abort", None)])
def test_a_browser_stop_runs_with_access_journal_and_lock_blocked(stopped_world, path, expect):
    ran, journal_ = stopped_world
    with TestClient(_app()) as c:
        out: list = []
        done = threading.Event()

        def go():
            out.append(c.post(path, json={}))
            done.set()

        threading.Thread(target=go, daemon=True).start()
        deadline = time.time() + 10
        while journal_.at_first_note is None and time.time() < deadline:
            time.sleep(0.01)
        # The journal was reached only after the handler: the stop had run before any note.
        assert journal_.at_first_note is not None, "the stop never reached its journal note"
        if expect:
            assert expect in journal_.at_first_note
        journal_.release.set()
        assert done.wait(10)
        assert out[0].status_code == 200


def test_the_deadman_is_fed_where_base_fed_it(loop, monkeypatch):
    # Base: the door touches once before the handler; abort touches again before it parks.
    order: list = []

    async def parked(h, reason, *a, **k):
        order.append("park")
        return {}

    monkeypatch.setattr(importlib.import_module("afm_litho.service.host.safe_park"),
                        "request_park", parked)
    monkeypatch.setattr(hub, "touch_control", lambda tok: order.append("touch") or True)
    flex_door.handle(hub, "abort", {"token": "t"}, role="command", ctx=None)
    assert order == ["touch", "touch", "park"]
    order.clear()
    monkeypatch.setattr(fd, "_h_safe_park", lambda *a, **k: order.append("stop") or {"ok": True})
    flex_door.handle(hub, "safePark", {"token": "t"}, role="command", ctx=None)
    assert order == ["touch", "stop"]


def test_the_scan_abort_row_reaches_stop_scan_on_the_process_hub(monkeypatch):
    seen: list = []

    async def stop_scan(h, body=None):
        seen.append((h, body))
        return {"ok": True, "withdraw": True}

    monkeypatch.setattr(scan_cmds, "stop_scan", stop_scan)
    with TestClient(_app()) as c:
        reply = c.post("/api/scan/abort", json={"withdraw": True})
    assert reply.status_code == 200 and reply.json() == {"ok": True, "withdraw": True}
    assert seen == [(hub, {"withdraw": True})]


@pytest.mark.parametrize("path", ["/api/safe_park", "/api/withdraw", "/api/scan/abort",
                                  "/api/litho/abort"])
@pytest.mark.parametrize("raw,headers", [(b"{bad", JSON), (b"[]", JSON), (b'"s"', JSON),
                                         (b"not json", {})])
def test_no_stop_is_refused_for_its_body(monkeypatch, path, raw, headers):
    ran: list = []

    async def stop(*a, **k):
        ran.append(path)
        return {"backend": "twin"}

    async def parked(h, reason, *a, **k):
        ran.append(path)
        return {}

    monkeypatch.setattr(scan_router, "request_park", parked)    # for routers.scan.do_withdraw
    monkeypatch.setattr(importlib.import_module("afm_litho.service.host.safe_park"),
                        "request_park", parked)
    monkeypatch.setattr(scan_cmds, "withdraw", stop)
    monkeypatch.setattr(litho_cmds, "abort_write", stop)
    with TestClient(_app()) as c:
        reply = c.post(path, content=raw, headers=headers)
    assert reply.status_code == 200, reply.text
    assert path in ran or path == "/api/scan/abort"


# --- F46: the browser Withdraw --------------------------------------------------------------


class StalledWindow:
    """A browser window that never reads: its send never returns."""

    async def send_text(self, data):
        await asyncio.sleep(3600)


def test_withdraw_takes_in_memory_runs_then_announces(monkeypatch):
    order: list = []
    seen: dict = {}

    async def fake_withdraw(h, body):
        order.append("withdraw")
        seen["control_at_withdraw"] = hub.control

        return {"z_v": 0.0}

    async def announce(self, prev, route):
        order.append("announce")
        seen["prev"], seen["route"] = prev, route

    monkeypatch.setattr(scan_cmds, "withdraw", fake_withdraw)
    monkeypatch.setattr(type(hub), "announce_revoked", announce)
    rec = access.script_requests(hub, "lab-script")
    with TestClient(_app()) as c:
        assert c.post("/api/withdraw", json={}).status_code == 200
        for _ in range(100):
            if "announce" in order:
                break
            time.sleep(0.02)
    assert order == ["withdraw", "announce"]
    assert seen["control_at_withdraw"] is None          # the script had already lost control
    assert seen["prev"].token == rec.token and seen["route"] == "/api/withdraw"


def test_a_stalled_browser_window_cannot_hold_the_withdraw_up(monkeypatch):
    order: list = []

    async def fake_withdraw(h, body):
        order.append("withdraw")
        return {"z_v": 0.0}

    monkeypatch.setattr(scan_cmds, "withdraw", fake_withdraw)
    window = StalledWindow()
    hub.clients.add(window)
    try:
        access.script_requests(hub, "lab-script")
        with TestClient(_app()) as c:
            started = time.time()
            reply = c.post("/api/withdraw", json={})
            elapsed = time.time() - started
            time.sleep(0.2)                              # the notice is now stuck in its send
            assert reply.status_code == 200 and order == ["withdraw"]
            assert elapsed < 3, "the withdraw waited on the stalled window"
            assert hub.control is None
            assert len(hub._control_bg_tasks) == 1       # the notice is pending, not awaited
    finally:
        hub.clients.discard(window)
        hub._control_bg_tasks.clear()


def test_the_previous_holder_is_told_it_was_revoked_after_the_withdraw(monkeypatch):
    lines: list = []
    announced = threading.Event()

    async def fake_withdraw(h, body):
        lines.append("withdraw")
        return {}

    async def log(self, text):
        lines.append(text)

    async def broadcast_control(self):
        lines.append("control frame")
        if any("REVOKED" in x for x in lines):   # other tasks may broadcast earlier
            announced.set()

    monkeypatch.setattr(scan_cmds, "withdraw", fake_withdraw)
    # On the instance: another test leaves a bound `hub.log` there, which hides a class patch.
    monkeypatch.setattr(hub, "log", lambda text: log(hub, text))
    monkeypatch.setattr(hub, "broadcast_control", lambda: broadcast_control(hub))
    access.script_requests(hub, "lab-script")
    with TestClient(_app()) as c:
        c.post("/api/withdraw", json={})
        assert announced.wait(30), "the revoke notice never reached the control frame"
    notice = next(x for x in lines if "REVOKED" in x)
    assert lines[0] == "withdraw" and lines[-1] == "control frame"
    assert lines.index(notice) > 0
    assert "lab-script" in notice and "/api/withdraw" in notice


def test_a_withdraw_with_no_script_holding_announces_nothing(monkeypatch):
    calls: list = []

    async def fake_withdraw(h, body):
        return {}

    async def announce(self, prev, route):
        calls.append(route)

    monkeypatch.setattr(scan_cmds, "withdraw", fake_withdraw)
    monkeypatch.setattr(type(hub), "announce_revoked", announce)
    with TestClient(_app()) as c:
        assert c.post("/api/withdraw", json={}).status_code == 200
        time.sleep(0.2)
    assert calls == []


def test_abort_routes_leave_a_script_holding_control(monkeypatch):
    async def parked(h, reason, *a, **k):
        return {}

    monkeypatch.setattr(scan_router, "request_park", parked)    # for routers.scan.do_withdraw
    monkeypatch.setattr(importlib.import_module("afm_litho.service.host.safe_park"),
                        "request_park", parked)
    rec = access.script_requests(hub, "lab-script")
    with TestClient(_app()) as c:
        for path in ("/api/safe_park", "/api/scan/abort", "/api/litho/abort"):
            assert c.post(path, json={}).status_code == 200
            assert hub.control is not None and hub.control.token == rec.token


# --- the table's assert ---------------------------------------------------------------------


def _row_on_withdraw(control, gate):
    return Row(name="t", kind="command", control=control, handler=lambda r: 1,
               http=Http(method="POST", path="/api/withdraw", intake=lambda call: None,
                         gate=gate))


@pytest.mark.parametrize("control,gate,ok", [
    ("stop", "operator_stop", True), ("stop", "operator_tip", False),
    ("stop", "stop", False), ("tip", "operator_tip", True), ("tip", "operator_stop", False)])
def test_withdraw_is_the_documented_stop_exception_to_the_tip_gate(monkeypatch, control,
                                                                   gate, ok):
    monkeypatch.setattr(table, "ROWS", [_row_on_withdraw(control, gate)])
    if ok:
        table._check_tip_routes_are_gated()
    else:
        with pytest.raises(RuntimeError):
            table._check_tip_routes_are_gated()


def test_the_four_stops_and_the_scan_abort_are_stop_rows():
    for verb in ("abort", "safePark", "withdraw", "abortWrite"):
        assert table.flex_row(verb).control == "stop"
        assert table.flex_row(verb).flex.gate == "stop"
    gates = {r.http.path: r.http.gate for r in table.http_rows() if r.control == "stop"}
    assert gates == {"/api/safe_park": "stop", "/api/withdraw": "operator_stop",
                     "/api/litho/abort": "stop", "/api/scan/abort": "stop"}
