"""Slice K3: restart after death, and the keeper-death half of decision 8.

A restarted service makes the tip safe before any other Igor command (COM first, then the
socket as soon as Igor dials in), latches a fault, and reports Igor's real mark and fault on
READY. A dead keeper lets the object being written finish, refuses the next one and every
later write until a keeper is back in contact; Stop, park and withdraw still work.

The Igor side is the golden harness's stub with its COM door tapped, so the tap's `raw` list
is the order in which Igor received every command, COM and socket together.
"""

from __future__ import annotations

import contextlib
import socket
import subprocess
import sys

import pytest
from golden.harness import (
    _FRESH,
    _HUB_STATE,
    MODEL_FILE,
    GoldenRig,
    GoldenStub,
    StubCom,
    _clear_leftovers,
    _shallow,
    load_fixture,
    platform_for,
)
from golden.scenarios import PAD, WIRE
from hub_fields import owner

from afm_litho import clock
from afm_litho.goldens.wire_tap import TappedCom
from afm_litho.hardware import wire
from afm_litho.hardware.hardware_arc import HardwareARC
from afm_litho.keeper.protocol import AFTER_DEATH_ARG, KEEPER_PID_ENV, parse_line
from afm_litho.service import keeper_link
from afm_litho.service.host import after_death

pytestmark = pytest.mark.xdist_group("stop-port")


@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


@pytest.fixture(autouse=True)
def _no_keeper_left():
    keeper_link.forget_keeper()
    after_death.reset()
    yield
    keeper_link.forget_keeper()
    after_death.reset()
    # A writer that stopped on a broken pipe leaves what it had not read; the next test's
    # writer would read it first.
    while not keeper_link._outbox.empty():
        keeper_link._outbox.get_nowait()


def _run_catching(fn, arg, failures: list) -> None:
    try:
        fn(arg)
    except BaseException as exc:                         # noqa: BLE001 - recorded for the test
        failures.append(exc)


def _sleeper() -> subprocess.Popen:
    """A live process to stand in for a keeper."""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])


# ------------------------------------------------------- the keeper's half --

def test_the_keeper_restarts_the_service_with_after_death_and_tells_it_its_pid(tmp_path):
    from afm_litho.keeper.cli import child_specs
    from afm_litho.keeper.process import PopenHandle
    from afm_litho.keeper.supervisor import ChildSpec

    service = next(c for c in child_specs(7461, "127.0.0.1", recorder_port=1)
                   if c.name == "service")
    assert service.restart_args == (AFTER_DEATH_ARG,) == ("--after-death",)

    import os
    import time
    spec = ChildSpec("echo", [sys.executable, "-c",
                              "import os; from afm_litho.keeper.beatfile import from_env; "
                              f"b = from_env(); b.write('READY mark=0 fault=0 pid=' "
                              f"+ os.environ['{KEEPER_PID_ENV}']); b.flush()"])
    child = PopenHandle(spec)
    end = time.monotonic() + 30
    while child.poll() is None and time.monotonic() < end:
        time.sleep(0.05)
    assert parse_line(child.drain()[0]).pid == os.getpid()


def test_service_main_makes_safe_before_it_serves(monkeypatch):
    from afm_litho import service_main, site

    order = []
    monkeypatch.setattr(after_death, "boot",
                        lambda hub, last: order.append(f"make-safe {last}"))
    monkeypatch.setattr(site, "serve", lambda port, s: order.append("serve"))
    monkeypatch.setattr(site, "resolve_site", lambda port: None)
    monkeypatch.setattr(service_main, "_quiet_windows_disconnects", lambda: None)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert service_main.main(["--port", str(port), "--after-death",
                              "--last-bridge", "up"]) == 0
    assert order == ["make-safe up", "serve"]
    order.clear()
    assert service_main.main(["--port", str(port), "--after-death"]) == 0
    assert order == ["make-safe None", "serve"]
    order.clear()
    assert service_main.main(["--port", str(port)]) == 0
    assert order == ["serve"]


def test_a_service_whose_port_is_taken_stops_nothing(monkeypatch):
    from afm_litho import service_main

    called = []
    monkeypatch.setattr(after_death, "boot", lambda hub, last: called.append(hub))
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        port = holder.getsockname()[1]
        assert service_main.main(["--port", str(port), "--after-death"]) == 3
    assert called == []


def test_the_service_attaches_igor_through_connect_hardware():
    """The tests below drive `app.connect_hardware`; the service must too."""
    import inspect

    from afm_litho.service import app as app_module
    src = inspect.getsource(app_module._startup)
    assert "connect_hardware(lambda: HardwareARC.serve(port), LazyIgor)" in src


# ------------------------------------------------- a restarted hardware service --

class _ComWithNoSocket(StubCom):
    """Igor's COM door works with no socket connected: AFM_Abort sends its terminal frame
    only to a live socket (afm_litho_run.ipf, AFM_Abort), so here it goes nowhere."""

    def execute(self, command, *, check=True):
        if self.stub.link is not None:
            return super().execute(command, check=check)
        from afm_litho.hardware.igor_stub_v3 import _NullLink
        self.stub.link = _NullLink()
        try:
            return super().execute(command, check=check)
        finally:
            self.stub.link = None


@contextlib.contextmanager
def _hardware_station(hub, monkeypatch, *, stub_setup=None):
    """A hardware station as the service starts it, before Igor dials in, with the stub's
    COM door on the fail-safe thread. Yields (stub, connect), where connect() runs app.py's
    own `connect_hardware` (Python dials the stub in place of Igor dialling in) and returns
    the bound arc."""
    from afm_litho.service.instrument_model import load_model
    from afm_litho.service.host.safe_park import watchdog_for

    saved = {k: _shallow(getattr(owner(hub, k), k, None)) for k in _HUB_STATE}
    for k, v in _FRESH.items():
        setattr(owner(hub, k), k, _shallow(v))
    _clear_leftovers(hub)
    fx = load_fixture()
    stub = GoldenStub(dict(fx["values"]), microscope_id=int(fx["microscope_id"]),
                      done_delay=0.0)
    stub.crosspoint = dict(fx["crosspoint"])
    stub.values["Height"] = 70.0                       # the dead service left the tip down
    if stub_setup:
        stub_setup(stub)
    hub.host.instrument_model = load_model(platform=platform_for(stub.microscope_id),
                                      startup_cal=None, config_path=MODEL_FILE,
                                      overrides_log=None)
    monkeypatch.setattr(hub.host, "backend", "hardware")
    hub.host.hardware_arc = None
    wd = watchdog_for(hub)
    wd_com = wd.com_igor
    wd.com_igor = TappedCom(_ComWithNoSocket(stub), stub.tap, "failsafe")
    port = stub.serve_in_thread()
    stub.serve_port = port
    arcs = []

    def accept():
        arc = HardwareARC.connect("127.0.0.1", port, timeout=30.0)   # stands in for serve()
        arcs.append(arc)
        return arc

    def connect():
        from afm_litho.service.app import connect_hardware
        return connect_hardware(accept, lambda: TappedCom(StubCom(stub), stub.tap, "com"))

    try:
        yield stub, connect
    finally:
        for arc in arcs:
            arc.router.on_closed = None
            with contextlib.suppress(Exception):
                arc.close()
        wd.com_igor = wd_com
        stub.stop()
        for k, v in saved.items():
            with contextlib.suppress(Exception):
                setattr(owner(hub, k), k, v)
        _clear_leftovers(hub)


_HEIGHT_OVER_COM = '@COM fprintf 0, "%.15g", (td_ReadValue("Height"))'


@pytest.mark.time_sensitive
def test_kill_with_the_tip_down_parks_over_com_before_any_other_igor_command(hub, monkeypatch):
    with _hardware_station(hub, monkeypatch) as (stub, connect):
        assert after_death.boot(hub, "up") is True     # the service that died had Igor
        raw = stub.tap.raw
        assert raw[0] == "@SAFEPARK", raw[:4]           # COM AFM_SafePark, first of all
        assert raw[1] == _HEIGHT_OVER_COM               # then Height confirmed over COM
        assert not [r for r in raw if r.startswith("@DOSCANFUNC")]   # no vendor withdraw
        assert stub.values["Height"] == 0.0             # exactly 0.0 counts as under 1 V
        assert hub.host.last_park.verified is True and hub.host.last_park.z_out_v == 0.0
        assert hub.fault["reason"] == "after_death"
        assert hub.fault["detail"] == "service restarted after a death; tip state unknown"

        stub.tap.raw.clear()
        arc = connect()
        raw = stub.tap.raw
        assert raw[0] == "ABORT", raw[:4]                        # the socket make-safe
        assert raw[1].startswith("STARTUP ")                    # then the connect's reads
        assert stub.td_stops == 1                       # the second abort skipped td_Stop
        assert arc.read("Height") < 1.0
        assert hub.fault["reason"] == "after_death"             # stands until cleared


@pytest.mark.time_sensitive
def test_a_restart_whose_com_confirm_fails_falls_back_to_the_vendor_withdraw(hub, monkeypatch):
    def setup(stub):
        stub.safe_park_zeros = False                    # the abort leaves Height up

    with _hardware_station(hub, monkeypatch, stub_setup=setup) as (stub, _connect):
        assert after_death.boot(hub, "up") is True
        raw = stub.tap.raw
        assert raw[0] == "@SAFEPARK"
        vendor = [i for i, r in enumerate(raw) if r.startswith("@DOSCANFUNC")]
        assert len(vendor) == 1, raw                     # the vendor withdraw, over COM, once
        assert vendor[0] > raw.index(_HEIGHT_OVER_COM)
        assert hub.host.last_park.verified is True and hub.host.last_park.method == "vendor_withdraw"
        assert hub.fault["reason"] == "after_death"


@pytest.mark.time_sensitive
def test_a_restart_whose_tip_is_not_confirmed_up_latches_that_fault(hub, monkeypatch):
    def setup(stub):
        stub.safe_park_zeros = False
        stub.vendor_withdraw = lambda: None             # the vendor withdraw moves nothing

    with _hardware_station(hub, monkeypatch, stub_setup=setup) as (stub, _connect):
        assert after_death.boot(hub, "up") is True
        assert stub.tap.raw[0] == "@SAFEPARK"
        assert hub.host.last_park.verified is False and hub.host.last_park.critical is True
        assert hub.host.last_park.z_out_v == 70.0
        assert hub.fault["reason"] == "park_unverified"
        assert hub.fault["detail"] == after_death.NOT_CONFIRMED_UP


@pytest.mark.time_sensitive
def test_every_stop_before_the_new_bridge_goes_over_com(hub, client, monkeypatch):
    from afm_litho.instrument import WithdrawTip
    from afm_litho.instrument.asylum import AsylumInstrument
    from afm_litho.service import app as app_module
    from afm_litho.service import flex_dispatch
    from afm_litho.service.host.safe_park import (
        ROUTE_COM_AFTER_RESTART,
        ROUTE_HARDWARE,
        safe_park,
        stop_route,
    )
    from test_stop_port import _Line

    with _hardware_station(hub, monkeypatch) as (stub, connect):
        after_death.boot(hub, "up")
        assert stop_route(hub) == ROUTE_COM_AFTER_RESTART
        stub.tap.raw.clear()
        stub.values["Height"] = 70.0                     # down again: each stop must park

        cockpit = client.post("/api/safe_park", json={}).json()["safe_park"]
        assert _safeparks(stub) == 1, stub.tap.raw
        assert (cockpit["transport"], cockpit["verified"]) == ("com", True)
        assert cockpit["z_v"] == 0.0 and cockpit["backend"] == "hardware"

        stub.values["Height"] = 70.0
        flex = flex_dispatch._h_safe_park(hub, {}, None)
        assert _safeparks(stub) == 2 and (flex["transport"], flex["verified"]) == ("com", True)

        stub.values["Height"] = 70.0
        line = _Line(app_module._stop_port.port, "STOP k3 restarted\n")
        try:
            assert line.reply() == "sent"
            assert line.reply(40) == "confirmed"
        finally:
            line.close()
        assert _safeparks(stub) == 3, stub.tap.raw

        stub.values["Height"] = 70.0
        withdrawn = client.post("/api/withdraw", json={}).json()
        assert _safeparks(stub) == 4 and withdrawn["verified"] is True
        assert "not_connected" not in withdrawn

        stub.values["Height"] = 70.0
        res = AsylumInstrument(hub).withdraw(WithdrawTip())
        assert _safeparks(stub) == 5 and res.verified is True

        stub.values["Height"] = 70.0
        assert safe_park(hub, "sync park").verified is True
        assert _safeparks(stub) == 6
        assert not [r for r in stub.tap.raw if not r.startswith("@")]   # no socket yet

        # Once the new bridge binds, the normal routes apply.
        connect()
        assert stop_route(hub) == ROUTE_HARDWARE


@pytest.mark.time_sensitive
@pytest.mark.parametrize("last_bridge", ["down", None])
def test_a_service_that_died_without_igor_makes_nothing_safe(hub, monkeypatch, last_bridge):
    with _hardware_station(hub, monkeypatch) as (stub, connect):
        assert after_death.boot(hub, last_bridge) is False
        assert stub.tap.raw == []                       # nothing sent, COM included
        assert hub.fault is None
        connect()
        assert stub.tap.raw[0].startswith("STARTUP ")
        assert "ABORT" not in stub.tap.raw and "@SAFEPARK" not in stub.tap.raw
        assert hub.fault is None


@pytest.mark.time_sensitive
def test_a_park_before_igor_dials_in_sends_nothing_and_claims_nothing(hub, monkeypatch):
    import asyncio

    from afm_litho.service import app as app_module
    from afm_litho.service.host.safe_park import (
        NOT_CONNECTED,
        safe_park,
        safe_park_async,
        start_park,
        watchdog_for,
    )
    from test_stop_port import _Line

    with _hardware_station(hub, monkeypatch) as (stub, _connect):
        hub.host.engaged = True
        # The cockpit and FLEX park.
        res = asyncio.run_coroutine_threadsafe(safe_park_async(hub, "cold park"),
                                               hub._loop).result(30)
        assert (res["verified"], res["critical"], res["transport"]) == (False, False, None)
        assert res["backend"] == "disconnected" and res["detail"] == NOT_CONNECTED
        # The Asylum park and abort's route (the fail-safe thread), and the sync primitive.
        req = watchdog_for(hub).request("cold watchdog")
        assert req.done.wait(30) and req.result["not_connected"] is True
        assert req.result["verified"] is False and req.result["critical"] is False
        pr = safe_park(hub, "cold sync")
        assert (pr.verified, pr.critical, pr.method) == (False, False, "not_connected")
        req = start_park(hub, "cold start_park")
        assert req.done.is_set() and req.result["detail"] == NOT_CONNECTED
        # The stop port.
        line = _Line(app_module._stop_port.port, "STOP k3 cold\n")
        try:
            assert line.reply() == "sent"
            assert line.reply() == "NOT confirmed (not connected: nothing done)"
        finally:
            line.close()
        assert stub.tap.raw == []                        # nothing reached Igor
        assert hub.fault is None and hub.host.engaged is True  # nothing claimed either way


def _withdraw_by_every_route(hub, client):
    """Each withdraw route once: cockpit, FLEX, dead-man/idle cap, plugin, end-of-run.
    Returns what each answered, keyed by route."""
    import asyncio
    import queue
    from unittest import mock

    from afm_litho.instrument import WithdrawTip
    from afm_litho.instrument.asylum import AsylumInstrument
    from afm_litho.service import flex_dispatch
    from afm_litho.service.host import deadman, litho_worker

    answers = {"cockpit": client.post("/api/withdraw", json={}).json(),
               "flex": flex_dispatch._h_withdraw(hub, {}, None),
               "internal": asyncio.run_coroutine_threadsafe(deadman._withdraw(hub),
                                                            hub._loop).result(30)}
    plugin = AsylumInstrument(hub)
    answers["plugin"] = plugin.withdraw(WithdrawTip())
    plugin._make_safe("k3 cold make-safe")
    q = queue.Queue()
    answers["end_of_run"] = litho_worker._end_of_run_withdraw(q, mock.MagicMock())
    answers["end_of_run_log"] = [m.get("text", m.get("msg", m)) for m in list(q.queue)]
    return answers


@pytest.mark.time_sensitive
def test_a_withdraw_before_igor_dials_in_sends_nothing_and_claims_nothing(
        hub, client, monkeypatch):
    from afm_litho.service.host.safe_park import NOT_CONNECTED

    sent = []
    real_broadcast = hub.broadcast

    async def spy(msg, *a, **k):
        sent.append(msg)
        return await real_broadcast(msg, *a, **k)

    monkeypatch.setattr(hub, "broadcast", spy)
    with _hardware_station(hub, monkeypatch) as (stub, _connect):
        hub.host.engaged = True
        answers = _withdraw_by_every_route(hub, client)
        assert stub.tap.raw == []                        # nothing reached Igor
        assert hub.fault is None and hub.host.engaged is True  # nothing claimed
        assert not [m for m in sent if isinstance(m, dict)
                    and m.get("type") in ("withdrawn", "engaged_idle_withdraw")]
        cockpit = answers["cockpit"]
        assert cockpit["not_connected"] is True and cockpit["detail"] == NOT_CONNECTED
        assert (cockpit["verified"], cockpit["critical"]) == (False, False)
        flex = answers["flex"]
        assert (flex["verified"], flex["critical"], flex["not_connected"]) == (
            False, False, True)
        assert flex["detail"] == NOT_CONNECTED and flex["z_v"] is None
        assert answers["internal"]["not_connected"] is True
        res = answers["plugin"]
        assert res.done and res.verified is False and res.reason == NOT_CONNECTED
        assert answers["end_of_run"] is None
        assert any(NOT_CONNECTED in str(m) for m in answers["end_of_run_log"])


@pytest.mark.time_sensitive
def test_the_engaged_idle_cap_before_igor_dials_in_announces_no_withdraw(hub, monkeypatch):
    import asyncio

    from afm_litho.service.host import deadman

    sent = []
    real_broadcast = hub.broadcast

    async def spy(msg, *a, **k):
        sent.append(msg)
        return await real_broadcast(msg, *a, **k)

    monkeypatch.setattr(hub, "broadcast", spy)
    with _hardware_station(hub, monkeypatch) as (stub, _connect):
        monkeypatch.setattr(deadman, "pass_state", lambda h: "engaged")
        hub.host.engaged = True
        hub.host._engaged_since = 0.0
        act = asyncio.run_coroutine_threadsafe(deadman.engaged_idle_tick(hub, 1e9),
                                               hub._loop).result(30)
        assert act["not_connected"] is True
        assert stub.tap.raw == [] and hub.host.engaged is True
        assert not [m for m in sent if isinstance(m, dict)
                    and m.get("type") in ("withdrawn", "engaged_idle_withdraw")]


@pytest.mark.time_sensitive
def test_a_withdraw_after_igor_dials_in_reaches_igor_as_today(hub, client, monkeypatch):
    with _hardware_station(hub, monkeypatch) as (stub, connect):
        connect()
        stub.tap.raw.clear()
        hub.host.engaged = True
        res = client.post("/api/withdraw", json={}).json()
        assert "not_connected" not in res, res
        assert stub.tap.raw, "the withdraw sent nothing to Igor after dial-in"
        assert hub.host.engaged is False


@contextlib.contextmanager
def _dropped_station(hub, monkeypatch):
    """A hardware station Igor dialled into, whose bridge socket then closed from Igor's
    side with the tip engaged and nothing running (so no automatic park fires). Igor's COM
    door still works with no socket, as on the instrument. Yields the stub, its tap cleared."""
    import time

    with _hardware_station(hub, monkeypatch) as (stub, connect):
        arc = connect()
        arc.com_igor = TappedCom(_ComWithNoSocket(stub), stub.tap, "com")
        hub.host.engaged = True
        stub.drop_host_socket()
        end = time.monotonic() + 10.0
        while not hub.host.bridge_closed.is_set() and time.monotonic() < end:
            time.sleep(0.02)
        assert hub.host.bridge_closed.is_set(), "the drop never reached the hub"
        assert hub.backend_live() == "twin"              # the read that used to mislead
        stub.link = None
        stub.tap.raw.clear()
        try:
            yield stub
        finally:
            hub.host.bridge_closed.clear()
            hub.host.bridge_closed_at = None


def _safeparks(stub) -> int:
    return stub.tap.raw.count("@SAFEPARK")


@pytest.mark.time_sensitive
def test_every_stop_after_the_bridge_drops_sends_safepark_over_com(hub, client, monkeypatch):
    from afm_litho.service import app as app_module
    from afm_litho.service import flex_dispatch
    from test_stop_port import _Line

    with _dropped_station(hub, monkeypatch) as stub:
        cockpit = client.post("/api/safe_park", json={}).json()["safe_park"]
        assert _safeparks(stub) == 1, stub.tap.raw
        assert cockpit["transport"] == "com" and cockpit["backend"] == "hardware"
        assert cockpit["verified"] is False and cockpit["z_v"] is None
        assert hub.host.last_park.verified is False

        flex = flex_dispatch._h_safe_park(hub, {}, None)
        assert _safeparks(stub) == 2, stub.tap.raw
        assert (flex["transport"], flex["verified"]) == ("com", False)

        line = _Line(app_module._stop_port.port, "STOP k3 dropped\n")
        try:
            assert line.reply() == "sent"
            assert line.reply(40) == "NOT confirmed"
        finally:
            line.close()
        assert _safeparks(stub) == 3, stub.tap.raw
        assert hub.host.engaged is True                       # nothing confirmed, nothing claimed


@pytest.mark.time_sensitive
def test_a_withdraw_after_the_bridge_drops_sends_safepark_over_com(hub, client, monkeypatch):
    from afm_litho.service import flex_dispatch

    sent = []
    real_broadcast = hub.broadcast

    async def spy(msg, *a, **k):
        sent.append(msg)
        return await real_broadcast(msg, *a, **k)

    monkeypatch.setattr(hub, "broadcast", spy)
    with _dropped_station(hub, monkeypatch) as stub:
        from afm_litho.service.host.safe_park import BRIDGE_DOWN_UNCONFIRMED
        res = client.post("/api/withdraw", json={}).json()
        assert _safeparks(stub) == 1, stub.tap.raw
        assert (res["transport"], res["verified"]) == ("com", False)
        assert res["detail"] == BRIDGE_DOWN_UNCONFIRMED
        flex = flex_dispatch._h_withdraw(hub, {}, None)
        assert _safeparks(stub) == 2, stub.tap.raw
        assert flex["verified"] is False and flex["z_v"] is None
        assert hub.host.engaged is True
        assert not [m for m in sent if isinstance(m, dict) and m.get("type") == "withdrawn"]


def test_stop_route_is_decided_by_whether_igor_ever_dialled_in():
    import threading

    from afm_litho.service.host.safe_park import (
        ROUTE_COM_AFTER_RESTART, ROUTE_COM_ESCAPE, ROUTE_HARDWARE, ROUTE_NOT_CONNECTED,
        ROUTE_TWIN, stop_route)

    class _Hub:
        backend = "hardware"
        hardware_arc = None
        bridge_closed = threading.Event()

        @property
        def host(self):                # fake: host fields live on the fake itself
            return self

    h = _Hub()
    assert stop_route(h) == ROUTE_NOT_CONNECTED          # cold start
    after_death._predecessor_had_bridge = True           # --after-death --last-bridge up
    assert stop_route(h) == ROUTE_COM_AFTER_RESTART
    after_death.reset()
    h.hardware_arc = object()
    assert stop_route(h) == ROUTE_HARDWARE
    h.bridge_closed.set()
    assert stop_route(h) == ROUTE_COM_ESCAPE             # dropped: still Igor, over COM
    h.backend, h.hardware_arc = "twin", None
    assert stop_route(h) == ROUTE_TWIN


def test_the_keeper_passes_the_last_bridge_state_to_the_restart():
    from afm_litho.keeper.protocol import heartbeat_line, ready_line
    from afm_litho.keeper.supervisor import ChildSpec, Supervisor

    class Child:
        def __init__(self, pid):
            self.pid, self.code, self.lines = pid, None, []

        def poll(self):
            return self.code

        def drain(self):
            out, self.lines = self.lines, []
            return out

        def terminate(self):
            self.code = 1

    started = []

    def launch(spec, extra):
        started.append(extra)
        return Child(100 + len(started))

    spec = ChildSpec("service", ["x"], restart_args=(AFTER_DEATH_ARG,))
    keeper = Supervisor([spec], launch, out=lambda m: None, err=lambda m: None)
    keeper.start()
    child = keeper.slots[0].handle
    child.lines = [ready_line(0, 0, child.pid, "down"), heartbeat_line(child.pid, "up")]
    child.code = 1
    keeper.step()
    assert started[-1] == ("--after-death", "--last-bridge", "up")
    child = keeper.slots[0].handle                       # the restart never reports
    child.code = 1
    keeper.step()
    assert started[-1] == ("--after-death", "--last-bridge", "down")
    assert keeper.snapshot()["children"][0]["bridge"] is None


def test_the_service_says_whether_its_bridge_is_up(hub):
    assert keeper_link.bridge_state(hub) == "down"            # the twin never has Igor
    assert parse_line("HEARTBEAT pid=1 bridge=up").bridge == "up"


@pytest.mark.time_sensitive
def test_a_service_that_did_not_die_sends_no_make_safe(hub, monkeypatch):
    with _hardware_station(hub, monkeypatch) as (stub, connect):
        connect()
        assert stub.tap.raw[0].startswith("STARTUP ")
        assert "ABORT" not in stub.tap.raw and "@SAFEPARK" not in stub.tap.raw
        assert hub.fault is None


@pytest.mark.time_sensitive
@pytest.mark.parametrize("igor_fault", [False, True])
def test_ready_reports_igors_mark_and_the_fault_read_at_connect(hub, monkeypatch,
                                                              igor_fault):
    def setup(stub):
        stub.abort_stopped = True               # a make-safe nobody confirmed
        stub.abort_fault = igor_fault

    sent = []
    monkeypatch.setattr(keeper_link, "_send", sent.append)
    with wire.use_version(3), _hardware_station(hub, monkeypatch, stub_setup=setup) as (
            stub, connect):
        keeper_link.send_ready(hub)             # before Igor dials in: mark not read
        connect()
    first, bound = (parse_line(line) for line in sent)
    assert first.mark is None and first.fault == 0
    assert bound.mark == 1 and bound.fault == int(igor_fault)
    assert (first.bridge, bound.bridge) == ("down", "up")


@pytest.mark.time_sensitive
def test_a_restarted_service_reports_the_real_mark_and_fault(hub, monkeypatch):
    sent = []
    monkeypatch.setattr(keeper_link, "_send", sent.append)
    with wire.use_version(3), _hardware_station(hub, monkeypatch) as (stub, connect):
        after_death.boot(hub, "up")
        keeper_link.send_ready(hub)
        connect()
        assert stub.abort_stopped                # nobody has confirmed the make-safe
    first, bound = (parse_line(line) for line in sent)
    assert (first.mark, first.fault) == (None, 1)
    assert (bound.mark, bound.fault) == (1, 1)


def test_ready_on_the_twin_has_no_stop_mark(hub):
    mark, fault = keeper_link.read_marks(hub)
    assert mark == 0 and fault == (1 if hub.fault else 0)


# ----------------------------------------------------------- keeper death --

def test_a_dead_keeper_is_reported_once_and_refuses_until_contact_is_restored(hub, capsys):
    keeper = _sleeper()
    keeper_link.watch_keeper(keeper.pid)
    assert keeper_link.keeper_lost(hub) is None
    keeper.kill()
    keeper.wait()
    why = keeper_link.keeper_lost(hub)
    assert f"pid {keeper.pid}" in why and "Stop, park and withdraw still work" in why
    assert keeper_link.keeper_lost(hub) == why
    assert capsys.readouterr().out.count("SAFETY: the keeper") == 1
    other = _sleeper()
    try:
        keeper_link.restore_contact(other.pid, hub)
        assert keeper_link.keeper_lost(hub) is None
    finally:
        other.kill()
        other.wait()


def test_a_missed_beat_does_not_make_a_live_keeper_dead(tmp_path, monkeypatch):
    """Death is the operating system's word: a beat file that cannot be written says nothing."""
    import threading

    from afm_litho.keeper.beatfile import BeatFile

    other = _sleeper()
    try:
        keeper_link.watch_keeper(other.pid)
        blocker = tmp_path / "file"
        blocker.write_text("x")
        link = BeatFile(blocker / "beat.txt")            # its folder cannot be made
        monkeypatch.setattr(keeper_link, "_link", link)
        keeper_link._send("HEARTBEAT pid=1")
        keeper_link._outbox.put_nowait(None)
        failures = []
        writer = threading.Thread(
            target=lambda: _run_catching(keeper_link._writer, link, failures))
        writer.start()
        writer.join(timeout=5)
        assert not writer.is_alive()
        assert failures == []                            # the writer did not raise
        assert keeper_link.keeper_lost() is None
    finally:
        other.kill()
        other.wait()


def test_a_different_process_with_the_keepers_pid_does_not_keep_it_alive(hub, monkeypatch):
    """The handle opened at startup names the keeper itself; a bare pid lookup would say
    alive for whoever took the number."""
    from afm_litho.keeper import pids

    keeper = _sleeper()
    keeper_link.watch_keeper(keeper.pid)
    keeper.kill()
    keeper.wait()
    monkeypatch.setattr(pids, "pid_alive", lambda pid: True)       # the number is reused
    assert keeper_link.keeper_lost(hub) is not None


def test_a_keeper_that_cannot_be_opened_is_lost(hub, monkeypatch):
    if sys.platform != "win32":
        pytest.skip("process handles are Windows-only")
    import ctypes
    kernel32 = ctypes.windll.kernel32
    monkeypatch.setattr(kernel32, "OpenProcess", lambda *a: 0)   # the OS refuses
    keeper = _sleeper()
    try:
        keeper_link.watch_keeper(keeper.pid)                     # a live process
        assert keeper_link.keeper_lost(hub) is not None
    finally:
        keeper.kill()
        keeper.wait()


def test_the_contact_route_takes_only_a_live_keeper_pid(client, monkeypatch, tmp_path):
    from afm_litho.keeper import status
    monkeypatch.setenv("AFM_LITHO_RUNTIME", str(tmp_path))
    monkeypatch.delenv(KEEPER_PID_ENV, raising=False)
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    for body in ({}, {"pid": "12"}, {"pid": True}, {"pid": -3}):
        assert "error" in client.post("/api/keeper/contact", json=body).json()
    other = _sleeper()
    try:
        # A live process that is not a keeper lifts nothing.
        assert "is not a keeper" in client.post("/api/keeper/contact",
                                                json={"pid": other.pid}).json()["error"]
        monkeypatch.setenv(KEEPER_PID_ENV, str(done.pid))
        assert "no running process" in client.post("/api/keeper/contact",
                                                   json={"pid": done.pid}).json()["error"]
        # A restarted keeper names itself in its state file.
        status.write_state(status.state_path(), {"children": []})
        state = status.read_state(status.state_path())
        state["keeper_pid"] = other.pid
        status.state_path().write_text(__import__("json").dumps(state), encoding="utf-8")
        assert client.post("/api/keeper/contact", json={"pid": other.pid}).json() == {
            "ok": True, "pid": other.pid}
    finally:
        other.kill()
        other.wait()


def _acquire(door) -> str:
    return door.call("acquireControl", {"client": "k3", "deadman_s": 120})["token"]


def _write(door, tok, objects) -> dict:
    door.call("loadPattern", {"token": tok, "objects": [dict(o) for o in objects]})
    return door.call("startWrite", {"token": tok, "objects": "all", "setpoint_reference": "absolute", "setpoint": 0.0,
                                    "amp_gain": 25.0})


@pytest.mark.time_sensitive
def test_keeper_death_mid_run_finishes_the_object_refuses_the_next_and_stop_works(
        hub, client, monkeypatch):
    from afm_litho.service import app as app_module
    from test_stop_port import _Line

    keeper = _sleeper()
    keeper_link.watch_keeper(keeper.pid)
    with GoldenRig(hub, "keeper_death") as rig:
        stub = rig.stub
        real_dispatch = stub._dispatch
        fired = []

        def dispatch(line, link):
            # The keeper dies while the pad's first stroke is in Igor's hands.
            if line.startswith("ARMFIRE ") and not fired:
                keeper.kill()
                keeper.wait()
            if line.startswith("ARMFIRE "):
                fired.append(line)
            real_dispatch(line, link)

        monkeypatch.setattr(stub, "_dispatch", dispatch)
        door = rig.door
        tok = _acquire(door)
        run = _write(door, tok, [PAD, WIRE])
        door.wait_state("idle", "engaged", "parked")
        res = door.call("getWriteResult", {"run_id": run["run_id"]})
        assert res["aborted"] is True, res
        pad_strokes = run["segments"] - 1                # the wire is one stroke
        assert pad_strokes > 1 and len(fired) == pad_strokes   # the pad finished, no wire
        assert hub.fault is None                         # refused, not faulted
        assert hub.host.engaged is False                      # the end-of-run withdraw ran

        door.call("loadPattern", {"token": tok, "objects": [dict(WIRE)]})
        err = door.call_refused("startWrite", {"token": tok, "objects": "all",
                                               "setpoint_reference": "absolute", "setpoint": 0.0, "amp_gain": 25.0})
        assert err["data"]["reason"] == "keeper_lost", err
        assert len(fired) == pad_strokes

        # Stop, park and withdraw are never refused.
        assert door.call("safePark", {"reason": "k3 keeper dead"})["verified"] is True
        line = _Line(app_module._stop_port.port, "STOP k3 keeper dead\n")
        try:
            assert line.reply() == "sent"
            assert line.reply() == "confirmed"
        finally:
            line.close()
        # The cockpit Withdraw takes access from the script (an operator tip route).
        withdrawn = client.post("/api/withdraw", json={})
        assert withdrawn.status_code == 200 and "error" not in withdrawn.json(), withdrawn.json()
        tok = _acquire(door)

        # A keeper back in contact unblocks writes.
        other = _sleeper()
        monkeypatch.setenv(KEEPER_PID_ENV, str(other.pid))
        try:
            assert client.post("/api/keeper/contact", json={"pid": other.pid}).json()["ok"]
            again = _write(door, tok, [WIRE])
            door.wait_state("idle", "engaged", "parked")
            res = door.call("getWriteResult", {"run_id": again["run_id"]})
            assert res["aborted"] is False, res
            assert len(fired) == pad_strokes + 1
        finally:
            other.kill()
            other.wait()
        door.call("releaseControl", {"token": tok})
        rig.quiesce()



class _ScriptedCom:
    """A COM handle whose Height reads come from a list; records every command."""

    def __init__(self, heights):
        self.heights, self.sent = list(heights), []

    def execute(self, command, *, check=True):
        from afm_litho.hardware.igor import IgorResult
        self.sent.append(command)
        if "td_ReadValue" in command:
            return IgorResult(0, "", "", repr(self.heights.pop(0)))
        return IgorResult(0, "", "", "")


class _BareHub:
    backend = "hardware"
    hardware_arc = None
    fault = None
    last_park = None

    def log(self, msg):
        pass

    @property
    def host(self):                # fake: host fields live on the fake itself
        return self


def test_a_height_of_exactly_zero_over_com_confirms_the_park():
    hub = _BareHub()
    com = _ScriptedCom([0.0])
    out = after_death.park_over_com(hub, "zero", com)
    assert (out["verified"], out["critical"], out["z_v"]) == (True, False, 0.0)
    assert com.sent[0] == "AFM_SafePark()" and len(com.sent) == 2
    assert hub.fault is None


class _StuckHighCom(_ScriptedCom):
    """Height never comes down: every read answers 66 V."""

    def __init__(self):
        super().__init__([])

    def execute(self, command, *, check=True):
        from afm_litho.hardware.igor import IgorResult
        if "td_ReadValue" in command:
            self.sent.append(command)
            # Cap on a runaway poll (one that never sleeps). read_height_over_com swallows
            # this raise as a failed read, which ends the poll; the sleep asserts then fail.
            if sum("td_ReadValue" in c for c in self.sent) > 100:
                raise AssertionError("poll did not stop")
            return IgorResult(0, "", "", "66.0")
        return super().execute(command, check=check)


def _poll_sleeps():
    """The Height poll's recorded sleeps, in seconds."""
    return [w.seconds for w in clock.waits() if w.kind == "sleep"]


def test_a_failed_com_confirm_latches_the_not_confirmed_up_fault():
    hub = _BareHub()
    com = _StuckHighCom()
    with clock.using(clock.TESTING):
        out = after_death.park_over_com(hub, "stuck", com)
        sleeps = _poll_sleeps()
    assert (out["verified"], out["critical"], out["z_v"]) == (False, True, 66.0)
    # Two 2.5 s windows (before and after the vendor withdraw), 10 sleeps of 0.25 s each.
    assert sleeps == [0.25] * 20
    assert 'DoScanFunc("StopEngageButton")' in com.sent
    assert hub.fault["detail"] == after_death.NOT_CONFIRMED_UP


def test_a_cold_start_counts_as_not_connected():
    from afm_litho.service.host.safe_park import before_igor_dials_in, previously_connected
    hub = _BareHub()
    assert before_igor_dials_in(hub) and not previously_connected(hub)
    after_death._predecessor_had_bridge = True
    try:
        assert previously_connected(hub) and not before_igor_dials_in(hub)
        hub.host.backend = "twin"
        assert not previously_connected(hub)          # only a hardware station
    finally:
        after_death.reset()


def test_a_critical_park_with_height_under_1_v_does_not_say_otherwise():
    from afm_litho.service.host.safe_park import _outcome_line
    line = _outcome_line("latched", {"critical": True, "verified": False, "z_v": 0.0})
    assert "did NOT come under 1 V" not in line and "Height = 0.000 V" in line
    line = _outcome_line("up", {"critical": True, "verified": False, "z_v": 66.0})
    assert "did NOT come under 1 V" in line


def test_a_failed_com_read_ends_the_poll_at_once():
    class _BrokenRead(_ScriptedCom):
        def execute(self, command, *, check=True):
            if "td_ReadValue" in command:
                self.sent.append(command)
                raise RuntimeError("COM is down")
            return super().execute(command, check=check)

    hub = _BareHub()
    com = _BrokenRead([])
    out = after_death.park_over_com(hub, "broken read", com)
    assert (out["verified"], out["z_v"]) == (False, None)
    reads = [c for c in com.sent if "td_ReadValue" in c]
    assert len(reads) == 2                    # one per confirm, not a 2.5 s poll each


def test_no_vendor_withdraw_once_a_fault_is_latched():
    hub = _BareHub()
    hub.fault = {"reason": "after_death"}
    com = _StuckHighCom()
    with clock.using(clock.TESTING):
        out = after_death.park_over_com(hub, "latched", com)
        sleeps = _poll_sleeps()
    assert sleeps == [0.25] * 10                      # the one 2.5 s window, no second poll
    assert out["verified"] is False
    assert com.sent[0] == "AFM_SafePark()"
    assert 'DoScanFunc("StopEngageButton")' not in com.sent


def test_the_height_poll_waits_out_the_whole_settle_window_while_height_is_high():
    from afm_litho.service.host.safe_park import _VERIFY_POLL_S, _VERIFY_SETTLE_S
    com = _StuckHighCom()
    with clock.using(clock.TESTING):
        start = clock.now()
        height = after_death._poll_height_over_com(com)
        elapsed = clock.now() - start
        sleeps = _poll_sleeps()
    assert height == 66.0
    assert sleeps == [_VERIFY_POLL_S] * round(_VERIFY_SETTLE_S / _VERIFY_POLL_S)
    assert elapsed == pytest.approx(_VERIFY_SETTLE_S)
    assert len([c for c in com.sent if "td_ReadValue" in c]) == len(sleeps) + 1


def test_the_height_poll_stops_as_soon_as_height_is_under_1_v():
    com = _ScriptedCom([66.0, 66.0, 0.2])
    with clock.using(clock.TESTING):
        height = after_death._poll_height_over_com(com)
        sleeps = _poll_sleeps()
    assert height == 0.2 and len(sleeps) == 2
