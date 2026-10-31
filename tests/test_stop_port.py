"""The host's stop port (slice K1): `STOP <who> <why>` on loopback parks the tip through the
fail-safe thread, answers `sent` at once and then `confirmed` or `NOT confirmed`.

The rig tests drive a real park on the Igor stub (the golden harness) and judge Igor's
commands against the frozen safe_park golden.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import socket
import threading
import time

import pytest
from golden.harness import HERE, MFP3D_MICROSCOPE_ID, GoldenRig
from hub_fields import owner

from afm_litho.hardware.failsafe_thread import ParkRequest
from afm_litho.service.host import stop_port
from afm_litho.service.host.stop_port import (
    CONFIRMED,
    DEFAULT_PORT,
    NOT_CONFIRMED,
    SENT,
    StopPort,
    parse_stop,
)

pytestmark = pytest.mark.xdist_group("stop-port")


# ----------------------------------------------------------------- the line --

class _Line:
    """One client connection to a stop port, reading reply lines."""

    def __init__(self, port: int, text: bytes | str | None = None):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=5.0)
        self.buf = b""
        if text is not None:
            self.send(text)

    def send(self, text: bytes | str) -> None:
        self.sock.sendall(text if isinstance(text, bytes) else text.encode("ascii"))

    def reply(self, timeout: float = 20.0) -> str | None:
        """The next reply line, or None when none arrives within `timeout` (or EOF)."""
        end = time.monotonic() + timeout
        while b"\n" not in self.buf:
            left = end - time.monotonic()
            if left <= 0:
                return None
            self.sock.settimeout(left)
            try:
                chunk = self.sock.recv(256)
            except TimeoutError:
                return None
            if not chunk:
                return None
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return line.decode("ascii")

    def close(self) -> None:
        self.sock.close()


class _FakeWatchdog:
    """Stands in for the fail-safe thread: records each request; the test completes it."""

    alive = True

    def __init__(self):
        self.requests: list[ParkRequest] = []
        self.queued = threading.Event()

    def request(self, reason, *, report=False):
        req = ParkRequest(reason, report=report)
        self.requests.append(req)
        self.queued.set()
        return req


class _FakeHub:
    """A hardware station with its bridge bound and a live fail-safe thread."""

    backend = "hardware"

    def __init__(self):
        self._failsafe = _FakeWatchdog()
        self._safe_park_now = threading.Event()
        self._safe_park_reason = None
        self.hardware_arc = object()          # not the twin's SoftwareARC

    def backend_live(self):
        return "hardware"

    @property
    def host(self):                # fake: host fields live on the fake itself
        return self


def _finish(req: ParkRequest, verified: bool) -> None:
    """Complete a request as the fail-safe thread does: result, future, then done."""
    req.result = {"verified": verified, "critical": not verified, "transport": "com",
                  "z_v": 0.0 if verified else 60.0}
    req.future.set_result(req.result)
    req.done.set()


@pytest.fixture
def fake_port():
    hub = _FakeHub()
    sp = StopPort(hub, 0).start()
    yield hub, sp
    sp.close()


def test_the_line_format():
    assert parse_stop("STOP keeper operator pressed S") == ("keeper", "operator pressed S")
    for bad in ("", "STOP", "STOP keeper", "STOP keeper  ", "stop keeper why",
                "HALT keeper why", "STOP kéeper why", "STOP keeper why\x00"):
        with pytest.raises(ValueError):
            parse_stop(bad)


@pytest.mark.time_sensitive
@pytest.mark.parametrize("line", [b"\n", b"STOP\n", b"STOP keeper\n", b"HALT keeper why\n",
                                  b"stop keeper why\n", b"STOP k\xc3\xa9eper why\n",
                                  b"STOP keeper " + b"x" * 600 + b"\n"])
def test_a_malformed_line_is_answered_with_an_error_and_parks_nothing(fake_port, line):
    hub, sp = fake_port
    c = _Line(sp.port, line)
    try:
        reply = c.reply()
        assert reply is not None and reply.startswith("error "), reply
        assert c.reply(timeout=0.5) is None          # nothing more: no `sent`
    finally:
        c.close()
    assert hub.host._failsafe.requests == []
    assert not hub.host._safe_park_now.is_set()


@pytest.mark.time_sensitive
@pytest.mark.parametrize("verified, outcome", [(True, CONFIRMED), (False, NOT_CONFIRMED)])
def test_sent_at_once_then_the_confirm_outcome(fake_port, verified, outcome):
    hub, sp = fake_port
    c = _Line(sp.port, "STOP test the confirm\n")
    try:
        assert c.reply() == SENT
        assert hub.host._failsafe.queued.wait(5)
        (req,) = hub.host._failsafe.requests
        assert req.reason == "STOP from test: the confirm"
        assert c.reply(timeout=0.3) is None          # no outcome before the park reports
        _finish(req, verified)
        assert c.reply() == outcome
    finally:
        c.close()


@pytest.mark.time_sensitive
def test_a_stop_ended_by_the_sender_closing_parks_at_once(fake_port):
    hub, sp = fake_port
    c = _Line(sp.port, "STOP test cut off")
    t0 = time.monotonic()
    c.sock.shutdown(socket.SHUT_WR)
    try:
        assert c.reply() == SENT
        assert time.monotonic() - t0 < 1.0       # not the 5 s read timeout
        assert hub.host._failsafe.requests[0].reason == "STOP from test: cut off"
        _finish(hub.host._failsafe.requests[0], True)
        assert c.reply() == CONFIRMED
    finally:
        c.close()


def test_it_binds_to_loopback_only(fake_port):
    _, sp = fake_port
    assert sp._srv.getsockname()[0] == "127.0.0.1"
    try:
        addrs = {a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None,
                                                       socket.AF_INET)}
    except OSError:
        addrs = set()
    for addr in addrs - {"127.0.0.1"}:
        with pytest.raises(OSError):
            socket.create_connection((addr, sp.port), timeout=1.0).close()


@pytest.mark.time_sensitive
def test_a_dead_fail_safe_thread_is_replaced_before_sent(fake_port, monkeypatch):
    sp_mod = importlib.import_module("afm_litho.service.host.safe_park")
    hub, sp = fake_port
    hub.host._failsafe.alive = False
    fresh = _FakeWatchdog()
    monkeypatch.setattr(sp_mod, "_build_watchdog", lambda _hub, _cls: fresh)
    c = _Line(sp.port, "STOP test dead thread\n")
    try:
        assert c.reply() == SENT
        assert hub.host._failsafe.requests == []       # never queued to the dead thread
        assert fresh.queued.wait(5)
        _finish(fresh.requests[0], True)
        assert c.reply() == CONFIRMED
    finally:
        c.close()


@pytest.mark.time_sensitive
def test_with_no_fail_safe_thread_the_retract_is_fired_directly_and_not_confirmed(
        fake_port, monkeypatch):
    sp_mod = importlib.import_module("afm_litho.service.host.safe_park")
    hub, sp = fake_port
    hub.host._failsafe.alive = False
    fired = []

    def _no_thread(_hub, _cls):
        raise RuntimeError("no thread")

    monkeypatch.setattr(sp_mod, "_build_watchdog", _no_thread)
    monkeypatch.setattr(sp_mod, "_park_on_bridge_drop",
                        lambda _hub, reason, log: fired.append(reason))
    c = _Line(sp.port, "STOP test no thread\n")
    try:
        assert c.reply() == SENT
        assert c.reply() == NOT_CONFIRMED
    finally:
        c.close()
    assert fired == ["STOP from test: no thread"]


def test_the_port_comes_from_the_env_with_a_fixed_default(monkeypatch):
    monkeypatch.delenv(stop_port.PORT_ENV, raising=False)
    assert stop_port.port_from_env() == DEFAULT_PORT == 9210
    assert DEFAULT_PORT not in (7461, 7462, 7463, 29180, 29181, 9193, 9200)
    monkeypatch.setenv(stop_port.PORT_ENV, "9311")
    assert stop_port.port_from_env() == 9311


def test_the_service_starts_its_stop_port_on_loopback(client):
    from afm_litho.service import app as app_module
    sp = app_module._stop_port
    assert sp is not None and sp.port
    assert sp._srv.getsockname()[0] == "127.0.0.1"


def test_a_taken_port_is_reported_and_the_service_goes_on(capsys):
    holder = StopPort(_FakeHub(), 0).start()
    try:
        assert stop_port.start(_FakeHub(), holder.port) is None
        assert "NOT listening" in capsys.readouterr().out
    finally:
        holder.close()


# ------------------------------------------------------- a real park on the stub --

@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


@pytest.fixture
def rig(hub):
    with GoldenRig(hub, "safe_park") as r:
        sp = StopPort(hub, 0).start()
        try:
            yield r, sp
        finally:
            sp.close()


def _gate_the_abort(stub):
    """Hold Igor inside AFM_SafePark (the COM call) until the test opens the gate."""
    entered, gate = threading.Event(), threading.Event()
    real = stub.run_unrecorded

    def held(line):
        if line == "ABORT":
            entered.set()
            gate.wait(30)
        real(line)

    stub.run_unrecorded = held
    return entered, gate


def _frozen(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


@pytest.fixture
def twin_hub(hub, monkeypatch):
    """The twin as the service leaves it, with the park bookkeeping put back afterwards."""
    from afm_litho.service.hub import real_bridge
    assert hub.host.backend != "hardware" and real_bridge(hub) is None
    for name in ("_safe_park", "last_park", "fault", "engaged"):
        monkeypatch.setattr(owner(hub, name), name, getattr(owner(hub, name), name))
    hub.host._safe_park = dict(hub.host._safe_park)
    return hub


@pytest.mark.time_sensitive
def test_on_the_twin_a_stop_ends_the_sim_scan_as_the_cockpit_abort_does(twin_hub):
    from afm_litho.service import app as app_module
    from afm_litho.service.passes import ScanConfig
    hub = twin_hub
    st = hub.host.inst.start_scan(ScanConfig())
    assert st.running
    c = _Line(app_module._stop_port.port, "STOP test twin scan\n")
    try:
        assert c.reply() == SENT
        assert c.reply() == CONFIRMED
    finally:
        c.close()
    assert hub.host.inst.scan.running is False and hub.host.inst.scan.aborted is True
    assert hub.host.last_park.reason == "STOP from test: twin scan"


@pytest.mark.time_sensitive
def test_with_no_event_loop_yet_a_confirmed_stop_still_clears_engaged_and_is_recorded(
        twin_hub, monkeypatch):
    hub = twin_hub
    monkeypatch.setattr(hub, "_loop", None)       # the port starts before app.py sets it
    hub.host.engaged = True
    count = hub.host._safe_park.get("count", 0)
    sp = StopPort(hub, 0).start()
    try:
        c = _Line(sp.port, "STOP test no loop\n")
        assert c.reply() == SENT
        assert c.reply() == CONFIRMED
        c.close()
    finally:
        sp.close()
    assert hub.host.engaged is False
    assert hub.host._safe_park["count"] == count + 1
    assert hub.host._safe_park["last_reason"] == "STOP from test: no loop"
    assert hub.host._safe_park["verified"] is True and hub.host._safe_park["transport"] == "twin"


@pytest.mark.time_sensitive
@pytest.mark.parametrize("arc", ["none", "software_arc"])
def test_a_hardware_station_before_igor_dials_in_is_never_confirmed(twin_hub, monkeypatch,
                                                                    arc):
    from afm_litho.service import app as app_module
    hub = twin_hub
    monkeypatch.setattr(hub.host, "backend", "hardware")
    monkeypatch.setattr(hub.host, "hardware_arc", None if arc == "none" else hub.host.arc)
    c = _Line(app_module._stop_port.port, "STOP keeper before bind\n")
    try:
        assert c.reply() == SENT
        assert c.reply() == stop_port.NO_LINK == "NOT confirmed (not connected: nothing done)"
    finally:
        c.close()


@pytest.mark.time_sensitive
@pytest.mark.parametrize("mfp", [False, True], ids=["cypher", "mfp3d"])
def test_a_stop_line_commands_exactly_the_park_golden(hub, mfp):
    with GoldenRig(hub, "safe_park",
                   microscope_id=MFP3D_MICROSCOPE_ID if mfp else None) as rig:
        sp = StopPort(hub, 0).start()
        try:
            c = _Line(sp.port, "STOP golden stop port\n")
            assert c.reply() == SENT
            assert c.reply() == CONFIRMED
            c.close()
        finally:
            sp.close()
        rig.quiesce()
    if mfp:
        frozen = _frozen("safe_park.mfp3d.golden.json")
        assert rig.golden_wire("flex")["wire"] == frozen["wire"]
    else:
        frozen = _frozen("safe_park.golden.json")
        now = rig.golden("flex")
        assert now["igor"] == frozen["igor"]
        assert now["wire"] == frozen["wire"]


@pytest.mark.time_sensitive
def test_it_parks_and_confirms_while_the_event_loop_is_blocked(rig, hub):
    r, sp = rig
    blocked, release = threading.Event(), threading.Event()

    async def _block_the_loop():
        blocked.set()
        release.wait(60)                 # holds the loop thread itself

    asyncio.run_coroutine_threadsafe(_block_the_loop(), hub._loop)
    try:
        assert blocked.wait(5)
        c = _Line(sp.port, "STOP test loop is blocked\n")
        assert c.reply() == SENT
        assert c.reply() == CONFIRMED
        assert not release.is_set()      # the loop never ran in between
        assert r.stub.td_stops == 1 and r.stub.abort_confirms == 1
        c.close()
    finally:
        release.set()
    r.quiesce()


@pytest.mark.time_sensitive
def test_sent_arrives_before_the_park_finishes(rig):
    r, sp = rig
    entered, gate = _gate_the_abort(r.stub)
    c = _Line(sp.port, "STOP test early sent\n")
    try:
        assert c.reply() == SENT
        assert entered.wait(10)          # Igor is inside AFM_SafePark, held
        assert c.reply(timeout=0.5) is None
        assert r.stub.abort_confirms == 0
        gate.set()
        assert c.reply() == CONFIRMED
    finally:
        gate.set()
        c.close()
    r.quiesce()


@pytest.mark.time_sensitive
def test_two_stops_together_are_both_answered_and_neither_waits(rig):
    r, sp = rig
    entered, gate = _gate_the_abort(r.stub)
    first = _Line(sp.port, "STOP keeper first\n")
    second = None
    try:
        assert first.reply() == SENT
        assert entered.wait(10)          # the first park is running, held in Igor
        second = _Line(sp.port, "STOP cockpit second\n")
        assert second.reply(timeout=5) == SENT
        gate.set()
        assert first.reply() == CONFIRMED
        assert second.reply() == CONFIRMED
    finally:
        gate.set()
        first.close()
        if second is not None:
            second.close()
    r.quiesce()
    # One park each at most (the second may coalesce into the first), each confirmed.
    assert 1 <= r.stub.safe_parks <= 2
    assert r.stub.abort_confirms == r.stub.safe_parks


@pytest.mark.time_sensitive
def test_not_confirmed_when_height_never_comes_under_1_v(rig, hub):
    r, sp = rig
    r.inst.set_value("ZSensor", 3.0)
    r.inst.set_value("Height", 60.0)
    r.inst.set_value("Output.Z", 60.0)
    r.inst.park_leaves_tip_down()
    r.stub.vendor_withdraw = lambda: None    # the vendor Withdraw fails too
    c = _Line(sp.port, "STOP test tip stays down\n")
    try:
        assert c.reply() == SENT
        assert c.reply(timeout=30) == NOT_CONFIRMED
    finally:
        c.close()
    assert hub.fault is not None and hub.fault["reason"] == "park_unverified"
    assert r.stub.abort_confirms == 0


@pytest.mark.time_sensitive
def test_a_stop_still_parks_and_answers_while_the_keeper_pipe_is_never_read(rig, hub,
                                                                            monkeypatch):
    """A suspended keeper stops reading the pipe: the heartbeat writer blocks, and the
    event loop and the stop port go on working."""
    from afm_litho.service import keeper_link

    r, sp = rig
    release = threading.Event()

    class NeverRead:
        def write(self, _):
            release.wait(60)

        def flush(self):
            pass

    async def _start():
        keeper_link.start(hub)

    async def _stop():
        keeper_link.stop()

    monkeypatch.setattr(keeper_link, "_link", NeverRead())
    asyncio.run_coroutine_threadsafe(_start(), hub._loop).result(5)
    try:
        for _ in range(5000):
            keeper_link._send("HEARTBEAT pid=1")
        c = _Line(sp.port, "STOP keeper pipe is full\n")
        try:
            assert c.reply() == SENT
            assert c.reply() == CONFIRMED
        finally:
            c.close()
        assert r.stub.td_stops == 1 and r.stub.abort_confirms == 1
    finally:
        release.set()
        asyncio.run_coroutine_threadsafe(_stop(), hub._loop).result(5)
    r.quiesce()
