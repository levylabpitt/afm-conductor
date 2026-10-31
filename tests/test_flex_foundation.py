"""FLEX foundation (milestone 1): the stage-0 fail-safe + truthful-status fixes.

Truth document: docs/FLEX_INTEGRATION_PLAN.md (Sec B.2/B.3, C.1, F.2, A.3), the
plan of record for exposing afm-litho as a FLEX provider. NONE of this is FLEX code
-- it is the set of live defects that letting anything unattended near the tip would
amplify, so it ships first and is verified on the twin here (and at the rig, per
docs/FLEX_RIG_RUNBOOK.md "01 Foundation").

Zero hardware: a `HardwareARC` over `FakeIgorBridge`, the same double the hardware
tests use, plus TestClient service checks over the twin.

Covered:
  1. the COM interrupt path is owned by a DEDICATED thread (not an asyncio pool
     thread), which is the thread that CoInitialize's COM and owns the LazyIgor;
  2. bounded fail-safe/verify reads -- a stalled bridge cannot make safe_park or
     hw_withdraw hang, and the reply says it is unverified;
  3. the COM->TCP fallback is never silent (transport + com_error in the result);
  4. abort stops AND retracts;
  5. the bridge-closed liveness signal: backend_live, the fault state, /api/state;
  6. the telemetry cache's `t` / `x_um` / `y_um`, unconditionally;
  7. `_scan_repeat_stop` is settable from a foreign thread.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time

import pytest

from afm_litho.hardware.bridge_router import BridgeReadTimeout, send_abort
from afm_litho.hardware.hardware_arc import HardwareARC
from afm_litho.hardware.igor_stub import FakeIgorBridge
# MIGRATED (B1, docs/INSTRUMENT_MODEL_SPEC.md Sec 3 + 10): the async out-of-band
# hardware orchestrator this file exercises (COM/TCP trigger, failsafe watchdog,
# bounded Output.Z verify) is now `safe_park_async`; the bare `safe_park` is the new
# synchronous model-driven ParkResult primitive (tests/test_safe_park_model.py).
from afm_litho.service.host.safe_park import safe_park_async as safe_park

safe_park_mod = sys.modules[safe_park.__module__]


@pytest.fixture(autouse=True)
def _fast_verify_settle(monkeypatch):
    """Keep the units fast: the real verify waits ~2.5 s for the Output.Z ramp. The
    READ timeout is deliberately NOT patched -- these tests are about that bound."""
    monkeypatch.setattr(safe_park_mod, "_VERIFY_SETTLE_S", 0.05)


class _Store:
    def __init__(self):
        self.records: list[tuple] = []

    def record(self, kind, payload):
        self.records.append((kind, payload))
        return {"id": "rec", "kind": kind}


class _FakeCom:
    """A COM Igor handle stand-in that records the expression AND the thread it was
    called on -- the whole point of fix 1 is WHICH thread makes the COM call."""

    def __init__(self, fail: bool = False):
        self.calls: list[str] = []
        self.threads: list[str] = []
        self.fail = fail

    def execute(self, cmd, check=False):
        self.calls.append(cmd)
        self.threads.append(threading.current_thread().name)
        if self.fail:
            raise RuntimeError("COM rejected: CoInitialize has not been called")
        return 0


class _FakeHub:
    """The slice of the hub safe_park + its wrapper touch, built by hand. Now carries
    the `instrument_model` / `free_air` / `litho_worker` the sync retract+verify
    primitive reads (B1b: safe_park_async's retract IS that primitive)."""

    def __init__(self, *, hardware_arc=None):
        from _im_fixtures_guard import make_model

        from afm_litho.service.command_guard import FreeAir
        self.hardware_arc = hardware_arc
        self.arc = None
        self.inst = None
        self.engaged = True
        self.instrument_model = make_model(z_out_channel="Output.Z", z_park_v=0.0,
                                           z_park_tol_v=0.5)
        self.litho_worker = None
        self.free_air = FreeAir(value_v=0.0, ts=time.time(), mode="contact",
                                withdrawn=True)
        self.fault = None
        self.last_park = None
        self.scan_repeat = {"enabled": True, "stopped_reason": None}
        self._scan_repeat_stop = threading.Event()
        self._hw_abort = threading.Event()
        self._hw_litho_abort = threading.Event()
        self._safe_park_now = threading.Event()
        self._safe_park_reason = None
        self._failsafe = None
        self._loop = None
        self._safe_park = {"count": 0, "active": False, "last_reason": None,
                           "verified": None, "critical": False, "transport": None,
                           "z_v": None, "at": None}
        self.store = _Store()
        self.logs: list[str] = []
        self.broadcasts: list[dict] = []

    def backend_live(self):
        return "hardware" if self.hardware_arc is not None else "twin"

    async def log(self, text):
        self.logs.append(text)

    async def broadcast(self, msg):
        self.broadcasts.append(msg)

    @property
    def host(self):                # fake: host fields live on the fake itself
        return self


def _hw_over(stub: FakeIgorBridge, **cfg) -> HardwareARC:
    port = stub.serve_in_thread()
    return HardwareARC.connect("127.0.0.1", port, **cfg)


def _retracted_stub(height: float = 0.0) -> FakeIgorBridge:
    """A bridge whose ZSensor + Deflection read RETRACTED, so the sync primitive's
    sensor verify (B1b) confirms the park. `height` is the (now non-evidentiary)
    Output.Z readback."""
    return FakeIgorBridge({"height": height, "Output.Z": height,
                           "ZSensor": 0.0, "Deflection": 0.0})


def _shutdown(hub, hw, stub):
    wd = getattr(hub.host, "_failsafe", None)
    if wd is not None:
        wd.stop()
    hw.close()
    stub.stop()


# =============================================== 1. the dedicated COM thread ==

def test_safe_park_fires_com_from_the_dedicated_watchdog_thread():
    """FIX 1. The out-of-band trigger must NOT run on an arbitrary asyncio pool
    thread: COM has thread affinity, and nothing in the repo ever called
    pythoncom.CoInitialize(), so a GetActiveObject on an uninitialised thread can
    raise -- and send_safe_park swallowed every exception and fell through to the
    TCP path, silently degrading the COM-first fail-safe to the transport a comms
    failure has already broken (plan Sec B.2).

    Assert the property directly: the COM call happens on the dedicated
    `afm-failsafe` thread, which is neither the caller's thread nor an
    `asyncio.to_thread` pool worker."""
    # already retracted: the sensor verify confirms on the FIRST attempt, so the
    # COM path fires exactly once (no vendor fallback).
    stub = _retracted_stub(0.0)
    hw = _hw_over(stub)
    com = _FakeCom()
    hw.com_igor = com
    hub = _FakeHub(hardware_arc=hw)
    try:
        res = asyncio.run(safe_park(hub, "unit: which thread"))
        assert res["transport"] == "com"
        assert com.calls == ["AFM_SafePark()"]
        assert com.threads == ["afm-failsafe"]
        assert "asyncio" not in com.threads[0]
        assert com.threads[0] != threading.current_thread().name
        wd = hub.host._failsafe
        assert wd.alive and wd.parks == 1
        # the thread OWNS a COM handle for its lifetime (built on itself), even
        # though this test injected one on the ARC.
        assert wd.com_igor is not None
        # off the instrument PC pywin32 is absent, so CoInitialize is skipped --
        # explicitly recorded rather than assumed.
        assert wd.com_initialized in (True, False)
    finally:
        _shutdown(hub, hw, stub)


def test_watchdog_serves_an_out_of_band_trigger_with_no_loop():
    """Any thread may demand a verified retract with NO event loop and NO await --
    the property the whole watchdog exists for, and what a bridge-closed hook or a
    future ZMQ REP thread uses (plan Sec B.3)."""
    stub = _retracted_stub(66.0)
    hw = _hw_over(stub)
    hub = _FakeHub(hardware_arc=hw)
    try:
        wd = safe_park_mod.watchdog_for(hub)
        req = wd.park_now("unit: out-of-band")
        assert req.wait(10.0), "watchdog did not serve the request"
        assert req.result["verified"] is True
        assert stub.safe_parks >= 1
    finally:
        _shutdown(hub, hw, stub)


def test_bare_event_trigger_uses_the_recorded_reason():
    """`hub._safe_park_reason = ...; hub._safe_park_now.set()` -- the two-line,
    loop-free form the plan specifies."""
    stub = _retracted_stub(66.0)
    hw = _hw_over(stub)
    hub = _FakeHub(hardware_arc=hw)
    try:
        wd = safe_park_mod.watchdog_for(hub)
        hub.host._safe_park_reason = "unit: bare event"
        hub.host._safe_park_now.set()
        deadline = time.monotonic() + 10.0
        while wd.parks == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert wd.parks == 1
        assert stub.safe_parks >= 1
    finally:
        _shutdown(hub, hw, stub)


# ================================================== 2. bounded verify reads ==

def test_bounded_read_raises_bridge_read_timeout():
    """FIX 2. `td_ReadValue` takes an explicit bound; a read that gives up raises
    BridgeReadTimeout, so a fail-safe caller can tell "no answer in time" from "the
    bridge answered with an error"."""
    stub = FakeIgorBridge({"height": 66.0})
    stub.stall_getvalue = True
    hw = _hw_over(stub)
    try:
        t0 = time.monotonic()
        with pytest.raises(BridgeReadTimeout):
            hw.td_ReadValue("height", timeout=0.3)
        assert time.monotonic() - t0 < 3.0      # bounded, NOT the 30 s default
        assert hw.timeout == 30.0               # the default is untouched
    finally:
        hw.close()
        stub.stop()


def test_safe_park_returns_promptly_on_a_stalled_bridge():
    """FIX 2, the headline. With the 30 s default inherited, the worst-case park was
    trigger + 2.5 s settle + 30 s read, RETRIED ONCE ~= 65 s -- an emergency stop
    that appears to HANG (plan Sec F.2). The retract has already been COMMANDED by
    the time the verify read runs; only the CONFIRMATION is at stake, so the park
    must return saying it is unverified rather than block.

    The stub receives the ABORT (== AFM_SafePark) but never answers GETVALUE.

    CONTRADICTION (13.1 item 7): OLD -- the verify polled the Output.Z READBACK,
    retried the whole trigger ONCE, and reported `verify_timeout`/`retried`. NEW --
    the sync primitive reads the ZSensor to verify; a read that STALLS is short-
    circuited (never spun), so a failed sensor read is unverified + CRITICAL within a
    couple of bounded reads. The out-of-band retract itself still fires."""
    stub = FakeIgorBridge({"height": 66.0})
    stub.stall_getvalue = True
    hw = _hw_over(stub)
    hub = _FakeHub(hardware_arc=hw)
    try:
        t0 = time.monotonic()
        res = asyncio.run(safe_park(hub, "unit: stalled bridge"))
        elapsed = time.monotonic() - t0
        # a ZSensor read that stalls is short-circuited (one bounded read per verify,
        # ramp + vendor), ~4 s, and nowhere near the old ~65 s hang.
        assert elapsed < 12.0, f"safe_park took {elapsed:.1f}s -- not bounded"
        assert res["verified"] is False
        assert res["z_v"] is None               # the commanded readback was unreadable
        assert res["critical"] is True          # unverified -> operator must look
        assert stub.safe_parks >= 1             # the out-of-band RETRACT still fired
        assert any("CRITICAL" in m for m in hub.logs)
    finally:
        _shutdown(hub, hw, stub)


def test_hw_withdraw_returns_promptly_on_a_stalled_bridge():
    """FIX 2, the other unbounded fail-safe path: hw_withdraw's snapshot reads were
    2.5 s + two 30 s-capped reads ~= 62.5 s. Bounded now, and it SAYS the readback
    is unavailable instead of blocking."""
    from afm_litho.hardware import hw_approach

    stub = FakeIgorBridge({"height": 66.0})
    stub.stall_getvalue = True
    hw = _hw_over(stub)
    try:
        t0 = time.monotonic()
        snap = hw_approach.hw_withdraw(hw, max_seconds=0.05, settle_margin=0.0)
        elapsed = time.monotonic() - t0
        assert elapsed < 10.0, f"hw_withdraw took {elapsed:.1f}s -- not bounded"
        assert snap["engaged"] is False
        assert snap["z_v"] is None and snap["defl_v"] is None
        assert snap["verify_timeout"] is True
    finally:
        hw.close()
        stub.stop()


def test_engage_snapshot_keeps_the_backend_default():
    """The bound is for FAIL-SAFE paths only. An engage/tune snapshot passes no
    read_timeout, so it keeps the backend's own default -- this is not a place where
    returning early is safer than waiting."""
    from afm_litho.hardware import hw_approach

    seen: list = []

    class _Arc:
        is_cypher = False
        z_out_channel = "Output.Z"

        def td_ReadValue(self, name, **kw):
            seen.append(kw)
            return 0.0

    hw_approach.snapshot(_Arc(), engaged=True)
    assert seen and all(kw == {} for kw in seen)


# ============================================ 3. the fallback is never silent ==

def test_com_failure_surfaces_transport_and_com_error():
    """FIX 1/3. `send_safe_park` swallowed every COM exception and fell through to
    TCP. The retract still happens -- COM-FIRST is preserved -- but the downgrade is
    now VISIBLE: `transport: "tcp"` plus the exception text in `com_error`, and a
    log line. A fail-safe whose transport choice cannot be observed from outside is
    a fail-safe nobody can audit (plan Sec B.2)."""
    stub = _retracted_stub(66.0)
    hw = _hw_over(stub)
    com = _FakeCom(fail=True)
    hw.com_igor = com
    hub = _FakeHub(hardware_arc=hw)
    try:
        res = asyncio.run(safe_park(hub, "unit: com down"))
        assert com.calls == ["AFM_SafePark()"]           # COM was tried FIRST
        assert res["transport"] == "tcp"                 # and it fell back
        assert res["com_error"]                          # ... never silently
        assert "CoInitialize" in res["com_error"]
        assert res["verified"] is True                   # the TCP park still worked
    finally:
        _shutdown(hub, hw, stub)


def test_com_success_leaves_com_error_none():
    stub = _retracted_stub(0.0)              # sensors retracted -> verify passes
    hw = _hw_over(stub)
    hw.com_igor = _FakeCom()
    hub = _FakeHub(hardware_arc=hw)
    try:
        res = asyncio.run(safe_park(hub, "unit: com up"))
        assert res["transport"] == "com" and res["com_error"] is None
    finally:
        _shutdown(hub, hw, stub)


# ========================================== 4. abort stops AND retracts ======

def test_arc_abort_now_retracts():
    """FIX 3. `arc.abort()` reached a COM `AFM_Abort()` -- stop, NO retract -- and is
    called from the ARC wrapper, the litho run and EVERY chained-scan abort. It now
    fires `AFM_SafePark()`, which calls `AFM_Abort()` first (identical sequencing)
    and then retracts."""
    stub = FakeIgorBridge({"height": 66.0})
    hw = _hw_over(stub)
    com = _FakeCom()
    hw.com_igor = com
    try:
        assert hw.abort() == "com"
        assert com.calls == ["AFM_SafePark()"]
        com.calls.clear()
        hw.td_Stop()
        assert com.calls == ["AFM_SafePark()"]
    finally:
        hw.close()
        stub.stop()


def test_tcp_abort_reaches_the_safe_park_dispatch():
    """The TCP fallback was ALREADY stop-and-retract (the UID-aware .ipf dispatches
    the ABORT verb to AFM_SafePark), so the non-retracting behaviour was COM-only --
    the two transports of one abort disagreed about whether the tip came off the
    surface. Both now agree."""
    class _Link:
        def __init__(self):
            self.sent: list[str] = []

        def send(self, line):
            self.sent.append(line)

    link = _Link()
    assert send_abort(link, None) == "tcp"
    assert link.sent and link.sent[0].startswith("ABORT")


# ================================== 5. bridge liveness / fault / /api/state ==

def test_backend_live_flips_when_the_bridge_closes():
    """FIX 4. `hub.hardware_arc` is assigned once and never cleared, so
    `hardware_arc is not None` means "Igor connected ONCE". `bridge_closed`, set from
    the single choke point BridgeRouter._mark_closed, makes the answer LIVE."""
    from afm_litho.service.hub import Hub

    hub = Hub()
    stub = FakeIgorBridge({"height": 0.0})
    hw = _hw_over(stub)
    try:
        hub.bind_hardware(hw)
        assert hub.backend_live() == "hardware"
        assert hub.backend_label() in ("hardware",)
        assert hub.fault is None

        hw.router.stop()                     # every close path funnels through here
        deadline = time.monotonic() + 5.0
        while not hub.host.bridge_closed.is_set() and time.monotonic() < deadline:
            time.sleep(0.02)

        assert hub.host.bridge_closed.is_set()
        assert hub.host.bridge_closed_at is not None
        assert hub.host.hardware_arc is not None          # still latched, deliberately
        assert hub.backend_live() == "twin"          # ... but the ANSWER is truthful
        assert hub.fault is not None
        assert hub.fault["reason"] == "bridge_closed"
        assert hub.fault["state"] == "disconnected"
        assert hub.state_label() == "fault"
    finally:
        wd = getattr(hub.host, "_failsafe", None)
        if wd is not None:
            wd.stop()
        hw.close()
        stub.stop()


def test_bridge_close_during_a_run_demands_a_retract():
    """A closed-bridge transition while a run is IN FLIGHT must fire a verified
    retract -- through the watchdog, the only path that can still reach Igor when
    the bridge socket is the thing that died."""
    from afm_litho.service.hub import Hub

    hub = Hub()
    stub = FakeIgorBridge({"height": 66.0})
    hw = _hw_over(stub)
    try:
        hub.bind_hardware(hw)
        hub.host.scan_repeat["enabled"] = True             # a continuous run is armed
        assert hub.run_in_flight() is True

        hw.router.stop()
        deadline = time.monotonic() + 10.0
        while stub.safe_parks == 0 and time.monotonic() < deadline:
            time.sleep(0.02)

        assert stub.safe_parks >= 1, "no retract was demanded on bridge close"
        assert hub.fault["was_running"] is True
    finally:
        wd = getattr(hub.host, "_failsafe", None)
        if wd is not None:
            wd.stop()
        hw.close()
        stub.stop()


def test_bridge_close_while_idle_latches_the_fault_without_retracting():
    """Nothing was driving the tip, so there is nothing to stop: latch the fault,
    do NOT disturb the instrument."""
    from afm_litho.service.hub import Hub

    hub = Hub()
    stub = FakeIgorBridge({"height": 0.0})
    hw = _hw_over(stub)
    try:
        hub.bind_hardware(hw)
        assert hub.run_in_flight() is False
        hw.router.stop()
        deadline = time.monotonic() + 5.0
        while not hub.host.bridge_closed.is_set() and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.3)
        assert stub.safe_parks == 0
        assert hub.fault["was_running"] is False
    finally:
        wd = getattr(hub.host, "_failsafe", None)
        if wd is not None:
            wd.stop()
        hw.close()
        stub.stop()


def test_api_state_reports_backend_state_and_bridge(client):
    """/api/state built its snapshot from the TWIN arc regardless of backend, so on
    a hardware station the GUI's own status was twin-sourced. It now reports the
    truthful backend, a coarse state, and the bridge block."""
    st = client.get("/api/state").json()
    assert st["backend"] in ("twin", "hardware", "disconnected")
    assert st["state"] in ("idle", "scanning", "writing", "fault")
    assert st["bridge"] == {"closed": False, "closed_at": None}
    assert st["fault"] is None
    assert "safe_park" in st


def test_api_state_reports_fault_once_the_bridge_drops(client):
    """`disconnected` / `fault` are TERMINAL: HardwareARC.serve accepts ONE
    connection and closes its listener in a finally, so there is no re-accept path
    and no `disconnected -> idle` edge (plan Sec C.1 / risk 6)."""
    from afm_litho.service.hub import hub as real_hub

    saved = (real_hub.host.backend, real_hub.fault, real_hub.host.bridge_closed_at)
    try:
        real_hub.host.backend = "hardware"
        real_hub.host.bridge_closed.set()
        real_hub.host.bridge_closed_at = 1234.5
        real_hub.fault = {"reason": "bridge_closed", "at": 1234.5,
                          "state": "disconnected", "detail": None,
                          "was_running": False}
        st = client.get("/api/state").json()
        assert st["backend"] == "disconnected"
        assert st["state"] == "fault"
        assert st["bridge"]["closed"] is True
        assert st["fault"]["reason"] == "bridge_closed"
    finally:
        real_hub.host.backend, real_hub.fault, real_hub.host.bridge_closed_at = saved
        real_hub.host.bridge_closed.clear()


# ============================================= 6. telemetry cache fields =====

def test_telemetry_cache_carries_t_and_microns_without_show_tip():
    """FIX 5 (plan Sec A.3). The cache had no timestamp -- so a consumer could not
    tell a fresh frame from a FROZEN one, which is exactly what a dead bridge
    produces -- and no XY in microns, because the volts->um conversion happened only
    inside the tip_pos broadcast and only while the Show-Tip DISPLAY toggle was on.
    Both are unconditional now; the broadcast stays gated."""
    from afm_litho.service.hub import hub as real_hub
    from afm_litho.service.host import telemetry

    saved = (real_hub.host._hw_telemetry, dict(real_hub.host.hw_cal),
             real_hub._show_tip_display)
    try:
        real_hub._show_tip_display = False              # display OFF on purpose
        real_hub.host.hw_cal = {"x_lvdt_sens": 4.0e-6, "x_lvdt_offset": 0.0,
                           "y_lvdt_sens": 4.0e-6, "y_lvdt_offset": 0.0}
        t0 = time.time()
        telemetry._hw_telemetry_frame(1.0, -0.25, 66.0, 0.5, 6.1, 0.0)
        tel = real_hub.host._hw_telemetry
        assert tel["t"] >= t0
        assert tel["x_um"] == pytest.approx(4.0)
        assert tel["y_um"] == pytest.approx(-1.0)
        assert tel["defl_v"] == pytest.approx(0.5)
        assert tel["sum_v"] == pytest.approx(6.1)
        # kept for wire compatibility, but ZSensor is not polled, so never a number.
        assert tel["height_nm"] is None
    finally:
        (real_hub.host._hw_telemetry, real_hub.host.hw_cal,
         real_hub._show_tip_display) = saved


def test_telemetry_microns_are_none_when_the_volts_are_unreadable():
    from afm_litho.service.hub import hub as real_hub
    from afm_litho.service.host import telemetry

    saved = real_hub.host._hw_telemetry
    try:
        telemetry._hw_telemetry_frame(float("nan"), float("nan"), 0.0, 0.0, 0.0, 0.0)
        tel = real_hub.host._hw_telemetry
        assert tel["x_um"] is None and tel["y_um"] is None
        assert tel["t"] is not None
    finally:
        real_hub.host._hw_telemetry = saved


# ================================ 7. _scan_repeat_stop is thread-safe ========

def test_scan_repeat_stop_is_a_threading_event():
    """FIX 6. safe_park sets it from NON-loop code paths (the watchdog thread, the
    router reader, a worker). `asyncio.Event.set` from a foreign thread is a SILENT
    NO-OP for any waiter -- the stop would simply not land (plan Sec B.3)."""
    from afm_litho.service.hub import hub as real_hub

    assert isinstance(real_hub.host._scan_repeat_stop, threading.Event)
    assert not isinstance(real_hub.host._scan_repeat_stop, asyncio.Event)


def test_wait_stop_returns_early_when_set_from_another_thread():
    """The two `await stop.wait()` sites became `continuous._wait_stop`, which polls
    the threading.Event. Prove a foreign thread's `set()` actually lands."""
    from afm_litho.service.host.continuous import _wait_stop

    ev = threading.Event()
    threading.Timer(0.15, ev.set).start()

    async def _go():
        t0 = time.monotonic()
        landed = await _wait_stop(ev, 5.0)
        return landed, time.monotonic() - t0

    landed, elapsed = asyncio.run(_go())
    assert landed is True
    assert elapsed < 2.0                   # broke out early, did not sit out the 5 s


def test_wait_stop_times_out_when_never_set():
    from afm_litho.service.host.continuous import _wait_stop

    async def _go():
        return await _wait_stop(threading.Event(), 0.2)

    assert asyncio.run(_go()) is False


# ============================ 8. watchdog request/trigger channel separation ==

def test_a_queued_request_never_synthesises_a_second_bare_park():
    """The queued-request wake and the BARE `hub._safe_park_now` trigger are
    SEPARATE channels on purpose.

    If `request()` also set the shared trigger, the loop could not tell "somebody
    set the bare trigger" from "I already drained the request that set it" -- and a
    request queued between the clear and the drain would leave the trigger set with
    an empty queue on the next pass, firing a SPURIOUS SECOND retract. One request
    must produce exactly one park."""
    stub = FakeIgorBridge({"height": 66.0})
    hw = _hw_over(stub)
    hub = _FakeHub(hardware_arc=hw)
    try:
        wd = safe_park_mod.watchdog_for(hub)
        for _ in range(3):
            assert wd.request("unit: one each").wait(10.0)
        time.sleep(0.4)                       # give a spurious bare park time to fire
        assert wd.parks == 3
        assert hub.host._safe_park_now.is_set() is False
    finally:
        _shutdown(hub, hw, stub)


def test_several_requests_queued_at_once_are_coalesced_but_all_answered():
    """Requests queued faster than the thread drains them must EVERY ONE get an
    answer -- but they share ONE retract, not one each.

    Serially, each park costs up to ~4.5 s. A single interlock trip already produces
    two demands (the chained-scan abort plus the supervisor's own safe_park); add an
    operator Abort and the third waiter would blow past its wait bound and be told
    the park was pending/CRITICAL when it had in fact succeeded seconds earlier. The
    retract is idempotent, so one serves all."""
    stub = FakeIgorBridge({"height": 66.0})
    hw = _hw_over(stub)
    hub = _FakeHub(hardware_arc=hw)
    try:
        wd = safe_park_mod.watchdog_for(hub)
        reqs = [wd.request(f"unit: burst {i}") for i in range(4)]
        for r in reqs:
            assert r.wait(20.0), "a queued request was dropped"
            assert r.result is not None          # everyone gets an outcome
            assert r.future.done()
        assert wd.parks < 4, "demands were not coalesced"
        assert stub.safe_parks < 8, "one retract per demand, not one per batch"
        # a shared outcome must not be a shared DICT -- one caller must not be able
        # to rewrite another's result.
        reqs[0].result["verified"] = "tampered"
        assert reqs[1].result["verified"] != "tampered"
    finally:
        _shutdown(hub, hw, stub)


def test_concurrent_safe_parks_all_return_within_bound():
    """The end-to-end shape of the same property: N safe_park coroutines racing (an
    interlock trip plus an operator Abort) must ALL come back inside the bound with a
    real outcome, not a false `pending`/CRITICAL."""
    stub = _retracted_stub(66.0)                # sensors retracted -> verify passes
    hw = _hw_over(stub)
    hub = _FakeHub(hardware_arc=hw)

    async def _go():
        return await asyncio.gather(*[safe_park(hub, f"unit: concurrent {i}")
                                      for i in range(4)])
    try:
        t0 = time.monotonic()
        results = asyncio.run(_go())
        elapsed = time.monotonic() - t0
        assert elapsed < 20.0, f"concurrent parks took {elapsed:.1f}s"
        for res in results:
            assert res["transport"] != "pending", "a caller got a FALSE pending"
            assert res["verified"] is True
    finally:
        _shutdown(hub, hw, stub)


def test_stopped_watchdog_releases_a_waiter_instead_of_hanging():
    """Teardown must never strand a caller blocked on `ParkRequest.wait` -- a
    fail-safe that hangs its caller is worse than one that reports failure."""
    from afm_litho.hardware.failsafe_thread import FailsafeWatchdog

    gate = threading.Event()

    def _slow_work(reason, com):
        gate.wait(5.0)
        return {"transport": "tcp", "verified": True, "z_v": 0.0,
                "critical": False, "retried": False, "verify_timeout": False,
                "com_error": None}

    wd = FailsafeWatchdog(_slow_work).start()
    try:
        first = wd.request("unit: occupies the thread")
        second = wd.request("unit: still queued")
        wd.stop(join=0.2)                     # stop while the thread is busy
        gate.set()
        assert second.wait(5.0), "a queued request was left hanging by stop()"
        assert second.result["critical"] is True
        assert "stopped" in second.result["detail"]
        first.wait(5.0)
    finally:
        gate.set()
        wd.stop()


# ================== 9. review round 2: concurrency, ordering, gating ==========

def test_concurrent_watchdog_for_creates_exactly_one():
    """MUST-FIX 1. `watchdog_for` is called from the LOOP (safe_park) and from a
    FOREIGN THREAD (the bridge-router reader, via hub.mark_bridge_closed), and the
    check-then-create had no mutual exclusion: two callers could each see no live
    watchdog and each start one. The loser is ORPHANED -- it holds its own COM
    apartment for the life of the process and still watches the SHARED
    `hub._safe_park_now`, whose is_set()/clear() pair is not atomic across two
    threads, so ONE bare trigger could fire TWO retracts."""
    stub = FakeIgorBridge({"height": 0.0})
    hw = _hw_over(stub)
    hub = _FakeHub(hardware_arc=hw)
    barrier = threading.Barrier(8)
    got: list = []
    lock = threading.Lock()

    def _race():
        barrier.wait(10)
        wd = safe_park_mod.watchdog_for(hub)
        with lock:
            got.append(wd)

    threads = [threading.Thread(target=_race) for _ in range(8)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(20)
        assert len(got) == 8
        assert len({id(w) for w in got}) == 1, "watchdog_for raced into two watchdogs"
        assert got[0] is hub.host._failsafe
        # Assert on THIS hub's watchdog, not on a process-wide thread census. The
        # census version was deterministically order-dependent: any earlier test file
        # that left an `afm-failsafe` thread running made this fail, which looked like
        # a flake and was not. What the test is actually about is that eight racing
        # callers produced ONE watchdog and started ONE thread -- both local facts.
        wd = hub.host._failsafe
        assert wd.alive
        threads = {id(w._thread) for w in got}
        assert len(threads) == 1, "the racing callers started more than one thread"
        assert wd._thread is not None and wd._thread.name == "afm-failsafe"
    finally:
        _shutdown(hub, hw, stub)


def test_bind_hardware_catches_a_bridge_that_closed_before_binding():
    """MUST-FIX 2. `on_closed` is a ONE-SHOT EDGE and there is real work between
    accepting the socket and binding -- `arc.start()` reads the STARTUP calibration.
    A bridge that dies during that read fires the edge into a hook nobody has
    installed, and the service then reports `backend: "hardware"` forever on a dead
    bridge: a permanent lie, and this milestone's own done-condition. bind_hardware
    re-checks the LEVEL (`router.closed`) right after wiring."""
    from afm_litho.service.hub import Hub

    hub = Hub()
    stub = FakeIgorBridge({"height": 0.0})
    hw = _hw_over(stub)
    try:
        hw.router.stop()                       # the edge passes BEFORE we bind
        assert hw.router.closed is True
        assert hub.host.bridge_closed.is_set() is False

        hub.bind_hardware(hw)                  # must notice the level

        assert hub.host.bridge_closed.is_set() is True
        assert hub.backend_live() == "twin"
        assert hub.fault is not None
        assert "before bind" in (hub.fault.get("detail") or "")
    finally:
        wd = getattr(hub.host, "_failsafe", None)
        if wd is not None:
            wd.stop()
        hw.close()
        stub.stop()


def test_on_closed_fires_on_the_reader_own_exit_not_just_stop():
    """The close signal must come from the READER exit path (peer EOF / socket
    error), not only from an explicit `stop()` -- a bridge drop is never polite."""
    from afm_litho.hardware.bridge_router import BridgeRouter

    fired: list = []

    class _EofLink:
        def read_line(self, timeout=None):
            return None                        # peer closed

    r = BridgeRouter(_EofLink())
    r.on_closed = lambda *a: fired.append(True)
    r.start()
    deadline = time.monotonic() + 5.0
    while not fired and time.monotonic() < deadline:
        time.sleep(0.02)
    assert fired == [True], "on_closed did not fire on the reader EOF exit"
    assert r.closed is True


def test_loop_free_park_also_signals_the_cooperative_aborts():
    """MUST-FIX 3. Step 1 of a park (disable scan_repeat, set the cooperative abort
    Events) lived ONLY in the coroutine, so a park demanded from a foreign thread
    retracted the tip but left the continuous supervisor ARMED -- it would then arm
    the next frame against a dead bridge."""
    stub = FakeIgorBridge({"height": 66.0})
    hw = _hw_over(stub)
    hub = _FakeHub(hardware_arc=hw)
    try:
        wd = safe_park_mod.watchdog_for(hub)
        assert hub.host.scan_repeat["enabled"] is True
        hub.host._safe_park_reason = "unit: loop-free stop"
        hub.host._safe_park_now.set()
        deadline = time.monotonic() + 15.0
        while wd.parks == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert wd.parks == 1
        assert hub.host.scan_repeat["enabled"] is False
        assert hub.host.scan_repeat["stopped_reason"] == "safe_park"
        assert hub.host._scan_repeat_stop.is_set()
        assert hub.host._hw_abort.is_set()
        assert hub.host._hw_litho_abort.is_set()
    finally:
        _shutdown(hub, hw, stub)


def test_arc_abort_hands_off_to_the_watchdog_when_wired():
    """MUST-FIX 4. `arc.abort()` runs on the chained-scan reader worker thread, and
    LazyIgor attaches (and caches) its COM proxy on whichever thread uses it FIRST --
    so an abort from a pool thread could build the proxy in an apartment that later
    dies, poisoning the fail-safe own handle. The hand-off is non-blocking, so the
    chunk-boundary abort keeps its send-only, never-queues property."""
    stub = FakeIgorBridge({"height": 66.0})
    hw = _hw_over(stub)
    com = _FakeCom()
    hw.com_igor = com
    calls: list = []
    hw.failsafe_park = lambda reason: calls.append(reason)
    try:
        t0 = time.monotonic()
        assert hw.abort() == "watchdog"
        assert time.monotonic() - t0 < 1.0     # returned immediately
        assert calls == ["arc.abort"]
        assert com.calls == [], "abort touched COM on the calling thread"
    finally:
        hw.close()
        stub.stop()


def test_arc_abort_falls_back_when_the_handoff_raises():
    """A hand-off failure must never swallow an abort: fall through to the direct
    out-of-band path, which is what this method did before the hook existed."""
    stub = FakeIgorBridge({"height": 66.0})
    hw = _hw_over(stub)
    com = _FakeCom()
    hw.com_igor = com

    def _boom(reason):
        raise RuntimeError("watchdog gone")

    hw.failsafe_park = _boom
    try:
        assert hw.abort() == "com"
        assert com.calls == ["AFM_SafePark()"]
    finally:
        hw.close()
        stub.stop()


def test_arc_and_watchdog_do_not_share_one_lazy_igor(_flex_provider_off, monkeypatch):
    """The ARC must keep its OWN COM handle. A shared handle lets whichever thread
    aborts first decide which COM apartment the FAIL-SAFE proxy lives in.

    Runs the real start-up path with a fake bridge and a LazyIgor that only counts."""

    from fastapi.testclient import TestClient

    from afm_litho.hardware import igor as igor_mod
    from afm_litho.service.app import app
    from afm_litho.service.hub import hub

    class _Handle:
        """Stands in for LazyIgor: never touches COM, but each build is a new object."""

    class _FakeArc:
        com_igor = None
        router = None

        def start(self):
            raise RuntimeError("stop after the COM handles are wired")

    arc = _FakeArc()
    monkeypatch.setattr(igor_mod, "LazyIgor", _Handle)
    monkeypatch.setattr(HardwareARC, "serve", classmethod(lambda cls, port: arc))
    monkeypatch.setattr(hub.host, "backend", "hardware")
    wd = None
    try:
        with TestClient(app):
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not (
                    arc.com_igor is not None and getattr(hub.host, "_failsafe", None)):
                time.sleep(0.02)
            wd = hub.host._failsafe
            assert arc.com_igor is not None and wd.com_igor is not None
            assert arc.com_igor is not wd.com_igor
    finally:
        if wd is not None:
            wd.stop()
        hub.host._failsafe = None


def test_require_live_backend_refuses_a_configured_hardware_station_with_no_bridge():
    """MUST-FIX 8 / plan Sec C.2. PRE-CONNECT is the SILENT case: `hardware_arc` is
    None, so a scan or write falls straight through to the twin branch and returns
    ordinary success -- an unattended run would "write" a whole chip against
    simulated physics."""
    from afm_litho.service.hub import Hub

    hub = Hub()
    hub.host.backend = "hardware"
    hub.host.hardware_arc = None
    why = hub.require_live_backend()
    assert why and "has not connected" in why

    class _Arc:
        pass

    hub.host.hardware_arc = _Arc()
    assert hub.require_live_backend() is None      # live bridge: allowed
    hub.host.bridge_closed.set()
    why = hub.require_live_backend()
    assert why and "CLOSED" in why


def test_require_live_backend_never_refuses_a_twin_station():
    """A genuinely twin-configured station must be untouched by the guard."""
    from afm_litho.service.hub import Hub

    hub = Hub()
    hub.host.backend = "twin"
    hub.host.hardware_arc = None
    assert hub.require_live_backend() is None


@pytest.mark.parametrize(
    ("path", "body"),
    [("/api/scan/start", {"size_um": 5}),
     ("/api/litho/write", {}),
     ("/api/mode", {"mode": "contact"}),
     ("/api/approach", {})])
def test_mutating_endpoints_refuse_when_the_backend_is_not_live(client, path, body):
    """The four tip-mutating entry points refuse rather than silently driving the
    twin on a hardware-configured station with no live bridge."""
    from afm_litho.service.hub import hub as real_hub

    saved = real_hub.host.backend
    try:
        real_hub.host.backend = "hardware"          # configured hardware, nothing connected
        r = client.post(path, json=body).json()
        assert "error" in r, f"{path} did not refuse"
        assert r["reason"] == "backend_not_live"
        assert r["backend"] == "disconnected"
    finally:
        real_hub.host.backend = saved


def test_twin_station_endpoints_still_work(client):
    """... and the same endpoints are untouched on a twin station."""
    r = client.post("/api/mode", json={"mode": "contact"}).json()
    assert r.get("ok") is True and r["backend"] == "twin"


# ============================== 8. a dead watchdog must not swallow an abort ==

def test_park_now_raises_on_a_stopped_watchdog():
    """`park_now` on a watchdog whose thread is not running RAISES.

    The queue is served by the thread; with no thread, `request()` would return a
    ParkRequest nobody ever completes. `HardwareARC.abort` treats a returned request
    as "the watchdog owns this retract now", so a silent success there is an ABORT
    THAT NEVER HAPPENS. Raising is what lets `abort` fall through to the direct
    COM/TCP path."""
    from afm_litho.hardware.failsafe_thread import FailsafeWatchdog

    wd = FailsafeWatchdog(lambda reason, com: {"transport": "com", "verified": True})
    assert not wd.alive
    with pytest.raises(RuntimeError):                 # never started
        wd.park_now("unit: no thread")

    wd.start()
    assert wd.alive
    assert wd.park_now("unit: running").wait(10.0)

    wd.stop()
    assert not wd.alive
    with pytest.raises(RuntimeError):                 # stopped
        wd.park_now("unit: after stop")


def test_abort_falls_through_to_the_direct_path_when_the_watchdog_is_dead():
    """The end-to-end consequence: with a stopped watchdog wired as the hand-off
    hook, `abort()` must NOT report `"watchdog"` -- it must fire the retract itself
    and say which transport it used."""
    from afm_litho.hardware.failsafe_thread import FailsafeWatchdog

    stub = FakeIgorBridge({"height": 66.0})
    hw = _hw_over(stub)
    com = _FakeCom()
    hw.com_igor = com
    wd = FailsafeWatchdog(lambda reason, com_igor: {"transport": "com"})
    wd.start()
    wd.stop()                                        # the thread is gone
    hw.failsafe_park = wd.park_now
    try:
        transport = hw.abort()
        assert transport != "watchdog", "a dead watchdog silently swallowed the abort"
        assert transport == "com"
        assert com.calls == ["AFM_SafePark()"]
    finally:
        hw.close()
        stub.stop()


def test_abort_still_hands_off_to_a_live_watchdog():
    """... and the hand-off is unchanged while the thread is up."""
    from afm_litho.hardware.failsafe_thread import FailsafeWatchdog

    stub = FakeIgorBridge({"height": 66.0})
    hw = _hw_over(stub)
    com = _FakeCom()
    hw.com_igor = com
    served: list[str] = []
    wd = FailsafeWatchdog(lambda reason, com_igor: (served.append(reason),
                                                    {"transport": "com"})[1])
    wd.start()
    hw.failsafe_park = wd.park_now
    try:
        assert hw.abort() == "watchdog"
        deadline = time.monotonic() + 10.0
        while not served and time.monotonic() < deadline:
            time.sleep(0.02)
        assert served == ["arc.abort"]
        assert com.calls == []                       # the CALLER fired nothing
    finally:
        wd.stop()
        hw.close()
        stub.stop()


# =========================== E2: the twin abort door populates getLastPark ====

def test_flex_abort_on_the_twin_populates_last_park_for_getlastpark():
    """E2: the FLEX abort / safePark doors' whole retract IS `safe_park_async` (see
    `flex_dispatch._h_abort` / `_h_safe_park`), and `getLastPark` reads `hub.last_park`.
    Before E2 a FLEX abort on the twin went through the native `inst.withdraw` and never
    stored a ParkResult, so `getLastPark` returned `{}` after a twin abort while the same
    call on hardware was populated -- the twin-vs-Cypher mismatch. Now the twin routes
    through the SAME sync primitive, storing `hub.last_park`, so the FLEX read door
    reflects the abort on both backends."""
    import numpy as np

    from afm_litho.service import flex_dispatch as fd
    from afm_litho.twin.arc import SoftwareARC
    from afm_litho.twin.physics import Sample, World

    hub = _FakeHub(hardware_arc=None)                 # backend_live() -> "twin"
    hub.host.arc = SoftwareARC(World(sample=Sample(height_map=np.zeros((8, 8)),
                                              extent=1e-5)))
    hub.host.inst = None                                   # abort_scan is swallowed on twin

    # BEFORE any park: the FLEX read door is empty (not an error).
    assert fd._h_get_last_park(hub, {}) == {}

    # The FLEX abort door's retract path (what `_h_abort`'s coroutine awaits).
    res = asyncio.run(safe_park(hub, "flex abort (twin)"))
    assert res["backend"] == "twin" and res["verified"] is True

    # E2: the abort stored the ParkResult, so `getLastPark` now reflects it.
    assert hub.host.last_park is not None
    body = fd._h_get_last_park(hub, {})
    assert body["verified"] is True
    assert body["method"] in ("abort", "vendor_withdraw")
    assert body["reason"] == "flex abort (twin)"
    assert body == hub.host.last_park.to_dict()
