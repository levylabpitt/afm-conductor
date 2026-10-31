"""Stop timing and Stop windows, as they are today (H5.pre).

Characterisations pin what the code does now, including where it is not what one would want,
so that H6 and H8 cannot widen it unnoticed. The two xfails state what should hold; each flips
when a later slice fixes it (strict: an unplanned fix is loud).

Real time and real threads (these check ordering and waits), so every test is time-sensitive.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from golden.harness import GoldenRig
from golden.scenarios import WIRE, _acquire, _release

from afm_litho.hardware.failsafe_thread import FailsafeWatchdog
from afm_litho.service import board, journal as journal_module, keeper_stop
from afm_litho.service.host.safe_park import signal_run_abort
from afm_litho.service.journal import INTENT_WAIT_S, Journal
from afm_litho.service.safety import stop_pairs

pytestmark = [pytest.mark.time_sensitive, pytest.mark.xdist_group("stop-port")]

WRITE_BODY = {"objects": ["wire"], "setpoint_relative": 0.3, "amp_gain": 25.0}
SCAN_CFG = {"size_um": 1.0, "pixels": 32, "lines": 2, "line_rate_hz": 50.0, "x_offset_um": 0.0,
            "y_offset_um": 0.0, "angle_deg": 0.0, "direction": "up", "trace_retrace": True,
            "mode": "contact"}


@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


# ------------------------------------------------------- the second Stop waits --

def test_a_second_park_demand_waits_behind_the_park_that_is_running():
    """The fail-safe thread serves one retract at a time: a demand that arrives while a park
    runs is not served until it ends, and then gets a retract of its own
    (hardware/failsafe_thread.py `_loop`, `_serve_batch`)."""
    gate, started = threading.Event(), threading.Event()
    log: list[tuple[str, str]] = []

    def work(reason, _com):
        log.append((reason, "start"))
        started.set()
        if reason == "first":
            gate.wait(10)
        log.append((reason, "end"))
        return {"transport": "com", "verified": True, "z_v": 0.0, "critical": False}

    wd = FailsafeWatchdog(work).start()
    try:
        first = wd.request("first")
        assert started.wait(5)
        second = wd.request("second")
        assert not second.done.wait(0.5), "the second demand was served while the first ran"
        gate.set()
        assert first.done.wait(5) and second.done.wait(5)
    finally:
        wd.stop()
    assert log == [("first", "start"), ("first", "end"), ("second", "start"), ("second", "end")]


def test_start_once_holds_its_lock_while_the_first_park_is_being_started():
    """`stop_pairs.start_once` runs `start()` under one lock, so a Stop with a different id
    cannot even begin its own start until the first start returns (safety/stop_pairs.py)."""
    stop_pairs.clear()
    inside, release = threading.Event(), threading.Event()
    order: list[str] = []

    class Req:
        class future:                                    # noqa: N801 - stands in for a Future
            @staticmethod
            def add_done_callback(_fn):
                return None

    def slow_start():
        order.append("first-start")
        inside.set()
        release.wait(10)
        order.append("first-end")
        return Req()

    def other_start():
        order.append("second-start")
        return Req()

    t1 = threading.Thread(target=lambda: stop_pairs.start_once("a" * 16, slow_start))
    t2 = threading.Thread(target=lambda: stop_pairs.start_once("b" * 16, other_start))
    try:
        t1.start()
        assert inside.wait(5)
        t2.start()
        time.sleep(0.4)
        assert order == ["first-start"], "the second Stop started while the first was starting"
        release.set()
        t1.join(5)
        t2.join(5)
    finally:
        release.set()
        stop_pairs.clear()
    assert order == ["first-start", "first-end", "second-start"]


# ------------------------------------------------------------- the fsync window --

def test_a_stop_during_the_intent_fsync_is_not_seen_by_the_write_that_follows(hub, client,
                                                                              tmp_path):
    """TODAY'S MISS, pinned. A cockpit write journals its intent and waits for the disk, up to
    INTENT_WAIT_S, before the handler takes its stop count (`since`). A Stop pressed in that wait
    is therefore not seen: the write is not refused as `aborted_during_start`, and it goes on
    to drive the tip after the park. The fix is the owner's call; this keeps the window from growing."""
    block, release, entered = threading.Event(), threading.Event(), threading.Event()

    def sink(_line):
        if block.is_set():
            entered.set()
            release.wait(30)

    journal = Journal(tmp_path, sink=sink).start()
    try:
        with GoldenRig(hub, "fsync-window") as rig:
            door = rig.door
            tok = _acquire(door)
            door.call("loadPattern", {"token": tok, "objects": [dict(WIRE)]})
            _release(door, tok)
            client.post("/api/withdraw", json={})
            out = {}

            def write():
                t0 = time.monotonic()
                r = client.post("/api/litho/write", json=WRITE_BODY)
                out["reply"], out["took"] = r.json(), time.monotonic() - t0

            block.set()
            t = threading.Thread(target=write)
            t.start()
            assert entered.wait(10), "the write never reached its intent fsync"
            seq = hub.host._run_abort_seq
            park = client.post("/api/safe_park", json={"reason": "operator abort"}).json()
            assert park["safe_park"]["verified"] is True
            assert hub.host._run_abort_seq > seq, "the Stop did not signal the run"
            t.join(30)
            door.wait_state("idle", "engaged", "parked")
            rig.quiesce()
            verbs = [e["verb"] for e in rig.tap.snapshot()["actuation"]]
            release.set()
    finally:
        release.set()
        journal.stop()
        journal_module._running = None
    assert out["took"] >= INTENT_WAIT_S - 0.1, out
    assert out["reply"].get("reason") != "aborted_during_start", out["reply"]
    assert out["reply"].get("run_id"), out["reply"]
    assert "ARMFIRE" in verbs[verbs.index("@SAFEPARK"):], verbs


# ----------------------------------------------- Stop races inside a start's awaits --

def _hold_readxpt(rig):
    """Make the stub answer the next READXPT (the mode read a start awaits) only when told."""
    entered, release = threading.Event(), threading.Event()
    original = rig.stub._dispatch

    def dispatch(line, link):
        if line.startswith("READXPT") and not release.is_set():
            entered.set()
            release.wait(30)
        original(line, link)

    rig.stub._dispatch = dispatch
    return entered, release


def _clear_abort_flags(hub):
    """A Stop leaves the abort events set between runs; put them back as a fresh rig has them."""
    for name in ("_hw_abort", "_hw_litho_abort"):
        ev = getattr(hub.host, name, None)
        if ev is not None:
            ev.clear()


def _start_in_thread(door, method, params):
    box = {}

    def go():
        try:
            box["ok"] = door.call(method, params)
        except AssertionError as exc:
            box["refused"] = str(exc)

    t = threading.Thread(target=go)
    t.start()
    return t, box


def test_a_stop_during_a_scan_start_refuses_the_scan_and_arms_nothing(hub):
    """The scan start awaits the mode read, then compares `_run_abort_seq`
    (commands/scan_cmds.py `_scan_start_body`). A Stop signalled in that await (the function
    every Stop calls first, `signal_run_abort`) refuses the scan as `aborted_during_start`,
    before anything is armed. The stub is held, not parked, so the trace is the start alone."""
    with GoldenRig(hub, "scan-race") as rig:
        door = rig.door
        tok = _acquire(door)
        entered, release = _hold_readxpt(rig)
        t, box = _start_in_thread(door, "startScan", {"token": tok, **SCAN_CFG})
        assert entered.wait(10), "the scan start never reached its mode read"
        signal_run_abort(hub)
        release.set()
        t.join(30)
        door.wait_state("idle", "engaged", "parked")
        rig.quiesce()
        verbs = [e["verb"] for e in rig.tap.snapshot()["actuation"]]
        _release(door, tok)
        _clear_abort_flags(hub)
    assert "aborted_during_start" in box.get("refused", ""), box
    assert not {"SCANCHAIN", "WAVE", "OUTPAIR", "INPAIR"} & set(verbs), verbs


def test_a_stop_during_a_write_start_refuses_the_write_and_arms_nothing(hub):
    """The same race for a write started over FLEX: the stop count is taken first thing in
    `start_write`, and compared at the last check before the worker starts
    (commands/litho_cmds.py `_refuse_if_stopped`)."""
    with GoldenRig(hub, "write-race") as rig:
        door = rig.door
        tok = _acquire(door)
        door.call("loadPattern", {"token": tok, "objects": [dict(WIRE)]})
        entered, release = _hold_readxpt(rig)
        t, box = _start_in_thread(door, "startWrite", {"token": tok, "objects": "all",
                                                       "setpoint": 0.0, "amp_gain": 25.0})
        assert entered.wait(10), "the write start never reached its mode read"
        signal_run_abort(hub)
        release.set()
        t.join(30)
        door.wait_state("idle", "engaged", "parked")
        rig.quiesce()
        verbs = [e["verb"] for e in rig.tap.snapshot()["actuation"]]
        _release(door, tok)
        _clear_abort_flags(hub)
    assert "aborted_during_start" in box.get("refused", ""), box
    assert not {"ARMFIRE", "WAVE", "OUTPAIR", "FIRE"} & set(verbs), verbs


# ------------------------------------------------------------------ telemetry age --

def test_telemetry_on_the_twin_is_computed_on_demand_with_zero_age(hub, client):
    """getTelemetry on the twin builds its frame at the moment of the call: age_s is 0.0
    and `t` is now (board.telemetry)."""
    if hub.backend_live() == "hardware":
        pytest.skip("hub is on hardware")
    before = time.time()
    frame = board.telemetry(hub)
    assert frame["backend"] == "twin" and frame["age_s"] == 0.0
    assert before - 1.0 <= frame["t"] <= time.time() + 1.0


def test_telemetry_on_hardware_is_the_last_cached_frame_and_its_age_grows(hub):
    """On hardware getTelemetry is the polled frame (`hub._hw_telemetry`), never a fresh
    read: the values are the frame's, and age_s is the wall time since the frame's `t`."""
    with GoldenRig(hub, "telemetry-age") as rig:
        stamp = time.time() - 1.5
        hub.host._hw_telemetry = {"t": stamp, "row": None, "x_um": 1.0, "y_um": 2.0,
                             "defl_v": 0.25, "zdrive_v": 0.0, "sum_v": 5.0, "amp_v": None}
        rig.inst.set_value("Deflection", 3.0)           # the instrument moves on; the cache does not
        a = rig.door.call("getTelemetry")
        time.sleep(0.3)
        b = rig.door.call("getTelemetry")
        hub.host._hw_telemetry = None
    assert a["defl_v"] == b["defl_v"] == 0.25 and a["backend"] == "hardware"
    assert 1.4 <= a["age_s"] <= 2.2
    assert b["age_s"] > a["age_s"] + 0.2


# ------------------------------------------------------------------- the xfails --

@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "hole 1: the cockpit Abort's keeper copy is sent from inside an async handler "
    "(requests/stops.py `_browser_safe_park`), so with the event loop blocked the cockpit has "
    "no second route to the park; FLEX sends its copy from its own thread. The owner rules on the fix"))
def test_the_cockpit_abort_reaches_the_keeper_while_the_event_loop_is_blocked(hub, client,
                                                                              monkeypatch):
    from test_keeper_stop import FakeKeeper

    fake = FakeKeeper()
    monkeypatch.setenv(keeper_stop.PORT_ENV, str(fake.port))
    blocked, release = threading.Event(), threading.Event()

    async def _block():
        blocked.set()
        release.wait(60)

    try:
        with GoldenRig(hub, "hole-1") as rig:
            asyncio.run_coroutine_threadsafe(_block(), hub._loop)
            assert blocked.wait(5)
            poster = threading.Thread(
                target=lambda: client.post("/api/safe_park", json={"reason": "operator abort"}),
                daemon=True)
            poster.start()
            end = time.monotonic() + 2.0
            while not fake.lines and time.monotonic() < end:
                time.sleep(0.05)
            reached = bool(fake.lines)
            release.set()
            poster.join(30)
            rig.quiesce()
    finally:
        release.set()
        fake.close()
    assert reached, "no STOP line reached the keeper while the loop was blocked"


def test_a_silent_host_reads_disconnected_while_the_base_is_alive(hub):
    async def go():
        hub.host.stop_beat()                # the host goes silent; the base keeps running
        hub.host._last_heard = None
        hub.host.heard()
        try:
            await asyncio.sleep(board.STALE_AFTER_S + 0.7)
            return board.state(hub)
        finally:
            hub.host.start_beat()

    try:
        assert asyncio.run(go()) == "disconnected"
    finally:
        hub.host._last_heard = None
