"""The control plane (milestone 3): the commander token, the dead-man, the
engaged-idle cap, GUI preemption, the token-free fail-safe verbs, and the
single-writer claim.

Truth document: docs/FLEX_INTEGRATION_PLAN.md Sec E.1 (the hazard-based dead-man
table + the engagement cap), E.2 (one commander, where the claim is taken), B.4 (the
deadlock rule), F.2 (bounds). Wire reference: docs/FLEX_PROVIDER.md. Rig checks:
docs/FLEX_RIG_RUNBOOK.md "05".

TWO KINDS OF CASE, deliberately separated:

* **Wire cases** drive a real `zmq.REQ` client against a real provider on ephemeral
  ports, backed by a hub whose asyncio loop really runs in a background thread. The
  control verbs are the first handlers that MARSHAL TO THE LOOP, so a stubbed loop
  would test the one thing that cannot be stubbed -- the round trip a REP thread
  makes while holding no lock (plan Sec B.4).
* **Timer cases** call `control_tick(hub, now=...)` directly with an injected clock.
  No test in this file sleeps for a timer: the dead-man's floor is 5 s and the
  engagement cap's default is 900 s, and a suite that waited for either would be a
  suite nobody runs.

ZERO HARDWARE. The dead-man table is exercised on the twin, where `safe_park` runs
the simulated retract and verifies by construction.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from _im_fixtures_guard import GUARD_WRITE_KW, seed_free_air
from conftest import start_flex_provider

from afm_litho.service import control as ctl
from afm_litho.service import flex_dispatch as fd
from afm_litho.service import flex_provider as fp

zmq = pytest.importorskip("zmq", reason="pyzmq is the optional [flex] extra")


def _guard_write_body(real_hub):
    """The REST write body under the command guard (INSTRUMENT_MODEL_SPEC secs
    2/10, 13): every enabled object named explicitly (the sanctioned shape that
    replaces the old cockpit `all: true`), with `setpoint_relative` and
    `amp_gain`. The caller seeds `hub.free_air` first via `seed_free_air`.
    Naming the ids re-writes them regardless of the written set, exactly as
    `all: true` did."""
    ids = [o.name for o in real_hub.pattern.objects
           if getattr(o, "enabled", True)]
    return {"objects": ids, "setpoint_relative": 0.2, **GUARD_WRITE_KW}


# ------------------------------------------------------------------ fixtures --

class _FakeScan:
    """A scan pump task. It runs until the stop flag every pass watches (`hub._hw_abort`,
    set by an abort or a park) is raised, then reports done, as the pump does."""

    def __init__(self, hub):
        self._stop = hub.host._hw_abort

    def done(self):
        return self._stop.is_set()

    @property
    def aborted(self):
        return self._stop.is_set()


class _FakeInst:
    """The slice of TwinInstrument `safe_park` and `get_state` touch."""

    def __init__(self):
        self.withdrew = 0

    def abort_scan(self):
        pass

    def withdraw(self):
        self.withdrew += 1
        return {"z_v": None, "parked": True}


class _FakeStore:
    def __init__(self):
        self.records = []

    def record(self, kind, payload):
        self.records.append((kind, payload))
        return {"id": f"{kind}-1"}


def _twin_arc():
    """Build the twin `SoftwareARC` exactly as `Hub.__init__` does on the twin
    backend (hub.py: `Sample.from_ibw` -> `World` -> `SoftwareARC`), so the control
    plane's park verifies through the same simulated ZSensor/Deflection the real
    twin uses. Bound in `_LoopHub` to both `arc` and `hardware_arc`, matching the
    real hub's `if self.backend != "hardware": self.hardware_arc = self.arc`."""
    from afm_litho.io import read_ibw
    from afm_litho.service.deps import DEFAULT_SAMPLE, SAMPLES
    from afm_litho.twin.arc import SoftwareARC
    from afm_litho.twin.physics import Instrument, Sample, World

    path = SAMPLES / DEFAULT_SAMPLE
    sample = Sample.from_ibw(path)
    inst = Instrument.from_note(read_ibw(path).note_dict)
    return SoftwareARC(World(sample=sample, instrument=inst))


# Contact setpoint the park cases engage the twin at (V above free air). Inside the
# twin's contact band (0..1 V) but OUTSIDE park_plan's 0.5 V verify tolerance: at the
# approach default (0.2 V) an ENGAGED twin already reads ZSensor ~0.01 V and
# Deflection ~0.2 V -- both inside tolerance -- so a park "verified" there proves
# nothing (the sensors pass before any ramp). 0.8 V makes the verify pass ONLY once a
# correct-channel ramp (or the vendor fallback) has actually retracted the tip.
_ENGAGED_DEFL_V = 0.8


def _engage_twin(hub):
    """Put `hub`'s twin SoftwareARC truly in contact, the way the real twin engages:
    `TwinInstrument.approach` arms the loop-2 Z servo and steps the sim until
    deflection reaches setpoint, so the sim's OWN state is engaged (a settle `step`
    after a no-op ramp keeps it there, rather than snapping back to rest). Then assert
    the precondition that makes the park assertions non-vacuous: before the park, the
    independent verify sensors are OUTSIDE the park tolerance."""
    from afm_litho.service.command_guard import park_plan
    from afm_litho.service.passes import TwinInstrument

    TwinInstrument(hub.host.arc).approach(target_defl_v=_ENGAGED_DEFL_V)
    hub.host.engaged = True
    plan = park_plan(hub.host.instrument_model)
    zs = hub.host.arc.read("ZSensor")
    defl = hub.host.arc.read("Deflection")
    zs_ok = abs(zs - plan.park_v) <= plan.tol_v
    defl_ok = abs(defl - hub.host.free_air.value_v) <= plan.tol_v
    assert not (zs_ok and defl_ok), (
        f"twin not engaged outside park tolerance: ZSensor={zs}, Deflection={defl}")


class _LoopHub(ctl.ControlMixin):
    """A hub with arbitration, a real running loop, and exactly the attributes the
    control plane touches. Small on purpose: the point of the FakeHub pattern in this
    suite is that the surface a handler depends on is VISIBLE."""

    def __init__(self):
        from afm_litho.service.runs import RunRegistry

        self.backend = "twin"
        self.hardware_arc = None
        self.bridge_closed = threading.Event()
        self.bridge_closed_at = None
        self.fault = None
        self.engaged = False
        self.litho_thread = None
        self._scan_task = None
        self._approach_in_flight = 0
        self._scan_row = None
        self._hw_telemetry = None
        self._safe_park = {"count": 0, "active": False, "last_reason": None,
                           "verified": None, "critical": False, "transport": None,
                           "z_v": None, "at": None}
        self._scan_repeat_stop = threading.Event()
        self._hw_abort = threading.Event()
        self._hw_litho_abort = threading.Event()
        self.scan_repeat = {"enabled": False, "count": 0, "stopped_reason": None}
        self.written_accum = []
        self.written_objects = set()
        self.pattern = None
        self.hw_cal = {}
        # THE TWIN INSTRUMENT MODEL + ARC ARE PART OF THE SURFACE THE CONTROL PLANE
        # TOUCHES. Since E2 (830d401) every park -- the dead-man's, an operator
        # abort's, a FLEX safePark -- routes through the synchronous, model-driven
        # `safe_park` primitive, which reads `park_plan(hub.instrument_model)` for the
        # Z channel + park tolerance and PROVES the retract from the twin ARC's
        # ZSensor/Deflection (C1 bound the twin SoftwareARC to `hub.hardware_arc` for
        # exactly this). A bare stub could not verify, so this hub is built the way
        # the real hub builds a twin (hub.py: `load_model(platform="twin", ...)` for
        # the model; Sample -> World -> SoftwareARC bound to both `arc` and
        # `hardware_arc` -- `real_bridge` still reads a SoftwareARC as the twin) so
        # the dead-man/abort cases exercise the real park plan against real sim
        # sensors, not a hand-tuned success.
        from afm_litho.service.instrument_model import load_model
        self.instrument_model = load_model(
            platform="twin", startup_cal=None, config_path=None, overrides_log=None)
        self.arc = _twin_arc()
        self.hardware_arc = self.arc
        # A withdrawn free-air baseline, as the real twin park verifies deflection
        # against (seed_free_air overwrites it where a case needs a specific value).
        from afm_litho.service.command_guard import FreeAir
        self.free_air = FreeAir(value_v=0.0, ts=time.time(), mode="contact",
                                withdrawn=True)
        self.litho_worker = None
        self.last_park = None
        self._engaged_since = None
        self.inst = _FakeInst()
        self.store = _FakeStore()
        self.runs = RunRegistry()
        self.logs: list[str] = []
        self.frames: list[dict] = []
        self.withdrawals = 0
        self._loop = None
        self._init_control()

    # -- the two milestone-1 helpers every handler derives its answers from
    def backend_live(self):
        return "twin"

    def backend_label(self):
        return "twin"

    def require_live_backend(self):
        return None

    def next_run_id(self, kind):
        return self.runs.allocate(kind, backend="twin")

    def finish_run(self, run_id, **fields):
        return self.runs.finish(run_id, **fields)

    async def log(self, text):
        self.logs.append(str(text))

    def _log_sync(self, text):
        """The hub's log-from-a-non-coroutine path. The watcher's done-callback runs
        as a plain callback on the loop, where `await hub.log(...)` is not available
        -- and a CRITICAL line about the dead-man being down is the last thing that
        should depend on being in a coroutine."""
        self.logs.append(str(text))

    async def broadcast(self, msg):
        self.frames.append(msg)

    def logged(self, needle: str) -> bool:
        return any(needle in line for line in self.logs)

    @property
    def host(self):                # fake: host fields live on the fake itself
        return self


@pytest.fixture
def hub():
    """A `_LoopHub` whose asyncio loop really runs, on a background thread.

    The control verbs marshal to the loop (plan Sec B.4), so this is the fixture that
    makes the round trip real rather than mocked."""
    h = _LoopHub()
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True, name="test-loop")
    t.start()
    # Wait for it to actually be running: `run_coroutine_threadsafe` on a loop that
    # has not started yet schedules into the void.
    for _ in range(200):
        if loop.is_running():
            break
        time.sleep(0.005)
    h._loop = loop
    yield h
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=2.0)
    loop.close()


@pytest.fixture
def provider(hub):
    prov = start_flex_provider(fp, hub)
    yield prov
    prov.stop()


class _Client:
    def __init__(self, endpoint: str, timeout_s: float = 15.0):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
        self.sock.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
        self.sock.connect(endpoint)
        self._n = 0

    def send(self, method, params=None) -> dict:
        self._n += 1
        req = {"jsonrpc": "2.0", "method": method, "id": str(self._n)}
        if params is not None:
            req["params"] = params
        self.sock.send_string(json.dumps(req))
        return json.loads(self.sock.recv_string())

    def call(self, method, params=None):
        reply = self.send(method, params)
        assert "error" not in reply, reply
        return reply["result"]

    def close(self):
        self.sock.close(linger=0)
        self.ctx.term()


@pytest.fixture
def cmd(provider):
    c = _Client(provider.bound["command"])
    yield c
    c.close()


@pytest.fixture
def read(provider):
    c = _Client(provider.bound["read"])
    yield c
    c.close()


def _run(hub, coro):
    """Run a coroutine on the fixture hub's loop from the test thread."""
    return asyncio.run_coroutine_threadsafe(coro, hub._loop).result(10)


# ======================================== 1. acquire / release / heartbeat ======

def test_acquire_returns_a_token_and_a_clamped_deadman(cmd, hub):
    res = cmd.call("acquireControl", {"client": "flex-exp", "deadman_s": 45})
    assert isinstance(res["token"], str) and len(res["token"]) >= 16
    assert res["deadman_s"] == 45
    assert 0 < res["expires_s"] <= 45
    assert hub.control.client == "flex-exp"
    assert hub.control.kind == "script"


@pytest.mark.parametrize("asked,got", [(0, 5.0), (1, 5.0), (30, 30.0),
                                       (300, 300.0), (3600, 300.0), (45.5, 45.5)])
def test_deadman_is_clamped_never_disabled(cmd, asked, got):
    """Plan Sec E.1: default 30 s, extendable to 300 s, NEVER disabled. A NUMBER
    outside the range is clamped, not refused -- clamping fails safe, refusing just
    means the script has no dead-man while it works out why."""
    res = cmd.call("acquireControl", {"client": "c", "deadman_s": asked})
    assert res["deadman_s"] == got
    cmd.call("releaseControl", {"token": res["token"]})


@pytest.mark.parametrize("bad", ["nonsense", "30", [30], {"s": 30}, True])
def test_a_non_numeric_deadman_is_refused_not_silently_defaulted(cmd, bad):
    """Clamping a number is helping; substituting the 30 s default for a TYPO is not.
    `deadman_s: "abc"` used to come back as a perfectly ordinary 30 s token, so a
    script asking for five minutes got thirty seconds and no hint of it. (`True` is
    in the list on purpose: a bool is an int in Python, and `deadman_s: true` is not
    a duration.)"""
    reply = cmd.send("acquireControl", {"client": "c", "deadman_s": bad})
    assert reply["error"]["code"] == fd.INVALID_PARAMS
    assert "deadman_s" in reply["error"]["message"]


@pytest.mark.parametrize("bad", ["yes", "false", "0", 1, 0])
def test_withdraw_on_deadman_must_be_an_actual_bool(cmd, bad):
    """`bool("false")`, `bool("no")` and `bool("0")` are all True. This flag decides
    whether a silent script LEAVES THE TIP ON THE SURFACE, so it is never inferred
    from a string: a caller who wrote "false" would have got a withdraw, and one who
    meant it and mistyped would think they had asked."""
    reply = cmd.send("acquireControl", {"client": "c", "withdraw_on_deadman": bad})
    assert reply["error"]["code"] == fd.INVALID_PARAMS
    assert "withdraw_on_deadman" in reply["error"]["message"]


@pytest.mark.parametrize("asked", [True, False])
def test_withdraw_on_deadman_accepts_a_real_bool(cmd, asked):
    res = cmd.call("acquireControl", {"client": "c", "withdraw_on_deadman": asked})
    assert res["withdraw_on_deadman"] is asked
    cmd.call("releaseControl", {"token": res["token"]})


def test_acquire_is_refused_while_held(cmd):
    first = cmd.call("acquireControl", {"client": "one"})
    reply = cmd.send("acquireControl", {"client": "two"})
    assert reply["error"]["code"] == ctl.CONTROL_HELD == -32040
    assert "control_held" in reply["error"]["message"]
    assert reply["error"]["data"]["holder"] == "one"
    # and the incumbent is untouched
    assert cmd.call("heartbeat", {"token": first["token"]})["valid"] is True


def test_acquire_succeeds_once_the_incumbent_token_has_expired(cmd, hub):
    """An EXPIRED token is not a holder. The watcher normally clears it within a
    tick; a client arriving in between must not be told the instrument is held by
    something that has already gone silent."""
    cmd.call("acquireControl", {"client": "gone", "deadman_s": 5})
    hub.control = hub.control.__class__(**{**hub.control.__dict__,
                                           "expires_at": time.time() - 1})
    assert cmd.call("acquireControl", {"client": "fresh"})["token"]
    assert hub.control.client == "fresh"


def test_release_of_a_stale_token_is_never_an_error(cmd):
    """Plan Sec H puts `releaseControl` in a `finally`. A cleanup path that can throw
    is not a cleanup path."""
    assert cmd.call("releaseControl", {"token": "never-issued"}) == {"ok": True,
                                                                    "stale": True}
    tok = cmd.call("acquireControl", {"client": "c"})["token"]
    assert cmd.call("releaseControl", {"token": tok}) == {"ok": True, "stale": False}
    assert cmd.call("releaseControl", {"token": tok}) == {"ok": True, "stale": True}


def test_heartbeat_refreshes_the_deadman(cmd, hub):
    tok = cmd.call("acquireControl", {"client": "c", "deadman_s": 5})["token"]
    hub.touch_control(tok, now=time.time() - 4)      # pretend 4 s went by
    before = hub.control.expires_at
    res = cmd.call("heartbeat", {"token": tok})
    assert res["state"] == "idle" and res["valid"] is True
    assert hub.control.expires_at > before


def test_any_token_bearing_command_request_refreshes_the_deadman(cmd, hub):
    """Plan Sec E.1: a script that is actually commanding refreshes for free."""
    tok = cmd.call("acquireControl", {"client": "c", "deadman_s": 5})["token"]
    hub.touch_control(tok, now=time.time() - 4)
    before = hub.control.expires_at
    cmd.call("getState", {"token": tok})             # NOT a control verb
    assert hub.control.expires_at > before


def test_the_read_socket_never_refreshes_the_deadman(cmd, read, hub):
    """THE READ PLANE CANNOT AFFECT THE TIP -- and refreshing the dead-man is
    affecting the tip.

    A reviewer demonstrated it on the wire: with the refresh ungated, anything
    polling `getState` on 29181 with a crashed script's token kept that script's
    dead-man alive forever, and the retract that should have happened never would.
    Refresh is a command-plane act."""
    tok = cmd.call("acquireControl", {"client": "c", "deadman_s": 5})["token"]
    hub.touch_control(tok, now=time.time() - 4)
    before = hub.control.expires_at
    assert read.call("getState", {"token": tok}) == "idle"    # served, not refreshed
    assert read.call("getStatus", {"token": tok})["state"] == "idle"
    assert hub.control.expires_at == before


def test_an_unknown_token_never_refreshes_anybody_elses(cmd, hub):
    tok = cmd.call("acquireControl", {"client": "c", "deadman_s": 5})["token"]
    hub.touch_control(tok, now=time.time() - 4)
    before = hub.control.expires_at
    cmd.call("getState", {"token": "someone-elses"})
    assert hub.control.expires_at == before


def test_a_late_heartbeat_cannot_un_fire_the_deadman(cmd, hub):
    """The watcher clears an expired token within a tick. In that gap a late call
    must not be able to refresh it back to life: expiry is the timer's to act on, not
    a request's to undo -- otherwise a script that came back one second past its
    dead-man would keep driving a tip nobody was watching."""
    tok = cmd.call("acquireControl", {"client": "c", "deadman_s": 5})["token"]
    hub.control = hub.control.__class__(**{**hub.control.__dict__,
                                           "expires_at": time.time() - 0.5})
    assert hub.touch_control(tok) is False
    reply = cmd.send("heartbeat", {"token": tok})
    assert reply["error"]["code"] == ctl.CONTROL_REVOKED
    assert reply["error"]["data"]["reason"] == "deadman"


def test_heartbeat_without_a_token_is_control_required(cmd):
    reply = cmd.send("heartbeat", {})
    assert reply["error"]["code"] == ctl.CONTROL_REQUIRED == -32030
    assert reply["error"]["data"]["valid"] is False


def _wait_for_control_frame(hub, *, timeout_s: float = 3.0):
    """Poll `hub.frames` for the first `control` frame. The rebroadcast is SPAWNED
    (not awaited) off the heartbeat reply path, so it may land a beat after
    `cmd.call` returns -- asserting on `hub.frames` immediately would be a race."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        frames = [f for f in hub.frames if f.get("type") == "control"]
        if frames:
            return frames
        time.sleep(0.02)
    return [f for f in hub.frames if f.get("type") == "control"]


def test_an_accepted_heartbeat_rebroadcasts_control_with_fresh_expires_s(cmd, hub):
    """The cockpit countdown (ec-control.js) re-seeds off the `control` ws event, but
    that event used to fire only on acquire/release -- so a script heartbeating every
    few seconds never moved the chip, even though the server-side dead-man really was
    being reset. An ACCEPTED heartbeat must re-fire the SAME event `acquireControl`
    fires, with a fresh `expires_s`, so the countdown it drives actually reflects the
    refresh."""
    res = cmd.call("acquireControl", {"client": "c", "deadman_s": 60})
    tok = res["token"]
    hub.frames.clear()             # only care about what the heartbeat itself does

    reply = cmd.call("heartbeat", {"token": tok})
    assert reply["valid"] is True

    control_frames = _wait_for_control_frame(hub)
    assert len(control_frames) == 1
    frame = control_frames[0]
    assert frame["holder"] == "c"
    assert frame["deadman_s"] == 60
    assert 55.0 <= frame["expires_s"] <= 60.0


def test_a_rejected_heartbeat_broadcasts_nothing(cmd, hub):
    """A bad/absent token never reaches `touch_control` -- `require_token` raises
    first -- so it must never re-fire the `control` event either: a rejected
    heartbeat is not activity from the commander."""
    hub.frames.clear()

    reply = cmd.send("heartbeat", {"token": "never-issued"})
    assert reply["error"]["code"] == ctl.CONTROL_REQUIRED

    reply2 = cmd.send("heartbeat", {})
    assert reply2["error"]["code"] == ctl.CONTROL_REQUIRED

    # No task was ever spawned for a rejected heartbeat, so there is nothing to wait
    # for: an immediate check is not a race the way the accepted case is.
    assert [f for f in hub.frames if f.get("type") == "control"] == []


def test_heartbeat_does_not_block_on_a_slow_broadcast(cmd, hub):
    """Review of f8dd67c: the rebroadcast must be SPAWNED, not awaited, in the
    heartbeat reply path. `heartbeat` has only LOOP_CALL_FLOOR_S (1 s, see
    flex_provider.loop_timeout_for) to answer, and `hub.broadcast` is a sequential,
    untimed `await ws.send_text(...)` fan-out to every client -- so awaiting it
    inline would let one stalled client make the heartbeat reply late, the exact
    failure this fix exists to prevent. Proven here with a `broadcast` that sleeps
    2 s: the reply must come back long before that, and the frame must still show up
    once the slow send finally finishes."""
    tok = cmd.call("acquireControl", {"client": "c", "deadman_s": 60})["token"]
    hub.frames.clear()

    async def _slow_broadcast(msg):
        await asyncio.sleep(2.0)
        hub.frames.append(msg)

    hub.broadcast = _slow_broadcast    # instance override; the fixture hub is fresh

    start = time.time()
    reply = cmd.call("heartbeat", {"token": tok})
    elapsed = time.time() - start
    assert reply["valid"] is True
    assert elapsed < 1.0, f"heartbeat waited on the broadcast fan-out ({elapsed:.2f}s)"

    # not there yet -- the slow send is still in flight
    assert [f for f in hub.frames if f.get("type") == "control"] == []

    control_frames = _wait_for_control_frame(hub)
    assert len(control_frames) == 1
    assert control_frames[0]["holder"] == "c"


# ================================================ 2. socket separation (B.4) ====

@pytest.mark.parametrize("method", ["acquireControl", "releaseControl", "heartbeat",
                                    "abort", "safePark"])
def test_the_control_verbs_are_refused_on_the_read_socket(read, method):
    """The read plane must stay one on which NO request can move the tip -- and that
    has to be true of the socket, not just of the documentation."""
    reply = read.send(method, {"client": "c", "token": "t"})
    assert reply["error"]["code"] == fd.METHOD_NOT_FOUND
    assert "COMMAND socket only" in reply["error"]["message"]


def test_the_read_socket_still_answers_its_own_table(read):
    assert read.call("getState") == "idle"
    assert read.call("ACK") == "ACK"


def test_the_control_bounds_are_published(read):
    """The FLEX driver derives each call's timeout from `method_bounds` (plan Sec
    F.2), so the control verbs have to be IN it -- and the provider must not wait on
    the loop for longer than it published."""
    bounds = read.call("getCapabilities")["method_bounds"]
    assert bounds["acquireControl"] == 0.05
    assert bounds["abort"] == bounds["safePark"] == 10.0
    assert fp.loop_timeout_for("abort") <= 10.0
    assert fp.loop_timeout_for("acquireControl") >= 1.0    # the scheduling floor


def test_get_status_reports_whether_the_deadman_is_armed(cmd, read, hub):
    """A dead watcher looks exactly like a healthy one with nothing to do, so the
    only way a script can know the timer governing it is alive is to be told."""
    assert read.call("getStatus")["deadman_armed"] is False    # none started here
    hub.control_watch = _run(hub, _start_watch(hub))
    try:
        assert read.call("getStatus")["deadman_armed"] is True
        cmd.call("acquireControl", {"client": "c"})
        assert read.call("getStatus")["commander"]["deadman_armed"] is True
    finally:
        _run(hub, _stop_watch(hub))
    assert read.call("getStatus")["deadman_armed"] is False


async def _start_watch(hub):
    return ctl.start_control_watch(hub, interval=0.05)


async def _stop_watch(hub):
    ctl.stop_control_watch(hub)


@pytest.mark.time_sensitive
def test_a_watcher_that_dies_re_arms_itself(hub, monkeypatch):
    """The watcher IS the dead-man. If it ends and nothing restarts it, every token
    in the process is unpoliced and nothing on the wire says so -- so it re-arms, and
    says CRITICAL each time it has to."""
    boom = {"n": 0}

    async def _explode(h, **kw):
        boom["n"] += 1
        raise RuntimeError("tick exploded")

    monkeypatch.setattr(ctl, "control_tick", _explode)
    first = _run(hub, _start_watch(hub))
    try:
        for _ in range(300):                      # wait for it to die AND come back
            cur = getattr(hub, "control_watch", None)
            if cur is not None and cur is not first:
                break
            time.sleep(0.02)
        assert boom["n"] >= 1, "the patched tick never ran"
        assert hub.control_watch is not first, "the watcher did not re-arm"
        assert ctl.deadman_armed(hub) is True
        assert hub.logged("the control watcher STOPPED")
    finally:
        monkeypatch.undo()
        _run(hub, _stop_watch(hub))


def test_a_deliberate_stop_is_not_re_armed(hub):
    """The shutdown path must not resurrect the task it just cancelled."""
    _run(hub, _start_watch(hub))
    assert ctl.deadman_armed(hub) is True
    _run(hub, _stop_watch(hub))
    time.sleep(0.3)
    assert hub.control_watch is None
    assert ctl.deadman_armed(hub) is False


def test_get_status_reports_the_commander_without_the_token(cmd, read):
    assert read.call("getStatus")["commander"] is None
    res = cmd.call("acquireControl", {"client": "flex-exp", "deadman_s": 60})
    commander = read.call("getStatus")["commander"]
    assert commander["holder"] == "flex-exp" and commander["kind"] == "script"
    assert 0 < commander["expires_s"] <= 60
    # THE TOKEN IS NEVER ON THE READ PLANE: any observer may poll it, and a token
    # there would hand control to whoever polls fastest.
    assert res["token"] not in json.dumps(read.call("getStatus"))


# ======================================= 3. the token-free fail-safe verbs ======

def test_abort_is_token_free_and_retracts(cmd, hub):
    """An emergency stop must never be blocked by arbitration -- from EITHER side."""
    cmd.call("acquireControl", {"client": "someone-else"})
    _engage_twin(hub)
    hub.host._scan_task = _FakeScan(hub)
    res = cmd.call("abort")                       # no token at all
    assert res["ok"] is True
    assert res["state"] == "parked"
    assert res["safe_park"]["verified"] is True
    assert hub.host._scan_task.aborted is True
    # E2: the twin retract no longer goes through `inst.withdraw` -- it runs the
    # synchronous, sensor-verifying `safe_park` primitive (an ARC ramp of the model's
    # Z channel), which STOPS AND RETRACTS. Prove the retract ACTUALLY happened -- not
    # merely that a verb returned -- from the primitive's own stored evidence: a
    # sensor-verified retract via a real ramp, stored on `hub.last_park`.
    assert hub.host.last_park is not None
    assert hub.host.last_park.verified is True and hub.host.last_park.method == "abort"
    assert hub.host.engaged is False                   # a verified retract clears the gate


def test_abort_on_a_running_scan_leaves_a_retrievable_aborted_result(cmd, hub):
    """Plan Sec E.3: the run terminates, the tip retracts, and the terminal record
    stays fetchable with `aborted: true` -- "the files are whole" and "the run
    stopped early" are different questions (plan Sec A.1 rule 6)."""
    run_id = hub.next_run_id("scan")
    hub.host._scan_task = _FakeScan(hub)
    assert cmd.call("getState") == "scanning"
    cmd.call("abort", {"scope": "scan"})
    # the pump would close its own record; here we close it the way the pump does
    hub.runs.finish(run_id, completed=True, aborted=True, backend="twin")
    assert cmd.call("getState") == "parked"
    res = cmd.call("getScanResult", {"run_id": run_id})
    assert res["aborted"] is True and res["completed"] is True


def test_safe_park_returns_the_result_dict(cmd, hub):
    hub.host.engaged = True
    res = cmd.call("safePark", {"reason": "operator says so"})
    assert res["ok"] is True
    assert res["reason"] == "operator says so"
    assert res["verified"] is True and res["critical"] is False
    assert res["backend"] == "twin" and res["transport"] == "twin"
    assert hub.host._safe_park["active"] is True


def test_abort_never_calls_the_non_retracting_arc_abort(cmd, hub):
    """Plan Sec J risk 1: `arc.abort()` routes to a COM `AFM_Abort()` that stops
    WITHOUT retracting. The provider must reach the instrument only through
    `safe_park`, and this asserts the withdraw actually happened rather than
    trusting the call graph."""
    _engage_twin(hub)
    hub.host._scan_task = _FakeScan(hub)
    cmd.call("abort")
    # The non-retracting COM `arc.abort()` (AFM_Abort) is never on this path. Since E2
    # the retract runs the sync `safe_park` primitive, which STOPS AND RETRACTS; prove
    # the retract actually happened -- rather than trusting the call graph -- from the
    # primitive's own evidence: a sensor-verified retract via a real ramp.
    assert hub.host.last_park is not None
    assert hub.host.last_park.verified is True and hub.host.last_park.method == "abort"


# =========================================== 4. the dead-man table (Sec E.1) ====
#
# One case per row, driven with an INJECTED clock -- no test here waits for a timer.

def _acquire(hub, *, deadman_s=5.0, withdraw_on_deadman=False, kind="script"):
    return hub.acquire_control("flex-exp", kind, deadman_s=deadman_s,
                               withdraw_on_deadman=withdraw_on_deadman)


def _expire(hub, rec):
    """The instant just past this token's expiry."""
    return rec.expires_at + 0.001


def test_deadman_row1_in_flight_scan_aborts_and_parks(hub):
    """`approaching | scanning | writing | measuring` -> cooperative abort +
    safe_park + verified retract, state `parked`."""
    rec = _acquire(hub)
    _engage_twin(hub)
    hub.host._scan_task = _FakeScan(hub)
    acts = _run(hub, ctl.control_tick(hub, now=_expire(hub, rec)))
    assert [a["action"] for a in acts] == ["safe_park"]
    assert acts[0]["state"] == "scanning"
    assert acts[0]["safe_park"]["reason"] == "deadman"
    assert hub.host._scan_task.aborted is True
    # E2: the dead-man's retract runs the sync `safe_park` primitive (ARC ramp), not
    # `inst.withdraw`; prove the verified retract through the primitive's stored result.
    assert hub.host.last_park is not None
    assert hub.host.last_park.verified is True and hub.host.last_park.method == "abort"
    assert fd.get_state(hub) == "parked"
    assert hub.control is None
    assert hub.logged("DEAD-MAN")


def test_deadman_row1_covers_a_write_in_flight(hub):
    class _Thread:
        def is_alive(self):
            return True

    rec = _acquire(hub)
    hub.host.litho_thread = _Thread()
    acts = _run(hub, ctl.control_tick(hub, now=_expire(hub, rec)))
    assert acts[0]["action"] == "safe_park" and acts[0]["state"] == "writing"
    assert hub.host._safe_park["last_reason"] == "deadman"


def test_deadman_row2_engaged_does_not_withdraw_by_default(hub):
    """The DEFAULT, and the row the plan argues hardest for: a reflex withdraw is
    itself a disruption -- it abandons an engagement the operator may be re-using,
    and on a Cypher it can wedge the machine (risk 7). So: WARN, broadcast, release
    the token, and LEAVE THE TIP."""
    rec = _acquire(hub, withdraw_on_deadman=False)
    hub.host.engaged = True
    acts = _run(hub, ctl.control_tick(hub, now=_expire(hub, rec)))
    assert [a["action"] for a in acts] == ["warn"]
    assert hub.host.engaged is True
    assert hub.host.inst.withdrew == 0
    assert hub.control is None                      # the token IS released
    assert hub.logged("WARNING DEAD-MAN")
    assert any(f.get("type") == "control" for f in hub.frames)


def test_deadman_row2_engaged_withdraws_when_the_holder_asked(hub):
    rec = _acquire(hub, withdraw_on_deadman=True)
    hub.host.engaged = True
    acts = _run(hub, ctl.control_tick(hub, now=_expire(hub, rec)))
    assert [a["action"] for a in acts] == ["withdraw"]
    assert hub.host.inst.withdrew == 1
    assert hub.host.engaged is False
    assert hub.control is None


def test_deadman_row3_idle_releases_only(hub):
    rec = _acquire(hub)
    acts = _run(hub, ctl.control_tick(hub, now=_expire(hub, rec)))
    assert [a["action"] for a in acts] == ["release"]
    assert acts[0]["state"] == "idle"
    assert hub.host.inst.withdrew == 0
    assert hub.control is None


def test_deadman_row3_parked_releases_only(hub):
    rec = _acquire(hub)
    hub.host._safe_park["active"] = True
    acts = _run(hub, ctl.control_tick(hub, now=_expire(hub, rec)))
    assert [a["action"] for a in acts] == ["release"]
    assert acts[0]["state"] == "parked"
    assert hub.host.inst.withdrew == 0


def test_a_live_token_is_not_touched(hub):
    _acquire(hub, deadman_s=300)
    hub.host.engaged = True
    assert _run(hub, ctl.control_tick(hub, now=time.time())) == []
    assert hub.control is not None
    assert hub.host.inst.withdrew == 0


def test_operator_control_has_no_deadman(hub):
    """Plan Sec E.1: the human is present. Only the idle cap applies to them."""
    rec = _acquire(hub, kind="operator")
    assert rec.deadman_s is None and rec.expires_at is None
    hub.host.engaged = True
    acts = _run(hub, ctl.control_tick(hub, now=time.time() + 100_000))
    assert [a["action"] for a in acts] == []          # no dead-man ...
    assert hub.control is not None


# =================================== 5. the engaged-idle cap (Sec E.1) ==========

def test_engaged_idle_withdraws_with_no_token_at_all(hub, monkeypatch):
    """Independent of arbitration: "a tip has been left on a surface unattended" is a
    different failure from "a script stopped talking", and the dead-man's gentle
    default is only defensible because THIS exists."""
    monkeypatch.setenv("AFM_LITHO_ENGAGED_IDLE_MAX_S", "10")
    hub.host.engaged = True
    t0 = time.time()
    assert _run(hub, ctl.control_tick(hub, now=t0)) == []      # starts the clock
    assert hub.host._engaged_since == t0
    assert _run(hub, ctl.control_tick(hub, now=t0 + 9)) == []  # not yet
    acts = _run(hub, ctl.control_tick(hub, now=t0 + 11))
    assert [a["action"] for a in acts] == ["engaged_idle_withdraw"]
    assert hub.host.inst.withdrew == 1
    assert hub.logged("ENGAGED-IDLE LIMIT")
    assert any(f.get("type") == "engaged_idle_withdraw" for f in hub.frames)


def test_engaged_idle_applies_to_an_operator_too(hub, monkeypatch):
    monkeypatch.setenv("AFM_LITHO_ENGAGED_IDLE_MAX_S", "10")
    _acquire(hub, kind="operator")
    hub.host.engaged = True
    t0 = time.time()
    _run(hub, ctl.control_tick(hub, now=t0))
    acts = _run(hub, ctl.control_tick(hub, now=t0 + 11))
    assert [a["action"] for a in acts] == ["engaged_idle_withdraw"]


def test_a_run_in_flight_never_counts_toward_the_idle_cap(hub, monkeypatch):
    monkeypatch.setenv("AFM_LITHO_ENGAGED_IDLE_MAX_S", "10")
    hub.host.engaged = True
    t0 = time.time()
    _run(hub, ctl.control_tick(hub, now=t0))
    hub.host._scan_task = _FakeScan(hub)          # a long scan starts
    assert _run(hub, ctl.control_tick(hub, now=t0 + 11)) == []
    assert hub.host._engaged_since is None           # the clock RESET, not paused
    assert hub.host.inst.withdrew == 0


@pytest.mark.parametrize("env,expected", [
    (None, 900.0),          # the default
    ("60", 60.0),           # tighter: honoured
    ("3600", 900.0),        # LOOSER: clamped back to the default
    ("0", 900.0),           # "disable it": refused
    ("-5", 900.0),
    ("off", 900.0),         # unparseable
])
def test_the_idle_cap_can_only_be_tightened(monkeypatch, env, expected):
    """Plan Sec E.1 calls this timer NON-OPTIONAL. The environment may tighten it and
    nothing else -- there is no value that switches it off."""
    if env is None:
        monkeypatch.delenv("AFM_LITHO_ENGAGED_IDLE_MAX_S", raising=False)
    else:
        monkeypatch.setenv("AFM_LITHO_ENGAGED_IDLE_MAX_S", env)
    from afm_litho.service.host import deadman
    assert deadman.engaged_idle_max_s() == expected


# =========================================== 6. the watcher's own lifecycle =====

def test_the_watcher_starts_and_stops_with_the_app():
    """A dead watcher is a tip with no dead-man, so its lifecycle is asserted rather
    than assumed."""
    from fastapi.testclient import TestClient

    from afm_litho.service.app import app
    from afm_litho.service.hub import hub as real_hub

    # This client's startup repoints `hub._loop` at its own loop and its shutdown
    # CLOSES that loop, so the shared hub would be left holding a dead one for every
    # later case (a `run_coroutine_threadsafe` against it raises "Event loop is
    # closed"). Borrowed, then given back.
    prev_loop = real_hub._loop
    try:
        with TestClient(app) as c:
            c.get("/api/control")
            task = real_hub.control_watch
            assert task is not None and not task.done()
    finally:
        real_hub._loop = prev_loop
    assert real_hub.control_watch is None      # cancelled on shutdown


def test_the_watcher_survives_a_tick_that_raises(hub, monkeypatch):
    """One bad tick must not stop the timers. `control_tick` swallows and logs."""
    async def _boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(ctl, "_deadman_tick", _boom)
    hub.host.engaged = True
    assert _run(hub, ctl.control_tick(hub, now=time.time())) == []
    assert hub.logged("dead-man tick failed")


# ================================= 7. GUI preemption over HTTP (Sec E.2) ========

@pytest.fixture(scope="module")
def http(_flex_provider_off):
    """A TestClient on the REAL app, for the arbitration middleware. Module-scoped
    like the shared `client` fixture; each case claims and releases its own token."""

    from fastapi.testclient import TestClient

    from afm_litho.service.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _drain_the_process_global_hub():
    """Some cases below POST tip-mutating routes on the SHARED hub -- the preemption
    test hits every path in the curated set by design, which really does start a twin
    write and a twin scan. Left running, those leak into the next case as a live
    `litho_thread` or a held run claim, and the next case's write is refused for
    reasons that have nothing to do with what it tests.

    So: stop and join after EVERY case, bounded. A worker still stuck past the
    timeout is left alone rather than raised on -- this fixture must never be the
    thing that hangs the suite.
    """
    yield
    from afm_litho.service.hub import hub as real_hub

    from afm_litho.service.twin_plugin import abort as abort_twin
    real_hub.host.inst.abort_scan()
    abort_twin(real_hub, "test cleanup")
    ev = getattr(real_hub.host, "_hw_litho_abort", None)
    if ev is not None:
        ev.set()
    thread = getattr(real_hub.host, "litho_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=20)
    if ev is not None:
        ev.clear()
    task = getattr(real_hub.host, "_scan_task", None)
    loop = getattr(real_hub, "_loop", None)
    if task is not None and not task.done() and loop is not None and loop.is_running():
        async def _cancel_and_await():
            task.cancel()
            try:
                await task
            except BaseException:                             # noqa: BLE001,S110
                pass

        try:
            asyncio.run_coroutine_threadsafe(_cancel_and_await(), loop).result(5)
        except Exception:                                     # noqa: BLE001,S110
            pass
    # the claims are released by the pumps; clear any that outlived their run so one
    # stuck worker cannot fail every later case with `busy`.
    for _ in range(100):
        if real_hub._run_claim is None:
            break
        time.sleep(0.05)
    real_hub._run_claim = None


@pytest.fixture
def script_token(http):
    """A script token on the process-global hub, cleaned up after the case."""
    from afm_litho.service.hub import hub as real_hub

    real_hub.control = None
    rec = real_hub.acquire_control("flex-exp", "script", deadman_s=300)
    yield rec
    real_hub.control = None


def test_take_control_preempts_a_script(http, script_token):
    res = http.post("/api/control/take", json={}).json()
    assert res["preempted"] == "flex-exp"
    assert res["control"]["kind"] == "operator"
    assert res["control"]["expires_s"] is None      # no dead-man for a human
    http.post("/api/control/release", json={})


def test_a_revoked_token_gets_control_revoked_not_control_required(http,
                                                                  script_token):
    """The difference matters to a script: -32011 means "a human took it" (do NOT
    just re-acquire), -32030 means "we have never heard of that"."""
    from afm_litho.service.hub import hub as real_hub

    http.post("/api/control/take", json={})
    with pytest.raises(ctl.ControlRevoked):
        real_hub.require_token(script_token.token)
    with pytest.raises(ctl.ControlRequired):
        real_hub.require_token("a-token-we-never-issued")
    http.post("/api/control/release", json={})


@pytest.mark.parametrize("path", sorted(ctl.TIP_MUTATING))
def test_every_tip_mutating_route_preempts(http, script_token, path):
    """The curated set (plan Sec E.2): a human who reaches for one of these has
    already decided. The route's own outcome is irrelevant -- what is asserted is
    that arbitration moved."""
    from afm_litho.service.hub import hub as real_hub

    assert real_hub.control is not None
    http.post(path, json={})
    assert real_hub.control is None, f"{path} did not preempt"


@pytest.mark.parametrize("path", ["/api/speed/8"])
def test_an_ordinary_post_never_touches_arbitration(http, script_token, path):
    """THE FAILURE THIS PRECISION EXISTS TO PREVENT: the cockpit POSTs ordinary UI
    settings (such as the twin speed) all the time, so a blanket "any mutation preempts"
    would let a bystander moving the mouse across the panel revoke an overnight
    token (plan Sec E.2 rejects that draft by name)."""
    from afm_litho.service.hub import hub as real_hub

    http.post(path, json={"nothing": True})
    assert real_hub.control is script_token


@pytest.mark.parametrize("path", ["/api/safe_park", "/api/scan/abort"])
def test_the_abort_routes_are_token_free_and_do_not_preempt(http, script_token,
                                                            path):
    """An emergency stop must not also be an arbitration event: whoever was driving
    is still driving when the tip is safe again."""
    from afm_litho.service.hub import hub as real_hub

    http.post(path, json={})
    assert real_hub.control is script_token


def test_get_control_reports_the_holder_without_the_token(http, script_token):
    body = http.get("/api/control").json()
    assert body["control"]["holder"] == "flex-exp"
    assert script_token.token not in json.dumps(body)


# ===================== 7b. the dead-man on the REAL hub, end to end ============

def test_the_deadman_parks_a_real_twin_scan(http):
    """One case with NOTHING faked: the real hub, a real twin scan, the real
    `safe_park`.

    The table above is exercised on a hand-built hub, where every row can be posed
    exactly. This is the other half -- proof that the pieces are wired to each other
    on the object the service actually runs: a scan really in flight, a token that
    really expires, `safe_park` really firing, and a terminal record a script can
    still fetch afterwards with `aborted: true`.
    """
    from afm_litho.service import flex_dispatch as _fd
    from afm_litho.service.hub import hub as real_hub

    real_hub.control = None
    real_hub.host._safe_park["active"] = False
    r = http.post("/api/scan/start",
                  json={"size_um": 5, "pixels": 32, "lines": 32,
                        "line_rate_hz": 20.0}).json()
    assert r.get("ok"), r
    run_id = r["run_id"]
    assert _fd.get_state(real_hub) == "scanning"

    rec = real_hub.acquire_control("flex-exp", "script", deadman_s=5)
    # Expire it by fiat rather than by waiting: the dead-man's floor is 5 s, and a
    # suite that waits for a timer is a suite nobody runs.
    real_hub.control = rec.__class__(**{**rec.__dict__,
                                        "expires_at": time.time() - 0.001})
    loop = real_hub._loop
    assert loop is not None and not loop.is_closed(), \
        "the shared hub is holding a dead loop -- see the nested-TestClient note"
    acts = asyncio.run_coroutine_threadsafe(
        ctl.control_tick(real_hub), loop).result(30)

    assert [a["action"] for a in acts] == ["safe_park"], acts
    assert acts[0]["state"] == "scanning"
    assert acts[0]["safe_park"]["reason"] == "deadman"
    assert real_hub.control is None
    assert real_hub.host._safe_park["last_reason"] == "deadman"
    # The cooperative scan abort needs a loop tick to wind the sim `_scan_task` down;
    # `get_state` honestly reports "scanning" until that task is done, so poll briefly
    # for "parked" rather than racing the wind-down (the run-record poll below waits on
    # the same completion). safe_park itself has already verified the retract.
    for _ in range(200):
        if _fd.get_state(real_hub) == "parked":
            break
        time.sleep(0.02)
    assert _fd.get_state(real_hub) == "parked"

    # the run is still fetchable, and says it stopped early
    for _ in range(200):
        http.get("/api/state")
        record = real_hub.runs.get(run_id)
        if record is not None and record.finished is not None:
            break
        time.sleep(0.05)
    record = real_hub.runs.get(run_id)
    assert record is not None and record.aborted is True, record
    real_hub.host._safe_park["active"] = False


# ============================ 8. the single-writer claim (Sec E.2 race) =========

def test_two_concurrent_litho_writes_start_exactly_one(http):
    """THE REVIEWER-FOUND RACE. `POST /api/litho/write` checks
    `litho_thread.is_alive()` and then AWAITS the AC-guard crosspoint read before
    starting the thread, so two requests interleaving in that window both passed the
    check and both drove the tip. A script issues requests faster than a human can
    click, which is what made a latent race worth fixing now."""
    from afm_litho.service.hub import hub as real_hub

    http.post("/api/litho/load", json={"which": "demo"})
    real_hub.host.written_objects = set()
    seed_free_air(real_hub)
    body = _guard_write_body(real_hub)

    results: list[dict] = []
    lock = threading.Lock()

    def _fire():
        r = http.post("/api/litho/write", json=body).json()
        with lock:
            results.append(r)

    threads = [threading.Thread(target=_fire) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    started = [r for r in results if r.get("ok")]
    refused = [r for r in results if not r.get("ok")]
    assert len(started) == 1, results
    assert len(refused) == 1, results
    # Either refusal is correct -- the claim ("busy") or the older is_alive() check
    # ("write already running") -- but it must be a REFUSAL, never a second write.
    assert refused[0].get("error") in ("busy", "write already running")
    if refused[0].get("error") == "busy":
        assert refused[0]["run_id"] == started[0]["run_id"]

    thread = real_hub.host.litho_thread
    if thread is not None:
        thread.join(timeout=30)


def test_a_later_write_is_never_refused_by_a_finished_run(http):
    """A leaked claim would refuse every later start for the life of the process.

    Asserted BEHAVIOURALLY -- "can the next write start?" -- rather than by peeking
    at the slot, because there are two ways it stays clean and both count: the pump
    releases the claim at pass end, and `claim_run` reclaims a handed-over claim
    whose worker is demonstrably gone. The second matters because a `finally` is a
    promise the interpreter keeps only if the coroutine runs again, and a pump whose
    loop is torn down under it never gets that chance.
    """
    from afm_litho.service.hub import hub as real_hub

    http.post("/api/litho/load", json={"which": "demo"})
    real_hub.host.written_objects = set()
    seed_free_air(real_hub)
    first = http.post("/api/litho/write", json=_guard_write_body(real_hub)).json()
    assert first.get("ok"), first
    thread = real_hub.host.litho_thread
    if thread is not None:
        thread.join(timeout=30)
    # Drive the loop while waiting: under TestClient the service loop only runs
    # while a request is in flight, so a bare sleep would never let the pump reach
    # its own release.
    for _ in range(100):
        http.get("/api/state")
        if real_hub._run_claim is None:
            break
        time.sleep(0.05)
    real_hub.host.written_objects = set()
    seed_free_air(real_hub)
    second = http.post("/api/litho/write", json=_guard_write_body(real_hub)).json()
    assert second.get("error") != "busy", second


def test_an_unstarted_worker_never_reads_as_stale():
    """The claim's staleness is anchored to ITS OWN worker, and a `Thread` that was
    created but never started is NOT finished.

    The earlier version asked `hub.litho_thread.is_alive()`, which answers False in
    that gap -- so a second writer could take the claim between the thread being
    constructed and started, and only the statement ordering in two files stood
    between that and two writes driving the tip."""
    from afm_litho.service.hub import hub as real_hub

    real_hub._run_claim = None
    claim = real_hub.claim_run("write")
    unstarted = threading.Thread(target=lambda: None)      # created, never started
    claim.hand_over("write-99", unstarted)
    assert claim.worker_alive() is True
    with pytest.raises(ctl.RunBusy):
        real_hub.claim_run("write")
    unstarted.start()
    unstarted.join(5)
    assert claim.worker_alive() is False
    real_hub.claim_run("write")                 # a finished worker IS reclaimable
    real_hub._run_claim = None


def test_claim_run_is_synchronous_and_exclusive():
    """The unit property the endpoints depend on: no await between the check and the
    set, and identity-checked release so a late finisher cannot free a newer claim."""
    from afm_litho.service.hub import hub as real_hub

    real_hub._run_claim = None
    first = real_hub.claim_run("write")
    with pytest.raises(ctl.RunBusy) as exc:
        real_hub.claim_run("scan")
    assert exc.value.kind == "write"
    real_hub.release_claim(object())          # a stranger's release does nothing
    assert real_hub._run_claim is first
    real_hub.release_claim(first)
    assert real_hub._run_claim is None
    second = real_hub.claim_run("scan")       # free again
    real_hub.release_claim(first)             # the OLD claim cannot free the new one
    assert real_hub._run_claim is second
    real_hub.release_claim(second)


# Two short strokes, drawn the way this lab draws: 1 SVG unit = 1 nm. A couple of
# seconds of simulated write instead of the default routing's eight minutes.
_TWO_STROKE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg">'
    '<path d="M 0 0 L 3000 0"/>'
    '<path d="M 0 1500 L 3000 1500"/>'
    "</svg>"
)


def test_back_to_back_writes_each_persist_their_own_run(http, monkeypatch):
    """THE PERSISTENCE CLOBBER, as a regression guard.

    The pump releases the claim at PASS END so the next start is not refused for the
    length of a file write -- but everything it persists afterwards used to be read
    from the hub, and a starting write resets exactly those accumulators
    (`_litho_pump_body` clears `written_accum`, `litho_write` clears
    `hw_litho_capture`). A reviewer started run 2 inside run 1's persistence window
    and run 1's session record went to disk with an empty `written`: a run that
    really happened, and a file that says it wrote nothing.

    So: two writes back to back, and BOTH session records must carry their own
    objects. The pump snapshots before releasing; nothing below the release reads a
    hub field a later run can reset.

    THE DESIGN IS TINY ON PURPOSE. This used to load the default `auto` routing --
    eight objects and a 477-SECOND simulated pass -- and then poll 180 s for each of
    two writes to persist. On a loaded machine that is a coin flip, and the case
    failed for being slow rather than for being wrong (the reviewer measured it at
    roughly one full-suite run in two). Two short strokes exercise exactly the same
    path: run 1 persists while run 2 starts, which is the whole property.
    """
    from afm_litho.service.hub import hub as real_hub

    # The load needs a scan frame to place the design in; set its own, not an earlier test's.
    monkeypatch.setattr(real_hub.host, "last_scan_config", {"size_um": 30.0})
    loaded = http.post("/api/litho/load", json={"source": _TWO_STROKE_SVG}).json()
    assert "error" not in loaded, loaded
    ids = []
    for _ in range(2):
        assert _write_idle(http, real_hub), "a previous write never finished"
        real_hub.host.written_objects = set()
        seed_free_air(real_hub)
        r = http.post("/api/litho/write", json=_guard_write_body(real_hub)).json()
        assert r.get("ok"), r
        ids.append(r["run_id"])
        assert _run_persisted(http, real_hub, ids[-1]), "the pump never persisted"

    assert len(ids) == 2 and ids[0] != ids[1]
    for run_id in ids:
        rec = real_hub.runs.get(run_id)
        assert rec is not None and rec.finished is not None, run_id
        submitted = (rec.stats or {}).get("objects_submitted")
        # THE ASSERTION THAT WOULD HAVE CAUGHT IT, on the artifact that was wrong:
        # the SESSION RECORD on disk, and the run's own object count.
        assert (rec.stats or {}).get("objects") == submitted, rec.stats
        session = real_hub.store.get(rec.record_id)
        assert session is not None, rec.record_id
        assert len(session.get("written") or []) == submitted, (
            f"{run_id} persisted {len(session.get('written') or [])} of "
            f"{submitted} objects -- a later run cleared the accumulator")


def _write_idle(http, real_hub, timeout: float = 180.0) -> bool:
    """Wait until the previous write is REALLY finished, DRIVING THE LOOP while
    waiting.

    Both halves matter under TestClient: the service loop only runs while a request
    is in flight, so a bare sleep would never let a pump finish -- and a neighbouring
    case's write can still be going when this one starts, which has nothing to do
    with what is under test.

    "REALLY finished" is BOTH conditions, and waiting on only the first is what made
    this case load-sensitive in the full suite. The worker thread dying does not end
    the run: the PUMP still has to drain the worker's last messages and release the
    single-writer claim, and until it does, the next `POST /api/litho/write` is
    correctly refused `busy` -- so the case failed on its own precondition, not on
    the property it exists to guard. The claim is the honest signal, because it is
    exactly what the next start path checks."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        thread = getattr(real_hub.host, "litho_thread", None)
        if ((thread is None or not thread.is_alive())
                and getattr(real_hub, "_run_claim", None) is None):
            return True
        http.get("/api/state")
        time.sleep(0.05)
    return False


def _run_persisted(http, real_hub, run_id: str, timeout: float = 180.0) -> bool:
    """Wait until `run_id`'s terminal record is closed -- i.e. the pump has been
    through its persistence block, which is the thing under test."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        rec = real_hub.runs.get(run_id)
        if rec is not None and rec.finished is not None:
            return True
        http.get("/api/state")
        time.sleep(0.05)
    return False


def test_busy_maps_to_the_same_vocabulary_on_both_surfaces():
    """HTTP says `{"error": "busy", "run_id": ...}`; the wire reserves -32020 with
    the same `run_id` in `data` -- one vocabulary, so a mixed GUI/script session
    reads one story (plan Sec F.1)."""
    from afm_litho.service.hub import hub as real_hub

    body = real_hub.busy_body(ctl.RunBusy("write", "write-9"))
    assert body == {"error": "busy", "reason": "busy", "kind": "write",
                    "run_id": "write-9"}
    assert fd.BUSY == -32020
