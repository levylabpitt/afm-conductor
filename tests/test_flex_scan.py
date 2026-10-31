"""Scan control on the wire (slice 07): `getMode`, `setMode`, `startApproach`,
`withdraw`, `setZGain`, `startScan`, `setContinuous`.

Truth document: docs/FLEX_INTEGRATION_PLAN.md Sec A (the instrument-operations
table), C (the `service/commands/` layer these verbs and the cockpit's endpoints
share), E.2 (token, single-writer claim), F.1/F.3 (start+poll, busy, the state
machine), F.2 (bounds). Wire reference: docs/FLEX_PROVIDER.md. Rig checks:
docs/FLEX_RIG_RUNBOOK.md "07 Scripted imaging".

REAL SOCKETS, REAL HUB, REAL LOOP. The provider binds on ephemeral ports against the
PROCESS hub -- the same object the FastAPI app drives -- so `startApproach` really
engages the twin, `startScan` really runs a raster through the pump, and
`getScanResult` really reads the record the pump closed after persisting its files.
A fake hub would test the handler's plumbing and none of the things that have
actually broken here: a claim not released, an in-flight marker not cleared, a run
record nobody closes.

ZERO HARDWARE. Every case runs on the twin backend. The two hardware-only rules
(`setZGain` refused off-hardware, `startScan` refusing a setpoint without `engage`)
are asserted against the service layer with a hub stub that merely SAYS it is
hardware -- they are refusals, so nothing is driven either way.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import ClassVar

import pytest
from conftest import start_flex_provider

from afm_litho.service import flex_dispatch as fd
from afm_litho.service import flex_provider as fp
from afm_litho.service.host import scan_cmds
from afm_litho.service.commands.errors import CommandError

zmq = pytest.importorskip("zmq", reason="pyzmq is the optional [flex] extra")

# Long enough that no test's polling loop can let the dead-man fire mid-scan: reads
# ride the READ socket, which deliberately does NOT refresh it (that is the rule that
# stops a crashed script being kept alive by anything polling `getState`).
DEADMAN_S = 300

# A raster small enough to finish in a couple of seconds on the twin, and still a
# real frame: the pump streams it, persists it and closes its run record.
SMALL_SCAN = {"size_um": 2.0, "pixels": 16, "lines": 8, "line_rate_hz": 40.0}


class _Arc:
    """The one method `set_mode` calls on the hardware ARC. Records, drives nothing."""

    def __init__(self):
        self.modes = []

    def set_imaging_mode(self, mode):
        self.modes.append(mode)
        return 0


# ------------------------------------------------------------------ fixtures --

class _Client:
    def __init__(self, endpoint: str, timeout_s: float = 30.0):
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


@pytest.fixture(scope="module")
def hub(client):
    """The PROCESS hub, with the app running (so `hub._loop` is a live loop and the
    control watcher, the run registry and the twin instrument are all real).

    Depends on the `client` fixture purely for that startup; the HTTP door is used by
    two cases here and is otherwise incidental."""
    from afm_litho.service.hub import hub as real_hub
    return real_hub


@pytest.fixture(scope="module")
def provider(hub):
    prov = start_flex_provider(fp, hub)
    yield prov
    prov.stop()


@pytest.fixture(scope="module")
def cmd(provider):
    c = _Client(provider.bound["command"])
    yield c
    c.close()


@pytest.fixture(scope="module")
def read(provider):
    c = _Client(provider.bound["read"])
    yield c
    c.close()


@pytest.fixture(autouse=True)
def _clean(hub, cmd, litho_ready_hub):
    """Leave the instrument the way we found it.

    The hub is a process global shared with every other test module, so a case that
    left the tip "engaged", a token held or a continuous loop armed would leak into
    whatever runs next -- and the leak would look like a bug in THAT file."""
    prev_speed, hub.host.speed = hub.host.speed, 8.0       # these scans only need to finish in time
    yield
    hub.host.speed = prev_speed
    try:
        cmd.send("setContinuous", {"token": _token(cmd), "enabled": False})
    except Exception:                                    # noqa: BLE001,S110
        pass
    cmd.send("withdraw", {})
    _release(cmd, hub)
    hub.host.imaging_mode = None


def _token(cmd, **kw) -> str:
    """Become the commander, taking over from a previous case's leftovers."""
    reply = cmd.send("acquireControl",
                     {"client": "test-07", "deadman_s": DEADMAN_S, **kw})
    if "error" in reply:                    # held by a token this module abandoned
        from afm_litho.service.hub import hub as real_hub
        real_hub.control = None
        reply = cmd.send("acquireControl",
                         {"client": "test-07", "deadman_s": DEADMAN_S, **kw})
    return reply["result"]["token"]


def _release(cmd, hub) -> None:
    cur = getattr(hub, "control", None)
    if cur is not None:
        cmd.send("releaseControl", {"token": cur.token})


def _wait_state(read, *want, timeout=60.0) -> str:
    """Poll `getState` on the READ socket until it is one of `want`."""
    deadline = time.monotonic() + timeout
    seen = None
    while time.monotonic() < deadline:
        seen = read.call("getState")
        if seen in want:
            return seen
        time.sleep(0.05)
    raise AssertionError(f"state stayed {seen!r}, wanted one of {want}")


def _wait_run(hub, run_id: str, timeout=90.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rec = hub.runs.get(run_id)
        if rec is not None and rec.finished is not None:
            return rec
        time.sleep(0.05)
    raise AssertionError(f"{run_id} never finished")


# ============================================================ 1. the surface ====

def test_the_scan_verbs_are_no_longer_reserved():
    """They answered `-32601 not implemented in this build` until this milestone."""
    for name in ("getMode", "setMode", "startApproach", "withdraw", "setZGain",
                 "startScan", "setContinuous"):
        assert name in fd.REGISTRY, name
        assert name not in fd.RESERVED, name
    # The write half left RESERVED with slice 09 (tests/test_flex_litho.py); what is
    # still honestly reserved is the measure verb and the arbitrary-pass endpoint.
    assert {"executePass"} <= fd.RESERVED


def test_capabilities_publishes_a_bound_for_every_new_verb(read):
    bounds = read.call("getCapabilities")["method_bounds"]
    for name in ("getMode", "setMode", "startApproach", "withdraw", "setZGain",
                 "startScan", "setContinuous"):
        assert name in bounds, name
        assert bounds[name] > 0
    # The honest ones, asserted as VALUES because a driver derives its timeout from
    # them: a start verb that published 35 s would make every script wait for a reply
    # that arrives in milliseconds, and a withdraw that published 1 s would time out.
    # setMode is send + settle, NOT the 35 s "popup plus a readback" an earlier draft
    # invented: `set_imaging_mode` is fire-and-forget, with no readback to wait for.
    assert bounds["setMode"] == pytest.approx(scan_cmds.mode_settle_s() + 1.0)
    assert bounds["setMode"] <= 10.0
    assert bounds["withdraw"] == 6.5
    assert bounds["startApproach"] <= 1.0 and bounds["startScan"] <= 1.0
    # ...and the provider honours them without ever waiting past the backstop.
    assert fp.loop_timeout_for("setMode") >= bounds["setMode"]
    assert fp.loop_timeout_for("withdraw") >= 6.5


def test_every_command_socket_bound_stays_under_ten_seconds():
    """THE SERIALISATION INVARIANT. 29180 handles one request at a time, so a verb
    that blocks for half a minute blocks `abort` and `safePark` behind it for half a
    minute. Anything genuinely slower has to become a start+poll verb instead of
    growing a longer bound -- and `loop_timeout_for` must never lift its own ceiling
    to accommodate one, which it briefly did for a fabricated 35 s `setMode`."""
    for name in fd.COMMAND_ONLY:
        assert fd.REGISTRY[name].bound_s <= fp.LOOP_CALL_MAX_S, name
        assert fp.loop_timeout_for(name) <= fp.LOOP_CALL_MAX_S, name


@pytest.mark.parametrize("raw,want", [(None, 1.0), ("0.1", 1.0), ("2.5", 2.5),
                                      ("nonsense", 1.0), ("-3", 1.0), ("900", 5.0)])
def test_the_mode_settle_may_only_be_lengthened(monkeypatch, raw, want):
    """`AFM_LITHO_MODE_SETTLE_S` is an only-lengthen knob, like every other safety
    timer here: a station that MEASURES a longer vendor call can say so; nothing can
    shorten the wait that stands between a `setMode` and a `startApproach` engaging
    into a half-applied mode change."""
    monkeypatch.delenv("AFM_LITHO_MODE_SETTLE_S", raising=False)
    if raw is not None:
        monkeypatch.setenv("AFM_LITHO_MODE_SETTLE_S", raw)
    assert scan_cmds.mode_settle_s() == want


def test_setmode_waits_out_the_settle_before_returning(cmd, monkeypatch):
    """The twin has no vendor call to settle, so the wait is hardware-only -- asserted
    on the coroutine with a fake hardware hub rather than by sleeping through it."""
    slept = []

    class _Hw:
        hardware_arc = _Arc()
        backend = "hardware"
        engaged = False
        imaging_mode = "contact"
        inst = None
        litho_thread = None
        _scan_task = None
        scan_repeat: ClassVar[dict] = {"enabled": False}

        def backend_label(self):
            return "hardware"

        def backend_live(self):
            return "hardware"

        def require_live_backend(self):
            return None

        async def log(self, text):
            pass

        @property
        def host(self):                # fake: host fields live on the fake itself
            return self

    async def _go():
        import asyncio as _a
        real = _a.sleep

        async def _spy(seconds, *a, **kw):
            slept.append(seconds)
            return await real(0)

        monkeypatch.setattr(_a, "sleep", _spy)
        return await scan_cmds.set_mode(_Hw(), {"mode": "ac"})

    res = asyncio.run(_go())
    assert res == {"ok": True, "mode": "ac", "backend": "hardware"}
    assert scan_cmds.mode_settle_s() in slept, slept


@pytest.mark.parametrize("method", ["setMode", "startApproach", "withdraw",
                                    "setZGain", "startScan", "setContinuous"])
def test_the_mutating_verbs_are_refused_by_the_read_socket(read, method):
    """29181 must stay a plane on which no request can affect the tip -- and that has
    to be true of the SOCKET, not just of the documentation."""
    reply = read.send(method, {"token": "x"})
    assert reply["error"]["code"] == fd.METHOD_NOT_FOUND
    assert "COMMAND socket" in reply["error"]["message"]


def test_getmode_is_served_by_both_sockets(read, cmd):
    """The exception, and deliberately: it is a memory read, and a script deciding
    whether `setMode` would even be accepted should not have to queue behind a
    withdraw to find out."""
    for sock in (read, cmd):
        res = sock.call("getMode")
        assert set(res) == {"mode", "engaged", "gate", "backend"}
        assert res["backend"] == "twin"


# ======================================================== 2. token and backend ==

@pytest.mark.parametrize("method,params", [
    ("setMode", {"mode": "contact"}),
    ("startApproach", {"setpoint": 0.2}),
    ("setZGain", {"igain": 10}),
    ("startScan", dict(SMALL_SCAN)),
    ("setContinuous", {"enabled": False}),
])
def test_every_mutating_verb_needs_the_commander_token(cmd, hub, method, params):
    _release(cmd, hub)
    reply = cmd.send(method, params)
    assert reply["error"]["code"] == fd.CONTROL_REQUIRED == -32030
    assert "control_required" in reply["error"]["message"]


@pytest.mark.parametrize("method,params", [
    ("setMode", {"mode": "contact"}),
    ("startApproach", {}),
    ("startScan", dict(SMALL_SCAN)),
])
def test_a_revoked_token_is_told_so_rather_than_ignored(cmd, hub, method, params):
    """-32011, not -32030: "you HAD this and a human took it" is a different fact
    from "we never issued that", and only one of them means re-acquiring is wrong."""
    tok = _token(cmd)
    asyncio.run_coroutine_threadsafe(
        hub.preempt_for_operator("/api/approach"), hub._loop).result(10)
    reply = cmd.send(method, {**params, "token": tok})
    assert reply["error"]["code"] == fd.CONTROL_REVOKED == -32011


def test_withdraw_needs_no_token(cmd, hub):
    """Plan Sec A: a retract must never be blocked by arbitration, from either side.
    Someone else may even hold control -- getting the tip off the surface wins."""
    _token(cmd)                                   # a commander IS holding the lock
    res = cmd.call("withdraw")
    assert res["ok"] is True
    assert res["state"] in ("idle", "parked")
    assert hub.host.engaged is False
    # ...and the retract did not quietly become an arbitration event: the commander
    # still holds the instrument. A token-free verb must not revoke anyone's token.
    assert hub.control is not None


def test_setzgain_is_refused_on_the_twin_not_silently_ignored():
    """Plan Sec A, HW-only: the twin models no servo to re-arm. A script that
    believes it tuned a loop it did not is worse off than one that was told no."""

    class _NoHardware:
        hardware_arc = None
        backend = "twin"

        def backend_label(self):
            return "twin"

        @property
        def host(self):                # fake: host fields live on the fake itself
            return self

    with pytest.raises(CommandError) as caught:
        asyncio.run(scan_cmds.set_zgain(_NoHardware(), {"igain": 10}))
    assert caught.value.code == fd.BACKEND_MISMATCH == -32010
    assert "hardware-only" in caught.value.message


def test_setzgain_on_the_twin_over_the_wire(cmd):
    tok = _token(cmd)
    reply = cmd.send("setZGain", {"token": tok, "pgain": 0.0, "igain": 10.0})
    assert reply["error"]["code"] == fd.BACKEND_MISMATCH
    assert "backend_mismatch" in reply["error"]["message"]


class _EngagedHardware:
    """A hub that says it is hardware, for the two gates `setZGain` applies before it
    touches a loop. `engaged` is the thing under test; nothing here can move a tip."""

    def __init__(self, engaged: bool):
        self.hardware_arc = object()
        self.backend = "hardware"
        self.engaged = engaged
        self.logs: list[str] = []

    def backend_label(self):
        return "hardware"

    async def log(self, text):
        self.logs.append(str(text))

    @property
    def host(self):                # fake: host fields live on the fake itself
        return self


def test_setzgain_is_refused_while_the_tip_is_withdrawn():
    """THE ENGAGE-WEARING-A-TUNE'S-NAME CASE. `hw_set_zgain` re-arms loop 2
    unconditionally, so from a withdrawn tip it CLOSES the Z servo -- at the 1.0 V
    default, because an un-approached `z_loop` is empty -- and drives the piezo toward
    the surface while `hub.engaged` stays False and `getState` reads `idle`.

    The hub stub never gets as far as a loop: the refusal happens before any call."""
    hub = _EngagedHardware(engaged=False)
    with pytest.raises(CommandError) as caught:
        asyncio.run(scan_cmds.set_zgain(hub, {"igain": 10}))
    assert caught.value.code == fd.REFUSED == -32021
    assert "not engaged" in caught.value.message
    assert caught.value.http_body()["reason"] == "not_engaged"
    assert any("REFUSED" in line for line in hub.logs)


def test_setzgain_refusal_is_the_same_on_the_http_door(client, hub):
    """Both doors, one gate: the panel must not be able to do it either."""
    was, hub.host.engaged = hub.host.engaged, False
    try:
        body = client.post("/api/zgain", json={"igain": 10}).json()
    finally:
        hub.host.engaged = was
    # the twin has no bridge at all, so it stops one gate earlier -- either way the
    # request is REFUSED and nothing armed a loop.
    assert "error" in body
    assert hub.host.engaged is was


def test_setzgain_gets_past_the_engaged_gate_when_the_tip_is_down():
    """The gate is about the STATE, not about the verb: with the tip engaged the call
    proceeds (and then fails on the stub's missing loop API, which is the next line of
    the real path)."""
    hub = _EngagedHardware(engaged=True)
    with pytest.raises(Exception) as caught:
        asyncio.run(scan_cmds.set_zgain(hub, {"igain": 10}))
    assert not (isinstance(caught.value, CommandError)
                and caught.value.code == fd.REFUSED)


# ================================================================== 3. setMode ==

def test_setmode_switches_and_is_reported(cmd, read):
    tok = _token(cmd)
    assert cmd.call("setMode", {"token": tok, "mode": "ac"}) == {
        "ok": True, "mode": "ac", "backend": "twin"}
    assert read.call("getMode")["mode"] == "ac"
    assert cmd.call("setMode", {"token": tok, "mode": "contact"})["mode"] == "contact"


def test_setmode_rejects_a_mode_it_does_not_have(cmd):
    tok = _token(cmd)
    reply = cmd.send("setMode", {"token": tok, "mode": "tapping"})
    assert reply["error"]["code"] == fd.INVALID_PARAMS
    assert "contact" in reply["error"]["message"]


def test_setmode_is_refused_while_the_tip_is_engaged(cmd, read, hub):
    """SAFETY-CRITICAL: the vendor switch changes the Z feedback channel and the gain
    sign and does NOT withdraw, so with the tip down it is a crash. -32021 refused,
    not -32020 busy: retrying cannot fix it, only a withdraw can."""
    from _im_fixtures_guard import seed_free_air
    seed_free_air(hub)          # B1: approach now runs check_approach (needs free-air)
    tok = _token(cmd)
    run = cmd.call("startApproach", {"token": tok, "setpoint": 0.2})
    _wait_state(read, "engaged")
    _wait_run(hub, run["run_id"])
    reply = cmd.send("setMode", {"token": tok, "mode": "ac"})
    assert reply["error"]["code"] == fd.REFUSED == -32021
    assert "REFUSED" in reply["error"]["message"]
    assert "withdraw first" in reply["error"]["message"].lower()
    # and the switch really did not happen
    assert read.call("getMode")["mode"] != "ac"


# ============================================================ 4. startApproach ==

def test_start_approach_is_start_and_poll(cmd, read, hub):
    """`{run_id, est_s}` back in well under a second, `approaching` visible while it
    runs, `engaged` after, and a run record that CLOSES."""
    from _im_fixtures_guard import seed_free_air
    seed_free_air(hub)          # B1: approach now runs check_approach (needs free-air)
    tok = _token(cmd)
    t0 = time.monotonic()
    res = cmd.call("startApproach", {"token": tok, "setpoint": 0.2, "settle_s": 1.0})
    assert time.monotonic() - t0 < 1.0, "a start verb must not block on the pass"
    assert res["run_id"].startswith("approach-")
    assert res["state"] == "approaching"
    assert res["est_s"] == pytest.approx(1.0 + scan_cmds.APPROACH_MARGIN_S)
    assert _wait_state(read, "engaged") == "engaged"
    rec = _wait_run(hub, res["run_id"])
    assert rec.kind == "approach" and rec.completed is True
    # the in-flight marker and the single-writer claim are both released
    assert not hub.host._approach_in_flight
    assert hub._run_claim is None


def test_getstatus_names_the_approach_run(cmd, read, hub):
    """A run kind rather than a null id, so `getStatus.run_id` shows what is in
    flight -- an approach is otherwise the one tip-moving operation with nothing of
    its own to point at."""
    tok = _token(cmd)
    res = cmd.call("startApproach", {"token": tok, "settle_s": 1.0})
    status = read.call("getStatus")
    assert status["run_id"] == res["run_id"]
    assert status["run_kind"] == "approach"
    assert status["state"] in ("approaching", "engaged")
    _wait_run(hub, res["run_id"])


def test_a_second_start_while_busy_is_busy_with_the_run_id(cmd, read, hub):
    """Plan Sec F.1: a start verb arriving while busy errors -32020 and NEVER
    queues, and the error names what is in flight so a script can poll it."""
    tok = _token(cmd)
    started = cmd.call("startScan", {"token": tok, **SMALL_SCAN,
                                     "line_rate_hz": 4.0})
    reply = cmd.send("startScan", {"token": tok, **SMALL_SCAN})
    assert reply["error"]["code"] == fd.BUSY == -32020
    assert "busy" in reply["error"]["message"]
    assert reply["error"]["data"]["run_id"] is not None
    second = cmd.send("startApproach", {"token": tok})
    assert second["error"]["code"] == fd.BUSY
    _wait_run(hub, started["run_id"])


def test_the_same_bad_input_is_refused_identically_on_both_doors(client, cmd, hub):
    """THE ONE-CODE-PATH PROOF. `POST /api/approach` and `startApproach` share
    `commands.scan_cmds`, so the same nonsense setpoint must produce the same
    sentence -- if the two ever drift apart, a cockpit that accepts something a
    script is refused (or the reverse) is exactly the class of bug the extraction
    exists to prevent.

    HTTP first, ON PURPOSE: `/api/approach` is in the curated tip-mutating set, so it
    preempts any script token. The wire call acquires afterwards."""
    bad = {"setpoint": "abc"}
    http = client.post("/api/approach", json=bad).json()
    tok = _token(cmd)
    wire = cmd.send("startApproach", {"token": tok, **bad})
    assert http["reason"] == "bad_params"
    assert wire["error"]["code"] == fd.INVALID_PARAMS
    assert http["error"] == wire["error"]["message"]
    assert "setpoint" in http["error"]
    # and neither door started anything
    assert not hub.host._approach_in_flight


# =============================================================== 5. startScan ===

def test_start_scan_runs_a_frame_and_the_result_is_fetchable(cmd, read, hub):
    """The whole point of start+poll: a run id, a state that leaves `scanning`, and
    a result whose `completed` is gated on PERSISTENCE, so the paths are whole."""
    tok = _token(cmd)
    res = cmd.call("startScan", {"token": tok, **SMALL_SCAN})
    assert res["run_id"].startswith("scan-")
    assert res["state"] == "scanning"
    # 8 lines at 40 Hz, trace+retrace, plus the margin
    assert res["est_s"] == pytest.approx(8 / 40 * 2 + scan_cmds.EST_MARGIN_S)
    _wait_run(hub, res["run_id"])
    result = read.call("getScanResult", {"run_id": res["run_id"]})
    assert result["run_id"] == res["run_id"]
    assert result["kind"] == "scan"
    if result["completed"]:
        for path in result["files"].values():
            assert os.path.isfile(path), path


def test_start_scan_rejects_a_setpoint_without_engage_on_hardware():
    """Plan Sec A: on hardware a `setpoint` is INERT unless `engage: true` -- the
    scan runs at whatever the current engagement is. Accept-and-ignore is the shape
    of a script that images a whole array at the wrong force and never finds out.

    Asserted against the service layer with a hub that merely SAYS it is hardware:
    this is a refusal, so nothing is driven either way, and the twin cannot exercise
    a hardware-only rule."""

    class _SaysHardware:
        backend = "hardware"

        def backend_live(self):
            return "hardware"

        @property
        def host(self):                # fake: host fields live on the fake itself
            return self

    with pytest.raises(CommandError) as caught:
        asyncio.run(scan_cmds.start_scan(_SaysHardware(),
                                         {**SMALL_SCAN, "setpoint": 0.5}))
    assert caught.value.code == fd.INVALID_PARAMS == -32602
    assert "engage" in caught.value.message
    assert "inert" in caught.value.message


def test_the_same_scan_with_engage_passes_the_setpoint_check():
    """The refusal is about the COMBINATION, not about setpoints -- a scan that says
    `engage: true` means the setpoint, and must get past this gate (it then fails on
    the fake hub's missing instrument, which is the next line of the real path)."""

    class _SaysHardware:
        backend = "hardware"

        def backend_live(self):
            return "hardware"

        @property
        def host(self):                # fake: host fields live on the fake itself
            return self

    with pytest.raises(Exception) as caught:
        asyncio.run(scan_cmds.start_scan(
            _SaysHardware(), {**SMALL_SCAN, "setpoint": 0.5, "engage": True}))
    assert not (isinstance(caught.value, CommandError)
                and caught.value.code == fd.INVALID_PARAMS
                and "engage" in caught.value.message)


@pytest.mark.parametrize("param", ["pgain", "igain"])
def test_start_scan_rejects_a_z_gain_without_engage_on_hardware(param):
    """Same rule as `setpoint` (plan Sec A), extended to the optional Z-loop gain
    override: `pgain`/`igain` are INERT on hardware unless `engage: true` -- the scan
    would run with whatever Z gains are already armed, not these. Asserted against
    the service layer with a hub that merely SAYS it is hardware, same as the
    setpoint case above."""

    class _SaysHardware:
        backend = "hardware"

        def backend_live(self):
            return "hardware"

        @property
        def host(self):                # fake: host fields live on the fake itself
            return self

    with pytest.raises(CommandError) as caught:
        asyncio.run(scan_cmds.start_scan(_SaysHardware(),
                                         {**SMALL_SCAN, param: 2.5}))
    assert caught.value.code == fd.INVALID_PARAMS == -32602
    assert param in caught.value.message
    assert "engage" in caught.value.message
    assert "inert" in caught.value.message


def test_start_scan_echoes_pgain_and_igain_on_the_twin(cmd, hub):
    """Twin: `pgain`/`igain` are accepted and ignored, exactly like `setpoint` --
    echoed back in the config (so a caller can see what it sent), never acted on
    (the twin models no servo to re-arm)."""
    tok = _token(cmd)
    res = cmd.call("startScan", {"token": tok, **SMALL_SCAN, "engage": True,
                                 "pgain": 3.0, "igain": 25.0})
    assert res["config"]["pgain"] == 3.0
    assert res["config"]["igain"] == 25.0
    _wait_run(hub, res["run_id"])


@pytest.mark.parametrize("param", ["pgain", "igain"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")],
                         ids=["nan", "inf", "-inf"])
def test_start_scan_rejects_a_non_finite_z_gain_over_the_wire(cmd, param, bad):
    """`pgain`/`igain` must be FINITE numbers -- a NaN/Inf gain reaching the real
    `ir_SetPISLoop` is a silent instrument fault, not a refusal a script would ever
    see coming (see `scan_cmds._finite_number`). Sent as REAL JSON `NaN`/`Infinity`
    literals over the actual socket: `json.dumps`'s default `allow_nan=True` writes
    them as bare (non-standard) tokens, and the provider's `json.loads(raw)`
    (`flex_provider.py`) accepts them back the same way -- so this exercises the
    validation the service layer actually sees, on the twin backend (no `engage`
    needed: `_finite_number` runs in `build_scan_config`, before any backend or
    engage gate)."""
    tok = _token(cmd)
    reply = cmd.send("startScan", {"token": tok, **SMALL_SCAN, param: bad})
    assert reply["error"]["code"] == fd.INVALID_PARAMS == -32602
    assert param in reply["error"]["message"]


@pytest.mark.parametrize("falsy", [False, 0, "false", ""], ids=["false", "zero", "str", "empty"])
def test_start_scan_refuses_an_explicit_unchained_request(cmd, falsy):
    """The unchained per-line scan was retired; `chained: false` is refused, never
    silently changed. `chained: true` or omitted is the one scan path."""
    tok = _token(cmd)
    reply = cmd.send("startScan", {"token": tok, **SMALL_SCAN, "chained": falsy})
    assert reply["error"]["code"] == fd.INVALID_PARAMS
    assert "unchained scan was retired" in reply["error"]["message"]


# ================================================================ 6. withdraw ===

def test_withdraw_reports_a_verified_park(cmd, read, hub):
    from _im_fixtures_guard import seed_free_air
    seed_free_air(hub)          # B1: approach now runs check_approach (needs free-air)
    tok = _token(cmd)
    run = cmd.call("startApproach", {"token": tok, "settle_s": 1.0})
    _wait_state(read, "engaged")
    _wait_run(hub, run["run_id"])
    res = cmd.call("withdraw", {"token": tok})
    assert res["ok"] is True
    assert res["verify_timeout"] is False
    assert hub.host.engaged is False
    assert read.call("getState") in ("idle", "parked")


# =========================================================== 7. setContinuous ===

def test_set_continuous_rail_limits_may_only_tighten(cmd, hub):
    """Plan Sec A: "the provider may only TIGHTEN, never loosen". A script asking for
    a 149 V rail ceiling on a guard armed at 140 gets 140 -- the request is floored,
    not honoured and not refused."""
    tok = _token(cmd)
    default = hub.host.rail_guard.limits
    res = cmd.call("setContinuous", {
        "token": tok, "enabled": True, "config": dict(SMALL_SCAN),
        "max_frames": 1,
        "rail": {"v_hi": default.v_hi + 9.0,        # LOOSER -> must be ignored
                 "v_lo": default.v_lo - 5.0,        # LOOSER -> must be ignored
                 "consec_lines": max(1, default.consec_lines - 2)}})   # tighter
    assert res["ok"] is True and res["enabled"] is True
    assert res["rail"]["v_hi"] == default.v_hi
    assert res["rail"]["v_lo"] == default.v_lo
    assert res["rail"]["consec_limit"] == max(1, default.consec_lines - 2)
    stop = cmd.call("setContinuous", {"token": tok, "enabled": False})
    assert stop["enabled"] is False
    # the guard's own limits are restored for whatever runs next
    hub.host.rail_guard.limits = default


def test_set_continuous_refuses_a_rail_that_is_not_an_object(cmd):
    tok = _token(cmd)
    reply = cmd.send("setContinuous", {"token": tok, "enabled": True,
                                       "config": dict(SMALL_SCAN), "rail": 140})
    assert reply["error"]["code"] == fd.INVALID_PARAMS
    assert "rail" in reply["error"]["message"]


def test_disabling_continuous_is_allowed_and_idempotent(cmd):
    """Stopping must never be the call that is refused."""
    tok = _token(cmd)
    for _ in range(2):
        assert cmd.call("setContinuous", {"token": tok,
                                          "enabled": False})["enabled"] is False


# ============================================================== 8. the dead-man ==

@pytest.mark.parametrize("method,params", [
    ("getMode", {}),
    ("withdraw", {}),
    ("setMode", {"mode": "contact"}),
])
def test_these_verbs_refresh_the_deadman(cmd, hub, method, params):
    """Plan Sec E.1: any COMMAND-socket request bearing a valid token refreshes it,
    so a script that is actually commanding the instrument refreshes for free."""
    tok = _token(cmd)
    hub.touch_control(tok)
    before = hub.control.expires_s()
    time.sleep(1.0)
    assert hub.control.expires_s() < before - 0.9    # the clock really is running
    cmd.send(method, {**params, "token": tok})
    # The refresh happens when the request ARRIVES, so by the time we look, the
    # verb's own duration has ticked off it again -- a withdraw really does drive the
    # twin. The slack only has to separate "refreshed" from "not refreshed at all",
    # which the 1 s sleep above makes an unambiguous gap.
    assert hub.control.expires_s() > before - 0.5


def test_the_read_socket_never_refreshes_the_deadman(read, cmd, hub):
    """The rule that stops a CRASHED script being kept alive for ever by anything
    polling `getState` with its token."""
    tok = _token(cmd)
    hub.touch_control(tok)
    before = hub.control.expires_s()
    time.sleep(0.25)
    read.send("getMode", {"token": tok})
    read.send("getState", {"token": tok})
    assert hub.control.expires_s() < before


# ===================================== 8b. a refusal changes nothing ==============

def test_a_refused_setcontinuous_does_not_stop_a_running_loop(cmd, hub):
    """PARTIAL STATE. The arming block unwinds any running supervisor, clears the stop
    event and drops the safe_park latch; a `-32602` raised from inside the update
    literal (as `withdraw_on_stop` used to be) would therefore have STOPPED the
    operator's loop on the way to refusing the call."""
    from afm_litho.service.host import continuous_cmds

    tok = _token(cmd)
    started = cmd.call("setContinuous", {
        "token": tok, "enabled": True, "config": dict(SMALL_SCAN), "max_frames": 3})
    assert started["enabled"] is True
    supervisor = hub.host._scan_repeat_task

    # Driven on the service layer, not the wire: the wire door refuses a re-arm while
    # a frame is in flight (-32020) and would never reach the validation this case is
    # about. The layer underneath is where both doors meet it.
    fut = asyncio.run_coroutine_threadsafe(
        continuous_cmds.set_continuous(hub, {
            "enabled": True, "config": dict(SMALL_SCAN),
            "withdraw_on_stop": "yes"}),      # a string, not a bool -> -32602
        hub._loop)
    with pytest.raises(CommandError) as caught:
        fut.result(30)
    assert caught.value.code == fd.INVALID_PARAMS
    assert "withdraw_on_stop" in caught.value.message
    # the loop that was running is STILL running, on its ORIGINAL supervisor
    assert hub.host.scan_repeat.get("enabled") is True
    assert hub.host._scan_repeat_task is supervisor
    assert not hub.host._scan_repeat_stop.is_set()
    cmd.call("setContinuous", {"token": tok, "enabled": False})


def test_a_refused_startscan_does_not_arm_the_raster(cmd, hub):
    """The twin used to call `inst.start_scan` and only then discover the pump slot
    was taken -- leaving the raster ARMED behind a busy reply."""
    tok = _token(cmd)
    first = cmd.call("startScan", {"token": tok, **SMALL_SCAN, "line_rate_hz": 4.0})
    running_before = bool(getattr(hub.host.inst.scan, "running", False))
    reply = cmd.send("startScan", {"token": tok, **SMALL_SCAN})
    assert reply["error"]["code"] == fd.BUSY
    # the in-flight frame is untouched, and no SECOND raster was armed under it
    assert bool(getattr(hub.host.inst.scan, "running", False)) == running_before
    assert hub.runs.get(first["run_id"]) is not None
    _wait_run(hub, first["run_id"])


# ========================================================== 9. the full sequence ==

def test_approach_scan_result_withdraw(cmd, read, hub):
    """The slice 07 deliverable, end to end on the twin: exactly the sequence
    docs/FLEX_RIG_RUNBOOK.md "07 Scripted imaging" runs at the instrument, minus the
    instrument.

    Everything a script would do -- acquire, engage, poll, scan three offsets, fetch
    each result by path, withdraw, release -- and every poll on the READ socket, so
    none of it queues behind the command plane."""
    from _im_fixtures_guard import seed_free_air
    seed_free_air(hub)          # B1: approach now runs check_approach (needs free-air)
    tok = _token(cmd)
    assert read.call("getStatus")["backend"] == "twin"

    engage = cmd.call("startApproach", {"token": tok, "setpoint": 0.2,
                                        "settle_s": 1.0})
    assert _wait_state(read, "engaged") == "engaged"
    assert _wait_run(hub, engage["run_id"]).completed is True

    results = []
    for offset in (0.0, 1.0, -1.0):
        run = cmd.call("startScan", {"token": tok, **SMALL_SCAN,
                                     "x_offset_um": offset})
        assert read.call("getState") == "scanning"
        _wait_run(hub, run["run_id"])
        _wait_state(read, "engaged", "idle")
        res = read.call("getScanResult", {"run_id": run["run_id"]})
        assert res["run_id"] == run["run_id"] and res["aborted"] is False
        results.append(res)
    assert len({r["run_id"] for r in results}) == 3

    out = cmd.call("withdraw", {"token": tok})
    assert out["ok"] is True
    assert _wait_state(read, "idle", "parked") in ("idle", "parked")
    assert cmd.call("releaseControl", {"token": tok}) == {"ok": True, "stale": False}
