"""Park via the instrument model, sensor-verified, vendor fallback (post-incident
fix; docs/INSTRUMENT_MODEL_SPEC.md Sec 3 + 10, wave-1 slice T3).

RED against today's `service/host/safe_park.py`, which is an ASYNC coroutine
`safe_park(hub, reason) -> dict` hard-wired to `Output.Z` / the vendor alias
"height" (the old `_VERIFY_CHANNEL`) and has no `command_guard.park_plan`,
no `InstrumentModel`, no `hub.litho_worker` (only `hub.litho_run` + a bare thread
today -- see `service/host/litho_worker.py`), and no `hardware_arc.vendor_withdraw`.
The new contract (Sec 10) is synchronous and channel-driven off the model:

    safe_park(hub, reason) -> ParkResult
    ParkResult(verified: bool, method: "ramp" | "vendor_withdraw",
               z_out_v, zsensor_v, defl_v, critical: bool)

and `command_guard.park_plan(model) -> ParkPlan` (`.stop_loop, .ramp_channel,
.park_v, .tol_v, .verify_channels`) is the ONLY source of which channel gets
ramped -- never a hard-coded "Output.Z" / "height" alias.

This is the incident itself: the old code verified a fixed alias reading back a
plausible value while the ACTUAL Height-loop channel on a Cypher never moved --
"the commanded value alone is never evidence" (Sec 3 step 3).

Per the 2026-09-22 spec update (40ace53 + 97a4296 + 12121c0), Sec 10 now names the
object `safe_park` drives the ramp/read through explicitly: ONE object at
`hub.hardware_arc` exposing `.ramp(channel, target_v)`, `.read(channel) -> float`
and `.vendor_withdraw()` on BOTH backends (the twin's `SoftwareARC` implements the
same three; its `vendor_withdraw` ignores `ramp_noop`) -- this file's `_Bridge`
double targets that exact surface, and `.commands` is its command log. The model
lives at `hub.instrument_model` (not `hub.instrument` -- Sec 10 renamed it to avoid
a collision), `GuardRefused` carries JSON-RPC code `-32050`, and every `check_*`
takes a keyword `fault: object | None = None`, refusing first with reason
`"fault_latched"` when truthy (callers pass `hub.fault`). `hub.free_air` is still
stood in locally as a tiny `FreeAir`-shaped dataclass (`value_v, ts, mode,
withdrawn`) since `command_guard.FreeAir` does not exist yet (T2's slice), and
`InstrumentModel` is still stood in with a minimal `Field`-shaped (`.value`) duck
type since `instrument_model.py` (T1's slice) does not exist yet -- only the SHAPE
`park_plan`/`check_approach` are documented to read is exercised, not the real
dataclass.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import pytest

from afm_litho.service import command_guard
from afm_litho.service.host.safe_park import safe_park
from afm_litho.twin.arc import SoftwareARC
from afm_litho.twin.physics import Sample, World

# ======================================================================= fakes ==

class _Field:
    """Duck-typed stand-in for `instrument_model.Field` (`.value`, `.source`)."""

    def __init__(self, value):
        self.value = value
        self.source = "test"


class _FakeModel:
    """Duck-typed stand-in for `instrument_model.InstrumentModel` -- only the
    fields `command_guard.park_plan` is documented to read (Sec 10)."""

    def __init__(self, *, platform: str, z_park_v: float = 0.0,
                 z_park_tol_v: float = 0.5,
                 z_verify_channels=("ZSensor", "Deflection")):
        z_out = "Height" if platform == "cypher" else "Output.Z"
        self.platform = _Field(platform)
        self.z_out_channel = _Field(z_out)
        self.z_park_v = _Field(z_park_v)
        self.z_park_tol_v = _Field(z_park_tol_v)
        self.z_verify_channels = _Field(z_verify_channels)
        # B1 fix (section-10 contradiction, reported): the docstring above says this
        # fake stands in for the fields "park_plan/check_approach" read, and section
        # 10 lists these three as read by `check_approach` (free-air freshness + the
        # contact/AC setpoint bands). Without them, `check_approach(fault=None)` in
        # `test_no_fault_does_not_trigger_fault_latched` raises AttributeError inside
        # command_guard instead of completing/raising GuardRefused. Defaults chosen so
        # that test's setpoint 0.5 (rel 0.5, band [0,1]) passes cleanly.
        self.free_air_max_age_s = _Field(30)
        self.contact_setpoint_band_v = _Field((0.0, 1.0))
        self.ac_setpoint_fraction_band = _Field((0.3, 0.95))


@dataclass(frozen=True)
class _FreeAir:
    """Stand-in for `command_guard.FreeAir`."""

    value_v: float
    ts: float
    mode: str = "contact"
    withdrawn: bool = True


class _Bridge:
    """Records every abort/confirm/read/vendor-withdraw safe_park issues (since slice AB the
    retract is Igor's one abort function, `abort_retract` on the twin, and the confirm is Height < 1 V). `order`, if
    given, is a list SHARED with a `_RecordingWorker` so cross-object ordering
    (worker join before any Z command) can be asserted.

    `move_zsensor` / `move_defl` let a test move the two verify sensors
    INDEPENDENTLY on a successful ramp (MUST-FIX 1: neither sensor alone may
    stand in for the other -- a stuck ZSensor with fine deflection, or fine
    ZSensor with stuck deflection, must both fail verify). `z_read_v` is the
    separate READBACK of the commanded channel itself (`plan.ramp_channel`,
    e.g. "Height"/"Output.Z"), served by `read()` alongside "ZSensor" /
    "Deflection" (MUST-FIX 2) -- this is what lets a test build exactly the
    incident shape: the commanded channel reads back the target while the
    REAL sensor (ZSensor) never moved.
    """

    def __init__(self, *, zsensor_v: float, defl_v: float,
                 abort_noop: bool = False, vendor_ok: bool = True,
                 order: list | None = None,
                 move_zsensor: bool = True, move_defl: bool = True,
                 height_v: float = 60.0):
        self.commands: list[tuple] = []
        self.zsensor_v = zsensor_v
        self.defl_v = defl_v
        self.abort_noop = abort_noop
        self.vendor_ok = vendor_ok
        self.vendor_withdraw_calls = 0
        self._order = order
        self.move_zsensor = move_zsensor
        self.move_defl = move_defl
        self._z_channel_v = height_v       # the Height readback the confirm reads

    def abort_retract(self) -> None:
        """Igor's abort function: the re-arm drives Height to its 0 V minimum."""
        self.commands.append(("abort",))
        if self._order is not None:
            self._order.append("abort")
        if not self.abort_noop:
            self._z_channel_v = 0.0
            if self.move_zsensor:
                self.zsensor_v = 0.0
            if self.move_defl:
                self.defl_v = 0.0

    def confirm_retract(self) -> None:
        self.commands.append(("confirm",))

    def read(self, channel: str) -> float:
        if channel == "ZSensor":
            return self.zsensor_v
        if channel == "Deflection":
            return self.defl_v
        if channel in ("Height", "Output.Z"):    # the commanded z_out_channel
            return self._z_channel_v
        raise AssertionError(f"unexpected verify channel {channel!r}")

    def vendor_withdraw(self) -> dict:
        self.vendor_withdraw_calls += 1
        self.commands.append(("vendor_withdraw",))
        if self._order is not None:
            self._order.append("vendor_withdraw")
        if self.vendor_ok:
            self.zsensor_v = 0.0
            self.defl_v = 0.0
            self._z_channel_v = 0.0
        return {"ok": self.vendor_ok}


class _RecordingWorker:
    """Stand-in for `hub.litho_worker` (Sec 10: `.request_abort()`, `.join(timeout)`)."""

    def __init__(self, order: list | None = None):
        self._order = order
        self.abort_requested = False
        self.join_timeout = "unset"
        self._alive = True

    def request_abort(self) -> None:
        self.abort_requested = True
        if self._order is not None:
            self._order.append("request_abort")

    def join(self, timeout=None) -> None:
        self.join_timeout = timeout
        self._alive = False
        if self._order is not None:
            self._order.append("join")

    def is_alive(self) -> bool:
        return self._alive


class _FakeHub:
    def __init__(self, model, bridge, *, litho_worker=None, free_air=None):
        self.instrument_model = model
        self.hardware_arc = bridge
        self.litho_worker = litho_worker
        self.free_air = free_air
        self.fault = None
        self.logs: list[str] = []
        self.records: list[dict] = []

    def log(self, msg: str) -> None:      # sync per Sec 10's sync safe_park
        self.logs.append(msg)

    @property
    def host(self):                # fake: host fields live on the fake itself
        return self


def _model(platform: str, **kw) -> _FakeModel:
    return _FakeModel(platform=platform, **kw)


def _free_air(value_v: float = 0.0) -> _FreeAir:
    return _FreeAir(value_v=value_v, ts=time.time())


# ============================================================= park_plan unit ==

def test_park_plan_cypher_confirms_on_height():
    plan = command_guard.park_plan(_model("cypher"))
    assert plan.z_channel == "Height"
    assert plan.verify_channels == ("ZSensor", "Deflection")


def test_park_plan_mfp3d_confirms_on_its_model_channel():
    plan = command_guard.park_plan(_model("mfp3d"))
    assert plan.z_channel == "Output.Z"


# ======================================================== safe_park: channel ==

def test_safe_park_cypher_runs_the_abort_and_confirms_with_no_ramp():
    model = _model("cypher", z_park_v=0.0, z_park_tol_v=0.5)
    bridge = _Bridge(zsensor_v=-5.0, defl_v=2.0)
    hub = _FakeHub(model, bridge, free_air=_free_air(0.0))

    result = safe_park(hub, "test")

    assert bridge.commands == [("abort",), ("confirm",)]     # no STOPLOOP, no ramp
    assert result.method == "abort"
    assert result.verified is True
    assert bridge.vendor_withdraw_calls == 0    # SHOULD-FIX 4: no fallback needed


def test_safe_park_mfp3d_parks_the_same_way():
    model = _model("mfp3d", z_park_v=0.0, z_park_tol_v=0.5)
    bridge = _Bridge(zsensor_v=-5.0, defl_v=2.0)
    hub = _FakeHub(model, bridge, free_air=_free_air(0.0))

    result = safe_park(hub, "test")

    assert bridge.commands == [("abort",), ("confirm",)]
    assert result.method == "abort"
    assert result.verified is True
    assert bridge.vendor_withdraw_calls == 0    # SHOULD-FIX 4: no fallback needed


# ================================================== safe_park: Height confirm ==

def test_already_retracted_height_confirms_and_records_the_sensors():
    model = _model("cypher", z_park_v=0.0, z_park_tol_v=0.5)
    bridge = _Bridge(zsensor_v=0.0, defl_v=0.0, height_v=0.0)
    hub = _FakeHub(model, bridge, free_air=_free_air(0.0))

    result = safe_park(hub, "test")

    assert result.verified is True
    assert result.zsensor_v == pytest.approx(0.0)
    assert result.defl_v == pytest.approx(0.0)


def test_height_under_one_volt_confirms_even_with_the_sensors_elsewhere():
    """2026-10-27: Height (the meter's Z Voltage) is the confirm. ZSensor and
    Deflection are read and recorded beside it; they no longer gate `verified`."""
    model = _model("cypher", z_park_v=0.0, z_park_tol_v=0.5)
    bridge = _Bridge(zsensor_v=3.17, defl_v=-0.6, move_zsensor=False, move_defl=False)
    hub = _FakeHub(model, bridge, free_air=_free_air(0.0))

    result = safe_park(hub, "test")

    assert result.verified is True
    assert result.z_out_v == pytest.approx(0.0)
    assert result.zsensor_v == pytest.approx(3.17)
    assert result.defl_v == pytest.approx(-0.6)
    assert any("ZSensor=3.170 V" in m for m in hub.logs)


@pytest.mark.parametrize("height_v, confirmed", [(0.99, True), (1.0, False), (60.0, False)])
def test_the_confirm_threshold_is_height_under_one_volt(height_v, confirmed):
    model = _model("cypher")
    bridge = _Bridge(zsensor_v=0.0, defl_v=0.0, abort_noop=True, vendor_ok=False,
                     height_v=height_v)
    hub = _FakeHub(model, bridge, free_air=_free_air(0.0))

    result = safe_park(hub, "test")

    assert result.verified is confirmed
    assert (("confirm",) in bridge.commands) is confirmed


def test_z_out_v_reports_the_height_read():
    model = _model("cypher", z_park_v=0.0, z_park_tol_v=0.5)
    bridge = _Bridge(zsensor_v=-5.0, defl_v=2.0, abort_noop=True, vendor_ok=False,
                     height_v=42.0)
    hub = _FakeHub(model, bridge, free_air=_free_air(0.0))

    result = safe_park(hub, "test")

    assert result.z_out_v == pytest.approx(42.0)
    assert result.verified is False


def test_z_out_v_matches_bridge_read_on_a_normal_verified_park():
    model = _model("mfp3d", z_park_v=0.0, z_park_tol_v=0.5)
    bridge = _Bridge(zsensor_v=-5.0, defl_v=2.0)
    hub = _FakeHub(model, bridge, free_air=_free_air(0.0))

    result = safe_park(hub, "test")

    plan = command_guard.park_plan(model)
    assert result.z_out_v == pytest.approx(bridge.read(plan.z_channel))
    assert result.verified is True


# ================================================== safe_park: vendor fallback ==

def test_unconfirmed_abort_triggers_vendor_withdraw_exactly_once():
    model = _model("cypher", z_park_v=0.0, z_park_tol_v=0.5)
    bridge = _Bridge(zsensor_v=-5.0, defl_v=2.0, abort_noop=True, vendor_ok=True)
    hub = _FakeHub(model, bridge, free_air=_free_air(0.0))

    result = safe_park(hub, "test")

    assert [c for c in bridge.commands if c[0] == "abort"] == [("abort",)]
    assert bridge.commands[-1] == ("confirm",)        # the vendor withdraw confirmed
    assert bridge.vendor_withdraw_calls == 1
    assert result.method == "vendor_withdraw"
    assert result.verified is True                    # vendor withdraw succeeded


def test_vendor_withdraw_also_failing_is_critical_and_latches_fault():
    model = _model("cypher", z_park_v=0.0, z_park_tol_v=0.5)
    bridge = _Bridge(zsensor_v=-5.0, defl_v=2.0, abort_noop=True, vendor_ok=False)
    hub = _FakeHub(model, bridge, free_air=_free_air(0.0))

    result = safe_park(hub, "test")

    assert result.verified is False
    assert result.critical is True
    assert hub.fault                                    # latched, truthy


def test_critical_park_log_says_park_unverified_never_withdrawn():
    model = _model("cypher", z_park_v=0.0, z_park_tol_v=0.5)
    bridge = _Bridge(zsensor_v=-5.0, defl_v=2.0, abort_noop=True, vendor_ok=False)
    hub = _FakeHub(model, bridge, free_air=_free_air(0.0))

    safe_park(hub, "test")

    assert any("park_unverified" in m for m in hub.logs)
    assert not any("withdrawn" in m and "park_unverified" not in m for m in hub.logs)


def test_verified_park_never_logs_park_unverified():
    model = _model("mfp3d", z_park_v=0.0, z_park_tol_v=0.5)
    bridge = _Bridge(zsensor_v=0.0, defl_v=0.0)
    hub = _FakeHub(model, bridge, free_air=_free_air(0.0))

    safe_park(hub, "test")

    assert not any("park_unverified" in m for m in hub.logs)


def test_repeated_safe_park_calls_during_one_failure_do_not_repeat_the_abort():
    """Asserted at the `safe_park` level: two separate calls issue exactly one
    ramp between them. An implementer MAY choose to route this dedup through
    `hub.fault` at the CALLER (i.e. a caller that already sees `hub.fault` set
    skips calling `safe_park` again at all) rather than inside `safe_park`
    itself -- either satisfies this assertion, since it only counts commands
    on the shared bridge across both calls, but they must report which one
    they built, per Sec 10's instruction not to silently deviate."""
    model = _model("cypher", z_park_v=0.0, z_park_tol_v=0.5)
    bridge = _Bridge(zsensor_v=-5.0, defl_v=2.0, abort_noop=True, vendor_ok=False)
    hub = _FakeHub(model, bridge, free_air=_free_air(0.0))

    safe_park(hub, "first")
    safe_park(hub, "second")

    assert [c for c in bridge.commands if c[0] == "abort"] == [("abort",)]
    assert bridge.vendor_withdraw_calls == 1


# ================================================== safe_park: subsequent refusal =

def test_latched_fault_refuses_a_subsequent_approach_with_fault_latched():
    """A critical park latches `hub.fault`; every `check_*` in command_guard takes
    `fault: object | None = None` and refuses FIRST with reason `"fault_latched"`
    when it is truthy (Sec 10, 2026-09-22 update), regardless of how otherwise
    valid the approach parameters are -- the operator must clear the fault via
    `POST /api/fault/clear {confirm: true}` before anything re-engages."""
    model = _model("cypher", z_park_v=0.0, z_park_tol_v=0.5)
    bridge = _Bridge(zsensor_v=-5.0, defl_v=2.0, abort_noop=True, vendor_ok=False)
    hub = _FakeHub(model, bridge, free_air=_free_air(0.0))

    safe_park(hub, "test")
    assert hub.fault

    with pytest.raises(command_guard.GuardRefused) as excinfo:
        command_guard.check_approach(
            model, mode="contact", setpoint=0.5, free_air=hub.host.free_air,
            now=time.time(), fault=hub.fault)
    assert excinfo.value.reason == "fault_latched"
    assert excinfo.value.code == -32050


def test_no_fault_does_not_trigger_fault_latched():
    """The `fault` kwarg is opt-in evidence, not an ambient global -- a
    `check_approach` call with `fault=None` on an otherwise-fine setup must not
    itself raise `fault_latched` (it may still refuse for an unrelated reason,
    which is fine; only that specific reason is asserted against here)."""
    model = _model("mfp3d", z_park_v=0.0, z_park_tol_v=0.5)
    try:
        command_guard.check_approach(
            model, mode="contact", setpoint=0.5, free_air=_free_air(0.0),
            now=time.time(), fault=None)
    except command_guard.GuardRefused as exc:
        assert exc.reason != "fault_latched"


# =========================================================== worker join order ==

def test_worker_abort_and_join_happen_before_any_z_command():
    order: list[str] = []
    worker = _RecordingWorker(order=order)
    model = _model("cypher", z_park_v=0.0, z_park_tol_v=0.5)
    bridge = _Bridge(zsensor_v=-5.0, defl_v=2.0, order=order)
    hub = _FakeHub(model, bridge, litho_worker=worker, free_air=_free_air(0.0))

    safe_park(hub, "abort mid-write")

    assert worker.abort_requested is True
    assert worker.join_timeout is not None and worker.join_timeout != "unset"
    assert worker.is_alive() is False

    z_command_indices = [i for i, ev in enumerate(order)
                         if ev == "abort" or ev == "vendor_withdraw"]
    join_index = order.index("join")
    abort_index = order.index("request_abort")
    assert abort_index < join_index
    assert all(join_index < i for i in z_command_indices)


class _OrderingProxy:
    """Wraps a REAL `SoftwareARC` and forwards `.abort_retract` / `.read` /
    `.vendor_withdraw` to it, recording call order into a list SHARED with the
    real-thread worker below. Today `SoftwareARC` has none of these methods
    (T3's twin-cypher-profile RED), so this raises `AttributeError` the first
    time `safe_park` actually tries to command the Z channel -- which is
    itself the correct RED reason for MUST-FIX 3."""

    def __init__(self, arc, order: list):
        self._arc = arc
        self._order = order

    def abort_retract(self) -> None:
        self._order.append("abort")
        return self._arc.abort_retract()

    def confirm_retract(self) -> None:
        return self._arc.confirm_retract()

    def read(self, channel: str) -> float:
        return self._arc.read(channel)

    def vendor_withdraw(self):
        self._order.append("vendor_withdraw")
        return self._arc.vendor_withdraw()


class _RealThreadTwinWorker:
    """MUST-FIX 3: a REAL `threading.Thread` + `threading.Event`, driving a
    long "litho fill" directly against the twin ARC (`arc.step` in a loop),
    so `join()` genuinely blocks on real work finishing rather than a fake
    immediately reporting not-alive. Bounded to `_STEP_DEADLINE_S` on its OWN
    so a `safe_park` that never calls `request_abort()`/`join()` at all still
    cannot hang the suite -- the thread gives up and exits by itself."""

    _STEP_DEADLINE_S = 5.0

    def __init__(self, arc, order: list):
        self._arc = arc
        self._order = order
        self._abort = threading.Event()
        self._started = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        self._started.wait(timeout=2.0)

    def _run(self) -> None:
        self._started.set()
        deadline = time.monotonic() + self._STEP_DEADLINE_S
        while not self._abort.is_set() and time.monotonic() < deadline:
            self._arc.step(50)
            time.sleep(0.001)

    def request_abort(self) -> None:
        self._abort.set()
        self._order.append("request_abort")

    def join(self, timeout=None) -> None:
        # a real join: if the caller passes a timeout too short for the
        # thread to actually finish, it genuinely returns early and the
        # thread is genuinely still alive -- there is no fake to fool.
        self._thread.join(timeout=timeout)
        self._order.append("join")

    def is_alive(self) -> bool:
        return self._thread.is_alive()


def test_real_thread_worker_is_dead_before_any_z_command_on_the_twin(monkeypatch):
    """MUST-FIX 3 / Sec 6: "park during a running write joins the worker first
    (no Z command before join)" -- proved here against REAL threading
    primitives and a REAL `SoftwareARC` long fill, not a fake that reports
    not-alive without ever having been alive. Bounded to
    `_RealThreadTwinWorker._STEP_DEADLINE_S` (5 s) so a `safe_park` that never
    joins at all cannot hang the suite -- the worker thread gives up on its
    own and the assertions below simply fail instead."""
    monkeypatch.setenv("AFM_LITHO_TWIN_PLATFORM", "cypher")
    arc = SoftwareARC(World(sample=Sample(height_map=np.zeros((8, 8)), extent=1e-5)))

    order: list[str] = []
    worker = _RealThreadTwinWorker(arc, order=order)
    worker.start()
    assert worker.is_alive() is True        # genuinely running, not a stub

    model = _model("cypher", z_park_v=0.0, z_park_tol_v=0.5)
    bridge = _OrderingProxy(arc, order)
    hub = _FakeHub(model, bridge, litho_worker=worker, free_air=_free_air(0.0))

    safe_park(hub, "abort mid-write")

    assert worker.is_alive() is False, (
        "join(timeout) must actually wait for the real thread to exit, not "
        "return early while it is still running")
    assert "request_abort" in order
    join_index = order.index("join")
    abort_index = order.index("request_abort")
    assert abort_index < join_index
    z_indices = [i for i, ev in enumerate(order)
                if ev == "abort" or ev == "vendor_withdraw"]
    assert z_indices, "safe_park issued no Z command at all"
    assert all(join_index < i for i in z_indices)


def test_no_worker_present_is_a_harmless_noop_for_the_join_step():
    model = _model("mfp3d", z_park_v=0.0, z_park_tol_v=0.5)
    bridge = _Bridge(zsensor_v=-5.0, defl_v=2.0)
    hub = _FakeHub(model, bridge, litho_worker=None, free_air=_free_air(0.0))

    result = safe_park(hub, "idle park")

    assert result.method == "abort"


# ===================================================== REST: /api/fault/clear ==

def test_fault_clear_requires_confirm_true(client):
    from afm_litho.service.hub import hub as real_hub

    prev = real_hub.fault
    real_hub.fault = {"reason": "park_unverified", "critical": True}
    try:
        r = client.post("/api/fault/clear", json={})
        assert r.status_code != 200 or real_hub.fault is not None
    finally:
        real_hub.fault = prev


def test_fault_clear_with_confirm_true_clears_the_latch(client):
    from afm_litho.service.hub import hub as real_hub

    prev = real_hub.fault
    real_hub.fault = {"reason": "park_unverified", "critical": True}
    try:
        r = client.post("/api/fault/clear", json={"confirm": True})
        assert r.status_code == 200
        assert real_hub.fault is None
    finally:
        real_hub.fault = prev
