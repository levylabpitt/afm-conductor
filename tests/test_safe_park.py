"""Host-side fail-safe: the `safe_park` primitive + every failure path wired to it.

This is the safety-critical build (docs/SAFETY_KERNEL_PLAN.md Sec 0): safe_park =
abort = fail-safe retract -- ONE out-of-band primitive, two triggers (operator Abort
OR any auto-detected failure). The audit found the live system "holds feedback on
failure"; the rule is now ANY failure -> safe_park. Three layers, all zero-hardware:

- UNIT over the primitive itself (service/host/safe_park.py) driven against a
  HardwareARC over FakeIgorBridge (the same double the hardware tests use): the
  out-of-band trigger (COM-first / TCP-fallback), the Output.Z readback VERIFY,
  the retry-once-then-CRITICAL branch, idempotency, the twin path, abort-awareness.

- UNIT over the free-air 3-read guard truth table (host/continuous).

- SERVICE over the twin via the TestClient: EACH continuous-loop failure class
  (a cap, a single terminal pass fault, a supervisor exception, a reref NaN read, a
  3-read disagreement, free-air / effective > 0.9*SUM, SUM-loss, a router-failure
  read) trips safe_park; and the hardware-pump fault -> terminal stop (blocker #3 +
  fix 3) + the chained-scan error surfacing (blocker #2).

`client` comes from tests/conftest.py (a TestClient).
"""

from __future__ import annotations

import asyncio
import math
import sys
import time

import pytest

from afm_litho.hardware.hardware_arc import HardwareARC
from afm_litho.hardware.igor_stub import FakeIgorBridge
# MIGRATED (B1, docs/INSTRUMENT_MODEL_SPEC.md Sec 3 + 10): the async out-of-band
# hardware orchestrator this whole file exercises (COM/TCP trigger, failsafe
# watchdog, bounded Output.Z verify, ws broadcast) is now named `safe_park_async`.
# The bare name `safe_park` is the NEW synchronous, model-driven ParkResult
# primitive (tested in tests/test_safe_park_model.py). Aliasing keeps every case
# below unchanged while pointing them at the orchestrator they were written for.
from afm_litho.service.host.safe_park import safe_park_async as safe_park

# the SUBMODULE object (the package __init__ rebinds the `safe_park` attribute to the
# function, so `import ...safe_park as m` would grab the function -- reach the module
# through sys.modules instead), for patching its _VERIFY_SETTLE_S in the units.
safe_park_mod = sys.modules[safe_park.__module__]

# Park verify, the failsafe done-wait and the stop polls are answered by other real
# threads, so every case here keeps real waits (recorded) rather than the testing clock.
pytestmark = pytest.mark.time_sensitive


@pytest.fixture(autouse=True)
def _fast_verify_settle(monkeypatch):
    """Keep the primitive units fast. The real hardware verify WAITS
    safe_park._VERIFY_SETTLE_S (~2.5 s, == hw_withdraw's settle) for the Output.Z
    ramp before reading back; patch it down for the units. The dedicated async-ramp
    ordering test overrides both this settle and the stub's ramp delay to assert the
    ordering explicitly, so it does not rely on this default."""
    monkeypatch.setattr(safe_park_mod, "_VERIFY_SETTLE_S", 0.05)


# ============================================================ primitive units ==
#
# REWRITTEN (B1b, spec 13.1 item 7): `safe_park_async` is now a thin transport/watchdog
# WRAPPER whose retract + verify IS the synchronous, model-driven `safe_park` primitive.
# The old contract verified from the Output.Z READBACK reaching ~0 (the commanded value
# -- exactly the 2026-09-22 incident path); the single contract verifies from the
# INDEPENDENT ZSensor + Deflection sensors and never from the commanded value. Each test
# below names the specific old-vs-new contradiction it was migrated across.
#
# The fakes gain what the primitive reads: a twin-defaults `instrument_model`
# (park_plan channel Output.Z, park_v 0, tol 0.5), a fresh withdrawn `free_air`
# baseline, and a `hardware_arc` whose ZSensor/Deflection the test drives (via the
# FakeIgorBridge value store) to model an engaged / retracting / stuck tip. The
# out-of-band COM/TCP AFM_SafePark trigger, the ws broadcast and the safety record are
# UNCHANGED and still asserted.

import numpy as np

from _im_fixtures_guard import make_model


class _Store:
    def __init__(self):
        self.records: list[tuple] = []

    def record(self, kind, payload):
        self.records.append((kind, payload))
        return {"id": "rec", "kind": kind}


class _Inst:
    """Minimal twin instrument for the safe_park twin branch (only `abort_scan` is
    called now -- the twin retract itself is the sync primitive on `hub.arc`)."""
    def __init__(self):
        self.calls: list[str] = []

    def abort_scan(self):
        self.calls.append("abort_scan")

    def withdraw(self):                     # kept for safety; not called any more
        self.calls.append("withdraw")
        return {"z_um": -10.0}


class _Com:
    """A COM Igor handle stand-in. `reject_safepark` makes the out-of-band
    `AFM_SafePark()` trigger raise (forcing the TCP fallback); `on_withdraw` fires when
    the vendor withdraw fallback (`DoScanFunc("StopEngageButton")`) is called, so a test can model the vendor retract
    recovering the sensors."""
    def __init__(self, *, reject_safepark: bool = False, on_withdraw=None):
        self.calls: list[str] = []
        self.reject_safepark = reject_safepark
        self._on_withdraw = on_withdraw

    def execute(self, cmd, check=False):
        self.calls.append(cmd)
        if "StopEngageButton" in cmd:               # the vendor withdraw button's action
            if self._on_withdraw:
                self._on_withdraw()
            return 0
        if self.reject_safepark:
            raise RuntimeError("COM rejected")
        return 0


class _FakeHub:
    """The slice of the hub the park primitive + its wrapper touch, hand-built so the
    unit runs without the whole service. Now carries the `instrument_model`,
    `free_air`, `litho_worker` and `last_park` the sync primitive reads."""
    def __init__(self, *, hardware_arc=None, inst=None, arc=None, free_air_v=0.0):
        import threading
        import time as _t

        from afm_litho.service.command_guard import FreeAir
        self.hardware_arc = hardware_arc
        self.arc = arc
        self.inst = inst
        self.instrument_model = make_model(z_out_channel="Output.Z", z_park_v=0.0,
                                           z_park_tol_v=0.5)
        self.litho_worker = None
        self.free_air = FreeAir(value_v=free_air_v, ts=_t.time(), mode="contact",
                                withdrawn=True)
        self.fault = None
        self.last_park = None
        self.engaged = True
        self.scan_repeat = {"enabled": True, "stopped_reason": None}
        self._scan_repeat_stop = threading.Event()
        self._hw_abort = threading.Event()
        self._hw_litho_abort = threading.Event()
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


def _stub(*, zsensor: float, defl: float, height: float = 0.0) -> FakeIgorBridge:
    """A bridge whose ZSensor/Deflection (what the sync primitive VERIFIES from) and
    Output.Z/height readback the test controls."""
    return FakeIgorBridge({"ZSensor": zsensor, "Deflection": defl,
                           "height": height, "Output.Z": height})


def test_safe_park_hardware_verifies_from_the_sensors_not_the_command():
    """CONTRADICTION (13.1 item 7): OLD -- verified when the Output.Z READBACK reached
    ~0 (the commanded value). NEW -- verified when the INDEPENDENT ZSensor + Deflection
    read retracted; the commanded value is never evidence. The out-of-band AFM_SafePark
    trigger (TCP fallback, no COM), the ws broadcast and the safety record are unchanged."""
    stub = _stub(zsensor=0.0, defl=0.0, height=5.0)   # sensors already retracted
    hw = _hw_over(stub)
    hub = _FakeHub(hardware_arc=hw)
    try:
        res = asyncio.run(safe_park(hub, "unit: verify sensors"))
        assert res["backend"] == "hardware"
        assert res["transport"] == "tcp"        # no COM handle -> TCP AFM_SafePark
        assert res["verified"] is True and res["critical"] is False
        assert res["zsensor_v"] == pytest.approx(0.0)
        assert res["defl_v"] == pytest.approx(0.0)
        assert stub.safe_parks >= 1             # AFM_SafePark actually fired
        assert any(k == "safety" for k, _ in hub.store.records)
        assert any(m.get("type") == "safe_park" for m in hub.broadcasts)
    finally:
        hw.close()
        stub.stop()


def test_safe_park_unverified_sensors_fire_vendor_then_flag_critical():
    """CONTRADICTION (13.1 item 7 / Sec 3 steps 4-5): OLD -- retry the Output.Z trigger
    ONCE, then CRITICAL if it still would not read 0. NEW -- a failed SENSOR verify
    fires the ONE vendor_withdraw fallback (here it cannot recover: no COM handle), then
    CRITICAL with a latched fault. The ZSensor stays engaged throughout."""
    stub = _stub(zsensor=5.0, defl=2.0, height=50.0)  # ZSensor stuck engaged
    stub.safe_park_zeros = False                      # the retract does NOT recover it
    hw = _hw_over(stub)                                # no COM -> vendor_withdraw raises
    hub = _FakeHub(hardware_arc=hw)
    try:
        res = asyncio.run(safe_park(hub, "unit: stuck engaged"))
        assert res["verified"] is False and res["critical"] is True
        assert res["zsensor_v"] == pytest.approx(5.0)
        assert hub.fault["reason"] == "park_unverified"   # latched with the reason
        assert any("CRITICAL" in m for m in hub.logs)     # operator-facing critical log
    finally:
        hw.close()
        stub.stop()


def test_safe_park_vendor_withdraw_fallback_recovers_the_sensors():
    """CONTRADICTION (13.1 item 7): OLD -- the retract verified only on the SECOND
    AFM_SafePark (a re-fire of the same commanded ramp). NEW -- the ramp verify fails,
    the ONE vendor_withdraw fallback retracts (the sensors recover), method
    'vendor_withdraw'."""
    stub = _stub(zsensor=5.0, defl=2.0, height=50.0)
    stub.safe_park_zeros = False                      # only the vendor withdraw recovers

    def _recover():
        stub.values.update(ZSensor=0.0, Deflection=0.0, Height=0.0)
        stub.values["Output.Z"] = 0.0

    com = _Com(on_withdraw=_recover)
    hw = _hw_over(stub, com_igor=com)
    hub = _FakeHub(hardware_arc=hw)
    try:
        res = asyncio.run(safe_park(hub, "unit: vendor recovers"))
        assert res["verified"] is True and res["critical"] is False
        assert res["method"] == "vendor_withdraw"
        assert 'DoScanFunc("StopEngageButton")' in com.calls
        assert hub.fault is None
    finally:
        hw.close()
        stub.stop()


def test_safe_park_polls_the_sensors_across_the_settle_window():
    """Slice AB: polls the Height confirm (< 1 V) across the settle window; it must wait
    PAST Igor's retract (>= the recovery delay) and stop once Height lands, not sit out the
    whole window."""
    stub = _stub(zsensor=66.0, defl=3.0, height=66.0)   # engaged ~66 V (rig-measured)
    stub.safe_park_zeros = False                        # only the delayed timer recovers
    import threading as _thr
    _thr.Timer(0.25, lambda: stub.values.update(
        {"ZSensor": 0.0, "Deflection": 0.0, "Height": 0.0, "Output.Z": 0.0})).start()
    hw = _hw_over(stub)
    hub = _FakeHub(hardware_arc=hw)
    try:
        safe_park_mod._VERIFY_SETTLE_S = 0.8            # (autouse patched 0.05; override)
        safe_park_mod._VERIFY_POLL_S = 0.02
        t0 = time.monotonic()
        res = asyncio.run(safe_park(hub, "unit: async ramp"))
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.20                          # waited for the sensors to land
        assert elapsed < 2.0                            # ... and stopped once they did
        assert res["verified"] is True and res["critical"] is False
        assert res["method"] == "abort"
    finally:
        hw.close()
        stub.stop()


def test_safe_park_sensors_that_never_settle_flag_critical():
    """CONTRADICTION (13.1 item 7): the flip side -- when the ZSensor never returns to
    the withdrawn value within the window (and no COM vendor fallback is available),
    the park is CRITICAL with a latched fault. Proves the verify is the sensor, not the
    command (a host that trusted the commanded ramp would false-'verify' here)."""
    stub = _stub(zsensor=66.0, defl=3.0, height=66.0)   # never recovers
    stub.safe_park_zeros = False                        # the retract does NOT recover it
    hw = _hw_over(stub)                                  # no COM
    hub = _FakeHub(hardware_arc=hw)
    try:
        safe_park_mod._VERIFY_SETTLE_S = 0.05
        res = asyncio.run(safe_park(hub, "unit: never settles"))
        assert res["verified"] is False and res["critical"] is True
        assert res["zsensor_v"] == pytest.approx(66.0)
        assert hub.fault
    finally:
        hw.close()
        stub.stop()


def test_safe_park_total_transport_failure_criticals_without_raising():
    """When BOTH transports are dead (COM rejects AND TCP link.send raises) the
    out-of-band trigger reports transport 'failed'; the sync primitive that follows
    cannot read/ramp/withdraw either, so the park is CRITICAL -- but safe_park NEVER
    raises (that would skip the CRITICAL log/record/broadcast in the exact comms-dead
    case it exists for)."""
    class _DeadLink:
        def send(self, line):
            raise OSError("link dead")

    class _DeadHw:
        def __init__(self):
            self.link = _DeadLink()
            self.com_igor = _Com(reject_safepark=True)   # COM rejects

    hub = _FakeHub(hardware_arc=_DeadHw())
    res = asyncio.run(safe_park(hub, "unit: dead link"))   # must NOT raise
    assert res["backend"] == "hardware"
    assert res["transport"] == "failed"
    assert res["critical"] is True and res["verified"] is False
    assert any("CRITICAL" in m for m in hub.logs)
    assert any(k == "safety" and p.get("critical") for k, p in hub.store.records)


def test_safe_park_uses_com_first_when_available():
    """With a COM handle the out-of-band AFM_SafePark fires over COM (transport 'com',
    no TCP ABORT); the retract then verifies from the sensors (already retracted)."""
    stub = _stub(zsensor=0.0, defl=0.0, height=0.0)
    com = _Com()
    hw = _hw_over(stub, com_igor=com)
    hub = _FakeHub(hardware_arc=hw)
    try:
        res = asyncio.run(safe_park(hub, "unit: com path"))
        assert res["transport"] == "com"
        assert com.calls == ["AFM_SafePark()"]      # verified on the ramp -> no vendor
        assert stub.aborts == 0                     # COM path did NOT send a TCP ABORT
        assert res["verified"] is True
    finally:
        hw.close()
        stub.stop()


def test_safe_park_falls_back_to_tcp_when_com_rejects():
    """A COM-level rejection of the trigger falls back to the TCP AFM_SafePark; the
    sensor verify (already retracted) still confirms the park."""
    stub = _stub(zsensor=0.0, defl=0.0, height=5.0)
    com = _Com(reject_safepark=True)
    hw = _hw_over(stub, com_igor=com)
    hub = _FakeHub(hardware_arc=hw)
    try:
        res = asyncio.run(safe_park(hub, "unit: com rejects"))
        assert res["transport"] == "tcp"
        assert stub.aborts >= 1                     # fell back to the TCP ABORT
        assert res["verified"] is True
    finally:
        hw.close()
        stub.stop()


def test_safe_park_is_abort_aware_breaks_the_loop_and_signals_stops():
    """safe_park breaks any running loop AT ONCE: it disables continuous and sets every
    cooperative abort event, so an in-flight pass/scan/write stops (unchanged)."""
    stub = _stub(zsensor=0.0, defl=0.0, height=0.0)
    hw = _hw_over(stub)
    hub = _FakeHub(hardware_arc=hw)
    try:
        asyncio.run(safe_park(hub, "unit: abort-aware"))
        assert hub.host.scan_repeat["enabled"] is False
        assert hub.host._scan_repeat_stop.is_set()
        assert hub.host._hw_abort.is_set()
        assert hub.host._hw_litho_abort.is_set()
        assert hub.host._safe_park["active"] is True
        assert hub.host._safe_park["count"] == 1
    finally:
        hw.close()
        stub.stop()


def test_safe_park_is_idempotent():
    """Re-triggering is harmless, so safe_park is safe to call repeatedly (every
    failure path may call it): two verified parks, count 2, no latched fault."""
    stub = _stub(zsensor=0.0, defl=0.0, height=0.0)
    hw = _hw_over(stub)
    hub = _FakeHub(hardware_arc=hw)
    try:
        r1 = asyncio.run(safe_park(hub, "first"))
        r2 = asyncio.run(safe_park(hub, "second"))
        assert r1["verified"] and r2["verified"]
        assert hub.host._safe_park["count"] == 2
        assert hub.fault is None
    finally:
        hw.close()
        stub.stop()


def test_safe_park_twin_stops_and_parks_through_the_sync_primitive():
    """CONTRADICTION (13.1 item 7 vs 8, E2): OLD -- the twin path stopped the sim scan
    then ran the twin's native `inst.withdraw`, verified by construction and NEVER
    stored on `hub.last_park` (so `GET /api/park/last` / `getLastPark` returned `{}`
    after a twin abort while hardware was populated). NEW -- C1 bound hub.hardware_arc
    to the twin SoftwareARC, so the twin routes through the SAME model-driven sync
    primitive as hardware: it ramps `park_plan(model).ramp_channel` on the twin ARC,
    VERIFIES from the twin's ZSensor + Deflection, and STORES the ParkResult on
    `hub.last_park`. `inst.withdraw` is no longer called (only the sim scan is stopped);
    `method` is "abort"/"vendor_withdraw" exactly as on the rig."""
    from afm_litho.twin.arc import SoftwareARC
    from afm_litho.twin.physics import Sample, World

    arc = SoftwareARC(World(sample=Sample(height_map=np.zeros((8, 8)), extent=1e-5)))
    inst = _Inst()
    hub = _FakeHub(hardware_arc=None, arc=arc, inst=inst)
    res = asyncio.run(safe_park(hub, "unit: twin"))
    assert res["backend"] == "twin"
    assert res["verified"] is True and res["critical"] is False
    assert res["transport"] == "twin"
    assert res["method"] == "abort"                        # through the sync primitive
    assert inst.calls == ["abort_scan"]                   # sim scan stopped; NO native withdraw
    # E2: the twin abort now STORES the ParkResult, so the read doors are populated.
    assert hub.host.last_park is not None
    assert hub.host.last_park.verified is True
    assert hub.host.last_park.reason == "unit: twin"


class _TipDownPlugin:
    """A twin plugin whose tip stays down however it is told to withdraw."""

    def __init__(self):
        self.told = []

    def abort(self, reason):
        self.told.append(reason)

    def withdrawn(self):
        return False


def test_a_twin_park_is_not_verified_while_the_plugins_tip_is_down(monkeypatch):
    """The old twin's sensors say the tip is up, but the plugin that ran the write or scan
    still has it down: the park must not claim a retract, and (the plugin is withdrawing,
    not stuck) must not latch a fault."""
    import importlib
    sp = importlib.import_module("afm_litho.service.host.safe_park")
    from afm_litho.twin.arc import SoftwareARC
    from afm_litho.twin.physics import Sample, World

    monkeypatch.setattr(sp, "_VERIFY_SETTLE_S", 0.05)
    arc = SoftwareARC(World(sample=Sample(height_map=np.zeros((8, 8)), extent=1e-5)))
    hub = _FakeHub(hardware_arc=None, arc=arc, inst=_Inst())
    hub.host._twin_plugin = _TipDownPlugin()
    res = asyncio.run(safe_park(hub, "unit: plugin tip down"))
    assert res["verified"] is False and res["critical"] is True
    assert hub.host.last_park.verified is False
    assert hub.fault is None
    assert hub.host._twin_plugin.told                      # it was told to withdraw


def test_a_hardware_park_ignores_a_stray_twin_plugin():
    """A plugin left on a hardware station never delays or fails the rig's park."""
    from afm_litho.service.host.safe_park import _twin_tip_still_down

    class _Hw:
        backend = "hardware"
        _twin_plugin = _TipDownPlugin()

        @property
        def host(self):                # fake: host fields live on the fake itself
            return self
    assert _twin_tip_still_down(_Hw()) is False


@pytest.mark.parametrize("twin_platform,z_out_channel",
                         [("mfp3d", "Height"), ("cypher", "Height")])
def test_safe_park_twin_populates_last_park_on_both_platforms(
        monkeypatch, twin_platform, z_out_channel):
    """E2: the async door wrapper (`safe_park_async` -- what POST /api/safe_park, FLEX
    `abort` and `safePark` all call) stores `hub.last_park` on the twin for BOTH
    `AFM_LITHO_TWIN_PLATFORM` profiles, ramping that profile's own Z output channel
    (Output.Z on mfp3d, Height on cypher) and verifying from the twin's sensors, so a
    park verifies via "abort" and the ParkResult is populated exactly as on the rig."""
    monkeypatch.setenv("AFM_LITHO_TWIN_PLATFORM", twin_platform)
    from afm_litho.twin.arc import SoftwareARC
    from afm_litho.twin.physics import Sample, World

    arc = SoftwareARC(World(sample=Sample(height_map=np.zeros((8, 8)), extent=1e-5)))
    assert arc.z_out_channel == z_out_channel     # sanity: the env selected the profile
    inst = _Inst()
    hub = _FakeHub(hardware_arc=None, arc=arc, inst=inst)
    # The twin model reports z_out_channel for the active profile (park_plan drives it).
    hub.host.instrument_model = make_model(platform="twin", z_out_channel=z_out_channel,
                                      z_park_v=0.0, z_park_tol_v=0.5)
    res = asyncio.run(safe_park(hub, f"unit: twin {twin_platform}"))
    assert res["backend"] == "twin" and res["transport"] == "twin"
    assert res["verified"] is True and res["critical"] is False
    assert res["method"] == "abort"
    assert inst.calls == ["abort_scan"]
    assert hub.host.last_park is not None and hub.host.last_park.verified is True
    assert hub.host.last_park.method == "abort"
    # the ramp aimed at THIS profile's Z output channel
    assert ("abort", 0.0) in arc.command_log              # Igor's abort function, no ramp


# ================================================ free-air 3-read guard table ==

def test_free_air_3read_guard_truth_table(monkeypatch):
    """The exact accept/escalate/park logic of the free-air 3-read 10% guard,
    including the first-frame (no last-accepted) branch."""
    import afm_litho.service.host.continuous as cont

    def run(values, last):
        it = iter(values)

        async def _r():
            return next(it)

        monkeypatch.setattr(cont, "_read_free_air_deflection", _r)
        return asyncio.run(cont._read_free_air_guarded(last))

    # fast path: r1 within 10% of the last accepted -> accept on ONE read
    v, why = run([0.105], 0.10)
    assert why is None and v == pytest.approx(0.105)

    # real drift: r1 far from last, but r2 agrees with r1 -> accept r2 (2 reads)
    v, why = run([0.30, 0.305], 0.10)
    assert why is None and v == pytest.approx(0.305)

    # first frame (no last): two must agree -> accept
    v, why = run([0.10, 0.105], None)
    assert why is None and v == pytest.approx(0.105)

    # first frame: r1,r2 disagree, escalate, r3 agrees with r2 -> use r3
    v, why = run([0.10, 0.30, 0.31], None)
    assert why is None and v == pytest.approx(0.31)

    # three inconsistent reads -> safe_park (unstable)
    v, why = run([0.10, 0.50, 0.99], None)
    assert v is None and "unstable" in why

    # a bad read (NaN) at any step is itself a safe_park
    v, why = run([math.nan], 0.10)
    assert v is None and "NaN" in why
    v, why = run([0.50, math.nan], 0.10)
    assert v is None and "NaN" in why


def test_read_sum_reads_detector_sum_not_sum():
    """Regression for the DetectorSum alias fix (2026-09-15): lowercase "sum"
    reads 0 V on the Cypher, so _read_sum must read the controller alias
    "DetectorSum". On the stub-backed hardware path, a value store keyed only
    "DetectorSum" (no lowercase "sum") must still resolve through _read_sum -- and
    asking the SAME stub for the retired "sum" name must come back distinguishable
    (GETVALUE has no such key -> NaN), not silently answer with the sum value."""
    import afm_litho.service.host.continuous as cont

    stub = FakeIgorBridge({"DetectorSum": 4.2})
    port = stub.serve_in_thread()
    hw = HardwareARC.connect("127.0.0.1", port)
    prev = cont.hub.host.hardware_arc
    cont.hub.host.hardware_arc = hw
    try:
        v = asyncio.run(cont._read_sum())
        assert v == pytest.approx(4.2), v

        bad = hw.td_ReadValue("sum")
        assert math.isnan(bad), (
            "the stub answered the retired 'sum' name instead of rejecting it "
            f"distinguishably: got {bad!r}"
        )
    finally:
        cont.hub.host.hardware_arc = prev
        hw.close()
        stub.stop()


def test_read_sum_maps_router_failure_to_nan():
    """A router failure (timeout / BridgeClosed) during the SUM read surfaces as NaN
    (-> the SUM-loss interlock fails safe), never an unhandled exception."""
    import afm_litho.service.host.continuous as cont
    from afm_litho.hardware.bridge_router import BridgeClosed

    class _Boom:
        def td_ReadValue(self, name):
            raise BridgeClosed("link dropped")

    prev = cont.hub.host.hardware_arc
    cont.hub.host.hardware_arc = _Boom()
    try:
        v = asyncio.run(cont._read_sum())
        assert math.isnan(v)
    finally:
        cont.hub.host.hardware_arc = prev


# ================================================= service: failure -> safe_park ==

def _wait(cond, timeout=40.0, interval=0.03):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return False


def _tmp_store(tmp_path):
    from afm_litho.service.app import hub
    from afm_litho.service.sessions import SessionStore
    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    return hub


def _stop_and_reset(client, hub):
    client.post("/api/scan/continuous", json={"enabled": False})
    _wait(lambda: client.get("/api/scan/continuous").json()["enabled"] is False, 10)
    hub.host.scan_repeat.update({
        "enabled": False, "config": None, "count": 0, "started_at": None,
        "errors": 0, "max_frames": None, "max_seconds": None,
        "interval_s": 0.0, "withdraw_on_stop": False, "stopped_reason": None,
        "_sum_low_streak": 0})
    hub.host.relative_setpoint.update({
        "enabled": False, "settle_s": 4.0,
        "last_free_air_v": None, "last_effective_v": None})
    hub.host._safe_park["active"] = False


def _spy_safe_park(monkeypatch, cont):
    """Replace cont.safe_park with a spy that records the reason and mimics its
    idempotency latch (so the supervisor finally doesn't double-park)."""
    from afm_litho.service.app import hub
    calls: list[str] = []

    async def _spy(_hub, reason):
        calls.append(reason)
        hub.host._safe_park["active"] = True
        return {"reason": reason, "backend": "twin", "transport": "twin",
                "verified": True, "z_v": None, "critical": False}

    monkeypatch.setattr(cont, "safe_park", _spy)
    return calls


_TINY = {"size_um": 5.0, "pixels": 16, "lines": 4, "line_rate_hz": 200.0,
         "mode": "contact", "setpoint": 0.20}


def test_max_frames_termination_safeparks(client, tmp_path, monkeypatch):
    import afm_litho.service.host.continuous as cont
    hub = _tmp_store(tmp_path)
    _stop_and_reset(client, hub)
    calls = _spy_safe_park(monkeypatch, cont)
    try:
        client.post("/api/scan/continuous",
                    json={"enabled": True, "config": _TINY, "max_frames": 2})
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False)
        assert client.get("/api/scan/continuous").json()["stopped_reason"] == "max_frames"
        assert any("max_frames" in r for r in calls), calls
    finally:
        _stop_and_reset(client, hub)


def test_max_seconds_termination_safeparks(client, tmp_path, monkeypatch):
    import afm_litho.service.host.continuous as cont
    hub = _tmp_store(tmp_path)
    _stop_and_reset(client, hub)
    calls = _spy_safe_park(monkeypatch, cont)
    try:
        client.post("/api/scan/continuous",
                    json={"enabled": True, "config": _TINY, "max_seconds": 0.01,
                          "max_frames": 100})
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False)
        assert client.get("/api/scan/continuous").json()["stopped_reason"] == "max_seconds"
        assert any("max_seconds" in r for r in calls), calls
    finally:
        _stop_and_reset(client, hub)


def test_single_pass_fault_is_terminal_and_safeparks(client, tmp_path, monkeypatch):
    """Fix 3 (operator decision): a hardware pass fault is TERMINAL -- there is NO
    retry tolerance. ONE injected pass exception stops the loop AT ONCE (reason
    "fault") and safe_parks; it never arms another frame on the unknown Z state."""
    import afm_litho.service.host.continuous as cont
    hub = _tmp_store(tmp_path)
    _stop_and_reset(client, hub)
    calls = _spy_safe_park(monkeypatch, cont)

    frames = {"n": 0}

    async def _boom(_cfg):
        frames["n"] += 1
        raise RuntimeError("wedged rig")

    monkeypatch.setattr(cont, "scan_start", _boom)
    try:
        client.post("/api/scan/continuous", json={"enabled": True, "config": _TINY})
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False)
        st = client.get("/api/scan/continuous").json()
        assert st["stopped_reason"] == "fault", st
        assert st["count"] == 0, st                 # no frame ever completed
        assert frames["n"] == 1, frames             # stopped after the FIRST fault
        assert any("fault" in r for r in calls), calls
    finally:
        _stop_and_reset(client, hub)


def test_supervisor_exception_safeparks(client, tmp_path, monkeypatch):
    """An exception in the supervisor loop body (not caught by a frame) still
    terminates through safe_park -- no path leaves the tip engaged."""
    import afm_litho.service.host.continuous as cont
    hub = _tmp_store(tmp_path)
    _stop_and_reset(client, hub)
    calls = _spy_safe_park(monkeypatch, cont)

    async def _explode():
        raise RuntimeError("supervisor blew up")

    monkeypatch.setattr(cont, "_run_one_frame", _explode)
    try:
        client.post("/api/scan/continuous", json={"enabled": True, "config": _TINY})
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False)
        assert calls, "supervisor exception did not safe_park"
    finally:
        _stop_and_reset(client, hub)


def test_hardware_pump_fault_is_terminal_and_safeparks(client, tmp_path, monkeypatch):
    """Blocker #3 + Fix 3: the hardware scan pump used to SWALLOW a pass exception,
    so `await task` never raised and the fault was invisible -- a wedged rig looped
    forever. The pump now records the fault on hub._scan_pump_error; _run_one_frame
    reads it AFTER awaiting the pump and returns a frame FAILURE, which is now
    TERMINAL: the loop stops on the FIRST such fault (no retry) and safe_parks."""
    import afm_litho.service.host.continuous as cont
    hub = _tmp_store(tmp_path)
    _stop_and_reset(client, hub)
    calls = _spy_safe_park(monkeypatch, cont)

    frames = {"n": 0}

    async def _fault(_cfg):
        frames["n"] += 1

        async def _pump():
            hub.host._scan_pump_error = "hardware scan failed: boom"   # swallowed fault
        hub.host._scan_task = asyncio.create_task(_pump())
        return {"ok": True}

    monkeypatch.setattr(cont, "scan_start", _fault)
    try:
        client.post("/api/scan/continuous", json={"enabled": True, "config": _TINY})
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False)
        st = client.get("/api/scan/continuous").json()
        assert st["stopped_reason"] == "fault", st
        assert st["count"] == 0, st                # no frame ever "succeeded"
        assert frames["n"] == 1, frames            # stopped after the FIRST fault
        assert any("fault" in r for r in calls), calls
    finally:
        _stop_and_reset(client, hub)


def test_safe_park_endpoint_is_the_operator_abort(client, tmp_path):
    """Abort == safe_park unification: POST /api/safe_park is the one primitive the
    operator Abort button fires (the twin path here). It stops + retracts and
    records the park."""
    hub = _tmp_store(tmp_path)
    _stop_and_reset(client, hub)
    try:
        j = client.post("/api/safe_park", json={"reason": "operator abort"}).json()
        assert j["ok"] is True
        assert j["safe_park"]["backend"] == "twin"
        assert j["safe_park"]["verified"] is True
        assert j["safe_park"]["critical"] is False
        assert hub.host._safe_park["last_reason"] == "operator abort"
    finally:
        _stop_and_reset(client, hub)


# ----- reactive reref monitors (relative mode) -----

def _arm_relative(client, cfg, max_frames=10):
    client.post("/api/scan/relative", json={"enabled": True, "settle_s": 0})
    client.post("/api/scan/continuous",
                json={"enabled": True, "config": cfg, "max_frames": max_frames})


def test_reref_nan_read_safeparks(client, tmp_path, monkeypatch):
    import afm_litho.service.host.continuous as cont
    hub = _tmp_store(tmp_path)
    _stop_and_reset(client, hub)
    calls = _spy_safe_park(monkeypatch, cont)

    async def _read():
        return math.nan

    monkeypatch.setattr(cont, "_read_free_air_deflection", _read)
    try:
        _arm_relative(client, _TINY)
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False)
        assert any("free-air read failed" in r for r in calls), calls
    finally:
        _stop_and_reset(client, hub)


def test_reref_3read_disagreement_safeparks(client, tmp_path, monkeypatch):
    import afm_litho.service.host.continuous as cont
    hub = _tmp_store(tmp_path)
    _stop_and_reset(client, hub)
    calls = _spy_safe_park(monkeypatch, cont)
    seq = iter([0.10, 0.50, 0.99] * 4)          # three wildly disagreeing reads

    async def _read():
        return next(seq)

    monkeypatch.setattr(cont, "_read_free_air_deflection", _read)
    try:
        _arm_relative(client, _TINY)
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False)
        assert any("unstable" in r for r in calls), calls
    finally:
        _stop_and_reset(client, hub)


def test_reref_free_air_over_point9_sum_safeparks(client, tmp_path, monkeypatch):
    """A free-air read past 0.9*SUM is unphysical (|deflection| <= SUM) -> safe_park."""
    import afm_litho.service.host.continuous as cont
    hub = _tmp_store(tmp_path)
    _stop_and_reset(client, hub)
    calls = _spy_safe_park(monkeypatch, cont)

    async def _read():
        return 0.5                               # free-air 0.5 V

    async def _sum():
        return 0.2                               # 0.9*SUM = 0.18 < 0.5

    monkeypatch.setattr(cont, "_read_free_air_deflection", _read)
    monkeypatch.setattr(cont, "_read_sum", _sum)
    try:
        _arm_relative(client, _TINY)
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False)
        assert any("free-air read" in r and "0.9*SUM" in r for r in calls), calls
    finally:
        _stop_and_reset(client, hub)


def test_reref_effective_over_point9_sum_safeparks(client, tmp_path, monkeypatch):
    """The free-air read is under 0.9*SUM but free_air + base setpoint exceeds it ->
    safe_park on the effective setpoint."""
    import afm_litho.service.host.continuous as cont
    hub = _tmp_store(tmp_path)
    _stop_and_reset(client, hub)
    calls = _spy_safe_park(monkeypatch, cont)
    cfg = dict(_TINY, setpoint=0.6)              # base 0.6 -> effective 0.5+0.6 = 1.1

    async def _read():
        return 0.5                               # < 0.9*SUM (0.9)

    async def _sum():
        return 1.0                               # 0.9*SUM = 0.9; effective 1.1 > 0.9

    monkeypatch.setattr(cont, "_read_free_air_deflection", _read)
    monkeypatch.setattr(cont, "_read_sum", _sum)
    try:
        _arm_relative(client, cfg)
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False)
        assert any("effective setpoint" in r and "0.9*SUM" in r for r in calls), calls
    finally:
        _stop_and_reset(client, hub)


def test_reref_sum_loss_two_low_reads_safeparks(client, tmp_path, monkeypatch):
    """Two CONSECUTIVE SUM reads below 0.1 V (laser/lever gone, feedback blind) ->
    safe_park. Kept in-bounds otherwise so ONLY the SUM-loss interlock trips."""
    import afm_litho.service.host.continuous as cont
    hub = _tmp_store(tmp_path)
    _stop_and_reset(client, hub)
    calls = _spy_safe_park(monkeypatch, cont)
    cfg = dict(_TINY, setpoint=0.01)             # tiny base so 0.9*SUM never trips first

    async def _read():
        return 0.01                              # tiny free-air, within 0.9*0.05

    async def _sum():
        return 0.05                              # < 0.1 V every reref

    monkeypatch.setattr(cont, "_read_free_air_deflection", _read)
    monkeypatch.setattr(cont, "_read_sum", _sum)
    try:
        _arm_relative(client, cfg)
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False)
        assert any("SUM lost" in r for r in calls), calls
    finally:
        _stop_and_reset(client, hub)


def test_reref_router_failure_read_safeparks(client, tmp_path, monkeypatch):
    """A router failure (timeout / BridgeClosed) during the SUM read surfaces as NaN
    -> the SUM-loss interlock treats it as a fail-safe -> safe_park."""
    import afm_litho.service.host.continuous as cont
    hub = _tmp_store(tmp_path)
    _stop_and_reset(client, hub)
    calls = _spy_safe_park(monkeypatch, cont)

    async def _sum():
        return math.nan                          # router failure -> NaN

    monkeypatch.setattr(cont, "_read_sum", _sum)
    try:
        _arm_relative(client, _TINY)
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False)
        assert any("SUM read failed" in r for r in calls), calls
    finally:
        _stop_and_reset(client, hub)


# ----- blocker #2: a chained-scan error surfaces (distinct from an operator abort) --

def test_chained_scan_error_surfaces_in_summary():
    """A chained-scan bridge/router fault comes back as summary['error'] (with
    aborted=True), so the pump can treat it as a FRAME FAILURE -- not a silent
    operator-style abort (blocker #2)."""
    from afm_litho.service.hw_scan import run_hw_scan_chained
    from afm_litho.service.passes import ScanConfig

    stub = FakeIgorBridge({})
    stub.scanchain_err = "router closed mid-scan"
    hw = _hw_over(stub)
    cfg = ScanConfig(size_um=5.0, pixels=8, lines=4, line_rate_hz=50.0,
                     mode="contact", trace_retrace=False, chained=True)
    cal = {"z_lvdt_sens": 1.0, "z_lvdt_offset": 0.0,
           "x_lvdt_sens": 1.0, "x_lvdt_offset": 0.0,
           "y_lvdt_sens": 1.0, "y_lvdt_offset": 0.0,
           "x_pgain": 0.0, "x_igain": 1.0, "y_pgain": 0.0, "y_igain": 1.0}

    async def _bc(_msg):
        return None

    try:
        summary = asyncio.run(run_hw_scan_chained(hw, cfg, cal, _bc))
        assert summary["aborted"] is True
        assert summary["error"] and "router closed" in summary["error"]
    finally:
        hw.close()
        stub.stop()


# ================================================ slice AB: Igor's one abort function ==

def test_park_confirms_on_height_and_clears_igors_abort_flag():
    """The out-of-band trigger runs AFM_SafePark (td_Stop once, bias 0, -10 V re-arm); the
    host sends no Z command of its own, confirms Height < 1 V and tells Igor."""
    stub = _stub(zsensor=3.17, defl=-0.5, height=50.0)
    hw = _hw_over(stub)
    hub = _FakeHub(hardware_arc=hw)
    try:
        res = asyncio.run(safe_park(hub, "unit: confirm"))
        assert res["verified"] is True and res["method"] == "abort"
        assert stub.td_stops == 1 and stub.abort_confirms == 1
        assert stub.abort_stopped is False and stub._backup is None
        assert "STOPLOOP" not in stub.transcript and "RAMP" not in stub.transcript
        assert "ABORTFAULT" in stub.transcript
        assert hub.fault is None
    finally:
        hw.close()
        stub.stop()


def test_park_latches_the_fault_igors_backup_left():
    """Igor's 5 s backup ran the vendor Withdraw while nobody confirmed: the next park reads
    Igor's fault flag (and Igor clears it) and latches `hub.fault`."""
    stub = _stub(zsensor=0.0, defl=0.0, height=0.0)
    stub.abort_fault = True
    hw = _hw_over(stub)
    hub = _FakeHub(hardware_arc=hw)
    try:
        asyncio.run(safe_park(hub, "unit: igor fault"))
        assert hub.fault and "5 s abort backup" in hub.fault["detail"]
    finally:
        hw.close()
        stub.stop()


def test_unconfirmed_abort_leaves_igors_flag_set_for_the_backup():
    """Height never comes under 1 V and the vendor withdraw cannot run (no COM): the park is
    CRITICAL, no ABORTCONFIRM is sent, and Igor's flag and 5 s backup stay armed."""
    stub = _stub(zsensor=5.0, defl=2.0, height=50.0)
    stub.safe_park_zeros = False
    hw = _hw_over(stub)
    hub = _FakeHub(hardware_arc=hw)
    try:
        res = asyncio.run(safe_park(hub, "unit: unconfirmed"))
        assert res["critical"] is True and hub.fault["reason"] == "park_unverified"
        assert stub.abort_confirms == 0
        assert stub.abort_stopped is True and stub._backup is not None
    finally:
        hw.close()
        stub.stop()


def test_operator_fault_clear_resets_igors_flags_so_the_next_abort_stops(monkeypatch):
    """Review must-fix (2026-10-27): a park that never confirms leaves Igor's stopped flag
    set (no ABORTCONFIRM). The operator's fault clear sends CLEARFAULT, so the abort of the
    next pass calls td_Stop again."""
    from afm_litho.hardware import bridge_protocol as P
    from afm_litho.service import hub as hub_mod
    from afm_litho.service.routers import instrument

    stub = _stub(zsensor=5.0, defl=2.0, height=50.0)
    stub.safe_park_zeros = False
    hw = _hw_over(stub)
    hub = _FakeHub(hardware_arc=hw)
    try:
        res = asyncio.run(safe_park(hub, "unit: fault path"))
        assert res["critical"] is True and stub.abort_stopped is True
        monkeypatch.setattr(hub_mod, "real_bridge", lambda h: hw)
        monkeypatch.setattr(instrument.hub, "fault", dict(hub.fault))
        assert instrument.post_fault_clear({"confirm": True})["cleared"] is True
        assert instrument.hub.fault is None
        for _ in range(200):
            if stub.fault_clears:
                break
            time.sleep(0.01)
        assert stub.fault_clears == 1 and stub.abort_stopped is False
        hw.link.send(P.cmd_abort())                     # the next pass's abort
        for _ in range(200):
            if stub.aborts >= 2:
                break
            time.sleep(0.01)
        assert stub.td_stops == 2
    finally:
        hw.close()
        stub.stop()


def test_reading_igors_fault_leaves_it_set_until_the_operator_clears_it():
    """ABORTFAULT is read only: a park latches Igor's fault but Igor keeps it, so a reply lost
    to a timeout cannot lose the fault."""
    stub = _stub(zsensor=0.0, defl=0.0, height=0.0)
    stub.abort_fault = True
    hw = _hw_over(stub)
    hub = _FakeHub(hardware_arc=hw)
    try:
        asyncio.run(safe_park(hub, "unit: igor fault"))
        assert hub.fault and stub.abort_fault is True
        assert stub.fault_clears == 0
    finally:
        hw.close()
        stub.stop()


# ---- re-review fixes (2026-10-27): CLEARFAULT reported, sent only when latched; a failed
# ---- connect-time fault read latches a fault

def _clear_route(monkeypatch, arc, fault):
    from afm_litho.service import hub as hub_mod
    from afm_litho.service.routers import instrument
    monkeypatch.setattr(hub_mod, "real_bridge", lambda h: arc)
    monkeypatch.setattr(instrument.hub, "fault", fault)
    logs: list[str] = []
    monkeypatch.setattr(instrument.hub, "_log_sync", logs.append)
    return instrument.post_fault_clear({"confirm": True}), logs


def _wait_until(pred, timeout=2.0):
    for _ in range(int(timeout / 0.01)):
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def test_a_clear_with_nothing_latched_leaves_igors_backup_running(monkeypatch):
    """A clear clicked while a park runs (nothing latched yet) sends no CLEARFAULT, so Igor's
    5 s backup stays armed under the park."""
    from afm_litho.hardware import bridge_protocol as P
    stub = _stub(zsensor=0.0, defl=0.0, height=50.0)
    stub.safe_park_zeros = False
    hw = _hw_over(stub)
    try:
        hw.link.send(P.cmd_abort())
        assert _wait_until(lambda: stub.aborts == 1)
        reply, _ = _clear_route(monkeypatch, hw, None)
        assert reply == {"ok": True, "cleared": True}
        time.sleep(0.1)
        assert stub.fault_clears == 0 and stub.abort_stopped and stub._backup is not None
    finally:
        hw.close()
        stub.stop()


def test_a_failed_clearfault_still_clears_python_and_says_so(monkeypatch):
    stub = _stub(zsensor=0.0, defl=0.0, height=0.0)
    hw = _hw_over(stub)

    def broken():
        raise ConnectionError("link down")

    monkeypatch.setattr(hw, "clear_fault", broken)
    try:
        reply, logs = _clear_route(monkeypatch, hw, {"reason": "park_unverified"})
        from afm_litho.service.routers import instrument
        assert reply["cleared"] is True and instrument.hub.fault is None
        assert reply["igor_cleared"] is False and "link down" in reply["igor_detail"]
        assert any(m.startswith("WARNING fault clear") for m in logs)
    finally:
        hw.close()
        stub.stop()


def test_a_latched_clear_sends_clearfault_and_reports_it(monkeypatch):
    stub = _stub(zsensor=0.0, defl=0.0, height=0.0)
    stub.abort_stopped = stub.abort_fault = True
    hw = _hw_over(stub)
    try:
        reply, _ = _clear_route(monkeypatch, hw, {"reason": "park_unverified"})
        assert reply["igor_cleared"] is True
        assert _wait_until(lambda: stub.fault_clears == 1)
        assert not stub.abort_stopped and not stub.abort_fault
    finally:
        hw.close()
        stub.stop()


def test_a_clear_with_no_igor_bridge_is_logged(monkeypatch):
    reply, logs = _clear_route(monkeypatch, None, {"reason": "park_unverified"})
    assert reply["igor_cleared"] is None and "no Igor bridge" in reply["igor_detail"]
    assert any("no Igor bridge" in m for m in logs)


class _SilentAbortFaultStub(FakeIgorBridge):
    """An Igor that never answers ABORTFAULT (an older .ipf, or dry-run)."""

    def _dispatch(self, line, link):
        if line.startswith("ABORTFAULT"):
            self._record(line, "ABORTFAULT")
            return
        super()._dispatch(line, link)


def test_a_connect_whose_fault_read_fails_latches_a_fault(monkeypatch):
    from afm_litho.hardware import hardware_arc
    from afm_litho.service.hub import Hub
    monkeypatch.setattr(hardware_arc, "_PARK_READ_TIMEOUT_S", 0.2)
    stub = _SilentAbortFaultStub({"ZSensor": 0.0, "Deflection": 0.0})
    hw = _hw_over(stub)
    hub = _FakeHub(hardware_arc=hw)
    try:
        Hub._latch_igor_abort_fault(hub, hw)
        assert hub.fault and "could not read Igor's abort fault at connect" in hub.fault["detail"]
    finally:
        hw.close()
        stub.stop()


@pytest.mark.parametrize("igor_fault, latched", [(True, True), (False, False)])
def test_a_connect_latches_igors_fault_only_when_set(igor_fault, latched):
    from afm_litho.service.hub import Hub
    stub = _stub(zsensor=0.0, defl=0.0, height=0.0)
    stub.abort_fault = igor_fault
    hw = _hw_over(stub)
    hub = _FakeHub(hardware_arc=hw)
    try:
        Hub._latch_igor_abort_fault(hub, hw)
        assert bool(hub.fault) is latched
    finally:
        hw.close()
        stub.stop()
