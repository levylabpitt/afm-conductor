"""Engage-time safety wiring shared by every hardware engage (approach, scan,
write) on both doors (docs/INSTRUMENT_MODEL_SPEC.md section 12 + 12.1).

Three responsibilities, reused so a FLEX script and the cockpit meet identical
guards:
  * `establish_free_air` -- take a valid free-air baseline via the pure 3-read
    agreement + one dwell-retry (`hardware.free_air.read_free_air`) and latch it on
    `hub.free_air`. Reused by approach, withdraw and the mode switch, in every
    imaging mode; the record is tagged with the mode it was taken for.
  * `check_engage_sum_ceiling` -- read DetectorSum FRESH and apply the 0.9*SUM
    ceiling / SUM floor (`command_guard.check_sum_ceiling`).
  * `arm_interlocks` / `disarm_interlocks` -- the always-on runtime interlocks
    (`WriteInterlocks`), armed for the engage and fed from the telemetry consumer
    (the hardware `_hw_telemetry_frame`, or the twin feed task started here). On a
    trip the callback safe_parks (`host.safe_park.safe_park`).

Kept OUT of the pure `free_air` / `write_interlocks` / `command_guard` modules
because it touches the hub, the twin ARC and the event loop -- exactly the side
effects the caveat says to keep in ONE place per door.
"""

from __future__ import annotations

import asyncio
import threading
import time

from ..command_guard import FreeAir, GuardRefused, check_sum_ceiling
from ...hardware.free_air import FreeAirDisagree, read_free_air
from ..hub import real_bridge
from .safe_park import PARK_WATCHDOG, request_park
from ..safety.write_interlocks import InterlockConfig, WriteInterlocks

# ---------------------------------------------------------------- SUM read ---

def read_detector_sum(hub) -> float | None:
    """A FRESH DetectorSum read: the bridge on hardware, the twin world otherwise.
    None on a failed read (the ceiling then refuses as `sum_missing`)."""
    hw = real_bridge(hub)
    if hw is not None:
        from ...hardware.hw_approach import SUM_CHANNEL
        try:
            return float(hw.td_ReadValue(SUM_CHANNEL))
        except Exception:                                 # noqa: BLE001
            return None
    try:
        return float(hub.host.inst.read_sum())
    except Exception:                                     # noqa: BLE001
        return None


def check_engage_sum_ceiling(hub, setpoint_abs_v: float) -> None:
    """Read DetectorSum FRESH and apply `check_sum_ceiling` (12.1). Raises
    `GuardRefused` (`sum_missing` / `setpoint_exceeds_sum`)."""
    sum_v = read_detector_sum(hub)
    check_sum_ceiling(hub.host.instrument_model, setpoint_abs_v, sum_v)


# ------------------------------------------------------------- free air ------

def free_air_is_fresh(hub, mode: str) -> bool:
    """True when `hub.free_air` is a fresh, withdrawn baseline in `mode` -- so the
    engage need not re-read it (keeps the continuous re-reference's own baseline)."""
    fa = getattr(hub.host, "free_air", None)
    if fa is None or not getattr(fa, "withdrawn", False):
        return False
    if getattr(fa, "mode", None) != mode:
        return False
    try:
        max_age = float(hub.host.instrument_model.free_air_max_age_s.value)
    except Exception:                                     # noqa: BLE001
        max_age = 30.0
    return (time.time() - float(fa.ts)) <= max_age


# Internal key that selects the amplitude reader; never sent to Igor.
_FREE_AMPLITUDE = "amplitude"


def _twin_free_air_reader(hub, quantity: str):
    """A `read_one` for the twin: the world free amplitude (`amplitude`) or the
    world free-air deflection ("deflection"); made to JITTER (first three reads
    disagree) when `faults.free_air_jitter` is engaged."""
    arc = getattr(hub.host, "arc", None)
    jitter = bool(getattr(getattr(arc, "faults", None), "free_air_jitter", False))
    state = {"i": 0}

    def _raw() -> float:
        if quantity == _FREE_AMPLITUDE:
            return float(hub.host.inst.world.lever.free_amplitude_v)
        return float(hub.host.inst.read_free_air_deflection())

    def read_one() -> float:
        state["i"] += 1
        v = _raw()
        if jitter and state["i"] <= 3:
            # Three deterministically-disagreeing reads, then the settled value.
            return v + state["i"] * 1.0
        return v

    return read_one


def _agreed_read(hub, quantity: str, mode: str):
    """One 3-read agreement (+ one dwell-retry) of `quantity` ("deflection" |
    `amplitude`). Raises `GuardRefused("free_air_missing")` if the reads disagree
    twice."""
    hw = real_bridge(hub)
    if hw is not None:
        from ...hardware.hw_approach import hw_read_free_air, hw_read_free_amplitude

        reader = hw_read_free_amplitude if quantity == _FREE_AMPLITUDE else hw_read_free_air

        def read_one() -> float:
            return float(reader(hw))
    else:
        read_one = _twin_free_air_reader(hub, quantity)
    abs_band = float(hub.host.instrument_model.free_air_abs_band_v.value)
    try:
        return read_free_air(read_one, now=time.time, sleep=time.sleep, mode=mode,
                             abs_band_v=abs_band)
    except FreeAirDisagree as exc:
        raise GuardRefused(
            "free_air_missing",
            f"free-air {quantity} reads disagree by more than the agreement band "
            "after a dwell-and-retry") from exc


def establish_free_air(hub, mode: str, *, force: bool = False):
    """Take a valid free-air baseline in `mode` and latch it on `hub.free_air`,
    tagged with `mode` (the guard refuses a baseline whose tag differs).

    `defl_v` is always the withdrawn deflection (what park verify compares
    against). `value_v` is the quantity `command_guard._check_setpoint_band`
    uses: the withdrawn free AMPLITUDE in AC (fraction band), the deflection in
    every other mode (relative band).

    A no-op when a fresh withdrawn baseline in `mode` is already on record (unless
    `force`), so the continuous re-reference's own baseline is preserved. Runs the
    read SYNCHRONOUSLY (the caller off-threads it): the dwell-and-retry really
    sleeps, so the door blocks the documented >5 s on a jittery baseline. Raises
    `GuardRefused("free_air_missing")` if the reads disagree twice.
    """
    mode = str(mode).strip().lower()
    if not force and free_air_is_fresh(hub, mode):
        return hub.host.free_air
    defl = _agreed_read(hub, "deflection", mode)
    band = _agreed_read(hub, _FREE_AMPLITUDE, mode) if mode == "ac" else defl
    hub.host.free_air = FreeAir(value_v=band.value_v, ts=band.ts, mode=mode,
                           withdrawn=True, defl_v=defl.value_v)
    return hub.host.free_air


async def establish_free_air_async(hub, mode: str, *, force: bool = False):
    """`establish_free_air` off the event loop (its dwell really sleeps)."""
    return await asyncio.to_thread(establish_free_air, hub, mode, force=force)


# ------------------------------------------------------- runtime interlocks --

def _twin_interlock_sample(hub) -> tuple[float, float, float]:
    """Read (z_out, SUM, |defl| proxy) for the twin interlock feed, honouring the
    twin fault hooks: `z_rail_stuck` (z reads 148), `sum_drop` (SUM 0.1),
    `defl_over_sum` (deflection above SUM).

    Reads the deterministic world knobs directly (NOT `arc.read`, which shares the
    ARC lock with a write/scan worker's `step()`); a fed sample must never block on a
    long step block, or the continuous-time clock would stall under load. The fault
    flags reproduce exactly what `arc.read`/the running interlock would see."""
    faults = hub.host.arc.faults
    z = 148.0 if faults.z_rail_stuck else float(hub.host.inst.read_z_drive())
    sum_v = 0.1 if faults.sum_drop else float(hub.host.inst.read_sum())
    if faults.defl_over_sum:
        defl = sum_v + 1.0
    else:
        defl = float(hub.host.inst.read_free_air_deflection())
    return z, sum_v, defl


def arm_interlocks(hub) -> None:
    """Arm the runtime interlocks for the current engage. Fed by the always-on
    telemetry consumer on hardware; on the twin, this also starts the feed task."""
    model = hub.host.instrument_model
    baseline = read_detector_sum(hub)
    if baseline is None:
        baseline = 0.0

    def on_trip(reason: str) -> None:
        # Stash the reason; the feed (twin) or the failsafe trigger (hardware)
        # performs the actual safe_park off this callback.
        hub.host._pending_interlock_reason = reason
        if real_bridge(hub) is not None:
            # Hardware: trip from the router thread -> the out-of-band failsafe
            # watchdog performs the verified retract (never block the reader).
            try:
                request_park(hub, reason, PARK_WATCHDOG)
            except Exception:                             # noqa: BLE001,S110
                pass

    wi = WriteInterlocks(model, InterlockConfig(), on_trip=on_trip)
    wi.arm(float(baseline))
    hub.host.write_interlocks = wi
    hub.host._pending_interlock_reason = None
    if real_bridge(hub) is None:
        # The twin feed runs in a DAEMON THREAD, not an asyncio task: a busy event
        # loop (a whole test suite, a fast continuous scan) must never starve the
        # continuous-time clock so a trip is missed before the run ends. Mirrors the
        # hardware side, where the always-on TELEM consumer feeds from the router
        # thread. One at a time.
        thread = getattr(hub.host, "_interlock_feed_thread", None)
        if thread is None or not thread.is_alive():
            thread = threading.Thread(target=_twin_feed, args=(hub,),
                                      name="twin-interlock-feed", daemon=True)
            hub.host._interlock_feed_thread = thread
            thread.start()


def disarm_interlocks(hub) -> None:
    """Disarm the runtime interlocks (a normal withdraw / engage end)."""
    wi = getattr(hub.host, "write_interlocks", None)
    if wi is not None:
        wi.disarm()


def _twin_feed(hub) -> None:
    """The twin's always-on interlock feed (daemon thread): sample the twin
    (fault-aware) every ~0.1 s and feed the armed `WriteInterlocks`. On a trip it
    safe_parks (sync, from this thread -- the same way the failsafe watchdog does)
    with the interlock's reason and exits."""
    from .safe_park import safe_park as _safe_park
    while True:
        wi = getattr(hub.host, "write_interlocks", None)
        if wi is None or not wi.armed:
            return
        try:
            z, sum_v, defl = _twin_interlock_sample(hub)
            wi.feed(t=time.monotonic(), z_out_v=z, sum_v=sum_v, defl_v=defl)
        except Exception:                                 # noqa: BLE001,S110
            pass
        reason = getattr(hub.host, "_pending_interlock_reason", None)
        if reason is not None:
            hub.host._pending_interlock_reason = None
            hub.host._safe_park_reason = reason
            try:
                _safe_park(hub, reason)
            except Exception:                             # noqa: BLE001,S110
                pass
            # Kernel rule #22 (never a silent stop): record the interlock park as a
            # "safety" session record. The sync `safe_park` primitive does not write
            # one (the async orchestrator's `_finish` did); the interlock trip is our
            # trigger, so we record it here, with the SAME label the scan-path rail
            # guard uses (`safe_park: <reason>`).
            try:
                hub.store.record("safety", {
                    "label": f"safe_park: {reason}",
                    "event": "safe_park", "reason": reason,
                    "trigger": "write_interlock"})
            except Exception:                             # noqa: BLE001,S110
                pass
            return
        time.sleep(0.05)     # a background periodic loop: real time on every clock mode


def feed_hw_interlocks(hub, *, z_out_v, sum_v, defl_v) -> None:
    """Feed one hardware telemetry point into the armed interlocks (called from the
    always-on TELEM consumer, `host/telemetry._hw_telemetry_frame`). A no-op
    unless armed."""
    wi = getattr(hub.host, "write_interlocks", None)
    if wi is None or not wi.armed:
        return
    wi.feed(t=time.monotonic(), z_out_v=z_out_v, sum_v=sum_v, defl_v=defl_v)
