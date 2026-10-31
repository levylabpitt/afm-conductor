"""Continuous imaging: a backend-owned auto-repeat scan loop (Scan tab).

The operator leaves the instrument imaging ONE frame overnight and watches drift
correction accumulate. The loop lives in the SERVICE, not the page, so a browser
reload or a dropped websocket never stops it. It only RE-TRIGGERS scans -- every
frame goes through the SAME scan/start path (routers/scan.py), so Connect Phase 2's
drift COMPENSATE applies per frame. Continuous imaging and drift correction
COMPOSE: this re-images, the Connect toggle corrects; neither reimplements the
other, and with drift off this simply re-images the same window.

MECHANISM. One asyncio supervisor task: it arms a scan with the stored config,
AWAITS the pump to completion (hub._scan_task -- so the frame's drift registration
+ COMPENSATE have landed), then arms the next, optionally after interval_s. Serial
by construction -- no re-entrant calls inside a pump's finally.

ABORT WINS. scan/abort disables the loop and sets hub._scan_repeat_stop, so the
supervisor breaks the instant the in-flight pump releases; the stop event is also
what the interval sleep waits on, so a stop lands mid-wait too. Every step is
best-effort: a hiccup in the loop can never wedge the service or strand the tip.

The auto-repeat supervisor, the rail-dwell guard and the free-air and SUM guards. The
enable/disable adapters are in `routers/continuous.py`. The telemetry tick that feeds the guards is `host/telemetry.py`.
"""

from __future__ import annotations

import asyncio
import math
import time


from ... import clock
from ..hub import hub
from ..hub import real_bridge as _real_bridge
from .safe_park import safe_park_async as safe_park
from .safe_park import (
    PARK_DEFERRED,
    STOP_REPEAT_LOOP,
    STOP_SCAN_PASS,
    _signal_stop,
    request_park,
)
from ..routers.scan import approach, scan_start, withdraw


# NO retry tolerance on a hardware fault (operator decision, 2026-08-28 audit fix 3).
# A spontaneous pass fault means the Z state is UNKNOWN with the tip engaged, so the
# loop must NOT arm another frame to "retry" -- ANY fault is terminal: stop + safe_park
# so the operator/AI can diagnose. (The old MAX_CONSECUTIVE_ERRORS=3 retry armed up to
# two more frames on an unknown Z state; that tolerance is removed -- see _supervise.)

# Reactive fail-safe monitors (host-side, docs/SAFETY_KERNEL_PLAN.md Sec 0).
# FREE_AIR_JUMP_FRAC: the "within 10%" band of the 3-read free-air guard
#   (|a - b| <= FREE_AIR_JUMP_FRAC * |b|); a single anomalous free-air read is
#   re-checked, and only three inconsistent reads -> safe_park.
# SUM_LOSS_V: the SUM-loss floor -- two CONSECUTIVE reads below it (laser/lever
#   gone, feedback blind) -> safe_park immediately.
# SUM_CLAMP_FRAC: |deflection| <= SUM physically, so any deflection-domain value
#   (the accepted free-air read OR the effective setpoint) past this fraction of a
#   fresh SUM read is unphysical/unsafe -> safe_park.
FREE_AIR_JUMP_FRAC = 0.10
SUM_LOSS_V = 0.1
SUM_CLAMP_FRAC = 0.9


def _flip_direction(direction: str) -> str:
    """The opposite raster direction. Anything that isn't "down" flips to "down",
    so a missing/"up" config alternates up<->down cleanly."""
    return "up" if str(direction).lower() == "down" else "down"


def _state() -> dict:
    """The JSON-safe continuous state for the UI to hydrate/poll (the relative-
    setpoint config + last readout ride along, so the one GET hydrates both)."""
    sr = hub.host.scan_repeat
    started = sr.get("started_at")
    rel = hub.host.relative_setpoint
    guard = getattr(hub.host, "rail_guard", None)
    # The rail-dwell guard now arms on BOTH backends: on hardware it is fed from the
    # always-on TELEM poll (host/telemetry._hw_telemetry_frame), not a host bridge
    # read, so there is no shared-link race and no `pending` gate any more.
    rail = guard.readout() if guard is not None else None
    return {
        "enabled": bool(sr.get("enabled")),
        "count": int(sr.get("count", 0)),
        "started_at": started,
        "elapsed_s": (time.time() - started) if started else 0.0,
        "max_frames": sr.get("max_frames"),
        "max_seconds": sr.get("max_seconds"),
        "interval_s": sr.get("interval_s", 0.0),
        "errors": int(sr.get("errors", 0)),
        "stopped_reason": sr.get("stopped_reason"),
        "relative": {
            "enabled": bool(rel.get("enabled")),
            "settle_s": rel.get("settle_s", 4.0),
            "last_free_air_v": rel.get("last_free_air_v"),
            "last_effective_v": rel.get("last_effective_v"),
        },
        # Rail-dwell safety guard live state (armed indicator + counter), so the
        # one GET hydrates the Scan tab's guard readout too (service/safety.py).
        "rail": rail,
    }


async def _wait_stop(stop, timeout: float, tick: float = 0.05) -> bool:
    """Sleep up to `timeout` seconds, returning EARLY (True) once `stop` is set.

    `hub._scan_repeat_stop` is a `threading.Event`, not an `asyncio.Event`, because
    it is set from NON-loop threads -- the failsafe watchdog, the bridge-router
    reader, a pass worker -- and `asyncio.Event.set` from a foreign thread is a
    silent no-op for any waiter (docs/FLEX_INTEGRATION_PLAN.md Sec B.3). A
    threading.Event has no awaitable `wait`, so this polls it on a short tick
    instead. Polling (rather than `to_thread(stop.wait, timeout)`) also keeps the
    default executor free: these waits are seconds-to-minutes long and would
    otherwise pin a pool thread for the whole interval.

    Returns True when the stop landed, False when the full timeout elapsed.
    """
    deadline = clock.deadline(max(0.0, float(timeout)))
    while True:
        if stop.is_set():
            return True
        remaining = deadline.remaining()
        if remaining <= 0:
            return False
        await clock.async_sleep(min(tick, remaining))


async def _broadcast(stopped_reason: str | None = None):
    """Push the live readout (start / per-frame increment / stop-with-reason)."""
    st = _state()
    msg = {"type": "continuous", "enabled": st["enabled"], "count": st["count"],
           "elapsed_s": st["elapsed_s"]}
    if stopped_reason:
        msg["stopped_reason"] = stopped_reason
    await hub.broadcast(msg)


def _relative_active() -> bool:
    """Relative re-referencing applies only when ENABLED AND only for a
    contact-mode frame -- the vendor records a free-air deflection baseline only
    for the contact family (ScanMaster.ipf:4665); AC's amplitude setpoint is a
    different mechanism. Outside contact, relative mode is inert."""
    rel = hub.host.relative_setpoint
    if not rel.get("enabled"):
        return False
    cfg = hub.host.scan_repeat.get("config") or {}
    return str(cfg.get("mode", "contact")).lower() == "contact"


async def _read_free_air_deflection() -> float:
    """Read the off-contact free-air deflection (V). Reuses the hardware free-air
    read (td_ReadValue(`DeflectionValue`)) on the real backend, the simulated
    drifting baseline on the twin. Must be called after the withdraw + settle."""
    import math

    hw = _real_bridge(hub)
    if hw is not None:
        from ...hardware.hw_approach import hw_read_free_air
        v = float(await asyncio.to_thread(hw_read_free_air, hw))
    else:
        raw = await asyncio.to_thread(hub.host.inst.read_free_air_deflection)
        v = float(raw) if raw is not None else math.nan
    if not (math.isnan(v) or math.isinf(v)):
        # Every WITHDRAWN free-air read taken as a baseline populates
        # hub.free_air (docs/INSTRUMENT_MODEL_SPEC.md section 10) -- this call is
        # made ONLY after a full withdraw + settle (the re-reference's own
        # contract), and relative-setpoint re-referencing is CONTACT-ONLY.
        from ..command_guard import FreeAir
        hub.host.free_air = FreeAir(value_v=v, ts=time.time(), mode="contact",
                               withdrawn=True)
    return v


async def _read_sum() -> float:
    """Read the photodiode SUM voltage (V). Hardware: td_ReadValue(SUM_CHANNEL)
    (`DetectorSum` -- CONFIRMED on the Cypher and the MFP-3D, 2026-09-15;
    lowercase "sum" reads 0 V on the Cypher and must not be used). Twin:
    hub.inst.read_sum (world.sum_voltage). A router failure (timeout /
    BridgeClosed) is caught and returned as NaN, which the SUM-loss interlock
    treats as a fail-safe. Module-level so tests can monkeypatch a controlled
    sequence, exactly like _read_free_air_deflection."""
    hw = _real_bridge(hub)
    if hw is not None:
        from ...hardware.hw_approach import SUM_CHANNEL
        try:
            return float(await asyncio.to_thread(hw.td_ReadValue, SUM_CHANNEL))
        except Exception:                                # noqa: BLE001 (router failure)
            return math.nan
    v = await asyncio.to_thread(hub.host.inst.read_sum)
    return float(v) if v is not None else math.nan


def _bad(v) -> bool:
    """A non-finite (None/NaN/Inf) read -- a bad read is itself a fail-safe trigger."""
    return v is None or math.isnan(v) or math.isinf(v)


def _within(a: float, b: float) -> bool:
    """The 3-read guard's "within 10%": |a - b| <= FREE_AIR_JUMP_FRAC * |b|."""
    return abs(a - b) <= FREE_AIR_JUMP_FRAC * abs(b)


async def _read_free_air_guarded(last_accepted: float | None):
    """The free-air 3-read 10% guard. Returns (value, None) on accept, or
    (None, reason) when the reads are unstable and the caller must safe_park.

    Normal (a `last_accepted` exists): read r1; if within 10% of last -> accept it
    (the fast path -- the baseline barely moved). Else the read is suspicious, so
    re-read r2; if r2 agrees with r1 (within 10% of each other) -> accept r2 (a
    consistent reading, i.e. real drift). Else r3; if r3 agrees with r2 -> use r3.
    Else three inconsistent reads -> safe_park. First frame (no last accepted): two
    reads must agree, escalate to a third, else safe_park. A bad read (None/NaN/Inf)
    at ANY step is itself a safe_park."""
    r1 = await _read_free_air_deflection()
    if _bad(r1):
        return None, "free-air read failed (NaN)"
    if last_accepted is not None and _within(r1, last_accepted):
        return r1, None                                  # fast path: within 10% of last
    # r1 is far from the last accepted (or this is the first frame): re-check.
    r2 = await _read_free_air_deflection()
    if _bad(r2):
        return None, "free-air read failed (NaN)"
    if _within(r2, r1):
        return r2, None                                  # two agree -> accept
    r3 = await _read_free_air_deflection()
    if _bad(r3):
        return None, "free-air read failed (NaN)"
    if _within(r3, r2):
        return r3, None                                  # third agrees -> use it
    return None, (f"free-air deflection unstable: 3 reads disagree by >10% "
                  f"({r1:+.3f}, {r2:+.3f}, {r3:+.3f} V)")


async def _rereference() -> bool:
    """Between-frame re-reference (relative mode): full withdraw -> settle ->
    read free-air deflection -> effective setpoint = free_air + the scan setpoint
    (the base force above free-air, captured at loop start as base_setpoint so it
    never compounds) -> re-approach on the CURRENT gains with that setpoint ->
    store the effective on the config so the next armed frame uses it. Returns
    True on success.

    FAIL-SAFE: a failed read (NaN) or a raised re-approach calls safe_park (stop +
    verified retract) and returns False -- the supervisor then STOPS rather than
    scan the next frame at a stale/wrong setpoint. A full withdraw (not a partial
    lift) is mandatory: a partial lift does not free the probe from the tip-sample
    interaction, so the free-air read would be contaminated by contact.

    REACTIVE SAFETY MONITORS (host-side fail-safe, run each re-reference): a fresh
    SUM read gates a SUM-loss interlock (two consecutive reads < 0.1 V -> safe_park)
    BEFORE any 0.9*SUM math; a 3-read free-air guard rejects an unstable baseline;
    and a 0.9*SUM clamp rejects any deflection-domain value (the accepted free-air
    read OR the effective setpoint) that exceeds |deflection| <= SUM physically.
    Every trip -> safe_park (docs/SAFETY_KERNEL_PLAN.md Sec 0).
    """
    rel = hub.host.relative_setpoint
    sr = hub.host.scan_repeat
    stop = hub.host._scan_repeat_stop

    async def _fail(why: str) -> bool:
        """Park and report the failure under the SPECIFIC reason.

        The stamp must land BEFORE safe_park, not after. `safe_park` clears
        `enabled` and stamps the generic `stopped_reason = "safe_park"` when the
        slot is still None -- and clearing `enabled` is what an observer (the Scan
        panel's poll, a test, a FLEX script) treats as "the loop has stopped", so it
        reads the reason in that same breath. The supervisor's own `finally` does
        write "reref_failed", but only once it has unwound, which is later. Stamping
        first means the specific reason is the only one anybody ever sees.
        """
        if sr.get("stopped_reason") is None:
            sr["stopped_reason"] = "reref_failed"
        await request_park(hub, why, full_park=safe_park)
        return False

    try:
        await withdraw({})                                  # full retract, off-contact
    except Exception as exc:                                # noqa: BLE001
        return await _fail(f"relative-setpoint withdraw failed ({exc})")
    # Settle so the free-air baseline stabilizes; interruptible so an abort/stop
    # lands mid-dwell (the tip is already withdrawn, so a stop here is safe).
    settle = float(rel.get("settle_s", 4.0) or 0.0)
    if settle > 0:
        await _wait_stop(stop, settle)
    if stop.is_set() or not sr.get("enabled"):
        return False                                        # aborted during settle

    # --- SUM-loss interlock FIRST (before any 0.9*SUM math). A NaN/failed SUM read,
    #     or two CONSECUTIVE reads < 0.1 V (laser/lever gone -> feedback blind), is a
    #     fail-safe: the feedback would be driving on garbage.
    sum_v = await _read_sum()
    if _bad(sum_v):
        return await _fail("SUM read failed (NaN) -- detector blind")
    if sum_v < SUM_LOSS_V:
        streak = int(sr.get("_sum_low_streak", 0)) + 1
        sr["_sum_low_streak"] = streak
        if streak >= 2:
            return await _fail(f"SUM lost: two consecutive reads < {SUM_LOSS_V:g} V "
                               f"(laser/lever gone, feedback blind)")
    else:
        sr["_sum_low_streak"] = 0

    # --- free-air 3-read 10% guard -> the accepted free-air deflection (or safe_park).
    free_air, why = await _read_free_air_guarded(rel.get("last_free_air_v"))
    if free_air is None:
        return await _fail(f"relative setpoint: {why}")

    # base_setpoint = the scan setpoint the loop started with (the target force
    # above free-air). Reading it from base_setpoint (NOT config["setpoint"], which
    # we overwrite with the effective below) keeps successive re-references from
    # compounding the offset onto itself.
    base = float(sr.get("base_setpoint", (sr.get("config") or {}).get("setpoint", 0.2)))
    effective = free_air + base

    # --- 0.9*SUM clamp: |deflection| <= SUM physically, so a free-air read OR the
    #     effective setpoint past 0.9*SUM is unphysical/unsafe.
    clamp = SUM_CLAMP_FRAC * sum_v
    for label, val in (("free-air read", free_air), ("effective setpoint", effective)):
        if abs(val) > clamp:
            return await _fail(f"relative setpoint: {label} {val:+.3f} V exceeds "
                               f"0.9*SUM ({clamp:.3f} V; SUM={sum_v:.3f} V)")

    # Record the just-read WITHDRAWN free-air baseline so the command guard on the
    # re-approach below (`check_approach`, wired in B1) sees a FRESH baseline
    # (docs/INSTRUMENT_MODEL_SPEC.md secs 2/10): the re-reference IS a withdraw +
    # free-air read, so this is exactly the baseline the guard requires, and
    # `effective = free_air + base` then has a relative offset of `base` (the scan
    # setpoint), which sits in the contact band. Without this, the guard would refuse
    # the loop's own between-frame re-approach with `free_air_missing`.
    from ..command_guard import FreeAir
    hub.host.free_air = FreeAir(value_v=float(free_air), ts=time.time(),
                           mode="contact", withdrawn=True)

    # Re-approach on the CURRENT gains (from the last approach / z_loop), with the
    # new effective setpoint -- the existing approach path, not a reinvented one.
    z = getattr(hub.host, "z_loop", None) or {}
    body: dict = {"setpoint": effective}
    if "pgain" in z:
        body["pgain"] = z["pgain"]
    if "igain" in z:
        body["igain"] = z["igain"]
    try:
        res = await approach(body)
    except Exception as exc:                                # noqa: BLE001
        return await _fail(f"relative-setpoint re-approach failed ({exc})")
    # A REFUSAL IS NOT AN EXCEPTION. `approach` answers with an error DICT for the
    # silent-twin gate and, since the single-writer claim landed, for `busy` -- and a
    # bare `await` swallowed both: the loop would record the new effective setpoint
    # and arm the next frame believing the tip had been re-approached, when it is
    # withdrawn (the re-reference withdraws first). Scanning from there is a crash
    # into an unknown Z. Fail safe instead, which stops the loop with the tip already
    # off the surface.
    if isinstance(res, dict) and res.get("error"):
        return await _fail(f"relative-setpoint re-approach refused "
                           f"({res.get('reason') or res['error']})")
    sr["config"]["setpoint"] = effective        # the next frame arms with this
    rel["last_free_air_v"] = round(free_air, 4)
    rel["last_effective_v"] = round(effective, 4)
    await hub.log(f"Relative setpoint: free-air {free_air:+.3f} V + scan setpoint "
                  f"{base:+.3f} V -> setpoint {effective:+.3f} V (re-referenced)")
    await hub.broadcast({"type": "relative_setpoint",
                         "free_air_v": rel["last_free_air_v"],
                         "effective_v": rel["last_effective_v"]})
    return True


# --------------------------------------------------------- rail-dwell guard --
# The Z-drive rail-dwell telemetry watchdog (safety kernel entry #1). The guard
# DECISION lives in service/safety.py (pure, unit-tested); here is the RUNTIME:
# the ~10 Hz HV Z-drive poll, the trip action (stop-with-withdraw), and the
# reporting. Active only while a continuous scan runs (a single scan never trips).
# See docs/SAFETY_KERNEL_RAIL_DWELL.md.


async def _read_zdrive() -> float:
    """Read the HV Z-drive voltage (V, -10..+150 V domain). Hardware backend:
    td_ReadValue on the instrument's Z output channel via hw_read_zdrive
    (`Height` on both; Output.Z on the MFP-3D, see
    hw_approach._Z_DRIVE + HardwareARC.z_out_channel). Twin: the simulated
    Z-drive (world.z_drive_voltage).
    Module-level so tests can monkeypatch a controlled sequence, exactly like
    _read_free_air_deflection."""
    hw = _real_bridge(hub)
    if hw is not None:
        from ...hardware.hw_approach import hw_read_zdrive
        return float(await asyncio.to_thread(hw_read_zdrive, hw))
    v = await asyncio.to_thread(hub.host.inst.read_z_drive)
    return float(v) if v is not None else math.nan


async def _broadcast_rail(guard) -> None:
    """Push the guard's live readout so the Scan tab can show the armed indicator
    and watch the consecutive-railed-line counter climb toward the trip."""
    await hub.broadcast({"type": "rail_guard", **guard.readout()})




async def _rail_poll(guard) -> None:
    """~POLL_HZ poll of the HV Z-drive during ONE frame. Each tick reads the
    Z-drive voltage and the live raster row (hub._scan_row, set by the scan pump);
    when the row advances it closes the previous line (Guard B). A Guard B trip is
    the ONLY mid-frame abort -- it calls _rail_trip (stop + abort + report), then
    the poll returns (the single withdraw runs post-pump in _run_one_frame).

    Best-effort and abort-aware: a read hiccup is dropped, a set stop event ends
    the poll within a tick, and CancelledError (frame end) unwinds cleanly -- the
    guard can never wedge the loop, and an abort always wins. The stop event is
    polled with is_set() (never awaited) on purpose: it is a process-global Event
    shared across the service's supervisor, so awaiting it here would bind it to
    this poll's loop and break the next run on a fresh loop (the TestClient case)."""
    period = 1.0 / max(1.0, float(guard.limits.poll_hz))
    stop = hub.host._scan_repeat_stop
    last_row: int | None = None
    while not stop.is_set():
        row = getattr(hub.host, "_scan_row", None)
        if row is not None and row != last_row:
            if last_row is not None:                     # a completed line -> Guard B
                trip = guard.close_line(last_row)
                if trip is not None:
                    await _rail_trip(guard, trip)
                    return
            last_row = row
        try:
            v = await _read_zdrive()
        except Exception:                                # noqa: BLE001  (read hiccup)
            v = None
        guard.add_sample(v)
        await _broadcast_rail(guard)
        # A background periodic loop: real time on every clock mode.
        await asyncio.sleep(period)                      # abort ends it within a tick


async def _rail_trip(guard, trip: dict) -> None:
    """Act on a guard trip: build the reason, stop the continuous loop AT ONCE,
    abort any in-flight frame, and REPORT (log + session record -- kernel rule #22,
    never a silent stop). Does NOT withdraw -- the single retract runs post-pump in
    _run_one_frame so it can't race the twin/pass mid-frame. Idempotent: the first
    trip of a run wins."""
    if guard.tripped:
        return
    limits = guard.limits
    if trip.get("guard") == "B":
        reason = (f"rail-dwell: Z drive railed for {limits.consec_lines} "
                  f"consecutive lines; frame aborted, tip retracted")
        frac = None
    else:
        frac = float(trip.get("frac", guard.frame_frac))
        reason = (f"rail-dwell: Z drive at a rail for {frac:.0%} of the frame "
                  f"(> {limits.frame_frac:.0%}); continuous stopped, tip retracted")
    guard.tripped = {"guard": trip.get("guard"), "reason": reason,
                     "consec": int(guard.consec),
                     "frame_frac": round(guard.frame_frac, 4),
                     "frac": (round(frac, 4) if frac is not None else None),
                     "withdrawn": False}
    # stop the loop AT ONCE -- the supervisor arms no further frame.
    hub.host.scan_repeat["enabled"] = False
    hub.host.scan_repeat["stopped_reason"] = "rail_dwell"
    _signal_stop(hub, STOP_REPEAT_LOOP)
    # abort the in-flight frame so the pump stops now (Guard B mid-frame). For a
    # Guard A trip the frame is already done, so these are harmless no-ops.
    if _real_bridge(hub) is not None:
        _signal_stop(hub, STOP_SCAN_PASS)
    try:
        hub.host.inst.abort_scan()
    # a failed abort_scan must not stop the rest of the safety stop
    except Exception:  # noqa: BLE001, S110
        pass
    await hub.log(f"SAFETY {reason}")
    try:                                # report to the session recorder (rule #22)
        hub.store.record("safety", {
            "label": reason, "hazard": 23, "guard": trip.get("guard"),
            "reason": reason, "consecutive_lines": int(guard.consec),
            "frame_frac": round(guard.frame_frac, 4), "limits": limits.as_dict()})
    except Exception as exc:                             # noqa: BLE001
        await hub.log(f"Rail-dwell: session record failed ({exc})")
    await _broadcast_rail(guard)


async def _rail_withdraw(guard) -> None:
    """Retract the tip after a trip -- the SINGLE withdraw site, run once the frame
    pump has released (so it never races the twin/pass). A safety trip ALWAYS
    retracts, regardless of withdraw_on_stop. Reuses the existing hw/twin withdraw
    path; on hardware WITH a pass still holding the bridge link it defers to the
    scan pump's finally (the Stop+Withdraw pattern), else withdraws inline."""
    if not guard.tripped or guard.tripped.get("withdrawn"):
        return
    hw = _real_bridge(hub)
    scan_active = hub.host._scan_task is not None and not hub.host._scan_task.done()
    try:
        if hw is not None and scan_active:
            # pass worker owns the link -> deferred, stamped with the scan epoch
            request_park(hub, "rail-dwell trip", PARK_DEFERRED)
            await hub.log("Rail-dwell trip: tip will withdraw when the pass stops")
        else:
            # Unified fail-safe: route the retract through safe_park (rail-dwell is
            # twin-only today -- gated off on hardware -- so this is the twin path).
            await request_park(hub, guard.tripped.get("reason", "rail-dwell trip"),
                               full_park=safe_park)
    except Exception as exc:                             # noqa: BLE001
        await hub.log(f"Rail-dwell: withdraw failed ({exc})")
    guard.tripped["withdrawn"] = True
    await _broadcast_rail(guard)


async def _run_one_frame() -> bool:
    """Arm one scan through the SAME scan/start path and await the pump to
    completion. Returns True on a clean frame, False on a scan error (a raised
    handler, or a `{"error": ...}` result). Best-effort: never raises.

    ALTERNATING RASTER: a frame-up ends at the top, so the next frame runs down
    from there (no fly-back). This frame uses the stored config's current
    direction -- the FIRST frame is whatever Frame button armed the loop -- then
    the stored direction is flipped so the next call arms the opposite way. The
    cached image is NOT mirrored between directions (the capture indexes rows by
    physical slow position, not scan order), so drift tracking stays intact.
    """
    sr = hub.host.scan_repeat
    cfg = dict(sr["config"])
    sr["config"]["direction"] = _flip_direction(cfg.get("direction", "up"))
    # Race-guard (docs/STOP_RETRACT_SPEC.md Sec 2): a fresh continuous frame is a
    # new pass that (re)claims the tip, so any deferred withdraw stamped for an
    # earlier frame/scan is now stale. scan_start bumps the epoch too; this is the
    # explicit frame-start bump the spec calls for (robust to future refactors).
    hub.host._scan_epoch += 1
    # Rail-dwell guard (safety kernel entry #1): while armed, run a ~10 Hz HV
    # Z-drive poll ALONGSIDE this frame (Guard B, the only mid-frame abort), then
    # evaluate Guard A at frame end. Inert when the guard is disarmed.
    guard = getattr(hub.host, "rail_guard", None)
    # The guard arms on BOTH backends now. TWIN: an async ~10 Hz poll reads the
    # simulated Z-drive (_rail_poll). HARDWARE: the always-on TELEM consumer
    # (_hw_telemetry_frame) feeds the guard from the polled Output.Z -- no host poll
    # loop, no shared-link read. _rail_feeding bounds the consumer's feed to this frame.
    armed = (guard is not None and guard.armed)
    poll = None
    if armed:
        guard.start_frame()
        hub.host._scan_row = None            # no stale row from the previous frame
        hub.host._rail_last_row = None
        hub.host._rail_pending_trip = None
        if _real_bridge(hub) is None:
            poll = asyncio.create_task(_rail_poll(guard))   # twin: async z-drive poll
        else:
            hub.host._rail_feeding = True                        # hardware: consumer feeds
    hub.host._scan_pump_error = None       # clear before the frame (blocker #3: see below)
    try:
        res = await scan_start(cfg)
        if not isinstance(res, dict) or res.get("error"):
            await hub.log(f"Continuous: scan did not start "
                          f"({res.get('error') if isinstance(res, dict) else res})")
            return False
        task = hub.host._scan_task
        if task is not None:
            await task            # frame + its drift register/COMPENSATE have run
        # Blocker #3/#2: the hardware scan pump (host/scan_pump._hw_scan_pump) runs in
        # its own task and used to SWALLOW a pass exception (and report a silent
        # chained-scan `err` as aborted=True), so `await task` never raised and the
        # error was never counted -- a wedged rig would loop forever. The pump now
        # records the fault on hub._scan_pump_error; a spontaneous pass exception, a
        # chained `err`, or a router failure surfaced there is a FRAME FAILURE.
        if getattr(hub.host, "_scan_pump_error", None):
            await hub.log(f"Continuous: frame failed "
                          f"({hub.host._scan_pump_error})")
            return False
        return True
    except Exception as exc:                              # noqa: BLE001
        await hub.log(f"Continuous: frame failed ({exc})")
        return False
    finally:
        if armed:
            hub.host._rail_feeding = False       # stop the hardware consumer feeding first
            if poll is not None:                # stop the twin poll (frame is over)
                poll.cancel()
                try:
                    await poll
                except asyncio.CancelledError:
                    pass
            # S3 INTERLOCK trip (SUM-collapse / deflection): the consumer stashed a
            # pending interlock trip mid-frame + set _hw_abort to stop the pass. Route
            # it through the SINGLE post-pump withdraw site -> safe_park with the
            # interlock's stopped_reason. Takes precedence over the rail finish (the
            # pass already aborted); safe_park stops the loop + retracts + reports, so
            # the supervisor then breaks on the not-enabled/stop flag with this reason.
            ilk = getattr(hub.host, "_interlock_pending_trip", None)
            if ilk is not None:
                hub.host._interlock_pending_trip = None
                hub.host.scan_repeat["enabled"] = False
                hub.host.scan_repeat["stopped_reason"] = ilk["stopped_reason"]
                _signal_stop(hub, STOP_REPEAT_LOOP)
                await request_park(hub, ilk["reason"], full_park=safe_park)
            elif not guard.tripped:
                # HARDWARE: a Guard B trip the TELEM consumer detected mid-frame is
                # stashed as _rail_pending_trip (it already set _hw_abort to stop the
                # pass); act on it now (log/record). Otherwise, on a CLEAN frame end,
                # finalize the last line's Guard-B counting then evaluate Guard A over
                # the whole frame. The lock serializes the guard against a late
                # consumer add_sample on the (now stopped) feed.
                pend = getattr(hub.host, "_rail_pending_trip", None)
                if pend is not None:
                    await _rail_trip(guard, pend)
                else:
                    with hub.host._rail_lock:
                        fin_row = getattr(hub.host, "_scan_row", None)
                        tb = guard.close_line(fin_row) if fin_row is not None else None
                        ta = guard.finish_frame() if tb is None else None
                    if tb is not None:
                        await _rail_trip(guard, tb)
                    elif ta is not None:
                        await _rail_trip(guard, ta)
            # SINGLE withdraw site: a trip always retracts, once the pump released.
            if guard.tripped:
                await _rail_withdraw(guard)


async def _supervise():
    """Run scans back-to-back while enabled, within bounds, and un-aborted.

    Stop reasons (broadcast on the way out): "max_frames", "max_seconds",
    "fault" (any hardware pass fault -- terminal, no retry), "rail_dwell",
    "reref_failed", "aborted", "disabled". `aborted`/`disabled` are set by the
    caller that flipped the toggle (scan/abort, or the disable endpoint); the caps
    and faults are decided here. EVERY termination safe_parks in the finally (stop +
    verified retract) -- there is no path that leaves the tip engaged.
    """
    sr = hub.host.scan_repeat
    stop = hub.host._scan_repeat_stop
    reason = None
    first = True
    try:
        while True:
            if stop.is_set() or not sr.get("enabled"):
                reason = sr.get("stopped_reason") or "disabled"
                break
            if sr.get("max_frames") and sr["count"] >= sr["max_frames"]:
                reason = "max_frames"
                break
            if sr.get("max_seconds") and sr.get("started_at") \
                    and (time.time() - sr["started_at"]) >= sr["max_seconds"]:
                reason = "max_seconds"
                break

            # Relative-setpoint re-reference BETWEEN frames: the first frame uses
            # the setpoint the loop started with; from the second frame on (and
            # only when we've cleared the bounds above, i.e. another frame really
            # is coming) we withdraw/settle/read/re-approach so the setpoint
            # tracks the drifting free-air baseline. A failure fails safe: stop
            # the loop, tip already withdrawn (never scan at a stale setpoint).
            if not first and _relative_active():
                if not await _rereference():
                    reason = "reref_failed"
                    break
                if stop.is_set() or not sr.get("enabled"):
                    reason = sr.get("stopped_reason") or "disabled"
                    break
            first = False

            ok = await _run_one_frame()
            # Rail-dwell safety trip (Guard A/B): _run_one_frame already stopped the
            # loop, aborted any in-flight frame, and withdrew the tip. Guard B aborts
            # MID-frame (that frame does not count); Guard A fires at a CLEAN frame
            # end (that frame completed, so it counts). Then break with the reason.
            guard = getattr(hub.host, "rail_guard", None)
            trip = guard.tripped if guard is not None else None
            if trip:
                if trip.get("guard") == "A" and ok:
                    sr["count"] += 1
                reason = "rail_dwell"
                break
            # An abort/disable can land WHILE the frame runs -- honour it before
            # counting the frame, so no extra increment slips past the stop.
            if stop.is_set() or not sr.get("enabled"):
                reason = sr.get("stopped_reason") or "disabled"
                break
            if ok:
                sr["count"] += 1
                sr["errors"] = 0
                await _broadcast()
            else:
                # A hardware pass fault (a spontaneous pass exception, a chained-scan
                # `err`, or a router failure surfaced on hub._scan_pump_error) is
                # TERMINAL -- operator decision (audit fix 3): "if an error occurred,
                # it should be parked until the operator/AI can diagnose." There is NO
                # retry: arming another frame would drive a scan on an UNKNOWN Z state
                # with the tip still engaged. Stop now; the finally safe_parks (stop +
                # verified retract). errors is bumped only for the readout.
                sr["errors"] += 1
                reason = "fault"
                break

            interval = float(sr.get("interval_s") or 0.0)
            if interval > 0:
                await _wait_stop(stop, interval)   # interruptible: a stop breaks it
    except Exception as exc:                              # noqa: BLE001
        reason = reason or "fault"
        await hub.log(f"Continuous: supervisor stopped on error ({exc})")
    finally:
        sr["enabled"] = False
        sr["stopped_reason"] = reason
        try:
            from .engage import disarm_interlocks
            disarm_interlocks(hub)                     # continuous run over: disarm interlocks
        except Exception:                             # noqa: BLE001,S110
            pass
        guard = getattr(hub.host, "rail_guard", None)      # rail-dwell guard: polls only
        if guard is not None:                         # while a continuous scan runs
            guard.disarm()
            await _broadcast_rail(guard)
        await hub.log(f"Continuous imaging stopped ({reason}) after "
                      f"{sr['count']} frame(s)")
        # FAIL-SAFE (host-side, docs/SAFETY_KERNEL_PLAN.md Sec 0): ANY continuous-loop
        # termination -> safe_park (stop + verified retract). There is NO "stop and
        # leave engaged" -- a cap (max_frames/max_seconds), the consecutive-error
        # cap, an operator abort/disable, a reref failure, a rail-dwell trip and a
        # supervisor exception ALL get the tip off the surface. If a reactive monitor
        # (reref / SUM / rail) already safe-parked THIS episode, don't re-park (it is
        # idempotent, but avoid the redundant retract + record) -- just note it.
        if not hub.host._safe_park.get("active"):
            await request_park(hub, f"continuous stopped: {reason}",
                               full_park=safe_park)
        else:
            await hub.log(f"Continuous: already safe-parked this run ({reason})")
        await _broadcast(stopped_reason=reason)


async def _stop_supervisor(reason: str):
    """Signal the running supervisor to stop and wait for it to unwind."""
    sr = hub.host.scan_repeat
    sr["enabled"] = False
    sr["stopped_reason"] = reason
    _signal_stop(hub, STOP_REPEAT_LOOP)
    task = hub.host._scan_repeat_task
    if task is not None and not task.done():
        try:
            await task
        # a repeat task's own failure must not abort the stop
        except Exception:  # noqa: BLE001, S110
            pass
    hub.host._scan_repeat_task = None


