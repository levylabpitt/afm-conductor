"""The always-on telemetry tick: the host end of Igor's TELEM poll.

`_hw_telemetry_frame`
runs inline on the bridge reader thread, so it stays non-blocking and never touches
the socket. It feeds the SUM-collapse and deflection interlocks and the rail-dwell
guard; the supervisor (`host/continuous.py`) acts on the trips post-pump.
"""

from __future__ import annotations

import asyncio
import math
import time

from ..hub import hub
from .safe_park import STOP_SCAN_PASS, _signal_stop


# ------------------------------------------------- always-on TELEM consumer --
# The host end of the always-on telemetry+safety poll (group-2 redo). Igor pushes one
# `TELEM x y zdrive defl sum` per tick (afm_litho_run.ipf AFM_TipTick); the bridge
# router routes it to this consumer on its SINGLE reader thread. It is installed ONCE
# per hardware backend (install_hw_telemetry) and stays for the life of the link -- it
# is NOT armed per pass, and it reads NO in-wave capture bank (the whole point of the
# redo). CONTRACT: runs INLINE on the reader thread, so it MUST be non-blocking and
# MUST NOT touch the socket/router. It does only: a dict write (hub._hw_telemetry), a
# thread-safe Event set (hub._hw_abort on a trip), a lock-guarded list append
# (rail_guard feed), and it hands any WS broadcast to the event loop via
# run_coroutine_threadsafe (fire-and-forget, never awaited here).


def _finite(v):
    """A telemetry channel coerced to a finite float, or None (NaN/Inf/None)."""
    if v is None:
        return None
    v = float(v)
    return None if (math.isnan(v) or math.isinf(v)) else v


# SETTLED THRESHOLDS (2026-08-30; plan shiny-tickling-duckling.md S3). These are
# SAFETY-CRITICAL and must NOT be silently changed -- any change needs a new rationale.
# The debounce is per TELEMETRY POINT (each _hw_telemetry_frame call, ~4-10 Hz), NOT
# per scan line: a lithography pass over a rough sample momentarily blips SUM when the
# tip hits a feature, so a single good point resets the streak and only a sustained
# collapse trips.
SUM_ABS_FLOOR_V = 0.5        # absolute SUM floor. NOT 0.1 V -- 0.1 V is the laser-OFF
                             # dark floor; 0.5 V = serious signal degradation with the
                             # laser ON (SUM < 0.5 V -> detector effectively blind).
SUM_BASELINE_FRAC = 0.5      # relative arm: SUM below 50% of the pre-scan baseline.
INTERLOCK_DEBOUNCE_POINTS = 5  # consecutive telemetry points before a SUM/deflection
                             # trip (blip-tolerant; one good point resets the streak).
# Deflection interlock: |Deflection| > SUM at the SAME telemetry point means deflection
# exceeds the total detector signal (physically |defl| <= SUM), i.e. saturation /
# detector-integrity loss (deflection's normal range is +/-10 V). No baseline; same
# 5-point debounce.


def _eval_interlocks(defl, sum_v) -> None:
    """Evaluate the SUM-collapse and deflection interlocks for ONE telemetry point.
    The caller has already gated this to a live scan (guard armed + _rail_feeding). Each
    interlock debounces INTERLOCK_DEBOUNCE_POINTS consecutive points; a None/NaN channel
    leaves that streak UNCHANGED (neither trips nor clears). On a trip it stashes a
    pending interlock trip ({reason, stopped_reason}) and sets _hw_abort so the in-flight
    pass stops NOW -- the supervisor safe_parks on it POST-PUMP (the single withdraw
    site), never a second mid-frame withdraw. Runs on the router thread: lock-guarded,
    no await, no bridge read. Best-effort (the router catches any exception)."""
    with hub.host._rail_lock:
        if (hub.host._interlock_pending_trip is not None
                or hub.host._rail_pending_trip is not None):
            return                                       # already tripping this frame
        s = _finite(sum_v)
        d = _finite(defl)
        base = getattr(hub.host, "_sum_baseline", None)
        # SUM interlock: absolute floor OR (baseline known) below 50% of baseline. A
        # None/NaN SUM leaves the streak unchanged (neither trips nor clears).
        if s is not None:
            collapsed = (s < SUM_ABS_FLOOR_V
                         or (base is not None and s < SUM_BASELINE_FRAC * base))
            if collapsed:
                hub.host._sum_low_streak += 1
                if hub.host._sum_low_streak >= INTERLOCK_DEBOUNCE_POINTS:
                    hub.host._interlock_pending_trip = {
                        "reason": (f"SUM lost: {INTERLOCK_DEBOUNCE_POINTS} consecutive "
                                   f"telemetry points below {SUM_ABS_FLOOR_V:g} V or "
                                   f"50% of the pre-scan baseline -- detector signal "
                                   f"collapsed"),
                        "stopped_reason": "sum_lost"}
                    _signal_stop(hub, STOP_SCAN_PASS)
                    return
            else:
                hub.host._sum_low_streak = 0
        # Deflection interlock: |defl| > SUM at the same point. None on EITHER channel
        # leaves the streak unchanged.
        if d is not None and s is not None:
            if abs(d) > s:
                hub.host._defl_high_streak += 1
                if hub.host._defl_high_streak >= INTERLOCK_DEBOUNCE_POINTS:
                    hub.host._interlock_pending_trip = {
                        "reason": (f"deflection integrity: |Deflection| > SUM for "
                                   f"{INTERLOCK_DEBOUNCE_POINTS} consecutive telemetry "
                                   f"points (saturation / detector-integrity loss)"),
                        "stopped_reason": "deflection"}
                    _signal_stop(hub, STOP_SCAN_PASS)
                    return
            else:
                hub.host._defl_high_streak = 0


def current_scan_is_ac() -> bool:
    """True when the current (or most recent) scan config is AC / tapping mode -- the
    ONLY mode in which amplitude is a meaningful readout. Contact and current modes
    drive Z on deflection (amplitude is not the feedback channel), so the host blanks
    the amplitude display there. The active continuous run config takes precedence (an
    overnight run), else the last single-scan config; an unknown/absent mode -> not AC.
    Used by the /ws heartbeat to gate snap['amp_v'] (telemetry/display only -- there is
    NO amplitude safety interlock)."""
    cfg = hub.host.scan_repeat.get("config") or hub.host.last_scan_config or {}
    return str(cfg.get("mode", "")).lower() == "ac"


def _hw_telemetry_frame(x, y, zdrive, defl, sum_v, amp) -> None:
    """Consume one TELEM frame (RAW volts): update hub._hw_telemetry, broadcast the
    live tip marker when Show-Tip DISPLAY is on, and feed the rail-dwell guard from the
    polled Z-drive during a continuous scan frame. The Z-drive channel is selected on
    the Igor side (AFM_TipTick: `Output.Z` MFP-3D / `Height` Cypher, matching the
    servo/withdraw fix) and arrives positionally as `zdrive`. Best-effort throughout: an
    exception here is caught by the router so a bad frame can never wedge the reader.

    `amp` (Amplitude V) is carried for DISPLAY only -- the /ws heartbeat surfaces it in
    AC mode and blanks it otherwise. It drives NO safety interlock (only SUM and
    deflection do); it is projected onto hub._hw_telemetry so the readout can show it."""
    row = getattr(hub.host, "_scan_row", None)
    # (1) latest snapshot for the /ws heartbeat + the Scan-tab readout.
    #
    #     `t` (docs/FLEX_INTEGRATION_PLAN.md Sec A.3): the cache carried NO timestamp,
    #     so a consumer could not tell a fresh frame from a FROZEN one -- precisely
    #     the failure a dead bridge produces. A reader derives `age_s` from this.
    #
    #     `x_um` / `y_um`: the poll receives tip position as raw XSensor/YSensor
    #     volts, and the conversion to lab microns used to happen ONLY inside the
    #     tip_pos broadcast below, and only while the Show-Tip DISPLAY toggle was on.
    #     A scripted consumer correlating a signal against tip position cannot depend
    #     on a GUI toggle, so the conversion is done UNCONDITIONALLY here, through the
    #     same tip_stream helper and the same hub.hw_cal the broadcast uses. None when
    #     the raw volts were unreadable.
    #
    #     `height_nm` is kept for wire compatibility but is ALWAYS None: ZSensor
    #     (topography) is NOT one of the polled channels, so no true height is
    #     derivable from this poll. It is dropped from the FLEX wire surface rather
    #     than shipped as a permanently-null field.
    tx0, ty0 = _finite(x), _finite(y)
    x_um = y_um = None
    if tx0 is not None and ty0 is not None:
        try:
            from ..tip_stream import tippos_to_um
            x_um, y_um = tippos_to_um(tx0, ty0, hub.host.hw_cal)
            x_um, y_um = round(x_um, 3), round(y_um, 3)
        except Exception:                                # noqa: BLE001
            x_um = y_um = None                           # never break the safety feed
    hub.host._hw_telemetry = {
        "t": time.time(),
        "row": row,
        "x_um": x_um,
        "y_um": y_um,
        "defl_v": _finite(defl),
        "zdrive_v": _finite(zdrive),
        "sum_v": _finite(sum_v),
        "amp_v": _finite(amp),
        "height_nm": None,
    }
    # ---- S3: SUM-collapse + deflection INTERLOCKS, evaluated per TELEMETRY POINT.
    #      Bounded to a live scan by the SAME armed/_rail_feeding gate as the rail feed
    #      below, and routed through the SAME single post-pump withdraw site (a pending
    #      interlock trip + _hw_abort, which the supervisor safe_parks on). Evaluated
    #      before the marker/guard-feed so a trip preempts them this frame.
    _guard = getattr(hub.host, "rail_guard", None)
    if (_guard is not None and _guard.armed
            and getattr(hub.host, "_rail_feeding", False)):
        _eval_interlocks(defl, sum_v)
    # Fourth-slice runtime interlocks (12.1): feed the always-on TELEM point into the
    # armed WriteInterlocks (armed for approach/scan/write). A no-op unless armed; on a
    # trip its on_trip fires the out-of-band failsafe retract (never blocks this thread).
    try:
        from .engage import feed_hw_interlocks
        feed_hw_interlocks(hub, z_out_v=zdrive, sum_v=sum_v, defl_v=defl)
    except Exception:                                    # noqa: BLE001,S110
        pass
    # (2) live tip marker: broadcast tip_pos WHEN display is on. The conversion
    #     itself is no longer gated on this toggle (see the cache write above); only
    #     the BROADCAST is.
    tx, ty = tx0, ty0
    if getattr(hub, "_show_tip_display", False) and tx is not None and ty is not None:
        loop = getattr(hub, "_loop", None)
        if loop is not None:
            try:
                from ..tip_stream import tip_pos_msg
                msg = tip_pos_msg(x, y, hub.host.hw_cal)
                asyncio.run_coroutine_threadsafe(hub.broadcast(msg), loop)
            except Exception:                                # noqa: BLE001,S110
                pass    # display is best-effort; MUST NOT abort the guard feed below
    # (3) rail-dwell guard feed during a continuous scan FRAME (Guard A/B). Fed from
    #     the poll, NOT a host bridge read (which would race the scan pass's reader).
    #     _rail_feeding bounds this to the live frame, mirroring the twin _rail_poll.
    guard = getattr(hub.host, "rail_guard", None)
    if guard is None or not guard.armed or not getattr(hub.host, "_rail_feeding", False):
        return
    with hub.host._rail_lock:
        if (hub.host._rail_pending_trip is not None
                or hub.host._interlock_pending_trip is not None):
            return                                           # already tripping this frame
        last = hub.host._rail_last_row
        if row is not None and row != last:
            if last is not None:                             # a completed line -> Guard B
                trip = guard.close_line(last)
                if trip is not None:
                    # signal the in-flight scan to abort NOW (thread-safe Event) and
                    # stash the descriptor; the supervisor acts on it post-pump (the
                    # single withdraw site, so it never races the pass mid-frame).
                    hub.host._rail_pending_trip = trip
                    _signal_stop(hub, STOP_SCAN_PASS)
                    return
            hub.host._rail_last_row = row
        guard.add_sample(zdrive)


def install_hw_telemetry(arc) -> None:
    """Install the always-on TELEM consumer as the hardware backend's single router
    fan-out (arc.router.on_telem). Call once when the HardwareARC connects (app.py
    startup, or a test that attaches hub.hardware_arc). Idempotent: re-installing just
    re-points the one slot at the same consumer."""
    if getattr(arc, "router", None) is not None:
        arc.router.on_telem = _hw_telemetry_frame
