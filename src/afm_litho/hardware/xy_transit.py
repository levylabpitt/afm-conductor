"""XY slew guard: the ONE helper for every hardware XY move that is not a raster or a
write stroke (docs/XY_SLEW_SPEC.md, hazard #11). A setpoint jump to a far target is fast
enough to stick-slip the sample on the scanner and shift the tip relative to it.

  1. The move is ONE straight-line out-wave PAIR on the X/Y setpoints, fired as one
     event at `MOVE_SPEED_UM_S` (compile_object's spacing/timing), with an
     XSensor/YSensor capture pair armed on the SAME event: a bare FIRE with no input
     bank returns DONE immediately (afm_litho_run.ipf `AFM_FireEvent`), before the tip
     has moved.
  2. An ENGAGED tip is VERIFIED-withdrawn first; re-approach only if asked.
  3. Arrival is verified by a live sensor readback. A short landing is retried once,
     and never from a single readback: two reads must agree first, else raise.
  4. ANY failure -> withdraw and re-raise (never a silent hold).
  5. Every fire logs wave end vs readback; success logs one line with timing.

Withdraw reuses `hw_approach.hw_withdraw` (looked up at call time): one implementation
of the Z withdraw.

Re-approach is deliberately NOT `hw_approach.hw_engage`: that scales PANEL gains x100,
while the caller's z_igain is already RAW, so it would be ~100x too strong (crash risk).
Loop 2 is armed directly with the caller's raw igain instead.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from ..litho.chunker import MAX_OUTWAVE_POINTS
from ..litho.compiler import DEFAULT_INTERP_UM, time_segment

from .. import clock
from . import channels

# XY translate speed (um/s) for every guarded move; fixed by the spec.
MOVE_SPEED_UM_S = 5.0
# IR_SetRamp's cap (s); a longer hop uses rate=0 (vendor "maxSeconds is the duration").
# Used by `_plan_ramp` for lithography_run._hover_move.
RAMP_MAX_SECONDS = 10.0
# Settle floor on a ramp's planned duration; also the fire-wait timeout margin.
RAMP_SETTLE_MARGIN_S = 0.5
# A ramp's settle is the larger of the margin and this fraction of the move time.
RAMP_SETTLE_FRACTION = 0.25

# Pacing gap after every bridge command here; HardwareARC adds none of its own.
CMD_GAP_S = 0.02

# Arrival tolerance: the LARGER of a noise floor and a fraction of the planned distance.
ARRIVAL_TOL_UM = 1.0
ARRIVAL_TOL_FRAC = 0.02

# Gap between the two reads that must agree before a retry (tests pass 0).
RETRY_VERIFY_GAP_S = 0.2

# Early-DONE guard is skipped below this planned duration: a zero/sub-tick move's
# duration is microseconds, so the guard would only measure bridge noise.
GUARD_MIN_DURATION_S = 0.05

# Bank 0 / event "0", the same pair litho strokes and scans arm. Safe to share: this
# arm -> fire -> await is synchronous, callers arm bank 0 only AFTER it returns, and
# OUTPAIR always rebinds the bank before the next FIRE, so nothing rides a stale wave.
_MOVE_BANK = 0
_MOVE_EVENT = "0"

# Withdraw bounds: match hw_withdraw's defaults; tests pass 0.
DEFAULT_WITHDRAW_MAX_S = 2.0
DEFAULT_WITHDRAW_SETTLE_S = 0.5

# XY tracking loops 0/1 and Z loop 2: the validated channels every pass path uses.
_X_LOOP, _X_IN, _X_OUT = 0, channels.X_SENSOR, channels.OUTPUT_X
_Y_LOOP, _Y_IN, _Y_OUT = 1, channels.Y_SENSOR, channels.OUTPUT_Y
_X_SETPOINT = "PIDSLoop.0.Setpoint"
_Y_SETPOINT = "PIDSLoop.1.Setpoint"
Z_LOOP = 2
_DEFL = channels.DEFLECTION
_AMPL = channels.AMPLITUDE


def mode_loop_params(mode: str) -> tuple[str, float]:
    """(InputChannel, gain_sign) for loop 2 by mode; mirrors hw_approach's copy."""
    if str(mode).lower() == "ac":
        return _AMPL, -1.0
    return _DEFL, 1.0


def _um_to_v(um: float, sens: float, offset: float) -> float:
    return float(um) * 1e-6 / float(sens) + float(offset)


def _v_to_um(v: float, sens: float, offset: float) -> float:
    return (float(v) - float(offset)) * float(sens) * 1e6


def _mult32(n: int) -> int:
    """InWave lengths must be a multiple of 32 (td_xSetInWavePair)."""
    return ((int(n) + 31) // 32) * 32


def _read_xy_um(arc, cal: dict) -> tuple[float, float]:
    """Tip XY (um) from a LIVE sensor read, never a cache."""
    x_v = float(arc.td_ReadValue(_X_IN))
    y_v = float(arc.td_ReadValue(_Y_IN))
    x_um = _v_to_um(x_v, cal.get("x_lvdt_sens", 1.0), cal.get("x_lvdt_offset", 0.0))
    y_um = _v_to_um(y_v, cal.get("y_lvdt_sens", 1.0), cal.get("y_lvdt_offset", 0.0))
    return x_um, y_um


def _plan_ramp(dx_um: float, dy_um: float, cal: dict,
               move_speed_um_s: float) -> tuple[float, float, float]:
    """(rate_x, rate_y, maxsec) for a rate-limited XY ramp."""
    speed = move_speed_um_s if move_speed_um_s > 0 else MOVE_SPEED_UM_S
    dist_um = max(abs(dx_um), abs(dy_um))
    move_time = dist_um / speed
    need = move_time + max(RAMP_SETTLE_MARGIN_S, RAMP_SETTLE_FRACTION * move_time)
    if need > RAMP_MAX_SECONDS:
        return 0.0, 0.0, RAMP_MAX_SECONDS
    rate_x = speed * 1e-6 / float(cal.get("x_lvdt_sens", 1.0))
    rate_y = speed * 1e-6 / float(cal.get("y_lvdt_sens", 1.0))
    return rate_x, rate_y, need


def _build_wavepair_v(cur_x_um: float, cur_y_um: float, x_um: float, y_um: float,
                      cal: dict, move_speed_um_s: float
                      ) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """The straight-line XY setpoint out-wave PAIR (LVDT volts) for one move, sampled
    like a litho stroke (`DEFAULT_INTERP_UM` spacing, `time_segment` timing). `interp`
    is SIGNED ticks/point (negative == linear). Returns (xs_v, ys_v, interp,
    duration_s, n)."""
    speed = move_speed_um_s if move_speed_um_s > 0 else MOVE_SPEED_UM_S
    dist_um = math.hypot(x_um - cur_x_um, y_um - cur_y_um)
    n = max(2, round(dist_um / DEFAULT_INTERP_UM) + 1)
    if n > MAX_OUTWAVE_POINTS:
        raise ValueError(
            f"xy_transit: a {dist_um:.1f} um move needs {n:,} points, over the "
            f"{MAX_OUTWAVE_POINTS:,}-point out-wave cap -- the instrument would "
            f"refuse it.")
    frac = np.linspace(0.0, 1.0, n)
    xs_um = cur_x_um + frac * (x_um - cur_x_um)
    ys_um = cur_y_um + frac * (y_um - cur_y_um)
    t = time_segment(np.column_stack([xs_um, ys_um]), speed)
    interp = -float(t["ticks_per_point"])
    x_sens, x_off = float(cal.get("x_lvdt_sens", 1.0)), float(cal.get("x_lvdt_offset", 0.0))
    y_sens, y_off = float(cal.get("y_lvdt_sens", 1.0)), float(cal.get("y_lvdt_offset", 0.0))
    xs_v = xs_um * 1e-6 / x_sens + x_off
    ys_v = ys_um * 1e-6 / y_sens + y_off
    return xs_v, ys_v, interp, float(t["duration_s"]), n


class TransitAborted(RuntimeError):
    """The run was aborted during the move: no retry, no re-approach."""


def xy_transit(arc, cal: dict, x_um: float, y_um: float, *,
               engaged: bool, reapproach: bool, setpoint: float | None = None,
               mode: str = "contact",
               x_pgain: float = 0.0, x_igain: float = 0.0,
               y_pgain: float = 0.0, y_igain: float = 0.0,
               z_pgain: float = 0.0, z_igain: float = 1200.0,
               move_speed_um_s: float = MOVE_SPEED_UM_S,
               ramp_settle_s: float = RAMP_SETTLE_MARGIN_S,
               withdraw_rate: float = 80.0,
               withdraw_max_s: float = DEFAULT_WITHDRAW_MAX_S,
               withdraw_settle_s: float = DEFAULT_WITHDRAW_SETTLE_S,
               cmd_gap_s: float = CMD_GAP_S,
               retry_verify_gap_s: float = RETRY_VERIFY_GAP_S,
               should_abort: Callable[[], bool] | None = None,
               log: Callable[[str], None] | None = None) -> dict:
    """Move the tip to (x_um, y_um), withdrawn, as one captured out-wave-pair move.

    `arc`: HardwareARC or LithographyRun (ir_SetPISLoop, td_xSetOutWavePair,
    td_xSetInWavePair, fire, td_ReadValue, is_cypher).
    `engaged`: the CALLER's knowledge that the tip is on the surface; only then does a
    verified withdraw run first.
    `reapproach`: re-arm loop 2 at `setpoint` afterwards. Pass False when the caller's
    next step re-engages anyway.
    `ramp_settle_s`: fire-wait timeout margin; >0 also enables the early-DONE guard.
    All waits are real safety-tuned defaults; tests pass 0.
    `should_abort`: the run's abort flag, checked on entry, after the withdraw and after a
    move (an abort cuts the move short). Once set it raises TransitAborted: no XY move, no
    retry, no re-approach, and no withdraw here, since the abort's own park retracts.

    Raises (after withdrawing) on any failure. Returns a dict with `retried` (a second
    attempt was needed) and `withdraw` (the verified withdraw's result, None if the tip was
    not engaged), and whose `timing_s` maps
    entry_readback / withdraw_done / loops_armed / wave_uploaded / fired /
    done_received / readback_done to seconds since entry.
    """
    if reapproach and setpoint is None:
        raise ValueError("xy_transit: reapproach=True requires a setpoint")
    log = log or (lambda msg: None)
    from .hw_approach import hw_withdraw  # looked up at call time: tests patch it

    t0 = clock.now()
    timing: dict[str, float] = {}

    def _elapsed() -> float:
        return clock.now() - t0

    # LithographyRun has no z_out_channel; its loop-2 channel is `z_out` (Height on a Cypher).
    withdraw_z_out = getattr(arc, "z_out_channel", None) or getattr(arc, "z_out", None)

    def _do_withdraw() -> dict:
        return hw_withdraw(arc, rate=withdraw_rate, max_seconds=withdraw_max_s,
                           settle_margin=withdraw_settle_s, z_out=withdraw_z_out)

    def _stop_if_aborted() -> None:
        if should_abort is not None and should_abort():
            raise TransitAborted("XY transit: the run was aborted during the move")

    def _pace() -> None:
        if cmd_gap_s:
            clock.sleep(cmd_gap_s)

    def _fire_move(from_x_um: float, from_y_um: float
                   ) -> tuple[float, float, float | None, float | None]:
        """One move attempt: arm loops 0/1 at the start, arm out-wave + capture pair,
        fire, then read back. Returns (readback_x_um, readback_y_um, wave_end_x_um,
        wave_end_y_um); wave end is None if the bridge returned no capture."""
        start_x = _um_to_v(from_x_um, cal.get("x_lvdt_sens", 1.0),
                           cal.get("x_lvdt_offset", 0.0))
        start_y = _um_to_v(from_y_um, cal.get("y_lvdt_sens", 1.0),
                           cal.get("y_lvdt_offset", 0.0))
        # Arm at the move START so arming does not itself jump; the loops need real
        # gains or Output.X/Y never moves.
        arc.ir_SetPISLoop(_X_LOOP, "Always,Never", _X_IN, start_x,
                          x_pgain, x_igain, 0.0, _X_OUT, *channels.LOOP_OUTPUT_LIMITS_V)
        _pace()
        arc.ir_SetPISLoop(_Y_LOOP, "Always,Never", _Y_IN, start_y,
                          y_pgain, y_igain, 0.0, _Y_OUT, *channels.LOOP_OUTPUT_LIMITS_V)
        _pace()
        timing["loops_armed"] = _elapsed()

        xs_v, ys_v, interp, duration_s, n = _build_wavepair_v(
            from_x_um, from_y_um, x_um, y_um, cal, move_speed_um_s)
        ticks = max(1, abs(round(interp)))         # one capture sample per point,
        cap_len = _mult32(n)                       # padded to a multiple of 32
        # `arc.fire` has no per-call timeout; raise `arc.timeout` to cover this move,
        # then restore it.
        orig_timeout = getattr(arc, "timeout", None)
        if orig_timeout is not None:
            arc.timeout = max(float(orig_timeout), duration_s + float(ramp_settle_s))
        try:
            arc.td_xSetOutWavePair(_MOVE_BANK, _MOVE_EVENT, _X_SETPOINT, xs_v,
                                   _Y_SETPOINT, ys_v, interp)
            _pace()
            # Arm capture on the SAME bank/event BEFORE firing, so DONE is the real
            # capture-fill completion, not the bare-FIRE immediate DONE.
            arc.td_xSetInWavePair(_MOVE_BANK, _MOVE_EVENT, _X_IN, _Y_IN,
                                  cap_len, ticks)
            _pace()
            timing["wave_uploaded"] = _elapsed()
            fire_start = clock.now()
            timing["fired"] = _elapsed()
            cap = arc.fire(_MOVE_EVENT, n)
            timing["done_received"] = _elapsed()
            _pace()
            # Defense in depth: if DONE still beat the planned duration, wait out the
            # remainder + ramp_settle_s before trusting the readback.
            if ramp_settle_s > 0 and duration_s > GUARD_MIN_DURATION_S:
                remaining = duration_s - (clock.now() - fire_start)
                if remaining > 0:
                    clock.sleep(remaining + float(ramp_settle_s))
        finally:
            if orig_timeout is not None:
                arc.timeout = orig_timeout

        wave_x_um = wave_y_um = None
        xs_cap = cap.get(_X_IN) if cap else None
        ys_cap = cap.get(_Y_IN) if cap else None
        if xs_cap is not None and len(xs_cap) and ys_cap is not None and len(ys_cap):
            wave_x_um = _v_to_um(float(xs_cap[-1]), cal.get("x_lvdt_sens", 1.0),
                                 cal.get("x_lvdt_offset", 0.0))
            wave_y_um = _v_to_um(float(ys_cap[-1]), cal.get("y_lvdt_sens", 1.0),
                                 cal.get("y_lvdt_offset", 0.0))

        rx, ry = _read_xy_um(arc, cal)
        timing["readback_done"] = _elapsed()
        if wave_x_um is not None:
            log(f"XY transit: wave ended at ({wave_x_um:.3f}, {wave_y_um:.3f}) um, "
                f"readback ({rx:.3f}, {ry:.3f}) um")
        return rx, ry, wave_x_um, wave_y_um

    withdrew = False
    retried = False
    snap: dict | None = None
    try:
        _stop_if_aborted()
        cur_x_um, cur_y_um = _read_xy_um(arc, cal)
        timing["entry_readback"] = _elapsed()
        dist_um = math.hypot(x_um - cur_x_um, y_um - cur_y_um)

        if engaged:
            snap = _do_withdraw()
            withdrew = True
            timing["withdraw_done"] = _elapsed()
            # A timed-out withdraw verify before an XY ramp is logged LOUDLY, never
            # silent.
            if snap.get("verify_timeout"):
                log(f"WARNING: XY transit withdraw verify timed out (z_v="
                    f"{snap.get('z_v')!r}) -- the retract may not be confirmed")

        tol_um = max(ARRIVAL_TOL_UM, ARRIVAL_TOL_FRAC * dist_um)
        _stop_if_aborted()

        # Attempt 1: from wherever the tip actually IS (the entry readback).
        rx, ry, _wx, _wy = _fire_move(cur_x_um, cur_y_um)
        err_um = math.hypot(x_um - rx, y_um - ry)
        verified = err_um <= tol_um

        if not verified:
            _stop_if_aborted()
            log(f"XY transit: landed {err_um:.3f} um short of target "
                f"(tol {tol_um:.3f} um) -- retrying")
            # Retry safety: never re-arm loops 0/1 from a single (possibly stale)
            # readback. A second read must AGREE with the first, or raise.
            read1 = (rx, ry)
            if retry_verify_gap_s:
                clock.sleep(retry_verify_gap_s)
            read2 = _read_xy_um(arc, cal)
            disagree_um = math.hypot(read2[0] - read1[0], read2[1] - read1[1])
            if disagree_um > tol_um:
                log(f"XY transit: retry-safety abort -- two reads "
                    f"{retry_verify_gap_s:g} s apart disagree by {disagree_um:.3f} "
                    f"um (tol {tol_um:.3f} um): read1 ({read1[0]:.3f}, "
                    f"{read1[1]:.3f}) um, read2 ({read2[0]:.3f}, {read2[1]:.3f}) "
                    f"um -- refusing to retry from an untrusted position")
                raise RuntimeError(
                    f"XY transit: two position reads {retry_verify_gap_s:g} s "
                    f"apart disagree by {disagree_um:.3f} um (tol {tol_um:.3f} "
                    f"um) -- refusing to retry from an untrusted position")
            retried = True
            rx, ry, _wx, _wy = _fire_move(read2[0], read2[1])
            err_um = math.hypot(x_um - rx, y_um - ry)
            verified = err_um <= tol_um

        if not verified:
            _stop_if_aborted()
            raise RuntimeError(
                f"XY transit: {err_um:.3f} um short of destination after one "
                f"retry (tol {tol_um:.3f} um)")

        if reapproach:
            _stop_if_aborted()
            z_in, z_sign = mode_loop_params(mode)
            z_out = getattr(arc, "z_out_channel", None) or getattr(arc, "z_out", channels.Z_OUT)
            arc.ir_SetPISLoop(Z_LOOP, "Always,Never", z_in, float(setpoint),
                              float(z_pgain) * z_sign, float(z_igain) * z_sign,
                              0.0, z_out, *channels.LOOP_OUTPUT_LIMITS_V)
    except TransitAborted:
        raise
    except Exception:
        try:
            _do_withdraw()
        except Exception:               # noqa: BLE001,S110 -- never mask the original error
            pass
        raise

    state = "tip withdrawn" if withdrew else "tip was not engaged"
    # Per-phase timing: arm / upload (out+in wave) / move (fire wait) / verify.
    arm_ref = timing.get("withdraw_done", timing.get("entry_readback", 0.0))
    arm_s = timing.get("loops_armed", arm_ref) - arm_ref
    upload_s = timing.get("wave_uploaded", 0.0) - timing.get("loops_armed", 0.0)
    move_s = timing.get("done_received", 0.0) - timing.get("fired", 0.0)
    verify_s = timing.get("readback_done", 0.0) - timing.get("done_received", 0.0)
    log(f"XY transit: {dist_um:.1f} um at {move_speed_um_s:g} um/s, {state}, verified "
        f"[arm {arm_s:.2f}s, upload {upload_s:.2f}s, move {move_s:.2f}s, "
        f"verify {verify_s:.2f}s]")
    return {"distance_um": dist_um, "withdrawn": withdrew,
            "reapproached": bool(reapproach), "verified": True, "retried": retried,
            "withdraw": snap, "timing_s": timing}
