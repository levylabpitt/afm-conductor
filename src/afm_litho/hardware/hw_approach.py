"""Real approach (fine engage) and withdraw on the hardware backend.

Engage closes loop 2 (feedback input -> Z output) at a positive setpoint so the tip
tracks the surface. Withdraw (both platforms): re-arm loop 2
so the servo drives Height back to 0 V. Vendor recipe and citations:
docs/engage_withdraw.md. The coarse approach must already be within Z-piezo range.

Sign convention (safety-critical): positive Output.Z = piezo extended TOWARD the
surface; 0 V = fully retracted / safe. In free air Deflection < setpoint, so a POSITIVE
setpoint with positive gain drives Output.Z up until Deflection == setpoint.
"""
from __future__ import annotations

import math

from .. import clock
from ..hardware import channels, retract
from ..hardware.channels import Z_OUT

# Vendor Height-loop channels (docs/engage_withdraw.md).
Z_LOOP = 2
_DEFL = channels.DEFLECTION            # contact-mode feedback input (AC uses Amplitude, below)
_Z_OUT = Z_OUT           # 0 V retracted, positive into surface
Z_MIN, Z_MAX = channels.LOOP_OUTPUT_LIMITS_V  # scanning Z range (the hardware channel clamp)

# Photodiode SUM channel on both Cypher and MFP-3D: `DetectorSum` (lowercase "sum"
# reads 0 V on the Cypher). afm_litho_run.ipf AFM_TipTick repeats the literal; keep in sync.
SUM_CHANNEL = channels.DETECTOR_SUM
# CYPHER retract, through the loop: Z is a loop-driven servo output (`Height`), NOT a
# raw DAC, so a direct ramp does nothing (confirmed on the rig). Re-arm loop 2 at -10 V
# (Deflection never reaches it off-surface, so the servo drives Height back); OutputMin
# 0, NOT -10, stops Height at 0 V = retracted. Validated command (it retracts):
#   ir_SetPISLoop(2,"Always,Never",`Deflection`,-10,0,1000,0,`Height`,0,150)
# Mirrors the vendor Cypher withdraw (HeightLoop.Setpoint = -10*GainSign, Lithography.ipf:3873).
_Z_RETRACT_SETPOINT = -10.0    # deeply-negative Deflection setpoint -> Height driven fully back
_Z_RETRACT_IGAIN = 1000.0      # raw loop integral gain (already controller units, NOT panel*IScale)
Z_RETRACT_OUTMIN = 0.0        # OUTPUT min for withdraw: Height stops at 0 V (engaged servo uses -10)

# AC / TAPPING Z loop (SAFETY-CRITICAL: a wrong channel or gain sign drives the tip INTO
# the sample). Input `Amplitude`; gain NEGATED on BOTH P and I, because amplitude FALLS
# on approach (deflection rises). Setpoint is an ABSOLUTE amplitude below free
# amplitude; relative setpoint is contact-only. Vendor: HardWare.ipf:8923/8964
# (GainSign), 9388-9391 (applied to P and I). Panel truth: contact +1000/Deflection,
# AC -1000/Amplitude. Full audit trail: docs/engage_withdraw.md.
_AMPL = channels.AMPLITUDE            # AC-mode feedback input (the PIDS-loop InputChannel)
_CONTACT_GAIN_SIGN = 1.0       # contact: deflection RISES on approach -> positive gain
_AC_GAIN_SIGN = -1.0           # AC: amplitude FALLS on approach -> NEGATED gain (panel -1000)


def mode_loop_params(mode: str) -> tuple[str, float]:
    """(InputChannel, gain_sign) for loop 2: AC -> Amplitude, NEGATED gain; contact and
    current -> Deflection, positive. The AC inversion keeps the loop driving TOWARD the
    setpoint (safety-critical; see the _AMPL block)."""
    if str(mode).lower() == "ac":
        return _AMPL, _AC_GAIN_SIGN
    return _DEFL, _CONTACT_GAIN_SIGN


# Free-air deflection read: the vendor's metered `DeflectionValue` (ScanMaster.ipf:4665),
# not the feedback input name `Deflection`.
_FREE_AIR_DEFL = channels.DEFLECTION_VALUE

# MFP-3D HV Z drive (-10..150 V), the rail-dwell channel (the DRIVE, not ZSensor).
# Fallback only when the arc has no z_out_channel (HardWare.ipf:9595-9596).
_Z_DRIVE = Z_OUT

# hw_engage/hw_set_zgain take PANEL gains; the controller wants IScale=100 / PScale=0.1
# (ARGetFeedbackParms, HardWare.ipf:8874-8875). Raw panel numbers make the loop ~100x too
# weak. (Litho's z_igain 1200 is already a raw controller gain.)
_ISCALE = 100.0
_PSCALE = 0.1


# BOUNDED read (s) on every WITHDRAW-path read, never the 30 s default: the retract is
# already commanded, so an unconfirmed `verify_timeout: True` beats an apparent hang
# (docs/FLEX_INTEGRATION_PLAN.md §F.2).
VERIFY_READ_TIMEOUT_S = 2.0


def _read_ex(arc, param: str, *, timeout: float | None = None) -> tuple[float, bool]:
    """Read one channel; return `(value_or_nan, timed_out)`. `timeout` bounds the read
    where the backend supports it (the twin and stubs fall back to unbounded)."""
    try:
        if timeout is not None:
            try:
                return float(arc.td_ReadValue(param, timeout=timeout)), False
            except TypeError:
                pass                       # backend has no bounded-read kwarg
        return float(arc.td_ReadValue(param)), False
    except Exception as exc:  # noqa: BLE001 - a read hiccup becomes NaN and must not break engage
        return math.nan, type(exc).__name__ == "BridgeReadTimeout"


def _read(arc, param: str, *, timeout: float | None = None) -> float:
    return _read_ex(arc, param, timeout=timeout)[0]


def _jsafe(v: float) -> float | None:
    """NaN/inf -> None, so a failed read cannot 500 the withdraw reply."""
    return None if (v is None or math.isnan(v) or math.isinf(v)) else v


def snapshot(arc, *, engaged: bool, read_timeout: float | None = None) -> dict:
    """Deflection and Z output for the panel readout, NaN-safe. `read_timeout` bounds
    both reads (withdraw path); a timed-out read gives a null field plus
    `verify_timeout: True`."""
    defl, defl_to = _read_ex(arc, _DEFL, timeout=read_timeout)
    z, z_to = _read_ex(arc, channels.HEIGHT_LOWERCASE, timeout=read_timeout)  # vendor "height" == Output.Z
    snap = {"engaged": engaged, "defl_v": _jsafe(defl), "z_v": _jsafe(z)}
    if defl_to or z_to:
        snap["verify_timeout"] = True
    return snap


# hw_engage's defaults: how close to the setpoint counts as reached, and how long it polls.
# The litho set-down shares them (lithography_run: the least engage step, the engage timeout).
ENGAGE_TOL_V = 0.05
ENGAGE_SETTLE_S = 6.0


def hw_engage(arc, *, setpoint: float, pgain: float, igain: float,
              settle_s: float = ENGAGE_SETTLE_S, tol: float = ENGAGE_TOL_V,
              poll_s: float = 0.2, mode: str = "contact") -> dict:
    """Close the Z loop, then poll the feedback channel until it reaches the setpoint
    (within `tol`) or `settle_s` elapses; `reached_setpoint` False means still driving.

    mode: SAFETY-CRITICAL, selects channel AND gain sign (contact/current Deflection
      positive; AC Amplitude NEGATED).
    setpoint: volts, POSITIVE (contact: above free air; AC: absolute amplitude).
    pgain/igain: PANEL gains, scaled here (P*0.1, I*100) and sign-adjusted.
    """
    in_chan, gain_sign = mode_loop_params(mode)
    # The Z output is the Height alias: a wrong channel drives NOTHING and the tip does
    # not track the surface (crash).
    arc.ir_SetPISLoop(Z_LOOP, "Always,Never", in_chan, float(setpoint),
                      float(pgain) * _PSCALE * gain_sign,
                      float(igain) * _ISCALE * gain_sign,
                      0.0, arc.z_out_channel, Z_MIN, Z_MAX)
    reached = False
    if settle_s > 0:
        steps = max(1, int(round(settle_s / poll_s)))
        for i in range(steps):
            d = _read(arc, in_chan)
            if not math.isnan(d) and abs(d - float(setpoint)) <= tol:
                reached = True
                break
            if i < steps - 1:
                clock.sleep(poll_s)
    snap = snapshot(arc, engaged=True)
    snap["reached_setpoint"] = reached
    return snap


def hw_read_free_air(arc) -> float:
    """Free-air deflection (V) for the relative-setpoint re-reference. Call only AFTER a
    full withdraw + settle. NaN on a failed read, which the caller treats as fail-safe."""
    return _read(arc, _FREE_AIR_DEFL)


def hw_read_free_amplitude(arc) -> float:
    """Free-air AC amplitude (V), the AC baseline the fraction band divides by. Call
    only withdrawn with the AC drive on. NaN on a failed read."""
    return _read(arc, _AMPL)


def hw_read_zdrive(arc) -> float:
    """HV Z-drive voltage for the rail-dwell guard. NaN on a failed read (the guard drops
    it: never fabricate or clear a rail hit). Reads arc.z_out_channel, because the
    Cypher's Output.Z DAC never moves."""
    return _read(arc, getattr(arc, "z_out_channel", _Z_DRIVE))


def hw_set_zgain(arc, *, setpoint: float, pgain: float, igain: float,
                 mode: str = "contact") -> dict:
    """Re-arm the Z loop with new gains WHILE engaged (StartEvent "Always" rewrites the
    running loop in place). No settle poll.

    mode MUST match the engaged mode (SAFETY-CRITICAL): a mismatch flips an AC loop to
    Deflection / positive gain mid-run and crashes the tip."""
    in_chan, gain_sign = mode_loop_params(mode)
    # Output is the Height alias (see hw_engage).
    arc.ir_SetPISLoop(Z_LOOP, "Always,Never", in_chan, float(setpoint),
                      float(pgain) * _PSCALE * gain_sign,
                      float(igain) * _ISCALE * gain_sign,
                      0.0, arc.z_out_channel, Z_MIN, Z_MAX)
    return snapshot(arc, engaged=True)


def hw_withdraw(arc, *, rate: float = 80.0, max_seconds: float = 2.0,
                settle_margin: float = 0.5, z_out: str | None = None) -> dict:
    """Retract, then WAIT out the move before reading back. Both platforms re-arm loop 2
    at -10 with OutputMin 0, so Height parks at 0 V (_Z_RETRACT_* block); stopping the
    loop and ramping does NOT retract a Cypher.

    `rate` is unused now that no ramp is sent; the signature is pinned by the callers.
    `z_out`: the loop-2 output channel. Callers whose `arc` has no `z_out_channel`
    (LithographyRun) must pass it; otherwise `arc.z_out_channel` is used. Never a
    hard-coded `Output.Z`: on a Cypher that channel is inert (INSTRUMENT_MODEL_SPEC §3)."""
    if z_out is None:
        z_out = getattr(arc, "z_out_channel", _Z_OUT)
    retract.withdraw_rearm(arc, z_out, loop=Z_LOOP, defl_channel=_DEFL,
                           setpoint=_Z_RETRACT_SETPOINT, igain=_Z_RETRACT_IGAIN,
                           out_min=Z_RETRACT_OUTMIN, out_max=Z_MAX)
    wait = max_seconds + settle_margin
    if wait > 0:
        clock.sleep(wait)
    # BOUNDED readback. This is not the independent-sensor proof: that is safe_park's.
    return snapshot(arc, engaged=False, read_timeout=VERIFY_READ_TIMEOUT_S)
