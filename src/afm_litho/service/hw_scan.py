"""Hardware scan path: a raster on the REAL backend, as one callback-chained pass.

On the simulation backend the scan endpoint keeps the fast `lfilter` renderer
(`passes.scan_line`). On the real backend `run_hw_scan_chained` arms the whole raster
(`raster_pass` builds the waves) and Igor advances the lines on the capture-fill
callback. ZSensor (topography) and Deflection/Amplitude (error) are captured; each
completed row is projected and broadcast as a `{"type":"line"}` message so the browser
fills line by line. No `HeightImage0` harvest; we measure it ourselves.
"""

from __future__ import annotations

import asyncio
import math
import queue as _queue
import time
from typing import Awaitable, Callable

import numpy as np

from ..hardware import channels
from ..hardware.channels import Z_OUT
from ..hardware import bridge_protocol as P
from ..hardware.xy_transit import TransitAborted, xy_transit
from ..pass_executor import raster_pass, raster_path_um
from ..hardware.hw_approach import _ISCALE, _PSCALE, Z_LOOP, mode_loop_params
from .host.safe_park import PARK_ARC_ABORT, request_park
from .step_notes import NO_NOTES

_HEIGHT = channels.Z_SENSOR          # measured Z position -> topography
_ERR = channels.DEFLECTION          # DEFAULT (contact) error-capture channel name
_BIAS = channels.USER_IN2           # MEASURED tip bias: BNC In2, teed bias line


def _err_chan(mode) -> str:
    """The scan's ERROR *capture/display* channel by imaging mode: contact ->
    `Deflection` (cantilever bend), AC/tapping -> `Amplitude` (the meaningful error
    signal near the surface). Mirrors the twin/sim vocabulary (host.scan_pump._scan_pump
    and passes.scan_line, {contact:Deflection, ac:Amplitude}). This is the RECORDED
    image/scope channel only -- the Z feedback servo's input channel + gain sign are a
    SEPARATE concern handled by hw_approach.mode_loop_params (which the chained arm
    already uses for loop 2). `Amplitude` is the rig scalar/channel name confirmed by
    the engage + telemetry paths; it is the second (non-callback) channel of the bank-0
    in-wave pair, so only its NAME changes -- no extra bank."""
    return channels.AMPLITUDE if str(mode).lower() == "ac" else _ERR

# Loops 0/1 (XY tracking) and 2 (Z) are armed with output limits of -10..150 V.
LOOP_LIMITS_V = channels.LOOP_OUTPUT_LIMITS_V
XY_LOOPS = (0, 1)   # X and Y tracking loops; the Z loop is Z_LOOP


def scan_line_count(config) -> int:
    """Lines in the raster: the request, or one per pixel when the request leaves it 0."""
    return int(config.lines) or int(config.pixels)


def scan_points_per_line(pixels: int, trace_retrace: bool) -> int:
    """Points played per line: trace and retrace double it."""
    return 2 * pixels if trace_retrace else pixels


def scan_interpolation(pixels: int, line_rate_hz: float) -> int:
    """The negative out-wave interpolation that plays `pixels` points per line at
    `line_rate_hz` on the 50 kHz clock."""
    pts_per_s = max(1.0, pixels * float(line_rate_hz))
    return -max(1, round(50000.0 / pts_per_s))


def scan_decimation(interp: int) -> int:
    """One captured sample per played point."""
    return max(1, abs(int(interp)))


def scan_capture_channels(mode) -> tuple[str, str, str]:
    """Channels a scan records: topography, the mode's error channel, measured tip bias."""
    return (_HEIGHT, _err_chan(mode), _BIAS)


def scan_reapproaches(engaged: bool, config) -> bool:
    """The transit re-engages when the tip was engaged on entry and the scan will not
    engage itself."""
    return bool(engaged and not bool(getattr(config, "engage", False)))


# Z HEIGHT loop (SAFETY-CRITICAL). Index 2 is the SAME binding hw_approach.py
# engages -- rig-verified as the Height loop (AC engage armed loop 2 correctly).
# The concrete channels below REPLACE the mode-agnostic aliases "Fast%Input" /
# "Z%Output" this scan used to arm: those aliases do NOT resolve on this rig's
# crosspoint (ae_BadParm) -- the same reject lithography_run.py documents and the
# hardware-bringup fix (Deflection/Output.Z) proved. The INPUT channel and gain
# SIGN are mode-dependent (contact -> Deflection/+; AC -> Amplitude/-, because the
# amplitude error inverts near the surface), so they come from the shared
# mode_loop_params pinned to v16 HardWare.ipf (ARGetImagingMode: cContactMode
# GainSign=+1 Input=`Deflection` :8923-8925; cACMode GainSign=-1 Input=`Amplitude`
# :8964-8968, applied to P+I at :9388-9391; absolute amplitude setpoint). Only the
# input string + gain sign differ by mode; the Output.Z / -10..150 rail is
# mode-independent (SetupSlowLoop :9595-9596).
_Z_OUT = Z_OUT         # the Height alias (the rig rejects "Z%Output")


def _publish_scan_row(row) -> None:
    """Publish the live raster row on the hub so the rail-dwell safety guard's
    ~10 Hz Z-drive poll can tag its samples by line (host/continuous._rail_poll).
    Best-effort + lazy-imported (hub does not import hw_scan): a hiccup here must
    never touch the scan. Inert unless a continuous scan has the guard armed."""
    try:
        from .hub import hub
        if row is not None:
            hub.host._scan_row = int(row)
    except Exception:  # noqa: BLE001, S110 - a telemetry stamp must never touch the scan
        pass


def _finite(v, default: float = 0.0) -> float:
    """A calibration/gain value coerced to a finite float (NaN / Inf / None / non-
    numeric -> default). Off-instrument, missing gains parse to NaN -> default, so
    simulation behaviour is unchanged."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _flatten1(y: np.ndarray) -> np.ndarray:
    """Per-line 1st-order least-squares line subtraction -- mirrors Asylum's default
    topography "Flatten 1" (ScanMaster.ipf `switch(DoPlaneFit)` case 2): fit y=a*x+b
    over the row's own pixels (x = 0..N-1) and subtract. DISPLAY ONLY -- this greys the
    image; the RAW height persisted/written against flows via the capture, untouched.
    Degenerate rows (N<2, <2 finite samples, zero-variance x) pass through unchanged, so
    it never divides by zero. See docs/ASYLUM_REALTIME_FLATTEN.md."""
    a = np.asarray(y, dtype=np.float64)
    n = a.size
    if n < 2:
        return a
    x = np.arange(n, dtype=np.float64)
    ok = np.isfinite(a)
    if int(ok.sum()) < 2 or np.ptp(x[ok]) == 0.0:
        return a
    slope, intercept = np.polyfit(x[ok], a[ok], 1)
    return a - (slope * x + intercept)


def _rows_to_lines(z, e, row_start: int, n_rows: int, config, cal: dict,
                   trace_retrace: bool = False, v=None, err_chan: str = _ERR) -> list[dict]:
    """Project a block of capture (n_rows whole lines from global row `row_start`) into
    per-row {"type":"line"} messages. ZSensor volts -> height nm via the Z LVDT cal
    (Height[m] = -(ZSensor - offset) * sens).

    trace_retrace: each captured line is 2*pixels points -- forward (trace) then reverse
    (retrace). We split them and send `height`/`error` (trace) alongside `retrace`/
    `error_retrace` (the reverse sweep, flipped back so it overlays the trace spatially).
    Single-sweep (snake): odd rows are captured reversed, so un-snake by parity, and
    retrace mirrors the trace."""
    pixels = int(config.pixels)
    lines = int(config.lines) or pixels
    ppl = 2 * pixels if trace_retrace else pixels        # points per captured line
    z_sens = float(cal.get("z_lvdt_sens", 1.0))
    z_off = float(cal.get("z_lvdt_offset", 0.0))
    zb = np.asarray(z, dtype=np.float64)
    eb = np.asarray(e, dtype=np.float64)
    need = ppl * n_rows
    if len(zb) < need:
        zb = np.pad(zb, (0, need - len(zb)))
    if len(eb) < need:
        eb = np.pad(eb, (0, need - len(eb)))
    # MEASURED tip bias (UserIn2). Optional -- absent when the channel is not
    # captured (or was not ADC-routable and the fail-safe arm skipped it); the line then
    # simply carries no bias field, exactly as before.
    vb = None
    if v is not None:
        vb = np.asarray(v, dtype=np.float64)
        if len(vb) < need:
            vb = np.pad(vb, (0, need - len(vb)))

    def h(vals):
        # ZSensor volts -> physical height (nm), unrounded (flattened before display rounding)
        return -(np.asarray(vals, dtype=np.float64) - z_off) * z_sens * 1e9

    msgs: list[dict] = []
    for r in range(n_rows):
        row = row_start + r
        z_row = zb[r * ppl:(r + 1) * ppl]
        e_row = eb[r * ppl:(r + 1) * ppl]
        v_row = vb[r * ppl:(r + 1) * ppl] if vb is not None else None
        if trace_retrace:
            z_tr, z_re = z_row[:pixels], z_row[pixels:][::-1]   # retrace flipped to align
            e_tr, e_re = e_row[:pixels], e_row[pixels:][::-1]
            if v_row is not None:
                v_tr, v_re = v_row[:pixels], v_row[pixels:][::-1]
        else:
            if row % 2 == 1:                      # snake: odd rows captured reversed
                z_row, e_row = z_row[::-1], e_row[::-1]
                if v_row is not None:
                    v_row = v_row[::-1]
            z_tr = z_re = z_row
            e_tr = e_re = e_row
            if v_row is not None:
                v_tr = v_re = v_row
        slow_idx = row if config.direction == "up" else (lines - 1 - row)
        # DISPLAY-ONLY per-line flatten: mirror Asylum's default topography "Flatten 1"
        # (1st-order line subtraction) so the greyscale image matches the instrument.
        # Trace & retrace fit INDEPENDENTLY over their own pixels. Only the ZSensor-
        # derived height/zsensor/retrace display fields are flattened; deflection (error)
        # and bias (tipv) stay RAW, and the persisted capture (ZSensor volts) is untouched.
        h_tr = [round(float(x), 3) for x in _flatten1(h(z_tr))]
        h_re = [round(float(x), 3) for x in _flatten1(h(z_re))]
        msg = {
            "type": "line", "row": row, "slow_idx": slow_idx,
            "rows": lines, "pixels": pixels, "size_um": config.size_um,
            "direction": config.direction, "angle_deg": config.angle_deg,
            "errName": err_chan, "trace_retrace": bool(trace_retrace),
            "height": h_tr, "retrace": h_re,
            "error": [round(float(x), 4) for x in e_tr],
            "error_retrace": [round(float(x), 4) for x in e_re],
            "zsensor": h_tr,
            "slow_um": (row / max(1, lines)) * config.size_um,
        }
        if v_row is not None:
            msg["tipv"] = [round(float(x), 4) for x in v_tr]
            msg["tipv_retrace"] = [round(float(x), 4) for x in v_re]
        msgs.append(msg)
    return msgs


async def _pre_scan_xy_transit(arc, config, cal: dict, broadcast, *, engaged: bool,
                               x_pg: float, x_ig: float, y_pg: float, y_ig: float,
                               trace_retrace: bool, notes=NO_NOTES,
                               should_abort: Callable[[], bool] | None = None) -> None:
    """XY_SLEW_SPEC (docs/XY_SLEW_SPEC.md): move to the raster's first point at
    MOVE_SPEED_UM_S with the tip withdrawn, BEFORE this scan's own loop arm/engage.
    Runs off-loop (`xy_transit` blocks: arc I/O + settle waits).

    Re-approach only in the one case the spec calls out: the tip was engaged on
    entry AND this scan will NOT itself engage -- otherwise the scan's own engage
    (below, inside the chained `driver()`) restores contact, and a
    second re-approach here would just be redundant (and outrun by it)."""
    pixels = int(config.pixels)
    lines = scan_line_count(config)
    xs, ys = raster_path_um(pixels, lines, config.size_um,
                            x_offset_um=config.x_offset_um, y_offset_um=config.y_offset_um,
                            angle_deg=config.angle_deg, direction=config.direction,
                            trace_retrace=trace_retrace)
    reapproach = scan_reapproaches(engaged, config)
    log_lines: list[str] = []
    moved = await asyncio.to_thread(
        xy_transit, arc, cal, float(xs[0]), float(ys[0]),
        engaged=engaged, reapproach=reapproach,
        setpoint=(float(config.setpoint) if reapproach else None),
        mode=str(config.mode), x_pgain=x_pg, x_igain=x_ig, y_pgain=y_pg, y_igain=y_ig,
        should_abort=should_abort, log=log_lines.append)
    notes.outcome("move", lambda: {"how": "transit", "to_um": (float(xs[0]), float(ys[0])),
                                   **moved})
    if reapproach:
        notes.outcome("engage", lambda: {"setpoint_v": float(config.setpoint),
                                         "reapproach": True})
    for line in log_lines:
        await broadcast({"type": "log", "t": time.time(), "msg": line})


async def _aborted_before_raster(broadcast, notes, pixels: int) -> dict:
    """The scan was aborted before its raster fired: report an aborted, empty frame."""
    notes.abort("scan aborted before the raster")
    await broadcast({"type": "scan_done", "aborted": True})
    return {"lines": 0, "pixels": pixels, "aborted": True, "error": None,
            "capture": {}, "line_indices": []}


def _stop_if_cancelled(cancelled) -> None:
    """The service cancelled the scan pump: stop here, as a cancelled coroutine stops at its
    next await. The Z arm and the raster fire are never sent after this."""
    if cancelled is not None and cancelled():
        raise asyncio.CancelledError


async def run_hw_scan_chained(arc, config, cal: dict,
                              broadcast: Callable[[dict], Awaitable],
                              *, interp: int | None = None,
                              z_setpoint: float | None = None,
                              should_abort: Callable[[], bool] | None = None,
                              engaged: bool = False, notes=NO_NOTES,
                              cancelled: Callable[[], bool] | None = None,
                              igor_aborted: Callable[[], bool] | None = None) -> dict:
    """Drive a raster as a callback-CHAINED scan: send the full XY setpoint waves once,
    then SCANCHAIN, and let Igor advance line-to-line on the capture-fill callback (no
    per-line socket round-trip -- the vendor's OutWave-callback idea). Each line streams
    back as a CAPTURE("L<n>") pair; we project and broadcast it, ending on SCANDONE.
    The whole link conversation runs in one worker thread. `engaged` (default False) is
    the caller's own knowledge of whether the tip is on the surface BEFORE this call -- see `_pre_scan_xy_transit` (XY_SLEW_SPEC).
    `notes` (service/step_notes.py, optional) journals the outcome of each step.
    `igor_aborted` (optional): True once an abort started in Igor stopped this scan; its
    SCANDONE -1 then ends the frame as aborted."""
    pixels = int(config.pixels)
    lines = scan_line_count(config)
    trace_retrace = bool(getattr(config, "trace_retrace", True))
    ppl = scan_points_per_line(pixels, trace_retrace)
    _, err_chan, _ = scan_capture_channels(config.mode)   # contact->Deflection, AC->Amplitude

    if interp is None:
        interp = scan_interpolation(pixels, config.line_rate_hz)
    decim = scan_decimation(interp)

    if z_setpoint is None and bool(getattr(config, "engage", False)):
        z_setpoint = float(config.setpoint)
        z_in, z_sign = mode_loop_params(config.mode)
        await broadcast({"type": "log", "t": time.time(),
                         "msg": f"Imaging engage: Z servo ({config.mode}) at setpoint "
                                f"{z_setpoint:g} V, feedback {z_in}->{_Z_OUT}"
                                + (" (AC: gain sign negated)"
                                   if z_sign < 0 else "")})

    # Optional Z-servo gain override (startScan pgain/igain): PANEL-gain convention,
    # the SAME units setZGain/startApproach take -- see ScanConfig.pgain/igain and
    # hw_approach._PSCALE/_ISCALE/mode_loop_params. The mode sign is baked in HERE,
    # once, at Pass-build time -- so `driver()`'s ir_SetPISLoop call below can use
    # p.z_pgain/p.z_igain AS-IS, with no second multiplication. For the omitted case
    # this reproduces the OLD `p.z_igain * z_sign` done at that call site exactly
    # (1200.0 raw * mode sign), so it stays byte-identical when pgain/igain are
    # not given; the override applies the same sign to the panel-scaled value.
    z_gain_kwargs: dict[str, float] = {}
    if z_setpoint is not None:
        _, z_sign = mode_loop_params(config.mode)
        z_gain_kwargs["z_pgain"] = (
            float(config.pgain) * _PSCALE if config.pgain is not None else 0.0) * z_sign
        z_gain_kwargs["z_igain"] = (
            float(config.igain) * _ISCALE if config.igain is not None else 1200.0) * z_sign
        if config.pgain is not None or config.igain is not None:
            await broadcast({"type": "log", "t": time.time(),
                             "msg": f"Z loop gain override ({config.mode}): "
                                    f"pgain={z_gain_kwargs['z_pgain']:g} "
                                    f"igain={z_gain_kwargs['z_igain']:g}  [chained]"})

    x_pg, x_ig = _finite(cal.get("x_pgain")), _finite(cal.get("x_igain"))
    y_pg, y_ig = _finite(cal.get("y_pgain")), _finite(cal.get("y_igain"))
    await broadcast({"type": "log", "t": time.time(),
                     "msg": f"XY loop gains from calibration: X(P={x_pg:g} I={x_ig:g}) "
                            f"Y(P={y_pg:g} I={y_ig:g})  [chained]"})
    if x_ig == 0.0 or y_ig == 0.0:
        await broadcast({"type": "log", "t": time.time(),
                         "msg": "WARNING: an XY loop I-gain is 0/absent in STARTUP."})

    # XY_SLEW_SPEC: move to the raster's first point at a controlled 5 um/s with the
    # tip withdrawn, BEFORE the driver's own loop arm below. See _pre_scan_xy_transit.
    # An abort during the move (Igor's own abort panel included) ends the scan here: no
    # loop arm, no Z engage, no raster.
    try:
        await _pre_scan_xy_transit(arc, config, cal, broadcast, engaged=engaged,
                                   x_pg=x_pg, x_ig=x_ig, y_pg=y_pg, y_ig=y_ig,
                                   trace_retrace=trace_retrace, notes=notes,
                                   should_abort=should_abort)
    except TransitAborted:
        return await _aborted_before_raster(broadcast, notes, pixels)
    _stop_if_cancelled(cancelled)
    if should_abort is not None and should_abort():
        return await _aborted_before_raster(broadcast, notes, pixels)

    # reuse raster_pass to build the full setpoint waves (um -> LVDT volts, centered at
    # the LVDT offset) and hold the loop-gain/z fields.
    p = raster_pass(pixels, lines, config.size_um, cal, interp,
                    x_offset_um=config.x_offset_um, y_offset_um=config.y_offset_um,
                    angle_deg=config.angle_deg, direction=config.direction,
                    trace_retrace=trace_retrace, capture=list(scan_capture_channels(config.mode)),
                    z_setpoint=z_setpoint, x_pgain=x_pg, x_igain=x_ig,
                    y_pgain=y_pg, y_igain=y_ig, **z_gain_kwargs)

    q: "_queue.Queue" = _queue.Queue()

    def aborted_now() -> bool:
        return should_abort is not None and should_abort()

    def driver():
        # servo loops (XY tracking at the tuned gains; Z only if engaging)
        arc.ir_SetPISLoop(0, "Always,Never", channels.X_SENSOR, float(p.x_v[0]),
                          p.x_pgain, p.x_igain, 0.0, channels.OUTPUT_X, *LOOP_LIMITS_V)
        arc.ir_SetPISLoop(1, "Always,Never", channels.Y_SENSOR, float(p.y_v[0]),
                          p.y_pgain, p.y_igain, 0.0, channels.OUTPUT_Y, *LOOP_LIMITS_V)
        if aborted_now():                            # stopped while arming: no engage
            q.put(("done", True))
            return
        if p.z_setpoint is not None:
            # SAFETY-CRITICAL Z arm: mode-aware channel + gain sign (see Z_LOOP /
            # mode_loop_params). Contact -> `Deflection` + positive gain; AC ->
            # `Amplitude` + NEGATED gain (the amplitude error inverts near the
            # surface, so a positive gain would drive the piezo the wrong way).
            # p.z_pgain/p.z_igain already carry that sign -- baked in once, where
            # the Pass was built (the z_gain_kwargs block above; the default z_igain
            # is 1200.0*sign, matching the OLD `p.z_igain * z_sign` done here, and an
            # startScan pgain/igain override is scaled + signed the SAME way, via
            # hw_approach._PSCALE/_ISCALE/mode_loop_params -- the setZGain
            # convention). So they are used AS-IS here, with no second multiply. The
            # Z loop OUTPUT: arc.z_out_channel is `Height` on both platforms. Arming
            # loop 2 with the wrong output drives nothing and the scan does not track
            # the surface (crash). RIG-UNVERIFIED on the MFP-3D.
            z_in, _ = mode_loop_params(config.mode)
            arc.ir_SetPISLoop(Z_LOOP, "Always,Never", z_in, float(p.z_setpoint),
                              p.z_pgain, p.z_igain, 0.0, arc.z_out_channel, *LOOP_LIMITS_V)
            notes.outcome("engage", lambda: {"setpoint_v": float(p.z_setpoint), "on": z_in,
                                             "out": arc.z_out_channel})
        xname = arc._load_wave(p.x_v)                # WAVE, full frame, once
        yname = arc._load_wave(p.y_v)
        # Live tip marker + rail-dwell safety are BOTH fed from the always-on TELEM
        # poll now (installed once as arc.router.on_telem by the hardware backend --
        # service/host/telemetry.install_hw_telemetry), NOT armed per-scan here.
        # So this driver no longer touches on_telem; it would clobber that single
        # always-on consumer. The interleaved TELEM frames route to that consumer,
        # which broadcasts tip_pos and feeds the guard while this scan runs.
        # Register the scan pass under a fresh UID and fire the chain: every per-line
        # CAPTURE + the terminal SCANDONE (tagged with this uid by Igor) route to this
        # waiter, so a concurrent td_ReadValue on the shared link can never steal them.
        timeout = float(getattr(arc, "timeout", 30.0))
        if aborted_now():                            # stopped while loading: no raster
            q.put(("done", True))
            return
        w = arc.router.open_waiter()
        # Pass the mode-aware ERROR channel name so Igor arms the bank-0 in-wave PAIR
        # as (ZSensor + err_chan): the FIRST channel keeps the ZSensor advance callback;
        # only the SECOND (error) channel name changes (Deflection<->Amplitude).
        arc.link.send(P.cmd_scanchain(w.uid, xname, yname, ppl, lines, interp, decim,
                                      err_chan))

        pending: dict = {}
        aborted = False
        try:
            while True:
                try:
                    verb, rest = w.get(timeout)
                except Exception as exc:                  # noqa: BLE001 (closed / timeout)
                    q.put(("err", f"chained scan read failed: {exc}"))
                    return
                if verb == P.CAPTURE:
                    name, _, tail = rest.partition(" ")
                    chan, _, body = tail.partition(" ")
                    try:
                        idx = int(name[1:])
                    except ValueError:
                        continue
                    vals = [float(v) for v in body.split(";") if v != ""]
                    # ACCUMULATE per (line, channel): a line's capture (ppl points) is
                    # small (<< kCapChunkPoints), so normally each channel arrives as ONE
                    # frame and this reaches ppl at once. If a channel were ever chunked,
                    # its frames append in arrival order and the line completes only once
                    # BOTH driving channels reach ppl -- never on a partial chunk.
                    d = pending.setdefault(idx, {})
                    d.setdefault(chan, []).extend(vals)
                    if len(d.get(_HEIGHT, ())) >= ppl and len(d.get(err_chan, ())) >= ppl:
                        z = np.array(d[_HEIGHT], dtype=np.float64)
                        e = np.array(d[err_chan], dtype=np.float64)
                        # _BIAS (UserIn2) is streamed BEFORE the pair when armed, so it
                        # is fully buffered here; None when the fail-safe arm skipped it.
                        v = (np.array(d[_BIAS], dtype=np.float64)
                             if _BIAS in d else None)
                        q.put(("line", idx, z, e, v))
                        pending.pop(idx, None)
                        if should_abort is not None and should_abort() and not aborted:
                            aborted = True
                            # out-of-band: COM first, TCP fallback -- and it now
                            # RETRACTS (AFM_SafePark, docs/ABORT_SEQUENCE.md). The
                            # deferred safe_park that follows is idempotent, so this
                            # is not a double-retract race.
                            # no hub here: the ARC_ABORT style only needs `arc`
                            request_park(hub=None, reason="scan aborted",
                                         how=PARK_ARC_ABORT, arc=arc)
                elif verb in (P.SCANDONE, P.ABORTED):
                    # Flush any straggler line (a short final capture that never reached
                    # ppl) that still has both driving channels, so it is not lost.
                    for idx in sorted(pending):
                        d = pending[idx]
                        if _HEIGHT in d and err_chan in d:
                            v = (np.array(d[_BIAS], dtype=np.float64)
                                 if _BIAS in d else None)
                            q.put(("line", idx, np.array(d[_HEIGHT], dtype=np.float64),
                                   np.array(d[err_chan], dtype=np.float64), v))
                    pending.clear()
                    q.put(("done", aborted or (igor_aborted is not None and igor_aborted())))
                    return
                elif verb == P.ERR:
                    q.put(("err", rest.strip()))
                    return
                # anything else: keep reading until SCANDONE
        finally:
            arc.router.release(w.uid)

    # The thread must be built inside `untraced()`: to_thread copies the context, and the
    # reader thread outlives this request.
    with P.untraced():
        task = asyncio.ensure_future(asyncio.to_thread(driver))
    sent, aborted = 0, False
    err_msg: str | None = None          # a chained `err` (router/bridge fault) -> summary["error"]
    frames: dict[int, tuple] = {}       # line idx -> (raw ZSensor, Deflection, UserIn2|None)
    while True:
        try:
            item = q.get_nowait()
        except _queue.Empty:
            if task.done() and q.empty():
                break
            _stop_if_cancelled(cancelled)
            # real time: the queue is filled by a real to_thread driver
            await asyncio.sleep(0.005)
            continue
        if item[0] == "line":
            _, idx, z, e, v = item
            vv = None if v is None else np.asarray(v, dtype=np.float64)
            frames[idx] = (np.asarray(z, dtype=np.float64),
                           np.asarray(e, dtype=np.float64), vv)
            _publish_scan_row(idx)              # rail-dwell guard poll tags by row
            for msg in _rows_to_lines(z, e, idx, 1, config, cal, trace_retrace, v=vv,
                                      err_chan=err_chan):
                await broadcast(msg)
                sent += 1
        elif item[0] == "done":
            aborted = bool(item[1])
            break
        elif item[0] == "err":
            await broadcast({"type": "log", "t": time.time(),
                             "msg": f"Chained scan error: {item[1]}"})
            err_msg = str(item[1])       # a REAL fault (router/bridge), not an operator abort
            aborted = True
            break
    await task
    notes.outcome("pass", fire=P.SCANCHAIN, lines_planned=lines, lines_delivered=len(frames),
                  aborted=aborted, error=err_msg)
    if aborted:
        notes.abort(err_msg or "scan aborted")
    # assemble the RAW measured capture in line order (row-major, ppl/line) so the
    # A4 recorder persists the chained frame as for any scan.
    capture: dict = {}
    if frames:
        order = sorted(frames)
        capture = {_HEIGHT: np.concatenate([frames[i][0] for i in order]),
                   err_chan: np.concatenate([frames[i][1] for i in order])}
        # Include the MEASURED tip bias only if EVERY line carried it (armed cleanly),
        # so the persisted channel is a full, row-major frame or absent -- never ragged.
        if all(frames[i][2] is not None for i in order):
            capture[_BIAS] = np.concatenate([frames[i][2] for i in order])
    await broadcast({"type": "scan_done", "aborted": aborted})
    # the REAL line indices delivered (sparse/short on a partial scan); the cache
    # places each captured row at its OWN index, not its ordinal in the concat.
    # `error` (when set) marks a chained fault the pump treats as a FRAME FAILURE
    # (blocker #2) -- distinct from an operator-requested abort (aborted, no error).
    return {"lines": sent, "pixels": pixels, "aborted": aborted, "error": err_msg,
            "capture": capture, "line_indices": sorted(frames)}
