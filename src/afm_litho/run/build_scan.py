"""Build the run of a scan from the same inputs today's scan starts with.

Pure: no I/O, no hub, no wire. Every number and rule comes from the scan's own functions
(hw_scan, scan_cmds, hardware.xy_transit), so the run and the scan cannot disagree about
them. Not imported by `run/__init__` because these modules import the service.
"""

from __future__ import annotations

from ..hardware import channels
from ..hardware import xy_transit as transit
from ..hardware.hardware_arc import WAVE_CHUNK_POINTS
from ..service import hw_scan
from ..service.host.scan_cmds import mode_differs, mode_settle_s, norm_mode, tracked_mode
from ..hardware.hw_approach import Z_LOOP, Z_RETRACT_OUTMIN, mode_loop_params
from .model import (
    Capture,
    Check,
    End,
    Engage,
    Execution,
    Move,
    Recipe,
    Run,
    Scan,
    ScanFrame,
    SetMode,
    Setpoint,
    Withdraw,
)

# What the transit waits after a move: the larger of a margin and a fraction of the move time.
TRANSIT_HOW = (f"transit, then settle max({transit.RAMP_SETTLE_MARGIN_S:g} s,"
               f" {transit.RAMP_SETTLE_FRACTION:g} x move time)")


def build_scan_run(config, *, platform: str, execution: Execution, imaging_mode,
                   engaged: bool, mode_gate_closed: bool | None = None,
                   drift_offset: bool = False, frame_source: str = "scan request",
                   ends_with_withdraw: bool = True) -> Run:
    """The run of one scan.

    `config` is the scan's `ScanConfig` (offsets already include any drift compensation;
    `drift_offset` says so). `platform` is "cypher" or "mfp3d".

    `imaging_mode` is the hub's tracked software mode as the scan reads it (None or "" is
    unknown; the scan then reads the instrument's own mode and switches only if it differs,
    so a setmode step is planned with that condition).

    `engaged` is the hub's tip state BEFORE the scan starts, which is what the transit sees
    once the scan has marked itself engaged (`engaged or config.engage`). The mode gate is
    separate: `mode_gate_closed` is `engaged_or_scanning(hub)[0]` (a tip down, a raster
    running, continuous imaging, a write running); default: the same as `engaged`.

    `ends_with_withdraw`: a scan lifts the tip when it ends on its own (the routine withdraw a
    write ends with). False for a continuous frame, whose loop parks when it ends. A Stop or an
    abort replaces the withdraw with its park at run time.
    """
    mode = norm_mode(config.mode)
    pixels = int(config.pixels)
    lines = hw_scan.scan_line_count(config)
    trace_retrace = bool(config.trace_retrace)
    setpoint = Setpoint("absolute", float(config.setpoint))
    engage_here = bool(config.engage)
    transit_engaged = bool(engaged or engage_here)
    gate_closed = engaged if mode_gate_closed is None else bool(mode_gate_closed)
    current = tracked_mode(imaging_mode)

    def engage_step(loop: int, on: str, sign: float, when: str) -> Engage:
        return Engage(setpoint=setpoint, loop=loop, on=on,
                      out=channels.Z_OUT,
                      limits_v=channels.LOOP_OUTPUT_LIMITS_V, sign=int(sign), when=when)

    body: list = [Check(items=("no-latched-fault", "frame-inside-xy-range"))]
    if mode_differs(current, mode) and not gate_closed:
        when = ("only if withdrawn and the mode differs" if current
                else "only if the instrument's own mode read differs")
        body.append(SetMode(mode=mode, when=when, settle_s=mode_settle_s()))
    body.append(Move(how=TRANSIT_HOW, to="first point of the raster",
                     withdraw_first="verified" if transit_engaged else "verified-if-engaged",
                     speed_um_s=transit.MOVE_SPEED_UM_S))
    if hw_scan.scan_reapproaches(transit_engaged, config):
        z_in, z_sign = transit.mode_loop_params(mode)
        body.append(engage_step(transit.Z_LOOP, z_in, z_sign,
                                "the transit's own re-engage after the move: the tip was"
                                " engaged and the scan does not engage"))
    if engage_here:
        z_in, z_sign = mode_loop_params(mode)
        body.append(engage_step(Z_LOOP, z_in, z_sign, "only when engage is requested"))
    interp = hw_scan.scan_interpolation(pixels, config.line_rate_hz)
    body.append(Scan(lines=lines,
                     points_per_line=hw_scan.scan_points_per_line(pixels, trace_retrace),
                     interpolation=interp, decimation=hw_scan.scan_decimation(interp),
                     error_channel=hw_scan.scan_capture_channels(mode)[1],
                     xy_loops=hw_scan.XY_LOOPS,
                     xy_outputs=(channels.OUTPUT_X, channels.OUTPUT_Y),
                     limits_v=hw_scan.LOOP_LIMITS_V, wave_line_points=WAVE_CHUNK_POINTS))
    if ends_with_withdraw:
        body.append(Withdraw("verified", Z_RETRACT_OUTMIN, why="end of scan"))
    else:
        body.append(End(engaged=transit_engaged))

    frame = ScanFrame(source=frame_source, size_um=float(config.size_um), lines=lines,
                      pixels=pixels, angle_deg=float(config.angle_deg),
                      offset_um=(float(config.x_offset_um), float(config.y_offset_um)),
                      drift_offset=drift_offset, direction=str(config.direction),
                      line_rate_hz=float(config.line_rate_hz),
                      trace_and_retrace=trace_retrace)
    recipe = Recipe(kind="scan", platform=platform, mode=mode, setpoint=setpoint,
                    capture=Capture(channels=hw_scan.scan_capture_channels(mode)),
                    script_gone="park", body=tuple(body), frame=frame)
    return Run(recipe=recipe, execution=execution)
