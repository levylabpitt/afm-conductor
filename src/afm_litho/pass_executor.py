"""A compiled raster pass and its tip path.

A *pass* is a trajectory (XY setpoint waveforms in LVDT volts) plus what to capture.
`raster_pass` compiles a scan into one; `service/hw_scan.run_hw_scan_chained` arms it
and Igor plays it line by line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hardware import channels
from .hardware.channels import Z_OUT
from .litho.frame import rotate

_Z_DAC = Z_OUT


@dataclass
class Pass:
    """A compiled pass: XY setpoint waveforms (LVDT volts) + how to capture it."""
    x_v: np.ndarray                       # X setpoint per commanded point
    y_v: np.ndarray                       # Y setpoint
    interp: int                           # signed ticks/point (negative == linear)
    capture: list[str] = field(default_factory=lambda: [_Z_DAC, channels.INPUT_FAST])
    bias_v: np.ndarray | None = None      # optional bias out-wave (litho); None = none
    z_setpoint: float | None = None       # engage the Z servo at this setpoint; None = leave Z
    z_pgain: float = 0.0
    z_igain: float = 1200.0
    line_len: int | None = None           # chunk on this many points (raster: pixels/line)
    # XY closed-loop scanner gains, the instrument's own -- the hardware scan path
    # derives them from calibration the vendor's way (10^GV("XIGain")*sign(GV(
    # "XPiezoSens")); see hw_scan and docs/asylum_PIS_gains.md), so a real scan drives
    # the loops with real gains and never an invented one (a zero-gain loop does not
    # track the commanded setpoint; PASS_MODEL_PLAN B4). Default 0 == "caller did not
    # supply"; a gain-0 pass is legitimate open-loop command-and-measure (retracted
    # dry run), not accurate positioning.
    x_pgain: float = 0.0
    x_igain: float = 0.0
    y_pgain: float = 0.0
    y_igain: float = 0.0


# ---------------------------------------------------------- raster as a pass --

def raster_path_um(pixels: int, lines: int, size_um: float, *,
                   x_offset_um: float = 0.0, y_offset_um: float = 0.0,
                   angle_deg: float = 0.0, direction: str = "up",
                   snake: bool = True,
                   trace_retrace: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The raster tip path in microns (litho/frame.py rotation, same fast/slow layout
    as passes.scan_line).

    trace_retrace=True scans each line BOTH ways: forward (trace) then reverse
    (retrace) at the same slow position, so each line is 2*pixels points and the two
    sweeps can be compared (the vendor's Trace/Retrace). trace_retrace=False keeps the
    boustrophedon single-sweep snake (odd lines reversed so the tip never flies back)."""
    ux, uy = rotate(1.0, 0.0, angle_deg)          # fast unit vector
    vx, vy = rotate(0.0, 1.0, angle_deg)          # slow unit vector
    fast = np.linspace(-size_um / 2, size_um / 2, pixels)
    xs_all, ys_all = [], []
    for row in range(lines):
        slow_idx = row if direction == "up" else (lines - 1 - row)
        slow = -size_um / 2 + size_um * slow_idx / max(lines - 1, 1)
        if trace_retrace:
            f = np.concatenate([fast, fast[::-1]])       # forward then reverse, same slow
        else:
            f = fast[::-1] if (snake and row % 2 == 1) else fast
        xs_all.append(x_offset_um + ux * f + vx * slow)
        ys_all.append(y_offset_um + uy * f + vy * slow)
    return np.concatenate(xs_all), np.concatenate(ys_all)


def raster_pass(pixels: int, lines: int, size_um: float, cal: dict, interp: int,
                *, x_offset_um: float = 0.0, y_offset_um: float = 0.0,
                angle_deg: float = 0.0, direction: str = "up", snake: bool = True,
                trace_retrace: bool = False,
                capture: list[str] | None = None,
                z_setpoint: float | None = None,
                z_pgain: float = 0.0, z_igain: float = 1200.0,
                x_pgain: float = 0.0, x_igain: float = 0.0,
                y_pgain: float = 0.0, y_igain: float = 0.0) -> Pass:
    """Compile a raster scan into a Pass: the snake tip path converted to LVDT
    setpoint volts (Volts = Position[m]/Sensitivity + Offset), with the capture
    channels and `line_len = pixels` so chunk seams fall on line boundaries.
    `cal` carries x/y_lvdt_sens and x/y_lvdt_offset. The XY loop gains are the
    instrument's own, derived from calibration by the caller (10^XIGain *
    sign(XPiezoSens)); they are NOT invented here (closed loops can oscillate --
    PASS_MODEL_PLAN)."""
    x_um, y_um = raster_path_um(pixels, lines, size_um, x_offset_um=x_offset_um,
                                y_offset_um=y_offset_um, angle_deg=angle_deg,
                                direction=direction, snake=snake,
                                trace_retrace=trace_retrace)
    # Center at the LVDT offset (position 0 == scanner centre, verified on hardware:
    # at Asylum offset 0,0 the sensors read the LVDT offsets). Sens does the um<->volt
    # scaling; offset places position 0 at the scanner centre.
    x_v = x_um * 1e-6 / cal["x_lvdt_sens"] + cal["x_lvdt_offset"]
    y_v = y_um * 1e-6 / cal["y_lvdt_sens"] + cal["y_lvdt_offset"]
    line_len = 2 * pixels if trace_retrace else pixels   # a full line incl. retrace
    return Pass(x_v=x_v, y_v=y_v, interp=interp, line_len=line_len,
                capture=capture or [_Z_DAC, channels.INPUT_FAST], z_setpoint=z_setpoint,
                z_pgain=z_pgain, z_igain=z_igain,
                x_pgain=x_pgain, x_igain=x_igain, y_pgain=y_pgain, y_igain=y_igain)

