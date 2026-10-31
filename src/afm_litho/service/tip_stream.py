"""Host-side conversion + WS framing for the live tip-position stream.

The Igor bridge's always-on poll pushes `TELEM <x> <y> <zdrive> <defl> <sum>`
frames carrying RAW XSensor/YSensor volts (plus the safety channels) at ~4-10 Hz.
The host consumer (host/telemetry._hw_telemetry_frame) converts the tip volts
to lab microns via `hub.hw_cal`, read live per frame (refreshed at connect, scan
start and approach by service/xy_cal.py, since the Cypher rescales the XY LVDT
sensitivities on every motor move) -- the SAME LVDT
convention the pass path uses (`Volts = Position[m]/Sensitivity + Offset`,
inverted to `Position[m] = (Volts - Offset)*Sensitivity`) -- and broadcasts a
dedicated `tip_pos{x_um,y_um}` message the cockpit draws as a live marker/trail.

The message type is DELIBERATELY separate from `litho_progress`: the tip marker
is a low-resolution real-time marker and must not gate or replace the per-object
end-of-object capture (the high-resolution written record). The two coexist -- the
cockpit draws the live marker from `tip_pos` and the accumulated high-res written
paths from `litho_object_done`.
"""

from __future__ import annotations


def tippos_to_um(xv: float, yv: float, cal: dict) -> tuple[float, float]:
    """Convert raw XSensor/YSensor volts to lab microns via the LVDT calibration.

    Inverse of the pass path's forward map (raster_pass / lithography_run:
    `volts = um*1e-6/sens + offset`), so `um = (volts - offset)*sens*1e6`. There
    is no sign flip here (unlike Z height in hw_scan, which negates because height
    grows as Z retracts); X/Y position maps straight through. A missing cal key
    falls back to identity-ish defaults so an off-instrument call never divides by
    a phantom or throws."""
    x_sens = float(cal.get("x_lvdt_sens", 1.0))
    x_off = float(cal.get("x_lvdt_offset", 0.0))
    y_sens = float(cal.get("y_lvdt_sens", 1.0))
    y_off = float(cal.get("y_lvdt_offset", 0.0))
    x_um = (float(xv) - x_off) * x_sens * 1e6
    y_um = (float(yv) - y_off) * y_sens * 1e6
    return x_um, y_um


def tip_pos_msg(xv: float, yv: float, cal: dict) -> dict:
    """The WS message the cockpit draws: `tip_pos{x_um,y_um}` (rounded to nm)."""
    x_um, y_um = tippos_to_um(xv, yv, cal)
    return {"type": "tip_pos", "x_um": round(x_um, 3), "y_um": round(y_um, 3)}
