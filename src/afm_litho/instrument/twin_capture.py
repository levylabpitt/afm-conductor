"""Seed the twin from one of the program's own hardware scan captures (a `.npz` of raw channels).

The capture holds ZSensor, Amplitude and UserIn2 in volts, each `lines * 2 * pixels` long:
every line is the trace then the retrace, as hw_scan.py splits it. The file has no geometry,
so a square image is assumed and `size_um` is the caller's (default 30 um).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..hardware import channels

# Cypher Z LVDT calibration in m/V (Instrument default, twin/physics.py); the capture file does
# not carry the rig's own.
Z_LVDT_SENS_M_PER_V = 1.3036e-6
Z_LVDT_OFFSET_V = 0.0


def split_trace_retrace(samples, pixels: int, lines: int | None = None):
    """(trace, retrace) as lines x pixels; the retrace is flipped to align with the trace."""
    flat = np.asarray(samples, dtype=np.float64)
    lines = lines or flat.size // (2 * pixels)
    grid = flat[:lines * 2 * pixels].reshape(lines, 2, pixels)
    return grid[:, 0, :], grid[:, 1, ::-1]


def load_capture(path, size_um: float = 30.0, offset_um: tuple[float, float] = (0.0, 0.0),
                 z_sens_m_per_v: float = Z_LVDT_SENS_M_PER_V,
                 z_offset_v: float = Z_LVDT_OFFSET_V) -> dict:
    """The twin's sample arrays from a capture: height in nm (trace, retrace) and amplitude in V.

    Pixels per line come from the size (square image: size = 2 * pixels^2). ZSensor volts turn
    into nm as hw_scan does: -(V - offset) * sens * 1e9 (the rig's ZSensor is sign-inverted).
    """
    with np.load(Path(path)) as z:
        zs, amp = np.asarray(z[channels.Z_SENSOR]), np.asarray(z[channels.AMPLITUDE])
    pixels = round((zs.size / 2) ** 0.5)
    if 2 * pixels * pixels != zs.size:
        raise ValueError(f"{zs.size} points is not a square trace+retrace capture")

    def to_nm(volts):
        return -(volts - z_offset_v) * z_sens_m_per_v * 1e9
    z_tr, z_re = split_trace_retrace(zs, pixels)
    a_tr, a_re = split_trace_retrace(amp, pixels)
    return {"height": to_nm(z_tr), "retrace": to_nm(z_re),
            "amplitude": a_tr, "amplitude_retrace": a_re,
            "size_um": float(size_um), "x_offset_um": float(offset_um[0]),
            "y_offset_um": float(offset_um[1]), "angle_deg": 0.0}


def capture_to_sample_npz(path, dest, **options) -> Path:
    """Write the twin plugin's sample `.npz` (with the real retrace) from a capture."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dest, **load_capture(path, **options))
    return dest
