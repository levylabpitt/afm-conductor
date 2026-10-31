"""Project a RAW measured capture into titled 2-D channel grids.

A capture (`service/captures.py`) is a dict of `{channel -> 1-D samples}`. Gwyddion
and the on-demand exporter both want each channel as a `(rows, pixels)` grid in base
SI. This module owns that projection so the export writers never duplicate the
reshape/scale math:

  * a 2-D SCAN channel is `rows*2*pixels` long (trace THEN retrace per line). EVERY
    channel is projected as BOTH `<chan> trace` and `<chan> retrace` (their difference
    reveals feedback lag): topography (`ZSensor`) as volts -> height in METRES the same
    way `hw_scan._rows_to_lines` does (`-(v - z_off) * z_sens`, no *1e9); every other
    channel (Deflection/error, Output.A) in raw volts.
  * any other length is a 1-D litho trajectory readback: one thin `1 x N` grid per
    channel, no trace/retrace split (`ZSensor` still volts -> metres).

The split/flip is identical to `hw_scan._rows_to_lines`, minus the display-only
per-line flatten -- RAW/unflattened. First output row = top of the image.
"""

from __future__ import annotations

import numpy as np
from ..hardware import channels

_TOPO = channels.Z_SENSOR          # the one channel projected as height (METRES), not volts


def _split_grid(arr, rows: int, pixels: int, half: str,
                direction: str = "up") -> np.ndarray:
    """Reshape a scan channel's `2*pixels`-per-line samples into a `(rows, pixels)`
    grid, keeping one sweep: `half="trace"` -> `line[:pixels]`; `half="retrace"` ->
    `line[pixels:][::-1]` (flipped so it overlays the trace spatially). SAME split/flip
    as `hw_scan._rows_to_lines`; the slow axis is placed by `direction`."""
    ppl = 2 * pixels
    need = ppl * rows
    a = np.asarray(arr, dtype=np.float64)
    if a.size < need:
        a = np.pad(a, (0, need - a.size))
    grid = np.full((rows, pixels), np.nan, dtype=np.float64)
    for r in range(rows):
        line = a[r * ppl:(r + 1) * ppl]
        row_vals = line[pixels:][::-1] if half == "retrace" else line[:pixels]
        slow_idx = r if direction == "up" else (rows - 1 - r)
        grid[slow_idx, :] = row_vals[:pixels]
    return grid


def _capture_to_grid(arr, rows: int, pixels: int, trace_retrace: bool,
                     direction: str = "up") -> np.ndarray:
    """Reshape a channel's 1-D samples into a `(rows, pixels)` grid, matching
    `hw_scan._rows_to_lines`: a trace+retrace line (`2*pixels`) keeps its TRACE half; a
    single-sweep snake un-reverses odd rows; the slow axis is placed by `direction`.
    Short arrays are zero-padded; first output row = top."""
    ppl = 2 * pixels if trace_retrace else pixels
    need = ppl * rows
    a = np.asarray(arr, dtype=np.float64)
    if a.size < need:
        a = np.pad(a, (0, need - a.size))
    grid = np.full((rows, pixels), np.nan, dtype=np.float64)
    for r in range(rows):
        line = a[r * ppl:(r + 1) * ppl]
        if trace_retrace:
            row_vals = line[:pixels]
        elif r % 2 == 1:                      # snake: odd rows captured reversed
            row_vals = line[::-1]
        else:
            row_vals = line
        slow_idx = r if direction == "up" else (rows - 1 - r)
        grid[slow_idx, :] = row_vals[:pixels]
    return grid


def capture_channels_2d(cap: dict, config, cal: dict) -> list[dict]:
    """Project a RAW capture dict into a list of channel entries, one per titled field:

        {"name": str, "grid2d": (rows, pixels) float64 array (first row = top),
         "rows": int, "pixels": int, "x_real_m": float, "y_real_m": float,
         "z_units": "m"|"V"}

    `config` supplies the grid (`.pixels`, `.lines`, `.size_um`, `.trace_retrace`,
    `.direction`); `cal` supplies the Z LVDT cal (`z_lvdt_sens`/`z_lvdt_offset`). RAW,
    unflattened, base SI (ZSensor topography in metres, all other channels raw volts).
    Returns `[]` for an empty capture. A 2-D scan (`rows*2*pixels` per channel) yields
    trace + retrace for EVERY channel (ZSensor in metres, others in volts); any other
    length is a 1-D litho readback stored as a `1 x N` field per channel.
    """
    if not cap:
        return []
    pixels = int(config.pixels)
    rows = int(getattr(config, "lines", 0)) or pixels
    trace_retrace = bool(getattr(config, "trace_retrace", False))
    direction = str(getattr(config, "direction", "up"))
    size_m = float(config.size_um) * 1e-6
    z_sens = float(cal.get("z_lvdt_sens", 1.0)) if cal else 1.0
    z_off = float(cal.get("z_lvdt_offset", 0.0)) if cal else 0.0
    scan_len = rows * 2 * pixels                     # trace+retrace line block

    def topo(grid):
        return -(grid - z_off) * z_sens              # volts -> height in METRES

    out: list[dict] = []

    def add(name: str, grid, z_units: str, x_real: float, y_real: float) -> None:
        g = np.ascontiguousarray(np.asarray(grid, dtype=np.float64))
        yres, xres = g.shape
        out.append({"name": name, "grid2d": g, "rows": int(yres),
                    "pixels": int(xres), "x_real_m": float(x_real),
                    "y_real_m": float(y_real), "z_units": z_units})

    for chan, samples in cap.items():
        if int(np.asarray(samples).size) == scan_len:
            # 2-D scan line block: EVERY channel -> trace + retrace (their difference
            # reveals feedback lag). Topography (ZSensor) in metres, all others in raw
            # volts. Field is `size_m` square.
            for half in ("trace", "retrace"):
                grid = _split_grid(samples, rows, pixels, half, direction)
                if chan == _TOPO:
                    add(f"{chan} {half}", topo(grid), "m", size_m, size_m)
                else:
                    add(f"{chan} {half}", grid, "V", size_m, size_m)
        else:
            # 1-D litho trajectory readback (or any non-scan length): one thin (rows,
            # pixels) grid per channel, no trace/retrace split. For a single-row profile
            # the row is one pixel tall (`x_real / pixels`) so a sample stays ~square.
            grid = _capture_to_grid(samples, rows, pixels, trace_retrace, direction)
            y_real = size_m / pixels if rows == 1 and pixels else size_m
            if chan == _TOPO:
                add(chan, topo(grid), "m", size_m, y_real)
            else:
                add(chan, grid, "V", size_m, y_real)
    return out
