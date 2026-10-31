"""Pacing and sample marking for the twin's lithography write.

The twin plays a write by advancing along the compiled points on the wall clock
(`passes.LithoRun`), not by stepping the 50 kHz controller model. This module holds the
two parts of that which do not depend on the run: the clock (`Pacer`) and the mark the
tip leaves on the simulated sample (`stamp_dose`), plus the conductance the live
current is read from.

`twin_speed` scales every simulated duration of a write: 1 is real time, N is N times
faster, 0 is unlimited (no waiting). It never scales anything the instrument side
measures in wall time (dead-man, heartbeat, reply bounds, the interlock feed, the page
frame rate).
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

from ..twin.arc import DT
from ..twin.physics import METAL_SEED_DOSE

SPEED_ENV = "AFM_LITHO_TWIN_SPEED"
DEFAULT_SPEED = 1.0
# The longest the write worker sleeps without looking at its stop flag. An Abort, a
# Withdraw, a park and the dead-man are all seen within one slice at any speed.
WAIT_SLICE_S = 0.05


def parse_speed(value) -> float:
    """A `twin_speed` as a finite number >= 0, or ValueError. A bool is refused:
    `float(True)` is 1.0, and `twin_speed: true` is a typo, not a speed."""
    try:
        speed = math.nan if isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        speed = math.nan
    if not math.isfinite(speed) or speed < 0:
        raise ValueError(f"twin_speed must be a number >= 0, got {value!r}")
    return speed


def speed_from_env() -> float:
    """The station's `twin_speed`: env AFM_LITHO_TWIN_SPEED, real time when unset."""
    raw = os.environ.get(SPEED_ENV, "").strip()
    if not raw:
        return DEFAULT_SPEED
    try:
        return parse_speed(raw)
    except ValueError:
        raise ValueError(f"{SPEED_ENV}={raw!r} is not a number >= 0") from None


class Pacer:
    """The write worker's clock: wall time scaled by `speed`, and waits that end early
    on a stop. `stopped` is polled, never waited on, so nothing can delay a stop."""

    def __init__(self, speed: float, stopped, clock=time.monotonic, sleep=time.sleep):
        self.speed = parse_speed(speed)
        self._stopped = stopped
        self.now = clock
        self._sleep = sleep

    @property
    def unlimited(self) -> bool:
        return self.speed == 0.0

    def wall(self, sim_s: float) -> float:
        """Wall seconds a simulated duration takes (0 when unlimited)."""
        return 0.0 if self.unlimited else float(sim_s) / self.speed

    def sim(self, wall_s: float) -> float:
        """Simulated seconds elapsed over a wall duration (infinite when unlimited)."""
        return math.inf if self.unlimited else float(wall_s) * self.speed

    def wait_until(self, deadline: float) -> bool:
        """Sleep until `deadline` (this clock) in slices of at most WAIT_SLICE_S.
        False as soon as a stop is seen."""
        while not self._stopped():
            remaining = deadline - self.now()
            if remaining <= 0:
                return True
            self._sleep(min(remaining, WAIT_SLICE_S))
        return False

    def dwell(self, sim_s: float) -> bool:
        """Wait out a simulated pause (tip down, relax, lift, settle)."""
        if self.unlimited:
            return not self._stopped()
        return self.wait_until(self.now() + self.wall(sim_s))


def _cells(sample, x_m, y_m):
    """`Sample._dose_idx` for arrays: the dose-grid cell under each position."""
    n = sample.dose_n
    i = ((np.asarray(x_m, dtype=float) / sample.dose_range + 0.5) * (n - 1) + 0.5)
    j = ((np.asarray(y_m, dtype=float) / sample.dose_range + 0.5) * (n - 1) + 0.5)
    return (np.clip(i.astype(np.int64), 0, n - 1),
            np.clip(j.astype(np.int64), 0, n - 1))


def conductance_at(sample, x_m, y_m) -> np.ndarray:
    """`Sample.conductance` for arrays."""
    i, j = _cells(sample, x_m, y_m)
    d = sample.dose[i, j].astype(float)
    return d * d / (1.0 + d * d)


def _stamp_weights(sample, i, j, weight):
    """Sum `weight` over the 3x3 stamp around each cell (i, j). Returns the sums and
    the dose-grid slices they cover (the stamps' bounding box). Stamp cells off the
    grid are dropped, as the slice in `Sample.deposit` drops them."""
    n = sample.dose_n
    i0, i1 = max(int(i.min()) - 1, 0), min(int(i.max()) + 2, n)
    j0, j1 = max(int(j.min()) - 1, 0), min(int(j.max()) + 2, n)
    ni, nj = i1 - i0, j1 - j0
    acc = np.zeros(ni * nj)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ii, jj = i + di - i0, j + dj - j0
            ok = (ii >= 0) & (ii < ni) & (jj >= 0) & (jj < nj)
            acc += np.bincount(ii[ok] * nj + jj[ok], weights=weight[ok],
                               minlength=ni * nj)
    return acc.reshape(ni, nj), (slice(i0, i1), slice(j0, j1))


def stamp_dose(sample, x_m, y_m, bias_v, dwell_s) -> None:
    """Mark the sample for a tip that sat `dwell_s` at each (x_m, y_m) under `bias_v`.

    The rule is `Sample.deposit` applied once per controller tick of each dwell: a 3x3
    stamp, +bias * dt * 8 for a writing bias, a per-tick factor of 1 + bias * dt * 40
    for an erasing one, nothing below 1 V, and metal never erased below its seed dose.
    Consecutive positions of one sign are applied together; a sign change keeps its
    order, because erasing then writing a cell differs from writing then erasing it.
    """
    x_m = np.atleast_1d(np.asarray(x_m, dtype=float))
    y_m = np.atleast_1d(np.asarray(y_m, dtype=float))
    bias = np.broadcast_to(np.asarray(bias_v, dtype=float), x_m.shape)
    dwell = np.broadcast_to(np.asarray(dwell_s, dtype=float), x_m.shape)
    active = np.flatnonzero(np.abs(bias) >= 1.0)
    if not active.size:
        return
    i_all, j_all = _cells(sample, x_m, y_m)
    writing = bias[active] > 0
    cuts = np.flatnonzero(np.diff(writing)) + 1
    for run in np.split(np.arange(active.size), cuts):
        idx = active[run]
        i, j, b, t = i_all[idx], j_all[idx], bias[idx], dwell[idx]
        if b[0] > 0:
            add, box = _stamp_weights(sample, i, j, b * t * 8.0)
            sample.dose[box] += add.astype(sample.dose.dtype)
            continue
        per_tick = np.maximum(1.0 + b * DT * 40.0, 1e-300)
        log_factor, box = _stamp_weights(sample, i, j, (t / DT) * np.log(per_tick))
        touched = log_factor < 0
        region = sample.dose[box]
        region[touched] *= np.exp(log_factor[touched]).astype(region.dtype)
        metal = getattr(sample, "metal_mask", None)
        if metal is not None:
            keep = touched & metal[box]
            region[keep] = np.maximum(region[keep], METAL_SEED_DOSE)


def substeps(sample, x_m, y_m) -> int:
    """How many equal parts each point-to-point move is marked in, so consecutive
    stamps are at most half a dose-grid cell apart and the mark has no gaps."""
    if len(x_m) < 2:
        return 1
    pitch = sample.dose_range / max(sample.dose_n - 1, 1)
    step = float(np.max(np.hypot(np.diff(x_m), np.diff(y_m))))
    return max(1, math.ceil(step / (pitch / 2.0)))


def stamp_path(sample, x_m, y_m, bias_v, dt_point_s: float, parts: int) -> None:
    """Mark the moves between consecutive points of a path: each takes `dt_point_s`,
    with position and bias linear between its two ends (the out-wave interpolates).
    A sample that only holds the position at 0 V is not a move and leaves no mark."""
    if len(x_m) < 2:
        return
    x_m, y_m, bias_v = (np.asarray(a, dtype=float) for a in (x_m, y_m, bias_v))
    hold = (np.diff(x_m) == 0) & (np.diff(y_m) == 0) & (bias_v[1:] == 0)
    f = (np.arange(parts) + 0.5) / parts

    def mid(a):
        return (a[:-1, None] + np.diff(a)[:, None] * f)[~hold].ravel()

    stamp_dose(sample, mid(x_m), mid(y_m), mid(bias_v), dt_point_s / parts)
