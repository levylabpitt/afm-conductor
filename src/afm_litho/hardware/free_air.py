"""Free-air baseline reads -- the pure, reused definition of a valid free-air
reading (docs/INSTRUMENT_MODEL_SPEC.md sections 12 + 12.1).

The caveat (2026-09-22): changes to the read path have broken the whole
service before, so the AGREEMENT rule and the 3-read/dwell/3-read retry live
here as a PURE function over injected `read_one` / `now` / `sleep` callables,
with NO instrument access and NO hub coupling. Both the scan re-reference
(host/continuous.py), the approach door and the write set-down call this one
implementation; the callers own the side effects (setting `hub.free_air`).

Agreement (12.1 + 13.1's near-zero addition): a sample set agrees when its
spread `max - min` sits inside `max(frac * |mean|, abs_band_v)`. The absolute
floor `abs_band_v` (model field `free_air_abs_band_v`, 0.02 V placeholder) keeps
the band physically sane when the mean is essentially zero, where the relative
term `frac * |mean|` would otherwise collapse and reject any nonzero jitter.
Boundary inclusive; a NaN/Inf anywhere never agrees.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# Default absolute floor (V) of the free-air agreement band, owned here; LithographyRun and
# service/instrument_model.py both take it from this constant. 0.02 V is a PLACEHOLDER
# above expected withdrawn-deflection noise: measure on the rig.
FREE_AIR_ABS_BAND_V = 0.02


@dataclass(frozen=True)
class FreeAirRead:
    """One accepted free-air baseline: the mean the caller latches, the three raw
    samples it was computed from, the timestamp (`now()`), and the imaging mode
    the read was taken in."""

    value_v: float
    samples: tuple[float, float, float]
    ts: float
    mode: str


class FreeAirDisagree(Exception):
    """Raised when both triples fail to agree. `.samples` carries ALL SIX reads
    (the first triple followed by the second), so the operator/log sees the whole
    failed sequence."""

    def __init__(self, samples: Sequence[float]):
        self.samples = tuple(samples)
        super().__init__(
            "free-air reads disagree after a dwell-and-retry: "
            f"{self.samples}")


def agree_within(samples: Sequence[float], frac: float = 0.10,
                 abs_band_v: float = 0.0) -> bool:
    """True when `samples` all sit inside one agreement band (pure, order-
    independent).

    Band = `max(frac * |mean|, abs_band_v)`; agreement is `(max - min) <= band`,
    boundary INCLUSIVE. A NaN or Inf anywhere returns False unconditionally,
    evaluated BEFORE any spread/mean arithmetic so a bad read can never cancel
    out (e.g. `inf - inf` is NaN, which would make `<=` silently False for the
    wrong reason).
    """
    vals = [float(s) for s in samples]
    if not vals:
        return False
    if any(math.isnan(v) or math.isinf(v) for v in vals):
        return False
    mean = sum(vals) / len(vals)
    spread = max(vals) - min(vals)
    band = max(frac * abs(mean), abs_band_v)
    return spread <= band


def read_free_air(read_one: Callable[[], float], *, now: Callable[[], float],
                  sleep: Callable[[float], None], mode: str,
                  dwell_s: float = 5.0, frac: float = 0.10,
                  abs_band_v: float = 0.0) -> FreeAirRead:
    """Take a valid free-air baseline, with one dwell-and-retry.

    Reads three samples; if they agree (`agree_within`) returns their mean at
    once. Otherwise sleeps `dwell_s` (tip still withdrawn) and reads three more;
    if THOSE agree returns the second triple's mean. If they disagree again,
    raises `FreeAirDisagree` carrying all six reads. A `read_one` that raises
    propagates immediately -- a broken read channel must surface, never be
    retried into a different failure.
    """
    first = (read_one(), read_one(), read_one())
    if agree_within(first, frac=frac, abs_band_v=abs_band_v):
        return FreeAirRead(value_v=sum(first) / 3.0, samples=first,
                           ts=float(now()), mode=mode)
    sleep(dwell_s)
    second = (read_one(), read_one(), read_one())
    if agree_within(second, frac=frac, abs_band_v=abs_band_v):
        return FreeAirRead(value_v=sum(second) / 3.0, samples=second,
                           ts=float(now()), mode=mode)
    raise FreeAirDisagree(first + second)
