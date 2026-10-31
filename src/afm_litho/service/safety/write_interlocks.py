"""Runtime write interlocks -- the always-on, continuous-time guards armed for the
whole duration of EVERY hardware engage (approach, scan, write) on both doors
(docs/INSTRUMENT_MODEL_SPEC.md section 12 + 12.1).

Three conditions, each tripping `safe_park` when it holds CONTINUOUSLY for its
dwell window (the scan-path's line/point-count guards become seconds here):
  * `rail_dwell`          -- `z_out_v` above `model.z_rail_v` (140 V);
  * `sum_collapse`        -- DetectorSum below `config.sum_floor_v` (0.5 V), OR
                             below `config.sum_frac` (0.5) of its pre-engage
                             baseline;
  * `deflection_integrity`-- `|defl_v| > sum_v` (deflection cannot physically
                             exceed the total detector signal).

Continuous-time (12.1): the FIRST sample at which a condition becomes true starts
a per-condition clock; a single clean sample resets that clock; when the elapsed
time reaches the dwell (INCLUSIVE) the interlock trips ONCE (calling `on_trip`)
and self-disarms, so every later `feed()` is a no-op. `feed()` before `arm()` is
a silent no-op.

This is the pure state machine. The caller (the always-on telemetry consumer,
`host/telemetry._hw_telemetry_frame` on hardware; the twin's telemetry/step
loop on the twin) constructs one per engage, `arm()`s it with the pre-engage SUM
baseline, `feed()`s it every telemetry point, and passes an `on_trip` that calls
`safe_park(hub, reason)`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class InterlockConfig:
    """The running-interlock thresholds NOT sourced from the instrument model
    (12.1). The rail threshold is the one model-sourced value (`model.z_rail_v`);
    everything else lives here so a caller can tune the SUM/deflection windows
    without an override."""

    rail_v: float = 140.0        # documentary default; the running rail reads model.z_rail_v
    rail_s: float = 5.0
    sum_floor_v: float = 0.5
    sum_frac: float = 0.5
    sum_s: float = 5.0
    defl_s: float = 5.0


def _mval(field) -> float:
    """A model threshold, whether the model stores it as an `InstrumentModel.Field`
    (`.value`) or as a plain float (the tests' minimal stand-in model)."""
    return float(getattr(field, "value", field))


def _finite(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


class WriteInterlocks:
    """One instance per engage. Not thread-safe on its own -- the caller feeds it
    from a single telemetry consumer (or holds a lock, as the continuous
    supervisor already does around the rail guard)."""

    def __init__(self, model, config: InterlockConfig, *,
                 on_trip: Callable[[str], None]):
        self._model = model
        self._config = config
        self._on_trip = on_trip
        self._armed = False
        self._baseline: float | None = None
        # Per-condition continuous-time clocks: the t at which the condition
        # first became true, or None when it is currently clean.
        self._since: dict[str, float | None] = {
            "rail_dwell": None, "sum_collapse": None, "deflection_integrity": None}

    @property
    def armed(self) -> bool:
        return self._armed

    def arm(self, sum_baseline_v: float) -> None:
        """Arm the interlocks for an engage, latching the pre-engage SUM baseline
        (used by the relative SUM-collapse condition). Resets every clock."""
        self._armed = True
        self._baseline = _finite(sum_baseline_v)
        for k in self._since:
            self._since[k] = None

    def disarm(self) -> None:
        """Stop evaluating. Idempotent."""
        self._armed = False
        for k in self._since:
            self._since[k] = None

    def feed(self, *, t: float, z_out_v: float, sum_v: float,
             defl_v: float) -> None:
        """Consume one telemetry point. A no-op unless armed. Trips at most once,
        then self-disarms."""
        if not self._armed:
            return
        z = _finite(z_out_v)
        s = _finite(sum_v)
        d = _finite(defl_v)

        rail_active = None if z is None else (z > _mval(self._model.z_rail_v))

        if s is None:
            sum_active = None
        else:
            collapsed = s < float(self._config.sum_floor_v)
            if not collapsed and self._baseline is not None:
                collapsed = s < float(self._config.sum_frac) * self._baseline
            sum_active = collapsed

        # |defl| > SUM needs BOTH channels this point; a bad read on either leaves
        # the clock unchanged rather than trips or clears.
        defl_active = None if (d is None or s is None) else (abs(d) > s)

        conditions = (
            ("rail_dwell", rail_active, float(self._config.rail_s)),
            ("sum_collapse", sum_active, float(self._config.sum_s)),
            ("deflection_integrity", defl_active, float(self._config.defl_s)),
        )
        for reason, active, dwell_s in conditions:
            if active is None:
                continue                      # unknown read: hold the clock
            if not active:
                self._since[reason] = None
                continue
            if self._since[reason] is None:
                self._since[reason] = float(t)
            if (float(t) - self._since[reason]) >= dwell_s:
                self._trip(reason)
                return

    def _trip(self, reason: str) -> None:
        self._armed = False
        for k in self._since:
            self._since[k] = None
        self._on_trip(reason)
