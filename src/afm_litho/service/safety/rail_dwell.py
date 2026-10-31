"""Safety kernel -- entry #1: the Z-drive rail-dwell telemetry watchdog.

(Lives in the service/safety/ package; re-exported from
service/safety/__init__.py, so `from ..safety import RailDwellGuard` is unchanged.)

This is the FIRST concrete piece of the safety kernel (docs/SAFETY_KERNEL_PLAN.md
is the architecture; docs/SAFETY_KERNEL_RAIL_DWELL.md is this guard's spec). It
introduces a hazard class the plan's 22 command-clamp guards do not cover -- a
*telemetry watchdog*: it watches a MEASURED signal (the HV Z-drive voltage) over
time and trips on a dwell condition, rather than clamping a commanded value.

Hazard #23 -- prolonged piezo rail-dwell -> depolarization. Holding the Z piezo
fully extended (near +150 V) or retracted (near/below 0 V) for a long time
depolarizes it. Under CONTINUOUS unattended imaging, Z thermal drift can walk the
servo to a rail and pin it there for a whole night. The vendor Z HV piezo rail is
-10 -> +150 V (ScanMaster.ipf:812, HardWare.ipf:9595-9596); the trip thresholds
sit ~10 V inside each rail.

This module is deliberately LEAN and pure (no hub / asyncio / hardware deps), so
every guard decision is unit-testable off-instrument. The continuous supervisor
(host/continuous.py) owns the ~10 Hz poll, the trip action (stop-with-withdraw),
and the reporting; it consults a `RailDwellGuard` for the decision. K1's SafeARC
command proxy and the rest of the kernel grow into this module later.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class RailDwellLimits:
    """Sourced operator-config limits, ONLY-TIGHTEN (plan Sec 2.4: a client may
    request a tighter ceiling, never a looser one).

    Defaults (docs/SAFETY_KERNEL_RAIL_DWELL.md):
      v_hi / v_lo   -- rail thresholds ~10 V inside the -10..+150 V Z HV rail.
      frame_frac    -- Guard A: stop if > this fraction of a FRAME's samples railed.
      line_frac     -- Guard B: a LINE counts railed only if > this fraction of its
                       samples railed (pinned at a rail almost end to end).
      consec_lines  -- Guard B: abort after this many consecutive railed lines.
      poll_hz       -- the HV Z-drive sample rate during a continuous scan.
    """

    v_hi: float = 140.0
    v_lo: float = 0.0
    frame_frac: float = 0.5
    line_frac: float = 0.95
    consec_lines: int = 5
    poll_hz: float = 10.0

    def tighten(self, **overrides) -> "RailDwellLimits":
        """Return limits with the requested overrides applied ONLY in the tightening
        direction (a trip that fires EARLIER); a loosening request is silently
        floored/capped to the current value, never applied. `poll_hz` is a sampling
        rate, not a trip threshold, so it takes the requested value as-is (a faster
        poll is only more sensitive)."""
        out = dict(v_hi=self.v_hi, v_lo=self.v_lo, frame_frac=self.frame_frac,
                   line_frac=self.line_frac, consec_lines=self.consec_lines,
                   poll_hz=self.poll_hz)
        if "v_hi" in overrides and overrides["v_hi"] is not None:
            out["v_hi"] = min(self.v_hi, float(overrides["v_hi"]))   # lower = tighter
        if "v_lo" in overrides and overrides["v_lo"] is not None:
            out["v_lo"] = max(self.v_lo, float(overrides["v_lo"]))   # higher = tighter
        if "frame_frac" in overrides and overrides["frame_frac"] is not None:
            out["frame_frac"] = min(self.frame_frac, float(overrides["frame_frac"]))
        if "line_frac" in overrides and overrides["line_frac"] is not None:
            out["line_frac"] = min(self.line_frac, float(overrides["line_frac"]))
        if "consec_lines" in overrides and overrides["consec_lines"] is not None:
            out["consec_lines"] = min(self.consec_lines, int(overrides["consec_lines"]))
        if "poll_hz" in overrides and overrides["poll_hz"] is not None:
            out["poll_hz"] = max(1.0, float(overrides["poll_hz"]))
        return replace(self, **out)

    def as_dict(self) -> dict:
        return asdict(self)


class RailDwellGuard:
    """The rail-dwell decision state a continuous scan consults. PURE logic: fed
    Z-drive samples grouped into lines, it tracks consecutive railed lines (Guard
    B) and the per-frame railed fraction (Guard A). The supervisor owns the poll,
    the trip action, and the reporting; this only DECIDES.

    Lifecycle over one continuous run:
        arm()                         -- at run start (resets everything)
          start_frame()               -- before each frame (resets frame totals;
                                         the consecutive-line counter PERSISTS
                                         across frames -- the tip sitting railed
                                         does not un-rail at a frame boundary)
            add_sample(v) * N         -- each ~10 Hz poll sample of the CURRENT line
            close_line(row) -> trip?  -- when the pump advances to the next line
          finish_frame() -> trip?     -- after the frame (Guard A)
        disarm()                      -- at run end / on a trip
    """

    def __init__(self, limits: RailDwellLimits | None = None):
        self.limits = limits or RailDwellLimits()
        self.armed = False
        # `tripped` is set by the SUPERVISOR (via host/continuous._rail_trip)
        # once it has acted on a trip this guard returned, so re-evaluation and the
        # single withdraw site stay idempotent; None means no trip yet this run.
        self.tripped: dict | None = None
        self.reset()

    # ---------------------------------------------------------------- lifecycle --

    def reset(self) -> None:
        self._line: list[float] = []      # current line's samples
        self.consec = 0                   # consecutive railed lines (persists per frame)
        self._frame_hits = 0
        self._frame_total = 0
        self.last_line_frac: float | None = None
        self.last_row: int | None = None
        self.tripped = None

    def arm(self, limits: RailDwellLimits | None = None) -> None:
        if limits is not None:
            self.limits = limits
        self.armed = True
        self.reset()

    def disarm(self) -> None:
        self.armed = False

    def start_frame(self) -> None:
        """Reset the per-frame accumulators before a frame. The consecutive-railed
        counter is intentionally NOT reset (Guard B is about the tip dwelling at a
        rail, which spans frame boundaries); the current-line buffer is cleared so
        a previous frame's unclosed final line cannot leak into this one."""
        self._line = []
        self._frame_hits = 0
        self._frame_total = 0

    # ------------------------------------------------------------------- feeding --

    def _is_rail(self, v: float) -> bool:
        return v >= self.limits.v_hi or v <= self.limits.v_lo

    def add_sample(self, v: float | None) -> None:
        """Append one poll sample to the current line. A failed/None/NaN read is
        dropped (a read hiccup must never fabricate a rail hit or clear one)."""
        if not self.armed or v is None:
            return
        v = float(v)
        if math.isnan(v) or math.isinf(v):
            return
        self._line.append(v)

    def close_line(self, row: int | None = None) -> dict | None:
        """Finalize the current line and return a Guard-B trip descriptor, or None.

        A line is RAILED only if > line_frac of its samples were at a rail. A line
        with ZERO samples is 'no data' -- it is skipped (neither counted railed NOR
        allowed to reset the consecutive counter), so a poll that briefly missed a
        line cannot silently break a genuine rail streak. Reaching consec_lines
        consecutive railed lines returns the trip."""
        samples = self._line
        self._line = []
        n = len(samples)
        if n == 0:
            return None                                   # no data -> skip, don't reset
        hits = sum(1 for v in samples if self._is_rail(v))
        self._frame_hits += hits
        self._frame_total += n
        frac = hits / n
        self.last_line_frac = frac
        self.last_row = row
        if frac > self.limits.line_frac:
            self.consec += 1
        else:
            self.consec = 0                               # any clear line breaks the streak
        if self.consec >= self.limits.consec_lines:
            return {"guard": "B", "consec": self.consec}
        return None

    def finish_frame(self) -> dict | None:
        """Evaluate Guard A at frame end. Any residual unclosed line is folded into
        the frame totals first (so its samples count toward the fraction). Returns
        an A trip descriptor when > frame_frac of the frame's samples railed."""
        if self._line:
            samples = self._line
            self._line = []
            hits = sum(1 for v in samples if self._is_rail(v))
            self._frame_hits += hits
            self._frame_total += len(samples)
        frac = self.frame_frac
        if self._frame_total and frac > self.limits.frame_frac:
            return {"guard": "A", "frac": frac}
        return None

    # ------------------------------------------------------------------- readout --

    @property
    def frame_frac(self) -> float:
        """Running railed fraction of the current frame (0..1)."""
        return (self._frame_hits / self._frame_total) if self._frame_total else 0.0

    def readout(self) -> dict:
        """JSON-safe live state for the Scan tab (armed indicator + counter climb)."""
        return {
            "armed": bool(self.armed),
            "consec": int(self.consec),
            "consec_limit": int(self.limits.consec_lines),
            "line_frac": (round(self.last_line_frac, 4)
                          if self.last_line_frac is not None else None),
            "frame_frac": round(self.frame_frac, 4),
            "frame_frac_limit": float(self.limits.frame_frac),
            "last_row": self.last_row,
            "v_hi": float(self.limits.v_hi),
            "v_lo": float(self.limits.v_lo),
            "tripped": self.tripped,
        }
