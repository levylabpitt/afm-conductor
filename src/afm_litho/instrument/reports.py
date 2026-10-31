"""What an instrument pushes to the program unasked: telemetry frames and status messages."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

from ..run.model import Part
from .commands import Citation

# Igor's always-on tip poll (AFM_TipDesiredPeriod, igor/afm_litho_run.ipf) runs on the 60 Hz
# tick at period 6 (10 Hz) while AFM_ltActive or AFM_scActive is set, else period 15 (4 Hz).
# AFM_ltActive is set only by AFM_LithoChainStart (a chained, LITHOCHAIN write) and
# AFM_scActive only by AFM_ScanChainStart, so a one-chunk ARMFIRE write polls at 4 Hz.
# On the Asylum side Igor sets the rate itself; the twin copies the same rule.
TELEMETRY_HZ_CHAINED = 10.0
TELEMETRY_HZ_IDLE = 4.0

Activity = Literal["idle", "moving", "engaging", "engaged", "withdrawing", "reading",
                   "writing", "scanning", "parking", "parked"]
ACTIVITIES: tuple[Activity, ...] = ("idle", "moving", "engaging", "engaged", "withdrawing",
                                    "reading", "writing", "scanning", "parking", "parked")


def telemetry_hz(activity: Activity, chained_write: bool = False) -> float:
    """Igor's tip-poll rate while the instrument is doing `activity`.

    `chained_write` is whether the object being written fires as a chain (more than one
    chunk); a scan always fires as a chain.
    """
    if activity not in ACTIVITIES:
        raise ValueError(f"{activity!r} is not an activity")
    if activity == "scanning" or (activity == "writing" and chained_write):
        return TELEMETRY_HZ_CHAINED
    return TELEMETRY_HZ_IDLE


@dataclass(frozen=True)
class TelemetryFrame:
    """One sample of the tip poll: today's `TELEM x y zdrive defl sum amp`, with a time.

    `t_s` is afm_litho.clock.now() when the values were read. x and y are the tip position in
    micrometres (the plugin converts the X/Y sensor volts with its own calibration); the rest
    are volts. A channel that could not be read is NaN, so one bad channel never drops the
    frame. Amplitude means something only in AC mode.
    """

    t_s: float
    x_um: float
    y_um: float
    z_drive_v: float
    deflection_v: float
    sum_v: float
    amplitude_v: float

    def __post_init__(self) -> None:
        for name in ("t_s", "x_um", "y_um", "z_drive_v", "deflection_v", "sum_v",
                     "amplitude_v"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"TelemetryFrame.{name}: expected a number, got {value!r}")
            object.__setattr__(self, name, float(value))
            if math.isinf(value):
                raise ValueError(f"TelemetryFrame.{name}: infinite (NaN marks an unread channel)")
        if math.isnan(self.t_s):
            raise ValueError("TelemetryFrame.t_s: a frame always has a time")


@dataclass(frozen=True)
class Status(Part):
    """What the instrument is doing now, in words a user reads ("writing object 3: wire_a").

    `cite` is the run step being carried out, when the command carried one.
    """

    t_s: float
    activity: Activity
    text: str
    cite: Citation | None = None


TelemetryListener = Callable[[TelemetryFrame], None]
StatusListener = Callable[[Status], None]
