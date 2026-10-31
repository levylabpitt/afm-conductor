"""The commands the program sends an instrument, and the result each one comes back with.

Positions are lab-frame micrometres, voltages are volts at the tip or the detector, and
channels are named by their Asylum read alias (Height, Deflection, ZSensor, DetectorSum,
XSensor, ...). Nothing here is Igor-, wire- or brand-specific: the Asylum plugin converts to
LVDT and DAC volts itself, and the twin plays the same numbers directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

# Messages are checked on construction with the run contract's rules (allowed values,
# finite numbers, typeable ASCII), so a bad command is refused exactly as a bad run part is.
from ..run.model import EXECUTION_ID, Mode, Part, ScanFrame, Setpoint, quotable

# A read alias is one word. Dotted or percent names (Output.Z, Input.Fast, Z%Output) are
# platform literals or twin names, which the program never uses to read a channel.
ALIAS = re.compile(r"[A-Za-z][A-Za-z0-9]*")

Outcome = Literal["done", "refused", "aborted"]
Verb = Literal["move", "engage", "withdraw", "read", "park", "write", "scan"]
VERBS: tuple[Verb, ...] = ("move", "engage", "withdraw", "read", "park", "write", "scan")


def _aliases(channels: tuple[str, ...], where: str) -> None:
    for name in channels:
        if not (isinstance(name, str) and ALIAS.fullmatch(name)):
            raise ValueError(f"{where}: {name!r} is not a channel alias")
    if len(set(channels)) != len(channels):
        raise ValueError(f"{where}: a channel is named twice")


def _points(value, where: str) -> np.ndarray:
    """One commanded value per point: a read-only 1-D float array, every value finite."""
    array = np.array(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{where}: expected one value per point")
    if not np.isfinite(array).all():
        raise ValueError(f"{where}: every point must be finite")
    array.setflags(write=False)
    return array


def _positive(value: float, where: str) -> None:
    if value <= 0:
        raise ValueError(f"{where}: must be more than 0, got {value}")


def _transit_speed(value: float | None, where: str) -> None:
    if value is not None:
        _positive(value, f"{where}.transit_speed_um_s")


def _one_run_in_order(steps: tuple[Citation, ...], where: str) -> None:
    if len({c.run for c in steps}) > 1:
        raise ValueError(f"{where}: every step is from one run")
    numbers = [c.step for c in steps]
    if numbers != sorted(set(numbers)):
        raise ValueError(f"{where}: steps are distinct and in run order")


@dataclass(frozen=True)
class Citation(Part):
    """The run step a command carries out: Phase 3's `run.cite(n)`, as `Citation(**run.cite(n))`."""

    run: str
    step: int

    def check(self) -> None:
        if not EXECUTION_ID.match(self.run):
            raise ValueError(f"Citation.run: {self.run!r} is not an execution id")
        if self.step < 0:
            raise ValueError("Citation.step: step numbers start at 0")


# --- single verbs: approach, withdraw, park, the end-of-run withdraw ----------


@dataclass(frozen=True)
class MoveTo(Part):
    """Move the tip in XY to (x, y) at a speed, off the surface (withdrawn first if engaged)."""

    x_um: float
    y_um: float
    speed_um_s: float
    cite: Citation | None = None

    def check(self) -> None:
        _positive(self.speed_um_s, "MoveTo.speed_um_s")


@dataclass(frozen=True)
class EngageAt(Part):
    """Engage the tip on the surface at a setpoint, in a mode. The setpoint has no default."""

    setpoint: Setpoint
    mode: Mode
    cite: Citation | None = None


@dataclass(frozen=True)
class WithdrawTip(Part):
    """Withdraw the tip from the surface; the result says whether the withdraw was verified."""

    cite: Citation | None = None


@dataclass(frozen=True)
class ReadChannels(Part):
    """Read the present value of each named channel once.

    A channel that cannot be read makes the whole read `refused` with the reason; a value is
    never made up.
    """

    channels: tuple[str, ...]
    cite: Citation | None = None

    def check(self) -> None:
        if not self.channels:
            raise ValueError("ReadChannels.channels: name at least one channel")
        _aliases(self.channels, "ReadChannels.channels")


@dataclass(frozen=True)
class ParkTip(Part):
    """Park: make the instrument safe (tip off the surface, XY still, tip voltage zero)."""

    reason: str
    cite: Citation | None = None

    def check(self) -> None:
        if not self.reason.strip():
            raise ValueError("ParkTip.reason: say why the instrument is parking")


# --- bundles: one command is one of today's whole jobs -------------------------


@dataclass(frozen=True, eq=False)
class WriteObject(Part):
    """Write one design object: today's write_object cycle, on both plugins.

    Hover (same `source` as the previous object) or transit to the first point, a free-air
    read after a transit, engage at `setpoint`, fire the points (tip follows (x, y) at
    `speed_um_s` with `tip_v` at each point), back-off withdraw. One setpoint per object
    (2026-10-26). `steps` cites each run step the cycle carries out, in order; the instrument
    sends one status message per step. The result's readings hold the free-air read it took.
    `chained` is set from the run's `LithoObject.fire == "LITHOCHAIN"`; the twin polls telemetry
    at 10 Hz for it, and the Asylum plugin ignores it (Igor decides the rate).
    `transit_speed_um_s` is the XY speed of the move to the first point; the twin needs it
    (it has no speed of its own), the Asylum plugin ignores it. Equality is not defined
    because the points are arrays.
    """

    name: str
    x_um: np.ndarray
    y_um: np.ndarray
    tip_v: np.ndarray
    speed_um_s: float
    setpoint: Setpoint
    record: tuple[str, ...]
    source: str | None = None
    steps: tuple[Citation, ...] = ()
    chained: bool = False
    transit_speed_um_s: float | None = None

    def __post_init__(self) -> None:
        for name in ("x_um", "y_um", "tip_v"):
            object.__setattr__(self, name, _points(getattr(self, name), f"WriteObject.{name}"))
        super().__post_init__()

    def check(self) -> None:
        if not self.name:
            raise ValueError("WriteObject.name: an object has a name")
        quotable(self.name, "WriteObject.name")
        if len(self.x_um) < 2:
            raise ValueError("WriteObject: an object has at least two points")
        if not len(self.x_um) == len(self.y_um) == len(self.tip_v):
            raise ValueError("WriteObject: x_um, y_um and tip_v must have one value per point")
        _positive(self.speed_um_s, "WriteObject.speed_um_s")
        _transit_speed(self.transit_speed_um_s, "WriteObject")
        if not self.record:
            raise ValueError("WriteObject.record: a write records at least one channel")
        _aliases(self.record, "WriteObject.record")
        _one_run_in_order(self.steps, "WriteObject.steps")


@dataclass(frozen=True)
class ScanRaster(Part):
    """Take one scan of `frame`: today's chained scan, on both plugins.

    Transit to the first point (re-engaging there if the tip was engaged and the scan does
    not engage itself), reconcile the imaging mode, engage at `setpoint` when `engage` is
    set, then the raster at the frame's line rate, trace and retrace as the frame says.
    `steps` cites each run step it carries out, in order.
    """

    frame: ScanFrame
    mode: Mode
    setpoint: Setpoint
    engage: bool
    record: tuple[str, ...]
    steps: tuple[Citation, ...] = ()
    transit_speed_um_s: float | None = None

    def check(self) -> None:
        _transit_speed(self.transit_speed_um_s, "ScanRaster")
        if not self.record:
            raise ValueError("ScanRaster.record: a scan records at least one channel")
        _aliases(self.record, "ScanRaster.record")
        _one_run_in_order(self.steps, "ScanRaster.steps")


Command = MoveTo | EngageAt | WithdrawTip | ReadChannels | ParkTip | WriteObject | ScanRaster


# --- results -------------------------------------------------------------------


@dataclass(frozen=True)
class Reading(Part):
    """One channel's value, read by the instrument. Never NaN: a failed read is refused."""

    channel: str
    volts: float

    def check(self) -> None:
        _aliases((self.channel,), "Reading.channel")


@dataclass(frozen=True, eq=False)
class Recorded:
    """One channel captured during a write or a scan, in volts.

    A write gives one value per point (`part` "write"); a scan gives one row per line for
    each of "trace" and "retrace". Measured data may hold NaN, so values are not checked
    finite.
    """

    channel: str
    part: Literal["write", "trace", "retrace"]
    volts: np.ndarray

    def __post_init__(self) -> None:
        _aliases((self.channel,), "Recorded.channel")
        if self.part not in ("write", "trace", "retrace"):
            raise ValueError(f"Recorded.part: {self.part!r} is not write, trace or retrace")
        array = np.array(self.volts, dtype=float)
        if array.ndim != (1 if self.part == "write" else 2):
            raise ValueError("Recorded.volts: a write is 1-D, a scan part is lines x points")
        if array.size == 0:
            raise ValueError("Recorded.volts: no values")
        array.setflags(write=False)
        object.__setattr__(self, "volts", array)


@dataclass(frozen=True, eq=False)
class Result:
    """How a command ended, in instrument terms.

    done: it finished. refused: the instrument did not start it (the reason says why) and is
    as it was. aborted: it stopped part way (the reason says why); the instrument then makes
    itself safe as it does today.

    `verified` is required for a park and a withdraw and absent for every other verb, so an
    unverified park or withdraw can never read as safe. `critical` is today's park flag (the
    park could not be verified and needs the operator) and is given for a park only.
    `landed` is given only on an aborted write whose pass was fired: the fraction (0 to 1) of
    the pass the controller had captured when it stopped; `recorded` then holds that prefix.
    """

    verb: Verb
    outcome: Outcome
    reason: str = ""
    verified: bool | None = None
    critical: bool | None = None
    readings: tuple[Reading, ...] = ()
    recorded: tuple[Recorded, ...] = ()
    landed: float | None = None

    def __post_init__(self) -> None:
        if self.verb not in VERBS:
            raise ValueError(f"Result.verb: {self.verb!r} is not one of {VERBS}")
        if self.outcome not in ("done", "refused", "aborted"):
            raise ValueError(f"Result.outcome: {self.outcome!r} is not done, refused or aborted")
        if not (isinstance(self.reason, str) and self.reason.isascii()
                and self.reason.isprintable()):
            raise ValueError("Result.reason must be typeable ASCII")
        if self.outcome != "done" and not self.reason.strip():
            raise ValueError(f"Result: a {self.outcome} result says why")
        self._check_safety_flags()
        if self.landed is not None and not (
                self.verb == "write" and self.outcome == "aborted" and 0.0 <= self.landed <= 1.0):
            raise ValueError("Result.landed: a fraction from 0 to 1, given only on an aborted write")
        readings = tuple(self.readings)
        if not all(isinstance(r, Reading) for r in readings):
            raise ValueError("Result.readings: expected Reading values")
        _aliases(tuple(r.channel for r in readings), "Result.readings")
        recorded = tuple(self.recorded)
        if not all(isinstance(r, Recorded) for r in recorded):
            raise ValueError("Result.recorded: expected Recorded values")
        object.__setattr__(self, "readings", readings)
        object.__setattr__(self, "recorded", recorded)

    def _check_safety_flags(self) -> None:
        if self.verb in ("park", "withdraw"):
            if not isinstance(self.verified, bool):
                raise ValueError(f"Result: a {self.verb} result says whether it was verified")
        elif self.verified is not None:
            raise ValueError(f"Result.verified: not given for a {self.verb}")
        if self.verb == "park":
            if not isinstance(self.critical, bool):
                raise ValueError("Result: a park result says whether it is critical")
        elif self.critical is not None:
            raise ValueError(f"Result.critical: not given for a {self.verb}")

    @property
    def done(self) -> bool:
        return self.outcome == "done"

    def reading(self, channel: str) -> float:
        """The value read for `channel`; KeyError if it was not read."""
        for r in self.readings:
            if r.channel == channel:
                return r.volts
        raise KeyError(channel)
