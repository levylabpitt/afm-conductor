"""The run contract, version 1: a recipe (header + numbered body) and one execution of it.

A run is a series of passes with the prep and cleanup steps between them. A pass is what the
user asked the AFM to do (one litho object, one scan); steps are what the system adds to get
there and to finish clean. The recipe is everything that says what to do and is fingerprinted;
the execution is this one go at it (its id, start time and the files it wrote). Outcomes (free
air read, points played, aborts) are not here: they go to the journal, cited by (execution, n).

Every part is a frozen dataclass checked on construction against its own type hints, so a
builder cannot leave a setpoint without its reference, put lambda in a name, or pass 150 where
150.0 is meant (the int would change the fingerprint of an identical recipe).
"""

from __future__ import annotations

import math
import re
import types
import typing
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import ClassVar, Literal

VERSION = 1

RunKind = Literal["write", "scan"]
Platform = Literal["cypher", "mfp3d"]
Mode = Literal["contact", "ac"]

# The ARC has in-wave banks 0, 1 and 2 only; naming a bank above 2 breaks the Igor program.
IN_BANKS = (0, 1, 2)

EXECUTION_ID = re.compile(r"^[ws]-\d{4}-\d{2}-\d{2}-\d{4}-\d{2}$")
HEX = re.compile(r"^[0-9a-f]{8,64}$")


# --- checking a part against its type hints --------------------------------


def _check_value(hint, value, where: str):
    """Return `value` checked against the type hint `hint`, normalised for the fingerprint.

    For example, with hint `tuple[float, float]` the value `[-10, 150]` comes back as
    `(-10.0, 150.0)`, and `["a", 1]` raises ValueError naming `where`. Normalising matters
    because the fingerprint hashes the JSON: 150 and 150.0, or [] and (), must hash the same.
    """
    origin = typing.get_origin(hint)
    if origin is Literal:
        # Matched on type too: True == 1 and 1.0 == 1 in Python, but not in the JSON.
        if not any(value == a and type(value) is type(a) for a in typing.get_args(hint)):
            raise ValueError(f"{where}: {value!r} is not one of {typing.get_args(hint)}")
        return value
    if origin in (typing.Union, types.UnionType):
        # The only unions in the contract are `X | None`.
        options = typing.get_args(hint)
        if value is None and type(None) in options:
            return None
        (inner,) = [o for o in options if o is not type(None)]
        return _check_value(inner, value, where)
    if origin is tuple:
        # JSON gives lists; parts hold tuples so they stay frozen and hashable.
        if not isinstance(value, (tuple, list)):
            raise ValueError(f"{where}: expected a list, got {value!r}")
        args = typing.get_args(hint)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_check_value(args[0], v, f"{where}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ValueError(f"{where}: expected {len(args)} values, got {len(value)}")
        return tuple(_check_value(a, v, f"{where}[{i}]")
                     for i, (a, v) in enumerate(zip(args, value)))
    if hint is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{where}: expected true or false, got {value!r}")
        return value
    if hint is int:
        # bool is a subclass of int; True is not a step number.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where}: expected a whole number, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where}: expected a number, got {value!r}")
        # JSON has no NaN or infinity, and a NaN setpoint or limit means nothing.
        if not math.isfinite(value):
            raise ValueError(f"{where}: {value!r} is not finite")
        # Adding 0.0 turns -0.0 into 0.0, which would otherwise fingerprint differently.
        return float(value) + 0.0
    if hint is str:
        # Names must be typeable: no Greek letters, no tabs or newlines.
        if not isinstance(value, str):
            raise ValueError(f"{where}: expected text, got {value!r}")
        if not (value.isascii() and value.isprintable()):
            raise ValueError(f"{where}: {value!r} must be typeable ASCII")
        return value
    if isinstance(value, hint):  # a nested part, already checked when it was built
        return value
    raise ValueError(f"{where}: expected {hint.__name__}, got {type(value).__name__}")


def quotable(name: str, where: str) -> None:
    """The text form prints names in double quotes, so a name may not contain one."""
    if '"' in name:
        raise ValueError(f"{where}: {name!r} may not contain a double quote")


class Part:
    """Base of every run part: checks and normalises its fields after construction."""

    def __post_init__(self) -> None:
        hints = typing.get_type_hints(type(self))
        for f in fields(self):
            value = _check_value(hints[f.name], getattr(self, f.name),
                                 f"{type(self).__name__}.{f.name}")
            object.__setattr__(self, f.name, value)
        self.check()

    def check(self) -> None:
        """Rules across fields; parts override it."""


# --- header parts ------------------------------------------------------------


@dataclass(frozen=True)
class Setpoint(Part):
    """A setpoint always says what it is measured from; there is no default reference."""

    reference: Literal["relative", "absolute"]
    volts: float


@dataclass(frozen=True)
class Placement(Part):
    """Where a write's design sits on the sample, and where that frame came from."""

    source: str
    size_um: float
    angle_deg: float
    centre_um: tuple[float, float]


@dataclass(frozen=True)
class ScanFrame(Part):
    """A scan's raster as requested. `pixels` is points per line in the image (trace only)."""

    source: str
    size_um: float
    lines: int
    pixels: int
    angle_deg: float
    offset_um: tuple[float, float]
    drift_offset: bool
    direction: Literal["up", "down"]
    line_rate_hz: float
    trace_and_retrace: bool


@dataclass(frozen=True)
class Design(Part):
    """The design a write was compiled from, linked by fingerprints, never by point arrays."""

    name: str
    rev: int
    objects: int
    snapshot_fingerprint: str
    compile_settings_fingerprint: str
    compiler_version: str

    def check(self) -> None:
        quotable(self.name, "Design.name")
        for name in ("snapshot_fingerprint", "compile_settings_fingerprint"):
            if not HEX.match(getattr(self, name)):
                raise ValueError(f"Design.{name} must be lowercase hex")


@dataclass(frozen=True)
class Bias(Part):
    """Tip bias: the amplifier gain between the ARC output and the tip, and the allowed window."""

    amp_gain: float
    window_v: tuple[float, float]


@dataclass(frozen=True)
class Capture(Part):
    """Channels every pass records, by Asylum alias. `pair` shares one in-bank; when that bank
    cannot take the pair, each entry of `pair_fallbacks` is tried in turn (empty = record none)."""

    channels: tuple[str, ...]
    pair: tuple[str, ...] = ()
    pair_bank: int | None = None
    pair_fallbacks: tuple[tuple[str, ...], ...] = ()

    def check(self) -> None:
        if self.pair and self.pair_bank not in IN_BANKS:
            raise ValueError(f"Capture.pair_bank must be one of {IN_BANKS}")
        if not self.pair and (self.pair_bank is not None or self.pair_fallbacks):
            raise ValueError("Capture: pair_bank and pair_fallbacks need a pair")


# --- body: steps (prep and cleanup) -------------------------------------------


class Entry(Part):
    """One numbered line of a run's body. `KIND` is its name in the JSON and the text."""

    KIND: ClassVar[str]


class Step(Entry):
    """Prep or cleanup the system adds around the passes."""


class Pass(Entry):
    """Work the user asked the AFM to do."""


@dataclass(frozen=True)
class Check(Step):
    """Preconditions tested before anything moves, by name (e.g. `sum-below-ceiling`)."""

    KIND: ClassVar[str] = "check"
    items: tuple[str, ...]


@dataclass(frozen=True)
class Read(Step):
    """Pre-engage reads in free air: only the channels this work needs. The values read are
    outcomes and go to the journal."""

    KIND: ClassVar[str] = "read"
    channels: tuple[str, ...]
    agreeing: int | None = None
    z_drive_max_v: float | None = None


@dataclass(frozen=True)
class Move(Step):
    """XY travel between passes. `how` is open-ended (lifted, transit, hover, contour, ...) and
    always stated by the builder."""

    KIND: ClassVar[str] = "move"
    how: str
    to: str
    withdraw_first: Literal["none", "verified", "verified-if-engaged"] = "verified"
    hold_v: float | None = None
    speed_um_s: float | None = None
    settle_s: float | None = None


@dataclass(frozen=True)
class Settle(Step):
    """Wait with nothing newly commanded; `why` says when the rule applies."""

    KIND: ClassVar[str] = "settle"
    seconds: float
    why: str = ""


@dataclass(frozen=True)
class SetMode(Step):
    """Switch the imaging mode; `when` states the condition the code applies."""

    KIND: ClassVar[str] = "setmode"
    mode: Mode
    when: str = ""
    settle_s: float | None = None


@dataclass(frozen=True)
class Engage(Step):
    """Close the Z loop at a setpoint: loop `loop` reads `on` and drives `out` within limits."""

    KIND: ClassVar[str] = "engage"
    setpoint: Setpoint
    loop: int
    on: str
    out: str
    limits_v: tuple[float, float]
    sign: Literal[-1, 1] | None = None
    settle_s: float | None = None
    when: str = ""


@dataclass(frozen=True)
class Withdraw(Step):
    """`back-off` drops the Z setpoint after a pass; `verified` withdraws and checks the tip
    left the surface. `hold_v` is where the Z drive is held (the Height alias on the Cypher)."""

    KIND: ClassVar[str] = "withdraw"
    how: Literal["back-off", "verified"]
    hold_v: float
    setpoint: Setpoint | None = None
    settle_s: float | None = None
    why: str = ""


@dataclass(frozen=True)
class End(Step):
    """The state a run leaves the tip in when it does not finish with a withdraw."""

    KIND: ClassVar[str] = "end"
    engaged: bool


# --- body: passes ---------------------------------------------------------------


@dataclass(frozen=True)
class Param(Part):
    """One named shape parameter, in typeable units (um, nm, deg, V, s)."""

    name: str
    value: float
    unit: str


@dataclass(frozen=True)
class LithoObject(Pass):
    """One design object written in one fire. More than one chunk means a chained write with
    the bias left on across each seam. `points_fingerprint` hashes the fired points in
    micrometres (after padding to a multiple of 32, before the LVDT and amp-gain conversion,
    which depend on the calibration at the moment of the move). `chunk_cuts` are the indices
    where the chain is cut: empty for one chunk, `chunks - 1` of them otherwise."""

    KIND: ClassVar[str] = "litho-object"
    index: int
    name: str
    shape: str
    points: int
    chunks: int
    fire: Literal["ARMFIRE", "LITHOCHAIN"]
    ends_at_v: float
    wave_line_points: int
    points_fingerprint: str
    params: tuple[Param, ...] = ()
    speed_um_s: float | None = None
    chunk_cuts: tuple[int, ...] = ()

    def check(self) -> None:
        quotable(self.name, "LithoObject.name")
        chained = self.chunks > 1
        if self.chunks < 1 or (self.fire == "LITHOCHAIN") != chained:
            raise ValueError("LithoObject: one chunk fires ARMFIRE, several fire LITHOCHAIN")
        if len(self.chunk_cuts) != self.chunks - 1:
            raise ValueError("LithoObject: chunk_cuts needs exactly chunks - 1 cuts")
        if list(self.chunk_cuts) != sorted(set(self.chunk_cuts)) or any(
                not 0 < c < self.points for c in self.chunk_cuts):
            raise ValueError("LithoObject: chunk_cuts must increase and fall inside the stroke")
        if not HEX.match(self.points_fingerprint):
            raise ValueError("LithoObject.points_fingerprint must be lowercase hex")


@dataclass(frozen=True)
class Scan(Pass):
    """One raster scan. `lines` must agree with the frame. `points_per_line` is what is played
    per line (trace + retrace), set by the scan code, not derived here from the image pixels.
    `decimation` and `error_channel` are the other SCANCHAIN arguments: ticks per captured
    sample, and the mode's error channel paired with ZSensor."""

    KIND: ClassVar[str] = "scan"
    lines: int
    points_per_line: int
    interpolation: int
    decimation: int
    error_channel: str
    xy_loops: tuple[int, int]
    xy_outputs: tuple[str, str]
    limits_v: tuple[float, float]
    wave_line_points: int
    fire: Literal["SCANCHAIN"] = "SCANCHAIN"


ENTRY_KINDS: dict[str, type[Entry]] = {
    c.KIND: c for c in (Check, Read, Move, Settle, SetMode, Engage, Withdraw, End,
                        LithoObject, Scan)
}


# --- the recipe, the execution, the run ---------------------------------------


@dataclass(frozen=True)
class Recipe(Part):
    """What to do. Its canonical JSON is what the recipe fingerprint hashes. Body entries are
    numbered by position from 0. A write carries placement, design and bias; a scan a frame.
    `setpoint` is the run's stated request; every engage in the body must use exactly it."""

    kind: RunKind
    platform: Platform
    mode: Mode
    setpoint: Setpoint
    capture: Capture
    script_gone: Literal["park"]
    body: tuple[Entry, ...]
    placement: Placement | None = None
    frame: ScanFrame | None = None
    design: Design | None = None
    bias: Bias | None = None
    script_gone_note: str = ""
    version: int = VERSION

    def check(self) -> None:
        if self.version != VERSION:
            raise ValueError(f"Recipe.version {self.version} is not {VERSION}")
        passes = [e for e in self.body if isinstance(e, Pass)]
        if not passes:
            raise ValueError("Recipe.body has no pass")
        if self.kind == "write":
            ok = (self.placement and self.design and self.bias and self.frame is None
                  and all(isinstance(p, LithoObject) for p in passes))
        else:
            ok = (self.frame and self.placement is None and self.design is None
                  and self.bias is None and all(isinstance(p, Scan) for p in passes))
        if not ok:
            raise ValueError(f"Recipe: header or passes do not fit a {self.kind} run")
        for n, e in enumerate(self.body):
            if isinstance(e, Engage) and e.setpoint != self.setpoint:
                raise ValueError(f"Recipe: engage at step {n} does not use the run's setpoint")
        if self.design:
            indexes = [p.index for p in passes]
            if len(set(indexes)) != len(indexes) or not all(
                    0 <= i < self.design.objects for i in indexes):
                raise ValueError("Recipe: litho objects must be distinct objects of the design")
        if self.frame and any(p.lines != self.frame.lines for p in passes):
            raise ValueError("Recipe: scan lines do not match the frame")


@dataclass(frozen=True)
class StepFile(Part):
    """A file one step saved, cited by the step's number."""

    step: int
    path: str


@dataclass(frozen=True)
class Execution(Part):
    """One go at a recipe: its id, start time and the files it saved. Writes save a float32
    commanded X/Y file per litho pass that was fired (a pass cut short by an abort has
    "partial" in its file name); scans save none (the raster rebuilds exactly). `captures`
    are the measured-data files of the whole run (npz, and gwy/ibw where written).

    `started` is ISO 8601 with a UTC offset; the id carries the same minute in UTC (see ids.py).
    """

    execution_id: str
    started: str
    design_snapshot: str | None = None
    compile_settings: str | None = None
    commanded_xy: tuple[StepFile, ...] = ()
    captures: tuple[str, ...] = ()

    def check(self) -> None:
        if not EXECUTION_ID.match(self.execution_id) or self.execution_id.endswith("-00"):
            raise ValueError(f"Execution.execution_id {self.execution_id!r}"
                             " is not k-YYYY-MM-DD-HHMM-NN with NN from 01")
        try:
            started = datetime.fromisoformat(self.started)
        except ValueError:
            raise ValueError(f"Execution.started {self.started!r} is not ISO 8601") from None
        if started.utcoffset() is None:
            raise ValueError("Execution.started must carry a UTC offset")
        minute = started.astimezone(timezone.utc).strftime("%Y-%m-%d-%H%M")
        if self.execution_id[2:-3] != minute:
            raise ValueError(f"Execution: id minute is not the start minute in UTC ({minute})")


@dataclass(frozen=True)
class Run(Part):
    """One recipe and one execution of it."""

    recipe: Recipe
    execution: Execution

    def check(self) -> None:
        if self.execution.execution_id[0] != self.recipe.kind[0]:
            raise ValueError("Run: execution id letter does not match the run kind")
        for f in self.execution.commanded_xy:
            if not (0 <= f.step < len(self.recipe.body)
                    and isinstance(self.recipe.body[f.step], LithoObject)):
                raise ValueError(f"Run: commanded X/Y file cites step {f.step}, not a litho pass")

    def cite(self, n: int) -> dict:
        """Journal fields that tie an outcome to step `n` of this execution."""
        if not 0 <= n < len(self.recipe.body):
            raise ValueError(f"step {n} is not in this run")
        return {"run": self.execution.execution_id, "step": n}
