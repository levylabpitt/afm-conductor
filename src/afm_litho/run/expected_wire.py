"""The wire a stored run says should be sent, as an ordered list of facts.

Pure: no I/O, no hub, nothing sent. Not called by the service. It is the oracle for the (run,
wire) check: a test projects a stored `Run` with `expected_wire` and compares the result with
the same facts read off a recorded wire (`tests/wire_facts.py`), using `differences`.

A fact is a tuple whose first item names it. `None` inside an expected fact means the run does
not claim that value; `differences` skips it. In order, per body step:

  ("withdraw", setpoint_v, out_min, out_max)
                                         verified withdraw: loop 2 at the retract setpoint
                                         (RETRACT_SETPOINT_V), output held at out_min V (0 V
                                         on the Cypher) up to out_max (the loop output limit).
                                         Emitted before a move whose `withdraw_first` is
                                         "verified", and for a "verified" Withdraw step. The
                                         setpoint and out_max are the code's constants, not
                                         values the run holds.
  ("move", "transit" | "hover")          XY travel: a FIRE of the transit waves, or a RAMP of
                                         the XY setpoints with the tip down.
  ("engage", loop, on, out, setpoint_v, out_min, out_max, gain_sign)
                                         the loop-2 line that sets the tip down.
  ("back-off", setpoint_v, out_min)      the loop-2 line after a pass.
  ("litho", fire, name, points, wave_lines, cap_len, bounds)
                                         ARMFIRE or LITHOCHAIN of one stroke: its X wave is
                                         `points` long, loaded in `wave_lines` lines; ARMFIRE
                                         captures `cap_len` points (None for a chain);
                                         `bounds` is (n, sha) of the chunk-cut wave, None for
                                         ARMFIRE.
  ("scan", fire, points_per_line, lines, interpolation, decimation, error_channel,
   wave_points, wave_lines, xy_loops)    SCANCHAIN and its arguments; `xy_loops` is
                                         ((loop, output, out_min, out_max), ...) of the loops
                                         armed for the raster.

What it checks: the order of these facts and every value named above.

What it does not claim, because the run does not hold it: free-air readings (a relative
engage setpoint is claimed only when the caller passes `free_air_v`), the live start point of
a transit and so the length of its waves, the LVDT and amp-gain conversion of the stroke's
points, a litho stroke's interpolation and decimation (they follow from the path's spacing,
which the run keeps only as a fingerprint), settle times, the Check, Read and SetMode steps
(no wire line, or conditional on a run-time read), and what a park or abort sends.

Also unclaimed: a move whose `withdraw_first` is "verified-if-engaged" projects like "none",
because the run does not say whether the tip was engaged when the move began. So a scan's
withdraw before its move, and the transit's own re-engage after it, are unchecked until a
golden starts a scan engaged (docs/migration/SLICES.md, F57).

An abort ends the wire early: the recorded facts are then a prefix of the expected ones, which
the caller states.
"""

from __future__ import annotations

import math
from math import ceil

import numpy as np

from ..goldens.expander import summarise
from ..hardware import channels
from ..hardware.lithography_run import RETRACT_SETPOINT_V
from .model import (Check, End, Engage, LithoObject, Move, Read, Run, Scan, SetMode, Settle,
                    Step, Withdraw)

Fact = tuple

# A chunk-cut wave is summarised the way the goldens summarise every wave: length and a hash
# of its float32 values.
BOUNDS_WAVE = "LithoBounds"

# Steps that send nothing a golden records.
_SILENT = (Check, Read, Settle, SetMode, End)

# The verified withdraw's upper output limit: the loop output limit shared with the engage.
_WITHDRAW_OUT_MAX_V = channels.LOOP_OUTPUT_LIMITS_V[1]


def expected_wire(run: Run, *, free_air_v: float | None = None) -> list[Fact]:
    """The facts `run` says the wire carries, in order. `free_air_v` is the free-air
    deflection the write read, which turns a relative engage setpoint into an absolute one."""
    facts: list[Fact] = []
    for step in run.recipe.body:
        facts += _step_facts(step, free_air_v)
    return facts


def collapse_repeats(facts: list[Fact]) -> list[Fact]:
    """`facts` without a withdraw or a transit that repeats the fact before it. A repeated
    transit is its retry after landing short, and a repeated withdraw is the same withdraw sent
    twice (a cockpit withdraw just before a write, a failure handler's after one); the run
    states neither, so both sides of the check collapse them. A repeated hover is two moves.
    Two consecutive transit moves would collapse too; that cannot happen today, because each
    transit of a run is preceded by a verified withdraw."""
    out: list[Fact] = []
    for fact in facts:
        repeatable = fact[0] == "withdraw" or fact == ("move", "transit")
        if not (out and repeatable and fact == out[-1]):
            out.append(fact)
    return out


def _step_facts(step: Step, free_air_v: float | None) -> list[Fact]:
    if isinstance(step, Move):
        return _move_facts(step)
    if isinstance(step, Engage):
        return [_engage_fact(step, free_air_v)]
    if isinstance(step, Withdraw):
        if step.how == "back-off":
            sp = step.setpoint
            volts = sp.volts if sp and sp.reference == "absolute" else None
            return [("back-off", volts, step.hold_v)]
        return [_withdraw_fact(step.hold_v)]
    if isinstance(step, LithoObject):
        return [_litho_fact(step)]
    if isinstance(step, Scan):
        return [_scan_fact(step)]
    if isinstance(step, _SILENT):
        return []
    raise ValueError(f"expected_wire: no rule for a {type(step).__name__} step")


def _move_facts(move: Move) -> list[Fact]:
    kind = move.how.split(",")[0].strip()
    if kind not in ("transit", "hover"):
        raise ValueError(f"expected_wire: no rule for a move of kind {kind!r}")
    withdraw = [_withdraw_fact(move.hold_v)] if move.withdraw_first == "verified" else []
    return [*withdraw, ("move", kind)]


def _withdraw_fact(out_min: float | None) -> Fact:
    return ("withdraw", RETRACT_SETPOINT_V, out_min, _WITHDRAW_OUT_MAX_V)


def _engage_fact(step: Engage, free_air_v: float | None) -> Fact:
    sp = step.setpoint
    if sp.reference == "absolute":
        volts = sp.volts
    else:
        volts = None if free_air_v is None else free_air_v + sp.volts
    return ("engage", step.loop, step.on, step.out, volts,
            step.limits_v[0], step.limits_v[1], step.sign)


def _litho_fact(p: LithoObject) -> Fact:
    bounds = None
    cap_len = p.points
    if p.fire == "LITHOCHAIN":
        wave = summarise(BOUNDS_WAVE, np.array([0, *p.chunk_cuts, p.points], dtype="<f4"))
        bounds = (wave.n, wave.sha)
        cap_len = None
    return ("litho", p.fire, p.name, p.points, ceil(p.points / p.wave_line_points), cap_len,
            bounds)


def _scan_fact(p: Scan) -> Fact:
    points = p.points_per_line * p.lines
    loops = tuple((loop, out, p.limits_v[0], p.limits_v[1])
                  for loop, out in zip(p.xy_loops, p.xy_outputs))
    return ("scan", p.fire, p.points_per_line, p.lines, p.interpolation, p.decimation,
            p.error_channel, points, ceil(points / p.wave_line_points), loops)


def _same(expected, recorded) -> bool:
    if expected is None:
        return True
    if isinstance(expected, float) or isinstance(recorded, float):
        return (isinstance(expected, (int, float)) and isinstance(recorded, (int, float))
                and math.isclose(expected, recorded, rel_tol=1e-9, abs_tol=1e-9))
    if isinstance(expected, tuple) and isinstance(recorded, tuple):
        return len(expected) == len(recorded) and all(map(_same, expected, recorded))
    return expected == recorded


def differences(expected: list[Fact], recorded: list[Fact]) -> list[str]:
    """Where `recorded` is not `expected`, one line each (empty when they agree). A `None`
    in an expected fact matches anything; floats match to 1e-9."""
    out = []
    for i, (want, got) in enumerate(zip(expected, recorded)):
        if not _same(want, got):
            out.append(f"fact {i}: run says {want}, wire has {got}")
    if len(expected) != len(recorded):
        out.append(f"{len(expected)} facts expected, {len(recorded)} recorded")
    return out


__all__ = ["BOUNDS_WAVE", "Fact", "collapse_repeats", "differences", "expected_wire"]
