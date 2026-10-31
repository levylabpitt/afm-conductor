"""The optional Deflection check that a litho set-down has engaged before anything fires.

Rig, Cypher (2026-10-28): at the start of an object the bias came on and XY moved while
the integral-only loop 2 was still bringing the tip down. The default fix is the longer fixed
settle (lithography_run.SETTLE_MOVE_S). This check is the measured alternative: it stays OFF
until it is proven on the rig, and lives here alone so it can be deleted in one piece.

Switched on with AFM_LITHO_CONFIRM_ENGAGE=1. When on, LithographyRun:
- refuses an object before its set-down when it has no agreed free-air read, or when the
  setpoint is within MIN_ENGAGE_STEP_V of free air (the engage cannot be seen over noise);
- after the set-down, loads and fires nothing until Deflection has gone ENGAGE_FRACTION of the
  way from free air to the setpoint, ENGAGE_READS reads in a row; a timeout raises
  EngageNotConfirmed and the caller's failure path does the verified withdraw.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable

from .. import clock
from .hw_approach import ENGAGE_SETTLE_S, ENGAGE_TOL_V

ENV = "AFM_LITHO_CONFIRM_ENGAGE"

# One-sided: a free-air read never counts, whatever the step (2026-10-28).
ENGAGE_FRACTION = 0.9
ENGAGE_READS = 3
ENGAGE_POLL_S = 0.05
ENGAGE_TIMEOUT_S = ENGAGE_SETTLE_S          # hw_engage's own poll limit, 6 s
# The usual relative step is 0.1 V; under 0.05 V (hw_engage's "reached" tolerance) the
# engage cannot be told from noise.
MIN_ENGAGE_STEP_V = ENGAGE_TOL_V


def from_env(environ=None) -> bool:
    """On only for "1"; unset or empty is off; anything else is refused with a clear message."""
    raw = (os.environ if environ is None else environ).get(ENV, "").strip()
    if raw in ("", "0"):
        return False
    if raw == "1":
        return True
    raise ValueError(f'{ENV} must be "0" or "1", not {raw!r}')


# The switch, read once at import. It exists because the fixed 1.0 s settle is the shipped
# fix and this measured check is not yet trusted: off until it is proven on the rig.
ENABLED = from_env()


@contextlib.contextmanager
def use(enabled: bool):
    """Set ENABLED for the block (tests), then restore it."""
    global ENABLED
    before = ENABLED
    ENABLED = bool(enabled)
    try:
        yield
    finally:
        ENABLED = before


class EngageNotConfirmed(RuntimeError):
    """After the set-down, Deflection did not reach the setpoint within ENGAGE_TIMEOUT_S:
    nothing was loaded or fired for this object. The caller's failure path does the verified
    withdraw (Asylum._write_stopped -> _make_safe, or the worker's end-of-run withdraw)."""


def engage_progress(deflection_v: float, free_air_v: float, setpoint_v: float) -> float:
    """How far Deflection has gone from free air toward the setpoint: 0 in free air, 1 at the
    setpoint, negative the wrong way, NaN for a NaN read. `step_refusal` keeps the step at or
    above MIN_ENGAGE_STEP_V, so the division is safe."""
    return (deflection_v - free_air_v) / (setpoint_v - free_air_v)


def step_refusal(name: str, setpoint_v: float, free_air_v: float) -> str | None:
    """Why the write must be refused before the set-down, or None when the step is big enough."""
    if abs(setpoint_v - free_air_v) >= MIN_ENGAGE_STEP_V:
        return None
    return (f"write {name}: setpoint {setpoint_v:.4f} V is within {MIN_ENGAGE_STEP_V:g} V of "
            f"free air {free_air_v:.4f} V, so the engage cannot be seen; write refused "
            "before the set-down")


def confirm_engage(name: str, setpoint_v: float, free_air_v: float, *,
                   read: Callable[[], float], check_abort: Callable[[], None],
                   wait: Callable[[float], None]) -> float:
    """Poll `read` (Deflection, NaN on a failed read) until ENGAGE_READS reads in a row have
    gone ENGAGE_FRACTION of the step; return the last read. A short or NaN read restarts the
    count. `check_abort` and `wait` raise on an abort; nothing here sends a command."""
    deadline = clock.Deadline(ENGAGE_TIMEOUT_S)
    engaged_reads = 0
    while True:
        check_abort()
        defl = read()
        far_enough = engage_progress(defl, free_air_v, setpoint_v) >= ENGAGE_FRACTION
        engaged_reads = engaged_reads + 1 if far_enough else 0
        if engaged_reads >= ENGAGE_READS:
            return defl
        if deadline.expired():
            raise EngageNotConfirmed(
                f"write {name}: tip did not engage within {ENGAGE_TIMEOUT_S:g} s "
                f"(deflection {defl:.4f} V, setpoint {setpoint_v:.4f} V, "
                f"free air {free_air_v:.4f} V); nothing fired")
        wait(ENGAGE_POLL_S)
