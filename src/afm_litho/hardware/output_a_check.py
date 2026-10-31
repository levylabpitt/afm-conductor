"""Before a write moves to its start, stop any feedback loop that drives Output.A.

Output.A is the tip bias. A PIDS loop left driving it (from an earlier session) would
fight the write's bias wave, so the write stops such a loop first, logs it, and goes on.
The check must never block or fail a write: any failed read is logged once as a FAILED
check (never as a pass) and the write carries on. It is not run on an abort or a park; an abort never waits.
"""

from __future__ import annotations

import math
import queue

from . import bridge_protocol as P
from . import channels

# What td_ReadString returns on a bad name or a lost device instead of raising (td reference,
# asylum-td-command-reference.txt ~1455). The .ipf replies ERR for it; an older .ipf passed it on.
_TD_READ_FAILED = "*ERROR*"

# The off switch (2026-10-27): the loop reads are not yet proven on the rig. False means
# Igor sees exactly the commands it saw before this check existed.
CHECK_OUTPUT_A_LOOPS = True

_LOOPS = range(6)            # PIDSLoop.0 .. PIDSLoop.5 (td reference, "PIDSLoop.n Group")
_READ_TIMEOUT_S = 5.0        # one read; a silent Igor must not hold the write for long
_LOOP_RUNNING = 1            # PIDSLoop.<n>.Status: 1 = active


_warned_unverified = False   # the MFP-3D note is logged once per session, not on every write


def stop_loops_driving_output_a(run, log, cypher: bool, abort_requested=lambda: False) -> bool:
    """Stop every running loop whose output is Output.A, then set Output.A to 0 so the zero
    is not overridden. `cypher` picks the Status read form. Returns False when a read
    failed (logged as a FAILED check; the write goes on). Never raises."""
    if not CHECK_OUTPUT_A_LOOPS:
        return True
    try:
        _stop_loops(run, log, cypher, abort_requested)
    except Exception as exc:                       # noqa: BLE001 -- must never fail a write
        log(f"Output.A loop check FAILED (exception), the write goes on: {exc}")
        return False
    return True


def _stop_loops(run, log, cypher: bool, abort_requested) -> None:
    global _warned_unverified
    # Rig fact (Cypher, 2026-10-27): the bare PIDSLoop.<n>.Status reads -1 even while the loop
    # runs; only the Cypher.-prefixed name is true. OutputChannel reads fine with the bare name.
    # The MFP-3D form is unknown: the bare name is tried and logged as unverified.
    status_prefix = "Cypher." if cypher else ""
    if not cypher and not _warned_unverified:
        _warned_unverified = True
        log("Output.A loop check: MFP-3D Status read form is unverified")
    stopped = False
    for n in _LOOPS:
        if abort_requested():
            break
        if _read_number(run, f"{status_prefix}PIDSLoop.{n}.Status") != _LOOP_RUNNING:
            continue
        output = _read_text(run, f"PIDSLoop.{n}.OutputChannel")
        if _is_output_a(output):
            run.ir_StopPISLoop(n)
            stopped = True
            log(f"Output.A loop check: loop {n} was driving {output}; stopped it")
    if stopped:
        run.link.send(P.cmd_setvalue(channels.OUTPUT_A, 0))


def _is_output_a(channel: str) -> bool:
    """Output.A, or a prefixed form such as Cypher.Output.A."""
    return channel.strip().lower().endswith("output.a")


def _read_number(run, name: str) -> float:
    verb, rest = _ask(run, P.cmd_getvalue, name)
    if verb == P.VALUE and math.isfinite(float(rest.strip())):
        return float(rest.strip())
    raise RuntimeError(f"read of {name} failed: {verb} {rest.strip()}")


def _read_text(run, name: str) -> str:
    verb, rest = _ask(run, P.cmd_getstr, name)
    fields = P.parse_xpt(rest) if verb == P.XPT else {}
    if name in fields and fields[name].strip() != _TD_READ_FAILED:
        return fields[name]
    raise RuntimeError(f"read of {name} failed: {verb} {rest.strip()}")


def _ask(run, command, name: str) -> tuple[str, str]:
    """Send one uid-correlated read and return the reply (verb, rest)."""
    waiter = run.router.open_waiter()
    try:
        run.link.send(command(waiter.uid, name))
        return waiter.get(_READ_TIMEOUT_S)
    except queue.Empty:
        raise RuntimeError(f"no reply to read of {name} within {_READ_TIMEOUT_S:g} s") from None
    finally:
        run.router.release(waiter.uid)
