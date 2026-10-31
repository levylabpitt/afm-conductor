"""Journal what each step of a run did, cited by (execution id, step n).

The stored run says what was planned; the journal says what happened. `StepNotes` is built
from the stored Run and maps each outcome (a free-air read, a move, a pass...) to the step number
of that line in the run's body, so the numbering is the run's own. Without a run ("no run: ..."),
outcomes are still noted, without a citation.

`outcome()` and `abort()` are called from the write and scan threads at the point the code
already decided something. They only build a dict and queue it (`journal.note`), never raise,
never wait, and take no lock, so they are safe on a path that ends in a withdraw. A caller whose
note body needs computing passes a function (`make`), which runs inside the same guard.

Only the hardware write and scan emit step notes. The twin's write and scan do not (twin runs
are Phase 4).

Each note is src "step", `what` = the outcome kind, arrival field `run` = execution id, body
`step` = n. The same (run, step) is stamped on the wire entries the sending thread makes while
that step is in progress (`bridge_protocol.wire_step`).
"""

from __future__ import annotations

from ..hardware.bridge_protocol import wire_step
from ..run import Run
from ..run.model import Pass, Withdraw
from . import journal

SRC = "step"

# What a body line is called in a note. A pass is "pass" whatever its kind; a withdraw after a
# pass backs the setpoint off ("withdraw"); the verified one finishes the run ("end").
_END = "end"


def _kind(entry) -> str:
    if isinstance(entry, Pass):
        return "pass"
    if isinstance(entry, Withdraw):
        return "withdraw" if entry.how == "back-off" else _END
    return entry.KIND


def step_groups(run: Run) -> list[dict[str, int]]:
    """The run's body as one group per pass: {kind: step n} for the steps that lead to a pass,
    the pass, and the one line after it (its back-off, or a scan's end). The steps after the
    last such line (a write's closing verified withdraw) form a final group.

    A write's group k holds that object's move, settle, read, engage, pass and withdraw; the
    check is in group 0 and the closing "end" is in the last group.
    """
    groups: list[dict[str, int]] = [{}]
    after_pass = False
    for n, entry in enumerate(run.recipe.body):
        groups[-1][_kind(entry)] = n
        if after_pass:
            groups.append({})
            after_pass = False
        elif isinstance(entry, Pass):
            after_pass = True
    return groups


class StepNotes:
    """Notes for one execution. `kind` is "write" or "scan" (used only in the note body).

    `stroke` is the index of the pass being run (a write sets it before each object; a scan
    has one pass). `_next` is the first step not completed: the one after the last step completed.
    """

    def __init__(self, run: Run | None, kind: str) -> None:
        self.run = run
        self.kind = kind
        self.stroke = 0
        self._groups = step_groups(run) if run is not None else []
        self._next = 0
        self._stamping = False

    # -- mapping ----------------------------------------------------------

    def step_of(self, kind: str) -> int | None:
        """The run's step number for an outcome of `kind` in the pass now running, or None when
        there is no run or its body has no such line."""
        if self.run is None:
            return None
        if kind == "check":
            return self._groups[0].get(kind)
        if kind == _END:
            return next((g[_END] for g in reversed(self._groups) if _END in g), None)
        if self.stroke >= len(self._groups):
            return None
        return self._groups[self.stroke].get(kind)

    # -- noting -----------------------------------------------------------

    def outcome(self, kind: str, make=None, **body) -> None:
        """Note that the step of this `kind` happened, with `body` (and the dict `make()`
        returns, if given) as what it found. A step that did not finish (`aborted`) or was not
        done (`skipped`) is noted but does not count as completed."""
        try:
            if make is not None:
                body = {**make(), **body}
            self._note(kind, self.step_of(kind), body)
        except Exception:  # noqa: BLE001,S110 - a note must never stop the instrument work
            pass

    def abort(self, reason: str) -> None:
        """Note that the run stopped, citing the first step not yet completed."""
        try:
            step = self._next if self.run is not None else None
            during = None
            if step is not None and step < len(self.run.recipe.body):
                during = _kind(self.run.recipe.body[step])
            self._note("abort", step, {"reason": reason, "during": during})
        except Exception:  # noqa: BLE001,S110 - as above
            pass

    def stamp_step(self, kind: str) -> None:
        """Mark the step of this `kind` as the one in progress, before its commands are sent."""
        try:
            step = self.step_of(kind)
            if step is not None:
                self._next = step
                self._stamp()
        except Exception:  # noqa: BLE001,S110 - as above
            pass

    def begin(self) -> None:
        """Start stamping wire entries made by the calling thread or task with the step in
        progress. Call it in the thread or task that sends the run's commands, not before."""
        self._stamping = True
        self._stamp()

    def _note(self, what: str, step: int | None, body: dict) -> None:
        fields: dict = {"run_kind": self.kind}
        run_id = None
        if self.run is not None:
            run_id = self.run.execution.execution_id
            if step is not None:
                cite = self.run.cite(step)
                run_id, fields["step"] = cite["run"], cite["step"]
        journal.running().note(SRC, what, run=run_id, **fields, **body)
        if step is not None and what != "abort" and not (
                body.get("aborted") or body.get("skipped")):
            self._next = step + 1
            self._stamp()

    def _stamp(self) -> None:
        """Mark the step now in progress for wire entries made by this thread or task; clear
        it after the run's last step."""
        if self.run is None or not self._stamping:
            return
        last = len(self.run.recipe.body) - 1
        if self._next > last:
            wire_step.set(None)
        else:
            wire_step.set((self.run.execution.execution_id, self._next))


class _NoNotes:
    """Stands in when a caller has no notes, so it never has to check."""

    def outcome(self, *args, **body) -> None:
        return None

    abort = stamp_step = begin = outcome

    def step_of(self, kind: str) -> None:
        return None


NO_NOTES = _NoNotes()


def make_notes(run: Run | None, kind: str) -> StepNotes:
    """Notes for `run`. If they cannot be built from it, uncited notes: the write or scan
    goes on exactly as without them."""
    try:
        return StepNotes(run, kind)
    except Exception:  # noqa: BLE001 - building notes must never stop the instrument work
        return StepNotes(None, kind)
