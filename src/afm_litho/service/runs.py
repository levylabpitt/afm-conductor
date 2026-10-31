"""Run-id-keyed TERMINAL RECORDS -- the registry a scripted client fetches results
from (docs/FLEX_INTEGRATION_PLAN.md Sec A.1, F.1).

WHY
---
The service already writes a session-store record per run (`scan-<ms>`, `litho-<ms>`,
`iv-<ms>`) and drops `.npz` / `.gwy` / `.ibw` files beside it. What it did NOT have is
a handle a caller can hold from the moment a run STARTS to the moment its files are on
disk. A FLEX script polls `getState`, sees the run leave the busy state, and opens the
path it was given -- and on hardware the `scan_done` event is broadcast from inside the
scan worker while the router pump writes the files AFTERWARDS, in a best-effort
try/except. Without a record the script races the writer and gets a truncated or absent
file.

So every start verb allocates a monotonic `run_id` and the pump CLOSES the record after
its persistence block:

    run_id = hub.next_run_id("scan")        # at start; returned to the caller
    ...
    hub.runs.finish(run_id, completed=True, files=..., shape=..., channels=...)

`completed` is therefore gated on PERSISTENCE, not on the done event -- it is the one
flag that says "the paths in `files` exist and are whole". `files` lists only paths that
were `stat`-checked; an export that failed is simply ABSENT (never present-but-broken)
and `partial` is set.

NO LOCK, ON PURPOSE
-------------------
This registry is read from the ZMQ REP threads and written from the asyncio loop. Plan
Sec B.4's deadlock rule forbids the REP thread from holding a `threading.Lock` that a
loop coroutine can also acquire -- the REP thread holds the lock, blocks on
`run_coroutine_threadsafe(...).result()`, and the coroutine it waits on takes the same
lock: both wait forever and the tip stays where it is. Rather than reason about whether
this particular lock is ever held across a marshal, there is no lock at all. Every
mutation is a single dict store or an `itertools.count` bump -- atomic under the GIL --
and every read copies before returning. The worst a race can do is order two records
that were allocated in the same instant.
"""

from __future__ import annotations

import itertools
import logging
import os
import time
from dataclasses import asdict, dataclass, field

log = logging.getLogger(__name__)


def on_pass_ended(hub, run_id, outcome, written, files) -> None:
    """The base's end-of-pass handler (HostState.pass_ended calls it): release the
    single-writer claim held by `run_id`, synchronously, then log."""
    hub.release_run(run_id)
    log.debug("pass ended: run=%s outcome=%s written=%s files=%d",
              run_id, outcome, written, len(files))


# Retained terminal records, newest last. Plan Sec F.1: "terminal records for the last
# N runs (default 32) are retained after persistence completes, so a poll arriving late
# still finds its result".
KEEP_RUNS = 32


@dataclass
class RunRecord:
    """One run, from allocation to persistence.

    `kind` is the wire vocabulary -- "scan" / "write" -- which is what
    `getScanResult` / `getWriteResult` select on. `record_id` is
    the session-store id (`scan-<ms>`) that names the files on disk, kept so a result
    can be joined back to the session the GUI shows.
    """

    run_id: str
    kind: str
    started: float
    backend: str = "twin"
    finished: float | None = None
    completed: bool = False          # true ONLY after the persistence block finished
    aborted: bool = False
    partial: bool = False            # an attempted export is missing from `files`
    files: dict[str, str] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    shape: list[int] | None = None
    channels: list[str] = field(default_factory=list)
    record_id: str | None = None
    # The stored run (service/run_store.py): {execution_id, path}, or "no run: <reason>";
    # None when no run was attempted (the twin).
    run_ref: dict | str | None = None
    error: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class RunRegistry:
    """The last `keep` runs, keyed by a process-monotonic id."""

    def __init__(self, keep: int = KEEP_RUNS):
        self._runs: dict[str, RunRecord] = {}      # insertion-ordered = allocation order
        self._seq = itertools.count(1)             # next() is atomic under the GIL
        self._keep = int(keep)

    # ------------------------------------------------------------- allocation --

    def allocate(self, kind: str, *, backend: str = "twin") -> str:
        """Mint the next run id and open an IN-FLIGHT record for it.

        Ids are `"<kind>-<n>"` with `n` monotonic across ALL kinds, so they sort by
        start order globally and still say what they are: scan-1, write-2, scan-3.
        """
        run_id = f"{kind}-{next(self._seq)}"
        self._runs[run_id] = RunRecord(run_id=run_id, kind=str(kind),
                                       started=time.time(), backend=str(backend))
        self._trim()
        return run_id

    def _trim(self) -> None:
        # Oldest first: plain dicts preserve insertion order.
        while len(self._runs) > self._keep:
            try:
                self._runs.pop(next(iter(self._runs)))
            except (StopIteration, KeyError):      # a concurrent pop got there first
                return

    # ---------------------------------------------------------------- mutation --

    def annotate(self, run_id: str | None, **fields) -> RunRecord | None:
        """Update a record WITHOUT closing it. Used at start time to stash what the
        run was asked to do (e.g. how many objects a write submitted), which is what
        `getStatus` needs as a progress denominator while the run is still going."""
        if not run_id:
            return None
        rec = self._runs.get(run_id)
        if rec is None:
            return None
        for key, value in fields.items():
            if hasattr(rec, key):
                setattr(rec, key, value)
        return rec

    def finish(self, run_id: str | None, **fields) -> RunRecord | None:
        """Close a record. Unknown/None ids are a NO-OP, never an error: a pump must
        never fail a run because its bookkeeping id went missing."""
        rec = self.annotate(run_id, **fields)
        if rec is None:
            return None
        if rec.finished is None:
            rec.finished = time.time()
        return rec

    # ------------------------------------------------------------------- reads --

    def get(self, run_id: str) -> RunRecord | None:
        return self._runs.get(str(run_id))

    def last(self, kind: str, *, completed_only: bool = True) -> RunRecord | None:
        """The most recently ALLOCATED record of a kind. `completed_only` keeps a
        still-running (or failed-to-persist) run from being handed back as a result --
        `getScanResult` with no run_id means "the last COMPLETED scan"."""
        for rec in reversed(list(self._runs.values())):
            if rec.kind != kind:
                continue
            if completed_only and not rec.completed:
                continue
            return rec
        return None

    def close_if_open(self, run_id: str | None, **fields) -> RunRecord | None:
        """Close a record ONLY if nothing has closed it yet.

        The `finally` form of `finish`. A pump's normal path closes the record with
        the real outcome; its `finally` then calls this, which is a no-op on the
        happy path and the safety net on every other one -- an exception, a
        `CancelledError`, a worker that died before it could report."""
        rec = self._runs.get(str(run_id)) if run_id else None
        if rec is None or rec.finished is not None:
            return rec
        return self.finish(run_id, **fields)

    def in_flight(self) -> RunRecord | None:
        """The newest record that has not finished -- what `getStatus` reports as the
        current `run_id` / `run_kind`.

        ONLY the newest allocation counts. Belt and braces against a leaked record:
        if some future pump still manages to die without closing its record, the leak
        is bounded to that one run instead of being reported as "in flight" forever,
        because the next allocation supersedes it. Two runs are never legitimately in
        flight at once -- one tip, one lr-class pass."""
        newest = self.newest()
        if newest is not None and newest.finished is None:
            return newest
        return None

    def newest(self) -> RunRecord | None:
        for rec in reversed(list(self._runs.values())):
            return rec
        return None

    def all(self) -> list[RunRecord]:
        return list(self._runs.values())

    def clear(self) -> None:
        """Tests only -- a fresh registry between cases."""
        self._runs.clear()


# ------------------------------------------------------------------- file paths --

def wire_path(path) -> str:
    """One absolute, forward-slash, UTF-8 path string (plan Sec A.1), so a script can
    use it unmodified from Python on Windows."""
    return os.path.abspath(os.fspath(path)).replace("\\", "/")


def existing_files(candidates: dict) -> tuple[dict[str, str], bool]:
    """Stat every ATTEMPTED export and return `(files, partial)`.

    Pass only what the pump actually tried to write. A value that is None/empty, or a
    path that does not exist on disk, is an export that FAILED: it is left out of
    `files` and sets `partial`. Never list a path you have not stat'd -- that is the
    whole point of the record (plan Sec A.1).
    """
    files: dict[str, str] = {}
    partial = False
    for name, path in candidates.items():
        if not path:
            partial = True
            continue
        try:
            if os.path.isfile(os.fspath(path)):
                files[name] = wire_path(path)
            else:
                partial = True
        except (OSError, TypeError, ValueError):
            partial = True
    return files, partial
