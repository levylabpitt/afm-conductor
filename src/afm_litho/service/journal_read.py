"""Read a journal file and answer questions from it, and from it alone.

Plain functions over the list of entries `load()` returns, so a person can use them from a
notebook: `entries = load(path)`, then `in_flight(entries)`, `stop_vs_chunk(entries, 1)`,
`commander(entries, step)`. Nothing here touches the service or its memory. The file format
is in `journal.py`.

What the file can and cannot say (docs/migration/slices/J3-journal-questions.md):
- A wire entry carries `trace` when the send ran inside a door request's own context. Plain
  threads a request starts (the write worker, the fail-safe thread that retracts) carry none,
  and long-lived tasks are started without one (`bridge_protocol.spawn_untraced`), so
  `commander()` says how it decided (`basis`).
- Stop rows are journalled after their handler, so a stop's request entry comes after the
  wire lines it caused. The wire line itself (`AFM_SafePark()` over COM, or `ABORT`) is the
  moment the stop landed.
- Some entries a stop causes (a read-back ABORTED, `litho_object_partial`) can be noted just
  before the stop line, because the line is noted after the COM call returns.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Entry = dict[str, Any]

# Frames the instrument sends up. A twin's stub sends them through the same tap as the host,
# so they are told apart by verb.
UPWARD = frozenset({"VALUE", "XPT", "CAPTURE", "DONE", "ABORTED", "ERR", "SCANDONE", "TELEM",
                    "STARTUP"})
# Commands that start a pass and are answered by CAPTUREs and a final DONE or ABORTED.
PASS_STARTS = frozenset({"FIRE", "ARMFIRE", "LITHOCHAIN", "SCANCHAIN"})
# Commands that only read; the instrument is unchanged by them. The telemetry tick sends them
# at any moment, so they would blur "the last thing it was told to do".
READS = frozenset({"GETVALUE", "GETGV", "READXPT"})
PASS_ENDS = frozenset({"DONE", "ABORTED", "SCANDONE", "ERR"})
# The request rows that start a run, and the stream event that ends it.
RUN_ROWS = {"startWrite": "litho_done", "startScan": "scan_done"}
# Rows that stop or retract (control "stop" in requests/table.py; a test pins this list).
STOP_ROWS = frozenset({"abort", "safePark", "withdraw", "abortWrite", "abortScan"})
# Texts only ever sent to stop (journal wire `com` and `send` lines).
STOP_LINES = frozenset({"AFM_SafePark()", "ABORT"})
# The thread name the retract runs on (hardware/failsafe_thread.py).
FAILSAFE_THREAD = "afm-failsafe"
# Every litho pass arms the same fixed in-wave set (Output.A, ZSensor, Deflection, XSensor,
# YSensor: the "fixed channel set" of bridge_protocol.ARMFIRE), so a chunk is whole when all
# five CAPTUREs are in. It cannot be learned from a file that stops mid-chunk.
CAPTURE_CHANNELS = 5
LAST_COMMANDS = 5


def load(path: str | Path) -> list[Entry]:
    """Every entry in the file, in file order. A torn last line (the service died mid-write)
    is dropped, and `is_closed()` is then False. A bad line anywhere else raises."""
    lines = Path(path).read_text("utf-8").splitlines()
    entries: list[Entry] = []
    for i, line in enumerate(lines):
        try:
            entries.append(json.loads(line))
        except ValueError as exc:
            if i == len(lines) - 1:
                break
            raise ValueError(f"{path}: line {i + 1} is not JSON ({exc}); the file is damaged "
                             "before its end") from exc
    return _in_arrival_order(entries)


def _in_arrival_order(entries: list[Entry]) -> list[Entry]:
    """Late lines from the recorder's spill come after newer ones in the file. Put the numbered
    entries back in arrival order (n) in the slots they occupy, keeping the header, gap and
    closing lines where they are, and drop a repeated n (a replayed line the recorder had
    already written), keeping the one that is not marked late."""
    best: dict[int, Entry] = {}
    for e in entries:
        n = e.get("n")
        if n is not None and (n not in best or best[n].get("late")):
            best[n] = e
    ordered = iter(sorted(best.values(), key=lambda e: e["n"]))
    seen: set[int] = set()
    out = []
    for e in entries:
        n = e.get("n")
        if n is None:
            out.append(e)
        elif n not in seen:
            seen.add(n)
            out.append(next(ordered))
    return out


def is_closed(entries: list[Entry]) -> bool:
    """True when the service stopped cleanly (the writer wrote its closing line)."""
    return bool(entries) and entries[-1].get("what") == "closed"


def _body(e: Entry) -> dict:
    return e.get("body") or {}


def text_of(e: Entry) -> str:
    return str(_body(e).get("text", ""))


def verb_of(e: Entry) -> str:
    return text_of(e).split(" ", 1)[0]


def uid_of(e: Entry) -> str | None:
    parts = text_of(e).split()
    return parts[1] if len(parts) > 1 else None


def _thread(e: Entry) -> str:
    return str(_body(e).get("thread", ""))


def commands(entries: list[Entry]) -> list[Entry]:
    """The host's wire commands in file order: every TCP line it sent and every Igor COM
    command. These are the "steps" of question 3."""
    return [e for e in entries if e.get("src") == "wire" and (
        e["what"] == "com" or (e["what"] == "send" and verb_of(e) not in UPWARD))]


def instrument_frames(entries: list[Entry]) -> list[Entry]:
    """What the host read back from the instrument (`read` entries)."""
    return [e for e in entries if e.get("src") == "wire" and e["what"] == "read"]


# --- requests and runs ------------------------------------------------------


@dataclass
class Request:
    n: int                    # arrival number of the request entry
    trace: int
    row: str
    actor: str
    reply_n: int | None       # the reply or error entry, if there is one
    result: Any = None
    failed: bool = False      # the handler raised: a stop that failed did not run


def requests(entries: list[Entry]) -> list[Request]:
    """Every door request, paired with its reply by trace."""
    out: dict[int, Request] = {}
    for e in entries:
        if e.get("src") != "door" or "trace" not in e:
            continue
        if e["what"] == "request":
            out[e["trace"]] = Request(e["n"], e["trace"], _body(e).get("row", ""),
                                      e.get("actor", ""), None)
        elif e["what"] in ("reply", "error") and e["trace"] in out:
            out[e["trace"]].reply_n = e["n"]
            out[e["trace"]].failed = e["what"] == "error"
            out[e["trace"]].result = _body(e).get("result", _body(e).get("error"))
    return sorted(out.values(), key=lambda r: r.n)


@dataclass
class Run:
    run_id: str
    request: Request          # the startWrite or startScan request that made it
    started_n: int            # its reply
    ended_n: int | None       # the run's end event; None if the file shows none
    limit_n: float            # the next run's request: nothing later belongs to this run

    def window(self) -> tuple[int, float]:
        """The arrivals that belong to this run: from its request to its end event, or, if the
        file shows none, to the next run's request (or the end of the file). The one definition
        every question uses. A park can outlive it (see `_retract_commander`)."""
        return self.request.n, (self.ended_n if self.ended_n is not None else self.limit_n)


def runs(entries: list[Entry]) -> list[Run]:
    """Runs the file shows starting: a startWrite or startScan reply that names a run id,
    ended by the next `litho_done` or `scan_done` event before the following run starts."""
    starts = [r for r in requests(entries) if r.row in RUN_ROWS and r.reply_n is not None
              and isinstance(r.result, dict) and r.result.get("run_id")]
    found: list[Run] = []
    for i, r in enumerate(starts):
        limit = starts[i + 1].n if i + 1 < len(starts) else math.inf
        end = next((e["n"] for e in entries if r.reply_n < e.get("n", 0) < limit
                    and e.get("src") == "stream" and e["what"] == RUN_ROWS[r.row]), None)
        found.append(Run(r.result["run_id"], r, r.reply_n, end, limit))
    return found


def step_outcomes(entries: list[Entry], execution_id: str, step: int | None = None
                  ) -> list[Entry]:
    """What happened at step `step` of the run stored under `execution_id`, in file order: the
    "step" entries that cite it (`what` is the outcome kind, the body holds what it found).
    Without `step`: every outcome of that execution, which reads as the run's story. Wire
    entries sent while a step was in progress carry the same `run` and `body.step`.
    Outcomes of a write whose run could not be stored have no `run` and are not found here."""
    return [e for e in entries
            if e.get("src") == "step" and e.get("run") == execution_id
            and (step is None or _body(e).get("step") == step)]


def _contains(run: Run, n: int) -> bool:
    """True when arrival `n` falls inside the run's window, after its reply."""
    return run.started_n < n <= run.window()[1]


def _run_at(entries: list[Entry], n: int) -> Run | None:
    return next((r for r in runs(entries) if _contains(r, n)), None)


def _pick_run(entries: list[Entry], run_id: str | None) -> Run | None:
    return next((r for r in runs(entries) if r.run_id == run_id), None)


# --- stops -------------------------------------------------------------------


def stop_marker(entries: list[Entry], after: int = 0, before: float = math.inf) -> Entry | None:
    """The wire line where a stop landed: `AFM_SafePark()` over COM, or the TCP `ABORT`.
    The first one with arrival number between `after` and `before`; None if there is none."""
    return next((e for e in commands(entries) if after < e["n"] < before
                 and ((e["what"] == "com" and text_of(e) == "AFM_SafePark()")
                      or (e["what"] == "send" and text_of(e) == "ABORT"))), None)


def _stop_in(entries: list[Entry], run: Run) -> Entry | None:
    lo, hi = run.window()
    return stop_marker(entries, lo, hi + 1)     # the stop line comes before the end event


def _stop_for(entries: list[Entry], run_id: str | None) -> tuple[Run | None, Entry | None]:
    """The run asked about and its stop line. With a `run_id`: that run's first stop after it
    started. Without: the first run that has a stop; failing that, the file's first stop line
    (a retract with no run)."""
    if run_id is not None:
        run = _pick_run(entries, run_id)
        if run is None:
            raise KeyError(f"no run {run_id!r} in the file")
        return run, _stop_in(entries, run)
    for run in runs(entries):
        marker = _stop_in(entries, run)
        if marker is not None:
            return run, marker
    return None, stop_marker(entries)


# --- question 1: what was in flight ----------------------------------------


@dataclass
class InFlight:
    at_n: int                         # the arrival number the question is asked at
    stopped: bool                     # True when `at_n` is the stop line, False for "the end"
    run_id: str | None = None
    request: Request | None = None    # what asked for the run
    object: str | None = None         # the object being written, from the event stream
    pending: list[str] = field(default_factory=list)   # pass commands with no answer yet
    last_commands: list[str] = field(default_factory=list)  # newest last


def in_flight(entries: list[Entry], at: int | None = None,
              run_id: str | None = None) -> InFlight:
    """What was going on at arrival `at`. Without `at`: the stop line of the run (see
    `_stop_for`), else the end of the file (a service that died).

    The run (and the request that made it), the object being written, the pass commands the
    instrument had not yet answered, and the last few commands the host sent that act on it
    (reads are left out: the telemetry tick sends them at any moment).
    """
    run, marker = _stop_for(entries, run_id) if at is None else (None, None)
    stopped = at is None and marker is not None
    at_n = at if at is not None else (marker["n"] if marker else _last_n(entries))
    # At a stop, the stop line itself is the answer to "what stopped it", not part of what was
    # going on.
    before = [e for e in entries
              if e.get("n", 0) < at_n or (not stopped and e.get("n", 0) == at_n)]
    run = run or _run_at(entries, at_n)
    sent = [e for e in commands(before) if verb_of(e) not in READS]
    return InFlight(
        at_n=at_n, stopped=stopped,
        run_id=run.run_id if run else None, request=run.request if run else None,
        object=_object_at(before, run, stopped), pending=_pending(before, stopped, run),
        last_commands=[text_of(e) for e in sent[-LAST_COMMANDS:]])


def _last_n(entries: list[Entry]) -> int:
    return max((e["n"] for e in entries if "n" in e), default=0)


def _object_at(before: list[Entry], run: Run | None, stopped: bool) -> str | None:
    """The object whose `litho_object_start` came last, unless it already finished. At a stop
    line a `litho_object_partial` does not count as finishing: it is the stop's consequence and
    can be noted just before the line."""
    if run is None:
        return None
    ends = ("litho_object_done",) if stopped else ("litho_object_done", "litho_object_partial")
    current = None
    for e in before:
        if e.get("src") != "stream" or not run.window()[0] < e["n"] <= run.window()[1]:
            continue
        msg = _body(e).get("msg") or {}
        if e["what"] == "litho_object_start":
            current = msg.get("name")
        elif e["what"] in ends:
            current = None if current == msg.get("name") else current
    return current


def _pending(before: list[Entry], stopped: bool, run: Run | None) -> list[str]:
    """Pass commands of this run sent and not yet answered by DONE, SCANDONE, ERR or ABORTED.
    At a stop line an ABORTED does not count: the stop's own ABORT can be read back just
    before the line is noted, and it answers the stop, not the pass."""
    ends = PASS_ENDS - {"ABORTED"} if stopped else PASS_ENDS
    answered = {uid_of(e) for e in instrument_frames(before) if verb_of(e) in ends}
    lo = run.request.n if run else 0
    return [text_of(e) for e in commands(before)
            if e["what"] == "send" and verb_of(e) in PASS_STARTS and e["n"] > lo
            and uid_of(e) not in answered]


# --- question 2: did the stop land before or after chunk n ------------------


def chunks_delivered(entries: list[Entry], name: str, before_n: int,
                     uid: str | None = None, after_n: int = 0) -> int:
    """Whole chunks of object `name` the host read between arrivals `after_n` and `before_n`
    (only the pass `uid`, if given): the fewest CAPTUREs seen on any one channel. A chunk with
    some channels still to come is not counted. The wire carries no chunk number; chunks
    follow one another."""
    per_channel: dict[str, int] = {}
    for e in instrument_frames(entries):
        parts = text_of(e).split()
        if (after_n < e["n"] < before_n and parts[0] == "CAPTURE" and len(parts) == 4
                and parts[2] == name and (uid is None or parts[1] == uid)):
            per_channel[parts[3]] = per_channel.get(parts[3], 0) + 1
    if len(per_channel) < CAPTURE_CHANNELS:
        return 0
    return min(per_channel.values())


@dataclass
class StopLanding:
    run_id: str | None
    object: str | None
    stop_n: int
    chunks_before: int        # whole chunks of this pass read before the stop line

    def after_chunk(self, n: int) -> bool:
        """True when chunk `n` (counting from 0) was delivered before the stop landed."""
        return self.chunks_before > n


def stop_landing(entries: list[Entry], run_id: str | None = None) -> StopLanding | None:
    """Where a stop landed in the object being written; None with no stop. Without `run_id`
    it is the first run that has a stop (see `_stop_for`); give a `run_id` to ask about a
    later run or a later stop in a file with several."""
    run, marker = _stop_for(entries, run_id)
    if marker is None:
        return None
    state = in_flight(entries, run_id=run.run_id if run else None)
    name = state.object
    chunks = 0
    if name:
        launch = [e for e in commands(entries) if e["what"] == "send" and e["n"] < marker["n"]
                  and verb_of(e) in PASS_STARTS and text_of(e).split()[2:3] == [name]
                  and (run is None or e["n"] > run.request.n)]
        uid = uid_of(launch[-1]) if launch else None
        chunks = chunks_delivered(entries, name, marker["n"], uid,
                                  launch[-1]["n"] if launch else 0)
    return StopLanding(state.run_id, name, marker["n"], chunks)


def stop_vs_chunk(entries: list[Entry], n: int, run_id: str | None = None) -> str:
    """"after" or "before": did the stop land after chunk `n` was delivered. "no stop" if the
    file shows none for that run."""
    landing = stop_landing(entries, run_id)
    if landing is None:
        return "no stop"
    return "after" if landing.after_chunk(n) else "before"


# --- question 3: who commanded step k ---------------------------------------


@dataclass
class Commander:
    actor: str
    row: str
    trace: int | None
    basis: str    # "trace", "run", "stop" or "fault"; see commander()


def commander(entries: list[Entry], step: Entry) -> Commander | None:
    """Who commanded one wire command (an entry from `commands()`), and how we know. Only
    "trace" is proof; "run", "stop" and "fault" are rules over file order and thread names.

    - "trace": the send ran inside that request's own context. Exact.
    - "stop": no trace, sent by the fail-safe thread (the retract), or a stop line
      (`AFM_SafePark()`, `ABORT`) from any thread. The first `safe_park` event after it is
      its park; the stop request that caused that park (`_stop_of_park`) asked for it.
    - "fault": the same, but no stop request in this run's window before the park, and none
      that is only its own (see `_stop_of_park`): the service parked itself (dead-man,
      interlock, bridge drop). The park's reason is given as the row.
    - "run": no trace, and the sending thread first appeared after a run started (the write
      worker, the scan driver) while that run was going: whoever's startWrite/startScan made
      the run. A thread that was already sending before the run began (the main thread's
      telemetry reads, a reused pool thread) is not taken for the run's: None.
    - None: the file cannot say.
    """
    by_trace = {r.trace: r for r in requests(entries)}
    if step.get("trace") in by_trace:
        r = by_trace[step["trace"]]
        return Commander(r.actor, r.row, r.trace, "trace")
    if _thread(step) == FAILSAFE_THREAD or text_of(step) in STOP_LINES:
        return _retract_commander(entries, step)
    run = _run_at(entries, step["n"])
    if run is not None and _thread_began_after(entries, _thread(step), run.started_n):
        return Commander(run.request.actor, run.request.row, run.request.trace, "run")
    return None


def _thread_began_after(entries: list[Entry], thread: str, n: int) -> bool:
    """True when the thread's first wire entry in the file came after arrival `n`."""
    first = next((e["n"] for e in entries if e.get("src") == "wire" and _thread(e) == thread),
                 None)
    return first is not None and first > n


def _retract_commander(entries: list[Entry], step: Entry) -> Commander | None:
    parks = [e for e in entries if e.get("src") == "stream" and e["what"] == "safe_park"]
    park = next((e for e in parks if e["n"] > step["n"]), None)
    if park is None:
        return None
    stop = _stop_of_park(entries, park, parks)
    if stop is not None:
        return Commander(stop.actor, stop.row, stop.trace, "stop")
    reason = (_body(park).get("msg") or {}).get("reason", "")
    return Commander("service", f"safe_park: {reason}", None, "fault")


def _stop_of_park(entries: list[Entry], park: Entry, parks: list[Entry]) -> Request | None:
    """The stop request that caused `park`, or None (the service parked itself).

    The park belongs to the run that began last before it (a scan's deferred park comes after
    its `scan_done`, so the run's window is too short for it). Only stops that actually ran
    count (a request that errored retracted nothing). Then either
    - a stop in that run that came before the park (a scan Stop answers at once and the pump
      parks later; one Stop can cause several parks): the nearest such; or
    - a stop that came after it (a write abort waits for its park, then is journalled) when no
      other park lies between, and none follows that stop either: any other park would be that
      later stop's own, so this one was not.
    """
    run = max((r for r in runs(entries) if r.request.n < park["n"]),
              key=lambda r: r.request.n, default=None)
    lo, hi = (run.request.n, run.limit_n) if run else (0, math.inf)
    stops = [r for r in requests(entries) if r.row in STOP_ROWS and not r.failed
             and lo < r.n < hi]
    earlier = [r for r in stops if r.n < park["n"]]
    if earlier:
        return max(earlier, key=lambda r: r.n)
    later = next((r for r in stops if r.n > park["n"]), None)
    if later is None:
        return None
    boundary = next((r.n for r in stops if r.n > later.n), hi)
    others = [e for e in parks if e["n"] > park["n"] and e["n"] < boundary and e is not park]
    return later if not others else None
