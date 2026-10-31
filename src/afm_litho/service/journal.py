"""The journal: one JSON Lines file per service run, written off the caller's thread.

Callers only ever call `note()` on `running()`. It puts one finished line on a bounded queue and
returns; a single writer thread drains the queue to disk. `note()` never blocks and
never raises, so a stalled disk cannot slow a stop or any other instrument command.
When the queue is full the entry is dropped and a `gap` entry is written where it was lost.
Journal files are kept, never deleted.

File format v1 (docs/migration/slices/J1-journal.md has an example). Every line has
`src`, `what`, `t` and `mono`; only arrivals (entries from `note()`) also carry `n`:
  header   what "header", body {journal, started, platform, commit}
  arrival  n, src, what, t, mono, actor?, trace?, run?, chunk?, body?
  gap      src "journal", what "gap", body {dropped, first_n, last_n}
  gap      src "recorder", what "gap", body {entries, first_n, last_n, down_since}: the recorder
           process was unreachable; the entries after it carry `late: true` and keep their `n`
  closed   src "journal", what "closed", body {entries, write_failures}
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import math
import os
import platform
import queue
import re
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .. import clock

# Wire entries must match the golden tap, so arrays are summarised by the tap's own function:
# sha256 over little-endian float32, first 16 hex characters.
from ..goldens.expander import summarise as golden_summary
from .deps import JOURNAL_DIR, REPO

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
QUEUE_SIZE = 10_000
INTENT_WAIT_S = 1.0  # longest an intent waits for its entry to reach the disk
INLINE_LIST_MAX = 16  # longer lists and byte strings are summarised, not stored
STRING_MAX = 4096  # longer strings are cut, with their length noted
SECRET_KEY = re.compile(r"token|secret|password|authorization|cookie|api[-_]?key", re.I)
MASK = "***"


# --- turning a body into plain JSON ---------------------------------------


def summarise_array(items: Any) -> dict:
    """Count, first, last and the golden tap's hash; the whole array is never written."""
    wave = golden_summary("", np.asarray(items, dtype=float))
    return {"array": wave.n, "first": clean(wave.first), "last": clean(wave.last),
            "sha256": wave.sha}


def _is_number(x: Any) -> bool:
    return isinstance(x, int | float | np.number) and not isinstance(x, bool)


def _summarise_long_list(items: list | tuple) -> dict:
    """A number list gets the array summary; anything else gets count, first and last."""
    if _is_number(items[0]) and _is_number(items[-1]):
        with contextlib.suppress(TypeError, ValueError):
            return summarise_array(items)
    return {"list": len(items), "first": clean(items[0]), "last": clean(items[-1])}


def clean(value: Any) -> Any:
    """Plain, strict-JSON, bounded: secrets masked, arrays summarised, long text cut."""
    if isinstance(value, dict):
        return {
            str(k): MASK if SECRET_KEY.search(str(k)) else clean(v) for k, v in value.items()
        }
    if isinstance(value, np.ndarray):
        return summarise_array(value)
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        if len(value) <= STRING_MAX:
            return value
        return {"text": value[:STRING_MAX], "length": len(value)}
    if isinstance(value, bytes | bytearray):
        if len(value) <= INLINE_LIST_MAX:
            return {"bytes": bytes(value).hex()}
        return {"bytes": len(value), "head": bytes(value[:INLINE_LIST_MAX]).hex()}
    if isinstance(value, list | tuple):
        if len(value) > INLINE_LIST_MAX:
            return _summarise_long_list(value)
        return [clean(v) for v in value]
    return value


# --- the file and its header ----------------------------------------------


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO, capture_output=True, text=True, timeout=2, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - a missing git must not stop the service starting
        return "unknown"


def _open_run_file(folder: Path, started: datetime) -> Path:
    """`<folder>/<start time>.jsonl`; a second start in the same second gets `-2`, `-3`."""
    folder.mkdir(parents=True, exist_ok=True)
    stem = started.strftime("%Y-%m-%dT%H-%M-%S")
    for k in itertools.count(1):
        path = folder / (f"{stem}.jsonl" if k == 1 else f"{stem}-{k}.jsonl")
        try:
            path.open("x").close()
            return path
        except FileExistsError:
            continue
    raise AssertionError("unreachable")


class FileSink:
    """Appends lines to the run's file, flushing each one."""

    def __init__(self, path: Path) -> None:
        self._handle = path.open("a", encoding="utf-8", buffering=1)

    def __call__(self, line: str) -> None:
        self._handle.write(line + "\n")
        self._handle.flush()

    def sync(self) -> None:
        """Force what was written onto the disk, not just into the OS cache."""
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()


# --- the journal ----------------------------------------------------------


class Journal:
    """One service run's journal. Whoever creates it calls `start()` and `stop()`.

    It owns its queue and writer thread; callers only see `note()`. Before `start()` and
    after `stop()`, `note()` does nothing. `sink` is the one place a line reaches the disk;
    tests replace it to stall the writer. `sink_for(path, gap_line)` builds the sink once the run
    file exists; the keeper-started service uses it to send lines to the recorder process.
    """

    def __init__(self, folder: Path | None = None, *, queue_size: int = QUEUE_SIZE,
                 sink: Callable[[str], None] | None = None,
                 sink_for: Callable[[Path, Callable[[dict], str]], Any] | None = None) -> None:
        self._folder = Path(folder or JOURNAL_DIR)
        self._queue: queue.Queue[tuple[int, str, threading.Event | None]] = queue.Queue(
            maxsize=queue_size)
        self._given_sink = sink
        self._sink_for = sink_for
        self._numbers = itertools.count(1)
        self._lost_lock = threading.Lock()
        self._lost: list[int] = []  # arrival numbers dropped and not yet reported in a gap line
        self._accepting = False
        self._started = False
        self._stopping = threading.Event()
        self._writer: threading.Thread | None = None
        self._t0 = time.monotonic()
        self._last_n = 0
        self.write_failures = 0
        self.path: Path | None = None

    def start(self) -> Journal:
        """Create the folder, open the file, read the git commit, queue the header, start the
        writer and become the running journal. The owner's startup pays for the disk and git."""
        global _running
        if self._started:
            raise RuntimeError("journal already started; make a new Journal for a new run")
        self._started = True
        started = datetime.now().astimezone()
        self._t0 = time.monotonic()
        self.path = _open_run_file(self._folder, started)
        self._sink = self._given_sink or (
            self._sink_for(self.path, self._gap_line) if self._sink_for else FileSink(self.path))
        header = {"journal": FORMAT_VERSION, "started": started.isoformat(timespec="milliseconds"),
                  "platform": platform.platform(), "commit": git_commit()}
        self._queue.put_nowait((0, self._line(None, "journal", "header", header), None))
        self._writer = threading.Thread(target=self._drain, name="journal-writer", daemon=True)
        self._writer.start()
        self._accepting = True  # only once the writer is running, so no note outruns the header
        _running = self
        return self

    def stop(self, timeout: float = 2.0) -> None:
        """Stop taking entries, write what is queued and a closing line, join within `timeout`."""
        global _running
        self._accepting = False
        if _running is self:
            _running = None
        self._stopping.set()
        if self._writer is not None:
            self._writer.join(timeout)

    def _line(self, n: int | None, src: str, what: str, body: dict | None = None,
              **top: Any) -> str:
        entry: dict[str, Any] = {} if n is None else {"n": n}
        entry.update(src=src, what=what,
                     t=datetime.now().astimezone().isoformat(timespec="milliseconds"),
                     mono=round(time.monotonic() - self._t0, 4))
        entry.update({k: clean(v) for k, v in top.items() if v is not None})
        if body:
            entry["body"] = clean(body)
        # default=str: an object json cannot encode is written as its text, never an error.
        return json.dumps(entry, default=str, allow_nan=False)

    def _gap_line(self, body: dict) -> str:
        """A `gap` line from the recorder's sink: entries that reached the file late."""
        return self._line(None, "recorder", "gap", body)

    def note(self, src: str, what: str, /, *, actor: Any = None, trace: Any = None,
             run: Any = None, chunk: Any = None, **body: Any) -> None:
        """Queue one entry. Never blocks, never raises: a failure is counted as a gap."""
        self._enqueue(None, src, what, actor, trace, run, chunk, body)

    def note_intent(self, src: str, what: str, /, *, actor: Any = None, trace: Any = None,
                    run: Any = None, chunk: Any = None, **body: Any) -> int | None:
        """Queue an entry for a request that STARTS a run, and return only once it is on disk
        (fsync), or after INTENT_WAIT_S. On a timeout it logs and goes on: the run is never
        refused for a slow disk. A wedged or dead writer therefore makes every start cost up
        to INTENT_WAIT_S. Never use this on a stop, park or withdraw.
        Returns the arrival number, or None when the journal is not running."""
        done = threading.Event()
        n = self._enqueue(done, src, what, actor, trace, run, chunk, body)
        if n is not None and not clock.wait(done, INTENT_WAIT_S):
            log.warning("journal: entry %d (%s %s) not on disk after %.1f s; run goes on",
                        n, src, what, INTENT_WAIT_S)
        return n

    def _enqueue(self, done: threading.Event | None, src: str, what: str, actor: Any,
                 trace: Any, run: Any, chunk: Any, body: dict) -> int | None:
        if not self._accepting:
            return None
        n = next(self._numbers)  # taken before the try, so the failure path can report it
        try:
            line = self._line(n, src, what, body, actor=actor, trace=trace, run=run, chunk=chunk)
            self._queue.put_nowait((n, line, done))
        except Exception:  # noqa: BLE001 - note() is called on the stop path and must not raise
            with self._lost_lock:
                self._lost.append(n)
            return None  # dropped: nothing will ever set `done`, so the caller must not wait
        return n

    def _take_gap_line(self, before_n: float) -> str | None:
        """A gap line for the drops that came before arrival `before_n`, if there were any."""
        with self._lost_lock:
            lost = sorted(x for x in self._lost if x < before_n)
            self._lost = [x for x in self._lost if x >= before_n]
        if not lost:
            return None
        return self._line(None, "journal", "gap",
                          {"dropped": len(lost), "first_n": lost[0], "last_n": lost[-1]})

    def _drain(self) -> None:
        while not (self._stopping.is_set() and self._queue.empty()):
            try:
                n, line, done = self._queue.get(timeout=0.1)  # timeout: how a stop is noticed
            except queue.Empty:
                self._idle()
                continue
            gap = self._take_gap_line(n)
            if gap:
                self._write(gap)
            self._write(line)
            self._last_n = max(self._last_n, n)
            if done is not None:
                self._sync()
                done.set()
        gap = self._take_gap_line(math.inf)
        if gap:
            self._write(gap)
        self._write(self._line(None, "journal", "closed", {
            "entries": self._last_n, "write_failures": self.write_failures}))
        if hasattr(self._sink, "close"):
            with contextlib.suppress(Exception):
                self._sink.close()

    def _idle(self) -> None:
        idle = getattr(self._sink, "idle", None)
        if idle is not None:
            try:
                idle()
            except Exception:  # noqa: BLE001 - a sink fault must not kill the writer thread
                self.write_failures += 1

    def _sync(self) -> None:
        sync = getattr(self._sink, "sync", None)
        if sync is not None:
            try:
                sync()
            except Exception:  # noqa: BLE001 - a failed fsync is a slow disk, not a refusal
                self.write_failures += 1

    def _write(self, line: str) -> None:
        try:
            self._sink(line)
        except Exception:  # noqa: BLE001 - a bad disk must not kill the writer thread
            self.write_failures += 1
            if self.write_failures == 1:
                log.warning("journal write failed; lines are being lost (%s)", self.path)


# --- the running journal --------------------------------------------------

_running: Journal | None = None


class _NoJournal:
    """Stands in when no journal is running, so a caller never has to check."""

    def note(self, *args: Any, **fields: Any) -> None:
        return None

    def note_intent(self, *args: Any, **fields: Any) -> None:
        return None


_NO_JOURNAL = _NoJournal()


def running() -> Journal | _NoJournal:
    """The journal the service started, or one that ignores notes."""
    return _running or _NO_JOURNAL
