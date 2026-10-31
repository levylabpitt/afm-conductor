"""Persist the RAW measured capture of a scan/pass alongside its session record.

The whitepaper's recorder logs "every command, waveform, capture, and scan", and
`.ibw` export is a projection of the session store -- so the store must hold the
real measured samples, not just the config. The session archive (sessions.jsonl)
stays small (metadata only); the bulk per-channel arrays live in a per-session
compressed `.npz` named by the session id, so recall/export loads them by id.
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
from pathlib import Path

import numpy as np

from .. import clock

log = logging.getLogger(__name__)

_DIR = "captures"
QUEUE_SIZE = 8          # captures waiting for the writer; one scan or write each
WAIT_S = 30.0           # how long a caller waits for its own save before saving it itself


def _capture_dir(runtime_dir) -> Path:
    d = Path(runtime_dir) / _DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


class _RawWriter:
    """The one thread that writes raw captures, fed through a bounded queue.

    A caller hands its save over and waits for the result, so `save_capture` still returns
    only once the file is on disk. When the queue is full, or the writer does not answer
    within WAIT_S, the caller saves its own capture instead: a raw capture is never
    dropped and no caller waits on a backlog. Stop, park and withdraw never come here.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_SIZE)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def _ensure_started(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="raw-writer", daemon=True)
                self._thread.start()

    def _run(self) -> None:
        while True:
            args, kwargs, box, done = self._queue.get()
            if box.get("cancelled"):          # the caller gave up and saved it itself
                continue
            try:
                box["ref"] = save_capture_now(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - handed to the caller; the thread lives on
                box["error"] = exc
            done.set()

    def _save_here(self, args: tuple, kwargs: dict, reason: str) -> str:
        from . import journal
        journal.running().note("capture", "gap", reason=reason, session_id=args[1])
        log.warning("%s; saving %s on the caller", reason, args[1])
        return save_capture_now(*args, **kwargs)

    def save(self, args: tuple, kwargs: dict) -> str:
        if sys.is_finalizing():
            return save_capture_now(*args, **kwargs)
        self._ensure_started()
        if self._thread is None or not self._thread.is_alive():
            return self._save_here(args, kwargs, "raw writer thread is not running")
        box: dict = {}
        done = threading.Event()
        try:
            self._queue.put_nowait((args, kwargs, box, done))
        except queue.Full:
            return self._save_here(args, kwargs, "raw writer queue full")
        if not clock.wait(done, WAIT_S):
            box["cancelled"] = True       # the writer skips it if it has not started yet
            return self._save_here(
                args, kwargs, f"raw writer did not answer in {WAIT_S:.0f} s")
        if "error" in box:
            raise box["error"]
        return box["ref"]


_writer = _RawWriter()


def save_capture(runtime_dir, session_id: str, cap: dict, *, out_dir=None) -> str:
    """Save `cap` (see `save_capture_now`) through the raw writer thread."""
    if not cap:
        return ""
    return _writer.save((runtime_dir, session_id, cap), {"out_dir": out_dir})


def save_capture_now(runtime_dir, session_id: str, cap: dict, *, out_dir=None) -> str:
    """Write {channel: samples} to `<out_dir>/<session_id>.npz` (default:
    captures/<session_id>.npz, the old flat layout -- kept as the default so a
    caller that doesn't know about the new per-kind layout still works).
    Callers on the new layout pass a stem for `session_id` and
    `out_dir=stems.dir_for(runtime_dir, kind, "npz")`. Returns the reference,
    relative to `runtime_dir` when `out_dir` lives under it, else the bare
    path string. Channel names with dots (Output.Z) are archived verbatim as
    npz keys. A None/empty capture writes nothing."""
    if not cap:
        return ""
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in cap.items()}
    dest_dir = Path(out_dir) if out_dir is not None else _capture_dir(runtime_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{session_id}.npz"
    # WRITE TO A TEMP NAME, THEN RENAME. `np.savez_compressed` streams a zip into the
    # final path, so for the length of the write there is a file AT THAT PATH that is a
    # truncated archive -- and everything downstream (`load_capture`,
    # `getScanResult`/`getWriteResult` inline, a cockpit capture download) finds it by
    # path and opens it. The reviewer reproduced the resulting `BadZipFile`. `os.replace`
    # is atomic on Windows and POSIX alike, so a reader sees either the previous file or
    # the whole new one, never half of one.
    # THE TEMP NAME MUST NOT END IN `.npz`, and it must not look like a capture.
    # Everything that finds captures does so by globbing `*.npz` in this directory
    # (the cockpit's recall, the exporters, a couple of tests), so a half-written
    # `<id>.npz.<pid>.part.npz` sitting beside the real one is the same hazard in a new
    # costume -- a glob picks it up and opens a partial archive. A leading dot plus a
    # `.tmp` suffix keeps it out of every such glob.
    #
    # The archive is written through an OPEN FILE HANDLE rather than by name, because
    # `np.savez_compressed` appends `.npz` to any path that does not already end in it
    # -- which is exactly how the first version of this fix reintroduced the bug.
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)       # a no-op after a successful replace
        except OSError:
            pass
    try:
        return str(path.relative_to(Path(runtime_dir))).replace("\\", "/")
    except ValueError:
        return str(path)


def load_capture(runtime_dir, session_id: str) -> dict:
    """Load the raw capture for a session/run id, or {} if none was recorded.

    Resolves through `stems.lookup` first (the new layout: the id names a
    `stem_index.jsonl` entry, and the npz lives at
    `dir_for(runtime_dir, kind, "npz")/<stem>.npz`). Falls back to the old flat
    `captures/<id>.npz` path when the id was never registered -- an
    un-migrated id, or a caller (a test) that wrote straight via
    `save_capture`'s default `out_dir`."""
    from . import stems
    rec = stems.lookup(runtime_dir, session_id)
    if rec is not None:
        path = Path(runtime_dir) / stems.kind_dir(rec["kind"]) / "npz" / f"{rec['stem']}.npz"
    else:
        path = Path(runtime_dir) / _DIR / f"{session_id}.npz"
    if not path.is_file():
        return {}
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
