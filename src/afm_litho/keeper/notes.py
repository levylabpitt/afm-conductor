"""The keeper's notes: one JSON line per Stop it passed on, fsynced, in
runtime/keeper/notes.jsonl. The format is docs/migration/plan/keeper-notes-spec.md; the
keeper writes it from that spec, never from the service's journal code (stdlib only).

A note is written after its Stop has been passed on and answered, never before. Each note
also goes to the recorder as a late entry (`late: true`) in journal/keeper/notes.jsonl,
whenever the recorder is up; the file here is the record, the recorder's copy a convenience.
"""

from __future__ import annotations

import json
import os
import queue
import socket
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

from afm_litho import clock
from afm_litho.keeper import status

FORMAT_VERSION = 1
HOST = "127.0.0.1"
RECORDER_CONNECT_S = 0.2   # loopback: a live recorder accepts at once
RECORDER_ACK_S = 1.0       # the recorder answers each message once it is written
RECORDER_RETRY_S = 2.0     # between attempts while the recorder is down
RECORDER_BACKLOG = 1000    # notes held for a recorder that stays down; the file has them all


def notes_path() -> Path:
    return status.runtime_dir() / "keeper" / "notes.jsonl"


def recorder_journal_path() -> Path:
    """Where the recorder writes the keeper's late entries: inside the journal folder the
    recorder accepts (service/deps.py JOURNAL_DIR, the same runtime folder)."""
    return status.runtime_dir() / "journal" / "keeper" / "notes.jsonl"


def _iso(wall: float | None) -> str | None:
    if wall is None:
        return None
    return datetime.fromtimestamp(wall).astimezone().isoformat(timespec="milliseconds")


def stop_note(*, who: str, why: str, received_wall: float, mono: float, host_port: int,
              relay) -> dict:
    """The note for one Stop, as keeper-notes-spec.md lays it out."""
    return {
        "keeper_notes": FORMAT_VERSION,
        "src": "keeper",
        "what": "stop",
        "t": _iso(received_wall),
        "mono": round(mono, 4),
        "body": {
            "who": who,
            "why": why,
            "sent_to": f"{HOST}:{host_port}",
            "connected": relay.connected,
            "forwarded_t": _iso(relay.forwarded_wall),
            "replies": list(relay.replies),
            "outcome": relay.outcome,
            "confirmed": relay.outcome == "confirmed",
        },
    }


def contact_note(words: str) -> dict:
    """The note for a revived keeper that could not reach the service."""
    return {"keeper_notes": FORMAT_VERSION, "src": "keeper", "what": "contact-failed",
            "t": _iso(clock.wall()), "mono": round(clock.now(), 4), "body": {"words": words}}


class Notes:
    """Appends notes to the file (flush + fsync each) and hands each to the recorder feed."""

    def __init__(self, path: Path | None = None, feed: RecorderFeed | None = None) -> None:
        self.path = Path(path) if path is not None else notes_path()
        self._feed = feed
        self._lock = threading.Lock()

    def write_stop(self, **fields) -> dict:
        note = stop_note(**fields)
        self.write(note)
        return note

    def write(self, note: dict) -> None:
        """One line, on disk before this returns. A disk error is printed, never raised: the
        Stop it describes has already been passed on."""
        line = json.dumps(note, default=str, allow_nan=False)
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    f.flush()
                    os.fsync(f.fileno())
        except OSError as exc:
            print(f"[keeper] could not write the note to {self.path} ({exc}): {line}",
                  flush=True)
        if self._feed is not None:
            self._feed.send(line)

    def close(self) -> None:
        if self._feed is not None:
            self._feed.close()


class RecorderFeed:
    """Sends notes to the recorder as late lines, on a thread of its own, retrying while the
    recorder is down. Wire: service/recorder_main.py (`open`, then `line` with `late`)."""

    _CLOSE = object()

    def __init__(self, port: int, journal_path: Path | None = None) -> None:
        self._port = int(port)
        self._path = Path(journal_path) if journal_path is not None else recorder_journal_path()
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="keeper-notes-feed", daemon=True)
        self._thread.start()
        self.delivered = 0

    def send(self, line: str) -> None:
        self._q.put(line)

    def close(self) -> None:
        self._q.put(self._CLOSE)

    def _run(self) -> None:
        pending: deque[str] = deque(maxlen=RECORDER_BACKLOG)
        while True:
            try:
                item = clock.queue_get(self._q, RECORDER_RETRY_S if pending else None)
            except queue.Empty:
                item = None
            while item is not None:
                if item is self._CLOSE:
                    return
                pending.append(item)
                try:
                    item = self._q.get_nowait()
                except queue.Empty:
                    item = None
            if pending:
                self._deliver(pending)

    def _deliver(self, pending: deque[str]) -> None:
        """Send what is pending; whatever the recorder acknowledged leaves the queue."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with socket.create_connection((HOST, self._port), timeout=RECORDER_CONNECT_S) as s:
                s.settimeout(RECORDER_ACK_S)
                with s.makefile("r", encoding="utf-8", newline="\n") as acks:
                    k = 1
                    self._ask(s, acks, {"k": k, "op": "open", "path": str(self._path)})
                    while pending:
                        k += 1
                        self._ask(s, acks, {"k": k, "op": "line", "line": pending[0],
                                            "late": True})
                        pending.popleft()
                        self.delivered += 1
        except (OSError, ValueError):
            pass                         # the recorder is down; try again later

    @staticmethod
    def _ask(sock: socket.socket, acks, msg: dict) -> None:
        sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))
        answer = acks.readline()
        if not answer or json.loads(answer).get("k") != msg["k"]:
            raise ValueError("the recorder did not acknowledge")
