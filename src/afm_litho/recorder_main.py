"""The recorder process:  python -m afm_litho.recorder_main --port N

Started by the keeper (AFM_LITHO_RECORDER_PORT carries the port). It owns the journal file's
writes; the service builds each line and the arrival numbers, the recorder only writes them.
It never touches Igor and is on no Stop route.

Wire, JSON lines over 127.0.0.1 only. A client sends numbered messages `k`:
    {"k": 1, "op": "open", "path": "<run file>.jsonl"}        append to this file from now on
    {"k": 2, "op": "line", "line": "<journal line>"}          write it, flushed
    {"k": 3, "op": "line", "line": "...", "late": true}       write it with "late": true added
    {"k": 4, "op": "sync"}                                    fsync what was written
The recorder answers {"k": N} once message N is handled (a sync only after the fsync). `late`
marks an entry written after its time, replayed from a spill file or sent by the keeper.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import sys
import threading
from pathlib import Path

from afm_litho import clock
from afm_litho.keeper import beatfile
from afm_litho.keeper.protocol import (
    HEARTBEAT_EVERY_S,
    PORT_TAKEN_EXIT,
    heartbeat_line,
    ready_line,
)
from afm_litho.service.deps import JOURNAL_DIR
from afm_litho.service.journal import FileSink

HOST = "127.0.0.1"
CLOSING_GRACE_S = 10.0  # after Ctrl+C, wait this long for the service to finish its last lines


def late_line(line: str) -> str:
    """The same journal line with `late: true` added."""
    entry = json.loads(line)
    entry["late"] = True
    return json.dumps(entry, default=str, allow_nan=False)


class Recorder:
    """Accepts clients on loopback and writes their lines. One thread per client. With a
    `folder`, a client may only open journal files inside it."""

    def __init__(self, port: int = 0, folder: Path | None = None) -> None:
        self._folder = None if folder is None else Path(folder).resolve()
        self._listener = socket.socket()
        self._listener.bind((HOST, port))
        self._listener.listen(8)
        self.port = self._listener.getsockname()[1]
        self._clients = 0
        self._lock = threading.Lock()
        self._conns: set[socket.socket] = set()
        self._stopped = False
        threading.Thread(target=self._accept, name="recorder-accept", daemon=True).start()

    @property
    def clients(self) -> int:
        with self._lock:
            return self._clients

    def stop(self) -> None:
        """Close the listener and every client connection."""
        self._stopped = True
        with contextlib.suppress(OSError):
            self._listener.close()
        with self._lock:
            conns = list(self._conns)
        for conn in conns:
            with contextlib.suppress(OSError):
                conn.shutdown(socket.SHUT_RDWR)  # close() alone waits for the reader's makefile

    def _accept(self) -> None:
        while not self._stopped:
            try:
                conn, _ = self._listener.accept()
            except OSError:
                return
            with self._lock:
                self._clients += 1
                self._conns.add(conn)
            threading.Thread(target=self._serve, args=(conn,), name="recorder-client",
                             daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        sink: FileSink | None = None
        try:
            with conn, conn.makefile("r", encoding="utf-8", newline="\n") as lines:
                for raw in lines:
                    msg = json.loads(raw)
                    op = msg["op"]
                    if op == "open":
                        path = Path(msg["path"])
                        if path.suffix != ".jsonl" or not (
                                self._folder is None or path.resolve().is_relative_to(self._folder)):
                            raise ValueError(f"not a journal file: {path}")
                        if sink is not None:
                            sink.close()
                        sink = FileSink(path)
                    elif op == "line" and sink is not None:
                        sink(late_line(msg["line"]) if msg.get("late") else msg["line"])
                    elif op == "sync" and sink is not None:
                        sink.sync()
                    else:
                        raise ValueError(f"unexpected message {op!r}")
                    conn.sendall((json.dumps({"k": msg["k"]}) + "\n").encode())
        except (OSError, ValueError, KeyError):
            # A bad message or a disk error ends this client; its sender spills and replays.
            pass
        finally:
            if sink is not None:
                with contextlib.suppress(OSError):
                    sink.close()
            with self._lock:
                self._clients -= 1
                self._conns.discard(conn)


def _keeper_beat(beats, line: str) -> None:
    """Put a protocol line in the recorder's beat file; a keeper that is gone changes nothing."""
    if beats is not None:
        beats.write(line + "\n")
        beats.flush()


def _heartbeat(beats) -> None:
    while True:
        _keeper_beat(beats, heartbeat_line(os.getpid()))
        clock.sleep(HEARTBEAT_EVERY_S)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="afm-litho-recorder", description="Run the journal recorder")
    ap.add_argument("--port", type=int, default=int(os.environ.get("AFM_LITHO_RECORDER_PORT") or 0))
    args = ap.parse_args(argv)
    try:
        recorder = Recorder(args.port, JOURNAL_DIR)
    except OSError:
        print(f"Recorder port {args.port} is already in use.", file=sys.stderr)
        return PORT_TAKEN_EXIT
    beats = beatfile.from_env()
    _keeper_beat(beats, ready_line(None, 0, os.getpid()))
    threading.Thread(target=_heartbeat, args=(beats,), name="recorder-heartbeat",
                     daemon=True).start()
    try:
        while True:
            clock.sleep(0.25)
    except KeyboardInterrupt:
        # The service got the same Ctrl+C and is writing its last lines; wait for it to leave.
        grace = clock.deadline(CLOSING_GRACE_S)
        while recorder.clients and not grace.expired():
            clock.sleep(0.1)
    recorder.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
