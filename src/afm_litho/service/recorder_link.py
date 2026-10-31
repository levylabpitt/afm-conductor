"""The service's side of the recorder: a journal sink that sends lines to the recorder process.

The Journal still builds every line and takes every arrival number; this sink only carries the
finished lines to `afm_litho.recorder_main` (wire described there). It is called from the
journal's one writer thread, so a slow or dead recorder never reaches a caller of `note()`.

Recorder down (not running yet, killed, or not answering within ACK_WAIT_S):
  - lines go to a spill file next to the run file (`<run file>.<pid>.spill`); an intent's sync
    fsyncs it, so the intent counts as written and the run is never refused;
  - once the recorder is back, the sink sends one `gap` line (src "recorder") per outage and
    then every spilled line marked `late`, in arrival order, and deletes the spill file once
    the recorder acknowledged them. A replay that fails keeps only the lines not yet acknowledged;
  - spill files left by an earlier service run (that run's process is gone) are replayed into
    their own run file the same way. One whose run file is missing is logged and kept.
A line the recorder wrote but could not acknowledge before dying is sent again, so it can
appear twice (the second marked `late`, same `n`).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import threading
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import clock
from ..keeper.pids import pid_alive

log = logging.getLogger(__name__)

HOST = "127.0.0.1"
ACK_WAIT_S = 0.5      # longest the writer thread waits for the recorder, then it spills
INTENT_ACK_S = 0.3    # an intent's fsync waits this long for the recorder, then fsyncs the spill
PROBE_S = 0.3         # a reconnected recorder must answer `open` this fast before a replay
CONNECT_S = 0.2       # loopback: a live recorder accepts at once
RETRY_S = 0.5         # between connection attempts while the recorder is down
RETRY_AFTER_SILENCE_S = 2.0  # a recorder that accepted but did not answer is tried less often,
MAX_RETRY_S = 30.0           # doubling each time up to this
REPLAY_LINES_PER_S = 20_000  # extra acknowledgement allowance when replaying a spill file
SPILL_SUFFIX = ".spill"


class RecorderSink:
    """`__call__(line)`, `sync()`, `idle()` and `close()`, as `Journal` expects of a sink."""

    def __init__(self, port: int, path: Path, gap_line: Callable[[dict], str]) -> None:
        self._port = port
        self._path = Path(path)
        self._gap_line = gap_line
        self._spill_path = self._path.parent / f"{self._path.name}.{os.getpid()}{SPILL_SUFFIX}"
        self._spill: Any = None
        self._spilled = 0
        self._down_since: float | None = None
        self._lock = threading.Lock()   # guards everything below; never held while waiting
        self._sock: socket.socket | None = None
        self._gen = 0                   # which connection; a reader of an old one does nothing
        self._k = 0
        self._acked = 0
        self._pending: deque[tuple[int, str]] = deque()   # sent lines not yet acknowledged
        self._waiters: dict[int, threading.Event] = {}
        self._next_try = float("-inf")
        self._silent_for = RETRY_AFTER_SILENCE_S
        self._gap_due = True            # this outage has no gap line in the journal yet
        self._log_unmatched_spills()

    # --- what the Journal calls --------------------------------------------

    def __call__(self, line: str) -> None:
        self._ensure_connected()
        with self._lock:
            if self._sock is not None and self._send({"op": "line", "line": line}, line):
                return
            self._spill_line(line)

    def sync(self) -> None:
        """Return once what was written is on disk: the recorder's fsync, or the spill's."""
        if self._barrier(INTENT_ACK_S):
            return
        with self._lock:
            if self._sock is not None:   # connected but silent: treat the recorder as down
                self._go_down()
                self._back_off()
            self._spill_fsync()

    def idle(self) -> None:
        """Called by the writer when its queue is empty: reconnect and replay."""
        self._ensure_connected()

    def close(self) -> None:
        with self._lock:
            self._next_try = float("-inf")
        self._ensure_connected()
        self._barrier(ACK_WAIT_S)
        with self._lock:
            self._drop_connection()
            if self._spill is not None:
                self._spill.close()
                self._spill = None
            if self._spilled:
                log.warning("recorder was down at shutdown; %d entries stay in %s",
                            self._spilled, self._spill_path)

    # --- connection -----------------------------------------------------------

    def _ensure_connected(self) -> None:
        with self._lock:
            if self._sock is not None or clock.now() < self._next_try:
                return
            self._next_try = clock.now() + RETRY_S
            try:
                sock = socket.create_connection((HOST, self._port), timeout=CONNECT_S)
            except OSError:
                return
            sock.settimeout(ACK_WAIT_S)
            self._gen += 1
            self._sock = sock
            threading.Thread(target=self._read_acks, args=(sock, self._gen),
                             name="recorder-acks", daemon=True).start()
            if not self._send({"op": "open", "path": str(self._path)}):
                return
            orphans = self._orphan_spills()
            if not self._spilled and not orphans:
                self._silent_for = RETRY_AFTER_SILENCE_S
                return
        # Something to replay. A recorder that is accepting but hung is found out here, cheaply.
        if not self._barrier(PROBE_S):
            with self._lock:
                if self._sock is not None:
                    self._go_down()
                self._back_off()
            return
        for spill_path, run_path in orphans:
            if not self._replay_file(spill_path, run_path, gap_due=True)[0]:
                return
        if self._spilled:
            done, self._gap_due = self._replay_file(self._spill_path, self._path, self._gap_due)
            if not done:
                return
        with self._lock:
            self._silent_for = RETRY_AFTER_SILENCE_S

    def _back_off(self) -> None:
        """Lock held. Try the recorder again later, and later still if it stays silent."""
        self._next_try = clock.now() + self._silent_for
        self._silent_for = min(self._silent_for * 2, MAX_RETRY_S)

    def _drop_connection(self) -> None:
        sock, self._sock = self._sock, None
        self._gen += 1
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.close()

    def _go_down(self) -> None:
        """Lock held. The recorder is gone: what it never acknowledged goes to the spill."""
        if self._down_since is None:
            self._down_since = clock.wall()
            self._gap_due = True
        self._drop_connection()
        for _, line in self._pending:
            self._spill_line(line)
        self._pending.clear()
        for event in self._waiters.values():
            event.set()
        self._waiters.clear()

    def _send(self, msg: dict, line: str | None = None) -> bool:
        """Lock held. False when the socket failed (and the sink is now down)."""
        self._k += 1
        msg["k"] = self._k
        try:
            self._sock.sendall((json.dumps(msg) + "\n").encode())
        except OSError:
            self._go_down()
            return False
        if line is not None:
            self._pending.append((self._k, line))
        return True

    def _read_acks(self, sock: socket.socket, gen: int) -> None:
        buf = b""
        while True:
            try:
                data = sock.recv(4096)
            except TimeoutError:
                if self._gen != gen:
                    return
                continue
            except OSError:
                break
            if not data:
                break
            buf += data
            *done, buf = buf.split(b"\n")
            for raw in done:
                try:
                    self._on_ack(int(json.loads(raw)["k"]), gen)
                except (ValueError, KeyError, TypeError):
                    pass
        with self._lock:
            if self._gen == gen:
                self._go_down()

    def _on_ack(self, k: int, gen: int) -> None:
        with self._lock:
            if gen != self._gen:
                return
            self._acked = max(self._acked, k)
            while self._pending and self._pending[0][0] <= k:
                self._pending.popleft()
            for key in [w for w in self._waiters if w <= k]:
                self._waiters.pop(key).set()

    def _barrier(self, timeout: float) -> bool:
        """Ask the recorder to fsync and wait for its answer; False if it is down or silent."""
        with self._lock:
            if self._sock is None or not self._send({"op": "sync"}):
                return False
            k = self._k
            event = self._waiters[k] = threading.Event()
        clock.wait(event, timeout)
        with self._lock:
            self._waiters.pop(k, None)
            return self._acked >= k

    # --- the spill file ---------------------------------------------------------

    def _spill_line(self, line: str) -> None:
        if self._spill is None:
            self._spill = self._spill_path.open("a", encoding="utf-8", buffering=1)
        if self._spilled == 0 and self._down_since is None:
            self._down_since = clock.wall()
            self._gap_due = True
        self._spill.write(line + "\n")
        self._spill.flush()
        self._spilled += 1

    def _spill_fsync(self) -> None:
        if self._spill is not None:
            with contextlib.suppress(OSError):
                os.fsync(self._spill.fileno())

    def _orphan_spills(self) -> list[tuple[Path, Path]]:
        """(spill file, its run file) for spills of earlier runs whose process is gone,
        oldest first. Lock held or not: it only reads the folder."""
        found = []
        for spill in self._path.parent.glob(f"*{SPILL_SUFFIX}"):
            if spill == self._spill_path:
                continue
            run_name, _, pid = spill.name[:-len(SPILL_SUFFIX)].rpartition(".")
            run_path = spill.with_name(run_name)
            if (pid.isdigit() and int(pid) != os.getpid() and pid_alive(int(pid))
                    or not run_path.exists()):
                continue
            found.append((spill, run_path))
        return sorted(found, key=lambda pair: pair[0].stat().st_mtime)

    def _log_unmatched_spills(self) -> None:
        for spill in self._path.parent.glob(f"*{SPILL_SUFFIX}"):
            if spill != self._spill_path and not spill.with_name(
                    spill.name[:-len(SPILL_SUFFIX)].rpartition(".")[0]).exists():
                log.error("journal spill %s has no run file; its entries are NOT in any "
                          "journal. Look at it by hand.", spill)

    def _replay_file(self, spill_path: Path, run_path: Path, gap_due: bool) -> tuple[bool, bool]:
        """Send a spill file as `late` lines (after a gap line when one is due) into `run_path`,
        and delete it once the recorder acknowledged them. Returns (done, gap_still_due). On a
        failure only the lines not yet acknowledged stay in the file."""
        own = spill_path == self._spill_path
        with self._lock:
            if self._sock is None:
                return False, gap_due
            if own and self._spill is not None:
                self._spill.flush()
            lines = spill_path.read_text("utf-8").splitlines()
            base = self._k
            ok, extra = True, 0
            if run_path != self._path:
                ok, extra = self._send({"op": "open", "path": str(run_path)}), 1
            gapped = ok and gap_due
            if gapped:
                gap = self._gap_line(_gap_body(lines, self._down_since if own else None))
                ok = self._send({"op": "line", "line": gap})
            if ok:
                for line in lines:
                    if not self._send({"op": "line", "line": line, "late": True}):
                        break
            if run_path != self._path and self._sock is not None:
                self._send({"op": "open", "path": str(self._path)})
        ok = self._sock is not None and self._barrier(
            ACK_WAIT_S + len(lines) / REPLAY_LINES_PER_S)
        with self._lock:
            if ok:
                if own and self._spill is not None:
                    self._spill.close()
                    self._spill = None
                    self._spilled = 0
                    self._down_since = None
                spill_path.unlink(missing_ok=True)
                return True, False
            heard = max(0, min(self._acked - base, self._k - base))
            gap_acked = gapped and heard > extra
            lines_acked = max(0, heard - extra - (1 if gapped else 0))
            keep = lines[lines_acked:]
            if own and self._spill is not None:
                self._spill.close()
                self._spill = None
            spill_path.write_text("".join(f"{x}\n" for x in keep), "utf-8")
            if own:
                self._spilled = len(keep)
            if self._sock is not None:
                self._go_down()
            self._back_off()
            return False, gap_due and not gap_acked


def _gap_body(lines: list[str], down_since: float | None) -> dict:
    numbers = []
    for line in lines:
        with contextlib.suppress(ValueError):
            n = json.loads(line).get("n")
            if isinstance(n, int):
                numbers.append(n)
    body: dict[str, Any] = {"reason": "recorder was not reachable", "entries": len(lines)}
    if numbers:
        body.update(first_n=min(numbers), last_n=max(numbers))
    if down_since is not None:
        body["down_since"] = datetime.fromtimestamp(down_since).astimezone().isoformat(
            timespec="milliseconds")
    return body


def sink_factory_from_env() -> Callable[[Path, Callable[[dict], str]], RecorderSink] | None:
    """A factory for `Journal(sink_for=...)` when the keeper gave this service a recorder port,
    else None (the journal writes its own file, as under the test client)."""
    try:
        port = int(os.environ["AFM_LITHO_RECORDER_PORT"])
    except (KeyError, ValueError):
        return None
    return lambda path, gap_line: RecorderSink(port, path, gap_line)
