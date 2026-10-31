"""The keeper's Stop listener (docs/migration/plan/phase6-board/plan.md, slice K4).

A door sends `STOP <who> <why>` to the keeper's loopback port, beside its own route. The
keeper passes it straight on to the host's stop port and relays the host's replies: `sent`,
then `confirmed` or `NOT confirmed`. Only then does it write its note, so the note never
delays the Stop (withdraw before the forensic record).

The keeper never talks to Igor. When the host does not answer, the keeper says so and
writes the note; it makes nothing safe itself.

    host alive   relays `sent`, then the host's outcome
    host slow    no `sent` within SENT_WAIT_S: `NOT confirmed (sent, no answer from the host)`
    host dead    no connection within CONNECT_S: `NOT confirmed (host unreachable)`

Loopback only, as the host's port: anything on this machine may send a Stop, nothing else.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
from typing import Callable

from afm_litho import clock

PORT_ENV = "AFM_LITHO_KEEPER_PORT"
HOST_PORT_ENV = "AFM_LITHO_STOP_PORT"
# Beside the host's stop port (9210, service/host/stop_port.py), clear of the cockpit
# (7461 and the 20 after it), FLEX (29180-29182), the image stream (9193) and the bridge (9200).
DEFAULT_PORT = 9211
DEFAULT_HOST_PORT = 9210
HOST = "127.0.0.1"

SENT = "sent"
CONFIRMED = "confirmed"
NOT_CONFIRMED = "NOT confirmed"

_MAX_LINE = 512            # the host's limit for one STOP line
_READ_TIMEOUT_S = 5.0      # a client that connects and sends nothing is dropped
_ACCEPT_TICK_S = 0.5       # bounds how long close() waits for the accept loop
_MAX_CONNECTIONS = 32      # more at once are answered `error busy`; that many Stops are in hand
CONNECT_S = 0.5            # a live host accepts on loopback at once
# The host answers `sent` once its park is handed on (stop_port.py); a cold start of its
# fail-safe thread (COM attach) can take about 2 s first.
SENT_WAIT_S = 3.0
# The host answers its outcome within its watchdog bound, 12 s (safe_park._watchdog_wait_bound).
OUTCOME_WAIT_S = 15.0
_VIA = " (via keeper)"


def port_from_env() -> int:
    """The keeper's own Stop port: `AFM_LITHO_KEEPER_PORT` if set (0: the OS chooses)."""
    raw = os.environ.get(PORT_ENV, "").strip()
    return int(raw) if raw else DEFAULT_PORT


def host_port_from_env() -> int:
    """The host's stop port, read the way the service reads it (stop_port.port_from_env)."""
    raw = os.environ.get(HOST_PORT_ENV, "").strip()
    return int(raw) if raw else DEFAULT_HOST_PORT


def parse_stop(line: str) -> tuple[str, str]:
    """`(who, why)` from `STOP <who> <why>`; raises ValueError for anything else. The same
    rule as the host's port, written out here because the keeper may not import the service."""
    if not (line.isascii() and line.isprintable()):
        raise ValueError("the line must be printable ASCII")
    parts = line.split(None, 2)
    if len(parts) != 3 or parts[0] != "STOP" or not parts[2].strip():
        raise ValueError("expected: STOP <who> <why>")
    return parts[1], parts[2].strip()


def forward_line(who: str, why: str) -> str:
    """The line sent to the host: the door's own, marked as relayed, within the host's limit."""
    head = f"STOP {who} "
    room = _MAX_LINE - len(head) - len(_VIA)
    return head + why[:max(room, 1)] + _VIA


class Relay:
    """What happened to one Stop passed to the host: `replies` as relayed, and `outcome`."""

    def __init__(self) -> None:
        self.replies: list[str] = []
        self.outcome = ""
        self.connected = False
        self.forwarded_wall: float | None = None


def relay(host_port: int, who: str, why: str, reply: Callable[[str], None]) -> Relay:
    """Pass one Stop to the host's stop port and hand each answer to `reply` as it comes."""
    r = Relay()
    try:
        sock = socket.create_connection((HOST, host_port), timeout=CONNECT_S)
    except OSError as exc:
        r.outcome = f"{NOT_CONFIRMED} (host unreachable: {type(exc).__name__})"
        reply(r.outcome)
        return r
    with sock:
        r.connected = True
        try:
            sock.sendall((forward_line(who, why) + "\n").encode("ascii"))
            r.forwarded_wall = clock.wall()
        except OSError as exc:
            r.outcome = f"{NOT_CONFIRMED} (host unreachable: {type(exc).__name__})"
            reply(r.outcome)
            return r
        lines = _Lines(sock)
        first = lines.next(SENT_WAIT_S)
        if first is None:
            r.outcome = f"{NOT_CONFIRMED} (sent, no answer from the host)"
            reply(r.outcome)
            return r
        r.replies.append(first)
        if first != SENT:
            r.outcome = f"{NOT_CONFIRMED} (host answered: {first})"
            reply(r.outcome)
            return r
        reply(SENT)
        second = lines.next(OUTCOME_WAIT_S)
        if second is None:
            r.outcome = f"{NOT_CONFIRMED} (no outcome from the host)"
        else:
            r.replies.append(second)
            r.outcome = second
        reply(r.outcome)
    return r


class _Lines:
    """Reply lines from the host, each waited for up to its own real-time bound."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock, self._buf = sock, b""

    def next(self, timeout_s: float) -> str | None:
        self._sock.settimeout(timeout_s)
        while b"\n" not in self._buf:
            try:
                chunk = self._sock.recv(256)
            except OSError:            # a timeout or a reset: no answer
                return None
            if not chunk:
                return None
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode("ascii", "replace").rstrip("\r")


class StopListener:
    """The keeper's Stop port: a listener thread and one thread per connection.
    `start()` binds and returns self; `port` is the bound port; `close()` stops accepting."""

    def __init__(self, notes, *, port: int | None = None, host_port: int | None = None):
        self._notes = notes
        self._wanted = port_from_env() if port is None else int(port)
        self.host_port = host_port_from_env() if host_port is None else int(host_port)
        self.port: int | None = None
        self._srv: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._slots = threading.BoundedSemaphore(_MAX_CONNECTIONS)

    def start(self) -> StopListener:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Windows: refuse to share the port, so another process cannot take over our Stops.
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            srv.bind((HOST, self._wanted))
            srv.listen(8)
        except OSError:
            srv.close()
            raise
        srv.settimeout(_ACCEPT_TICK_S)
        self._srv = srv
        self.port = srv.getsockname()[1]
        self._thread = threading.Thread(target=self._accept_loop, name="keeper-stop-port",
                                        daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        srv, self._srv = self._srv, None
        if srv is not None:
            try:
                srv.close()
            except OSError:
                pass
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(2 * _ACCEPT_TICK_S)

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            srv = self._srv
            if srv is None:
                return
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                continue
            if not self._slots.acquire(blocking=False):
                with conn:
                    _reply(conn, "error busy")
                continue
            threading.Thread(target=self._serve_slot, args=(conn,), name="keeper-stop-line",
                             daemon=True).start()

    def _serve_slot(self, conn: socket.socket) -> None:
        try:
            self._serve(conn)
        finally:
            self._slots.release()

    def _serve(self, conn: socket.socket) -> None:
        with conn:
            try:
                who, why = parse_stop(_read_line(conn))
            except ValueError as exc:
                _reply(conn, f"error {exc}")
                return
            received = clock.wall()
            mono = clock.now()
            r = relay(self.host_port, who, why, lambda text: _reply(conn, text))
        # The Stop has been passed on and answered; only now the note.
        self._notes.write_stop(who=who, why=why, received_wall=received, mono=mono,
                               host_port=self.host_port, relay=r)


def _read_line(conn: socket.socket) -> str:
    """One line, without its end; a line ended by the sender closing counts at once."""
    conn.settimeout(_READ_TIMEOUT_S)
    buf = b""
    while b"\n" not in buf and len(buf) <= _MAX_LINE:
        try:
            chunk = conn.recv(256)
        except OSError:
            break
        if not chunk:
            break
        buf += chunk
    line = buf.split(b"\n", 1)[0]
    if len(line) > _MAX_LINE:
        raise ValueError("line too long")
    if not line.strip():
        raise ValueError("no line received")
    try:
        return line.decode("ascii").rstrip("\r")
    except UnicodeDecodeError:
        raise ValueError("the line must be printable ASCII") from None


def _reply(conn: socket.socket, text: str) -> None:
    """Send one reply line. A sender that has gone away changes nothing: the Stop goes on."""
    try:
        conn.sendall(text.encode("ascii", "replace") + b"\n")
    except OSError:
        pass


def start(notes, *, port: int | None = None, host_port: int | None = None,
          say: Callable[[str], None] | None = None) -> StopListener | None:
    """Start the keeper's Stop port; on failure say so and return None. The keeper runs on:
    every door still has its own route to the host."""
    say = say or (lambda m: print(m, flush=True))
    try:
        sl = StopListener(notes, port=port, host_port=host_port).start()
    except (OSError, ValueError) as exc:
        wanted = port_from_env() if port is None else port
        say(f"[keeper] Stop port NOT listening on {HOST}:{wanted} ({exc}). Stop still works "
            f"from the cockpit and FLEX; free the port or set {PORT_ENV}.")
        return None
    say(f"[keeper] Stop port listening on {HOST}:{sl.port}; passes Stop to {HOST}:"
        f"{sl.host_port}")
    return sl
