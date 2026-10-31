"""The host's stop port: one line on a loopback TCP port parks the tip, with no event loop
in the way (docs/migration/plan/phase6-board/plan.md, slice K1).

The line is `STOP <who> <why>`. The port answers `sent` as soon as the park is under way,
then `confirmed` (Height came under 1 V) or `NOT confirmed`. A line that is not a STOP is
answered `error <what>` and parks nothing.

The park is the cockpit Abort's and FLEX safePark's: `safe_park.start_park`, the steps
`request_park(PARK_FULL)` takes before it awaits, called here on the connection's thread.
Hardware queues the retract on the fail-safe thread; the twin stops the sim scan and runs
`safe_park` on a thread. The outcome is recorded with `finish_without_loop`: on the loop
when there is one, directly when there is none yet (the port starts before it).

Each connection has its own thread, so a second STOP gets its `sent` at once, whatever the
first is doing. Its answer is the outcome of the park served after it arrived (the
fail-safe thread folds it into one already queued, or serves it after the running one).
A line ending `#<id>` is a door's Stop relayed by the keeper: if the door's own route
already started that park, the line shares it and parks nothing more (safety/stop_pairs.py).

Loopback only: the service has no auth, so the port must never bind another address.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading

from ... import clock
from ...hardware.failsafe_thread import ParkRequest
from ..safety.stop_pairs import split_id, start_once
from .safe_park import (
    _watchdog_wait_bound,
    finish_without_loop,
    full_park_result,
    start_park,
)

_LOG = logging.getLogger(__name__)

# Clear of the cockpit (7461 and the next 20 it slides to), the FLEX ports (29180-29182),
# the Igor image stream (9193) and the Igor bridge (9200).
DEFAULT_PORT = 9210
PORT_ENV = "AFM_LITHO_STOP_PORT"
HOST = "127.0.0.1"

# A STOP line is short; anything longer is not one.
_MAX_LINE = 512
# A client that connects and sends nothing is dropped after this; the listener is unaffected.
_READ_TIMEOUT_S = 5.0
# Bounds how long close() waits for the accept loop to notice.
_ACCEPT_TICK_S = 0.5
# More STOP connections than this at once are answered `error busy`: that many parks are
# already demanded, so one more changes nothing.
_MAX_CONNECTIONS = 32

SENT = "sent"
CONFIRMED = "confirmed"
NOT_CONFIRMED = "NOT confirmed"
# A hardware station before Igor has dialled in: the park sends nothing and checks nothing
# (safe_park.before_igor_dials_in; 2026-10-28).
NO_LINK = "NOT confirmed (not connected: nothing done)"


def port_from_env() -> int:
    """The stop port: `AFM_LITHO_STOP_PORT` if set (0 lets the OS choose), else 9210."""
    raw = os.environ.get(PORT_ENV, "").strip()
    return int(raw) if raw else DEFAULT_PORT


def parse_stop(line: str) -> tuple[str, str]:
    """`(who, why)` from `STOP <who> <why>`; raises ValueError for anything else."""
    if not (line.isascii() and line.isprintable()):
        raise ValueError("the line must be printable ASCII")
    parts = line.split(None, 2)
    if len(parts) != 3 or parts[0] != "STOP":
        raise ValueError("expected: STOP <who> <why>")
    return parts[1], parts[2].strip()


def demand_park(hub, reason: str, stop_id: str | None = None) -> ParkRequest:
    """Start the park (`safe_park.start_park`, the steps every full park takes) and return
    its request at once. The outcome is recorded when the park ends, whether or not this
    connection is still waiting for it, and without the event loop if there is none.
    A `stop_id` whose park the door already started returns that park, recorded there."""
    req, first = start_once(stop_id, lambda: start_park(hub, reason))
    if not first:
        return req

    def _record(fut):
        try:
            result = fut.result()
        except Exception as exc:                          # noqa: BLE001
            result = {"verified": False, "critical": True, "detail": repr(exc)}
        backend = "twin" if result.get("transport") == "twin" else "hardware"
        finish_without_loop(hub, reason, full_park_result(reason, result, backend))

    req.future.add_done_callback(_record)
    return req


def no_instrument_link(hub) -> bool:
    """A hardware station Igor has not dialled into: the park sends nothing."""
    from .safe_park import ROUTE_NOT_CONNECTED, stop_route
    return stop_route(hub) == ROUTE_NOT_CONNECTED


def park_outcome(request) -> str:
    """`confirmed` when the park's Height confirm passed, else `NOT confirmed` (a failed
    confirm, or no outcome within the watchdog bound)."""
    # The setter is another real thread, so this is a real-time wait (clock rule b).
    clock.wait(request.done, _watchdog_wait_bound())
    result = request.result or {}
    return CONFIRMED if result.get("verified") is True else NOT_CONFIRMED


class StopPort:
    """The listener thread and its per-connection threads. `start()` binds and returns
    self; `port` is the bound port; `close()` stops accepting (a park in flight goes on)."""

    def __init__(self, hub, port: int | None = None):
        self.hub = hub
        self._wanted = port_from_env() if port is None else int(port)
        self.port: int | None = None
        self._srv: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._slots = threading.BoundedSemaphore(_MAX_CONNECTIONS)

    def start(self) -> StopPort:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Windows: refuse to share the port, so another process cannot take over our STOPs.
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
        self._thread = threading.Thread(target=self._accept_loop, name="afm-stop-port",
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
            threading.Thread(target=self._serve_slot, args=(conn,), name="afm-stop-line",
                             daemon=True).start()

    def _serve_slot(self, conn: socket.socket) -> None:
        try:
            self._serve(conn)
        finally:
            self._slots.release()

    def _serve(self, conn: socket.socket) -> None:
        with conn:
            try:
                line = _read_line(conn)
                who, why = parse_stop(line)
                why, stop_id = split_id(why)
            except ValueError as exc:
                _reply(conn, f"error {exc}")
                return
            # The park is demanded before anything else is done with the line.
            unlinked = no_instrument_link(self.hub)
            try:
                request = demand_park(self.hub, f"STOP from {who}: {why}", stop_id)
            except Exception as exc:                      # noqa: BLE001
                _LOG.error("stop port: STOP from %s was not handed on: %r", who, exc)
                _reply(conn, f"error the park was not handed on ({exc!r})")
                return
            _reply(conn, SENT)
            outcome = park_outcome(request)
            if unlinked or no_instrument_link(self.hub):
                outcome = NO_LINK
            _reply(conn, outcome)


def _read_line(conn: socket.socket) -> str:
    """One line, without its end. A line ended by the sender closing counts at once; one
    with neither a newline nor a close counts after the 5 s read timeout. A STOP that
    lost its newline is still a STOP."""
    conn.settimeout(_READ_TIMEOUT_S)
    buf = b""
    while b"\n" not in buf and len(buf) <= _MAX_LINE:
        try:
            chunk = conn.recv(256)
        except (TimeoutError, OSError):
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
    """Send one reply line. A sender that has gone away changes nothing: the park goes on."""
    try:
        conn.sendall(text.encode("ascii", "replace") + b"\n")
    except OSError:
        pass


def start(hub, port: int | None = None) -> StopPort | None:
    """Start the service's stop port; on failure say so loudly and return None, because a
    service without its stop port must still serve its other Stop routes."""
    try:
        sp = StopPort(hub, port).start()
    except (OSError, ValueError) as exc:
        wanted = port_from_env() if port is None else port
        print(f"[stop port] NOT listening on {HOST}:{wanted} ({exc}). Stop still works "
              f"from the cockpit and FLEX; free the port or set {PORT_ENV}.", flush=True)
        _LOG.error("stop port did not start on %s:%s: %s", HOST, wanted, exc)
        return None
    print(f"[stop port] listening on {HOST}:{sp.port}", flush=True)
    return sp
