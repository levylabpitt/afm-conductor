"""A door's second Stop: `STOP <who> <why> #<id>` to the keeper's port, beside the door's
own route.

The keeper passes it to the host's stop port and keeps a note (keeper/stop_listener.py).
The door never waits for it: the line is sent from a thread of its own with a short
connect bound, so the door's route runs as before whether the keeper is up, slow or gone.
The line carries the id the door's own park runs under, so the host parks once for the
pair, whichever arrives first (safety/stop_pairs.py).

The keeper puts its port in AFM_LITHO_KEEPER_PORT for the service it starts. Without it
(the service started by hand, or under tests) nothing is sent.
"""

from __future__ import annotations

import os
import socket
import threading

PORT_ENV = "AFM_LITHO_KEEPER_PORT"
HOST = "127.0.0.1"
CONNECT_S = 0.2          # loopback: a live keeper accepts at once
_MAX_WHY = 200


def keeper_port() -> int | None:
    """The keeper's Stop port, or None when this service has no keeper to tell."""
    raw = os.environ.get(PORT_ENV, "").strip()
    try:
        port = int(raw)
    except ValueError:
        return None
    return port if port > 0 else None


def stop_line(who: str, why: str, stop_id: str | None = None) -> bytes:
    """One STOP line in printable ASCII; anything else in `why` (and any `#`, which marks
    the id) becomes `?`. The id, if any, goes last as `#<id>`."""
    clean = "".join(c if c.isascii() and c.isprintable() and c != "#" else "?"
                    for c in str(why))
    clean = " ".join(clean.split())[:_MAX_WHY] or "stop"
    tail = f" #{stop_id}" if stop_id else ""
    return f"STOP {who} {clean}{tail}\n".encode("ascii")


def tell_keeper(who: str, why: str, stop_id: str | None = None) -> threading.Thread | None:
    """Send the Stop to the keeper from a new thread and return at once (the thread, for
    tests; None when there is no keeper port). Never raises."""
    port = keeper_port()
    if port is None:
        return None
    try:
        line = stop_line(who, why, stop_id)
        t = threading.Thread(target=_send, args=(port, line), name="afm-keeper-stop",
                             daemon=True)
        t.start()
        return t
    except Exception:                                   # noqa: BLE001
        return None


def _send(port: int, line: bytes) -> None:
    """Connect, send, close. The keeper's replies are its own business (its notes)."""
    try:
        with socket.create_connection((HOST, port), timeout=CONNECT_S) as s:
            s.sendall(line)
            s.shutdown(socket.SHUT_WR)
    except OSError:
        pass
