"""Send one Stop to the keeper, or to the host's stop port when the keeper does not answer.

Used by `python -m afm_litho.keeper stop` (the `Stop AFM.cmd` launcher) and by the S key in
the keeper's console. A STOP with no `#id` always parks. Stdlib only.
"""

from __future__ import annotations

import socket
from typing import Callable

from afm_litho.keeper import stop_listener
from afm_litho.keeper.stop_listener import HOST, NOT_CONFIRMED

CONNECT_S = 1.0
# The keeper says `sent` within 3 s or it is not working: then go to the host at once.
SENT_WAIT_S = 3.0
# Longer than the keeper's own outcome bound (15 s), so its verdict arrives first.
OUTCOME_WAIT_S = 20.0


def _exchange(port: int, who: str, why: str, say: Callable[[str], None]) -> list[str] | None:
    """Send the line, say each reply as it comes. None: nothing was reached or answered."""
    try:
        sock = socket.create_connection((HOST, port), timeout=CONNECT_S)
    except OSError:
        return None
    replies: list[str] = []
    with sock:
        try:
            sock.sendall(f"STOP {who} {why}\n".encode("ascii"))
        except OSError:
            return None
        buf = b""
        while True:
            sock.settimeout(SENT_WAIT_S if not replies else OUTCOME_WAIT_S)
            try:
                chunk = sock.recv(256)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode("ascii", "replace").rstrip("\r")
                replies.append(text)
                say(text)
    return replies or None


def send_stop(who: str, why: str, say: Callable[[str], None] = print, *,
              keeper_port: int | None = None, host_port: int | None = None) -> str:
    """The keeper first; the host's port only if the keeper gave no answer. Returns the
    last reply line (the outcome), or a `NOT confirmed` line when neither answered."""
    keeper_port = stop_listener.port_from_env() if keeper_port is None else keeper_port
    host_port = stop_listener.host_port_from_env() if host_port is None else host_port
    replies = _exchange(keeper_port, who, why, say)
    if replies is None:
        say(f"keeper on {HOST}:{keeper_port} not answering; sending to the host stop port "
            f"{host_port}")
        replies = _exchange(host_port, who, why, say)
    if replies is None:
        outcome = f"{NOT_CONFIRMED} (neither the keeper nor the host stop port answered)"
        say(outcome)
        return outcome
    return replies[-1]


def main_stop(argv: list[str] | None = None) -> int:
    """`python -m afm_litho.keeper stop`: exit 0 when the outcome is `confirmed`."""
    outcome = send_stop("stop-cmd", "double-click")
    return 0 if outcome == stop_listener.CONFIRMED else 1
