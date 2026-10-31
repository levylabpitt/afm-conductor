"""The S key in the keeper's console window sends a Stop to the keeper's own Stop port.

A small daemon thread polls the keyboard (msvcrt on Windows). Ctrl+C is not touched, and
there is nothing to do where the keeper has no console. Stdlib only.
"""

from __future__ import annotations

import sys
import threading
from typing import Callable

from afm_litho import clock
from afm_litho.keeper import stop_client

POLL_S = 0.1
WHO, WHY = "keeper-console", "key"


def _console_poll() -> Callable[[], str | None] | None:
    """A function returning the next pressed key or None; None itself when no console."""
    if sys.platform != "win32":
        return None
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return None
        import msvcrt
    except (ImportError, ValueError, OSError):
        return None

    def poll() -> str | None:
        return msvcrt.getwch() if msvcrt.kbhit() else None
    return poll


def run(poll: Callable[[], str | None], send: Callable[[], None], stop: threading.Event,
        *, tick_s: float = POLL_S) -> None:
    """One `send()` per S or s pressed; every other key is ignored."""
    while not stop.is_set():
        key = poll()
        if key is None:
            clock.wait(stop, tick_s)
        elif key in ("s", "S"):
            send()


def start(*, keeper_port: int | None = None, host_port: int | None = None,
          poll: Callable[[], str | None] | None = None,
          say: Callable[[str], None] | None = None) -> threading.Event | None:
    """Start the key thread; returns its stop event, or None when there is no console."""
    poll = poll or _console_poll()
    if poll is None:
        return None
    say = say or (lambda m: print(m, flush=True))

    def send() -> None:
        say("[keeper] S pressed: sending Stop")
        stop_client.send_stop(WHO, WHY, lambda m: say(f"[keeper] {m}"),
                              keeper_port=keeper_port, host_port=host_port)

    stop = threading.Event()
    threading.Thread(target=run, args=(poll, send, stop), name="keeper-console-key",
                     daemon=True).start()
    say("[keeper] press S to Stop (park)")
    return stop
