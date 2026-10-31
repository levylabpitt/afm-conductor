"""One keeper at a time, and a keeper that said it was leaving.

`keeper.lock` is held with an exclusive operating-system lock for as long as a keeper is
alive. The OS releases it when the process dies, however it dies, so a leftover file never
blocks the next keeper.

`clean-shutdown` is written before a keeper exits on purpose (Ctrl+C, or the children
quitting); a keeper that dies without it died by accident. A starting keeper clears it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from afm_litho import clock

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

# Exit code of a keeper that found another one alive.
ANOTHER_KEEPER_EXIT = 5


def lock_path(runtime: Path) -> Path:
    return runtime / "keeper" / "keeper.lock"


def marker_path(runtime: Path) -> Path:
    return runtime / "keeper" / "clean-shutdown"


class KeeperLock:
    """The held lock. `release()` lets go; process death does it too."""

    def __init__(self, fh):
        self._fh = fh

    def release(self) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            if sys.platform == "win32":
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        fh.close()


def revive_lock_path(runtime: Path) -> Path:
    return runtime / "keeper" / "revive.lock"


def acquire(runtime: Path) -> KeeperLock | None:
    """Take the keeper lock, or None when another keeper holds it."""
    return acquire_file(lock_path(runtime))


def acquire_file(path: Path) -> KeeperLock | None:
    """Take an exclusive lock on this file, or None when another holder has it. Each call
    opens its own handle, so two callers in one process exclude each other too."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+b")                      # noqa: SIM115 - held until release()
    try:
        if sys.platform == "win32":
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return KeeperLock(fh)


def write_clean_marker(runtime: Path) -> None:
    path = marker_path(runtime)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"pid={os.getpid()} wall={clock.wall():.3f}\n", encoding="utf-8")
    except OSError:
        pass            # a keeper that cannot write it still exits, and would be revived


def clear_clean_marker(runtime: Path) -> None:
    try:
        marker_path(runtime).unlink()
    except OSError:
        pass


def clean_shutdown_marked(runtime: Path) -> bool:
    return marker_path(runtime).exists()


_close_handler = None       # kept referenced: the OS calls it after this function returns


def mark_on_console_close(runtime: Path) -> None:
    """Closing the keeper's console window is a deliberate exit too (CTRL_CLOSE_EVENT):
    write the marker before Windows ends the process. Does nothing off Windows."""
    global _close_handler
    if sys.platform != "win32":
        return
    import ctypes

    handler_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_ulong)

    def handler(event: int) -> int:
        if event == 2:                  # CTRL_CLOSE_EVENT
            write_clean_marker(runtime)
        return 0                        # not handled here: the default action follows

    _close_handler = handler_type(handler)
    ctypes.windll.kernel32.SetConsoleCtrlHandler(_close_handler, True)
