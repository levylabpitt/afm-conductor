"""Bring a dead keeper back (docs/migration/plan/phase6-board/keeper-revival.md).

Any process may call `Reviver.revive()`; today it is the service. The operating system
decides a keeper is dead (the service's process handle, service/keeper_link.py), never a
late heartbeat. Exactly one revived keeper results, however many callers try at once:

    revive.lock   held by the reviver from the first check until the new keeper holds
                  keeper.lock, so a second reviver finds it taken and does nothing.
    keeper.lock   probed, not kept: if someone can take it, no keeper is alive. It is let
                  go before the new keeper starts, because a lock does not pass to a child
                  on Windows; revive.lock covers that gap.

A keeper that exited on purpose left the clean-shutdown marker and is not revived. The new
keeper runs in adopt mode (keeper/adopt.py): it takes over the running children.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path

from afm_litho import clock
from afm_litho.keeper import lock
from afm_litho.keeper.beatfile import BEAT_FILE_ENV
from afm_litho.keeper.protocol import KEEPER_PID_ENV

ADOPT_ARG = "--adopt"
REVIVED_BY_ARG = "--revived-by"   # pid of the reviver: a recorded child
MAX_REVIVALS = 3
REVIVAL_WINDOW_S = 600.0       # 3 revivals in 10 minutes, then stop trying
UP_WAIT_S = 20.0               # the new keeper holds the lock within this, or it failed
UP_POLL_S = 0.1

REVIVED, CLEAN, ALIVE, BUSY, LIMIT, FAILED = (
    "revived", "clean-shutdown", "keeper-alive", "another-reviver", "limit", "failed")


def spawn_adopt_keeper(runtime: Path, by_pid: int) -> subprocess.Popen:
    """Start a keeper in adopt mode. On Windows it gets a console window of its own (the S
    key) and is not tied to this process, so it outlives the service."""
    env = {k: v for k, v in os.environ.items()
           if k not in (BEAT_FILE_ENV, KEEPER_PID_ENV, "AFM_LITHO_KEEPER_CHILD")}
    env["AFM_LITHO_RUNTIME"] = str(runtime)
    argv = [sys.executable, "-m", "afm_litho.keeper", ADOPT_ARG, REVIVED_BY_ARG,
            str(by_pid)]
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
        return subprocess.Popen(argv, env=env, creationflags=flags, close_fds=True)
    return subprocess.Popen(argv, env=env, start_new_session=True, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Reviver:
    def __init__(self, runtime: Path, spawn=spawn_adopt_keeper):
        self._runtime = Path(runtime)
        self._spawn = spawn
        self._times: deque[float] = deque()      # when this process started a revival

    def revive(self) -> tuple[str, str]:
        """(outcome, words). REVIVED means a new keeper holds the lock; it is in contact
        only once the service has heard from it (service/keeper_link.py)."""
        guard = lock.acquire_file(lock.revive_lock_path(self._runtime))
        if guard is None:
            return BUSY, "another process is reviving the keeper"
        try:
            return self._revive_locked()
        finally:
            guard.release()

    def _revive_locked(self) -> tuple[str, str]:
        if lock.clean_shutdown_marked(self._runtime):
            return CLEAN, "the keeper shut down on purpose; not reviving it"
        probe = lock.acquire(self._runtime)
        if probe is None:
            return ALIVE, "a keeper is alive (it holds the keeper lock)"
        probe.release()
        now = clock.now()
        while self._times and now - self._times[0] > REVIVAL_WINDOW_S:
            self._times.popleft()
        if len(self._times) >= MAX_REVIVALS:
            return LIMIT, (f"the keeper died again after {MAX_REVIVALS} revivals in "
                           f"{REVIVAL_WINDOW_S / 60:.0f} minutes; NOT reviving it again")
        self._times.append(now)
        try:
            proc = self._spawn(self._runtime, os.getpid())
        except OSError as exc:
            return FAILED, f"could not start a keeper ({exc})"
        return self._wait_until_up(proc)

    def _wait_until_up(self, proc) -> tuple[str, str]:
        end = clock.now() + UP_WAIT_S
        while clock.now() < end:
            if proc.poll() is not None:
                return FAILED, f"the new keeper exited at once (code {proc.poll()})"
            held = lock.acquire(self._runtime)
            if held is None:
                return REVIVED, f"a new keeper (pid {proc.pid}) holds the keeper lock"
            held.release()
            clock.sleep(UP_POLL_S)
        return FAILED, f"the new keeper (pid {proc.pid}) did not take the lock in {UP_WAIT_S:.0f} s"
