"""A real child process, as the supervisor sees it.

No job object: on Windows a child outlives its parent unless it is tied to a job,
and the service must survive a keeper crash (2026-10-28).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from afm_litho.keeper import status
from afm_litho.keeper.beatfile import BEAT_FILE_ENV, read_lines
from afm_litho.keeper.pids import ProcessHandle
from afm_litho.keeper.protocol import KEEPER_PID_ENV
from afm_litho.keeper.supervisor import ChildSpec


def beat_path(name: str) -> Path:
    return status.runtime_dir() / "keeper" / f"beat-{name}.txt"


class _BeatDrain:
    """Reads a child's beat file (keeper/beatfile.py), which any keeper can read."""

    _beats: Path
    _seen: list[str]

    def drain(self) -> list[str]:
        """The READY and HEARTBEAT lines that changed since the last call."""
        now = read_lines(self._beats)
        if now is None:                 # the file could not be read this tick: nothing changed
            return []
        new = [line for i, line in enumerate(now)
               if line and (i >= len(self._seen) or self._seen[i] != line)]
        self._seen = now
        return new


class PopenHandle(_BeatDrain):
    """The child's stdout is the keeper's console. Its READY and heartbeat lines come
    through its beat file, which a revived keeper can read too."""

    def __init__(self, spec: ChildSpec, extra_args: tuple[str, ...] = ()):
        self._beats = beat_path(spec.name)
        self._beats.parent.mkdir(parents=True, exist_ok=True)
        self._beats.unlink(missing_ok=True)     # a restart must not read the last run's beat
        self._seen = []
        env = {**os.environ, **spec.env, "AFM_LITHO_KEEPER_CHILD": "1",
               KEEPER_PID_ENV: str(os.getpid()), BEAT_FILE_ENV: str(self._beats),
               "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        self._proc = subprocess.Popen(
            [*spec.argv, *extra_args], stdin=subprocess.DEVNULL, env=env)
        self.pid = self._proc.pid

    def poll(self) -> int | None:
        return self._proc.poll()

    def terminate(self) -> None:
        self._proc.terminate()


class AdoptedHandle(_BeatDrain):
    """A child the previous keeper started, still running: watched by an open process
    handle, read through its beat file. This keeper did not start it and is not its console,
    so it neither waits for it at Ctrl+C nor ends it (`owned` is False)."""

    owned = False

    def __init__(self, name: str, pid: int):
        self.pid = pid
        self._beats = beat_path(name)
        self._seen = []
        self._process = ProcessHandle(pid)

    def poll(self) -> int | None:
        return self._process.exit_code()        # the real code, so a clean quit is not restarted

    def terminate(self) -> None:
        pass            # an adopted child is never ended by the keeper

    def close(self) -> None:
        self._process.close()


def launch(spec: ChildSpec, extra_args: tuple[str, ...] = ()) -> PopenHandle:
    return PopenHandle(spec, extra_args)
