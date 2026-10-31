"""The heartbeat channel: one small file per child, in runtime/keeper/.

A child rewrites its file every second (atomically); any keeper, the first or a revived
one, reads it. Nothing is shared with the keeper process, so the keeper dying cannot
break the child's write, and a stuck keeper cannot block it (the K2 rule).

The file holds two lines: the child's latest READY line, then its latest HEARTBEAT line
with a counter (`n=`) so every beat reads differently from the last.
"""

from __future__ import annotations

import os
from pathlib import Path

BEAT_FILE_ENV = "AFM_LITHO_BEAT_FILE"


class BeatFile:
    """Write side, shaped like the stream the child used to write to: `write` takes
    protocol lines, `flush` puts the latest of each on disk. A disk error skips that beat;
    only the operating system decides a keeper is dead, never a missed write."""

    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._ready = ""
        self._beat = ""
        self._count = 0

    def write(self, text: str) -> int:
        for line in text.splitlines():
            if line.startswith("READY"):
                self._ready = line
            elif line.startswith("HEARTBEAT"):
                self._count += 1
                self._beat = f"{line} n={self._count}"
        return len(text)

    def flush(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(f"{self._ready}\n{self._beat}\n", encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError:
            pass            # the reader had the file open (Windows); the next beat follows


def from_env() -> BeatFile | None:
    """The child's beat file, or None when no keeper started it."""
    path = os.environ.get(BEAT_FILE_ENV, "").strip()
    return BeatFile(path) if path else None


def read_lines(path: Path | str) -> list[str] | None:
    """The READY and HEARTBEAT lines in the file now; None when it cannot be read (not
    written yet, or replaced at this very moment)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    return text.splitlines()
