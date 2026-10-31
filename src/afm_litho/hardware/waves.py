"""Hand a numpy wave to Igor by file -- the capture's Make/LoadWave transport.

A path wave is far too large to pass through the ActiveX command string, so the
capture (igor-commands-commented.txt:48-56) writes the values to a temp text
file, `Make`s a wave of the right length, `LoadWave`s the file (which auto-names
it <name>_temp), then copies into the target. We reproduce that exactly, so what
reaches Igor is the flow the LabVIEW program already uses.

The LoadWave flags and the name-header format are taken from the capture; they
are the part to confirm on the AFM PC, since the round-trip cannot run without a
live Igor. These are Igor operations, not ARC functions, so they go through
execute() (Igor-level errors), not arc_call().
"""

from __future__ import annotations

import os
import tempfile


def to_igor_path(win_path: str) -> str:
    """Windows path -> Igor's colon-separated path (C:\\a\\b.txt -> C:a:b.txt).

    Igor uses ':' as its native path separator. The drive's own colon plus the
    first backslash would make a double colon (C:\\ -> C::), so collapse runs of
    colons back to one. (Local drive paths only -- UNC is not handled.)
    """
    p = win_path.replace("\\", ":").replace("/", ":")
    while "::" in p:
        p = p.replace("::", ":")
    return p


def write_temp_wave(array, name: str, tmpdir: str | None = None) -> str:
    """Write `array` to an Igor-loadable temp text file; return its path.

    Format for LoadWave/J/D/W: a one-line name header (<name>_temp) then one
    value per line. %.12g preserves double precision compactly.
    """
    tmpdir = tmpdir or tempfile.gettempdir()
    path = os.path.join(tmpdir, f"afm_{name}.txt")
    with open(path, "w", encoding="ascii") as f:
        f.write(f"{name}_temp\n")
        for v in array:
            f.write(f"{float(v):.12g}\n")
    return path


def push_wave(igor, name: str, array, tmpdir: str | None = None) -> int:
    """Load `array` into an Igor wave `name` via the capture's file hand-off.

    Returns the point count. Mirrors the capture's three-step move: Make
    (pre-allocate), LoadWave (reads the file, auto-names <name>_temp), copy.
    """
    n = len(array)
    path = write_temp_wave(array, name, tmpdir)
    igor.execute(f"Make/O/N={n} {name}")
    igor.execute(f'LoadWave/O/A/J/D/W/K=0 "{to_igor_path(path)}"')
    igor.execute(f"{name} = {name}_temp")
    return n
