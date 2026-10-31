"""Shared helpers for the Igor install scripts: locate Igor's own folders by
asking the running Igor over COM (SpecialDirPath), with a filesystem fallback.

Igor 6 returns old Mac/HFS colon paths on Windows; win_path() converts them.
find_igor_folder() returns the 'Igor Pro Folder' -- the one that contains the
app-folder 'Igor Extensions' this Asylum Igor actually scans (user-files
extensions are NOT scanned here, learned the hard way).
"""

from __future__ import annotations

from pathlib import Path

from .igor import Igor, IgorError


def win_path(p: str) -> str:
    """Convert an Igor-6 HFS colon path ("C:Users:<user>:...:") to a Windows path.
    Paths already using \\ or / are returned unchanged."""
    p = p.strip()
    if not p or "\\" in p or "/" in p:
        return p
    parts = [x for x in p.split(":") if x != ""]
    return "\\".join([parts[0] + ":"] + parts[1:]) if parts else p


def igor_dir(igor: Igor, which: str) -> Path | None:
    """A folder Igor reports via SpecialDirPath, or None if the id is unknown."""
    try:
        p = igor.string(f'SpecialDirPath("{which}", 0, 0, 0)').strip()
    except IgorError:
        return None
    return Path(win_path(p)) if p else None


def find_igor_folder(igor: Igor,
                     search_roots=(Path("C:/AsylumResearch"),)) -> Path | None:
    """The 'Igor Pro Folder' -- the one holding 'Igor Extensions'."""
    cands: list[Path] = []
    for which in ("Igor Application", "Igor Executable"):
        p = igor_dir(igor, which)
        if p:
            cands += [p, p.parent]
    for c in cands:
        if (c / "Igor Extensions").is_dir():
            return c
    for root in search_roots:
        if root.is_dir():
            for ext in root.rglob("Igor Extensions"):
                if (ext.parent / "More Extensions").is_dir():
                    return ext.parent
    return None


def igor_exe_dir(folder: Path) -> Path:
    """Directory containing Igor.exe (where dependent DLLs are reliably found),
    or the folder itself if Igor.exe isn't located under it."""
    for exe in folder.rglob("Igor.exe"):
        return exe.parent
    return folder
