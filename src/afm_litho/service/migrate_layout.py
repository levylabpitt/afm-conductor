"""One-shot migration from the old flat `captures/` + `scans/` layout to the
new `scans/`, `writes/`, `measurements/` per-format layout with date-time
stems (docs/IMAGE_SAVING_NOTES.md, docs/PLAN.md "Layout and naming"). Runs
once at startup (service/app.py `_startup`, via `asyncio.to_thread` --
this module does blocking filesystem work only).

MOVE ONLY, NEVER DELETE. Idempotent: a migrated file lands under its new
kind/format directory, which the old flat globs below never see again, so a
restart after a partial or complete migration is a no-op on every file it
already moved.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from . import stems

log = logging.getLogger(__name__)

# Old session ids: "scan-1758123456789", "litho-...", "iv-...".
_ID_RE = re.compile(r"^(scan|litho|iv)-(\d+)$")

# `captures/*.{npz,gwy,ibw,txt}` -- everything `captures.py`/`io.gwy`/`io.ibw`
# used to write next to a session record, before out_dir/stem existed.
_CAPTURE_EXTS = ("npz", "gwy", "ibw", "txt")


def _kind_and_when(name: str, mtime: float) -> tuple[str, float, bool]:
    """(kind, epoch-seconds, unknown?) for an old flat filename stem (the
    session id, or -- if unrecognised -- just whatever the file was called).
    A recognised id gives its real kind and the exact ms-epoch it was created
    at; an unrecognised name gets kind "scan" (migrate_runtime_layout lands it
    under `scans/`) and the file's own mtime, so it is still findable rather
    than skipped."""
    m = _ID_RE.match(name)
    if m:
        try:
            return m.group(1), int(m.group(2)) / 1000.0, False
        except ValueError:
            pass
    return "scan", mtime, True


def _unique_dest(dest_dir: Path, name: str) -> Path:
    """Never overwrite: if `name` is taken, insert a zero-padded `_01`, `_02`,
    ... before the first extension -- the SAME collision-suffix convention
    `stems.make_stem` uses -- so a stem collision during migration (two old
    ids that happen to land on the same date-time stem) can't clobber a file.
    The `.height.npz` double-extension keeps its whole tail as the
    "extension" here, so the result still ends in `.height.npz` and the
    newest-by-name glob (`scan_cache.load_latest_scan_frame`) still finds
    it."""
    dest = dest_dir / name
    if not dest.exists():
        return dest
    head, _, tail = name.partition(".")
    n = 1
    while True:
        cand = dest_dir / f"{head}_{n:02d}.{tail}"
        if not cand.exists():
            return cand
        n += 1


def migrate_runtime_layout(runtime_dir) -> dict:
    """Move every file under the old `captures/` and `scans/` flat layout into
    the new per-kind/per-format layout, registering each recognised id in
    `stem_index.jsonl` so `captures.load_capture` can still find it by its old
    session id. Returns a summary dict -- ALSO logged as one line:
    `{"moved": n, "unknown": n, "errors": [str, ...]}`. Safe on every startup.
    """
    root = Path(runtime_dir)
    moved = 0
    unknown = 0
    errors: list[str] = []

    old_captures = root / "captures"
    if old_captures.is_dir():
        for ext in _CAPTURE_EXTS:
            for path in sorted(old_captures.glob(f"*.{ext}")):
                run_id = path.stem
                try:
                    kind, when, is_unknown = _kind_and_when(run_id, path.stat().st_mtime)
                    if is_unknown:
                        unknown += 1
                        log.warning("migrate_layout: unrecognised capture id %r "
                                    "(%s) -> scans/, stem from file mtime",
                                    run_id, path.name)
                    stem = time.strftime("%Y%m%d_%H%M%S", time.localtime(when))
                    dest_dir = stems.dir_for(root, kind, ext)
                    dest = _unique_dest(dest_dir, f"{stem}.{ext}")
                    path.rename(dest)
                    # Only a RECOGNISED id is registered: `run_id` here is the
                    # raw filename stem, and for an unrecognised name that is
                    # not a real session id -- indexing it would make
                    # `stems.lookup` succeed for a string that was never a run
                    # id, which is worse than a miss (matches the old_scans
                    # loop below, which never registers either).
                    if not is_unknown:
                        stems.register(root, run_id, kind, dest.stem)
                    moved += 1
                except OSError as exc:
                    errors.append(f"{path}: {exc}")

    old_scans = root / "scans"
    if old_scans.is_dir():
        for path in sorted(old_scans.glob("*.npz")):
            # a directory already migrated into (scans/npz, scans/png, ...)
            # sits inside `scans/` too -- glob("*.npz") on the directory
            # itself only ever matches FILES directly in `scans/`, never
            # `scans/npz/*.npz`, so this is safe without an extra guard.
            run_id = path.stem
            try:
                kind, when, is_unknown = _kind_and_when(run_id, path.stat().st_mtime)
                if is_unknown:
                    unknown += 1
                    log.warning("migrate_layout: unrecognised scan-cache id %r "
                                "(%s) -> scans/, stem from file mtime",
                                run_id, path.name)
                stem = time.strftime("%Y%m%d_%H%M%S", time.localtime(when))
                dest_dir = stems.dir_for(root, "scan", "npz")
                dest = _unique_dest(dest_dir, f"{stem}.height.npz")
                path.rename(dest)
                moved += 1
            except OSError as exc:
                errors.append(f"{path}: {exc}")

    summary = {"moved": moved, "unknown": unknown, "errors": errors}
    log.info("migrate_layout: moved=%d unknown=%d errors=%d",
              moved, unknown, len(errors))
    return summary
