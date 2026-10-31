"""Single owner of the run stem / on-disk layout convention (docs/PLAN.md
"Layout and naming", docs/IMAGE_SAVING_NOTES.md).

Every run of one kind (scan/litho/iv) gets ONE stem -- `YYYYMMDD_HHMMSS` from its
timestamp -- shared across every file that run writes (npz, gwy, ibw, png, json
sidecar), so the files for the scan just taken sit together under a human-
findable name instead of today's flat `captures/scan-1758123456789.npz`.

    runtime/
      scans/         npz/ gwy/ ibw/ png/ json/ svg/
      writes/        npz/ gwy/ ibw/
      measurements/  txt/ ibw/            (npz dropped for new IV runs)
      stem_index.jsonl                    (run id -> kind, stem)

A same-second collision is disambiguated with a zero-padded `_NN` suffix
(`_01`, `_02`, ...) rather than a raw counter, so lexicographic filename order
still sorts newest last -- the plain stem is a strict string-prefix of its
suffixed collisions ("20260917_143012" < "20260917_143012_01"), and "_01" <
"_02" digit-by-digit. Two readers depend on this: `scan_cache.
load_latest_scan_frame` (newest-by-filename wins) and drift registration, which
reads through it.

`register`/`lookup` are the run_id -> (kind, stem) map, in the same append/scan
style as `SessionStore.record`/`get` (service/sessions.py) -- fine at this
volume (one line per scan/write/IV run). This is what lets `captures.
load_capture(runtime_dir, run_id)` -- called by `scripts/export_capture.py` and
tests, keyed by the OLD session id -- find a file that is now named by a human
stem instead.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

_INDEX = "stem_index.jsonl"

_KIND_DIR = {"scan": "scans", "litho": "writes", "iv": "measurements"}

# Every format directory a stem might land a file under, for collision probing.
_PROBE_FMTS = ("npz", "gwy", "ibw", "png", "json", "svg", "txt", "xy")


def kind_dir(kind: str) -> str:
    """The top-level folder name for a run kind. An unrecognised kind falls
    back to `scans/` (mirrors migrate_layout's unknown-id fallback) rather than
    raising -- naming must never be the reason a capture fails to save."""
    return _KIND_DIR.get(kind, "scans")


def dir_for(runtime_dir, kind: str, fmt: str) -> Path:
    """`<runtime_dir>/<kind_dir>/<fmt>/`, created if missing. This is a WRITE
    path (it mkdirs); a reader that must not create directories as a side
    effect of checking "is there anything here yet" builds the path itself
    from `kind_dir`."""
    d = Path(runtime_dir) / kind_dir(kind) / fmt
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_stem(runtime_dir, kind: str, when: float | None = None) -> str:
    """Mint `YYYYMMDD_HHMMSS` from `when` (default: now, local time -- the
    stems are for a human at the bench). A same-second collision against any
    file already on disk for this kind gets a zero-padded `_NN` suffix; see the
    module docstring for why that keeps newest-last ordering intact.
    """
    ts = time.localtime(when if when is not None else time.time())
    base = time.strftime("%Y%m%d_%H%M%S", ts)
    if not _collides(runtime_dir, kind, base):
        return base
    for n in range(1, 100):
        candidate = f"{base}_{n:02d}"
        if not _collides(runtime_dir, kind, candidate):
            return candidate
    # Exceedingly unlikely (100 saves of the same kind in the same second) --
    # fall back to a microsecond-suffixed stem rather than raise.
    us = int((when if when is not None else time.time()) * 1e6) % 1_000_000
    return f"{base}_{us:06d}"


def _collides(runtime_dir, kind: str, stem: str) -> bool:
    """True if any file under any of this kind's format directories already
    uses `stem` -- as `<stem>.ext`, `<stem>.height.npz` (the scan_cache frame
    cache), or `<stem>_<channel>.png` (the per-channel PNGs)."""
    root = Path(runtime_dir) / kind_dir(kind)
    for fmt in _PROBE_FMTS:
        d = root / fmt
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p.name == stem or p.name.startswith(stem + ".") or \
               p.name.startswith(stem + "_"):
                return True
    return False


def register(runtime_dir, run_id: str, kind: str, stem: str) -> None:
    """Append `{run_id, kind, stem}` to `stem_index.jsonl` (append-only, same
    pattern as `SessionStore.record`)."""
    path = Path(runtime_dir) / _INDEX
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"run_id": str(run_id), "kind": str(kind), "stem": str(stem)}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def lookup(runtime_dir, run_id: str) -> dict | None:
    """The `{run_id, kind, stem}` record for `run_id`, or None if it was never
    registered (an old un-migrated id, or a run that never wrote a stem-named
    file). Linear scan of `stem_index.jsonl`, same pattern as
    `SessionStore.get`."""
    path = Path(runtime_dir) / _INDEX
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("run_id") == run_id:
            return rec
    return None
