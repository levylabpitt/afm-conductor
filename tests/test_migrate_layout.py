"""`service/migrate_layout.py` -- the one-shot startup migration from the old
flat `captures/`+`scans/` layout to the new per-kind/per-format layout
(docs/IMAGE_SAVING_NOTES.md, docs/PLAN.md "Layout and naming"). MOVE ONLY,
NEVER DELETE, and idempotent: a migrated file must never be picked up again by
the old flat globs.
"""

from __future__ import annotations

import logging
import time

from afm_litho.service import stems
from afm_litho.service.migrate_layout import migrate_runtime_layout


def _stem_for(ms_epoch: int) -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(ms_epoch / 1000.0))


def test_moves_recognized_capture_files_into_new_layout(tmp_path):
    old = tmp_path / "captures"
    old.mkdir()
    (old / "scan-1758123456789.npz").write_bytes(b"npz-bytes")
    (old / "scan-1758123456789.gwy").write_bytes(b"gwy-bytes")
    (old / "litho-1758123499000.ibw").write_bytes(b"ibw-bytes")

    summary = migrate_runtime_layout(tmp_path)
    assert summary == {"moved": 3, "unknown": 0, "errors": []}

    stem = _stem_for(1758123456789)
    dest = tmp_path / "scans" / "npz" / f"{stem}.npz"
    assert dest.is_file() and dest.read_bytes() == b"npz-bytes"
    assert (tmp_path / "scans" / "gwy" / f"{stem}.gwy").is_file()

    litho_stem = _stem_for(1758123499000)
    assert (tmp_path / "writes" / "ibw" / f"{litho_stem}.ibw").is_file()

    # nothing left behind in the old flat folder
    assert not list(old.glob("*.npz"))
    assert not list(old.glob("*.gwy"))
    assert not list(old.glob("*.ibw"))

    # by-id lookup now resolves through the stem index
    rec = stems.lookup(tmp_path, "scan-1758123456789")
    assert rec == {"run_id": "scan-1758123456789", "kind": "scan", "stem": stem}


def test_moves_old_scan_cache_to_height_npz(tmp_path):
    old = tmp_path / "scans"
    old.mkdir()
    (old / "scan-1758123456789.npz").write_bytes(b"frame-bytes")

    summary = migrate_runtime_layout(tmp_path)
    assert summary["moved"] == 1 and summary["errors"] == []

    stem = _stem_for(1758123456789)
    dest = tmp_path / "scans" / "npz" / f"{stem}.height.npz"
    assert dest.is_file() and dest.read_bytes() == b"frame-bytes"
    assert not (old / "scan-1758123456789.npz").exists()


def test_unrecognised_id_falls_back_to_mtime_stem_and_warns(tmp_path, caplog):
    old = tmp_path / "captures"
    old.mkdir()
    weird = old / "whatever_this_was.npz"
    weird.write_bytes(b"orphan")

    with caplog.at_level(logging.WARNING):
        summary = migrate_runtime_layout(tmp_path)
    assert summary["moved"] == 1 and summary["unknown"] == 1
    assert any("unrecognised" in r.message for r in caplog.records)

    # lands under scans/ (kind_dir's fallback), stem derived from the file's
    # own mtime rather than an id it doesn't have
    files = list((tmp_path / "scans" / "npz").glob("*.npz"))
    assert len(files) == 1
    assert files[0].read_bytes() == b"orphan"
    # NOT registered: "whatever_this_was" is a raw filename, never a real run
    # id, and indexing it would make a later `stems.lookup` on that string
    # succeed for something that was never a run id -- worse than a miss (the
    # old_scans loop below already skips registration for the same reason).
    assert stems.lookup(tmp_path, "whatever_this_was") is None


def test_second_run_is_a_no_op(tmp_path):
    old = tmp_path / "captures"
    old.mkdir()
    (old / "scan-1758123456789.npz").write_bytes(b"npz-bytes")

    first = migrate_runtime_layout(tmp_path)
    assert first["moved"] == 1

    second = migrate_runtime_layout(tmp_path)
    assert second == {"moved": 0, "unknown": 0, "errors": []}

    # the migrated file is still exactly where the first run put it
    stem = _stem_for(1758123456789)
    dest = tmp_path / "scans" / "npz" / f"{stem}.npz"
    assert dest.is_file() and dest.read_bytes() == b"npz-bytes"


def test_stem_collision_during_migration_gets_a_unique_suffix(tmp_path):
    """Two old ids whose ms-epoch floors to the SAME second (a real
    same-second collision from before this layout existed) must not clobber
    each other -- `_unique_dest` inserts a zero-padded `_01`, `_02`, ...,
    the SAME convention `stems.make_stem` uses, rather than overwriting."""
    old = tmp_path / "captures"
    old.mkdir()
    (old / "scan-1758123456001.npz").write_bytes(b"first")
    (old / "scan-1758123456999.npz").write_bytes(b"second")

    summary = migrate_runtime_layout(tmp_path)
    assert summary["moved"] == 2 and summary["errors"] == []

    stem = _stem_for(1758123456001)
    assert stem == _stem_for(1758123456999)          # same second, by construction
    d = tmp_path / "scans" / "npz"
    assert (d / f"{stem}.npz").read_bytes() == b"first"
    assert (d / f"{stem}_01.npz").read_bytes() == b"second"


def test_no_old_folders_is_a_clean_no_op(tmp_path):
    assert migrate_runtime_layout(tmp_path) == {"moved": 0, "unknown": 0, "errors": []}
