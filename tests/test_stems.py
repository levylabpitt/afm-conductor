"""`service/stems.py` -- the single owner of the run stem / on-disk layout
convention (docs/PLAN.md "Layout and naming"): `make_stem` mints a date-time
stem and disambiguates a same-second collision with a zero-padded `_NN` suffix
that MUST keep newest-last string-sort order (scan_cache.load_latest_scan_frame
and drift registration depend on it); `dir_for` is the per-kind/per-format
directory; `register`/`lookup` are the run_id -> {kind, stem} index.
"""

from __future__ import annotations

from pathlib import Path

from afm_litho.service import stems


def test_kind_dir_mapping_and_unknown_fallback():
    assert stems.kind_dir("scan") == "scans"
    assert stems.kind_dir("litho") == "writes"
    assert stems.kind_dir("iv") == "measurements"
    assert stems.kind_dir("nonsense") == "scans"     # never raises


def test_dir_for_creates_the_kind_fmt_directory(tmp_path):
    d = stems.dir_for(tmp_path, "scan", "png")
    assert d == tmp_path / "scans" / "png"
    assert d.is_dir()


def test_make_stem_is_a_date_time_string(tmp_path):
    stem = stems.make_stem(tmp_path, "scan", when=1758123456.789)
    import time
    assert stem == time.strftime("%Y%m%d_%H%M%S", time.localtime(1758123456.789))


def test_make_stem_collision_appends_zero_padded_suffix_newest_last(tmp_path):
    when = 1758123456.0
    first = stems.make_stem(tmp_path, "scan", when=when)
    # simulate the first stem having actually been written (an npz landed under
    # scans/npz/) before minting the next one at the SAME second
    (stems.dir_for(tmp_path, "scan", "npz") / f"{first}.npz").touch()

    second = stems.make_stem(tmp_path, "scan", when=when)
    assert second == f"{first}_01"
    (stems.dir_for(tmp_path, "scan", "npz") / f"{second}.npz").touch()

    third = stems.make_stem(tmp_path, "scan", when=when)
    assert third == f"{first}_02"

    # the whole point: lexicographic filename order must sort newest last
    assert first < second < third


def test_make_stem_collision_detected_across_every_format_dir(tmp_path):
    """A collision on the height-frame cache (npz dir, `.height.npz` suffix) or
    a per-channel PNG must be caught too -- not just a bare `<stem>.<ext>`."""
    when = 1758123456.0
    first = stems.make_stem(tmp_path, "scan", when=when)
    (stems.dir_for(tmp_path, "scan", "npz") / f"{first}.height.npz").touch()
    second = stems.make_stem(tmp_path, "scan", when=when)
    assert second == f"{first}_01"

    (stems.dir_for(tmp_path, "scan", "png") / f"{second}_height.png").touch()
    third = stems.make_stem(tmp_path, "scan", when=when)
    assert third == f"{first}_02"


def test_collision_is_scoped_to_its_own_kind(tmp_path):
    """A scan stem and a litho stem minted at the same instant do not collide
    with each other -- separate top-level folders, separate namespaces."""
    when = 1758123456.0
    scan_stem = stems.make_stem(tmp_path, "scan", when=when)
    (stems.dir_for(tmp_path, "scan", "npz") / f"{scan_stem}.npz").touch()
    litho_stem = stems.make_stem(tmp_path, "litho", when=when)
    assert litho_stem == scan_stem               # no "_01" -- different kind


def test_register_and_lookup_round_trip(tmp_path):
    stems.register(tmp_path, "scan-1758123456789", "scan", "20260917_143012")
    stems.register(tmp_path, "litho-1758123499000", "litho", "20260917_143059")

    rec = stems.lookup(tmp_path, "scan-1758123456789")
    assert rec == {"run_id": "scan-1758123456789", "kind": "scan",
                    "stem": "20260917_143012"}

    rec2 = stems.lookup(tmp_path, "litho-1758123499000")
    assert rec2["stem"] == "20260917_143059" and rec2["kind"] == "litho"

    assert stems.lookup(tmp_path, "no-such-id") is None
    assert stems.lookup(tmp_path, "scan-1758123456789") is not None  # index file exists

    index = Path(tmp_path / "stem_index.jsonl")
    assert index.is_file()
    assert len(index.read_text(encoding="utf-8").splitlines()) == 2


def test_lookup_on_missing_index_returns_none(tmp_path):
    assert stems.lookup(tmp_path, "scan-1") is None
