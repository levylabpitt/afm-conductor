"""The Surface comes back from disk.

User request: "it used to open with a set of data topography. that's gone."

It had not gone. Every pass has always been written to runtime/raw as an
.npz, on the stated principle that the raw measurement is what is kept and
the fused model is a derived view. But nothing ever read them back, so the
Surface began every run empty while 94 files of real measurements sat in the
directory next door. The only reason it had ever looked otherwise was a
server left running for days, holding the fused model in memory; restarting
it revealed that the persistence had only ever been half a loop.

A derived view that cannot be rederived is not derived, it is lost.
"""

from __future__ import annotations

import numpy as np
import pytest

from afm_litho.service.surface import SurfaceModel


def _raster(n=48, size_um=20.0, centre=(0.0, 0.0), bump=40e-9):
    ax = np.linspace(-size_um / 2, size_um / 2, n)
    xs, ys = np.meshgrid(ax + centre[0], ax + centre[1], indexing="xy")
    r2 = (xs - centre[0]) ** 2 + (ys - centre[1]) ** 2
    h = bump * np.exp(-r2 / (2 * (size_um / 6) ** 2))
    return h, xs, ys, float(ax[1] - ax[0])


def _covered(s):
    return float(np.isfinite(s.value).mean())


def test_a_raster_survives_a_restart(tmp_path):
    a = SurfaceModel(raw_dir=tmp_path, range_um=90.0)
    h, xs, ys, px = _raster()
    a.ingest_raster("scan-1", h, xs, ys, px)
    before = _covered(a)
    assert before > 0

    b = SurfaceModel(raw_dir=tmp_path, range_um=90.0)
    assert _covered(b) == 0, "a fresh model starts empty, as it always did"
    assert b.reload() == 1
    assert _covered(b) == pytest.approx(before, rel=0.25)
    assert len(b.contributions) == 1


def test_a_lithography_path_survives_too(tmp_path):
    """Paths are measurements as much as rasters are: the virtual electrodes
    are scans that happen not to be square."""
    a = SurfaceModel(raw_dir=tmp_path, range_um=90.0)
    t = np.linspace(0, 1, 400)
    a.ingest_path("litho-wire-1", 10 * t, 5 * np.sin(6 * t), 20 * t)
    b = SurfaceModel(raw_dir=tmp_path, range_um=90.0)
    assert b.reload() == 1
    assert b.contributions[0].kind == "path"


def test_everything_on_disk_comes_back(tmp_path):
    a = SurfaceModel(raw_dir=tmp_path, range_um=90.0)
    for i in range(4):
        h, xs, ys, px = _raster(centre=(i * 6.0, 0.0))
        a.ingest_raster(f"scan-{i}", h, xs, ys, px)
    a.ingest_path("litho-1", np.linspace(0, 9, 200), np.zeros(200), np.zeros(200))

    b = SurfaceModel(raw_dir=tmp_path, range_um=90.0)
    assert b.reload() == 5
    kinds = {c.kind for c in b.contributions}
    assert kinds == {"raster", "path"}


def test_replaying_does_not_rewrite_the_files_it_is_reading(tmp_path):
    """Otherwise a rebuild is wasted IO at best, and a torn file if the
    process dies partway through."""
    a = SurfaceModel(raw_dir=tmp_path, range_um=90.0)
    h, xs, ys, px = _raster()
    a.ingest_raster("scan-1", h, xs, ys, px)
    stamps = {p.name: (p.stat().st_mtime_ns, p.stat().st_size)
              for p in tmp_path.glob("*.npz")}

    b = SurfaceModel(raw_dir=tmp_path, range_um=90.0)
    b.reload()
    after = {p.name: (p.stat().st_mtime_ns, p.stat().st_size)
             for p in tmp_path.glob("*.npz")}
    assert after == stamps, "the raw files were touched during a replay"


def test_the_replay_flag_is_cleared_even_if_something_throws(tmp_path):
    """A stuck flag would silently stop every later pass being persisted --
    the raw data quietly ceasing to be written, which is the one thing this
    module promises never to happen."""
    s = SurfaceModel(raw_dir=tmp_path, range_um=90.0)

    def boom(files):
        raise RuntimeError("bad file")
    s._replay = boom
    with pytest.raises(RuntimeError):
        s.reload()
    assert s._replaying is False

    h, xs, ys, px = _raster()
    s.ingest_raster("scan-after", h, xs, ys, px)
    assert list(tmp_path.glob("scan-after.npz")), "persistence stayed off"


def test_an_unreadable_file_does_not_stop_the_others(tmp_path):
    a = SurfaceModel(raw_dir=tmp_path, range_um=90.0)
    h, xs, ys, px = _raster()
    a.ingest_raster("good-1", h, xs, ys, px)
    a.ingest_raster("good-2", h, xs + 8, ys, px)
    (tmp_path / "torn.npz").write_bytes(b"not an npz at all")

    b = SurfaceModel(raw_dir=tmp_path, range_um=90.0)
    assert b.reload() == 2, "one bad file must not lose the good ones"


def test_no_raw_directory_is_simply_nothing_to_restore(tmp_path):
    assert SurfaceModel(raw_dir=None).reload() == 0
    assert SurfaceModel(raw_dir=tmp_path / "never-made").reload() == 0


def test_the_pixel_pitch_is_recovered_from_the_coordinates(tmp_path):
    """It is not stored, so it has to be inferred; a wrong pitch changes the
    fusion weight and the splat radius."""
    _, xs, _, px = _raster(n=64, size_um=12.0)
    assert SurfaceModel._pixel_of(xs) == pytest.approx(px, rel=1e-6)


def test_a_limit_takes_the_most_recent(tmp_path):
    a = SurfaceModel(raw_dir=tmp_path, range_um=90.0)
    for i in range(5):
        h, xs, ys, px = _raster(centre=(i * 6.0, 0.0))
        a.ingest_raster(f"scan-{i}", h, xs, ys, px)
    b = SurfaceModel(raw_dir=tmp_path, range_um=90.0)
    assert b.reload(limit=2) == 2
