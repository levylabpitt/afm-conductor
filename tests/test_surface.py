"""Surface model: smart leveling, fusion, drift registration, path ingest."""

import numpy as np

from afm_litho.service.surface import SurfaceModel, smart_level


def _terraces(n=128, step=0.4, span_px=40):
    gx, gy = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
    return (np.floor(gx / span_px) * step).astype(float)


def test_smart_level_preserves_features():
    """A tall feature crossing many lines must not streak (the z artifact
    that naive per-line slope removal creates)."""
    n = 128
    img = _terraces(n) * 1e-9
    img[40:90, 30:100] += 50e-9          # a mesa spanning many rows
    img += np.linspace(0, 200e-9, n)[:, None]      # slow-axis tilt
    lev = smart_level(img)
    # inside-mesa rows: the background left of the mesa should still match
    # the background of non-mesa rows (no streaking offset)
    bg_mesa_rows = np.median(lev[45:85, 5:25])
    bg_other_rows = np.median(lev[5:35, 5:25])
    assert abs(bg_mesa_rows - bg_other_rows) < 2e-9
    # and the mesa is still ~50 nm tall
    mesa = np.median(lev[45:85, 40:90]) - bg_other_rows
    assert 40e-9 < mesa < 60e-9


def test_fusion_and_drift_registration():
    surf = SurfaceModel(raw_dir=None)
    n = 96
    base = _terraces(n) * 1e-9
    xs, ys = np.meshgrid(np.linspace(-5, 5, n), np.linspace(-5, 5, n), indexing="xy")
    surf.ingest_raster("a", base, xs, ys, pixel_um=10 / n)
    # second overlapping frame, shifted by exactly 2 pixels of drift in x
    shift_px = 2
    true_shift = shift_px * (10 / n)
    drifted = np.roll(base, -shift_px, axis=1)
    c = surf.ingest_raster("b", drifted, xs, ys, pixel_um=10 / n)
    dx = c.offset_um[0]
    assert abs(dx - true_shift) < 0.08   # registration recovers the drift
    v = surf.snapshot(64, -5, -5, 10)
    assert np.isfinite(v).mean() > 0.5


def test_path_ingest():
    surf = SurfaceModel(raw_dir=None)
    t = np.linspace(0, 1, 300)
    xs = -3 + 6 * t
    ys = np.zeros_like(t)
    zs = 5 * np.sin(2 * np.pi * 3 * t)
    c = surf.ingest_path("wire", xs, ys, zs)
    assert c.n_samples == 300
    v = surf.snapshot(128, -4, -1, 8)
    assert np.isfinite(v).any()


def test_a_segment_name_with_a_slash_does_not_break_the_raw_writer(tmp_path):
    """The 8192-point split names pieces `lead-source[1/2]`, and the id is
    also a filename. That slash killed a whole run: the raw writer threw,
    the broadcast pump died with it, and the panel never learned the write
    had finished. Sanitising happens at the filesystem boundary, so no
    caller has to know."""
    from afm_litho.service.surface import SurfaceModel

    m = SurfaceModel(raw_dir=tmp_path)
    c = m.ingest_path("litho-lead-source[1/2]-1785987090209",
                      [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [1.0, 1.1, 1.2, 1.1, 1.0, 1.05])
    assert c is not None
    written = list(tmp_path.glob("*.npz"))
    assert len(written) == 1
    assert "/" not in written[0].name and "\\" not in written[0].name
    # the id itself is untouched: it is for people, the filename is for disk
    assert c.cid == "litho-lead-source[1/2]-1785987090209"
