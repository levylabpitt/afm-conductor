"""The Surface: one drift-corrected model of the sample, fused from every pass.

Every trajectory the tip executes is a measurement: raster frames at any
angle and aspect, and lithography paths (the "virtual electrodes" are scans
that happen not to be square). All of them contribute height samples to a
single multi-resolution surface over the full scanner range.

Principles (2026-08-04):
- Raw data is sacred: every contribution's raw arrays are stored untouched
  (runtime/raw/*.npz); leveling and drift correction are derived views.
- Smart leveling: per-line alignment by the median of pointwise differences
  between adjacent lines (robust to features crossing lines, which is what
  breaks naive 1st-order flattening), then a global plane fit under the
  working assumption that the sample is overall flat.
- Drift: slow x/y/z drift is modeled per contribution. New rasters are
  registered against the fused surface in their overlap (phase correlation
  for xy, median difference for z); accepted offsets update a piecewise
  drift track dx(t), dy(t), dz(t).
- Fusion: weighted average with resolution-aware weights (finer pixels win).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

RANGE_UM = 90.0                   # default (MFP-3D); per-instrument at runtime

# Characters no filesystem will take, plus the ones Windows reserves.
_UNSAFE = '<>:"/\\|?*'


def _safe_name(cid: str) -> str:
    """A contribution id, made safe to use as a filename.

    An id carries a name a person chose (an object, a pass, a segment) and is
    ALSO used as a filename, and those are different alphabets. The historical
    case was `lead-source[1/2]`, from the compiler's 8,192-point stroke cut: the
    slash made the raw writer try to create a file in a directory that does not
    exist, and the exception surfaced two layers away, inside the broadcast pump,
    where it stopped a run from ever reporting that it had finished. That cut is
    opt-in and off since 2026-09-11, so the default write path no longer produces
    that name -- but a user-chosen object name is still arbitrary text, and
    `compile_pattern(max_segment_points=...)` still produces it on request, so
    sanitising is not obsolete.

    Sanitising here, at the filesystem boundary, fixes it for every caller at
    once. The alternative, forbidding slashes in names that are meant to be
    read, would only fix the one name we happened to notice.
    """
    return "".join("_" if c in _UNSAFE or ord(c) < 32 else c
                   for c in str(cid)).strip(" .") or "unnamed"
GRID_N = 1024
PX_UM = RANGE_UM / GRID_N         # module default; instances use self.px_um


@dataclass
class Contribution:
    cid: str
    kind: str                     # "raster" | "path"
    t: float
    n_samples: int
    pixel_um: float
    offset_um: tuple[float, float, float] = (0.0, 0.0, 0.0)   # applied drift
    bounds_um: tuple[float, float, float, float] = (0, 0, 0, 0)


def smart_level(img: np.ndarray) -> np.ndarray:
    """Robust leveling for display/fusion. Raw input is never modified.

    1) Row alignment: each row is shifted by the running median of pointwise
       differences to the previous row (features spanning rows survive;
       naive per-row mean/slope removal streaks them into z artifacts).
    2) Global first-order plane removal (sample assumed overall flat).
    """
    a = np.array(img, dtype=np.float64, copy=True)
    rows, cols = a.shape
    gx, gy = np.meshgrid(np.arange(cols), np.arange(rows))
    m = np.isfinite(a)
    if m.sum() < 16:
        return a

    def robust_plane(b):
        inl = np.isfinite(b)
        plane = np.zeros_like(b)
        for _ in range(3):
            A = np.column_stack([gx[inl], gy[inl], np.ones(inl.sum())])
            coef, *_ = np.linalg.lstsq(A, b[inl], rcond=None)
            plane = coef[0] * gx + coef[1] * gy + coef[2]
            res = b - plane
            med = np.median(res[np.isfinite(res)])
            mad = np.median(np.abs(res[np.isfinite(res)] - med)) + 1e-15
            new_inl = np.isfinite(b) & (np.abs(res - med) < 3.0 * mad)
            if new_inl.sum() < 16 or new_inl.sum() == inl.sum():
                break
            inl = new_inl
        return plane

    def rank_median(x):
        # median over the least-deviating half, selected BY RANK: robust even
        # when a feature covers most of a line (sequential adjacent-row
        # differencing fails there; a global column reference does not)
        x = x[np.isfinite(x)]
        if not len(x):
            return 0.0
        k = max(4, len(x) // 2)
        idx = np.argsort(np.abs(x))[:k]
        return float(np.median(x[idx]))

    off = np.zeros(rows)
    for _ in range(3):
        b = a - off[:, None]
        plane = robust_plane(b)
        resid = b - plane
        ref = np.nanmedian(resid, axis=0)          # per-column flat reference
        for r in range(rows):
            off[r] += rank_median(resid[r] - ref)
    return a - off[:, None] - plane


class SurfaceModel:
    def __init__(self, raw_dir=None, range_um: float = RANGE_UM):
        self.range_um = float(range_um)
        self.px_um = self.range_um / GRID_N
        self.value = np.full((GRID_N, GRID_N), np.nan, dtype=np.float32)
        self.weight = np.zeros((GRID_N, GRID_N), dtype=np.float32)
        self.contributions: list[Contribution] = []
        self.drift: list[dict] = [{"t": time.time(), "dx": 0.0, "dy": 0.0, "dz": 0.0}]
        self.raw_dir = raw_dir
        self._replaying = False
        if raw_dir is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- drift --

    def current_offset(self) -> tuple[float, float, float]:
        d = self.drift[-1]
        return d["dx"], d["dy"], d["dz"]

    def _register(self, xs, ys, zs, pixel_um) -> tuple[float, float, float]:
        """Estimate this contribution's offset against the fused surface.

        xy from the best shift of a coarse cross-correlation over the overlap
        (search +-0.5 um), z from the median difference. Falls back to the
        last known drift when overlap is too small.
        """
        gi, gj = self._idx(xs, ys)
        have = np.isfinite(self.value[gi, gj])
        if have.sum() < 200:
            return self.current_offset()
        zi = zs[have]
        vi = self.value[gi[have], gj[have]]
        dz = float(np.median(vi - zi))
        xh, yh = xs[have], ys[have]

        def err_at(sx, sy):
            gi2, gj2 = self._idx(xh + sx, yh + sy)
            v2 = self.value[gi2, gj2]
            ok = np.isfinite(v2)
            if ok.sum() < 100:
                return None
            return float(np.mean(np.abs((v2[ok] - zi[ok]) - dz)))

        # two-stage search: coarse over +-span, then fine around the best
        best = (0.0, 0.0)
        span = max(self.px_um, min(0.5, 6 * pixel_um))
        for stage_span in (span, span / 3.0):
            cx, cy = best
            best_err = None
            for sx in np.linspace(cx - stage_span, cx + stage_span, 7):
                for sy in np.linspace(cy - stage_span, cy + stage_span, 7):
                    e = err_at(sx, sy)
                    if e is not None and (best_err is None or e < best_err):
                        best_err, best = e, (float(sx), float(sy))
        return best[0], best[1], dz

    # ---------------------------------------------------------------- ingest --

    def _idx(self, xs_um, ys_um):
        r = self.range_um
        gi = np.clip(((np.asarray(xs_um) / r + 0.5) * (GRID_N - 1) + 0.5).astype(int), 0, GRID_N - 1)
        gj = np.clip(((np.asarray(ys_um) / r + 0.5) * (GRID_N - 1) + 0.5).astype(int), 0, GRID_N - 1)
        return gi, gj

    def _splat(self, xs, ys, zs, w, pixel_um: float = 0.0):
        # each sample covers its true pixel footprint on the fused grid, so
        # coarse frames fill solid and fine frames override them (higher w)
        r = max(0, int(round(pixel_um / self.px_um / 2)))
        gi, gj = self._idx(xs, ys)
        val = self.value
        wt = self.weight
        n = GRID_N
        for i, j, z in zip(gi, gj, zs):
            if not np.isfinite(z):
                continue
            i0, i1 = max(i - r, 0), min(i + r + 1, n)
            j0, j1 = max(j - r, 0), min(j + r + 1, n)
            for ii in range(i0, i1):
                for jj in range(j0, j1):
                    if wt[ii, jj] <= 0 or not np.isfinite(val[ii, jj]):
                        val[ii, jj] = z
                        wt[ii, jj] = w
                    elif w >= wt[ii, jj]:
                        f = w / (w + wt[ii, jj])
                        val[ii, jj] = val[ii, jj] * (1 - f) + z * f
                        wt[ii, jj] = max(wt[ii, jj], w)

    def reload(self, limit: int | None = None) -> int:
        """Rebuild the Surface from the raw passes already on disk.

        Every pass has always been written to `raw_dir` as an .npz, on the
        stated principle that the raw measurement is the thing kept and the
        fused model is a derived view. But nothing ever read them back, so
        the Surface began every run empty while the evidence for it sat in
        the directory next door: 94 files here, and a panel that opened with
        nothing on it until somebody scanned again.

        A derived view that cannot be rederived is not a derived view, it is
        just lost work. This replays them, oldest first, so the fusion order
        matches the order they were measured in.

        Returns the number of passes fused. Failures are skipped rather than
        fatal: one unreadable file from an interrupted write must not stop
        the other ninety-three from coming back.
        """
        if self.raw_dir is None or not self.raw_dir.is_dir():
            return 0
        files = sorted(self.raw_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime)
        if limit:
            files = files[-limit:]
        n = 0
        self._replaying = True
        try:
            n = self._replay(files)
        finally:
            self._replaying = False
        return n

    def _replay(self, files) -> int:
        n = 0
        for f in files:
            try:
                with np.load(f) as d:
                    keys = set(d.files)
                    if {"height", "x_um", "y_um"} <= keys:
                        # A raster. Re-fusing runs registration again, which
                        # is correct: drift is measured against what is
                        # already known, and rebuilding in order reproduces
                        # the same sequence of comparisons.
                        self.ingest_raster(f.stem, d["height"], d["x_um"],
                                           d["y_um"], self._pixel_of(d["x_um"]))
                    elif {"x_um", "y_um", "z_nm"} <= keys:
                        self.ingest_path(f.stem, d["x_um"], d["y_um"], d["z_nm"])
                    else:
                        continue
                n += 1
            except Exception:  # noqa: BLE001, S112 - a torn file is skipped
                continue
        return n

    @staticmethod
    def _pixel_of(xs: np.ndarray) -> float:
        """Pixel pitch inferred from the stored coordinates, since it is not
        itself stored. The median absolute step along a row is robust to the
        turnaround and to a rotated frame."""
        a = np.asarray(xs, float)
        if a.ndim == 2 and a.shape[1] > 1:
            d = np.abs(np.diff(a, axis=1))
        else:
            d = np.abs(np.diff(a.ravel()))
        d = d[np.isfinite(d) & (d > 0)]
        return float(np.median(d)) if d.size else 0.06

    def ingest_raster(self, cid: str, height_m: np.ndarray, xs_um: np.ndarray,
                      ys_um: np.ndarray, pixel_um: float) -> Contribution:
        """Fuse a completed (or partial) raster: any angle, any aspect.

        height_m: (lines, pixels) raw heights; xs/ys same shape, sample-frame
        coordinates in um for every pixel.
        """
        # `not self._replaying`: reload() feeds these back in, and writing
        # each file out again as it is read is at best wasted IO and at worst
        # a torn file if the process dies mid-rebuild.
        if self.raw_dir is not None and not self._replaying:
            np.savez_compressed(self.raw_dir / f"{_safe_name(cid)}.npz",
                                height=height_m, x_um=xs_um, y_um=ys_um)
        lev = smart_level(np.where(np.isfinite(height_m), height_m, np.nan)) * 1e9
        flat = np.isfinite(lev.ravel())
        xs = xs_um.ravel()[flat]
        ys = ys_um.ravel()[flat]
        zs = lev.ravel()[flat]
        dx, dy, dz = self._register(xs, ys, zs, pixel_um)
        self.drift.append({"t": time.time(), "dx": dx, "dy": dy, "dz": dz})
        self._splat(xs + dx, ys + dy, zs + dz, w=1.0 / max(pixel_um, 1e-4),
                    pixel_um=pixel_um)
        c = Contribution(cid=cid, kind="raster", t=time.time(),
                         n_samples=int(flat.sum()), pixel_um=pixel_um,
                         offset_um=(dx, dy, dz),
                         bounds_um=(float(xs.min()), float(ys.min()),
                                    float(xs.max()), float(ys.max())))
        self.contributions.append(c)
        return c

    def ingest_path(self, cid: str, xs_um, ys_um, zs_nm, pixel_um: float = 0.06):
        """Fuse a trajectory measurement (lithography pass, spiral, anything)."""
        xs = np.asarray(xs_um, float)
        ys = np.asarray(ys_um, float)
        zs = np.asarray(zs_nm, float)
        if self.raw_dir is not None and not self._replaying:
            np.savez_compressed(self.raw_dir / f"{_safe_name(cid)}.npz",
                                x_um=xs, y_um=ys, z_nm=zs)
        ok = np.isfinite(zs)
        if not ok.any():
            return None
        zs = zs - np.median(zs[ok])       # flat-sample working assumption
        dx, dy, dz = self.current_offset()
        self._splat(xs[ok] + dx, ys[ok] + dy, zs[ok] + dz, w=1.0 / pixel_um,
                    pixel_um=pixel_um)
        c = Contribution(cid=cid, kind="path", t=time.time(),
                         n_samples=int(ok.sum()), pixel_um=pixel_um,
                         offset_um=(dx, dy, dz),
                         bounds_um=(float(xs.min()), float(ys.min()),
                                    float(xs.max()), float(ys.max())))
        self.contributions.append(c)
        return c

    # ---------------------------------------------------------------- views --

    def snapshot(self, n: int = 512, x0=None, y0=None, size=None):
        """Downsampled view of the fused surface for the UI."""
        size = self.range_um if size is None else size
        x0 = -self.range_um / 2 if x0 is None else x0
        y0 = -self.range_um / 2 if y0 is None else y0
        ax = np.linspace(x0, x0 + size, n)
        ay = np.linspace(y0, y0 + size, n)
        gx, gy = np.meshgrid(ax, ay, indexing="ij")
        gi, gj = self._idx(gx.ravel(), gy.ravel())
        v = self.value[gi, gj].reshape(n, n)
        return v

    def drift_track(self):
        t0 = self.drift[0]["t"]
        return [{"t_s": round(d["t"] - t0, 1), "dx": round(d["dx"], 4),
                 "dy": round(d["dy"], 4), "dz": round(d["dz"], 2)}
                for d in self.drift]
