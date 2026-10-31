"""FFT phase-correlation drift estimator for the Connect tab (Phase 2).

Thermal drift walks the sample under the tip over minutes. Once the operator has
marked electrodes as PORTS (Phase 1), we keep those marks locked to the moving
features: on every completed scan we estimate the (dx, dy) TRANSLATION of the
sample versus a designated REFERENCE frame, then optionally TRACK (shift the
ports) and COMPENSATE (shift the next scan's offset) by it.

We mirror the vendor's approach -- ScanMaster.ipf `CalcImageOffset` cross-
correlates two frames' FFTs and refines the peak to subpixel -- rather than the
coarse grid search in `surface.py::_register`, because phase correlation is
sharper and its peak is a clean delta.

SIGN / AXIS CONVENTION (pinned by tests/test_drift.py's twin ground-truth)
-------------------------------------------------------------------------
`estimate_shift(ref, new, ...)` returns the displacement of the sample's
FEATURES in the image, going FROM the reference frame TO the new frame, in the
image's own (fast, slow) axes:

  * `dx_um` > 0  -> a feature sits FURTHER along the +fast axis (higher column)
                   in `new` than it did in `ref`.
  * `dy_um` > 0  -> a feature sits FURTHER along the +slow axis (higher row)
                   in `new` than it did in `ref`.

At angle_deg == 0 the fast axis is lab +x and the slow axis is lab +y, so
(dx_um, dy_um) are the lab drift directly; at a non-zero scan angle the caller
rotates the frame-axis result into lab XY with `litho.frame.rotate`.

Because the feature moved by (+dx, +dy) in the image, to keep a mark on its
electrode both the port and the scan window follow the feature the SAME way:

  * PORTS (TRACK):      `position_um += (dx_um, dy_um)`  (lab).
  * SCAN OFFSET (COMP): `offset_um   += (total_x, total_y)`  (lab, accumulated).

Both add the drift -- one consistent sign, verified end to end by the twin test.

Deps: numpy.fft only (no new dependency). Per-line flattening reuses
`hw_scan._flatten1` so the preprocessing matches the display pipeline.
"""

from __future__ import annotations

import time

import numpy as np

from .hw_scan import _flatten1

# A phase-correlation peak on two genuinely-overlapping broadband frames is a
# sharp delta -- peak/mean(|c|) runs many tens upward (measured ~40-90 on real
# and synthetic shifts). Two UNCORRELATED (or blank/low-contrast) frames give a
# flat correlation surface with the ratio ~5-6, so rejecting below this keeps
# ports from being shoved around on noise while comfortably accepting a noisy
# but genuine registration.
PEAK_RATIO_MIN = 10.0


def _prep(grid, rows: int, pixels: int) -> np.ndarray | None:
    """Flatten, fill, and window one (rows, pixels) height frame for the FFT.

    Per-line 1st-order flatten (reuse `_flatten1`) removes scan tilt; NaN
    (unscanned pixels) becomes the finite-pixel mean so the FFT sees a filled
    frame; a 2-D Hann window tapers the edges to nothing to kill the wrap-around
    the periodic FFT would otherwise correlate on. Returns None for an all-NaN
    frame (nothing to register).
    """
    a = np.asarray(grid, dtype=np.float64).reshape(rows, pixels)
    a = np.vstack([_flatten1(a[r]) for r in range(rows)])
    finite = np.isfinite(a)
    if not finite.any():
        return None
    a = np.where(finite, a, float(a[finite].mean()))
    a = a - a.mean()
    win = np.hanning(rows)[:, None] * np.hanning(pixels)[None, :]
    return a * win


def _parabolic(cm1: float, c0: float, cp1: float) -> float:
    """Subpixel peak offset from a 3-point parabola fit (peak at index+return).

    Standard vertex of the parabola through (-1, cm1), (0, c0), (+1, cp1);
    clamped to +-1 so a degenerate/flat triple can never throw the peak a whole
    pixel. Returns 0.0 on a zero curvature denominator.
    """
    denom = cm1 - 2.0 * c0 + cp1
    if denom == 0.0:
        return 0.0
    return float(np.clip(0.5 * (cm1 - cp1) / denom, -1.0, 1.0))


def estimate_shift(ref, new, size_um: float, pixels: int, rows: int) -> dict:
    """Estimate the (dx, dy) drift of `new` relative to `ref` by phase correlation.

    `ref`, `new`: (rows, pixels) height grids (nm) of the SAME geometry. Returns
    `{dx_um, dy_um, peak, quality, ok}` in the image's (fast=+x-col, slow=+y-row)
    axes -- see the module docstring for the sign convention. `ok` is False (and
    the caller should NOT move ports) when `quality < PEAK_RATIO_MIN` or a frame
    is empty.
    """
    a = _prep(ref, rows, pixels)
    b = _prep(new, rows, pixels)
    if a is None or b is None:
        return {"dx_um": 0.0, "dy_um": 0.0, "peak": 0.0, "quality": 0.0, "ok": False}

    # Cross-power spectrum with F1 = fft(new), F2 = fft(ref): R = F1 * conj(F2),
    # normalized to unit magnitude. The ifft peak then lands at the (row, col)
    # by which `new`'s features are shifted relative to `ref` -- so a positive
    # peak offset from centre IS a positive feature displacement (dx, dy).
    f1 = np.fft.fft2(b)
    f2 = np.fft.fft2(a)
    r = f1 * np.conj(f2)
    r /= np.abs(r) + 1e-12
    c = np.fft.fftshift(np.fft.ifft2(r).real)

    pr, pc = np.unravel_index(int(np.argmax(c)), c.shape)
    peak = float(c[pr, pc])
    quality = peak / (float(np.abs(c).mean()) + 1e-12)

    # subpixel refinement per axis (wrap the neighbour indices at the borders)
    col = c[:, pc]
    line = c[pr, :]
    d_r = _parabolic(col[(pr - 1) % rows], col[pr], col[(pr + 1) % rows])
    d_c = _parabolic(line[(pc - 1) % pixels], line[pc], line[(pc + 1) % pixels])

    drow = (pr - rows // 2) + d_r          # +slow-axis (row) feature shift, px
    dcol = (pc - pixels // 2) + d_c        # +fast-axis (col) feature shift, px

    dx_um = dcol * size_um / pixels        # fast axis == lab +x at angle 0
    dy_um = drow * size_um / rows          # slow axis == lab +y at angle 0
    return {"dx_um": float(dx_um), "dy_um": float(dy_um), "peak": peak,
            "quality": float(quality), "ok": bool(quality >= PEAK_RATIO_MIN)}


# --------------------------------------------------------------- integration --

def new_state() -> dict:
    """The hub's fresh drift state (also the shape a re-base/clear returns to)."""
    return {"enabled": False, "reference": None, "track": [], "total": [0.0, 0.0]}


def reference_meta(ref: dict | None) -> dict | None:
    """JSON-safe geometry of a reference frame (drops the height array)."""
    if not ref:
        return None
    return {k: ref[k] for k in ("rows", "pixels", "size_um", "angle_deg",
                                "x_offset_um", "y_offset_um", "t") if k in ref}


def frame_to_reference(frame: dict) -> dict:
    """Build a reference record (height as a (rows,pixels) array) from a cached
    scan frame -- the wire dict `scan_cache.load_latest_scan_frame` returns, or
    the in-memory frame the scan pump just wrote."""
    rows = int(frame["rows"])
    pixels = int(frame["pixels"])
    height = np.asarray(frame["height"], dtype=np.float64).reshape(rows, pixels)
    return {"height": height, "rows": rows, "pixels": pixels,
            "size_um": float(frame["size_um"]),
            "angle_deg": float(frame["angle_deg"]),
            "x_offset_um": float(frame["x_offset_um"]),
            "y_offset_um": float(frame["y_offset_um"]),
            "t": float(frame.get("t", time.time()))}


def compensation_offset(hub) -> tuple[float, float]:
    """The accumulated scan-offset compensation (lab µm) the NEXT scan adds, or
    (0, 0) when drift is off or no reference is set. The BACKEND owns compensation
    (see docs/CONNECT_PHASE2.md): scan_start adds this to the incoming offset."""
    d = getattr(hub, "drift", None) or {}
    if d.get("enabled") and d.get("reference"):
        tx, ty = d.get("total", [0.0, 0.0])
        return float(tx), float(ty)
    return 0.0, 0.0


def _geometry_matches(ref: dict, frame: dict) -> bool:
    return (int(ref["rows"]) == int(frame["rows"])
            and int(ref["pixels"]) == int(frame["pixels"])
            and abs(float(ref["size_um"]) - float(frame["size_um"])) < 1e-6
            and abs(float(ref["angle_deg"]) - float(frame["angle_deg"])) < 1e-6)


def register_frame(hub, runtime_dir, frame: dict) -> dict | None:
    """Estimate this completed frame's drift against the reference and, when the
    toggle is on, TRACK the ports and accumulate the COMPENSATE offset.

    Returns a JSON-safe broadcast dict (`{"type": "drift", ...}`) or None when
    there is nothing to do (no reference / geometry mismatch). Never raises for a
    caller reason -- callers still wrap it best-effort so a drift hiccup cannot
    break a scan. Mutates `hub.drift`, and (when enabled + accepted) persists and
    reloads `hub.ports`.
    """
    from . import electrodes

    d = hub.drift
    ref = d.get("reference")
    if not ref:
        return None
    if not _geometry_matches(ref, frame):
        return {"type": "drift", "rejected": True, "reason": "geometry mismatch",
                "enabled": bool(d.get("enabled")),
                "total": list(d.get("total", [0.0, 0.0]))}

    rows = int(frame["rows"])
    pixels = int(frame["pixels"])
    size_um = float(frame["size_um"])
    angle_deg = float(frame["angle_deg"])
    new_h = np.asarray(frame["height"], dtype=np.float64).reshape(rows, pixels)
    res = estimate_shift(ref["height"], new_h, size_um, pixels, rows)

    t = float(frame.get("t", time.time()))
    if not res["ok"]:
        return {"type": "drift", "rejected": True, "reason": "low quality",
                "quality": res["quality"], "t": t,
                "enabled": bool(d.get("enabled")),
                "total": list(d.get("total", [0.0, 0.0]))}

    # frame (fast, slow) axes -> lab (x, y). Identity at angle 0.
    from ..litho.frame import rotate
    lab_dx, lab_dy = rotate(res["dx_um"], res["dy_um"], angle_deg)

    d["track"].append({"t": t, "dx_um": lab_dx, "dy_um": lab_dy})
    d["total"] = [d["total"][0] + lab_dx, d["total"][1] + lab_dy]

    ports_moved = False
    rebased = False
    if d.get("enabled"):
        # TRACK: every marked electrode follows its feature by the same (dx, dy).
        ports, seq = electrodes.load_store(runtime_dir, hub.sample_name)
        for p in ports:
            p["position_um"] = [p["position_um"][0] + lab_dx,
                                p["position_um"][1] + lab_dy]
        if ports:
            electrodes.save_ports(runtime_dir, hub.sample_name, ports, seq)
            hub.ports, hub.ports_seq = ports, seq
            ports_moved = True

        # RE-BASE: once the accumulated drift is a quarter of the frame, adopt
        # this frame as the new reference and zero the total, so a long session
        # never correlates against an ever-more-different reference (mirrors the
        # vendor's maxoffset = scanSize * threshold). COMPENSATE keeps the offset
        # already applied -- only the delta resets.
        tx, ty = d["total"]
        if (tx * tx + ty * ty) ** 0.5 > 0.25 * size_um:
            d["reference"] = frame_to_reference(frame)
            d["total"] = [0.0, 0.0]
            rebased = True

    return {"type": "drift", "rejected": False, "dx_um": lab_dx, "dy_um": lab_dy,
            "total": list(d["total"]), "offset": list(d["total"]), "t": t,
            "quality": res["quality"], "enabled": bool(d.get("enabled")),
            "ports_moved": ports_moved, "rebased": rebased,
            "sample": hub.sample_name}
