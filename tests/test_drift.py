"""Drift estimator + Track-and-compensate integration (Connect Phase 2).

The SIGN CONVENTION lives or dies here. Two independent pins:

  * Synthetic: a known analytic field shifted by a known (integer + subpixel)
    amount -- the estimator must recover magnitude AND direction, robust to
    Gaussian noise and a gentle plane, and REJECT a low-contrast frame.
  * Twin ground-truth: a KNOWN `World.drift_v` walks the sample between two
    consecutive sim scans of the same region; the estimator must recover that
    drift with the right sign, and `register_frame` must then shift the ports
    and the next scan's compensation offset the SAME way.

See src/afm_litho/service/drift.py for the convention the assertions pin.
"""

from __future__ import annotations

import types

import numpy as np

from afm_litho.service import electrodes
from afm_litho.service.drift import (
    PEAK_RATIO_MIN,
    compensation_offset,
    estimate_shift,
    frame_to_reference,
    new_state,
    register_frame,
)

# --------------------------------------------------------------- synthetic --

def _frames(dcol, drow, rows=64, pixels=64, noise=0.0, plane=0.0, seed=0):
    """Reference grid and a copy whose FEATURES are moved by (+dcol, +drow) px.

    The reference is a BROADBAND random field (real AFM topography is broadband;
    phase correlation's spectral whitening needs it). The shift is applied
    EXACTLY -- including subpixel -- by the Fourier shift theorem, so
    `new[r,c] = base[r - drow, c - dcol]`: a feature at reference column x lands
    at column x + dcol in new, exactly the +dx / +dy the estimator reports."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((rows, pixels))
    F = np.fft.fft2(base)
    ky = np.fft.fftfreq(rows)[:, None]
    kx = np.fft.fftfreq(pixels)[None, :]
    ramp = np.exp(-2j * np.pi * (ky * drow + kx * dcol))
    new = np.fft.ifft2(F * ramp).real
    ref = base
    yy, xx = np.mgrid[0:rows, 0:pixels].astype(float)
    if plane:
        new = new + plane * (0.7 * xx + yy)          # gentle skew/trend
    if noise:
        ref = ref + rng.normal(0, noise, ref.shape)
        new = new + rng.normal(0, noise, new.shape)
    return ref, new


def test_recovers_exact_subpixel_shift_and_direction():
    """No noise: the estimator recovers a subpixel shift to ~0.05 px, and the
    SIGN is right on both axes (dx>0 for +col, dy<0 for -row)."""
    size_um, rows, pixels = 10.0, 64, 64
    dcol, drow = 5.3, -3.7
    ref, new = _frames(dcol, drow, rows, pixels)
    r = estimate_shift(ref, new, size_um, pixels, rows)
    assert r["ok"] and r["quality"] >= PEAK_RATIO_MIN
    px = size_um / pixels
    assert r["dx_um"] > 0 and r["dy_um"] < 0          # direction
    # subpixel: the per-line flatten + Hann window bias recovery by ~0.12 px
    assert abs(r["dx_um"] - dcol * px) < 0.2 * px     # magnitude
    assert abs(r["dy_um"] - drow * (size_um / rows)) < 0.2 * (size_um / rows)


def test_recovers_shift_through_noise_and_skew():
    """Add Gaussian noise and a gentle plane: still recovered within ~0.25 px."""
    size_um, rows, pixels = 12.0, 64, 64
    dcol, drow = -4.6, 2.9
    ref, new = _frames(dcol, drow, rows, pixels, noise=0.35, plane=0.02, seed=3)
    r = estimate_shift(ref, new, size_um, pixels, rows)
    assert r["ok"]
    px, py = size_um / pixels, size_um / rows
    assert abs(r["dx_um"] - dcol * px) < 0.25 * px
    assert abs(r["dy_um"] - drow * py) < 0.25 * py
    assert r["dx_um"] < 0 and r["dy_um"] > 0


def test_low_contrast_frame_is_rejected():
    """A near-blank frame gives a flat correlation surface -> low quality ->
    ok=False, so the caller never shoves ports around on noise."""
    rows = pixels = 64
    rng = np.random.default_rng(7)
    ref = np.ones((rows, pixels)) + rng.normal(0, 1e-4, (rows, pixels))
    new = np.ones((rows, pixels)) + rng.normal(0, 1e-4, (rows, pixels))
    r = estimate_shift(ref, new, 10.0, pixels, rows)
    assert not r["ok"] and r["quality"] < PEAK_RATIO_MIN


def test_all_nan_frame_is_safe():
    """An all-NaN (never-scanned) frame returns ok=False, not a crash."""
    a = np.full((32, 32), np.nan)
    b = np.zeros((32, 32))
    assert estimate_shift(a, b, 10.0, 32, 32)["ok"] is False


# ----------------------------------------------------------- twin ground-truth --

def _twin(drift_v):
    """A fresh twin over the default sample with a known XY drift rate."""
    from afm_litho.io import read_ibw
    from afm_litho.service.deps import DEFAULT_SAMPLE, SAMPLES
    from afm_litho.service.passes import TwinInstrument
    from afm_litho.twin.arc import SoftwareARC
    from afm_litho.twin.physics import Instrument, Sample, World

    path = SAMPLES / DEFAULT_SAMPLE
    sample = Sample.from_ibw(path)
    inst = Instrument.from_note(read_ibw(path).note_dict)
    world = World(sample=sample, instrument=inst)
    world.t = 0.0
    world.drift_v = drift_v
    return TwinInstrument(SoftwareARC(world)), world


def _run_frame(twin, cfg):
    """Drive one sim frame to completion; return its Height grid (rows,pixels)."""
    twin.start_scan(cfg)
    while twin.scan and twin.scan.running:
        if twin.scan_line() is None:
            break
    return np.asarray(twin.scan.channels["Height"], dtype=float)


def test_up_and_down_frames_are_not_vertically_flipped():
    """Continuous imaging alternates the PHYSICAL scan direction each frame, but
    the cached frame must keep row i = the same physical y regardless of the scan
    direction (the twin indexes channels by physical slow_idx, not scan order).
    If an up-frame and a down-frame of the SAME static scene came out vertically
    mirrored, the drift estimator would read the flip as an enormous bogus dy and
    drift tracking would break during continuous imaging. Guard:
    estimate_shift(up, down) ~ 0 -- NOT a ~full-frame dy."""
    from afm_litho.service.passes import ScanConfig

    size_um, n = 20.0, 48
    twin, _world = _twin((0.0, 0.0))              # NO drift: an identical scene
    up = _run_frame(twin, ScanConfig(size_um=size_um, pixels=n, lines=n,
                                     line_rate_hz=float(n), mode="contact",
                                     direction="up"))
    down = _run_frame(twin, ScanConfig(size_um=size_um, pixels=n, lines=n,
                                       line_rate_hz=float(n), mode="contact",
                                       direction="down"))
    r = estimate_shift(up, down, size_um, n, n)
    assert r["ok"], f"up/down frames did not correlate (quality {r['quality']:.1f})"
    px = size_um / n
    # a vertical mirror would read as ~ n/2 rows of dy; both axes must be ~0.
    assert abs(r["dy_um"]) < 1.5 * px, r
    assert abs(r["dx_um"]) < 1.5 * px, r


def test_register_frame_tracks_the_ports_and_accumulates_the_compensation(tmp_path):
    """The drift registration on a synthetic pair (the twin no longer drifts): a mark follows
    its electrode by the estimator's (dx, dy), and the next scan's compensation moves the
    same way."""
    size_um, n = 20.0, 64
    px = size_um / n
    h1, h2 = _frames(dcol=3.0, drow=-2.0, rows=n, pixels=n)       # features moved +x, -y
    sample_name = "synthetic"
    p0 = [1.0, 2.0]
    electrodes.save_ports(tmp_path, sample_name, [{
        "id": "e0", "label": "1", "position_um": list(p0),
        "direction_deg": 0.0, "width_um": 1.0}], 1)

    ref = frame_to_reference({"height": h1, "rows": n, "pixels": n,
                              "size_um": size_um, "angle_deg": 0.0,
                              "x_offset_um": 0.0, "y_offset_um": 0.0})
    hub = types.SimpleNamespace(sample_name=sample_name, ports=[], ports_seq=0,
                                drift={**new_state(), "enabled": True,
                                       "reference": ref})
    frame2 = {"height": h2, "rows": n, "pixels": n, "size_um": size_um,
              "angle_deg": 0.0, "x_offset_um": 0.0, "y_offset_um": 0.0}
    msg = register_frame(hub, tmp_path, frame2)
    assert msg and not msg.get("rejected") and msg["ports_moved"]

    moved = electrodes.load_ports(tmp_path, sample_name)[0]["position_um"]
    assert abs(moved[0] - (p0[0] + msg["dx_um"])) < 1e-6
    assert abs(moved[1] - (p0[1] + msg["dy_um"])) < 1e-6
    assert moved[0] > p0[0] and moved[1] < p0[1]        # +x, -y, right way

    cox, coy = compensation_offset(hub)
    assert cox > 0 and coy < 0
    assert abs(cox - 3.0 * px) < 0.5 * px and abs(coy + 2.0 * px) < 0.5 * px
