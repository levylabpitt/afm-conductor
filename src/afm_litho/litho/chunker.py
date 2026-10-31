"""Split an over-long AFM out-wave into DSP-loadable chunks.

An out-wave armed on the DSP is capped at `MAX_OUTWAVE_POINTS`; a longer trajectory is split
into contiguous sub-waves run back-to-back, with seams on natural boundaries (whole raster
lines) so the reload pause lands somewhere harmless. Works on index ranges only (numpy +
stdlib); it never looks at what the points mean.
"""

from __future__ import annotations

import numpy as np

# The per-wave DSP cap, one source of truth for both backends and the wire. Two numbers exist;
# use the smaller, the vendor's own:
#   * 87,380 -- the v16 td reference (docs/reference/asylum-td-command-reference.txt:1808,
#     td_xSetOutWavePair);
#   * 87,000 -- `Constant cMaxOutWave` in the v19 vendor source (Temp1.ipf:13; cited in
#     docs/asylum_safeguards.md:269,277 and docs/history/2026-08-21-AUDIT.md:129-132 M5).
# The rig runs v19/v21, so an 87,001..87,380-point wave would pass 87,380 and be refused or
# silently truncated by the controller.
MAX_OUTWAVE_POINTS = 87_000

# One chain chunk, deliberately under the arm cap. Igor pads each chunk's capture up to a
# multiple of 32 (`AFM_LithoCapPoints`, mirror of `capture_points`): 87,000 = 2718*32 + 24
# pads to an 87,008-point in-wave, over the cap. Whether cMaxOutWave binds in-waves is
# unknown (the td reference gives a maximum only for td_xSetOutWavePair; td_xSetInWavePair
# documents only the multiple-of-32 rule). 86,976 = 2718*32 keeps capLen == m, so both
# waves are inside the cap, at a cost of 24 points per chunk.
CHAIN_CHUNK_POINTS = 86_976
assert CHAIN_CHUNK_POINTS % 32 == 0 and CHAIN_CHUNK_POINTS <= MAX_OUTWAVE_POINTS


def is_chained(n: int, cap: int = CHAIN_CHUNK_POINTS) -> bool:
    """A stroke of `n` points is fired as a chunk chain (LITHOCHAIN) only above the chunk
    size `cap`, not the arm cap: its in-wave has the out-wave's length, and 86,976 is the
    largest multiple of 32 inside the cap (docs/LITHO_CHUNK_CHAIN_PLAN.md)."""
    return n > cap


def check_outwave_len(channel: str, wave) -> np.ndarray:
    """Refuse to ARM a single out-wave longer than the DSP cap.

    Shared by both backends (hardware_arc.py, twin/arc.py) so a twin test cannot pass where
    the controller would truncate an over-cap wave and report success. The cap binds at the
    bank arm (td_xSetOutWave/Pair), not per transferred wave: HardwareARC._load_wave builds
    longer waves that SCANCHAIN/LITHOCHAIN slice Igor-side a chunk at a time.
    """
    vals = np.asarray(wave, dtype=np.float64)
    if len(vals) > MAX_OUTWAVE_POINTS:
        raise ValueError(
            f"out-wave for {channel} is {len(vals):,} points, over the "
            f"{MAX_OUTWAVE_POINTS:,}-point DSP cap. Chunk it "
            f"(litho.chunker.chunk_indices), or fire it as "
            f"a chunk chain (hardware.lithography_run); the controller would "
            f"silently truncate it.")
    return vals


def _penup_mask(*arrays) -> np.ndarray:
    """Per-point pen-up mask. Vendor invariant: the bias/XY waves carry NaN at a pen-up
    (LithoVolts is NaN where XLitho/YLitho are), so a NaN in ANY array marks a pen-up."""
    a0 = np.asarray(arrays[0], dtype=np.float64)
    mask = np.zeros(len(a0), dtype=bool)
    for a in arrays:
        mask |= np.isnan(np.asarray(a, dtype=np.float64))
    return mask


def chunk_boundaries(n: int, penup: np.ndarray,
                     cap: int = CHAIN_CHUNK_POINTS) -> list[int]:
    """Split an n-point object into consecutive chunk boundaries [0, c1, ..., n], each
    chunk `[b[k]:b[k+1]]` at most `cap` points (docs/LITHO_CHUNK_CHAIN_PLAN.md §4). `cap`
    defaults to CHAIN_CHUNK_POINTS, not MAX_OUTWAVE_POINTS (32-alignment, see there).

    PEN-UP PREFERRING: each cut falls at the LAST pen-up (NaN) index within
    `(start, start+cap]`, where the tip is lifted and the bias off; only a continuous
    stroke longer than `cap` is cut mid-stroke at `start+cap`. The same indices slice x, y
    and v, so position and voltage stay registered across a seam.

    On the hardware write path the preference never fires: the compiler splits at pen-ups
    first, so `penup` is all-False and every cut is mid-stroke, the tip stationary at full
    bias while Igor re-arms.
    """
    penup = np.asarray(penup, dtype=bool)
    bounds = [0]
    start = 0
    while n - start > cap:
        lo, hi = start + 1, start + cap                # candidate cut range (inclusive)
        window = penup[lo:hi + 1]
        idxs = np.nonzero(window)[0]
        if len(idxs):
            cut = lo + int(idxs[-1])                    # last pen-up in the window
        else:
            cut = start + cap                          # continuous stroke: mid-stroke cut
        bounds.append(cut)
        start = cut
    bounds.append(n)
    return bounds


def _block_split(start: int, stop: int, max_points: int, block: int) -> list[tuple[int, int]]:
    """Tile [start, stop) with slices of the largest multiple of ``block`` that fits in
    ``max_points``; the last slice is the remainder."""
    span = stop - start
    if span <= max_points:
        return [(start, stop)]

    step = (max_points // block) * block
    if step <= 0:
        # One block exceeds the cap: fall back to unaligned max_points slices.
        step = max_points

    out: list[tuple[int, int]] = []
    pos = start
    while pos < stop:
        end = min(pos + step, stop)
        out.append((pos, end))
        pos = end
    return out


def chunk_indices(
    n_points: int,
    *,
    max_points: int = MAX_OUTWAVE_POINTS,
    block: int = 32,
    line_len: int | None = None,
) -> list[tuple[int, int]]:
    """Tile ``[0, n_points)`` exactly with contiguous ``[start, stop)`` slices, each
    ``<= max_points``.

    With ``line_len`` (a raster) every chunk is whole lines, so seams fall on line
    boundaries; a single line wider than the cap is sub-split block-aligned. Without it,
    chunks are the largest multiple of ``block`` that fits. ``n_points == 0`` -> ``[]``;
    ``line_len <= 0`` is treated as None.
    """
    if n_points < 0:
        raise ValueError(f"n_points must be non-negative, got {n_points}")
    if max_points < 1:
        raise ValueError(f"max_points must be >= 1, got {max_points}")
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")

    if n_points == 0:
        return []
    if n_points <= max_points:
        return [(0, n_points)]

    if line_len is not None and line_len <= 0:
        line_len = None

    if line_len is None:
        return _block_split(0, n_points, max_points, block)

    lines_per_chunk = max(1, max_points // line_len)
    chunk_len = lines_per_chunk * line_len

    out: list[tuple[int, int]] = []
    pos = 0
    while pos < n_points:
        end = min(pos + chunk_len, n_points)
        if lines_per_chunk == 1 and line_len > max_points:
            # One line is wider than the cap: sub-split it block-aligned.
            out.extend(_block_split(pos, end, max_points, block))
        else:
            out.append((pos, end))
        pos = end
    return out


def chunk_arrays(arrays: list[np.ndarray], **kw) -> list[list[np.ndarray]]:
    """Slice equal-length arrays (e.g. ``[x, y, v]``) per :func:`chunk_indices` chunk
    (``**kw`` forwarded). Raises ``ValueError`` on unequal lengths."""
    if not arrays:
        return []

    n = len(arrays[0])
    for i, a in enumerate(arrays):
        if len(a) != n:
            raise ValueError(
                f"all arrays must be the same length; arrays[0] has {n} "
                f"but arrays[{i}] has {len(a)}"
            )

    slices = chunk_indices(n, **kw)
    return [[a[start:stop] for a in arrays] for start, stop in slices]
