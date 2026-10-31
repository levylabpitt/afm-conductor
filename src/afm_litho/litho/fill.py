"""Filling a closed outline with a tip path.

Inkscape drew the shape; the old lithography program decided how to sweep
the tip across it. This is that second half, and it carries the lab's
conventions rather than a generic rasteriser's:

- Fill lines run along the shape's LONG axis. A virtual electrode is a
  narrow funnel, and sweeping it the short way means thousands of tiny
  strokes and thousands of turnarounds. Along the long direction it is a
  handful of passes.
- Consecutive lines alternate direction and are joined at their ends, so
  the tip walks a continuous serpentine instead of flying back. There is no
  square cap: the joins ARE the zigzag end.
- Holes work: the even-odd rule means a shape drawn inside another shape is
  a hole, exactly as it renders in Inkscape.
- Concave shapes work: a scan line crossing a C-shape produces two spans,
  and the tip must not be dragged across the gap between them at write
  bias, so spans are ordered and connected without ever crossing outside
  the shape.
"""

from __future__ import annotations

import math

import numpy as np

from .bezier import signed_area


def long_axis_angle(poly: np.ndarray) -> float:
    """Orientation of the shape's long axis, radians.

    Principal component of the outline vertices. For a tapered lead this is
    the direction the lead runs, which is the direction the lab writes.
    """
    p = np.asarray(poly, dtype=np.float64)
    if len(p) < 3:
        return 0.0
    q = p - p.mean(axis=0)
    # weight by edge length so densely-sampled curves do not dominate
    seg = np.hypot(*np.diff(np.vstack([p, p[:1]]), axis=0).T)
    w = (seg + np.roll(seg, 1)) / 2
    cov = (q * w[:, None]).T @ q / max(w.sum(), 1e-12)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, int(np.argmax(vals))]
    return float(np.arctan2(v[1], v[0]))


def _spans(polys, y: float, x0: float, x1: float) -> list[tuple[float, float]]:
    """Even-odd crossings of the horizontal line at `y`, as inside spans."""
    xs: list[float] = []
    for poly in polys:
        n = len(poly)
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % n]
            if ay == by:
                continue
            # half-open rule: counts each vertex once, so a scan line
            # through a vertex does not produce a doubled crossing
            if (ay <= y < by) or (by <= y < ay):
                t = (y - ay) / (by - ay)
                xs.append(ax + t * (bx - ax))
    if len(xs) < 2:
        return []
    xs.sort()
    out = []
    for i in range(0, len(xs) - 1, 2):
        a, b = xs[i], xs[i + 1]
        if b - a > 1e-9:
            out.append((max(a, x0), min(b, x1)))
    return [(a, b) for a, b in out if b - a > 1e-9]


def _scanline_setup(polys, angle_rad, inset_um, spacing_um):
    """Shared setup for the parallel-scanline fills: rotate so the fill axis
    is horizontal, inset the bounds, and place the scan lines. Returns
    (rot, Rb, x0, x1, ys) -- the rotated polys, the inverse rotation to map
    strokes back, the inset x-extent for span clipping, and the scan-line
    heights -- or None if there is nothing to fill."""
    polys = [np.asarray(p, dtype=np.float64) for p in polys if len(p) >= 3]
    if not polys:
        return None
    if angle_rad is None:
        angle_rad = long_axis_angle(np.vstack(polys))

    # rotate into a frame where the long axis is horizontal, fill, rotate back
    c, s = np.cos(-angle_rad), np.sin(-angle_rad)
    R = np.array([[c, -s], [s, c]])
    rot = [p @ R.T for p in polys]
    allp = np.vstack(rot)
    x0, x1 = allp[:, 0].min(), allp[:, 0].max()
    y0, y1 = allp[:, 1].min(), allp[:, 1].max()

    # inset keeps the written material off the designed edge (the tip has a
    # finite radius, so a fill that runs to the outline writes wider than it)
    x0, x1 = x0 + inset_um, x1 - inset_um
    y0, y1 = y0 + inset_um, y1 - inset_um
    if x1 <= x0 or y1 <= y0:
        return None

    height = y1 - y0
    n_lines = max(1, int(round(height / max(spacing_um, 1e-9))) + 1)
    ys = (np.linspace(y0, y1, n_lines) if n_lines > 1
          else np.array([(y0 + y1) / 2]))

    Rb = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                   [np.sin(angle_rad), np.cos(angle_rad)]])
    return rot, Rb, x0, x1, ys


def fill_polygons(polys, spacing_um: float, angle_rad: float | None = None,
                  inset_um: float = 0.0, outline: bool = False) -> list[np.ndarray]:
    """Serpentine tip path(s) filling the given closed polygons.

    `polys` is a list of vertex arrays in microns; nesting makes holes by
    the even-odd rule. Returns one array per continuous stroke: usually one,
    but a shape that breaks into disconnected regions (a C swept the wrong
    way, or a genuine second island) yields one per region, because joining
    them would drag the biased tip across unwritten sample.
    """
    setup = _scanline_setup(polys, angle_rad, inset_um, spacing_um)
    if setup is None:
        return []
    rot, Rb, x0, x1, ys = setup

    rows = [(y, _spans(rot, y, x0, x1)) for y in ys]
    rows = [(y, sp) for y, sp in rows if sp]
    if not rows:
        return []

    # Walk the rows, extending a stroke while the next row has a span that
    # overlaps the one we are on. Overlap is the condition for the join to
    # stay inside the shape.
    strokes: list[list[tuple[float, float]]] = []
    used = [[False] * len(sp) for _, sp in rows]

    def overlaps(a, b):
        return min(a[1], b[1]) - max(a[0], b[0]) > -1e-9

    for ri in range(len(rows)):
        for si in range(len(rows[ri][1])):
            if used[ri][si]:
                continue
            stroke: list[tuple[float, float]] = []
            r, k = ri, si
            flip = False
            while True:
                used[r][k] = True
                y = rows[r][0]
                a, b = rows[r][1][k]
                stroke += [(b, y), (a, y)] if flip else [(a, y), (b, y)]
                flip = not flip
                if r + 1 >= len(rows):
                    break
                cand = [(j, sp) for j, sp in enumerate(rows[r + 1][1])
                        if not used[r + 1][j] and overlaps(sp, (a, b))]
                if not cand:
                    break
                # prefer the span whose centre is nearest where the tip is
                here = a if flip else b
                j, _ = min(cand, key=lambda t: abs((t[1][0] + t[1][1]) / 2 - here))
                r, k = r + 1, j
            if stroke:
                strokes.append(stroke)

    out = [np.asarray(st, dtype=np.float64) @ Rb.T for st in strokes]
    if outline:
        out = [p @ Rb.T for p in rot] + out
    return out


def fill_funnel_from_curves(curve_a, curve_b, spacing_um: float,
                            tol_um: float = 0.01) -> list[np.ndarray]:
    """The 2013 funnel family, stitched into one continuous serpentine.

    Both cubics run base-corner -> apex. Blend their CONTROL POINTS,
    cx = x*A + (1-x)*B, and flatten each -- because a Bezier is linear in its
    control points this is identical to blending the two evaluated curves, and
    every sweep runs from a point on the base to the shared apex. The sweep
    count comes from the base width, where the family is widest.

    Then connect the sweeps into ONE stroke, so the tip never lifts inside the
    funnel: start at one base corner, up an edge to the apex, down the next
    curve, a little along the base to the next start, up again, and so on --
    the apexes coincide (a shared point) and the base steps are along the base.
    Both boundary curves are the first and last sweeps, so the edges are
    written. Returned as a single-element list, the funnel's one continuous
    write.

    This uses the two ACTUAL edges -- what an imported funnel outline gives
    us -- so the sides written are the outline's own curves.
    """
    from .bezier import flatten_cubic
    a = [np.asarray(p, dtype=np.float64) for p in curve_a]
    b = [np.asarray(p, dtype=np.float64) for p in curve_b]
    base_w = float(np.hypot(*(a[0] - b[0])))
    n = max(2, int(math.ceil(base_w / max(spacing_um, 1e-6))) + 1)
    sweeps = []
    for x in np.linspace(0.0, 1.0, n):
        c = [x * ai + (1.0 - x) * bi for ai, bi in zip(a, b)]
        # flatten_cubic drops the curve's start point, so put it back: every
        # sweep must BEGIN on the base (c[0]), or the base comes out ragged.
        pts = flatten_cubic(c[0], c[1], c[2], c[3], tol_um)
        sweeps.append(np.asarray([c[0], *pts], dtype=np.float64))
    path = [sweeps[0]]                            # base_0 -> apex (an edge)
    for i in range(1, n):
        seg = sweeps[i] if i % 2 == 0 else sweeps[i][::-1]
        if np.allclose(path[-1][-1], seg[0]):     # apexes coincide: no repeat
            seg = seg[1:]
        path.append(seg)                          # base steps stay as segments
    return [np.concatenate(path)]


def fill_zigzag(polys, spacing_um: float, angle_rad: float | None = None,
                inset_um: float = 0.0) -> list[np.ndarray]:
    """Unidirectional raster: every row written the SAME direction.

    Unlike fill_polygons (a serpentine that writes rows in alternating
    directions), this returns one stroke per row, all running the same way, so
    the retrace between rows is a non-writing move (a bias-off transit) -- the
    AFM's own trace/retrace scan. In lithography the write is direction-
    dependent, so alternating rows would come out uneven; a zig zag keeps every
    line consistent.
    """
    setup = _scanline_setup(polys, angle_rad, inset_um, spacing_um)
    if setup is None:
        return []
    rot, Rb, x0, x1, ys = setup
    out = []
    for y in ys:
        for a, b in _spans(rot, y, x0, x1):
            out.append(np.asarray([(a, y), (b, y)], dtype=np.float64) @ Rb.T)  # same way
    return out


def fill_shape(paths, spacing_um: float, angle_rad: float | None = None,
               tol_um: float = 0.005, inset_um: float = 0.0,
               outline: bool = False) -> list[np.ndarray]:
    """Same, starting from BezierPaths: flatten, then fill."""
    polys = []
    for p in paths:
        q = p.flatten(tol_um)
        if len(q) >= 3:
            polys.append(q)
    return fill_polygons(polys, spacing_um, angle_rad, inset_um, outline)


# --------------------------------------------------------- contour filling --

def _signed_area(poly: np.ndarray) -> float:
    return signed_area(poly)


def _offset_rings(poly: np.ndarray, first: float, step: float,
                  max_rings: int = 4000):
    """Every inward offset of `poly`, from `first` inward by `step`.

    A raster erosion, not a Clipper-style analytic offset: an inward offset of
    a concave shape SPLITS (a dumbbell eroded past its waist becomes two loops),
    and eroding a mask then re-tracing its boundaries gets that topology change
    for free, because connected components are just what labelling finds. The
    distance field is built once and thresholded at each depth rather than
    rebuilt per ring.
    """
    if len(poly) < 3 or step <= 0:
        return
    lo = poly.min(axis=0) - 2 * step
    hi = poly.max(axis=0) + 2 * step
    px = max(step / 6.0, float(max(hi - lo)) / 1200.0, 1e-4)
    nx = int(np.ceil((hi[0] - lo[0]) / px)) + 1
    ny = int(np.ceil((hi[1] - lo[1]) / px)) + 1
    if nx < 3 or ny < 3 or nx * ny > 4_000_000:
        return

    gx = lo[0] + np.arange(nx) * px
    gy = lo[1] + np.arange(ny) * px
    X, Y = np.meshgrid(gx, gy, indexing="xy")
    inside = _points_in_poly(poly, X.ravel(), Y.ravel()).reshape(ny, nx)
    try:
        from scipy.ndimage import distance_transform_edt, label
    except ImportError:                            # scipy is a dependency
        return
    dist = distance_transform_edt(inside, sampling=(px, px))
    reach = float(dist.max())

    depth = float(first)
    for _ in range(max_rings):
        if depth > reach:
            return
        lab, n = label(dist >= depth)
        got = []
        for k in range(1, n + 1):
            comp = lab == k
            if comp.sum() < 8:
                continue
            got += [r for r in _trace_boundary(comp, gx, gy) if len(r) >= 4]
        if got:
            yield got
        depth += step


def _points_in_poly(poly: np.ndarray, xs, ys) -> np.ndarray:
    """Even-odd test, vectorised over all query points at once."""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    inside = np.zeros(xs.shape, dtype=bool)
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    for a, b, c, d in zip(x1, y1, x2, y2):
        cond = ((b > ys) != (d > ys))
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (c - a) * (ys - b) / np.where(d != b, d - b, np.nan) + a
        inside ^= cond & (xs < xint)
    return inside


def _trace_boundary(mask: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> list[np.ndarray]:
    """Marching-squares boundary of a binary mask, in world coordinates."""
    try:
        from skimage import measure
        contours = measure.find_contours(mask.astype(float), 0.5)
    except ImportError:
        contours = _find_contours_fallback(mask)
    out = []
    for c in contours:
        # find_contours yields (row, col) = (y index, x index)
        ys = gy[0] + c[:, 0] * (gy[1] - gy[0])
        xs = gx[0] + c[:, 1] * (gx[1] - gx[0])
        out.append(np.column_stack([xs, ys]))
    return out


def _find_contours_fallback(mask: np.ndarray) -> list[np.ndarray]:
    """Boundary cells in walk order, when scikit-image is not installed.

    Moore-neighbourhood tracing: from a boundary cell, keep turning around
    the component and record what you touch. Linear in the length of the
    boundary.

    It used to order the boundary cells by repeatedly scanning the remaining
    set for the nearest one, which is quadratic, and scikit-image is NOT
    installed here so this is the path that actually runs. A contour fill at
    the 10 nm pitch that is now the default took 72 seconds, essentially all
    of it in that scan. The same rings now take under a second.

    Only the OUTER boundary of each component, as before: a ring with a hole
    in it loses the hole's edge. Marching squares would give both.
    """
    m = np.asarray(mask, bool)
    if m.sum() < 4:
        return []
    from scipy.ndimage import find_objects, label

    lab, n = label(m)
    nb = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    ny, nx = m.shape
    out: list[np.ndarray] = []

    for k, sl in enumerate(find_objects(lab), start=1):
        if sl is None:
            continue
        sub = lab[sl] == k
        if sub.sum() < 4:
            continue
        r0, c0 = sl[0].start, sl[1].start
        first = np.argwhere(sub)[0]
        start = (int(first[0]) + r0, int(first[1]) + c0)

        def here(p, _k=k):
            return (0 <= p[0] < ny and 0 <= p[1] < nx and lab[p[0], p[1]] == _k)

        ring = [start]
        cur, back = start, (start[0], start[1] - 1)
        cap = 4 * int(sub.sum()) + 16
        while len(ring) < cap:
            d0 = nb.index((back[0] - cur[0], back[1] - cur[1]))
            step = None
            for t in range(1, 9):
                d = (d0 + t) % 8
                p = (cur[0] + nb[d][0], cur[1] + nb[d][1])
                if here(p):
                    prev = nb[(d - 1) % 8]
                    step = (p, (cur[0] + prev[0], cur[1] + prev[1]))
                    break
            if step is None:                      # a single isolated cell
                break
            cur, back = step
            if cur == start:
                break
            ring.append(cur)
        if len(ring) >= 4:
            out.append(np.array(ring, dtype=float))
    return out


# ---------------------------------------------------- space-filling curves --

def hilbert_cells(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Cell coordinates of a Hilbert curve on a 2**order square, in visit
    order. Built by expansion rather than by converting an index per cell:
    four transformed copies of the previous order, which is the definition.
    """
    x = np.array([0, 0, 1, 1], dtype=np.int64)
    y = np.array([0, 1, 1, 0], dtype=np.int64)
    for i in range(2, int(order) + 1):
        n = 1 << (i - 1)
        xo, yo = x, y
        x = np.concatenate([yo, xo, xo + n, 2 * n - 1 - yo])
        y = np.concatenate([xo, yo + n, yo + n, n - 1 - xo])
    return x, y


def _region_mask(polys, cell: float, inset_um: float, max_cells: int):
    """Fine-cell occupancy of the region, at EXACTLY `cell` microns.

    Returns (mask, ox, oy) with mask[i, j] True where the centre of fine cell
    (i, j) is inside. The grid dimensions are even, because the covering path
    below pairs cells into 2x2 blocks.
    """
    from scipy import ndimage

    allp = np.vstack(polys)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    nx = int(math.ceil((hi[0] - lo[0]) / cell)) + 3
    ny = int(math.ceil((hi[1] - lo[1]) / cell)) + 3
    nx += nx % 2
    ny += ny % 2
    if nx * ny > max_cells:
        return None, 0.0, 0.0
    ox, oy = lo[0] - cell, lo[1] - cell

    mask = np.zeros((nx, ny), dtype=bool)
    ax = ox + (np.arange(nx) + 0.5) * cell
    for j in range(ny):
        y = oy + (j + 0.5) * cell
        for a, b in _spans(polys, y, -np.inf, np.inf):
            mask[(ax >= a) & (ax <= b), j] = True
    if inset_um > 0:
        r = int(round(inset_um / cell))
        if r > 0:
            g = np.mgrid[-r:r + 1, -r:r + 1]
            mask = ndimage.binary_erosion(mask, np.hypot(g[0], g[1]) <= r)
    return mask, ox, oy


def _spanning_forest(free: np.ndarray, rank: np.ndarray):
    """Spanning forest of the free 2x2 blocks, grown in Hilbert order.

    `east[I, J]` means a tree edge from block (I, J) to (I+1, J); `north` the
    same upward. Depth-first, taking the neighbour of lowest Hilbert rank
    first, so the tree -- and therefore the path that circumnavigates it --
    keeps the locality the Hilbert curve is chosen for.

    **A branch that continues straight is taken last.** Without that the tree
    grows long spines and the path runs beside them: measured on a 10 um
    square, straight runs of 11 to 47 cells, which is a raster wearing a
    different hat and would undo the one property this fill exists for. With
    it the longest run is SEVEN cells whatever the pitch or the size of the
    region -- 70 nm at the 10 nm pitch -- because the bound comes from the
    local branching rule and not from the geometry.
    """
    cx, cy = free.shape
    east = np.zeros((cx, cy), dtype=bool)
    north = np.zeros((cx, cy), dtype=bool)
    seen = np.zeros((cx, cy), dtype=bool)
    roots: list[tuple[int, int]] = []

    order = int(math.ceil(math.log2(max(cx, cy, 2))))
    hx, hy = hilbert_cells(order)
    keep = (hx < cx) & (hy < cy)
    seq = list(zip(hx[keep].tolist(), hy[keep].tolist()))

    for s in seq:
        if not free[s] or seen[s]:
            continue
        roots.append(s)
        seen[s] = True
        stack = [(s, (0, 0))]
        while stack:
            (i, j), came = stack.pop()
            cand = []
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < cx and 0 <= b < cy and free[a, b] and not seen[a, b]:
                    straight = 1 if (di, dj) == came else 0
                    cand.append((straight, int(rank[a, b]), a, b, di, dj))
            # highest key pushed first, so the lowest -- a turn, then the
            # lowest Hilbert rank -- is popped first
            cand.sort(reverse=True)
            for _, _, a, b, di, dj in cand:
                if seen[a, b]:
                    continue
                seen[a, b] = True
                if di == 1:
                    east[i, j] = True
                elif di == -1:
                    east[a, b] = True
                elif dj == 1:
                    north[i, j] = True
                else:
                    north[a, b] = True
                stack.append(((a, b), (di, dj)))
    return east, north, roots


def _cover_path(east, north, root) -> list[tuple[int, int]]:
    """Fine cells of one spanning tree, in the order the tip visits them.

    The path circumnavigates the tree: it never crosses a tree edge, and it
    leaves a block only where one exists. Every fine cell of the tree then
    has exactly two permitted neighbours, so the walk is forced -- which is
    the whole trick, and is why this covers each cell once with no choices to
    get wrong.
    """
    cx, cy = east.shape

    def nexts(i, j):
        big_i, big_j = i >> 1, j >> 1
        u, v = i & 1, j & 1
        out = []
        # inside the block: blocked where a tree edge runs between the pair
        vert = east[big_i, big_j] if u else (east[big_i - 1, big_j]
                                             if big_i > 0 else False)
        if not vert:
            out.append((i, j ^ 1))
        horiz = north[big_i, big_j] if v else (north[big_i, big_j - 1]
                                               if big_j > 0 else False)
        if not horiz:
            out.append((i ^ 1, j))
        # out of the block: only along a tree edge, on either side of it
        if u and east[big_i, big_j]:
            out.append((i + 1, j))
        if not u and big_i > 0 and east[big_i - 1, big_j]:
            out.append((i - 1, j))
        if v and north[big_i, big_j]:
            out.append((i, j + 1))
        if not v and big_j > 0 and north[big_i, big_j - 1]:
            out.append((i, j - 1))
        return out

    start = (root[0] * 2, root[1] * 2)
    path = [start]
    prev, here = None, start
    while True:
        step = [q for q in nexts(*here) if q != prev]
        if not step:
            break
        nxt = step[0]
        if nxt == start:
            break
        path.append(nxt)
        prev, here = here, nxt
    return path


def fill_hilbert(polys, spacing_um: float, inset_um: float = 0.0,
                 outline: bool = False,
                 max_cells: int = 4_000_000) -> list[np.ndarray]:
    """Fill a region with a space-filling curve, at exactly `spacing_um`.

    A raster and a set of contour rings both cover a region; neither is a
    space-filling curve, and for writing with a biased tip the difference is
    physical, not aesthetic.

    - **One pass, no pen-ups.** A serpentine breaks wherever a span ends and
      contour rings are separate closed loops, so both put joins inside the
      region. This visits every cell of a connected region in one continuous
      path, so there is no dose discontinuity where strokes meet.
    - **No preferred direction.** A raster writes long parallel runs, and
      whatever is anisotropic about the tip, the scanner or the sample gets
      imprinted along that one axis. Runs here are a few cells at most,
      against a serpentine's full width every line. On KTO the angle between
      a feature and the crystal axes is a physical variable (docs/KTOME.md),
      and a fill that carries its own axis confounds it.
    - **No turnaround dwell.** A serpentine reverses at both ends of every
      line, and the velocity ceiling means the tip lingers there, so dose
      piles up along two edges. The turns here are spread evenly.
    - **Adjacent in time is adjacent in space.** Drift and decay during a
      long fill show up as a local gradient rather than smeared across the
      region, which is what makes them correctable.

    Path length is area / spacing, the same as a raster at the same pitch.

    **`spacing_um` is the line separation, exactly.** It was not, in the
    first version: a Hilbert curve lives on a 2**k square, so the pitch came
    out as the region's size over a power of two and a 10 nm request could
    be written at 7. A user asked for the separation to be the parameter, so
    the curve is now a spanning-tree covering path: 2x2 blocks of cells at
    the requested pitch, a spanning tree over the blocks, and a path that
    circumnavigates the tree. The tree is grown in Hilbert order, so the
    locality survives; what goes is only the requirement that the region be
    a power-of-two square.

    A disconnected region gives one stroke per component. The tip is never
    dragged across unwritten sample at bias, the same rule fill_polygons and
    fill_contour follow. The boundary is covered to within one block (two
    pitches), so pass `outline=True` when the edge has to be crisp.
    """
    polys = [np.asarray(p, float) for p in polys if len(np.asarray(p)) >= 3]
    if not polys or spacing_um <= 0:
        return []
    cell = float(spacing_um)
    mask, ox, oy = _region_mask(polys, cell, inset_um, max_cells)
    if mask is None or not mask.any():
        return []

    # a block is usable only if all four of its cells are inside, so the path
    # never steps outside the outline
    free = (mask[0::2, 0::2] & mask[1::2, 0::2]
            & mask[0::2, 1::2] & mask[1::2, 1::2])
    if not free.any():
        return []

    cx, cy = free.shape
    order = int(math.ceil(math.log2(max(cx, cy, 2))))
    hx, hy = hilbert_cells(order)
    rank = np.zeros((1 << order, 1 << order), dtype=np.int64)
    rank[hx, hy] = np.arange(hx.size)
    rank = rank[:cx, :cy]

    east, north, roots = _spanning_forest(free, rank)

    out: list[np.ndarray] = []
    if outline:
        out += [np.vstack([p, p[:1]]) for p in polys]
    for root in roots:
        cells = _cover_path(east, north, root)
        if len(cells) < 2:
            continue
        arr = np.asarray(cells, dtype=np.float64)
        out.append(np.column_stack([ox + (arr[:, 0] + 0.5) * cell,
                                    oy + (arr[:, 1] + 0.5) * cell]))
    return out


# ------------------------------------------------------------ radial fills --

def fill_radial(polys, spacing_um: float, centre=None, inward: bool = False,
                outline: bool = False, tol_um: float = 0.001,
                max_points: int = 400_000) -> list[np.ndarray]:
    """Fill by spiralling out from a point (or in towards it).

    User request, 2026-08-08: convex shapes filled "along some axis or radially
    outward/inward from a point". The axis case is fill_polygons; this is
    the other one.

    An Archimedean spiral, r = spacing * theta / 2pi, so successive turns are
    one `spacing_um` apart along every ray: the line separation is the
    parameter here too. One continuous pass while the spiral stays inside,
    which for a disc is the whole fill.

    `inward` only reverses the path. It is not cosmetic: writing outward
    leaves the tip at the rim and writing inward leaves it at the centre, and
    with decay in play the order in which a region is written is the order in
    which it ages.

    A region that is not a disc will be left and re-entered as the spiral
    grows past it -- a square is covered to the inscribed circle in one pass
    and the corners in arcs -- and each of those is a separate stroke rather
    than a chord dragged across the outside at bias.

    `centre` defaults to the region's centroid.
    """
    polys = [np.asarray(p, float) for p in polys if len(np.asarray(p)) >= 3]
    if not polys or spacing_um <= 0:
        return []
    if centre is None:
        centre = _centroid(polys[0])
    cxy = np.asarray(centre, dtype=np.float64).reshape(2)

    allp = np.vstack(polys)
    r_max = float(np.hypot(*(allp - cxy).T).max()) + spacing_um
    if r_max <= 0:
        return []

    # step in angle so the chord never departs from the arc by more than
    # tol_um: coarse near the centre, fine at the rim, which is where a
    # fixed step would either alias or waste points
    pts: list[tuple[float, float]] = []
    theta, k = 0.0, spacing_um / (2 * math.pi)
    theta_max = r_max / k
    while theta <= theta_max and len(pts) < max_points:
        r = k * theta
        pts.append((cxy[0] + r * math.cos(theta), cxy[1] + r * math.sin(theta)))
        step = 0.3 if r < tol_um else min(0.3, 2.0 * math.sqrt(2 * tol_um / r))
        theta += max(step, 1e-4)
    p = np.asarray(pts, dtype=np.float64)
    if len(p) < 2:
        return []

    keep = np.flatnonzero(_points_inside(polys, p))
    if keep.size == 0:
        return []
    runs = np.split(keep, np.flatnonzero(np.diff(keep) != 1) + 1)

    out: list[np.ndarray] = []
    if outline:
        out += [np.vstack([q, q[:1]]) for q in polys]
    for r in runs:
        if r.size < 2:
            continue
        arc = p[r]
        out.append(arc[::-1] if inward else arc)
    if inward:
        out.reverse()
    return out


def _centroid(poly: np.ndarray) -> np.ndarray:
    """Area centroid, falling back to the vertex mean for a degenerate ring."""
    p = np.asarray(poly, float)
    x, y = p[:, 0], p[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-12:
        return p.mean(axis=0)
    return np.array([((x + np.roll(x, -1)) * cross).sum() / (6 * a),
                     ((y + np.roll(y, -1)) * cross).sum() / (6 * a)])


def _points_inside(polys, pts: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon for many points at once, holes included."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    for poly in polys:
        n = len(poly)
        for k in range(n):
            ax, ay = poly[k]
            bx, by = poly[(k + 1) % n]
            if ay == by:
                continue
            hit = ((ay <= y) & (y < by)) | ((by <= y) & (y < ay))
            if not hit.any():
                continue
            t = np.where(hit, (y - ay) / (by - ay), 0.0)
            inside ^= hit & (x < ax + t * (bx - ax))
    return inside


def fill_contour(polys, spacing_um: float, inset_um: float = 0.0,
                 outline: bool = True) -> list[np.ndarray]:
    """Fill a region with rings that follow its own outline inward.

    The "self-similar shape-filling" strategy (2026-08-08). Each ring
    is the boundary offset inward by another `spacing_um`, so the toolpath is
    everywhere parallel to the edge instead of crossing it at an angle.

    Why it is worth having alongside the directional fill:

      - the written edge is one continuous pass, so the boundary of the shape
        comes out as smooth as the shape, rather than as the ends of a
        thousand parallel lines;
      - it handles concave regions without the long retraces a parallel
        sweep needs to get from one span to the next;
      - and when the region necks down, the rings SPLIT, which is correct and
        is why this is done by eroding a mask rather than offsetting an
        outline analytically.

    A directional sweep is still the better answer for a convex region where
    a single fast axis matters -- that is fill_polygons, and both are offered.
    """
    polys = [np.asarray(p, float) for p in polys if len(np.asarray(p)) >= 3]
    if not polys or spacing_um <= 0:
        return []
    rings: list[np.ndarray] = []
    for poly in polys:
        if outline and inset_um <= 0:
            rings.append(np.vstack([poly, poly[:1]]))
        depth = inset_um + (0.0 if (outline and inset_um <= 0) else spacing_um)
        depth = max(depth, spacing_um) if not outline else depth + spacing_um
        # ONE distance field, thresholded at every depth. It used to
        # rasterise and transform the polygon again for each ring, which is
        # the same work over and over: 13 seconds for a 2 um square at the
        # 10 nm pitch that is now the default, and that reads as a freeze.
        for got in _offset_rings(poly, depth, spacing_um):
            for ring in got:
                rings.append(np.vstack([ring, ring[:1]]))
    return rings
