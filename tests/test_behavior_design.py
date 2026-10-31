"""Behavioural (invariant) tests for the design/geometry endpoints.

Where test_api_design.py drives the endpoints and checks their reported shape
(status, message strings, object counts), this file asserts the GEOMETRY the
endpoints actually produce. Each op is judged by a quantity computed from the
returned object's `points_um` -- polygon area, centroid, bounding box -- so a
regression that keeps the HTTP contract but corrupts the geometry (an outset
that shrinks, a translate that also scales, a union that double-counts the
overlap) fails here even though the endpoint tests would still pass.

Harness: the module-scoped `client` fixture (tests/conftest.py) is a FastAPI
a TestClient. The `hub` is a PROCESS GLOBAL shared across
every test, so each test below replaces the whole design first (import a fresh
SVG, which sets hub.pattern outright, or POST /api/sketch/clear) and never
leans on state another test left behind.
"""

from __future__ import annotations

import math

# `client` comes from tests/conftest.py (a TestClient).

TOL = 1e-6


# --------------------------------------------------------------- geometry --

def _area(pts):
    """Unsigned polygon area by the shoelace formula.

    Treats the vertex list as a closed ring (last joined back to first), so a
    ring that already repeats its first point as its last contributes a zero
    edge and the area is unaffected.
    """
    n = len(pts)
    s = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return abs(s) / 2.0


def _centroid(pts):
    """Area-weighted centroid of the closed polygon (the real centroid).

    Under a rigid translation every vertex shifts by the same vector, so this
    centroid shifts by exactly that vector -- which is the invariant the
    translate/duplicate tests below pin. Falls back to the vertex mean if the
    polygon is degenerate (zero signed area).
    """
    n = len(pts)
    a2 = 0.0          # twice the signed area
    cx = cy = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        a2 += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(a2) < 1e-12:
        return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)
    return (cx / (3.0 * a2), cy / (3.0 * a2))


def _bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


# ---------------------------------------------------------------- fixtures --

# A design-origin import needs the scan size the design was drawn over; the
# cockpit writes it in as a `topo_bgd` background image (width = size_um * 1000).
# These helpers embed one so imports resolve a size the way a real drawing does.
# Every assertion below is area/centroid/ordering based and so is invariant under
# the resulting translate + y-flip, so only the presence of the image is new.
def _bg(span_units):
    return (f'<image id="topo_bgd" x="0" y="0" '
            f'width="{span_units}" height="{span_units}"/>')


def _svg_rect(x, y, w, h):
    span = max(x + w, y + h)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">{_bg(span)}'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="black"/></svg>')


def _svg_two(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    span = max(ax + aw, bx + bw, ay + ah, by + bh)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{span}" height="{span}" '
            f'viewBox="0 0 {span} {span}">{_bg(span)}'
            f'<rect x="{ax}" y="{ay}" width="{aw}" height="{ah}" fill="black"/>'
            f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="black"/></svg>')


def _post_svg(client, svg, name="t.svg"):
    """Import an SVG. This REPLACES hub.pattern outright, so it doubles as the
    per-test reset: whatever an earlier test left is gone after this returns."""
    r = client.post(f"/api/design/svg?name={name}", content=svg,
                    headers={"content-type": "image/svg+xml"})
    j = r.json()
    assert j.get("ok"), j
    return j["design"]


def _op(client, **body):
    r = client.post("/api/design/op", json=body).json()
    assert r.get("ok") is True, r
    return r


def _objs(design):
    return design["objects"]


def _pts(design, i):
    return _objs(design)[i]["points_um"]


# ============================================================ offset ops ==

def test_outset_strictly_increases_area(client):
    d = _post_svg(client, _svg_rect(0, 0, 12000, 8000))
    before = _area(_pts(d, 0))
    r = _op(client, op="outset", indices=[0], delta_um=0.5)
    after = _area(_pts(r["design"], 0))
    assert after > before, (before, after)
    # a 12x8 um box grown 0.5 um on every side is ~13x9; the growth is real,
    # not a rounding wobble
    assert after - before > 1.0


def test_inset_strictly_decreases_area(client):
    d = _post_svg(client, _svg_rect(0, 0, 12000, 8000))
    before = _area(_pts(d, 0))
    r = _op(client, op="inset", indices=[0], delta_um=0.5)
    after = _area(_pts(r["design"], 0))
    assert after < before, (before, after)
    assert before - after > 1.0


# ============================================================ transform ==

def test_translate_moves_centroid_by_exact_offset_and_preserves_area(client):
    d = _post_svg(client, _svg_two((0, 0, 12000, 12000), (6000, 6000, 12000, 12000)))
    p0 = _pts(d, 0)
    c0 = _centroid(p0)
    a0 = _area(p0)

    dx, dy = 3.0, -1.5
    r = _op(client, op="transform", indices=[0], dx_um=dx, dy_um=dy)
    p1 = _pts(r["design"], 0)
    c1 = _centroid(p1)
    a1 = _area(p1)

    assert abs((c1[0] - c0[0]) - dx) < TOL
    assert abs((c1[1] - c0[1]) - dy) < TOL
    # a rigid move changes position, not size
    assert abs(a1 - a0) < TOL


# ============================================================ duplicate ==

def test_duplicate_adds_one_offset_copy_and_leaves_original_untouched(client):
    d = _post_svg(client, _svg_two((0, 0, 12000, 12000), (16000, 16000, 12000, 12000)))
    n0 = len(_objs(d))
    orig_pts = _pts(d, 0)
    orig_centroid = _centroid(orig_pts)

    dx, dy = 5.0, 2.0
    r = _op(client, op="duplicate", indices=[0], dx_um=dx, dy_um=dy)
    objs = _objs(r["design"])

    # object count went up by exactly one
    assert len(objs) == n0 + 1
    # the panel selects the new copy; use that index rather than assuming order
    assert len(r["selection"]) == 1
    copy_pts = objs[r["selection"][0]]["points_um"]
    copy_centroid = _centroid(copy_pts)
    assert abs((copy_centroid[0] - orig_centroid[0]) - dx) < TOL
    assert abs((copy_centroid[1] - orig_centroid[1]) - dy) < TOL
    # the original object is byte-for-byte where it was
    assert objs[0]["points_um"] == orig_pts


# ================================================================ union ==

def test_union_of_overlapping_rects_is_one_region_within_area_bounds(client):
    d = _post_svg(client, _svg_two((0, 0, 12000, 12000), (6000, 6000, 12000, 12000)))
    a0 = _area(_pts(d, 0))
    a1 = _area(_pts(d, 1))

    r = _op(client, op="union", indices=[0, 1])
    objs = _objs(r["design"])
    assert len(objs) == 1                         # two overlapping regions -> one
    au = _area(objs[0]["points_um"])

    # the union covers at least the larger input and less than both summed:
    # the shared overlap is counted once, not twice
    assert au >= max(a0, a1) - TOL
    assert au < a0 + a1 - TOL


# =========================================================== difference ==

def test_difference_strictly_shrinks_the_first_shape(client):
    d = _post_svg(client, _svg_two((0, 0, 12000, 12000), (6000, 6000, 12000, 12000)))
    a0 = _area(_pts(d, 0))

    r = _op(client, op="difference", indices=[0, 1])
    objs = _objs(r["design"])
    assert len(objs) >= 1
    # subtracting the overlap leaves strictly less than the first shape had
    total_after = sum(_area(o["points_um"]) for o in objs)
    assert total_after < a0 - TOL
    assert total_after > 0.0


# ============================================================== z-order ==

def _index_of(design, name):
    return [o["name"] for o in _objs(design)].index(name)


def test_raise_moves_target_one_step_up_the_stack(client):
    d = _post_svg(client, _svg_two((0, 0, 12000, 12000), (16000, 16000, 12000, 12000)))
    target = _objs(d)[0]["name"]
    i0 = _index_of(d, target)                     # 0
    r = _op(client, op="raise", indices=[0])
    i1 = _index_of(r["design"], target)
    assert i1 == i0 + 1                            # rose one place


def test_lower_moves_target_one_step_down_the_stack(client):
    d = _post_svg(client, _svg_two((0, 0, 12000, 12000), (16000, 16000, 12000, 12000)))
    target = _objs(d)[1]["name"]
    i0 = _index_of(d, target)                     # 1
    r = _op(client, op="lower", indices=[1])
    i1 = _index_of(r["design"], target)
    assert i1 == i0 - 1                            # sank one place


def test_top_moves_target_to_the_end_and_bottom_to_the_start(client):
    d = _post_svg(client, _svg_two((0, 0, 12000, 12000), (16000, 16000, 12000, 12000)))
    first = _objs(d)[0]["name"]
    r = _op(client, op="top", indices=[0])
    assert _index_of(r["design"], first) == len(_objs(r["design"])) - 1

    d2 = _post_svg(client, _svg_two((0, 0, 12000, 12000), (16000, 16000, 12000, 12000)))
    last = _objs(d2)[-1]["name"]
    last_idx = len(_objs(d2)) - 1
    r2 = _op(client, op="bottom", indices=[last_idx])
    assert _index_of(r2["design"], last) == 0


# ======================================================= SVG scale fidelity ==

def test_svg_rect_imports_at_one_nm_per_unit(client):
    # the service fixes 1 SVG unit = 1 nm = 1e-3 um; a W x H unit rect must land
    # as a shape whose bbox is W*1e-3 x H*1e-3 um. Two sizes so a coincidence at
    # one aspect ratio cannot pass.
    for w_units, h_units in [(20000, 10000), (7000, 30000)]:
        d = _post_svg(client, _svg_rect(0, 0, w_units, h_units))
        x0, y0, x1, y1 = _bbox(_pts(d, 0))
        assert math.isclose(x1 - x0, w_units * 1e-3, rel_tol=1e-3, abs_tol=1e-3)
        assert math.isclose(y1 - y0, h_units * 1e-3, rel_tol=1e-3, abs_tol=1e-3)


# ======================================================= sketch round-trip ==

def test_sketch_polyline_traces_the_requested_points(client):
    client.post("/api/sketch/clear")
    # an L: right along the bottom, then up the right side
    poly = [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]]
    r = client.post("/api/sketch",
                    json={"pen": "write", "points_um": poly, "mode": "literal"}).json()
    assert r.get("ok") is True
    pts = _objs(r["design"])[0]["points_um"]
    assert len(pts) >= 2

    # the service resamples the stroke but keeps its geometry: endpoints land on
    # the requested endpoints and the traced bbox is the polyline's bbox. (Probed
    # empirically -- literal-mode sketches are NOT recentred.)
    assert math.isclose(pts[0][0], poly[0][0], abs_tol=1e-3)
    assert math.isclose(pts[0][1], poly[0][1], abs_tol=1e-3)
    assert math.isclose(pts[-1][0], poly[-1][0], abs_tol=1e-3)
    assert math.isclose(pts[-1][1], poly[-1][1], abs_tol=1e-3)

    bx0, by0, bx1, by1 = _bbox(pts)
    want = _bbox(poly)
    assert math.isclose(bx0, want[0], abs_tol=1e-3)
    assert math.isclose(by0, want[1], abs_tol=1e-3)
    assert math.isclose(bx1, want[2], abs_tol=1e-3)
    assert math.isclose(by1, want[3], abs_tol=1e-3)

    # the corner really is on the trace, so this is the L and not the diagonal
    assert any(math.isclose(x, 5.0, abs_tol=1e-2) and math.isclose(y, 0.0, abs_tol=1e-2)
               for x, y in pts)
    client.post("/api/sketch/clear")
