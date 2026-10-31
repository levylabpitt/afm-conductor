"""Path operations: union, difference, offset, simplify.

Inkscape's `Ctrl +`, `Ctrl -`, `Ctrl *`, `Ctrl ^`, `Ctrl (`, `Ctrl )` and
`Ctrl L`, on this program's lithography objects. `docs/INKSCAPE.md` argues
these are not graphics luxuries here and that argument is the reason this
file exists:

  **outset is how a lead gets its width.** A wire is drawn as a centre line
  and written as an area, and until now the only thing that could turn one
  into the other was a one-off lead builder that composed with nothing.
  `offset` on an open path does it for any
  path, which is Inkscape's stroke-to-path, and the result is an ordinary
  shape that the next operation can take as input.

  **union and difference are how a device is drawn.** Two overlapping regions
  written as two objects are written twice, with a seam where they meet and
  the tip lifting between them; unioned first, they are one region with one
  fill and no seam.

  **simplify is toolpath reduction**, which the compiler wants anyway.

The geometry is klayout's, not mine. Polygon booleans are the classic thing
that looks like an afternoon and is not: degenerate edges, self-intersection,
holes inside holes. `klayout.db` already does them correctly and is already
what `oasis.py` reads and writes the canonical format with, so the design
surface and the interchange format agree by construction rather than by two
implementations happening to match.

Everything here works in microns at the boundary and in integer database
units inside. `DBU_PER_UM` is 10^4, so the grid is 0.1 nm: two and a half
orders of magnitude below the 25 nm feature this instrument writes, and far
enough below it that snapping is never what moved an edge.
"""

from __future__ import annotations

import math
import operator

import numpy as np

from .compiler import LithoObject
from .pens import chiral_settings, chiral_spine, is_chiral, rebake_chiral

DBU_PER_UM = 10_000.0          # 0.1 nm grid

#: The operations, named as Inkscape names them, with its keys.
BOOLEAN_OPS = {
    "union": ("Union", "Ctrl +"),
    "difference": ("Difference", "Ctrl -"),
    "intersection": ("Intersection", "Ctrl *"),
    "exclusion": ("Exclusion", "Ctrl ^"),
}

# How each boolean op combines two regions (keys mirror BOOLEAN_OPS): the region
# type overloads +, -, &, ^, so the op is just a binary operator to fold with.
_REGION_COMBINE = {
    "union": operator.add,
    "difference": operator.sub,
    "intersection": operator.and_,
    "exclusion": operator.xor,
}


class PathOpError(Exception):
    """A refusal with a reason a person can act on."""


class ChiralRefused(PathOpError):
    """A refusal about a chiral wire, with a stable `reason` code for the caller."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def _db():
    try:
        import klayout.db as db
    except ImportError as exc:                       # pragma: no cover
        raise PathOpError(
            "the geometry library is not installed: pip install klayout") from exc
    return db


# ------------------------------------------------------------- to and from --

def _ring_dbu(db, pts):
    return [db.Point(int(round(x * DBU_PER_UM)), int(round(y * DBU_PER_UM)))
            for x, y in pts]


def _closed(pts: list) -> list[tuple[float, float]]:
    """Drop a repeated last point: klayout closes a polygon itself."""
    out = [(float(x), float(y)) for x, y in pts]
    while len(out) > 1 and math.isclose(out[0][0], out[-1][0], abs_tol=1e-9) \
            and math.isclose(out[0][1], out[-1][1], abs_tol=1e-9):
        out.pop()
    return out


def is_area(o: LithoObject) -> bool:
    """Whether this object encloses an area, as against being a line."""
    if o.kind == "rect":
        return len(o.points_um) >= 2
    return o.kind == "shape" and len(_closed(o.points_um)) >= 3


def to_region(o: LithoObject, width_um: float = 0.0):
    """One object as a klayout Region.

    An open path has no area, so a boolean on one is a question with no
    answer and is refused by name rather than by silently returning nothing.
    `width_um` is the one exception: it is what `offset` uses to give a line
    a width, which is the whole point of offsetting a line.
    """
    db = _db()
    if o.kind == "rect" and len(o.points_um) >= 2:
        (x0, y0), (x1, y1) = o.points_um[0], o.points_um[1]
        return db.Region(db.Box(*_ring_dbu(db, [(min(x0, x1), min(y0, y1)),
                                                (max(x0, x1), max(y0, y1))])))
    if o.kind == "shape":
        hull = _closed(o.points_um)
        if len(hull) < 3:
            raise PathOpError(f"{o.name} has no outline to work with")
        poly = db.Polygon(_ring_dbu(db, hull))
        for h in (o.holes_um or []):
            ring = _closed(h)
            if len(ring) >= 3:
                poly.insert_hole(_ring_dbu(db, ring))
        return db.Region(poly)
    if width_um > 0 and len(o.points_um) >= 2:
        path = db.Path(_ring_dbu(db, o.points_um),
                       int(round(width_um * DBU_PER_UM)))
        return db.Region(path.polygon())
    raise PathOpError(
        f"{o.name} is a line, not a region. Give it a width first "
        f"(outset, Ctrl+parenthesis) and then it can be combined.")


def _um(p) -> tuple[float, float]:
    return (p.x / DBU_PER_UM, p.y / DBU_PER_UM)


def from_region(region, template: LithoObject, name: str) -> list[LithoObject]:
    """A Region back into lithography objects, one per merged polygon.

    Everything about how the result gets WRITTEN -- bias, speed, fill pitch --
    comes from the template, which is the first object the user selected.
    Only the geometry is new. A boolean that quietly reset the write bias to
    a default would be a geometry tool that changes the physics.
    """
    out: list[LithoObject] = []
    region = region.merged()
    for i, poly in enumerate(region.each()):
        hull = [_um(p) for p in poly.each_point_hull()]
        if len(hull) < 3:
            continue
        holes = []
        for h in range(poly.holes()):
            ring = [_um(p) for p in poly.each_point_hole(h)]
            if len(ring) >= 3:
                holes.append(ring + [ring[0]])
        out.append(LithoObject(
            name=name if i == 0 else f"{name}.{i}",
            kind="shape",
            points_um=hull + [hull[0]],
            holes_um=holes,
            # geometry only: the outline is no longer any Bezier's flattening,
            # so saying otherwise would leave the
            # node editor dragging control points that describe a shape that
            # no longer exists
            bezier=None,
            profile=None,
            voltage=template.voltage,
            voltage_end=template.voltage_end,
            speed_um_s=template.speed_um_s,
            interpolation_um=template.interpolation_um,
            fill_spacing_um=template.fill_spacing_um,
            fill_angle_deg=template.fill_angle_deg,
            inset_um=template.inset_um,
            outline_pass=template.outline_pass,
            enabled=template.enabled,
        ))
    return out


# ------------------------------------------------------------------ booleans --

def boolean(objs: list[LithoObject], op: str,
            name: str | None = None) -> list[LithoObject]:
    """Combine objects. The first one is the base, as in Inkscape.

    Difference is `bottom minus everything above it`, so which object is
    which matters and z-order is the answer, exactly as Inkscape does it.
    Union and intersection do not care about order; difference and division
    do, and having one rule for all four is what makes the result
    predictable.
    """
    if op not in BOOLEAN_OPS:
        raise PathOpError(f"no such operation: {op}")
    if len(objs) < 2:
        raise PathOpError(f"{BOOLEAN_OPS[op][0]} needs at least two objects "
                          f"selected ({len(objs)} selected)")
    combine = _REGION_COMBINE[op]           # op is validated against BOOLEAN_OPS above
    base = to_region(objs[0])
    for o in objs[1:]:
        base = combine(base, to_region(o))
    base = base.merged()
    if base.is_empty():
        raise PathOpError(
            f"{BOOLEAN_OPS[op][0]} leaves nothing: the objects do not overlap "
            f"the way that operation needs. Nothing was changed.")
    return from_region(base, objs[0], name or f"{op}{abs(hash(op)) % 97}")


# ------------------------------------------------------------------- offset --

def offset(o: LithoObject, delta_um: float,
           name: str | None = None) -> list[LithoObject]:
    """Grow (outset) or shrink (inset) an object by delta.

    On an open path this is stroke-to-path: the line becomes a region of
    width 2*delta centred on it. That is how a wire drawn by hand becomes a
    lead with a width, and it is the operation this program most needed and
    did not have.
    """
    if abs(delta_um) < 1e-9:
        raise PathOpError("an offset of zero would change nothing")
    if is_area(o):
        region = to_region(o)
        region.size(int(round(delta_um * DBU_PER_UM)))
        if region.is_empty():
            raise PathOpError(
                f"inset by {abs(delta_um):g} um consumes {o.name} entirely "
                f"(it is narrower than that). Nothing was changed.")
    else:
        if delta_um < 0:
            raise PathOpError(
                f"{o.name} is a line with no width, so there is nothing to "
                f"take away. Outset gives it one.")
        region = to_region(o, width_um=2.0 * delta_um)
    return from_region(region, o, name or f"{o.name}+")


# ----------------------------------------------------------------- simplify --

def _dp(pts: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker. Endpoints are never moved."""
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        ab = b - a
        n = math.hypot(*ab)
        seg = pts[i + 1:j] - a
        if n < 1e-12:
            d = np.hypot(seg[:, 0], seg[:, 1])
        else:
            d = np.abs(seg[:, 0] * ab[1] - seg[:, 1] * ab[0]) / n
        k = int(np.argmax(d))
        if d[k] > tol:
            k += i + 1
            keep[k] = True
            stack += [(i, k), (k, j)]
    return pts[keep]


def simplify(o: LithoObject, tol_um: float,
             name: str | None = None) -> LithoObject:
    """Drop vertices that are not carrying the shape.

    Douglas-Peucker with the tolerance stated in microns, so the guarantee
    is the one that matters on an instrument: no point of the original
    outline is further than `tol_um` from the simplified one. The tolerance
    is what the caller should think about, not a vertex count.
    """
    if tol_um <= 0:
        raise PathOpError("a tolerance of zero would drop nothing")
    pts = np.asarray(o.points_um, dtype=np.float64)
    if len(pts) < 3:
        raise PathOpError(f"{o.name} has nothing to simplify")
    closed = bool(np.allclose(pts[0], pts[-1]))
    body = _dp(pts[:-1] if closed else pts, tol_um)
    if closed and len(body) < 3:
        raise PathOpError(
            f"simplifying {o.name} at {tol_um:g} um leaves less than a "
            f"triangle. Nothing was changed.")
    kept = np.vstack([body, body[:1]]) if closed else body
    holes = []
    for h in (o.holes_um or []):
        hp = np.asarray(h, dtype=np.float64)
        hc = bool(len(hp) > 2 and np.allclose(hp[0], hp[-1]))
        hb = _dp(hp[:-1] if hc else hp, tol_um)
        if len(hb) >= 3:
            holes.append([(float(x), float(y))
                          for x, y in (np.vstack([hb, hb[:1]]) if hc else hb)])
    out = LithoObject(
        name=name or o.name, kind=o.kind,
        points_um=[(float(x), float(y)) for x, y in kept],
        holes_um=holes,
        # the control points described the old vertex set; keeping them would
        # let the next node edit undo the simplification without saying so
        bezier=None, profile=None,
        voltage=o.voltage, voltage_end=o.voltage_end,
        speed_um_s=o.speed_um_s, interpolation_um=o.interpolation_um,
        fill_spacing_um=o.fill_spacing_um, fill_angle_deg=o.fill_angle_deg,
        inset_um=o.inset_um, outline_pass=o.outline_pass, enabled=o.enabled,
    )
    # A simplified wire keeps its bias profile, resampled onto the vertices
    # that survived, by arc length. Dropping it would turn a superlattice
    # into a plain wire as a side effect of tidying its geometry.
    if o.v_profile:
        out.v_profile = list(o.v_profile)
    return out


# ---------------------------------------------------------------- transform --

def transform(o: LithoObject, dx_um: float = 0.0, dy_um: float = 0.0,
              sx: float = 1.0, sy: float = 1.0, rot_deg: float = 0.0,
              about: tuple[float, float] | None = None) -> LithoObject:
    """Move, scale, rotate and flip, about a point.

    Flipping is `sx = -1` or `sy = -1` -- Inkscape's `h` and `v` -- which is
    why there is one function and not four. Everything derived from the
    geometry is transformed WITH it: the Bezier control points, the chiral
    spine.

    A chiral wire is its spine plus its chiral settings, so a scale, rotation or
    flip is applied to the spine and the wire is re-baked from it: pitch and
    amplitude stay as set. A pure move translates the baked points, which is the
    same wire. A chiral wire that cannot be re-baked (no spine or settings, or the
    builder refuses) raises ChiralRefused with `o` untouched; one whose spine the
    transform maps onto itself is returned unchanged.
    """
    c = np.asarray(about if about is not None else centre_of(o), dtype=np.float64)
    th = math.radians(rot_deg)
    ct, st = math.cos(th), math.sin(th)

    def T(pts):
        p = np.asarray(pts, dtype=np.float64)
        if p.size == 0:
            return p
        q = (p - c) * np.array([sx, sy])
        x = q[:, 0] * ct - q[:, 1] * st
        y = q[:, 0] * st + q[:, 1] * ct
        return np.column_stack([x, y]) + c + np.array([dx_um, dy_um])

    def L(pts):
        return [(float(a), float(b)) for a, b in T(pts)]

    rebaked = False
    if is_chiral(o) and (sx != 1.0 or sy != 1.0 or rot_deg != 0.0):
        spine = chiral_spine(o)
        if spine is None or chiral_settings(o) is None:
            raise ChiralRefused(
                "chiral_unrebakeable",
                f"object {o.name!r} is a chiral wire with no spine or settings to "
                f"re-bake from, so it cannot be scaled, rotated or flipped. Re-apply "
                f"its chiral settings.")
        moved = L(spine)
        if np.allclose(moved, spine, rtol=0.0, atol=1e-12):
            return o
        before = getattr(o, "axis_um", None)
        o.axis_um = moved
        try:
            rebake_chiral(o)
        except ValueError as exc:
            o.axis_um = before
            raise ChiralRefused(
                "chiral_build_refused",
                f"object {o.name!r} is a chiral wire and cannot be re-baked after "
                f"this transform: {exc}") from exc
        rebaked = True
    if not rebaked:
        o.points_um = L(o.points_um)
        # The base axis a modulation was built from travels WITH the geometry too,
        # or a later regenerate-from-axis (node edit / chiral re-parametrize) would
        # snap the wire back to where the axis was left behind.
        if getattr(o, "axis_um", None):
            o.axis_um = L(o.axis_um)
    o.holes_um = [L(h) for h in (o.holes_um or [])]
    # The radial fills radiate from this point; left behind, a moved or pasted
    # region would fill from its old centre.
    if getattr(o, "fill_centre_um", None) is not None:
        o.fill_centre_um = L([o.fill_centre_um])[0]
    o.bezier = _bez(o.bezier, T)
    if o.profile and not rebaked:
        p = dict(o.profile)
        for key in ("axis", "axis_um", "points_um", "path_um"):
            if isinstance(p.get(key), list) and p[key] and \
                    isinstance(p[key][0], (list, tuple)):
                p[key] = L(p[key])
        o.profile = p
    return o


def _bez(bez, T):
    if not bez or not bez.get("nodes"):
        return bez
    b = {k: v for k, v in bez.items() if k != "nodes"}
    nodes = []
    for nd in bez["nodes"]:
        n = dict(nd)
        (n["x"], n["y"]) = [float(v) for v in T([[nd["x"], nd["y"]]])[0]]
        for h in ("h_in", "h_out"):
            if n.get(h):
                n[h] = [float(v) for v in T([n[h]])[0]]
        nodes.append(n)
    b["nodes"] = nodes
    return b


def bbox_of(objs) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for o in objs:
        for x, y in (o.points_um or []):
            xs.append(float(x))
            ys.append(float(y))
    if not xs:
        raise PathOpError("nothing to measure")
    return min(xs), min(ys), max(xs), max(ys)


def centre_of(o) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox_of([o])
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
