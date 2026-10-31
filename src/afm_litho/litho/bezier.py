"""Cubic Bezier paths, with the semantics the lab already knows from Inkscape.

Virtual electrodes were drawn in Inkscape for years: a closed outline, its
sides pulled into a smooth funnel with Bezier handles, exported as SVG, and
raster-filled by the lithography program. That workflow is worth keeping;
Inkscape as a dependency is not.

So this module is the geometry half of it. A path is a list of nodes, each
carrying an anchor and two handles, exactly as Inkscape models them:

    cusp        handles independent, corner allowed
    smooth      handles collinear, lengths free
    symmetric   handles collinear and equal length

Curves are flattened by adaptive subdivision to a stated tolerance in
microns, which is the honest control: the tolerance is a lithography
parameter (how far the written path may stray from the designed curve), not
a rendering detail. Nothing here rasterises or fills; see fill.py.

SVG path data parses and re-emits, so files from the archive open directly
and anything drawn here can go back out to a tool that speaks SVG.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

CUSP, SMOOTH, SYMMETRIC = "cusp", "smooth", "symmetric"


def signed_area(poly) -> float:
    """Signed shoelace area of an implicitly-closed polygon; 0.0 if < 3 points."""
    p = np.asarray(poly, dtype=np.float64)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


@dataclass
class Node:
    """One anchor and its two handles, in microns, absolute coordinates.

    `h_in` governs the segment arriving at this node and `h_out` the one
    leaving it. Either may be None, which means "coincident with the
    anchor", i.e. that side is a straight line.
    """
    x: float
    y: float
    h_in: tuple[float, float] | None = None
    h_out: tuple[float, float] | None = None
    kind: str = CUSP

    @property
    def p(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)

    def handle_in(self) -> np.ndarray:
        return self.p if self.h_in is None else np.asarray(self.h_in, dtype=np.float64)

    def handle_out(self) -> np.ndarray:
        return self.p if self.h_out is None else np.asarray(self.h_out, dtype=np.float64)

    def move_to(self, x: float, y: float) -> None:
        """Drag the anchor; handles travel with it, as in a node editor."""
        dx, dy = x - self.x, y - self.y
        if self.h_in is not None:
            self.h_in = (self.h_in[0] + dx, self.h_in[1] + dy)
        if self.h_out is not None:
            self.h_out = (self.h_out[0] + dx, self.h_out[1] + dy)
        self.x, self.y = x, y

    def set_handle(self, which: str, x: float, y: float) -> None:
        """Move one handle, enforcing this node's continuity rule.

        This is the whole reason node kinds exist: on a smooth node the
        opposite handle swings to stay collinear (keeping its length), and
        on a symmetric node it mirrors exactly. A cusp lets the two move
        independently, which is how you get a corner.
        """
        v = np.array([x, y], dtype=np.float64) - self.p
        if which == "out":
            self.h_out = (x, y)
            other, keep = "h_in", self.handle_in()
        else:
            self.h_in = (x, y)
            other, keep = "h_out", self.handle_out()
        if self.kind == CUSP:
            return
        n = float(np.hypot(*v))
        if n < 1e-12:
            return
        u = -v / n
        length = float(np.hypot(*(keep - self.p)))
        if self.kind == SYMMETRIC or length < 1e-12:
            length = n
        q = self.p + u * length
        setattr(self, other, (float(q[0]), float(q[1])))

    def to_dict(self) -> dict:
        return {"x": round(self.x, 5), "y": round(self.y, 5),
                "h_in": None if self.h_in is None else [round(self.h_in[0], 5),
                                                        round(self.h_in[1], 5)],
                "h_out": None if self.h_out is None else [round(self.h_out[0], 5),
                                                          round(self.h_out[1], 5)],
                "kind": self.kind}

    @staticmethod
    def from_dict(d: dict) -> "Node":
        return Node(float(d["x"]), float(d["y"]),
                    None if d.get("h_in") is None else tuple(d["h_in"]),
                    None if d.get("h_out") is None else tuple(d["h_out"]),
                    d.get("kind", CUSP))


def _flatness(p0, p1, p2, p3) -> float:
    """Greatest distance of the control points from the chord.

    The usual cheap proxy for how far the curve can stray from the straight
    line joining its ends: a cubic lies inside the convex hull of its four
    control points, so this bounds the error conservatively.
    """
    chord = p3 - p0
    L = float(np.hypot(*chord))
    if L < 1e-12:
        return float(max(np.hypot(*(p1 - p0)), np.hypot(*(p2 - p0))))
    n = np.array([-chord[1], chord[0]]) / L
    return float(max(abs(np.dot(p1 - p0, n)), abs(np.dot(p2 - p0, n))))


def flatten_cubic(p0, p1, p2, p3, tol_um: float, depth: int = 0) -> list:
    """Adaptive subdivision of one cubic to within tol_um of the true curve."""
    p0, p1, p2, p3 = (np.asarray(q, dtype=np.float64) for q in (p0, p1, p2, p3))
    if depth >= 18 or _flatness(p0, p1, p2, p3) <= tol_um:
        return [p3]
    # de Casteljau split at t = 1/2
    a = (p0 + p1) / 2
    b = (p1 + p2) / 2
    c = (p2 + p3) / 2
    d = (a + b) / 2
    e = (b + c) / 2
    m = (d + e) / 2
    return (flatten_cubic(p0, a, d, m, tol_um, depth + 1)
            + flatten_cubic(m, e, c, p3, tol_um, depth + 1))


@dataclass
class BezierPath:
    nodes: list[Node] = field(default_factory=list)
    closed: bool = False

    def __len__(self) -> int:
        return len(self.nodes)

    def segments(self):
        """Yield (p0, c1, c2, p3) for each segment, closing if closed."""
        n = len(self.nodes)
        if n < 2:
            return
        last = n if self.closed else n - 1
        for i in range(last):
            a = self.nodes[i]
            b = self.nodes[(i + 1) % n]
            yield a.p, a.handle_out(), b.handle_in(), b.p

    def flatten(self, tol_um: float = 0.005) -> np.ndarray:
        """Polyline within tol_um of the curve. Microns in, microns out."""
        if not self.nodes:
            return np.empty((0, 2))
        pts = [self.nodes[0].p]
        for p0, c1, c2, p3 in self.segments():
            pts.extend(flatten_cubic(p0, c1, c2, p3, tol_um))
        out = np.asarray(pts, dtype=np.float64)
        if len(out) > 1:                      # drop repeats from touching ends
            keep = np.concatenate([[True], (np.hypot(*np.diff(out, axis=0).T) > 1e-12)])
            out = out[keep]
        return out

    def as_funnel(self):
        """A funnel outline: a flat base and two curved sides meeting at an apex.

        If this closed path is one -- exactly one straight segment (the base)
        and two curved segments (the sides) that share an apex node -- return
        the two side cubics as (p0, c1, c2, p3) control tuples, each running
        base-corner -> apex, plus the apex point. Otherwise None. This is the
        shape the SVG funnels have, and the family the 2013 fill was written
        for (fill.fill_funnel_from_curves).
        """
        if not self.closed or len(self.nodes) != 3:
            return None
        segs = list(self.segments())
        if len(segs) != 3:
            return None

        def straight(s):
            p0, c1, c2, p3 = (np.asarray(q, dtype=np.float64) for q in s)
            chord = float(np.hypot(*(p3 - p0)))
            if chord < 1e-9:
                return True
            cr = lambda a, b: abs(a[0] * b[1] - a[1] * b[0])  # noqa: E731
            return (cr(p3 - p0, c1 - p0) + cr(p3 - p0, c2 - p0)) <= 0.05 * chord * chord

        lines = [i for i, s in enumerate(segs) if straight(s)]
        if len(lines) != 1:                       # need exactly one flat base
            return None
        b = lines[0]
        s1, s2 = segs[(b + 1) % 3], segs[(b + 2) % 3]
        curve_a = tuple(np.asarray(q, dtype=np.float64) for q in s1)          # corner1->apex
        curve_b = tuple(np.asarray(q, dtype=np.float64)                       # corner0->apex
                        for q in (s2[3], s2[2], s2[1], s2[0]))                # (reversed)
        return curve_a, curve_b, np.asarray(s1[3], dtype=np.float64)

    def as_rect(self):
        """If this closed outline is a (possibly rotated) rectangle -- four
        straight sides meeting at right angles -- return the angle (radians) of
        its LONGER pair of edges, the direction a raster should run parallel to.
        Else None.

        A near-square rectangle has no stable principal axis, so the generic
        long-axis fill comes out diagonal; this gives the fill the rectangle's
        own edge direction instead.
        """
        if not self.closed or len(self.nodes) != 4:
            return None
        segs = list(self.segments())
        if len(segs) != 4:
            return None
        pts = [np.asarray(n.p, dtype=np.float64) for n in self.nodes]
        edges = [pts[(i + 1) % 4] - pts[i] for i in range(4)]

        def straight(s):
            p0, c1, c2, p3 = (np.asarray(q, dtype=np.float64) for q in s)
            chord = float(np.hypot(*(p3 - p0)))
            if chord < 1e-9:
                return False
            cr = lambda a, b: abs(a[0] * b[1] - a[1] * b[0])  # noqa: E731
            return (cr(p3 - p0, c1 - p0) + cr(p3 - p0, c2 - p0)) <= 0.05 * chord * chord

        if not all(straight(s) for s in segs):
            return None

        def unit(v):
            n = float(np.hypot(*v))
            return v / n if n > 1e-9 else v
        for i in range(4):                        # adjacent edges ~perpendicular
            if abs(float(np.dot(unit(edges[i]), unit(edges[(i + 1) % 4])))) > 0.08:
                return None
        e = edges[int(np.argmax([np.hypot(*ed) for ed in edges]))]   # longer pair
        return float(np.arctan2(e[1], e[0]))

    def bounds(self, tol_um: float = 0.01):
        p = self.flatten(tol_um)
        if not len(p):
            return (0.0, 0.0, 0.0, 0.0)
        return (float(p[:, 0].min()), float(p[:, 1].min()),
                float(p[:, 0].max()), float(p[:, 1].max()))

    def area(self, tol_um: float = 0.01) -> float:
        """Signed area by the shoelace formula; sign gives the winding."""
        return signed_area(self.flatten(tol_um))

    def to_dict(self) -> dict:
        return {"nodes": [n.to_dict() for n in self.nodes], "closed": self.closed}

    @staticmethod
    def from_dict(d: dict) -> "BezierPath":
        return BezierPath([Node.from_dict(n) for n in d.get("nodes", [])],
                          bool(d.get("closed", False)))

    def to_svg_d(self, precision: int = 4) -> str:
        """Emit SVG path data. Straight segments stay L, curves become C."""
        if not self.nodes:
            return ""
        f = lambda v: f"{v:.{precision}f}".rstrip("0").rstrip(".") or "0"  # noqa: E731
        out = [f"M {f(self.nodes[0].x)},{f(self.nodes[0].y)}"]
        for p0, c1, c2, p3 in self.segments():
            straight = (np.allclose(c1, p0, atol=1e-9)
                        and np.allclose(c2, p3, atol=1e-9))
            if straight:
                out.append(f"L {f(p3[0])},{f(p3[1])}")
            else:
                out.append(f"C {f(c1[0])},{f(c1[1])} {f(c2[0])},{f(c2[1])} "
                           f"{f(p3[0])},{f(p3[1])}")
        if self.closed:
            out.append("Z")
        return " ".join(out)


# ------------------------------------------------------------ SVG path data --

_TOKEN = re.compile(r"[MmZzLlHhVvCcSsQqTtAa]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def _arc_to_cubics(p0, rx, ry, phi_deg, large, sweep, p1):
    """Elliptical arc -> cubic segments (SVG F.6). Inkscape emits these for
    circles and rounded corners, so a parser that skips them silently loses
    geometry."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    if np.allclose(p0, p1):
        return []
    rx, ry = abs(rx), abs(ry)
    if rx < 1e-12 or ry < 1e-12:
        return [(p0, p0, p1, p1)]
    phi = math.radians(phi_deg)
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    d = (p0 - p1) / 2
    x1 = cos_p * d[0] + sin_p * d[1]
    y1 = -sin_p * d[0] + cos_p * d[1]
    lam = (x1 / rx) ** 2 + (y1 / ry) ** 2
    if lam > 1:                       # radii too small: scale up per spec
        s = math.sqrt(lam)
        rx, ry = rx * s, ry * s
    num = rx ** 2 * ry ** 2 - rx ** 2 * y1 ** 2 - ry ** 2 * x1 ** 2
    den = rx ** 2 * y1 ** 2 + ry ** 2 * x1 ** 2
    co = math.sqrt(max(num / den, 0.0)) if den > 0 else 0.0
    if large == sweep:
        co = -co
    cx1 = co * rx * y1 / ry
    cy1 = -co * ry * x1 / rx
    cx = cos_p * cx1 - sin_p * cy1 + (p0[0] + p1[0]) / 2
    cy = sin_p * cx1 + cos_p * cy1 + (p0[1] + p1[1]) / 2

    def angle(ux, uy, vx, vy):
        dot = ux * vx + uy * vy
        n = math.hypot(ux, uy) * math.hypot(vx, vy)
        a = math.acos(max(-1.0, min(1.0, dot / n))) if n else 0.0
        return -a if ux * vy - uy * vx < 0 else a

    th0 = angle(1, 0, (x1 - cx1) / rx, (y1 - cy1) / ry)
    dth = angle((x1 - cx1) / rx, (y1 - cy1) / ry,
                (-x1 - cx1) / rx, (-y1 - cy1) / ry)
    if not sweep and dth > 0:
        dth -= 2 * math.pi
    elif sweep and dth < 0:
        dth += 2 * math.pi

    n_seg = max(1, int(math.ceil(abs(dth) / (math.pi / 2))))
    out = []
    step = dth / n_seg
    k = 4 / 3 * math.tan(step / 4)
    for i in range(n_seg):
        a0 = th0 + i * step
        a1 = a0 + step

        def pt(a):
            x = rx * math.cos(a)
            y = ry * math.sin(a)
            return np.array([cos_p * x - sin_p * y + cx, sin_p * x + cos_p * y + cy])

        def dpt(a):
            x = -rx * math.sin(a)
            y = ry * math.cos(a)
            return np.array([cos_p * x - sin_p * y, sin_p * x + cos_p * y])

        s0, s1 = pt(a0), pt(a1)
        out.append((s0, s0 + k * dpt(a0), s1 - k * dpt(a1), s1))
    return out


def parse_svg_path(d: str) -> list[BezierPath]:
    """SVG path data -> BezierPaths. Every command in the spec is handled.

    Node kinds are inferred the way Inkscape would show them: a node whose
    two handles are collinear is smooth (symmetric if they are also equal in
    length), otherwise it is a cusp. That means a file drawn in Inkscape
    opens here with the handles behaving as they did there.
    """
    toks = _TOKEN.findall(d or "")
    i, cmd = 0, None
    cur = np.zeros(2)
    start = np.zeros(2)
    prev_ctrl = None          # for S/T reflection
    paths: list[BezierPath] = []
    path: BezierPath | None = None

    def num():
        nonlocal i
        v = float(toks[i])
        i += 1
        return v

    def ensure():
        nonlocal path
        if path is None:
            path = BezierPath([Node(float(cur[0]), float(cur[1]))])
        return path

    def add_curve(c1, c2, p3):
        p = ensure()
        p.nodes[-1].h_out = (float(c1[0]), float(c1[1]))
        p.nodes.append(Node(float(p3[0]), float(p3[1]),
                            h_in=(float(c2[0]), float(c2[1]))))

    def add_line(p3):
        p = ensure()
        p.nodes.append(Node(float(p3[0]), float(p3[1])))

    def finish():
        nonlocal path
        if path is not None and len(path.nodes) > 1:
            paths.append(path)
        path = None

    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            cmd = t
            i += 1
            if cmd in "Zz":
                if path is not None and len(path.nodes) > 1:
                    # a closing Z that lands on the start point means the
                    # last node IS the first: drop the duplicate
                    if np.allclose(path.nodes[-1].p, path.nodes[0].p, atol=1e-9):
                        h = path.nodes[-1].h_in
                        path.nodes.pop()
                        path.nodes[0].h_in = h
                    path.closed = True
                    finish()
                cur = start.copy()
                prev_ctrl = None
                continue
        if cmd is None:
            i += 1
            continue
        rel = cmd.islower()
        c = cmd.upper()
        base = cur if rel else np.zeros(2)

        match c:
            case "M":
                finish()
                cur = base + np.array([num(), num()])
                start = cur.copy()
                path = BezierPath([Node(float(cur[0]), float(cur[1]))])
                cmd = "l" if rel else "L"      # subsequent pairs are implicit L
                prev_ctrl = None
            case "L":
                cur = base + np.array([num(), num()])
                add_line(cur)
                prev_ctrl = None
            case "H":
                cur = np.array([(cur[0] if rel else 0.0) + num(), cur[1]])
                add_line(cur)
                prev_ctrl = None
            case "V":
                cur = np.array([cur[0], (cur[1] if rel else 0.0) + num()])
                add_line(cur)
                prev_ctrl = None
            case "C":
                c1 = base + np.array([num(), num()])
                c2 = base + np.array([num(), num()])
                p3 = base + np.array([num(), num()])
                add_curve(c1, c2, p3)
                cur, prev_ctrl = p3, c2
            case "S":
                c2 = base + np.array([num(), num()])
                p3 = base + np.array([num(), num()])
                c1 = cur if prev_ctrl is None else 2 * cur - prev_ctrl
                add_curve(c1, c2, p3)
                cur, prev_ctrl = p3, c2
            case "Q":
                q = base + np.array([num(), num()])
                p3 = base + np.array([num(), num()])
                c1 = cur + 2.0 / 3.0 * (q - cur)       # quadratic -> cubic, exact
                c2 = p3 + 2.0 / 3.0 * (q - p3)
                add_curve(c1, c2, p3)
                cur, prev_ctrl = p3, q
            case "T":
                q = cur if prev_ctrl is None else 2 * cur - prev_ctrl
                p3 = base + np.array([num(), num()])
                c1 = cur + 2.0 / 3.0 * (q - cur)
                c2 = p3 + 2.0 / 3.0 * (q - p3)
                add_curve(c1, c2, p3)
                cur, prev_ctrl = p3, q
            case "A":
                rx, ry, rot = num(), num(), num()
                large, sweep = int(num()), int(num())
                p3 = base + np.array([num(), num()])
                for (_, c1, c2, e) in _arc_to_cubics(cur, rx, ry, rot, large, sweep, p3):
                    add_curve(c1, c2, e)
                cur, prev_ctrl = p3, None
            case _:
                i += 1
                continue

    finish()
    for p in paths:
        _infer_kinds(p)
    return paths


def _infer_kinds(path: BezierPath) -> None:
    """Label nodes cusp / smooth / symmetric from their handle geometry."""
    n = len(path.nodes)
    for idx, nd in enumerate(path.nodes):
        if nd.h_in is None or nd.h_out is None:
            if not (path.closed and (idx == 0 or idx == n - 1)):
                nd.kind = CUSP
                continue
        a = nd.handle_in() - nd.p
        b = nd.handle_out() - nd.p
        na, nb = float(np.hypot(*a)), float(np.hypot(*b))
        if na < 1e-9 or nb < 1e-9:
            nd.kind = CUSP
            continue
        cosang = float(np.dot(a, b) / (na * nb))
        if cosang < -0.9995:                       # anti-parallel: collinear
            nd.kind = SYMMETRIC if abs(na - nb) <= 1e-6 * max(na, nb) else SMOOTH
        else:
            nd.kind = CUSP


# --------------------------------------------------------------- SVG files --

_LEN = re.compile(r"^\s*([-+0-9.eE]+)\s*([a-z%]*)\s*$")
# SVG user units per real-world unit, at the spec's 96 dpi
_PER_MM = 96.0 / 25.4
_UNITS = {"": 1.0, "px": 1.0, "mm": _PER_MM, "cm": 10 * _PER_MM,
          "in": 96.0, "pt": 96.0 / 72.0, "pc": 16.0, "m": 1000 * _PER_MM}


def _length(text: str, default: float = 0.0) -> tuple[float, str]:
    m = _LEN.match(text or "")
    if not m:
        return default, ""
    return float(m.group(1)), m.group(2)


def _parse_transform(text: str) -> np.ndarray:
    """SVG transform list -> 3x3 matrix. Inkscape leans on these heavily."""
    M = np.eye(3)
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", text or ""):
        v = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", args)]
        T = np.eye(3)
        match name:
            case "matrix" if len(v) == 6:
                T = np.array([[v[0], v[2], v[4]], [v[1], v[3], v[5]], [0, 0, 1]])
            case "translate":
                T[0, 2] = v[0]
                T[1, 2] = v[1] if len(v) > 1 else 0.0
            case "scale":
                T[0, 0] = v[0]
                T[1, 1] = v[1] if len(v) > 1 else v[0]
            case "rotate":
                a = math.radians(v[0])
                R = np.array([[math.cos(a), -math.sin(a), 0],
                              [math.sin(a), math.cos(a), 0], [0, 0, 1]])
                if len(v) >= 3:
                    T1 = np.eye(3); T1[0, 2], T1[1, 2] = v[1], v[2]
                    T2 = np.eye(3); T2[0, 2], T2[1, 2] = -v[1], -v[2]
                    R = T1 @ R @ T2
                T = R
            case "skewX":
                T[0, 1] = math.tan(math.radians(v[0]))
            case "skewY":
                T[1, 0] = math.tan(math.radians(v[0]))
            case _:                          # unknown transform (or matrix w/o 6 args): identity
                pass
        M = M @ T
    return M


def _apply(M: np.ndarray, p) -> tuple[float, float]:
    q = M @ np.array([p[0], p[1], 1.0])
    return float(q[0]), float(q[1])


def load_svg(text: str, um_per_unit: float | None = None,
             center: bool = True, preserve_origin: bool = False) -> list[BezierPath]:
    """Read an SVG into BezierPaths in microns.

    Handles the parts of SVG a design actually uses: <path>, <rect>,
    <circle>, <ellipse>, <polygon>, <polyline>, <line>, nested <g>
    transforms, and the document's own width/height/viewBox scaling.

    Scale: if um_per_unit is given it is used directly. Otherwise the
    document's physical width is read (an Inkscape drawing declares
    something like width="120mm") and treated as microns of sample, which
    is the convention the lab's archive already uses: the page IS the
    write field. Y is flipped, because SVG counts down the page and the
    sample counts up.

    `preserve_origin` keeps the PAGE ORIGIN instead of flipping y against the
    viewBox bottom and re-centring on the design's bounding box. Coordinates
    come out as `x_um = (x_svg - vx) * s` and `y_um = (y_svg - vy) * s`, i.e.
    microns measured RIGHT and DOWN from the top-left of the page, with no
    flip and no centring. This is what the design-origin rule wants (the
    upper-left of the page is the upper-left of the scan, see
    docs/DESIGN_ORIGIN_SPEC.md): the caller then maps page-origin microns to
    lab coordinates once, through `litho.frame.svg_to_lab`. When it is set,
    `center` is ignored. The default (False) is the historic behaviour --
    viewBox-bottom flip plus optional bbox centring -- kept for the callers
    and tests that rely on it.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(text)
    vb = (root.get("viewBox") or "").replace(",", " ").split()
    w_txt, h_txt = root.get("width", ""), root.get("height", "")
    w_val, w_unit = _length(w_txt)
    h_val, _ = _length(h_txt)

    if len(vb) == 4:
        vx, vy, vw, vh = (float(v) for v in vb)
    else:
        vx, vy, vw, vh = 0.0, 0.0, w_val or 100.0, h_val or 100.0

    if um_per_unit is None:
        # physical page width -> microns; fall back to 1 unit = 1 um
        if w_val and w_unit in ("mm", "cm", "in", "m", "pt", "pc"):
            page_um = {"mm": 1e3, "cm": 1e4, "in": 25.4e3,
                       "m": 1e6, "pt": 25.4e3 / 72, "pc": 25.4e3 / 6}[w_unit] * w_val
            um_per_unit = page_um / vw if vw else 1.0
        else:
            um_per_unit = 1.0

    paths: list[BezierPath] = []
    elem_seq = [0]                    # a distinct id per drawable ELEMENT (not subpath)

    def walk(el, M):
        M = M @ _parse_transform(el.get("transform", ""))
        tag = el.tag.split("}")[-1]
        d = None
        match tag:
            case "path":
                d = el.get("d")
            case "rect":
                x, _ = _length(el.get("x", "0")); y, _ = _length(el.get("y", "0"))
                w, _ = _length(el.get("width", "0")); h, _ = _length(el.get("height", "0"))
                d = f"M {x},{y} H {x + w} V {y + h} H {x} Z"
            case "circle" | "ellipse":
                cx, _ = _length(el.get("cx", "0")); cy, _ = _length(el.get("cy", "0"))
                if tag == "circle":
                    r, _ = _length(el.get("r", "0")); rx = ry = r
                else:
                    rx, _ = _length(el.get("rx", "0")); ry, _ = _length(el.get("ry", "0"))
                d = (f"M {cx - rx},{cy} A {rx},{ry} 0 1 0 {cx + rx},{cy} "
                     f"A {rx},{ry} 0 1 0 {cx - rx},{cy} Z")
            case "polygon" | "polyline":
                pts = (el.get("points") or "").replace(",", " ").split()
                if len(pts) >= 4:
                    d = "M " + " L ".join(f"{pts[i]},{pts[i + 1]}"
                                          for i in range(0, len(pts) - 1, 2))
                    if tag == "polygon":
                        d += " Z"
            case "line":
                x1, _ = _length(el.get("x1", "0")); y1, _ = _length(el.get("y1", "0"))
                x2, _ = _length(el.get("x2", "0")); y2, _ = _length(el.get("y2", "0"))
                d = f"M {x1},{y1} L {x2},{y2}"

        if d:
            # The element's `id` (if any) and a per-element sequence number are
            # carried on every subpath this element produces, so an importer can name
            # an object by its SVG id and still tell one element's subpaths (which
            # share an id) apart from two elements that collide on one.
            eid = el.get("id") or None
            this_elem = elem_seq[0]
            elem_seq[0] += 1
            for p in parse_svg_path(d):
                for nd in p.nodes:
                    nd.x, nd.y = _apply(M, (nd.x, nd.y))
                    if nd.h_in is not None:
                        nd.h_in = _apply(M, nd.h_in)
                    if nd.h_out is not None:
                        nd.h_out = _apply(M, nd.h_out)
                p.svg_id = eid
                p.svg_elem = this_elem
                paths.append(p)
        for child in el:
            walk(child, M)

    walk(root, np.eye(3))

    # viewBox -> microns. Default flips y against the viewBox bottom (SVG
    # counts down the page, the sample counts up); preserve_origin keeps the
    # page origin, y measured DOWN from the top, and does the flip later.
    s = um_per_unit
    for p in paths:
        for nd in p.nodes:
            if preserve_origin:
                nd.x = (nd.x - vx) * s
                nd.y = (nd.y - vy) * s
                if nd.h_in is not None:
                    nd.h_in = ((nd.h_in[0] - vx) * s, (nd.h_in[1] - vy) * s)
                if nd.h_out is not None:
                    nd.h_out = ((nd.h_out[0] - vx) * s, (nd.h_out[1] - vy) * s)
            else:
                nd.x = (nd.x - vx) * s
                nd.y = (vh - (nd.y - vy)) * s
                if nd.h_in is not None:
                    nd.h_in = ((nd.h_in[0] - vx) * s, (vh - (nd.h_in[1] - vy)) * s)
                if nd.h_out is not None:
                    nd.h_out = ((nd.h_out[0] - vx) * s, (vh - (nd.h_out[1] - vy)) * s)
        _infer_kinds(p)

    if center and not preserve_origin and paths:
        xs = [b for p in paths for b in (p.bounds()[0], p.bounds()[2])]
        ys = [b for p in paths for b in (p.bounds()[1], p.bounds()[3])]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        for p in paths:
            for nd in p.nodes:
                nd.move_to(nd.x - cx, nd.y - cy)
    return paths


def to_svg(paths: list[BezierPath], stroke: str = "#000",
           pad_um: float = 2.0) -> str:
    """Write paths back out as an SVG, in microns, y flipped back."""
    if not paths:
        return '<svg xmlns="http://www.w3.org/2000/svg"/>'
    bs = [p.bounds() for p in paths]
    x0 = min(b[0] for b in bs) - pad_um
    y0 = min(b[1] for b in bs) - pad_um
    x1 = max(b[2] for b in bs) + pad_um
    y1 = max(b[3] for b in bs) + pad_um
    w, h = x1 - x0, y1 - y0
    body = []
    for p in paths:
        q = BezierPath([Node(n.x - x0, y1 - n.y,
                             None if n.h_in is None else (n.h_in[0] - x0, y1 - n.h_in[1]),
                             None if n.h_out is None else (n.h_out[0] - x0, y1 - n.h_out[1]),
                             n.kind) for n in p.nodes], p.closed)
        body.append(f'<path d="{q.to_svg_d()}" fill="none" stroke="{stroke}" '
                    f'stroke-width="0.05"/>')
    # Coordinates are microns, so the page must be that many microns across:
    # declaring w mm against a viewBox of w units would make one unit a
    # millimetre, and the file would re-import a thousand times too big.
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w / 1000:.6g}mm" '
            f'height="{h / 1000:.6g}mm" viewBox="0 0 {w:.6g} {h:.6g}">'
            + "".join(body) + "</svg>")
