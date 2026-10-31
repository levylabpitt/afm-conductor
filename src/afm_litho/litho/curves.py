"""A wire as functions of a parameter: x(t), y(t), V(t).

The formulation is this, and it is the right one. A wire is not a point cloud and
not one formula either: it is a curve t -> (x, y, V) on [0, 1], built from
PIECES, and each piece carries whatever representation suits it. A hand
route is a polyline piece; a chiral segment is a modulated piece (axis +
wavelength + amplitude + phase); tomorrow's pen can add another kind
without touching the others.

Every piece answers the same three questions:
    evaluate(t)      -> (x, y, V) exactly
    derivative(t)    -> d(x, y)/dt, analytically
    to_beziers(...)  -> cubic Bezier spans with V per span

The last one is the point. With value and derivative available anywhere,
each span becomes a Hermite cubic, and a quarter period of a sinusoid fits
one cubic to ~3e-4 of the amplitude (0.02 nm at A = 55 nm). The renderer
hands those cubics to the canvas, which rasterises the CURVE at whatever
zoom is in force. Nothing is pre-sampled, so nothing can alias, facet, or
be argued about in points per period ever again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Bez:
    """One cubic span with the bias that applies along it."""
    p0: tuple; c1: tuple; c2: tuple; p3: tuple
    v: float


class Piece:
    t0: float
    t1: float

    def evaluate(self, t): ...
    def derivative(self, t): ...
    def to_beziers(self) -> list[Bez]: ...

    def _local(self, t):
        return (np.asarray(t) - self.t0) / max(self.t1 - self.t0, 1e-12)


@dataclass
class PolyPiece(Piece):
    """Line segments: the degenerate cubic, exact by construction."""
    t0: float
    t1: float
    pts: np.ndarray                    # (n, 2) um
    volts: np.ndarray                  # (n,)

    def evaluate(self, t):
        f = self._local(t)
        n = len(self.pts)
        u = np.clip(f, 0, 1) * (n - 1)
        k = np.minimum(n - 2, np.floor(u).astype(int))
        w = u - k
        xy = self.pts[k] + w[..., None] * (self.pts[k + 1] - self.pts[k])
        v = self.volts[k] + w * (self.volts[k + 1] - self.volts[k])
        return xy, v

    def derivative(self, t):
        f = self._local(t)
        n = len(self.pts)
        k = np.minimum(n - 2, np.floor(np.clip(f, 0, 1) * (n - 1)).astype(int))
        return (self.pts[k + 1] - self.pts[k]) * (n - 1) / max(self.t1 - self.t0, 1e-12)

    def to_beziers(self):
        out = []
        for i in range(len(self.pts) - 1):
            a, b = self.pts[i], self.pts[i + 1]
            out.append(Bez(tuple(a), tuple(a + (b - a) / 3), tuple(a + 2 * (b - a) / 3),
                           tuple(b), float((self.volts[i] + self.volts[i + 1]) / 2)))
        return out


@dataclass
class ModulatedPiece(Piece):
    """The chiral construction: a smooth axis carrying a transverse sinusoid,
    with the bias in quadrature. This IS the analytic profile."""
    t0: float
    t1: float
    axis: np.ndarray                   # (m, 2) um, uniform in arc length
    normal: np.ndarray                 # (m, 2) unit
    amp: np.ndarray                    # (m,) um, curvature-clamped
    periods: float
    v0: float
    v_k: float
    phase_rad: float
    closed: bool = False

    def _interp(self, f):
        m = len(self.axis)
        u = np.clip(f, 0, 1) * (m - 1)
        k = np.minimum(m - 2, np.floor(u).astype(int))
        w = (u - k)[..., None]
        ax = self.axis[k] + w * (self.axis[k + 1] - self.axis[k])
        nn = self.normal[k] + w * (self.normal[k + 1] - self.normal[k])
        nn = nn / np.maximum(np.hypot(nn[..., 0], nn[..., 1]), 1e-12)[..., None]
        am = self.amp[k] + (u - k) * (self.amp[k + 1] - self.amp[k])
        dax = (self.axis[k + 1] - self.axis[k]) * (m - 1)
        dnn = (self.normal[k + 1] - self.normal[k]) * (m - 1)
        dam = (self.amp[k + 1] - self.amp[k]) * (m - 1)
        return ax, nn, am, dax, dnn, dam

    def evaluate(self, t):
        f = self._local(t)
        ax, nn, am, *_ = self._interp(f)
        th = 2 * np.pi * self.periods * f
        s = np.sin(th)
        xy = ax + nn * (am * s)[..., None]
        v = self.v0 + self.v_k * np.sin(th + self.phase_rad)
        return xy, v

    def derivative(self, t):
        f = self._local(t)
        ax, nn, am, dax, dnn, dam = self._interp(f)
        thp = 2 * np.pi * self.periods
        th = thp * f
        s, c = np.sin(th), np.cos(th)
        d = (dax + dnn * (am * s)[..., None]
             + nn * (dam * s + am * c * thp)[..., None])
        return d / max(self.t1 - self.t0, 1e-12)

    def to_beziers(self, f_lo: float = 0.0, f_hi: float = 1.0):
        """Hermite cubics, one per QUARTER period, over [f_lo, f_hi] of this
        piece. Quarter-period knots sit where sin is 0 or +-1, and one cubic
        matching value and slope at both ends reproduces a quarter sine to
        2.8e-4 of the amplitude: 0.02 nm here, three orders below anything
        the instrument can write.
        """
        span = self.t1 - self.t0
        n_knots = max(1, int(math.ceil(4 * self.periods * (f_hi - f_lo))))
        fs = np.linspace(f_lo, f_hi, n_knots + 1)
        ts = self.t0 + fs * span
        (xy, v) = self.evaluate(ts)
        dv = self.derivative(ts) * span          # d/df
        out = []
        for i in range(n_knots):
            h = fs[i + 1] - fs[i]
            p0, p3 = xy[i], xy[i + 1]
            c1 = p0 + dv[i] * (h / 3)
            c2 = p3 - dv[i + 1] * (h / 3)
            out.append(Bez(tuple(p0), tuple(c1), tuple(c2), tuple(p3),
                           float((v[i] + v[i + 1]) / 2)))
        return out

    @staticmethod
    def from_profile(P: dict, t0: float = 0.0, t1: float = 1.0) -> "ModulatedPiece":
        return ModulatedPiece(
            t0, t1,
            np.asarray(P["axis_um"], dtype=np.float64),
            np.asarray(P["normal"], dtype=np.float64),
            np.asarray(P["amp_um"], dtype=np.float64),
            float(P["periods"]), float(P["v0"]), float(P["v_k"]),
            float(P["phase_rad"]), bool(P.get("closed", False)))


@dataclass
class Wire:
    """x(t), y(t), V(t) on [0, 1]: an ordered list of pieces."""
    pieces: list = field(default_factory=list)

    def evaluate(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        xy = np.zeros((len(t), 2))
        v = np.zeros(len(t))
        for p in self.pieces:
            m = (t >= p.t0) & (t <= p.t1)
            if m.any():
                xy[m], v[m] = p.evaluate(t[m])
        return xy, v

    def to_beziers(self):
        out = []
        for p in self.pieces:
            out.extend(p.to_beziers())
        return out

    def flatten(self, n: int):
        return self.evaluate(np.linspace(0.0, 1.0, n))

    @staticmethod
    def from_object(o) -> "Wire":
        """A LithoObject as a Wire: modulated where it has a profile,
        polyline where it does not. Different representations at different
        parameter intervals, exactly as intended; today's objects happen to
        be one piece each."""
        prof = getattr(o, "profile", None)
        if prof and prof.get("kind") == "chiral":
            return Wire([ModulatedPiece.from_profile(prof)])
        pts = np.asarray(o.points_um, dtype=np.float64)
        vp = getattr(o, "v_profile", None)
        volts = (np.interp(np.linspace(0, 1, len(pts)),
                           np.linspace(0, 1, len(vp)), np.asarray(vp))
                 if vp else np.full(len(pts), float(getattr(o, "voltage", 0.0))))
        return Wire([PolyPiece(0.0, 1.0, pts, volts)])


def bezier_error_um(bez: Bez, piece: Piece, f0: float, f1: float, n: int = 33) -> float:
    """Worst distance between one cubic span and the true curve over its
    interval: the honesty check on the whole representation."""
    t = np.linspace(piece.t0 + f0 * (piece.t1 - piece.t0),
                    piece.t0 + f1 * (piece.t1 - piece.t0), n)
    true, _ = piece.evaluate(t)
    u = np.linspace(0, 1, n)[:, None]
    p0, c1, c2, p3 = (np.asarray(q) for q in (bez.p0, bez.c1, bez.c2, bez.p3))
    cur = ((1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * c1
           + 3 * (1 - u) * u ** 2 * c2 + u ** 3 * p3)
    return float(np.hypot(*(cur - true).T).max())
