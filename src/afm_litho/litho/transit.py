"""Getting the tip from the end of one stroke to the start of the next.

A compiled object is a continuous trajectory, but a pattern is a list of
them, and the gaps in between were previously nothing at all: the XY
setpoint jumped and the scanner chased it through its own lag. That is
both unphysical (no scanner slews 30 um in a millisecond) and dangerous
(the tip is dragged across the surface at a speed nothing bounds, which is
how tips and samples get destroyed).

Every gap now gets a planned transit: a real trajectory, at a bounded
speed, with the bias held at zero so nothing is written on the way. Four
policies, in increasing order of how much they protect the surface:

  none     jump the setpoint, as before. Kept only so the difference can
           be measured; not a sane choice on a real instrument.
  slew     stay in contact, move in a straight line at transit speed with
           the bias at zero. Cheapest and keeps the z servo in its normal
           regime, but the tip ploughs across everything between the two
           points.
  contour  stay in contact but pre-programme z from the known surface, so
           the servo is asked to follow a ramp it already knows instead of
           discovering a step. Needs a surface model; falls back to slew.
  lift     retract z by a clearance, move, land, settle. Touches nothing in
           between. Costs time and a z cycle per gap, and the landing is
           the one moment worth being careful about.

The speed ceiling applies to every mode, including `none`, because it is a
property of the scanner rather than of the policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TRANSIT_MODES = ("lift", "slew", "contour", "none")


@dataclass
class TransitPolicy:
    mode: str = "lift"
    speed_um_s: float = 20.0        # how fast a transit is meant to run
    max_speed_um_s: float = 40.0    # hard ceiling: the scanner cannot exceed this
    lift_nm: float = 150.0          # z clearance for "lift"
    settle_s: float = 0.03          # dwell after landing, before the bias goes on
    step_um: float = 0.05           # trajectory point spacing
    accel_um_s2: float = 400.0      # ramp in and out instead of stepping to speed

    @property
    def speed(self) -> float:
        return min(self.speed_um_s, self.max_speed_um_s)

    def to_dict(self) -> dict:
        return {"mode": self.mode, "speed_um_s": self.speed_um_s,
                "max_speed_um_s": self.max_speed_um_s, "lift_nm": self.lift_nm,
                "settle_s": self.settle_s}


def _trapezoid(distance: float, speed: float, accel: float, step: float):
    """Points along 0..distance under a trapezoidal speed profile.

    Returns (s, dt) with s monotonic in um and dt the per-point time. The
    profile matters: a constant-speed move still starts with an infinite
    acceleration, and the scanner answers that with a ringing overshoot.
    """
    if distance <= 0:
        return np.zeros(1), np.zeros(1)
    accel = max(accel, 1e-6)
    t_ramp = speed / accel
    s_ramp = 0.5 * accel * t_ramp ** 2
    if 2 * s_ramp >= distance:            # triangular: never reaches speed
        t_ramp = math.sqrt(distance / accel)
        speed = accel * t_ramp
        t_total = 2 * t_ramp
    else:
        t_total = 2 * t_ramp + (distance - 2 * s_ramp) / speed

    n = max(3, int(round(distance / max(step, 1e-6))) + 1)
    t = np.linspace(0.0, t_total, n)
    s = np.where(
        t <= t_ramp,
        0.5 * accel * t ** 2,
        np.where(
            t <= t_total - t_ramp,
            0.5 * accel * t_ramp ** 2 + speed * (t - t_ramp),
            distance - 0.5 * accel * np.maximum(t_total - t, 0.0) ** 2,
        ),
    )
    return np.clip(s, 0.0, distance), t


def plan_transit(p0, p1, policy: TransitPolicy, surface=None) -> dict:
    """Plan the move from p0 to p1 (both um).

    Returns a dict with the trajectory (`path_um`), the bias to apply at
    each point (`v`, all zero by construction), the commanded tip height
    relative to the surface (`z_nm`, zero means in contact), the duration
    and the peak speed actually reached. An empty plan (`n` == 0) means the
    two points coincide and there is nothing to do.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    d = p1 - p0
    dist = float(np.hypot(*d))
    if dist < 1e-9:
        return {"n": 0, "path_um": np.empty((0, 2)), "v": np.empty(0),
                "z_nm": np.empty(0), "duration_s": 0.0, "peak_um_s": 0.0,
                "mode": policy.mode, "distance_um": 0.0}

    mode = policy.mode if policy.mode in TRANSIT_MODES else "lift"

    if mode == "none":
        # Two points: the jump. Still reported honestly so the cost of the
        # old behaviour is visible next to the alternatives.
        path = np.vstack([p0, p1])
        return {"n": 2, "path_um": path, "v": np.zeros(2), "z_nm": np.zeros(2),
                "duration_s": 0.0, "peak_um_s": float("inf"),
                "mode": mode, "distance_um": dist}

    s, t = _trapezoid(dist, policy.speed, policy.accel_um_s2, policy.step_um)
    u = d / dist
    path = p0[None, :] + u[None, :] * s[:, None]
    n = len(s)
    duration = float(t[-1])
    peak = float(np.max(np.diff(s) / np.maximum(np.diff(t), 1e-12))) if n > 1 else 0.0

    if mode == "lift":
        # z profile: up over the first tenth, across, down over the last
        # tenth, then a settle dwell at contact before the next stroke arms.
        z = np.full(n, float(policy.lift_nm))
        rise = max(2, n // 10)
        z[:rise] = np.linspace(0.0, policy.lift_nm, rise)
        z[-rise:] = np.linspace(policy.lift_nm, 0.0, rise)
        n_settle = max(1, int(policy.settle_s / max(duration / n, 1e-6)))
        path = np.vstack([path, np.repeat(p1[None, :], n_settle, axis=0)])
        z = np.concatenate([z, np.zeros(n_settle)])
        duration += policy.settle_s
        n += n_settle
    elif mode == "contour" and surface is not None:
        # Ride the known topography: the servo sees a ramp it was told
        # about rather than a step it has to discover. A small standoff
        # keeps the tip off asperities the map does not resolve.
        try:
            z = np.asarray(surface(path[:, 0], path[:, 1]), dtype=np.float64)
            z = z - float(np.min(z)) + 2.0
        except Exception:
            z = np.zeros(n)
    else:
        z = np.zeros(n)                      # slew: in contact the whole way

    return {"n": n, "path_um": path, "v": np.zeros(n), "z_nm": z,
            "duration_s": duration, "peak_um_s": peak,
            "mode": mode, "distance_um": dist}


def plan_pattern_transits(objects_paths, start_um=(0.0, 0.0),
                          policy: TransitPolicy | None = None,
                          surface=None) -> list[dict]:
    """Plan every gap in a pattern, including the move to the first stroke.

    `objects_paths` is a list of (name, path_um) in write order.
    """
    policy = policy or TransitPolicy()
    out = []
    here = np.asarray(start_um, dtype=np.float64)
    for name, path in objects_paths:
        path = np.asarray(path, dtype=np.float64)
        if len(path) == 0:
            continue
        plan = plan_transit(here, path[0], policy, surface)
        plan["to"] = name
        out.append(plan)
        here = path[-1]
    return out


def transit_report(plans: list[dict]) -> dict:
    """What the chosen policy costs, so the choice can be made on numbers."""
    plans = [p for p in plans if p.get("n")]
    if not plans:
        return {"moves": 0, "distance_um": 0.0, "seconds": 0.0, "peak_um_s": 0.0}
    peak = max(p["peak_um_s"] for p in plans)
    return {
        "moves": len(plans),
        "distance_um": round(sum(p["distance_um"] for p in plans), 2),
        "seconds": round(sum(p["duration_s"] for p in plans), 2),
        # None, not inf: an unbounded jump has no speed to report, and JSON
        # cannot carry infinity anyway
        "peak_um_s": None if math.isinf(peak) else round(peak, 1),
    }
