"""Everything is a Timeline client.

A scan is a function. A write is a function. A transit is a function. The
producers here reduce each plan to (x(t), y(t), V(t)) pieces and enqueue
them; TimelinePlayer is the one loop that touches the controller. This is
the pass-engine unification: three executors collapse into one engine that
never learns what it is playing, and a hardware driver has exactly one
job, keep the controller fed from the timeline.

What the producers do NOT do, deliberately:

- z is not on the timeline. The timeline speaks (x, y, V); tip height is a
  servo binding, so lifted transits and approach stay with the pass layer.
  A transit produced here is the in-contact trajectory at zero bias.
- The imaging CHANNELS (height, deflection, current) are synthesis, not
  trajectory. enqueue_scan serves the raster the tip flies; what the
  instrument records while flying it belongs to whoever is listening.

The litho producer keeps the hardest-won property in the file: the pattern
is compiled ONCE into the instrument's NaN-separated wire format and the
timeline plays what `split_segments` walks back out, so the engine runs
the wave the hardware driver will hand the machine. Between strokes the
bias piece is zero by construction, and the timeline's own bridging does
the vendor cycle without being told: stroke end to transit start is a
bias ramp to zero in place, transit end to stroke start ramps the bias
back up at the new position.
"""

from __future__ import annotations

import math

import numpy as np

from ..litho.compiler import compile_pattern, split_segments
from ..litho.curves import PolyPiece
from ..litho.frame import rotate
from ..litho.transit import plan_transit
from .timeline import Segment, Timeline


def enqueue_transit(tl: Timeline, p0_um, p1_um, policy, surface=None,
                    label: str = "transit") -> dict:
    """The move between strokes, as a function: the planned path at the
    planned speed, bias identically zero. A `none`-mode plan wants a jump,
    which is not a function the engine serves; it goes on the timeline at
    the xy slew limit instead, and the plan's own zero duration remains in
    the report as the honest cost of the old behaviour."""
    plan = plan_transit(p0_um, p1_um, policy, surface)
    if not plan["n"]:
        return plan
    dur = float(plan["duration_s"])
    if not dur or not math.isfinite(plan["peak_um_s"]):
        dur = max(float(plan["distance_um"]) / tl.xy_slew_um_s, 1e-3)
    path = np.asarray(plan["path_um"], dtype=np.float64)
    piece = PolyPiece(0.0, 1.0, path, np.zeros(len(path)))
    plan["segment"] = tl.enqueue(piece, dur, label=label)
    return plan


def enqueue_scan(tl: Timeline, config, label: str = "scan") -> list[Segment]:
    """The raster as a function: per line one trace and one retrace at the
    line rate, through the SAME rotate() lithography uses, so the flown
    frame and the written frame cannot disagree about the angle. The
    slow-axis step between lines is not produced at all: it is exactly the
    discontinuity the timeline bridges at the slew limit."""
    n_lines = config.lines
    size = float(config.size_um)
    half = 1.0 / float(config.line_rate_hz) / 2.0
    ux, uy = rotate(1.0, 0.0, config.angle_deg)      # fast unit vector
    vx, vy = rotate(0.0, 1.0, config.angle_deg)      # slow unit vector
    bias = float(config.bias_v) if config.mode == "current" else 0.0
    cx, cy = float(config.x_offset_um), float(config.y_offset_um)
    segs: list[Segment] = []
    for row in range(n_lines):
        slow_idx = row if config.direction == "up" else (n_lines - 1 - row)
        slow = -size / 2 + size * slow_idx / max(n_lines - 1, 1)
        a = np.array([cx + ux * (-size / 2) + vx * slow,
                      cy + uy * (-size / 2) + vy * slow])
        b = np.array([cx + ux * (size / 2) + vx * slow,
                      cy + uy * (size / 2) + vy * slow])
        segs.append(tl.enqueue(
            PolyPiece(0.0, 1.0, np.vstack([a, b]), np.full(2, bias)),
            half, label=f"{label}:row{row}:trace"))
        segs.append(tl.enqueue(
            PolyPiece(0.0, 1.0, np.vstack([b, a]), np.full(2, bias)),
            half, label=f"{label}:row{row}:retrace"))
    return segs


def enqueue_pattern(tl: Timeline, pattern, x_sens: float, y_sens: float,
                    x_offset: float = 0.0, y_offset: float = 0.0,
                    frame=None, policy=None, surface=None,
                    start_um=None) -> dict:
    """A whole write as functions: compile once to the wire format, walk
    the NaN waves back out with the executor's own arithmetic, and enqueue
    each parsed segment at its compiled duration.

    The vendor's per-object cycle (bias to zero, move, land, bias up,
    stroke, bias to zero) is encoded entirely in what gets enqueued. Every
    stroke is preceded by a zero-bias PIN at its first point, so the bias
    only comes up once the tip is there (a pure in-place ramp, courtesy of
    the bridge), and every move between strokes happens at zero bias:
    through the planned transit when a policy is given, through a zero-bias
    pin at the departure point otherwise, so the bridge that serves the
    move has zero bias at both ends. Without the pins a residual stroke
    bias would ramp along the move and write a fading line between objects,
    which is precisely the bug the direct executor guards against.

    Returns the compile report (`waves`) plus `segments_enqueued`, one
    entry per parsed wire segment with its Segment on the timeline.
    """
    waves = compile_pattern(pattern, x_sens, y_sens, x_offset, y_offset,
                            frame=frame)
    parsed = split_segments(waves["x"], waves["y"], waves["v"])
    here = None if start_um is None else np.asarray(start_um, dtype=np.float64)
    out: list[dict] = []

    def pin(p, label):
        return tl.enqueue(PolyPiece(0.0, 1.0, np.vstack([p, p]),
                                    np.zeros(2)), 1e-3, label=label)

    for seg, comp in zip(parsed, waves["segments"]):
        pts_um = np.column_stack([(seg["x"] - x_offset) * x_sens * 1e6,
                                  (seg["y"] - y_offset) * y_sens * 1e6])
        entry = {"name": comp["name"], "object": comp["object"],
                 "voltage": comp["voltage"],
                 "duration_s": comp["duration_s"],
                 "length_um": comp["length_um"]}
        if here is not None:
            if policy is not None:
                entry["transit"] = enqueue_transit(
                    tl, here, pts_um[0], policy, surface,
                    label=f"transit->{comp['name']}")
            else:
                pin(here, f"bias-off->{comp['name']}")
        pin(pts_um[0], f"pin->{comp['name']}")
        entry["segment"] = tl.enqueue(
            PolyPiece(0.0, 1.0, pts_um, np.asarray(seg["v"], dtype=np.float64)),
            float(comp["duration_s"]), label=comp["name"])
        here = pts_um[-1]
        out.append(entry)
    if here is not None:
        pin(here, "bias-off")
    waves["segments_enqueued"] = out
    return waves
