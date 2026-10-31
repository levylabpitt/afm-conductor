"""Build the run of a lithography write from what the write is given, without performing it.

Every rule the run states (hover or transit between strokes, ARMFIRE or LITHOCHAIN, the chunk
cut, the setpoint reference, padding of a stroke) is decided by the write's own function, called
here, so the run cannot drift from what `LithographyRun.write_object` sends. Nothing is read
from the instrument and nothing is sent: values only the execution knows (free-air readings,
the live start point of a transit) belong to the journal, not here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from datetime import datetime

import numpy as np

from ..hardware import channels
from ..hardware import bridge_protocol as P
from ..hardware.lithography_run import (
    FIRST_OBJECT_SETTLE_S,
    RETRACT_SETPOINT_V,
    SETTLE_MOVE_S,
    WAVE_CHUNK_POINTS,
    Z_IGAIN,
    WriteSetpoint,
    as_write_setpoint,
    chunk_bounds,
    is_hover_move,
    pad_fired,
    stroke_name,
)
from ..hardware.xy_transit import MOVE_SPEED_UM_S
from ..litho.chunker import is_chained
from ..hardware.hw_approach import Z_MAX, Z_MIN, Z_RETRACT_OUTMIN
from .ids import next_execution_id
from .model import (
    Bias,
    Capture,
    Check,
    Design,
    Engage,
    Execution,
    LithoObject,
    Move,
    Placement,
    Platform,
    Read,
    Recipe,
    Run,
    Settle,
    Setpoint,
    Withdraw,
)

# What the Igor file records on every stroke (afm_litho_run.ipf AFM_ArmAndFireLitho, lines
# 635-653): ZSensor and Deflection on in-bank 0, XSensor and YSensor on in-bank 1, and the
# measured tip bias UserIn2 paired with Lateral on in-bank 2. A refused pair falls back to
# UserIn2 alone (lErr rung), and if that is refused too, nothing is recorded on bank 2.
WRITE_CAPTURE = Capture(
    channels=(channels.Z_SENSOR, channels.DEFLECTION, channels.X_SENSOR, channels.Y_SENSOR),
    pair=(channels.USER_IN2, channels.LATERAL),
    pair_bank=2,
    pair_fallbacks=((channels.USER_IN2,), ()),
)

# The tip is parked at 0 V of Z drive by a verified withdraw (hw_withdraw: OutputMin 0).
WITHDRAWN_HOLD_V = Z_RETRACT_OUTMIN

# The pre-engage read every object takes in free air. The Z-drive and agreement guards live
# in the write's `_free_air_setpoint`, which is where the values read are judged.
FREE_AIR_READ = Read(channels=(channels.DEFLECTION,))

# The write door (`_ensure_write_free_air`) takes the first free-air baseline before anything
# moves, but only when the tip is withdrawn and no fresh contact baseline is on record.
FREE_AIR_BASELINE = "free-air-baseline-if-withdrawn-and-stale"

# Z loop 2 reads Deflection; its output is the platform's z_out alias (Height on the Cypher).
Z_LOOP = 2


class UnsupportedWrite(ValueError):
    """The write does something the run contract cannot state."""


def points_fingerprint(x_um, y_um, v_tip) -> str:
    """SHA-256 of a stroke's fired points: x, y (um) and tip bias (V) as little-endian
    float64, in that order, after padding and before the LVDT and amp-gain conversion (which
    depend on the calibration at the time of the move, unknown when the run is built)."""
    h = hashlib.sha256()
    for a in (x_um, y_um, v_tip):
        h.update(np.ascontiguousarray(a, dtype="<f8").tobytes())
    return h.hexdigest()


def fired_points(meta: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A compiled stroke's x and y (um) and tip bias (V) as `write_object` fires them."""
    path = np.asarray(meta["path_um"], dtype=np.float64)
    return pad_fired(path[:, 0], path[:, 1], meta["v"])


def _litho_pass(index: int, obj, meta: dict) -> LithoObject:
    """The pass for one stroke: whether it chains, where it is cut, how its waves are loaded."""
    x, y, v = fired_points(meta)
    n = len(x)
    bounds = chunk_bounds(n, v, x, y)
    return LithoObject(
        index=index, name=stroke_name(meta["name"]), shape=obj.kind, points=n,
        chunks=len(bounds) - 1, fire=P.LITHOCHAIN if is_chained(n) else P.ARMFIRE,
        ends_at_v=0.0, wave_line_points=WAVE_CHUNK_POINTS,
        points_fingerprint=points_fingerprint(x, y, v), speed_um_s=float(obj.speed_um_s),
        chunk_cuts=tuple(bounds[1:-1]))


def build_write_run(
    *,
    segments: Sequence[dict],
    objects: Sequence,
    setpoint: float | WriteSetpoint,
    amp_gain: float,
    platform: Platform,
    placement: Placement,
    design: Design,
    bias_window_v: tuple[float, float],
    started: datetime,
    taken_ids: Iterable[str] = (),
    first_object_settle_s: float = FIRST_OBJECT_SETTLE_S,
    settle_move_s: float = SETTLE_MOVE_S,
    move_speed_um_s: float = MOVE_SPEED_UM_S,
    retract: float = RETRACT_SETPOINT_V,
) -> Run:
    """The run of a write of `objects`, from `segments` (`compile_pattern(...)["segments"]`,
    one entry per stroke, in firing order).

    `placement`, `design` and `bias_window_v` are facts the caller holds (the frame and its
    source, the design file and its fingerprints, the tip-voltage window). `started` must be
    timezone-aware; `taken_ids` are the execution ids already used.

    A write can only hold one stroke per object here: the contract has one fire per object, so
    a filled region (several strokes of one object, with hover moves between them) raises
    UnsupportedWrite.
    """
    ask = as_write_setpoint(setpoint)
    request = Setpoint("relative" if ask.relative else "absolute", ask.value_v)
    z_out = channels.Z_OUT
    names = [o.name for o in objects]

    body = [Check((FREE_AIR_BASELINE, "xy-calibration-refreshed", "sum-below-ceiling"))]
    last_source = None
    for stroke, meta in enumerate(segments):
        source = meta["object"]
        if source not in names:
            raise UnsupportedWrite(f"stroke {meta['name']!r}: object {source!r} is not in objects")
        index = names.index(source)
        if sum(m["object"] == source for m in segments) > 1:
            raise UnsupportedWrite(
                f"object {source!r} compiles to several strokes (a filled region); the run "
                "contract has one fire per object")
        first = stroke == 0
        where = f"start of object {index}"
        if is_hover_move(first, source, last_source):
            body.append(Move("hover", where, withdraw_first="none", speed_um_s=move_speed_um_s,
                             settle_s=settle_move_s))
        else:
            body.append(Move("transit", where, withdraw_first="verified",
                             hold_v=WITHDRAWN_HOLD_V, speed_um_s=move_speed_um_s,
                             settle_s=settle_move_s))
            if first and first_object_settle_s:
                body.append(Settle(first_object_settle_s, "first object only"))
            body.append(FREE_AIR_READ)
        last_source = source
        body.append(Engage(request, Z_LOOP, channels.DEFLECTION, z_out, (Z_MIN, Z_MAX),
                           sign=1 if Z_IGAIN > 0 else -1, settle_s=settle_move_s))
        body.append(_litho_pass(index, objects[index], meta))
        body.append(Withdraw("back-off", retract, Setpoint("absolute", retract),
                             settle_s=settle_move_s))
    body.append(Withdraw("verified", WITHDRAWN_HOLD_V, why="end of run"))

    recipe = Recipe(
        kind="write", platform=platform, mode="contact", setpoint=request,
        capture=WRITE_CAPTURE, script_gone="park", script_gone_note="write in progress",
        body=tuple(body), placement=placement, design=design,
        bias=Bias(float(amp_gain), bias_window_v))
    execution = Execution(execution_id=next_execution_id("write", started, taken_ids),
                          started=started.isoformat())
    return Run(recipe, execution)


__all__ = ["WRITE_CAPTURE", "UnsupportedWrite", "build_write_run", "fired_points",
           "points_fingerprint"]
