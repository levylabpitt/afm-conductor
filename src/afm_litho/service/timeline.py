"""The engine as a dumb function player.

The architecture, stated in one line: the system ALWAYS sends out
(x(t), y(t), V(t)). There are no modes and no idle state. When nothing is
scheduled, the functions are constants, x(t) = x0, y(t) = y0, V(t) = 0,
which is what "holding position with the bias off" IS. A scan is a
function. A write is a function. A transit is a function. The engine never
knows which one it is playing: it evaluates whatever covers the current
tick and holds the last value when its future runs out.

The robustness comes from what is NOT here. No arm/fire sequencing, no
start/stop edges, no special case for "between things", no way to be in a
wrong mode, and underrun is not an error: a queue that runs dry degrades
to a hold at the last commanded point, which is exactly what a real tip
should do when its controller has nothing newer to say.

Everything upstream reduces to one verb: serve the engine its future, as
curve segments appended to the timeline before the clock reaches them.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from ..hardware import channels


@dataclass
class Segment:
    """One span of the future: t in [t0, t0+dur) plays `piece`.

    `piece` is anything with evaluate(u) -> (xy, v) on its own [t0, t1]
    domain (litho/curves.py pieces qualify verbatim). `label` is for the
    log and the session record, never for behaviour: the engine must not
    care whether it is playing a scan line or a chiral wire.
    """
    t0: float
    dur: float
    piece: object
    label: str = ""

    @property
    def t1(self) -> float:
        return self.t0 + self.dur

    def sample(self, t: np.ndarray):
        u = self.piece.t0 + (self.piece.t1 - self.piece.t0) * \
            np.clip((t - self.t0) / max(self.dur, 1e-12), 0.0, 1.0)
        return self.piece.evaluate(u)


@dataclass
class Hold:
    """The constant function: what idle IS."""
    x: float
    y: float
    v: float = 0.0
    t0: float = 0.0
    t1: float = 1.0

    def evaluate(self, u):
        u = np.atleast_1d(u)
        return (np.column_stack([np.full(len(u), self.x),
                                 np.full(len(u), self.y)]),
                np.full(len(u), self.v))


@dataclass
class RampPiece:
    """The bridge: linear in position and in bias over its span.

    Vtip(t) with steps is not a function the instrument can follow and not
    one the electrochemistry should see. Rather than every producer
    remembering to ramp, the TIMELINE refuses to serve a discontinuous
    future: enqueue() measures the mismatch between where the last segment
    ends and where the next begins, in position and in bias, and inserts
    one of these sized by the slew limits. The engine stays dumb; continuity
    is a property of what it is fed.
    """
    ax: float; ay: float; av: float
    bx: float; by: float; bv: float
    t0: float = 0.0
    t1: float = 1.0

    def evaluate(self, u):
        u = np.atleast_1d(np.asarray(u, dtype=np.float64))
        w = np.clip((u - self.t0) / max(self.t1 - self.t0, 1e-12), 0.0, 1.0)
        xy = np.column_stack([self.ax + (self.bx - self.ax) * w,
                              self.ay + (self.by - self.ay) * w])
        return xy, self.av + (self.bv - self.av) * w


class Timeline:
    """The future, as served to the engine.

    Append-only while playing: segments are queued with `enqueue`, which
    schedules each after the last (or at a requested absolute time), and
    the engine consumes them by asking `sample`. Between and after
    segments the timeline holds the last commanded value, so there is
    never a tick without an answer.
    """

    def __init__(self, x0: float = 0.0, y0: float = 0.0, v0: float = 0.0,
                 v_slew_per_s: float = 100.0, xy_slew_um_s: float = 20.0):
        self._lock = threading.Lock()
        self._segments: list[Segment] = []
        self._hold = Hold(x0, y0, v0)
        self._horizon = 0.0            # end of the scheduled future
        self.v_slew_per_s = float(v_slew_per_s)
        self.xy_slew_um_s = float(xy_slew_um_s)

    # ---------------------------------------------------------- scheduling --

    def enqueue(self, piece, dur: float, t0: float | None = None,
                label: str = "") -> Segment:
        """Serve the engine some future. Defaults to right after whatever is
        already scheduled, so callers can stream a plan segment by segment
        and never think about the clock."""
        with self._lock:
            start = self._horizon if t0 is None else max(t0, self._horizon)
            # continuity is enforced HERE: where does the future currently
            # end, and where does this piece want to begin?
            if self._segments:
                last = self._segments[-1]
                exy, ev = last.sample(np.array([last.t1]))
                px, py, pv = float(exy[0, 0]), float(exy[0, 1]), float(ev[0])
            else:
                px, py, pv = self._hold.x, self._hold.y, self._hold.v
            sxy, sv = piece.evaluate(np.array([piece.t0]))
            nx, ny, nv = float(sxy[0, 0]), float(sxy[0, 1]), float(sv[0])
            dxy = float(np.hypot(nx - px, ny - py))
            dv = abs(nv - pv)
            if dxy > 1e-4 or dv > 0.01:
                bdur = max(dv / self.v_slew_per_s,
                           dxy / self.xy_slew_um_s, 1e-3)
                bridge = Segment(start, bdur,
                                 RampPiece(px, py, pv, nx, ny, nv),
                                 label="bridge")
                self._segments.append(bridge)
                start = bridge.t1
            seg = Segment(start, float(dur), piece, label)
            self._segments.append(seg)
            self._horizon = seg.t1
            return seg

    def hold_at(self, x: float, y: float, v: float = 0.0) -> None:
        """Reset the idle function (e.g. after an approach moved the tip)."""
        with self._lock:
            self._hold = Hold(x, y, v)

    @property
    def horizon(self) -> float:
        return self._horizon

    def future(self, t: float) -> float:
        """Seconds of scheduled future beyond t: what a scheduler watches to
        decide when to serve more."""
        return max(0.0, self._horizon - t)

    # ------------------------------------------------------------ playback --

    def sample(self, t: np.ndarray):
        """(x, y, V) at the given times. Total: every t gets an answer.

        Gaps and exhaustion are not errors; they are the hold. After a
        segment finishes, the hold value becomes that segment's endpoint,
        which is where the tip physically is.
        """
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        xy = np.empty((len(t), 2))
        v = np.empty(len(t))
        with self._lock:
            segs = list(self._segments)
            hold = self._hold

        covered = np.zeros(len(t), dtype=bool)
        for seg in segs:
            m = (t >= seg.t0) & (t < seg.t1)
            if m.any():
                xy[m], v[m] = seg.sample(t[m])
                covered[m] = True
            # past segments update the hold: where the tip ended up
            if t.max() >= seg.t1:
                end_xy, end_v = seg.sample(np.array([seg.t1]))
                hold = Hold(float(end_xy[0, 0]), float(end_xy[0, 1]),
                            float(end_v[0]))
        if (~covered).any():
            # before everything, between segments, or after the future ran
            # out: each uncovered time takes the hold in force just before
            # it, which is the endpoint of the most recent finished segment
            for i in np.flatnonzero(~covered):
                h = self._hold
                for seg in segs:
                    if seg.t1 <= t[i]:
                        end_xy, end_v = seg.sample(np.array([seg.t1]))
                        h = Hold(float(end_xy[0, 0]), float(end_xy[0, 1]),
                                 float(end_v[0]))
                    else:
                        break
                xy[i] = (h.x, h.y)
                v[i] = h.v
        with self._lock:
            self._hold = hold
        return xy, v

    def prune(self, before_t: float) -> int:
        """Drop segments wholly in the past. The hold they established is
        kept, so pruning never changes what the engine outputs."""
        with self._lock:
            n0 = len(self._segments)
            keep = [s for s in self._segments if s.t1 > before_t]
            if len(keep) < n0 and (not keep or keep[0].t0 > before_t):
                past = [s for s in self._segments if s.t1 <= before_t]
                if past:
                    end_xy, end_v = past[-1].sample(np.array([past[-1].t1]))
                    self._hold = Hold(float(end_xy[0, 0]), float(end_xy[0, 1]),
                                      float(end_v[0]))
            self._segments = keep
            return n0 - len(self._segments)


class TimelinePlayer:
    """The dumb engine: ticks the clock, evaluates the timeline, commands
    the controller. It does not know what it is playing.

    Against the twin it writes the XY servo setpoints and the bias exactly
    as LithoRun does, so material deposited through the timeline is
    deposited by the same physics; against hardware the same loop becomes
    the wave-serving thread.
    """

    def __init__(self, arc, timeline: Timeline, clock_hz: int = 50_000):
        self.arc = arc
        self.timeline = timeline
        self.clock_hz = int(clock_hz)
        self.t = 0.0

    def play(self, seconds: float, chunk: int = 500) -> None:
        w = self.arc.world
        # XY loops in setpoint-tracking mode (gain 0), as the capture defines
        # them and as every path executor here arms them. Without this the
        # setpoints land in loops that are not running, and the engine
        # commands a tip that never moves.
        self.arc.ir_SetPISLoop(0, "Always,Never", channels.X_SENSOR,
                               self.arc.sig[channels.X_SENSOR], 0.0, 0.0, 0.0,
                               channels.OUTPUT_X, -10.0, 150.0)
        self.arc.ir_SetPISLoop(1, "Always,Never", channels.Y_SENSOR,
                               self.arc.sig[channels.Y_SENSOR], 0.0, 0.0, 0.0,
                               "Y%Output", -10.0, 150.0)
        ticks = int(seconds * self.clock_hz)
        for start in range(0, ticks, chunk):
            n = min(chunk, ticks - start)
            tt = self.t + (np.arange(n) + 1) / self.clock_hz
            xy, v = self.timeline.sample(tt)
            # command the endpoint of this chunk; the ARC's own dynamics do
            # the tick-by-tick following, as they do for every other path
            self.arc.td_WriteValue("PIDSLoop.0.Setpoint",
                                   float(xy[-1, 0]) * 1e-6 / w.x_lvdt_sens + w.x_lvdt_offset)
            self.arc.td_WriteValue("PIDSLoop.1.Setpoint",
                                   float(xy[-1, 1]) * 1e-6 / w.y_lvdt_sens + w.y_lvdt_offset)
            self.arc.td_WriteValue(channels.OUTPUT_A, float(v[-1]))
            self.arc.step(n)
            self.t = float(tt[-1])
