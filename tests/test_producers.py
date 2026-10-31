"""Everything is a Timeline client: scans, writes and transits reduce to
(x(t), y(t), V(t)) pieces served to one engine, and the engine never
learns what it is playing."""

import numpy as np
import pytest

from afm_litho.litho.compiler import LithoObject, Pattern
from afm_litho.litho.frame import rotate
from afm_litho.litho.transit import TransitPolicy
from afm_litho.service.passes import ScanConfig, TwinInstrument
from afm_litho.service.producers import (enqueue_pattern, enqueue_scan,
                                         enqueue_transit)
from afm_litho.service.timeline import Timeline, TimelinePlayer
from afm_litho.twin.arc import SoftwareARC
from afm_litho.twin.physics import Sample, World


def _two_stroke_pattern():
    return Pattern(name="t", objects=[
        LithoObject("wire", "path", [(-3, 0), (3, 0)], voltage=8.0,
                    speed_um_s=6.0, interpolation_um=0.05),
        LithoObject("cut", "path", [(0, 1), (0, -1)], voltage=-8.0,
                    speed_um_s=6.0, interpolation_um=0.05),
    ])


# ---------------------------------------------------------------- the scan --

def test_the_scan_producer_flies_the_rotated_raster():
    """The flown frame and the written frame share one rotate(): each line
    of the produced scan lies exactly where scan_line's renderer samples
    the surface, at the vendor's (cos A, -sin A) fast axis."""
    cfg = ScanConfig(size_um=10.0, pixels=8, lines=4, line_rate_hz=2.0,
                     angle_deg=30.0, x_offset_um=1.0, y_offset_um=-2.0,
                     direction="down")
    tl = Timeline(0.0, 0.0)
    segs = enqueue_scan(tl, cfg)
    assert len(segs) == 2 * cfg.lines
    ux, uy = rotate(1.0, 0.0, 30.0)
    vx, vy = rotate(0.0, 1.0, 30.0)
    eps = 1e-6
    for row in range(cfg.lines):
        slow_idx = cfg.lines - 1 - row                 # direction="down"
        slow = -5.0 + 10.0 * slow_idx / (cfg.lines - 1)
        a = np.array([1.0 + ux * -5.0 + vx * slow, -2.0 + uy * -5.0 + vy * slow])
        b = np.array([1.0 + ux * 5.0 + vx * slow, -2.0 + uy * 5.0 + vy * slow])
        trace, retrace = segs[2 * row], segs[2 * row + 1]
        xy, v = tl.sample(np.array([trace.t0 + eps, trace.t1 - eps,
                                    retrace.t1 - eps]))
        assert np.allclose(xy[0], a, atol=1e-3)        # trace departs a
        assert np.allclose(xy[1], b, atol=1e-3)        # trace arrives b
        assert np.allclose(xy[2], a, atol=1e-3)        # retrace returns to a
        assert np.allclose(v, 0.0)                     # contact mode: no bias
        # a line takes half the line period each way
        assert trace.t1 - trace.t0 == pytest.approx(0.25)


def test_a_current_scan_carries_its_bias():
    cfg = ScanConfig(size_um=4.0, pixels=4, lines=2, mode="current", bias_v=1.5)
    tl = Timeline(0.0, 0.0)
    segs = enqueue_scan(tl, cfg)
    _, v = tl.sample(np.array([(s.t0 + s.t1) / 2 for s in segs]))
    assert np.allclose(v, 1.5)


# --------------------------------------------------------------- the write --

def test_the_litho_producer_plays_the_parsed_wire():
    """Compile once, walk the NaN wave back out, play THAT: the stroke on
    the timeline is the wire-format segment, not a parallel geometry."""
    tl = Timeline(0.0, 0.0)
    w = enqueue_pattern(tl, _two_stroke_pattern(), 6e-6, 6e-6)
    names = [e["name"] for e in w["segments_enqueued"]]
    assert names == ["wire", "cut"]
    e0 = w["segments_enqueued"][0]
    seg = e0["segment"]
    xy, v = tl.sample(np.array([(seg.t0 + seg.t1) / 2]))
    assert xy[0, 0] == pytest.approx(0.0, abs=0.1)     # mid-wire
    assert xy[0, 1] == pytest.approx(0.0, abs=0.01)
    assert v[0] == pytest.approx(8.0, abs=0.01)
    assert seg.t1 - seg.t0 == pytest.approx(e0["duration_s"], rel=0.01)


def test_the_tip_never_moves_while_biased_between_strokes():
    """The vendor cycle as an invariant of the served functions: outside a
    stroke the tip is either unbiased or stationary. A residual bias
    ramping along the move would write a fading line between objects."""
    tl = Timeline(0.0, 0.0)
    w = enqueue_pattern(tl, _two_stroke_pattern(), 6e-6, 6e-6)
    strokes = [(e["segment"].t0, e["segment"].t1)
               for e in w["segments_enqueued"]]
    t = np.linspace(0.0, tl.horizon, 8001)
    xy, v = tl.sample(t)
    speed = np.hypot(*np.diff(xy, axis=0).T) / np.diff(t)
    # an interval is "inside" if it touches a stroke at all, so the two
    # samples that straddle a boundary do not read as motion-while-biased
    inside = np.zeros(len(t) - 1, dtype=bool)
    for t0, t1 in strokes:
        inside |= (t[:-1] <= t1 + 1e-9) & (t[1:] >= t0 - 1e-9)
    moving = speed > 0.05                              # um/s
    biased = np.abs(v[:-1]) > 0.6
    offenders = (~inside) & moving & biased
    assert not offenders.any()


def test_transits_go_at_zero_bias_under_the_policy():
    tl = Timeline(0.0, 0.0)
    plan = enqueue_transit(tl, (0.0, 0.0), (5.0, 5.0),
                           TransitPolicy(mode="slew"))
    seg = plan["segment"]
    t = np.linspace(seg.t0, seg.t1, 200)
    xy, v = tl.sample(t)
    assert np.allclose(v, 0.0)
    assert np.allclose(xy[-1], [5.0, 5.0], atol=1e-3)


# --------------------------------------------------- one engine, all three --

def test_one_engine_plays_scan_transit_and_write():
    """The unification, end to end: a raster, a move, and a two-stroke
    write queue onto ONE timeline, one player serves it to the ARC, and
    the sample afterwards carries the wire with the erased notch, exactly
    what the direct executor produces, while the raster deposited
    nothing."""
    s = Sample(height_map=np.zeros((64, 64)), extent=30e-6)
    inst = TwinInstrument(SoftwareARC(World(sample=s)))
    inst.approach(0.2)

    tl = Timeline(0.0, 0.0)
    enqueue_scan(tl, ScanConfig(size_um=8.0, pixels=8, lines=2,
                                line_rate_hz=2.0))
    enqueue_transit(tl, (-4.0, -4.0), (-3.0, 0.0),
                    TransitPolicy(mode="slew"))
    enqueue_pattern(tl, _two_stroke_pattern(), 6e-6, 6e-6,
                    policy=TransitPolicy(mode="slew"), start_um=(-3.0, 0.0))

    # V(t) is continuous across every boundary the three producers made
    eps = 1e-6
    with tl._lock:
        bounds = sorted({s2.t0 for s2 in tl._segments}
                        | {s2.t1 for s2 in tl._segments})
    for tb in bounds:
        if tb - eps < 0:
            continue
        _, vv = tl.sample(np.array([tb - eps, tb + eps]))
        assert abs(vv[1] - vv[0]) < 0.05, f"V steps at t={tb}"

    TimelinePlayer(inst.arc, tl).play(seconds=float(tl.horizon) + 0.2)

    assert s.conductance(-2e-6, 0.0) > 0.8             # the wire is there
    assert s.conductance(2e-6, 0.0) > 0.8
    assert s.conductance(0.0, 0.0) < 0.1               # the notch is cut
    assert s.conductance(0.0, 3e-6) < 0.02             # the raster wrote nothing
