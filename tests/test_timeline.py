"""The engine as a dumb function player: always (x(t), y(t), V(t))."""

import numpy as np
import pytest

from afm_litho.litho.curves import PolyPiece, Wire
from afm_litho.litho.pens import render_stroke
from afm_litho.service.timeline import Timeline, TimelinePlayer


def _line(x0, y0, x1, y1, v=8.0, n=50):
    pts = np.column_stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)])
    return PolyPiece(0.0, 1.0, pts, np.full(n, float(v)))


def test_idle_is_the_constant_function():
    """Nothing scheduled is not an error and not a mode: it is x=x0, y=y0."""
    tl = Timeline(x0=3.0, y0=-2.0, v0=0.0)
    xy, v = tl.sample(np.array([0.0, 1.0, 100.0]))
    assert np.allclose(xy, [[3.0, -2.0]] * 3)
    assert np.allclose(v, 0.0)


def test_every_tick_gets_an_answer_before_between_and_after():
    tl = Timeline(0.0, 0.0)
    a = tl.enqueue(_line(0, 0, 10, 0), dur=1.0, t0=1.0)
    b = tl.enqueue(_line(10, 0, 10, 5), dur=1.0, t0=3.0)
    mid_a = (a.t0 + a.t1) / 2
    gap = (a.t1 + b.t0) / 2 if b.t0 > a.t1 else None
    t = [0.2, mid_a] + ([gap] if gap else []) + [(b.t0 + b.t1) / 2, b.t1 + 5.0]
    xy, v = tl.sample(np.array(t))
    assert np.allclose(xy[0], [0, 0])              # before: initial hold
    assert xy[1, 0] == pytest.approx(5.0)          # mid first segment
    if gap:
        assert np.allclose(xy[2], [10, 0])         # gap: hold at segment end
    assert xy[-2, 1] == pytest.approx(2.5)         # mid second segment
    assert np.allclose(xy[-1], [10, 5])            # future exhausted: hold


def test_underrun_degrades_to_hold_not_to_failure():
    """The queue running dry is the tip staying put, which is correct."""
    tl = Timeline(0.0, 0.0)
    seg = tl.enqueue(_line(0, 0, 4, 0, v=9.0), dur=0.5)
    xy, v = tl.sample(np.array([(seg.t0 + seg.t1) / 2, seg.t1, seg.t1 + 7.0]))
    assert xy[0, 0] == pytest.approx(2.0)
    assert np.allclose(xy[1:], [[4, 0], [4, 0]])
    assert v[2] == pytest.approx(9.0)              # holds the last commanded bias


def test_segments_stream_back_to_back_by_default():
    tl = Timeline(0.0, 0.0, v0=8.0)                # start at the bias in use
    a = tl.enqueue(_line(0, 0, 1, 0), dur=2.0)
    b = tl.enqueue(_line(1, 0, 1, 1), dur=3.0)
    assert b.t0 == pytest.approx(a.t1)             # no gap between them
    assert tl.horizon == pytest.approx(b.t1)
    assert tl.future(b.t1 - 1.0) == pytest.approx(1.0)


def test_a_chiral_wire_is_just_another_function():
    """The engine must not know what it is playing."""
    o = render_stroke([(0.0, 0.0), (4.0, 0.0)], "chiral", mode="clean")[0]
    piece = Wire.from_object(o).pieces[0]
    tl = Timeline(0.0, 0.0)
    seg = tl.enqueue(piece, dur=2.0, label="chiral")
    t = np.linspace(seg.t0, seg.t1, 4001)
    xy, v = tl.sample(t)
    y = xy[:, 1]
    crossings = int((np.diff(np.sign(y - y.mean())) != 0).sum())
    assert crossings == pytest.approx(2 * 160, rel=0.05)   # 4 um / 25 nm periods
    assert v.max() == pytest.approx(15.0, abs=0.2)
    assert v.min() == pytest.approx(5.0, abs=0.2)


def test_prune_never_changes_the_output():
    tl = Timeline(0.0, 0.0)
    seg = tl.enqueue(_line(0, 0, 6, 0, v=7.0), dur=1.0)
    tl.sample(np.array([seg.t1 + 1.0]))            # establish the hold
    dropped = tl.prune(before_t=seg.t1 + 0.5)
    assert dropped >= 1                            # the segment and its bridge
    xy, v = tl.sample(np.array([seg.t1 + 2.0]))
    assert np.allclose(xy[0], [6, 0])
    assert v[0] == pytest.approx(7.0)


def test_the_player_writes_through_the_same_physics():
    """Play a write through the dumb engine and the sample gains the same
    wire LithoRun would have written: same deposit path, no new physics."""
    from afm_litho.service.passes import TwinInstrument
    from afm_litho.twin.arc import SoftwareARC
    from afm_litho.twin.physics import Sample, World

    s = Sample(height_map=np.zeros((64, 64)), extent=30e-6)
    inst = TwinInstrument(SoftwareARC(World(sample=s)))
    inst.approach(0.2)

    tl = Timeline(0.0, 0.0)
    tl.enqueue(_line(-5, 0, 5, 0, v=9.0, n=200), dur=2.0, label="wire")
    TimelinePlayer(inst.arc, tl).play(seconds=float(tl.horizon) + 0.5)

    ax = np.linspace(-4e-6, 4e-6, 30)
    g_on = max(s.conductance(float(x), 0.0) for x in ax)
    g_off = max(s.conductance(float(x), 3e-6) for x in ax)
    assert g_on > 0.2                          # the wire is there
    assert g_off < 0.02                        # and nowhere else


def test_vtip_is_never_discontinuous():
    """The requirement as a property: whatever is enqueued, the served
    V(t) has no steps. A chiral wire wants to start at 15 V from a 0 V
    hold; the timeline bridges with a slew-limited ramp instead of serving
    the step. The modulation's own slope (vk * omega, thousands of V/s at
    speed) is the intended signal and is NOT a discontinuity: the test is
    the limit across every boundary, not a slope bound.
    """
    o = render_stroke([(2.0, 1.0), (6.0, 1.0)], "chiral", mode="clean")[0]
    piece = Wire.from_object(o).pieces[0]
    tl = Timeline(0.0, 0.0, 0.0, v_slew_per_s=100.0, xy_slew_um_s=20.0)
    tl.enqueue(piece, dur=2.0, label="chiral")
    tl.enqueue(_line(6, 1, 6, 4, v=-8.0), dur=1.0)     # sign flip between strokes
    eps = 1e-6
    with tl._lock:
        bounds = [s.t0 for s in tl._segments] + [s.t1 for s in tl._segments]
    for tb in bounds:
        xy, v = tl.sample(np.array([tb - eps, tb + eps]))
        assert abs(v[1] - v[0]) < 0.05, f"V steps at t={tb}"
        assert np.hypot(*(xy[1] - xy[0])) < 1e-3, f"position steps at t={tb}"
    # and the bridges are real ramps, not teleports: V rate inside any
    # bridge stays at the slew limit
    with tl._lock:
        bridges = [s for s in tl._segments if s.label == "bridge"]
    assert bridges, "the 0 V -> 15 V start must have been bridged"
    for b in bridges:
        t = np.linspace(b.t0, b.t1, 101)
        _, v = tl.sample(t)
        dv = np.abs(np.diff(v)) / (t[1] - t[0])
        assert dv.max() <= tl.v_slew_per_s * 1.05
