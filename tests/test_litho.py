"""Litho compiler, OASIS round-trip, and write-through-the-ARC tests."""

import queue

import numpy as np
import pytest

from afm_litho.litho.compiler import LithoObject, compile_object, demo_pattern
from afm_litho.service.passes import LithoRun, TwinInstrument
from afm_litho.twin.arc import SoftwareARC
from afm_litho.twin.physics import Sample, World


def test_interpolation_equal_spacing():
    obj = LithoObject("l", "path", [(0, 0), (10, 0), (10, 5)], interpolation_um=0.1)
    comp = compile_object(obj, 6e-6, 6e-6)
    # the path points; the stroke's last sample repeats the last one at 0 V
    p = comp["path_um"][:comp["path_points"]]
    d = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
    # 159 path points over 15 um: the corner at (10, 0) falls between two samples,
    # so the sampled path cuts it, by at most (1 - 1/sqrt(2)) of one step
    step = 15.0 / (len(p) - 1)
    assert 0.0 <= 15.0 - comp["length_um"] <= 0.2929 * step + 1e-9
    assert d.std() / d.mean() < 0.05          # equal arc-length spacing
    # the tip moving, then one tick more: the step to the 0 V sample
    tick_s = comp["ticks_per_point"] / 50_000
    assert abs(comp["duration_s"] - tick_s - 15.0 / obj.speed_um_s) < 0.01


def test_rect_raster():
    obj = LithoObject("r", "rect", [(0, 0), (2, 1)], fill_spacing_um=0.25)
    comp = compile_object(obj, 6e-6, 6e-6)
    assert comp["length_um"] > 2 * 4          # several fill lines
    ys = np.unique(np.round(comp["path_um"][:, 1], 3))
    assert len(ys) >= 5


def test_voltage_ramp():
    obj = LithoObject("v", "path", [(0, 0), (10, 0)], voltage=2.0, voltage_end=8.0)
    comp = compile_object(obj, 6e-6, 6e-6)
    assert comp["v"][0] == pytest.approx(2.0)
    assert comp["v"][-2] == pytest.approx(8.0)     # the last path point
    assert comp["v"][-1] == 0.0                    # the stroke ends at 0 V


def test_oasis_roundtrip(tmp_path):
    oasis = pytest.importorskip("afm_litho.litho.oasis")
    pat = demo_pattern()
    f = tmp_path / "demo.oas"
    oasis.pattern_to_oasis(pat, f)
    back = oasis.oasis_to_pattern(f)
    assert len(back.objects) == len(pat.objects)
    by_name = {o.name: o for o in back.objects}
    for o in pat.objects:
        b = by_name[o.name]
        assert b.voltage == pytest.approx(o.voltage)
        assert b.speed_um_s == pytest.approx(o.speed_um_s)
        a0 = np.asarray(o.points_um)
        b0 = np.asarray(b.points_um)
        assert np.allclose(a0, b0, atol=0.01)


def test_write_and_erase_through_arc():
    sample = Sample(height_map=np.zeros((64, 64)), extent=30e-6)
    arc = SoftwareARC(World(sample=sample))
    inst = TwinInstrument(arc)
    from afm_litho.litho.compiler import Pattern
    pat = Pattern(name="t", objects=[
        LithoObject("wire", "path", [(-3, 0), (3, 0)], voltage=8.0,
                    speed_um_s=6.0, interpolation_um=0.05),
        LithoObject("cut", "path", [(0, 1), (0, -1)], voltage=-8.0,
                    speed_um_s=6.0, interpolation_um=0.05),
    ])
    q = queue.Queue()
    LithoRun(inst, pat, q).run()
    types = []
    while not q.empty():
        types.append(q.get()["type"])
    assert types.count("litho_object_done") == 2
    # wire written, notch erased
    assert sample.conductance(-2e-6, 0.0) > 0.8
    assert sample.conductance(2e-6, 0.0) > 0.8
    assert sample.conductance(0.0, 0.0) < 0.1


def test_object_start_messages_tile_the_objects_path():
    """Each piece announces the span of its design object it covers, so
    the panel can grow the written wire live. Pieces of one object tile
    [0, 1] in order.

    The pieces are the object's real DISCONNECTED strokes (an annulus fills as two
    regions), NOT an arbitrary cut of one stroke: since 2026-09-11 the compiler no
    longer chops a stroke at 8,192 points, so a plain wire is one piece however
    long it is (see `test_a_long_wire_is_ONE_piece_now` below)."""
    sample = Sample(height_map=np.zeros((64, 64)), extent=30e-6)
    inst = TwinInstrument(SoftwareARC(World(sample=sample)))
    from afm_litho.litho.compiler import Pattern
    th = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    pat = Pattern(name="t", objects=[
        LithoObject("ring", "shape",
                    [(6.0 * np.cos(t), 6.0 * np.sin(t)) for t in th],
                    holes_um=[[(3.0 * np.cos(t), 3.0 * np.sin(t)) for t in th]],
                    voltage=8.0, speed_um_s=50.0),
    ])
    q = queue.Queue()
    LithoRun(inst, pat, q).run()
    spans = []
    while not q.empty():
        m = q.get()
        if m["type"] == "litho_object_start":
            assert m["object"] == "ring"
            spans.append((m["object_frac0"], m["object_frac1"]))
    assert len(spans) >= 2                       # two disconnected fill regions
    assert spans[0][0] == 0.0
    assert spans[-1][1] == pytest.approx(1.0, abs=1e-3)
    for (a, b), (c, d) in zip(spans, spans[1:]):
        assert b == pytest.approx(c, abs=1e-3)   # contiguous, in order


def _two_line_pattern():
    from afm_litho.litho.compiler import Pattern
    return Pattern(name="t", objects=[
        LithoObject("wire", "path", [(-3, 0), (3, 0)], voltage=8.0,
                    speed_um_s=6.0, interpolation_um=0.05),
        LithoObject("cut", "path", [(0, 1), (0, -1)], voltage=-8.0,
                    speed_um_s=6.0, interpolation_um=0.05),
    ])


def _grouped(q):
    msgs_by_type: dict[str, list] = {}
    while not q.empty():
        m = q.get()
        msgs_by_type.setdefault(m["type"], []).append(m)
    return msgs_by_type


class _SleepClock:
    """A clock that moves only when the twin's player sleeps on it, so a write at
    `twin_speed` 1 runs its real schedule without taking real time. `on_sleep` runs
    after every sleep: where a test lands its abort."""

    def __init__(self):
        self.t = 0.0
        self.on_sleep = None

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s
        if self.on_sleep is not None:
            self.on_sleep()


def _paced_run(q):
    """A two-line write played in real time on a `_SleepClock`."""
    sample = Sample(height_map=np.zeros((64, 64)), extent=30e-6)
    inst = TwinInstrument(SoftwareARC(World(sample=sample)))
    clock = _SleepClock()
    run = LithoRun(inst, _two_line_pattern(), q, twin_speed=1.0,
                   clock=clock.now, sleep=clock.sleep)
    return run, clock


def _abort_mid_cut(run, clock):
    """Abort on the third wait slice of the SECOND object's stroke.

    `td_xSetInWavePair` is only called in an object's OWN stroke setup (never
    during a transit) -- its 2nd call marks the start of "cut"'s stroke. That
    stroke is 2 um at 6 um/s, a third of a second, so the third 50 ms slice is
    genuinely mid-stroke, not at a boundary."""
    arc = run.inst.arc
    state = {"set_in_wave_calls": 0, "in_cut_stream": False, "slices": 0}
    real_set_in_wave_pair = arc.td_xSetInWavePair

    def counting_set_in_wave_pair(*a, **kw):
        state["set_in_wave_calls"] += 1
        if state["set_in_wave_calls"] == 2:
            state["in_cut_stream"] = True
        return real_set_in_wave_pair(*a, **kw)

    def on_sleep():
        if state["in_cut_stream"]:
            state["slices"] += 1
            if state["slices"] == 3:
                run.abort = True

    arc.td_xSetInWavePair = counting_set_in_wave_pair
    clock.on_sleep = on_sleep


def _abort_in_second_transit(run, clock):
    """Abort on the first wait slice of the transit to the SECOND object -- before
    any of that object's own stroke has happened."""
    orig_transit = run._transit_to
    calls = {"n": 0}

    def abort_now():
        run.abort = True

    def counting_transit(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:                 # transiting to "cut"
            clock.on_sleep = abort_now
        return orig_transit(*a, **kw)

    run._transit_to = counting_transit


def test_abort_mid_stroke_withholds_the_cut_objects_done():
    """An abort that lands inside the SECOND object's stroke must not claim that
    object as done: cleanup (bias to zero) still runs, but `litho_object_done` is
    withheld for it -- matching the hardware worker's break between `write_object`
    and its own done-put on `res.aborted`.
    """
    q = queue.Queue()
    run, clock = _paced_run(q)
    _abort_mid_cut(run, clock)
    run.run()

    msgs_by_type = _grouped(q)
    assert len(msgs_by_type.get("litho_object_start", [])) == 2
    # only the first object's completion is reported
    done = msgs_by_type.get("litho_object_done", [])
    assert len(done) == 1
    assert done[0]["object"] == "wire"
    assert msgs_by_type["litho_done"][0]["aborted"] is True
    # confirm this really was a MID-stroke cut, not a lucky full completion
    cut_progress = [m for m in msgs_by_type.get("litho_progress", [])
                    if m["name"] == "cut"]
    assert cut_progress, "abort landed before any progress was captured for cut"
    assert cut_progress[-1]["frac"] < 1.0
    assert run.inst.arc.sig["Output.A"] == 0.0


def test_abort_mid_stroke_reports_a_partial_object():
    """The mid-stroke cut the previous test withholds `litho_object_done` for must
    not vanish without a trace: `LithoRun.run` reports it on a side message,
    `litho_object_partial`, carrying how far the wire actually got -- POINTS
    PLAYED, not the planned stroke -- from the point the player reached, the same
    index `litho_progress`'s `frac` uses.
    """
    sample = Sample(height_map=np.zeros((64, 64)), extent=30e-6)
    inst = TwinInstrument(SoftwareARC(World(sample=sample)))
    q_full = queue.Queue()
    LithoRun(inst, _two_line_pattern(), q_full).run()
    full_cut_done = next(m for m in _grouped(q_full)["litho_object_done"]
                         if m["object"] == "cut")
    full_points = len(full_cut_done["v_written"])

    q = queue.Queue()
    run, clock = _paced_run(q)
    _abort_mid_cut(run, clock)
    run.run()

    msgs_by_type = _grouped(q)
    partial = msgs_by_type.get("litho_object_partial", [])
    assert len(partial) == 1
    p = partial[0]
    assert p["object"] == "cut" and p["name"] == "cut"
    assert p["object_index"] == 1                 # "cut" is pattern.objects[1]
    assert p["frac0"] == pytest.approx(0.0)
    assert 0.0 < p["frac1"] < 1.0
    assert len(p["path_um"]) == len(p["v_written"])
    assert 0 < len(p["v_written"]) < full_points   # sliced to what was played
    # the partial and the last progress frame agree on how far the stroke got
    cut_progress = [m for m in msgs_by_type["litho_progress"] if m["name"] == "cut"]
    assert p["frac1"] == pytest.approx(cut_progress[-1]["frac"], abs=1e-3)
    # still never claimed as done
    assert not any(m["object"] == "cut"
                   for m in msgs_by_type.get("litho_object_done", []))


def test_abort_during_transit_withholds_the_next_objects_done():
    """An abort that lands while transiting to an object's start -- before a single
    point of it has been written -- must not report that object done either
    (the `if self.abort: break` right after `_transit_to` in `LithoRun.run`)."""
    q = queue.Queue()
    run, clock = _paced_run(q)
    _abort_in_second_transit(run, clock)
    run.run()

    msgs_by_type = _grouped(q)
    # the first object started and completed; the second only started -- its
    # transit is where the abort landed, so it never got a chance to write
    assert len(msgs_by_type.get("litho_object_start", [])) == 2
    done = msgs_by_type.get("litho_object_done", [])
    assert len(done) == 1
    assert done[0]["object"] == "wire"
    assert msgs_by_type["litho_done"][0]["aborted"] is True
    # never even reached the "cut" object's own stroke setup
    assert not any(m["name"] == "cut"
                   for m in msgs_by_type.get("litho_progress", []))


def test_abort_during_transit_reports_no_partial_object():
    """Zero points played is zero points played, whether the abort lands in the
    transit (this case) or between the object's arm-and-fire and its first advance
    -- neither synthesizes a partial."""
    q = queue.Queue()
    run, clock = _paced_run(q)
    _abort_in_second_transit(run, clock)
    run.run()

    msgs_by_type = _grouped(q)
    assert not msgs_by_type.get("litho_object_partial")


# =================== the tip stays DOWN inside a stroke (twin) ===================
#
# The hardware half of this invariant lives in tests/test_litho_chunk_split.py
# (`test_no_lift_inside_a_stroke_over_8192_points`, `test_a_chain_boundary_moves_
# nothing_and_zeroes_the_bias_once`). Both backends have to pin it, or a twin test
# passes where the instrument lifts.

def _traced_run(pattern, extent=30e-6, grid=64):
    """Run a pattern on the twin, returning (events, arc).

    `events` is an ORDERED log of the only three things that matter here:
      ("transit",)                  -- LithoRun._transit_to actually planned a move
      ("z", setpoint)               -- the Z servo was re-armed (a lift or a set-down)
      ("arm", n_points, live_bias, bias_now) -- an Output.A out-wave was armed,
                                    and the bias the tip was at when it happened
      ("bias0",)                    -- Output.A was driven to 0
    The invariant is a statement about the ORDER of those, which is exactly what the
    lift bug violated: a ("bias0",)/("z", -10) pair appeared between two ("arm", ...)
    of the same stroke.
    """
    sample = Sample(height_map=np.zeros((grid, grid)), extent=extent)
    inst = TwinInstrument(SoftwareARC(World(sample=sample)))
    arc = inst.arc
    events = []

    run = LithoRun(inst, pattern, queue.Queue())
    orig_transit = run._transit_to
    orig_set_out = arc.td_xSetOutWave
    orig_write = arc.td_WriteValue
    import afm_litho.service.passes as passes_mod
    orig_arm_z = passes_mod.arm_z_servo

    def transit(target_um, setpoint=0.2):
        plan = orig_transit(target_um, setpoint)
        if plan.get("n"):
            events.append(("transit",))
        return plan

    def set_out(bank, event, channel, wave, interpolation):
        if channel == "Output.A":
            vals = np.asarray(wave, dtype=float)
            live = bool(vals.size and np.nanmax(np.abs(vals)) > 0)
            # the bias the tip is ACTUALLY at as this arm happens. At a chain
            # boundary it must still be the previous chunk's last point -- the
            # out-wave stopped playing but nothing drove the channel to 0.
            events.append(("arm", int(vals.size), live,
                           float(arc.sig.get("Output.A", 0.0))))
        return orig_set_out(bank, event, channel, wave, interpolation)

    def write_value(channel, value):
        if channel == "Output.A" and float(value) == 0.0:
            events.append(("bias0",))
        return orig_write(channel, value)

    def arm_z(a, setpoint, mode="contact"):
        events.append(("z", float(setpoint)))
        return orig_arm_z(a, setpoint, mode)

    run._transit_to = transit
    arc.td_xSetOutWave = set_out
    arc.td_WriteValue = write_value
    passes_mod.arm_z_servo = arm_z
    try:
        run.run()
    finally:
        passes_mod.arm_z_servo = orig_arm_z
    return events, arc


def _arm_runs(events):
    """Maximal runs of CONSECUTIVE live-bias arms in the event log.

    One run is one stroke: several arms only where the stroke was over the out-wave
    cap and had to chain. "Consecutive" is the whole assertion -- any lift, set-down,
    transit or bias-zeroing between two chunks of one stroke splits the run in two,
    which is precisely the bug this pins.
    """
    runs, cur = [], None
    for e in events:
        if e[0] == "arm" and e[2]:
            if cur is None:
                cur = []
                runs.append(cur)
            cur.append(e)
        else:
            cur = None
    return runs


def test_a_long_wire_is_ONE_piece_now():
    """A 24 um wire at 2 nm spacing is 12,000 points -- comfortably past the old
    8,192-point cut, comfortably inside the out-wave cap. One transit, one arm, one
    completion: the tip goes down once and comes up once."""
    from afm_litho.litho.compiler import Pattern
    pat = Pattern(name="t", objects=[
        LithoObject("wire", "path", [(-12, 0), (12, 0)], voltage=8.0,
                    speed_um_s=2000.0, interpolation_um=0.002),
    ])
    events, _arc = _traced_run(pat)
    writes = [e for e in events if e[0] == "arm" and e[2]]
    assert len(writes) == 1 and writes[0][1] > 8192
    assert len([e for e in events if e[0] == "transit"]) == 1
    # the ONE stroke is ONE uninterrupted arm: no lift, no set-down, no bias zeroing
    assert _arm_runs(events) == [writes]
    # bias zeroings: one at the START of the transit (tip about to move, nothing to
    # write) and one at the END of the stroke. Never between the two.
    assert [e[0] for e in events].count("bias0") == 2


def test_an_over_cap_stroke_chains_on_the_twin_without_lifting():
    """Past the out-wave cap the twin cannot arm the stroke in one go, so it chains --
    and the chain must be the ONLY difference. Between the chunks there is no
    ("z", ...) and no ("bias0",): the servo holds and the bias holds at the last
    point, mirroring Igor's AFM_LithoAdvance."""
    from afm_litho.litho.chunker import MAX_OUTWAVE_POINTS

    # a sawtooth: long arc length, small bounding box, so the twin's sample grid
    # stays cheap. ~100k points at the default 0.02 um spacing.
    from afm_litho.litho.compiler import Pattern, compile_pattern
    pts = [(i * 0.2, 20.0 if i % 2 else 0.0) for i in range(101)]
    pat = Pattern(name="t", objects=[
        LithoObject("saw", "path", pts, voltage=8.0, speed_um_s=20000.0)])
    n_total = len(compile_pattern(pat, 7e-6, 7e-6)["segments"][0]["path_um"])
    assert n_total > MAX_OUTWAVE_POINTS
    events, _arc = _traced_run(pat, extent=60e-6)

    writes = [e for e in events if e[0] == "arm" and e[2]]
    assert len(writes) >= 2, "the stroke must have chained"
    assert all(n <= MAX_OUTWAVE_POINTS for _t, n, _l, _b in writes)
    # ONE transit for the whole stroke, and nothing tip-moving between the chunks:
    # every chunk is in ONE uninterrupted run of arms.
    assert len([e for e in events if e[0] == "transit"]) == 1
    assert _arm_runs(events) == [writes]
    # the chunks sum back to the whole stroke -- nothing dropped at a seam
    assert sum(n for _t, n, _l, _b in writes) == n_total
    # THE SEAM ITSELF: every chunk AFTER the first was armed while the bias was
    # still live at the previous chunk's last point. A lift would have zeroed it.
    assert all(abs(b) > 0 for _t, _n, _l, b in writes[1:]),         [b for *_x, b in writes]
    # bias zeroings: the transit's, and one after the LAST chunk. None at a seam --
    # which is what `_arm_runs(events) == [writes]` above says.
    assert [e[0] for e in events].count("bias0") == 2


def test_two_objects_get_exactly_one_lifting_transit_between_them():
    """The lift that SHOULD happen still happens. Two separate wires are separate
    geometry, so the tip comes up, moves and lands once between them -- and only
    there."""
    from afm_litho.litho.compiler import Pattern
    pat = Pattern(name="t", objects=[
        LithoObject("a", "path", [(-8, -4), (8, -4)], voltage=8.0, speed_um_s=2000.0),
        LithoObject("b", "path", [(-8, 4), (8, 4)], voltage=8.0, speed_um_s=2000.0),
    ])
    events, _arc = _traced_run(pat)
    writes = [e for e in events if e[0] == "arm" and e[2]]
    assert len(writes) == 2                       # one arm per wire
    assert len([e for e in events if e[0] == "transit"]) == 2   # approach + between
    # two strokes, each ONE uninterrupted arm -- the lift falls between them, not
    # inside either
    assert _arm_runs(events) == [[writes[0]], [writes[1]]]
    # two per stroke: the transit's, and the end-of-stroke cleanup
    assert [e[0] for e in events].count("bias0") == 4
