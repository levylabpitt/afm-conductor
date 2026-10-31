"""The twin write is played by a paced player (docs/TWIN_PLAYER_SPEC.md).

`LithoRun` advances along the compiled points on the wall clock scaled by
`twin_speed`, reports about TWIN_FRAME_HZ positions per second of wall time, marks the
sample for the points it has passed, and looks at its stop flag at least every
WAIT_SLICE_S.

Every case runs on a clock that only moves when the player sleeps, so nothing here
depends on how fast the machine is and no case waits in real time.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import pytest

from afm_litho.litho.compiler import LithoObject, Pattern
from afm_litho.litho.transit import TransitPolicy
from afm_litho.service import passes, twin_player
from afm_litho.service.passes import LithoRun, TwinInstrument
from afm_litho.service.twin_player import WAIT_SLICE_S, Pacer
from afm_litho.twin.arc import DT, SoftwareARC
from afm_litho.twin.physics import Sample, World

A_START, A_END = (-4.0, -2.0), (1.0, -2.0)
B_START, B_END = (1.0, 2.0), (6.0, 2.0)
STROKE_S = 2.5                       # 5 um at 2 um/s
REPORT_S = passes.REPORT_S
TICK = 1e-6                          # slack for float sums of wait slices


class _Clock:
    """A clock that moves only when it is slept on. `on_sleep(clock)` runs after each
    sleep, which is where a case lands its stop."""

    def __init__(self):
        self.t = 100.0
        self.sleeps: list[float] = []
        self.on_sleep = None

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s
        if self.on_sleep is not None:
            self.on_sleep(self)


class _Stamped(queue.Queue):
    """A queue that notes the clock time of every message put on it."""

    def __init__(self, clock: _Clock):
        super().__init__()
        self.clock = clock
        self.log: list[tuple[float, dict]] = []

    def put(self, item, *a, **kw):
        self.log.append((self.clock.t, item))
        return super().put(item, *a, **kw)


def _pattern() -> Pattern:
    kw = {"voltage": 6.0, "speed_um_s": 2.0, "interpolation_um": 0.05}
    return Pattern(name="player", objects=[
        LithoObject("a", "path", [A_START, A_END], **kw),
        LithoObject("b", "path", [B_START, B_END], **kw)])


def _inst() -> TwinInstrument:
    sample = Sample(height_map=np.zeros((64, 64)), extent=30e-6)
    return TwinInstrument(SoftwareARC(World(sample=sample)))


def _tip_frame(x_um, y_um):
    return {"type": "tip_pos", "x_um": round(x_um, 3), "y_um": round(y_um, 3)}


def _make(speed, *, tips=True):
    clock = _Clock()
    q = _Stamped(clock)
    # a slow transit, so the move between the objects has many reports of its own
    run = LithoRun(_inst(), _pattern(), q, TransitPolicy(speed_um_s=2.0),
                   tip_frame=_tip_frame if tips else None, twin_speed=speed,
                   clock=clock.now, sleep=clock.sleep)
    return run, clock, q


def _play(speed, **kw):
    run, clock, q = _make(speed, **kw)
    run.run()
    return [m for _, m in q.log], run, clock, q


def _of(msgs, kind):
    return [m for m in msgs if m["type"] == kind]


def _abort_in_stroke(run, clock, after_sleeps: int) -> None:
    """Set the run's abort on the n-th sleep after the first stroke is armed."""
    arc = run.inst.arc
    real_arm = arc.td_xSetInWavePair
    state = {"armed": False, "n": 0}

    def arm(*a, **kw):
        state["armed"] = True
        return real_arm(*a, **kw)

    def on_sleep(c):
        if state["armed"]:
            state["n"] += 1
            if state["n"] == after_sleeps:
                run.abort = True

    arc.td_xSetInWavePair = arm
    clock.on_sleep = on_sleep


# ------------------------------------------------------------------ the clock --

def test_speed_is_a_number_not_below_zero():
    assert twin_player.parse_speed("2.5") == 2.5
    assert twin_player.parse_speed(0) == 0.0
    for bad in (-1, "fast", None, True, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="twin_speed"):
            twin_player.parse_speed(bad)


def test_the_station_speed_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv(twin_player.SPEED_ENV, raising=False)
    assert twin_player.speed_from_env() == 1.0            # real time by default
    monkeypatch.setenv(twin_player.SPEED_ENV, "10")
    assert twin_player.speed_from_env() == 10.0
    monkeypatch.setenv(twin_player.SPEED_ENV, "0")
    assert twin_player.speed_from_env() == 0.0
    monkeypatch.setenv(twin_player.SPEED_ENV, "quick")
    with pytest.raises(ValueError, match=twin_player.SPEED_ENV):
        twin_player.speed_from_env()


def test_a_wait_is_sliced_and_scaled_by_the_speed():
    clock = _Clock()
    pacer = Pacer(4.0, lambda: False, clock.now, clock.sleep)
    assert pacer.dwell(1.0) is True
    assert clock.t - 100.0 == pytest.approx(0.25)         # 1 s simulated at 4x
    assert max(clock.sleeps) <= WAIT_SLICE_S + TICK and len(clock.sleeps) >= 5


def test_a_wait_ends_at_the_first_slice_that_sees_a_stop():
    clock = _Clock()
    stop = {"set": False}
    pacer = Pacer(1.0, lambda: stop["set"], clock.now, clock.sleep)

    def on_sleep(c):
        if len(c.sleeps) == 3:
            stop["set"] = True

    clock.on_sleep = on_sleep
    assert pacer.dwell(60.0) is False
    assert len(clock.sleeps) == 3 and clock.t - 100.0 <= 3 * WAIT_SLICE_S + TICK


def test_unlimited_speed_never_sleeps():
    clock = _Clock()
    pacer = Pacer(0, lambda: False, clock.now, clock.sleep)
    assert pacer.unlimited and pacer.dwell(1e6) is True and pacer.wall(1e6) == 0.0
    assert clock.sleeps == []
    _, _, run_clock, _ = _play(0)
    assert run_clock.sleeps == []


# ------------------------------------------------------------------- duration --

def _stroke_spans(q):
    """Wall time from each object's tip-down (its `litho_transit`) to its done."""
    spans, t_transit = [], None
    for t, m in q.log:
        if m["type"] == "litho_transit":
            t_transit = t
        elif m["type"] == "litho_object_done":
            spans.append(t - t_transit)
    return spans


@pytest.mark.parametrize("speed", [1.0, 10.0])
def test_a_stroke_and_a_transit_take_their_duration_over_the_speed(speed):
    msgs, _, clock, q = _play(speed)
    dwell_s = (passes.TIP_DOWN_TICKS + passes.RELAX_TICKS) * DT
    for start, span in zip(_of(msgs, "litho_object_start"), _stroke_spans(q)):
        assert start["duration_s"] == pytest.approx(STROKE_S, abs=0.05)
        assert span == pytest.approx((start["duration_s"] + dwell_s) / speed, rel=0.01)
    transits = sum(m["seconds"] for m in _of(msgs, "litho_transit"))
    strokes = sum(m["duration_s"] for m in _of(msgs, "litho_object_start"))
    total = clock.t - 100.0
    assert (strokes + transits) / speed <= total * 1.01
    # every other pause of the cycle is short and scaled too
    assert total <= (strokes + transits + 2 * (dwell_s + 0.2)) / speed


def test_no_sleep_is_longer_than_one_slice():
    for speed in (0.25, 1.0, 10.0):
        _, _, clock, _ = _play(speed)
        assert clock.sleeps and max(clock.sleeps) <= WAIT_SLICE_S + TICK


# -------------------------------------------------------------------- reports --

@pytest.mark.parametrize("speed", [1.0, 10.0])
def test_reports_come_at_the_report_rate_of_wall_time_at_any_speed(speed):
    _, _, _, q = _play(speed)
    for name in ("a", "b"):
        times = [t for t, m in q.log
                 if m["type"] == "litho_progress" and m["name"] == name]
        gaps = np.diff(times)
        expect = STROKE_S / speed / REPORT_S
        assert expect * 0.9 <= len(times) <= expect * 1.1 + 1
        assert gaps.max() <= REPORT_S + WAIT_SLICE_S + TICK
        # the stroke's last report may follow the one before it closely; no other does
        assert gaps[:-1].min() >= REPORT_S / 2 - TICK
    # the instrument's telemetry poll reports the tip at 4-10 Hz
    assert 4.0 <= passes.TWIN_FRAME_HZ <= 10.0


def test_a_stroke_report_is_progress_then_the_tip_at_the_same_point():
    msgs, _, _, _ = _play(1.0)
    for i, m in enumerate(msgs):
        if m["type"] != "litho_progress":
            continue
        tip = msgs[i + 1]
        assert tip["type"] == "tip_pos"
        assert (tip["x_um"], tip["y_um"]) == (round(m["x_um"], 3), round(m["y_um"], 3))
        assert m["trail_um"][-1] == [tip["x_um"], tip["y_um"]]


def test_a_transit_reports_tip_positions_along_the_move():
    msgs, _, _, _ = _play(1.0)
    i0 = next(i for i, m in enumerate(msgs) if m["type"] == "litho_object_done")
    i1 = next(i for i, m in enumerate(msgs) if i > i0 and m["type"] == "litho_transit")
    travel = [m for m in msgs[i0 + 1:i1] if m["type"] == "tip_pos"]
    ys = [m["y_um"] for m in travel]
    assert len(travel) >= 15                              # 4 um at 2 um/s, 10 per s
    assert ys == sorted(ys) and ys[0] > A_END[1] and ys[-1] == B_START[1]
    assert not any(m["type"] == "litho_progress" for m in msgs[i0 + 1:i1])


def test_the_reports_that_end_something_are_always_sent():
    for speed in (1.0, 0):
        msgs, _, _, _ = _play(speed)
        types = [m["type"] for m in msgs]
        assert types[0] == "litho_pattern_armed"
        assert msgs[-1] == {"type": "litho_done", "aborted": False}
        for kind in ("litho_object_start", "litho_transit", "litho_object_done"):
            assert types.count(kind) == 2
        # the last progress frame of each object reads frac 1.0 and precedes its done
        for name in ("a", "b"):
            i_done = next(i for i, m in enumerate(msgs)
                          if m["type"] == "litho_object_done" and m["object"] == name)
            mine = [(i, m) for i, m in enumerate(msgs)
                    if m["type"] == "litho_progress" and m["name"] == name]
            assert mine[-1][1]["frac"] == 1.0 and mine[-1][0] < i_done
            fracs = [m["frac"] for _, m in mine]
            assert fracs == sorted(fracs)
        # the tip where each transit and each stroke stopped, in order
        ends = [A_START, A_END, B_START, B_END]
        tips = [(m["x_um"], m["y_um"]) for m in _of(msgs, "tip_pos")]
        seen = [p for p in tips if p in ends]
        assert [p for i, p in enumerate(seen) if i == 0 or p != seen[i - 1]] == ends
        assert tips[-1] == B_END


def test_unlimited_speed_sends_one_report_per_stroke_and_per_transit():
    msgs, _, _, _ = _play(0)
    progress = _of(msgs, "litho_progress")
    assert [p["frac"] for p in progress] == [1.0, 1.0]
    assert [(m["x_um"], m["y_um"]) for m in _of(msgs, "tip_pos")] == [
        A_START, A_END, B_START, B_END]


def test_the_trail_carries_everything_passed_since_the_last_report():
    """The page draws its live trace from `trail_um`: whatever the speed, the trace
    has a position at least every 40 ms of controller time, in path order."""
    slow, _, _, _ = _play(1.0)
    fast, _, _, _ = _play(0)
    for name, (x0, x1) in (("a", (A_START[0], A_END[0])), ("b", (B_START[0], B_END[0]))):
        every = [tuple(p) for m in _of(fast, "litho_progress") if m["name"] == name
                 for p in m["trail_um"]]
        trail = [tuple(p) for m in _of(slow, "litho_progress") if m["name"] == name
                 for p in m["trail_um"]]
        assert set(every) <= set(trail)
        for pts in (every, trail):
            xs = [p[0] for p in pts]
            assert xs == sorted(xs) and xs[-1] == x1 and xs[0] > x0
            assert max(np.diff(xs)) <= 2.0 * 0.04 + 1e-3  # 40 ms at 2 um/s
    # the live current rides along with it
    assert all(m["i_na"] for m in _of(slow, "litho_progress"))


def test_without_a_tip_hook_only_progress_is_reported():
    msgs, _, _, _ = _play(1.0, tips=False)
    assert not _of(msgs, "tip_pos") and len(_of(msgs, "litho_progress")) > 20


# ----------------------------------------------------- what the write leaves --

def test_the_written_record_and_the_mark_do_not_depend_on_the_speed():
    ref_msgs, ref_run, _, _ = _play(0)
    ref_dose = ref_run.inst.world.sample.dose
    assert ref_dose.max() > 1.0
    for speed in (1.0, 10.0):
        msgs, run, _, _ = _play(speed)
        for kind in ("litho_object_done", "litho_transit", "litho_pattern_armed"):
            assert _of(msgs, kind) == _of(ref_msgs, kind)
        np.testing.assert_allclose(run.inst.world.sample.dose, ref_dose,
                                   rtol=1e-4, atol=1e-5)


def test_the_twin_is_left_where_the_last_report_said():
    msgs, run, _, _ = _play(1.0)
    arc, w = run.inst.arc, run.inst.world
    snap = arc.snapshot()
    last = _of(msgs, "tip_pos")[-1]
    assert (snap["x_um"], snap["y_um"]) == (pytest.approx(last["x_um"], abs=1e-3),
                                            pytest.approx(last["y_um"], abs=1e-3))
    assert snap["bias_v"] == 0.0 and snap["engaged"] is True
    assert snap["defl_v"] == pytest.approx(0.2, abs=0.02)   # servo at its setpoint
    assert arc.sig["XSensor"] == w.x_v and arc.sig["YSensor"] == w.y_v
    assert arc.params["PIDSLoop.0.Setpoint"] == pytest.approx(w.x_v)
    assert not any(b.active for b in arc.out_banks.values())
    # the controller model holds that state when someone else steps it afterwards
    arc.step(5000)
    after = arc.snapshot()
    assert (after["x_um"], after["y_um"]) == (pytest.approx(snap["x_um"], abs=1e-3),
                                              pytest.approx(snap["y_um"], abs=1e-3))
    assert after["z_um"] == pytest.approx(snap["z_um"], abs=1e-3) and after["engaged"]
    # and a withdraw from there lifts the tip as it always did
    assert run.inst.withdraw()["engaged"] is False


def test_telemetry_follows_the_tip_during_a_stroke():
    run, clock, _ = _make(1.0)
    seen: list[tuple] = []

    def on_sleep(c):
        snap = run.inst.arc.snapshot()
        seen.append((snap["x_um"], snap["y_um"], snap["bias_v"], snap["engaged"]))

    clock.on_sleep = on_sleep
    run.run()
    writing = [s for s in seen if s[2] != 0.0]
    assert len(writing) > 50 and all(s[3] for s in writing)
    xs_a = [s[0] for s in writing if abs(s[1] - A_START[1]) < 1e-6]
    assert xs_a == sorted(xs_a) and len(set(xs_a)) > 20


def test_the_arc_lock_is_never_held_across_a_wait():
    """`safe_park` steps the twin under the same lock; a wait that held it would
    make a park wait for the write."""
    run, clock, _ = _make(1.0)
    lock = run.inst.arc._lock
    free: list[bool] = []

    def on_sleep(c):
        if len(c.sleeps) % 25:
            return

        def probe():
            ok = lock.acquire(timeout=2.0)
            free.append(ok)
            if ok:
                lock.release()

        t = threading.Thread(target=probe)
        t.start()
        t.join()

    clock.on_sleep = on_sleep
    run.run()
    assert len(free) > 3 and all(free)


# ----------------------------------------------------------------------- stop --

def test_an_abort_mid_stroke_is_seen_within_one_slice_and_marks_what_was_written():
    run, clock, q = _make(1.0)
    _abort_in_stroke(run, clock, after_sleeps=20)         # about 1 s into stroke a
    run.run()
    msgs = [m for _, m in q.log]
    assert run.abort
    assert max(clock.sleeps) <= WAIT_SLICE_S + TICK

    progress = _of(msgs, "litho_progress")
    assert progress and 0.0 < progress[-1]["frac"] < 1.0
    partial = _of(msgs, "litho_object_partial")
    assert len(partial) == 1 and partial[0]["object"] == "a"
    assert partial[0]["frac1"] == pytest.approx(progress[-1]["frac"], abs=1e-3)
    assert not _of(msgs, "litho_object_done")
    assert msgs[-1] == {"type": "litho_done", "aborted": True}

    # the last report is where the stop landed, and the twin is left there, bias off
    x_stop = progress[-1]["x_um"]
    tip = _of(msgs, "tip_pos")[-1]
    snap = run.inst.arc.snapshot()
    assert tip["x_um"] == pytest.approx(x_stop, abs=1e-3)
    assert snap["x_um"] == pytest.approx(x_stop, abs=1e-3)
    assert snap["bias_v"] == 0.0 and snap["engaged"] is True
    assert A_START[0] < x_stop < A_END[0]
    # the partial's path ends there too (every 4th point: within 4 points of it)
    assert partial[0]["path_um"][-1][0] == pytest.approx(x_stop, abs=0.2)

    # the mark reaches the stop and not past it
    sample = run.inst.world.sample
    y = A_START[1] * 1e-6
    assert sample.conductance((x_stop - 0.5) * 1e-6, y) > 0.5
    assert sample.conductance((A_START[0] + 0.3) * 1e-6, y) > 0.5
    assert sample.conductance((x_stop + 0.5) * 1e-6, y) == 0.0


def test_the_time_from_a_stop_to_the_end_of_the_run_is_under_one_slice():
    """At any speed and in any phase: nothing is waited out after a stop."""
    for speed in (0.1, 1.0, 50.0):
        for at_s in (0.05, 0.4, 1.3, 3.0, 4.5):          # simulated seconds in
            run, clock, q = _make(speed)
            t_stop: dict = {}

            def on_sleep(c, run=run, t_stop=t_stop, at=at_s / speed):
                if c.t - 100.0 >= at and not t_stop:
                    run.abort = True
                    t_stop["t"] = c.t

            clock.on_sleep = on_sleep
            run.run()
            assert q.log[-1][1] == {"type": "litho_done", "aborted": True}
            assert q.log[-1][0] - t_stop["t"] <= WAIT_SLICE_S + TICK
            assert run.inst.arc.sig["Output.A"] == 0.0


def test_an_abort_in_transit_reports_where_the_tip_stopped():
    run, clock, q = _make(1.0)

    def on_sleep(c):
        if len(c.sleeps) == 12:                           # in the transit to A_START
            run.abort = True

    clock.on_sleep = on_sleep
    run.run()
    msgs = [m for _, m in q.log]
    assert not _of(msgs, "litho_progress") and not _of(msgs, "litho_object_partial")
    assert msgs[-1] == {"type": "litho_done", "aborted": True}
    tip = _of(msgs, "tip_pos")[-1]
    snap = run.inst.arc.snapshot()
    assert (tip["x_um"], tip["y_um"]) == (pytest.approx(snap["x_um"], abs=1e-3),
                                          pytest.approx(snap["y_um"], abs=1e-3))
    assert (tip["x_um"], tip["y_um"]) != A_START
    assert run.inst.world.sample.dose.max() == 0.0


def test_an_abort_while_the_tip_settles_ends_the_run_without_writing():
    """Tip down, relax, lift and settle are waits in slices, not one long step."""
    run, clock, q = _make(1.0)
    real_transit = run._transit_to
    state = {"after_transit": False}

    def transit(*a, **kw):
        out = real_transit(*a, **kw)
        state["after_transit"] = True
        return out

    def on_sleep(c):
        if state["after_transit"]:
            run.abort = True                 # first slice of the tip-down wait

    run._transit_to = transit
    clock.on_sleep = on_sleep
    run.run()
    msgs = [m for _, m in q.log]
    assert msgs[-1] == {"type": "litho_done", "aborted": True}
    assert not _of(msgs, "litho_progress") and not _of(msgs, "litho_object_partial")
    assert run.inst.world.sample.dose.max() == 0.0
    assert run.inst.arc.sig["Output.A"] == 0.0


def test_a_foreign_step_at_the_fire_does_not_replay_the_stroke():
    """Firing Event.0 starts no bank: a block stepped by anyone else right at the
    fire (here, from inside the fire itself) plays nothing and marks nothing."""
    ref_run, _, _ = _make(0)
    ref_run.run()
    run, _, _ = _make(0)
    arc = run.inst.arc
    real = arc.td_WriteString
    seen: list[bool] = []

    def fire_then_step(name, value):
        out = real(name, value)
        if name == "Event.0":
            seen.append(any(b.active for b in (*arc.out_banks.values(),
                                               *arc.in_banks.values())))
            arc.step(2000)
        return out

    arc.td_WriteString = fire_then_step
    run.run()
    assert seen and not any(seen)
    np.testing.assert_array_equal(run.inst.world.sample.dose,
                                  ref_run.inst.world.sample.dose)
    assert not any(b.active for b in arc.out_banks.values())


def test_a_tip_lifted_mid_stroke_leaves_no_mark_for_the_rest():
    """The mark needs contact, as in the controller model. Something else lifting
    the tip (the Z loop re-armed to retract) while the write goes on stops the mark
    where the lift happened; the stroke itself plays to its end."""
    run, clock, q = _make(1.0)
    # only stroke a: a next object's tip-down would land the tip again
    run.pattern = Pattern(name="one", objects=_pattern().objects[:1])
    state = {"armed": False, "n": 0, "x": None}
    real_arm = run.inst.arc.td_xSetInWavePair

    def arm(*a, **kw):
        state["armed"] = True
        return real_arm(*a, **kw)

    def on_sleep(c):
        if state["armed"] and state["x"] is None:
            state["n"] += 1
            if state["n"] == 20:                         # about 1 s into the stroke
                state["x"] = run.inst.arc.snapshot()["x_um"]
                passes.arm_z_servo(run.inst.arc, -10.0)

    run.inst.arc.td_xSetInWavePair = arm
    clock.on_sleep = on_sleep
    run.run()
    msgs = [m for _, m in q.log]
    assert msgs[-1] == {"type": "litho_done", "aborted": False}
    x_lift, y = state["x"], A_START[1] * 1e-6
    assert A_START[0] < x_lift < A_END[0]
    sample = run.inst.world.sample
    assert sample.conductance((x_lift - 0.5) * 1e-6, y) > 0.5
    assert sample.conductance((x_lift + 0.5) * 1e-6, y) == 0.0
    assert sample.conductance((A_END[0] - 0.2) * 1e-6, y) == 0.0


def test_a_compiled_stroke_plays_for_its_duration_and_marks_to_its_last_path_point():
    """Against the compiler's own output: the stroke lasts `duration_s`, and the 0 V
    sample that closes it is neither marked past nor traced."""
    from afm_litho.litho.compiler import compile_pattern
    run, _, q = _make(1.0)
    run.pattern = Pattern(name="one", objects=_pattern().objects[:1])
    w = run.inst.world
    comp = compile_pattern(run.pattern, w.x_lvdt_sens, w.y_lvdt_sens, w.x_lvdt_offset,
                           w.y_lvdt_offset, frame=run.frame)["segments"][0]
    assert comp["v"][-1] == 0.0 and comp["path_points"] == len(comp["v"]) - 1
    run.run()
    t_arm = next(t for t, m in q.log if m["type"] == "litho_transit")
    t_full = next(t for t, m in q.log
                  if m["type"] == "litho_progress" and m["frac"] == 1.0)
    tip_down_s = passes.TIP_DOWN_TICKS * DT
    assert t_full - t_arm - tip_down_s == pytest.approx(comp["duration_s"], abs=1e-6)
    msgs = [m for _, m in q.log]
    trail = [tuple(p) for m in _of(msgs, "litho_progress") for p in m["trail_um"]]
    assert trail[-1] == A_END and trail.count(A_END) == 1
    done = _of(msgs, "litho_object_done")[0]
    assert tuple(done["path_um"][-1]) == A_END and done["v_written"][-1] == 6.0
    sample, y = w.sample, A_END[1] * 1e-6
    assert sample.conductance((A_END[0] - 0.05) * 1e-6, y) > 0.5
    assert sample.conductance((A_END[0] + 0.3) * 1e-6, y) == 0.0


# ------------------------------------------------------------------- the mark --

def _by_ticks(sample, x_m, y_m, bias_v, dwell_s):
    for x, y, b, t in zip(x_m, y_m, bias_v, dwell_s):
        for _ in range(round(t / DT)):
            sample.deposit(float(x), float(y), float(b), DT)


def _marked_sample() -> Sample:
    s = Sample(height_map=np.zeros((8, 8)), extent=30e-6)
    s.dose[:] = 3.0                                       # something to erase
    s.metal_mask = np.zeros_like(s.dose, dtype=bool)
    s.metal_mask[:, : s.dose_n // 2] = True
    return s


@pytest.mark.parametrize("bias", [6.0, -6.0, 0.5])
def test_the_vectorised_mark_is_the_per_tick_rule(bias):
    rng = np.random.default_rng(5)
    x = np.linspace(-2e-6, 2e-6, 60)
    y = rng.uniform(-1e-7, 1e-7, 60)
    dwell = np.full(60, 40 * DT)
    a, b = _marked_sample(), _marked_sample()
    _by_ticks(a, x, y, np.full(60, bias), dwell)
    twin_player.stamp_dose(b, x, y, np.full(60, bias), dwell)
    np.testing.assert_allclose(b.dose, a.dose, rtol=2e-4, atol=1e-5)
    assert (abs(bias) < 1.0) == bool(np.all(b.dose == 3.0))


def test_a_sign_change_in_one_batch_keeps_its_order():
    x = np.zeros(4)
    bias = np.array([8.0, 8.0, -8.0, -8.0])
    dwell = np.full(4, 200 * DT)
    a = Sample(height_map=np.zeros((8, 8)), extent=30e-6)
    b = Sample(height_map=np.zeros((8, 8)), extent=30e-6)
    _by_ticks(a, x, x, bias, dwell)
    twin_player.stamp_dose(b, x, x, bias, dwell)
    np.testing.assert_allclose(b.dose, a.dose, rtol=2e-4, atol=1e-6)
    assert 0.0 < b.dose.max() < 8.0 * 400 * DT * 8.0


def test_a_fast_stroke_is_marked_without_gaps():
    s = Sample(height_map=np.zeros((8, 8)), extent=30e-6)
    x = np.linspace(-3e-6, 3e-6, 7)                       # 1 um apart; cells far smaller
    y = np.zeros(7)
    twin_player.stamp_path(s, x, y, np.full(7, 9.0), 0.05,
                           twin_player.substeps(s, x, y))
    along = [s.conductance(v, 0.0) for v in np.linspace(-2.9e-6, 2.9e-6, 200)]
    assert min(along) > 0.3 and max(along) < 2 * min(along)


def test_a_closing_sample_that_holds_the_position_at_zero_volts_leaves_no_mark():
    """A compiled stroke may end in a 0 V sample repeated at its last position; the
    out-wave's ramp down to it is not a move of the tip."""
    x = np.array([0.0, 1e-7, 2e-7, 2e-7])
    y = np.zeros(4)
    with_hold = Sample(height_map=np.zeros((8, 8)), extent=30e-6)
    without = Sample(height_map=np.zeros((8, 8)), extent=30e-6)
    twin_player.stamp_path(with_hold, x, y, np.array([8.0, 8.0, 8.0, 0.0]), 0.01, 1)
    twin_player.stamp_path(without, x[:3], y[:3], np.full(3, 8.0), 0.01, 1)
    assert without.dose.max() > 0
    np.testing.assert_array_equal(with_hold.dose, without.dose)


def test_a_stroke_with_a_closing_hold_sample_is_played_but_not_traced():
    """The stroke's points and timing come from the compiled output's own fields:
    `path_points` says which samples are path, `duration_s` how long the stroke
    lasts."""
    run, clock, q = _make(1.0)
    w = run.inst.world
    n_path = 31
    x_um = np.append(np.linspace(0.0, 3.0, n_path), 3.0)
    seg = {"x": x_um * 1e-6 / w.x_lvdt_sens + w.x_lvdt_offset,
           "y": np.full(n_path + 1, w.y_lvdt_offset),
           "v": np.append(np.full(n_path, 8.0), 0.0)}
    tpp = 1000                                            # 20 ms a point
    comp = {"name": "s", "ticks_per_point": tpp, "path_points": n_path,
            "duration_s": (n_path + 1) * tpp * DT}
    passes.arm_z_servo(run.inst.arc, 0.2)
    t0 = clock.t
    assert run._play_stroke(comp, seg) == n_path
    assert clock.t - t0 == pytest.approx(comp["duration_s"], abs=1e-6)
    progress = _of([m for _, m in q.log], "litho_progress")
    assert progress[-1]["frac"] == 1.0
    assert progress[-1]["x_um"] == pytest.approx(3.0)
    trail = [tuple(p) for m in progress for p in m["trail_um"]]
    assert trail.count((3.0, 0.0)) == 1 and len(trail) == len(set(trail))
    ss = run._surface_samples(comp, seg)
    assert ss["x_um"][-1] == 3.0 and ss["x_um"].count(3.0) == 1
    assert run.inst.arc.sig["Output.A"] == 0.0            # the hold sample's bias
