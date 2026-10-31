"""The twin plugin: positions and ramps are calculated from the commands, not simulated."""

import threading
import time

import numpy as np
import pytest

from afm_litho import clock
from afm_litho.instrument import (Citation, EngageAt, MoveTo, ParkTip, PluginChoice, ReadChannels,
                                  ScanRaster, WithdrawTip, WriteObject)
from afm_litho.instrument import twin as twin_module
from afm_litho.instrument.twin import TwinInstrument, open_twin
from afm_litho.run.model import ScanFrame, Setpoint

RELATIVE = Setpoint("relative", 0.3)       # deflection -0.2 + 0.3 = 0.1 V
RUN = "w-2026-10-26-1200-00"
WITHDRAWN = twin_module.WITHDRAWN_Z_V


@pytest.fixture(autouse=True)
def testing_clock():
    with clock.using("testing"):
        yield


@pytest.fixture
def make():
    """Build twins; every one is closed at the end of the test."""
    made = []

    def build(**options):
        made.append(TwinInstrument(**options))
        return made[-1]
    yield build
    for twin in made:
        twin.close()


@pytest.fixture
def sample_file(tmp_path):
    """A 5 um sample, 100 x 100 pixels (50 nm each), a gentle slope, centred on the origin."""
    path = tmp_path / "sample.height.npz"
    height = np.add.outer(np.arange(100), np.arange(100)) * 0.1
    np.savez(path, height=height, size_um=5.0, x_offset_um=0.0, y_offset_um=0.0,
             angle_deg=0.0, pixels=100, rows=100)
    return path


class Log:
    """A fake telemetry and status sink; `hooks` run on the thread that sent the status."""

    def __init__(self, twin):
        self.frames, self.statuses, self.hooks = [], [], []
        twin.listen(self.frames.append, self._status)

    def _status(self, status):
        self.statuses.append(status)
        for hook in self.hooks:
            hook(status)

    def time_of(self, text_start):
        return next(s.t_s for s in self.statuses if s.text.startswith(text_start))

    def activities(self):
        return [s.activity for s in self.statuses]


TRANSIT_UM_S = 50.0      # the speed these tests give the twin's move to the first point


def wire(name="wire", x=(-2.0, 2.0), y=(0.0, 0.0), v=(1.0, 1.0), speed=1.0, **kw):
    kw.setdefault("transit_speed_um_s", TRANSIT_UM_S)
    return WriteObject(name, x, y, v, speed, RELATIVE, ("Height", "Deflection"), **kw)


def frame5(offset=(0.0, 0.0), size=5.0, lines=5, pixels=5, retrace=True, angle=0):
    return ScanFrame("scan", size, lines, pixels, angle, offset, False, "up", 1.0, retrace)


def scan(twin, frame, engage=True, record=("Height",)):
    return twin.scan(ScanRaster(frame, "contact", RELATIVE, engage, record,
                                transit_speed_um_s=TRANSIT_UM_S))


def test_defaults_to_the_cypher(make):
    assert make().platform == "cypher"
    assert open_twin(PluginChoice("twin")).platform == "cypher"
    assert open_twin(PluginChoice("twin", "mfp3d")).platform == "mfp3d"


def test_position_and_speed_along_a_multi_segment_pass(make):
    twin = make()
    log = Log(twin)
    assert twin.write(wire(x=(0.0, 4.0, 4.0), y=(0.0, 0.0, 3.0), v=(1, 1, 1), speed=2.0)).done
    t = log.time_of("writing")
    assert twin.position_at(t) == pytest.approx((0, 0))
    assert twin.position_at(t + 1) == pytest.approx((2, 0))
    assert twin.speed_at(t + 1) == pytest.approx(2.0)
    assert twin.position_at(t + 2) == pytest.approx((4, 0))
    assert twin.position_at(t + 3.5) == pytest.approx((4, 3))
    assert twin.position_at(t + 2.75) == pytest.approx((4, 1.5))
    assert twin.speed_at(t + 2.75) == pytest.approx(2.0)
    assert twin.speed_at(t + 4) == 0.0


def test_a_move_runs_at_the_move_speed(make):
    twin = make()
    t0 = clock.now()
    assert twin.move(MoveTo(10, 0, 5.0)).done
    assert twin.position_at(t0 + 1) == pytest.approx((5, 0))
    assert twin.speed_at(t0 + 1) == pytest.approx(5.0)
    assert clock.now() - t0 == pytest.approx(2.0)
    assert twin.position_at(clock.now() + 9) == pytest.approx((10, 0))


def test_a_move_from_an_engaged_tip_withdraws_before_the_xy_moves(make):
    twin = make()
    log = Log(twin)
    twin.engage(EngageAt(RELATIVE, "contact"))
    t0 = clock.now()
    assert twin.move(MoveTo(10, 0, 5.0)).done
    assert [a for a in log.activities() if a in ("withdrawing", "moving")] == [
        "withdrawing", "moving"]
    assert twin.position_at(t0 + 2.0) == pytest.approx((0, 0))      # still, while Z comes down
    assert twin.position_at(t0 + 2.5 + 1.0) == pytest.approx((5, 0))


def test_z_ramps_on_engage_and_withdraw_and_deflection_follows(make):
    twin = make()
    t0 = clock.now()
    assert twin.engage(EngageAt(RELATIVE, "contact")).done
    z = lambda t: twin.frame_at(t).z_drive_v            # noqa: E731
    defl = lambda t: twin.frame_at(t).deflection_v      # noqa: E731
    assert z(t0 + 1.0) == pytest.approx(twin_module.ENGAGED_Z_V / 2)
    assert defl(t0 + 1.0) == pytest.approx(-0.2)         # free air until the Z drive has risen
    assert defl(t0 + 2.25) == pytest.approx(-0.05)       # half way through the settle
    assert defl(t0 + 2.5) == pytest.approx(0.1)
    t1 = clock.now()
    result = twin.withdraw(WithdrawTip())
    assert result.done and result.verified is True
    assert z(t1 + 1.25) == pytest.approx(twin_module.ENGAGED_Z_V / 2)
    assert z(t1 + 2.5) == pytest.approx(0.0)
    assert defl(t1 + 2.5) == pytest.approx(-0.2)
    assert clock.now() - t1 == pytest.approx(2.5)


def test_telemetry_values_in_contact_and_in_ac(make):
    twin = make()
    idle = twin.frame_at(clock.now())
    assert (idle.sum_v, idle.deflection_v, idle.z_drive_v, idle.amplitude_v) == (10, -0.2, 0, 0)
    twin.engage(EngageAt(Setpoint("absolute", 0.6), "ac"))
    ac = twin.frame_at(clock.now())
    assert (ac.sum_v, ac.amplitude_v) == (10, pytest.approx(0.6))
    assert ac.deflection_v == pytest.approx(-0.2)
    twin.withdraw(WithdrawTip())
    assert twin.frame_at(clock.now() + 3).amplitude_v == pytest.approx(1.0)


def test_telemetry_rate_follows_the_hardware_rule(make):
    twin = make()
    for activity, hz in (("idle", 4), ("moving", 4), ("engaged", 4), ("scanning", 10)):
        twin._activity = activity
        assert twin.telemetry_period_s() == pytest.approx(1 / hz)
    periods, log = [], Log(twin)
    log.hooks.append(lambda s: s.activity == "writing" and periods.append(
        twin.telemetry_period_s()))
    twin.write(wire())
    twin.write(wire(chained=True))
    assert periods == [pytest.approx(0.25), pytest.approx(0.1)]


def test_scan_polls_at_10_hz_while_scanning(make):
    twin = make()
    seen, log = [], Log(twin)
    log.hooks.append(lambda s: s.activity == "scanning" and seen.append(twin.telemetry_period_s()))
    scan(twin, frame5())
    assert seen == [pytest.approx(0.1)]


def test_telemetry_is_pushed_on_a_real_thread_until_close_and_can_restart(make):
    with clock.using("hardware"):
        twin = make()
        log = Log(twin)
        time.sleep(0.6)
        count = len(log.frames)
        twin.close()
        assert 2 <= count <= 4                      # 4 Hz, idle
        time.sleep(0.4)
        assert len(log.frames) == count
        assert log.frames[0].sum_v == 10 and isinstance(log.frames[0].t_s, float)
        twin.listen(log.frames.append, log._status)         # close() then listen() starts again
        time.sleep(0.1)
        assert len(log.frames) > count


def test_a_written_wire_is_drawn_stays_and_negative_bias_erases_it(make, sample_file):
    twin = make(sample=sample_file)
    before = scan(twin, frame5()).recorded[0].volts
    assert twin.write(wire()).done
    diff = scan(twin, frame5()).recorded[0].volts - before
    assert diff[2] == pytest.approx(twin_module.WIRE_HEIGHT_NM * twin_module.HEIGHT_V_PER_NM)
    assert np.count_nonzero(diff) == 5 and np.count_nonzero(diff[2]) == 5
    assert twin.write(wire(v=(-1.0, -1.0))).done
    assert np.allclose(scan(twin, frame5()).recorded[0].volts, before)


def test_each_segment_takes_the_bias_at_its_start(make, sample_file):
    twin = make(sample=sample_file)
    twin.write(wire(x=(-2.0, 2.0), v=(1.0, 1.0)))
    left, right = (twin.sample.height_nm([x], [0.0])[0] for x in (-1.0, 1.0))
    twin.write(wire(x=(-2.0, 0.0, 2.0), y=(0, 0, 0), v=(-1.0, 1.0, 1.0)))
    assert twin.sample.height_nm([-1.0], [0.0])[0] == pytest.approx(left - 2.0)   # erased
    assert twin.sample.height_nm([1.0], [0.0])[0] == pytest.approx(right)        # kept
    twin.write(wire(x=(-2.0, 0.0, 2.0), y=(0, 0, 0), v=(1.0, -1.0, 1.0)))
    assert twin.sample.height_nm([-1.0], [0.0])[0] == pytest.approx(left)        # drawn again
    assert twin.sample.height_nm([1.0], [0.0])[0] == pytest.approx(right - 2.0)  # erased


def test_scan_returns_the_sample_cropped_line_by_line(make, sample_file):
    twin = make(sample=sample_file)
    t0 = clock.now()
    result = scan(twin, frame5(offset=(0.01, 0.01), size=2.5))
    assert result.done and [r.part for r in result.recorded] == ["trace", "retrace"]
    trace, retrace = (r.volts for r in result.recorded)
    xs = 0.01 - 1.25 + (np.arange(5) + 0.5) * 0.5
    index = (xs + 2.5) // 0.05
    assert trace.shape == (5, 5)
    assert np.allclose(trace, np.add.outer(index, index) * 0.1 * 0.001)
    assert np.allclose(retrace, trace)               # aligned with the trace, as on hardware
    assert clock.now() - t0 >= 5.0                     # five lines at 1 Hz, after engaging
    assert twin.frame_at(clock.now()).z_drive_v == twin_module.ENGAGED_Z_V


def test_an_angled_scan_frame_and_an_angled_sample_rotate(make, sample_file, tmp_path):
    twin = make(sample=sample_file)
    xs = (np.arange(5) + 0.5) * 0.5 - 1.25
    uu, vv = np.meshgrid(xs, xs)
    turned = scan(twin, frame5(size=2.5, angle=90)).recorded[0].volts
    assert np.allclose(turned, twin.sample.height_nm(vv, -uu) * 0.001)    # the frame turned
    assert not np.allclose(turned, scan(twin, frame5(size=2.5)).recorded[0].volts)
    path = tmp_path / "turned.npz"
    np.savez(path, height=np.add.outer(np.arange(100), np.arange(100)) * 0.1, size_um=5.0,
             x_offset_um=0.0, y_offset_um=0.0, angle_deg=90.0)
    other = make(sample=path)
    # the image turned 90 degrees (the vendor's way: litho/frame.py rotate) reads, at a lab
    # point, what the upright one reads at the point turned back
    assert other.sample.height_nm([1.0], [0.5])[0] == pytest.approx(
        twin.sample.height_nm([-0.5], [1.0])[0])


def test_scan_without_retrace_and_refusals(make):
    twin = make()
    assert len(scan(twin, frame5(retrace=False)).recorded) == 1
    refused = scan(twin, frame5(), record=("Height", "Deflection"))
    assert refused.outcome == "refused" and "Deflection" in refused.reason
    twin.withdraw(WithdrawTip())
    assert scan(twin, frame5(), engage=False).outcome == "refused"


def test_a_scan_that_does_not_engage_re_engages_an_engaged_tip(make):
    twin = make()
    log = Log(twin)
    twin.engage(EngageAt(RELATIVE, "contact"))
    assert scan(twin, frame5(), engage=False).done
    tail = log.activities()[2:]
    assert tail == ["withdrawing", "moving", "engaging", "scanning", "engaged"]
    assert twin.frame_at(clock.now()).z_drive_v == twin_module.ENGAGED_Z_V


def test_unmeasured_pixels_refuse_a_read_and_stay_nan_in_a_scan(make, tmp_path):
    path = tmp_path / "holes.npz"
    np.savez(path, height=np.full((100, 100), np.nan), size_um=5.0, x_offset_um=0.0,
             y_offset_um=0.0, angle_deg=0.0)
    twin = make(sample=path)
    twin.engage(EngageAt(RELATIVE, "contact"))
    assert twin.read(ReadChannels(("Height",))).outcome == "refused"
    assert np.isnan(scan(twin, frame5()).recorded[0].volts).all()
    assert twin.write(wire()).done


def test_abort_mid_write_returns_aborted_and_withdraws(make, sample_file):
    twin = make(sample=sample_file)
    log = Log(twin)
    took = []

    def pull_the_plug(status):
        if status.activity == "writing":
            started = time.perf_counter()
            twin.abort("operator")
            took.append(time.perf_counter() - started)
    log.hooks.append(pull_the_plug)
    result = twin.write(wire())
    assert result.outcome == "aborted" and "operator" in result.reason
    assert took[0] < 0.1
    assert twin.frame_at(clock.now() + 2.5).z_drive_v == pytest.approx(WITHDRAWN)
    assert log.statuses[-1].activity == "idle"
    assert scan(twin, frame5()).done                    # the twin takes commands again
    twin.abort("again")                                  # harmless twice
    twin.abort("again")


def test_an_abort_during_engage_is_not_overwritten_by_the_next_step(make):
    twin = make()
    log = Log(twin)
    log.hooks.append(lambda s: s.activity == "engaging" and twin.abort("stop"))
    result = twin.write(wire())
    assert result.outcome == "aborted"
    assert "writing" not in log.activities()
    for later in (0.0, 1.0, 10.0):
        assert twin.frame_at(clock.now() + later).z_drive_v == WITHDRAWN
    assert twin.position_at(clock.now() + 10) == pytest.approx((-2, 0))   # still at the start


def test_abort_freezes_xy_and_draws_only_the_wire_reached(make, sample_file):
    twin = make(sample=sample_file)
    log = Log(twin)
    log.hooks.append(lambda s: s.activity == "writing" and clock.timer(
        2.5, twin.abort, "half way").start())
    result = twin.write(wire(x=(-2.0, 0.0, 2.0), y=(0, 0, 0), speed=1.0, v=(1, 1, 1)))
    assert result.outcome == "aborted"
    stopped = twin.position_at(clock.now())
    assert stopped[0] == pytest.approx(0.5, abs=0.1)
    assert twin.position_at(clock.now() + 10) == pytest.approx(stopped)
    assert twin.speed_at(clock.now() + 1) == 0.0
    drawn = [bool(twin.sample.overlay[row[0], col[0]])
             for row, col, _ in (twin.sample._index([x], [0.0]) for x in (-1.0, 1.0))]
    assert drawn == [True, False]


def test_park_from_another_thread_stops_a_running_write_and_stays_parked(make):
    with clock.using("testing-time-sensitive"):
        twin = make(speed_factor=50.0)
        log = Log(twin)
        parked = []

        def park_soon(status):
            if status.activity == "writing":
                threading.Timer(0.2, lambda: parked.append(twin.park(ParkTip("pull away")))
                                ).start()
        log.hooks.append(park_soon)
        result = twin.write(wire(x=(0.0, 1000.0), y=(0.0, 0.0), speed=1.0))
        assert result.outcome == "aborted" and "park" in result.reason
        deadline = time.perf_counter() + 2
        while not parked and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert parked[0].verified is True and parked[0].critical is False
        assert twin.frame_at(clock.now() + 10).z_drive_v == WITHDRAWN


def test_abort_from_another_thread_stops_a_running_command(make):
    with clock.using("testing-time-sensitive"):
        twin = make(speed_factor=50.0)
        log = Log(twin)
        log.hooks.append(lambda s: s.activity == "writing" and threading.Timer(
            0.2, twin.abort, ["from afar"]).start())
        result = twin.write(wire(x=(0.0, 1000.0), y=(0.0, 0.0), speed=1.0))
        assert result.outcome == "aborted" and "from afar" in result.reason


def test_park_is_one_visible_verified_park(make):
    twin = make()
    log = Log(twin)
    twin.engage(EngageAt(RELATIVE, "contact"))
    result = twin.park(ParkTip("end of session"))
    assert (result.outcome, result.verified, result.critical) == ("done", True, False)
    assert log.activities()[-2:] == ["parking", "parked"]
    assert twin.frame_at(clock.now()).z_drive_v == 0.0
    before = clock.now()
    assert twin.park(ParkTip("again")).verified is True       # already withdrawn: at once
    assert clock.now() == before


def test_write_reports_each_step_with_the_citation_of_its_job(make):
    twin = make()
    log = Log(twin)
    cites = tuple(Citation(RUN, n) for n in range(10, 16))    # Move, Settle, Read, Engage, Fire, Withdraw
    result = twin.write(wire(steps=cites, source="design"))
    assert [(s.activity, s.cite) for s in log.statuses[:5]] == [
        ("moving", cites[0]), ("reading", cites[2]), ("engaging", cites[3]),
        ("writing", cites[4]), ("withdrawing", cites[5])]
    assert "wire" in log.statuses[3].text
    assert [r.part for r in result.recorded] == ["write", "write"]
    assert result.recorded[1].volts == pytest.approx(0.1)          # deflection at the setpoint
    assert result.reading("Deflection") == pytest.approx(-0.2)
    assert result.reading("ZSensor") == 0.0
    hover = twin.write(wire(source="design", steps=cites[:1] + cites[3:]))
    assert any(s.text.startswith("hovering") for s in log.statuses)
    assert hover.readings == ()                                      # no free-air read on a hover


def test_a_write_in_ac_and_unknown_channels_are_refused(make):
    twin = make()
    twin.engage(EngageAt(Setpoint("absolute", 0.6), "ac"))
    assert twin.write(wire()).outcome == "refused"
    assert twin.read(ReadChannels(("XSensor",))).outcome == "refused"
    twin.withdraw(WithdrawTip())
    odd = WriteObject("w", (0, 1), (0, 0), (1, 1), 1.0, RELATIVE, ("Lateral",))
    assert twin.write(odd).outcome == "refused"


def test_read_gives_the_present_values(make):
    twin = make()
    twin.engage(EngageAt(RELATIVE, "contact"))
    result = twin.read(ReadChannels(("Deflection", "DetectorSum", "ZSensor", "Height")))
    assert result.reading("Deflection") == pytest.approx(0.1)
    assert result.reading("DetectorSum") == 10 and result.reading("ZSensor") == 75
    assert result.reading("Height") == 0.0


def test_speed_factor_zero_plays_instantly(make):
    with clock.using("hardware"):
        twin = make(speed_factor=0)
        started = time.perf_counter()
        result = twin.write(wire(x=(0.0, 1000.0), y=(0.0, 0.0), speed=1.0))
        assert result.done and time.perf_counter() - started < 1.0
        assert twin.move(MoveTo(500, 0, 1.0)).done
        assert twin.park(ParkTip("done")).verified is True
        fast = make(speed_factor=1000.0)
        started = time.perf_counter()
        assert fast.engage(EngageAt(RELATIVE, "contact")).done
        assert time.perf_counter() - started < 0.5


# --- what the service needs of the twin (4.3b) ---------------------------------------------


def test_a_write_or_scan_without_a_transit_speed_is_refused(make):
    twin = make()
    assert twin.write(wire(transit_speed_um_s=None)).outcome == "refused"
    refused = twin.scan(ScanRaster(frame5(), "contact", RELATIVE, True, ("Height",)))
    assert refused.outcome == "refused" and "transit_speed_um_s" in refused.reason


def test_the_move_to_the_first_point_goes_at_the_commanded_speed(make):
    twin = make()
    log = Log(twin)
    twin.write(wire(x=(10.0, 11.0), transit_speed_um_s=5.0))      # 10 um from the origin
    moving = log.time_of("moving")
    engaging = log.time_of("engaging")
    assert 1.9 < engaging - moving < 2.5          # 10 um at 5 um/s, not at some speed of its own


def test_a_stop_signalled_before_the_command_begins_aborts_it_untouched(make):
    stop = threading.Event()
    twin = make(should_stop=lambda verb: stop.is_set())
    assert twin.write(wire()).done
    stop.set()
    for result in (twin.write(wire()), scan(twin, frame5())):
        assert result.outcome == "aborted" and "signalled" in result.reason
    assert twin.frame_at(clock.now()).z_drive_v == WITHDRAWN


def test_each_write_step_is_reported_with_what_it_found(make):
    twin = make()
    steps = []
    twin.on_step = lambda kind, make_body: steps.append((kind, make_body()))
    twin.write(wire(source="design"))
    assert [kind for kind, _ in steps] == ["move", "read", "engage", "pass", "withdraw"]
    assert steps[1][1]["free_air_v"] == pytest.approx(-0.2)
    assert steps[3][1]["aborted"] is False
    steps.clear()
    twin.write(wire(source="design"))                              # same object: a hover
    assert [kind for kind, _ in steps] == ["move", "engage", "pass", "withdraw"]
    assert steps[0][1]["how"] == "hover"


def test_an_aborted_pass_reports_how_far_it_got(make):
    twin = make()
    log = Log(twin)
    log.hooks.append(lambda s: s.activity == "writing" and clock.timer(
        2.0, twin.abort, "half way").start())
    result = twin.write(wire(x=(-2.0, 2.0), y=(0, 0), speed=1.0))      # a 4 s pass
    assert result.outcome == "aborted"
    assert result.landed == pytest.approx(0.5, abs=0.05)
    assert len(result.recorded[0].volts) == 1 and len(result.recorded) == 2    # the first point


def test_scan_lines_are_handed_over_as_their_time_passes(make, sample_file):
    twin = make(sample=sample_file)
    seen = []
    twin.on_line = lambda index, trace_nm, retrace_nm: seen.append(
        (index, clock.now(), trace_nm.copy(), retrace_nm))
    t0 = clock.now()
    result = scan(twin, frame5(offset=(0.01, 0.01), size=2.5))
    assert [i for i, *_ in seen] == [0, 1, 2, 3, 4]
    assert seen[0][1] - t0 >= 2.0 + 0.5 + 1.0 - 1e-6        # engage, settle, then one line
    assert np.allclose(np.array([row for _, _, row, _ in seen]) * 0.001, result.recorded[0].volts)
    assert np.allclose(seen[2][3], seen[2][2])
    seen.clear()
    twin.on_line = lambda *args: seen.append(args)
    twin.speed_factor = 0
    scan(twin, frame5(retrace=False))
    assert len(seen) == 5 and seen[0][2] is None


@pytest.mark.parametrize("retrace, per_line", [(False, 1.0), (True, 2.0)])
def test_scan_lines_are_paced_at_the_line_rate(make, sample_file, retrace, per_line):
    """At 1 Hz a line takes 1 s, or 2 s with retrace (the hardware plays twice the points)."""
    twin = make(sample=sample_file)
    seen = []
    twin.on_line = lambda index, *rest: seen.append(clock.now())
    scan(twin, frame5(offset=(0.01, 0.01), size=2.5, retrace=retrace))
    assert np.diff(seen) == pytest.approx([per_line] * 4, abs=0.1)
