"""The stop poll and the fake controller's time go through the clock; the park verify,
watchdog done-wait and future bound stay on real time (set by the watchdog thread).

Testing mode covers waits whose setter is virtual or absent. A wait whose setter is
another real thread (the stop poll, the failsafe done-wait) is marked `time_sensitive` and
runs on the recording real-time clock.
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from afm_litho import clock
from afm_litho.hardware.failsafe_thread import FailsafeWatchdog
from afm_litho.hardware.igor_stub import SafetySimulation
from afm_litho.service.host.continuous import _wait_stop

def _waits(kind: str, where: str) -> list[clock.Wait]:
    return [w for w in clock.waits() if w.kind == kind and w.caller.startswith(where)]


def test_wait_stop_runs_out_its_timeout_in_virtual_time():
    with clock.using(clock.TESTING):
        started = time.monotonic()
        landed = asyncio.run(_wait_stop(threading.Event(), 30.0))
        ticks = _waits("async_sleep", "service/host/continuous.py")
    assert landed is False
    assert time.monotonic() - started < 2.0
    assert sum(w.seconds for w in ticks) == pytest.approx(30.0)


@pytest.mark.time_sensitive
def test_wait_stop_lands_within_a_tick_of_a_real_setter():
    """Stop latency: a stop set by another real thread ends a 30 s wait at once."""
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    started = time.monotonic()
    landed = asyncio.run(_wait_stop(stop, 30.0))
    assert landed is True
    assert time.monotonic() - started < 1.5


@pytest.mark.time_sensitive
def test_failsafe_done_wait_is_woken_by_the_watchdog_thread():
    """The park's done-wait is set by the watchdog's own thread, so it keeps real time."""
    wd = FailsafeWatchdog(lambda reason, com: {"verified": True}).start()
    try:
        req = wd.request("unit")
        assert req.wait(5.0) is True
        assert req.result == {"verified": True}
    finally:
        wd.stop()


def test_fake_controller_real_time_mode_follows_the_clock():
    """SafetySimulation(real_time=True) reads the clock, so a test clock moves it."""
    with clock.using(clock.TESTING):
        sim = SafetySimulation({}, real_time=True)
        clock.advance(1.5)
        sim.sync()
        assert sim.now == pytest.approx(1.5, abs=0.01)
