"""The clock: hardware mode really waits; testing mode records and skips."""
import asyncio
import queue
import threading
import time

import pytest

from afm_litho import clock


@pytest.fixture(autouse=True)
def _back_to_hardware():
    yield
    clock.use(clock.HARDWARE)


def test_default_mode_is_hardware():
    assert clock.mode() == clock.HARDWARE
    assert clock.waits() == []


def test_hardware_sleep_really_waits():
    t0 = time.monotonic()
    clock.sleep(0.05)
    assert 0.04 <= time.monotonic() - t0 < 1.0


def test_hardware_event_wait_times_out_and_wakes():
    ev = threading.Event()
    t0 = time.monotonic()
    assert clock.wait(ev, 0.05) is False
    assert time.monotonic() - t0 >= 0.04
    threading.Timer(0.02, ev.set).start()
    assert clock.wait(ev, 2.0) is True


def test_hardware_queue_get_and_timeout():
    q = queue.Queue()
    q.put(7)
    assert clock.queue_get(q, 0.05) == 7
    with pytest.raises(queue.Empty):
        clock.queue_get(q, 0.05)


def test_hardware_timer_fires_and_cancels():
    fired = threading.Event()
    clock.timer(0.02, fired.set).start()
    assert fired.wait(2.0)
    never = threading.Event()
    t = clock.timer(0.05, never.set)
    t.start()
    t.cancel()
    assert not never.wait(0.15)


def test_hardware_async_sleep_and_wait_for():
    async def go():
        t0 = time.monotonic()
        await clock.async_sleep(0.05)
        assert time.monotonic() - t0 >= 0.04
        with pytest.raises(TimeoutError):
            await clock.wait_for(asyncio.sleep(5), 0.05)
    asyncio.run(go())


def test_testing_sleep_records_and_advances_virtual_time():
    with clock.using(clock.TESTING):
        t0, w0 = clock.now(), clock.wall()
        real0 = time.monotonic()
        clock.sleep(30.0)
        assert time.monotonic() - real0 < 1.0
        assert clock.now() - t0 == pytest.approx(30.0)
        assert clock.wall() - w0 == pytest.approx(30.0)
        (rec,) = clock.waits()
        assert rec.kind == "sleep" and rec.seconds == 30.0
        assert rec.caller.startswith("test_clock.py:")
    assert clock.mode() == clock.HARDWARE


def test_testing_mode_starts_fresh_each_time():
    with clock.using(clock.TESTING):
        clock.sleep(1.0)
    with clock.using(clock.TESTING):
        assert clock.waits() == []


def test_manual_advance_only_in_testing():
    with pytest.raises(RuntimeError):
        clock.advance(1.0)
    with clock.using(clock.TESTING):
        t0 = clock.now()
        clock.advance(5.0)
        assert clock.now() - t0 == pytest.approx(5.0)


def test_timers_fire_in_virtual_time_in_order():
    fired = []
    with clock.using(clock.TESTING):
        for name, due in (("c", 3.0), ("a", 1.0), ("b", 2.0)):
            clock.timer(due, fired.append, name).start()
        clock.sleep(0.5)
        assert fired == []
        clock.sleep(1.0)
        assert fired == ["a"]
        clock.sleep(10.0)
        assert fired == ["a", "b", "c"]


def test_timer_sees_its_own_due_time_and_can_be_cancelled():
    seen = []
    with clock.using(clock.TESTING):
        start = clock.now()
        clock.timer(2.0, lambda: seen.append(clock.now() - start)).start()
        dropped = clock.timer(1.0, seen.append, "dropped")
        dropped.start()
        dropped.cancel()
        clock.sleep(5.0)
    assert seen == [pytest.approx(2.0)]


def test_event_wait_returns_state_after_virtual_timeout():
    ev = threading.Event()
    with clock.using(clock.TESTING):
        assert clock.wait(ev, 4.0) is False
        clock.timer(2.0, ev.set).start()
        assert clock.wait(ev, 4.0) is True      # the timer fired inside the wait
        assert clock.wait(ev, 4.0) is True      # already set: no time advances
        kinds = [(w.kind, w.seconds) for w in clock.waits() if w.kind == "event"]
    assert kinds == [("event", 4.0)] * 3


def test_queue_get_empty_raises_after_the_real_bound_and_timer_can_fill():
    q = queue.Queue()
    with clock.using(clock.TESTING):
        t0, r0 = clock.now(), time.monotonic()
        with pytest.raises(queue.Empty):
            clock.queue_get(q, 0.3)
        assert time.monotonic() - r0 >= 0.3            # the real bound was waited
        assert clock.now() - t0 == pytest.approx(0.3)
        clock.timer(1.0, q.put, "frame").start()
        assert clock.queue_get(q, 3.0) == "frame"


def test_testing_queue_get_keeps_the_real_bound_for_a_slow_real_sender():
    """A bridge reply from a real thread, later than one grace slice, still arrives."""
    q = queue.Queue()
    threading.Timer(0.15, q.put, [("VALUE", "1.0")]).start()
    with clock.using(clock.TESTING):
        assert clock.queue_get(q, 5.0) == ("VALUE", "1.0")


def test_testing_queue_get_receives_a_frame_a_timer_hands_to_another_thread():
    q = queue.Queue()

    def fire():
        threading.Timer(0.03, q.put, ["frame"]).start()
    with clock.using(clock.TESTING):
        clock.timer(1.0, fire).start()
        assert clock.queue_get(q, 5.0) == "frame"


def test_leaving_the_testing_clock_with_a_pending_timer_raises_and_cleans_up():
    fired = []
    with pytest.raises(RuntimeError, match="test_clock.py"):
        with clock.using(clock.TESTING):
            clock.timer(1.0, fired.append, 1).start()
    assert fired == [] and clock.mode() == clock.HARDWARE
    assert not [t for t in threading.enumerate() if t.name == "clock-timers" and t.is_alive()]


def test_a_failing_block_is_not_masked_by_a_pending_timer_error():
    with pytest.raises(ValueError):
        with clock.using(clock.TESTING):
            clock.timer(1.0, print).start()
            raise ValueError("the test's own failure")
    assert clock.mode() == clock.HARDWARE


def test_async_sleep_and_wait_for_in_testing_mode():
    async def go():
        t0 = clock.now()
        real0 = time.monotonic()
        await clock.async_sleep(20.0)
        assert clock.now() - t0 == pytest.approx(20.0)

        async def quick():
            return 5
        assert await clock.wait_for(quick(), 10.0) == 5

        with pytest.raises(TimeoutError):
            await clock.wait_for(asyncio.Event().wait(), 60.0)
        assert clock.now() - t0 == pytest.approx(80.0)
        assert time.monotonic() - real0 < 5.0
        return [(w.kind, w.seconds) for w in clock.waits()]
    with clock.using(clock.TESTING):
        kinds = asyncio.run(go())
    assert kinds == [("async_sleep", 20.0), ("wait_for", 10.0), ("wait_for", 60.0)]


def test_deadline_loop_terminates_in_virtual_time():
    with clock.using(clock.TESTING):
        real0 = time.monotonic()
        limit = clock.deadline(2.5)
        polls = 0
        while not limit.expired():
            clock.sleep(0.25)
            polls += 1
        assert polls == 10
        assert limit.remaining() == 0.0
        assert time.monotonic() - real0 < 1.0


def test_time_sensitive_mode_waits_really_and_records():
    with clock.using(clock.TESTING_TIME_SENSITIVE):
        t0 = time.monotonic()
        clock.sleep(0.05)
        assert clock.wait(threading.Event(), 0.05) is False
        assert time.monotonic() - t0 >= 0.09
        assert [(w.kind, w.seconds) for w in clock.waits()] == [
            ("sleep", 0.05), ("event", 0.05)]
        assert clock.waits()[0].caller.startswith("test_clock.py:")


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        clock.use("fast")


def test_startup_refuses_test_mode_with_hardware_backend():
    clock.refuse_test_mode_with_hardware("hardware")
    clock.refuse_test_mode_with_hardware("twin")
    for m in (clock.TESTING, clock.TESTING_TIME_SENSITIVE):
        with clock.using(m):
            with pytest.raises(RuntimeError):
                clock.refuse_test_mode_with_hardware("hardware")
            clock.refuse_test_mode_with_hardware("twin")
