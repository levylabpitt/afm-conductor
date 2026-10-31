"""The testing clock with real threads: monotonic time, timer thread, real setters."""
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


def test_virtual_time_never_goes_backwards_with_two_sleepers():
    gate = threading.Event()
    with clock.using(clock.TESTING):
        clock.timer(5.0, gate.wait).start()         # a callback that blocks
        t0 = clock.now()
        sleeper = threading.Thread(target=clock.sleep, args=(10.0,))
        sleeper.start()
        time.sleep(0.1)                             # sleeper is inside the 5 s timer
        clock.sleep(15.0)
        after_main = clock.now() - t0
        gate.set()
        sleeper.join(5.0)
        assert clock.now() - t0 >= after_main >= 15.0


def test_timer_callback_runs_off_the_advancing_thread_and_takes_its_lock():
    lock = threading.Lock()
    ran_on = []

    def callback():
        with lock:
            ran_on.append(threading.current_thread())
    with clock.using(clock.TESTING):
        clock.timer(1.0, callback).start()
        with lock:
            clock.sleep(2.0)                        # must not deadlock
        time.sleep(0.2)                             # callback gets the lock afterwards
    assert len(ran_on) == 1
    assert ran_on[0] is not threading.current_thread()


def test_event_set_by_a_real_thread_is_seen():
    done = threading.Event()
    with clock.using(clock.TESTING):
        t0 = clock.now()
        threading.Timer(0.01, done.set).start()
        assert clock.wait(done, 12.0) is True
        assert clock.now() - t0 < 1.0               # no virtual timeout was spent


def test_queue_item_put_by_a_real_thread_is_seen():
    q = queue.Queue()
    with clock.using(clock.TESTING):
        threading.Timer(0.01, q.put, ("frame",)).start()
        assert clock.queue_get(q, 12.0) == "frame"


def test_no_timeout_wait_is_released_by_a_virtual_timer():
    ev = threading.Event()
    q = queue.Queue()
    with clock.using(clock.TESTING):
        clock.timer(3.0, ev.set).start()
        assert clock.wait(ev, None) is True
        clock.timer(3.0, q.put, "x").start()
        assert clock.queue_get(q, None) == "x"


def test_testing_wait_for_cancels_inner_task_when_caller_is_cancelled():
    state = {}

    async def inner():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    async def go():
        caller = asyncio.ensure_future(clock.wait_for(inner(), 60.0))
        await asyncio.sleep(0.01)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        await asyncio.sleep(0.01)
    with clock.using(clock.TESTING):
        asyncio.run(go())
    assert state == {"cancelled": True}


def test_caller_is_relative_to_the_package():
    from afm_litho.hardware import hw_approach
    assert clock._short(hw_approach.__file__) == "hardware/hw_approach.py"


def test_service_refuses_to_start_on_hardware_in_a_test_mode(monkeypatch):
    from fastapi.testclient import TestClient

    from afm_litho.service.app import app
    from afm_litho.service.hub import hub
    monkeypatch.setattr(hub.host, "backend", "hardware")
    with clock.using(clock.TESTING):
        with pytest.raises(RuntimeError, match="hardware"):
            with TestClient(app):
                pass
