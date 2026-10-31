"""The one clock for every wait the product makes.

Hardware mode (the default) calls plain time / asyncio / threading, so the
instrument waits exactly as written. The two test modes are chosen only by test
fixtures through use() / using(); no environment variable or config file selects
them, and refuse_test_mode_with_hardware() stops a service from running a real
instrument on one.

Testing mode records each commanded wait, advances a virtual clock and returns at
once. Testing-time-sensitive mode really waits and also records, for the few tests
that check timing itself (an abort landing during a wait, the deadman, stop latency).

Rules for using testing mode
(a) Background periodic loops (failsafe tick, control watcher, telemetry) never run
    on the virtual clock: they stay real or are stopped in testing-mode tests. A
    looping thread would otherwise spin virtual time away.
(b) A wait whose setter is another real thread is time-sensitive (park verify,
    failsafe done-wait, safe_park's future bound, deadman, stop latency).
(c) A deadline loop converts its clock reads (clock.now) in the same commit as its
    sleep, so one loop never mixes real and virtual time.
(d) Fake-Igor timer callbacks never take product locks.
"""
from __future__ import annotations

import asyncio
import contextlib
import heapq
import itertools
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Awaitable, Callable, NamedTuple

HARDWARE = "hardware"
TESTING = "testing"
TESTING_TIME_SENSITIVE = "testing-time-sensitive"
MODES = (HARDWARE, TESTING, TESTING_TIME_SENSITIVE)

# Testing mode gives a wait this many loop turns (async only), then this much real
# time, for a real setter to act before the wait counts as timed out.
_YIELDS_BEFORE_TIMEOUT = 100
_REAL_GRACE_S = 0.05
# Longest real time an advancing call waits for one timer callback to return.
_FIRE_WAIT_S = 0.5


_PACKAGE = Path(__file__).resolve().parent


class Wait(NamedTuple):
    """One wait the product commanded: what kind, how long, and who asked."""
    kind: str          # sleep, async_sleep, event, queue, wait_for, timer
    seconds: float | None
    caller: str        # "path/to/file.py:line" of the code that asked


def _caller() -> str:
    """File:line of the nearest frame outside this module."""
    frame = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename == __file__:
        frame = frame.f_back
    if frame is None:
        return "?"
    return f"{_short(frame.f_code.co_filename)}:{frame.f_lineno}"


def _short(filename: str) -> str:
    """Path relative to the package for product files, else the file name."""
    try:
        return Path(filename).resolve().relative_to(_PACKAGE).as_posix()
    except ValueError:
        return os.path.basename(filename)


class Deadline:
    """A point in clock time; poll loops ask it instead of comparing time.monotonic."""

    def __init__(self, seconds: float):
        self._end = now() + seconds

    def remaining(self) -> float:
        return max(0.0, self._end - now())

    def expired(self) -> bool:
        return now() >= self._end


class _HardwareClock:
    """Plain time / asyncio / threading, unrecorded."""

    waits: tuple = ()

    def now(self) -> float:
        return time.monotonic()

    def wall(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    async def async_sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    def wait(self, event: threading.Event, timeout: float | None) -> bool:
        return event.wait(timeout)

    def queue_get(self, q: "queue.Queue", timeout: float | None) -> Any:
        return q.get(timeout=timeout)

    async def wait_for(self, awaitable: Awaitable, timeout: float | None) -> Any:
        return await asyncio.wait_for(awaitable, timeout)

    def timer(self, seconds: float, fn: Callable, *args, **kwargs) -> threading.Timer:
        return threading.Timer(seconds, fn, args, kwargs)


class _TimeSensitiveClock(_HardwareClock):
    """Real waits, each also recorded."""

    def __init__(self):
        self.waits: list[Wait] = []

    def _record(self, kind: str, seconds: float | None, caller: str) -> None:
        self.waits.append(Wait(kind, seconds, caller))

    def sleep(self, seconds):
        self._record("sleep", seconds, _caller())
        time.sleep(seconds)

    async def async_sleep(self, seconds):
        self._record("async_sleep", seconds, _caller())
        await asyncio.sleep(seconds)

    def wait(self, event, timeout):
        self._record("event", timeout, _caller())
        return event.wait(timeout)

    def queue_get(self, q, timeout):
        self._record("queue", timeout, _caller())
        return q.get(timeout=timeout)

    async def wait_for(self, awaitable, timeout):
        self._record("wait_for", timeout, _caller())
        return await asyncio.wait_for(awaitable, timeout)

    def timer(self, seconds, fn, *args, **kwargs):
        self._record("timer", seconds, _caller())
        return threading.Timer(seconds, fn, args, kwargs)


class _VirtualTimer:
    """A threading.Timer stand-in that fires when virtual time passes its due time."""

    def __init__(self, clock: "_TestingClock", seconds, fn, args, kwargs, caller="?"):
        self._clock = clock
        self._caller = caller
        self._seconds = seconds
        self._fn, self._args, self._kwargs = fn, args, kwargs
        self.daemon = True          # accepted for threading.Timer compatibility
        self._state = "new"         # new -> pending -> fired | cancelled
        self.finished = threading.Event()   # set once the callback has returned

    def start(self) -> None:
        self._state = "pending"
        self._clock._schedule(self, self._seconds)

    def cancel(self) -> None:
        if self._state != "fired":
            self._state = "cancelled"

    def is_alive(self) -> bool:
        return self._state == "pending"

    def _fire(self) -> None:
        if self._state == "pending":
            self._state = "fired"
            self._fn(*self._args, **self._kwargs)


class _TestingClock:
    """Virtual time: waits are recorded, advance the clock, and return at once.

    Timers fire in due-time order on ONE dedicated thread owned by this clock
    (as threading.Timer calls back off the caller's thread), so a callback never
    runs inside the locks of the thread that advanced time. The advancing call
    hands each due timer over and waits for its callback to return, but for at most
    _FIRE_WAIT_S of real time; a callback blocked longer is left running and time
    moves on. Virtual time never goes backwards, whichever thread advances it.

    Timeout waits (event, queue, wait_for) give a real setter thread _REAL_GRACE_S
    to act, then advance virtual time by the full timeout so a virtual timer that
    would have set the event or filled the queue has fired before the wait looks.
    A wait with no timeout waits in _REAL_GRACE_S slices and, each slice, jumps
    virtual time to the next pending timer, so a virtual timer can still release it.
    """

    def __init__(self):
        self.waits: list[Wait] = []
        self._base_mono = time.monotonic()
        self._base_wall = time.time()
        self._elapsed = 0.0
        self._timers: list = []
        self._order = itertools.count()
        self._lock = threading.RLock()
        self._to_fire: "queue.Queue[_VirtualTimer]" = queue.Queue()
        self._firer: threading.Thread | None = None

    def now(self) -> float:
        return self._base_mono + self._elapsed

    def wall(self) -> float:
        return self._base_wall + self._elapsed

    def advance(self, seconds: float) -> None:
        """Move virtual time forward, firing every timer that comes due on the way."""
        with self._lock:
            target = self._elapsed + max(0.0, seconds)
        while True:
            with self._lock:
                if not self._timers or self._timers[0][0] > target:
                    self._elapsed = max(self._elapsed, target)
                    return
                due, _, timer = heapq.heappop(self._timers)
                self._elapsed = max(self._elapsed, due)
            self._fire_and_wait(timer)

    def _advance_to_next_timer(self, virtual_end: float | None = None) -> None:
        """Move virtual time to the next pending timer, but not past `virtual_end`."""
        with self._lock:
            pending = [t for t in self._timers if t[2]._state == "pending"]
            if not pending:
                return
            gap = min(pending)[0] - self._elapsed
            if virtual_end is not None:
                gap = min(gap, virtual_end - self._elapsed)
        self.advance(gap)

    def close(self) -> list[str]:
        """Cancel pending virtual timers, stop the timer thread, return the cancelled
        timers' callers. Called when the testing clock is switched out."""
        with self._lock:
            pending = [t[2] for t in self._timers if t[2]._state == "pending"]
            for timer in pending:
                timer.cancel()
            self._timers.clear()
            firer = self._firer
        if firer is not None:
            self._to_fire.put(None)
        return [t._caller for t in pending]

    def _schedule(self, timer: _VirtualTimer, seconds: float) -> None:
        with self._lock:
            heapq.heappush(self._timers,
                           (self._elapsed + seconds, next(self._order), timer))

    def _fire_and_wait(self, timer: _VirtualTimer) -> None:
        self._start_firer()
        self._to_fire.put(timer)
        if threading.current_thread() is not self._firer:
            timer.finished.wait(_FIRE_WAIT_S)

    def _start_firer(self) -> None:
        with self._lock:
            if self._firer is None:
                self._firer = threading.Thread(target=self._fire_forever, daemon=True,
                                               name="clock-timers")
                self._firer.start()

    def _fire_forever(self) -> None:
        while True:
            timer = self._to_fire.get()
            if timer is None:
                return
            try:
                timer._fire()
            except Exception:                       # noqa: BLE001
                traceback.print_exc()               # as threading.Timer would
            finally:
                timer.finished.set()

    def _record_and_advance(self, kind, seconds, caller) -> None:
        self.waits.append(Wait(kind, seconds, caller))
        self.advance(seconds or 0.0)

    def sleep(self, seconds):
        self._record_and_advance("sleep", seconds, _caller())

    async def async_sleep(self, seconds):
        self._record_and_advance("async_sleep", seconds, _caller())
        await asyncio.sleep(0)      # let other tasks run, as a real sleep would

    def wait(self, event, timeout):
        """Event state once the timeout has virtually elapsed."""
        self.waits.append(Wait("event", timeout, _caller()))
        if timeout is None:
            while not event.wait(_REAL_GRACE_S):
                self._advance_to_next_timer()
            return True
        if event.wait(min(timeout, _REAL_GRACE_S)):
            return True
        self.advance(timeout)
        return event.is_set()

    def queue_get(self, q, timeout):
        """Next item, waiting up to the REAL bound `timeout` in _REAL_GRACE_S slices.

        A bridge reply from a real thread keeps its real timeout. Each empty slice moves
        virtual time to the next pending virtual timer only (never past the virtual
        timeout), so a fake-Igor DONE still arrives at once. queue.Empty is raised only
        once the real bound has passed; virtual time then moves on to the full timeout."""
        self.waits.append(Wait("queue", timeout, _caller()))
        real_end = None if timeout is None else time.monotonic() + timeout
        with self._lock:
            virtual_end = None if timeout is None else self._elapsed + timeout
        while True:
            slice_s = (_REAL_GRACE_S if real_end is None
                       else max(0.0, min(_REAL_GRACE_S, real_end - time.monotonic())))
            try:
                return q.get(timeout=slice_s)
            except queue.Empty:
                if real_end is not None and time.monotonic() >= real_end:
                    with self._lock:
                        rest = virtual_end - self._elapsed
                    self.advance(rest)
                    raise
                self._advance_to_next_timer(virtual_end)

    async def wait_for(self, awaitable, timeout):
        """Result if the awaitable finishes promptly; else virtual timeout, TimeoutError.

        "Promptly" is _YIELDS_BEFORE_TIMEOUT loop turns plus _REAL_GRACE_S of real time.
        The inner task is cancelled if the caller is cancelled, as asyncio.wait_for does.
        """
        self.waits.append(Wait("wait_for", timeout, _caller()))
        task = asyncio.ensure_future(awaitable)
        try:
            for _ in range(_YIELDS_BEFORE_TIMEOUT):
                if task.done():
                    return task.result()
                await asyncio.sleep(0)
            if timeout is None:
                while not task.done():
                    await asyncio.wait({task}, timeout=_REAL_GRACE_S)
                    self._advance_to_next_timer()
                return task.result()
            await asyncio.wait({task}, timeout=_REAL_GRACE_S)
            if task.done():
                return task.result()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            self.advance(timeout)
            raise TimeoutError
        finally:
            if not task.done():
                task.cancel()

    def timer(self, seconds, fn, *args, **kwargs):
        caller = _caller()
        self.waits.append(Wait("timer", seconds, caller))
        return _VirtualTimer(self, seconds, fn, args, kwargs, caller)


_current: _HardwareClock = _HardwareClock()
_mode = HARDWARE


def _make(mode: str) -> _HardwareClock:
    if mode == HARDWARE:
        return _HardwareClock()
    if mode == TESTING:
        return _TestingClock()
    if mode == TESTING_TIME_SENSITIVE:
        return _TimeSensitiveClock()
    raise ValueError(f"unknown clock mode {mode!r}; expected one of {MODES}")


def use(mode: str) -> str:
    """Switch mode (test fixtures only); a test mode starts with fresh records and time.

    Leaving the testing clock with virtual timers still pending is a test bug (their
    callbacks never ran): the timers are cancelled, the timer thread stopped, and
    RuntimeError names the callers that set them."""
    global _current, _mode
    previous = _mode
    old = _current
    _current, _mode = _make(mode), mode
    pending = old.close() if isinstance(old, _TestingClock) else []
    if pending:
        raise RuntimeError(
            f"left the testing clock with {len(pending)} virtual timer(s) pending "
            f"(never fired, now cancelled): {pending}")
    return previous


@contextlib.contextmanager
def using(mode: str):
    """Run a block in `mode`, then put the previous mode back (a pending-timer error is
    raised only when the block itself did not fail)."""
    previous = use(mode)
    try:
        yield _current
    except BaseException:
        with contextlib.suppress(RuntimeError):
            use(previous)
        raise
    use(previous)


def mode() -> str:
    return _mode


def waits() -> list[Wait]:
    """Waits recorded in the current test mode (empty in hardware mode)."""
    return list(_current.waits)


def advance(seconds: float) -> None:
    """Move the virtual clock by hand (testing mode only)."""
    if _mode != TESTING:
        raise RuntimeError("advance() needs the testing clock")
    _current.advance(seconds)


def refuse_test_mode_with_hardware(backend: str) -> None:
    """Service startup check: a real instrument never runs on a test clock."""
    if backend == "hardware" and _mode != HARDWARE:
        raise RuntimeError(
            f"clock is in {_mode!r} mode but the backend is hardware; "
            "refusing to start with skipped or recorded waits on a real instrument")


def now() -> float:
    return _current.now()


def wall() -> float:
    return _current.wall()


def sleep(seconds: float) -> None:
    _current.sleep(seconds)


async def async_sleep(seconds: float) -> None:
    await _current.async_sleep(seconds)


def wait(event: threading.Event, timeout: float | None) -> bool:
    return _current.wait(event, timeout)


def queue_get(q: "queue.Queue", timeout: float | None) -> Any:
    return _current.queue_get(q, timeout)


async def wait_for(awaitable: Awaitable, timeout: float | None) -> Any:
    return await _current.wait_for(awaitable, timeout)


def timer(seconds: float, fn: Callable, *args, **kwargs):
    return _current.timer(seconds, fn, *args, **kwargs)


def deadline(seconds: float) -> Deadline:
    return Deadline(seconds)
