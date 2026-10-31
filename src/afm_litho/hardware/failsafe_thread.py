"""The dedicated fail-safe WATCHDOG THREAD: the one thread allowed to fire the
out-of-band retract over COM (docs/FLEX_INTEGRATION_PLAN.md §B.2/B.3,
docs/SAFETY_KERNEL_PLAN.md §0, docs/ABORT_SEQUENCE.md).

Why not a pool thread:
  1. COM has thread affinity. `LazyIgor` uses `GetActiveObject`, which can raise on a
     thread that never called `CoInitialize()`; `send_safe_park` would then silently
     fall through to TCP, the transport a comms failure has already broken.
  2. A pool thread is not guaranteed: the default executor is shared, so the retract
     could queue behind unrelated work, and it must not depend on the asyncio loop.

So: ONE non-pool daemon thread that CoInitialize()s once (guarded: no pywin32 off the
instrument PC), OWNS a `LazyIgor` for life, waits on an Event or a request queue, runs
the retract SYNCHRONOUSLY, and hands the outcome to the loop best-effort, never
blocking the retract.

Generic by design: the retract sequence is injected as `work` by
`service/host/safe_park.py`, so `hardware/` never imports `service/`.
"""
from __future__ import annotations

import queue
import threading
import time
from concurrent import futures

# Per-wait block before re-checking the stop flag (bounds shutdown latency).
_TICK_S = 0.1


class ParkRequest:
    """One queued retract request. The watchdog fills `result` BEFORE setting `done`,
    so a woken waiter always sees a complete outcome."""

    __slots__ = ("at", "done", "future", "reason", "report", "result")

    def __init__(self, reason: str, *, report: bool = False):
        self.reason = str(reason)
        self.report = bool(report)
        self.done = threading.Event()
        # For asyncio callers (`wrap_future`), so the wait holds no executor thread.
        self.future: futures.Future[dict] = futures.Future()
        self.result: dict | None = None
        self.at = time.time()

    def wait(self, timeout: float | None = None) -> bool:
        return self.done.wait(timeout)       # real time: set by the watchdog thread


class FailsafeWatchdog:
    """The single COM-owning fail-safe thread.

    Parameters
    ----------
    work
        ``work(reason, com_igor) -> dict``: the SYNCHRONOUS retract, run on this
        thread. It must never raise (a fail-safe that raises did not happen); anything
        that escapes becomes a CRITICAL outcome.
    report
        Optional ``report(reason, result)`` for requests with no awaiting caller.
        Failures are swallowed: reporting must never affect the retract.
    trigger
        The bare Event any thread may set to demand a park (`hub._safe_park_now`).
    reason_getter
        Reads a bare-trigger park's reason (`hub._safe_park_reason`, written BEFORE
        the Event is set).
    """

    def __init__(self, work, *, report=None, trigger: threading.Event | None = None,
                 reason_getter=None, log=None, name: str = "afm-failsafe"):
        self._work = work
        self._report = report
        self.trigger = trigger if trigger is not None else threading.Event()
        self._reason_getter = reason_getter or (lambda: None)
        self._log = log or (lambda msg: None)
        self._name = name
        self._q: queue.Queue[ParkRequest] = queue.Queue()
        # The thread's OWN wake for `request()`, deliberately NOT the shared `trigger`:
        # sharing it lets a request queued between clear and drain leave `trigger` set
        # with an empty queue, synthesising a SPURIOUS SECOND PARK.
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        # COM handle OWNED by this thread, created on it after CoInitialize.
        self.com_igor = None
        self.com_initialized: bool | None = None   # None = not started; False = no pywin32
        self.parks = 0                             # observability
        # Reason for a BARE `trigger.set()`, consumed by `_bare_reason`.
        self._reason_override: str | None = None

    # --------------------------------------------------------------- lifecycle --

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> FailsafeWatchdog:
        """Start the thread (idempotent) and block briefly until it is up, so a park
        submitted right after cannot race COM initialisation."""
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._loop, name=self._name,
                                        daemon=True)
        self._thread.start()
        self._started.wait(2.0)              # real time: set by the watchdog thread
        return self

    def stop(self, *, join: float = 2.0) -> None:
        """Ask the thread to exit (teardown only; it normally lives for the process)."""
        self._stop.set()
        self._wake.set()                          # wake it out of its wait
        t = self._thread
        if t is not None and t.is_alive():
            t.join(join)
        self._thread = None

    # ---------------------------------------------------------------- requests --

    def request(self, reason: str, *, report: bool = False) -> ParkRequest:
        """Queue a retract and wake the thread. Returns at once; wait on the
        returned `ParkRequest` for the outcome. Safe from any thread."""
        req = ParkRequest(reason, report=report)
        self._q.put(req)
        self._wake.set()                          # the thread's OWN wake, not `trigger`
        return req

    def park_now(self, reason: str) -> ParkRequest:
        """Loop-free trigger for ANY thread: queue the retract and return; the watchdog
        retracts and REPORTS to the loop itself.

        RAISES when the thread is not running (safety-load-bearing): `HardwareARC.abort`
        treats a returned request as "the watchdog owns the retract", so a silent
        success on a dead watchdog is an abort that never happened. Raising lets
        `abort` fall through to direct COM/TCP. `request()` does not raise; its callers
        start the thread via `watchdog_for` first.
        """
        if not self.alive:
            raise RuntimeError(
                "failsafe watchdog is not running; refusing to queue a retract "
                "nobody will serve")
        return self.request(reason, report=True)

    # ------------------------------------------------------------- thread body --

    def _co_initialize(self) -> None:
        """Initialise COM for THIS thread, once (guarded: no pywin32 off the instrument)."""
        try:
            import pythoncom
        except Exception:                                     # noqa: BLE001
            self.com_initialized = False
            self._log("failsafe watchdog: pythoncom unavailable (not the instrument "
                      "PC?) -- COM is not initialised on this thread; the COM-first "
                      "trigger will fail over to TCP and SAY SO.")
            return
        try:
            pythoncom.CoInitialize()
            self.com_initialized = True
        except Exception as exc:                              # noqa: BLE001
            self.com_initialized = False
            self._log(f"failsafe watchdog: CoInitialize failed ({exc!r}); the "
                      f"COM-first trigger may fail over to TCP")

    def _co_uninitialize(self) -> None:
        if not self.com_initialized:
            return
        try:
            import pythoncom
            pythoncom.CoUninitialize()
        except Exception:                                     # noqa: BLE001,S110
            pass

    def _make_com_igor(self):
        """Create this thread's LazyIgor; it attaches on first use, on this thread, so
        it lives in the apartment `_co_initialize` set up."""
        try:
            from .igor import LazyIgor
            return LazyIgor()
        except Exception as exc:                              # noqa: BLE001
            self._log(f"failsafe watchdog: could not build a COM handle ({exc!r}); "
                      f"the fail-safe falls back to the TCP transport")
            return None

    def _loop(self) -> None:
        self._co_initialize()
        self.com_igor = self._make_com_igor()
        self._started.set()
        try:
            while not self._stop.is_set():
                # Instant on `request()`; the tick bounds a BARE trigger's latency.
                # A background periodic loop: stays on real time on every clock mode.
                self._wake.wait(_TICK_S)
                self._wake.clear()
                if self._stop.is_set():
                    break
                reqs = self._drain_queue()
                if self.trigger.is_set():
                    # Bare demand. Cleared BEFORE serving so a second demand during
                    # the retract is not lost; synthesised only when actually set.
                    self.trigger.clear()
                    reqs.append(ParkRequest(self._bare_reason(), report=True))
                if reqs:
                    self._serve_batch(reqs)
        finally:
            self._co_uninitialize()
            # Anyone still waiting must not hang on a dead watchdog.
            for req in self._drain_queue():
                self._finish_request(req, {
                    "transport": "failed", "verified": False, "z_v": None,
                    "critical": True, "retried": False, "verify_timeout": False,
                    "com_error": None, "detail": "failsafe watchdog stopped"})

    def _drain_queue(self) -> list[ParkRequest]:
        """Take every queued request, oldest first."""
        out: list[ParkRequest] = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out

    def _bare_reason(self) -> str:
        """Reason for a BARE trigger. A demand with no reason still parks under a
        generic label: an unexplained retract beats a skipped one."""
        reason = self._reason_override
        self._reason_override = None
        if reason is None:
            try:
                reason = self._reason_getter()
            except Exception:                                 # noqa: BLE001
                reason = None
        return reason or "out-of-band fail-safe trigger"

    def _serve_batch(self, reqs: list[ParkRequest]) -> None:
        """Run ONE retract for every demand drained this pass, then release them all.

        COALESCED ON PURPOSE: the retract is IDEMPOTENT (an absolute ramp of the
        model's Z channel), and serving N demands in sequence is harmful: each can
        take several seconds (worker join, settle, vendor fallback), and one interlock
        trip already yields two demands, so a third waiter would time out and report
        CRITICAL after a successful park. One retract, one outcome, fanned out.
        (Relation to INSTRUMENT_MODEL_SPEC §3 "no coalescing": docs/TECH_DEBT.md.)
        """
        reason = reqs[0].reason
        if len(reqs) > 1:
            others = ", ".join(r.reason for r in reqs[1:])
            self._log(f"failsafe watchdog: coalescing {len(reqs)} park demands into "
                      f"one retract ({reason}; also: {others})")
        try:
            result = self._work(reason, self.com_igor)
        except Exception as exc:                              # noqa: BLE001
            # Never die on an exception: report CRITICAL so it is still logged.
            self._log(f"failsafe watchdog: retract raised ({exc!r}) -- CRITICAL")
            result = {"transport": "failed", "verified": False, "z_v": None,
                      "critical": True, "retried": False, "verify_timeout": False,
                      "com_error": None, "detail": f"watchdog work raised: {exc}"}
        self.parks += 1
        if len(reqs) > 1:
            result = dict(result, coalesced=len(reqs))
        for req in reqs:
            # A copy each, so one caller cannot rewrite another's result.
            self._finish_request(req, dict(result))

    def _finish_request(self, req: ParkRequest, result: dict) -> None:
        req.result = result
        if not req.future.done():
            req.future.set_result(result)
        req.done.set()
        if req.report and self._report is not None:
            try:
                self._report(req.reason, result)
            except Exception as exc:                          # noqa: BLE001
                self._log(f"failsafe watchdog: report failed ({exc!r})")
