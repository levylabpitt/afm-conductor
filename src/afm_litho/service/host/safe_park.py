"""safe_park = abort = fail-safe retract: stop everything and get the tip to a safe spot.

Two triggers (operator Abort, any auto-detected failure), one implementation: ANY
failure -> safe_park. The trigger is out-of-band (COM first, TCP fallback) because a
comms failure is exactly when the router cannot be trusted. It runs Igor's one abort
function, `AFM_SafePark()`: td_Stop once per abort, bias 0, loop 2 re-armed at -10 V so
Height parks at 0 V (bridge_router.send_safe_park, afm_litho_run.ipf).

The sync `safe_park` below sends no Z command of its own: it confirms the retract from
Height < 1 V (2026-10-27; the meter's "Z Voltage"), tells Igor (ABORTCONFIRM), and
falls back to the vendor Withdraw once; still unconfirmed is CRITICAL and latches
`hub.fault`. ZSensor and Deflection are read and logged beside it. Design:
docs/migration/plan/abort-redesign.md.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import threading
import time
from dataclasses import dataclass

from ... import clock
from ...hardware import channels, retract
from ..command_guard import park_plan

_LOG = logging.getLogger(__name__)

# The plain withdraw's Height band (V): |Height| <= 0.5 V reads as withdrawn. Used only by
# the routine withdraw's readback (flex_dispatch withdraw verify, instrument/asylum.py
# _withdraw_verified); the park confirms on Height < 1 V (retract.RETRACTED_HEIGHT_V).
# Rig data: engaged ~66 V, withdrawn 0 V. Do not loosen without new data.
_WITHDRAWN_HEIGHT_TOL_V = 0.5

# Igor's retract loop drives Height down on its own after the call returns, so the confirm
# polls for up to 2 s + 0.5 s margin, the same bound as hw_approach.hw_withdraw. Igor's own
# backup confirms at 5 s (afm_litho_run.ipf AFM_AbortBackup), after this window.
_RETRACT_MAX_S = 2.0
_RETRACT_SETTLE_MARGIN_S = 0.5
_VERIFY_SETTLE_S = _RETRACT_MAX_S + _RETRACT_SETTLE_MARGIN_S      # 2.5 s, == hw_withdraw

# Poll inside the settle window rather than sleeping it out: same worst case, fast typical.
_VERIFY_POLL_S = 0.25

# Per-read bound (s) assumed by _watchdog_wait_bound; the park read itself is bounded by
# hardware_arc._PARK_READ_TIMEOUT_S, never the 30 s default, so an emergency stop cannot
# appear to hang (docs/FLEX_INTEGRATION_PLAN.md §F.2).
_VERIFY_READ_TIMEOUT_S = 2.0


def signal_run_abort(hub) -> None:
    """Signal every cooperative abort an IN-FLIGHT PASS watches (scan and write), and
    nothing else.

    `withdraw` needs exactly this without `_signal_stop`'s `scan_repeat` disable: the
    continuous loop's between-frame re-reference calls `withdraw`, and would stop itself.
    Thread-safe (Event.set + one attribute write), so the watchdog can run it; a no-op
    when idle. Never raises.
    """
    for ev_name in ("_hw_abort", "_hw_litho_abort"):
        ev = getattr(hub.host, ev_name, None)
        if ev is not None:
            try:
                ev.set()
            except Exception:                             # noqa: BLE001,S110
                pass
    # The twin plugin has no event to watch: it is told, and withdraws its own tip. Without
    # this a park would leave a TWIN write or scan running behind a "parked" state.
    from ..twin_plugin import abort as abort_twin
    abort_twin(hub, "run aborted")
    # Counted LAST, once every flag above is set (and so before any caller goes on to
    # the Igor command): a caller that awaited something can tell a signal arrived
    # meanwhile (`start_write` across its estimate), and whoever sees the count change
    # finds the flags already set. The events stay set between runs and cannot show a
    # new signal themselves. Only compared for change, so a lost increment is harmless.
    try:
        hub.host._run_abort_seq = int(getattr(hub.host, "_run_abort_seq", 0) or 0) + 1
    except Exception:                                     # noqa: BLE001,S110
        pass


# Which stop flags `_signal_stop` sets. The narrow ones exist because the scan, the
# continuous loop and the write each have a Stop that must not stop the others.
STOP_ALL = "all"                  # every park: loop disable + every cooperative abort
STOP_RUN = "run"                  # the in-flight pass only (what `withdraw` needs)
STOP_SCAN_PASS = "scan_pass"      # `hub._hw_abort` and the twin plugin: the running scan pass
STOP_REPEAT_LOOP = "repeat_loop"  # `hub._scan_repeat_stop`: the continuous supervisor
STOP_LITHO_RUN = "litho_run"      # the twin plugin only


def _signal_stop(hub, which: str = STOP_ALL) -> None:
    """The ONE place a stop flag is set. `which` picks the flags; the default is
    STEP 1 of every park: break any running loop AT ONCE (arm no further frame) and
    signal every cooperative abort, so an in-flight pass / scan / write stops.

    Thread-safe (dict writes + Event.set), so the watchdog can run it; a no-op when
    idle. Never raises for STOP_ALL / STOP_RUN.
    """
    if which == STOP_RUN:
        signal_run_abort(hub)
        return
    if which == STOP_SCAN_PASS:
        hub.host._hw_abort.set()
        from ..twin_plugin import abort as abort_twin
        abort_twin(hub, "scan stopped")
        return
    if which == STOP_REPEAT_LOOP:
        hub.host._scan_repeat_stop.set()
        return
    if which == STOP_LITHO_RUN:
        from ..twin_plugin import abort as abort_twin
        abort_twin(hub, "litho abort")
        return
    try:
        hub.host.scan_repeat["enabled"] = False
        if hub.host.scan_repeat.get("stopped_reason") is None:
            hub.host.scan_repeat["stopped_reason"] = "safe_park"
    except Exception:  # noqa: BLE001, S110 - failure here must not stop the park
        pass
    signal_run_abort(hub)
    for ev_name in ("_scan_repeat_stop",):
        ev = getattr(hub.host, ev_name, None)
        if ev is not None:
            try:
                ev.set()
            except Exception:  # noqa: BLE001, S110 - failure here must not stop the park
                pass


# ---------------------------------------------------------------- request_park --
# The one decision point. Every trigger names why it parks and in which of today's
# styles; each style does exactly what that trigger did before, nothing more.
PARK_FULL = "full"            # stop, out-of-band park, verify: returns an awaitable
PARK_WATCHDOG = "watchdog"    # bare trigger for the fail-safe thread (any thread)
PARK_ARC_ABORT = "arc_abort"  # `arc.abort()` from the chained-scan reader
PARK_WITHDRAW = "withdraw"    # the plain withdraw: returns an awaitable
PARK_DEFERRED = "deferred"    # ask the scan pump to park once the pass lets go
PARK_BRIDGE_DROP = "bridge_drop"   # reader thread: watchdog, else a direct park


def request_park(hub, reason: str, how: str = PARK_FULL, *, full_park=None,
                 body=None, arc=None, log=None, stop_id: str | None = None):
    """Every park trigger calls this. `how` is the style (PARK_*). PARK_FULL and
    PARK_WITHDRAW return an awaitable; the others act at once and return None
    (PARK_ARC_ABORT returns the transport). Only plain calls: nothing here waits, locks
    or writes a journal, so an abort is never slower for going through it.

    `full_park` lets a caller supply the full-park coroutine function it already holds.
    `stop_id` (PARK_FULL) pairs a door's park with the keeper's copy of its Stop
    (safety/stop_pairs.py): whichever reaches the host second shares the first one's park.
    """
    if how == PARK_FULL:
        if full_park is None:
            full_park = safe_park_async
        if stop_id is not None:
            return full_park(hub, reason, stop_id=stop_id)
        return full_park(hub, reason)
    if how == PARK_WATCHDOG:
        hub.host._safe_park_reason = reason
        hub.host._safe_park_now.set()
        return None
    if how == PARK_ARC_ABORT:
        return arc.abort()
    if how == PARK_WITHDRAW:
        from . import scan_cmds
        return scan_cmds.withdraw(hub, body)
    if how == PARK_DEFERRED:
        hub.host._hw_withdraw_after = True
        hub.host._hw_withdraw_epoch = hub.host._scan_epoch
        return None
    if how == PARK_BRIDGE_DROP:
        _park_on_bridge_drop(hub, reason, log)
        return None
    raise ValueError(f"unknown park style {how!r}")


def _park_on_bridge_drop(hub, reason: str, log) -> None:
    """The bridge dropped mid-run (reader thread, so nothing here may block)."""
    _park_off_loop(hub, reason, log, "bridge closed during a run")


# The park reason for an abort that started in Igor (its AFM Abort panel, not the service).
IGOR_SIDE_ABORT = "Igor-side abort"


def park_on_igor_abort(hub, frame: str, log) -> None:
    """Igor reported an abort the service did not send, while a run was in flight (reader
    thread, so nothing here may block). Treated as the cockpit Abort: every stop flag set
    at once, so the pass that wakes on `frame` stops and nothing re-engages, then the full
    park. Igor has already parked; the park's AFM_SafePark re-fire is harmless (td_Stop runs
    once per abort, the re-arm is absolute) and its confirm records the park as
    IGOR_SIDE_ABORT, latching a fault only if Height is not confirmed down. Never raises."""
    try:
        _signal_stop(hub)
        hub.host._safe_park_reason = IGOR_SIDE_ABORT
        log(f"SAFETY: Igor reported an abort the service did not send ({frame}); "
            "stopping the run")
    except Exception:                                     # noqa: BLE001,S110
        pass
    _park_off_loop(hub, IGOR_SIDE_ABORT, log, "Igor reported an abort during a run")


def _park_off_loop(hub, reason: str, log, what: str) -> None:
    """Hand the park to the watchdog, else fire the retract directly. For the reader
    thread, which must not block. Never raises."""
    # `existing_watchdog`, never `watchdog_for`: this runs inline on the reader
    # thread, which must not block, and watchdog_for would START a thread and wait.
    # bind_hardware already started one; park_now is a queue put plus an Event set.
    # park_now RAISES on a dead watchdog; do not swallow it (a retract nobody
    # serves). Say so and fall through to the direct out-of-band path.
    try:
        wd = existing_watchdog(hub)
        if wd is not None:
            wd.park_now(reason)
            return
        log(f"SAFETY: {what} and there is NO "
            "fail-safe watchdog; firing the retract directly.")
    except Exception as exc:                          # noqa: BLE001
        log(f"SAFETY: {what} and the fail-safe "
            f"watchdog could not take it ({exc!r}); firing the "
            f"retract directly.")
    # Direct out-of-band retract on THIS thread. Last resort (may fall back to
    # TCP): an attempted unverified retract beats one never sent.
    try:
        from ...hardware.bridge_router import send_safe_park
        arc = hub.host.hardware_arc
        send_safe_park(getattr(arc, "link", None),
                       getattr(arc, "com_igor", None),
                       log=log)
    except Exception as exc:                          # noqa: BLE001
        log(f"SAFETY CRITICAL: the direct retract also failed "
            f"({exc!r}). The tip may still be engaged.")


async def safe_park_async(hub, reason: str, *, stop_id: str | None = None) -> dict:
    """Stop everything and get the tip to a safe spot, then REPORT it.

    Transport/watchdog wrapper called by the Abort button, dead-man, bridge-drop reader,
    continuous supervisor and scan/write abort paths; the retract + verify IS the sync
    `safe_park` below. Hardware: out-of-band `AFM_SafePark()` trigger (COM first, TCP
    `ABORT` fallback) then `safe_park`'s confirm on the watchdog thread. Twin: stop the
    sim scan, then `safe_park` off-loop.

    Every call is logged, recorded and broadcast as a `safe_park` ws frame (never a
    silent stop). Returns the result dict. Never raises: a fail-safe must not fail.
    """
    is_hw = stop_route(hub) in TO_IGOR

    # 1-2. Signal the stop and start the retract + confirm (`start_park`), unless the
    #      keeper's copy of this Stop already started it (then this awaits that park).
    from ..safety.stop_pairs import start_once
    req, first = start_once(stop_id, lambda: start_park(hub, reason, hardware=is_hw))

    result: dict = {
        "reason": reason,
        # Before Igor dials in, the not-connected result overrides this with "disconnected".
        "backend": "hardware" if is_hw else "twin",
        "transport": None, "verified": None, "z_v": None,
        "critical": False, "retried": False, "verify_timeout": False,
        "com_error": None, "at": time.time()}

    if is_hw:
        result.update(await _await_hardware_park(req))
    else:
        result.update(await asyncio.wrap_future(req.future))

    if not first:
        return result                  # the stop port that started this park records it
    return await _finish(hub, reason, result)


# Where a park or a withdraw goes (`stop_route`).
ROUTE_NOT_CONNECTED = "not_connected"   # hardware station, Igor never dialled in
ROUTE_HARDWARE = "hardware"             # Igor dialled in and the bridge socket is open
ROUTE_COM_ESCAPE = "com_escape"         # Igor dialled in, the bridge socket has closed
ROUTE_COM_AFTER_RESTART = "com_after_restart"   # restarted after a death, no new bridge yet
ROUTE_TWIN = "twin"
TO_IGOR = (ROUTE_HARDWARE, ROUTE_COM_ESCAPE, ROUTE_COM_AFTER_RESTART)


def stop_route(hub) -> str:
    """The one decision of where a park or a withdraw goes (2026-10-28).

    Before Igor ever dials in: nowhere (no attempt, no guarantee). A service restarted after
    a death whose predecessor had the bridge is previously connected: until its new bridge
    binds, every stop goes to Igor over COM (after_death.park_over_com). Once the bridge
    has bound, every stop goes to Igor, even after its socket closes: the fail-safe thread
    sends AFM_SafePark over COM, the direct escape. Keyed on `real_bridge`, which stays
    set after a drop, never on `backend_live()`, which then reads "twin"."""
    from ..hub import real_bridge
    if before_igor_dials_in(hub):
        return ROUTE_NOT_CONNECTED
    if real_bridge(hub) is None:
        return ROUTE_COM_AFTER_RESTART if previously_connected(hub) else ROUTE_TWIN
    closed = getattr(hub.host, "bridge_closed", None)
    if closed is not None and closed.is_set():
        return ROUTE_COM_ESCAPE
    return ROUTE_HARDWARE


def start_park(hub, reason: str, *, report: bool = False, hardware: bool | None = None):
    """Steps 1-2 of every full park, on any thread and with no event loop: signal the
    stop, then start the retract + confirm, returning its `ParkRequest` at once.

    Hardware: queued on the fail-safe thread (out-of-band `AFM_SafePark()`, then
    `safe_park`'s confirm there); with no fail-safe thread the retract is fired directly,
    as the bridge-drop path does, and the request comes back already failed. Twin: the
    sim scan is stopped, then `safe_park` runs on a thread of its own (`_signal_stop`
    does not drive the sim's scan loop; it does tell the twin plugin to withdraw, and
    `safe_park` confirms only once the plugin's own Z is up). `report` hands the outcome
    to the loop afterwards; `hardware` is the caller's backend choice (decided here when
    None).
    """
    from ...hardware.failsafe_thread import ParkRequest

    # 1. Break loops + signal aborts. Repeated on the watchdog thread for loop-free
    #    parks; idempotent.
    _signal_stop(hub)
    if before_igor_dials_in(hub):
        req = ParkRequest(reason, report=report)
        _complete(hub, req, not_connected_result())
        return req
    if hardware is None:
        hardware = stop_route(hub) in TO_IGOR

    # 2. Retract + confirm via the sync `safe_park` on both backends.
    if hardware:
        try:
            wd = watchdog_for(hub)
        except Exception as exc:                          # noqa: BLE001
            _LOG.error("safe_park: no fail-safe thread (%r); firing the retract directly",
                       exc)
            request_park(hub, reason, PARK_BRIDGE_DROP,
                         log=lambda msg: _park_log(hub, msg))
            req = ParkRequest(reason, report=report)
            _complete(hub, req, {
                "transport": "failed", "verified": False, "z_v": None,
                "critical": True, "retried": False, "verify_timeout": False,
                "com_error": None, "detail": f"no fail-safe thread: {exc!r}"})
            return req
        return wd.request(reason, report=report)

    req = ParkRequest(reason, report=report)

    def _twin_park():
        try:
            hub.host.inst.abort_scan()
        except Exception:                                 # noqa: BLE001,S110
            pass
        try:
            out = _parkresult_to_dict(safe_park(hub, reason))
        except Exception as exc:                          # noqa: BLE001
            out = {"verified": False, "critical": True, "z_v": None,
                   "detail": f"safe_park raised: {exc!r}"}
        out["transport"] = "twin"
        _complete(hub, req, out)

    # A thread of its own, not the default executor: a park must not queue behind a
    # saturated pool, and it must run with no event loop at all.
    threading.Thread(target=_twin_park, name="afm-twin-park", daemon=True).start()
    return req


def _complete(hub, req, result: dict) -> None:
    """Fill a `ParkRequest` (result before `done`, as the fail-safe thread does), then
    report it to the loop if it asked to be."""
    req.result = result
    if not req.future.done():
        req.future.set_result(result)
    req.done.set()
    if req.report:
        _report_threadsafe(hub, req.reason, result, backend="twin"
                           if result.get("transport") == "twin" else "hardware")


def _parkresult_to_dict(pr: ParkResult) -> dict:
    """Render a `ParkResult` as the `safe_park_async` / `_finish` dict. `z_v` is the
    Height readback the confirm is gated on (2026-10-27)."""
    return {"verified": pr.verified, "critical": pr.critical, "z_v": pr.z_out_v,
            "zsensor_v": pr.zsensor_v, "defl_v": pr.defl_v, "method": pr.method,
            "reason": pr.reason, "retried": False, "verify_timeout": False}


async def _finish(hub, reason: str, result: dict) -> dict:
    """Steps 3-4 of a park: latch the engaged gate, record the outcome on the hub, then
    REPORT it (operator log + session record + ws broadcast).

    Shared with out-of-band parks (demanded via `hub._safe_park_now`, run by the
    watchdog), which `_report_threadsafe` marshals here. Never a silent stop.
    """
    _record_on_hub(hub, reason, result)

    # 4. Report. CRITICAL is logged loudly: the tip may still be engaged.
    await hub.log(_outcome_line(reason, result))
    failed = _record_in_session(hub, reason, result)
    if failed:
        await hub.log(failed)
    try:
        await hub.broadcast(_park_frame(reason, result))
    # the park result is returned even if the broadcast fails
    except Exception:  # noqa: BLE001, S110
        pass
    return result


def finish_without_loop(hub, reason: str, result: dict) -> None:
    """`_finish` for a park whose service has no event loop yet (the stop port starts
    before the loop is captured): the hub and session records directly, the log line to
    the module logger. With a loop, `_finish` itself runs there. Never raises."""
    loop = getattr(hub, "_loop", None)
    if loop is not None:
        try:
            asyncio.run_coroutine_threadsafe(_finish(hub, reason, result), loop)
            return
        except Exception:                                 # noqa: BLE001,S110
            pass
    try:
        _record_on_hub(hub, reason, result)
        _LOG.warning("%s", _outcome_line(reason, result))
        failed = _record_in_session(hub, reason, result)
        if failed:
            _LOG.warning("%s", failed)
    except Exception as exc:                              # noqa: BLE001
        _LOG.error("safe_park: recording the park failed (%r)", exc)


def _record_on_hub(hub, reason: str, result: dict) -> None:
    """Step 3 of a park: latch the engaged gate and record the outcome on the hub."""
    # Clear the engaged gate ONLY on a VERIFIED retract. A CRITICAL park (tip MAY still be
    # engaged) deliberately LEAVES engaged True, so a subsequent mode switch stays REFUSED
    # until an operator withdraws: an unverified clear would allow a channel/gain-sign
    # flip on a possibly-engaged tip.
    if result.get("verified"):
        hub.host.engaged = False

    # 3. Record it on the hub.
    sp = getattr(hub.host, "_safe_park", None)
    if isinstance(sp, dict):
        sp["count"] = int(sp.get("count", 0)) + 1
        sp["active"] = True
        sp["last_reason"] = reason
        sp["verified"] = result["verified"]
        sp["critical"] = result["critical"]
        sp["transport"] = result["transport"]
        sp["z_v"] = result["z_v"]
        sp["at"] = result["at"]


def _outcome_line(reason: str, result: dict) -> str:
    """The operator log line. CRITICAL is loud: the tip may still be engaged."""
    zbit = (f" (Height = {result['z_v']:.3f} V)"
            if result["z_v"] is not None else "")
    if result.get("not_connected"):
        return f"safe_park: {reason} -- {NOT_CONNECTED}"
    if result["critical"] and retract.is_retracted(result["z_v"]):
        # Height reads under 1 V, but the park is not claimed: a latched fault (no further
        # Z command is sent), or the twin plugin's tip still withdrawing.
        return (f"CRITICAL safe_park: {reason} -- park NOT verified{zbit}; "
                f"{result.get('detail') or 'a fault is latched or the tip is still moving'}"
                f", OPERATOR MUST INTERVENE")
    if result["critical"]:
        return (f"CRITICAL safe_park: {reason} -- Height did NOT come under 1 V"
                f"{zbit}; tip may still be engaged, OPERATOR MUST INTERVENE")
    return (f"SAFETY safe_park: {reason} -- tip retracted"
            f" (verified={result['verified']}){zbit}")


def _record_in_session(hub, reason: str, result: dict) -> str | None:
    """The session record; returns the line to log if it failed."""
    try:
        hub.store.record("safety", {
            "label": f"safe_park: {reason}",
            "event": "safe_park", "reason": reason,
            "backend": result["backend"], "transport": result["transport"],
            "verified": result["verified"], "z_v": result["z_v"],
            "critical": result["critical"], "retried": result["retried"]})
    except Exception as exc:                              # noqa: BLE001
        return f"safe_park: session record failed ({exc})"
    return None


def _park_frame(reason: str, result: dict) -> dict:
    return {"type": "safe_park", "reason": reason,
            "backend": result["backend"],
            "transport": result["transport"],
            "verified": result["verified"], "z_v": result["z_v"],
            "critical": result["critical"]}


# ------------------------------------------------------- the watchdog thread --
#
# The hardware retract runs on ONE dedicated non-pool daemon thread that CoInitialize'd
# COM and owns a LazyIgor (hardware/failsafe_thread.py; docs/FLEX_INTEGRATION_PLAN.md
# §B.2/B.3). Any thread may demand a park without a loop:
#     hub.host._safe_park_reason = "..."; hub.host._safe_park_now.set()   # or park_now("...")
# and the watchdog hands the log / record / broadcast back to the loop itself.

# Bound on the async wait for the watchdog's outcome before reporting unverified. Meant
# to fire only when the watchdog is wedged (a stuck `link.send`). Known too short for
# the current sequence (worker join + vendor fallback): docs/TECH_DEBT.md.
_WATCHDOG_MARGIN_S = 3.0


def _watchdog_wait_bound() -> float:
    return 2.0 * (_VERIFY_SETTLE_S + _VERIFY_READ_TIMEOUT_S) + _WATCHDOG_MARGIN_S


# Guards get-or-create of the per-hub watchdog. `watchdog_for` runs on the LOOP and on
# FOREIGN THREADS (bridge-router reader), so without it two callers could each start
# one; the orphan keeps its COM apartment and still watches the shared
# `hub._safe_park_now` (is_set/clear not atomic), so one trigger could fire TWO retracts.
_WATCHDOG_LOCK = threading.Lock()


def watchdog_for(hub):
    """The hub's ONE fail-safe watchdog thread (`hub._failsafe`), started on first use.

    Lazy so a park always has a thread; `app.py` / `hub.bind_hardware` also start it
    eagerly so COM is initialised BEFORE the first emergency. Safe from any thread, but
    BLOCKS briefly on a cold start, so a non-blocking caller (the bridge-router reader)
    must use `existing_watchdog` instead.
    """
    from ...hardware.failsafe_thread import FailsafeWatchdog
    with _WATCHDOG_LOCK:
        wd = getattr(hub.host, "_failsafe", None)
        if wd is not None and wd.alive:
            return wd
        return _build_watchdog(hub, FailsafeWatchdog)


def existing_watchdog(hub):
    """The hub's watchdog IF one is already running, else None. NEVER starts one, so
    it is safe to call from a thread that must not block (the bridge-router reader,
    whose contract is documented at BridgeRouter.on_telem/on_closed)."""
    wd = getattr(hub.host, "_failsafe", None)
    return wd if (wd is not None and wd.alive) else None


def _build_watchdog(hub, FailsafeWatchdog):
    """Construct + start the watchdog. Called with `_WATCHDOG_LOCK` held."""

    def _work(reason, com_igor):
        # STEP 1 here too, so a loop-free park (bridge drop, ZMQ thread) also breaks
        # loops and signals aborts. Idempotent.
        _signal_stop(hub)
        route = stop_route(hub)
        if route == ROUTE_NOT_CONNECTED:
            return not_connected_result()
        if route == ROUTE_COM_AFTER_RESTART:
            from .after_death import park_over_com
            return park_over_com(hub, reason, com_igor)
        # STEP 2: out-of-band trigger so the tip moves even if the router is wedged.
        # This is NOT the verify (INSTRUMENT_MODEL_SPEC §13.1 item 7).
        transport, com_error = _fire_out_of_band(hub, com_igor)
        # STEP 3: the confirm (Height < 1 V), on this COM-owning thread so its
        # vendor_withdraw fallback can use COM safely.
        pr = safe_park(hub, reason)
        out = _parkresult_to_dict(pr)
        out["transport"] = transport
        out["com_error"] = com_error
        if (stop_route(hub) == ROUTE_COM_ESCAPE and not pr.verified
                and transport in ("com", "com_raw")):
            out["detail"] = BRIDGE_DOWN_UNCONFIRMED
        return out

    def _report(reason, result):
        _report_threadsafe(hub, reason, result)

    def _log(msg):
        _log_threadsafe(hub, msg)

    wd = FailsafeWatchdog(
        _work, report=_report, log=_log,
        trigger=getattr(hub.host, "_safe_park_now", None),
        reason_getter=lambda: getattr(hub.host, "_safe_park_reason", None))
    wd.start()
    try:
        hub.host._failsafe = wd          # published only AFTER it is actually running
    except Exception:                                     # noqa: BLE001,S110
        pass
    return wd


def before_igor_dials_in(hub) -> bool:
    """A hardware station with no bridge bound yet that was never connected. Rule
    (2026-10-28): before the bridge is up, the program makes no attempt and no guarantee to
    control Igor, so a park sends nothing, and it must not confirm from the twin's
    simulated Height either."""
    from ..hub import real_bridge
    return (getattr(hub.host, "backend", None) == "hardware" and real_bridge(hub) is None
            and not previously_connected(hub))


def previously_connected(hub) -> bool:
    """A hardware station restarted after a death whose predecessor had a live bridge
    (`--after-death --last-bridge up`). Igor may still hold the tip down, so this service
    counts as connected from boot (2026-10-28)."""
    from .after_death import predecessor_had_bridge
    return getattr(hub.host, "backend", None) == "hardware" and predecessor_had_bridge()


NOT_CONNECTED = "not connected: Igor has not dialled in, so nothing was sent or checked"
BRIDGE_DOWN_UNCONFIRMED = ("the bridge socket is closed: AFM_SafePark was sent over COM, "
                           "but Height cannot be read, so the retract is NOT confirmed")


def not_connected_result() -> dict:
    """A park's outcome before Igor dials in: nothing done, and not a failure of the park."""
    return {"transport": None, "backend": "disconnected", "verified": False, "z_v": None,
            "critical": False, "retried": False, "verify_timeout": False, "com_error": None,
            "not_connected": True, "detail": NOT_CONNECTED}


def _log_threadsafe(hub, msg: str) -> None:
    """Log from a NON-loop thread, best effort. `hub.log` is a coroutine, so it is
    marshalled onto the captured loop; with no loop (a unit test) the message is
    dropped rather than raising on the fail-safe path."""
    loop = getattr(hub, "_loop", None)
    if loop is None:
        return
    try:
        asyncio.run_coroutine_threadsafe(hub.log(msg), loop)
    except Exception:                                     # noqa: BLE001,S110
        pass


def full_park_result(reason: str, result: dict | None, backend: str = "hardware") -> dict:
    """A park outcome with every field `_finish` reads, `result` laid over the defaults."""
    full = {"reason": reason, "backend": backend, "transport": None,
            "verified": None, "z_v": None, "critical": False, "retried": False,
            "verify_timeout": False, "com_error": None, "at": time.time()}
    full.update(result or {})
    return full


def _report_threadsafe(hub, reason: str, result: dict, backend: str = "hardware") -> None:
    """Hand an OUT-OF-BAND park's outcome back to the loop for the log / session
    record / broadcast. Best effort and non-blocking: the retract already happened,
    and reporting must never be allowed to delay or fail it."""
    loop = getattr(hub, "_loop", None)
    if loop is None:
        return
    full = full_park_result(reason, result, backend)
    try:
        asyncio.run_coroutine_threadsafe(_finish(hub, reason, full), loop)
    except Exception:                                     # noqa: BLE001,S110
        pass


# --------------------------------------------- the synchronous retract sequence --


def _fire_out_of_band(hub, com_igor) -> tuple[str | None, str | None]:
    """Fire AFM_SafePark OUT-OF-BAND (COM first, TCP fallback) so the tip starts
    retracting even when the router is wedged: Igor's one abort function, never delayed.
    The confirm is the sync `safe_park` that follows.

    Runs on the watchdog thread, the one whose apartment can service COM. A COM failure
    that falls back to TCP is captured into `com_error` and logged, never silent
    (docs/FLEX_INTEGRATION_PLAN.md §B.2). Returns `(transport, com_error)`; never raises.
    """
    from ...hardware.bridge_router import send_safe_park

    hw = getattr(hub.host, "hardware_arc", None)
    if hw is None:                                        # backend vanished mid-park
        return "failed", None
    # The ARC's COM handle, else the watchdog's LazyIgor. On the instrument these are
    # the SAME object (app.py), attached only from this thread.
    com = getattr(hw, "com_igor", None) or com_igor
    com_err: list[str] = []

    def _log(msg):
        com_err.append(str(msg))
        _log_threadsafe(hub, "SAFETY (WARNING) " + str(msg))

    try:
        transport = send_safe_park(hw.link, com, log=_log)
    except Exception as exc:                              # noqa: BLE001
        # Raises only when BOTH COM and link.send failed; the sync primitive that
        # follows still confirms, falls back to vendor_withdraw and reports the outcome.
        _log_threadsafe(hub, f"SAFETY (WARNING) out-of-band trigger failed: {exc}")
        return "failed", (com_err[-1] if com_err else str(exc))
    return transport, (com_err[-1] if com_err else None)


async def _await_hardware_park(req) -> dict:
    """Await the watchdog thread's outcome, BOUNDED, without blocking the loop. On
    timeout report UNVERIFIED + CRITICAL rather than hang: the retract is still queued
    to a thread that does not depend on this loop; only the CONFIRMATION failed."""
    bound = _watchdog_wait_bound()
    # wrap_future, NOT `to_thread(req.wait)`: to_thread holds a default-executor thread,
    # so the confirmation would queue behind a saturated pool during exactly the incident
    # that saturates it. wrap_future costs no thread.
    try:
        # real time: set by the watchdog thread
        return await asyncio.wait_for(asyncio.wrap_future(req.future), bound)
    except TimeoutError:                # asyncio.TimeoutError is an alias
        pass
    if req.result is not None:                            # landed in the same breath
        return req.result
    return {"transport": "pending", "verified": False, "z_v": None,
            "critical": True, "retried": False, "verify_timeout": True,
            "com_error": None,
            "detail": ("failsafe watchdog did not report within "
                       f"{bound:g}s; the retract is queued to it and does not "
                       "depend on this loop")}


# ============================================================================
# The synchronous park primitive. The retract itself is Igor's one abort function
# (AFM_SafePark), fired before this runs; Python sends no Z command of its own. Rig
# 2026-10-27 (Cypher): only the -10 V loop re-arm retracts, a Height ramp does not move Z,
# and td_Stop undefines the loops (docs/guides/INSTRUMENT_CONFIG.md section 6).
# ============================================================================

# Bounded wait for an aborted litho worker, BEFORE any Z command: a park must never race
# a pass (§3 step 1), and a wedged worker must not hang the park.
_WORKER_JOIN_TIMEOUT_S = 5.0


@dataclass
class ParkResult:
    """Outcome of a `safe_park` (§10, §12.1). `z_out_v` is the Height readback the confirm
    is gated on (< 1 V); `zsensor_v` + `defl_v` are the independent sensors, read and
    logged beside it."""

    verified: bool
    method: str                 # "abort" | "vendor_withdraw"
    z_out_v: float | None
    zsensor_v: float | None
    defl_v: float | None
    critical: bool
    reason: str
    ts: float | None = None     # store time, set by `_store_last_park`

    def to_dict(self) -> dict:
        """Shape for `GET /api/park/last`, `getLastPark` and `/api/instrument`'s
        `last_park`. Uses `z_out_v`, not the legacy `z_v` of `_parkresult_to_dict`."""
        return {"verified": self.verified, "method": self.method,
                "z_out_v": self.z_out_v, "zsensor_v": self.zsensor_v,
                "defl_v": self.defl_v, "critical": self.critical,
                "reason": self.reason, "ts": self.ts}


def _park_log(hub, msg: str) -> None:
    """Log a park message on either a sync test hub (`def log`) or the real async
    hub (`async def log`). Never raises; a fail-safe must not fail on its own log."""
    log = getattr(hub, "log", None)
    if log is None:
        _LOG.warning("%s", msg)
        return
    try:
        res = log(msg)
    except Exception:                                     # noqa: BLE001
        _LOG.warning("%s", msg)
        return
    if inspect.isawaitable(res):
        # Async hub.log: marshal onto its loop; with no loop, close the coroutine and
        # mirror to the module logger so the message is never lost.
        loop = getattr(hub, "_loop", None)
        if loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(res, loop)
                return
            except Exception:                            # noqa: BLE001,S110
                pass
        try:
            res.close()
        except Exception:                                # noqa: BLE001,S110
            pass
        _LOG.warning("%s", msg)


def _read_retract(arc, plan) -> tuple[bool, float | None, float | None, float | None]:
    """One confirm read: `(confirmed, height, zsensor, defl)`. Confirmed when Height is
    under 1 V (2026-10-27). ZSensor and Deflection are read for the record only."""
    height = _safe_read(arc, plan.z_channel)
    if height is None:
        # Never a confirm; and on a stalled bridge each further read would block for the
        # full bound, so stop here.
        return False, None, None, None
    zsensor = _safe_read(arc, channels.Z_SENSOR)
    defl = _safe_read(arc, channels.DEFLECTION)
    return retract.is_retracted(height), height, zsensor, defl


def _safe_read(arc, channel: str) -> float | None:
    """One sensor read; None on any failure (a failed read is never evidence)."""
    try:
        v = float(arc.read(channel))
    except Exception:                                     # noqa: BLE001
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


# Twin sim blocks (count, size) advanced per settle iteration.
_TWIN_SETTLE_STEPS = 40
_TWIN_SETTLE_BLOCK = 50


def _is_hardware_arc(arc) -> bool:
    """True for a real bridge/COM ARC, which Igor's abort function retracts on its own and
    whose confirm must be polled. False for the twin (stepped) and test doubles."""
    return (getattr(arc, "link", None) is not None
            or getattr(arc, "com_igor", None) is not None)


def _poll_retract(arc, plan):
    """Confirm the retract, letting Z settle where it is not synchronous: test doubles
    return at once, the twin is stepped, the real bridge is POLLED for up to the settle
    window. Returns `(confirmed, height, zsensor, defl)`."""
    res = _read_retract(arc, plan)
    if res[0]:
        return res
    # A FAILED read will not change by waiting: do not spin a stalled bridge.
    if res[1] is None:
        return res
    stepper = getattr(arc, "step", None)
    can_step = callable(stepper)
    if not can_step and not _is_hardware_arc(arc):
        return res                       # fake double: settling cannot change it
    deadline = time.monotonic() + _VERIFY_SETTLE_S      # real time: Igor's loop is the setter
    while time.monotonic() < deadline:
        if can_step:
            for _ in range(_TWIN_SETTLE_STEPS):
                try:
                    stepper(_TWIN_SETTLE_BLOCK)
                except Exception:                        # noqa: BLE001
                    break
        else:
            time.sleep(_VERIFY_POLL_S)
        res = _read_retract(arc, plan)
        if res[0]:
            return res
        if res[1] is None:               # bridge went stale mid-poll; stop spinning
            return res
    return res


def _twin_tip_still_down(hub) -> bool:
    """True when the twin plugin's own tip is not withdrawn once the settle window is over.
    The old twin's sensors say nothing about the plugin, which runs the write and the scan.
    The withdraw was commanded when the park began (`_signal_stop`); this only reads the
    plugin's Z until it gets there, as the rig's verify polls its sensors, and never delays
    the command."""
    from ..twin_plugin import twin_only
    twin = getattr(hub.host, "_twin_plugin", None)
    if twin is None or not twin_only(hub):
        return False                  # a hardware park never waits on, or is judged by, the twin
    end = clock.now() + _VERIFY_SETTLE_S
    while not twin.withdrawn():
        if clock.now() >= end:
            return True
        clock.sleep(_VERIFY_POLL_S)
    return False


def _unverified_twin_park(hub, reason: str, how: str, z_out, zsensor, defl) -> ParkResult:
    """The old twin's sensors verify but the plugin's tip is still down: not verified, and
    the tip is not safe yet. No fault is latched (the plugin is withdrawing, not stuck)."""
    result = ParkResult(verified=False, method=how, z_out_v=z_out, zsensor_v=zsensor,
                        defl_v=defl, critical=True, reason=reason)
    _park_log(hub, f"CRITICAL safe_park: {reason} -- park_unverified; the twin plugin's "
                   f"tip is still down (it is withdrawing); not claiming a retract")
    _store_last_park(hub, result)
    return result


def safe_park(hub, reason: str) -> ParkResult:
    """Stop everything and confirm the tip is retracted (INSTRUMENT_MODEL_SPEC §3, §10;
    docs/migration/plan/abort-redesign.md):

      1. Abort + JOIN any litho worker BEFORE anything else (never race a pass).
      2. The retract is Igor's one abort function. On hardware the out-of-band trigger has
         already run it; on the twin it runs here (`arc.abort_retract`). No STOPLOOP, no
         ramp: on the Cypher only the loop re-arm retracts.
      3. Confirm from Height < 1 V, polled up to 2.5 s; ZSensor and Deflection are read
         and logged. Confirmed: tell Igor (ABORTCONFIRM).
      4. Not confirmed: `vendor_withdraw()` ONCE, confirm again.
      5. Still not: critical, latch `hub.fault`, log CRITICAL.
      6. Igor's own fault flag (its 5 s backup ran the vendor Withdraw) latches `hub.fault`.

    Stores `hub.last_park`. Idempotent under a latched fault (no further Z commands).
    Before Igor dials in: nothing. Restarted after a death with no new bridge yet: the
    COM park on the fail-safe thread (`_park_on_failsafe_thread`). Never raises: a
    fail-safe must not fail.
    """
    route = stop_route(hub)
    if route == ROUTE_NOT_CONNECTED:
        result = ParkResult(verified=False, method="not_connected", z_out_v=None,
                            zsensor_v=None, defl_v=None, critical=False, reason=reason)
        _park_log(hub, f"safe_park: {reason} -- {NOT_CONNECTED}")
        _store_last_park(hub, result)
        return result
    if route == ROUTE_COM_AFTER_RESTART:
        return _park_on_failsafe_thread(hub, reason)
    # The one ARC driven for read/confirm/vendor_withdraw (§10): the real HardwareARC, or
    # the twin's SoftwareARC (bound to hardware_arc, with `hub.arc` as fallback).
    arc = getattr(hub.host, "hardware_arc", None) or getattr(hub.host, "arc", None)
    plan = park_plan(hub.host.instrument_model)

    # Break loops / signal aborts at once (harmless when idle).
    try:
        _signal_stop(hub)
    except Exception:                                     # noqa: BLE001,S110
        pass

    # 1. ABORT + JOIN the litho worker BEFORE any Z command (§3 step 1).
    worker = getattr(hub.host, "litho_worker", None)
    if worker is not None:
        try:
            alive = worker.is_alive()
        except Exception:                                # noqa: BLE001
            alive = True
        if alive:
            try:
                worker.request_abort()
            except Exception:                            # noqa: BLE001,S110
                pass
            try:
                worker.join(_WORKER_JOIN_TIMEOUT_S)
            except Exception:                            # noqa: BLE001,S110
                pass

    result = _park_once(hub, reason, arc, plan)
    if arc is not None:
        _latch_igor_fault(hub, arc, reason)
    return result


def _park_on_failsafe_thread(hub, reason: str) -> ParkResult:
    """`safe_park` for a previously connected service with no bridge yet: only COM reaches
    Igor, and COM is set up on the fail-safe thread, so the park runs there (`_work` ->
    after_death.park_over_com) and this waits for its outcome, boundedly."""
    out: dict = {}
    try:
        request = watchdog_for(hub).request(reason)
        # The setter is another real thread, so this is a real-time wait (clock rule b).
        clock.wait(request.done, _watchdog_wait_bound())
        out = request.result or {}
    except Exception as exc:                              # noqa: BLE001
        _LOG.error("safe_park: the fail-safe thread could not take the park (%r)", exc)
    verified = out.get("verified") is True
    result = ParkResult(verified=verified, method=out.get("method") or "abort",
                        z_out_v=out.get("z_v"), zsensor_v=None, defl_v=None,
                        critical=not verified, reason=reason)
    _store_last_park(hub, result)
    return result


def _park_once(hub, reason: str, arc, plan) -> ParkResult:
    """Steps 2-5 of `safe_park`."""
    # IDEMPOTENT under a latched fault: the abort + one vendor fallback already fired for
    # this episode, so issue NO further Z commands; report the latched state (§3 step 4).
    if getattr(hub, "fault", None):
        result = ParkResult(verified=False, method="vendor_withdraw",
                            z_out_v=_safe_read(arc, plan.z_channel) if arc else None,
                            zsensor_v=_safe_read(arc, channels.Z_SENSOR) if arc else None,
                            defl_v=_safe_read(arc, channels.DEFLECTION) if arc else None,
                            critical=True, reason=reason)
        _store_last_park(hub, result)
        return result

    if arc is None:
        # No ARC to command: report CRITICAL, do not pretend.
        result = ParkResult(verified=False, method="abort", z_out_v=None,
                            zsensor_v=None, defl_v=None, critical=True,
                            reason=reason)
        _latch_fault(hub, reason)
        _park_log(hub, f"CRITICAL safe_park: {reason} -- park_unverified, no "
                       f"ARC to command; tip may still be engaged")
        _store_last_park(hub, result)
        return result

    # 2. The twin runs Igor's abort function here; on hardware it has already run.
    if not _is_hardware_arc(arc):
        try:
            retract.twin_abort(arc)
        except Exception as exc:                          # noqa: BLE001
            _LOG.warning("safe_park abort retract failed: %s", exc)

    # 3. Confirm from Height.
    confirmed, z_out, zsensor, defl = _poll_retract(arc, plan)
    if confirmed and _twin_tip_still_down(hub):
        return _unverified_twin_park(hub, reason, "abort", z_out, zsensor, defl)
    if confirmed:
        retract.confirm_retract(arc)
        result = ParkResult(verified=True, method="abort", z_out_v=z_out,
                            zsensor_v=zsensor, defl_v=defl, critical=False,
                            reason=reason)
        _park_log(hub, f"SAFETY safe_park: {reason} -- tip retracted (verified; "
                       f"Height={_fmt_v(z_out)}, ZSensor={_fmt_v(zsensor)}, "
                       f"Deflection={_fmt_v(defl)})")
        _store_last_park(hub, result)
        return result

    # 4. Not confirmed -> the vendor full withdraw ONCE, then confirm again.
    retract.vendor_withdraw(arc)
    confirmed, z_out, zsensor, defl = _poll_retract(arc, plan)
    if confirmed and _twin_tip_still_down(hub):
        return _unverified_twin_park(hub, reason, "vendor_withdraw", z_out, zsensor, defl)
    if confirmed:
        retract.confirm_retract(arc)
        result = ParkResult(verified=True, method="vendor_withdraw", z_out_v=z_out,
                            zsensor_v=zsensor, defl_v=defl, critical=False,
                            reason=reason)
        _park_log(hub, f"SAFETY safe_park: {reason} -- tip retracted via vendor "
                       f"withdraw (verified; Height={_fmt_v(z_out)}, "
                       f"ZSensor={_fmt_v(zsensor)}, Deflection={_fmt_v(defl)})")
        _store_last_park(hub, result)
        return result

    # 5. Still unconfirmed after the vendor fallback -> CRITICAL, latch the fault.
    #    NEVER log "withdrawn"/"retracted" on this path: the tip may still be engaged.
    result = ParkResult(verified=False, method="vendor_withdraw", z_out_v=z_out,
                        zsensor_v=zsensor, defl_v=defl, critical=True, reason=reason)
    _latch_fault(hub, reason)
    _park_log(hub, f"CRITICAL safe_park: {reason} -- park_unverified; the abort AND "
                   f"the vendor withdraw both failed the Height confirm "
                   f"(ZSensor={_fmt_v(zsensor)}, Deflection={_fmt_v(defl)}, "
                   f"commanded {plan.z_channel}={_fmt_v(z_out)}); tip may still "
                   f"be engaged, OPERATOR MUST INTERVENE and clear the fault")
    _store_last_park(hub, result)
    return result


def _latch_igor_fault(hub, arc, reason: str) -> None:
    """Read Igor's fault flag (its 5 s backup found Height not retracted and ran the vendor
    Withdraw) and latch it as `hub.fault`. The read leaves Igor's flag set: only the
    operator's fault clear resets it (CLEARFAULT), so a lost reply loses nothing. An ARC
    with no such flag (the twin, test doubles) or a failed read latches nothing."""
    reader = getattr(arc, "read_abort_fault", None)
    if not callable(reader):
        return
    try:
        igor_fault = bool(reader())
    except Exception as exc:                              # noqa: BLE001
        _LOG.warning("reading Igor's abort fault failed: %s", exc)
        return
    if igor_fault:
        latch_igor_abort_fault(hub, reason)


def latch_igor_abort_fault(hub, reason: str, detail: str | None = None) -> None:
    """Latch `hub.fault` for Igor's own backup fault (or, with `detail`, for a fault read
    that failed) and say so. Never overwrites a latched fault."""
    what = detail or ("Igor's 5 s abort backup found Height not under 1 V and ran the "
                      "vendor withdraw")
    try:
        if getattr(hub, "fault", None) is None:
            hub.fault = {"reason": "park_unverified", "at": clock.wall(),
                         "state": "fault", "critical": True,
                         "detail": f"{what} ({reason})"}
    except Exception:                                    # noqa: BLE001,S110
        pass
    _park_log(hub, f"CRITICAL safe_park: {reason} -- {what}; tip may still be engaged, "
                   f"OPERATOR MUST INTERVENE and clear the fault")


def _fmt_v(v: float | None) -> str:
    return f"{v:.3f} V" if isinstance(v, (int, float)) else "-"


def _latch_fault(hub, reason: str) -> None:
    """Latch `hub.fault` so every engage is refused until an operator clears it (POST
    /api/fault/clear). Never overwrites a latched fault."""
    try:
        if getattr(hub, "fault", None) is None:
            hub.fault = {"reason": "park_unverified", "at": time.time(),
                         "state": "fault", "critical": True,
                         "detail": f"safe_park could not verify a retract: {reason}"}
    except Exception:                                    # noqa: BLE001,S110
        pass


def _store_last_park(hub, result: ParkResult) -> None:
    # `ts` is the park time, stable once set (not a per-read now()), so getInstrument
    # stays idempotent (REST == FLEX parity).
    if result.ts is None:
        result.ts = time.time()
    try:
        hub.host.last_park = result
    except Exception:                                    # noqa: BLE001,S110
        pass
