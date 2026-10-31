"""`--after-death`: a service the keeper restarted makes the tip safe before anything else
talks to Igor, if the service that died had Igor (docs/migration/plan/phase6-board/plan.md,
slice K3).

Rule (2026-10-28): before the bridge is up, the program makes no attempt and no guarantee to
control Igor, unless the service that died had a live bridge. So:
  - the service that died had a live bridge (`--last-bridge up`): the tip may be down, and
    this service counts as previously connected from boot. Before anything else it sends
    Igor's abort `AFM_SafePark()` over COM and confirms Height < 1 V over COM, then latches
    the restart fault. Until a new bridge binds, every stop goes to Igor over COM
    (`park_over_com`). When Igor dials in, the socket make-safe goes first (`on_link`);
  - it never had one (`--last-bridge down`, or none given): it never drove Igor, so
    nothing is sent and nothing is claimed, as on a cold start.

Igor plays the pass it was given to its end (decision 6). A repeat make-safe is
harmless: Igor's abortStopped skips the second td_Stop. The keeper itself never sends COM.
"""

from __future__ import annotations

from ... import clock
from ...keeper.protocol import BRIDGE_UP

REASON = "service restarted after a death; tip state unknown"
BOOT_PARK_REASON = "service restarted after a death"
NOT_CONFIRMED_UP = ("service restarted after a death: AFM_SafePark was sent over COM, but "
                    "Height did not read under 1 V, so the tip is NOT confirmed up")

# The service that died had a live bridge to Igor (`--last-bridge up`).
_predecessor_had_bridge = False
# The socket make-safe is still to be sent when Igor first dials in.
_socket_due = False


def predecessor_had_bridge() -> bool:
    """Whether this service was restarted after a death with `--last-bridge up`."""
    return _predecessor_had_bridge


def boot(hub, last_bridge: str | None) -> bool:
    """Step 1, before the app is imported. Returns whether the dead service had Igor (and
    so whether the tip was parked over COM). Never raises."""
    global _socket_due, _predecessor_had_bridge
    _predecessor_had_bridge = _socket_due = last_bridge == BRIDGE_UP
    if not _predecessor_had_bridge:
        print("[after death] the service that died had no bridge to Igor: nothing to make "
              "safe", flush=True)
        return False
    verified = _park_at_boot(hub)
    latch_fault(hub)
    print(f"[after death] AFM_SafePark sent over COM; Height under 1 V: {verified}. "
          f"Fault latched: {hub.fault['detail']}. The socket make-safe goes first when "
          f"Igor dials in.", flush=True)
    return True


def _park_at_boot(hub) -> bool:
    """The park every stop takes before the new bridge binds, run once now and waited for.
    Returns whether Height was confirmed under 1 V."""
    from .safe_park import (_watchdog_wait_bound, finish_without_loop, full_park_result,
                            watchdog_for)
    try:
        request = watchdog_for(hub).request(BOOT_PARK_REASON)
        # The setter is another real thread, so this is a real-time wait (clock rule b).
        clock.wait(request.done, _watchdog_wait_bound())
        result = request.result
    except Exception as exc:                              # noqa: BLE001
        result = None
        print(f"[after death] the park at boot failed: {exc!r}", flush=True)
    if result is None:
        result = {"transport": "failed", "verified": False, "z_v": None,
                  "critical": True, "detail": "the fail-safe thread did not report in time"}
        latch_not_confirmed(hub)
    finish_without_loop(hub, BOOT_PARK_REASON, full_park_result(BOOT_PARK_REASON, result))
    return result.get("verified") is True


def park_over_com(hub, reason: str, com_igor) -> dict:
    """The park of a previously connected service before its new bridge binds, on the
    fail-safe thread (the one whose COM apartment is set up). Everything goes through COM
    Execute2, the only way to Igor with no socket:

      1. `AFM_SafePark()`: td_Stop once per abort, bias 0, loop 2 re-armed to retract.
      2. Height polled over COM until under 1 V, for up to 2.5 s.
      3. Not under 1 V and no fault latched yet: the vendor Withdraw
         `DoScanFunc("StopEngageButton")` once, then Height polled again.
      4. Still not: latch a fault saying the tip is not confirmed up.

    No ABORTCONFIRM is sent (a socket verb): Igor's 5 s backup clears its own abort flag once
    it reads Height under 1 V. Returns the park result dict; never raises."""
    from ...hardware import retract
    from .safe_park import ParkResult, _park_log, _store_last_park

    result = {"transport": "com", "verified": False, "z_v": None, "critical": True,
              "retried": False, "verify_timeout": False, "com_error": None}
    if com_igor is None:
        result.update(transport="failed", com_error="no COM handle on the fail-safe thread")
    else:
        try:
            com_igor.execute("AFM_SafePark()", check=True)
        except Exception as exc:                          # noqa: BLE001
            result.update(transport="failed", com_error=str(exc))
    method = "abort"
    height = _poll_height_over_com(com_igor) if com_igor is not None else None
    # The vendor withdraw once per episode, as `_park_once`: a latched fault means it has
    # already run (or the boot park confirmed before the restart fault was latched).
    vendor_due = getattr(hub, "fault", None) is None
    if not retract.is_retracted(height) and com_igor is not None and vendor_due:
        method = "vendor_withdraw"
        try:
            retract.vendor_withdraw_over_com(com_igor)
        except Exception as exc:                          # noqa: BLE001
            result["com_error"] = str(exc)
        height = _poll_height_over_com(com_igor)
    verified = retract.is_retracted(height)
    result.update(verified=verified, critical=not verified, z_v=height, method=method)
    if verified:
        _park_log(hub, f"SAFETY safe_park: {reason} -- tip retracted over COM (verified; "
                       f"Height={height:.3f} V)")
    else:
        result["detail"] = NOT_CONFIRMED_UP
        latch_not_confirmed(hub)
        _park_log(hub, f"CRITICAL safe_park: {reason} -- {NOT_CONFIRMED_UP}; OPERATOR MUST "
                       f"INTERVENE and clear the fault")
    _store_last_park(hub, ParkResult(verified=verified, method=method, z_out_v=height,
                                     zsensor_v=None, defl_v=None, critical=not verified,
                                     reason=reason))
    return result


def _poll_height_over_com(com_igor) -> float | None:
    """Height over COM, polled until under 1 V or the park's 2.5 s settle window ends."""
    from ...hardware import retract
    from .safe_park import _VERIFY_POLL_S, _VERIFY_SETTLE_S
    end = clock.now() + _VERIFY_SETTLE_S
    while True:
        height = retract.read_height_over_com(com_igor)
        # A failed read (None) ends the poll: a broken COM will not answer the next one.
        if height is None or retract.is_retracted(height) or clock.now() >= end:
            return height
        clock.sleep(_VERIFY_POLL_S)


def latch_fault(hub) -> None:
    """The fault refuses every engage and write until the operator clears it."""
    if getattr(hub, "fault", None) is None:
        hub.fault = {"reason": "after_death", "at": clock.wall(), "state": "fault",
                     "critical": True, "detail": REASON}


def latch_not_confirmed(hub) -> None:
    """The fault for a park over COM whose Height confirm failed. Never overwrites one."""
    if getattr(hub, "fault", None) is None:
        hub.fault = {"reason": "park_unverified", "at": clock.wall(), "state": "fault",
                     "critical": True, "detail": NOT_CONFIRMED_UP}


def on_link(hub, arc) -> None:
    """Step 2, once: Igor has just dialled in, and nothing has been sent on the socket.
    Never raises."""
    global _socket_due
    if not _socket_due:
        return
    _socket_due = False
    from ...hardware.bridge_router import send_safe_park
    try:
        send_safe_park(arc.link, None)
    except Exception as exc:                              # noqa: BLE001
        hub._log_sync(f"SAFETY (WARNING) after death: the socket make-safe failed ({exc})")
        return
    hub._log_sync("SAFETY after death: the socket make-safe went out before the connect")


def reset() -> None:
    """Forget the restart state (tests)."""
    global _socket_due, _predecessor_had_bridge
    _socket_due = False
    _predecessor_had_bridge = False
