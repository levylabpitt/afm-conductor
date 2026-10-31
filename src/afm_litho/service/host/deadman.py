"""What the tip is doing, and what the dead-man and the engaged-idle cap do about it.

Plan Sec E.1. The base keeps the commander record: token expiry, the holder check, the
release and the `control` frame (service/access.py, service/control.py). When a script's
token lapses the base calls `commander_lapsed`, which reads the tip's state here and acts:

| state at the lapse                         | action                          |
| approaching/scanning/writing/measuring     | abort + safe_park -> `parked`   |
| engaged, nothing in flight                 | withdraw ONLY if asked, else warn |
| idle / parked / anything else              | nothing (the token is released) |

A reflex withdraw is itself a disruption (it abandons an engagement the operator may be
re-using, and on a Cypher it can wedge the machine, plan risk 7), so a silent script
leaves an engaged tip down unless it asked otherwise. The engaged-idle cap is what bounds
that: a tip engaged with nothing running is withdrawn after `engaged_idle_max_s()`,
whoever holds control.

`commander_lapsed` takes no holder check, so a base link close (H8c) can call it directly.
"""

from __future__ import annotations

import os

from .. import journal

# The unattended-engagement cap (plan Sec E.1). Non-optional: the environment may only
# TIGHTEN it, never loosen it past the default and never disable it.
ENGAGED_IDLE_MAX_DEFAULT_S = 900.0
ENGAGED_IDLE_MIN_S = 1.0

PASS_STATES = ("approaching", "scanning", "writing", "measuring")


def engaged_idle_max_s() -> float:
    """The engaged-idle cap in seconds. `AFM_LITHO_ENGAGED_IDLE_MAX_S` may only
    TIGHTEN it: a value above the default is clamped down to the default, and a
    value that is absent, unparseable or non-positive leaves the default in place.
    There is no way to switch this timer off, which is the whole point of it."""
    raw = os.environ.get("AFM_LITHO_ENGAGED_IDLE_MAX_S")
    if raw is None:
        return ENGAGED_IDLE_MAX_DEFAULT_S
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return ENGAGED_IDLE_MAX_DEFAULT_S
    if val <= 0:
        return ENGAGED_IDLE_MAX_DEFAULT_S
    return max(ENGAGED_IDLE_MIN_S, min(ENGAGED_IDLE_MAX_DEFAULT_S, val))


# ------------------------------------------------------------ the tip's state --

def pass_state(hub) -> str:
    """The state machine (plan Sec F.3). First match wins; the rows and the reasons for
    their order are the getState table in service/flex_dispatch.py's docstring. Pure
    in-memory: no bridge read, ever.

    Two inputs are base state today: `hub.backend_live()` and the run registry read by
    `write_in_flight`. Dropping either would change a verdict; both are on the H8a list."""
    host = hub.host
    if getattr(host, "fault", None) is not None:
        return "fault"
    if getattr(host, "backend", None) == "hardware" and hub.backend_live() != "hardware":
        return "disconnected"
    if write_in_flight(hub):
        return "writing"
    if getattr(host, "_approach_in_flight", 0):
        return "approaching"
    task = getattr(host, "_scan_task", None)
    if task is not None and not task.done():
        return "scanning"
    if (getattr(host, "scan_repeat", None) or {}).get("enabled"):
        return "scanning"
    if (getattr(host, "_safe_park", None) or {}).get("active"):
        return "parked"
    if getattr(host, "engaged", False):
        return "engaged"
    return "idle"


def write_in_flight(hub) -> bool:
    """Is a write still going, persistence included?

    The run record first: it stays open until the pump has saved the run, so the state
    does not leave `writing` while the .npz/.gwy/.ibw are written (plan Sec A.1 gates
    `completed` on persistence). Not the claim, which is released at pass end, before
    the save. The worker thread is the fallback for the moment before the record is
    allocated, and for a pump driven directly (tests).
    """
    try:
        rec = hub.runs.in_flight()
        if rec is not None and rec.kind == "write":
            return True
    except Exception:                                # noqa: BLE001,S110
        pass
    thread = getattr(hub.host, "litho_thread", None)
    return thread is not None and thread.is_alive()


# ------------------------------------------------------------- the dead-man --

def lapse_action(state: str, withdraw_on_deadman: bool) -> str:
    """The table at the top of this file: "safe_park" | "withdraw" | "warn" | "release"."""
    if state in PASS_STATES:
        return "safe_park"
    if state == "engaged":
        return "withdraw" if withdraw_on_deadman else "warn"
    return "release"


def commander_lapsed(hub, holder: str, withdraw_on_deadman: bool):
    """A commander went silent and its token is already released. Reads the tip's state
    and journals the verdict now, then returns the awaitable that logs and acts; it gives
    `{action, state, holder, ...}`.

    The state is read in this call, not in the awaitable: the base releases the token and
    calls this in one synchronous run, so the state is the one at the release, before the
    base awaits its `control` frame."""
    state = pass_state(hub)
    action = lapse_action(state, withdraw_on_deadman)
    # The commander's journal entry (src "access"), with the verdict it got.
    journal.running().note("access", "lapse", actor=holder, state=state, action=action)
    return _act_on_lapse(hub, holder, state, action)


async def _act_on_lapse(hub, holder: str, state: str, action: str) -> dict:
    if action == "safe_park":
        await hub.log(f"DEAD-MAN: {holder} went silent while {state} -- aborting and "
                      f"safe-parking (verified retract)")
        from . import safe_park
        res = await safe_park.request_park(hub, "deadman")
        return {"action": "safe_park", "state": state, "holder": holder,
                "safe_park": res}
    if action == "withdraw":
        await hub.log(f"DEAD-MAN: {holder} went silent while engaged -- "
                      f"withdrawing (withdraw_on_deadman was set)")
        snap = await _withdraw(hub)
        if snap.get("not_connected"):
            return {"action": "withdraw", "state": state, "holder": holder,
                    "not_connected": True}
        return {"action": "withdraw", "state": state, "holder": holder}
    if action == "warn":
        # The default: warn and leave the tip (see the top of this file). The
        # engaged-idle cap below bounds it.
        await hub.log(f"WARNING DEAD-MAN: {holder} went silent while ENGAGED. The "
                      f"token is released and the tip is LEFT ON THE SURFACE "
                      f"(withdraw_on_deadman was not set). The engaged-idle limit "
                      f"({engaged_idle_max_s():g}s) still applies.")
        return {"action": "warn", "state": state, "holder": holder}
    await hub.log(f"DEAD-MAN: {holder} went silent while {state} -- token released, "
                  f"no instrument action")
    return {"action": "release", "state": state, "holder": holder}


async def _withdraw(hub) -> dict:
    """The plain withdraw both timers use, the same call as `routers/scan.do_withdraw`:
    stop the scan; on hardware re-arm the Z loop to drive Height to 0; the simulated
    withdraw on the twin."""
    from . import safe_park
    snap = await safe_park.request_park(hub, "withdraw", safe_park.PARK_WITHDRAW, body={})
    return snap if isinstance(snap, dict) else {}


# ------------------------------------------------------- the engaged-idle cap --

async def engaged_idle_tick(hub, now: float) -> dict | None:
    """The unattended-engagement cap (plan Sec E.1), independent of any token.

    `hub.host._engaged_since` is "engaged with nothing in flight, since". This tick is
    its only writer: any state other than `engaged` resets it, so a long scan never
    counts toward the cap.
    """
    host = hub.host
    if pass_state(hub) != "engaged":
        host._engaged_since = None
        return None
    if host._engaged_since is None:
        host._engaged_since = now
        return None
    limit = engaged_idle_max_s()
    if (now - host._engaged_since) < limit:
        return None
    host._engaged_since = None
    await hub.log(f"ENGAGED-IDLE LIMIT: the tip has been engaged with nothing "
                  f"running for over {limit:g}s -- withdrawing. This limit is not "
                  f"optional and applies whether or not anyone holds control.")
    snap = await _withdraw(hub)
    if snap.get("not_connected"):
        # Nothing was sent, so no withdraw is announced.
        return {"action": "engaged_idle_withdraw", "limit_s": limit, "not_connected": True}
    await hub.broadcast({"type": "engaged_idle_withdraw", "t": now,
                         "limit_s": limit})
    return {"action": "engaged_idle_withdraw", "limit_s": limit}
