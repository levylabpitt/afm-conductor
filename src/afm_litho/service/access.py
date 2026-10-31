"""Who may command the instrument: today's access rules, one named function each.

Moved unchanged out of `control.py`. The state (`hub.control`, the revoked-token list,
`hub._control_lock`) stays on the hub; these functions are the rules that read and
swap it. Each docstring names the `A` item of docs/migration/plan/phase-2-doors.md
where today differs from the intended access model. Stop verbs never call this file.

Every change is written to the journal under src "access" with its actor; a lapse is
written by the host with its verdict (host/deadman.py). A token is never written (the
journal masks any key containing "token").
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, replace

from . import control as _c
from . import journal


def _note(what: str, actor, **body) -> None:
    journal.running().note("access", what, actor=actor, **body)


def _install(hub, client, kind, deadman_s, withdraw_on_deadman, now) -> _c.Control:
    """Install a new commander, or raise `ControlHeld`. One body shared by the script
    request and the operator take, as it was in `acquire_control`."""
    now = time.time() if now is None else now
    operator = (kind == "operator")
    with hub._control_lock:
        cur = hub.control
        if cur is not None and not cur.expired(now) and not operator:
            raise _c.ControlHeld(
                f"control is held by {cur.holder} ({cur.kind})",
                {"holder": cur.holder, "kind": cur.kind,
                 "expires_s": cur.expires_s(now)})
        if cur is not None:
            # An operator take-over is never refused; it revokes what is there.
            # An EXPIRED script token is likewise not a holder -- the watcher
            # normally clears it within a tick, and a client arriving in between
            # must not be told the instrument is held by something already gone.
            hub._remember_revoked(cur.token)
        dm = None if operator else _c.clamp_deadman(
            _c.DEADMAN_DEFAULT_S if deadman_s is None else deadman_s)
        rec = _c.Control(
            token=secrets.token_urlsafe(16),
            client=str(client or ("operator" if operator else "script")),
            kind="operator" if operator else "script",
            acquired_at=now, last_seen=now, deadman_s=dm,
            withdraw_on_deadman=bool(withdraw_on_deadman) and not operator,
            expires_at=None if operator else now + dm)
        hub.control = rec                # ONE assignment: see control.py's head
    return rec


def script_requests(hub, actor, deadman_s=None, withdraw_on_deadman=False, *,
                    now=None) -> _c.Control:
    """A script asks for access. Granted at once if free or expired; refused with
    -32040 if anyone holds unexpired access. Dead-man clamped to 5-300 s.
    A3: no grant step; a free instrument is granted at once."""
    _note("request", actor, deadman_s=deadman_s,
          withdraw_on_deadman=bool(withdraw_on_deadman))
    try:
        rec = _install(hub, actor, "script", deadman_s, withdraw_on_deadman, now)
    except _c.ControlHeld as held:
        _note("refused", actor, holder=(held.data or {}).get("holder"))
        raise
    _note("grant", actor, deadman_s=rec.deadman_s,
          withdraw_on_deadman=bool(rec.withdraw_on_deadman))
    return rec


def operator_takes(hub, client="operator", *, now=None):
    """The operator takes by hand. Revokes any script token; never refused; operator
    access never expires. Returns (what was held, the new record)."""
    prev = hub.control
    rec = _install(hub, client, "operator", None, False, now)
    _note("take", "operator", client=rec.client,
          from_holder=prev.holder if prev is not None else None)
    return prev, rec


def operator_tip_action(hub, route: str) -> _c.Control | None:
    """The automatic take on the six HTTP tip routes, run before the handler even if
    the request is then refused. Only a SCRIPT token is revoked. Returns it, or None.
    A1: the intended model has the operator take by hand. A2: fires even when refused."""
    with hub._control_lock:
        cur = hub.control
        if cur is None or cur.kind != "script":
            return None
        hub._remember_revoked(cur.token)
        hub.control = None
    _note("auto_take", "operator", route=route, from_holder=cur.holder)
    return cur


def release(hub, token, *, actor=None) -> dict:
    """Give up access. NEVER raises. Only the matching current token releases;
    anything else is `stale` and changes nothing. HTTP passes the holder's token (C4).
    The journal names `actor` (the operator's button passes "operator"), else the holder.
    A4: no hand-back; after a release the old token is revoked and the script must
    request again."""
    with hub._control_lock:
        cur = hub.control
        if cur is None or not isinstance(token, str) or token != cur.token:
            return {"ok": True, "stale": True}
        hub._remember_revoked(cur.token)
        hub.control = None
    _note("release", actor or cur.holder, holder=cur.holder)
    return {"ok": True, "stale": False}


def check(hub, token, *, now=None) -> _c.Control:
    """The gate for every token-bearing verb. -32011 if the token was ours and is
    revoked or expired; -32030 if this service never issued it."""
    cur = hub.control
    if isinstance(token, str) and token and cur is not None and token == cur.token:
        if cur.expired(now):
            raise _c.ControlRevoked("control expired (dead-man)",
                                    {"reason": "deadman", "valid": False})
        return cur
    if isinstance(token, str) and token and token in hub._revoked_tokens:
        raise _c.ControlRevoked(
            "control was revoked (preempted or expired)",
            {"reason": "revoked", "valid": False,
             "holder": cur.holder if cur is not None else None})
    raise _c.ControlRequired(
        "this call needs the commander token; call acquireControl first",
        {"valid": False,
         "holder": cur.holder if cur is not None else None})


def refresh(hub, token, *, now=None) -> bool:
    """Dead-man refresh, on a heartbeat and on any command-socket request that carries
    the token. False (not an error) when the token is not the current holder's. An
    already-expired record is never refreshed: expiry is the timer's to act on."""
    now = time.time() if now is None else now
    with hub._control_lock:
        cur = hub.control
        if cur is None or not isinstance(token, str) or token != cur.token:
            return False
        if cur.expired(now):
            return False
        hub.control = replace(
            cur, last_seen=now,
            expires_at=None if cur.deadman_s is None else now + cur.deadman_s)
    return True


@dataclass(frozen=True)
class Lapse:
    """Whose token lapsed, and whether they asked to be withdrawn if it did."""

    holder: str
    withdraw_on_deadman: bool


def lapsed(hub, now) -> Lapse | None:
    """The dead-man's base half. A script's token has expired: release it FIRST and say
    whose it was. The host then reads the tip's state, acts and journals the lapse with
    its verdict (host/deadman.commander_lapsed). None when there is no lapse, or the
    record changed under us. Operator access never lapses.
    A5: while engaged the tip stays down unless the script asked for withdraw."""
    cur = hub.control
    if cur is None or cur.deadman_s is None or not cur.expired(now):
        return None
    # RELEASE FIRST, always. Whatever the tip is doing, the commander has gone
    # silent, so the token must not survive the action that follows -- and the GUI
    # must see the countdown end even if the retract then fails.
    with hub._control_lock:
        if hub.control is not cur:            # released or preempted under us
            return None
        hub._remember_revoked(cur.token)
        hub.control = None
    return Lapse(cur.holder, bool(cur.withdraw_on_deadman))
