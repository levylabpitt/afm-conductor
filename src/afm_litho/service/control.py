"""Single-writer arbitration: the commander token, the dead-man, the engaged-idle
cap, and the run claim.

Truth document: docs/FLEX_INTEGRATION_PLAN.md Sec E.1 (the hazard-based dead-man
table plus the separate unattended-engagement cap), E.2 (one commander, GUI
preemption, and *where* the single-writer claim is taken), B.4 (the deadlock rule),
F.3 (the state machine). Wire surface: `flex_dispatch.py` / docs/FLEX_PROVIDER.md.

Who may command the instrument (request, take, release, check, refresh, lapse) lives in
`access.py`; the hub methods below only call it.

FOUR THINGS LIVE HERE, and they are deliberately separate:

1. **`Control`** -- the commander record. ONE commander at a time, script or
   operator. It is a FROZEN dataclass published by plain attribute assignment
   (`hub.control = record`), so ANY thread may read it with no lock: a reader sees
   either the whole old record or the whole new one, never a half-updated one. That
   is what lets a REP thread answer `getStatus` while the loop is mutating
   arbitration state, without the lock plan Sec B.4 forbids.

2. **The dead-man** (`control_tick`) -- an expiry timer whose action is graded by
   what the tip is DOING, per the plan Sec E.1 table. This file releases the expired
   token; the host reads the tip's state and acts (`host/deadman.py`, which holds the
   table and why silence while merely `engaged` does NOT withdraw by default).

3. **The engaged-idle cap** (`host/deadman.engaged_idle_max_s`) -- independent of any
   token, and NOT disableable. "A script stopped talking" (dead-man, gentle) and "a tip has been
   left on a surface unattended" (this, decisive) are different failures; neither
   timer alone covers both, which is why the plan's earlier single-timer draft was
   wrong.

4. **The run claim** (`claim_run`) -- the single-writer claim, taken SYNCHRONOUSLY
   before the first `await` of a start path. `POST /api/litho/write` checked
   `litho_thread.is_alive()` and then awaited the AC-guard crosspoint read before
   starting the thread, so two interleaved requests both passed the check and both
   proceeded. A script issues requests faster than a human can click, so FLEX makes
   that pre-existing race easy to hit.

THREADING RULES, stated once:

  * READS of `hub.control` are lock-free from any thread (see 1).
  * WRITES take `hub._control_lock` for the two or three statements that swap the
    record, and NEVER hold it across an `await` or across
    `run_coroutine_threadsafe(...).result()`. That is what plan Sec B.4's deadlock
    rule actually forbids -- a REP thread parked on the loop while holding a lock the
    loop needs. A lock held only over a compare-and-swap cannot deadlock, and it is
    what closes the one real interleaving: a heartbeat refresh landing on a REP
    thread just as the loop releases the token would otherwise RESURRECT a released
    commander.
  * `claim_run` is synchronous ON THE LOOP: check and set with no `await` between
    them, which is the entire point of it.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, replace

from .. import clock

# --------------------------------------------------------------- error codes --
# The arbitration slice of the provider's error vocabulary (docs/FLEX_PROVIDER.md).
# Kept HERE, beside the state they describe, and imported by flex_dispatch so there
# is one definition rather than two that can drift.
CONTROL_REVOKED = -32011      # the token existed and was preempted or expired
CONTROL_REQUIRED = -32030     # no token, or one this service has never issued
CONTROL_HELD = -32040         # someone else is the commander right now

# `deadman_s` bounds (plan Sec E.1: default 30 s, extendable to 300 s, never
# disabled). A request outside the range is CLAMPED, not refused -- a script that
# asks for 0 gets the floor, which is the safe direction.
DEADMAN_DEFAULT_S = 30.0
DEADMAN_MIN_S = 5.0
DEADMAN_MAX_S = 300.0

# How often the watcher looks. One second is fine for both timers -- the shortest
# thing either can measure is 5 s -- and it is what bounds shutdown latency.
WATCH_INTERVAL_S = 1.0

# How many revoked tokens are remembered, so a script that was preempted is told
# `control_revoked` (actionable: a human took over) rather than `control_required`
# (which reads like it never acquired). Bounded, because this is a courtesy, not
# state anything depends on.
_REVOKED_KEEP = 32

# The CURATED tip-mutating path set (plan Sec E.2). Reaching for one of these from
# the GUI preempts a script token, because a human who clicks them has already
# decided. Each is a POST row of the request table whose gate takes access in `dispatch`
# (table.py refuses to import otherwise). EXACT paths, never a blanket "any POST": the cockpit POSTs UI preferences
# on a four-second pointer-dwell debounce, so a bystander moving the mouse across
# the panel would otherwise revoke an overnight token.
#
# `/api/safe_park`, `/api/scan/abort` and `/api/litho/abort` are deliberately ABSENT:
# they are token-free from both sides, and an emergency stop must not also be an
# arbitration event.
TIP_MUTATING = frozenset({
    "/api/approach", "/api/scan/start", "/api/litho/write", "/api/withdraw",
    "/api/mode", "/api/scan/continuous",
})


def clamp_deadman(value) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return DEADMAN_DEFAULT_S
    return max(DEADMAN_MIN_S, min(DEADMAN_MAX_S, val))


# ------------------------------------------------------------- the exceptions --

class ControlError(Exception):
    """Base for the arbitration refusals. `code` is the JSON-RPC code the dispatcher
    renders; the HTTP side reports the same vocabulary in its body."""

    code = CONTROL_REQUIRED

    def __init__(self, message: str, data=None):
        super().__init__(message)
        self.message = str(message)
        self.data = data


class ControlRequired(ControlError):
    """No token, or a token this service has never issued (-32030)."""

    code = CONTROL_REQUIRED


class ControlRevoked(ControlError):
    """A token that WAS this service's and is not any more: preempted by the
    operator, or expired by the dead-man (-32011)."""

    code = CONTROL_REVOKED


class ControlHeld(ControlError):
    """Someone else is the commander (-32040). `data` names the holder, so a script
    can log WHO it is waiting for instead of retrying blind."""

    code = CONTROL_HELD


class RunBusy(Exception):
    """A start path arrived while another one holds the claim (`-32020 busy` on the
    wire, `{"error": "busy", "run_id": ...}` over HTTP)."""

    def __init__(self, kind: str | None, run_id: str | None):
        super().__init__(f"busy: a {kind or 'run'} is already starting or running")
        self.kind = kind
        self.run_id = run_id


# ---------------------------------------------------------------- the records --

@dataclass(frozen=True)
class Control:
    """The commander. FROZEN on purpose: every mutation replaces the whole record in
    one attribute assignment, so a reader on another thread can never observe a
    half-updated commander (see the module docstring)."""

    token: str
    client: str
    kind: str                     # "script" | "operator"
    acquired_at: float
    last_seen: float
    deadman_s: float | None       # None for an operator: a human is present
    withdraw_on_deadman: bool
    expires_at: float | None      # None for an operator

    def expires_s(self, now: float | None = None) -> float | None:
        if self.expires_at is None:
            return None
        return max(0.0, self.expires_at - (time.time() if now is None else now))

    def expired(self, now: float | None = None) -> bool:
        if self.expires_at is None:
            return False
        return (time.time() if now is None else now) >= self.expires_at

    @property
    def holder(self) -> str:
        """What the GUI chip and the `control` ws event show: the script's own name,
        or the bare string "operator"."""
        return "operator" if self.kind == "operator" else self.client

    def to_wire(self, now: float | None = None) -> dict:
        return {"holder": self.holder, "client": self.client, "kind": self.kind,
                "expires_s": self.expires_s(now), "deadman_s": self.deadman_s,
                "withdraw_on_deadman": bool(self.withdraw_on_deadman),
                "acquired_at": self.acquired_at, "last_seen": self.last_seen}


class RunClaim:
    """The single-writer claim (plan Sec E.2). Identity IS the claim: `release_claim`
    clears the slot only when it still holds THIS object, so a late release from a
    finished run can never free a claim a newer run has taken."""

    __slots__ = ("at", "kind", "run_id", "worker")

    def __init__(self, kind: str, at: float):
        self.kind = kind
        self.at = at
        self.run_id: str | None = None      # stamped once the id is minted
        # The thread or task that owns this claim, stamped at hand-over by
        # `hand_over`. Liveness is asked of THIS object -- see `_claim_is_stale`.
        self.worker = None

    def hand_over(self, run_id: str | None, worker) -> None:
        """Give the claim to the worker that will run the pass.

        ONE call taking both facts at once, because they must become true together:
        until it is handed over the claim is "held by the starting coroutine" and is
        never stale, and from then on its validity is judged by `worker`. Stamping
        the id without the worker is what let a not-yet-started thread read as
        finished.
        """
        self.run_id = run_id
        self.worker = worker

    def worker_alive(self) -> bool:
        """Is the owning worker still going?

        A `Thread` that was created but never started answers False to `is_alive()`
        and True here: it has not FINISHED, and a claim must never be reclaimed from
        a worker that has not run yet."""
        w = self.worker
        if w is None:
            return False
        is_alive = getattr(w, "is_alive", None)
        if is_alive is not None:                     # threading.Thread
            started = getattr(w, "_started", None)
            if started is not None and not started.is_set():
                return True                          # created, not started yet
            return bool(is_alive())
        done = getattr(w, "done", None)
        if done is not None:                         # asyncio.Task / Future
            return not done()
        return True

    def __repr__(self) -> str:              # pragma: no cover - debugging aid
        return f"RunClaim({self.kind!r}, run_id={self.run_id!r})"


# ------------------------------------------------------------------ the mixin --

class ControlMixin:
    """Arbitration on the hub. Mixed into `Hub` rather than written inline there so
    the state and the rules that govern it read as one thing."""

    # Declared here for the reader; `Hub.__init__` sets them (a mixin must not own
    # __init__ or the hub's own field declarations stop being the single place to
    # look).
    control: Control | None
    _control_lock: threading.Lock
    _revoked_tokens: list
    _run_claim: RunClaim | None
    control_watch: object
    _control_bg_tasks: set

    def _init_control(self) -> None:
        # Guards the WRITE side only, over a handful of statements at a time; see the
        # threading rules in the module docstring for why this is NOT the lock plan
        # Sec B.4 forbids.
        self._control_lock = threading.Lock()
        self.control = None
        self._revoked_tokens = []          # bounded FIFO of tokens we DID issue
        self._run_claim = None
        self.control_watch = None          # the watcher task, while the app runs
        # Fire-and-forget `control` re-broadcasts (spawn_control_broadcast), kept
        # here so nothing gets GC'd mid-flight -- see that function.
        self._control_bg_tasks = set()

    # ------------------------------------------------------------ the token --

    def _remember_revoked(self, token: str) -> None:
        self._revoked_tokens.append(token)
        if len(self._revoked_tokens) > _REVOKED_KEEP:
            del self._revoked_tokens[:-_REVOKED_KEEP]

    def control_wire(self, now: float | None = None) -> dict:
        """The `control` ws event / `GET /api/control` body. Null holder means the
        instrument is unclaimed -- which is the normal state for a GUI session."""
        cur = self.control
        if cur is None:
            return {"holder": None, "client": None, "kind": None,
                    "expires_s": None, "deadman_s": None,
                    "withdraw_on_deadman": False}
        return cur.to_wire(now)

    def acquire_control(self, client: str, kind: str = "script", *,
                        deadman_s=None, withdraw_on_deadman: bool = False,
                        now: float | None = None) -> Control:
        """Kept for callers; the rule is `access.script_requests` (or `operator_takes` for kind operator)."""
        from . import access
        if kind == "operator":
            return access.operator_takes(self, client, now=now)[1]
        return access.script_requests(self, client, deadman_s, withdraw_on_deadman,
                                      now=now)

    def release_control(self, token, *, now: float | None = None) -> dict:
        """Kept for callers; the rule is `access.release`."""
        from . import access
        return access.release(self, token)

    def rearm_deadman(self, token, seconds, *, now: float | None = None) -> float | None:
        """Change the commander's dead-man window and restart it. Returns the window
        actually armed, or None when `token` is not the current commander's.

        `startWrite {deadman_s}` (plan Sec A) is why this exists: a write is minutes
        long, and a script that had to re-acquire control to lengthen its window would
        be re-acquiring on the one path where dropping the token for an instant is
        worst. CLAMPED, never disabled -- `clamp_deadman` is the same [5, 300] rule
        `acquireControl` applies, so there is one bound, not two.

        An OPERATOR record is left alone: a human at the panel has no dead-man, and a
        script must not be able to give one.
        """
        now = time.time() if now is None else now
        with self._control_lock:
            cur = self.control
            if cur is None or not isinstance(token, str) or token != cur.token:
                return None
            if cur.kind == "operator" or cur.deadman_s is None:
                return None
            if cur.expired(now):
                return None       # expiry is the watcher's to act on, not ours to undo
            dm = clamp_deadman(seconds)
            self.control = replace(cur, deadman_s=dm, last_seen=now,
                                   expires_at=now + dm)
        return dm

    def touch_control(self, token, *, now: float | None = None) -> bool:
        """Kept for callers; the rule is `access.refresh`."""
        from . import access
        return access.refresh(self, token, now=now)

    def require_token(self, token, *, now: float | None = None) -> Control:
        """Kept for callers; the rule is `access.check`."""
        from . import access
        return access.check(self, token, now=now)

    # ------------------------------------------------- operator preemption --

    async def take_control(self, *, client: str = "operator",
                           reason: str | None = None) -> dict:
        """The GUI's "Take control": revoke any script token and become the
        commander. Plan Sec E.2's normal path -- a human deliberately taking the
        instrument back, and SEEING that they did."""
        from . import access
        prev, rec = access.operator_takes(self, client)
        if prev is not None and prev.kind == "script":
            await self.log(f"Control: operator took control from {prev.holder}"
                           + (f" ({reason})" if reason else ""))
        await self.broadcast_control()
        return {"ok": True, "control": rec.to_wire(),
                "preempted": (prev.holder if prev is not None
                              and prev.kind == "script" else None)}

    async def preempt_for_operator(self, reason: str) -> bool:
        """A curated tip-mutating HTTP route was hit: the operator wins, always.

        Only a SCRIPT token is preempted, and only when one is actually held -- so
        an ordinary GUI session (no commander at all) never touches arbitration, and
        neither does an operator who already holds it."""
        from . import access
        cur = access.operator_tip_action(self, reason)
        if cur is None:
            return False
        await self.announce_revoked(cur, reason)
        return True

    async def announce_revoked(self, prev: Control, route: str) -> None:
        """Tell the browser (log line and control frame) that the operator revoked `prev`."""
        await self.log(f"Control: {prev.holder}'s token REVOKED -- the operator used "
                       f"{route} from the panel")
        await self.broadcast_control()

    def spawn_revoked_notice(self, prev: Control, route: str) -> None:
        """`announce_revoked` as a background task: the Withdraw has already run, and a
        stalled browser window (`ws.send_text` has no timeout) must not hold it up (F46)."""
        task = asyncio.get_running_loop().create_task(self.announce_revoked(prev, route))
        self._control_bg_tasks.add(task)

        def _done(t: asyncio.Task, hub=self) -> None:
            hub._control_bg_tasks.discard(t)
            if not t.cancelled() and t.exception() is not None:   # pragma: no cover
                hub._log_sync(f"control watch: revoked notice failed ({t.exception()!r})")

        task.add_done_callback(_done)

    async def broadcast_control(self) -> None:
        """One ws frame per arbitration change; the cockpit chip is driven by it."""
        try:
            await self.broadcast({"type": "control", "t": time.time(),
                                  "deadman_armed": deadman_armed(self),
                                  **self.control_wire()})
        except Exception:                                  # noqa: BLE001,S110
            pass

    def spawn_control_broadcast(self) -> None:
        """Fire `broadcast_control()` WITHOUT waiting for it.

        `hub.broadcast` is a sequential `await ws.send_text(...)` per client with no
        per-send timeout (a separate, pre-existing concern -- not this fix). A caller
        on a bounded reply budget -- `heartbeat` has a 1 s loop budget -- must not
        `await` that fan-out: one stalled client would make the reply late, which is
        the exact failure this re-broadcast exists to fix.

        The task is kept in `_control_bg_tasks` until it finishes so nothing GC's it
        mid-flight (a bare `create_task` result with no reference is eligible for
        collection before it runs). `broadcast_control` already swallows its own
        exceptions, so the done-callback's `except` is a second net, not the first.
        """
        task = asyncio.get_running_loop().create_task(self.broadcast_control())
        self._control_bg_tasks.add(task)

        def _done(t: asyncio.Task, hub=self) -> None:
            hub._control_bg_tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:                            # pragma: no cover
                hub._log_sync(f"control watch: background control broadcast "
                              f"failed ({exc!r})")

        task.add_done_callback(_done)

    # ------------------------------------------------------- the run claim --

    def claim_run(self, kind: str) -> RunClaim:
        """Take the single-writer claim, SYNCHRONOUSLY (plan Sec E.2).

        There is no `await` between the check and the set, which is the whole
        property: `POST /api/litho/write` checks `litho_thread.is_alive()`, then
        awaits the AC-guard crosspoint read, and only then starts the thread -- so
        without this, two requests interleaving in that window BOTH passed the check
        and both proceeded to drive the tip.

        Deliberately narrow: it arbitrates the START paths against each other and
        nothing else. It does NOT consult `run_in_flight()`, which reports True for
        the whole life of a continuous loop -- claiming on that would refuse the
        loop's own next frame.
        """
        held = self._run_claim
        if held is not None and not self._claim_is_stale(held):
            raise RunBusy(held.kind, held.run_id or self._in_flight_run_id())
        claim = RunClaim(str(kind), time.time())
        self._run_claim = claim
        return claim

    def _claim_is_stale(self, claim) -> bool:
        """True when a HANDED-OVER claim's OWN worker is demonstrably finished.

        The claim is released by the pump that owns it -- but a `finally` is a
        promise the interpreter keeps only if the coroutine runs again, and a pump
        task whose event loop is torn down under it (a TestClient closing, a process
        quiescing mid-run) never gets that chance. A claim stuck that way would
        refuse every later start for the life of the process: a wedge, caused by
        bookkeeping, on the path that drives the tip.

        So the claim's authority is anchored to its run's own liveness. Two rules,
        and the first is the safety-critical one:

        * NOT handed over (`worker is None`) -- the claim is held by a coroutine
          between the busy check and the thread start, which is exactly the window
          the claim exists for, so it is NEVER stale. An earlier version asked
          `hub.litho_thread.is_alive()` instead, which answers False for a thread
          that has been CREATED but not STARTED -- so a second writer could have
          taken the claim in that gap, and only the statement ordering in two files
          stood between that and two writes driving the tip.
        * handed over -- ask THIS claim's worker, never a hub field that a newer run
          may already have replaced.
        """
        if claim.worker is None:
            return False
        return not claim.worker_alive()

    def release_claim(self, claim) -> None:
        """Release, by IDENTITY. A no-op when the slot has moved on, so a late
        release from a finished run cannot free a newer run's claim. Never raises:
        it lives in `finally` blocks."""
        if claim is not None and self._run_claim is claim:
            self._run_claim = None

    def release_run(self, run_id) -> None:
        """Release the claim held by `run_id` (the run-id twin of `release_claim`). A no-op
        when the slot is empty or belongs to another run. Never raises."""
        claim = self._run_claim
        if run_id is not None and claim is not None and claim.run_id == run_id:
            self._run_claim = None

    def _in_flight_run_id(self) -> str | None:
        try:
            rec = self.runs.in_flight()
        except Exception:                                  # noqa: BLE001
            return None
        return rec.run_id if rec is not None else None

    def busy_body(self, busy: RunBusy) -> dict:
        """The HTTP shape of a refused claim, matching the wire's `-32020 busy`
        (data: run_id) so a mixed GUI/script session reads one vocabulary."""
        return {"error": "busy", "reason": "busy", "kind": busy.kind,
                "run_id": busy.run_id or self._in_flight_run_id()}


# ------------------------------------------------------------- the dead-man ----

async def control_tick(hub, *, now: float | None = None) -> list[dict]:
    """One pass of both timers. Returns what it did (a list of action records), so a
    test can assert on the OUTCOME rather than on a sleep.

    Order matters: the dead-man is applied first, because an expiry that withdraws
    also ends the engagement the idle cap is counting. That is why the host's cap
    (`host/deadman.engaged_idle_tick`) is still driven from this tick, after the dead-man.
    """
    now = time.time() if now is None else now
    actions: list[dict] = []
    try:
        act = await _deadman_tick(hub, now)
        if act is not None:
            actions.append(act)
    except Exception as exc:                               # noqa: BLE001
        # A timer that raises must not stop the OTHER timer, and must not kill the
        # watcher: a dead watcher is a tip with no dead-man at all.
        await hub.log(f"control watch: dead-man tick failed ({exc})")
    try:
        from .host import deadman
        act = await deadman.engaged_idle_tick(hub, now)
        if act is not None:
            actions.append(act)
    except Exception as exc:                               # noqa: BLE001
        await hub.log(f"control watch: engaged-idle tick failed ({exc})")
    return actions


async def _deadman_tick(hub, now: float) -> dict | None:
    """Plan Sec E.1's dead-man. The base half: a script's token has expired, so release
    it (`access.lapsed`) and tell the pages. The host half reads what the tip is doing
    and acts (`host/deadman.commander_lapsed`, which holds the table).

    Operator control has NO dead-man (the human is present), so it never lands here.
    """
    from . import access
    from .host.api import HostApi

    lapse = access.lapsed(hub, now)
    if lapse is None:
        return None
    # Called before the `control` frame's await, so the host reads the tip's state in
    # the same synchronous run as the release.
    try:
        act = HostApi(hub).commander_lapsed(lapse.holder, lapse.withdraw_on_deadman)
    except Exception:
        # The token is already released: the pages must still see the countdown end.
        await hub.broadcast_control()
        raise
    try:
        await hub.broadcast_control()
    except BaseException:
        act.close()                    # cancelled before acting: nothing was sent
        raise
    return await act


async def control_watch_loop(hub, *, interval: float = WATCH_INTERVAL_S) -> None:
    """The watcher task: started with the app, cancelled on shutdown.

    OWNED ON THE LOOP (plan Sec B.3/B.4). The REP threads never run it, so the
    timers keep running even if a provider thread dies -- and the retract each one
    fires goes through `safe_park`, which hands the hardware retract to the failsafe
    watchdog THREAD, so it does not depend on this loop being healthy either.
    """
    try:
        while True:
            # A background periodic loop: stays on real time on every clock mode, or a
            # virtual clock would run the dead-man's expiry away.
            await asyncio.sleep(interval)
            await control_tick(hub)
    except asyncio.CancelledError:
        raise
    except Exception as exc:                               # noqa: BLE001
        # Unreachable in practice (control_tick swallows its own failures); if it
        # ever is reached, say so LOUDLY -- a silent dead watcher is a tip with no
        # dead-man.
        await hub.log(f"CRITICAL: the control watcher STOPPED ({exc}). The dead-man "
                      f"and the engaged-idle limit are no longer running -- restart "
                      f"the service.")


# How long to wait before re-arming a watcher that died, and how far to back off if
# it keeps dying. Bounded, because a watcher that fails instantly in a tight restart
# loop would bury the CRITICAL line that says so.
REARM_DELAY_S = 1.0
REARM_MAX_DELAY_S = 30.0


def deadman_armed(hub) -> bool:
    """Are the timers actually running?

    Published in `getStatus` (and in `commander`) because the failure this reports is
    otherwise INVISIBLE: a dead watcher looks exactly like a healthy one with nothing
    to do, and an unattended script would go on driving a tip whose dead-man stopped
    hours ago. A status field that can say "no" is what makes that detectable from
    the outside.
    """
    task = getattr(hub, "control_watch", None)
    return task is not None and not task.done()


def start_control_watch(hub, *, interval: float = WATCH_INTERVAL_S,
                        rearm_delay: float = REARM_DELAY_S):
    """Start the watcher if it is not already running, and make it SELF-HEALING.

    A done-callback re-arms it whenever it ends for any reason other than a
    deliberate `stop_control_watch` (the shutdown path clears `hub.control_watch`
    first, so a cancellation there is recognised as intentional and left alone).
    Each re-arm logs CRITICAL and backs off, so a watcher that keeps dying is loud
    and bounded rather than a tight restart loop that buries its own message.

    Idempotent: calling it twice returns the running task.
    """
    task = getattr(hub, "control_watch", None)
    if task is not None and not task.done():
        return task
    delay = max(0.0, min(REARM_MAX_DELAY_S, rearm_delay))

    def _on_done(finished) -> None:
        if getattr(hub, "control_watch", None) is not finished:
            return                     # replaced or deliberately stopped: not ours
        hub.control_watch = None
        if finished.cancelled():
            return                     # shutdown, or an explicit stop
        exc = None
        try:
            exc = finished.exception()
        except Exception:                                  # noqa: BLE001,S110
            pass
        hub._log_sync(
            f"CRITICAL: the control watcher STOPPED ({exc!r}). The dead-man and the "
            f"engaged-idle limit are NOT running; re-arming in {delay:g}s.")

        async def _rearm():
            await clock.async_sleep(delay)
            if getattr(hub, "control_watch", None) is None:
                start_control_watch(
                    hub, interval=interval,
                    rearm_delay=min(REARM_MAX_DELAY_S,
                                    delay * 2 if delay else REARM_DELAY_S))

        try:
            asyncio.get_running_loop().create_task(_rearm())
        except RuntimeError:                               # no loop: nothing to do
            pass

    hub.control_watch = asyncio.create_task(
        control_watch_loop(hub, interval=interval), name="afm-control-watch")
    hub.control_watch.add_done_callback(_on_done)
    return hub.control_watch


def stop_control_watch(hub) -> None:
    """Cancel the watcher. Safe to call twice, and safe when it never started.

    Clears `hub.control_watch` BEFORE cancelling, which is how the done-callback
    tells a deliberate stop from a death worth re-arming."""
    task = getattr(hub, "control_watch", None)
    hub.control_watch = None
    if task is not None and not task.done():
        task.cancel()
