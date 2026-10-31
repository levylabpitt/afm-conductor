"""The one place a request is run: journal the request, run the handler, journal the reply.

Steps 1, 2, 4, 5 and 6 of the plan are here; step 3 (the dead-man refresh) is still in the FLEX
door. Step 2 is the row's declared gate: "token" calls `access.check`, "operator_tip" is the
automatic operator take, anything else does nothing. Every request gets a trace number and an
actor label; nothing is refused because of the label.

A stop row runs its handler first and journals afterwards, so a journal fault cannot fail it. The
actor is resolved after the handler. Before a stop the FLEX door still refreshes the dead-man when
the request carries a token (under the control lock), and FLEX abort does so again. The browser Withdraw (gate "operator_stop", F46) also first
revokes a script's token in memory, and tells the browser afterwards in a background task.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import functools
import inspect
import itertools
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from ...hardware.bridge_protocol import wire_trace
from .. import access, journal
from .context import current_hub
from .table import Row

_log = logging.getLogger(__name__)
_traces = itertools.count(1)
# Intent waits get their own worker so they can never queue behind (or in front of) the
# hardware-safety calls on the default executor.
_intent_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="journal-intent")


def new_trace() -> int:
    """The number that ties one request's journal entries together."""
    return next(_traces)


@contextlib.contextmanager
def _wire_trace(trace: int):
    """Wire entries made while the handler runs carry this request's trace (one in-memory
    assignment; nothing waits and nothing is written)."""
    token = wire_trace.set(trace)
    try:
        yield
    finally:
        wire_trace.reset(token)


def _note_request(row: Row, actor: str, trace: int, request: Any, *,
                  start: bool = False) -> None:
    """Journal the request. `start` is set only on the accepted path just before the handler of
    an intent row, where the entry is fsynced (capped at 1 s) first. Refusals and stop rows use
    plain `note`: a stop journals after its handler and never waits on the disk."""
    if not row.journalled:
        return
    running = journal.running()
    note = running.note_intent if start and row.intent and row.control != "stop"         else running.note
    note("door", "request", actor=actor, trace=trace, row=row.name, params=request)


def _note_reply(row: Row, actor: str, trace: int, result: Any) -> None:
    if not row.journalled:
        return
    journal.running().note("door", "reply", actor=actor, trace=trace, row=row.name,
                           result=result)


def _note_error(row: Row, actor: str, trace: int, exc: BaseException) -> None:
    if not row.journalled:
        return
    journal.running().note("door", "error", actor=actor, trace=trace, row=row.name,
                           error=f"{type(exc).__name__}: {exc}",
                           code=getattr(exc, "code", None))


def _safely(note, row, actor, trace, value) -> None:
    """After a stop's handler has run, a journal fault must never turn the stop into an error."""
    try:
        note(row, actor() if callable(actor) else actor, trace, value)
    except Exception:                            # noqa: BLE001,S110
        pass


Actor = str | Callable[[], str]


def _check_token(row: Row, actor: Actor, trace: int, request: Any, gate: str,
                 token: Any) -> None:
    """Step 2 for a token row: -32011 or -32030 before the handler. The refusal is journalled
    as a request followed by its error."""
    if gate != "token":
        return
    try:
        access.check(current_hub(), token)
    except Exception as exc:
        _note_request(row, actor, trace, request)
        _note_error(row, actor, trace, exc)
        raise


async def _operator_tip_take(gate: str, route: str | None) -> None:
    """Step 2 for an operator tip row: a script's token is revoked, logged, broadcast, even if
    the handler then refuses. Bookkeeping never fails the request."""
    if gate != "operator_tip":
        return
    try:
        await current_hub().preempt_for_operator(route)
    except Exception:                            # noqa: BLE001,S110
        pass


def dispatch(row: Row, actor: Actor, request: Any, *, trace: int | None = None,
             gate: str = "none", token: Any = None, wait_for_disk: bool = True) -> Any:
    """Run a row's handler for `actor`, journalling the request and then the reply or error.

    `gate` is the carrier's declared gate for this row. A stop row is never gated.
    `wait_for_disk=False` journals an intent row's request with plain `note`. FLEX uses it:
    stops share the command socket with starts, so a start's wait would delay a queued stop."""
    trace = new_trace() if trace is None else trace
    if gate in ("operator_tip", "operator_stop"):
        raise ValueError(f"a {gate} gate needs dispatch_async (the take announces itself)")
    if row.control != "stop":
        _check_token(row, actor, trace, request, gate, token)
    if row.control == "stop":
        try:
            with _wire_trace(trace):
                outcome = row.handler(request)
        except Exception as exc:
            _safely(_note_request, row, actor, trace, request)
            _safely(_note_error, row, actor, trace, exc)
            raise
        _safely(_note_request, row, actor, trace, request)
        _safely(_note_reply, row, actor, trace, outcome)
        return outcome
    else:
        _note_request(row, actor, trace, request, start=wait_for_disk)
        try:
            with _wire_trace(trace):
                outcome = row.handler(request)
        except Exception as exc:
            _note_error(row, actor, trace, exc)
            raise
    _note_reply(row, actor, trace, outcome)
    return outcome


def _take_for_withdraw(gate: str, route: str | None):
    """The browser Withdraw's take (F46): the script's token is revoked in memory, under the
    control lock only. Nothing is sent and nothing is awaited, so no browser window can hold
    the withdraw up. Returns the revoked holder, or None."""
    if gate != "operator_stop":
        return None
    try:
        return access.operator_tip_action(current_hub(), route)
    except Exception:                            # noqa: BLE001
        _log.exception("revoking the script's control for %s failed; the withdraw goes on", route)
        return None


def _announce_after_withdraw(revoked, route: str | None) -> None:
    """Tell the browser, in a background task, that `revoked` lost control. It cannot delay
    or fail the withdraw, which has already run."""
    if revoked is None:
        return
    try:
        current_hub().spawn_revoked_notice(revoked, route)
    except Exception:                            # noqa: BLE001,S110
        pass


async def dispatch_async(row: Row, actor: Actor, request: Any = None, *,
                         trace: int | None = None, gate: str = "none", token: Any = None,
                         route: str | None = None,
                         prepare: Callable[[], Awaitable[Any]] | None = None) -> Any:
    """`dispatch` for handlers that are coroutine functions (the HTTP routes' handlers).

    `prepare` reads and checks the request body. It runs after the operator take, because the
    take fires even when the request is then refused, a malformed body included (A2).
    The browser Withdraw (gate "operator_stop") takes in memory first, runs, and announces
    the revocation afterwards."""
    revoked = _take_for_withdraw(gate, route)
    try:
        return await _dispatch_async(row, actor, request, trace=trace, gate=gate,
                                     token=token, route=route, prepare=prepare)
    finally:
        _announce_after_withdraw(revoked, route)


async def _dispatch_async(row: Row, actor: Actor, request: Any = None, *,
                          trace: int | None = None, gate: str = "none", token: Any = None,
                          route: str | None = None,
                          prepare: Callable[[], Awaitable[Any]] | None = None) -> Any:
    trace = new_trace() if trace is None else trace
    if row.control != "stop":
        await _operator_tip_take(gate, route)
    if prepare is not None:
        try:
            request = await prepare()
        except Exception as exc:
            _safely(_note_request, row, actor, trace, None)
            _safely(_note_error, row, actor, trace, exc)
            raise
    if row.control != "stop":
        _check_token(row, actor, trace, request, gate, token)
    if row.control == "stop":
        try:
            with _wire_trace(trace):
                outcome = row.handler(request)
                if inspect.isawaitable(outcome):
                    outcome = await outcome
        except Exception as exc:
            _safely(_note_request, row, actor, trace, request)
            _safely(_note_error, row, actor, trace, exc)
            raise
        _safely(_note_request, row, actor, trace, request)
        _safely(_note_reply, row, actor, trace, outcome)
        return outcome
    else:
        if row.intent:  # the fsync wait must not hold the event loop
            await asyncio.get_running_loop().run_in_executor(
                _intent_executor,
                functools.partial(_note_request, row, actor, trace, request, start=True))
        else:
            _note_request(row, actor, trace, request)
        try:
            with _wire_trace(trace):
                outcome = row.handler(request)
                if inspect.isawaitable(outcome):
                    outcome = await outcome
        except Exception as exc:
            _note_error(row, actor, trace, exc)
            raise
    _note_reply(row, actor, trace, outcome)
    return outcome
