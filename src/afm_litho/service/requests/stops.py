"""Rows for the stops: FLEX abort, safePark, withdraw and abortWrite, and the browser's
/api/safe_park, /api/withdraw, /api/scan/abort and /api/litho/abort.

A stop row (control "stop") runs its handler first. Nothing is gated, no token is required,
the actor is resolved after, and the journal cannot fail it. Abort, safePark and the two
abort routes take no control. The browser Withdraw is the one exception (F46): `dispatch`
revokes a script's token in memory before it runs and tells the browser afterwards, so a
stalled browser window cannot hold the retract up (gate "operator_stop").
No stop is refused for its body: a malformed or non-object body reads as none (litho/abort
never read one).

The cockpit Abort (/api/safe_park) and FLEX abort and safePark also send their Stop to the
keeper (`keeper_stop.tell_keeper`), from a thread, before their own route runs. Both carry
one id, so the host parks once for the pair (safety/stop_pairs.py).
"""

from __future__ import annotations

from .. import flex_dispatch as fd
from ..host.api import HostApi
from ..keeper_stop import tell_keeper
from ..safety.stop_pairs import new_id
from .cards import CARDS
from .context import current_ctx, current_hub
from .control import raw
from .table import Flex, Http, HttpCall, Row


def _flex(verb: str) -> Flex:
    return Flex(verb=verb, socket="command", bound_s=CARDS[verb].bound_s,
                intake=raw, gate="stop")


def _lenient(call: HttpCall) -> dict | None:
    """A stop is never refused for its body: anything but an object reads as no body."""
    return call.body if isinstance(call.body, dict) else None


def _why(params, key: str, index: int, default: str) -> str:
    """The caller's words for the keeper's line; never an error, whatever the params."""
    try:
        return fd._stop_str(params, key, index) or default
    except Exception:                                   # noqa: BLE001
        return default


def _abort(params):
    sid = new_id()
    scope = _why(params, "scope", 1, "")
    tell_keeper("flex", f"abort ({scope})" if scope else "abort", sid)
    return fd._h_abort(current_hub(), params, current_ctx(), stop_id=sid)


def _safe_park(params):
    sid = new_id()
    tell_keeper("flex", _why(params, "reason", 0, "safePark"), sid)
    return fd._h_safe_park(current_hub(), params, current_ctx(), stop_id=sid)


def _withdraw(params):
    return fd._h_withdraw(current_hub(), params, current_ctx())


def _abort_write(params):
    return fd._h_abort_write(current_hub(), params, current_ctx())


# The browser's forms. The safe_park and litho-abort bodies stay in the routers, imported
# inside the functions because the routers load after the table; withdraw and scan abort
# call their HostApi verbs.

async def _browser_safe_park(body):
    from ..routers.scan import safe_park_endpoint
    sid = new_id()
    tell_keeper("cockpit", (body or {}).get("reason") or "operator abort", sid)
    return await safe_park_endpoint(body, stop_id=sid)


async def _browser_withdraw(body):
    return await HostApi(current_hub()).withdraw(body)


async def _browser_scan_abort(body):
    # The process hub, which a running scan pump reads; a loaded module, so the import
    # cannot raise.
    from ..hub import hub
    return await HostApi(hub).stop_scan(body)


async def _browser_litho_abort(_ignored):
    from ..routers.litho import litho_abort
    return await litho_abort()


ROWS = [
    Row(name="abort", kind="command", control="stop", handler=_abort,
        flex=_flex("abort"), help=CARDS["abort"].doc),
    Row(name="safePark", kind="command", control="stop", handler=_safe_park,
        flex=_flex("safePark"),
        http=Http(method="POST", path="/api/safe_park", intake=_lenient, gate="stop",
                  handler=_browser_safe_park, body="lenient"),
        help=CARDS["safePark"].doc),
    Row(name="withdraw", kind="command", control="stop", handler=_withdraw,
        flex=_flex("withdraw"),
        http=Http(method="POST", path="/api/withdraw", intake=_lenient, gate="operator_stop",
                  handler=_browser_withdraw, body="lenient"),
        help=CARDS["withdraw"].doc),
    Row(name="abortWrite", kind="command", control="stop", handler=_abort_write,
        flex=_flex("abortWrite"),
        http=Http(method="POST", path="/api/litho/abort", intake=lambda call: None,
                  gate="stop", handler=_browser_litho_abort, body="ignore"),
        help=CARDS["abortWrite"].doc),
    Row(name="abortScan", kind="command", control="stop", handler=_browser_scan_abort,
        http=Http(method="POST", path="/api/scan/abort", intake=_lenient, gate="stop",
                  body="lenient")),
]
