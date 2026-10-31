"""Rows for the control requests: a script asks for, keeps and gives up the instrument, and the
operator's Take control / Release buttons and the chip's read.

The rules live in `service/access.py`. FLEX acquire/release/heartbeat keep their registry
handlers, which validate their own parameters; the browser's take, release and state handlers
are the old routes' bodies, moved here. `heartbeat` is gated by `dispatch` (`access.check`);
release is never gated, so a script can call it from a `finally` block (C4).
"""

from __future__ import annotations

from typing import Any

from .. import access
from .. import flex_dispatch as fd
from .cards import CARDS
from .context import current_ctx, current_hub
from .table import Flex, Http, HttpCall, Row, object_body


def raw(params: Any) -> Any:
    """The handlers validate their own parameters, as they always have."""
    return params


def body_or_empty(call: HttpCall) -> dict:
    """A missing body is `{}`; anything but an object is refused with FastAPI's own 422 detail."""
    return object_body(call) or {}


def _flex(verb: str, gate: str) -> Flex:
    return Flex(verb=verb, socket="command", bound_s=CARDS[verb].bound_s,
                intake=raw, gate=gate)


def _acquire(params):
    return fd._h_acquire_control(current_hub(), params, current_ctx())


def _release(params):
    return fd._h_release_control(current_hub(), params, current_ctx())


def _heartbeat(params):
    # Dispatch has checked the token; the handler checks again on the loop, where the
    # refresh happens, so a token that lapses in between is still refused.
    return fd._h_heartbeat(current_hub(), params, current_ctx())


async def _operator_release(body):
    """The operator hands the instrument back. With no token in the body, the current holder's
    token is used, so this releases a script too (C4). Idempotent: nothing held is stale."""
    hub = current_hub()
    cur = hub.control
    token = body.get("token") or (cur.token if cur is not None else None)
    out = access.release(hub, token, actor="operator")
    if not out["stale"]:
        await hub.log("Control: operator released the instrument")
        await hub.broadcast_control()
    return {**out, "control": hub.control_wire()}


async def _operator_take(body):
    """Any script token is revoked and the operator becomes the commander. Operator control has
    no dead-man (the human is present); the engaged-idle cap still applies."""
    name = str(body.get("client") or "operator")
    return await current_hub().take_control(client=name, reason=body.get("reason"))


async def _state(_body):
    """Who holds the instrument. The token is never in this body: any tab can call it."""
    from ..control import deadman_armed
    hub = current_hub()
    return {"control": hub.control_wire(), "engaged": bool(hub.host.engaged),
            # The chip warns when the timers are not running: they are why it is safe to
            # leave a script driving.
            "deadman_armed": deadman_armed(hub)}


ROWS = [
    Row(name="acquireControl", kind="command", control="none", handler=_acquire,
        flex=_flex("acquireControl", "none"), help=CARDS["acquireControl"].doc),
    Row(name="releaseControl", kind="command", control="none", handler=_release,
        flex=_flex("releaseControl", "none"),
        http=Http(method="POST", path="/api/control/release", intake=body_or_empty,
                  gate="none", handler=_operator_release),
        help=CARDS["releaseControl"].doc),
    Row(name="heartbeat", kind="command", control="none", handler=_heartbeat,
        flex=_flex("heartbeat", "token"), help=CARDS["heartbeat"].doc),
    Row(name="takeControl", kind="command", control="none", handler=_operator_take,
        http=Http(method="POST", path="/api/control/take", intake=body_or_empty, gate="none")),
    Row(name="getControl", kind="read", control="none", handler=_state,
        http=Http(method="GET", path="/api/control", intake=lambda _call: None, gate="none")),
]
