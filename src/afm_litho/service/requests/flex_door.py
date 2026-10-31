"""The FLEX adapter: translates one FLEX request into `dispatch` and the reply back.

`flex_provider.handle_message` calls `handle`.
A verb with no row is refused -32601, still through `dispatch`, so every request is
journalled with its actor. Nothing is refused because of the actor label.
"""

from __future__ import annotations

from typing import Any

from .. import control as _ctl
from ..commands.errors import CommandError
from ..flex_dispatch import (
    _CODE_NAMES,
    BUSY,
    METHOD_NOT_FOUND,
    READ_ONLY,
    RpcError,
    refuse_unknown,
)
from .context import using_hub
from .dispatch import dispatch, new_trace
from .table import Row, flex_row

UNNAMED_ACTOR = "flex (unnamed)"


def actor_for(hub, params: Any) -> str:
    """The script's client name if the request carries the current token, else unnamed.

    Reads only; a token that is not current simply gives the unnamed label.
    """
    held = getattr(hub, "control", None)
    if held is not None and isinstance(params, dict) and params.get("token") == held.token:
        return held.client or UNNAMED_ACTOR
    return UNNAMED_ACTOR


def handle(hub, method: str, params: Any, *, role: str, ctx: Any) -> Any:
    """Run one FLEX request and return its result, or raise the `RpcError` to send."""
    row = flex_row(method)
    if row is not None:
        with using_hub(hub, ctx):
            return _run_row(row, hub, params, role)
    return _refuse_unknown_verb(hub, method, params, role)


def _refuse_unknown_verb(hub, method, params, role) -> Any:
    """No row: a command-socket token still feeds the dead-man, then the verb is refused."""
    def refuse(_request):
        _refresh_deadman(hub, params, role)
        refuse_unknown(method)
    row = Row(name=method, kind="unknown", control="none", handler=refuse)
    return dispatch(row, actor_for(hub, params), params, trace=new_trace())


def _refresh_deadman(hub, params: Any, role: str) -> None:
    """A command-socket request bearing a token keeps the dead-man fed.

    The read socket is excluded: polling must not keep a crashed script's tip down.
    """
    if role == "command" and isinstance(params, dict) and isinstance(params.get("token"), str):
        try:
            hub.touch_control(params["token"])
        except Exception:                        # noqa: BLE001,S110
            pass


def _token_of(params: Any) -> Any:
    """The token as the verb's handler reads it: named, or first of a positional list
    (`heartbeat` accepts `[token]`)."""
    if isinstance(params, dict):
        return params.get("token")
    if isinstance(params, list) and params:
        return params[0]
    return None


def _refuse(row: Row, actor, params: Any, error: Exception) -> Any:
    """Raise `error` through `dispatch`, so the refusal is journalled as a request followed by
    its error like any other, and the reply is the error itself."""
    def refuse(_request):
        raise error
    refusal = Row(name=row.name, kind=row.kind, control=row.control, handler=refuse)
    return dispatch(refusal, actor, params, trace=new_trace())


def _refuse_on_read_socket(row: Row, actor, params: Any) -> Any:
    """A command-only verb asked of the read socket is -32601."""
    return _refuse(row, actor, params, RpcError(
        METHOD_NOT_FOUND, f"{row.flex.verb} is served by the COMMAND socket only",
        {"method": row.flex.verb, "socket": "command", "methods": sorted(READ_ONLY)}))


def _run_row(row: Row, hub, params: Any, role: str) -> Any:
    actor = (lambda: actor_for(hub, params)) if row.control == "stop" else actor_for(hub, params)
    if row.flex.socket == "command" and role != "command":
        return _refuse_on_read_socket(row, actor, params)
    _refresh_deadman(hub, params, role)
    try:
        request = row.flex.intake(params)
    except Exception as exc:                       # noqa: BLE001  (the intake's own refusal)
        return _refuse(row, actor, params, exc)
    try:
        return dispatch(row, actor, request, gate=row.flex.gate, token=_token_of(params),
                        wait_for_disk=False)  # stops share this socket: never queue them
    except CommandError as exc:
        name = _CODE_NAMES.get(exc.code)
        raise RpcError(exc.code, f"{name}: {exc.message}" if name else exc.message,
                       exc.data) from exc
    except _ctl.ControlError as exc:
        raise RpcError(exc.code, f"{_CODE_NAMES.get(exc.code, 'control')}: {exc.message}",
                       exc.data) from exc
    except _ctl.RunBusy as exc:
        raise RpcError(BUSY, f"busy: {exc}", {"run_id": exc.run_id, "kind": exc.kind}) from exc
