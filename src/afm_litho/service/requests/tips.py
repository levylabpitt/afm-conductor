"""Rows for the requests that act on the tip or the instrument's mode (the "tip" controls).

A script's `setMode` is gated by its token; the cockpit's `POST /api/mode` is gated by the
operator take, which fires even when the switch is then refused (A2). The refusal while the
tip is engaged lives in `commands.scan_cmds.set_mode`, which both carriers reach.

`setZGain` is a tip row whose HTTP side takes no control and has no token (C2, 2026-10-19:
gain changes need an operator at the instrument), so `/api/zgain` stays out of `TIP_MUTATING`.

`startScan` and `setContinuous` are gated by the token over FLEX and by the operator take over
HTTP. The two carriers keep their own handlers, so the REST continuous route still lacks the
silent-twin gate and the busy check the FLEX verb has (C6, kept as it is).

`startWrite` is gated by the token over FLEX and by the operator take over HTTP. The carriers
differ only in how they ask for the setpoint, and that lives in `litho_cmds.flex_intake` and
`litho_cmds.http_intake` (`setpoint` + `setpoint_reference`, required on both, "absolute" by
default; the cockpit sends `setpoint_relative`, which means a free-air `setpoint`).
"""

from __future__ import annotations

from functools import partial

from .. import flex_dispatch as fd
from ..host.api import HostApi
from ..commands.errors import CommandError
from .cards import CARDS
from .context import current_ctx, current_hub
from .table import Flex, Http, Row, object_body


def _params_object_for(verb):
    """Base's refusal of a non-object `params`, raised before the token gate can answer
    -32030 for a request that never had a place to put a token."""
    def intake(params):
        if params is not None and not isinstance(params, dict):
            fd._body(params, verb)
        return params
    return intake


def _script_set_mode(params):
    return fd._h_set_mode(current_hub(), params, current_ctx())


async def _operator_set_mode(body):
    """The cockpit's reply is a body, not an error: a refusal comes back as `{"error": ...}`."""
    try:
        return await HostApi(current_hub()).set_mode(body)
    except CommandError as exc:
        return exc.http_body()


def _script_start_approach(params):
    return fd._h_start_approach(current_hub(), params, current_ctx())


async def _operator_approach(body):
    """Blocking fine engage; a refusal comes back as `{"error", "reason"}`, status 200."""
    try:
        return await HostApi(current_hub()).approach(body)
    except CommandError as exc:
        return exc.http_body()


def _script_set_zgain(params):
    return fd._h_set_zgain(current_hub(), params, current_ctx())


async def _cockpit_set_zgain(body):
    try:
        return await HostApi(current_hub()).set_zgain(body)
    except CommandError as exc:
        return exc.http_body()


def _script_start_scan(params):
    return fd._h_start_scan(current_hub(), params, current_ctx())


async def _cockpit_start_scan(body):
    """The old route's body, still in `routers.scan` because the continuous loop calls it."""
    from ..routers.scan import scan_start
    return await scan_start(body)


def _script_set_continuous(params):
    return fd._h_set_continuous(current_hub(), params, current_ctx())


async def _cockpit_set_continuous(body):
    from ..routers.continuous import continuous_set
    return await continuous_set(body)


def _script_start_write(params):
    return fd._h_start_write(current_hub(), params, current_ctx())


async def _cockpit_write(body):
    """The old route's body, still in `routers.litho`; it reaches `litho_write` with the
    cockpit's intake (`setpoint_relative`)."""
    from ..routers.litho import litho_write
    return await litho_write(body)


ROWS = [
    Row(name="setMode", kind="command", control="tip", handler=_script_set_mode,
        flex=Flex(verb="setMode", socket="command", bound_s=CARDS["setMode"].bound_s,
                  intake=_params_object_for("setMode"), gate="token"),
        http=Http(method="POST", path="/api/mode", intake=object_body, gate="operator_tip",
                  handler=_operator_set_mode),
        help=CARDS["setMode"].doc),
    Row(name="startApproach", kind="command", control="tip", handler=_script_start_approach,
        flex=Flex(verb="startApproach", socket="command",
                  bound_s=CARDS["startApproach"].bound_s,
                  intake=_params_object_for("startApproach"), gate="token"),
        http=Http(method="POST", path="/api/approach", intake=object_body, gate="operator_tip",
                  handler=_operator_approach),
        help=CARDS["startApproach"].doc),
    Row(name="setZGain", kind="command", control="tip", handler=_script_set_zgain,
        flex=Flex(verb="setZGain", socket="command", bound_s=CARDS["setZGain"].bound_s,
                  intake=_params_object_for("setZGain"), gate="token"),
        http=Http(method="POST", path="/api/zgain", intake=object_body, gate="none",
                  handler=_cockpit_set_zgain),
        help=CARDS["setZGain"].doc),
    Row(name="startScan", kind="command", control="tip", intent=True,
        handler=_script_start_scan,
        flex=Flex(verb="startScan", socket="command", bound_s=CARDS["startScan"].bound_s,
                  intake=_params_object_for("startScan"), gate="token"),
        http=Http(method="POST", path="/api/scan/start", intake=partial(object_body, required=True),
                  gate="operator_tip", handler=_cockpit_start_scan),
        help=CARDS["startScan"].doc),
    Row(name="setContinuous", kind="command", control="tip", handler=_script_set_continuous,
        flex=Flex(verb="setContinuous", socket="command",
                  bound_s=CARDS["setContinuous"].bound_s,
                  intake=_params_object_for("setContinuous"), gate="token"),
        http=Http(method="POST", path="/api/scan/continuous", intake=object_body,
                  gate="operator_tip", handler=_cockpit_set_continuous),
        help=CARDS["setContinuous"].doc),
    Row(name="startWrite", kind="command", control="tip", intent=True,
        handler=_script_start_write,
        flex=Flex(verb="startWrite", socket="command", bound_s=CARDS["startWrite"].bound_s,
                  intake=_params_object_for("startWrite"), gate="token"),
        http=Http(method="POST", path="/api/litho/write", intake=object_body, gate="operator_tip",
                  handler=_cockpit_write),
        help=CARDS["startWrite"].doc),
]
