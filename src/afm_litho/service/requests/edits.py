"""Rows for the other edits: instrument override and reload, electrode ports, drift, the
relative setpoint, the toolpath preview, Show Tip, fault clear, keeper contact, and the
pattern loads.

None of these is in TIP_MUTATING, so none takes access: the gate is "none" everywhere. Each handler runs the function the old route
ran. A route that was a plain `def` ran on a worker thread; it still does (`_in_thread`).
A route that never declared a body parameter ignores whatever body it is sent, so its row
is bound with body="text", which reads the body without parsing it.

`loadPattern` is where the carriers differ. Over FLEX the script may name a file path or
pass `objects`; the cockpit may only name a design inside data/designs. That difference is
the `allow_paths` field of the request each carrier's intake builds, not a branch.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from .. import flex_dispatch as fd
from .cards import CARDS
from .context import current_ctx, current_hub
from .table import Flex, Http, HttpCall, Row, object_body


@dataclass(frozen=True)
class PatternSource:
    """What the browser asks `loadPattern` to read, and whether it may name a path."""

    source: Any
    allow_paths: bool


def _optional_body(call: HttpCall) -> dict | None:
    return object_body(call)


def _required_body(call: HttpCall) -> dict:
    return object_body(call, required=True)


def _ignored_body(_call: HttpCall) -> None:
    return None


def _pattern_source(call: HttpCall) -> PatternSource:
    body = object_body(call) or {}
    return PatternSource(source=body.get("source", "auto"), allow_paths=False)


def raw(params: Any) -> Any:
    """The FLEX handlers validate their own parameters, in their own order."""
    return params


# Each handler imports its router when it runs: the router modules need the hub, which loads
# after the table.


async def _in_thread(function, *args):
    return await asyncio.to_thread(function, *args)


async def _instrument_override(body):
    from ..host.api import HostApi
    from ..hub import hub
    return await _in_thread(HostApi(hub).set_override, body)


async def _instrument_reload(_body):
    from ..routers import instrument
    return await _in_thread(instrument.post_instrument_reload)


async def _instrument_open_config(_body):
    from ..routers import instrument
    return await _in_thread(instrument.post_instrument_open_config)


async def _fault_clear(body):
    from ..host.api import HostApi
    from ..hub import hub
    return await _in_thread(HostApi(hub).clear_fault, body)


async def _keeper_contact(body):
    from ..routers import instrument
    return await _in_thread(instrument.post_keeper_contact, body)


async def _ports_add(body):
    from ..routers import connect
    return await connect.connect_ports_add(body)


async def _ports_update(body):
    from ..routers import connect
    return await connect.connect_ports_update(body)


async def _ports_delete(body):
    from ..routers import connect
    return await connect.connect_ports_delete(body)


async def _ports_replace(body):
    from ..routers import connect
    return await connect.connect_ports_replace(body)


async def _drift_reference(body):
    from ..routers import drift
    return await drift.drift_reference(body)


async def _drift_enable(body):
    from ..routers import drift
    return await drift.drift_enable(body)


async def _scan_relative(body):
    from ..routers import continuous
    return await continuous.relative_set(body)


async def _toolpath(_body):
    from ..routers import litho
    return await litho.litho_toolpath()


async def _tip_show(body):
    from ..routers import tip
    return await tip.tip_show(body)


async def _load_pattern_for_browser(request: PatternSource):
    from ..routers import litho
    return await litho.load_design(request.source, request.allow_paths)


def _load_pattern(params):
    return fd._h_load_pattern(current_hub(), params, current_ctx())


def _set_objects(params):
    return fd._h_set_objects(current_hub(), params, current_ctx())


def _post(name: str, handler, path: str, intake, body: str = "json") -> Row:
    http = Http(method="POST", path=path, intake=intake, gate="none", body=body)
    return Row(name=name, kind="edit", control="none", handler=handler, http=http)


def _flex_command(verb: str) -> Flex:
    # Gate "none": these handlers check the token themselves, after their own parameter
    # checks, and the order of those refusals is part of the wire.
    return Flex(verb=verb, socket="command", bound_s=CARDS[verb].bound_s,
                intake=raw, gate="none")


ROWS = [
    _post("instrumentOverride", _instrument_override, "/api/instrument/override",
          _optional_body),
    _post("instrumentReload", _instrument_reload, "/api/instrument/reload", _ignored_body,
          body="text"),
    _post("instrumentOpenConfig", _instrument_open_config, "/api/instrument/open_config",
          _ignored_body, body="text"),
    _post("faultClear", _fault_clear, "/api/fault/clear", _optional_body),
    _post("keeperContact", _keeper_contact, "/api/keeper/contact", _required_body),
    _post("connectPortsAdd", _ports_add, "/api/connect/ports/add", _required_body),
    _post("connectPortsUpdate", _ports_update, "/api/connect/ports/update", _required_body),
    _post("connectPortsDelete", _ports_delete, "/api/connect/ports/delete", _required_body),
    _post("connectPortsReplace", _ports_replace, "/api/connect/ports/replace",
          _required_body),
    _post("driftReference", _drift_reference, "/api/drift/reference", _optional_body),
    _post("driftEnable", _drift_enable, "/api/drift/enable", _optional_body),
    _post("scanRelative", _scan_relative, "/api/scan/relative", _optional_body),
    _post("lithoToolpath", _toolpath, "/api/litho/toolpath", _ignored_body, body="text"),
    _post("tipShow", _tip_show, "/api/tip/show", _optional_body),
    Row(name="loadPattern", kind="command", control="none", handler=_load_pattern,
        flex=_flex_command("loadPattern"),
        http=Http(method="POST", path="/api/litho/load", intake=_pattern_source,
                  gate="none", handler=_load_pattern_for_browser),
        help=CARDS["loadPattern"].doc),
    Row(name="setObjects", kind="command", control="none", handler=_set_objects,
        flex=_flex_command("setObjects"), help=CARDS["setObjects"].doc),
]
