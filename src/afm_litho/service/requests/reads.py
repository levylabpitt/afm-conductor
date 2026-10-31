"""Rows for the reads: no tip, answered from memory, config or files on disk.

Each row's FLEX bound and HELP card come from `cards.py` (the one copy of those
numbers and words). The handlers are the same functions the verbs and routes always ran.
A row with only an HTTP binding (`_web_row`) calls its old route function directly.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .. import flex_dispatch as fd
from ..host import litho_cmds, scan_cmds
from .cards import CARDS
from .context import current_hub
from .table import Flex, Http, Row


def no_params(raw: Any) -> None:
    """These reads take nothing; a stray body or params list is ignored, as it is today."""
    return None


def with_params(raw: Any) -> Any:
    """The verb reads its own named or positional values (HELP, the result reads)."""
    return raw


def _flex(verb: str, intake=no_params) -> Flex:
    return Flex(verb=verb, socket="read", bound_s=CARDS[verb].bound_s,
                intake=intake, gate="none")


def _http(path: str, handler=None) -> Http:
    return Http(method="GET", path=path, intake=no_params, gate="none", handler=handler)


def _row(verb: str, handler, http: Http | None = None, intake=no_params) -> Row:
    # Named after the verb so a journal reads the same as before the verb had a row.
    return Row(name=verb, kind="read", control="none", handler=handler,
               flex=_flex(verb, intake), http=http, help=CARDS[verb].doc)


def _web_row(name: str, path: str, handler, intake=no_params) -> Row:
    """A browser-only read: FLEX has no verb for it, or its verb has its own derivation."""
    return Row(name=name, kind="read", control="none", handler=handler,
               http=Http(method="GET", path=path, intake=intake, gate="none"))


def _ack(_req):
    return fd._h_ack(current_hub(), None)


def _idn(_req):
    return fd._h_idn(current_hub(), None)


def _capabilities(_req):
    return fd._h_get_capabilities(current_hub(), None)


def _instrument(_req):
    return fd._h_get_instrument(current_hub(), None)


async def _instrument_http(_req):
    # The old route was a plain `def`, run in a worker thread: it reads config files.
    return await asyncio.to_thread(fd._h_get_instrument, current_hub(), None)


def _last_park(_req):
    return fd._h_get_last_park(current_hub(), None)


def _estimate(_req):
    return litho_cmds.write_estimate(current_hub())


async def _estimate_http(_req):
    return await litho_cmds.write_estimate_async(current_hub())


def _mode(_req):
    return scan_cmds.get_mode(current_hub())


def _help(req):
    return fd._h_help(current_hub(), req)


def _state(_req):
    return fd._h_get_state(current_hub(), None)


def _status(_req):
    return fd._h_get_status(current_hub(), None)


def _telemetry(_req):
    return fd._h_get_telemetry(current_hub(), None)


def _pattern(_req):
    return fd._h_get_pattern(current_hub(), None)


def _written(_req):
    return fd._h_get_written(current_hub(), None)


def _scan_result(req):
    return fd._h_get_scan_result(current_hub(), req)


def _write_result(req):
    return fd._h_get_write_result(current_hub(), req)


# The browser's own reads: each calls the route function it always ran. The routers import
# the process hub at load, so they are imported here, when the request arrives.

async def _web_state(_req):
    from ..routers import meta
    return await meta.state()


async def _web_scan_last(_req):
    from ..routers import scan
    return await scan.scan_last()


async def _web_drift(_req):
    from ..routers import drift
    return await drift.drift_get()


async def _web_ports(_req):
    from ..routers import connect
    return await connect.connect_ports()


async def _web_continuous(_req):
    from ..routers import continuous
    return await continuous.continuous_get()


async def _web_design(_req):
    from ..routers import design
    return await design.design_get()


async def _web_design_capabilities(_req):
    from ..routers import design
    return await design.design_capabilities()


async def _web_design_svg(_req):
    from ..routers import design
    return await design.design_svg_export()


async def _web_design_background_channels(_req):
    from ..routers import design
    return await design.design_background_channels()


async def _web_design_files(_req):
    from ..routers import design
    return await design.design_files()


ROWS = [
    _row("ACK", _ack),
    _row("IDN", _idn),
    _row("getCapabilities", _capabilities),
    _row("getInstrument", _instrument,
         _http("/api/instrument", _instrument_http)),
    _row("getLastPark", _last_park, _http("/api/park/last")),
    _row("getEstimate", _estimate, _http("/api/litho/estimate", _estimate_http)),
    _row("getMode", _mode, _http("/api/mode")),
    _row("HELP", _help, intake=with_params),
    # getState and GET /api/state are two rows on purpose: different derivations.
    _row("getState", _state),
    _row("getStatus", _status),
    _row("getTelemetry", _telemetry),
    _row("getPattern", _pattern),
    _row("getWritten", _written),
    _row("getScanResult", _scan_result, intake=with_params),
    _row("getWriteResult", _write_result, intake=with_params),
    _web_row("stateSnapshot", "/api/state", _web_state),
    _web_row("lastScan", "/api/scan/last", _web_scan_last),
    _web_row("drift", "/api/drift", _web_drift),
    _web_row("connectPorts", "/api/connect/ports", _web_ports),
    _web_row("scanContinuous", "/api/scan/continuous", _web_continuous),
    _web_row("design", "/api/design", _web_design),
    _web_row("designCapabilities", "/api/design/capabilities", _web_design_capabilities),
    _web_row("designSvg", "/api/design/svg", _web_design_svg),
    _web_row("designBackgroundChannels", "/api/design/background/channels",
             _web_design_background_channels),
    _web_row("designFiles", "/api/design/files", _web_design_files),
]
