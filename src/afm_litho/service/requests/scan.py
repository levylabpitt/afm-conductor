"""Rows for the scan panel's own routes: the live parameter record, and the three twin
refusals (force curve, thermal tune, frequency sweep). None moves the tip, so the control
is "none". Each handler runs the function the old route ran, in `routers/scan.py`.
"""

from __future__ import annotations

from .table import Http, HttpCall, Row, object_body


def _nothing(_call: HttpCall) -> None:
    return None


def _optional_body(call: HttpCall) -> dict | None:
    return object_body(call)


async def _params(body):
    from ..routers import scan
    return await scan.scan_params(body)


async def _force_curve(_request):
    from ..routers import scan
    return await scan.force_curve()


async def _tune_thermal(_request):
    from ..routers import scan
    return await scan.tune_thermal()


async def _tune_sweep(_request):
    from ..routers import scan
    return await scan.tune_sweep()


def _post(name: str, handler, path: str, intake, body: str = "json") -> Row:
    http = Http(method="POST", path=path, intake=intake, gate="none", body=body)
    return Row(name=name, kind="edit", control="none", handler=handler, http=http,
               journalled=False)


ROWS = [
    _post("scanParams", _params, "/api/scan/params", _optional_body),
    _post("forceCurve", _force_curve, "/api/force_curve", _nothing, body="text"),
    _post("tuneThermal", _tune_thermal, "/api/tune/thermal", _nothing, body="text"),
    _post("tuneSweep", _tune_sweep, "/api/tune/sweep", _nothing, body="text"),
]
