"""Rows for the two pages (`/` redirects to the cockpit), the simulation speed and the
optics (CCD) basemap.

Nothing here moves the tip, so the control is "none". Each handler runs the function the
old route ran, in `routers/meta.py`.
"""

from __future__ import annotations

from .table import Http, HttpCall, Row, path_float


def _nothing(_call: HttpCall) -> None:
    return None


def _speed_factor(call: HttpCall) -> float:
    return path_float(call, "factor")


async def _index(_request):
    from ..routers import meta
    return await meta.index()


async def _edit_cockpit(_request):
    from ..routers import meta
    return await meta.edit_cockpit()


async def _optics(_request):
    from ..routers import meta
    return await meta.optics()


async def _speed(factor):
    from ..routers import meta
    return await meta.set_speed(factor)


def _row(name: str, handler, path: str, intake, method: str = "GET", body: str = "json",
         kind: str = "read") -> Row:
    http = Http(method=method, path=path, intake=intake, gate="none", body=body)
    return Row(name=name, kind=kind, control="none", handler=handler, http=http,
               journalled=False)


ROWS = [
    _row("indexPage", _index, "/", _nothing),
    _row("editCockpitPage", _edit_cockpit, "/edit-cockpit", _nothing),
    _row("optics", _optics, "/api/optics", _nothing),
    _row("simSpeed", _speed, "/api/speed/{factor}", _speed_factor, method="POST",
         body="text", kind="edit"),
]
