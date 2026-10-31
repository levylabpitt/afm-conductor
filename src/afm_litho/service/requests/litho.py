"""Rows for the litho panel's own routes: transit policy, litho frame, the demo toolpath
and the written (conductance) map. Nothing here moves the tip; each handler runs the
function the old route ran, in `routers/litho.py`.
"""

from __future__ import annotations

from .table import Http, HttpCall, Row, object_body, query_float, query_int


def _nothing(_call: HttpCall) -> None:
    return None


def _required_body(call: HttpCall) -> dict:
    return object_body(call, required=True)


def _written_args(call: HttpCall) -> dict:
    return {"size_um": query_float(call, "size_um", 30.0),
            "x_off": query_float(call, "x_off", 0.0),
            "y_off": query_float(call, "y_off", 0.0),
            "n": query_int(call, "n", 256)}


async def _transit_get(_request):
    from ..routers import litho
    return await litho.transit_get()


async def _transit_set(body):
    from ..routers import litho
    return await litho.transit_set(body)


async def _frame_get(_request):
    from ..routers import litho
    return await litho.litho_frame_get()


async def _frame_set(body):
    from ..routers import litho
    return await litho.litho_frame_set(body)


async def _demo(_request):
    from ..routers import litho
    return await litho.litho_demo()


async def _written(args):
    from ..routers import litho
    return await litho.litho_written(**args)


def _row(name: str, handler, path: str, intake, method: str = "GET") -> Row:
    http = Http(method=method, path=path, intake=intake, gate="none")
    return Row(name=name, kind="read" if method == "GET" else "edit", control="none",
               handler=handler, http=http, journalled=False)


ROWS = [
    _row("transitGet", _transit_get, "/api/transit", _nothing),
    _row("transitSet", _transit_set, "/api/transit", _required_body, method="POST"),
    _row("lithoFrameGet", _frame_get, "/api/litho/frame", _nothing),
    _row("lithoFrameSet", _frame_set, "/api/litho/frame", _required_body, method="POST"),
    _row("lithoDemo", _demo, "/api/litho/demo", _nothing),
    _row("lithoWritten", _written, "/api/litho/written", _written_args),
]
