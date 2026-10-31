"""Rows for the design edits: HTTP only, the cockpit's way of changing the working design.

These are edits, not tip actions: nothing here moves the tip, so the control is "none".
Each handler runs the function the old route ran, which still holds the editing code and
sends the `design_changed` event; a row only gives it a name, a journal entry and an intake.
"""

from __future__ import annotations

from fastapi.exceptions import RequestValidationError

from .table import Http, HttpCall, Row, object_body, path_int, query_float


def _design():
    # Imported at call time: the router module needs the hub, which loads after the table.
    from ..routers import design
    return design


def _body(call: HttpCall) -> dict:
    return object_body(call, required=True)


def _optional_body(call: HttpCall) -> dict | None:
    return object_body(call)


def _bezier_index(call: HttpCall) -> int:
    return path_int(call, "index")


def _nothing(_call: HttpCall) -> None:
    return None


def _shape_index(call: HttpCall) -> int:
    return path_int(call, "index")


def _shape_update(call: HttpCall) -> tuple[int, dict]:
    """The path value and the body are both checked, and both refusals are reported
    together, as FastAPI reported them for the old route."""
    problems = []
    index = body = None
    try:
        index = path_int(call, "index")
    except RequestValidationError as exc:
        problems += exc.errors()
    try:
        body = object_body(call, required=True)
    except RequestValidationError as exc:
        problems += exc.errors()
    if problems:
        raise RequestValidationError(problems)
    return index, body


def _svg_import(call: HttpCall) -> dict:
    q = call.query
    return {"text": call.body or "", "name": q.get("name", "design.svg"),
            "voltage": query_float(call, "voltage", 8.0),
            "speed_um_s": query_float(call, "speed_um_s", 4.0),
            "fill_spacing_um": query_float(call, "fill_spacing_um", 0.25)}


async def _reorder(body):
    return await _design().design_reorder(body)


async def _add(body):
    return await _design().design_add(body)


async def _points(body):
    return await _design().design_points(body)


async def _bezier_get(index):
    return await _design().design_bezier_get(index)


async def _bezier(body):
    return await _design().design_bezier(body)


async def _op(body):
    return await _design().design_op(body)


async def _chiral(body):
    return await _design().design_chiral(body)


async def _pens(_request):
    return await _design().pens()


async def _sketch(body):
    return await _design().sketch(body)


async def _sketch_undo(body):
    return await _design().sketch_undo(body)


async def _sketch_clear(_request):
    return await _design().sketch_clear()


async def _shapes(_request):
    return await _design().shapes()


async def _shape_update_handler(request):
    index, body = request
    return await _design().shape_update(index, body)


async def _shape_toolpath(index):
    return await _design().shape_toolpath(index)


async def _svg(args):
    return await _design().import_svg(**args)


async def _background(body):
    return await _design().design_background(body)


async def _save(body):
    return await _design().design_save(body)


async def _open(body):
    return await _design().design_open(body)


def _row(name: str, handler, path: str, intake, method: str = "POST", body: str = "json",
         kind: str = "edit", journalled: bool = True) -> Row:
    http = Http(method=method, path=path, intake=intake, gate="none", body=body)
    return Row(name=name, kind=kind, control="none", handler=handler, http=http,
               journalled=journalled)


ROWS = [
    _row("designReorder", _reorder, "/api/design/reorder", _body),
    _row("designAdd", _add, "/api/design/add", _body),
    _row("designPoints", _points, "/api/design/points", _body),
    _row("designBezierGet", _bezier_get, "/api/design/bezier/{index}", _bezier_index,
         method="GET", kind="read"),
    _row("designBezier", _bezier, "/api/design/bezier", _body),
    _row("designOp", _op, "/api/design/op", _body),
    _row("designChiral", _chiral, "/api/design/chiral", _body),
    _row("designSvgImport", _svg, "/api/design/svg", _svg_import, body="text"),
    _row("designBackground", _background, "/api/design/background", _optional_body),
    _row("designSave", _save, "/api/design/save", _optional_body),
    _row("designOpen", _open, "/api/design/open", _optional_body),
    _row("pens", _pens, "/api/pens", _nothing, method="GET", kind="read", journalled=False),
    _row("sketch", _sketch, "/api/sketch", _body, journalled=False),
    _row("sketchUndo", _sketch_undo, "/api/sketch/undo", _optional_body,
         journalled=False),
    _row("sketchClear", _sketch_clear, "/api/sketch/clear", _nothing, body="text", journalled=False),
    _row("shapes", _shapes, "/api/shapes", _nothing, method="GET", kind="read", journalled=False),
    _row("shapeUpdate", _shape_update_handler, "/api/shapes/{index}", _shape_update, journalled=False),
    _row("shapeToolpath", _shape_toolpath, "/api/shapes/{index}/toolpath", _shape_index,
         method="GET", kind="read", journalled=False),
]
