"""The browser adapter: one route per table row that has an HTTP binding.

Routes added here are registered ahead of the routers; a path with a row has no route
left in them. An intake receives one `HttpCall` holding the
body, the path values and the query values. The actor is always "operator"; the label
refuses nothing.
"""

from __future__ import annotations

import dataclasses
import json

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError

from .context import using_hub
from .dispatch import dispatch_async
from .table import HttpCall, Row, http_rows

OPERATOR = "operator"
_JSON_NULL = object()   # a body of literal `null`: present, unlike an empty body


async def _body_of(request: Request, mode: str):
    """The request body as FastAPI's own `body: dict | None = None` would read it: None when
    empty (the cockpit POSTs release with no body), parsed JSON for a JSON content type, else
    the text (which an object intake then refuses). Bad JSON gives FastAPI's own 422 detail.
    A row bound with body="text" (the SVG upload) always gets the text, and body="bytes" (the
    .ibw upload) the raw bytes. A stop is bound "lenient" (bad JSON reads as no body) or
    "ignore" (never parsed), so it is never refused."""
    if request.method not in ("POST", "PUT") or mode == "ignore":
        return None
    raw = await request.body()
    if mode == "bytes":
        return raw
    if mode == "text":
        return raw.decode("utf-8", errors="replace")
    if not raw:
        return None
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type != "application/json" and not content_type.endswith("+json"):
        return raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(raw)
        return _JSON_NULL if parsed is None else parsed
    except json.JSONDecodeError as exc:
        if mode == "lenient":
            return None
        raise RequestValidationError(
            [{"type": "json_invalid", "loc": ("body", exc.pos), "msg": "JSON decode error",
              "input": {}, "ctx": {"error": exc.msg}}], body=exc.doc) from exc


def _route_for(row: Row):
    if row.http.handler is not None:
        row = dataclasses.replace(row, handler=row.http.handler)

    async def endpoint(request: Request):
        from ..hub import hub
        async def prepare():
            body = await _body_of(request, row.http.body)
            null = body is _JSON_NULL
            return row.http.intake(HttpCall(body=None if null else body, json_null=null,
                                            path=dict(request.path_params),
                                            query=dict(request.query_params)))

        with using_hub(hub):
            return await dispatch_async(row, OPERATOR, gate=row.http.gate,
                                        route=row.http.path, prepare=prepare)

    return endpoint


def install(app: FastAPI) -> None:
    """Register a route for every table row with an HTTP binding."""
    for row in http_rows():
        app.add_api_route(row.http.path, _route_for(row), methods=[row.http.method],
                          name=f"request_{row.name}")
