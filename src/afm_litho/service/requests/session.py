"""Rows for the session archive: list the recorded scans and writes, recall one."""

from __future__ import annotations

from .table import Http, HttpCall, Row, query_text


def _nothing(_call: HttpCall) -> None:
    return None


def _recall_id(call: HttpCall) -> str:
    return query_text(call, "id")


async def _list(_request):
    from ..routers import session
    return await session.sessions_list()


async def _recall(session_id):
    from ..routers import session
    return await session.sessions_recall(session_id)


def _row(name: str, handler, path: str, intake) -> Row:
    http = Http(method="GET", path=path, intake=intake, gate="none")
    return Row(name=name, kind="read", control="none", handler=handler, http=http,
               journalled=False)


ROWS = [
    _row("sessions", _list, "/api/sessions", _nothing),
    _row("sessionRecall", _recall, "/api/sessions/recall", _recall_id),
]
