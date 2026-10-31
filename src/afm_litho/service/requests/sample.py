"""Rows for the sample routes: the list of scans, loading or uploading one, and the contacts
marked on it. Loading a scan swaps the twin's world, but the tip is not driven, so the
control is "none". Each handler runs the function the old route ran, in `routers/sample.py`.
"""

from __future__ import annotations

from .table import Http, HttpCall, Row, object_body, query_text


def _nothing(_call: HttpCall) -> None:
    return None


def _required_body(call: HttpCall) -> dict:
    return object_body(call, required=True)


def _upload_args(call: HttpCall) -> dict:
    """The raw .ibw bytes and the `name` query value (default scan.ibw)."""
    return {"data": call.body if call.body is not None else b"",
            "name": query_text(call, "name", "scan.ibw")}


async def _samples(_request):
    from ..routers import sample
    return await sample.samples()


async def _load(body):
    from ..routers import sample
    return await sample.sample_load(body)


async def _upload(args):
    from ..routers import sample
    return await sample.sample_upload(**args)


async def _contacts_get(_request):
    from ..routers import sample
    return await sample.contacts_get()


async def _contacts_set(body):
    from ..routers import sample
    return await sample.contacts_set(body)


def _row(name: str, handler, path: str, intake, method: str = "GET", body: str = "json") -> Row:
    http = Http(method=method, path=path, intake=intake, gate="none", body=body)
    return Row(name=name, kind="read" if method == "GET" else "edit", control="none",
               handler=handler, http=http, journalled=False)


ROWS = [
    _row("samples", _samples, "/api/samples", _nothing),
    _row("sampleLoad", _load, "/api/sample/load", _required_body, method="POST"),
    _row("sampleUpload", _upload, "/api/sample/upload", _upload_args, method="POST",
         body="bytes"),
    _row("contactsGet", _contacts_get, "/api/contacts", _nothing),
    _row("contactsSet", _contacts_set, "/api/contacts", _required_body, method="POST"),
]
