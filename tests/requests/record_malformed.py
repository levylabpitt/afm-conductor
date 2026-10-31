"""Records malformed_design.json and malformed_edits.json from the OLD routers.

Those routers were retired in slice 2.10, so this only runs on a checkout from before it
(arch-migration at 592165e, or any commit where `routers/design.py` and friends still carry
their route decorators). From that checkout's root, with this file copied next to the case
lists in `tests/requests/`:

    AFM_LITHO_RUNTIME=<empty temp folder> python -m pytest tests/requests/record_malformed.py \
        -k record -p no:cacheprovider

It writes both JSON files next to itself. For every case it also checks that the row answers
exactly as the old router did, so a recording is only written from a state where they agree.
The pytest name `record_malformed.py` is not collected by default; the command above names it.
"""

from __future__ import annotations

import json
import pathlib

from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_design_rows import CASES as DESIGN
from test_edit_rows import CASES as EDITS

from afm_litho.service.requests import http_door

HERE = pathlib.Path(__file__).parent


def _apps(routers):
    new, old = FastAPI(), FastAPI()
    http_door.install(new)
    for r in routers:
        old.include_router(r.router)
    return (TestClient(new, raise_server_exceptions=False),
            TestClient(old, raise_server_exceptions=False))


def _record(apps, cases, name):
    new, old = apps
    out = {}
    for label, method, url, kw in cases:
        want = old.request(method, url, **kw)
        got = new.request(method, url, **kw)
        assert got.status_code == want.status_code, label
        detail = None
        if want.status_code == 422:
            detail = [[d["type"], list(d["loc"]), d["msg"]] for d in want.json()["detail"]]
        else:
            assert got.text == want.text, label
        out[label] = {"status": want.status_code, "detail": detail,
                      "text": None if want.status_code == 422 else want.text}
    (HERE / name).write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")


def test_record_design():
    from afm_litho.service.routers import design
    _record(_apps([design]), DESIGN, "malformed_design.json")


def test_record_edits():
    from afm_litho.service.routers import connect, continuous, drift, instrument, tip
    _record(_apps([connect, continuous, drift, instrument, tip]), EDITS,
            "malformed_edits.json")
