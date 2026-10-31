"""The remaining reads (slice 2.3): rows for FLEX getState..HELP and the browser's GET routes."""

from __future__ import annotations

import pytest

from afm_litho.service import flex_dispatch as fd
from afm_litho.service import journal
from afm_litho.service.hub import hub
from afm_litho.service.requests import flex_door, table

STABLE_VERBS = ["HELP", "getState", "getPattern", "getWritten"]
WEB_ROWS = {
    "/api/state": "stateSnapshot", "/api/drift": "drift",
    "/api/connect/ports": "connectPorts",
    "/api/scan/continuous": "scanContinuous", "/api/design": "design",
    "/api/design/capabilities": "designCapabilities", "/api/design/files": "designFiles",
    "/api/design/background/channels": "designBackgroundChannels",
    "/api/design/svg": "designSvg",
}


class Recorder:
    def __init__(self):
        self.events = []

    def note(self, src, what, /, **fields):
        self.events.append((what, fields))


@pytest.fixture
def events(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(journal, "_running", rec)
    return rec.events


@pytest.mark.parametrize("verb", STABLE_VERBS)
def test_flex_read_goes_through_its_row_and_is_journalled(verb, events):
    flex_door.handle(hub, verb, None, role="read", ctx=None)
    assert [w for w, _ in events][:2] == ["request", "reply"]
    assert events[0][1]["row"] == verb


def test_status_telemetry_and_result_reads_have_rows():
    for verb in ("getStatus", "getTelemetry", "getScanResult", "getWriteResult"):
        assert table.flex_row(verb).flex.socket == "read"


def test_help_card_and_unknown_command_pass_through_the_row():
    card = flex_door.handle(hub, "HELP", {"command": "getState"}, role="read", ctx=None)
    assert card["method"] == "getState"
    with pytest.raises(fd.RpcError) as err:
        flex_door.handle(hub, "HELP", ["nope"], role="read", ctx=None)
    assert err.value.code == fd.INVALID_PARAMS


def test_result_read_with_unknown_run_id_is_the_same_refusal():
    with pytest.raises(fd.RpcError) as err:
        flex_door.handle(hub, "getScanResult", {"run_id": "r-nope"}, role="read", ctx=None)
    assert err.value.code == fd.INVALID_PARAMS


def test_getState_and_api_state_are_separate_rows():
    flex, web = table.flex_row("getState"), [r for r in table.ROWS if r.name == "stateSnapshot"]
    assert flex.http is None and len(web) == 1 and web[0].flex is None
    assert flex.handler is not web[0].handler


OLD_ROUTES = {
    "/api/state": ("meta", "state"), "/api/drift": ("drift", "drift_get"),
    "/api/connect/ports": ("connect", "connect_ports"),
    "/api/scan/continuous": ("continuous", "continuous_get"),
    "/api/design": ("design", "design_get"),
    "/api/design/capabilities": ("design", "design_capabilities"),
    "/api/design/files": ("design", "design_files"),
    "/api/design/background/channels": ("design", "design_background_channels"),
    "/api/design/svg": ("design", "design_svg_export"),
}


def _old_reply(path):
    """What the route function the browser always ran answers, called directly."""
    import asyncio
    import importlib

    from fastapi import Response
    from fastapi.encoders import jsonable_encoder
    module, name = OLD_ROUTES[path]
    func = getattr(importlib.import_module(f"afm_litho.service.routers.{module}"), name)
    out = asyncio.run(func())
    return out.body.decode() if isinstance(out, Response) else jsonable_encoder(out)


@pytest.mark.parametrize("path,name", sorted(WEB_ROWS.items()))
def test_browser_read_is_served_by_its_row_as_operator(path, name, client, events):
    reply = client.get(path)
    assert reply.status_code == 200
    ours = [f for w, f in events if w == "request" and f["row"] == name]
    assert ours and ours[0]["actor"] == "operator"
    old = _old_reply(path)
    got = reply.text if isinstance(old, str) else reply.json()
    if path == "/api/scan/continuous":    # elapsed_s is read from the clock at each call
        got, old = ({k: v for k, v in d.items() if k != "elapsed_s"} for d in (got, old))
    assert got == old


def test_scan_last_404_is_journalled_as_an_error_and_still_a_404(client, events, monkeypatch):
    import afm_litho.service.scan_cache as sc
    monkeypatch.setattr(sc, "load_latest_scan_frame", lambda _p: None)
    assert client.get("/api/scan/last").status_code == 404
    assert [w for w, f in events if f["row"] == "lastScan"] == ["request", "error"]
