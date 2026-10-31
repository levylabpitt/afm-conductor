"""The edit cockpit's header badge shows the service's LIVE backend (GET /api/state)."""

# ruff: noqa: F811 -- tests take the imported fixtures as parameters.

from __future__ import annotations

import shutil
import socket

import pytest
from _service import Service
from test_instrument_cockpit import (
    _WS_CAPTURE_INIT,
    browser,  # noqa: F401 -- fixture re-export
    page,  # noqa: F401 -- fixture re-export
    service,  # noqa: F401 -- fixture re-export
)

_BADGE = (
    "() => { const e = document.getElementById('backendBadge');"
    " return [e.textContent, e.className]; }"
)


def test_badge_reads_twin_on_the_twin(page, service):
    status, state = service.get("/api/state")
    assert status == 200 and state["backend"] == "twin"
    text, cls = page.evaluate(_BADGE)
    assert text == "twin" and "badge-twin" in cls


def test_badge_says_bridge_down_for_an_unconnected_hardware_station(browser, monkeypatch):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("AFM_LITHO_BACKEND", "hardware")
    monkeypatch.setenv("AFM_LITHO_BRIDGE_PORT", str(port))
    svc = Service(flex=True)
    try:
        svc.start()
    except Exception as exc:      # noqa: BLE001 -- pragma: no cover - environment
        shutil.rmtree(svc.runtime, ignore_errors=True)
        pytest.skip(f"hardware-configured service would not start: {exc}")
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000}, permissions=[])
    try:
        ctx.add_init_script(_WS_CAPTURE_INIT)
        p = ctx.new_page()
        p.goto(svc.url + "/edit-cockpit", wait_until="load", timeout=45000)
        p.wait_for_timeout(800)
        text, cls = p.evaluate(_BADGE)
        assert text == "hardware (bridge down)" and "badge-disconnected" in cls
    finally:
        ctx.close()
        svc.stop()
        shutil.rmtree(svc.runtime, ignore_errors=True)
