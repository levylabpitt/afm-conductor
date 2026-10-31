"""Changing the Voltage field on a chiral wire in the edit cockpit re-bakes the wire
on the service, and the page redraws it from the new profile (it re-adopts the
design on the `design_changed` broadcast the edit triggers).

Reuses `test_instrument_cockpit.py`'s service/browser plumbing on its own ports.
"""

# ruff: noqa: F811 -- fixtures imported by name shadow their definitions when taken
# as test parameters; the standard pytest cross-file fixture-reuse pattern.

from __future__ import annotations

import math
import shutil

import pytest
from _service import Service
from test_cockpit_fill_and_copy import _select, _tab, _wait
from test_instrument_cockpit import (
    browser,  # noqa: F401 -- fixture re-export
)

V_K = 5.0


@pytest.fixture(scope="module")
def service():
    svc = Service(flex=True)
    try:
        svc.start()
    except Exception as exc:      # noqa: BLE001 -- pragma: no cover - environment
        shutil.rmtree(svc.runtime, ignore_errors=True)
        pytest.skip(f"service would not start: {exc}")
    try:
        yield svc
    finally:
        svc.stop()
        shutil.rmtree(svc.runtime, ignore_errors=True)


def _wire(service) -> dict:
    status, d = service.get("/api/design")
    assert status == 200, (status, d)
    return next(o for o in d["objects"] if o["name"] == "w")


def _page_wire(p) -> dict:
    return p.evaluate(
        "async () => { const {state} = await import('/static/ec-state.js');"
        " const o = state.objects.find((x) => x.name === 'w');"
        " return {voltage: o.voltage, v0: o.profile.v0,"
        "         vmin: Math.min(...o.v_profile), vmax: Math.max(...o.v_profile),"
        "         field: document.getElementById('fVoltage').value}; }")


def _seed_chiral_wire(service) -> None:
    _status, d = service.get("/api/design")
    if d and d.get("objects"):
        service.post("/api/design/op", {"op": "delete",
                                        "indices": list(range(len(d["objects"])))})
    _status, r = service.post("/api/design/add", {
        "kind": "path", "name": "w", "points_um": [[5.0, 5.0], [6.0, 5.0]]})
    assert r and r.get("ok"), r
    _status, r = service.post("/api/design/chiral", {
        "indices": [r["index"]], "enabled": True, "lambda_um": 0.025,
        "y_amp_um": 0.055, "v_k": V_K, "phase_deg": 90, "v0": 8})
    assert r and r.get("ok"), r
    assert _wire(service)["profile"]["v0"] == 8.0


def test_a_refused_voltage_shows_a_banner_and_the_services_value_again(service, browser):
    """28 V is in the window, 28 + v_k is not: refused on the rebuilt profile."""
    _seed_chiral_wire(service)
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000}, permissions=[])
    try:
        p = ctx.new_page()
        p.goto(service.url + "/edit-cockpit", wait_until="load", timeout=45000)
        p.wait_for_timeout(800)
        _tab(p, "write")
        _select(p, "w")
        p.fill("#fVoltage", "28")
        p.dispatch_event("#fVoltage", "change")

        def banner() -> bool:
            shown = p.eval_on_selector("#guardNotice",
                                       "e => getComputedStyle(e).display") != "none"
            text = p.eval_on_selector("#guardNoticeText", "e => e.textContent")
            return shown and "tip_voltage_out_of_window" in text and "'w'" in text

        _wait(p, banner, "the refusal banner")
        _wait(p, lambda: float(_page_wire(p)["field"]) == 8.0,
              "the Voltage field to show the service's 8 V again")
        w = _page_wire(p)
        assert (w["voltage"], w["v0"]) == (8.0, 8.0), w
        o = _wire(service)
        assert o["voltage"] == 8.0 and o["profile"]["v0"] == 8.0
    finally:
        ctx.close()


def test_the_voltage_field_rebakes_a_chiral_wire_and_the_page_redraws_it(service, browser):
    _seed_chiral_wire(service)

    ctx = browser.new_context(viewport={"width": 1600, "height": 1000}, permissions=[])
    try:
        p = ctx.new_page()
        p.goto(service.url + "/edit-cockpit", wait_until="load", timeout=45000)
        p.wait_for_timeout(800)
        _tab(p, "write")
        _select(p, "w")
        assert _page_wire(p)["v0"] == 8.0

        p.fill("#fVoltage", "6")
        p.dispatch_event("#fVoltage", "change")

        def service_rebaked() -> bool:
            o = _wire(service)
            return (o["voltage"] == 6.0 and o["profile"]["v0"] == 6.0
                    and abs(max(o["v_profile"]) - (6.0 + V_K)) < 0.05
                    and abs(min(o["v_profile"]) - (6.0 - V_K)) < 0.05)

        _wait(p, service_rebaked, "the service to re-bake the wire at 6 V")

        def page_redrawn() -> bool:
            w = _page_wire(p)
            return (w["v0"] == 6.0 and w["voltage"] == 6.0
                    and abs(w["vmax"] - (6.0 + V_K)) < 0.05
                    and abs(w["vmin"] - (6.0 - V_K)) < 0.05)

        _wait(p, page_redrawn, "the page to re-adopt the re-baked wire")
        assert float(_page_wire(p)["field"]) == 6.0
    finally:
        ctx.close()


RING_PITCH = 0.0251


def _open(browser, service):
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000}, permissions=[])
    p = ctx.new_page()
    p.goto(service.url + "/edit-cockpit", wait_until="load", timeout=45000)
    p.wait_for_timeout(800)
    _tab(p, "write")
    return ctx, p


def _banner_text(p) -> str | None:
    if p.eval_on_selector("#guardNotice", "e => getComputedStyle(e).display") == "none":
        return None
    return p.eval_on_selector("#guardNoticeText", "e => e.textContent")


def test_the_pitch_field_shows_the_asked_for_pitch_and_a_refused_edit_restores_it(
        service, browser):
    _status, d = service.get("/api/design")
    if d and d.get("objects"):
        service.post("/api/design/op", {"op": "delete",
                                        "indices": list(range(len(d["objects"])))})
    ring = [[5.0 + math.cos(2 * math.pi * k / 240), 5.0 + math.sin(2 * math.pi * k / 240)]
            for k in range(240)]
    ring.append(ring[0])
    _status, r = service.post("/api/design/add", {
        "kind": "path", "name": "w", "points_um": ring})
    assert r and r.get("ok"), r
    _status, r = service.post("/api/design/chiral", {
        "indices": [r["index"]], "enabled": True, "lambda_um": RING_PITCH,
        "y_amp_um": 0.055, "v_k": V_K, "phase_deg": 90, "v0": 8})
    assert r and r.get("ok"), r
    prof = _wire(service)["profile"]
    assert prof["lambda_set_um"] == RING_PITCH and prof["lambda_um"] != RING_PITCH

    ctx, p = _open(browser, service)
    try:
        _select(p, "w")
        assert float(p.eval_on_selector("#fLambda", "e => e.value")) == RING_PITCH

        p.fill("#fLambda", "")
        p.dispatch_event("#fLambda", "change")
        _wait(p, lambda: "chiral_bad_setting" in (_banner_text(p) or ""),
              "the refusal banner")
        assert "lambda_um" in _banner_text(p)
        _wait(p, lambda: p.eval_on_selector("#fLambda", "e => e.value") != "",
              "the Pitch field to show the service's value again")
        assert float(p.eval_on_selector("#fLambda", "e => e.value")) == RING_PITCH
        assert _wire(service)["profile"] == prof
    finally:
        ctx.close()


@pytest.mark.parametrize("reason", ["chiral_bias_fixed", "chiral_unrebakeable",
                                    "chiral_build_refused", "chiral_bad_setting"])
def test_each_chiral_refusal_reason_raises_the_banner(service, browser, reason):
    ctx, p = _open(browser, service)
    try:
        assert _banner_text(p) is None
        p.evaluate("async (reason) => {"
                   " const {showRefusal} = await import('/static/ec-instrument.js');"
                   " showRefusal({reason, error: 'object w'}); }", reason)
        assert _banner_text(p) == f"refused ({reason}): object w"
    finally:
        ctx.close()
