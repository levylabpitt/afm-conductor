"""Usertwin coverage for the DEFAULT write setpoint above free air
(docs/INSTRUMENT_MODEL_SPEC.md section 15).

`#lithoSetpoint` is RELATIVE (V above free air) and, since 13.1 item 14, an empty
box refuses. Section 15 gives it a real, non-zero default so it never loads empty
or at 0:

  hydration order (on load / design_changed / reload):
    hub.last_write_params.setpoint_relative from THIS service process if present,
    else the model's write_setpoint_relative_v (code default 0.1).

Section 15 item 4 checklist covered here:
  * fresh service + fresh browser, no write -> box reads 0.1 (the model default,
    also exposed on GET /api/instrument);
  * after a write at 0.3 and a refresh -> 0.3 (last-write wins over the default);
  * override write_setpoint_relative_v to 0.2 via the Config dialog, fresh browser,
    no write yet -> 0.2 (the overridden default wins, and the field IS editable in
    the dialog);
  * an emptied box still refuses (13.1 item 14 preserved).

Reuses `test_instrument_cockpit.py`'s `Service`/`browser` plumbing, exactly as
`test_instrument_config_dialog.py` does, plus a FUNCTION-scoped fresh service so
the "default" and "override" cases start from a clean hub (no accumulated write /
override state from the module-scoped `service`).
"""

# ruff: noqa: F811 -- fixtures imported by name shadow their definitions when taken
# as test parameters; the standard pytest cross-file fixture-reuse pattern.

from __future__ import annotations

import shutil

import pytest
from _service import Service
from test_instrument_cockpit import (  # reuse plumbing, see module docstring
    _seed_free_air,
    _seed_write_setpoint,
    _to_write_tab,
    browser,  # noqa: F401 -- fixture re-export
    page,  # noqa: F401 -- fixture re-export
    service,  # noqa: F401 -- fixture re-export
)

# --------------------------------------------------------- fresh-service --

@pytest.fixture
def fresh_service():
    """A brand-new twin service on its own port/runtime, with NO writes and NO
    overrides -- so `#lithoSetpoint` hydrates from the model default (or an
    override made in-test), not from another test's leftover last_write_params."""
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


def _load_write_tab(svc, browser):
    """A fresh browser context on `svc`, parked on the Write tab. Returns
    (context, page) so the caller can close the context."""
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000}, permissions=[])
    p = ctx.new_page()
    p.goto(svc.url + "/edit-cockpit", wait_until="load", timeout=45000)
    p.wait_for_timeout(600)
    _to_write_tab(p)
    return ctx, p


def _setpoint_value(p) -> str:
    el = p.query_selector("#lithoSetpoint")
    assert el is not None, "#lithoSetpoint not found"
    return el.evaluate("e => e.value")


# ------------------------------------------------------------ item 4 (a) --

def test_fresh_service_hydrates_to_model_default_0_1(fresh_service, browser):
    """Section 15 item 4: a fresh service (no write) + a fresh browser shows the
    model's write_setpoint_relative_v default (0.1) in #lithoSetpoint, and the
    field is exposed on GET /api/instrument."""
    status, model = fresh_service.get("/api/instrument")
    assert status == 200, (status, model)
    assert "write_setpoint_relative_v" in model, model
    assert model["write_setpoint_relative_v"]["value"] == pytest.approx(0.1)

    ctx, p = _load_write_tab(fresh_service, browser)
    try:
        shown = _setpoint_value(p)
        assert shown not in (None, ""), "#lithoSetpoint loaded empty"
        assert float(shown) == pytest.approx(0.1), (
            f"#lithoSetpoint shows {shown!r}, expected the model default 0.1")
    finally:
        ctx.close()


# ------------------------------------------------------------ item 4 (b) --

def test_after_a_write_the_box_hydrates_to_the_last_relative_setpoint(
        fresh_service, browser):
    """Section 15 item 4: after a write at 0.3, a refresh shows 0.3 -- the
    process's last_write_params.setpoint_relative wins over the model default."""
    # before any write: the model default.
    ctx, p = _load_write_tab(fresh_service, browser)
    try:
        assert float(_setpoint_value(p)) == pytest.approx(0.1)
    finally:
        ctx.close()

    # a real write at 0.3 (seeds hub.last_write_params on THIS service process).
    _seed_free_air(fresh_service)
    _seed_write_setpoint(fresh_service, setpoint_relative=0.3)

    ctx, p = _load_write_tab(fresh_service, browser)
    try:
        shown = _setpoint_value(p)
        assert shown not in (None, ""), "#lithoSetpoint loaded empty after a write"
        assert float(shown) == pytest.approx(0.3), (
            f"#lithoSetpoint shows {shown!r}, expected the last write's 0.3")
    finally:
        ctx.close()


# ------------------------------------------------------------ item 4 (c) --

def test_override_default_via_config_dialog_then_fresh_browser_reads_it(
        fresh_service, browser):
    """Section 15 items 1/4: write_setpoint_relative_v is editable in the Config
    dialog; overriding it to 0.2 (no write) makes a fresh browser hydrate to 0.2.
    Proves the field appears in the dialog and the overridden default wins."""
    ctx, p = _load_write_tab(fresh_service, browser)
    try:
        # open the Config dialog and enter edit mode.
        p.click("#configBtn", timeout=5000)
        p.wait_for_selector("#configDialog", state="visible", timeout=5000)
        p.click("#configEdit", timeout=5000)
        # the field appears automatically as an editable row (item 1: "shown and
        # editable in the Config dialog like every other field").
        inp = '[data-field-input="write_setpoint_relative_v"]'
        p.wait_for_selector(inp, timeout=5000)
        p.fill(inp, "0.2")
        p.click("#configSave", timeout=5000)
        p.wait_for_timeout(600)
    finally:
        ctx.close()

    # the override persisted to disk on this service.
    status, model = fresh_service.get("/api/instrument")
    assert status == 200, (status, model)
    assert model["write_setpoint_relative_v"]["value"] == pytest.approx(0.2)
    assert model["write_setpoint_relative_v"]["source"] == "override"

    # a fresh browser, no write -> the overridden default.
    ctx, p = _load_write_tab(fresh_service, browser)
    try:
        shown = _setpoint_value(p)
        assert shown not in (None, ""), "#lithoSetpoint loaded empty after override"
        assert float(shown) == pytest.approx(0.2), (
            f"#lithoSetpoint shows {shown!r}, expected the overridden default 0.2")
    finally:
        ctx.close()


# ------------------------------------------------------------ item 4 (d) --

def test_an_emptied_setpoint_still_refuses(service, page):
    """Section 15 item 2 / 13.1 item 14: the default hydration must NOT weaken
    the empty-field refusal -- if the operator clears the box, #perform is
    disabled and the hint shows (a write at free air is never sent silently)."""
    _to_write_tab(page)
    sp = page.query_selector("#lithoSetpoint")
    assert sp is not None, "#lithoSetpoint not found"
    # clear the box (fill "" then fire input so the gate re-evaluates).
    sp.fill("")
    sp.dispatch_event("input")
    page.wait_for_timeout(150)

    perform = page.query_selector("#perform")
    assert perform is not None, "#perform not found"
    assert perform.is_disabled(), "#perform is enabled with an empty setpoint"
    hint = page.query_selector("#lithoSetpointHint")
    assert hint is not None and hint.is_visible(), (
        "the empty-setpoint hint is not shown")
