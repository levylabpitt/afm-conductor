"""Usertwin coverage for the cockpit CONFIG DIALOG (docs/INSTRUMENT_MODEL_SPEC.md
section 14): the per-tab `#instrumentPanel` is gone, replaced by a modal opened
from the header's Config button, with Edit / Save / Cancel that post
`POST /api/instrument/override` for the CHANGED fields only, and platform now a
file-owned model Field (item 4) whose change is refused while engaged and
re-derives the platform defaults.

Reuses `test_instrument_cockpit.py`'s plumbing (the `Service`/`browser`/`page`
fixtures and the HTTP seeding helpers) rather than re-inventing it, exactly as
`test_instrument_cockpit_b3b.py` does.

Section 14 item 5 checklist covered here:
  * the dialog opens from every (normal) tab;
  * edit then Cancel restores the values and sends nothing;
  * edit then Save persists across a browser reload AND on disk (i.e. a service
    restart, since `load_model` reads `[override]` from the file at startup);
  * a refused value keeps the dialog in edit mode with the reason on that row,
    and a field applied before it stays applied;
  * a platform change is refused while engaged;
  * a platform change re-derives `z_out_channel`;
  * no `#instrumentPanel` remains in the DOM.

The header (and so `#configBtn`) shows on every tab
(Scan / Connect / Edit / Write).
"""

# ruff: noqa: F811 -- fixtures imported by name shadow their definitions when taken
# as test parameters; the standard pytest cross-file fixture-reuse pattern (see the
# same note in test_instrument_cockpit_b3b.py).

from __future__ import annotations

import re

from test_instrument_cockpit import (  # reuse plumbing, see module docstring
    WEB_DIR,
    _ensure_withdrawn,
    _seed_amp_gain_override,
    browser,  # noqa: F401 -- fixture re-export
    page,  # noqa: F401 -- fixture re-export
    service,  # noqa: F401 -- fixture re-export
)

# Connect is hidden from the tab strip (docs/UI_LAYOUT_SPEC.md), so it is not a
# tab the operator can open the dialog from.
NORMAL_TABS = ["scan", "edit", "write", "file"]


# ------------------------------------------------------------------ helpers --

def _open_config(page) -> None:
    """Open the Config dialog from the header and wait for it."""
    page.click("#configBtn", timeout=5000)
    page.wait_for_selector("#configDialog", state="visible", timeout=5000)
    page.wait_for_timeout(200)


def _close_config(page) -> None:
    btn = page.query_selector("#configClose")
    if btn and btn.is_visible():
        btn.click()
        page.wait_for_timeout(150)


def _row_value_text(page, field: str) -> str | None:
    el = page.query_selector(f'#configRows tr[data-field="{field}"] .configval')
    return (el.text_content() or "").strip() if el else None


def _instrument_model(service) -> dict:
    status, model = service.get("/api/instrument")
    assert status == 200, (status, model)
    return model


# ------------------------------------------------------- no instrument panel --

def test_no_instrument_panel_remains_in_the_dom(page):
    """Section 14 item 1/5: `#instrumentPanel` is removed entirely."""
    assert page.query_selector("#instrumentPanel") is None, (
        "#instrumentPanel is still in the DOM after section 14 removed it")
    # and it is not in the served markup either
    html = (WEB_DIR / "edit-cockpit.html").read_text(encoding="utf-8")
    assert 'id="instrumentPanel"' not in html, (
        "edit-cockpit.html still ships an #instrumentPanel")


def test_config_dialog_opens_from_every_normal_tab(page):
    """Section 14 item 5: the dialog opens from every tab (the header's
    Config button shows on all four normal tabs)."""
    for tab in NORMAL_TABS:
        page.click(f'button.tab[data-tab="{tab}"]', timeout=5000)
        page.wait_for_timeout(150)
        cfg = page.query_selector("#configBtn")
        assert cfg is not None and cfg.is_visible(), (
            f"#configBtn not visible on the {tab} tab")
        _open_config(page)
        dlg = page.query_selector("#configDialog")
        assert dlg is not None and dlg.is_visible(), (
            f"#configDialog did not open from the {tab} tab")
        # every model field is shown as a row
        rows = page.eval_on_selector_all(
            "#configRows tr", "els => els.map(e => e.getAttribute('data-field'))")
        assert "tip_amp_gain" in rows and "z_out_channel" in rows, rows
        _close_config(page)
        assert not page.query_selector("#configDialog").is_visible(), (
            f"#configDialog did not close on the {tab} tab")


# ----------------------------------------------------------- edit / cancel --

def test_edit_then_cancel_restores_and_sends_nothing(service, page):
    """Section 14 item 3: Cancel restores the pre-edit values and posts nothing."""
    _open_config(page)
    before = _row_value_text(page, "z_park_tol_v")
    assert before is not None

    overrides = []
    page.on("request", lambda req: overrides.append(req.url)
            if "/api/instrument/override" in req.url else None)

    page.click("#configEdit", timeout=5000)
    page.wait_for_selector('[data-field-input="z_park_tol_v"]', timeout=5000)
    page.fill('[data-field-input="z_park_tol_v"]', "0.987")
    page.wait_for_timeout(100)

    page.click("#configCancel", timeout=5000)
    page.wait_for_timeout(200)

    # back to read-only, restored value, no override posted
    assert page.query_selector("#configEdit").is_visible(), "Edit did not come back after Cancel"
    save = page.query_selector("#configSave")
    assert save is None or not save.is_visible(), "Save still visible after Cancel"
    after = _row_value_text(page, "z_park_tol_v")
    assert after == before, f"Cancel did not restore z_park_tol_v ({before!r} -> {after!r})"
    assert not overrides, f"Cancel posted an override: {overrides}"


# ------------------------------------------------------------- edit / save --

def test_edit_then_save_persists_across_reload_and_on_disk(service, page):
    """Section 14 item 3/5: Save posts an override for the changed field, then
    the value survives a browser reload (re-fetched from the service) and is on
    disk in `[override]` (so it survives a service restart, which reads the file
    at startup)."""
    new_val = 43.0
    _open_config(page)
    page.click("#configEdit", timeout=5000)
    page.wait_for_selector('[data-field-input="free_air_max_age_s"]', timeout=5000)
    page.fill('[data-field-input="free_air_max_age_s"]', str(int(new_val)))
    page.click("#configSave", timeout=5000)
    # Save re-reads the model and closes on success.
    page.wait_for_selector("#configDialog", state="hidden", timeout=8000)

    # service now reports the override
    model = _instrument_model(service)
    assert float(model["free_air_max_age_s"]["value"]) == new_val, model["free_air_max_age_s"]
    assert model["free_air_max_age_s"]["source"] == "override", model["free_air_max_age_s"]

    # on disk under [override] (survives a restart)
    toml_text = (service.runtime / "instrument.toml").read_text(encoding="utf-8")
    assert "[override]" in toml_text, toml_text
    assert re.search(r"free_air_max_age_s\s*=\s*43", toml_text), toml_text

    # survives a browser reload (value + source re-fetched from the service)
    page.reload(wait_until="load", timeout=45000)
    page.wait_for_timeout(600)
    _open_config(page)
    shown = _row_value_text(page, "free_air_max_age_s")
    assert shown is not None and float(shown) == new_val, (
        f"reloaded dialog shows {shown!r}, expected {new_val}")
    src = page.query_selector(
        '#configRows tr[data-field="free_air_max_age_s"] .configsrc')
    assert src and "override" in (src.text_content() or ""), (
        "the reloaded row does not show the override source")


def test_refused_value_stays_in_edit_mode_with_reason_and_keeps_prior_field(
        service, page):
    """Section 14 item 3: a refused field (here `invalid_params` -- a
    contact_setpoint_band_v that is not a 2-element band) keeps the dialog in
    edit mode with the reason on that row; a valid field CHANGED BEFORE it in
    order stays applied (durable on disk)."""
    _open_config(page)
    model_before = _instrument_model(service)
    zpv_before = float(model_before["z_park_v"]["value"])
    zpv_new = zpv_before + 0.25

    page.click("#configEdit", timeout=5000)
    page.wait_for_selector('[data-field-input="z_park_v"]', timeout=5000)
    # a valid change BEFORE contact_setpoint_band_v in CONFIG_FIELDS order
    page.fill('[data-field-input="z_park_v"]', str(zpv_new))
    # an invalid band (a single value, not [lo, hi]) -> invalid_params
    page.fill('[data-field-input="contact_setpoint_band_v"]', "5")
    page.click("#configSave", timeout=5000)
    page.wait_for_timeout(500)

    # dialog stays open, still in edit mode (inputs present, Save visible)
    assert page.query_selector("#configDialog").is_visible(), (
        "dialog closed despite a refused field")
    assert page.query_selector('[data-field-input="contact_setpoint_band_v"]') is not None, (
        "dialog left edit mode after a refused field")
    err = page.query_selector('[data-field-err="contact_setpoint_band_v"]')
    assert err is not None and (err.text_content() or "").strip(), (
        "the refused row shows no reason")
    assert "invalid_params" in (err.text_content() or "") or "refused" in (err.text_content() or ""), (
        f"the refused row does not name the reason: {err.text_content()!r}")

    # the earlier valid field WAS applied and is durable on disk
    model_after = _instrument_model(service)
    assert float(model_after["z_park_v"]["value"]) == zpv_new, (
        "the field before the refusal was rolled back (must stay applied)")
    assert model_after["z_park_v"]["source"] == "override", model_after["z_park_v"]
    toml_text = (service.runtime / "instrument.toml").read_text(encoding="utf-8")
    assert re.search(rf"z_park_v\s*=\s*{re.escape(str(zpv_new))}", toml_text), toml_text


# ------------------------------------------------- item 4: platform in dialog --

def test_platform_change_refused_while_engaged(service, page):
    """Section 14 item 4/5: a platform change from the dialog is refused while the
    tip is engaged (reason `engaged`); the dialog stays in edit mode with the
    reason on the platform row, and nothing is written to disk."""
    _ensure_withdrawn(service)
    # Engage the twin so _run_active() is true (approach sets hub.engaged).
    status, r = service.post("/api/approach", {"mode": "contact"})
    assert status == 200 and not (r or {}).get("error"), (status, r)
    try:
        _open_config(page)
        page.click("#configEdit", timeout=5000)
        page.wait_for_selector('[data-field-input="platform"]', timeout=5000)
        page.fill('[data-field-input="platform"]', "mfp3d")
        page.click("#configSave", timeout=5000)
        page.wait_for_timeout(500)

        assert page.query_selector("#configDialog").is_visible(), (
            "dialog closed despite the engaged refusal")
        assert page.query_selector('[data-field-input="platform"]') is not None, (
            "dialog left edit mode after the engaged refusal")
        err = page.query_selector('[data-field-err="platform"]')
        assert err is not None and "engaged" in (err.text_content() or ""), (
            f"the platform row does not show the engaged refusal: "
            f"{err.text_content() if err else None!r}")
        # nothing hit the file
        toml_text = (service.runtime / "instrument.toml").read_text(encoding="utf-8")
        tail = toml_text.split("[override]", 1)[1] if "[override]" in toml_text else ""
        assert "platform" not in tail, "a refused platform change still hit the file"
    finally:
        _ensure_withdrawn(service)      # leave the twin retracted


def test_platform_change_rederives_z_out_channel(service, page):
    """Section 14 item 4/5: an accepted platform change from the dialog re-derives
    the platform defaults. Both platforms drive Z by the alias Height (M2), so the
    switch to `cypher` leaves z_out_channel at Height; the model and the dialog must
    agree after the change. (Amp gain AND the sample tip-voltage window are seeded to
    the file first so the cypher model is not model_incomplete for want of a
    config-owned tip_amp_gain / tip_voltage_window_v.)"""
    _ensure_withdrawn(service)
    _seed_amp_gain_override(service)     # persist tip_amp_gain so cypher is complete
    # R1/S1: on a hardware platform (cypher) a missing [sample] tip_voltage_window_v
    # raises ModelIncomplete, so persist a window override alongside the gain -- the
    # same override path -- before the switch, or the platform save refuses
    # model_incomplete.
    status, r = service.post("/api/instrument/override", {
        "field": "tip_voltage_window_v", "value": [-8.0, 30.0],
        "reason": "test", "confirm": True})
    assert status == 200 and r and r.get("applied"), (
        f"POST /api/instrument/override (window) did not apply: {status} {r}")

    status, model = service.get("/api/instrument")
    assert status == 200, (status, model)
    assert model["z_out_channel"]["value"] == "Height", model["z_out_channel"]

    _open_config(page)
    page.click("#configEdit", timeout=5000)
    page.wait_for_selector('[data-field-input="platform"]', timeout=5000)
    page.fill('[data-field-input="platform"]', "cypher")
    page.click("#configSave", timeout=5000)
    page.wait_for_selector("#configDialog", state="hidden", timeout=8000)

    status, model2 = service.get("/api/instrument")
    assert status == 200, (status, model2)
    assert model2["platform"]["value"] == "cypher", model2["platform"]
    assert model2["z_out_channel"]["value"] == "Height", (
        f"platform change did not re-derive z_out_channel: {model2['z_out_channel']}")

    # the reopened dialog shows the re-derived value
    _open_config(page)
    assert _row_value_text(page, "z_out_channel") == "Height", (
        "the reopened dialog does not show the re-derived z_out_channel")
