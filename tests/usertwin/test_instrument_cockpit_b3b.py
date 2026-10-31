"""Coverage for the B3b adversarial-review must-fixes
(docs/INSTRUMENT_MODEL_SPEC.md sec 13.1 items 11-14), against the real edit
cockpit in a real browser -- same spirit as `test_instrument_cockpit.py`,
which this file reuses plumbing from rather than re-inventing it (the
`Service`/`_FlexClient` classes, the `service`/`flex`/`browser`/`page`
fixtures, and the seeding helpers are imported, not copied).

Items covered:

  11. The FLEX doors `set_objects`/`load_pattern` did NOT broadcast
      `design_changed` -- a script's change left the cockpit and
      `#writeEstimate` stale (the incident door). Tests here drive BOTH over
      the real FLEX ZMQ socket (not a second REST client, which the base
      file already covers) and check the cockpit re-adopts without a reload,
      and that a disabled object is excluded from `#perform`'s explicit id
      list.
  12. Guard refusals were log-only (`perform()` swallowed `{error, reason}`).
      Tests trigger an `amp_gain_mismatch` refusal (the write door no longer
      refuses over free air, docs/PER_OBJECT_FREE_AIR_SPEC.md)
      through the cockpit's OWN `ec-api.js:api()` (via a dynamic import in
      the page, so the refusal hook that `ec-instrument.js:initInstrument()`
      registered is the SAME one that fires -- not a synthetic DOM poke),
      and check `#guardNotice` becomes visible with the reason text, in both
      normal and presentation-tab view, and never overlaps `#abortBtn`. A
      third test drives the SAME refusal by first mutating an object's
      voltage out of the sample's window over the FLEX door (set_objects
      does not itself validate the window yet -- known gap 13.1 item 9, a
      different owner -- so the refusal only appears at the next write) to
      prove the FLEX-originated case is surfaced too, not just a
      REST-originated one.
  13. `litho-cockpit.html` was a divergent served cockpit; it now redirects
      to `/edit-cockpit` and can no longer write. `litho-debug.html`'s
      "Write all" stopped sending `{all: true}`.
  14. An empty `#lithoSetpoint` used to hydrate to 0 and write at free air;
      it must instead disable `#perform`, show `#lithoSetpointHint`, issue
      no write request, and re-enable once a value is typed.
"""

# ruff: noqa: F811 -- every helper/test below takes `service`/`page`/`flex`/
# `browser` as a parameter, shadowing the fixture functions of the same name
# imported below (the standard pytest cross-file fixture-reuse pattern: a
# fixture must live in this module's own namespace, under its own name, to be
# discovered here). That is a normal nested-scope parameter, not an actual
# redefinition; ruff does not special-case it the way pytest-aware linters do.

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from test_instrument_cockpit import (  # reuse plumbing, see module docstring
    _current_amp_gain,
    _enabled_object_names,
    _object_rows,
    _seed_free_air,
    _to_write_tab,
    browser,  # noqa: F401 -- fixture re-export
    flex,  # noqa: F401 -- fixture re-export
    page,  # noqa: F401 -- fixture re-export
    service,  # noqa: F401 -- fixture re-export
)

# ------------------------------------------------------------------ helpers --

def _get_raw(service, path: str) -> tuple[int, str]:
    """A GET that returns the raw response body (not JSON-decoded) -- needed
    for `litho-cockpit.html`, which is HTML, not a JSON-RPC body."""
    req = urllib.request.Request(service.url + path)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def _instrument_model(service) -> dict:
    status, model = service.get("/api/instrument")
    assert status == 200, (status, model)
    return model


def _add_object(service, name: str, voltage: float = 5.0) -> str:
    status, r = service.post("/api/design/add", {
        "kind": "path", "name": name, "voltage": voltage,
        "points_um": [[1.0, 1.0], [6.0, 6.0]]})
    assert status == 200 and r and r.get("ok"), (status, r)
    return name


def _api_write_via_page(page, *, objects, setpoint_relative, amp_gain) -> dict:
    """Fire `POST /api/litho/write` through the COCKPIT'S OWN `ec-api.js:api()`
    (a dynamic `import()` in the page, resolving to the SAME module instance
    `ec-main.js`'s `<script type="module">` tag already loaded -- the browser
    dedups ES module instances by resolved URL) rather than a raw `fetch` or a
    server-side POST. This is deliberate: item 12 is about the COCKPIT'S
    refusal-sink wiring (`ec-api.js`'s `onRefusal` hook -> `ec-instrument.js:
    showRefusal` -> `#guardNotice`), and only a call through the real `api()`
    exercises that hook the way `#perform`'s click handler does. `#perform`
    itself cannot construct an intentionally-wrong body (amp_gain always comes
    from the read-only model), so this is the only way to deterministically
    provoke `amp_gain_mismatch` from inside the browser."""
    return page.evaluate(
        """async ({objects, setpoint_relative, amp_gain}) => {
             const m = await import('/static/ec-api.js');
             return await m.api('litho/write', {objects, setpoint_relative, amp_gain});
           }""",
        {"objects": objects, "setpoint_relative": setpoint_relative,
         "amp_gain": amp_gain})


def _ensure_flex_control(flex) -> None:
    """Re-acquire FLEX control if a REST `TIP_MUTATING` POST since the last
    acquire revoked it (`dispatch`'s operator_tip gate), WITHOUT blowing up
    when it did not: re-sending `acquireControl` while `flex`'s own token is
    still the live holder is refused `-32040 control_held` (the server does
    not special-case "the same client already holds it" -- `acquire_control`
    just checks whether ANYONE unexpired holds it), which is not an error
    here -- `flex.token` is already good, keep it."""
    reply = flex.send("acquireControl",
                      {"client": "im-tests-T4-usertwin", "deadman_s": 300})
    if "result" in reply:
        flex.token = reply["result"]["token"]
    elif reply.get("error", {}).get("code") != -32040:   # not "control_held"
        raise AssertionError(f"acquireControl failed unexpectedly: {reply}")
    # else: -32040 control_held means our own prior token is still the live
    # holder (nothing preempted it) -- flex.token from the earlier acquire()
    # remains valid, no action needed.


def _clear_safety_banner(page) -> None:
    """`_seed_free_air`'s withdraw fires a real (non-critical) `safe_park`
    broadcast that `ec-safety.js` shows as `#ecSafetyBanner`, docked over the
    tab bar for up to 8 s -- long enough to intercept a `page.click` on the
    Write tab right after seeding on an ALREADY-CONNECTED
    page (unlike the base file's tests, which reload the page after seeding,
    getting a fresh websocket that never sees the stale frame). Hide it
    directly rather than adding an unconditional reload to every test here."""
    page.evaluate(
        "() => { const el = document.getElementById('ecSafetyBanner'); "
        "if (el) el.style.display = 'none'; }")


def _guard_notice_text(page) -> str | None:
    el = page.query_selector("#guardNotice")
    if el is None or not el.is_visible():
        return None
    txt = page.query_selector("#guardNoticeText")
    return (txt.text_content() or "").strip() if txt else ""


# --------------------------------------------------- item 12: guard refusals --

def test_amp_gain_mismatch_refusal_renders_visible_guardnotice(service, page):
    """Seeds a real free-air baseline (so the ONLY thing wrong with the write is
    the gain) then sends a deliberately wrong `amp_gain` -- refused
    `amp_gain_mismatch` -- and checks the same visible-notice contract."""
    _seed_free_air(service)
    _clear_safety_banner(page)
    name = _add_object(service, f"agm-{int(time.time() * 1000)}")
    wrong_gain = _current_amp_gain(service) + 3.5
    _to_write_tab(page)

    result = _api_write_via_page(
        page, objects=[name], setpoint_relative=0.2, amp_gain=wrong_gain)
    assert result and result.get("reason") == "amp_gain_mismatch", (
        f"expected an amp_gain_mismatch refusal, got {result}")

    page.wait_for_timeout(300)
    shown = _guard_notice_text(page)
    assert shown is not None, "#guardNotice did not become visible on a guard refusal"
    assert "amp_gain_mismatch" in shown, (
        f"#guardNotice does not name the refusal reason: {shown!r}")


def test_guardnotice_never_overlaps_abortbtn_when_visible(service, page):
    """Should-fix parity with the existing critical-notice test: a docked
    `#guardNotice` must never cover `#abortBtn`'s centre OR a corner (section
    4: "Critical notices never overlap the Abort control")."""
    _seed_free_air(service)
    _clear_safety_banner(page)
    name = _add_object(service, f"overlap-{int(time.time() * 1000)}")
    wrong_gain = _current_amp_gain(service) + 3.5
    _to_write_tab(page)

    result = _api_write_via_page(
        page, objects=[name], setpoint_relative=0.2, amp_gain=wrong_gain)
    assert result and result.get("reason") == "amp_gain_mismatch", result
    page.wait_for_timeout(300)

    notice = page.query_selector("#guardNotice")
    assert notice is not None and notice.is_visible(), "#guardNotice not visible"

    abort_btn = page.query_selector("#abortBtn")
    assert abort_btn is not None, "#abortBtn not found"
    box = abort_btn.bounding_box()
    assert box is not None
    # Baseline first (must run BEFORE the assertion below reads too much into a
    # failure): a rounded corner can legitimately show the element behind the
    # button at that exact pixel with NO notice up at all, which would be a
    # CSS artifact, not a #guardNotice overlap. Only points that resolve to
    # #abortBtn with the notice hidden are meaningful to re-check with it shown
    # -- same two points test_abort_button_stays_clickable_under_a_critical_notice
    # (test_instrument_cockpit.py) already established as button, not rounding.
    candidates = {
        "centre": (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2),
        "top-right corner": (box["x"] + box["width"] - 1, box["y"] + 1),
        "bottom-left corner": (box["x"] + 1, box["y"] + box["height"] - 1),
    }
    notice.evaluate("e => e.style.display = 'none'")
    meaningful = {}
    for label, (x, y) in candidates.items():
        top_id = page.evaluate(
            "([x, y]) => { const e = document.elementFromPoint(x, y); "
            "return e ? (e.id || e.closest('[id]')?.id || e.tagName) : null; }",
            [x, y])
        if top_id == "abortBtn":
            meaningful[label] = (x, y)
    assert meaningful, "no point of #abortBtn's own box resolves to #abortBtn at rest"
    notice.evaluate("e => e.style.display = ''")

    for label, (x, y) in meaningful.items():
        top_id = page.evaluate(
            "([x, y]) => { const e = document.elementFromPoint(x, y); "
            "return e ? (e.id || e.closest('[id]')?.id || e.tagName) : null; }",
            [x, y])
        assert top_id == "abortBtn", (
            f"#guardNotice covers #abortBtn's {label}: elementFromPoint -> {top_id!r}")


def test_flex_out_of_window_voltage_refusal_is_visible_not_log_only(service, page, flex):
    """B3 review must-fix 11/12 combined: a write-time `tip_voltage_out_of_window`
    refusal, on an object a SCRIPT (not the cockpit) mutated, must still render the
    visible `#guardNotice` -- proving the sink is not wired to only the cockpit's own
    writes.

    Since 13.1 item 19 (D3) the FLEX `setObjects` door ALSO validates the window, so
    the out-of-window voltage can no longer be staged into the design through it
    directly. Widen the sample window for the duration of the FLEX mutation (a real
    such voltage would carry a window that admits it), then restore the window, so the
    script's mutation succeeds and the *write* -- under the real window -- is the door
    that refuses. The refusal is what must reach `#guardNotice`."""
    _seed_free_air(service)
    _clear_safety_banner(page)
    name = _add_object(service, f"flexwin-{int(time.time() * 1000)}", voltage=5.0)
    model = _instrument_model(service)
    _lo, hi = model["tip_voltage_window_v"]["value"]
    too_high = float(hi) + 50.0

    # `_seed_free_air` POSTs to /api/scan/continuous (a TIP_MUTATING REST path) --
    # `dispatch`'s operator_tip gate revokes ANY existing control on such a POST
    # (plan Sec E.2's GUI-preemption rule), including the token `flex` acquired
    # during its own fixture setup a moment earlier. Re-acquire right before the
    # FLEX call rather than assuming the fixture's token is still live.
    _ensure_flex_control(flex)
    # D3 (13.1 item 19): setObjects now refuses an out-of-window voltage, so widen the
    # window for the mutation and restore it before the write, leaving the write as the
    # door that refuses (the object is left at `too_high` in the design either way).
    status, _r = service.post("/api/instrument/override", {
        "field": "tip_voltage_window_v", "value": [_lo, too_high + 1.0],
        "reason": "test D3: admit the staged voltage for the FLEX mutation",
        "confirm": True})
    assert status == 200, (status, _r)
    try:
        r = flex.call("setObjects", {"token": flex.token,
                                     "updates": {name: {"voltage": too_high}}})
        assert r is not None  # accepted under the widened window
    finally:
        status, _r = service.post("/api/instrument/override", {
            "field": "tip_voltage_window_v", "value": [_lo, hi],
            "reason": "test D3: restore the real window before the write",
            "confirm": True})
        assert status == 200, (status, _r)

    gain = _current_amp_gain(service)
    _to_write_tab(page)
    result = _api_write_via_page(
        page, objects=[name], setpoint_relative=0.2, amp_gain=gain)
    assert result and result.get("reason") == "tip_voltage_out_of_window", (
        f"expected tip_voltage_out_of_window, got {result}")

    page.wait_for_timeout(300)
    shown = _guard_notice_text(page)
    assert shown is not None, (
        "a FLEX-originated write refusal did not render #guardNotice "
        "(log-only regression)")
    assert "tip_voltage_out_of_window" in shown, shown


# ---------------------------------------- item 11: FLEX design_changed doors --

def test_flex_set_objects_disable_excludes_object_from_explicit_write_ids(
        service, page, flex):
    """`setObjects` over the real FLEX socket disables an object; the cockpit
    must re-adopt (`design_changed`) WITHOUT a reload, and `#perform`'s
    explicit id list must exclude the now-disabled object -- never
    `{all:true}`, and never a stale id list from before the FLEX mutation."""
    keep = _add_object(service, f"flexkeep-{int(time.time() * 1000)}")
    drop = _add_object(service, f"flexdrop-{int(time.time() * 1000)}")
    _to_write_tab(page)

    deadline = time.time() + 10
    names = []
    while time.time() < deadline:
        names = [r["name"] for r in _object_rows(page)]
        if keep in names and drop in names:
            break
        page.wait_for_timeout(200)
    assert keep in names and drop in names, (
        f"cockpit did not pick up both added objects: {names}")

    # Any earlier test's REST write (/api/litho/write is TIP_MUTATING) may have
    # preempted this module-scoped `flex` fixture's control; re-acquire so this
    # test does not depend on running-order to have a live token.
    _ensure_flex_control(flex)
    r = flex.call("setObjects", {"token": flex.token,
                                 "updates": {drop: {"enabled": False}}})
    assert r is not None

    deadline = time.time() + 10
    enabled = []
    while time.time() < deadline:
        enabled = _enabled_object_names(page)
        if drop not in enabled and keep in enabled:
            break
        page.wait_for_timeout(200)
    assert drop not in enabled, (
        f"cockpit still shows {drop!r} enabled after a FLEX setObjects disable "
        f"(design_changed not broadcast/consumed): {enabled}")
    assert keep in enabled

    captured = {}

    def handler(route):
        try:
            captured["body"] = json.loads(route.request.post_data or "{}")
        except Exception:      # noqa: BLE001
            captured["body"] = None
        route.fulfill(status=200, content_type="application/json",
                     body=json.dumps({"ok": True, "writing": 0, "skipped": 0,
                                       "run_id": "flex-set-objects-test"}))

    page.route("**/api/litho/write", handler)
    perform = page.query_selector("#perform")
    assert perform is not None
    # The setpoint may be empty at this point (item 14) -- fill it so the click
    # is not refused by the empty-setpoint gate before it can build a body.
    sp = page.query_selector("#lithoSetpoint")
    if sp is not None and (sp.evaluate("e => e.value") or "") == "":
        sp.fill("0.3")
    perform.click()
    page.wait_for_timeout(500)

    assert "body" in captured, "#perform did not POST /api/litho/write"
    body = captured["body"]
    assert body is not None
    assert "all" not in body, f"#perform still sends 'all': {body}"
    assert drop not in body.get("objects", []), (
        f"#perform sent the FLEX-disabled object {drop!r}: {body}")
    assert keep in body.get("objects", []), (
        f"#perform dropped the still-enabled {keep!r}: {body}")


def test_flex_load_pattern_broadcasts_design_changed_and_cockpit_rehydrates(
        service, page, flex):
    """`loadPattern` (the `objects` door, a full pattern REPLACE) over FLEX
    must also broadcast `design_changed` -- the base file's REST-only
    `set_props` mutation test does not cover this door at all."""
    _to_write_tab(page)
    before = {r["name"] for r in _object_rows(page)}

    unique = f"flexload-{int(time.time() * 1000)}"
    _ensure_flex_control(flex)   # same TIP_MUTATING-preemption reasoning as above
    r = flex.call("loadPattern", {"token": flex.token, "objects": [
        {"kind": "region", "name": unique, "shape": "rect",
         "x_um": 2.0, "y_um": 2.0, "w_um": 3.0, "h_um": 3.0,
         "pitch_um": 0.5, "voltage": 6.0},
    ]})
    assert r is not None and not (isinstance(r, dict) and r.get("error")), r

    deadline = time.time() + 10
    names = set()
    while time.time() < deadline:
        names = {row["name"] for row in _object_rows(page)}
        if unique in names:
            break
        page.wait_for_timeout(200)
    assert unique in names, (
        f"cockpit did not pick up a FLEX loadPattern replacement without a "
        f"reload (before={before}, after={names})")


# ----------------------------------------------------- item 13: no {all:true} --

_WEB = Path(__file__).resolve().parents[2] / "src" / "afm_litho" / "web"
_needs_old_pages = pytest.mark.skipif(
    not ((_WEB / "litho-cockpit.html").exists() and (_WEB / "litho-debug.html").exists()),
    reason="the old litho pages are not in this tree")

@_needs_old_pages
def test_litho_cockpit_html_redirects_to_edit_cockpit_and_has_no_write_control(service):
    status, body = _get_raw(service, "/static/litho-cockpit.html")
    assert status == 200, (status, body[:500])
    assert "/edit-cockpit" in body, (
        f"litho-cockpit.html no longer redirects to /edit-cockpit: {body[:800]!r}")
    assert 'id="perform"' not in body, (
        "litho-cockpit.html still ships its own write control instead of "
        "redirecting to the maintained cockpit")


@_needs_old_pages
def test_litho_debug_write_all_sends_explicit_ids_never_all_true(service, browser):
    """Exercised live (not just grepped): loads `/static/litho-debug.html`,
    lets it build its demo pattern, clicks "Write all", and captures the
    intercepted POST body."""
    ctx = browser.new_context(viewport={"width": 1400, "height": 900})
    p = ctx.new_page()
    try:
        p.goto(service.url + "/static/litho-debug.html", wait_until="load", timeout=45000)
        p.wait_for_timeout(800)      # buildDemoPattern() runs on load

        captured = {}

        def handler(route):
            try:
                captured["body"] = json.loads(route.request.post_data or "{}")
            except Exception:      # noqa: BLE001
                captured["body"] = None
            route.fulfill(status=200, content_type="application/json",
                         body=json.dumps({"writing": 0, "skipped": 0}))

        p.route("**/api/litho/write", handler)
        write_all = p.query_selector("#writeAll")
        assert write_all is not None, "#writeAll not found on litho-debug.html"
        write_all.click()
        p.wait_for_timeout(500)

        assert "body" in captured, "#writeAll did not POST /api/litho/write"
        body = captured["body"]
        assert body is not None
        assert "all" not in body, f"litho-debug Write all still sends 'all': {body}"
        assert body.get("objects"), (
            f"litho-debug Write all did not send an explicit non-empty objects "
            f"list: {body}")
    finally:
        ctx.close()


# --------------------------------------------------------- item 14: empty sp --

def test_empty_lithosetpoint_disables_perform_shows_hint_and_reenables(service, page):
    _seed_free_air(service)
    _clear_safety_banner(page)
    _add_object(service, f"emptysp-{int(time.time() * 1000)}")
    _to_write_tab(page)

    sp = page.query_selector("#lithoSetpoint")
    hint = page.query_selector("#lithoSetpointHint")
    perform = page.query_selector("#perform")
    assert sp is not None and hint is not None and perform is not None

    sp.fill("")
    page.wait_for_timeout(200)
    assert hint.is_visible(), "#lithoSetpointHint is not shown while #lithoSetpoint is empty"
    assert perform.is_disabled(), "#perform is enabled with an empty #lithoSetpoint"

    sp.fill("0.42")
    page.wait_for_timeout(200)
    assert not hint.is_visible(), "#lithoSetpointHint stayed up after a value was typed"
    assert not perform.is_disabled(), "#perform stayed disabled after a value was typed"


def test_empty_lithosetpoint_issues_no_write_request(service, page):
    """N2: an empty setpoint must refuse rather than coerce to 0 -- checked at
    the network layer (no `/api/litho/write` request at all), not just the
    disabled-button DOM state above."""
    _seed_free_air(service)
    _clear_safety_banner(page)
    _add_object(service, f"emptysp2-{int(time.time() * 1000)}")
    _to_write_tab(page)

    requests = []
    page.on("request", lambda req: requests.append(req.url)
            if "/api/litho/write" in req.url else None)

    sp = page.query_selector("#lithoSetpoint")
    perform = page.query_selector("#perform")
    assert sp is not None and perform is not None
    sp.fill("")
    page.wait_for_timeout(200)

    # Keyboard-activation belt-and-suspenders: dispatch a click even though the
    # button is disabled (a disabled button swallows a real click; this proves
    # the click HANDLER also refuses, not just the DOM disabled attribute).
    perform.evaluate("e => e.click()")
    page.wait_for_timeout(400)

    assert not requests, (
        f"a write request went out with an empty #lithoSetpoint: {requests}")
