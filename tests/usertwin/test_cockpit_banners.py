"""Cockpit banners fit their text and are all clearable
(docs/COCKPIT_BANNERS_SPEC.md, spec 13.1 item 24), in a real browser against the
twin service.

Every docked banner (`#guardNotice`, `#ecSafetyBanner`) must
sit in the message row under the header, overlap neither the tab strip nor
`#abortBtn`, and carry a close control. A closed STATE banner collapses to a chip in
the message row that re-expands
it, stays collapsed while its condition holds, drops its chip when the condition
clears, and shows in full on the next occurrence. A closed EVENT notice goes
entirely. Closing never re-enables `#perform` while the Write gate holds.

Plumbing (service subprocess, browser, page with the ws-capture hook) is reused
from `test_instrument_cockpit.py`.
"""

# ruff: noqa: F811 -- tests take the imported fixtures as parameters (see
# test_instrument_cockpit_b3b.py for the same pattern).

from __future__ import annotations

import json
import shutil

import pytest
from _service import Service
from test_instrument_cockpit import (
    _WS_CAPTURE_INIT,
    _emit_critical_safe_park,
    _ensure_withdrawn,
    _to_write_tab,
    browser,  # noqa: F401 -- fixture re-export
    page,  # noqa: F401 -- fixture re-export
    service,  # noqa: F401 -- fixture re-export
)

# ------------------------------------------------------------------ helpers --

def _box(page, selector: str) -> dict:
    el = page.query_selector(selector)
    assert el is not None, f"{selector} not found"
    box = el.bounding_box()
    assert box is not None, f"{selector} has no bounding box (not rendered)"
    return box


def _overlaps(a: dict, b: dict) -> bool:
    return (a["x"] < b["x"] + b["width"] and b["x"] < a["x"] + a["width"]
            and a["y"] < b["y"] + b["height"] and b["y"] < a["y"] + a["height"])


def _assert_docked_clear(page, selector: str) -> None:
    """The banner lies inside the message row and the window, and clears the tab
    strip, the header (Abort included) and the zoom bar; Abort's centre still hits
    Abort."""
    el = page.query_selector(selector)
    assert el is not None and el.is_visible(), f"{selector} is not visible"
    box = _box(page, selector)
    row = _box(page, "#ecMessageRow")
    vw = page.viewport_size["width"]
    tol = 0.6
    assert (box["x"] >= row["x"] - tol and box["y"] >= row["y"] - tol
            and box["x"] + box["width"] <= min(row["x"] + row["width"], vw) + tol
            and box["y"] + box["height"] <= row["y"] + row["height"] + tol), (
        f"{selector} is not inside the message row and the window: {box} / {row}")
    for other in ("#abortBtn", "header"):
        assert not _overlaps(box, _box(page, other)), f"{selector} overlaps {other}: {box}"
    strip = page.query_selector(".tabstrip")
    if strip is not None and strip.is_visible():
        assert not _overlaps(box, _box(page, ".tabstrip")), (
            f"{selector} overlaps the tab strip: {box}")
    bar = page.query_selector(".rendertools")
    if bar is not None and bar.is_visible():
        assert not _overlaps(box, _box(page, ".rendertools")), (
            f"{selector} overlaps the render toolbar (.rendertools): {box}")
    ab = _box(page, "#abortBtn")
    top_id = page.evaluate(
        "([x, y]) => { const e = document.elementFromPoint(x, y); "
        "return e ? (e.id || e.closest('[id]')?.id || e.tagName) : null; }",
        [ab["x"] + ab["width"] / 2, ab["y"] + ab["height"] / 2])
    assert top_id == "abortBtn", f"#abortBtn's centre resolves to {top_id!r}"


def _close(page, selector: str) -> None:
    btn = page.query_selector(f"{selector} .banner-close")
    assert btn is not None, f"{selector} has no close control (.banner-close)"
    btn.click()
    page.wait_for_timeout(150)


def _chip(page, banner_id: str):
    return page.query_selector(f'#ecBannerChips .banner-chip[data-banner="{banner_id}"]')


def _chip_visible(page, banner_id: str) -> bool:
    chip = _chip(page, banner_id)
    return chip is not None and chip.is_visible()


def _approach(service) -> None:
    status, r = service.post("/api/approach", {"mode": "contact"})
    assert status == 200 and r and not r.get("error"), (status, r)


def _hide_park_notice(page) -> None:
    """A withdraw fires a non-critical `safe_park` notice (auto-hides after 8 s);
    close it so it does not stack into the dock under test."""
    page.evaluate(
        "() => { const b = document.querySelector('#ecSafetyBanner .banner-close'); "
        "const el = document.getElementById('ecSafetyBanner'); "
        "if (b && el && el.style.display !== 'none') b.click(); }")


def _emit_safe_park(page, **frame) -> None:
    body = {"type": "safe_park", "reason": "usertwin banner test", "transport": "test"}
    body.update(frame)
    page.evaluate(
        "(data) => window.__wsInstances.forEach(ws => ws.dispatchEvent("
        "new MessageEvent('message', {data})))", json.dumps(body))
    page.wait_for_timeout(300)


def _refuse_next_write(page, reason: str = "fault_latched") -> None:
    """Answer the cockpit's next `litho/write` with a guard refusal, so the real
    ec-api refusal sink -> `#guardNotice` path runs without needing the service
    to be in a refusing state."""
    def handler(route):
        route.fulfill(status=200, content_type="application/json",
                      body=json.dumps({"error": "usertwin: fault latched",
                                       "reason": reason}))
    page.route("**/api/litho/write", handler)


def _fire_write_via_page(page) -> dict:
    return page.evaluate(
        """async () => {
             const m = await import('/static/ec-api.js');
             return await m.api('litho/write',
               {objects: ['x'], setpoint_relative: 0.2, amp_gain: 25});
           }""")


# ------------------------------------------------------ guard refusal (event) --

def test_guard_notice_fits_clears_and_closes_entirely(service, page):
    _to_write_tab(page)
    _hide_park_notice(page)
    _refuse_next_write(page)
    result = _fire_write_via_page(page)
    assert result and result.get("reason") == "fault_latched", result
    page.wait_for_timeout(200)
    _assert_docked_clear(page, "#guardNotice")
    text = page.query_selector("#guardNoticeText").text_content() or ""
    assert "fault_latched" in text, text

    _close(page, "#guardNotice")
    assert not page.query_selector("#guardNotice").is_visible()
    assert not _chip_visible(page, "guardNotice"), "an event notice left a chip"

    # a new refusal shows it again
    _fire_write_via_page(page)
    page.wait_for_timeout(200)
    assert page.query_selector("#guardNotice").is_visible(), "a new refusal did not show"
    _close(page, "#guardNotice")


# ------------------------------------------ safe_park / fault latch (the X accepts) --

def _emit_fault(page, fault) -> None:
    page.evaluate(
        "(data) => window.__wsInstances.forEach(ws => ws.dispatchEvent("
        "new MessageEvent('message', {data})))",
        json.dumps({"type": "fault", "fault": fault}))
    page.wait_for_timeout(300)


def _close_title(page, selector: str) -> str:
    return page.eval_on_selector(f"{selector} .banner-close", "b => b.title")


def test_the_x_on_a_critical_park_accepts_it_and_never_leaves_a_chip(page):
    _emit_critical_safe_park(page)
    _assert_docked_clear(page, "#ecSafetyBanner")
    assert _close_title(page, "#ecSafetyBanner") == "accept and clear the fault"

    # a park result while latched does not hide the latch
    _emit_safe_park(page, critical=False, verified=True, z_v=0.0)
    assert page.query_selector("#ecSafetyBanner").is_visible()

    _close(page, "#ecSafetyBanner")
    page.wait_for_timeout(300)
    assert not page.query_selector("#ecSafetyBanner").is_visible(), (
        "accepting the fault did not remove the banner")
    assert not _chip_visible(page, "ecSafetyBanner"), "the fault banner left a chip"

    # with nothing latched a park result is an event notice: close removes it
    _emit_safe_park(page, critical=False, verified=True, z_v=0.0)
    assert page.query_selector("#ecSafetyBanner").is_visible()
    assert _close_title(page, "#ecSafetyBanner") == "close"
    _close(page, "#ecSafetyBanner")
    assert not page.query_selector("#ecSafetyBanner").is_visible()
    assert not _chip_visible(page, "ecSafetyBanner"), "a closed event notice left a chip"


def test_a_fault_frame_shows_the_latch_until_a_clear_frame(page):
    rec = {"reason": "withdraw_failed", "at": 1_800_000_000.0, "state": "fault",
           "detail": "usertwin frame"}
    _emit_fault(page, rec)
    el = page.query_selector("#ecSafetyBanner")
    assert el.is_visible(), "a fault frame did not show the banner"
    assert "withdraw_failed" in (el.text_content() or "")
    _assert_docked_clear(page, "#ecSafetyBanner")
    _emit_fault(page, rec)
    assert el.is_visible()
    assert page.query_selector("#faultStatus").is_visible(), "the status box does not say FAULT"
    _emit_fault(page, None)
    assert not el.is_visible() and not _chip_visible(page, "ecSafetyBanner")
    assert not page.query_selector("#faultStatus").is_visible(), "FAULT outlived the latch"


def _opened(page, selector: str) -> bool:
    return page.eval_on_selector(selector, "e => e.classList.contains('open')")


def test_a_critical_park_alone_shows_fault_in_the_status_box(page):
    assert not page.query_selector("#faultStatus").is_visible()
    _emit_critical_safe_park(page)
    assert page.query_selector("#faultStatus").is_visible(), (
        "a critical park did not put FAULT in the status box")
    assert _opened(page, "#ecSafetyBanner"), "a critical park did not arrive opened"
    _close(page, "#ecSafetyBanner")
    page.wait_for_timeout(300)
    assert not page.query_selector("#ecSafetyBanner").is_visible()
    assert not page.query_selector("#faultStatus").is_visible(), "FAULT outlived the accept"


def _hits(page, selector: str) -> bool:
    b = _box(page, selector)
    return page.evaluate(
        "([x, y, id]) => { const e = document.elementFromPoint(x, y); "
        "return !!e && (e.id === id || e.closest('[id]')?.id === id); }",
        [b["x"] + b["width"] / 2, b["y"] + b["height"] / 2, selector.lstrip("#")])


def test_an_opened_long_fault_leaves_abort_stop_and_withdraw_reachable(page):
    page.set_viewport_size({"width": 948, "height": 700})
    try:
        _emit_fault(page, {"reason": "withdraw_failed", "at": 1_800_000_000.0,
                           "state": "fault", "detail": "x" * 2500})
        # a latched fault arrives opened, its whole message shown, with "less"
        assert _opened(page, "#ecSafetyBanner"), "a latched fault did not arrive opened"
        page.wait_for_timeout(200)
        less = page.query_selector("#ecSafetyBanner .banner-more")
        assert less.is_visible() and less.text_content() == "less", less.text_content()
        for sel in ("#abortBtn", "#stop", "#withdraw"):
            assert _hits(page, sel), f"{sel}'s centre is not reachable under an opened fault"
    finally:
        _emit_fault(page, None)


_LONG_FAULT = {"reason": "withdraw_failed", "at": 1_800_000_000.0,
               "state": "fault", "detail": "x" * 2500}


def _open_scan_fold(page, is_open: bool) -> None:
    page.evaluate("o => { const d = document.getElementById('scanMore'); if (d) d.open = o; }",
                  is_open)


@pytest.mark.parametrize("size,fault", [
    ((1896, 700), False), ((1896, 700), True), ((948, 700), False), ((948, 700), True),
    ((948, 560), True),
], ids=["1896x700", "1896x700-fault", "948x700", "948x700-fault", "948x560-fault"])
def test_withdraw_and_stop_stay_in_view_after_every_scan_column_scrolls_to_its_end(
        page, size, fault):
    """Hard rule 3: Withdraw (and Stop) are never hidden on the Scan tab, with the
    folded settings open and every scrollable box in the tab pane at its end."""
    page.set_viewport_size({"width": size[0], "height": size[1]})
    page.click('button.tab[data-tab="scan"]')
    if fault:
        _emit_fault(page, _LONG_FAULT)
        assert _opened(page, "#ecSafetyBanner")
    try:
        _open_scan_fold(page, True)
        page.wait_for_timeout(200)
        page.evaluate(
            """() => { for (const e of document.querySelectorAll(
                   '.rightpanel, .rightbody, #tabScan, .scancols, .scanbtns, .btncol,'
                   + ' .scanfields, .fieldlist'))
                 { e.scrollTop = e.scrollHeight; e.scrollLeft = e.scrollWidth; } }""")
        page.wait_for_timeout(100)
        for sel in ("#withdraw", "#stop", "#abortBtn"):
            assert _hits(page, sel), f"{size} fault={fault}: {sel}'s centre is not {sel}"
    finally:
        if fault:
            _emit_fault(page, None)
        _open_scan_fold(page, False)
        page.set_viewport_size({"width": 1600, "height": 1000})


@pytest.mark.parametrize("tab", ["scan", "edit", "write"])
def test_half_screen_keeps_abort_withdraw_and_stop_reachable(page, tab):
    """At half of a 1896 x 1036 screen the half-screen arrangement is in force, the
    panes split the width equally, the document does not scroll either way, Abort is
    hittable on every tab and Withdraw and Stop on Scan."""
    page.set_viewport_size({"width": 948, "height": 1036})
    try:
        page.click(f'button.tab[data-tab="{tab}"]')
        page.wait_for_timeout(300)
        facts = page.evaluate(
            """() => {
                 const w = s => document.querySelector(s).getBoundingClientRect().width;
                 const se = document.scrollingElement;
                 return {arr: getComputedStyle(document.getElementById('renderBody'))
                           .getPropertyValue('--arr').trim(),
                         render: w('.scanpanel'), right: w('.rightpanel'),
                         sx: se.scrollWidth - se.clientWidth,
                         sy: se.scrollHeight - se.clientHeight};
               }""")
        assert facts["arr"] == "half", facts
        assert abs(facts["render"] - facts["right"]) <= 2, f"panes not split equally: {facts}"
        assert facts["sx"] <= 0 and facts["sy"] <= 0, f"the document scrolls: {facts}"
        for sel in ("#abortBtn",) + (("#withdraw", "#stop") if tab == "scan" else ()):
            assert _hits(page, sel), f"half screen, {tab}: {sel}'s centre is not {sel}"
    finally:
        page.click('button.tab[data-tab="scan"]')
        page.set_viewport_size({"width": 1600, "height": 1000})


@pytest.mark.parametrize("size", [(948, 480), (1896, 400)], ids=lambda s: f"{s[0]}x{s[1]}")
def test_the_document_never_scrolls_abort_off_screen_in_a_short_window(page, size):
    """With an opened fault in a short window, neither script nor keyboard focus can
    scroll the document, so Abort, Withdraw and Stop all stay hittable."""
    page.set_viewport_size({"width": size[0], "height": size[1]})
    page.click('button.tab[data-tab="scan"]')
    _emit_fault(page, _LONG_FAULT)
    assert _opened(page, "#ecSafetyBanner")
    try:
        page.wait_for_timeout(200)
        top = page.evaluate(
            """() => {
                 const se = document.scrollingElement;
                 const sel = 'button, input, select, textarea, summary, [tabindex]';
                 const all = [...document.querySelectorAll(sel)]
                   .filter(e => e.offsetParent !== null && !e.disabled);
                 for (const e of all) e.focus();
                 all[all.length - 1].focus();
                 se.scrollTop = 1e6;
                 return se.scrollTop;
               }""")
        assert top == 0, f"{size}: the document scrolled to {top}"
        page.keyboard.press("End")
        page.keyboard.press("PageDown")
        assert page.evaluate("() => document.scrollingElement.scrollTop") == 0
        for sel in ("#abortBtn", "#withdraw", "#stop"):
            assert _hits(page, sel), f"{size}: {sel}'s centre is not {sel}"
    finally:
        page.evaluate("() => document.activeElement && document.activeElement.blur()")
        _emit_fault(page, None)
        page.set_viewport_size({"width": 1600, "height": 1000})


def test_a_failed_clear_keeps_the_fault_banner_and_says_why(page):
    def refuse(route):
        route.fulfill(status=500, content_type="application/json",
                      body=json.dumps({"error": "usertwin: clear refused"}))
    page.route("**/api/fault/clear", refuse)
    _emit_fault(page, {"reason": "withdraw_failed", "at": 1_800_000_000.0,
                       "state": "fault", "detail": "usertwin frame"})
    _close(page, "#ecSafetyBanner")
    page.wait_for_timeout(300)
    el = page.query_selector("#ecSafetyBanner")
    assert el.is_visible(), "a failed clear hid the fault banner"
    text = el.text_content() or ""
    assert "clear failed" in text and "usertwin: clear refused" in text, text
    assert not _chip_visible(page, "ecSafetyBanner")


def test_stacked_banners_do_not_overlap_each_other(service, page):
    _to_write_tab(page)
    try:
        _approach(service)
        _refuse_next_write(page)
        _fire_write_via_page(page)
        _emit_critical_safe_park(page)
        boxes = {s: _box(page, s) for s in
                 ("#guardNotice", "#ecSafetyBanner")}
        for s in boxes:
            _assert_docked_clear(page, s)
        names = list(boxes)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                assert not _overlaps(boxes[a], boxes[b]), f"{a} overlaps {b}"
        # the safety banner docks first in the row, ahead of every other banner
        s, g = boxes["#ecSafetyBanner"], boxes["#guardNotice"]
        assert (round(s["y"]), s["x"]) < (round(g["y"]), g["x"]), (s, g)
    finally:
        _ensure_withdrawn(service)


# ------------------------------------------- a real latch on the service side --

# No HTTP door latches a fault on the twin, so this service latches one at start-up,
# the same record shape as the litho end-of-run withdraw_failed latch.
_LATCHED_LAUNCH = """
import time, uvicorn
from afm_litho.service.app import app
from afm_litho.service.hub import hub
hub.fault = {"reason": "withdraw_failed", "at": time.time(), "state": "fault",
             "detail": "usertwin: latched before any page opened"}
uvicorn.run(app, host="127.0.0.1", port={port}, log_level="warning")
"""


@pytest.fixture(scope="module")
def latched_service():
    svc = Service(flex=True)
    svc.launch_code = _LATCHED_LAUNCH
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


def _open(browser, svc):
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000}, permissions=[])
    ctx.add_init_script(_WS_CAPTURE_INIT)
    p = ctx.new_page()
    p.goto(svc.url + "/edit-cockpit", wait_until="load", timeout=45000)
    p.wait_for_timeout(800)
    return ctx, p


_BOTH_HIDDEN = (
    "() => document.getElementById('ecSafetyBanner').style.display === 'none'"
    " && document.querySelector('.banner-chip[data-banner=\"ecSafetyBanner\"]')"
    ".style.display === 'none'")


def _approach_reason(svc) -> str | None:
    status, r = svc.post("/api/approach", {"mode": "contact"})
    assert status == 200, (status, r)
    return (r or {}).get("reason")


def test_a_latched_fault_shows_on_load_everywhere_and_the_x_clears_it(
        latched_service, browser):
    status, st = latched_service.get("/api/state")
    assert status == 200 and st["fault"] and st["fault"]["reason"] == "withdraw_failed", st
    assert _approach_reason(latched_service) == "fault_latched", (
        "precondition: the latch does not refuse an approach")

    ctx1, p1 = _open(browser, latched_service)
    ctx2 = None
    try:
        el = p1.query_selector("#ecSafetyBanner")
        assert el.is_visible(), "a fault latched before the page opened is invisible"
        assert "withdraw_failed" in (el.text_content() or "")
        _assert_docked_clear(p1, "#ecSafetyBanner")
        assert _close_title(p1, "#ecSafetyBanner") == "accept and clear the fault"

        ctx2, p2 = _open(browser, latched_service)
        assert p2.query_selector("#ecSafetyBanner").is_visible(), (
            "a second page does not show the latched fault")

        _close(p1, "#ecSafetyBanner")
        for p in (p1, p2):
            p.wait_for_function(_BOTH_HIDDEN, timeout=5000)

        status, st = latched_service.get("/api/state")
        assert st["fault"] is None, f"the X did not clear the latch: {st['fault']}"
        assert _approach_reason(latched_service) != "fault_latched", (
            "approach is still refused fault_latched after the operator accepted it")
    finally:
        _ensure_withdrawn(latched_service)
        ctx1.close()
        if ctx2 is not None:
            ctx2.close()


# ------------------------------------------ replies that are older than a frame --

_NEW_LATCH = {"reason": "park_unverified", "at": 1_800_000_200.0, "state": "fault",
              "detail": "usertwin: latched while the clear was in flight"}


def test_a_latch_that_arrives_while_the_clear_is_in_flight_stays_shown(page):
    """Server clears -> `fault: null` -> a NEW latch -> `fault: rec`, and only then
    the clear's HTTP reply lands: the reply must not wipe the new latch."""
    _emit_fault(page, {"reason": "withdraw_failed", "at": 1_800_000_000.0,
                       "state": "fault", "detail": "usertwin frame"})
    page.evaluate(
        """(newLatch) => {
             const real = window.fetch;
             window.fetch = async (url, init) => {
               if (String(url).includes('/api/fault/clear')) {
                 const send = (fault) => window.__wsInstances.forEach(ws =>
                   ws.dispatchEvent(new MessageEvent('message',
                     {data: JSON.stringify({type: 'fault', fault})})));
                 send(null);
                 send(newLatch);
                 return new Response(JSON.stringify({ok: true, cleared: true}),
                   {status: 200, headers: {'Content-Type': 'application/json'}});
               }
               return real(url, init);
             };
           }""", _NEW_LATCH)
    _close(page, "#ecSafetyBanner")
    page.wait_for_timeout(300)
    el = page.query_selector("#ecSafetyBanner")
    assert el.is_visible(), "the clear's reply hid a fault latched after it"
    assert "park_unverified" in (el.text_content() or "")


def test_a_state_read_older_than_a_fault_frame_does_not_hide_the_latch(page):
    """A reconnect's GET /api/state read before a latch, answered after that latch's
    frame, must not hide it."""
    page.evaluate(
        """() => {
             const real = window.fetch;
             window.fetch = async (url, init) => {
               if (String(url).includes('/api/state')) {
                 await new Promise(r => setTimeout(r, 600));
                 return new Response(JSON.stringify({fault: null}),
                   {status: 200, headers: {'Content-Type': 'application/json'}});
               }
               return real(url, init);
             };
             // re-run only ec-safety's (re)connect seed
             window.__wsInstances.forEach(ws => {
               if (ws.onopen && String(ws.onopen).includes('seedFault')) ws.onopen();
             });
           }""")
    _emit_fault(page, _NEW_LATCH)
    page.wait_for_timeout(900)
    el = page.query_selector("#ecSafetyBanner")
    assert el.is_visible(), "a stale state read hid a latched fault"
    assert "park_unverified" in (el.text_content() or "")
