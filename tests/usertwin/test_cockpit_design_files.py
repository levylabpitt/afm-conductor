"""The cockpit's Save design / Open design dialog, in a real browser against a twin
service that runs with the design autosave on and a private runtime.

Each test builds its own design over HTTP, then drives the dialog by element id:
Save; Open (add) with a clashing name and the rename in the notice; Open (replace)
and reopening `_previous`; Save and Open while only the placeholder objects show;
the keyboard and the reach of Withdraw and Abort while the dialog is open; refusals.
"""

# ruff: noqa: F811 -- fixtures imported by name shadow their definitions when taken
# as test parameters; the standard pytest cross-file fixture-reuse pattern.

from __future__ import annotations

import json
import os
import shutil
import time

import pytest
from _service import Service
from test_instrument_cockpit import (
    _ensure_withdrawn,
    _object_names,
    browser,  # noqa: F401 -- fixture re-export
)

PLACEHOLDERS = ["wire-1", "rect-1", "funnel-1", "wire-2", "rect-2"]


@pytest.fixture(scope="module")
def service():
    svc = Service(flex=True)
    prev = os.environ.get("AFM_LITHO_DESIGN_AUTOSAVE")
    os.environ["AFM_LITHO_DESIGN_AUTOSAVE"] = "1"      # the child copies os.environ
    try:
        svc.start()
    except Exception as exc:      # noqa: BLE001 -- pragma: no cover - environment
        shutil.rmtree(svc.runtime, ignore_errors=True)
        pytest.skip(f"service would not start: {exc}")
    finally:
        if prev is None:
            os.environ.pop("AFM_LITHO_DESIGN_AUTOSAVE", None)
        else:
            os.environ["AFM_LITHO_DESIGN_AUTOSAVE"] = prev
    try:
        yield svc
    finally:
        svc.stop()
        shutil.rmtree(svc.runtime, ignore_errors=True)


@pytest.fixture
def page(service, browser):
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000}, permissions=[])
    p = ctx.new_page()
    yield p
    ctx.close()


# ------------------------------------------------------------------ helpers --

def _names(service) -> list[str]:
    status, d = service.get("/api/design")
    assert status == 200, (status, d)
    return [o["name"] for o in d["objects"]]


def _set_design(service, names: list[str]) -> None:
    """Make the service's design exactly one short path per name."""
    n = len(_names(service))
    if n:
        status, r = service.post("/api/design/op",
                                 {"op": "delete", "indices": list(range(n))})
        assert status == 200 and not (r or {}).get("error"), (status, r)
    for i, name in enumerate(names):
        status, r = service.post("/api/design/add", {
            "kind": "path", "name": name,
            "points_um": [[1.0, 1.0 + i], [4.0, 2.0 + i]]})
        assert status == 200 and r and r.get("ok"), (status, r)
    assert _names(service) == names


def _saved(service) -> dict[str, dict]:
    status, r = service.get("/api/design/files")
    assert status == 200 and r and isinstance(r.get("designs"), list), (status, r)
    return {d["name"]: d for d in r["designs"]}


def _save_over_http(service, name: str) -> None:
    status, r = service.post("/api/design/save", {"name": name})
    assert status == 200 and r and r.get("ok"), (status, r)


def _load(p, service) -> None:
    p.goto(service.url + "/edit-cockpit", wait_until="load", timeout=45000)
    p.wait_for_timeout(800)
    p.click('button.tab[data-tab="write"]', timeout=5000)
    p.wait_for_timeout(200)


def _wait(p, pred, what: str, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if pred():
            return
        p.wait_for_timeout(150)
    raise AssertionError(f"timed out waiting for {what}")


def _shown(p, selector: str) -> bool:
    return p.eval_on_selector(selector, "e => getComputedStyle(e).display") != "none"


def _text(p, selector: str) -> str:
    return p.eval_on_selector(selector, "e => e.textContent")


def _notice_lines(p) -> list[str]:
    return p.eval_on_selector_all("#designNoticeText > div",
                                  "els => els.map(e => e.textContent)")


def _listed(p) -> list[str]:
    return p.eval_on_selector_all("#designFileList .designfile",
                                  "els => els.map(e => e.dataset.name)")


def _press(p, selector: str, keep_tab: bool = False) -> None:
    """Press Open design or Save design, which live on the File tab. `keep_tab`
    clicks it by script instead, so the current tab stays selected (leaving Edit
    abandons a pen draft)."""
    if keep_tab:
        p.evaluate("id => document.getElementById(id).click()", selector.lstrip("#"))
        return
    p.click('button.tab[data-tab="file"]', timeout=5000)
    p.click(selector)


def _save_through_dialog(p, name: str | None) -> None:
    _press(p, "#saveDesign")
    assert _shown(p, "#designDialog") and _shown(p, "#designSaveSection")
    assert not _shown(p, "#designOpenSection")
    if name is not None:
        p.fill("#designSaveName", name)
    p.click("#designSaveConfirm")


def _open_through_dialog(p, name: str, mode: str) -> None:
    _press(p, "#openDesign")
    assert _shown(p, "#designDialog") and _shown(p, "#designOpenSection")
    _wait(p, lambda: name in _listed(p), f"{name} in the list of saved designs")
    assert p.is_checked("#designOpenModeAdd"), "add must be the default way to open"
    p.click(f'#designFileList .designfile[data-name="{name}"]')
    if mode == "replace":
        p.check("#designOpenModeReplace")
    p.click("#designOpenConfirm")
    _wait(p, lambda: not _shown(p, "#designDialog"), "the dialog to close after the open")


# -------------------------------------------------------------------- tests --

def test_the_child_service_autosaves_into_its_private_runtime(service):
    _set_design(service, ["probe"])
    autosave = service.runtime / "designs" / "_autosave.design.json"
    deadline = time.time() + 10
    while time.time() < deadline and not autosave.is_file():
        time.sleep(0.2)
    assert autosave.is_file(), "the child service is not running with the autosave on"
    assert "_autosave" not in _saved(service)


def test_save_writes_the_design_and_remembers_the_name(service, page):
    _set_design(service, ["wire-a", "pad-b"])
    _load(page, service)
    assert _object_names(page) == ["wire-a", "pad-b"]

    _save_through_dialog(page, "dev one")
    _wait(page, lambda: "dev_one" in _saved(service), "the saved design in the listing")
    assert _saved(service)["dev_one"]["object_count"] == 2
    assert (service.runtime / "designs" / "dev_one.design.json").is_file()
    _wait(page, lambda: not _shown(page, "#designDialog"), "the dialog to close")
    assert _notice_lines(page) == ["design saved as dev_one (2 objects)"]

    # the name is kept across a reload, and the dialog says the file will be replaced
    _load(page, service)
    _press(page, "#saveDesign")
    assert page.input_value("#designSaveName") == "dev_one"
    _wait(page, lambda: _shown(page, "#designSaveExists"), "the already-saved hint")
    assert "dev_one" in _text(page, "#designSaveExists")
    assert "replaces" in _text(page, "#designSaveExists")
    page.fill("#designSaveName", "something-new")
    assert not _shown(page, "#designSaveExists")
    page.fill("#designSaveName", "dev_one")
    assert _shown(page, "#designSaveExists")
    assert page.is_enabled("#designSaveConfirm"), "an existing name must not block Save"
    page.click("#designSaveConfirm")
    _wait(page, lambda: not _shown(page, "#designDialog"), "the dialog to close")
    assert _notice_lines(page) == [
        "design saved as dev_one (2 objects), replacing the earlier file"]
    assert not _shown(page, "#guardNotice")


def test_open_add_renames_a_clashing_name_and_the_notice_shows_it(service, page):
    _set_design(service, ["wire-a", "pad-b"])
    _save_over_http(service, "clash")
    _set_design(service, ["wire-a"])
    _load(page, service)
    assert _object_names(page) == ["wire-a"]

    _open_through_dialog(page, "clash", "add")
    _wait(page, lambda: len(_object_names(page)) == 3, "the opened objects in the list")
    names = _names(service)
    assert names[0] == "wire-a" and names[2] == "pad-b" and names[1] != "wire-a", names
    assert _object_names(page) == names

    assert _shown(page, "#designNotice")
    lines = _notice_lines(page)
    assert lines[0] == "clash opened: 2 objects added (3 in the design now)", lines
    assert lines[1:] == [f"renamed: wire-a → {names[1]}"], (
        f"only the name that changed belongs in the notice: {lines}")
    assert not _shown(page, "#guardNotice")


def test_open_replace_keeps_the_earlier_design_as_previous(service, page):
    _set_design(service, ["wire-a", "pad-b"])
    _save_over_http(service, "kept")
    _set_design(service, ["scratch"])
    _load(page, service)

    _open_through_dialog(page, "kept", "replace")
    _wait(page, lambda: _object_names(page) == ["wire-a", "pad-b"],
          "the saved design in place of the earlier one")
    assert _names(service) == ["wire-a", "pad-b"]
    lines = _notice_lines(page)
    assert lines[0] == "kept opened: 2 objects in place of the earlier design", lines
    assert lines[1:] == ["the earlier design is kept as _previous"], lines
    assert "_previous" in _saved(service)

    _press(page, "#openDesign")
    _wait(page, lambda: "_previous" in _listed(page), "_previous in the list")
    assert "_autosave" not in _listed(page)
    page.click("#designDialogClose")
    assert not _shown(page, "#designDialog")

    _open_through_dialog(page, "_previous", "replace")
    _wait(page, lambda: _object_names(page) == ["scratch"], "the earlier design back")
    assert _names(service) == ["scratch"]


def test_saving_the_placeholders_says_they_are_not_a_design(service, page):
    _set_design(service, [])
    _load(page, service)
    assert _object_names(page) == PLACEHOLDERS

    before = set(_saved(service))
    _save_through_dialog(page, "placeholders")
    _wait(page, lambda: _shown(page, "#guardNotice"), "the refusal notice")
    text = _text(page, "#guardNoticeText")
    assert "no_design" in text and "placeholders, not a design" in text, text
    assert "placeholders, not a design" in _text(page, "#designDialogMsg")
    assert set(_saved(service)) == before
    assert _names(service) == []


def test_opening_over_the_placeholders_shows_the_opened_design(service, page):
    _set_design(service, ["wire-a", "pad-b"])
    _save_over_http(service, "first")
    _set_design(service, [])
    _load(page, service)
    assert _object_names(page) == PLACEHOLDERS

    _open_through_dialog(page, "first", "add")
    _wait(page, lambda: _object_names(page) == ["wire-a", "pad-b"],
          "the opened design in place of the placeholders")
    assert _names(service) == ["wire-a", "pad-b"]
    assert _notice_lines(page) == ["first opened: 2 objects added (2 in the design now)"]

    # the page now holds a real design: saving it goes through
    _save_through_dialog(page, "first-copy")
    _wait(page, lambda: "first-copy" in _saved(service), "the save after the hand-over")
    assert not _shown(page, "#guardNotice")


def test_a_refused_open_is_shown_with_the_service_reason(service, page):
    _set_design(service, ["wire-a"])
    _save_over_http(service, "gone")
    _load(page, service)
    _press(page, "#openDesign")
    _wait(page, lambda: "gone" in _listed(page), "the saved design in the list")
    (service.runtime / "designs" / "gone.design.json").unlink()
    page.click('#designFileList .designfile[data-name="gone"]')
    page.click("#designOpenConfirm")
    _wait(page, lambda: _shown(page, "#guardNotice"), "the refusal notice")
    text = _text(page, "#guardNoticeText")
    assert "not_found" in text and "gone" in text, text
    assert _shown(page, "#designDialog"), "a refusal leaves the dialog open"
    assert _names(service) == ["wire-a"]


# ------------------------------------------------- keyboard, reach, guards --

def _posts(p, path: str) -> list[str]:
    seen: list[str] = []
    p.on("request", lambda r: seen.append(r.url)
         if r.method == "POST" and r.url.endswith(path) else None)
    return seen


def _hold(p, pattern: str) -> list:
    """Catch requests matching `pattern` without answering them; the test answers."""
    held: list = []
    p.route(pattern, lambda route: held.append(route))
    return held


def _refusal(p, timeout_s: float = 10.0) -> str:
    _wait(p, lambda: _shown(p, "#guardNotice"), "the refusal notice", timeout_s)
    return _text(p, "#guardNoticeText")


def _log_rows(p) -> list[str]:
    return p.eval_on_selector_all("#log .row", "els => els.map(e => e.textContent)")


def _ready_to_write(service) -> None:
    """An Abort can leave the fault latched; later tests in this module write."""
    service.post("/api/fault/clear", {"confirm": True})
    _ensure_withdrawn(service)


def test_escape_closes_the_dialog_wherever_focus_is(service, page):
    _set_design(service, ["wire-a"])
    _load(page, service)

    _press(page, "#openDesign")
    assert _shown(page, "#designDialog")
    page.keyboard.press("Escape")
    assert not _shown(page, "#designDialog"), "Escape must close the dialog in Open mode"

    _press(page, "#openDesign")
    page.evaluate("document.activeElement && document.activeElement.blur()")
    page.keyboard.press("Escape")
    assert not _shown(page, "#designDialog"), "Escape must close it with focus on the body"

    _press(page, "#saveDesign")
    for _ in range(6):
        page.keyboard.press("Tab")
    assert page.evaluate("!document.getElementById('designDialog')"
                         ".contains(document.activeElement)"), "focus should have left"
    page.keyboard.press("Escape")
    assert not _shown(page, "#designDialog"), "Escape must close it after tabbing out"


def test_escape_under_the_dialog_does_not_commit_a_pen_draft(service, page):
    """Guards the code path only: Open design lives on the File tab and leaving Edit
    abandons a pen draft, so a user can no longer reach this state; the dialog is
    opened by script to keep the draft."""
    _set_design(service, ["wire-a"])
    _load(page, service)
    page.click('button.tab[data-tab="edit"]')
    page.click('button.tool[data-tool="pen"]')
    box = page.query_selector("#overlay").bounding_box()
    # low in the image, clear of where the dialog floats
    x, y = box["x"] + 80, box["y"] + box["height"] - 60
    for dx in (0, 70, 140):
        page.mouse.click(x + dx, y - dx / 4)
        page.wait_for_timeout(100)

    _press(page, "#openDesign", keep_tab=True)
    page.keyboard.press("Escape")
    assert not _shown(page, "#designDialog")
    page.wait_for_timeout(800)
    assert _names(service) == ["wire-a"], "Escape for the dialog committed the pen draft"

    # control: the draft was real, and Escape commits it once the dialog is closed
    page.keyboard.press("Escape")
    _wait(page, lambda: len(_names(service)) == 2, "the pen draft to be committed")
    page.click('button.tool[data-tool="select"]')


def test_shortcuts_do_not_reach_the_design_while_the_dialog_is_open(service, page):
    _set_design(service, ["wire-a"])
    _save_over_http(service, "for-paste")
    _load(page, service)
    page.click('#objList .objrow[data-i="0"]')
    page.keyboard.press("Control+c")

    _press(page, "#openDesign")
    page.click('button.tab[data-tab="edit"]')        # the dialog stays open across tabs
    assert _shown(page, "#designDialog")
    _wait(page, lambda: "for-paste" in _listed(page), "the list of saved designs")
    page.keyboard.press("Control+v")
    page.click('#designFileList .designfile[data-name="for-paste"]')
    page.keyboard.press("Control+v")
    page.wait_for_timeout(800)
    assert _names(service) == ["wire-a"], "a paste edited the design under the dialog"

    # control: the clipboard held the object, and paste works once the dialog is closed
    page.keyboard.press("Escape")
    assert not _shown(page, "#designDialog")
    page.keyboard.press("Control+v")
    _wait(page, lambda: len(_names(service)) == 2, "the paste after the dialog closed")


TIP_CONTROLS = ["#withdraw", "#abortBtn", "#stop", "#approach", "#up", "#down"]
SIZES = [(1920, 1080), (1366, 768), (1024, 700), (800, 500), (640, 400), (608, 700),
         (480, 700)]


def _box(p, selector: str) -> dict | None:
    el = p.query_selector(selector)
    return el.bounding_box() if el is not None and el.is_visible() else None


def _overlap(a: dict, b: dict) -> bool:
    return (a["x"] < b["x"] + b["width"] and b["x"] < a["x"] + a["width"]
            and a["y"] < b["y"] + b["height"] and b["y"] < a["y"] + a["height"])


def _assert_panel_is_clear(p, where: str) -> None:
    """The panel is inside the scan panel and the window, and over no tip control."""
    dlg = _box(p, "#designDialog")
    assert dlg is not None, f"{where}: the dialog is not shown"
    scan = _box(p, ".scanpanel")
    vw, vh = p.viewport_size["width"], p.viewport_size["height"]
    tol = 0.6
    assert dlg["x"] >= max(scan["x"], 0) - tol and dlg["y"] >= max(scan["y"], 0) - tol, (
        f"{where}: the dialog starts outside the scan panel: {dlg} / {scan}")
    assert dlg["x"] + dlg["width"] <= min(scan["x"] + scan["width"], vw) + tol, (
        f"{where}: the dialog runs past the scan panel or the window: {dlg} / {scan}")
    assert dlg["y"] + dlg["height"] <= min(scan["y"] + scan["height"], vh) + tol, (
        f"{where}: the dialog runs below the scan panel or the window: {dlg} / {scan}")
    for sel in TIP_CONTROLS:
        b = _box(p, sel)
        if b is not None:
            assert not _overlap(dlg, b), f"{where}: the dialog lies over {sel}: {dlg} / {b}"


def _open_dialog(p, opener: str) -> None:
    # by script: at the narrowest sizes the header buttons are not all on screen
    p.evaluate("id => document.getElementById(id).click()", opener.lstrip("#"))
    assert _shown(p, "#designDialog")
    p.wait_for_timeout(300)


@pytest.mark.parametrize("opener", ["#saveDesign", "#openDesign"])
@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_the_dialog_never_covers_a_tip_control(service, page, opener, size):
    _set_design(service, ["wire-a"])
    _save_over_http(service, "reach")
    page.set_viewport_size({"width": size[0], "height": size[1]})
    _load(page, service)
    for tab in ("scan", "write", "edit", "scan"):
        page.click(f'button.tab[data-tab="{tab}"]')
        if not _shown(page, "#designDialog"):
            _open_dialog(page, opener)
        page.wait_for_timeout(150)
        _assert_panel_is_clear(page, f"{size} {opener} tab {tab}")
    # the panel's own buttons first, before an Abort adds its notice to the message
    # row (below about 520 px of window width the scan panel is a sliver, and so is the
    # panel: nothing in it can be clicked there, and Escape is what closes it)
    dlg = _box(page, "#designDialog")
    if dlg["width"] >= 100 and dlg["height"] >= 100:
        confirm = "#designSaveConfirm" if opener == "#saveDesign" else "#designOpenConfirm"
        for own in ("#designDialogClose", confirm):
            page.click(own, trial=True, timeout=3000)
    else:
        assert size == (480, 700), f"the panel is too small to use at {size}: {dlg}"
        page.keyboard.press("Escape")
        assert not _shown(page, "#designDialog")
        _open_dialog(page, opener)
    try:
        with page.expect_request(lambda r: r.url.endswith("/api/withdraw"), timeout=5000):
            page.click("#withdraw", timeout=3000)
        with page.expect_request(lambda r: r.url.endswith("/api/safe_park"), timeout=5000):
            page.click("#abortBtn", timeout=3000)
        assert _shown(page, "#designDialog"), "neither control should close the dialog"
    finally:
        _ready_to_write(service)


@pytest.mark.parametrize("opener", ["#saveDesign", "#openDesign"])
def test_the_dialog_stays_clear_when_the_window_is_resized(service, page, opener):
    _set_design(service, ["wire-a"])
    page.set_viewport_size({"width": 1920, "height": 1080})
    _load(page, service)
    page.click('button.tab[data-tab="scan"]')
    _open_dialog(page, opener)
    for w, h in [(560, 700), (608, 700), (480, 400), (1920, 1080), (640, 400)]:
        page.set_viewport_size({"width": w, "height": h})
        page.wait_for_timeout(300)
        _assert_panel_is_clear(page, f"resized to {w}x{h} {opener}")
    page.click("#withdraw", trial=True, timeout=3000)
    page.click("#abortBtn", trial=True, timeout=3000)


def test_the_open_button_stays_in_view_with_a_long_list(service, page):
    _set_design(service, ["wire-a"])
    many = [{"name": f"design-{i:03d}", "saved_at": "2026-10-05T12:00:00", "object_count": 1}
            for i in range(300)]
    page.route("**/api/design/files", lambda route: route.fulfill(
        status=200, content_type="application/json",
        body=json.dumps({"ok": True, "designs": many})))
    page.set_viewport_size({"width": 800, "height": 500})
    _load(page, service)
    page.click('button.tab[data-tab="scan"]')
    _open_dialog(page, "#openDesign")
    _wait(page, lambda: len(_listed(page)) == 300, "the long list")
    _assert_panel_is_clear(page, "800x500 with 300 files")
    dlg = _box(page, "#designDialog")
    for own in ("#designOpenConfirm", "#designDialogClose"):
        b = _box(page, own)
        assert b is not None, f"{own} is not visible"
        inside = (b["y"] >= dlg["y"] - 0.6
                  and b["y"] + b["height"] <= dlg["y"] + dlg["height"] + 0.6)
        assert inside, f"{own} is scrolled out of the panel: {b} / {dlg}"
        page.click(own, trial=True, timeout=3000)


def test_a_double_click_opens_or_saves_once(service, page):
    _set_design(service, ["wire-a"])
    _save_over_http(service, "twice")
    _load(page, service)

    opens = _posts(page, "/api/design/open")
    held = _hold(page, "**/api/design/open")
    _press(page, "#openDesign")
    _wait(page, lambda: "twice" in _listed(page), "the list of saved designs")
    page.click('#designFileList .designfile[data-name="twice"]')
    page.dblclick("#designOpenConfirm")
    page.wait_for_timeout(400)
    assert len(opens) == 1 and len(held) == 1, opens
    said = _text(page, "#designDialogMsg")
    assert "an open of twice is still in progress" in said, (
        f"the dropped click was not explained: {said!r}")
    held[0].continue_()
    _wait(page, lambda: not _shown(page, "#designDialog"), "the dialog to close")
    page.wait_for_timeout(400)
    assert len(_names(service)) == 2, "the file was opened twice"
    page.unroute("**/api/design/open")

    saves = _posts(page, "/api/design/save")
    held = _hold(page, "**/api/design/save")
    _press(page, "#saveDesign")
    page.fill("#designSaveName", "twice-saved")
    page.click("#designSaveConfirm")
    page.wait_for_timeout(200)
    assert "saving twice-saved" in _text(page, "#designDialogMsg")
    page.click("#designSaveConfirm")
    page.wait_for_timeout(300)
    assert len(saves) == 1 and len(held) == 1, saves
    assert "a save as twice-saved is still in progress" in _text(page, "#designDialogMsg")
    held[0].continue_()
    _wait(page, lambda: not _shown(page, "#designDialog"), "the dialog to close")
    assert "twice-saved" in _saved(service)


def test_the_guard_is_released_by_the_reply_not_by_the_refetch(service, page):
    _set_design(service, ["wire-a"])
    _save_over_http(service, "quick")
    _load(page, service)
    held = _hold(page, "**/api/litho/toolpath")
    _open_through_dialog(page, "quick", "add")
    _wait(page, lambda: _shown(page, "#designNotice"), "the opened notice")
    assert _notice_lines(page)[0].startswith("quick opened"), _notice_lines(page)

    _save_through_dialog(page, "while-toolpath-held")
    _wait(page, lambda: "while-toolpath-held" in _saved(service),
          "a save while the toolpath request is unanswered")
    _wait(page, lambda: not _shown(page, "#designDialog"), "the dialog to close")
    assert held, "the toolpath request was never held; the test proves nothing"


def test_an_earlier_reply_leaves_a_reopened_dialog_alone(service, page):
    _set_design(service, ["wire-a"])
    _save_over_http(service, "slow")
    _load(page, service)
    held = _hold(page, "**/api/design/open")
    _press(page, "#openDesign")
    _wait(page, lambda: "slow" in _listed(page), "the list of saved designs")
    page.click('#designFileList .designfile[data-name="slow"]')
    page.click("#designOpenConfirm")
    _wait(page, lambda: len(held) == 1, "the held open")
    page.keyboard.press("Escape")

    _press(page, "#saveDesign")
    assert not _shown(page, "#designDialogMsg"), (
        f"the Save dialog shows the earlier call: {_text(page, '#designDialogMsg')!r}")
    page.fill("#designSaveName", "typed-meanwhile")
    page.click("#designSaveConfirm")
    _wait(page, lambda: _shown(page, "#designDialogMsg"), "the dropped click explained")
    assert "an open of slow is still in progress" in _text(page, "#designDialogMsg")

    held[0].continue_()
    _wait(page, lambda: _shown(page, "#designNotice"), "the opened notice")
    assert _notice_lines(page)[0].startswith("slow opened"), _notice_lines(page)
    assert _shown(page, "#designDialog"), "the earlier reply closed the Save dialog"
    assert _shown(page, "#designSaveSection")
    assert page.input_value("#designSaveName") == "typed-meanwhile"
    assert not _shown(page, "#designDialogMsg"), "the in-progress message was left up"
    page.unroute("**/api/design/open")

    page.click("#designSaveConfirm")
    _wait(page, lambda: "typed-meanwhile" in _saved(service), "the save after the open")


def test_only_ok_true_counts_as_success(service, page):
    _set_design(service, ["wire-a"])
    _save_over_http(service, "odd")
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    _load(page, service)

    page.route("**/api/design/save", lambda route: route.fulfill(
        status=200, content_type="text/html", body="<html>proxy says hello</html>"))
    _save_through_dialog(page, "kept-name")
    text = _refusal(page)
    assert "save design failed" in text and "not understood" in text, text
    assert "undefined" not in text
    assert _shown(page, "#designDialog")
    assert not _shown(page, "#designNotice"), "a failed save was announced as saved"
    page.keyboard.press("Escape")
    _press(page, "#saveDesign")
    assert page.input_value("#designSaveName") != "undefined"
    page.keyboard.press("Escape")
    page.unroute("**/api/design/save")

    page.route("**/api/design/open", lambda route: route.fulfill(
        status=200, content_type="application/json", body='{"weird": 1}'))
    _press(page, "#openDesign")
    _wait(page, lambda: "odd" in _listed(page), "the list of saved designs")
    page.click('#designFileList .designfile[data-name="odd"]')
    page.click("#designOpenConfirm")
    _wait(page, lambda: "open design failed" in _text(page, "#guardNoticeText"),
          "the failed open in the notice")
    text = _text(page, "#guardNoticeText")
    assert "not understood" in text and "weird" in text, text
    assert "opening" not in _text(page, "#designDialogMsg")
    assert errors == [], errors
    assert _names(service) == ["wire-a"]


def test_the_list_is_driven_from_the_keyboard(service, page):
    _set_design(service, ["wire-a"])
    _save_over_http(service, "kb-one")
    time.sleep(1.1)                             # saved_at has one-second resolution
    _save_over_http(service, "kb-two")
    _load(page, service)
    opens = _posts(page, "/api/design/open")
    _press(page, "#openDesign")
    _wait(page, lambda: {"kb-one", "kb-two"} <= set(_listed(page)), "the list")
    order = _listed(page)

    def selected() -> str:
        return page.eval_on_selector("#designFileList .designfile.sel", "e => e.dataset.name")

    assert selected() == order[0]
    page.keyboard.press("ArrowDown")
    assert selected() == order[1]
    page.keyboard.press("ArrowUp")
    page.keyboard.press("ArrowUp")
    assert selected() == order[0]
    page.keyboard.press("ArrowDown")
    page.keyboard.press("Enter")
    _wait(page, lambda: not _shown(page, "#designDialog"), "the dialog to close")
    page.wait_for_timeout(300)
    assert len(opens) == 1, opens
    assert _notice_lines(page)[0].startswith(f"{order[1]} opened"), _notice_lines(page)


def test_save_says_when_the_list_could_not_be_read(service, page):
    _set_design(service, ["wire-a"])
    _load(page, service)
    page.route("**/api/design/files", lambda route: route.fulfill(
        status=500, content_type="application/json", body='{"detail": "boom"}'))
    _press(page, "#saveDesign")
    _wait(page, lambda: _shown(page, "#designSaveExists"), "the hint about the list")
    text = _text(page, "#designSaveExists")
    assert "could not be listed" in text and "HTTP 500" in text, text
    assert "not known" in text, text


def test_the_list_says_loading_then_why_it_failed(service, page):
    _set_design(service, ["wire-a"])
    _load(page, service)
    opens = _posts(page, "/api/design/open")
    held = _hold(page, "**/api/design/files")

    _press(page, "#openDesign")
    _wait(page, lambda: len(held) == 1, "the list request")
    state = page.query_selector("#designListState")
    assert state.get_attribute("data-state") == "loading"
    assert "loading" in _text(page, "#designListState")
    assert "no saved designs" not in _text(page, "#designListState")

    held[0].fulfill(status=500, content_type="application/json", body='{"detail": "boom"}')
    _wait(page, lambda: state.get_attribute("data-state") == "failed", "the failed state")
    text = _text(page, "#designListState")
    assert "could not be listed" in text and "HTTP 500" in text, text
    assert "no saved designs" not in text

    page.click("#designOpenConfirm")
    _wait(page, lambda: _shown(page, "#designDialogMsg"), "the nothing-selected message")
    assert "no saved design is selected" in _text(page, "#designDialogMsg")
    page.wait_for_timeout(300)
    assert opens == [], "Open posted with nothing selected"


# The service's own test for a write in flight (flex_dispatch._write_in_flight) is a
# write run record or a live `hub.litho_thread`. This service is started with such a
# thread, so the refusal is the real one without depending on a twin write's timing.
_WRITING_LAUNCH = """
import threading, time, uvicorn
from afm_litho.service.app import app
from afm_litho.service.hub import hub
hub.host.litho_thread = threading.Thread(target=time.sleep, args=(3600,), daemon=True)
hub.host.litho_thread.start()
uvicorn.run(app, host="127.0.0.1", port={port}, log_level="warning")
"""


@pytest.fixture
def writing_service():
    svc = Service(flex=True)
    svc.launch_code = _WRITING_LAUNCH
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


def test_open_during_a_write_is_refused_with_write_in_flight(writing_service, browser):
    service = writing_service
    _set_design(service, ["wire-a"])
    _save_over_http(service, "later")              # a save is allowed during a write
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000}, permissions=[])
    try:
        page = ctx.new_page()
        _load(page, service)
        _press(page, "#openDesign")
        _wait(page, lambda: "later" in _listed(page), "the list of saved designs")
        page.click('#designFileList .designfile[data-name="later"]')
        page.click("#designOpenConfirm")
        text = _refusal(page)
        assert "write_in_flight" in text, text
        assert _shown(page, "#designDialog")
        assert _names(service) == ["wire-a"]
    finally:
        ctx.close()


def test_a_file_that_is_not_a_design_is_listed_plainly_and_refused_on_open(service, page):
    _set_design(service, ["wire-a"])
    (service.runtime / "designs").mkdir(exist_ok=True)
    stray = service.runtime / "designs" / "stray.design.json"
    stray.write_text('{"something": "else"}', "utf-8")
    try:
        assert _saved(service)["stray"] == {
            "name": "stray", "saved_at": None, "object_count": None}
        _load(page, service)
        _press(page, "#openDesign")
        _wait(page, lambda: "stray" in _listed(page), "the stray file in the list")
        row = _text(page, '#designFileList .designfile[data-name="stray"]')
        assert "null" not in row and "undefined" not in row, row
        assert "no details in the file" in row, row
        page.click('#designFileList .designfile[data-name="stray"]')
        page.click("#designOpenConfirm")
        text = _refusal(page)
        assert "not_a_design_file" in text and "stray" in text, text
        assert _names(service) == ["wire-a"]
    finally:
        stray.unlink(missing_ok=True)


def test_a_failed_save_is_shown_with_save_failed(service, page):
    _set_design(service, ["wire-a"])
    blocker = service.runtime / "designs" / "blocked.design.json"
    blocker.mkdir(parents=True)                 # a folder where the file would go
    try:
        _load(page, service)
        _save_through_dialog(page, "blocked")
        text = _refusal(page)
        assert "save_failed" in text and "blocked" in text, text
        assert _shown(page, "#designDialog")
        assert _names(service) == ["wire-a"]
    finally:
        shutil.rmtree(blocker, ignore_errors=True)


def test_a_voltage_refusal_on_open_is_shown_once(service, page):
    _set_design(service, ["hot"])
    _save_over_http(service, "too-hot")
    path = service.runtime / "designs" / "too-hot.design.json"
    doc = json.loads(path.read_text("utf-8"))
    doc["objects"][0]["voltage"] = 500.0
    path.write_text(json.dumps(doc), "utf-8")
    _set_design(service, ["wire-a"])
    _load(page, service)

    _press(page, "#openDesign")
    _wait(page, lambda: "too-hot" in _listed(page), "the list of saved designs")
    page.click('#designFileList .designfile[data-name="too-hot"]')
    page.click("#designOpenConfirm")
    text = _refusal(page)
    assert "tip_voltage_out_of_window" in text, text
    page.wait_for_timeout(300)
    shown = [row for row in _log_rows(page) if "REFUSED tip_voltage_out_of_window" in row]
    assert len(shown) == 1, f"a guard reason was put in the notice {len(shown)} times"
    assert _names(service) == ["wire-a"]


def test_a_reserved_name_is_refused_and_a_later_save_clears_the_notice(service, page):
    _set_design(service, ["wire-a"])
    _load(page, service)
    _save_through_dialog(page, "_previous")
    text = _refusal(page)
    assert "reserved_name" in text, text
    assert _shown(page, "#designDialog")

    page.fill("#designSaveName", "after-refusal")
    page.click("#designSaveConfirm")
    _wait(page, lambda: not _shown(page, "#designDialog"), "the dialog to close")
    assert "after-refusal" in _saved(service)
    assert not _shown(page, "#guardNotice"), "the earlier refusal is still up"
    assert _shown(page, "#designNotice")


def test_no_answer_from_the_service_is_shown_in_a_notice(service, page):
    _set_design(service, ["wire-a"])
    _load(page, service)
    page.route("**/api/design/save", lambda route: route.abort())
    _save_through_dialog(page, "unreachable")
    text = _refusal(page)
    assert "save design failed" in text and "did not answer" in text, text
    assert "did not answer" in _text(page, "#designDialogMsg")
    assert _shown(page, "#designDialog")


# ------------------------------------------- keys aimed at another control --

CONTROL_REQUESTS = {"#withdraw": "/api/withdraw", "#abortBtn": "/api/safe_park",
                    "#stop": "/api/scan/abort", "#approach": "/api/approach",
                    "#up": "/api/scan/start", "#down": "/api/scan/start"}


def _settle_instrument(service) -> None:
    service.post("/api/scan/abort", {"withdraw": True})
    _ready_to_write(service)


@pytest.mark.parametrize("size", [(1366, 768), (1920, 1080)], ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("opener", ["#saveDesign", "#openDesign"])
def test_a_key_on_a_focused_tip_control_goes_to_that_control(service, page, opener, size):
    _set_design(service, ["wire-a"])
    _save_over_http(service, "bystander")
    page.set_viewport_size({"width": size[0], "height": size[1]})
    _load(page, service)
    page.click('button.tab[data-tab="scan"]')
    opens = _posts(page, "/api/design/open")
    saves = _posts(page, "/api/design/save")
    _press(page, opener)
    page.click('button.tab[data-tab="scan"]')       # the dialog stays open across tabs
    if opener == "#openDesign":
        _wait(page, lambda: "bystander" in _listed(page), "the list of saved designs")
        page.check("#designOpenModeReplace")

    def sent(action, selector: str, path: str, what: str) -> None:
        with page.expect_request(lambda r: r.url.endswith(path), timeout=5000):
            action()
        page.wait_for_timeout(120)
        assert opens == [] and saves == [], (
            f"{what} on {selector} reached the dialog: {opens} {saves}")
        assert _shown(page, "#designDialog"), f"{what} on {selector} closed the dialog"

    try:
        for selector, path in CONTROL_REQUESTS.items():
            for key in ("Enter", "Space", "NumpadEnter"):
                page.focus(selector)
                sent(lambda key=key: page.keyboard.press(key), selector, path, key)
                # focus left there by a real click, then the key
                sent(lambda selector=selector: page.click(selector, timeout=3000),
                     selector, path, "a click")
                assert page.evaluate("s => document.activeElement === document.querySelector(s)",
                                     selector), f"a click did not leave focus on {selector}"
                sent(lambda key=key: page.keyboard.press(key), selector, path,
                     f"{key} after a click")
            _settle_instrument(service)
    finally:
        _settle_instrument(service)
    assert _names(service) == ["wire-a"]


def test_arrow_keys_in_another_field_stay_in_that_field(service, page):
    _set_design(service, ["wire-a"])
    _save_over_http(service, "arrow-one")
    _save_over_http(service, "arrow-two")
    _load(page, service)
    # stay on Write, whose setpoint is a number field beside the open dialog
    _press(page, "#openDesign", keep_tab=True)
    _wait(page, lambda: {"arrow-one", "arrow-two"} <= set(_listed(page)), "the list")
    selected = page.eval_on_selector("#designFileList .designfile.sel", "e => e.dataset.name")
    page.fill("#lithoSetpoint", "10")
    page.focus("#lithoSetpoint")
    page.keyboard.press("ArrowUp")
    assert float(page.input_value("#lithoSetpoint")) > 10, "ArrowUp did not step the field"
    page.keyboard.press("ArrowDown")
    page.keyboard.press("ArrowDown")
    assert float(page.input_value("#lithoSetpoint")) < 10, "ArrowDown did not step the field"
    assert page.eval_on_selector("#designFileList .designfile.sel",
                                 "e => e.dataset.name") == selected


# ----------------------------------- Enter needs focus inside the dialog --

def _focus_is_in_dialog(p) -> bool:
    return p.evaluate(
        "document.getElementById('designDialog').contains(document.activeElement)")


def test_focus_moves_into_the_dialog_when_it_opens(service, page):
    _set_design(service, ["wire-a"])
    _save_over_http(service, "focus")
    _load(page, service)
    held = _hold(page, "**/api/design/files")
    _press(page, "#openDesign")
    _wait(page, lambda: len(held) == 1, "the list request")
    assert page.evaluate("document.activeElement.id") == "designDialog"
    held[0].continue_()
    _wait(page, lambda: "focus" in _listed(page), "the list of saved designs")
    assert page.evaluate("document.activeElement.id") == "designFileList"
    page.keyboard.press("Escape")
    page.unroute("**/api/design/files")
    _press(page, "#saveDesign")
    assert page.evaluate("document.activeElement.id") == "designSaveName"


def test_enter_with_focus_on_the_background_opens_nothing(service, page):
    _set_design(service, ["wire-a"])
    _save_over_http(service, "background")
    _load(page, service)
    opens = _posts(page, "/api/design/open")
    _press(page, "#openDesign")
    _wait(page, lambda: "background" in _listed(page), "the list of saved designs")
    page.check("#designOpenModeReplace")

    # a point on the image clear of the panel: the panel is centred on the render pane,
    # so the image's bottom corners are tried first
    box = page.query_selector("#overlay").bounding_box()
    dlg = page.query_selector("#designDialog .configgearbox").bounding_box()
    corners = [(box["x"] + 30, box["y"] + box["height"] - 30),
               (box["x"] + box["width"] - 30, box["y"] + box["height"] - 30),
               (box["x"] + 30, box["y"] + 30)]
    clear = [(x, y) for x, y in corners
             if not (dlg["x"] <= x <= dlg["x"] + dlg["width"]
                     and dlg["y"] <= y <= dlg["y"] + dlg["height"])]
    assert clear, f"the panel {dlg} covers the whole image {box}"
    page.mouse.click(*clear[0])
    assert not _focus_is_in_dialog(page)
    for key in ("Enter", "NumpadEnter", "ArrowDown"):
        page.keyboard.press(key)
    page.wait_for_timeout(300)

    # a run control that disables itself on activation drops focus to the body
    page.focus("#perform")
    page.evaluate("document.getElementById('perform').disabled = true")
    page.keyboard.press("Enter")
    page.evaluate("document.activeElement.blur()")
    assert page.evaluate("document.activeElement === document.body")
    page.keyboard.press("Enter")
    page.wait_for_timeout(300)
    assert opens == [], f"Enter outside the dialog opened a file: {opens}"
    assert _shown(page, "#designDialog")
    assert _names(service) == ["wire-a"]


@pytest.mark.parametrize("opener", ["#saveDesign", "#openDesign"])
def test_a_held_enter_on_the_header_button_saves_and_opens_nothing(service, page, opener):
    _set_design(service, ["wire-a"])
    _save_over_http(service, "held-key")
    _load(page, service)
    _save_through_dialog(page, "held-key")           # so the name field is pre-filled
    _wait(page, lambda: not _shown(page, "#designDialog"), "the dialog to close")
    opens = _posts(page, "/api/design/open")
    saves = _posts(page, "/api/design/save")

    page.click('button.tab[data-tab="file"]', timeout=5000)
    page.focus(opener)
    for _ in range(8):
        page.keyboard.down("Enter")                  # the second and later are repeats
        page.wait_for_timeout(60)
    page.keyboard.up("Enter")
    page.wait_for_timeout(400)
    assert _shown(page, "#designDialog"), "the first Enter should have opened the dialog"
    assert opens == [] and saves == [], f"a held Enter went through: {opens} {saves}"

    # the dialog's own buttons ignore a held Enter too; one press still works
    confirm = "#designSaveConfirm" if opener == "#saveDesign" else "#designOpenConfirm"
    held = _hold(page, f"**/api/design/{'save' if opener == '#saveDesign' else 'open'}")
    page.focus(confirm)
    for _ in range(6):
        page.keyboard.down("Enter")
        page.wait_for_timeout(60)
    page.keyboard.up("Enter")
    page.wait_for_timeout(300)
    assert len(opens) + len(saves) == 1 and len(held) == 1, (opens, saves)
    held[0].continue_()
    _wait(page, lambda: not _shown(page, "#designDialog"), "the dialog to close")


# --------------------------------------------- replies, refetch, late calls --

def _answer(p, pattern: str, body, status: int = 200) -> None:
    p.route(pattern, lambda route: route.fulfill(
        status=status, content_type="application/json", body=json.dumps(body)))


def test_a_listing_with_bad_rows_shows_the_good_ones(service, page):
    _set_design(service, ["wire-a"])
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    _load(page, service)

    _answer(page, "**/api/design/files", {"ok": True, "designs": [
        None, 7, "text", {}, {"name": 3},
        {"name": "good", "saved_at": {"odd": 1}, "object_count": "many"}]})
    _press(page, "#openDesign")
    _wait(page, lambda: _listed(page) == ["good"], "the one usable row")
    assert page.get_attribute("#designListState", "data-state") == "ok"
    row = _text(page, '#designFileList .designfile[data-name="good"]')
    assert "object Object" not in row and "many" not in row, row
    page.keyboard.press("Escape")
    page.unroute("**/api/design/files")

    _answer(page, "**/api/design/files", {"ok": True, "designs": "not a list"})
    _press(page, "#openDesign")
    _wait(page, lambda: page.get_attribute("#designListState", "data-state") == "failed",
          "the listing reported as not understood")
    assert "not understood" in _text(page, "#designListState")
    assert errors == [], errors


def test_a_success_reply_with_missing_fields_is_not_announced(service, page):
    _set_design(service, ["wire-a"])
    _save_over_http(service, "one")
    _load(page, service)

    _answer(page, "**/api/design/save", {"ok": True, "name": "ghost"})
    _save_through_dialog(page, "ghost")
    text = _refusal(page)
    assert "not understood" in text and "undefined" not in text, text
    assert not _shown(page, "#designNotice")
    page.keyboard.press("Escape")
    page.unroute("**/api/design/save")

    for body in ({"ok": True, "name": "one"},
                 {"ok": True, "name": "one", "mode": "add", "object_count": 1,
                  "objects": [], "error": "but it did not happen"}):
        _answer(page, "**/api/design/open", body)
        _press(page, "#openDesign")
        _wait(page, lambda: "one" in _listed(page), "the list of saved designs")
        page.click('#designFileList .designfile[data-name="one"]')
        page.click("#designOpenConfirm")
        _wait(page, lambda: _shown(page, "#designDialogMsg")
              and "opening" not in _text(page, "#designDialogMsg"), "the failure message")
        said = _text(page, "#guardNoticeText") + " " + _text(page, "#designDialogMsg")
        assert "undefined" not in said and "null" not in said, said
        assert not _shown(page, "#designNotice"), f"announced as opened: {_notice_lines(page)}"
        page.keyboard.press("Escape")
        page.unroute("**/api/design/open")
    assert "but it did not happen" in _text(page, "#guardNoticeText")


def test_a_failed_refresh_after_an_open_is_said_and_retried(service, page):
    _set_design(service, ["wire-a", "pad-b"])
    _save_over_http(service, "fresh")
    _set_design(service, ["old-one"])
    _load(page, service)
    assert _object_names(page) == ["old-one"]

    failing = {"on": True}

    def design(route):
        if failing["on"] and route.request.method == "GET":
            route.fulfill(status=500, content_type="application/json", body="{}")
        else:
            route.continue_()

    page.route("**/api/design", design)
    _open_through_dialog(page, "fresh", "replace")
    _wait(page, lambda: "could not be refreshed" in _text(page, "#guardNoticeText")
          and _shown(page, "#guardNotice"), "the notice that the screen is stale")
    assert "fresh was opened" in _text(page, "#guardNoticeText")
    assert _object_names(page) == ["old-one"]
    assert _names(service) == ["wire-a", "pad-b"]

    # another refusal takes the notice; the next retry must not write over it
    _save_through_dialog(page, "_previous")
    _wait(page, lambda: "reserved_name" in _text(page, "#guardNoticeText"), "the refusal")
    page.keyboard.press("Escape")
    page.wait_for_timeout(2600)
    assert "reserved_name" in _text(page, "#guardNoticeText"), (
        f"a retry wrote over another notice: {_text(page, '#guardNoticeText')!r}")

    failing["on"] = False
    _wait(page, lambda: _object_names(page) == ["wire-a", "pad-b"],
          "a retry to bring the opened design on screen", timeout_s=15.0)


def test_a_late_refusal_is_not_written_into_a_reopened_dialog(service, page):
    _set_design(service, ["wire-a"])
    _save_over_http(service, "late")
    _load(page, service)
    held = _hold(page, "**/api/design/open")
    _press(page, "#openDesign")
    _wait(page, lambda: "late" in _listed(page), "the list of saved designs")
    page.click('#designFileList .designfile[data-name="late"]')
    page.click("#designOpenConfirm")
    _wait(page, lambda: len(held) == 1, "the held open")
    page.keyboard.press("Escape")
    _press(page, "#saveDesign")
    page.fill("#designSaveName", "typed-meanwhile")

    held[0].fulfill(status=200, content_type="application/json",
                    body=json.dumps({"error": "late is gone", "reason": "not_found"}))
    text = _refusal(page)
    assert "late is gone" in text, text
    page.wait_for_timeout(200)
    assert not _shown(page, "#designDialogMsg"), (
        f"the earlier call wrote into the Save dialog: {_text(page, '#designDialogMsg')!r}")
    assert page.input_value("#designSaveName") == "typed-meanwhile"


def test_the_list_refreshes_when_a_save_lands(service, page):
    _set_design(service, ["wire-a"])
    _load(page, service)
    held = _hold(page, "**/api/design/save")
    _save_through_dialog(page, "landed")
    _wait(page, lambda: len(held) == 1, "the held save")
    page.keyboard.press("Escape")
    _press(page, "#openDesign")
    page.wait_for_timeout(500)
    assert "landed" not in _listed(page)
    held[0].continue_()
    _wait(page, lambda: "landed" in _listed(page), "the new file in the open list")
    assert _shown(page, "#designDialog") and _shown(page, "#designOpenSection")


def test_no_answer_in_60_s_is_unknown_and_the_late_reply_corrects_it(service, page):
    _set_design(service, ["wire-a", "pad-b"])
    _save_over_http(service, "slow-build")
    _set_design(service, ["wire-a"])
    page.clock.install()
    _load(page, service)
    held = _hold(page, "**/api/design/open")
    _press(page, "#openDesign")
    _wait(page, lambda: "slow-build" in _listed(page), "the list of saved designs")
    page.click('#designFileList .designfile[data-name="slow-build"]')
    page.click("#designOpenConfirm")
    _wait(page, lambda: len(held) == 1, "the held open")
    page.clock.fast_forward(59000)
    page.wait_for_timeout(200)
    assert "opening slow-build" in _text(page, "#designDialogMsg")
    page.clock.fast_forward(2000)
    text = _refusal(page)
    assert "no answer after 60 s; it may still complete" in text, text
    assert "failed" not in text, text
    assert "it may still complete" in _text(page, "#designDialogMsg")

    held[0].continue_()
    _wait(page, lambda: _shown(page, "#designNotice"), "the real result")
    lines = _notice_lines(page)
    assert lines[0] == "slow-build opened: 2 objects added (3 in the design now)", lines
    assert any(line.startswith("renamed: wire-a") for line in lines), lines
    _wait(page, lambda: not _shown(page, "#guardNotice"), "the no-answer notice to clear")
    assert "may still complete" not in _text(page, "#designDialogMsg")
    assert len(_names(service)) == 3


# ------------------------------------------------- names the service lists --

def _click_listed(p, name: str) -> None:
    p.evaluate(
        "name => [...document.querySelectorAll('#designFileList .designfile')]"
        ".find(e => e.dataset.name === name).click()", name)


def test_a_listed_name_is_sent_exactly_as_listed(service, page):
    _set_design(service, ["wire-a"])
    _save_over_http(service, "plain")
    folder = service.runtime / "designs"
    odd = "sp ace & 'q'; é"
    shutil.copy(folder / "plain.design.json", folder / f"{odd}.design.json")
    try:
        _load(page, service)
        sent: list[dict] = []
        page.on("request", lambda r: sent.append(r.post_data_json)
                if r.url.endswith("/api/design/open") else None)
        _press(page, "#openDesign")
        _wait(page, lambda: odd in _listed(page), "the oddly named file in the list")
        _click_listed(page, odd)
        page.click("#designOpenConfirm")
        _wait(page, lambda: not _shown(page, "#designDialog"), "the dialog to close")
        assert sent and sent[0]["name"] == odd, sent
        _wait(page, lambda: len(_names(service)) == 2, "the oddly named file opened")
    finally:
        (folder / f"{odd}.design.json").unlink(missing_ok=True)


def test_a_set_aside_autosave_is_labelled_and_opens_what_it_can(service, page):
    _set_design(service, ["wire-a", "pad-b"])
    _save_over_http(service, "plain")
    _set_design(service, ["scratch"])
    folder = service.runtime / "designs"
    doc = json.loads((folder / "plain.design.json").read_text("utf-8"))
    doc["objects"][1]["kind"] = "no-such-kind"
    aside = folder / "_autosave.bad.json"
    aside.write_text(json.dumps(doc), "utf-8")
    try:
        listed = _saved(service)
        assert listed["_autosave.bad"].get("recovered") is True, listed
        assert "recovered" not in listed["plain"]
        _load(page, service)
        _press(page, "#openDesign")
        _wait(page, lambda: "_autosave.bad" in _listed(page), "the set-aside copy in the list")
        row = _text(page, '#designFileList .designfile[data-name="_autosave.bad"]')
        assert "recovered autosave copy" in row, row
        assert "recovered" not in _text(page, '#designFileList .designfile[data-name="plain"]')
        _click_listed(page, "_autosave.bad")
        page.click("#designOpenConfirm")
        _wait(page, lambda: not _shown(page, "#designDialog"), "the dialog to close")
        _wait(page, lambda: _names(service) == ["scratch", "wire-a"], "what could be opened")
        lines = _notice_lines(page)
        assert lines[0].startswith("_autosave.bad opened: 1 object added"), lines
        left = [line for line in lines if line.startswith("left out: ")]
        assert len(left) == 1 and "pad-b" in left[0], lines

        _press(page, "#saveDesign")
        assert page.input_value("#designSaveName") != "_autosave.bad"
        page.fill("#designSaveName", "_autosave.bad")
        page.click("#designSaveConfirm")
        text = _refusal(page)
        assert "reserved_name" in text, text
        page.keyboard.press("Escape")

        for record in doc["objects"]:
            record["kind"] = "no-such-kind"
        aside.write_text(json.dumps(doc), "utf-8")
        _press(page, "#openDesign")
        _wait(page, lambda: "_autosave.bad" in _listed(page), "the set-aside copy in the list")
        _click_listed(page, "_autosave.bad")
        page.click("#designOpenConfirm")
        _wait(page, lambda: "no_buildable_record" in _text(page, "#guardNoticeText"),
              "the refusal of a copy with nothing to open")
        assert _names(service) == ["scratch", "wire-a"]
    finally:
        aside.unlink(missing_ok=True)


# ----------------------------------------------------------- chiral wires --

PITCH_UM = 0.025


def _seed_chiral(service) -> dict:
    _set_design(service, [])
    status, r = service.post("/api/design/add", {
        "kind": "path", "name": "w", "points_um": [[5.0, 5.0], [6.0, 5.0]]})
    assert status == 200 and r and r.get("ok"), (status, r)
    status, r = service.post("/api/design/chiral", {
        "indices": [r["index"]], "enabled": True, "lambda_um": PITCH_UM,
        "y_amp_um": 0.055, "v_k": 5.0, "phase_deg": 90, "v0": 8})
    assert status == 200 and r and r.get("ok"), (status, r)
    return _object(service, "w")


def _object(service, name: str) -> dict:
    status, d = service.get("/api/design")
    assert status == 200, (status, d)
    return next(o for o in d["objects"] if o["name"] == name)


def _page_object(p, name: str) -> dict:
    return p.evaluate(
        "async (name) => { const {state} = await import('/static/ec-state.js');"
        " const o = state.objects.find((x) => x.name === name);"
        " return o ? {points: o.points_um, v_profile: o.v_profile,"
        "             profile: o.profile} : null; }", name)


@pytest.mark.parametrize("mode", ["add", "replace"])
def test_an_opened_chiral_wire_is_drawn_as_saved_and_keeps_its_pitch(service, page, mode):
    saved = _seed_chiral(service)
    assert len(saved["points_um"]) > 2 and saved["profile"], saved
    _save_over_http(service, "chiral")
    _set_design(service, ["other"])
    _load(page, service)
    gets: list[str] = []
    page.on("request", lambda r: gets.append(r.url)
            if r.method == "GET" and r.url.endswith("/api/design") else None)

    _open_through_dialog(page, "chiral", mode)
    expected = ["w"] if mode == "replace" else ["other", "w"]
    _wait(page, lambda: _object_names(page) == expected, "the opened wire in the list")
    opened = _object(service, "w")
    assert opened["points_um"] == saved["points_um"], "the wire's path changed on open"
    assert opened["v_profile"] == saved["v_profile"]

    on_page = _page_object(page, "w")
    assert on_page["points"] == opened["points_um"], "the page draws a different path"
    assert on_page["v_profile"] == opened["v_profile"]

    page.click('button.tab[data-tab="write"]')
    page.click(f'#objList .objrow[data-i="{expected.index("w")}"]')
    _wait(page, lambda: page.input_value("#fLambda") != "", "the Pitch field")
    assert float(page.input_value("#fLambda")) == PITCH_UM
    page.wait_for_timeout(500)
    assert 1 <= len(gets) <= 2, f"the design was fetched {len(gets)} times after one open"
    assert not _shown(page, "#guardNotice")
