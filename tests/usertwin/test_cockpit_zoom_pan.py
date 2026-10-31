"""Usertwin coverage for docs/COCKPIT_ZOOM_PAN_SPEC.md, in a real browser against a
twin service: Ctrl+wheel (a touchpad pinch) zooms about the pointer and clamps at
both limits, plain wheel pans, left-drag pans on Scan only, right-drag and
Space+drag pan on Edit without touching an object, no context menu over the canvas,
double-click on Scan returns to the scan frame, the view readout tracks the view and
`state.view` drives the canvas, and a space typed in a field does not start a pan.

Reuses `test_instrument_cockpit.py`'s service/browser/FLEX plumbing.
"""

# ruff: noqa: F811 -- fixtures imported by name shadow their definitions when taken
# as test parameters; the standard pytest cross-file fixture-reuse pattern.

from __future__ import annotations

import re

import pytest
from test_instrument_cockpit import (  # reuse plumbing, see module docstring
    browser,  # noqa: F401 -- fixture re-export
    flex,  # noqa: F401 -- fixture re-export
    service,  # noqa: F401 -- fixture re-export
)
from test_instrument_cockpit_b3b import _ensure_flex_control

# the object the Edit-tab cases press on, loaded over FLEX
BOX = {"kind": "region", "name": "zp_box", "shape": "rect", "x_um": 2.0, "y_um": 2.0,
       "w_um": 4.0, "h_um": 3.0, "pitch_um": 0.15, "voltage": 6.0}

VIEW_MIN_UM = 0.1
VIEW_BOUNDS_MARGIN = 0.1


@pytest.fixture
def page(service, flex, browser):
    _ensure_flex_control(flex)
    r = flex.call("loadPattern", {"token": flex.token, "objects": [dict(BOX)]})
    assert r is not None and not (isinstance(r, dict) and r.get("error")), r
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000}, permissions=[])
    p = ctx.new_page()
    p.goto(service.url + "/edit-cockpit", wait_until="load", timeout=45000)
    p.wait_for_function(
        "async () => (await import('/static/ec-state.js')).state.objects"
        ".some(o => o.name === 'zp_box')", timeout=15000)
    p.wait_for_timeout(300)
    yield p
    ctx.close()


def _tab(p, name: str) -> None:
    sel = f'button.tab[data-tab="{name}"]'
    # Connect's tab button is hidden but its handler still switches the tab.
    if name == "connect":
        p.eval_on_selector(sel, "b => b.click()")
    else:
        p.click(sel, timeout=5000)
    p.wait_for_timeout(200)


def _view(p) -> dict:
    return p.evaluate("async () => { const {state} = await import('/static/ec-state.js');"
                      " return {...(state.view || state.SCAN_FRAME)}; }")


def _set_view(p, v: dict) -> None:
    p.evaluate("async (v) => { const {state} = await import('/static/ec-state.js');"
               " state.view = v; }", v)


def _settle(p) -> None:
    p.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")


def _um_to_client(p, x: float, y: float) -> tuple[float, float]:
    return tuple(p.evaluate(
        """async ([x, y]) => {
             const {state} = await import('/static/ec-state.js');
             const g = await import('/static/ec-geom.js');
             const cv = document.getElementById('overlay');
             const r = cv.getBoundingClientRect();
             const [px, py] = g.fitTransform(state.view || state.SCAN_FRAME,
                                             cv.width, cv.height)(x, y);
             return [r.left + px * r.width / cv.width, r.top + py * r.height / cv.height];
           }""", [x, y]))


def _client_to_um(p, cx: float, cy: float) -> tuple[float, float]:
    return tuple(p.evaluate(
        """async ([cx, cy]) => {
             const {state} = await import('/static/ec-state.js');
             const g = await import('/static/ec-geom.js');
             const cv = document.getElementById('overlay');
             const r = cv.getBoundingClientRect();
             return g.invTransform(state.view || state.SCAN_FRAME, cv.width, cv.height)(
               (cx - r.left) * cv.width / r.width, (cy - r.top) * cv.height / r.height);
           }""", [cx, cy]))


def _css_px_per_um(p) -> float:
    return p.evaluate(
        """async () => {
             const {state} = await import('/static/ec-state.js');
             const g = await import('/static/ec-geom.js');
             const cv = document.getElementById('overlay');
             const r = cv.getBoundingClientRect();
             return g.pxPerUm(state.view || state.SCAN_FRAME, cv.width, cv.height)
                    * r.width / cv.width;
           }""")


def _canvas_box(p) -> dict:
    return p.eval_on_selector("#overlay", "e => { const r = e.getBoundingClientRect();"
                              " return {x: r.left, y: r.top, w: r.width, h: r.height}; }")


def _bounds(p) -> dict:
    """The navigable area, computed here from its parts: the scan frame, the design
    bounds and the instrument's xy_range_um (centred on the origin), plus a margin."""
    b = p.evaluate(
        """async () => {
             const {state} = await import('/static/ec-state.js');
             const g = await import('/static/ec-geom.js');
             const m = await (await fetch('/api/instrument')).json();
             const f = state.SCAN_FRAME, d = g.boundsOfObjects(state.objects);
             const r = m.xy_range_um.value;
             const rb = {x0: -r[0] / 2, y0: -r[1] / 2, w: r[0], h: r[1]};
             const bs = [f, d, rb];
             return {x0: Math.min(...bs.map(q => q.x0)), y0: Math.min(...bs.map(q => q.y0)),
                     x1: Math.max(...bs.map(q => q.x0 + q.w)),
                     y1: Math.max(...bs.map(q => q.y0 + q.h))};
           }""")
    m = max(b["x1"] - b["x0"], b["y1"] - b["y0"]) * VIEW_BOUNDS_MARGIN
    return {"x0": b["x0"] - m, "y0": b["y0"] - m,
            "x1": b["x1"] + m, "y1": b["y1"] + m}


def _max_view_um(p) -> float:
    b = _bounds(p)
    return max(b["x1"] - b["x0"], b["y1"] - b["y0"])


def _box_obj(p) -> dict:
    return p.evaluate("async () => { const {state} = await import('/static/ec-state.js');"
                      " const i = state.objects.findIndex(o => o.name === 'zp_box');"
                      " return {i, sel: state.selected,"
                      " pts: state.objects[i].points_um.map(q => [q[0], q[1]])}; }")


def _drag(p, x0, y0, x1, y1, button="left") -> None:
    # whole pixels: mouse events report integer client coordinates
    x0, y0, x1, y1 = (round(c) for c in (x0, y0, x1, y1))
    p.mouse.move(x0, y0)
    p.mouse.down(button=button)
    for k in range(1, 6):
        p.mouse.move(x0 + (x1 - x0) * k / 5, y0 + (y1 - y0) * k / 5)
    p.mouse.up(button=button)
    _settle(p)


def _ctrl_wheel(p, x, y, dy, times=1) -> None:
    p.mouse.move(x, y)
    p.keyboard.down("Control")
    try:
        for _ in range(times):
            p.mouse.wheel(0, dy)
            _settle(p)
    finally:
        p.keyboard.up("Control")
    _settle(p)


def _same_view(a: dict, b: dict, tol: float = 1e-9) -> bool:
    return all(abs(a[k] - b[k]) <= tol * max(1.0, abs(b[k])) for k in ("x0", "y0", "w", "h"))


def _empty_point(p) -> tuple[float, float]:
    """A canvas point well clear of the loaded box (left edge, mid height)."""
    b = _canvas_box(p)
    return b["x"] + 30, b["y"] + b["h"] / 2


# -------------------------------------------------------------------- tests --

def test_ctrl_wheel_zooms_about_the_pointer_and_clamps(page):
    p = page
    b = _canvas_box(p)
    # whole pixels: the browser reports wheel clientX/clientY as integers
    cx, cy = round(b["x"] + b["w"] * 0.3), round(b["y"] + b["h"] * 0.65)
    before = _view(p)
    u0 = _client_to_um(p, cx, cy)
    _ctrl_wheel(p, cx, cy, -100)
    after = _view(p)
    assert after["w"] < before["w"], (before, after)
    u1 = _client_to_um(p, cx, cy)
    assert abs(u1[0] - u0[0]) < 1e-6 and abs(u1[1] - u0[1]) < 1e-6, (u0, u1)
    # the page itself was not zoomed by the browser
    assert p.evaluate("() => window.visualViewport ? window.visualViewport.scale : 1") == 1

    _ctrl_wheel(p, cx, cy, -500, times=40)
    v = _view(p)
    assert max(v["w"], v["h"]) == pytest.approx(VIEW_MIN_UM, rel=1e-6), v
    u2 = _client_to_um(p, cx, cy)
    assert abs(u2[0] - u0[0]) < 1e-6 and abs(u2[1] - u0[1]) < 1e-6, (u0, u2)

    _ctrl_wheel(p, cx, cy, 500, times=60)
    v = _view(p)
    assert max(v["w"], v["h"]) == pytest.approx(_max_view_um(p), rel=1e-6), v


def test_pan_and_zoom_out_stop_at_the_design_and_scan_range_bounds(page):
    p = page
    _tab(p, "scan")
    b = _canvas_box(p)
    cx, cy = round(b["x"] + b["w"] / 2), round(b["y"] + b["h"] / 2)
    bd = _bounds(p)
    # the scan range is the larger part on the twin, so it sets the area
    assert bd["x1"] - bd["x0"] > 50, bd
    # a long pan to the right and down stops at the bounds' edges
    p.mouse.move(cx, cy)
    for _ in range(80):
        p.mouse.wheel(400, 400)
    _settle(p)
    v = _view(p)
    assert v["x0"] + v["w"] == pytest.approx(bd["x1"], abs=1e-6), (v, bd)
    assert v["y0"] == pytest.approx(bd["y0"], abs=1e-6), (v, bd)
    # and the other way
    _drag(p, cx - 200, cy - 150, cx + 200, cy + 150)
    for _ in range(80):
        p.mouse.wheel(-400, -400)
    _settle(p)
    v = _view(p)
    assert v["x0"] == pytest.approx(bd["x0"], abs=1e-6), (v, bd)
    assert v["y0"] + v["h"] == pytest.approx(bd["y1"], abs=1e-6), (v, bd)
    # zoomed fully out, the view's larger side spans the bounds, centred on them,
    # and the view stays inside them
    _ctrl_wheel(p, cx, cy, 500, times=60)
    v = _view(p)
    assert max(v["w"], v["h"]) == pytest.approx(_max_view_um(p), rel=1e-6), v
    assert v["x0"] + v["w"] / 2 == pytest.approx((bd["x0"] + bd["x1"]) / 2, abs=1e-6)
    assert bd["y0"] - 1e-6 <= v["y0"] and v["y0"] + v["h"] <= bd["y1"] + 1e-6, (v, bd)
    p.screenshot(path="tmp-shots/7b-pan-bounds.png")


def test_small_pinch_deltas_zoom_smoothly(page):
    p = page
    b = _canvas_box(p)
    cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
    w0 = _view(p)["w"]
    _ctrl_wheel(p, cx, cy, -2)
    w1 = _view(p)["w"]
    assert 0.98 < w1 / w0 < 1.0, (w0, w1)


def test_plain_wheel_pans(page):
    p = page
    b = _canvas_box(p)
    cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
    s = _css_px_per_um(p)
    v0 = _view(p)
    p.mouse.move(cx, cy)
    p.mouse.wheel(0, 120)
    _settle(p)
    v1 = _view(p)
    assert v1["w"] == pytest.approx(v0["w"]) and v1["h"] == pytest.approx(v0["h"])
    assert v1["y0"] == pytest.approx(v0["y0"] - 120 / s, abs=0.6 / s), (v0, v1)
    assert v1["x0"] == pytest.approx(v0["x0"])
    p.mouse.wheel(80, 0)
    _settle(p)
    v2 = _view(p)
    assert v2["x0"] == pytest.approx(v1["x0"] + 80 / s, rel=1e-6), (v1, v2)
    # the page did not scroll
    assert p.evaluate("() => window.scrollY") == 0


def test_left_drag_pans_on_scan(page):
    p = page
    _tab(p, "scan")
    b = _canvas_box(p)
    s = _css_px_per_um(p)
    v0 = _view(p)
    x, y = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
    _drag(p, x, y, x + 100, y + 60)
    v1 = _view(p)
    assert v1["x0"] == pytest.approx(v0["x0"] - 100 / s, abs=0.6 / s), (v0, v1)
    assert v1["y0"] == pytest.approx(v0["y0"] + 60 / s, abs=0.6 / s), (v0, v1)
    assert v1["w"] == pytest.approx(v0["w"])


@pytest.mark.parametrize("tab", ["edit", "connect"])
def test_left_drag_does_not_pan_off_scan(page, tab):
    p = page
    _tab(p, tab)
    if tab == "edit":
        p.click('[data-tool="select"]')
    v0 = _view(p)
    x, y = _empty_point(p)
    _drag(p, x, y, x + 120, y + 40)
    assert _same_view(_view(p), v0), (v0, _view(p))


@pytest.mark.parametrize("how", ["right", "space"])
def test_right_drag_and_space_drag_pan_on_edit_without_touching_objects(page, how):
    p = page
    _tab(p, "edit")
    p.click('[data-tool="select"]')
    before = _box_obj(p)
    assert before["sel"] == -1, before
    pts = before["pts"]
    mx = sum(q[0] for q in pts) / len(pts)
    my = sum(q[1] for q in pts) / len(pts)
    x, y = _um_to_client(p, mx, my)
    s = _css_px_per_um(p)
    v0 = _view(p)
    if how == "space":
        p.mouse.move(x, y)
        p.keyboard.down(" ")
        try:
            _drag(p, x, y, x + 90, y - 50)
        finally:
            p.keyboard.up(" ")
    else:
        _drag(p, x, y, x + 90, y - 50, button="right")
    v1 = _view(p)
    assert v1["x0"] == pytest.approx(v0["x0"] - 90 / s, abs=0.6 / s), (v0, v1)
    assert v1["y0"] == pytest.approx(v0["y0"] - 50 / s, abs=0.6 / s), (v0, v1)
    after = _box_obj(p)
    assert after["sel"] == -1, "the pan selected the object"
    assert after["pts"] == pts, "the pan moved the object"
    # the select tool still works once the pan is over
    x2, y2 = _um_to_client(p, mx, my)
    p.mouse.click(x2, y2)
    _settle(p)
    assert _box_obj(p)["sel"] == before["i"]


def test_no_context_menu_over_the_canvas_only(page):
    p = page
    p.evaluate("() => { window.__cm = []; window.addEventListener('contextmenu',"
               " e => window.__cm.push([e.target.id, e.defaultPrevented])); }")
    for tab in ("scan", "edit"):
        _tab(p, tab)
        x, y = _empty_point(p)
        p.mouse.click(x, y, button="right")
    lg = p.eval_on_selector("#log", "e => { const r = e.getBoundingClientRect();"
                            " return [r.left + 10, r.top + 10]; }")
    p.mouse.click(lg[0], lg[1], button="right")
    p.keyboard.press("Escape")
    cm = p.evaluate("() => window.__cm")
    assert cm[:2] == [["overlay", True], ["overlay", True]], cm
    assert cm[2][0] != "overlay" and cm[2][1] is False, cm


def test_double_click_on_scan_returns_to_the_scan_frame(page):
    p = page
    _tab(p, "scan")
    b = _canvas_box(p)
    cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
    _ctrl_wheel(p, cx, cy, -300)
    p.mouse.wheel(40, 40)
    _settle(p)
    frame = p.evaluate("async () => ({...(await import('/static/ec-state.js')).state.SCAN_FRAME})")
    assert not _same_view(_view(p), frame)
    p.mouse.dblclick(cx, cy)
    _settle(p)
    assert _same_view(_view(p), frame), (_view(p), frame)


def _view_cap(p) -> tuple[float, float, float, float]:
    """(w, h, cx, cy) parsed from the #viewCap readout a zoom or pan writes."""
    txt = p.inner_text("#viewCap")
    m = re.search(r"([-\d.]+)\D+?([-\d.]+) µm @ \(([-\d.]+), ([-\d.]+)\)", txt)
    assert m, txt
    return tuple(float(g) for g in m.groups())


def test_view_readout_tracks_the_view_and_state_view_drives_it(page):
    p = page
    _tab(p, "scan")
    b = _canvas_box(p)
    cx, cy = b["x"] + b["w"] * 0.7, b["y"] + b["h"] * 0.4
    v0 = _view(p)
    _ctrl_wheel(p, cx, cy, -250)
    _drag(p, cx, cy, cx - 70, cy + 30)
    v = _view(p)
    assert v["w"] < 0.9 * v0["w"] and abs(v["x0"] - v0["x0"]) > 0.01, (v0, v)
    w, h, ox, oy = _view_cap(p)
    tol = 0.01 * max(v["w"], v["h"])
    assert abs(w - v["w"]) <= tol and abs(h - v["h"]) <= tol, (w, h, v)
    assert abs(ox - (v["x0"] + v["w"] / 2)) <= tol, (ox, v)
    assert abs(oy - (v["y0"] + v["h"] / 2)) <= tol, (oy, v)
    # a view set from the page is the one the next gesture starts from
    assert not _same_view(v, {"x0": -2.0, "y0": -2.0, "w": 4.0, "h": 4.0}), v
    _set_view(p, {"x0": -2.0, "y0": -2.0, "w": 4.0, "h": 4.0})
    x, y = round(b["x"] + b["w"] / 2), round(b["y"] + b["h"] / 2)
    ux, uy = _client_to_um(p, x, y)
    _ctrl_wheel(p, x, y, -100)
    v2 = _view(p)
    assert 0.5 * 4.0 < v2["w"] < 4.0, v2
    u2 = _client_to_um(p, x, y)
    assert abs(u2[0] - ux) < 1e-6 and abs(u2[1] - uy) < 1e-6, (ux, uy, u2)
    w2, *_ = _view_cap(p)
    assert abs(w2 - v2["w"]) <= 0.01 * v2["w"], (w2, v2)


def test_view_readout_follows_a_frame_up(page, service):
    p = page
    _tab(p, "scan")
    b = _canvas_box(p)
    _ctrl_wheel(p, b["x"] + b["w"] / 2, b["y"] + b["h"] / 2, -250)
    p.fill("#size", "3")
    try:
        p.click("#up")
        _settle(p)
        frame = p.evaluate(
            "async () => ({...(await import('/static/ec-state.js')).state.SCAN_FRAME})")
        assert frame["w"] == pytest.approx(3.0), frame
        assert _same_view(_view(p), frame), (_view(p), frame)
        assert p.inner_text("#viewCap") == "view: scan frame (3×3 µm)", p.inner_text("#viewCap")
    finally:
        service.post("/api/scan/abort", {"withdraw": True})


def _half_screen(browser, service, stored_px=None):
    ctx = browser.new_context(viewport={"width": 948, "height": 1036}, permissions=[])
    if stored_px is not None:
        ctx.add_init_script(
            f"try {{ localStorage.setItem('ec.rightPaneWidthPx.half', '{stored_px}'); }}"
            " catch {}")
    p = ctx.new_page()
    p.goto(service.url + "/edit-cockpit", wait_until="load", timeout=45000)
    p.wait_for_timeout(800)
    return ctx, p


def _cockpit(browser, service, width, stored=None):
    """A fresh page at `width` x 1036 with `stored` ({key: value}) in localStorage."""
    ctx = browser.new_context(viewport={"width": width, "height": 1036}, permissions=[])
    if stored:
        ctx.add_init_script(
            "try { " + " ".join(f"localStorage.setItem('{k}', '{v}');"
                                for k, v in stored.items()) + " } catch {}")
    p = ctx.new_page()
    p.goto(service.url + "/edit-cockpit", wait_until="load", timeout=45000)
    p.wait_for_timeout(800)
    return ctx, p


def _right_width(p):
    return p.eval_on_selector(".rightpanel", "e => e.getBoundingClientRect().width")


def _arr(p):
    return p.evaluate("() => getComputedStyle(document.getElementById('renderBody'))"
                      ".getPropertyValue('--arr').trim()")


def _resize(p, width):
    p.set_viewport_size({"width": width, "height": 1036})
    p.wait_for_timeout(300)


def test_a_width_saved_under_the_old_single_key_is_dropped(service, browser):
    ctx, p = _cockpit(browser, service, 1896, {"ec.rightPaneWidthPx": 470})
    try:
        assert _right_width(p) == pytest.approx(600, abs=1), "full screen did not start at 600"
        assert p.evaluate("() => localStorage.getItem('ec.rightPaneWidthPx')") is None
        _resize(p, 948)
        assert abs(_right_width(p) - p.eval_on_selector(
            ".scanpanel", "e => e.getBoundingClientRect().width")) <= 2, "half not split equally"
    finally:
        ctx.close()


def test_the_full_and_half_screen_widths_are_stored_apart(service, browser):
    ctx, p = _cockpit(browser, service, 1896, {"ec.rightPaneWidthPx.full": 700,
                                                "ec.rightPaneWidthPx.half": 500})
    try:
        assert _right_width(p) == pytest.approx(700, abs=1)
        _resize(p, 948)
        assert _right_width(p) == pytest.approx(500, abs=1)
        p.focus("#paneDivider")
        p.keyboard.press("ArrowLeft")      # the right pane grows by one step
        p.wait_for_timeout(100)
        stored = p.evaluate("() => [localStorage.getItem('ec.rightPaneWidthPx.full'),"
                            " localStorage.getItem('ec.rightPaneWidthPx.half')]")
        assert stored == ["700", "524"], stored
        _resize(p, 1896)
        assert _right_width(p) == pytest.approx(700, abs=1), "the half drag moved full screen"
    finally:
        ctx.close()


def test_the_log_folds_when_the_half_arrangement_turns_on(service, browser):
    ctx, p = _cockpit(browser, service, 1896)
    try:
        folded = lambda: p.eval_on_selector(  # noqa: E731
            ".logpanel", "e => e.classList.contains('collapsed')")
        assert _arr(p) == "" and not folded()
        _resize(p, 948)
        assert _arr(p) == "half" and folded(), "half screen left the log open"
        _resize(p, 1896)
        assert _arr(p) == "" and not folded(), "leaving half screen kept the log folded"
    finally:
        ctx.close()


def test_the_arrangement_switches_at_a_659_px_render_pane(service, browser):
    ctx, p = _cockpit(browser, service, 1600)
    try:
        for outer, want in ((660, ""), (659, "half")):
            got = p.evaluate(
                """o => { const main = document.querySelector('main');
                     const dw = document.getElementById('paneDivider').offsetWidth;
                     document.querySelector('.rightpanel').style.width =
                       (main.clientWidth - dw - o) + 'px';
                     return document.querySelector('.scanpanel').getBoundingClientRect().width; }""",
                outer)
            assert got == pytest.approx(outer, abs=0.5), (outer, got)
            assert _arr(p) == want, f"render pane {outer} px: --arr {_arr(p)!r}"
    finally:
        ctx.close()


def test_no_tab_scrolls_the_page_sideways_with_the_right_pane_at_its_minimum(service, browser):
    ctx, p = _half_screen(browser, service, stored_px=1)
    try:
        min_px = p.evaluate("async () => (await import('/static/ec-panes.js')).MIN_RIGHT_PX")
        width = p.eval_on_selector(".rightpanel", "e => e.getBoundingClientRect().width")
        assert width == pytest.approx(min_px, abs=1), (width, min_px)
        for tab in ("scan", "connect", "edit", "write", "file"):
            _tab(p, tab)
            over = p.evaluate(
                """() => ['html', 'body', 'main', '.rightpanel', '.rightbody']
                     .map(s => [s, document.querySelector(s)])
                     .filter(([, e]) => e.scrollWidth > e.clientWidth)
                     .map(([s, e]) => [s, e.scrollWidth, e.clientWidth])""")
            assert over == [], f"{tab}: content wider than its box: {over}"
    finally:
        ctx.close()


def test_render_toolbar_scrolls_sideways_by_wheel_and_drag(service, browser):
    ctx, p = _half_screen(browser, service)
    try:
        bar = p.eval_on_selector("#renderTools", "e => { const r = e.getBoundingClientRect();"
                                 " return {x: r.left, y: r.top, w: r.width, h: r.height,"
                                 " max: e.scrollWidth - e.clientWidth}; }")
        assert bar["max"] > 0, "the toolbar fits at half screen; nothing to scroll"
        scroll = lambda: p.eval_on_selector("#renderTools", "e => e.scrollLeft")  # noqa: E731
        p.mouse.move(bar["x"] + bar["w"] / 2, bar["y"] + bar["h"] / 2)
        p.mouse.wheel(0, 2000)
        _settle(p)
        assert scroll() == pytest.approx(bar["max"], abs=1), scroll()
        # Zoom to scan is now on screen and takes a real click
        z = p.eval_on_selector("#zoomScan", "e => { const r = e.getBoundingClientRect();"
                               " return [r.left + r.width / 2, r.top + r.height / 2]; }")
        assert z[0] < bar["x"] + bar["w"], z
        assert p.evaluate("([x, y]) => document.elementFromPoint(x, y).id", z) == "zoomScan"
        # a drag that starts on the gap between the first two groups scrolls it back
        p.eval_on_selector("#renderTools", "e => { e.scrollLeft = 0; }")
        gx = p.eval_on_selector(
            "#renderTools", "e => (e.children[0].getBoundingClientRect().right"
            " + e.children[1].getBoundingClientRect().left) / 2")
        gy = bar["y"] + bar["h"] / 2
        assert p.evaluate("([x, y]) => document.elementFromPoint(x, y).id",
                          [gx, gy]) == "renderTools"
        _drag(p, gx, gy, gx - 60, gy)
        assert scroll() == pytest.approx(min(60, bar["max"]), abs=2), scroll()
    finally:
        ctx.close()


def test_space_typed_in_a_field_does_not_start_a_pan(page):
    p = page
    # Write: a left-drag there pans only while Space is held, as on Edit
    _tab(p, "write")
    p.evaluate("() => { window.__kd = []; window.addEventListener('keydown',"
               " e => window.__kd.push(e.defaultPrevented)); }")
    v0 = _view(p)
    p.focus("#lithoSetpoint")
    p.keyboard.down(" ")
    try:
        x, y = _empty_point(p)
        _drag(p, x, y, x + 120, y + 40)
    finally:
        p.keyboard.up(" ")
    assert _same_view(_view(p), v0), (v0, _view(p))
    assert p.evaluate("() => window.__kd") == [False]


def test_right_drag_pan_released_off_the_canvas_opens_no_menu(page):
    p = page
    _tab(p, "edit")
    p.evaluate("() => { window.__cm = []; window.addEventListener('contextmenu',"
               " e => window.__cm.push([e.target.id, e.defaultPrevented])); }")
    b = _canvas_box(p)
    x, y = _empty_point(p)
    v0 = _view(p)
    # past the canvas's right edge, over the objects panel
    _drag(p, x, y, b["x"] + b["w"] + 200, y, button="right")
    assert not _same_view(_view(p), v0)
    p.wait_for_timeout(600)
    # a later plain right-click off the canvas still gets its menu
    lg = p.eval_on_selector("#log", "e => { const r = e.getBoundingClientRect();"
                            " return [r.left + 10, r.top + 10]; }")
    p.mouse.click(lg[0], lg[1], button="right")
    p.keyboard.press("Escape")
    cm = p.evaluate("() => window.__cm")
    assert len(cm) == 2, cm
    assert cm[0][0] != "overlay" and cm[0][1] is True, cm
    assert cm[1][0] != "overlay" and cm[1][1] is False, cm


@pytest.mark.parametrize("mod", ["Control", "Alt", "Meta"])
def test_space_with_a_modifier_does_not_start_a_pan(page, mod):
    p = page
    _tab(p, "edit")
    p.click('[data-tool="select"]')
    v0 = _view(p)
    x, y = _empty_point(p)
    p.mouse.move(x, y)
    p.keyboard.down(mod)
    p.keyboard.down(" ")
    try:
        _drag(p, x, y, x + 120, y + 40)
    finally:
        p.keyboard.up(" ")
        p.keyboard.up(mod)
    assert _same_view(_view(p), v0), (v0, _view(p))
