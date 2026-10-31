"""Usertwin coverage for the render-pane display choices (docs/UI_LAYOUT_SPEC.md, new
behaviour 6-9): the manual z range on the z bar, the per-write history list behind
`written ▾`, the scan view switch `live | last | both`, and the stale-telemetry
look. Each runs in a real browser against a twin service, at the wide (1896 px) and
the half-screen (948 px) arrangement where the item differs between them.

Scan lines and litho frames are fed through the page's own websocket dispatcher
(`ec-api.js` dispatchWs), so a test controls exactly which frames the page sees.

Reuses `test_instrument_cockpit.py`'s service/browser plumbing.
"""

# ruff: noqa: F811 -- fixtures imported by name shadow their definitions when taken
# as test parameters; the standard pytest cross-file fixture-reuse pattern.

from __future__ import annotations

import json
import re

import pytest
from test_instrument_cockpit import (  # reuse plumbing, see module docstring
    _WS_CAPTURE_INIT,
    browser,  # noqa: F401 -- fixture re-export
    service,  # noqa: F401 -- fixture re-export
)

P = 16          # pixels per synthetic line
R = 16          # rows per synthetic frame


def _open(browser, service, width):
    ctx = browser.new_context(viewport={"width": width, "height": 1036}, permissions=[])
    ctx.add_init_script(_WS_CAPTURE_INIT)
    p = ctx.new_page()
    p.goto(service.url + "/edit-cockpit", wait_until="load", timeout=45000)
    p.wait_for_timeout(800)
    return ctx, p


@pytest.fixture(params=[1896, 948], ids=["wide", "half"])
def page(request, service, browser):
    ctx, p = _open(browser, service, request.param)
    yield p
    ctx.close()


def _settle(p) -> None:
    p.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")


def _line(p, si, base, rows=R):
    """Dispatch one synthetic scan line: height = base + column index (nm)."""
    p.evaluate(
        """async ([si, base, P, R]) => {
             const {dispatchWs} = await import('/static/ec-api.js');
             const h = Array.from({length: P}, (_, i) => base + i);
             dispatchWs({type: 'line', row: si, slow_idx: si, pixels: P, rows: R,
                         size_um: 5, angle_deg: 0, height: h, retrace: h,
                         error: h.map(v => v / 100), error_retrace: h.map(v => v / 100)});
           }""", [si, base, P, rows])


def _frame(p, base, rows=R):
    for si in range(rows):
        _line(p, si, base + si, rows)
    _settle(p)


def _range(p) -> dict:
    return p.evaluate("async () => ({...(await import('/static/ec-state.js')).state.scanRange})")


def _zbar(p) -> dict:
    return p.eval_on_selector(
        "#zScaleCv",
        "c => { const r = c.getBoundingClientRect(); return {left: r.left, top: r.top,"
        " axis: c.dataset.axis, lo: +c.dataset.loPx, hi: +c.dataset.hiPx,"
        " w: r.width, h: r.height}; }")


def _drag_end(p, end: str, frac: float) -> None:
    """Drag one end of the z bar by `frac` of the ramp's length toward its other end."""
    z = _zbar(p)
    a = z[end]
    b = z["lo" if end == "hi" else "hi"]
    to = a + (b - a) * frac
    if z["axis"] == "y":
        x = z["left"] + 8
        p.mouse.move(x, z["top"] + a)
        p.mouse.down()
        for k in range(1, 6):
            p.mouse.move(x, z["top"] + a + (to - a) * k / 5)
    else:
        y = z["top"] + 8
        p.mouse.move(z["left"] + a, y)
        p.mouse.down()
        for k in range(1, 6):
            p.mouse.move(z["left"] + a + (to - a) * k / 5, y)
    p.mouse.up()
    _settle(p)


# ------------------------------------------------------------ manual z range --

def test_z_bar_ends_set_the_grey_range_by_hand_and_auto_returns(page):
    p = page
    _frame(p, 0)
    auto = _range(p)
    assert auto["manual"] is False and auto["hi"] > auto["lo"], auto
    assert p.eval_on_selector("#zAuto", "b => !b.hidden && b.classList.contains('active')")
    span = auto["hi"] - auto["lo"]

    _drag_end(p, "hi", 0.25)
    r = _range(p)
    assert r["manual"] is True, r
    assert r["lo"] == pytest.approx(auto["lo"])
    assert r["hi"] == pytest.approx(auto["hi"] - 0.25 * span, rel=0.05), (auto, r)
    _drag_end(p, "lo", 0.2)
    r2 = _range(p)
    assert r2["lo"] > auto["lo"] and r2["hi"] == pytest.approx(r["hi"]), (r, r2)
    assert not p.eval_on_selector("#zAuto", "b => b.classList.contains('active')")
    p.screenshot(path=f"tmp-shots/7b-z-manual-{p.viewport_size['width']}.png")

    # a new line on the same channel keeps the hand-set range
    _line(p, 3, 50)
    _settle(p)
    assert _range(p)["hi"] == pytest.approx(r2["hi"]) and _range(p)["manual"] is True

    p.click("#zAuto")
    _settle(p)
    back = _range(p)
    assert back["manual"] is False
    assert back["hi"] != pytest.approx(r2["hi"])


def test_changing_the_channel_returns_the_z_range_to_auto(page):
    p = page
    _frame(p, 0)
    _drag_end(p, "hi", 0.3)
    assert _range(p)["manual"] is True
    p.select_option("#scanChan", "error")
    _settle(p)
    r = _range(p)
    assert r["manual"] is False and r["chan"] == "error", r
    # and back: the height channel starts at auto too, not at the old hand-set range
    p.select_option("#scanChan", "height")
    _settle(p)
    assert _range(p)["manual"] is False


# ---------------------------------------------------- per-write history list --

def _litho(p, frames):
    p.evaluate("""async (fs) => { const {dispatchWs} = await import('/static/ec-api.js');
                                  for (const f of fs) dispatchWs(f); }""", frames)
    _settle(p)


def _stroke(name, obj, x, v=3.0, partial=False):
    return {"type": "litho_object_partial" if partial else "litho_object_done",
            "name": name, "object": obj, "voltage": v,
            "path_um": [[x, 1.0], [x + 1.0, 2.0]], "v_written": [v, v]}


def _shown_strokes(p) -> list[str]:
    """The object of every written-trace entry the overlay currently draws."""
    return p.evaluate(
        """async () => { const {state} = await import('/static/ec-state.js');
             const wr = await import('/static/ec-written.js');
             return state.litho.written.filter(wr.strokeShown).map(e =>
               state.litho.writes.find(w => w.id === e.write).object); }""")


def _rows(p) -> list[dict]:
    return p.eval_on_selector_all(
        "#writtenList .wrow",
        "rs => rs.map(r => ({name: r.querySelector('.wname').textContent,"
        " meta: r.querySelector('em').textContent,"
        " on: r.querySelector('input').checked}))")


def test_written_toggle_and_per_write_history_list(page):
    p = page
    start = {"type": "litho_object_start", "name": "box#0", "object": "box", "voltage": 3.0,
             "duration_s": 1, "length_um": 1}
    _litho(p, [start, _stroke("box#0", "box", 1.0), _stroke("box#1", "box", 2.0),
               {**start, "name": "wire", "object": "wire", "voltage": -1.5},
               _stroke("wire", "wire", 4.0, v=-1.5),
               {"type": "litho_done", "aborted": False},
               # a second run writes the box again: a write of its own
               start, _stroke("box#0", "box", 1.5, partial=True),
               {"type": "litho_done", "aborted": True}])
    assert sorted(_shown_strokes(p)) == ["box", "box", "box", "wire"]

    p.click("#writtenMenu")
    assert p.is_visible("#writtenList")
    rows = _rows(p)
    # newest first; the cut stroke marks its write partial
    assert [r["name"] for r in rows] == ["box (partial)", "wire", "box"], rows
    assert all(r["on"] for r in rows)
    assert rows[1]["meta"].endswith("-1.5 V") and rows[2]["meta"].endswith("+3.0 V"), rows
    p.screenshot(path=f"tmp-shots/7b-written-list-{p.viewport_size['width']}.png")

    # one write hidden: only its strokes leave the overlay
    p.locator("#writtenList .wrow").nth(2).locator("input").uncheck()
    _settle(p)
    assert sorted(_shown_strokes(p)) == ["box", "wire"]
    # the global box hides the whole written trace, and brings back what was shown
    p.keyboard.press("Escape")
    assert not p.is_visible("#writtenList")
    p.uncheck("#showWritten")
    _settle(p)
    assert _shown_strokes(p) == []
    p.check("#showWritten")
    _settle(p)
    assert sorted(_shown_strokes(p)) == ["box", "wire"]

    # Clear written trace empties the list too
    p.click("#clearTrace")
    p.click("#writtenMenu")
    assert _rows(p) == [] and "no writes yet" in p.inner_text("#writtenList")
    # a press elsewhere closes it
    p.mouse.click(5, 5)
    assert not p.is_visible("#writtenList")


# ------------------------------------------------------- scan view live|last|both --

def _img_rows(p) -> list[int]:
    """Mean grey of each row of the offscreen scan image, top row first."""
    return p.evaluate(
        """async () => { const {state} = await import('/static/ec-state.js');
             const c = state.scanImg; if (!c) return null;
             const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
             const out = [];
             for (let r = 0; r < c.height; r++) {
               let t = 0;
               for (let i = 0; i < c.width; i++) t += d[(r * c.width + i) * 4];
               out.push(Math.round(t / c.width));
             }
             return out; }""")


def _view_btn(p, v):
    p.click(f'#scanViewSeg .seg-btn[data-view="{v}"]')
    _settle(p)


def test_scan_view_switch_shows_live_last_or_both(page):
    p = page
    assert p.get_attribute('#scanViewSeg [data-view="both"]', "aria-pressed") == "true"
    _frame(p, 0)                               # frame A: 0..30 nm
    p.evaluate("async () => (await import('/static/ec-api.js')).dispatchWs("
               "{type: 'scan_done', aborted: false})")
    for si in range(4):                        # frame B starts: four rows at 100+ nm
        _line(p, si, 100 + si)
    _settle(p)

    both = _range(p)
    assert both["lo"] < 10 and both["hi"] > 100, both
    rows = _img_rows(p)
    # bottom four rows (image rows R-4..R-1) are frame B, the rest still frame A
    assert min(rows[-4:]) > max(rows[:-4]), rows

    _view_btn(p, "live")
    live = _range(p)
    assert live["lo"] >= 100, live
    rows = _img_rows(p)
    assert all(v == 0 for v in rows[:-4]) and min(rows[-4:]) > 0, rows
    p.screenshot(path=f"tmp-shots/7b-scan-live-{p.viewport_size['width']}.png")

    _view_btn(p, "last")
    last = _range(p)
    assert last["hi"] < 40, last
    assert p.get_attribute('#scanViewSeg [data-view="last"]', "aria-pressed") == "true"
    assert p.get_attribute('#scanViewSeg [data-view="both"]', "aria-pressed") == "false"
    # a line arriving while "last" is shown leaves the previous frame in view
    _line(p, 4, 104)
    _settle(p)
    assert _range(p)["hi"] == pytest.approx(last["hi"])
    p.screenshot(path=f"tmp-shots/7b-scan-last-{p.viewport_size['width']}.png")

    _view_btn(p, "both")
    assert _range(p)["hi"] > 100


# --------------------------------------------------------------- stale telemetry --

@pytest.fixture(params=[1896, 948], ids=["wide", "half"])
def routed(request, service, browser):
    """A page whose /ws frames pass through the test, which can freeze hw_telemetry.t
    (a stalled hardware poll keeps sending the same frame time)."""
    ctl = {"freeze": None}

    def rewrite(msg):
        if ctl["freeze"] is None or not isinstance(msg, str) or '"telemetry"' not in msg:
            return msg
        m = json.loads(msg)
        if m.get("type") == "telemetry" and isinstance(m.get("hw_telemetry"), dict):
            m["hw_telemetry"]["t"] = ctl["freeze"]
            return json.dumps(m)
        return msg

    def route(ws):
        server = ws.connect_to_server()
        ws.on_message(lambda m: server.send(m))
        server.on_message(lambda m: ws.send(rewrite(m)))

    ctx = browser.new_context(viewport={"width": request.param, "height": 1036}, permissions=[])
    ctx.add_init_script(_WS_CAPTURE_INIT)
    ctx.route_web_socket(re.compile(r".*/ws$"), route)
    p = ctx.new_page()
    p.goto(service.url + "/edit-cockpit", wait_until="load", timeout=45000)
    p.wait_for_function("() => document.getElementById('ecTelemetry')"
                        ".classList.contains('has-data')", timeout=15000)
    yield p, ctl
    ctx.close()


def _stale(p) -> tuple[bool, str, str]:
    return tuple(p.evaluate(
        "() => { const e = document.getElementById('ecTelemetry'),"
        " t = document.getElementById('telemStale');"
        " return [e.classList.contains('stale'), t.hidden ? '' : t.textContent,"
        " getComputedStyle(e.querySelector('.tnum')).color]; }"))


def test_telemetry_greys_out_and_shows_its_age_when_not_live(routed):
    p, ctl = routed
    p.wait_for_function("() => !document.getElementById('ecTelemetry')"
                        ".classList.contains('stale')", timeout=10000)
    live_colour = _stale(p)[2]

    # the frame time stops advancing: stale after TELEMETRY_STALE_S (5 s), with its age
    ctl["freeze"] = 12345.0
    p.wait_for_function("() => document.getElementById('ecTelemetry')"
                        ".classList.contains('stale')", timeout=10000)
    stale, text, colour = _stale(p)
    age = float(re.fullmatch(r"stale (\d+\.\d) s", text).group(1))
    assert age >= 5.0, text
    assert colour != live_colour, (colour, live_colour)
    p.wait_for_timeout(1200)
    assert float(_stale(p)[1].split()[1]) > age        # the age keeps counting
    p.screenshot(path=f"tmp-shots/7b-telemetry-stale-{p.viewport_size['width']}.png")
    ctl["freeze"] = None
    p.wait_for_function("() => !document.getElementById('ecTelemetry')"
                        ".classList.contains('stale')", timeout=10000)
    assert _stale(p)[1] == ""

    # the websocket closes: stale at once, live again once it reconnects
    p.evaluate("() => window.__wsInstances.forEach(w => w.close())")
    p.wait_for_function("() => document.getElementById('ecTelemetry')"
                        ".classList.contains('stale')", timeout=1500)
    assert _stale(p)[1].startswith("stale")
    p.wait_for_function("() => !document.getElementById('ecTelemetry')"
                        ".classList.contains('stale')", timeout=10000)


# ------------------------------------------------------- mode dropdown, unread --

def _mode_look(p) -> dict:
    return p.evaluate(
        "() => { const m = document.getElementById('mode'),"
        " t = document.getElementById('modeUnread'), cs = getComputedStyle(m);"
        " return {tag: !t.hidden && t.offsetWidth > 0, outline: cs.outlineStyle,"
        " inline: m.style.outline, title: m.title}; }")


def test_unread_mode_shows_a_tag_not_a_dashed_outline(page):
    p = page
    # the service has read no mode yet (as on a fresh twin, whatever an earlier test set)
    p.route("**/api/mode", lambda r: r.fulfill(json={"mode": None})
            if r.request.method == "GET" else r.fallback())
    p.reload(wait_until="load")
    p.wait_for_timeout(800)
    look = _mode_look(p)
    assert look["tag"] is True, look
    assert look["outline"] != "dashed" and look["inline"] == "", look
    assert "unknown until read" in look["title"], look
    p.screenshot(path=f"tmp-shots/7b-mode-unread-{p.viewport_size['width']}.png",
                 clip={"x": 0, "y": 90, "width": min(940, p.viewport_size["width"]),
                       "height": 50})
    # a mode applied from the dropdown is known: the tag goes
    try:
        p.select_option("#mode", "ac")
        p.wait_for_function("() => document.getElementById('modeUnread').hidden",
                            timeout=8000)
        assert _mode_look(p)["title"] == ""
    finally:
        p.select_option("#mode", "contact")
        p.wait_for_timeout(500)


# ------------------------------------------------------------ scan-line marker --

def _marker(p) -> dict:
    """The marked row, and the image rows (top first) whose centre pixel is the marker
    colour."""
    return p.evaluate(
        """() => { const c = document.getElementById('scan');
             const css = getComputedStyle(document.documentElement)
               .getPropertyValue('--scanline').trim();
             const want = [1, 3, 5].map(i => parseInt(css.slice(i, i + 2), 16));
             const d = c.getContext('2d').getImageData(Math.floor(c.width / 2), 0, 1,
                                                       c.height).data;
             const ys = [];
             for (let y = 0; y < c.height; y++)
               if (want.every((v, k) => Math.abs(d[y * 4 + k] - v) < 32)) ys.push(y);
             return {row: c.dataset.scanRow ?? null, ys}; }""")


def _fault_frame(p, rec):
    # through the page's own sockets, as the service's `fault` broadcast arrives
    p.evaluate("(rec) => window.__wsInstances.forEach(w => w.onmessage && w.onmessage("
               "{data: JSON.stringify({type: 'fault', fault: rec})}))", rec)
    _settle(p)


def test_scan_line_marker_shows_the_row_being_scanned_only_while_scanning(page):
    p = page
    assert _marker(p) == {"row": None, "ys": []}           # idle: no marker
    for si in range(6):
        _line(p, si, si)
    _settle(p)
    m = _marker(p)
    assert m["row"] == "5" and m["ys"], m
    p.screenshot(path=f"tmp-shots/7b-scanline-{p.viewport_size['width']}.png")
    y5 = sum(m["ys"]) / len(m["ys"])
    _line(p, 6, 6)
    _settle(p)
    m6 = _marker(p)
    # slow_idx counts up from the bottom row, so the marker moves up the image
    assert m6["row"] == "6" and sum(m6["ys"]) / len(m6["ys"]) < y5, (m, m6)

    # hidden on the last view, and while a fault is latched
    _view_btn(p, "last")
    assert _marker(p)["row"] is None
    _view_btn(p, "both")
    _fault_frame(p, {"reason": "test latch", "at": 0})
    assert _marker(p) == {"row": None, "ys": []}
    _fault_frame(p, None)
    assert _marker(p)["row"] == "6"

    # scan_done: idle again
    p.evaluate("async () => (await import('/static/ec-api.js')).dispatchWs("
               "{type: 'scan_done', aborted: false})")
    _settle(p)
    assert _marker(p) == {"row": None, "ys": []}


def test_scan_line_marker_clears_when_the_websocket_closes(page):
    p = page
    for si in range(3):
        _line(p, si, si)
    _settle(p)
    assert _marker(p)["row"] == "2"
    # the link drops before any scan_done: the marker must not keep claiming a scan
    p.evaluate("() => window.__wsInstances.forEach(w => w.close())")
    p.wait_for_function("() => document.getElementById('scan').dataset.scanRow === undefined",
                        timeout=3000)
    assert _marker(p) == {"row": None, "ys": []}


def _done(p, aborted):
    p.evaluate("async (a) => (await import('/static/ec-api.js')).dispatchWs("
               "{type: 'scan_done', aborted: a})", aborted)


def test_last_keeps_the_most_recent_scanned_grid_even_partial_or_aborted(page):
    p = page
    _frame(p, 0)                               # frame A: 0..30 nm, finished
    _done(p, False)
    for si in range(4):                        # frame B: four rows at 100+, aborted
        _line(p, si, 100 + si)
    _done(p, True)
    _line(p, 0, 200)                           # frame C starts: one row at 200+
    _settle(p)
    _view_btn(p, "last")
    last = _range(p)
    # "last" is B as it ended: its four rows over A, without C's row
    assert 100 <= last["hi"] < 200, last
    rows = _img_rows(p)
    assert min(rows[-4:]) > max(rows[:-4]), rows
    # C never gets a scan_done; the next frame's start keeps it as "last"
    p.evaluate("async () => (await import('/static/ec-api.js')).dispatchWs({type: "
               f"'scan_reset', pixels: {P}, rows: {R}, size_um: 5, angle_deg: 0}})")
    _settle(p)
    assert _range(p)["hi"] >= 200, _range(p)


def test_unticking_written_keeps_the_stroke_being_written(page):
    p = page
    pt = [2.0, 2.0]
    _litho(p, [{"type": "litho_object_start", "name": "w1", "object": "w1", "voltage": 3.0,
                "duration_s": 1, "length_um": 1},
               {"type": "litho_progress", "name": "w1", "frac": 0.5,
                "x_um": pt[0], "y_um": pt[1], "trail_um": [pt]}])
    p.uncheck("#showWritten")
    _settle(p)
    alpha = p.evaluate(
        """async ([x, y]) => { const {state} = await import('/static/ec-state.js');
             const g = await import('/static/ec-geom.js');
             const c = document.getElementById('overlay');
             const [px, py] = g.fitTransform(state.view || state.SCAN_FRAME,
                                             c.width, c.height)(x, y);
             return c.getContext('2d').getImageData(Math.round(px), Math.round(py), 1, 1)
               .data[3]; }""", pt)
    assert alpha > 0, "the live point cloud was hidden with the written trace"
    assert p.evaluate("async () => (await import('/static/ec-state.js'))"
                      ".state.litho.live.length") == 1
    _litho(p, [{"type": "litho_done", "aborted": False}])


# ------------------------------------------------------- telemetry does not flicker --

def test_telemetry_arrangement_is_steady_over_live_frames(page):
    """Over about 2 s of live twin frames the readout keeps one arrangement: the same
    classes, bars, history graphs and height on every sample."""
    p = page
    p.wait_for_function("() => document.getElementById('ecTelemetry')"
                        ".classList.contains('has-data')", timeout=10000)
    p.wait_for_timeout(500)
    js = ("() => { const t = document.getElementById('ecTelemetry');"
          " return [t.className, t.querySelectorAll('.tch.has-bar').length,"
          " t.querySelectorAll('.tch.has-spark').length, t.offsetHeight].join('|'); }")
    seen = set()
    for _ in range(80):
        seen.add(p.evaluate(js))
        p.wait_for_timeout(25)
    assert len(seen) == 1, seen
    # one readout, one safety banner: no module runs twice
    assert p.evaluate("() => document.querySelectorAll('#ecSafetyBanner').length") == 1
    loads = p.evaluate("() => performance.getEntriesByType('resource')"
                       ".map(e => e.name).filter(n => n.includes('/ec-telemetry.js'))")
    assert len(loads) == 1, loads


# ------------------------------------------- log lines never resize the render --

def test_folded_log_lines_never_change_the_canvas_height(page):
    """Long and short lines alternate in the folded log's one-line view (as on a node
    drag's release): the log bar and the overlay canvas keep their heights, so the
    image never refits."""
    p = page
    if not p.eval_on_selector(".logpanel", "e => e.classList.contains('collapsed')"):
        p.click("#logToggle")
    _settle(p)
    js = ("() => [document.getElementById('overlay').getBoundingClientRect().height,"
          " document.querySelector('.logbar').getBoundingClientRect().height]")
    first = p.evaluate(js)
    long = "x" * 3000
    seen = {tuple(first)}
    for k in range(16):
        p.evaluate("async (t) => (await import('/static/ec-log.js')).logLine(t, 'info')",
                   long if k % 2 else "ok")
        _settle(p)
        seen.add(tuple(p.evaluate(js)))
    assert seen == {tuple(first)}, seen
    # the long line is cut with an ellipsis on one line, not wrapped
    assert p.eval_on_selector("#logLatest", "e => e.scrollWidth > e.clientWidth")
    for sel in ("#counts", ".logbar label"):
        assert p.eval_on_selector(sel, "e => e.getBoundingClientRect().height") < 20, sel
