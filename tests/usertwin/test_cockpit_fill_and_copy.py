"""Usertwin coverage for docs/COCKPIT_FILL_AND_COPY_SPEC.md, in a real browser
against a twin service.

1. The Fill spacing field (`#fDensity`) shows the pitch the model holds -- after
   the change, after deselect/reselect, after a `design_changed` broadcast and
   after a page reload -- for every kind that rasters a fill: a rectangle drawn
   with the Rectangle tool, a closed pen outline, an electrode, and a FLEX region.
2. Ctrl+C / Ctrl+V copy the selected object through the service's duplicate op:
   the copy has every property of the source, geometry shifted +1/-1 um per
   paste, a new name, and is unwritten. The shortcuts do nothing while focus is
   in a field, with nothing selected, or with an empty clipboard.

Reuses `test_instrument_cockpit.py`'s service/browser/FLEX plumbing.
"""

# ruff: noqa: F811 -- fixtures imported by name shadow their definitions when taken
# as test parameters; the standard pytest cross-file fixture-reuse pattern.

from __future__ import annotations

import re
import time

import pytest
from test_instrument_cockpit import (  # reuse plumbing, see module docstring
    _object_names,
    browser,  # noqa: F401 -- fixture re-export
    flex,  # noqa: F401 -- fixture re-export
    service,  # noqa: F401 -- fixture re-export
)
from test_instrument_cockpit_b3b import _ensure_flex_control

STEP_UM = 1.0
TOL = 1e-3


# ------------------------------------------------------------------ helpers --

@pytest.fixture
def page(service, browser):
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000}, permissions=[])
    p = ctx.new_page()
    _open(p, service)
    yield p
    ctx.close()


def _open(p, service) -> None:
    p.goto(service.url + "/edit-cockpit", wait_until="load", timeout=45000)
    p.wait_for_timeout(800)
    _tab(p, "write")


def _tab(p, name: str) -> None:
    p.click(f'button.tab[data-tab="{name}"]', timeout=5000)
    p.wait_for_timeout(200)


def _design(service) -> list[dict]:
    status, d = service.get("/api/design")
    assert status == 200, (status, d)
    return d["objects"]


def _obj(service, name: str) -> dict:
    found = [o for o in _design(service) if o["name"] == name]
    assert found, f"{name!r} not in the design: {[o['name'] for o in _design(service)]}"
    return found[0]


def _index(service, name: str) -> int:
    return [o["name"] for o in _design(service)].index(name)


def _selected_name(p) -> str | None:
    if p.eval_on_selector("#setFields", "e => e.style.display") == "none":
        return None
    return p.eval_on_selector("#setNameText", "e => e.textContent").rsplit(" (", 1)[0]


def _wait_for_row(p, name: str, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if name in _object_names(p):
            return
        p.wait_for_timeout(150)
    raise AssertionError(f"{name!r} never appeared in the list: {_object_names(p)}")


def _click_row(p, name: str) -> None:
    _wait_for_row(p, name)
    i = _object_names(p).index(name)
    p.click(f'#objList .objrow[data-i="{i}"]')
    p.wait_for_timeout(300)


def _select(p, name: str) -> None:
    if _selected_name(p) != name:
        _click_row(p, name)
    assert _selected_name(p) == name, (name, _selected_name(p))


def _deselect(p, name: str) -> None:
    assert _selected_name(p) == name
    _click_row(p, name)                         # clicking the selected row clears it
    assert _selected_name(p) is None


def _field(p) -> str:
    return p.eval_on_selector("#fDensity", "e => e.value")


def _row_visible(p) -> bool:
    return p.eval_on_selector("#densityRow", "e => getComputedStyle(e).display") != "none"


def _estimate(p) -> str:
    return p.eval_on_selector("#writeEstimate", "e => e.textContent")


def _wait(p, pred, what: str, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if pred():
            return
        p.wait_for_timeout(150)
    raise AssertionError(f"timed out waiting for {what}")


def _log_lines(p) -> list[str]:
    return p.eval_on_selector_all("#log .row", "els => els.map(e => e.textContent)")


def _load(flex, objects: list[dict]) -> None:
    """Replace the whole design over FLEX (the `objects` door)."""
    _ensure_flex_control(flex)
    r = flex.call("loadPattern", {"token": flex.token, "objects": objects})
    assert r is not None and not (isinstance(r, dict) and r.get("error")), r


def _flex_view(flex, name: str) -> dict:
    listing = flex.call("getPattern", {})
    found = [o for o in listing["objects"] if o["id"] == name]
    assert found, f"{name!r} not in getPattern: {[o['id'] for o in listing['objects']]}"
    return found[0]


# ------------------------------------------------------------ object makers --

def _make_rect_tool(p, service, flex) -> str:
    """A rectangle drawn with the Rectangle tool (kind `shape` + a rect bezier)."""
    before = {o["name"] for o in _design(service)}
    _tab(p, "edit")
    p.click('button.tool[data-tool="rect"]')
    box = p.query_selector("#overlay").bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    p.mouse.move(cx - 150, cy - 100)
    p.mouse.down()
    p.mouse.move(cx - 50, cy - 20, steps=5)
    p.mouse.up()
    p.wait_for_timeout(800)
    p.click('button.tool[data-tool="select"]')
    _tab(p, "write")
    new = [o["name"] for o in _design(service) if o["name"] not in before]
    assert len(new) == 1, new
    return new[0]


def _make_pen_shape(p, service, flex) -> str:
    """A closed pen outline (kind `shape`, no rect signature)."""
    before = {o["name"] for o in _design(service)}
    _tab(p, "edit")
    p.click('button.tool[data-tool="pen"]')
    box = p.query_selector("#overlay").bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    pts = [(cx + 50, cy - 100), (cx + 150, cy - 90), (cx + 140, cy), (cx + 60, cy + 10)]
    for x, y in pts:
        p.mouse.click(x, y)
        p.wait_for_timeout(100)
    p.mouse.dblclick(*pts[0])
    p.wait_for_timeout(800)
    p.click('button.tool[data-tool="select"]')
    _tab(p, "write")
    new = [o for o in _design(service) if o["name"] not in before]
    assert len(new) == 1 and new[0]["kind"] == "shape", new
    return new[0]["name"]


def _make_electrode(p, service, flex) -> str:
    status, r = service.post("/api/design/add", {
        "kind": "electrode", "name": "lead",
        "points_um": [[1.0, 1.0], [4.0, 1.0], [3.0, 3.0], [2.0, 3.0], [1.0, 1.0]],
        "voltage": 6.0})
    assert status == 200 and r.get("ok"), r
    return r["name"]


def _make_region(p, service, flex) -> str:
    name = f"pad-{int(time.time() * 1000)}"
    _load(flex, [{"kind": "region", "name": name, "shape": "rect", "x_um": 2.0,
                  "y_um": 2.0, "w_um": 2.0, "h_um": 1.5, "pitch_um": 0.15,
                  "voltage": 6.0}])
    return name


MAKERS = {"rect_tool": _make_rect_tool, "pen_shape": _make_pen_shape,
          "electrode": _make_electrode, "region": _make_region}


# ------------------------------------------------------ 1. the fill field --

@pytest.mark.parametrize("maker", list(MAKERS))
def test_fill_spacing_field_shows_the_model_value(service, flex, page, maker):
    name = MAKERS[maker](page, service, flex)
    _wait_for_row(page, name)
    _select(page, name)
    assert _row_visible(page), f"Fill spacing row hidden for {maker} ({name})"

    idx = _index(service, name)
    points_before = service.get(f"/api/shapes/{idx}/toolpath")[1]["points"]
    for value in (0.05, 0.1):
        est_before = _estimate(page)
        page.fill("#fDensity", str(value))
        page.press("#fDensity", "Tab")
        _wait(page, lambda v=value: _obj(service, name)["fill_spacing_um"] == pytest.approx(v),
              f"the model to take {value}")
        _wait(page, lambda eb=est_before: _estimate(page) != eb,
              f"the write estimate to change after {value}")
        # the set_props broadcast re-adopts the design; the field must survive it
        page.wait_for_timeout(1000)
        assert float(_field(page)) == pytest.approx(value), (
            f"{maker}: typed {value}, field shows {_field(page)!r}, model holds "
            f"{_obj(service, name)['fill_spacing_um']}")
        assert _flex_view(flex, name)["properties"].get("fill", {}).get(
            "spacing_um", value) == pytest.approx(value)
    points_after = service.get(f"/api/shapes/{_index(service, name)}/toolpath")[1]["points"]
    assert points_after != points_before, "the toolpath did not change with the pitch"

    _deselect(page, name)
    _select(page, name)
    assert float(_field(page)) == pytest.approx(0.1), f"after reselect: {_field(page)!r}"

    # a design_changed from another client (a voltage edit over REST)
    status, r = service.post("/api/design/op", {"op": "set_props",
                                                "indices": [_index(service, name)],
                                                "voltage": 5.5})
    assert status == 200 and r.get("ok"), r
    _wait(page, lambda: page.eval_on_selector("#fVoltage", "e => e.value") == "5.5",
          "the broadcast to reach the panel")
    assert float(_field(page)) == pytest.approx(0.1), f"after broadcast: {_field(page)!r}"

    _open(page, service)
    _select(page, name)
    assert float(_field(page)) == pytest.approx(0.1), f"after reload: {_field(page)!r}"


# -------------------------------------------------------- 2. copy / paste --

_COORD_KEYS = ("axis", "axis_um", "points_um", "path_um")


def _shifted(a, b, n: int) -> bool:
    return len(a) == len(b) and all(
        abs(bx - ax - n * STEP_UM) < TOL and abs(by - ay + n * STEP_UM) < TOL
        for (ax, ay), (bx, by) in zip(a, b))


def _bezier_nodes(bez):
    """Every anchor and handle of a serialized BezierPath, in order."""
    out = []
    for nd in (bez or {}).get("nodes", []):
        out.append((nd["x"], nd["y"]))
        out += [tuple(nd[k]) for k in ("h_in", "h_out") if nd.get(k) is not None]
    return out


def _assert_copy(service, flex, src_name: str, copy_name: str, n: int) -> None:
    src, cp = _obj(service, src_name), _obj(service, copy_name)
    assert copy_name != src_name
    for key in ("kind", "voltage", "voltage_end", "v_profile", "speed_um_s",
                "enabled", "fill_spacing_um"):
        assert cp.get(key) == src.get(key), (key, src.get(key), cp.get(key))
    assert _shifted(src["points_um"], cp["points_um"], n), "points not shifted by the offset"
    if src.get("profile"):
        sp = {k: v for k, v in src["profile"].items() if k not in _COORD_KEYS}
        cpp = {k: v for k, v in (cp.get("profile") or {}).items() if k not in _COORD_KEYS}
        assert cpp == sp, "chiral profile differs"
    if src.get("bezier"):
        assert cp["bezier"].get("closed") == src["bezier"].get("closed")
        assert _shifted(_bezier_nodes(src["bezier"]), _bezier_nodes(cp["bezier"]), n)

    fs, fc = _flex_view(flex, src_name), _flex_view(flex, copy_name)
    assert fc["written"] is False
    assert fc["properties"] == fs["properties"], (fs["properties"], fc["properties"])
    assert _shifted(fs["vertices_um"], fc["vertices_um"], n), "vertices not shifted"


def _copy_paste_case(page, service, flex, src_name: str) -> None:
    _wait_for_row(page, src_name)
    _select(page, src_name)
    page.keyboard.press("Control+c")
    names_before = set(_object_names(page))
    for n in (1, 2):
        page.keyboard.press("Control+v")
        _wait(page, lambda k=n: len(set(_object_names(page)) - names_before) == k,
              f"paste {n} to add an object")
        copy_name = _selected_name(page)
        assert copy_name and copy_name not in names_before, (
            f"paste {n}: the copy is not the selection ({copy_name!r})")
        stem = re.sub(r"-\d+$", "", src_name)
        assert copy_name == (f"{stem}-copy" if n == 1 else f"{stem}-copy-{n}"), copy_name
        _assert_copy(service, flex, src_name, copy_name, n)
        assert any(f"pasted {src_name} as {copy_name}" in ln and "enabled" in ln
                   for ln in _log_lines(page)), _log_lines(page)[-5:]


def test_copy_paste_a_chiral_path(service, flex, page):
    name = f"wire-{int(time.time() * 1000)}"
    _load(flex, [{"kind": "chiral", "name": name, "points_um": [[1.0, 1.0], [5.0, 1.5]],
                  "lambda_um": 0.5, "y_amp_um": 0.08, "v0": 4.0, "v_k": 2.0,
                  "phase_deg": 90.0}])
    _copy_paste_case(page, service, flex, name)


def test_copy_paste_a_rect_with_a_non_default_fill(service, flex, page):
    name = _make_rect_tool(page, service, flex)
    status, r = service.post("/api/design/op", {"op": "set_props",
                                                "indices": [_index(service, name)],
                                                "fill_spacing_um": 0.07, "voltage": -3.0})
    assert status == 200 and r.get("ok"), r
    assert _obj(service, name)["fill_spacing_um"] == pytest.approx(0.07)
    _open(page, service)
    _copy_paste_case(page, service, flex, name)


def test_copy_paste_an_electrode(service, flex, page):
    name = _make_electrode(page, service, flex)
    _open(page, service)
    _copy_paste_case(page, service, flex, name)


def test_shortcuts_do_nothing_in_a_field_with_no_selection_or_empty_clipboard(
        service, flex, page):
    name = _make_electrode(page, service, flex)
    _open(page, service)
    count = len(_design(service))

    # empty clipboard: Ctrl+V does nothing
    page.keyboard.press("Control+v")
    # nothing selected: Ctrl+C does nothing, logs nothing
    assert _selected_name(page) is None
    page.keyboard.press("Control+c")
    page.keyboard.press("Control+v")

    # Ctrl+C inside a field copies text, not the object
    _select(page, name)
    page.click("#fVoltage")
    page.keyboard.press("Control+a")
    page.keyboard.press("Control+c")
    page.evaluate("document.activeElement.blur()")
    page.keyboard.press("Control+v")
    page.wait_for_timeout(1200)
    assert len(_design(service)) == count, "a copy made from inside a field"
    assert not any("copied" in ln for ln in _log_lines(page)), _log_lines(page)[-5:]

    # a real copy, then Ctrl+V inside a field pastes text, not the object
    _select(page, name)
    page.keyboard.press("Control+c")
    page.click("#fSpeed")
    page.keyboard.press("Control+v")
    page.wait_for_timeout(1200)
    assert len(_design(service)) == count, "Ctrl+V inside a field pasted an object"
    assert not any("pasted" in ln for ln in _log_lines(page))


def test_ctrl_c_with_page_text_selected_is_the_browser_copy(service, flex, page):
    name = _make_electrode(page, service, flex)
    _open(page, service)
    _select(page, name)
    page.evaluate("""() => {
        window.__copied = 0;
        document.addEventListener('copy', () => { window.__copied += 1; });
        const row = document.querySelector('#log .row');
        const r = document.createRange(); r.selectNodeContents(row);
        const s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
    }""")
    assert page.evaluate("window.getSelection().toString().length") > 0
    page.keyboard.press("Control+c")
    page.wait_for_timeout(300)
    assert page.evaluate("window.__copied") >= 1, "the browser copy event never fired"
    assert not any("copied" in ln for ln in _log_lines(page)), _log_lines(page)[-5:]


def test_paste_is_refused_while_a_write_is_in_progress(service, flex, page):
    name = _make_electrode(page, service, flex)
    _open(page, service)
    _select(page, name)
    page.keyboard.press("Control+c")
    count = len(_design(service))
    # the flag markWriting(true) sets when #perform starts a write
    page.evaluate("document.getElementById('perform').dataset.writing = '1'")
    page.keyboard.press("Control+v")
    page.wait_for_timeout(1200)
    assert len(_design(service)) == count, "pasted during a write"
    assert any("paste refused: a write is in progress" in ln for ln in _log_lines(page))
    page.evaluate("document.getElementById('perform').dataset.writing = '0'")
    page.keyboard.press("Control+v")
    _wait(page, lambda: len(_design(service)) == count + 1, "the paste after the write")


# ------------------------------------------- the node tool leaves the keys --

def _tool(p, tool: str) -> None:
    p.click(f'button.tool[data-tool="{tool}"]')
    p.wait_for_timeout(200)


_COPY_PROBE = """() => {
    window.__copied = 0; window.__prevented = 0;
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key.toLowerCase() === 'c')
            setTimeout(() => { if (e.defaultPrevented) window.__prevented += 1; });
    });
    document.addEventListener('copy', () => { window.__copied += 1; });
}"""


def test_node_tool_ctrl_c_copies_no_object_and_is_not_prevented(service, flex, page):
    name = _make_electrode(page, service, flex)
    _open(page, service)
    _select(page, name)
    _tab(page, "edit")
    _tool(page, "node")
    page.evaluate(_COPY_PROBE)
    page.keyboard.press("Control+c")
    page.wait_for_timeout(300)
    assert page.evaluate("window.__prevented") == 0, "Ctrl+C was preventDefault-ed"
    assert not any("copied" in ln for ln in _log_lines(page)), _log_lines(page)[-5:]


def test_node_tool_ctrl_v_does_not_paste_and_select_tool_still_does(service, flex, page):
    name = _make_electrode(page, service, flex)
    _open(page, service)
    _select(page, name)
    _tab(page, "edit")
    _tool(page, "select")
    page.keyboard.press("Control+c")
    assert any(f"copied {name}" in ln for ln in _log_lines(page))
    count = len(_design(service))

    _tool(page, "node")
    page.keyboard.press("Control+v")
    page.wait_for_timeout(1200)
    assert len(_design(service)) == count, "pasted with the node tool active"
    assert not any("pasted" in ln for ln in _log_lines(page))

    _tool(page, "select")
    page.keyboard.press("Control+v")
    _wait(page, lambda: len(_design(service)) == count + 1,
          "the earlier copy to paste once the select tool is back")


def test_node_tool_ctrl_c_with_log_text_selected_is_the_browser_copy(service, flex, page):
    name = _make_electrode(page, service, flex)
    _open(page, service)
    _select(page, name)
    _tab(page, "edit")
    _tool(page, "node")
    page.evaluate(_COPY_PROBE)
    page.evaluate("""() => {
        const r = document.createRange();
        r.selectNodeContents(document.querySelector('#log .row'));
        const s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
    }""")
    page.keyboard.press("Control+c")
    page.wait_for_timeout(300)
    assert page.evaluate("window.__copied") >= 1, "the browser copy event never fired"
    assert page.evaluate("window.__prevented") == 0
    assert not any("copied" in ln for ln in _log_lines(page))
