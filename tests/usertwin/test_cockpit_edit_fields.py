"""Usertwin coverage for the Edit-tab readouts and fields (docs/UI_LAYOUT_SPEC.md,
"New behaviour" 1-4), in a real browser against a twin service, at a full
screen (1896 x 1036) and half screen (948 x 1036):

1. the coordinate bar shows the pointer in lab µm, "—" off the canvas;
2. Select and Node options show the selected object's length (and, for Node, its
   node count and whether it is closed);
3. the selected node's x/y fields move it exactly as dragging it there does;
4. the Select options' x/y/w/h, rotate and skew fields, and the copy / paste /
   duplicate buttons.

Every field edit is checked against the design read back from the service.
Reuses `test_instrument_cockpit.py`'s service/browser plumbing.
"""

# ruff: noqa: F811 -- fixtures imported by name shadow their definitions when taken
# as test parameters; the standard pytest cross-file fixture-reuse pattern.

from __future__ import annotations

import math
import re
import time

import numpy as np
import pytest
from test_instrument_cockpit import (  # reuse plumbing, see module docstring
    browser,  # noqa: F401 -- fixture re-export
    service,  # noqa: F401 -- fixture re-export
)

VIEWPORTS = {"full": (1896, 1036), "half": (948, 1036)}
# the design GET serves coordinates to 1e-4 µm, so a value typed from one served
# coordinate lands within that of the dragged object's
TOL_UM = 2e-4


def _cusp(x, y, h_in=None, h_out=None, kind="cusp"):
    return {"x": x, "y": y, "h_in": h_in, "h_out": h_out, "kind": kind}


# the three object kinds every field is checked on, as /api/design/add bodies
KINDS = {
    "path": {"kind": "path", "bezier": {"closed": False, "nodes": [
        _cusp(3.0, 10.0, h_out=[4.0, 12.0]), _cusp(7.0, 10.0, h_in=[6.0, 12.0]),
        _cusp(10.0, 11.0)]}},
    "rect": {"kind": "shape", "bezier": {"closed": True, "nodes": [
        _cusp(12.0, 3.0), _cusp(16.0, 3.0), _cusp(16.0, 6.0), _cusp(12.0, 6.0)]}},
    # a flat base and two curved sides to an apex: the service files it as a funnel
    "funnel": {"kind": "shape", "bezier": {"closed": True, "nodes": [
        _cusp(4.0, 4.0, h_in=[4.6, 6.0]), _cusp(8.0, 4.0, h_out=[7.4, 6.0]),
        _cusp(6.0, 8.0, h_in=[6.8, 7.0], h_out=[5.2, 7.0])]}},
}


# ------------------------------------------------------------------ helpers --

@pytest.fixture(params=list(VIEWPORTS))
def page(request, service, browser):
    w, h = VIEWPORTS[request.param]
    ctx = browser.new_context(viewport={"width": w, "height": h}, permissions=[])
    p = ctx.new_page()
    p.goto(service.url + "/edit-cockpit", wait_until="load", timeout=45000)
    p.wait_for_timeout(800)
    yield p
    ctx.close()


def _tab(p, name: str) -> None:
    p.click(f'button.tab[data-tab="{name}"]', timeout=5000)
    p.wait_for_timeout(200)


def _wait(pred, what: str, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if pred():
            return
        time.sleep(0.15)
    raise AssertionError(f"timed out waiting for {what}")


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


def _canvas_box(p) -> dict:
    return p.query_selector("#overlay").bounding_box()


def _design(service) -> list[dict]:
    status, d = service.get("/api/design")
    assert status == 200, (status, d)
    return d["objects"]


def _obj(service, name: str) -> dict:
    found = [o for o in _design(service) if o["name"] == name]
    assert found, f"{name!r} not in the design"
    return found[0]


def _add(service, kind: str) -> str:
    body = dict(KINDS[kind], voltage=6.0, speed_um_s=2.0)
    status, r = service.post("/api/design/add", body)
    assert status == 200 and r.get("ok"), r
    return r["name"]


def _names(p) -> list[str]:
    return p.evaluate("async () => (await import('/static/ec-state.js'))"
                      ".state.objects.map(o => o.name)")


def _select_row(p, name: str) -> None:
    _wait(lambda: name in _names(p), f"{name} in the page")
    sel = p.evaluate("async () => { const {state} = await import('/static/ec-state.js');"
                     " return state.selected >= 0 ? state.objects[state.selected].name"
                     " : null; }")
    if sel != name:
        p.click(f'#objList .objrow[data-i="{_names(p).index(name)}"]')
        p.wait_for_timeout(300)


def _tool(p, tool: str) -> None:
    p.click(f'button.tool[data-tool="{tool}"]')
    p.wait_for_timeout(300)


def _text(p, sel: str) -> str:
    return p.eval_on_selector(sel, "e => e.textContent")


def _bez_length(bez: dict) -> float:
    from afm_litho.litho.bezier import BezierPath
    pts = np.asarray(BezierPath.from_dict(bez).flatten(0.004))
    return float(np.hypot(*np.diff(pts, axis=0).T).sum())


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


def _frame(p, names: list[str]) -> None:
    """Zoom the view onto the named objects so their nodes are far apart on screen."""
    p.evaluate(
        """async (names) => {
             const {state} = await import('/static/ec-state.js');
             const g = await import('/static/ec-geom.js');
             const b = g.boundsOfObjects(state.objects.filter(o => names.includes(o.name)));
             const m = Math.max(b.w, b.h) * 0.3 + 0.5;
             state.view = {x0: b.x0 - m, y0: b.y0 - m, w: b.w + 2 * m, h: b.h + 2 * m};
           }""", names)
    p.wait_for_timeout(100)


def _pick_node(p, name: str, k: int) -> None:
    """Node tool: click node k of the named object (selecting the object first)."""
    _select_row(p, name)
    _tool(p, "node")
    n = _obj_bezier_node(p, name, k)
    p.mouse.click(*_um_to_client(p, n["x"], n["y"]))
    p.wait_for_timeout(300)


def _obj_bezier_node(p, name: str, k: int) -> dict:
    return p.evaluate(
        """async ([name, k]) => {
             const {state} = await import('/static/ec-state.js');
             const o = state.objects.find(q => q.name === name);
             return o.bezier.nodes[k];
           }""", [name, k])


def _geometry(o: dict) -> dict:
    return {"kind": o["kind"], "closed": o["bezier"]["closed"],
            "nodes": o["bezier"]["nodes"], "points_um": o["points_um"]}


def _assert_same_geometry(a: dict, b: dict, offset: tuple[float, float]) -> None:
    """a == b shifted by offset: the field-edited copy against the dragged one."""
    ga, gb = _geometry(a), _geometry(b)
    assert ga["kind"] == gb["kind"] and ga["closed"] == gb["closed"]
    assert len(ga["nodes"]) == len(gb["nodes"])
    for na, nb in zip(ga["nodes"], gb["nodes"], strict=True):
        assert na["kind"] == nb["kind"]
        assert na["x"] == pytest.approx(nb["x"] + offset[0], abs=TOL_UM)
        assert na["y"] == pytest.approx(nb["y"] + offset[1], abs=TOL_UM)
        for h in ("h_in", "h_out"):
            assert (na[h] is None) == (nb[h] is None)
            if na[h] is not None:
                assert na[h][0] == pytest.approx(nb[h][0] + offset[0], abs=TOL_UM)
                assert na[h][1] == pytest.approx(nb[h][1] + offset[1], abs=TOL_UM)
    pa, pb = np.asarray(ga["points_um"]), np.asarray(gb["points_um"])
    assert pa.shape == pb.shape
    assert np.allclose(pa, pb + np.asarray(offset), atol=TOL_UM)


def _add_at(service, kind: str, dx: float) -> str:
    """The kind's object shifted by dx in x (a twin to compare a drag against)."""
    import copy
    body = copy.deepcopy(KINDS[kind])
    for n in body["bezier"]["nodes"]:
        n["x"] += dx
        for h in ("h_in", "h_out"):
            if n[h]:
                n[h][0] += dx
    status, r = service.post("/api/design/add", dict(body, voltage=6.0, speed_um_s=2.0))
    assert status == 200 and r.get("ok"), r
    return r["name"]


def _um(text: str) -> float:
    m = re.fullmatch(r"(-?[\d.]+) µm", text)
    assert m, text
    return float(m.group(1))


# ------------------------------------------------------- 1. cursor readout --

def _cursor_text(p) -> str:
    return p.eval_on_selector("#cursorPos", "e => e.textContent")


@pytest.mark.parametrize("tab", ["scan", "edit"])
def test_cursor_readout_follows_the_pointer_in_lab_um(page, tab):
    _tab(page, tab)
    box = _canvas_box(page)
    for fx, fy in ((0.5, 0.5), (0.2, 0.7), (0.8, 0.25)):
        cx, cy = box["x"] + box["width"] * fx, box["y"] + box["height"] * fy
        page.mouse.move(cx, cy)
        page.wait_for_timeout(50)
        m = re.fullmatch(r"cursor x (-?[\d.]+)\s+y (-?[\d.]+) µm", _cursor_text(page))
        assert m, _cursor_text(page)
        ex, ey = _client_to_um(page, cx, cy)
        assert float(m.group(1)) == pytest.approx(ex, abs=2e-3)
        assert float(m.group(2)) == pytest.approx(ey, abs=2e-3)
    page.mouse.move(5, 5)                       # the header: off the canvas
    page.wait_for_timeout(50)
    assert _cursor_text(page) == "cursor —"


# ---------------------------------------------------------- 2. path length --

@pytest.mark.parametrize("kind", list(KINDS))
def test_options_show_the_selected_objects_length(service, page, kind):
    name = _add(service, kind)
    _tab(page, "edit")
    _tool(page, "select")
    assert _text(page, "#selLength") == "—"
    _select_row(page, name)
    o = _obj(service, name)
    if kind == "funnel":
        assert o["kind"] == "electrode", o["kind"]
    want = _bez_length(o["bezier"])
    assert _um(_text(page, "#selLength")) == pytest.approx(want, abs=2e-3)
    assert page.is_visible("#selLength")

    _tool(page, "node")
    _wait(lambda: _text(page, "#nodeLength") != "—", "the node length")
    assert _um(_text(page, "#nodeLength")) == pytest.approx(want, abs=2e-3)
    if kind == "rect":                   # rectangles are not node-editable
        assert _text(page, "#nodeCount") == "—"
    else:
        _wait(lambda: _text(page, "#nodeCount") != "—", "the node count")
        assert _text(page, "#nodeCount") == str(len(o["bezier"]["nodes"]))
        assert _text(page, "#nodeClosed") == ("yes" if o["bezier"]["closed"] else "no")
    assert math.isfinite(want) and want > 0


# ------------------------------------------------------- 3. node x/y fields --

def _field_value(p, sel: str) -> str:
    return p.eval_on_selector(sel, "e => e.value")


def _disabled(p, sel: str) -> bool:
    return p.eval_on_selector(sel, "e => e.disabled")


@pytest.mark.parametrize("kind", ["path", "funnel"])
def test_node_fields_move_the_node_as_a_drag_does(service, page, kind):
    shift = 20.0
    dragged, typed = _add_at(service, kind, 0.0), _add_at(service, kind, shift)
    _tab(page, "edit")
    _frame(page, [dragged, typed])
    k = 1

    # the drag: node k of `dragged` pulled up and right
    _pick_node(page, dragged, k)
    assert not _disabled(page, "#nodeX")
    n0 = _obj_bezier_node(page, dragged, k)
    assert float(_field_value(page, "#nodeX")) == pytest.approx(n0["x"], abs=1e-3)
    assert float(_field_value(page, "#nodeY")) == pytest.approx(n0["y"], abs=1e-3)
    sx, sy = _um_to_client(page, n0["x"], n0["y"])
    page.mouse.move(sx, sy)
    page.mouse.down()
    page.mouse.move(sx + 30, sy - 20, steps=6)
    page.mouse.up()
    _wait(lambda: _obj(service, dragged)["bezier"]["nodes"][k]["x"] != n0["x"], "the drag")
    page.wait_for_timeout(300)
    moved = _obj(service, dragged)["bezier"]["nodes"][k]
    # the fields follow the drag
    assert float(_field_value(page, "#nodeX")) == pytest.approx(moved["x"], abs=1e-3)

    # the field edit: the same node of the twin, typed to the dragged position
    _pick_node(page, typed, k)
    posts = []
    page.on("request", lambda r: posts.append(r.url) if r.method == "POST"
            and r.url.endswith("/api/design/bezier") else None)
    page.fill("#nodeX", repr(moved["x"] + shift))
    page.press("#nodeX", "Enter")
    page.fill("#nodeY", repr(moved["y"]))
    page.press("#nodeY", "Tab")                # leaving the field applies it too
    _wait(lambda: _obj(service, typed)["bezier"]["nodes"][k]["y"] == pytest.approx(
        moved["y"], abs=1e-9), "the typed y")
    page.wait_for_timeout(300)
    _assert_same_geometry(_obj(service, typed), _obj(service, dragged), (shift, 0.0))
    assert len(posts) == 2, f"one persist per field edit, as one per drag: {posts}"

    # deselecting the node disables the fields
    box = _canvas_box(page)
    page.mouse.click(box["x"] + 3, box["y"] + 3)
    page.wait_for_timeout(300)
    assert _disabled(page, "#nodeX") and _disabled(page, "#nodeY")
    assert _field_value(page, "#nodeX") == ""


def test_node_fields_stay_disabled_on_a_rectangle(service, page):
    name = _add(service, "rect")
    _tab(page, "edit")
    _select_row(page, name)
    _tool(page, "node")
    assert _disabled(page, "#nodeX") and _disabled(page, "#nodeY")


def test_node_fields_stack_in_half_screen(service, page):
    _tab(page, "edit")
    _tool(page, "node")
    bx = page.query_selector("#nodeX").bounding_box()
    by = page.query_selector("#nodeY").bounding_box()
    half = page.viewport_size["width"] < 1200
    if half:
        assert by["y"] > bx["y"] + bx["height"] - 1, "y sits under x in half screen"
    else:
        assert abs(by["y"] - bx["y"]) < 2, "x and y share a line in full screen"


# ------------------------------------------------- 4. select transform fields --

def _bbox(o: dict) -> tuple[float, float, float, float]:
    pts = np.asarray(o["points_um"])
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def _type(p, sel: str, value: float, key: str = "Enter") -> None:
    p.fill(sel, repr(value))
    p.press(sel, key)


def _bezier_affine(bez: dict, f) -> list[dict]:
    out = []
    for n in bez["nodes"]:
        m = dict(n)
        m["x"], m["y"] = f(n["x"], n["y"])
        for h in ("h_in", "h_out"):
            if n[h] is not None:
                m[h] = list(f(*n[h]))
        out.append(m)
    return out


def _assert_nodes(nodes: list[dict], want: list[dict]) -> None:
    for a, b in zip(nodes, want, strict=True):
        assert a["x"] == pytest.approx(b["x"], abs=TOL_UM)
        assert a["y"] == pytest.approx(b["y"], abs=TOL_UM)
        for h in ("h_in", "h_out"):
            assert (a[h] is None) == (b[h] is None)
            if a[h] is not None:
                assert a[h] == pytest.approx(b[h], abs=TOL_UM)


def _changed(service, name: str, before: dict):
    return lambda: _obj(service, name)["points_um"] != before["points_um"]


def _body_point(o: dict) -> tuple[float, float]:
    """A point on the object away from its box handles: the box centre of a filled
    outline, a third of the way along a wire."""
    if o["kind"] == "path":
        q = o["points_um"][len(o["points_um"]) // 3]
        return q[0], q[1]
    x0, y0, x1, y1 = _bbox(o)
    return (x0 + x1) / 2, (y0 + y1) / 2


@pytest.mark.parametrize("kind", list(KINDS))
def test_select_fields_move_and_size_as_the_handles_do(service, page, kind):
    shift = 20.0
    dragged, typed = _add_at(service, kind, 0.0), _add_at(service, kind, shift)
    _tab(page, "edit")
    _tool(page, "select")
    _frame(page, [dragged, typed])

    # move: drag the body of `dragged`, then type the same offset into x and y
    _select_row(page, dragged)
    a0 = _obj(service, dragged)
    sx, sy = _um_to_client(page, *_body_point(a0))
    page.mouse.move(sx, sy)
    page.mouse.down()
    page.mouse.move(sx + 25, sy - 15, steps=6)
    page.mouse.up()
    _wait(_changed(service, dragged, a0), "the move drag")
    a1 = _obj(service, dragged)
    dx = a1["points_um"][0][0] - a0["points_um"][0][0]
    dy = a1["points_um"][0][1] - a0["points_um"][0][1]
    assert abs(dx) > 0.1 and abs(dy) > 0.1

    _select_row(page, typed)
    b0 = _obj(service, typed)
    x0, y0, _, _ = _bbox(b0)
    assert float(_field_value(page, "#selX")) == pytest.approx(x0, abs=1e-3)
    _type(page, "#selX", x0 + dx)
    _wait(_changed(service, typed, b0), "the typed x")
    b1 = _obj(service, typed)
    _type(page, "#selY", y0 + dy, key="Tab")
    _wait(_changed(service, typed, b1), "the typed y")
    page.wait_for_timeout(300)
    _assert_same_geometry(_obj(service, typed), _obj(service, dragged), (shift, 0.0))

    # size: drag `dragged`'s right-edge handle, then type its new width (unlocked)
    _select_row(page, dragged)
    _frame(page, [dragged])                  # handles far apart: the drag takes `e`
    a2 = _obj(service, dragged)
    ax0, ay0, ax1, ay1 = _bbox(a2)
    hx, hy = _um_to_client(page, ax1, (ay0 + ay1) / 2)
    page.mouse.move(hx, hy)
    page.mouse.down()
    page.mouse.move(hx + 40, hy, steps=6)
    page.mouse.up()
    _wait(_changed(service, dragged, a2), "the scale drag")
    a3 = _obj(service, dragged)
    new_w = _bbox(a3)[2] - _bbox(a3)[0]

    _select_row(page, typed)
    page.uncheck("#selLock")
    b2 = _obj(service, typed)
    _type(page, "#selW", new_w)
    _wait(_changed(service, typed, b2), "the typed width")
    page.wait_for_timeout(300)
    _assert_same_geometry(_obj(service, typed), a3, (shift, 0.0))

    # lock aspect: a typed height scales the width with it
    page.check("#selLock")
    b3 = _obj(service, typed)
    bx0, by0, bx1, by1 = _bbox(b3)
    _type(page, "#selH", (by1 - by0) * 1.5)
    _wait(_changed(service, typed, b3), "the typed height")
    bx0n, by0n, bx1n, by1n = _bbox(_obj(service, typed))
    assert (bx1n - bx0n) == pytest.approx((bx1 - bx0) * 1.5, abs=TOL_UM)
    assert (by1n - by0n) == pytest.approx((by1 - by0) * 1.5, abs=TOL_UM)
    assert (bx0n, by0n) == pytest.approx((bx0, by0), abs=TOL_UM), "scaled from lower-left"


@pytest.mark.parametrize("kind", list(KINDS))
def test_select_rotate_and_skew_fields(service, page, kind):
    shift = 20.0
    ref, typed = _add_at(service, kind, 0.0), _add_at(service, kind, shift)
    _tab(page, "edit")
    _tool(page, "select")
    _select_row(page, typed)

    # rotate: the same op:transform the rotate handle sends, about the box centre
    b0 = _obj(service, typed)
    x0, y0, x1, y1 = _bbox(b0)
    _type(page, "#selRot", 30.0)
    _wait(_changed(service, typed, b0), "the typed rotation")
    _wait(lambda: _field_value(page, "#selRot") == "0", "an applied amount to read 0 again")
    r0 = _obj(service, ref)
    rx0, ry0, rx1, ry1 = _bbox(r0)
    status, r = service.post("/api/design/op", {
        "op": "transform", "indices": [[o["name"] for o in _design(service)].index(ref)],
        "rot_deg": 30.0, "about_um": [(rx0 + rx1) / 2, (ry0 + ry1) / 2]})
    assert status == 200 and r.get("ok"), r
    page.wait_for_timeout(300)
    _assert_same_geometry(_obj(service, typed), _obj(service, ref), (shift, 0.0))

    if kind != "path":
        # filled outlines may carry holes the page cannot see: no skew
        assert _disabled(page, "#selSkewX") and _disabled(page, "#selSkewY")
        assert "open paths" in page.eval_on_selector("#selSkewX", "e => e.title")
        return
    # skew (open paths): an affine on the base Bezier about the box centre
    assert not _disabled(page, "#selSkewX")
    for sel, ax, ay in (("#selSkewX", 20.0, 0.0), ("#selSkewY", 0.0, -15.0)):
        before = _obj(service, typed)
        bx0, by0, bx1, by1 = _bbox(before)
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        tx, ty = math.tan(math.radians(ax)), math.tan(math.radians(ay))
        want = _bezier_affine(before["bezier"],
                              lambda x, y, tx=tx, ty=ty, cx=cx, cy=cy:
                              (x + tx * (y - cy), y + ty * (x - cx)))
        _type(page, sel, ax or ay)
        _wait(_changed(service, typed, before), f"the typed skew {sel}")
        page.wait_for_timeout(300)
        after = _obj(service, typed)
        _assert_nodes(after["bezier"]["nodes"], want)
        assert after["kind"] == before["kind"]
        _wait(lambda s=sel: _field_value(page, s) == "0", f"{sel} to read 0 again")


def test_select_clipboard_buttons(service, page):
    name = _add(service, "funnel")
    _tab(page, "edit")
    _tool(page, "select")
    _select_row(page, name)
    src = _obj(service, name)
    n0 = len(_design(service))
    page.click("#selCopy")

    def done(verb: str, n: int):
        # the page counts a paste once it has re-adopted the design and logged it
        return lambda: sum(f"{verb} {name} as" in r for r in _log_rows(page)) == n

    page.click("#selPaste")
    _wait(done("pasted", 1), "the paste")
    page.click("#selPaste")
    _wait(done("pasted", 2), "the second paste")
    _select_row(page, name)
    page.click("#selDup")
    _wait(done("duplicated", 1), "the duplicate")
    _select_row(page, name)
    page.click("#selDup")
    _wait(done("duplicated", 2), "the second duplicate")
    copies = _design(service)[n0:]
    # a second duplicate of the same source steps on, as a second paste does
    offsets = [(1.0, -1.0), (2.0, -2.0), (1.0, -1.0), (2.0, -2.0)]
    for c, (ox, oy) in zip(copies, offsets, strict=True):
        assert c["name"] != name and c["kind"] == src["kind"]
        assert np.allclose(np.asarray(c["points_um"]),
                           np.asarray(src["points_um"]) + [ox, oy], atol=TOL_UM)


def _log_rows(p) -> list[str]:
    return p.eval_on_selector_all("#log .row", "els => els.map(e => e.textContent)")


def _local_points(p, name: str) -> list:
    return p.evaluate("async (nm) => (await import('/static/ec-state.js')).state.objects"
                      ".find(o => o.name === nm).points_um", name)


@pytest.mark.parametrize("value", ["0", "-2", "1e300"])
def test_select_size_out_of_range_warns_once_and_shows_the_real_width(service, page, value):
    name = _add(service, "rect")
    _tab(page, "edit")
    _tool(page, "select")
    _select_row(page, name)
    real = _field_value(page, "#selW")
    before = _obj(service, name)
    posts = []
    page.on("request", lambda r: posts.append(r.url) if r.method == "POST"
            and "/api/design/" in r.url else None)
    n_log = len(_log_rows(page))
    page.fill("#selW", value)
    page.press("#selW", "Enter")
    page.press("#selW", "Tab")               # the change event after Enter
    page.wait_for_timeout(400)
    warns = [r for r in _log_rows(page)[n_log:] if "width must be" in r]
    assert len(warns) == 1, _log_rows(page)[n_log:]
    assert _field_value(page, "#selW") == real
    assert posts == [], posts
    assert _obj(service, name)["points_um"] == before["points_um"]


def test_select_refused_transform_restores_the_canvas(service, page):
    name = _add(service, "rect")
    _tab(page, "edit")
    _tool(page, "select")
    _select_row(page, name)
    real = _field_value(page, "#selW")
    local_before = _local_points(page, name)
    page.route("**/api/design/op", lambda route: route.fulfill(
        status=200, content_type="application/json",
        body='{"error": "chiral_build_refused", "detail": "refused for the test"}'))
    page.fill("#selW", "9.5")
    page.press("#selW", "Enter")
    page.wait_for_timeout(600)
    page.unroute("**/api/design/op")
    assert _local_points(page, name) == local_before, "the canvas kept a refused scale"
    assert _field_value(page, "#selW") == real


@pytest.mark.parametrize("kind", ["rect", "funnel"])
def test_select_skew_refused_for_a_filled_kind(service, page, kind):
    name = _add(service, kind)
    _tab(page, "edit")
    _tool(page, "select")
    _select_row(page, name)
    assert _disabled(page, "#selSkewX") and _disabled(page, "#selSkewY")
    assert not _disabled(page, "#selRot")
    before = _obj(service, name)
    posts = []
    page.on("request", lambda r: posts.append(r.url) if r.method == "POST"
            and "/api/design/" in r.url else None)
    n_log = len(_log_rows(page))
    page.eval_on_selector("#selSkewY", "e => { e.disabled = false; }")
    page.fill("#selSkewY", "10")
    page.press("#selSkewY", "Enter")
    page.wait_for_timeout(400)
    warns = [r for r in _log_rows(page)[n_log:] if "skew is for open paths" in r]
    assert len(warns) == 1, _log_rows(page)[n_log:]
    assert posts == [] and _obj(service, name)["points_um"] == before["points_um"]


def test_select_skew_refused_for_an_object_with_holes(service, page):
    name = _add(service, "path")
    _tab(page, "edit")
    _tool(page, "select")
    _select_row(page, name)
    assert not _disabled(page, "#selSkewX")
    # the service does not serialize holes today; the page refuses whenever it has them
    page.evaluate("""async (nm) => { const {state} = await import('/static/ec-state.js');
        const o = state.objects.find(q => q.name === nm);
        o.holes_um = [[[5, 10], [6, 10], [6, 11], [5, 11]]]; }""", name)
    _tool(page, "select")                    # a repaint resyncs the fields
    _wait(lambda: _disabled(page, "#selSkewX") and _disabled(page, "#selSkewY"),
          "the skew fields to disable")
    assert not _disabled(page, "#selRot"), "rotate goes through op:transform, holes and all"
    # a skew forced past the disabled field is still refused, once, and sends nothing
    before = _obj(service, name)
    posts = []
    page.on("request", lambda r: posts.append(r.url) if r.method == "POST"
            and "/api/design/" in r.url else None)
    n_log = len(_log_rows(page))
    page.eval_on_selector("#selSkewX", "e => { e.disabled = false; }")
    page.fill("#selSkewX", "20")
    page.press("#selSkewX", "Enter")
    page.wait_for_timeout(400)
    warns = [r for r in _log_rows(page)[n_log:] if "skew refused" in r]
    assert len(warns) == 1, _log_rows(page)[n_log:]
    assert posts == [] and _obj(service, name)["points_um"] == before["points_um"]
    assert _field_value(page, "#selSkewX") == "0"


def test_select_fields_stack_in_half_screen(service, page):
    name = _add(service, "rect")
    _tab(page, "edit")
    _tool(page, "select")
    _select_row(page, name)
    half = page.viewport_size["width"] < 1200
    for a, b in (("#selX", "#selY"), ("#selW", "#selH"), ("#selSkewX", "#selSkewY"),
                 ("#selCopy", "#selPaste")):
        ba = page.query_selector(a).bounding_box()
        bb = page.query_selector(b).bounding_box()
        if half:
            assert bb["y"] >= ba["y"] + ba["height"] - 1, f"{b} sits under {a}"
        else:
            assert abs(bb["y"] - ba["y"]) < 2, f"{a} and {b} share a line"
