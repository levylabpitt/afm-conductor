"""The tip marker on the twin, in a real browser (docs/TWIN_TIP_STREAM_SPEC.md).

The marker is visible at rest on a freshly loaded page (before any scan, in the
page's own start-up view), moves during a write driven over FLEX, is still there
between the objects and after the write, and the "show tip" checkbox hides it in
that page without telling the service anything. A hardware-configured station whose
bridge has not connected gets none of this.

Plumbing (service subprocess with a FLEX provider, browser, page) is reused from
`test_instrument_cockpit.py`.
"""

# ruff: noqa: F811 -- tests take the imported fixtures as parameters.

from __future__ import annotations

import itertools
import shutil
import socket
import time

import pytest
from _service import Service
from test_instrument_cockpit import (
    _WS_CAPTURE_INIT,
    _current_amp_gain,
    _ensure_withdrawn,
    _seed_free_air,
    browser,  # noqa: F401 -- fixture re-export
    flex,  # noqa: F401 -- fixture re-export
    page,  # noqa: F401 -- fixture re-export
    service,  # noqa: F401 -- fixture re-export
)

# The marker (the centre dot, or the edge arrowhead when the tip is outside the view)
# is the only thing drawLitho paints in near-opaque pure white, so counting those
# overlay pixels says whether it is on screen. The view is never changed here.
_MARKER_PIXELS = """
() => {
  const cv = document.getElementById('overlay');
  const d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
  let n = 0;
  for (let i = 0; i < d.length; i += 4)
    if (d[i] > 250 && d[i + 1] > 250 && d[i + 2] > 250 && d[i + 3] > 225) n++;
  return n;
}
"""

_TIP = ("async () => { const {state} = await import('/static/ec-state.js');"
        " return state.litho.tip ? [state.litho.tip.x, state.litho.tip.y] : null; }")

_VIEW = ("async () => { const {state} = await import('/static/ec-state.js');"
         " const v = state.view || state.SCAN_FRAME; return [v.x0, v.y0, v.w, v.h]; }")

# Logs EVERY value the page assigns to the marker (state.litho.tip) into
# window.__tipLog. Frames delayed by a busy service arrive in bursts, which a timed
# sampler would miss; a clear (null) between two frames would be missed too.
_RECORD = """
async () => {
  const {state} = await import('/static/ec-state.js');
  const log = window.__tipLog = [];
  let cur = state.litho.tip;
  Object.defineProperty(state.litho, 'tip', {
    configurable: true, enumerable: true,
    get() { return cur; },
    set(v) { cur = v; log.push({tip: v ? [v.x, v.y] : null, writing: state.litho.writing}); },
  });
}
"""
_STOP = """
async () => {
  const {state} = await import('/static/ec-state.js');
  const cur = state.litho.tip;
  delete state.litho.tip;
  state.litho.tip = cur;
  return window.__tipLog;
}
"""


def _wait_for(fn, timeout: float = 10.0, interval: float = 0.1):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = fn()
        if last:
            return last
        time.sleep(interval)
    return last


def _marker_drawn(page) -> bool:
    """True once a repaint (each tip_pos triggers one) shows the marker."""
    return bool(_wait_for(lambda: page.evaluate(_MARKER_PIXELS) > 0, timeout=5.0))


def _watch_tip_show(page) -> list[str]:
    """Every request the page makes to /api/tip/show from now on."""
    seen: list[str] = []
    page.on("request", lambda r: seen.append(r.url) if "tip/show" in r.url else None)
    return seen


def test_marker_is_visible_at_rest_on_a_fresh_page(service, browser):
    """Before any scan the view is the placeholder frame and the twin's tip sits just
    outside it; the marker must be on screen all the same. Loading the page tells the
    service nothing about Show Tip."""
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000}, permissions=[])
    try:
        p = ctx.new_page()
        tip_show = _watch_tip_show(p)
        p.goto(service.url + "/edit-cockpit", wait_until="load", timeout=45000)
        tip = _wait_for(lambda: p.evaluate(_TIP))
        assert tip, "no tip marker at rest on the twin"
        assert p.is_checked("#showTip")
        view = p.evaluate(_VIEW)
        assert _marker_drawn(p), f"tip at {tip} is not shown in the start-up view {view}"
        p.wait_for_timeout(500)
        assert tip_show == [], f"the page load posted {tip_show}"
    finally:
        ctx.close()


def test_marker_moves_during_a_flex_write_and_persists(page, service, flex):
    free_air = _seed_free_air(service)
    names = []
    for i, pts in enumerate(([[-4.0, -2.0], [-2.0, -2.0]], [[1.0, 2.0], [4.0, 2.0]])):
        name = f"tip-{i}-{int(time.time() * 1000)}"
        status, r = service.post("/api/design/add", {
            "kind": "path", "name": name, "points_um": pts})
        assert status == 200 and r and r.get("ok"), (status, r)
        names.append(name)
    assert _wait_for(lambda: page.evaluate(_TIP)), "no tip marker before the write"
    # The write is played in real time (`twin_speed` 1; the test service's default
    # is unlimited) and reports the tip ten times a second of wall time. At 2 um/s
    # the 5 um move between the objects takes 2.5 s, about 25 positions, so the few
    # asked for below land between the objects however the machine is loaded.
    status, r = service.post("/api/transit", {"speed_um_s": 2.0})
    assert status == 200 and r and "policy" in r, (status, r)

    try:
        page.evaluate(_RECORD)
        res = flex.call("startWrite", {
            "token": flex.acquire(), "objects": names, "twin_speed": 1.0,
            "setpoint_reference": "absolute", "setpoint": free_air + 0.2, "amp_gain": _current_amp_gain(service)})
        assert res["objects"] == 2, res
        deadline = time.time() + 60
        while time.time() < deadline:
            st, s = service.get("/api/state")
            if st == 200 and s and not s.get("writing", True):
                break
            time.sleep(0.1)
        page.wait_for_timeout(1500)                 # the stream goes on after the write
        log = page.evaluate(_STOP)
        _ensure_withdrawn(service)
    finally:
        service.post("/api/transit", {"speed_um_s": 20.0})   # TransitPolicy's default

    assert any(e["writing"] for e in log), "the page never saw the write"
    # present the whole time: during the strokes, between the objects, and after
    assert all(e["tip"] is not None for e in log), (
        f"the marker was cleared at assignment "
        f"{[e['tip'] is None for e in log].index(True)} of {len(log)}")
    seen = {tuple(e["tip"]) for e in log}
    assert len(seen) >= 10, f"the marker barely moved: {sorted(seen)}"
    # it travelled BETWEEN the objects: positions off both stroke lines (y -2 and +2)
    between = [p for p in seen if -1.5 < p[1] < 1.5 and -2.5 < p[0] < 1.5]
    assert len(between) >= 3, (
        f"too few marker positions between the two objects: {sorted(seen)}")
    # and rests where the second stroke ended
    assert log[-1]["tip"] == pytest.approx([4.0, 2.0], abs=0.1)
    assert _marker_drawn(page)


# Hands the page's own message handlers a write's frames in a chosen order, in ONE
# synchronous turn so no frame from the live twin stream lands in between, and returns
# every value the marker took plus the live trace. Marker frames (`tip_pos`) are at
# x = 2.0, trace frames (`litho_progress`) at x = 3.0 and 3.5.
_FEED = """
async (order) => {
  const {state} = await import('/static/ec-state.js');
  const {dispatchWs} = await import('/static/ec-api.js');
  const frames = {
    progress: {type: 'litho_progress', name: 'o', frac: 0.5, x_um: 3.0, y_um: 7.0,
               i_na: [], trail_um: [[2.5, 7.0], [3.0, 7.0]]},
    progress_hw: {type: 'litho_progress', name: 'o', frac: 0.6, x_um: 3.5, y_um: 7.0,
                  i_na: []},
    tip: {type: 'tip_pos', x_um: 2.0, y_um: 7.0},
  };
  const seen = [];
  let cur = state.litho.tip;
  Object.defineProperty(state.litho, 'tip', {
    configurable: true, enumerable: true,
    get() { return cur; },
    set(v) { cur = v; seen.push(v ? [v.x, v.y] : null); },
  });
  let live;
  try {
    dispatchWs({type: 'litho_object_start', name: 'o', voltage: 1});
    for (const k of order) dispatchWs(frames[k]);
    live = state.litho.live.map((p) => [p[0], p[1]]);
  } finally {
    delete state.litho.tip;
    state.litho.tip = cur;
    dispatchWs({type: 'litho_done', aborted: false});
  }
  return {seen, tip: cur ? [cur.x, cur.y] : null, live};
}
"""


@pytest.mark.parametrize("order", [
    ["progress", "tip"],                    # a newer trace frame, then an older marker frame
    ["tip", "progress"],                    # the reverse
    ["tip", "progress_hw", "progress"],     # the hardware frame shape (no trail_um) too
])
def test_only_tip_pos_moves_the_marker(page, order):
    assert _wait_for(lambda: page.evaluate(_TIP)), "no tip marker before the feed"
    r = page.evaluate(_FEED, order)
    assert r["tip"] == [2.0, 7.0], f"the marker ended off the tip_pos position: {r}"
    assert r["seen"] == [[2.0, 7.0]], (
        f"the marker took a position that no tip_pos carried: {r['seen']}")
    # the trace is still fed by litho_progress, in arrival order
    want = {"progress": [[2.5, 7.0], [3.0, 7.0]], "progress_hw": [[3.5, 7.0]], "tip": []}
    assert r["live"] == [p for k in order for p in want[k]]


def test_marker_never_steps_backward_along_a_stroke(page, service, flex):
    """One straight stroke in +x: every position the marker takes on it is at or ahead
    of the one before. Played at twice real time (3 s for the stroke, about 30
    positions); the speed of the machine only changes how many are checked."""
    free_air = _seed_free_air(service)
    name = f"mono-{int(time.time() * 1000)}"
    x0, x1, y = -6.0, 6.0, 3.0
    status, r = service.post("/api/design/add", {
        "kind": "path", "name": name, "points_um": [[x0, y], [x1, y]]})
    assert status == 200 and r and r.get("ok"), (status, r)
    assert _wait_for(lambda: page.evaluate(_TIP)), "no tip marker before the write"
    # The move to the stroke's first point is one jump, whatever the tip's position was
    # (it depends on the tests that ran before). A slower move reports intermediate
    # positions, and the last of them can fall just short of the first point but inside
    # its 0.05 um box, which the stroke check below then reads as a step backward.
    status, r = service.post("/api/transit", {"speed_um_s": 1000.0})
    assert status == 200 and r and "policy" in r, (status, r)

    page.evaluate(_RECORD)
    res = flex.call("startWrite", {
        "token": flex.acquire(), "objects": [name], "twin_speed": 2.0,
        "setpoint_reference": "absolute", "setpoint": free_air + 0.2, "amp_gain": _current_amp_gain(service)})
    assert res["objects"] == 1, res
    deadline = time.time() + 120
    while time.time() < deadline:
        st, s = service.get("/api/state")
        if st == 200 and s and not s.get("writing", True):
            break
        time.sleep(0.1)
    page.wait_for_timeout(1500)                 # the closing tip_pos of the stroke
    log = page.evaluate(_STOP)
    _ensure_withdrawn(service)
    service.post("/api/transit", {"speed_um_s": 20.0})      # TransitPolicy's default

    assert all(e["tip"] is not None for e in log), "the marker was cleared mid-write"
    tips = [e["tip"] for e in log]
    # The stroke starts where the marker first sits on its first point (the transit
    # there comes in from elsewhere) and runs to the end of the log: the tip stays on
    # the last point after the write.
    at_start = [i for i, t in enumerate(tips)
                if abs(t[0] - x0) < 0.05 and abs(t[1] - y) < 0.05]
    assert at_start, f"the marker never reached the stroke's first point: {tips[:20]}"
    stroke = tips[at_start[0]:]
    assert all(abs(t[1] - y) < 0.05 for t in stroke), f"the marker left the stroke: {stroke}"
    xs = [t[0] for t in stroke]
    back = [(i, round(a, 4), round(b, 4))
            for i, (a, b) in enumerate(itertools.pairwise(xs)) if b < a - 1e-6]
    assert not back, (
        f"the marker stepped backward {len(back)} times in {len(xs)} positions: {back[:8]}")
    assert len(set(xs)) >= 3, f"the marker did not travel along the stroke: {xs}"
    assert xs[-1] == pytest.approx(x1, abs=0.1)


def test_show_tip_checkbox_hides_the_marker_in_this_page_only(page, service, browser):
    assert _wait_for(lambda: page.evaluate(_TIP))
    assert _marker_drawn(page)
    other_ctx = browser.new_context(viewport={"width": 1600, "height": 1000},
                                    permissions=[])
    try:
        other = other_ctx.new_page()
        other.goto(service.url + "/edit-cockpit", wait_until="load", timeout=45000)
        assert _marker_drawn(other)

        tip_show = _watch_tip_show(page)
        page.uncheck("#showTip")
        page.wait_for_timeout(1500)             # longer than the at-rest period
        assert page.evaluate(_MARKER_PIXELS) == 0, "the tip marker is still drawn"
        assert tip_show == [], f"the twin page told the service: {tip_show}"
        # the service goes on streaming: the other page still has its marker
        assert other.is_checked("#showTip") and _marker_drawn(other)

        page.check("#showTip")
        assert _marker_drawn(page), "the marker did not come back"
        assert tip_show == []
    finally:
        other_ctx.close()


def test_a_hardware_station_without_a_bridge_gets_no_marker(browser, monkeypatch):
    """Configured for hardware, Igor not connected: the page must not take the station
    for the twin -- Show Tip stays off, and no simulated tip is drawn."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        bridge_port = s.getsockname()[1]
    monkeypatch.setenv("AFM_LITHO_BACKEND", "hardware")
    monkeypatch.setenv("AFM_LITHO_BRIDGE_PORT", str(bridge_port))
    svc = Service(flex=True)
    try:
        svc.start()
    except Exception as exc:      # noqa: BLE001 -- pragma: no cover - environment
        shutil.rmtree(svc.runtime, ignore_errors=True)
        pytest.skip(f"hardware-configured service would not start: {exc}")
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000}, permissions=[])
    try:
        status, state = svc.get("/api/state")
        assert status == 200 and state["backend"] == "disconnected", state
        ctx.add_init_script(_WS_CAPTURE_INIT)
        p = ctx.new_page()
        tip_show = _watch_tip_show(p)
        p.goto(svc.url + "/edit-cockpit", wait_until="load", timeout=45000)
        p.wait_for_timeout(2500)                # several at-rest periods
        assert not p.is_checked("#showTip")
        assert p.evaluate(_TIP) is None, "a simulated tip reached a hardware station"
        assert p.evaluate(_MARKER_PIXELS) == 0
        assert tip_show == []
    finally:
        ctx.close()
        svc.stop()
        shutil.rmtree(svc.runtime, ignore_errors=True)
