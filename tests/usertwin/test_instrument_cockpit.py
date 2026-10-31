"""RED (wave-1 tests-only) coverage for docs/INSTRUMENT_MODEL_SPEC.md sec 4
(cockpit) + sec 10 (the DOM-id/door contract), against the real edit cockpit
in a real browser -- a DOM stub would not catch a `value="0.2"` literal still
sitting in the HTML, or the Write button still sending `{all:true}`.

None of this exists on `instrument-model` today:

  * `#ampGainRO`, `#instrumentPanel`, `#openInstrumentConfig`,
    `#engagedNoBaseline`, `#abortBtn` are not in edit-cockpit.html at all
    (the amp-gain field is the EDITABLE `#ampGain`; the abort button's id is
    `#abort`, not `#abortBtn`).
  * `#lithoSetpoint` and the scan tab's `#setpoint` both carry literal
    `value="..."` in the served HTML.
  * `#perform`'s click handler posts `{all: true, amp_gain, setpoint}`
    (`web/ec-main.js`), never an explicit `objects` list, and never
    `setpoint_relative`.
  * `GET /api/instrument`, `POST /api/instrument/override`,
    `POST /api/instrument/open_config` do not exist (no
    `service/instrument_model.py`, no `/api/instrument` router).

`design_changed` IS already broadcast by every design-mutating door
(`routers/design.py`), so the "cockpit re-adopts the design on a
server-side mutation" test exercises real, already-shipped plumbing on the
frontend side -- only the cockpit's LISTENER for it is new (also not found
in `web/ec-main.js` today).

Coordinator corrections + adversarial-review must-fixes folded in:
  * (12a0593, sec 4+10) the write setpoint box is RELATIVE to free air
    ("setpoint (V above free air)"); the door key is `setpoint_relative`,
    never a bare `setpoint`, and `#perform`'s body is
    `{objects, setpoint_relative, amp_gain}` with no `all`.
  * every hydration test seeds a NON-DEFAULT value first (a non-25 amp gain
    via `POST /api/instrument/override`, a non-1.0 relative setpoint, a
    known free-air baseline) so a cockpit that hardcodes today's default
    cannot pass by accident.
  * every test seeds its OWN preconditions -- no test depends on another
    test's leftover service state or execution order.
  * the design_changed test also flips an existing object's `enabled` and
    `voltage` over a second HTTP client and checks both land in the DOM.
  * the abort-under-notice test checks a second (corner) point of
    `#abortBtn`, and looks for the park-failed notice's "vendor withdraw"
    control.
  * (cockpit-implementer finding) a write body's `amp_gain` must equal the
    model's `tip_amp_gain` -- `command_guard.check_write` (section 2)
    refuses any mismatch (`amp_gain_mismatch`) -- so every seeding write
    reads the model's current gain via `GET /api/instrument` rather than a
    literal. `hub.free_air` is seeded through the one production path that
    actually sets it (a fast Relative-setpoint continuous scan's
    between-frame re-reference) -- a bare approach+withdraw never sets it at
    all.

Uses its OWN service + browser plumbing: a throwaway runtime dir, an
ephemeral port, and the service as a child process that dies with the
module. Skips cleanly (see `_playwright` below) when no Chromium is
installed.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import pytest

playwright = pytest.importorskip(
    "playwright.sync_api", reason="playwright not installed (pip install -e .[dev])")

import afm_litho
from _service import Service

WEB_DIR = Path(afm_litho.__file__).resolve().parent / "web"


# --------------------------------------------------------------- service --

# ----------------------------------------------------------- FLEX door --

class _FlexClient:
    """A raw ZMQ REQ JSON-RPC client against the subprocess's FLEX command
    socket -- same shape as `tests/test_flex_litho.py::_Client`, kept local
    for the same reason `Service` is: this file owns its own plumbing
    rather than reaching into another suite's fixtures."""

    def __init__(self, endpoint: str, timeout_s: float = 30.0):
        import zmq
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
        self.sock.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
        self.sock.connect(endpoint)
        self._n = 0
        self.token: str | None = None

    def send(self, method: str, params: dict | None = None) -> dict:
        self._n += 1
        req = {"jsonrpc": "2.0", "method": method, "id": str(self._n)}
        if params is not None:
            req["params"] = params
        self.sock.send_string(json.dumps(req))
        return json.loads(self.sock.recv_string())

    def call(self, method: str, params: dict | None = None):
        reply = self.send(method, params)
        assert "error" not in reply, reply
        return reply["result"]

    def acquire(self) -> str:
        reply = self.send("acquireControl",
                          {"client": "im-tests-T4-usertwin", "deadman_s": 300})
        if "error" in reply:
            # a previous run in this same subprocess abandoned the token --
            # there is no other client racing a fresh subprocess for it.
            reply = self.send("acquireControl",
                              {"client": "im-tests-T4-usertwin", "deadman_s": 300})
        self.token = reply["result"]["token"]
        return self.token

    def close(self) -> None:
        try:
            if self.token:
                self.send("releaseControl", {"token": self.token})
        except Exception:  # noqa: BLE001, S110 - best-effort cleanup
            pass
        self.sock.close(linger=0)
        self.ctx.term()


@pytest.fixture(scope="module")
def flex(service):
    pytest.importorskip("zmq", reason="pyzmq is the optional [flex] extra")
    endpoint = f"tcp://127.0.0.1:{service.flex_port}"
    deadline = time.time() + 20
    last_exc = None
    client = None
    while time.time() < deadline:
        try:
            client = _FlexClient(endpoint, timeout_s=10.0)
            client.acquire()
            break
        except Exception as exc:      # noqa: BLE001 -- FLEX may not be bound yet
            last_exc = exc
            if client is not None:
                try:
                    client.sock.close(linger=0)
                    client.ctx.term()
                except Exception:  # noqa: BLE001, S110 - best-effort cleanup
                    pass
            client = None
            time.sleep(0.5)
    if client is None:
        pytest.skip(f"FLEX command socket at {endpoint} never came up: {last_exc}")
    yield client
    client.close()


@pytest.fixture(scope="module")
def service():
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


# ---------------------------------------------------------- HTTP seeding --

_NON_DEFAULT_GAIN = 17.5           # rig default is 25.0 (config-only, sec 1)
_NON_DEFAULT_SETPOINT_REL = 0.37   # cockpit default box is 0.2
_NON_DEFAULT_SCAN_SETPOINT = 0.63  # scan tab's #setpoint literal is "1"


def _seed_amp_gain_override(service, value: float = _NON_DEFAULT_GAIN) -> float:
    """`POST /api/instrument/override` (section 10) -- does not exist yet.
    A non-default value so a cockpit that hardcodes the rig's 25.0 default
    cannot pass this by accident."""
    status, r = service.post("/api/instrument/override", {
        "field": "tip_amp_gain", "value": value, "reason": "test", "confirm": True})
    assert status == 200 and r and r.get("applied"), (
        f"POST /api/instrument/override did not apply: {status} {r}")
    return value


def _ensure_withdrawn(service, timeout: float = 8.0) -> None:
    """Retract and confirm it stuck. `/api/state`'s `engaged` (twin-only: it
    comes from `hub.arc.snapshot()`'s `world.engaged`, a physics-derived
    tip-sample-gap flag -- NOT the same thing as the service-level
    `hub.engaged` safety gate) does not necessarily drop the instant
    `/api/withdraw` returns, and a litho WRITE genuinely puts the tip down on
    the sample (that is what writing means), so it is left physically
    engaged afterward unless something retracts it. `service` is
    module-scoped, so a seeding helper that skips this leaves the twin
    engaged for every later test in the file."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        service.post("/api/withdraw", {})
        st, s = service.get("/api/state")
        if st == 200 and s and not s.get("engaged", False):
            return
        time.sleep(0.3)


def _current_amp_gain(service) -> float:
    """`command_guard.check_write` (section 2, landed post-B3) REFUSES any
    write whose `amp_gain` != `model.tip_amp_gain` (`amp_gain_mismatch`) --
    caught by the cockpit implementer against the migrated helper below,
    which used to hardcode `1.0`. Reads the model's current value instead of
    a literal `25.0`, so this keeps working under a test that has already
    overridden the gain (e.g. `test_a_reload_does_not_change_...`, which
    seeds a non-default gain via `_seed_amp_gain_override` first)."""
    status, model = service.get("/api/instrument")
    assert status == 200, (status, model)
    return float(model["tip_amp_gain"]["value"])


def _seed_free_air(service, mode: str = "contact") -> float:
    """A fresh, withdrawn, contact-mode free-air baseline.

    `hub.free_air` (section 10) is populated by exactly ONE production path
    today (`service/host/continuous.py:_read_free_air_deflection`, called
    from the between-frame re-reference) -- a plain approach+withdraw does
    NOT set it. (`scan_cmds.py` DOES clear `hub.free_air` to `None`, but only
    on an imaging-MODE switch, "a free-air baseline is mode-specific" -- not
    on withdraw, so an explicit withdraw below is safe and does not undo the
    seed.) There is no dedicated "read free-air now" door, and this file's
    `Service` is a SUBPROCESS (unlike `tests/test_write_records.py`, which
    is in-process and can call `tests/_im_fixtures_guard.py::
    seed_free_air(hub, ...)` directly) -- no test-only HTTP seed route
    exists either (checked `routers/instrument.py`), so this drives the one
    real path there is: a fast Relative-setpoint continuous scan, run long
    enough for at least one between-frame re-reference to land, then
    stopped AND WITHDRAWN -- otherwise the twin is left mid-scan/engaged for
    every test that runs after this one (module-scoped `service`).
    `relative.last_free_air_v` (the same call's other side effect) is
    polled as the "it happened" signal and returned.
    """
    status, r = service.post("/api/scan/relative", {"enabled": True, "settle_s": 0.2})
    assert status == 200, (status, r)
    status, r = service.post("/api/scan/continuous", {
        "enabled": True,
        "config": {"size_um": 3.0, "pixels": 3, "lines": 3, "line_rate_hz": 80.0,
                  "angle_deg": 0.0, "mode": mode, "setpoint": 0.3,
                  "direction": "down", "x_offset_um": 0.0, "y_offset_um": 0.0},
        "max_frames": 4, "interval_s": 0.1})
    assert status == 200 and not (r or {}).get("error"), (status, r)

    deadline = time.time() + 25
    last_free_air = None
    while time.time() < deadline:
        st, s = service.get("/api/scan/continuous")
        if st == 200 and s:
            last_free_air = (s.get("relative") or {}).get("last_free_air_v")
            if last_free_air is not None:
                break
        time.sleep(0.2)

    service.post("/api/scan/continuous", {"enabled": False})
    service.post("/api/scan/relative", {"enabled": False})
    _ensure_withdrawn(service)     # leave the twin retracted for later tests
    assert last_free_air is not None, (
        "the continuous relative-setpoint re-reference never populated "
        "hub.free_air over HTTP within the deadline")
    return float(last_free_air)


def _seed_write_setpoint(service, setpoint_relative: float = _NON_DEFAULT_SETPOINT_REL,
                         amp_gain: float | None = None) -> str:
    """Adds one object and writes it with an explicit, non-default relative
    setpoint, so `hub.last_write_params` has something distinctive to
    hydrate `#lithoSetpoint` from.

    `amp_gain` defaults to the MODEL's current value (`GET /api/instrument`),
    never a literal -- `command_guard.check_write` refuses any write whose
    `amp_gain` does not equal `model.tip_amp_gain` (`amp_gain_mismatch`), and
    the twin's rig default is 25.0, not 1.0.
    """
    if amp_gain is None:
        amp_gain = _current_amp_gain(service)
    name = f"seed-{int(time.time() * 1000)}"
    status, r = service.post("/api/design/add", {
        "kind": "path", "name": name, "points_um": [[1.0, 1.0], [5.0, 5.0]]})
    assert status == 200 and r and r.get("ok"), (status, r)
    status, w = service.post("/api/litho/write", {
        "objects": [name], "setpoint_relative": setpoint_relative, "amp_gain": amp_gain})
    assert status == 200 and w and w.get("ok"), (status, w)
    deadline = time.time() + 20
    while time.time() < deadline:
        st, s = service.get("/api/state")
        if st == 200 and s and not s.get("writing", True):
            break
        time.sleep(0.2)
    # Writing genuinely puts the tip down on the sample -- withdraw so the
    # twin is not left engaged (no baseline) for every later test in this
    # module-scoped-service file (see `_ensure_withdrawn`).
    _ensure_withdrawn(service)
    return name


def _seed_scan_setpoint(service, setpoint: float = _NON_DEFAULT_SCAN_SETPOINT,
                        mode: str = "contact") -> None:
    """A small, fast scan with a distinctive, non-default setpoint, so
    `#setpoint` (scan tab) has something non-trivial to hydrate from
    `hub.last_scan_params` (section 10: "hub.last_scan_params per mode")."""
    status, r = service.post("/api/scan/start", {
        "size_um": 4.0, "pixels": 4, "lines": 4, "line_rate_hz": 50.0,
        "angle_deg": 0.0, "mode": mode, "setpoint": setpoint, "direction": "down",
        "x_offset_um": 0.0, "y_offset_um": 0.0})
    assert status == 200 and r and not r.get("error"), (status, r)
    deadline = time.time() + 15
    while time.time() < deadline:
        st, s = service.get("/api/state")
        if st == 200 and s and not s.get("scanning", True):
            break
        time.sleep(0.2)


# ------------------------------------------------------------------- page --

# Captures every WebSocket the page opens (ec-safety.js, ec-telemetry.js,
# ec-control.js each open their own) onto `window.__wsInstances`, so a test
# can synthesize an incoming server frame -- `dispatchEvent(new
# MessageEvent(...))` on a real WebSocket invokes its `onmessage` exactly as
# a genuine frame would -- WITHOUT needing the twin to actually drive a
# critical safe_park (the fault-injection hook the spec names,
# `twin.faults.ramp_noop`, does not exist yet either). This is the "test
# hook that emits the same ws event" alternative the brief allows.
# NOTE: `add_init_script` runs this string as a SCRIPT BODY, not a function to
# call -- a bare `() => {...}` arrow expression would just be evaluated and
# discarded, never invoked. It must be a self-invoking IIFE.
_WS_CAPTURE_INIT = """
(() => {
  window.__wsInstances = [];
  const NativeWS = window.WebSocket;
  function Wrapped(...args) {
    const inst = new NativeWS(...args);
    window.__wsInstances.push(inst);
    return inst;
  }
  Wrapped.prototype = NativeWS.prototype;
  Wrapped.CONNECTING = NativeWS.CONNECTING;
  Wrapped.OPEN = NativeWS.OPEN;
  Wrapped.CLOSING = NativeWS.CLOSING;
  Wrapped.CLOSED = NativeWS.CLOSED;
  window.WebSocket = Wrapped;
})();
"""


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    try:
        b = pw.chromium.launch(headless=True)
    except Exception as exc:      # noqa: BLE001 -- pragma: no cover - environment
        pw.stop()
        pytest.skip(f"no browser: {exc}. Run `playwright install chromium`.")
    yield b
    b.close()
    pw.stop()


@pytest.fixture
def page(service, browser):
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000}, permissions=[])
    ctx.add_init_script(_WS_CAPTURE_INIT)
    p = ctx.new_page()
    p.goto(service.url + "/edit-cockpit", wait_until="load", timeout=45000)
    p.wait_for_timeout(600)
    yield p
    ctx.close()


def _to_write_tab(page) -> None:
    """Select the Write tab -- `#perform`/`#lithoSetpoint`/`#ampGainRO`
    all ship `display:none` until it is (ec-main.js:
    setTab -- tab switching is CSS-only, so the element exists in the DOM
    either way, it just is not visible/clickable). Needed after every fresh
    load or reload (`ec-main.js` has no tab-in-URL/localStorage persistence,
    a fresh load always starts on Scan)."""
    page.click('button.tab[data-tab="write"]', timeout=5000)
    page.wait_for_timeout(200)


def _object_rows(page) -> list[dict]:
    """The object list as the DOM shows it (`ec-main.js:renderList`'s
    `#objList .objrow`), read through the DOM rather than the module-scoped
    `state` (`ec-main.js` is `type="module"`, so `state` -- imported from
    `ec-state.js` -- is not a global and `page.evaluate("state...")` throws a
    ReferenceError; nothing exposes it on `window`). Each row gives its name,
    whether it is shown disabled (`.dis`), and its swatch color -- the only
    DOM-visible proxy for voltage (`renderList`'s `voltageColor(o.voltage,
    vmax)`), since the row shows no voltage number."""
    return page.eval_on_selector_all(
        "#objList .objrow",
        "els => els.map(e => ({"
        "  name: e.querySelector('.nm')?.textContent,"
        "  disabled: e.classList.contains('dis'),"
        "  swatch: e.querySelector('.swatch')?.style.background"
        "}))")


def _object_names(page) -> list[str]:
    return [r["name"] for r in _object_rows(page)]


def _enabled_object_names(page) -> list[str]:
    return [r["name"] for r in _object_rows(page) if not r["disabled"]]


# ---------------------------------------------------------- static markup --

def _no_value_attribute(input_id: str) -> None:
    html = (WEB_DIR / "edit-cockpit.html").read_text(encoding="utf-8")
    assert f'id="{input_id}"' in html
    m = re.search(rf'<input[^>]*id="{input_id}"[^>]*>', html)
    assert m, f"no #{input_id} input tag found"
    assert "value=" not in m.group(0), (
        f"#{input_id} still carries a literal HTML value attribute: {m.group(0)!r}")


def test_lithosetpoint_carries_no_html_value_attribute_in_the_served_file():
    """Checked on the SOURCE FILE too (not just the DOM after JS runs), per
    the brief -- the DOM check below can be fooled by JS setting `.value`
    after load; the markup itself must not ship a literal."""
    _no_value_attribute("lithoSetpoint")


def test_scan_setpoint_carries_no_html_value_attribute_in_the_served_file():
    """Should-fix 6: the scan tab's own `#setpoint` (today `value="1"`) must
    also hydrate from the service (`hub.last_scan_params`), not ship a
    literal."""
    _no_value_attribute("setpoint")


# -------------------------------------------------------------- hydration --

def test_amp_gain_readout_hydrates_a_seeded_nondefault_value_and_is_not_editable(
        service, page):
    """Must-fix 2: seeds a NON-25 gain first via an accepted override, so a
    cockpit hardcoding the rig's 25.0 default cannot pass this by
    accident."""
    seeded = _seed_amp_gain_override(service)
    status, model = service.get("/api/instrument")
    assert status == 200, f"GET /api/instrument -> {status} {model}"
    assert "tip_amp_gain" in model, model
    assert float(model["tip_amp_gain"]["value"]) == pytest.approx(seeded)

    page.reload(wait_until="load", timeout=45000)
    page.wait_for_timeout(600)
    _to_write_tab(page)

    el = page.query_selector("#ampGainRO")
    assert el is not None, "#ampGainRO not found in the cockpit"
    tag = el.evaluate("e => e.tagName.toLowerCase()")
    editable = el.evaluate(
        "e => e.tagName.toLowerCase() === 'input' && !e.readOnly && !e.disabled")
    assert not editable, f"#ampGainRO ({tag}) is an editable input"

    shown = (page.text_content("#ampGainRO") or el.evaluate("e => e.value || ''")).strip()
    assert shown, "#ampGainRO shows nothing"
    assert float(shown) == pytest.approx(seeded), (
        f"#ampGainRO shows {shown!r}, expected the seeded {seeded} "
        f"(a hardcoded 25.0 default would show 25 here)")


def test_lithosetpoint_hydrates_from_hub_last_write_params_after_seeding(service, page):
    """Seeds a NON-default relative setpoint and a known free-air baseline
    first (must-fix 1 + 2), then loads a FRESH page and checks the displayed
    #lithoSetpoint matches -- not the 0.2 literal default."""
    _seed_free_air(service)
    seeded = _seed_write_setpoint(service, setpoint_relative=_NON_DEFAULT_SETPOINT_REL)
    assert seeded

    page.reload(wait_until="load", timeout=45000)
    page.wait_for_timeout(600)
    _to_write_tab(page)

    el = page.query_selector("#lithoSetpoint")
    assert el is not None
    shown = el.evaluate("e => e.value")
    assert shown not in (None, ""), "#lithoSetpoint shows nothing after hydration"
    assert float(shown) == pytest.approx(_NON_DEFAULT_SETPOINT_REL), (
        f"#lithoSetpoint shows {shown!r}, expected the seeded "
        f"{_NON_DEFAULT_SETPOINT_REL}")


def test_scan_setpoint_hydrates_from_hub_last_scan_params_after_seeding(service, page):
    """Should-fix 6: the scan tab's `#setpoint` must hydrate from
    `hub.last_scan_params` for the current mode, the same way #lithoSetpoint
    hydrates from `hub.last_write_params` -- section 10 names both."""
    _seed_scan_setpoint(service, setpoint=_NON_DEFAULT_SCAN_SETPOINT, mode="contact")

    page.reload(wait_until="load", timeout=45000)
    page.wait_for_timeout(600)   # Scan is the default tab -- no click needed

    el = page.query_selector("#setpoint")
    assert el is not None, "#setpoint not found"
    shown = el.evaluate("e => e.value")
    assert shown not in (None, ""), "#setpoint shows nothing after hydration"
    assert float(shown) == pytest.approx(_NON_DEFAULT_SCAN_SETPOINT), (
        f"#setpoint shows {shown!r}, expected the seeded {_NON_DEFAULT_SCAN_SETPOINT}")


def test_a_reload_does_not_change_amp_gain_ro_or_litho_setpoint(service, page):
    """Self-seeds its own known, non-default gain/setpoint (must-fix 8: no
    dependence on another test's leftover state or file order) and checks
    they read back identically before and after a reload."""
    seeded_gain = _seed_amp_gain_override(service, value=_NON_DEFAULT_GAIN + 1.0)
    _seed_free_air(service)
    _seed_write_setpoint(service, setpoint_relative=_NON_DEFAULT_SETPOINT_REL + 0.05)

    page.reload(wait_until="load", timeout=45000)
    page.wait_for_timeout(600)
    _to_write_tab(page)

    gain_el = page.query_selector("#ampGainRO")
    sp_el = page.query_selector("#lithoSetpoint")
    assert gain_el is not None, "#ampGainRO not found"
    assert sp_el is not None, "#lithoSetpoint not found"
    before_gain = gain_el.evaluate("e => e.value ?? e.textContent")
    before_sp = sp_el.evaluate("e => e.value")
    assert float(before_gain) == pytest.approx(seeded_gain), before_gain

    page.reload(wait_until="load", timeout=45000)
    page.wait_for_timeout(600)
    _to_write_tab(page)

    gain_el = page.query_selector("#ampGainRO")
    sp_el = page.query_selector("#lithoSetpoint")
    assert gain_el is not None, "#ampGainRO not found after reload"
    assert sp_el is not None, "#lithoSetpoint not found after reload"
    after_gain = gain_el.evaluate("e => e.value ?? e.textContent")
    after_sp = sp_el.evaluate("e => e.value")
    assert before_gain == after_gain
    assert before_sp == after_sp


# ---------------------------------------------------------- design_changed --

def test_cockpit_reflects_a_design_mutation_from_a_second_client_without_reload(
        service, page):
    """Self-seeds its own object (must-fix 8), then flips BOTH `enabled` and
    `voltage` over a second HTTP client (should-fix 5) and checks the DOM
    reflects both without a reload: the row gains the `.dis` class, and its
    swatch color (the only DOM-visible proxy for voltage -- the list shows no
    voltage number) changes."""
    unique_name = f"usertwin-mutation-{int(time.time() * 1000)}"
    status, r = service.post("/api/design/add", {
        "kind": "path", "name": unique_name, "voltage": 8.0,
        "points_um": [[2.0, 2.0], [9.0, 9.0]]})
    assert status == 200 and r and r.get("ok"), (status, r)
    idx = r["index"]

    deadline = time.time() + 10
    row = None
    while time.time() < deadline:
        rows = {row["name"]: row for row in _object_rows(page)}
        if unique_name in rows:
            row = rows[unique_name]
            break
        page.wait_for_timeout(200)
    assert row is not None, (
        "the cockpit's object list did not pick up the added object "
        "(design_changed) without a reload")
    assert not row["disabled"]
    before_swatch = row["swatch"]

    status, r = service.post("/api/design/op", {
        "op": "set_props", "indices": [idx], "enabled": False, "voltage": 30.0})
    assert status == 200 and "error" not in (r or {}), (status, r)

    deadline = time.time() + 10
    seen = False
    while time.time() < deadline:
        rows = {row["name"]: row for row in _object_rows(page)}
        row = rows.get(unique_name)
        if row and row["disabled"] and row["swatch"] != before_swatch:
            seen = True
            break
        page.wait_for_timeout(200)
    assert seen, (
        "the cockpit did not reflect a second client's enabled+voltage "
        f"mutation without a reload (last seen: {row})")


# ------------------------------------------------------------------ write --

def test_perform_sends_explicit_object_ids_never_all(page):
    """Coordinator correction (instrument-model 12a0593, sec 4+10) + must-fix
    1: the body is `{objects, setpoint_relative, amp_gain}` -- no `all`, and
    no bare `setpoint` (the setpoint box is relative to free air)."""
    captured = {}

    def handler(route):
        try:
            captured["body"] = json.loads(route.request.post_data or "{}")
        except Exception:      # noqa: BLE001 -- best-effort JSON decode of the intercepted body
            captured["body"] = None
        route.fulfill(status=200, content_type="application/json",
                     body=json.dumps({"ok": True, "writing": 0, "skipped": 0,
                                       "run_id": "write-test"}))

    page.route("**/api/litho/write", handler)
    _to_write_tab(page)
    enabled = _enabled_object_names(page)

    perform = page.query_selector("#perform")
    assert perform is not None, "#perform not found"
    perform.click()
    page.wait_for_timeout(500)

    assert "body" in captured, "#perform did not POST /api/litho/write"
    body = captured["body"]
    assert body is not None, "the write body was not valid JSON"
    assert "all" not in body, f"#perform still sends an 'all' key: {body}"
    assert "setpoint" not in body, (
        f"#perform sends a bare 'setpoint' key -- it must be 'setpoint_relative': {body}")
    assert "objects" in body, f"#perform did not send an explicit objects list: {body}"
    assert "setpoint_relative" in body, (
        f"#perform did not send setpoint_relative: {body}")
    assert "amp_gain" in body, f"#perform did not send amp_gain: {body}"
    assert sorted(body["objects"], key=str) == sorted(enabled, key=str), (
        f"objects sent ({body['objects']}) != enabled objects shown on screen ({enabled})")


# ------------------------------------------------- Write is enabled engaged --

def test_perform_is_enabled_after_approach_and_no_engaged_banner_exists(service, page):
    """docs/COCKPIT_WRITE_GATE_SPEC.md: the run reads free air before every object,
    so an engaged tip does not disable Write and no engaged banner is shown."""
    _to_write_tab(page)
    sp = page.query_selector("#lithoSetpoint")
    assert sp is not None
    sp.fill("0.2")
    sp.dispatch_event("input")
    try:
        status, r = service.post("/api/approach", {"mode": "contact"})
        assert status == 200 and r and not r.get("error"), (status, r)
        deadline = time.time() + 8
        while time.time() < deadline:
            st, s = service.get("/api/state")
            if st == 200 and s and s.get("engaged"):
                break
            time.sleep(0.2)
        assert s and s.get("engaged"), f"the twin did not engage: {s}"
        page.wait_for_timeout(800)   # let the engaged ws frame reach the page

        perform = page.query_selector("#perform")
        assert perform is not None
        assert not perform.is_disabled(), "#perform is disabled while engaged"
        assert page.query_selector("#engagedNoBaseline") is None, (
            "the #engagedNoBaseline banner still exists")
        assert page.query_selector(
            '.banner-chip[data-banner="engagedNoBaseline"]') is None, (
            "the engaged banner's header chip still exists")
    finally:
        _ensure_withdrawn(service)


def test_perform_is_disabled_while_a_write_runs(service, page):
    """With engagement no longer gating Write, the write-in-progress flag is what
    stops a second Perform during a write: setting it and re-running the gate (an
    input event on #lithoSetpoint) disables #perform; clearing it re-enables it."""
    _to_write_tab(page)
    sp = page.query_selector("#lithoSetpoint")
    perform = page.query_selector("#perform")
    assert sp is not None and perform is not None
    sp.fill("0.2")
    sp.dispatch_event("input")
    assert not perform.is_disabled(), "precondition: #perform disabled with a setpoint"
    try:
        page.evaluate("() => { document.getElementById('perform').dataset.writing = '1'; }")
        sp.dispatch_event("input")
        assert perform.is_disabled(), "#perform is enabled while a write runs"
    finally:
        page.evaluate("() => { document.getElementById('perform').dataset.writing = '0'; }")
        sp.dispatch_event("input")
    assert not perform.is_disabled(), "#perform stayed disabled after the write ended"


# --------------------------------------------------------------- abort UI --

def _emit_critical_safe_park(page) -> None:
    ws_count = page.evaluate("() => window.__wsInstances.length")
    assert ws_count > 0, "no WebSocket connections were opened by the cockpit"
    page.evaluate("""
      () => window.__wsInstances.forEach(ws => ws.dispatchEvent(
        new MessageEvent('message', {data: JSON.stringify({
          type: 'safe_park', reason: 'usertwin critical-notice test',
          critical: true, verified: false, z_v: 3.2, transport: 'test'})})))
    """)
    page.wait_for_timeout(400)


def test_abort_button_stays_clickable_under_a_critical_notice(page):
    _emit_critical_safe_park(page)

    abort_btn = page.query_selector("#abortBtn")
    assert abort_btn is not None, "#abortBtn not found (today's id is #abort)"
    assert abort_btn.is_visible(), "#abortBtn is not visible under a critical notice"

    box = abort_btn.bounding_box()
    assert box is not None
    # centre AND a corner (should-fix 7): a notice that only overlaps one
    # edge of the button would pass a centre-only check while still
    # covering, say, the top-right corner a fast click might land on.
    points = {
        "centre": (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2),
        "top-right corner": (box["x"] + box["width"] - 1, box["y"] + 1),
    }
    for label, (x, y) in points.items():
        top_id = page.evaluate(
            "([x, y]) => { const e = document.elementFromPoint(x, y); "
            "return e ? (e.id || e.closest('[id]')?.id || e.tagName) : null; }",
            [x, y])
        assert top_id == "abortBtn", (
            f"a critical notice covers #abortBtn's {label}: elementFromPoint -> {top_id!r}")


def test_park_failed_notice_carries_a_vendor_withdraw_control(page):
    """Section 4: "The park-failed notice carries a 'vendor withdraw' button
    that calls the step-4 path." """
    _emit_critical_safe_park(page)
    btn = page.get_by_role("button", name=re.compile("vendor withdraw", re.IGNORECASE))
    assert btn.count() > 0, (
        "no 'vendor withdraw' control found on the park-failed critical notice")
    assert btn.first.is_visible()


# ------------------------------------------------------------ writeEstimate --

def _write_estimate_seconds(page) -> float:
    el = page.query_selector("#writeEstimate")
    assert el is not None, "#writeEstimate not found"
    text = (el.text_content() or "").strip()
    m = re.search(r"[-+]?\d*\.?\d+", text)
    assert m, f"no number found in #writeEstimate text: {text!r}"
    return float(m.group(0))


def test_write_estimate_shows_the_service_total_after_load(service, page):
    """Coordinator addition (instrument-model ce30dfa, sec 4+10): `#writeEstimate`
    shows `GET /api/litho/estimate`'s `total_s`, refreshed on load."""
    status, r = service.post("/api/design/add", {
        "kind": "path", "name": f"est-{int(time.time() * 1000)}", "voltage": 6.0,
        "speed_um_s": 2.0, "points_um": [[2.0, 2.0], [22.0, 2.0]]})
    assert status == 200 and r and r.get("ok"), (status, r)

    page.reload(wait_until="load", timeout=45000)
    page.wait_for_timeout(600)
    _to_write_tab(page)

    status, est = service.get("/api/litho/estimate")
    assert status == 200, (status, est)

    shown = _write_estimate_seconds(page)
    assert shown == pytest.approx(est["total_s"], abs=0.5), (
        f"#writeEstimate shows {shown}, service total_s is {est['total_s']}")


def test_write_estimate_rises_after_a_second_client_enables_an_object(service, page):
    """The stale-view detector: a second client (the FLEX door, or here a
    second HTTP client on the same door) enables an object the cockpit did
    not have enabled -- `#writeEstimate` must rise to the new service total
    WITHOUT a reload, the same `design_changed`-driven refresh as the object
    list."""
    _to_write_tab(page)
    name = f"est-toggle-{int(time.time() * 1000)}"
    status, r = service.post("/api/design/add", {
        "kind": "path", "name": name, "voltage": 6.0, "speed_um_s": 1.0,
        "points_um": [[2.0, 2.0], [12.0, 2.0]]})
    assert status == 200 and r and r.get("ok"), (status, r)
    idx = r["index"]
    status, r2 = service.post("/api/design/op", {
        "op": "set_props", "indices": [idx], "enabled": False})
    assert status == 200 and "error" not in (r2 or {}), (status, r2)

    page.reload(wait_until="load", timeout=45000)
    page.wait_for_timeout(600)
    _to_write_tab(page)
    before = _write_estimate_seconds(page)

    status, r3 = service.post("/api/design/op", {
        "op": "set_props", "indices": [idx], "enabled": True})
    assert status == 200 and "error" not in (r3 or {}), (status, r3)

    deadline = time.time() + 10
    after = before
    while time.time() < deadline:
        after = _write_estimate_seconds(page)
        if after > before + 1e-6:
            break
        page.wait_for_timeout(200)
    assert after > before, (
        f"#writeEstimate did not rise after a second client enabled an "
        f"object without a reload (before={before}, after={after})")

    status, est = service.get("/api/litho/estimate")
    assert status == 200, (status, est)
    assert after == pytest.approx(est["total_s"], abs=0.5), (
        f"#writeEstimate ({after}) does not match the new service total_s "
        f"({est['total_s']})")


# --------------------------------------------------------- open config file --

def test_open_instrument_config_posts_exactly_once(page):
    calls = []

    def handler(route):
        calls.append(route.request.post_data)
        route.fulfill(status=200, content_type="application/json",
                     body=json.dumps({"path": "runtime/instrument.toml"}))

    page.route("**/api/instrument/open_config", handler)
    # Section 14 item 2: #openInstrumentConfig moved from the removed
    # #instrumentPanel into the #configDialog modal, opened from the command-log
    # bar's Config button. Open the dialog before the button is reachable.
    page.click("#configBtn", timeout=5000)
    page.wait_for_timeout(300)

    btn = page.query_selector("#openInstrumentConfig")
    assert btn is not None, "#openInstrumentConfig not found"
    btn.click()
    page.wait_for_timeout(500)

    assert len(calls) == 1, (
        f"#openInstrumentConfig click produced {len(calls)} "
        f"POST /api/instrument/open_config calls, expected exactly 1")
