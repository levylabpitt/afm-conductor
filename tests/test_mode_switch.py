"""The imaging-mode switch route (POST /api/mode) across the sim/hardware seam.

The cockpit's mode dropdown must actually switch the instrument between contact
and AC by calling the vendor's own mode-profile function -- but ONLY when the tip
is off the surface. Switching imaging mode reconfigures the Z feedback channel +
gain sign and does NOT withdraw, so doing it while engaged is a tip crash. The
decision (option a): REFUSE the switch when engaged/scanning; never auto-withdraw.

These tests assert, off-instrument via the fake Igor bridge:
  * REFUSE when engaged (hub.engaged) or scanning -> NO SETMODE reaches the bridge,
    and hub.imaging_mode is not changed;
  * SEND SETMODE when clear -> the stub records the correct vendor mode token
    ("Contact" / "AC", the MainPopupFunc arg), and hub.imaging_mode is updated;
  * the twin path tracks the mode with no bridge and no error;
  * the litho-AC guard refuses a write when hub.imaging_mode == "ac".

`client` comes from tests/conftest.py (a TestClient). `hub` is a
PROCESS GLOBAL, so each test restores hub.engaged / hub.imaging_mode.
"""

from __future__ import annotations

import contextlib
import time

from _im_fixtures_guard import GUARD_WRITE_KW, seed_free_air


def _guard_write_body(hub, **extra):
    """The REST `/api/litho/write` body under the command guard
    (INSTRUMENT_MODEL_SPEC secs 2/10, 13): a fresh withdrawn free-air baseline
    seeded on the hub, the enabled objects named explicitly (replacing `all:
    true`), plus `setpoint_relative` and `amp_gain`."""
    seed_free_air(hub)
    ids = [o.name for o in hub.pattern.objects if getattr(o, "enabled", True)]
    return {"objects": ids, "setpoint_relative": 0.2, **GUARD_WRITE_KW, **extra}


def _wait(cond, timeout=5.0, interval=0.02):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return False


def _ready_ws(ws):
    """Block until the /ws handler has registered this client, so a write/scan
    started next can't broadcast before we're subscribed."""
    ws.send_text("ping")
    for _ in range(200):
        if ws.receive_json().get("type") == "telemetry":
            return
    raise AssertionError("ws never confirmed registration")


def _attach_bridge(hub):
    """Inject a HardwareARC over a fresh fake Igor bridge (no instrument)."""
    from afm_litho.hardware.hardware_arc import HardwareARC
    from afm_litho.hardware.igor_stub import FakeIgorBridge
    stub = FakeIgorBridge({}, done_delay=0.0)
    port = stub.serve_in_thread()
    hub.host.hardware_arc = HardwareARC.connect("127.0.0.1", port)
    return stub


def _detach_bridge(hub, stub):
    with contextlib.suppress(Exception):
        hub.host.hardware_arc.close()
    hub.host.hardware_arc = None
    stub.stop()


# --------------------------------------------------------------- wire mapping --
# The exact SETMODE -> MainPopupFunc mapping the .ipf implements, asserted at the
# protocol layer (independent of the route) so the vendor call cannot silently drift.

def test_cmd_setmode_maps_to_vendor_tokens():
    from afm_litho.hardware import bridge_protocol as P
    assert P.cmd_setmode("contact") == "SETMODE Contact"
    assert P.cmd_setmode("ac") == "SETMODE AC"
    assert P.cmd_setmode("AC") == "SETMODE AC"           # case-insensitive
    assert P.cmd_setmode("Contact") == "SETMODE Contact"


# ------------------------------------------------------ hardware: SENDS switch --

def test_mode_sends_setmode_when_not_engaged(client):
    from afm_litho.service.app import hub
    stub = _attach_bridge(hub)
    prev_mode = hub.host.imaging_mode
    hub.host.engaged = False
    hub.host.imaging_mode = None
    hub.host.scan_repeat["enabled"] = False
    try:
        r = client.post("/api/mode", json={"mode": "ac"})
        body = r.json()
        assert body == {"ok": True, "mode": "ac", "backend": "hardware"}, r.text
        # the vendor mode token reached the bridge (MainPopupFunc(...,2,"AC Mode"))
        assert _wait(lambda: stub.modes == ["AC"]), stub.modes
        assert hub.host.imaging_mode == "ac"

        # and a contact switch maps to the Contact token
        r = client.post("/api/mode", json={"mode": "contact"})
        assert r.json()["mode"] == "contact"
        assert _wait(lambda: stub.modes == ["AC", "Contact"]), stub.modes
        assert hub.host.imaging_mode == "contact"
    finally:
        _detach_bridge(hub, stub)
        hub.host.imaging_mode = prev_mode
        hub.host.engaged = False


# --------------------------------------------------- hardware: REFUSE (engaged) --
# The safety gate: engaged -> NO SETMODE may reach the bridge, and the tracked mode
# must be untouched. This is the invariant a separate auditor checks can't be bypassed.

def test_mode_refused_when_engaged_sends_no_setmode(client):
    from afm_litho.service.app import hub
    stub = _attach_bridge(hub)
    prev_mode = hub.host.imaging_mode
    hub.host.imaging_mode = "contact"
    hub.host.engaged = True                                   # tip on the surface
    hub.host.scan_repeat["enabled"] = False
    try:
        r = client.post("/api/mode", json={"mode": "ac"})
        body = r.json()
        assert body.get("reason") == "engaged", r.text
        assert "error" in body
        # NOTHING was sent, and the tracked mode is unchanged -- the switch was blocked.
        time.sleep(0.15)                                 # give any (wrong) send time to land
        assert stub.modes == [], stub.modes
        assert hub.host.imaging_mode == "contact"
    finally:
        _detach_bridge(hub, stub)
        hub.host.imaging_mode = prev_mode
        hub.host.engaged = False


# --------------------------------------------------- hardware: REFUSE (scanning) --

def test_mode_refused_when_scanning_sends_no_setmode(client):
    from afm_litho.service.app import hub
    stub = _attach_bridge(hub)
    prev_mode = hub.host.imaging_mode
    hub.host.imaging_mode = "contact"
    hub.host.engaged = False
    hub.host.scan_repeat["enabled"] = True                    # a continuous run is live
    try:
        r = client.post("/api/mode", json={"mode": "ac"})
        assert r.json().get("reason") == "engaged", r.text
        time.sleep(0.15)
        assert stub.modes == [], stub.modes
        assert hub.host.imaging_mode == "contact"
    finally:
        hub.host.scan_repeat["enabled"] = False
        _detach_bridge(hub, stub)
        hub.host.imaging_mode = prev_mode


# -------------------------------------------------------------- twin path -------

def test_mode_twin_tracks_without_bridge(client):
    from afm_litho.service.app import hub
    # 13.1 item 8: the twin now BINDS its SoftwareARC to hub.hardware_arc so the
    # park/interlock primitives run on the twin; the real-hardware BRANCH keys on
    # hub.hw_bridge() (None on the twin), so assert that instead of `is None`.
    assert hub.hw_bridge() is None                      # dev-machine default
    prev_mode = hub.host.imaging_mode
    hub.host.engaged = False
    hub.host.scan_repeat["enabled"] = False
    try:
        r = client.post("/api/mode", json={"mode": "ac"})
        assert r.json() == {"ok": True, "mode": "ac", "backend": "twin"}, r.text
        assert hub.host.imaging_mode == "ac"
    finally:
        hub.host.imaging_mode = prev_mode


def test_mode_rejects_bad_mode(client):
    from afm_litho.service.app import hub
    prev_mode = hub.host.imaging_mode
    try:
        r = client.post("/api/mode", json={"mode": "tapping"})
        body = r.json()
        assert body.get("reason") == "bad_mode", r.text
        assert hub.host.imaging_mode == prev_mode             # untouched by a bad request
    finally:
        hub.host.imaging_mode = prev_mode


# -------------------------------------------------------- GET /api/mode sync ----
# The cockpit syncs its dropdown to the ACTUAL instrument mode on load, so browser and
# software never disagree. GET returns the tracked mode + the live gate state.

def test_get_mode_reflects_tracked_mode_and_gate(client):
    from afm_litho.service.app import hub
    prev_mode = hub.host.imaging_mode
    hub.host.engaged = False
    hub.host.scan_repeat["enabled"] = False
    try:
        hub.host.imaging_mode = "ac"
        d = client.get("/api/mode").json()
        assert d["mode"] == "ac" and d["engaged"] is False, d
        hub.host.engaged = True
        d = client.get("/api/mode").json()
        assert d["engaged"] is True and d["gate"] == "engaged", d
    finally:
        hub.host.imaging_mode = prev_mode
        hub.host.engaged = False


# ------------------------------------------ defense-in-depth: approach reconcile --
# A hardware approach must guarantee the SOFTWARE mode matches the mode it engages in:
# an AC engage with the software in Contact is a dead crosspoint. When the tip is
# withdrawn (approach's precondition) a mismatch AUTO-SWITCHES the software first.

def test_approach_auto_switches_software_mode_when_mismatched(client):
    import asyncio

    from afm_litho.hardware.hardware_arc import HardwareARC
    from afm_litho.hardware.igor_stub import FakeIgorBridge
    from afm_litho.service.app import hub
    from afm_litho.service.routers.scan import approach

    # The switch clears any baseline, so the approach reads a fresh one from the stub:
    # steady DeflectionValue/Amplitude so the 3-read agreement passes, free amplitude
    # 1.3 so the setpoint 0.65 is in the AC fraction band (0.5 in [0.3, 0.95]), and a
    # DetectorSum for the 0.9*SUM ceiling -- the engage itself is reached.
    stub = FakeIgorBridge({"Amplitude": 1.3, "DeflectionValue": -0.2,
                           "Deflection": 1.5, "DetectorSum": 5.0, "height": 20.0},
                          done_delay=0.0)
    port = stub.serve_in_thread()
    prev = (hub.host.hardware_arc, hub.host.imaging_mode, getattr(hub.host, "z_loop", None), hub.host.engaged)
    hub.host.hardware_arc = HardwareARC.connect("127.0.0.1", port)
    hub.host.imaging_mode = "contact"                 # software says contact ...
    hub.host.engaged = False                          # ... and the tip is withdrawn
    prev_fa = getattr(hub.host, "free_air", None)
    hub.host.free_air = None
    try:
        # ... but we approach in AC -> the software mode must be switched FIRST.
        res = asyncio.run(approach({"mode": "ac", "setpoint": 0.65, "settle_s": 0.0}))
        assert res.get("ok") is True, res
        assert _wait(lambda: stub.modes == ["AC"]), stub.modes
        assert hub.host.imaging_mode == "ac"
        assert hub.host.free_air.mode == "ac" and hub.host.free_air.value_v == 1.3
    finally:
        hub.host.hardware_arc.close()
        hub.host.hardware_arc, hub.host.imaging_mode = prev[0], prev[1]
        hub.host.z_loop = prev[2] if prev[2] is not None else {}
        hub.host.engaged = prev[3]
        hub.host.free_air = prev_fa
        stub.stop()


def test_approach_does_not_switch_mode_while_engaged(client):
    """A mismatch WHILE the tip is engaged must NOT switch (a mid-engage channel/gain-
    sign flip is a crash) -- it warns instead. NO SETMODE reaches the bridge."""
    import asyncio

    from afm_litho.hardware.hardware_arc import HardwareARC
    from afm_litho.hardware.igor_stub import FakeIgorBridge
    from afm_litho.service.app import hub
    from afm_litho.service.routers.scan import approach

    stub = FakeIgorBridge({"Amplitude": 0.65, "Deflection": 1.5, "height": 20.0},
                          done_delay=0.0)
    port = stub.serve_in_thread()
    prev = (hub.host.hardware_arc, hub.host.imaging_mode, getattr(hub.host, "z_loop", None), hub.host.engaged)
    hub.host.hardware_arc = HardwareARC.connect("127.0.0.1", port)
    hub.host.imaging_mode = "contact"
    hub.host.engaged = True                           # already engaged: a switch would crash
    try:
        asyncio.run(approach({"mode": "ac", "settle_s": 0.0}))
        import time as _t
        _t.sleep(0.15)
        assert stub.modes == [], stub.modes       # no switch was sent
        assert hub.host.imaging_mode == "contact"      # tracked mode unchanged
    finally:
        hub.host.hardware_arc.close()
        hub.host.hardware_arc, hub.host.imaging_mode = prev[0], prev[1]
        hub.host.z_loop = prev[2] if prev[2] is not None else {}
        hub.host.engaged = prev[3]
        stub.stop()


# ------------------------------------------------ litho guard consults the mode --
# A hardware litho write must be refused when the INSTRUMENT is actually in AC, as
# tracked on hub.imaging_mode -- not only when a scan config was AC.

def test_litho_guard_refuses_on_hub_imaging_mode_ac():
    from afm_litho.service.app import hub
    from afm_litho.service.routers.litho import _ac_mode_active
    prev_mode = hub.host.imaging_mode
    prev_zloop = getattr(hub.host, "z_loop", None)
    prev_cfg = hub.host.scan_repeat.get("config")
    prev_last = hub.host.last_scan_config
    try:
        # z_loop + scan configs all CONTACT, but the instrument mode is AC -> refuse.
        hub.host.z_loop = {"mode": "contact"}
        hub.host.scan_repeat["config"] = {"mode": "contact"}
        hub.host.last_scan_config = {"mode": "contact"}
        hub.host.imaging_mode = "ac"
        assert _ac_mode_active() is True
        # clearing the instrument mode back to contact lifts the guard
        hub.host.imaging_mode = "contact"
        assert _ac_mode_active() is False
    finally:
        hub.host.imaging_mode = prev_mode
        hub.host.z_loop = prev_zloop if prev_zloop is not None else {}
        hub.host.scan_repeat["config"] = prev_cfg
        hub.host.last_scan_config = prev_last


# ------------------------------- gate: the litho-thread witness (unit) ----------
# _engaged_or_scanning must read True while a hardware litho write is in progress even
# if hub.engaged were somehow not set yet -- the running worker thread is an independent
# witness that the tip is on the surface (write_object self-engages the Z loop). This is
# the endpoint->worker gap the audit flagged; assert the thread branch on its own.

def test_gate_flags_a_live_litho_thread_independently():
    import threading

    from afm_litho.service.app import hub
    from afm_litho.service.routers.scan import _engaged_or_scanning

    prev = (hub.host.engaged, hub.host.litho_thread, hub.host.scan_repeat.get("enabled"))
    release = threading.Event()
    t = threading.Thread(target=release.wait, daemon=True)
    try:
        hub.host.engaged = False                 # isolate: ONLY the thread should trip the gate
        hub.host.scan_repeat["enabled"] = False
        t.start()
        hub.host.litho_thread = t
        engaged, why = _engaged_or_scanning()
        assert engaged is True and why == "writing", (engaged, why)
    finally:
        release.set()
        t.join(timeout=5.0)
        hub.host.engaged, hub.host.litho_thread = prev[0], prev[1]
        hub.host.scan_repeat["enabled"] = prev[2]


# ---------------------------- REFUSE during a hardware litho write --------------
# A hardware litho write SELF-ENGAGES (no prior /api/approach), so a mode switch during
# the write is a mid-engage channel/gain-sign flip -> crash. The write worker sets
# hub.engaged True at its start; POST /api/mode must REFUSE and send NO SETMODE while
# the worker is alive. (Fake bridge, done_delay keeps the write in-flight.)

def test_mode_refused_during_hardware_litho_write(client, tmp_path):
    from afm_litho.hardware.hardware_arc import HardwareARC
    from afm_litho.hardware.igor_stub import FakeIgorBridge
    from afm_litho.litho.compiler import LithoObject, Pattern
    from afm_litho.service.app import hub
    from afm_litho.service.sessions import SessionStore

    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    stub = FakeIgorBridge({}, done_delay=1.5)        # hold DONE so the worker stays alive
    port = stub.serve_in_thread()
    hub.host.hardware_arc = HardwareARC.connect("127.0.0.1", port)
    hub.host.hw_cal = {"x_lvdt_sens": 7.46e-6, "x_lvdt_offset": -0.63,
                  "y_lvdt_sens": 7.04e-6, "y_lvdt_offset": 0.07,
                  "z_lvdt_sens": 5e-6, "z_lvdt_offset": 0.0}
    hub.pattern = Pattern(name="gate-test", objects=[
        LithoObject("wire", "path", [(-1.0, 0.0), (1.0, 0.0)], voltage=6.0,
                    speed_um_s=8.0, interpolation_um=0.1)])
    hub.host.written_objects = set()
    hub.host.z_loop = {"mode": "contact"}                 # litho is contact-only (guard precondition)
    hub.host.imaging_mode = "contact"                     # instrument mode (guard + gate read it)
    hub.host.engaged = False
    hub.host.scan_repeat["enabled"] = False
    try:
        with client.websocket_connect("/ws") as ws:
            _ready_ws(ws)
            r = client.post("/api/litho/write", json=_guard_write_body(hub))
            assert r.json().get("backend") == "hardware", r.text
            # the write is in-flight: the worker is alive AND set the engaged gate.
            assert _wait(lambda: hub.host.litho_thread and hub.host.litho_thread.is_alive()
                         and hub.host.engaged is True), "write worker never engaged"
            # a mode switch NOW must be refused, and NO SETMODE may reach the bridge.
            mr = client.post("/api/mode", json={"mode": "ac"})
            assert mr.json().get("reason") == "engaged", mr.text
            time.sleep(0.15)                         # give any (wrong) send time to land
            assert stub.modes == [], stub.modes
            assert hub.host.imaging_mode == "contact"     # tracked mode untouched
            # let the write finish so the worker thread exits cleanly.
            end = time.monotonic() + 20.0
            while time.monotonic() < end:
                if ws.receive_json().get("type") == "litho_done":
                    break
    finally:
        with contextlib.suppress(Exception):
            hub.host.hardware_arc.close()
        hub.host.hardware_arc = None
        hub.host.hw_cal = {}
        hub.host.z_loop = {}
        hub.host.imaging_mode = None
        hub.host.engaged = False
        stub.stop()
        _wait(lambda: not (hub.host.litho_thread and hub.host.litho_thread.is_alive()))


# --------------------- REFUSE after a COMPLETED engaged single scan -------------
# An engaging hardware scan ends with the routine withdraw (loop 2 re-armed), so when the pump
# task finishes the tip is OFF the surface: hub.engaged is False and a mode switch after the
# scan completes is allowed.

def test_mode_allowed_after_completed_engaged_scan(client, tmp_path):
    from afm_litho.hardware.hardware_arc import HardwareARC
    from afm_litho.hardware.igor_stub import FakeIgorBridge
    from afm_litho.service.app import hub
    from afm_litho.service.sessions import SessionStore

    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    stub = FakeIgorBridge({}, done_delay=0.0)
    port = stub.serve_in_thread()
    hub.host.hardware_arc = HardwareARC.connect("127.0.0.1", port)
    hub.host.hw_cal = {"x_lvdt_sens": 7.46e-6, "x_lvdt_offset": -0.63,
                  "y_lvdt_sens": 7.04e-6, "y_lvdt_offset": 0.07,
                  "z_lvdt_sens": 5e-6, "z_lvdt_offset": 0.0}
    hub.host.imaging_mode = "contact"
    hub.host.engaged = False
    hub.host.scan_repeat["enabled"] = False
    try:
        with client.websocket_connect("/ws") as ws:
            _ready_ws(ws)
            # engage:true -> the scan closes the Z loop at setpoint, then withdraws at its end.
            r = client.post("/api/scan/start", json={
                "size_um": 5.0, "pixels": 32, "lines": 4, "line_rate_hz": 200.0,
                "mode": "contact", "engage": True, "setpoint": 0.3})
            assert r.json().get("backend") == "hardware", r.text
            end = time.monotonic() + 12.0
            done = False
            while time.monotonic() < end and not done:
                done = ws.receive_json().get("type") == "scan_done"
            assert done, "the engaged scan never reached scan_done"
            # the pump task has fully finished, and the tip is up.
            assert _wait(lambda: hub.host._scan_task is None or hub.host._scan_task.done())
            assert hub.host.engaged is False, "a completed scan left the tip engaged"
            # the mode switch is allowed and SETMODE flows.
            mr = client.post("/api/mode", json={"mode": "ac"})
            assert mr.json() == {"ok": True, "mode": "ac", "backend": "hardware"}, mr.text
            assert _wait(lambda: stub.modes == ["AC"]), stub.modes
    finally:
        with contextlib.suppress(Exception):
            hub.host.hardware_arc.close()
        hub.host.hardware_arc = None
        hub.host.hw_cal = {}
        hub.host.imaging_mode = None
        hub.host.engaged = False
        stub.stop()
        _wait(lambda: hub.host._scan_task is None or hub.host._scan_task.done())
