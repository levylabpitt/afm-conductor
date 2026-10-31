"""The litho write/abort route across the sim/hardware seam.

Mirrors test_behavior_scan.py's hardware-backend test: with `hub.hardware_arc`
None (the dev-machine default) the write MUST take the twin path unchanged, and
with a HardwareARC injected over the fake Igor bridge the write MUST take the
hardware branch and emit the SAME litho_* WS contract the cockpit consumes. No
real hardware is required -- the fake bridge answers STARTUP/GETVALUE and fires
DONE after ARMFIRE, so the whole write runs off-instrument.

`client` comes from tests/conftest.py. `hub` is a PROCESS GLOBAL, so each test
redirects hub.store to a temp file and always tears its write down (a live
worker thread makes the next write return {"error": "write already running"}).
"""

from __future__ import annotations

import contextlib
import time

import pytest
from _im_fixtures_guard import GUARD_WRITE_KW, seed_free_air

# `client` comes from tests/conftest.py (a TestClient).


@pytest.fixture(autouse=True)
def _idle_hub(litho_ready_hub):
    """Every test here starts from a hub with no scan config left by an earlier test."""


def _guard_write_body(hub, **extra):
    """The REST `/api/litho/write` body under the command guard
    (INSTRUMENT_MODEL_SPEC secs 2/10, 13): a fresh withdrawn free-air baseline
    seeded on the hub, every enabled object named explicitly (replacing the old
    `all: true`), plus `setpoint_relative` and `amp_gain`."""
    seed_free_air(hub)
    ids = [o.name for o in hub.pattern.objects if getattr(o, "enabled", True)]
    return {"objects": ids, "setpoint_relative": 0.2, **GUARD_WRITE_KW, **extra}


def _wait(cond, timeout=20.0, interval=0.02):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return False


def _tmp_store(tmp_path):
    """Point the session archive at a temp file so a completing write records
    there, not in the real runtime/sessions.jsonl."""
    from afm_litho.service.app import hub
    from afm_litho.service.sessions import SessionStore
    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    return hub


def _ready_ws(ws):
    """Block until the /ws handler has registered this client, so a write
    started next can't broadcast before we're subscribed."""
    ws.send_text("ping")
    for _ in range(200):
        if ws.receive_json().get("type") == "telemetry":
            return
    raise AssertionError("ws never confirmed registration")


def _tiny_pattern():
    from afm_litho.litho.compiler import LithoObject, Pattern
    return Pattern(name="hw-litho-test", objects=[
        LithoObject("wire", "path", [(-1.0, 0.0), (1.0, 0.0)], voltage=6.0,
                    speed_um_s=8.0, interpolation_um=0.1)])


# ------------------------------------------------ twin path (default) --------
# hardware_arc is None on the dev machine, so /api/litho/write must behave
# exactly as before: the shared worker on the twin plugin, a response with no "backend"
# key, and litho_abort routing to the twin stop.

def test_write_takes_twin_path_when_no_hardware(client, tmp_path):
    from afm_litho.service.app import hub
    _tmp_store(tmp_path)
    # 13.1 item 8: the twin now BINDS its SoftwareARC to hub.hardware_arc so the
    # park/interlock primitives run on the twin; the real-hardware BRANCH keys on
    # hub.hw_bridge() (None on the twin), so assert that instead of `is None`.
    assert hub.hw_bridge() is None                 # dev-machine default
    hub.pattern = _tiny_pattern()
    hub.host.written_objects = set()
    # Drive through a ws so the finally can drain the async _litho_pump to its
    # litho_done: that task is not tracked on hub (unlike hub._scan_task), so a
    # lingering pump would otherwise broadcast this run's litho_done onto a LATER
    # test's socket.
    with client.websocket_connect("/ws") as ws:
        _ready_ws(ws)
        try:
            r = client.post("/api/litho/write", json=_guard_write_body(hub))
            body = r.json()
            assert body.get("ok") is True, r.text
            assert body["writing"] == 1
            assert "backend" not in body            # twin response is unchanged
            # the twin worker was started, on the twin plugin (not the hardware worker)
            assert hub.host._twin_plugin is not None
        finally:
            # twin abort: Fix 4 routes it through the unified safe_park (sets the
            # run's abort flag + the twin stop/withdraw), backend-tagged "twin".
            ar = client.post("/api/litho/abort")
            assert ar.json() == {"ok": True, "backend": "twin"}
            _wait(lambda: not (hub.host.litho_thread and hub.host.litho_thread.is_alive()))
            # drain the socket until the pump emits litho_done, so the pump task
            # has fully ended before the next test connects.
            end = time.monotonic() + 15.0
            while time.monotonic() < end:
                if ws.receive_json().get("type") == "litho_done":
                    break


# ------------------------------------------- hardware branch (injected) ------
# With a HardwareARC over the fake Igor bridge, the write drives LithographyRun
# over that same bridge link and emits the cockpit's litho_* contract. Injected
# here exactly as the scan hardware test injects its backend -- no instrument.

def test_write_takes_hardware_branch_and_emits_contract(client, tmp_path):
    from afm_litho.hardware.hardware_arc import HardwareARC
    from afm_litho.hardware.igor_stub import FakeIgorBridge
    from afm_litho.service.app import hub

    _tmp_store(tmp_path)
    stub = FakeIgorBridge({}, done_delay=0.0)
    port = stub.serve_in_thread()
    hub.host.hardware_arc = HardwareARC.connect("127.0.0.1", port)
    # calibration the driver's um->LVDT conversion needs (as the connect hook
    # would have read via STARTUP); same shape as the scan hardware test.
    hub.host.hw_cal = {"x_lvdt_sens": 7.46e-6, "x_lvdt_offset": -0.63,
                  "y_lvdt_sens": 7.04e-6, "y_lvdt_offset": 0.07,
                  "z_lvdt_sens": 5e-6, "z_lvdt_offset": 0.0}
    hub.pattern = _tiny_pattern()
    hub.host.written_objects = set()
    hub.host.z_loop = {"mode": "contact"}                 # litho is contact-only (guard precondition)
    hub.host.imaging_mode = "contact"                      # ... and the instrument mode too (guard reads it)
    try:
        with client.websocket_connect("/ws") as ws:
            _ready_ws(ws)
            r = client.post("/api/litho/write", json=_guard_write_body(hub))
            body = r.json()
            assert body.get("backend") == "hardware", r.text
            assert body["ok"] is True and body["writing"] == 1

            # collect the litho_* stream in order, stepping over log/telemetry
            wanted = {"litho_object_start", "litho_progress",
                      "litho_object_done", "litho_done"}
            seen = []
            end = time.monotonic() + 120.0
            while time.monotonic() < end:
                m = ws.receive_json()
                t = m.get("type")
                # start collecting only at our object_start, so a stray trailing
                # litho_done from an earlier run's pump can't truncate the stream
                if not seen and t != "litho_object_start":
                    continue
                if t in wanted:
                    seen.append(m)
                    if t == "litho_done":
                        break
            types = [m["type"] for m in seen]
            assert types == ["litho_object_start", "litho_progress",
                             "litho_object_done", "litho_done"], types

            start, prog, done, fin = seen
            # the exact keys the cockpit reads off each message
            assert {"name", "voltage", "length_um", "duration_s"} <= set(start)
            assert start["name"] == "wire"
            assert {"name", "frac", "x_um", "y_um"} <= set(prog)
            assert prog["frac"] == 1.0
            assert {"name", "path_um", "v_written", "voltage"} <= set(done)
            assert len(done["path_um"]) >= 2 and len(done["v_written"]) >= 2
            assert fin["aborted"] is False
    finally:
        try:
            hub.host.hardware_arc.close()
        except Exception:  # noqa: S110 - teardown must not mask the test result
            pass
        hub.host.hardware_arc = None                     # restore simulation
        hub.host.hw_cal = {}
        hub.host.z_loop = {}
        hub.host.imaging_mode = None
        stub.stop()
        _wait(lambda: not (hub.host.litho_thread and hub.host.litho_thread.is_alive()))


# ---------------------------------------- hardware pass capture persist ------
# A hardware litho pass must be SAVED like a scan: the worker accumulates the
# RAW per-channel capture the bridge streams (UserIn2 bias readback, Lateral,
# sensors), and the pump records it under the session id via save_capture, loadable
# back. lateral_armed=False is the .ipf fallback: bank 2 refused the UserIn2 + Lateral
# pair and armed UserIn2 alone, so the bias is still captured and Lateral is absent.

@pytest.mark.parametrize("lateral_armed", [True, False])
def test_hardware_litho_pass_saves_loadable_capture(client, tmp_path, lateral_armed):
    from afm_litho.hardware.hardware_arc import HardwareARC
    from afm_litho.hardware.igor_stub import FakeIgorBridge
    from afm_litho.service.app import hub
    from afm_litho.service.captures import load_capture

    _tmp_store(tmp_path)
    # the fake Igor "measures" a bias readback + sensors on the write's capture
    # banks (as AFM_LithoDone streams AFM_SendCapture per channel).
    def _cap(name, cap_len):
        return {"UserIn2": [0.3] * cap_len, "Lateral": [-0.51] * cap_len,
                "ZSensor": [0.1] * cap_len, "Deflection": [0.0] * cap_len,
                "XSensor": [0.05] * cap_len, "YSensor": [0.02] * cap_len}
    stub = FakeIgorBridge({}, done_delay=0.0, capture=_cap)
    stub.lateral_armed = lateral_armed
    port = stub.serve_in_thread()
    hub.host.hardware_arc = HardwareARC.connect("127.0.0.1", port)
    hub.host.hw_cal = {"x_lvdt_sens": 7.46e-6, "x_lvdt_offset": -0.63,
                  "y_lvdt_sens": 7.04e-6, "y_lvdt_offset": 0.07,
                  "z_lvdt_sens": 5e-6, "z_lvdt_offset": 0.0}
    hub.pattern = _tiny_pattern()
    hub.host.written_objects = set()
    hub.host.z_loop = {"mode": "contact"}                 # litho is contact-only (guard precondition)
    hub.host.imaging_mode = "contact"                      # ... and the instrument mode too (guard reads it)
    try:
        with client.websocket_connect("/ws") as ws:
            _ready_ws(ws)
            r = client.post("/api/litho/write", json=_guard_write_body(hub))
            assert r.json().get("backend") == "hardware", r.text
            # drain to litho_done so the pump's record+save has been reached
            end = time.monotonic() + 20.0
            while time.monotonic() < end:
                if ws.receive_json().get("type") == "litho_done":
                    break

            def _litho_id():
                for rr in hub.store.list(limit=5):
                    if rr["kind"] == "litho":
                        return rr["id"]
                return None

            assert _wait(lambda: _litho_id() is not None), "no litho session recorded"
            sid = _litho_id()
            # the pump saves the capture in a worker thread after litho_done, so poll
            assert _wait(lambda: bool(load_capture(tmp_path, sid))), "capture not saved"
            cap = load_capture(tmp_path, sid)
            assert "UserIn2" in cap and "ZSensor" in cap
            assert set(cap) >= {"UserIn2", "ZSensor", "Deflection",
                                "XSensor", "YSensor", "output_a_cmd_v"}
            assert len(cap["UserIn2"]) > 0
            import numpy as np
            assert np.allclose(cap["UserIn2"], 0.3)
            if lateral_armed:
                assert len(cap["Lateral"]) == len(cap["UserIn2"])
                assert np.allclose(cap["Lateral"], -0.51)
            else:
                assert "Lateral" not in cap
            # the session metadata records the channels, and the .gwy autosave ran
            # (one native file holding every captured channel), under the run's
            # date-time stem in writes/gwy/ (service/stems.py) rather than the
            # old flat captures/<id>.gwy
            assert list((tmp_path / "writes" / "gwy").glob("*.gwy"))
    finally:
        try:
            hub.host.hardware_arc.close()
        except Exception:  # noqa: S110 - teardown must not mask the test result
            pass
        hub.host.hardware_arc = None
        hub.host.hw_cal = {}
        hub.host.hw_litho_capture = {}
        hub.host.z_loop = {}
        hub.host.imaging_mode = None
        stub.stop()
        _wait(lambda: not (hub.host.litho_thread and hub.host.litho_thread.is_alive()))


# ---------------------------------------- litho contact-only guard -----------
# SAFETY-CRITICAL: lithography is ALWAYS contact. On the hardware backend a write
# must be REFUSED when the instrument is in AC/tapping mode (the engaged Z-servo
# mode, hub.z_loop), so the litho Z loop can never servo on amplitude. No worker
# thread starts; the response carries the ac_mode_active reason.

def test_litho_write_refused_in_ac_mode_on_hardware(client, tmp_path):
    from afm_litho.hardware.hardware_arc import HardwareARC
    from afm_litho.hardware.igor_stub import FakeIgorBridge
    from afm_litho.service.app import hub

    _tmp_store(tmp_path)
    stub = FakeIgorBridge({}, done_delay=0.0)
    stub.crosspoint = {"InFast": "ACDefl"}           # AUTHORITATIVE: instrument is in AC
    port = stub.serve_in_thread()
    hub.host.hardware_arc = HardwareARC.connect("127.0.0.1", port)
    hub.host.hw_cal = {"x_lvdt_sens": 7.46e-6, "x_lvdt_offset": -0.63,
                  "y_lvdt_sens": 7.04e-6, "y_lvdt_offset": 0.07,
                  "z_lvdt_sens": 5e-6, "z_lvdt_offset": 0.0}
    hub.pattern = _tiny_pattern()
    hub.host.written_objects = set()
    hub.host.z_loop = {"setpoint": 0.65, "pgain": 0.0, "igain": 10.0, "mode": "ac"}
    try:
        r = client.post("/api/litho/write", json=_guard_write_body(hub))
        body = r.json()
        assert body.get("reason") == "ac_mode_active", r.text
        assert "error" in body and "contact-only" in body["error"]
        # nothing was written and no worker thread was spawned
        assert not (getattr(hub.host, "litho_thread", None) and hub.host.litho_thread.is_alive())
        assert hub.host.written_objects == set()
    finally:
        hub.host.hardware_arc.close()
        hub.host.hardware_arc = None
        hub.host.hw_cal = {}
        hub.host.z_loop = {}
        hub.host.imaging_mode = None
        stub.stop()


def test_litho_write_allowed_in_contact_mode_on_hardware(client, tmp_path):
    """The guard is AC-specific: a contact-engaged instrument writes normally."""
    from afm_litho.hardware.hardware_arc import HardwareARC
    from afm_litho.hardware.igor_stub import FakeIgorBridge
    from afm_litho.service.app import hub

    _tmp_store(tmp_path)
    stub = FakeIgorBridge({}, done_delay=0.0)
    stub.crosspoint = {"InFast": "FilterOut"}         # AUTHORITATIVE: instrument is in contact
    port = stub.serve_in_thread()
    hub.host.hardware_arc = HardwareARC.connect("127.0.0.1", port)
    hub.host.hw_cal = {"x_lvdt_sens": 7.46e-6, "x_lvdt_offset": -0.63,
                  "y_lvdt_sens": 7.04e-6, "y_lvdt_offset": 0.07,
                  "z_lvdt_sens": 5e-6, "z_lvdt_offset": 0.0}
    hub.pattern = _tiny_pattern()
    hub.host.written_objects = set()
    hub.host.z_loop = {"setpoint": 0.3, "pgain": 0.0, "igain": 10.0, "mode": "contact"}
    hub.host.imaging_mode = "contact"                      # instrument mode: contact (guard reads it)
    try:
        with client.websocket_connect("/ws") as ws:
            _ready_ws(ws)
            r = client.post("/api/litho/write", json=_guard_write_body(hub))
            body = r.json()
            assert body.get("backend") == "hardware", r.text
            assert body["ok"] is True
            end = time.monotonic() + 20.0
            while time.monotonic() < end:
                if ws.receive_json().get("type") == "litho_done":
                    break
    finally:
        hub.host.hardware_arc.close()
        hub.host.hardware_arc = None
        hub.host.hw_cal = {}
        hub.host.z_loop = {}
        hub.host.imaging_mode = None
        stub.stop()
        _wait(lambda: not (hub.host.litho_thread and hub.host.litho_thread.is_alive()))


# -------------------------------------------- hardware abort routing ---------
# Fix 4: litho_abort on the hardware backend routes through the unified safe_park,
# which fires AFM_SafePark OUT-OF-BAND (COM-first, TCP ABORT fallback -- here TCP,
# no COM handle) and sets the cooperative flag. So the worker is told to stop, an
# ABORT reaches the bridge, and the reply is hardware-tagged. (Verify settle patched
# down so the test doesn't sit through the real ~2.5 s Output.Z ramp wait.)

def test_abort_routes_to_bridge_on_hardware(client, tmp_path, monkeypatch):
    import sys

    from afm_litho.hardware.hardware_arc import HardwareARC
    from afm_litho.hardware.igor_stub import FakeIgorBridge
    from afm_litho.service.app import hub
    from afm_litho.service.host.safe_park import safe_park as _sp_fn

    # patch the SUBMODULE's settle (the package rebinds the `safe_park` attr to the
    # function, so reach the module via sys.modules) so the test skips the ~2.5 s wait.
    monkeypatch.setattr(
        sys.modules[_sp_fn.__module__], "_VERIFY_SETTLE_S", 0.05)
    _tmp_store(tmp_path)
    stub = FakeIgorBridge({}, done_delay=0.0)
    port = stub.serve_in_thread()
    hub.host.hardware_arc = HardwareARC.connect("127.0.0.1", port)
    try:
        hub.host._hw_litho_abort.clear()
        r = client.post("/api/litho/abort")
        assert r.json() == {"ok": True, "backend": "hardware"}
        assert hub.host._hw_litho_abort.is_set()         # worker told to stop
        # the ABORT (== AFM_SafePark on the wire) reached the bridge
        assert _wait(lambda: stub.aborts >= 1)
    finally:
        try:
            hub.host.hardware_arc.close()
        except Exception:  # noqa: S110 - teardown must not mask the test result
            pass
        hub.host.hardware_arc = None
        hub.host._hw_litho_abort.clear()
        hub.host._safe_park["active"] = False
        stub.stop()


# ------------------------------------ per-object free air: setpoint carry -----
# docs/PER_OBJECT_FREE_AIR_SPEC.md: the door hands the run the setpoint as it was
# requested. REST (relative) is re-resolved against the run's own free-air read before
# the object; FLEX (absolute) is used as sent. The stub's DeflectionValue (-0.35 V)
# differs from the door's seeded baseline (-0.2 V), so the set-down shows which won.

def _z_setdowns(stub):
    """The set-down setpoints: loop-2 re-arms at anything but the -10 V retract."""
    return [lp[2] for lp in stub.loops if lp[0] == 2 and lp[2] != -10.0]


def _hw_write_setdowns(client, tmp_path, send):
    stub = _hw_write(client, tmp_path, send)
    return _z_setdowns(stub)


def _hw_write(client, tmp_path, send, *, engaged=False, free_air_v=-0.35):
    """Run one hardware write over the fake bridge (DeflectionValue `free_air_v`, door
    baseline seeded at -0.2 V); `send(hub)` starts it. Returns the stopped stub, with
    the run's litho_done message as `stub.litho_done`."""
    from afm_litho.hardware.hardware_arc import HardwareARC
    from afm_litho.hardware.igor_stub import FakeIgorBridge
    from afm_litho.service.app import hub

    _tmp_store(tmp_path)
    stub = FakeIgorBridge({"DeflectionValue": free_air_v}, done_delay=0.0)
    port = stub.serve_in_thread()
    hub.host.hardware_arc = HardwareARC.connect("127.0.0.1", port)
    hub.host.hw_cal = {"x_lvdt_sens": 7.46e-6, "x_lvdt_offset": -0.63,
                  "y_lvdt_sens": 7.04e-6, "y_lvdt_offset": 0.07,
                  "z_lvdt_sens": 5e-6, "z_lvdt_offset": 0.0}
    hub.pattern = _tiny_pattern()
    hub.host.written_objects = set()
    hub.host.z_loop = {"mode": "contact"}
    hub.host.imaging_mode = "contact"
    try:
        seed_free_air(hub, value_v=-0.2)
        hub.host.engaged = engaged
        with client.websocket_connect("/ws") as ws:
            _ready_ws(ws)
            send(hub)
            end = time.monotonic() + 30.0
            while time.monotonic() < end:
                m = ws.receive_json()
                if m.get("type") == "litho_done":
                    stub.litho_done = m
                    break
        assert _wait(lambda: not (hub.host.litho_thread and hub.host.litho_thread.is_alive()))
        return stub
    finally:
        hub.host.engaged = False
        with contextlib.suppress(Exception):
            hub.host.hardware_arc.close()
        hub.host.hardware_arc = None
        hub.host.hw_cal = {}
        hub.host.z_loop = {}
        hub.host.imaging_mode = None
        stub.stop()
        _wait(lambda: not (hub.host.litho_thread and hub.host.litho_thread.is_alive()))


def test_rest_relative_setpoint_is_resolved_against_the_runs_free_air(client, tmp_path):
    def send(hub):
        # seeds the door's baseline at -0.2 V; setpoint_relative 0.2
        r = client.post("/api/litho/write", json=_guard_write_body(hub))
        assert r.json().get("backend") == "hardware", r.text

    # -0.35 V (the run's own read before the object) + 0.2, not the door's -0.2 + 0.2
    assert _hw_write_setdowns(client, tmp_path, send) == [pytest.approx(-0.15)]


def test_flex_absolute_setpoint_is_used_as_sent(client, tmp_path):
    import functools

    from afm_litho.service.host import litho_cmds

    def send(hub):
        ids = [o.name for o in hub.pattern.objects if getattr(o, "enabled", True)]
        body = {"objects": ids, "setpoint": 0.1, "setpoint_reference": "absolute", **GUARD_WRITE_KW}
        res = client.portal.call(functools.partial(litho_cmds.litho_write, hub, body,
                                                   intake=litho_cmds.flex_intake))
        assert res.get("backend") == "hardware", res

    assert _hw_write_setdowns(client, tmp_path, send) == [pytest.approx(0.1)]


def test_flex_free_air_setpoint_is_the_runs_free_air_plus_the_setpoint(client, tmp_path):
    import functools

    from afm_litho.service.host import litho_cmds

    def send(hub):
        ids = [o.name for o in hub.pattern.objects if getattr(o, "enabled", True)]
        body = {"objects": ids, "setpoint": 0.2, "setpoint_reference": "free_air",
                **GUARD_WRITE_KW}
        res = client.portal.call(functools.partial(litho_cmds.litho_write, hub, body,
                                                   intake=litho_cmds.flex_intake))
        assert res.get("backend") == "hardware", res

    # the same -0.35 V + 0.2 the cockpit's setpoint_relative gets
    assert _hw_write_setdowns(client, tmp_path, send) == [pytest.approx(-0.15)]


def test_flex_setpoint_with_no_reference_is_used_as_sent(client, tmp_path):
    import functools

    from afm_litho.service.host import litho_cmds

    def send(hub):
        ids = [o.name for o in hub.pattern.objects if getattr(o, "enabled", True)]
        body = {"objects": ids, "setpoint": 0.1, **GUARD_WRITE_KW}
        res = client.portal.call(functools.partial(litho_cmds.litho_write, hub, body,
                                                   intake=litho_cmds.flex_intake))
        assert res.get("backend") == "hardware", res

    # an old script's absolute value keeps its meaning: exactly 0.1, no free air added
    assert _hw_write_setdowns(client, tmp_path, send) == [pytest.approx(0.1)]


def test_rest_absolute_setpoint_is_used_as_sent(client, tmp_path):
    def send(hub):
        body = {**_guard_write_body(hub), "setpoint": 0.1, "setpoint_reference": "absolute"}
        body.pop("setpoint_relative", None)
        r = client.post("/api/litho/write", json=body)
        assert r.json().get("backend") == "hardware", r.text

    assert _hw_write_setdowns(client, tmp_path, send) == [pytest.approx(0.1)]


def test_engaged_write_with_an_old_baseline_is_accepted_and_reads_fresh_free_air(
        client, tmp_path, monkeypatch):
    """docs/PER_OBJECT_FREE_AIR_SPEC.md "Start-of-write free-air age check dropped": a
    hardware write started ENGAGED with a baseline older than free_air_max_age_s is
    accepted, and its first object reads fresh free air before the approach."""
    from afm_litho.service.command_guard import FreeAir
    from afm_litho.service.instrument_model import Field, FieldSource

    def send(hub):
        monkeypatch.setattr(hub.host.instrument_model, "free_air_max_age_s",
                            Field(1, FieldSource.config))
        body = _guard_write_body(hub)
        hub.host.free_air = FreeAir(value_v=-0.2, ts=time.time() - 60.0, mode="contact",
                               withdrawn=True)
        r = client.post("/api/litho/write", json=body)
        assert r.json().get("backend") == "hardware", r.text

    stub = _hw_write(client, tmp_path, send, engaged=True)
    setdowns = _z_setdowns(stub)
    assert setdowns == [pytest.approx(-0.15)]           # fresh -0.35 V + 0.2
    assert stub.transcript.count("ARMFIRE") == 1


def test_write_record_carries_each_objects_free_air_and_setpoint(client, tmp_path):
    """The saved write record (writes/json/<stem>.json) lists each object's own
    free-air read and the setpoint it was written at, not only the door-time value."""
    import json

    def send(hub):
        r = client.post("/api/litho/write", json=_guard_write_body(hub))
        assert r.json().get("backend") == "hardware", r.text

    _hw_write(client, tmp_path, send)
    assert _wait(lambda: list((tmp_path / "writes" / "json").glob("*.json")))
    rec = json.loads(next((tmp_path / "writes" / "json").glob("*.json")).read_text())
    per = rec["object_setpoints"]
    assert len(per) == 1
    assert per[0]["free_air_v"] == pytest.approx(-0.35)
    assert per[0]["setpoint_v"] == pytest.approx(-0.15)
    assert rec["setpoint"] == pytest.approx(0.0)        # the door-time value, unchanged


def test_abort_during_the_first_object_settle_stops_the_write(client, tmp_path,
                                                              monkeypatch):
    """Abort stops everything, from anywhere (review of 577d456): POST /api/litho/abort
    0.3 s after the first object's XY-move FIRE, inside the 3 s settle. The run must not
    approach or fire the object afterwards."""
    import sys

    from afm_litho.service.host.safe_park import safe_park as _sp_fn
    monkeypatch.setattr(sys.modules[_sp_fn.__module__], "_VERIFY_SETTLE_S", 0.05)

    def send(hub):
        hub.host._hw_litho_abort.clear()
        r = client.post("/api/litho/write", json=_guard_write_body(hub))
        assert r.json().get("backend") == "hardware", r.text
        assert _wait(lambda: len(stub_ref[0].fires) >= 1)
        time.sleep(0.3)
        assert client.post("/api/litho/abort").json()["ok"] is True

    stub_ref = []
    from afm_litho.hardware import igor_stub
    orig_init = igor_stub.FakeIgorBridge.__init__

    def keep(self, *a, **kw):
        orig_init(self, *a, **kw)
        stub_ref.append(self)

    monkeypatch.setattr(igor_stub.FakeIgorBridge, "__init__", keep)
    from afm_litho.service.app import hub
    try:
        stub = _hw_write(client, tmp_path, send)
    finally:
        hub.host._hw_litho_abort.clear()
        hub.host._safe_park["active"] = False
    assert stub.aborts >= 1
    assert not _z_setdowns(stub)
    assert "ARMFIRE" not in stub.transcript and "LITHOCHAIN" not in stub.transcript
    assert stub.transcript.count("FIRE") == 1            # only the XY move


def test_a_dead_free_air_read_stops_the_write_with_the_reason(client, tmp_path):
    """A NaN free-air deflection before the first object stops the write before
    any approach; litho_done carries the reason and the end-of-run withdraw runs."""
    def send(hub):
        r = client.post("/api/litho/write", json=_guard_write_body(hub))
        assert r.json().get("backend") == "hardware", r.text

    stub = _hw_write(client, tmp_path, send, free_air_v=float("nan"))
    assert stub.litho_done["aborted"] is True
    assert "free-air deflection read failed on wire" in stub.litho_done["error"]
    assert not _z_setdowns(stub)
    assert "ARMFIRE" not in stub.transcript
    assert stub.stopped_loops == []                  # the withdraws re-arm, never stop
    assert len([lp for lp in stub.loops if lp[0] == 2]) == 2   # first-object + end-of-run withdraw
