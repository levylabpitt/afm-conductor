"""The XY LVDT calibration is re-read from Igor at scan start and approach.

The Cypher rescales XLVDTSens / YLVDTSens on every motor move, so the values read
at connect go stale (INSTRUMENT_MODEL_SPEC 13.1 item 34). Fixture = the two rig
pairs of 2026-09-25: connect with the tip RAISED, scan with the tip ENGAGED.
"""

import asyncio

import numpy as np
import pytest
from hub_fields import owner

from afm_litho.hardware.hardware_arc import HardwareARC
from afm_litho.hardware.igor_stub import FakeIgorBridge
from afm_litho.pass_executor import raster_pass
from afm_litho.service.passes import ScanConfig

# Rig 2026-09-25, um = (V - offset) * sens: X (-3.11956 + 0.02) * 5.07983 = -15.75,
# Y (2.88476 + 0.38) * 4.75782 = 15.53 with the tip raised.
RAISED = {"XLVDTSens": 5.07983e-6, "YLVDTSens": 4.75782e-6,
          "XLVDTOffset": -0.02, "YLVDTOffset": -0.38,
          "ZLVDTSens": 5e-6, "ZLVDTOffset": 0.0}
ENGAGED = {"XLVDTSens": 4.9163e-6, "YLVDTSens": 4.59453e-6}
RAW_XY_V = (-3.11956, 2.88476)           # raw XSensor / YSensor at a vendor 15 um corner


@pytest.fixture
def rig(tmp_path):
    """A hub bound to a FakeIgorBridge that connected with the RAISED values, then
    had its GV values changed to ENGAGED (a coarse approach between connect and scan)."""
    from afm_litho.service.hub import hub
    from afm_litho.service.sessions import SessionStore
    from afm_litho.service.xy_cal import note_xy_cal

    stub = FakeIgorBridge(dict(RAISED), done_delay=0.0)
    port = stub.serve_in_thread()
    hw = HardwareARC.connect("127.0.0.1", port)
    saved = (hub.store, hub.host.hardware_arc, hub.host.hw_cal, dict(hub.host.xy_cal_status))
    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    hub.host.hardware_arc = hw
    hub.host.hw_cal = hw.start()
    note_xy_cal(hub, "connect")
    hub.host.xy_cal_status["refreshed_at"] -= 10.0       # outside the refresh debounce
    assert hub.host.hw_cal["x_lvdt_sens"] == pytest.approx(RAISED["XLVDTSens"])
    stub.values.update(ENGAGED)                       # the motor moved; Igor rescaled
    try:
        yield hub, stub
    finally:
        hub.store, hub.host.hardware_arc, hub.host.hw_cal = saved[:3]
        hub.host.xy_cal_status = saved[3]
        hw.close()
        stub.stop()


def _raster_x_volts(stub):
    """The X raster volts the scan sent: the full X wave named by its SCANCHAIN."""
    assert stub.scanchains
    return np.asarray(stub.waves[stub.scanchains[-1][0]], dtype=np.float64)


def test_scan_start_refreshes_the_cal_and_drives_with_it(rig):
    """(a) The raster volts are computed with the ENGAGED sensitivity read at scan
    start, not the RAISED one read at connect."""
    from afm_litho.service.host.scan_pump import _hw_scan_pump

    hub, stub = rig
    connect_dict = hub.host.hw_cal
    cfg = ScanConfig(size_um=5.0, pixels=32, lines=4)
    asyncio.run(_hw_scan_pump(cfg))

    assert hub.host.hw_cal is connect_dict                 # updated in place, not replaced
    assert hub.host.hw_cal["x_lvdt_sens"] == pytest.approx(ENGAGED["XLVDTSens"])
    assert hub.host.hw_cal["y_lvdt_sens"] == pytest.approx(ENGAGED["YLVDTSens"])
    assert hub.host.xy_cal_status["source"] == "scan"
    assert hub.host.xy_cal_status["error"] is None

    x = _raster_x_volts(stub)
    fresh = raster_pass(cfg.pixels, cfg.lines, cfg.size_um, hub.host.hw_cal, 1,
                        trace_retrace=cfg.trace_retrace)
    stale = raster_pass(cfg.pixels, cfg.lines, cfg.size_um,
                        dict(hub.host.hw_cal, x_lvdt_sens=RAISED["XLVDTSens"]), 1,
                        trace_retrace=cfg.trace_retrace)
    span = float(x.max() - x.min())
    assert span == pytest.approx(float(np.ptp(fresh.x_v)), rel=1e-6)
    assert span != pytest.approx(float(np.ptp(stale.x_v)), rel=1e-3)


def test_telemetry_uses_the_refreshed_cal(rig):
    """(b) The TELEM consumer converts with the live hub.hw_cal: after the refresh
    the vendor's (15, 15) Y reading comes out 15.00 um, not the stale 15.53."""
    from afm_litho.service.host import telemetry

    hub, _stub = rig
    saved = hub.host._hw_telemetry
    try:
        telemetry._hw_telemetry_frame(*RAW_XY_V, 0.0, 0.0, 5.0, 0.0)
        stale = dict(hub.host._hw_telemetry)
        asyncio.run(hub.refresh_xy_cal("approach"))
        telemetry._hw_telemetry_frame(*RAW_XY_V, 0.0, 0.0, 5.0, 0.0)
        tel = hub.host._hw_telemetry
    finally:
        hub.host._hw_telemetry = saved
    assert stale["x_um"] == pytest.approx(-15.745, abs=0.01)
    assert stale["y_um"] == pytest.approx(15.53, abs=0.01)
    assert tel["x_um"] == pytest.approx((RAW_XY_V[0] + 0.02) * 4.9163, abs=0.002)
    assert tel["y_um"] == pytest.approx(15.00, abs=0.01)


def test_api_instrument_shows_the_refreshed_values(rig, client):
    """(c) GET /api/instrument carries the current four values and the read time;
    FLEX getInstrument is the same function."""
    hub, _stub = rig
    before = client.get("/api/instrument").json()["xy_cal"]
    assert before["x_lvdt_sens"] == pytest.approx(RAISED["XLVDTSens"])
    assert before["source"] == "connect"

    asyncio.run(hub.refresh_xy_cal("approach"))
    after = client.get("/api/instrument").json()["xy_cal"]
    assert after["x_lvdt_sens"] == pytest.approx(ENGAGED["XLVDTSens"])
    assert after["y_lvdt_sens"] == pytest.approx(ENGAGED["YLVDTSens"])
    assert after["x_lvdt_offset"] == pytest.approx(-0.02)
    assert after["y_lvdt_offset"] == pytest.approx(-0.38)
    assert after["source"] == "approach"
    assert after["refreshed_at"] >= before["refreshed_at"]
    assert after["error"] is None


def test_refresh_reads_the_four_globals_over_getgv(rig):
    """Four GETGV reads, no repeated STARTUP handshake."""
    from afm_litho.service.xy_cal import refresh_xy_cal

    hub, stub = rig
    n_startup = stub.transcript.count("STARTUP")
    stub.getgvs.clear()
    refresh_xy_cal(hub, "scan")
    assert stub.getgvs == ["XLVDTSens", "YLVDTSens", "XLVDTOffset", "YLVDTOffset"]
    assert stub.transcript.count("STARTUP") == n_startup


def test_refresh_keeps_old_values_when_a_global_reads_nan(rig):
    """An unknown global (or a GV error on the rig) answers nan; nothing changes."""
    from afm_litho.service.xy_cal import refresh_xy_cal

    hub, stub = rig
    del stub.values["YLVDTOffset"]
    lines = refresh_xy_cal(hub, "scan")
    assert "REJECTED" in lines[0]
    assert hub.host.hw_cal["x_lvdt_sens"] == pytest.approx(RAISED["XLVDTSens"])
    assert hub.host.hw_cal["y_lvdt_offset"] == pytest.approx(-0.38)


def test_refresh_refuses_an_implausible_sensitivity(rig):
    """A sensitivity 100x off would put ~300 V on the XY setpoint; anything more than
    10 % from the previous value is refused and the scan runs on the old values."""
    from afm_litho.service.host.scan_pump import _hw_scan_pump

    hub, stub = rig
    stub.values["XLVDTSens"] = 5e-8
    cfg = ScanConfig(size_um=5.0, pixels=32, lines=4)
    asyncio.run(_hw_scan_pump(cfg))
    assert hub.host.hw_cal["x_lvdt_sens"] == pytest.approx(RAISED["XLVDTSens"])
    assert hub.host.hw_cal["y_lvdt_sens"] == pytest.approx(RAISED["YLVDTSens"])
    assert "x_lvdt_sens" in hub.host.xy_cal_status["error"]
    assert float(np.abs(_raster_x_volts(stub)).max()) < 10.0


def test_refresh_refuses_an_offset_jump(rig):
    from afm_litho.service.xy_cal import refresh_xy_cal

    hub, stub = rig
    stub.values["YLVDTOffset"] = 0.5                  # 0.88 V from -0.38
    lines = refresh_xy_cal(hub, "scan")
    assert "REJECTED" in lines[0] and "y_lvdt_offset" in lines[0]
    assert hub.host.hw_cal["y_lvdt_offset"] == pytest.approx(-0.38)
    assert hub.host.hw_cal["x_lvdt_sens"] == pytest.approx(RAISED["XLVDTSens"])


def test_failed_read_keeps_old_values_and_the_scan_still_runs(rig, monkeypatch):
    from afm_litho.service.host.scan_pump import _hw_scan_pump

    hub, stub = rig

    def boom(name, *, timeout=None):
        raise TimeoutError(f"no reply for {name}")

    monkeypatch.setattr(hub.host.hardware_arc, "get_gv", boom)
    cfg = ScanConfig(size_um=5.0, pixels=32, lines=4)
    asyncio.run(_hw_scan_pump(cfg))
    assert hub.host.hw_cal["x_lvdt_sens"] == pytest.approx(RAISED["XLVDTSens"])
    assert "GETGV read failed" in hub.host.xy_cal_status["error"]
    assert hub.host.xy_cal_status["source"] == "connect"
    assert stub.scanchains                                    # the scan ran


def test_stalled_getgv_times_out_and_keeps_old_values(rig, monkeypatch):
    from afm_litho.service import xy_cal

    hub, stub = rig
    monkeypatch.setattr(xy_cal, "REFRESH_TIMEOUT_S", 0.2)
    stub.stall_getvalue = True
    lines = xy_cal.refresh_xy_cal(hub, "scan")
    stub.stall_getvalue = False
    assert "FAILED" in lines[0] and "GETGV" in lines[0]
    assert hub.host.hw_cal["x_lvdt_sens"] == pytest.approx(RAISED["XLVDTSens"])


def test_refresh_is_debounced_within_one_second(rig):
    """The continuous loop's re-reference approach and the frame's scan start come
    back to back; the second read is skipped."""
    from afm_litho.service.xy_cal import refresh_xy_cal

    hub, stub = rig
    refresh_xy_cal(hub, "approach")
    stub.getgvs.clear()
    assert refresh_xy_cal(hub, "scan") == []
    assert stub.getgvs == []
    assert hub.host.xy_cal_status["source"] == "approach"


def test_refresh_logs_the_percent_delta(rig):
    from afm_litho.service.xy_cal import refresh_xy_cal

    hub, stub = rig
    lines = refresh_xy_cal(hub, "scan")
    assert len(lines) == 1
    assert "x_lvdt_sens" in lines[0] and "-3.22 %" in lines[0]
    assert "y_lvdt_sens" in lines[0] and "-3.43 %" in lines[0]
    assert "offset" not in lines[0]                   # unchanged offsets are not logged
    hub.host.xy_cal_status["refreshed_at"] -= 10.0         # past the debounce
    stub.getgvs.clear()
    assert refresh_xy_cal(hub, "scan") == []          # no change, no line
    assert len(stub.getgvs) == 4                      # it did read


def test_refresh_rejects_a_zero_sensitivity_and_keeps_the_old_values(rig):
    """A zero or non-finite sensitivity would put inf volts on the XY drive."""
    from afm_litho.service.xy_cal import refresh_xy_cal

    hub, stub = rig
    stub.values["XLVDTSens"] = 0.0
    lines = refresh_xy_cal(hub, "scan")
    assert "REJECTED" in lines[0]
    assert hub.host.hw_cal["x_lvdt_sens"] == pytest.approx(RAISED["XLVDTSens"])
    assert hub.host.hw_cal["y_lvdt_sens"] == pytest.approx(RAISED["YLVDTSens"])
    assert "x_lvdt_sens=0" in hub.host.xy_cal_status["error"]


def test_refresh_is_a_noop_on_the_twin():
    from types import SimpleNamespace

    from afm_litho.service.xy_cal import refresh_xy_cal

    fake = SimpleNamespace(hardware_arc=None, hw_cal={"x_lvdt_sens": 1.0})
    fake.host = fake  # fake: host fields live on the fake itself
    assert refresh_xy_cal(fake, "scan") == []
    assert fake.hw_cal == {"x_lvdt_sens": 1.0}


def test_approach_refreshes_the_cal_before_engaging(rig):
    """The approach handler re-reads the calibration at its top, so a scan image
    taken after a coarse approach is scaled with the post-move values."""
    from _im_fixtures_guard import seed_free_air

    from afm_litho.service.routers.scan import approach

    hub, stub = rig
    stub.values.update({"Deflection": 1.5, "height": 20.0})
    saved = {k: getattr(owner(hub, k), k, None) for k in
             ("last_scan_config", "z_loop", "imaging_mode", "engaged", "free_air")}
    hub.host.last_scan_config = {"mode": "contact"}
    hub.host.scan_repeat["config"] = None
    hub.host.imaging_mode = "contact"      # the instrument's mode; unset, approach must read it back
    seed_free_air(hub, 1.0, mode="contact")
    try:
        asyncio.run(approach({"setpoint": 1.5, "pgain": 0.0, "igain": 10.0,
                              "settle_s": 0.0}))
    finally:
        for k, v in saved.items():
            setattr(owner(hub, k), k, v)
    assert hub.host.hw_cal["x_lvdt_sens"] == pytest.approx(ENGAGED["XLVDTSens"])
    assert hub.host.xy_cal_status["source"] == "approach"
    assert any(lp[0] == 2 for lp in stub.loops)       # the engage itself still ran


def test_hardware_write_refreshes_the_cal_before_the_worker(rig, client, monkeypatch):
    """The write's LithographyRun snapshots hub.hw_cal when its worker starts, so the
    write start re-reads it first (source "write")."""
    import time

    from _im_fixtures_guard import GUARD_WRITE_KW, seed_free_air

    from afm_litho.hardware import lithography_run
    from afm_litho.litho.compiler import LithoObject, Pattern

    hub, _stub = rig
    seen = {}
    real_init = lithography_run.LithographyRun.__init__

    def spy_init(self, *a, **kw):
        real_init(self, *a, **kw)
        seen["x_sens_at_run_build"] = hub.host.hw_cal["x_lvdt_sens"]

    monkeypatch.setattr(lithography_run.LithographyRun, "__init__", spy_init)
    saved = {k: getattr(owner(hub, k), k, None) for k in
             ("pattern", "written_objects", "z_loop", "imaging_mode", "free_air")}
    hub.pattern = Pattern(name="xy-cal-write", objects=[
        LithoObject("wire", "path", [(-1.0, 0.0), (1.0, 0.0)], voltage=6.0,
                    speed_um_s=8.0, interpolation_um=0.1)])
    hub.host.written_objects = set()
    hub.host.z_loop = {"mode": "contact"}
    hub.host.imaging_mode = "contact"
    seed_free_air(hub)
    try:
        r = client.post("/api/litho/write", json={
            "objects": ["wire"], "setpoint_relative": 0.2, **GUARD_WRITE_KW})
        assert r.json().get("backend") == "hardware", r.text
        assert hub.host.xy_cal_status["source"] == "write"
        assert hub.host.hw_cal["x_lvdt_sens"] == pytest.approx(ENGAGED["XLVDTSens"])
        end = time.monotonic() + 20.0
        while hub.host.litho_thread is not None and hub.host.litho_thread.is_alive():
            assert time.monotonic() < end
            time.sleep(0.02)
        assert seen["x_sens_at_run_build"] == pytest.approx(ENGAGED["XLVDTSens"])
    finally:
        for k, v in saved.items():
            setattr(owner(hub, k), k, v)


_PASS_VERBS = ("FIRE", "ARMFIRE", "LITHOCHAIN", "SCANCHAIN")


def test_abort_during_the_write_refresh_refuses_the_write(rig, client, monkeypatch):
    """An abort or park landing during the refresh's bridge reads must not be wiped
    by the write start: the write is refused and nothing fires."""
    from _im_fixtures_guard import GUARD_WRITE_KW, seed_free_air

    from afm_litho.litho.compiler import LithoObject, Pattern
    from afm_litho.service.host.safe_park import signal_run_abort

    hub, stub = rig

    async def refresh_then_abort(source):
        signal_run_abort(hub)

    monkeypatch.setattr(hub, "refresh_xy_cal", refresh_then_abort)
    saved = {k: getattr(owner(hub, k), k, None) for k in
             ("pattern", "written_objects", "z_loop", "imaging_mode", "free_air")}
    hub.pattern = Pattern(name="xy-cal-abort", objects=[
        LithoObject("wire", "path", [(-1.0, 0.0), (1.0, 0.0)], voltage=6.0,
                    speed_um_s=8.0, interpolation_um=0.1)])
    hub.host.written_objects = set()
    hub.host.z_loop = {"mode": "contact"}
    hub.host.imaging_mode = "contact"
    seed_free_air(hub)
    try:
        r = client.post("/api/litho/write", json={
            "objects": ["wire"], "setpoint_relative": 0.2, **GUARD_WRITE_KW})
        body = r.json()
        assert body.get("reason") == "aborted_during_start", r.text
        assert not body.get("ok")
        assert not any(v in _PASS_VERBS for v in stub.transcript)
        assert hub.host.litho_thread is None or not hub.host.litho_thread.is_alive()
    finally:
        for k, v in saved.items():
            setattr(owner(hub, k), k, v)


def test_probe_marks_an_ipf_without_getgv_unsupported(rig, client, monkeypatch):
    """A pre-GETGV ipf never answers; the connect probe records it once and later
    refreshes skip the read instead of each waiting out the timeout."""
    from afm_litho.hardware.bridge_router import BridgeReadTimeout
    from afm_litho.service.xy_cal import UNSUPPORTED_MSG, probe_getgv, refresh_xy_cal

    hub, _stub = rig
    calls = []

    def no_reply(name, *, timeout=None):
        calls.append(name)
        raise BridgeReadTimeout(f"expected VALUE reply for {name!r}")

    monkeypatch.setattr(hub.host.hardware_arc, "get_gv", no_reply)
    assert probe_getgv(hub, hub.host.hardware_arc) == UNSUPPORTED_MSG
    calls.clear()
    assert refresh_xy_cal(hub, "scan") == []
    assert calls == []
    assert hub.host.hw_cal["x_lvdt_sens"] == pytest.approx(RAISED["XLVDTSens"])
    x = client.get("/api/instrument").json()["xy_cal"]
    assert x["unsupported"] is True and x["error"] == UNSUPPORTED_MSG


def test_probe_passes_on_an_ipf_with_getgv(rig):
    from afm_litho.service.xy_cal import probe_getgv

    hub, stub = rig
    assert probe_getgv(hub, hub.host.hardware_arc) is None
    assert hub.host.xy_cal_status["unsupported"] is False
    assert stub.getgvs[-1] == "XLVDTSens"


def test_large_change_rejection_says_how_to_accept_it(rig):
    from afm_litho.service.xy_cal import refresh_xy_cal

    hub, stub = rig
    stub.values["XLVDTSens"] = RAISED["XLVDTSens"] * 1.2
    lines = refresh_xy_cal(hub, "scan")
    assert "reconnect the bridge (restart the service)" in lines[0]
    assert "reconnect the bridge (restart the service)" in hub.host.xy_cal_status["error"]


def test_debounce_does_not_apply_after_a_failed_read(rig):
    from afm_litho.service.xy_cal import refresh_xy_cal

    hub, stub = rig
    refresh_xy_cal(hub, "approach")                   # success: stamps refreshed_at
    stub.values["XLVDTSens"] = 0.0                    # the next read is refused
    hub.host.xy_cal_status["refreshed_at"] -= 10.0
    assert "REJECTED" in refresh_xy_cal(hub, "scan")[0]
    hub.host.xy_cal_status["refreshed_at"] += 10.0         # recent success on record ...
    stub.values["XLVDTSens"] = ENGAGED["XLVDTSens"]
    stub.getgvs.clear()
    refresh_xy_cal(hub, "scan")                       # ... but the last read failed
    assert len(stub.getgvs) == 4
    assert hub.host.xy_cal_status["error"] is None


def test_park_during_the_write_guards_refuses_the_write(rig, client, monkeypatch):
    """The SUM-ceiling read blocks the loop thread while a park can land from the
    watchdog or telemetry guard; the write must not clear that signal and engage."""
    from _im_fixtures_guard import GUARD_WRITE_KW, seed_free_air

    from afm_litho.litho.compiler import LithoObject, Pattern
    from afm_litho.service.host import engage
    from afm_litho.service.host.safe_park import signal_run_abort

    hub, stub = rig
    real_read = engage.read_detector_sum

    def park_then_read(h):
        signal_run_abort(hub)
        return real_read(h)

    monkeypatch.setattr(engage, "read_detector_sum", park_then_read)
    stub.values["DetectorSum"] = 5.0
    saved = {k: getattr(owner(hub, k), k, None) for k in
             ("pattern", "written_objects", "z_loop", "imaging_mode", "free_air")}
    hub.pattern = Pattern(name="xy-cal-guard-park", objects=[
        LithoObject("wire", "path", [(-1.0, 0.0), (1.0, 0.0)], voltage=6.0,
                    speed_um_s=8.0, interpolation_um=0.1)])
    hub.host.written_objects = set()
    hub.host.z_loop = {"mode": "contact"}
    hub.host.imaging_mode = "contact"
    seed_free_air(hub)
    try:
        r = client.post("/api/litho/write", json={
            "objects": ["wire"], "setpoint_relative": 0.2, **GUARD_WRITE_KW})
        body = r.json()
        assert body.get("reason") == "aborted_during_start", r.text
        assert not any(v in _PASS_VERBS for v in stub.transcript)
        assert hub.host.litho_thread is None or not hub.host.litho_thread.is_alive()
        wi = getattr(hub.host, "write_interlocks", None)
        assert wi is None or not wi.armed
    finally:
        for k, v in saved.items():
            setattr(owner(hub, k), k, v)
