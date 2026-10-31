"""The service's twin write and scan run on the twin plugin (slice 4.3b).

Checks the seams between the service and `instrument/twin.py`: the sample converter, the one
plugin per sample, the transit speed the worker gives, a scan's frame and its stop, and the
wire a write leaves in a later scan. The plugin itself is tested in test_instrument_twin.py.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from _im_fixtures_guard import GUARD_WRITE_KW, seed_free_air

from afm_litho import clock
from afm_litho.litho.compiler import LithoObject, Pattern
from afm_litho.service.twin_plugin import plugin, sample_to_npz

SMALL_SCAN = {"size_um": 4.0, "pixels": 16, "lines": 8, "line_rate_hz": 200.0,
              "mode": "contact"}


@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


@pytest.fixture(autouse=True)
def _quiet(hub, tmp_path):
    from afm_litho.service.sessions import SessionStore
    prev_store, prev_speed = hub.store, hub.host.speed
    prev_frame, hub.litho_frame = getattr(hub, "litho_frame", None), None   # a pinned frame moves wires
    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    yield
    _wait(lambda: hub.state_label() == "idle", 60.0)
    hub.store, hub.host.speed = prev_store, prev_speed
    hub.litho_frame = prev_frame
    hub.host.written_objects = set()


def _wait(cond, timeout=30.0, interval=0.05) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return False


def _scan(client, hub, body=None):
    r = client.post("/api/scan/start", json=body or SMALL_SCAN).json()
    assert r.get("ok") is True, r
    # the frame is cached after the pass ends: wait for the run record to close
    assert _wait(lambda: hub.runs.get(r["run_id"]).finished is not None, 60.0)


def test_the_converter_keeps_the_maps_orientation(hub, tmp_path):
    """The IBW sample is indexed [x, y] in metres; the plugin's file is rows (Y) by pixels
    (X) in nm, centred on the origin, as wide as the sample."""
    sample = hub.host.world.sample
    path = sample_to_npz(sample, tmp_path / "twin" / "s.npz")
    with np.load(path) as z:
        height, size_um = z["height"], float(z["size_um"])
        assert (float(z["x_offset_um"]), float(z["y_offset_um"]), float(z["angle_deg"])) == (0, 0, 0)
    nx, ny = sample.height_map.shape
    assert height.shape == (ny, nx) and size_um == pytest.approx(sample.extent * 1e6)
    assert height[3, 5] == pytest.approx(sample.height_map[5, 3] * 1e9)
    from afm_litho.instrument.twin import TwinInstrument
    twin = TwinInstrument(sample=path)
    x_um = (5 + 0.5) / nx * size_um - size_um / 2
    y_um = (3 + 0.5) / ny * size_um - size_um / 2
    assert twin.sample.height_nm([x_um], [y_um])[0] == pytest.approx(height[3, 5])


def test_one_plugin_per_sample_and_a_new_sample_starts_a_new_one(client, hub):
    first = plugin(hub)
    assert plugin(hub) is first
    from afm_litho.service.deps import DEFAULT_SAMPLE
    assert client.post("/api/sample/load", json={"file": DEFAULT_SAMPLE}).json().get("ok")
    assert hub.host._twin_plugin is None                    # rebuilt from the new sample on use
    assert plugin(hub) is not first


def test_a_hardware_bridge_drops_the_plugin(hub):
    plugin(hub)
    hub.host.twin_scan_angle_deg = 12.0
    hub.forget_twin_state()
    assert hub.host._twin_plugin is None and hub.host.twin_scan_angle_deg is None


def test_the_worker_gives_the_twin_the_hardware_transit_speed(client, hub):
    """The twin has no speed of its own; the settings travel in the command, and the value
    is the rig's `MOVE_SPEED_UM_S` (5 um/s), not the 50 um/s the old twin used."""
    from afm_litho.hardware.xy_transit import MOVE_SPEED_UM_S
    twin = plugin(hub)
    seen = []
    real_write = twin.write
    twin.write = lambda command: (seen.append(command.transit_speed_um_s), real_write(command))[1]
    try:
        hub.pattern = Pattern(name="one", objects=[LithoObject(
            "a", "path", [(0.0, 0.0), (0.5, 0.0)], voltage=6.0, speed_um_s=5.0,
            interpolation_um=0.05)])
        hub.host.written_objects = set()
        seed_free_air(hub)
        r = client.post("/api/litho/write", json={
            "objects": ["a"], "setpoint_relative": 0.2, **GUARD_WRITE_KW}).json()
        assert r.get("ok") is True, r
        assert _wait(lambda: seen and hub.state_label() == "idle", 60.0)
    finally:
        del twin.write
    assert seen == [MOVE_SPEED_UM_S] == [5.0]


def test_a_twin_scan_delivers_every_line_and_caches_the_frame(client, hub):
    from afm_litho.service.scan_cache import load_latest_scan_frame
    twin = plugin(hub)
    with client.websocket_connect("/ws") as ws:
        ws.send_text("ping")
        r = client.post("/api/scan/start", json=SMALL_SCAN).json()
        assert r.get("ok") is True, r
        lines, done = [], None
        while done is None:
            m = ws.receive_json()
            if m["type"] == "line":
                lines.append(m)
            elif m["type"] == "scan_done":
                done = m
    assert done["aborted"] is False
    assert [m["row"] for m in lines] == list(range(8))
    assert all(len(m["height"]) == 16 and m["errName"] == "Deflection" for m in lines)
    frame = load_latest_scan_frame(hub.store.path.parent)
    assert frame is not None and np.isfinite(frame["height"]).all()
    # the cached rows are the sample as the plugin reads it, in nm
    from afm_litho.instrument.twin import raster_points
    from afm_litho.run.model import ScanFrame
    scan_frame = ScanFrame("t", 4.0, 8, 16, 0.0, (0.0, 0.0), False, "up", 200.0, False)
    gx, gy = raster_points(scan_frame)
    cached = np.asarray(frame["height"], dtype=float).reshape(8, 16)
    assert np.allclose(cached, twin.sample.height_nm(gx, gy), atol=1e-2)
    assert hub.host.twin_scan_angle_deg == 0.0


def test_a_scan_in_current_mode_is_refused_on_the_twin(client, hub):
    r = client.post("/api/scan/start", json={**SMALL_SCAN, "mode": "current"}).json()
    assert "current" in r.get("error", ""), r


def test_a_written_wire_shows_in_the_next_scan_as_two_nanometres(client, hub):
    twin = plugin(hub)
    twin.sample.overlay[:] = False
    # fine enough that a 100 nm wire is under some scan lines (the sample's pixel is 118 nm)
    big = {"size_um": 4.0, "pixels": 64, "lines": 64, "line_rate_hz": 200.0,
           "mode": "contact"}
    _scan(client, hub, big)
    before = np.array(_cached_height(hub))
    hub.pattern = Pattern(name="w", objects=[LithoObject(
        "a", "path", [(-2.0, 0.0), (2.0, 0.0)], voltage=6.0, speed_um_s=50.0,
        interpolation_um=0.05)])
    hub.host.written_objects = set()
    seed_free_air(hub)
    r = client.post("/api/litho/write", json={
        "objects": ["a"], "setpoint_relative": 0.2, **GUARD_WRITE_KW}).json()
    assert r.get("ok") is True, r
    assert _wait(lambda: hub.state_label() == "idle", 60.0)
    assert _wait(lambda: twin.sample.overlay.any(), 10.0), r
    _scan(client, hub, big)
    after = np.array(_cached_height(hub))
    raised = after - before
    assert raised.max() == pytest.approx(2.0, abs=1e-2)
    assert (raised > 1.0).any() and (np.abs(raised[raised < 1.0]) < 1e-2).all()


def _cached_height(hub):
    from afm_litho.service.scan_cache import load_latest_scan_frame
    return load_latest_scan_frame(hub.store.path.parent)["height"]


def test_a_stop_during_a_twin_scan_withdraws_the_plugins_tip(client, hub):
    hub.host.speed = 1.0
    body = {**SMALL_SCAN, "line_rate_hz": 2.0, "lines": 20}
    r = client.post("/api/scan/start", json=body).json()
    assert r.get("ok") is True, r
    twin = plugin(hub)
    assert _wait(lambda: twin._activity == "scanning", 30.0)
    assert client.post("/api/scan/abort", json={}).json()["ok"] is True
    assert _wait(lambda: hub.host._scan_task.done(), 30.0)
    assert twin.frame_at(clock.now() + 3.0).z_drive_v == 0.0       # withdrawn
    record = hub.runs.get(r["run_id"])
    assert record.aborted is True


def test_plugin_called_from_many_threads_builds_one_instance(tmp_path, monkeypatch):
    """Two threads (the scan pump and a caller) must never each build a TwinInstrument."""
    import threading
    from types import SimpleNamespace

    from afm_litho.service import twin_plugin as tp

    built = []

    class SlowTwin:
        def __init__(self, **kwargs):
            time.sleep(0.05)               # holds the check-then-build window open
            built.append(self)

        def listen(self, *args):
            pass

    monkeypatch.setattr(tp, "TwinInstrument", SlowTwin)
    monkeypatch.setattr(tp, "sample_to_npz", lambda sample, dest: dest)
    hub = SimpleNamespace(world=SimpleNamespace(sample=object()), _twin_plugin=None,
                          _twin_plugin_sample=None, sample_name="s",
                          store=SimpleNamespace(path=tmp_path / "sessions.jsonl"))
    hub.host = hub  # fake: host fields live on the fake itself
    n = 8
    barrier, got = threading.Barrier(n), []

    def call():
        barrier.wait()
        got.append(tp.plugin(hub))

    threads = [threading.Thread(target=call) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(built) == 1
    assert len(got) == n and all(g is got[0] for g in got)
