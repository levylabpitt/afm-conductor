"""The litho cockpit paints the NEWEST scan under its design overlay.

That means two things must hold: (1) the on-disk cache round-trips a frame --
newest wins, height comes back flat/nm/right length, geometry preserved; and
(2) GET /api/scan/last 404s before anything is cached and serves the wire
contract dict once a frame is saved into the hub's runtime dir.
"""


import numpy as np
from fastapi.testclient import TestClient

from afm_litho.service.scan_cache import load_latest_scan_frame, save_scan_frame


def _client():
    from afm_litho.service.app import app
    return TestClient(app)


def test_save_load_round_trip_newest_wins(tmp_path):
    older = np.arange(2 * 3, dtype=float).reshape(3, 2)          # rows=3, pixels=2
    newer = (np.arange(2 * 3, dtype=float).reshape(3, 2) + 100.0)
    save_scan_frame(tmp_path, {
        "stamp": "scan-1000000000001", "pixels": 2, "rows": 3,
        "size_um": 15.0, "x_offset_um": 1.0, "y_offset_um": -2.0,
        "angle_deg": 0.0, "height": older})
    save_scan_frame(tmp_path, {
        "stamp": "scan-1000000000002", "pixels": 2, "rows": 3,
        "size_um": 20.0, "x_offset_um": 3.0, "y_offset_um": 4.0,
        "angle_deg": 0.0, "height": newer})

    f = load_latest_scan_frame(tmp_path)
    assert f is not None
    # newest by stamp is the "newer" frame -> its geometry
    assert f["pixels"] == 2 and f["rows"] == 3
    assert f["size_um"] == 20.0
    assert f["x_offset_um"] == 3.0 and f["y_offset_um"] == 4.0
    # height is flat, row-major (index = slow_idx*pixels + i), length pixels*rows
    assert isinstance(f["height"], list) and len(f["height"]) == 2 * 3
    assert f["height"] == newer.reshape(-1).tolist()


def test_load_none_when_empty(tmp_path):
    assert load_latest_scan_frame(tmp_path) is None            # dir missing
    (tmp_path / "scans").mkdir()
    assert load_latest_scan_frame(tmp_path) is None            # dir empty


def test_hw_scan_pump_caches_the_completed_frame(tmp_path):
    """A hardware(-twin) scan driven through _hw_scan_pump must (1) populate the
    /api/scan/last cache with a frame of the right pixels/rows and a flat height of
    length rows*pixels, and (2) still persist the RAW per-channel capture separately
    (the write/targeting path). The cached height is DISPLAY-only (flattened)."""
    import asyncio

    from afm_litho.hardware.hardware_arc import HardwareARC
    from afm_litho.hardware.igor_stub import FakeIgorBridge
    from afm_litho.service.captures import load_capture
    from afm_litho.service.hub import hub
    from afm_litho.service.passes import ScanConfig
    from afm_litho.service.host.scan_pump import _hw_scan_pump
    from afm_litho.service.sessions import SessionStore

    cal = {"x_lvdt_sens": 7.46e-6, "x_lvdt_offset": -0.63,
           "y_lvdt_sens": 7.04e-6, "y_lvdt_offset": 0.07,
           "z_lvdt_sens": 5e-6, "z_lvdt_offset": 0.0}
    cfg = ScanConfig(size_um=5.0, pixels=32, lines=4)

    stub = FakeIgorBridge({}, done_delay=0.0)
    port = stub.serve_in_thread()
    hw = HardwareARC.connect("127.0.0.1", port)

    prev = (hub.store, hub.host.hardware_arc, hub.host.hw_cal)
    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    hub.host.hardware_arc = hw
    hub.host.hw_cal = cal
    try:
        asyncio.run(_hw_scan_pump(cfg))

        # (1) the cache now serves a right-shaped frame
        f = load_latest_scan_frame(hub.store.path.parent)
        assert f is not None
        assert f["pixels"] == 32 and f["rows"] == 4
        assert len(f["height"]) == 32 * 4
        assert f["size_um"] == 5.0

        # (2) the RAW capture is still persisted separately, under the session id
        rec = hub.store.list(limit=1)[0]
        raw = load_capture(hub.store.path.parent, rec["id"])
        assert "ZSensor" in raw and len(raw["ZSensor"]) > 0
    finally:
        hub.store, hub.host.hardware_arc, hub.host.hw_cal = prev
        hw.close()
        stub.stop()


def test_twin_scan_writes_shared_stem_png_and_json_sidecar(tmp_path):
    """Slice A end-to-end (docs/PLAN.md "Layout and naming"): a completed TWIN
    scan must leave npz/png/json under `scans/` sharing ONE stem, the PNG must
    decode to (rows, pixels), and the FLEX-visible terminal record
    (`getScanResult.files`) must list `png`/`json`."""
    import json
    import struct
    import time as _time

    from afm_litho.service.hub import hub
    from afm_litho.service.sessions import SessionStore

    prev_store = hub.store
    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    try:
        run_id = None
        with _client() as c, c.websocket_connect("/ws") as ws:
            ws.send_text("ping")
            for _ in range(200):
                if ws.receive_json().get("type") == "telemetry":
                    break
            r = c.post("/api/scan/start", json={
                "size_um": 5.0, "pixels": 8, "lines": 4,
                "line_rate_hz": 400.0, "mode": "contact"})
            assert r.json().get("ok") is True, r.text
            run_id = r.json().get("run_id")
            end = _time.monotonic() + 20.0
            done = False
            while _time.monotonic() < end:
                if ws.receive_json().get("type") == "scan_done":
                    done = True
                    break
            assert done, "scan never finished"

        png_files = list((tmp_path / "scans" / "png").glob("*_height.png"))
        assert len(png_files) == 1, png_files
        stem = png_files[0].name[:-len("_height.png")]
        assert (tmp_path / "scans" / "npz" / f"{stem}.height.npz").is_file()
        json_path = tmp_path / "scans" / "json" / f"{stem}.json"
        assert json_path.is_file()

        sidecar = json.loads(json_path.read_text(encoding="utf-8"))
        assert sidecar["row0"] == "bottom"
        assert sidecar["channels"] == ["height"]
        assert sidecar["pixels"] == 8 and sidecar["rows"] == 4
        assert sidecar["stamp"] == stem

        data = png_files[0].read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        width, height = struct.unpack(">II", data[16:24])
        assert (width, height) == (8, 4)

        assert run_id is not None
        rec = hub.runs.get(run_id)
        assert rec is not None and rec.completed
        assert "png" in rec.files and rec.files["png"].endswith(f"{stem}_height.png")
        assert "json" in rec.files and rec.files["json"].endswith(f"{stem}.json")
    finally:
        hub.store = prev_store


def test_scan_last_404_then_frame(tmp_path):
    from afm_litho.service.hub import hub
    from afm_litho.service.sessions import SessionStore

    prev_store = hub.store
    hub.store = SessionStore(tmp_path / "sessions.jsonl")       # fresh runtime dir
    try:
        with _client() as c:
            r = c.get("/api/scan/last")
            assert r.status_code == 404, r.text

            save_scan_frame(hub.store.path.parent, {
                "stamp": "scan-1700000000000", "pixels": 4, "rows": 2,
                "size_um": 10.0, "x_offset_um": 0.5, "y_offset_um": -0.5,
                "angle_deg": 0.0,
                "height": np.arange(4 * 2, dtype=float).reshape(2, 4)})

            r = c.get("/api/scan/last")
            assert r.status_code == 200, r.text
            f = r.json()
            assert f["pixels"] == 4 and f["rows"] == 2
            assert f["size_um"] == 10.0
            assert f["x_offset_um"] == 0.5 and f["y_offset_um"] == -0.5
            assert f["angle_deg"] == 0.0
            assert len(f["height"]) == 4 * 2
            assert f["height"] == list(range(8))
    finally:
        hub.store = prev_store


def test_scan_last_sanitizes_nan_to_null(tmp_path):
    """A partial/in-progress cached frame carries NaN for unscanned pixels; the
    endpoint must map non-finite floats to null (not 500 via allow_nan=False)."""
    from afm_litho.service.hub import hub
    from afm_litho.service.sessions import SessionStore

    prev_store = hub.store
    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    try:
        with _client() as c:
            height = np.array([[0.0, np.nan, 2.0, np.inf],
                               [-np.inf, 5.0, np.nan, 7.0]], dtype=float)
            save_scan_frame(hub.store.path.parent, {
                "stamp": "scan-1700000000001", "pixels": 4, "rows": 2,
                "size_um": 10.0, "x_offset_um": 0.0, "y_offset_um": 0.0,
                "angle_deg": 0.0, "height": height})

            r = c.get("/api/scan/last")
            assert r.status_code == 200, r.text       # was a 500 before the fix
            f = r.json()
            # non-finite pixels come back as null; finite ones unchanged
            assert f["height"] == [0.0, None, 2.0, None, None, 5.0, None, 7.0]
    finally:
        hub.store = prev_store
