"""6b2 on a live twin service: the recorder dies in the middle of a scan. The scan finishes, no
door call costs more than the intent cap, and the journal has a gap line and late entries.
Real time: the recorder is in-process, the service is a child process (as the keeper starts it)."""

from __future__ import annotations

import json
import time

from _service import Service, free_port
from test_recorder import DOOR_CAP_S, restart_recorder, wait_until

from afm_litho.recorder_main import Recorder

SCAN = {"size_um": 5.0, "pixels": 64, "lines": 128, "line_rate_hz": 40.0, "mode": "contact"}


def _entries(folder) -> list[dict]:
    (path,) = folder.glob("*.jsonl")
    return [json.loads(x) for x in path.read_text("utf-8").splitlines()]


def _timed(call, *args) -> float:
    t0 = time.monotonic()
    call(*args)
    return time.monotonic() - t0


def test_killing_the_recorder_mid_scan(tmp_path, monkeypatch):
    port = free_port()
    monkeypatch.setenv("AFM_LITHO_RECORDER_PORT", str(port))
    rec = Recorder(port)
    svc = Service(runtime=tmp_path / "rt", twin_speed=1.0).start()
    folder = tmp_path / "rt" / "journal"
    slowest = 0.0
    try:
        svc.post("/api/approach", {"setpoint": 0.2, "igain": 12.0, "mode": "contact"})
        status, _ = svc.post("/api/scan/start", SCAN)
        assert status == 200
        rec.stop()                                      # the recorder dies mid-scan
        for _ in range(8):
            slowest = max(slowest, _timed(svc.get, "/api/state"),
                          _timed(svc.post, "/api/scan/start", SCAN))  # refused: one is running
            time.sleep(0.1)
        rec = restart_recorder(port)
        wait_until(lambda: any(e["what"] == "gap" and e["src"] == "recorder"
                               for e in _entries(folder)), 20)
        wait_until(lambda: not svc.get("/api/scan/continuous")[1].get("running", False)
                   and svc.get("/api/scan/last")[0] == 200, 60)  # the scan finished
    finally:
        svc.stop()
        rec.stop()
    assert slowest < DOOR_CAP_S
    entries = _entries(folder)
    numbers = [e["n"] for e in entries if "n" in e]
    assert sorted(set(numbers)) == list(range(1, max(numbers) + 1))
    assert numbers == sorted(numbers)
    assert any(e.get("late") for e in entries)
