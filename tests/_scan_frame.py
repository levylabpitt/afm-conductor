"""Shared setup: give the hub a scan frame the way an operator does, by scanning."""

from __future__ import annotations

import time

TINY_SCAN = {"size_um": 40.0, "pixels": 16, "lines": 4, "line_rate_hz": 200.0,
             "mode": "contact"}


def run_tiny_scan(client, timeout=30.0):
    """Run a tiny twin scan through the public API and wait for it to finish, so the
    hub has a scan frame (`last_scan_config`) for a design import to be placed against."""
    from afm_litho.service.app import hub
    r = client.post("/api/scan/start", json=TINY_SCAN).json()
    assert r.get("ok") is True, r
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        t = hub.host._scan_task
        if t is not None and t.done() and hub.state_label() == "idle":
            return
        time.sleep(0.05)
    raise AssertionError("the setup scan never finished")

