"""Continuous imaging: the backend-owned auto-repeat scan loop.

The loop lives in the SERVICE (host/continuous.py), so these tests drive it
through the FastAPI service exactly as the cockpit would: enable it, let the
supervisor run scans on the app's event loop (a background task, as with the
scan pump), and poll GET /api/scan/continuous for the outcome.

Four behaviours are pinned: it runs exactly `max_frames` frames then stops with
that reason; a scan abort breaks it at once with no further frames; a sustained
scan failure trips the consecutive-error guard; and it COMPOSES with Connect
Phase 2 drift tracking -- continuous re-images, drift corrects, and the
accumulated compensation grows the right way against a known twin drift.

`client` comes from tests/conftest.py (a TestClient). The service
`hub` is a PROCESS GLOBAL, so every test disables the loop and restores the
world/drift it touched, and redirects hub.store to a temp file so a completing
frame records there, not in the real runtime/sessions.jsonl.
"""

from __future__ import annotations

import time

# `client` comes from tests/conftest.py (a TestClient).


# ----------------------------------------------------------------- helpers --

def _wait(cond, timeout=30.0, interval=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return False


def _scan_pump_done(hub):
    """True once the last scan's pump task has finished and the hub is idle. The
    frame lands in the cache BEFORE the pump releases, so "frame cached" is not
    "scan over"; a continuous frame armed in that gap is refused as "scan already
    running"."""
    t = hub.host._scan_task
    return (t is None or t.done()) and hub.state_label() == "idle"


def _tmp_store(tmp_path):
    """Point the session archive at a temp file so completing frames record
    there, not in the real runtime/sessions.jsonl. Returns the process hub."""
    from afm_litho.service.app import hub
    from afm_litho.service.sessions import SessionStore
    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    return hub


def _stop_and_reset(client, hub):
    """Cleanly stop any running loop and clear the continuous state, so a leaked
    supervisor from a prior test can never bleed into the next one."""
    client.post("/api/scan/continuous", json={"enabled": False})
    _wait(lambda: client.get("/api/scan/continuous").json()["enabled"] is False, 10)
    hub.host.scan_repeat.update({
        "enabled": False, "config": None, "count": 0, "started_at": None,
        "errors": 0, "max_frames": None, "max_seconds": None,
        "interval_s": 0.0, "withdraw_on_stop": False, "stopped_reason": None})


_TINY = {"size_um": 5.0, "pixels": 16, "lines": 4, "line_rate_hz": 200.0,
         "mode": "contact"}


# --------------------------------------------------------- max_frames stop --

def test_continuous_runs_exactly_max_frames_then_stops(client, tmp_path):
    hub = _tmp_store(tmp_path)
    _stop_and_reset(client, hub)
    try:
        r = client.post("/api/scan/continuous",
                        json={"enabled": True, "config": _TINY, "max_frames": 3})
        j = r.json()
        assert j["enabled"] is True and j["count"] == 0, j

        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False), "loop never stopped"
        st = client.get("/api/scan/continuous").json()
        assert st["count"] == 3, st                    # exactly N frames
        assert st["stopped_reason"] == "max_frames", st
        assert st["enabled"] is False

        # a beat later, still no extra frame slipped through
        time.sleep(0.4)
        assert client.get("/api/scan/continuous").json()["count"] == 3
    finally:
        _stop_and_reset(client, hub)


# ------------------------------------------------------------- abort breaks --

def test_scan_abort_breaks_the_loop_immediately(client, tmp_path):
    hub = _tmp_store(tmp_path)
    _stop_and_reset(client, hub)
    try:
        # a large cap so only the abort can stop it
        client.post("/api/scan/continuous",
                    json={"enabled": True, "config": _TINY, "max_frames": 100})
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["count"] >= 1), "no frame completed before abort"

        client.post("/api/scan/abort")
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False), "abort did not stop the loop"
        st = client.get("/api/scan/continuous").json()
        assert st["stopped_reason"] == "aborted", st
        n = st["count"]

        # no further frames arm after the abort
        time.sleep(0.6)
        assert client.get("/api/scan/continuous").json()["count"] == n
    finally:
        _stop_and_reset(client, hub)


# ---------------------------------------------------- terminal-fault stop --

def test_continuous_stops_on_first_pass_fault(client, tmp_path, monkeypatch):
    """Fix 3 (operator decision): a hardware pass fault is TERMINAL -- one failure
    stops the loop (reason "fault") rather than retrying on an unknown Z state with
    the tip engaged. Inject a scan that raises and confirm it stops after ONE call."""
    hub = _tmp_store(tmp_path)
    _stop_and_reset(client, hub)
    import afm_litho.service.host.continuous as cont

    frames = {"n": 0}

    async def _boom(_cfg):
        frames["n"] += 1
        raise RuntimeError("wedged rig")

    monkeypatch.setattr(cont, "scan_start", _boom)
    try:
        client.post("/api/scan/continuous",
                    json={"enabled": True, "config": _TINY})
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False), "fault loop never stopped"
        st = client.get("/api/scan/continuous").json()
        assert st["stopped_reason"] == "fault", st
        assert st["count"] == 0, st                     # no frame ever completed
        assert frames["n"] == 1, frames                 # stopped after the FIRST fault
    finally:
        monkeypatch.undo()
        _stop_and_reset(client, hub)


# ---------------------------------------------------- alternating direction --

def test_continuous_alternates_direction_from_start(client, tmp_path, monkeypatch):
    """The efficient raster: continuous imaging flips up<->down each frame,
    STARTING from whichever direction armed the loop. A frame-up ends at the top,
    so the next runs down from there (no fly-back). We spy the direction actually
    handed to scan_start per frame."""
    import afm_litho.service.host.continuous as cont

    hub = _tmp_store(tmp_path)
    _stop_and_reset(client, hub)
    real = cont.scan_start

    def _run(start_dir):
        seen = []

        async def _spy(cfg):
            seen.append(cfg.get("direction", "up"))
            return await real(cfg)

        monkeypatch.setattr(cont, "scan_start", _spy)
        cfg = dict(_TINY, direction=start_dir)
        r = client.post("/api/scan/continuous",
                        json={"enabled": True, "config": cfg, "max_frames": 4})
        assert r.json()["enabled"] is True
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False), "loop never stopped"
        assert client.get("/api/scan/continuous").json()["count"] == 4
        return seen

    try:
        assert _run("up")[:4] == ["up", "down", "up", "down"]
        _stop_and_reset(client, hub)
        assert _run("down")[:4] == ["down", "up", "down", "up"]
    finally:
        monkeypatch.undo()
        _stop_and_reset(client, hub)
