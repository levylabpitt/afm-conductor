"""Relative (drift-referenced) setpoint for continuous imaging.

The feature lives in the continuous supervisor (host/continuous.py): between
frames, in RELATIVE mode, it withdraws fully, settles, reads the free-air
deflection, sets the next frame's setpoint to free_air + the scan setpoint, and
re-approaches on the current gains -- so a drifting baseline never rails the Z servo
(docs/RELATIVE_SETPOINT.md). These tests drive it through the FastAPI service on
the twin, exactly as the cockpit would, and pin:

- the commanded setpoint TRACKS free_air + offset each frame (absolute holds fixed);
- the per-frame SEQUENCE (withdraw -> settle -> off-contact read -> re-approach on
  the current gains with the new setpoint);
- CONTACT-MODE GATING (relative is inert for a non-contact frame);
- the FAIL-SAFE (a failed re-reference STOPS the loop with the tip WITHDRAWN).

`client` comes from tests/conftest.py (a TestClient). The service
`hub` is a PROCESS GLOBAL, so every test disables the loop and restores the
world/relative state it touched, and redirects hub.store to a temp file.
"""

from __future__ import annotations

import math
import time

import pytest

# `client` comes from tests/conftest.py (a TestClient).


# ----------------------------------------------------------------- helpers --

def _wait(cond, timeout=30.0, interval=0.02):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return False


def _tmp_store(tmp_path):
    from afm_litho.service.app import hub
    from afm_litho.service.sessions import SessionStore
    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    return hub


def _reset(client, hub):
    """Stop any running loop, clear continuous + relative state, so a leaked
    supervisor or a prior test's config can never bleed into the next test."""
    client.post("/api/scan/continuous", json={"enabled": False})
    _wait(lambda: client.get("/api/scan/continuous").json()["enabled"] is False, 10)
    hub.host.scan_repeat.update({
        "enabled": False, "config": None, "count": 0, "started_at": None,
        "errors": 0, "max_frames": None, "max_seconds": None,
        "interval_s": 0.0, "withdraw_on_stop": False, "stopped_reason": None})
    hub.host.relative_setpoint.update({
        "enabled": False, "settle_s": 4.0,
        "last_free_air_v": None, "last_effective_v": None})


# small, fast frames; settle 0 so the tests don't dwell for real seconds.
_TINY = {"size_um": 5.0, "pixels": 16, "lines": 4, "line_rate_hz": 200.0,
         "mode": "contact", "setpoint": 0.20}


def _spy_setpoints(monkeypatch, cont):
    """Record the setpoint handed to scan_start each frame (real scan still runs)."""
    real = cont.scan_start
    seen: list[float] = []

    async def _spy(cfg):
        seen.append(float(cfg.get("setpoint")))
        return await real(cfg)

    monkeypatch.setattr(cont, "scan_start", _spy)
    return seen


# --------------------------------------------------- tracking / absolute --

def test_relative_tracks_free_air_plus_setpoint(client, tmp_path, monkeypatch):
    """RELATIVE on: frame 1 uses the started setpoint; every later frame arms at
    the freshly-read free-air deflection + the SCAN setpoint (the single "relative
    setpoint" checkbox -- there is no separate offset, the scan setpoint IS the
    target above free-air). Drive a KNOWN free-air sequence and assert the
    commanded setpoint follows free_air + scan setpoint exactly."""
    import afm_litho.service.host.continuous as cont
    hub = _tmp_store(tmp_path)
    _reset(client, hub)
    try:
        reads = [0.100, 0.130, 0.170]        # free-air baseline BEFORE frames 2,3,4
        sp = _TINY["setpoint"]               # the scan setpoint is the base offset (0.20)
        state = {"n": 0}                     # frames armed so far

        # STABLE within one re-reference: the 3-read free-air guard reads the
        # baseline up to 3x per reref, so the mock returns the SAME value across
        # those reads (a real baseline is momentarily steady) and steps per FRAME.
        # The reref before frame k reads reads[k-1]; state["n"] = frames armed so far.
        async def _read():
            return reads[min(max(0, state["n"] - 1), len(reads) - 1)]

        real = cont.scan_start
        seen: list[float] = []

        async def _spy(cfg):
            seen.append(float(cfg.get("setpoint")))
            state["n"] += 1
            return await real(cfg)

        monkeypatch.setattr(cont, "_read_free_air_deflection", _read)
        monkeypatch.setattr(cont, "scan_start", _spy)

        client.post("/api/scan/relative", json={"enabled": True, "settle_s": 0})
        client.post("/api/scan/continuous",
                    json={"enabled": True, "config": _TINY, "max_frames": 4})
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False), "loop never stopped"
        assert client.get("/api/scan/continuous").json()["count"] == 4

        assert len(seen) == 4, seen
        assert seen[0] == pytest.approx(sp)               # frame 1: started setpoint
        assert seen[1] == pytest.approx(reads[0] + sp)    # then free_air + scan setpoint
        assert seen[2] == pytest.approx(reads[1] + sp)    # base never compounds
        assert seen[3] == pytest.approx(reads[2] + sp)

        # the readout reflects the last re-reference
        st = client.get("/api/scan/continuous").json()["relative"]
        assert st["last_free_air_v"] == pytest.approx(reads[2])
        assert st["last_effective_v"] == pytest.approx(reads[2] + sp)
    finally:
        _reset(client, hub)


def test_absolute_mode_holds_setpoint_fixed(client, tmp_path, monkeypatch):
    """ABSOLUTE (default): no withdraw/re-reference; the setpoint is used as-is
    every frame even if the free-air baseline is drifting underneath."""
    import afm_litho.service.host.continuous as cont
    hub = _tmp_store(tmp_path)
    _reset(client, hub)
    try:
        called = {"read": 0, "withdraw": 0}

        async def _read():
            called["read"] += 1
            return 0.9

        async def _withdraw(_body=None):
            called["withdraw"] += 1
            return {}

        monkeypatch.setattr(cont, "_read_free_air_deflection", _read)
        monkeypatch.setattr(cont, "withdraw", _withdraw)
        seen = _spy_setpoints(monkeypatch, cont)

        # a drifting real baseline is present, but absolute mode ignores it
        hub.host.world.free_air_defl_v = 0.2
        hub.host.world.free_air_defl_drift_v = 0.05
        client.post("/api/scan/continuous",
                    json={"enabled": True, "config": _TINY, "max_frames": 3})
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False), "loop never stopped"

        assert seen == [pytest.approx(0.20)] * 3, seen
        assert called["read"] == 0 and called["withdraw"] == 0, called
    finally:
        hub.host.world.free_air_defl_v = 0.0
        hub.host.world.free_air_defl_drift_v = 0.0
        _reset(client, hub)


# ------------------------------------------------- per-frame sequence --

def test_rereference_sequence_withdraw_settle_read_reapproach(client, tmp_path,
                                                              monkeypatch):
    """The between-frame step, in order: full withdraw -> settle dwell -> off-
    contact free-air read -> re-approach on the CURRENT gains with free_air +
    offset. Spy each primitive (calling the real one) and assert the ordering,
    the settle, and the approach body."""
    import afm_litho.service.host.continuous as cont
    hub = _tmp_store(tmp_path)
    _reset(client, hub)
    prev_zloop = getattr(hub.host, "z_loop", None)
    hub.host.z_loop = {"setpoint": 0.2, "pgain": 0.3, "igain": 7.0}   # "current gains"
    hub.host.world.free_air_defl_v = 0.15                             # a real baseline
    hub.host.world.free_air_defl_drift_v = 0.0
    events: list[tuple] = []
    try:
        real_wd, real_ap = cont.withdraw, cont.approach
        real_read = cont._read_free_air_deflection

        async def _wd(body=None):
            events.append(("withdraw", time.monotonic()))
            return await real_wd(body or {})

        async def _read():
            v = await real_read()
            events.append(("read", time.monotonic(), v))
            return v

        async def _ap(body=None):
            events.append(("approach", time.monotonic(), dict(body or {})))
            return await real_ap(body or {})

        monkeypatch.setattr(cont, "withdraw", _wd)
        monkeypatch.setattr(cont, "_read_free_air_deflection", _read)
        monkeypatch.setattr(cont, "approach", _ap)

        client.post("/api/scan/relative",
                    json={"enabled": True, "settle_s": 0.25})
        client.post("/api/scan/continuous",
                    json={"enabled": True, "config": _TINY, "max_frames": 2})
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False, timeout=30), "loop never stopped"
        assert client.get("/api/scan/continuous").json()["count"] == 2

        # exactly one re-reference (between the two frames), in order. The free-air
        # 3-read guard reads TWICE on the FIRST re-reference (the first-frame "two
        # must agree" rule; the stable twin baseline agrees, so it accepts): so the
        # sequence is withdraw -> read -> read -> approach.
        kinds = [e[0] for e in events]
        assert kinds == ["withdraw", "read", "read", "approach"], kinds
        wd_t, rd = events[0][1], events[1]
        ap = events[3]
        # settle dwell honoured between withdraw and read
        assert rd[1] - wd_t >= 0.25 - 0.02, (rd[1] - wd_t)
        # the read is off-contact: it returned the free-air baseline, not a
        # contact deflection (the twin read is the world's free-air value)
        assert rd[2] == pytest.approx(0.15)
        # re-approach on the current gains + the new effective setpoint
        # (free_air + the SCAN setpoint from _TINY, 0.20 -- no separate offset)
        assert ap[2]["setpoint"] == pytest.approx(0.15 + _TINY["setpoint"])
        assert ap[2]["pgain"] == pytest.approx(0.3)
        assert ap[2]["igain"] == pytest.approx(7.0)
    finally:
        if prev_zloop is None:
            if hasattr(hub.host, "z_loop"):
                delattr(hub.host, "z_loop")
        else:
            hub.host.z_loop = prev_zloop
        hub.host.world.free_air_defl_v = 0.0
        _reset(client, hub)


# --------------------------------------------------- contact-mode gating --

def test_relative_inert_outside_contact_mode(client, tmp_path, monkeypatch):
    """Relative mode is CONTACT-ONLY: the vendor records a free-air deflection
    baseline only for the contact family (ScanMaster.ipf:4665). With an AC-mode
    frame, relative is inert -- no withdraw/read, and the setpoint holds fixed."""
    import afm_litho.service.host.continuous as cont
    hub = _tmp_store(tmp_path)
    _reset(client, hub)
    try:
        touched = {"read": 0, "withdraw": 0}

        async def _read():
            touched["read"] += 1
            return 0.5

        async def _withdraw(_body=None):
            touched["withdraw"] += 1
            return {}

        monkeypatch.setattr(cont, "_read_free_air_deflection", _read)
        monkeypatch.setattr(cont, "withdraw", _withdraw)
        seen = _spy_setpoints(monkeypatch, cont)

        ac_cfg = dict(_TINY, mode="ac", setpoint=0.35)
        client.post("/api/scan/relative", json={"enabled": True, "settle_s": 0})
        client.post("/api/scan/continuous",
                    json={"enabled": True, "config": ac_cfg, "max_frames": 3})
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False), "loop never stopped"

        assert seen == [pytest.approx(0.35)] * 3, seen      # setpoint never moved
        assert touched == {"read": 0, "withdraw": 0}, touched
    finally:
        _reset(client, hub)


# ------------------------------------------------------------- fail-safe --

def test_failed_reference_fails_safe_tip_withdrawn(client, tmp_path, monkeypatch):
    """A failed free-air read (NaN) between frames STOPS the loop with reason
    `reref_failed` -- it must NEVER scan the next frame at a stale/wrong setpoint,
    and the tip is left WITHDRAWN (the re-reference withdrew and never re-approached)."""
    import afm_litho.service.host.continuous as cont
    hub = _tmp_store(tmp_path)
    _reset(client, hub)
    counts = {"withdraw": 0, "approach": 0}
    try:
        real_wd = cont.withdraw

        async def _read():
            return math.nan                       # the read fails

        async def _wd(body=None):
            counts["withdraw"] += 1
            return await real_wd(body or {})

        async def _ap(body=None):
            counts["approach"] += 1
            return {}

        monkeypatch.setattr(cont, "_read_free_air_deflection", _read)
        monkeypatch.setattr(cont, "withdraw", _wd)
        monkeypatch.setattr(cont, "approach", _ap)

        client.post("/api/scan/relative", json={"enabled": True, "settle_s": 0})
        client.post("/api/scan/continuous",
                    json={"enabled": True, "config": _TINY, "max_frames": 10})
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False), "loop never stopped on failure"
        st = client.get("/api/scan/continuous").json()
        assert st["stopped_reason"] == "reref_failed", st
        assert st["count"] == 1, st               # only the first frame ran
        # tip left withdrawn: the re-reference withdrew and, the read having
        # failed, never re-approached.
        assert counts["withdraw"] >= 1, counts
        assert counts["approach"] == 0, counts
    finally:
        _reset(client, hub)


def test_a_refused_reapproach_fails_safe_and_never_arms_the_next_frame(
        client, tmp_path, monkeypatch):
    """A REFUSAL IS NOT AN EXCEPTION, and the loop used to swallow it.

    `_rereference` did a bare `await approach(body)` inside a try that only catches
    exceptions -- but `approach` answers with an error DICT for the silent-twin gate
    and, since the single-writer claim landed, for `busy`. Both fell straight
    through: the loop recorded the new effective setpoint and armed the next frame
    believing the tip had been re-approached, when the re-reference had just
    WITHDRAWN it. Scanning from there is a raster into an unknown Z.

    Fails safe now: the loop stops with `reref_failed`, tip already off the surface.
    """
    import afm_litho.service.host.continuous as cont
    hub = _tmp_store(tmp_path)
    _reset(client, hub)
    seen = {"approach": 0}
    try:
        real_wd = cont.withdraw

        async def _read():
            return 0.10                            # a perfectly good free-air read

        async def _wd(body=None):
            return await real_wd(body or {})

        async def _busy(body=None):
            seen["approach"] += 1
            return {"error": "busy", "reason": "busy", "run_id": "write-1"}

        monkeypatch.setattr(cont, "_read_free_air_deflection", _read)
        monkeypatch.setattr(cont, "withdraw", _wd)
        monkeypatch.setattr(cont, "approach", _busy)

        client.post("/api/scan/relative", json={"enabled": True, "settle_s": 0})
        client.post("/api/scan/continuous",
                    json={"enabled": True, "config": _TINY, "max_frames": 10})
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False), "loop never stopped on a refusal"
        st = client.get("/api/scan/continuous").json()
        assert st["stopped_reason"] == "reref_failed", st
        assert st["count"] == 1, st                # the second frame never armed
        assert seen["approach"] == 1, seen
    finally:
        _reset(client, hub)
