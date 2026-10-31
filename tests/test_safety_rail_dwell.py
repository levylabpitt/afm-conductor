"""Safety kernel entry #1: the Z-drive rail-dwell telemetry watchdog.

Two layers, both zero-hardware:

- UNIT tests over the pure `RailDwellGuard` (service/safety.py): the exact Guard B
  (consecutive railed lines) and Guard A (per-frame railed fraction) decision
  logic, the >95%-of-a-line rule, the empty-line skip, and only-tighten limits.
  These are the deterministic "feed a frame" tests the spec calls for.

- SERVICE integration over the twin (host/continuous.py): the ~10 Hz HV
  Z-drive poll wired into the continuous supervisor -- a railed Z trips the guard,
  which aborts the frame (Guard B) / stops after it (Guard A), withdraws the tip,
  and reports (log + session record). Scope: a single scan (continuous off) never
  trips.

`client` comes from tests/conftest.py (a TestClient). The service
`hub` is a PROCESS GLOBAL, so the integration tests disable the loop, reset the
guard, and restore the world/speed they touched, and redirect hub.store to a temp
file so a safety record lands there, not in the real runtime/sessions.jsonl.
"""

from __future__ import annotations

import math
import time

from afm_litho.service.safety import RailDwellGuard, RailDwellLimits

# `client` comes from tests/conftest.py (a TestClient).


# ============================================================ unit: the guard ==

def _feed_line(g: RailDwellGuard, n: int, rail_frac: float,
               rail_v: float = 150.0, ok_v: float = 75.0) -> None:
    """Feed one line of `n` samples with the first ~rail_frac fraction at a rail."""
    k = int(round(n * rail_frac))
    for i in range(n):
        g.add_sample(rail_v if i < k else ok_v)


def _armed() -> RailDwellGuard:
    g = RailDwellGuard(RailDwellLimits())
    g.arm()
    g.start_frame()
    return g


# ------------------------------------------------------------------- Guard B --

def test_guard_b_trips_on_five_consecutive_railed_lines():
    g = _armed()
    trips = []
    for row in range(5):
        _feed_line(g, 20, 1.0)                      # 100% railed
        trips.append(g.close_line(row))
    assert trips[:4] == [None, None, None, None]
    assert trips[4] == {"guard": "B", "consec": 5}


def test_guard_b_resets_on_a_clear_line():
    g = _armed()
    for row in range(4):                            # 4 railed -> consec 4, no trip
        _feed_line(g, 20, 1.0)
        assert g.close_line(row) is None
    assert g.consec == 4
    _feed_line(g, 20, 0.0)                          # a clear line breaks the streak
    assert g.close_line(4) is None
    assert g.consec == 0
    trip = None                                     # now it takes 5 more
    for row in range(5, 10):
        _feed_line(g, 20, 1.0)
        trip = g.close_line(row)
    assert trip == {"guard": "B", "consec": 5}


def test_line_railed_only_above_line_frac():
    """A line is railed only if > 95% of its samples railed: 90% and exactly 95%
    do NOT count (strict >)."""
    g = _armed()
    _feed_line(g, 20, 0.90)                         # 18/20 = 0.90 -> not railed
    g.close_line(0)
    assert g.consec == 0
    _feed_line(g, 20, 0.95)                         # 19/20 = 0.95 -> not > 0.95
    g.close_line(1)
    assert g.consec == 0
    _feed_line(g, 40, 39 / 40)                      # 0.975 -> railed
    g.close_line(2)
    assert g.consec == 1


def test_empty_line_is_skipped_not_reset():
    """A line the poll collected zero samples for is 'no data' -- skipped, and it
    must NOT reset a genuine rail streak."""
    g = _armed()
    _feed_line(g, 20, 1.0)
    g.close_line(0)
    assert g.consec == 1
    assert g.close_line(1) is None                  # empty line -> skip
    assert g.consec == 1                            # streak intact


# ------------------------------------------------------------------- Guard A --

def test_guard_a_trips_on_frame_fraction():
    """> 50% of a FRAME's samples railed (without 5 straight railed lines) -> A."""
    g = _armed()
    _feed_line(g, 20, 1.0)                          # railed
    assert g.close_line(0) is None
    _feed_line(g, 20, 0.0)                          # clear (breaks any B streak)
    assert g.close_line(1) is None
    _feed_line(g, 20, 1.0)                          # railed
    assert g.close_line(2) is None
    trip = g.finish_frame()                         # 40/60 = 0.667 > 0.5
    assert trip is not None and trip["guard"] == "A"
    assert trip["frac"] > 0.5


def test_guard_a_no_trip_under_frac():
    g = _armed()
    _feed_line(g, 20, 1.0)
    g.close_line(0)
    _feed_line(g, 20, 0.0)
    g.close_line(1)
    _feed_line(g, 20, 0.0)                          # 20/60 = 0.333 < 0.5
    g.close_line(2)
    assert g.finish_frame() is None


# ------------------------------------------------------ scope / robustness --

def test_disarmed_guard_ignores_samples():
    g = RailDwellGuard(RailDwellLimits())           # NOT armed
    for _ in range(20):
        g.add_sample(150.0)
    assert g.close_line(0) is None                  # no samples recorded
    assert g.consec == 0


def test_nan_and_none_reads_are_dropped():
    g = _armed()
    g.add_sample(float("nan"))
    g.add_sample(None)
    g.add_sample(math.inf)
    g.add_sample(150.0)                             # the only real sample: railed
    g.close_line(0)
    assert g.consec == 1                            # 1/1 railed, hiccups dropped


# ------------------------------------------------------- limits: only-tighten --

def test_tighten_only_tightens():
    base = RailDwellLimits()
    t = base.tighten(v_hi=120, v_lo=5, frame_frac=0.3, line_frac=0.8, consec_lines=3)
    assert (t.v_hi, t.v_lo, t.frame_frac, t.line_frac, t.consec_lines) == \
        (120, 5, 0.3, 0.8, 3)
    lo = base.tighten(v_hi=200, v_lo=-5, frame_frac=0.9, line_frac=0.99,
                      consec_lines=99)              # loosening requests are ignored
    assert (lo.v_hi, lo.v_lo, lo.frame_frac, lo.line_frac, lo.consec_lines) == \
        (140.0, 0.0, 0.5, 0.95, 5)


def test_tightening_consec_moves_trip_point():
    g = RailDwellGuard(RailDwellLimits(consec_lines=3))
    g.arm()
    g.start_frame()
    trip = None
    for row in range(3):
        _feed_line(g, 20, 1.0)
        trip = g.close_line(row)
    assert trip == {"guard": "B", "consec": 3}      # trips at 3, not 5


def test_tightening_vhi_moves_trip_point():
    """A Z drive at 130 V is NOT railed under the default v_hi=140, but IS under a
    tightened v_hi=125."""
    g = RailDwellGuard(RailDwellLimits())
    g.arm()
    g.start_frame()
    for _ in range(20):
        g.add_sample(130.0)
    g.close_line(0)
    assert g.consec == 0
    g2 = RailDwellGuard(RailDwellLimits(v_hi=125.0))
    g2.arm()
    g2.start_frame()
    for _ in range(20):
        g2.add_sample(130.0)
    g2.close_line(0)
    assert g2.consec == 1


# ============================================== service: twin integration ==

def _wait(cond, timeout=40.0, interval=0.05):
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


def _stop_and_reset(client, hub):
    client.post("/api/scan/continuous", json={"enabled": False})
    _wait(lambda: client.get("/api/scan/continuous").json()["enabled"] is False, 10)
    hub.host.scan_repeat.update({
        "enabled": False, "config": None, "count": 0, "started_at": None,
        "errors": 0, "max_frames": None, "max_seconds": None,
        "interval_s": 0.0, "withdraw_on_stop": False, "stopped_reason": None})
    hub.host.rail_guard.disarm()
    hub.host.rail_guard.reset()


def test_guard_b_aborts_frame_and_withdraws_on_twin(client, tmp_path):
    """Guard B end to end: a Z drive pinned at the +150 V rail from line 1 rails
    every line; the guard reaches 5 consecutive railed lines and ABORTS the frame
    mid-run, withdraws the tip, stops continuous, and reports a safety record."""
    hub = _tmp_store(tmp_path)
    _stop_and_reset(client, hub)
    prev = (hub.host.speed, hub.host.world.z_drive_v, hub.host.world.z_drive_drift_v, hub.host.world.t)
    hub.host.speed = 2.0                                 # slow the frame so the poll samples
    hub.host.world.z_drive_v = 150.0                     # pin the Z drive at the top rail
    hub.host.world.z_drive_drift_v = 0.0
    hub.host.world.t = 0.0
    cfg = {"size_um": 5.0, "pixels": 8, "lines": 24, "line_rate_hz": 2.0,
           "mode": "contact"}
    try:
        r = client.post("/api/scan/continuous",
                        json={"enabled": True, "config": cfg, "max_frames": 5})
        assert r.json()["enabled"] is True
        # wait for the FULL unwind: the trip flips enabled=False, then the frame
        # pump releases and the tip withdraws (the deferred, race-free retract).
        assert _wait(lambda: hub.host.rail_guard.tripped
                     and hub.host.rail_guard.tripped.get("withdrawn")), "guard never tripped"
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False)
        st = client.get("/api/scan/continuous").json()
        assert st["stopped_reason"] == "rail_dwell", st
        assert st["count"] == 0, st                 # frame 1 aborted before completing
        g = hub.host.rail_guard
        assert g.tripped and g.tripped["guard"] == "B", g.tripped
        assert g.tripped["withdrawn"] is True, g.tripped
        # reporting (kernel rule #22): a safety session record was written
        kinds = [rec["kind"] for rec in hub.store.list(50)]
        assert "safety" in kinds, kinds
    finally:
        (hub.host.speed, hub.host.world.z_drive_v, hub.host.world.z_drive_drift_v,
         hub.host.world.t) = prev
        _stop_and_reset(client, hub)


def test_guard_a_stops_after_frame_and_withdraws_on_twin(client, tmp_path):
    """Guard A end to end: with the per-line rule disabled (line_frac > 1, so Guard
    B can never fire) but the Z drive railed for the whole frame, the per-frame
    fraction clears 50% -> the frame COMPLETES, then continuous stops and the tip
    withdraws. The completed frame counts; no further frame arms."""
    hub = _tmp_store(tmp_path)
    _stop_and_reset(client, hub)
    prev = (hub.host.speed, hub.host.world.z_drive_v, hub.host.world.t, hub.host.rail_guard.limits)
    hub.host.speed = 2.0
    hub.host.world.z_drive_v = 150.0
    hub.host.world.t = 0.0
    hub.host.rail_guard.limits = RailDwellLimits(line_frac=1.1)   # B disabled; A on frac
    cfg = {"size_um": 5.0, "pixels": 8, "lines": 8, "line_rate_hz": 2.0,
           "mode": "contact"}
    try:
        client.post("/api/scan/continuous",
                    json={"enabled": True, "config": cfg, "max_frames": 3})
        assert _wait(lambda: hub.host.rail_guard.tripped
                     and hub.host.rail_guard.tripped.get("withdrawn")), \
            "Guard A never stopped the loop"
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False)
        st = client.get("/api/scan/continuous").json()
        assert st["stopped_reason"] == "rail_dwell", st
        g = hub.host.rail_guard
        assert g.tripped and g.tripped["guard"] == "A", g.tripped
        assert st["count"] == 1, st                 # the railed frame completed, then stop
        assert g.tripped["withdrawn"] is True, g.tripped
    finally:
        (hub.host.speed, hub.host.world.z_drive_v, hub.host.world.t,
         hub.host.rail_guard.limits) = prev
        _stop_and_reset(client, hub)


def test_guard_inert_for_a_single_scan(client, tmp_path):
    """Scope: the guard polls only during CONTINUOUS imaging. A single attended
    scan with the Z drive railed must NOT arm the guard or trip -- it is
    operator-watched and out of scope."""
    hub = _tmp_store(tmp_path)
    _stop_and_reset(client, hub)
    prev_z = hub.host.world.z_drive_v
    hub.host.world.z_drive_v = 150.0                     # railed, but a single scan is exempt
    try:
        r = client.post("/api/scan/start",
                        json={"size_um": 5.0, "pixels": 8, "lines": 6,
                              "line_rate_hz": 50.0, "mode": "contact"})
        assert r.json().get("ok")
        assert _wait(lambda: hub.host._scan_task is not None and hub.host._scan_task.done(),
                     20), "single scan never finished"
        assert hub.host.rail_guard.armed is False        # never armed off continuous
        assert hub.host.rail_guard.tripped is None
        assert client.get("/api/scan/continuous").json()["enabled"] is False
    finally:
        hub.host.world.z_drive_v = prev_z
        _stop_and_reset(client, hub)


def test_rail_guard_arms_on_hardware_backend(client, tmp_path):
    """Group-2 redo: the rail-dwell guard now ARMS on the HARDWARE backend too. The
    old shared-link race is gone -- the guard is fed from the always-on TELEM poll
    (Igor's autonomous scalar poll -> host/telemetry._hw_telemetry_frame), so
    nothing on the host reads the bridge link. Enabling continuous with a hardware
    backend set ARMS the guard (no `pending` gate any more). The end-to-end poll feed
    + trip is covered in test_safety_monitor_poll.py; here we assert only the
    synchronous arm decision continuous_set makes."""
    hub = _tmp_store(tmp_path)
    _stop_and_reset(client, hub)
    prev_arc = getattr(hub.host, "hardware_arc", None)
    hub.host.hardware_arc = object()                      # a non-None backend sentinel
    cfg = {"size_um": 5.0, "pixels": 8, "lines": 4, "line_rate_hz": 2.0,
           "mode": "contact"}
    try:
        # continuous_set makes the arm decision synchronously (and builds the response
        # state) BEFORE the supervisor task runs, so the POST response reflects the
        # armed guard even though the supervisor -- driving a bare sentinel arc --
        # faults and disarms moments later. Assert on the response, not the live flag.
        st = client.post("/api/scan/continuous",
                         json={"enabled": True, "config": cfg, "max_frames": 1}).json()
        assert st["rail"] is not None and st["rail"]["armed"] is True, st["rail"]
        # no misleading `pending` flag any more -- the guard is genuinely protecting.
        assert "pending" not in st["rail"], st["rail"]
    finally:
        client.post("/api/scan/continuous", json={"enabled": False})
        _wait(lambda: client.get("/api/scan/continuous").json()["enabled"] is False, 10)
        hub.host.hardware_arc = prev_arc
        _stop_and_reset(client, hub)
