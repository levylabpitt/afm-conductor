"""Safety monitor group-2 (REDO): the always-on TELEM poll feed.

The reverted attempt fed the rail-dwell guard from an in-wave CAPTURE armed on
Igor banks 2 AND 3 -- but there is NO bank 3 on the hardware, so it broke every
scan line. This redo feeds the guard from a SCALAR poll: Igor's autonomous
AFM_TipTick background pushes one `TELEM x y zdrive defl sum amp` per tick, the bridge
router routes it to the host consumer (host/telemetry._hw_telemetry_frame),
and THAT feeds the guard from the polled Output.Z. NO in-wave capture bank.

Coverage here (all zero-hardware, on the fake bridge / the pure consumer):
- the consumer populates hub._hw_telemetry (defl/zdrive/sum/amp/row; height None) and
  is per-channel NaN-safe;
- the consumer feeds the guard and signals a Guard B trip (pending + _hw_abort);
- end to end on the fake hardware bridge: a TELEM stream with Output.Z pinned at
  the +150 V rail trips Guard B mid-frame, the frame aborts, and the tip retracts
  through safe_park -- stopped_reason rail_dwell (NO capture bank involved);
- Guard A (per-frame railed fraction) trips the same way from the poll;
- the /ws heartbeat carries hw_telemetry on hardware and carries the same key set on the twin;
- the stub emits real TELEM frames;
- HARDWARE HONESTY: the .ipf + stub reference NO in-wave bank > 2 (the phantom-bank
  bug class cannot recur).

The pure Guard A/B decision + only-tighten limits are covered by
tests/test_safety_rail_dwell.py; `client` comes from tests/conftest.py.
"""

from __future__ import annotations

import contextlib
import re
import time
from pathlib import Path

from afm_litho.hardware.hardware_arc import HardwareARC
from afm_litho.hardware.igor_stub import FakeIgorBridge
from afm_litho.service.host import telemetry as C
from afm_litho.service.safety import RailDwellGuard, RailDwellLimits

CAL = {"x_lvdt_sens": 7.46e-6, "x_lvdt_offset": -0.63,
       "y_lvdt_sens": 7.04e-6, "y_lvdt_offset": 0.07,
       "z_lvdt_sens": 5e-6, "z_lvdt_offset": 0.0}


def _wait(cond, timeout=40.0, interval=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return False


# ============================================ unit: the TELEM consumer =========

def test_consumer_populates_hw_telemetry():
    """Every TELEM frame refreshes hub._hw_telemetry with the safety channels; a NaN
    channel lands as None (per-channel NaN-safe), height_nm is None (ZSensor is not a
    polled channel), and the current raster row rides along."""
    from afm_litho.service.app import hub
    prev = (hub.host._hw_telemetry, hub.host._scan_row, hub.host.rail_guard.armed)
    hub.host.rail_guard.disarm()                              # not feeding the guard here
    hub.host._scan_row = 3
    try:
        C._hw_telemetry_frame(0.1, 0.2, 66.0, 0.25, 5.0, 3.0)
        t = hub.host._hw_telemetry
        assert t["row"] == 3
        assert abs(t["defl_v"] - 0.25) < 1e-9
        assert abs(t["zdrive_v"] - 66.0) < 1e-9
        assert abs(t["sum_v"] - 5.0) < 1e-9
        assert abs(t["amp_v"] - 3.0) < 1e-9              # amplitude (display-only)
        assert t["height_nm"] is None                    # ZSensor not polled
        # a NaN channel -> None, without dropping the rest of the frame
        C._hw_telemetry_frame(0.1, 0.2, float("nan"), 0.25, float("nan"), float("nan"))
        t = hub.host._hw_telemetry
        assert t["zdrive_v"] is None and t["sum_v"] is None and t["amp_v"] is None
        assert abs(t["defl_v"] - 0.25) < 1e-9
    finally:
        hub.host._hw_telemetry, hub.host._scan_row = prev[0], prev[1]
        if prev[2]:
            hub.host.rail_guard.arm()


def test_consumer_feeds_guard_and_signals_guard_b_trip():
    """During a continuous frame the consumer feeds the guard from the polled Output.Z
    and, on five (here three) consecutive fully-railed lines, stashes a Guard B trip
    and sets _hw_abort so the in-flight scan stops -- all on the router thread, no
    await, no bridge read."""
    from afm_litho.service.app import hub
    prev_guard = hub.host.rail_guard
    prev = (hub.host._rail_feeding, hub.host._rail_last_row, hub.host._rail_pending_trip,
            hub.host._scan_row)
    g = RailDwellGuard(RailDwellLimits(consec_lines=3, line_frac=0.5))
    g.arm()
    g.start_frame()
    hub.host.rail_guard = g
    hub.host._rail_feeding = True
    hub.host._rail_last_row = None
    hub.host._rail_pending_trip = None
    hub.host._hw_abort.clear()
    try:
        # rows 0..3, each fed several fully-railed (+150 V) samples. A row advance
        # closes the previous line; closing rows 0,1,2 reaches consec == 3 -> trip.
        for row in range(4):
            hub.host._scan_row = row
            for _ in range(4):
                C._hw_telemetry_frame(0.0, 0.0, 150.0, 0.2, 5.0, 3.0)
        assert hub.host._rail_pending_trip is not None
        assert hub.host._rail_pending_trip["guard"] == "B", hub.host._rail_pending_trip
        assert hub.host._hw_abort.is_set()                    # scan told to abort NOW
    finally:
        hub.host.rail_guard = prev_guard
        (hub.host._rail_feeding, hub.host._rail_last_row, hub.host._rail_pending_trip,
         hub.host._scan_row) = prev
        hub.host._hw_abort.clear()


def test_consumer_inert_when_not_feeding():
    """Between frames (_rail_feeding False) the consumer updates telemetry but does
    NOT feed the guard -- so a railed reading while no frame is live never trips."""
    from afm_litho.service.app import hub
    prev_guard = hub.host.rail_guard
    prev = (hub.host._rail_feeding, hub.host._rail_pending_trip, hub.host._hw_abort.is_set())
    g = RailDwellGuard(RailDwellLimits(consec_lines=2, line_frac=0.5))
    g.arm()
    g.start_frame()
    hub.host.rail_guard = g
    hub.host._rail_feeding = False                            # no live frame
    hub.host._rail_pending_trip = None
    hub.host._hw_abort.clear()
    try:
        for row in range(6):
            hub.host._scan_row = row
            C._hw_telemetry_frame(0.0, 0.0, 150.0, 0.2, 5.0, 3.0)
        assert hub.host._rail_pending_trip is None            # never fed -> never trips
        assert g.consec == 0
    finally:
        hub.host.rail_guard = prev_guard
        hub.host._rail_feeding, hub.host._rail_pending_trip = prev[0], prev[1]
        hub.host._hw_abort.clear()


# ============================== unit: SUM + deflection interlocks (S3) ==========
# The interlocks live at the SAME TELEM seam as the rail feed and share its gate
# (guard armed + _rail_feeding). Here we drive the pure consumer directly and assert
# the pending interlock trip; the constants under test are C.SUM_ABS_FLOOR_V (0.5 V),
# C.SUM_BASELINE_FRAC (0.5) and C.INTERLOCK_DEBOUNCE_POINTS (5). Output.Z is held at a
# safe 30 V and the raster row is held constant, so the rail guard never fires here.


@contextlib.contextmanager
def _interlock_ctx(*, baseline=None):
    """Arm a guard + feeding so C._hw_telemetry_frame evaluates the interlocks, with a
    known SUM baseline. Restores all touched hub state on exit."""
    from afm_litho.service.app import hub
    prev = (hub.host.rail_guard, hub.host._rail_feeding, hub.host._rail_last_row,
            hub.host._rail_pending_trip, hub.host._interlock_pending_trip, hub.host._scan_row,
            hub.host._sum_baseline, hub.host._sum_low_streak, hub.host._defl_high_streak,
            hub.host._hw_abort.is_set())
    g = RailDwellGuard(RailDwellLimits(v_hi=140.0, v_lo=0.0))
    g.arm()
    g.start_frame()
    hub.host.rail_guard = g
    hub.host._rail_feeding = True
    hub.host._rail_last_row = None
    hub.host._scan_row = 0                                    # constant row -> no line closes
    hub.host._rail_pending_trip = None
    hub.host._interlock_pending_trip = None
    hub.host._sum_baseline = baseline
    hub.host._sum_low_streak = 0
    hub.host._defl_high_streak = 0
    hub.host._hw_abort.clear()
    try:
        yield hub
    finally:
        hub.host.rail_guard = prev[0]
        (hub.host._rail_feeding, hub.host._rail_last_row, hub.host._rail_pending_trip,
         hub.host._interlock_pending_trip, hub.host._scan_row, hub.host._sum_baseline,
         hub.host._sum_low_streak, hub.host._defl_high_streak) = prev[1:9]
        hub.host._hw_abort.clear()


def _frame(hub, *, sum_v, defl=0.0, zdrive=30.0, amp=3.0):
    C._hw_telemetry_frame(0.0, 0.0, zdrive, defl, sum_v, amp)


def test_sum_interlock_absolute_floor_trips_after_five_points():
    """SUM < 0.5 V for 5 consecutive telemetry points -> pending sum_lost trip + abort;
    four points is not enough. No baseline (relative arm off), absolute floor only."""
    with _interlock_ctx(baseline=None) as hub:
        for _ in range(4):
            _frame(hub, sum_v=0.1)
        assert hub.host._interlock_pending_trip is None       # 4 < debounce
        _frame(hub, sum_v=0.1)                            # the 5th
        assert hub.host._interlock_pending_trip is not None
        assert hub.host._interlock_pending_trip["stopped_reason"] == "sum_lost"
        assert hub.host._hw_abort.is_set()


def test_sum_interlock_relative_arm_trips_below_half_baseline():
    """With a 6 V baseline, SUM at 2.5 V (< 50% of 6 V but > the 0.5 V absolute floor)
    is a collapse via the RELATIVE arm; 5 points -> sum_lost."""
    with _interlock_ctx(baseline=6.0) as hub:
        for _ in range(5):
            _frame(hub, sum_v=2.5)
        assert hub.host._interlock_pending_trip is not None
        assert hub.host._interlock_pending_trip["stopped_reason"] == "sum_lost"


def test_sum_interlock_single_good_point_resets_streak():
    """A single good SUM point resets the streak (blip tolerance): 4 low + 1 good + 4
    low never reaches 5 consecutive -> no trip."""
    with _interlock_ctx(baseline=None) as hub:
        for _ in range(4):
            _frame(hub, sum_v=0.1)
        _frame(hub, sum_v=5.0)                            # good -> streak resets
        for _ in range(4):
            _frame(hub, sum_v=0.1)
        assert hub.host._interlock_pending_trip is None
        assert hub.host._hw_abort.is_set() is False


def test_sum_interlock_nan_point_neither_trips_nor_resets():
    """A None/NaN SUM point leaves the streak UNCHANGED: 4 low (streak 4), then a NaN
    (no trip, no reset), then 1 low reaches 5 -> trip (proving the NaN did not reset)."""
    with _interlock_ctx(baseline=None) as hub:
        for _ in range(4):
            _frame(hub, sum_v=0.1)
        _frame(hub, sum_v=float("nan"))                  # unchanged: no trip, no reset
        assert hub.host._interlock_pending_trip is None
        _frame(hub, sum_v=0.1)                            # 5th real low -> trip
        assert hub.host._interlock_pending_trip is not None
        assert hub.host._interlock_pending_trip["stopped_reason"] == "sum_lost"


def test_nan_baseline_disables_relative_arm_absolute_still_trips():
    """A None baseline DISABLES the relative arm: SUM at 2.5 V (would be < 50% of a 6 V
    baseline) never trips; but the absolute 0.5 V floor still trips (SUM 0.3 V)."""
    with _interlock_ctx(baseline=None) as hub:
        for _ in range(8):
            _frame(hub, sum_v=2.5)                        # relative off -> not a collapse
        assert hub.host._interlock_pending_trip is None
        for _ in range(5):
            _frame(hub, sum_v=0.3)                        # absolute floor still trips
        assert hub.host._interlock_pending_trip is not None
        assert hub.host._interlock_pending_trip["stopped_reason"] == "sum_lost"


def test_deflection_interlock_trips_after_five_points():
    """|Deflection| > SUM for 5 consecutive points -> pending deflection trip. SUM is
    held at 1.0 V (> the 0.5 V floor, no baseline) so ONLY the deflection arm fires."""
    with _interlock_ctx(baseline=None) as hub:
        for _ in range(4):
            _frame(hub, sum_v=1.0, defl=2.0)
        assert hub.host._interlock_pending_trip is None
        _frame(hub, sum_v=1.0, defl=2.0)                  # the 5th
        assert hub.host._interlock_pending_trip is not None
        assert hub.host._interlock_pending_trip["stopped_reason"] == "deflection"
        assert hub.host._hw_abort.is_set()


def test_deflection_interlock_single_in_range_point_resets():
    """A single |defl| <= SUM point resets the deflection streak: 4 high + 1 in-range +
    4 high -> no trip."""
    with _interlock_ctx(baseline=None) as hub:
        for _ in range(4):
            _frame(hub, sum_v=1.0, defl=2.0)
        _frame(hub, sum_v=1.0, defl=0.5)                  # |0.5| <= 1.0 -> resets
        for _ in range(4):
            _frame(hub, sum_v=1.0, defl=2.0)
        assert hub.host._interlock_pending_trip is None


def test_deflection_interlock_below_threshold_never_trips():
    """|Deflection| <= SUM never trips, however many points."""
    with _interlock_ctx(baseline=None) as hub:
        for _ in range(20):
            _frame(hub, sum_v=1.0, defl=0.5)
        assert hub.host._interlock_pending_trip is None


# ================================= end-to-end on the fake hardware bridge =======

def _hw_setup(tmp_path, values, *, done_delay=0.03, telem_hz=300.0):
    """Attach a fake hardware bridge that emits an always-on TELEM stream, install
    the host consumer, and redirect the session store. Returns (hub, stub, hw, prev)."""
    from afm_litho.service.app import hub
    from afm_litho.service.sessions import SessionStore
    stub = FakeIgorBridge(values, done_delay=done_delay, telem_hz=telem_hz)
    port = stub.serve_in_thread()
    hw = HardwareARC.connect("127.0.0.1", port)
    prev = (hub.store, hub.host.hardware_arc, hub.host.hw_cal)
    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    hub.host.hardware_arc = hw
    hub.host.hw_cal = CAL
    C.install_hw_telemetry(hw)                           # single always-on consumer
    return hub, stub, hw, prev


def _hw_teardown(client, hub, stub, hw, prev):
    client.post("/api/scan/continuous", json={"enabled": False})
    _wait(lambda: client.get("/api/scan/continuous").json()["enabled"] is False, 10)
    hub.host.rail_guard.disarm()
    hub.host.rail_guard.reset()
    hub.host._hw_abort.clear()
    hub.host._hw_withdraw_after = False
    hub.host._hw_withdraw_epoch = None
    hub.host._rail_feeding = False
    hub.host._rail_pending_trip = None
    hub.host._interlock_pending_trip = None
    hub.host._sum_baseline = None
    hub.host._sum_low_streak = 0
    hub.host._defl_high_streak = 0
    hub.host._hw_telemetry = None
    hub.store, hub.host.hardware_arc, hub.host.hw_cal = prev
    hw.close()
    stub.stop()


def test_continuous_hardware_rail_trip_retracts_via_poll(client, tmp_path):
    """End to end on the fake hardware bridge: continuous imaging drives a chained
    scan while the always-on TELEM poll streams Output.Z pinned at the +150 V rail.
    The consumer feeds the guard from the POLL (no capture, no bank), Guard B reaches
    five railed lines, the frame aborts, and the tip retracts through safe_park (the
    fake zeros the Output.Z readback on ABORT, so the verify passes) -- stopped_reason
    rail_dwell. hub._hw_telemetry carries the railed Z-drive along the way."""
    # Output.Z railed at +150 V in the value store -> every TELEM frame carries it.
    values = {"XSensor": 0.0, "YSensor": 0.0, "Output.Z": 150.0,
              "Deflection": 0.2, "DetectorSum": 5.0, "height": 66.0}
    hub, stub, hw, prev = _hw_setup(tmp_path, values)
    cfg = {"size_um": 5.0, "pixels": 32, "lines": 12, "line_rate_hz": 5.0,
           "mode": "contact", "chained": True, "trace_retrace": False}
    try:
        r = client.post("/api/scan/continuous",
                        json={"enabled": True, "config": cfg, "max_frames": 3})
        assert r.json()["enabled"] is True
        # the poll populated the hardware telemetry with the railed Z-drive
        assert _wait(lambda: hub.host._hw_telemetry is not None
                     and hub.host._hw_telemetry.get("zdrive_v") == 150.0), hub.host._hw_telemetry
        # Guard B trips from the poll feed, then the tip withdraws (deferred retract).
        assert _wait(lambda: hub.host.rail_guard.tripped
                     and hub.host.rail_guard.tripped.get("withdrawn")), \
            "hardware rail guard never tripped+withdrew from the TELEM poll feed"
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False)
        g = hub.host.rail_guard
        assert g.tripped["guard"] == "B", g.tripped
        st = client.get("/api/scan/continuous").json()
        assert st["stopped_reason"] == "rail_dwell", st
        assert stub.safe_parks >= 1                      # retract went through safe_park
        assert hub.host._safe_park.get("verified") is True, hub.host._safe_park
    finally:
        _hw_teardown(client, hub, stub, hw, prev)


def test_continuous_hardware_guard_a_trips_from_poll(client, tmp_path):
    """Guard A (per-frame railed fraction) from the poll: with the per-line rule off
    (line_frac > 1, so Guard B can never fire) but Output.Z railed the whole frame,
    the per-frame fraction clears 50% -> the frame COMPLETES, then continuous stops
    and the tip withdraws. stopped_reason rail_dwell, guard A."""
    values = {"XSensor": 0.0, "YSensor": 0.0, "Output.Z": 150.0,
              "Deflection": 0.2, "DetectorSum": 5.0, "height": 66.0}
    hub, stub, hw, prev = _hw_setup(tmp_path, values, done_delay=0.04)
    hub.host.rail_guard.limits = RailDwellLimits(line_frac=1.1)   # B disabled; A on fraction
    cfg = {"size_um": 5.0, "pixels": 32, "lines": 6, "line_rate_hz": 5.0,
           "mode": "contact", "chained": True, "trace_retrace": False}
    try:
        client.post("/api/scan/continuous",
                    json={"enabled": True, "config": cfg, "max_frames": 3})
        assert _wait(lambda: hub.host.rail_guard.tripped
                     and hub.host.rail_guard.tripped.get("withdrawn")), \
            "Guard A never stopped the loop from the poll"
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False)
        g = hub.host.rail_guard
        assert g.tripped["guard"] == "A", g.tripped
        st = client.get("/api/scan/continuous").json()
        assert st["stopped_reason"] == "rail_dwell", st
        assert stub.safe_parks >= 1
    finally:
        hub.host.rail_guard.limits = RailDwellLimits()
        _hw_teardown(client, hub, stub, hw, prev)


def test_continuous_hardware_sum_lost_retracts_via_poll(client, tmp_path):
    """End to end on the fake bridge: the always-on TELEM poll streams SUM pinned at
    0.1 V (below the 0.5 V absolute floor) while Output.Z sits at a safe 30 V. The SUM
    interlock debounces 5 points, aborts the frame, and retracts through safe_park --
    stopped_reason sum_lost (NO rail trip, NO capture bank)."""
    values = {"XSensor": 0.0, "YSensor": 0.0, "Output.Z": 30.0,
              "Deflection": 0.05, "DetectorSum": 0.1, "height": 66.0}
    hub, stub, hw, prev = _hw_setup(tmp_path, values)
    cfg = {"size_um": 5.0, "pixels": 32, "lines": 12, "line_rate_hz": 5.0,
           "mode": "contact", "chained": True, "trace_retrace": False}
    try:
        client.post("/api/scan/continuous",
                    json={"enabled": True, "config": cfg, "max_frames": 3})
        # baseline read at enable saw the collapsed SUM (0.1 V)
        assert abs(hub.host._sum_baseline - 0.1) < 1e-9, hub.host._sum_baseline
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False), "SUM interlock never stopped the loop"
        st = client.get("/api/scan/continuous").json()
        assert st["stopped_reason"] == "sum_lost", st
        assert hub.host.rail_guard.tripped is None            # rail did NOT trip; interlock did
        assert stub.safe_parks >= 1
        assert hub.host._safe_park.get("verified") is True, hub.host._safe_park
    finally:
        _hw_teardown(client, hub, stub, hw, prev)


def test_continuous_hardware_deflection_retracts_via_poll(client, tmp_path):
    """End to end: |Deflection| (2.0 V) > SUM (1.0 V) for 5 points -> deflection
    interlock aborts the frame and retracts through safe_park -- stopped_reason
    deflection. SUM at 1.0 V stays above the floor so only the deflection arm fires."""
    values = {"XSensor": 0.0, "YSensor": 0.0, "Output.Z": 30.0,
              "Deflection": 2.0, "DetectorSum": 1.0, "height": 66.0}
    hub, stub, hw, prev = _hw_setup(tmp_path, values)
    cfg = {"size_um": 5.0, "pixels": 32, "lines": 12, "line_rate_hz": 5.0,
           "mode": "contact", "chained": True, "trace_retrace": False}
    try:
        client.post("/api/scan/continuous",
                    json={"enabled": True, "config": cfg, "max_frames": 3})
        assert _wait(lambda: client.get("/api/scan/continuous")
                     .json()["enabled"] is False), "deflection interlock never stopped"
        st = client.get("/api/scan/continuous").json()
        assert st["stopped_reason"] == "deflection", st
        assert hub.host.rail_guard.tripped is None
        assert stub.safe_parks >= 1
        assert hub.host._safe_park.get("verified") is True, hub.host._safe_park
    finally:
        _hw_teardown(client, hub, stub, hw, prev)


def test_continuous_set_populates_sum_baseline(client, tmp_path):
    """continuous_set takes a ONE-TIME pre-scan SUM baseline from the rig (here 6.0 V)
    and stores it on hub before the frame loop; a healthy SUM (6 V, defl 0.2) runs
    without tripping any interlock."""
    values = {"XSensor": 0.0, "YSensor": 0.0, "Output.Z": 30.0,
              "Deflection": 0.2, "DetectorSum": 6.0, "height": 66.0}
    hub, stub, hw, prev = _hw_setup(tmp_path, values)
    cfg = {"size_um": 5.0, "pixels": 32, "lines": 6, "line_rate_hz": 5.0,
           "mode": "contact", "chained": True, "trace_retrace": False}
    try:
        client.post("/api/scan/continuous",
                    json={"enabled": True, "config": cfg, "max_frames": 1})
        assert abs(hub.host._sum_baseline - 6.0) < 1e-9, hub.host._sum_baseline
        # healthy signal -> the interlocks never latch a pending trip
        assert hub.host._interlock_pending_trip is None
    finally:
        _hw_teardown(client, hub, stub, hw, prev)


def test_continuous_set_nan_baseline_when_sum_unreadable(client, tmp_path):
    """When the pre-scan SUM read is unreadable (no `DetectorSum` key -> NaN), the
    baseline is left None (relative arm disabled) and the poll's NaN SUM never trips."""
    values = {"XSensor": 0.0, "YSensor": 0.0, "Output.Z": 30.0,
              "Deflection": 0.05, "height": 66.0}     # no `DetectorSum` -> GETVALUE NaN
    hub, stub, hw, prev = _hw_setup(tmp_path, values)
    cfg = {"size_um": 5.0, "pixels": 32, "lines": 6, "line_rate_hz": 5.0,
           "mode": "contact", "chained": True, "trace_retrace": False}
    try:
        client.post("/api/scan/continuous",
                    json={"enabled": True, "config": cfg, "max_frames": 1})
        assert hub.host._sum_baseline is None                 # NaN read -> disabled
        assert hub.host._interlock_pending_trip is None       # NaN SUM stream never trips
    finally:
        _hw_teardown(client, hub, stub, hw, prev)


# ============================================= telemetry on the /ws heartbeat ===

def test_heartbeat_carries_hw_telemetry_on_hardware(client):
    """On the hardware backend the /ws telemetry frame carries the poll's
    hw_telemetry snapshot (Deflection / Z-drive / Sum / row)."""
    from afm_litho.service.app import hub
    prev = (hub.host.hardware_arc, hub.host._hw_telemetry)
    hub.host.hardware_arc = object()                          # a non-None backend sentinel
    hub.host._hw_telemetry = {"row": 4, "defl_v": 0.25, "zdrive_v": 66.0,
                         "sum_v": 5.0, "height_nm": None}
    try:
        with client.websocket_connect("/ws") as ws:
            ws.send_text("ping")
            for _ in range(6):
                m = ws.receive_json()
                if m.get("type") == "telemetry":
                    assert m.get("hw_telemetry") == hub.host._hw_telemetry, m
                    return
        raise AssertionError("no telemetry frame seen")
    finally:
        hub.host.hardware_arc, hub.host._hw_telemetry = prev


def test_heartbeat_twin_hw_telemetry_matches_hardware_keys(client):
    """The twin's /ws telemetry frame carries `hw_telemetry` with exactly the key set the
    hardware poll builds, so the cockpit readout shows on the twin. sum_v is the twin's constant
    10 V."""
    from afm_litho.service.app import hub
    from afm_litho.service.host import telemetry
    saved = hub.host._hw_telemetry
    try:
        telemetry._hw_telemetry_frame(1.0, -0.25, 66.0, 0.5, 6.1, 0.0)
        hw_keys = set(hub.host._hw_telemetry)
    finally:
        hub.host._hw_telemetry = saved
    assert hub.hw_bridge() is None                      # dev-machine default
    with client.websocket_connect("/ws") as ws:
        ws.send_text("ping")
        for _ in range(6):
            m = ws.receive_json()
            if m.get("type") == "telemetry":
                block = m["hw_telemetry"]
                assert set(block) == hw_keys, block
                assert block["sum_v"] == 10.0
                assert abs(block["t"] - time.time()) < 5.0
                assert m["bias_v"] is None
                assert isinstance(block["zdrive_v"], float)
                return
    raise AssertionError("no telemetry frame seen")


def test_ws_frame_hardware_backend_without_bridge_has_no_hw_telemetry(client, monkeypatch):
    """Before the bridge binds, a hardware-configured station claims nothing."""
    from afm_litho.service import board
    from afm_litho.service.app import hub
    monkeypatch.setattr(hub.host, "backend", "hardware")
    monkeypatch.setattr(hub.host, "_hw_telemetry", None)
    assert hub.hw_bridge() is None
    assert "hw_telemetry" not in board.ws_frame(hub)


# ================================================= the stub emits real TELEM ====

def test_stub_emits_telem_stream():
    """The stub models the real always-on poll: with telem_hz set it autonomously
    streams `TELEM x y zdrive defl sum amp` from its value store (a SCALAR poll -- no
    in-wave bank), served to the router's on_telem."""
    values = {"XSensor": 0.11, "YSensor": 0.22, "Output.Z": 66.0,
              "Deflection": 0.25, "DetectorSum": 5.0, "amplitude": 3.0}
    stub = FakeIgorBridge(values, telem_hz=200.0)
    port = stub.serve_in_thread()
    hw = HardwareARC.connect("127.0.0.1", port)
    seen: list[tuple] = []
    hw.on_telem = lambda *a: seen.append(a)
    try:
        assert _wait(lambda: len(seen) >= 2, timeout=5.0), "stub emitted no TELEM"
        x, y, z, d, s, a = seen[-1]
        assert abs(x - 0.11) < 1e-6 and abs(y - 0.22) < 1e-6
        assert abs(z - 66.0) < 1e-6 and abs(d - 0.25) < 1e-6 and abs(s - 5.0) < 1e-6
        assert abs(a - 3.0) < 1e-6                        # amplitude rides every frame
    finally:
        hw.close()
        stub.stop()


# ================================================= hardware honesty (no bank>2) ==

_IGOR = Path(__file__).resolve().parents[1] / "igor" / "afm_litho_run.ipf"
_STUB = (Path(__file__).resolve().parents[1] / "src" / "afm_litho" / "hardware"
         / "igor_stub.py")

# Every in-/out-wave bank call carries the bank as its FIRST integer argument.
_BANK_CALL = re.compile(
    r"\b(?:td_xSetInWave|td_xSetInWavePair|td_xSetOutWave|td_xSetOutWavePair|"
    r"ir_StopInWaveBank|ir_StopOutWaveBank|AFM_ArmInWave|AFM_ArmInPair|"
    r"AFM_ArmOutWave|AFM_ArmOutPair)\s*\(\s*([0-9]+)")


def _bank_literals(text: str) -> list[int]:
    """Every literal bank INDEX passed to a wave-bank call (ignores calls whose bank
    is a variable, e.g. the generic AFM_ArmInWave(bank,...) wrappers)."""
    return [int(m.group(1)) for m in _BANK_CALL.finditer(text)]


def test_ipf_references_no_in_wave_bank_above_2():
    """HARDWARE HONESTY (the phantom-bank bug class must not recur): the Igor .ipf
    scan/litho arming references ONLY wave banks 0-2. The reverted attempt armed a
    capture on bank 3 (`ir_StopInWaveBank(3)`), which the hardware does not have, and
    broke every scan line. Assert no bank literal > 2 anywhere, and that the specific
    banned bank-3 calls are absent."""
    text = _IGOR.read_text(encoding="latin1")
    banks = _bank_literals(text)
    assert banks, "no wave-bank calls found -- the regex or file moved"
    assert max(banks) <= 2, f"an in-/out-wave bank > 2 is referenced: {sorted(set(banks))}"
    assert "ir_StopInWaveBank(3" not in text
    assert "td_xSetInWave(3" not in text and "td_xSetInWavePair(3" not in text


def test_stub_models_only_real_banks():
    """The stub must model real hardware -- no phantom bank > 2 in its scan/litho
    simulation (green tests off a fake bank-3 machine were the false-confidence that
    shipped the bug).

    The stub hard-codes NO bank of its own: it is a pure reflector that records
    int(parts[1]) from whatever wave-bank command the host arms (igor_stub._dispatch
    OUTWAVE/OUTPAIR/INWAVE/INPAIR). So a SOURCE scan for `td_xSet*(N` literals is
    vacuous -- the stub issues none, and `_bank_literals(stub_text)` is empty, which
    made the old `if banks:` assertion SKIP; a phantom `td_xSetInWave(5, ...)` would
    have slipped through. Assert instead on the stub's ACTUAL parsed bank
    representation: driven with the REAL host arm sequence it records ONLY banks 0-2,
    and an embedded bank-5 probe proves the max-bank guard is LIVE (it would FAIL if
    the stub ever recorded a bank > 2)."""
    from afm_litho.hardware import bridge_protocol as P

    def _recorded_banks(stub) -> list[int]:
        # the bank rides as the FIRST field of every out_binds / in_channels record.
        return [rec[0] for rec in stub.out_binds] + [rec[0] for rec in stub.in_channels]

    # 1) Feed the stub the EXACT banks the host arms (passes.py):
    #    OUT bank 0 (setpoint pair), OUT bank 1 (Output.A bias), IN bank 2
    #    (Input.A/XSensor capture pair). No in-wave capture bank beyond 2 exists.
    stub = FakeIgorBridge({})
    for line in (
        P.cmd_outpair(0, "0", "PIDSLoop.0.Setpoint", "XW", "PIDSLoop.1.Setpoint", "YW", -1.0),
        P.cmd_outwave(1, "0", "Output.A", "VW", -1.0),
        P.cmd_inpair(2, "0", "Input.A", "XSensor", 32, 1),
    ):
        stub._dispatch(line, None)
    banks = _recorded_banks(stub)
    assert banks, "stub recorded no banks -- the arm sequence or the parser moved"
    assert max(banks) <= 2, \
        f"stub modelled an in-/out-wave bank > 2 from the real arm seq: {sorted(set(banks))}"
    assert 3 not in banks                                # the reverted attempt's phantom bank

    # 2) NON-VACUITY: the parser surfaces any bank verbatim, so the max<=2 guard above
    #    is live -- a phantom bank-5 arm (the exact bug class) IS recorded and caught.
    phantom = FakeIgorBridge({})
    phantom._dispatch(P.cmd_inwave(5, "0", "Sum", 32, 1), None)
    assert max(_recorded_banks(phantom)) > 2             # guard (1) WOULD fail on this stub
