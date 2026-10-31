"""The XY slew guard (docs/XY_SLEW_SPEC.md, hazard #11): every hardware XY move
that is not a raster or a write stroke goes through `hardware.xy_transit.
xy_transit` -- withdraw (if engaged) -> arm loops 0/1 at current -> ONE straight-
line out-wave pair on the setpoint channels, fired as a single event -> wait for
completion -> verify arrival by readback (retry once if short) -> re-approach
(only if asked). Exercised off-instrument against `FakeIgorBridge`, which mirrors
real Igor's SERVOLOOP/STOPLOOP/RAMP/OUTPAIR/FIRE/GETVALUE handling.

2026-09-16: this replaced an earlier two-independent-`ramp_output`-calls
implementation (no pacing, no arrival check) with the single out-wave-pair move
xy_transit now shares with `lithography_run.LithographyRun`'s own proven litho-
stroke fire/wait cycle (`_fire_and_wait`) -- see xy_transit.py's module docstring
and lithography_run.py's `_fire_and_wait`/`td_xSetOutWavePair`/`fire`.
"""

from __future__ import annotations

import asyncio

import pytest

from afm_litho import clock
from afm_litho.hardware.hardware_arc import HardwareARC
from afm_litho.hardware.igor_stub import FakeIgorBridge
from afm_litho.hardware.lithography_run import LithographyRun
from afm_litho.hardware.xy_transit import (
    ARRIVAL_TOL_UM,
    CMD_GAP_S,
    MOVE_SPEED_UM_S,
    xy_transit,
)
from afm_litho.litho.compiler import DEFAULT_INTERP_UM
from afm_litho.service.hw_scan import run_hw_scan_chained
from afm_litho.service.passes import ScanConfig

CAL = {"x_lvdt_sens": 7.46e-6, "x_lvdt_offset": -0.63,
       "y_lvdt_sens": 7.04e-6, "y_lvdt_offset": 0.07,
       "z_lvdt_sens": 5e-6, "z_lvdt_offset": 0.0}

_X_SP = "PIDSLoop.0.Setpoint"
_Y_SP = "PIDSLoop.1.Setpoint"


def _connect(values=None):
    stub = FakeIgorBridge(dict(values or {}), done_delay=0.0)
    port = stub.serve_in_thread()
    hw = HardwareARC.connect("127.0.0.1", port)
    return stub, hw


def _v_of(x_um: float, y_um: float, cal: dict = CAL) -> tuple[float, float]:
    """(x_v, y_v) for a lab-um position under `cal` -- the same LVDT convention
    xy_transit itself uses, spelled out here so a test can stage a `FakeIgorBridge`
    readback without reaching into xy_transit's own private helpers."""
    return (x_um * 1e-6 / cal["x_lvdt_sens"] + cal["x_lvdt_offset"],
            y_um * 1e-6 / cal["y_lvdt_sens"] + cal["y_lvdt_offset"])


def _setpoint_out_binds(stub) -> list[tuple]:
    """Every recorded OUTPAIR bind (see `FakeIgorBridge.out_binds`) touching the
    XY setpoint channels -- each OUTPAIR call appends two entries (one per
    channel), so this list has 2 entries per xy_transit move attempt."""
    return [b for b in stub.out_binds if b[2] in (_X_SP, _Y_SP)]


# ============================== (a) scan start ==============================

def test_scan_start_withdraws_arms_one_wavepair_and_approaches():
    """A scan starting far from the tip's CURRENT XY (engaged from a previous
    action, and this scan will itself engage) withdraws first, then moves as ONE
    out-wave pair on the setpoint channels (first sample = current, last sample =
    destination), fired as a single event and verified by readback -- never two
    independent `ramp_output` calls. Only THEN does the scan's own approach/
    loop-arm run -- never a re-approach from xy_transit itself (the scan's engage
    covers it)."""
    # tip currently far from the raster's first point (0.63 V offset == um 0 for
    # this calibration; sit it well away so the move is a real, nonzero hop).
    stub, hw = _connect({"XSensor": 2.0, "YSensor": 2.0})
    cfg = ScanConfig(size_um=5.0, pixels=32, lines=2, engage=True, setpoint=0.3,
                     mode="contact")
    msgs: list[dict] = []

    async def broadcast(m):
        msgs.append(m)

    try:
        asyncio.run(run_hw_scan_chained(hw, cfg, CAL, broadcast, engaged=True))
    finally:
        hw.close()
        stub.stop()

    t = stub.transcript
    # withdraw first: loop 2 re-armed at -10 V (both platforms); no loop is stopped and
    # NO ramp ever touches a setpoint channel.
    assert stub.loops[0][:3] == (2, "Deflection", -10.0)
    assert stub.stopped_loops == []
    assert not any(r[0] in (_X_SP, _Y_SP) for r in stub.ramps)
    z_ramp_i = t.index("SERVOLOOP")

    # exactly one OUTPAIR on the setpoint pair, then exactly one FIRE for it --
    # with an XSensor/YSensor INPAIR armed BEFORE that FIRE (the Cypher "no input
    # bank armed -> immediate DONE" fix), capLen a multiple of 32.
    sp_binds = _setpoint_out_binds(stub)
    assert len(sp_binds) == 2
    assert len(stub.fires) == 1
    outpair_i = t.index("OUTPAIR")
    inpair_i = t.index("INPAIR")
    fire_i = t.index("FIRE")
    assert z_ramp_i < outpair_i < inpair_i < fire_i
    xy_in = [c for c in stub.in_channels if c[2] in ("XSensor", "YSensor")]
    assert len(xy_in) == 2
    assert {c[2] for c in xy_in} == {"XSensor", "YSensor"}
    assert all(c[3] % 32 == 0 for c in xy_in)             # capLen a multiple of 32

    # the bound wave's first sample is the CURRENT position, the last sample is
    # the raster's first point -- both in LVDT volts, and the point count matches
    # the litho stroke's own arc-length-at-DEFAULT_INTERP_UM sampling convention
    # for a straight line at MOVE_SPEED_UM_S.
    x_name = next(b[3] for b in stub.out_binds if b[2] == _X_SP)
    y_name = next(b[3] for b in stub.out_binds if b[2] == _Y_SP)
    xs = stub.waves[x_name]
    ys = stub.waves[y_name]
    # the seeded XSensor/YSensor readback IS the starting volts (GETVALUE returns
    # the stub's raw scalar; xy_transit converts volts -> um internally).
    assert xs[0] == pytest.approx(2.0)
    assert ys[0] == pytest.approx(2.0)
    assert len(xs) == len(ys)
    # exact destination is the raster's first point in scanner um; recompute the
    # expected point count from the ACTUAL commanded distance between the two
    # endpoints (volts -> um via the same calibration), rather than re-deriving
    # the raster geometry here.
    dx_um = (xs[-1] - xs[0]) * CAL["x_lvdt_sens"] * 1e6
    dy_um = (ys[-1] - ys[0]) * CAL["y_lvdt_sens"] * 1e6
    moved_um = (dx_um ** 2 + dy_um ** 2) ** 0.5
    assert moved_um > 1.0                       # a real, nonzero hop
    expected_n = max(2, round(moved_um / DEFAULT_INTERP_UM) + 1)
    assert len(xs) == expected_n

    # approach/loop-arm: the scan's own SERVOLOOP arms (X=0, Y=1, Z=2 engage) --
    # they come from run_hw_scan_chained's driver(), AFTER xy_transit's withdraw
    # + move. is_cypher False here, so the Z engage is a SERVOLOOP too.
    servoloop_indices = [i for i, v in enumerate(t) if v == "SERVOLOOP"]
    after_fire = [i for i in servoloop_indices if i > fire_i]
    assert len(after_fire) >= 3
    by_index = {lp[0]: lp for lp in stub.loops}
    assert 2 in by_index and by_index[2][2] == pytest.approx(0.3)   # Z engaged at setpoint
    assert 0 in by_index and 1 in by_index                          # X/Y loop armed

    # two XY-transit log lines: the per-fire "wave ended at ..., readback ..."
    # diagnostic, then the one-line summary with the ", verified" suffix and a
    # per-phase timing breakdown appended.
    xy_logs = [m for m in msgs if m.get("type") == "log"
              and str(m.get("msg", "")).startswith("XY transit:")]
    assert len(xy_logs) == 2
    assert any(m["msg"].startswith("XY transit: wave ended at") for m in xy_logs)
    summary = next(m["msg"] for m in xy_logs if ", verified" in m["msg"])
    assert "um at 5 um/s, tip withdrawn, verified" in summary
    assert summary.endswith("]") and "[arm " in summary and "move " in summary


def test_scan_start_not_engaged_skips_withdraw():
    """The typical scan-start state: the tip is NOT engaged on entry (default
    `engaged=False`) -- the move still fires (rule 1 is unconditional) but no
    withdraw command is sent at all (nothing to withdraw from)."""
    stub, hw = _connect({"XSensor": 0.5, "YSensor": 0.5})
    cfg = ScanConfig(size_um=5.0, pixels=32, lines=2, engage=False)
    msgs: list[dict] = []

    async def broadcast(m):
        msgs.append(m)

    try:
        asyncio.run(run_hw_scan_chained(hw, cfg, CAL, broadcast))   # engaged defaults False
    finally:
        hw.close()
        stub.stop()

    assert stub.stopped_loops == []                     # no withdraw
    assert not any(r[0] == "Output.Z" for r in stub.ramps)
    assert not any(r[0] in (_X_SP, _Y_SP) for r in stub.ramps)   # never a setpoint RAMP
    sp_binds = _setpoint_out_binds(stub)
    assert len(sp_binds) == 2                            # the move still fires
    assert len(stub.fires) == 1
    xy_logs = [m for m in msgs if m.get("type") == "log"
              and str(m.get("msg", "")).startswith("XY transit:")]
    assert len(xy_logs) == 2
    summary = next(m["msg"] for m in xy_logs if ", verified" in m["msg"])
    assert "um at 5 um/s, tip was not engaged, verified" in summary


# ============================ (b) two-object write ===========================

def test_two_object_write_withdraws_moves_approaches_between_strokes():
    """First object: no withdraw (fresh run, never engaged). Second object: the
    tip is engaged from the first object's set-down, so its move withdraws ->
    fires the single out-wave pair on the setpoint channels -> approaches
    (set-down) -> THEN fires the stroke's own ARMFIRE -- in that order."""
    from afm_litho.hardware.igor_stub import values_from_note
    note = {"XLVDTSens": CAL["x_lvdt_sens"], "XLVDTOffset": CAL["x_lvdt_offset"],
            "YLVDTSens": CAL["y_lvdt_sens"], "YLVDTOffset": CAL["y_lvdt_offset"]}
    stub = FakeIgorBridge(values_from_note(note), done_delay=0.0)
    stub.values["XSensor"], stub.values["YSensor"] = CAL["x_lvdt_offset"], CAL["y_lvdt_offset"]
    port = stub.serve_in_thread()
    try:
        run = LithographyRun.connect("127.0.0.1", port, settle_move_s=0.0,
                                     cmd_gap_s=0.0, xy_withdraw_max_s=0.0,
                                     xy_withdraw_settle_s=0.0)
        run.start()
        run.write_object("a", [0.0, 1.0], [0.0, 0.0], [3.0, 3.0], setpoint=0.4, speed=5.0)
        run.get_value("XSensor")            # barrier: "a"'s trailing lift is fire-
                                             # and-forget: wait for the stub to have
                                             # processed it before snapshotting below.
        n_before = len(stub.transcript)
        run.write_object("b", [0.0, 1.0], [0.0, 0.0], [3.0, 3.0], setpoint=0.4, speed=5.0)
        run.get_value("XSensor")            # barrier
    finally:
        stub.stop()
    run.close()

    t = stub.transcript[n_before:]          # only the SECOND object's commands
    assert "SERVOLOOP" in t                 # withdraw: loop 2 re-armed at -10 V
    assert stub.stopped_loops == []
    z_setpoints = [lp[2] for lp in stub.loops if lp[0] == 2]
    assert z_setpoints[-3:] == [-10.0, 0.4, -10.0]          # withdraw, set-down, lift
    assert not any(r[0] in (_X_SP, _Y_SP) for r in stub.ramps)   # never a setpoint RAMP
    # order: withdraw -> OUTPAIR(setpoint pair) -> FIRE -> approach (set-down
    # re-arm) -> stroke (ARMFIRE) -> lift (re-arm)
    withdraw_i = t.index("SERVOLOOP")
    outpair_i = t.index("OUTPAIR")
    fire_i = t.index("FIRE")
    armfire_i = t.index("ARMFIRE")
    arms_after_fire = [i for i, v in enumerate(t) if v == "SERVOLOOP" and i > fire_i]
    assert withdraw_i < outpair_i < fire_i
    assert any(i < armfire_i for i in arms_after_fire)     # set-down before the stroke fires
    assert any(i > armfire_i for i in arms_after_fire)     # lift after
    # exactly one OUTPAIR/FIRE pair for THIS object's move (the first object's
    # move contributed the other entries before n_before).
    sp_binds_after = [b for b in stub.out_binds[len(stub.out_binds) - 2:] if b[2] in (_X_SP, _Y_SP)]
    assert len(sp_binds_after) == 2


# ==================== (c) failure mid-move -> withdrawn =========================

def test_xy_transit_failure_mid_move_leaves_tip_withdrawn_and_raises():
    """Any failure inside xy_transit's move (a broken out-wave-pair arm here) ->
    the tip is withdrawn and the exception propagates -- CLAUDE.md's hardware
    fail-safe rule. `engaged=False` isolates that the ONLY withdraw came from the
    failure path (nothing to withdraw from on entry)."""
    stub, hw = _connect({"XSensor": 0.0, "YSensor": 0.0})

    def _boom(bank, event, ch_a, wave_a, ch_b, wave_b, interp):
        raise RuntimeError("simulated mid-arm bridge failure")

    hw.td_xSetOutWavePair = _boom

    try:
        with pytest.raises(RuntimeError, match="simulated mid-arm"):
            xy_transit(hw, CAL, 10.0, 10.0, engaged=False, reapproach=False,
                      withdraw_max_s=0.0, withdraw_settle_s=0.0,
                      ramp_settle_s=0.0, cmd_gap_s=0.0)
    finally:
        hw.close()
        stub.stop()

    # the failure handler withdrew despite engaged=False on entry: loop 2 re-armed at
    # -10 V, hw_withdraw's sequence on both platforms.
    assert stub.stopped_loops == []
    z_loops = [lp for lp in stub.loops if lp[0] == 2]
    assert len(z_loops) == 1 and z_loops[0][2] == -10.0 and z_loops[0][5] == "Height"
    assert stub.ramps == []


# ==================== withdraw verify_timeout -> WARNING logged ==================

def test_withdraw_verify_timeout_logs_a_warning():
    """`hw_withdraw`'s own verify_timeout handling is untouched (a separate
    decision) -- xy_transit just has to make a timed-out verify VISIBLE, since a
    silent unverified withdraw right before a big XY move is exactly the case
    worth a loud WARNING in the cockpit log."""
    stub, hw = _connect({"XSensor": 5.0, "YSensor": 5.0})

    class BridgeReadTimeout(Exception):
        """Named exactly like the real one -- hw_approach._read_ex matches on the
        exception's CLASS NAME, not isinstance, so this is enough to simulate a
        timed-out verify read without importing the bridge's real transport."""

    real_td_read = hw.td_ReadValue

    def _flaky(name, **kw):
        # xy_transit's own position reads (XSensor/YSensor) must still succeed;
        # only the withdraw's verify snapshot (Deflection / height) times out.
        if name in ("Deflection", "height"):
            raise BridgeReadTimeout("simulated stalled verify read")
        return real_td_read(name, **kw)

    hw.td_ReadValue = _flaky
    msgs: list[str] = []

    try:
        xy_transit(hw, CAL, 10.0, 10.0, engaged=True, reapproach=False,
                  withdraw_max_s=0.0, withdraw_settle_s=0.0, ramp_settle_s=0.0,
                  cmd_gap_s=0.0, log=msgs.append)
    finally:
        hw.close()
        stub.stop()

    assert [lp[2] for lp in stub.loops if lp[0] == 2] == [-10.0]   # the withdraw still ran
    warnings = [m for m in msgs if m.startswith("WARNING")]
    assert len(warnings) == 1 and "verify" in warnings[0].lower()
    # the transit's own one-line summary still logs too, now with ", verified"
    # followed by the per-phase timing breakdown.
    assert any(m.startswith("XY transit:") and ", verified [" in m and m.endswith("]")
              for m in msgs)


# ==================== arrival verify: short -> retry -> raise ==================

def test_move_short_of_destination_retries_once_then_raises():
    """Arrival is verified by an XSensor/YSensor readback after the move's fire
    completes (docs/XY_SLEW_SPEC.md rule 3): landing short of the target retries
    the SAME move once (from wherever it actually landed); still short after that
    raises. `FakeIgorBridge` never simulates real motion (its `values` dict is
    static unless changed), so the readback is staged directly here: the tip
    starts at (0, 0) and reports short of the (10, 10) target on BOTH the first
    attempt and the retry."""
    stub, hw = _connect({"XSensor": 0.0, "YSensor": 0.0})
    start_x_v, start_y_v = _v_of(0.0, 0.0)
    short_x_v, short_y_v = _v_of(9.0, 9.0)      # ~1.41 um short of (10, 10)
    calls = {"n": 0}
    real_read = hw.td_ReadValue

    def _staged(name, **kw):
        if name not in ("XSensor", "YSensor"):
            return real_read(name, **kw)
        calls["n"] += 1
        if calls["n"] <= 2:                      # the very first current-position read
            return start_x_v if name == "XSensor" else start_y_v
        return short_x_v if name == "XSensor" else short_y_v   # every verify read

    hw.td_ReadValue = _staged
    try:
        with pytest.raises(RuntimeError, match="short of destination"):
            xy_transit(hw, CAL, 10.0, 10.0, engaged=False, reapproach=False,
                      withdraw_max_s=0.0, withdraw_settle_s=0.0,
                      ramp_settle_s=0.0, cmd_gap_s=0.0, retry_verify_gap_s=0.0)
    finally:
        hw.close()
        stub.stop()

    # two full attempts: two OUTPAIR arms, two FIREs -- never a third.
    sp_binds = _setpoint_out_binds(stub)
    assert len(sp_binds) == 4
    assert len(stub.fires) == 2
    # the fixed tolerance floor ARRIVAL_TOL_UM (1.0 um, 2026-09-16) is what
    # made 1.41 um short a genuine failure, not a rounding fluke.
    assert ARRIVAL_TOL_UM < 1.41


# ============================== pacing ==============================

def test_pacing_between_bridge_commands():
    """Every bridge command xy_transit itself issues is separated by
    >= `cmd_gap_s` (docs/XY_SLEW_SPEC.md): the OLD two-ramp move fired
    `ramp_output` on X then Y back to back with NO gap at all -- part of how the
    two axes could desync on the Cypher. Runs on the testing clock, which records
    each commanded wait and never blocks."""
    stub, hw = _connect({"XSensor": 0.0, "YSensor": 0.0})

    with clock.using(clock.TESTING):
        try:
            xy_transit(hw, CAL, 5.0, 5.0, engaged=False, reapproach=False,
                      withdraw_max_s=0.0, withdraw_settle_s=0.0,
                      ramp_settle_s=0.0, cmd_gap_s=0.05)
        finally:
            hw.close()
            stub.stop()
        sleeps = [w.seconds for w in clock.waits()
                  if w.kind == "sleep" and w.caller.startswith("hardware/xy_transit.py")]

    # ir_SetPISLoop(X), ir_SetPISLoop(Y), td_xSetOutWavePair, td_xSetInWavePair,
    # fire -- 5 paced gaps for one successful (no-retry) attempt (the INPAIR arm
    # added 2026-09-16 for the capture-verified fire).
    assert len(sleeps) == 5
    assert all(s == pytest.approx(0.05) for s in sleeps)


def test_cmd_gap_s_default_matches_lithography_run_convention():
    """`CMD_GAP_S` is pulled out to one module constant rather than repeated as a
    magic number (docs/XY_SLEW_SPEC.md) -- and `xy_transit`'s default keeps it, so
    a real (non-test) caller that never overrides `cmd_gap_s` still gets pacing."""
    assert CMD_GAP_S == pytest.approx(0.02)
    assert xy_transit.__kwdefaults__["cmd_gap_s"] == pytest.approx(CMD_GAP_S)
    assert MOVE_SPEED_UM_S == pytest.approx(5.0)


# =================== capture-armed fire (the Cypher defect fix) ===================

def test_inpair_armed_before_fire_with_cap_len_multiple_of_32():
    """The fix (docs/XY_SLEW_SPEC.md, 2026-09-16 Cypher defect): an XSensor/
    YSensor INPAIR is armed on the SAME bank/event as the setpoint OUTPAIR,
    BEFORE the FIRE -- so the fire blocks on the bridge's own real completion
    instead of the "no input bank armed -> DONE now" shortcut a bare OUTPAIR/
    FIRE takes on the real controller. capLen is the out-wave's own point count
    (n), rounded UP to a multiple of 32 (the in-wave rule)."""
    stub, hw = _connect({"XSensor": 0.0, "YSensor": 0.0})
    try:
        xy_transit(hw, CAL, 30.0, 0.0, engaged=False, reapproach=False,
                  withdraw_max_s=0.0, withdraw_settle_s=0.0,
                  ramp_settle_s=0.0, cmd_gap_s=0.0, retry_verify_gap_s=0.0)
    finally:
        hw.close()
        stub.stop()

    t = stub.transcript
    assert t.index("OUTPAIR") < t.index("INPAIR") < t.index("FIRE")
    xy_in = [c for c in stub.in_channels if c[2] in ("XSensor", "YSensor")]
    assert len(xy_in) == 2
    assert {c[2] for c in xy_in} == {"XSensor", "YSensor"}
    cap_len = xy_in[0][3]
    assert cap_len % 32 == 0
    assert xy_in[0][3] == xy_in[1][3]                 # X and Y armed with the SAME capLen
    # capLen is n (the bound wave's own point count) padded UP to the next
    # multiple of 32 -- not just "happens to be a multiple of 32".
    x_name = next(b[3] for b in stub.out_binds if b[2] == _X_SP)
    n = len(stub.waves[x_name])
    assert cap_len == n + (32 - n % 32) % 32


def test_bare_fire_sees_a_stale_readback_but_xy_transit_waits_for_the_move():
    """The Cypher defect (docs/XY_SLEW_SPEC.md, 2026-09-16): on the real
    controller, a bare OUTPAIR/FIRE with no input bank armed completes at DONE
    immediately, even though the tip has not actually finished moving.
    `FakeIgorBridge.sensor_lag_s` reproduces that here (a real-seconds delay,
    decoupled from DONE's own still-immediate timing): the readback only
    reaches the commanded destination `sensor_lag_s` after the OUTPAIR bind --
    read right after a bare FIRE, it is still stale. This is the failure mode
    the next test's xy_transit call (capture-armed, waits) does not have."""
    lag_s = 0.08
    stub, hw = _connect({"XSensor": 0.0, "YSensor": 0.0})
    stub.sensor_lag_s = lag_s
    try:
        x1_v, y1_v = _v_of(1.0, 1.0)
        # bank 0 / event "0" -- the SAME pair xy_transit itself uses -- with NO
        # INPAIR armed (the old, bug-prone shape).
        hw.td_xSetOutWavePair(0, "0", _X_SP, [0.0, x1_v], _Y_SP, [0.0, y1_v], -1)
        hw.fire("0", 2)
        assert hw.td_ReadValue("XSensor") == pytest.approx(0.0)   # still stale
    finally:
        hw.close()
        stub.stop()


def test_transit_verifies_via_done_gating_alone_with_the_guard_disabled():
    """xy_transit's OWN move is capture-armed (the fix): `FakeIgorBridge` gates
    an armed FIRE's DONE on `sensor_lag_s` (not just `done_delay`) when
    XSensor/YSensor are among the armed channels, so `arc.fire` genuinely does
    not return until the tip has "landed". `ramp_settle_s=0` disables the
    SEPARATE belt-and-braces guard entirely, so ONLY that DONE-gating can make
    this verify -- proving the capture arm itself is load-bearing, not the
    belt-and-braces fallback (see the sibling test below for that one alone).

    Mutation check (reported, not automated): commenting out xy_transit's
    `arc.td_xSetInWavePair(...)` call makes this SAME test fail with "short of
    destination after one retry" -- the FIRE reverts to the no-input-armed,
    immediate-DONE branch (`igor_stub.py`'s FIRE dispatch), so the readback
    right after `arc.fire()` returns is still stale."""
    lag_s = 0.08
    stub, hw = _connect({"XSensor": 0.0, "YSensor": 0.0})
    stub.sensor_lag_s = lag_s
    try:
        result = xy_transit(hw, CAL, 1.0, 1.0, engaged=False, reapproach=False,
                            withdraw_max_s=0.0, withdraw_settle_s=0.0,
                            ramp_settle_s=0.0, cmd_gap_s=0.0, retry_verify_gap_s=0.0)
    finally:
        hw.close()
        stub.stop()

    assert result["verified"] is True


def test_belt_and_braces_alone_rescues_an_early_done_despite_the_capture_arm():
    """Defense in depth (module docstring): even with the capture armed, an
    anomalous controller could still report DONE before the capture data it
    just promised is really ready. `sensor_lag_gates_done=False` forces exactly
    that in the stub (an armed FIRE's DONE uses `done_delay` alone, ignoring
    `sensor_lag_s` -- see FakeIgorBridge.__init__), bypassing the DONE-gating
    the capture arm normally buys. With `ramp_settle_s > 0` the belt-and-braces
    guard ALONE -- not the capture-arm's own DONE-gating, which this stub
    setting defeats on purpose -- still catches it: it sleeps out the
    remainder of the move's own planned duration before the readback, so the
    tip has landed by the time xy_transit reads."""
    lag_s = 0.08
    stub, hw = _connect({"XSensor": 0.0, "YSensor": 0.0})
    stub.sensor_lag_s = lag_s
    stub.sensor_lag_gates_done = False        # DONE ignores sensor_lag_s -> "early"
    try:
        result = xy_transit(hw, CAL, 1.0, 1.0, engaged=False, reapproach=False,
                            withdraw_max_s=0.0, withdraw_settle_s=0.0,
                            ramp_settle_s=0.05, cmd_gap_s=0.0, retry_verify_gap_s=0.0)
    finally:
        hw.close()
        stub.stop()

    assert result["verified"] is True


# ======================= retry safety: disagreeing double read =====================

def test_retry_disagreeing_double_read_raises_without_rearming_loops():
    """Retry safety (docs/XY_SLEW_SPEC.md rule 3): a short landing is never
    retried from a SINGLE readback. Two reads `retry_verify_gap_s` apart must
    AGREE (within tolerance) before the second is trusted as the retry's start
    -- if they disagree, the move raises INSTEAD of re-arming loops 0/1 for a
    second attempt (never yank the tip toward an unknown position)."""
    stub, hw = _connect({"XSensor": 0.0, "YSensor": 0.0})
    start_x_v, start_y_v = _v_of(0.0, 0.0)
    short_x_v, short_y_v = _v_of(9.0, 9.0)        # attempt 1's short readback
    disagree_x_v, disagree_y_v = _v_of(7.0, 7.0)  # the second (disagreeing) read
    calls = {"n": 0}
    real_read = hw.td_ReadValue

    def _staged(name, **kw):
        if name not in ("XSensor", "YSensor"):
            return real_read(name, **kw)
        calls["n"] += 1
        if calls["n"] <= 2:
            return start_x_v if name == "XSensor" else start_y_v
        if calls["n"] <= 4:                       # attempt 1's own readback (= read1)
            return short_x_v if name == "XSensor" else short_y_v
        return disagree_x_v if name == "XSensor" else disagree_y_v   # read2

    hw.td_ReadValue = _staged
    msgs: list[str] = []
    try:
        with pytest.raises(RuntimeError, match="untrusted position"):
            xy_transit(hw, CAL, 10.0, 10.0, engaged=False, reapproach=False,
                      withdraw_max_s=0.0, withdraw_settle_s=0.0,
                      ramp_settle_s=0.0, cmd_gap_s=0.0, retry_verify_gap_s=0.0,
                      log=msgs.append)
    finally:
        hw.close()
        stub.stop()

    # only ONE attempt ever fired -- a disagreeing double read means NO retry.
    sp_binds = _setpoint_out_binds(stub)
    assert len(sp_binds) == 2
    assert len(stub.fires) == 1
    # loop 0 + loop 1 armed ONCE -- never re-armed for a phantom retry.
    assert sorted(lp[0] for lp in stub.loops) == [0, 1, 2]   # XY once each + the loop-2 withdraw
    # the failure path still withdraws (CLAUDE.md's hardware rule), even though
    # engaged=False on entry.
    assert [lp[2] for lp in stub.loops if lp[0] == 2] == [-10.0]
    assert any("read1" in m and "read2" in m for m in msgs)


# ========================= zero / short-distance moves =========================

def test_zero_distance_move_is_well_formed_and_verifies():
    """A destination equal to the tip's CURRENT position (dist == 0) still runs
    the FULL sequence -- arm loops, upload a wave pair with a capture, fire,
    wait for real DONE, verify -- NOT a short-circuited no-op (2026-09-16).
    The wave itself must be well-formed: n >= 2 samples, capLen a multiple of
    32, no division by zero in the rate/duration math."""
    x0_v, y0_v = _v_of(0.0, 0.0)
    stub, hw = _connect({"XSensor": x0_v, "YSensor": y0_v})
    try:
        result = xy_transit(hw, CAL, 0.0, 0.0, engaged=False, reapproach=False,
                            withdraw_max_s=0.0, withdraw_settle_s=0.0,
                            ramp_settle_s=0.0, cmd_gap_s=0.0, retry_verify_gap_s=0.0)
    finally:
        hw.close()
        stub.stop()

    assert result["verified"] is True
    assert result["distance_um"] == pytest.approx(0.0)
    assert len(stub.fires) == 1                    # the full sequence still ran
    x_name = next(b[3] for b in stub.out_binds if b[2] == _X_SP)
    n = len(stub.waves[x_name])
    assert n >= 2
    xy_in = [c for c in stub.in_channels if c[2] in ("XSensor", "YSensor")]
    assert len(xy_in) == 2 and all(c[3] % 32 == 0 for c in xy_in)


def test_half_micron_move_is_well_formed_and_verifies():
    """A very short (0.5 um) move: the same well-formedness contract as a zero-
    distance move -- n >= 2, capLen a multiple of 32 -- and verifies."""
    x0_v, y0_v = _v_of(0.0, 0.0)
    stub, hw = _connect({"XSensor": x0_v, "YSensor": y0_v})
    try:
        result = xy_transit(hw, CAL, 0.5, 0.0, engaged=False, reapproach=False,
                            withdraw_max_s=0.0, withdraw_settle_s=0.0,
                            ramp_settle_s=0.0, cmd_gap_s=0.0, retry_verify_gap_s=0.0)
    finally:
        hw.close()
        stub.stop()

    assert result["verified"] is True
    assert result["distance_um"] == pytest.approx(0.5)
    x_name = next(b[3] for b in stub.out_binds if b[2] == _X_SP)
    n = len(stub.waves[x_name])
    assert n >= 2
    xy_in = [c for c in stub.in_channels if c[2] in ("XSensor", "YSensor")]
    assert len(xy_in) == 2 and all(c[3] % 32 == 0 for c in xy_in)


# ============================== timing_s ==============================

def test_timing_s_keys_exist_and_are_monotonic():
    """`timing_s` (module docstring, 2026-09-16 -- "3 s with no way to attribute
    it"): {step: seconds since entry} for entry_readback, withdraw_done (only
    when a withdraw ran), loops_armed, wave_uploaded, fired, done_received,
    readback_done -- a per-phase breakdown, not just a single "it was slow"
    symptom. Runs on the testing clock (virtual time, advanced by each commanded
    wait) rather than real wall-clock waits."""
    stub, hw = _connect({"XSensor": 2.0, "YSensor": 2.0})

    with clock.using(clock.TESTING):
        try:
            result = xy_transit(hw, CAL, 20.0, 20.0, engaged=True, reapproach=False,
                                withdraw_max_s=0.0, withdraw_settle_s=0.0,
                                ramp_settle_s=0.1, cmd_gap_s=0.0, retry_verify_gap_s=0.0)
        finally:
            hw.close()
            stub.stop()

    timing = result["timing_s"]
    keys = ["entry_readback", "withdraw_done", "loops_armed", "wave_uploaded",
           "fired", "done_received", "readback_done"]
    assert all(k in timing for k in keys)      # engaged=True -> withdraw_done present
    values = [timing[k] for k in keys]
    assert values == sorted(values)            # monotonic (non-decreasing)
