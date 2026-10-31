"""Drive LithographyRun against the fake Igor bridge, asserting the converted
waves and the per-object choreography against real calibration.

The expected numbers are computed INDEPENDENTLY here from the note's LVDT
calibration and the vendor timing rule, so this is a golden test of the driver,
not an echo of it.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from afm_litho import clock
from afm_litho.hardware import bridge_protocol as P
from afm_litho.hardware.igor_stub import FakeIgorBridge
from afm_litho.hardware.lithography_run import LithographyRun

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "SA40656B0002.ibw"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(),
                                reason="sample .ibw not present")


@pytest.fixture(autouse=True)
def _testing_clock(request):
    """The run's waits are skipped (and recorded). A test whose abort is set by another real
    thread during a wait is marked `time_sensitive` and keeps real time."""
    mode = (clock.TESTING_TIME_SENSITIVE if request.node.get_closest_marker("time_sensitive")
            else clock.TESTING)
    with clock.using(mode):
        yield


def _cal():
    from afm_litho.io.ibw import read_ibw
    n = read_ibw(SAMPLE).note_dict
    return {
        "x_sens": float(n["XLVDTSens"]), "x_off": float(n["XLVDTOffset"]),
        "y_sens": float(n["YLVDTSens"]), "y_off": float(n["YLVDTOffset"]),
    }


def _ar_get_deci(rate, clock=50_000):
    return max(int(round(clock / rate)), 1) if rate > 0 else 1


def _serve():
    stub = FakeIgorBridge.from_ibw(SAMPLE, done_delay=0.0)
    port = stub.serve_in_thread()
    return stub, port


def _run(port, **cfg):
    # zero the settle waits so the tests don't actually sleep 300 ms per move;
    # xy_withdraw_max_s/settle_s zero the verified-withdraw's own waits the same
    # way (docs/XY_SLEW_SPEC.md -- xy_transit's withdraw step, hw_withdraw).
    # first_object_settle_s zeroes the first object's free-air settle
    # (docs/PER_OBJECT_FREE_AIR_SPEC.md) the same way.
    kw = {"settle_move_s": 0.0, "cmd_gap_s": 0.0, "xy_withdraw_max_s": 0.0,
          "xy_withdraw_settle_s": 0.0, "first_object_settle_s": 0.0}
    kw.update(cfg)
    return LithographyRun.connect("127.0.0.1", port, **kw)


def test_start_parses_calibration():
    stub, port = _serve()
    try:
        run = _run(port)
        run.start()
    finally:
        stub.stop()
    assert run.cal["x_lvdt_sens"] == pytest.approx(7.4643e-6)
    assert run.cal["x_lvdt_offset"] == pytest.approx(-0.63)
    assert run.cal["y_lvdt_sens"] == pytest.approx(7.0428e-6)
    assert run.cal["y_lvdt_offset"] == pytest.approx(0.07)
    # the rest of the startup values are kept available too
    assert run.startup["pixels"] == pytest.approx(256)
    run.close()


def test_write_object_conversion_and_cycle():
    cal = _cal()
    stub, port = _serve()
    # xy_transit (docs/XY_SLEW_SPEC.md) reads the CURRENT tip position live off
    # XSensor/YSensor -- seed the stub at the LVDT offset (== the scanner centre,
    # um 0), matching this object's own first point (um 0) so the move is a ZERO
    # hop, same as the old software-assumed "starts at centre" default.
    stub.values["XSensor"], stub.values["YSensor"] = cal["x_off"], cal["y_off"]
    x_um, y_um, v_tip = [0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [3.0, 3.0, 3.0]
    amp_gain, setpoint, speed, retract, decim = 10.0, 0.4, 5.0, -10.0, 50
    try:
        run = _run(port, amp_gain=amp_gain, retract=retract, decimation=decim)
        run.start()
        res = run.write_object("obj0", x_um, y_um, v_tip,
                               setpoint=setpoint, speed=speed)
        # barrier: a round-trip guarantees the stub has processed the
        # fire-and-forget commands (incl. the post-DONE lift) before we assert.
        run.get_value("XSensor")
    finally:
        stub.stop()

    # --- converted waves, computed independently from the calibration ------
    exp_x = [xi * 1e-6 / cal["x_sens"] + cal["x_off"] for xi in x_um]
    exp_y = [yi * 1e-6 / cal["y_sens"] + cal["y_off"] for yi in y_um]
    exp_v = [vi / amp_gain for vi in v_tip]
    # These three points did not come from the compiler, so the driver fires them
    # with the last position repeated at 0 V up to 32 points: the controller holds
    # the last out-wave sample until the 32-sample capture fills.
    assert stub.waves["XLitho"] == pytest.approx(exp_x + [exp_x[-1]] * 29)
    assert stub.waves["YLitho"] == pytest.approx(exp_y + [exp_y[-1]] * 29)
    assert stub.waves["VLitho"] == pytest.approx(exp_v + [0.0] * 29)  # [0.3]*3, then 0 V

    # --- interp: signed ticks/point from the vendor timing rule ------------
    max_step = 1.0  # each step is 1 um along x
    exp_tpp = _ar_get_deci(speed / max_step)      # round(50000/5) = 10000
    exp_interp = -float(exp_tpp)
    assert stub.arms[0][0] == "obj0"
    assert stub.arms[0][1] == pytest.approx(exp_interp)
    assert res.interp == pytest.approx(exp_interp)

    # --- capture length is a multiple of 32 --------------------------------
    cap_len = stub.arms[0][2]
    assert cap_len % 32 == 0
    assert cap_len == 32                      # == the 32 points fired
    assert res.cap_len == cap_len
    # capture decimation now tracks the trajectory speed (one sample per
    # interpolated point), not the fixed self.decimation. It is |round(interp)|.
    assert stub.arms[0][3] == max(1, abs(int(round(exp_interp))))
    assert res.points == 32                   # the fired count
    assert not res.aborted

    # --- first-object withdraw -> XY transit move -> set down -> ... -> lift, in order
    # servo loops: the first object's withdraw re-arms loop 2 at the retract setpoint,
    # then X and Y are parked at the first point.
    assert stub.loops[0][:3] == (2, "Deflection", -10.0)
    assert stub.loops[1] == (0, "XSensor", pytest.approx(exp_x[0]),
                             pytest.approx(0.0), pytest.approx(0.0), "Output.X")
    assert stub.loops[2] == (1, "YSensor", pytest.approx(exp_y[0]),
                             pytest.approx(0.0), pytest.approx(0.0), "Output.Y")
    # loop-2 re-arms: the withdraw, the set-down at the litho setpoint, the lift to retract
    assert [lp[2] for lp in stub.loops if lp[0] == 2] == [
        pytest.approx(-10.0), pytest.approx(setpoint), pytest.approx(retract)]
    assert stub.sets == []
    # the move is a SINGLE straight-line out-wave PAIR on the XY loop setpoints to
    # the first point, fired as one event, NOT a RAMP (XY_SLEW_SPEC.md, 2026-09-16):
    # one OUTPAIR binding both PIDSLoop.0/1.Setpoint, then one FIRE, no RAMP at all
    # on either setpoint channel.
    assert not any(r[0] in ("PIDSLoop.0.Setpoint", "PIDSLoop.1.Setpoint")
                   for r in stub.ramps)
    sp_binds = [b for b in stub.out_binds
               if b[2] in ("PIDSLoop.0.Setpoint", "PIDSLoop.1.Setpoint")]
    assert len(sp_binds) == 2
    x_name = next(b[3] for b in sp_binds if b[2] == "PIDSLoop.0.Setpoint")
    y_name = next(b[3] for b in sp_binds if b[2] == "PIDSLoop.1.Setpoint")
    assert stub.waves[x_name][-1] == pytest.approx(exp_x[0])
    assert stub.waves[y_name][-1] == pytest.approx(exp_y[0])
    assert len(stub.fires) == 1
    run.close()


def test_move_speed_has_a_conservative_default():
    """The repositioning translate speed is a separate, conservative default."""
    stub, port = _serve()
    try:
        run = _run(port)
        assert run.move_speed_um_s == pytest.approx(5.0)
    finally:
        stub.stop()
    run.close()


def test_move_is_rate_limited_ramp_not_a_jump():
    """Two STROKES OF THE SAME SOURCE OBJECT (docs/XY_SLEW_SPEC.md rule 5 -- e.g.
    two rows of one filled shape, `source="shape"` both calls): the move between
    them is the OLD mild hover-lift (`_hover_move`) -- a RATE-LIMITED ramp of the
    loop setpoint at move_speed, the loops armed at the ramp START (the
    software-tracked previous point, `_last_xy_um`, NOT jumped straight to the
    target), and crucially NO verified withdraw between the strokes -- a fill row's
    gap inside one shape does not pay the full Z-loop open/close cycle. See test_move_between_distinct_objects_withdraws_first for
    the DIFFERENT-source case, which does withdraw."""
    cal = _cal()
    speed = 4.0

    def to_x(um):
        return um * 1e-6 / cal["x_sens"] + cal["x_off"]

    def to_y(um):
        return um * 1e-6 / cal["y_sens"] + cal["y_off"]

    # two strokes of ONE source object ("shape"): stroke 1's first point 10 um out
    # from center; stroke 2's first point elsewhere.
    s1 = ("shape#1", [10.0, 12.0], [0.0, 0.0], [3.0, 3.0])
    s2 = ("shape#2", [20.0, 22.0], [5.0, 5.0], [3.0, 3.0])
    stub, port = _serve()
    # stroke 1 is the FIRST move of a fresh run -> it goes through xy_transit's
    # LIVE XSensor/YSensor read (not _last_xy_um); seed it at the LVDT offset ==
    # um 0 (the scanner centre) so the arm-start assertion below matches the old
    # "starts at centre" expectation. Stroke 2 (the hover path) uses `_last_xy_um`
    # instead, so it needs no such seeding.
    stub.values["XSensor"], stub.values["YSensor"] = cal["x_off"], cal["y_off"]
    try:
        run = _run(port, move_speed_um_s=speed)
        run.start()
        run.write_object(*s1, setpoint=0.4, speed=5.0, source="shape")
        run.write_object(*s2, setpoint=0.4, speed=5.0, source="shape")
        run.get_value("XSensor")                 # barrier: stub processed everything
    finally:
        stub.stop()

    # The only verified withdraw is the FIRST object's (PER_OBJECT_FREE_AIR_SPEC);
    # none between the two strokes -- both belong to the same source.
    assert stub.stopped_loops == [] and stub.sets == []

    # loops: the first object's withdraw [Z], stroke 1's XY_TRANSIT arm [X, Y], its
    # set-down and lift [Z, Z], then stroke 2's _hover_move [Z lift, X, Y] and its
    # set-down and lift [Z, Z] -> indices 0 | 1,2 | 3,4 | 5,6,7 | 8,9.
    assert len(stub.loops) == 10
    # FIRST move: X/Y loops armed at the CENTER (0 um), not at the 10 um target.
    assert stub.loops[1][2] == pytest.approx(to_x(0.0))   # arm start = center
    assert stub.loops[1][2] != pytest.approx(to_x(10.0))  # NOT jumped to target
    # BETWEEN strokes: _hover_move's mild Z lift (loop 2, still CLOSED -- not a
    # withdraw), THEN X/Y armed at stroke 1's LAST point (12 um).
    assert stub.loops[5] == (2, "Deflection", pytest.approx(run.retract),
                             pytest.approx(0.0), pytest.approx(run.z_igain),
                             "Height")
    assert stub.loops[6][2] == pytest.approx(to_x(12.0))

    # stroke 1's move goes through xy_transit (fresh run, first move): a SINGLE
    # out-wave pair fired once (XY_SLEW_SPEC.md, 2026-09-16), NOT a RAMP -- to
    # stroke 1's first point.
    sp_binds = [b for b in stub.out_binds
               if b[2] in ("PIDSLoop.0.Setpoint", "PIDSLoop.1.Setpoint")]
    assert len(sp_binds) == 2
    x_name = next(b[3] for b in sp_binds if b[2] == "PIDSLoop.0.Setpoint")
    y_name = next(b[3] for b in sp_binds if b[2] == "PIDSLoop.1.Setpoint")
    assert stub.waves[x_name][-1] == pytest.approx(to_x(10.0))
    assert stub.waves[y_name][-1] == pytest.approx(to_y(0.0))
    assert len(stub.fires) == 1

    # stroke 2's move is the SAME-source `_hover_move` -- UNCHANGED, still a
    # RATE-LIMITED ramp of the loop setpoints, from stroke 1's last point (12) to
    # stroke 2's first point (20).
    x_ramps = [r for r in stub.ramps if r[0] == "PIDSLoop.0.Setpoint"]
    y_ramps = [r for r in stub.ramps if r[0] == "PIDSLoop.1.Setpoint"]
    assert len(x_ramps) == 1 and len(y_ramps) == 1
    assert x_ramps[0][1] == pytest.approx(to_x(20.0))
    assert x_ramps[0][2] == pytest.approx(speed * 1e-6 / cal["x_sens"])  # rate = move_speed
    assert y_ramps[0][1] == pytest.approx(to_y(5.0))
    assert y_ramps[0][2] == pytest.approx(speed * 1e-6 / cal["y_sens"])
    # maxsec stays within IR_SetRamp's 10 s cap.
    assert all(0 < r[3] <= 10.0 for r in x_ramps + y_ramps)
    run.close()


def test_move_between_distinct_objects_withdraws_first():
    """Two DISTINCT source objects (`source="A"` then `source="B"`,
    docs/XY_SLEW_SPEC.md rule 5): the move between them withdraws FIRST -- the tip
    is engaged from A's set-down -- loop 2 re-armed at -10 V, THEN
    ramps X/Y at move_speed, exactly like the FIRST move of a run. Compare
    test_move_is_rate_limited_ramp_not_a_jump, where two strokes of the SAME
    source object never withdraw."""
    cal = _cal()
    speed = 4.0

    def to_x(um):
        return um * 1e-6 / cal["x_sens"] + cal["x_off"]

    def to_y(um):
        return um * 1e-6 / cal["y_sens"] + cal["y_off"]

    # object A first point 10 um out from center; B first point elsewhere.
    a = ("A", [10.0, 12.0], [0.0, 0.0], [3.0, 3.0])
    b = ("B", [20.0, 22.0], [5.0, 5.0], [3.0, 3.0])
    stub, port = _serve()
    # xy_transit reads the tip's CURRENT position live off XSensor/YSensor rather
    # than a software-tracked "last point" -- seed the stub at the scanner centre
    # (um 0) for the FIRST move, then move it to A's last point (12 um) before B's
    # move, so the assertions below can tell the two moves' starts apart. A dumb
    # stub does not simulate the ramp's physical effect, so this stands in for it.
    stub.values["XSensor"], stub.values["YSensor"] = cal["x_off"], cal["y_off"]
    try:
        run = _run(port, move_speed_um_s=speed)
        run.start()
        run.write_object(*a, setpoint=0.4, speed=5.0, source="A")
        stub.values["XSensor"] = to_x(a[1][-1])       # simulate: tip now at A's last point
        stub.values["YSensor"] = to_y(a[2][-1])
        run.write_object(*b, setpoint=0.4, speed=5.0, source="B")
        run.get_value("XSensor")                 # barrier: stub processed everything
    finally:
        stub.stop()

    # loops per object: [Z withdraw, X, Y, Z set-down, Z lift] -> A: 0-4 ; B: 5-9.
    assert len(stub.loops) == 10
    # FIRST move: X/Y loops armed at the CENTER (0 um), not at the 10 um target.
    assert stub.loops[1][2] == pytest.approx(to_x(0.0))   # arm start = center
    assert stub.loops[1][2] != pytest.approx(to_x(10.0))  # NOT jumped to target
    # Every object withdraws first, by re-arming loop 2 at -10 V: A as the first
    # object (PER_OBJECT_FREE_AIR_SPEC), B because A engaged; B is armed at A's LAST
    # point (12 um).
    assert [stub.loops[i][:3] for i in (0, 5)] == [(2, "Deflection", -10.0)] * 2
    assert stub.stopped_loops == []
    assert stub.loops[6][2] == pytest.approx(to_x(12.0))

    # BOTH moves go through xy_transit (distinct sources each time): a SINGLE
    # out-wave pair fired once per move (XY_SLEW_SPEC.md, 2026-09-16), NEVER a
    # RAMP on either setpoint channel; the withdraw ramps nothing.
    assert stub.ramps == []
    sp_binds = [b for b in stub.out_binds
               if b[2] in ("PIDSLoop.0.Setpoint", "PIDSLoop.1.Setpoint")]
    assert len(sp_binds) == 4                       # two OUTPAIRs (A's move, B's move)
    assert len(stub.fires) == 2
    # B's move (the LAST one -- LithographyRun.td_xSetOutWavePair reuses FIXED
    # wave names across calls, the same /O-overwrite convention XLitho/YLitho use
    # per object, so only the most recent content survives to check here): to B's
    # first point.
    x_name_b = sp_binds[2][3] if sp_binds[2][2] == "PIDSLoop.0.Setpoint" else sp_binds[3][3]
    y_name_b = sp_binds[2][3] if sp_binds[2][2] == "PIDSLoop.1.Setpoint" else sp_binds[3][3]
    assert stub.waves[x_name_b][-1] == pytest.approx(to_x(20.0))
    assert stub.waves[y_name_b][-1] == pytest.approx(to_y(5.0))
    run.close()


def test_done_round_trips():
    stub, port = _serve()
    try:
        run = _run(port)
        run.start()
        # returns without hanging once DONE arrives
        res = run.write_object("solo", [0.0, 1.0], [0.0, 0.0], [5.0, 5.0],
                               setpoint=0.4, speed=2.0)
        assert res.name == "solo"
        assert not res.aborted
    finally:
        stub.stop()
    run.close()


def test_abort_is_handled_cleanly():
    stub, port = _serve()
    try:
        run = _run(port)
        run.start()
        how = run.abort()                        # out-of-band (TCP -- no COM in tests)
        assert how == "tcp"
        # abort is send-only now (no drain), so round-trip once as a barrier: the
        # stub has processed the ABORT (before this GETVALUE) by the time VALUE lands.
        run.get_value("XSensor")
        assert stub.aborts == 1
    finally:
        stub.stop()
    run.close()


def test_get_value_round_trips():
    stub, port = _serve()
    stub.values["XSensor"] = 1.25
    try:
        run = _run(port)
        run.start()
        assert run.get_value("XSensor") == pytest.approx(1.25)
    finally:
        stub.stop()
    run.close()


def test_serve_accepts_client_bridge():
    """Hardware topology: Python is the SERVER (LithographyRun.serve accepts),
    the bridge is the CLIENT and connects out -- the mirror of the other tests.
    Proves start()/write_object work identically regardless of who dialed."""
    from afm_litho.hardware.bridge_protocol import BridgeLink

    stub = FakeIgorBridge.from_ibw(SAMPLE, done_delay=0.0)
    srv, port = BridgeLink.listen("127.0.0.1", 0)
    result: dict = {}

    def drive():
        link = BridgeLink.accept_on(srv, timeout=5)
        srv.close()
        run = LithographyRun(link, settle_move_s=0.0, cmd_gap_s=0.0)
        run.start()
        result["cal"] = dict(run.cal)
        res = run.write_object("obj0", [0.0, 1.0], [0.0, 0.0], [5.0, 5.0],
                               setpoint=0.4, speed=2.0)
        result["done"] = not res.aborted
        run.close()

    t = threading.Thread(target=drive)
    t.start()
    stub.connect_in_thread("127.0.0.1", port)   # the bridge dials in
    t.join(timeout=8)
    stub.stop()

    assert not t.is_alive()                      # the run completed
    assert result["cal"]["x_lvdt_sens"] == pytest.approx(7.4643e-6)
    assert result["done"] is True
    assert stub.arms[0][0] == "obj0"


def test_write_object_returns_the_capture():
    """The bridge streams the measured input-wave capture back on completion;
    write_object collects the CAPTURE frames into the result."""
    stub = FakeIgorBridge.from_ibw(SAMPLE, done_delay=0.0,
                                   capture=lambda name, cap_len: {
                                       "Output.Z": [float(i) for i in range(cap_len)]})
    port = stub.serve_in_thread()
    try:
        run = _run(port)
        run.start()
        res = run.write_object("obj", [0.0, 1.0], [0.0, 0.0], [5.0, 5.0],
                               setpoint=0.4, speed=2.0)
        assert list(res.capture) == ["Output.Z"]
        assert len(res.capture["Output.Z"]) == res.cap_len
        assert res.capture["Output.Z"][:3] == [0.0, 1.0, 2.0]
        assert not res.aborted
    finally:
        stub.stop()
    run.close()


def test_cypher_engage_re_arms_loop2_at_contact_setpoint():
    """The Z set-down engages by RE-ARMING loop 2 at the
    CONTACT setpoint on the Height output (the proven scan/engage primitive), NOT by
    poking PIDSLoop.2.Setpoint -- the running Cypher Height loop ignores that poke, so
    a poke would leave the loop at its -10 arm setpoint and the tip pinned to the -10
    rail (no write). The lift re-arms at the retract setpoint the same way."""
    stub, port = _serve()
    setpoint, retract = 0.4, -10.0
    try:
        run = _run(port, retract=retract, z_out="Height")
        run.start()
        run.write_object("obj0", [0.0, 1.0], [0.0, 0.0], [3.0, 3.0],
                         setpoint=setpoint, speed=5.0)
        run.get_value("XSensor")                   # barrier
    finally:
        stub.stop()
    # NO bare PIDSLoop.2.Setpoint pokes on the Cypher -- every Z transition is a re-arm
    assert all(s[0] != "PIDSLoop.2.Setpoint" for s in stub.sets)
    # loop 2 re-armed on "Height" input Deflection: the first object's withdraw
    # (PER_OBJECT_FREE_AIR_SPEC; hw_withdraw's own retract gain), then set-down
    # (contact) and end lift (-10) at the contact gain.
    from afm_litho.hardware.hw_approach import _Z_RETRACT_IGAIN, _Z_RETRACT_SETPOINT
    z_loops = [lp for lp in stub.loops if lp[0] == 2]
    assert len(z_loops) == 3
    assert all(lp[1] == "Deflection" and lp[5] == "Height" for lp in z_loops)
    assert z_loops[0][2] == pytest.approx(_Z_RETRACT_SETPOINT)   # first-object withdraw
    assert z_loops[0][4] == pytest.approx(_Z_RETRACT_IGAIN)
    for lp in z_loops[1:]:
        assert lp[4] == pytest.approx(run.z_igain)     # integral-only, contact gain
    assert z_loops[1][2] == pytest.approx(setpoint)     # set-down ENGAGE
    assert z_loops[2][2] == pytest.approx(retract)      # end-of-object lift
    run.close()


def test_cypher_between_object_withdraw_re_arms_loop2_on_height():
    """CYPHER, two DISTINCT source objects, built as host/litho_worker.py builds the run
    (z_out="Height"): the between-object withdraw (xy_transit ->
    hw_withdraw) re-arms loop 2 at the retract setpoint on "Height". "Output.Z" is
    inert on the Cypher, so a withdraw naming it does not retract
    (INSTRUMENT_MODEL_SPEC 13.1 item 44)."""
    from afm_litho.hardware.hw_approach import _Z_RETRACT_IGAIN, _Z_RETRACT_SETPOINT
    cal = _cal()
    stub, port = _serve()
    stub.values["XSensor"], stub.values["YSensor"] = cal["x_off"], cal["y_off"]
    try:
        run = _run(port, retract=-10.0, z_out="Height")
        run.start()
        run.write_object("A", [10.0, 12.0], [0.0, 0.0], [3.0, 3.0],
                         setpoint=0.4, speed=5.0, source="A")
        run.get_value("XSensor")                   # barrier: A's end-of-object lift recorded
        n_after_a = len(stub.loops)
        run.write_object("B", [20.0, 22.0], [5.0, 5.0], [3.0, 3.0],
                         setpoint=0.4, speed=5.0, source="B")
        run.get_value("XSensor")                   # barrier
    finally:
        stub.stop()
    z_b = [lp for lp in stub.loops[n_after_a:] if lp[0] == 2]
    # B's first Z command is the between-object withdraw: the Cypher retract re-arm
    withdraw = z_b[0]
    assert withdraw[1] == "Deflection"
    assert withdraw[2] == pytest.approx(_Z_RETRACT_SETPOINT)
    assert withdraw[4] == pytest.approx(_Z_RETRACT_IGAIN)
    assert withdraw[5] == "Height"
    # no Z command anywhere in the run names Output.Z, as a loop output or a ramp
    assert all(lp[5] == "Height" for lp in stub.loops if lp[0] == 2)
    assert not any(r[0] == "Output.Z" for r in stub.ramps)
    assert 2 not in stub.stopped_loops
    run.close()


def test_write_object_sanitizes_whitespace_in_name():
    """A space in the object name would split the space-delimited ARMFIRE command
    and break the CAPTURE/DONE <name> the completion matches on -- hanging the
    write. write_object collapses whitespace to underscores so the round-trip
    still completes. A short timeout makes a regression fail fast, not in 30 s."""
    stub, port = _serve()
    try:
        run = _run(port, timeout=3.0)
        run.start()
        res = run.write_object("wire 3 left", [0.0, 1.0], [0.0, 0.0], [2.0, 2.0],
                               setpoint=0.3, speed=5.0)
    finally:
        stub.stop()
    assert res.name == "wire_3_left"           # whitespace collapsed
    assert not res.aborted                      # completion matched -> no hang
    run.close()


# ---------------------------------------------- per-object free air -----------
# docs/PER_OBJECT_FREE_AIR_SPEC.md: every object of a hardware write is withdraw ->
# XY move -> (first object: settle 3 s) -> free-air read (3-read agreement) ->
# approach -> write. A RELATIVE setpoint is re-resolved against each object's own
# free air; an ABSOLUTE one is used as sent.

def _record_order(stub, monkeypatch, defl_reads=None):
    """Record every bridge line the stub receives, in order, plus the run's
    sleeps of 1 s or more the clock recorded (as "SLEEP <s>"; shorter ones are command
    gaps), placed before the next line the stub receives. `defl_reads`: a list the
    DeflectionValue reads are answered from, in order (else the stub's value)."""
    lines: list[str] = []
    seen = 0
    orig = stub._dispatch
    reads = list(defl_reads) if defl_reads is not None else None

    def rec(line, link):
        nonlocal seen
        waits = clock.waits()
        lines.extend(f"SLEEP {w.seconds:g}" for w in waits[seen:]
                     if w.kind == "sleep" and w.seconds >= 1.0)
        seen = len(waits)
        lines.append(line)
        if reads and line.split(" ")[0] == P.GETVALUE and line.endswith(" DeflectionValue"):
            stub.values["DeflectionValue"] = reads.pop(0)
        orig(line, link)

    stub._dispatch = rec
    return lines


def _events(lines, retract=-10.0):
    """Classify the recorded lines into the per-object choreography."""
    from afm_litho.hardware.hw_approach import _Z_RETRACT_IGAIN, _Z_RETRACT_SETPOINT
    ev = []
    for ln in lines:
        p = ln.split(" ")
        verb = p[0]
        if verb == P.STOPLOOP and p[1] == "2":
            ev.append("withdraw")                                   # MFP-3D withdraw
        elif verb == P.SERVOLOOP and p[1] == "2":
            sp, ig = float(p[3]), float(p[5])
            if sp == _Z_RETRACT_SETPOINT and ig == _Z_RETRACT_IGAIN:
                ev.append("withdraw")                               # Cypher withdraw
            elif sp == retract:
                ev.append("lift")
            else:
                ev.append(("approach", round(sp, 6)))
        elif verb == P.SETVALUE and p[1] == "PIDSLoop.2.Setpoint":
            sp = float(p[2])
            ev.append("lift" if sp == retract else ("approach", round(sp, 6)))
        elif verb == P.FIRE:
            ev.append("move")
        elif verb == P.GETVALUE and p[2] == "DeflectionValue":
            ev.append("free_air")
        elif verb in (P.ARMFIRE, P.LITHOCHAIN):
            ev.append("write")
        elif verb == "SLEEP":
            ev.append(("settle", float(p[1])))
    return ev


_PLATFORMS = [pytest.param({"z_out": "Height"}, id="height")]


def _two_objects(run, stub, setpoint, free_air_b=None):
    run.write_object("A", [10.0, 12.0], [0.0, 0.0], [3.0, 3.0],
                     setpoint=setpoint, speed=5.0, source="A")
    if free_air_b is not None:
        stub.values["DeflectionValue"] = free_air_b
    run.write_object("B", [20.0, 22.0], [5.0, 5.0], [3.0, 3.0],
                     setpoint=setpoint, speed=5.0, source="B")
    run.get_value("XSensor")                       # barrier: B's lift recorded


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_two_object_relative_write_reads_free_air_before_each_approach(platform,
                                                                       monkeypatch):
    """(a)+(e) Relative setpoint, two distinct objects, MFP-3D and Cypher: exact order
    withdraw -> move -> [first: settle 3 s] -> 3 free-air reads -> approach -> write
    for BOTH objects; each approach = that object's own free air + the relative."""
    from afm_litho.hardware.lithography_run import FIRST_OBJECT_SETTLE_S, WriteSetpoint
    stub, port = _serve()
    stub.values["DeflectionValue"] = -0.2
    lines = _record_order(stub, monkeypatch)
    logs: list[str] = []
    try:
        run = _run(port, first_object_settle_s=FIRST_OBJECT_SETTLE_S,
                   log=logs.append, **platform)
        run.start()
        _two_objects(run, stub, WriteSetpoint(0.25, relative=True), free_air_b=-0.3)
    finally:
        stub.stop()
    assert FIRST_OBJECT_SETTLE_S == 3.0
    assert _events(lines) == [
        "withdraw", "move", ("settle", 3.0), "free_air", "free_air", "free_air",
        ("approach", 0.05), "write", "lift",
        "withdraw", "move", "free_air", "free_air", "free_air",
        ("approach", -0.05), "write", "lift"]
    per_object = [m for m in logs if "free air" in m]
    assert len(per_object) == 2
    assert "free air -0.2000 V, setpoint 0.0500 V" in per_object[0]
    assert "free air -0.3000 V, setpoint -0.0500 V" in per_object[1]
    run.close()


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_two_object_absolute_write_keeps_its_setpoint(platform, monkeypatch):
    """(b) An absolute setpoint (FLEX; a bare float too) is used as sent on every
    object, whatever the free air reads; the reads still happen before each approach."""
    from afm_litho.hardware.lithography_run import WriteSetpoint
    for sp in (WriteSetpoint(0.15, relative=False), 0.15):
        stub, port = _serve()
        stub.values["DeflectionValue"] = -0.2
        lines = _record_order(stub, monkeypatch)
        try:
            run = _run(port, **platform)
            run.start()
            _two_objects(run, stub, sp, free_air_b=-0.3)
        finally:
            stub.stop()
        assert _events(lines) == [
            "withdraw", "move", "free_air", "free_air", "free_air",
            ("approach", 0.15), "write", "lift",
            "withdraw", "move", "free_air", "free_air", "free_air",
            ("approach", 0.15), "write", "lift"]
        run.close()


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_disagreeing_reads_on_a_later_object_reuse_the_previous_setpoint(platform,
                                                                         monkeypatch):
    """(c) Object B's three free-air reads disagree (one triple, no retry): no stop, B is approached at A's absolute setpoint, one log line says so."""
    from afm_litho.hardware.lithography_run import WriteSetpoint
    stub, port = _serve()
    reads = [-0.2, -0.2, -0.2] + [0.0, 0.5, 1.0]
    lines = _record_order(stub, monkeypatch, defl_reads=reads)
    logs: list[str] = []
    try:
        run = _run(port, log=logs.append, **platform)
        run.start()
        _two_objects(run, stub, WriteSetpoint(0.25, relative=True))
    finally:
        stub.stop()
    assert _events(lines) == [
        "withdraw", "move", "free_air", "free_air", "free_air",
        ("approach", 0.05), "write", "lift",
        "withdraw", "move"] + ["free_air"] * 3 + [("approach", 0.05), "write", "lift"]
    disagree = [m for m in logs if "disagree" in m]
    assert len(disagree) == 1
    assert "write B" in disagree[0]
    assert "previous object's setpoint 0.0500 V" in disagree[0]
    run.close()


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_disagreeing_reads_on_the_first_object_refuse_a_relative_write(platform,
                                                                        monkeypatch):
    """(d) A RELATIVE write's first-object reads disagree: no earlier setpoint exists,
    so the write is refused with the tip withdrawn -- no approach, nothing armed or
    fired."""
    from afm_litho.hardware.lithography_run import FreeAirRefused, WriteSetpoint
    stub, port = _serve()
    lines = _record_order(stub, monkeypatch,
                          defl_reads=[0.0, 0.5, 1.0])
    logs: list[str] = []
    try:
        run = _run(port, log=logs.append, **platform)
        run.start()
        with pytest.raises(FreeAirRefused):
            run.write_object("A", [10.0, 12.0], [0.0, 0.0], [3.0, 3.0],
                             setpoint=WriteSetpoint(0.25, relative=True), speed=5.0,
                             source="A")
        run.get_value("XSensor")                   # barrier
    finally:
        stub.stop()
    assert _events(lines) == ["withdraw", "move"] + ["free_air"] * 3
    assert stub.arms == [] and stub.lithochains == []
    refused = [m for m in logs if "refused" in m]
    assert len(refused) == 1 and "first object" in refused[0]
    run.close()


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_disagreeing_reads_never_refuse_or_change_an_absolute_write(platform,
                                                                    monkeypatch):
    """An ABSOLUTE setpoint is never refused or changed over free air: reads
    that disagree on the first object AND on a later one only log; both objects are
    approached at the setpoint as sent, in the same sequence."""
    from afm_litho.hardware.lithography_run import WriteSetpoint
    stub, port = _serve()
    bad = [0.0, 0.5, 1.0]
    lines = _record_order(stub, monkeypatch, defl_reads=bad + bad)
    logs: list[str] = []
    try:
        run = _run(port, log=logs.append, **platform)
        run.start()
        _two_objects(run, stub, WriteSetpoint(0.15, relative=False))
    finally:
        stub.stop()
    assert _events(lines) == (
        ["withdraw", "move"] + ["free_air"] * 3 + [("approach", 0.15), "write", "lift"]
        + ["withdraw", "move"] + ["free_air"] * 3
        + [("approach", 0.15), "write", "lift"])
    disagree = [m for m in logs if "disagree" in m]
    assert len(disagree) == 2
    assert "write A" in disagree[0] and "write B" in disagree[1]
    assert all("setpoint 0.1500 V (absolute, as sent)" in m for m in disagree)
    assert not any("refused" in m for m in logs)
    run.close()


@pytest.mark.parametrize("platform", _PLATFORMS)
@pytest.mark.time_sensitive
def test_abort_during_the_first_object_settle_stops_before_the_approach(platform,
                                                                        monkeypatch):
    """Abort stops everything: the abort event set during the first object's settle
    ends the wait at once; no free-air read, no approach, nothing armed or fired."""
    from afm_litho.hardware.lithography_run import WriteAborted, WriteSetpoint
    stub, port = _serve()
    lines = _record_order(stub, monkeypatch)
    ab = threading.Event()
    threading.Timer(0.2, ab.set).start()
    try:
        run = _run(port, first_object_settle_s=30.0, abort_event=ab, **platform)
        run.start()
        t0 = time.monotonic()
        with pytest.raises(WriteAborted):
            run.write_object("A", [10.0, 12.0], [0.0, 0.0], [3.0, 3.0],
                             setpoint=WriteSetpoint(0.25, relative=True), speed=5.0,
                             source="A")
        assert time.monotonic() - t0 < 10.0          # the 30 s settle was cut short
        run.get_value("XSensor")                   # barrier
    finally:
        stub.stop()
    assert _events(lines) == ["withdraw", "move"]
    assert stub.arms == [] and stub.lithochains == []
    run.close()


@pytest.mark.parametrize("platform", _PLATFORMS)
@pytest.mark.time_sensitive
def test_abort_during_the_free_air_reads_stops_before_the_approach(platform, monkeypatch):
    """The abort set between two free-air reads: no further read, no approach."""
    from afm_litho.hardware.lithography_run import WriteAborted, WriteSetpoint
    stub, port = _serve()
    ab = threading.Event()
    lines = _record_order(stub, monkeypatch)
    orig = stub._dispatch

    def abort_on_first_read(line, link):
        # Set before the reply goes out: the host cannot see the first read's value, and so
        # cannot start a second read, until the abort is already set.
        if line.endswith(" DeflectionValue"):
            ab.set()
        orig(line, link)

    stub._dispatch = abort_on_first_read
    try:
        run = _run(port, abort_event=ab, **platform)
        run.start()
        with pytest.raises(WriteAborted):
            run.write_object("A", [10.0, 12.0], [0.0, 0.0], [3.0, 3.0],
                             setpoint=WriteSetpoint(0.25, relative=True), speed=5.0,
                             source="A")
        run.get_value("XSensor")                   # barrier
    finally:
        stub.stop()
    assert _events(lines) == ["withdraw", "move", "free_air"]
    assert stub.arms == [] and stub.lithochains == []
    run.close()


def _abort_after_send(run, ab, starts, armed=lambda: True):
    """Set `ab` on the run's own thread right after it sends a line starting with
    `starts` (while `armed()`), so the abort lands at that exact point of the cycle."""
    orig = run._send

    def send(cmd):
        orig(cmd)
        if cmd.startswith(starts) and armed():
            ab.set()

    run._send = send


@pytest.mark.parametrize("platform", _PLATFORMS)
@pytest.mark.time_sensitive
def test_abort_while_the_waves_load_never_fires(platform, monkeypatch):
    """Abort set while the litho waves load (after the set-down): no ARMFIRE, no
    LITHOCHAIN, and no later Z command (no lift, no re-arm) that could undo SafePark."""
    from afm_litho.hardware.lithography_run import WriteAborted, WriteSetpoint
    stub, port = _serve()
    stub.values["DeflectionValue"] = -0.2
    lines = _record_order(stub, monkeypatch)
    ab = threading.Event()
    try:
        run = _run(port, abort_event=ab, **platform)
        run.start()
        _abort_after_send(run, ab, f"{P.WAVE} XLitho ")
        with pytest.raises(WriteAborted, match="before the fire"):
            run.write_object("A", [10.0, 12.0], [0.0, 0.0], [3.0, 3.0],
                             setpoint=WriteSetpoint(0.25, relative=True), speed=5.0,
                             source="A")
        run.get_value("XSensor")                   # barrier
    finally:
        stub.stop()
    assert _events(lines) == ["withdraw", "move"] + ["free_air"] * 3 + [("approach", 0.05)]
    assert stub.arms == [] and stub.lithochains == []
    run.close()


@pytest.mark.parametrize("platform", _PLATFORMS)
@pytest.mark.time_sensitive
def test_abort_during_a_hover_move_stops_before_the_next_approach(platform, monkeypatch):
    """Abort set during the same-source hover ramp to stroke 2: stroke 2 is neither
    approached nor armed; stroke 1 alone was written."""
    from afm_litho.hardware.lithography_run import WriteAborted, WriteSetpoint
    stub, port = _serve()
    stub.values["DeflectionValue"] = -0.2
    lines = _record_order(stub, monkeypatch)
    ab = threading.Event()
    second = threading.Event()
    sp = WriteSetpoint(0.25, relative=True)
    try:
        run = _run(port, abort_event=ab, **platform)
        run.start()
        _abort_after_send(run, ab, f"{P.RAMP} PIDSLoop.0.Setpoint ", armed=second.is_set)
        run.write_object("A#1", [10.0, 12.0], [0.0, 0.0], [3.0, 3.0],
                         setpoint=sp, speed=5.0, source="A")
        second.set()
        with pytest.raises(WriteAborted):
            run.write_object("A#2", [20.0, 22.0], [5.0, 5.0], [3.0, 3.0],
                             setpoint=sp, speed=5.0, source="A")
        run.get_value("XSensor")                   # barrier
    finally:
        stub.stop()
    assert _events(lines) == (["withdraw", "move"] + ["free_air"] * 3
                              + [("approach", 0.05), "write", "lift", "lift"])
    assert [a[0] for a in stub.arms] == ["A#1"] and stub.lithochains == []
    run.close()


@pytest.mark.parametrize("relative", [True, False], ids=["relative", "absolute"])
@pytest.mark.parametrize("platform", _PLATFORMS)
def test_a_dead_free_air_read_stops_the_write_before_the_approach(platform, relative,
                                                                   monkeypatch):
    """A DEAD deflection read (NaN, or a read error, which the reader returns as
    NaN) stops the write, relative or absolute: no approach, the tip stays withdrawn,
    and the reason names the object. Before object B (A written), and on the first."""
    from afm_litho.hardware.lithography_run import FreeAirReadFailed, WriteSetpoint
    sp = WriteSetpoint(0.25 if relative else 0.15, relative=relative)
    approach = 0.05 if relative else 0.15
    for first in (False, True):
        stub, port = _serve()
        nan = float("nan")
        reads = ([nan] if first else [-0.2, -0.2, -0.2, nan])
        lines = _record_order(stub, monkeypatch, defl_reads=reads)
        logs: list[str] = []
        try:
            run = _run(port, log=logs.append, **platform)
            run.start()
            with pytest.raises(FreeAirReadFailed, match="read failed on " +
                               ("A" if first else "B")):
                _two_objects(run, stub, sp)
            run.get_value("XSensor")               # barrier
        finally:
            stub.stop()
        done_a = [] if first else ["free_air"] * 3 + [("approach", approach), "write",
                                                       "lift", "withdraw", "move"]
        assert _events(lines) == ["withdraw", "move"] + done_a + ["free_air"]
        assert len(stub.arms) == (0 if first else 1)
        assert run.object_setpoints[-1]["setpoint_v"] is None
        assert "read failed" in logs[-1]
        run.close()


# ------------------------------------------ Z-drive withdrawn check -----------
# docs/PER_OBJECT_FREE_AIR_SPEC.md: a free-air read counts only if the run's Z piezo
# drive (its z_out alias: "Height" on the Cypher, "Output.Z" on the MFP-3D) reads
# BELOW 1.0 V; otherwise it is handled like reads that disagree. A dead Z read stops.

def _zchan(platform):
    return platform.get("z_out", "Output.Z")


def _write_n(run, n, sp):
    for k in range(n):
        name = "ABCDE"[k]
        run.write_object(name, [10.0 + k, 12.0 + k], [0.0, 0.0], [3.0, 3.0],
                         setpoint=sp, speed=5.0, source=name)


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_z_drive_left_high_does_not_let_the_setpoint_climb(platform, monkeypatch):
    """The reviewer's repro: from object B on the Z drive stays at 70 V after the
    withdraw and DeflectionValue follows the last contact setpoint. Without the check
    each object's setpoint would be last + relative (climbing); with it B..E reuse A's."""
    from afm_litho.hardware.lithography_run import WriteSetpoint
    stub, port = _serve()
    stub.values["DeflectionValue"] = -0.2
    orig = stub._dispatch

    def follow(line, link):
        orig(line, link)
        p = line.split(" ")
        if p[0] == P.SETVALUE and p[1] == "PIDSLoop.2.Setpoint" and float(p[2]) != -10.0:
            stub.values["DeflectionValue"] = float(p[2])
        if p[0] == P.SERVOLOOP and p[1] == "2" and float(p[3]) not in (-10.0,):
            stub.values["DeflectionValue"] = float(p[3])
        if p[0] in (P.ARMFIRE,):
            stub.values[_zchan(platform)] = 70.0          # tip stays down from here on

    stub._dispatch = follow
    lines = _record_order(stub, monkeypatch)
    logs: list[str] = []
    try:
        run = _run(port, log=logs.append, **platform)
        run.start()
        _write_n(run, 5, WriteSetpoint(0.25, relative=True))
        run.get_value("XSensor")
    finally:
        stub.stop()
    approaches = [e[1] for e in _events(lines) if isinstance(e, tuple) and e[0] == "approach"]
    assert approaches == [pytest.approx(0.05)] * 5
    assert [r["setpoint_v"] for r in run.object_setpoints] == [pytest.approx(0.05)] * 5
    high = [m for m in logs if "tip not confirmed withdrawn" in m]
    assert len(high) == 4 and all("previous object's setpoint 0.0500 V" in m for m in high)
    run.close()


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_z_drive_high_on_the_first_object(platform, monkeypatch):
    """Z drive >= 1.0 V before the first object's read: a relative write is refused
    before any approach; an absolute one proceeds at the setpoint as sent."""
    from afm_litho.hardware.lithography_run import FreeAirRefused, WriteSetpoint
    for relative in (True, False):
        stub, port = _serve()
        stub.values[_zchan(platform)] = 1.0                  # "below 1.0" fails at 1.0
        stub.values["DeflectionValue"] = -0.2
        lines = _record_order(stub, monkeypatch)
        logs: list[str] = []
        try:
            run = _run(port, log=logs.append, **platform)
            run.start()
            sp = WriteSetpoint(0.25 if relative else 0.15, relative=relative)
            if relative:
                with pytest.raises(FreeAirRefused, match="not confirmed withdrawn"):
                    _write_n(run, 1, sp)
                refused = [m for m in logs if "refused" in m]
                assert len(refused) == 1
                assert "before any approach" in refused[0]
                assert "tip withdrawn" not in refused[0]
            else:
                _write_n(run, 1, sp)
            run.get_value("XSensor")
        finally:
            stub.stop()
        approaches = [e for e in _events(lines) if isinstance(e, tuple)
                      and e[0] == "approach"]
        assert approaches == ([] if relative else [("approach", 0.15)])
        run.close()


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_z_drive_at_minus_ten_passes(platform, monkeypatch):
    """A retract can leave the Z drive at -10 V: below 1.0 V, so the read counts."""
    from afm_litho.hardware.lithography_run import WriteSetpoint
    stub, port = _serve()
    stub.values[_zchan(platform)] = -10.0
    stub.values["DeflectionValue"] = -0.3
    lines = _record_order(stub, monkeypatch)
    try:
        run = _run(port, **platform)
        run.start()
        _write_n(run, 1, WriteSetpoint(0.25, relative=True))
        run.get_value("XSensor")
    finally:
        stub.stop()
    assert ("approach", -0.05) in _events(lines)
    assert run.object_setpoints[0]["free_air_v"] == pytest.approx(-0.3)
    run.close()


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_a_dead_z_drive_read_stops_the_write(platform, monkeypatch):
    """A NaN Z drive stops the write before the approach, relative or absolute."""
    from afm_litho.hardware.lithography_run import FreeAirReadFailed, WriteSetpoint
    for sp in (WriteSetpoint(0.25, relative=True), WriteSetpoint(0.15, relative=False)):
        stub, port = _serve()
        stub.values[_zchan(platform)] = float("nan")
        lines = _record_order(stub, monkeypatch)
        try:
            run = _run(port, **platform)
            run.start()
            with pytest.raises(FreeAirReadFailed, match="Z drive .* read failed on A"):
                _write_n(run, 1, sp)
            run.get_value("XSensor")
        finally:
            stub.stop()
        ev = _events(lines)
        assert "free_air" not in ev
        assert not any(isinstance(e, tuple) and e[0] == "approach" for e in ev)
        assert stub.arms == []
        run.close()
