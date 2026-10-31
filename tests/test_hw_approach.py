"""Real approach/withdraw on the hardware backend: engage arms the deflection->Z
feedback loop; withdraw re-arms it at -10 V with output minimum 0 so Height parks at 0 V."""

from afm_litho.hardware.hardware_arc import HardwareARC
from afm_litho.hardware.igor_stub import FakeIgorBridge
from afm_litho.hardware.hw_approach import (
    hw_engage,
    hw_read_zdrive,
    hw_set_zgain,
    hw_withdraw,
)


def _connect(defl: float = 1.5):
    stub = FakeIgorBridge({"Deflection": defl, "height": 20.0}, done_delay=0.0)
    port = stub.serve_in_thread()
    return stub, HardwareARC.connect("127.0.0.1", port)


def _connect_ac(amp: float = 0.65):
    """AC-mode stub: the feedback channel is Amplitude, not Deflection."""
    stub = FakeIgorBridge({"Amplitude": amp, "Deflection": 0.0, "height": 20.0},
                          done_delay=0.0)
    port = stub.serve_in_thread()
    return stub, HardwareARC.connect("127.0.0.1", port)


def test_hw_engage_arms_the_z_feedback_loop_and_confirms_setpoint():
    stub, hw = _connect(defl=1.5)
    try:
        # stub reads Deflection == 1.5, so a 1.5 V setpoint converges on the first
        # poll (no wait): reached_setpoint True.
        snap = hw_engage(hw, setpoint=1.5, pgain=0.0, igain=25.0)
    finally:
        hw.close()
        stub.stop()
    # loop 2: Deflection -> Height at the positive setpoint and gains
    by_index = {lp[0]: lp for lp in stub.loops}
    assert 2 in by_index
    _, inp, sp, pg, ig, outp = by_index[2]
    assert inp == "Deflection" and outp == "Height"
    assert sp == 1.5 and pg == 0.0 and ig == 2500.0   # panel igain 25 * IScale 100
    assert snap["engaged"] is True and snap["reached_setpoint"] is True
    assert snap["defl_v"] == 1.5
    # engaged arm keeps the full -10..150 output range (servoloop-limits fix,
    # 2026-09-16) -- only withdraw narrows it.
    limits2 = [lim for lp, lim in zip(stub.loops, stub.loop_limits) if lp[0] == 2]
    assert limits2[-1] == (-10.0, 150.0)


# --- The Z Height-loop (loop 2) OUTPUT is "Height" on both platforms (Cypher: the live
# PIDS panel; MFP-3D: the decision of 2026-10-27, ir_readAlias("Height") = Output.Z
# there). Arming/ramping the wrong output drives NOTHING (tip crash / no retract).
# The INPUT stays "Deflection" on both platforms.


def _make_cypher(arc):
    arc.startup = {"microscope_id": 4}      # cMicroscopeCypher == 4
    return arc


def test_cypher_engage_arms_loop2_on_height_output():
    stub, hw = _connect(defl=1.5)
    _make_cypher(hw)
    try:
        hw_engage(hw, setpoint=1.5, pgain=0.0, igain=25.0, settle_s=0.0)
    finally:
        hw.close()
        stub.stop()
    _, inp, _sp, _pg, _ig, outp = {lp[0]: lp for lp in stub.loops}[2]
    assert inp == "Deflection"              # input unchanged
    assert outp == "Height"                 # Cypher Z output, NOT "Output.Z"


def test_cypher_withdraw_rearms_loop2_through_the_height_servo():
    # Cypher retract goes THROUGH the loop: re-arm loop 2 (Deflection->Height) at the -10
    # retract setpoint (validated command). NO stop-loop, NO ramp (Height is a
    # loop-driven servo output, not a rampable DAC).
    stub, hw = _connect()
    _make_cypher(hw)
    try:
        hw_engage(hw, setpoint=1.0, pgain=0.0, igain=10.0, settle_s=0.0)
        hw_withdraw(hw, max_seconds=0.0, settle_margin=0.0)
    finally:
        hw.close()
        stub.stop()
    # the LAST loop-2 arm is the retract arm: Deflection -> Height, setpoint -10, igain 1000
    loops2 = [lp for lp in stub.loops if lp[0] == 2]
    _, inp, sp, _pg, ig, outp = loops2[-1]
    assert inp == "Deflection" and outp == "Height"
    assert sp == -10.0 and ig == 1000.0
    assert 2 not in stub.stopped_loops                  # NOT stopped -- retract is through the loop
    assert not any(r[0] in ("Height", "Output.Z") for r in stub.ramps)  # NO ramp
    # servoloop-limits fix (2026-09-16): the withdraw's 0 output-min must actually
    # reach Igor (it used to be dropped, and AFM_SetupServoLoop hardcoded -10/150,
    # so Height parked at -10 V instead of 0).
    limits2 = [lim for lp, lim in zip(stub.loops, stub.loop_limits) if lp[0] == 2]
    assert limits2[-1] == (0.0, 150.0)


def test_hw_read_zdrive_reads_height_on_both_platforms():
    # The rail-dwell / telemetry Z read follows arc.z_out_channel: the Height alias on
    # every platform. The stub holds a distinct value on the dead Output.Z.
    stub = FakeIgorBridge({"Deflection": 0.0, "Output.Z": 5.0, "Height": 99.0},
                          done_delay=0.0)
    port = stub.serve_in_thread()
    hw = HardwareARC.connect("127.0.0.1", port)
    try:
        assert hw.is_cypher is False
        assert hw_read_zdrive(hw) == 99.0
        _make_cypher(hw)
        assert hw.is_cypher is True
        assert hw_read_zdrive(hw) == 99.0
    finally:
        hw.close()
        stub.stop()


def test_mfp3d_withdraw_rearms_loop2_like_the_cypher():
    # No microscope_id (MFP-3D): the same re-arm, no STOPLOOP, no ramp (2026-10-27).
    stub, hw = _connect()
    try:
        hw_engage(hw, setpoint=1.0, pgain=0.0, igain=10.0, settle_s=0.0)
        snap = hw_withdraw(hw, max_seconds=0.0, settle_margin=0.0)
    finally:
        hw.close()
        stub.stop()
    assert hw.is_cypher is False
    _, inp, sp, _pg, ig, outp = [lp for lp in stub.loops if lp[0] == 2][-1]
    assert (inp, sp, ig, outp) == ("Deflection", -10.0, 1000.0, "Height")
    assert 2 not in stub.stopped_loops and stub.ramps == []
    limits2 = [lim for lp, lim in zip(stub.loops, stub.loop_limits) if lp[0] == 2]
    assert limits2[-1] == (0.0, 150.0)
    assert snap["engaged"] is False


def test_hw_set_zgain_rearms_the_z_loop_live():
    """Live gain tuning: re-arm loop 2 (Deflection->Height) with a new igain at the
    same setpoint, no withdraw. The last loop-2 arm carries the new gain."""
    stub, hw = _connect()
    try:
        hw_engage(hw, setpoint=1.0, pgain=0.0, igain=10.0, settle_s=0.0)
        snap = hw_set_zgain(hw, setpoint=1.0, pgain=0.0, igain=500.0)
    finally:
        hw.close()
        stub.stop()
    loops2 = [lp for lp in stub.loops if lp[0] == 2]
    assert loops2[-1][1] == "Deflection" and loops2[-1][5] == "Height"
    assert loops2[-1][4] == 50000.0                # panel igain 500 * IScale 100
    assert snap["engaged"] is True


# --- AC / tapping mode (SAFETY-CRITICAL): Amplitude channel + NEGATED gain sign.
# Pinned to the v16 source ARGetImagingMode cACMode branch: GainSign=-1,
# Input="Amplitude" (HardWare.ipf:8964-8968), sign applied to P and I gains
# (HardWare.ipf:9388-9391). Contact must stay Deflection + positive gain.


def test_hw_engage_ac_uses_amplitude_channel_and_negated_gain():
    """AC engage: loop 2 servos on Amplitude at a positive absolute setpoint with a
    NEGATED integral gain (amplitude falls on approach). Poll reads Amplitude, so the
    0.65 V setpoint converges on the first poll (stub Amplitude == 0.65)."""
    stub, hw = _connect_ac(amp=0.65)
    try:
        snap = hw_engage(hw, setpoint=0.65, pgain=0.0, igain=25.0, mode="ac")
    finally:
        hw.close()
        stub.stop()
    by_index = {lp[0]: lp for lp in stub.loops}
    assert 2 in by_index
    _, inp, sp, pg, ig, outp = by_index[2]
    assert inp == "Amplitude" and outp == "Height"      # AC feedback channel
    assert sp == 0.65                                      # absolute amplitude setpoint
    assert ig == -2500.0            # panel igain 25 * IScale 100 * GainSign(-1) NEGATED
    assert pg == 0.0                                       # 0 gain, sign is moot
    # reached_setpoint proves the poll read Amplitude (0.65), not Deflection (0.0).
    assert snap["engaged"] is True and snap["reached_setpoint"] is True


def test_hw_engage_ac_negates_a_nonzero_pgain_too():
    """GainSign multiplies BOTH P and I (HardWare.ipf:9388-9391), so a nonzero AC
    pgain is negated as well -- not just the integral gain."""
    stub, hw = _connect_ac(amp=0.65)
    try:
        hw_engage(hw, setpoint=0.65, pgain=4.0, igain=10.0, mode="ac", settle_s=0.0)
    finally:
        hw.close()
        stub.stop()
    _, _inp, _sp, pg, ig, _outp = {lp[0]: lp for lp in stub.loops}[2]
    assert pg == -0.4              # panel pgain 4 * PScale 0.1 * GainSign(-1)
    assert ig == -1000.0           # panel igain 10 * IScale 100 * GainSign(-1)


def test_hw_engage_contact_stays_deflection_and_positive_gain():
    """Contact path is unchanged: Deflection channel, POSITIVE gain. Guards against
    the AC change leaking into the contact servo."""
    stub, hw = _connect(defl=1.5)
    try:
        hw_engage(hw, setpoint=1.5, pgain=6.0, igain=10.0, mode="contact")
    finally:
        hw.close()
        stub.stop()
    _, inp, _sp, pg, ig, outp = {lp[0]: lp for lp in stub.loops}[2]
    assert inp == "Deflection" and outp == "Height"
    assert pg == 0.6 and ig == 1000.0            # POSITIVE (panel 6*0.1, 10*100)


def test_hw_engage_default_mode_is_contact_deflection_positive():
    """No mode argument -> contact behaviour, byte-identical to before the AC change."""
    stub, hw = _connect(defl=1.5)
    try:
        hw_engage(hw, setpoint=1.5, pgain=0.0, igain=25.0)   # no mode kwarg
    finally:
        hw.close()
        stub.stop()
    _, inp, _sp, _pg, ig, _outp = {lp[0]: lp for lp in stub.loops}[2]
    assert inp == "Deflection" and ig == 2500.0             # positive gain


def test_hw_set_zgain_ac_keeps_amplitude_and_negated_gain():
    """Live gain tune in AC must NOT flip the running loop back to Deflection/positive
    -- it re-arms on Amplitude with the negated gain (crash guard for a mid-run tune)."""
    stub, hw = _connect_ac(amp=0.65)
    try:
        hw_engage(hw, setpoint=0.65, pgain=0.0, igain=10.0, mode="ac", settle_s=0.0)
        hw_set_zgain(hw, setpoint=0.65, pgain=0.0, igain=500.0, mode="ac")
    finally:
        hw.close()
        stub.stop()
    loops2 = [lp for lp in stub.loops if lp[0] == 2]
    assert loops2[-1][1] == "Amplitude" and loops2[-1][5] == "Height"
    assert loops2[-1][4] == -50000.0            # panel igain 500 * IScale 100 * (-1)


# --- Route-level plumbing: the approach route derives the servo mode from the scan
# config (current_scan_is_ac) and passes it to hw_engage, so an AC scan engages on
# Amplitude + negated gain end-to-end. Contact/current stay Deflection + positive.


def _route_engage(mode_cfg, body):
    """Run the /api/approach coroutine against a fake AC-capable bridge with the
    given last-scan-config mode, and return the stub's loop-2 arm tuple."""
    import asyncio

    from afm_litho.service.app import hub
    from afm_litho.service.routers.scan import approach

    stub = FakeIgorBridge({"Amplitude": 0.65, "Deflection": 1.5, "height": 20.0},
                          done_delay=0.0)
    port = stub.serve_in_thread()
    prev_arc, prev_cfg = hub.host.hardware_arc, getattr(hub.host, "last_scan_config", None)
    prev_zloop = getattr(hub.host, "z_loop", None)   # approach() writes hub.z_loop; restore it
    prev_mode = getattr(hub.host, "imaging_mode", None)  # approach() may switch it; restore it
    prev_engaged = getattr(hub.host, "engaged", False)
    hub.host.hardware_arc = HardwareARC.connect("127.0.0.1", port)
    hub.host.last_scan_config = dict(mode_cfg)
    hub.host.scan_repeat["config"] = None            # so current_scan_is_ac() uses last_scan_config
    # B1 migration (docs/INSTRUMENT_MODEL_SPEC.md secs 2/10/13): the approach route now
    # runs `check_approach`, which needs a fresh withdrawn free-air baseline in the
    # engage mode, with the setpoint in-band (contact: setpoint-free_air in [0,1];
    # AC: setpoint/free_air in [0.3,0.95]). Seed one that keeps this call's setpoint
    # in-band so the plumbing under test (loop-2 arming) is still reached.
    from _im_fixtures_guard import seed_free_air
    prev_free_air = getattr(hub.host, "free_air", None)
    _m = str(body.get("mode") or mode_cfg.get("mode") or "contact").lower()
    _sp = float(body.get("setpoint", 0.0))
    _fa = (_sp / 0.6 if _sp else 1.0) if _m == "ac" else (_sp - 0.5)
    seed_free_air(hub, _fa, mode=_m)
    # Already in the engage mode, so the approach's pre-read mode reconcile does not
    # switch (a switch clears the seeded baseline, as set_mode does).
    hub.host.imaging_mode = "ac" if _m == "ac" else "contact"
    try:
        # settle_s 0 so the route does not block on the poll loop.
        asyncio.run(approach({**body, "settle_s": 0.0}))
        return {lp[0]: lp for lp in stub.loops}[2]
    finally:
        hub.host.hardware_arc.close()
        hub.host.hardware_arc = prev_arc
        hub.host.last_scan_config = prev_cfg
        # Restore the engaged Z-servo mode + the instrument imaging mode + the engaged
        # flag so an AC approach here does not leak into a later hardware litho write
        # (the litho contact-only guard reads hub.imaging_mode and hub.z_loop).
        hub.host.z_loop = prev_zloop
        hub.host.imaging_mode = prev_mode
        hub.host.engaged = prev_engaged
        hub.host.free_air = prev_free_air
        stub.stop()


def test_approach_route_ac_config_engages_on_amplitude_negated():
    """An AC scan config -> the approach route servos loop 2 on Amplitude with a
    NEGATED integral gain (the safety-critical end-to-end plumbing)."""
    _, inp, sp, _pg, ig, outp = _route_engage(
        {"mode": "ac"}, {"setpoint": 0.65, "pgain": 0.0, "igain": 10.0})
    assert inp == "Amplitude" and outp == "Height"
    assert sp == 0.65 and ig == -1000.0         # panel 10 * IScale 100 * GainSign(-1)


def test_approach_route_contact_config_stays_deflection_positive():
    """A contact scan config -> Deflection + POSITIVE gain, unchanged."""
    _, inp, _sp, _pg, ig, outp = _route_engage(
        {"mode": "contact"}, {"setpoint": 1.5, "pgain": 0.0, "igain": 10.0})
    assert inp == "Deflection" and outp == "Height"
    assert ig == 1000.0                         # POSITIVE


def test_approach_route_explicit_body_mode_overrides_config():
    """An explicit body `mode` wins over the scan config (contact config, AC body)."""
    _, inp, _sp, _pg, ig, _outp = _route_engage(
        {"mode": "contact"}, {"setpoint": 0.65, "pgain": 0.0, "igain": 10.0,
                              "mode": "ac"})
    assert inp == "Amplitude" and ig == -1000.0
