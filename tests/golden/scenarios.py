"""Golden scenarios: FLEX verbs only, every parameter stated.

Each takes a door with `call(method, params)` and `wait_state(*states)`. A scenario states
every parameter it relies on, so a changed service default shows up as a changed scenario
rather than as a silent golden drift.
"""

from __future__ import annotations

import time

CLIENT = "golden"


def _acquire(door) -> str:
    return door.call("acquireControl", {"client": CLIENT, "deadman_s": 120})["token"]


def _release(door, tok: str) -> None:
    door.call("releaseControl", {"token": tok})


def scan_chained_contact(door) -> None:
    """A small chained contact frame: 32 px, 2 lines, trace and retrace, at the current
    engagement (no engage)."""
    tok = _acquire(door)
    run = door.call("startScan", {
        "token": tok, "size_um": 1.0, "pixels": 32, "lines": 2, "line_rate_hz": 50.0,
        "x_offset_um": 0.0, "y_offset_um": 0.0, "angle_deg": 0.0, "direction": "up",
        "trace_retrace": True, "mode": "contact"})
    assert run["backend"] == "hardware" and run["config"]["chained"] is True, run
    door.wait_state("idle", "engaged", "parked")
    res = door.call("getScanResult", {"run_id": run["run_id"]})
    assert res["aborted"] is False, res
    _release(door, tok)


def litho_single(door) -> None:
    """One straight chiral wire (no wobble, constant 5 V), absolute setpoint, amp gain 25."""
    tok = _acquire(door)
    door.call("loadPattern", {"token": tok, "objects": [{
        "kind": "chiral", "name": "wire", "points_um": [[0.0, 0.0], [0.3, 0.0]],
        "lambda_um": 0.1, "y_amp_um": 0.0, "v0": 5.0, "v_k": 0.0, "phase_deg": 0.0,
        "hand": 1.0, "speed_um_s": 1.0}]})
    run = door.call("startWrite", {"token": tok, "objects": "all", "setpoint_reference": "absolute", "setpoint": 0.0,
                                   "amp_gain": 25.0})
    assert run["backend"] == "hardware" and run["objects"] == 1, run
    door.wait_state("idle", "engaged", "parked")
    res = door.call("getWriteResult", {"run_id": run["run_id"]})
    assert res["aborted"] is False, res
    _release(door, tok)


def litho_lateral_refused(door, inst) -> None:
    """`litho_single` with bank 2 refusing the UserIn2 + Lateral pair: the write goes on with
    UserIn2 alone (the .ipf fail-safe ladder's second rung)."""
    inst.lateral_refused()
    litho_single(door)


def safe_park(door) -> None:
    """The token-free safe-park: COM `AFM_SafePark()` (td_Stop once, bias 0, loop 2 re-armed),
    then the host confirms Height < 1 V (ABORTCONFIRM) and reads Igor's fault flag. No Z
    command of the host's own (slice AB fixed defect-1)."""
    res = door.call("safePark", {"reason": "golden safe-park"})
    assert res["transport"] == "com" and res["verified"] is True, res


def safe_park_com_down(door, inst) -> None:
    """The token-free safe-park with no COM server: `AFM_SafePark()` fails, so the park falls
    back to the wire ABORT (which the .ipf dispatches to AFM_SafePark), then the host's
    confirm."""
    inst.com_down()
    res = door.call("safePark", {"reason": "golden com down"})
    assert res["transport"] == "tcp" and res["verified"] is True, res


def safe_park_vendor_withdraw(door, inst) -> None:
    """A park whose abort does not bring Height under 1 V: the confirm fails, the park fires
    the vendor Withdraw button (`DoScanFunc("StopEngageButton")`, COM) once and confirms
    again."""
    inst.set_value("ZSensor", 3.0)
    inst.set_value("Height", 60.0)
    inst.set_value("Output.Z", 60.0)
    inst.park_leaves_tip_down()
    res = door.call("safePark", {"reason": "golden vendor withdraw"})
    assert res["transport"] == "com" and res["verified"] is True, res
    assert door.call("getLastPark")["method"] == "vendor_withdraw"


def _scan(door, tok, *, mode, lines=2, **extra) -> dict:
    cfg = {"token": tok, "size_um": 1.0, "pixels": 32, "lines": lines, "line_rate_hz": 50.0,
           "x_offset_um": 0.0, "y_offset_um": 0.0, "angle_deg": 0.0, "direction": "up",
           "trace_retrace": True, "mode": mode}
    cfg.update(extra)
    run = door.call("startScan", cfg)
    assert run["backend"] == "hardware" and run["config"]["chained"] is True, run
    return run


AC_CROSSPOINT = {"InFast": "ACDefl"}


def scan_chained_ac(door, inst) -> None:
    """A small chained AC frame at the current engagement (no engage), on an instrument whose
    software is already in AC: the chain arms Amplitude as its error channel."""
    inst.set_crosspoint(AC_CROSSPOINT)
    tok = _acquire(door)
    run = _scan(door, tok, mode="ac")
    door.wait_state("idle", "engaged", "parked")
    res = door.call("getScanResult", {"run_id": run["run_id"]})
    assert res["aborted"] is False, res
    _release(door, tok)


def scan_abort_mid_line(door, inst) -> None:
    """A chained contact frame of 4 lines, aborted (token-free `abort`) while line 1 is the last
    one the instrument has delivered."""
    tok = _acquire(door)
    inst.hold_scan_after_line(1)
    run = _scan(door, tok, mode="contact", lines=4)
    inst.wait_held()
    door.wait_scan_row(1)
    door.call("abort", {"scope": "scan"})
    door.wait_state("idle", "engaged", "parked")
    res = door.call("getScanResult", {"run_id": run["run_id"]})
    # SLICES F30(1), frozen as it behaves today: the stop landed between lines, so the run
    # is recorded as completed, not aborted.
    assert res["completed"] is True and res["aborted"] is False, res
    _release(door, tok)


SCAN_CFG = {"size_um": 1.0, "pixels": 32, "lines": 2, "line_rate_hz": 50.0, "x_offset_um": 0.0,
            "y_offset_um": 0.0, "angle_deg": 0.0, "direction": "up", "trace_retrace": True,
            "mode": "contact", "chained": True}


def continuous_two_frames(door) -> None:
    """Continuous imaging on the chained path, 2 frames: the loop arms a frame, then the
    next one, and its stop safe-parks."""
    tok = _acquire(door)
    door.call("setContinuous", {
        "token": tok, "enabled": True, "config": dict(SCAN_CFG), "max_frames": 2,
        "max_seconds": 120.0, "interval_s": 0.0, "withdraw_on_stop": False})
    door.wait_state("idle", "parked")
    _release(door, tok)


def engage_contact_tune_withdraw(door) -> None:
    """A contact engage, a live Z-gain tune on the engaged loop, then the withdraw."""
    tok = _acquire(door)
    door.call("startApproach", {"token": tok, "mode": "contact", "setpoint": 0.5,
                                "pgain": 0.0, "igain": 10.0, "settle_s": 0.2})
    assert door.wait_state("engaged", "parked") == "engaged"
    door.call("setZGain", {"token": tok, "mode": "contact", "setpoint": 0.5,
                           "pgain": 0.0, "igain": 20.0})
    res = door.call("withdraw", {})
    assert res["verified"] is True, res
    _release(door, tok)


def engage_ac_tune_withdraw(door, inst) -> None:
    """An AC engage on an instrument already in AC (free amplitude 1 V), a live Z-gain tune,
    then the withdraw."""
    inst.set_crosspoint(AC_CROSSPOINT)
    inst.set_value("Amplitude", 1.0)
    tok = _acquire(door)
    door.call("startApproach", {"token": tok, "mode": "ac", "setpoint": 0.5,
                                "pgain": 0.0, "igain": 10.0, "settle_s": 0.2})
    assert door.wait_state("engaged", "parked") == "engaged"
    door.call("setZGain", {"token": tok, "mode": "ac", "setpoint": 0.5,
                           "pgain": 0.0, "igain": 20.0})
    res = door.call("withdraw", {})
    assert res["verified"] is True, res
    _release(door, tok)


def mode_switch(door) -> None:
    """The imaging mode switched to AC and back to contact while withdrawn (SETMODE each)."""
    tok = _acquire(door)
    assert door.call("setMode", {"token": tok, "mode": "ac"})["mode"] == "ac"
    assert door.call("setMode", {"token": tok, "mode": "contact"})["mode"] == "contact"
    _release(door, tok)


def show_tip(http) -> None:
    """The live tip marker switched on and off (SHOWTIP ON/OFF; the cockpit has no FLEX verb)."""
    assert http.post("/api/tip/show", {"on": True})["on"] is True
    assert http.post("/api/tip/show", {"on": False})["on"] is False


def connect_reads(door) -> None:
    """Nothing but the connect: the STARTUP handshake and the GETGV calibration reads, then
    status reads. No actuation at all."""
    assert door.call("getStatus")["backend"] == "hardware"
    door.call("getInstrument")
    door.call("getMode")


def write_refused_in_ac(door, inst) -> None:
    """Lithography is contact-only: with the instrument in AC, startWrite reads the crosspoint
    (READXPT) and is refused before anything is armed."""
    inst.set_crosspoint(AC_CROSSPOINT)
    tok = _acquire(door)
    door.call("loadPattern", {"token": tok, "objects": [dict(WIRE)]})
    err = door.call_refused("startWrite", {"token": tok, "objects": "all", "setpoint_reference": "absolute", "setpoint": 0.0,
                                           "amp_gain": 25.0})
    assert "AC" in err["message"] or "ac" in err["message"], err
    _release(door, tok)


WIRE = {"kind": "chiral", "name": "wire", "points_um": [[0.0, 0.0], [0.3, 0.0]],
        "lambda_um": 0.1, "y_amp_um": 0.0, "v0": 5.0, "v_k": 0.0, "phase_deg": 0.0,
        "hand": 1.0, "speed_um_s": 1.0}
WIRE_B = {**WIRE, "name": "wire_b", "points_um": [[0.5, 0.2], [0.8, 0.2]]}
# 10 um at the chiral pen's 0.1 nm spacing floor (lambda 0.8 nm) is about 100,000 points:
# over the 86,976-point chunk size, so it is fired as a two-chunk chain.
WIRE_LONG = {**WIRE, "name": "wire_long", "points_um": [[0.0, 0.0], [10.0, 0.0]],
             "lambda_um": 0.0008}
# A filled rectangle expands into several strokes of one source object.
PAD = {"kind": "region", "name": "pad", "shape": "rect", "x_um": 0.0, "y_um": 0.0,
       "w_um": 0.2, "h_um": 0.1, "pitch_um": 0.05, "voltage": 5.0, "speed_um_s": 1.0,
       "fill": "directional"}


def _start_write(door, tok, objects) -> dict:
    door.call("loadPattern", {"token": tok, "objects": [dict(o) for o in objects]})
    run = door.call("startWrite", {"token": tok, "objects": "all", "setpoint_reference": "absolute", "setpoint": 0.0,
                                   "amp_gain": 25.0})
    assert run["backend"] == "hardware" and run["objects"] == len(objects), run
    return run


def litho_multi_object(door) -> None:
    """A filled pad (several strokes of one source: hover moves between them), then a wire
    (a new source: a full transit with the verified withdraw), then the end-of-write
    withdraw."""
    tok = _acquire(door)
    run = _start_write(door, tok, [PAD, WIRE])
    door.wait_state("idle", "engaged", "parked")
    res = door.call("getWriteResult", {"run_id": run["run_id"]})
    assert res["aborted"] is False, res
    _release(door, tok)


def litho_chain_two_chunks(door) -> None:
    """One stroke over the chunk size: LITHOCHAIN over two chunks, tip down across the seam."""
    tok = _acquire(door)
    run = _start_write(door, tok, [WIRE_LONG])
    door.wait_state("idle", "engaged", "parked")
    res = door.call("getWriteResult", {"run_id": run["run_id"]})
    assert res["aborted"] is False, res
    _release(door, tok)


def write_relative_setpoint(door, http) -> None:
    """The cockpit's write: a withdraw records the free-air baseline, then the write takes
    `setpoint_relative` and the service adds the baseline (FLEX states an absolute setpoint)."""
    tok = _acquire(door)
    door.call("loadPattern", {"token": tok, "objects": [dict(WIRE)]})
    _release(door, tok)
    http.post("/api/withdraw", {})
    res = http.post("/api/litho/write", {"objects": ["wire"], "setpoint_relative": 0.3,
                                         "amp_gain": 25.0})
    assert res.get("run_id"), res
    door.wait_state("idle", "engaged", "parked")


def abort_in_transit(door, inst) -> None:
    """Two wires; the abort lands while the second wire's transit move is in flight."""
    tok = _acquire(door)
    inst.hold_next("FIRE", 2)
    _start_write(door, tok, [WIRE, WIRE_B])
    inst.wait_held()
    door.call("abortWrite", {})
    door.wait_state("idle", "engaged", "parked")
    _release(door, tok)


def abort_mid_stroke(door, inst) -> None:
    """The abort lands while the first wire's stroke is playing."""
    tok = _acquire(door)
    inst.hold_next("ARMFIRE", 1)
    _start_write(door, tok, [WIRE])
    inst.wait_held()
    door.call("abortWrite", {})
    door.wait_state("idle", "engaged", "parked")
    _release(door, tok)


def withdraw_flex(door) -> None:
    """The withdraw through the FLEX door: a contact engage, then `withdraw` (H5.pre: one
    golden per withdraw door)."""
    tok = _acquire(door)
    door.call("startApproach", {"token": tok, "mode": "contact", "setpoint": 0.5,
                                "pgain": 0.0, "igain": 10.0, "settle_s": 0.2})
    assert door.wait_state("engaged", "parked") == "engaged"
    res = door.call("withdraw", {})
    assert res["verified"] is True, res
    _release(door, tok)


def withdraw_http(door, http) -> None:
    """The withdraw through the cockpit's HTTP door: a contact engage over FLEX, then
    `POST /api/withdraw`."""
    tok = _acquire(door)
    door.call("startApproach", {"token": tok, "mode": "contact", "setpoint": 0.5,
                                "pgain": 0.0, "igain": 10.0, "settle_s": 0.2})
    assert door.wait_state("engaged", "parked") == "engaged"
    _release(door, tok)
    http.post("/api/withdraw", {})
    door.wait_state("idle", "parked")


def cockpit_abort_mid_stroke(door, inst, http) -> None:
    """The cockpit's Abort button (`POST /api/safe_park`, reason "operator abort") while the
    first wire's stroke is playing."""
    tok = _acquire(door)
    inst.hold_next("ARMFIRE", 1)
    _start_write(door, tok, [WIRE])
    inst.wait_held()
    http.post("/api/safe_park", {"reason": "operator abort"})
    door.wait_state("idle", "engaged", "parked")
    _release(door, tok)


def cockpit_abort_in_transit(door, inst, http) -> None:
    """The cockpit's Abort button while the second wire's transit move is in flight."""
    tok = _acquire(door)
    inst.hold_next("FIRE", 2)
    _start_write(door, tok, [WIRE, WIRE_B])
    inst.wait_held()
    http.post("/api/safe_park", {"reason": "operator abort"})
    door.wait_state("idle", "engaged", "parked")
    _release(door, tok)


def litho_abort_http_mid_stroke(door, inst, http) -> None:
    """The write abort over HTTP (`POST /api/litho/abort`, the pair of `abortWrite`) while the
    first wire's stroke is playing."""
    tok = _acquire(door)
    inst.hold_next("ARMFIRE", 1)
    _start_write(door, tok, [WIRE])
    inst.wait_held()
    http.post("/api/litho/abort", {})
    door.wait_state("idle", "engaged", "parked")
    _release(door, tok)


def abort_at_chain_seam(door, inst) -> None:
    """The abort lands at the seam of a two-chunk chain: chunk 0 delivered, chunk 1 not."""
    tok = _acquire(door)
    inst.hold_chain_after_chunk(0)
    _start_write(door, tok, [WIRE_LONG])
    inst.wait_held()
    door.call("abortWrite", {})
    door.wait_state("idle", "engaged", "parked")
    _release(door, tok)


def transit_retry(door, inst) -> None:
    """The first wire's transit lands short once: two agreeing reads, then the retry arrives."""
    tok = _acquire(door)
    inst.land_short(1)
    run = _start_write(door, tok, [WIRE])
    door.wait_state("idle", "engaged", "parked")
    res = door.call("getWriteResult", {"run_id": run["run_id"]})
    assert res["aborted"] is False, res
    _release(door, tok)


def transit_failure_withdraw(door, inst) -> None:
    """The transit lands short twice: it raises after a withdraw, and the write fails."""
    tok = _acquire(door)
    inst.land_short(2)
    run = _start_write(door, tok, [WIRE])
    door.wait_state("idle", "engaged", "parked")
    door.call("getWriteResult", {"run_id": run["run_id"]})
    _release(door, tok)


def multi_pass(door) -> None:
    """The F9 shape: a wire, then a chained scan (its own transit, with the withdraw), then a
    second wire (a transit to a new source). Frozen as today's behaviour."""
    tok = _acquire(door)
    _start_write(door, tok, [WIRE])
    door.wait_state("idle", "engaged", "parked")
    run = _scan(door, tok, mode="contact")
    door.wait_state("idle", "engaged", "parked")
    door.call("getScanResult", {"run_id": run["run_id"]})
    _start_write(door, tok, [WIRE_B])
    door.wait_state("idle", "engaged", "parked")
    _release(door, tok)


def scan_stop_deferred_withdraw(door, inst, http) -> None:
    """The cockpit's Stop during a scan: the pass owns the link, so the retract is deferred
    until the pass halts (the next line arrives, the chain driver sees the stop)."""
    tok = _acquire(door)
    inst.hold_scan_after_line(1)
    _scan(door, tok, mode="contact", lines=4)
    inst.wait_held()
    door.wait_scan_row(1)
    http.post("/api/scan/abort", {})
    inst.release_hold()
    door.wait_state("idle", "engaged", "parked")
    _release(door, tok)


def interlock_trip(door, inst) -> None:
    """An engage with the always-on poll running; DetectorSum then collapses. The interlock
    trips after its 5 s dwell and the fail-safe thread parks the tip."""
    tok = _acquire(door)
    door.call("startApproach", {"token": tok, "mode": "contact", "setpoint": 0.5,
                                "pgain": 0.0, "igain": 10.0, "settle_s": 0.2})
    assert door.wait_state("engaged", "parked") == "engaged"
    inst.set_value("DetectorSum", 0.1)
    door.wait_state("parked", timeout=60.0)
    _release(door, tok)


def deadman_during_write(door, inst) -> None:
    """The script goes silent while a stroke plays: the dead-man (5 s) aborts and safe-parks."""
    tok = door.call("acquireControl", {"client": CLIENT, "deadman_s": 5.0,
                                       "withdraw_on_deadman": False})["token"]
    inst.hold_next("ARMFIRE", 1)
    _start_write(door, tok, [WIRE])
    while not inst.is_held():              # the start itself outlasts the 5 s dead-man
        door.call("heartbeat", {"token": tok})
        time.sleep(0.5)
    inst.wait_held()
    door.wait_state("parked", timeout=60.0)
    _release(door, tok)


def deadman_engaged_withdraw(door) -> None:
    """The script goes silent while engaged and asked for `withdraw_on_deadman`: the dead-man
    withdraws (no safe-park)."""
    tok = door.call("acquireControl", {"client": CLIENT, "deadman_s": 5.0,
                                       "withdraw_on_deadman": True})["token"]
    door.call("startApproach", {"token": tok, "mode": "contact", "setpoint": 0.5,
                                "pgain": 0.0, "igain": 10.0, "settle_s": 0.2})
    assert door.wait_state("engaged", "parked") == "engaged"
    door.wait_state("idle", "parked", timeout=60.0)
    _release(door, tok)


def engaged_idle_cap(door) -> None:
    """Engaged with nothing running, a long dead-man: the engaged-idle cap (set to its 1 s
    floor for the recording) withdraws."""
    tok = door.call("acquireControl", {"client": CLIENT, "deadman_s": 300.0,
                                       "withdraw_on_deadman": False})["token"]
    door.call("startApproach", {"token": tok, "mode": "contact", "setpoint": 0.5,
                                "pgain": 0.0, "igain": 10.0, "settle_s": 0.2})
    assert door.wait_state("engaged", "parked") == "engaged"
    door.wait_state("idle", "parked", timeout=60.0)
    _release(door, tok)


# Defects a golden freezes as today's behaviour. Stamped into the golden file, so an intended
# fix shows as a changed golden plus a removed entry, never as a silent re-freeze.
F30_1 = {
    "id": "f30-1",
    "what": ("an abort between two lines of a chained scan is recorded as completed: the "
             "run reports completed true, aborted false (the commands are right)"),
    "where": "docs/migration/SLICES.md F30(1)",
}
F30_3 = {
    "id": "f30-3",
    "what": "a scan Stop parks twice: two full @SAFEPARK and confirm sequences",
    "where": "docs/migration/SLICES.md F30(3)",
}
F30_4 = {
    "id": "f30-4",
    "what": ("a transit failure ends the write in a plain withdraw (Igor's abort function "
             "over the socket, no fallback), not a safe-park"),
    "where": "docs/migration/SLICES.md F30(4)",
}
_F30 = {"scan_abort_mid_line": F30_1, "scan_stop_deferred_withdraw": F30_3,
        "transit_failure_withdraw": F30_4}
# defect-1 (the park's overlapping loop-2 commands) is fixed by slice AB: the park is Igor's
# one abort function and the host sends no Z command of its own.
KNOWN_DEFECTS: dict[str, list[dict]] = {}
for _n, _d in _F30.items():
    for _k in (_n, f"{_n}@mfp3d"):
        KNOWN_DEFECTS[_k] = [*KNOWN_DEFECTS.get(_k, []), _d]


# Per-scenario recording options: `telem_hz` (the always-on tip poll), `env` (variables set while
# recording), `mfp` (False, or a reason string, when the scenario is not also recorded as an
# MFP-3D at the wire level; every scenario is today), `clock` ("testing-time-sensitive" for a
# scenario whose outcome depends on real elapsed time, an abort or dead-man landing while
# another real thread is mid-wait; the default is the testing clock, which skips waits).
OPTIONS: dict[str, dict] = {
    "scan_abort_mid_line": {"clock": "testing-time-sensitive"},
    "scan_stop_deferred_withdraw": {"clock": "testing-time-sensitive"},
    "deadman_during_write": {"clock": "testing-time-sensitive"},
    "deadman_engaged_withdraw": {"clock": "testing-time-sensitive"},
    "interlock_trip": {"telem_hz": 50.0},
    "engaged_idle_cap": {"env": {"AFM_LITHO_ENGAGED_IDLE_MAX_S": "1"},
                         "clock": "testing-time-sensitive"},
}


SCENARIOS = {
    "scan_chained_contact": scan_chained_contact,
    "litho_single": litho_single,
    "litho_lateral_refused": litho_lateral_refused,
    "safe_park": safe_park,
    "safe_park_com_down": safe_park_com_down,
    "safe_park_vendor_withdraw": safe_park_vendor_withdraw,
    "scan_chained_ac": scan_chained_ac,
    "scan_abort_mid_line": scan_abort_mid_line,
    "continuous_two_frames": continuous_two_frames,
    "engage_contact_tune_withdraw": engage_contact_tune_withdraw,
    "engage_ac_tune_withdraw": engage_ac_tune_withdraw,
    "mode_switch": mode_switch,
    "show_tip": show_tip,
    "connect_reads": connect_reads,
    "write_refused_in_ac": write_refused_in_ac,
    "litho_multi_object": litho_multi_object,
    "litho_chain_two_chunks": litho_chain_two_chunks,
    "write_relative_setpoint": write_relative_setpoint,
    "abort_in_transit": abort_in_transit,
    "abort_mid_stroke": abort_mid_stroke,
    "abort_at_chain_seam": abort_at_chain_seam,
    "withdraw_flex": withdraw_flex,
    "withdraw_http": withdraw_http,
    "cockpit_abort_mid_stroke": cockpit_abort_mid_stroke,
    "cockpit_abort_in_transit": cockpit_abort_in_transit,
    "litho_abort_http_mid_stroke": litho_abort_http_mid_stroke,
    "transit_retry": transit_retry,
    "transit_failure_withdraw": transit_failure_withdraw,
    "multi_pass": multi_pass,
    "scan_stop_deferred_withdraw": scan_stop_deferred_withdraw,
    "interlock_trip": interlock_trip,
    "deadman_during_write": deadman_during_write,
    "deadman_engaged_withdraw": deadman_engaged_withdraw,
    "engaged_idle_cap": engaged_idle_cap,
}
