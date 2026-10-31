"""The rows of the safe-state matrix: each way the program is told to make the instrument safe.

A row drives one door (FLEX or HTTP) on the fake-Igor safety simulation with the tip down, and returns what
the program *said* about it: `Said(safe=True)` when the reply or state flag claims the
instrument is safe. The harness then judges the instrument itself from GETVALUE reads.

Every row first checks that the instrument really was unsafe in the way the row is about
(tip down; bias on or XY moving where the row has them), so a passing cell cannot be vacuous.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from safe_state_harness import Safe, SafeRig, engage, wait_for

from afm_litho.service import control as ctl
from afm_litho.service.host import deadman


@dataclass(frozen=True)
class Said:
    """What the program reported: `safe` is whether that report claims a safe instrument."""
    safe: bool
    text: str


@dataclass(frozen=True)
class Row:
    name: str
    drive: object                    # (rig, token) -> Said
    rig_kw: dict
    # A write is ~10 s of the program's own waits, and on the MFP-3D it never sets the tip
    # down (DECISIONS, third round), so the write rows run on the Cypher only.
    cypher_only: bool = False
    mfp_too: bool = False            # run on the MFP-3D as well, where that is cheap
    fault_injection: bool = False    # the retract is made to fail; only the report is judged
    after_reply: object = None       # (rig) -> None: lets a deferred step finish after the reply read


WIRE = {"kind": "chiral", "name": "wire", "points_um": [[0.0, 0.0], [1.0, 0.0]],
        "lambda_um": 0.1, "y_amp_um": 0.0, "v0": 5.0, "v_k": 0.0, "phase_deg": 0.0,
        "hand": 1.0, "speed_um_s": 1.0}


def _over_cap_svg() -> str:
    """A sawtooth whose toolpath is over the out-wave cap, so the write is a chain of chunks."""
    legs = " ".join(f"L {i * 200} {20000 if i % 2 else 0}" for i in range(1, 101))
    return ('<svg xmlns="http://www.w3.org/2000/svg">'
            '<image id="topo_bgd" x="0" y="0" width="20000" height="20000"/>'
            '<path d="M 0 0 ' + legs + '"/></svg>')


SLOW_SCAN = {"size_um": 1.0, "pixels": 32, "lines": 4, "line_rate_hz": 0.5, "x_offset_um": 0.0,
             "y_offset_um": 0.0, "angle_deg": 0.0, "direction": "up", "trace_retrace": True,
             "mode": "contact"}
# Where the program's pre-move transit ends, in XSensor/YSensor volts (read off the wire).
SCAN_START_V = (-0.697024107, -0.00102272723)
WIRE_START_V = (-0.629999995, 0.0700000003)
INTO_LINE_S = 0.5                # after the scan chain arms, to be well into its first line
INTO_STROKE_S = 0.1              # a 1 um wire at 1 um/s is a one-second stroke
INTO_TRANSIT_S = 0.15            # the transit's wave plays for about a second
SEAM_WAIT_S = 30.0
WRITE_DONE_WAIT_S = 40.0
COLLAPSED_SUM_V = 0.1            # below the interlock's 0.5 V floor
TELEM_HZ = 50.0                  # the tip poll that feeds the interlocks
INTERLOCK_PARK_WAIT_S = 40.0     # the interlock's real 5 s dwell, then the park
DEFERRED_WITHDRAW_WAIT_S = 30.0   # a bound for a hang; the retract reports in about a second
PAST_EXPIRY_S = 1.0              # the injected clock reads this far past the deadline


def _park_said(res: dict) -> Said:
    park = res.get("safe_park", res)
    ok = bool(park.get("verified")) and not park.get("critical")
    return Said(ok, f"verified={park.get('verified')} critical={park.get('critical')}")


def start_slow_scan(rig: SafeRig, tok: str) -> None:
    rig.door.call("startScan", {"token": tok, **SLOW_SCAN})
    assert rig.wait_for_actuation("SCANCHAIN"), "the scan never started"
    time.sleep(INTO_LINE_S)
    down, moving = rig.tip_is_down(), rig.xy_is_moving()
    assert down and moving, f"not mid-line: tip down {down}, xy moving {moving}"


def start_wire(rig: SafeRig, tok: str, pattern: dict | None = None) -> None:
    rig.door.call("loadPattern", {"token": tok, "objects": [pattern or WIRE]})
    rig.door.call("startWrite", {"token": tok, "objects": "all", "setpoint_reference": "absolute", "setpoint": 0.0,
                                 "amp_gain": 25.0})


def _post(rig: SafeRig, path: str, body: dict | None = None) -> dict:
    return rig.client.post(path, json=body or {}).json()


def _on_loop(rig: SafeRig, coro):
    return asyncio.run_coroutine_threadsafe(coro, rig.hub._loop).result(60)


# ---- aborts: a pass is running ----

def abort_scan_line(rig, tok):
    start_slow_scan(rig, tok)
    return _park_said(rig.door.call("abort", {"scope": "scan"}))


def abort_litho_stroke(rig, tok):
    start_wire(rig, tok)
    assert rig.wait_for_actuation("ARMFIRE", "wire"), "the stroke never started"
    time.sleep(INTO_STROKE_S)
    assert rig.tip_is_down() and rig.bias_is_on(), "not mid-stroke"
    return _park_said(rig.door.call("abortWrite", {}))


def abort_chain_seam(rig, tok):
    rig.door.call("loadPattern", {"token": tok, "source": _over_cap_svg()})
    rig.stub.hold_at_seam()
    rig.door.call("startWrite", {"token": tok, "objects": "all", "setpoint_reference": "absolute", "setpoint": 0.0,
                                 "amp_gain": 25.0})
    assert rig.stub.seam_reached.wait(SEAM_WAIT_S), "the chain never reached its seam"
    assert rig.tip_is_down() and rig.bias_is_on(), "not at the seam with the tip down"
    return _park_said(rig.door.call("abort", {"scope": "write"}))


def abort_transit(rig, tok):
    start_wire(rig, tok)
    assert rig.wait_for_actuation("FIRE", "XYTransit"), "the transit never started"
    time.sleep(INTO_TRANSIT_S)
    assert rig.xy_is_moving(), "not mid-transit"
    reply = _post(rig, "/api/litho/abort")
    return Said(bool(reply.get("ok")) and rig.door.call("getState") != "fault",
                f"reply ok={reply.get('ok')} state={rig.door.call('getState')}")


# ---- stop, withdraw, park ----

def stop_scan(rig, tok):
    start_slow_scan(rig, tok)
    rig.parked_before_stop = rig.door.call("getLastPark").get("ts")
    reply = _post(rig, "/api/scan/abort")
    return Said(bool(reply.get("ok")) and not rig.hub.host.engaged,
                f"reply ok={reply.get('ok')} engaged={rig.hub.host.engaged}")


def after_scan_stop(rig):
    """The held scan sends no further line, so the chain driver never tests the stop and the
    deferred retract never runs. One more line is what a real scan delivers; the retract then
    shows as a new park through the FLEX door."""
    rig.stub.let_one_scan_line_through()
    assert wait_for(lambda: rig.door.call("getLastPark").get("ts") != rig.parked_before_stop,
                    DEFERRED_WITHDRAW_WAIT_S), "the deferred retract never reported a park"


def withdraw_http(rig, tok):
    reply = _post(rig, "/api/withdraw")
    return Said(not rig.hub.host.engaged and not reply.get("error"),
                f"engaged={rig.hub.host.engaged} snapshot={reply}")


def withdraw_http_stuck_z(rig, tok):
    rig.safety_sim.z_stuck = True                       # the retract does not take
    return withdraw_http(rig, tok)


def withdraw_flex_stuck_z(rig, tok):
    rig.safety_sim.z_stuck = True
    return withdraw_flex(rig, tok)


def withdraw_flex(rig, tok):
    reply = rig.door.call("withdraw", {})
    return Said(bool(reply.get("verified")), f"verified={reply.get('verified')} "
                                             f"z_v={reply.get('z_v')}")


def end_of_write_withdraw(rig, tok):
    start_wire(rig, tok)
    assert rig.wait_for_actuation("ARMFIRE", "wire"), "the stroke never started"
    assert wait_for(lambda: rig.door.call("getState") != "writing", WRITE_DONE_WAIT_S), "the write never ended"
    rig.wait_quiet()
    return Said(not rig.hub.host.engaged, f"engaged={rig.hub.host.engaged}")


def safe_park_button(rig, tok):
    return _park_said(_post(rig, "/api/safe_park"))


# ---- unattended: nobody pressed anything ----

def interlock_trip(rig, tok):
    start_slow_scan(rig, tok)
    parks_before = rig.hub.host._failsafe.parks
    rig.stub.values["DetectorSum"] = COLLAPSED_SUM_V             # the detector signal collapses
    assert wait_for(lambda: rig.hub.host._failsafe.parks > parks_before, INTERLOCK_PARK_WAIT_S), "no interlock park"
    rig.wait_quiet()
    park = rig.hub.host.last_park
    return Said(bool(park and park.verified), f"last_park={park}")


def deadman_lapse(rig, tok):
    start_slow_scan(rig, tok)
    expired = rig.hub.control.expires_at + PAST_EXPIRY_S
    state = rig.door.call("getState")
    actions = _on_loop(rig, ctl.control_tick(rig.hub, now=expired))
    rig.wait_quiet()
    park = (actions[0].get("safe_park") if actions else None) or {}
    return Said(bool(park.get("verified")) and not park.get("critical"),
                f"state={state} actions={[a['action'] for a in actions]} "
                f"park verified={park.get('verified')}")


def engaged_idle_cap(rig, tok):
    start = time.time()
    _on_loop(rig, ctl.control_tick(rig.hub, now=start))              # starts the idle clock
    limit = deadman.engaged_idle_max_s()
    actions = _on_loop(rig, ctl.control_tick(rig.hub, now=start + limit + PAST_EXPIRY_S))
    rig.wait_quiet()
    return Said(not rig.hub.host.engaged and bool(actions),
                f"actions={[a['action'] for a in actions]} engaged={rig.hub.host.engaged}")


_SCAN = {"hold_scan": True, "tip_at_volts": SCAN_START_V}
_WIRE = {"tip_at_volts": WIRE_START_V}

ROWS = [
    Row("abort_scan_line", abort_scan_line, _SCAN, mfp_too=True),
    Row("abort_litho_stroke", abort_litho_stroke, {**_WIRE, "fill_scale": 1.0},
        cypher_only=True),
    Row("abort_chain_seam", abort_chain_seam, _WIRE, cypher_only=True),
    Row("abort_transit", abort_transit, {"fill_scale": 1.0}),
    Row("stop_scan", stop_scan, _SCAN, after_reply=after_scan_stop),
    Row("withdraw_http", withdraw_http, {}, mfp_too=True),
    Row("withdraw_flex", withdraw_flex, {}, mfp_too=True),
    Row("end_of_write_withdraw", end_of_write_withdraw, {**_WIRE, "fill_scale": 1.0},
        cypher_only=True),
    Row("safe_park_button", safe_park_button, {}, mfp_too=True),
    Row("interlock_trip", interlock_trip, {**_SCAN, "telem_hz": TELEM_HZ}),
    Row("deadman_lapse", deadman_lapse, _SCAN),
    Row("engaged_idle_cap", engaged_idle_cap, {}, mfp_too=True),
    # The retract is made not to take (the Z drive is stuck): only what the program says counts.
    Row("withdraw_http_stuck_z", withdraw_http_stuck_z, {}, mfp_too=True, fault_injection=True),
    Row("withdraw_flex_stuck_z", withdraw_flex_stuck_z, {}, mfp_too=True, fault_injection=True),
]


@dataclass(frozen=True)
class Outcome:
    """What one row did: what the program said, and what the instrument read the moment the
    door answered and again once everything had wound down."""
    said: Said
    at_reply: Safe
    settled: Safe


def run_row(hub, client, row: Row, microscope_id: int,
            height_ramp_after_stoploop: bool = False) -> Outcome:
    """One row on a fresh rig: engage, drive the door, judge from reads at the reply and
    again after the program has gone quiet."""
    with SafeRig(hub, row.name, microscope_id=microscope_id, client=client,
                 height_ramp_after_stoploop=height_ramp_after_stoploop, **row.rig_kw) as rig:
        tok = engage(rig)
        said = row.drive(rig, tok)
        at_reply = rig.judge()
        if row.after_reply:
            row.after_reply(rig)
        rig.wait_quiet()
        return Outcome(said, at_reply, rig.judge())
