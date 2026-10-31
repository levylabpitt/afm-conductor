"""An abort pressed in Igor (the AFM Abort panel's AFM_SafePark, not sent by the service) stops
the whole run, in every phase (rig finding, Cypher, 2026-10-28: pressed during the move to an
object's first point, the service retried the move, re-engaged and wrote the object).

Igor's button is played on the golden rig's stub as Igor runs it: AFM_SafePark on the
instrument side, its ABORTED / SCANDONE -1 on the socket, nothing on the wire from the
service. The service's own park that follows is the cockpit Abort's, recorded as
"Igor-side abort".
"""

from __future__ import annotations

import time

import pytest
from golden.harness import HERE, GoldenRig
from golden.scenarios import WIRE, WIRE_B, _acquire, _release, _scan, _start_write

from afm_litho import clock
from afm_litho.goldens import golden_format as G
from afm_litho.hardware import bridge_protocol as P
from afm_litho.hardware.bridge_router import BridgeRouter
from afm_litho.hardware.lithography_run import LithographyRun
from afm_litho.service.host.safe_park import IGOR_SIDE_ABORT


@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


def _wait(cond, timeout: float = 30.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        time.sleep(0.01)
    raise AssertionError("condition never held")


def press_igor_abort(rig) -> None:
    """Igor's AFM Abort panel button: AFM_SafePark runs in Igor (v2: ABORT's function)."""
    rig.stub.run_unrecorded("ABORT")


def _wire(rig) -> list[str]:
    return G.wire_level(rig.tap)["actuation"]


def _engages(rig) -> list[str]:
    """Loop-2 arms at a contact setpoint (the -10 V arm is the lift / retract)."""
    return [ln for ln in _wire(rig) if ln.startswith("SERVOLOOP 2 ")
            and ln.split(" ")[3] != "-10"]


def _count(rig, verb: str) -> int:
    return sum(1 for v in rig.stub.transcript if v == verb)


def _parked_by_igor(hub) -> None:
    _wait(lambda: (hub.host._safe_park or {}).get("last_reason") == IGOR_SIDE_ABORT)
    assert hub.host._safe_park["verified"] is True
    assert hub.fault is None


def _write_aborted(rig, run) -> None:
    rig.door.wait_state("idle", "engaged", "parked")
    res = rig.door.call("getWriteResult", {"run_id": run["run_id"]})
    assert res["aborted"] is True, res


def test_igor_abort_during_the_transit_ends_the_run_with_no_retry_and_no_engage(hub, client):
    with clock.using(clock.TESTING), GoldenRig(hub, "igor_abort_transit") as rig:
        door, inst = rig.door, rig.inst
        tok = _acquire(door)
        inst.land_short(1)                  # the abort cuts the move short of its target
        inst.hold_next("FIRE", 1)
        run = _start_write(door, tok, [WIRE, WIRE_B])
        inst.wait_held()
        press_igor_abort(rig)
        _write_aborted(rig, run)
        _parked_by_igor(hub)
        rig.quiesce()
        assert _count(rig, "FIRE") == 1, "the transit was retried"
        assert _engages(rig) == [], "the tip was set down after the abort"
        assert _count(rig, "ARMFIRE") == 0 and _count(rig, "LITHOCHAIN") == 0
        _release(door, tok)


def test_igor_abort_during_the_engage_writes_nothing(hub, client, monkeypatch):
    real_arm = LithographyRun._arm_z

    def arm_then_press(self, setpoint):
        real_arm(self, setpoint)
        if setpoint != self.retract and not pressed:
            pressed.append(setpoint)
            press_igor_abort(rig)
            _wait(hub.host._hw_litho_abort.is_set)

    pressed: list = []
    monkeypatch.setattr(LithographyRun, "_arm_z", arm_then_press)
    with clock.using(clock.TESTING), GoldenRig(hub, "igor_abort_engage") as rig:
        door = rig.door
        tok = _acquire(door)
        run = _start_write(door, tok, [WIRE, WIRE_B])
        _write_aborted(rig, run)
        _parked_by_igor(hub)
        rig.quiesce()
        assert pressed and len(_engages(rig)) == 1, "the tip was set down again"
        assert _count(rig, "ARMFIRE") == 0 and _count(rig, "FIRE") == 1
        _release(door, tok)


def test_igor_abort_between_objects_does_not_start_the_next(hub, client, monkeypatch):
    real_write = LithographyRun.write_object

    def write_then_press(self, *a, **kw):
        out = real_write(self, *a, **kw)
        if not pressed:
            pressed.append(True)
            press_igor_abort(rig)
            _wait(hub.host._hw_litho_abort.is_set)
        return out

    pressed: list = []
    monkeypatch.setattr(LithographyRun, "write_object", write_then_press)
    with clock.using(clock.TESTING), GoldenRig(hub, "igor_abort_between") as rig:
        door = rig.door
        tok = _acquire(door)
        run = _start_write(door, tok, [WIRE, WIRE_B])
        _write_aborted(rig, run)
        _parked_by_igor(hub)
        rig.quiesce()
        assert _count(rig, "ARMFIRE") == 1, "the second object was written"
        assert _count(rig, "FIRE") == 1, "the second object's move started"
        assert len(_engages(rig)) == 1
        _release(door, tok)


def test_igor_abort_mid_stroke_stops_the_stroke(hub, client):
    with clock.using(clock.TESTING), GoldenRig(hub, "igor_abort_stroke") as rig:
        door, inst = rig.door, rig.inst
        tok = _acquire(door)
        inst.hold_next("ARMFIRE", 1)
        run = _start_write(door, tok, [WIRE, WIRE_B])
        inst.wait_held()
        press_igor_abort(rig)
        _write_aborted(rig, run)
        _parked_by_igor(hub)
        rig.quiesce()
        assert _count(rig, "ARMFIRE") == 1 and _count(rig, "FIRE") == 1
        assert len(_engages(rig)) == 1
        _release(door, tok)


def test_igor_abort_mid_frame_stops_the_scan_like_the_cockpit_abort(hub, client):
    """The same wire as the cockpit abort of scan_abort_mid_line: the park, its confirm, and
    no end-of-scan withdraw."""
    with clock.using(clock.TESTING_TIME_SENSITIVE), GoldenRig(hub, "igor_abort_scan") as rig:
        door, inst = rig.door, rig.inst
        tok = _acquire(door)
        inst.hold_scan_after_line(1)
        run = _scan(door, tok, mode="contact", lines=4)
        inst.wait_held()
        door.wait_scan_row(1)
        press_igor_abort(rig)
        door.wait_state("idle", "engaged", "parked")
        _parked_by_igor(hub)
        rig.quiesce()
        res = door.call("getScanResult", {"run_id": run["run_id"]})
        assert res["aborted"] is True, res
        frozen = G.load(HERE / "scan_abort_mid_line.golden.json")
        assert _wire(rig) == frozen["wire"]["actuation"]
        assert hub.host.scan_repeat.get("enabled") is False
        _release(door, tok)


# ---- the router hands every abort report to the hook, before the waiter -------------------

class _NoLink:
    def read_line(self, timeout=None):
        return None


def _router():
    seen: list[str] = []
    r = BridgeRouter(_NoLink())
    r.on_igor_abort = seen.append
    return r, seen


def test_router_reports_aborted_and_scandone_minus_one_and_nothing_else():
    r, seen = _router()
    r._route(P.msg_aborted("", "XYTransit"))           # stale uid: no waiter
    r._route(P.msg_scandone("7", -1))
    r._route(P.msg_scandone("8", 4))
    r._route(P.msg_done("9", "wire"))
    assert seen == ["ABORTED  XYTransit", "SCANDONE 7 -1"]


def test_router_sets_the_flag_before_the_pass_wakes():
    r, seen = _router()
    w = r.open_waiter()
    waiter_empty_at_hook: list[bool] = []
    r.on_igor_abort = lambda frame: waiter_empty_at_hook.append(w._q.empty())
    r._route(P.msg_aborted(w.uid, "XYTransit"))
    assert waiter_empty_at_hook == [True]
    assert w.get(0.1) == (P.ABORTED, "XYTransit")


def test_no_run_in_flight_parks_nothing_but_marks_the_stop(hub, client, monkeypatch):
    import importlib
    called: list = []
    safe_park = importlib.import_module("afm_litho.service.host.safe_park")
    monkeypatch.setattr(safe_park, "park_on_igor_abort", lambda *a: called.append(a))
    monkeypatch.setattr(type(hub), "run_in_flight", lambda self: False)
    before = hub.host._run_abort_seq
    hub.note_igor_abort("ABORTED 3 XYTransit")
    assert called == []
    assert hub.host._run_abort_seq != before          # a start mid-way through its reads refuses


def test_igor_abort_during_the_scan_transit_arms_nothing(hub, client):
    with clock.using(clock.TESTING_TIME_SENSITIVE), GoldenRig(hub, "igor_abort_scan_move") as rig:
        door, inst = rig.door, rig.inst
        tok = _acquire(door)
        inst.land_short(1)
        inst.hold_next("FIRE", 1)
        _scan(door, tok, mode="contact", lines=4, engage=True, setpoint=0.5)
        inst.wait_held()
        press_igor_abort(rig)
        door.wait_state("idle", "engaged", "parked")
        _parked_by_igor(hub)
        rig.quiesce()
        assert _count(rig, "FIRE") == 1, "the transit was retried"
        assert _count(rig, "SCANCHAIN") == 0 and _engages(rig) == []
        _release(door, tok)


def test_igor_abort_during_the_transit_withdraw_moves_nothing(hub, client, monkeypatch):
    """The first object's move withdraws first; an abort landing in that withdraw stops the
    move before any XY loop arm or FIRE."""
    from afm_litho.hardware import hw_approach
    real_withdraw = hw_approach.hw_withdraw
    pressed: list = []

    def withdraw_then_press(*a, **kw):
        out = real_withdraw(*a, **kw)
        if not pressed:
            pressed.append(True)
            press_igor_abort(rig)
            _wait(hub.host._hw_litho_abort.is_set)
        return out

    monkeypatch.setattr(hw_approach, "hw_withdraw", withdraw_then_press)
    with clock.using(clock.TESTING), GoldenRig(hub, "igor_abort_withdraw") as rig:
        door = rig.door
        tok = _acquire(door)
        run = _start_write(door, tok, [WIRE, WIRE_B])
        _write_aborted(rig, run)
        _parked_by_igor(hub)
        rig.quiesce()
        assert pressed
        assert _count(rig, "FIRE") == 0 and _count(rig, "OUTPAIR") == 0, "XY moved"
        assert not any(ln.startswith(("SERVOLOOP 0 ", "SERVOLOOP 1 ")) for ln in _wire(rig))
        assert _engages(rig) == []
        _release(door, tok)


def _press_and_wait(rig, hub) -> None:
    before = hub.host._run_abort_seq
    press_igor_abort(rig)
    _wait(lambda: hub.host._run_abort_seq != before)


def test_igor_abort_during_the_write_start_reads_refuses_the_write(hub, client, monkeypatch):
    """No worker runs yet during the start's reads; the start refuses, and the operator's
    next start is allowed (no latch)."""
    from afm_litho.service.host import litho_cmds
    real_ac = litho_cmds.ac_mode_active
    pressed: list = []

    def read_then_press(h):
        out = real_ac(h)
        if not pressed:
            pressed.append(True)
            _press_and_wait(rig, hub)
        return out

    monkeypatch.setattr(litho_cmds, "ac_mode_active", read_then_press)
    with clock.using(clock.TESTING), GoldenRig(hub, "igor_abort_write_start") as rig:
        door = rig.door
        tok = _acquire(door)
        door.call("loadPattern", {"token": tok, "objects": [dict(WIRE)]})
        err = door.call_refused("startWrite", {"token": tok, "objects": "all",
                                               "setpoint_reference": "absolute", "setpoint": 0.0, "amp_gain": 25.0})
        assert (err.get("data") or {}).get("reason") == "aborted_during_start", err
        assert _count(rig, "FIRE") == 0 and _count(rig, "ARMFIRE") == 0
        assert _engages(rig) == []
        run = _start_write(door, tok, [WIRE])           # the next start runs
        door.wait_state("idle", "engaged", "parked")
        res = door.call("getWriteResult", {"run_id": run["run_id"]})
        assert res["aborted"] is False, res
        _release(door, tok)


def test_igor_abort_during_the_scan_start_refuses_the_scan(hub, client, monkeypatch):
    from afm_litho.service.host import scan_cmds
    real_mode = scan_cmds.ensure_software_mode
    pressed: list = []

    async def mode_then_press(h, mode, *a, **kw):
        out = await real_mode(h, mode, *a, **kw)
        if not pressed:
            pressed.append(True)
            _press_and_wait(rig, hub)
        return out

    monkeypatch.setattr(scan_cmds, "ensure_software_mode", mode_then_press)
    with clock.using(clock.TESTING_TIME_SENSITIVE), GoldenRig(hub, "igor_abort_scan_start") as rig:
        door = rig.door
        tok = _acquire(door)
        cfg = {"token": tok, "size_um": 1.0, "pixels": 32, "lines": 2, "line_rate_hz": 50.0,
               "x_offset_um": 0.0, "y_offset_um": 0.0, "angle_deg": 0.0, "direction": "up",
               "trace_retrace": True, "mode": "contact"}
        runs_before = _stored_runs(hub)
        err = door.call_refused("startScan", cfg)
        assert (err.get("data") or {}).get("reason") == "aborted_during_start", err
        assert _count(rig, "FIRE") == 0 and _count(rig, "SCANCHAIN") == 0
        assert _stored_runs(hub) == runs_before, "the refused start stored a run"
        run = _scan(door, tok, mode="contact", lines=2)  # the next start runs
        door.wait_state("idle", "engaged", "parked")
        res = door.call("getScanResult", {"run_id": run["run_id"]})
        assert res["aborted"] is False, res
        assert _stored_runs(hub) != runs_before         # a started scan does store one
        _release(door, tok)


def _stored_runs(hub) -> set[str]:
    from afm_litho.service import run_store
    folder = run_store.store_dir(hub.store.path.parent)
    return {p.name for p in folder.glob("*.run.json")} if folder.is_dir() else set()


def test_igor_abort_after_the_scan_task_starts_aborts_the_scan(hub, client, monkeypatch):
    """The report lands once the pump task exists but before its first line, which clears
    the abort flag: the pump sees the stop counter moved and the scan never moves."""
    from afm_litho.hardware import bridge_protocol
    real_spawn = bridge_protocol.spawn_untraced
    pressed: list = []

    def spawn_then_press(coro):
        task = real_spawn(coro)
        if not pressed and getattr(coro, "__name__", "") == "_hw_scan_pump":
            pressed.append(True)
            _press_and_wait(rig, hub)
        return task

    monkeypatch.setattr(bridge_protocol, "spawn_untraced", spawn_then_press)
    with clock.using(clock.TESTING_TIME_SENSITIVE), GoldenRig(hub, "igor_abort_pump") as rig:
        door = rig.door
        tok = _acquire(door)
        run = _scan(door, tok, mode="contact", lines=2)
        door.wait_state("idle", "engaged", "parked")
        rig.quiesce()
        assert pressed
        res = door.call("getScanResult", {"run_id": run["run_id"]})
        assert res["aborted"] is True, res
        assert _count(rig, "FIRE") == 0 and _count(rig, "SCANCHAIN") == 0
        assert _engages(rig) == []
        _release(door, tok)
