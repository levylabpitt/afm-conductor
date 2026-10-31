"""The fake Igor speaks wire v3 (docs/migration/plan/phase-5-wire-v3-contract.md, slice W3).

Calls go to `FakeIgorBridge._dispatch` with a recording link, on the virtual clock, so the
timeline of make-safe (2.5 s, 5 s) and the dead-socket park run without sleeping. Line formats
come only from the contract (and its section 11 of W2's agreed points).
"""

from __future__ import annotations

import time

import pytest

from afm_litho import clock
from afm_litho.hardware import bridge_protocol as P
from afm_litho.hardware.igor_stub import FakeIgorBridge
from afm_litho.hardware.igor_stub_v3 import K_SOCK_DEAD_PARK_S, RING_SIZE

CYPHER = {"MicroscopeID": 4.0, "Height": 20.0}


class _Link:
    def __init__(self):
        self.sent: list[str] = []
        self._buf = b""

    def send(self, line: str) -> None:
        self.sent.append(line)

    def take(self, prefix: str) -> list[str]:
        return [x for x in self.sent if x.startswith(prefix)]


@pytest.fixture
def vclock():
    with clock.using(clock.TESTING):
        yield clock


class Rig:
    def __init__(self, values=None, **kw):
        self.stub = FakeIgorBridge(dict(values or CYPHER), **kw)
        self.link = _Link()
        self.n = 0

    def send(self, text: str, link=None) -> str:
        """Dispatch one v3 line; return the REPLY body for its seq."""
        self.n += 1
        link = link or self.link
        line = text.replace("#", str(self.n), 1)
        before = len(link.sent)
        self.stub._dispatch(line, link)
        for x in link.sent[before:]:
            if x.startswith(f"REPLY {self.n} "):
                return x[len(f"REPLY {self.n} "):]
        return ""

    def hello(self, proto=3):
        return self.send(f"HELLO # {proto} test")

    def reports(self):
        return [x.split(" ", 7) for x in self.link.sent if x.startswith("REPORT ")]

    def events(self):
        return [r[7] for r in self.reports()]

    def run_timers(self):
        clock.advance(0.0)
        time.sleep(0.05)            # the clock's timer thread runs the callbacks

    def close(self):
        self.stub.stop()


@pytest.fixture
def rig(vclock):
    r = Rig()
    yield r
    r.close()


def _load(r, name, vals, total=None):
    return r.send(f"LOAD # {name} {total or len(vals)} " + ";".join(str(v) for v in vals))


# ------------------------------------------------------------------------------ HELLO

def test_hello_reports_the_state_and_the_startup_keys(rig):
    body = rig.hello()
    assert body.startswith("ok proto=3;ipf=3;sha=0123456789abcdef;MicroscopeID=")
    for key in ("clock=", "ring=0-0;", "mark=0;", "fault=0;", "size_m="):
        assert key in body
    assert rig.stub.speaks_v3


def test_hello_with_another_version_is_a_mismatch_and_refuses_the_state_verbs(vclock):
    r = Rig(v3_proto=4)
    try:
        body = r.hello(proto=3)
        assert body.startswith("err mismatch proto=4;")
        assert _load(r, "w", [1, 2]) == "err host"
        assert r.send("PLAY # move r1 a b -200 32 200") == "err host"
        assert r.send("SET # value Output.A 1") == "err host"
        assert r.send("READ # value Height") == "ok 20"           # reads still work
        assert r.send("MAKESAFE #") == "ok"                       # and so does make-safe
    finally:
        r.close()


def test_state_verbs_need_a_hello_first(rig):
    assert _load(rig, "w", [1]) == "err host"
    assert rig.send("SET # mode AC") == "err host"
    assert rig.send("CLEARFAULT #") == "err host"


# ------------------------------------------------------------------------------ reads

def test_read_value_gv_and_xpt(rig):
    rig.stub.values["XSensor"] = 1.5
    rig.stub.crosspoint = {"InFast": "FilterOut"}
    assert rig.send("READ # value XSensor") == "ok 1.5"
    assert rig.send("READ # value NoSuch") == "ok nan"
    rig.stub.values["ScanSize"] = 3e-5
    assert rig.send("READ # gv ScanSize") == "ok 3e-05"
    assert rig.send("READ # xpt Crosspoint") == "ok InFast=FilterOut;"
    assert rig.stub.getgvs == ["ScanSize"]


def test_status_idle(rig):
    rig.hello()
    body = rig.send("STATUS #")
    assert body.startswith("ok pass=none;run=-;step=0/0;mark=0;fault=0;tipdown=0;loop2=none;")
    assert "sock_dead_s=0;" in body and "ring=0-0;" in body


# ------------------------------------------------------------------------------ LOAD

def test_load_replies_only_after_the_piece_is_stored(rig):
    rig.hello()
    assert _load(rig, "XLitho", [1, 2, 3], 6) == "ok XLitho 3/6"
    assert rig.stub.waves["XLitho"] == [1.0, 2.0, 3.0]
    assert rig.send("LOADAPP # XLitho 4;5;6") == "ok XLitho 6/6"
    assert rig.stub.waves["XLitho"] == [1, 2, 3, 4, 5, 6]


def test_load_ack_waits_for_a_choking_igor(rig):
    rig.hello()
    rig.stub.v3.load_delay_s = 1.5
    t0 = clock.now()
    assert _load(rig, "w", [1, 2]) == "ok w 2/2"
    assert clock.now() - t0 >= 1.5          # the reply came after the delay, not before


def test_a_missing_ack_is_silent_but_the_piece_is_stored(rig):
    rig.hello()
    rig.stub.v3.drop_load_acks_from = 2
    assert _load(rig, "w", [1, 2], 4) == "ok w 2/4"
    assert rig.send("LOADAPP # w 3;4") == ""
    assert rig.stub.waves["w"] == [1, 2, 3, 4]


def test_load_refuses_a_piece_over_8192_points(rig):
    rig.hello()
    assert _load(rig, "w", list(range(8193))).startswith("err bad piece")


def test_load_records_the_v2_wave_line_for_the_golden_tap(rig):
    rig.hello()
    _load(rig, "XLitho", [1, 2, 3])
    assert rig.stub.tap.raw[-1] == "WAVE XLitho 1;2;3"


# ------------------------------------------------------------------------------ SET

def test_set_value_loop_mode_showtip(rig):
    rig.hello()
    assert rig.send("SET # value Output.A 2.5") == "ok value Output.A 2.5"
    assert rig.stub.sets == [("Output.A", 2.5)]
    assert (rig.send("SET # loop 2 Deflection 0.5 0 1000 Height")
            == "ok loop 2 Deflection 0.5 0 1000 Height -10 150 err=0")
    assert rig.stub.loops == [(2, "Deflection", 0.5, 0.0, 1000.0, "Height")]
    assert rig.stub.loop_limits == [(-10.0, 150.0)]
    assert (rig.send("SET # loop 2 Deflection -10 0 1000 Height 0 150")
            == "ok loop 2 Deflection -10 0 1000 Height 0 150 err=0")
    assert rig.send("SET # mode AC") == "ok mode AC"
    assert rig.stub.modes == ["AC"]
    assert rig.send("SET # showtip ON") == "ok"
    assert rig.stub.showtips == [True]


def test_set_ramp_and_loopstop_stay_for_the_hover_move_and_old_park(rig):
    rig.hello()
    assert (rig.send("SET # ramp PIDSLoop.0.Setpoint 1 2 3")
            == "ok ramp PIDSLoop.0.Setpoint 1 2 3 err=0")
    assert rig.stub.ramps == [("PIDSLoop.0.Setpoint", 1.0, 2.0, 3.0)]
    assert rig.send("SET # loopstop 2") == "ok loopstop 2 err=0"
    assert rig.stub.stopped_loops == [2]


# ------------------------------------------------------------------------------ PLAY

def _arm_xy_waves(r):
    r.stub.waves["xa"] = [0.0] * 64
    r.stub.waves["ya"] = [0.0] * 64


def test_play_move_runs_the_v2_outpair_inpair_fire_and_reports(rig):
    rig.hello()
    _arm_xy_waves(rig)
    assert rig.send("PLAY # move r1 xa ya -200 64 200") == "ok armed r1"
    rig.run_timers()
    assert rig.stub.out_binds[0][:4] == (0, "0", "PIDSLoop.0.Setpoint", "xa")
    assert rig.stub.in_channels[0] == (0, "0", "XSensor", 64, 200)
    assert rig.stub.fires == [("move", "0")]
    assert rig.events()[0] == "armed"
    assert rig.events()[-2:] == ["step 0 1 64", "done 64"]
    caps = rig.link.take("CAP ")
    assert [c.split(" ")[6] for c in caps] == ["XSensor", "YSensor"]


def test_play_litho1_equals_the_v2_armfire(rig):
    rig.hello()
    for name in ("XLitho", "YLitho", "VLitho"):
        _load(rig, name, [0.0] * 64)
    assert rig.send("PLAY # litho1 r2 obj XLitho YLitho VLitho -500 64 500") == "ok armed r2"
    rig.run_timers()
    assert rig.stub.arms == [("obj", -500.0, 64, 500)]
    assert rig.stub.bank_fires[0]["out"][0][2] == ["XLitho", "YLitho"]
    assert rig.events()[-1].startswith("done")


def test_play_litho1_with_other_wave_names_aliases_them(rig):
    rig.hello()
    for name in ("a", "b", "c"):
        _load(rig, name, [1.0, 2.0] * 32)
    rig.send("PLAY # litho1 r2 obj a b c -500 64 500")
    assert rig.stub.waves["XLitho"] == rig.stub.waves["a"]
    assert rig.stub.tap.raw.count("WAVE XLitho " + ";".join(["1", "2"] * 32)) == 1
    rig.run_timers()


def test_play_scanchain_reports_a_step_per_line(rig):
    rig.hello()
    _load(rig, "AFM_scLX", [0.0] * 128)
    _load(rig, "AFM_scLY", [0.0] * 128)
    assert (rig.send("PLAY # scanchain r3 AFM_scLX AFM_scLY 64 2 -100 100 Deflection")
            == "ok armed r3")
    deadline = time.monotonic() + 5
    while not any(e.startswith("done") for e in rig.events()) and time.monotonic() < deadline:
        rig.run_timers()
    ev = rig.events()
    assert ev[0] == "armed" and ev[-1] == "done 2"
    assert [e for e in ev if e.startswith("step")] == ["step 0 2 64", "step 1 2 64"]
    assert rig.stub.scanchains == [("AFM_scLX", "AFM_scLY", 64, 2, -100.0, 100)]


def test_play_lithochain_reports_a_step_per_chunk(rig):
    rig.hello()
    for name in ("cx", "cy", "cv"):
        _load(rig, name, [0.0] * 128)
    _load(rig, "bnd", [0, 64, 128])
    assert rig.send("PLAY # lithochain r4 obj cx cy cv bnd -500 500") == "ok armed r4"
    rig.run_timers()
    assert rig.stub.lithochains[0][0] == "obj"
    ev = rig.events()
    assert [e for e in ev if e.startswith("step")] == ["step 0 2 64", "step 1 2 64"]
    assert ev[-1] == "done 128"
    assert rig.stub.bias_zeros == [("obj", 1)]          # only at the true end


def test_play_while_a_pass_is_active_is_busy(rig):
    rig.hello()
    _arm_xy_waves(rig)
    rig.stub.done_delay = 3600.0
    assert rig.send("PLAY # move r1 xa ya -200 64 200") == "ok armed r1"
    assert rig.send("PLAY # move r2 xa ya -200 64 200") == "err busy"
    assert rig.send("STATUS #").startswith("ok pass=move;run=r1;")


def test_up_messages_are_stamped_and_useq_counts_every_up_message(rig):
    rig.hello()
    _arm_xy_waves(rig)
    rig.send("PLAY # move r1 xa ya -200 64 200")
    rig.run_timers()
    seen = []
    for x in rig.link.sent:
        if x.startswith(("REPORT ", "CAP ")):
            f = x.split(" ")
            seen.append(int(f[1]))
            assert f[3] == "0"                          # mark
    assert seen == list(range(1, len(seen) + 1))        # one series, no gaps
    rep = rig.reports()[0]
    assert rep[3:7] == ["0", "0", "sock", "r1"]         # mark fault route run


# ------------------------------------------------------------------------------ ring

def test_fetch_resends_the_reports_from_a_useq(rig):
    rig.hello()
    for i in range(3):
        rig.stub.v3.up_report(f"r{i}", "armed")
    rig.link.sent.clear()
    body = rig.send("FETCH # 2")
    assert [x.split(" ")[1] for x in rig.link.take("REPORT ")] == ["2", "3"]
    assert body == "ok 1-3"


def test_the_ring_keeps_the_last_256_and_fetch_says_gone(rig):
    rig.hello()
    for i in range(RING_SIZE + 10):
        rig.stub.v3.up_report("r", "armed")
    assert rig.stub.v3.ring_span() == "11-266"
    assert rig.send("FETCH # 5") == "err gone 11"
    assert rig.send("FETCH # 11").startswith("ok 11-266")


def test_captures_are_not_ringed(rig):
    rig.hello()
    rig.stub.v3.up_cap("r", 0, "ZSensor", "1;2")
    assert rig.stub.v3.ring_span() == "0-0"
    assert rig.stub.v3.useq == 1


def test_cap_pieces_are_8192_points(rig):
    rig.hello()
    rig.stub.v3.up_cap("r", 0, "ZSensor", ";".join(["1"] * 20000))
    sizes = [len(x.split(" ", 7)[7].split(";")) for x in rig.link.take("CAP ")]
    assert sizes == [8192, 8192, 3616]


# ------------------------------------------------------------------------------ make-safe

def _engage(r):
    r.hello()
    r.send("SET # loop 2 Deflection 0.5 0 1000 Height")
    r.send("SET # value Output.A 3")


def test_makesafe_runs_the_park_sequence_and_replies_at_once(rig):
    _engage(rig)
    assert rig.send("MAKESAFE #") == "ok"
    v3 = rig.stub.v3
    assert v3.igor_calls == [
        "td_Stop()", 'td_WriteValue("Output.A",0)',
        'ir_SetPISLoop(2,"Always,Never","Deflection",-10,0,1000,0,"Height",0,150)']
    assert v3.abort_stopped
    assert rig.events()[-1] == "madesafe stop"
    assert "ABORT" in rig.stub.tap.raw
    status = rig.send("STATUS #")
    assert "mark=1;" in status and "loop2=-10,0,150;" in status


def test_a_repeated_makesafe_does_not_call_td_stop_again(rig):
    _engage(rig)
    rig.send("MAKESAFE #")
    rig.send("MAKESAFE #")
    calls = rig.stub.v3.igor_calls
    assert calls.count("td_Stop()") == 1
    assert calls.count('td_WriteValue("Output.A",0)') == 2          # bias and re-arm every time
    assert sum(c.startswith("ir_SetPISLoop") for c in calls) == 2


def test_confirm_under_one_volt_clears_the_mark(rig):
    _engage(rig)
    rig.send("MAKESAFE #")
    assert rig.stub.values["Height"] == 0.0                         # the re-arm retracted
    assert rig.send("CONFIRMSAFE #") == "ok height=0"
    assert not rig.stub.v3.abort_stopped
    assert rig.events()[-1] == "confirmed 0"
    assert not rig.stub.v3.tip_down
    assert rig.reports()[-1][3] == "0"                              # stamped after the clear


def test_confirm_refuses_while_height_is_high_and_the_mark_stays(vclock):
    r = Rig()
    try:
        r.stub.safe_park_zeros = False
        _engage(r)
        r.send("MAKESAFE #")
        assert r.send("CONFIRMSAFE #") == "err height=20"
        assert r.stub.v3.abort_stopped
        r.stub.set_height(0.4)
        assert r.send("CONFIRMSAFE #") == "ok height=0.4"
    finally:
        r.close()


def test_vendor_withdraw_runs_once_and_sets_the_fault(vclock):
    r = Rig()
    try:
        r.stub.safe_park_zeros = False
        _engage(r)
        r.send("MAKESAFE #")
        assert r.send("VENDORWITHDRAW #") == "ok ran"
        assert r.send("VENDORWITHDRAW #") == "ok already"
        assert r.stub.v3.abort_fault
        assert r.stub.v3.igor_calls.count('DoScanFunc("StopEngageButton")') == 1
        assert "@DOSCANFUNC StopEngageButton" in r.stub.tap.raw
        assert r.stub.values["Height"] == 0.0
        assert "fault=1;" in r.send("STATUS #")
    finally:
        r.close()


def test_the_5s_backup_clears_the_mark_when_height_is_low(rig):
    _engage(rig)
    rig.send("MAKESAFE #")
    clock.advance(4.9)
    rig.run_timers()
    assert rig.stub.v3.abort_stopped
    clock.advance(0.2)
    rig.run_timers()
    assert not rig.stub.v3.abort_stopped
    assert rig.events()[-1] == "backup 0 cleared"
    assert not rig.stub.v3.abort_fault


def test_the_5s_backup_runs_the_vendor_withdraw_and_faults_when_height_is_high(vclock):
    r = Rig()
    try:
        r.stub.safe_park_zeros = False
        _engage(r)
        r.send("MAKESAFE #")
        clock.advance(5.1)
        r.run_timers()
        assert r.events()[-1] == "backup 20 withdraw"
        assert r.stub.v3.abort_fault and r.stub.v3.abort_stopped      # mark stays until confirm
        assert r.reports()[-1][4] == "1"                              # the fault stamp
        assert r.send("PLAY # move r1 a b -200 32 200") == "err mark"
        assert r.send("SET # value Output.A 0") == "ok value Output.A 0"   # SET is never refused
    finally:
        r.close()


def test_each_makesafe_restarts_the_5s_backup(vclock):
    r = Rig()
    try:
        r.stub.safe_park_zeros = False
        _engage(r)
        r.send("MAKESAFE #")
        clock.advance(3.0)
        r.send("MAKESAFE #")
        clock.advance(3.0)
        r.run_timers()
        assert r.stub.backup_runs == []                  # 5 s from the LAST call not yet up
        clock.advance(2.1)
        r.run_timers()
        assert r.stub.backup_runs == ["fault"]
    finally:
        r.close()


def test_play_is_refused_while_the_mark_or_the_fault_is_set(rig):
    _engage(rig)
    rig.send("MAKESAFE #")
    assert rig.send("PLAY # move r1 a b -200 32 200") == "err mark"
    rig.send("CONFIRMSAFE #")
    assert rig.send("PLAY # move r1 a b -200 32 200") == "ok armed r1"
    rig.stub.v3.abort_fault = True
    rig.stub.v3.run = None
    assert rig.send("PLAY # move r2 a b -200 32 200") == "err fault"


def test_clearfault_zeroes_both_flags_and_abortfault_is_read_only(vclock):
    r = Rig()
    try:
        r.stub.safe_park_zeros = False
        _engage(r)
        r.send("MAKESAFE #")
        assert r.send("READ # flag abortFault") == "ok 0"
        clock.advance(5.1)
        r.run_timers()
        assert r.send("READ # flag abortFault") == "ok 1"
        assert r.send("READ # flag nosuch").startswith("err bad")
        assert r.send("CLEARFAULT #") == "ok fault=0"
        assert not r.stub.abort_stopped and not r.stub.abort_fault
        assert r.send("READ # flag abortFault") == "ok 0"
    finally:
        r.close()


def test_v2_clearfault_without_a_seq_still_goes_to_the_v2_path(rig):
    rig.stub.abort_fault = True
    rig.stub._dispatch("CLEARFAULT", rig.link)
    assert not rig.stub.abort_fault and rig.stub.fault_clears == 1


def test_read_str_of_a_running_loop(rig):
    rig.hello()
    rig.send("SET # loop 2 Deflection 0.5 0 1000 Height")
    assert rig.send("READ # str PIDSLoop.2.OutputChannel") == "ok PIDSLoop.2.OutputChannel=Height;"
    assert rig.send("READ # str PIDSLoop.1.OutputChannel").startswith("err")


def test_makesafe_ends_a_running_pass_with_one_aborted(rig):
    rig.hello()
    _arm_xy_waves(rig)
    rig.stub.done_delay = 3600.0
    rig.send("PLAY # move r1 xa ya -200 64 200")
    rig.send("MAKESAFE #")
    assert rig.events()[-2:] == ["aborted", "madesafe stop"]
    assert rig.events().count("aborted") == 1
    assert rig.stub.v3.run is None


# ------------------------------------------------------------------------------ rig facts, sim

def _sim_rig():
    return Rig({"MicroscopeID": 4.0, "Height": 74.0}, safety_sim=True, safety_sim_real_time=True)


def test_td_stop_undefines_loop_2_and_the_rearm_alone_retracts(vclock):
    r = _sim_rig()
    try:
        r.hello()
        r.send("SET # loop 2 Deflection 0.5 0 1000 Height")
        sim = r.stub.safety_sim
        assert 2 in sim.loop_state
        r.send("MAKESAFE #")
        assert sim.loop_state[2]["setpoint"] == -10             # re-armed after td_Stop
        clock.advance(0.5)
        assert r.send("READ # value Height") != "ok 74"        # Z is moving
        clock.advance(1.0)
        assert r.send("READ # value Height") == "ok 0"
        assert r.send("CONFIRMSAFE #") == "ok height=0"
    finally:
        r.close()


def test_a_height_ramp_with_loop_2_stopped_is_inert(vclock):
    r = _sim_rig()
    try:
        r.hello()
        r.send("SET # loop 2 Deflection 0.5 0 1000 Height")
        r.send("SET # loopstop 2")
        r.send("SET # ramp Height 0 80 2")
        clock.advance(2.0)
        assert r.send("READ # value Height") == "ok 74"
        assert r.stub.safety_sim.ignored_ramps
    finally:
        r.close()


def test_confirm_at_2_5s_and_the_vendor_withdraw_when_height_is_stuck(vclock):
    r = _sim_rig()
    try:
        r.hello()
        r.send("SET # loop 2 Deflection 0.5 0 1000 Height")
        r.send("MAKESAFE #")
        r.stub.safety_sim.undefine_loops()                      # the retract never happens
        clock.advance(2.5)
        assert r.send("CONFIRMSAFE #") == "err height=74"
        assert r.send("VENDORWITHDRAW #") == "ok ran"
        assert r.send("READ # value Height") == "ok 0"
    finally:
        r.close()


# ------------------------------------------------------------------------------ socket

def test_a_closed_socket_does_not_stop_the_run_and_its_reports_are_ringed(rig):
    rig.hello()
    _arm_xy_waves(rig)
    rig.stub.done_delay = 5.0
    rig.send("PLAY # move r1 xa ya -200 64 200")
    rig.stub.v3.on_close(rig.link)
    sent_before = len(rig.link.sent)
    clock.advance(5.1)
    rig.run_timers()
    assert len(rig.link.sent) == sent_before                    # nothing reaches a dead socket
    ring = [line for _, line in rig.stub.v3.ring]
    assert [r.split(" ", 7)[7] for r in ring][-2:] == ["step 0 1 64", "done 64"]
    assert rig.stub.v3.run is None


def test_a_dead_socket_with_the_tip_down_and_no_run_parks_after_5s(rig):
    _engage(rig)
    rig.send("SET # loop 2 Deflection 0.5 0 1000 Height")
    rig.stub.v3.on_close(rig.link)
    clock.advance(K_SOCK_DEAD_PARK_S - 0.5)
    rig.run_timers()
    assert not rig.stub.v3.abort_stopped
    clock.advance(0.6)
    rig.run_timers()
    v3 = rig.stub.v3
    assert v3.abort_stopped and v3.igor_calls[0] == "td_Stop()"
    assert v3.ring[-1][1].split(" ", 7)[7] == "madesafe socket"


def test_a_dead_socket_with_the_tip_up_does_not_park(rig):
    rig.hello()
    rig.send("SET # loop 2 Deflection 0.5 0 1000 Height")
    rig.send("SET # loop 2 Deflection -10 0 1000 Height 0 150")        # lifted
    rig.stub.v3.on_close(rig.link)
    clock.advance(K_SOCK_DEAD_PARK_S + 1)
    rig.run_timers()
    assert not rig.stub.v3.abort_stopped and rig.stub.v3.igor_calls == []


def test_a_dead_socket_waits_for_the_run_to_end_then_parks(rig):
    rig.hello()
    rig.send("SET # loop 2 Deflection 0.5 0 1000 Height")           # engaged, tip down
    _arm_xy_waves(rig)
    rig.stub.done_delay = 8.0
    rig.send("PLAY # move r1 xa ya -200 64 200")
    rig.stub.v3.on_close(rig.link)
    clock.advance(6.0)
    rig.run_timers()
    assert not rig.stub.v3.abort_stopped                            # the run is still going
    clock.advance(3.0)
    rig.run_timers()
    assert rig.stub.v3.abort_stopped                                # parked once it ended


def test_td_stop_alone_leaves_the_tip_state_as_it_was(rig):
    _engage(rig)
    rig.send("MAKESAFE #")
    rig.send("CLEARFAULT #")
    # tip_down is cleared only by a confirmed make-safe or a -10 loop-2 command
    assert rig.stub.v3.tip_down is True


def test_a_new_hello_cancels_the_dead_socket_park(rig):
    _engage(rig)
    rig.send("SET # loop 2 Deflection 0.5 0 1000 Height")
    rig.stub.v3.on_close(rig.link)
    clock.advance(2.0)
    new = _Link()
    rig.hello_new = rig.send("HELLO # 3 t", link=new)
    clock.advance(10.0)
    rig.run_timers()
    assert not rig.stub.v3.abort_stopped
    assert rig.stub.v3.sock_dead_s() == 0.0


# ------------------------------------------------------------------------------ TCP

def test_makesafe_jumps_the_queue_and_drops_what_is_behind_it(vclock):
    r = Rig()
    port = r.stub.serve_in_thread()
    link = P.BridgeLink.connect("127.0.0.1", port)
    try:
        link.send("HELLO 1 3 t")
        assert link.read_line(5).startswith("REPLY 1 ok proto=3")
        link.sock.sendall(b"SET 2 value Output.A 1\nPLAY 3 move r1 a b -200 32 200\n"
                          b"MAKESAFE 4\nREAD 5 value Height\n")
        got = [link.read_line(5) for _ in range(5)]
        replies = [x for x in got if x.startswith("REPLY")]
        assert replies[0] == "REPLY 4 ok"                                   # first
        assert "REPLY 2 err dropped" in replies and "REPLY 3 err dropped" in replies
        assert any(x.startswith("REPLY 5 ok") for x in replies)
        assert r.stub.sets == []                                            # SET never ran
    finally:
        link.close()
        r.close()


def test_dropping_the_socket_from_the_instrument_side_starts_the_dead_clock(vclock):
    r = Rig()
    port = r.stub.serve_in_thread()
    link = P.BridgeLink.connect("127.0.0.1", port)
    try:
        link.send("HELLO 1 3 t")
        link.read_line(5)
        r.stub.drop_host_socket()
        deadline = time.monotonic() + 5
        while r.stub.v3.dead_since is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert r.stub.v3.dead_since is not None
    finally:
        link.close()
        r.close()


def test_a_v3_connection_gets_tick_instead_of_telem(vclock):
    r = Rig(telem_hz=50.0)
    port = r.stub.serve_in_thread()
    link = P.BridgeLink.connect("127.0.0.1", port)
    try:
        first = link.read_line(5)
        assert first.startswith("TELEM ")
        link.send("HELLO 1 3 t")
        frames = [link.read_line(5) for _ in range(40)]
        ticks = [f for f in frames if f.startswith("TICK ")]
        assert ticks
        f = ticks[0].split(" ")
        assert f[0] == "TICK" and f[4] == "0" and f[5] == "-" and len(f) == 12
    finally:
        link.close()
        r.close()
