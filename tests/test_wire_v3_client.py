"""Wire v3 client (hardware/wire_v3.py) against a scripted fake socket.

The fake is a thread on the far end of a socketpair: it records every line and answers
each one with the lines a script function returns. It is not the v3 fake Igor (slice W3);
it only checks the client's side of the contract (phase-5-wire-v3-contract.md).
"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest

from afm_litho.hardware import bridge_protocol as P
from afm_litho.hardware import bridge_router, wire, wire_v3
from afm_litho.hardware.bridge_protocol import BridgeLink
from afm_litho.hardware.bridge_router import BridgeRouter
from afm_litho.hardware.hardware_arc import HardwareARC

HELLO_BODY = "proto=3;ipf=3;sha=;MicroscopeID=4;clock=100;ring=1-1;mark=0;fault=0;microscope_id=4"


class FakeIgor:
    """Far end of the socket: `script(line, fake)` returns the lines to send back."""

    def __init__(self, script):
        a, b = socket.socketpair()
        self.client = BridgeLink(a)
        self._far = BridgeLink(b)
        self.script = script
        self.got: list[str] = []
        self.useq = 0
        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stamp(self) -> int:
        self.useq += 1
        return self.useq

    def report(self, run, event, fields="", mark=0, fault=0):
        return f"REPORT {self.stamp()} 100 {mark} {fault} sock {run} {event} {fields}".rstrip()

    def cap(self, run, step, alias, values):
        return f"CAP {self.stamp()} 100 0 {run} {step} {alias} " + ";".join(
            f"{v:g}" for v in values)

    def tick(self, fault=0, mark=0):
        return f"TICK {self.stamp()} 100 {mark} {fault} - 1 2 3 4 5 6"

    def push(self, line: str) -> None:
        self._far.send(line)

    def _loop(self):
        while not self._stop:
            try:
                line = self._far.read_line(timeout=0.2)
            except socket.timeout:
                continue
            except OSError:
                return
            if line is None:
                return
            self.got.append(line)
            for out in self.script(line, self) or ():
                if callable(out):
                    out()
                else:
                    self._far.send(out)

    def verbs(self) -> list[str]:
        return [g.split(" ")[0] for g in self.got]

    def close(self):
        self._stop = True
        self.client.close()
        self._far.close()


def seq_of(line: str) -> str:
    return line.split(" ")[1]


def igor_v3(line, fake, *, height=0.2):
    """A well-behaved v3 Igor for the verbs these tests use."""
    verb = line.split(" ")[0]
    s = seq_of(line) if " " in line else ""
    if verb == "HELLO":
        return [f"REPLY {s} ok {HELLO_BODY}"]
    if verb == "STARTUP":                              # permanent safe set, still sent (F100)
        return [f"STARTUP {s} proto=2;microscope_id=4;x_lvdt_sens=1"]
    if verb == "READ":
        kind = line.split(" ")[2]
        if kind == "xpt":
            return [f"REPLY {s} ok InFast=FilterOut;"]
        return [f"REPLY {s} ok {height:g}"]
    if verb == "SET":
        return [f"REPLY {s} ok {' '.join(line.split(' ')[2:])} err=0"]
    if verb in ("LOAD", "LOADAPP"):
        return [f"REPLY {s} ok {line.split(' ')[2]} 1/1"]
    if verb == "CONFIRMSAFE":
        return [f"REPLY {s} ok height={height:g}"]
    if verb == "VENDORWITHDRAW":
        return [f"REPLY {s} ok ran"]
    if verb == "MAKESAFE":
        return [f"REPLY {s} ok", fake.report("-", "madesafe", "stop", mark=1)]
    if verb == "CLEARFAULT":
        return [f"REPLY {s} ok fault=0"]
    if verb == "FETCH":
        return [f"REPLY {s} ok 1-9"]
    return []


@pytest.fixture
def v3():
    with wire.use_version(3):
        yield


def make_arc(script, **kw):
    fake = FakeIgor(script)
    arc = HardwareARC(fake.client, timeout=5.0, **kw)
    return fake, arc


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


# ------------------------------------------------------------------ switch --

def test_switch_defaults_to_v2():
    assert wire.WIRE_VERSION == 2
    fake, arc = make_arc(lambda line, f: [])
    try:
        assert type(arc.router) is BridgeRouter
        assert not hasattr(arc.link, "load_wave")
    finally:
        arc.close()
        fake.close()


def test_use_version_restores():
    with wire.use_version(3):
        assert wire.WIRE_VERSION == 3
    assert wire.WIRE_VERSION == 2
    with pytest.raises(ValueError):
        with wire.use_version(4):
            pass


# ------------------------------------------------------------- HELLO, gate --

def test_hello_match_answers_startup(v3):
    fake, arc = make_arc(igor_v3)
    try:
        startup = arc.start()
        assert fake.got[0].startswith("HELLO ") and fake.got[0].split(" ")[2] == "3"
        assert arc.link.state == "v3" and arc.link.version_banner is None
        assert arc.is_cypher and startup["x_lvdt_sens"] == 1.0
        assert fake.verbs()[:2] == ["HELLO", "STARTUP"]  # STARTUP still runs MacroStartup
    finally:
        arc.close()
        fake.close()


def test_version_mismatch_reads_and_make_safe_work_set_and_play_refused(v3):
    def script(line, fake):
        if line.startswith("HELLO"):
            return [f"REPLY {seq_of(line)} err mismatch proto=4;ipf=4;sha=x;"]
        return igor_v3(line, fake)
    fake, arc = make_arc(script)
    try:
        arc.start()                                    # does not brick the connect
        assert arc.link.state == "mismatch"
        assert arc.link.version_banner.startswith(
            "Igor procedures are v4, this program needs v3. Park the tip, then run "
            "`python scripts/install_afm_ipf.py`")
        assert arc.td_ReadValue("Height") == pytest.approx(0.2)
        with pytest.raises(wire_v3.WireRefused):
            arc.ir_SetPISLoop(2, "Always,Never", "Deflection", 0, 0, 1000, 0, "Height", -10, 150)
        with pytest.raises(wire_v3.WireRefused):
            arc._load_wave(np.zeros(4))
        bridge_router.send_safe_park(arc.link, None)
        # another version's procedures get the permanent ABORT, not MAKESAFE (§7)
        assert wait_for(lambda: "ABORT" in fake.verbs())
        assert not any(v in ("SET", "LOAD", "PLAY", "MAKESAFE") for v in fake.verbs())
    finally:
        arc.close()
        fake.close()


def test_hand_edited_ipf_waits_for_operator(v3, monkeypatch):
    monkeypatch.setattr(wire_v3, "expected_ipf_sha", lambda path=None: "aaaa")
    def script(line, fake):
        if line.startswith("HELLO"):
            return [f"REPLY {seq_of(line)} ok proto=3;ipf=3;sha=bbbb;"]
        return igor_v3(line, fake)
    fake, arc = make_arc(script)
    try:
        arc.start()
        assert arc.link.state == "hand_edit" and "differ" in arc.link.version_banner
        with pytest.raises(wire_v3.WireRefused):
            arc.td_WriteValue("Output.A", 0)
        arc.link.accept_hand_edit()
        arc.td_WriteValue("Output.A", 0)
        assert any(g.startswith("SET ") and " value Output.A 0" in g for g in fake.got)
    finally:
        arc.close()
        fake.close()


def test_old_ipf_falls_back_to_v2(v3):
    def script(line, fake):
        if line.startswith("HELLO"):
            return ["ERR - unknown verb: HELLO"]
        if line.startswith("STARTUP"):
            return [f"STARTUP {seq_of(line)} proto=2;microscope_id=4"]
        return []
    fake, arc = make_arc(script)
    try:
        arc.start()
        assert arc.link.state == "v2"
        arc.td_WriteValue("Output.A", 0)
        arc._load_wave(np.zeros(3))
        assert wait_for(lambda: len(fake.got) >= 4)
        assert fake.got[2] == "SETVALUE Output.A 0" and fake.got[3].startswith("WAVE ow0 ")
    finally:
        arc.close()
        fake.close()


def test_hello_bad_proto_answer_needs_hello_first(v3):
    fake, arc = make_arc(igor_v3)
    try:
        arc.td_WriteValue("Output.A", 0)               # a set before start() says HELLO first
        assert fake.verbs()[:2] == ["HELLO", "SET"]
    finally:
        arc.close()
        fake.close()


# ------------------------------------------------------------------- verbs --

def test_reads(v3):
    fake, arc = make_arc(igor_v3)
    try:
        arc.start()
        assert arc.td_ReadValue("Deflection") == pytest.approx(0.2)
        assert arc.get_gv("XLVDTSens") == pytest.approx(0.2)
        assert arc.check_mode() == "contact"
        kinds = [g.split(" ")[2:] for g in fake.got if g.startswith("READ")]
        assert kinds == [["value", "Deflection"], ["gv", "XLVDTSens"], ["xpt", "Crosspoint"]]
    finally:
        arc.close()
        fake.close()


def test_read_err_raises(v3):
    def script(line, fake):
        if line.startswith("READ"):
            return [f"REPLY {seq_of(line)} err no such alias"]
        return igor_v3(line, fake)
    fake, arc = make_arc(script)
    try:
        arc.start()
        with pytest.raises(RuntimeError, match="no such alias"):
            arc.td_ReadValue("Bogus")
    finally:
        arc.close()
        fake.close()


def test_every_set_verb(v3):
    fake, arc = make_arc(igor_v3)
    try:
        arc.start()
        arc.td_WriteValue("PIDSLoop.2.Setpoint", 0.5)
        arc.ir_SetPISLoop(2, "Always,Never", "Deflection", -10, 0, 1000, 0, "Height", 0, 150)
        arc.ir_StopPISLoop(2)
        arc.ramp_output("PIDSLoop.0.Setpoint", -0.63, 0.67, 0.52)
        arc.set_imaging_mode("ac")
        arc.link.send(P.cmd_showtip(True))
        sets = [" ".join(g.split(" ")[2:]) for g in fake.got if g.startswith("SET ")]
        assert sets == [
            "value PIDSLoop.2.Setpoint 0.5",
            "loop 2 Deflection -10 0 1000 Height 0 150",
            "loopstop 2",
            "ramp PIDSLoop.0.Setpoint -0.63 0.67 0.52",
            "mode AC",
            "showtip ON",
        ]
        seqs = [seq_of(g) for g in fake.got]
        assert len(seqs) == len(set(seqs))             # one counter, unique per connection
    finally:
        arc.close()
        fake.close()


def test_set_refused_by_igor_raises(v3):
    def script(line, fake):
        if line.startswith("SET"):
            return [f"REPLY {seq_of(line)} err host"]
        return igor_v3(line, fake)
    fake, arc = make_arc(script)
    try:
        arc.start()
        with pytest.raises(wire_v3.WireRefused, match="err|host"):
            arc.td_WriteValue("Output.A", 0)
    finally:
        arc.close()
        fake.close()


def test_set_without_reply_times_out(v3, monkeypatch):
    monkeypatch.setattr(wire_v3, "PIECE_TIMEOUT_S", 0.3)
    def script(line, fake):
        return [] if line.startswith("SET") else igor_v3(line, fake)
    fake, arc = make_arc(script)
    try:
        arc.start()
        with pytest.raises(wire_v3.WireTimeout):
            arc.td_WriteValue("Output.A", 0)
    finally:
        arc.close()
        fake.close()


def test_unknown_v2_verb_has_no_v3_line(v3):
    fake, arc = make_arc(igor_v3)
    try:
        arc.start()
        with pytest.raises(ValueError, match="OUTWAVE"):
            arc.link.send(P.cmd_outwave(2, "14", "Output.A", "ow0", -500))
    finally:
        arc.close()
        fake.close()


# ---------------------------------------------------------------- loading --

def test_load_is_paced_by_replies(v3):
    """Piece k+1 is not sent before Igor's reply to piece k."""
    order: list[str] = []
    gate = threading.Event()

    def script(line, fake):
        verb = line.split(" ")[0]
        if verb in ("LOAD", "LOADAPP"):
            order.append(f"got {verb}")
            if verb == "LOAD":
                def later():
                    gate.wait(0.4)                     # Igor is slow to store piece 0
                    order.append("reply LOAD")
                    fake.push(f"REPLY {seq_of(line)} ok {line.split(' ')[2]} 8192/9000")
                threading.Thread(target=later, daemon=True).start()
                return []
            order.append("reply LOADAPP")
            return [f"REPLY {seq_of(line)} ok {line.split(' ')[2]} 9000/9000"]
        return igor_v3(line, fake)
    fake, arc = make_arc(script)
    try:
        arc.start()
        name = arc._load_wave(np.arange(9000, dtype=float))
        assert order == ["got LOAD", "reply LOAD", "got LOADAPP", "reply LOADAPP"]
        load = next(g for g in fake.got if g.startswith("LOAD "))
        head = load.split(" ")
        assert head[2] == name and head[3] == "9000"
        assert len(head[4].split(";")) == wire_v3.LOAD_PIECE_POINTS
        app = next(g for g in fake.got if g.startswith("LOADAPP "))
        assert app.split(" ")[3].split(";")[0] == "8192"
    finally:
        arc.close()
        fake.close()


def test_load_missing_reply_fails(v3, monkeypatch):
    monkeypatch.setattr(wire_v3, "PIECE_TIMEOUT_S", 0.3)
    def script(line, fake):
        return [] if line.startswith("LOAD") else igor_v3(line, fake)
    fake, arc = make_arc(script)
    try:
        arc.start()
        with pytest.raises(wire_v3.WireTimeout):
            arc._load_wave(np.zeros(100))
    finally:
        arc.close()
        fake.close()


def test_load_pause_fallback_off_by_default_and_on_by_switch(v3, monkeypatch):
    assert wire_v3.LOAD_PAUSE_ON is False
    slept: list[float] = []
    monkeypatch.setattr(wire_v3.clock, "sleep", lambda s: slept.append(s))
    fake, arc = make_arc(igor_v3)
    try:
        arc.start()
        arc._load_wave(np.zeros(5 * 8192))
        assert slept == []
        monkeypatch.setattr(wire_v3, "LOAD_PAUSE_ON", True)
        arc._load_wave(np.zeros(5 * 8192))
        assert slept == [wire_v3.LOAD_PAUSE_S] * 4
        slept.clear()
        arc._load_wave(np.zeros(4 * 8192))             # 4 pieces: at the limit, no pause
        assert slept == []
    finally:
        arc.close()
        fake.close()


# ------------------------------------------------------------------- PLAY --

def play_script(capture_steps, *, done_extra=""):
    """Igor that answers PLAY with armed, the steps' CAPs, and done."""
    def script(line, fake):
        if line.startswith("PLAY"):
            t = line.split(" ")
            s, run = t[1], t[3]
            out = [f"REPLY {s} ok armed {run}", fake.report(run, "armed")]
            for k, (alias, vals) in enumerate(capture_steps):
                out.append(fake.report(run, "step", f"{k} {len(capture_steps)} {len(vals)}"))
                out.append(fake.cap(run, k, alias, vals))
            out.append(fake.report(run, "done", done_extra))
            return out
        return igor_v3(line, fake)
    return script


def test_play_litho1(v3):
    fake, arc = make_arc(play_script([("ZSensor", [1, 2, 3])]))
    try:
        arc.start()
        w = arc.router.open_waiter()
        arc.link.send(P.cmd_armfire(w.uid, "obj", -500, 64, 500))
        frames = [w.get(3), w.get(3)]
        assert frames == [(P.CAPTURE, "obj ZSensor 1;2;3"), (P.DONE, "obj")]
        play = next(g for g in fake.got if g.startswith("PLAY"))
        assert play == f"PLAY {w.uid} litho1 {w.uid} obj XLitho YLitho VLitho -500 64 500"
    finally:
        arc.close()
        fake.close()


def test_play_lithochain_fields(v3):
    fake, arc = make_arc(play_script([("ZSensor", [1]), ("ZSensor", [2])]))
    try:
        arc.start()
        w = arc.router.open_waiter()
        arc.link.send(P.cmd_lithochain(w.uid, "big", -50, 50, "LithoBounds"))
        assert [w.get(3) for _ in range(3)][-1] == (P.DONE, "big")
        play = next(g for g in fake.got if g.startswith("PLAY"))
        assert play == (f"PLAY {w.uid} lithochain {w.uid} big XLitho YLitho VLitho "
                        "LithoBounds -50 50")
    finally:
        arc.close()
        fake.close()


def test_play_move_from_transit_arm(v3):
    fake, arc = make_arc(play_script([("XSensor", [0.5]), ("YSensor", [0.25])]))
    try:
        arc.start()
        arc.td_xSetOutWavePair(0, "0", "PIDSLoop.0.Setpoint", np.zeros(32),
                               "PIDSLoop.1.Setpoint", np.zeros(32), -200)
        arc.td_xSetInWavePair(0, "0", "XSensor", "YSensor", 32, 200)
        cap = arc.fire("0", 32)
        assert cap["XSensor"].tolist() == [0.5] and cap["YSensor"].tolist() == [0.25]
        assert not any(g.split(" ")[0] in ("OUTPAIR", "INPAIR", "FIRE") for g in fake.got)
        play = next(g for g in fake.got if g.startswith("PLAY"))
        assert play.split(" ")[2] == "move" and play.split(" ")[4:] == [
            "ow0", "ow1", "-200", "32", "200"]
    finally:
        arc.close()
        fake.close()


def test_play_move_refuses_a_non_transit_arm(v3):
    fake, arc = make_arc(igor_v3)
    try:
        arc.start()
        arc.td_xSetOutWavePair(0, "14", "PIDSLoop.0.Setpoint", np.zeros(32),
                               "PIDSLoop.1.Setpoint", np.zeros(32), -200)
        arc.td_xSetInWavePair(0, "14", "XSensor", "YSensor", 32, 200)
        with pytest.raises(ValueError, match="transit only"):
            arc.fire("14", 32)
    finally:
        arc.close()
        fake.close()


def test_play_scanchain_lines_and_scandone(v3):
    fake, arc = make_arc(play_script([("ZSensor", [1, 2]), ("ZSensor", [3, 4])]))
    try:
        arc.start()
        w = arc.router.open_waiter()
        arc.link.send(P.cmd_scanchain(w.uid, "ow0", "ow1", 2, 2, -100, 100, "Amplitude"))
        frames = [w.get(3) for _ in range(3)]
        assert frames == [(P.CAPTURE, "L0 ZSensor 1;2"), (P.CAPTURE, "L1 ZSensor 3;4"),
                          (P.SCANDONE, "2")]
        play = next(g for g in fake.got if g.startswith("PLAY"))
        assert play.split(" ")[2:] == ["scanchain", w.uid, "ow0", "ow1", "2", "2", "-100",
                                       "100", "Amplitude"]
    finally:
        arc.close()
        fake.close()


@pytest.mark.parametrize("why", ["busy", "mark", "fault"])
def test_play_refused_reaches_the_driver_as_err(v3, why):
    def script(line, fake):
        if line.startswith("PLAY"):
            return [f"REPLY {seq_of(line)} err {why}"]
        return igor_v3(line, fake)
    fake, arc = make_arc(script)
    try:
        arc.start()
        w = arc.router.open_waiter()
        arc.link.send(P.cmd_armfire(w.uid, "obj", -500, 64, 500))
        assert w.get(3) == (P.ERR, f"PLAY refused: {why}")
    finally:
        arc.close()
        fake.close()


def test_aborted_and_error_reports_end_the_run(v3):
    def script(line, fake):
        if line.startswith("PLAY"):
            run = line.split(" ")[3]
            return [f"REPLY {seq_of(line)} ok armed {run}", fake.report(run, "aborted")]
        return igor_v3(line, fake)
    fake, arc = make_arc(script)
    try:
        arc.start()
        w = arc.router.open_waiter()
        arc.link.send(P.cmd_armfire(w.uid, "obj", -500, 64, 500))
        assert w.get(3) == (P.ABORTED, "obj")
    finally:
        arc.close()
        fake.close()


def test_short_capture_is_a_known_hole(v3):
    def script(line, fake):
        if line.startswith("PLAY"):
            run = line.split(" ")[3]
            return [f"REPLY {seq_of(line)} ok armed {run}",
                    fake.report(run, "step", "0 1 4"), fake.cap(run, 0, "ZSensor", [1, 2]),
                    fake.report(run, "done")]
        return igor_v3(line, fake)
    fake, arc = make_arc(script)
    try:
        arc.start()
        w = arc.router.open_waiter()
        arc.link.send(P.cmd_armfire(w.uid, "obj", -500, 64, 500))
        w.get(3)
        w.get(3)
        assert arc.router.holes == [f"run {w.uid} step 0 ZSensor: 2 of 4 points"]
    finally:
        arc.close()
        fake.close()


# --------------------------------------------- stamps, ring, fault latch --

def test_tick_feeds_telemetry_and_fault_stamp_latches_once(v3):
    fake, arc = make_arc(igor_v3)
    telem, faults = [], []
    try:
        arc.start()
        arc.on_telem = lambda *v: telem.append(v)
        arc.router.on_fault = faults.append
        fake.push(fake.tick())
        fake.push(fake.tick(fault=1))
        fake.push(fake.report("-", "backup", "50 withdraw", mark=1, fault=1))
        assert wait_for(lambda: len(telem) == 2 and arc.router.last_useq == 3)
        assert telem[0] == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        assert len(faults) == 1 and arc.router.igor_fault and arc.router.mark
    finally:
        arc.close()
        fake.close()


def test_useq_gap_is_logged(v3):
    logs: list[str] = []
    fake = FakeIgor(igor_v3)
    arc = HardwareARC(fake.client, timeout=5.0, log=logs.append)
    try:
        arc.start()
        fake.push("TICK 1 100 0 0 - 1 2 3 4 5 6")
        fake.push("TICK 5 100 0 0 - 1 2 3 4 5 6")
        assert wait_for(lambda: any("useq gap, 1 then 5" in m for m in logs))
    finally:
        arc.close()
        fake.close()


def test_reconnect_fetches_the_missed_reports(v3):
    def script(line, fake):
        if line.startswith("HELLO"):
            return [f"REPLY {seq_of(line)} ok proto=3;ring=1-9;"]
        return igor_v3(line, fake)
    fake = FakeIgor(script)
    link, router = wire_v3.open_v3(fake.client, resume_from_useq=5)
    try:
        link.say_hello()
        fetch = next(g for g in fake.got if g.startswith("FETCH"))
        assert fetch.split(" ")[2] == "6"
    finally:
        router.stop()
        fake.close()


def test_ring_gone_is_logged(v3):
    logs: list[str] = []
    def script(line, fake):
        if line.startswith("HELLO"):
            return [f"REPLY {seq_of(line)} ok proto=3;ring=300-555;"]
        if line.startswith("FETCH"):
            return [f"REPLY {seq_of(line)} err gone 300"]
        return igor_v3(line, fake)
    fake = FakeIgor(script)
    link, router = wire_v3.open_v3(fake.client, log=logs.append, resume_from_useq=10)
    try:
        link.say_hello()
        assert any("gone" in m for m in logs)
    finally:
        router.stop()
        fake.close()


def test_no_fetch_on_a_first_connect(v3):
    fake = FakeIgor(igor_v3)
    link, router = wire_v3.open_v3(fake.client)
    try:
        link.say_hello()
        assert "FETCH" not in fake.verbs()
    finally:
        router.stop()
        fake.close()


def test_hub_latches_the_stamp_as_the_park_igor_fault():
    from afm_litho.service.hub import Hub
    hub = Hub.__new__(Hub)
    hub.__dict__["host"] = __import__("types").SimpleNamespace(fault=None)  # the latch lives on HostState
    hub.__dict__["_loop"] = None
    hub._log_sync = lambda msg: None
    hub.latch_igor_fault("x")
    assert hub.fault["reason"] == "park_unverified" and "fault stamp" in hub.fault["detail"]


# -------------------------------------------------------------- make-safe --

class FakeCOM:
    """COM Igor: `fail` lines raise from Execute2; `arc_reject` lines run but the ARC
    function returns an error code, which only arc_call reports (Execute2 hides it)."""

    def __init__(self, fail=(), arc_reject=()):
        self.fail = set(fail)
        self.arc_reject = set(arc_reject)
        self.ran: list[str] = []

    def execute(self, cmd, check=True):
        self.ran.append(cmd)
        if cmd in self.fail or "*" in self.fail:
            raise RuntimeError(f"Igor error on {cmd}")

    def arc_call(self, call):
        self.execute(call)
        if call in self.arc_reject:
            raise RuntimeError(f"ARC error 1001 from {call}")
        return 0


def test_park_words_carried_as_v3(v3):
    """AB's park (safe_park) confirms, reads Igor's fault and clears it; v3 only carries it."""
    def script(line, fake):
        if line.startswith("READ") and line.endswith("flag abortFault"):
            return [f"REPLY {seq_of(line)} ok 1"]
        if line.startswith("READ") and " str " in line:
            return [f"REPLY {seq_of(line)} ok PIDSLoop.2.OutputChannel=Height;"]
        return igor_v3(line, fake)
    fake, arc = make_arc(script)
    try:
        arc.start()
        bridge_router.send_safe_park(arc.link, None)
        arc.confirm_retract()
        assert arc.read_abort_fault() is True
        arc.router.igor_fault = True
        arc.clear_fault()
        assert wait_for(lambda: not arc.router.igor_fault)      # cleared on Igor's ok
        w = arc.router.open_waiter()
        arc.link.send(P.cmd_getstr(w.uid, "PIDSLoop.2.OutputChannel"))
        assert w.get(3) == (P.XPT, "PIDSLoop.2.OutputChannel=Height;")
        assert wait_for(lambda: "CLEARFAULT" in fake.verbs())
        lines = [" ".join(g.split(" ")[:1] + g.split(" ")[2:]) for g in fake.got[2:]]
        assert lines == ["MAKESAFE", "CONFIRMSAFE", "READ flag abortFault", "CLEARFAULT",
                         "READ str PIDSLoop.2.OutputChannel"]
    finally:
        arc.close()
        fake.close()


def test_park_words_to_another_version_stay_v2(v3):
    def script(line, fake):
        if line.startswith("HELLO"):
            return [f"REPLY {seq_of(line)} err mismatch proto=4;"]
        if line.startswith("STARTUP"):
            return [f"STARTUP {seq_of(line)} proto=2"]
        if line.startswith("ABORTFAULT"):
            return [f"VALUE {seq_of(line)} 0"]
        return []
    fake, arc = make_arc(script)
    try:
        arc.start()
        bridge_router.send_safe_park(arc.link, None)
        arc.confirm_retract()
        assert arc.read_abort_fault() is False
        arc.clear_fault()
        assert wait_for(lambda: len(fake.got) == 6)
        assert [g.split(" ")[0] for g in fake.got[2:]] == [
            "ABORT", "ABORTCONFIRM", "ABORTFAULT", "CLEARFAULT"]
    finally:
        arc.close()
        fake.close()


def test_abort_before_hello_is_sent_at_once_as_abort(v3):
    fake, arc = make_arc(lambda line, f: [])          # Igor answers nothing at all
    try:
        t0 = time.monotonic()
        arc.link.send(P.cmd_abort())
        assert time.monotonic() - t0 < 0.5
        assert wait_for(lambda: fake.verbs() == ["ABORT"])
    finally:
        arc.close()
        fake.close()


def test_abort_after_hello_is_makesafe_without_waiting(v3):
    replies = {"HELLO", "STARTUP"}
    fake, arc = make_arc(lambda line, f: igor_v3(line, f) if line.split(" ")[0] in replies
                         else [])
    try:
        arc.start()
        t0 = time.monotonic()
        arc.link.send(P.cmd_abort())
        assert time.monotonic() - t0 < 0.5
        assert wait_for(lambda: fake.verbs() == ["HELLO", "STARTUP", "MAKESAFE"])
    finally:
        arc.close()
        fake.close()


def test_old_ipf_abort_is_the_v2_abort(v3):
    def script(line, fake):
        if line.startswith("HELLO"):
            return ["ERR - unknown verb: HELLO"]
        if line.startswith("STARTUP"):
            return [f"STARTUP {seq_of(line)} proto=2"]
        return []
    fake, arc = make_arc(script)
    try:
        arc.start()
        bridge_router.send_safe_park(arc.link, None)
        assert wait_for(lambda: fake.verbs()[-1:] == ["ABORT"])
        assert "READ" not in fake.verbs()              # no v3 confirm on a v2 .ipf
    finally:
        arc.close()
        fake.close()


# ------------------------------- raw COM fallback (v3 only, contract section 12) --

RAW = ["td_Stop()", 'td_WriteValue("Output.A",0)',
       'ir_SetPISLoop(2,"Always,Never","Deflection",-10,0,1000,0,"Height",0,150)']


class SentLink:
    def __init__(self):
        self.sent: list[str] = []

    def send(self, line):
        self.sent.append(line)


def test_v2_has_no_raw_fallback():
    """On v2, today's order exactly -- COM AFM_SafePark(), then the socket ABORT."""
    com, link = FakeCOM(fail={"AFM_SafePark()"}), SentLink()
    assert bridge_router.send_safe_park(link, com) == "tcp"
    assert com.ran == ["AFM_SafePark()"] and link.sent == ["ABORT"]


def test_safepark_ok_sends_nothing_else():
    com, link = FakeCOM(), SentLink()
    assert bridge_router.send_safe_park(link, com) == "com"
    assert com.ran == ["AFM_SafePark()"] and link.sent == []


def _v3_after_hello():
    fake, arc = make_arc(igor_v3)
    arc.start()
    return fake, arc


def test_v3_raw_lines_when_safepark_fails(v3):
    fake, arc = _v3_after_hello()
    try:
        com = FakeCOM(fail={"AFM_SafePark()"})
        assert bridge_router.send_safe_park(arc.link, com) == "com_raw"
        assert com.ran == ["AFM_SafePark()"] + RAW
        assert "MAKESAFE" not in fake.verbs()
    finally:
        arc.close()
        fake.close()


def test_v3_raw_rearm_rejected_by_arc_still_sends_makesafe(v3):
    """Reviewer's repro (must-fix 1): Execute2 succeeds but the ARC rejects the re-arm.
    That counts as a failure, so the socket MAKESAFE still goes out."""
    fake, arc = _v3_after_hello()
    try:
        com = FakeCOM(fail={"AFM_SafePark()"}, arc_reject={RAW[2]})
        assert bridge_router.send_safe_park(arc.link, com) == "tcp"
        assert com.ran == ["AFM_SafePark()"] + RAW
        assert wait_for(lambda: "MAKESAFE" in fake.verbs())
    finally:
        arc.close()
        fake.close()


def test_v3_raw_bias_zero_failing_still_sends_makesafe(v3):
    fake, arc = _v3_after_hello()
    try:
        com = FakeCOM(fail={"AFM_SafePark()", RAW[1]})
        assert bridge_router.send_safe_park(arc.link, com) == "tcp"
        assert com.ran == ["AFM_SafePark()"] + RAW       # every line tried, in order
        assert wait_for(lambda: "MAKESAFE" in fake.verbs())
    finally:
        arc.close()
        fake.close()


def test_v3_raw_counts_when_only_td_stop_fails(v3):
    fake, arc = _v3_after_hello()
    try:
        com = FakeCOM(fail={"AFM_SafePark()"}, arc_reject={RAW[0]})
        assert bridge_router.send_safe_park(arc.link, com) == "com_raw"
    finally:
        arc.close()
        fake.close()


def test_v3_raw_skips_td_stop_while_an_abort_runs(v3):
    """The raw fallback does not undefine a retract already under way."""
    fake, arc = _v3_after_hello()
    try:
        com = FakeCOM(fail={"AFM_SafePark()"})
        bridge_router.send_safe_park(arc.link, com)      # first abort: td_Stop runs
        com.ran.clear()
        bridge_router.send_safe_park(arc.link, com)      # repeat while it retracts
        assert com.ran == ["AFM_SafePark()"] + RAW[1:]
        arc.confirm_retract()                            # the park confirmed
        com.ran.clear()
        bridge_router.send_safe_park(arc.link, com)
        assert com.ran == ["AFM_SafePark()"] + RAW
    finally:
        arc.close()
        fake.close()


def test_v3_raw_skips_td_stop_on_igor_mark(v3):
    fake, arc = _v3_after_hello()
    try:
        fake.push(fake.tick(mark=1))                     # Igor: a make-safe unconfirmed
        assert wait_for(lambda: arc.router.mark)
        com = FakeCOM(fail={"AFM_SafePark()"})
        bridge_router.send_safe_park(arc.link, com)
        assert com.ran == ["AFM_SafePark()"] + RAW[1:]
    finally:
        arc.close()
        fake.close()


def test_v3_all_com_dead_falls_to_makesafe(v3):
    fake, arc = _v3_after_hello()
    try:
        com = FakeCOM(fail={"*"})
        assert bridge_router.send_safe_park(arc.link, com) == "tcp"
        assert wait_for(lambda: "MAKESAFE" in fake.verbs())
    finally:
        arc.close()
        fake.close()


# ------------------------------------------------- review must-fixes 5-7 --

def test_stale_fault_tick_after_clearfault_does_not_relatch(v3):
    """Review item 5: a TICK stamped fault=1 that Igor sent before it acted on CLEARFAULT
    must not latch hub.fault again; the flag clears only on Igor's ok."""
    held: list[str] = []

    def script(line, fake):
        if line.startswith("CLEARFAULT"):
            held.append(seq_of(line))
            return []                                  # Igor has not acted yet
        return igor_v3(line, fake)
    fake, arc = make_arc(script)
    faults: list[str] = []
    try:
        arc.start()
        arc.router.on_fault = faults.append
        fake.push(fake.tick(fault=1))
        assert wait_for(lambda: len(faults) == 1)
        faults.clear()                                 # the operator clears hub.fault
        arc.clear_fault()
        assert wait_for(lambda: held)
        fake.push(fake.tick(fault=1))                  # in flight before Igor cleared
        assert wait_for(lambda: arc.router.last_useq == 2)
        assert faults == [] and arc.router.igor_fault
        fake.push(f"REPLY {held[0]} ok fault=0")
        fake.push(fake.tick(fault=0))
        assert wait_for(lambda: not arc.router.igor_fault)
        assert faults == []
    finally:
        arc.close()
        fake.close()


@pytest.mark.parametrize("answer", ["ok nan", "err no such flag", "ok 2"])
def test_fault_flag_read_that_is_not_0_or_1_is_an_error_not_a_fault(v3, answer):
    """Review item 6: `READ flag abortFault` answering nan (or anything but 0/1) raises,
    so the park logs a failed read instead of latching a false fault."""
    def script(line, fake):
        if line.startswith("READ") and " flag " in line:
            return [f"REPLY {seq_of(line)} {answer}"]
        return igor_v3(line, fake)
    fake, arc = make_arc(script)
    try:
        arc.start()
        with pytest.raises(RuntimeError, match="flag read"):
            arc.read_abort_fault()
    finally:
        arc.close()
        fake.close()


def test_hello_fault_field_latches(v3):
    """Review item 7: HELLO's fault=1 latches, and wire_igor_fault (wired before start)
    passes it to hub.fault."""
    from afm_litho.service.hub import Hub

    def script(line, fake):
        if line.startswith("HELLO"):
            return [f"REPLY {seq_of(line)} ok proto=3;mark=0;fault=1;"]
        return igor_v3(line, fake)
    fake, arc = make_arc(script)
    hub = Hub.__new__(Hub)
    hub.__dict__["host"] = __import__("types").SimpleNamespace(fault=None)  # the latch lives on HostState
    hub.__dict__["_loop"] = None
    hub._log_sync = lambda msg: None
    try:
        hub.wire_igor_fault(arc.router)
        arc.start()
        assert arc.router.igor_fault and hub.fault["reason"] == "park_unverified"
    finally:
        arc.close()
        fake.close()


def test_fault_seen_before_wiring_latches_at_bind():
    from afm_litho.service.hub import Hub
    fake = FakeIgor(igor_v3)
    with wire.use_version(3):
        link, router = wire_v3.open_v3(fake.client)
    hub = Hub.__new__(Hub)
    hub.__dict__["host"] = __import__("types").SimpleNamespace(fault=None)  # the latch lives on HostState
    hub.__dict__["_loop"] = None
    hub._log_sync = lambda msg: None
    try:
        fake.push(fake.tick(fault=1))                  # no on_fault wired yet
        assert wait_for(lambda: router.igor_fault)
        assert hub.fault is None
        hub.wire_igor_fault(router)
        assert hub.fault["reason"] == "park_unverified"
    finally:
        router.stop()
        fake.close()


def test_load_of_no_points_raises_and_sends_nothing(v3):
    fake, arc = make_arc(igor_v3)
    try:
        arc.start()
        with pytest.raises(ValueError, match="no points"):
            arc._load_wave(np.zeros(0))
        assert "LOAD" not in fake.verbs()
    finally:
        arc.close()
        fake.close()
