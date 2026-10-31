"""Verb-level conformance of wire v3: every test cites the contract section it checks.

Runs against any target in conftest (`igor`): today the reference model, later the fake Igor of
slice W3. Raw lines only; the Python client is not involved.
"""

from __future__ import annotations

import re

import pytest

from . import contract as C
from .target import LineSession, require

SIX = "0;0;0;0;0;0;0;0"


def _lines(igor):
    return list(igor.igor_lines) if hasattr(igor, "igor_lines") else None


def _play(s, kind="move", run="r1"):
    args = {"move": (run, "XW", "YW", -200, 256, 200),
            "litho1": (run, "w", "XW", "YW", "VW", -500, 32, 500)}[kind]
    return s.ask("PLAY", kind, *args)


# ---- HELLO and version rules (section 2a, section 7) ---------------------------------------

def test_hello_reply_fields(igor):
    s = LineSession(igor.port)
    r = s.hello()
    kv = C.parse_kv(r.body)
    assert r.ok and kv["proto"] == "3"
    assert re.fullmatch(r"[0-9a-f]{16}", kv["sha"]), kv
    for key in ("ipf", "MicroscopeID", "clock", "ring", "mark", "fault"):
        assert key in kv, f"HELLO reply lacks {key}: {r.body}"
    assert re.fullmatch(r"\d+-\d+", kv["ring"])


def test_hello_wrong_proto_is_mismatch_with_same_body(igor):
    s = LineSession(igor.port)
    r = s.ask("HELLO", 2, "old")
    assert not r.ok and r.body.startswith("mismatch")
    assert "ipf=" in r.body and "sha=" in r.body


def test_mismatch_refuses_load_play_set_but_keeps_reads_and_makesafe(igor):
    s = LineSession(igor.port)
    s.ask("HELLO", 2, "old")
    for verb, args in (("LOAD", ("w", 2, "1;2")), ("PLAY", ("move", "r1", "a", "b", -1, 8, 1)),
                       ("SET", ("value", "Output.A", 0))):
        assert not s.ask(verb, *args).ok, f"{verb} must be refused after a mismatch"
    assert s.ask("READ", "value", "Height").ok
    assert s.ask("STATUS").ok
    assert s.ask("MAKESAFE").ok


def test_no_hello_means_no_rights_but_the_safe_set_works(igor, session):
    for verb, args in (("LOAD", ("w", 1, "1")), ("PLAY", ("move", "r1", "a", "b", -1, 8, 1)),
                       ("SET", ("value", "Output.A", 0)), ("CLEARFAULT", ())):
        assert not session.ask(verb, *args).ok, f"{verb} without HELLO must be refused"
    assert session.ask("READ", "value", "Height").ok          # section 2a read-only
    assert session.ask("STATUS").ok
    assert session.ask("MAKESAFE").ok                          # section 7 safe set


def test_v2_safe_set_still_answers(igor, session):
    session.raw("STARTUP u1 3", "GETVALUE u2 Height")
    got = []
    for _ in range(40):                       # v2 TELEM frames may be interleaved
        got.append(session.lines.get(timeout=2))
        if any(g.startswith("STARTUP") for g in got) and any(g.startswith("VALUE") for g in got):
            return
    raise AssertionError(got[-5:])


def test_second_connection_is_not_the_host(igor, host):
    require(igor, "concurrent_connections")
    other = LineSession(igor.port)
    assert not other.ask("SET", "value", "Output.A", 0).ok     # section 1: one host


# ---- reads (section 2a) --------------------------------------------------------------------

def test_status_fields(igor, session):
    kv = C.parse_kv(session.ask("STATUS").body)
    for key in ("pass", "run", "step", "mark", "fault", "tipdown", "loop2", "sock_dead_s",
                "ring", "tick"):
        assert key in kv, f"STATUS lacks {key}"
    # td_Stop leaves loop 2 undefined and the fake then says "none" (section 11, W3)
    assert kv["loop2"] == "none" or len(kv["loop2"].split(",")) == 3


def test_read_value_gv_xpt(igor, session):
    assert session.ask("READ", "value", "Height").ok
    gv = session.ask("READ", "gv", "abortFault")
    assert gv.ok and (gv.body == "nan" or float(gv.body) is not None)
    assert session.ask("READ", "xpt", "Crosspoint").ok


# ---- ack-paced LOAD (section 2b, 2d) -------------------------------------------------------

def test_load_replies_after_each_piece_with_progress(host):
    first, rest = [0.5] * C.PIECE_POINTS, [0.25] * 100
    total = len(first) + len(rest)
    r1 = host.ask("LOAD", "XLitho", total, C.wave_body(first))
    assert r1.ok and r1.body == f"XLitho {len(first)}/{total}"
    r2 = host.ask("LOADAPP", "XLitho", C.wave_body(rest))
    assert r2.ok and r2.body == f"XLitho {total}/{total}"


def test_load_reply_is_one_per_piece_and_in_order(host):
    seqs = []
    for k in range(3):
        verb = "LOAD" if k == 0 else "LOADAPP"
        args = ("w", 30, C.wave_body([1] * 10)) if k == 0 else ("w", C.wave_body([1] * 10))
        r = host.ask(verb, *args)
        seqs.append(r.body)
    assert seqs == ["w 10/30", "w 20/30", "w 30/30"]


# ---- PLAY, busy, mark, fault (section 2b) -------------------------------------------------

def test_play_arms_then_reports_and_done(igor, host):
    require(igor, "advance")
    r = _play(host)
    assert r.ok and r.body == "armed r1"
    igor.advance(10)
    events = [u.event for u in host.reports()]
    assert events[0] == "armed" and "done" in events


def test_play_while_active_is_busy(host):
    _play(host)
    r = _play(host, run="r2")
    assert not r.ok and r.body.startswith("busy")


def test_play_refused_while_mark_set(host):
    host.ask("MAKESAFE")
    r = _play(host)
    assert not r.ok and r.body.startswith(("mark", "fault"))


def test_play_refused_while_fault_set_and_set_is_not(igor, host):
    require(igor, "set_height")
    igor.set_height(100.0)
    host.ask("MAKESAFE")
    assert host.ask("VENDORWITHDRAW").ok
    r = _play(host)
    assert not r.ok and r.body.startswith(("mark", "fault"))
    # section 2b: SET is never refused for mark or fault (operator keeps the loops)
    assert host.ask("SET", "loop", 2, "Deflection", 0.5, 0, 1000, "Height").ok
    assert host.ask("SET", "value", "Output.A", 0).ok


def test_clearfault_clears_both_flags_with_no_height_check_and_play_is_accepted(igor, host):
    # Section 12: the banner CLEARFAULT zeroes the flags with no Height check and PLAY
    # is then accepted. Height stays high here, so a check would have refused it.
    require(igor, "set_height")
    igor.set_height(100.0)
    host.ask("MAKESAFE")
    host.ask("VENDORWITHDRAW")
    assert not _play(host).ok                       # refused while the flags are set
    r = host.ask("CLEARFAULT")
    assert r.ok and r.body == "fault=0"
    assert host.ask("READ", "flag", "abortFault").body == "0"
    kv = C.parse_kv(host.ask("STATUS").body)
    assert kv["mark"] == "0" and kv["fault"] == "0"
    assert _play(host).ok


def test_clearfault_is_host_only(igor, session):
    assert not session.ask("CLEARFAULT").ok


def test_read_flag_abortfault_is_read_only_zero_or_one(igor, host):
    require(igor, "set_height")
    assert host.ask("READ", "flag", "abortFault").body == "0"
    igor.set_height(100.0)
    host.ask("MAKESAFE")
    host.ask("VENDORWITHDRAW")
    assert host.ask("READ", "flag", "abortFault").body == "1"
    assert not host.ask("READ", "flag", "noSuchFlag").ok     # never a fault by accident
    assert not host.ask("SET", "flag", "abortFault", 0).ok   # no way to write it


# ---- SET echoes (section 2b) ---------------------------------------------------------------

def test_set_loop_echo_has_error_code(host):
    r = host.ask("SET", "loop", 2, "Deflection", 0.5, 0, 1000, "Height")
    assert r.ok and r.body.startswith("loop 2 ") and re.search(r"err=-?\d+$", r.body)


def test_set_mode_and_value_echo(host):
    assert host.ask("SET", "mode", "AC").body.startswith("mode ")
    assert host.ask("SET", "value", "Output.A", 0).body.startswith("value Output.A ")


# ---- stamps, ring and FETCH (section 1, 2c) ------------------------------------------------

def test_every_report_is_stamped_and_useq_has_no_gaps(igor, host):
    require(igor, "advance")
    _play(host)
    igor.advance(10)
    host.ask("MAKESAFE")
    ups = [u for u in host.pump(0.3) + host.seen if u.kind in ("REPORT", "TICK", "CAP")]
    ups = list({u.raw: u for u in ups}.values())
    assert ups
    for u in ups:
        assert None not in (u.tick, u.useq, u.mark)
        assert u.mark in (0, 1)
        if u.kind != "CAP":                   # section 11 point 3: CAP carries `mark` only
            assert u.fault in (0, 1) and u.route in ("sock", "com", "-")
    useqs = sorted(u.useq for u in ups)
    assert useqs == list(range(useqs[0], useqs[0] + len(useqs))), "a gap in useq means loss"


def test_ticks_are_dash_routed_and_paced(igor, host):
    require(igor, "advance")
    igor.advance(2.0)
    ticks = [u for u in host.pump(0.4) if u.kind == "TICK"]
    assert ticks and all(t.route == "-" for t in ticks)
    assert len(ticks[0].body.split()) == 6               # x y z defl sum amp
    assert 5 <= len(ticks) <= 11, "4 Hz idle: about 8 ticks in 2 s"


def test_fetch_resends_from_useq_and_reports_the_ring(igor, host):
    require(igor, "advance")
    _play(host)
    igor.advance(10)
    host.pump(0.3)
    first = host.reports()[0].useq
    host.seen.clear()
    r = host.ask("FETCH", first)
    resent = [u for u in host.seen if u.kind == "REPORT"]
    assert r.ok and re.fullmatch(r"\d+-\d+", r.body)
    events = [u.event for u in resent]
    assert events[0] == "armed" and "done" in events[1:]
    assert all(u.useq >= first for u in resent)


def test_ring_keeps_only_the_last_256_and_fetch_says_gone(igor, host):
    require(igor, "advance")
    for k in range(C.RING_SIZE // 2 + 10):               # armed + done each => > 256 reports
        _play(host, run=f"r{k}")
        igor.advance(5)
    host.pump(0.3)
    # useq is one series shared with TICK and CAP, so the ring's span is wider than 256 numbers;
    # what it holds is the last 256 REPORTs.
    oldest = int(host.ask("STATUS").body.split("ring=")[1].split("-")[0])
    host.seen.clear()
    assert host.ask("FETCH", oldest).ok
    assert len([u for u in host.seen if u.kind == "REPORT"]) == C.RING_SIZE
    r = host.ask("FETCH", 1)
    assert not r.ok and r.body == f"gone {oldest}"


def test_captures_are_not_ringed(igor, host):
    require(igor, "advance")
    _play(host)
    igor.advance(10)
    before = {u.raw for u in host.seen if u.kind == "CAP"}
    host.seen.clear()
    host.ask("FETCH", 1)
    assert not [u for u in host.seen if u.kind == "CAP" and u.raw in before]


def test_reconnect_hello_reports_ring_and_fetch_recovers_the_gap(igor, host):
    require(igor, "advance", "drop_host")
    _play(host)
    igor.drop_host()
    igor.advance(5)
    again = LineSession(igor.port)
    kv = C.parse_kv(again.hello().body)
    oldest, newest = (int(x) for x in kv["ring"].split("-"))
    assert newest >= 2                                   # armed + done, though nobody listened
    again.seen.clear()
    assert again.ask("FETCH", oldest).ok
    assert "done" in [u.event for u in again.seen if u.kind == "REPORT"]


# ---- make-safe (section 5a, 6) -------------------------------------------------------------

def test_makesafe_is_ok_at_once_then_reports_madesafe_and_sets_mark(igor, host):
    r = host.ask("MAKESAFE")
    assert r.ok
    ms = host.reports("madesafe")
    assert ms and ms[0].body.split()[0] == "stop" and ms[0].mark == 1
    assert C.parse_kv(host.ask("STATUS").body)["mark"] == "1"


def test_makesafe_runs_the_contract_sequence_once(igor, host):
    lines = _lines(igor)
    if lines is None:
        require(igor, "igor_lines")
    host.ask("MAKESAFE")
    host.pump(0.2)
    assert list(igor.igor_lines) == list(C.MAKE_SAFE_LINES)
    assert not any("StopPISLoop" in ln or "SetRamp" in ln for ln in igor.igor_lines)


def test_repeat_abort_rearms_retract_but_skips_second_td_stop(igor, host):
    require(igor, "igor_lines", "com_safe_park")
    host.ask("MAKESAFE")
    igor.com_safe_park()
    host.ask("MAKESAFE")
    host.pump(0.2)
    got = list(igor.igor_lines)
    assert got.count(C.TD_STOP) == 1
    assert got.count(C.RETRACT) == 3 and got.count(C.BIAS_ZERO) == 3


def test_com_route_stamps_com_and_runs_same_sequence(igor, host):
    require(igor, "igor_lines", "com_safe_park")
    igor.com_safe_park()
    ms = host.reports("madesafe")
    assert ms and ms[0].route == "com"
    assert list(igor.igor_lines) == list(C.MAKE_SAFE_LINES)


def test_makesafe_jumps_the_queue_and_drops_the_rest(host):
    host.raw(C.down(101, "SET", "value", "Output.A", 1),
             C.down(102, "PLAY", "move", "rq", "XW", "YW", -200, 256, 200),
             C.down(103, "MAKESAFE"))
    r103 = host.reply(103)
    assert r103.ok
    for seq in (101, 102):
        r = host.reply(seq)
        assert not r.ok and r.body.startswith("dropped"), r.raw
    order = [u.seq for u in host.seen if u.kind == "REPLY" and u.seq in (101, 102, 103)]
    assert order[0] == 103, "MAKESAFE must be answered before what was queued ahead of it"


def test_confirmsafe_clears_mark_only_on_igors_own_low_reading(igor, host):
    require(igor, "set_height")
    igor.set_height(100.0)
    host.ask("MAKESAFE")
    r = host.ask("CONFIRMSAFE")
    assert not r.ok and r.body.startswith("height=100")
    assert C.parse_kv(host.ask("STATUS").body)["mark"] == "1"
    igor.set_height(0.4)
    r = host.ask("CONFIRMSAFE")
    assert r.ok and r.body == "height=0.4"
    assert C.parse_kv(host.ask("STATUS").body)["mark"] == "0"


def test_vendorwithdraw_runs_once_per_makesafe_and_sets_fault(igor, host):
    require(igor, "igor_lines", "set_height")
    igor.set_height(100.0)
    host.ask("MAKESAFE")
    assert host.ask("VENDORWITHDRAW").body == "ran"
    assert host.ask("VENDORWITHDRAW").body == "already"
    assert [ln for ln in igor.igor_lines if "StopEngageButton" in ln] == \
        ['DoScanFunc("StopEngageButton")']
    assert C.parse_kv(host.ask("STATUS").body)["fault"] == "1"


def test_backup_clears_mark_at_5s_when_height_is_low(igor, host):
    require(igor, "advance")
    host.ask("MAKESAFE")
    igor.advance(C.BACKUP_S - 0.5)
    assert C.parse_kv(host.ask("STATUS").body)["mark"] == "1"
    igor.advance(1.0)
    b = host.reports("backup")
    assert b and b[0].body.split()[1] == "cleared"
    assert C.parse_kv(host.ask("STATUS").body)["mark"] == "0"


def test_backup_withdraws_and_faults_at_5s_when_height_is_high(igor, host):
    require(igor, "advance", "set_height", "vendor_withdraws")
    igor.set_height(100.0)
    host.ask("MAKESAFE")
    igor.advance(C.BACKUP_S + 0.5)
    b = host.reports("backup")
    assert b and b[0].body.split()[1] == "withdraw"
    kv = C.parse_kv(host.ask("STATUS").body)
    assert kv["fault"] == "1" and kv["mark"] == "1"
    assert igor.vendor_withdraws() == 1


def test_backup_does_nothing_after_a_confirm(igor, host):
    require(igor, "advance")
    host.ask("MAKESAFE")
    assert host.ask("CONFIRMSAFE").ok
    host.seen.clear()
    igor.advance(C.BACKUP_S + 1)
    assert not host.reports("backup")


# ---- the dead socket (section 5b) ---------------------------------------------------------

def _tip_down(host):
    assert host.ask("SET", "loop", 2, "Deflection", 0.5, 0, 1000, "Height").ok


def test_dead_socket_with_tip_down_parks_after_5s(igor, host):
    require(igor, "advance", "drop_host", "igor_lines")
    _tip_down(host)
    igor.drop_host()
    igor.advance(C.SOCK_DEAD_PARK_S - 0.5)
    assert C.TD_STOP not in igor.igor_lines
    igor.advance(1.0)
    assert list(igor.igor_lines)[-3:] == list(C.MAKE_SAFE_LINES)
    again = LineSession(igor.port)
    again.hello()
    why = [u.body.split()[0] for u in again.reports("madesafe")]
    assert why == [] or why[-1] == "socket"      # the ring is fetched, not pushed


def test_dead_socket_with_tip_up_does_not_park(igor, host):
    require(igor, "advance", "drop_host", "igor_lines")
    host.ask("SET", "loop", 2, "Deflection", -10, 0, 1000, "Height")
    igor.drop_host()
    igor.advance(C.SOCK_DEAD_PARK_S + 5)
    assert C.TD_STOP not in igor.igor_lines


def test_dead_socket_during_a_run_lets_it_finish_then_parks(igor, host):
    require(igor, "advance", "drop_host", "igor_lines")
    _tip_down(host)
    _play(host)
    igor.drop_host()
    igor.advance(1.0)
    assert C.TD_STOP not in igor.igor_lines              # run active: no park
    igor.advance(1.5)                                    # run (2 s) is over
    assert C.TD_STOP not in igor.igor_lines              # dead for < 5 s
    igor.advance(C.SOCK_DEAD_PARK_S + 1)
    assert C.TD_STOP in igor.igor_lines
    again = LineSession(igor.port)
    again.hello()
    again.seen.clear()
    again.ask("FETCH", 1)
    events = [u.event for u in again.seen if u.kind == "REPORT"]
    assert "done" in events, "the run's own REPORTs sit in the ring for the reconnect"


# ---- boundary probes: the 1 V and 5 s edges (sections 5a, 5b) -------------------------------

@pytest.mark.parametrize(("volts", "confirms"), [(0.9, True), (1.5, False)])
def test_confirmsafe_edge_at_one_volt(igor, host, volts, confirms):
    require(igor, "set_height")
    igor.set_height(volts)
    host.ask("MAKESAFE")
    assert host.ask("CONFIRMSAFE").ok is confirms
    assert C.parse_kv(host.ask("STATUS").body)["mark"] == ("0" if confirms else "1")


@pytest.mark.parametrize(("volts", "outcome"), [(0.9, "cleared"), (1.5, "withdraw")])
def test_backup_edge_at_one_volt(igor, host, volts, outcome):
    require(igor, "advance", "set_height")
    igor.set_height(volts)
    host.ask("MAKESAFE")
    igor.advance(C.BACKUP_S + 0.2)
    b = host.reports("backup")
    assert b and b[0].body.split()[1] == outcome


def test_dead_socket_park_edge_at_five_seconds(igor, host):
    require(igor, "advance", "drop_host", "igor_lines")
    _tip_down(host)
    igor.drop_host()
    igor.advance(4.8)
    assert C.TD_STOP not in igor.igor_lines
    igor.advance(0.4)                                     # 5.2 s dead
    assert C.TD_STOP in igor.igor_lines
