"""The Output.A loop check: before a write moves to its start, a loop driving the tip bias
is stopped and logged; a failed read never fails the write; the switch turns it off."""

from __future__ import annotations

from pathlib import Path

import pytest

from afm_litho import clock
from afm_litho.hardware import output_a_check
from afm_litho.hardware.igor_stub import FakeIgorBridge
from afm_litho.hardware.lithography_run import LithographyRun

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "SA40656B0002.ibw"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample .ibw not present")

OBJECT = ([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [3.0, 3.0, 3.0])


@pytest.fixture(autouse=True)
def _testing_clock(monkeypatch):
    monkeypatch.setattr(output_a_check, "_warned_unverified", False)
    with clock.using(clock.TESTING):
        yield


def _write(stub_setup=None, cypher=True):
    """One write against the fake Igor. Returns (stub, log lines, result)."""
    stub = FakeIgorBridge.from_ibw(SAMPLE, done_delay=0.0)
    port = stub.serve_in_thread()
    log: list[str] = []
    try:
        stub.values["MicroscopeID"] = 4 if cypher else 1
        if stub_setup:
            stub_setup(stub)
        run = LithographyRun.connect(
            "127.0.0.1", port, settle_move_s=0.0, cmd_gap_s=0.0, xy_withdraw_max_s=0.0,
            xy_withdraw_settle_s=0.0, first_object_settle_s=0.0, log=log.append)
        run.start()
        run.startup["microscope_id"] = stub.values["MicroscopeID"]   # real Igor reports it
        res = run.write_object("obj0", *OBJECT, setpoint=0.4, speed=5.0)
        run.get_value("XSensor")                      # barrier: the stub has seen it all
    finally:
        stub.stop()
    return stub, log, res


def _loop_on_output_a(stub):
    stub.pid_loops[4] = "Cypher.Output.A"


def test_a_loop_driving_output_a_is_stopped_and_logged():
    stub, log, res = _write(_loop_on_output_a)
    assert 4 in stub.stopped_loops
    assert ("Output.A", 0.0) in stub.sets          # the zero is set after the stop
    assert any("loop 4" in m and "Cypher.Output.A" in m for m in log)
    assert not res.aborted


def test_no_loop_on_output_a_means_no_stop():
    stub, log, _ = _write(lambda s: s.pid_loops.update({3: "Height"}))
    assert 3 not in stub.stopped_loops
    assert not any("Output.A loop check" in m for m in log)
    assert any(" PIDSLoop.3.OutputChannel" in t for t in stub.tap.raw)


def test_failed_read_leaves_the_write_running():
    # Status reads back NaN: the check logs once and the write completes untouched.
    stub, log, res = _write(lambda s: s.values.update({"Cypher.PIDSLoop.0.Status": float("nan")}))
    skipped = [m for m in log if "Output.A loop check FAILED" in m]
    assert len(skipped) == 1
    assert res.points == 32 and not res.aborted


def test_failed_string_read_leaves_the_write_running():
    # Loop 5 reads as running but its OutputChannel is unknown to the stub: ERR reply.
    stub, log, res = _write(lambda s: s.values.update({"Cypher.PIDSLoop.5.Status": 1.0}))
    assert len([m for m in log if "Output.A loop check FAILED" in m]) == 1
    assert 5 not in stub.stopped_loops
    assert not res.aborted


def test_switch_off_sends_no_loop_reads(monkeypatch):
    monkeypatch.setattr(output_a_check, "CHECK_OUTPUT_A_LOOPS", False)
    stub, log, _ = _write(_loop_on_output_a)
    assert 4 not in stub.stopped_loops
    assert not any("PIDSLoop.0.Status" in t or "GETSTR" in t for t in stub.tap.raw)


def test_output_a_name_variants():
    assert output_a_check._is_output_a("Output.A")
    assert output_a_check._is_output_a("Cypher.Output.A")
    assert not output_a_check._is_output_a("Output.Z")
    assert not output_a_check._is_output_a("Height")


def test_cypher_reads_status_with_the_cypher_prefix():
    stub, _, _ = _write()
    assert any(t.endswith(" Cypher.PIDSLoop.0.Status") for t in stub.tap.raw)
    assert not any(t.endswith(" PIDSLoop.0.Status") for t in stub.tap.raw)


def test_mfp3d_reads_the_bare_name_and_logs_it_unverified():
    stub, log, _ = _write(_loop_on_output_a, cypher=False)
    assert any(t.endswith(" PIDSLoop.0.Status") for t in stub.tap.raw)
    assert any("unverified" in m for m in log)
    assert 4 in stub.stopped_loops


def _connected_run(cypher=True):
    stub = FakeIgorBridge.from_ibw(SAMPLE, done_delay=0.0)
    stub.values["MicroscopeID"] = 4 if cypher else 1
    run = LithographyRun.connect("127.0.0.1", stub.serve_in_thread(), cmd_gap_s=0.0)
    run.start()
    return stub, run


def test_read_timeout_skips_the_check(monkeypatch):
    monkeypatch.setattr(output_a_check, "_READ_TIMEOUT_S", 0.2)
    stub, run = _connected_run()
    stub.stall_getvalue = True                       # Igor answers no read
    log: list[str] = []
    try:
        output_a_check.stop_loops_driving_output_a(run, log.append, True)
    finally:
        stub.stop()
    assert len([m for m in log if "Output.A loop check FAILED" in m]) == 1
    assert stub.stopped_loops == []


def test_abort_during_the_check_stops_it_at_the_next_loop():
    stub, run = _connected_run()
    calls = []

    def abort_after_first_loop():
        calls.append(1)
        return len(calls) > 1

    try:
        output_a_check.stop_loops_driving_output_a(run, lambda m: None, True,
                                                   abort_after_first_loop)
        run.get_value("XSensor")
    finally:
        stub.stop()
    reads = [t for t in stub.tap.raw if "PIDSLoop" in t and "Status" in t]
    assert len(reads) == 1


def test_abort_sends_no_loop_reads():
    stub, run = _connected_run()
    try:
        run.abort()
        run.get_value("XSensor")
    finally:
        stub.stop()
    assert not any("PIDSLoop" in t or "GETSTR" in t for t in stub.tap.raw)


def test_only_the_write_start_calls_the_check():
    src = Path(__file__).resolve().parents[1] / "src" / "afm_litho"
    users = [p.name for p in src.rglob("*.py")
             if "stop_loops_driving_output_a(" in p.read_text(encoding="utf-8")
             and p.name != "output_a_check.py"]
    assert users == ["lithography_run.py"]


def test_unverified_note_is_logged_once_per_session():
    _, log1, _ = _write(cypher=False)
    _, log2, _ = _write(cypher=False)
    assert len([m for m in log1 + log2 if "unverified" in m]) == 1


def test_a_failed_string_read_is_named_an_exception_by_igor():
    """The .ipf (and the stub) answer a td_ReadString failure with ERR <uid> exception: ...,
    and the check reports it as FAILED, not as a pass."""
    stub, log, res = _write(lambda s: s.values.update({"Cypher.PIDSLoop.5.Status": 1.0}))
    failed = [m for m in log if "Output.A loop check FAILED" in m]
    assert len(failed) == 1 and "exception: td_ReadString(PIDSLoop.5.OutputChannel)" in failed[0]
    assert 5 not in stub.stopped_loops and not res.aborted


def test_the_error_text_from_an_older_ipf_is_a_failed_read_not_a_channel():
    """An older .ipf replied XPT <uid> <name>=*ERROR*; that is a failed read, never a channel
    name that happens not to be Output.A."""
    def setup(s):
        s.values["Cypher.PIDSLoop.5.Status"] = 1.0
        s.getstr_passes_error_text = True
    stub, log, res = _write(setup)
    failed = [m for m in log if "Output.A loop check FAILED" in m]
    assert len(failed) == 1 and "*ERROR*" in failed[0]
    assert not res.aborted


def test_the_check_returns_false_on_a_failed_read_and_true_otherwise():
    class _Run:
        def __init__(self, reply):
            self.reply = reply
            self.link = self
            self.router = self

        def open_waiter(self):
            return self

        uid = "u1"

        def send(self, line):
            pass

        def get(self, timeout):
            return self.reply

        def release(self, uid):
            pass

    from afm_litho.hardware import bridge_protocol as P
    log: list[str] = []
    assert output_a_check.stop_loops_driving_output_a(_Run((P.VALUE, "0")), log.append, True)
    assert not output_a_check.stop_loops_driving_output_a(
        _Run((P.ERR, "exception: x")), log.append, True)
    assert any("FAILED" in m for m in log)
