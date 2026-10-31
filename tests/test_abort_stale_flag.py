"""A stale root:AFM_abortStopped must never keep an abort from stopping a pass (Cypher rig,
2026-10-28: after service restarts the flag stayed 1, the Igor abort panel skipped td_Stop, and
the write kept playing). AFM_Abort calls td_Stop when the flag is clear OR a pass armed since
the last td_Stop (AFM_PassMayPlay); only a repeat abort with nothing armed since skips it. At
connect, with no fault latched here and Height under 1 V, the service sends CLEARFAULT, and a
fault clear that could not reach Igor is sent then too."""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

from afm_litho import clock
from afm_litho.goldens import expander as X
from afm_litho.hardware import wire
from afm_litho.hardware.hardware_arc import HardwareARC
from afm_litho.hardware.igor_stub import _XY_AXIS, FakeIgorBridge
from afm_litho.service.hub import Hub

IPF = Path(__file__).resolve().parents[1] / "igor" / "afm_litho_run.ipf"


class _Link:
    def __init__(self):
        self.sent: list[str] = []

    def send(self, line: str) -> None:
        self.sent.append(line)


@pytest.fixture
def vclock():
    with clock.using(clock.TESTING):
        yield clock


def _write_playing(stub: FakeIgorBridge, link: _Link) -> None:
    """A single-shot litho write armed and fired (ARMFIRE), left playing on the virtual clock."""
    ramp = ";".join(str(0.01 * i) for i in range(64))
    for name in ("XLitho", "YLitho", "VLitho"):
        stub._dispatch(f"WAVE {name} {ramp}", link)
    stub._dispatch("ARMFIRE u1 obj -500 64 500", link)


def _xy_playing(stub: FakeIgorBridge) -> bool:
    return bool(set(_XY_AXIS) & set(stub.safety_sim._plays))


# ---------------------------------------------------------------- the abort, wire v2 --

def test_a_stale_flag_does_not_stop_an_abort_from_stopping_a_playing_write(vclock):
    stub = FakeIgorBridge({"Height": 0.0}, safety_sim=True)
    link = _Link()
    try:
        stub.abort_stopped = True                   # left from an earlier, unconfirmed park
        _write_playing(stub, link)
        assert _xy_playing(stub)
        stub._dispatch("ABORT", link)
        assert stub.td_stops == 1                   # td_Stop ran despite the stale flag
        assert not _xy_playing(stub)                # and the write stopped
        assert stub.abort_stopped and not stub.pass_armed
    finally:
        stub.stop()


def test_a_repeat_abort_while_retracting_with_nothing_playing_skips_td_stop(vclock):
    stub = FakeIgorBridge({"Height": 0.0}, safety_sim=True)
    link = _Link()
    try:
        _write_playing(stub, link)
        stub._dispatch("ABORT", link)
        stub._dispatch("ABORT", link)               # the retract is still running
        assert stub.td_stops == 1
    finally:
        stub.stop()


def test_a_pass_fired_after_an_abort_is_stopped_by_the_next_abort(vclock):
    stub = FakeIgorBridge({"Height": 0.0}, safety_sim=True)
    link = _Link()
    try:
        stub._dispatch("ABORT", link)               # flag set, never confirmed
        _write_playing(stub, link)
        stub._dispatch("ABORT", link)
        assert stub.td_stops == 2 and not _xy_playing(stub)
    finally:
        stub.stop()


# ---------------------------------------------------------------- the abort, wire v3 --

def test_v3_makesafe_stops_a_playing_run_despite_a_stale_flag(vclock):
    stub = FakeIgorBridge({"MicroscopeID": 4.0, "Height": 0.0}, safety_sim=True)
    link = _Link()
    try:
        stub._dispatch("HELLO 1 3 test", link)
        ramp = ";".join(str(0.01 * i) for i in range(64))
        for name in ("XLitho", "YLitho", "VLitho"):
            stub._dispatch(f"LOAD 2 {name} 64 {ramp}", link)
        stub._dispatch("PLAY 3 litho1 r1 obj XLitho YLitho VLitho -500 64 500", link)
        assert any(x.startswith("REPLY 3 ok armed") for x in link.sent)
        stub.abort_stopped = True                   # stale, as after a lost CLEARFAULT
        stub._dispatch("MAKESAFE 4", link)
        assert stub.td_stops == 1 and not _xy_playing(stub)
        assert stub.v3.igor_calls.count("td_Stop()") == 1
        stub._dispatch("MAKESAFE 5", link)          # the repeat, nothing armed since
        assert stub.td_stops == 1
    finally:
        stub.stop()


# ---------------------------------------------------------------- the .ipf and the expander --

def _ipf() -> str:
    return IPF.read_bytes().replace(b"\r\n", b"\n").decode("latin-1")


def test_the_ipf_abort_stops_when_a_pass_may_play():
    src = _ipf()
    abort = re.search(r"^Function AFM_Abort\(\)\n(.*?)^End", src, re.M | re.S).group(1)
    assert "if(!NVAR_Exists(stopped) || !stopped || AFM_PassMayPlay())" in abort
    assert re.search(r"td_Stop\(\)\n\s*AFM_StopRan\(\)", abort)
    ran = re.search(r"^Function AFM_StopRan\(\)\n(.*?)^End", src, re.M | re.S).group(1)
    for line in ("root:AFM_abortStopped = 1", "root:AFM_passArmed = 0",
                 "root:AFM_v3StopSeen = 0"):
        assert line in ran


def _functions(src: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in
            re.finditer(r"^Function(?:/\w+)?\s+(\w+)\(.*?\n(.*?)^End", src, re.M | re.S)}


def test_every_function_that_fires_or_ramps_marks_a_pass_armed():
    """Every function body that fires an event (`"Once")`) or ramps an output (IR_SetRamp)
    sets root:AFM_passArmed, so an abort after it runs td_Stop even with a stale flag."""
    movers = {name: body for name, body in _functions(_ipf()).items()
              if '"Once")' in body or "IR_SetRamp(" in body}
    assert {"AFM_FireEvent", "AFM_ArmAndFireLitho", "AFM_ScanArmLine", "AFM_LithoArmChunk",
            "AFM_RampOutput"} <= set(movers)
    missing = [n for n, b in movers.items() if "Variable/G root:AFM_passArmed = 1" not in b]
    assert not missing, missing


def test_the_expander_runs_td_stop_for_a_fire_after_a_stale_flag():
    exp = X.IgorExpander(microscope_id=4)
    exp.g.update(abortStopped=1)                    # stale
    exp._abort()                                    # nothing armed: skipped, as before
    assert [a for a in exp.out.actuation if a.fn == "td_Stop"] == []
    exp._fire_event("n", "0")
    exp._abort()
    exp._abort()                                    # the repeat during the retract
    assert len([a for a in exp.out.actuation if a.fn == "td_Stop"]) == 1


# ---------------------------------------------------------------- self-heal at connect --

class _Hub:
    """The slice of the hub the connect clear reads."""

    def __init__(self, fault=None):
        self.fault = fault
        self.igor_clear_pending = False
        self._safe_park = {"active": False}
        self.logs: list[str] = []

    def _log_sync(self, msg: str) -> None:
        self.logs.append(msg)

    @property
    def host(self):                # fake: host fields live on the fake itself
        return self


def _wait(pred, timeout=2.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def _connect(stub: FakeIgorBridge) -> HardwareARC:
    return HardwareARC.connect("127.0.0.1", stub.serve_in_thread(), timeout=5.0)


@pytest.mark.parametrize("version", [2, 3])
def test_a_connect_with_no_python_fault_clears_stale_igor_flags(version):
    with wire.use_version(version):
        stub = FakeIgorBridge({"MicroscopeID": 4.0, "Height": 0.0})
        stub.abort_stopped = True
        hw = _connect(stub)
        hub = _Hub()
        try:
            Hub._clear_stale_igor_abort_flags(hub, hw)
            assert _wait(lambda: stub.fault_clears == 1)
            assert not stub.abort_stopped
            assert any("CLEARFAULT sent" in m for m in hub.logs)
        finally:
            hw.close()
            stub.stop()


@pytest.mark.parametrize("version", [2, 3])
def test_a_connect_with_a_python_fault_leaves_igors_flags(version):
    with wire.use_version(version):
        stub = FakeIgorBridge({"MicroscopeID": 4.0, "Height": 0.0})
        stub.abort_stopped = True
        hw = _connect(stub)
        hub = _Hub(fault={"reason": "park_unverified"})
        try:
            Hub._clear_stale_igor_abort_flags(hub, hw)
            time.sleep(0.1)
            assert stub.fault_clears == 0 and stub.abort_stopped
            assert any("a fault is latched" in m for m in hub.logs)
        finally:
            hw.close()
            stub.stop()


@pytest.mark.parametrize("version", [2, 3])
def test_a_connect_while_height_is_up_leaves_igors_backup_alone(version):
    """Height not under 1 V may be a park whose 5 s backup CLEARFAULT would cancel."""
    with wire.use_version(version):
        stub = FakeIgorBridge({"MicroscopeID": 4.0, "Height": 50.0})
        stub.abort_stopped = True
        hw = _connect(stub)
        hub = _Hub()
        try:
            Hub._clear_stale_igor_abort_flags(hub, hw)
            time.sleep(0.1)
            assert stub.fault_clears == 0 and stub.abort_stopped
            assert any("not under 1 V" in m for m in hub.logs)
        finally:
            hw.close()
            stub.stop()


@pytest.mark.parametrize("version", [2, 3])
def test_a_fault_cleared_while_the_bridge_is_down_sends_clearfault_at_the_next_connect(
        version, monkeypatch):
    from afm_litho.service import hub as hub_mod
    from afm_litho.service.routers import instrument
    monkeypatch.setattr(hub_mod, "real_bridge", lambda h: None)
    monkeypatch.setattr(instrument.hub, "fault", {"reason": "park_unverified"})
    monkeypatch.setattr(instrument.hub.host, "igor_clear_pending", False)
    logs: list[str] = []
    monkeypatch.setattr(instrument.hub, "_log_sync", logs.append)
    reply = instrument.post_fault_clear({"confirm": True})
    assert reply["igor_cleared"] is None and "next connect" in reply["igor_detail"]
    assert instrument.hub.host.igor_clear_pending is True
    with wire.use_version(version):
        stub = FakeIgorBridge({"MicroscopeID": 4.0, "Height": 0.0})
        stub.abort_stopped = True                   # the flag the lost clear should have reset
        hw = _connect(stub)
        try:
            Hub._clear_stale_igor_abort_flags(instrument.hub, hw)
            assert _wait(lambda: stub.fault_clears == 1)
            assert not stub.abort_stopped
            assert instrument.hub.host.igor_clear_pending is False
            assert any("while the bridge was down" in m for m in logs)
        finally:
            hw.close()
            stub.stop()
