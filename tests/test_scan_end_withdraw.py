"""A scan lifts the tip when it ends on its own: the plain routine withdraw a write ends with
(loop 2 re-armed, no td_Stop, no park). A Stop or a park keeps its own route, and a continuous
run withdraws only when its loop ends."""

import asyncio

import pytest
from golden.recordings import recorded

from afm_litho.instrument import asylum as A
from afm_litho.service.host import scan_pump as S
from afm_litho.service.passes import ScanConfig

WITHDRAW = 'ir_SetPISLoop(2,"Always,Never","Deflection",-10,0,1000,0,"Height",0,150)'


class _Arc:
    is_cypher = True
    timeout = 5.0


@pytest.fixture
def pump(tmp_path, monkeypatch):
    from afm_litho.service.sessions import SessionStore
    hub = S.hub
    monkeypatch.setattr(hub, "store", SessionStore(tmp_path / "sessions.jsonl"))
    monkeypatch.setattr(hub.host, "hardware_arc", _Arc())
    monkeypatch.setattr(hub.host, "hw_cal", {})
    monkeypatch.setattr(hub.host, "scan_repeat", {"enabled": False})
    monkeypatch.setattr(hub.host, "_hw_withdraw_after", False)
    seen = {"withdraws": 0, "parks": []}

    async def no_cal(source):
        return None

    def withdraw():
        seen["withdraws"] += 1
        return {"verified": True}

    def park(h, reason, how=None, **kwargs):
        # The real one returns a coroutine for the full park and None for the watchdog's.
        seen["parks"].append(reason)
        return asyncio.sleep(0) if how is None else None

    monkeypatch.setattr(hub, "refresh_xy_cal", no_cal)
    monkeypatch.setattr(A.AsylumInstrument, "_reconcile_mode", lambda *a, **k: None)
    monkeypatch.setattr(A.AsylumInstrument, "_refresh_xy_cal", lambda *a, **k: None)
    monkeypatch.setattr(S, "_hardware_withdraw", withdraw)
    monkeypatch.setattr(S, "request_park", park)

    def run(fake):
        monkeypatch.setattr(A, "run_hw_scan_chained", fake)
        hub.host._hw_abort.clear()
        hub.host.engaged = True
        try:
            asyncio.run(S._hw_scan_pump(ScanConfig(size_um=1.0, pixels=8, lines=2)))
        finally:
            hub.host._hw_abort.clear()
        return seen

    return run


def _summary(**extra):
    return {"lines": 0, "pixels": 8, "capture": {}, "line_indices": [], "aborted": False,
            "error": None, **extra}


async def _finished(*args, **kwargs):
    return _summary()


def test_a_scan_that_finishes_withdraws_once(pump):
    seen = pump(_finished)
    assert seen["withdraws"] == 1 and seen["parks"] == []
    assert S.hub.host.engaged is False


def test_a_chained_error_withdraws(pump):
    async def errored(*args, **kwargs):
        return _summary(aborted=True, error="boom")

    assert pump(errored)["withdraws"] == 1


def test_a_raised_failure_withdraws(pump):
    async def raised(*args, **kwargs):
        raise RuntimeError("boom")

    assert pump(raised)["withdraws"] == 1


def test_a_stop_parks_by_its_own_route_and_adds_no_second_withdraw(pump):
    async def stopped(arc, config, cal, sink, **kwargs):
        S.hub.host._hw_abort.set()
        S.hub.host._hw_withdraw_after = True          # scan Stop: PARK_DEFERRED
        S.hub.host._hw_withdraw_epoch = S.hub.host._scan_epoch
        return _summary(aborted=True)

    seen = pump(stopped)
    assert seen["withdraws"] == 0 and seen["parks"] == ["scan stopped"]


def test_an_abort_mid_scan_adds_no_withdraw(pump):
    async def aborted(arc, config, cal, sink, **kwargs):
        S.hub.host._hw_abort.set()                    # the park has signalled the pass to stop
        return _summary(aborted=True)

    seen = pump(aborted)
    assert seen["withdraws"] == 0 and seen["parks"] == []


def test_a_newer_pass_that_took_the_tip_is_not_withdrawn(pump):
    async def superseded(arc, config, cal, sink, **kwargs):
        S.hub.host._scan_epoch += 1
        return _summary()

    assert pump(superseded)["withdraws"] == 0


def test_continuous_frames_do_not_withdraw_until_the_loop_ends(pump):
    S.hub.host.scan_repeat["enabled"] = True
    assert pump(_finished)["withdraws"] == 0
    assert pump(_finished)["withdraws"] == 0
    S.hub.host.scan_repeat["enabled"] = False         # the loop's last frame, after it ended
    assert pump(_finished)["withdraws"] == 1


def test_a_failed_withdraw_hands_the_retract_to_the_failsafe(pump, monkeypatch):
    def broken():
        raise RuntimeError("link down")

    monkeypatch.setattr(S.hub, "fault", None)
    monkeypatch.setattr(S, "_hardware_withdraw", broken)
    seen = pump(_finished)
    assert len(seen["parks"]) == 1 and "link down" in seen["parks"][0]
    assert S.hub.fault["reason"] == "withdraw_failed"


def test_a_slow_save_does_not_delay_the_withdraw(pump, monkeypatch):
    order = []
    from afm_litho.service import captures

    def slow_save(*args, **kwargs):
        order.append("save")
        return None

    monkeypatch.setattr(captures, "save_capture", slow_save)
    monkeypatch.setattr(S, "_hardware_withdraw",
                        lambda: order.append("withdraw") or {"z_v": 0.0})
    pump(_finished)
    assert order == ["withdraw", "save"]


def test_a_pass_that_starts_during_the_withdraw_is_not_lifted_by_it(pump, monkeypatch):
    def withdraw_while_a_write_starts():
        S.hub.host._scan_epoch += 1               # the write engages as the thread returns
        return {"z_v": 0.0}

    monkeypatch.setattr(S, "_hardware_withdraw", withdraw_while_a_write_starts)
    pump(_finished)
    assert S.hub.host.engaged is True


def test_a_mode_switch_is_still_refused_while_the_scan_has_the_tip_down(pump):
    from afm_litho.service.host.scan_cmds import engaged_or_scanning
    seen = []

    async def scanning(*args, **kwargs):
        seen.append(engaged_or_scanning(S.hub))
        return _summary()

    pump(scanning)
    assert seen == [(True, "engaged")]


# ---- the Igor level, on the stub through the FLEX door ---------------------------------------

@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


def _lines(hub, client, name):
    return recorded(hub, client, name).golden["igor"]["actuation"]


def test_a_flex_scan_ends_with_the_routine_withdraw(hub, client):
    lines = _lines(hub, client, "scan_chained_contact")
    assert lines[-1] == WITHDRAW
    assert not any(line.startswith("td_Stop") for line in lines)


def test_a_flex_scan_then_a_write_still_runs_the_write(hub, client):
    lines = _lines(hub, client, "multi_pass")
    scan_end = max(i for i, line in enumerate(lines) if "AFM_ScanAdvance" in line)
    second_write = max(i for i, line in enumerate(lines)
                       if "XYTransitA" in line or "XYTransitB" in line)
    assert lines[scan_end + 1:].count(WITHDRAW) >= 2
    assert second_write > scan_end
    assert lines[-1] == WITHDRAW
