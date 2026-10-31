"""The hardware engage and withdraw doors drive the Asylum plugin with `service_owns_approach`:
the service has stopped the pass, reconciled the mode, run the guards and records the
engagement itself. The plugin then sends only the Igor commands and must do none of the rest.
Each test names one step the plugin skips; the plugin without the flag still does it."""

import asyncio
import importlib
import threading
from types import SimpleNamespace

import pytest

from afm_litho.instrument import EngageAt, WithdrawTip
from afm_litho.instrument import asylum as A
from afm_litho.instrument.asylum import AsylumInstrument
from afm_litho.run.model import Setpoint
from afm_litho.service.host import scan_cmds

SNAP = {"engaged": True, "defl_v": 0.5, "z_v": 10.0, "reached_setpoint": True}
ENGAGE = EngageAt(Setpoint("absolute", 0.5), "contact")


@pytest.fixture
def rig(monkeypatch):
    """A plugin on a fake hub; hw_engage, hw_withdraw, the mode reconcile, the stop signal
    and the park are counting fakes."""
    sp = importlib.import_module("afm_litho.service.host.safe_park")
    seen = SimpleNamespace(engages=[], withdraws=0, reconciles=0, parks=0, stops=[],
                           engage_error=None)
    hub = SimpleNamespace(
        _hw_abort=threading.Event(), _hw_litho_abort=threading.Event(), engaged=False,
        _scan_epoch=7, _safe_park={"active": True}, fault=None)
    hub.host = hub  # fake: host fields live on the fake itself
    arc = SimpleNamespace(is_cypher=True)

    def engage(arc_, **kwargs):
        if seen.engage_error:
            raise seen.engage_error
        seen.engages.append(kwargs)
        return dict(SNAP)

    def withdraw(arc_):
        seen.withdraws += 1
        return {"engaged": False, "defl_v": 0.0, "z_v": 0.0}

    def park_requested(hub_):
        seen.parks += 1
        done = SimpleNamespace(wait=lambda bound: None,
                               result={"verified": True, "critical": False})
        return SimpleNamespace(request=lambda reason, report: done)

    def reconcile(self, mode):
        seen.reconciles += 1

    monkeypatch.setattr(A, "hw_engage", engage)
    monkeypatch.setattr(A, "hw_withdraw", withdraw)
    monkeypatch.setattr(A.AsylumInstrument, "_reconcile_mode", reconcile)
    monkeypatch.setattr(sp, "watchdog_for", park_requested)
    monkeypatch.setattr(sp, "_signal_stop", lambda hub_, which="all": seen.stops.append(which))
    seen.hub, seen.arc = hub, arc

    def plugin(owns):
        asylum = AsylumInstrument(hub, engage_pgain=2.0, engage_igain=15.0,
                                  engage_settle_s=0.0, service_owns_approach=owns)
        asylum._arc = lambda: arc
        return asylum

    seen.plugin = plugin
    return seen


def test_the_plugin_sends_the_engage_with_the_service_gains(rig):
    asylum = rig.plugin(True)
    assert asylum.engage(ENGAGE).done
    assert rig.engages == [dict(setpoint=0.5, pgain=2.0, igain=15.0, settle_s=0.0,
                                mode="contact")]
    assert asylum.snapshot == SNAP


def test_an_owned_engage_leaves_the_mode_to_the_service(rig):
    rig.plugin(True).engage(ENGAGE)
    assert rig.reconciles == 0
    rig.plugin(False).engage(ENGAGE)
    assert rig.reconciles == 1


def test_an_owned_engage_leaves_the_z_loop_record_to_the_service(rig):
    rig.plugin(True).engage(ENGAGE)
    assert not hasattr(rig.hub.host, "z_loop")
    rig.plugin(False).engage(ENGAGE)
    assert rig.hub.host.z_loop["setpoint"] == 0.5


def test_an_owned_engage_leaves_engaged_to_the_service(rig):
    rig.plugin(True).engage(ENGAGE)
    assert rig.hub.host.engaged is False
    rig.plugin(False).engage(ENGAGE)
    assert rig.hub.host.engaged is True


def test_an_owned_engage_leaves_the_scan_epoch_to_the_service(rig):
    rig.plugin(True).engage(ENGAGE)
    assert rig.hub.host._scan_epoch == 7
    rig.plugin(False).engage(ENGAGE)
    assert rig.hub.host._scan_epoch == 8


def test_an_owned_engage_leaves_the_park_latch_to_the_service(rig):
    rig.plugin(True).engage(ENGAGE)
    assert rig.hub.host._safe_park["active"] is True
    rig.plugin(False).engage(ENGAGE)
    assert rig.hub.host._safe_park["active"] is False


def test_a_failed_owned_engage_is_not_parked_by_the_plugin(rig):
    rig.engage_error = OSError("bridge gone")
    asylum = rig.plugin(True)
    result = asylum.engage(ENGAGE)
    assert result.outcome == "aborted"
    assert asylum.failure is rig.engage_error
    assert rig.parks == 0 and rig.hub.host.engaged is False


def test_a_failed_engage_without_the_flag_is_still_parked_by_the_plugin(rig):
    rig.engage_error = OSError("bridge gone")
    rig.plugin(False).engage(ENGAGE)
    assert rig.parks == 1


def test_an_owned_withdraw_signals_no_stop(rig):
    assert rig.plugin(True).withdraw(WithdrawTip()).verified is True
    assert rig.withdraws == 1 and rig.stops == []
    rig.plugin(False).withdraw(WithdrawTip())
    assert rig.stops == ["run"]


def test_an_owned_withdraw_leaves_engaged_to_the_service(rig):
    rig.hub.host.engaged = True
    rig.plugin(True).withdraw(WithdrawTip())
    assert rig.hub.host.engaged is True
    rig.plugin(False).withdraw(WithdrawTip())
    assert rig.hub.host.engaged is False


def test_a_failed_owned_withdraw_keeps_the_exception(rig, monkeypatch):
    def broken(arc):
        raise OSError("bridge gone")
    monkeypatch.setattr(A, "hw_withdraw", broken)
    asylum = rig.plugin(True)
    result = asylum.withdraw(WithdrawTip())
    assert result.outcome == "aborted" and result.verified is False
    assert isinstance(asylum.failure, OSError)


# --- the service doors ------------------------------------------------------------


@pytest.fixture
def door(rig, monkeypatch):
    """The app's hub over the same fake arc."""
    from afm_litho.service.app import hub
    monkeypatch.setattr(hub.host, "hardware_arc", rig.arc)
    monkeypatch.setattr(hub.host, "engaged", True)
    return hub


def test_the_engage_door_sends_the_service_values_through_the_plugin(rig, door):
    snap = scan_cmds._engage_on_instrument(door, setpoint=0.7, pgain=1.0, igain=5.0,
                                           settle_s=3.0, mode="ac")
    assert rig.engages == [dict(setpoint=0.7, pgain=1.0, igain=5.0, settle_s=3.0, mode="ac")]
    assert snap == SNAP


@pytest.mark.parametrize("word", ["current", "tapping"])
def test_a_mode_word_other_than_contact_or_ac_reaches_the_engage_as_given(rig, door, word):
    scan_cmds._engage_on_instrument(door, setpoint=0.7, pgain=1.0, igain=5.0,
                                    settle_s=3.0, mode=word)
    assert rig.engages == [dict(setpoint=0.7, pgain=1.0, igain=5.0, settle_s=3.0, mode=word)]


def test_current_mode_engages_on_the_deflection_loop_with_a_positive_gain(monkeypatch):
    from afm_litho.hardware import hw_approach
    sent = []
    arc = SimpleNamespace(z_out_channel="Height",
                          ir_SetPISLoop=lambda *a: sent.append(a),
                          td_ReadValue=lambda name, **k: 0.5)
    hub = SimpleNamespace(hardware_arc=arc, engaged=False)
    hub.host = hub  # fake: host fields live on the fake itself
    scan_cmds._engage_on_instrument(hub, setpoint=0.5, pgain=2.0, igain=15.0,
                                    settle_s=0.0, mode="current")
    base = []
    arc.ir_SetPISLoop = lambda *a: base.append(a)
    hw_approach.hw_engage(arc, setpoint=0.5, pgain=2.0, igain=15.0, settle_s=0.0,
                          mode="current")
    assert sent == base and sent[0][2] == "Deflection" and sent[0][4] > 0


def test_a_failed_engage_raises_the_original_exception_to_the_service(rig, door):
    rig.engage_error = OSError("bridge gone")
    with pytest.raises(OSError, match="bridge gone"):
        scan_cmds._engage_on_instrument(door, setpoint=0.7, pgain=1.0, igain=5.0,
                                        settle_s=3.0, mode="contact")
    assert rig.parks == 0


def test_the_withdraw_door_retracts_through_the_plugin_and_the_service_does_the_rest(
        rig, door, monkeypatch):
    after = []
    monkeypatch.setattr(scan_cmds, "_post_withdraw", lambda hub: after.append("disarmed"))

    async def free_air(hub):
        after.append("free_air")
    monkeypatch.setattr(scan_cmds, "_establish_free_air_on_withdraw", free_air)
    snap = asyncio.run(scan_cmds.withdraw(door))
    assert rig.withdraws == 1 and snap["z_v"] == 0.0
    assert door.host.engaged is False
    assert after == ["disarmed", "free_air"]
    assert rig.stops == ["run"]            # the service's own stop, sent once
