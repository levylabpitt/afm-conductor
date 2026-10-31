"""The Asylum plugin without an instrument: what it refuses, how it chains the telemetry hook,
how run steps are matched to sub-steps. The wire proof is test_instrument_asylum_goldens.py."""

import math
import types
import threading
from types import MappingProxyType, SimpleNamespace

import pytest

from afm_litho import clock
import numpy as np

from afm_litho.instrument import (Citation, EngageAt, ParkTip, ReadChannels, ScanRaster,
                                  WithdrawTip, WriteObject)
from afm_litho.instrument import asylum as A
from afm_litho.instrument.asylum import AsylumInstrument, _align_to_end
from afm_litho.run.model import ScanFrame, Setpoint

RUN = "w-2026-10-26-1200-00"
CAL = {"x_lvdt_sens": 2e-5, "x_lvdt_offset": 0.5, "y_lvdt_sens": 4e-5, "y_lvdt_offset": -0.25}


class FakeArc:
    """Just enough of a HardwareARC for real_bridge, read and the telemetry hook."""

    is_cypher = True

    def __init__(self, values=None):
        self.router = SimpleNamespace(on_telem=None)
        self.values = values or {}

    def td_ReadValue(self, name):
        value = self.values[name]
        if isinstance(value, Exception):
            raise value
        return value


def plugin(arc=None, **hub_fields):
    hub = SimpleNamespace(hardware_arc=arc or FakeArc(), hw_cal=dict(CAL), **hub_fields)
    hub.host = hub  # fake: host fields live on the fake itself
    return AsylumInstrument(hub, amp_gain=25.0)


def frame(**kw):
    values = dict(source="t", size_um=1.0, lines=2, pixels=32, angle_deg=0.0,
                  offset_um=(0.0, 0.0), drift_offset=False, direction="up", line_rate_hz=50.0,
                  trace_and_retrace=True)
    return ScanFrame(**{**values, **kw})


def write_command(**kw):
    values = dict(name="w", x_um=[0.0, 1.0], y_um=[0.0, 0.0], tip_v=[1.0, 0.0], speed_um_s=1.0,
                  setpoint=Setpoint("absolute", 0.0), record=("Deflection",))
    return WriteObject(**{**values, **kw})


def test_the_amplifier_gain_is_positive_and_a_write_needs_it():
    with pytest.raises(ValueError):
        AsylumInstrument(SimpleNamespace(), amp_gain=0.0)
    result = AsylumInstrument(SimpleNamespace()).write(write_command())
    assert result.outcome == "refused"


def test_a_write_that_records_what_the_capture_cannot_give_is_refused():
    result = plugin().write(write_command(record=("Deflection", "Amplitude")))
    assert result.outcome == "refused" and "Amplitude" in result.reason


def test_a_scan_records_only_its_fixed_capture():
    contact = ScanRaster(frame(), "contact", Setpoint("absolute", 0.2), False, ("Amplitude",))
    ac = ScanRaster(frame(), "ac", Setpoint("absolute", 0.2), False, ("Deflection",))
    assert plugin().scan(contact).outcome == "refused"
    assert plugin().scan(ac).outcome == "refused"
    height = ScanRaster(frame(), "contact", Setpoint("absolute", 0.2), False, ("Height",))
    assert "ZSensor" in plugin().scan(height).reason


def test_a_relative_setpoint_is_refused_for_an_engage_and_a_scan():
    assert plugin().engage(EngageAt(Setpoint("relative", 0.3), "contact")).outcome == "refused"
    scan = ScanRaster(frame(), "contact", Setpoint("relative", 0.3), True, ("ZSensor",))
    assert plugin().scan(scan).outcome == "refused"


def test_a_read_that_fails_is_refused_with_the_reason_and_no_value():
    asylum = plugin(FakeArc({"Deflection": 0.25, "DetectorSum": RuntimeError("bridge closed")}))
    ok = asylum.read(ReadChannels(("Deflection",)))
    assert ok.done and ok.reading("Deflection") == 0.25
    refused = asylum.read(ReadChannels(("Deflection", "DetectorSum")))
    assert refused.outcome == "refused" and "DetectorSum" in refused.reason
    assert refused.readings == ()


def test_a_read_that_is_not_a_number_is_refused():
    asylum = plugin(FakeArc({"Deflection": math.nan}))
    assert asylum.read(ReadChannels(("Deflection",))).outcome == "refused"


def test_telemetry_x_and_y_are_micrometres_and_the_service_consumer_still_gets_the_frame():
    arc = FakeArc()
    seen = []
    arc.router.on_telem = lambda *frame_: seen.append(frame_)
    asylum = plugin(arc)
    frames = []
    asylum.listen(frames.append, lambda status: None)
    arc.router.on_telem(0.6, -0.2, 12.0, 0.1, 9.5, math.inf)
    assert seen == [(0.6, -0.2, 12.0, 0.1, 9.5, math.inf)]
    (got,) = frames
    assert got.x_um == pytest.approx((0.6 - 0.5) * 2e-5 * 1e6)
    assert got.y_um == pytest.approx((-0.2 + 0.25) * 4e-5 * 1e6)
    assert (got.z_drive_v, got.deflection_v, got.sum_v) == (12.0, 0.1, 9.5)
    assert math.isnan(got.amplitude_v)
    assert got.t_s == pytest.approx(clock.now(), abs=5.0)


def test_close_puts_the_service_consumer_back():
    arc = FakeArc()
    original = arc.router.on_telem = lambda *frame_: None
    asylum = plugin(arc)
    asylum.listen(lambda f: None, lambda s: None)
    assert arc.router.on_telem != original
    asylum.close()
    assert arc.router.on_telem is original


def test_run_steps_are_matched_to_the_last_sub_steps():
    cites = tuple(Citation(RUN, n) for n in (3, 4, 5))
    kinds = ["move", "settle", "read", "engage", "pass", "withdraw"]
    got = _align_to_end(kinds, cites)
    assert {kind: c.step for kind, c in got.items()} == {"engage": 3, "pass": 4, "withdraw": 5}
    assert _align_to_end(kinds, ()) == {}


# --- stops, faults and failures, on a hub of fakes ---------------------------------------


class Written:
    aborted = False
    capture = types.MappingProxyType({})
    cap_len = 0


class Request:
    def __init__(self, result):
        self.result = result

    def wait(self, timeout):
        return self.result is not None


@pytest.fixture
def rig(monkeypatch):
    """A plugin on a fake hub; write_object, the watchdog and the XY refresh are fakes."""
    import afm_litho.hardware.lithography_run as LR
    import afm_litho.service.host.litho_cmds as lc
    import importlib
    sp = importlib.import_module("afm_litho.service.host.safe_park")
    import afm_litho.service.xy_cal as xc
    state = SimpleNamespace(fired=[], withdrawn=[], park_outcome={"verified": True,
                                                                  "critical": False})
    hub = SimpleNamespace(
        _hw_litho_abort=threading.Event(), _hw_abort=threading.Event(),
        _scan_repeat_stop=threading.Event(), scan_repeat={}, hw_cal={}, engaged=False,
        litho_worker=None, fault=None, _run_abort_seq=0, _safe_park={}, _scan_epoch=0)
    hub.host = hub  # fake: host fields live on the fake itself
    arc = SimpleNamespace(link=None, z_out_channel="Height", is_cypher=True, router=None,
                          com_igor=None)

    def fire(self, name, *a, **k):
        state.fired.append(self._abort_event.is_set())
        return Written()

    monkeypatch.setattr(LR.LithographyRun, "write_object", fire)
    monkeypatch.setattr(lc, "ac_mode_active", lambda hub: False)
    monkeypatch.setattr(xc, "refresh_xy_cal", lambda hub, source: None)
    monkeypatch.setattr(sp, "watchdog_for", lambda hub: SimpleNamespace(
        request=lambda reason, report: Request(state.park_outcome)))
    monkeypatch.setattr(A, "hw_withdraw", lambda a: state.withdrawn.append(
        hub.host._hw_litho_abort.is_set()) or state.snap)
    state.snap = {"z_v": 0.0}
    asylum = AsylumInstrument(hub, amp_gain=25.0)
    asylum._arc = lambda: arc
    state.hub, state.asylum = hub, asylum
    return state


WRITE = dict(name="w", x_um=np.array([0.0, 1.0]), y_um=np.array([0.0, 0.0]),
             tip_v=np.array([1.0, 0.0]), speed_um_s=1.0, setpoint=Setpoint("absolute", 0.0),
             record=("Deflection",), source="w")


def test_a_write_after_a_park_is_refused_while_the_park_is_in_force(rig):
    rig.hub.host._hw_litho_abort.set()          # what a park leaves set
    rig.hub.fault = {"reason": "park_unverified"}
    result = rig.asylum.write(WriteObject(**WRITE))
    assert result.outcome == "refused" and rig.fired == [] and not rig.hub.host.engaged


def test_a_stop_signalled_while_a_write_starts_refuses_it(rig, monkeypatch):
    import afm_litho.service.xy_cal as xc

    def stop_during_refresh(hub, source):
        hub.host._run_abort_seq += 1
    monkeypatch.setattr(xc, "refresh_xy_cal", stop_during_refresh)
    result = rig.asylum.write(WriteObject(**WRITE))
    assert result.outcome == "refused" and rig.fired == []


def test_a_clean_write_after_an_earlier_park_goes_ahead(rig):
    assert rig.asylum.park(ParkTip("operator")).verified
    assert rig.asylum.write(WriteObject(**WRITE)).done and rig.fired == [False]


def test_a_withdraw_stops_the_pass_in_flight_first(rig):
    result = rig.asylum.withdraw(WithdrawTip())
    assert rig.withdrawn == [True] and result.verified is True


@pytest.mark.parametrize("snap", [{"z_v": 0.6}, {"z_v": 0.0, "verify_timeout": True},
                                  {"z_v": None}])
def test_a_withdraw_that_is_not_confirmed_is_not_verified(rig, snap):
    rig.snap = snap
    assert rig.asylum.withdraw(WithdrawTip()).verified is False


@pytest.mark.parametrize("outcome,verified,critical", [
    ({"verified": True, "critical": False}, True, False),
    ({"verified": False, "critical": True}, False, True),
    ({"verified": False, "critical": False}, False, True),
    (None, False, True)])
def test_the_park_outcome_maps_to_verified_and_critical(rig, outcome, verified, critical):
    rig.park_outcome = outcome
    result = rig.asylum.park(ParkTip("operator"))
    assert (result.verified, result.critical) == (verified, critical)


def test_a_failed_engage_parks_and_reports_aborted(rig, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("bridge closed")
    monkeypatch.setattr(A, "hw_engage", broken)
    monkeypatch.setattr(A.AsylumInstrument, "_reconcile_mode", lambda self, mode: None)
    rig.hub.host.z_loop = {}
    rig.park_outcome = {"verified": False, "critical": True}
    result = rig.asylum.engage(EngageAt(Setpoint("absolute", 0.5), "contact"))
    assert result.outcome == "aborted" and "critical=True" in result.reason
    assert rig.hub.host.engaged is False


def test_an_abort_with_no_instrument_does_not_raise():
    asylum = AsylumInstrument(SimpleNamespace(), amp_gain=25.0)
    asylum.abort("no bridge")


class Cut(Written):
    """A pass the controller cut half way: 16 of 32 captured points."""

    aborted = True
    cap_len = 32
    capture = MappingProxyType({"Deflection": [0.1] * 16})


def test_a_cut_pass_gives_its_captured_prefix_and_the_fraction_landed(rig, monkeypatch):
    import afm_litho.hardware.lithography_run as LR
    monkeypatch.setattr(LR.LithographyRun, "write_object", lambda self, *a, **k: Cut())
    result = rig.asylum.write(WriteObject(**WRITE))
    assert result.outcome == "aborted" and result.landed == 0.5
    assert result.recorded[0].channel == "Deflection" and len(result.recorded[0].volts) == 16


def test_a_stopped_write_is_withdrawn_by_the_plugin_unless_the_service_owns_the_write(
        rig, monkeypatch):
    import afm_litho.hardware.lithography_run as LR
    monkeypatch.setattr(LR.LithographyRun, "write_object", lambda self, *a, **k: Cut())
    assert rig.asylum.write(WriteObject(**WRITE)).outcome == "aborted"
    assert len(rig.withdrawn) == 1 and rig.hub.host.engaged is False
    rig.withdrawn.clear()
    rig.asylum._service_owns_write = True
    result = rig.asylum.write(WriteObject(**WRITE))
    assert result.outcome == "aborted" and result.landed == 0.5 and rig.withdrawn == []


def test_a_failure_is_kept_for_the_caller_and_the_write_is_aborted_not_raised(rig, monkeypatch):
    import afm_litho.hardware.lithography_run as LR

    def boom(self, *a, **k):
        raise OSError("link gone")
    monkeypatch.setattr(LR.LithographyRun, "write_object", boom)
    result = rig.asylum.write(WriteObject(**WRITE))
    assert result.outcome == "aborted" and result.landed is None
    assert isinstance(rig.asylum.failure, OSError) and len(rig.withdrawn) == 1


def test_an_unconfirmed_engage_is_withdrawn_and_reported(rig, monkeypatch):
    import afm_litho.hardware.lithography_run as LR
    from afm_litho.hardware.engage_check import EngageNotConfirmed

    def no_engage(self, *a, **k):
        raise EngageNotConfirmed("write w: tip did not engage within 6 s")
    monkeypatch.setattr(LR.LithographyRun, "write_object", no_engage)
    result = rig.asylum.write(WriteObject(**WRITE))
    assert result.outcome == "aborted" and "did not engage" in result.reason
    assert isinstance(rig.asylum.failure, EngageNotConfirmed) and len(rig.withdrawn) == 1
    assert rig.hub.host.engaged is False


def test_a_fault_latched_mid_write_refuses_the_later_strokes_with_no_commands(rig, monkeypatch):
    import afm_litho.hardware.lithography_run as LR

    def fire_then_fault(self, name, *a, **k):
        rig.fired.append(name)
        rig.hub.fault = {"reason": "rail_guard"}
        return Written()
    monkeypatch.setattr(LR.LithographyRun, "write_object", fire_then_fault)
    assert rig.asylum.write(WriteObject(**WRITE)).done
    later = rig.asylum.write(WriteObject(**{**WRITE, "name": "w2"}))
    assert later.outcome == "refused" and rig.fired == ["w"] and rig.withdrawn == []


def test_a_clean_write_reports_each_step(rig, monkeypatch):
    import afm_litho.hardware.lithography_run as LR
    seen = []
    rig.asylum.on_step = lambda kind, make: seen.append(kind)

    def fire(self, name, *a, **k):
        self._outcome("pass", lambda: {"aborted": False})
        return Written()
    monkeypatch.setattr(LR.LithographyRun, "write_object", fire)
    assert rig.asylum.write(WriteObject(**WRITE)).done
    assert seen == ["pass"] and rig.withdrawn == []
