"""Instrument interface (slice 4.0): the message types build, check and refuse as specified."""

from __future__ import annotations

import math

import numpy as np
import pytest

from afm_litho.instrument import (TELEMETRY_HZ_CHAINED, TELEMETRY_HZ_IDLE, Citation, EngageAt,
                                  Instrument, MoveTo, ParkTip, PluginChoice,
                                  PluginRegistry, ReadChannels, Reading, Recorded, Result,
                                  ScanRaster, Status, TelemetryFrame, WithdrawTip,
                                  WriteObject, telemetry_hz)
from afm_litho.run import ScanFrame, Setpoint

RELATIVE = Setpoint("relative", 0.05)
CITE = Citation("w-2026-10-24-1802-01", 3)


def an_object(**changes) -> WriteObject:
    fields = dict(name="wire_a", x_um=[0, 1, 2], y_um=[0, 0, 0], tip_v=[0, 5, 0],
                  speed_um_s=0.5, setpoint=RELATIVE, record=("Deflection", "UserIn2"))
    fields.update(changes)
    return WriteObject(**fields)


def a_frame() -> ScanFrame:
    return ScanFrame("scan 13:58", 5, 256, 256, 0, (0, 0), False, "up", 1.0, True)


# --- commands ------------------------------------------------------------------


def test_commands_build_with_their_values():
    move = MoveTo(12.4, -3, 2, cite=CITE)
    assert (move.x_um, move.y_um, move.speed_um_s) == (12.4, -3.0, 2.0)
    assert type(move.y_um) is float and move.cite.step == 3
    assert EngageAt(RELATIVE, "contact").setpoint == RELATIVE
    assert WithdrawTip().cite is None
    assert ReadChannels(["Deflection", "DetectorSum"]).channels == ("Deflection", "DetectorSum")
    assert ParkTip("operator pressed Park").reason == "operator pressed Park"
    scan = ScanRaster(a_frame(), "ac", Setpoint("absolute", 1.0), True, ("Height", "Amplitude"),
                      steps=(CITE, Citation(CITE.run, 4)))
    assert scan.frame.line_rate_hz == 1.0 and scan.engage and scan.mode == "ac"


def test_setpoint_must_be_explicit():
    with pytest.raises(TypeError):
        EngageAt(mode="contact")
    with pytest.raises(ValueError, match="setpoint"):
        EngageAt(0.05, "contact")
    with pytest.raises(ValueError, match="setpoint"):
        an_object(setpoint=0.05)
    with pytest.raises(TypeError):
        Setpoint(0.05)


def test_engage_mode_is_contact_or_ac():
    with pytest.raises(ValueError, match="mode"):
        EngageAt(RELATIVE, "tapping")


@pytest.mark.parametrize("speed", [0, -1, math.nan, math.inf, True])
def test_speeds_are_positive_numbers(speed):
    with pytest.raises(ValueError):
        MoveTo(0, 0, speed)
    with pytest.raises(ValueError):
        an_object(speed_um_s=speed)


def test_citation_is_an_execution_id_and_a_step():
    assert Citation(run="s-2026-10-24-1802-02", step=0).step == 0
    with pytest.raises(ValueError, match="execution id"):
        Citation("run-1", 0)
    with pytest.raises(ValueError, match="step"):
        Citation("w-2026-10-24-1802-01", -1)


def test_object_points_are_read_only_arrays_of_equal_length():
    p = an_object()
    assert p.x_um.dtype == float and p.tip_v.tolist() == [0.0, 5.0, 0.0]
    with pytest.raises(ValueError):
        p.x_um[0] = 9.0
    with pytest.raises(ValueError, match="one value per point"):
        an_object(tip_v=[0, 5])
    with pytest.raises(ValueError, match="two points"):
        an_object(x_um=[0], y_um=[0], tip_v=[0])
    with pytest.raises(ValueError, match="finite"):
        an_object(y_um=[0, math.nan, 0])
    with pytest.raises(ValueError, match="one value per point"):
        an_object(x_um=[[0, 1, 2]])


def test_object_points_are_copied():
    x = np.array([0.0, 1.0, 2.0])
    p = an_object(x_um=x)
    x[0] = 7.0
    assert p.x_um[0] == 0.0


@pytest.mark.parametrize("channel", ["Output.Z", "Input.Fast", "Z%Output", "", "Höhe",
                                     "1Height", "Height\n", "Height Z"])
def test_channels_are_named_by_alias(channel):
    with pytest.raises(ValueError):
        ReadChannels((channel,))
    with pytest.raises(ValueError):
        an_object(record=(channel,))


def test_channels_are_named_once_and_reads_name_one():
    with pytest.raises(ValueError, match="twice"):
        ReadChannels(("Deflection", "Deflection"))
    with pytest.raises(ValueError, match="at least one"):
        ReadChannels(())
    with pytest.raises(ValueError, match="at least one"):
        ScanRaster(a_frame(), "contact", RELATIVE, False, ())
    with pytest.raises(ValueError, match="at least one"):
        an_object(record=())


@pytest.mark.parametrize("name", ["", "wire\ta", "wireα", 'wire "a"'])
def test_object_names_are_typeable_ascii(name):
    with pytest.raises(ValueError):
        an_object(name=name)


def test_park_says_why():
    for blank in ("", "   "):
        with pytest.raises(ValueError, match="why"):
            ParkTip(blank)
    with pytest.raises(ValueError, match="ASCII"):
        ParkTip("café")


# --- results -------------------------------------------------------------------


def test_results():
    done = Result("read", "done", readings=[Reading("Deflection", -0.2)])
    assert done.done and done.reading("Deflection") == -0.2
    with pytest.raises(KeyError):
        done.reading("DetectorSum")
    refused = Result("write", "refused", "a write is refused in AC mode")
    assert not refused.done
    with pytest.raises(ValueError, match="says why"):
        Result("scan", "aborted")
    with pytest.raises(ValueError, match="says why"):
        Result("scan", "aborted", "  ")
    with pytest.raises(ValueError, match="outcome"):
        Result("move", "failed", "no")
    with pytest.raises(ValueError, match="verb"):
        Result("dance", "done")
    with pytest.raises(ValueError, match="ASCII"):
        Result("move", "refused", "µm")


def test_a_failed_read_is_refused_never_nan():
    with pytest.raises(ValueError, match="finite"):
        Reading("Deflection", math.nan)
    assert not Result("read", "refused", "DetectorSum did not answer").readings


def test_readings_are_readings_named_once():
    with pytest.raises(ValueError, match="Reading"):
        Result("read", "done", readings=[("Deflection", -0.2)])
    with pytest.raises(ValueError, match="twice"):
        Result("read", "done", readings=[Reading("Deflection", -0.2), Reading("Deflection", 0)])


def test_write_result_carries_its_free_air_read():
    result = Result("write", "done", readings=[Reading("Deflection", -0.2),
                                               Reading("Height", 12.0)])
    assert result.reading("Deflection") == -0.2


def test_park_and_withdraw_say_whether_they_were_verified():
    with pytest.raises(ValueError, match="verified"):
        Result("park", "done", critical=False)
    with pytest.raises(ValueError, match="verified"):
        Result("withdraw", "done")
    with pytest.raises(ValueError, match="critical"):
        Result("park", "done", verified=True)
    unverified = Result("park", "done", verified=False, critical=True)
    assert unverified.done and unverified.verified is False and unverified.critical
    assert Result("withdraw", "aborted", "abort pressed", verified=False).verified is False
    with pytest.raises(ValueError, match="verified"):
        Result("park", "done", verified=1, critical=False)
    with pytest.raises(ValueError, match="not given"):
        Result("move", "done", verified=True)
    with pytest.raises(ValueError, match="not given"):
        Result("withdraw", "done", verified=True, critical=False)


def test_recorded_shapes():
    assert Recorded("Deflection", "write", [0.1, math.nan]).volts.shape == (2,)
    trace = Recorded("Height", "trace", np.zeros((4, 8)))
    assert trace.volts.shape == (4, 8)
    with pytest.raises(ValueError):
        trace.volts[0, 0] = 1.0
    with pytest.raises(ValueError, match="lines x points"):
        Recorded("Height", "retrace", [0.1, 0.2])
    with pytest.raises(ValueError, match="write, trace or retrace"):
        Recorded("Height", "both", np.zeros((1, 1)))
    with pytest.raises(ValueError, match="no values"):
        Recorded("Height", "write", [])
    for name in ("Output.Z", "Height\n", "Height Z"):
        with pytest.raises(ValueError, match="alias"):
            Recorded(name, "write", [0.0])


def test_steps_are_one_run_in_order():
    other = Citation("w-2026-10-24-1803-01", 4)
    with pytest.raises(ValueError, match="one run"):
        an_object(steps=(CITE, other))
    with pytest.raises(ValueError, match="run order"):
        an_object(steps=(Citation(CITE.run, 4), CITE))
    with pytest.raises(ValueError, match="run order"):
        an_object(steps=(CITE, CITE))
    assert an_object(steps=(CITE, Citation(CITE.run, 5))).steps[1].step == 5


# --- telemetry and status --------------------------------------------------------


def test_telemetry_frame_matches_the_telem_line():
    frame = TelemetryFrame(12.5, 1, 2, 30.0, -0.2, 10, math.nan)
    assert (frame.x_um, frame.sum_v) == (1.0, 10.0)
    assert math.isnan(frame.amplitude_v)
    with pytest.raises(ValueError, match="time"):
        TelemetryFrame(math.nan, 0, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="infinite"):
        TelemetryFrame(0, 0, 0, math.inf, 0, 0, 0)
    with pytest.raises(ValueError, match="number"):
        TelemetryFrame(0, "1", 0, 0, 0, 0, 0)


def test_telemetry_rate_follows_the_tip_poll():
    assert telemetry_hz("scanning") == TELEMETRY_HZ_CHAINED == 10.0
    assert telemetry_hz("writing", chained_write=True) == 10.0
    # A one-chunk ARMFIRE write does not set AFM_ltActive, so Igor stays at 4 Hz.
    assert telemetry_hz("writing") == TELEMETRY_HZ_IDLE == 4.0
    for activity in ("idle", "moving", "engaging", "engaged", "withdrawing", "parked"):
        assert telemetry_hz(activity, chained_write=True) == 4.0
    with pytest.raises(ValueError):
        telemetry_hz("dancing")


def test_status_carries_the_run_step():
    status = Status(3.0, "writing", "writing object 3: wire_a", cite=CITE)
    assert status.cite.run == "w-2026-10-24-1802-01"
    assert Status(0, "idle", "idle").cite is None
    with pytest.raises(ValueError, match="activity"):
        Status(0, "dancing", "dancing")
    with pytest.raises(ValueError, match="ASCII"):
        Status(0, "moving", "moving to 3 µm")


# --- plugin choice -----------------------------------------------------------------


def test_plugin_choice_from_the_setting():
    assert PluginChoice.from_setting("twin") == PluginChoice("twin", "cypher")
    assert PluginChoice.from_setting(" twin:mfp3d ").platform == "mfp3d"
    assert PluginChoice.from_setting("asylum") == PluginChoice("asylum")
    assert PluginChoice("asylum").platform is None
    with pytest.raises(ValueError, match="platform"):
        PluginChoice.from_setting("asylum:cypher")
    with pytest.raises(ValueError):
        PluginChoice.from_setting("bruker")
    with pytest.raises(ValueError):
        PluginChoice.from_setting("twin:nanoman")


class _Nothing:
    platform = "cypher"


def test_registry_opens_the_chosen_plugin():
    registry = PluginRegistry()
    opened = []
    registry.register("twin", lambda choice: opened.append(choice) or _Nothing())
    assert registry.open(PluginChoice("twin")).platform == "cypher"
    assert opened == [PluginChoice("twin", "cypher")]
    with pytest.raises(ValueError, match="not registered"):
        registry.open(PluginChoice("asylum"))
    with pytest.raises(ValueError, match="already"):
        registry.register("twin", lambda choice: _Nothing())
    with pytest.raises(ValueError, match="not a plugin"):
        registry.register("bruker", lambda choice: _Nothing())


def test_instrument_names_every_verb():
    verbs = {"listen", "abort", "move", "engage", "withdraw", "read", "park", "write", "scan",
             "close"}
    assert verbs <= set(dir(Instrument))


def test_write_object_chained_defaults_to_false_and_is_a_bool():
    assert an_object().chained is False
    assert an_object(chained=True).chained is True
    with pytest.raises(ValueError):
        an_object(chained="yes")
