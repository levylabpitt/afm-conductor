"""Behaviour taken from the shipped Asylum v16 source.

Each of these mirrors a specific rule in the vendor code rather than a
plausible guess about it. See docs/asylum-v16-findings.md for where.
"""

import pytest

from afm_litho.litho.compiler import (LithoObject, ar_get_deci, capture_points,
                                      compile_object)
from afm_litho.twin.physics import Instrument


def test_ar_get_deci_matches_the_instrument():
    """ARGetDeci in HardWare.ipf: max(round(50000/rate), 1)."""
    assert ar_get_deci(100) == 500
    assert ar_get_deci(2) == 25000
    assert ar_get_deci(60_000) == 1        # never below one tick
    assert ar_get_deci(0) == 1             # degenerate input, not a crash


def test_capture_buffers_are_padded_to_32():
    """Lithography.ipf pads inputPoints up to a multiple of 32."""
    assert capture_points(100) == 128
    assert capture_points(128) == 128      # already aligned: unchanged
    assert capture_points(1) == 32


@pytest.mark.parametrize("asked", [0.5, 1.0, 2.0, 3.0, 7.0, 13.0])
def test_write_speed_is_a_ceiling_and_is_quantised(asked):
    """The panel speed is a target, not a promise.

    Timing comes from the largest step against the velocity setting, and
    the interpolation factor is an integer, so the achieved speed lands on
    max_step * 50000/k. Note ARGetDeci ROUNDS rather than rounding up, so
    the result can sit slightly ABOVE the requested speed (3.0 um/s comes
    back as 3.0030). That is the instrument's behaviour, not ours: a strict
    ceiling would need a ceil() the vendor code does not do.
    """
    o = LithoObject("w", "path", [(0, 0), (15, 0)], speed_um_s=asked,
                    interpolation_um=0.02)
    c = compile_object(o, 6e-6, 6e-6)
    got = c["speed_achieved_um_s"]
    assert got == pytest.approx(asked, rel=0.01)
    # the error is exactly one decimation step's worth, never more
    k = c["ticks_per_point"]
    assert abs(1.0 / got - 1.0 / asked) <= (1.0 / asked) / max(k - 1, 1) + 1e-12
    # the tip moving, then one tick more: the step to the 0 V sample
    assert c["duration_s"] == pytest.approx(15.0 / got + k / 50_000, rel=1e-6)
    assert k >= 1


def test_one_long_step_slows_the_whole_segment():
    """The rate is set by the worst step, so a single long jump slows
    everything. This is a real property of the machine, not an artifact."""
    fine = LithoObject("a", "path", [(0, 0), (10, 0)], speed_um_s=2.0,
                       interpolation_um=0.02)
    a = compile_object(fine, 6e-6, 6e-6)
    # same path, but interpolated ten times more coarsely: bigger steps
    coarse = LithoObject("b", "path", [(0, 0), (10, 0)], speed_um_s=2.0,
                         interpolation_um=0.2)
    b = compile_object(coarse, 6e-6, 6e-6)
    assert b["points_per_second"] < a["points_per_second"]
    # both still honour the ceiling, to the tick's rounding (ARGetDeci rounds to
    # nearest: at most half a tick over)
    for c in (a, b):
        assert c["speed_achieved_um_s"] <= 2.0 * (1 + 0.5 / c["ticks_per_point"])


def test_tip_holder_gain_scales_the_commanded_bias():
    """Lithography.ipf: LithoBias /= TipParms.TipGain. The panel number is
    the bias at the tip; a HV holder (TipGain 15) reaches it from a much
    smaller DAC output."""
    hv = Instrument(tip_gain=15.0)
    assert hv.bias_to_dac(10.0) == pytest.approx(10.0 / 15.0)
    assert hv.dac_to_bias(hv.bias_to_dac(10.0)) == pytest.approx(10.0)
    plain = Instrument()
    assert plain.bias_to_dac(10.0) == pytest.approx(10.0)


def test_orca_current_is_signed():
    """cOrcaSign = -1: the current channel is inverted."""
    i = Instrument(orca_gain=1e-9, is_orca=True)
    assert i.current_from_volts(1.0) == pytest.approx(-1e-9)
    assert i.current_from_volts(-2.0) == pytest.approx(2e-9)


def test_instrument_reads_tip_and_orca_from_a_wave_note():
    inst = Instrument.from_note({
        "MicroscopeModel": "MFP3D", "TipGain": "15", "OrcaGain": "2e-9",
        "OrcaGain2": "2e-7",
    })
    assert inst.tip_gain == pytest.approx(15.0)
    assert inst.orca_gain == pytest.approx(2e-9)
    assert inst.is_orca
    assert inst.current_from_volts(1.0, gain2=True) == pytest.approx(-2e-7)


def test_missing_tip_gain_never_divides_by_zero():
    """A note with TipGain 0 (or absent) must behave as unity gain."""
    assert Instrument.from_note({"TipGain": "0"}).bias_to_dac(5.0) == 5.0
    assert Instrument.from_note({}).bias_to_dac(5.0) == 5.0
