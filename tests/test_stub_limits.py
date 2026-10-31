"""The Igor stub enforces the controller's limits and models event-bound banks (SLICES 0.5).

Real Igor discards the td_x* return code of a failed bind and sends no ERR
(afm_litho_run.ipf:309-320, 610-649), so a rejected call is silent here too: it lands in
`stub.violations` and, where Igor would never send DONE, no DONE comes back. Calls go to
`FakeIgorBridge._dispatch` with a recording link; numbers are literals on purpose.
"""

import asyncio
import socket
import time

import pytest

from afm_litho.hardware import bridge_protocol as P
from afm_litho.hardware.igor_stub import FakeIgorBridge, Violation

# Most tests here break the 32 rule on purpose and assert the recorded violation.
pytestmark = pytest.mark.expects_in_wave_violation


class _Link:
    def __init__(self):
        self.sent: list[str] = []

    def send(self, line: str) -> None:
        self.sent.append(line)


def _stub(**kw):
    return FakeIgorBridge({}, **kw), _Link()


def _d(s, link, line):
    s._dispatch(line, link)


def _rules(s):
    return [v.rule for v in s.violations]


def _no_err(link):
    assert not [x for x in link.sent if x.startswith("ERR")]


def _silent(link):
    """No ERR frame, and (after any timer) no DONE."""
    time.sleep(0.15)
    assert not [x for x in link.sent if x.startswith(("ERR", "DONE"))]


def _done(link):
    time.sleep(0.15)
    return [x for x in link.sent if x.startswith("DONE")]


# ---------------------------------------------------------------- out-wave cap

def test_out_wave_of_87381_points_is_rejected_silently():
    s, link = _stub()
    s.waves["W"] = [0.0] * 87381
    _d(s, link, P.cmd_outwave(0, "14", "Output.A", "W", -1))
    assert s.violations == [Violation("out_wave_over_cap", P.OUTWAVE, 0, 87381)]
    assert s.out_banks == {}
    _no_err(link)
    s.stop()


def test_out_wave_of_87380_points_passes():
    s, link = _stub()
    s.waves["W"] = [0.0] * 87380
    _d(s, link, P.cmd_outwave(1, "14", "Output.A", "W", -1))
    assert s.violations == [] and 1 in s.out_banks
    s.stop()


def test_out_pair_second_wave_over_cap_is_rejected():
    s, link = _stub()
    s.waves["X"] = [0.0] * 10
    s.waves["Y"] = [0.0] * 87381
    _d(s, link, P.cmd_outpair(0, "14", "PIDSLoop.0.Setpoint", "X",
                              "PIDSLoop.1.Setpoint", "Y", -1))
    assert s.violations == [Violation("out_wave_over_cap", P.OUTPAIR, 0, 87381)]
    assert s.out_banks == {}


def test_out_pair_at_87380_passes():
    s, link = _stub()
    s.waves["X"] = [0.0] * 87380
    s.waves["Y"] = [0.0] * 87380
    _d(s, link, P.cmd_outpair(0, "14", "PIDSLoop.0.Setpoint", "X",
                              "PIDSLoop.1.Setpoint", "Y", -1))
    assert s.violations == [] and 0 in s.out_banks


def test_armfire_xlitho_over_cap_is_silent_and_the_pass_still_completes():
    s, link = _stub()
    s.waves["XLitho"] = [0.0] * 87381
    _d(s, link, P.cmd_armfire("u1", "o1", -1, 64, 1))
    assert s.violations == [Violation("out_wave_over_cap", P.ARMFIRE, 0, 87381)]
    assert 0 not in s.out_banks and 2 in s.out_banks
    assert len(_done(link)) == 1         # the in-banks still fill; only the XY wave is lost
    _no_err(link)
    s.stop()


def test_armfire_vlitho_over_cap_leaves_out_bank_2_unbound():
    s, link = _stub()
    s.waves["VLitho"] = [0.0] * 87381
    _d(s, link, P.cmd_armfire("u1", "o1", -1, 64, 1))
    assert s.violations == [Violation("out_wave_over_cap", P.ARMFIRE, 2, 87381)]
    assert 2 not in s.out_banks and 0 in s.out_banks
    s.stop()


def test_armfire_at_87380_passes():
    s, link = _stub()
    for n in ("XLitho", "YLitho", "VLitho"):
        s.waves[n] = [0.0] * 87380
    _d(s, link, P.cmd_armfire("u1", "o1", -1, 64, 1))
    assert s.violations == []
    s.stop()


def test_scanchain_ppl_over_cap_is_recorded_and_the_lines_still_run():
    s, link = _stub()
    _d(s, link, f"{P.SCANCHAIN} u9 XS YS 87392 1 -1 1")     # 32 x 2731, over the cap
    s._sc_thread.join(5)
    assert s.violations == [Violation("out_wave_over_cap", P.SCANCHAIN, 0, 87392)]
    assert any(x.startswith("SCANDONE") for x in link.sent)
    _no_err(link)
    s.stop()


def test_lithochain_chunk_over_cap_is_recorded():
    s, link = _stub()
    s.waves["B"] = [0.0, 87381.0]
    _d(s, link, f"{P.LITHOCHAIN} u1 o1 XLitho YLitho VLitho B -1 1")
    assert s.violations == [Violation("out_wave_over_cap", P.LITHOCHAIN, 0, 87381)]
    _no_err(link)
    s.stop()


# ------------------------------------------------------------ in-wave multiple

def test_armfire_capture_of_33_is_silent_and_never_completes():
    s, link = _stub()
    _d(s, link, P.cmd_armfire("u1", "o1", -1, 33, 1))
    assert s.violations == [Violation("in_wave_not_multiple_of_32", P.ARMFIRE, 0, 33)]
    assert s.in_banks == {}
    _silent(link)
    s.stop()


def test_armfire_capture_of_64_passes_and_completes():
    s, link = _stub()
    _d(s, link, P.cmd_armfire("u1", "o1", -1, 64, 1))
    assert s.violations == []
    assert len(_done(link)) == 1
    s.stop()


def test_general_in_wave_of_33_is_silent_and_fire_never_completes():
    s, link = _stub()
    _d(s, link, P.cmd_inwave(0, "14", "ZSensor", 33, 1))
    assert s.violations == [Violation("in_wave_not_multiple_of_32", P.INWAVE, 0, 33)]
    _d(s, link, P.cmd_fire("u1", "p", "14"))
    _silent(link)
    s.stop()


def test_in_pair_of_33_is_rejected_and_64_passes():
    s, link = _stub()
    _d(s, link, P.cmd_inpair(0, "14", "ZSensor", "Deflection", 33, 1))
    assert s.violations == [Violation("in_wave_not_multiple_of_32", P.INPAIR, 0, 33)]
    assert s.in_banks == {}
    _d(s, link, P.cmd_inpair(0, "14", "ZSensor", "Deflection", 64, 1))
    assert len(s.violations) == 1 and 0 in s.in_banks
    s.stop()


def test_a_working_in_bank_still_completes_when_another_bind_failed():
    s, link = _stub()
    _d(s, link, P.cmd_inpair(0, "14", "ZSensor", "Deflection", 64, 1))
    _d(s, link, P.cmd_inwave(2, "14", "Output.A", 33, 1))
    _d(s, link, P.cmd_fire("u1", "p", "14"))
    assert len(s.violations) == 1
    assert len(_done(link)) == 1
    s.stop()


def test_the_caller_times_out_when_the_fire_never_completes():
    s = FakeIgorBridge({})
    port = s.serve_in_thread()
    link = P.BridgeLink.connect("127.0.0.1", port)
    try:
        link.send(P.cmd_inwave(0, "14", "ZSensor", 33, 1))
        link.send(P.cmd_fire("u1", "p", "14"))
        with pytest.raises(socket.timeout):
            link.read_line(timeout=0.4)
    finally:
        link.close()
        s.stop()


# ----------------------------------------------------------------------- banks

@pytest.mark.parametrize("line,verb", [
    (P.cmd_outwave(3, "14", "Output.A", "W", -1), P.OUTWAVE),
    (P.cmd_inwave(3, "14", "ZSensor", 32, 1), P.INWAVE),
    (P.cmd_inpair(3, "14", "ZSensor", "Deflection", 32, 1), P.INPAIR),
])
def test_bank_3_is_rejected_as_above_2(line, verb):
    s, link = _stub()
    s.waves["W"] = [0.0] * 8
    _d(s, link, line)
    assert s.violations == [Violation("bank_above_2", verb, 3, None)]
    assert s.out_banks == {} and s.in_banks == {}


def test_bank_minus_1_is_its_own_rule():
    s, link = _stub()
    _d(s, link, P.cmd_inwave(-1, "14", "ZSensor", 32, 1))
    assert s.violations == [Violation("bank_below_0", P.INWAVE, -1, None)]


@pytest.mark.parametrize("bank", [0, 1, 2])
def test_banks_0_to_2_pass(bank):
    s, link = _stub()
    s.waves["W"] = [0.0] * 8
    _d(s, link, P.cmd_outwave(bank, "14", "Output.A", "W", -1))
    _d(s, link, P.cmd_inwave(bank, "14", "ZSensor", 32, 1))
    assert s.violations == []
    assert bank in s.out_banks and bank in s.in_banks


def test_out_pair_on_bank_2_is_rejected_and_on_0_and_1_passes():
    s, link = _stub()
    s.waves["X"] = s.waves["Y"] = [0.0] * 8
    for bank in (0, 1):
        _d(s, link, P.cmd_outpair(bank, "14", "PIDSLoop.0.Setpoint", "X",
                                  "PIDSLoop.1.Setpoint", "Y", -1))
    assert s.violations == []
    _d(s, link, P.cmd_outpair(2, "14", "PIDSLoop.0.Setpoint", "X",
                              "PIDSLoop.1.Setpoint", "Y", -1))
    assert s.violations == [Violation("pair_on_bank_2", P.OUTPAIR, 2, None)]
    assert 2 not in s.out_banks


def test_in_pair_on_bank_2_is_rejected_and_a_single_channel_passes():
    s, link = _stub()
    _d(s, link, P.cmd_inwave(2, "14", "Output.A", 32, 1))
    assert s.violations == []
    _d(s, link, P.cmd_inpair(2, "14", "ZSensor", "Deflection", 32, 1))
    assert _rules(s) == ["pair_on_bank_2"]


def test_a_violating_bind_is_still_recorded_as_sent():
    s, link = _stub()
    _d(s, link, P.cmd_inwave(5, "14", "Sum", 32, 1))
    assert s.in_channels and s.in_channels[0][0] == 5
    assert s.in_banks == {}


def test_two_violations_in_one_pass_are_both_recorded():
    s, link = _stub()
    s.waves["W"] = [0.0] * 87381
    _d(s, link, P.cmd_outwave(3, "14", "Output.A", "W", -1))            # bank and length
    _d(s, link, P.cmd_inpair(2, "14", "ZSensor", "Deflection", 33, 1))  # pair and length
    assert _rules(s) == ["bank_above_2", "out_wave_over_cap",
                         "pair_on_bank_2", "in_wave_not_multiple_of_32"]


def test_xsensor_ysensor_on_different_banks_is_rejected():
    s, link = _stub()
    _d(s, link, P.cmd_inwave(0, "14", "XSensor", 32, 1))
    assert s.violations == []
    _d(s, link, P.cmd_inwave(1, "14", "YSensor", 32, 1))
    assert s.violations == [Violation("xy_sensors_split", P.INWAVE, 1, None)]


def test_xsensor_ysensor_as_one_pair_passes():
    s, link = _stub()
    _d(s, link, P.cmd_inpair(1, "14", "XSensor", "YSensor", 32, 1))
    assert s.violations == []


def test_xsensor_ysensor_split_across_pairs_is_rejected():
    s, link = _stub()
    _d(s, link, P.cmd_inpair(0, "14", "ZSensor", "XSensor", 32, 1))
    _d(s, link, P.cmd_inpair(1, "14", "YSensor", "Deflection", 32, 1))
    assert _rules(s) == ["xy_sensors_split"]


def test_xy_sensors_left_by_an_earlier_pass_do_not_count_as_a_split():
    s, link = _stub()
    _d(s, link, P.cmd_inpair(1, "14", "XSensor", "YSensor", 32, 1))
    _d(s, link, P.cmd_fire("u1", "p", "14"))
    _d(s, link, P.cmd_inpair(0, "14", "XSensor", "YSensor", 32, 1))
    assert s.violations == []
    s.stop()


# ------------------------------------------------------------------ bank model

def test_fire_plays_only_banks_bound_to_that_event():
    s, link = _stub()
    s.waves["W"] = [0.0] * 8
    _d(s, link, P.cmd_outwave(0, "3", "Output.A", "W", -1))
    _d(s, link, P.cmd_outwave(1, "14", "Output.A", "W", -1))
    _d(s, link, P.cmd_fire("u1", "p", "14"))
    assert [b for (b, _c, _w) in s.bank_fires[-1]["out"]] == [1]
    s.stop()


def test_a_litho_arm_binds_and_plays_the_five_banks_on_event_14():
    s, link = _stub()
    _d(s, link, P.cmd_armfire("u1", "o1", -1, 64, 1))
    play = s.bank_fires[-1]
    assert [b for (b, _c, _w) in play["out"]] == [0, 2]
    assert [b for (b, _c) in play["in"]] == [0, 1, 2]
    assert s.stale_plays == []
    s.stop()


def test_a_second_litho_arm_releases_and_rebinds_with_nothing_stale():
    s, link = _stub()
    _d(s, link, P.cmd_armfire("u1", "o1", -1, 64, 1))
    _d(s, link, P.cmd_armfire("u2", "o2", -1, 64, 1))
    assert s.stale_plays == []
    s.stop()


def _scan(s, link):
    _d(s, link, f"{P.SCANCHAIN} u9 XS YS 64 2 -1 1")
    s._sc_thread.join(5)


# F9 pin: the next two tests flip together. When the scan releases out-bank 2, the first
# starts failing and the strict xfail below XPASSes; then delete the first and the marker.
def test_f9_litho_out_bank_2_plays_again_on_a_later_scan_event_14():
    """Known defect F9 as the stub records it: the scan releases only out-bank 0, so the
    litho bias wave on out-bank 2 plays on every scan line."""
    s, link = _stub()
    s.waves["XS"] = s.waves["YS"] = [0.0] * 128
    _d(s, link, P.cmd_armfire("u1", "o1", -1, 64, 1))
    _scan(s, link)
    assert set(s.stale_plays) == {("14", 2, "VLitho")}
    s.stop()


@pytest.mark.xfail(strict=True, reason="F9 (SLICES): the chained scan never releases "
                   "out-bank 2, so a litho bias wave replays on the scan's Event 14")
def test_f9_no_stale_bank_plays_on_a_scan_after_a_write():
    s, link = _stub()
    s.waves["XS"] = s.waves["YS"] = [0.0] * 128
    _d(s, link, P.cmd_armfire("u1", "o1", -1, 64, 1))
    _scan(s, link)
    assert s.stale_plays == []
    s.stop()


def test_general_path_pass_after_a_litho_write_also_sees_the_stale_bank():
    """The 1.3 hazard: a transit-shaped OUTPAIR + FIRE on Event 14 after a write."""
    s, link = _stub()
    s.waves["X"] = s.waves["Y"] = [0.0] * 8
    _d(s, link, P.cmd_armfire("u1", "o1", -1, 64, 1))
    _d(s, link, P.cmd_outpair(0, "14", "PIDSLoop.0.Setpoint", "X",
                              "PIDSLoop.1.Setpoint", "Y", -1))
    _d(s, link, P.cmd_fire("u2", "move", "14"))
    assert [(b, w) for (_e, b, w) in s.stale_plays] == [(2, "VLitho")]
    s.stop()


# ---------------------------------------------------------- chained scan, ppl

def _chained(pixels):
    from afm_litho.hardware.hardware_arc import HardwareARC
    from afm_litho.service.hw_scan import run_hw_scan_chained
    from afm_litho.service.passes import ScanConfig

    cal = {"x_lvdt_sens": 7.46e-6, "x_lvdt_offset": -0.63,
           "y_lvdt_sens": 7.04e-6, "y_lvdt_offset": 0.07,
           "z_lvdt_sens": 5e-6, "z_lvdt_offset": 0.0}
    stub = FakeIgorBridge({"XSensor": 0.0, "YSensor": 0.0})
    port = stub.serve_in_thread()
    hw = HardwareARC.connect("127.0.0.1", port)
    hw.timeout = 3.0           # a rejected SCANCHAIN stalls; do not wait out 30 s

    async def broadcast(_m):
        return None

    try:
        summary = asyncio.run(run_hw_scan_chained(
            hw, ScanConfig(size_um=5.0, pixels=pixels, lines=2, line_rate_hz=200.0),
            cal, broadcast))
    finally:
        hw.close()
        stub.stop()
    return summary, stub


def test_chained_scan_with_32_pixels_passes():
    _summary, stub = _chained(32)
    assert stub.violations == []


@pytest.mark.xfail(strict=True, reason="SLICES 0.5 / F28: the chained scan arms its capture at "
                   "`ppl` unpadded (afm_litho_run.ipf:876), not a multiple of 32")
def test_chained_scan_pixel_counts_the_scan_code_does_not_pad():
    _summary, stub = _chained(6)
    assert "in_wave_not_multiple_of_32" not in _rules(stub)


def test_in_wave_of_31_and_1_points_are_rejected_and_32_passes():
    s, link = _stub()
    _d(s, link, P.cmd_inwave(0, "14", "ZSensor", 31, 1))
    _d(s, link, P.cmd_inwave(1, "14", "ZSensor", 1, 1))
    assert [v.length for v in s.violations] == [31, 1]
    _d(s, link, P.cmd_inwave(0, "14", "ZSensor", 32, 1))
    assert len(s.violations) == 2 and 0 in s.in_banks
    s.stop()


def test_scanchain_with_a_ppl_of_33_records_the_violation_and_never_finishes():
    s, link = _stub()
    s.waves["X"] = [0.0] * 66
    s.waves["Y"] = [0.0] * 66
    _d(s, link, P.cmd_scanchain("u1", "X", "Y", 33, 2, -50, 50))
    time.sleep(0.15)
    assert s.violations == [Violation("in_wave_not_multiple_of_32", P.SCANCHAIN, 0, 33)]
    assert not [x for x in link.sent if x.startswith(("ERR", "SCANDONE", "CAPTURE"))]
    s.stop()

