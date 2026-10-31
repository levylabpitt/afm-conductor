"""Hand-built end-of-line expansions, checked against audit C section 1 and the `.ipf`.

Each expected list is written from the `.ipf` (line numbers in the comments), not from the
expander. Wire lines are built with the driver's own `bridge_protocol` builders.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from afm_litho.goldens import ExpanderError, IgorExpander, expand
from afm_litho.hardware import bridge_protocol as P

CYPHER = 4
MFP = 0
SP0, SP1 = '"PIDSLoop.0.Setpoint"', '"PIDSLoop.1.Setpoint"'
CY_RETRACT = 'ir_SetPISLoop(2,"Always,Never","Deflection",-10,0,1000,0,"Height",0,150)'
MFP_RAMP_Z = 'IR_SetRamp(2,"Output.Z",80,0,"",0,0,"",0,0,"")'
FIRE14 = 'td_WriteString("Event.14","Once")'
ZERO_BIAS = 'td_WriteValue("Output.A",0)'


def wv(name: str, values) -> str:
    """Expected wave summary: values as sent on the wire (.12g), held by Igor as float32."""
    v = np.array([float(f"{x:.12g}") for x in values], dtype="<f4")
    sha = hashlib.sha256(v.tobytes()).hexdigest()[:16]
    return f"wave({name},n={len(v)},first={float(v[0]):.9g},last={float(v[-1]):.9g},sha={sha})"


def cap(name: str, n: int) -> str:
    return f"cap({name},n={n})"


X100 = [0.1 * i - 3.3 for i in range(100)]
Y100 = [2.5 - 0.07 * i for i in range(100)]
V100 = [0.0 if i % 25 == 0 else 0.4 for i in range(100)]


def litho_waves(n: int) -> list[str]:
    return [P.cmd_wave("XLitho", X100[:n]), P.cmd_wave("YLitho", Y100[:n]),
            P.cmd_wave("VLitho", V100[:n])]


def litho_release() -> list[str]:
    return ["ir_StopOutWaveBank(0)", "ir_StopOutWaveBank(2)", "ir_StopInWaveBank(0)",
            "ir_StopInWaveBank(1)", "ir_StopInWaveBank(2)"]


# ------------------------------------------------------------------ scenarios --

def sc_cypher_withdraw():
    # hw_withdraw on the Cypher: one SERVOLOOP with limits 0/150 (audit C 1.B; .ipf :570).
    return expand([P.cmd_servoloop(2, "Deflection", -10, 0, 1000, "Height", 0, 150)])


def sc_mfp_withdraw():
    return expand([P.cmd_stoploop(2), P.cmd_ramp("Output.Z", 0, 80, 2)])


def sc_safepark_cypher_wire():
    return expand([P.cmd_abort()], microscope_id=CYPHER)


def sc_safepark_cypher_com():
    return expand([("@SAFEPARK", ())], microscope_id=CYPHER)


def sc_safepark_mfp():
    return expand([P.cmd_abort()], microscope_id=MFP)


def sc_safepark_composite():
    # The live park: COM AFM_SafePark, then safe_park's own STOPLOOP / RAMP, then the vendor
    # withdraw when the verify fails (F-gaps gap 4; defect 1).
    return expand([("@SAFEPARK", ()), P.cmd_stoploop(2), P.cmd_ramp("Height", 0, 80, 2),
                   ("@DOSCANFUNC", ("StopEngageButton",))], microscope_id=CYPHER)


def sc_armfire(rc: int = 0, pair_rc: int = 0):
    return expand([*litho_waves(40), P.cmd_armfire("7", "obj_1", -3, 64, 3), ("@FILL", ())],
                  vcap_rc=rc, lateral_rc=pair_rc)


def sc_armfire_vcap_fail():
    # the pair refused, then UserIn2 alone refused too
    return sc_armfire(rc=100, pair_rc=100)


def sc_lithochain():
    return expand([*litho_waves(100), P.cmd_wave("LithoBounds", [0, 60, 100]),
                   P.cmd_lithochain("9", "obj_2", -2, 2, "LithoBounds"),
                   ("@FILL", ()), ("@FILL", ())])


def sc_lithochain_abort_at_seam():
    return expand([*litho_waves(100), P.cmd_wave("LithoBounds", [0, 60, 100]),
                   P.cmd_lithochain("9", "obj_2", -2, 2, "LithoBounds"),
                   ("@FILL", ()), P.cmd_abort(), ("@LITHOTICK", ())],
                  microscope_id=CYPHER, auto_tick=False)


SX = [0.01 * i for i in range(64)]
SY = [-1.0] * 32 + [-0.9] * 32


def sc_scanchain(err: str = "Amplitude"):
    return expand([P.cmd_wave("sx", SX), P.cmd_wave("sy", SY),
                   P.cmd_scanchain("5", "sx", "sy", 32, 2, -5, 5, err),
                   ("@FILL", ()), ("@FILL", ())])


def sc_servoloop_no_limits():
    return expand([P.cmd_servoloop(2, "Deflection", 0.5, 0, 1200, "Height")])


def sc_servoloop_limits():
    return expand([P.cmd_servoloop(2, "Amplitude", 0.8, -0.5, -1500, "Height", -10, 150),
                   P.cmd_servoloop(2, "Deflection", 0.5, 0, 1200, "Height", 0, 150)])


XS = [-1.25 + 0.05 * i for i in range(50)]
YS = [0.75 - 0.02 * i for i in range(50)]


def xy_transit_lines(uid: str = "11") -> list:
    return [P.cmd_servoloop(0, "XSensor", -1.25, 0, 1234.5, "Output.X", -10, 150),
            P.cmd_servoloop(1, "YSensor", 0.75, 0, -987.25, "Output.Y", -10, 150),
            P.cmd_wave("ow0", XS), P.cmd_wave("ow1", YS),
            P.cmd_outpair(0, "0", "PIDSLoop.0.Setpoint", "ow0", "PIDSLoop.1.Setpoint", "ow1", -4),
            P.cmd_inpair(0, "0", "XSensor", "YSensor", 64, 4),
            P.cmd_fire(uid, "pass0", "0")]


def sc_xy_transit():
    return expand([*xy_transit_lines(), ("@FILL", ())])


def sc_setmode():
    return expand([P.cmd_setmode("AC"), P.cmd_setmode("Contact"), "SETMODE ac"])


def sc_startup_and_reads():
    return expand([P.cmd_startup("1"), P.cmd_getvalue("2", "ZSensor"),
                   P.cmd_getgv("3", "XLVDTSens"), P.cmd_readxpt("4"), ("@TIPTICK", ())],
                  microscope_id=CYPHER, gv={"ImagingMode": 1})


def sc_run_end():
    return expand([("@RUNEND", ())])


ALL_SCENARIOS = [
    sc_cypher_withdraw, sc_mfp_withdraw, sc_safepark_cypher_wire, sc_safepark_cypher_com,
    sc_safepark_mfp, sc_safepark_composite, sc_armfire, sc_armfire_vcap_fail, sc_lithochain,
    sc_lithochain_abort_at_seam, sc_scanchain, sc_servoloop_no_limits, sc_servoloop_limits,
    sc_xy_transit, sc_setmode, sc_startup_and_reads, sc_run_end,
]


# ---------------------------------------------------------------- withdraw / park --

def test_cypher_withdraw():
    out = sc_cypher_withdraw()
    assert out.lines() == [CY_RETRACT]
    assert out.reads == []


def test_mfp_withdraw():
    assert sc_mfp_withdraw().lines() == ["ir_StopPISLoop(2)", MFP_RAMP_Z]


@pytest.mark.parametrize("scenario", [sc_safepark_cypher_wire, sc_safepark_cypher_com])
def test_safepark_cypher(scenario):
    # AFM_SafePark :826-832 -> AFM_Abort :729-730, the re-arm :830. No platform read.
    out = scenario()
    assert out.lines() == ["td_Stop()", ZERO_BIAS, CY_RETRACT]
    assert out.lines("read") == []
    assert out.frames == [("ABORTED", "", "")]


def test_safepark_mfp():
    # The MFP-3D parks the Cypher's way (2026-10-27).
    out = sc_safepark_mfp()
    assert out.lines() == ["td_Stop()", ZERO_BIAS, CY_RETRACT]


def test_safepark_composite_with_python_follow_up():
    assert sc_safepark_composite().lines() == [
        "td_Stop()", ZERO_BIAS, CY_RETRACT, "ir_StopPISLoop(2)",
        'IR_SetRamp(2,"Height",80,0,"",0,0,"",0,0,"")', 'DoScanFunc("StopEngageButton")']


# ------------------------------------------------------------------- litho single --

def bank2_pair(v: str, lat: str, n: int, decim: int) -> str:
    return (f'td_xSetInWavePair(2,"14","UserIn2",{cap(v, n)},"Lateral",{cap(lat, n)},'
            f'"",{decim})')


def bank2_fallback(v: str, n: int, decim: int, armed: bool) -> list[str]:
    """The .ipf ladder after a refused pair: release, UserIn2 alone, release again if refused."""
    return ["ir_StopInWaveBank(2)", f'td_xSetInWave(2,"14","UserIn2",{cap(v, n)},"",{decim})',
            *([] if armed else ["ir_StopInWaveBank(2)"])]


def armfire_expected(rc_fail: bool, lateral_fail: bool = False) -> list[str]:
    tail = bank2_fallback("VcapLitho", 64, 3, armed=not rc_fail) if rc_fail or lateral_fail else []
    return [
        *litho_release(),
        (f'td_xSetOutWavePair(0,"14",{SP0},{wv("XLitho", X100[:40])},{SP1},'
        f'{wv("YLitho", Y100[:40])},-3)'),
        f'td_xSetOutWave(2,"14","Output.A",{wv("VLitho", V100[:40])},-3)',
        (f'td_xSetInWavePair(0,"14","ZSensor",{cap("ZcapLitho", 64)},"Deflection",'
        f'{cap("DcapLitho", 64)},"AFM_LithoDone()",3)'),
        (f'td_xSetInWavePair(1,"14","XSensor",{cap("XcapLitho", 64)},"YSensor",'
        f'{cap("YcapLitho", 64)},"",3)'),
        bank2_pair("VcapLitho", "LcapLitho", 64, 3),
        *tail,
        FIRE14,
        ZERO_BIAS,
    ]


def test_armfire_single_object():
    # AFM_ArmAndFireLitho :618-666, then AFM_LithoDone :717 on the bank-0 fill.
    out = sc_armfire()
    assert out.lines() == armfire_expected(rc_fail=False)
    assert out.frames == [("CAPTURE", "7", "obj_1", ch, 64) for ch in
                          ("ZSensor", "Deflection", "XSensor", "YSensor", "UserIn2",
                           "Lateral")] + [("DONE", "7", "obj_1")]


def test_armfire_bias_readback_not_armed():
    # Pair refused and UserIn2 alone refused too (:647-656): both banks released, no
    # UserIn2 or Lateral capture frame.
    out = sc_armfire_vcap_fail()
    assert out.lines() == armfire_expected(rc_fail=True)
    assert ("CAPTURE", "7", "obj_1", "UserIn2", 64) not in out.frames
    assert ("CAPTURE", "7", "obj_1", "Lateral", 64) not in out.frames


def test_armfire_lateral_refused_falls_back_to_bias_alone():
    out = expand([*litho_waves(40), P.cmd_armfire("7", "obj_1", -3, 64, 3), ("@FILL", ())],
                 lateral_rc=100)
    assert out.lines() == armfire_expected(rc_fail=False, lateral_fail=True)
    assert out.frames == [("CAPTURE", "7", "obj_1", ch, 64) for ch in
                          ("ZSensor", "Deflection", "XSensor", "YSensor", "UserIn2")] + [
        ("DONE", "7", "obj_1")]


def test_armfire_bias_alone_failing_does_not_refuse_the_pair():
    # The .ipf's lErr is the pair call's rc alone (:647); vErr only follows a refused pair.
    out = sc_armfire(rc=100)
    assert out.lines() == armfire_expected(rc_fail=False)
    assert ("CAPTURE", "7", "obj_1", "UserIn2", 64) in out.frames
    assert ("CAPTURE", "7", "obj_1", "Lateral", 64) in out.frames


def test_armfire_drives_fixed_wave_names():
    # The wire name is a report label; the driven waves are root:XLitho/YLitho/VLitho (:623).
    with pytest.raises(ExpanderError, match="XLitho"):
        expand([P.cmd_armfire("7", "obj_1", -3, 64, 3)])
    out = expand([P.cmd_armfire("7", "obj_1", -3, 64, 3)], strict=False)
    assert "wave(XLitho,MISSING)" in out.lines()[5]


def test_armfire_capture_length_is_used_as_sent():
    # :614 Make/O/N=(captureLen): Igor does not pad; padding to 32 is the driver's job.
    out = expand([*litho_waves(40), P.cmd_armfire("7", "obj_1", -3, 50, 3)])
    assert out.lines()[7:10] == [
        (f'td_xSetInWavePair(0,"14","ZSensor",{cap("ZcapLitho", 50)},"Deflection",'
         f'{cap("DcapLitho", 50)},"AFM_LithoDone()",3)'),
        (f'td_xSetInWavePair(1,"14","XSensor",{cap("XcapLitho", 50)},"YSensor",'
         f'{cap("YcapLitho", 50)},"",3)'),
        bank2_pair("VcapLitho", "LcapLitho", 50, 3),
    ]


# -------------------------------------------------------------------- litho chain --

def chunk_expected(a: int, b: int, cap_n: int) -> list[str]:
    return [
        *litho_release(),
        (f'td_xSetOutWavePair(0,"14",{SP0},{wv("AFM_ltLX", X100[a:b])},{SP1},'
        f'{wv("AFM_ltLY", Y100[a:b])},-2)'),
        f'td_xSetOutWave(2,"14","Output.A",{wv("AFM_ltLV", V100[a:b])},-2)',
        (f'td_xSetInWavePair(0,"14","ZSensor",{cap("AFM_ltZcap", cap_n)},"Deflection",'
        f'{cap("AFM_ltDcap", cap_n)},"AFM_LithoAdvance()",2)'),
        (f'td_xSetInWavePair(1,"14","XSensor",{cap("AFM_ltXcap", cap_n)},"YSensor",'
        f'{cap("AFM_ltYcap", cap_n)},"",2)'),
        bank2_pair("AFM_ltVcap", "AFM_ltLcap", cap_n, 2),
        FIRE14,
    ]


def test_lithochain_two_chunks():
    # Chunk k slices [bounds[k], bounds[k+1]) (:1089-1099); capture padded to 32 (:1092);
    # chunk 1 is armed by AFM_LithoTick (:1217); Output.A = 0 only at the true end (:1182).
    out = sc_lithochain()
    assert out.lines() == [*chunk_expected(0, 60, 64), *chunk_expected(60, 100, 64), ZERO_BIAS]
    done = [f for f in out.frames if f[0] == "DONE"]
    assert done == [("DONE", "9", "obj_2")]
    assert len([f for f in out.frames if f[0] == "CAPTURE"]) == 12


def test_lithochain_seam_has_no_bias_write():
    lines = sc_lithochain().lines()
    seam = lines[len(chunk_expected(0, 60, 64)) - 1:len(lines) - 1]
    assert ZERO_BIAS not in seam


def test_lithochain_abort_at_seam_arms_no_further_chunk():
    out = sc_lithochain_abort_at_seam()
    assert out.lines() == [*chunk_expected(0, 60, 64), "td_Stop()", ZERO_BIAS, CY_RETRACT]
    assert out.frames[-1] == ("ABORTED", "9", "obj_2")


# -------------------------------------------------------------------- chained scan --

def line_expected(ln: int, err: str) -> list[str]:
    a = ln * 32
    return [
        "ir_StopOutWaveBank(0)", "ir_StopInWaveBank(0)", "ir_StopInWaveBank(1)",
        (f'td_xSetOutWavePair(0,"14",{SP0},{wv("AFM_scLX", SX[a:a + 32])},{SP1},'
        f'{wv("AFM_scLY", SY[a:a + 32])},-5)'),
        (f'td_xSetInWavePair(0,"14","ZSensor",{cap("AFM_scCZ", 32)},"{err}",'
        f'{cap("AFM_scCD", 32)},"AFM_ScanAdvance()",5)'),
        f'td_xSetInWave(1,"14","UserIn2",{cap("AFM_scCV", 32)},"",5)',
        FIRE14,
    ]


def test_scanchain_two_lines():
    # AFM_ScanArmLine :889-914 per line; line 1 armed by AFM_ScanTick (:978). No Output.A
    # write anywhere in a completed chained scan.
    out = sc_scanchain()
    assert out.lines() == [*line_expected(0, "Amplitude"), *line_expected(1, "Amplitude")]
    assert out.frames[-1] == ("SCANDONE", "5", 2)
    assert [f[2:4] for f in out.frames if f[0] == "CAPTURE"] == [
        ("L0", "UserIn2"), ("L0", "ZSensor"), ("L0", "Amplitude"),
        ("L1", "UserIn2"), ("L1", "ZSensor"), ("L1", "Amplitude")]


def test_scanchain_capture_length_is_ppl_unpadded():
    # :876 Make/O/N=(ppl) for slices and captures alike: 50 stays 50 (slice 0.5 records the
    # missing pad to 32 as a strict xfail; this pins today's behaviour).
    sx = [0.02 * i for i in range(100)]
    sy = [0.5 - 0.01 * i for i in range(100)]
    out = expand([P.cmd_wave("sx", sx), P.cmd_wave("sy", sy),
                  P.cmd_scanchain("5", "sx", "sy", 50, 2, -5, 5), ("@FILL", ())])
    line1 = out.lines()[7:]
    assert line1 == [
        "ir_StopOutWaveBank(0)", "ir_StopInWaveBank(0)", "ir_StopInWaveBank(1)",
        (f'td_xSetOutWavePair(0,"14",{SP0},{wv("AFM_scLX", sx[50:100])},{SP1},'
         f'{wv("AFM_scLY", sy[50:100])},-5)'),
        (f'td_xSetInWavePair(0,"14","ZSensor",{cap("AFM_scCZ", 50)},"Deflection",'
         f'{cap("AFM_scCD", 50)},"AFM_ScanAdvance()",5)'),
        f'td_xSetInWave(1,"14","UserIn2",{cap("AFM_scCV", 50)},"",5)',
        FIRE14,
    ]
    assert f'{cap("AFM_scCZ", 50)}' in out.lines()[4]


def test_scanchain_error_channel_defaults_to_deflection():
    line = P.cmd_scanchain("5", "sx", "sy", 32, 2, -5, 5).rsplit(" ", 1)[0]
    out = expand([P.cmd_wave("sx", SX), P.cmd_wave("sy", SY), line])
    assert out.lines() == line_expected(0, "Deflection")


def test_scanchain_slice_past_wave_end_clips():
    out = expand([P.cmd_wave("sx", SX[:40]), P.cmd_wave("sy", SY[:40]),
                  P.cmd_scanchain("5", "sx", "sy", 32, 2, -5, 5), ("@FILL", ())])
    line1_x = out.actuation[10].args[3]
    assert (line1_x.n, line1_x.first, line1_x.last) == (32, np.float32(SX[32]),
                                                       np.float32(SX[39]))


def test_scanchain_abort_mid_line():
    out = expand([P.cmd_wave("sx", SX), P.cmd_wave("sy", SY),
                  P.cmd_scanchain("5", "sx", "sy", 32, 2, -5, 5), P.cmd_abort()],
                 microscope_id=CYPHER)
    assert out.lines()[-3:] == ["td_Stop()", ZERO_BIAS, CY_RETRACT]
    assert out.frames == [("SCANDONE", "5", -1)]


# ------------------------------------------------------------------ servo loops --

def test_servoloop_without_limit_tokens_defaults():
    # :285-289: fewer than 9 items -> -10/150.
    assert sc_servoloop_no_limits().lines() == [
        'ir_SetPISLoop(2,"Always,Never","Deflection",0.5,0,1200,0,"Height",-10,150)']


def test_servoloop_with_limit_tokens():
    assert sc_servoloop_limits().lines() == [
        'ir_SetPISLoop(2,"Always,Never","Amplitude",0.8,-0.5,-1500,0,"Height",-10,150)',
        'ir_SetPISLoop(2,"Always,Never","Deflection",0.5,0,1200,0,"Height",0,150)']


def test_servoloop_one_limit_token_is_ignored():
    out = expand(["SERVOLOOP 2 Deflection 0.5 0 1200 Height 0"])
    assert out.lines()[0].endswith('"Height",-10,150)')


# ------------------------------------------------------------------ general path --

def test_xy_transit_event_zero():
    # Audit C 1.G; .ipf :570, :441, :486, :495, then AFM_Done :523 on fill.
    out = sc_xy_transit()
    assert out.lines() == [
        'ir_SetPISLoop(0,"Always,Never","XSensor",-1.25,0,1234.5,0,"Output.X",-10,150)',
        'ir_SetPISLoop(1,"Always,Never","YSensor",0.75,0,-987.25,0,"Output.Y",-10,150)',
        f'td_xSetOutWavePair(0,"0",{SP0},{wv("ow0", XS)},{SP1},{wv("ow1", YS)},-4)',
        (f'td_xSetInWavePair(0,"0","XSensor",{cap("cap_XSensor", 64)},"YSensor",'
        f'{cap("cap_YSensor", 64)},"AFM_Done()",4)'),
        'td_WriteString("Event.0","Once")',
        ZERO_BIAS,
    ]
    assert out.frames == [("CAPTURE", "11", "pass0", "XSensor", 64),
                          ("CAPTURE", "11", "pass0", "YSensor", 64), ("DONE", "11", "pass0")]


def test_general_pass_two_banks_completes_once():
    out = expand([P.cmd_inpair(0, "0", "ZSensor", "Deflection", 64, 2),
                  P.cmd_inwave(1, "0", "Output.A", 64, 2), P.cmd_fire("3", "pass1", "0"),
                  ("@FILL", ())])
    assert out.lines().count(ZERO_BIAS) == 1
    assert out.lines()[1] == (f'td_xSetInWave(1,"0","Output.A",{cap("cap_Output_A", 64)},'
                              f'"AFM_Done()",2)')


def test_bare_fire_completes_at_once():
    # AFM_FireEvent :502-504: no capture recorded -> AFM_Done right after the fire.
    out = expand([P.cmd_fire("3", "bare", "0")])
    assert out.lines() == ['td_WriteString("Event.0","Once")', ZERO_BIAS]
    with pytest.raises(ExpanderError):
        IgorExpander().run([P.cmd_fire("3", "bare", "0"), ("@FILL", ())])


def test_aborted_general_pass_leaves_capture_record():
    # AFM_Abort does not clear root:AFM_capChans, so a later bare FIRE no longer
    # completes at once and its bias zero (:503 -> :523) does not happen.
    out = expand([*xy_transit_lines(), P.cmd_abort(), P.cmd_fire("12", "bare", "0")],
                 microscope_id=CYPHER)
    assert out.lines()[-1] == 'td_WriteString("Event.0","Once")'


def test_wave_and_waveapp_build_one_wave():
    full = [0.5 * i for i in range(20)]
    out = expand([P.cmd_wave("w", full[:8]), P.cmd_waveapp("w", full[8:]),
                  P.cmd_outwave(1, "0", "Output.A", "w", -1)])
    assert out.lines() == [f'td_xSetOutWave(1,"0","Output.A",{wv("w", full)},-1)']


# ---------------------------------------------------------------- mode, reads, run --

def test_setmode():
    # AFM_SetMode :1252-1256: the compare is case-sensitive, so "ac" selects Contact.
    assert sc_setmode().lines() == [
        'MainPopupFunc("ImagingModePopup_0",2,"AC Mode")',
        'MainPopupFunc("ImagingModePopup_0",1,"Contact")',
        'MainPopupFunc("ImagingModePopup_0",1,"Contact")']


def test_reads_go_to_the_reads_list():
    out = sc_startup_and_reads()
    assert out.actuation == []
    reads = out.lines("read")
    assert reads[:7] == ['GV("ScanSize")', 'GV("ScanRate")', 'GV("ScanPoints")',
                         'GV("ScanLines")', 'GV("ScanAngle")', 'GV("ImagingMode")',
                         'GV("ImagingMode")']
    assert reads[7] == 'GV("AmplitudeSetpointVolts")'
    assert reads[-10:] == [
        'GV("MicroscopeID")', 'td_ReadValue("ZSensor")', 'GV("XLVDTSens")',
        'td_ReadGroup("Crosspoint",cap(AFM_xptTmp,n=0))',
        'td_ReadValue("XSensor")', 'td_ReadValue("YSensor")', 'td_ReadValue("Height")',
        'td_ReadValue("Deflection")', 'td_ReadValue("DetectorSum")',
        'td_ReadValue("amplitude")']
    assert [e.tag for e in out.timeline][-6:] == ["tip"] * 6


def test_tip_reads_do_not_change_actuation():
    base = sc_armfire().lines()
    noisy = expand([*litho_waves(40), ("@TIPTICK", ()), P.cmd_armfire("7", "obj_1", -3, 64, 3),
                    ("@TIPTICK", ()), ("@FILL", ()), ("@TIPTICK", ())])
    assert noisy.lines() == base
    assert len(noisy.reads) == 18


def test_dry_run_commands_nothing():
    out = expand([("@DRYRUN", ("1",)), *litho_waves(40), P.cmd_armfire("7", "o", -3, 64, 3),
                  P.cmd_abort(), P.cmd_getvalue("8", "Height")])
    assert out.actuation == []
    assert out.lines("read") == ['td_ReadValue("Height")']
    assert out.frames == [("DONE", "7", "o"), ("ABORTED", "", ""), ("VALUE", "8")]


def test_run_end():
    assert sc_run_end().lines() == [ZERO_BIAS]
    with pytest.raises(ExpanderError):
        expand([("@RUNEND", ()), P.cmd_stoploop(2)])
    assert expand([("@RUNEND", ()), ("@TIPTICK", ())]).reads == []


def test_run_end_does_not_stop_a_litho_chain():
    # AFM_LithoRunEnd :132-141 stops AFMdispatch and AFMtip only; AFMlitho re-arms chunk 1.
    out = expand([*litho_waves(100), P.cmd_wave("LithoBounds", [0, 60, 100]),
                  P.cmd_lithochain("9", "obj_2", -2, 2, "LithoBounds"), ("@RUNEND", ()),
                  ("@FILL", ())])
    assert out.lines()[-len(chunk_expected(60, 100, 64)):] == chunk_expected(60, 100, 64)


def test_unknown_verb_and_pseudo():
    with pytest.raises(ExpanderError):
        expand(["NOPE 1 2"])
    out = expand(["NOPE 1 2"], strict=False)       # Igor's own reply to an unknown verb
    assert out.actuation == [] and out.frames == [("ERR", "-", "unknown verb: NOPE")]
    with pytest.raises(ExpanderError):
        expand([("@NOPE", ())])


def test_missing_chain_waves_arm_nothing():
    out = expand([P.cmd_scanchain("5", "sx", "sy", 32, 2, -5, 5),
                  P.cmd_lithochain("9", "o", -2, 2, "LithoBounds")])
    assert out.actuation == []
    assert [f[0] for f in out.frames] == ["ERR", "ERR"]


def test_sites_are_recorded():
    out = sc_armfire()
    assert out.actuation[0].site == "igor/afm_litho_run.ipf:634"
    assert out.actuation[-1].site.startswith("igor/afm_litho_run.ipf:554 (via :725)")
