"""The `.ipf` call table: what each dispatched verb and each bridge function calls, in order.

Derived line by line from `igor/afm_litho_run.ipf` and the two files it includes. Each `Step` is
one call written in a function body, in source order, with its line number:

  act    a vendor call that commands the controller (td_ write/arm/fire/stop, ir_, IR_SetRamp,
         MainPopupFunc)
  read   a vendor read (td_ReadValue, td_ReadGroup, GV)
  afm    a call to another bridge function (followed by `resolve`)
  xport  a SOCKIT transport call

Each argument is marked by where its value comes from: `wire` (a token of the dispatched line, or
a wave the wire loaded), `igor` (computed Igor-side), `literal` (written in the `.ipf`; `note` is
the exact source text), `state` (Igor/controller state the wire does not decide), or `param`
(the enclosing function's parameter; `resolve` replaces it with the caller's argument).

`tests/test_igor_expander_table.py` checks every entry against the `.ipf` text: call names and
order per body, line numbers, argument counts, literal text and parameter lists, plus a pinned
hash per file so any `.ipf` edit forces the table to be re-checked.
"""

from __future__ import annotations

from dataclasses import dataclass

RUN = "igor/afm_litho_run.ipf"
IGOR = "igor/afm_igor.ipf"
SOCK = "igor/afm_sockit.ipf"

# sha256 of each file with CRLF normalised to LF, so a checkout's line-ending setting does not
# matter. A mismatch means the .ipf changed: re-derive the affected entries, then re-pin.
IPF_SHA256 = {
    RUN: "303ff69ebf767486f2e20c745fc4c75f705244d50be9a5184a497dbddc306f8d",
    IGOR: "f872d554da9633006288fe88b670d40929d005ce030f19ecb5918e5855adfd05",
    SOCK: "e7041374992885d45d699caa8cc94f3b91732ff302047dd6e7b0899ea09f6cf0",
}

IPF_CONSTANTS = {
    "kBiasInChannel": ("UserIn2", f"{RUN}:73"),
    "kLateralInChannel": ("Lateral", f"{RUN}:77"),
    "kEvent": ("14", f"{RUN}:81"),
    "kCapChunkPoints": (8192, f"{RUN}:87"),
    "cMicroscopeCypher": (4, "vendor HardWare.ipf:10, cited at afm_litho_run.ipf:790"),
}


@dataclass(frozen=True)
class Arg:
    src: str
    note: str

    def __str__(self) -> str:
        return f"{self.src}:{self.note}"


@dataclass(frozen=True)
class Step:
    name: str
    line: int
    kind: str
    args: tuple[Arg, ...] = ()
    when: str = ""


@dataclass(frozen=True)
class FnEntry:
    name: str
    file: str
    start: int
    end: int
    params: tuple[str, ...]
    steps: tuple[Step, ...]
    note: str = ""


@dataclass(frozen=True)
class VerbEntry:
    verb: str
    line: int
    steps: tuple[Step, ...]
    note: str = ""


def W(note: str) -> Arg:
    return Arg("wire", note)


def I(note: str) -> Arg:
    return Arg("igor", note)


def L(text: str) -> Arg:
    return Arg("literal", text)


def S(note: str) -> Arg:
    return Arg("state", note)


def P(name: str) -> Arg:
    return Arg("param", name)


def act(name: str, line: int, *args: Arg, when: str = "") -> Step:
    return Step(name, line, "act", args, when)


def read(name: str, line: int, *args: Arg, when: str = "") -> Step:
    return Step(name, line, "read", args, when)


def afm(name: str, line: int, *args: Arg, when: str = "") -> Step:
    return Step(name, line, "afm", args, when)


def xport(name: str, line: int, *args: Arg, when: str = "") -> Step:
    return Step(name, line, "xport", args, when)


SOCK_ARG = I("root:AFM_lrSock")
REPLY = I("reply text")
EV14 = L("kEvent")
FIRE14 = I('"Event." + kEvent')
OUT_A = L('"Output.A"')
BIAS_IN = L("kBiasInChannel")        # "UserIn2": the measured tip bias
LATERAL_IN = L("kLateralInChannel")  # "Lateral": its bank-2 pair partner
SP0 = L('"PIDSLoop.0.Setpoint"')
SP1 = L('"PIDSLoop.1.Setpoint"')


def _send(line: int, when: str = "") -> Step:
    return afm("AFM_SendStr", line, SOCK_ARG, REPLY, when=when)


def _frame(line: int, when: str = "") -> Step:
    """A pass frame: wire v3 for the PLAY in flight (AFM_SendRunFrame), else as today."""
    return afm("AFM_SendRunFrame", line, SOCK_ARG, REPLY, when=when)


def _cap(line: int, chan: Arg, path: Arg | str, when: str = "") -> Step:
    path_arg = L(path) if isinstance(path, str) else path
    return afm("AFM_SendCapture", line, SOCK_ARG, I("pass uid global"), I("report name"),
               chan, path_arg, when=when)


# ------------------------------------------------------------ AFM_Dispatch cases --
# Live cases (:269-374). The `.ipf` reads tokens with StringFromList(i, line, " ").

VERBS: dict[str, VerbEntry] = {e.verb: e for e in (
    VerbEntry("STARTUP", 270, (
        _send(272),
        afm("AFM_MacroStartup", 272),
    ), "reply STARTUP <tok1> proto=2;<startup values>"),
    VerbEntry("GETVALUE", 274, (
        afm("AFM_GetValue", 276, W("tok 2")),
        _send(277),
    )),
    VerbEntry("GETGV", 279, (
        _send(281),
        afm("AFM_GetGVTok", 281, W("tok 2")),
    )),
    VerbEntry("SETVALUE", 283, (
        afm("AFM_SetValue", 284, W("tok 1"), W("str2num(tok 2)")),
    )),
    VerbEntry("SERVOLOOP", 286, (
        afm("AFM_SetupServoLoop", 300, W("str2num(tok 1)"), W("tok 2"), W("str2num(tok 3)"),
            W("str2num(tok 4)"), W("str2num(tok 5)"), W("tok 6"),
            W("str2num(tok 7) if ItemsInList(line) >= 9, else literal -10 (:295-298)"),
            W("str2num(tok 8) if ItemsInList(line) >= 9, else literal 150 (:295-298)")),
    )),
    VerbEntry("STOPLOOP", 302, (
        afm("AFM_StopLoop", 303, W("str2num(tok 1)")),
    )),
    VerbEntry("RAMP", 305, (
        afm("AFM_RampOutput", 306, W("tok 1"), W("str2num(tok 2)"), W("str2num(tok 3)"),
            W("str2num(tok 4)")),
    )),
    VerbEntry("WAVE", 308, (
        afm("AFM_StoreWave", 309, W("tok 1"), W("tok 2")),
    )),
    VerbEntry("WAVEAPP", 311, (
        afm("AFM_AppendWave", 312, W("tok 1"), W("tok 2")),
    )),
    VerbEntry("ARMFIRE", 314, (
        afm("AFM_ArmAndFireLitho", 317, W("tok 2"), W("str2num(tok 3)"), W("str2num(tok 4)"),
            W("str2num(tok 5)")),
    ), "tok 1 (uid) -> root:AFM_lrUID"),
    VerbEntry("OUTWAVE", 319, (
        afm("AFM_ArmOutWave", 320, W("str2num(tok 1)"), W("tok 2"), W("tok 3"), W("tok 4"),
            W("str2num(tok 5)")),
    )),
    VerbEntry("OUTPAIR", 322, (
        afm("AFM_ArmOutPair", 323, W("str2num(tok 1)"), W("tok 2"), W("tok 3"), W("tok 4"),
            W("tok 5"), W("tok 6"), W("str2num(tok 7)")),
    )),
    VerbEntry("INWAVE", 325, (
        afm("AFM_ArmInWave", 326, W("str2num(tok 1)"), W("tok 2"), W("tok 3"),
            W("str2num(tok 4)"), W("str2num(tok 5)")),
    )),
    VerbEntry("INPAIR", 328, (
        afm("AFM_ArmInPair", 329, W("str2num(tok 1)"), W("tok 2"), W("tok 3"), W("tok 4"),
            W("str2num(tok 5)"), W("str2num(tok 6)")),
    )),
    VerbEntry("FIRE", 331, (
        afm("AFM_FireEvent", 334, W("tok 2"), W("tok 3")),
    ), "tok 1 (uid) -> root:AFM_lrUID"),
    VerbEntry("SCANCHAIN", 336, (
        afm("AFM_ScanChainStart", 345, W("tok 2"), W("tok 3"), W("str2num(tok 4)"),
            W("str2num(tok 5)"), W("str2num(tok 6)"), W("str2num(tok 7)"),
            W("errChan = tok 8, or \"Deflection\" when absent (:341-344)")),
    ), "tok 1 (uid) -> root:AFM_scUID"),
    VerbEntry("LITHOCHAIN", 347, (
        afm("AFM_LithoChainStart", 350, W("tok 2"), W("tok 3"), W("tok 4"), W("tok 5"),
            W("tok 6"), W("str2num(tok 7)"), W("str2num(tok 8)")),
    ), "tok 1 (uid) -> root:AFM_lrUID"),
    VerbEntry("SHOWTIP", 352, (
        afm("AFM_ShowTip", 353, I('cmpstr(tok 1, "ON", 1) == 0 (case-sensitive)')),
    )),
    VerbEntry("SETMODE", 355, (
        afm("AFM_SetMode", 358, W("tok 1")),
    )),
    VerbEntry("READXPT", 360, (
        afm("AFM_ReadXPT", 364, W("tok 1"), W("tok 2")),
    )),
    VerbEntry("ABORT", 366, (
        afm("AFM_SafePark", 367),
    )),
    VerbEntry("ABORTCONFIRM", 369, (
        afm("AFM_AbortConfirmed", 370),
    )),
    VerbEntry("ABORTFAULT", 372, (
        _send(373),
        afm("AFM_ReadAbortFault", 373),
    ), "reply VALUE <tok1> <Igor's fault flag 0|1>; read only"),
    VerbEntry("CLEARFAULT", 375, (
        afm("AFM_V3Dispatch", 377, W("verb"), W("line"), when="CLEARFAULT <seq>: wire v3"),
        afm("AFM_ClearFault", 379, when="no seq: v2"),
    ), "the operator cleared the fault: root:AFM_abortStopped and root:AFM_abortFault to 0"),
    VerbEntry("default", 382, (
        afm("AFM_DispatchOther", 383, SOCK_ARG, W("verb"), W("line")),
    ), "wire v3 verbs and GETSTR are answered there; any other verb -> ERR - unknown verb"),
)}

# Dry-run cases (:233-267): nothing is commanded; reads still answer.
DRY_VERBS: dict[str, VerbEntry] = {e.verb: e for e in (
    VerbEntry("STARTUP", 236, (_send(237), afm("AFM_MacroStartup", 237))),
    VerbEntry("GETVALUE", 239, (afm("AFM_GetValue", 240, W("tok 2")), _send(241))),
    VerbEntry("GETGV", 243, (_send(244), afm("AFM_GetGVTok", 244, W("tok 2")))),
    VerbEntry("READXPT", 246, (afm("AFM_ReadXPT", 248, W("tok 1"), W("tok 2")),)),
    VerbEntry("FIRE", 250, (_send(251),), "reply DONE <tok1> <tok2>"),
    VerbEntry("ARMFIRE", 253, (_send(254),), "reply DONE <tok1> <tok2>"),
    VerbEntry("ABORT", 256, (_send(257),), "reply ABORTED; no stop, no park"),
    VerbEntry("SCANCHAIN", 259, (_send(260),), "reply SCANDONE <tok1> 0"),
    VerbEntry("LITHOCHAIN", 262, (_send(263),), "reply DONE <tok1> <tok2>"),
)}

_RAMP_TAIL = (L('""'), L("0"), L("0"), L('""'), L("0"), L("0"), L('""'))

# ------------------------------------------------------------------- functions --

FUNCTIONS: dict[str, FnEntry] = {e.name: e for e in (
    FnEntry("AFM_LithoRunStart", RUN, 103, 139, ("host", "port"), (
        afm("AFM_SockConnect", 111, P("host"), P("port")),
        afm("AFM_V3LinkOpened", 116),
        afm("AFM_TipDesiredPeriod", 134),
    ), "starts AFMdispatch (period 3) and AFMtip; run by hand"),
    FnEntry("AFM_LithoRunEnd", RUN, 141, 150, (), (
        afm("AFM_SetValue", 144, OUT_A, L("0")),
        afm("AFM_SockClose", 147, I("root:AFM_lrSock")),
    ), "stops AFMdispatch and AFMtip only; AFMscan / AFMlitho keep running"),
    FnEntry("AFM_LithoRunTrace", RUN, 161, 166, ("on",), ()),
    FnEntry("AFM_LithoRunDryRun", RUN, 168, 176, ("on",), ()),
    FnEntry("AFM_DispatchTick", RUN, 184, 206, ("s",), (
        xport("SOCKITpeek", 190, I("root:AFM_lrSock")),
        afm("AFM_V3Intake", 193, I("the buffer")),
        afm("AFM_DispatchFromSocket", 203, I("one whole line from the buffer")),
    )),
    FnEntry("AFM_Dispatch", RUN, 210, 386, ("line",), (),
            "per-case steps: DRY_VERBS then VERBS"),
    FnEntry("AFM_StoreWave", RUN, 388, 412, ("name", "body"), (),
            "Make/O/N=(n) root:$name, single precision; ';'-separated str2num values"),
    FnEntry("AFM_AppendWave", RUN, 419, 431, ("name", "body"), (
        afm("AFM_StoreWave", 423, P("name"), P("body"), when="wave does not exist"),
        afm("AFM_StoreWave", 425, L('"AFM_appTmp"'), P("body")),
    )),
    FnEntry("AFM_ArmOutWave", RUN, 442, 447,
            ("bank", "event", "channel", "name", "interp"), (
        act("td_xSetOutWave", 446, P("bank"), P("event"), P("channel"),
            W("wave root:$name (values from WAVE/WAVEAPP)"), P("interp")),
    )),
    FnEntry("AFM_ArmOutPair", RUN, 450, 456,
            ("bank", "event", "chA", "nameA", "chB", "nameB", "interp"), (
        act("td_xSetOutWavePair", 455, P("bank"), P("event"), P("chA"),
            W("wave root:$nameA"), P("chB"), W("wave root:$nameB"), P("interp")),
    )),
    FnEntry("AFM_CapWaveName", RUN, 460, 466, ("channel",), ()),
    FnEntry("AFM_CapRecord", RUN, 469, 476, ("channel", "waveName"), ()),
    FnEntry("AFM_ArmInWave", RUN, 480, 488, ("bank", "event", "channel", "capLen", "decim"), (
        afm("AFM_CapWaveName", 483, P("channel")),
        afm("AFM_CapRecord", 486, P("channel"), I("wn")),
        act("td_xSetInWave", 487, P("bank"), P("event"), P("channel"),
            I("Make/O/N=(capLen) root:cap_<channel>"), L('"AFM_Done()"'), P("decim")),
    )),
    FnEntry("AFM_ArmInPair", RUN, 491, 501, ("bank", "event", "chA", "chB", "capLen", "decim"), (
        afm("AFM_CapWaveName", 494, P("chA")),
        afm("AFM_CapWaveName", 494, P("chB")),
        afm("AFM_CapRecord", 498, P("chA"), I("wnA")),
        afm("AFM_CapRecord", 499, P("chB"), I("wnB")),
        act("td_xSetInWavePair", 500, P("bank"), P("event"), P("chA"),
            I("Make/O/N=(capLen) root:cap_<chA>"), P("chB"),
            I("Make/O/N=(capLen) root:cap_<chB>"), L('"AFM_Done()"'), P("decim")),
    )),
    FnEntry("AFM_FireEvent", RUN, 505, 520, ("name", "event"), (
        act("td_WriteString", 509, I('"Event." + event'), L('"Once"')),
        afm("AFM_Done", 517, when="no capture channel recorded (root:AFM_capChans empty)"),
    )),
    FnEntry("AFM_Done", RUN, 526, 546, (), (
        afm("AFM_SetValue", 537, OUT_A, L("0"), when="first call since the FIRE"),
        _cap(540, I("each recorded channel"), I("root:<recorded wave>"), when="per channel"),
        _frame(542),
    ), "in-bank fill callback of the general path; guarded by root:AFM_doneSent"),
    FnEntry("AFM_SetValue", RUN, 551, 555, ("param", "value"), (
        act("td_WriteValue", 554, P("param"), P("value")),
    )),
    FnEntry("AFM_GetValue", RUN, 558, 561, ("param",), (
        read("td_ReadValue", 560, P("param")),
    )),
    FnEntry("AFM_GetGVTok", RUN, 566, 573, ("name",), (
        read("GV", 568, P("name")),
        afm("AFM_TelemTok", 572, I("v")),
    )),
    FnEntry("AFM_SetupServoLoop", RUN, 581, 587,
            ("loop", "inChan", "setpoint", "pgain", "igain", "outChan", "outMin", "outMax"), (
        act("ir_SetPISLoop", 584, P("loop"), L('"Always,Never"'), P("inChan"), P("setpoint"),
            P("pgain"), P("igain"), L("0"), P("outChan"), P("outMin"), P("outMax")),
        afm("AFM_V3NoteLoop", 585, P("loop"), P("setpoint"), P("outMin"), P("outMax")),
    )),
    FnEntry("AFM_StopLoop", RUN, 592, 595, ("loop",), (
        act("ir_StopPISLoop", 594, P("loop")),
    )),
    FnEntry("AFM_RampOutput", RUN, 599, 605, ("chan", "dest", "rate", "maxsec"), (
        act("IR_SetRamp", 603, P("maxsec"), P("chan"), P("rate"), P("dest"), *_RAMP_TAIL),
    )),
    FnEntry("AFM_ArmAndFireLitho", RUN, 626, 674,
            ("name", "interp", "captureLen", "decimation"), (
        act("ir_StopOutWaveBank", 634, L("0")),
        act("ir_StopOutWaveBank", 635, L("2")),
        act("ir_StopInWaveBank", 636, L("0")),
        act("ir_StopInWaveBank", 637, L("1")),
        act("ir_StopInWaveBank", 638, L("2")),
        act("td_xSetOutWavePair", 639, L("0"), EV14, SP0, W("wave root:XLitho"), SP1,
            W("wave root:YLitho"), P("interp")),
        act("td_xSetOutWave", 641, L("2"), EV14, OUT_A, W("wave root:VLitho"), P("interp")),
        act("td_xSetInWavePair", 643, L("0"), EV14, L('"ZSensor"'),
            I("root:ZcapLitho, captureLen points"), L('"Deflection"'),
            I("root:DcapLitho, captureLen points"), L('"AFM_LithoDone()"'), P("decimation")),
        act("td_xSetInWavePair", 645, L("1"), EV14, L('"XSensor"'),
            I("root:XcapLitho, captureLen points"), L('"YSensor"'),
            I("root:YcapLitho, captureLen points"), L('""'), P("decimation")),
        act("td_xSetInWavePair", 655, L("2"), EV14, BIAS_IN,
            I("root:VcapLitho, captureLen points"), LATERAL_IN,
            I("root:LcapLitho, captureLen points"), L('""'), P("decimation")),
        act("ir_StopInWaveBank", 659, L("2"), when="lErr != 0 (state: vendor return code)"),
        act("td_xSetInWave", 661, L("2"), EV14, BIAS_IN,
            I("root:VcapLitho, captureLen points"), L('""'), P("decimation"),
            when="lErr != 0 (pair refused: UserIn2 alone)"),
        act("ir_StopInWaveBank", 664, L("2"),
            when="lErr != 0 and vErr != 0 (state: vendor return code)"),
        act("td_WriteString", 671, FIRE14, L('"Once"')),
    ), "the wire name is only the report name; the driven waves are fixed"),
    FnEntry("AFM_SendCapture", RUN, 690, 715, ("sock", "uid", "name", "channel", "wavePath"), (
        _frame(711, when="per frame of at most kCapChunkPoints points"),
    )),
    FnEntry("AFM_LithoDone", RUN, 721, 740, (), (
        afm("AFM_SetValue", 725, OUT_A, L("0")),
        _cap(726, L('"ZSensor"'), '"root:ZcapLitho"'),
        _cap(727, L('"Deflection"'), '"root:DcapLitho"'),
        _cap(728, L('"XSensor"'), '"root:XcapLitho"'),
        _cap(729, L('"YSensor"'), '"root:YcapLitho"'),
        _cap(732, BIAS_IN, '"root:VcapLitho"', when="root:AFM_VcapArmed"),
        _cap(736, LATERAL_IN, '"root:LcapLitho"', when="root:AFM_LcapArmed"),
        _frame(738),
    ), "ARMFIRE bank-0 fill callback"),
    FnEntry("AFM_Abort", RUN, 747, 786, (), (
        afm("AFM_PassMayPlay", 755, when="root:AFM_abortStopped is set"),
        act("td_Stop", 756, when="root:AFM_abortStopped is clear, or root:AFM_passArmed is set"),
        afm("AFM_StopRan", 757, when="td_Stop ran"),
        afm("AFM_SetValue", 759, OUT_A, L("0")),
        afm("AFM_GlobalStr", 765, L('"root:AFM_scUID"'), L('""'),
            when="a chained scan is active"),
        _frame(766, when="a chained scan is active: SCANDONE <scUID> -1"),
        afm("AFM_GlobalStr", 779, L('"root:AFM_lrUID"'), L('""'),
            when="otherwise, root:AFM_lrName exists"),
        _frame(780, when="otherwise, root:AFM_lrName exists: ABORTED <lrUID> <lrName>"),
    ), "sets scAbort/ltAbort = 1 and scPending/ltPending = 0 first"),
    FnEntry("AFM_AbortConfirmed", RUN, 803, 808, (), (),
            "clears root:AFM_abortStopped and stops AFMabortBackup"),
    FnEntry("AFM_ReadAbortFault", RUN, 811, 814, (), (),
            "returns root:AFM_abortFault (read only)"),
    FnEntry("AFM_ClearFault", RUN, 817, 823, (), (),
            "clears root:AFM_abortStopped and root:AFM_abortFault, stops AFMabortBackup"),
    FnEntry("AFM_SafePark", RUN, 836, 843, (), (
        afm("AFM_Abort", 837),
        act("ir_SetPISLoop", 839, L("2"), L('"Always,Never"'), L('"Deflection"'), L("-10"),
            L("0"), L("1000"), L("0"), L('"Height"'), L("0"), L("150")),
        afm("AFM_ArmAbortBackup", 840),
        afm("AFM_V3ParkEnds", 841),
    )),
    FnEntry("AFM_ScanChainStart", RUN, 856, 894,
            ("xname", "yname", "ppl", "nlines", "interp", "decim", "errChan"), (
        _frame(868, when="xname or yname wave missing: ERR, nothing armed"),
        afm("AFM_GlobalStr", 868, L('"root:AFM_scUID"'), L('"-"'),
            when="xname or yname wave missing"),
        afm("AFM_ScanArmLine", 892, L("0")),
    ), "starts AFMscan (period 1, proc AFM_ScanTick)"),
    FnEntry("AFM_ScanArmLine", RUN, 898, 948, ("ln",), (
        act("ir_StopOutWaveBank", 921, L("0")),
        act("ir_StopInWaveBank", 922, L("0")),
        act("ir_StopInWaveBank", 923, L("1")),
        act("td_xSetOutWavePair", 924, L("0"), EV14, SP0,
            I("root:AFM_scLX = x wave[ln*ppl + p], ppl points"), SP1,
            I("root:AFM_scLY = y wave[ln*ppl + p], ppl points"), W("interp (SCANCHAIN tok 6)")),
        act("td_xSetInWavePair", 930, L("0"), EV14, L('"ZSensor"'),
            I("root:AFM_scCZ, ppl points"), W("errChan (SCANCHAIN tok 8 or Deflection)"),
            I("root:AFM_scCD, ppl points"), L('"AFM_ScanAdvance()"'),
            W("decim (SCANCHAIN tok 7)")),
        act("td_xSetInWave", 938, L("1"), EV14, BIAS_IN, I("root:AFM_scCV, ppl points"), L('""'),
            W("decim (SCANCHAIN tok 7)")),
        act("ir_StopInWaveBank", 941, L("1"), when="vErr != 0 (state: vendor return code)"),
        act("td_WriteString", 946, FIRE14, L('"Once"')),
    ), "capture length is ppl, not padded to 32"),
    FnEntry("AFM_ScanAdvance", RUN, 954, 985, (), (
        _cap(965, BIAS_IN, '"root:AFM_scCV"', when="root:AFM_scVArmed"),
        _cap(967, L('"ZSensor"'), '"root:AFM_scCZ"'),
        _cap(972, I("errChan"), '"root:AFM_scCD"'),
        _frame(978, when="last line or abort, scan active: SCANDONE <uid> <lines done>"),
    ), "SCANCHAIN bank-0 fill callback; sets AFM_scPending, arms nothing"),
    FnEntry("AFM_ScanTick", RUN, 991, 1012, ("s",), (
        afm("AFM_ScanArmLine", 1010, I("root:AFM_scLine"), when="pending, active, no abort"),
    )),
    FnEntry("AFM_LithoCapPoints", RUN, 1035, 1041, ("n",), (),
            "max(n, 1) padded up to a multiple of 32"),
    FnEntry("AFM_LithoChainStart", RUN, 1044, 1079,
            ("name", "xname", "yname", "vname", "boundsName", "interp", "decim"), (
        _frame(1054, when="any of the four waves missing: ERR, nothing armed"),
        afm("AFM_GlobalStr", 1054, L('"root:AFM_lrUID"'), L('"-"'),
            when="any of the four waves missing"),
        afm("AFM_LithoArmChunk", 1077, L("0")),
    ), "starts AFMlitho (period 1, proc AFM_LithoTick); nChunks = numpnts(bounds) - 1"),
    FnEntry("AFM_LithoArmChunk", RUN, 1088, 1161, ("k",), (
        afm("AFM_LithoCapPoints", 1103, I("m = bounds[k+1] - bounds[k]")),
        act("ir_StopOutWaveBank", 1120, L("0")),
        act("ir_StopOutWaveBank", 1121, L("2")),
        act("ir_StopInWaveBank", 1122, L("0")),
        act("ir_StopInWaveBank", 1123, L("1")),
        act("ir_StopInWaveBank", 1124, L("2")),
        act("td_xSetOutWavePair", 1125, L("0"), EV14, SP0,
            I("root:AFM_ltLX = XLitho[bounds[k] + p], m points"), SP1,
            I("root:AFM_ltLY = YLitho[bounds[k] + p], m points"),
            W("interp (LITHOCHAIN tok 7)")),
        act("td_xSetOutWave", 1127, L("2"), EV14, OUT_A,
            I("root:AFM_ltLV = VLitho[bounds[k] + p], m points"),
            W("interp (LITHOCHAIN tok 7)")),
        act("td_xSetInWavePair", 1129, L("0"), EV14, L('"ZSensor"'),
            I("root:AFM_ltZcap, capLen points"), L('"Deflection"'),
            I("root:AFM_ltDcap, capLen points"), L('"AFM_LithoAdvance()"'),
            W("decim (LITHOCHAIN tok 8)")),
        act("td_xSetInWavePair", 1131, L("1"), EV14, L('"XSensor"'),
            I("root:AFM_ltXcap, capLen points"), L('"YSensor"'),
            I("root:AFM_ltYcap, capLen points"), L('""'), W("decim (LITHOCHAIN tok 8)")),
        act("td_xSetInWavePair", 1139, L("2"), EV14, BIAS_IN,
            I("root:AFM_ltVcap, capLen points"), LATERAL_IN,
            I("root:AFM_ltLcap, capLen points"), L('""'), W("decim (LITHOCHAIN tok 8)")),
        act("ir_StopInWaveBank", 1143, L("2"), when="lErr != 0 (state: vendor return code)"),
        act("td_xSetInWave", 1147, L("2"), EV14, BIAS_IN,
            I("root:AFM_ltVcap, capLen points"), L('""'), W("decim (LITHOCHAIN tok 8)"),
            when="lErr != 0 (pair refused: UserIn2 alone)"),
        act("ir_StopInWaveBank", 1150, L("2"),
            when="lErr != 0 and vErr != 0 (state: vendor return code)"),
        act("td_WriteString", 1158, FIRE14, L('"Once"')),
    )),
    FnEntry("AFM_LithoAdvance", RUN, 1168, 1203, (), (
        _cap(1180, BIAS_IN, '"root:AFM_ltVcap"', when="root:AFM_ltVArmed"),
        _cap(1184, LATERAL_IN, '"root:AFM_ltLcap"', when="root:AFM_ltLArmed"),
        _cap(1186, L('"ZSensor"'), '"root:AFM_ltZcap"'),
        _cap(1187, L('"Deflection"'), '"root:AFM_ltDcap"'),
        _cap(1188, L('"XSensor"'), '"root:AFM_ltXcap"'),
        _cap(1189, L('"YSensor"'), '"root:AFM_ltYcap"'),
        afm("AFM_SetValue", 1193, OUT_A, L("0"), when="last chunk or abort (true end)"),
        _frame(1196, when="last chunk or abort, chain active: DONE <uid> <name>"),
    ), "LITHOCHAIN bank-0 fill callback; no Output.A write at a seam"),
    FnEntry("AFM_LithoTick", RUN, 1209, 1230, ("s",), (
        afm("AFM_LithoArmChunk", 1228, I("root:AFM_ltChunk"), when="pending, active, no abort"),
    )),
    FnEntry("AFM_TipDesiredPeriod", RUN, 1271, 1281, (), (),
            "6 while ltActive or scActive, else 15 (60/period Hz)"),
    FnEntry("AFM_SetMode", RUN, 1298, 1306, ("mode",), (
        act("MainPopupFunc", 1301, L('"ImagingModePopup_0"'), L("2"), L('"AC Mode"'),
            when='cmpstr(mode, "AC", 1) == 0 (case-sensitive)'),
        act("MainPopupFunc", 1303, L('"ImagingModePopup_0"'), L("1"), L('"Contact"'),
            when="any other token"),
    )),
    FnEntry("AFM_GlobalStr", RUN, 1501, 1508, ("path", "dflt"), (),
            "the string global at `path`, or `dflt` when it does not exist"),
    FnEntry("AFM_DispatchOther", RUN, 1477, 1497, ("sock", "verb", "line"), (
        afm("AFM_V3IsVerb", 1481, P("verb")),
        afm("AFM_V3Dispatch", 1482, P("verb"), P("line"), when="a wire v3 verb with a seq"),
        _send(1485, when="verb is not GETSTR: ERR - unknown verb: <verb>"),
        read("td_ReadString", 1490, W("tok 2")),
        _send(1492, when='read failed (a runtime error or the text "*ERROR*"): '
                         'ERR <uid> exception: td_ReadString(<name>) failed'),
        _send(1495, when="XPT <uid> <name>=<text>;"),
    )),
    FnEntry("AFM_ReadXPT", RUN, 1320, 1357, ("uid", "group"), (
        read("td_ReadGroup", 1328, I('group, "Crosspoint" when empty'),
             I("text wave root:AFM_xptTmp")),
        _send(1333, when="no data: ERR"),
        _send(1354, when="data: XPT <uid> InFast=<v>;"),
    )),
    FnEntry("AFM_ShowTip", RUN, 1364, 1376, ("on",), (
        afm("AFM_TipDesiredPeriod", 1369, when="on"),
    ), "(re)starts AFMtip on ON; never stops it"),
    FnEntry("AFM_TipTick", RUN, 1383, 1428, ("s",), (
        afm("AFM_TipDesiredPeriod", 1392),
        read("td_ReadValue", 1411, L('"XSensor"')),
        read("td_ReadValue", 1412, L('"YSensor"')),
        read("td_ReadValue", 1416, I('zChan = "Height" (:1415)')),
        read("td_ReadValue", 1417, L('"Deflection"')),
        read("td_ReadValue", 1421, L('"DetectorSum"')),
        read("td_ReadValue", 1422, L('"amplitude"')),
        afm("AFM_TelemTok", 1424, I("xv")),
        afm("AFM_TelemTok", 1424, I("yv")),
        afm("AFM_TelemTok", 1424, I("zv")),
        afm("AFM_TelemTok", 1425, I("dv")),
        afm("AFM_TelemTok", 1425, I("sv")),
        afm("AFM_TelemTok", 1425, I("av")),
        afm("AFM_SendTelem", 1426, SOCK_ARG, I("msg")),
    ), "AFMtip background task: reads only"),
    FnEntry("AFM_TelemTok", RUN, 1433, 1441, ("v",), ()),
    FnEntry("AFM_ArmAbortBackup", RUN, 1446, 1451, (), (),
            "restarts AFMabortBackup (proc AFM_AbortBackup) to fire once, 5 s from now"),
    FnEntry("AFM_AbortBackup", RUN, 1456, 1471, ("s",), (
        read("td_ReadValue", 1460, L('"Height"'), when="root:AFM_abortStopped is set"),
        afm("AFM_V3BackupRan", 1463, I("h"), L("0"), when="Height under kRetractedHeightV"),
        act("DoScanFunc", 1465, L('"StopEngageButton"'),
            when="Height is not under kRetractedHeightV; sets root:AFM_abortFault"),
        afm("AFM_V3BackupRan", 1467, I("h"), L("1"),
            when="Height is not under kRetractedHeightV"),
    ), "AFMabortBackup one-shot: Height < 1 V clears root:AFM_abortStopped"),
    # ---- wire v3 (phase-5-wire-v3-contract.md; verb map: docs/migration/plan/w1-ipf-map.md)
    FnEntry("AFM_V3Init", RUN, 1529, 1562, (), ()),
    FnEntry("AFM_V3LinkOpened", RUN, 1565, 1569, (), (
        afm("AFM_V3Init", 1566),
    )),
    FnEntry("AFM_V3IsVerb", RUN, 1571, 1574, ("verb",), ()),
    FnEntry("AFM_DispatchFromSocket", RUN, 1577, 1583, ("line",), (
        afm("AFM_Dispatch", 1580, P('line')),
    )),
    FnEntry("AFM_V3Intake", RUN, 1588, 1626, ("buf",), (
        afm("AFM_DispatchFromSocket", 1603, I('oneLine')),
        afm("AFM_V3Reply", 1620, I('StringFromList(1, oneLine, " ")'), L('"err"'), L('"dropped"')),
    )),
    FnEntry("AFM_V3Dispatch", RUN, 1629, 1671, ("verb", "line"), (
        afm("AFM_V3Init", 1631),
        afm("AFM_V3Hello", 1635, I('seq'), P('line')),
        afm("AFM_V3Read", 1638, I('seq'), P('line')),
        afm("AFM_V3Reply", 1641, I('seq'), L('"ok"'), I('AFM_V3Status()')),
        afm("AFM_V3Status", 1641),
        afm("AFM_V3Fetch", 1644, I('seq'), I('str2num(StringFromList(2, line, " "))')),
        afm("AFM_V3Load", 1648, I('seq'), P('line')),
        afm("AFM_V3Play", 1651, I('seq'), P('line')),
        afm("AFM_V3Set", 1654, I('seq'), P('line')),
        afm("AFM_V3Reply", 1657, I('seq'), L('"ok"'), L('""')),
        afm("AFM_SafePark", 1658),
        afm("AFM_V3ConfirmSafe", 1661, I('seq')),
        afm("AFM_V3VendorWithdraw", 1664, I('seq')),
        afm("AFM_V3ClearFault", 1667, I('seq')),
    )),
    FnEntry("AFM_V3Reply", RUN, 1675, 1686, ("seq", "status", "body"), (
        afm("AFM_SendStr", 1685, I('sock'), I('msg + "\\n"')),
    )),
    FnEntry("AFM_V3Host", RUN, 1689, 1697, (), ()),
    FnEntry("AFM_V3ToHost", RUN, 1702, 1715, ("msg",), (
        afm("AFM_V3Host", 1704),
        afm("AFM_SendStr", 1709, I('sock'), P('msg')),
        afm("AFM_V3SockDied", 1712),
    )),
    FnEntry("AFM_V3SockDied", RUN, 1718, 1726, (), ()),
    FnEntry("AFM_V3Mark", RUN, 1728, 1731, (), ()),
    FnEntry("AFM_V3Stamps", RUN, 1733, 1735, (), (
        afm("AFM_V3Mark", 1734),
        afm("AFM_ReadAbortFault", 1734),
    )),
    FnEntry("AFM_V3Num", RUN, 1738, 1752, ("v",), ()),
    FnEntry("AFM_V3Chop", RUN, 1755, 1762, ("s", "tail"), ()),
    FnEntry("AFM_V3Report", RUN, 1764, 1773, ("run", "event", "route"), (
        afm("AFM_V3Init", 1766),
        afm("AFM_V3Stamps", 1769),
        afm("AFM_V3Ring", 1771, I('useq'), I('msg')),
        afm("AFM_V3ToHost", 1772, I('msg + "\\n"')),
    )),
    FnEntry("AFM_V3Ring", RUN, 1775, 1795, ("useq", "msg"), ()),
    FnEntry("AFM_V3RingSpan", RUN, 1797, 1805, (), ()),
    FnEntry("AFM_SendTelem", RUN, 1808, 1818, ("sock", "msg"), (
        afm("AFM_V3Host", 1811),
        afm("AFM_SendStr", 1812, P('sock'), P('msg')),
        afm("AFM_V3ToHost", 1817, I('"TICK " + num2istr(useq) + " " + num2istr(ticks) + " " + AFM_V3Stamps() + " -" + reads')),
        afm("AFM_V3Stamps", 1817),
    )),
    FnEntry("AFM_SendRunFrame", RUN, 1825, 1833, ("sock", "msg"), (
        afm("AFM_V3RunFrame", 1830, P('msg')),
        afm("AFM_SendStr", 1832, P('sock'), P('msg')),
    )),
    FnEntry("AFM_V3RunFrame", RUN, 1838, 1874, ("msg",), (
        afm("AFM_V3Chop", 1852, P('msg'), L('"\\n"')),
        afm("AFM_V3Cap", 1857, P('msg')),
        afm("AFM_V3RunEnds", 1859, L('"done"')),
        afm("AFM_V3RunEnds", 1864, L('"done"')),
        afm("AFM_V3RunEnds", 1871, I('"error -1 " + text')),
    )),
    FnEntry("AFM_V3Cap", RUN, 1878, 1915, ("msg",), (
        afm("AFM_V3Chop", 1892, I('vals'), L('";"')),
        afm("AFM_V3CloseStep", 1903),
        afm("AFM_V3ToHost", 1914, I('"CAP " + num2istr(useq) + " " + num2istr(ticks) + " " + num2istr(AFM_V3Mark()) + " " + run + " " + num2istr(k) + " " + chan + " " + vals + "\\n"')),
        afm("AFM_V3Mark", 1914),
    )),
    FnEntry("AFM_V3CloseStep", RUN, 1918, 1933, (), (
        afm("AFM_V3Report", 1928, I('run'), I('"step " + num2istr(step) + " " + num2istr(steps) + " " + num2istr(stepPts)'), L('"sock"')),
    )),
    FnEntry("AFM_V3RunEnds", RUN, 1936, 1953, ("event",), (
        afm("AFM_V3CloseStep", 1942),
        afm("AFM_V3Report", 1952, I('ended'), P('event'), L('"sock"')),
    )),
    FnEntry("AFM_V3HelloBody", RUN, 1957, 1962, (), (
        afm("AFM_GetGVTok", 1959, L('"MicroscopeID"')),
        afm("AFM_V3RingSpan", 1960),
        afm("AFM_V3Mark", 1960),
        afm("AFM_ReadAbortFault", 1960),
        afm("AFM_MacroStartup", 1961),
    )),
    FnEntry("AFM_V3Hello", RUN, 1965, 1982, ("seq", "line"), (
        afm("AFM_V3HelloBody", 1969),
        afm("AFM_V3Reply", 1974, P('seq'), L('"err"'), I('"mismatch " + body')),
        afm("AFM_V3Reply", 1981, P('seq'), L('"ok"'), I('body')),
    )),
    FnEntry("AFM_V3Read", RUN, 1985, 2005, ("seq", "line"), (
        afm("AFM_V3Reply", 1991, P('seq'), L('"ok"'), I('AFM_V3Num(AFM_GetValue(name))')),
        afm("AFM_V3Num", 1991, I('AFM_GetValue(name)')),
        afm("AFM_GetValue", 1991, I('name')),
        afm("AFM_V3Reply", 1993, P('seq'), L('"ok"'), I('AFM_GetGVTok(name)')),
        afm("AFM_GetGVTok", 1993, I('name')),
        afm("AFM_V3Reply", 1996, P('seq'), L('"err"'), I('"bad flag " + name')),
        afm("AFM_V3Reply", 1998, P('seq'), L('"ok"'), I('num2istr(AFM_ReadAbortFault())')),
        afm("AFM_ReadAbortFault", 1998),
        afm("AFM_V3ReadStr", 2000, P('seq'), I('name')),
        afm("AFM_V3ReadXpt", 2002, P('seq'), I('name')),
        afm("AFM_V3Reply", 2004, P('seq'), L('"err"'), I('"bad read " + what')),
    )),
    FnEntry("AFM_V3ReadStr", RUN, 2008, 2015, ("seq", "name"), (
        read("td_ReadString", 2010, P('name')),
        afm("AFM_V3Reply", 2012, P('seq'), L('"err"'), I('"td_ReadString(" + name + ") failed"')),
        afm("AFM_V3Reply", 2014, P('seq'), L('"ok"'), I('name + "=" + text + ";"')),
    )),
    FnEntry("AFM_V3ReadXpt", RUN, 2019, 2046, ("seq", "group"), (
        read("td_ReadGroup", 2026, P('group'), I('tmp')),
        afm("AFM_V3Reply", 2043, P('seq'), L('"err"'), I('"td_ReadGroup(" + group + ") returned no data (err " + num2istr(err) + ")"')),
        afm("AFM_V3Reply", 2045, P('seq'), L('"ok"'), I('body')),
    )),
    FnEntry("AFM_V3DeadSeconds", RUN, 2048, 2054, (), ()),
    FnEntry("AFM_V3Status", RUN, 2056, 2070, (), (
        afm("AFM_V3Mark", 2067),
        afm("AFM_ReadAbortFault", 2067),
        afm("AFM_V3Num", 2068, I('AFM_V3DeadSeconds()')),
        afm("AFM_V3DeadSeconds", 2068),
        afm("AFM_V3RingSpan", 2069),
    )),
    FnEntry("AFM_V3Fetch", RUN, 2073, 2098, ("seq", "from"), (
        afm("AFM_V3Reply", 2089, P('seq'), L('"err"'), I('"gone " + num2istr(k)')),
        afm("AFM_SendStr", 2094, I('sock'), I('txt[k] + "\\n"')),
        afm("AFM_V3Reply", 2097, P('seq'), L('"ok"'), I('AFM_V3RingSpan()')),
        afm("AFM_V3RingSpan", 2097),
    )),
    FnEntry("AFM_V3NotHost", RUN, 2100, 2107, ("seq",), (
        afm("AFM_V3Host", 2102),
        afm("AFM_V3Reply", 2105, P('seq'), L('"err"'), L('"host"')),
    )),
    FnEntry("AFM_V3Load", RUN, 2111, 2147, ("seq", "line"), (
        afm("AFM_V3NotHost", 2113, P('seq')),
        afm("AFM_V3Reply", 2122, P('seq'), L('"err"'), I('"bad piece " + num2istr(nPiece)')),
        afm("AFM_V3Reply", 2126, P('seq'), L('"err"'), I('"bad noname " + name')),
        afm("AFM_AppendWave", 2130, I('name'), I('body')),
        afm("AFM_StoreWave", 2134, I('name'), I('body')),
        afm("AFM_V3Reply", 2146, P('seq'), L('"ok"'), I('name + " " + num2istr(have) + "/" + num2istr(total)')),
    )),
    FnEntry("AFM_V3Play", RUN, 2151, 2181, ("seq", "line"), (
        afm("AFM_V3NotHost", 2153, P('seq')),
        afm("AFM_V3Mark", 2157),
        afm("AFM_V3Reply", 2158, P('seq'), L('"err"'), L('"mark"')),
        afm("AFM_ReadAbortFault", 2159),
        afm("AFM_V3Reply", 2160, P('seq'), L('"err"'), L('"fault"')),
        afm("AFM_V3Reply", 2162, P('seq'), L('"err"'), L('"busy"')),
        afm("AFM_V3PlaySteps", 2166, I('kind'), P('line')),
        afm("AFM_V3Reply", 2168, P('seq'), L('"err"'), I('"bad kind " + kind')),
        afm("AFM_V3Reply", 2170, P('seq'), L('"err"'), L('"bad args"')),
        afm("AFM_V3MissingWave", 2172, I('kind'), P('line')),
        afm("AFM_V3Reply", 2174, P('seq'), L('"err"'), I('"bad noname " + missing')),
        afm("AFM_V3RunBegins", 2176, I('kind'), I('token'), I('steps')),
        afm("AFM_V3Arm", 2177, I('kind'), P('line')),
        afm("AFM_V3Reply", 2178, P('seq'), L('"ok"'), I('"armed " + token')),
        afm("AFM_V3Report", 2179, I('token'), L('"armed"'), L('"sock"')),
        afm("AFM_V3Release", 2180),
    )),
    FnEntry("AFM_V3PlaySteps", RUN, 2185, 2216, ("kind", "line"), ()),
    FnEntry("AFM_V3RunBegins", RUN, 2218, 2233, ("kind", "token", "steps"), ()),
    FnEntry("AFM_V3Arm", RUN, 2237, 2267, ("kind", "line"), (
        afm("AFM_ArmOutPair", 2245, L('0'), L('"0"'), L('"PIDSLoop.0.Setpoint"'), I('t4'), L('"PIDSLoop.1.Setpoint"'), I('t5'), I('str2num(t6)')),
        afm("AFM_ArmInPair", 2246, L('0'), L('"0"'), L('"XSensor"'), L('"YSensor"'), I('str2num(t7)'), I('str2num(t8)')),
        afm("AFM_FireEvent", 2248, L('"move"'), L('"0"')),
        afm("AFM_V3LithoWave", 2251, I('t5'), L('"XLitho"')),
        afm("AFM_V3LithoWave", 2252, I('t6'), L('"YLitho"')),
        afm("AFM_V3LithoWave", 2253, I('t7'), L('"VLitho"')),
        afm("AFM_ArmAndFireLitho", 2255, I('t4'), I('str2num(t8)'), I('str2num(t9)'), I('str2num(t10)')),
        afm("AFM_LithoChainStart", 2259, I('t4'), I('t5'), I('t6'), I('t7'), I('t8'), I('str2num(t9)'), I('str2num(t10)')),
        afm("AFM_ScanChainStart", 2263, I('t4'), I('t5'), I('str2num(t6)'), I('str2num(t7)'), I('str2num(t8)'), I('str2num(t9)'), I('errChan=t10')),
    )),
    FnEntry("AFM_V3MissingWave", RUN, 2270, 2285, ("kind", "line"), ()),
    FnEntry("AFM_V3LithoWave", RUN, 2288, 2295, ("have", "want"), ()),
    FnEntry("AFM_V3Release", RUN, 2298, 2308, (), (
        afm("AFM_V3RunFrame", 2304, I('held[i]')),
    )),
    FnEntry("AFM_V3Set", RUN, 2312, 2344, ("seq", "line"), (
        afm("AFM_V3NotHost", 2314, P('seq')),
        afm("AFM_SetValue", 2323, I('a'), I('str2num(b)')),
        afm("AFM_V3Reply", 2324, P('seq'), L('"ok"'), I('"value " + a + " " + AFM_V3Num(AFM_GetValue(a))')),
        afm("AFM_V3Num", 2324, I('AFM_GetValue(a)')),
        afm("AFM_GetValue", 2324, I('a')),
        afm("AFM_V3SetLoop", 2326, P('seq'), P('line')),
        afm("AFM_RampOutput", 2328, I('a'), I('str2num(b)'), I('str2num(c)'), I('str2num(d)')),
        afm("AFM_V3Reply", 2329, P('seq'), L('"ok"'), I('"ramp " + a + " " + b + " " + c + " " + d + " err=" + num2istr(err)')),
        afm("AFM_StopLoop", 2331, I('str2num(a)')),
        afm("AFM_V3Reply", 2335, P('seq'), L('"ok"'), I('"loopstop " + a + " err=" + num2istr(err)')),
        afm("AFM_SetMode", 2337, I('a')),
        afm("AFM_V3Reply", 2338, P('seq'), L('"ok"'), I('"mode " + a')),
        afm("AFM_ShowTip", 2340, I('cmpstr(a, "ON", 1) == 0')),
        afm("AFM_V3Reply", 2341, P('seq'), L('"ok"'), L('""')),
        afm("AFM_V3Reply", 2343, P('seq'), L('"err"'), I('"bad set " + what')),
    )),
    FnEntry("AFM_V3SetLoop", RUN, 2347, 2359, ("seq", "line"), (
        afm("AFM_SetupServoLoop", 2357, I('str2num(n)'), I('inCh'), I('str2num(sp)'), I('str2num(pg)'), I('str2num(ig)'), I('outCh'), I('str2num(lo)'), I('str2num(hi)')),
        afm("AFM_V3Reply", 2358, P('seq'), L('"ok"'), I('"loop " + n + " " + inCh + " " + sp + " " + pg + " " + ig + " " + outCh + " " + lo + " " + hi + " err=" + num2istr(err)')),
    )),
    FnEntry("AFM_V3NoteLoop", RUN, 2363, 2372, ("loop", "setpoint", "outMin", "outMax"), (
        afm("AFM_V3Init", 2368),
        afm("AFM_V3Num", 2369, P('setpoint')),
        afm("AFM_V3Num", 2369, P('outMin')),
        afm("AFM_V3Num", 2369, P('outMax')),
    )),
    FnEntry("AFM_V3ConfirmSafe", RUN, 2375, 2385, ("seq",), (
        read("td_ReadValue", 2377, L('"Height"')),
        afm("AFM_AbortConfirmed", 2379),
        afm("AFM_V3Reply", 2381, P('seq'), L('"ok"'), I('"height=" + AFM_V3Num(h)')),
        afm("AFM_V3Num", 2381, I('h')),
        afm("AFM_V3Report", 2382, L('"-"'), I('"confirmed " + AFM_V3Num(h)'), L('"sock"')),
        afm("AFM_V3Num", 2382, I('h')),
        afm("AFM_V3Reply", 2384, P('seq'), L('"err"'), I('"height=" + AFM_V3Num(h)')),
        afm("AFM_V3Num", 2384, I('h')),
    )),
    FnEntry("AFM_V3VendorWithdraw", RUN, 2388, 2399, ("seq",), (
        afm("AFM_V3Reply", 2392, P('seq'), L('"ok"'), L('"already"')),
        act("DoScanFunc", 2394, L('"StopEngageButton"')),
        afm("AFM_V3Reply", 2398, P('seq'), L('"ok"'), L('"ran"')),
    )),
    FnEntry("AFM_V3ClearFault", RUN, 2403, 2411, ("seq",), (
        afm("AFM_V3NotHost", 2405, P('seq')),
        afm("AFM_ClearFault", 2408),
        afm("AFM_V3Reply", 2410, P('seq'), L('"ok"'), L('"fault=0"')),
    )),
    FnEntry("AFM_V3ParkEnds", RUN, 2418, 2436, (), (
        afm("AFM_V3Init", 2419),
        afm("AFM_V3RunEnds", 2433, L('"aborted"')),
        afm("AFM_V3Report", 2435, I('parked'), I('"madesafe " + why'), I('route')),
    )),
    FnEntry("AFM_V3BackupRan", RUN, 2439, 2450, ("h", "withdrew"), (
        afm("AFM_V3Init", 2441),
        afm("AFM_V3Report", 2445, L('"-"'), I('"backup " + AFM_V3Num(h) + " withdraw"'), L('"-"')),
        afm("AFM_V3Num", 2445, P('h')),
        afm("AFM_V3Report", 2449, L('"-"'), I('"backup " + AFM_V3Num(h) + " cleared"'), L('"-"')),
        afm("AFM_V3Num", 2449, P('h')),
    )),
    FnEntry("AFM_V3GuardTick", RUN, 2455, 2477, ("s",), (
        afm("AFM_SafePark", 2472),
    )),
    # The operator's abort panel (Macros > AFM Abort panel); nothing on the wire reaches it.
    FnEntry("AFM_ShowAbortPanel", RUN, 2490, 2502, (), ()),
    FnEntry("AFM_AbortButtonProc", RUN, 2506, 2518, ("ba",), (
        afm("AFM_SafePark", 2511),
        read("td_ReadValue", 2512, L('"Height"')),
    ), "then arms the one-shot AFMabortPanel status check"),
    FnEntry("AFM_AbortPanelTick", RUN, 2521, 2539, ("s",), (
        read("td_ReadValue", 2523, L('"Height"')),
    ), "rewrites the panel status from Height and the abort flags; returns 1"),
    FnEntry("AFM_PassMayPlay", RUN, 2547, 2550, (), (),
            "returns root:AFM_passArmed: set where a pass arms out-waves or fires, cleared by td_Stop"),
    FnEntry("AFM_StopRan", RUN, 2554, 2559, (), (),
            "sets root:AFM_abortStopped, clears root:AFM_passArmed and root:AFM_v3StopSeen"),
    # ---- afm_igor.ipf (STARTUP reads)
    FnEntry("AFM_State", IGOR, 22, 33, (), (
        read("td_ReadValue", 24, L('"XSensor"')),
        read("td_ReadValue", 25, L('"YSensor"')),
        read("td_ReadValue", 26, L('"Height"')),
        read("td_ReadValue", 27, L('"Deflection"')),
        read("td_ReadValue", 28, L('"Amplitude"')),
        read("td_ReadValue", 29, L('"DetectorSum"')),
        read("td_ReadValue", 30, L('"Output.A"')),
        read("GV", 31, L('"ARSysScanBusy"')),
    )),
    FnEntry("AFM_GetScanParams", IGOR, 39, 48, (), tuple(
        read("GV", ln, L(f'"{nm}"')) for nm, ln in (
            ("ScanSize", 41), ("ScanRate", 42), ("ScanPoints", 43), ("ScanLines", 44),
            ("ScanAngle", 45), ("ImagingMode", 46)))),
    FnEntry("AFM_GetSetpoint", IGOR, 53, 64, (), (
        read("GV", 55, L('"ImagingMode"')),
        read("GV", 57, L('"AmplitudeSetpointVolts"'), when="ImagingMode == 1 (state)"),
        read("GV", 60, L('"DeflectionSetpointVolts"'), when="otherwise"),
    )),
    FnEntry("AFM_GetScanner", IGOR, 67, 72, (), (
        read("GV", 69, L('"MaxScanSize"')),
        read("GV", 70, L('"InputRange"')),
    )),
    FnEntry("AFM_GetGains", IGOR, 75, 80, (), (
        read("GV", 77, L('"IntegralGain"')),
        read("GV", 78, L('"ProportionalGain"')),
    )),
    FnEntry("AFM_GetCantilever", IGOR, 83, 93, (), tuple(
        read("GV", ln, L(f'"{nm}"')) for nm, ln in (
            ("ThermalFrequency", 85), ("ThermalQ", 86), ("SpringConstant", 87),
            ("InvOLS", 88), ("AmpInvOLS", 89), ("DriveFrequency", 90),
            ("DriveAmplitude", 91)))),
    FnEntry("AFM_GetCal", IGOR, 96, 114, (), tuple(
        read("GV", ln, L(f'"{nm}"')) for nm, ln in (
            ("XLVDTSens", 101), ("YLVDTSens", 103), ("ZLVDTSens", 105),
            ("XLVDTOffset", 107), ("YLVDTOffset", 109), ("ZLVDTOffset", 111)))),
    FnEntry("AFM_GetXYLoopGains", IGOR, 128, 154, (), tuple(
        read("GV", ln, L(f'"{nm}"')) for nm, ln in (
            ("XPiezoSens", 130), ("YPiezoSens", 130), ("XPGain", 131), ("XIGain", 131),
            ("YPGain", 132), ("YIGain", 132)))),
    FnEntry("AFM_MacroStartup", IGOR, 161, 176, (), (
        afm("AFM_GetScanParams", 163),
        afm("AFM_GetSetpoint", 164),
        afm("AFM_GetScanner", 165),
        afm("AFM_GetGains", 166),
        afm("AFM_GetCantilever", 167),
        afm("AFM_GetCal", 168),
        afm("AFM_GetXYLoopGains", 169),
        afm("AFM_State", 170),
        read("GV", 174, L('"MicroscopeID"')),
    )),
    # ---- afm_sockit.ipf (transport)
    FnEntry("AFM_SockConnect", SOCK, 23, 28, ("host", "port"), (
        xport("SOCKITopenconnectionF", 27, P("host"), P("port"), L("5")),
    )),
    FnEntry("AFM_SockClose", SOCK, 58, 62, ("sock",), (
        xport("SOCKITcloseConnection", 61, P("sock")),
    )),
    FnEntry("AFM_SendStr", SOCK, 283, 288, ("sock", "msg"), (
        xport("SOCKITsendmsgF", 287, P("sock"), P("msg")),
    )),
)}

# Fill callbacks registered by string in an in-wave arm; the caller supplies `@FILL`.
CALLBACKS = {
    "AFM_Done()": ("AFM_Done", f"{RUN}:487, :500"),
    "AFM_LithoDone()": ("AFM_LithoDone", f"{RUN}:643"),
    "AFM_ScanAdvance()": ("AFM_ScanAdvance", f"{RUN}:930"),
    "AFM_LithoAdvance()": ("AFM_LithoAdvance", f"{RUN}:1129"),
}

# Background tasks started by name (CtrlNamedBackground ... proc=).
BACKGROUND = {
    "AFMdispatch": ("AFM_DispatchTick", f"{RUN}:127"),
    "AFMtip": ("AFM_TipTick", f"{RUN}:136, :1371"),
    "AFMscan": ("AFM_ScanTick", f"{RUN}:890"),
    "AFMlitho": ("AFM_LithoTick", f"{RUN}:1075"),
    "AFMabortBackup": ("AFM_AbortBackup", f"{RUN}:1448"),
    "AFMv3guard": ("AFM_V3GuardTick", f"{RUN}:1722"),
    "AFMabortPanel": ("AFM_AbortPanelTick", f"{RUN}:2516"),
}

# Entry points outside the socket, as the expander's pseudo-verbs.
PSEUDO = {
    "@FILL": "the in-bank fill callbacks of the last fired event (CALLBACKS)",
    "@SCANTICK": "AFM_ScanTick",
    "@LITHOTICK": "AFM_LithoTick",
    "@TIPTICK": "AFM_TipTick",
    "@SAFEPARK": "AFM_SafePark",
    "@RUNSTART": "AFM_LithoRunStart",
    "@RUNEND": "AFM_LithoRunEnd",
    "@DRYRUN": "AFM_LithoRunDryRun",
    "@DOSCANFUNC": "vendor DoScanFunc over COM, below the line (one opaque call)",
    "@PRINT": "Igor Print over COM; no vendor call",
    "@COM": "any other COM command; unmodelled, so the expander refuses it",
    "@COMFAIL": "a COM command Igor rejected with an error code; the expander refuses it",
}


@dataclass(frozen=True)
class Resolved:
    """A vendor call reached from a verb, its arguments traced back to wire/igor/literal."""
    fn: str
    kind: str
    args: tuple[Arg, ...]
    when: str
    cite: str

    def __str__(self) -> str:
        cond = f"   [when {self.when}]" if self.when else ""
        return f"{self.fn}({', '.join(map(str, self.args))})  {self.cite}{cond}"


def _walk(steps, env: dict[str, Arg], file: str, when: str, out: list[Resolved]) -> None:
    for s in steps:
        args = tuple(env.get(a.note, I(f"default of optional {a.note}"))
                     if a.src == "param" else a for a in s.args)
        cond = " and ".join(c for c in (when, s.when) if c)
        if s.kind in ("act", "read"):
            out.append(Resolved(s.name, s.kind, args, cond, f"{file}:{s.line}"))
        elif s.kind == "afm" and s.name in FUNCTIONS:
            fn = FUNCTIONS[s.name]
            _walk(fn.steps, dict(zip(fn.params, args, strict=False)), fn.file, cond, out)


def resolve(verb: str, *, dry: bool = False) -> list[Resolved]:
    """Every vendor call `verb` reaches through direct calls (callbacks and ticks excluded)."""
    entry = (DRY_VERBS if dry else VERBS)[verb]
    out: list[Resolved] = []
    _walk(entry.steps, {}, RUN, "", out)
    return out


def resolve_function(name: str) -> list[Resolved]:
    """Same as `resolve`, for a callback, tick or pseudo-verb entry function."""
    fn = FUNCTIONS[name]
    out: list[Resolved] = []
    _walk(fn.steps, {p: P(p) for p in fn.params}, fn.file, "", out)
    return out


def format_table() -> str:
    """The resolved table as text, one block per verb and entry function."""
    blocks = []
    for verb, entry in VERBS.items():
        blocks.append(f"{verb}  ({RUN}:{entry.line})")
        blocks += [f"  {r}" for r in resolve(verb)] or ["  (no vendor call)"]
    for name in ("AFM_Done", "AFM_LithoDone", "AFM_ScanAdvance", "AFM_LithoAdvance",
                 "AFM_ScanTick", "AFM_LithoTick", "AFM_TipTick", "AFM_LithoRunEnd"):
        blocks.append(f"{name}  ({FUNCTIONS[name].file}:{FUNCTIONS[name].start})")
        blocks += [f"  {r}" for r in resolve_function(name)] or ["  (no vendor call)"]
    return "\n".join(blocks)


if __name__ == "__main__":
    print(format_table())
