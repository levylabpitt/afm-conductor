"""The wire-tap -> expander contract (SLICES.md F12) and the golden comparison rules."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest
from golden.harness import GoldenStub, StubCom, load_fixture

from afm_litho.goldens import ExpanderError, expand
from afm_litho.goldens import golden_format as G
from afm_litho.goldens.expander import WIRE_VERBS, summarise
from afm_litho.goldens.wire_tap import (
    VERB_CLASS,
    TappedCom,
    TappedComRefused,
    WireTap,
    com_entry,
)
from afm_litho.hardware import bridge_protocol as P
from afm_litho.hardware.failsafe_thread import FailsafeWatchdog
from afm_litho.hardware.hardware_arc import HardwareARC
from afm_litho.hardware.lithography_run import LithographyRun

PROTOCOL = Path(P.__file__)


class _Ok:
    error_code = 0


class _Rejected:
    error_code = 7


class _Com:
    """A COM double: returns `reply` for every command, or raises it if it is an exception."""

    def __init__(self, reply=None):
        self.reply = _Ok() if reply is None else reply
        self.calls: list[str] = []

    def execute(self, cmd, *, check=True):
        self.calls.append(cmd)
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply


def _single_object_tap() -> WireTap:
    values = dict(load_fixture()["values"], XSensor=-0.63, YSensor=0.07)
    stub = GoldenStub(values, microscope_id=4, done_delay=0.0)
    port = stub.serve_in_thread()
    try:
        run = LithographyRun.connect(
            "127.0.0.1", port, settle_move_s=0.0, cmd_gap_s=0.0, xy_withdraw_max_s=0.0,
            xy_withdraw_settle_s=0.0, first_object_settle_s=0.0, amp_gain=25.0,
            retract=-10.0, decimation=50, z_out="Height")
        run.start()
        run.write_object("obj0", [0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [3.0, 3.0, 3.0],
                         setpoint=0.4, speed=5.0)
        run.get_value("XSensor")             # barrier: the post-DONE lift has been dispatched
    finally:
        stub.stop()
    return stub.tap


# ------------------------------------------- (a) one transcript the expander reads --

def test_the_expander_reads_the_taps_raw_transcript():
    tap = _single_object_tap()
    raw = tap.expander_input()
    assert raw[0].startswith("STARTUP ")
    assert raw.count("@FILL") == 2                 # the transit FIRE, then the ARMFIRE
    out = expand(raw, microscope_id=4)
    acts = out.lines()
    assert any(a.startswith("td_xSetOutWavePair(0,\"14\"") for a in acts)
    assert acts.count('td_WriteValue("Output.A",0)') == 2     # AFM_Done, then AFM_LithoDone
    assert [f[2] for f in out.frames if f[0] == "DONE"] == ["XYTransit", "obj0"]


def test_raw_lines_are_verbatim_and_summaries_stay_small():
    tap = WireTap()
    line = "WAVE w " + ";".join(["1.5"] * 5000)
    tap.record(line)
    assert tap.expander_input() == [line]
    assert len(tap.to_json()) < 400               # the summary, never the payload
    assert "raw" not in tap.snapshot()


# ------------------------------------------------------------ (b) one wave hash --

def test_tap_and_expander_hash_a_wave_identically():
    tap = _single_object_tap()
    out = expand(tap.expander_input(), microscope_id=4)
    wire_xlitho = [G.render_entry(e) for e in tap.actuation
                   if e["verb"] == "WAVE" and e["args"][0][0] == "XLitho"]
    bound = [a for a in out.lines() if "wave(XLitho," in a]
    assert len(wire_xlitho) == 1 and len(bound) == 1
    assert wire_xlitho[0].split(" ", 1)[1] in bound[0]


def test_one_hash_across_appends_case_and_igor_parsing():
    vals = np.array([0.1, -2.5, 3.0, np.nan, 7.123456789012345], dtype="<f4")
    txt = [f"{v:.12g}" for v in vals.astype(float)]
    tap = WireTap()
    tap.record("WAVE Wv " + ";".join(txt[:2]))
    tap.record("WAVEAPP wV " + ";".join(txt[2:]) + ";")       # Igor names fold case
    _name, n, _first, _last, sha = tap.actuation[-1]["args"][0]
    ref = summarise("Wv", vals)
    assert (n, sha) == (5, ref.sha) == (5, hashlib.sha256(vals.tobytes()).hexdigest()[:16])
    exp = expand(["WAVE Wv " + ";".join(txt[:2]), "WAVEAPP wV " + ";".join(txt[2:]) + ";",
                  "OUTWAVE 2 14 Output.A wv -1"])
    assert f"sha={sha}" in exp.lines()[0]


# ------------------------------------------- (c) one spelling for out-of-socket --

@pytest.mark.parametrize("text,entry", [
    ("AFM_SafePark()", ("@SAFEPARK", [])),
    ('DoScanFunc("StopEngageButton")', ("@DOSCANFUNC", ["StopEngageButton"])),
    ('DoScanFunc("Withdraw")', ("@DOSCANFUNC", ["Withdraw"])),
    ('AFM_LithoRunStart("127.0.0.1", 9200)', ("@RUNSTART", [])),
    ("AFM_LithoRunEnd()", ("@RUNEND", [])),
    ("AFM_LithoRunDryRun(1)", ("@DRYRUN", ["1"])),
    ('Print "afm-litho: DoScanFunc(\\"Withdraw\\")"', ("@PRINT", [])),
    ('PV("ScanSize",1e-06)', ("@COM", ['PV("ScanSize",1e-06)'])),
    ('Print "a";td_Stop()', ("@COM", ['Print "a";td_Stop()'])),
    ('AFM_LithoRunStart("h",1);td_Stop()', ("@COM", ['AFM_LithoRunStart("h",1);td_Stop()'])),
    ("AFM_SafePark();td_Stop()", ("@COM", ["AFM_SafePark();td_Stop()"])),
    ('DoScanFunc("x") td_Stop()', ("@COM", ['DoScanFunc("x") td_Stop()'])),
    ('Print "a" td_Stop()', ("@COM", ['Print "a" td_Stop()'])),
])
def test_com_entry_spelling(text, entry):
    assert com_entry(text) == entry


def test_every_com_spelling_is_an_expander_pseudo_verb():
    from afm_litho.goldens.expander import PSEUDO_VERBS
    for text in ("AFM_SafePark()", 'DoScanFunc("x")', "AFM_LithoRunStart(1)",
                 "AFM_LithoRunEnd()", "AFM_LithoRunDryRun(0)", 'Print "x"', "Junk()"):
        assert com_entry(text)[0] in PSEUDO_VERBS


def test_failsafe_threads_com_is_tapped(monkeypatch):
    tap = WireTap()
    monkeypatch.setattr(FailsafeWatchdog, "_make_com_igor",
                        lambda self: TappedCom(_Com(), tap, "failsafe"))

    def work(reason, com):
        com.execute("AFM_SafePark()", check=True)
        return {"verified": True}

    wd = FailsafeWatchdog(work).start()
    try:
        assert wd.request("t").wait(5.0)
    finally:
        wd.stop()
    assert tap.actuation == [{"seq": 0, "kind": "ext", "verb": "@SAFEPARK", "args": [],
                              "source": "failsafe"}]
    assert tap.expander_input() == ["@SAFEPARK"]


def test_stub_com_safepark_runs_the_park_without_a_wire_record():
    stub = GoldenStub({"Output.Z": 50.0}, microscope_id=4)
    port = stub.serve_in_thread()
    arc = HardwareARC.connect("127.0.0.1", port)
    try:
        arc.start()
        com = TappedCom(StubCom(stub), stub.tap, "com")
        com.execute("AFM_SafePark()", check=True)
        assert stub.safe_parks == 1 and stub.values["Output.Z"] == 0.0
        assert [e["verb"] for e in stub.tap.actuation] == ["@SAFEPARK"]
        with pytest.raises(RuntimeError):
            com.execute('PV("x",1)', check=True)
        assert len(stub.tap.actuation) == 1
    finally:
        arc.close()
        stub.stop()


# ------------------------------------------------- (d) the expander never goes quiet --

def test_the_0_1_review_repro_now_raises():
    tap = _single_object_tap()
    with pytest.raises(ExpanderError, match="Tap summaries cannot be expanded"):
        expand(tap.actuation, microscope_id=4)


@pytest.mark.parametrize("transcript", [
    ["NOPE 1 2"],
    [("NOPE", ())],
    [("@NOPE", ())],
    ["@COM PV(\"x\",1)"],
    [{"verb": "SETVALUE", "args": ["a", "1"]}],
    [("WAVE", [["w", 1, "0", "0", "abc"]])],
    [("SETVALUE", "a 1")],
    [42],
    ["@DRYRUN 1", "NOPE 1"],
])
def test_unknown_verbs_pseudo_verbs_and_shapes_raise(transcript):
    with pytest.raises(ExpanderError):
        expand(transcript, microscope_id=4)


def test_rendered_wave_summaries_do_not_expand():
    golden = G.load(Path(__file__).parent / "golden" / "litho_single.golden.json")
    with pytest.raises(ExpanderError):
        expand(golden["wire"]["actuation"], microscope_id=4)
    with pytest.raises(ExpanderError, match="summarised wave"):
        expand(["WAVE w wave(w,n=2,first=0,last=1,sha=abc)"])
    with pytest.raises(ExpanderError, match="no WAVE loaded"):
        expand(["OUTWAVE 2 14 Output.A never_loaded -1"])


def test_strict_off_keeps_igors_err_reply():
    assert expand(["NOPE 1"], strict=False).frames == [("ERR", "-", "unknown verb: NOPE")]


# ----------------------------------------------- (e) read / actuation class pinned --

PINNED_CLASS = {
    "STARTUP": "read", "GETVALUE": "read", "GETGV": "read", "GETSTR": "read",
    "READXPT": "read",
    "SETVALUE": "act", "SERVOLOOP": "act", "STOPLOOP": "act", "RAMP": "act", "WAVE": "act",
    "WAVEAPP": "act", "ARMFIRE": "act", "SHOWTIP": "act", "SETMODE": "act", "ABORT": "act",
    "ABORTCONFIRM": "act", "ABORTFAULT": "read", "CLEARFAULT": "act",
    "OUTWAVE": "act", "OUTPAIR": "act", "INWAVE": "act", "INPAIR": "act", "FIRE": "act",
    "SCANCHAIN": "act", "LITHOCHAIN": "act",
}


def _driver_verbs() -> set[str]:
    text = PROTOCOL.read_text(encoding="utf-8")
    block = text.split("# driver -> bridge verbs", 1)[1].split("# bridge -> driver verbs", 1)[0]
    return {v for k, v in re.findall(r'^([A-Z]+) = "([A-Z]+)"', block, re.MULTILINE) if k == v}


def test_every_protocol_verb_has_a_pinned_class():
    assert _driver_verbs() == set(PINNED_CLASS)
    assert VERB_CLASS == PINNED_CLASS
    assert set(VERB_CLASS) == WIRE_VERBS


@pytest.mark.parametrize("verb", sorted(PINNED_CLASS))
def test_the_tap_files_each_verb_by_its_class(verb):
    tap = WireTap()
    tap.record(f"{verb} 1 2")
    assert bool(tap.reads) == (PINNED_CLASS[verb] == "read")
    assert bool(tap.actuation) == (PINNED_CLASS[verb] == "act")


# ------------------------------------------ (f) TappedCom records only what ran --

def test_tapped_com_records_igor_rejections_and_skips_calls_that_never_reached_igor():
    from afm_litho.hardware.igor import IgorError
    tap = WireTap()
    with pytest.raises(OSError):
        TappedCom(_Com(OSError("COM down")), tap).execute("AFM_SafePark()", check=True)
    assert tap.actuation == []                        # never reached Igor
    TappedCom(_Com(_Rejected()), tap).execute("AFM_SafePark()", check=False)
    with pytest.raises(IgorError):
        TappedCom(_Com(IgorError(3, "boom", "AFM_SafePark()")), tap).execute(
            "AFM_SafePark()", check=True)
    TappedCom(_Com(), tap).execute("AFM_SafePark()", check=True)
    assert [(e["verb"], e["args"]) for e in tap.actuation] == [
        ("@COMFAIL", ["AFM_SafePark()", "7"]), ("@COMFAIL", ["AFM_SafePark()", "3"]),
        ("@SAFEPARK", [])]
    assert tap.expander_input()[0] == "@COMFAIL AFM_SafePark() 7"
    with pytest.raises(ExpanderError, match="failed in Igor"):
        expand(tap.expander_input(), microscope_id=4)


@pytest.mark.parametrize("method", ["arc_call", "value", "string"])
def test_tapped_com_refuses_the_bypassing_entry_points(method):
    inner = _Com()
    with pytest.raises(TappedComRefused):
        getattr(TappedCom(inner, WireTap()), method)('td_ReadValue("x")')
    assert inner.calls == []


# ---------------------------------------------------------- the comparison rules --

def _g(act, reads, ipf=None):
    lvl = {"actuation": act, "reads": reads}
    return {"format": 2, "scenario": "s", "door": "flex", "microscope_id": 4,
            "ipf_sha256": ipf or {"a": "1"}, "fixture": {}, "known_defects": [],
            "wire": lvl, "igor": dict(lvl)}


def test_compare_is_exact_on_actuation_and_superset_on_reads():
    frozen = _g(["A", "B"], ["r1", "r2"])
    assert G.compare(frozen, _g(["A", "B"], ["r0", "r1", "r2"])) == []
    assert any("reads lost" in p for p in G.compare(frozen, _g(["A", "B"], ["r1"])))
    assert any("actuation" in p for p in G.compare(frozen, _g(["B", "A"], ["r1", "r2"])))
    assert any("actuation" in p for p in G.compare(frozen, _g(["A", "B", "C"], ["r1", "r2"])))
    assert any(p.startswith("ipf_sha256") for p in
               G.compare(frozen, _g(["A", "B"], ["r1", "r2"], ipf={"a": "2"})))


def test_function_names_compare_case_insensitively():
    frozen = _g(["A"], [])
    frozen["igor"] = {"actuation": ['IR_SetRamp(2,"Height",80,0)'],
                      "reads": ['td_ReadValue("XSensor")']}
    now = _g(["A"], [])
    now["igor"] = {"actuation": ['ir_SetRamp(2,"Height",80,0)'],
                   "reads": ['td_readValue("XSensor")']}
    assert G.compare(frozen, now) == []


def test_string_arguments_stay_exact():
    frozen, now = _g(["A"], []), _g(["A"], [])
    frozen["igor"] = {"actuation": ['td_xSetOutWave(2,"14","Output.A",w,-1)'],
                      "reads": ['td_ReadValue("amplitude")']}
    now["igor"] = {"actuation": ['td_xSetOutWave(2,"14","output.a",w,-1)'],
                   "reads": ['td_ReadValue("Amplitude")']}
    problems = G.compare(frozen, now)
    assert any(p.startswith("igor.actuation") for p in problems)
    assert any(p.startswith("igor.reads lost") for p in problems)
    ev = _g(["A"], [])
    ev["igor"] = {"actuation": ['ir_SetPISLoop(2,"always,never")'], "reads": []}
    frozen["igor"] = {"actuation": ['ir_SetPISLoop(2,"Always,Never")'], "reads": []}
    assert G.compare(frozen, ev)


def test_known_defects_are_stamped_and_compared():
    g = G.load(Path(__file__).parent / "golden" / "scan_abort_mid_line.golden.json")
    assert [d["id"] for d in g["known_defects"]] == ["f30-1"]
    fixed = dict(g, known_defects=[])
    assert any(p.startswith("known_defects") for p in G.compare(g, fixed))


def test_frozen_goldens_stamp_the_current_ipf_and_format():
    for path in sorted((Path(__file__).parent / "golden").glob("*.golden.json")):
        g = G.load(path)
        assert g["format"] == G.GOLDEN_FORMAT
        assert g["ipf_sha256"] == G.ipf_hashes(), f"{path.name}: .ipf changed; regenerate"
