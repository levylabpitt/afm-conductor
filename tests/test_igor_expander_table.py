"""The `.ipf` call table against the `.ipf` text: completeness, hashes, citations, order."""

from __future__ import annotations

import hashlib
import re
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from afm_litho.goldens import expander as X
from afm_litho.goldens import ipf_table as T
from afm_litho.hardware import bridge_protocol as P

ROOT = Path(__file__).resolve().parents[1]
IPF_FILES = (T.RUN, T.IGOR, T.SOCK)
CALL_RE = re.compile(r"\b(AFM_\w+|td_\w+|ir_\w+|IR_\w+|GV|MainPopupFunc|DoScanFunc|SOCKIT\w+)\s*\(")
FUNC_RE = re.compile(r"^\s*Function(?:/\w+)?\s+(\w+)\s*\(([^)]*)\)")
END_RE = re.compile(r"^\s*End\b")
ACT_FNS = {"td_Stop", "td_WriteValue", "td_WriteString", "td_xSetOutWave",
           "td_xSetOutWavePair", "td_xSetInWave", "td_xSetInWavePair", "ir_SetPISLoop",
           "ir_StopPISLoop", "IR_SetRamp", "ir_StopOutWaveBank", "ir_StopInWaveBank",
           "MainPopupFunc", "DoScanFunc"}
READ_FNS = {"td_ReadValue", "td_ReadGroup", "td_ReadString", "GV"}
ROOTS = ("AFM_Dispatch", "AFM_DispatchTick", "AFM_LithoRunStart", "AFM_LithoRunEnd",
         "AFM_LithoRunDryRun", "AFM_LithoRunTrace", "AFM_SafePark")
# Igor built-ins the three files call today. Any other callee (a new vendor call such as
# DoScanFunc or PV, or an unclassified td_/ir_ name) must be tabled before this test passes.
IGOR_BUILTINS = {
    "cmpstr", "DimSize", "GetDimLabel", "GetRTError", "ItemsInList", "max", "mod", "num2istr",
    "NumberByKey", "ReplaceNumberByKey", "WhichListItem",
    "num2str", "numpnts", "numtype", "NVAR_Exists", "PadString", "ParamIsDefault",
    "ReplaceString", "SelectString", "sign", "stopMSTimer", "str2num", "StringFromList",
    "strlen", "strsearch", "SVAR_Exists", "WaveExists"}
IGOR_KEYWORDS = {"if", "elseif", "while", "for", "switch", "strswitch", "return"}
ANY_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def _blank(raw: str, keep_strings: bool = False) -> str:
    """The line with any `//` comment cut and string contents blanked (columns kept)."""
    out, in_str, i = [], False, 0
    while i < len(raw):
        c = raw[i]
        if in_str:
            if c == "\\" and i + 1 < len(raw):
                out += "  "
                i += 2
                continue
            if c == '"':
                in_str = False
                out.append(c)
            else:
                out.append(c if keep_strings else " ")
        elif c == '"':
            in_str = True
            out.append(c)
        elif raw.startswith("//", i):
            break
        else:
            out.append(c)
        i += 1
    return "".join(out)


@cache
def raw_lines(path: str) -> list[str]:
    text = (ROOT / path).read_bytes().replace(b"\r\n", b"\n").decode("latin-1")
    return [""] + text.split("\n")


@cache
def code_lines(path: str) -> list[str]:
    return [_blank(r) for r in raw_lines(path)]


@cache
def uncommented_lines(path: str) -> list[str]:
    return [_blank(r, keep_strings=True) for r in raw_lines(path)]


@cache
def functions(path: str) -> dict[str, tuple[int, int, tuple[str, ...]]]:
    out, cur = {}, None
    for i, line in enumerate(code_lines(path)):
        m = FUNC_RE.match(line)
        if m:
            params = tuple(p.strip().strip("[]").strip()
                           for p in m.group(2).replace("[", ",").split(",") if p.strip("[] "))
            cur = (m.group(1), i, params)
        elif cur and END_RE.match(line):
            out[cur[0]] = (cur[1], i, cur[2])
            cur = None
    return out


def calls_in(path: str, start: int, end: int) -> list[tuple[str, int]]:
    """Calls written in lines start..end, in source order."""
    out = []
    for i in range(start, end + 1):
        out += [(m.group(1), i) for m in CALL_RE.finditer(code_lines(path)[i])]
    return out


def call_args(path: str, line: int, name: str, occurrence: int) -> list[str]:
    """Top-level argument texts of the occurrence-th call to `name` on `line`."""
    code, raw = code_lines(path)[line], raw_lines(path)[line]
    hits = [m for m in CALL_RE.finditer(code) if m.group(1) == name]
    assert len(hits) > occurrence, f"{path}:{line} has no call #{occurrence} to {name}"
    i = hits[occurrence].end()
    depth, cur, args = 0, [], []
    while True:
        c = code[i]
        if c == "(":
            depth += 1
        elif c == ")":
            if depth == 0:
                break
            depth -= 1
        elif c == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
            i += 1
            continue
        cur.append(raw[i])
        i += 1
    tail = "".join(cur).strip()
    if tail or args:
        args.append(tail)
    return args


@cache
def dispatch_cases() -> dict[int, dict[str, tuple[int, int]]]:
    """{switch index: {verb: (case line, last line of its block)}} inside AFM_Dispatch."""
    start, end, _ = functions(T.RUN)["AFM_Dispatch"]
    switches: dict[int, dict[str, tuple[int, int]]] = {}
    idx, open_case = -1, None
    for i in range(start, end + 1):
        line = uncommented_lines(T.RUN)[i]
        m = re.match(r'\s*case\s+"(\w+)"\s*:', line)
        is_default = re.match(r"\s*default\s*:", line)
        if re.match(r"\s*strswitch\b", line):
            idx += 1
            switches[idx] = {}
            continue
        if open_case and (m or is_default or re.match(r"\s*endswitch\b", line)):
            switches[idx][open_case[0]] = (open_case[1], i - 1)
            open_case = None
        if m:
            open_case = (m.group(1), i)
        elif is_default:
            open_case = ("default", i)
    return switches


def _steps_with_files():
    for fn in T.FUNCTIONS.values():
        for s in fn.steps:
            yield fn.file, s
    for table in (T.VERBS, T.DRY_VERBS):
        for e in table.values():
            for s in e.steps:
                yield T.RUN, s


# ---------------------------------------------------------------------- (b) hashes --

@pytest.mark.parametrize("path", IPF_FILES)
def test_ipf_hash_pinned(path):
    digest = hashlib.sha256((ROOT / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert digest == T.IPF_SHA256[path], (
        f"{path} changed: re-derive the ipf_table entries it touches, then re-pin its hash")


# ---------------------------------------------------------------- (a) completeness --

def test_every_dispatch_case_has_an_entry():
    cases = dispatch_cases()
    assert set(cases) == {0, 1}
    assert set(cases[0]) == set(T.DRY_VERBS)
    assert set(cases[1]) == set(T.VERBS)
    for table, sw in ((T.DRY_VERBS, 0), (T.VERBS, 1)):
        for verb, e in table.items():
            assert cases[sw][verb][0] == e.line, verb


def test_every_reachable_function_has_an_entry():
    defined = {name: path for path in IPF_FILES for name in functions(path)}
    graph: dict[str, set[str]] = {}
    for name, path in defined.items():
        start, end, _ = functions(path)[name]
        body_raw = "\n".join(uncommented_lines(path)[start + 1:end + 1])
        callees = {c for c, _ in calls_in(path, start + 1, end)}
        callees |= set(re.findall(r'"(AFM_\w+)\(\)"', body_raw))
        callees |= set(re.findall(r"proc=(AFM_\w+)", body_raw))
        graph[name] = {c for c in callees if c in defined}
    seen, todo = set(), list(ROOTS)
    while todo:
        f = todo.pop()
        if f not in seen:
            seen.add(f)
            todo += graph.get(f, ())
    missing = sorted(seen - set(T.FUNCTIONS))
    assert not missing, f"reachable functions with no table entry: {missing}"
    assert not sorted(set(functions(T.RUN)) - set(T.FUNCTIONS))


def test_every_callee_is_tabled_or_an_allowed_builtin():
    defined = {name for path in IPF_FILES for name in functions(path)}
    vendor = ACT_FNS | READ_FNS
    unknown = []
    for path in IPF_FILES:
        for i, line in enumerate(code_lines(path)):
            if FUNC_RE.match(line):
                continue
            for name in ANY_CALL_RE.findall(line):
                ok = (name in IGOR_KEYWORDS or name in IGOR_BUILTINS or name in vendor
                      or name.startswith("SOCKIT")
                      or (name.startswith("AFM_") and name in defined))
                if not ok:
                    unknown.append(f"{path}:{i} {name}")
    assert not unknown, f"callees that are neither tabled nor allowed built-ins: {unknown}"


def test_callbacks_and_background_tasks_are_tabled():
    text = "\n".join(uncommented_lines(T.RUN))
    assert set(re.findall(r'"(AFM_\w+\(\))"', text)) == set(T.CALLBACKS)
    procs = dict(re.findall(r"CtrlNamedBackground\s+(\w+),[^\n]*proc=(AFM_\w+)", text))
    assert procs == {k: v[0] for k, v in T.BACKGROUND.items()}


def test_expander_handles_exactly_the_tabled_verbs():
    # GETSTR is answered by the default branch (AFM_DispatchOther), not by a case.
    assert X.WIRE_VERBS - {"GETSTR"} == set(T.VERBS) - {"default"}
    assert X.DRY_RUN_VERBS == set(T.DRY_VERBS)
    assert X.PSEUDO_VERBS == set(T.PSEUDO)


# ------------------------------------------------------- table against .ipf text --

@pytest.mark.parametrize("name", sorted(T.FUNCTIONS))
def test_function_entry_matches_ipf(name):
    e = T.FUNCTIONS[name]
    start, end, params = functions(e.file)[name]
    assert (e.start, e.end) == (start, end)
    assert e.params == params
    found = calls_in(e.file, start + 1, end)
    if name == "AFM_Dispatch":
        steps = [s for table in (T.DRY_VERBS, T.VERBS)
                 for v in sorted(table.values(), key=lambda v: v.line) for s in v.steps]
    else:
        steps = list(e.steps)
    assert [(s.name, s.line) for s in steps] == found


def test_verb_steps_lie_in_their_case_block():
    cases = dispatch_cases()
    for table, sw in ((T.DRY_VERBS, 0), (T.VERBS, 1)):
        for verb, e in table.items():
            lo, hi = cases[sw][verb]
            assert all(lo <= s.line <= hi for s in e.steps), verb


def test_step_arguments_match_ipf_text():
    seen: dict[tuple[str, int, str], int] = {}
    for path, s in _steps_with_files():
        key = (path, s.line, s.name)
        k = seen.get(key, 0)
        seen[key] = k + 1
        src = call_args(path, s.line, s.name, k)
        assert len(src) == len(s.args), f"{path}:{s.line} {s.name}: {src}"
        for a, text in zip(s.args, src, strict=True):
            if a.src == "literal":
                assert a.note == text, f"{path}:{s.line} {s.name}: {a.note!r} != {text!r}"
            assert a.src in {"wire", "igor", "literal", "state", "param"}


def test_param_args_name_a_parameter():
    for fn in T.FUNCTIONS.values():
        for s in fn.steps:
            for a in s.args:
                if a.src == "param":
                    assert a.note in fn.params, (fn.name, s.name, a.note)


def test_step_kinds():
    for _, s in _steps_with_files():
        if s.name in ACT_FNS:
            assert s.kind == "act", s
        elif s.name in READ_FNS:
            assert s.kind == "read", s
        elif s.name.startswith("SOCKIT"):
            assert s.kind == "xport", s
        else:
            assert s.kind == "afm" and s.name.startswith("AFM_"), s


def test_resolve_reaches_only_wire_igor_literal_or_state():
    for verb in T.VERBS:
        for r in T.resolve(verb):
            assert all(a.src in {"wire", "igor", "literal", "state"} for a in r.args), r
    for name in ("AFM_Done", "AFM_LithoDone", "AFM_ScanAdvance", "AFM_LithoAdvance",
                 "AFM_ScanTick", "AFM_LithoTick"):
        for r in T.resolve_function(name):
            assert all(a.src != "param" for a in r.args), r
    assert "ir_SetPISLoop" in T.format_table()


def test_resolved_servoloop_row():
    (r,) = T.resolve("SERVOLOOP")
    assert r.fn == "ir_SetPISLoop" and r.cite == f"{T.RUN}:584"
    assert [a.src for a in r.args] == ["wire", "literal", "wire", "wire", "wire", "wire",
                                       "literal", "wire", "wire", "wire"]


# ------------------------------------------------------ expander citations are real --

NUM_RE = re.compile(r"^-?\d+(\.\d*)?$")


def _matches(src: list[str], args: tuple) -> bool:
    """Every literal in the source call equals the emitted argument at that position."""
    if len(src) != len(args):
        return False
    for text, a in zip(src, args, strict=True):
        if text in T.IPF_CONSTANTS:
            text = f'"{T.IPF_CONSTANTS[text][0]}"'
        if text.startswith('"') and text.endswith('"') and text.count('"') == 2:
            if a != text[1:-1]:
                return False
        elif NUM_RE.match(text) and float(a) != float(text):
            return False
    return True


def test_expander_sites_cite_the_call():
    from test_igor_expander import ALL_SCENARIOS

    checked = 0
    for scenario in ALL_SCENARIOS:
        out = scenario()
        for call in out.actuation + out.reads:
            if call.fn == "DoScanFunc":
                continue
            path, line = call.site.split(" ")[0].rsplit(":", 1)
            names = [m.group(1) for m in CALL_RE.finditer(code_lines(path)[int(line)])]
            assert call.fn in names, f"{call} cites {call.site}"
            occurrences = [call_args(path, int(line), call.fn, k)
                           for k in range(names.count(call.fn))]
            assert any(_matches(src, call.args) for src in occurrences), (
                f"{call} does not match the literals at {call.site}: {occurrences}")
            checked += 1
    assert checked > 100


# ------------------------------------------------ expander follows the resolved table --

def _resolved(verb: str, drop: tuple[str, ...] = (), kind: str = "act") -> list[str]:
    return [r.fn for r in T.resolve(verb)
            if r.kind == kind and not any(d in r.when for d in drop)]


def test_expander_follows_table_for_litho_and_park():
    from test_igor_expander import (
        sc_armfire,
        sc_armfire_vcap_fail,
        sc_safepark_cypher_wire,
        sc_safepark_mfp,
    )

    assert [c.fn for c in sc_armfire().actuation][:-1] == _resolved("ARMFIRE", drop=("lErr",))
    assert [c.fn for c in sc_armfire_vcap_fail().actuation][:-1] == _resolved("ARMFIRE")
    assert [c.fn for c in sc_safepark_cypher_wire().actuation] == _resolved(
        "ABORT", drop=("not Cypher",))
    assert [c.fn for c in sc_safepark_mfp().actuation] == _resolved(
        "ABORT", drop=("cMicroscopeCypher",))


def test_expander_follows_table_for_chain_first_arm():
    sc = X.expand([P.cmd_wave("sx", np.zeros(64)), P.cmd_wave("sy", np.zeros(64)),
                   P.cmd_scanchain("5", "sx", "sy", 32, 2, -5, 5)])
    assert [c.fn for c in sc.actuation] == _resolved("SCANCHAIN", drop=("vErr",))
    lc = X.expand([P.cmd_wave(n, np.zeros(10)) for n in ("XLitho", "YLitho", "VLitho")]
                  + [P.cmd_wave("B", [0, 10]), P.cmd_lithochain("9", "o", -2, 2, "B")])
    assert [c.fn for c in lc.actuation] == _resolved("LITHOCHAIN", drop=("lErr",))


def _lit(note: str):
    if note in T.IPF_CONSTANTS:
        return T.IPF_CONSTANTS[note][0]
    if note.startswith('"') and note.endswith('"'):
        return note[1:-1]
    return float(note)


def _check_fn(name: str, calls, *, wire_line: str = "", sizes=None, drop=()) -> None:
    """Expander calls for one callback or tick equal resolve_function(name), value by value:
    literals by text, wire tokens by the dispatched line, wave arguments by length."""
    want = [r for r in T.resolve_function(name)
            if r.kind == "act" and not any(d in r.when for d in drop)]
    assert [c.fn for c in calls] == [r.fn for r in want], name
    for c, r in zip(calls, want, strict=True):
        assert len(c.args) == len(r.args), (name, str(c))
        for a, ra in zip(c.args, r.args, strict=True):
            if ra.src == "literal":
                lit = _lit(ra.note)
                assert (a == lit) if isinstance(lit, str) else float(a) == lit, (name, str(c))
            elif ra.src == "wire" and (m := re.search(r"tok (\d+)", ra.note)):
                tok = wire_line.split(" ")[int(m.group(1))]
                assert (a == tok) if isinstance(a, str) else float(a) == X.str2num(tok), (
                    name, str(c), ra.note)
            elif isinstance(a, X.Wave | X.CapWave):
                (n,) = [v for k, v in (sizes or {}).items() if k in ra.note]
                assert a.n == n, (name, str(c), ra.note)


def _acts_at(out, step: int):
    return [e.call for e in out.timeline if e.step == step and e.kind == "act"]


def test_callbacks_and_ticks_follow_resolve_function():
    from test_igor_expander import SX, SY, V100, X100, Y100, litho_waves, xy_transit_lines

    tr = X.expand([*xy_transit_lines(), ("@FILL", ())], auto_tick=False)
    _check_fn("AFM_Done", _acts_at(tr, 7))

    af = X.expand([*litho_waves(40), P.cmd_armfire("7", "o", -3, 64, 3), ("@FILL", ())])
    _check_fn("AFM_LithoDone", _acts_at(af, 4))

    scan = P.cmd_scanchain("5", "sx", "sy", 32, 2, -5, 5, "Amplitude")
    sc = X.expand([P.cmd_wave("sx", SX), P.cmd_wave("sy", SY), scan, ("@FILL", ()),
                   ("@SCANTICK", ()), ("@FILL", ())], auto_tick=False)
    _check_fn("AFM_ScanAdvance", _acts_at(sc, 3))
    _check_fn("AFM_ScanTick", _acts_at(sc, 4), wire_line=scan, sizes={"ppl points": 32},
              drop=("vErr",))
    _check_fn("AFM_ScanAdvance", _acts_at(sc, 5))

    chain = P.cmd_lithochain("9", "o", -2, 2, "B")
    lc = X.expand([P.cmd_wave("XLitho", X100), P.cmd_wave("YLitho", Y100),
                   P.cmd_wave("VLitho", V100), P.cmd_wave("B", [0, 60, 100]), chain,
                   ("@FILL", ()), ("@LITHOTICK", ()), ("@FILL", ())], auto_tick=False)
    _check_fn("AFM_LithoAdvance", _acts_at(lc, 5), drop=("last chunk",))
    _check_fn("AFM_LithoTick", _acts_at(lc, 6), wire_line=chain,
              sizes={"m points": 40, "capLen points": 64}, drop=("lErr",))
    _check_fn("AFM_LithoAdvance", _acts_at(lc, 7))


def test_expander_follows_table_for_startup_reads():
    out = X.expand(["STARTUP 1 2"], microscope_id=4)
    want = [(r.fn, r.args[0].note.strip('"')) for r in T.resolve("STARTUP")
            if "ImagingMode == 1" not in r.when]
    assert [(c.fn, c.args[0]) for c in out.reads] == want
