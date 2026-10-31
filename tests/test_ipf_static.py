"""Compile-class checks on the Igor procedures, run off the instrument.

Igor 6.37 rejects the whole procedure set when one function fails to compile, and the dev box
cannot compile it. These checks catch the compile errors that a text scan can find: unbalanced
blocks, a local declared twice in one function, a user function called with the wrong number
of arguments, a return of the wrong type, and names over Igor 6's 31 bytes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from test_igor_expander_table import _blank

IGOR = Path(__file__).resolve().parents[1] / "igor"
FILES = ("afm_litho_run.ipf", "afm_igor.ipf", "afm_sockit.ipf")
FUNC = re.compile(r"^\s*Function(/S|/D)?\s+(\w+)\s*\(([^)]*)\)", re.I)
END = re.compile(r"^\s*End\s*$", re.I)
DECL = re.compile(r"^\s*(Variable|String|NVAR|SVAR|WAVE|Wave)((?:/\w+)*)\s+(.*)$")
CALL = re.compile(r"\b(AFM_\w+)\s*\(")
IGOR_NAME_MAX = 31


def _code(name: str) -> list[str]:
    text = (IGOR / name).read_bytes().replace(b"\r\n", b"\n").decode("latin-1")
    return [_blank(r) for r in text.split("\n")]


def _raw(name: str) -> list[str]:
    text = (IGOR / name).read_bytes().replace(b"\r\n", b"\n").decode("latin-1")
    return text.split("\n")


def _functions(name: str):
    """(name, kind, params, optional params, first line index, body lines) per function."""
    code, out, cur = _code(name), [], None
    for i, line in enumerate(code):
        m = FUNC.match(line)
        if m:
            sig = m.group(3)
            opt = re.findall(r"\[([^\]]*)\]", sig)
            optional = [p.strip() for o in opt for p in o.split(",") if p.strip()]
            required = [p.strip() for p in re.sub(r"\[[^\]]*\]", "", sig).split(",")
                        if p.strip()]
            cur = (m.group(2), (m.group(1) or "").upper(), required, optional, i, [])
        elif cur and END.match(line):
            out.append(cur)
            cur = None
        elif cur:
            cur[5].append((i, line))
    return out


def _split_top(text: str) -> list[str]:
    parts, depth, cur = [], 0, []
    for c in text:
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(c)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _all_functions():
    return [(f, fn) for f in FILES for fn in _functions(f)]


DEFINED = {fn[0]: fn for _, fn in _all_functions()}


@pytest.mark.parametrize("name", FILES)
def test_blocks_balance_in_every_function(name):
    pairs = (("if", r"^\s*if\s*\(", r"^\s*endif\b"),
             ("switch", r"^\s*(str)?switch\s*\(", r"^\s*endswitch\b"),
             ("do", r"^\s*do\b", r"^\s*while\s*\("),
             ("for", r"^\s*for\s*\(", r"^\s*endfor\b"))
    for fn, _kind, _req, _opt, start, body in _functions(name):
        for label, op, cl in pairs:
            depth = 0
            for i, line in body:
                depth += bool(re.match(op, line)) - bool(re.match(cl, line))
                assert depth >= 0, f"{name}:{i + 1} {fn}: {label} closed before it opened"
            assert depth == 0, f"{name}:{start + 1} {fn}: unbalanced {label}"


@pytest.mark.parametrize("name", FILES)
def test_no_local_declared_twice(name):
    for fn, _kind, req, opt, _start, body in _functions(name):
        seen: dict[str, int] = {}
        params = set(req) | set(opt)
        for i, line in body:
            m = DECL.match(line)
            if not m or "/G" in m.group(2).upper():
                continue
            for item in _split_top(m.group(3)):
                local = re.match(r"&?\s*(\w+)", item.lstrip("&")).group(1)
                if local in params:
                    params.discard(local)              # the parameter's own declaration
                    continue
                assert local not in seen, (
                    f"{name}:{i + 1} {fn}: '{local}' declared again (first at {seen[local]})")
                seen[local] = i + 1


def test_user_calls_have_the_right_argument_count():
    bad = []
    for name, (fn, _kind, _req, _opt, _start, body) in _all_functions():
        for i, line in body:
            for m in CALL.finditer(line):
                callee = m.group(1)
                if callee not in DEFINED:
                    bad.append(f"{name}:{i + 1} {fn} calls undefined {callee}")
                    continue
                depth, j, cur = 0, m.end(), []
                while j < len(line):
                    c = line[j]
                    if c in "([":
                        depth += 1
                    elif c in ")]":
                        if depth == 0:
                            break
                        depth -= 1
                    cur.append(c)
                    j += 1
                args = _split_top("".join(cur))
                pos = [a for a in args if not re.match(r"^\w+\s*=", a)]
                _n, _k, req, opt, _s, _b = DEFINED[callee]
                if not (len(req) <= len(pos) <= len(req) + len(opt)):
                    bad.append(f"{name}:{i + 1} {fn}: {callee} takes {len(req)} (+{len(opt)}),"
                               f" given {len(pos)}")
    assert not bad, "\n".join(bad)


def test_returns_match_the_function_type():
    bad = []
    string_fns = {n for n, fn in DEFINED.items() if fn[1] == "/S"}
    for name, (fn, kind, _req, _opt, _start, body) in _all_functions():
        for i, line in body:
            m = re.match(r"^\s*return\b\s*(.*)$", line)
            if not m:
                continue
            expr = m.group(1).strip()
            raw = _raw(name)[i].split("//")[0]
            if not expr:
                bad.append(f"{name}:{i + 1} {fn}: return with no value")
            elif kind == "/S" and re.fullmatch(r"-?\d+", expr):
                bad.append(f"{name}:{i + 1} {fn}: Function/S returns a number")
            elif kind != "/S" and ('"' in raw.split("return", 1)[1][:2]
                                   or (CALL.match(expr) and CALL.match(expr).group(1)
                                       in string_fns and expr.endswith(")")
                                       and expr.count("(") == 1)):
                bad.append(f"{name}:{i + 1} {fn}: numeric Function returns a string")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("name", FILES)
def test_names_fit_igor6(name):
    raw = "\n".join(_raw(name))
    names = set(re.findall(r"^\s*Function(?:/\w+)?\s+(\w+)", raw, re.M | re.I))
    names |= set(re.findall(r"root:(\w+)", raw))
    names |= set(re.findall(r"^\s*(?:Str)?Constant\s+(\w+)", raw, re.M))
    names |= set(re.findall(r"CtrlNamedBackground\s+(\w+)", raw))
    long = sorted(n for n in names if len(n) > IGOR_NAME_MAX)
    assert not long, f"names over {IGOR_NAME_MAX} bytes: {long}"


def test_background_procs_have_the_documented_signature():
    raw = "\n".join(_raw("afm_litho_run.ipf"))
    procs = set(re.findall(r"^\s*CtrlNamedBackground\b.*\bproc=(\w+)", raw, re.M))
    assert procs
    for proc in procs:
        m = re.search(rf"^Function {proc}\((\w+)\)\s*\n\s*STRUCT WMBackgroundStruct &\1",
                      raw, re.M)
        assert m, f"{proc} lacks Function {proc}(s) + STRUCT WMBackgroundStruct &s"


def test_abort_button_runs_safe_park_and_nothing_else():
    """The abort panel's button is the escape when the service is dead: on mouse up it calls
    AFM_SafePark and no other procedure; it only reads Height back for the status line."""
    body = _body("afm_litho_run.ipf", "AFM_AbortButtonProc")
    code = "\n".join(line for _i, line in DEFINED["AFM_AbortButtonProc"][5])  # strings blanked
    calls = re.findall(r"\b(\w+)\s*\(", code)
    assert calls == ["if", "AFM_SafePark", "td_ReadValue", "num2str"], calls
    assert 'td_ReadValue("Height")' in body
    assert "ba.eventCode != 2" in body
    assert "proc=AFM_AbortPanelTick" in body             # the one-shot 6 s status check
    raw = "\n".join(_raw("afm_litho_run.ipf"))
    assert re.search(r"^Function AFM_AbortButtonProc\(ba\) : ButtonControl\s*\n"
                     r"\s*STRUCT WMButtonAction &ba", raw, re.M)
    assert re.search(r"^\s*Button abortBtn,.*\bproc=AFM_AbortButtonProc\b", raw, re.M)
    assert re.search(r'^\s*"AFM Abort panel", AFM_ShowAbortPanel\(\)', raw, re.M)


SEND = re.compile(r"\b(AFM_SendStr|SOCKIT\w+)\s*\(")
CLEAR = re.compile(r"^\s*(Variable\s+)?\w+\s*=\s*GetRTError\(1\)\s*$")


def _clears_after(body: list[tuple[int, str]], k: int) -> bool:
    rest = [line for _i, line in body[k + 1:] if line.strip()]
    return bool(rest) and bool(CLEAR.match(rest[0]))


def _unguarded_sends(fn: str, seen: set[str]) -> list[str]:
    """Sends reachable from `fn` whose runtime error is not cleared on the next code line,
    either right after the raw send or right after a call that leads to one."""
    if fn in seen or fn not in DEFINED:
        return []
    seen.add(fn)
    body = DEFINED[fn][5]
    bad = []
    for k, (i, line) in enumerate(body):
        hits = [m.group(1) for m in SEND.finditer(line)]
        hits += [c for c in CALL.findall(line) if c != fn and _sends(c, set())]
        for callee in hits:
            if _clears_after(body, k):
                continue
            if SEND.match(callee + "("):
                bad.append(f"{fn}:{i + 1} {callee}")
            else:
                bad += _unguarded_sends(callee, seen)
    return bad


def _sends(fn: str, seen: set[str]) -> bool:
    if fn in seen or fn not in DEFINED:
        return False
    seen.add(fn)
    body = DEFINED[fn][5]
    return any(SEND.search(line) or any(_sends(c, seen) for c in CALL.findall(line))
               for _i, line in body)


def test_abort_path_survives_a_dead_socket():
    """Review (abort panel): a send to a closed link may raise a runtime error, and an uncleared
    one stops the function before the retract. Every socket send reachable from AFM_SafePark
    is followed by GetRTError(1), at the send or at the call that leads to it."""
    assert _sends("AFM_Abort", set()) and _sends("AFM_SafePark", set())
    assert _unguarded_sends("AFM_SafePark", set()) == []
    abort = DEFINED["AFM_Abort"][5]
    frames = [k for k, (_i, line) in enumerate(abort) if "AFM_SendRunFrame(" in line]
    assert len(frames) == 2 and all(_clears_after(abort, k) for k in frames)


def test_ipf_sha_constant_is_on_one_line_only():
    """The install script blanks the one line holding the hash before hashing (contract 11.7)."""
    hits = [ln for ln in _raw("afm_litho_run.ipf") if "StrConstant kIpfSha" in ln]
    assert len(hits) == 1 and re.fullmatch(r'StrConstant kIpfSha = "[0-9a-f]{16}"', hits[0])


def test_install_stamp_is_the_hash_the_service_expects(tmp_path):
    """scripts/install_afm_ipf.py writes kIpfSha; the hash blanks that line, so stamping keeps
    it, and it equals what the v3 client expects of the shipped copy (contract 11.7)."""
    import importlib.util

    from afm_litho.hardware import wire_v3

    spec = importlib.util.spec_from_file_location(
        "install_afm_ipf", IGOR.parent / "scripts" / "install_afm_ipf.py")
    inst = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inst)
    src = (IGOR / "afm_litho_run.ipf").read_bytes()
    stamped = inst.stamp_ipf_sha(src)
    want = wire_v3.expected_ipf_sha(IGOR / "afm_litho_run.ipf")
    assert inst.ipf_sha(src) == inst.ipf_sha(stamped) == want
    assert f'StrConstant kIpfSha = "{want}"'.encode() in stamped
    assert len(stamped.split(b"\n")) == len(src.split(b"\n"))
    (tmp_path / "User Procedures").mkdir()
    assert inst._copy(IGOR / "afm_litho_run.ipf", tmp_path / "User Procedures")
    assert (tmp_path / "User Procedures" / "afm_litho_run.ipf").read_bytes() == stamped


@pytest.mark.parametrize("name", FILES)
def test_lines_stay_under_400_characters(name):
    """The 400-character ceiling (ipf-syntax-checklist item 37), held for procedure lines too."""
    long = [i + 1 for i, ln in enumerate(_raw(name)) if len(ln) >= 400]
    assert not long, f"{name}: lines of 400 characters or more: {long}"


def _body(name: str, fn: str) -> str:
    for f, _k, _r, _o, _s, body in _functions(name):
        if f == fn:
            raw = _raw(name)
            return "\n".join(raw[i] for i, _line in body)
    raise AssertionError(f"{fn} not in {name}")


def test_run_frames_route_by_the_live_run_not_the_uid():
    """Review M1: AFM_LithoRunStart blanks AFM_lrUID/AFM_scUID on a reconnect, so a pass frame
    sent after it carries uid "". Routing on the uid left the v3 run live forever (STATUS stuck,
    PLAY busy, the guard waiting). The router must look only at whether a v3 run is live, and
    CAP must cut the header by the frame's own uid."""
    route = _body("afm_litho_run.ipf", "AFM_SendRunFrame")
    assert "StringFromList(1" not in route and "AFM_v3Run" in route
    cap = _body("afm_litho_run.ipf", "AFM_V3Cap")
    assert 'strlen("CAPTURE " + uid' in cap and "uid = StringFromList(1, msg" in cap
    start = _body("afm_litho_run.ipf", "AFM_LithoRunStart")
    assert "AFM_V3LinkOpened()" in start and "AFM_v3Run" not in start   # v3 run state survives


SHIPPED = tuple(f for f in ("afm_litho.ipf", *FILES) if (IGOR / f).exists())       # what scripts/install_afm_ipf.py installs together
PROC = re.compile(r"^\s*(?:Static\s+)?(?:ThreadSafe\s+)?(?:Function|Macro|Proc|Window|Picture)"
                  r"(?:/\w+)*\s+(\w+)", re.I)
GLOBAL_DECL = re.compile(r"^\s*(?:Variable|String)((?:/\w+)*)\s+(.*)$", re.I)
REF_DECL = re.compile(r"^\s*(?:NVAR|SVAR|WAVE)(?:/\w+)*\s+(.*)$", re.I)
MAKE = re.compile(r"^\s*Make((?:\s*/\w+(?:=(?:\([^)]*\)|[^\s/]+))?)*)\s+(.*)$", re.I)
BG_TASK = re.compile(r"\bCtrlNamedBackground\s+(\w+)", re.I)


def _leaf(ref: str) -> str | None:
    """The object name at the end of a path such as root:AFM_x, or None for a $name."""
    path = ref.split("=")[0].split("[")[0].strip()
    leaf = path.split(":")[-1]
    return leaf if "$" not in path and re.fullmatch(r"\w+", leaf) else None


def _globals(name: str) -> dict[str, str]:
    """Lower-cased name -> "file:line" for every global the procedures create or bind."""
    out = {}
    for i, line in enumerate(_code(name)):
        for stmt in line.split(";"):
            names = list(BG_TASK.findall(stmt))
            m = GLOBAL_DECL.match(stmt)
            if m and "/G" in m.group(1).upper():
                names += [_leaf(p) for p in _split_top(m.group(2))]
            m = REF_DECL.match(stmt)
            if m:
                names += [_leaf(p.split("=", 1)[-1]) for p in _split_top(m.group(1))]
            m = MAKE.match(stmt)
            if m:
                names += [_leaf(p) for p in _split_top(m.group(2).split("{")[0])]
            for n in names:
                if n:
                    out.setdefault(n.lower(), f"{name}:{i + 1} {n}")
    return out


def test_no_procedure_shares_a_name_with_a_global():
    """Igor names are case-insensitive, and a function named like an existing global
    (AFM_AbortFault vs root:AFM_abortFault) fails to compile with "name already exists as a
    variable" once that global exists, so every recompile after the first fault failed."""
    procs, made = {}, {}
    for f in SHIPPED:
        for i, line in enumerate(_code(f)):
            m = PROC.match(line)
            if m:
                procs.setdefault(m.group(1).lower(), f"{f}:{i + 1} {m.group(1)}")
        for k, where in _globals(f).items():
            made.setdefault(k, where)
    assert len(procs) > 100 and len(made) > 50          # the scan sees the procedure set
    clash = [f"{procs[k]} vs {made[k]}" for k in sorted(procs.keys() & made.keys())]
    assert not clash, "function and global share a name: " + "; ".join(clash)
