"""The frozen golden-trace file: two levels, rendered from one recording.

A golden holds, per scenario:

* `wire`: what the driver sent the stub. `actuation` is the exact ordered list (UIDs labelled,
  waves summarised, out-of-socket and stub-side entries as pseudo-verbs); `reads` is the sorted
  set of distinct reads with UIDs dropped, checked as a subset of a new recording.
* `igor`: the expander's end-of-line vendor calls for that recording, same two rules, except
  that Igor function names compare case-insensitively (Igor resolves them so; SLICES F22).
  String arguments stay exact.
* `known_defects`: the defects this golden freezes as today's behaviour. An intended fix
  re-freezes the golden and removes its entry.

Format and the regeneration rule: docs/migration/slices/0.2a-golden-format.md and
tests/golden/README.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from . import ipf_table
from .expander import Expansion
from .wire_tap import WireTap

GOLDEN_FORMAT = 2
REGEN_ENV = "AFM_LITHO_REGEN_GOLDENS"
REPO = Path(__file__).resolve().parents[3]
IPF_FILES = (ipf_table.RUN, ipf_table.IGOR, ipf_table.SOCK)


def regen_requested() -> bool:
    return os.environ.get(REGEN_ENV, "").strip() == "1"


def file_sha256(path: Path) -> str:
    """sha256 with CRLF normalised to LF, so a checkout's line-ending setting does not matter."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def ipf_hashes() -> dict[str, str]:
    return {rel: file_sha256(REPO / rel) for rel in IPF_FILES}


def render_entry(e: dict, *, uid: bool = True) -> str:
    """One tap entry as one line. `uid=False` drops a read's label, which counts reads."""
    args = list(e["args"])
    if e["verb"] in ("WAVE", "WAVEAPP") and args and isinstance(args[0], list):
        name, n, first, last, sha = args[0]
        args = [f"wave({name},n={n},first={first},last={last},sha={sha})"]
    elif not uid and e["kind"] == "read" and args and str(args[0]).startswith("r"):
        args = args[1:]
    return " ".join([e["verb"], *(str(a) for a in args)])


def wire_level(tap: WireTap) -> dict:
    snap = tap.snapshot()
    return {"actuation": [render_entry(e) for e in snap["actuation"]],
            "reads": sorted({render_entry(e, uid=False) for e in snap["reads"]})}


def igor_level(out: Expansion) -> dict:
    return {"actuation": out.lines("act"), "reads": sorted(set(out.lines("read")))}


def build(scenario: str, tap: WireTap, out: Expansion, *, microscope_id: int,
          door: str, fixture: dict[str, str], known_defects: list[dict] | None = None) -> dict:
    return {"format": GOLDEN_FORMAT, "scenario": scenario, "door": door,
            "microscope_id": microscope_id, "ipf_sha256": ipf_hashes(), "fixture": fixture,
            "known_defects": list(known_defects or []),
            "wire": wire_level(tap), "igor": igor_level(out)}


_FN = re.compile(r"^([A-Za-z_]\w*)\(")


def fold_fn(line: str) -> str:
    """An end-of-line call with its function name case-folded; the arguments untouched."""
    return _FN.sub(lambda m: m.group(1).casefold() + "(", line, count=1)


def dumps(golden: dict) -> bytes:
    """Canonical bytes: sorted keys, one list item per line, LF, trailing newline."""
    return (json.dumps(golden, sort_keys=True, indent=1, ensure_ascii=True) + "\n").encode()


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(expected: dict, actual: dict) -> list[str]:
    """Problems found judging `actual` against the frozen `expected`; empty means a match.

    Actuation must be equal at both levels; frozen reads must all still occur (new reads are
    allowed). Format, scenario, platform, `.ipf` and fixture hashes and the known defects must
    match. At the `igor` level function names compare case-folded, arguments exactly.
    """
    problems: list[str] = []
    for key in ("format", "scenario", "door", "microscope_id", "ipf_sha256", "fixture",
                "known_defects"):
        if expected.get(key) != actual.get(key):
            problems.append(f"{key}: frozen {expected.get(key)!r} != now {actual.get(key)!r}")
    for level in ("wire", "igor"):
        exp, act = expected[level], actual[level]
        if level == "igor":
            exp = {k: [fold_fn(x) for x in v] for k, v in exp.items()}
            act = {k: [fold_fn(x) for x in v] for k, v in act.items()}
        if exp["actuation"] != act["actuation"]:
            problems.append(f"{level}.actuation differs:\n" + _first_diff(
                exp["actuation"], act["actuation"]))
        missing = sorted(set(exp["reads"]) - set(act["reads"]))
        if missing:
            problems.append(f"{level}.reads lost {len(missing)}: {missing[:5]}")
    return problems


def _first_diff(a: list[str], b: list[str]) -> str:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return f"  [{i}] frozen: {x}\n  [{i}] now:    {y}"
    return f"  length frozen {len(a)} != now {len(b)}"


__all__ = ["GOLDEN_FORMAT", "REGEN_ENV", "build", "compare", "dumps", "file_sha256",
           "fold_fn", "igor_level", "ipf_hashes", "load", "regen_requested", "render_entry", "wire_level"]
