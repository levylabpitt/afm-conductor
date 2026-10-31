"""The session record and the stem names a scan and a write leave behind (H5.pre).

The Cypher golden scenarios judge what reaches Igor; nothing judged what the host writes
afterwards. This records one chained scan and one single-wire write on the Igor stub and
freezes: which files appear in the runtime folder (stem and run id masked), the session line
and the stem-index line each run appends, and the shape of the run record. The persistence
steps move at H5b/H7.5, and a step dropped, renamed or reordered shows here.

Frozen in tests/golden/persistence/session_and_stems.json. Regenerate only with
AFM_LITHO_REGEN_GOLDENS=1, in a commit of its own.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from golden.harness import HERE, record

from afm_litho.goldens import golden_format as G

FROZEN = HERE / "persistence" / "session_and_stems.json"
SCENARIOS = ("scan_chained_contact", "litho_single")
SESSION_KEYS = ("kind", "label", "backend", "aborted", "capture_channels")

STEM = re.compile(r"(?<![0-9])20\d{6}_\d{6}(?:_\d{2}(?![0-9]))?")
_MASKS = (
    (STEM, "<stem>"),
    (re.compile(r"\b(scan|litho|iv)-\d{10,}\b"), r"\1-<ms>"),
    (re.compile(r"\b([sw])-\d{4}-\d{2}-\d{2}-\d{4}-\d{2}\b"), r"\1-<date>-<n>"),
)


def _mask(text: str) -> str:
    for pat, repl in _MASKS:
        text = pat.sub(repl, text)
    return text


def _files(root: Path) -> dict[str, int]:
    return {p.relative_to(root).as_posix(): p.stat().st_size for p in root.rglob("*")
            if p.is_file()}


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _summarise(root: Path, sample: str, before_files: dict, before_sessions: int,
               before_index: int) -> dict:
    now = _files(root)
    new = sorted(k for k, v in now.items()
                 if before_files.get(k) != v and not k.startswith("journal/"))
    session = _jsonl(root / "sessions.jsonl")[before_sessions:]
    index = _jsonl(root / "stem_index.jsonl")[before_index:]
    assert len(session) == 1 and len(index) == 1, (session, index)
    (sess,), (idx,) = session, index
    runs = [json.loads((root / k).read_text(encoding="utf-8")) for k in new
            if k.startswith("runs/") and k.endswith(".run.json")]
    assert len(runs) == 1, new
    run = runs[0]
    stems = {m.group(0) for k in new for m in re.finditer(r"\b20\d{6}_\d{6}(_\d{2})?\b", k)}
    # one stem names every file the run wrote, and the index and the session agree on the run
    assert stems == {idx["stem"]}, (stems, idx)
    assert sess["id"] == idx["run_id"], (sess["id"], idx)
    return {
        "files": sorted({_mask(k) for k in new}),
        "all_files_non_empty": all(now[k] > 0 for k in new),
        "session_keys": sorted(sess),
        "session": {k: sess.get(k) for k in SESSION_KEYS} | {
            "label": str(sess.get("label")).replace(sample, "<sample>")},
        "written": [w["name"] for w in sess.get("written", [])],
        "stem_index": {"keys": sorted(idx), "kind": idx["kind"],
                       "run_id": _mask(idx["run_id"]), "stem": _mask(idx["stem"])},
        "run_record": {
            "keys": sorted(run),
            "execution_keys": sorted(run.get("execution", {})),
            "execution_id": _mask(str(run.get("execution", {}).get("execution_id"))),
            "recipe_kind": run["recipe"]["kind"],
            "recipe_steps": [s["kind"] for s in run["recipe"]["body"]],
            "recipe_capture_channels": run["recipe"]["capture"]["channels"],
        },
    }


def _record_all(hub, client) -> dict:
    root = Path(hub.store.path.parent)
    out = {}
    for name in SCENARIOS:
        before = _files(root)
        n_sessions = len(_jsonl(root / "sessions.jsonl"))
        n_index = len(_jsonl(root / "stem_index.jsonl"))
        record(hub, name, client=client)
        out[name] = _summarise(root, hub.sample_name, before, n_sessions, n_index)
    return out


def _diff(want, got, path="") -> list[str]:
    """Where `got` differs from `want`, one line per difference."""
    if isinstance(want, dict) and isinstance(got, dict):
        out = []
        for k in sorted(set(want) | set(got)):
            out += _diff(want.get(k), got.get(k), f"{path}/{k}")
        return out
    return [] if want == got else [f"{path}: frozen {want!r}, now {got!r}"]


@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


def test_a_scan_and_a_write_leave_the_frozen_session_record_and_stems(hub, client):
    got = _record_all(hub, client)
    if G.regen_requested():
        FROZEN.parent.mkdir(exist_ok=True)
        FROZEN.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n", encoding="utf-8",
                          newline="\n")
        pytest.skip("regenerated session_and_stems.json")
    assert FROZEN.exists(), f"no frozen file; regenerate with {G.REGEN_ENV}=1"
    assert not _diff(json.loads(FROZEN.read_text(encoding="utf-8")), got)
