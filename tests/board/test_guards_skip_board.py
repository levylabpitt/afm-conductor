"""The safety guards read HostState directly and never the board.

A dead board must not silence the dead-man (phase6-board plan, contracts). The guard
modules may not import `service.board`, in any import form.
"""

from __future__ import annotations

import ast
from pathlib import Path

SERVICE = Path(__file__).parents[2] / "src" / "afm_litho" / "service"
SRC = SERVICE.parent

GUARD_FILES = [
    SERVICE / "access.py",             # dead-man token release
    SERVICE / "control.py",            # dead-man watcher
    SERVICE / "host_state.py",
    SERVICE / "routers" / "continuous.py",   # rail guard and interlock feed
    SERVICE / "routers" / "tip.py",
    *sorted((SERVICE / "safety").rglob("*.py")),
    SERVICE / "host" / "safe_park.py",
    SERVICE / "host" / "deadman.py",   # dead-man verdict, engaged-idle cap
    SERVICE / "host" / "engage.py",
    SERVICE / "host" / "stop_port.py",
    SERVICE / "host" / "after_death.py",
    *sorted((SRC / "instrument").rglob("*.py")),
]


def _imports_board(path: Path) -> list[int]:
    hits = []
    for node in ast.walk(ast.parse(path.read_text("utf-8"))):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[-1] == "board" for a in node.names):
                hits.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            names = {a.name for a in node.names}
            if (node.module or "").split(".")[-1] == "board" or (
                    "board" in names and (node.module or "").split(".")[-1] in ("", "service")):
                hits.append(node.lineno)
    return hits


def test_guard_files_exist():
    assert all(p.exists() for p in GUARD_FILES)
    assert len(GUARD_FILES) > 6


def test_no_guard_imports_the_board():
    offenders = {str(p.relative_to(SRC)): _imports_board(p) for p in GUARD_FILES}
    assert {k: v for k, v in offenders.items() if v} == {}


def test_the_detector_sees_a_board_import(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("from .. import board\nfrom ..board import status\nimport x.board\n")
    assert _imports_board(bad) == [1, 2, 3]


def test_get_state_never_calls_the_board(monkeypatch):
    """The dead-man and the cap derive their state with host deadman.pass_state, which
    flex_dispatch.get_state calls, so neither may touch the board at run time or by name."""
    from afm_litho.service import board, flex_dispatch
    from afm_litho.service.hub import hub

    def boom(*_a, **_k):
        raise AssertionError("get_state called the board")

    for name in ("state", "status", "telemetry", "snapshot", "ws_frame"):
        monkeypatch.setattr(board, name, boom)
    assert flex_dispatch.get_state(hub) in {
        "idle", "engaged", "fault", "disconnected", "parked", "scanning", "writing",
        "approaching"}

    tree = ast.parse((SERVICE / "flex_dispatch.py").read_text("utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("get_state", "_write_in_flight"):
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            attrs = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
            assert "board" not in names | attrs, node.name
