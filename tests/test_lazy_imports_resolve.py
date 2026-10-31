"""Every lazy import on the stop path resolves (H5.pre).

A function-level import is not run until its function is, so a module moved by `git mv`
leaves a stale one that fails only when a withdraw or park runs. This walks the stop-path
files, resolves each in-function import of afm_litho code to a module and, for `from M import
N`, checks N is an attribute or a submodule of M. Third-party imports (pythoncom) are
skipped. An optional extra that is not installed (OPTIONAL_PACKAGES: klayout, the `geometry`
extra) is skipped; any other missing package is reported.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
PKG = SRC / "afm_litho"

# Everything the stops run through: the whole safety package, the host's park, stop port and
# after-death and continuous files, the fail-safe watchdog thread, the hardware withdraw and
# retract code, the stop rows, the keeper copy and the Asylum plugin.
STOP_PATH_FILES = sorted(
    [*(PKG / "service" / "safety").glob("*.py"),
     PKG / "service" / "host" / "safe_park.py",
     PKG / "service" / "host" / "stop_port.py",
     PKG / "service" / "host" / "after_death.py",
     PKG / "service" / "host" / "continuous.py",
     PKG / "service" / "host" / "scan_cmds.py",
     PKG / "service" / "host" / "litho_cmds.py",
     PKG / "service" / "host" / "continuous_cmds.py",
     PKG / "service" / "host" / "api.py",
     PKG / "hardware" / "failsafe_thread.py",
     PKG / "hardware" / "hw_approach.py",
     PKG / "hardware" / "retract.py",
     PKG / "service" / "requests" / "stops.py",
     PKG / "service" / "keeper_stop.py",
     PKG / "instrument" / "asylum.py"])


# Optional extras a dev box may lack; a stop-path module failing on any other import is a defect.
OPTIONAL_PACKAGES = {"klayout"}


def _absolute(home: Path, level: int, module: str | None) -> str:
    parts = list(home.relative_to(SRC).with_suffix("").parts)
    is_pkg = parts[-1] == "__init__"
    if is_pkg:
        parts.pop()
    base = parts if is_pkg else parts[:-1]
    base = base[: len(base) - (level - 1)]
    return ".".join([*base, *([module] if module else [])])


def lazy_imports(source: str, home: Path) -> list[tuple[int, str, list[str]]]:
    """(line, absolute module, imported names) for every afm_litho import inside a function.
    `home` is the path the source lives at, which fixes what a relative import means."""
    out = []
    for fn in ast.walk(ast.parse(source)):
        if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.ImportFrom):
                mod = _absolute(home, node.level, node.module) if node.level else node.module
                out.append((node.lineno, mod, [a.name for a in node.names]))
            elif isinstance(node, ast.Import):
                out += [(node.lineno, a.name, []) for a in node.names]
    return [t for t in out if t[1] and t[1].split(".")[0] == "afm_litho"]


def problems(source: str, home: Path) -> list[str]:
    bad = []
    for line, mod, names in lazy_imports(source, home):
        try:
            m = importlib.import_module(mod)
        except ImportError as exc:
            if exc.name and exc.name.split(".")[0] in OPTIONAL_PACKAGES:
                continue                     # a missing third-party package, not a stale path
            bad.append(f"{home.name}:{line} {mod}: {exc}")
            continue
        for n in names:
            if n == "*" or hasattr(m, n):
                continue
            try:
                importlib.import_module(f"{mod}.{n}")
            except ImportError as exc:
                if exc.name and exc.name.split(".")[0] in OPTIONAL_PACKAGES:
                    continue
                bad.append(f"{home.name}:{line} {mod} has no {n}")
    return bad


CASES = [pytest.param(p, id=str(p.relative_to(PKG)).replace("\\", "/")) for p in STOP_PATH_FILES]


def test_the_stop_path_files_exist_and_have_lazy_imports():
    assert all(p.is_file() for p in STOP_PATH_FILES)
    n = sum(len(lazy_imports(p.read_text(encoding="utf-8"), p)) for p in STOP_PATH_FILES)
    assert n >= 10


@pytest.mark.parametrize("path", CASES)
def test_every_lazy_import_resolves(path):
    assert not problems(path.read_text(encoding="utf-8"), path)


def test_the_check_catches_a_stale_import():
    home = PKG / "service" / "host" / "safe_park.py"
    source = home.read_text(encoding="utf-8")
    assert "from . import scan_cmds" in source
    stale = source.replace("from . import scan_cmds",
                           "from . import scan_cmds_gone")
    found = problems(stale, home)
    assert found and "scan_cmds_gone" in found[0]
    moved = source.replace("from . import scan_cmds",
                           "from ..host_commands import scan_cmds")
    assert any("host_commands" in p for p in problems(moved, home))


def test_a_missing_third_party_package_is_not_a_problem(monkeypatch):
    home = PKG / "service" / "host" / "api.py"
    real = importlib.import_module

    def without_klayout(name, *args):
        if name.endswith("oasis"):
            raise ModuleNotFoundError("No module named 'klayout'", name="klayout")
        return real(name, *args)

    monkeypatch.setattr(importlib, "import_module", without_klayout)
    assert problems("def f():\n    from afm_litho.litho import oasis\n", home) == []
    assert problems("def f():\n    from afm_litho.litho.oasis import read\n", home) == []
    monkeypatch.undo()

    def without_other(name, *args):
        if name.endswith("oasis"):
            raise ModuleNotFoundError("No module named 'numpyish'", name="numpyish")
        return real(name, *args)

    monkeypatch.setattr(importlib, "import_module", without_other)
    other = problems("def f():\n    from afm_litho.litho.oasis import read\n", home)
    assert len(other) == 1 and "numpyish" in other[0]
    monkeypatch.undo()
    found = problems("def f():\n    from afm_litho.no_such_module import x\n", home)
    assert len(found) == 1 and "no_such_module" in found[0]
