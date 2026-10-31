"""Every module under src/ must compile, including ones nothing imports yet.

A SyntaxError in a module the service imports at start-up kills the whole service. Compiling
every file finds all of them at once. The repo has no CI config, so this test is the lint step.
"""

from __future__ import annotations

import py_compile
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def test_every_module_under_src_compiles(tmp_path):
    """The bytecode goes to `tmp_path`, so the run never writes under src/."""
    modules = sorted(SRC.rglob("*.py"))
    assert len(modules) > 10, f"no modules found under {SRC}"
    broken = []
    for module in modules:
        out = tmp_path / module.relative_to(SRC).with_suffix(".pyc")
        try:
            py_compile.compile(str(module), cfile=str(out), doraise=True)
        except py_compile.PyCompileError as exc:
            broken.append(str(exc))
    assert not broken, "\n".join(broken)
