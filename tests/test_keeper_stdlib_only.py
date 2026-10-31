"""The keeper stays up when the service's dependencies are what broke, and it never
talks to Igor. So it may import only the standard library, plus afm_litho.clock, afm_litho.wire_version,
afm_litho.site and its own package."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

KEEPER = Path(__file__).resolve().parents[1] / "src" / "afm_litho" / "keeper"
ALLOWED_AFM = ("afm_litho.keeper", "afm_litho.clock", "afm_litho.wire_version",
               "afm_litho.site", "afm_litho.__main__")
BANNED_TOP_LEVEL = ("fastapi", "starlette", "uvicorn", "numpy", "scipy", "pydantic",
                    "nptdms", "win32com", "pythoncom", "zmq", "httpx", "websockets")


def _imported_modules(path: Path) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.append(node.module)
            if node.module == "afm_litho":
                names += [f"afm_litho.{a.name}" for a in node.names]
    return names


def test_every_import_in_the_keeper_source_is_standard_library_or_allowed():
    files = sorted(KEEPER.glob("*.py"))
    assert len(files) >= 6
    for path in files:
        for name in _imported_modules(path):
            top = name.split(".")[0]
            if name == "afm_litho":
                continue
            if top == "afm_litho":
                assert name.startswith(ALLOWED_AFM), f"{path.name} imports {name}"
            else:
                assert top in sys.stdlib_module_names, f"{path.name} imports {name}"


def test_importing_the_keeper_loads_no_service_hardware_or_third_party_module():
    """Checked in a fresh interpreter, since this one has the service loaded."""
    code = (
        "import sys\n"
        "import afm_litho.keeper.cli, afm_litho.keeper.process, afm_litho.keeper.status\n"
        "import afm_litho.keeper.stop_listener, afm_litho.keeper.notes\n"
        "import afm_litho.__main__\n"
        "print('\\n'.join(sorted(sys.modules)))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.split()
    loaded = set(out)
    for banned in BANNED_TOP_LEVEL:
        assert banned not in loaded, f"importing the keeper loaded {banned}"
    afm = sorted(m for m in loaded if m == "afm_litho" or m.startswith("afm_litho."))
    for m in afm:
        assert m == "afm_litho" or m.startswith(ALLOWED_AFM), f"the keeper loaded {m}"
