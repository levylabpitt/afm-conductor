"""The committed JS cross-check fixtures must match what the Python originals produce now.

Covers bezier flatten / set_handle / move_to and design-file naming.
"""
import importlib.util
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "js" / "fixtures"
CHECKED = ["flatten", "set_handle", "move_to", "suffix", "file_name"]
REGEN = ".venv/Scripts/python.exe tests/js/fixtures/generate_fixtures.py"
FLOAT_TOL = 1e-9  # same code, same machine class: any real change is far above this


def _load_generator():
    path = FIXTURES / "generate_fixtures.py"
    spec = importlib.util.spec_from_file_location("generate_fixtures", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _diff(a, b, path=""):
    """First difference between two JSON trees, or None. Strings exact, floats within FLOAT_TOL."""
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return None if a is b else f"{path}: {a!r} != {b!r}"
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return None if a == pytest.approx(b, abs=FLOAT_TOL, rel=0) else f"{path}: {a!r} != {b!r}"
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return f"{path}: keys {sorted(a)} != {sorted(b)}"
        for k in a:
            d = _diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if type(a) is type(b) and a == b else f"{path}: {a!r} != {b!r}"


@pytest.mark.parametrize("section", CHECKED)
def test_js_fixtures_match_python(section):
    fresh = json.loads(json.dumps(_load_generator().build_fixtures(), ensure_ascii=False))
    committed = json.loads((FIXTURES / "python_fixtures.json").read_text(encoding="utf-8"))
    d = _diff(fresh[section], committed[section], section)
    assert d is None, (
        f"python_fixtures.json is stale ({d}). Regenerate from the repo root with:\n"
        f"    {REGEN}\nthen re-run `npx vitest run` and commit the result."
    )
