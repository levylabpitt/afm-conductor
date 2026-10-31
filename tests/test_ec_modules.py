"""Run tests/ec_modules.mjs under node, as part of the suite.

The edit cockpit's Layer-1 / import-safe modules (ec-log's pure formatters,
ec-api's NaN-safe ws parse + onWs registry) are real behaviour with no DOM
dependency, imported directly under node -- the same pattern as
tests/pure_layer.mjs for the main panel. This wrapper makes pytest notice when
they break.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

JS = Path(__file__).resolve().parent / "ec_modules.mjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_ec_modules_behave():
    r = subprocess.run(["node", str(JS)], capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout or "") + (r.stderr or "")
