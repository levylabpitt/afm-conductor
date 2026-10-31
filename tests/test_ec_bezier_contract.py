"""JS<->Python bezier contract: ec-bezier.js `bezFlatten` must match litho/bezier.py.

The node tool previews a curve by flattening its control points in the browser
(ec-bezier.js), while the instrument writes the polyline the backend flattens
from the SAME control points (litho/bezier.py). If the two subdivisions diverge,
the operator sees one path and the tip writes another. This pins them together:
Python flattens a set of BezierPaths, the node runner flattens the same dicts in
JS, and the polylines must agree point-for-point within 1e-6 µm.

The two implementations are the same adaptive de Casteljau subdivision (identical
flatness proxy, tolerance, and depth cap), so with the same tolerance the output
sequences are expected to match exactly bar floating-point last bits.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from afm_litho.litho.bezier import BezierPath, Node

RUNNER = Path(__file__).resolve().parent / "ec_bezier_flatten_runner.mjs"


def _case(name: str, path: BezierPath, tol: float) -> dict:
    return {"name": name, "bez": path.to_dict(), "tol": tol,
            "expected": path.flatten(tol).tolist()}


def _cases() -> list[dict]:
    return [
        # a straight open segment -> just its two endpoints
        _case("line", BezierPath([Node(0, 0), Node(10, 0)]), 0.004),
        # a single smooth arc pulled up by two handles
        _case("arc", BezierPath([Node(0.0, 0.0, h_out=(0.0, 4.0)),
                                 Node(8.0, 0.0, h_in=(8.0, 4.0))]), 0.01),
        # a two-segment S: down-curve then up-curve
        _case("s-curve", BezierPath([
            Node(0.0, 0.0, h_out=(2.0, 3.0)),
            Node(6.0, 0.0, h_in=(4.0, -3.0), h_out=(8.0, 3.0)),
            Node(12.0, 0.0, h_in=(10.0, -3.0)),
        ]), 0.004),
        # a tight arc that forces deep subdivision at a fine tolerance
        _case("tight", BezierPath([Node(0.0, 0.0, h_out=(0.0, 20.0)),
                                   Node(1.0, 0.0, h_in=(1.0, 20.0))]), 0.001),
    ]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_bezflatten_matches_python():
    fixtures = {"cases": _cases()}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        json.dump(fixtures, f)
        fn = f.name
    try:
        r = subprocess.run(["node", str(RUNNER), fn], capture_output=True, text=True)
        assert r.returncode == 0, (r.stdout or "") + (r.stderr or "")
    finally:
        os.unlink(fn)
