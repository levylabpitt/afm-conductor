"""Regenerate python_fixtures.json: reference outputs of the Python originals that
the cockpit JS duplicates. Run from the repo root with the repo venv:

    .venv/Scripts/python.exe tests/js/fixtures/generate_fixtures.py

The JS tests (tests/js/crosscheck.test.js) assert the JS modules reproduce these.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from afm_litho.litho.bezier import BezierPath, Node
from afm_litho.service import design_store


def bez_dict(nodes, closed):
    return {"closed": closed, "nodes": nodes}


def flatten_case(name, nodes, closed, tol):
    path = BezierPath([Node.from_dict(n) for n in nodes], closed)
    pts = path.flatten(tol)
    return {"name": name, "bez": bez_dict(nodes, closed), "tol": tol,
            "points": [[float(x), float(y)] for x, y in pts]}


def n(x, y, h_in=None, h_out=None, kind="cusp"):
    return {"x": x, "y": y, "h_in": h_in, "h_out": h_out, "kind": kind}


def build_fixtures():
    """Return the fixture dict; writes nothing."""
    flatten = [
        flatten_case("straight_open", [n(0, 0), n(3, 4), n(6, 0)], False, 0.004),
        flatten_case("s_curve", [n(0, 0, None, [2, 3], "smooth"),
                                 n(10, 0, [8, -3], None, "smooth")], False, 0.004),
        flatten_case("closed_blob", [n(0, 0, [-2, 1], [2, -1], "smooth"),
                                     n(5, 5, [4, 3], [6, 7], "symmetric"),
                                     n(10, 0, [11, 3], [9, -3], "smooth")], True, 0.005),
        flatten_case("coarse_tol", [n(0, 0, None, [0, 8]), n(8, 0, [8, 8], None)], False, 0.5),
    ]

    handle = []
    for kind in ("cusp", "smooth", "symmetric"):
        for which, tgt in (("out", (3.0, 4.0)), ("in", (-2.0, 1.5))):
            node = Node(1.0, 2.0, (0.5, 2.5), (3.0, 2.0), kind)
            node.set_handle(which, *tgt)
            handle.append({"kind": kind, "which": which, "target": list(tgt),
                           "start": {"x": 1.0, "y": 2.0, "h_in": [0.5, 2.5],
                                     "h_out": [3.0, 2.0], "kind": kind},
                           "result": node.to_dict()})

    move = Node(1.0, 2.0, (0.5, 2.5), (3.0, 2.0), "smooth")
    move.move_to(4.0, -1.0)
    move_case = {"start": {"x": 1.0, "y": 2.0, "h_in": [0.5, 2.5], "h_out": [3.0, 2.0],
                           "kind": "smooth"}, "to": [4.0, -1.0], "result": move.to_dict()}

    names = ["dev1", "dev1.design.json", "  spaced name ", "a/b\\c", "..", "dev.1", "dev.2",
             "ünï-cødé", "x:y*z?", "", "   ", "café", "n²", "tab\tname", "A.DESIGN.JSON",
             "é", "日本語", "a b  c"]
    file_names = [{"raw": r, "name": design_store.file_name(r)} for r in names]

    return {"suffix": design_store.SUFFIX, "flatten": flatten, "set_handle": handle,
            "move_to": move_case, "file_name": file_names}


def write_fixtures():
    dest = Path(__file__).with_name("python_fixtures.json")
    text = json.dumps(build_fixtures(), indent=1, ensure_ascii=False) + chr(10)
    dest.write_text(text, encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    write_fixtures()
