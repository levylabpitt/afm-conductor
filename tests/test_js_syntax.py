"""The front-end JavaScript must at least parse.

User request: "now there is no history of scans to load." The sessions list was
empty, and so was everything else, because app.js had a SYNTAX ERROR and the
browser never executed a line of it. A confirm() string had a literal newline
in the middle of it -- I had written the file with a Python replacement and
the escape did not survive -- so the whole panel was dead: no telemetry, no
log, no session history, nothing.

It cost an hour of looking in the wrong place, because a page with broken
JavaScript still renders perfectly. The markup, the stylesheet and the
service were all fine. Only the behaviour was missing, and "the list is
empty" reads as a data problem, not a parse error.

There are 383 Python tests here and there was nothing checking the language
the user interface is actually written in. node is present on this machine
and `node --check` costs milliseconds.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "src" / "afm_litho" / "web"
JS = sorted(WEB.glob("*.js"))


def test_there_are_javascript_files_to_check():
    """A control: if the glob ever comes back empty, the rest passes
    vacuously and proves nothing."""
    assert JS, f"no .js found in {WEB}"
    names = {p.name for p in JS}
    old = {"app.js", "ui-adapt.js", "render.js"}
    if old & names:        # the old main window's scripts are not in every tree
        assert old <= names


@pytest.mark.parametrize("path", JS, ids=lambda p: p.name)
def test_it_parses(path: Path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed; cannot check JavaScript syntax")
    r = subprocess.run([node, "--check", str(path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, (
        f"{path.name} does not parse, so the browser will execute NONE of it:\n"
        + (r.stderr or r.stdout))


@pytest.mark.skipif(not (WEB / "app.js").exists(), reason="app.js is not in this tree")
def test_no_literal_newline_inside_a_string(path=WEB / "app.js"):
    """The specific shape of the bug: an unterminated string literal. `node
    --check` catches it, and this says plainly what to look for when it
    does."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    r = subprocess.run([node, "--check", str(path)], capture_output=True, text=True)
    assert "Invalid or unexpected token" not in (r.stderr or ""), (
        "an unterminated string literal, most likely a newline inside quotes")
