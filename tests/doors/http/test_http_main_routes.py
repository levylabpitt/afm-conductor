"""The MAIN-only HTTP routes, recorded (H5.pre).

The 26 routes the doors work left without a transcript (`MAIN_ONLY` in
tests/doors/test_inventory.py). Same recorder, own directory, so the existing transcripts
are not touched. Regenerate only on purpose:

    AFM_LITHO_REGEN_HTTP_TRANSCRIPTS=1 pytest tests/doors/http/test_http_main_routes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import http_door_recorder as rec
from http_door_recorder import Group, check_or_regen, run_group
from main_route_cases import main_routes

MAIN_TRANSCRIPTS = Path(__file__).parent / "transcripts_main"


def test_main_routes_transcript(monkeypatch):
    monkeypatch.setattr(rec, "TRANSCRIPT_DIR", MAIN_TRANSCRIPTS)
    entries = run_group(Group("main_routes"), main_routes)
    assert entries
    check_or_regen("main_routes", entries)


def test_the_transcript_covers_every_main_route_that_had_none():
    sys.path.insert(0, str(Path(__file__).parents[1]))
    import test_inventory as inv

    routes = inv._actual_http_routes()
    other = {r for _g, (m, p) in inv._recorded_steps() if (r := inv._route_of(m, p, routes))}
    doc = json.loads((MAIN_TRANSCRIPTS / "main_routes.json").read_text(encoding="utf-8"))
    mine = {r for e in doc["entries"] if (req := e.get("request"))
            if (r := inv._route_of(req["method"], req["path"], routes))}
    missing = inv.MAIN_ONLY - other - mine
    assert not missing, f"MAIN routes with no transcript anywhere: {sorted(missing)}"
