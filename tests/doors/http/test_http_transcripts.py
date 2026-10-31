"""The HTTP door, recorded (slice 2.0b).

One test per transcript group. Each starts its own twin service, replays the
scripted requests in `http_door_cases`, and compares the masked result with the
canonical JSON under `transcripts/`. A route that changes its status code, its
body or the WS events it causes fails here. Regenerate only on purpose:

    AFM_LITHO_REGEN_HTTP_TRANSCRIPTS=1 pytest tests/doors/http/test_http_transcripts.py
"""

from __future__ import annotations

import pytest
from http_door_cases import GROUPS
from http_door_recorder import Group, check_or_regen, run_group

pytest.importorskip("zmq", reason="the control group sets up a script holder over FLEX")


@pytest.mark.parametrize("name", sorted(GROUPS))
def test_http_door_transcript(name):
    flex, steps = GROUPS[name]
    entries = run_group(Group(name, flex=flex), steps)
    assert entries, f"group {name} recorded nothing"
    check_or_regen(name, entries)
