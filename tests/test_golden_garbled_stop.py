"""An abort with garbled params leaves the same wire and Igor trace as a clean abort (2.7c).

Replays the frozen `abort_mid_stroke` scenario with its `abortWrite` sent a list, a number
and a string as params, and judges each against the frozen golden (read only).
"""

from __future__ import annotations

import pytest
from golden import scenarios
from golden.harness import HERE, record

from afm_litho.goldens import golden_format as G


@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


@pytest.mark.parametrize("garbled", [[1], 5, "s"], ids=repr)
def test_a_garbled_abort_write_has_the_clean_abort_trace(hub, client, monkeypatch, garbled):
    def abort_mid_stroke(door, inst) -> None:
        tok = scenarios._acquire(door)
        inst.hold_next("ARMFIRE", 1)
        scenarios._start_write(door, tok, [scenarios.WIRE])
        inst.wait_held()
        door.call("abortWrite", garbled)
        door.wait_state("idle", "engaged", "parked")
        scenarios._release(door, tok)

    monkeypatch.setitem(scenarios.SCENARIOS, "abort_mid_stroke", abort_mid_stroke)
    got = record(hub, "abort_mid_stroke", client=client)
    frozen = G.load(HERE / "abort_mid_stroke.golden.json")
    problems = G.compare(frozen, got)
    assert not problems, "\n".join(problems)
