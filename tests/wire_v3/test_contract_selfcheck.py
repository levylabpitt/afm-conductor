"""The suite's own pins: codec, the frozen goldens it derives from, and its scenario coverage.

Always runs (no v3 target needed). If these fail, the conformance tests are not judging what
the contract says.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from golden.scenarios import OPTIONS, SCENARIOS

from afm_litho.goldens import golden_format as G

from . import contract as C

GOLDENS = Path(__file__).resolve().parents[1] / "golden"


def _frozen(name: str) -> dict:
    return G.load(GOLDENS / f"{name}.golden.json")


def test_parse_up_reply_report_cap_tick():
    r = C.parse_up("REPLY 7 ok w 8192/9000")
    assert (r.seq, r.ok, r.body) == (7, True, "w 8192/9000")
    e = C.parse_up("REPLY 8 err busy")
    assert e.ok is False and e.body == "busy"
    p = C.parse_up("REPORT 12 600 1 0 com r1 madesafe stop")
    assert (p.useq, p.tick, p.mark, p.fault, p.route, p.run, p.event, p.body) == \
        (12, 600, 1, 0, "com", "r1", "madesafe", "stop")
    c = C.parse_up("CAP 13 601 0 r1 2 ZSensor 1;2;3")
    assert (c.step, c.alias, c.body) == ("2", "ZSensor", "1;2;3")
    t = C.parse_up("TICK 14 602 0 0 - 0.1 0.2 0.3 0.4 5 0.6")
    assert t.route == "-" and len(t.body.split()) == 6


@pytest.mark.parametrize("bad", ["", "VALUE 1 2", "REPORT x", "REPORT 1 2"])
def test_parse_up_rejects_garbage(bad):
    with pytest.raises(ValueError):
        C.parse_up(bad)


def test_number_format_is_12_significant_digits():
    assert C.num(0.1 + 0.2) == "0.3" and C.num(1 / 3) == "0.333333333333"
    assert C.wave_body([1, 2.5, float("nan")]) == "1;2.5;nan"


def test_every_golden_scenario_is_in_the_catalogue_this_suite_walks():
    assert set(SCENARIOS) == {p.name.removesuffix(".golden.json")
                              for p in GOLDENS.glob("*.golden.json")
                              if ".mfp3d." not in p.name}
    assert len(SCENARIOS) >= 29, "the catalogue shrank; did a scenario get dropped?"


def test_contract_igor_changes_only_the_retired_park_lines():
    changed = []
    for name in SCENARIOS:
        frozen = _frozen(name)["igor"]["actuation"]
        new = C.contract_igor(frozen)
        dropped = [ln for ln in frozen if ln not in new or frozen.count(ln) != new.count(ln)]
        if new != frozen:
            changed.append(name)
            assert all(C._RETIRED_PARK[0].match(ln) or C._RETIRED_PARK[1].match(ln)
                       for ln in dropped), (name, dropped)
    # Slice AB re-froze the park goldens, so today nothing changes; a golden that still holds the
    # retired lines would be listed here and only those lines may go.
    assert changed == [] or all(_frozen(n)["known_defects"] for n in changed)


def test_contract_igor_is_idempotent_and_leaves_no_stoploop_or_ramp():
    for name in SCENARIOS:
        once = C.contract_igor(_frozen(name)["igor"]["actuation"])
        assert C.contract_igor(once) == once
        assert not C.retired_park_lines(once)


def test_every_park_golden_ends_with_the_contract_retract():
    parks = [n for n in SCENARIOS
             if any(x.startswith(("@SAFEPARK", "ABORT")) for x in _frozen(n)["wire"]["actuation"])]
    assert parks, "no park goldens found"
    for name in parks:
        got = C.contract_igor(_frozen(name)["igor"]["actuation"])
        assert C.RETRACT in got, f"{name}: no loop-2 retract left after the contract transform"
        assert C.BIAS_ZERO in got


def test_scenarios_that_need_real_time_are_known():
    # The equivalence suite runs them on the same clock options as the v2 goldens.
    for name, opts in OPTIONS.items():
        assert name in SCENARIOS, name
        json.dumps(opts)


def test_no_frozen_golden_still_holds_the_retired_park_lines():
    """Slice AB re-froze the park goldens: the STOPLOOP and Height ramp are gone from every
    one, so `contract_igor` has nothing to drop. If this fails, a golden was re-frozen with
    the retired lines (defect 1) and the transform is hiding it."""
    left = {n: C.retired_park_lines(_frozen(n)["igor"]["actuation"]) for n in SCENARIOS}
    assert {n: v for n, v in left.items() if v} == {}
