"""The golden scenarios that take the instrument side (`inst`), listed (H5.pre).

`inst` sets what Igor would report or holds the stub at a chosen moment. It never goes through
a door, so a seam proof that replays only door traffic cannot reproduce these scenarios. Each
is named here with the instrument calls it makes. A scenario that gains or loses `inst`, or
changes which calls it makes, fails this test and must be classed again.

Disposition `excluded`: left out of any proof that drives only the doors, because the moment
the stub is held at (or the value Igor reports) cannot be stated through a door. None has a
door-driven twin yet.
"""

from __future__ import annotations

import inspect
import re

from golden.harness import scenario_args
from golden.scenarios import SCENARIOS

INST_SCENARIOS = {
    "litho_lateral_refused": {"lateral_refused"},
    "safe_park_com_down": {"com_down"},
    "safe_park_vendor_withdraw": {"park_leaves_tip_down", "set_value"},
    "scan_chained_ac": {"set_crosspoint"},
    "scan_abort_mid_line": {"hold_scan_after_line", "wait_held"},
    "engage_ac_tune_withdraw": {"set_crosspoint", "set_value"},
    "write_refused_in_ac": {"set_crosspoint"},
    "abort_in_transit": {"hold_next", "wait_held"},
    "abort_mid_stroke": {"hold_next", "wait_held"},
    "abort_at_chain_seam": {"hold_chain_after_chunk", "wait_held"},
    "cockpit_abort_mid_stroke": {"hold_next", "wait_held"},
    "cockpit_abort_in_transit": {"hold_next", "wait_held"},
    "litho_abort_http_mid_stroke": {"hold_next", "wait_held"},
    "transit_retry": {"land_short"},
    "transit_failure_withdraw": {"land_short"},
    "scan_stop_deferred_withdraw": {"hold_scan_after_line", "release_hold", "wait_held"},
    "interlock_trip": {"set_value"},
    "deadman_during_write": {"hold_next", "is_held", "wait_held"},
}


def _inst_calls(fn) -> set[str]:
    return set(re.findall(r"\binst\.(\w+)", inspect.getsource(fn)))


def test_the_scenarios_that_take_inst_are_the_listed_ones():
    actual = {n for n, fn in SCENARIOS.items() if "inst" in scenario_args(fn)}
    assert actual == set(INST_SCENARIOS), sorted(actual ^ set(INST_SCENARIOS))


def test_each_makes_the_listed_instrument_calls():
    for name, calls in INST_SCENARIOS.items():
        assert _inst_calls(SCENARIOS[name]) == calls, name


def test_the_door_only_scenarios_never_touch_inst():
    for name, fn in SCENARIOS.items():
        if name not in INST_SCENARIOS:
            assert "inst" not in _inst_calls(fn) and not _inst_calls(fn), name
