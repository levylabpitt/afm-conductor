"""Slice 3.4: the (run, wire) check.

`expected_wire(run)` is what a stored run says the wire carries; `wire_facts(lines)` reads the
same facts off a golden's recorded wire. For every write and scan golden that has a run, the
run the golden's session stored is projected and compared with the FROZEN golden file. The
mutation tests then change one rule in the builders (not in the service, so the wire does not
move with it) and show the check fails.

The Cypher goldens are all checked. Of the MFP-3D wire goldens only the scans are: the MFP-3D
write goldens are skipped (engage and withdraw mechanics stay in the driver, and they carry the
known `mfp3d-setdown` defect), a gap logged as F58 in docs/migration/SLICES.md.
"""

from __future__ import annotations

import json

import pytest
from golden.harness import HERE
from golden.recordings import golden_group, recorded
from golden.scenarios import SCENARIOS, WIRE, WIRE_B, WIRE_LONG
from test_build_write_run import build
from test_run_build_scan import SCAN_GOLDENS, _build
from wire_facts import golden_actuation, wire_facts

from afm_litho.hardware import lithography_run
from afm_litho.hardware.lithography_run import WriteSetpoint
from afm_litho.run import build_write as bw
from afm_litho.run.expected_wire import collapse_repeats, differences, expected_wire
from afm_litho.service import hw_scan

# The free-air deflection the stub reports (fixture DeflectionValue); the relative golden's
# engage setpoint on the wire is this plus the 0.3 V asked for.
FREE_AIR_V = json.loads((HERE / "fixture" / "igor_values.json").read_text())["values"][
    "DeflectionValue"]

# Goldens with a run. `prefix`: the wire stops at an abort or failure after exactly that many
# facts, a strictly shorter prefix of the run's (pinned, so a golden that aborts earlier
# fails instead of checking less). `tail`: facts the failure handler sends after the prefix. `free_air_v`: turns a relative engage setpoint into the absolute one on the wire.
VERIFIED_WITHDRAW = ("withdraw", -10.0, 0.0, 150.0)

CHECKED = {
    "litho_single": {},
    "litho_lateral_refused": {},
    "litho_chain_two_chunks": {},
    "write_relative_setpoint": {"free_air_v": FREE_AIR_V},
    "transit_retry": {},
    "multi_pass": {},
    "scan_chained_contact": {},
    "scan_chained_ac": {},
    "continuous_two_frames": {},
    "scan_abort_mid_line": {"prefix": 2},
    "scan_stop_deferred_withdraw": {"prefix": 2},
    "abort_in_transit": {"prefix": 7},
    "abort_mid_stroke": {"prefix": 4},
    "cockpit_abort_mid_stroke": {"prefix": 4},
    "litho_abort_http_mid_stroke": {"prefix": 4},
    "cockpit_abort_in_transit": {"prefix": 7},
    "abort_at_chain_seam": {"prefix": 4},
    "deadman_during_write": {"prefix": 4},
    "transit_failure_withdraw": {"prefix": 2, "tail": [VERIFIED_WITHDRAW]},
}

SKIPPED = {
    "litho_multi_object": ("no run: the filled pad compiles to several strokes of one object "
                           "and the run contract has one fire per object (UnsupportedWrite)"),
}

# Goldens in which no write or scan runs.
NO_PASS = {
    "safe_park", "safe_park_com_down", "safe_park_vendor_withdraw", "engage_contact_tune_withdraw",
    "engage_ac_tune_withdraw", "mode_switch", "show_tip", "connect_reads", "write_refused_in_ac",
    "interlock_trip", "deadman_engaged_withdraw", "engaged_idle_cap", "withdraw_flex",
    "withdraw_http",
}


def compare(name, runs, *, free_air_v=None, prefix=None, tail=(), mfp3d=False):
    """Problems between the runs' projected wire and the frozen golden's recorded wire. With
    `prefix`, the recorded facts (less `tail`) must be exactly that many, strictly fewer than
    the run's."""
    expected = collapse_repeats(
        [f for run in runs for f in expected_wire(run, free_air_v=free_air_v)])
    recorded = wire_facts(golden_actuation(name, mfp3d=mfp3d))
    if prefix is not None:
        keep = len(recorded) - len(tail)
        if recorded[keep:] != list(tail):
            return [f"after the abort the wire has {recorded[keep:]}, not {list(tail)}"]
        recorded = recorded[:keep]
        if len(recorded) != prefix or prefix >= len(expected):
            return [f"{len(recorded)} facts recorded, {prefix} pinned, {len(expected)} in the run"]
        expected = expected[:prefix]
    return differences(expected, recorded)


def test_every_golden_is_checked_skipped_or_has_no_pass():
    assert set(CHECKED) | set(SKIPPED) | NO_PASS == set(SCENARIOS)
    assert not set(CHECKED) & (set(SKIPPED) | NO_PASS)


@pytest.mark.parametrize("name,reason", SKIPPED.items())
def test_skipped_golden(name, reason):
    pytest.skip(reason)


@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


def stored_runs(hub, client, name, microscope_id=None):
    """The runs the scenario stored on the golden rig, in the order they began (one
    recording per scenario, shared with the golden tests)."""
    return recorded(hub, client, name, microscope_id).runs


@pytest.mark.parametrize("name", [pytest.param(n, marks=golden_group(n)) for n in CHECKED])
def test_the_stored_run_predicts_the_golden_wire(hub, client, name):
    runs = stored_runs(hub, client, name)
    assert runs, f"{name} stored no run"
    problems = compare(name, runs, **CHECKED[name])
    assert not problems, "\n".join(problems)


# The MFP-3D scan goldens: the wire-level goldens of the scenarios that only scan.
MFP_SCANS = ("scan_chained_contact", "scan_chained_ac", "continuous_two_frames",
             "scan_abort_mid_line", "scan_stop_deferred_withdraw")
MFP_WRITES_SKIPPED = ("engage and withdraw mechanics stay in the driver, and the MFP-3D write "
                      "goldens carry the known mfp3d-setdown defect (SLICES F58)")


@pytest.mark.parametrize("name", [pytest.param(n, marks=golden_group(n)) for n in MFP_SCANS])
def test_the_stored_scan_run_predicts_the_mfp3d_golden_wire(hub, client, name):
    from golden.harness import MFP3D_MICROSCOPE_ID
    runs = stored_runs(hub, client, name, MFP3D_MICROSCOPE_ID)
    assert runs and all(r.recipe.platform == "mfp3d" for r in runs)
    problems = compare(name, runs, mfp3d=True, **CHECKED[name])
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("name", [n for n in SCENARIOS if n not in MFP_SCANS
                                  and n not in NO_PASS])
def test_mfp3d_write_golden_is_skipped(name):
    pytest.skip(MFP_WRITES_SKIPPED)


# --- the check fails when a rule the run states changes ------------------------------------
# Each mutation changes the BUILDER only (a rule it calls, as a module attribute it reads), so
# the run says something the recorded wire does not.

def two_wires():
    return [build([WIRE, WIRE_B])]


def long_wire():
    return [build([WIRE_LONG])]


def relative_wire():
    return [build([WIRE], setpoint=WriteSetpoint(0.3, relative=True))]


def contact_scan():
    return [_build(SCAN_GOLDENS["scan_chained_contact"])]


INPUTS = (
    ("abort_in_transit", two_wires, {"prefix": 7}),
    ("litho_chain_two_chunks", long_wire, {}),
    ("write_relative_setpoint", relative_wire, {"free_air_v": FREE_AIR_V}),
    ("scan_chained_contact", contact_scan, {}),
)


@pytest.mark.parametrize("name,runs,opts", INPUTS, ids=[i[0] for i in INPUTS])
def test_runs_built_from_the_golden_inputs_agree(name, runs, opts):
    assert compare(name, runs(), **opts) == []


def test_a_changed_hover_transit_rule_fails_the_check(monkeypatch):
    monkeypatch.setattr(bw, "is_hover_move", lambda first, source, last: not first)
    problems = compare("abort_in_transit", two_wires(), prefix=7)
    assert any("hover" in p for p in problems), problems


def test_a_changed_chunk_cut_fails_the_check(monkeypatch):
    monkeypatch.setattr(lithography_run, "CHAIN_CHUNK_POINTS",
                        lithography_run.CHAIN_CHUNK_POINTS - 32)
    problems = compare("litho_chain_two_chunks", long_wire())
    assert any("litho" in p for p in problems), problems


def test_a_changed_scan_interpolation_fails_the_check(monkeypatch):
    monkeypatch.setattr(hw_scan, "scan_interpolation", lambda pixels, rate_hz: -30)
    problems = compare("scan_chained_contact", contact_scan())
    assert any("scan" in p for p in problems), problems


def test_a_changed_engage_setpoint_sign_fails_the_check(monkeypatch):
    real = bw.as_write_setpoint

    def negated(setpoint):
        ask = real(setpoint)
        return WriteSetpoint(-ask.value_v, ask.relative)

    monkeypatch.setattr(bw, "as_write_setpoint", negated)
    problems = compare("write_relative_setpoint", relative_wire(), free_air_v=FREE_AIR_V)
    assert any("engage" in p for p in problems), problems


def test_a_changed_back_off_setpoint_fails_the_check(monkeypatch):
    kw = {**bw.build_write_run.__kwdefaults__, "retract": -9.0}
    monkeypatch.setattr(bw.build_write_run, "__kwdefaults__", kw)
    problems = compare("abort_in_transit", two_wires(), prefix=7)
    assert any("back-off" in p for p in problems), problems


def test_a_changed_engage_gain_sign_fails_the_check(monkeypatch):
    """The sign of the write's loop-2 gain as the builder sees it, flipped: an inverted Z
    feedback. Every write golden engages, so any of them shows it."""
    monkeypatch.setattr(bw, "Z_IGAIN", -lithography_run.Z_IGAIN)
    problems = compare("litho_single", [build([WIRE])])
    assert any("engage" in p for p in problems), problems
