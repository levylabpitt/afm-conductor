"""The safe-state matrix (slice 0.6): every way the program is told to make the instrument safe,
driven through a door (FLEX or HTTP) on the fake-Igor safety simulation, judged only from what the program can
read.

Rows are in `safe_state_rows.py`, the rig and the judge in `safe_state_harness.py`. Columns:

  tip_off     the Z drive is off the surface (Height and ZSensor below the surface level)
  xy_at_rest  XSensor and YSensor are unchanged across a wait
  bias_zero   Output.A reads zero
  report      what the program said (the reply or state flag) matches the instrument: it says
              safe when safe and not-safe when not

A cell either passes or is a strict xfail naming the defect it records (`KNOWN`). These tests
record today's behaviour, so the migration cannot make it worse; nothing here is a fix. When a
defect is fixed its cell starts passing, the strict xfail fails, and the entry is deleted.

Run with `-n 4 --dist loadgroup` to keep each row on one worker (a row takes several seconds
of the program's own waits; without the grouping every cell would redo its row).
"""

from __future__ import annotations

import pytest
from safe_state_harness import BIAS_TOL_V, XY_LOOK_S, SafeRig, engage, wait_for
from safe_state_rows import (
    ROWS, SLOW_SCAN, WIRE_START_V, WRITE_DONE_WAIT_S, Outcome, run_row, start_wire)

CYPHER, MFP3D = 4, 0
CELLS = ("tip_off", "xy_at_rest", "bias_zero", "report")
PLATFORM = {CYPHER: "cypher", MFP3D: "mfp3d"}

STOP_REPLIES_EARLY = ("scan/abort answers ok before the deferred retract finishes "
                      "(host/scan_cmds.py stop_scan defers to _hw_scan_pump). With the tip "
                      "engaged at ~70 V the Cypher reads 70 V at the reply and 0 V once the "
                      "retract has run. Unresolved: why the MFP-3D row does not show it.")
STOP_XY_STILL_MOVING = ("scan/abort answers ok while the scan's XY out-wave is still playing; "
                        "XY reads moving at the reply and at rest once the deferred retract "
                        "has run (F38).")
AUDIT_A_2 = ("audit A section 2: HTTP withdraw clears 'engaged' and answers ok without "
             "reading back that the tip came off (the FLEX withdraw does verify)")

# (row, platform, cell) -> the defect it records. A cell listed here is a strict xfail.
KNOWN: dict[tuple[str, str, str], str] = {
    ("stop_scan", "cypher", "tip_off"): STOP_REPLIES_EARLY,
    ("stop_scan", "cypher", "xy_at_rest"): STOP_XY_STILL_MOVING,
    ("withdraw_http_stuck_z", "cypher", "report"): AUDIT_A_2,
    ("withdraw_http_stuck_z", "mfp3d", "report"): AUDIT_A_2,
}


@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


_OUTCOMES: dict[tuple[str, int, bool], Outcome] = {}


def outcome(hub, client, row, microscope_id: int, ramp_hypothesis: bool = False) -> Outcome:
    """One run of a row per platform and Cypher park hypothesis, shared by its cells."""
    key = (row.name, microscope_id, ramp_hypothesis)
    if key not in _OUTCOMES:
        _OUTCOMES[key] = run_row(hub, client, row, microscope_id, ramp_hypothesis)
    return _OUTCOMES[key]


def cell_holds(out: Outcome, cell: str) -> bool:
    """A physical cell holds when the instrument read right at the reply and after it settled;
    the report cell holds when what the program said matches the instrument as it read at the
    reply, the moment the claim was made."""
    if cell == "report":
        return out.said.safe == out.at_reply.all
    return getattr(out.at_reply, cell) and getattr(out.settled, cell)


def _params():
    for row in ROWS:
        platforms = [CYPHER] if row.cypher_only or not row.mfp_too else [CYPHER, MFP3D]
        for mid in platforms:
            cells = ("report",) if row.fault_injection else CELLS
            for cell in cells:
                name = PLATFORM[mid]
                marks = [pytest.mark.xdist_group(f"{row.name}-{name}")]
                why = KNOWN.get((row.name, name, cell))
                if why:
                    marks.append(pytest.mark.xfail(reason=why, strict=True))
                yield pytest.param(row, mid, cell, marks=marks,
                                   id=f"{row.name}-{name}-{cell}")


@pytest.mark.parametrize("row,microscope_id,cell", _params())
def test_cell(hub, client, row, microscope_id, cell):
    out = outcome(hub, client, row, microscope_id)
    assert cell_holds(out, cell), (
        f"{row.name} on the {PLATFORM[microscope_id]}: the program said "
        f"safe={out.said.safe} ({out.said.text}); the instrument read, at the reply: "
        f"{out.at_reply}; settled: {out.settled}")


# ---- the Cypher park under both hypotheses about a Height ramp after STOPLOOP ----

PARK_HYPOTHESIS_A = ("Cypher park hypothesis A (ramp inert): decided by the rig on 2026-10-27: a "
                     "Height ramp with loop 2 stopped does not move Z, only the -10 V re-arm "
                     "retracts; SLICES F5, F77.")
PARK_ROWS = ("abort_scan_line", "stop_scan", "safe_park_button", "interlock_trip",
             "deadman_lapse")
# Rows whose tip stays up when the ramp is inert. None since slice AB: the park is Igor's
# one abort function (the -10 V re-arm), with no host ramp to depend on.
A_FAILS: set[str] = set()


def _hypothesis_params():
    for row in ROWS:
        if row.name not in PARK_ROWS:
            continue
        for hyp in (True, False):
            marks = [pytest.mark.xdist_group(f"{row.name}-cypher")]
            if not hyp and row.name in A_FAILS:
                marks.append(pytest.mark.xfail(reason=PARK_HYPOTHESIS_A, strict=True))
            elif row.name == "stop_scan":
                marks.append(pytest.mark.xfail(reason=STOP_REPLIES_EARLY, strict=True))
            yield pytest.param(row, hyp, marks=marks,
                               id=f"{row.name}-ramp_{'effective' if hyp else 'inert'}")


@pytest.mark.parametrize("row,hyp", _hypothesis_params())
def test_cypher_tip_off_under_both_park_hypotheses(hub, client, row, hyp):
    out = outcome(hub, client, row, CYPHER, hyp)
    assert cell_holds(out, "tip_off"), f"{row.name}: {out.at_reply}; settled: {out.settled}"


# ---- an abort from a safe state is harmless ----

SAFE_STATE_DOORS = (
    ("flex", "abort", {}),
    ("flex", "safePark", {}),
    ("flex", "withdraw", {}),
    ("flex", "abortWrite", {}),
    ("http", "/api/safe_park", {}),
    ("http", "/api/withdraw", {}),
    ("http", "/api/scan/abort", {}),
    ("http", "/api/litho/abort", {}),
)


@pytest.mark.xdist_group("already-safe")
@pytest.mark.parametrize("microscope_id", [CYPHER, MFP3D], ids=["cypher", "mfp3d"])
def test_every_stop_from_an_already_safe_state_is_harmless(hub, client, microscope_id):
    """Nothing is engaged: tip up, XY still, bias zero. Each stop door is issued in turn; none
    raises, none latches a fault, and the instrument reads safe after every one."""
    with SafeRig(hub, "already-safe", microscope_id=microscope_id, client=client) as rig:
        assert rig.judge().all, "the rig did not start safe"
        for kind, door, body in SAFE_STATE_DOORS:
            if kind == "flex":
                rig.door.call(door, body)                 # raises on any error reply
            else:
                reply = rig.client.post(door, json=body)
                assert reply.status_code < 400, f"{door}: HTTP {reply.status_code}"
            rig.wait_quiet()
            after = rig.judge()
            assert after.all, f"{door} from a safe state left it unsafe: {after}"
            assert hub.fault is None, f"{door} from a safe state latched {hub.fault}"


# ---- F9: a bias wave left bound after a write ----

F9_SAMPLES = 8                   # reads of Output.A, XY_LOOK_S apart, during the scan

F9 = ("Stub model (the fake Igor's stale-play rule; vendor persistence is unconfirmed), F9: a litho write binds the bias wave (out-bank 2) to Event 14 and the chained scan "
      "releases only out-bank 0, so the scan's line fire replays the bias on Output.A while the "
      "tip scans. Modelled by the stub's stale-play rule (0.5 bank model); vendor persistence "
      "of the binding is not confirmed")


@pytest.mark.xdist_group("f9")
@pytest.mark.xfail(reason=F9, strict=True)
def test_a_scan_after_a_write_puts_no_voltage_on_the_tip(hub, client):
    with SafeRig(hub, "f9", client=client, hold_scan=True, tip_at_volts=WIRE_START_V) as rig:
        tok = engage(rig)
        start_wire(rig, tok)
        assert wait_for(lambda: rig.door.call("getState") != "writing", WRITE_DONE_WAIT_S), "the write hung"
        rig.wait_quiet()
        assert not rig.bias_is_on(), "the write left a voltage on the tip"
        rig.door.call("startScan", {"token": tok, **SLOW_SCAN})
        assert rig.wait_for_actuation("SCANCHAIN"), "the scan never started"
        biases = []
        for _ in range(F9_SAMPLES):
            biases.append(abs(rig.read("Output.A")))
            rig.stub.advance(XY_LOOK_S)
        assert max(biases) <= BIAS_TOL_V, f"the scan put {max(biases):.2f} V on the tip: {biases}"
