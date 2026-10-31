"""RED tests for `service/command_guard.check_sum_ceiling` and
`service/safety/write_interlocks.WriteInterlocks` (wave-1 slice T5,
docs/INSTRUMENT_MODEL_SPEC.md section 12 + 12.1's interface contract).

Neither `check_sum_ceiling` nor `service/safety/write_interlocks.py` exist yet,
so every test in this file is expected to fail today -- most at COLLECTION
(`ModuleNotFoundError: afm_litho.service.safety.write_interlocks`) and the
`check_sum_ceiling` cases at an `ImportError`/`AttributeError` on
`afm_litho.service.command_guard` (which itself does not exist as a module yet
either, per tests/test_command_guard.py's own module docstring). `pytest
--collect-only -q` must still collect every other test file uneventfully.

Contract under test (12.1):
    def check_sum_ceiling(model, setpoint_abs_v: float, sum_v: float | None) -> None
    # sum_v None or < model.sum_floor_v (0.5) -> GuardRefused("sum_missing");
    # abs(setpoint_abs_v) > model.sum_ceiling_frac (0.9) * sum_v ->
    # GuardRefused("setpoint_exceeds_sum")

    @dataclass
    class InterlockConfig: rail_v=140.0; rail_s=5.0; sum_floor_v=0.5;
                           sum_frac=0.5; sum_s=5.0; defl_s=5.0
    class WriteInterlocks:
        def __init__(self, model, config: InterlockConfig, *, on_trip)
        def arm(self, sum_baseline_v: float) -> None
        def disarm(self) -> None
        def feed(self, *, t, z_out_v, sum_v, defl_v) -> None
        # on_trip(reason) fires ONCE, reason in {"rail_dwell", "sum_collapse",
        # "deflection_integrity"}, after 5s CONTINUOUS; self-disarms.

Model fields added (12.1): sum_floor_v (0.5), sum_ceiling_frac (0.9), z_rail_v
(140.0), interlock_s (5.0). Twin fault hooks: twin.faults.free_air_jitter,
z_rail_stuck (z_out reads 148V regardless of ramps), sum_drop (SUM reads 0.1V),
defl_over_sum.

ASSUMPTIONS beyond 12.1 (flagged for the implementer to confirm or correct):

  A1. `check_sum_ceiling`'s `model` argument only needs `.sum_floor_v` and
      `.sum_ceiling_frac` for this pure function -- it is exercised here against
      a minimal frozen stand-in (`_CeilingModel`), NOT the full section-10
      `InstrumentModel`/`tests/_im_fixtures_guard.make_model` (which does not
      even declare these two new fields yet). The door-level integration tests
      further down DO go through the real `hub.instrument_model` (however wave 2
      shapes it), since that is the only way to prove the guard is actually
      wired into `/api/approach` and `/api/litho/write`.
  A2. Both ceiling comparisons are INCLUSIVE at the boundary and EXCLUSIVE just
      past it, matching the brief's own worked example verbatim ("0.45 with SUM
      0.5 passes; 0.46 refused" - > i.e. `abs(setpoint) > ceiling` refuses,
      `==` passes) and the floor ("SUM 0.49 -> sum_missing" implies the floor
      check is `sum_v < model.sum_floor_v`, so `sum_v == 0.5` itself passes).
  A3. `check_sum_ceiling` checks `sum_missing` BEFORE `setpoint_exceeds_sum`.
      12.1 does not spell out an order between the two, but a None/too-low SUM
      makes `0.9 * sum_v` meaningless (undefined for `None`, and an
      already-untrustworthy bound for a below-floor SUM), so refusing on the
      SUM's own validity first is the only order under which the second
      check's arithmetic is ever well-formed. (NIT fix: the earlier draft cited
      `test_command_guard.py`'s fault-latched-first convention here, which is a
      different file's precedence rule for different checks and does not
      actually establish this one -- removed.)
  A4. `WriteInterlocks`'s `model` argument only needs `.z_rail_v` for these
      tests (the ONLY model-sourced threshold 12.1 names -- "rail uses
      model.z_rail_v"); a minimal frozen stand-in (`_RailModel`) is used rather
      than the full `InstrumentModel`. SUM and deflection thresholds are read
      from `InterlockConfig` (`sum_floor_v`, `sum_frac`), which 12.1's own
      dataclass literal already carries -- flagged as a discrepancy: the brief
      says "SUM collapse via floor 0.5" without naming which of
      model.sum_floor_v / config.sum_floor_v feeds the RUNNING interlock (as
      opposed to `check_sum_ceiling`'s pre-engage floor, which is unambiguously
      `model.sum_floor_v`). Since 12.1's own `InterlockConfig` literal already
      carries `sum_floor_v: float = 0.5`, config is used for the running
      interlock is this file's working assumption.
  A5. "Continuously" is implemented as: the FIRST sample at which a condition
      becomes true starts a clock; every subsequent `feed()` while the
      condition remains true is compared against that start time; a single
      clean (non-triggering) sample resets the clock to None. The 5s boundary
      is INCLUSIVE (`elapsed >= condition_s` trips), matching this same
      section's other "at the line trips" convention.
  A6. `on_trip`'s self-disarm means `WriteInterlocks.armed` becomes False the
      moment a trip fires, and every subsequent `feed()` call is then a no-op
      (never raises, never calls `on_trip` again) -- consistent with "the
      interlock self-disarms" read literally.
  A7. `feed()` called before `arm()` is a silent no-op (never trips, never
      raises) -- `armed` defaults False until `arm()` is called, mirroring
      `RailDwellGuard.add_sample`'s existing "if not self.armed: return"
      pattern in `service/safety/rail_dwell.py`.
  A8. PINNED (coordinator, instrument-model b30c9fd, end of 12.1): the twin
      fault hooks (`z_rail_stuck`, `sum_drop`, `defl_over_sum`) live on
      `SoftwareARC.faults`, reachable as `hub.hardware_arc.faults` on the twin
      backend -- NOT `hub.arc.faults` (this file's earlier guess, corrected;
      see test_free_air_read.py's A7 for the same correction).
  A9. PINNED (same coordinator note): every `safe_park` stores
      `hub.last_park: ParkResult` with a `.reason` field, and a tripped
      interlock whose park VERIFIES does NOT latch `hub.fault` -- confirming
      (rather than merely guessing, as this file's earlier draft did) that
      `hub.fault` is the wrong observable here. A tripped write's safe_park is
      asserted two ways: `hub.last_park.reason == expected_reason` and
      `hub.last_park.verified is True` (the twin's withdraw verifies by
      construction -- `host/safe_park.py`'s `_safe_park_twin` docstring),
      plus the SAME `"safety"` kind session record (matched by its `label`,
      `f"safe_park: {reason}"`, since `SessionStore.list()` only surfaces
      `id/ts/kind/label`, not the full payload) that
      `tests/test_safety_rail_dwell.py`'s twin integration tests already assert
      on for the scan-path rail guard.
  A10. The write record's persisted `abort_reason` key (new; not in
      `tests/test_write_records.py`'s `_RECORD_KEYS`) is read from the same
      litho JSON sidecar that file already establishes
      (`<tmp_path>/writes/json/<stem>.json`, looked up via
      `afm_litho.service.stems.lookup`), alongside the existing `aborted` key.
  A11. One writable object 30 um long at 2 um/s (a 15s stroke) gives enough
      margin for a 5s-continuous interlock to trip mid-stroke (around the 5-6s
      mark, per the brief) while the write is still comfortably running, without
      making the whole test suite unreasonably slow.
  A12. FLEX's `startApproach` refuses SYNCHRONOUSLY (an immediate JSON-RPC
      error, never a `run_id` to poll), mirroring `startWrite`'s already-
      proven synchronous refusal pattern -- see the door-level section below
      for the full reasoning; this assumption is scoped to the `check_sum_
      ceiling` wiring only, not to every possible `startApproach` refusal.

ADVERSARIAL REVIEW FIXES APPLIED (this revision): M1 (agree_within lives in
test_free_air_read.py, fixed there) -- N/A here; M2 isolated the deflection-
integrity cases from sum_collapse contamination (baseline 1.0/sum_v 0.6, not
baseline 5.0/sum_v 0.5); M3 added continuous-time discrimination (4.9/clean/
4.9, sub-5s many-sample, exact-5s, on_trip-once, self-disarm) for sum_collapse
and deflection_integrity, mirroring rail_dwell's existing coverage; M4 added
door-level `check_sum_ceiling` wiring proof on REST (approach + write) and
FLEX (startApproach + startWrite), plus a fresh-read proof; S1 added an
elapsed >= 4.0s lower bound and a transient (~2s, cleared) non-trip case; S2
added twin approach/scan tests proving the running interlocks are armed
outside the write path too; S3 isolated the absolute SUM floor from the
baseline-fraction condition; S4 pinned `|defl| == sum` as NOT tripping; A3's
citation was corrected; the twin fixtures (`_tmp_store`) now reset `hub.
free_air`/`hub.last_park`/`hub.fault`/`hub.write_interlocks`/`hub.engaged`
per test. S5 (near-zero free-air mean) is out of scope for THIS file (it is
`agree_within`-specific) -- its stub lives in test_free_air_read.py.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass

import pytest
from afm_litho.service.command_guard import GuardRefused, check_sum_ceiling
from afm_litho.service.safety.write_interlocks import InterlockConfig, WriteInterlocks

# ===================================================== check_sum_ceiling ==

@dataclass(frozen=True)
class _CeilingModel:
    sum_floor_v: float = 0.5
    sum_ceiling_frac: float = 0.9


def test_setpoint_at_the_ceiling_passes():
    # 0.9 * 0.5 = 0.45 exactly -> passes (A2: inclusive at the boundary).
    assert check_sum_ceiling(_CeilingModel(), 0.45, 0.5) is None


def test_setpoint_just_past_the_ceiling_is_refused():
    with pytest.raises(GuardRefused) as ei:
        check_sum_ceiling(_CeilingModel(), 0.46, 0.5)
    assert ei.value.reason == "setpoint_exceeds_sum"


def test_sum_none_is_sum_missing():
    with pytest.raises(GuardRefused) as ei:
        check_sum_ceiling(_CeilingModel(), 0.1, None)
    assert ei.value.reason == "sum_missing"


def test_sum_below_floor_is_sum_missing_even_with_a_tiny_setpoint():
    # 0.49 < 0.5 floor -> sum_missing, even though 0.01 would pass any ceiling
    # math against 0.49 (0.9 * 0.49 = 0.441 > 0.01).
    with pytest.raises(GuardRefused) as ei:
        check_sum_ceiling(_CeilingModel(), 0.01, 0.49)
    assert ei.value.reason == "sum_missing"


def test_sum_exactly_at_the_floor_does_not_refuse_as_missing():
    # 0.5 == floor -> not "missing"; 0.1 is well inside 0.9*0.5=0.45 -> passes.
    assert check_sum_ceiling(_CeilingModel(), 0.1, 0.5) is None


def test_negative_setpoint_beyond_the_ceiling_is_refused_via_abs():
    with pytest.raises(GuardRefused) as ei:
        check_sum_ceiling(_CeilingModel(), -0.46, 0.5)
    assert ei.value.reason == "setpoint_exceeds_sum"


def test_negative_setpoint_within_the_ceiling_passes():
    assert check_sum_ceiling(_CeilingModel(), -0.45, 0.5) is None


def test_sum_missing_checked_before_setpoint_exceeds_sum():
    # A3: SUM below floor AND a setpoint that would also exceed a real ceiling
    # -> still sum_missing, not setpoint_exceeds_sum, so precedence is provable.
    with pytest.raises(GuardRefused) as ei:
        check_sum_ceiling(_CeilingModel(), 10.0, 0.1)
    assert ei.value.reason == "sum_missing"


def test_custom_model_ceiling_and_floor_are_honored():
    model = _CeilingModel(sum_floor_v=1.0, sum_ceiling_frac=0.5)
    assert check_sum_ceiling(model, 1.0, 2.0) is None   # 0.5*2.0 = 1.0 -> passes
    with pytest.raises(GuardRefused) as ei:
        check_sum_ceiling(model, 1.01, 2.0)
    assert ei.value.reason == "setpoint_exceeds_sum"
    with pytest.raises(GuardRefused) as ei2:
        check_sum_ceiling(model, 0.1, 0.99)   # below the custom 1.0 floor
    assert ei2.value.reason == "sum_missing"


# =========================================================== WriteInterlocks ==

@dataclass(frozen=True)
class _RailModel:
    z_rail_v: float = 140.0


def _on_trip_recorder():
    calls = []
    return calls, (lambda reason: calls.append(reason))


def _wi(model=None, config=None, on_trip=None):
    calls, cb = _on_trip_recorder()
    w = WriteInterlocks(model or _RailModel(), config or InterlockConfig(),
                        on_trip=on_trip or cb)
    return w, calls


def _clean(t, **overrides):
    base = {"t": t, "z_out_v": 75.0, "sum_v": 5.0, "defl_v": 0.0}
    base.update(overrides)
    return base


# ------------------------------------------------------------- rail_dwell --

def test_rail_dwell_49s_then_clean_sample_then_49s_more_does_not_trip():
    """A5: 4.9s of rail, then ONE clean sample (resets the clock), then 4.9s
    more of rail -- neither streak alone reaches 5s continuous, so this must
    NEVER trip despite 9.8s of total rail exposure."""
    w, calls = _wi()
    w.arm(sum_baseline_v=5.0)
    w.feed(**_clean(0.0, z_out_v=141.0))
    w.feed(**_clean(4.9, z_out_v=141.0))
    assert calls == [], "must not trip at 4.9s continuous"
    w.feed(**_clean(4.95, z_out_v=100.0))          # one clean sample resets
    assert calls == []
    w.feed(**_clean(4.96, z_out_v=141.0))          # new streak starts here
    w.feed(**_clean(9.86, z_out_v=141.0))          # +4.9s from the new start
    assert calls == [], (
        "a clean sample must reset the continuous clock -- 9.8s of TOTAL rail "
        "exposure across two 4.9s streaks must not trip")


def test_rail_dwell_trips_at_exactly_5s_continuous():
    w, calls = _wi()
    w.arm(sum_baseline_v=5.0)
    w.feed(**_clean(0.0, z_out_v=141.0))
    w.feed(**_clean(5.0, z_out_v=141.0))
    assert calls == ["rail_dwell"], calls


def test_rail_dwell_on_trip_fires_exactly_once_even_if_feeding_continues():
    w, calls = _wi()
    w.arm(sum_baseline_v=5.0)
    w.feed(**_clean(0.0, z_out_v=141.0))
    w.feed(**_clean(5.0, z_out_v=141.0))
    assert calls == ["rail_dwell"]
    w.feed(**_clean(6.0, z_out_v=141.0))
    w.feed(**_clean(20.0, z_out_v=141.0))
    assert calls == ["rail_dwell"], (
        "on_trip must fire exactly once, not once per feed() past the trip")


def test_interlock_self_disarms_after_a_trip():
    w, calls = _wi()
    w.arm(sum_baseline_v=5.0)
    assert w.armed is True
    w.feed(**_clean(0.0, z_out_v=141.0))
    w.feed(**_clean(5.0, z_out_v=141.0))
    assert calls == ["rail_dwell"]
    assert w.armed is False, "A6: the interlock must self-disarm after a trip"


def test_disarm_stops_evaluation():
    w, calls = _wi()
    w.arm(sum_baseline_v=5.0)
    w.feed(**_clean(0.0, z_out_v=141.0))
    w.disarm()
    w.feed(**_clean(4.0, z_out_v=141.0))
    w.feed(**_clean(10.0, z_out_v=141.0))
    assert calls == [], "disarm() must stop the interlock from ever tripping"
    assert w.armed is False


def test_feed_before_arm_is_ignored():
    w, calls = _wi()
    assert w.armed is False
    w.feed(**_clean(0.0, z_out_v=141.0))
    w.feed(**_clean(10.0, z_out_v=141.0))
    assert calls == [], "A7: feed() before arm() must be a silent no-op"


def test_rail_threshold_uses_model_z_rail_v_not_a_hardcoded_140():
    """A4: swapping model.z_rail_v changes the trip point, proving the running
    threshold is read from the model, not baked into WriteInterlocks."""
    w, calls = _wi(model=_RailModel(z_rail_v=100.0))
    w.arm(sum_baseline_v=5.0)
    w.feed(**_clean(0.0, z_out_v=101.0))     # above the LOWERED rail
    w.feed(**_clean(5.0, z_out_v=101.0))
    assert calls == ["rail_dwell"], (
        "101V must trip against a model.z_rail_v of 100.0")


def test_rail_dwell_139_9v_for_10s_does_not_trip():
    w, calls = _wi()
    w.arm(sum_baseline_v=5.0)
    w.feed(**_clean(0.0, z_out_v=139.9))
    w.feed(**_clean(10.0, z_out_v=139.9))
    assert calls == [], "139.9V is below model.z_rail_v=140.0 -- must not trip"


def test_rail_dwell_140_1v_trips():
    w, calls = _wi()
    w.arm(sum_baseline_v=5.0)
    w.feed(**_clean(0.0, z_out_v=140.1))
    w.feed(**_clean(5.0, z_out_v=140.1))
    assert calls == ["rail_dwell"], "140.1V is above model.z_rail_v=140.0"


# ------------------------------------------------------------ sum_collapse --

def test_sum_collapse_via_absolute_floor():
    w, calls = _wi(config=InterlockConfig(sum_floor_v=0.5))
    w.arm(sum_baseline_v=5.0)
    w.feed(**_clean(0.0, sum_v=0.4))
    w.feed(**_clean(5.0, sum_v=0.4))
    assert calls == ["sum_collapse"], calls


def test_sum_collapse_via_baseline_fraction_trips_at_99pct_of_half_baseline():
    """baseline=2.0, sum_frac=0.5 -> threshold 1.0; feed 0.99 for 5s -> trips."""
    w, calls = _wi(config=InterlockConfig(sum_frac=0.5, sum_floor_v=0.1))
    w.arm(sum_baseline_v=2.0)
    w.feed(**_clean(0.0, sum_v=0.99))
    w.feed(**_clean(5.0, sum_v=0.99))
    assert calls == ["sum_collapse"], calls


def test_sum_collapse_via_baseline_fraction_101pct_does_not_trip():
    w, calls = _wi(config=InterlockConfig(sum_frac=0.5, sum_floor_v=0.1))
    w.arm(sum_baseline_v=2.0)
    w.feed(**_clean(0.0, sum_v=1.01))
    w.feed(**_clean(10.0, sum_v=1.01))
    assert calls == [], "1.01 is above 0.5*2.0=1.0 -- must not trip"


# ------------------------------------------------------ deflection_integrity --
#
# M2 (adversarial review): the earlier draft armed baseline=5.0 and fed
# sum_v=0.5, which by ITSELF trips sum_collapse (0.5 < default sum_frac 0.5 *
# baseline 5.0 = 2.5) regardless of deflection -- so the "does not trip" case
# could never pass a correct implementation, and the "trips" case could not
# prove WHICH condition tripped. Isolated here with baseline=1.0, sum_v=0.6:
# 0.6 is above BOTH the absolute floor (0.5) and the fraction threshold
# (0.5*1.0=0.5), so sum_collapse stays clear and only deflection is exercised.

def test_deflection_integrity_trips_when_abs_defl_exceeds_sum_for_5s():
    w, calls = _wi()
    w.arm(sum_baseline_v=1.0)
    w.feed(**_clean(0.0, sum_v=0.6, defl_v=0.7))
    w.feed(**_clean(5.0, sum_v=0.6, defl_v=0.7))
    assert calls == ["deflection_integrity"], calls


def test_deflection_integrity_does_not_trip_when_within_sum():
    w, calls = _wi()
    w.arm(sum_baseline_v=1.0)
    w.feed(**_clean(0.0, sum_v=0.6, defl_v=0.5))
    w.feed(**_clean(10.0, sum_v=0.6, defl_v=0.5))
    assert calls == [], "|0.5| <= 0.6 must never trip deflection_integrity"


def test_deflection_integrity_equal_to_sum_does_not_trip():
    """S4: |defl| == sum is NOT a trip -- the rule is strictly `>`."""
    w, calls = _wi()
    w.arm(sum_baseline_v=1.0)
    w.feed(**_clean(0.0, sum_v=0.6, defl_v=0.6))
    w.feed(**_clean(10.0, sum_v=0.6, defl_v=-0.6))
    assert calls == [], "|defl| == sum must not trip (strict > only)"


def test_reason_strings_are_exact():
    """All three reasons, spelled exactly as 12.1 names them. Each uses values
    isolated to ONLY its own condition (M2)."""
    w, calls = _wi()
    w.arm(sum_baseline_v=5.0)
    w.feed(**_clean(0.0, z_out_v=141.0))
    w.feed(**_clean(5.0, z_out_v=141.0))
    assert calls[-1] == "rail_dwell"

    w2, calls2 = _wi()
    w2.arm(sum_baseline_v=5.0)
    w2.feed(**_clean(0.0, sum_v=0.1))
    w2.feed(**_clean(5.0, sum_v=0.1))
    assert calls2[-1] == "sum_collapse"

    w3, calls3 = _wi()
    w3.arm(sum_baseline_v=1.0)
    w3.feed(**_clean(0.0, sum_v=0.6, defl_v=-0.7))
    w3.feed(**_clean(5.0, sum_v=0.6, defl_v=-0.7))
    assert calls3[-1] == "deflection_integrity"


# ---------------------------------------- S3: isolate the absolute SUM floor --

def test_sum_collapse_via_floor_alone_when_fraction_would_not_trip():
    """baseline=0.6 -> fraction threshold = 0.5*0.6 = 0.3; sum_v=0.45 is ABOVE
    that (0.45 > 0.3, so the fraction condition alone would not trip) but BELOW
    the absolute floor (0.5), so the floor alone must still trip sum_collapse."""
    w, calls = _wi(config=InterlockConfig(sum_floor_v=0.5, sum_frac=0.5))
    w.arm(sum_baseline_v=0.6)
    w.feed(**_clean(0.0, sum_v=0.45))
    w.feed(**_clean(5.0, sum_v=0.45))
    assert calls == ["sum_collapse"], (
        "0.45 is below the 0.5 floor even though it clears the 0.3 fraction "
        f"threshold -- the floor alone must trip: {calls}")


def test_sum_collapse_exactly_at_the_floor_does_not_trip():
    """Should-fix (coordinator re-review): the floor is a STRICT `<`, not
    `<=`. baseline=0.9 keeps the fraction threshold (0.5*0.9=0.45) below
    sum_v=0.5 too, so this isolates the floor comparison alone -- sum_v ==
    0.5 (exactly the default `sum_floor_v`) held for well over 5s continuous
    must never trip."""
    w, calls = _wi()
    w.arm(sum_baseline_v=0.9)
    w.feed(**_clean(0.0, sum_v=0.5))
    w.feed(**_clean(7.0, sum_v=0.5))
    assert calls == [], (
        "sum_v == model floor (0.5) exactly must NOT trip sum_collapse -- "
        f"the rule is strictly BELOW the floor: {calls}")


# ============================================ M3: continuous-time coverage ==
# for sum_collapse and deflection_integrity, mirroring rail_dwell's own
# 4.9s/clean/4.9s, sub-5s, and exact-5s cases above, plus on_trip-once and
# self-disarm. A sample-COUNT based implementation (trip after N feed() calls)
# is deliberately defeated by feeding many samples within a sub-5s window.

def _active_kwargs(condition: str, active: bool) -> dict:
    """Full kwargs for `feed()` (t, z_out_v, sum_v, defl_v are ALL required --
    there is no default in the 12.1 signature), with every channel OTHER than
    the one under test held at an inert value so only the named condition can
    possibly trip."""
    if condition == "sum_collapse":
        return {"z_out_v": 75.0, "sum_v": 0.1 if active else 5.0, "defl_v": 0.0}
    if condition == "deflection_integrity":
        # baseline=1.0 below (isolated per M2): sum_v=0.6 never trips
        # sum_collapse on its own (above both the 0.5 floor and the 0.5
        # fraction threshold), so only defl_v varies here.
        return {"z_out_v": 75.0, "sum_v": 0.6, "defl_v": 0.7 if active else 0.5}
    raise ValueError(condition)


def _arm_for(condition: str):
    if condition == "deflection_integrity":
        return _wi(), 1.0
    return _wi(), 5.0


_CONTINUOUS_CASES = ["sum_collapse", "deflection_integrity"]


@pytest.mark.parametrize("condition", _CONTINUOUS_CASES)
def test_condition_49s_then_clean_then_49s_more_does_not_trip(condition):
    (w, calls), baseline = _arm_for(condition)
    w.arm(sum_baseline_v=baseline)
    w.feed(t=0.0, **_active_kwargs(condition, True))
    w.feed(t=4.9, **_active_kwargs(condition, True))
    assert calls == [], f"{condition} must not trip at 4.9s continuous"
    w.feed(t=4.95, **_active_kwargs(condition, False))       # resets the clock
    assert calls == []
    w.feed(t=4.96, **_active_kwargs(condition, True))        # new streak starts
    w.feed(t=9.86, **_active_kwargs(condition, True))        # +4.9s from restart
    assert calls == [], (
        f"{condition}: a clean sample must reset the continuous clock -- 9.8s "
        f"of TOTAL exposure across two 4.9s streaks must not trip")


@pytest.mark.parametrize("condition", _CONTINUOUS_CASES)
def test_condition_many_samples_under_5s_does_not_trip(condition):
    """Defeats a sample-COUNT implementation: 7 triggering samples, all within
    a 3s window (well under the 5s continuous requirement)."""
    (w, calls), baseline = _arm_for(condition)
    w.arm(sum_baseline_v=baseline)
    for t in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        w.feed(t=t, **_active_kwargs(condition, True))
    assert calls == [], (
        f"{condition}: 7 samples spanning only 3.0s continuous must not trip "
        f"a TIME-based implementation, even though a COUNT-based one might")


@pytest.mark.parametrize("condition", _CONTINUOUS_CASES)
def test_condition_trips_at_exactly_5s_continuous(condition):
    (w, calls), baseline = _arm_for(condition)
    w.arm(sum_baseline_v=baseline)
    w.feed(t=0.0, **_active_kwargs(condition, True))
    w.feed(t=5.0, **_active_kwargs(condition, True))
    assert calls == [condition], calls


@pytest.mark.parametrize("condition", _CONTINUOUS_CASES)
def test_condition_on_trip_fires_once_and_self_disarms(condition):
    (w, calls), baseline = _arm_for(condition)
    w.arm(sum_baseline_v=baseline)
    w.feed(t=0.0, **_active_kwargs(condition, True))
    w.feed(t=5.0, **_active_kwargs(condition, True))
    assert calls == [condition]
    w.feed(t=6.0, **_active_kwargs(condition, True))
    w.feed(t=20.0, **_active_kwargs(condition, True))
    assert calls == [condition], (
        f"{condition}: on_trip must fire exactly once, and the interlock must "
        f"self-disarm (feed() after the trip must be a no-op)")
    assert w.armed is False


# ================================================ twin integration (A8-A11) ==

def _wait(cond, timeout=8.0, interval=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return False


@pytest.fixture(scope="module")
def client(_flex_provider_off_module):
    from fastapi.testclient import TestClient

    from afm_litho.service.app import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def _flex_provider_off_module():
    import os
    prev = os.environ.get("AFM_LITHO_FLEX")
    os.environ["AFM_LITHO_FLEX"] = "0"
    yield
    if prev is None:
        os.environ.pop("AFM_LITHO_FLEX", None)
    else:
        os.environ["AFM_LITHO_FLEX"] = prev


# ============================================ M4: door-level sum-ceiling ==
# wiring proof. `check_sum_ceiling` (unit-tested above) is never proven to be
# ACTUALLY CALLED at engage time by the unit tests alone -- these hit both
# doors (REST approach/write, FLEX startApproach/startWrite) and prove the SUM
# read is FRESH (not cached from an earlier engage) by changing the twin's SUM
# between two engages and checking the outcome flips.
#
# A12 (new, this section): FLEX's `startApproach` runs its validation
# SYNCHRONOUSLY before spawning the engage task -- `commands/scan_cmds.py:
# start_approach`'s own docstring says so explicitly ("Validation... happen
# HERE, synchronously... a refusal is a refusal of the CALL"), matching
# `startWrite`'s already-proven synchronous-refusal pattern
# (`test_command_guard.py::test_flex_start_write_refused_by_guard`). This
# section assumes `check_sum_ceiling` lands at that SAME synchronous point,
# so a refusal is an immediate JSON-RPC error, never a `run_id` a script has
# to poll to discover a refusal.

def _sum_ceiling_hub(tmp_path):
    hub = _tmp_store(tmp_path)
    hub.host.world.detector_sum_drift_v = 0.0
    return hub


def test_rest_approach_refused_by_sum_ceiling_missing_and_exceeds(client, tmp_path):
    hub = _sum_ceiling_hub(tmp_path)
    hub.host.world.detector_sum_v = 0.3
    try:
        r = client.post("/api/approach", json={"setpoint": 0.2})
        assert r.json().get("reason") == "sum_missing", r.json()

        hub.host.world.detector_sum_v = 2.0
        r2 = client.post("/api/approach", json={"setpoint": 1.9})
        assert r2.json().get("reason") == "setpoint_exceeds_sum", r2.json()
    finally:
        with contextlib.suppress(Exception):
            client.post("/api/withdraw")


def test_rest_write_refused_by_sum_ceiling_missing_and_exceeds(client, tmp_path):
    hub = _sum_ceiling_hub(tmp_path)
    _seed_free_air(hub, value_v=0.0)
    ids = _one_stroke_object(client)
    hub.host.written_objects = set()

    hub.host.world.detector_sum_v = 0.3
    r = client.post("/api/litho/write", json={
        "objects": ids, "setpoint_relative": 0.2, "amp_gain": 3.5})
    assert r.json().get("reason") == "sum_missing", r.json()

    hub.host.written_objects = set()
    hub.host.world.detector_sum_v = 2.0
    r2 = client.post("/api/litho/write", json={
        "objects": ids, "setpoint_relative": 1.9, "amp_gain": 3.5})
    assert r2.json().get("reason") == "setpoint_exceeds_sum", r2.json()


def test_sum_ceiling_engage_uses_a_fresh_sum_read_not_a_cached_one(client, tmp_path):
    """The refusal at one engage must not stick around once the twin's SUM
    genuinely recovers -- proves `check_sum_ceiling` reads DetectorSum FRESH
    at each engage, not once and cached."""
    hub = _sum_ceiling_hub(tmp_path)
    hub.host.world.detector_sum_v = 2.0
    try:
        r1 = client.post("/api/approach", json={"setpoint": 1.9})
        assert r1.json().get("reason") == "setpoint_exceeds_sum", r1.json()

        hub.host.world.detector_sum_v = 5.0    # 0.9*5.0 = 4.5 -- 1.9 now well inside
        r2 = client.post("/api/approach", json={"setpoint": 1.9})
        assert r2.json().get("reason") != "setpoint_exceeds_sum", (
            f"a stale/cached SUM read would still refuse this: {r2.json()}")
        assert r2.json().get("reason") != "sum_missing", r2.json()
    finally:
        with contextlib.suppress(Exception):
            client.post("/api/withdraw")


# ---------------------------------------------------------- FLEX door (A12) --

# NOT a module-level `pytest.importorskip("zmq")`: that would skip this ENTIRE
# file (including every pure check_sum_ceiling/WriteInterlocks unit test above)
# whenever pyzmq (the optional [flex] extra) is absent -- exactly the
# environment this file was authored in. Scoped to just the two FLEX tests via
# `_needs_zmq` instead.
try:
    import zmq
    _HAS_ZMQ = True
except ImportError:
    zmq = None
    _HAS_ZMQ = False

_needs_zmq = pytest.mark.skipif(
    not _HAS_ZMQ, reason="pyzmq is the optional [flex] extra")
# CI/pre-merge NOTE: pyzmq must actually be INSTALLED in whatever environment
# gates this PR -- a bare `skipif` protects this file's OTHER tests from being
# wiped out by a missing optional extra (see the comment above), but it also
# means `test_flex_start_approach_refused_by_sum_ceiling` and
# `test_flex_start_write_refused_by_sum_ceiling` will silently SKIP (not fail)
# if pyzmq is missing, and the FLEX side of the sum-ceiling wiring proof (M4)
# would never actually run. Confirm `pip install -e .[dev]` (which does not
# pull pyzmq -- see pyproject.toml's `[flex]` extra) also has `[flex]`
# installed, or these two tests pass as skips rather than green.


def _free_port() -> int:
    import socket
    for _ in range(50):
        a = socket.socket()
        b = socket.socket()
        try:
            a.bind(("127.0.0.1", 0))
            port = a.getsockname()[1]
            b.bind(("127.0.0.1", port + 1))
            return port
        except OSError:
            continue
        finally:
            a.close()
            b.close()
    raise RuntimeError("no free port pair")


class _Client:
    def __init__(self, endpoint: str, timeout_s: float = 30.0):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
        self.sock.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
        self.sock.connect(endpoint)
        self._n = 0

    def send(self, method, params=None) -> dict:
        self._n += 1
        req = {"jsonrpc": "2.0", "method": method, "id": str(self._n)}
        if params is not None:
            req["params"] = params
        self.sock.send_string(json.dumps(req))
        return json.loads(self.sock.recv_string())

    def close(self):
        self.sock.close(linger=0)
        self.ctx.term()


@pytest.fixture(scope="module")
def _flex_hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


@pytest.fixture(scope="module")
def _provider(_flex_hub):
    from afm_litho.service import flex_provider as fp
    prov = fp.FlexProvider(_flex_hub, bind="127.0.0.1", port=_free_port(),
                           log=lambda msg: None)
    prov.start()
    yield prov
    prov.stop()


@pytest.fixture(scope="module")
def cmd(_provider):
    c = _Client(_provider.bound["command"])
    yield c
    c.close()


def _svg_one_object() -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<image id="topo_bgd" x="0" y="0" width="4000" height="4000"/>'
        '<path d="M 0 0 L 4000 0"/>'
        "</svg>"
    )


@_needs_zmq
def test_flex_start_approach_refused_by_sum_ceiling(cmd, _flex_hub, tmp_path):
    hub = _sum_ceiling_hub(tmp_path)
    reply = cmd.send("acquireControl", {"client": "test-sum-ceiling",
                                        "deadman_s": 60})
    if "error" in reply:
        hub.control = None
        reply = cmd.send("acquireControl", {"client": "test-sum-ceiling",
                                            "deadman_s": 60})
    token = reply["result"]["token"]
    try:
        hub.host.world.detector_sum_v = 0.3
        r = cmd.send("startApproach", {"token": token, "setpoint": 0.2})
        assert "error" in r, (
            "A12: expected an immediate synchronous refusal, not a run_id: "
            f"{r}")
        assert (r["error"].get("data") or {}).get("reason") == "sum_missing", r

        hub.host.world.detector_sum_v = 2.0
        r2 = cmd.send("startApproach", {"token": token, "setpoint": 1.9})
        assert "error" in r2, r2
        assert (r2["error"].get("data") or {}).get("reason") == (
            "setpoint_exceeds_sum"), r2
    finally:
        # Both engages above were REFUSED (never actually engaged), so there
        # is nothing to withdraw here.
        cur = getattr(hub, "control", None)
        if cur is not None:
            cmd.send("releaseControl", {"token": cur.token})


@_needs_zmq
def test_flex_start_write_refused_by_sum_ceiling(cmd, _flex_hub, tmp_path):
    hub = _sum_ceiling_hub(tmp_path)
    reply = cmd.send("acquireControl", {"client": "test-sum-ceiling-write",
                                        "deadman_s": 60})
    if "error" in reply:
        hub.control = None
        reply = cmd.send("acquireControl", {"client": "test-sum-ceiling-write",
                                            "deadman_s": 60})
    token = reply["result"]["token"]
    try:
        load = cmd.send("loadPattern", {"token": token, "source": _svg_one_object()})
        assert "error" not in load, load

        hub.host.world.detector_sum_v = 0.3
        r = cmd.send("startWrite", {"token": token, "objects": "all",
                                    "setpoint_reference": "absolute", "setpoint": 0.2, "amp_gain": 1})
        assert "error" in r, r
        assert (r["error"].get("data") or {}).get("reason") == "sum_missing", r

        hub.host.world.detector_sum_v = 2.0
        r2 = cmd.send("startWrite", {"token": token, "objects": "all",
                                     "setpoint_reference": "absolute", "setpoint": 1.9, "amp_gain": 1})
        assert "error" in r2, r2
        assert (r2["error"].get("data") or {}).get("reason") == (
            "setpoint_exceeds_sum"), r2
    finally:
        cur = getattr(hub, "control", None)
        if cur is not None:
            cmd.send("releaseControl", {"token": cur.token})


_SPEED_UM_S = 2.0
_STROKE_UM = 30.0    # -> a 15s stroke, ample margin around a 5-6s trip (A11)


@dataclass(frozen=True)
class _FreeAirStub:
    value_v: float
    ts: float
    mode: str
    withdrawn: bool


@pytest.fixture(autouse=True)
def _end_the_park_episode():
    """A tripped interlock leaves `hub._safe_park["active"]` set on the shared hub. It is
    cleared here, as a new engagement does (scan_cmds), so rigs that refuse to run with a
    park in force (tests/golden/harness.py) are not blocked by a test that ran before
    them. No door clears it alone: engaging moves the instrument."""
    yield
    from afm_litho.service.app import hub
    hub.host._safe_park["active"] = False


def _tmp_store(tmp_path):
    """Fresh session store PLUS a clean slate on the cross-test hub singleton
    (NIT, adversarial review): `hub.free_air`, `hub.last_park`, `hub.fault`,
    and `hub.write_interlocks` are all reset here so a prior twin integration
    test's leftover state (a stale FreeAir baseline, a stale ParkResult, a
    latched fault) can never make a LATER test pass or fail for the wrong
    reason. `hub.engaged` is reset too -- a prior test's park normally clears
    it, but a test that errors before its own `finally` could leave it set.
    `hub.world.detector_sum_v`/`detector_sum_drift_v` are reset to a healthy
    default (5.0 V, no drift) too -- a prior sum-ceiling test that left SUM at
    0.3/2.0/5.0 must not leak into the next test's engage."""
    from afm_litho.service.app import hub
    from afm_litho.service.sessions import SessionStore
    hub.store = SessionStore(tmp_path / "s.jsonl")
    hub.host.free_air = None
    hub.host.last_park = None
    hub.fault = None
    hub.host.write_interlocks = None
    hub.host.engaged = False
    hub.host.world.detector_sum_v = 5.0
    hub.host.world.detector_sum_drift_v = 0.0
    return hub


def _seed_free_air(hub, value_v=0.0):
    try:
        from afm_litho.service.command_guard import FreeAir
        fa = FreeAir(value_v=value_v, ts=time.time(), mode="contact", withdrawn=True)
    except ImportError:
        fa = _FreeAirStub(value_v=value_v, ts=time.time(), mode="contact",
                          withdrawn=True)
    hub.host.free_air = fa
    return fa


def _one_stroke_object(client) -> list[str]:
    r = client.post("/api/design/add", json={
        "kind": "path", "name": "interlockWire", "voltage": 6.0,
        "speed_um_s": _SPEED_UM_S,
        "points_um": [[2.0, 2.0], [2.0 + _STROKE_UM, 2.0]]})
    assert r.json().get("ok"), r.json()
    return ["interlockWire"]


def _n_litho(hub):
    return len([s for s in hub.store.list() if s["kind"] == "litho"])


def _last_litho_record(hub):
    recs = [s for s in hub.store.list() if s["kind"] == "litho"]
    assert recs, "no litho session record was created"
    return recs[-1]


def _sidecar_for(tmp_path, hub, rec_id):
    from afm_litho.service import stems
    info = stems.lookup(tmp_path, rec_id)
    assert info is not None, (
        f"no stem registered for litho session {rec_id!r}")
    sidecar_path = tmp_path / "writes" / "json" / f"{info['stem']}.json"
    assert sidecar_path.is_file(), f"no litho sidecar at {sidecar_path}"
    return json.loads(sidecar_path.read_text(encoding="utf-8"))


def _start_write(client, hub, tmp_path):
    _seed_free_air(hub)
    ids = _one_stroke_object(client)
    hub.host.written_objects = set()
    n0 = _n_litho(hub)
    # amp_gain MUST equal the model's tip_amp_gain (25.0, canonical per spec/memory
    # "Tip-bias amplifier gain = 25"); the guard checks it exactly, so 3.5 here was a
    # contradiction that refused every write with amp_gain_mismatch before it could
    # start. (The check_sum_ceiling cases above pass 3.5 on purpose -- the ceiling
    # refusal precedes the amp_gain check, so those never reach it.)
    # `twin_speed` 1: the interlocks time 5 s of WALL clock, so the write has to
    # last in real time; the suite's default plays it at unlimited speed.
    r = client.post("/api/litho/write", json={
        "objects": ids, "setpoint_relative": 0.2, "amp_gain": 25.0,
        "twin_speed": 1.0})
    assert r.json().get("ok"), r.json()
    return n0


# Slice AB: the park confirms on Height < 1 V. `z_rail_stuck` makes the Height readback lie
# at 148 V, so that park cannot confirm: the vendor fallback runs, the park is critical and the
# fault latches. Its record lands after the two 2.5 s confirm windows, not within 6.5 s.
@pytest.mark.parametrize("fault_flag,expected_reason,confirms", [
    ("z_rail_stuck", "rail_dwell", False),
    ("sum_drop", "sum_collapse", True),
    ("defl_over_sum", "deflection_integrity", True),
])
def test_twin_write_faulted_interlock_parks_within_6s_and_records_abort_reason(
        client, tmp_path, fault_flag, expected_reason, confirms):
    """A8-A10, end to end: a twin write with the named fault engaged trips its
    matching interlock within 5-6 s, safe_park runs and VERIFIES (observable
    via hub.last_park and a "safety" session record -- A9), the write worker
    dies, and the closed write record's sidecar carries aborted: true and
    abort_reason == expected_reason.
    """
    hub = _tmp_store(tmp_path)

    assert hasattr(hub.host, "hardware_arc") and hasattr(hub.host.hardware_arc, "faults"), (
        "hub.hardware_arc.faults (A8, pinned) does not exist yet -- the twin "
        "fault-injection surface these interlock integration tests target is "
        "not built")
    prev_flag = getattr(hub.host.hardware_arc.faults, fault_flag, None)
    setattr(hub.host.hardware_arc.faults, fault_flag, True)
    try:
        n0 = _start_write(client, hub, tmp_path)

        t0 = time.monotonic()
        bound = 6.5 if confirms else 14.0
        if not confirms:
            parked = _wait(
                lambda: getattr(getattr(hub.host, "last_park", None), "reason", None)
                == expected_reason,
                timeout=bound)
            assert parked, (
                f"hub.last_park.reason never became {expected_reason!r} "
                f"within {bound}s (fault={fault_flag})")
            assert hub.host.last_park.verified is False and hub.host.last_park.critical is True
            assert hub.fault and hub.fault["reason"] == "park_unverified"
            return
        # The trip is timed on its own: the interlock disarms itself the moment it trips.
        # The park that follows waits for the twin plugin's fixed 2.5 s withdraw, so it is
        # waited for separately and not timed.
        tripped = _wait(lambda: not hub.host.write_interlocks.armed, timeout=bound)
        elapsed = time.monotonic() - t0
        assert tripped, f"the interlock never tripped within {bound}s (fault={fault_flag})"
        assert 4.0 <= elapsed <= 6.5, (
            f"S1: expected the 5s-continuous interlock to trip in roughly the "
            f"5-6s window (with slack for scheduling), got {elapsed:.2f}s -- "
            f"outside [4.0, 6.5]")
        parked = _wait(
            lambda: getattr(getattr(hub.host, "last_park", None), "reason", None)
            == expected_reason,
            timeout=20.0)
        assert parked, (
            f"hub.last_park.reason never became {expected_reason!r} "
            f"after the trip (fault={fault_flag})")
        assert hub.host.last_park.verified is True, (
            f"a tripped interlock's park must VERIFY on the twin (by "
            f"construction -- no HV rail, synchronous withdraw): "
            f"{hub.host.last_park}")
        assert getattr(hub, "fault", None) is None, (
            "A9 (pinned): a tripped interlock whose park verifies must NOT "
            f"latch hub.fault: {hub.fault}")

        # SessionStore.list() only surfaces id/ts/kind/label (sessions.py); the
        # reason rides in the label safe_park's `_finish` stamps
        # (`f"safe_park: {reason}"`), same observable test_safety_rail_dwell.py
        # relies on ("safety" in the kind list) but pinned to the SPECIFIC reason.
        expected_label = f"safe_park: {expected_reason}"
        kinds_and_labels = [(rec["kind"], rec.get("label"))
                            for rec in hub.store.list(50)]
        assert any(k == "safety" and lbl == expected_label
                  for k, lbl in kinds_and_labels), (
            f"no 'safety' session record labelled {expected_label!r}: "
            f"{kinds_and_labels}")

        assert _wait(lambda: _n_litho(hub) > n0, timeout=10.0), (
            "the interlock-tripped write never closed its litho session record")
        assert _wait(lambda: not hub.host.litho_thread.is_alive(), timeout=5.0), (
            "the write worker thread must be dead once the write record closed")

        rec = _last_litho_record(hub)
        sidecar = _sidecar_for(tmp_path, hub, rec["id"])
        assert sidecar.get("aborted") is True, sidecar
        assert sidecar.get("abort_reason") == expected_reason, sidecar
    finally:
        setattr(hub.host.hardware_arc.faults, fault_flag, bool(prev_flag))
        hub.fault = None
        with contextlib.suppress(Exception):
            client.post("/api/litho/abort")
        with contextlib.suppress(Exception):
            client.post("/api/withdraw")


def test_twin_write_transient_fault_cleared_before_5s_does_not_park(client, tmp_path):
    """S1: the fault is held for ~2s (well under the 5s continuous
    requirement) then cleared -- the interlock must NEVER trip, and the write
    must complete normally, proving the interlock is genuinely time-based
    rather than latching on the FIRST bad sample."""
    hub = _tmp_store(tmp_path)
    assert hasattr(hub.host, "hardware_arc") and hasattr(hub.host.hardware_arc, "faults"), (
        "hub.hardware_arc.faults (A8, pinned) does not exist yet")
    hub.host.hardware_arc.faults.z_rail_stuck = True
    try:
        n0 = _start_write(client, hub, tmp_path)
        time.sleep(2.0)
        hub.host.hardware_arc.faults.z_rail_stuck = False
        # Give it another 5s past the clear: if the interlock were latched on
        # the first bad sample (rather than requiring 5s CONTINUOUS), it would
        # have tripped by now.
        parked = _wait(
            lambda: getattr(getattr(hub.host, "last_park", None), "reason", None)
            == "rail_dwell",
            timeout=5.0)
        assert not parked, (
            "a fault held only ~2s (cleared before the 5s continuous "
            "threshold) must never trip rail_dwell")
        assert _wait(lambda: _n_litho(hub) > n0, timeout=40.0), (
            "the write never completed after the transient fault cleared")
        rec = _last_litho_record(hub)
        sidecar = _sidecar_for(tmp_path, hub, rec["id"])
        assert sidecar.get("aborted") is False, sidecar
    finally:
        hub.host.hardware_arc.faults.z_rail_stuck = False
        with contextlib.suppress(Exception):
            client.post("/api/litho/abort")
        with contextlib.suppress(Exception):
            client.post("/api/withdraw")


# ============ S2: the running interlocks are armed for approach/scan too ==
# (section 12: "armed for the whole duration of every hardware engage
# (approach, scan, write)") -- not just the write path exercised above.

def test_twin_approach_with_z_rail_stuck_is_parked_with_rail_dwell(client, tmp_path):
    hub = _tmp_store(tmp_path)
    assert hasattr(hub.host, "hardware_arc") and hasattr(hub.host.hardware_arc, "faults"), (
        "hub.hardware_arc.faults (A8, pinned) does not exist yet")
    hub.host.hardware_arc.faults.z_rail_stuck = True
    try:
        client.post("/api/approach", json={"setpoint": 0.2})
        # 14 s: the stuck Height readback cannot confirm the park (slice AB), so its record
        # lands after both 2.5 s confirm windows.
        assert _wait(
            lambda: getattr(getattr(hub.host, "last_park", None), "reason", None)
            == "rail_dwell",
            timeout=14.0), (
            "S2: a twin APPROACH with z_rail_stuck must be parked with "
            "rail_dwell -- the running rail interlock must be armed for "
            "approach, not just for a write")
    finally:
        hub.host.hardware_arc.faults.z_rail_stuck = False
        hub.fault = None
        with contextlib.suppress(Exception):
            client.post("/api/withdraw")


# The park is signalled by `last_park`; this is only the hang guard. The interlock's 5 s dwell
# is on the wall clock and starts after the scan engages, which a loaded machine delays.
RAIL_DWELL_PARK_WAIT_S = 60.0


def test_twin_scan_with_z_rail_stuck_is_parked_with_rail_dwell(client, tmp_path, litho_ready_hub):
    hub = _tmp_store(tmp_path)
    assert hasattr(hub.host, "hardware_arc") and hasattr(hub.host.hardware_arc, "faults"), (
        "hub.hardware_arc.faults (A8, pinned) does not exist yet")
    hub.host.hardware_arc.faults.z_rail_stuck = True
    cfg = {"size_um": 5.0, "pixels": 8, "lines": 24, "line_rate_hz": 2.0,
          "mode": "contact"}
    try:
        client.post("/api/scan/continuous",
                    json={"enabled": True, "config": cfg, "max_frames": 5})
        assert _wait(
            lambda: getattr(getattr(hub.host, "last_park", None), "reason", None)
            == "rail_dwell",
            timeout=RAIL_DWELL_PARK_WAIT_S), (
            "S2: a twin SCAN with z_rail_stuck must be parked with "
            "rail_dwell -- the running rail interlock must be armed for a "
            "continuous scan too")
    finally:
        hub.host.hardware_arc.faults.z_rail_stuck = False
        hub.fault = None              # the stuck Height readback cannot confirm the park
        with contextlib.suppress(Exception):
            client.post("/api/scan/continuous", json={"enabled": False})
        with contextlib.suppress(Exception):
            client.post("/api/withdraw")


def test_clean_twin_write_is_not_tripped(client, tmp_path):
    """No fault engaged: a normal twin write must complete without any
    interlock firing -- aborted: false, no rail_dwell/sum_collapse/
    deflection_integrity safety record."""
    hub = _tmp_store(tmp_path)
    for flag in ("z_rail_stuck", "sum_drop", "defl_over_sum", "free_air_jitter"):
        if hasattr(getattr(hub.host, "hardware_arc", None), "faults"):
            setattr(hub.host.hardware_arc.faults, flag, False)

    n0 = _start_write(client, hub, tmp_path)
    assert _wait(lambda: _n_litho(hub) > n0, timeout=40.0), (
        "a clean twin write never completed")

    rec = _last_litho_record(hub)
    sidecar = _sidecar_for(tmp_path, hub, rec["id"])
    assert sidecar.get("aborted") is False, sidecar
    assert sidecar.get("abort_reason") in (None, ""), (
        "a clean write must not carry an abort_reason")

    labels = [(rec2["kind"], rec2.get("label")) for rec2 in hub.store.list(50)]
    bad_labels = {f"safe_park: {r}" for r in
                 ("rail_dwell", "sum_collapse", "deflection_integrity")}
    assert not any(k == "safety" and lbl in bad_labels for k, lbl in labels), labels
