"""RED tests for `src/afm_litho/hardware/free_air.py` (wave-1 slice T5,
docs/INSTRUMENT_MODEL_SPEC.md section 12 + 12.1's interface contract).

`hardware/free_air.py` does not exist yet, so this whole file is expected to fail
at COLLECTION with a `ModuleNotFoundError` today; `pytest --collect-only -q` must
still collect every other test file uneventfully.

Contract under test (12.1, plus the 12.1-addendum spec 57613f4 "near-zero
free-air mean"):
    @dataclass(frozen=True)
    class FreeAirRead: value_v: float; samples: tuple[float, float, float]; ts: float; mode: str
    class FreeAirDisagree(Exception): samples: tuple[float, ...]
    def agree_within(samples: Sequence[float], frac: float = 0.10,
                     abs_band_v: float = ...) -> bool   # pure
    # agreement band = max(frac * abs(mean(samples)), abs_band_v) -- NOT just
    # frac * mean any more (57613f4): a near-zero mean collapses the relative
    # term toward zero, so an ABSOLUTE floor keeps the band physically sane.
    def read_free_air(read_one, *, now, sleep, mode, dwell_s=5.0, frac=0.10,
                      abs_band_v=...) -> FreeAirRead
    # takes 3 samples; if not agree_within -> sleep(dwell_s), 3 more; if still not
    # -> raise FreeAirDisagree.

New model field (57613f4): `free_air_abs_band_v`, platform default 0.02 V
(explicitly a PLACEHOLDER -- "TO FIX ON THE RIG", not yet measured against the
Cypher's real withdrawn-deflection noise), overridable. Callers (approach,
write set-down, scan re-reference) are expected to pass
`model.free_air_abs_band_v.value` as `read_free_air`'s `abs_band_v`.

ASSUMPTIONS beyond 12.1 (section 12 does not spell out the exact arithmetic; each
is flagged for the implementer to confirm or correct):

  A1. `agree_within`'s "agreement" for an N-sample sequence is defined as
      `(max(samples) - min(samples)) <= max(frac * abs(mean(samples)),
      abs_band_v)` (57613f4) -- a symmetric, order-independent generalization
      of the existing 2-read `_within(a, b)` helper in `host/continuous.py`
      (`abs(a - b) <= frac * abs(b)`, now with the absolute floor added),
      chosen because `agree_within` takes an arbitrary `Sequence[float]` (used
      here on 3-tuples) rather than a fixed pair, and section 12's own text
      ("three reads disagree by >10%") reads as a single band all three must
      sit inside, not a chain of pairwise comparisons against one arbitrarily-
      chosen reference sample.
  A2. The boundary is INCLUSIVE: spread `== frac * mean` agrees. Chosen because
      12.1's `check_sum_ceiling` example ("0.45 with SUM 0.5 passes; 0.46
      refused") demonstrates the same "at the line passes" convention elsewhere
      in this same section, so it is taken as the house style pending
      confirmation. Tested at `frac=0.5` with binary-exact (dyadic) sample
      values rather than `frac=0.10` -- `agree_within([0.95, 1.0, 1.05], 0.10)`
      looks like an exact boundary on paper, but `1.05 - 0.95 ==
      0.10000000000000009` in IEEE-754 double, not `0.1`, so a test built on it
      would be asserting on a rounding accident rather than the actual
      boundary rule (caught in adversarial review).
  A3. NaN or Inf ANYWHERE in `samples` makes `agree_within` return False
      unconditionally (12 explicitly requires "NaN/Inf never agree"), evaluated
      BEFORE the spread arithmetic so a bad read cannot accidentally cancel out
      in a mean/spread computation (inf - inf is NaN, which would otherwise make
      a `<=` comparison silently False for the wrong reason -- this test pins it
      as an explicit, not incidental, guarantee).
  A4. `read_free_air`'s three normal-path calls to `read_one` happen with no
      sleep between them (only the one `dwell_s` sleep gates the SECOND triple);
      `sleep` is otherwise never called on the happy path or on the immediate-
      agreement path.
  A5. On the double-disagreement path, `FreeAirDisagree.samples` carries ALL SIX
      reads (the first triple followed by the second), per the brief's "disagree
      twice -> FreeAirDisagree with all samples" -- not just the second triple --
      so the caller/log can show the operator the whole failed sequence.
  A6. `read_one` raising propagates the raw exception (not wrapped), and no
      `sleep`/further `read_one` calls happen after a raise -- consistent with
      12's fail-safe-first house style (a broken read channel must surface
      immediately, not be retried into a different failure).
  A7. PINNED (coordinator, instrument-model b30c9fd, end of 12.1): the TWIN
      fault-injection hooks (`twin.faults.free_air_jitter` etc.) live on
      `SoftwareARC.faults`, reachable as `hub.hardware_arc.faults` on the twin
      backend -- NOT `hub.arc.faults` (this file's earlier guess, corrected).
      Same object T3 already puts on `SoftwareARC`
      (`tests/twin/test_cypher_profile.py`: "`.faults.ramp_noop: bool`").
  A8. With `free_air_jitter` engaged, the first three `_read_free_air_deflection`
      calls disagree with each other (by construction of the twin's fault path)
      and the next three all return the WORLD's true `free_air_deflection()`
      value (`hub.world.free_air_defl_v`, held constant here via
      `free_air_defl_drift_v = 0.0`) -- so this test sets a known, distinctive
      baseline and asserts `hub.free_air.value_v` equals it exactly after the
      dwell, rather than asserting anything about the jittered first triple's
      values (which 12.1 leaves fully unspecified).
  A9. The approach door (`POST /api/approach`) is the one exercised for the twin
      integration case: 12.1 says `read_free_air` is "Used by: the scan
      re-reference..., approach, write set-down (both doors)", and approach is
      the simplest single blocking call among those to await a >5 s dwell on.
  A10. `agree_within`'s and `read_free_air`'s `abs_band_v` DEFAULT TO 0.0 (both
      take primitives, not an `InstrumentModel` -- there is nowhere inside
      either pure function to source 0.02 from), i.e. a caller that omits it
      gets today's pre-57613f4 pure-relative behavior back. Most unit tests
      below pass it explicitly anyway for clarity; one test
      (`test_read_free_air_near_zero_mean_without_abs_band_v_dwells_and_
      disagrees`) deliberately OMITS it to prove the 0.0 default is real, not
      a required positional the implementer might make mandatory. The twin
      integration test relies on the REST door (`approach`) reading
      `model.free_air_abs_band_v.value` and passing it through -- the
      door-level GET `/api/instrument` test below proves the field's VALUE
      (0.02, `platform_default`) exists on the model, not that the plumbing
      from model to `read_free_air` call exists end to end; that latter proof
      is the twin integration test itself succeeding with a near-zero
      baseline.
  A11 (GET /api/instrument shape). Per `routers/instrument.py`'s already-
      merged `to_dict()`, a model field serializes as
      `{"value": ..., "source": "platform_default"}` -- this file asserts
      `body["free_air_abs_band_v"] == {"value": pytest.approx(0.02),
      "source": "platform_default"}` accordingly (not against
      `tests/test_instrument_model.py`, which is T1's file -- per the
      coordinator's explicit instruction this round).

RESOLVED (was S5, open question raised in adversarial review): it was decided the
near-zero rule in spec 57613f4 (see the Contract section above) -- the
`xfail(strict=False)` TODO stub this file previously carried is REPLACED below
by real tests of the `max(frac * |mean|, abs_band_v)` rule.
"""

from __future__ import annotations

import contextlib
import time

import pytest
from afm_litho.hardware.free_air import (
    FreeAirDisagree,
    FreeAirRead,
    agree_within,
    read_free_air,
)

# ============================================================ agree_within ==
#
# M1 (adversarial review): the boundary must be proven with BINARY-EXACT
# (dyadic) values, never an epsilon or a decimal literal that merely LOOKS
# exact on paper. `frac=0.5` with samples 0.75/1.0/1.25 is exact in IEEE-754
# double (all are n/4 for small integer n): mean = (0.75+1.0+1.25)/3 = 1.0
# exactly, spread = 1.25-0.75 = 0.5 exactly, frac*mean = 0.5*1.0 = 0.5 exactly
# -> the boundary comparison is bit-for-bit, not approximate.

def test_agree_within_true_exactly_at_the_dyadic_boundary():
    # mean = 1.0, spread = 0.5 == frac * mean, both bit-exact -> agrees (A2).
    # abs_band_v=0.0 (A10): the absolute floor is neutral here since
    # frac*mean (0.5) dominates it.
    assert agree_within([0.75, 1.0, 1.25], frac=0.5, abs_band_v=0.0) is True


def test_agree_within_false_clearly_outside_the_dyadic_boundary():
    # mean = 1.0 exactly (0.5+1.0+1.5=3.0/3); spread = 1.0, DOUBLE frac*mean
    # (0.5) -- clearly outside, no epsilon involved.
    assert agree_within([0.5, 1.0, 1.5], frac=0.5, abs_band_v=0.0) is False


def test_agree_within_true_for_identical_samples():
    assert agree_within([2.0, 2.0, 2.0], frac=0.10, abs_band_v=0.0) is True


def test_agree_within_false_for_widely_spread_samples():
    assert agree_within([0.0, 5.0, 10.0], frac=0.10, abs_band_v=0.0) is False


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_agree_within_nan_or_inf_never_agrees(bad):
    assert agree_within([1.0, 1.0, bad], frac=0.10, abs_band_v=0.0) is False


def test_agree_within_all_nan_never_agrees():
    assert agree_within([float("nan")] * 3, frac=0.10, abs_band_v=0.0) is False


def test_agree_within_default_frac_is_10_percent_directionally():
    # NOT a boundary claim (see M1) -- just proves the default parameter value
    # is really 0.10 by bracketing it with a clearly-inside and a clearly-
    # outside case at that frac. abs_band_v=0.0 keeps the absolute floor out
    # of it entirely.
    assert agree_within([0.99, 1.0, 1.01], abs_band_v=0.0) is True   # spread 0.02 << 0.10
    assert agree_within([0.0, 5.0, 10.0], abs_band_v=0.0) is False    # spread 10 >> 0.10


# ==================================== 57613f4: near-zero free-air mean ==
#
# S5, RESOLVED: agreement band = max(frac * |mean|, abs_band_v). Model
# field `free_air_abs_band_v` defaults to 0.02 V (a rig placeholder -- see the
# module docstring). These replace the earlier xfail TODO stub.

def test_agree_within_near_zero_mean_small_jitter_agrees_via_abs_band():
    """mean is essentially 0 ((-0.001+0.0+0.001)/3 == 0.0 exactly), so
    frac*|mean| == 0.0 -- WITHOUT the absolute floor this would disagree on
    ANY nonzero spread. With abs_band_v=0.02 (the rig placeholder default),
    spread=0.002 is comfortably inside the absolute floor -- agrees."""
    samples = [-0.001, 0.0, 0.001]
    assert sum(samples) == 0.0            # mean is bit-exact zero -- no epsilon
    assert agree_within(samples, frac=0.10, abs_band_v=0.02) is True


def test_agree_within_near_zero_mean_same_spread_disagrees_without_abs_band():
    """Same near-zero-mean samples as above, but abs_band_v=0.0 (no floor):
    frac*|mean| == 0.0, so ANY nonzero spread must disagree -- proving the
    absolute floor is what makes the previous test pass, not some other
    leniency in the comparison."""
    samples = [-0.001, 0.0, 0.001]
    assert agree_within(samples, frac=0.10, abs_band_v=0.0) is False


def test_agree_within_near_zero_mean_larger_spread_disagrees_even_with_abs_band():
    """[0.0, 0.03, 0.0]: mean = 0.01, spread = 0.03. frac*mean = 0.10*0.01 =
    0.001 and abs_band_v = 0.02 -- band = max(0.001, 0.02) = 0.02. spread
    (0.03) is past BOTH the relative term and the 0.02 absolute floor, so
    this must still disagree -- the floor raises the bar near zero, it does
    not make agreement unconditional."""
    samples = [0.0, 0.03, 0.0]
    assert agree_within(samples, frac=0.10, abs_band_v=0.02) is False


def test_agree_within_large_mean_10pct_still_dominates_the_floor():
    """Away from zero, the RELATIVE term (frac*mean) must still be the one
    that governs once it exceeds abs_band_v (0.02) -- the floor must not
    silently widen the band for an ordinary, non-near-zero baseline. mean=10.0
    exactly for both samples (symmetric), so frac*mean=1.0 >> abs_band_v=0.02
    either way; a clearly-inside spread (0.2) agrees, a clearly-outside one
    (5.0) does not -- no boundary-epsilon risk (M1), just direction."""
    assert agree_within([9.9, 10.0, 10.1], frac=0.10, abs_band_v=0.02) is True
    assert agree_within([7.5, 10.0, 12.5], frac=0.10, abs_band_v=0.02) is False


def test_read_free_air_near_zero_mean_agrees_immediately_with_abs_band_v():
    """`read_free_air` must actually PASS `abs_band_v` through to
    `agree_within` (A10): with abs_band_v=0.02 and near-zero, small-jitter
    reads, the first triple agrees immediately -- no dwell, exactly 3
    read_one calls."""
    reads = [-0.001, 0.0, 0.001]
    calls = {"read": 0}

    def read_one():
        calls["read"] += 1
        return reads[calls["read"] - 1]

    sleeps = []
    out = read_free_air(read_one, now=_now_stub(), sleep=_sleep_recorder(sleeps),
                        mode="contact", abs_band_v=0.02)
    assert calls["read"] == 3, (
        "abs_band_v=0.02 should let a near-zero, small-jitter triple agree "
        f"immediately, not dwell: {calls}")
    assert sleeps == []
    assert out.value_v == pytest.approx(0.0, abs=1e-9)


def test_read_free_air_near_zero_mean_without_abs_band_v_dwells_and_disagrees():
    """Same near-zero, small-jitter reads, but abs_band_v NOT passed (A10:
    defaults to 0.0, the pre-57613f4 pure-relative behavior) -- both triples
    disagree (frac*|mean| collapses to ~0 near zero), so this must dwell
    once and then raise FreeAirDisagree, proving `abs_band_v` is genuinely
    load-bearing rather than a no-op parameter."""
    reads = [-0.001, 0.0, 0.001, -0.001, 0.0, 0.001]
    calls = {"read": 0}

    def read_one():
        calls["read"] += 1
        return reads[calls["read"] - 1]

    sleeps = []
    with pytest.raises(FreeAirDisagree):
        read_free_air(read_one, now=_now_stub(), sleep=_sleep_recorder(sleeps),
                      mode="contact")
    assert calls["read"] == 6
    assert sleeps == [5.0]


# ========================================================== read_free_air ==

def _now_stub(value=123456.0):
    return lambda: value


def _sleep_recorder(calls):
    def _sleep(seconds):
        calls.append(seconds)
    return _sleep


def test_read_free_air_happy_path_returns_mean_and_samples():
    reads = [1.0, 1.0, 1.0]
    calls = {"read": 0}

    def read_one():
        calls["read"] += 1
        return reads[calls["read"] - 1]

    sleeps = []
    out = read_free_air(read_one, now=_now_stub(999.0), sleep=_sleep_recorder(sleeps),
                        mode="contact")

    assert isinstance(out, FreeAirRead)
    assert out.value_v == pytest.approx(1.0)
    assert out.samples == (1.0, 1.0, 1.0)
    assert out.ts == pytest.approx(999.0)
    assert out.mode == "contact"
    assert calls["read"] == 3, "happy path must take exactly 3 samples"
    assert sleeps == [], "no dwell on the happy (agreeing) path"


def test_read_free_air_disagree_then_agree_sleeps_once_and_uses_second_triple():
    """First triple disagrees wildly; dwell once; second triple agrees at 2.0.
    Exactly 6 read_one calls, exactly one sleep(5.0), and the returned value is
    the mean of the SECOND triple, not the first."""
    reads = [0.0, 10.0, 20.0, 2.0, 2.0, 2.0]
    calls = {"read": 0}

    def read_one():
        calls["read"] += 1
        return reads[calls["read"] - 1]

    sleeps = []
    out = read_free_air(read_one, now=_now_stub(42.0), sleep=_sleep_recorder(sleeps),
                        mode="contact")

    assert calls["read"] == 6, f"expected 6 read_one calls, got {calls['read']}"
    assert sleeps == [5.0], f"expected exactly one sleep(5.0), got {sleeps}"
    assert out.value_v == pytest.approx(2.0)
    assert out.samples == (2.0, 2.0, 2.0)
    assert out.ts == pytest.approx(42.0)


def test_read_free_air_disagree_twice_raises_with_all_samples():
    """Both triples disagree -> FreeAirDisagree, read_one called exactly 6 times,
    sleep called exactly once. A6/A5: the exception carries all six samples."""
    reads = [0.0, 10.0, 20.0, 0.0, 10.0, 20.0]
    calls = {"read": 0}

    def read_one():
        calls["read"] += 1
        return reads[calls["read"] - 1]

    sleeps = []
    with pytest.raises(FreeAirDisagree) as ei:
        read_free_air(read_one, now=_now_stub(), sleep=_sleep_recorder(sleeps),
                      mode="contact")

    assert calls["read"] == 6, f"expected 6 read_one calls, got {calls['read']}"
    assert sleeps == [5.0], f"expected exactly one sleep(5.0), got {sleeps}"
    assert tuple(ei.value.samples) == (0.0, 10.0, 20.0, 0.0, 10.0, 20.0), (
        "FreeAirDisagree.samples must carry all six reads (both triples)")


def test_read_free_air_uses_custom_dwell_and_frac():
    reads = [0.0, 100.0, 200.0, 5.0, 5.0, 5.0]
    calls = {"read": 0}

    def read_one():
        calls["read"] += 1
        return reads[calls["read"] - 1]

    sleeps = []
    out = read_free_air(read_one, now=_now_stub(), sleep=_sleep_recorder(sleeps),
                        mode="ac", dwell_s=1.5, frac=0.5)
    assert sleeps == [1.5]
    assert out.mode == "ac"
    assert out.value_v == pytest.approx(5.0)


def test_read_one_raising_propagates_immediately():
    calls = {"read": 0}

    def read_one():
        calls["read"] += 1
        raise RuntimeError("bridge closed")

    sleeps = []
    with pytest.raises(RuntimeError, match="bridge closed"):
        read_free_air(read_one, now=_now_stub(), sleep=_sleep_recorder(sleeps),
                      mode="contact")
    assert calls["read"] == 1, (
        "a raising read_one must propagate immediately, not be retried")
    assert sleeps == [], "no dwell after a raised read"


def test_read_one_raising_on_the_second_triple_propagates():
    """A disagreement on the first triple dwells once, then a raise on the
    second triple's reads still propagates (not swallowed into FreeAirDisagree)."""
    reads = [0.0, 10.0, 20.0]
    calls = {"read": 0}

    def read_one():
        calls["read"] += 1
        if calls["read"] <= 3:
            return reads[calls["read"] - 1]
        raise RuntimeError("bridge closed mid-dwell")

    sleeps = []
    with pytest.raises(RuntimeError, match="bridge closed mid-dwell"):
        read_free_air(read_one, now=_now_stub(), sleep=_sleep_recorder(sleeps),
                      mode="contact")
    assert sleeps == [5.0]
    assert calls["read"] == 4, (
        "the raise on the 4th call (first of the second triple) must propagate "
        "immediately, not be retried into a 5th/6th read")


def test_read_free_air_ts_comes_from_injected_now_not_wall_clock():
    reads = [1.0, 1.0, 1.0]
    calls = {"read": 0}

    def read_one():
        calls["read"] += 1
        return reads[calls["read"] - 1]

    sentinel_ts = 31415.9265
    out = read_free_air(read_one, now=lambda: sentinel_ts, sleep=lambda s: None,
                        mode="contact")
    assert out.ts == sentinel_ts
    # Sanity: the sentinel is nowhere near the real wall clock, so a stub that
    # silently fell back to time.time() would fail this by a wide margin.
    assert abs(out.ts - time.time()) > 1000


def test_free_air_disagree_is_an_exception_subclass():
    exc = FreeAirDisagree((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    assert isinstance(exc, Exception)
    assert tuple(exc.samples) == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


# =============================================== twin integration (A7-A9) ==

def _wait(cond, timeout=8.0, interval=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return False


@pytest.fixture()
def _twin_client():
    """Own module-independent client: the free-air jitter integration test needs
    a live wall-clock dwell (>5 s), so it is kept out of the fast unit-test
    module scope used elsewhere in this repo's suite."""
    import os
    prev_flex = os.environ.get("AFM_LITHO_FLEX")
    os.environ["AFM_LITHO_FLEX"] = "0"
    from fastapi.testclient import TestClient

    from afm_litho.service.app import app
    with TestClient(app) as c:
        yield c
    if prev_flex is None:
        os.environ.pop("AFM_LITHO_FLEX", None)
    else:
        os.environ["AFM_LITHO_FLEX"] = prev_flex


def test_twin_approach_survives_free_air_jitter_after_one_dwell(_twin_client):
    """A7/A8/A9: with `hub.hardware_arc.faults.free_air_jitter` engaged, an approach still
    succeeds (the dwell-and-retry recovers), and the free-air baseline the
    service latches equals the WORLD's true (settled) free-air deflection, not
    the jittered first triple.

    Real wall-clock dwell (>5 s): this test is intentionally slow, matching the
    house style of the other cross-cutting twin integration tests in this suite
    (e.g. tests/test_safety_rail_dwell.py's twin tests, which also block on real
    time rather than a mocked clock, so the service's ACTUAL sleep(5.0) path is
    exercised end to end).
    """
    from afm_litho.service.app import hub

    known_free_air_v = 0.0777
    prev_defl = hub.host.world.free_air_defl_v
    prev_drift = hub.host.world.free_air_defl_drift_v
    prev_engaged = hub.host.engaged
    prev_free_air = getattr(hub.host, "free_air", None)
    prev_last_park = getattr(hub.host, "last_park", None)
    hub.host.world.free_air_defl_v = known_free_air_v
    hub.host.world.free_air_defl_drift_v = 0.0
    hub.host.free_air = None
    hub.host.last_park = None

    assert hasattr(hub.host, "hardware_arc") and hasattr(hub.host.hardware_arc, "faults"), (
        "hub.hardware_arc.faults (A7, pinned) does not exist yet -- the twin "
        "fault-injection surface this integration test targets is not built")
    hub.host.hardware_arc.faults.free_air_jitter = True
    try:
        t0 = time.monotonic()
        r = _twin_client.post("/api/approach", json={"setpoint": 0.2})
        elapsed = time.monotonic() - t0
        body = r.json()
        assert body.get("ok") is True, (
            f"approach must still succeed after the dwell-and-retry: {body}")
        assert elapsed >= 5.0, (
            f"approach returned in {elapsed:.2f}s -- the 5s dwell-and-retry "
            f"does not appear to have run")

        assert hub.host.free_air is not None
        assert hub.host.free_air.value_v == pytest.approx(known_free_air_v, abs=1e-6), (
            f"hub.free_air.value_v ({hub.host.free_air.value_v}) should equal the "
            f"settled world baseline ({known_free_air_v}), i.e. the mean of the "
            f"SECOND (agreeing) triple, not the jittered first one")
    finally:
        hub.host.hardware_arc.faults.free_air_jitter = False
        hub.host.world.free_air_defl_v = prev_defl
        hub.host.world.free_air_defl_drift_v = prev_drift
        with contextlib.suppress(Exception):
            _twin_client.post("/api/withdraw")
        hub.host.engaged = prev_engaged
        hub.host.free_air = prev_free_air
        hub.host.last_park = prev_last_park


def test_twin_approach_near_zero_free_air_with_small_jitter_succeeds(_twin_client):
    """57613f4, twin integration: a near-zero settled free-air baseline (the
    exact regime where `frac * |mean|` alone would collapse toward zero) with
    the twin's jitter fault engaged must still let approach SUCCEED, once the
    door passes `model.free_air_abs_band_v.value` through to `read_free_air`.

    ASSUMPTION (reuses A7/A8's `free_air_jitter` fault -- there is no
    DEDICATED "small jitter" fault hook named anywhere in the spec, near-zero
    or otherwise): this is the twin's only existing deterministic way to make
    consecutive `_read_free_air_deflection` calls disagree, so it stands in
    for "small jitter" here too. Unlike the non-near-zero test above, this
    does NOT assert on `elapsed` (whether the fix lets the first triple agree
    immediately, or the dwell-and-retry still runs once and then succeeds on
    the second triple, both count as "succeeds"); the precise no-dwell-at-
    near-zero claim is proven at the pure-function level instead
    (`test_read_free_air_near_zero_mean_agrees_immediately_with_abs_band_v`
    above), which does not depend on how big the twin's jitter fault happens
    to be.
    """
    from afm_litho.service.app import hub

    known_free_air_v = 0.001   # near-zero, but not exactly 0.0
    prev_defl = hub.host.world.free_air_defl_v
    prev_drift = hub.host.world.free_air_defl_drift_v
    prev_engaged = hub.host.engaged
    prev_free_air = getattr(hub.host, "free_air", None)
    prev_last_park = getattr(hub.host, "last_park", None)
    hub.host.world.free_air_defl_v = known_free_air_v
    hub.host.world.free_air_defl_drift_v = 0.0
    hub.host.free_air = None
    hub.host.last_park = None

    assert hasattr(hub.host, "hardware_arc") and hasattr(hub.host.hardware_arc, "faults"), (
        "hub.hardware_arc.faults (A7, pinned) does not exist yet")
    hub.host.hardware_arc.faults.free_air_jitter = True
    try:
        r = _twin_client.post("/api/approach", json={"setpoint": 0.01})
        body = r.json()
        assert body.get("ok") is True, (
            f"approach with a near-zero free-air baseline and jitter must "
            f"still succeed once abs_band_v is wired in: {body}")
        assert hub.host.free_air is not None
        assert hub.host.free_air.value_v == pytest.approx(known_free_air_v, abs=1e-6), (
            f"hub.free_air.value_v ({hub.host.free_air.value_v}) should settle on "
            f"the true near-zero world baseline ({known_free_air_v})")
    finally:
        hub.host.hardware_arc.faults.free_air_jitter = False
        hub.host.world.free_air_defl_v = prev_defl
        hub.host.world.free_air_defl_drift_v = prev_drift
        with contextlib.suppress(Exception):
            _twin_client.post("/api/withdraw")
        hub.host.engaged = prev_engaged
        hub.host.free_air = prev_free_air
        hub.host.last_park = prev_last_park


# ============================ 57613f4: GET /api/instrument door proof ==

def test_get_instrument_reports_free_air_abs_band_v_platform_default(_twin_client):
    """New model field (57613f4): `free_air_abs_band_v`, value 0.02 (a rig
    placeholder, TO FIX per the spec -- not yet measured), source
    `platform_default`. Per the coordinator's explicit instruction this round:
    asserted via `GET /api/instrument` in THIS file, not against
    `tests/test_instrument_model.py` (T1's file). Shape per
    `routers/instrument.py`'s already-merged `InstrumentModel.to_dict()`:
    `{"value": ..., "source": "..."}` per field."""
    r = _twin_client.get("/api/instrument")
    body = r.json()
    field = body.get("free_air_abs_band_v")
    assert field is not None, (
        f"GET /api/instrument has no 'free_air_abs_band_v' field yet: "
        f"{sorted(body.keys())}")
    assert field.get("value") == pytest.approx(0.02), field
    assert field.get("source") == "platform_default", field
