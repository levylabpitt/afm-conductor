"""RED tests for `src/afm_litho/service/command_guard.py` (wave-1 slice T2,
docs/INSTRUMENT_MODEL_SPEC.md sections 2 and 10).

Neither `command_guard.py` nor `instrument_model.py` exist yet (both are wave-2
work); this file is expected to fail at COLLECTION with an ImportError until they
land, and `pytest --collect-only -q` must still collect every other test file
uneventfully.

Scope: `check_write`, `check_approach`, `check_scan`, `park_plan` as pure functions
over a hand-built `InstrumentModel` (see `tests/_im_fixtures_guard.py` -- built
directly from the section-10 dataclasses, not through T1's `load_model`/config
loading), plus door-level integration tests (REST `/api/litho/write`, FLEX
`startWrite`, and -- per spec item 9 -- `/api/design/op set_props`, the door that
lets `voltage_end`/`v_profile` reach an object in the first place) asserting the
guard is actually wired into every one of them.

ASSUMPTIONS (section 10 leaves these implicit; documented per the brief rather than
guessed silently):

  A1. `check_write`'s per-object voltage checks (window, DAC rail) and the
      amp_gain checks are independent axes: each test below varies exactly one
      thing and holds the model otherwise "obviously fine", so this file does not
      assert an inter-check PRECEDENCE order except where the brief explicitly
      names one (amp_gain checked ahead of nothing in particular here).
  A2. `object_ids` for `check_write` is the explicit id list (LithoObject.name,
      confirmed against litho_cmds.select_objects's `by_name` lookup and
      `getPattern`'s `"id": obj.name`). Passing the string `"all"` must NOT be
      silently treated as "every object" -- section 10 types it as `list[str]` and
      cockpit callers are required to send the explicit on-screen id list (section
      4). This file only asserts it does not succeed as a select-all; it does not
      pin the exact exception, since section 10 does not name one for this case.
  A3. `FreeAir.mode` records which mode ("contact" | "ac") the reading was taken
      in; `check_approach`'s contact rule requires a free-air record whose mode is
      "contact", and AC's rule requires "ac". (`check_write` has no free-air age or
      mode rule: the hardware run reads free air before every object.) This is not stated
      explicitly in section 10 but follows from `ac_setpoint_fraction_band` being
      meaningless against a contact deflection baseline and vice versa.
  A4. "taken not withdrawn" (section 2's free-air rule) is tested as refused with
      `free_air_stale` -- section 10 does not add a distinct reason for this case,
      and `free_air_missing` reads as "no reading at all" rather than "an
      untrustworthy one", so `free_air_stale` is the closer fit of the two listed
      reasons. Flagged for the implementer to confirm or correct.
  A5. `GuardRefused`'s JSON-RPC code is -32050 (section 10, updated after the
      original -32030 draft collided with `CONTROL_REQUIRED` in
      `service/control.py` / `flex_dispatch.py` -- flagged in an earlier pass of
      this file and fixed in the spec).
  A6. `park_plan`'s `.z_channel` is asserted equal to `model.z_out_channel`'s
      value (not a separately-computed channel), matching section 10's one-line
      description ("cypher -> z_channel Height ... mfp3d -> Output.Z").
  A7. The scan-frame integration test's corner math (45-degree rotation) is
      computed independently in this file (numpy), not copied from any
      implementation, so it exercises `check_scan`'s actual geometry rather than
      restating it.
  A8. `fault` (instrument-model 40ace53) is typed `object | None` -- section 10
      gives no concrete shape, so these tests pass an arbitrary truthy sentinel
      (a string, `"park_unverified"`) and assert only that ANY truthy value
      refuses with `fault_latched` ahead of every other rule; they do not assume
      a dataclass, exception instance, or any particular attribute on it. Each
      fault test is paired with a SEPARATE, independently-refusing violation (a
      missing amp_gain, a stale free-air, an out-of-range scan) so a guard that
      checks `fault` last -- rather than first -- fails the test with the OTHER
      reason instead of `fault_latched`.
  A9. `check_write` with an EMPTY `object_ids` list refuses `unknown_object`.
      Section 10 does not name a distinct reason for "nothing was selected"; an
      empty selection matches no id in the pattern, and `unknown_object` reads
      closer than silently succeeding as a no-op write. Flagged for the
      implementer to confirm or correct.
  A10. The REST-door test proving `setpoint = hub.free_air.value_v +
      setpoint_relative` (instrument-model 12a0593/672a8b1) assumes the TWIN's
      shipped instrument-model defaults for `contact_setpoint_band_v` ([0.0,
      1.0]) and `tip_amp_gain` (25.0) match section 10's example
      `instrument.toml` and the rig's known amplifier gain (CLAUDE.md / memory:
      "Tip-bias amplifier gain = 25"). Section 1 says only that "the twin
      backend ships its own defaults", not their exact values. If wave 2's twin
      defaults differ, the test's literal numbers need adjusting. NOTE (per
      spec 672a8b1): the band applies to `setpoint - free_air.value_v`, and on
      the REST door that reduces to exactly `setpoint_relative` -- so
      `free_air.value_v` CANCELS OUT of the band check and pass/refuse
      behaviour across different `free_air` values proves nothing about whether
      it was read. This test instead asserts the applied ABSOLUTE value via
      `hub.last_write_params["setpoint"]` -- see assumption A11.
  A11. `hub.last_write_params["setpoint"]` (section 10) is not populated by
      either round-A implementation branch that existed when this test was
      written (`im-impl-A1`: model + guard; `im-impl-A2`: + twin cypher
      profile) -- that population belongs to round A's third slice, "records"
      (section 9), not yet landed anywhere. This test targets the documented
      section-10 key directly rather than either branch's current partial
      state; see the test's own docstring for the full reasoning, including why
      "the worker's received setpoint" (the brief's suggested fallback) isn't
      available either on the twin path.
"""

from __future__ import annotations

import json
import math
import socket
import time

import numpy as np
import pytest
from afm_litho.service.command_guard import FreeAir, GuardRefused

from afm_litho.service import command_guard
from afm_litho.service.commands.errors import CommandError

from _im_fixtures_guard import (
    chiral_object,
    fill_object,
    make_model,
    make_pattern,
    malformed_chiral_object,
    plain_object,
    profiled_object,
    profiled_ramped_object,
    ramped_object,
)

# S2 (instrument-model spec item 10): a module-level `importorskip("zmq")` used to
# skip this WHOLE file -- including every pure check_write/check_approach/check_scan
# unit test above, which has nothing to do with FLEX -- on a box without the
# optional [flex] extra. Import it softly instead, and skip only the FLEX-socket
# fixtures (`_provider`, `cmd`) that actually need it; `test_flex_start_write_
# refused_by_guard` is skipped transitively through its `cmd` fixture.
try:
    import zmq
except ImportError:                              # pragma: no cover - optional extra
    zmq = None

NOW = 1_700_000_000.0


# ---------------------------------------------------- door-integration fixtures --
# Same construction as tests/conftest.py (`client`) and tests/test_flex_litho.py
# (`hub`/`provider`/`cmd`) -- kept local and module-scoped rather than imported, so
# this file has no import-time dependency on those modules' fixture names changing.

def _free_port() -> int:
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


@pytest.fixture(scope="module", autouse=True)
def _flex_provider_off():
    import os
    prev = os.environ.get("AFM_LITHO_FLEX")
    os.environ["AFM_LITHO_FLEX"] = "0"
    yield
    if prev is None:
        os.environ.pop("AFM_LITHO_FLEX", None)
    else:
        os.environ["AFM_LITHO_FLEX"] = prev


@pytest.fixture(scope="module")
def client(_flex_provider_off):
    from fastapi.testclient import TestClient

    from afm_litho.service.app import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


@pytest.fixture(scope="module")
def _provider(hub):
    if zmq is None:
        pytest.skip("pyzmq is the optional [flex] extra")
    from afm_litho.service import flex_provider as fp
    prov = fp.FlexProvider(hub, bind="127.0.0.1", port=_free_port(),
                           log=lambda msg: None)
    prov.start()
    yield prov
    prov.stop()


@pytest.fixture(scope="module")
def cmd(_provider):
    c = _Client(_provider.bound["command"])
    yield c
    c.close()


# ---------------------------------------------------------------- GuardRefused --

def test_guard_refused_is_a_command_error_with_the_stable_code():
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_write(
            make_model(tip_amp_gain=None), make_pattern(plain_object("a")),
            ["a"], 0.2, None, FreeAir(0.0, NOW, "contact", True), now=NOW)
    assert isinstance(ei.value, CommandError)
    assert ei.value.code == -32050
    assert ei.value.reason == "amp_gain_missing"


# ------------------------------------------------------------- fault (refuses first) --
# instrument-model 40ace53: every check_* takes keyword `fault: object | None = None`
# (callers pass `hub.fault`) and refuses FIRST with `fault_latched` when it is
# truthy -- ahead of every other check. Each case below is DELIBERATELY paired
# with a second, independently-refusing violation, so a guard that checks `fault`
# LAST (rather than first) would report that OTHER reason and fail the test --
# an otherwise-valid call would pass either ordering and prove nothing.

def test_check_write_fault_latched_refused_first():
    # paired violation: amp_gain=None, which ALONE gives amp_gain_missing
    model = make_model(tip_amp_gain=25.0)
    pattern = make_pattern(plain_object("a", voltage=5.0))
    free_air = FreeAir(0.0, NOW, "contact", True)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_write(model, pattern, ["a"], 0.2, None, free_air,
                                  now=NOW, fault="park_unverified")
    assert ei.value.reason == "fault_latched"


def test_check_approach_fault_latched_refused_first():
    # paired violation: a stale free_air, which ALONE gives free_air_stale
    model = make_model(free_air_max_age_s=30)
    stale = FreeAir(0.0, NOW - 3600.0, "contact", True)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_approach(model, "contact", 0.2, stale, now=NOW,
                                     fault="park_unverified")
    assert ei.value.reason == "fault_latched"


def test_check_scan_fault_latched_refused_first():
    # paired violation: an out-of-range frame, which ALONE gives scan_out_of_range
    model = make_model(xy_range_um=30.0)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_scan(model, 30.0, 10.0, 0.0, 0.0,
                                 fault="park_unverified")
    assert ei.value.reason == "fault_latched"


# --------------------------------------------------------------- amp_gain rules --

def test_check_write_amp_gain_none_is_amp_gain_missing():
    model = make_model(tip_amp_gain=25.0)
    pattern = make_pattern(plain_object("a", voltage=5.0))
    free_air = FreeAir(0.0, NOW, "contact", True)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_write(model, pattern, ["a"], 0.2, None, free_air, now=NOW)
    assert ei.value.reason == "amp_gain_missing"


def test_check_write_amp_gain_mismatch():
    model = make_model(tip_amp_gain=25.0)
    pattern = make_pattern(plain_object("a", voltage=5.0))
    free_air = FreeAir(0.0, NOW, "contact", True)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_write(model, pattern, ["a"], 0.2, 1.0, free_air, now=NOW)
    assert ei.value.reason == "amp_gain_mismatch"


def test_check_write_amp_gain_match_passes_that_check():
    model = make_model(tip_amp_gain=25.0)
    pattern = make_pattern(plain_object("a", voltage=5.0))
    free_air = FreeAir(0.0, NOW, "contact", True)
    # a correct amp_gain plus an otherwise clean write goes all the way through
    assert float(model.tip_amp_gain.value) == 25.0
    assert command_guard.check_write(model, pattern, ["a"], 0.2, 25.0, free_air,
                                     now=NOW) is None


# ----------------------------------------------------------- tip voltage window --

def test_tip_voltage_window_object_at_max_passes():
    model = make_model(tip_voltage_window_v=(-8.0, 30.0))
    pattern = make_pattern(plain_object("a", voltage=30.0))
    free_air = FreeAir(0.0, NOW, "contact", True)
    assert set(command_guard.commanded_voltages(pattern.objects[0])) == {30.0}  # the limit
    assert command_guard.check_write(model, pattern, ["a"], 0.2, 25.0, free_air,
                                     now=NOW) is None


def test_tip_voltage_window_object_over_max_refused():
    model = make_model(tip_voltage_window_v=(-8.0, 30.0))
    pattern = make_pattern(plain_object("a", voltage=30.5))
    free_air = FreeAir(0.0, NOW, "contact", True)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_write(model, pattern, ["a"], 0.2, 25.0, free_air, now=NOW)
    assert ei.value.reason == "tip_voltage_out_of_window"


def test_tip_voltage_window_chiral_within_window_passes():
    # v0=26.5, v_k=3.5 -> commanded voltages 23.0 and 30.0, both inside [-8, 30]
    model = make_model(tip_voltage_window_v=(-8.0, 30.0))
    pattern = make_pattern(chiral_object("wire", v0=26.5, v_k=3.5))
    free_air = FreeAir(0.0, NOW, "contact", True)
    assert {23.0, 30.0} <= set(command_guard.commanded_voltages(pattern.objects[0]))
    assert command_guard.check_write(model, pattern, ["wire"], 0.2, 25.0, free_air,
                                     now=NOW) is None


def test_tip_voltage_window_chiral_over_window_refused():
    # v0=30, v_k=0.5 -> commanded voltages 29.5 and 30.5; 30.5 is out of window
    model = make_model(tip_voltage_window_v=(-8.0, 30.0))
    pattern = make_pattern(chiral_object("wire", v0=30.0, v_k=0.5))
    free_air = FreeAir(0.0, NOW, "contact", True)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_write(model, pattern, ["wire"], 0.2, 25.0, free_air,
                                  now=NOW)
    assert ei.value.reason == "tip_voltage_out_of_window"


def test_tip_voltage_window_fill_at_min_passes():
    model = make_model(tip_voltage_window_v=(-8.0, 30.0))
    pattern = make_pattern(fill_object("region", voltage=-8.0))
    free_air = FreeAir(0.0, NOW, "contact", True)
    assert set(command_guard.commanded_voltages(pattern.objects[0])) == {-8.0}  # the limit
    assert command_guard.check_write(model, pattern, ["region"], 0.2, 25.0, free_air,
                                     now=NOW) is None


def test_tip_voltage_window_fill_under_min_refused():
    model = make_model(tip_voltage_window_v=(-8.0, 30.0))
    pattern = make_pattern(fill_object("region", voltage=-8.5))
    free_air = FreeAir(0.0, NOW, "contact", True)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_write(model, pattern, ["region"], 0.2, 25.0, free_air,
                                  now=NOW)
    assert ei.value.reason == "tip_voltage_out_of_window"


# ------------------------------------------------- voltage_end / v_profile / chiral --
# instrument-model spec item 9 (A1 review + T2, f7ec2b0): the guard used to derive
# commanded voltages from `voltage` and chiral `v0 +/- v_k` only -- `compile_object`
# also honours `voltage_end` (linear ramp) and `v_profile` (per-point bias), both
# settable via `/api/design/op set_props` with no range check, so `voltage=5,
# voltage_end=300` passed the guard and reached the DAC at 12 V. MUST-FIX: the guard
# enumerates every voltage source compile_object emits.

def test_ramp_end_over_window_refused():
    # voltage=5 (in window), voltage_end=300 (WAY over) -- both ends are commanded
    model = make_model(tip_voltage_window_v=(-8.0, 30.0))
    pattern = make_pattern(ramped_object("ramp", voltage=5.0, voltage_end=300.0))
    free_air = FreeAir(0.0, NOW, "contact", True)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_write(model, pattern, ["ramp"], 0.2, 25.0, free_air,
                                  now=NOW)
    assert ei.value.reason == "tip_voltage_out_of_window"


def test_ramp_end_within_window_passes():
    model = make_model(tip_voltage_window_v=(-8.0, 30.0))
    pattern = make_pattern(ramped_object("ramp", voltage=5.0, voltage_end=30.0))
    free_air = FreeAir(0.0, NOW, "contact", True)
    assert set(command_guard.commanded_voltages(pattern.objects[0])) == {5.0, 30.0}
    assert command_guard.check_write(model, pattern, ["ramp"], 0.2, 25.0, free_air,
                                     now=NOW) is None


def test_ramp_end_over_dac_rail_refused():
    # window wide enough that 30 V is inside it; amp_gain 1 -> |30/1| = 30 > 10,
    # same as the plain-voltage dac-rail case but tripped via voltage_end instead
    model = make_model(tip_voltage_window_v=(-40.0, 40.0), dac_rail_v=10.0,
                       tip_amp_gain=1.0)
    pattern = make_pattern(ramped_object("ramp", voltage=5.0, voltage_end=30.0))
    free_air = FreeAir(0.0, NOW, "contact", True)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_write(model, pattern, ["ramp"], 0.2, 1.0, free_air,
                                  now=NOW)
    assert ei.value.reason == "dac_rail_exceeded"


def test_v_profile_sample_over_window_refused():
    model = make_model(tip_voltage_window_v=(-8.0, 30.0))
    pattern = make_pattern(profiled_object("wavy", v_profile=[5.0, 250.0, 5.0]))
    free_air = FreeAir(0.0, NOW, "contact", True)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_write(model, pattern, ["wavy"], 0.2, 25.0, free_air,
                                  now=NOW)
    assert ei.value.reason == "tip_voltage_out_of_window"


def test_v_profile_within_window_passes():
    model = make_model(tip_voltage_window_v=(-8.0, 30.0))
    pattern = make_pattern(profiled_object("wavy", v_profile=[5.0, 10.0, 15.0]))
    free_air = FreeAir(0.0, NOW, "contact", True)
    assert {5.0, 15.0} <= set(command_guard.commanded_voltages(pattern.objects[0]))
    assert command_guard.check_write(model, pattern, ["wavy"], 0.2, 25.0, free_air,
                                     now=NOW) is None


# ------------------------------------------- M1: benign v_profile masks voltage_end --
# Adversarial review finding M1 / spec 13.1 item 9. `commanded_voltages` used to
# enumerate `voltage`/`voltage_end` ONLY in the `v_profile`-absent branch, so an
# object with a benign `v_profile` and an out-of-window `voltage_end` slipped past
# BOTH check_write and the set_props door -- yet `compiler.expand` strips
# `v_profile` from the fill strokes it emits, which then linspace
# `voltage`->`voltage_end`, so the out-of-window value reached the DAC. These tests
# pin the exact repro (voltage=5, voltage_end=300, v_profile=[1,2]).

def test_benign_v_profile_with_ramp_end_over_window_refused():
    """The exact M1 repro: a benign per-point profile [1, 2] (well inside the
    window) alongside `voltage_end=300` (WAY over). The window check must refuse
    on the enumerated `voltage_end`, not be fooled by the benign profile."""
    model = make_model(tip_voltage_window_v=(-8.0, 30.0))
    pattern = make_pattern(profiled_ramped_object(
        "electrode", voltage=5.0, voltage_end=300.0, v_profile=[1.0, 2.0]))
    free_air = FreeAir(0.0, NOW, "contact", True)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_write(model, pattern, ["electrode"], 0.2, 25.0,
                                  free_air, now=NOW)
    assert ei.value.reason == "tip_voltage_out_of_window"


def test_benign_v_profile_with_ramp_end_over_dac_rail_refused():
    """The DAC-rail variant: a window WIDE enough to allow 300 V at the tip, so
    the window check would pass -- but `|300 / 25| = 12 V > 10 V` rail. A benign
    `v_profile` must not hide the ramp end from the DAC-rail check either."""
    model = make_model(tip_voltage_window_v=(-1000.0, 1000.0), dac_rail_v=10.0,
                       tip_amp_gain=25.0)
    pattern = make_pattern(profiled_ramped_object(
        "electrode", voltage=5.0, voltage_end=300.0, v_profile=[1.0, 2.0]))
    free_air = FreeAir(0.0, NOW, "contact", True)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_write(model, pattern, ["electrode"], 0.2, 25.0,
                                  free_air, now=NOW)
    assert ei.value.reason == "dac_rail_exceeded"


@pytest.mark.parametrize("obj_builder", [
    lambda: plain_object("a", voltage=17.0),
    lambda: ramped_object("a", voltage=5.0, voltage_end=300.0),
    lambda: profiled_object("a", v_profile=[5.0, 250.0, 5.0]),
    lambda: profiled_ramped_object("a", voltage=5.0, voltage_end=300.0,
                                   v_profile=[1.0, 2.0]),
    lambda: profiled_ramped_object("a", voltage=-40.0, voltage_end=12.0,
                                   v_profile=[2.0, 3.0]),
    lambda: chiral_object("a", v0=10.0, v_k=7.0),
    lambda: fill_object("a", voltage=22.0),
])
def test_commanded_voltages_is_a_superset_of_every_compiled_voltage(obj_builder):
    """Property: for ANY object, the max magnitude the compiler actually commands
    over the expanded strokes is bounded by `max(|commanded_voltages(obj)|)`
    (spec 13.1 item 9 -- the guard's enumerated set must be a safe SUPERSET of
    whatever reaches the DAC, so no compiled point can exceed what the guard saw).
    Computed against the real `compiler.expand` + `compile_object`, not a
    restatement of the guard's own arithmetic."""
    from afm_litho.litho.compiler import compile_object, expand

    obj = obj_builder()
    guard_max = max(abs(v) for v in command_guard.commanded_voltages(obj))
    compiled_max = 0.0
    for stroke in expand(obj):
        v = compile_object(stroke, x_sens=1.0, y_sens=1.0)["v"]
        if len(v):
            compiled_max = max(compiled_max, float(np.max(np.abs(v))))
    assert compiled_max <= guard_max + 1e-9, (
        f"compiler commanded |{compiled_max}| V but the guard only enumerated "
        f"up to |{guard_max}| V for {obj.name!r}")


def test_malformed_chiral_v0_none_refuses_not_typeerror():
    # spec item 9: "Malformed chiral v0 must refuse, not raise" -- asserting
    # GuardRefused specifically (not the broader (GuardRefused, TypeError,
    # ValueError) tuple used elsewhere in this file for genuinely-unspecified
    # cases) because the spec text is explicit that this one must NOT raise a
    # bare TypeError.
    model = make_model()
    pattern = make_pattern(malformed_chiral_object("wire", v0=None, v_k=3.5))
    free_air = FreeAir(0.0, NOW, "contact", True)
    with pytest.raises(GuardRefused):
        command_guard.check_write(model, pattern, ["wire"], 0.2, 25.0, free_air,
                                  now=NOW)


# ---------------------------------------------------------------------- dac rail --

def test_dac_rail_exceeded_with_gain_one():
    # window wide enough that 30 V is inside it; amp_gain 1 -> |30/1| = 30 > 10
    model = make_model(tip_voltage_window_v=(-40.0, 40.0), dac_rail_v=10.0,
                       tip_amp_gain=1.0)
    pattern = make_pattern(plain_object("a", voltage=30.0))
    free_air = FreeAir(0.0, NOW, "contact", True)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_write(model, pattern, ["a"], 0.2, 1.0, free_air, now=NOW)
    assert ei.value.reason == "dac_rail_exceeded"


def test_dac_rail_passes_with_gain_twentyfive():
    # |30/25| = 1.2 V <= 10 V rail
    model = make_model(tip_voltage_window_v=(-40.0, 40.0), dac_rail_v=10.0,
                       tip_amp_gain=25.0)
    pattern = make_pattern(plain_object("a", voltage=30.0))
    free_air = FreeAir(0.0, NOW, "contact", True)
    assert set(command_guard.commanded_voltages(pattern.objects[0])) == {30.0}
    assert command_guard.check_write(model, pattern, ["a"], 0.2, 25.0, free_air,
                                     now=NOW) is None
    # the same gain refuses once the commanded voltage crosses rail * gain (250 V)
    too_high = make_pattern(plain_object("a", voltage=260.0))
    wide = make_model(tip_voltage_window_v=(-400.0, 400.0), dac_rail_v=10.0,
                      tip_amp_gain=25.0)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_write(wide, too_high, ["a"], 0.2, 25.0, free_air, now=NOW)
    assert ei.value.reason == "dac_rail_exceeded"


# ---------------------------------------------------------------- free-air rules --

@pytest.mark.parametrize("free_air", [
    None,                                              # no baseline on record
    FreeAir(0.0, NOW - 31.0, "contact", True),         # older than free_air_max_age_s
    FreeAir(0.0, NOW - 1.0, "contact", False),         # not taken withdrawn
    FreeAir(0.0, NOW - 1.0, "ac", True),               # taken in another mode
], ids=["none", "older_than_max_age", "not_withdrawn", "other_mode"])
def test_check_write_does_not_refuse_over_the_free_air_baseline(free_air):
    """docs/PER_OBJECT_FREE_AIR_SPEC.md "Start-of-write free-air age check dropped":
    the hardware run reads free air before every object, so the write guard no longer
    refuses a missing, old, non-withdrawn or other-mode baseline. The APPROACH guard
    still does (below)."""
    model = make_model(free_air_max_age_s=30)
    pattern = make_pattern(plain_object("a", voltage=5.0))
    assert command_guard.check_write(model, pattern, ["a"], 0.2, 25.0, free_air,
                                     now=NOW) is None
    if free_air is not None:
        with pytest.raises(GuardRefused) as ei:
            command_guard.check_approach(model, "contact", 0.2, free_air, now=NOW)
        assert ei.value.reason == "free_air_stale"


def test_check_write_setpoint_out_of_band_low():
    model = make_model(contact_setpoint_band_v=(0.0, 1.0))
    pattern = make_pattern(plain_object("a", voltage=5.0))
    free_air = FreeAir(0.0, NOW, "contact", True)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_write(model, pattern, ["a"], -0.1, 25.0, free_air,
                                  now=NOW)
    assert ei.value.reason == "setpoint_out_of_band"


def test_check_write_setpoint_out_of_band_high():
    model = make_model(contact_setpoint_band_v=(0.0, 1.0))
    pattern = make_pattern(plain_object("a", voltage=5.0))
    free_air = FreeAir(0.0, NOW, "contact", True)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_write(model, pattern, ["a"], 1.2, 25.0, free_air,
                                  now=NOW)
    assert ei.value.reason == "setpoint_out_of_band"


def test_check_write_setpoint_in_band_passes():
    model = make_model(contact_setpoint_band_v=(0.0, 1.0))
    pattern = make_pattern(plain_object("a", voltage=5.0))
    free_air = FreeAir(0.0, NOW, "contact", True)
    # the guard returns nothing; the out-of-band partners below prove the band is applied
    assert command_guard.check_write(model, pattern, ["a"], 0.2, 25.0, free_air,
                                     now=NOW) is None


@pytest.mark.parametrize("rel,refused", [(1.9, True), (-0.1, True), (0.2, False)],
                         ids=["above", "below", "inside"])
def test_check_write_relative_setpoint_without_baseline_is_band_checked(rel, refused):
    """PER_OBJECT_FREE_AIR_SPEC review fix M2: with no baseline on record, a REST
    relative setpoint is checked against contact_setpoint_band_v directly, the same
    result as setpoint - free_air with a baseline."""
    model = make_model(contact_setpoint_band_v=(0.0, 1.0))
    pattern = make_pattern(plain_object("a", voltage=5.0))
    with_baseline = FreeAir(-0.3, NOW, "contact", True)
    for free_air, sp in ((None, rel), (with_baseline, -0.3 + rel)):
        if refused:
            with pytest.raises(GuardRefused) as ei:
                command_guard.check_write(model, pattern, ["a"], sp, 25.0, free_air,
                                          now=NOW, setpoint_relative=rel)
            assert ei.value.reason == "setpoint_out_of_band"
        else:
            assert command_guard.check_write(model, pattern, ["a"], sp, 25.0, free_air,
                                             now=NOW, setpoint_relative=rel) is None


def test_check_write_absolute_setpoint_without_baseline_is_not_band_checked():
    """The FLEX door's absolute setpoint (no setpoint_relative) with no baseline adds
    no refusal: the band is relative to free air."""
    model = make_model(contact_setpoint_band_v=(0.0, 1.0))
    pattern = make_pattern(plain_object("a", voltage=5.0))
    assert 1.9 > model.contact_setpoint_band_v.value[1]   # outside the band, still allowed
    assert command_guard.check_write(model, pattern, ["a"], 1.9, 25.0, None,
                                     now=NOW) is None


# ------------------------------------------------------- object id / enable rules --

def test_check_write_unknown_object_id_refused():
    model = make_model()
    pattern = make_pattern(plain_object("a", voltage=5.0))
    free_air = FreeAir(0.0, NOW, "contact", True)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_write(model, pattern, ["does-not-exist"], 0.2, 25.0,
                                  free_air, now=NOW)
    assert ei.value.reason == "unknown_object"


def test_check_write_disabled_object_refused():
    model = make_model()
    pattern = make_pattern(plain_object("a", voltage=5.0, enabled=False))
    free_air = FreeAir(0.0, NOW, "contact", True)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_write(model, pattern, ["a"], 0.2, 25.0, free_air,
                                  now=NOW)
    assert ei.value.reason == "object_disabled"


def test_check_write_rejects_all_string_not_an_explicit_list():
    # see module docstring A2: "all" must not be silently treated as select-all.
    # The pattern's only object is named "wire-1" -- NOT a single character of
    # "all" -- so a buggy guard that iterates the STRING "all" char-by-char
    # ('a', 'l', 'l') can't accidentally match a real id and appear to pass for
    # the wrong reason (as it would if the object were named "a").
    model = make_model()
    pattern = make_pattern(plain_object("wire-1", voltage=5.0))
    free_air = FreeAir(0.0, NOW, "contact", True)
    with pytest.raises((GuardRefused, TypeError, ValueError)):
        command_guard.check_write(model, pattern, "all", 0.2, 25.0, free_air,
                                  now=NOW)


def test_check_write_empty_object_ids_refused_unknown_object():
    # see module docstring A9: an empty selection is refused, not silently
    # treated as a vacuous no-op write
    model = make_model()
    pattern = make_pattern(plain_object("a", voltage=5.0))
    free_air = FreeAir(0.0, NOW, "contact", True)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_write(model, pattern, [], 0.2, 25.0, free_air, now=NOW)
    assert ei.value.reason == "unknown_object"


# ------------------------------------------------------------------ check_approach --

def test_check_approach_contact_in_band_passes():
    model = make_model(contact_setpoint_band_v=(0.0, 1.0))
    free_air = FreeAir(0.0, NOW, "contact", True)
    # the guard returns nothing; the out-of-band test below proves the band is applied
    assert command_guard.check_approach(model, "contact", 0.2, free_air, now=NOW) is None


def test_check_approach_contact_out_of_band_refused():
    model = make_model(contact_setpoint_band_v=(0.0, 1.0))
    free_air = FreeAir(0.0, NOW, "contact", True)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_approach(model, "contact", 1.5, free_air, now=NOW)
    assert ei.value.reason == "setpoint_out_of_band"


def test_check_approach_ac_uses_fraction_band_passes():
    # free-air amplitude 2.0 V; setpoint 1.0 V -> fraction 0.5, inside [0.3, 0.95]
    model = make_model(ac_setpoint_fraction_band=(0.3, 0.95))
    free_air = FreeAir(2.0, NOW, "ac", True)
    # the guard returns nothing; the fraction-below-band test below proves the band is applied
    assert command_guard.check_approach(model, "ac", 1.0, free_air, now=NOW) is None


def test_check_approach_ac_fraction_below_band_refused():
    # fraction 0.25 < 0.3
    model = make_model(ac_setpoint_fraction_band=(0.3, 0.95))
    free_air = FreeAir(2.0, NOW, "ac", True)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_approach(model, "ac", 0.5, free_air, now=NOW)
    assert ei.value.reason == "setpoint_out_of_band"


def test_check_approach_missing_free_air_refused():
    model = make_model()
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_approach(model, "contact", 0.2, None, now=NOW)
    assert ei.value.reason == "free_air_missing"


def test_check_approach_stale_free_air_refused():
    model = make_model(free_air_max_age_s=30)
    stale = FreeAir(0.0, NOW - 60.0, "contact", True)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_approach(model, "contact", 0.2, stale, now=NOW)
    assert ei.value.reason == "free_air_stale"


# ----------------------------------------------------------------------- check_scan --

def test_check_scan_full_size_at_zero_offset_passes():
    model = make_model(xy_range_um=30.0)
    assert _rotated_corner_reach(30.0, 0.0, 0.0, 0.0) == (15.0, 15.0)   # exactly the edge
    assert command_guard.check_scan(model, 30.0, 0.0, 0.0, 0.0) is None


def test_check_scan_full_size_offset_by_one_refused():
    model = make_model(xy_range_um=30.0)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_scan(model, 30.0, 1.0, 0.0, 0.0)
    assert ei.value.reason == "scan_out_of_range"


def _rotated_corner_reach(size_um, x_offset_um, y_offset_um, angle_deg):
    """Independent corner-math check (assumption A7): the max |x| and max |y| any
    corner of a size x size square, centered at (x_offset, y_offset) and rotated by
    angle_deg about its own center, reaches."""
    half = size_um / 2.0
    corners = np.array([[sx * half, sy * half]
                        for sx in (-1, 1) for sy in (-1, 1)])
    theta = math.radians(angle_deg)
    rot = np.array([[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)]])
    rotated = corners @ rot.T
    rotated += np.array([x_offset_um, y_offset_um])
    return float(np.max(np.abs(rotated[:, 0]))), float(np.max(np.abs(rotated[:, 1])))


def test_check_scan_rotated_frame_out_of_range_refused():
    size_um, x_off, y_off, angle = 20.0, 8.0, 8.0, 45.0
    reach_x, reach_y = _rotated_corner_reach(size_um, x_off, y_off, angle)
    half_range = 30.0 / 2.0
    assert reach_x > half_range or reach_y > half_range, (
        "fixture assumption broken: this frame should leave the 30 um range")
    model = make_model(xy_range_um=30.0)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_scan(model, size_um, x_off, y_off, angle)
    assert ei.value.reason == "scan_out_of_range"


def test_check_scan_rotated_frame_within_range_passes():
    # same rotation, no offset: bounding reach is size/sqrt(2)*sqrt(2)=size*0.707*2
    # (14.14 um for a 20 um square), well inside a 30 um range
    size_um, angle = 20.0, 45.0
    reach_x, reach_y = _rotated_corner_reach(size_um, 0.0, 0.0, angle)
    half_range = 30.0 / 2.0
    assert reach_x <= half_range and reach_y <= half_range, (
        "fixture assumption broken: this frame should fit the 30 um range")
    model = make_model(xy_range_um=30.0)
    assert command_guard.check_scan(model, size_um, 0.0, 0.0, angle) is None


def test_check_scan_rotation_matters_axis_aligned_fits_rotated_does_not():
    """A guard that ignored `angle_deg` entirely -- just checking the unrotated
    half-extent against the range -- would wrongly PASS this: size 22 (half 11)
    fits inside a 15 um half-range on its own. Only accounting for the actual
    45-degree rotation (bbox half = 11 * sqrt(2) = 15.56 um) catches that the
    frame's rotated corners leave the range."""
    size_um, angle = 22.0, 45.0
    half = size_um / 2.0
    half_range = 30.0 / 2.0
    assert half <= half_range, (
        "fixture assumption broken: the UNROTATED half-extent should fit")
    reach_x, reach_y = _rotated_corner_reach(size_um, 0.0, 0.0, angle)
    assert reach_x > half_range or reach_y > half_range, (
        "fixture assumption broken: the ROTATED frame should leave the range")
    model = make_model(xy_range_um=30.0)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_scan(model, size_um, 0.0, 0.0, angle)
    assert ei.value.reason == "scan_out_of_range"


def test_check_scan_rotation_matters_naive_diagonal_would_wrongly_refuse():
    """The mirror case: a guard that approximates ANY rotation with the
    worst-case diagonal (half * sqrt(2), i.e. "45 degrees always") rather than
    the actual angle would wrongly REFUSE this. size 20 (half 10) offset
    (1, 0) at 30 degrees has a true rotated bbox reach of
    10 * (|cos 30| + |sin 30|) = 13.66 um, so total reach 1 + 13.66 = 14.66 um
    fits a 15 um half-range -- but the diagonal approximation
    (1 + 10 * sqrt(2) = 15.14 um) does not. Passing here proves the guard used
    the real angle, not a worst-case stand-in."""
    size_um, x_off, y_off, angle = 20.0, 1.0, 0.0, 30.0
    half = size_um / 2.0
    half_range = 30.0 / 2.0
    reach_x, reach_y = _rotated_corner_reach(size_um, x_off, y_off, angle)
    diagonal_reach = x_off + half * math.sqrt(2.0)
    assert reach_x <= half_range and reach_y <= half_range, (
        "fixture assumption broken: the TRUE rotated frame should fit")
    assert diagonal_reach > half_range, (
        "fixture assumption broken: the diagonal approximation should NOT fit -- "
        "otherwise this case can't distinguish the two implementations")
    model = make_model(xy_range_um=30.0)
    assert command_guard.check_scan(model, size_um, x_off, y_off, angle) is None


# ------------------------------------------------------------------------ park_plan --

def test_park_plan_cypher():
    model = make_model(platform="cypher", z_out_channel="Height",
                       z_verify_channels=("ZSensor", "Deflection"))
    plan = command_guard.park_plan(model)
    assert plan.z_channel == "Height"
    assert tuple(plan.verify_channels) == ("ZSensor", "Deflection")


def test_park_plan_mfp3d():
    model = make_model(platform="mfp3d", z_out_channel="Output.Z",
                       z_verify_channels=("ZSensor", "Deflection"))
    plan = command_guard.park_plan(model)
    assert plan.z_channel == "Output.Z"
    assert tuple(plan.verify_channels) == ("ZSensor", "Deflection")


# --------------------------------------------- check_design_voltages (item 19) --
# The design-door window + DAC-rail check (spec 13.1 item 19), a pure function over
# the same `commanded_voltages` enumeration `check_write` uses, shared by REST
# set_props and the FLEX setObjects/loadPattern/SVG doors. These tests exercise the
# function directly, on a list of objects (no pattern, no free-air, no amp_gain arg
# -- the amp gain is read from the model).

def test_design_voltages_within_window_passes():
    model = make_model(tip_voltage_window_v=(-10.0, 10.0))
    obj = plain_object("a", voltage=8.0)
    assert set(command_guard.commanded_voltages(obj)) == {8.0}
    assert command_guard.check_design_voltages(model, [obj]) is None


def test_design_voltages_over_window_refused_with_id_and_value():
    model = make_model(tip_voltage_window_v=(-10.0, 10.0))
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_design_voltages(model, [plain_object("a", voltage=30.0)])
    assert ei.value.reason == "tip_voltage_out_of_window"
    assert ei.value.data["object_id"] == "a"
    assert ei.value.data["value"] == 30.0


def test_design_voltages_none_model_is_noop():
    # No model loaded yet -- nothing to check against (matches _check_voltage_window).
    command_guard.check_design_voltages(None, [plain_object("a", voltage=1e6)])


def test_design_voltages_rail_with_gain_one_refused():
    # window wide enough that 12 V is inside it; amp_gain 1 -> |12/1| = 12 > 10 rail.
    model = make_model(tip_voltage_window_v=(-40.0, 40.0), dac_rail_v=10.0,
                       tip_amp_gain=1.0)
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_design_voltages(model, [plain_object("a", voltage=12.0)])
    assert ei.value.reason == "dac_rail_exceeded"
    assert ei.value.data["object_id"] == "a"


def test_design_voltages_rail_ok_with_gain_twentyfive():
    # |12/25| = 0.48 V <= 10 V rail, and 12 V inside the wide window -> passes.
    model = make_model(tip_voltage_window_v=(-40.0, 40.0), dac_rail_v=10.0,
                       tip_amp_gain=25.0)
    obj = plain_object("a", voltage=12.0)
    assert set(command_guard.commanded_voltages(obj)) == {12.0}
    assert command_guard.check_design_voltages(model, [obj]) is None
    wide = make_model(tip_voltage_window_v=(-400.0, 400.0), dac_rail_v=10.0,
                      tip_amp_gain=25.0)
    with pytest.raises(GuardRefused) as ei:     # same gain, past rail * gain = 250 V
        command_guard.check_design_voltages(wide, [plain_object("a", voltage=260.0)])
    assert ei.value.reason == "dac_rail_exceeded"


def test_design_voltages_v_profile_extremum_over_window_refused():
    model = make_model(tip_voltage_window_v=(-10.0, 10.0))
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_design_voltages(
            model, [profiled_object("p", [1.0, 2.0, 40.0])])
    assert ei.value.reason == "tip_voltage_out_of_window"
    assert ei.value.data["value"] == 40.0


def test_design_voltages_voltage_end_over_window_refused():
    # voltage inside, voltage_end outside -> the ramp end is a commanded voltage.
    model = make_model(tip_voltage_window_v=(-10.0, 10.0))
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_design_voltages(
            model, [ramped_object("r", voltage=5.0, voltage_end=300.0)])
    assert ei.value.reason == "tip_voltage_out_of_window"


def test_design_voltages_chiral_extremum_over_window_refused():
    # v0=8, v_k=5 -> extrema 3 and 13; 13 is outside [-10, 10].
    model = make_model(tip_voltage_window_v=(-10.0, 10.0))
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_design_voltages(
            model, [chiral_object("wire", v0=8.0, v_k=5.0)])
    assert ei.value.reason == "tip_voltage_out_of_window"


def test_design_voltages_chiral_within_window_passes():
    model = make_model(tip_voltage_window_v=(-10.0, 10.0))
    obj = chiral_object("wire", v0=5.0, v_k=3.0)
    assert {2.0, 8.0} <= set(command_guard.commanded_voltages(obj))
    assert command_guard.check_design_voltages(model, [obj]) is None


def test_design_voltages_malformed_chiral_v0_refuses_not_raises():
    model = make_model(tip_voltage_window_v=(-10.0, 10.0))
    with pytest.raises(GuardRefused) as ei:
        command_guard.check_design_voltages(model, [malformed_chiral_object("bad")])
    assert ei.value.reason == "tip_voltage_out_of_window"


# ------------------------------------------------------- every refusal's shape --

@pytest.mark.parametrize("reason,build", [
    ("amp_gain_missing", lambda: command_guard.check_write(
        make_model(), make_pattern(plain_object("a")), ["a"], 0.2, None,
        FreeAir(0.0, NOW, "contact", True), now=NOW)),
    ("amp_gain_mismatch", lambda: command_guard.check_write(
        make_model(tip_amp_gain=25.0), make_pattern(plain_object("a")), ["a"], 0.2,
        1.0, FreeAir(0.0, NOW, "contact", True), now=NOW)),
    ("tip_voltage_out_of_window", lambda: command_guard.check_write(
        make_model(), make_pattern(plain_object("a", voltage=99.0)), ["a"], 0.2,
        25.0, FreeAir(0.0, NOW, "contact", True), now=NOW)),
    ("dac_rail_exceeded", lambda: command_guard.check_write(
        make_model(tip_voltage_window_v=(-40.0, 40.0), tip_amp_gain=1.0),
        make_pattern(plain_object("a", voltage=30.0)), ["a"], 0.2, 1.0,
        FreeAir(0.0, NOW, "contact", True), now=NOW)),
    ("free_air_missing", lambda: command_guard.check_approach(
        make_model(), "contact", 0.2, None, now=NOW)),
    ("free_air_stale", lambda: command_guard.check_approach(
        make_model(), "contact", 0.2,
        FreeAir(0.0, NOW - 3600.0, "contact", True), now=NOW)),
    ("setpoint_out_of_band", lambda: command_guard.check_write(
        make_model(), make_pattern(plain_object("a")), ["a"], 5.0, 25.0,
        FreeAir(0.0, NOW, "contact", True), now=NOW)),
    ("unknown_object", lambda: command_guard.check_write(
        make_model(), make_pattern(plain_object("a")), ["nope"], 0.2, 25.0,
        FreeAir(0.0, NOW, "contact", True), now=NOW)),
    ("object_disabled", lambda: command_guard.check_write(
        make_model(), make_pattern(plain_object("a", enabled=False)), ["a"], 0.2,
        25.0, FreeAir(0.0, NOW, "contact", True), now=NOW)),
    ("scan_out_of_range", lambda: command_guard.check_scan(
        make_model(xy_range_um=30.0), 30.0, 10.0, 0.0, 0.0)),
    ("fault_latched", lambda: command_guard.check_write(
        make_model(), make_pattern(plain_object("a")), ["a"], 0.2, 25.0,
        FreeAir(0.0, NOW, "contact", True), now=NOW, fault="park_unverified")),
])
def test_every_refusal_carries_its_exact_reason_and_the_stable_code(reason, build):
    with pytest.raises(GuardRefused) as ei:
        build()
    assert ei.value.reason == reason
    assert isinstance(ei.value, CommandError)
    assert ei.value.code == -32050


# `model_incomplete` (section 10) is raised by `load_model` on the hardware
# backend (section 1), not by `check_write`/`check_approach`/`check_scan`
# themselves -- it belongs to T1's `load_model`/config-loading tests
# (tests/test_instrument_model.py), not here, so this file does not construct
# a GuardRefused for it directly (that would over-specify a constructor section
# 10 never fixes beyond `class GuardRefused(CommandError): reason: str`).


# --------------------------------------------------------------- door integration --

def _svg_two_objects() -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<image id="topo_bgd" x="0" y="0" width="4000" height="4000"/>'
        '<path d="M 0 0 L 4000 0"/>'
        '<path d="M 0 2000 L 4000 2000"/>'
        "</svg>"
    )


_GUARD_REASONS = {
    "model_incomplete", "amp_gain_missing", "amp_gain_mismatch",
    "tip_voltage_out_of_window", "dac_rail_exceeded", "free_air_missing",
    "free_air_stale", "setpoint_out_of_band", "unknown_object", "object_disabled",
    "scan_out_of_range", "fault_latched",
}


def _worker_alive(hub) -> bool:
    """True if a write worker appears to be running -- checked under BOTH the
    current attribute name (`hub.litho_thread`) and the section-10 name
    (`hub.litho_worker`), so this does not pass vacuously once wave 2 renames it
    (should-fix #3)."""
    thread = getattr(hub.host, "litho_thread", None)
    if thread is not None and getattr(thread, "is_alive", lambda: False)():
        return True
    worker = getattr(hub.host, "litho_worker", None)
    if worker is not None:
        is_alive = getattr(worker, "is_alive", None)
        if callable(is_alive) and is_alive():
            return True
    return False


def _assert_write_refused_and_untouched(hub, written_before) -> None:
    assert not _worker_alive(hub), (
        "a guard/param refusal must never start the write worker")
    assert getattr(hub.host, "written_objects", set()) == written_before, (
        "a guard/param refusal must never mark objects written")


def _load_two_object_design(client, hub, name: str) -> list[str]:
    r = client.post(f"/api/design/svg?name={name}",
                    content=_svg_two_objects(),
                    headers={"content-type": "image/svg+xml"})
    assert r.json().get("ok")
    hub.host.written_objects = set()
    return [o.name for o in hub.pattern.objects]


def test_rest_litho_write_with_no_setpoint_is_refused_and_touches_nothing(client, tmp_path):
    """A write must say where it presses: `setpoint` (volts, free air by default) or the
    cockpit's `setpoint_relative`. With neither it is refused as `setpoint_missing`."""
    from afm_litho.service.app import hub

    names = _load_two_object_design(client, hub, "guard-no-setpoint.svg")
    written_before = set(hub.host.written_objects)
    assert not _worker_alive(hub)

    resp = client.post("/api/litho/write", json={"objects": names, "amp_gain": 25})
    body = resp.json()
    assert body.get("ok") is not True, body
    assert body.get("reason") == "setpoint_missing", body

    _assert_write_refused_and_untouched(hub, written_before)


@pytest.mark.parametrize("baseline", ["none", "stale"])
def test_rest_litho_write_engaged_old_or_missing_baseline_is_accepted(client, tmp_path,
                                                                      baseline):
    """docs/PER_OBJECT_FREE_AIR_SPEC.md "Start-of-write free-air age check dropped": a
    write started ENGAGED with no baseline, or one older than free_air_max_age_s, is
    accepted (the hardware run reads free air before every object). The door takes no
    read while engaged, so the baseline on record is left as it was."""
    from afm_litho.service.app import hub

    names = _load_two_object_design(client, hub, f"guard-engaged-{baseline}.svg")
    fa = (None if baseline == "none" else
          FreeAir(value_v=-0.3, ts=time.time() - 10_000.0, mode="contact",
                  withdrawn=True))
    hub.host.free_air = fa
    hub.host.engaged = True
    try:
        resp = client.post(
            "/api/litho/write",
            json={"objects": names, "setpoint_relative": 0.1, "amp_gain": 25})
    finally:
        hub.host.engaged = False
    body = resp.json()
    assert body.get("ok") is True, body
    assert hub.host.free_air is fa, "an engaged write must not take a free-air read"
    _finish_or_abort_write(client, hub)


def test_rest_litho_write_engaged_no_baseline_relative_out_of_band_refused(client,
                                                                          tmp_path):
    """PER_OBJECT_FREE_AIR_SPEC review fix M2: engaged, no baseline, a relative
    setpoint outside the twin's contact band [0, 1] -> refused setpoint_out_of_band."""
    from afm_litho.service.app import hub

    names = _load_two_object_design(client, hub, "guard-engaged-out-of-band.svg")
    written_before = set(hub.host.written_objects)
    hub.host.free_air = None
    hub.host.engaged = True
    try:
        resp = client.post(
            "/api/litho/write",
            json={"objects": names, "setpoint_relative": 1.9, "amp_gain": 25})
    finally:
        hub.host.engaged = False
    body = resp.json()
    assert body.get("ok") is not True, body
    assert body.get("reason") == "setpoint_out_of_band", body
    _assert_write_refused_and_untouched(hub, written_before)


def _assert_fresh_write_baseline(hub, t0: float) -> None:
    ctx = hub._litho_write_ctx
    assert ctx["free_air_v"] is not None, ctx
    assert ctx["free_air_ts"] is not None and ctx["free_air_ts"] >= t0, ctx
    assert hub.host.free_air.mode == "contact" and hub.host.free_air.withdrawn


def test_rest_litho_write_withdrawn_no_baseline_establishes_free_air(client, tmp_path):
    """13.1 item 27 (a): withdrawn + no baseline -> the write door reads free air
    itself and proceeds; the record context carries the fresh read."""
    from afm_litho.service.app import hub

    names = _load_two_object_design(client, hub, "guard-e5-missing.svg")
    hub.host.engaged = False
    hub.host.free_air = None
    t0 = time.time()
    resp = client.post(
        "/api/litho/write",
        json={"objects": names, "setpoint_relative": 0.1, "amp_gain": 25})
    body = resp.json()
    assert body.get("ok") is True, body
    _assert_fresh_write_baseline(hub, t0)
    assert hub.host.last_write_params["setpoint"] == pytest.approx(
        hub._litho_write_ctx["free_air_v"] + 0.1)
    _finish_or_abort_write(client, hub)


def test_rest_litho_write_withdrawn_stale_baseline_rereads(client, tmp_path):
    """13.1 item 27 (b): withdrawn + stale baseline -> re-read, then proceed."""
    from afm_litho.service.app import hub

    names = _load_two_object_design(client, hub, "guard-e5-stale.svg")
    hub.host.engaged = False
    stale_ts = time.time() - 10_000.0
    hub.host.free_air = FreeAir(value_v=-0.3, ts=stale_ts, mode="contact",
                           withdrawn=True)
    t0 = time.time()
    resp = client.post(
        "/api/litho/write",
        json={"objects": names, "setpoint_relative": 0.1, "amp_gain": 25})
    body = resp.json()
    assert body.get("ok") is True, body
    _assert_fresh_write_baseline(hub, t0)
    _finish_or_abort_write(client, hub)


def test_mode_switch_while_withdrawn_reestablishes_contact_free_air(client):
    """13.1 item 27 (d): ac -> contact while withdrawn leaves a fresh contact
    baseline on the hub instead of None."""
    from afm_litho.service.app import hub

    hub.host.engaged = False
    prev_mode = getattr(hub.host, "imaging_mode", None)
    try:
        r = client.post("/api/mode", json={"mode": "ac"})
        assert r.json().get("ok") is True, r.json()
        t0 = time.time()
        r = client.post("/api/mode", json={"mode": "contact"})
        assert r.json().get("ok") is True, r.json()
        fa = hub.host.free_air
        assert fa is not None
        assert fa.mode == "contact" and fa.withdrawn and fa.ts >= t0
    finally:
        if prev_mode and prev_mode != "contact":
            client.post("/api/mode", json={"mode": prev_mode})


def test_rest_litho_write_all_key_is_unknown_object(client, tmp_path):
    """13.1 item 1: the REST door takes an EXPLICIT list of object ids; the `all`
    string / `{"all": true}` select-all convenience is a FLEX-door concept. On REST
    it must refuse `unknown_object` -- BEFORE the amp_gain check (which is why the
    body deliberately omits amp_gain: a pre-fix guard reached it and returned
    amp_gain_missing, masking the real problem)."""
    from afm_litho.service.app import hub

    # No free-air seed needed: the `all` refusal precedes the free-air/amp_gain guard.
    _load_two_object_design(client, hub, "guard-all-key.svg")
    written_before = set(hub.host.written_objects)

    # The `all` KEY is the REST select-all convenience the FLEX door owns; on REST it
    # is not a valid explicit object id. (An `objects: "all"` STRING is still expanded
    # to select-all on REST -- that is a separate, still-supported form.)
    resp = client.post("/api/litho/write",
                       json={"all": True, "setpoint_relative": 0.2})
    j = resp.json()
    assert j.get("ok") is not True, j
    assert j.get("reason") == "unknown_object", j

    _assert_write_refused_and_untouched(hub, written_before)


def _wait_write_idle(hub, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _worker_alive(hub):
            return True
        time.sleep(0.05)
    return False


def _finish_or_abort_write(client, hub) -> None:
    """A passing write in this file actually starts a real twin worker; the NEXT
    `/api/litho/write` POST in the same test would otherwise fail `busy` while it
    is still running. Bounded-poll for it to finish on its own first (short
    single-shape writes usually do, fast); if the budget runs out, abort it and
    poll again."""
    if _wait_write_idle(hub, timeout=5.0):
        return
    client.post("/api/litho/abort")
    assert _wait_write_idle(hub, timeout=25.0), (
        "write worker never went idle after abort")


def test_rest_litho_write_setpoint_relative_reads_hub_free_air(client, tmp_path):
    """Proves the conversion is real: `setpoint = hub.free_air.value_v +
    setpoint_relative`, computed before `check_write` (section 10) -- not merely
    accepting `setpoint_relative` as if it were already absolute.

    instrument-model 672a8b1 (section 2) PINS the contact band onto the RELATIVE
    value `setpoint - free_air.value_v`, never the absolute setpoint. On the REST
    door `setpoint = free_air.value_v + setpoint_relative`, so the relative value
    the guard actually checks is `(free_air.value_v + setpoint_relative) -
    free_air.value_v == setpoint_relative` exactly -- `free_air.value_v` CANCELS
    OUT of the band check by construction. A pass/refuse difference across
    free_air values therefore proves NOTHING about whether `hub.free_air` was
    read (an earlier version of this test got this backwards). What DOES prove
    it is the actual APPLIED absolute value, so this test asserts that directly
    and separately exercises the (now free_air-value-independent) band with the
    spec's own worked example.

    Observable used, and why (as instructed, checked against the two round-A
    implementation branches that exist at the time of writing):
    `hub.last_write_params["setpoint"]` (section 10: `hub.last_write_params:
    {"setpoint": float}`). On `origin/im-impl-A1` (model + guard) and
    `origin/im-impl-A2` (twin cypher profile), `hub.last_write_params` exists
    (`hub.py` initializes it to `{}`) but nothing yet WRITES to it -- that
    population is section 9 round A's THIRD slice, "records (routers/litho +
    hardware capture)", which had not landed on any branch yet. The brief's
    fallback -- "the worker's received setpoint" -- does not exist at all on the
    twin path either: `_twin_litho_worker`/`LithoRun` in both branches take no
    setpoint argument (`hub.litho_run = LithoRun(hub.inst, run_pattern, q,
    _transit_policy(), surf, frame=...)`). With neither option populated yet,
    this test targets the documented section-10 CONTRACT
    (`hub.last_write_params["setpoint"]`) rather than either branch's current
    partial state -- consistent with this whole file being wave-1 RED tests
    against the contract, not against any one implementation slice. Flagged as
    assumption A11: if the records slice ends up keying this dict differently
    (e.g. storing `setpoint_relative` instead, as
    `tests/test_write_records.py` on `im-impl-A2` does for a DIFFERENT purpose),
    this assertion needs updating to match.
    """
    from afm_litho.service.app import hub

    names = _load_two_object_design(client, hub, "guard-relative-setpoint.svg")

    # Worked example, spec 672a8b1: free_air -0.3, setpoint_relative 0.2 ->
    # relative +0.2 -> passes band [0, 1]; absolute applied = -0.3 + 0.2 = -0.1.
    hub.host.free_air = FreeAir(value_v=-0.3, ts=time.time(), mode="contact",
                           withdrawn=True)
    resp = client.post(
        "/api/litho/write",
        json={"objects": names, "setpoint_relative": 0.2, "amp_gain": 25})
    body = resp.json()
    assert body.get("ok") is True, body
    assert hub.host.last_write_params.get("setpoint") == pytest.approx(-0.1), (
        hub.host.last_write_params)
    _finish_or_abort_write(client, hub)

    # setpoint_relative -0.1 -> relative -0.1, below band [0, 1] -> refused,
    # regardless of free_air.value_v (it cancels out -- see docstring)
    hub.host.written_objects = set()
    hub.host.free_air = FreeAir(value_v=-0.3, ts=time.time(), mode="contact",
                           withdrawn=True)
    resp2 = client.post(
        "/api/litho/write",
        json={"objects": names, "setpoint_relative": -0.1, "amp_gain": 25})
    body2 = resp2.json()
    assert body2.get("reason") == "setpoint_out_of_band", body2

    # setpoint_relative 1.2 -> relative 1.2, above band [0, 1] -> refused
    hub.host.written_objects = set()
    resp3 = client.post(
        "/api/litho/write",
        json={"objects": names, "setpoint_relative": 1.2, "amp_gain": 25})
    body3 = resp3.json()
    assert body3.get("reason") == "setpoint_out_of_band", body3


def test_flex_start_write_refused_by_guard(hub, cmd):
    """Same refusal through the FLEX door (`startWrite`), same choke point (section
    2: "Consumers (all of them, both doors, no exceptions)"). Follows the dispatch
    pattern in test_flex_litho.py: real sockets, real hub, `acquireControl` first.
    FLEX keeps ABSOLUTE `setpoint` (unaffected by instrument-model 12a0593, which
    only changed the REST/cockpit door -- see the REST tests above)."""
    reply = cmd.send("acquireControl",
                     {"client": "test-guard", "deadman_s": 60})
    if "error" in reply:
        hub.control = None
        reply = cmd.send("acquireControl", {"client": "test-guard", "deadman_s": 60})
    token = reply["result"]["token"]
    try:
        load = cmd.send("loadPattern", {"token": token, "source": _svg_two_objects()})
        assert "error" not in load, load

        written_before = set(getattr(hub.host, "written_objects", set()))
        assert not _worker_alive(hub)

        reply = cmd.send("startWrite",
                         {"token": token, "objects": "all", "setpoint_reference": "absolute", "setpoint": 0.2,
                          "amp_gain": 1})
        assert "error" in reply, reply
        assert reply["error"]["code"] == -32050, reply
        data = reply["error"].get("data") or {}
        assert data.get("reason") in _GUARD_REASONS, reply

        _assert_write_refused_and_untouched(hub, written_before)
    finally:
        cur = getattr(hub, "control", None)
        if cur is not None:
            cmd.send("releaseControl", {"token": cur.token})


def test_flex_start_write_withdrawn_no_baseline_establishes_free_air(hub, cmd):
    """13.1 item 27 (e): the FLEX `startWrite` door behaves like REST -- withdrawn
    with no baseline, it reads free air itself and proceeds."""
    reply = cmd.send("acquireControl",
                     {"client": "test-guard-e5", "deadman_s": 60})
    if "error" in reply:
        hub.control = None
        reply = cmd.send("acquireControl",
                         {"client": "test-guard-e5", "deadman_s": 60})
    token = reply["result"]["token"]
    try:
        load = cmd.send("loadPattern", {"token": token, "source": _svg_two_objects()})
        assert "error" not in load, load
        hub.host.written_objects = set()
        hub.host.engaged = False
        hub.host.free_air = None
        # FLEX speaks ABSOLUTE setpoint: aim 0.1 V above the twin's free air.
        setpoint = float(hub.host.inst.read_free_air_deflection()) + 0.1
        t0 = time.time()
        reply = cmd.send("startWrite",
                         {"token": token, "objects": "all", "setpoint_reference": "absolute", "setpoint": setpoint,
                          "amp_gain": 25})
        assert "error" not in reply, reply
        _assert_fresh_write_baseline(hub, t0)
        assert _wait_write_idle(hub, timeout=30.0) or cmd.send(
            "abortWrite", {"token": token})
        _wait_write_idle(hub, timeout=25.0)
    finally:
        cur = getattr(hub, "control", None)
        if cur is not None:
            cmd.send("releaseControl", {"token": cur.token})


def test_rest_design_set_props_refuses_out_of_window_voltage_end(client, tmp_path):
    """instrument-model spec item 9 (A1 review + T2, f7ec2b0): "`voltage=5,
    voltage_end=300` passes the guard and reaches the DAC at 12 V. MUST-FIX ...
    the set_props door refuses out-of-window values at entry" -- i.e. NOT only at
    write time (the unit tests above), but also at `/api/design/op set_props`
    itself, so a bad value can never even be SAVED into the design.

    The exact response shape `/api/design/op` will use for this new refusal is
    not pinned by section 10 (that door's existing convention is a bare
    `{"error": str}`, unlike the write doors' `reason`-carrying bodies -- see
    `src/afm_litho/service/routers/design.py`'s many `return {"error": ...}`
    sites). This test therefore accepts EITHER a structured `reason` field or the
    string `"tip_voltage_out_of_window"` appearing in the error message, and
    pins the one thing section 10 DOES require: the object is left unchanged.
    """
    from afm_litho.service.app import hub

    _load_two_object_design(client, hub, "guard-set-props.svg")
    obj = hub.pattern.objects[0]
    voltage_before = obj.voltage
    voltage_end_before = obj.voltage_end

    resp = client.post("/api/design/op", json={
        "op": "set_props", "indices": [0], "voltage_end": 300.0})
    body = resp.json()
    payload = json.dumps(body)
    assert body.get("reason") == "tip_voltage_out_of_window" or (
        "tip_voltage_out_of_window" in payload), body

    assert hub.pattern.objects[0].voltage == voltage_before
    assert hub.pattern.objects[0].voltage_end == voltage_end_before, (
        "a refused set_props must leave the object unchanged")


def test_rest_design_set_props_refuses_out_of_window_v_profile(client, tmp_path):
    """Same door, tripped via `v_profile` instead of `voltage_end` (spec item 9
    names both as the guard's blind spots)."""
    from afm_litho.service.app import hub

    _load_two_object_design(client, hub, "guard-set-props-vp.svg")
    obj = hub.pattern.objects[0]
    v_profile_before = obj.v_profile

    resp = client.post("/api/design/op", json={
        "op": "set_props", "indices": [0], "v_profile": [5.0, 250.0, 5.0]})
    body = resp.json()
    payload = json.dumps(body)
    assert body.get("reason") == "tip_voltage_out_of_window" or (
        "tip_voltage_out_of_window" in payload), body

    assert hub.pattern.objects[0].v_profile == v_profile_before, (
        "a refused set_props must leave the object unchanged")


def test_rest_design_set_props_refuses_benign_profile_with_over_window_ramp_end(
        client, tmp_path):
    """The exact M1 repro at the set_props door (spec 13.1 item 9): setting a
    benign `v_profile=[1, 2]` AND `voltage_end=300` in one call must be refused.
    With the pre-fix guard the benign profile masked the ramp end, so this door
    SAVED an object that would then reach the DAC at 300 V / 12 V on the tip."""
    from afm_litho.service.app import hub

    _load_two_object_design(client, hub, "guard-set-props-m1.svg")
    obj = hub.pattern.objects[0]
    voltage_end_before = obj.voltage_end
    v_profile_before = obj.v_profile

    resp = client.post("/api/design/op", json={
        "op": "set_props", "indices": [0],
        "v_profile": [1.0, 2.0], "voltage_end": 300.0})
    body = resp.json()
    payload = json.dumps(body)
    assert body.get("reason") == "tip_voltage_out_of_window" or (
        "tip_voltage_out_of_window" in payload), body

    assert hub.pattern.objects[0].voltage_end == voltage_end_before, (
        "a refused set_props must leave the object unchanged")
    assert hub.pattern.objects[0].v_profile == v_profile_before, (
        "a refused set_props must leave the object unchanged")
