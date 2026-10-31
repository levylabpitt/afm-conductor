"""Twin cypher profile (docs/INSTRUMENT_MODEL_SPEC.md Sec 6 + 10, wave-1 slice T3).

RED against today's `SoftwareARC`, which has no platform concept at all: loop 2's
Z output is hard-wired to the single "Z%Output"/"Output.Z" pair (see
`SoftwareARC.step`'s `match lp.output: case "Z%Output" | "Output.Z":`) and there is
no `AFM_LITHO_TWIN_PLATFORM` env handling, no `.platform`/`.is_cypher`/
`.z_out_channel` surface (mirroring `HardwareARC`'s, which already has it -- see
`src/afm_litho/hardware/hardware_arc.py`), no fire-and-forget `.ramp()` primitive,
and no `.faults` fault-injection object.

Contract under test (Sec 10):
  - env `AFM_LITHO_TWIN_PLATFORM=cypher|mfp3d`, default `mfp3d`.
  - cypher profile: loop-2 output `Height`; `Output.Z` ramps are inert; `ZSensor`
    follows `Height`. mfp3d keeps today's `Output.Z` behaviour.
  - fault injection: `twin.faults.ramp_noop = True` makes every ramp a no-op.
  - the model reports platform `"cypher"` from the twin.

Per the 2026-09-22 spec update (40ace53), this is now Sec 10's official twin
contract (names mirror `HardwareARC`'s -- hardware/hardware_arc.py):
`SoftwareARC.platform: str`, `.is_cypher: bool`, `.z_out_channel: str`;
`.ramp(channel: str, target_v: float) -> None` (fire-and-forget, the twin's
IR_SetRamp); `.read(channel: str) -> float`; `.command_log: list[tuple[str, float]]`
of every ramp issued; `.faults.ramp_noop: bool` (default False). `safe_park` drives
ramp/read/vendor_withdraw through this SAME object (`hub.hardware_arc` on both
backends); the twin's `vendor_withdraw` is a full retract that IGNORES
`ramp_noop` (exercised via the fake-bridge doubles in `tests/test_safe_park_model.py`,
not here -- this file stays scoped to `SoftwareARC` itself).
"""

from __future__ import annotations

import numpy as np
import pytest

from afm_litho.twin.arc import SoftwareARC
from afm_litho.twin.physics import Sample, World


def _world() -> World:
    return World(sample=Sample(height_map=np.zeros((8, 8)), extent=1e-5))


def _arc(monkeypatch, platform: str | None):
    if platform is None:
        monkeypatch.delenv("AFM_LITHO_TWIN_PLATFORM", raising=False)
    else:
        monkeypatch.setenv("AFM_LITHO_TWIN_PLATFORM", platform)
    return SoftwareARC(_world())


# --------------------------------------------------------------- platform id --

def test_default_platform_is_mfp3d_when_env_unset(monkeypatch):
    arc = _arc(monkeypatch, None)
    assert arc.platform == "mfp3d"
    assert arc.is_cypher is False
    assert arc.z_out_channel == "Height"


def test_env_var_selects_cypher_profile(monkeypatch):
    arc = _arc(monkeypatch, "cypher")
    assert arc.platform == "cypher"
    assert arc.is_cypher is True
    assert arc.z_out_channel == "Height"


def test_env_var_explicit_mfp3d_matches_default(monkeypatch):
    arc = _arc(monkeypatch, "mfp3d")
    assert arc.platform == "mfp3d"
    assert arc.is_cypher is False
    assert arc.z_out_channel == "Height"


def test_model_reports_platform_cypher_from_the_twin(monkeypatch):
    """"the model reports platform cypher from the twin" (T3 brief) -- the twin
    itself is the source of truth `load_model(platform=...)` is expected to read
    off of when the backend is the twin (Sec 1: platform source order is STARTUP,
    config; for the twin backend STARTUP is this env var)."""
    arc = _arc(monkeypatch, "cypher")
    assert arc.platform == "cypher"


# ---------------------------------------------------------- loop-2 Z identity --

def test_cypher_loop2_z_output_identity_is_height(monkeypatch):
    arc = _arc(monkeypatch, "cypher")
    arc.loops[2].output = arc.z_out_channel
    assert arc.loops[2].output == "Height"


def test_mfp3d_loop2_z_output_identity_is_output_z(monkeypatch):
    arc = _arc(monkeypatch, "mfp3d")
    arc.loops[2].output = arc.z_out_channel
    assert arc.loops[2].output == "Height"


# --------------------------------------------------------------- ramp behaviour --

def _settle(arc, blocks: int = 200, n: int = 50) -> None:
    for _ in range(blocks):
        arc.step(n)


def test_cypher_ramp_of_output_z_is_inert_on_zsensor(monkeypatch):
    """This is the incident: a Height-loop rig commanded through the wrong channel
    (Output.Z) must move nothing on ZSensor, however "successful" the commanded
    value looks."""
    arc = _arc(monkeypatch, "cypher")
    z0 = arc.sig["ZSensor"]
    arc.ramp("Output.Z", -5.0)
    _settle(arc)
    assert arc.sig["ZSensor"] == pytest.approx(z0, abs=1e-9)


def test_cypher_ramp_of_height_moves_zsensor(monkeypatch):
    arc = _arc(monkeypatch, "cypher")
    z0 = arc.sig["ZSensor"]
    arc.ramp("Height", -5.0)
    _settle(arc)
    assert arc.sig["ZSensor"] != pytest.approx(z0, abs=1e-9)


def test_mfp3d_ramp_of_output_z_moves_zsensor_today(monkeypatch):
    """mfp3d keeps today's single-channel behaviour."""
    arc = _arc(monkeypatch, "mfp3d")
    z0 = arc.sig["ZSensor"]
    arc.ramp("Output.Z", -5.0)
    _settle(arc)
    assert arc.sig["ZSensor"] != pytest.approx(z0, abs=1e-9)


def test_ramp_appends_channel_and_target_to_command_log(monkeypatch):
    arc = _arc(monkeypatch, "cypher")
    arc.ramp("Height", -5.0)
    assert ("Height", -5.0) in arc.command_log
    assert not any(ch == "Output.Z" for ch, _v in arc.command_log)


def test_read_reports_the_named_sensor_channel(monkeypatch):
    arc = _arc(monkeypatch, "cypher")
    arc.ramp("Height", -5.0)
    _settle(arc)
    assert arc.read("ZSensor") == pytest.approx(arc.sig["ZSensor"])


# --------------------------------------------------------------- fault injection --

def test_ramp_noop_fault_makes_every_ramp_inert(monkeypatch):
    arc = _arc(monkeypatch, "cypher")
    arc.faults.ramp_noop = True
    z0 = arc.sig["ZSensor"]
    arc.ramp("Height", -5.0)
    _settle(arc)
    assert arc.sig["ZSensor"] == pytest.approx(z0, abs=1e-9)
    # the command is still logged (it was issued) -- only its effect is suppressed,
    # so safe_park's "first failed verify" branch has something truthful to see.
    assert ("Height", -5.0) in arc.command_log


def test_ramp_noop_fault_defaults_false(monkeypatch):
    arc = _arc(monkeypatch, "cypher")
    assert arc.faults.ramp_noop is False


def test_vendor_withdraw_ignores_ramp_noop(monkeypatch):
    """"vendor_withdraw on the twin is a full retract that ignores ramp_noop"
    (Sec 10) -- the whole point of the fallback path is that it still works when
    the primary ramp is faulted."""
    arc = _arc(monkeypatch, "cypher")
    arc.faults.ramp_noop = True
    z0 = arc.sig["ZSensor"]
    arc.vendor_withdraw()
    _settle(arc)
    assert arc.sig["ZSensor"] != pytest.approx(z0, abs=1e-9)


def test_vendor_withdraw_is_recorded_in_the_command_log(monkeypatch):
    """SHOULD-FIX (review): `command_log` must show the vendor fallback too, not
    only ramps -- a test/implementer reading it back (as `safe_park`'s tests do)
    needs the full sequence of what was actually commanded, including the
    fallback. Sec 10 documents `command_log` as ramp tuples only; this extends
    it to also record `("vendor_withdraw", <target after the retract>)`, flagged
    here rather than silently assumed."""
    arc = _arc(monkeypatch, "cypher")
    arc.vendor_withdraw()
    assert any(entry[0] == "vendor_withdraw" for entry in arc.command_log)
