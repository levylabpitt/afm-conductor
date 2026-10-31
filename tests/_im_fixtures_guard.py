"""Small, distinctly-named fixture helpers for tests/test_command_guard.py (wave-1
slice T2, docs/INSTRUMENT_MODEL_SPEC.md sections 2 and 10).

Kept out of test_command_guard.py itself only so the test file reads as cases, not
plumbing. Builds `InstrumentModel` instances directly from the section-10 dataclasses
(`InstrumentModel`, `Field`, `FieldSource`) -- it does NOT depend on T1's
`load_model`/config-file loading, per the T2 brief.

Every field is tagged `FieldSource.config` here; nothing in command_guard's contract
(section 10) branches on WHICH source a field carries, only on the field's value, so
the exact source tag is not itself under test.
"""

from __future__ import annotations

import time

from afm_litho.litho.compiler import LithoObject, Pattern
from afm_litho.service.instrument_model import Field, FieldSource, InstrumentModel

# ---------------------------------------------------------------------------
# Wave-2 legacy-migration helpers (docs/INSTRUMENT_MODEL_SPEC.md section 13).
#
# The command guard (section 2) now refuses any write / approach that arrives
# without `amp_gain` or without a fresh, WITHDRAWN free-air baseline on the hub.
# Legacy tests that predate the guard are migrated by (a) passing `amp_gain` on
# every write body and (b) seeding `hub.free_air` before the engage. These two
# helpers are the single place that shape lives so a migrated test reads as its
# original case, not as guard plumbing.
# ---------------------------------------------------------------------------

# The one amp-gain a migrated write body carries. Equal to `model.tip_amp_gain`
# in `make_model` above and to the on-instrument tip-bias amplifier gain
# (memory: "Tip-bias amplifier gain = 25"), so `check_write` accepts it.
GUARD_WRITE_KW = {"amp_gain": 25.0}


def seed_free_air(hub, value_v=-0.2, mode="contact", withdrawn=True, now=None):
    """Seed `hub.free_air` with a fresh withdrawn free-air baseline so the
    approach guard's `free_air_missing` / `free_air_stale` checks pass and the
    write guard's setpoint band has a baseline.

    Mirrors `host/continuous.py`'s re-reference: a `FreeAir` taken WITHDRAWN
    with a current timestamp. `value_v` defaults to -0.2 V, in the Cypher's
    real free-air range this week (-0.16..-0.33 V, spec section 2), so a
    `setpoint_relative` of +0.2 lands near an absolute -0.0 V setpoint.

    Returns the seeded `FreeAir` so a caller that computes an absolute FLEX
    `setpoint = free_air.value_v + relative` can read the baseline back.
    """
    from afm_litho.service.command_guard import FreeAir

    fa = FreeAir(
        value_v=value_v,
        ts=time.time() if now is None else now,
        mode=mode,
        withdrawn=withdrawn,
    )
    hub.host.free_air = fa
    return fa


def field(value):
    return Field(value=value, source=FieldSource.config)


def make_model(
    *,
    platform="twin",
    z_out_channel="Output.Z",
    z_verify_channels=("ZSensor", "Deflection"),
    z_park_v=0.0,
    z_park_tol_v=0.05,
    dac_rail_v=10.0,
    tip_amp_gain=25.0,
    tip_voltage_window_v=(-8.0, 30.0),
    contact_setpoint_band_v=(0.0, 1.0),
    ac_setpoint_fraction_band=(0.3, 0.95),
    free_air_max_age_s=30,
    xy_range_um=30.0,
    free_air_abs_band_v=0.02,
    sum_floor_v=0.5,
    sum_ceiling_frac=0.9,
    z_rail_v=140.0,
    interlock_s=5.0,
    write_setpoint_relative_v=0.1,
) -> InstrumentModel:
    """A complete, valid InstrumentModel, every field overridable by keyword so each
    test only names what it is actually varying."""
    return InstrumentModel(
        platform=field(platform),
        z_out_channel=field(z_out_channel),
        z_verify_channels=field(z_verify_channels),
        z_park_v=field(z_park_v),
        z_park_tol_v=field(z_park_tol_v),
        dac_rail_v=field(dac_rail_v),
        tip_amp_gain=field(tip_amp_gain),
        tip_voltage_window_v=field(tip_voltage_window_v),
        contact_setpoint_band_v=field(contact_setpoint_band_v),
        ac_setpoint_fraction_band=field(ac_setpoint_fraction_band),
        free_air_max_age_s=field(free_air_max_age_s),
        xy_range_um=field(xy_range_um),
        free_air_abs_band_v=field(free_air_abs_band_v),
        sum_floor_v=field(sum_floor_v),
        sum_ceiling_frac=field(sum_ceiling_frac),
        z_rail_v=field(z_rail_v),
        interlock_s=field(interlock_s),
        write_setpoint_relative_v=field(write_setpoint_relative_v),
    )


def plain_object(name, voltage=8.0, *, enabled=True, kind="path") -> LithoObject:
    """A plain path/rect-style object: its commanded tip voltage is `voltage`
    (section 10: "object voltage")."""
    return LithoObject(
        name=name, kind=kind,
        points_um=[(0.0, 0.0), (1.0, 0.0)],
        voltage=voltage, enabled=enabled,
    )


def chiral_object(name, v0, v_k, *, enabled=True) -> LithoObject:
    """A chiral wire: commanded tip voltages are v0 - v_k and v0 + v_k (section 10),
    carried the way `litho.pens.chiral_profile` stamps them -- `profile["kind"] ==
    "chiral"`, `profile["v0"]`, `profile["v_k"]` -- with `voltage` mirroring v0 the
    way the real pen sets it (`pens.py` `_render_chiral`)."""
    return LithoObject(
        name=name, kind="path",
        points_um=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
        voltage=v0, enabled=enabled,
        profile={"kind": "chiral", "v0": float(v0), "v_k": float(v_k)},
    )


def malformed_chiral_object(name, *, v0=None, v_k=3.5, enabled=True) -> LithoObject:
    """instrument-model spec item 9 (A1 review + T2, f7ec2b0): "Malformed chiral
    v0 must refuse, not raise." Builds the profile WITHOUT `chiral_object`'s
    `float(...)` conversion (which would itself raise `TypeError` at fixture-
    build time, before `check_write` ever runs) so a genuinely malformed
    `v0=None` reaches the guard as-is."""
    return LithoObject(
        name=name, kind="path",
        points_um=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
        voltage=8.0, enabled=enabled,
        profile={"kind": "chiral", "v0": v0, "v_k": v_k},
    )


def ramped_object(name, voltage, voltage_end, *, enabled=True) -> LithoObject:
    """A linear-ramp object: `compile_object` linspaces `voltage` -> `voltage_end`
    (compiler.py `voltage_end`), so BOTH ends are commanded tip voltages the
    guard must check (instrument-model spec item 9)."""
    return LithoObject(
        name=name, kind="path",
        points_um=[(0.0, 0.0), (1.0, 0.0)],
        voltage=voltage, voltage_end=voltage_end, enabled=enabled,
    )


def profiled_object(name, v_profile, *, enabled=True) -> LithoObject:
    """An object carrying an arbitrary per-point bias profile (`v_profile`,
    compiler.py: "arbitrary bias profile sampled uniformly over the object's arc
    length"); every sample is a commanded tip voltage (instrument-model spec item
    9)."""
    return LithoObject(
        name=name, kind="path",
        points_um=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
        voltage=v_profile[0] if v_profile else 0.0,
        v_profile=list(v_profile), enabled=enabled,
    )


def profiled_ramped_object(name, *, voltage, voltage_end, v_profile,
                           kind="shape", fill_kind="directional",
                           enabled=True) -> LithoObject:
    """The M1 repro object (adversarial review finding M1, spec 13.1 item 9): a
    filled shape/electrode carrying a BENIGN `v_profile` AND an out-of-window
    `voltage`/`voltage_end`. `compile_object` uses the benign `v_profile` on a
    plain write, so a guard that enumerated `voltage`/`voltage_end` only in the
    `v_profile`-absent branch waved the object through -- yet `compiler.expand`
    strips `v_profile` from the fill strokes, which then linspace
    `voltage`->`voltage_end`, so the out-of-window `voltage_end` reaches the DAC.
    `commanded_voltages` must therefore enumerate every source unconditionally."""
    return LithoObject(
        name=name, kind=kind,
        points_um=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        fill_kind=fill_kind,
        voltage=voltage, voltage_end=voltage_end,
        v_profile=list(v_profile), enabled=enabled,
    )


def fill_object(name, voltage, *, enabled=True) -> LithoObject:
    """A filled region: fill strokes inherit the object's own `voltage` (no separate
    fill-voltage field on LithoObject -- see compiler.py `to_dict`/fill expansion)."""
    return LithoObject(
        name=name, kind="shape",
        points_um=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        fill_kind="directional",
        voltage=voltage, enabled=enabled,
    )


def make_pattern(*objects: LithoObject) -> Pattern:
    return Pattern(name="guard-test-pattern", objects=list(objects))
