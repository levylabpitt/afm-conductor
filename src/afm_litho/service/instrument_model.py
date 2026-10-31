"""The instrument model: one explicit, source-tagged description of the rig that
every command guard and the cockpit read instead of a scattered pile of literals.

Truth document: docs/INSTRUMENT_MODEL_SPEC.md sections 1 and 10 (the interface
contract). Every field records WHERE its value came from
(`platform_default` < `startup_cal` < `config` < `override`), so a refusal or a
cockpit readout can say not just "25.0" but "why 25.0".

Config file: `runtime/instrument.toml` (path resolved by the caller -- normally
`AFM_LITHO_INSTRUMENT_CONFIG`, else `<runtime>/instrument.toml`). Missing on a
HARDWARE platform (`cypher` / `mfp3d`) -> `ModelIncomplete`: the service must start
REFUSING rather than driving real hardware from platform-default guesses. The twin
platform ships complete defaults and never needs the file.

Overrides persist to two places (`apply_override`): one line appended to
`<runtime>/instrument_overrides.jsonl` (the audit trail) and the `[override]` table
written back into the TOML itself (so a restart's `load_model` sees it too, without
depending on this process's memory).
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ..hardware import channels
from ..hardware.channels import Z_OUT
from ..hardware.free_air import FREE_AIR_ABS_BAND_V

# ------------------------------------------------------------------- vocabulary --

_HARDWARE_PLATFORMS = {"cypher", "mfp3d"}

# Per-platform defaults (section 1 table + section 10 baseline values). Only the
# fields that actually differ by platform are listed per-platform; the rest share
# one constant default below.
_PLATFORM_XY_RANGE_UM = {"cypher": (90.0, 90.0), "mfp3d": (90.0, 90.0),
                         "twin": (90.0, 90.0)}

_DEFAULT_Z_VERIFY_CHANNELS = (channels.Z_SENSOR, channels.DEFLECTION)
_DEFAULT_Z_PARK_V = 0.0
_DEFAULT_Z_PARK_TOL_V = 0.5
_DEFAULT_DAC_RAIL_V = 10.0

# The TWIN ships COMPLETE defaults (section 1: "does not need the file at all") --
# these are the only numbers not pinned by section 10's example instrument.toml,
# so they are chosen to MATCH that example (the rig's known values, CLAUDE.md /
# memory: "Tip-bias amplifier gain = 25").
_TWIN_TIP_AMP_GAIN = 25.0
_TWIN_TIP_VOLTAGE_WINDOW_V = (-8.0, 30.0)
_TWIN_CONTACT_SETPOINT_BAND_V = (0.0, 1.0)
_TWIN_AC_SETPOINT_FRACTION_BAND = (0.3, 0.95)
_TWIN_FREE_AIR_MAX_AGE_S = 30

# Fourth-slice runtime-interlock defaults (section 12.1), platform-independent --
# one constant default across every platform (twin and hardware alike), overridable
# via [override]/config. `free_air_abs_band_v` is the absolute floor on the free-air
# 3-read agreement band (13.1 near-zero addition); 0.02 V is a PLACEHOLDER above the
# Cypher's expected withdrawn-deflection noise -- TO FIX ON THE RIG (measure the real
# noise on Thursday, override in instrument.toml if needed).
_DEFAULT_FREE_AIR_ABS_BAND_V = FREE_AIR_ABS_BAND_V
_DEFAULT_SUM_FLOOR_V = 0.5
_DEFAULT_SUM_CEILING_FRAC = 0.9
_DEFAULT_Z_RAIL_V = 140.0
_DEFAULT_INTERLOCK_S = 5.0

# Section 15: the default value the Write tab's #lithoSetpoint hydrates to (RELATIVE
# to free air, V above the free-air deflection) when this service process has no
# last write. Platform-independent code default 0.1 V, `[instrument]
# write_setpoint_relative_v` in the TOML, overridable; NEVER 0 (an empty box refuses,
# 13.1 item 14, so the default must be a real, non-zero force above free air).
_DEFAULT_WRITE_SETPOINT_RELATIVE_V = 0.1


class FieldSource(str, Enum):
    platform_default = "platform_default"
    startup_cal = "startup_cal"
    config = "config"
    override = "override"


@dataclass(frozen=True)
class Field:
    value: Any
    source: FieldSource


OVERRIDABLE_FIELDS = (
    "platform", "z_out_channel", "z_verify_channels", "z_park_v", "z_park_tol_v",
    "dac_rail_v", "tip_amp_gain", "tip_voltage_window_v", "contact_setpoint_band_v",
    "ac_setpoint_fraction_band", "free_air_max_age_s", "xy_range_um",
    "free_air_abs_band_v", "sum_floor_v", "sum_ceiling_frac", "z_rail_v",
    "interlock_s", "write_setpoint_relative_v",
)


@dataclass
class InstrumentModel:
    platform: Field                # "cypher" | "mfp3d" | "twin"
    z_out_channel: Field           # `Height` | `Output.Z`
    z_verify_channels: Field       # (`ZSensor`, `Deflection`)
    z_park_v: Field                # float
    z_park_tol_v: Field            # float
    dac_rail_v: Field              # 10.0
    tip_amp_gain: Field            # float; NO default; None => model_incomplete
    tip_voltage_window_v: Field    # (v_min, v_max) for the active sample
    contact_setpoint_band_v: Field # (lo, hi) relative to free air
    ac_setpoint_fraction_band: Field
    free_air_max_age_s: Field
    xy_range_um: Field
    # Fourth-slice runtime interlocks (section 12.1): the free-air absolute
    # agreement floor, the pre-engage 0.9*SUM ceiling + SUM floor, the running
    # rail-dwell threshold, and the continuous-interlock dwell window.
    free_air_abs_band_v: Field   # absolute free-air agreement floor (V)
    sum_floor_v: Field           # SUM below this at engage -> sum_missing (V)
    sum_ceiling_frac: Field      # |setpoint| > frac*SUM -> setpoint_exceeds_sum
    z_rail_v: Field              # running rail-dwell threshold (V)
    interlock_s: Field           # continuous-interlock dwell (s)
    # Section 15: default #lithoSetpoint hydration value (V above free air).
    write_setpoint_relative_v: Field
    # Not one of section 10's fixed dataclass Fields -- the [sample] section's id,
    # surfaced plainly for the cockpit's sample identity (see the T1 module
    # docstring's ASSUMPTION on this). None off a hand-built model / the twin.
    sample_id: str | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {}
        for f in fields(self):
            if f.name == "sample_id":
                continue
            field: Field = getattr(self, f.name)
            value = field.value
            if isinstance(value, tuple):
                value = list(value)
            d[f.name] = {"value": value, "source": field.source.value}
        if self.sample_id is not None:
            d["sample_id"] = self.sample_id
        return d

    def snapshot_hash(self) -> str:
        import hashlib
        parts = []
        for f in fields(self):
            if f.name == "sample_id":
                parts.append(("sample_id", self.sample_id))
                continue
            field: Field = getattr(self, f.name)
            parts.append((f.name, field.value))
        blob = json.dumps(parts, sort_keys=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def instrument_config_path() -> Path:
    """`AFM_LITHO_INSTRUMENT_CONFIG`, else `<runtime>/instrument.toml` (section
    10). Read fresh on every call -- not cached -- so `POST
    /api/instrument/reload` and a monkeypatched env var in a test both take
    effect without a process restart."""
    from .deps import RUNTIME
    v = os.environ.get("AFM_LITHO_INSTRUMENT_CONFIG")
    return Path(v) if v else (RUNTIME / "instrument.toml")


def file_owned_platform(config_path: Path | None) -> str | None:
    """The platform declared in the config file (section 14 item 4): `[override]
    platform` beats `[instrument] platform`. `None` when the file is missing,
    unreadable, or declares no platform. Never raises -- callers use it only to
    pick a hardware fallback and to warn on an env-var disagreement."""
    try:
        doc = _read_toml(config_path)
    except Exception:                                    # noqa: BLE001
        return None
    ov = doc.get("override") if isinstance(doc.get("override"), dict) else {}
    if isinstance(ov.get("platform"), str) and ov["platform"]:
        return ov["platform"]
    inst = doc.get("instrument") if isinstance(doc.get("instrument"), dict) else {}
    if isinstance(inst.get("platform"), str) and inst["platform"]:
        return inst["platform"]
    return None


def overrides_log_path(config_path: Path) -> Path:
    """The overrides jsonl, derived as a SIBLING of the config file (section 10
    only names `AFM_LITHO_INSTRUMENT_CONFIG`; there is no independent env var for
    this path)."""
    return Path(config_path).parent / "instrument_overrides.jsonl"


class ModelIncomplete(Exception):
    """Raised by `load_model` on a hardware platform (`cypher`/`mfp3d`) whose
    config is missing, unreadable, or missing/mistyped `tip_amp_gain` or
    `tip_voltage_window_v` -- the two config-only fields with no safe code
    default (section 1, section 10: "ModelIncomplete when amp_gain OR window are
    absent"). The twin ships complete defaults and never raises."""


# --------------------------------------------------------------------- loading --

def _as_tuple2(value, name: str) -> tuple[float, float]:
    if (not isinstance(value, (list, tuple))) or len(value) != 2:
        raise ValueError(f"{name} must be a 2-element list, got {value!r}")
    return (float(value[0]), float(value[1]))


def _read_toml(config_path: Path | None) -> dict:
    if config_path is None or not Path(config_path).exists():
        return {}
    text = Path(config_path).read_text(encoding="utf-8")
    return tomllib.loads(text)


def _read_overrides_table(config_path: Path | None) -> dict:
    doc = _read_toml(config_path)
    ov = doc.get("override")
    return ov if isinstance(ov, dict) else {}


def load_model(*, platform: str, startup_cal: dict | None,
               config_path: Path | None, overrides_log: Path | None) -> InstrumentModel:
    """Build the model: `platform_default < startup_cal < config < override`.

    `platform in {"cypher", "mfp3d"}` is a HARDWARE platform: a missing config file,
    or a config missing/mistyping `tip_amp_gain` or `tip_voltage_window_v` (the two
    config-only fields, section 10), raises `ModelIncomplete` rather than silently
    driving real hardware off platform-default guesses. `platform == "twin"` never
    raises -- it ships complete defaults and reads the file only if one happens to
    be given.
    """
    # The PASSED platform reflects the caller's backend intent (hardware vs twin)
    # and gates the missing-file check (a missing config on a hardware backend can
    # never be read to discover the file-owned platform, so it must still raise).
    backend_is_hw = platform in _HARDWARE_PLATFORMS
    doc = {}
    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            try:
                doc = tomllib.loads(path.read_text(encoding="utf-8"))
            except tomllib.TOMLDecodeError:
                if backend_is_hw:
                    raise ModelIncomplete(
                        f"instrument config at {path} is not valid TOML") from None
                raise
        elif backend_is_hw:
            raise ModelIncomplete(f"instrument config not found: {path}")
    elif backend_is_hw:
        raise ModelIncomplete("no instrument config path given for a hardware platform")

    instrument_tbl = doc.get("instrument") if isinstance(doc.get("instrument"), dict) else {}
    sample_tbl = doc.get("sample") if isinstance(doc.get("sample"), dict) else {}
    override_tbl = doc.get("override") if isinstance(doc.get("override"), dict) else {}
    startup_cal = startup_cal or {}

    # ---- platform: FILE-OWNED (section 14 item 4) ---------------------------
    # `[instrument] platform` (overridden by `[override] platform`) is the single
    # source of truth. The EFFECTIVE platform then drives is_hw AND every
    # platform_default below, so changing platform re-derives z_out_channel,
    # z_park_v, z_rail_v, xy_range, ... The AFM_LITHO_PLATFORM env var is no longer
    # a source (Hub warns if it disagrees with the file; the file wins).
    if "platform" in override_tbl:
        platform, platform_field = (str(override_tbl["platform"]),
                                    Field(str(override_tbl["platform"]), FieldSource.override))
    elif "platform" in instrument_tbl:
        platform, platform_field = (str(instrument_tbl["platform"]),
                                    Field(str(instrument_tbl["platform"]), FieldSource.config))
    elif "platform" in startup_cal or "microscope_id" in startup_cal:
        platform_field = Field(platform, FieldSource.startup_cal)
    else:
        platform_field = Field(platform, FieldSource.platform_default)

    # is_hw now follows the EFFECTIVE platform (a hardware platform requires
    # tip_amp_gain from config; the twin ships defaults).
    is_hw = platform in _HARDWARE_PLATFORMS

    def resolve(name: str, platform_default, *, caster=lambda v: v):
        # override beats everything
        if name in override_tbl:
            return Field(caster(override_tbl[name]), FieldSource.override)
        if name in instrument_tbl:
            return Field(caster(instrument_tbl[name]), FieldSource.config)
        if name in startup_cal:
            return Field(caster(startup_cal[name]), FieldSource.startup_cal)
        return Field(platform_default, FieldSource.platform_default)

    # ---- z_out_channel / z_verify_channels / z_park / dac_rail (platform default) --
    z_out_channel = resolve("z_out_channel", Z_OUT, caster=str)
    z_verify_channels = resolve("z_verify_channels", _DEFAULT_Z_VERIFY_CHANNELS,
                                caster=lambda v: tuple(v))
    z_park_v = resolve("z_park_v", _DEFAULT_Z_PARK_V, caster=float)
    z_park_tol_v = resolve("z_park_tol_v", _DEFAULT_Z_PARK_TOL_V, caster=float)
    dac_rail_v = resolve("dac_rail_v", _DEFAULT_DAC_RAIL_V, caster=float)

    # ---- tip_amp_gain: CONFIG ONLY, NO CODE DEFAULT on hardware --------------
    if "tip_amp_gain" in override_tbl:
        raw_gain = override_tbl["tip_amp_gain"]
        gain_source = FieldSource.override
    elif "tip_amp_gain" in instrument_tbl:
        raw_gain = instrument_tbl["tip_amp_gain"]
        gain_source = FieldSource.config
    else:
        raw_gain = None
        gain_source = FieldSource.platform_default

    if raw_gain is None:
        if is_hw:
            raise ModelIncomplete("tip_amp_gain is required on a hardware platform "
                                  "and has no code default")
        gain_value = _TWIN_TIP_AMP_GAIN if platform == "twin" else None
        tip_amp_gain = Field(gain_value, gain_source if gain_value is None
                             else FieldSource.platform_default)
    else:
        if isinstance(raw_gain, bool) or not isinstance(raw_gain, (int, float)):
            # A quoted "25.0" (or any other non-numeric type) must never load
            # silently as a rig-safety gain. On hardware this is effectively an
            # absent value; elsewhere it is a hard type error.
            if is_hw:
                raise ModelIncomplete(
                    f"tip_amp_gain must be a number, got {raw_gain!r}")
            raise TypeError(f"tip_amp_gain must be a number, got {raw_gain!r}")
        tip_amp_gain = Field(float(raw_gain), gain_source)

    # ---- config-sourced bands / window ---------------------------------------
    # tip_voltage_window_v lives under [sample] in the TOML (per-sample, section
    # 10's example), not [instrument] -- checked separately from `resolve`, which
    # only reads [instrument]/[override]/startup_cal.
    window_default = _TWIN_TIP_VOLTAGE_WINDOW_V if platform == "twin" else (0.0, 0.0)
    if "tip_voltage_window_v" in override_tbl:
        tip_voltage_window_v = Field(
            _as_tuple2(override_tbl["tip_voltage_window_v"], "tip_voltage_window_v"),
            FieldSource.override)
    elif "tip_voltage_window_v" in sample_tbl:
        tip_voltage_window_v = Field(
            _as_tuple2(sample_tbl["tip_voltage_window_v"], "tip_voltage_window_v"),
            FieldSource.config)
    elif "tip_voltage_window_v" in startup_cal:
        tip_voltage_window_v = Field(
            _as_tuple2(startup_cal["tip_voltage_window_v"], "tip_voltage_window_v"),
            FieldSource.startup_cal)
    elif is_hw:
        # Section 10: ModelIncomplete when amp_gain OR window are absent on a
        # hardware platform. A window is per-sample (`[sample]
        # tip_voltage_window_v`) and has NO safe code default -- falling back to
        # `(0.0, 0.0)` at platform_default would let the service start driving a
        # real tip with an all-zero (or, worse, a stale) voltage window instead
        # of REFUSING. Raise naming the missing key so the operator knows what to
        # add to `instrument.toml`. The twin ships a real default and is
        # unaffected (it never reaches this branch: window_default is set).
        raise ModelIncomplete(
            "tip_voltage_window_v is required on a hardware platform and has no "
            "code default; add [sample] tip_voltage_window_v to the instrument "
            "config")
    else:
        tip_voltage_window_v = Field(window_default, FieldSource.platform_default)
    contact_default = (_TWIN_CONTACT_SETPOINT_BAND_V if platform == "twin"
                       else (0.0, 1.0))
    contact_setpoint_band_v = resolve(
        "contact_setpoint_band_v", contact_default,
        caster=lambda v: _as_tuple2(v, "contact_setpoint_band_v"))
    ac_default = (_TWIN_AC_SETPOINT_FRACTION_BAND if platform == "twin"
                 else (0.3, 0.95))
    ac_setpoint_fraction_band = resolve(
        "ac_setpoint_fraction_band", ac_default,
        caster=lambda v: _as_tuple2(v, "ac_setpoint_fraction_band"))
    free_air_default = _TWIN_FREE_AIR_MAX_AGE_S if platform == "twin" else 30
    free_air_max_age_s = resolve("free_air_max_age_s", free_air_default, caster=float)

    xy_default = _PLATFORM_XY_RANGE_UM.get(platform, (90.0, 90.0))
    xy_range_um = resolve("xy_range_um", xy_default,
                          caster=lambda v: _as_tuple2(v, "xy_range_um"))

    # Fourth-slice interlock fields (section 12.1): one platform-independent code
    # default each, overridable via [instrument]/[override]/startup_cal.
    free_air_abs_band_v = resolve("free_air_abs_band_v", _DEFAULT_FREE_AIR_ABS_BAND_V,
                                  caster=float)
    sum_floor_v = resolve("sum_floor_v", _DEFAULT_SUM_FLOOR_V, caster=float)
    sum_ceiling_frac = resolve("sum_ceiling_frac", _DEFAULT_SUM_CEILING_FRAC,
                               caster=float)
    z_rail_v = resolve("z_rail_v", _DEFAULT_Z_RAIL_V, caster=float)
    interlock_s = resolve("interlock_s", _DEFAULT_INTERLOCK_S, caster=float)
    write_setpoint_relative_v = resolve(
        "write_setpoint_relative_v", _DEFAULT_WRITE_SETPOINT_RELATIVE_V, caster=float)

    sample_id = sample_tbl.get("id") if isinstance(sample_tbl.get("id"), str) else None

    return InstrumentModel(
        platform=platform_field,
        z_out_channel=z_out_channel,
        z_verify_channels=z_verify_channels,
        z_park_v=z_park_v,
        z_park_tol_v=z_park_tol_v,
        dac_rail_v=dac_rail_v,
        tip_amp_gain=tip_amp_gain,
        tip_voltage_window_v=tip_voltage_window_v,
        contact_setpoint_band_v=contact_setpoint_band_v,
        ac_setpoint_fraction_band=ac_setpoint_fraction_band,
        free_air_max_age_s=free_air_max_age_s,
        xy_range_um=xy_range_um,
        free_air_abs_band_v=free_air_abs_band_v,
        sum_floor_v=sum_floor_v,
        sum_ceiling_frac=sum_ceiling_frac,
        z_rail_v=z_rail_v,
        interlock_s=interlock_s,
        write_setpoint_relative_v=write_setpoint_relative_v,
        sample_id=sample_id,
    )


# ---------------------------------------------------------------- overrides ----

def _toml_str(s: str) -> str:
    esc = str(s).replace("\\", "\\\\").replace('"', '\\"')
    esc = esc.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return f'"{esc}"'


def _toml_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_scalar(x) for x in v) + "]"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    return _toml_str(v)


def _rewrite_toml_with_override(config_path: Path, field: str, value: Any) -> None:
    """Read the current TOML (if any), set `[override].<field> = value`, and
    rewrite the whole file. Deliberately simple (a handful of scalar keys across
    three flat tables) rather than a general TOML writer."""
    doc = _read_toml(config_path if config_path.exists() else None)
    instrument_tbl = doc.get("instrument") if isinstance(doc.get("instrument"), dict) else {}
    sample_tbl = doc.get("sample") if isinstance(doc.get("sample"), dict) else {}
    override_tbl = dict(doc.get("override") or {})
    override_tbl[field] = value

    lines: list[str] = []
    if instrument_tbl:
        lines.append("[instrument]")
        for k, v in instrument_tbl.items():
            lines.append(f"{k} = {_toml_scalar(v)}")
        lines.append("")
    if sample_tbl:
        lines.append("[sample]")
        for k, v in sample_tbl.items():
            lines.append(f"{k} = {_toml_scalar(v)}")
        lines.append("")
    lines.append("[override]")
    for k, v in override_tbl.items():
        lines.append(f"{k} = {_toml_scalar(v)}")
    lines.append("")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("\n".join(lines), encoding="utf-8")


def apply_override(model: InstrumentModel, field: str, value, *, reason: str,
                   client: str, config_path: Path, overrides_log: Path) -> InstrumentModel:
    """Persist an override to BOTH the jsonl audit trail and the TOML `[override]`
    table, and return a model with that field's `Field.source == override`.

    `config_path`/`overrides_log` are REQUIRED keywords (no default) -- an override
    that is not written down survives nothing, and section 1 requires it to survive
    a restart.
    """
    if field not in OVERRIDABLE_FIELDS:
        raise KeyError(f"not an overridable instrument-model field: {field!r}")
    old_field: Field = getattr(model, field)
    old = old_field.value
    if isinstance(old, tuple):
        old = list(old)

    new_value = value
    if isinstance(old_field.value, tuple):
        new_value = _as_tuple2(value, field)

    config_path = Path(config_path)
    overrides_log = Path(overrides_log)
    overrides_log.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now(UTC).isoformat(), "client": client, "field": field,
          "old": old, "new": (list(new_value) if isinstance(new_value, tuple)
                              else new_value),
          "reason": reason}
    with overrides_log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")

    _rewrite_toml_with_override(config_path, field,
                                list(new_value) if isinstance(new_value, tuple)
                                else new_value)

    # Platform is file-owned and re-derives every platform_default field
    # (section 14 item 4): reload from disk (which now reads the just-written
    # [override] platform) rather than copying the stale z_out_channel/z_park/...
    # from `model`. Every other field is a local change, so the cheap copy stands.
    if field == "platform":
        return load_model(platform=str(new_value), startup_cal=None,
                          config_path=config_path, overrides_log=overrides_log)

    new_field = Field(new_value, FieldSource.override)
    kwargs = {f.name: getattr(model, f.name) for f in fields(model)
             if f.name != "sample_id"}
    kwargs[field] = new_field
    return InstrumentModel(sample_id=model.sample_id, **kwargs)
