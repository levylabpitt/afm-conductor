"""Instrument model + config + overrides + REST/FLEX read (Wave 1, T1).

Truth document: docs/INSTRUMENT_MODEL_SPEC.md, sections 1 and 10 (the interface
contract "tests and implementation both target exactly this"). These tests are
written RED, against section 10 exactly, before
`src/afm_litho/service/instrument_model.py` exists. They cover:

  * platform defaults per platform (cypher/mfp3d/twin), every field carrying a
    `source`;
  * `load_model` raising `ModelIncomplete` on a hardware platform with no config
    file / no `tip_amp_gain`, and loading fine on the twin platform without one;
  * config precedence (platform_default < startup_cal < config < override) and
    `to_dict()` / `snapshot_hash()`;
  * `apply_override` persisting to both the overrides jsonl and the TOML
    `[override]` table, surviving a fresh `load_model` (i.e. surviving restart);
  * REST `GET/POST /api/instrument*` and FLEX `getInstrument`;
  * the hardware-backend-with-missing-config REFUSING state (`hub.fault` +
    `model_incomplete`).

ZERO HARDWARE. Every file this suite touches is under `tmp_path`.

ASSUMPTIONS made where section 10 does not pin the answer (see the T1 report for
the full list -- these are NOT to be treated as spec changes):

  * `load_model` decides "hardware platform" (subject to `ModelIncomplete`) from
    `platform in {"cypher", "mfp3d"}`; `platform == "twin"` never raises.
  * The model lives at `hub.instrument_model` (spec fix, commit 97a4296):
    `hub.instrument` stays the twin's `twin.physics.Instrument` calibration
    object, unchanged. The only refusal reason for a missing/incomplete model
    is `model_incomplete` -- there is no separate `instrument_model_missing`
    string anywhere in the contract.
  * IMPLEMENTER TO CONFIRM: REST endpoints (and `Hub()` construction) read the
    config/overrides paths from the environment (`AFM_LITHO_INSTRUMENT_CONFIG`)
    PER REQUEST / PER CONSTRUCTION rather than caching them once at process
    start -- required for `POST /api/instrument/reload` to mean anything, and
    assumed to hold for override/open_config/GET and for a freshly-constructed
    `Hub()` too (the persistence-survives-a-rebuild REST test below relies on
    this). Each test sets this env var (via `monkeypatch`) before acting. If the
    real implementation instead resolves the path once at process/hub startup,
    these tests need a fixture that restarts the process (or re-imports the hub
    module) rather than just setting the env var.
  * `POST /api/instrument/open_config` calls `os.startfile` via the `os` module
    (`os.startfile(path)`, not `from os import startfile`), so
    `monkeypatch.setattr("os.startfile", ...)` intercepts it.
  * The overrides log path is not independently configurable by env var (section
    10 only names `AFM_LITHO_INSTRUMENT_CONFIG`); it is derived as
    `<config_path's runtime dir>/instrument_overrides.jsonl`, i.e. a sibling of
    the config file. Unit-level `apply_override` tests avoid this question by
    passing both paths explicitly, as the contract's signature allows.
"""

from __future__ import annotations

import json

import pytest

# ------------------------------------------------------------------- imports --
# RED today: this module does not exist yet. Every test below is expected to
# fail at collection with ImportError until it does.
from afm_litho.service.instrument_model import (
    Field,
    FieldSource,
    InstrumentModel,
    ModelIncomplete,
    apply_override,
    load_model,
)

pytestmark = pytest.mark.usefixtures("_flex_provider_off")


# ------------------------------------------------------------------- helpers --

def _write_toml(path, *, platform="cypher", tip_amp_gain=25.0,
                 sample_id="SA40656B0000", window=(-8.0, 30.0),
                 contact_band=(0.0, 1.0), ac_band=(0.3, 0.95),
                 free_air_max_age_s=30, extra=""):
    path.write_text(
        "[instrument]\n"
        f'platform = "{platform}"\n'
        f"tip_amp_gain = {tip_amp_gain}\n"
        f"contact_setpoint_band_v = [{contact_band[0]}, {contact_band[1]}]\n"
        f"ac_setpoint_fraction_band = [{ac_band[0]}, {ac_band[1]}]\n"
        f"free_air_max_age_s = {free_air_max_age_s}\n"
        "[sample]\n"
        f'id = "{sample_id}"\n'
        f"tip_voltage_window_v = [{window[0]}, {window[1]}]\n"
        f"{extra}\n",
        encoding="utf-8",
    )
    return path


def _load(tmp_path, *, platform="cypher", config=True, overrides=True, **kw):
    config_path = tmp_path / "instrument.toml"
    if config:
        _write_toml(config_path, platform=platform, **kw)
    overrides_log = tmp_path / "instrument_overrides.jsonl"
    return load_model(
        platform=platform,
        startup_cal=None,
        config_path=config_path if config else (tmp_path / "missing.toml"),
        overrides_log=overrides_log if overrides else None,
    )


# ================================================================== 1. defaults

@pytest.mark.parametrize("platform, z_out_channel", [
    ("cypher", "Height"),
    ("mfp3d", "Height"),
])
def test_platform_default_z_out_channel(tmp_path, platform, z_out_channel):
    model = _load(tmp_path, platform=platform)
    assert model.z_out_channel.value == z_out_channel


def test_twin_platform_defaults_without_a_config_file():
    # The twin backend ships its own defaults and does not need the file at all
    # (section 1): no config_path, no overrides_log.
    model = load_model(platform="twin", startup_cal=None, config_path=None,
                       overrides_log=None)
    assert model.platform.value == "twin"
    assert model.z_out_channel.value in ("Height", "Output.Z")


def test_every_field_carries_a_source(tmp_path):
    model = _load(tmp_path, platform="cypher")
    for name in ("platform", "z_out_channel", "z_verify_channels", "z_park_v",
                 "z_park_tol_v", "dac_rail_v", "tip_amp_gain",
                 "tip_voltage_window_v", "contact_setpoint_band_v",
                 "ac_setpoint_fraction_band", "free_air_max_age_s",
                 "xy_range_um"):
        field = getattr(model, name)
        assert isinstance(field, Field), name
        assert isinstance(field.source, FieldSource), name


def test_dac_rail_v_platform_default_is_10(tmp_path):
    model = _load(tmp_path, platform="cypher")
    assert model.dac_rail_v.value == 10.0
    assert model.dac_rail_v.source == FieldSource.platform_default


def test_z_verify_channels_platform_default(tmp_path):
    model = _load(tmp_path, platform="mfp3d")
    assert tuple(model.z_verify_channels.value) == ("ZSensor", "Deflection")


def test_config_sourced_bands_and_window_carry_their_toml_values(tmp_path):
    """Values, not just sources: a stub that hard-codes the spec's example
    figures (which happen to be this file's other fixture defaults) would pass
    a source-only check, so this test uses DISTINCT numbers nothing else in the
    file uses."""
    model = _load(tmp_path, platform="cypher", window=(-6.5, 22.0),
                 contact_band=(0.1, 0.8), ac_band=(0.25, 0.88),
                 sample_id="SA_DISTINCT_0001")
    assert tuple(model.tip_voltage_window_v.value) == (-6.5, 22.0)
    assert model.tip_voltage_window_v.source == FieldSource.config
    assert tuple(model.contact_setpoint_band_v.value) == (0.1, 0.8)
    assert model.contact_setpoint_band_v.source == FieldSource.config
    assert tuple(model.ac_setpoint_fraction_band.value) == (0.25, 0.88)
    assert model.ac_setpoint_fraction_band.source == FieldSource.config
    # The [sample] section's id, per the TOML layout in section 10. Not one of
    # the fixed dataclass Fields listed there -- ASSUMPTION: exposed as a plain
    # "sample_id" key in to_dict() (or a `model.sample_id` attribute), since the
    # implementer must surface it SOMEHOW for the cockpit's sample identity and
    # section 10 does not name a Field for it.
    d = model.to_dict()
    sample_id = d.get("sample_id", getattr(model, "sample_id", None))
    assert sample_id == "SA_DISTINCT_0001"


# ============================================================ 2. ModelIncomplete

@pytest.mark.parametrize("platform", ["cypher", "mfp3d"])
def test_load_model_raises_on_hardware_with_no_config_file(tmp_path, platform):
    missing = tmp_path / "instrument.toml"
    assert not missing.exists()
    with pytest.raises(ModelIncomplete):
        load_model(platform=platform, startup_cal=None, config_path=missing,
                   overrides_log=tmp_path / "instrument_overrides.jsonl")


@pytest.mark.parametrize("platform", ["cypher", "mfp3d"])
def test_load_model_raises_on_hardware_with_no_tip_amp_gain(tmp_path, platform):
    path = tmp_path / "instrument.toml"
    # Config file exists, but omits tip_amp_gain -- "config only, no code default".
    path.write_text(
        f'[instrument]\nplatform = "{platform}"\n'
        "contact_setpoint_band_v = [0.0, 1.0]\n"
        "ac_setpoint_fraction_band = [0.3, 0.95]\n"
        "free_air_max_age_s = 30\n"
        '[sample]\nid = "SA1"\ntip_voltage_window_v = [-8.0, 30.0]\n',
        encoding="utf-8",
    )
    with pytest.raises(ModelIncomplete):
        load_model(platform=platform, startup_cal=None, config_path=path,
                   overrides_log=tmp_path / "instrument_overrides.jsonl")


@pytest.mark.parametrize("platform", ["cypher", "mfp3d"])
def test_load_model_raises_on_hardware_with_no_tip_voltage_window(tmp_path, platform):
    # S1 (spec section 10): "ModelIncomplete when amp_gain OR window are absent"
    # on a hardware platform. Config exists and HAS tip_amp_gain but omits the
    # per-sample [sample] tip_voltage_window_v -- it used to fall to the
    # platform_default (0.0, 0.0) window, which would let the service start
    # driving a real tip against an all-zero window.
    path = tmp_path / "instrument.toml"
    path.write_text(
        "[instrument]\n"
        f'platform = "{platform}"\n'
        "tip_amp_gain = 25.0\n"
        "contact_setpoint_band_v = [0.0, 1.0]\n"
        "ac_setpoint_fraction_band = [0.3, 0.95]\n"
        "free_air_max_age_s = 30\n"
        "[sample]\n"
        'id = "SA1"\n',                       # NO tip_voltage_window_v
        encoding="utf-8",
    )
    with pytest.raises(ModelIncomplete) as ei:
        load_model(platform=platform, startup_cal=None, config_path=path,
                   overrides_log=tmp_path / "instrument_overrides.jsonl")
    assert "tip_voltage_window_v" in str(ei.value), (
        "the ModelIncomplete message must name the missing key so the fault "
        "detail (hub.py) tells the operator what to add")


def test_twin_loads_without_a_tip_voltage_window(tmp_path):
    # The twin ships a real window default and must NOT raise when the config
    # omits the window (S1: "twin platform unchanged").
    path = tmp_path / "instrument.toml"
    path.write_text(
        "[instrument]\n"
        'platform = "twin"\n'
        "[sample]\n"
        'id = "SA1"\n',                       # no window, no amp gain
        encoding="utf-8",
    )
    model = load_model(platform="twin", startup_cal=None, config_path=path,
                       overrides_log=tmp_path / "instrument_overrides.jsonl")
    assert model.tip_voltage_window_v.source == FieldSource.platform_default
    assert len(tuple(model.tip_voltage_window_v.value)) == 2


def test_twin_loads_without_any_file_or_paths():
    model = load_model(platform="twin", startup_cal=None, config_path=None,
                       overrides_log=None)
    assert model.tip_amp_gain.value is None or isinstance(
        model.tip_amp_gain.value, float)


# ============================================================== 3. precedence

def test_config_overrides_platform_default(tmp_path):
    model = _load(tmp_path, platform="cypher", free_air_max_age_s=45)
    assert model.free_air_max_age_s.value == 45
    assert model.free_air_max_age_s.source == FieldSource.config


def test_startup_cal_overrides_platform_default_but_not_config(tmp_path):
    config_path = tmp_path / "instrument.toml"
    _write_toml(config_path, platform="cypher")
    model = load_model(
        platform="cypher",
        startup_cal={"xy_range_um": (55.0, 55.0), "microscope_id": 12},
        config_path=config_path,
        overrides_log=tmp_path / "instrument_overrides.jsonl",
    )
    assert model.xy_range_um.value == (55.0, 55.0)
    assert model.xy_range_um.source == FieldSource.startup_cal
    # tip_amp_gain came only from config, never from startup_cal
    assert model.tip_amp_gain.source == FieldSource.config


def test_to_dict_reports_source_per_field(tmp_path):
    model = _load(tmp_path, platform="cypher")
    d = model.to_dict()
    assert d["tip_amp_gain"] == {"value": 25.0, "source": "config"}
    assert d["dac_rail_v"]["source"] == "platform_default"


# A hand-built, fully-populated model per field, bypassing load_model/TOML
# entirely -- so "does snapshot_hash react to THIS field" is tested directly
# against the dataclass, independent of whether that field happens to be
# config-overridable in the current implementation (dac_rail_v and
# z_out_channel are platform-default-only per section 10's table, so two
# legitimately-loaded models could never otherwise differ only in those).
_BASELINE_FIELD_VALUES = {
    "platform": "cypher",
    "z_out_channel": "Height",
    "z_verify_channels": ("ZSensor", "Deflection"),
    "z_park_v": 0.0,
    "z_park_tol_v": 0.5,
    "dac_rail_v": 10.0,
    "tip_amp_gain": 25.0,
    "tip_voltage_window_v": (-8.0, 30.0),
    "contact_setpoint_band_v": (0.0, 1.0),
    "ac_setpoint_fraction_band": (0.3, 0.95),
    "free_air_max_age_s": 30,
    "xy_range_um": (90.0, 90.0),
    "free_air_abs_band_v": 0.02,
    "sum_floor_v": 0.5,
    "sum_ceiling_frac": 0.9,
    "z_rail_v": 140.0,
    "interlock_s": 5.0,
    "write_setpoint_relative_v": 0.1,
}


def _full_model(**value_overrides):
    kwargs = {}
    for name, value in _BASELINE_FIELD_VALUES.items():
        kwargs[name] = Field(value=value_overrides.get(name, value),
                             source=FieldSource.platform_default)
    return InstrumentModel(**kwargs)


@pytest.mark.parametrize("field, changed_value", [
    ("tip_voltage_window_v", (-5.0, 30.0)),
    ("contact_setpoint_band_v", (0.0, 2.0)),
    ("ac_setpoint_fraction_band", (0.3, 0.9)),
    ("dac_rail_v", 12.0),
    ("z_park_v", 1.0),
    ("z_out_channel", "Output.Z"),
    ("platform", "mfp3d"),
    ("free_air_max_age_s", 45),
    ("tip_amp_gain", 30.0),
    ("write_setpoint_relative_v", 0.3),
])
def test_snapshot_hash_changes_for_every_field(field, changed_value):
    baseline = _full_model()
    changed = _full_model(**{field: changed_value})
    assert baseline.snapshot_hash() != changed.snapshot_hash(), field


# --------------------------------------------------------------------------
# Section 15: default write setpoint above free air (write_setpoint_relative_v).
# --------------------------------------------------------------------------

def test_write_setpoint_relative_v_code_default_is_0_1():
    """Section 15 item 1: platform-independent code default 0.1, present with a
    platform_default source on the twin without any config file."""
    model = load_model(platform="twin", startup_cal=None, config_path=None,
                       overrides_log=None)
    f = model.write_setpoint_relative_v
    assert f.value == 0.1
    assert f.source == FieldSource.platform_default
    d = model.to_dict()
    assert d["write_setpoint_relative_v"] == {"value": 0.1,
                                              "source": "platform_default"}


def test_write_setpoint_relative_v_from_config(tmp_path):
    """Section 15 item 1: `[instrument] write_setpoint_relative_v` in the TOML
    wins over the code default, source `config`."""
    config_path = tmp_path / "instrument.toml"
    # write_setpoint_relative_v belongs under [instrument]; write the file directly.
    config_path.write_text(
        chr(10).join([
            "[instrument]",
            'platform = "cypher"',
            "tip_amp_gain = 25.0",
            "write_setpoint_relative_v = 0.4",
            "[sample]",
            'id = "S"',
            "tip_voltage_window_v = [-8.0, 30.0]",
            "",
        ]),
        encoding="utf-8")
    model = load_model(platform="cypher", startup_cal=None, config_path=config_path,
                       overrides_log=tmp_path / "ov.jsonl")
    assert model.write_setpoint_relative_v.value == 0.4
    assert model.write_setpoint_relative_v.source == FieldSource.config


def test_write_setpoint_relative_v_is_overridable_and_survives_reload(tmp_path):
    """Section 15 item 1: overridable like every other field; the override
    persists to the TOML `[override]` table and survives a fresh load_model."""
    from afm_litho.service.instrument_model import apply_override
    config_path = tmp_path / "instrument.toml"
    overrides_log = tmp_path / "instrument_overrides.jsonl"
    model = load_model(platform="cypher", startup_cal=None,
                       config_path=_write_toml(config_path),
                       overrides_log=overrides_log)
    updated = apply_override(model, "write_setpoint_relative_v", 0.2,
                            reason="test", client="t",
                            config_path=config_path, overrides_log=overrides_log)
    assert updated.write_setpoint_relative_v.value == 0.2
    assert updated.write_setpoint_relative_v.source == FieldSource.override
    reloaded = load_model(platform="cypher", startup_cal=None,
                          config_path=config_path, overrides_log=overrides_log)
    assert reloaded.write_setpoint_relative_v.value == 0.2
    assert reloaded.write_setpoint_relative_v.source == FieldSource.override


def test_snapshot_hash_stable_for_identical_models(tmp_path):
    p1 = tmp_path / "one"
    p1.mkdir()
    p2 = tmp_path / "two"
    p2.mkdir()
    m1 = _load(p1, platform="cypher", tip_amp_gain=25.0)
    m2 = _load(p2, platform="cypher", tip_amp_gain=25.0)
    assert m1.snapshot_hash() == m2.snapshot_hash()


# ============================================================ 4. apply_override

def test_apply_override_requires_persisting_paths(tmp_path):
    model = _load(tmp_path, platform="cypher")
    with pytest.raises(TypeError):
        apply_override(model, "tip_amp_gain", 30.0, reason="rig check",
                       client="operator")   # missing config_path/overrides_log


def test_apply_override_appends_one_jsonl_line_with_the_documented_keys(tmp_path):
    config_path = tmp_path / "instrument.toml"
    overrides_log = tmp_path / "instrument_overrides.jsonl"
    model = load_model(platform="cypher", startup_cal=None,
                       config_path=_write_toml(config_path),
                       overrides_log=overrides_log)
    apply_override(model, "tip_amp_gain", 30.0, reason="rig check",
                   client="operator", config_path=config_path,
                   overrides_log=overrides_log)
    lines = overrides_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert {"ts", "client", "field", "old", "new", "reason"} <= set(rec)
    assert rec["field"] == "tip_amp_gain"
    assert rec["old"] == 25.0
    assert rec["new"] == 30.0
    assert rec["client"] == "operator"
    assert rec["reason"] == "rig check"


def test_apply_override_writes_the_override_table_in_the_toml(tmp_path):
    config_path = tmp_path / "instrument.toml"
    overrides_log = tmp_path / "instrument_overrides.jsonl"
    model = load_model(platform="cypher", startup_cal=None,
                       config_path=_write_toml(config_path),
                       overrides_log=overrides_log)
    apply_override(model, "tip_amp_gain", 30.0, reason="rig check",
                   client="operator", config_path=config_path,
                   overrides_log=overrides_log)
    text = config_path.read_text(encoding="utf-8")
    assert "[override]" in text
    assert "tip_amp_gain" in text.split("[override]", 1)[1]


def test_apply_override_returns_a_model_with_source_override(tmp_path):
    config_path = tmp_path / "instrument.toml"
    overrides_log = tmp_path / "instrument_overrides.jsonl"
    model = load_model(platform="cypher", startup_cal=None,
                       config_path=_write_toml(config_path),
                       overrides_log=overrides_log)
    updated = apply_override(model, "tip_amp_gain", 30.0, reason="rig check",
                             client="operator", config_path=config_path,
                             overrides_log=overrides_log)
    assert updated.tip_amp_gain.value == 30.0
    assert updated.tip_amp_gain.source == FieldSource.override


def test_override_survives_a_fresh_load_model_after_restart(tmp_path):
    config_path = tmp_path / "instrument.toml"
    overrides_log = tmp_path / "instrument_overrides.jsonl"
    model = load_model(platform="cypher", startup_cal=None,
                       config_path=_write_toml(config_path),
                       overrides_log=overrides_log)
    apply_override(model, "tip_amp_gain", 30.0, reason="rig check",
                   client="operator", config_path=config_path,
                   overrides_log=overrides_log)

    # Simulate a restart: a brand new load_model call against the same files.
    reloaded = load_model(platform="cypher", startup_cal=None,
                          config_path=config_path, overrides_log=overrides_log)
    assert reloaded.tip_amp_gain.value == 30.0
    assert reloaded.tip_amp_gain.source == FieldSource.override


def test_override_beats_config_precedence_order(tmp_path):
    # platform_default < startup_cal < config < override, end to end.
    config_path = tmp_path / "instrument.toml"
    overrides_log = tmp_path / "instrument_overrides.jsonl"
    _write_toml(config_path, free_air_max_age_s=45)
    model = load_model(platform="cypher", startup_cal=None,
                       config_path=config_path, overrides_log=overrides_log)
    assert model.free_air_max_age_s.source == FieldSource.config
    updated = apply_override(model, "free_air_max_age_s", 60, reason="rig",
                             client="operator", config_path=config_path,
                             overrides_log=overrides_log)
    assert updated.free_air_max_age_s.value == 60
    assert updated.free_air_max_age_s.source == FieldSource.override


def test_apply_override_on_an_unknown_field_raises(tmp_path):
    """ASSUMPTION: section 10 does not name the exception for an unknown field
    name; a `KeyError` (bad dict/attr key) or `ValueError` (bad enum-like value)
    both read as reasonable, so either is accepted here -- confirm with the
    implementer which one lands, and tighten this test then."""
    config_path = tmp_path / "instrument.toml"
    overrides_log = tmp_path / "instrument_overrides.jsonl"
    model = load_model(platform="cypher", startup_cal=None,
                       config_path=_write_toml(config_path),
                       overrides_log=overrides_log)
    with pytest.raises((KeyError, ValueError, AttributeError)):
        apply_override(model, "not_a_real_field", 1, reason="typo",
                       client="operator", config_path=config_path,
                       overrides_log=overrides_log)


def test_tip_amp_gain_as_a_toml_string_never_loads_silently(tmp_path):
    """`tip_amp_gain = "25.0"` (quoted) in the TOML must not be accepted as a
    float lookalike. ASSUMPTION: the implementer may choose to treat a wrong
    type as ModelIncomplete (this rig-safety field is effectively "absent" if
    it isn't the right type) or as a more generic type error; either satisfies
    the requirement that it never SILENTLY loads as a string."""
    path = tmp_path / "instrument.toml"
    path.write_text(
        '[instrument]\nplatform = "cypher"\ntip_amp_gain = "25.0"\n'
        "contact_setpoint_band_v = [0.0, 1.0]\n"
        "ac_setpoint_fraction_band = [0.3, 0.95]\n"
        "free_air_max_age_s = 30\n"
        '[sample]\nid = "SA1"\ntip_voltage_window_v = [-8.0, 30.0]\n',
        encoding="utf-8",
    )
    try:
        model = load_model(platform="cypher", startup_cal=None,
                           config_path=path,
                           overrides_log=tmp_path / "instrument_overrides.jsonl")
    except (ModelIncomplete, TypeError, ValueError):
        return
    assert not isinstance(model.tip_amp_gain.value, str)


# ===================================================================== 5. REST

@pytest.fixture
def _instrument_config_env(tmp_path, monkeypatch):
    """Point the service's instrument-config env var at an isolated tmp_path file,
    pre-populated so hardware-backend reads (the shared app fixture's hub defaults
    to the twin backend, but the endpoints must not care) never hit ModelIncomplete
    mid-test. See the module docstring's per-request-env assumption."""
    config_path = tmp_path / "instrument.toml"
    _write_toml(config_path)
    monkeypatch.setenv("AFM_LITHO_INSTRUMENT_CONFIG", str(config_path))
    return config_path


def test_get_instrument_returns_to_dict_plus_overrides_tail(
        client, _instrument_config_env):
    r = client.get("/api/instrument")
    assert r.status_code == 200, r.text[:300]
    j = r.json()
    assert "tip_amp_gain" in j
    assert j["tip_amp_gain"]["source"] in (s.value for s in FieldSource)
    assert "overrides_tail" in j
    assert isinstance(j["overrides_tail"], list)


def _overrides_log_for(config_path):
    # See the module docstring: no independent env var for the overrides log,
    # derived as a sibling of the config file.
    return config_path.parent / "instrument_overrides.jsonl"


def test_post_override_without_confirm_is_a_no_op(client, _instrument_config_env):
    overrides_log = _overrides_log_for(_instrument_config_env)
    toml_before = _instrument_config_env.read_text(encoding="utf-8")
    jsonl_before = (overrides_log.read_text(encoding="utf-8")
                    if overrides_log.exists() else "")
    r = client.post("/api/instrument/override", json={
        "field": "tip_amp_gain", "value": 40.0, "reason": "dry run",
    })
    assert r.status_code == 200, r.text[:300]
    j = r.json()
    assert j["applied"] is False
    assert "would_change" in j
    assert _instrument_config_env.read_text(encoding="utf-8") == toml_before
    jsonl_after = (overrides_log.read_text(encoding="utf-8")
                   if overrides_log.exists() else "")
    assert jsonl_after == jsonl_before            # nothing written to the log either


def test_post_override_with_confirm_persists_to_disk_and_survives_a_rebuild(
        client, _instrument_config_env, monkeypatch):
    overrides_log = _overrides_log_for(_instrument_config_env)
    r = client.post("/api/instrument/override", json={
        "field": "tip_amp_gain", "value": 40.0, "reason": "rig check",
        "confirm": True,
    })
    assert r.status_code == 200, r.text[:300]

    # 1) On-disk TOML carries the override, in [override], not just in memory.
    toml_text = _instrument_config_env.read_text(encoding="utf-8")
    assert "[override]" in toml_text
    assert "40.0" in toml_text.split("[override]", 1)[1]

    # 2) The overrides log has exactly one line with the documented keys.
    lines = overrides_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert {"ts", "client", "field", "old", "new", "reason"} <= set(rec)
    assert rec["field"] == "tip_amp_gain"
    assert rec["new"] == 40.0

    # 3) A REBUILT hub/model (independent of the live app's in-memory state)
    # reads the override back with source "override" -- an in-memory-only
    # implementation would fail this because a fresh Hub() re-reads the files.
    from afm_litho.service.hub import Hub

    rebuilt = Hub()
    assert rebuilt.host.instrument_model.tip_amp_gain.value == 40.0
    assert rebuilt.host.instrument_model.tip_amp_gain.source == FieldSource.override

    # 4) And the live app's own GET agrees.
    r2 = client.get("/api/instrument")
    assert r2.json()["tip_amp_gain"]["value"] == 40.0
    assert r2.json()["tip_amp_gain"]["source"] == "override"


def test_post_reload_reflects_a_manual_toml_edit(client, _instrument_config_env):
    r0 = client.get("/api/instrument")
    assert r0.json()["free_air_max_age_s"]["value"] == 30
    _write_toml(_instrument_config_env, free_air_max_age_s=99)
    r = client.post("/api/instrument/reload")
    assert r.status_code == 200, r.text[:300]
    j = r.json()
    # Tightened per review: the diff must NAME the changed field and its old
    # and new values, not just be present as an opaque blob.
    assert "diff" in j
    diff_text = json.dumps(j["diff"])
    assert "free_air_max_age_s" in diff_text
    assert "30" in diff_text
    assert "99" in diff_text
    r2 = client.get("/api/instrument")
    assert r2.json()["free_air_max_age_s"]["value"] == 99


def test_post_reload_with_malformed_toml_errors_and_keeps_the_previous_model(
        client, _instrument_config_env):
    r0 = client.get("/api/instrument")
    good_gain = r0.json()["tip_amp_gain"]["value"]
    _instrument_config_env.write_text("this is not [ valid toml", encoding="utf-8")
    r = client.post("/api/instrument/reload")
    assert r.status_code >= 400 or "error" in r.json()
    r2 = client.get("/api/instrument")
    assert r2.json()["tip_amp_gain"]["value"] == good_gain


def test_post_open_config_returns_path_and_calls_os_startfile(
        client, _instrument_config_env, monkeypatch):
    calls = []
    monkeypatch.setattr("os.startfile", lambda p: calls.append(p), raising=False)
    r = client.post("/api/instrument/open_config")
    assert r.status_code == 200, r.text[:300]
    j = r.json()
    assert j["path"] == str(_instrument_config_env)
    assert calls == [str(_instrument_config_env)]


def test_post_open_config_with_a_missing_file_errors(tmp_path, monkeypatch, client):
    missing = tmp_path / "does_not_exist.toml"
    monkeypatch.setenv("AFM_LITHO_INSTRUMENT_CONFIG", str(missing))
    r = client.post("/api/instrument/open_config")
    assert r.status_code >= 400 or "error" in r.json()


@pytest.fixture
def _latched_fault(monkeypatch):
    """Latch hub.fault directly (bypassing the missing-config path, which is
    exercised elsewhere) so the fault/clear tests are about the CLEAR endpoint,
    not about how a fault gets set in the first place."""
    from afm_litho.service.hub import hub

    previous = hub.fault
    hub.fault = {"reason": "model_incomplete", "at": 0.0, "state": "fault"}
    yield hub
    hub.fault = previous


def test_post_fault_clear_without_confirm_leaves_the_fault_latched(_latched_fault,
                                                                    client):
    r = client.post("/api/fault/clear", json={})
    assert r.status_code == 200, r.text[:300]
    assert _latched_fault.fault is not None
    assert _latched_fault.fault.get("reason") == "model_incomplete"


def test_post_fault_clear_with_confirm_clears_it(_latched_fault, client):
    assert _latched_fault.fault is not None            # sanity: it was latched
    r = client.post("/api/fault/clear", json={"confirm": True})
    assert r.status_code == 200, r.text[:300]
    assert _latched_fault.fault is None


# ===================================================================== 6. FLEX

def test_flex_get_instrument_matches_rest_get(client, _instrument_config_env):
    from afm_litho.service.hub import hub
    from afm_litho.service.requests import flex_door

    rest = client.get("/api/instrument").json()
    flex = flex_door.handle(hub, "getInstrument", None, role="read", ctx=None)
    assert flex == rest


def test_flex_get_instrument_is_read_socket_only(client, _instrument_config_env):
    """getInstrument must be dispatchable on the READ role -- the whole point of
    FLEX's read/command split (see flex_door's command-only check)."""
    from afm_litho.service.hub import hub
    from afm_litho.service.requests import flex_door

    # Must not raise -32601 "served by the COMMAND socket only".
    flex_door.handle(hub, "getInstrument", None, role="read", ctx=None)


def test_flex_get_last_park_matches_rest_get(client, _instrument_config_env):
    """E1: FLEX `getLastPark` returns exactly what `GET /api/park/last` returns,
    whatever the current park state (both read `hub.last_park`)."""
    from afm_litho.service.hub import hub
    from afm_litho.service.requests import flex_door

    rest = client.get("/api/park/last").json()
    flex = flex_door.handle(hub, "getLastPark", None, role="read", ctx=None)
    assert flex == rest


def test_flex_get_last_park_is_read_socket_only(client, _instrument_config_env):
    """getLastPark is a read verb -- dispatchable on the READ role, never -32601."""
    from afm_litho.service.hub import hub
    from afm_litho.service.requests import flex_door

    flex_door.handle(hub, "getLastPark", None, role="read", ctx=None)


# ============================================ 7. hardware backend REFUSING state

def test_hub_instrument_model_is_the_instrument_model(monkeypatch):
    """Section 10's Hub state line: `hub.instrument_model: InstrumentModel`, kept
    distinct from `hub.instrument` (the twin's `twin.physics.Instrument`
    calibration object, unchanged by this work -- spec fix, commit 97a4296)."""
    monkeypatch.delenv("AFM_LITHO_ARC_BACKEND", raising=False)
    monkeypatch.setenv("AFM_LITHO_BACKEND", "twin")
    from afm_litho.service.hub import Hub
    from afm_litho.twin.physics import Instrument as TwinInstrument

    hub = Hub()
    assert isinstance(hub.host.instrument_model, InstrumentModel)
    # Nit (review): confirm the rename did NOT touch the twin-physics object.
    assert isinstance(hub.host.instrument, TwinInstrument)


def test_hub_faults_on_hardware_backend_with_missing_instrument_config(
        tmp_path, monkeypatch):
    """service/fault-state consumer of ModelIncomplete (section 1): a hardware
    backend with no `instrument.toml` must NOT raise out of `Hub()` -- it starts
    REFUSING (`hub.fault` set) rather than crashing the service."""
    missing = tmp_path / "instrument.toml"
    monkeypatch.delenv("AFM_LITHO_BACKEND", raising=False)
    monkeypatch.setenv("AFM_LITHO_ARC_BACKEND", "hardware")
    monkeypatch.setenv("AFM_LITHO_INSTRUMENT_CONFIG", str(missing))
    from afm_litho.service.hub import Hub

    hub = Hub()
    assert hub.host.backend == "hardware"
    assert hub.fault is not None
    assert hub.fault.get("reason") == "model_incomplete"


# ================================ 8. platform is file-owned + re-derives (sec 14 item 4)

def test_file_platform_beats_the_passed_platform(tmp_path):
    """Section 14 item 4: the config file OWNS the platform. A cypher file loaded
    with a "twin" passed arg still derives the cypher park plan -- the passed arg
    is only the backend-intent gate + fallback, not the source of truth."""
    config_path = tmp_path / "instrument.toml"
    _write_toml(config_path, platform="cypher")
    overrides_log = tmp_path / "instrument_overrides.jsonl"
    model = load_model(platform="twin", startup_cal=None, config_path=config_path,
                       overrides_log=overrides_log)
    assert model.platform.value == "cypher"
    assert model.platform.source == FieldSource.config
    assert model.z_out_channel.value == "Height"


def test_platform_override_rederives_platform_default_fields(tmp_path):
    """Section 14 item 4: changing platform re-derives every platform_default
    field. A cypher model whose platform is overridden to mfp3d reports the new
    platform; z_out_channel is the Height alias on both, so it does not change."""
    config_path = tmp_path / "instrument.toml"
    _write_toml(config_path, platform="cypher")
    overrides_log = tmp_path / "instrument_overrides.jsonl"
    model = load_model(platform="cypher", startup_cal=None, config_path=config_path,
                       overrides_log=overrides_log)
    assert model.z_out_channel.value == "Height"

    updated = apply_override(model, "platform", "mfp3d", reason="rig", client="operator",
                             config_path=config_path, overrides_log=overrides_log)
    assert updated.platform.value == "mfp3d"
    assert updated.platform.source == FieldSource.override
    assert updated.z_out_channel.value == "Height", (
        "platform override did not re-derive the platform-default fields")

    # File-owned: a fresh load (a restart) agrees, the file wins over the passed arg.
    reloaded = load_model(platform="cypher", startup_cal=None, config_path=config_path,
                          overrides_log=overrides_log)
    assert reloaded.platform.value == "mfp3d"
    assert reloaded.z_out_channel.value == "Height"


def test_hub_hardware_platform_from_file_not_env_var(tmp_path, monkeypatch, caplog):
    """Section 14 item 4: the hardware backend takes the platform from
    instrument.toml, not AFM_LITHO_PLATFORM. A set env var that disagrees warns
    and the file wins."""
    config_path = tmp_path / "instrument.toml"
    _write_toml(config_path, platform="mfp3d")
    monkeypatch.delenv("AFM_LITHO_BACKEND", raising=False)
    monkeypatch.setenv("AFM_LITHO_ARC_BACKEND", "hardware")
    monkeypatch.setenv("AFM_LITHO_INSTRUMENT_CONFIG", str(config_path))
    monkeypatch.setenv("AFM_LITHO_PLATFORM", "cypher")   # disagrees with the file
    from afm_litho.service.hub import Hub

    with caplog.at_level("WARNING"):
        hub = Hub()
    assert hub.host.backend == "hardware"
    assert hub.fault is None, "a complete mfp3d config should not fault"
    assert hub.host.instrument_model.platform.value == "mfp3d", "the file must win over the env var"
    assert hub.host.instrument_model.z_out_channel.value == "Height"
    assert any("disagrees" in rec.message for rec in caplog.records), (
        "no warning logged for the env/file platform disagreement")


def test_platform_override_refused_while_engaged(client, _instrument_config_env):
    """Section 14 item 4: the override door refuses a platform change while the
    tip is engaged or a run is active (reason `engaged`), because the park plan
    changes underneath a live loop."""
    from afm_litho.service.hub import hub as live_hub

    prev = live_hub.host.engaged
    live_hub.host.engaged = True
    try:
        r = client.post("/api/instrument/override", json={
            "field": "platform", "value": "mfp3d", "confirm": True})
        assert r.status_code == 200, r.text[:300]
        j = r.json()
        assert j["applied"] is False, j
        assert j["reason"] == "engaged", j
    finally:
        live_hub.host.engaged = prev
    # Refused BEFORE apply_override: nothing written to the file's [override].
    toml_text = _instrument_config_env.read_text(encoding="utf-8")
    tail = toml_text.split("[override]", 1)[1] if "[override]" in toml_text else ""
    assert "platform" not in tail, "a refused platform change still hit the file"


def test_platform_override_via_route_rederives_z_out_channel(client, _instrument_config_env):
    """Section 14 item 4: an accepted platform override through the REST door
    reports the new platform on the next GET (z_out_channel stays Height)."""
    from afm_litho.service.hub import hub as live_hub

    live_hub.host.engaged = False        # not engaged: the change is allowed
    r0 = client.get("/api/instrument").json()
    assert r0["platform"]["value"] == "cypher"
    assert r0["z_out_channel"]["value"] == "Height"

    r = client.post("/api/instrument/override", json={
        "field": "platform", "value": "mfp3d", "confirm": True})
    assert r.status_code == 200, r.text[:300]
    assert r.json().get("applied") is True, r.json()

    r2 = client.get("/api/instrument").json()
    assert r2["platform"]["value"] == "mfp3d"
    assert r2["z_out_channel"]["value"] == "Height", (
        "the platform override did not re-derive z_out_channel over REST")
