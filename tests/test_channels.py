"""The channel table (hardware/channels.py): read aliases are one value, actuation
targets are per platform, and the literals still outside the table are listed."""

from __future__ import annotations

import re
from pathlib import Path


from afm_litho.hardware import channels
from afm_litho.service.instrument_model import load_model

SRC = Path(channels.__file__).resolve().parents[1]


def test_read_aliases_are_single_strings() -> None:
    # A read alias that became {"cypher": ..., "mfp3d": ...} would be a per-platform
    # branch in disguise; Asylum resolves these names itself.
    for name, value in channels.READ_ALIASES.items():
        assert isinstance(value, str) and value, f"read alias {name!r} must be one string"


def test_z_out_is_the_height_alias_on_every_platform() -> None:
    assert channels.Z_OUT == channels.HEIGHT == "Height"


def test_module_constants_match_the_read_alias_table() -> None:
    constants = {k: v for k, v in vars(channels).items()
                 if k.isupper() and not k.startswith("_") and k != "Z_OUT"
                 and isinstance(v, str)}
    assert sorted(constants.values()) == sorted(channels.READ_ALIASES.values())


def test_both_amplitude_spellings_are_kept() -> None:
    assert channels.READ_ALIASES["amplitude"] == "Amplitude"
    assert channels.READ_ALIASES["amplitude_lowercase"] == "amplitude"


def test_instrument_model_default_is_the_z_out_channel(tmp_path: Path) -> None:
    # The twin needs no config file; hardware platforms refuse to load without one.
    model = load_model(platform="twin", startup_cal=None,
                       config_path=tmp_path / "missing.toml",
                       overrides_log=tmp_path / "overrides.jsonl")
    assert model.z_out_channel.value == channels.Z_OUT


# A channel name spelled as a quoted string anywhere in src/ but the table fails the test.
# Exempt: twin/ (its own vocabulary), goldens/ (the independent oracle) and igor_stub.py
# (the fake Igor, and its v3 file, speak Igor's names deliberately).
_LITERAL = re.compile(
    r"[\"'](Height|Deflection|DeflectionValue|ZSensor|XSensor|YSensor|DetectorSum|"
    r"Amplitude|Output\.[A-Z]|Input\.[A-Za-z]+)[\"']")
_EXEMPT = ("twin/", "goldens/", "hardware/channels.py", "hardware/igor_stub.py", "hardware/igor_stub_v3.py")


def literals_outside_the_table() -> dict[str, int]:
    found: dict[str, int] = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel.startswith(_EXEMPT):
            continue
        count = len(_LITERAL.findall(path.read_text(encoding="utf-8")))
        if count:
            found[rel] = count
    return found


def test_no_channel_literals_outside_the_table() -> None:
    assert literals_outside_the_table() == {}, literals_outside_the_table()
