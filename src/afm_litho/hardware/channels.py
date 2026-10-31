"""The one table of Asylum channel names. Every name is one value on every platform:
Asylum resolves ``Height`` and ``Deflection`` itself, and Z is driven through ``Height`` on
both instruments. The twin's own names (``Z%Output``, ``Fast%Input``) are not here."""

from __future__ import annotations

# --- 1. Read aliases: same string on every platform ---
# tests/test_pinned_constants.py spells this table out; a per-platform value fails there.

HEIGHT = "Height"
DEFLECTION = "Deflection"
DEFLECTION_VALUE = "DeflectionValue"
Z_SENSOR = "ZSensor"
X_SENSOR = "XSensor"
Y_SENSOR = "YSensor"
DETECTOR_SUM = "DetectorSum"
# Both spellings are in use today; each call site keeps the one it already sends.
AMPLITUDE = "Amplitude"
AMPLITUDE_LOWERCASE = "amplitude"
# The withdraw verify reads Output.Z under this vendor alias; it must stay lowercase.
HEIGHT_LOWERCASE = "height"
# Fixed output channels: the same on every platform.
# Output limits (V) every PIS loop is armed with: the Z loop and the XY tracking loops.
LOOP_OUTPUT_LIMITS_V = (-10.0, 150.0)

OUTPUT_A = "Output.A"
OUTPUT_X = "Output.X"
OUTPUT_Y = "Output.Y"
# Input channels.
# The measured tip bias: front BNC In2, teed off the bias line. LATERAL is its capture-bank partner.
USER_IN2 = "UserIn2"
LATERAL = "Lateral"
INPUT_A = "Input.A"
INPUT_FAST = "Input.Fast"

READ_ALIASES: dict[str, str] = {
    "height": HEIGHT,
    "deflection": DEFLECTION,
    "deflection_value": DEFLECTION_VALUE,
    "z_sensor": Z_SENSOR,
    "x_sensor": X_SENSOR,
    "y_sensor": Y_SENSOR,
    "detector_sum": DETECTOR_SUM,
    "amplitude": AMPLITUDE,
    "amplitude_lowercase": AMPLITUDE_LOWERCASE,
    "height_lowercase": HEIGHT_LOWERCASE,
    "output_a": OUTPUT_A,
    "output_x": OUTPUT_X,
    "output_y": OUTPUT_Y,
    "user_in2": USER_IN2,
    "lateral": LATERAL,
    "input_a": INPUT_A,
    "input_fast": INPUT_FAST,
}

# --- 2. The Z drive: the same on every platform ----------------------------------------

# Both platforms drive loop 2 through the Height alias; the MFP-3D resolves it to Output.Z
# (ir_readAlias("Height"), 2026-10-27). Loop 2 as its Z loop is still to confirm.
Z_OUT = HEIGHT
