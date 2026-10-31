"""The command guard: one choke point every hardware-moving verb goes through,
on both doors (REST/cockpit and FLEX), with no second implementation anywhere.

Truth document: docs/INSTRUMENT_MODEL_SPEC.md sections 2 and 10. Pure functions
over an `InstrumentModel` + the request + the live free-air state -- they never
touch the instrument and never modify the request. Every refusal is a
`GuardRefused` (a `CommandError`, JSON-RPC code -32050) carrying a stable
`.reason` string from the fixed vocabulary section 10 names.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .commands.errors import CommandError

GUARD_CODE = -32050


class GuardRefused(CommandError):
    """A command-guard refusal. `.reason` is one of the fixed strings section 10
    names (`amp_gain_missing`, `tip_voltage_out_of_window`, ...)."""

    def __init__(self, reason: str, message: str | None = None,
                extra: dict | None = None):
        data = {"reason": reason}
        if extra:
            data.update(extra)
        super().__init__(GUARD_CODE, message or reason, data)
        self.reason = reason


@dataclass(frozen=True)
class FreeAir:
    value_v: float
    ts: float
    mode: str                # "contact" | "ac"
    withdrawn: bool
    # Withdrawn deflection (V), the park-verify reference. Equals value_v outside AC;
    # in AC value_v is the free amplitude. None on records that predate the field.
    defl_v: float | None = None


@dataclass(frozen=True)
class ParkPlan:
    z_channel: str              # the Z drive the park confirms on (Height, by the model)
    park_v: float
    tol_v: float
    verify_channels: tuple


def _refuse(reason: str, message: str | None = None, **extra) -> None:
    raise GuardRefused(reason, message, extra or None)


def _check_fault(fault) -> None:
    if fault:
        _refuse("fault_latched", "the instrument fault is latched; clear it first")


def _num(value, default: float = 0.0) -> float:
    """`float(value)`, coercing `None` (a legitimately-absent optional field, or
    a malformed request that set one to null) to `default` rather than letting
    `float(None)` raise -- see `commanded_voltages`'s `v0`/`v_k` note (adversarial
    review finding S3: a chiral profile with `v0: null` must refuse cleanly, not
    500)."""
    return default if value is None else float(value)


def commanded_voltages(obj) -> list[float]:
    """Every commanded tip voltage extremum for one LithoObject, enumerated
    EXACTLY as `litho.compiler.compile_object` derives `v` (adversarial review
    finding M1, docs/INSTRUMENT_MODEL_SPEC.md section 10): only `obj.voltage`
    and a chiral `v0 +/- v_k` used to be checked here, but `compile_object`
    (compiler.py ~533-539) builds the Output.A wave from `obj.v_profile` when
    present (ANY value in that list reaches the DAC, not just `voltage`), else a
    linspace from `obj.voltage` to `obj.voltage_end` (defaulting to `voltage`
    when unset) -- and BOTH are settable with no range check via `POST
    /api/design/op {op: "set_props"}` (routers/design.py). Both fields are
    checked here regardless of which one the compiler will actually use for a
    GIVEN write, so a value that can never legitimately reach the tip is refused
    up front rather than only in the specific case that exercises it.

    A filled shape's expanded strokes (`compiler.expand`) inherit only
    `voltage`/`voltage_end` from the parent object (never `v_profile` -- see
    `expand`'s `LithoObject(...)` construction), so checking the parent's own
    fields here already covers every fill stroke; there is no separate
    fill-voltage field to enumerate.

    The `profile` dict's `v0`/`v_k` (set by `pens.chiral_profile` alongside the
    `v_profile` array it samples) are ALSO checked, for defense in depth and
    backward compatibility with hand-built chiral fixtures that set `profile`
    without `v_profile` -- a real chiral wire's `v_profile` is the actual
    compiled source and already covers its extrema, but a v0/v_k pair that
    disagrees with a stale/hand-set v_profile must still be caught.
    """
    out: list[float] = []
    profile = getattr(obj, "profile", None)
    if isinstance(profile, dict) and profile.get("kind") == "chiral":
        v0_raw = profile.get("v0")
        if v0_raw is None:
            # A chiral wire's centre voltage is not an optional field the way
            # `v_k` is (0 chirality is a legitimate, safe default; no centre
            # voltage is not) -- a missing/null v0 cannot be silently defaulted
            # to something that reads as "safe", so it refuses (S3), rather than
            # coercing to a value that might slip inside the window by luck.
            raise ValueError(
                f"object {getattr(obj, 'name', '?')!r}: chiral profile has no v0")
        v0 = float(v0_raw)
        v_k = _num(profile.get("v_k"), 0.0)
        out.extend([v0 - v_k, v0 + v_k])
    # ALWAYS enumerate BOTH `v_profile` extrema (when present) AND
    # `voltage`/`voltage_end`, never one-or-the-other (adversarial review
    # finding M1, spec 13.1 item 9): `compile_object` uses `v_profile` for a
    # GIVEN write when it is set, else a linspace(`voltage`, `voltage_end`), but
    # all three fields are independently settable via `POST /api/design/op
    # {op:"set_props"}` with no per-field range check, and `compiler.expand`
    # splits a filled shape/electrode into fill strokes that carry
    # `voltage`/`voltage_end` (NO `v_profile`) which `compile_object` then
    # linspaces. Checking `voltage`/`voltage_end` only in the `v_profile`-absent
    # branch let an object with a benign `v_profile` and `voltage_end=300` slip
    # past `check_write` and the set_props door and reach the DAC. Enumerating
    # every source unconditionally is a safe SUPERSET of whatever a write will
    # actually command, so the guard refuses up front.
    v_profile = getattr(obj, "v_profile", None)
    if v_profile:
        values = [_num(x, 0.0) for x in v_profile]
        out.extend([min(values), max(values)])
    v0 = _num(getattr(obj, "voltage", None), 0.0)
    voltage_end = getattr(obj, "voltage_end", None)
    v1 = v0 if voltage_end is None else _num(voltage_end, v0)
    out.extend([v0, v1])
    return out


def check_design_voltages(model, objs, *, fault: object | None = None) -> None:
    """Refuse any design object whose commanded tip voltage would leave the
    model's `tip_voltage_window_v`, or which, divided by the tip amplifier gain,
    would exceed the DAC rail (`|v / tip_amp_gain| > dac_rail_v`).

    The ONE choke point every design-mutating door runs (spec 13.1 item 19):
    REST `POST /api/design/op {op:"set_props"}` (via `routers/design.
    _check_voltage_window`, a thin wrapper), the FLEX `setObjects` and
    `loadPattern` doors, and the SVG import door -- BEFORE the design is
    installed, so an out-of-window or over-rail value never even enters the
    saved pattern. Before this landed only the write doors and set_props
    refused, so a script could push voltage=300 into the design over FLEX and
    the cockpit's Write would then be the one to trip on it.

    Voltage sources are enumerated by `commanded_voltages` -- exactly the set
    `litho.compiler.compile_object` can emit (`voltage`, `voltage_end`,
    `v_profile` extrema, chiral `v0 +/- v_k`, and a filled shape's inherited
    `voltage`/`voltage_end`). A malformed source (non-numeric, or a chiral with
    no `v0`) refuses as `tip_voltage_out_of_window` rather than raising, the same
    way `check_write` handles it -- it can never be proven inside the window.

    Refusals carry the offending object id and value in the payload
    (`object_id`, `value`), so a door can name the object without re-deriving it.
    Pure: `objs` and the request are never modified; `model is None` is a no-op
    (no model loaded yet -- nothing to check against, matching `check_write`'s
    callers' twin/no-model tolerance)."""
    _check_fault(fault)
    if model is None:
        return
    window = model.tip_voltage_window_v.value
    v_min, v_max = float(window[0]), float(window[1])
    dac_rail_v = float(model.dac_rail_v.value)
    amp_gain_raw = model.tip_amp_gain.value
    amp_gain = None if amp_gain_raw is None else float(amp_gain_raw)
    for obj in objs:
        oid = getattr(obj, "name", "?")
        try:
            voltages = commanded_voltages(obj)
        except (TypeError, ValueError) as exc:
            _refuse("tip_voltage_out_of_window",
                   f"object {oid!r}: could not determine its commanded tip "
                   f"voltage ({exc})", object_id=oid)
            continue                                     # pragma: no cover -- _refuse raises
        for v in voltages:
            if not (v_min <= v <= v_max):
                _refuse("tip_voltage_out_of_window",
                       f"object {oid!r}: commanded tip voltage {v:g} V outside "
                       f"[{v_min:g}, {v_max:g}]", object_id=oid, value=v)
            # The rail can only bite when the window is wider than
            # `dac_rail_v * amp_gain` (e.g. a config that opened the window past
            # what the amplifier can drive); with a missing amp gain
            # (model_incomplete on the hardware backend) it cannot be computed,
            # so it is skipped here -- the engage/write doors refuse that model
            # with `model_incomplete` before any hardware move regardless.
            if amp_gain is not None and abs(v / amp_gain) > dac_rail_v:
                _refuse("dac_rail_exceeded",
                       f"object {oid!r}: |{v:g}/{amp_gain:g}| exceeds the "
                       f"{dac_rail_v:g} V DAC rail", object_id=oid, value=v)


def _check_free_air(model, mode: str, free_air: FreeAir | None, *, now: float) -> None:
    if free_air is None:
        _refuse("free_air_missing", "no free-air baseline is on record")
    if not free_air.withdrawn:
        _refuse("free_air_stale",
               "the free-air reading was not taken withdrawn")
    max_age = float(model.free_air_max_age_s.value)
    if (now - float(free_air.ts)) > max_age:
        _refuse("free_air_stale",
               f"the free-air reading is older than {max_age:g} s")
    if free_air.mode != mode:
        _refuse("free_air_stale",
               f"the free-air reading was taken in {free_air.mode!r} mode, "
               f"not {mode!r}")


def _check_setpoint_band(model, mode: str, setpoint: float,
                         free_air: FreeAir) -> None:
    """Band check on `setpoint` relative to the free-air baseline.

    AC: `setpoint / free_air.value_v` (an absolute amplitude target, checked as a
    FRACTION of the free-air amplitude -- section 10's worked example).

    CONTACT (spec correction, 672a8b1): `setpoint - free_air.value_v` is checked
    against `contact_setpoint_band_v` -- the RELATIVE value, per section 2's
    prose. `setpoint` reaching this function is the ABSOLUTE value either way
    (a free-air ask: `free_air.value_v + setpoint`, computed by `litho_cmds`
    before the guard runs; an absolute ask: as given), so this re-derives
    the relative offset to compare against the band. (An earlier revision of
    this function compared `setpoint` directly, driven by a since-corrected test
    fixture; a real rig's free-air baseline is negative, e.g. -0.16..-0.33 V, so
    a direct comparison would refuse every real write.)"""
    if mode == "ac":
        lo, hi = model.ac_setpoint_fraction_band.value
        amp = float(free_air.value_v)
        fraction = (setpoint / amp) if amp else math.inf
        if not (lo <= fraction <= hi):
            _refuse("setpoint_out_of_band",
                   f"AC setpoint fraction {fraction:g} outside [{lo:g}, {hi:g}]")
    else:
        lo, hi = model.contact_setpoint_band_v.value
        rel = setpoint - float(free_air.value_v)
        if not (lo <= rel <= hi):
            _refuse("setpoint_out_of_band",
                   f"setpoint - free_air = {rel:g} V outside [{lo:g}, {hi:g}]")


def check_write(model, pattern, object_ids, setpoint: float,
                amp_gain: float | None, free_air: FreeAir | None, *, now: float,
                fault: object | None = None,
                setpoint_relative: float | None = None) -> None:
    """Refuse a write that would put the instrument, the sample, or the tip in
    danger. Consumers (both doors): `litho_cmds._litho_write_body`.

    `setpoint_relative`: the setpoint of a free-air ask, band-checked directly when
    no baseline is on record (None when the ask was absolute)."""
    _check_fault(fault)

    if amp_gain is None:
        _refuse("amp_gain_missing", "amp_gain is required and was not given")
    model_gain = model.tip_amp_gain.value
    if model_gain is None:
        _refuse("model_incomplete", "the instrument model has no tip_amp_gain")
    if float(amp_gain) != float(model_gain):
        _refuse("amp_gain_mismatch",
               f"amp_gain {amp_gain!r} does not match the model's "
               f"{model_gain!r}")

    if not isinstance(object_ids, (list, tuple)):
        raise TypeError(
            "check_write: object_ids must be an explicit list of object ids, "
            f"got {type(object_ids).__name__}")

    by_name = {getattr(o, "name", None): o for o in getattr(pattern, "objects", [])}
    if not object_ids:
        _refuse("unknown_object", "no objects were selected")
    objs = []
    for oid in object_ids:
        obj = by_name.get(oid)
        if obj is None:
            _refuse("unknown_object", f"no object named {oid!r} in the pattern")
        if not getattr(obj, "enabled", True):
            _refuse("object_disabled", f"object {oid!r} is disabled")
        objs.append(obj)

    window = model.tip_voltage_window_v.value
    v_min, v_max = float(window[0]), float(window[1])
    dac_rail_v = float(model.dac_rail_v.value)
    for obj in objs:
        try:
            voltages = commanded_voltages(obj)
        except (TypeError, ValueError) as exc:
            # A malformed voltage source (a non-numeric v_profile entry, an
            # unconvertible voltage/voltage_end) must refuse, not fail open to
            # a 500 (adversarial review finding S3) -- it can never be proven
            # inside the window, so it is refused the same way an out-of-window
            # value is.
            _refuse("tip_voltage_out_of_window",
                   f"object {getattr(obj, 'name', '?')!r}: could not determine "
                   f"its commanded tip voltage ({exc})")
            continue                                     # pragma: no cover -- _refuse raises
        for v in voltages:
            if not (v_min <= v <= v_max):
                _refuse("tip_voltage_out_of_window",
                       f"commanded tip voltage {v:g} V outside "
                       f"[{v_min:g}, {v_max:g}]")
            if abs(v / float(amp_gain)) > dac_rail_v:
                _refuse("dac_rail_exceeded",
                       f"|{v:g}/{amp_gain:g}| exceeds the {dac_rail_v:g} V DAC rail")

    # Litho is always contact. No free-air age/presence check for a write: the hardware
    # run reads free air before every object (docs/PER_OBJECT_FREE_AIR_SPEC.md). The band
    # is relative to a baseline; with none on record a REST relative value is checked
    # against it directly, the same result as setpoint - free_air with a baseline.
    if free_air is not None:
        _check_setpoint_band(model, "contact", setpoint, free_air)
    elif setpoint_relative is not None:
        lo, hi = model.contact_setpoint_band_v.value
        rel = float(setpoint_relative)
        if not (lo <= rel <= hi):
            _refuse("setpoint_out_of_band",
                   f"relative setpoint {rel:g} V outside [{lo:g}, {hi:g}]")


def check_approach(model, mode: str, setpoint: float, free_air: FreeAir | None,
                   *, now: float, fault: object | None = None) -> None:
    """Refuse an engage (approach, scan-with-engage, write set-down) with a stale
    or missing free-air baseline, or a setpoint outside its band."""
    _check_fault(fault)
    _check_free_air(model, mode, free_air, now=now)
    _check_setpoint_band(model, mode, setpoint, free_air)


def check_sum_ceiling(model, setpoint_abs_v: float, sum_v: float | None) -> None:
    """Refuse a contact engage whose deflection setpoint is unsafe against a FRESH
    DetectorSum read (docs/INSTRUMENT_MODEL_SPEC.md section 12).

    Two refusals, checked in this order so the ceiling arithmetic is only ever
    reached with a trustworthy SUM:
      * `sum_missing` -- `sum_v` is None, or below `model.sum_floor_v` (0.5 V): the
        detector is effectively blind, so `sum_ceiling_frac * sum_v` is meaningless.
        The floor is a STRICT `<`: `sum_v == sum_floor_v` passes.
      * `setpoint_exceeds_sum` -- `abs(setpoint_abs_v) > model.sum_ceiling_frac
        (0.9) * sum_v`. Boundary INCLUSIVE: `abs(setpoint) == 0.9*sum` passes, one
        step past it refuses (12.1's worked example: 0.45 with SUM 0.5 passes, 0.46
        refused).

    Pure: called at EVERY contact engage (approach, scan, write set-down) on both
    doors, AFTER `check_write`/`check_approach`, with a freshly-read SUM."""
    floor = float(getattr(model.sum_floor_v, "value", model.sum_floor_v))
    if sum_v is None or float(sum_v) < floor:
        _refuse("sum_missing",
               f"DetectorSum {sum_v!r} is missing or below the {floor:g} V floor")
    frac = float(getattr(model.sum_ceiling_frac, "value", model.sum_ceiling_frac))
    ceiling = frac * float(sum_v)
    if abs(float(setpoint_abs_v)) > ceiling:
        _refuse("setpoint_exceeds_sum",
               f"|setpoint| {abs(float(setpoint_abs_v)):g} V exceeds "
               f"{frac:g}*SUM ({ceiling:g} V; "
               f"SUM={float(sum_v):g} V)")


def check_scan(model, size_um: float, x_offset_um: float, y_offset_um: float,
              angle_deg: float, *, fault: object | None = None) -> None:
    """Refuse a scan frame (offset +/- size/2, rotated by angle_deg about its own
    centre) that would leave `model.xy_range_um`."""
    _check_fault(fault)
    half = float(size_um) / 2.0
    theta = math.radians(float(angle_deg))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    max_x = max_y = 0.0
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            cx, cy = sx * half, sy * half
            rx = cx * cos_t - cy * sin_t
            ry = cx * sin_t + cy * cos_t
            max_x = max(max_x, abs(rx + float(x_offset_um)))
            max_y = max(max_y, abs(ry + float(y_offset_um)))
    xy_range = model.xy_range_um.value
    if isinstance(xy_range, (list, tuple)):
        range_x = float(xy_range[0])
        range_y = float(xy_range[1]) if len(xy_range) > 1 else range_x
    else:
        range_x = range_y = float(xy_range)
    if max_x > range_x / 2.0 or max_y > range_y / 2.0:
        _refuse("scan_out_of_range",
               f"scan frame reaches ({max_x:g}, {max_y:g}) um, outside the "
               f"[{range_x:g}, {range_y:g}] um range")


def park_plan(model) -> ParkPlan:
    """The Z channel the park confirms on and the sensors it reads, from the model alone
    (no live state)."""
    return ParkPlan(
        z_channel=model.z_out_channel.value,
        park_v=float(model.z_park_v.value),
        tol_v=float(model.z_park_tol_v.value),
        verify_channels=tuple(model.z_verify_channels.value),
    )
