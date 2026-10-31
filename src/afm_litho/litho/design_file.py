"""The design file: a design's objects as JSON records, and back.

A record stores the object's geometry and settings, not the points the AFM would
write. A chiral wire is a spine with a modulation added: whatever its kind, its
record is the spine and the chiral parameters, never its points, and it is built
again on load (one with no spine, from the old panel's pen, has nothing else and
keeps them). The build is `pens.bake_chiral`, the one the cockpit uses, at the
pitch the operator asked for, so a wire reloads as it is in memory; a `v_profile`
set by hand on a chiral wire is replaced by the rebuilt one, as it is by the next
edit in memory. An
object with a `bezier` is flattened again on load when that gives its points back
within `COORD_TOL_UM` at equal length; otherwise, and for anything with no other
definition, the points are stored. A record that carries `points_um` is loaded
with those points as they are.

Before a Bezier is flattened or a chiral path is built on load, the size of that
one build is estimated by arithmetic, and an object far beyond anything the
instrument can write is refused (`object_too_large`). There is no limit on the
points of a design.

A field is written only when it carries something: one that is not set (null, 0,
false) is left out, and so is a chiral path's interpolation step when it is the
one the rebuild computes. Voltage, speed and enabled are always written.
"""

from __future__ import annotations

import dataclasses
import json
import math
import threading
from itertools import chain

import numpy as np

from .compiler import LithoObject, Pattern

FORMAT = "afm-litho-design"
VERSION = 1
# The rebuild a file relies on: `BezierPath.flatten(0.004)` and
# `pens.chiral_profile`. Raised when either changes what it returns.
BUILDER = 1
KINDS = ("path", "shape", "electrode", "rect", "taper")
FILLED_KINDS = ("shape", "electrode", "rect", "taper")
BEZIER_FLATTEN_TOL_UM = 0.004
MAX_EXTENT_UM = 2000.0
COORD_TOL_UM = 1e-5
# A guard against one absurd curve, not a limit on a design: 200 nodes with
# handles anywhere in a 90 um scan range estimate to about 52,000 points. A curve
# the service itself holds past this is saved with its points, so it reopens. A
# chiral wire is held to the builder's own limit, `pens._CHIRAL_MAX_POINTS`.
MAX_BEZIER_POINTS = 250_000
_FIELDS = {f.name: f for f in dataclasses.fields(LithoObject)}
_FILL = {"spacing_um": "fill_spacing_um", "kind": "fill_kind",
         "angle_deg": "fill_angle_deg", "centre_um": "fill_centre_um",
         "inset_um": "inset_um", "outline_pass": "outline_pass"}
_FILL_KEY = {attr: f"fill.{key}" for key, attr in _FILL.items()}
_DIRECT = ("name", "kind", "points_um", "bezier", "holes_um", "voltage", "voltage_end",
           "v_profile", "speed_um_s", "enabled", "interpolation_um", "end_binding")
# Every key a file may hold.
# Fields of the removed electrode pen; files saved with them still open and the values are dropped.
_LEGACY_EXTRA = ("centerline_um", "half_width_um", "generator")
DOCUMENT_KEYS = ("format", "version", "builder", "name", "saved_at", "object_count",
                 "pattern_name", "frame", "objects")
RECORD_KEYS = (*_DIRECT, "fill", "chiral", "extra")
CHIRAL_KEYS = ("lambda_um", "y_amp_um", "v_k", "phase_deg", "axis_um", "profile")
FILL_KEYS = tuple(_FILL)
_ALWAYS = ("voltage", "speed_um_s", "enabled")
_ALWAYS_FILL = ("spacing_um", "kind")
_CHIRAL = ("axis_um", "profile")
_CHIRAL_PARAMS = ("lambda_um", "y_amp_um", "v_k", "phase_deg")
_EXTRA = tuple(n for n in _FIELDS
               if n not in _DIRECT and n not in _CHIRAL and n not in _FILL.values())
_POINT_LISTS = ("points_um", "axis_um")
_NUMBER_LISTS = ("v_profile",)
_NUMBERS = ("voltage", "speed_um_s", "interpolation_um", "fill_spacing_um",
            "inset_um", "width0_um", "width1_um")
_POSITIVE = ("fill_spacing_um", "speed_um_s", "interpolation_um")
_OPTIONAL_NUMBERS = ("voltage_end", "fill_angle_deg")
_BOOLS = ("enabled", "outline_pass")
_DICTS = ("bezier", "profile", "end_binding")

# One C-level call over a long list holds the GIL for its whole length, so long
# lists are encoded, converted and compared in slices and the thread yields between.
_JSON_CHUNK = 4000
_ARRAY_CHUNK = 20000


class BuildCancelled(Exception):
    """The build was told to stop between two records."""


class DesignFileError(ValueError):
    """A document or record that cannot be loaded. `reason` is `not_a_design_file`,
    `unsupported_version`, `file_too_large`, `bad_record` or `object_too_large`;
    `object_name` and `field` name the record."""

    def __init__(self, reason: str, message: str, *, object_name: str | None = None,
                 field: str | None = None):
        super().__init__(message)
        self.reason = reason
        self.object_name = object_name
        self.field = field


def unique_name(base: str, existing) -> str:
    """`base`, or `base-k` with the smallest k >= 2 not in `existing`."""
    taken = set(existing)
    name, k = base, 1
    while name in taken:
        k += 1
        name = f"{base}-{k}"
    return name


def unique_names(bases, existing) -> list[str]:
    """`unique_name` for each of `bases` in turn, every name given joining
    `existing` for the next, in time linear in the number of names."""
    taken = set(existing)
    last: dict[str, int] = {}
    out = []
    for base in bases:
        name = base
        if name in taken:
            k = last.get(base, 1)
            while name in taken:
                k += 1
                name = f"{base}-{k}"
            last[base] = k
        taken.add(name)
        out.append(name)
    return out


def _default(name: str):
    f = _FIELDS[name]
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return None if f.default is dataclasses.MISSING else f.default


def _is_default(name: str, value) -> bool:
    default = _default(name)
    try:
        if isinstance(value, float) and isinstance(default, float):
            return (value == default
                    and math.copysign(1.0, value) == math.copysign(1.0, default))
        return bool(value == default)
    except Exception:                                      # noqa: BLE001
        return False


# ------------------------------------------------- building from the source --

def flatten_bezier(bezier: dict) -> list[tuple[float, float]]:
    from .bezier import BezierPath
    pts = BezierPath.from_dict(bezier).flatten(BEZIER_FLATTEN_TOL_UM)
    return [(float(x), float(y)) for x, y in pts]


def chiral_step(params: dict) -> float:
    """The interpolation step the chiral control gives a path."""
    from .pens import chiral_spacing_um
    lam = max(float(params["lambda_um"]), 1e-4)
    return chiral_spacing_um(lam, float(params["y_amp_um"]))


def flatten_points(bezier: dict) -> int:
    """About how many points `flatten_bezier` returns, from each segment's control
    polygon: each halving of a cubic divides its flatness by about four."""
    from .bezier import BezierPath
    total = 1
    for p0, c1, c2, p3 in BezierPath.from_dict(bezier).segments():
        chord = p3 - p0
        length = float(np.hypot(*chord))
        if length < 1e-12:
            flatness = float(max(np.hypot(*(c1 - p0)), np.hypot(*(c2 - p0))))
        else:
            normal = np.array([-chord[1], chord[0]]) / length
            flatness = float(max(abs(np.dot(c1 - p0, normal)),
                                 abs(np.dot(c2 - p0, normal))))
        depth = 0
        if flatness > BEZIER_FLATTEN_TOL_UM:
            depth = min(18, math.ceil(math.log(flatness / BEZIER_FLATTEN_TOL_UM, 4)) + 1)
        total += 2 ** depth
    return total


def chiral_samples(axis, params: dict) -> int:
    """How many samples `chiral_profile` takes for this spine and these
    parameters, by its own rule."""
    from .pens import is_closed
    arr = np.asarray(axis, dtype=np.float64)
    length = float(np.hypot(*np.diff(arr, axis=0).T).sum())
    if is_closed(arr):
        length += float(np.hypot(*(arr[0] - arr[-1])))
    lam = max(float(params["lambda_um"]), 1e-4)
    density = math.hypot(1.0, 2 * math.pi * float(params["y_amp_um"]) / lam)
    per_period = max(24, int(density / 0.04 / 4) + 1)
    return max(len(arr), int(length / lam * per_period) + 2)


def _too_large(name: str, field: str, estimate: int, limit: int, what: str):
    return DesignFileError(
        "object_too_large",
        f"object {name!r}: {field}: building it would take about {estimate:,} {what}; "
        f"the most one object may take is {limit:,}", object_name=name, field=field)


def _phase_deg(phase_rad: float) -> float:
    """The angle in degrees that `math.radians` turns back into exactly
    `phase_rad`, when a neighbouring float does where the plain conversion does not."""
    deg = math.degrees(phase_rad)
    for candidate in (deg, math.nextafter(deg, math.inf), math.nextafter(deg, -math.inf)):
        if math.radians(candidate) == phase_rad:
            return candidate
    return deg


def _chiral_params(obj) -> dict | None:
    """The chiral settings of an object as the file holds them (`pens.
    chiral_settings`: the pitch is the one asked for), or None when it has none."""
    from .pens import chiral_settings
    settings = chiral_settings(obj)
    if settings is None:
        return None
    return {"lambda_um": settings["lambda_um"], "y_amp_um": settings["y_amp_um"],
            "v_k": settings["v_k"],
            "phase_deg": _phase_deg(float(obj.profile["phase_rad"]))}


def _close(a, b, tol: float) -> bool:
    """Two lists of numbers or points of equal length, nowhere further apart than
    `tol`."""
    try:
        if len(a) != len(b):
            return False
        for start in range(0, len(a), _ARRAY_CHUNK):
            x = np.asarray(a[start:start + _ARRAY_CHUNK], dtype=np.float64)
            y = np.asarray(b[start:start + _ARRAY_CHUNK], dtype=np.float64)
            if x.shape != y.shape or not (np.abs(x - y) <= tol).all():
                return False
        return True
    except Exception:                                      # noqa: BLE001
        return False


def _is_spine(points) -> bool:
    return isinstance(points, (list, tuple)) and len(points) >= 2


def _rebuilds(obj: LithoObject) -> tuple[bool, bool]:
    """`(source, bezier)`: whether the record can leave the object's points out,
    and whether the Bezier alone then gives them (for a chiral path, its spine).

    A chiral wire with a spine always can; its Bezier stands for the spine only
    when it flattens to exactly that, because a spine moved by the Bezier's
    rounding shifts the phase along the whole wire. A Bezier object can when
    flattening gives its points back within the tolerance. A Bezier the load
    would refuse to flatten never stands in for points."""
    try:
        flat = None
        if obj.bezier is not None and flatten_points(obj.bezier) <= MAX_BEZIER_POINTS:
            flat = flatten_bezier(obj.bezier)
        if _chiral_params(obj) is not None:
            if _is_spine(flat) and (obj.axis_um is None
                                    or _close(flat, obj.axis_um, 0.0)):
                return True, True
            return _is_spine(obj.axis_um), False
        if flat is not None and obj.profile is None:
            return _close(flat, obj.points_um, COORD_TOL_UM), True
        return False, False
    except Exception:                                      # noqa: BLE001
        return False, False


class RebuildCache:
    """Remembers `_rebuilds` for objects whose lists are the same list objects as
    before, so a save does not flatten every Bezier again. Holding the lists keeps
    their ids from being reused; an entry no document has used for two documents
    is dropped. Safe to share between threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._entries: dict = {}
        self._generation = 0

    @staticmethod
    def _key(obj: LithoObject):
        held = (obj.points_um, obj.bezier, obj.axis_um, obj.profile, obj.v_profile)
        return (*map(id, held), obj.kind, obj.voltage), held

    def rebuilds(self, obj: LithoObject) -> tuple[bool, bool]:
        key, held = self._key(obj)
        with self._lock:
            hit = self._entries.get(key)
        result = hit[1] if hit else _rebuilds(obj)
        with self._lock:
            self._entries[key] = (held, result, self._generation)
        return result

    def roll(self) -> None:
        with self._lock:
            self._generation += 1
            oldest = self._generation - 2
            self._entries = {k: v for k, v in self._entries.items() if v[2] >= oldest}


def object_to_record(obj: LithoObject, cache: RebuildCache | None = None) -> dict:
    """The record of one object. Large lists are referenced, not copied."""
    from_source, from_bezier = cache.rebuilds(obj) if cache else _rebuilds(obj)
    params = _chiral_params(obj)
    baked = from_source and params is not None
    rec = {"name": obj.name, "kind": obj.kind}
    if not from_source:
        rec["points_um"] = obj.points_um
    for key in _DIRECT[3:]:
        value = getattr(obj, key)
        if key == "v_profile" and baked:
            continue
        if key == "interpolation_um":
            # Not a "not set" default, so it is written; a chiral path's is left
            # out when the rebuild computes the same.
            if not (baked and value == chiral_step(params)):
                rec[key] = value
        elif key in _ALWAYS or not _is_default(key, value):
            rec[key] = value
    extra = {}
    if obj.kind in FILLED_KINDS:
        rec["fill"] = {key: getattr(obj, attr) for key, attr in _FILL.items()
                       if key in _ALWAYS_FILL or not _is_default(attr, getattr(obj, attr))}
    else:
        extra.update({attr: getattr(obj, attr) for attr in _FILL.values()
                      if not _is_default(attr, getattr(obj, attr))})
    if params is not None:
        rec["chiral"] = dict(params)
        if not baked:
            rec["chiral"].update(axis_um=obj.axis_um, profile=obj.profile)
        elif not from_bezier:
            rec["chiral"]["axis_um"] = obj.axis_um
    else:
        extra.update({name: getattr(obj, name) for name in _CHIRAL
                      if getattr(obj, name) is not None})
    extra.update({name: getattr(obj, name) for name in _EXTRA
                  if not _is_default(name, getattr(obj, name))})
    if extra:
        rec["extra"] = extra
    return rec


def pattern_to_document(objects, *, name: str, saved_at: str, frame: dict | None = None,
                        pattern_name: str | None = None,
                        cache: RebuildCache | None = None) -> dict:
    objects = list(objects)
    doc = {"format": FORMAT, "version": VERSION, "builder": BUILDER, "name": name,
           "saved_at": saved_at, "object_count": len(objects),
           "pattern_name": pattern_name, "frame": frame,
           "objects": [object_to_record(o, cache) for o in objects]}
    if cache is not None:
        cache.roll()
    return doc


def find_non_finite(objects) -> tuple[str, str] | None:
    """`(object name, field)` of the first number that is not finite, or None."""
    for obj in objects:
        for name in _FIELDS:
            value = getattr(obj, name, None)
            try:
                if name in _POINT_LISTS or name in _NUMBER_LISTS or name == "holes_um":
                    rows = value if name == "holes_um" else [value]
                    bad = any(row is not None and len(row)
                              and not np.isfinite(np.asarray(row, dtype=float)).all()
                              for row in rows or [])
                elif isinstance(value, dict):
                    bad = not _tree_is_finite(value)
                elif isinstance(value, (list, tuple)):
                    bad = not _tree_is_finite(list(value))
                else:
                    bad = isinstance(value, float) and not math.isfinite(value)
            except (TypeError, ValueError):
                bad = True
            if bad:
                return str(obj.name), _FILL_KEY.get(name, name)
    return None


# ------------------------------------------------------------------ writing --

def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _dumps(value) -> str:
    return json.dumps(value, allow_nan=False, default=_json_default)


def iter_json(value, indent: int = 0):
    """Yield `value` as indented JSON text in pieces, none of them a long encoder
    call. An object of plain values is one line; a list of points is one point
    per line; a list of numbers is one line. Raises ValueError on a non-finite
    number."""
    pad = " " * (indent + 2)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        if all(_is_flat(item) for item in value.values()):
            yield _dumps({str(key): item for key, item in value.items()})
            return
        yield "{"
        for i, (key, item) in enumerate(value.items()):
            yield ("," if i else "") + "\n" + pad + json.dumps(str(key)) + ": "
            yield from iter_json(item, indent + 2)
        yield "\n" + " " * indent + "}"
    elif isinstance(value, (list, tuple)):
        if not len(value):
            yield "[]"
        elif not isinstance(value[0], (list, tuple, dict, np.ndarray)):
            yield "["
            for start in range(0, len(value), _JSON_CHUNK):
                yield (", " if start else "") + _dumps(value[start:start + _JSON_CHUNK])[1:-1]
            yield "]"
        else:
            points = _is_flat(value[0]) and not isinstance(value[0], dict)
            yield "["
            for start in range(0, len(value), _JSON_CHUNK):
                piece = value[start:start + _JSON_CHUNK]
                yield ("," if start else "") + "\n" + pad
                if points:
                    yield _dumps(piece)[1:-1].replace("], [", "],\n" + pad + "[")
                else:
                    for i, item in enumerate(piece):
                        if i:
                            yield ",\n" + pad
                        yield from iter_json(item, indent + 2)
            yield "\n" + " " * indent + "]"
    else:
        yield _dumps(value)


def _is_flat(value) -> bool:
    """A scalar, or a short sequence of scalars (a point)."""
    if isinstance(value, (list, tuple)):
        return len(value) <= 8 and not any(
            isinstance(item, (list, tuple, dict, np.ndarray)) for item in value)
    return not isinstance(value, (dict, np.ndarray))


# ------------------------------------------------------------------ reading --

def _bad(name, field, message):
    return DesignFileError("bad_record", f"object {name!r}: {field}: {message}",
                           object_name=name, field=field)


def _number(value, name, field) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _bad(name, field, f"must be a number, got {type(value).__name__}")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise _bad(name, field, "is not a finite number")
    return float(value)


def _only_numbers(items) -> bool:
    return all(t is not bool and issubclass(t, (int, float)) for t in set(map(type, items)))


def _array(value, name, field, *, points: bool) -> np.ndarray:
    shape = "[[x, y], ...] microns" if points else "a list of numbers"
    if not isinstance(value, (list, tuple)):
        raise _bad(name, field, f"must be {shape}")
    parts = []
    for start in range(0, len(value), _ARRAY_CHUNK):
        piece = value[start:start + _ARRAY_CHUNK]
        if points:
            if not all(issubclass(t, (list, tuple)) for t in set(map(type, piece))):
                raise _bad(name, field, f"must be {shape}")
            items = chain.from_iterable(piece)
        else:
            items = piece
        if not _only_numbers(items):
            raise _bad(name, field, f"must be {shape}; it holds something that is "
                                    f"not a number")
        try:
            arr = np.asarray(piece, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _bad(name, field, f"must be {shape}") from exc
        if arr.ndim != (2 if points else 1) or (points and arr.shape[1] != 2):
            raise _bad(name, field, f"must be {shape}")
        if not np.isfinite(arr).all():
            raise _bad(name, field, "holds a number that is not finite")
        parts.append(arr)
    if not parts:
        return np.empty((0, 2) if points else (0,))
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def _numbers(value, name, field) -> list[float]:
    return _array(value, name, field, points=False).tolist()


def _points(value, name, field, minimum: int = 0) -> list[tuple[float, float]]:
    arr = _array(value, name, field, points=True)
    if len(arr) < minimum:
        raise _bad(name, field, f"needs at least {minimum} points")
    out: list = []
    for start in range(0, len(arr), _ARRAY_CHUNK):
        out.extend(map(tuple, arr[start:start + _ARRAY_CHUNK].tolist()))
    return out


def _tree_is_finite(value) -> bool:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, float) and not math.isfinite(item):
            return False
    return True


def _coerce(attr: str, value, name: str, field: str):
    """One stored value, checked and normalised to the type the object holds."""
    if attr in _POINT_LISTS:
        return None if value is None else _points(value, name, field)
    if attr == "holes_um":
        if not isinstance(value, (list, tuple)):
            raise _bad(name, field, "must be a list of outlines")
        return [_points(hole, name, field) for hole in value]
    if attr in _NUMBER_LISTS:
        return None if value is None else _numbers(value, name, field)
    if attr == "fill_centre_um":
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise _bad(name, field, "must be [x, y] microns")
        return (_number(value[0], name, field), _number(value[1], name, field))
    if attr in _NUMBERS:
        return _number(value, name, field)
    if attr in _OPTIONAL_NUMBERS:
        return None if value is None else _number(value, name, field)
    if attr in _BOOLS:
        if not isinstance(value, bool):
            raise _bad(name, field, "must be true or false")
        return value
    if attr in _DICTS:
        if value is not None and not isinstance(value, dict):
            raise _bad(name, field, "must be an object")
        if not _tree_is_finite(value):
            raise _bad(name, field, "holds a number that is not finite")
        if attr == "bezier" and value is not None:
            _check_bezier(value, name, field)
        return value
    if attr == "fill_kind":
        if not isinstance(value, str):
            raise _bad(name, field, "must be a string")
        return value
    raise _bad(name, field, "is not a field this version knows")


def _check_bezier(value: dict, name: str, field: str) -> None:
    from .bezier import BezierPath
    try:
        nodes = BezierPath.from_dict(value).nodes
    except Exception as exc:
        raise _bad(name, field, f"is not a Bezier path ({exc})") from exc
    if len(nodes) < 2:
        raise _bad(name, field, "needs at least 2 nodes")


def _check_extent(points, name: str, field: str) -> None:
    arr = np.asarray(points, dtype=np.float64)
    if arr.size and float(np.abs(arr).max()) > MAX_EXTENT_UM:
        raise _bad(name, field, f"reaches past {MAX_EXTENT_UM:g} um")


def _bezier_extent(bezier: dict, name: str) -> None:
    for node in bezier.get("nodes", []):
        for point in ((node.get("x"), node.get("y")), node.get("h_in"),
                      node.get("h_out")):
            if point is not None:
                _check_extent(point, name, "bezier")


def _flatten(bezier: dict, name: str) -> list[tuple[float, float]]:
    _bezier_extent(bezier, name)
    try:
        estimate = flatten_points(bezier)
        if estimate <= MAX_BEZIER_POINTS:
            return flatten_bezier(bezier)
    except Exception as exc:
        raise _bad(name, "bezier", str(exc)) from exc
    raise _too_large(name, "bezier", estimate, MAX_BEZIER_POINTS, "flattened points")


def _build_points(values: dict, chiral: dict | None, name: str) -> None:
    """Fill in the points of a record that does not carry them."""
    bezier = values.get("bezier")
    if chiral is not None:
        params = {}
        for key in _CHIRAL_PARAMS:
            if key not in chiral:
                raise _bad(name, f"chiral.{key}", "is missing")
            params[key] = _number(chiral[key], name, f"chiral.{key}")
        if chiral.get("axis_um") is not None:
            axis = _points(chiral["axis_um"], name, "chiral.axis_um", minimum=2)
            _check_extent(axis, name, "chiral.axis_um")
        elif bezier is not None:
            axis = _flatten(bezier, name)
        else:
            raise _bad(name, "chiral.axis_um", "is missing, and there is no bezier "
                                               "to take the axis from")
        from .pens import _CHIRAL_MAX_POINTS
        samples = chiral_samples(axis, params)
        if samples > _CHIRAL_MAX_POINTS:
            raise _too_large(name, "chiral", samples, _CHIRAL_MAX_POINTS,
                             "samples of the chiral builder")
        from .pens import bake_chiral
        wire = LithoObject(name, values["kind"], axis,
                           voltage=values.get("voltage", _default("voltage")))
        try:
            bake_chiral(wire, params["lambda_um"], params["y_amp_um"], params["v_k"],
                        params["phase_deg"], 1.0, spine=axis)
        except ValueError as exc:
            raise _bad(name, "chiral", str(exc)) from exc
        values.setdefault("interpolation_um", wire.interpolation_um)
        values.update(axis_um=wire.axis_um, points_um=wire.points_um,
                      v_profile=wire.v_profile, profile=wire.profile)
    elif bezier is not None:
        values["points_um"] = _flatten(bezier, name)
    else:
        raise _bad(name, "points_um", "is missing, and there is no bezier or "
                                      "chiral block to build it from")


def record_to_object(rec, index: int = 0, *,
                     warnings: list | None = None) -> LithoObject:
    """One object from its record. Raises `DesignFileError`.

    With a `warnings` list the record is loaded for a restore: a setting that an
    open refuses (a spacing, speed or interpolation that is not positive) is kept
    as saved and described in the list instead."""
    label = rec.get("name") if isinstance(rec, dict) else None
    label = label if isinstance(label, str) and label else f"#{index}"
    try:
        return _record_to_object(rec, label, warnings)
    except DesignFileError:
        raise
    except (ArithmeticError, AttributeError, LookupError, RecursionError, TypeError,
            ValueError) as exc:
        raise _bad(label, "record", f"cannot be read ({type(exc).__name__})") from exc


def _record_to_object(rec, label: str, warnings: list | None) -> LithoObject:
    if not isinstance(rec, dict):
        raise _bad(label, "record", "must be an object")
    name = rec.get("name")
    if not isinstance(name, str) or not name:
        raise _bad(label, "name", "must be a non-empty string")
    kind = rec.get("kind")
    if kind not in KINDS:
        raise _bad(name, "kind", f"must be one of {', '.join(KINDS)}")
    values: dict = {"name": name, "kind": kind}
    for key in _DIRECT[2:]:
        if key in rec:
            values[key] = _coerce(key, rec[key], name, key)
    fill = rec.get("fill")
    if fill is not None:
        if not isinstance(fill, dict):
            raise _bad(name, "fill", "must be an object")
        for key, value in fill.items():
            if key not in _FILL:
                raise _bad(name, f"fill.{key}", "is not a fill setting")
            values[_FILL[key]] = _coerce(_FILL[key], value, name, f"fill.{key}")
    extra = rec.get("extra")
    if extra is not None:
        if not isinstance(extra, dict):
            raise _bad(name, "extra", "must be an object")
        for key, value in extra.items():
            if key in _LEGACY_EXTRA:
                continue
            if key not in _FIELDS or key in ("name", "kind"):
                raise _bad(name, f"extra.{key}", "is not a field this version knows")
            values[key] = _coerce(key, value, name, f"extra.{key}")
    for attr in _POSITIVE:
        if attr in values and not values[attr] > 0:
            field = _FILL_KEY.get(attr, attr)
            if warnings is None:
                raise _bad(name, field, f"must be greater than 0, got {values[attr]:g}")
            warnings.append(f"object {name!r}: {field} is {values[attr]:g}")
    chiral = rec.get("chiral")
    if chiral is not None and not isinstance(chiral, dict):
        raise _bad(name, "chiral", "must be an object")
    if values.get("points_um") is None:
        _build_points(values, chiral, name)
    else:
        if warnings is None:
            _check_extent(values["points_um"], name, "points_um")
        for key in _CHIRAL:
            if chiral is not None and key in chiral:
                values[key] = _coerce(key, chiral[key], name, f"chiral.{key}")
    if len(values["points_um"]) < 2:
        raise _bad(name, "points_um", "needs at least 2 points")
    return LithoObject(**values)


def check_document(doc) -> None:
    """Raise `DesignFileError` unless `doc` is a design document this version reads."""
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise DesignFileError("not_a_design_file",
                              f"not a design file (its format is not {FORMAT!r})")
    version = doc.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise DesignFileError("not_a_design_file",
                              "not a design file (it has no version number)")
    builder = doc.get("builder", BUILDER)
    if isinstance(builder, bool) or not isinstance(builder, int):
        raise DesignFileError("not_a_design_file",
                              "not a design file (its builder tag is not a number)")
    if version > VERSION or builder > BUILDER:
        raise DesignFileError(
            "unsupported_version",
            f"this design file is version {version}, builder {builder}; this "
            f"service reads up to version {VERSION}, builder {BUILDER}")
    if not isinstance(doc.get("objects"), list):
        raise DesignFileError("not_a_design_file",
                              "not a design file (it has no objects list)")


def document_to_objects(doc, *, warnings: list | None = None,
                        skipped: list | None = None, cancelled=None) -> list[LithoObject]:
    """Every object of a document, in file order.

    One bad record refuses them all, unless a `skipped` list is given: then the
    record is left out and its `DesignFileError` is put in the list. `cancelled`
    is an event looked at before each record; once set, `BuildCancelled` is
    raised and what was built is dropped."""
    check_document(doc)
    objects = []
    for i, rec in enumerate(doc["objects"]):
        if cancelled is not None and cancelled.is_set():
            raise BuildCancelled
        try:
            objects.append(record_to_object(rec, i, warnings=warnings))
        except DesignFileError as exc:
            if skipped is None:
                raise
            skipped.append(exc)
    return objects


def document_to_pattern(doc, **kw) -> Pattern:
    objects = document_to_objects(doc, **kw)
    name = doc.get("pattern_name") or doc.get("name") or "design"
    return Pattern(name=str(name), objects=objects)
