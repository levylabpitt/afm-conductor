"""The run's JSON form and its recipe fingerprint.

Canonical JSON is sorted keys, no spaces, ASCII only, no NaN: the same recipe always gives the
same bytes. The recipe fingerprint is the SHA-256 of the recipe's canonical JSON alone, so the
execution id, start time and saved files never change it, and outcomes are not in the run at
all. Body entries carry `kind` and their number `n`; `n` is checked on reading.
"""

from __future__ import annotations

import hashlib
import json
import typing
from dataclasses import MISSING, fields

from .model import ENTRY_KINDS, Entry, Recipe, Run, Part


def to_plain(part: Part) -> dict:
    """A part as plain JSON values (dicts, lists, numbers, text, null)."""
    out: dict = {}
    if isinstance(part, Entry):
        out["kind"] = part.KIND
    for f in fields(part):
        out[f.name] = _plain_value(getattr(part, f.name))
    if isinstance(part, Recipe):
        for n, entry in enumerate(out["body"]):
            entry["n"] = n
    return out


def _plain_value(value):
    if isinstance(value, Part):
        return to_plain(value)
    if isinstance(value, tuple):
        return [_plain_value(v) for v in value]
    return value


def from_plain(cls: type[Part], data: dict) -> Part:
    """Build `cls` from plain JSON values; unknown or missing fields are errors."""
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__}: expected an object, got {data!r}")
    data = dict(data)
    if issubclass(cls, Entry):
        data.pop("kind", None)
    hints = typing.get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown fields {sorted(unknown)}")
    missing = {f.name for f in fields(cls)
               if f.default is MISSING and f.default_factory is MISSING} - set(data)
    if missing:
        raise ValueError(f"{cls.__name__}: missing fields {sorted(missing)}")
    if cls is Recipe:
        data["body"] = _body_from_plain(data["body"])
    built = {name: _value_from_plain(hints[name], v) for name, v in data.items()}
    return cls(**built)


def _body_from_plain(items: list) -> tuple[Entry, ...]:
    if not isinstance(items, list):
        raise ValueError(f"Recipe.body: expected a list, got {items!r}")
    body = []
    for n, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"body entry {n}: expected an object, got {item!r}")
        item = dict(item)
        if item.pop("n", None) != n:
            raise ValueError(f"body entry {n} is numbered {item.get('n')!r}")
        cls = ENTRY_KINDS.get(item.get("kind"))
        if cls is None:
            raise ValueError(f"body entry {n}: unknown kind {item.get('kind')!r}")
        body.append(from_plain(cls, item))
    return tuple(body)


def _value_from_plain(hint, value):
    """Turn nested objects back into parts; leave the rest for the part's own check."""
    if isinstance(value, dict):
        # `hint` is a part, or `part | None`; anything else cannot hold an object.
        options = [h for h in (typing.get_args(hint) or (hint,)) if isinstance(h, type)]
        parts = [h for h in options if issubclass(h, Part)]
        if not parts:
            raise ValueError(f"expected {hint}, got an object {value!r}")
        return from_plain(parts[0], value)
    if isinstance(value, list):
        args = typing.get_args(hint)
        if not args:
            return tuple(value)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_value_from_plain(args[0], v) for v in value)
        return tuple(_value_from_plain(a, v) for a, v in zip(args, value))
    return value


def canonical(plain) -> str:
    """The one byte form of plain JSON values: sorted keys, no spaces, ASCII escapes, and
    NaN or infinity refused (they are not JSON, and would not hash the same everywhere)."""
    return json.dumps(plain, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def recipe_fingerprint(recipe: Recipe) -> str:
    """SHA-256 hex of the recipe's canonical JSON."""
    return hashlib.sha256(canonical(to_plain(recipe)).encode("ascii")).hexdigest()


def run_to_json(run: Run) -> str:
    """The stored form: the fingerprint, the recipe and the execution, canonical."""
    return canonical({"fingerprint": recipe_fingerprint(run.recipe),
                      "recipe": to_plain(run.recipe),
                      "execution": to_plain(run.execution)})


def run_from_json(text: str) -> Run:
    """Read a stored run. Anything malformed raises ValueError, as does a recipe that no
    longer matches its stored fingerprint."""
    data = json.loads(text)
    if not isinstance(data, dict) or set(data) != {"fingerprint", "recipe", "execution"}:
        raise ValueError("run JSON must be an object of fingerprint, recipe and execution")
    try:
        run = from_plain(Run, {"recipe": data["recipe"], "execution": data["execution"]})
    except (TypeError, KeyError) as e:  # a shape the field checks did not foresee
        raise ValueError(f"run JSON is malformed: {e}") from e
    if recipe_fingerprint(run.recipe) != data["fingerprint"]:
        raise ValueError("run JSON: recipe does not match its fingerprint")
    return run
