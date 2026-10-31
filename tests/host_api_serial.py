"""SerializingHostApi: the H8 dress rehearsal, in the test suite only.

Every HostApi verb and `HostState.pass_ended` is wrapped on the class. A call's arguments
and its reply (the awaited result, for a verb that returns an awaitable) go through
`json.loads(json.dumps(x))`; the copy is what the callee and the caller see, and a copy that
differs from the original fails the call. So whatever crosses base<->host in the suite has
proved it survives a socket. Nothing here waits, sleeps or reads a clock, and a stop verb
still returns its awaitable at once (the awaited reply is checked on the way back).

A shape log records (verb, argument keys, reply keys), never values. `tests/golden/
host_api_calls.json` is that log frozen once; a run with a shape not in it fails.
Switch: on by default; `--no-serial-host-api` turns it off; `--freeze-host-api-log` merges
this run's shapes into the golden. Production code never imports this file.
"""
from __future__ import annotations

import functools
import inspect
import json
import math
import os
from pathlib import Path

from afm_litho.service.host import api as host_api
from afm_litho.service.host_state import HostState

GOLDEN = Path(__file__).parent / "golden" / "host_api_calls.json"
LOG_DIR_ENV = "AFM_HOST_API_LOG_DIR"
FREEZING_ENV = "AFM_HOST_API_FREEZING"

# Replies/arguments already known not to survive the round trip, as (verb, side). Shrink-only:
# each entry is a product shape that a maintainer decides about (J5); the wrapper passes the original
# through for these instead of failing, and says so in the shape log.
KNOWN_NOT_JSON: dict[tuple[str, str], str] = {}


# Set False for one test by the `host_api_unserialized` marker (conftest). Shrink-only: see
# EXEMPT_TESTS in tests/test_host_api_serial.py.
ENABLED = True


class NotJsonPlain(AssertionError):
    """A value crossing the HostApi did not survive json.dumps/json.loads unchanged."""


def _kind(x) -> type:
    """The JSON kind of `x`. A float subclass (numpy float64) encodes as a float, so it
    survives a socket; a tuple, a numpy array or an arbitrary object does not."""
    for base in (bool, int, float, str, dict, list):
        if isinstance(x, base):
            return base
    return type(x)


def _same(a, b) -> bool:
    """Equality that keeps types (tuple != list, int != float key) and treats NaN as equal."""
    if _kind(a) is not _kind(b):
        return False
    if isinstance(a, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, dict):
        return (list(a) == list(b)) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _diff(a, b, path="$"):
    """Path to the first place `a` and its decoded copy `b` differ."""
    if _kind(a) is not _kind(b):
        return f"{path} ({type(a).__name__} -> {type(b).__name__})"
    if isinstance(a, dict):
        for k in a:
            if k not in b:
                return f"{path}[{k!r}] (key {type(k).__name__} -> str)"
            d = _diff(a[k], b[k], f"{path}[{k!r}]")
            if d:
                return d
    if isinstance(a, list):
        for i, (x, y) in enumerate(zip(a, b)):
            d = _diff(x, y, f"{path}[{i}]")
            if d:
                return d
    return "" if _same(a, b) else f"{path} (value)"


def round_trip(value, where: str):
    """The decoded copy of `value`; raises NotJsonPlain if it cannot be encoded or differs."""
    try:
        copy = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise NotJsonPlain(f"{where}: not JSON-encodable ({exc})") from exc
    if not _same(value, copy):
        raise NotJsonPlain(f"{where}: changed by the JSON round trip at {_diff(value, copy)}")
    return copy


def _shape(value):
    if isinstance(value, dict):
        return sorted(str(k) for k in value)
    return "None" if value is None else type(value).__name__


class _ShapeLog:
    def __init__(self) -> None:
        self.seen: set[str] = set()
        self.frozen: set[str] | None = None
        self.loaded = False

    def _load_frozen(self) -> set[str] | None:
        if os.environ.get(FREEZING_ENV) or not GOLDEN.exists():
            return None
        return {json.dumps(r, sort_keys=True) for r in json.loads(GOLDEN.read_text("utf-8"))}

    def record(self, verb: str, args: list, kwargs: dict, reply) -> None:
        key = json.dumps({"verb": verb, "args": [_shape(a) for a in args],
                          "kwargs": {k: _shape(v) for k, v in sorted(kwargs.items())},
                          "reply": _shape(reply)}, sort_keys=True)
        if key in self.seen:
            return
        self.seen.add(key)
        if not self.loaded:
            self.loaded, self.frozen = True, self._load_frozen()
        if self.frozen is not None and key not in self.frozen:
            raise NotJsonPlain(f"{verb}: call shape not in tests/golden/host_api_calls.json: "
                               f"{key}")
        log_dir = os.environ.get(LOG_DIR_ENV)
        if log_dir:
            with open(Path(log_dir) / f"{os.getpid()}.jsonl", "a", encoding="utf-8") as f:
                f.write(key + "\n")


_LOG = _ShapeLog()


def _checked(verb: str, args: tuple, kwargs: dict):
    where = f"{verb} argument"
    if (verb, "args") in KNOWN_NOT_JSON:
        return list(args), dict(kwargs)
    return ([round_trip(a, where) for a in args],
            {k: round_trip(v, f"{where} {k!r}") for k, v in kwargs.items()})


def _reply(verb: str, args: list, kwargs: dict, reply):
    if (verb, "reply") not in KNOWN_NOT_JSON:
        reply = round_trip(reply, f"{verb} reply")
    _LOG.record(verb, args, kwargs, reply)
    return reply


def wrap(verb: str, fn):
    """`fn` with its arguments and reply round-tripped. An async fn stays async; a plain fn
    that returns an awaitable still returns an awaitable at once, with the check run on its
    awaited result."""
    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def serial_async(self, *args, **kwargs):
            if not ENABLED:
                return await fn(self, *args, **kwargs)
            a, k = _checked(verb, args, kwargs)
            return _reply(verb, a, k, await fn(self, *a, **k))
        return serial_async

    @functools.wraps(fn)
    def serial(self, *args, **kwargs):
        if not ENABLED:
            return fn(self, *args, **kwargs)
        a, k = _checked(verb, args, kwargs)
        out = fn(self, *a, **k)
        if inspect.isawaitable(out):
            async def awaited():
                return _reply(verb, a, k, await out)
            return awaited()
        return _reply(verb, a, k, out)
    return serial


def _targets() -> list[tuple[type, str]]:
    verbs = [(host_api.HostApi, n) for n, _ in host_api.VERBS if hasattr(host_api.HostApi, n)]
    return verbs + [(HostState, "pass_ended")]


def install() -> list:
    """Wrap every verb on its class. Returns the undo list for `uninstall`."""
    undo = []
    for cls, name in _targets():
        original = cls.__dict__[name]
        setattr(cls, name, wrap(name, original))
        undo.append((cls, name, original))
    return undo


def uninstall(undo: list) -> None:
    for cls, name, original in undo:
        setattr(cls, name, original)


def merge_log_dir(log_dir: Path, golden: Path = GOLDEN) -> int:
    """Union the per-worker shape files (and any existing golden) into the golden."""
    rows = {json.dumps(r, sort_keys=True) for r in
            (json.loads(golden.read_text("utf-8")) if golden.exists() else [])}
    for f in Path(log_dir).glob("*.jsonl"):
        rows |= {line.strip() for line in f.read_text("utf-8").splitlines() if line.strip()}
    ordered = [json.loads(r) for r in sorted(rows)]
    golden.parent.mkdir(parents=True, exist_ok=True)
    golden.write_text(json.dumps(ordered, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return len(ordered)
