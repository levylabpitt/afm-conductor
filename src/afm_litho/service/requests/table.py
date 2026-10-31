"""The request table: one row per thing a client can ask for, shared by both doors.

Rows live in topic files (`reads.py`, `design.py`) and are gathered at the bottom. The FLEX
verb list (`flex_dispatch.REGISTRY`) is built from the rows, below.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi.exceptions import RequestValidationError

# "stop" is a retract or stop: nothing may run or be written before its handler.
CONTROLS = ("tip", "stop", "none")


@dataclass(frozen=True)
class Flex:
    """How a row is reached over FLEX: verb, socket, honest time bound, parameter intake."""

    verb: str
    socket: str          # "command" or "read"
    bound_s: float
    intake: Callable[[Any], Any]
    gate: str            # "token", "stop" or "none"


@dataclass(frozen=True)
class HttpCall:
    """Everything an HTTP request carries, handed whole to the row's intake.

    `path` and `query` hold the raw strings from the URL (the intake converts them);
    `body` is the parsed JSON, or the text itself for a row bound with body="text".
    """

    body: Any = None
    json_null: bool = False   # the body was the literal `null` (None, but not "missing")
    path: dict[str, str] = field(default_factory=dict)
    query: dict[str, str] = field(default_factory=dict)


def _refuse(kind: str, where: tuple, msg: str, value: Any) -> RequestValidationError:
    return RequestValidationError([{"type": kind, "loc": where, "msg": msg, "input": value}])


def object_body(call: HttpCall, required: bool = False) -> dict | None:
    """The body as an object, refused with FastAPI's 422 detail as `body: dict` was.
    A missing body is None when optional; required, it is the "missing" error."""
    if call.body is None and not required:
        return None
    if call.body is None and not call.json_null:
        raise _refuse("missing", ("body",), "Field required", None)
    if not isinstance(call.body, dict):
        raise _refuse("dict_type", ("body",), "Input should be a valid dictionary", call.body)
    return call.body


def path_int(call: HttpCall, name: str) -> int:
    value = call.path[name]
    try:
        return int(value)
    except ValueError:
        raise _refuse("int_parsing", ("path", name),
                      "Input should be a valid integer, unable to parse string as an integer",
                      value) from None


def path_float(call: HttpCall, name: str) -> float:
    value = call.path[name]
    try:
        return float(value)
    except ValueError:
        raise _refuse("float_parsing", ("path", name),
                      "Input should be a valid number, unable to parse string as a number",
                      value) from None


def query_float(call: HttpCall, name: str, default: float) -> float:
    if name not in call.query:
        return default
    value = call.query[name]
    try:
        return float(value)
    except ValueError:
        raise _refuse("float_parsing", ("query", name),
                      "Input should be a valid number, unable to parse string as a number",
                      value) from None


def query_int(call: HttpCall, name: str, default: int) -> int:
    if name not in call.query:
        return default
    value = call.query[name]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        number = float(value)       # pydantic reads "8.0" as 8, but not "8.5"
        if number.is_integer():
            return int(number)
    except (ValueError, OverflowError):
        pass
    raise _refuse("int_parsing", ("query", name),
                  "Input should be a valid integer, unable to parse string as an integer",
                  value)


def query_text(call: HttpCall, name: str, default: str | None = None) -> str:
    """A text query value; with no default, a missing one is refused as FastAPI refused it."""
    if name in call.query:
        return call.query[name]
    if default is None:
        raise _refuse("missing", ("query", name), "Field required", {})
    return default


@dataclass(frozen=True)
class Http:
    """How a row is reached over HTTP: method, path, and the intake of an `HttpCall`."""

    method: str
    path: str
    intake: Callable[[HttpCall], Any]
    gate: str            # "operator_tip", "operator_stop", "stop" or "none"
    handler: Callable[[Any], Any] | None = None   # the browser's own form of the handler
    # "json" (empty reads as None), "text", "bytes", "lenient" (a bad body is None), "ignore"
    body: str = "json"


@dataclass(frozen=True)
class Row:
    name: str                       # one name for both carriers
    kind: str                       # "command", "edit" or "read"
    control: str                    # one of CONTROLS
    handler: Callable[[Any], Any]   # the coroutine or function, carrier-neutral
    flex: Flex | None = None
    http: Http | None = None
    claims: str | None = None       # run claim kind, or None
    help: str = ""                  # the HELP card text
    intent: bool = False            # starts a run: its request entry is fsynced before the handler
    journalled: bool = True         # False: dispatch notes nothing (pages, design edits)


ROWS: list[Row] = []


def flex_row(verb: str) -> Row | None:
    for row in ROWS:
        if row.flex is not None and row.flex.verb == verb:
            return row
    return None


def http_rows() -> list[Row]:
    return [row for row in ROWS if row.http is not None]


from .cards import CARDS  # noqa: E402
from . import control as _control  # noqa: E402  (these build Rows, so they load after them)
from . import design as _design  # noqa: E402
from . import edits as _edits  # noqa: E402
from . import litho as _litho  # noqa: E402
from . import meta as _meta  # noqa: E402
from . import reads as _reads  # noqa: E402
from . import sample as _sample  # noqa: E402
from . import scan as _scan  # noqa: E402
from . import session as _session  # noqa: E402
from . import stops as _stops  # noqa: E402
from . import tips as _tips  # noqa: E402

ROWS.extend(_reads.ROWS)
ROWS.extend(_control.ROWS)
ROWS.extend(_design.ROWS)
ROWS.extend(_stops.ROWS)
ROWS.extend(_edits.ROWS)
ROWS.extend(_meta.ROWS)
ROWS.extend(_litho.ROWS)
ROWS.extend(_scan.ROWS)
ROWS.extend(_sample.ROWS)
ROWS.extend(_session.ROWS)
ROWS.extend(_tips.ROWS)


def _check_tip_routes_are_gated() -> None:
    """A row on a tip-mutating path takes the operator's access in `dispatch`; nothing else
    does. The one exception is a stop row (the browser Withdraw), which takes control in
    memory before it runs and announces it afterwards: gate "operator_stop"."""
    from ..control import TIP_MUTATING
    for row in http_rows():
        if row.http.method == "POST" and row.http.path in TIP_MUTATING:
            expected = "operator_stop" if row.control == "stop" else "operator_tip"
            if row.http.gate != expected:
                raise RuntimeError(f"row {row.name} is on {row.http.path} but its gate is "
                                   f"{row.http.gate!r}, not {expected!r}")


_check_tip_routes_are_gated()


def _check_every_tip_route_has_a_row() -> None:
    """A path in TIP_MUTATING with no POST row would take no operator access at all."""
    from ..control import TIP_MUTATING
    served = {row.http.path for row in http_rows() if row.http.method == "POST"}
    missing = sorted(TIP_MUTATING - served)
    if missing:
        raise RuntimeError(f"tip-mutating routes with no POST row: {missing}")


_check_every_tip_route_has_a_row()


def _check_every_verb_has_a_card() -> None:
    verbs = {row.flex.verb for row in ROWS if row.flex is not None}
    if verbs != set(CARDS):
        raise RuntimeError(f"verbs and HELP cards differ: {sorted(verbs ^ set(CARDS))}")


_check_every_verb_has_a_card()

_registry: dict | None = None


def registry() -> dict:
    """The FLEX verb list, `flex_dispatch.REGISTRY`: one entry per row with a FLEX binding, in
    the card order HELP and getCapabilities have always printed. Built once; a test that widens
    a bound edits this one dict."""
    global _registry
    if _registry is None:
        from ..flex_dispatch import Method
        _registry = {
            verb: Method(name=verb, params=card.params, result=card.result, bound_s=card.bound_s,
                         doc=card.doc, command_only=flex_row(verb).flex.socket == "command")
            for verb, card in CARDS.items()}
    return _registry


def read_only() -> frozenset:
    """The verbs the READ socket serves: nothing in this set can move the tip."""
    return frozenset(name for name, m in registry().items() if not m.command_only)


def command_only() -> frozenset:
    return frozenset(name for name, m in registry().items() if m.command_only)
