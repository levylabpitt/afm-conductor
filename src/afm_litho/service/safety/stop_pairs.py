"""One park for a door's Stop and the keeper's copy of it.

A door (cockpit Abort, FLEX abort/safePark) parks by its own route and also sends its Stop
to the keeper with an id, `STOP <who> <why> #<id>`; the keeper relays that line to the
stop port. Whichever of the two reaches the host first starts the park; the other attaches
to the same request and reports its outcome, so Igor sees one AFM_SafePark, as it does with
no keeper. A Stop with no id (the keeper's own console key, Stop AFM.cmd) always parks.

An id is forgotten EXPIRE_AFTER_S after its park's outcome is known: a copy that arrives
later parks again, which is the safe side.
"""

from __future__ import annotations

import re
import secrets
import threading

from ... import clock

EXPIRE_AFTER_S = 10.0
# Any id whose park never reports (it always does, within the watchdog bound) is dropped
# after this, so the table cannot grow without end.
_FORGET_ANYWAY_S = 120.0

_ID = re.compile(r"(?:^|\s)#([0-9a-f]{16})(?=\s|$)")

_lock = threading.Lock()
# id -> [request, forget-at on the clock]
_pairs: dict[str, list] = {}


def new_id() -> str:
    return secrets.token_hex(8)


def split_id(why: str) -> tuple[str, str | None]:
    """`(why without the id, id)`; the id is None when the line carries none."""
    m = _ID.search(why)
    if m is None:
        return why, None
    return " ".join((why[:m.start()] + why[m.end():]).split()), m.group(1)


def start_once(stop_id: str | None, start):
    """`(request, first)`. The first caller with this id runs `start()` (which returns the
    park's ParkRequest); a later caller gets that same request and `first=False`. With no
    id, `start()` always runs."""
    if stop_id is None:
        return start(), True
    now = clock.now()
    with _lock:
        for k in [k for k, (_, until) in _pairs.items() if until <= now]:
            del _pairs[k]
        held = _pairs.get(stop_id)
        if held is not None:
            return held[0], False
        req = start()
        entry = [req, now + _FORGET_ANYWAY_S]
        _pairs[stop_id] = entry
    req.future.add_done_callback(lambda _f: _expire(stop_id, entry))
    return req, True


def _expire(stop_id: str, entry: list) -> None:
    with _lock:
        if _pairs.get(stop_id) is entry:
            entry[1] = min(entry[1], clock.now() + EXPIRE_AFTER_S)


def clear() -> None:
    """Forget every id (tests)."""
    with _lock:
        _pairs.clear()
