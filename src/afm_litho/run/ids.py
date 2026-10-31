"""Execution ids: `<w|s>-YYYY-MM-DD-HHMM-NN`, kind letter, start minute in UTC, a number.

UTC so ids sort in start order across daylight-saving changes; the execution's `started`
keeps the local time with its offset for reading. The same recipe run twice has one
fingerprint and two execution ids.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone

KIND_LETTER = {"write": "w", "scan": "s"}
MAX_PER_MINUTE = 99


def id_prefix(kind: str, started: datetime) -> str:
    """The part of an id shared by every run of `kind` started in the same UTC minute."""
    if started.utcoffset() is None:
        raise ValueError("started must be timezone-aware")
    return f"{KIND_LETTER[kind]}-{started.astimezone(timezone.utc):%Y-%m-%d-%H%M}-"


def next_execution_id(kind: str, started: datetime, taken: Iterable[str] = ()) -> str:
    """The id for a `kind` run started at `started` (timezone-aware): the lowest number from
    01 not in `taken`. The caller passes the ids already used (in memory or in the run store);
    uniqueness across restarts is the store's job."""
    prefix = id_prefix(kind, started)
    taken = set(taken)
    for number in range(1, MAX_PER_MINUTE + 1):
        candidate = f"{prefix}{number:02d}"
        if candidate not in taken:
            return candidate
    raise RuntimeError(f"more than {MAX_PER_MINUTE} {kind} runs started in one minute")
