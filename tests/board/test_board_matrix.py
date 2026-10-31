"""The board answers exactly what the pre-board code answered, in every state.

`state_matrix.json` was recorded from the code before 6c1 (hub-reaching reads) with
`AFM_RECORD_BOARD_MATRIX=1`. Each state is set on the process hub's HostState, then
getState, getStatus, getTelemetry, GET /api/state and the /ws frame are read through
their public entry points. Clock fields are masked: they differ per read by design.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from afm_litho.service.hub import Hub, hub
from afm_litho.service.requests import table  # noqa: F401  (import order)
from afm_litho.service.requests import reads, stream
from afm_litho.service.requests.context import using_hub

FIXTURE = Path(__file__).with_name("state_matrix.json")
CLOCK_KEYS = {"t", "age_s", "refreshed_at", "at", "closed_at", "updated_at", "saved_at"}


class _Task:
    def __init__(self, done: bool):
        self._done = done

    def done(self) -> bool:
        return self._done


class _Thread:
    def is_alive(self) -> bool:
        return True


def _set(monkeypatch, **fields):
    for name, value in fields.items():
        monkeypatch.setattr(hub.host, name, value)


STATES = {
    "idle": lambda mp: None,
    "engaged": lambda mp: _set(mp, engaged=True),
    "scanning": lambda mp: _set(mp, engaged=True, _scan_task=_Task(False), _scan_row=7),
    "writing": lambda mp: _set(mp, engaged=True, litho_thread=_Thread()),
    "faulted": lambda mp: _set(mp, fault={"reason": "test fault", "critical": True}),
    "parked": lambda mp: _set(
        mp, engaged=True,
        _safe_park={"count": 1, "active": True, "last_reason": "test", "verified": True,
                    "critical": False, "transport": "socket", "z_v": 0.0, "at": 1.0}),
}


def _mask(value):
    if isinstance(value, dict):
        return {k: (None if k in CLOCK_KEYS else _mask(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask(v) for v in value]
    return value


def _answers() -> dict:
    with using_hub(hub):
        web = asyncio.run(reads._web_state(None))
        out = {
            "getState": reads._state(None),
            "getStatus": reads._status(None),
            "getTelemetry": reads._telemetry(None),
            "api_state": web,
            "ws_frame": stream.telemetry_frame(hub),
        }
    return _mask(json.loads(json.dumps(out)))


@pytest.fixture(autouse=True)
def _pristine_live_hub(monkeypatch):
    """Read the process hub as it is NOW (an earlier test may have replaced the module's
    hub) with every instrument field and the run record put back to a new hub's values,
    so nothing an earlier test left behind leaks in; then make the host live."""
    from afm_litho.service import hub as hub_module
    from afm_litho.service.host_state import HOST_FIELD_NAMES
    from afm_litho.service.runs import RunRegistry
    current, fresh = hub_module.hub, Hub()
    for name in HOST_FIELD_NAMES:
        if name != "_sink":
            monkeypatch.setattr(current.host, name, getattr(fresh.host, name))
    monkeypatch.setattr(current, "runs", RunRegistry())
    current.host.heard()
    monkeypatch.setitem(globals(), "hub", current)


@pytest.mark.parametrize("name", list(STATES))
def test_board_matches_pre_board_answers(name, monkeypatch):
    STATES[name](monkeypatch)
    got = _answers()
    if os.environ.get("AFM_RECORD_BOARD_MATRIX"):
        data = json.loads(FIXTURE.read_text("utf-8")) if FIXTURE.exists() else {}
        data[name] = got
        FIXTURE.write_text(json.dumps(data, indent=1, sort_keys=True), "utf-8")
        return
    want = json.loads(FIXTURE.read_text("utf-8"))[name]
    def leaves(a, b, path=""):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                yield from leaves(a.get(k, "<missing>"), b.get(k, "<missing>"), path + "/" + k)
        elif a != b:
            yield path, a, b
    assert list(leaves(got, want)) == []
