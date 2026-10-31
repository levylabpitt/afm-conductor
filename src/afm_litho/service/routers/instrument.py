"""The instrument-model REST surface (docs/INSTRUMENT_MODEL_SPEC.md sections 1
and 10): `GET /api/instrument`, `POST /api/instrument/override`,
`POST /api/instrument/reload`, `POST /api/instrument/open_config`,
`POST /api/fault/clear`, and `POST /api/keeper/contact`.

Every endpoint here re-derives the model from `AFM_LITHO_INSTRUMENT_CONFIG`
(current env value, read fresh on each call -- see
`instrument_model.instrument_config_path`) rather than trusting a value cached at
`Hub()` construction, so `POST /api/instrument/reload` (and a test's monkeypatched
env var) take effect without a process restart. A successful load also refreshes
`hub.instrument_model`, so the write/scan guards see the same model this door just
reported.
"""

from __future__ import annotations

import json
import os
import tomllib

from fastapi import APIRouter

from ..host import instrument_cmds
from ..hub import hub
from ..instrument_model import (
    ModelIncomplete,
    instrument_config_path,
    overrides_log_path,
)

router = APIRouter()


def _load_current():
    """(model, config_path, overrides_log) for THIS request, from the live env."""
    return instrument_cmds.load_current(hub)


def _overrides_tail(overrides_log, n: int = 20) -> list[dict]:
    if not overrides_log.exists():
        return []
    lines = overrides_log.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def get_instrument():
    config_path = instrument_config_path()
    try:
        model, config_path, overrides_log = _load_current()
    except (ModelIncomplete, tomllib.TOMLDecodeError, ValueError, TypeError) as exc:
        model = getattr(hub.host, "instrument_model", None)
        if model is None:
            return {"error": str(exc), "config_path": str(config_path)}
        overrides_log = overrides_log_path(config_path)
    else:
        hub.host.instrument_model = model
    d = model.to_dict()
    d["overrides_tail"] = _overrides_tail(overrides_log)
    # The config-file path the cockpit's Config dialog shows (section 14 item 2).
    # A stable string (not wall-clock), so it does not break getInstrument
    # idempotency (REST == FLEX).
    d["config_path"] = str(config_path)
    d["last_write_params"] = getattr(hub.host, "last_write_params", {})
    d["last_scan_params"] = getattr(hub.host, "last_scan_params", {})
    # The live free-air baseline (13.1 item 2), reported for display. Its age gates the
    # approach door only; a write reads free air before every object itself. None until
    # an approach/withdraw/scan re-reference establishes it.
    _fa = getattr(hub.host, "free_air", None)
    if _fa is not None:
        # value_v/ts/mode/withdrawn/defl_v only -- NOT a server-computed age_s: getInstrument
        # is idempotent (test_flex_get_instrument_matches_rest_get pins REST == FLEX),
        # and a wall-clock age would differ between two calls. The cockpit derives
        # freshness from `ts` (age = now - ts).
        d["free_air"] = {
            "value_v": _fa.value_v, "ts": _fa.ts, "mode": _fa.mode,
            "withdrawn": bool(_fa.withdrawn),
            "defl_v": getattr(_fa, "defl_v", None)}
    else:
        d["free_air"] = None
    # The most recent park outcome (E1), so a script gets it in the same call as the
    # model. Same dict `GET /api/park/last` / FLEX `getLastPark` return. `ParkResult.ts`
    # is the park time (stable), so this does not break getInstrument idempotency
    # (REST == FLEX). None until the first `safe_park`.
    _lp = getattr(hub.host, "last_park", None)
    d["last_park"] = _lp.to_dict() if _lp is not None else None
    # The XY LVDT calibration scans and telemetry convert with, and when it was last
    # read from Igor (refreshed at connect, every scan start and every approach).
    from ..xy_cal import xy_cal_payload
    d["xy_cal"] = xy_cal_payload(hub)
    return d


def post_instrument_override(body: dict | None = None):
    """Override one model field, on the process hub (`host.instrument_cmds.set_override`)."""
    return instrument_cmds.set_override(hub, body)


def post_instrument_reload():
    try:
        before = getattr(hub.host, "instrument_model", None)
        model, _config_path, _overrides_log = _load_current()
    except (ModelIncomplete, tomllib.TOMLDecodeError) as exc:
        return {"error": str(exc)}
    diff = {}
    if before is not None:
        before_d = before.to_dict()
        after_d = model.to_dict()
        for key, after_field in after_d.items():
            if key == "sample_id":
                continue
            before_field = before_d.get(key)
            if before_field is None or before_field.get("value") != after_field.get("value"):
                diff[key] = {"old": before_field.get("value") if before_field else None,
                            "new": after_field.get("value")}
    hub.host.instrument_model = model
    return {"ok": True, "diff": diff}


def post_instrument_open_config():
    config_path = instrument_config_path()
    if not config_path.exists():
        return {"error": f"instrument config not found: {config_path}"}
    try:
        os.startfile(str(config_path))
    except Exception as exc:                             # noqa: BLE001
        return {"error": str(exc)}
    return {"path": str(config_path)}


def post_fault_clear(body: dict | None = None):
    """Clear the latched fault, on the process hub (`host.instrument_cmds.clear_fault`)."""
    return instrument_cmds.clear_fault(hub, body)


def post_keeper_contact(body: dict):
    """A keeper is in contact again (a restarted one): `{"pid": n}` of a live keeper.
    Writes refused since the last keeper died are allowed again (decision 8).
    The pid must be a keeper's: the one that started this service, or the one the keeper
    state file names (a restarted keeper writes it), so no other process can lift it."""
    from ...keeper.pids import pid_alive
    from .. import keeper_link
    pid = body.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return {"error": "pid must be a positive integer"}
    if pid not in _keeper_pids():
        return {"error": f"pid {pid} is not a keeper"}
    if not pid_alive(pid):
        return {"error": f"no running process has pid {pid}"}
    keeper_link.restore_contact(pid, hub)
    return {"ok": True, "pid": pid}


def _keeper_pids() -> set[int]:
    """The keeper pids this service accepts: AFM_LITHO_KEEPER_PID and the state file's."""
    from ...keeper import status
    from ...keeper.protocol import KEEPER_PID_ENV
    pids = set()
    env = os.environ.get(KEEPER_PID_ENV, "").strip()
    if env.isdigit():
        pids.add(int(env))
    state = status.read_state(status.state_path()) or {}
    if isinstance(state.get("keeper_pid"), int):
        pids.add(state["keeper_pid"])
    return pids
