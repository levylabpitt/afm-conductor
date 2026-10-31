"""The instrument-model commands: clear the latched fault, override one model field.

Both are plain functions (sync); the doors run them in a worker thread. Each takes its
hub explicitly. The model is re-derived from `AFM_LITHO_INSTRUMENT_CONFIG` on each call
(`instrument_model.instrument_config_path`), so a reload or a test's env var takes effect
without a restart.
"""

from __future__ import annotations

import threading
import time
import tomllib

from ..instrument_model import (
    ModelIncomplete,
    file_owned_platform,
    instrument_config_path,
    load_model,
    overrides_log_path,
)


def current_platform(hub, config_path) -> str:
    """The platform to load with. The config file owns it (section 14 item 4):
    `[override] platform` beats `[instrument] platform`; else whatever
    `hub.instrument_model` carries (typically "twin" off-instrument)."""
    p = file_owned_platform(config_path)
    if p:
        return p
    model = getattr(hub.host, "instrument_model", None)
    return model.platform.value if model is not None else "twin"


def load_current(hub):
    """(model, config_path, overrides_log) for this request, from the live env."""
    config_path = instrument_config_path()
    overrides_log = overrides_log_path(config_path)
    platform = current_platform(hub, config_path)
    model = load_model(platform=platform, startup_cal=None, config_path=config_path,
                       overrides_log=overrides_log)
    return model, config_path, overrides_log


def _run_active(hub) -> bool:
    """True while the tip is engaged or a scan or write is running: a platform change
    then re-derives the park plan under a live loop, so it is refused (section 14
    item 4). getattr defaults so a partly built hub never raises here."""
    if bool(getattr(hub.host, "engaged", False)):
        return True
    thread = getattr(hub.host, "litho_thread", None)
    if thread is not None and thread.is_alive():
        return True
    if getattr(hub.host, "litho_worker", None) is not None:
        return True
    scan_task = getattr(hub.host, "_scan_task", None)
    return scan_task is not None and not scan_task.done()


def set_override(hub, body: dict | None = None) -> dict:
    """Override one instrument-model field; a dry run (`would_change`) unless `confirm`."""
    from ..instrument_model import apply_override

    body = dict(body or {})
    field = body.get("field")
    value = body.get("value")
    reason = body.get("reason") or ""
    client = body.get("client") or "cockpit"
    confirm = bool(body.get("confirm"))
    if not isinstance(field, str) or not field:
        return {"error": "override: field is required", "applied": False}

    try:
        model, config_path, overrides_log = load_current(hub)
    except (ModelIncomplete, tomllib.TOMLDecodeError) as exc:
        return {"error": str(exc), "applied": False}

    # Checked against the overridable set first: a name such as "to_dict" resolves by
    # getattr to a method, not a `Field`, and would 500 on `.value`.
    from ..instrument_model import OVERRIDABLE_FIELDS, Field
    if field not in OVERRIDABLE_FIELDS:
        return {"error": f"not an overridable field: {field!r}", "applied": False}
    current = getattr(model, field, None)
    if not isinstance(current, Field):
        return {"error": f"not an overridable field: {field!r}", "applied": False}

    # Platform re-derives the whole park plan (z_out_channel, z_park_v, z_rail_v, ...),
    # so it may not change under a live loop. Other fields are not gated by this.
    if field == "platform" and confirm and _run_active(hub):
        return {"error": "platform cannot be changed while engaged or a run is "
                         "active (the park plan changes underneath)",
               "reason": "engaged", "applied": False}

    if not confirm:
        return {"would_change": {"field": field, "old": current.value,
                                 "new": value},
               "applied": False}

    # A malformed value or a hardware platform with no tip_amp_gain is a clean refusal
    # the cockpit shows inline (section 14 item 3), never a 500.
    try:
        updated = apply_override(model, field, value, reason=reason, client=client,
                                 config_path=config_path, overrides_log=overrides_log)
    except (ValueError, TypeError) as exc:
        return {"error": str(exc), "reason": "invalid_params", "applied": False}
    except ModelIncomplete as exc:
        return {"error": str(exc), "reason": "model_incomplete", "applied": False}
    hub.host.instrument_model = updated
    return {"applied": True, "field": field, "value": getattr(updated, field).value}


_FAULT_CLEAR_LOCK = threading.Lock()


def _clear_igor_abort_flags(hub) -> tuple[bool | None, str]:
    """CLEARFAULT to Igor (2026-10-27): its stopped flag and fault flag go back to 0, so
    the next abort calls td_Stop again even after a park that never confirmed. Returns
    `(igor_cleared, detail)`: True when sent, False when the send failed, None with no Igor
    bridge (the twin). A failure is logged and noted (`hub.igor_clear_pending`). At the next
    connect, if Igor still reports a fault, it is latched again and the operator clears once
    more; otherwise the connect's CLEARFAULT resets Igor's stale flags
    (Hub._clear_stale_igor_abort_flags). The operator's clear stands either way."""
    from ..hub import real_bridge
    arc = real_bridge(hub)
    clear = getattr(arc, "clear_fault", None)
    if not callable(clear):
        hub.host.igor_clear_pending = True
        detail = ("no Igor bridge connected, so Igor's abort flags were not cleared; noted: "
                  "at the next connect a fault Igor still reports is latched again and must "
                  "be cleared once more")
        hub._log_sync(f"fault clear: {detail}")
        return None, detail
    try:
        clear()
    except Exception as exc:                             # noqa: BLE001
        hub.host.igor_clear_pending = True
        detail = (f"CLEARFAULT to Igor failed ({exc}); Igor's abort flags may still be "
                  f"set; noted: at the next connect a fault Igor still reports is latched "
                  f"again and must be cleared once more")
        hub._log_sync(f"WARNING fault clear: {detail}")
        return False, detail
    hub.host.igor_clear_pending = False
    return True, "CLEARFAULT sent"


def clear_fault(hub, body: dict | None = None) -> dict:
    """Clear the latched fault when `confirm` is true; without it, report the fault."""
    body = dict(body or {})
    if not bool(body.get("confirm")):
        return {"ok": True, "cleared": False, "fault": hub.fault}
    # read-and-clear under one lock: the doors run this in worker threads, and two pages
    # accepting together must log exactly one clear
    with _FAULT_CLEAR_LOCK:
        was = hub.fault
        hub.fault = None
    if was is None:
        # Nothing was latched: a park may be running, and CLEARFAULT would stop Igor's 5 s
        # backup under it, so Igor is left alone.
        return {"ok": True, "cleared": True}
    # the one record of an operator clear, and when (console + every page's log)
    hub._log_sync(f"fault cleared by operator at {time.strftime('%H:%M:%S')}: "
                  f"{was.get('reason') if isinstance(was, dict) else was}")
    igor_cleared, detail = _clear_igor_abort_flags(hub)
    return {"ok": True, "cleared": True, "igor_cleared": igor_cleared, "igor_detail": detail}
