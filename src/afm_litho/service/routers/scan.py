"""The scan domain: approach/withdraw, raster scans, force curves and tuning.

Everything here is a door. `scan_start` arms a raster on the instrument and hands the
streaming to the scan pump, which now lives in `service/host/scan_pump.py` (the two
pumps, their persistence and the drift registration -- the streaming half, which no
wire verb starts on its own).

THE VERB BODIES LIVE IN `service/commands/scan_cmds.py` (docs/
FLEX_INTEGRATION_PLAN.md Sec C). `/api/approach`, `/api/mode`, `/api/withdraw`,
`/api/zgain` and `/api/scan/start` are thin adapters over that service layer, so the
cockpit and a FLEX script meet the same guards, the same single-writer claim and the
same state machine. What stays here: those adapters and the force-curve and tune
endpoints.
"""

from __future__ import annotations

import asyncio

from ..host import scan_cmds
from ..host.api import HostApi
from ..commands.errors import CommandError
from ..hub import hub
from ..host.safe_park import (
    PARK_WITHDRAW,
    STOP_REPEAT_LOOP,
    _signal_stop,
    request_park,
)



# ------------------------------------------------------------------ adapters ----
# Every endpoint below is a THIN ADAPTER over `service/commands/scan_cmds.py`
# (docs/FLEX_INTEGRATION_PLAN.md Sec C). The guards, the single-writer claim, the
# state machine and the validation all live in the service layer, because a FLEX
# script arrives through `flex_dispatch` and must meet EXACTLY the same ones: "one
# validation path, one guard set, one state machine" is the whole of plan goal 3, and
# it is delivered by the shared layer rather than by changing the panel's transport
# (plan Sec D).
#
# An adapter's only job is the HTTP dialect. This service answers HTTP 200 with
# `{"error": ..., "reason": ...}` bodies rather than status codes, and a large part of
# the suite asserts those bodies verbatim -- they are the regression guard that the
# extraction did not change what the cockpit sees. `CommandError.http_body()` carries
# the exact legacy shape from the raise site.


async def approach(body: dict | None = None):
    """Fine engage (`commands.scan_cmds.approach`), blocking, returning the engage snapshot.
    `POST /api/approach` is served by the startApproach tip row (requests/tips.py); this
    function stays for in-process callers (the continuous loop, tests)."""
    try:
        return await scan_cmds.approach(hub, body)
    except CommandError as exc:
        return exc.http_body()


def _engaged_or_scanning() -> tuple[bool, str | None]:
    """The "tip is on the surface (or about to be)" gate, against the process hub.
    The rule itself is `commands.scan_cmds.engaged_or_scanning`; this is the
    no-argument form the litho router and the tests have always imported."""
    return scan_cmds.engaged_or_scanning(hub)


async def withdraw(body: dict | None = None):
    """Retract. On the hardware backend: stop any running scan, stop the Z
    feedback loop, then re-arm loop 2 at -10 V with output minimum 0 so Height parks
    at 0 V (the retracted/safe position). On the twin: the simulated withdraw."""
    return await do_withdraw(hub, body)


async def do_withdraw(hub, body: dict | None = None):
    """The retract itself, taking its hub EXPLICITLY
    (`commands.scan_cmds.withdraw`).

    The dead-man and the engaged-idle cap (host/deadman.py `_withdraw`) make the
    same `request_park` call with their own hub: an endpoint that closes over the
    process-global hub silently ignores whichever hub the caller meant.
    """
    return await request_park(hub, "withdraw", PARK_WITHDRAW, body=body)


async def scan_params(body: dict | None = None):
    """A live scan-parameter change from the panel (size/rate/points/lines/
    angle/mode/setpoint). The twin builds its ScanConfig from the scan/start
    body, so this just records the latest values -- its real job is to give
    mirror mode a request to echo, so changing a control writes to Igor at once.
    """
    hub.scan_params.update(body or {})
    return {"ok": True}


async def scan_start(cfg: dict):
    """Arm a raster and hand the streaming to a pump (`commands.scan_cmds.
    scan_start`). The silent-twin guard, the ScanConfig coercion, the single-writer
    claim and the run-id allocation live there, so the wire's `startScan` gets all of
    them; the pumps themselves live in `host/scan_pump.py`."""
    try:
        return await scan_cmds.scan_start(hub, cfg)
    except CommandError as exc:
        return exc.http_body()


async def scan_last():
    """The most recently cached scan frame (newest on disk), for the litho
    cockpit background. 404 when nothing has been scanned yet."""
    from fastapi import HTTPException

    from ..scan_cache import load_latest_scan_frame
    frame = await asyncio.to_thread(load_latest_scan_frame, hub.store.path.parent)
    if not frame:
        raise HTTPException(status_code=404, detail="no scan cached yet")
    # A partial/in-progress cached frame carries NaN for unscanned pixels (and could
    # carry inf); Starlette's JSONResponse serializes with allow_nan=False and 500s
    # on any non-finite float. Map NaN/inf -> None in the float grids (the frontend
    # already paints null as unscanned) so the instrument's partial frames serve
    # cleanly instead of erroring.
    import math
    for key, val in frame.items():
        if isinstance(val, list):
            frame[key] = [None if isinstance(x, float) and not math.isfinite(x) else x
                          for x in val]
    return frame


async def scan_abort(body: dict | None = None):
    """The operator Stop on the process hub (`host.scan_cmds.stop_scan`): stop the scan
    and retract the tip."""
    return await scan_cmds.stop_scan(hub, body)


async def safe_park_endpoint(body: dict | None = None, stop_id: str | None = None):
    """Operator ABORT = the unified fail-safe primitive: stop everything and get the
    tip to a safe spot (docs/SAFETY_KERNEL_PLAN.md Sec 0). The one button behind the
    two triggers -- operator abort OR any auto-detected failure -- so the Abort
    control and every failure path share ONE implementation (service/host/
    safe_park.py). Out-of-band on hardware (COM-first, TCP-fallback) with a Z-drive
    readback verify; a stop + withdraw on the twin. Breaks any continuous loop too.
    """
    body = body or {}
    reason = str(body.get("reason") or "operator abort")
    # Break the continuous auto-repeat loop at once (an operator abort of an
    # unattended run); the supervisor sees it and arms no further frame. safe_park
    # itself also does this, but doing it here makes the endpoint self-contained.
    if hub.host.scan_repeat.get("enabled"):
        hub.host.scan_repeat["enabled"] = False
        hub.host.scan_repeat["stopped_reason"] = "safe_park"
        _signal_stop(hub, STOP_REPEAT_LOOP)
    res = await HostApi(hub).park(reason, stop_id=stop_id)
    # `transport` + `com_error` + `verify_timeout` are surfaced deliberately: a
    # COM-first trigger that silently fell back to TCP, or a verify read that gave
    # up, must be visible to the operator and to a scripted caller, never inferred.
    return {"ok": True, "safe_park": {k: res.get(k) for k in
            ("reason", "backend", "transport", "com_error", "verified", "z_v",
             "critical", "verify_timeout")}}


def _twin_lacks(what: str) -> str:
    return f"{what} is not available on the twin"


async def force_curve():
    return {"error": _twin_lacks("a force curve")}


async def tune_thermal():
    return {"error": _twin_lacks("a thermal tune")}


async def tune_sweep():
    return {"error": _twin_lacks("a frequency sweep")}
