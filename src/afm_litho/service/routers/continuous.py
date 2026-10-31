"""Continuous imaging: the endpoints behind the Scan tab's auto-repeat toggle.

The loop itself (the supervisor, the rail-dwell guard and the free-air and SUM guards)
lives in `service/host/continuous.py`; this module keeps the adapters the wire rows and
in-process callers use to read and set it. Like routers/drift.py, it owns the
enable/disable and validates its input, returning `{"error": ...}` rather than a 500.
"""

from __future__ import annotations

from fastapi import APIRouter

from ..commands.errors import CommandError
from ..host.continuous import _state
from ..hub import hub


router = APIRouter()


async def continuous_get():
    """The current continuous-imaging state (toggle, frame count, elapsed, bounds,
    consecutive errors) so the Scan panel can hydrate and poll it."""
    return _state()


async def continuous_set(body: dict | None = None):
    """Enable or disable the auto-repeat scan loop. Body:
    `{enabled, config?, max_frames?, max_seconds?, interval_s?, withdraw_on_stop?,
    rail?}`.

    A THIN ADAPTER over `commands.continuous_cmds.set_continuous` (docs/
    FLEX_INTEGRATION_PLAN.md Sec C): the arming, the bounds, the rail-guard
    only-tighten pass-through and the SUM baseline live in the service layer, so the
    wire's `setContinuous` arms an unattended run through exactly the same code the
    panel's toggle does. The supervisor and every guard it drives live in `host/continuous.py`.
    """
    from ..host.continuous_cmds import set_continuous
    try:
        return await set_continuous(hub, body)
    except CommandError as exc:
        return exc.http_body()


async def relative_set(body: dict | None = None):
    """Configure the relative (drift-referenced) setpoint for continuous imaging.
    Body: `{enabled?: bool, settle_s?}`. Either field may be sent alone (the rest
    keep their current value). Returns the full continuous state (the relative
    config rides along in `relative`).

    A SINGLE toggle: when `enabled`, the effective per-frame setpoint tracks the
    free-air baseline as free_air + the CURRENT scan setpoint -- there is no
    separate offset, the scan setpoint IS the target force above free-air.

    Contact-mode only: `enabled` may be set here, but the re-reference is INERT
    unless the running continuous frame is contact mode (see `_relative_active` --
    the vendor records a free-air deflection baseline only for the contact family,
    ScanMaster.ipf:4665). Takes effect from the NEXT frame's between-frame
    boundary; changing it mid-run does not disturb the in-flight frame.
    """
    body = body or {}
    rel = hub.host.relative_setpoint
    if "enabled" in body:
        rel["enabled"] = bool(body["enabled"])
    if body.get("settle_s") not in (None, ""):
        rel["settle_s"] = max(0.0, float(body["settle_s"]))
    cfg = hub.host.scan_repeat.get("config") or {}
    inert = (rel["enabled"]
             and str(cfg.get("mode", "contact")).lower() != "contact")
    await hub.log(
        f"Relative setpoint: {'on' if rel['enabled'] else 'off'} "
        f"(settle {rel['settle_s']:g} s)"
        + ("; INERT -- current frame is not contact mode" if inert else ""))
    await hub.broadcast({"type": "relative_setpoint", "enabled": rel["enabled"],
                         "settle_s": rel["settle_s"],
                         "free_air_v": rel.get("last_free_air_v"),
                         "effective_v": rel.get("last_effective_v")})
    return _state()
