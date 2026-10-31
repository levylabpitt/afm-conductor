"""The Connect tab, Phase 2: Track and compensate drift.

Once electrodes are marked (Phase 1 ports), the operator designates a scan as
the drift REFERENCE and flips the toggle on. Every completed scan is then
registered against that reference by `service.drift`; the (dx, dy) estimate is
always MEASURED and broadcast, and -- while the toggle is on -- the ports track
the moving features and the next scan's offset compensates for the drift.

This router owns the toggle and the reference; the actual per-scan estimation
happens in the scan pump (host/scan_pump.py) after a frame is cached. Handlers
validate their input and return `{"error": ...}` rather than a 500, mirroring
the Connect (Phase 1) and design routers.
"""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter

from .. import drift as drift_mod
from ..hub import hub
from ..scan_cache import load_latest_scan_frame

router = APIRouter()


def _capture_reference() -> dict | None:
    """Snapshot the newest cached scan frame as a drift reference, or None when
    nothing has been scanned yet."""
    frame = load_latest_scan_frame(hub.store.path.parent)
    if not frame:
        return None
    frame.setdefault("t", time.time())
    return drift_mod.frame_to_reference(frame)


def _state() -> dict:
    """The JSON-safe drift state for the UI to hydrate/plot."""
    d = hub.drift
    return {
        "enabled": bool(d.get("enabled")),
        "has_reference": d.get("reference") is not None,
        "reference": drift_mod.reference_meta(d.get("reference")),
        "total": list(d.get("total", [0.0, 0.0])),
        "track": list(d.get("track", [])),
    }


async def drift_get():
    """The current drift state (toggle, reference geometry, accumulated total,
    and the per-scan track) so the Connect panel can hydrate and plot it."""
    return _state()


async def drift_reference(body: dict | None = None):
    """Designate a scan as the drift REFERENCE and clear the track/total.

    Captures the CURRENT newest cached scan frame (no body needed). Returns the
    reference geometry, or `{"error": ...}` when no scan has been cached yet.
    """
    ref = await asyncio.to_thread(_capture_reference)
    if ref is None:
        return {"error": "no scan cached yet -- run a scan first, then set the "
                         "reference"}
    hub.drift["reference"] = ref
    hub.drift["track"] = []
    hub.drift["total"] = [0.0, 0.0]
    await hub.log(f"Drift reference set from the latest {ref['rows']}x"
                  f"{ref['pixels']} scan ({ref['size_um']:g} µm)")
    await hub.broadcast({"type": "drift", "reference_set": True, **_state()})
    return _state()


async def drift_enable(body: dict | None = None):
    """Turn Track-and-compensate-drift on or off. Body: `{enabled: bool}`.

    Enabling with no reference set auto-captures the latest scan as the
    reference; if there is no cached scan either, it returns `{error}` asking to
    scan first and leaves the toggle off.
    """
    body = body or {}
    enabled = bool(body.get("enabled", True))
    if enabled and hub.drift.get("reference") is None:
        ref = await asyncio.to_thread(_capture_reference)
        if ref is None:
            return {"error": "no scan cached yet -- run a scan first so drift has "
                             "a reference to track against"}
        hub.drift["reference"] = ref
        hub.drift["track"] = []
        hub.drift["total"] = [0.0, 0.0]
        await hub.log(f"Drift reference auto-captured from the latest "
                      f"{ref['rows']}x{ref['pixels']} scan")
    hub.drift["enabled"] = enabled
    await hub.log(f"Track and compensate drift: {'ON' if enabled else 'off'}")
    await hub.broadcast({"type": "drift", **_state()})
    return _state()
