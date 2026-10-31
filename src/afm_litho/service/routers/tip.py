"""The live tip-position marker: the DISPLAY toggle for the always-on TELEM poll.

`POST /api/tip/show {on}` sets a HOST display flag (hub._show_tip_display). The
Igor-side telemetry+safety poll (AFM_TipTick) is ALWAYS running -- it pushes one
`TELEM <x> <y> <zdrive> <defl> <sum>` per tick over the SOCKIT link regardless of
this toggle, because those frames also carry the safety channels (rail-dwell / SUM /
deflection). The router's always-on consumer (host/telemetry._hw_telemetry_frame)
converts the RAW tip volts to lab microns via `hub.hw_cal` and broadcasts a dedicated
`tip_pos{x_um,y_um}` message the cockpit draws as a live marker/trail -- but only WHEN
this display flag is on. So the toggle governs the marker DISPLAY, never the poll.

This is ADDITIVE: it is a low-resolution real-time marker and does NOT gate or
replace the per-object end-of-object capture (the high-resolution written record).
The marker streams idle and during a pass alike (the poll is always on), so it needs
no per-pass reader wiring any more.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..hub import hub
from ..hub import real_bridge as _real_bridge

router = APIRouter()


async def tip_show(body: dict | None = None):
    """Toggle the live tip-marker DISPLAY. Body: {on: bool} (default true).

    Sets the host display flag either way. Hardware-guarded like the litho write:
    `hub.hardware_arc` is None off-instrument, so that branch is a clean twin no-op
    (no bridge). On the instrument it ALSO sends `SHOWTIP ON|OFF` over the bridge --
    send-only, exactly like the litho abort path (BridgeLink.send is locked, so it
    can't interleave with a worker's mid-arm send while a pass owns the read side).
    The .ipf treats SHOWTIP as display-only now (it never stops the always-on poll)."""
    body = body or {}
    on = bool(body.get("on", True))
    # DISPLAY intent, host-side: the telemetry+safety poll ALWAYS runs (it carries the
    # rail-dwell / SUM channels), so this toggle only controls whether the host
    # broadcasts the tip_pos marker from each TELEM frame (see
    # host/telemetry._hw_telemetry_frame). It never stops the poll.
    hub._show_tip_display = on
    hw = _real_bridge(hub)
    if hw is None:
        # twin / dev machine: no bridge, so nothing streams. Keep it a clean no-op
        # (the toggle still reflects intent) rather than an error.
        await hub.log(f"Show tip {'on' if on else 'off'}: no hardware bridge "
                      f"(twin no-op)")
        # backend_label, not backend_live: a hardware station whose bridge has not
        # connected reports "disconnected", so a page cannot take it for the twin.
        return {"ok": True, "on": on, "backend": hub.backend_label()}
    from ...hardware import bridge_protocol as P
    try:
        hw.link.send(P.cmd_showtip(on))
    except Exception as exc:
        # A failed bridge send is a hardware-action failure, not a success: report it
        # with an HTTP error STATUS so a status-code-only client never reads it as ok.
        # 502 Bad Gateway -- the service is a gateway to the instrument and the
        # instrument bridge send failed. (Was: HTTP 200 {"ok": False, "error": ...},
        # which a client checking only the status code saw as success.)
        await hub.log(f"Show tip send failed: {exc}")
        raise HTTPException(
            status_code=502,
            detail=f"tip show send failed: {exc}") from exc
    await hub.log(f"Show tip {'on' if on else 'off'} (hardware): tip marker "
                  f"{'displayed from the always-on poll' if on else 'hidden'}")
    # literal: this branch has already sent SHOWTIP over the live bridge.
    return {"ok": True, "on": on, "backend": "hardware"}
