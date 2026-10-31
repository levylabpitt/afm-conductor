"""Fresh XY LVDT calibration for `hub.hw_cal`.

The Cypher recomputes XLVDTSens / YLVDTSens on every tip or focus motor move
(MotorControl.ipf AdjustXYLVDTSens, sample-height correction) and writes them back
into the same GV globals, so the values read at connect go stale after any coarse
approach (+3.3 % X / +3.6 % Y measured raised vs engaged, INSTRUMENT_MODEL_SPEC
13.1 item 34). `refresh_xy_cal` re-reads them over the bridge (four GETGV reads;
the STARTUP payload stays the connect-time read) and updates
`hub.hw_cal` IN PLACE, so every holder of that dict (telemetry, the scan pump,
`arc.startup` when it is the same object) sees the new values.

The conversion itself is unchanged: um = (V - offset) * sens * 1e6.
"""

from __future__ import annotations

import asyncio
import math

from .. import clock

# The STARTUP keys that carry the XY LVDT calibration (AFM_GetCal in afm_igor.ipf),
# each with the Asylum global it is read from.
XY_CAL_GV = {"x_lvdt_sens": "XLVDTSens", "y_lvdt_sens": "YLVDTSens",
             "x_lvdt_offset": "XLVDTOffset", "y_lvdt_offset": "YLVDTOffset"}
XY_CAL_KEYS = tuple(XY_CAL_GV)

# Bounded so a wedged bridge delays a scan start by at most this, not the 30 s default.
REFRESH_TIMEOUT_S = 5.0

# A change smaller than this is not worth a log line (the GV values are stored at
# ~6 significant figures; a real motor move changes them by percent).
LOG_DELTA_FRAC = 0.001

# Plausibility vs the previous values. A motor move changes the sensitivities by a
# few percent (3.3 / 3.6 % raised vs engaged) and leaves the offsets alone; a bigger
# jump is a bad read, and a 100x-off sensitivity would put ~300 V on the XY setpoint.
MAX_SENS_CHANGE_FRAC = 0.10
MAX_OFFSET_CHANGE_V = 0.5

# The continuous loop re-references (approach) and then starts the frame's scan
# back to back; a second read within this window with no error is skipped.
DEBOUNCE_S = 1.0


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None


def _valid(fresh: dict, old: dict) -> str | None:
    """None when all four values are usable, else why not; any failure refuses the
    whole set. A zero or non-finite sensitivity would put inf volts on the XY
    drive; a jump beyond the plausibility bounds from a usable previous value is
    treated as a bad read."""
    for k in XY_CAL_KEYS:
        if _num(fresh.get(k)) is None:
            return f"{k}={fresh.get(k)!r} is not a finite number"
    for k in ("x_lvdt_sens", "y_lvdt_sens"):
        n, o = fresh[k], _num(old.get(k))
        if n == 0:
            return f"{k}=0"
        if o and abs(n - o) > MAX_SENS_CHANGE_FRAC * abs(o):
            return (f"{k} {o:.6g} -> {n:.6g} is more than "
                    f"{MAX_SENS_CHANGE_FRAC:.0%} from the previous value")
    for k in ("x_lvdt_offset", "y_lvdt_offset"):
        n, o = fresh[k], _num(old.get(k))
        if o is not None and abs(n - o) > MAX_OFFSET_CHANGE_V:
            return (f"{k} {o:.6g} -> {n:.6g} V moved more than "
                    f"{MAX_OFFSET_CHANGE_V:g} V from the previous value")
    return None


UNSUPPORTED_MSG = ("afm_litho_run.ipf has no GETGV; XY calibration will not refresh; "
                   "reinstall the ipf and restart the service")

LARGE_CHANGE_HINT = ("reconnect the bridge (restart the service) to accept a large "
                     "calibration change")


def _status(hub) -> dict:
    st = getattr(hub.host, "xy_cal_status", None)
    if st is None:
        st = {"refreshed_at": None, "source": None, "error": None}
        hub.host.xy_cal_status = st
    return st


def note_xy_cal(hub, source: str) -> None:
    """Stamp the current `hub.hw_cal` as read now (used at connect, where
    `arc.start()` already did the read)."""
    st = _status(hub)
    st.update(refreshed_at=clock.wall(), source=source, error=None)


def probe_getgv(hub, arc) -> str | None:
    """At connect: one GETGV read. No reply means the installed ipf predates the
    verb; mark refresh unsupported so later scan/approach/write starts do not each
    wait out the timeout. Returns the one log line to print, or None."""
    st = _status(hub)
    try:
        arc.get_gv("XLVDTSens", timeout=REFRESH_TIMEOUT_S)
    except Exception:                                 # noqa: BLE001
        st["unsupported"] = True
        st["error"] = UNSUPPORTED_MSG
        return UNSUPPORTED_MSG
    st["unsupported"] = False
    return None


def refresh_xy_cal(hub, source: str) -> list[str]:
    """Blocking: re-read the XY calibration from the bridge into `hub.hw_cal`.

    Returns the log lines to emit. No-op (empty list) on the twin, whose cal does
    not drift. Never raises: on a failed or invalid read the previous values are
    kept and the error is recorded in `hub.xy_cal_status`, because a stale
    calibration is a placement error, not a reason to refuse the scan."""
    from .hub import real_bridge
    arc = real_bridge(hub)
    if arc is None or not hasattr(arc, "get_gv"):
        return []
    st = _status(hub)
    if st.get("unsupported"):
        return []
    last = st.get("refreshed_at")
    if (st.get("error") is None and last is not None
            and clock.wall() - last < DEBOUNCE_S):
        return []
    try:
        fresh = {k: arc.get_gv(gv, timeout=REFRESH_TIMEOUT_S)
                 for k, gv in XY_CAL_GV.items()}
    except Exception as exc:                          # noqa: BLE001
        st["error"] = f"GETGV read failed: {type(exc).__name__}: {exc}"
        return [(f"XY cal refresh ({source}) FAILED, keeping previous values: "
                 f"{st['error']} (an afm_litho_run.ipf without GETGV cannot answer)")]
    cal = hub.host.hw_cal
    old = {k: cal.get(k) for k in XY_CAL_KEYS}
    why = _valid(fresh, old)
    if why is not None:
        st["error"] = f"invalid read: {why}; {LARGE_CHANGE_HINT}"
        return [(f"XY cal refresh ({source}) REJECTED, keeping previous values: "
                 f"{why}; {LARGE_CHANGE_HINT}")]
    cal.update({k: float(fresh[k]) for k in XY_CAL_KEYS})
    st.update(refreshed_at=clock.wall(), source=source, error=None)
    changed = []
    for k in XY_CAL_KEYS:
        o, n = old[k], cal[k]
        if not isinstance(o, (int, float)) or o == 0:
            if o != n:
                changed.append(f"{k} {o!r} -> {n:.6g}")
            continue
        frac = (n - o) / abs(o)
        if abs(frac) > LOG_DELTA_FRAC:
            changed.append(f"{k} {o:.6g} -> {n:.6g} ({frac * 100:+.2f} %)")
    if not changed:
        return []
    return [f"XY cal refreshed ({source}): " + "; ".join(changed)]


async def refresh_xy_cal_async(hub, source: str) -> None:
    """`refresh_xy_cal` off the event loop, with its lines sent to the hub log."""
    from .hub import real_bridge
    if real_bridge(hub) is None:
        return
    lines = await asyncio.to_thread(refresh_xy_cal, hub, source)
    for line in lines:
        await hub.log(line)


def xy_cal_payload(hub) -> dict:
    """The `xy_cal` object of GET /api/instrument / FLEX getInstrument. Stable
    between calls (no wall-clock age), so REST == FLEX idempotency holds."""
    cal = getattr(hub.host, "hw_cal", None) or {}
    st = getattr(hub.host, "xy_cal_status", None) or {}
    out = {k: (float(cal[k]) if isinstance(cal.get(k), (int, float)) else None)
           for k in XY_CAL_KEYS}
    out["refreshed_at"] = st.get("refreshed_at")
    out["source"] = st.get("source")
    out["error"] = st.get("error")
    out["unsupported"] = bool(st.get("unsupported", False))
    return out
