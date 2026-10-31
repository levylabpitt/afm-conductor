"""The state board (the cortex): every read of "what is the instrument doing now".

getState, getStatus, getTelemetry, GET /api/state and the /ws telemetry frame are
answered here and nowhere else, so the later split into a separate process (H8)
replaces this one module. In-process the board holds no copy: each read is a pure
function of the hub's `HostState` (plus the door's run record and commander), so
the answer is never staler than the fields it reads and nothing has to hook the
many threads that write them (P3, r2-P3.md objection 1).

The safety guards (dead-man, rail guard, interlocks, fail-safe, park) read
`HostState` directly and must never import this module: a dead board must not
silence the dead-man (tests/board/test_guards_skip_board.py).

Staleness: the host stamps `_last_heard` on every `HostState.emit` and on its
own 0.5 s beat (host/beat.py); the base never stamps it. If the stamp is older than STALE_AFTER_S the board
answers `disconnected` rather than the last state: a stale "parked" would claim
safety falsely. A host that has never been heard (no tick yet) is not stale.

Pushes: only /ws pages subscribe (`stream.subscribe`). FLEX polls and is never pushed.
"""

from __future__ import annotations

from .. import clock
from ..instrument.twin import SUM_V as TWIN_SUM_V
from . import flex_dispatch as _fd
from .deps import _instrument_dict
from .design_store import autosave_status

STALE_AFTER_S = 2.0


def silent_for(hub) -> float | None:
    """Seconds since the host last spoke, or None if it has not yet."""
    heard = getattr(hub.host, "_last_heard", None)
    return None if heard is None else max(0.0, clock.now() - heard)


def disconnected_reason(hub) -> str | None:
    """Why the board says `disconnected`, or None while the host is live."""
    quiet = silent_for(hub)
    if quiet is None or quiet <= STALE_AFTER_S:
        return None
    return f"host silent for {quiet:.1f} s (limit {STALE_AFTER_S:g} s)"


def processes() -> dict | None:
    """Keeper and child pids from the keeper's state file, or None when it has no file."""
    from ..keeper import status as keeper_status
    body = keeper_status.read_state(keeper_status.state_path())
    if body is None:
        return None
    children = []
    for c in body.get("children", []):
        ready = c.get("ready")
        children.append({"name": c.get("name"),
                         "pid": ready["pid"] if ready else c.get("pid"),
                         "state": c.get("state")})
    return {"keeper_pid": body.get("keeper_pid"), "children": children}


def state(hub) -> str:
    """The FLEX state machine word (derivation table in flex_dispatch)."""
    if disconnected_reason(hub) is not None:
        return "disconnected"
    return _fd.get_state(hub)


def status(hub) -> dict:
    park = dict(getattr(hub.host, "_safe_park", None) or {})
    run = hub.runs.in_flight() or hub.runs.newest()
    progress, detail = _progress(hub)
    reason = disconnected_reason(hub)
    out = {
        "state": "disconnected" if reason else _fd.get_state(hub),
        "backend": hub.backend_label(),
        # The CONFIGURED backends (docs/FLEX_INTEGRATION_PLAN.md Sec J risk 2):
        # "backend" above stays the ARC's LIVE label (hardware/twin/disconnected,
        # derived from the bridge) for compatibility; "arc_backend" is the raw
        # sim/hardware config.
        "arc_backend": hub.host.backend,
        "run_id": run.run_id if run is not None else None,
        "run_kind": run.kind if run is not None else None,
        "progress": progress,
        "detail": detail,
        "safe_park": {"active": bool(park.get("active")),
                      "reason": park.get("last_reason"),
                      "verified": park.get("verified"),
                      "critical": bool(park.get("critical")),
                      "transport": park.get("transport")},
        # Single-writer arbitration (plan Sec E.2). Null when the instrument is
        # unclaimed -- the normal state for a GUI-only session. The token itself is
        # NEVER published here: `getStatus` is served by the read socket, which any
        # observer may poll, and a commander token on that surface would hand
        # control to whoever polls fastest.
        "commander": _fd._commander(hub),
        # IS THE DEAD-MAN ACTUALLY RUNNING? A dead watcher looks exactly like a
        # healthy one with nothing to do, so without this field an unattended script
        # could go on driving a tip whose dead-man stopped hours ago and nothing on
        # the wire would say so. The watcher re-arms itself; this is how a client
        # sees that it had to.
        "deadman_armed": _fd._ctl.deadman_armed(hub),
        "bridge": {"closed": bool(getattr(hub.host, "bridge_closed", None) is not None
                                  and hub.host.bridge_closed.is_set()),
                   "closed_at": getattr(hub.host, "bridge_closed_at", None)},
    }
    if reason:
        out["disconnected_reason"] = reason
    procs = processes()
    if procs is not None:
        out["processes"] = procs
    return out


def telemetry(hub) -> dict:
    """The last cached frame -- never a new instrument round-trip (plan Sec A.3).

    On HARDWARE this is the always-on TELEM poll's snapshot (`hub._hw_telemetry`),
    which milestone 1 gave a `t` and unconditional micrometre XY. On the TWIN there is
    no poll, so the SAME SHAPE is built from the twin's own state: the plan flagged
    this as a decision ("either the twin gains a poll that fills the cache, or
    getTelemetry says available:false"), and a rehearsal script that cannot see
    telemetry cannot rehearse the thing telemetry is for. The twin's `sum_v` is a
    constant 10 V, as a healthy detector reads.
    """
    now = clock.wall()
    frame = getattr(hub.host, "_hw_telemetry", None)
    if hub.backend_live() == "hardware" and isinstance(frame, dict):
        t = frame.get("t")
        return {"t": t, "age_s": (now - t) if t is not None else None,
                "x_um": frame.get("x_um"), "y_um": frame.get("y_um"),
                "zdrive_v": frame.get("zdrive_v"), "defl_v": frame.get("defl_v"),
                "sum_v": frame.get("sum_v"), "amp_v": frame.get("amp_v"),
                "row": frame.get("row"), "backend": "hardware"}
    if hub.backend_label() == "disconnected":
        # Configured hardware, no live bridge: there is no frame and there will not
        # be one. Nulls with a null age (plan: "if there is no frame yet, return
        # nulls with age_s: null").
        return _empty_telemetry("disconnected")
    arc = getattr(hub.host, "arc", None)
    if arc is None:
        return _empty_telemetry("twin")
    try:
        from .twin_plugin import twin_snapshot
        snap = twin_snapshot(hub, arc.snapshot())
        world = getattr(arc, "world", None)
        finite = _fd._finite
        return {"t": now, "age_s": 0.0,          # computed on demand -> zero age
                "x_um": finite(snap.get("x_um")), "y_um": finite(snap.get("y_um")),
                "zdrive_v": finite(getattr(world, "z_v", None)),
                "defl_v": finite(snap.get("defl_v")),
                "sum_v": TWIN_SUM_V,
                "amp_v": finite(snap.get("amp_v")),
                "row": getattr(hub.host, "_scan_row", None), "backend": "twin"}
    except Exception as exc:
        raise _fd.RpcError(_fd.INTERNAL_ERROR, f"twin telemetry unavailable: {exc}") from exc


def _empty_telemetry(backend: str) -> dict:
    return {"t": None, "age_s": None, "x_um": None, "y_um": None, "zdrive_v": None,
            "defl_v": None, "sum_v": None, "amp_v": None, "row": None,
            "backend": backend}


def _progress(hub) -> tuple[float | None, str | None]:
    """Best-effort `(progress, detail)` for getStatus. Never raises; a status read
    that cannot compute progress reports null rather than failing."""
    try:
        word = _fd.get_state(hub)
        if word == "writing":
            done = len(getattr(hub.host, "written_accum", None) or [])
            # THIS RUN'S submitted count, stashed on the record at allocation. The
            # whole pattern is the wrong denominator: writing one object out of
            # forty would report 2.5% and never reach 100%.
            run = hub.runs.in_flight()
            total = int((run.stats or {}).get("objects_submitted") or 0) \
                if run is not None and run.kind == "write" else 0
            frac = (done / total) if total else None
            return frac, f"object {done}/{total}" if total else f"{done} object(s)"
        if word == "scanning":
            row = getattr(hub.host, "_scan_row", None)
            inst = getattr(hub.host, "inst", None)
            scan = getattr(inst, "scan", None) if inst is not None else None
            cfg = getattr(scan, "config", None) if scan is not None else None
            rows = int(getattr(cfg, "lines", 0) or 0) if cfg is not None else 0
            if row is not None and rows:
                return min(1.0, row / rows), f"line {row}/{rows}"
            sr = getattr(hub.host, "scan_repeat", None) or {}
            if sr.get("enabled"):
                return None, f"continuous frame {sr.get('count', 0)}"
            return None, f"line {row}" if row is not None else None
        if word == "parked":
            park = getattr(hub.host, "_safe_park", None) or {}
            return None, park.get("last_reason")
        if word == "fault":
            fault = getattr(hub, "fault", None) or {}
            return None, fault.get("reason")
    except Exception:                            # noqa: BLE001
        return None, None
    return None, None


def _is_ac() -> bool:
    """Amplitude is meaningful only in AC mode."""
    from .host.telemetry import current_scan_is_ac
    return current_scan_is_ac()


def snapshot(hub) -> dict:
    """GET /api/state: the service's live state (docs/FLEX_INTEGRATION_PLAN.md Sec C.1).

    On hardware, polled TELEM values replace the twin's (same as the /ws frame).
    Extra fields:

      backend: "hardware" | "twin" | "disconnected"   (hub.backend_label, LIVE)
      arc_backend: CONFIGURED "sim" | "hardware"
      state:   "idle" | "scanning" | "writing" | "fault"
      bridge:  {closed, closed_at}
      fault:   the latched fault record, or null

    `disconnected` / `fault` are terminal for the process (plan Sec J risk 6):
    a script that reaches them must stop, not wait.
    """
    from .twin_plugin import twin_snapshot
    snap = twin_snapshot(hub, hub.host.arc.snapshot())
    live_hw = hub.backend_live() == "hardware"
    hwt = hub.host._hw_telemetry if live_hw else None
    if hwt is not None:
        # Real instrument values in place of the twin's stubs (as the /ws heartbeat does).
        for key in ("x_um", "y_um", "defl_v"):
            if hwt.get(key) is not None:
                snap[key] = hwt[key]
        snap["amp_v"] = hwt.get("amp_v") if _is_ac() else None
        snap["bias_v"] = None            # not polled; do not report the twin's stub
        snap["engaged"] = bool(hub.host.engaged)
        snap["hw_telemetry"] = hwt
    reason = disconnected_reason(hub)
    snap.update({
        "backend": hub.backend_label(),
        # CONFIGURED backends; "backend" above is the ARC's LIVE label.
        "arc_backend": hub.host.backend,
        "state": "disconnected" if reason else hub.state_label(),
        "bridge": {"closed": hub.host.bridge_closed.is_set(),
                   "closed_at": hub.host.bridge_closed_at},
        "fault": hub.fault,
        "sample": hub.sample_name,
        "speed": hub.host.speed,
        # UNION, not state_label(): on hardware the scan-pump task is the one scanning.
        # `scanning` and `writing` are independent flags; `state` is exclusive.
        "scanning": bool(hub.host._scan_task is not None and not hub.host._scan_task.done()),
        # Lets FLEX/MCP/user-twin know the tip is busy without trying a second write.
        "writing": bool(hub.host.litho_thread and hub.host.litho_thread.is_alive()),
        # LEGACY park summary; per-park sensor evidence is GET /api/park/last
        # (FLEX getLastPark, /api/instrument "last_park"), from hub.last_park.
        "safe_park": dict(hub.host._safe_park),
        "design_autosave": autosave_status(),
        "lever": {"f0_khz": hub.host.world.lever.f0 / 1e3, "Q": hub.host.world.lever.Q,
                  "k_nm": hub.host.world.lever.k},
        "instrument": _instrument_dict(hub.host.instrument),
    })
    if reason:
        snap["disconnected_reason"] = reason
    return snap


def _twin_hw_telemetry(hub, snap: dict) -> dict:
    """The twin's `hw_telemetry` block: the same keys and units as the hardware poll's
    (`host/telemetry._hw_telemetry_frame`), filled from the twin's simulated channels.
    `sum_v` is the twin's constant 10 V; `height_nm` is None on hardware too (ZSensor
    is not polled)."""
    finite = _fd._finite
    world = getattr(hub.host.arc, "world", None)
    plugin = getattr(hub.host, "_twin_plugin", None)
    if plugin is not None:
        frame = plugin.frame_at(clock.now())
        return {
            "t": clock.wall(), "row": getattr(hub.host, "_scan_row", None),
            "x_um": finite(frame.x_um), "y_um": finite(frame.y_um),
            "defl_v": finite(frame.deflection_v), "zdrive_v": finite(frame.z_drive_v),
            "sum_v": TWIN_SUM_V, "amp_v": finite(frame.amplitude_v), "height_nm": None,
        }
    return {
        "t": clock.wall(),
        "row": getattr(hub.host, "_scan_row", None),
        "x_um": finite(snap.get("x_um")),
        "y_um": finite(snap.get("y_um")),
        "defl_v": finite(snap.get("defl_v")),
        "zdrive_v": finite(getattr(world, "z_v", None)),
        "sum_v": TWIN_SUM_V,
        "amp_v": finite(snap.get("amp_v")),
        "height_nm": None,
    }


def ws_frame(hub) -> dict:
    """One /ws telemetry frame, built per connection per tick. Not journalled."""
    from .twin_plugin import twin_snapshot
    snap = twin_snapshot(hub, hub.host.arc.snapshot())
    snap["type"] = "telemetry"
    # On the twin a scan is the pump task (the plugin plays it); on hardware this flag has
    # never been set here.
    snap["scanning"] = bool(
        hub.hw_bridge() is None and hub.host._scan_task is not None
        and not hub.host._scan_task.done())
    # On hardware, ADD the real polled channels. A hardware-configured station with no
    # bridge yet sends no block at all: nothing is claimed about the instrument.
    if hub.hw_bridge() is not None and hub.host._hw_telemetry is not None:
        hwt = hub.host._hw_telemetry
        # Replace the twin-stubbed electrical fields with real ones:
        # defl_v always; amp_v only in AC mode; bias_v is not polled -> None.
        snap["defl_v"] = hwt.get("defl_v")
        snap["amp_v"] = hwt.get("amp_v") if _is_ac() else None
        snap["bias_v"] = None
        # The twin is not the one scanning on hardware; use the scan-pump
        # task (set for the hardware scan path too, host/scan_pump.py).
        task = hub.host._scan_task
        snap["scanning"] = bool(task is not None and not task.done())
        snap["hw_telemetry"] = hwt
    elif hub.host.backend != "hardware":
        snap["hw_telemetry"] = _twin_hw_telemetry(hub, snap)
        # Same display rules as hardware: amplitude only in AC mode, bias not polled.
        snap["amp_v"] = snap["hw_telemetry"]["amp_v"] if _is_ac() else None
        snap["bias_v"] = None
    return snap
