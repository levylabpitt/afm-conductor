"""`setContinuous` -- the auto-repeat imaging loop, as a service-layer verb.

Truth document: docs/FLEX_INTEGRATION_PLAN.md Sec A (`setContinuous` -- "existing
rail-guard and interlock config passes through untouched; the provider may only
*tighten*, never loosen"), C (this layer).

WHAT MOVED
----------
The body of `POST /api/scan/continuous` lives here; the router keeps the decorated
endpoint as an adapter. The SUPERVISOR and every guard it drives (`_supervise`,
`_run_one_frame`, `_rereference` with its free-air 3-read guard and 0.9*SUM clamp,
the rail poll, the interlocks) stay in `service/host/continuous.py` and are
untouched -- they are the run, not the start verb, and the plan's extraction rule is
"per verb, never a big-bang refactor".

THE HUB PARAMETER, HONESTLY
---------------------------
This verb takes its hub explicitly like the rest of the layer, but the supervisor it
arms reads the PROCESS hub (it always has). So `hub` here must be that hub; it is a
parameter for uniformity and for the day the supervisor takes one too, not a promise
that a second hub would work. Nothing in the service passes anything else.
"""

from __future__ import annotations

import math
import time

from ...hardware.bridge_protocol import spawn_untraced
from ..commands.errors import BACKEND_MISMATCH, BUSY, INVALID_PARAMS, CommandError

# The bounds a caller may set. `max_frames` is an integer count; the rest are seconds.
_CAPS = ("max_seconds", "interval_s")
# The rail-guard thresholds a caller may pass through. They may only TIGHTEN
# (`RailLimits.tighten` enforces that); a key that is absent keeps the armed value.
RAIL_KEYS = ("v_hi", "v_lo", "frame_frac", "line_frac", "consec_lines", "poll_hz")


def _flag(params: dict, key: str, default: bool) -> bool:
    """A boolean that must ARRIVE as a boolean.

    `bool("false")`, `bool("no")` and `bool("0")` are all True, and these flags decide
    whether an unattended loop runs and whether it retracts when it stops -- the same
    reasoning `acquireControl` applies to `withdraw_on_deadman`. A missing key keeps
    the default; a present non-bool is a `-32602`, not a guess.
    """
    if key not in params or params[key] is None:
        return default
    value = params[key]
    if not isinstance(value, bool):
        msg = (f"setContinuous: {key} must be true or false, got "
               f"{type(value).__name__} -- it governs an unattended run, so it is "
               f"never inferred from a string")
        raise CommandError(INVALID_PARAMS, msg, {"param": key},
                           http={"error": msg, "reason": "bad_params"})
    return value


def _cap(params: dict, key: str):
    """One optional numeric bound: None when absent or empty, else a float."""
    value = params.get(key)
    if value in (None, ""):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise _bad(key, value)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise _bad(key, value) from None


def _bad(key: str, value) -> CommandError:
    msg = (f"setContinuous: {key} must be a number, got "
           f"{type(value).__name__} {value!r}")
    return CommandError(INVALID_PARAMS, msg, {"param": key},
                        http={"error": msg, "reason": "bad_params"})


async def set_continuous(hub, params: dict | None = None) -> dict:
    """Enable or disable the auto-repeat scan loop. Params:
    `{enabled, config?, max_frames?, max_seconds?, interval_s?, withdraw_on_stop?,
    rail?}`.

    Enabling stores the config + bounds and starts the supervisor; enabling with no
    config falls back to the last scan's config, and refuses when nothing has been
    scanned yet. Disabling stops the supervisor. Idempotent: enabling while already
    running restarts with the new config/bounds.
    """
    from ..host import continuous as cont

    body = dict(params or {})
    enabled = _flag(body, "enabled", True)

    if not enabled:
        await cont._stop_supervisor("disabled")
        await hub.log("Continuous imaging: off")
        await cont._broadcast(stopped_reason="disabled")
        return cont._state()

    config = body.get("config") or hub.host.last_scan_config
    if not config:
        msg = ("no scan config -- start a scan (or send `config`) first, "
               "then enable continuous imaging")
        raise CommandError(INVALID_PARAMS, msg, {"param": "config"},
                           http={"error": msg})

    # EVERY REMAINING PARAMETER IS VALIDATED HERE, BEFORE THE FIRST MUTATION. The
    # arming below unwinds a running supervisor, clears the stop event and drops the
    # safe_park "already parked" latch; a `-32602` raised after any of that would have
    # stopped the operator's loop and re-armed the park on the way to REFUSING the
    # call. A refusal must leave the instrument exactly as it found it.
    withdraw_on_stop = _flag(body, "withdraw_on_stop", False)
    max_frames = body.get("max_frames")
    if max_frames in (None, ""):
        max_frames = None
    else:
        try:
            max_frames = int(max_frames)
        except (TypeError, ValueError):
            raise _bad("max_frames", max_frames) from None
    caps = {key: _cap(body, key) for key in _CAPS}

    # a restart from a still-running loop: unwind the old supervisor first.
    if hub.host._scan_repeat_task is not None and not hub.host._scan_repeat_task.done():
        await cont._stop_supervisor("disabled")

    hub.host._scan_repeat_stop.clear()
    # New run = a fresh fail-safe episode: clear the safe_park "already parked"
    # latch (so this run's first termination/monitor parks) and the SUM-loss streak.
    hub.host._safe_park["active"] = False
    hub.host.scan_repeat.update({
        "enabled": True, "config": dict(config), "count": 0,
        "_sum_low_streak": 0,
        # the scan setpoint the loop starts with = the base force above free-air
        # for relative re-referencing (kept separate so the reref can overwrite
        # config["setpoint"] with the effective value without losing the base).
        "base_setpoint": float(dict(config).get("setpoint", 0.2)),
        "started_at": time.time(), "errors": 0,
        "max_frames": max_frames,
        "max_seconds": caps["max_seconds"],
        "interval_s": float(caps["interval_s"] or 0.0),
        "withdraw_on_stop": withdraw_on_stop,
        "stopped_reason": None})
    # Arm the rail-dwell safety guard for this run (safety kernel entry #1). It
    # watches the HV Z-drive per line and trips on a rail dwell -> stop-with-
    # withdraw; it is inert (disarmed) whenever continuous is off. An optional
    # `rail` override may TIGHTEN the thresholds (only-tighten, plan Sec 2.4).
    guard = getattr(hub.host, "rail_guard", None)
    if guard is not None:
        limits = guard.limits
        rail = body.get("rail")
        if isinstance(rail, dict):
            limits = limits.tighten(**{k: rail.get(k) for k in RAIL_KEYS
                                       if k in rail})
        # Arm on BOTH backends (group-2 redo). The old shared-link race is gone: the
        # guard is no longer fed by a host ~10 Hz Z-drive read (which shared the scan
        # pass's bridge link). On HARDWARE it is fed from the always-on TELEM poll
        # (Igor's autonomous scalar poll -> _hw_telemetry_frame), so nothing on the
        # host reads the link; on the TWIN a small async poll reads the simulated
        # Z-drive. Reset the consumer's per-run feed bookkeeping so a stale row/trip
        # from a previous run cannot leak in.
        hub.host._rail_feeding = False
        hub.host._rail_last_row = None
        hub.host._rail_pending_trip = None
        guard.arm(limits)
        await cont._broadcast_rail(guard)
    # S3: pre-scan SUM baseline + per-run interlock state reset (safety-critical; the
    # thresholds + rationale live at the TELEM-consumer section). ONE SUM read before
    # the frame loop is the relative-collapse reference; a NaN/None read leaves the
    # baseline None, which DISABLES the relative arm (the absolute 0.5 V floor still
    # trips). Reset the streaks + any stale pending interlock trip so a previous run
    # cannot leak in. Best-effort: a failed baseline read must not 500 the endpoint.
    hub.host._sum_low_streak = 0
    hub.host._defl_high_streak = 0
    hub.host._interlock_pending_trip = None
    try:
        _base = await cont._read_sum()
    except Exception:                                    # noqa: BLE001
        _base = math.nan
    hub.host._sum_baseline = None if cont._bad(_base) else float(_base)
    # Fourth-slice runtime interlocks (12.1): armed ONCE for the whole continuous run
    # (not per frame -- a per-frame re-arm would reset the continuous-time clocks the
    # twin scan renders past faster than real time). Fed by the TELEM consumer on
    # hardware / the twin feed task; the supervisor disarms it on exit.
    from ..host.engage import arm_interlocks as _arm_interlocks
    _arm_interlocks(hub)
    hub.host._scan_repeat_task = spawn_untraced(cont._supervise())
    await hub.log(
        f"Continuous imaging: ON (max_frames={hub.host.scan_repeat['max_frames']}, "
        f"max_seconds={hub.host.scan_repeat['max_seconds']}, "
        f"interval={hub.host.scan_repeat['interval_s']:g}s). Drift CORRECTION still "
        f"needs the Connect 'Track and compensate drift' toggle.")
    await cont._broadcast()
    return cont._state()


async def set_continuous_wire(hub, params: dict | None = None) -> dict:
    """The WIRE door. Same verb, plus the two refusals a script needs stated as
    codes rather than inferred from a state dict: a `rail` override that is not an
    object, and the backend gate.

    The loop itself is a START verb whose state stays `scanning` for hours (plan Sec
    F.1), so the reply is deliberately small -- `{ok, enabled, state}` plus the bounds
    that were actually applied, so a script can log what it armed without parsing the
    cockpit's full state dict.
    """
    body = dict(params or {})
    rail = body.get("rail")
    if rail is not None and not isinstance(rail, dict):
        msg = (f"setContinuous: `rail` must be an object of "
               f"{{{', '.join(RAIL_KEYS)}}}, got {type(rail).__name__}")
        raise CommandError(INVALID_PARAMS, msg, {"param": "rail"})
    enabled = _flag(body, "enabled", True)
    if enabled:
        # The silent-twin guard, before anything is armed: an unattended loop is the
        # LAST thing that should run against simulated physics on a station that
        # believes it is driving hardware (plan Sec C.2).
        gate = hub.require_live_backend()
        if gate:
            await hub.log("Continuous imaging REFUSED: " + gate)
            raise CommandError(BACKEND_MISMATCH, gate,
                               {"backend": hub.backend_label(),
                                "required": getattr(hub.host, "backend", None)})
        # A scan that is ALREADY running is not a loop this call can arm cleanly:
        # the supervisor arms its own frames, so let the caller stop or wait.
        run = hub.runs.in_flight()
        if run is not None and run.kind == "scan":
            raise CommandError(BUSY, "a scan is already running",
                               {"run_id": run.run_id, "kind": run.kind})
    state = await set_continuous(hub, body)
    return {"ok": True, "enabled": bool(state.get("enabled")),
            "state": "scanning" if state.get("enabled") else "idle",
            "max_frames": state.get("max_frames"),
            "max_seconds": state.get("max_seconds"),
            "interval_s": state.get("interval_s"),
            "withdraw_on_stop": bool(hub.host.scan_repeat.get("withdraw_on_stop")),
            "rail": state.get("rail")}
