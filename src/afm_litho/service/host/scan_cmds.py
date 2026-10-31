"""The scan-domain service layer: mode, approach, withdraw, Z gain, raster start.

Truth document: docs/FLEX_INTEGRATION_PLAN.md Sec A (the instrument-operations
table), C (this layer -- "one coroutine per FLEX verb, each raising a typed
CommandError and returning plain dicts; each router body becomes a thin adapter"),
E.2 (where the single-writer claim is taken), F.1/F.3 (start+poll, the state
machine). Wire reference: docs/FLEX_PROVIDER.md.

WHAT MOVED, AND WHAT DELIBERATELY DID NOT
-----------------------------------------
The BODIES of `POST /api/mode`, `/api/approach`, `/api/withdraw`, `/api/zgain`,
`/api/scan/start` and `/api/scan/abort` live here now; `service/routers/scan.py` keeps the decorated
endpoints as adapters that call these and render a `CommandError` back into the
HTTP-200 `{"error": ...}` convention the cockpit has always seen. There is one
implementation of every guard, so a FLEX script and the panel cannot drift apart --
which is the entire point of the extraction (plan Sec D: "the behaviour the GUI would
gain from FLEX is delivered by the service layer, not by the transport").

What did NOT move: the two scan PUMPS (`_scan_pump`, `_hw_scan_pump`) and the
persistence they do. They are streaming machinery, not the start path, and they close
over the process hub through `host/scan_pump.py`. The start verbs import them lazily.

THE HUB IS A PARAMETER, AND WHAT THAT DOES NOT PROMISE
------------------------------------------------------
Every function here takes its hub explicitly, because the dead-man already learned
what an endpoint closing over the process-global hub costs (`withdraw`'s docstring).
The pumps, the continuous supervisor and the telemetry consumers still read the
process hub, so passing a DIFFERENT hub here is honest only for the paths that never
reach them (mode, withdraw, Z gain). Nothing in the service does that today; a test
that wants to must not start a pump. `stop_scan` signals a running pump, so every caller
passes it the process hub.
"""

from __future__ import annotations

import asyncio
import math
import time

from ... import clock
from ...hardware import bridge_protocol as P
from ...hardware import channels
from . import safe_park
from ..commands.errors import (
    BACKEND_MISMATCH,
    BUSY,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    REFUSED,
    CommandError,
)

# The margin added to every published estimate. An `est_s` a script SLEEPS on
# (the FLEX driver's blocking helpers do exactly that, plan Sec G) must not be
# systematically short, or every convenience helper starts its poll loop early and
# reports a timeout that never happened.
EST_MARGIN_S = 5.0
APPROACH_MARGIN_S = 2.0

# How long `setMode` waits after sending SETMODE before it returns, and why it waits
# at all: `HardwareARC.set_imaging_mode` is FIRE-AND-FORGET (`link.send(cmd_setmode)`,
# no UID, no readback), so without a settle the verb returns while the vendor's
# MainPopupFunc is still rewriting the crosspoint, the feedback channel, the gain sign
# and the DDS drive. A script that sends `setMode` and then `startApproach` back to
# back would engage INTO that reconfiguration -- the exact mid-flight channel/gain-sign
# state the engaged gate exists to prevent, arrived at from the other side.
#
# 1.0 s is a PLACEHOLDER, not a measurement: the correct value is whatever the vendor
# call actually takes on the rig, and docs/FLEX_RIG_RUNBOOK.md "07" (c) is where it
# gets measured. `AFM_LITHO_MODE_SETTLE_S` may only LENGTHEN it (a station that
# measures 2.5 s can say so; nothing can shorten it below the default), and it is
# capped so the published `setMode` bound stays honest and under the 10 s ceiling
# every command-socket verb keeps.
MODE_SETTLE_DEFAULT_S = 1.0
MODE_SETTLE_MAX_S = 5.0


def _real_bridge(hub):
    """The REAL hardware ARC bridge attached to `hub`, or None on the twin.
    Local + lazy (imports only SoftwareARC, never the hub singleton) so
    importing this module stays cheap -- see hub.real_bridge for the rule."""
    from ...twin.arc import SoftwareARC
    arc = getattr(hub.host, "hardware_arc", None)
    if arc is None or isinstance(arc, SoftwareARC):
        return None
    return arc


def mode_settle_s() -> float:
    """The post-SETMODE settle, in seconds. Only-lengthen; never disabled."""
    import os
    raw = os.environ.get("AFM_LITHO_MODE_SETTLE_S")
    if raw is None:
        return MODE_SETTLE_DEFAULT_S
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return MODE_SETTLE_DEFAULT_S
    if val <= 0:
        return MODE_SETTLE_DEFAULT_S
    return max(MODE_SETTLE_DEFAULT_S, min(MODE_SETTLE_MAX_S, val))


# =============================================================== validation ======

def _number(body: dict, key: str, default, *, verb: str):
    """One numeric parameter, or a `-32602` that NAMES it.

    Numeric strings are accepted (`"0.5"`), because JSON from a hand-written script
    and from a form field both arrive that way and the endpoints have always coerced
    them. A BOOL is refused: `float(True)` is 1.0, and `setpoint: true` is a typo,
    not a volt.
    """
    if not isinstance(body, dict) or body.get(key) is None:
        return default
    value = body[key]
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise _bad_param(verb, key, value)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise _bad_param(verb, key, value) from None


def _bad_param(verb: str, key: str, value) -> CommandError:
    msg = (f"{verb}: {key} must be a number, got "
           f"{type(value).__name__} {value!r}")
    return CommandError(INVALID_PARAMS, msg, {"param": key},
                        http={"error": msg, "reason": "bad_params"})


def _finite_number(body: dict, key: str, default, *, verb: str):
    """Like `_number`, but also refuses non-finite values (NaN/Inf). A scan-loop
    gain that reaches `ir_SetPISLoop` NaN or Inf is a silent instrument fault, not
    a refusal a script would ever see coming -- so this is the check `startScan`
    runs on `pgain`/`igain` before either reaches the Z arm."""
    value = _number(body, key, default, verb=verb)
    if value is not None and not math.isfinite(value):
        raise _bad_param(verb, key, body.get(key))
    return value


def _live_or_refuse(hub, what: str) -> CommandError | None:
    """The SILENT-TWIN GUARD (plan Sec C.2), in one place.

    A station CONFIGURED as hardware whose bridge is not live must not quietly
    execute against simulated physics and report ordinary success -- an overnight
    script would "image" a whole chip against a cartoon. Never refuses a genuinely
    twin-configured station. Returns the refusal rather than raising it, so the
    caller can log its own wording first (the log lines are asserted by tests).
    """
    gate = hub.require_live_backend()
    if not gate:
        return None
    return CommandError(BACKEND_MISMATCH, gate,
                        {"backend": hub.backend_label(), "verb": what,
                         "required": getattr(hub.host, "backend", None)},
                        http={"error": gate, "reason": "backend_not_live",
                              "backend": hub.backend_label()})


def _claim(hub, kind: str):
    """The single-writer claim (plan Sec E.2), taken SYNCHRONOUSLY -- no `await`
    between the busy check and the set. Raises the wire's `-32020 busy` carrying the
    in-flight `run_id`, and the HTTP body the routers have always returned."""
    from ..control import RunBusy
    try:
        return hub.claim_run(kind)
    except RunBusy as busy:
        raise CommandError(BUSY, str(busy),
                           {"run_id": busy.run_id or hub._in_flight_run_id(),
                            "kind": busy.kind},
                           http=hub.busy_body(busy)) from busy


# ============================================================== mode helpers =====

def norm_mode(mode: str) -> str:
    """The INSTRUMENT imaging mode a scan/approach mode maps to: "ac" for AC/tapping,
    "contact" for everything else. "current" (c-AFM) is a contact-family variant -- it
    servos on Deflection with the DC crosspoint, so it runs under the vendor Contact
    imaging mode, not AC."""
    return "ac" if str(mode).strip().lower() == "ac" else "contact"


def tracked_mode(raw) -> str:
    """The hub's imaging mode as the mode gate reads it: lowercase text, "" when unknown."""
    return str(raw or "").lower()


def mode_differs(current: str, want: str) -> bool:
    """True when the tracked software mode is not the one the scan needs."""
    return current != want


def engaged_or_scanning(hub) -> tuple[bool, str | None]:
    """The authoritative "the tip is on the surface (or about to be)" signal the mode
    switch gates on. True when ANY of:
      * hub.engaged -- set by approach, cleared by withdraw / safe_park (the primary,
        authoritative flag);
      * a raster is running (the scan pump task, on either backend);
      * continuous imaging is enabled (an unattended run keeps re-engaging per frame).
    Returns (engaged, why) so the caller can say WHY it refused. Conservative by
    design: a false positive only refuses a SAFE switch (annoying, recoverable by a
    withdraw); a false negative would allow a mid-engage feedback-channel + gain-sign
    flip -- a tip crash -- so every path that could leave the tip engaged must read
    True here."""
    from ..twin_plugin import twin_only
    if twin_only(hub):
        # A twin write sets the engaged flag as a rig write does, but the twin has always
        # named the write itself in the refusal.
        lt = getattr(hub.host, "litho_thread", None)
        if lt is not None and lt.is_alive():
            return True, "writing"
    if getattr(hub.host, "engaged", False):
        return True, "engaged"
    task = getattr(hub.host, "_scan_task", None)
    if task is not None and not task.done():
        return True, "scanning"
    if hub.host.scan_repeat.get("enabled"):
        return True, "scanning"
    # A HARDWARE litho write self-engages the Z loop (LithographyRun.write_object arms
    # loop 2 -- there is no prior /api/approach), so the tip is on the surface for the
    # whole write. hub.engaged is also set True by the worker, but check the running
    # worker thread too: it closes the gap between the endpoint spawning the thread and
    # the worker's first line, and it is a second, independent witness of the write.
    lt = getattr(hub.host, "litho_thread", None)
    if lt is not None and lt.is_alive():
        return True, "writing"
    return False, None


async def ensure_software_mode(hub, mode: str) -> bool:
    """Defense-in-depth (the core requirement): keep the ACTUAL Asylum software imaging
    mode consistent with the mode a HARDWARE approach/scan is about to run in, so the
    browser mode and the software mode can never disagree at engage/scan time (an AC scan
    with the software in Contact is WRONG -- crosspoint routed to DC/Defl, no drive).

    On hardware, comparing the requested instrument mode (norm_mode) with the tracked
    hub.imaging_mode:
      * already consistent -> nothing;
      * a mismatch, and the tip is NOT engaged/scanning -> send SETMODE first (safe
        while withdrawn -- approach engages AFTER this), settle, clear hub.free_air,
        then track it;
      * an UNKNOWN (None) tracked mode, not engaged/scanning -> read the live mode
        (check_mode): a match is recorded with no switch; a mismatch or an unreadable
        mode switches as above ONLY if fresh telemetry shows the Z drive retracted,
        else refuses `mode_unknown_tip_engaged` (after a restart hub.engaged is False
        even with the tip down; a stale AC read with the drive off would latch a ~0 V
        amplitude baseline, a SETMODE on an engaged tip can crash it);
      * unknown WHILE engaged/scanning -> record the requested mode, no switch;
      * a real mismatch WHILE engaged/scanning -> do NOT switch (a mid-engage channel/
        gain-sign flip is a crash); log a loud WARNING so the operator/auditor sees it.
    Returns True when it sent SETMODE. Twin: no-op (there is no software to
    reconcile)."""
    hw = _real_bridge(hub)
    if hw is None:
        return False
    want = norm_mode(mode)
    cur = tracked_mode(getattr(hub.host, "imaging_mode", None))
    if not mode_differs(cur, want):
        return False
    engaged, why = engaged_or_scanning(hub)
    if engaged and not cur:
        hub.host.imaging_mode = want
        return False
    if not cur:
        # Unknown tracked mode (service start): hub.engaged is in-memory only, so the
        # tip may be down. Ask the instrument, and never switch a tip that looks down.
        live = await asyncio.to_thread(_live_mode, hw)
        if live == want:
            hub.host.imaging_mode = want
            await hub.log(f"Imaging mode read back from the instrument: '{want}'.")
            return False
        if live == "unknown":
            await hub.log("Imaging mode unknown on the hub and the instrument crosspoint "
                          "read-back is unavailable; switching only if the Z drive shows "
                          "the tip retracted.")
        down, zdrive = _tip_appears_down(hub)
        if down:
            where = (f"Z drive {zdrive:g} V" if zdrive is not None
                     else "no fresh Z-drive telemetry")
            state = (f"is in '{live}'" if live != "unknown"
                     else "mode could not be read back")
            msg = (f"instrument software {state} but the tip appears engaged "
                   f"({where}); withdraw before switching to '{want}'")
            await hub.log("Mode switch REFUSED: " + msg)
            from ..command_guard import GuardRefused
            raise GuardRefused("mode_unknown_tip_engaged", msg,
                               {"live_mode": live, "want": want, "zdrive_v": zdrive})
    if engaged:
        await hub.log(
            f"WARNING: imaging-mode MISMATCH -- the instrument software is '{cur}' but "
            f"this {('scan' if why == 'scanning' else 'approach')} is '{want}'. NOT "
            f"switching while the tip is {why} (a mid-engage channel/gain-sign flip is a "
            f"crash). Withdraw, then change the mode, then retry.")
        return False
    await asyncio.to_thread(hw.set_imaging_mode, want)
    # Same settle and baseline invalidation as set_mode: SETMODE is fire-and-forget,
    # and a free-air baseline belongs to the mode it was read in.
    await clock.async_sleep(mode_settle_s())
    hub.host.imaging_mode = want
    hub.host.free_air = None
    await hub.log(f"Imaging mode auto-switched '{cur or 'unknown'}' -> '{want}' "
                  f"(hardware) to match this pass -- the tip was withdrawn, so the "
                  f"switch is safe.")
    return True


def _live_mode(hw) -> str:
    """The instrument's imaging mode from the crosspoint read-back ("contact" | "ac" |
    "unknown"); "unknown" on any failure or a backend without the read."""
    try:
        return str(hw.check_mode())
    except Exception:                                     # noqa: BLE001
        return "unknown"


# A TELEM frame older than this says nothing about where the tip is now.
_TELEM_FRESH_S = 10.0


def _tip_appears_down(hub) -> tuple[bool, float | None]:
    """(down, zdrive_v) from the last hardware telemetry frame: down when the Z drive
    is above `z_park_v + z_park_tol_v`. No fresh frame counts as down (unknown tip
    state never licenses a mode switch)."""
    frame = getattr(hub.host, "_hw_telemetry", None)
    if not isinstance(frame, dict):
        return True, None
    t, zdrive = frame.get("t"), frame.get("zdrive_v")
    if t is None or zdrive is None or (time.time() - float(t)) > _TELEM_FRESH_S:
        return True, None
    model = hub.host.instrument_model
    ceiling = float(model.z_park_v.value) + float(model.z_park_tol_v.value)
    return float(zdrive) > ceiling, float(zdrive)


# ==================================================================== getMode ====

def get_mode(hub) -> dict:
    """The ACTUAL instrument imaging mode the cockpit dropdown syncs to on load, plus the
    live engaged/scanning gate state (so the panel -- and a script deciding whether to
    call `setMode` at all -- can see whether a switch is allowed right now). `mode` is
    None until first established (a `setMode` / `POST /api/mode` switch, or a hardware
    approach).

    A pure in-memory read with no bridge I/O, which is why the wire serves it from the
    READ socket as well as the command socket."""
    engaged, why = engaged_or_scanning(hub)
    return {"mode": getattr(hub.host, "imaging_mode", None), "engaged": engaged,
            "gate": why, "backend": hub.backend_live()}


# ==================================================================== setMode ====

async def _establish_free_air_after_mode(hub, mode: str) -> None:
    """13.1 items 27/33: a mode switch (only allowed withdrawn) re-establishes free
    air in the NEW mode after the vendor profile has settled, rather than leaving
    `hub.free_air` None; every mode, because the guard demands a baseline tagged
    with the engage mode. Never raises -- a free-air hiccup must not fail the mode
    switch; the write/approach doors re-read anyway."""
    from ..host.engage import establish_free_air_async
    try:
        fa = await establish_free_air_async(hub, mode, force=True)
    except Exception as exc:                              # noqa: BLE001
        await hub.log(f"mode: free-air re-read after switch to {mode} failed: {exc}")
        return
    if fa is not None:
        await hub.log(f"mode: free air re-read after switch to {mode}: "
                      f"{fa.value_v:.4f} V")


async def set_mode(hub, params: dict | None = None) -> dict:
    """Switch the instrument's IMAGING MODE (contact <-> AC).

    SAFETY-CRITICAL (option a, the decision): the vendor switch does NOT withdraw --
    it stops the scan but leaves the tip on the surface, then changes the Z feedback
    InputChannel + gain sign (a crash if the tip is engaged). So this REFUSES the switch
    while the tip is ENGAGED or a scan is running (`engaged_or_scanning`) and tells the
    operator to withdraw first -- it never auto-withdraws.

    Not engaged + hardware backend: send `SETMODE contact|ac` over the bridge (the
    vendor's own MainPopupFunc mode-profile call; see hardware/bridge_protocol.cmd_setmode
    + afm_litho_run.ipf AFM_SetMode). Either backend records the commanded mode on
    `hub.imaging_mode`, so the mode-aware engage/scan and the litho-AC guard reflect the
    real instrument mode; the twin only tracks it (no real switch)."""
    body = params or {}
    mode = str(body.get("mode", "")).strip().lower()
    if mode not in ("contact", "ac"):
        msg = "mode must be 'contact' or 'ac'"
        raise CommandError(INVALID_PARAMS, msg, {"param": "mode"},
                           http={"error": msg, "reason": "bad_mode"})
    gate = _live_or_refuse(hub, "setMode")
    if gate is not None:
        await hub.log("Mode switch REFUSED: " + gate.message)
        raise gate
    engaged, why = engaged_or_scanning(hub)
    if engaged:
        state = {"engaged": "engaged", "scanning": "scanning",
                 "writing": "writing lithography"}.get(why, "on the surface")
        msg = ("Mode switch REFUSED: the tip is " + state
               + ". Switching imaging mode changes the Z feedback channel + gain sign "
                 "and does NOT withdraw -- withdraw first, then change the mode.")
        await hub.log(msg)
        raise CommandError(REFUSED, msg, {"reason": "engaged", "gate": why},
                           http={"error": msg, "reason": "engaged"})
    hw = _real_bridge(hub)
    if hw is not None:
        try:
            await asyncio.to_thread(hw.set_imaging_mode, mode)
        except Exception as exc:
            await hub.log(f"Mode switch send failed: {exc}")
            raise CommandError(INTERNAL_ERROR, str(exc), {"mode": mode},
                               http={"ok": False, "mode": mode,
                                     "error": str(exc)}) from exc
        # SETTLE BEFORE RETURNING (see MODE_SETTLE_DEFAULT_S). The send is
        # fire-and-forget, so this wait is the only thing standing between a scripted
        # `setMode` and a `startApproach` that engages mid-reconfiguration.
        await clock.async_sleep(mode_settle_s())
        hub.host.imaging_mode = mode
        # A free-air baseline is mode-specific (docs/INSTRUMENT_MODEL_SPEC.md
        # section 10: `FreeAir.mode`) -- a switch invalidates whatever was on
        # record.
        hub.host.free_air = None
        await hub.log(f"Imaging mode -> {mode} (hardware): instrument reconfigured "
                      f"via the vendor mode-profile call (feedback channel, gain sign, "
                      f"crosspoint, DDS drive); settled {mode_settle_s():g}s.")
        await _establish_free_air_after_mode(hub, mode)
        # literal, not backend_live(): this branch has ALREADY driven the
        # instrument, so reporting the derived value could only lie the
        # other way (say "twin" about a switch the hardware just made).
        return {"ok": True, "mode": mode, "backend": "hardware"}
    hub.host.imaging_mode = mode
    hub.host.free_air = None
    await hub.log(f"Imaging mode -> {mode} (twin): tracked on the hub (no real switch).")
    await _establish_free_air_after_mode(hub, mode)
    return {"ok": True, "mode": mode, "backend": hub.backend_live()}


# =================================================================== approach ====

def _approach_body(hub, params: dict | None) -> dict:
    """Validate and normalise the approach parameters ONCE, for both doors.

    This is the function that makes `POST /api/approach` and `startApproach` answer the
    same refusal for the same bad input -- a property a test asserts from both sides,
    because "one code path" is a claim that decays silently otherwise.
    """
    body = dict(params or {})
    for key in ("setpoint", "pgain", "igain", "settle_s"):
        _number(body, key, None, verb="approach")      # raises -32602, or is ignored
    return body


def _engage_on_instrument(hub, *, setpoint: float, pgain: float, igain: float,
                          settle_s: float, mode: str) -> dict:
    """The hardware engage's Igor commands, through the Asylum plugin. Everything around
    them (guards, mode, `hub.engaged`, interlocks) stays in `_approach_impl`; a failed
    engage raises the plugin's exception, as the direct call did. Call from a worker thread."""
    from ...instrument import EngageAt
    from ...instrument.asylum import AsylumInstrument
    from ...run.model import Setpoint
    asylum = AsylumInstrument(hub, engage_pgain=pgain, engage_igain=igain,
                              engage_settle_s=settle_s, service_owns_approach=True)
    asylum.engage_mode = mode       # raw: "current" is not a `Mode` but runs the contact loop
    result = asylum.engage(EngageAt(Setpoint("absolute", setpoint),
                                    "ac" if mode == "ac" else "contact"))
    if asylum.failure is not None:
        raise asylum.failure
    if not result.done:
        raise RuntimeError(result.reason)
    return asylum.snapshot


def _withdraw_on_instrument(hub) -> dict:
    """The hardware withdraw's Igor commands, through the Asylum plugin. The stop signal,
    `hub.engaged` and the interlocks stay in `withdraw`. Call from a worker thread."""
    from ...instrument import WithdrawTip
    from ...instrument.asylum import AsylumInstrument
    asylum = AsylumInstrument(hub, service_owns_approach=True)
    result = asylum.withdraw(WithdrawTip())
    if asylum.failure is not None:
        raise asylum.failure
    if not result.done:
        raise RuntimeError(result.reason)
    return asylum.snapshot


async def approach(hub, params: dict | None = None) -> dict:
    """Fine engage: close the deflection->Z feedback loop so the tip tracks the
    surface. BLOCKING -- it returns the engage snapshot (the HTTP door's shape;
    `start_approach` wraps the same work in a run record).

    On the hardware backend this drives the REAL Z loop (see hw_approach /
    docs/engage_withdraw.md); on the twin it runs the simulated approach. The coarse
    (motor) approach must already have the tip within Z-piezo range.

    The IN-FLIGHT MARKER is published here: an approach is the one tip-moving
    operation with no task, thread or flag of its own, so a status reader had no way
    to say `approaching` (plan Sec F.3). A counter, not a bool, because the continuous
    loop's between-frame re-reference calls this too and a nested/overlapping call must
    not clear the marker early.
    """
    body = _approach_body(hub, params)
    # The single-writer claim (plan Sec E.2), taken BEFORE the first await, exactly
    # as on the write path. An approach drives the tip toward the surface, so it must
    # not interleave with a scan or a write that is already starting. Held for the
    # whole call and released here: unlike scan and write there is no worker to hand
    # it to -- the approach IS this coroutine.
    claim = _claim(hub, "approach")
    hub.host._approach_in_flight = getattr(hub.host, "_approach_in_flight", 0) + 1
    try:
        return await _approach_impl(hub, body)
    finally:
        hub.host._approach_in_flight = max(0, getattr(hub.host, "_approach_in_flight", 1) - 1)
        hub.release_claim(claim)


def _append_to_refusal(exc: CommandError, note: str) -> None:
    """Append `note` to a refusal's message everywhere it is rendered (JSON-RPC
    message, exception args, the HTTP body's `error`)."""
    exc.message = exc.message + note
    exc.args = (exc.message,) + tuple(exc.args[1:])
    if exc.http is not None and isinstance(exc.http.get("error"), str):
        exc.http["error"] = exc.http["error"] + note


async def _approach_impl(hub, body: dict) -> dict:
    """The engage itself, with the claim and the in-flight marker already taken."""
    gate = _live_or_refuse(hub, "startApproach")
    if gate is not None:
        await hub.log("Approach REFUSED: " + gate.message)
        raise gate
    hw = _real_bridge(hub)
    await hub.refresh_xy_cal("approach")
    # ---- INSTRUMENT-MODEL COMMAND GUARD (docs/INSTRUMENT_MODEL_SPEC.md secs 2/10/13)
    # BEFORE the engage, on BOTH doors and BOTH backends: a latched fault, a stale or
    # missing free-air baseline, or a setpoint outside its band refuses the approach
    # here -- before the tip moves. The mode/setpoint are resolved exactly as the
    # backend branches below will resolve them (contact vs AC picks the band).
    from ..command_guard import check_approach
    from ..host.telemetry import current_scan_is_ac
    _guard_mode = str(body.get("mode")
                      or ("ac" if current_scan_is_ac() else "contact")).lower()
    _guard_default_sp = hub.scan_params.get("setpoint", 1.0 if hw is not None else 0.2)
    _guard_setpoint = float(body.get("setpoint", _guard_default_sp))
    # Establish a FRESH free-air baseline (3-read agreement + one dwell-retry)
    # BEFORE the guard, so a plain approach both records hub.free_air (13.1 item 2)
    # and gives check_approach a baseline to validate, tagged with _guard_mode. A
    # no-op when a fresh withdrawn baseline in that mode is already on record (the
    # continuous re-reference's own). The tip is withdrawn at approach start, so
    # this read is a valid withdrawn read.
    from ..host.engage import (
        arm_interlocks,
        check_engage_sum_ceiling,
        establish_free_air_async,
    )
    switched = False
    if hw is not None:
        # The instrument must be IN the engage mode before the free-air read: in AC
        # the baseline is the free amplitude, which reads ~0 with the drive off.
        switched = await ensure_software_mode(hub, _guard_mode)

    def _refused(exc: CommandError) -> CommandError:
        # The software mode switch above stands even though the engage is refused.
        if switched:
            _append_to_refusal(exc, f" (the instrument software imaging mode was "
                                    f"switched to '{norm_mode(_guard_mode)}')")
        return exc

    try:
        await establish_free_air_async(hub, _guard_mode)
    except CommandError as exc:
        _refused(exc)
        await hub.log("Approach REFUSED: " + exc.message)
        raise
    # 0.9*SUM ceiling + SUM floor, with a FRESH DetectorSum read (12.1). The detector-
    # integrity refusal (sum_missing / setpoint_exceeds_sum) is applied BEFORE the
    # setpoint-band check, so a setpoint past 0.9*SUM is named as such rather than
    # merely reported out-of-band.
    try:
        check_engage_sum_ceiling(hub, _guard_setpoint)
    except CommandError as exc:
        _refused(exc)
        await hub.log("Approach REFUSED: " + exc.message)
        raise
    try:
        check_approach(hub.host.instrument_model, _guard_mode, _guard_setpoint,
                       getattr(hub.host, "free_air", None), now=time.time(),
                       fault=getattr(hub, "fault", None))
    except CommandError as exc:
        _refused(exc)
        await hub.log("Approach REFUSED: " + exc.message)
        raise
    if hw is not None:
        setpoint = float(body.get("setpoint", hub.scan_params.get("setpoint", 1.0)))
        pgain = float(body.get("pgain", 0.0))
        igain = float(body.get("igain", 10.0))
        settle_s = float(body.get("settle_s", 6.0))
        # SAFETY-CRITICAL mode select: AC servos on Amplitude with a NEGATED gain,
        # contact/current on Deflection with a positive gain (hw_approach._AMPL). An
        # explicit body `mode` wins; otherwise derive it from the current/last scan
        # config (current_scan_is_ac). The relative-setpoint re-reference (contact-
        # only) passes no mode, so it resolves to the contact config -> "contact".
        from ..host.telemetry import current_scan_is_ac
        mode = str(body.get("mode")
                   or ("ac" if current_scan_is_ac() else "contact")).lower()
        # The software imaging mode was reconciled to `mode` before the free-air read
        # (above); `mode` and `_guard_mode` resolve identically, so this is a no-op
        # unless something switched it since.
        await ensure_software_mode(hub, mode)
        in_chan = channels.AMPLITUDE if mode == "ac" else channels.DEFLECTION
        hub.host.z_loop = {"setpoint": setpoint, "pgain": pgain, "igain": igain,
                      "mode": mode}
        await hub.log(f"Approach (hardware): {mode} Z feedback {in_chan}->Height, "
                      f"setpoint {setpoint:g} V (positive = into surface), gains "
                      f"P={pgain:g} I={igain:g}"
                      + (" (AC: gain sign negated)" if mode == "ac" else "")
                      + ". Coarse approach must be in range.")
        snap = await asyncio.to_thread(_engage_on_instrument, hub, setpoint=setpoint,
                                       pgain=pgain, igain=igain, settle_s=settle_s,
                                       mode=mode)
        hub.host.engaged = True            # tip on the surface -> gate the mode switch (safety)
        hub.host._scan_epoch += 1          # new engagement owns the tip (race-guard, Sec 2)
        arm_interlocks(hub)           # runtime interlocks armed for the whole engage
        # A NEW ENGAGEMENT ENDS THE PARK EPISODE. `_safe_park["active"]` is an
        # "already parked this episode" latch, not "the tip is parked right now" --
        # but the FLEX state machine reads it (row 7 beats row 8), so leaving it set
        # after a manual re-approach reported `parked` while the tip was ON THE
        # SURFACE. That is the dangerous direction of wrong: a script would believe
        # the tip was safe. Clearing it here also re-arms the next termination's
        # park, which is the same thing continuous_set does when it arms a run.
        hub.host._safe_park["active"] = False
        ok = "reached setpoint" if snap.get("reached_setpoint") else "still settling"

        # defl_v / z_v are NaN-safe (None when a read failed while the bridge was
        # unresponsive -- see hw_approach._jsafe). Format the number only when
        # present, else show "-", so a None read never 500s the approach.
        def _fmt(v):
            return f"{v:.3f} V" if isinstance(v, (int, float)) else "-"
        await hub.log(f"Engage ({ok}): defl = {_fmt(snap.get('defl_v'))}, "
                      f"Z out = {_fmt(snap.get('z_v'))}")
        await hub.broadcast({"type": "engaged", **snap})
        return {"ok": True, **snap}
    # Honour the requested setpoint (the relative-setpoint re-reference arms the
    # twin approach with a freshly-computed effective setpoint); default 0.2 V
    # preserves the old behaviour for callers that pass nothing.
    setpoint = float(body.get("setpoint", hub.scan_params.get("setpoint", 0.2)))
    await hub.log(f"Approach: z servo armed at setpoint {setpoint:g} V, engaging…")
    snap = await asyncio.to_thread(hub.host.inst.approach, setpoint)
    hub.host.engaged = True                # tip on the surface -> gate the mode switch (safety)
    hub.host._scan_epoch += 1              # new engagement owns the tip (race-guard, Sec 2)
    arm_interlocks(hub)               # runtime interlocks armed for the whole engage
    # A NEW ENGAGEMENT ENDS THE PARK EPISODE (see the hardware branch above for why
    # this latch must be cleared here and not only at withdraw).
    hub.host._safe_park["active"] = False
    await hub.log(f"Engaged at z = {snap['z_um']:.3f} um, defl = {snap['defl_v']:.3f} V")
    await hub.broadcast({"type": "engaged", **snap})
    return {"ok": True, **snap}


def approach_est_s(body: dict) -> float:
    """How long the engage should take: the settle the caller asked for, plus a
    margin. Honest rather than optimistic -- a driver SLEEPS on this before it polls."""
    try:
        settle = float(body.get("settle_s", 6.0))
    except (TypeError, ValueError):
        settle = 6.0
    return round(max(0.0, settle) + APPROACH_MARGIN_S, 3)


async def start_approach(hub, params: dict | None = None) -> dict:
    """START+POLL (plan Sec A/F.1): claim, allocate a run, spawn the engage, return
    `{run_id, est_s}` in well under a second. `getState` says `approaching` from the
    moment this returns until the run record closes.

    Validation, the silent-twin guard and the claim all happen HERE, synchronously,
    so a refusal is a refusal of the CALL -- a script must never be handed a `run_id`
    for a run that was never going to start.
    """
    body = _approach_body(hub, params)
    gate = _live_or_refuse(hub, "startApproach")
    if gate is not None:
        await hub.log("Approach REFUSED: " + gate.message)
        raise gate
    # 0.9*SUM ceiling / SUM floor, SYNCHRONOUSLY (12.1 / A12): startApproach validates
    # HERE, before spawning the engage task, so a detector-integrity refusal is a
    # refusal of the CALL (an immediate JSON-RPC error), never a run_id a script must
    # poll to discover. (_approach_impl re-checks it on the engage itself.)
    from ..host.engage import check_engage_sum_ceiling as _check_engage_sum_ceiling
    _sp = float(body.get("setpoint", hub.scan_params.get("setpoint", 0.2)))
    try:
        _check_engage_sum_ceiling(hub, _sp)
    except CommandError as exc:
        await hub.log("Approach REFUSED: " + exc.message)
        raise
    claim = _claim(hub, "approach")
    hub.host._approach_in_flight = getattr(hub.host, "_approach_in_flight", 0) + 1
    try:
        run_id = hub.next_run_id("approach")
        task = P.spawn_untraced(_approach_run(hub, body, run_id, claim))
        claim.hand_over(run_id, task)
    except BaseException:
        # Nothing was spawned, so nothing will ever decrement or release: unwind
        # here, or the state machine sticks on `approaching` for the process's life.
        hub.host._approach_in_flight = max(0, getattr(hub.host, "_approach_in_flight", 1) - 1)
        hub.release_claim(claim)
        raise
    return {"run_id": run_id, "est_s": approach_est_s(body), "state": "approaching"}


async def _approach_run(hub, body: dict, run_id: str, claim) -> None:
    """The spawned engage. Closes the run record, and on FAILURE parks the tip.

    Plan Sec F.3 is explicit that `approaching --fail--> parked (safe_park fired)`,
    and CLAUDE.md's rule is broader: any failure during a hardware pass is a stop and
    a verified withdraw. A script polling `getState` sees `parked`, and the run record
    carries the reason. The HTTP door keeps its own older behaviour of surfacing the
    exception to the caller, who is a human watching the panel and can act.
    """
    outcome = "failed"
    try:
        snap = await _approach_impl(hub, body)
        outcome = "completed"
        hub.finish_run(run_id, completed=True,
                       detail="engaged" if getattr(hub.host, "engaged", False) else "settled",
                       stats={k: v for k, v in (snap or {}).items()
                              if v is None or isinstance(v, (int, float, str, bool))})
    except BaseException as exc:
        detail = f"{type(exc).__name__}: {exc}"
        hub.finish_run(run_id, completed=False, error=detail)
        try:
            await hub.log(f"Approach FAILED ({detail}) -- parking the tip")
            from ..host.safe_park import request_park
            await request_park(hub, f"approach failed: {detail}")
        except Exception:                                    # noqa: BLE001,S110
            pass          # the park's own failure is recorded by safe_park itself
        if isinstance(exc, asyncio.CancelledError):
            raise
    finally:
        hub.host._approach_in_flight = max(0, getattr(hub.host, "_approach_in_flight", 1) - 1)
        hub.host.pass_ended(run_id, outcome, None, [])


# =================================================================== withdraw ====

def _post_withdraw(hub) -> None:
    """Disarm the runtime interlocks -- the engage is over (12.1)."""
    from ..host.engage import disarm_interlocks
    disarm_interlocks(hub)


async def _establish_free_air_on_withdraw(hub) -> None:
    """A withdraw is a natural moment to record a fresh free-air baseline (13.1
    item 2): the tip is off the surface. Taken in the hub's CURRENT imaging mode
    (the guard refuses a baseline tagged with another mode); "contact" when no mode
    is known. In AC this reads Deflection AND the Amplitude channel. Best-effort and SKIPPED during a continuous run, whose own
    between-frame re-reference owns hub.free_air. Never raises -- a free-air hiccup
    must not break a withdraw."""
    import contextlib
    if hub.host.scan_repeat.get("enabled"):
        return
    from ..host.engage import establish_free_air_async
    mode = str(getattr(hub.host, "imaging_mode", None) or "").strip().lower()
    if not mode:
        mode = "contact"
        with contextlib.suppress(Exception):
            await hub.log("withdraw: imaging mode unknown, free-air baseline taken "
                          "as 'contact'")
    with contextlib.suppress(Exception):
        await establish_free_air_async(hub, mode, force=True)


async def withdraw(hub, params: dict | None = None) -> dict:
    """Retract, from ANY state. On the hardware backend: stop any running scan, stop
    the Z feedback loop, then re-arm loop 2 at -10 V with output minimum 0 so Height
    parks at 0 V (the retracted/safe position), on both platforms. On the twin: the simulated withdraw.

    Taking its hub EXPLICITLY is load-bearing: the dead-man and the engaged-idle cap
    (service/control.py) perform the SAME retract, and an endpoint that closes over
    the process-global hub silently ignores whichever hub the caller meant -- which,
    in a test, is a timer that reports a withdraw it never performed.
    """
    # STOP EVERYTHING FIRST, INCLUDING A WRITE (docs/STOP_RETRACT_SPEC.md: every Stop
    # withdraws, and there is no stop-and-hold path).
    #
    # This used to set only the SCAN aborts, so a withdraw during a lithography write
    # retracted the tip and left the write RUNNING -- and on hardware that is the
    # dangerous direction: `LithographyRun.write_object` SELF-ENGAGES, so the next
    # object in the pass drove the tip straight back down onto the surface the
    # operator had just withdrawn from. `signal_run_abort` is the ONE place that knows
    # every cooperative flag an in-flight pass watches (the scan event,
    # `_hw_litho_abort`, and the twin plugin's abort), so withdraw signals through
    # it rather than keeping a second, shorter list that drifts.
    #
    # NOT the full `_signal_stop`: that also disables `scan_repeat`, and the
    # continuous loop's own between-frame re-reference CALLS this withdraw -- so
    # using it here would make a continuous run stop itself after its first frame.
    from ..host.safe_park import (
        NOT_CONNECTED, ROUTE_COM_AFTER_RESTART, ROUTE_COM_ESCAPE, ROUTE_NOT_CONNECTED,
        STOP_RUN, _signal_stop, not_connected_result, safe_park_async, stop_route)
    _signal_stop(hub, STOP_RUN)
    route = stop_route(hub)
    if route == ROUTE_NOT_CONNECTED:
        # Before Igor dials in nothing is sent, so nothing is claimed: `engaged` stays as it
        # is and no `withdrawn` goes out (2026-10-28; the same test as park's).
        await hub.log(f"Withdraw: {NOT_CONNECTED}")
        return not_connected_result()
    if route == ROUTE_COM_ESCAPE:
        # The bridge socket closed after Igor dialled in. COM is the only way left to Igor,
        # and its one stop over COM is AFM_SafePark: the park, which reports what it could
        # confirm and claims nothing more (no `withdrawn` frame).
        await hub.log("Withdraw: the bridge socket is closed; sending Igor's abort "
                      "(AFM_SafePark) over COM instead")
        return await safe_park_async(hub, "withdraw (bridge closed)")
    if route == ROUTE_COM_AFTER_RESTART:
        # Restarted after a death whose predecessor had the bridge; the new bridge is not
        # up yet. COM is the only way to Igor: the park, as above.
        await hub.log("Withdraw: no bridge since the restart; sending Igor's abort "
                      "(AFM_SafePark) over COM instead")
        return await safe_park_async(hub, "withdraw (restarted, no bridge yet)")
    hw = _real_bridge(hub)
    if hw is not None:
        snap = await asyncio.to_thread(_withdraw_on_instrument, hub)
        hub.host.engaged = False           # tip off the surface -> a mode switch is now safe
        _post_withdraw(hub)
        await _establish_free_air_on_withdraw(hub)
        await hub.log("Withdrawn (hardware): Z loop re-armed to retract, Height driven to 0 V")
        await hub.broadcast({"type": "withdrawn", **snap})
        return snap
    snap = await asyncio.to_thread(hub.host.inst.withdraw)
    hub.host.engaged = False               # tip off the surface -> a mode switch is now safe
    _post_withdraw(hub)
    await _establish_free_air_on_withdraw(hub)
    await hub.log("Withdrawn: tip parked at -10 V")
    await hub.broadcast({"type": "withdrawn", **snap})
    return snap


# ==================================================================== stopScan ===

async def stop_scan(hub, params: dict | None = None) -> dict:
    """The operator Stop: stop the scan and RETRACT the tip. Every Stop withdraws, as in
    the Asylum software (click Stop -> it retracts); a stop-and-hold left the tip engaged
    with a stopped Z loop, which crashes the next scan (docs/STOP_RETRACT_SPEC.md). The
    `withdraw` request field is accepted and ignored: the behaviour is always a retract.

    On hardware a running pass owns the bridge link, so the real retract DEFERS to the
    scan pump's finally (`_deferred_withdraw_if_current`); with no pass running it
    retracts inline. The deferred retract is stamped with the current scan epoch and
    skipped if a newer engage/scan has taken over the tip (the stale-withdraw guard).

    No await precedes the stop flags."""
    # Continuous imaging: a Stop breaks the auto-repeat loop AT ONCE, even while a frame
    # is in flight, so the supervisor arms no further frame. (No-op when off.)
    if hub.host.scan_repeat.get("enabled"):
        hub.host.scan_repeat["enabled"] = False
        hub.host.scan_repeat["stopped_reason"] = "aborted"
        safe_park._signal_stop(hub, safe_park.STOP_REPEAT_LOOP)
    hw = _real_bridge(hub)
    scan_active = hub.host._scan_task is not None and not hub.host._scan_task.done()
    # Cooperative: the scan worker owns the link, so no concurrent td_Stop; it stops at
    # the next chunk boundary. The same signal stops a twin scan (the twin withdraws).
    safe_park._signal_stop(hub, safe_park.STOP_SCAN_PASS)
    # Counted after the signal and before the first await, so a `start_write` still
    # computing its estimate does not start a write behind this Stop
    # (`litho_cmds._stop_seq`). The twin branch retracts without `signal_run_abort`,
    # which counts the other stops.
    try:
        hub.host._scan_stop_seq = int(getattr(hub.host, "_scan_stop_seq", 0) or 0) + 1
    except Exception:                                     # noqa: BLE001,S110
        pass
    if hw is not None and scan_active:
        # The pass owns the bridge link: the pump withdraws once it releases it.
        safe_park.request_park(hub, "scan stopped", safe_park.PARK_DEFERRED)
        await hub.log("Stopping scan; tip will retract as soon as the pass halts")
    elif hw is not None:
        # No pass running: the verified retract now (Height->0 backstop); safe_park
        # clears engaged, broadcasts and logs itself, and never raises.
        await safe_park.request_park(hub, "scan stopped")
    else:
        await asyncio.to_thread(hub.host.inst.withdraw)      # twin
        hub.host.engaged = False            # tip off the surface -> a mode switch is now safe
        await hub.log("Scan stopped; tip withdrawn")
    await hub.broadcast({"type": "aborted", "withdraw": True})
    return {"ok": True, "withdraw": True}


# ==================================================================== setZGain ===

async def set_zgain(hub, params: dict | None = None) -> dict:
    """Adjust the Z feedback gains LIVE while engaged/imaging, without withdrawing.

    Re-arms the Z loop (Deflection->Height) with new P/I gains at the current
    setpoint -- the vendor's "write to a running loop". Use it to sharpen a height
    channel that is lagging (low gain -> banded height, crisp error): bump `igain`
    and watch the height tracking improve. Hardware backend only; the twin models no
    servo to re-arm, so it is REFUSED (`-32010`) rather than silently ignored -- plan
    Sec A says so, and a script that believes it tuned a loop it did not is worse off
    than one that was told no.
    """
    body = params or {}
    for key in ("setpoint", "pgain", "igain"):
        _number(body, key, None, verb="setZGain")
    hw = _real_bridge(hub)
    if hw is None:
        msg = "Z gain adjust is hardware-only; no bridge connected."
        raise CommandError(BACKEND_MISMATCH, msg,
                           {"backend": hub.backend_label()},
                           http={"error": msg})
    # THE TIP MUST ALREADY BE ON THE SURFACE, and this gate is the difference between
    # a tune and an ENGAGE. `hw_set_zgain` re-arms loop 2 UNCONDITIONALLY (Deflection
    # -> Z, output -10..150): called with the tip withdrawn it CLOSES the Z servo at
    # whatever setpoint it was handed -- and with no prior approach `z_loop` is empty,
    # so the defaults below hand it 1.0 V and I=10, which drives the piezo toward the
    # surface. `hub.engaged` would still be False, so `getState` would read `idle`
    # while the tip went down: an engage wearing a tune's name, with none of the
    # approach path's guards. Refused on BOTH doors -- approach first, then tune.
    if not getattr(hub.host, "engaged", False):
        msg = ("Z gain adjust REFUSED: the tip is not engaged. This re-arms the "
               "RUNNING Z loop, so from a withdrawn tip it would close the servo and "
               "drive the piezo toward the surface. Approach first, then tune.")
        await hub.log(msg)
        raise CommandError(REFUSED, msg, {"reason": "not_engaged"},
                           http={"error": msg, "reason": "not_engaged"})
    from ...hardware.hw_approach import hw_set_zgain
    z = getattr(hub.host, "z_loop", None) or {}
    setpoint = float(body.get("setpoint",
                              z.get("setpoint", hub.scan_params.get("setpoint", 1.0))))
    pgain = float(body.get("pgain", z.get("pgain", 0.0)))
    igain = float(body.get("igain", z.get("igain", 10.0)))
    # SAFETY-CRITICAL: a live gain tune must keep the mode the loop was ENGAGED in,
    # so it re-arms the running loop on the same feedback channel + gain sign instead
    # of silently flipping an AC loop back to Deflection / positive gain mid-run (a
    # crash). Inherit the engaged mode from z_loop; an explicit body mode may override.
    mode = str(body.get("mode", z.get("mode", "contact"))).lower()
    hub.host.z_loop = {"setpoint": setpoint, "pgain": pgain, "igain": igain, "mode": mode}
    await hub.log(f"Z gain (live, {mode}): setpoint {setpoint:g} V, "
                  f"P={pgain:g} I={igain:g}")
    snap = await asyncio.to_thread(hw_set_zgain, hw, setpoint=setpoint,
                                   pgain=pgain, igain=igain, mode=mode)
    await hub.broadcast({"type": "engaged", **snap})
    return snap


# =================================================================== startScan ===

def scan_est_s(config) -> float:
    """Lines / line rate, doubled for trace+retrace, plus the margin."""
    try:
        lines = int(getattr(config, "lines", 0) or 0)
        rate = float(getattr(config, "line_rate_hz", 0) or 0)
    except (TypeError, ValueError):
        return EST_MARGIN_S
    if lines <= 0 or rate <= 0:
        return EST_MARGIN_S
    passes = 2 if bool(getattr(config, "trace_retrace", False)) else 1
    return round(lines / rate * passes + EST_MARGIN_S, 3)


_FALSY = {"false", "0", "no", "off", "none", ""}   # every spelling of "not chained"


def build_scan_config(cfg: dict):
    """The wire/HTTP body -> `ScanConfig`. One coercion, both doors."""
    from ..passes import ScanConfig
    cfg = cfg or {}
    if "chained" in cfg and str(cfg["chained"]).strip().lower() in _FALSY:
        msg = "startScan: unchained scan was retired; omit 'chained'"
        raise CommandError(INVALID_PARAMS, msg, {"param": "chained"},
                           http={"error": msg, "reason": "bad_params"})
    # PANEL-gain convention (see ScanConfig.pgain/igain): validated as finite
    # numbers here, same as any other body field, on BOTH doors. The engage-gate
    # (pgain/igain are inert without `engage: true`) is the wire door's own refusal
    # -- see `start_scan` -- exactly like `setpoint`.
    pgain = _finite_number(cfg, "pgain", None, verb="startScan")
    igain = _finite_number(cfg, "igain", None, verb="startScan")
    return ScanConfig(
        size_um=float(cfg.get("size_um", 30.0)),
        pixels=int(cfg.get("pixels", 256)),
        lines=int(cfg.get("lines", 0)),
        line_rate_hz=float(cfg.get("line_rate_hz", 1.0)),
        x_offset_um=float(cfg.get("x_offset_um", 0.0)),
        y_offset_um=float(cfg.get("y_offset_um", 0.0)),
        mode=str(cfg.get("mode", "contact")),
        setpoint=float(cfg.get("setpoint", 0.2)),
        angle_deg=float(cfg.get("angle_deg", 0.0)),
        direction=str(cfg.get("direction", "up")),
        trace_retrace=bool(cfg.get("trace_retrace", True)),
        engage=bool(cfg.get("engage", False)),
        chained=True,
        pgain=pgain,
        igain=igain,
    )


async def scan_start(hub, cfg: dict) -> dict:
    """Arm a raster and hand the streaming to a pump. Returns the HTTP door's shape
    (`{ok, config, run_id, backend?}`); `start_scan` renders the wire's."""
    gate = _live_or_refuse(hub, "startScan")
    if gate is not None:
        await hub.log("Scan start REFUSED: " + gate.message)
        raise gate
    config = build_scan_config(cfg)
    # The panel always sends a setpoint (it is a field), and on hardware it is INERT
    # unless the scan engages -- so say so once, in the log the operator is already
    # reading, rather than let the number sit in the body meaning nothing. The wire
    # door REFUSES this combination outright (`start_scan`) and never reaches here.
    if (hub.backend_live() == "hardware" and (cfg or {}).get("setpoint") is not None
            and not bool(getattr(config, "engage", False))):
        await hub.log(f"Scan setpoint {config.setpoint:g} V is INERT: this scan does "
                      f"not engage, so it runs at the CURRENT engagement force. Tick "
                      f"engage to make the setpoint mean something.")
    running = getattr(hub.host, "_scan_task", None)
    if running is not None and not running.done():
        msg = "scan already running"      # said before the claim, which would answer "busy"
        raise CommandError(BUSY, msg, {"run_id": hub._in_flight_run_id(), "kind": "scan"},
                           http={"error": msg})
    # THE SINGLE-WRITER CLAIM, BEFORE THE FIRST AWAIT (plan Sec E.2). Same race as
    # the write path: this path awaits the drift log and the software-mode reconcile
    # before it creates the pump task, so without the claim two starts can interleave
    # through that window. Handed to whichever pump task actually starts
    # (`claim.run_id` is stamped at hand-over); released here on every path that
    # starts none -- which includes the continuous loop's own frames, since the
    # supervisor awaits each pump before arming the next.
    claim = _claim(hub, "scan")
    try:
        return await _scan_start_body(hub, cfg, config, claim)
    finally:
        if claim.worker is None:
            hub.release_claim(claim)


def _store_scan_run(hub, config, tip_down, mode_gate_closed, mode_before, drift_offset):
    """Build and store this scan's run; returns (its reference or "no run: <reason>", the run
    or None). Never raises: any failure, in building or in the store's files, leaves the scan
    as it was."""
    try:
        from .. import run_store
        runtime = hub.store.path.parent
        out = run_store.store_scan_run(
            run_store.RunStore(run_store.store_dir(runtime)), runtime, config,
            platform=run_store.platform_name(hub.host.hardware_arc), imaging_mode=mode_before,
            engaged=tip_down, mode_gate_closed=mode_gate_closed, drift_offset=drift_offset,
            ends_with_withdraw=not hub.host.scan_repeat.get("enabled"))
    except Exception as exc:  # noqa: BLE001
        return f"no run: {type(exc).__name__}: {exc}", None
    return (out, None) if isinstance(out, str) else (out[1], out[0])


async def _refuse_stopped_scan_start(hub) -> None:
    """An abort (Igor's own abort panel included) arrived while the scan was starting,
    before its run was stored or anything armed: refuse. A stop after this point is the
    pump's (`stops_at_start`)."""
    msg = "scan start interrupted: an abort was signalled while the scan was starting"
    await hub.log("Scan start REFUSED: " + msg)
    raise CommandError(REFUSED, msg, {"reason": "aborted_during_start"},
                       http={"error": msg, "reason": "aborted_during_start"})


async def _scan_start_body(hub, cfg: dict, config, claim) -> dict:
    """The body of `scan_start`, run while holding the single-writer claim."""
    from ..drift import compensation_offset
    from ..host.scan_pump import _hw_scan_pump, _scan_pump
    from ..step_notes import make_notes
    # THE PUMP SLOT IS CHECKED HERE, BEFORE ANY MUTATION. A start that arms first (the
    # hardware branch sets `hub.engaged`) and only then asks whether a pump task is free
    # loses the race with the engaged flag set and a null run_id that its caller reports
    # as busy. A refusal must leave the instrument exactly as it found it.
    task = getattr(hub.host, "_scan_task", None)
    # The pump clears the abort flag when it starts, so a stop from here on is judged by
    # this counter, which is never cleared: the start refuses, or the pump aborts.
    stops_at_start = int(getattr(hub.host, "_run_abort_seq", 0) or 0)
    if task is not None and not task.done():
        msg = "scan already running"
        raise CommandError(BUSY, msg,
                           {"run_id": hub._in_flight_run_id(), "kind": "scan"},
                           http={"error": msg})
    # Remember the body so Continuous imaging can re-arm from the last scan when
    # enabled with no explicit config (see host/continuous.py).
    hub.host.last_scan_config = dict(cfg or {})
    # LAST-USED SCAN SETPOINT PER MODE (docs/INSTRUMENT_MODEL_SPEC.md sec 4/10): the
    # cockpit hydrates the scan #setpoint from hub.last_scan_params[mode] through GET
    # /api/instrument, so a reload never resets it to an HTML literal. Recorded here,
    # once the start is committed, keyed by the scan's mode. (No slice owned this; the
    # spec names it alongside hub.last_write_params, which B2 records on the write door.)
    _smode = str((cfg or {}).get("mode") or getattr(config, "mode", None) or "contact")
    hub.host.last_scan_params[_smode] = {"setpoint": getattr(config, "setpoint", None)}
    # COMPENSATE (Connect Phase 2): when Track-and-compensate drift is on, the
    # backend owns the compensation -- add the accumulated drift to the incoming
    # offset so the next scan re-centres on the features, and broadcast the
    # effective offset so the cockpit's offset fields follow. Same sign as the
    # port TRACK (both add the drift); no-op off, or when no reference is set.
    _cox, _coy = compensation_offset(hub)
    if _cox or _coy:
        config.x_offset_um += _cox
        config.y_offset_um += _coy
        await hub.log(f"Drift compensation: scan offset -> "
                      f"({config.x_offset_um:+.3f}, {config.y_offset_um:+.3f}) µm")
        await hub.broadcast({"type": "drift", "offset": [config.x_offset_um,
                             config.y_offset_um], "compensated": True})
    # ---- INSTRUMENT-MODEL COMMAND GUARD (docs/INSTRUMENT_MODEL_SPEC.md secs 2/10/13)
    # BEFORE any raster is armed, on BOTH doors and BOTH backends: a latched fault, or
    # a scan frame (offset +/- size/2, rotated by angle) that would leave the model's
    # xy_range, refuses the scan here. Checked on the EFFECTIVE (drift-compensated)
    # frame -- the frame the tip will actually raster. Needed for tiled imaging by
    # offset stepping (Sec 2).
    from ..command_guard import check_scan
    try:
        check_scan(hub.host.instrument_model, config.size_um, config.x_offset_um,
                   config.y_offset_um, getattr(config, "angle_deg", 0.0),
                   fault=getattr(hub, "fault", None))
    except CommandError as exc:
        await hub.log("Scan start REFUSED: " + exc.message)
        raise
    # REAL backend: drive the raster as a capturing pass on the instrument (the
    # controller does the 50 kHz work; we capture ZSensor/Deflection ourselves).
    # real_bridge is None off-instrument, so this branch is never taken in sim.
    if _real_bridge(hub) is not None:
        # Consistency guard: reconcile the software imaging mode with this scan's mode
        # (auto-switch when withdrawn, warn when engaged) so an AC scan can never run with
        # the software still in Contact (dead crosspoint, no drive) -- see
        # ensure_software_mode. A no-op when already consistent.
        # What the scan's own checks see, read before they act: the run is built from it.
        tip_down = bool(getattr(hub.host, "engaged", False))
        mode_gate_closed = engaged_or_scanning(hub)[0]
        mode_before = getattr(hub.host, "imaging_mode", None)
        switched = await ensure_software_mode(hub, config.mode)
        if int(getattr(hub.host, "_run_abort_seq", 0) or 0) != stops_at_start:
            await _refuse_stopped_scan_start(hub)
        # Off the loop, and before the tip state below changes: a stop never waits on the disk.
        run_ref, stored_run = await asyncio.to_thread(
            _store_scan_run, hub, config, tip_down, mode_gate_closed, mode_before,
            bool(_cox or _coy))
        notes = make_notes(stored_run, "scan")
        # SAFETY-CRITICAL (audit fix): an ENGAGING scan closes the Z servo (hw_scan
        # arms loop 2 at config.setpoint when config.engage) and does NOT withdraw on
        # completion -- so when the pump task finishes the tip is STILL on the surface.
        # Set hub.engaged now, so a completed engaged scan keeps the mode-switch gate
        # CLOSED (and ensure_software_mode won't auto-switch at the next scan) until a
        # withdraw/abort/safe_park clears it. A retracted scan (engage false) leaves the
        # tip off the surface, so the flag stays untouched.
        if bool(getattr(config, "engage", False)):
            hub.host.engaged = True
        if not hub.host.scan_repeat.get("enabled"):
            from ..host.engage import arm_interlocks as _arm_interlocks
            _arm_interlocks(hub)     # single scan: armed here (continuous arms in continuous_cmds)
        await hub.log(f"Scan started (hardware): {config.size_um:g} um, "
                      f"{config.pixels} px, {config.mode}")
        # Allocate the run id BEFORE the task, so the id in the response is the one
        # the pump closes (plan Sec A.1). Additive field: the GUI ignores it, a
        # scripted caller fetches getScanResult(run_id) with it.
        run_id = None
        if hub.host._scan_task is None or hub.host._scan_task.done():
            hub.host._scan_epoch += 1      # a new pass claims the tip (race-guard, Sec 2)
            run_id = hub.next_run_id("scan")
            hub.runs.annotate(run_id, run_ref=run_ref)
            hub.host._scan_task = P.spawn_untraced(
                _hw_scan_pump(config, run_id, claim, notes, bool(switched),
                              stops_at_start=stops_at_start))
            # Hand the claim to the pump TASK: from here its validity is judged by
            # that task, not by a hub field a later run may replace.
            claim.hand_over(run_id, hub.host._scan_task)
        # literal: the hardware scan pump is already running (see above).
        return {"ok": True, "config": config.__dict__, "backend": "hardware",
                "run_id": run_id}
    # SIMULATION backend (default): the twin plugin plays the scan (host/scan_pump.py).
    if config.mode not in ("contact", "ac"):
        msg = f"the twin cannot scan in {config.mode} mode; use contact or ac"
        await hub.log("Scan start REFUSED: " + msg)
        raise CommandError(REFUSED, msg, {"reason": "mode_not_on_twin"}, http={"error": msg})
    # Stored inline: the twin has no instrument to stop, and a thread hop here would shift the
    # scan's timing relative to the telemetry polls.
    run_ref, _ = _store_scan_run(
        hub, config, bool(getattr(hub.host, "engaged", False)),
        engaged_or_scanning(hub)[0], getattr(hub.host, "imaging_mode", None), bool(_cox or _coy))
    hub.host._scan_epoch += 1              # a new pass claims the tip (race-guard, Sec 2)
    if not hub.host.scan_repeat.get("enabled"):
        from ..host.engage import arm_interlocks as _arm_interlocks
        _arm_interlocks(hub)         # single scan: armed here (continuous arms once in continuous_cmds)
    await hub.log(
        f"Scan started: {config.size_um:g} um, {config.pixels} px, "
        f"{config.line_rate_hz:g} Hz, {config.mode}")
    run_id = None
    if hub.host._scan_task is None or hub.host._scan_task.done():
        run_id = hub.next_run_id("scan")
        hub.runs.annotate(run_id, run_ref=run_ref)
        hub.host._scan_task = P.spawn_untraced(_scan_pump(config, run_id, claim))
        claim.hand_over(run_id, hub.host._scan_task)   # the pump owns it from here
    return {"ok": True, "config": config.__dict__, "run_id": run_id}


def _refuse_inert_scan_fields(hub, cfg: dict) -> None:
    """The wire door's refusals, all decided before any await. On hardware a `setpoint`,
    `pgain` or `igain` without `engage: true` would be ignored, so refuse. Setpoint first."""
    if (hub.backend_live() == "hardware" and cfg.get("setpoint") is not None
            and not bool(cfg.get("engage"))):
        msg = ("startScan: `setpoint` is inert on hardware unless `engage: true` -- "
               "the scan would run at the CURRENT engagement force, not this one. "
               "Send engage:true to engage at this setpoint, or drop `setpoint`.")
        raise CommandError(INVALID_PARAMS, msg,
                           {"param": "setpoint", "engage": bool(cfg.get("engage"))})
    if (hub.backend_live() == "hardware" and not bool(cfg.get("engage"))
            and (cfg.get("pgain") is not None or cfg.get("igain") is not None)):
        bad = "pgain" if cfg.get("pgain") is not None else "igain"
        msg = (f"startScan: `{bad}` is inert on hardware unless `engage: true` -- "
               f"the scan would run with the CURRENT Z loop gains, not this one. "
               f"Send engage:true to engage with this {bad}, or drop `{bad}`.")
        raise CommandError(INVALID_PARAMS, msg,
                           {"param": bad, "engage": bool(cfg.get("engage"))})


async def start_scan(hub, params: dict | None = None) -> dict:
    """The WIRE door: `{run_id, est_s}`, start+poll.

    Extra refusals the HTTP door does not make, and deliberately so (plan Sec A):
    on HARDWARE a `setpoint`, `pgain` or `igain` is INERT unless `engage: true` --
    the scan runs at whatever the existing engagement/Z gains already are.
    Accept-and-ignore is the shape of a script that images a whole array at the
    wrong force (or gain) and never finds out. The cockpit's own scans always carry
    a setpoint (it is a panel field, sent whether or not the scan engages) and never
    a pgain/igain (there is no such panel field), so a blanket rule would refuse the
    panel on setpoint alone; this lives on the wire door, where the caller cannot
    see the panel state that makes the field meaningful. `pgain`/`igain` are PANEL
    gains, the same convention `setZGain` takes (see ScanConfig.pgain/igain).
    """
    cfg = dict(params or {})
    _refuse_inert_scan_fields(hub, cfg)
    res = await scan_start(hub, cfg)
    run_id = res.get("run_id")
    if run_id is None:
        # The pump slot was still occupied: the raster is armed but this call owns no
        # run. Report what IS in flight rather than hand back a null id a script would
        # poll for ever.
        raise CommandError(BUSY, "a scan pump is already running",
                           {"run_id": hub._in_flight_run_id(), "kind": "scan"})
    config = res.get("config") or {}
    est = scan_est_s(build_scan_config(config) if isinstance(config, dict) else config)
    return {"run_id": run_id, "est_s": est, "state": "scanning",
            "backend": hub.backend_label(), "config": config}
