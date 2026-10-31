"""The WRITE-domain service layer: load a design, start a write, abort one.

Truth document: docs/FLEX_INTEGRATION_PLAN.md Sec A (the instrument-operations rows
`loadPattern` / `startWrite` / `getWritten` / `getWriteResult`), C (this layer -- "one
coroutine per FLEX verb, each raising a typed CommandError and returning plain dicts;
each router body becomes a thin adapter"), E.2 (where the single-writer claim is
taken), F.1/F.3 (start+poll, the state machine), J risk 11 (the AC-guard ordering).
Wire reference: docs/FLEX_PROVIDER.md. Rig checks: docs/FLEX_RIG_RUNBOOK.md "09".

WHAT MOVED, AND WHAT DELIBERATELY DID NOT
-----------------------------------------
The BODIES of `POST /api/litho/load`, `/api/litho/write` and `/api/litho/abort` live
here now; `service/routers/litho.py` keeps the decorated endpoints as adapters that
call these and render a `CommandError` back into the HTTP-200 `{"error": ...}`
convention the cockpit has always seen. There is ONE implementation of the
contact-only guard, the silent-twin refusal and the single-writer claim, so a FLEX
script and the panel cannot drift apart (plan Sec D).

What did NOT move: the two write WORKERS (`_twin_litho_worker`, `_hw_litho_worker`)
and the PUMP (`_litho_pump`) that drains them onto the WebSocket and persists the
pass. They are streaming machinery, not the start path, and they close over the
process hub through `host/litho_worker.py`; the start verbs import them lazily, exactly
as `scan_cmds` imports the scan pumps.

THE AC-GUARD ORDERING IS COPIED VERBATIM (plan Sec J risk 11)
-------------------------------------------------------------
`check_mode` and `ac_mode_active` below are the litho router's functions, moved with
their priority order and their docstrings intact and NOT re-derived. The ordering is
the difference between "cannot prove we are in AC, so refuse" and "cannot prove
anything, so proceed": a FAILED crosspoint read must fall through to the engaged-mode
and conservative branches, and must NEVER be read as "contact". A test pins it
(tests/test_flex_litho.py, "an unreadable crosspoint never downgrades to contact") so
a later refactor cannot quietly reorder it.

THE HUB IS A PARAMETER, with the same caveat `scan_cmds` states: the pump and the
workers still read the process hub, so passing a DIFFERENT hub here is honest only
for the paths that never reach them (the guard functions, and every refusal).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import math
import queue as _queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from ...hardware import bridge_protocol as P

# The ONE limit a script needs, read off the module that owns it rather than retyped:
# MAX_OUTWAVE_POINTS (87,000, the v19 vendor `cMaxOutWave`) is the DSP's per-OUT-WAVE
# cap, and since 2026-09-11 it is the ONLY thing that chunks a write. The compiler's
# 8,192-point `MAX_SEGMENT_POINTS` used to and no longer does (it is opt-in now), so it
# is gone from this wire. ADVISORY -- not a refusal (decision of 2026-09-11).
from ...hardware.lithography_run import RAMP_SETTLE_MARGIN_S, SETTLE_MOVE_S, WriteSetpoint
from ...hardware.xy_transit import DEFAULT_WITHDRAW_MAX_S, DEFAULT_WITHDRAW_SETTLE_S
from ...litho.chunker import (
    CHAIN_CHUNK_POINTS,
    MAX_OUTWAVE_POINTS,
    chunk_boundaries,
    is_chained,
)
from ..commands.errors import BACKEND_MISMATCH, BUSY, INVALID_PARAMS, REFUSED, CommandError

# The margin added to a published `est_s`, for the same reason `scan_cmds` adds one:
# a driver's blocking helper SLEEPS on this before it starts polling, so an estimate
# that is systematically short reports timeouts that never happened.
EST_MARGIN_S = 5.0

# The per-object transit + arm + settle overhead a write pays beyond the trajectory
# time itself (lift, slew to the first point, arm the loops, fire, lift again). A
# rough constant on purpose: `est_s` is a SLEEP hint, not a schedule.
PER_OBJECT_OVERHEAD_S = 3.0

# The XY_SLEW_SPEC verified-withdraw cost that now runs on the move TO EACH NEW
# SOURCE OBJECT (docs/XY_SLEW_SPEC.md rule 5) -- NOT between strokes of the SAME
# one (see PER_STROKE_TRANSIT_S below, restored to its pre-withdraw value for
# exactly that reason: a fill row's gap inside one shape is still the old mild
# hover-lift, `hardware/lithography_run._hover_move`, no withdraw at all).
# `hardware.lithography_run.write_object` takes this path (`xy_transit`, reapproach
# left to the caller's own set-down) whenever the incoming stroke's `source`
# differs from the previous call's -- i.e. once per DISTINCT object, accounted here
# SEPARATELY from PER_OBJECT_OVERHEAD_S (a rough existing margin for the rest of
# the per-object cost: arm the loops, fire) rather than folded into it, so a later
# change to either is legible on its own:
#
#   hw_withdraw        the ramp's own bound              DEFAULT_WITHDRAW_MAX_S
#                      ... plus its settle-verify margin  DEFAULT_WITHDRAW_SETTLE_S
#   _settle_move       after the set-down _arm_z           SETTLE_MOVE_S
#   _settle_move       after the end-of-stroke lift _arm_z SETTLE_MOVE_S
#
# = DEFAULT_WITHDRAW_MAX_S + DEFAULT_WITHDRAW_SETTLE_S + 2 x SETTLE_MOVE_S = 4.5 s
# (the ZERO-HOP floor; a real hop at MOVE_SPEED_UM_S adds more -- the move itself
# still takes distance/MOVE_SPEED_UM_S). Charged once per object in `write_est_s`
# below, same rough-estimate shape as PER_OBJECT_OVERHEAD_S (it necessarily over-
# charges the FIRST object of a write, whose move pays no withdraw at all -- the
# tip was never engaged -- which is conservative, same as this estimator always
# was).
#
# 2026-09-16: DROPPED a `+ RAMP_SETTLE_MARGIN_S` term this used to carry for
# xy_transit's own move-completion wait. That wait used to be a real blocking
# `time.sleep` (a rate-limited `ramp_output` on each axis, waited out by the
# caller) -- XY_SLEW_SPEC.md's single out-wave-pair move instead blocks on the
# bridge's own DONE report (`LithographyRun._fire_and_wait`, the SAME cycle the
# litho stroke's own ARMFIRE/LITHOCHAIN uses), which is not a `time.sleep` this
# estimator can or should count -- it is already bounded by the pass's own
# timeout, not an extra sleep on top of it.
PER_OBJECT_WITHDRAW_S = (DEFAULT_WITHDRAW_MAX_S + DEFAULT_WITHDRAW_SETTLE_S
                         + 2 * SETTLE_MOVE_S)

# The overhead ONE INTRA-OBJECT TRANSIT costs, beyond the trajectory time: the gap
# between two DISCONNECTED STROKES OF THE SAME source object (the rows of a filled
# shape, the two regions of an annulus) -- `_hover_move`'s mild lift, never a
# verified withdraw (see PER_OBJECT_WITHDRAW_S above for the between-OBJECT cost).
# Count the sleeps `_hover_move` actually performs around one stroke, in order:
#
#   _settle_move       after the lift servoloop            SETTLE_MOVE_S
#   (ramp wait)        maxsec, floored at ...              RAMP_SETTLE_MARGIN_S
#                      ... plus its own settle             SETTLE_MOVE_S
#   _settle_move       after the set-down _arm_z            SETTLE_MOVE_S
#   _settle_move       after the end-of-stroke lift _arm_z SETTLE_MOVE_S
#
# = 4 x SETTLE_MOVE_S + RAMP_SETTLE_MARGIN_S = 4.5 s. This is the ZERO-HOP FLOOR: it
# is what a transit costs when the two strokes START AND END AT THE SAME POINT, since
# `_plan_ramp` bounds `maxsec` at `dist/MOVE_SPEED_UM_S + max(0.5, 0.25*that)` and a
# real hop makes it larger (5 um/s, so a 10 um gap between fill rows adds ~2 s).
#
# NOT charged inside a stroke any more: until 2026-09-11 the compiler cut every stroke
# at 8,192 points and this whole cycle ran at each cut, which is exactly the tip-lift
# bug that was fixed -- a stroke is now continuous whatever its length.
PER_STROKE_TRANSIT_S = 4 * SETTLE_MOVE_S + RAMP_SETTLE_MARGIN_S

# The dwell at ONE CHAIN BOUNDARY inside an over-cap stroke. No lift and no XY move,
# so none of SETTLE_MOVE_S / RAMP_SETTLE_MARGIN_S applies -- but this is NOT a cheap
# re-arm, and the number must not read like one:
#
#   * `AFM_LithoAdvance` runs INSIDE the capture-fill callback and, before it defers
#     anything, formats and sends FIVE captures of ~87k points each -- ~4.8 MB of
#     "%.9g;" text, built with an O(N^2) `s += tmp` string concat in the .ipf;
#   * only then does it set AFM_ltPending, so the 60 Hz `AFM_LithoTick` background
#     task waits at least one tick before `AFM_LithoArmChunk` runs;
#   * `AFM_LithoArmChunk` then does five `Make/O` of ~87k points and re-binds the
#     banks.
#
# That is SECONDS, not tenths -- and it happens with the tip STATIONARY AT FULL BIAS
# at one point of the trajectory, because the out-wave has stopped playing while the
# bias output holds. On the hardware write path the cut is always at exactly
# `start + CHAIN_CHUNK_POINTS`: `chunk_boundaries` prefers a pen-up, but a stroke
# reaching `write_object` has already been split at its pen-ups by the compiler, so
# there is no NaN in the window and the preference never fires. A stationary biased
# dwell of seconds at a single point is a DOSE-BLOB hazard, which is why `loadPattern`
# says so by name.
#
# 3 s is a defensible FLOOR, NOT A MEASUREMENT -- LITHOCHAIN has never fired on
# hardware. RIG-CALIBRATE: runbook 09 (h) step 6 times it off the Igor history
# (wall-clock between the chunk-0 and chunk-1 `[arm/chunk] FIRED` lines), not by
# subtracting the compiler's duration from the pass.
CHAIN_SEAM_S = 3.0

# How many write traces `getWriteResult` keeps in memory, and how many points each
# object contributes to one. Bounded on BOTH axes because this rides a REP-thread
# reply: goal 2 wants the current-along-path trace inline, and "inline when small" has
# to be enforced somewhere rather than hoped for.
TRACE_KEEP_RUNS = 8
TRACE_MAX_POINTS_PER_OBJECT = 512
# ...and a cap on the WHOLE run, because the per-object one is not a bound: a
# 200-object design hits 100k points per run and ~2.5 MB on a reply whose published
# bound is 200 ms. The run budget is spread evenly across the objects that actually
# have points, so a big design comes back coarser rather than enormous.
TRACE_MAX_POINTS_PER_RUN = 20_000

# The widest an imported SVG may be before the flatten is refused, in microns. Same
# number and same reasoning as `POST /api/design/svg` -- no scanner writes past a few
# hundred um, so a design that comes out metres wide is malformed, not ambitious.
MAX_DESIGN_UM = 2000.0


# =============================================================== validation ======

def _real_bridge(hub):
    """The REAL hardware ARC bridge attached to `hub`, or None on the twin.
    Local + lazy (imports only SoftwareARC, never the hub singleton) so
    importing this module stays cheap -- see hub.real_bridge for the rule."""
    from ...twin.arc import SoftwareARC
    arc = getattr(hub.host, "hardware_arc", None)
    if arc is None or isinstance(arc, SoftwareARC):
        return None
    return arc


async def _refuse_if_aborted(hub, during: str) -> None:
    """Refuse a hardware write start (reason `aborted_during_start`) when an abort or
    park was signalled, or a fault latched, since `_hw_litho_abort` was cleared at
    write start. The worker self-engages, so a missed park would re-engage the tip."""
    if not (hub.host._hw_litho_abort.is_set() or getattr(hub, "fault", None)):
        return
    why = ("a fault latched" if getattr(hub, "fault", None)
           else "an abort or park was signalled")
    msg = f"write start interrupted: {why} while {during}"
    await hub.log("Lithography REFUSED: " + msg)
    raise CommandError(REFUSED, msg, {"reason": "aborted_during_start"},
                       http={"error": msg, "reason": "aborted_during_start"})


async def _ensure_write_free_air(hub) -> None:
    """13.1 item 27: the write door measures free air itself, like the approach door.

    Withdrawn (not engaged, nothing running) and no fresh contact baseline on record
    (missing, stale, or taken in another mode) -> take a fresh withdrawn contact read
    BEFORE the setpoint conversion and `check_write`. Engaged -> leave `hub.free_air`
    alone and do not refuse an old or missing baseline: the hardware run reads free air
    before every object (docs/PER_OBJECT_FREE_AIR_SPEC.md). Logs the baseline on record
    at every write start.
    """
    from ..host.engage import establish_free_air_async, free_air_is_fresh
    from .scan_cmds import engaged_or_scanning
    engaged, _why = engaged_or_scanning(hub)
    fa = getattr(hub.host, "free_air", None)
    if not engaged and not free_air_is_fresh(hub, "contact"):
        if fa is None:
            reason = "missing"
        elif getattr(fa, "mode", None) != "contact":
            reason = "mode"
        else:
            reason = "stale"
        await hub.log(f"write: free air re-read (reason: {reason})")
        try:
            await establish_free_air_async(hub, "contact", force=True)
        except CommandError as exc:
            await hub.log("Lithography REFUSED: " + exc.message)
            raise
        fa = getattr(hub.host, "free_air", None)
    if fa is None:
        await hub.log("write: free-air baseline none")
    else:
        age = time.time() - float(fa.ts)
        await hub.log(f"write: free-air baseline {fa.value_v:.4f} V "
                      f"({fa.mode}, age {age:.1f} s)")


def _number(body: dict, key: str, default, *, verb: str):
    """One numeric parameter, or a `-32602` that NAMES it. Same rule as
    `scan_cmds._number`: numeric strings are accepted, a bool is not (`float(True)`
    is 1.0, and `setpoint: true` is a typo, not a volt)."""
    if not isinstance(body, dict) or body.get(key) is None:
        return default
    value = body[key]
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise _bad_param(verb, key, value)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise _bad_param(verb, key, value) from None


def _twin_speed(hub, body: dict, *, verb: str) -> float:
    """The speed this write is played at on the twin: the call's own `twin_speed`,
    else the station's (`hub.twin_speed`). Validated on every backend, so a typo is
    a -32602 rather than a write at a speed nobody asked for; the twin plugin plays at it,
    the rig ignores it."""
    from ..twin_player import DEFAULT_SPEED, parse_speed
    value = body.get("twin_speed") if isinstance(body, dict) else None
    if value is None:
        return float(getattr(hub.host, "twin_speed", DEFAULT_SPEED))
    try:
        return parse_speed(_number(body, "twin_speed", None, verb=verb))
    except ValueError:
        msg = f"{verb}: twin_speed must be a number >= 0, got {value!r}"
        raise CommandError(INVALID_PARAMS, msg, {"param": "twin_speed"},
                           http={"error": msg, "reason": "bad_params"}) from None


def _bad_param(verb: str, key: str, value) -> CommandError:
    msg = (f"{verb}: {key} must be a number, got "
           f"{type(value).__name__} {value!r}")
    return CommandError(INVALID_PARAMS, msg, {"param": key},
                        http={"error": msg, "reason": "bad_params"})


def _live_or_refuse(hub, what: str) -> CommandError | None:
    """The SILENT-TWIN GUARD (plan Sec C.2). Identical rule to `scan_cmds`, and it
    matters more here than anywhere: an overnight script must never "write" a whole
    chip against simulated physics while every call returns ordinary success.
    Returned rather than raised, so the caller logs its own wording first (the log
    lines are asserted by the cockpit's tests)."""
    gate = hub.require_live_backend()
    if not gate:
        return None
    return CommandError(BACKEND_MISMATCH, gate,
                        {"backend": hub.backend_label(), "verb": what,
                         "required": getattr(hub.host, "backend", None)},
                        http={"error": gate, "reason": "backend_not_live",
                              "backend": hub.backend_label()})


# ============================================================== the AC guard =====
#
# MOVED VERBATIM from `service/routers/litho.py` (plan Sec C: "the AC-guard's
# `_ac_mode_active` fail-open ordering in the litho router is correct as written and
# must be preserved verbatim"). The only change is that the hub arrives as an
# argument instead of being closed over; the router keeps zero-argument forms that
# bind the process hub, because tests and the mode-switch gate import those names.


def check_mode(hub) -> str:
    """The AUTHORITATIVE instrument imaging mode from the live hardware crosspoint:
    "contact" | "ac" | "unknown". Delegates to HardwareARC.check_mode (which reads the
    crosspoint's InFast discriminator: FilterOut -> contact, ACDefl -> ac). Only
    meaningful on the hardware backend; on the twin (no hardware_arc, no crosspoint) it
    returns "unknown" and the guard falls back to the twin's own tracking. NEVER raises
    -- read_crosspoint is failure-safe and returns {} -> "unknown"."""
    hw = _real_bridge(hub)
    if hw is None:
        return "unknown"
    try:
        return hw.check_mode()
    except Exception:                                   # noqa: BLE001
        return "unknown"                                # a failed read never downgrades


def ac_mode_active(hub) -> bool:
    """True when the instrument is currently in AC / tapping mode -- so a litho write
    must be REFUSED (litho is ALWAYS contact; the Z loop must never servo on amplitude).

    SAFETY-CRITICAL. Priority order, authoritative first:

      1. HARDWARE READ-BACK (hardware backend): check_mode() reads the ACTUAL crosspoint.
         "ac" -> refuse; "contact" -> allow (contact + current/c-AFM both route
         FilterOut). "unknown" (read failed / twin / unexpected) -> fall through to (2)
         -- a failed read must NEVER be read as contact and downgrade the guard.
      2. ENGAGED SERVO MODE: if the tip is engaged (hub.engaged) and the engaged Z loop
         records a mode (hub.z_loop["mode"], set by the approach), TRUST it -- an active
         contact engage clears the guard, an AC engage refuses. This fixes the observed
         bug (contact approach after an AC scan) even when the crosspoint can't be read.
      3. NOT ENGAGED (crosspoint state unknown, be conservative): refuse if the last
         commanded imaging mode (hub.imaging_mode) is AC, or the current/last scan
         config is AC (current_scan_is_ac).

    Refusing (not coercing) makes the operator do an explicit contact engage first,
    which re-routes the crosspoint and clears this. The stale-scan-config source is now
    consulted ONLY when not engaged, so a contact engage after an AC scan no longer
    trips on the earlier AC scan config (the bug)."""
    # (1) authoritative hardware read-back
    m = check_mode(hub)
    if m == "ac":
        return True
    if m == "contact":     # check_mode folds current/c-AFM into "contact"
        return False
    # (2) trust the engaged Z-servo mode when the tip is actually engaged
    if getattr(hub.host, "engaged", False):
        z = getattr(hub.host, "z_loop", None) or {}
        mode = str(z.get("mode", "")).lower()
        if mode:
            return mode == "ac"
    # (3) not engaged, no hardware read: conservative fallback
    if str(getattr(hub.host, "imaging_mode", None) or "").lower() == "ac":
        return True
    from ..host.telemetry import current_scan_is_ac
    return current_scan_is_ac()


# ============================================================== loadPattern ======

def _design_path(source: str, *, allow_paths: bool):
    """Resolve a design `source` to a file, or `None` for "no such design".

    TWO RESOLUTION RULES, ON PURPOSE -- and this is the ONLY thing the two doors
    resolve differently. The cockpit's door keeps the confinement it has always had:
    a BASENAME inside `data/designs`, so a name that climbs out
    (`../../pyproject.toml`) reads nothing there. The wire door additionally accepts a
    real path, because plan Sec H's worked script loads
    `source="D:/designs/hall_bar_v7.oas"`, and a script that must first copy its
    design into the service's own tree is a script nobody will write. Every guard
    below -- the compile, the cap, the contact-only rule -- is shared.
    """
    from ..deps import REPO
    if allow_paths:
        try:
            direct = Path(source)
            if direct.is_file():
                return direct
        except OSError:                                  # a name no filesystem likes
            pass
    confined = REPO / "data" / "designs" / Path(source).name
    return confined if confined.exists() else None


def _read_pattern(source: str, *, allow_paths: bool, hub):
    """`source` -> `(Pattern, origin)`, or a `CommandError` describing why not.

    Accepts, in this order: INLINE text (an SVG document -- the one format that
    arrives as text rather than as bytes), then a file, whose suffix picks the reader
    (`.svg` -> the Bezier loader; everything else -> the klayout reader, which takes
    OASIS and GDS through the same call).

    `origin` is the `{"size_um", "source"}` an SVG import was placed with (the
    design-origin rule needs `hub` to resolve the scan size), and `None` for an
    OASIS/GDS design, which carries its own absolute coordinates.
    """
    text = source if isinstance(source, str) else ""
    if "<svg" in text[:4096].lower():
        return _svg_pattern(text, name="inline.svg", hub=hub)
    path = _design_path(text, allow_paths=allow_paths)
    if path is None:
        msg = f"no such design: {source}"
        raise CommandError(INVALID_PARAMS, msg, {"param": "source", "source": source},
                           http={"error": msg})
    if path.suffix.lower() == ".svg":
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            msg = f"could not read {source}: {exc}"
            raise CommandError(INVALID_PARAMS, msg, {"param": "source"},
                               http={"error": msg}) from exc
        return _svg_pattern(body, name=path.name, hub=hub)
    from ...litho.oasis import oasis_to_pattern
    try:
        return oasis_to_pattern(path), None
    except Exception as exc:                             # malformed OASIS, not a 500
        msg = f"could not read {source}: {exc}"
        raise CommandError(INVALID_PARAMS, msg, {"param": "source"},
                           http={"error": msg}) from exc


def _svg_pattern(text: str, *, name: str, hub):
    """An SVG document -> `(Pattern, origin)`, on the SAME terms `POST /api/design/svg`
    uses: 1 SVG unit = 1 nm, always (this lab's files are drawn that way, and a
    per-file scale is one more thing to get wrong); closed outlines become filled
    shapes, open paths become wires.

    PLACED by the design-origin rule (docs/DESIGN_ORIGIN_SPEC.md): the page top-left is
    the scan top-left. The scan size that anchors it is resolved through the shared
    `resolve_import_size_um` (background image, else the current scan frame, else
    refuse), and the page-origin microns are mapped to lab coordinates once, so this
    door and `POST /api/design/svg` land identical geometry for the same SVG."""
    from ...litho.bezier import load_svg
    from ...litho.compiler import LithoObject, Pattern
    from ...litho.frame import bezier_paths_to_lab
    from ..design_import import DesignSizeError, resolve_import_size_um
    try:
        size_um, size_source = resolve_import_size_um(text, hub)
    except DesignSizeError as exc:
        msg = str(exc)
        raise CommandError(INVALID_PARAMS, msg, {"param": "source"},
                           http={"error": msg}) from exc
    try:
        paths = load_svg(text, 1e-3, preserve_origin=True)
    except Exception as exc:                             # malformed file, not a crash
        msg = f"could not read {name}: {exc}"
        raise CommandError(INVALID_PARAMS, msg, {"param": "source"},
                           http={"error": msg}) from exc
    # Page-origin microns -> lab: translate + flip y about the scan centre, once.
    bezier_paths_to_lab(paths, size_um)
    # GUARD THE FLATTEN, exactly as `POST /api/design/svg` does. At 1 nm/unit a normal
    # design is tens of um; a file that comes out metres wide is malformed (or not
    # drawn to the convention), and flattening a 40 mm curve to a 4 nm tolerance is
    # millions of points -- tens of GB and a killed process. No scanner writes past a
    # few hundred um, so bound the design from its control points (the curve lives
    # inside their hull) and refuse an absurd one BEFORE flattening it.
    import numpy as np
    lo = np.array([np.inf, np.inf])
    hi = -lo
    for bp in paths:
        for seg in bp.segments():
            for pt in seg:
                q = np.asarray(pt, dtype=float)
                lo = np.minimum(lo, q)
                hi = np.maximum(hi, q)
    span_um = float(np.max(hi - lo)) if np.isfinite(lo[0]) else 0.0
    if span_um > MAX_DESIGN_UM:
        msg = (f"design spans {span_um:.0f} um -- far larger than any scanner. The "
               f"file is malformed or not drawn 1 unit = 1 nm. Not importing.")
        raise CommandError(INVALID_PARAMS, msg, {"param": "source",
                                                 "span_um": span_um},
                           http={"error": msg})
    objs = []
    # NAME BY SVG `id` WHEN THE FILE GIVES ONE. A script that draws in Inkscape/KLayout
    # and then addresses the objects by name (setObjects, startWrite by id) needs the
    # id it drew with to survive the import verbatim, not be replaced by a positional
    # `<stem>-N`. An element with several subpaths shares one id, so the first takes it
    # and the rest are suffixed `<id>-1`, `<id>-2` (the same shape `path4-1` a drawing
    # program emits). TWO DIFFERENT elements sharing one id is an SVG error, not a
    # subpath split, and is refused by name -- the pattern is left untouched because
    # nothing here is assigned to the hub until load_pattern returns.
    id_elem: dict[str, int] = {}       # id -> the element that first claimed it
    id_next: dict[str, int] = {}       # id -> next `-N` suffix within that element
    used_names: set[str] = set()
    for i, bp in enumerate(paths):
        pts = bp.flatten(0.004)
        if len(pts) < 2:
            continue
        sid = getattr(bp, "svg_id", None)
        elem = getattr(bp, "svg_elem", None)
        if sid:
            if sid in id_elem and id_elem[sid] != elem:
                msg = (f"two SVG elements share the id {sid!r}; ids must be unique so "
                       f"a script can address the object by it. Not importing.")
                raise CommandError(INVALID_PARAMS, msg,
                                   {"param": "source", "id": sid},
                                   http={"error": msg})
            if sid not in id_elem:
                id_elem[sid] = elem
                id_next[sid] = 1
                label = sid
            else:
                label = f"{sid}-{id_next[sid]}"
                id_next[sid] += 1
        else:
            label = f"{Path(name).stem}-{i + 1}"
        if label in used_names:
            msg = (f"the object name {label!r} is claimed twice (an SVG id collides "
                   f"with another object's name). Not importing.")
            raise CommandError(INVALID_PARAMS, msg,
                               {"param": "source", "id": label},
                               http={"error": msg})
        used_names.add(label)
        kind = "path"
        if bp.closed and len(pts) >= 3:
            kind = "electrode" if bp.as_funnel() is not None else "shape"
        objs.append(LithoObject(
            label, kind, [(float(x), float(y)) for x, y in pts],
            bezier=bp.to_dict() if kind != "path" else None,
            voltage=8.0, speed_um_s=4.0))
    if not objs:
        msg = f"no drawable geometry in {name}"
        raise CommandError(INVALID_PARAMS, msg, {"param": "source"},
                           http={"error": msg})
    return (Pattern(name=Path(name).stem, objects=objs),
            {"size_um": size_um, "source": size_source})


# ================================================== loadPattern: `objects` ======
#
# WHY THIS EXISTS. `source` carries GEOMETRY and nothing else: SVG, OASIS and GDS
# have no place to put a tip voltage, let alone a voltage that is modulated in a
# chosen PHASE relation to the geometry. So the two experiments the capstone is for
# -- a chiral nanowire swept over phase, and an erase rectangle at a negative bias --
# could be drawn by a script but not ORDERED by one. `objects` is the parameter door:
# a script names the physics and the service builds the object.
#
# THE BUILDERS ARE NOT RE-IMPLEMENTED HERE. A chiral object is built by
# `litho.pens.chiral_profile` -- the same call `POST /api/design/chiral` makes, with
# the same argument order and the same `fine/8` interpolation spacing -- so the phase
# convention (phi = radians(phase_deg) * hand, registered against the modulated
# path's own ARC LENGTH) is defined in exactly one place. A region is built by
# `render_stroke(..., "region", ...)`, the same pen the cockpit's region tool uses,
# so the fill strategies and the pitch mean the same thing on both doors. If this
# file ever grows its own sinusoid, that is the bug.

OBJECT_KINDS = ("chiral", "region")

# Strict key sets. An UNKNOWN key is refused rather than ignored: `phase` for
# `phase_deg`, or `amp_um` for `y_amp_um`, is a phase sweep that silently writes the
# default phase at every point of the sweep, and a parameter study whose parameter
# never changed is worse than a refusal.
_CHIRAL_KEYS = frozenset({"kind", "name", "points_um", "lambda_um", "y_amp_um",
                          "v0", "v_k", "phase_deg", "hand", "speed_um_s"})
_REGION_KEYS = frozenset({"kind", "name", "shape", "x_um", "y_um", "w_um", "h_um",
                          "pitch_um", "voltage", "fill", "angle_deg", "speed_um_s"})


# ================================================= the SIZE POLICY, one place ===
#
# WHY THIS EXISTS. `source` has always been bounded -- `_svg_pattern` refuses a design
# whose extent is over MAX_DESIGN_UM before it flattens anything -- but `objects` is a
# door a SCRIPT drives, where the size is not drawn, it is typed. Two measured holes
# (adversarial review, 2026-09-12):
#
#   * a 400 x 400 um rect at a 0.5 um pitch was ACCEPTED: 16 million compiled points,
#     ~185 chain chunks and an 89-hour write, none of it refused;
#   * a 3000 x 3000 um rect took the compile past TEN MINUTES on a verb whose
#     published bound is 2 s -- so the caller's own timeout fired and the service was
#     still burning a thread on a pattern nobody would ever receive.
#
# Neither is a lithography request; both are a units slip (nm typed as um is exactly
# 1000x). So this door bounds THREE things, and the first two are arithmetic done
# BEFORE anything is rendered, so an absurd request is refused in milliseconds:
#
#   1. EXTENT   -- MAX_DESIGN_UM, the same 2000 um the SVG door uses, per object. A
#                  scanner is under 100 um; 2000 is already twenty times any of them.
#   2. POINTS   -- MAX_OBJECT_POINTS per object and MAX_TOTAL_POINTS over the pattern,
#                  checked against an ESTIMATE from extent/pitch arithmetic (for a
#                  region that estimate is exact to about 1%; see `_region_points`).
#   3. TIME     -- a hard wall clock on the build+compile, `load_pattern`'s
#                  `_objects_bound_s`, as the backstop for anything the arithmetic
#                  did not see coming.
#
# The point budget is NOT the out-wave cap and does not overlap it: an object over
# MAX_OUTWAVE_POINTS still CHAINS rather than being refused (rule of 2026-09-11).
# This is the far coarser question of whether the request is a lithography job at all.
MAX_OBJECT_POINTS = 2_000_000
MAX_TOTAL_POINTS = 4_000_000
# 2 million compiled points is, at the region pen's 0.02 um interpolation spacing,
# 40 mm of tip travel: about 11 hours at 1 um/s, and 23 chain seams (23 dose blobs).
# Anything past it is a units error, not an experiment -- and the 4 million total
# leaves room for a couple of such objects in one pattern and no more.


def _spec_error(index: int, field: str, msg: str) -> CommandError:
    """Every refusal on this path NAMES THE FIELD and the object index, because a
    script sending a dozen specs cannot otherwise tell which one it got wrong."""
    text = f"loadPattern: objects[{index}].{field}: {msg}"
    return CommandError(INVALID_PARAMS, text,
                        {"param": "objects", "index": index, "field": field},
                        http={"error": text})


def _spec_num(spec: dict, index: int, field: str, *, default=None,
              positive: bool = False, non_negative: bool = False) -> float:
    """One number out of a spec, or a `-32602` naming it.

    NaN and the infinities are refused HERE rather than left to surface as a silent
    NaN in an out-wave: a NaN in a litho trajectory is a pen-up marker to the chunker
    and an undefined DAC value to the controller, and neither is what a typo meant.
    """
    raw = spec.get(field, None)
    if raw is None:
        if default is None:
            raise _spec_error(index, field, "required")
        return float(default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise _spec_error(index, field,
                          f"must be a number, got {type(raw).__name__}")
    val = float(raw)
    if not math.isfinite(val):
        raise _spec_error(index, field, f"must be finite, got {raw!r}")
    if positive and val <= 0:
        raise _spec_error(index, field, f"must be > 0, got {val:g}")
    if non_negative and val < 0:
        raise _spec_error(index, field, f"must be >= 0, got {val:g}")
    return val


def _spec_keys(spec: dict, index: int, allowed: frozenset) -> None:
    unknown = sorted(set(spec) - allowed)
    if unknown:
        raise _spec_error(index, unknown[0],
                          f"unknown field for kind {spec.get('kind')!r}; this kind "
                          f"takes {', '.join(sorted(allowed))}")


def _chiral_object(spec: dict, index: int, name: str) -> tuple:
    """One `{"kind": "chiral", ...}` spec -> `(LithoObject, built, est_points)`.

    Built through `chiral_profile` with the SAME call `POST /api/design/chiral`
    makes, so phi = radians(phase_deg) * hand and the bias is registered against the
    modulated path's own arc length -- the whole point of the phase sweep. `axis_um`
    is set the way the HTTP door sets it, so the object can still be re-parametrized
    from its untouched base axis afterwards.
    """
    import numpy as np

    from ...litho.compiler import LithoObject
    from ...litho.pens import PENS, bake_chiral

    raw = spec.get("points_um")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise _spec_error(index, "points_um",
                          "must be a list of at least 2 [x, y] points (the base axis)")
    axis = []
    for j, pt in enumerate(raw):
        if (not isinstance(pt, (list, tuple)) or len(pt) != 2
                or any(isinstance(c, bool) or not isinstance(c, (int, float))
                       for c in pt)):
            raise _spec_error(index, "points_um",
                              f"point {j} is not a numeric [x, y] pair: {pt!r}")
        if not all(math.isfinite(float(c)) for c in pt):
            raise _spec_error(index, "points_um",
                              f"point {j} is not finite: {pt!r}")
        axis.append((float(pt[0]), float(pt[1])))

    lam = _spec_num(spec, index, "lambda_um", positive=True)
    amp = _spec_num(spec, index, "y_amp_um", non_negative=True)
    v0 = _spec_num(spec, index, "v0")
    v_k = _spec_num(spec, index, "v_k")
    phase_deg = _spec_num(spec, index, "phase_deg")
    hand = _spec_num(spec, index, "hand", default=1.0)
    if hand not in (1.0, -1.0):
        raise _spec_error(index, "hand", f"must be 1 or -1, got {hand:g}")
    speed = _spec_num(spec, index, "speed_um_s",
                      default=PENS["chiral"].params["speed_um_s"], positive=True)

    # A DEGENERATE AXIS is caught HERE, before the spline sees it, so the refusal
    # names `points_um`. Left to `chiral_profile` it arrives as a bare ValueError out
    # of the spline fit ("`x` must contain at least 2 elements", from the deduplicated
    # knot vector), which says nothing a caller can act on and would be attributed to
    # whichever field this function guessed.
    axis_arr = np.asarray(axis, dtype=float)
    axis_len = float(np.hypot(*np.diff(axis_arr, axis=0).T).sum())
    if axis_len <= 0.0:
        raise _spec_error(
            index, "points_um",
            f"all {len(axis)} points are at the same place "
            f"({axis[0][0]:g}, {axis[0][1]:g}), so the base axis has no length and "
            f"there is no wire to modulate. Give the axis two DIFFERENT endpoints.")

    # EXTENT, before anything is built. The wire spans its axis plus the modulation
    # either side of it, and the same 2000 um that bounds a design file bounds this:
    # a scanner is under 100 um, so past this the number is a units slip, not a wire.
    span_um = float(np.max(axis_arr.max(axis=0) - axis_arr.min(axis=0))) + 2.0 * amp
    if span_um > MAX_DESIGN_UM:
        raise _spec_error(
            index, "points_um",
            f"the wire spans {span_um:,.0f} um -- far larger than any scanner (the "
            f"limit is {MAX_DESIGN_UM:,.0f} um). Check the units: these are MICRONS, "
            f"and a nanometre figure typed here is 1000x too big.")

    o = LithoObject(name=name, kind="path", points_um=list(axis), voltage=v0,
                    speed_um_s=speed)
    try:
        build = bake_chiral(o, lam, amp, v_k, phase_deg, hand)
    except ValueError as exc:
        # The one ValueError the builder raises deliberately is the runaway sample
        # count, and it names the parameter actually out of scale: the amplitude when
        # the amplitude-to-pitch ratio is extreme, the axis LENGTH otherwise (a metre
        # of wire at a sane ratio is a `points_um` problem, not a `y_amp_um` one).
        # Anything else out of the spline is about the axis it was handed.
        text = str(exc)
        field = "y_amp_um" if "relative to the pitch" in text else "points_um"
        raise _spec_error(index, field, text) from exc
    if not build.arclen:                      # defence in depth on the same thing
        raise _spec_error(index, "points_um",
                          "the base axis has zero length -- nothing to modulate")

    clamp = build.clamp or {}

    # THE POINT BUDGET, chiral side. The arc length and the spacing are both known
    # exactly by now, so the compiled count is arithmetic, not a guess. It is checked
    # AFTER the build rather than before it (the region's is checked before) for one
    # reason: the runaway an ill-formed chiral actually produces is an
    # amplitude-to-pitch ratio, and `chiral_profile` refuses THAT by name with the
    # ratio in the message -- a far better diagnostic than a point count, and the
    # allocation it protects is already bounded by _CHIRAL_MAX_POINTS. What is left
    # for this check is the well-formed but enormous wire: a metre of arc at a 25 nm
    # pitch, which builds fine and would then compile into millions of points.
    est_points = int(build.arclen / o.interpolation_um) + 1
    if est_points > MAX_OBJECT_POINTS:
        raise _spec_error(
            index, "lambda_um",
            f"this wire compiles to about {est_points:,} points (the limit is "
            f"{MAX_OBJECT_POINTS:,}): {build.arclen:,.0f} um of tip travel at the "
            f"{o.interpolation_um * 1000:g} nm spacing a {lam * 1000:g} nm pitch "
            f"asks for. Shorten the axis, coarsen the pitch, or reduce the "
            f"amplitude.")

    # ECHO WHAT WAS ACTUALLY BUILT, not what was asked for. A closed axis has its
    # wavelength quantised to a whole number of periods and a curved one has its
    # amplitude clamped pointwise to the local radius; for a chirality experiment the
    # amplitude and the pitch ARE the physics, so a script has to be able to see that
    # it did not get the numbers it sent.
    prof = o.profile or {}
    fraction = float(clamp.get("fraction", 0.0))
    built = {
        "lambda_um": float(prof.get("lambda_um", lam)),
        "y_amp_um": float(prof.get("y_amp_um", amp)),
        "v0": float(v0), "v_k": float(v_k),
        "phase_deg": float(phase_deg), "hand": float(hand),
        "curvature_clamped": bool(fraction > 0.0),
        # A BOOLEAN IS NOT ENOUGH. The clamp is pointwise, so "clamped" spans
        # everything from one tight corner to a wire written at a third of the
        # amplitude that was ordered -- and the amplitude IS the measurement. So say
        # the MAGNITUDE: the smallest amplitude actually written, and how much of the
        # wire was reduced at all. Both equal the request when nothing was clamped.
        "y_amp_built_um": float(clamp.get("min_amp_um", amp)),
        "curvature_clamped_fraction": fraction,
    }
    return o, built, est_points


def _region_points(w_um: float, h_um: float, pitch_um: float) -> int:
    """How many points a filled rectangle COMPILES to, from extent/pitch arithmetic.

    An estimate, and it has to be one: the fill is resolved by `litho.fill` at compile
    time, and running it to find out is exactly the multi-minute wedge this guards.
    But it is a tight estimate, because a fill is just the region's area laid out as
    lines a pitch apart, resampled to the compiler's arc-length spacing:

        length = w*h/pitch (the fill lines) + 2*(w+h) (the outline pass)
        points = length / DEFAULT_INTERP_UM

    Measured against the real compiler (2026-09-12): exact to 1 part in 1000 for
    `directional` (1302 actual vs 1300 for the 0.6 x 0.4 um erase pad of the
    FLEX_PROVIDER example). For `contour`, `radial-*` and `hilbert` the estimate is
    NEVER LOW -- it errs high, which is the right direction for a guard -- but "high"
    is not the 2-8% first measured: a wider sweep (adversarial review, 2026-09-12)
    found it up to ~55% high for `hilbert`, whose recursive path is considerably
    shorter than the serpentine this formula models. Still nothing against a limit
    set two orders of magnitude above any real region, but say the real number rather
    than the optimistic first measurement.
    """
    from ...litho.compiler import DEFAULT_INTERP_UM
    length_um = w_um * h_um / pitch_um + 2.0 * (w_um + h_um)
    return int(length_um / DEFAULT_INTERP_UM) + 1


def _region_object(spec: dict, index: int, name: str) -> tuple:
    """One `{"kind": "region", "shape": "rect", ...}` spec ->
    `(LithoObject, {}, est_points)`.

    Built through the REGION PEN (`render_stroke`), so the object that lands in the
    pattern is the polygon plus its fill parameters -- exactly what the cockpit's
    region tool produces -- and `compiler.expand` resolves it into a toolpath at
    compile time through `litho.fill`. The bias is passed STRAIGHT THROUGH, signed:
    an erase is a region at a negative voltage, and coercing the sign here (as the
    `erase` pen does, with `-abs(voltage)`) would make a positive-bias region
    unorderable and a sign typo invisible.

    EVERY SIZE CHECK HERE RUNS BEFORE `render_stroke`, on arithmetic alone. That is
    the point of them: the object the pen returns is only the polygon, so a 3000 um
    rectangle renders instantly and then wedges the COMPILE for ten minutes -- long
    past the verb's 2 s bound, with the caller already gone. Refusing on the estimate
    costs microseconds.
    """
    from ...litho.pens import PENS, render_stroke

    shape = spec.get("shape")
    if shape != "rect":
        raise _spec_error(index, "shape",
                          f"must be \"rect\" (the only shape this door builds), "
                          f"got {shape!r}")
    x = _spec_num(spec, index, "x_um")
    y = _spec_num(spec, index, "y_um")
    w = _spec_num(spec, index, "w_um", positive=True)
    h = _spec_num(spec, index, "h_um", positive=True)
    pitch = _spec_num(spec, index, "pitch_um", positive=True)
    voltage = _spec_num(spec, index, "voltage")
    speed = _spec_num(spec, index, "speed_um_s",
                      default=PENS["region"].params["speed_um_s"], positive=True)
    fills = [c[0] for c in PENS["region"].choices["fill"]]
    fill = spec.get("fill", PENS["region"].params["fill"])
    if fill not in fills:
        raise _spec_error(index, "fill",
                          f"must be one of {', '.join(fills)}, got {fill!r}")
    # `angle_deg` is the direction the serpentine sweeps in, and ONLY `directional`
    # reads it -- every other strategy has its own geometry (a contour follows the
    # outline, a Hilbert curve has no preferred direction at all, which is the reason
    # to pick it). Sent with one of those it would be accepted and ignored, which is
    # the bug `_spec_keys` exists to prevent, so it is refused by name instead.
    angle = None
    if spec.get("angle_deg") is not None:
        if fill != "directional":
            raise _spec_error(
                index, "angle_deg",
                f"only the \"directional\" fill sweeps at a chosen angle; {fill!r} "
                f"has its own geometry and would ignore it. Drop `angle_deg`, or "
                f"set `fill` to \"directional\".")
        angle = _spec_num(spec, index, "angle_deg")

    # EXTENT, the same 2000 um bound `_svg_pattern` puts on a design file. `objects`
    # bypassed it entirely: a 400 x 400 um rect was accepted (16 M points, 89 hours).
    extent = max(w, h)
    if extent > MAX_DESIGN_UM:
        raise _spec_error(
            index, "w_um" if w >= h else "h_um",
            f"the rectangle is {w:,.0f} x {h:,.0f} um -- far larger than any scanner "
            f"(the limit is {MAX_DESIGN_UM:,.0f} um on a side). Check the units: "
            f"these are MICRONS, and a nanometre figure typed here is 1000x too big.")

    # PITCH SANITY. `pitch_um` is the separation between fill lines, so a pitch wider
    # than the rectangle leaves the fill as the outline plus a stripe or two down the
    # middle -- an "erase" that does not erase, and it looks like it worked. Measured:
    # a 0.4 um rect at `pitch_um: 10` compiled to an outline and ONE centre line.
    # Half the short side is the loosest rule that still guarantees more than one
    # interior line; a real erase wants the pitch at or under the written linewidth.
    if pitch > min(w, h) / 2.0:
        raise _spec_error(
            index, "pitch_um",
            f"a {pitch:g} um fill pitch is wider than half the {w:g} x {h:g} um "
            f"rectangle, so the fill would be the outline and at most one line down "
            f"the middle -- an erase that does not erase. Use a pitch at or under "
            f"the written linewidth (and never more than {min(w, h) / 2.0:g} um "
            f"here).")

    # THE POINT BUDGET, from extent/pitch arithmetic and BEFORE any rendering.
    est_points = _region_points(w, h, pitch)
    if est_points > MAX_OBJECT_POINTS:
        hours = est_points * 0.02 / speed / 3600.0
        raise _spec_error(
            index, "pitch_um",
            f"a {w:g} x {h:g} um rectangle at a {pitch:g} um pitch compiles to about "
            f"{est_points:,} points (the limit is {MAX_OBJECT_POINTS:,}) -- roughly "
            f"{hours:,.0f} hours of writing at {speed:g} um/s. Use a coarser pitch "
            f"or a smaller rectangle.")

    # `x_um, y_um` is the rectangle's CORNER and `w_um, h_um` its extent, so the
    # four corners are unambiguous; `shape: "rect"` then makes the pen take the
    # bounding box of exactly those points, which is the rectangle itself.
    corners = [[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]]
    params = {"voltage": voltage, "speed_um_s": speed, "pitch_um": pitch,
              "fill": fill, "shape": "rect"}
    if angle is not None:
        params["angle_deg"] = angle
    objs = render_stroke(corners, "region", params, mode="literal", name=name)
    if not objs:                              # AREA_EPS -- `w`/`h` are already > 0
        raise _spec_error(index, "w_um",
                          f"a {w:g} x {h:g} um rectangle has no area to fill")
    return objs[0], {}, est_points


def _objects_pattern(specs, deadline: float | None = None):
    """`objects: [ObjectSpec]` -> `(Pattern, {name: {"kind", "profile"}})`.

    The metadata comes back beside the pattern rather than on the objects because it
    describes what the SPEC ordered, which is not always what a `LithoObject` field
    can hold: a chiral wire is a `path` to everything downstream, and the fact that a
    script asked for it by parameters is this door's knowledge, not the compiler's.

    `deadline` (a `time.monotonic()` instant) is OPTIONAL and, when given, is checked
    at the top of every spec -- the one loop this function has that a multi-object
    request actually walks. This is the COOPERATIVE half of the wall clock in
    `_within`/`_submit_guarded`: that clock's own timeout already answers the caller
    with `-32021 build_timeout` the instant the budget runs out, but the thread
    itself (which cannot be killed -- see `_submit_guarded`) would otherwise keep
    building every remaining object in the list first. Checking here stops it after
    the object it is on, not the whole request -- cheap, because it is a
    `time.monotonic()` read, and it does NOT reach into `chiral_profile` or
    `render_stroke` (no pen is restructured for this): a single request holding one
    enormous spec is still bounded only by the outer wall clock, which is the
    backstop it was already relying on.
    """
    if not isinstance(specs, (list, tuple)) or not specs:
        msg = "loadPattern: objects must be a non-empty list of object specs"
        raise CommandError(INVALID_PARAMS, msg, {"param": "objects"},
                           http={"error": msg})
    from ...litho.compiler import Pattern
    objs, meta = [], {}
    total_points = 0
    for i, spec in enumerate(specs):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(
                f"objects build abandoned at object {i} of {len(specs)} -- the "
                f"verb's deadline has already passed")
        if not isinstance(spec, dict):
            msg = (f"loadPattern: objects[{i}] must be an object spec, got "
                   f"{type(spec).__name__}")
            raise CommandError(INVALID_PARAMS, msg,
                               {"param": "objects", "index": i},
                               http={"error": msg})
        kind = spec.get("kind")
        if kind not in OBJECT_KINDS:
            raise _spec_error(i, "kind",
                              f"must be one of {', '.join(OBJECT_KINDS)}, "
                              f"got {kind!r}")
        _spec_keys(spec, i, _CHIRAL_KEYS if kind == "chiral" else _REGION_KEYS)
        name = spec.get("name", None)
        if name is None:
            name = f"{kind}-{i + 1}"
        elif not isinstance(name, str) or not name.strip():
            raise _spec_error(i, "name", f"must be a non-empty string, got {name!r}")
        # The compile report is keyed BY NAME, so two objects sharing one would be
        # reported (and their point counts summed) as a single object. Refuse.
        if name in meta:
            raise _spec_error(i, "name",
                              f"duplicate name {name!r} -- object names must be "
                              f"unique within a pattern")
        build = _chiral_object if kind == "chiral" else _region_object
        o, profile, est_points = build(spec, i, name)
        # THE TOTAL BUDGET. Every object can be inside MAX_OBJECT_POINTS and the
        # pattern still be one nobody will ever write: fifty legal regions is fifty
        # legal regions' worth of compile, on a verb bounded at 2 s. Checked as the
        # specs are built, so a long list is refused at the object that crosses the
        # line rather than after all of them have been rendered.
        total_points += est_points
        if total_points > MAX_TOTAL_POINTS:
            raise _spec_error(
                i, "kind",
                f"this object takes the pattern to about {total_points:,} compiled "
                f"points, over the {MAX_TOTAL_POINTS:,}-point limit for one load. "
                f"Split the job across several loadPattern calls, or coarsen the "
                f"pitch.")
        objs.append(o)
        meta[name] = {"kind": kind, "profile": profile or None}
    return Pattern(name="objects", objects=objs), meta


def _compile_report(hub, pattern) -> dict:
    """Compile the pattern and MEASURE it, without arming anything.

    Returns per-object compiled point counts and durations plus the lab-frame bounding
    box. This is the whole of `loadPattern`'s work: a script gets to refuse on a number
    rather than discover a limit by tripping it (plan Sec A), and `startWrite` reads
    the durations back for its `est_s` instead of re-compiling on the start path, which
    has a sub-second bound to keep.
    """
    import numpy as np

    from ...litho.compiler import compile_pattern
    from ..host.litho_worker import _litho_frame
    w = hub.host.world
    waves = compile_pattern(pattern, w.x_lvdt_sens, w.y_lvdt_sens,
                            w.x_lvdt_offset, w.y_lvdt_offset,
                            frame=_litho_frame(hub))
    points: dict[str, int] = {}
    duration: dict[str, float] = {}
    # THREE different counts, because three different things are being asked:
    #
    #   strokes  -- continuous tip paths: one transit, one `litho_object_done`, one
    #               `getWritten` entry each. A filled shape expands into many; a wire
    #               is one however long it is. The pump counts completions against
    #               THIS (see `_marks`).
    #   chunks   -- out-waves actually ARMED. A stroke over the DSP's per-out-wave cap
    #               is fired as a seamless CHAIN of chunks: more arms, but the tip
    #               stays down and the bias stays live across every boundary.
    #   chained  -- how many chain boundaries the object contains (chunks - strokes),
    #               the only seams `est_s` charges for inside a stroke.
    strokes: dict[str, int] = {}
    segments: dict[str, int] = {}
    # What the machine SETTLES ON, per object. The panel's speed field is a ceiling
    # and the tick interpolation factor is an integer, so the achieved speed is
    # quantised down from the request (`compiler.time_segment`) -- sometimes by a lot
    # on a finely interpolated path. The slowest segment is the one that governs, so
    # that is the one reported.
    achieved: dict[str, float] = {}
    lo = np.array([np.inf, np.inf])
    hi = -lo
    total = 0
    for meta in waves["segments"]:
        name = str(meta["object"])
        n = len(meta["path_um"])
        points[name] = points.get(name, 0) + n
        duration[name] = duration.get(name, 0.0) + float(meta["duration_s"])
        strokes[name] = strokes.get(name, 0) + 1
        segments[name] = segments.get(name, 0) + _chunks_for(n)
        v_got = float(meta.get("speed_achieved_um_s") or 0.0)
        if v_got > 0:
            achieved[name] = min(achieved.get(name, v_got), v_got)
        total += n
        if n:
            p = np.asarray(meta["path_um"], dtype=float)
            lo = np.minimum(lo, p.min(axis=0))
            hi = np.maximum(hi, p.max(axis=0))
    bbox = ([round(float(lo[0]), 4), round(float(lo[1]), 4),
             round(float(hi[0]), 4), round(float(hi[1]), 4)]
            if np.isfinite(lo[0]) else None)
    return {"points": points, "duration": duration, "segments": segments,
            "strokes": strokes,
            "speed_achieved": achieved,
            # what was ASKED for, beside it, so `object_entries` can say whether the
            # quantisation actually moved the number
            "speed_requested": {str(o.name): float(o.speed_um_s)
                                for o in pattern.objects},
            # The COMPILER's own word for each object (`path` / `shape` /
            # `electrode`) -- what `getPattern` has always reported. It is the
            # fallback `object_entries` uses for an object that came from a design
            # file and therefore has no `objects` spec kind to report instead.
            "kinds": {str(o.name): str(o.kind) for o in pattern.objects},
            "chained": {k: segments[k] - strokes[k] for k in segments},
            "total_points": total,
            "segments_total": sum(segments.values()),
            "strokes_total": len(waves["segments"]),
            "order": [name for name in points],
            "bbox_um": bbox,
            "duration_s": round(float(waves["duration_s"]), 3)}


def _chunks_for(n_points: int) -> int:
    """How many OUT-WAVES one stroke of `n_points` is armed as.

    1 at or under CHAIN_CHUNK_POINTS -- the overwhelmingly common case, and the whole
    point of the 2026-09-11 fix: a stroke is one arm, tip down start to finish. Above
    it, the driver's own `chunk_boundaries` decides, so this asks that function rather
    than re-deriving a ceil() that could drift from it. The threshold is the CHUNK
    size, not the arm cap, for the same reason `write_object`'s is -- see there.
    """
    n = int(n_points)
    if not is_chained(n):
        return 1
    import numpy as np
    return len(chunk_boundaries(n, np.zeros(n, dtype=bool))) - 1


def multi_segment_objects(report: dict) -> list[dict]:
    """The objects that CHAIN, BY NAME: those containing a stroke over the DSP's
    per-out-wave cap, so the driver has to fire it as a continuation of chunks.

    Informational, not an error (decision of 2026-09-11 -- this replaced the
    `-32602 objects_over_cap` refusal). NARROWED 2026-09-11 (same day, second pass):
    it used to list every object of more than one fired segment, which meant every
    object over 8,192 points, because the compiler cut every stroke there and each
    cut was a full lift / move / set-down in the middle of a line. That cut is gone.
    What is left is genuinely different in kind:

      * a chain boundary is a DWELL -- tip down, bias live, same index either side
        (Igor's `AFM_LithoAdvance` zeroes the bias only at the object's true end);
      * the disconnected strokes of a filled shape are separated by real geometry, so
        the lift between them is correct, not an artifact, and is not reported here.

    So an entry means "this object is big enough that the controller cannot hold it
    in one out-wave", which is the one thing a script still cannot see for itself.
    """
    return [{"name": name, "points": report["points"][name],
             "segments": report["segments"][name],
             "strokes": report["strokes"][name],
             "chain_boundaries": n,
             "max_outwave_points": MAX_OUTWAVE_POINTS}
            for name, n in (report.get("chained") or {}).items() if n > 0]


def _chiral_view(obj) -> dict:
    """The five chiral parameters, read back off the stored profile the SAME way
    `litho.pens.chiral_settings` reads them: `hand` is folded into `phase_rad`
    (phi = radians(phase_deg) * hand), so the canonical readback is
    `phase_deg = degrees(phase_rad)` with `hand = 1`. `lambda_um` is the pitch asked
    for; `lambda_built_um` is the one a closed spine quantised it to, read-only (a
    script may send it back in a `chiral` block, where it is ignored). `v0` is NOT here -- it IS the
    object's `voltage`, which the properties block already carries."""
    prof = obj.profile or {}
    amp = prof.get("y_amp_um")
    if amp is None:
        amp = max(prof.get("amp_um") or [0.0])
    lam = prof.get("lambda_set_um")
    if lam is None:
        lam = prof["lambda_um"]
    return {"lambda_um": float(lam),
            "lambda_built_um": float(prof["lambda_um"]),
            "y_amp_um": float(amp),
            "v_k": float(prof["v_k"]),
            "phase_deg": math.degrees(float(prof["phase_rad"])),
            "hand": 1.0}


def _fill_view(obj) -> dict:
    """The region fill parameters: pitch, strategy and (for `directional`) the sweep
    angle. `angle_deg` is None for every strategy but `directional`, which the region
    pen also leaves None because those strategies have their own geometry."""
    angle = getattr(obj, "fill_angle_deg", None)
    return {"spacing_um": float(getattr(obj, "fill_spacing_um", 0.0) or 0.0),
            "kind": getattr(obj, "fill_kind", None),
            "angle_deg": (float(angle) if angle is not None else None)}


def _object_view(obj) -> dict:
    """The `id`/`vertices_um`/`properties` block shared by every per-object listing
    (`loadPattern` both doors, `getPattern`, `setObjects`), so the three cannot drift.

    `vertices_um` is the SOURCE polyline the object was built from: `points_um` for a
    path/rect/taper/shape, but for a CHIRAL path the ORIGINAL axis (`axis_um`), not the
    baked sinusoid -- that axis is what `setObjects` re-modulates and what a script has
    to edit. `properties.chiral` is present (possibly null) only for a `path`; `fill`
    only for a fillable region -- the two things `setObjects` can turn on and off."""
    kind = getattr(obj, "kind", None)
    axis = getattr(obj, "axis_um", None)
    is_chiral = (getattr(obj, "profile", None) or {}).get("kind") == "chiral"
    src = axis if (is_chiral and axis) else (getattr(obj, "points_um", None) or [])
    verts = [[float(x), float(y)] for x, y in src]
    voltage = getattr(obj, "voltage", None)
    props = {"voltage": float(voltage) if voltage is not None else 0.0,
             "speed_um_s": float(getattr(obj, "speed_um_s", 0.0) or 0.0),
             "enabled": bool(getattr(obj, "enabled", True))}
    if kind == "path":
        props["chiral"] = _chiral_view(obj) if is_chiral else None
    elif kind in ("rect", "taper", "shape"):
        props["fill"] = _fill_view(obj)
    return {"id": getattr(obj, "name", None), "vertices_um": verts,
            "properties": props}


def object_entries(report: dict, pattern=None, meta: dict | None = None) -> list[dict]:
    """One entry per object, in PATTERN order:
    `{index, name, id, kind, points, segments, strokes, vertices_um, properties}`,
    plus `profile` for a chiral one.

    `points` is the COMPILED point count (a filled shape is far more than its design
    polyline), `strokes` is how many continuous tip paths it is (one transit and one
    `getWritten` entry each) and `segments` is how many out-waves are actually armed
    -- the same number unless a stroke is over the cap and chains.

    WALKS `pattern.objects`, NOT the compile report's `order`, so a DISABLED object --
    which the compiler skips, so it never appears in the report -- is still listed,
    with its `enabled: false` visible and a fallback count from its design polyline.
    That is what lets a script see a disabled wire in the listing and re-enable it by
    id. When no pattern is given the function falls back to the report order.

    `id` (== `name`), `vertices_um` (the source polyline -- the base axis for a chiral
    wire) and `properties` (voltage, speed, enabled, and the chiral/fill block) come
    from `_object_view`, which `getPattern` and `setObjects` share so the listing shape
    is identical whichever verb produced it.

    `kind` is what the object IS, and the two doors answer it from different places
    on purpose: an object ordered through `objects` reports the SPEC kind it was
    ordered by (`chiral`, `region`), because that is the thing a script named and the
    thing it wants to read back; an object that came out of a design FILE has no spec,
    so it reports the compiler's own kind (`path`, `shape`, `electrode`) -- the same
    word `getPattern` has always used for it.
    """
    meta = meta or {}
    objects = getattr(pattern, "objects", None)
    names = ([getattr(o, "name", None) for o in objects] if objects is not None
             else list(report["order"]))
    by_name = {getattr(o, "name", None): o for o in (objects or [])}
    points = report.get("points") or {}
    segments = report.get("segments") or {}
    strokes = report.get("strokes") or {}
    out = []
    for i, name in enumerate(names):
        obj = by_name.get(name)
        # A disabled object is never compiled, so it is absent from the report; fall
        # back to its design polyline the way `getPattern` does for a pattern that
        # never went through `loadPattern`.
        n_design = len(getattr(obj, "points_um", None) or []) if obj is not None else 0
        entry = {"index": i, "name": name,
                 "points": int(points.get(name, n_design)),
                 "segments": int(segments.get(name, _chunks_for(n_design))),
                 "strokes": int(strokes.get(name, 1 if n_design else 0)),
                 "kind": (meta.get(name) or {}).get("kind")
                         or (report.get("kinds") or {}).get(name)
                         or (getattr(obj, "kind", None) if obj is not None else None)}
        if obj is not None:
            entry.update(_object_view(obj))
        profile = (meta.get(name) or {}).get("profile")
        if profile:
            entry["profile"] = profile
        # `speed_achieved_um_s` ONLY WHEN IT IS NEWS. The instrument's tick
        # interpolation factor is an integer, so the requested speed is a ceiling the
        # machine quantises DOWN from -- and on a chiral wire, whose interpolation
        # spacing is a fraction of the pitch, it can land well under what was asked.
        # A script sweeping dose has to know the real speed; a script that got the
        # speed it asked for does not need a key telling it so.
        want = (report.get("speed_requested") or {}).get(name)
        got = (report.get("speed_achieved") or {}).get(name)
        if want and got and abs(got - want) > 0.01 * want:
            entry["speed_achieved_um_s"] = round(got, 6)
        out.append(entry)
    return out


def _objects_bound_s() -> float:
    """How long the `objects` door gets, total, to build AND compile.

    THE VERB'S OWN PUBLISHED BOUND, read from the FLEX registry rather than written
    down a second time here: that is the number `method_bounds` advertises and the
    number the FLEX driver derives its client timeout from, so the two cannot drift
    (plan Sec F.2 -- the driver must not hardcode numbers that will drift, and neither
    must the service). Past it the caller has already given up; holding a thread on a
    result nobody will receive is the failure mode, not the protection against it.

    Imported inside the function because `flex_dispatch` imports THIS module (lazily,
    inside its handlers) -- at module scope the two would be a cycle. A default is
    kept for the cockpit's own HTTP door, which can reach `load_pattern` without the
    wire registry ever having been imported.
    """
    try:
        from ..flex_dispatch import REGISTRY
        return float(REGISTRY["loadPattern"].bound_s)
    except (ImportError, KeyError, AttributeError):   # no wire registry here
        return 2.0


# ============================================ the objects-door build executor ===
#
# WHY THIS EXISTS (adversarial review of 7bd011d, measured 2026-09-12). `_within`
# used to run the `objects` door's build+compile through bare `asyncio.to_thread`,
# i.e. the process-wide DEFAULT `ThreadPoolExecutor` -- the SAME pool every
# hardware-safety `to_thread` call uses (`withdraw`, `approach`, the free-air and SUM
# reads, `scan_cmds`/`host/continuous`). On a timeout the worker thread cannot be
# killed (Python cannot interrupt a running thread) and keeps burning CPU in the
# background; because the socket now replies promptly instead of hanging, a script
# issuing repeated slow `loadPattern` calls accumulates one orphaned thread per
# timed-out call until the default pool's fixed slots (20 here) are exhausted, and
# then `withdraw`/`approach`/the free-air reads QUEUE BEHIND THEM. Measured: an
# unrelated `to_thread` call delayed 2.8 s in a 3 s-timeout / 20-slot reproduction.
# Before the wall clock existed (pre-7bd011d) this could not accumulate -- there was
# no timeout to abandon a thread behind.
#
# THE FIX IS TWO PARTS.
#
#   1. A DEDICATED single-worker executor, so an orphan can only ever compete with
#      ITSELF -- never with a hardware-safety read on the default pool. `max_workers
#      = 1` is deliberate: the objects door has never needed concurrency (the wire is
#      serialised behind one REQ/REP command socket, plan Sec E.2), and a single
#      worker turns "how many orphans can pile up" into a yes/no question the next
#      call answers exactly.
#   2. TRACK THE ORPHAN, because a single worker becomes its own hazard otherwise: a
#      second `loadPattern {objects}` arriving while the first is still running would
#      queue SILENTLY behind it on this executor, and the caller would wait out
#      whatever is left of the orphan's runtime with zero feedback -- worse than the
#      shared-pool starvation this replaces, just narrower in scope. So the one
#      in-flight submission is tracked and a new one is refused immediately
#      (`-32020 busy`, `data.reason = "build_in_progress"`, `data.age_s`) rather than
#      queued. `source` loads never reach `_within`/this executor at all and are
#      unaffected.
#
# The orphan's own eventual result -- return value or exception -- is always
# discarded. The coroutine that submitted it already gave up and raised
# `-32021 build_timeout` to ITS caller, and `load_pattern` does not assign
# `hub.pattern` until AFTER its own `await` returns successfully, which for a
# timed-out call never happens. `_clear_orphan` below only clears the
# orphan-tracking bookkeeping; it never touches `hub` or `hub.pattern`.
#
# LAZY AND RECREATABLE, not built once at import time and kept forever: the
# service's `app.py` shuts this executor down alongside the FLEX provider on the
# app's `shutdown` event (see `shutdown_objects_executor`), and that event can fire
# more than once per PROCESS even though it fires once per real service LIFETIME --
# every test module that starts its own app/provider around the shared `hub`
# singleton triggers it again at that module's teardown. `_submit_guarded` recreates
# it on demand rather than assuming the one built at import time (or the previous
# app lifecycle's) is still alive.
#
# NOT A `ThreadPoolExecutor` (re-review of 450bebb, measured 2026-09-12). It was
# one when this section first shipped, and that was itself a bug: `ThreadPoolExecutor`
# workers are NOT daemon threads -- CPython registers every one in
# `concurrent.futures.thread._threads_queues`, and the interpreter's own `atexit`
# hook joins them UNCONDITIONALLY, regardless of what `shutdown(wait=False,
# cancel_futures=True)` asked for. So an orphan (see above -- it cannot be killed,
# only abandoned) still held up PROCESS EXIT despite `wait=False`: measured, a 3 s
# orphaned build made the process take 3.1 s to exit. `_DaemonSingleWorker` below is
# a minimal replacement with the same `submit`/`shutdown` call shape, built on a
# `threading.Thread(daemon=True)`, so an orphan left running past `shutdown()` -- or
# past interpreter exit -- is simply DETACHED: nothing waits on it, it never touches
# `hub`/`hub.pattern` (see `_clear_orphan` below), and it is bounded in practice by
# the size policy in `load_pattern` and by `_CHIRAL_MAX_POINTS` (`litho/pens.py`,
# ~2 s to build one chiral spec at that ceiling) -- NOT by the per-spec deadline
# check in `_objects_pattern`, which only stops a MULTI-spec request after the spec
# it is currently on, not mid-spec.
_orphan_lock = threading.Lock()
_orphan_future: concurrent.futures.Future | None = None
_orphan_since: float | None = None


class _DaemonSingleWorker:
    """A one-worker task queue with the same `submit`/`shutdown` shape as
    `concurrent.futures.ThreadPoolExecutor(max_workers=1)`, but built on a daemon
    thread so an abandoned (orphaned) job never delays interpreter exit -- see the
    section banner above for why that matters and what was measured.

    `submit` returns a real `concurrent.futures.Future`: `_within` awaits it through
    `asyncio.wrap_future`, which requires exactly that type, and `_clear_orphan` /
    `_submit_guarded` only ever call `.done()` / `.add_done_callback()` on it, both
    unchanged from stdlib `Future`. The worker thread resolves each Future itself
    (`set_result`/`set_exception`) rather than raising, so a failing `fn` cannot kill
    the thread -- one dead worker would silently stop the objects door forever.

    `submit` after `shutdown()` raises `RuntimeError("objects executor is shut down")`,
    matching stdlib `Executor.submit` -- it never silently enqueues behind a worker
    thread that has already been told to stop, whose sentinel would leave that Future
    unresolved forever.
    """

    def __init__(self, thread_name_prefix: str = "litho-build") -> None:
        self._queue: _queue.SimpleQueue = _queue.SimpleQueue()
        self._shutdown_lock = threading.Lock()
        self._is_shutdown = False
        self._thread = threading.Thread(
            target=self._run, name=thread_name_prefix, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:                       # shutdown()'s sentinel
                return
            fn, args, fut = item
            if not fut.set_running_or_notify_cancel():
                continue                           # cancelled before it ran
            try:
                result = fn(*args)
            except BaseException as exc:           # noqa: BLE001 -- resolved onto
                fut.set_exception(exc)             # the Future, not raised here
            else:                                   # (mirrors stdlib ThreadPoolExecutor's
                fut.set_result(result)              # own worker, which does the same)

    def submit(self, fn, *args) -> concurrent.futures.Future:
        with self._shutdown_lock:
            if self._is_shutdown:
                raise RuntimeError("objects executor is shut down")
            fut: concurrent.futures.Future = concurrent.futures.Future()
            self._queue.put((fn, args, fut))
        return fut

    def shutdown(self, wait: bool = False, cancel_futures: bool = False) -> None:
        """Same signature as `ThreadPoolExecutor.shutdown` (the two kwargs
        `shutdown_objects_executor` passes) so that call site did not need to
        change. Marks the worker shut down first, under the same lock `submit` uses,
        so no job can be enqueued after this call starts (see `submit` above).

        `cancel_futures=True` actually cancels: any job still sitting in the queue
        -- submitted but not yet picked up by the worker thread -- is drained and its
        Future is `.cancel()`-ed instead of being left to run after all. The job the
        worker is CURRENTLY running (already popped off the queue before this call)
        is not affected -- it cannot be interrupted -- and finishes normally; only
        work that has not started yet is cancellable. The sentinel put after the
        drain tells the worker to stop once that current job (if any) completes;
        since the thread is a daemon, nothing waits on it at interpreter exit
        regardless of `wait`."""
        with self._shutdown_lock:
            self._is_shutdown = True
            if cancel_futures:
                while True:
                    try:
                        item = self._queue.get_nowait()
                    except _queue.Empty:
                        break
                    if item is None:
                        continue                    # a stray earlier sentinel
                    _fn, _args, fut = item
                    fut.cancel()
            self._queue.put(None)
        if wait:
            self._thread.join()


_OBJECTS_EXECUTOR: _DaemonSingleWorker | None = None


def _clear_orphan(future: concurrent.futures.Future) -> None:
    """Done-callback on every submission: once it actually finishes (success,
    exception, or it was never the tracked one), the door reopens. Runs in whatever
    thread completes the future -- the dedicated worker, ordinarily -- so it is only
    a lock and two assignments: nothing here can block or raise."""
    global _orphan_future, _orphan_since
    with _orphan_lock:
        if _orphan_future is future:
            _orphan_future = None
            _orphan_since = None


def _submit_guarded(fn, *args) -> concurrent.futures.Future:
    """Submit `fn(*args)` to the dedicated objects-door executor, or raise
    `-32020 busy` (`reason: "build_in_progress"`) if a previous submission (build OR
    compile -- both route through here) timed out on the caller's side and is still
    running. See the section banner above for why a single worker makes this
    necessary rather than merely convenient."""
    global _orphan_future, _orphan_since, _OBJECTS_EXECUTOR
    with _orphan_lock:
        if _orphan_future is not None and _orphan_future.done():
            _orphan_future = None
            _orphan_since = None
        if _orphan_future is not None:
            age = time.monotonic() - _orphan_since
            msg = (f"loadPattern: a previous objects build/compile timed out "
                   f"{age:.1f} s ago and its worker is still running on the objects "
                   f"door's single dedicated thread -- a new one would only queue "
                   f"behind it, silently, for however long it has left. Wait for it "
                   f"to finish (it cannot be cancelled) and retry, or order a "
                   f"smaller/coarser job.")
            raise CommandError(BUSY, msg,
                               {"param": "objects", "reason": "build_in_progress",
                                "age_s": round(age, 1)},
                               http={"error": msg, "reason": "build_in_progress"})
        if _OBJECTS_EXECUTOR is None:          # first use, or recreated post-shutdown
            _OBJECTS_EXECUTOR = _DaemonSingleWorker(thread_name_prefix="litho-build")
        future = _OBJECTS_EXECUTOR.submit(fn, *args)
        future.add_done_callback(_clear_orphan)
        _orphan_future = future
        _orphan_since = time.monotonic()
    return future


def shutdown_objects_executor(*, wait: bool = False) -> None:
    """Called from the app's `shutdown` handler (`service/app.py`). `wait=False` by
    default: an orphan thread cannot be interrupted (see above) regardless, and
    `_DaemonSingleWorker`'s worker is a daemon thread, so leaving it running past
    this call never delays interpreter exit -- unlike the `ThreadPoolExecutor` this
    replaced, whose non-daemon workers the interpreter's `atexit` hook joined
    unconditionally (measured: `wait=False` bought nothing there either, a 3 s
    orphan still cost the process 3.1 s at exit). `wait=False` here is only about
    not blocking THIS call -- e.g. a test module's teardown, which runs this many
    times in one process -- not about process exit, which the daemon thread already
    settles on its own.

    DETACHES rather than merely shutting down: `_OBJECTS_EXECUTOR` is set back to
    `None` under the same lock `_submit_guarded` uses, so a LATER call (the next
    app lifecycle in the same process -- see the section banner above) lazily builds
    a fresh `_DaemonSingleWorker` instead of submitting to one already told to stop.
    The actual `shutdown()` call happens outside the lock, since it may (with
    `wait=True`) block, and nothing else needs to wait for it."""
    global _OBJECTS_EXECUTOR
    with _orphan_lock:
        executor, _OBJECTS_EXECUTOR = _OBJECTS_EXECUTOR, None
    if executor is not None:
        executor.shutdown(wait=wait, cancel_futures=True)


async def _within(coro, deadline: float, phase: str):
    """Await `coro`, but not past `deadline`; on timeout, REFUSE.

    `-32021 refused`, not `-32602`: nothing about the REQUEST was found to be invalid
    -- it was accepted, and the service ran out of its own budget. Retrying it
    unchanged will not help either, which is what separates this from `busy`, so
    `reason` says which half ran long and the message says what to change.

    The worker thread cannot be killed and keeps running to completion in the
    background; it is discarded. That is acceptable precisely BECAUSE the arithmetic
    guards above run first: by the time anything reaches here it has already been
    bounded in extent and in points, so the orphan is a slow compile, not a runaway
    allocation. The loaded pattern is untouched -- `hub.pattern` is not assigned until
    after the compile returns -- so a timeout leaves the previous design in place, the
    same rule every other refusal on this verb follows.

    `coro` is expected to be `asyncio.wrap_future(_submit_guarded(...))`, NOT a bare
    `asyncio.to_thread(...)` -- see the section above this function. That is what
    keeps the orphan off the shared default pool, and `_submit_guarded` is what
    refuses a second call while the orphan is still running rather than queuing
    behind it.
    """
    budget = deadline - time.monotonic()
    try:
        return await asyncio.wait_for(coro, max(budget, 0.001))
    except TimeoutError as exc:               # asyncio.TimeoutError is this, 3.11+
        msg = (f"loadPattern: the {phase} did not finish within the verb's "
               f"{_objects_bound_s():g} s bound and was abandoned; the previously "
               f"loaded pattern is unchanged. Order fewer or smaller objects (a "
               f"coarser pitch is the biggest single lever), or split the job across "
               f"several calls.")
        raise CommandError(REFUSED, msg,
                           {"param": "objects", "reason": f"{phase}_timeout"},
                           http={"error": msg}) from exc


async def load_pattern(hub, source=None, *, objects=None,
                       allow_paths: bool = False) -> dict:
    """Load a design, compile it, and report what it would cost. ARMS NOTHING.

    TWO DOORS INTO ONE PATTERN, and exactly one of them per call:

      * `source` -- GEOMETRY: `"auto"` (route to the detected contacts), a design
        name inside `data/designs`, a file path (wire door only -- see
        `_design_path`), or an inline SVG document.
      * `objects` -- PARAMETERS: a list of `{"kind": "chiral"|"region", ...}` specs
        (see `_objects_pattern`). SVG, OASIS and GDS carry no tip voltage, so the
        experiments that ARE a voltage -- a chiral wire's phase, an erase
        rectangle's negative bias -- cannot be ordered through `source` at all.

    Everything after the build is shared: the same compile, the same report, the same
    chunk accounting and the same refusal rules, so the two doors cannot drift.
    The compile IS the validation: a design that cannot be compiled is refused here,
    with the tip untouched, rather than at the first ARMFIRE.

    NO SIZE REFUSAL ON THE OUT-WAVE CAP (decision of 2026-09-11, superseding plan
    Sec A's "reject any object over the cap by name"). The cap is on ONE OUT-WAVE, not
    on an object: an object past it is fired as a seamless CHAIN of chunks, tip down
    and bias live across every boundary. This verb reports that -- `segments`
    (out-waves armed, per object and in total), `strokes` (continuous tip paths),
    `multi_segment_objects` (the ones that chain) and a WARNING naming them -- instead
    of refusing the load. Genuinely invalid input (unreadable, uncompilable, no
    drawable geometry, absurdly large) is still refused; for the `objects` door
    "absurdly large" is spelled out in the SIZE POLICY above (`MAX_OBJECT_POINTS`),
    and is a different question from the cap.
    """
    from ...litho.designs import auto_device
    meta: dict = {}
    deadline = time.monotonic() + _objects_bound_s()
    src = "auto" if source is None else str(source)
    if objects is not None:
        # EXACTLY ONE DOOR PER CALL. Both would be ambiguous about which pattern the
        # next `startWrite` acts on, and silently preferring one would make a script
        # that sent both write something it did not read back.
        if source is not None:
            msg = ("loadPattern: pass exactly one of `source` (geometry) or "
                   "`objects` (parameters), not both")
            raise CommandError(INVALID_PARAMS, msg, {"param": "objects"},
                               http={"error": msg})
        # Building runs on a thread for the same reason the compile does: a chiral
        # wire samples its spline at tens of thousands of points, and the command
        # socket is serialised behind `abort`.
        #
        # AND UNDER A WALL CLOCK, on the DEDICATED objects-door executor rather than
        # the shared default pool (see the section banner above `_within`) -- an
        # orphaned build must never compete with a hardware-safety `to_thread` read.
        # The size policy refuses the absurd request on arithmetic, in microseconds,
        # but arithmetic only catches what it was taught to look for; this is the
        # backstop that holds whatever it did not. The budget is the verb's OWN
        # published bound -- the number the FLEX driver derives its client timeout
        # from -- so the service gives up at the same moment the caller does, instead
        # of holding a thread on a result nobody is still waiting for. It covers
        # build AND compile together, because a spec can be cheap to build and
        # ruinous to compile (a region is one polygon until `litho.fill` resolves
        # it) -- so BOTH stages submit to `_OBJECTS_EXECUTOR` below, and share its
        # one-orphan-at-a-time busy guard.
        pattern, meta = await _within(
            asyncio.wrap_future(_submit_guarded(_objects_pattern, objects, deadline)),
            deadline, "build")
        load_desc = f"{len(pattern.objects)} object spec(s)"
        svg_origin = None
    elif src == "auto":
        pattern = auto_device(hub.host.world.sample)
        load_desc = f"auto-router ({len(hub.host.world.sample.contact_centroids_um)} contacts)"
        svg_origin = None
    else:
        pattern, svg_origin = await asyncio.to_thread(_read_pattern, src,
                                                      allow_paths=allow_paths, hub=hub)
        load_desc = f"design {src}"
    # Compiling is the expensive half (a filled shape expands into many strokes), so
    # it goes to a thread: `loadPattern` publishes a 2 s bound, and the loop has
    # telemetry and an Abort button to keep answering while a big design compiles.
    #
    # GUARDED, because the compile is now part of LOADING. A file that reads cleanly
    # but cannot be compiled (degenerate geometry, a fill that cannot be generated) used
    # to surface as an ordinary `{"error": ...}` from the reader; with the compile
    # moved into this verb an unguarded raise would be an HTTP 500 on the cockpit's
    # own load button.
    try:
        if objects is not None:
            # SAME dedicated executor and the same busy guard as the build stage
            # above -- a slow compile is exactly as capable of orphaning a thread as
            # a slow build, and the `source` door never reaches this branch at all.
            compiling = asyncio.wrap_future(
                _submit_guarded(_compile_report, hub, pattern))
            report = await _within(compiling, deadline, "compile")
        else:
            report = await asyncio.to_thread(_compile_report, hub, pattern)
    except CommandError:
        raise
    except Exception as exc:
        msg = f"could not compile {load_desc}: {exc}"
        await hub.log("Design load REFUSED: " + msg)
        raise CommandError(INVALID_PARAMS, msg,
                           {"param": "source", "reason": "compile_failed"},
                           http={"error": msg}) from exc
    multi = multi_segment_objects(report)
    if multi:
        shown = ", ".join(f"{o['name']} ({o['points']:,} points, {o['segments']} "
                          f"out-waves)" for o in multi[:5])
        more = f", and {len(multi) - 5} more" if len(multi) > 5 else ""
        await hub.log(
            f"WARNING: {len(multi)} object(s) contain a stroke over the "
            f"{CHAIN_CHUNK_POINTS:,}-point chunk size and are fired as a CHAIN of "
            f"chunks -- {shown}{more}. A chain boundary does NOT lift the tip -- and "
            f"that is the hazard, not the reassurance: the tip sits STATIONARY AT "
            f"FULL BIAS at a single point of the trajectory for a matter of SECONDS "
            f"while Igor streams ~4.8 MB of capture text and re-arms, so expect a "
            f"DOSE BLOB there. LITHOCHAIN has also never run on the instrument. "
            f"Treat these objects as unproven.")
    # VOLTAGE WINDOW + DAC RAIL, BEFORE the pattern is installed (spec 13.1 item
    # 19), covering all three sources -- the `objects` parameter door (chiral
    # phase, erase bias), `auto`, and an inline SVG/file design. On refusal
    # `hub.pattern` is untouched (the previous design stays loaded) and the
    # GuardRefused (JSON-RPC -32050 with `.reason`) crosses the FLEX wire like
    # the write door's refusal. Same `command_guard` choke point set_props and
    # setObjects use.
    from ..command_guard import check_design_voltages
    check_design_voltages(getattr(hub.host, "instrument_model", None),
                          pattern.objects)
    hub.pattern = pattern
    # The report is cached against THIS pattern object's identity, so `startWrite`
    # reads its estimate back instead of re-compiling on a sub-second start path. A
    # pattern set any other way (the sketch endpoints, a design edit) simply misses
    # the cache and falls back to the cheap geometric estimate.
    hub._pattern_report = {"pattern_id": id(pattern), **report}
    await hub.log(f"Design loaded from {load_desc}: {pattern.name} "
                  f"({len(pattern.objects)} objects)")
    # SANITY (not a refusal): an SVG design mapped partly outside the scan it was
    # drawn over is placed anyway, but flagged in `origin.outside_frame` and logged
    # so a script can see it before writing. Same helper the REST door uses.
    if svg_origin is not None:
        from ..design_import import frame_bbox_check
        outside, bbox = frame_bbox_check(pattern, svg_origin["size_um"])
        svg_origin = {**svg_origin, "outside_frame": outside}
        if outside:
            await hub.log(
                f"WARNING: {pattern.name} extends outside the "
                f"{svg_origin['size_um']:g} um frame (lab bbox {bbox} um) -- part "
                f"of it is off the scan.")
    await hub.notify_design_changed()   # M1: loadPattern (REST + FLEX) replaces the design
    # `objects` is a LIST of per-object entries (index/name/points/segments), not a
    # count: a script has to be able to see WHICH object is multi-segment without a
    # second round trip. `object_count` carries the number it used to be.
    return {"name": pattern.name,
            "object_count": len(pattern.objects),
            "objects": object_entries(report, pattern, meta),
            "total_points": report["total_points"], "bbox_um": report["bbox_um"],
            # `segments` is the TRUE out-wave count: one arm per fired out-wave, so
            # one per stroke plus one per chain chunk beyond the cap. `strokes` is the
            # count of continuous tip paths, which is what `getWritten` has an entry
            # per. There is deliberately NO `chunks` key (it counted a number nothing
            # ever armed) and NO `max_segment_points` (the compiler's 8,192 cut is
            # opt-in and off, so it no longer describes anything on this wire).
            "segments": report["segments_total"],
            "strokes": report["strokes_total"],
            "multi_segment_objects": multi,
            "max_outwave_points": MAX_OUTWAVE_POINTS,
            "duration_s": report["duration_s"],
            # WHERE an SVG design was placed: the scan size its page top-left was
            # anchored to, and whether that came from the design's background image
            # or the current scan frame (design-origin rule). Absent (None) for the
            # `objects`/`auto` doors, which carry their own coordinates.
            "origin": svg_origin,
            "design": pattern.to_dict()}


# ============================================================== setObjects ======
#
# WHY THIS EXISTS. `loadPattern` builds a pattern; nothing then let a script CHANGE
# one property of an object it had already loaded -- retune a bias, disable a wire,
# turn a plain path chiral -- without re-sending the whole design. `setObjects` is
# the edit door: a map of `{id: {property: value}}` merged onto the loaded pattern,
# recompiled through the SAME guarded executor `loadPattern`'s `objects` door uses,
# with the previous pattern left in place on any refusal (validation, a runaway
# rebuild, or a compile that runs past the verb's bound).
#
# THE BUILDERS ARE NOT RE-IMPLEMENTED HERE, for the same reason the `objects` door
# does not grow its own sinusoid: a chiral edit rebuilds through `chiral_profile`
# exactly as `routers/design._reapply_chiral` does (hand folded into phase_rad, bias
# registered against the modulated arc length), so the phase convention lives in one
# place whichever door touched the wire.

_SET_SCALARS = frozenset({"voltage", "speed_um_s", "enabled"})
_CHIRAL_FIELDS = frozenset({"lambda_um", "y_amp_um", "v_k", "phase_deg", "hand"})
_FILL_FIELDS = frozenset({"spacing_um", "kind", "angle_deg"})


def _set_error(oid, field: str, msg: str, reason: str | None = None) -> CommandError:
    """Every setObjects refusal NAMES the id and the field it choked on, and the whole
    call is refused: a script sending a dozen edits has to be told which one, and a
    partially applied pattern is one the next `startWrite` would go and write."""
    text = f"setObjects: {oid!r}.{field}: {msg}"
    data = {"param": "updates", "id": oid, "field": field}
    http = {"error": text}
    if reason:
        data |= {"reason": reason, "object_id": oid}
        http["reason"] = reason
    return CommandError(INVALID_PARAMS, text, data, http=http)


def _set_num(d: dict, oid, field: str, *, positive: bool = False,
             non_negative: bool = False) -> float:
    raw = d.get(field)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise _set_error(oid, field, f"must be a number, got {type(raw).__name__}")
    val = float(raw)
    if not math.isfinite(val):
        raise _set_error(oid, field, f"must be finite, got {raw!r}")
    if positive and val <= 0:
        raise _set_error(oid, field, f"must be > 0, got {val:g}")
    if non_negative and val < 0:
        raise _set_error(oid, field, f"must be >= 0, got {val:g}")
    return val


def _apply_chiral_update(obj, oid, val) -> None:
    """`chiral: null` collapses the wire back to its plain base axis; `chiral: {..}`
    merges onto the object's CURRENT chiral parameters (all five required only when
    turning chirality on from a plain path) and rebuilds the toolpath from the stored
    axis through `chiral_profile`. `v0` is the object's own `voltage`, taken AFTER any
    voltage in this same update has been applied."""
    from ...litho.compiler import DEFAULT_INTERP_UM
    from ...litho.pens import bake_chiral

    if obj.kind != "path":
        raise _set_error(oid, "chiral", f"only a path can be chiral, not a "
                         f"{obj.kind!r}")
    if val is None:                                   # restore the plain path
        base = getattr(obj, "axis_um", None)
        if base:
            obj.points_um = [(float(x), float(y)) for x, y in base]
        obj.axis_um = None
        obj.profile = None
        obj.v_profile = None
        obj.interpolation_um = DEFAULT_INTERP_UM
        return
    if not isinstance(val, dict):
        raise _set_error(oid, "chiral", f"must be an object of chiral parameters or "
                         f"null, got {type(val).__name__}")
    val = {k: v for k, v in val.items() if k != "lambda_built_um"}
    unknown = sorted(set(val) - _CHIRAL_FIELDS)
    if unknown:
        raise _set_error(oid, unknown[0], f"unknown chiral field; the chiral block "
                         f"takes {', '.join(sorted(_CHIRAL_FIELDS))}")
    prof = obj.profile or {}
    cur: dict = {}
    if prof.get("kind") == "chiral":                  # merge onto what is there
        cur = _chiral_view(obj)
    merged = {**cur, **val}
    missing = [k for k in sorted(_CHIRAL_FIELDS) if k not in merged]
    if missing:
        raise _set_error(oid, missing[0], "required to turn chirality on from a plain "
                         "path (all five: lambda_um, y_amp_um, v_k, phase_deg, hand)")
    lam = _set_num(merged, oid, "lambda_um", positive=True)
    amp = _set_num(merged, oid, "y_amp_um", non_negative=True)
    v_k = _set_num(merged, oid, "v_k")
    phase = _set_num(merged, oid, "phase_deg")
    hand = _set_num(merged, oid, "hand")
    if hand not in (1.0, -1.0):
        raise _set_error(oid, "hand", f"must be 1 or -1, got {hand:g}")
    try:
        bake_chiral(obj, lam, amp, v_k, phase, hand)
    except ValueError as exc:
        text = str(exc)
        field = ("y_amp_um" if "relative to the pitch" in text
                 else "chiral" if "no length" in text or "not finite" in text
                 else "lambda_um")
        raise _set_error(oid, field, text, "chiral_build_refused") from exc


def _apply_fill_update(obj, oid, val) -> None:
    """Merge a `{spacing_um, kind, angle_deg}` block onto a fillable region's own
    fill parameters. `angle_deg` is only meaningful for the `directional` strategy --
    every other one has its own geometry -- so it is refused on the others, exactly as
    the `objects` door refuses it, rather than accepted and ignored."""
    from ...litho.pens import PENS

    if obj.kind not in ("rect", "taper", "shape"):
        raise _set_error(oid, "fill", f"only a fillable region (rect/taper/shape) has "
                         f"a fill, not a {obj.kind!r}")
    if not isinstance(val, dict):
        raise _set_error(oid, "fill", f"must be an object of fill parameters, got "
                         f"{type(val).__name__}")
    unknown = sorted(set(val) - _FILL_FIELDS)
    if unknown:
        raise _set_error(oid, unknown[0], f"unknown fill field; the fill block takes "
                         f"{', '.join(sorted(_FILL_FIELDS))}")
    fills = [c[0] for c in PENS["region"].choices["fill"]]
    kind = obj.fill_kind
    if "kind" in val:
        if val["kind"] not in fills:
            raise _set_error(oid, "kind", f"must be one of {', '.join(fills)}, got "
                             f"{val['kind']!r}")
        kind = val["kind"]
    if "spacing_um" in val:
        obj.fill_spacing_um = _set_num(val, oid, "spacing_um", positive=True)
    if "angle_deg" in val:
        if kind != "directional":
            raise _set_error(oid, "angle_deg", f"only the \"directional\" fill sweeps "
                             f"at a chosen angle; {kind!r} would ignore it")
        obj.fill_angle_deg = _set_num(val, oid, "angle_deg")
    obj.fill_kind = kind


def _apply_object_update(obj, oid, changes: dict) -> None:
    """Validate and apply ONE object's `{property: value}` map onto `obj` (a copy of
    the loaded object -- the caller works on a deep copy so the live pattern survives
    a refusal). Scalars first, so a chiral rebuild in the same call reads the NEW
    voltage as its `v0`. A voltage change alone on a chiral wire re-bakes it too."""
    from ...litho.pens import is_chiral, rebake_chiral

    if not isinstance(changes, dict):
        raise _set_error(oid, "properties", f"must be an object of "
                         f"{{property: value}}, got {type(changes).__name__}")
    allowed = set(_SET_SCALARS) | {"chiral", "fill"}
    unknown = sorted(set(changes) - allowed)
    if unknown:
        raise _set_error(oid, unknown[0], "unknown property; settable properties are "
                         "voltage, speed_um_s, enabled, and chiral (a path) or fill "
                         "(a region)")
    if "voltage" in changes:
        obj.voltage = _set_num(changes, oid, "voltage")
    if "speed_um_s" in changes:
        obj.speed_um_s = _set_num(changes, oid, "speed_um_s", positive=True)
    if "enabled" in changes:
        if not isinstance(changes["enabled"], bool):
            raise _set_error(oid, "enabled", f"must be true or false, got "
                             f"{type(changes['enabled']).__name__}")
        obj.enabled = changes["enabled"]
    if "chiral" in changes:
        _apply_chiral_update(obj, oid, changes["chiral"])
    elif "voltage" in changes and is_chiral(obj):
        try:
            built = rebake_chiral(obj)
        except ValueError as exc:
            raise _set_error(oid, "voltage", f"this chiral wire cannot be re-baked: "
                             f"{exc}", "chiral_build_refused") from exc
        if built is None:
            raise _set_error(oid, "voltage", "this chiral wire has no spine or "
                             "settings to re-bake from; send its chiral block with "
                             "the voltage", "chiral_unrebakeable")
    if "fill" in changes:
        _apply_fill_update(obj, oid, changes["fill"])


async def set_objects(hub, updates) -> dict:
    """Merge `updates = {id: {property: value}}` onto the loaded pattern and recompile.

    THE WHOLE CALL IS ATOMIC. Every id and every property is validated and applied to
    a DEEP COPY first, so an unknown id, an unknown or wrong-typed property, a property
    invalid for that object's kind, or a runaway chiral rebuild refuses the call with
    `-32602` -- naming the id and field -- and the live pattern is untouched. Only once
    the copy compiles cleanly (on the SAME dedicated executor and 2 s wall clock as the
    `objects` door, with the same per-object/whole-pattern point caps) is it made the
    loaded pattern. Returns the per-object listing, the same shape `loadPattern` and
    `getPattern` return.
    """
    pattern = getattr(hub, "pattern", None)
    if not pattern or not getattr(pattern, "objects", None):
        msg = ("setObjects: no pattern is loaded. Call loadPattern first -- there is "
               "nothing to edit.")
        raise CommandError(REFUSED, msg, {"reason": "no_pattern"},
                           http={"error": msg, "reason": "no_pattern"})
    if not isinstance(updates, dict) or not updates:
        msg = "setObjects: updates must be a non-empty object of {id: {property: value}}"
        raise CommandError(INVALID_PARAMS, msg, {"param": "updates"},
                           http={"error": msg})

    import copy
    new_pattern = copy.deepcopy(pattern)
    by_name = {getattr(o, "name", None): o for o in new_pattern.objects}
    updated_objs = []
    for oid, changes in updates.items():
        obj = by_name.get(oid)
        if obj is None:
            raise _set_error(oid, "id", f"no object with id {oid!r} in the loaded "
                             f"pattern")
        _apply_object_update(obj, oid, changes)
        updated_objs.append(obj)

    # RECOMPILE on the dedicated objects-door executor, under the verb's own bound, so
    # a slow compile is abandoned (previous pattern kept) rather than blocking the
    # serialised command socket -- the same guard the `objects` door relies on.
    deadline = time.monotonic() + _objects_bound_s()
    compiling = asyncio.wrap_future(_submit_guarded(_compile_report, hub, new_pattern))
    report = await _within(compiling, deadline, "compile")

    # THE SAME SIZE CAPS the `objects` door applies, checked on the compiled counts:
    # an edit (a coarser-to-finer chiral pitch, a tighter fill) can push a pattern past
    # what any real write would be, and that is a units slip, not an experiment.
    if report["total_points"] > MAX_TOTAL_POINTS:
        msg = (f"setObjects: the edited pattern compiles to about "
               f"{report['total_points']:,} points, over the {MAX_TOTAL_POINTS:,}-"
               f"point limit for one pattern. Coarsen a pitch or shorten a wire.")
        raise CommandError(INVALID_PARAMS, msg, {"param": "updates"},
                           http={"error": msg})
    for name, n in report["points"].items():
        if n > MAX_OBJECT_POINTS:
            msg = (f"setObjects: object {name!r} now compiles to about {n:,} points, "
                   f"over the {MAX_OBJECT_POINTS:,}-point per-object limit. Coarsen "
                   f"its pitch or shorten it.")
            raise CommandError(INVALID_PARAMS, msg,
                               {"param": "updates", "id": name},
                               http={"error": msg})

    # VOLTAGE WINDOW + DAC RAIL, on the OBJECTS THIS EDIT TOUCHED, BEFORE the live
    # pattern is replaced (spec 13.1 item 19). Scoped to the updated objects, not the
    # whole pattern, to match the REST set_props door (which checks only the objects it
    # changed): an unrelated edit -- e.g. disabling one object -- must not be blocked by
    # some OTHER object that is already out of window (such a value can only pre-exist
    # via a door item 19 does not cover, e.g. /api/design/add). `new_pattern` is a deep
    # copy; on refusal `hub.pattern` is still the old pattern (the edit did not happen),
    # and the GuardRefused (a CommandError, JSON-RPC -32050 carrying `.reason`) crosses
    # the FLEX wire the way the write door's refusal does.
    from ..command_guard import check_design_voltages
    check_design_voltages(getattr(hub.host, "instrument_model", None), updated_objs)

    hub.pattern = new_pattern
    hub._pattern_report = {"pattern_id": id(new_pattern), **report}
    await hub.log(f"Pattern edited: {len(updates)} object(s) updated "
                  f"({', '.join(sorted(str(k) for k in updates))})")
    await hub.notify_design_changed()   # M1: setObjects (REST + FLEX) mutates the design
    return {"name": new_pattern.name,
            "object_count": len(new_pattern.objects),
            "objects": object_entries(report, new_pattern),
            "total_points": report["total_points"], "bbox_um": report["bbox_um"],
            "segments": report["segments_total"],
            "strokes": report["strokes_total"],
            "multi_segment_objects": multi_segment_objects(report),
            "max_outwave_points": MAX_OUTWAVE_POINTS,
            "duration_s": report["duration_s"]}


# =============================================================== startWrite ======

def select_objects(hub, body: dict, *,
                   commit: bool = True) -> tuple[list, int, bool]:
    """Which objects THIS write covers: `(todo, skipped, cleared)`.

    Three selections, and they are different questions:

      * `objects: [0, 3]` / `["path1", 3]` -- exactly these objects, by INDEX or by
        ID (name), and the two may be mixed in one list. Written whether or not they
        have been written before, and in the order given. A script names what it
        wants; it does not want the service second-guessing it from a per-session
        "already written" set.
      * `objects: "all"` / `{"all": true}` -- every enabled object, and the written
        set is cleared first (the cockpit's "Write all").
      * neither -- the cockpit's default: only what has not been written yet, which is
        the answer to "I drew one more thing, write that".

    A DISABLED object is never written (the compiler skips it too), so it is filtered
    out here and never marked written -- and an explicitly NAMED disabled object (by
    index or id) is a `-32602`, because silently dropping it would report a write of
    something that never happened.
    """
    done = getattr(hub.host, "written_objects", None)
    if done is None:
        done = hub.host.written_objects = set()
    objects = hub.pattern.objects
    by_name = {getattr(o, "name", None): idx for idx, o in enumerate(objects)}
    sel = body.get("objects")
    if isinstance(sel, (list, tuple)):
        todo = []
        seen: set[int] = set()          # INDICES, not object equality: two objects
        for raw in sel:                 # with identical geometry are still two objects
            if isinstance(raw, bool):
                msg = (f"startWrite: objects must be \"all\" or a list of object "
                       f"indices or ids, got {type(raw).__name__} {raw!r}")
                raise CommandError(INVALID_PARAMS, msg, {"param": "objects"},
                                   http={"error": msg, "reason": "bad_params"})
            if isinstance(raw, str):
                idx = by_name.get(raw)
                if idx is None:
                    msg = (f"startWrite: no object with id {raw!r} in the loaded "
                           f"design")
                    raise CommandError(INVALID_PARAMS, msg,
                                       {"param": "objects", "id": raw},
                                       http={"error": msg, "reason": "bad_params"})
                raw = idx
            elif not isinstance(raw, int):
                msg = (f"startWrite: objects must be \"all\" or a list of object "
                       f"indices or ids, got {type(raw).__name__} {raw!r}")
                raise CommandError(INVALID_PARAMS, msg, {"param": "objects"},
                                   http={"error": msg, "reason": "bad_params"})
            if raw < 0 or raw >= len(objects):
                msg = (f"startWrite: object index {raw} is out of range -- the loaded "
                       f"design has {len(objects)} object(s)")
                raise CommandError(INVALID_PARAMS, msg,
                                   {"param": "objects", "index": raw,
                                    "objects": len(objects)},
                                   http={"error": msg, "reason": "bad_params"})
            obj = objects[raw]
            if not getattr(obj, "enabled", True):
                msg = (f"startWrite: object {raw} ({obj.name}) is disabled and would "
                       f"not be written. Enable it, or drop it from `objects`.")
                raise CommandError(INVALID_PARAMS, msg,
                                   {"param": "objects", "index": raw,
                                    "name": obj.name},
                                   http={"error": msg, "reason": "bad_params"})
            if raw not in seen:         # a repeated index is one write, not two doses
                seen.add(raw)
                todo.append(obj)
        return todo, len(objects) - len(todo), False
    if body.get("all") or (isinstance(sel, str) and sel.strip().lower() == "all"):
        # `commit=False` is the ESTIMATE call (`start_write` sizes the run before it
        # claims anything). Clearing the written set there would mean a REFUSED
        # startWrite -- busy, AC mode, backend mismatch -- silently wiped the
        # cockpit's "already written" marks, and the operator's next panel write
        # would re-dose everything.
        if commit:
            done.clear()
        todo = [o for o in objects if getattr(o, "enabled", True)]
        return todo, len(objects) - len(todo), True
    todo = [o for o in objects
            if id(o) not in done and getattr(o, "enabled", True)]
    return todo, len(objects) - len(todo), False


def commit_selection(hub, body: dict) -> None:
    """Apply the part of the selection that MUTATES bookkeeping, once every guard has
    passed and the write is actually about to start.

    Only `objects: "all"` / `{"all": true}` mutates anything: it clears the written
    set so the pass covers the whole design again. Marking each object as written is
    NOT done here -- the pump does it per object, at completion (see `_marks`), so a
    mid-write abort leaves the design honestly half-marked instead of claiming every
    object was written.
    """
    sel = body.get("objects")
    if body.get("all") or (isinstance(sel, str) and sel.strip().lower() == "all"):
        done = getattr(hub.host, "written_objects", None)
        if done is None:
            hub.host.written_objects = set()
        else:
            done.clear()


def _marks(hub, todo: list) -> dict:
    """The pump's marking plan: `{object name: {"id": id(obj), "expect": n|None}}`.

    `expect` is how many STROKES that object becomes, taken from the compile
    `loadPattern` already did -- strokes, not out-waves: both workers emit exactly one
    `litho_object_done` per stroke, and a chain's chunks are not separate completions.
    The pump marks an object written only once that many
    `litho_object_done` messages have arrived, so an object cut in half by an abort is
    NOT marked -- the panel's next default write re-covers it, which is the safe
    direction (an under-marked object gets a second dose; an over-marked one stays
    unfinished for ever).

    `expect: None` when the pattern never went through `loadPattern` (a sketch, a
    design edit): the pump then marks on the first completion, which is still strictly
    better than the old mark-everything-at-start.
    """
    report = getattr(hub, "_pattern_report", None) or {}
    by_object = (report.get("strokes") or {}
                 if report.get("pattern_id") == id(getattr(hub, "pattern", None))
                 else {})
    return {getattr(o, "name", None): {"id": id(o),
                                       "expect": by_object.get(getattr(o, "name", None))}
            for o in todo}


def _world_cal(hub) -> tuple:
    """The LVDT sensitivities/offsets the compiler needs, with the twin defaults."""
    w = getattr(hub.host, "world", None)
    return (float(getattr(w, "x_lvdt_sens", 90e-6) or 90e-6),
            float(getattr(w, "y_lvdt_sens", 90e-6) or 90e-6),
            float(getattr(w, "x_lvdt_offset", 0.0) or 0.0),
            float(getattr(w, "y_lvdt_offset", 0.0) or 0.0))


def _object_trajectory(obj, x_sens, y_sens, x_off, y_off) -> tuple:
    """Trajectory seconds, length_um, stroke count and chain-seam count for ONE
    object, compiled EXACTLY as the write path compiles it (`expand` -> the very
    `compile_object` the writer runs). This is the SINGLE per-object measurement both
    `GET /api/litho/estimate` and the write-start estimate read (spec sec 10, 13.1
    item 18), so the two can never disagree. Raises to the caller's fallback when the
    geometry will not compile."""
    from ...litho.compiler import compile_object, expand
    seconds = 0.0
    length = 0.0
    strokes = 0
    chain_seams = 0
    for part in expand(obj):
        comp = compile_object(part, x_sens, y_sens, x_off, y_off)
        seconds += float(comp["duration_s"])
        length += float(comp["length_um"])
        strokes += 1
        chain_seams += max(0, _chunks_for(len(comp["path_um"])) - 1)
    return seconds, length, strokes, chain_seams


def _object_overhead_s(strokes: int, chain_seams: int) -> float:
    """The time one object costs a write BEYOND its trajectory. THREE different seams,
    charged differently because they cost differently: the fixed per-object arm/settle
    margin (PER_OBJECT_OVERHEAD_S) plus the verified withdraw on the move TO the object
    (PER_OBJECT_WITHDRAW_S, docs/XY_SLEW_SPEC.md rule 5); one mild hover-lift per gap
    between two disconnected strokes OF THE SAME object (PER_STROKE_TRANSIT_S); and one
    tip-down dwell per chain boundary inside an over-cap stroke (CHAIN_SEAM_S). A
    one-stroke object pays no PER_STROKE_TRANSIT_S however many points it has."""
    transits = max(0, int(strokes) - 1)
    return (PER_OBJECT_OVERHEAD_S + PER_OBJECT_WITHDRAW_S
            + transits * PER_STROKE_TRANSIT_S
            + max(0, int(chain_seams)) * CHAIN_SEAM_S)


def write_est_s(hub, todo: list) -> float:
    """How long this write should take, plus a margin. Honest rather than optimistic:
    a driver's blocking helper SLEEPS on this before it starts polling.

    ONE estimator (spec sec 10, 13.1 item 18): the same per-object trajectory +
    overhead `GET /api/litho/estimate` charges, so the write-start estimate and the
    published estimate can never disagree. `write_estimate(hub)["total_s"]` equals
    this over the same object set. Falls back to polyline-length over speed for a
    geometry the compiler cannot expand.
    """
    return _est_total_s(_world_cal(hub), todo)


def _est_total_s(cal: tuple, todo: list, stop=None) -> float:
    """`write_est_s` over a calibration and objects already read, so it can run off
    the service loop without touching the hub. `stop` (a threading.Event) ends it
    early, between objects, when nobody is waiting for the result any more."""
    x_sens, y_sens, x_off, y_off = cal
    total = 0.0
    for obj in todo:
        if stop is not None and stop.is_set():
            break
        try:
            seconds, _length, strokes, chain_seams = _object_trajectory(
                obj, x_sens, y_sens, x_off, y_off)
        except Exception:                                # noqa: BLE001
            seconds, strokes, chain_seams = _geometric_seconds(obj), 1, 0
        total += float(seconds) + _object_overhead_s(strokes, chain_seams)
    return round(total + EST_MARGIN_S, 3)


def write_estimate(hub) -> dict:
    """The per-object write-time estimate served by `GET /api/litho/estimate` and
    the FLEX `getEstimate` (docs/INSTRUMENT_MODEL_SPEC.md sections 4/10, 13.1 item 18).

    `{"total_s", "trajectory_s", "objects": [{"id" (object NAME), "seconds",
    "length_um", "speed_um_s", "enabled"}]}`, computed FROM THE SERVICE'S OWN enabled
    set -- never from a client's copy. Each object is compiled through the very
    `compile_object` the write path uses; the per-object `seconds` is that object's
    pure TRAJECTORY time (the same number the write-start log prints, "Writing <name>:
    <length_um> um at +V V (<duration_s> s)"). `trajectory_s` is the sum of the enabled
    objects' trajectory seconds; `total_s` is what the service will ACTUALLY spend --
    trajectory PLUS the per-object arm/withdraw/stroke/chain overhead PLUS a whole-run
    margin -- and equals `write_est_s` over the same set, closing the B2 contradiction
    where the published total omitted the overhead the write-start estimate charged.
    Disabled objects are listed with `enabled: false` and excluded from both totals,
    so a stale cockpit view shows up as a mismatch.

    Recomputed on every call from the live pattern -- no caching -- so a mutation
    between two calls is reflected immediately.
    """
    pattern = getattr(hub, "pattern", None)
    return _estimate_report(_world_cal(hub),
                            list(getattr(pattern, "objects", None) or []))


def _estimate_report(cal: tuple, objs: list) -> dict:
    """`write_estimate` over a calibration and objects already read, so it can run off
    the service loop without touching the hub."""
    x_sens, y_sens, x_off, y_off = cal
    out: list[dict] = []
    total = 0.0
    trajectory = 0.0
    any_enabled = False
    for obj in objs:
        strokes = 1
        chain_seams = 0
        try:
            seconds, length, strokes, chain_seams = _object_trajectory(
                obj, x_sens, y_sens, x_off, y_off)
        except Exception:                                # noqa: BLE001
            # A geometry the compiler cannot expand (too few points, a malformed
            # bezier) still gets an honest polyline-over-speed floor rather than
            # taking the whole estimate down.
            seconds = _geometric_seconds(obj)
            pts = getattr(obj, "points_um", None) or []
            length = 0.0
            for (x0, y0), (x1, y1) in pairwise(pts):
                length += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        enabled = bool(getattr(obj, "enabled", True))
        out.append({
            "id": getattr(obj, "name", None),
            "seconds": round(seconds, 3),
            "length_um": round(length, 3),
            "speed_um_s": float(getattr(obj, "speed_um_s", 0.0) or 0.0),
            "enabled": enabled,
        })
        if enabled:
            any_enabled = True
            trajectory += seconds
            total += seconds + _object_overhead_s(strokes, chain_seams)
    if any_enabled:
        total += EST_MARGIN_S
    return {"total_s": round(total, 3), "trajectory_s": round(trajectory, 3),
            "objects": out}


# ----------------------------------------------------- the estimate, off the loop --
#
# The estimate compiles every object, which is seconds of pure Python for a dense
# fill. On the service loop that is seconds in which Abort, Withdraw, heartbeats and
# `getState` wait, so the loop's two callers (`start_write` and
# `GET /api/litho/estimate`) compute it in a thread.
#
# Each compile runs on its OWN daemon thread, never in the default executor: that
# pool is the one `withdraw`, the twin park and the AC guard run in, a compile cannot
# be interrupted once started, and a caller that gives up (the FLEX reply bound, a
# closed page) leaves its compile running. The number of compiles alive is bounded
# instead: one for `start_write`, one for the endpoint.

def _detached(obj):
    """A copy of one design object that an edit on the loop cannot change under the
    estimate thread: the object and its lists are copied down to the point pairs
    (which are tuples or two-float lists), its dict fields in full."""
    def detach(val):
        return [detach(e) for e in val] if isinstance(val, list) else val

    c = copy.copy(obj)
    for key, val in list(vars(c).items()):
        if isinstance(val, list):
            setattr(c, key, detach(val))
        elif isinstance(val, dict):
            setattr(c, key, copy.deepcopy(val))
    return c


def _compile_thread(fn, *args) -> concurrent.futures.Future:
    """Run `fn(*args)` on a new daemon thread. Daemon, so a compile nobody is waiting
    for any more never delays interpreter exit."""
    fut: concurrent.futures.Future = concurrent.futures.Future()
    # Marked running at once: a waiter that is cancelled then cannot cancel the
    # future (`asyncio.wrap_future` tries), so `done()` stays false for exactly as
    # long as the thread is alive, which is what the callers bound themselves by.
    fut.set_running_or_notify_cancel()

    def run():
        try:
            result = fn(*args)
        except BaseException as exc:                     # noqa: BLE001
            fut.set_exception(exc)
        else:
            fut.set_result(result)

    threading.Thread(target=run, name="litho-estimate", daemon=True).start()
    return fut


class _EstimateQueue:
    """`GET /api/litho/estimate`, latest wins: one compile running and at most one
    more waiting, whatever the number of requests.

    A request that arrives while a compile runs is answered by the NEXT compile,
    which reads the design when it starts; every request waiting at that moment
    shares it. So a burst of edits costs two compiles, not one per edit, and no
    request is answered from a design older than the one it asked about."""

    def __init__(self, loop):
        self.loop = loop
        self.running: asyncio.Future | None = None
        self.waiting: asyncio.Future | None = None

    def request(self, hub) -> asyncio.Future:
        if self.running is None:
            return self._start(hub, self.loop.create_future())
        if self.waiting is None:
            self.waiting = self.loop.create_future()
        return self.waiting

    def _start(self, hub, fut: asyncio.Future) -> asyncio.Future:
        self.running = fut
        try:
            pattern = getattr(hub, "pattern", None)
            objs = [_detached(o) for o in (getattr(pattern, "objects", None) or [])]
            compile_ = _compile_thread(_estimate_report, _world_cal(hub), objs)
        except Exception as exc:                         # noqa: BLE001
            self._finish(hub, fut, None, exc)
            return fut

        def done(c: concurrent.futures.Future) -> None:  # on the compile thread
            exc = c.exception()
            try:
                self.loop.call_soon_threadsafe(
                    self._finish, hub, fut, None if exc else c.result(), exc)
            except RuntimeError:
                pass                                     # the loop has closed

        compile_.add_done_callback(done)
        return fut

    def _finish(self, hub, fut: asyncio.Future, result, exc) -> None:
        self.running = None
        nxt, self.waiting = self.waiting, None
        if nxt is not None:
            self._start(hub, nxt)
        if fut.done():
            return
        if exc is not None:
            fut.set_exception(exc)
        else:
            fut.set_result(result)


async def write_estimate_async(hub) -> dict:
    """`write_estimate(hub)`, computed off the loop from the design as it is when the
    compile starts (`_EstimateQueue`)."""
    loop = asyncio.get_running_loop()
    q = getattr(hub.host, "_estimate_queue", None)
    if q is None or q.loop is not loop:                  # first use, or a new loop
        q = hub.host._estimate_queue = _EstimateQueue(loop)
    # Shielded: a request that goes away must not cancel the compile's result for
    # the other requests sharing it.
    return await asyncio.shield(q.request(hub))


async def write_est_s_async(hub, todo: list) -> float:
    """`write_est_s(hub, todo)`, computed off the loop from copies of `todo`.

    One at a time. A caller that is cancelled while waiting (the FLEX reply bound)
    leaves its compile running, so a call that finds one still alive is refused
    (`-32020 busy`, `reason: "estimate_in_progress"`) rather than started beside it
    or queued behind it. The abandoned compile is told to stop and does so at its
    next object."""
    prev = getattr(hub.host, "_start_estimate", None)
    if prev is not None and not prev[0].done():
        age = time.monotonic() - prev[2]
        msg = (f"startWrite: the write estimate of an earlier startWrite, started "
               f"{age:.1f} s ago, is still computing. It cannot be interrupted; "
               f"retry when it has finished.")
        raise CommandError(BUSY, msg, {"reason": "estimate_in_progress",
                                       "age_s": round(age, 1)},
                           http={"error": msg, "reason": "estimate_in_progress"})
    stop = threading.Event()
    compile_ = _compile_thread(_est_total_s, _world_cal(hub),
                               [_detached(o) for o in todo], stop)
    hub.host._start_estimate = (compile_, stop, time.monotonic())
    try:
        return await asyncio.wrap_future(compile_)
    except asyncio.CancelledError:
        stop.set()
        raise


def _stop_seq(hub) -> tuple:
    """Changes whenever an abort, park or withdraw signals the in-flight pass
    (`host.safe_park.signal_run_abort`) or a scan is stopped (`host/scan_cmds.py`
    `stop_scan`, whose twin branch retracts without that signal). A change across
    an await means one of them arrived during it."""
    return (int(getattr(hub.host, "_run_abort_seq", 0) or 0),
            int(getattr(hub.host, "_scan_stop_seq", 0) or 0))


def _refuse_without_keeper(hub) -> None:
    """No write starts while the keeper is dead and nothing has restored contact (decision
    8). Stop, park and withdraw never come here."""
    from .. import keeper_link
    why = keeper_link.keeper_lost(hub)
    if why is None:
        return
    msg = f"write refused: {why}"
    raise CommandError(REFUSED, msg, {"reason": "keeper_lost"},
                       http={"error": msg, "reason": "keeper_lost"})


async def _refuse_if_stopped(hub, since: tuple, during: str) -> None:
    """Refuse a write start (reason `aborted_during_start`) when a stop was signalled
    since `since` (a `_stop_seq` taken when the start began).

    The abort flags cannot show this on their own: every park leaves them set, so the
    hardware start clears `_hw_litho_abort` part-way through, after its AC-guard and
    free-air reads, and a stop that landed during those reads is erased with it. The
    counters are never cleared. One comparison; nothing waits on it."""
    if _stop_seq(hub) == since:
        return
    msg = (f"write start interrupted: an abort, park, withdraw or scan stop was "
           f"signalled while {during}")
    await hub.log("Lithography REFUSED: " + msg)
    raise CommandError(REFUSED, msg, {"reason": "aborted_during_start"},
                       http={"error": msg, "reason": "aborted_during_start"})


def _fresh_report(hub) -> dict:
    """The cached `loadPattern` compile report, but ONLY if it belongs to the pattern
    that is loaded right now. A pattern set any other way (a sketch endpoint, a design
    edit) misses the cache, and every reader then falls back to its own estimate."""
    report = getattr(hub, "_pattern_report", None) or {}
    if report.get("pattern_id") != id(getattr(hub, "pattern", None)):
        return {}
    return report


def write_segments(hub, todo: list) -> int:
    """How many OUT-WAVES this write actually arms, summed over the selected
    objects: one per STROKE, plus one more per chain chunk for a stroke over the
    out-wave cap. Published by `startWrite` so a script knows, before it polls, how
    many arms the write contains -- and so the number it is told is the number of
    ARMFIREs (plus LITHOCHAIN chunks) the instrument will see."""
    segments = _fresh_report(hub).get("segments") or {}
    total = 0
    for obj in todo:
        name = getattr(obj, "name", None)
        n = segments.get(name)
        if n is None:
            # No compile report: floor it from the DESIGN point count. One stroke
            # unless the design polyline alone is over the cap. An underestimate for a
            # filled shape, exactly like `_geometric_seconds`.
            n = _chunks_for(len(getattr(obj, "points_um", None) or []))
        total += int(n)
    return total


def _geometric_seconds(obj) -> float:
    """Polyline length / speed, for a pattern that never went through `loadPattern`."""
    pts = getattr(obj, "points_um", None) or []
    speed = float(getattr(obj, "speed_um_s", 0.0) or 0.0)
    if len(pts) < 2 or speed <= 0:
        return 0.0
    length = 0.0
    for (x0, y0), (x1, y1) in pairwise(pts):
        length += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    return length / speed


class LithoWorkerHandle:
    """The abortable handle over a running write worker thread, registered on
    `hub.litho_worker` for the duration of a write (docs/INSTRUMENT_MODEL_SPEC.md
    Sec 3 + 10). `safe_park` calls `request_abort()` then `join(timeout)` on it
    BEFORE any Z command, so a park can never race a pass.

    `abort_event` is the worker's cooperative flag (`hub._hw_litho_abort`, honoured per
    stroke in `_run_write`). The twin plugin is stopped by `signal_run_abort`, which a park
    sends first, so the handle only sets the flag."""

    def __init__(self, thread, *, abort_event=None):
        self._thread = thread
        self._abort_event = abort_event

    def request_abort(self) -> None:
        if self._abort_event is not None:
            try:
                self._abort_event.set()
            except Exception:                             # noqa: BLE001,S110
                pass

    def join(self, timeout=None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())


def _claim(hub, kind: str):
    """The single-writer claim (plan Sec E.2), taken SYNCHRONOUSLY -- no `await`
    between the busy check and the set. Raises the wire's `-32020 busy` carrying the
    in-flight `run_id`, and the HTTP body the routers have always returned."""
    from ..control import RunBusy
    from ..commands.errors import BUSY
    try:
        return hub.claim_run(kind)
    except RunBusy as busy:
        raise CommandError(BUSY, str(busy),
                           {"run_id": busy.run_id or hub._in_flight_run_id(),
                            "kind": busy.kind},
                           http=hub.busy_body(busy)) from busy


@dataclass(frozen=True)
class WriteIntake:
    """How one carrier asks for a write. The two intakes below are the only places
    where the cockpit's request and a script's `startWrite` differ; everything after
    them (guards, claim, worker, journal) is shared (docs/INSTRUMENT_MODEL_SPEC.md
    sections 4/10/12).

    `verb` names the request in -32602 messages. `refuse_selection` runs before the
    guards. `setpoint` reads the setpoint out of the body as a `WriteSetpoint`."""

    verb: str
    refuse_selection: Callable[[dict], None]
    setpoint: Callable[[dict], Any]


def _flex_refuse_selection(body: dict) -> None:
    return None


def _refuse_params(msg: str) -> CommandError:
    return CommandError(INVALID_PARAMS, msg, {"reason": "invalid_params"},
                        http={"error": msg, "reason": "invalid_params"})


def _write_setpoint(body: dict, *, verb: str, accepts_setpoint_relative: bool = False):
    """The write's setpoint, as both doors ask for it: `setpoint` (volts, required) and
    `setpoint_reference` ("absolute" by default: used as given, so a script that sends an
    absolute value and no reference keeps its meaning; or "free_air": the fresh free-air
    deflection plus the setpoint). A write with no setpoint is refused.

    The door's "free_air" is the run contract's `Setpoint(reference="relative")`
    (`WriteSetpoint.relative`). The cockpit's `setpoint_relative` means the same as a
    free-air `setpoint`, so only the HTTP door accepts it and the panel keeps sending
    what it always has; FLEX refuses it."""
    given = body.get("setpoint_reference")
    reference = "absolute" if given is None else given
    if reference not in ("free_air", "absolute"):
        msg = (f'{verb}: setpoint_reference must be "free_air" or "absolute", '
               f"got {reference!r}")
        raise CommandError(INVALID_PARAMS, msg, {"param": "setpoint_reference"},
                           http={"error": msg, "reason": "bad_params"})
    cockpit_key = body.get("setpoint_relative") is not None
    if cockpit_key and not accepts_setpoint_relative:
        raise _refuse_params(f"{verb}: `setpoint_relative` is not accepted here; "
                             f"give `setpoint` (volts) with `setpoint_reference`")
    if cockpit_key and body.get("setpoint") is not None:
        raise _refuse_params(f"{verb}: give `setpoint` or `setpoint_relative`, not both")
    if cockpit_key and given == "absolute":
        raise _refuse_params(f"{verb}: `setpoint_relative` is relative to free air; "
                             f"use `setpoint` for an absolute one")
    volts = _number(body, "setpoint_relative" if cockpit_key else "setpoint", None, verb=verb)
    if volts is None:
        msg = (f"{verb}: `setpoint` (volts) is required, with `setpoint_reference` "
               f'"absolute" (default) or "free_air"')
        raise CommandError(INVALID_PARAMS, msg, {"reason": "setpoint_missing", "param": "setpoint"},
                           http={"error": msg, "reason": "setpoint_missing"})
    return WriteSetpoint(volts, relative=(cockpit_key or reference == "free_air"))


def _flex_setpoint(body: dict):
    return _write_setpoint(body, verb="startWrite")


def _http_refuse_selection(body: dict) -> None:
    """13.1 item 1: the cockpit takes an EXPLICIT list of object ids; the `all`
    select-all key is a FLEX `startWrite` convenience. Refused as `unknown_object`
    BEFORE the guard, so it reads as "no such object 'all'" rather than reaching the
    amp_gain check and returning amp_gain_missing."""
    if body.get("all"):
        msg = ("litho/write: the REST door takes an explicit list of object ids; "
               "the `all` select-all key is a FLEX startWrite convenience, not a "
               "valid object id")
        raise CommandError(REFUSED, msg, {"reason": "unknown_object"},
                           http={"error": msg, "reason": "unknown_object"})


def _http_setpoint(body: dict):
    return _write_setpoint(body, verb="litho/write", accepts_setpoint_relative=True)


flex_intake = WriteIntake("startWrite", _flex_refuse_selection, _flex_setpoint)
http_intake = WriteIntake("litho/write", _http_refuse_selection, _http_setpoint)


def _absolute_setpoint(hub, ask) -> float:
    """The applied setpoint: an absolute ask as sent, a relative one added to the free
    air on record (the relative number itself if there is none)."""
    if not ask.relative:
        return ask.value_v
    free_air = getattr(hub.host, "free_air", None)
    return (free_air.value_v + ask.value_v) if free_air is not None else ask.value_v


async def litho_write(hub, body: dict | None = None, *, intake: WriteIntake = http_intake,
                      since: tuple | None = None) -> dict:
    """Write what has not been written yet (or exactly what was asked for).

    User request: "when I draw one thing it does it. great. but then when I draw another
    thing it redraws the first one. it should remember that i already drew it."

    Sketching APPENDS to the pattern, and this used to write the whole pattern every
    time, so the second stroke re-traced the first. On a real sample that is not
    merely wasted time: writing over existing material at bias is a second dose
    nobody asked for, and the line comes out wider than the one that was measured.
    So an object that has been written carries the fact, and a write covers only the
    ones that have not. `{"all": true}` (or the wire's `objects: "all"`) deliberately
    goes over everything again.

    Returns the HTTP door's shape (`{ok, writing, skipped, run_id, backend?}`);
    `start_write` renders the wire's `{run_id, est_s, objects}`.

    `since` is the stop count (`_stop_seq`) at the moment the start began, taken here
    unless the caller began earlier (`start_write`, before its estimate). It is
    compared at the last check before the worker thread starts, so a stop that
    arrives at any point of the start refuses it.
    """
    if since is None:
        since = _stop_seq(hub)           # first statement: before any await
    body = dict(body or {})
    # SILENT-TWIN GUARD (plan Sec C.2). A station CONFIGURED as hardware whose bridge
    # is not live must not quietly execute against simulated physics and report
    # ordinary success -- pre-connect, this would fall straight through to the twin
    # branch below. Never refuses a genuinely twin-configured station.
    gate = _live_or_refuse(hub, "startWrite")
    if gate is not None:
        await hub.log("Lithography write REFUSED: " + gate.message)
        raise gate
    if getattr(hub.host, "litho_thread", None) and hub.host.litho_thread.is_alive():
        msg = "write already running"
        from ..commands.errors import BUSY
        raise CommandError(BUSY, msg,
                           {"run_id": hub._in_flight_run_id(), "kind": "write"},
                           http={"error": msg})
    # THE SINGLE-WRITER CLAIM, TAKEN BEFORE THE FIRST AWAIT (plan Sec E.2).
    #
    # The check above is not enough on its own: this path then AWAITS the AC-guard
    # crosspoint read (a bridge read, seconds long) before it starts the thread, so
    # two requests interleaving in that window BOTH passed `is_alive()` and both went
    # on to drive the tip. A human clicking twice rarely hits it; a script issuing
    # requests back to back hits it easily, which is why FLEX forced the fix.
    #
    # `claim_run` is synchronous -- no await between its check and its set -- so the
    # second caller is refused HERE rather than discovering the collision on the
    # instrument. Once a run really starts, `claim.run_id` is stamped and the pump's
    # finally owns the release; every path that starts nothing releases it below.
    claim = _claim(hub, "write")
    try:
        return await _litho_write_body(hub, body, claim, intake=intake, since=since)
    finally:
        if claim.worker is None:          # nothing started: nobody else will free it
            hub.release_claim(claim)


async def _litho_write_body(hub, body: dict, claim, *, intake: WriteIntake,
                            since: tuple | None = None) -> dict:
    """The body of `litho_write`, run while holding the single-writer claim. Split
    out only so the claim brackets the whole thing in one `try/finally`.

    `intake` is how the carrier asked (see `WriteIntake`): an absolute setpoint (the
    default) is used as given, a free-air one becomes `free_air + setpoint` before the
    guard runs, on both doors. `command_guard.check_write` is the SAME call either way -- one choke
    point, both doors, no exceptions.
    """
    from ...litho.compiler import Pattern, demo_pattern
    if since is None:
        since = _stop_seq(hub)
    from ..host.litho_worker import (
        _hw_litho_worker,
        _litho_pump,
        _twin_litho_worker,
    )
    if not getattr(hub, "pattern", None):
        hub.pattern = demo_pattern()

    intake.refuse_selection(body)
    _refuse_without_keeper(hub)

    # SIZE WITHOUT COMMITTING. Every guard below this line -- the contact-only AC
    # refusal, the parameter validation -- can still refuse the call, and a refusal
    # must leave the instrument and the bookkeeping exactly as it found them.
    # `objects: "all"` CLEARS the cockpit's "already written" set, and doing that
    # before the AC guard meant the one refusal an operator actually hits on hardware
    # (write, get told to engage in contact, engage, write again) silently wiped the
    # marks -- so the panel's default write then re-dosed the whole design.
    # `commit_selection` below runs once every guard has passed.
    todo, skipped, _cleared = select_objects(hub, body, commit=False)
    if not todo:
        await hub.log("Nothing new to write. Everything in the design has "
                      "already been written; use Write all to repeat it.")
        msg = "nothing new to write"
        raise CommandError(REFUSED, msg,
                           {"reason": "already_written",
                            "objects": len(hub.pattern.objects)},
                           http={"error": msg, "reason": "already_written",
                                 "objects": len(hub.pattern.objects)})

    # ---- INSTRUMENT-MODEL COMMAND GUARD (docs/INSTRUMENT_MODEL_SPEC.md secs 2/10,
    # 12) -- BEFORE any worker thread is started, on BOTH doors, no exceptions.
    ask = intake.setpoint(body)
    amp_gain_raw = body.get("amp_gain")
    amp_gain = (_number(body, "amp_gain", None,
                        verb=intake.verb)
               if amp_gain_raw is not None else None)
    twin_speed = _twin_speed(hub, body, verb=intake.verb)
    # GUARD (safety-critical): litho is contact-only. On HARDWARE, refuse if the
    # instrument is in AC/tapping mode -- BEFORE the SUM ceiling, so a wrong-mode
    # write is named ac_mode_active rather than masked by a SUM read the AC engage
    # never reaches. Off-loaded to a thread (blocking crosspoint read).
    if _real_bridge(hub) is not None and await asyncio.to_thread(ac_mode_active, hub):
        await hub.log("Lithography REFUSED: the instrument is in AC/tapping mode "
                      "and lithography is contact-only. Do a contact engage "
                      "(Approach, mode=contact) before writing.")
        msg = ("litho is contact-only; the instrument is in AC/tapping mode. Do "
               "a contact engage before writing.")
        raise CommandError(REFUSED, msg, {"reason": "ac_mode_active"},
                           http={"error": msg, "reason": "ac_mode_active"})
    # 13.1 item 27: withdrawn -> measure free air here (after the AC refusal, so a
    # contact baseline is never read with the instrument in AC), then convert a
    # free-air setpoint against the fresh baseline.
    await _ensure_write_free_air(hub)
    if _real_bridge(hub) is not None:
        # _hw_litho_worker snapshots hub.hw_cal into LithographyRun.cal; the Cypher
        # rescales the XY LVDT sensitivities on every motor move (service/xy_cal.py).
        # The flag is cleared first because every park leaves it set; an abort or
        # fault landing during the refresh's bridge reads must refuse the write.
        hub.host._hw_litho_abort.clear()
        await hub.refresh_xy_cal("write")
        await _refuse_if_aborted(hub, "the XY calibration was read")
    else:
        hub.host._hw_litho_abort.clear()     # same point as the rig: a stop after this one counts
    setpoint = _absolute_setpoint(hub, ask)
    # 0.9*SUM ceiling + SUM floor with a FRESH DetectorSum read (12.1), applied
    # BEFORE check_write's setpoint-band check so a setpoint past 0.9*SUM is named
    # setpoint_exceeds_sum rather than merely out-of-band. `setpoint` here is the
    # ABSOLUTE applied value (free_air + setpoint, or the setpoint as given).
    from ..host.engage import arm_interlocks, check_engage_sum_ceiling
    check_engage_sum_ceiling(hub, setpoint)
    from ..command_guard import check_write
    check_write(hub.host.instrument_model, hub.pattern,
               [getattr(o, "name", None) for o in todo], setpoint, amp_gain,
               getattr(hub.host, "free_air", None), now=time.time(),
               fault=getattr(hub, "fault", None),
               setpoint_relative=ask.value_v if ask.relative else None)

    # WRITE PARAMETERS, STAMPED AT WRITE START (docs/INSTRUMENT_MODEL_SPEC.md sec
    # 4/10). `hub.last_write_params` lets the cockpit hydrate the last attempted
    # values through GET /api/instrument; `hub._litho_write_ctx` is the full context
    # the record (writes/json/<stem>.json) is persisted from when the run closes --
    # captured HERE, before the worker starts, so an abort/exception still records
    # exactly what the write was armed with. `setpoint` is the ABSOLUTE applied value;
    # `setpoint_relative` is the force above free air (derived when the ask was absolute).
    _fa = getattr(hub.host, "free_air", None)
    if ask.relative:
        _rel = ask.value_v
    else:
        _rel = (setpoint - _fa.value_v) if _fa is not None else None
    _model = getattr(hub.host, "instrument_model", None)
    hub.host.last_write_params = {"setpoint_relative": _rel, "setpoint": setpoint,
                            "amp_gain": amp_gain}
    hub._litho_write_ctx = {
        "amp_gain": amp_gain,
        "setpoint": setpoint,
        "setpoint_relative": _rel,
        "free_air_v": (_fa.value_v if _fa is not None else None),
        "free_air_ts": (_fa.ts if _fa is not None else None),
        "model_hash": (_model.snapshot_hash() if _model is not None else None),
    }

    run_pattern = Pattern(name=hub.pattern.name, objects=todo)
    q = _queue.Queue()

    # REAL backend: drive the write over the same bridge the scan uses, as a
    # capturing pass driven by LithographyRun. hardware_arc is None off-instrument,
    # so this branch is never taken in sim -- exactly like the scan path -- and the
    # twin path below stays byte-identical there.
    if _real_bridge(hub) is not None:
        # The contact-only AC-mode guard already ran above (before the SUM ceiling).
        # `setpoint`/`amp_gain` were already resolved (and passed the command
        # guard) above -- the ABSOLUTE setpoint either way (free_air + setpoint, or
        # as given). Tip-bias amplifier gain: the
        # DAC drives V_tip / amp_gain so the tip sees V_tip after the external
        # amplifier.
        arm_interlocks(hub)              # runtime interlocks armed for the whole write
        # Not cleared here: the flag was cleared at write start, and a park since then
        # (the SUM-ceiling and interlock reads block this thread while the watchdog
        # or telemetry guard can fire) must stop the write, not be erased.
        try:
            await _refuse_if_aborted(hub, "the write guards ran")
            # The flag above was cleared part-way through this start; the counters
            # cover the reads before that (AC guard, free air) as well. Nothing
            # between here and `litho_thread.start()` awaits.
            await _refuse_if_stopped(hub, since, "the write was starting")
        except CommandError:
            from ..host.engage import disarm_interlocks
            disarm_interlocks(hub)
            raise
        commit_selection(hub, body)      # every guard passed: the write is happening
        hub.host.hw_litho_capture = {}                        # fresh capture accumulator
        # The run re-resolves a relative setpoint against each object's free air;
        # An absolute setpoint is used as sent (docs/PER_OBJECT_FREE_AIR_SPEC.md).
        run_setpoint = ask
        hub.host.litho_thread = threading.Thread(
            target=_hw_litho_worker, args=(q, run_pattern, run_setpoint, amp_gain),
            daemon=True)
        hub.host.litho_thread.start()
        # Register the abortable handle so `safe_park` can abort+join this worker
        # BEFORE any Z command (Sec 3 step 1). Cleared by the pump's finally.
        hub.host.litho_worker = LithoWorkerHandle(hub.host.litho_thread,
                                             abort_event=hub.host._hw_litho_abort)
        # Allocate the run id BEFORE the pump, so the id in the response is the one
        # the pump closes (plan Sec A.1). `objects` is THIS run's submitted count --
        # the denominator getStatus's progress needs. The whole pattern is the wrong
        # denominator: a write of one object out of forty would read as 2.5% and
        # never pass it.
        run_id = hub.next_run_id("write")
        hub.runs.annotate(run_id, stats={"objects_submitted": len(todo)})
        # Hand the claim to the worker: from here its validity is judged by THAT
        # thread's liveness, not by a hub field a later run may replace.
        claim.hand_over(run_id, hub.host.litho_thread)
        hub._litho_pump_task = P.spawn_untraced(
            _litho_pump(q, run_id, claim, marks=_marks(hub, todo)))
        await hub.log(
            f"Lithography started (hardware): {len(todo)} "
            f"object{'' if len(todo) == 1 else 's'}"
            + (f", {skipped} already written and skipped" if skipped else ""))
        return {"ok": True, "writing": len(todo), "skipped": skipped,
                "backend": "hardware", "run_id": run_id}

    # SIMULATION backend (default): the same worker as the rig, on the twin plugin.
    # LOUD twin-fallback: if the operator selected the hardware backend but no bridge
    # is attached (hub.hardware_arc is None -- start_bridge.py wasn't run, or its dial
    # never reached THIS service), the write silently went to the twin. That cost a
    # whole debugging session once; make it unmistakable in the log.
    if getattr(hub.host, "backend", None) == "hardware":
        await hub.log("WARNING: backend=hardware but NO bridge is attached "
                      "(hub.hardware_arc is None) -- this write is going to the "
                      "TWIN, not the instrument. Check the service console for a "
                      "'[hardware] bridge LIVE' line, and (re)run "
                      "scripts/start_bridge.py.")
    # A stop during any await of this start refuses it (the flag was cleared before the
    # guards, the counters cover the reads); nothing between here and `litho_thread.start()`
    # awaits, and the interlocks are not armed yet.
    await _refuse_if_aborted(hub, "the write guards ran")
    await _refuse_if_stopped(hub, since, "the write was starting")
    commit_selection(hub, body)          # every guard passed: the write is happening
    arm_interlocks(hub)                  # runtime interlocks armed for the whole write
    # The commanded Output.A of every stroke accumulates here (13.1 item 16), which is
    # what makes writes/npz/<stem>.npz non-empty for a twin write.
    hub.host.hw_litho_capture = {}
    from ..twin_plugin import plugin as twin_plugin
    twin_plugin(hub)                     # built here, not on the worker, so a failure is loud
    # As on the rig, the worker gets the setpoint as asked: the plugin resolves a relative
    # one per object against its own free air.
    hub.host.litho_thread = threading.Thread(
        target=_twin_litho_worker, args=(q, run_pattern, ask, amp_gain, twin_speed),
        daemon=True)
    hub.host.litho_thread.start()
    # Register the abortable handle so `safe_park` can abort+join this worker BEFORE
    # any Z command (Sec 3 step 1). Cleared by the pump's finally.
    hub.host.litho_worker = LithoWorkerHandle(hub.host.litho_thread,
                                              abort_event=hub.host._hw_litho_abort)
    run_id = hub.next_run_id("write")
    hub.runs.annotate(run_id, stats={"objects_submitted": len(todo)})
    claim.hand_over(run_id, hub.host.litho_thread)      # the worker owns it from here
    hub._litho_pump_task = P.spawn_untraced(
        _litho_pump(q, run_id, claim, marks=_marks(hub, todo)))
    await hub.log(
        f"Lithography started: {len(todo)} object{'' if len(todo) == 1 else 's'}"
        + (f", {skipped} already written and skipped" if skipped else ""))
    return {"ok": True, "writing": len(todo), "skipped": skipped, "run_id": run_id}


async def start_write(hub, params: dict | None = None) -> dict:
    """The WIRE door: `{run_id, est_s, objects, state}`, start+poll (plan Sec F.1).

    Validation happens BEFORE anything is claimed or armed, so a refusal is a refusal
    of the CALL -- a script must never be handed a `run_id` for a run that was never
    going to start. `objects` defaults to `"all"`: a script that has just loaded a
    design and says `startWrite` means "write this design", and inheriting the
    cockpit's per-session "already written" set would silently write a subset of what
    the caller asked for.
    """
    body = dict(params or {})
    sel = body.get("objects")
    if sel is None:
        body["objects"] = "all"
    elif isinstance(sel, str) and sel.strip().lower() != "all":
        msg = (f"startWrite: objects must be \"all\" or a list of object indices or "
               f"ids, got {sel!r}")
        raise CommandError(INVALID_PARAMS, msg, {"param": "objects"})
    elif not isinstance(sel, (str, list, tuple)):
        msg = (f"startWrite: objects must be \"all\" or a list of object indices or "
               f"ids, got {type(sel).__name__}")
        raise CommandError(INVALID_PARAMS, msg, {"param": "objects"})
    # Validate the numbers here too, so `setpoint: "loud"` is a -32602 rather than a
    # backend-branch surprise. `flex_intake.setpoint` also refuses a missing setpoint here,
    # before anything is claimed.
    flex_intake.setpoint(body)
    _number(body, "amp_gain", 1.0, verb="startWrite")
    _twin_speed(hub, body, verb="startWrite")
    if not getattr(hub, "pattern", None) or not hub.pattern.objects:
        msg = ("startWrite: no pattern is loaded. Call loadPattern first -- the wire "
               "door never substitutes the demo design.")
        raise CommandError(REFUSED, msg, {"reason": "no_pattern"})
    # VALIDATE deadman_s here (a typo must be -32602 before anything starts) but do
    # not APPLY it yet: re-arming on a call that then gets refused -- busy, AC mode,
    # backend mismatch -- would silently extend the dead-man of whatever IS running,
    # on behalf of a request that did nothing.
    if body.get("deadman_s") is not None:
        _check_deadman(body)
    # `todo` is recomputed inside `litho_write` under the claim; this pre-count is
    # only for the estimate, and any refusal it would raise is raised there too.
    todo, _skipped, _cleared = select_objects(hub, dict(body), commit=False)
    segments = write_segments(hub, todo)
    # The estimate runs in a thread and can take seconds. An Abort, Withdraw or park
    # that arrives meanwhile is served at once, and it must also stop this call: the
    # operator stopped the instrument, so the write does not start behind it.
    stop_seq = _stop_seq(hub)
    est = await write_est_s_async(hub, todo)
    if _stop_seq(hub) != stop_seq:
        msg = ("write start interrupted: an abort, park, withdraw or scan stop was "
               "signalled while the write estimate was computed")
        await hub.log("Lithography REFUSED: " + msg)
        raise CommandError(REFUSED, msg, {"reason": "aborted_during_start"})
    res = await litho_write(hub, body, intake=flex_intake, since=stop_seq)
    if body.get("deadman_s") is not None:
        _rearm_deadman(hub, body)                  # the write is real: arm for it
    return {"run_id": res.get("run_id"), "est_s": est,
            "objects": int(res.get("writing") or 0), "segments": segments,
            "state": "writing", "backend": hub.backend_label()}


def _check_deadman(body: dict) -> None:
    """Validate `deadman_s` WITHOUT applying it (see `_rearm_deadman` for why the two
    are separate). A NUMBER out of range is clamped later; a non-number is a typo, and
    silently substituting a default would hand a script a dead-man it never asked
    for."""
    value = body.get("deadman_s")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = (f"startWrite: deadman_s must be a number of seconds, got "
               f"{type(value).__name__}")
        raise CommandError(INVALID_PARAMS, msg, {"param": "deadman_s"})


def _rearm_deadman(hub, body: dict) -> None:
    """`startWrite {deadman_s}` -- re-arm the commander's dead-man for a long write.

    Validated, then clamped, exactly as `acquireControl` does and for the same reason:
    a NUMBER outside [5, 300] is a request the provider can honour approximately
    (clamping fails safe), but `deadman_s: "abc"` is a typo, and silently substituting
    a default would hand a script a dead-man it never asked for.

    This does NOT make a long write safe on its own. The dead-man is refreshed by
    COMMAND-socket requests, and a start+poll write is polled on the READ socket,
    which refreshes nothing by design -- so a write longer than `deadman_s` still has
    to heartbeat (docs/FLEX_PROVIDER.md, and check 09 (e) in the rig runbook).
    """
    _check_deadman(body)
    token = body.get("token")
    if not (isinstance(token, str) and token):
        return
    # REFUSED, not silently ignored, when the holder is an OPERATOR record: a human at
    # the panel has no dead-man and a script must not be able to give one. Returning
    # None there and saying nothing would let a script believe it had armed a timer
    # that does not exist.
    if hub.rearm_deadman(token, body["deadman_s"]) is None:
        msg = ("startWrite: deadman_s could not be armed -- the instrument is held by "
               "an operator, who has no dead-man. Drop deadman_s, or acquire control.")
        raise CommandError(REFUSED, msg, {"param": "deadman_s",
                                          "reason": "not_a_script_token"})


# =============================================================== abortWrite ======

async def abort_write(hub) -> dict:
    """Stop the write AND get the tip off the surface.

    ABORT/safety path -> the UNIFIED safe_park (stop + verified retract). Operator
    decision (2026-08-28 audit fix 4): the ABORT/safety paths all go through the one
    safe_park primitive, not a bespoke AFM_Abort + separate withdraw -- a litho abort
    must get the tip OFF the surface, not merely stop the write with a live Z loop.
    (A plain "Stop" -- scan/abort -- also retracts the tip now; every Stop withdraws,
    and there is no stop-and-hold path, docs/STOP_RETRACT_SPEC.md. But it stays on the
    NORMAL scan path; only ABORT routes here to the unified safe_park.)

    safe_park drives the out-of-band retract (hardware: AFM_SafePark over COM/TCP;
    twin: the plugin's own withdraw, verified from its Z) and sets every cooperative abort
    EVENT, so the write worker (which waits on hub._hw_litho_abort) stops. The twin plugin
    is told to abort by the same signal (`signal_run_abort`).

    TOKEN-FREE on the wire, like `abort` and `safePark`: an emergency stop must never
    be blocked by arbitration, from either side.
    """
    from ..host.safe_park import STOP_LITHO_RUN, _signal_stop, request_park

    _signal_stop(hub, STOP_LITHO_RUN)
    res = await request_park(hub, "litho abort")
    return {"ok": True, "backend": res.get("backend"), "safe_park": res}


# ========================================================= the inline trace ======

def remember_trace(hub, run_id: str | None, written: list) -> None:
    """Keep this run's written path + tip voltage in memory, bounded, so
    `getWriteResult(inline=true)` can hand a script the current-along-path trace
    without a file read on the REP thread.

    Goal 2 is "correlate the conductance jump against tip position", and the per-
    object path the pump already accumulated IS that trace's x-axis. It is decimated
    by the worker (`path_um[::4]`) and decimated again here, and only the last few
    runs are kept -- a reply that rides one REQ/REP round trip has to be bounded
    somewhere, and hoping the design is small is not a bound.
    """
    if not run_id:
        return
    traces = getattr(hub, "_write_traces", None)
    if traces is None:
        traces = hub._write_traces = {}
    entries = [e for e in (written or []) if (e.get("path_um") or [])]
    # The per-object budget is the SMALLER of the fixed cap and this run's fair share,
    # so both bounds hold at once and neither depends on how many objects there are.
    per_object = (max(1, TRACE_MAX_POINTS_PER_RUN // len(entries)) if entries
                  else TRACE_MAX_POINTS_PER_OBJECT)
    budget = min(TRACE_MAX_POINTS_PER_OBJECT, per_object)
    objects = []
    for entry in written or []:
        path = entry.get("path_um") or []
        volts = entry.get("v_written") or []
        step = max(1, -(-len(path) // budget))
        obj = {
            "name": entry.get("name"),
            "voltage": entry.get("voltage"),
            "length_um": entry.get("length_um"),
            "t_end": entry.get("t"),
            "decimate": step,
            "path_um": [[float(x), float(y)] for x, y in path[::step]],
            "v_tip": [float(v) for v in volts[::step]],
        }
        # Tagged by `_litho_finish` for the one object an abort cut mid-stroke
        # -- `written` never carries this, so only that spliced-in entry ever
        # sets it. A script reading `getWriteResult(inline=True).trace` needs
        # this to tell a real completed object from the partial one.
        if entry.get("partial"):
            obj["partial"] = True
        objects.append(obj)
    traces[run_id] = objects
    while len(traces) > TRACE_KEEP_RUNS:
        try:
            traces.pop(next(iter(traces)))
        except (StopIteration, KeyError):          # a concurrent pop got there first
            return


def trace_for(hub, run_id: str | None):
    """This run's remembered trace, or None. A pure memory read."""
    if not run_id:
        return None
    return (getattr(hub, "_write_traces", None) or {}).get(run_id)
