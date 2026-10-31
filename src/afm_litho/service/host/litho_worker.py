"""The litho workers and the pump that drains them.

`commands.litho_cmds` starts a
worker thread (`_hw_litho_worker` on the instrument, `_twin_litho_worker` on the twin)
that runs the pattern and posts progress into a queue; the async `_litho_pump` drains
that queue onto the WebSocket, fuses each finished pass into the Surface and persists
the pass. The workers and the pump read the process hub.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue as _queue
import time

import numpy as np

from ... import clock
from ...run import Placement
from .. import keeper_link
from . import litho_cmds
from ..hub import hub
from ..step_notes import StepNotes, make_notes


def _atomic_write_text(path, text: str) -> None:
    """Write `text` to `path` via a temp file + `os.replace`, so a reader never
    sees a half-written forensic file (docs/INSTRUMENT_MODEL_SPEC.md sec 5: the
    sidecar is "fully written ... before the session record"). The rename is
    atomic on the same filesystem; the temp file sits beside the target."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, "utf-8")
    os.replace(tmp, path)


def _build_write_sidecar(write_ctx: dict, *, aborted: bool,
                         abort_reason: str | None) -> dict:
    """The write-record sidecar dict (docs/INSTRUMENT_MODEL_SPEC.md sec 5/10):
    amp_gain, the absolute setpoint applied and the relative value, the free-air
    baseline and its timestamp, the model snapshot hash, whether the write aborted
    (with a reason), and on hardware `object_setpoints`: one entry per free-air re-read,
    i.e. per object, not per written segment (same-source hover rows add none). One
    builder, so the worker's pre-withdraw flush and `_litho_finish` write
    byte-identical sidecars."""
    sidecar = {
        "amp_gain": write_ctx.get("amp_gain"),
        "setpoint": write_ctx.get("setpoint"),
        "setpoint_relative": write_ctx.get("setpoint_relative"),
        "free_air_v": write_ctx.get("free_air_v"),
        "free_air_ts": write_ctx.get("free_air_ts"),
        "model_hash": write_ctx.get("model_hash"),
        "aborted": aborted,
    }
    # Hardware only: each object's own free-air read and the setpoint it was written at
    # (one entry per object's free-air re-read, not per segment).
    if write_ctx.get("object_setpoints") is not None:
        sidecar["object_setpoints"] = list(write_ctx["object_setpoints"])
    if write_ctx.get("run") is not None:
        sidecar["run"] = write_ctx["run"]       # {execution_id, path} or "no run: <reason>"
    if aborted:
        sidecar["abort_reason"] = abort_reason or "aborted"
    return sidecar


def _flush_forensics_sync(runtime, cap: dict, write_ctx: dict, *,
                          aborted: bool, abort_reason: str | None) -> tuple:
    """Write the forensic npz + sidecar for a write, SYNCHRONOUSLY, and return
    `(stem, npz_ref)`.

    Called from the hardware worker thread AFTER the withdraw (spec sec 5, 13.1 item
    17): no file work stands between a stop and the retract. It saves the npz with
    `save_capture_now`, on this thread, never through the raw writer.
    `_litho_finish` reuses the returned stem so the session record and the forensic
    files share one name. The sidecar goes through the same atomic temp+rename as the
    completion path.
    """
    from .. import stems
    stem = stems.make_stem(runtime, "litho")
    _sidecar_path = stems.dir_for(runtime, "litho", "json") / f"{stem}.json"
    _atomic_write_text(_sidecar_path, json.dumps(
        _build_write_sidecar(write_ctx, aborted=aborted,
                             abort_reason=abort_reason), indent=2))
    ref = None
    if cap:
        from ..captures import save_capture_now
        ref = save_capture_now(runtime, stem, cap,
                               out_dir=stems.dir_for(runtime, "litho", "npz"))
    return stem, ref




def _transit_policy(h=None):
    """The transit policy of `h` (default: the process hub), created on first use."""
    from ...litho.transit import TransitPolicy
    h = h or hub
    if not getattr(h, "transit", None):
        h.transit = TransitPolicy()
    return h.transit


def _litho_frame(h=None):
    """The frame lithography is written in.

    The ANGLE comes from the imaging scan, exactly as the vendor takes it
    from `RVW[%ScanAngle]`: you draw a pattern on a rotated image, so the
    pattern is written in that rotated frame or it lands somewhere else. The
    sizes and offsets are the litho panel's own parameters, and default to
    zero, which is the identity for the twin's centred design coordinates.

    Setting `angle_deg` explicitly through /api/litho/frame pins it and
    stops it following the scan. That is a deliberate escape hatch, not the
    normal case.

    `h` is the hub to read (default: the process hub).
    """
    from ...litho.frame import LithoFrame
    h = h or hub
    cfg = dict(getattr(h, "litho_frame", None) or {})
    if cfg.get("angle_deg") is None:
        cfg["angle_deg"] = h.host.twin_scan_angle_deg or 0.0
    return LithoFrame.from_dict(cfg)




# The frames whose `path_um` / `v_written` the cockpit keeps per stroke and redraws
# point by point on every overlay repaint.
_UI_TRACE_TYPES = frozenset(
    {"litho_object_done", "litho_object_partial", "litho_transit"})
# Same tolerance as the toolpath preview: 12.5 nm is below what the canvas resolves.
_UI_TRACE_TOL_UM = 0.0125
# Bias quantum: a point is kept wherever the bias crosses into another level of this
# size, so a modulated (chiral) wire still colours correctly.
_UI_TRACE_V_TOL = 0.1
# Hard ceiling on the points of one broadcast stroke, whatever the shape.
_UI_TRACE_MAX_POINTS = 4000
# Share of the ceiling the geometry may use when the bias is modulated; the rest is
# held for the bias points.
_UI_TRACE_GEOM_SHARE = 0.75
# Ceiling on point-to-chord distance evaluations for one stroke (about 5 ns each).
# Past it the simplification stops and a uniform stride fills in.
_UI_TRACE_WORK_BUDGET = 30_000_000
# Rows converted per numpy call. `np.asarray` on a Python list holds the GIL for the
# whole call; chunks this size let the service loop run between them.
_UI_TRACE_CHUNK = 10_000
_ui_trace_failure_logged = False


def _ui_array(seq, width: int | None):
    """`seq` as a finite float array, converted in GIL-yielding chunks. `width` is
    the expected column count (None for a flat list)."""
    n = len(seq)
    parts = [np.asarray(seq[i:i + _UI_TRACE_CHUNK], dtype=float)
             for i in range(0, n, _UI_TRACE_CHUNK)]
    a = np.concatenate(parts) if len(parts) > 1 else parts[0]
    want = (n,) if width is None else (n, width)
    if a.shape != want:
        raise ValueError(f"array has shape {a.shape}, expected {want}")
    # Finite and bounded, so the distance arithmetic below cannot overflow.
    return np.nan_to_num(a, nan=0.0, posinf=1.0e6, neginf=-1.0e6)


def _ui_rdp_keep(p, tol_um: float, max_keep: int, budget: int):
    """Keep-mask of a Douglas-Peucker simplification for the UI trace, largest
    deviation first, so stopping early leaves the most significant points.

    Distance is to the chord SEGMENT, not its infinite line: a collinear
    out-and-back keeps its turnaround. Stops at `max_keep` kept points, or after
    `budget` distance evaluations; in the second case the result is topped up with
    a uniform stride, since deep detail may not have been reached.
    """
    import heapq
    n = len(p)
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep
    keep[0] = keep[-1] = True
    if n < 3 or max_keep <= 2:
        return keep
    x, y = np.ascontiguousarray(p[:, 0]), np.ascontiguousarray(p[:, 1])

    def farthest(a: int, b: int):
        if b <= a + 1:
            return 0.0, -1
        rx = x[a + 1:b] - x[a]
        ry = y[a + 1:b] - y[a]
        dx, dy = x[b] - x[a], y[b] - y[a]
        L2 = dx * dx + dy * dy
        if L2 > 1e-24:
            t = np.clip((rx * dx + ry * dy) / L2, 0.0, 1.0)
            rx = rx - t * dx
            ry = ry - t * dy
        d2 = rx * rx + ry * ry
        i = int(np.argmax(d2))
        return float(d2[i]), a + 1 + i

    tol2 = tol_um * tol_um
    kept, work = 2, n
    d2, i = farthest(0, n - 1)
    heap = [(-d2, 0, n - 1, i)]
    while heap and kept < max_keep:
        neg, a, b, i = heapq.heappop(heap)
        if -neg <= tol2:
            return keep                      # every remaining chord is within tol
        keep[i] = True
        kept += 1
        work += b - a
        if work > budget:
            fill = np.round(np.linspace(0, n - 1, max_keep - kept + 2)).astype(int)
            keep[fill] = True
            return keep
        for lo, hi in ((a, i), (i, b)):
            d2, j = farthest(lo, hi)
            if j >= 0 and d2 > tol2:
                heapq.heappush(heap, (-d2, lo, hi, j))
    return keep


def _ui_run_ends(p):
    """Indices of `p` that are not strictly inside an exactly straight run: one
    vectorised pass that reduces an axis-aligned fill to its row ends. Exact
    collinearity only, so nothing the tolerance should judge is decided here; a
    reversal (the dot product is not positive) is never dropped.
    """
    n = len(p)
    if n < 3:
        return np.arange(n)
    d = np.diff(p, axis=0)
    cross = d[:-1, 0] * d[1:, 1] - d[:-1, 1] * d[1:, 0]
    dot = d[:-1, 0] * d[1:, 0] + d[:-1, 1] * d[1:, 1]
    live = np.ones(n, dtype=bool)
    live[1:-1] = ~((cross == 0.0) & (dot > 0.0))
    return np.flatnonzero(live)


def _ui_bias_keep(va, step: float):
    """Indices that carry a bias change, in one vectorised pass: the first sample of
    each new `step`-sized level, the sample before it when the change is an edge
    (so a step stays sharp), and the minimum and maximum.
    """
    q = np.rint(va / step)
    mark = np.zeros(len(va), dtype=bool)
    mark[1:] = q[1:] != q[:-1]
    edge = np.abs(np.diff(va)) >= step
    mark[:-1] |= edge & mark[1:]
    ends = np.array([int(np.argmin(va)), int(np.argmax(va))])
    return ends, np.flatnonzero(mark)


def _ui_trace_frame(msg: dict) -> dict:
    """The copy of a trace frame that goes to the WebSocket: `path_um` simplified by
    Douglas-Peucker, `v_written` taken at the SAME indices, `surface_samples` left
    out. `msg` and its lists are never modified -- the pump's records, sidecars,
    `written_accum` and surface ingest read the full-resolution original. Any other
    frame type is returned as the same object.
    """
    if msg.get("type") not in _UI_TRACE_TYPES:
        return msg
    out = {k: val for k, val in msg.items() if k != "surface_samples"}
    path = msg.get("path_um")
    if not path:
        return out
    n = len(path)
    p = _ui_array(path, 2)
    v = msg.get("v_written")
    has_v = isinstance(v, (list, tuple)) and len(v) > 0
    ends = changes = None
    if has_v:
        # A `v_written` of another length stays registered by index: the bias test
        # runs over the points both arrays have.
        m = min(n, len(v))
        va = _ui_array(v[:m], None)
        if m >= 3 and float(np.ptp(va)) >= _UI_TRACE_V_TOL:
            ends, changes = _ui_bias_keep(va, _UI_TRACE_V_TOL)
    cap = _UI_TRACE_MAX_POINTS
    geom_cap = cap if changes is None else int(cap * _UI_TRACE_GEOM_SHARE)
    runs = _ui_run_ends(p)
    keep = np.zeros(n, dtype=bool)
    keep[runs[_ui_rdp_keep(p[runs], _UI_TRACE_TOL_UM, geom_cap,
                           _UI_TRACE_WORK_BUDGET)]] = True
    if changes is not None:
        keep[ends] = True
        changes = changes[~keep[changes]]
        room = cap - int(np.count_nonzero(keep))
        # More bias changes than the ceiling has room for are taken at a uniform
        # stride: the modulation is then ALIASED on screen. Its minimum and maximum
        # are kept, so the colour range is still the true one.
        if len(changes) > room:
            changes = changes[np.round(
                np.linspace(0, len(changes) - 1, max(room, 0))).astype(int)]
        keep[changes] = True
    idx = np.flatnonzero(keep)
    out["path_um"] = [path[i] for i in idx]
    if has_v:
        nv = len(v)
        out["v_written"] = [v[i] for i in idx if i < nv]
    return out


async def _broadcast_ui(msg: dict) -> None:
    """Broadcast `msg`, with trace frames decimated for the panel. The decimation
    runs in a thread so Abort, Withdraw and telemetry keep being served; the pump
    awaits it, so frames reach the socket in queue order. A decimation failure sends
    the original frame: the display must never stop the pump, which is what closes
    the run record and releases the write claim.
    """
    global _ui_trace_failure_logged
    ui = msg
    if msg.get("type") in _UI_TRACE_TYPES:
        try:
            ui = await asyncio.to_thread(_ui_trace_frame, msg)
        except Exception as exc:                            # noqa: BLE001
            ui = msg
            if not _ui_trace_failure_logged:
                _ui_trace_failure_logged = True
                try:
                    await hub.log(
                        f"Trace decimation failed for {msg.get('type')} "
                        f"({type(exc).__name__}: {exc}); sending it undecimated.")
                except Exception:                           # noqa: BLE001,S110
                    pass
    await hub.broadcast(ui)




def _write_gain(amp_gain) -> float:
    """The tip-bias amplifier gain the write drives with: the DAC gets V_tip / gain, and a
    missing or zero gain means 1 (the commanded Output.A record and the run both use this)."""
    return float(amp_gain) if amp_gain else 1.0


def _emit_write_run(q, run_pattern, segments, setpoint, amp_gain, frame):
    """Build and store this write's run (service/run_store.py) before the first stroke, and
    note its reference in the write context. Returns the stored Run, or None. Failure is logged
    and recorded as "no run: <reason>"; the write goes on exactly as without it."""
    ctx = getattr(hub, "_litho_write_ctx", None)
    try:
        from ...litho import design_file
        from .. import design_store, run_store
        runtime = hub.store.path.parent
        w = hub.host.world
        model = getattr(hub.host, "instrument_model", None)
        window = tuple(model.tip_voltage_window_v.value)
        compile_doc = run_store.compile_settings(
            frame, w.x_lvdt_sens, w.y_lvdt_sens, w.x_lvdt_offset, w.y_lvdt_offset)
        design_doc = design_file.pattern_to_document(
            run_pattern.objects, name=run_pattern.name, saved_at=design_store.now_stamp(),
            frame=design_store.current_frame(hub), pattern_name=run_pattern.name)
        out = run_store.emit_write_run(
            run_store.RunStore(run_store.store_dir(runtime)), runtime,
            design_doc=design_doc, compile_doc=compile_doc, design_name=run_pattern.name,
            objects=run_pattern.objects, segments=segments, setpoint=setpoint,
            amp_gain=_write_gain(amp_gain),
            platform=run_store.platform_name(hub.host.hardware_arc),
            placement=Placement("litho frame (angle follows the scan)", frame.fast_size_um,
                                frame.angle_deg, (frame.fast_offset_um, frame.slow_offset_um)),
            bias_window_v=window)
    except Exception as exc:                            # noqa: BLE001
        out = f"no run: {type(exc).__name__}: {exc}"
    if isinstance(ctx, dict):
        if isinstance(out, str):
            ctx["run"] = out
        else:
            ctx["run"], ctx["_stored_run"] = out.ref, out
    if isinstance(out, str):
        q.put({"type": "log", "t": time.time(), "msg": f"Run not stored ({out})"})
        return None
    return out.run


def _begin_step_notes(stored_run) -> StepNotes:
    """Journal this write's outcomes against its stored run (uncited when there is none), note
    the door's checks as passed, and start stamping the wire with the step in progress. If any
    of that fails the write goes on without notes."""
    try:
        notes = make_notes(stored_run, "write")
        notes.begin()
        ctx = getattr(hub, "_litho_write_ctx", None) or {}
        notes.outcome("check", passed=True, free_air_baseline_v=ctx.get("free_air_v"),
                      setpoint_v=ctx.get("setpoint"))
        return notes
    except Exception:  # noqa: BLE001 - notes must never stop a write
        return make_notes(None, "write")


def _log_to(q: _queue.Queue, msg: str) -> None:
    """A line for the cockpit's log, through the write's queue."""
    q.put({"type": "log", "t": time.time(), "msg": msg})


def _flush_record(q: _queue.Queue, abort_reason: str) -> None:
    """Write the forensic npz + sidecar of an aborted write (docs/INSTRUMENT_MODEL_SPEC.md sec 5,
    13.1 item 17), after the tip is withdrawn: no file work stands between a stop and the
    retract. `_litho_finish` reuses the stem this leaves on hub._litho_forensics, so the flush
    must finish before `litho_done`. Best-effort: a failure here only costs the record."""
    try:
        _runtime = hub.store.path.parent
        _cap = {k: list(vv) for k, vv in hub.host.hw_litho_capture.items() if vv}
        _wctx = dict(getattr(hub, "_litho_write_ctx", None) or {})
        _stem, _ref = _flush_forensics_sync(
            _runtime, _cap, _wctx, aborted=True, abort_reason=abort_reason)
        hub._litho_forensics = {"stem": _stem, "ref": _ref, "abort_reason": abort_reason}
        _log_to(q, f"Write record flushed: writes/json/{_stem}.json")
    except Exception as exc:                            # noqa: BLE001
        _log_to(q, f"Record flush failed: {exc}")


def _hardware_withdraw() -> dict:
    from ...hardware.hw_approach import hw_withdraw
    return hw_withdraw(hub.host.hardware_arc)


def _twin_withdraw() -> dict:
    """The twin plugin's withdraw; anything but a verified one is a failure, like the rig's."""
    from ...instrument import WithdrawTip
    from ..twin_plugin import plugin
    result = plugin(hub).withdraw(WithdrawTip())
    if not (result.done and result.verified):
        raise RuntimeError(result.reason or "the twin tip was not verified withdrawn")
    return {"verified": True}


def _end_of_run_withdraw(q: _queue.Queue, notes: StepNotes, withdraw=None) -> str | None:
    """Park the tip at the end of the write; returns the failure text, None if it worked.

    Each object's lift ends at the -10 deflection setpoint, so the tip is withdrawn once here
    (the SAME retract the cockpit Withdraw button uses) whether the run finished or aborted.
    `withdraw` is the backend's retract (the rig's by default).
    """
    from .safe_park import NOT_CONNECTED, before_igor_dials_in
    if before_igor_dials_in(hub):
        # Before Igor dials in nothing is sent and nothing is claimed (2026-10-28).
        notes.outcome("end", withdrawn=False, error=NOT_CONNECTED)
        _log_to(q, f"End-of-run withdraw: {NOT_CONNECTED}")
        return None
    try:
        notes.stamp_step("end")
        withdrawn = (withdraw or _hardware_withdraw)()
        hub.host.engaged = False           # tip off the surface -> a mode switch is now safe
        notes.outcome("end", withdrawn=True, result=withdrawn)
        _log_to(q, "End-of-run withdraw: Z parked at 0 V")
        return None
    except Exception as exc:                            # noqa: BLE001
        # THE RETRACT FAILED WITH THE TIP ON THE SURFACE AT BIAS. Any failure during a
        # hardware pass is a stop AND a verified withdraw, so hand the retract to the
        # out-of-band fail-safe (its own COM thread) and latch the fault so no further
        # pass starts on top of it.
        error = f"end-of-run withdraw failed: {exc}"
        notes.outcome("end", withdrawn=False, error=error)
        _log_to(q, f"End-of-run withdraw failed: {exc} -- escalating to safe park")
        try:
            from .safe_park import PARK_WATCHDOG, request_park
            request_park(hub, f"litho {error}", PARK_WATCHDOG)
        except Exception:                               # noqa: BLE001,S110
            pass
        try:
            if getattr(hub, "fault", None) is None:
                hub.fault = {"reason": "withdraw_failed", "at": time.time(),
                             "state": "fault", "detail": f"litho {error}"}
        except Exception:                               # noqa: BLE001,S110
            pass
        return error


def _hw_litho_worker(q: _queue.Queue, run_pattern, setpoint,
                     amp_gain: float = 1.0) -> None:
    """The write on the rig: the Asylum plugin, one per write (it keeps the run, so a stroke
    of the same object hovers instead of transiting), closed when the write ends."""
    from ...instrument.asylum import WRITE_CAPTURE, AsylumInstrument
    _run_write(q, run_pattern, setpoint, amp_gain,
               lambda: AsylumInstrument(
                   hub, amp_gain=_write_gain(amp_gain), service_owns_write=True,
                   setpoint_log=lambda msg: _log_to(q, msg)),
               WRITE_CAPTURE, _hardware_withdraw, close_instrument=True)


def _put_twin_transit(q: _queue.Queue, tip_before, step: dict) -> None:
    """The `litho_transit` the panel draws as the pen-up travel to an object's first point.
    The old twin reported the transit policy's mode for every move, hover or not."""
    from ...hardware.xy_transit import MOVE_SPEED_UM_S
    x0, y0 = tip_before
    x1, y1 = step["to_um"]
    distance = float(np.hypot(x1 - x0, y1 - y0))
    q.put({"type": "litho_transit", "mode": _transit_policy().mode,
           "distance_um": round(distance, 3), "seconds": round(distance / MOVE_SPEED_UM_S, 3),
           "peak_um_s": MOVE_SPEED_UM_S,
           "path_um": [[round(float(x0), 3), round(float(y0), 3)],
                       [round(float(x1), 3), round(float(y1), 3)]]})


def _twin_surface_samples(instrument, path) -> dict:
    """The stroke as a height profile for the Surface: the sample's height under the path,
    thinned to about 200 points."""
    pts = np.asarray(path, dtype=float)[::max(1, len(path) // 200)]
    return {"x_um": pts[:, 0].tolist(), "y_um": pts[:, 1].tolist(),
            "z_nm": instrument.sample.height_nm(pts[:, 0], pts[:, 1]).tolist()}


def _twin_litho_worker(q: _queue.Queue, run_pattern, setpoint, amp_gain: float = 1.0,
                       twin_speed: float = 1.0) -> None:
    """The write on the twin: the twin plugin of this sample, played at `twin_speed` (1 real
    time, 0 instant). It outlives the write, so a wire written stays in later scans."""
    from ...hardware.channels import DEFLECTION
    from ..twin_plugin import plugin

    def twin():
        instrument = plugin(hub)
        instrument.speed_factor = twin_speed
        return instrument
    _run_write(q, run_pattern, setpoint, amp_gain, twin, (DEFLECTION,), _twin_withdraw,
               close_instrument=False, cockpit_events=True)


def _run_write(q: _queue.Queue, run_pattern, setpoint, amp_gain, make_instrument, record,
               withdraw, *, close_instrument: bool, cockpit_events: bool = False) -> None:
    """Drive the write through the instrument plugin `make_instrument()` gives and translate
    each object into the litho_* messages the cockpit's tip-trace and written overlay read.
    Both backends run this one worker; `record` is what the plugin captures, `withdraw` the
    end-of-run retract.

    Runs in a worker thread. The plugin reuses the connected bridge (`hub.hardware_arc.link`)
    and `hub.hw_cal`, which litho_write re-reads just before this thread starts. path_um from
    compile_pattern is in lab microns (frame-mapped, calibration-independent); the plugin
    re-applies the REAL LVDT conversion from hw_cal.

    When a write stops, this order holds: the journal's abort note and the "end" step label
    (memory only), the withdraw, then the forensic record.

    `cockpit_events`: also send what the twin write has always sent the panel and the
    surface: `litho_pattern_armed`, a `litho_transit` per object, and the object's height
    samples (`surface_samples`, which the pump fuses into the Surface).

    `setpoint`: a `WriteSetpoint` (relative or absolute as the door received it), or a
    bare float meaning absolute; each object's setpoint is resolved from its own
    free-air read (docs/PER_OBJECT_FREE_AIR_SPEC.md).
    """
    from ...hardware.lithography_run import as_write_setpoint
    from ...hardware.xy_transit import MOVE_SPEED_UM_S
    from ...instrument import WriteObject
    from ...litho.compiler import compile_pattern, trace_indices
    from ...run.model import Setpoint
    w = hub.host.world
    ab = hub.host._hw_litho_abort
    aborted = False
    notes = StepNotes(None, "write")   # replaced once the run is stored
    run_error = None      # the object-loop exception text, carried into abort_reason
    gain = _write_gain(amp_gain)
    written = as_write_setpoint(setpoint)
    command_setpoint = Setpoint("relative" if written.relative else "absolute", written.value_v)
    # No pre-flushed forensic record yet for this run (13.1 item 17). Cleared here so
    # a prior aborted run's flush can never be reused by this one.
    hub._litho_forensics = None
    # SAFETY-CRITICAL: a hardware litho write SELF-ENGAGES (the tip is on the surface for the
    # whole pass). Set the mode-switch gate BEFORE driving, so a mode switch during the write
    # is REFUSED (a mid-engage feedback-channel + gain-sign flip is a crash). Cleared by the
    # end-of-run withdraw (verified retract -> a switch is safe again).
    hub.host.engaged = True

    def keep_capture(result, v) -> None:
        """Add this object's RAW capture (measured bias, Z, deflection, XSensor, YSensor as
        streamed; empty when no channel was ADC-routable) and the COMMANDED Output.A,
        `v_tip / amp_gain` per point as loaded into the DAC, to the pass capture."""
        for recorded in result.recorded:
            hub.host.hw_litho_capture.setdefault(recorded.channel, []).extend(
                recorded.volts.tolist())
        hub.host.hw_litho_capture.setdefault("output_a_cmd_v", []).extend(
            float(val) / gain for val in v)

    instrument = None
    try:
        # One plugin for the whole write: it keeps the run so a stroke of the same object
        # hovers instead of transiting. The service ran the guards and withdraws itself.
        instrument = make_instrument()
        # Each object's free air and setpoint go into the write record (the plugin appends
        # to this same list as it goes, so an aborted run's flush has them too).
        _ctx = getattr(hub, "_litho_write_ctx", None)
        if isinstance(_ctx, dict):
            _ctx["object_setpoints"] = instrument.object_setpoints
        # The live tip marker is fed from the always-on TELEM poll (the router's on_telem),
        # not armed here; the per-object CAPTURE/DONE flow is untouched.
        frame = _litho_frame()
        waves = compile_pattern(run_pattern, w.x_lvdt_sens, w.y_lvdt_sens,
                                w.x_lvdt_offset, w.y_lvdt_offset, frame=frame)
        if cockpit_events:
            q.put({"type": "litho_pattern_armed", "segments": waves["segment_count"],
                   "nan_count": waves["nan_count"], "points": int(len(waves["x"])),
                   "duration_s": round(waves["duration_s"], 2), "frame": waves["frame"]})
        stored_run = _emit_write_run(q, run_pattern, waves["segments"], setpoint, amp_gain,
                                     frame)
        notes = _begin_step_notes(stored_run)
        instrument.on_step = notes.outcome
        tip_before = [0.0, 0.0]
        if cockpit_events:
            def on_step(kind, make=None, **body):
                notes.outcome(kind, make, **body)
                if kind == "move":
                    _put_twin_transit(q, tip_before, make() if make else body)
            instrument.on_step = on_step
        speed_by_obj = {o.name: float(o.speed_um_s) for o in run_pattern.objects}
        prof_by_obj = {o.name: o.profile for o in run_pattern.objects}
        writing = None          # the source object of the last stroke
        for stroke, meta in enumerate(waves["segments"]):
            if ab.is_set():
                aborted = True
                notes.abort("abort flag set before the stroke")
                break
            # A dead keeper: the object being written finishes, the next one does not
            # start (decision 8). The end-of-run withdraw below still runs.
            if meta["object"] != writing:
                lost = keeper_link.keeper_lost(hub)
                if lost is not None:
                    aborted = True
                    run_error = f"write stopped before {meta['object']}: {lost}"
                    notes.abort(run_error)
                    _log_to(q, f"Lithography stopped: {run_error}")
                    break
                writing = meta["object"]
            notes.stroke = stroke
            path = np.asarray(meta["path_um"], dtype=float)
            v = np.asarray(meta["v"], dtype=float)
            name = meta["name"]
            # the display traces hold path points only, never the 0 V end sample
            n_path = int(meta["path_points"])
            speed = speed_by_obj.get(meta["object"], 2.0)
            q.put({"type": "litho_object_start", "name": name,
                   "object": meta["object"], "voltage": float(meta["voltage"]),
                   "duration_s": round(float(meta["duration_s"]), 2),
                   "length_um": round(float(meta["length_um"]), 2)})
            # `source` is the PARENT object's name, unsplit (docs/XY_SLEW_SPEC.md rule 5): it
            # tells the plugin "the next row of the SAME filled shape" (a mild hover-lift)
            # from "a genuinely different object" (a verified withdraw).
            if cockpit_events:
                tip_before[:] = instrument.position_at(clock.now())
            res = instrument.write(WriteObject(name, path[:, 0], path[:, 1], v, speed,
                                           command_setpoint, record, meta["object"],
                                           transit_speed_um_s=MOVE_SPEED_UM_S))
            if not res.done:
                aborted = True
                failure = instrument.failure
                if failure is not None:
                    run_error = f"{type(failure).__name__}: {failure}"
                elif res.outcome == "refused" and not ab.is_set():
                    run_error = res.reason      # not started: a latched fault
                if res.landed is not None:
                    # This object was cut mid-write. Report the prefix that landed on a side
                    # channel (13.1 item 17) -- NOT a litho_object_done, which would
                    # over-mark undosed geometry. The controller ran the pass autonomously,
                    # so `landed` estimates how far it got; report only that prefix.
                    keep_capture(res, v)
                    _part = trace_indices(min(round(res.landed * len(path)), n_path))
                    _obj_index = next(
                        (i for i, od in enumerate(run_pattern.objects)
                         if od.name == meta["object"]), None)
                    q.put({"type": "litho_object_partial", "name": name,
                           "object": meta["object"], "object_index": _obj_index,
                           "voltage": float(meta["voltage"]),
                           "frac0": 0.0, "frac1": round(res.landed, 4),
                           "path_um": [[round(float(x), 3), round(float(y), 3)]
                                       for x, y in path[_part]],
                           "v_written": [round(float(val), 3) for val in v[_part]]})
                notes.abort(run_error or res.reason)
                if failure is not None:
                    _log_to(q, f"Hardware lithography failed: {failure}")
                break
            keep_capture(res, v)
            # The controller runs the pass autonomously, so there is no live per-point
            # position to stream; emit one progress at the object's end so the tip-trace
            # lands where the object finished.
            q.put({"type": "litho_progress", "name": name, "frac": 1.0,
                   "x_um": round(float(path[-1, 0]), 3),
                   "y_um": round(float(path[-1, 1]), 3)})
            _trace = trace_indices(n_path)
            q.put({"type": "litho_object_done", "name": name,
                   "object": meta["object"], "voltage": float(meta["voltage"]),
                   "profile": prof_by_obj.get(meta["object"]),
                   "path_um": [[round(float(x), 3), round(float(y), 3)]
                               for x, y in path[_trace]],
                   "v_written": [round(float(val), 3) for val in v[_trace]],
                   **({"surface_samples": _twin_surface_samples(instrument, path[:n_path])}
                      if cockpit_events else {})})
    except Exception as exc:                            # noqa: BLE001
        run_error = f"{type(exc).__name__}: {exc}"
        notes.abort(run_error)
        _log_to(q, f"Hardware lithography failed: {exc}")
        aborted = True
    # A failed withdraw's request_park overwrites hub._safe_park_reason, so read it first.
    abort_reason = run_error or getattr(hub.host, "_safe_park_reason", None) or "aborted"
    withdraw_error = _end_of_run_withdraw(q, notes, withdraw)
    if aborted:
        _flush_record(q, abort_reason)      # before litho_done: _litho_finish reuses its stem
    if instrument is not None:
        if close_instrument:
            instrument.close()
        else:
            instrument.on_step = None       # this write's notes end with it
    # `error` carries the reason `_litho_finish` stamps into the terminal record and,
    # when the worker did not pre-flush, the sidecar's abort_reason: the withdraw
    # failure if there was one, else the object-loop exception text (13.1 item 17).
    q.put({"type": "litho_done", "aborted": bool(aborted or withdraw_error or ab.is_set()),
           "error": withdraw_error or run_error})




async def _litho_pump(q: _queue.Queue, run_id: str | None = None, claim=None,
                      marks=None):
    """Drain the write worker's queue onto the WebSocket and, at `litho_done`, record
    the session + persist the pass capture.

    `run_id` keys the TERMINAL RECORD closed after that persistence block
    (service/runs.py); optional, and every registry call no-ops on None. This thin
    guard makes the close unconditional: a raise here, or an outright task
    cancellation, must not leave the record open, because `getStatus` would then
    report a write in flight for the life of the process."""
    try:
        await _litho_pump_body(q, run_id, claim, marks)
    except asyncio.CancelledError:
        hub.runs.close_if_open(run_id, completed=False, aborted=True,
                               backend=hub.backend_live(),
                               error="litho pump cancelled")
        raise                                       # never swallow a cancellation
    except Exception as exc:                                    # noqa: BLE001
        await hub.log(f"Lithography pump failed: {exc}")
        hub.runs.close_if_open(run_id, completed=False, aborted=True,
                               backend=hub.backend_live(),
                               error=f"litho pump failed: {exc}")
    finally:
        hub.runs.close_if_open(run_id, completed=False, aborted=True,
                               backend=hub.backend_live(),
                               error="litho pump exited before persistence")
        # RELEASE THE SINGLE-WRITER CLAIM (plan Sec E.2) on EVERY exit -- normal end,
        # raise, or cancellation. A leaked claim would refuse every later write for
        # the life of the process, so it is released in the same unconditional block
        # that closes the run record, never only on the happy path.
        hub.host.pass_ended(run_id, "failed", None, [])
        # Drop the abortable worker handle (Sec 3/10): the write is over, so there is
        # nothing for a `safe_park` to abort+join. A stale handle would be harmless
        # (its thread reports not-alive), but clear it so hub.litho_worker tracks
        # reality.
        hub.host.litho_worker = None


# How often the pump looks at its worker's queue when the queue is empty, and how
# long it waits after the worker has demonstrably finished before it decides nothing
# more is coming. Both are small: this loop is the only thing standing between a dead
# worker and a run record that never closes.
_PUMP_POLL_S = 0.02
_PUMP_GRACE_S = 1.0


def _pump_worker(claim):
    """The thread whose liveness says whether more messages can still arrive.

    `claim.worker` is stamped at hand-over and is the honest handle (it is judged per
    RUN, not per hub field a later run may replace); `hub.litho_thread` is the
    fallback for a pump driven directly, as the tests do.
    """
    worker = getattr(claim, "worker", None) if claim is not None else None
    return worker if worker is not None else getattr(hub.host, "litho_thread", None)


def _worker_finished(worker) -> bool:
    """True once `worker` has demonstrably run and stopped. A thread created but never
    started is NOT finished -- it has not run yet, and treating it as dead would close
    a run that is about to begin."""
    if worker is None:
        return False
    started = getattr(worker, "_started", None)
    if started is not None and not started.is_set():
        return False
    is_alive = getattr(worker, "is_alive", None)
    return bool(is_alive is not None and not is_alive())


async def _litho_pump_body(q: _queue.Queue, run_id: str | None = None,
                           claim=None, marks=None):
    from ..runs import existing_files
    hub.host.written_accum = []
    hub.host._litho_partial = None
    # No pre-flushed forensic record for this run yet (13.1 item 17). The hardware
    # worker sets this on an aborted flush; reset here so a twin write -- or a clean
    # write -- never reuses a prior hardware abort's stem.
    hub._litho_forensics = None
    # WHAT THIS RUN MAY MARK AS WRITTEN, and how many segment completions each object
    # needs before it counts (commands.litho_cmds._marks). Marking used to happen at
    # START, for every submitted object at once -- so a write aborted after one object
    # left the cockpit believing the whole design was written, and the operator's next
    # default write skipped geometry that was never dosed.
    marks = dict(marks or {})
    seen: dict[str, int] = {}
    worker = _pump_worker(claim)
    dead_since = None
    while True:
        # NO UNTIMED HOLD ON THE SHARED DEFAULT EXECUTOR.
        #
        # This used to be `await loop.run_in_executor(None, q.get)`: one pool thread
        # parked on a blocking `get` for the WHOLE write. On the instrument a pump
        # whose worker dies mid-write parks that thread FOREVER, and enough of those
        # starve every `asyncio.to_thread` in the service -- including `hw_withdraw`,
        # which is the call that gets the tip off the surface. The fail-safe must not
        # queue behind a bookkeeping leak.
        #
        # Polling the queue without a thread at all removes the hazard rather than
        # bounding it: `get_nowait` never blocks, and the 20 ms sleep hands the loop
        # back to telemetry and to Abort between messages.
        try:
            msg = q.get_nowait()
            dead_since = None
        except _queue.Empty:
            # BOTH workers guarantee a `litho_done` in a `finally`, so an empty queue
            # beside a finished worker means the thread died in a way Python could not
            # run that finally for (a hard kill, an interpreter tear-down, a test that
            # dropped it). Close the run as ABORTED with a reason instead of waiting
            # for a message that is never coming -- an open record makes `getStatus`
            # report a write in flight for the life of the process, and the claim it
            # holds refuses every later write.
            if _worker_finished(worker):
                now = time.monotonic()
                if dead_since is None:
                    dead_since = now            # grace: a last message may be in
                elif now - dead_since >= _PUMP_GRACE_S:   # flight behind the exit
                    msg = {"type": "litho_done", "aborted": True,
                           "error": "write worker exited without signalling "
                                    "completion"}
                    await hub.log("Lithography worker exited without a litho_done -- "
                                  "closing the run as aborted.")
                    await hub.broadcast(msg)
                    await _litho_finish(msg, run_id, claim, existing_files)
                    break
            await asyncio.sleep(_PUMP_POLL_S)
            continue
        await _broadcast_ui(msg)
        match msg["type"]:
            case "litho_object_start":
                await hub.log(
                    f"Writing {msg['name']}: {msg['length_um']} um at "
                    f"{msg['voltage']:+.1f} V ({msg['duration_s']} s)")
            case "litho_transit":
                if msg["distance_um"] > 0.05:
                    speed = ("instantaneous" if msg["peak_um_s"] is None
                             else f"peak {msg['peak_um_s']:g} um/s")
                    await hub.log(f"Transit {msg['distance_um']:g} um "
                                  f"({msg['mode']}, {speed})")
            case "litho_object_partial":
                # An abort cut this object mid-stroke (the plugin reports how far it got).
                # It is deliberately kept OUT of `written_accum` and the
                # `written_objects` marks -- those are "what may be counted
                # as dosed", and a partial stroke must not silently pass
                # `select_objects`' re-write skip. It lives in its own slot
                # so `_litho_finish` can still surface it to
                # `getWriteResult(inline=True).trace`, which is a read-only
                # diagnostic, not a completion claim.
                hub.host._litho_partial = {k: msg[k] for k in
                                      ("name", "object", "object_index",
                                       "voltage", "frac0", "frac1",
                                       "path_um", "v_written")} | {"t": time.time()}
            case "litho_object_done":
                # `t` is stamped HERE (the pump, on the loop) rather than in the
                # worker: it is the only ordering the wire cares about, and the
                # worker's message shape is shared with the twin run.
                # `object` rides along so `getWritten` / the inline trace can name
                # the DESIGN object a `name#k` stroke entry belongs to. Both
                # workers already emit it beside `name`; only the pump dropped it.
                hub.host.written_accum.append(
                    {k: msg[k] for k in ("name", "voltage", "path_um", "v_written")}
                    | {"t": time.time(), "duration_s": msg.get("duration_s"),
                       "length_um": msg.get("length_um"),
                       "object": msg.get("object")})
                ss = msg.get("surface_samples")
                if ss and len(ss["x_um"]) > 4:
                    cid = f"litho-{msg['name']}-{int(time.time() * 1000)}"
                    # Fusing a pass into the Surface is a side effect of writing,
                    # not part of it. It used to be able to take the whole run
                    # down with it: one bad filename threw here, the pump task
                    # died, and `litho_done` never reached the panel, which sat
                    # showing 100 % of a segment forever. Report and carry on.
                    try:
                        await asyncio.to_thread(
                            hub.surface.ingest_path, cid,
                            ss["x_um"], ss["y_um"], ss["z_nm"])
                        await hub.broadcast({"type": "surface_changed"})
                    # logged; an ingest failure must not stop the write
                    except Exception as exc:  # noqa: BLE001
                        await hub.log(f"Surface ingest failed for {msg['name']}: "
                                      f"{exc}")
                # MARK AT COMPLETION, once every segment of the object has landed.
                # `msg["object"]` is the parent object; `msg["name"]` may be one of
                # several strokes it expanded into.
                obj_name = msg.get("object") or msg.get("name")
                mark = marks.get(obj_name)
                if mark is not None:
                    seen[obj_name] = seen.get(obj_name, 0) + 1
                    expect = mark.get("expect")
                    if expect is None or seen[obj_name] >= int(expect):
                        done = getattr(hub.host, "written_objects", None)
                        if done is None:
                            done = hub.host.written_objects = set()
                        done.add(mark["id"])
                await hub.log(f"Object {msg['name']} complete")
            case "litho_done":
                await _litho_finish(msg, run_id, claim,
                                    existing_files)
                break


def _forensic_files(aborted: bool) -> list:
    """Paths of the forensic flush an aborted write already left on disk, else [].
    Never raises: it runs just before the claim release."""
    pre = getattr(hub, "_litho_forensics", None)
    if not aborted or not pre or not pre.get("stem"):
        return []
    try:
        from .. import stems
        sidecar = stems.dir_for(hub.store.path.parent, "litho", "json") / f"{pre['stem']}.json"
        return [str(p) for p in (sidecar, pre.get("ref")) if p]
    except Exception:                                     # noqa: BLE001
        return []


async def _litho_finish(msg, run_id, claim, existing_files) -> None:
    """Close out ONE write: snapshot, release the claim, record the session,
    persist the capture and close the terminal record.

    Extracted from the pump's `litho_done` branch so the two ways a write can end
    -- the worker's own `litho_done`, and the pump noticing the worker died without
    sending one -- go through exactly this code. A second, "simpler" close path for
    the failure case is how a run record ends up half-written.
    """
    # SNAPSHOT EVERYTHING THIS RUN IS ABOUT TO PERSIST -- BEFORE the
    # claim is released, and read from the locals from here on.
    #
    # Releasing the claim lets the NEXT write start, and a starting write
    # resets exactly the accumulators this block is still reading:
    # `_litho_pump_body` clears `written_accum` and `litho_write` clears
    # `hw_litho_capture`. A reviewer demonstrated it -- run 2 started
    # while run 1 was still persisting, and run 1's session record went
    # to disk with an empty `written` and no `capture_channels`. The run
    # was real, the file was a lie, which is the worst kind of bug this
    # code can have.
    #
    # Snapshotting is the fix rather than holding the claim longer: the
    # claim's job is the START race, and a run that has stopped driving
    # the tip must not refuse the operator's next start for the length of
    # a file write.
    written = list(getattr(hub.host, "written_accum", None) or [])
    # The mid-stroke-cut object (if any), snapshotted the same way and for
    # the same reason as `written` above -- the next run's start clears
    # `hub._litho_partial` too. It is spliced into the TRACE only, below;
    # `written`/`stats.objects`/`getWritten` must never see it.
    partial = getattr(hub.host, "_litho_partial", None)
    cap = {k: list(v) for k, v in
           getattr(hub.host, "hw_litho_capture", {}).items() if v}
    pattern = getattr(hub, "pattern", None)
    sample_name = hub.sample_name
    hw_cal = dict(getattr(hub.host, "hw_cal", {}) or {})
    # THE WRITE-PARAMETER CONTEXT stamped at write START (litho_cmds:
    # `_litho_write_ctx`), snapshotted here for the SAME reason as `written`/`cap`
    # above: releasing the claim lets the next write start and overwrite it. This is
    # what the record sidecar (writes/json/<stem>.json) persists
    # (docs/INSTRUMENT_MODEL_SPEC.md sec 5/10).
    write_ctx = dict(getattr(hub, "_litho_write_ctx", None) or {})
    # THE COMMANDED Output.A PER POINT (spec sec 5; 13.1 item 16) is now accumulated
    # at FULL RESOLUTION by each worker as `hub.hw_litho_capture["output_a_cmd_v"]`
    # -- exactly the `v_lit = v_tip / amp_gain` array LithographyRun loads on
    # hardware, and `seg["v"] / amp_gain` per stroke on the twin -- so it rides in
    # `cap` above alongside the measured channels. It is no longer reconstructed
    # from the decimated, rounded `v_written` UI stream (that decimation was the B2
    # review's REJECT). `cap` therefore already carries it for both backends; the
    # `if cap:` persistence block below writes the npz whenever any object was
    # written, twin or hardware.
    # RELEASE THE SINGLE-WRITER CLAIM HERE, at PASS END -- not after the
    # persistence block below (plan Sec E.2). The claim exists to close
    # the start-path race: it must cover the window from the busy check,
    # through the first await, to the moment the run is real. The pump's
    # `finally` releases it too, as the backstop for a raise or a
    # cancellation that never reaches this line.
    aborted = bool(msg.get("aborted"))
    worker_error = msg.get("error")     # set when the worker RAISED
    hub.host.pass_ended(
        run_id, "failed" if worker_error else "aborted" if aborted else "completed",
        len(written), _forensic_files(aborted))
    # The write engage is over: disarm the runtime interlocks (a clean write leaves
    # them armed; an interlock-tripped write already self-disarmed) (12.1).
    try:
        from .engage import disarm_interlocks
        disarm_interlocks(hub)
    except Exception:                                    # noqa: BLE001,S110
        pass
    # carry the submitted count through the close (annotated at start)
    _rec = hub.runs.get(run_id) if run_id else None
    _submitted = (_rec.stats.get("objects_submitted")
                  if _rec is not None else None)
    await hub.log("Lithography aborted" if aborted
                  else "Pattern complete: all objects written")
    runtime = hub.store.path.parent
    # ONE stem for every file this run writes (docs/PLAN.md "Layout and naming"),
    # same convention as a scan.
    from .. import stems
    # The reason an aborted write carries, shared by the sidecar and the terminal
    # record: the worker's exception/withdraw text (`msg["error"]`), else a safe-park
    # reason, else a bare "aborted".
    abort_reason = ((worker_error or getattr(hub.host, "_safe_park_reason", None)
                     or "aborted") if aborted else None)
    # FORENSIC FILES FIRST (docs/INSTRUMENT_MODEL_SPEC.md sec 5; 13.1 items 15/17).
    # If the hardware worker already flushed the npz + sidecar BEFORE its withdraw
    # (an aborted hardware write), reuse that stem -- the files, with the right
    # abort_reason, are already on disk -- so the session record shares the name and
    # nothing is rewritten. Otherwise (a twin write, or a clean completion) write the
    # files here, and BEFORE the session record below so the completion signal can
    # never precede the forensic files a reader loads.
    ref = gwy = ibw = None
    _pre = getattr(hub, "_litho_forensics", None)
    if _pre and _pre.get("stem"):
        stem = _pre["stem"]
        ref = _pre.get("ref")
        if _pre.get("abort_reason"):
            abort_reason = _pre["abort_reason"]
        await hub.log(f"Write record (flushed pre-withdraw): "
                      f"writes/json/{stem}.json")
    else:
        stem = await asyncio.to_thread(stems.make_stem, runtime, "litho")
        # THE WRITE RECORD SIDECAR (docs/INSTRUMENT_MODEL_SPEC.md sec 5/10): every
        # write, twin or hardware, completed or aborted, persists the parameters it
        # ran under to writes/json/<stem>.json -- amp_gain, the ABSOLUTE setpoint
        # applied (free_air_v + setpoint_relative) and the relative value, the
        # free-air baseline and its timestamp, the model snapshot hash, and whether
        # it aborted (with a reason). Atomic temp+rename, so a reader never sees a
        # half-written file. Best-effort: a sidecar write must not take the run down.
        try:
            _sidecar_dir = stems.dir_for(runtime, "litho", "json")
            _sidecar_path = _sidecar_dir / f"{stem}.json"
            await asyncio.to_thread(
                _atomic_write_text, _sidecar_path,
                json.dumps(_build_write_sidecar(
                    write_ctx, aborted=aborted, abort_reason=abort_reason),
                    indent=2))
            await hub.log(f"Write record: writes/json/{stem}.json")
        except Exception as exc:                            # noqa: BLE001
            await hub.log(f"Write-record sidecar failed for {stem}: {exc}")
    # Persist the pass's RAW capture under writes/npz/<stem>.npz, like a scan
    # does, and auto-export it to a single native Gwyddion .gwy holding every
    # channel. Best-effort: a save/export hiccup must never take the run down.
    # No PNG/sidecar for a litho pass -- its capture is a 1-D trajectory
    # readback, not a raster (docs/PLAN.md Slice A item 4).
    if cap:
        try:
            from ..captures import save_capture
            # The worker's pre-withdraw flush already saved the npz (ref set above);
            # only save here when it did not (twin, or a clean completion).
            if ref is None:
                ref = await asyncio.to_thread(
                    save_capture, runtime, stem, cap,
                    out_dir=stems.dir_for(runtime, "litho", "npz"))
            if ref:
                await hub.log(f"Pass capture saved: {ref}")
            # A litho capture is a 1-D readback along the trajectory, not a
            # raster; each channel becomes a thin 1 x N field in the .gwy
            # (nominal 1 um/sample axis; the values are exact).
            from types import SimpleNamespace
            n = max((len(v) for v in cap.values()), default=0)
            cfg = SimpleNamespace(pixels=n, lines=1, size_um=float(n),
                                  trace_retrace=False, direction="up")
            from ...io.gwy import export_capture_gwy
            gwy = await asyncio.to_thread(
                export_capture_gwy, runtime, stem, cap, cfg, hw_cal,
                out_dir=stems.dir_for(runtime, "litho", "gwy"))
            if gwy:
                await hub.log(f"GWY exported: {len(cap)} channel(s) -> "
                              f"writes/gwy/{stem}.gwy")
            from ...io.ibw import export_capture_ibw
            ibw = await asyncio.to_thread(
                export_capture_ibw, runtime, stem, cap, cfg, hw_cal,
                out_dir=stems.dir_for(runtime, "litho", "ibw"))
            if ibw:
                await hub.log(f"IBW exported: {len(cap)} channel(s) -> "
                              f"writes/ibw/{stem}.ibw")
        except Exception as exc:            # noqa: BLE001
            await hub.log(f"Pass capture save/export failed for "
                          f"{stem}: {exc}")
    # The stored run gets the commanded X/Y of the passes that were fired (named by the
    # capture's stem) and the capture links. Runs after the withdraw, from the pump; a
    # failure only loses the links.
    stored_run = write_ctx.get("_stored_run")
    if stored_run is not None:
        try:
            from .. import run_store
            _written = {item["name"] for item in written}
            _cut = ((partial["name"], partial.get("frac1")) if partial else None)
            await asyncio.to_thread(
                run_store.finish_write_run,
                run_store.RunStore(run_store.store_dir(runtime)), runtime, stem, stored_run,
                written=_written, partial=_cut,
                captures=run_store.capture_links(runtime, ref, gwy, ibw))
        except Exception as exc:                        # noqa: BLE001
            await hub.log(f"Run links not saved for {stem}: {exc}")
    # THE SESSION RECORD -- written ONLY now that the forensic files above are on
    # disk (13.1 item 15). This is the completion signal the sessions list / a
    # polling test observe, so it must come AFTER the files, never before. `cap` is
    # the RAW capture accumulated across the pass, snapshotted above.
    rec = hub.store.record("litho", {
        "label": f"{pattern.name if pattern else 'pattern'} on "
                 f"{sample_name}" + (" (aborted)" if aborted else ""),
        "sample": sample_name,
        "design": pattern.to_dict() if pattern else None,
        "written": written,
        "capture_channels": list(cap),
    })
    await hub.log(f"Session recorded: {rec['id']}")
    # Map the session id -> stem SYNCHRONOUSLY, right after the record and with no
    # await in between, so a reader that has just seen the record can resolve the
    # stem (and reach the forensic files, already on disk above).
    stems.register(runtime, rec["id"], "litho", stem)
    # TERMINAL RECORD (plan A.1). Closed after every file is on disk, so
    # `completed` is a promise about the paths. The TWIN write path
    # produces no capture at all -- it still completes, with an empty
    # `files` and `partial` false, because nothing was attempted.
    _attempted = ({"npz": (runtime / ref) if ref else None,
                   "gwy": gwy, "ibw": ibw} if cap else {})
    _files, _partial = await asyncio.to_thread(existing_files, _attempted)
    # The written path + tip voltage, kept in memory and bounded, so
    # `getWriteResult(inline=true)` can hand a script the current-along-path trace
    # without a file read on the REP thread (plan Sec A: "conductance-along-path is
    # the payload goal 2 cares about"). Written from the SNAPSHOT, like everything
    # else below the release -- never from `hub.written_accum`, which the next run
    # is free to clear.
    # `partial` is tagged and appended ONLY for the trace -- `remember_trace`
    # is the sole reader of this combined list; `written` above (persisted,
    # `stats.objects`, `getWritten`) stays exactly what completed.
    trace_source = written + [partial | {"partial": True}] if partial else written
    litho_cmds.remember_trace(hub, run_id, trace_source)
    hub.finish_run(run_id, completed=True, aborted=aborted,
                   error=worker_error,
                   backend=hub.backend_live(), files=_files,
                   partial=_partial, channels=list(cap),
                   record_id=rec["id"], run_ref=write_ctx.get("run"),
                   shape=[1, max((len(v) for v in cap.values()),
                                 default=0)] if cap else None,
                   stats={"objects": len(written),
                          "objects_submitted": _submitted,
                          "pattern": pattern.name if pattern else None})
    await hub.broadcast({"type": "sessions_changed"})
