"""The scan pumps: the hardware pump that drives one raster as a capturing pass, the twin
pump that plays one, and the drift registration both call after a frame is cached.

`scan_start` (`commands/scan_cmds.py`) arms a raster and hands the streaming to a pump, which
advances the scan line by line and broadcasts each row to connected clients. The pumps
read the process hub.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from dataclasses import dataclass

import numpy as np

from ... import clock
from ...hardware import channels
from ...hardware.xy_transit import MOVE_SPEED_UM_S
from ..hub import hub
from ..hw_scan import scan_line_count
from .safe_park import (
    _WITHDRAWN_HEIGHT_TOL_V,
    PARK_WATCHDOG,
    STOP_SCAN_PASS,
    _signal_stop,
    request_park,
)
from ..step_notes import NO_NOTES


async def _register_drift():
    """Register the just-cached frame against the drift reference and broadcast
    the estimate (Connect Phase 2). Best-effort: called from both scan pumps
    right after a frame is cached, and wrapped by the caller's try/except so a
    drift hiccup can never break a scan. No-op until a reference is set.

    Reads the NEWEST cached frame off disk (the one this pump just wrote), so it
    is identical for the sim and hardware paths and decoupled from either pump's
    in-memory frame shape. `register_frame` mutates hub.drift, tracks the ports
    when the toggle is on, and returns the {"type":"drift", ...} broadcast.
    """
    from .. import drift as drift_mod
    from ..scan_cache import load_latest_scan_frame
    if hub.drift.get("reference") is None:
        return
    frame = await asyncio.to_thread(load_latest_scan_frame, hub.store.path.parent)
    if not frame:
        return
    msg = await asyncio.to_thread(drift_mod.register_frame, hub,
                                  hub.store.path.parent, frame)
    if not msg:
        return
    await hub.broadcast(msg)
    if msg.get("rejected"):
        await hub.log(f"Drift: estimate rejected ({msg.get('reason', '?')})")
    else:
        note = ""
        if msg.get("ports_moved"):
            note += ", ports tracked"
        if msg.get("rebased"):
            note += ", reference re-based"
        await hub.log(f"Drift: dx={msg['dx_um'] * 1000:+.0f} nm "
                      f"dy={msg['dy_um'] * 1000:+.0f} nm "
                      f"(total {msg['total'][0] * 1000:+.0f}, "
                      f"{msg['total'][1] * 1000:+.0f} nm){note}")




async def _link_scan_captures(run_id, runtime, *files) -> None:
    """Add the scan's saved data files to its stored run. Best-effort, after the pass."""
    ref = getattr(hub.runs.get(run_id), "run_ref", None) if run_id else None
    if not isinstance(ref, dict):
        return
    try:
        from .. import run_store
        store = run_store.RunStore(run_store.store_dir(runtime))
        links = run_store.capture_links(runtime, *files)
        await asyncio.to_thread(run_store.link_captures, store, ref["execution_id"], links)
    except Exception as exc:                              # noqa: BLE001
        await hub.log(f"Run links not saved for {run_id}: {exc}")


def _scan_command(config):
    """The ScanRaster for a hardware scan's ScanConfig (the plugin takes the same fields)."""
    from ...instrument import ScanRaster
    from ...run.model import ScanFrame, Setpoint
    from ..hw_scan import scan_capture_channels, scan_line_count
    mode = config.mode
    frame = ScanFrame(
        source="scan request", size_um=float(config.size_um), lines=scan_line_count(config),
        pixels=int(config.pixels), angle_deg=float(config.angle_deg),
        offset_um=(float(config.x_offset_um), float(config.y_offset_um)), drift_offset=False,
        direction=str(config.direction), line_rate_hz=float(config.line_rate_hz),
        trace_and_retrace=bool(getattr(config, "trace_retrace", True)))
    return ScanRaster(frame, mode, Setpoint("absolute", float(config.setpoint)),
                      bool(config.engage), scan_capture_channels(mode))


async def _scan_on_instrument(config, notes, engaged: bool | None = None) -> dict:
    """Run the raster through the Asylum plugin on a worker thread (the plugin runs its own
    loop, so never on this one) and return what it came back with: the lines, the captured
    channels and the line indices, partial when the scan stopped. The pump keeps the guards,
    the saving and the deferred withdraw; the plugin does none of them (`service_owns_scan`)."""
    from ...instrument.asylum import AsylumInstrument
    loop = asyncio.get_running_loop()

    cancelled = threading.Event()

    def stream(message: dict) -> None:
        """Broadcast on the service loop, in order. Bounded by the link timeout, as the
        scan's own reads are; a message the loop cannot take in time is dropped."""
        sent = asyncio.run_coroutine_threadsafe(hub.broadcast(message), loop)
        try:
            sent.result(float(getattr(hub.host.hardware_arc, "timeout", 30.0)))
        except concurrent.futures.TimeoutError:
            sent.cancel()

    asylum = AsylumInstrument(hub, service_owns_scan=True)
    asylum.scan_message = stream
    asylum.scan_cancelled = cancelled.is_set
    asylum.scan_engaged = engaged
    asylum.scan_notes = notes
    asylum.scan_gains = (config.pgain, config.igain)
    try:
        result = await asyncio.to_thread(asylum.scan, _scan_command(config))
    except asyncio.CancelledError:
        cancelled.set()
        raise
    finally:
        asylum.close()
    if asylum.failure is not None:
        raise asylum.failure
    if asylum.scan_summary is None:
        raise RuntimeError(f"scan refused: {result.reason}")
    return asylum.scan_summary


async def _hw_scan_pump(config, run_id: str | None = None, claim=None, notes=NO_NOTES,
                        switched: bool = False, stops_at_start: int | None = None):
    """Drive one raster as a capturing pass on the real backend, stream the
    projected lines, and record the session.

    `run_id` (optional, allocated by scan_start) keys the TERMINAL RECORD this pump
    closes once the .npz/.gwy/.ibw block has finished -- see service/runs.py. It is
    None when the pump is driven directly (a unit test), and every registry call is a
    no-op on None, so nothing here depends on it. `notes` (service/step_notes.py) journals
    each step's outcome (`switched`: scan_start changed the imaging mode); the default
    journals none. `stops_at_start` is `hub._run_abort_seq` when scan_start began: a stop
    signalled since then (Igor's own abort panel included) aborts this scan before it moves."""
    from .. import stems
    from ..captures import save_capture
    from ..runs import existing_files
    hub.host._hw_abort.clear()
    if stops_at_start is not None and hub.host._run_abort_seq != stops_at_start:
        _signal_stop(hub, STOP_SCAN_PASS)
    hub.host._scan_pump_error = None       # blocker #3: clear before the pass (see below)
    epoch = hub.host._scan_epoch
    lifted = False
    try:
        # The tip state the scan's transit starts from, read before anything below changes it.
        _engaged = bool(getattr(hub.host, "engaged", False))
        # Before the plugin's scan converts um to drive volts with hub.hw_cal: the Cypher
        # rescales the XY LVDT sensitivities on every motor move (xy_cal.py).
        await hub.refresh_xy_cal("scan")
        notes.outcome("check", passed=True)
        if switched or notes.step_of("setmode") is not None:
            notes.outcome("setmode", switched=switched)
        notes.begin()
        # Igor advances the lines on the capture-fill callback: no host-side line pacing.
        summary = await _scan_on_instrument(config, notes, _engaged)
        notes.outcome("end", lambda: {"engaged": bool(getattr(hub.host, "engaged", False))})
        # Blocker #2: a chained-scan `err` (a router failure, a bridge ERR, a link
        # drop surfaced by the driver) returns aborted=True with an `error` string --
        # NOT an operator abort. Record it as a FRAME FAILURE so the continuous
        # supervisor counts it (and, at the cap, safe_parks) instead of silently
        # treating the wedged rig as a clean aborted frame.
        if summary.get("error") and not hub.host._hw_abort.is_set():
            hub.host._scan_pump_error = f"chained scan error: {summary['error']}"
        # Lift the tip first, while the claim is still held: the saving below is slow, and
        # a write or approach started after the release must not meet a late withdraw.
        lifted = True
        await _withdraw_at_scan_end(epoch, _hardware_withdraw)
        # SNAPSHOT WHAT THIS RUN PERSISTS, THEN release the claim (same rule as the
        # twin pump above and the litho pump: nothing below may read a hub field the
        # next run is free to reset). `summary` and `config` are already locals.
        cap = summary.get("capture") or {}
        sample_name = hub.sample_name
        hw_cal = dict(getattr(hub.host, "hw_cal", {}) or {})
        # RELEASE THE SINGLE-WRITER CLAIM HERE, at PASS END -- not after the
        # persistence block below (plan Sec E.2). The claim exists to close the
        # start-path race: it must cover the window from the busy check, through the
        # first await, to the moment the run is real. Holding it through a file write
        # would refuse the operator's next start for no reason. The pump's `finally`
        # releases it too, as the backstop for a raise or a cancellation that never
        # reaches this line.
        hub.host.pass_ended(
            run_id, "failed" if summary.get("error") else
            "aborted" if summary.get("aborted") else "completed", None, [])
        rec = hub.store.record("scan", {
            "label": f"{config.size_um:g} um {config.mode} scan of "
                     f"{sample_name} (hardware)"
                     + (" (aborted)" if summary.get("aborted") else ""),
            "sample": sample_name, "config": config.__dict__,
            "backend": "hardware", "aborted": bool(summary.get("aborted")),
            "capture_channels": list(cap)})
        # ONE stem for every file this run writes (docs/PLAN.md "Layout and
        # naming"); registered under the session id so `captures.load_capture`
        # and a FLEX/export-script caller can resolve this run's files by id.
        runtime = hub.store.path.parent
        stem = await asyncio.to_thread(stems.make_stem, runtime, "scan")
        await asyncio.to_thread(stems.register, runtime, rec["id"], "scan", stem)
        # persist the RAW measured capture next to the session store (recalled by
        # id via stems.lookup); co-located with sessions.jsonl so a redirected
        # store stays isolated.
        ref = await asyncio.to_thread(
            save_capture, runtime, stem, cap,
            out_dir=stems.dir_for(runtime, "scan", "npz"))
        await hub.log(f"Session recorded: {rec['id']} "
                      f"({summary.get('lines', 0)} lines"
                      + (f", capture -> {ref}" if ref else "") + ")")
        # Auto-export the RAW capture to a single native Gwyddion .gwy alongside the
        # npz (all channels in one openable file for diagnostics). Best-effort: a .gwy
        # hiccup must never fail the scan (same as the frame-cache below).
        gwy = ibw = None
        if cap:
            try:
                from ...io.gwy import export_capture_gwy
                gwy = await asyncio.to_thread(
                    export_capture_gwy, runtime, stem, cap, config, hw_cal,
                    out_dir=stems.dir_for(runtime, "scan", "gwy"))
                if gwy:
                    await hub.log(f"GWY exported: {len(cap)} channel(s) -> "
                                  f"scans/gwy/{stem}.gwy")
                from ...io.ibw import export_capture_ibw
                ibw = await asyncio.to_thread(
                    export_capture_ibw, runtime, stem, cap, config, hw_cal,
                    out_dir=stems.dir_for(runtime, "scan", "ibw"))
                if ibw:
                    await hub.log(f"IBW exported: {len(cap)} channel(s) -> "
                                  f"scans/ibw/{stem}.ibw")
            except Exception as exc:                    # noqa: BLE001
                await hub.log(f"GWY/IBW export failed for {rec['id']}: {exc}")
        # Cache the completed frame to disk so the litho cockpit's background stays
        # painted (GET /api/scan/last) instead of vanishing on scan_done -- mirror the
        # sim path (_scan_pump): a (rows,pixels) height grid in nm keyed by the session
        # id (ids sort by creation, so newest wins). We reuse the SAME display-flattened
        # `height` the live stream carried, so the cached
        # background matches what was shown; the RAW capture (save_capture above) is
        # untouched -- the write/targeting path keeps using it.
        # Best-effort: a caching hiccup must never fail the scan.
        fref = None
        png_paths: dict = {}          # {"height": Path, "<channel>": Path, ...}
        json_path = None
        try:
            import numpy as np

            from ..hw_scan import _HEIGHT, _err_chan, _rows_to_lines
            from ..scan_cache import save_scan_frame
            trace_retrace = bool(getattr(config, "trace_retrace", True))
            pixels = int(config.pixels)
            rows = int(config.lines) or pixels
            ppl = 2 * pixels if trace_retrace else pixels
            err_chan = _err_chan(config.mode)               # contact->Deflection, AC->Amplitude
            grid = [float("nan")] * (rows * pixels)         # NaN = unscanned (transparent)
            # Place each MEASURED line at its TRUE scan index (from line_indices),
            # not its ordinal in the concatenation. A short/aborted scan then leaves
            # the unscanned rows NaN (transparent) instead of zero-padded to grey,
            # and complete scans land pixel-identical to the live frame.
            cap_h = np.asarray(cap.get(_HEIGHT, []), dtype=float)
            cap_e = np.asarray(cap.get(err_chan, []), dtype=float)
            got = cap_h.size // ppl if ppl else 0
            indices = summary.get("line_indices") or list(range(min(rows, got)))
            for pos, idx in enumerate(indices):
                z_row = cap_h[pos * ppl:(pos + 1) * ppl]
                e_row = cap_e[pos * ppl:(pos + 1) * ppl]
                if z_row.size < ppl:
                    continue
                for msg in _rows_to_lines(z_row, e_row, int(idx), 1, config,
                                          hub.host.hw_cal, trace_retrace, err_chan=err_chan):
                    si, hh = msg["slow_idx"], msg["height"]
                    for i in range(min(pixels, len(hh))):
                        grid[si * pixels + i] = hh[i]
            frame = {"stamp": stem, "pixels": pixels, "rows": rows,
                     "size_um": float(config.size_um),
                     "x_offset_um": float(config.x_offset_um),
                     "y_offset_um": float(config.y_offset_um),
                     "angle_deg": float(config.angle_deg), "height": grid}
            fref = await asyncio.to_thread(save_scan_frame, runtime, frame)
            hub.last_scan = {k: v for k, v in frame.items() if k != "height"}  # metadata only
            hub.last_scan_ref = fref
            await hub.log(f"Scan cached: {rows}x{pixels} frame -> {fref}")
        except Exception as exc:                             # noqa: BLE001
            await hub.log(f"Scan cache skipped: {exc}")
        # One PNG per captured channel, plus a JSON sidecar (docs/
        # IMAGE_SAVING_NOTES.md, "Layout and naming"): height comes from the
        # display frame just cached above (flattened, what the cockpit shows);
        # every OTHER 2-D raw-capture channel gets its own auto-scaled PNG, so
        # a channel that looks better than height (amplitude, deflection) is
        # still available under the design. Best-effort, same as the frame
        # cache above -- a PNG hiccup must never fail the scan.
        if fref is not None:
            try:
                from ...io.capture_grid import capture_channels_2d
                from ...io.png import write_grey8_png
                from ..sidecar import write_json_atomic
                png_dir = stems.dir_for(runtime, "scan", "png")
                height_arr = np.asarray(grid, dtype=float).reshape(rows, pixels)
                png_paths["height"] = await asyncio.to_thread(
                    write_grey8_png, png_dir / f"{stem}_height.png", height_arr)
                if cap:
                    # `capture_channels_2d` splits every 2-D scan channel into
                    # "<chan> trace"/"<chan> retrace" (io/capture_grid.py); the
                    # PNG is the PRIMARY (trace) sweep only, one per channel,
                    # lower-cased. ZSensor is skipped -- that channel IS
                    # "height" above, but taken RAW (unflattened) here, so a
                    # second "zsensor.png" would silently disagree with the
                    # cockpit's own background.
                    for ch2d in await asyncio.to_thread(
                            capture_channels_2d, cap, config, hw_cal):
                        name = ch2d["name"]
                        if name.endswith(" retrace"):
                            continue
                        base = name.removesuffix(" trace")
                        if base == _HEIGHT:
                            continue
                        key = base.lower()
                        if key in png_paths:
                            continue
                        png_paths[key] = await asyncio.to_thread(
                            write_grey8_png, png_dir / f"{stem}_{key}.png",
                            ch2d["grid2d"])
                sidecar = {
                    "run_id": rec["id"], "stamp": stem,
                    "size_um": float(config.size_um), "pixels": pixels, "rows": rows,
                    "x_offset_um": float(config.x_offset_um),
                    "y_offset_um": float(config.y_offset_um),
                    "angle_deg": float(config.angle_deg), "mode": str(config.mode),
                    "line_rate_hz": float(config.line_rate_hz),
                    "direction": str(config.direction),
                    "row0": "bottom",         # io/png.py: PNG top row = grid row R-1
                    "channels": sorted(png_paths),
                }
                _run_ref = getattr(hub.runs.get(run_id), "run_ref", None) if run_id else None
                if _run_ref is not None:
                    sidecar["run"] = _run_ref
                json_path = await asyncio.to_thread(
                    write_json_atomic,
                    stems.dir_for(runtime, "scan", "json") / f"{stem}.json", sidecar)
                await hub.log(f"PNG/JSON written: {len(png_paths)} channel(s) -> "
                              f"scans/png/{stem}_*.png, scans/json/{stem}.json")
            except Exception as exc:                          # noqa: BLE001
                await hub.log(f"PNG/JSON export skipped: {exc}")
        await _link_scan_captures(run_id, runtime, ref if cap else None, gwy, ibw,
                                  (runtime / fref) if fref else None)
        # TERMINAL RECORD (plan A.1). Closed HERE -- after EVERY file this run
        # produces is on disk (the raw capture, the exports AND the cached frame) --
        # so `completed` is a promise about the paths, not about the done event.
        # Only ATTEMPTED writes are offered to the stat: with no capture nothing was
        # written, so nothing is missing and `partial` stays false.
        _base = runtime
        _attempted: dict = {}
        if cap:
            _attempted = {"npz": (_base / ref) if ref else None,
                          "gwy": gwy, "ibw": ibw}
        _attempted["frame"] = (_base / fref) if fref else None
        if fref is not None:
            for key, p in png_paths.items():
                _attempted["png" if key == "height" else f"png_{key}"] = p
            _attempted["json"] = json_path
        _files, _partial = await asyncio.to_thread(existing_files, _attempted)
        _pixels = int(config.pixels)
        _rows = int(config.lines) or _pixels
        hub.finish_run(run_id, completed=True, backend="hardware",
                       aborted=bool(summary.get("aborted")),
                       files=_files, partial=_partial,
                       shape=[_rows, _pixels], channels=list(cap),
                       record_id=rec["id"], error=summary.get("error"),
                       stats={"lines": int(summary.get("lines", 0) or 0),
                              "pixels": _pixels,
                              "size_um": float(config.size_um),
                              "mode": str(config.mode)})
        # Connect Phase 2: register this frame's drift against the reference
        # (measure always; track ports / accumulate compensation when enabled).
        # Best-effort so a drift hiccup never breaks a hardware scan.
        try:
            await _register_drift()
        except Exception as exc:                             # noqa: BLE001
            await hub.log(f"Drift registration skipped: {exc}")
        await hub.broadcast({"type": "sessions_changed"})
    except Exception as exc:                              # noqa: BLE001
        # Blocker #3: record the fault so the awaiting continuous frame SEES it (this
        # task swallowing the exception is why errors were never counted). A single
        # scan (continuous off) just logs it, as before.
        hub.host._scan_pump_error = f"hardware scan failed: {exc}"
        if not lifted:
            await _withdraw_at_scan_end(epoch, _hardware_withdraw)
        notes.abort(f"hardware scan failed: {exc}")
        await hub.log(f"Hardware scan failed: {exc}")
        # Close the record as NOT completed: a poll must be able to tell "the run
        # ended and there is nothing to fetch" from "still running" (plan A.1).
        hub.finish_run(run_id, completed=False, aborted=True, backend="hardware",
                       error=f"hardware scan failed: {exc}")
        await hub.broadcast({"type": "scan_done", "aborted": True})
    except asyncio.CancelledError:
        # `except Exception` does NOT catch this (it is BaseException), so without
        # this clause a cancelled pump left its record open forever.
        hub.runs.close_if_open(run_id, completed=False, aborted=True,
                               backend="hardware", error="scan pump cancelled")
        raise
    finally:
        # The record MUST be closed on every path. A no-op when the try block
        # already closed it with the real outcome.
        hub.runs.close_if_open(run_id, completed=False, aborted=True,
                               backend="hardware",
                               error="scan pump exited before persistence")
        # RELEASE THE SINGLE-WRITER CLAIM (plan Sec E.2). Before the deferred
        # withdraw, not after: the retract can block for seconds, and a claim held
        # across it would refuse the operator's next start for no reason. Identity-
        # checked inside, so a late release can never free a newer run's claim.
        hub.host.pass_ended(run_id, "failed", None, [])
        # deferred Stop retract (see scan_abort / the rail-dwell trip): the pass has
        # released the bridge link, so the real retract is now safe. Runs on normal
        # finish AND if the pass raised -- either way the worker thread is done with
        # the link. The epoch race-guard inside skips a stale withdraw.
        await _deferred_withdraw_if_current()


def _hardware_withdraw() -> dict:
    from ...hardware.hw_approach import hw_withdraw
    return hw_withdraw(hub.host.hardware_arc)


async def _withdraw_at_scan_end(epoch: int, withdraw) -> None:
    """Lift the tip when a scan ends on its own, by completing or by an error: the plain
    routine withdraw a write ends with (re-arm loop 2, no td_Stop, no park). Runs at pass end,
    before the claim is released and before the saving. Skipped when a continuous run owns the
    tip (its loop parks at the end, also after being stopped), when a Stop or a park already withdrew or will, and when
    a newer pass has taken the tip (the epoch moved, checked again after the withdraw)."""
    loop = getattr(hub.host, "_scan_repeat_task", None)
    if (hub.host.scan_repeat.get("enabled") or (loop is not None and not loop.done())
            or hub.host._hw_abort.is_set() or epoch != hub.host._scan_epoch):
        return
    try:
        snap = await asyncio.to_thread(withdraw)
    except Exception as exc:                                  # noqa: BLE001
        # The tip may still be down: hand the retract to the fail-safe thread and latch a
        # fault so no further pass starts on top of it (as the plugin's _make_safe does).
        message = f"end-of-scan withdraw failed: {exc}"
        await hub.log(f"{message}; parking")
        request_park(hub, message, PARK_WATCHDOG)
        if getattr(hub, "fault", None) is None:
            hub.fault = {"reason": "withdraw_failed", "at": clock.wall(), "state": "fault",
                         "detail": message}
        return
    if epoch != hub.host._scan_epoch:                  # a newer pass took the tip meanwhile
        await hub.log("End-of-scan withdraw returned after a newer pass took the tip")
        return
    snap = snap if isinstance(snap, dict) else {}
    hub.host.engaged = False                # tip off the surface -> a mode switch is now safe
    from .engage import disarm_interlocks
    disarm_interlocks(hub)
    z_v = snap.get("z_v")
    if (snap.get("verify_timeout") or not isinstance(z_v, (int, float))
            or z_v >= _WITHDRAWN_HEIGHT_TOL_V):
        await hub.log("End-of-scan withdraw sent: Z not confirmed at 0 V "
                      f"(Height {z_v}, readback timed out: {bool(snap.get('verify_timeout'))})")
    else:
        await hub.log("End-of-scan withdraw: Z parked at 0 V")
    await hub.broadcast({"type": "withdrawn", **snap})


async def _deferred_withdraw_if_current():
    """Perform the deferred Stop retract requested by scan_abort (or the rail-dwell
    trip), but ONLY if the scan epoch stamped at request time still matches the
    current epoch (docs/STOP_RETRACT_SPEC.md Sec 2). A newer engage/scan bumps the
    epoch and takes over the tip, so a stale deferred withdraw is SKIPPED -- it must
    NOT drive Height to 0 into the fresh pass -- and logged superseded. No-op when no
    deferred withdraw is pending. Called from _hw_scan_pump's finally, once the pass
    has released the bridge link."""
    if not getattr(hub.host, "_hw_withdraw_after", False):
        return
    hub.host._hw_withdraw_after = False
    stamped = getattr(hub.host, "_hw_withdraw_epoch", None)
    hub.host._hw_withdraw_epoch = None
    if stamped is not None and stamped != hub.host._scan_epoch:
        await hub.log(f"Deferred withdraw SKIPPED (superseded): stamped epoch "
                      f"{stamped} != current epoch {hub.host._scan_epoch}; a newer "
                      f"engage/scan now owns the tip.")
        await hub.broadcast({"type": "withdraw_superseded",
                             "stamped": stamped, "epoch": hub.host._scan_epoch})
        return
    # Route the real retract through the unified fail-safe primitive (not a raw
    # hw_withdraw): safe_park stops the pass, drives Height to 0, VERIFIES the
    # readback (retry-once-then-CRITICAL), clears hub.engaged on a verified retract,
    # and logs + records + broadcasts. So an operator Stop is a VERIFIED retract with
    # a backstop, not a fire-and-forget withdraw that only logs on failure. safe_park
    # never raises. It also clears engaged / broadcasts itself, so we do neither here.
    await request_park(hub, "scan stopped")




async def _scan_pump(config, run_id: str | None = None, claim=None):
    """Run one scan on the twin plugin, streaming each line to clients as it lands.

    A THIN GUARD around `_scan_pump_body`, whose job is to guarantee that the run
    record closes on EVERY exit path. The body has unguarded work in it (the surface
    ingest, the broadcast), and an asyncio task can also be cancelled outright -- and
    a record that never closes is reported by `getStatus` as `run_id: scan-N` beside
    `state: idle` forever. `close_if_open` is a no-op when the body already closed
    the record with the real outcome.
    """
    epoch = hub.host._scan_epoch
    lifted: list = []
    try:
        await _scan_pump_body(config, run_id, claim, epoch, lifted)
    except asyncio.CancelledError:
        hub.runs.close_if_open(run_id, completed=False, aborted=True,
                               backend="twin", error="scan pump cancelled")
        raise                                       # never swallow a cancellation
    except Exception as exc:                                    # noqa: BLE001
        await hub.log(f"Scan failed: {exc}")
        hub.runs.close_if_open(run_id, completed=False, aborted=True,
                               backend="twin", error=f"scan failed: {exc}")
        await hub.broadcast({"type": "scan_done", "aborted": True})
        if not lifted:
            await _withdraw_at_scan_end(epoch, hub.host.inst.withdraw)
    finally:
        hub.runs.close_if_open(run_id, completed=False, aborted=True,
                               backend="twin", error="scan pump exited early")
        # Disarm the runtime interlocks: this scan engage is over (12.1). The
        # continuous supervisor re-arms per frame via scan_start.
        try:
            if not hub.host.scan_repeat.get("enabled"):
                from .engage import disarm_interlocks
                disarm_interlocks(hub)
        except Exception:                                # noqa: BLE001,S110
            pass
        # Release the single-writer claim (plan Sec E.2) on every exit path,
        # including cancellation -- a leaked claim refuses every later start.
        hub.host.pass_ended(run_id, "failed", None, [])


@dataclass
class _TwinScan:
    """What a twin scan left: the config, the lines it delivered (metres, at their scan
    index; NaN where unscanned), where each pixel was, and whether it was cut short."""

    config: object
    channels: dict
    row: int
    aborted: bool


def _twin_line_message(config, index: int, trace_nm, retrace_nm, error_v=None) -> dict:
    """The cockpit's `line` frame, in the shape the rig's lines have. The error row is the
    sample's recorded amplitude under the line in either mode (the sample has no deflection
    layer); flat when the sample has no amplitude layer."""
    lines = scan_line_count(config)
    err_name = channels.AMPLITUDE if config.mode == "ac" else channels.DEFLECTION
    height = [round(float(v), 3) for v in trace_nm]
    retrace = height if retrace_nm is None else [round(float(v), 3) for v in retrace_nm]
    flat = ([0.0] * len(height) if error_v is None
            else [round(float(v), 4) for v in error_v])
    return {"type": "line", "row": index,
            "slow_idx": index if config.direction == "up" else lines - 1 - index,
            "rows": lines, "pixels": config.pixels, "size_um": config.size_um,
            "direction": config.direction, "angle_deg": config.angle_deg,
            "errName": err_name,
            "height": height, "retrace": retrace, "error": flat,
            "zsensor": height, "slow_um": (index / max(1, lines)) * config.size_um}


async def _play_twin_scan(config) -> _TwinScan:
    """Run `config` on the twin plugin, broadcasting each line as its time passes."""
    from ...instrument import ScanRaster
    from ...instrument.twin import raster_points
    from ...run.model import ScanFrame, Setpoint
    from ..twin_plugin import plugin
    lines, pixels = scan_line_count(config), int(config.pixels)
    frame = ScanFrame(source="twin scan", size_um=float(config.size_um), lines=lines,
                      pixels=pixels, angle_deg=float(config.angle_deg),
                      offset_um=(float(config.x_offset_um), float(config.y_offset_um)),
                      drift_offset=False, direction=str(config.direction),
                      line_rate_hz=float(config.line_rate_hz),
                      trace_and_retrace=bool(config.trace_retrace))
    grid = np.full((lines, pixels), np.nan)
    xs, ys = np.full_like(grid, np.nan), np.full_like(grid, np.nan)
    first_x, first_y = raster_points(frame)              # one row per line, in scan order
    delivered = [0]
    loop = asyncio.get_running_loop()

    def on_line(index: int, trace_nm, retrace_nm) -> None:
        slow = index if config.direction == "up" else lines - 1 - index
        grid[slow], xs[slow], ys[slow] = trace_nm * 1e-9, first_x[index], first_y[index]
        delivered[0] = index + 1
        hub.host._scan_row = index
        asyncio.run_coroutine_threadsafe(
            hub.broadcast(_twin_line_message(
                config, index, trace_nm, retrace_nm,
                twin.sample.recorded_line(twin.sample.amp_v, first_x[index], first_y[index]))),
            loop)

    twin = plugin(hub)
    twin.speed_factor = hub.host.speed           # the scan speed-up, as the old twin's pump used
    twin.on_line = on_line
    hub.host.twin_scan_angle_deg = float(config.angle_deg)
    hub.host._hw_abort.clear()
    try:
        result = await asyncio.to_thread(twin.scan, ScanRaster(
            frame, config.mode, Setpoint("absolute", float(config.setpoint)), True,
            (channels.HEIGHT,), transit_speed_um_s=MOVE_SPEED_UM_S))
    finally:
        twin.on_line = None
    return _TwinScan(config, {channels.HEIGHT: grid, "X_um": xs, "Y_um": ys}, delivered[0],
                     not result.done)


async def _scan_pump_body(config, run_id: str | None = None, claim=None,
                          epoch: int | None = None, lifted: list | None = None):
    from .. import stems
    from ..runs import existing_files
    epoch = hub.host._scan_epoch if epoch is None else epoch
    scan = await _play_twin_scan(config)
    # Lift the tip while the claim is still held (see _hw_scan_pump).
    if lifted is not None:
        lifted.append(True)
    await _withdraw_at_scan_end(epoch, hub.host.inst.withdraw)
    # The next scan builds its own _TwinScan, so what is persisted below is this pass's.
    done = scan is not None and not scan.aborted
    # RELEASE THE SINGLE-WRITER CLAIM HERE, at PASS END -- not after the persistence
    # block below (plan Sec E.2). The claim exists to close the start-path race: it
    # must cover the window from the busy check, through the first await, to the
    # moment the run is real. Holding it through a file write would refuse the
    # operator's next start for no reason -- "are the files whole?" is what the run
    # record's `completed` flag answers, not this. The pump's `finally` releases it
    # too, as the backstop for a raise or a cancellation that never reaches here.
    hub.host.pass_ended(run_id, "completed" if done else "aborted", None, [])
    sample_name = hub.sample_name
    if scan is not None and scan.row > 4:
        # every frame (even partial) is a surface contribution
        cfg = scan.config
        ch = scan.channels
        cid = f"scan-{int(time.time() * 1000)}"
        contrib = await asyncio.to_thread(
            hub.surface.ingest_raster, cid,
            ch[channels.HEIGHT].astype(float) ,
            ch["X_um"].astype(float), ch["Y_um"].astype(float),
            cfg.size_um / cfg.pixels)
        dx, dy, dz = contrib.offset_um
        await hub.log(
            f"Surface: frame fused ({contrib.n_samples} px); drift estimate "
            f"dx={dx * 1000:+.0f} nm dy={dy * 1000:+.0f} nm dz={dz:+.1f} nm")
        await hub.broadcast({"type": "surface_changed"})
    if done:
        await hub.log("Scan complete")
        cfg = scan.config
        rec = hub.store.record("scan", {
            "label": f"{cfg.size_um:g} um {cfg.mode} scan of {sample_name}",
            "sample": sample_name,
            "config": cfg.__dict__,
        })
        runtime = hub.store.path.parent
        stem = await asyncio.to_thread(stems.make_stem, runtime, "scan")
        await asyncio.to_thread(stems.register, runtime, rec["id"], "scan", stem)
        _frame_path = None
        _rows = _pixels = 0
        png_path = None
        json_path = None
        # Cache the completed frame to disk so the litho cockpit can paint the
        # newest scan under its design overlay (GET /api/scan/last). Best-effort:
        # a missing/odd Height must never crash the pump.
        try:
            import numpy as np

            from ..scan_cache import save_scan_frame
            ch = scan.channels
            height_nm = np.asarray(ch[channels.HEIGHT], float) * 1e9   # metres -> nm
            rows, pixels = height_nm.shape
            frame = {"stamp": stem, "pixels": int(pixels), "rows": int(rows),
                     "size_um": float(cfg.size_um), "x_offset_um": float(cfg.x_offset_um),
                     "y_offset_um": float(cfg.y_offset_um), "angle_deg": float(cfg.angle_deg),
                     "height": height_nm}
            ref = await asyncio.to_thread(save_scan_frame, runtime, frame)
            hub.last_scan = {k: v for k, v in frame.items() if k != "height"}  # metadata only
            hub.last_scan_ref = ref
            _frame_path = (runtime / ref) if ref else None
            _rows, _pixels = int(rows), int(pixels)
            await hub.log(f"Scan cached: {rows}x{pixels} frame -> {ref}")
        except Exception as exc:                                # noqa: BLE001
            await hub.log(f"Scan cache skipped: {exc}")
        # PNG + sidecar (docs/IMAGE_SAVING_NOTES.md). The twin has no raw
        # multi-channel capture, so height is the only channel PNG. A separate
        # try from the frame cache above: a PNG hiccup must not be reported as
        # "the frame cache failed" when it did not.
        if _frame_path is not None:
            try:
                from ...io.png import write_grey8_png
                from ..sidecar import write_json_atomic
                png_dir = stems.dir_for(runtime, "scan", "png")
                png_path = await asyncio.to_thread(
                    write_grey8_png, png_dir / f"{stem}_height.png", height_nm)
                sidecar = {
                    "run_id": rec["id"], "stamp": stem,
                    "size_um": float(cfg.size_um), "pixels": _pixels, "rows": _rows,
                    "x_offset_um": float(cfg.x_offset_um), "y_offset_um": float(cfg.y_offset_um),
                    "angle_deg": float(cfg.angle_deg), "mode": str(cfg.mode),
                    "line_rate_hz": float(cfg.line_rate_hz), "direction": str(cfg.direction),
                    "row0": "bottom",
                    "channels": ["height"],
                }
                _run_ref = getattr(hub.runs.get(run_id), "run_ref", None) if run_id else None
                if _run_ref is not None:
                    sidecar["run"] = _run_ref
                json_path = await asyncio.to_thread(
                    write_json_atomic,
                    stems.dir_for(runtime, "scan", "json") / f"{stem}.json", sidecar)
                await hub.log(f"PNG/JSON written -> scans/png/{stem}_height.png, "
                              f"scans/json/{stem}.json")
            except Exception as exc:                            # noqa: BLE001
                await hub.log(f"PNG/JSON export skipped: {exc}")
        # Connect Phase 2: register drift against the reference (see the hardware
        # pump). Best-effort so a drift hiccup never breaks the sim scan pump.
        try:
            await _register_drift()
        except Exception as exc:                                # noqa: BLE001
            await hub.log(f"Drift registration skipped: {exc}")
        await hub.broadcast({"type": "sessions_changed"})
    # TERMINAL RECORD (plan A.1), closed after the frame cache is on disk. The sim
    # path produces no raw capture and no .gwy/.ibw -- the frame npz, PNG and
    # sidecar are its whole output -- so that is what is offered to the stat.
    if done:
        await _link_scan_captures(run_id, runtime, None, None, None, _frame_path)
        _files, _partial = await asyncio.to_thread(
            existing_files, {"npz": _frame_path, "png": png_path, "json": json_path})
        # `completed` is a promise about the PATHS (plan A.1), so it is gated on the
        # frame cache having actually landed -- a scan whose only output failed to
        # write is not "completed with no files", it is a run with nothing to fetch.
        # measure.py already does this; the two now agree.
        hub.finish_run(run_id, completed=bool(_files), aborted=False, backend="twin",
                       files=_files, partial=_partial, record_id=rec["id"],
                       shape=[_rows, _pixels] if _rows and _pixels else None,
                       channels=["height"] if _files else [],
                       error=None if _files else "the frame cache did not write",
                       stats={"rows": _rows, "pixels": _pixels,
                              "size_um": float(cfg.size_um), "mode": str(cfg.mode)})
    else:
        hub.finish_run(run_id, completed=False, aborted=True, backend="twin")
    await hub.broadcast({"type": "scan_done", "aborted": not done})
