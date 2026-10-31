"""The twin's service: FastAPI + WebSocket telemetry + the web front panel.

Run:  uvicorn afm_litho.service.app:app --port 7461

The service is served on loopback only, through site.serve. There is no auth: running
uvicorn directly with --host on another address exposes instrument control.
"""

from __future__ import annotations

import asyncio
import json
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from . import deps, keeper_link, run_store
from .deps import RUNTIME, WEB
from .hub import hub
from .journal import Journal

app = FastAPI(title="afm-litho twin")
_journal: Journal | None = None
_stop_port = None       # host/stop_port.StopPort, or None when its port was not free

# Domain routers. Imported below `app` because they import hub/deps set up above.
from .routers import (
    connect,
    continuous,
    drift,
    instrument,
    tip,
)

# Rows of the request table get their routes ahead of the old routers.
from .requests import http_door as _http_door
from .requests import stream

_http_door.install(app)

for _router in (tip, connect, drift, continuous, instrument):
    app.include_router(_router.router)

# Helpers the startup primer below calls.
from .host.telemetry import install_hw_telemetry
from .routers.sample import restore_contacts  # startup restores the default sample's marks
from .routers.surface import schedule_rebuild  # startup Surface replay (guarded)


@app.on_event("startup")
async def _startup():
    """Startup: layout migration, loop capture, twin ARC bind,
    contact restore, FLEX provider, control watch, and (on hardware) the bridge."""
    from .. import clock
    clock.refuse_test_mode_with_hardware(hub.host.backend)

    # The stop port first, so a STOP line can park the tip from the moment the service is
    # up. A plain thread: it needs nothing below and never touches this loop.
    global _stop_port
    from .host import stop_port
    _stop_port = stop_port.start(hub)

    # One-shot layout migration (docs/PLAN.md "Layout and naming"), before anything
    # touches captures/ or scans/. Move-only and idempotent; a failure logs and
    # never blocks startup.
    from .migrate_layout import migrate_runtime_layout
    try:
        await asyncio.to_thread(migrate_runtime_layout, RUNTIME)
    except Exception as exc:                                  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning("startup layout migration failed: %s", exc)

    # The journal is owned here: started with the service, drained in the shutdown below.
    # The folder is read from deps at call time so it follows the runtime folder in use.
    global _journal
    from .recorder_link import sink_factory_from_env
    _journal = Journal(deps.RUNTIME / "journal", sink_for=sink_factory_from_env())
    try:
        await asyncio.to_thread(_journal.start)
    except Exception as exc:                                  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning("journal did not start: %s", exc)
        _journal = None
    # Asked once now so the first write's run does not wait for git.
    await asyncio.to_thread(run_store.compiler_version)

    # Captured so the bridge router's TELEM consumer thread can hand WS broadcasts
    # back via run_coroutine_threadsafe (host/telemetry._hw_telemetry_frame).
    hub._loop = asyncio.get_running_loop()

    # TWIN backend (docs/INSTRUMENT_MODEL_SPEC.md 13.1 item 8): (re)bind the twin's
    # SoftwareARC as hub.hardware_arc so park/interlock primitives run on the twin.
    # Repeated here because a hardware-path test can leave the singleton with
    # hardware_arc=None. Never clobbers a real bridge.
    from .hub import real_bridge as _real_bridge
    if hub.host.backend != "hardware" and _real_bridge(hub) is None:
        hub.host.hardware_arc = hub.host.arc

    # Restore the default sample's contact marks at startup, not only on sample
    # switch, so an erase survives a restart.
    n = restore_contacts(hub.host.world.sample, hub.sample_name)
    if n:
        hub.startup_note = f"Contacts: {n} restored from your marks"
    elif n == 0:
        hub.startup_note = ("Contacts: none, because you erased them. "
                            "Mark contacts to put some back.")

    from .design_store import restore_autosave
    design_note = await asyncio.to_thread(restore_autosave, hub)
    if design_note:
        hub.startup_note = "; ".join(filter(None, (hub.startup_note, design_note)))

    # FLEX provider (docs/FLEX_INTEGRATION_PLAN.md Sec B): two ZMQ REP sockets on
    # daemon threads sharing the hub. It serves tip-moving verbs (startScan,
    # startWrite, startApproach, ...). Optional: pyzmq missing, ports held, or
    # AFM_LITHO_FLEX=0 each log one line and the service runs without it. Started
    # after the loop capture above, and on the twin too (it reads the hub live).
    from .flex_provider import start_provider
    start_provider(hub)

    # Dead-man and engaged-idle cap (docs/FLEX_INTEGRATION_PLAN.md Sec E.1). A loop
    # task, not a thread, so it survives a dead provider thread; its retract goes
    # through safe_park to the failsafe watchdog thread, so it does not depend on
    # this loop either. Started on the twin too: the cap is about the tip.
    from .control import start_control_watch
    start_control_watch(hub)
    hub.host.start_beat()

    # Twin tip stream: `tip_pos` frames from the twin's own position, so the cockpit
    # marker moves on the twin as it does from the TELEM poll on hardware. `start`
    # creates no task on a hardware station (service/twin_tip_stream.py).
    from . import twin_tip_stream
    twin_tip_stream.start(hub)

    # The startup Surface replay is off by default: replaying every raw pass in the
    # sample's raw dir walks the whole directory in an uncancellable thread (~30 s+ per
    # TestClient lifespan on a box with thousands of .npz). Set
    # AFM_LITHO_REBUILD_SURFACE=1 (or true/yes) to run it.
    if os.environ.get("AFM_LITHO_REBUILD_SURFACE", "").strip().lower() in ("1", "true", "yes"):
        schedule_rebuild()
    else:
        _raw = getattr(getattr(hub, "surface", None), "raw_dir", None)
        _n_raw = len(list(_raw.glob("*.npz"))) if _raw is not None and _raw.is_dir() else 0
        _skip_msg = (
            f"Surface: startup replay skipped (AFM_LITHO_REBUILD_SURFACE off); "
            f"{_n_raw} raw pass{'' if _n_raw == 1 else 'es'} on disk. "
            f"Set AFM_LITHO_REBUILD_SURFACE=1 to replay them."
        )
        # hub.log only reaches WS clients CONNECTED NOW; a panel that connects
        # later would never learn why the Surface model starts empty. Also send it to the
        # service log at WARNING level: nothing in src/ configures logging and
        # uvicorn runs with log_level="warning", so an INFO line would be dropped.
        import logging
        logging.getLogger(__name__).warning(_skip_msg)
        await hub.log(_skip_msg)

    # Real-instrument image feed (off unless AFM_LITHO_IGOR_STREAM): listen for
    # Igor's SOCKIT push (afm_sockit.ipf AFM_StreamStart), broadcast each line.
    if os.environ.get("AFM_LITHO_IGOR_STREAM"):
        from ..hardware.sockit_receiver import run_receiver
        port = int(os.environ.get("AFM_LITHO_IGOR_STREAM_PORT", "9193"))
        asyncio.create_task(run_receiver(hub, port=port))

    # Real backend (hub.backend is the ARC backend only; MEASURE is independent):
    # accept Igor's inbound bridge (Igor dials out via AFM_LithoRunStart), read the
    # calibration, and hand the HardwareARC to the hub. A daemon thread so serve()'s
    # accept never blocks startup; the hub stays on the twin until Igor connects.
    if hub.host.backend == "hardware":
        import threading

        def _connect_hardware():
            from ..hardware.hardware_arc import HardwareARC
            port = int(os.environ.get("AFM_LITHO_BRIDGE_PORT", "9200"))
            # Every outcome prints to the console: a silent failure here leaves
            # the service in twin mode, simulating litho/scan with no indication.
            try:
                print(f"[hardware] backend=hardware: listening on :{port} for "
                      f"Igor to dial in (run scripts/start_bridge.py) ...",
                      flush=True)
                from ..hardware.igor import LazyIgor
                connect_hardware(lambda: HardwareARC.serve(port), LazyIgor)
            except Exception as exc:          # noqa: BLE001
                print(f"[hardware] bridge FAILED to attach "
                      f"({type(exc).__name__}: {exc}). The service stays in TWIN "
                      f"mode -- litho/scan will SIMULATE, not write. Likely: port "
                      f"{port} is still held by a previous service instance, or "
                      f"start_bridge.py dialed a different process. Kill the old "
                      f"service, restart this one, then re-run start_bridge.py.",
                      flush=True)

        threading.Thread(target=_connect_hardware, daemon=True).start()

    # Last: READY means the service is up, then the keeper's heartbeat (service/keeper_link.py).
    keeper_link.start(hub)


def connect_hardware(accept, make_com):
    """Attach Igor's bridge: `accept()` returns the HardwareARC once Igor has dialled in,
    `make_com()` its own COM handle. The step order is load-bearing: after a death, the
    socket make-safe is the first line on the link (host/after_death.py)."""
    arc = accept()
    # A restart after a death: the socket make-safe before any other line.
    from .host import after_death
    after_death.on_link(hub, arc)
    # Start the FAIL-SAFE WATCHDOG THREAD before anything can move the
    # tip (hardware/failsafe_thread.py; plan Sec B.3). Eager so COM is
    # initialised before the first emergency, and so mark_bridge_closed
    # (reader thread, must not block) always finds one running.
    from .host.safe_park import watchdog_for
    wd = watchdog_for(hub)
    print(f"[hardware] fail-safe watchdog thread up "
          f"(COM initialised: {wd.com_initialized})", flush=True)
    # WIRE THE CLOSE HOOK NOW, not at bind_hardware: arc.start() reads
    # STARTUP calibration over the bridge, and a drop during that read
    # would fire the one-shot _mark_closed into no hook, leaving the
    # service claiming "hardware" forever. bind_hardware re-checks the
    # level, so edge and already-passed edge are both covered.
    if arc.router is not None:
        arc.router.on_closed = hub.mark_bridge_closed
    hub.wire_igor_fault(arc.router)      # wire v3: a fault stamp before bind
    # Out-of-band abort escape: its OWN LazyIgor, NOT the watchdog's.
    # LazyIgor caches its COM proxy on first use, so sharing would let
    # whichever thread aborts first pick the fail-safe proxy's apartment.
    # bind_hardware routes arc.abort() via the watchdog; this is a fallback.
    arc.com_igor = make_com()
    print("[hardware] Igor connected; reading STARTUP calibration ...",
          flush=True)
    hub.host.hw_cal = arc.start()      # calibration (LVDT sens/offsets, ...)
    from .xy_cal import note_xy_cal, probe_getgv
    note_xy_cal(hub, "connect")
    _gv_msg = probe_getgv(hub, arc)
    if _gv_msg:
        print(f"[hardware] {_gv_msg}", flush=True)
    # Install the TELEM consumer BEFORE publishing the backend, so the
    # first poll frame is consumed (telemetry, rail-dwell guard, tip marker).
    install_hw_telemetry(arc)
    # bind_hardware, not a bare assignment: a dropped bridge then un-claims
    # "hardware" and retracts any in-flight run (plan Sec C.1).
    hub.bind_hardware(arc)
    print("[hardware] bridge LIVE -- hardware_arc attached. Litho/scan "
          "now drive the instrument.", flush=True)
    keeper_link.send_ready(hub)      # Igor's mark is readable from here on
    return arc


@app.on_event("shutdown")
async def _stop_flex_provider():
    """Close the FLEX sockets and the stop port cleanly and stop the control watcher.
    `stop()` unblocks the polls (they are never parked in a blocking recv) and terminates
    the ZMQ context, so a restart on the same ports does not fail to bind."""
    keeper_link.stop()
    from .flex_provider import stop_provider
    stop_provider()
    global _stop_port
    if _stop_port is not None:
        _stop_port.close()
        _stop_port = None
    from .control import stop_control_watch
    stop_control_watch(hub)
    hub.host.stop_beat()
    from . import twin_tip_stream
    twin_tip_stream.stop(hub)
    from .design_store import flush_autosave
    await asyncio.to_thread(flush_autosave, 5.0, retry_now=True)
    if _journal is not None:
        await asyncio.to_thread(_journal.stop)
    # The objects door's single-worker executor (litho_cmds._OBJECTS_EXECUTOR)
    # outlives a request; shut it down so the next app lifecycle in this process
    # (each test module starts one) builds a fresh one. wait=False: a running
    # orphan cannot be interrupted, and its daemon worker never delays exit.
    from .host.litho_cmds import shutdown_objects_executor
    shutdown_objects_executor()


class _NoCacheStatic(StaticFiles):
    """Serve static assets with Cache-Control: no-cache (revalidate every time).

    ES-module imports carry no ?v= buster, so without this a browser serves a
    stale module after the file changes. no-cache still allows cheap 304s.
    """

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", _NoCacheStatic(directory=WEB), name="static")


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    stream.subscribe(hub, websocket)
    try:
        while True:
            # telemetry heartbeat; also receives (ignored) client pings
            try:
                await asyncio.wait_for(websocket.receive_text(),
                                       timeout=stream.TELEMETRY_PERIOD_S)
            except TimeoutError:
                pass
            snap = stream.telemetry_frame(hub)
            await websocket.send_text(json.dumps(snap))
    except WebSocketDisconnect:
        stream.unsubscribe(hub, websocket)
    except Exception:  # noqa: BLE001 - any socket error just drops this client
        stream.unsubscribe(hub, websocket)
