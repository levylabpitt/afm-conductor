"""The Hub: one twin plus its websocket subscribers, and the process-global
`hub` singleton every router shares. Every field is declared in __init__."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from functools import partial

from fastapi import WebSocket

from ..hardware import channels
from ..hardware.channels import Z_OUT
from ..io import read_ibw
from ..twin.arc import SoftwareARC
from ..twin.physics import Instrument, Sample, World
from .control import ControlMixin
from .deps import DEFAULT_SAMPLE, RUNTIME, SAMPLES, raw_dir_for
from .host_state import HOST_FIELD_NAMES, HostState
from .passes import TwinInstrument
from .runs import on_pass_ended
from .sessions import Llab079Adapter, SessionStore
from .surface import SurfaceModel


def real_bridge(hub):
    """The REAL hardware ARC bridge attached to `hub`, or None on the twin.

    The twin binds its SoftwareARC to `hub.host.hardware_arc` (INSTRUMENT_MODEL_SPEC
    13.1 item 8), so branch on this, not `hardware_arc is not None`. Tolerates a
    stub whose `host` has no `hardware_arc`."""
    arc = getattr(hub.host, "hardware_arc", None)
    if arc is None or isinstance(arc, SoftwareARC):
        return None
    return arc


# Names Hub refuses to hold: every host field but `fault`.
_HOST_ONLY = frozenset(HOST_FIELD_NAMES) - {"fault"}


class Hub(ControlMixin):
    """One twin + its subscribers. `ControlMixin` (service/control.py) adds
    single-writer arbitration. Instrument state lives in `self.host` (HostState) and
    is read and written there; only `fault` also has a Hub property, which announces
    each change. `_loop` and `_show_tip_display` stay here as door fields:
    `_announce_fault` reads `_loop` to reach the pages, and the tip marker is display."""

    def __setattr__(self, name: str, value) -> None:
        # A host field set on the hub would be a second copy the host never reads.
        # `fault` passes through to its property, which writes host.fault.
        if name in _HOST_ONLY:
            raise AttributeError(f"hub.{name} is host state: set hub.host.{name}")
        object.__setattr__(self, name, value)

    def __init__(self):
        path = SAMPLES / DEFAULT_SAMPLE
        sample = Sample.from_ibw(path)
        instrument = Instrument.from_note(read_ibw(path).note_dict)
        world = World(sample=sample, instrument=instrument)
        arc = SoftwareARC(world)
        # The instrument's state: every name in HOST_FIELD_NAMES lives here only.
        self.host = HostState(instrument, world, arc, TwinInstrument(arc))
        self.host._sink = self._publish_to_pages
        self.host._on_pass_ended = partial(on_pass_ended, self)
        # Door state: page subscribers, sample, designs, sessions, runs.
        self.clients: set[WebSocket] = set()
        self.sample_name = sample.name
        self.store = SessionStore(RUNTIME / "sessions.jsonl")
        self.llab = Llab079Adapter()
        self.surface = SurfaceModel(raw_dir=raw_dir_for(sample.name),
                                    range_um=self.host.instrument.xy_range_um)
        self.pattern = None                 # current design (litho.compiler.Pattern)
        # The last uploaded SVG verbatim, so service/svg_background.py can edit the
        # original markup. Replaced on each upload, like `pattern`.
        self.design_source: dict | None = None   # {"name": <str>, "text": <svg str>}
        # Electrode ports for the current sample (service/electrodes.py); the connect
        # router re-reads disk per request so a sample switch never serves stale ports.
        from .electrodes import load_store as _load_ports_store
        self.ports, self.ports_seq = _load_ports_store(RUNTIME, sample.name)
        # Drift state (service/drift.py): reference frame or None; track = per-scan
        # [{t, dx_um, dy_um}]; total = accumulated drift (the compensation); enabled
        # = mutate only when on (always measured).
        from .drift import new_state as _new_drift_state
        self.drift = _new_drift_state()
        # Display-only tip marker streams (service/twin_tip_stream.py): the twin always
        # streams and each page shows or hides the marker itself.
        self._show_tip_display: bool = False
        self._twin_tip_sent_at: float = 0.0
        self._twin_tip_task: asyncio.Task | None = None
        # Captured at startup for run_coroutine_threadsafe from non-loop threads.
        self._loop = None
        # Litho settings the pages configure.
        self.transit = None
        self.litho_frame = None             # litho-frame dict, or None = from scan
        self.startup_note = None
        self.scan_params: dict = {}         # latest live param values from the panel
        # INSTRUMENT MODEL, distinct from the twin's physics `self.instrument`.
        self._init_instrument_model()
        # Run-id-keyed terminal records (plan Sec A.1/F.1): `completed` means the
        # files exist and are whole. Last 32 runs; lock-free by design (plan Sec B.4).
        from .runs import RunRegistry
        self.runs = RunRegistry()
        # SINGLE-WRITER ARBITRATION (plan Sec E): fields and rules in service/control.py.
        self._init_control()

    # ----------------------------------------------------- instrument model ----

    def _init_instrument_model(self) -> None:
        """Build `self.instrument_model` (INSTRUMENT_MODEL_SPEC sections 1, 10).

        Hardware: a missing/incomplete instrument.toml must NOT raise out of Hub();
        it latches `fault` = model_incomplete and installs a bare platform-default
        model, so `instrument_model` is never None."""
        from .instrument_model import (
            Field,
            FieldSource,
            InstrumentModel,
            ModelIncomplete,
            file_owned_platform,
            instrument_config_path,
            load_model,
        )

        if self.host.backend == "hardware":
            config_path = instrument_config_path()
            overrides_log = config_path.parent / "instrument_overrides.jsonl"
            # Platform is FILE-OWNED (section 14 item 4); a disagreeing env var only
            # warns. "cypher" is the fallback when the file declares none.
            file_platform = file_owned_platform(config_path)
            env_platform = os.environ.get("AFM_LITHO_PLATFORM")
            if env_platform and file_platform and env_platform != file_platform:
                logging.getLogger(__name__).warning(
                    "AFM_LITHO_PLATFORM=%s disagrees with instrument.toml "
                    "platform=%s; the file wins (section 14 item 4)",
                    env_platform, file_platform)
            platform = file_platform or "cypher"
            try:
                self.host.instrument_model = load_model(
                    platform=platform, startup_cal=None, config_path=config_path,
                    overrides_log=overrides_log)
                return
            except ModelIncomplete as exc:
                self.fault = {"reason": "model_incomplete", "at": time.time(),
                              "state": "fault", "detail": str(exc)}
                # Refused everywhere (tip_amp_gain None -> amp_gain_missing), never None.
                self.host.instrument_model = InstrumentModel(
                    platform=Field(platform, FieldSource.platform_default),
                    z_out_channel=Field(
                        Z_OUT,
                        FieldSource.platform_default),
                    z_verify_channels=Field((channels.Z_SENSOR, channels.DEFLECTION),
                                            FieldSource.platform_default),
                    z_park_v=Field(0.0, FieldSource.platform_default),
                    z_park_tol_v=Field(0.5, FieldSource.platform_default),
                    dac_rail_v=Field(10.0, FieldSource.platform_default),
                    tip_amp_gain=Field(None, FieldSource.platform_default),
                    tip_voltage_window_v=Field((0.0, 0.0), FieldSource.platform_default),
                    contact_setpoint_band_v=Field((0.0, 0.0), FieldSource.platform_default),
                    ac_setpoint_fraction_band=Field((0.0, 0.0), FieldSource.platform_default),
                    free_air_max_age_s=Field(30, FieldSource.platform_default),
                    xy_range_um=Field((90.0, 90.0), FieldSource.platform_default),
                    free_air_abs_band_v=Field(0.02, FieldSource.platform_default),
                    sum_floor_v=Field(0.5, FieldSource.platform_default),
                    sum_ceiling_frac=Field(0.9, FieldSource.platform_default),
                    z_rail_v=Field(140.0, FieldSource.platform_default),
                    interlock_s=Field(5.0, FieldSource.platform_default),
                    write_setpoint_relative_v=Field(0.1, FieldSource.platform_default),
                )
                return
        # Twin: read an instrument.toml if one exists (so overrides survive a
        # rebuild), but a malformed file falls back to defaults, never crashes Hub().
        config_path = instrument_config_path()
        loaded = False
        if config_path.exists():
            overrides_log = config_path.parent / "instrument_overrides.jsonl"
            try:
                self.host.instrument_model = load_model(
                    platform="twin", startup_cal=None, config_path=config_path,
                    overrides_log=overrides_log)
                loaded = True
            except Exception as exc:                     # noqa: BLE001
                logging.getLogger(__name__).warning(
                    "instrument config at %s could not be read (%s); using "
                    "twin defaults", config_path, exc)
        if not loaded:
            self.host.instrument_model = load_model(platform="twin", startup_cal=None,
                                               config_path=None, overrides_log=None)

    # ---------------------------------------------------------------- run ids --

    def next_run_id(self, kind: str) -> str:
        """Mint a run id (kind "scan"/"write"); open its in-flight record."""
        return self.runs.allocate(kind, backend=self.backend_live())

    def finish_run(self, run_id: str | None, **fields):
        """Close a run's terminal record; no-op on an unknown id (never fails a run)."""
        return self.runs.finish(run_id, **fields)

    # ------------------------------------------------------ backend liveness --

    def backend_live(self) -> str:
        """What is driving the tip right now: "hardware" only while a real bridge is
        attached AND not closed, else "twin". Every emitted `backend` field uses this.
        """
        return ("hardware"
                if (real_bridge(self) is not None and not self.host.bridge_closed.is_set())
                else "twin")

    def hw_bridge(self):
        """The REAL hardware ARC bridge, or None on the twin (see `real_bridge`)."""
        return real_bridge(self)

    def on_hardware(self) -> bool:
        """True only when a real hardware bridge is attached (not the twin ARC)."""
        return real_bridge(self) is not None

    def backend_label(self) -> str:
        """Like `backend_live`, but a configured-hardware station with its bridge down
        is "disconnected" rather than "twin". Used by status surfaces."""
        live = self.backend_live()
        if live == "hardware":
            return "hardware"
        return "disconnected" if self.host.backend == "hardware" else "twin"

    def state_label(self) -> str:
        """Coarse state: "fault" once latched, else "scanning"/"writing"/"idle".
        The full FLEX state is flex_dispatch.get_state."""
        if self.fault is not None:
            return "fault"
        if self.host.litho_thread is not None and self.host.litho_thread.is_alive():
            return "writing"
        task = self.host._scan_task
        if task is not None and not task.done():
            return "scanning"
        return "idle"

    def run_in_flight(self) -> bool:
        """True while a scan pass, litho write or armed continuous loop drives the
        tip. Decides whether a bridge drop fires a retract or only latches the fault."""
        if self.host.litho_thread is not None and self.host.litho_thread.is_alive():
            return True
        task = self.host._scan_task
        if task is not None and not task.done():
            return True
        return bool(self.host.scan_repeat.get("enabled"))

    def require_live_backend(self) -> str | None:
        """Refusal reason for a MUTATING call, or None when it may proceed.

        Closes the silent twin-fallback hazard (plan Sec C.2): a hardware-configured
        station must not run on the twin because Igor never connected or the bridge
        dropped. Twin stations are never refused. Also backs FLEX -32010.
        """
        if self.host.backend != "hardware":
            return None
        if real_bridge(self) is None:
            return ("backend is configured as hardware but Igor has not connected "
                    "(no bridge). Refusing rather than silently driving the TWIN -- "
                    "run scripts/start_bridge.py, then retry.")
        if self.host.bridge_closed.is_set():
            return ("the Igor bridge has CLOSED (state: disconnected). That is "
                    "terminal for this process -- restart the service and reconnect "
                    "the bridge.")
        return None

    def bind_hardware(self, arc) -> None:
        """Publish a connected hardware ARC, wire its bridge-close signal, and make
        sure the fail-safe watchdog is running. Wiring on_closed here keeps
        hardware/bridge_router.py free of a service import.

        THREE ordering hazards:

        1. **The close EDGE may already have passed.** `_mark_closed` is one-shot,
           and `arc.start()` (STARTUP calibration read) runs before this call. A
           bridge that died then would leave `backend: "hardware"` forever on a dead
           bridge, so after wiring the hook, re-check the LEVEL (`router.closed`).
        2. **`mark_bridge_closed` runs on the reader thread**, which must not block.
           Starting the watchdog blocks briefly, so it is started HERE, first.
        3. **`arc.abort()` must not attach COM from a pool thread** -- see
           `failsafe_park` below.
        """
        from .host.safe_park import watchdog_for
        wd = watchdog_for(self)              # started on THIS thread, not the reader's
        # Route the ARC out-of-band abort through the watchdog. arc.abort() runs on
        # the chained-scan worker thread, and LazyIgor caches its COM proxy on the
        # FIRST thread to use it; a dying pool thread would poison the fail-safe's
        # handle. park_now() is non-blocking, so the chunk-boundary abort never queues.
        arc.failsafe_park = wd.park_now
        router = getattr(arc, "router", None)
        if router is not None:
            router.on_closed = self.mark_bridge_closed
            router.on_igor_abort = self.note_igor_abort
        self.wire_igor_fault(router)
        banner = getattr(getattr(arc, "link", None), "version_banner", None)
        if banner:
            self._log_sync(f"SAFETY: {banner}")
        self.forget_twin_state()
        self.host.hardware_arc = arc
        if router is not None and router.closed:
            self.mark_bridge_closed("bridge closed before bind")
            return
        self._latch_igor_abort_fault(arc)
        self._clear_stale_igor_abort_flags(arc)

    def _latch_igor_abort_fault(self, arc) -> None:
        """Igor's 5 s abort backup may have run the vendor Withdraw while no service was
        connected (afm_litho_run.ipf AFM_AbortBackup): read its fault flag at connect and
        latch `hub.fault`. Igor keeps the flag until the operator clears the fault
        (CLEARFAULT), so a fault found here was never cleared. A failed read latches a fault
        too (fail-safe, 2026-10-27): an Igor that does not answer may be an older .ipf
        without the abort function, or in dry-run; the operator's clear then resets Igor.
        An ARC with no such read (the twin) latches nothing."""
        reader = getattr(arc, "read_abort_fault", None)
        if not callable(reader):
            return
        from .host.safe_park import latch_igor_abort_fault
        try:
            igor_fault = reader()
        except Exception as exc:                          # noqa: BLE001
            latch_igor_abort_fault(
                self, "found at connect",
                detail=f"could not read Igor's abort fault at connect ({exc})")
            return
        if igor_fault:
            latch_igor_abort_fault(self, "found at connect")

    def _clear_stale_igor_abort_flags(self, arc) -> None:
        """CLEARFAULT at connect when no fault is latched here, so Igor drops abort flags left
        from before (a fault clear made while no bridge was connected, or a service that died
        mid-park): a stale root:AFM_abortStopped refuses wire v3 PLAY with `mark`. A latched
        fault leaves Igor's flags for the operator's clear; this includes a fault Igor still
        reports after a clear made while the bridge was down (`igor_clear_pending` only notes
        that clear): it was latched again at connect, and the operator clears once more. A Python park under way, or Height
        not under 1 V, may be a park whose 5 s backup CLEARFAULT would cancel: Igor is left
        alone then. An ARC with no such verb (the twin) is skipped."""
        from ..hardware.retract import RETRACTED_HEIGHT_V
        clear, read = getattr(arc, "clear_fault", None), getattr(arc, "read", None)
        if not callable(clear) or not callable(read):
            return
        why = (" (a fault clear made while the bridge was down)"
               if getattr(self.host, "igor_clear_pending", False) else "")
        if getattr(self, "fault", None) is not None:
            again = ("; Igor still reports a fault, so it is latched again and the operator "
                     "clears once more" if why else "")
            self._log_sync(f"connect: a fault is latched, so Igor's abort flags are "
                           f"kept{why}{again}")
            return
        if getattr(self.host, "_safe_park", {}).get("active"):
            self._log_sync(f"connect: a park is running, so Igor's abort flags are kept{why}")
            return
        try:
            height = float(read(channels.HEIGHT))
        except Exception as exc:                          # noqa: BLE001
            self._log_sync(f"WARNING connect: Height read failed ({exc}), so Igor's abort "
                           f"flags were not cleared{why}")
            return
        if not height < RETRACTED_HEIGHT_V:              # NaN counts as not retracted
            self._log_sync(f"connect: Height {height:.3g} V is not under 1 V (a park may be "
                           f"under way), so Igor's abort flags are kept{why}")
            return
        try:
            clear()
        except Exception as exc:                          # noqa: BLE001
            self._log_sync(f"WARNING connect: CLEARFAULT to Igor failed ({exc}); Igor's "
                           f"abort flags may still be set{why}")
            return
        self.host.igor_clear_pending = False
        self._log_sync(f"connect: no fault latched, CLEARFAULT sent so Igor holds no stale "
                       f"abort flag{why}")

    def forget_twin_state(self) -> None:
        """Discard every hub field the twin produced that describes the instrument.

        Called from bind_hardware: this state is not the real instrument's and must
        not be trusted. Imaging mode goes back to unknown, so the next hardware
        approach reads it live (READXPT); the scan frame goes back to none, so the
        litho angle takes its no-scan default; the free-air baseline and the engaged Z
        loop record return to their service-start values. `engaged` is kept on
        purpose: the first hardware scan's XY transit withdraws on it, and the real
        tip may be down. Then the panel is told to re-read.
        """
        self.host.imaging_mode = None
        self.host.free_air = None
        self.host.z_loop = {}
        self.host.inst.scan = None
        from .twin_plugin import forget
        forget(self)
        self.host.relative_setpoint["last_free_air_v"] = None
        self.host.relative_setpoint["last_effective_v"] = None
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self.broadcast({"type": "instrument_rebound", "backend": "hardware",
                                "imaging_mode": None}), loop)
        except Exception:                                 # noqa: BLE001,S110
            pass

    def wire_igor_fault(self, router) -> None:
        """Wire v3: latch Igor's fault stamp as hub.fault. Called before arc.start() (app.py)
        and again at bind; a fault already seen (a TICK, or HELLO's fault=1) latches now."""
        if router is None or not hasattr(router, "on_fault"):
            return
        router.on_fault = self.latch_igor_fault
        if router.igor_fault:
            self.latch_igor_fault("Igor's fault was set before the service was bound")

    def latch_igor_fault(self, detail: str) -> None:
        """Wire v3: Igor's fault stamp rose to 1 (its 5 s backup ran the vendor withdraw).
        Latched as the park's own Igor fault. Runs on the router's reader thread."""
        from .host.safe_park import latch_igor_abort_fault
        latch_igor_abort_fault(self, "fault stamp on the wire", detail)

    def note_igor_abort(self, frame: str) -> None:
        """Igor reported an abort (`frame`). Runs on the ROUTER'S READER THREAD: nothing
        that can block. One the service did not start (Igor's AFM Abort panel) while a
        run is in flight stops the whole run, as the cockpit Abort does. The service's
        own stops set the run's flag before they reach Igor, so their reports only mark
        the stop again. With no run in flight the stop is still marked, so a start part
        way through its reads refuses; every start clears the flags, so nothing latches."""
        try:
            from .host.safe_park import park_on_igor_abort, signal_run_abort
            if self.run_in_flight() and not self._run_stop_signalled():
                self.host._igor_abort_seq += 1
                park_on_igor_abort(self, frame, self._log_sync)
            else:
                signal_run_abort(self)
        except Exception as exc:                          # noqa: BLE001
            self._log_sync(f"SAFETY CRITICAL: Igor reported an abort and stopping the "
                           f"run failed ({exc!r}).")

    def _run_stop_signalled(self) -> bool:
        """True once a stop was signalled for the run in flight (its own flag)."""
        if self.host.litho_thread is not None and self.host.litho_thread.is_alive():
            return self.host._hw_litho_abort.is_set()
        task = self.host._scan_task
        if task is not None and not task.done():
            return self.host._hw_abort.is_set()
        return self.host._scan_repeat_stop.is_set()

    def mark_bridge_closed(self, detail: str | None = None) -> None:
        """The bridge dropped. Runs on the ROUTER'S READER THREAD: no loop, no await,
        nothing that can block.

        Latches `bridge_closed` and a terminal `fault`. If a run was in flight, also
        demands a verified retract via the failsafe watchdog, the only path that can
        reach Igor when the bridge socket is what died (COM, CoInitialize'd thread).
        """
        if self.host.bridge_closed.is_set():
            return
        self.host.bridge_closed_at = time.time()
        self.host.bridge_closed.set()
        in_flight = False
        try:
            in_flight = self.run_in_flight()
        except Exception:                                 # noqa: BLE001,S110
            pass
        self.fault = {"reason": "bridge_closed", "at": self.host.bridge_closed_at,
                      "state": "disconnected", "detail": detail,
                      "was_running": in_flight}
        if not in_flight:
            return
        # The import is inside the try: this runs on the reader thread, which must
        # never raise.
        try:
            from .host.safe_park import PARK_BRIDGE_DROP, request_park
            request_park(self, "bridge_closed", PARK_BRIDGE_DROP, log=self._log_sync)
        except Exception as exc:                          # noqa: BLE001
            self._log_sync(f"SAFETY CRITICAL: the park request failed ({exc!r}). "
                           f"The tip may still be engaged.")

    # Every open page follows the latch: each change of `fault` (latch, clear) is
    # broadcast as {"type": "fault", "fault": <record or null>}. A page that connects
    # later seeds from GET /api/state instead. Latches happen on worker and reader
    # threads, so the send is scheduled on the service loop, never awaited here.
    @property
    def fault(self) -> dict | None:
        return self.host.fault

    @fault.setter
    def fault(self, value: dict | None) -> None:
        # The latch is assigned FIRST and the broadcast can never raise: a latch that a
        # failed send left unset would let the next engage through.
        old = self.host.fault
        self.host.fault = value
        try:
            if old != value:
                self._announce_fault(value)
        except Exception:                                 # noqa: BLE001,S110
            pass

    def _announce_fault(self, value: dict | None) -> None:
        try:
            loop = getattr(self, "_loop", None)
            if loop is None or loop.is_closed():
                return
            record = json.loads(json.dumps(value, default=str))
            asyncio.run_coroutine_threadsafe(
                self.broadcast({"type": "fault", "fault": record}), loop)
        except Exception:                                 # noqa: BLE001,S110
            pass

    def _log_sync(self, text: str) -> None:
        """Log from a NON-loop thread (hub.log is a coroutine). Console always (what
        the operator watches); the websocket too when a loop is running, best effort."""
        print(f"[hub] {text}", flush=True)
        loop = self._loop
        if loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self.log(text), loop)
        except Exception:                                 # noqa: BLE001,S110
            pass

    async def _publish_to_pages(self, msg: dict):
        from .requests import stream
        await stream.publish(self, msg)

    async def broadcast(self, msg: dict):
        """Door-side name for `host.emit`; the ~45 call sites keep this spelling."""
        await self.host.emit(msg)

    async def notify_design_changed(self):
        """The ONE design-mutation notification (INSTRUMENT_MODEL_SPEC sec 4).

        Every design-mutating door, REST and FLEX, calls this exactly once per
        mutation so the cockpit re-adopts the authoritative design and estimate."""
        from .design_store import request_autosave
        request_autosave(self)
        await self.broadcast({"type": "design_changed"})

    async def refresh_xy_cal(self, source: str) -> None:
        """Re-read the XY LVDT calibration into `hw_cal` (hardware only; the Cypher
        rescales it on every motor move). See service/xy_cal.py."""
        from .xy_cal import refresh_xy_cal_async
        await refresh_xy_cal_async(self, source)

    async def log(self, text: str):
        await self.broadcast({"type": "log", "t": time.time(), "msg": text})


hub = Hub()
