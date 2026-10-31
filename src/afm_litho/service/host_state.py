"""HostState: the instrument-owned state, split out of `Hub`.

Everything here describes the tip, the controller link and the passes that drive
them. The safety guards read it directly. `Hub` keeps the door, run-store and
design state; code reads and writes the names below on `hub.host`, and Hub refuses
to hold them. Each field lives on exactly one side (tests/test_hub_host_split.py).
"""

from __future__ import annotations

import asyncio
import os
import threading

from .. import clock
from .backends import arc_backend
from .safety import RailDwellGuard, RailDwellLimits
from .twin_player import speed_from_env

# Every attribute HostState defines, in group order. Hub refuses all of them except
# `fault`, whose Hub property reads host.fault and announces each change to the pages.
HOST_FIELD_NAMES = (
    # Where `emit` sends events, and when the host last spoke.
    "_sink", "_last_heard",
    # Where `pass_ended` goes (a base handler, not the pages) and the runs it has ended.
    "_on_pass_ended", "_ended_runs",
    # The twin plugin and its playback speed.
    "instrument", "world", "arc", "inst", "speed", "twin_speed", "_twin_plugin",
    "_twin_plugin_sample", "twin_scan_angle_deg",
    # The configured backend, the ARC and bridge handles, and the calibration.
    "backend", "hardware_arc", "hw_cal", "xy_cal_status", "bridge_closed",
    "bridge_closed_at", "instrument_model",
    # Tip engaged, imaging mode, approach, latched fault.
    "engaged", "_approach_in_flight", "imaging_mode", "fault", "igor_clear_pending", "z_loop",
    "free_air", "relative_setpoint",
    # Last-used setpoints and the last scan body.
    "last_write_params", "last_scan_params", "last_scan_config",
    # The scan pass: worker task, epoch, deferred withdraw, abort flags, stop marks.
    "_scan_task", "_scan_epoch", "_hw_withdraw_after", "_hw_withdraw_epoch",
    "_scan_pump_error", "_hw_abort", "_hw_litho_abort", "_run_abort_seq", "_igor_abort_seq",
    "_scan_stop_seq", "_scan_row",
    # The continuous-imaging supervisor.
    "scan_repeat", "_scan_repeat_task", "_scan_repeat_stop",
    # The write pass: worker, thread, run, capture, ledger, start estimate.
    "litho_worker", "litho_thread", "hw_litho_capture", "written_accum",
    "written_objects", "_litho_partial", "_start_estimate", "_estimate_queue",
    # Telemetry and the rail-dwell guard.
    "_hw_telemetry", "rail_guard", "_rail_lock", "_rail_feeding", "_rail_last_row",
    "_rail_pending_trip",
    # The sum and deflection interlocks.
    "_sum_baseline", "_sum_low_streak", "_defl_high_streak", "_interlock_pending_trip",
    "write_interlocks", "_interlock_feed_thread", "_pending_interlock_reason",
    # The park: last outcome, request flag and reason, fail-safe thread.
    "_safe_park", "last_park", "_safe_park_now", "_safe_park_reason", "_failsafe",
    # The engaged-idle cap's clock.
    "_engaged_since",
)


_ENDED_RUNS_KEPT = 32      # run ids remembered so a repeat pass_ended is dropped


class HostState:
    """The instrument's state. Built by `Hub`, which passes in the twin objects."""

    def __init__(self, instrument, world, arc, inst):
        # Where emitted events go; Hub wires it to the page broadcast.
        self._sink = None
        # clock.now() of the host's last emit or tick; None until the first one. The
        # board answers `disconnected` when this goes stale (service/board.py).
        self._last_heard: float | None = None
        # The base handler `pass_ended` calls, and the last few run ids already ended.
        self._on_pass_ended = None
        self._ended_runs: dict = {}
        # Twin plugin.
        self.instrument = instrument
        self.world = world
        self.arc = arc
        self.inst = inst
        self.speed = float(os.environ.get("AFM_LITHO_SCAN_SPEED") or 1.0)  # twin scan: 1 = real time
        self.twin_speed: float = speed_from_env()   # 1 = real time, 0 = unlimited
        # The twin plugin (service/twin_plugin.py), built on the first twin write or scan;
        # `twin_scan_angle_deg` is the last twin scan's angle, which the litho frame follows.
        self._twin_plugin = None
        self._twin_plugin_sample = None      # the sample object it was built from
        self.twin_scan_angle_deg: float | None = None
        # Backend and controller link. `hardware_arc` is the twin SoftwareARC on a twin
        # station (INSTRUMENT_MODEL_SPEC 13.1 item 8), None on hardware until
        # bind_hardware. `bridge_closed` is a threading.Event set on the reader thread.
        self.backend = arc_backend()
        self.hardware_arc = arc if self.backend != "hardware" else None
        self.hw_cal: dict = {}
        self.xy_cal_status: dict = {"refreshed_at": None, "source": None, "error": None}
        self.bridge_closed = threading.Event()
        self.bridge_closed_at: float | None = None
        self.instrument_model = None            # built by Hub._init_instrument_model
        # Tip and mode. `engaged` gates the mode switch; `fault` is the terminal latch
        # (no re-accept path); `free_air` is the withdrawn-read baseline.
        self.engaged: bool = False
        self._approach_in_flight: int = 0
        self.imaging_mode: str | None = None
        self.fault: dict | None = None
        # an operator fault clear that could not reach Igor; CLEARFAULT goes at the next connect
        self.igor_clear_pending: bool = False
        self.z_loop: dict = {}
        self.free_air = None
        self.relative_setpoint: dict = {
            "enabled": False, "settle_s": 4.0,
            "last_free_air_v": None, "last_effective_v": None}
        # Last-used setpoints and scan body.
        self.last_write_params: dict = {}
        self.last_scan_params: dict = {}
        self.last_scan_config: dict | None = None
        # Scan pass. The epoch guards a deferred withdraw against a fresh pass; the
        # abort events are cooperative flags polled by the workers; the stop counters
        # let start_write refuse when a stop arrived during its estimate.
        self._scan_task: asyncio.Task | None = None
        self._scan_epoch: int = 0
        self._hw_withdraw_after = False
        self._hw_withdraw_epoch: int | None = None
        self._scan_pump_error: str | None = None
        self._hw_abort = threading.Event()
        self._hw_litho_abort = threading.Event()
        self._run_abort_seq: int = 0
        # Counts the runs stopped by an abort that started in Igor (hub.note_igor_abort).
        self._igor_abort_seq: int = 0
        self._scan_stop_seq: int = 0
        self._scan_row: int | None = None
        # Continuous-imaging supervisor. The stop is a threading.Event because
        # safe_park sets it from non-loop threads.
        self.scan_repeat: dict = {
            "enabled": False, "config": None, "count": 0, "started_at": None,
            "errors": 0, "max_frames": None, "max_seconds": None,
            "interval_s": 0.0, "withdraw_on_stop": False, "stopped_reason": None}
        self._scan_repeat_task: asyncio.Task | None = None
        self._scan_repeat_stop = threading.Event()
        # Write pass. `litho_worker` is the abortable handle safe_park joins before any
        # Z command; `_litho_partial` is spliced into the trace only, never the ledger.
        self.litho_worker = None
        self.litho_thread = None
        self.hw_litho_capture: dict = {}
        self.written_accum: list[dict] = []
        self.written_objects: set = set()
        self._litho_partial: dict | None = None
        self._start_estimate = None
        self._estimate_queue = None
        # Telemetry and the rail-dwell guard. `_rail_lock` guards `rail_guard` between
        # the router thread and the loop's finish_frame.
        self._hw_telemetry: dict | None = None
        self.rail_guard = RailDwellGuard(RailDwellLimits())
        self._rail_lock = threading.Lock()
        self._rail_feeding: bool = False
        self._rail_last_row: int | None = None
        self._rail_pending_trip: dict | None = None
        # Sum-collapse and deflection interlocks (thresholds in host/telemetry.py).
        self._sum_baseline: float | None = None
        self._sum_low_streak: int = 0
        self._defl_high_streak: int = 0
        self._interlock_pending_trip: dict | None = None
        self.write_interlocks = None
        self._interlock_feed_thread = None
        self._pending_interlock_reason: str | None = None
        # Park. Any thread may set `_safe_park_reason` then `_safe_park_now`; the
        # fail-safe thread (`_failsafe`) retracts.
        self._safe_park: dict = {
            "count": 0, "active": False, "last_reason": None, "verified": None,
            "critical": False, "transport": None, "z_v": None, "at": None}
        self.last_park = None
        self._safe_park_now = threading.Event()
        self._safe_park_reason: str | None = None
        self._failsafe = None
        # The tick time at which the tip was first seen engaged with nothing in flight;
        # written only by the engaged-idle cap (host/deadman.py).
        self._engaged_since: float | None = None

    def heard(self) -> None:
        """Note that the instrument side is alive. Called by every emit and by the
        host's own beat (host/beat.py), so an idle but healthy host is never silent."""
        self._last_heard = clock.now()

    def pass_ended(self, run_id, outcome: str, written, files: list) -> None:
        """Tell the base a pass is over: `outcome` is "completed", "aborted" or "failed";
        `written` is the object count for a write, else None; `files` are paths already on
        disk. Each run id ends once however many exit paths reach here. Not a page frame:
        the handler is a base function, so the pages see no new event."""
        self.heard()
        if run_id is not None and run_id in self._ended_runs:
            return
        if self._on_pass_ended is not None:
            # The handler releases the claim; a failing one must not raise into the pass, or
            # the host skips its persistence after the release. A failed run is not marked
            # ended, so the pump's `finally` backstop calls again and retries the release.
            try:
                self._on_pass_ended(run_id, outcome, written, list(files))
            except Exception as exc:                      # noqa: BLE001
                print(f"[host] pass_ended handler failed for {run_id}: {exc!r}", flush=True)
                return
        if run_id is not None:
            self._ended_runs[run_id] = True
            while len(self._ended_runs) > _ENDED_RUNS_KEPT:
                del self._ended_runs[next(iter(self._ended_runs))]

    def start_beat(self) -> None:
        """Start the host's own beat (host/beat.py). Idempotent."""
        from .host.beat import start_host_heartbeat
        start_host_heartbeat(self)

    def stop_beat(self) -> None:
        from .host.beat import stop_host_heartbeat
        stop_host_heartbeat()

    async def emit(self, msg: dict) -> None:
        """The one place the instrument side publishes an event ({"type": ...}).

        Everything the host announces passes through here, in order, exactly once.
        The in-process board (6c1) subscribes at this point. Today the only
        subscriber is the page broadcast that Hub installs as the sink."""
        self.heard()
        if self._sink is not None:
            await self._sink(msg)
