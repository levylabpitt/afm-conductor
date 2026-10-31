"""The Asylum plugin: today's hardware code behind the `Instrument` interface.

Every verb calls the function the service calls today, with the same arguments in the same
order, so Igor runs exactly the commands it runs today (the golden traces prove it):

    write     hardware/lithography_run.py  LithographyRun.write_object
    move      hardware/xy_transit.py       xy_transit
    engage    hardware/hw_approach.py      hw_engage
    withdraw  hardware/hw_approach.py      hw_withdraw
    read      hardware/hardware_arc.py     HardwareARC.td_ReadValue
    park      service/host/safe_park.py  the fail-safe watchdog (out-of-band trigger + safe_park)
    abort     service/host/safe_park.py  _signal_stop, then the watchdog's park_now
    scan      service/hw_scan.py           run_hw_scan_chained

Known reference to revisit: the plugin is built with the service's `hub`. Today's functions
take it (safe_park(hub, ...), the XY calibration `hub.hw_cal`, the abort events, `hub.engaged`
that gates the imaging-mode switch), so the plugin reaches into the service until those move
behind the interface. The plugin looks the connected `hub.hardware_arc` up at every command, so
a reconnect is picked up (call `listen` again to re-attach the telemetry).

Not done here, because the service does it before it starts a pass, and 4.3 must keep it: the
command guards (SUM ceiling, free-air establishment, check_write, the fault gate), interlock
arming and disarming (arm_interlocks), run bookkeeping and the cockpit's line stream. A write
only refuses on a latched fault or a stop signalled while it starts.

Private service names used on purpose, to move with the hub reference: _signal_stop,
_watchdog_wait_bound, _WITHDRAWN_HEIGHT_TOL_V, and LithographyRun._object_setpoint / _last_source.
`write` sets `hub.litho_worker` for its duration so a park can join it, unless the service owns
the write (`service_owns_write`): then the service keeps that field.
`engage` and `withdraw` with `service_owns_approach` run only the Igor sequence; the service
keeps the guards, `hub.engaged`, the interlocks and the park (see the class docstring).

Calls block until the instrument has finished: call them from a worker thread, never from the
service's event loop. A scan and the imaging-mode reconcile run their own short loop with
asyncio.run, so on the event loop they fail with RuntimeError (the verb is refused).
"""

from __future__ import annotations

import asyncio
import math
import re
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .. import clock
from ..hardware import channels
from ..hardware.lithography_run import (FreeAirReadFailed, FreeAirRefused, LithographyRun,
                                        WriteAborted, WriteSetpoint)
from ..hardware.xy_transit import xy_transit
from ..run.model import Platform
from ..hardware.hw_approach import hw_engage, hw_withdraw
from ..service.hw_scan import run_hw_scan_chained, scan_capture_channels
from .commands import (Citation, EngageAt, MoveTo, ParkTip, ReadChannels, Reading, Recorded,
                       Result, ScanRaster, Verb, WithdrawTip, WriteObject)
from .reports import Activity, Status, StatusListener, TelemetryFrame, TelemetryListener

# The engage gains and settle the service's startApproach uses when a script names none.
ENGAGE_PGAIN = 0.0
ENGAGE_IGAIN = 10.0
ENGAGE_SETTLE_S = 6.0

# The input waves a write always captures (bank 0, 1 and 2). Lateral is armed only when bank 2
# takes the UserIn2 + Lateral pair; if it did not, the stream has no Lateral.
WRITE_CAPTURE = (channels.USER_IN2, channels.LATERAL, channels.Z_SENSOR, channels.DEFLECTION,
                 channels.X_SENSOR, channels.Y_SENSOR)


def _ascii(text: str) -> str:
    """Typeable ASCII on one line, for a Status or a Result reason."""
    return re.sub(r"\s+", " ", str(text)).encode("ascii", "replace").decode().strip() or "-"


def _number(value) -> float:
    """A float for a telemetry field: NaN for anything unreadable or infinite."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


@dataclass(frozen=True)
class _RasterConfig:
    """The fields `run_hw_scan_chained` reads from the service's ScanConfig."""

    size_um: float
    pixels: int
    lines: int
    line_rate_hz: float
    x_offset_um: float
    y_offset_um: float
    mode: str
    setpoint: float
    angle_deg: float
    direction: str
    trace_retrace: bool
    engage: bool
    pgain: float | None = None
    igain: float | None = None


class _ModeHub:
    """The hub as `ensure_software_mode` uses it, with its log going to the plugin's status
    instead of the service's websocket (that coroutine would run on this thread's own loop)."""

    def __init__(self, hub, say):
        object.__setattr__(self, "_hub", hub)
        object.__setattr__(self, "_say", say)

    def __getattr__(self, name):
        return getattr(self._hub, name)

    def __setattr__(self, name, value):
        setattr(self._hub, name, value)

    async def log(self, text: str) -> None:
        self._say("idle", text, None)


class _ScanNotes:
    """What `run_hw_scan_chained` reports as it goes (the `notes` it takes), as Status."""

    stroke = 0

    def __init__(self, say, cites: dict[str, Citation], forward=None):
        self._say, self._cites, self._forward = say, cites, forward

    def outcome(self, kind, make=None, **body) -> None:
        if self._forward is not None:
            self._forward.outcome(kind, make, **body)
        try:
            found = {**(make() if make is not None else {}), **body}
            text = {"move": "moved to the first point of the scan",
                    "engage": "engaged for the scan",
                    "pass": ("scan stopped" if found.get("aborted")
                             else f"scanned {found.get('lines_delivered', 0)} lines")}.get(kind)
            activity = {"move": "moving", "engage": "engaged", "pass": "scanning"}.get(kind)
            if text is not None:
                self._say(activity, text, self._cites.get(kind))
        except Exception:  # noqa: BLE001, S110 - a note never stops the scan
            pass

    def abort(self, reason) -> None:
        if self._forward is not None:
            self._forward.abort(reason)
        self._say("idle", f"scan aborted: {reason}", None)

    def stamp_step(self, kind) -> None:
        return None

    begin = stamp_step

    def step_of(self, kind) -> None:
        return None


class _WriteHandle:
    """The write in flight, as `safe_park` sees it on `hub.litho_worker`: a park signals the
    abort event, then waits (bounded) for the write command to return before any Z command."""

    def __init__(self, abort_event):
        self._abort_event = abort_event
        self._finished = threading.Event()

    def request_abort(self) -> None:
        self._abort_event.set()

    def join(self, timeout=None) -> None:
        self._finished.wait(timeout)

    def is_alive(self) -> bool:
        return not self._finished.is_set()

    def finish(self) -> None:
        self._finished.set()


def _align_to_end(kinds: list[str], steps: tuple[Citation, ...]) -> dict[str, Citation]:
    """Cite the last sub-steps with the command's steps, in order; sub-steps before the first
    cited step go uncited. The last ones (pass, back-off) are never optional."""
    return dict(zip(kinds[len(kinds) - len(steps):], steps, strict=False)) if steps else {}


class AsylumInstrument:
    """The Asylum/ARC instrument behind the interface.

    `amp_gain` is the external tip-amplifier gain the DAC output is divided by (the write's
    `amp_gain`); the interface has no field for it, and the instrument model holds it, so the
    caller passes it in. The engage gains and settle are the service's defaults.

    `service_owns_write=True`: the service ran the AC guard, refreshed the XY calibration and
    cleared the abort flag, owns the worker thread, and withdraws the tip itself when a write
    stops (so it can journal and set the step label first, and save its record after). The
    plugin then does none of those and leaves `hub.litho_worker` alone. Such a service may set:

    - `setpoint_log(msg)`: the one line per object (free air read, setpoint used);
    - `on_step(kind, make)`: each write step's outcome, `make()` giving what it found;
    - `object_setpoints` (read only): one record per object, filled as the write goes;
    - `failure`: the exception that stopped the last write, None for a plain abort or refusal.

    A stopped write's Result holds the captured prefix and `landed`.

    `service_owns_scan=True`: the service reconciled the imaging mode, refreshed the XY
    calibration, cleared the scan abort flag and set `hub.engaged` for an engaging scan; it
    owns the pump and the worker thread and handles a stopped scan itself (the abort already
    retracts inside the pass, and the service's deferred withdraw follows). The plugin then
    does none of those and withdraws nothing. Such a service may set:

    - `scan_message(message)`: each message the scan streams (log lines, the live rows and
      `scan_done`), called on the scan's own thread;
    - `scan_notes`: the service's step notes, given every outcome the scan reports;
    - `scan_gains`: the (pgain, igain) Z loop panel gains the scan engages with, or None;
    - `scan_cancelled()`: true once the service's pump was cancelled; the scan then stops
      before it arms the Z loop or fires the raster;
    - `scan_engaged`: whether the tip was down when the pump started (default: read at the scan);
    - `scan_summary` (read only): what the last scan came back with (its lines, captured
      channels and line indices, so a stopped scan's partial frame survives);
    - `failure`: the exception that stopped the last scan.

    `service_owns_approach=True`: the service stopped any pass in flight, reconciled the
    imaging mode, ran the free-air, SUM-ceiling and fault guards, and records the engagement
    itself (`hub.z_loop`, `hub.engaged`, the scan epoch, the interlocks, the park latch). The
    plugin's `engage` and `withdraw` then send only the Igor commands and read the result
    back: they set none of those, signal no stop, and a failed engage is not parked (the
    service parks a failed approach itself). Such a service reads:

    - `snapshot`: what the last engage or withdraw read back (deflection, Z output,
      `reached_setpoint`, `verify_timeout`);
    - `failure`: the exception that stopped the last engage or withdraw.

    The service may set `engage_mode`: the mode word as the door received it ("current" is a
    contact-style loop but not a `Mode`); `engage` hands it to `hw_engage` instead of the
    command's validated mode.

    The park is not part of this: it stays in the service (`safe_park_async`).
    """

    def __init__(self, hub, *, amp_gain: float | None = None, engage_pgain: float = ENGAGE_PGAIN,
                 engage_igain: float = ENGAGE_IGAIN, engage_settle_s: float = ENGAGE_SETTLE_S,
                 service_owns_write: bool = False,
                 setpoint_log: Callable[[str], None] | None = None,
                 service_owns_scan: bool = False,
                 service_owns_approach: bool = False):
        if amp_gain is not None and not amp_gain > 0:
            raise ValueError("AsylumInstrument: amp_gain must be more than 0")
        self._hub = hub
        self._amp_gain = None if amp_gain is None else float(amp_gain)
        self._engage = (float(engage_pgain), float(engage_igain), float(engage_settle_s))
        self._on_telemetry: TelemetryListener | None = None
        self._on_status: StatusListener | None = None
        self._router = None
        self._previous_telem = None
        self._run: LithographyRun | None = None     # the write run: hover needs the last object
        self._abort_reason: str | None = None
        self._service_owns_write = service_owns_write
        self._setpoint_log = setpoint_log
        self._object_setpoints: list[dict] = []
        self._service_owns_scan = service_owns_scan
        self._scan_summary: dict | None = None
        self._service_owns_approach = service_owns_approach
        self._snapshot: dict | None = None
        self.engage_mode: str | None = None
        self.scan_message: Callable[[dict], None] | None = None
        self.scan_notes = None
        self.scan_cancelled: Callable[[], bool] | None = None
        self.scan_engaged: bool | None = None
        self.scan_gains: tuple[float | None, float | None] | None = None
        self.on_step: Callable[..., None] | None = None
        self.failure: Exception | None = None

    @property
    def object_setpoints(self) -> list[dict]:
        return self._object_setpoints

    @property
    def snapshot(self) -> dict | None:
        return self._snapshot

    @property
    def scan_summary(self) -> dict | None:
        return self._scan_summary

    # --- the instrument we are connected to ----------------------------------------

    def _arc(self):
        from ..service.hub import real_bridge
        arc = real_bridge(self._hub)
        if arc is None:
            raise RuntimeError("no Asylum instrument is connected")
        return arc

    @property
    def platform(self) -> Platform:
        return "cypher" if self._arc().is_cypher else "mfp3d"

    # --- telemetry and status ------------------------------------------------------

    def listen(self, on_telemetry: TelemetryListener, on_status: StatusListener) -> None:
        """Set the subscriber and chain onto the router's TELEM hook: the service's own
        always-on consumer (tip marker, rail guard, interlocks) keeps receiving every frame."""
        self._on_telemetry, self._on_status = on_telemetry, on_status
        router = self._arc().router
        if self._router is not router:
            self._detach()
            self._router, self._previous_telem = router, router.on_telem
            router.on_telem = self._telem

    def _telem(self, x, y, zdrive, defl, sum_v, amp) -> None:
        """One TELEM frame (raw volts) from the router's reader thread: pass it on to the
        service's consumer, then push it with x and y in um by today's conversion."""
        previous = self._previous_telem
        if previous is not None:
            previous(x, y, zdrive, defl, sum_v, amp)
        push = self._on_telemetry
        if push is None:
            return
        from ..service.tip_stream import tippos_to_um
        try:
            x_um, y_um = tippos_to_um(x, y, self._hub.host.hw_cal)
        except (TypeError, ValueError):
            x_um = y_um = float("nan")
        push(TelemetryFrame(clock.now(), _number(x_um), _number(y_um), _number(zdrive),
                            _number(defl), _number(sum_v), _number(amp)))

    def _detach(self) -> None:
        if self._router is not None and self._router.on_telem == self._telem:
            self._router.on_telem = self._previous_telem
        self._router = self._previous_telem = None

    def close(self) -> None:
        """Stop pushing; the instrument is left as it is."""
        self._detach()
        self._on_telemetry = self._on_status = None

    def _say(self, activity: Activity, text: str, cite: Citation | None = None) -> None:
        listener = self._on_status
        if listener is not None:
            try:
                listener(Status(clock.now(), activity, _ascii(text), cite))
            except Exception:  # noqa: BLE001, S110 - a listener's fault never stops the instrument
                pass

    # --- abort and park --------------------------------------------------------------

    def abort(self, reason: str) -> None:
        """Today's abort: signal every stop flag, then hand the out-of-band retract to the
        fail-safe thread (direct COM/TCP when none runs). Sends and returns."""
        from ..service.host.safe_park import _signal_stop, existing_watchdog
        self._abort_reason = f"abort: {_ascii(reason)}"
        self._run = None
        try:
            _signal_stop(self._hub)
        except Exception:  # noqa: BLE001, S110 - the retract below still goes
            pass
        try:
            watchdog = existing_watchdog(self._hub)
            if watchdog is None:
                raise RuntimeError("no fail-safe watchdog is running")
            watchdog.park_now(f"abort: {reason}")
        except Exception:  # noqa: BLE001 - an abort is never lost: fire it directly
            try:
                self._arc().abort()
            except Exception:  # noqa: BLE001, S110 - no bridge: nothing left to send
                pass

    def park(self, command: ParkTip) -> Result:
        """Today's safe_park through the fail-safe thread: the out-of-band trigger, then the
        retract verified from the independent sensors. Waits for the outcome, boundedly."""
        from ..service.host.safe_park import _signal_stop, _watchdog_wait_bound, watchdog_for
        self._abort_reason = f"park: {_ascii(command.reason)}"
        self._run = None
        self._say("parking", f"parking: {command.reason}", command.cite)
        _signal_stop(self._hub)
        request = watchdog_for(self._hub).request(command.reason, report=True)
        request.wait(_watchdog_wait_bound())
        outcome = request.result
        if outcome is None:
            outcome = {"verified": False, "critical": True,
                       "detail": "the fail-safe thread did not report in time"}
        verified = bool(outcome.get("verified"))
        if verified:
            self._hub.host.engaged = False
        if outcome.get("not_connected"):
            # Before Igor dials in nothing is sent: not a failed park (2026-10-28).
            self._say("parking", "not connected to Igor: nothing done", command.cite)
            return Result("park", "done", _ascii(outcome["detail"]), verified=False,
                          critical=False)
        self._say("parked" if verified else "parking",
                  "parked" if verified else "park NOT verified, operator must intervene",
                  command.cite)
        return Result("park", "done", _ascii(outcome.get("detail") or ""), verified=verified,
                      critical=bool(outcome.get("critical")) or not verified)

    # --- single verbs ----------------------------------------------------------------

    def read(self, command: ReadChannels) -> Result:
        """One value per channel by its alias, read now; any failure refuses the whole read."""
        arc = self._arc()
        readings = []
        for name in command.channels:
            try:
                value = float(arc.td_ReadValue(name))
            except Exception as exc:  # noqa: BLE001
                return self._refused("read", f"could not read {name}: {exc}")
            if not math.isfinite(value):
                return self._refused("read", f"{name} read back {value}, not a number")
            readings.append(Reading(name, value))
        return Result("read", "done", readings=tuple(readings))

    def move(self, command: MoveTo) -> Result:
        """xy_transit to the point: a verified withdraw first if the tip is down."""
        arc = self._arc()
        self._begin()
        self._say("moving", f"moving to ({command.x_um:.3f}, {command.y_um:.3f}) um", command.cite)
        self._refresh_xy_cal("move")
        cal = self._hub.host.hw_cal
        try:
            xy_transit(arc, cal, command.x_um, command.y_um,
                       engaged=bool(self._hub.host.engaged), reapproach=False,
                       x_pgain=_gain(cal, "x_pgain"), x_igain=_gain(cal, "x_igain"),
                       y_pgain=_gain(cal, "y_pgain"), y_igain=_gain(cal, "y_igain"),
                       move_speed_um_s=command.speed_um_s)
        except Exception as exc:  # noqa: BLE001
            return self._failed("move", exc)
        self._hub.host.engaged = False
        return Result("move", "done")

    def engage(self, command: EngageAt) -> Result:
        if command.setpoint.reference != "absolute":
            return self._refused("engage", "the Asylum plugin engages at an absolute setpoint; "
                                 "resolve a relative one against free air first")
        arc = self._arc()
        self._begin()
        self.failure = None
        self._snapshot = None
        owns = self._service_owns_approach
        mode = self.engage_mode or command.mode
        if not owns:      # the service has done these three, and the engaged record, itself
            try:
                self._reconcile_mode(command.mode)
            except Exception as exc:  # noqa: BLE001
                return self._refused("engage", str(exc))
        hub, (pgain, igain, settle_s) = self._hub, self._engage
        self._say("engaging", f"engaging at {command.setpoint.volts:.3f} V", command.cite)
        if not owns:
            hub.host.z_loop = {"setpoint": command.setpoint.volts, "pgain": pgain, "igain": igain,
                          "mode": mode}
        try:
            snap = hw_engage(arc, setpoint=command.setpoint.volts, pgain=pgain, igain=igain,
                             settle_s=settle_s, mode=mode)
        except Exception as exc:  # noqa: BLE001
            self.failure = exc
            if owns:
                return self._failed("engage", exc)
            # The loop may be armed with the tip coming down: park, as a failed scripted engage does.
            parked = self.park(ParkTip(f"engage failed: {exc}", command.cite))
            return Result("engage", "aborted", _ascii(
                f"{type(exc).__name__}: {exc}; parked, verified={parked.verified}, "
                f"critical={parked.critical}"))
        self._snapshot = snap
        if not owns:
            hub.host.engaged = True
            hub.host._scan_epoch += 1
            hub.host._safe_park["active"] = False
        self._say("engaged", "engaged" if snap.get("reached_setpoint")
                  else "engaged, still settling to the setpoint", command.cite)
        return Result("engage", "done")

    def withdraw(self, command: WithdrawTip) -> Result:
        from ..service.host.safe_park import (
            NOT_CONNECTED, ROUTE_COM_AFTER_RESTART, STOP_RUN, _signal_stop,
            before_igor_dials_in, stop_route)
        if before_igor_dials_in(self._hub):
            # Before Igor dials in nothing is sent and nothing is claimed (2026-10-28).
            self.failure = None
            self._snapshot = None
            self._say("withdrawing", "not connected to Igor: nothing done", command.cite)
            return Result("withdraw", "done", _ascii(NOT_CONNECTED), verified=False)
        if stop_route(self._hub) == ROUTE_COM_AFTER_RESTART:
            # Restarted after a death, no new bridge yet: COM reaches Igor only through
            # its abort, so the withdraw is the park.
            parked = self.park(ParkTip("withdraw (restarted, no bridge yet)", command.cite))
            return Result("withdraw", "done", parked.reason, verified=parked.verified)
        arc = self._arc()
        self._begin()
        self.failure = None
        self._snapshot = None
        owns = self._service_owns_approach
        self._say("withdrawing", "withdrawing the tip", command.cite)
        if not owns:
            # Stop a pass in flight first, so its next object does not self-engage (as the cockpit's withdraw).
            _signal_stop(self._hub, STOP_RUN)
        try:
            snap = hw_withdraw(arc)
        except Exception as exc:  # noqa: BLE001
            self.failure = exc
            return self._failed("withdraw", exc, verified=False)
        self._snapshot = snap
        if not owns:
            self._hub.host.engaged = False
        verified = _withdraw_verified(snap)
        self._say("idle", "withdrawn" if verified else "withdraw NOT verified", command.cite)
        return Result("withdraw", "done", "" if verified else "the Z drive was not confirmed "
                      "at 0 V", verified=verified)

    # --- write -----------------------------------------------------------------------

    def write(self, command: WriteObject) -> Result:
        """One object by today's write_object cycle. On success the caller's `withdraw` is the
        end-of-run one. On an abort or a failure the plugin withdraws before it returns, unless
        `service_owns_write`: then the service withdraws. A stroke refused at its start (a
        latched fault, a stop signalled while starting) is never withdrawn by the plugin;
        without `service_owns_write` the caller must withdraw after it."""
        if self._amp_gain is None:
            return self._refused("write", "a write needs the tip-amplifier gain (amp_gain)")
        unsupported = [c for c in command.record if c not in WRITE_CAPTURE]
        if unsupported:
            return self._refused("write", "a write captures " + ", ".join(WRITE_CAPTURE)
                                 + f"; not {', '.join(unsupported)}")
        hub = self._hub
        self._abort_reason = None
        self.failure = None
        seq = getattr(hub.host, "_run_abort_seq", 0)
        try:
            if getattr(hub, "fault", None):
                raise _Refused("a fault is latched; clear it before writing")
            run = self._write_run(self._arc())
            # A park or abort signalled while this start was reading must stop the write: it
            # self-engages (today's _refuse_if_aborted / _refuse_if_stopped).
            if (hub.host._hw_litho_abort.is_set() or getattr(hub, "fault", None)
                    or getattr(hub.host, "_run_abort_seq", 0) != seq):
                raise _Refused("an abort or park was signalled, or a fault latched, "
                               "while the write was starting")
        except _Refused as why:
            self._run = None
            return self._refused("write", str(why))
        found = {}
        cites = _align_to_end(_write_kinds(run, command), command.steps)
        run.on_outcome = self._write_outcome(command.name, cites, found)
        handle = None if self._service_owns_write else _WriteHandle(hub.host._hw_litho_abort)
        if handle is not None:
            hub.host.litho_worker = handle
        hub.host.engaged = True
        self._say("writing", f"writing object {command.name}")
        try:
            return self._write_object(run, command, found)
        finally:
            run.on_outcome = None
            if handle is not None:
                handle.finish()
                if getattr(hub.host, "litho_worker", None) is handle:
                    hub.host.litho_worker = None

    def _write_object(self, run: LithographyRun, command: WriteObject, found: dict) -> Result:
        """write_object, and what it came back with. Runs before the park is let through
        (see _WriteHandle), so a park's Z commands follow this method's withdraw."""
        setpoint = WriteSetpoint(command.setpoint.volts,
                                 relative=command.setpoint.reference == "relative")
        try:
            result = run.write_object(command.name, command.x_um, command.y_um, command.tip_v,
                                      setpoint=setpoint, speed=command.speed_um_s,
                                      source=command.source)
        except (FreeAirRefused, FreeAirReadFailed) as exc:
            self.failure = exc
            return self._write_stopped("refused", str(exc))
        except WriteAborted as exc:
            return self._write_stopped("aborted", str(exc))
        except Exception as exc:  # noqa: BLE001
            self.failure = exc
            return self._write_stopped("aborted", f"{type(exc).__name__}: {exc}")
        if result.aborted or self._hub.host._hw_litho_abort.is_set():
            return self._write_stopped(
                "aborted", "the pass answered ABORTED" if result.aborted
                else "abort flag set after the pass", fired=result)
        readings = tuple(Reading(name, found[key]) for key, name in
                         (("free_air_v", channels.DEFLECTION), ("z_drive_v", channels.HEIGHT))
                         if found.get(key) is not None)
        recorded, missing = _recorded_write(result.capture, command.record)
        return Result("write", "done", f"not captured: {', '.join(missing)}" if missing else "",
                      readings=readings, recorded=recorded)

    def _write_run(self, arc) -> LithographyRun:
        """The run in progress, or a new one: hover between strokes of one object needs the
        same LithographyRun, and any other verb ends the run (the next write transits)."""
        if self._run is not None:
            return self._run
        hub = self._hub
        if not self._service_owns_write:
            from ..service.host.litho_cmds import ac_mode_active
            hub.host._hw_litho_abort.clear()
            if ac_mode_active(hub):
                raise _Refused("a write needs contact mode, the instrument is in AC")
            self._refresh_xy_cal("write")
        model = getattr(hub.host, "instrument_model", None)
        run = LithographyRun(arc.link, amp_gain=self._amp_gain, z_out=arc.z_out_channel,
                             router=arc.router, com_igor=arc.com_igor,
                             free_air_abs_band_v=None if model is None
                             else float(model.free_air_abs_band_v.value),
                             abort_event=hub.host._hw_litho_abort,
                             setpoint_log=self._setpoint_log)
        run.object_setpoints = self._object_setpoints
        run.cal = dict(hub.host.hw_cal)
        run.startup = dict(hub.host.hw_cal)
        self._run = run
        return run

    def _write_outcome(self, name: str, cites: dict[str, Citation], found: dict):
        """The per-step callback of write_object, as Status; keeps the free-air read."""
        def outcome(kind: str, make) -> None:
            if self.on_step is not None:
                self.on_step(kind, make)
            body = make()
            if kind == "read":
                found.update(free_air_v=body.get("free_air_v"), z_drive_v=body.get("z_drive_v"))
            activity, text = {
                "move": ("moving", f"moved to the start of {name} ({body.get('how')})"),
                "settle": ("moving", "settled after the move"),
                "read": ("reading", body.get("note", "read free air")),
                "engage": ("engaged", f"engaged for {name} at {body.get('setpoint_v')} V"),
                "pass": ("writing", f"{'stopped' if body.get('aborted') else 'wrote'} {name}"),
                "withdraw": ("withdrawing", "back-off skipped" if body.get("skipped")
                             else f"backed off after {name}"),
            }[kind]
            self._say(activity, text, cites.get(kind))
        return outcome

    def _write_stopped(self, outcome: str, reason: str, *, fired=None) -> Result:
        """The write did not finish: withdraw as today's end-of-run withdraw does (unless the
        service does that itself), then build the Result, so a fault building it cannot skip
        the withdraw."""
        self._run = None
        if not self._service_owns_write:
            self._make_safe(reason)
        if outcome == "refused":
            result = Result("write", "refused", _ascii(reason))
            self._say("idle", f"refused: {result.reason}")
            return result
        recorded, landed = (), None
        if fired is not None:                       # the pass was fired: keep what landed
            recorded, _ = _recorded_write(fired.capture, WRITE_CAPTURE)
            landed = _fraction_landed(fired)
        return Result("write", "aborted", _ascii(self._abort_reason or reason),
                      recorded=recorded, landed=landed)

    # --- scan ------------------------------------------------------------------------

    def scan(self, command: ScanRaster) -> Result:
        """Today's chained scan: mode reconcile, XY calibration, transit (re-approach if the
        tip was down), optional engage, the raster."""
        mode, frame, hub = command.mode, command.frame, self._hub
        capture = scan_capture_channels(mode)
        unsupported = [c for c in command.record if c not in capture]
        if unsupported:
            return self._refused("scan", f"a {mode} scan captures {', '.join(capture)}; "
                                 f"not {', '.join(unsupported)}")
        if command.setpoint.reference != "absolute":
            return self._refused("scan", "the Asylum plugin scans at an absolute setpoint; "
                                 "resolve a relative one against free air first")
        arc = self._arc()
        self._begin()
        self.failure = None
        self._scan_summary = None
        pgain, igain = self.scan_gains or (None, None)
        config = _RasterConfig(
            size_um=frame.size_um, pixels=frame.pixels, lines=frame.lines,
            line_rate_hz=frame.line_rate_hz, x_offset_um=frame.offset_um[0],
            y_offset_um=frame.offset_um[1], mode=mode, setpoint=command.setpoint.volts,
            angle_deg=frame.angle_deg, direction=frame.direction,
            trace_retrace=frame.trace_and_retrace, engage=command.engage,
            pgain=pgain, igain=igain)
        if not self._service_owns_scan:
            try:
                self._reconcile_mode(mode)
            except Exception as exc:  # noqa: BLE001
                return self._refused("scan", str(exc))
            if command.engage:
                hub.host.engaged = True
        engaged = bool(hub.host.engaged if self.scan_engaged is None else self.scan_engaged)
        kinds = ["move"] + (["engage"] if engaged or command.engage else []) + ["pass"]
        notes = _ScanNotes(self._say, _align_to_end(kinds, command.steps), self.scan_notes)
        if not self._service_owns_scan:
            hub.host._hw_abort.clear()
            self._refresh_xy_cal("scan")

        async def sink(message: dict) -> None:
            if self.scan_message is not None:
                self.scan_message(message)

        self._say("scanning", f"scanning {frame.lines} lines")
        igor_stops = getattr(hub.host, "_igor_abort_seq", 0)
        try:
            summary = asyncio.run(run_hw_scan_chained(
                arc, config, hub.host.hw_cal, sink, should_abort=hub.host._hw_abort.is_set,
                engaged=engaged, notes=notes, cancelled=self.scan_cancelled,
                igor_aborted=lambda: getattr(hub.host, "_igor_abort_seq", 0) != igor_stops))
        except Exception as exc:  # noqa: BLE001
            self.failure = exc
            if not self._service_owns_scan:
                self._make_safe(f"{type(exc).__name__}: {exc}")
            return self._failed("scan", exc)
        self._scan_summary = summary
        if summary.get("error"):
            reason = f"chained scan error: {summary['error']}"
            if not self._service_owns_scan:
                self._make_safe(reason)
            return Result("scan", "aborted", _ascii(reason))
        if summary.get("aborted"):
            return Result("scan", "aborted", _ascii(self._abort_reason or "the scan was aborted"))
        self._say("engaged" if hub.host.engaged else "idle", "scan finished")
        return Result("scan", "done", recorded=_recorded_scan(summary, config, command.record))

    # --- shared helpers --------------------------------------------------------------

    def _reconcile_mode(self, mode: str) -> None:
        """Keep the instrument software's imaging mode the same as the pass's (SETMODE only
        while the tip is up); raises if it cannot be done safely."""
        from ..service.host.scan_cmds import ensure_software_mode
        asyncio.run(ensure_software_mode(_ModeHub(self._hub, self._say), mode))

    def _refresh_xy_cal(self, source: str) -> None:
        from ..service.xy_cal import refresh_xy_cal
        refresh_xy_cal(self._hub, source)

    def _make_safe(self, reason: str) -> None:
        """Withdraw the tip as the service's end-of-run withdraw does; if that fails, hand the
        retract to the fail-safe thread and latch a fault (the worker's escalation)."""
        hub = self._hub
        from ..service.host.safe_park import before_igor_dials_in
        if before_igor_dials_in(hub):
            self._say("parking", "not connected to Igor: nothing done")
            return
        try:
            hw_withdraw(self._arc())
            hub.host.engaged = False
        except Exception as exc:  # noqa: BLE001
            from ..service.host.safe_park import PARK_WATCHDOG, request_park
            message = f"withdraw after '{reason}' failed: {exc}"
            self._say("parking", f"{message}; handing the retract to the fail-safe thread")
            try:
                request_park(hub, message, PARK_WATCHDOG)
            except Exception:  # noqa: BLE001, S110
                pass
            if getattr(hub, "fault", None) is None:
                hub.fault = {"reason": "withdraw_failed", "at": clock.wall(), "state": "fault",
                             "detail": message}

    def _begin(self) -> None:
        """A verb other than a write ends the write run (the next write transits) and clears
        the last abort's reason."""
        self._run = None
        self._abort_reason = None

    def _refused(self, verb: Verb, reason: str) -> Result:
        self._say("idle", f"refused: {reason}")
        return Result(verb, "refused", _ascii(reason))

    def _failed(self, verb: Verb, exc: Exception, **flags) -> Result:
        self._say("idle", f"{verb} failed: {exc}")
        return Result(verb, "aborted", _ascii(f"{type(exc).__name__}: {exc}"), **flags)


class _Refused(Exception):
    """A write that must not start."""


def _gain(cal: dict, key: str) -> float:
    """An XY loop gain from the STARTUP calibration, 0 when absent or not finite."""
    value = _number(cal.get(key))
    return value if math.isfinite(value) else 0.0


def _withdraw_verified(snap: dict) -> bool:
    """The service's withdraw verify: no read timed out and the Z drive is within the park
    tolerance of 0 V."""
    from ..service.host.safe_park import _WITHDRAWN_HEIGHT_TOL_V
    z = snap.get("z_v")
    return not snap.get("verify_timeout") and z is not None and abs(z) <= _WITHDRAWN_HEIGHT_TOL_V


def _write_kinds(run: LithographyRun, command: WriteObject) -> list[str]:
    """The sub-steps this write_object call will report, in order: a hover has no settle and
    no free-air read; the run's first object settles."""
    from ..hardware.lithography_run import is_hover_move
    first = run._object_setpoint is None
    hover = is_hover_move(first, command.source, run._last_source)
    return (["move"] + (["settle"] if first and run.first_object_settle_s else [])
            + ([] if hover else ["read"]) + ["engage", "pass", "withdraw"])


def _fraction_landed(result) -> float:
    """How much of the pass the controller captured before it stopped: the longest captured
    channel against the padded capture length (the estimate the cockpit's partial trace uses)."""
    lengths = [len(values) for values in (result.capture or {}).values()]
    if not lengths or not result.cap_len:
        return 0.0
    return min(1.0, max(lengths) / result.cap_len)


def _recorded_write(capture: dict, record: tuple[str, ...]):
    """The requested channels from a write's capture, volts as streamed (padded to the
    capture length); and the requested channels the stream did not carry."""
    recorded, missing = [], []
    for name in record:
        values = np.asarray(capture.get(name, ()), dtype=float)
        if values.size:
            recorded.append(Recorded(name, "write", values))
        else:
            missing.append(name)
    return tuple(recorded), missing


def _recorded_scan(summary: dict, config: _RasterConfig, record: tuple[str, ...]):
    """Each requested channel as lines x pixels for trace and, with retrace, retrace
    (flipped to line up with the trace, as the cockpit's line stream does)."""
    pixels, per_line = config.pixels, config.pixels * (2 if config.trace_retrace else 1)
    rows = summary.get("line_indices") or []
    recorded = []
    for name in record:
        flat = np.asarray(summary["capture"].get(name, ()), dtype=float)
        lines = flat[: len(rows) * per_line].reshape(-1, per_line)
        if not lines.size:
            continue
        if config.trace_retrace:
            recorded.append(Recorded(name, "trace", lines[:, :pixels]))
            recorded.append(Recorded(name, "retrace", lines[:, pixels:][:, ::-1]))
        else:
            snaked = np.array([line[::-1] if row % 2 else line for row, line in zip(rows, lines)])
            recorded.append(Recorded(name, "trace", snaked))
    return tuple(recorded)


def open_asylum(hub, choice, **options) -> AsylumInstrument:
    """The registry factory for the Asylum plugin: `PluginChoice("asylum")`."""
    return AsylumInstrument(hub, **options)
