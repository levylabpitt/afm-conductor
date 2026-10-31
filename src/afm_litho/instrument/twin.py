"""The twin plugin: a small simulated instrument behind the `Instrument` interface.

Calculated, not simulated. The tip's place and speed at any time come from the commanded
points (straight segments at the commanded speed, moves at the move speed); Z drive,
deflection and amplitude are straight ramps over the fixed durations below. There is no
physics and no stepping integrator. It knows nothing of Igor, td_ calls or the wire, and
imports nothing from twin/ or service/, and from hardware/ only the channel table.

Time: a command lays its movement out as keyframes on afm_litho.clock (`_Track`) and waits
for its end with clock.sleep. Telemetry runs on its own thread on REAL time (clock rule: a
background loop stays real) and stamps each frame with clock.now().

The sample is a height image from a scan `.npz` (the file service/scan_cache.py writes):
`height` (rows, pixels) in nm, optionally `retrace` (the same, aligned with the trace;
absent means the retrace is a copy of the trace), `size_um` (the image width), `x_offset_um` and `y_offset_um`
(its centre), `angle_deg` (its rotation about the centre, the vendor's: `litho/frame.py`
`rotate`). Pixels are square. Row r is slow
scan line r (host/scan_pump.py stores each line at its scan index), so for a direction "up"
scan row 0 is the lowest Y. Unscanned pixels are NaN. Written wires are drawn into an
overlay at the image resolution and added to the height, so they stay in later scans.
"""

from __future__ import annotations

import math
import threading
from bisect import bisect_right
from pathlib import Path
from typing import Callable

import numpy as np

from .. import clock
from ..hardware.channels import AMPLITUDE, DEFLECTION, DETECTOR_SUM, HEIGHT, Z_SENSOR
from ..litho.frame import rotate, unrotate
from ..run.model import Mode, Platform, Setpoint
from .commands import (Citation, EngageAt, MoveTo, ParkTip, ReadChannels, Reading, Recorded,
                       Result, ScanRaster, Verb, WithdrawTip, WriteObject)
from .reports import (Activity, Status, StatusListener, TelemetryFrame, TelemetryListener,
                      telemetry_hz)

# Fixed, stated durations (seconds at speed factor 1). Withdraw matches the hardware bound.
WITHDRAW_S = 2.5
ENGAGE_S = 2.0
DEFLECTION_SETTLE_S = 0.5      # after the Z drive stops rising, deflection goes to the setpoint

WITHDRAWN_Z_V = 0.0
ENGAGED_Z_V = 75.0
FREE_AIR_DEFLECTION_V = -0.2
SUM_V = 10.0
AC_FREE_AMPLITUDE_V = 1.0      # amplitude in free air in AC mode; 0 in contact mode
WIRE_WIDTH_NM = 100.0
WIRE_HEIGHT_NM = 2.0           # a written wire adds this to the height image
HEIGHT_V_PER_NM = 0.001        # the twin's Height channel calibration
TICK_S = 0.05                  # how often a waiting command looks for an abort

NO_TRANSIT_SPEED = "the twin has no speed of its own: the command must give transit_speed_um_s"
STOP_BEFORE_START = "an abort or park was signalled while the command was starting"

SUPPORTED = (HEIGHT, DEFLECTION, DETECTOR_SUM, AMPLITUDE, Z_SENSOR)


class _Aborted(Exception):
    """An abort or park cut the command short; `began` is when the cut step was laid."""

    def __init__(self, began: float | None = None):
        super().__init__()
        self.began = began


class _Track:
    """One value against time: keyframes joined by straight lines, constant past the ends."""

    def __init__(self, value: float):
        self.t = [0.0]
        self.v = [value]

    def value(self, t: float) -> float:
        i = bisect_right(self.t, t)
        if i == 0:
            return self.v[0]
        if i == len(self.t):
            return self.v[-1]
        span = self.t[i] - self.t[i - 1]
        return self.v[i - 1] + (self.v[i] - self.v[i - 1]) * (t - self.t[i - 1]) / span

    def slope(self, t: float) -> float:
        i = bisect_right(self.t, t)
        if i == 0 or i == len(self.t):
            return 0.0
        return (self.v[i] - self.v[i - 1]) / (self.t[i] - self.t[i - 1])

    def restart(self, t: float) -> None:
        """Forget everything planned from `t` on; hold the value it has at `t`."""
        here = self.value(t)
        keep = bisect_right(self.t, t - 1e-12)
        del self.t[keep:], self.v[keep:]
        self.t.append(t)
        self.v.append(here)

    def to(self, t_end: float, value: float) -> None:
        self.t.append(max(t_end, self.t[-1]))
        self.v.append(value)


class _Sample:
    """The height image and the overlay of written wires on it."""

    def __init__(self, height, size_um: float, x_um: float, y_um: float, angle_deg: float = 0.0,
                 retrace=None, zdrive_v=None, amp_v=None):
        self.zdrive_v = None if zdrive_v is None else np.asarray(zdrive_v, dtype=float)
        self.amp_v = None if amp_v is None else np.asarray(amp_v, dtype=float)
        self.height = np.asarray(height, dtype=float)
        self.retrace = None if retrace is None else np.asarray(retrace, dtype=float)
        self.rows, self.pixels = self.height.shape
        self.pitch_um = size_um / self.pixels
        self.centre = (x_um, y_um)
        self.angle_deg = angle_deg
        self.overlay = np.zeros(self.height.shape, dtype=bool)
        self._reach = max(0, round(WIRE_WIDTH_NM / 2 / (self.pitch_um * 1000)))

    @classmethod
    def load(cls, path) -> _Sample:
        with np.load(Path(path)) as z:
            retrace = z["retrace"] if "retrace" in z.files else None
            return cls(z["height"], float(z["size_um"]), float(z["x_offset_um"]),
                       float(z["y_offset_um"]), float(z["angle_deg"]), retrace,
                       z["zdrive_v"] if "zdrive_v" in z.files else None,
                       z["amp_v"] if "amp_v" in z.files else None)

    def _index(self, x, y):
        """Pixel row, column and whether it is on the image, for lab positions (um)."""
        dx, dy = np.asarray(x) - self.centre[0], np.asarray(y) - self.centre[1]
        u, v = unrotate(dx, dy, self.angle_deg)                         # into the image frame
        col = np.floor(u / self.pitch_um + self.pixels / 2).astype(int)
        row = np.floor(v / self.pitch_um + self.rows / 2).astype(int)
        return row, col, (col >= 0) & (col < self.pixels) & (row >= 0) & (row < self.rows)

    def recorded_v(self, grid, x: float, y: float) -> float | None:
        """The recorded volts of `grid` at the pixel under (x, y) um; None off the image."""
        if grid is None:
            return None
        row, col, inside = self._index(x, y)
        return float(grid[row, col]) if inside else None

    def recorded_line(self, grid, x, y) -> np.ndarray | None:
        """The recorded volts of `grid` under each (x, y) um; 0 off the image; None if no grid."""
        if grid is None:
            return None
        row, col, inside = self._index(x, y)
        row, col = np.where(inside, row, 0), np.where(inside, col, 0)
        return np.where(inside, grid[row, col], 0.0)

    def height_nm(self, x, y, retrace: bool = False) -> np.ndarray:
        """Image plus wires under each (x, y) in um; 0 off the image, NaN where unscanned.
        `retrace` reads the seed's own retrace image when it has one (else the trace)."""
        row, col, inside = self._index(x, y)
        row, col = np.where(inside, row, 0), np.where(inside, col, 0)
        image = self.retrace if retrace and self.retrace is not None else self.height
        total = image[row, col] + self.overlay[row, col] * WIRE_HEIGHT_NM
        return np.where(inside, total, 0.0)

    def draw(self, x: np.ndarray, y: np.ndarray, tip_v: np.ndarray) -> None:
        """Each segment takes its starting point's bias: positive writes, negative erases."""
        for i in range(len(x) - 1):
            if tip_v[i] == 0:
                continue
            length = math.hypot(x[i + 1] - x[i], y[i + 1] - y[i])
            n = max(2, math.ceil(length / (self.pitch_um / 2)) + 1)
            row, col, inside = self._index(np.linspace(x[i], x[i + 1], n),
                                           np.linspace(y[i], y[i + 1], n))
            for dr in range(-self._reach, self._reach + 1):
                for dc in range(-self._reach, self._reach + 1):
                    r, c = row[inside] + dr, col[inside] + dc
                    ok = (r >= 0) & (r < self.rows) & (c >= 0) & (c < self.pixels)
                    self.overlay[r[ok], c[ok]] = tip_v[i] > 0


class TwinInstrument:
    """The simulated instrument. `speed_factor` 1 is real time, 2 twice as fast, 0 instant.

    `sample` is a scan `.npz` path (a flat 10 um sample if none). The tip's calculated speed
    (`speed_at`) is what the twin reports for the cockpit's tip display.

    The twin has no speed of its own: a write's or scan's move to its first point goes at the
    command's `transit_speed_um_s`, and a command without one is refused.

    Hooks for a service that owns the job (carry existing data only, like the Asylum plugin's):
    `should_stop(verb)` says an abort or park was signalled before the command began, so an
    abort that lands between two commands is not lost; `on_step(kind, make)` gets each write
    step's outcome (move, read, engage, pass, withdraw); `on_line(index, trace_nm, retrace_nm)`
    gets each scan line as its time passes (retrace_nm is None without retrace); `failure` is
    always None (the twin raises nothing) and `object_setpoints` stays empty.
    """

    def __init__(self, platform: Platform = "cypher", sample: str | Path | None = None,
                 speed_factor: float = 1.0,
                 should_stop: Callable[[Verb], bool] | None = None):
        if speed_factor < 0:
            raise ValueError("speed_factor must be 0 (instant) or more")
        self.platform = platform
        self.speed_factor = speed_factor
        self.sample = _Sample.load(sample) if sample else _Sample(np.zeros((128, 128)), 10, 0, 0)
        self._lock = threading.RLock()
        self._x, self._y = _Track(0.0), _Track(0.0)
        self._z, self._defl = _Track(WITHDRAWN_Z_V), _Track(FREE_AIR_DEFLECTION_V)
        self._amp = _Track(0.0)
        self._mode: Mode = "contact"
        self._activity: Activity = "idle"
        self._chained = False
        self._last_source: str | None = None
        self._abort_reason: str | None = None
        self._halted_at = 0.0
        self._on_telemetry: TelemetryListener | None = None
        self._on_status: StatusListener | None = None
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._should_stop = should_stop
        self.on_step: Callable[..., None] | None = None
        self.on_line: Callable[..., None] | None = None
        self.failure: Exception | None = None
        self.object_setpoints: list[dict] = []

    # --- calculated state ------------------------------------------------------------

    def position_at(self, t: float) -> tuple[float, float]:
        return self._x.value(t), self._y.value(t)

    def speed_at(self, t: float) -> float:
        """Tip speed in um/s at time t."""
        return math.hypot(self._x.slope(t), self._y.slope(t))

    def frame_at(self, t: float) -> TelemetryFrame:
        with self._lock:
            x, y = self.position_at(t)
            z_v, amp_v = self._z.value(t), self._amp.value(t)
            if z_v == ENGAGED_Z_V:     # engaged: report what the instrument recorded here
                z_rec = self.sample.recorded_v(self.sample.zdrive_v, x, y)
                amp_rec = self.sample.recorded_v(self.sample.amp_v, x, y)
                z_v = z_rec if z_rec is not None and math.isfinite(z_rec) else z_v
                if self._mode == "ac" and amp_rec is not None and math.isfinite(amp_rec):
                    amp_v = amp_rec
            return TelemetryFrame(t, x, y, z_v, self._defl.value(t), SUM_V, amp_v)

    def telemetry_period_s(self) -> float:
        return 1.0 / telemetry_hz(self._activity, self._chained)

    def _withdrawn(self) -> bool:
        return math.isclose(self._z.value(clock.now()), WITHDRAWN_Z_V, abs_tol=1e-9)

    def withdrawn(self) -> bool:
        """The tip is off the surface and no Z move is still under way."""
        with self._lock:
            return self._withdrawn() and self._z.t[-1] <= clock.now()

    def _seconds(self, nominal: float) -> float:
        return nominal / self.speed_factor if self.speed_factor else 0.0

    # --- interface: listening, abort, close -------------------------------------------

    def listen(self, on_telemetry: TelemetryListener, on_status: StatusListener) -> None:
        self._on_telemetry, self._on_status = on_telemetry, on_status
        if self._thread is None:
            self._stopping.clear()
            self._thread = threading.Thread(target=self._telemetry_loop, daemon=True,
                                            name="twin-telemetry")
            self._thread.start()

    def _telemetry_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                self._on_telemetry(self.frame_at(clock.now()))
            except Exception:    # noqa: BLE001, S110 - a listener's fault never stops the twin
                pass
            self._stopping.wait(self.telemetry_period_s())    # real time, by clock rule (a)

    def close(self) -> None:
        self._stopping.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._thread = None

    def abort(self, reason: str) -> None:
        self._halt(f"abort: {reason}", "withdrawing", "aborted, withdrawing the tip")

    def _halt(self, reason: str, activity: Activity, text: str) -> float:
        """Stop where it is and withdraw; returns when the Z drive reaches the withdrawn value.

        Once set, the reason stops every later step from being laid (`_step`), so nothing
        raises the tip again; the next command clears it.
        """
        with self._lock:
            self._abort_reason = reason
            self._halted_at = clock.now()
            end = self._lay_withdraw(self._halted_at, freeze_xy=True)
        self._say(activity, text)
        return end

    def _say(self, activity: Activity, text: str, cite: Citation | None = None) -> None:
        self._activity = activity
        if self._on_status is not None:
            self._on_status(Status(clock.now(), activity, text, cite))

    # --- laying out tracks (each returns the time its movement ends) -------------------

    def _lay_withdraw(self, t0: float, freeze_xy: bool = False) -> float:
        if freeze_xy:
            self._x.restart(t0)
            self._y.restart(t0)
        if self._z.v[-1] == WITHDRAWN_Z_V:                  # already withdrawn or withdrawing
            return max(t0, self._z.t[-1])
        end = t0 + self._seconds(WITHDRAW_S)
        for track in (self._z, self._defl, self._amp):
            track.restart(t0)
        self._z.to(end, WITHDRAWN_Z_V)
        self._defl.to(end, FREE_AIR_DEFLECTION_V)
        self._amp.to(end, AC_FREE_AMPLITUDE_V if self._mode == "ac" else 0.0)
        return end

    def _lay_engage(self, t0: float, mode: Mode, setpoint: Setpoint) -> float:
        """Z rises over ENGAGE_S, then deflection (contact) or amplitude (AC) takes the setpoint."""
        self._mode = mode
        rising = self._z.value(t0) != ENGAGED_Z_V
        z_end = t0 + (self._seconds(ENGAGE_S) if rising else 0.0)
        end = z_end + self._seconds(DEFLECTION_SETTLE_S)
        for track in (self._z, self._defl, self._amp):
            track.restart(t0)
        self._z.to(z_end, ENGAGED_Z_V)
        self._defl.to(z_end, self._defl.value(t0))
        free_amp = AC_FREE_AMPLITUDE_V if mode == "ac" else 0.0
        self._amp.to(t0, free_amp)
        self._amp.to(z_end, free_amp)
        if mode == "ac":
            self._amp.to(end, _target(setpoint, AC_FREE_AMPLITUDE_V))
            self._defl.to(end, FREE_AIR_DEFLECTION_V)
        else:
            self._defl.to(end, _target(setpoint, FREE_AIR_DEFLECTION_V))
        return end

    def _lay_path(self, t0: float, times, x, y) -> float:
        """Reach point i at t0 + times[i]; the first point is where the tip is now."""
        self._x.restart(t0)
        self._y.restart(t0)
        for k in range(len(x)):
            self._x.to(t0 + float(times[k]), float(x[k]))
            self._y.to(t0 + float(times[k]), float(y[k]))
        return t0 + float(times[-1])

    def _path_times(self, x, y, speed: float) -> np.ndarray:
        length = np.hypot(np.diff(x), np.diff(y))
        return np.concatenate([[0.0], np.cumsum(length)]) / speed * self._seconds(1.0)

    def _lay_move(self, t0: float, x: float, y: float, speed: float) -> float:
        px, py = np.array([self._x.value(t0), x]), np.array([self._y.value(t0), y])
        return self._lay_path(t0, self._path_times(px, py, speed), px, py)

    def _lay_raster(self, t0: float, gx, gy, line_s: float, retrace: bool) -> float:
        """Each line goes along (gx[k], gy[k]) and, with retrace, back in the second half of line_s."""
        self._x.restart(t0)
        self._y.restart(t0)
        for k in range(len(gx)):
            t = t0 + k * line_s
            sweep = t + (line_s / 2 if retrace else line_s)
            for track, line in ((self._x, gx[k]), (self._y, gy[k])):
                track.to(t, float(line[0]))
                track.to(sweep, float(line[-1]))
                if retrace:
                    track.to(t + line_s, float(line[0]))
        return t0 + len(gx) * line_s

    # --- running a command -----------------------------------------------------------

    def _begin(self) -> None:
        with self._lock:
            self._abort_reason = None
            self._chained = False

    def _stop_signalled(self, verb: Verb) -> bool:
        return bool(self._should_stop is not None and self._should_stop(verb))

    def _note(self, kind: str, **body) -> None:
        if self.on_step is not None:
            try:
                self.on_step(kind, lambda: body)
            except Exception:    # noqa: BLE001, S110 - a note never stops a write
                pass

    def _step(self, activity: Activity, text: str, cite: Citation | None, lay,
              tick: Callable[[float], None] | None = None) -> float:
        """Say what is starting, lay it out, wait for its end; returns the time it began.

        Raises `_Aborted` instead of laying anything once an abort or park has been called.
        `tick(now)` is called on every look at the clock, last when the step ends or is cut.
        """
        self._say(activity, text, cite)
        with self._lock:
            if self._abort_reason is not None:
                raise _Aborted
            began = clock.now()
            end = lay(began)
        while self._abort_reason is None:
            left = end - clock.now()
            if tick is not None:
                tick(min(clock.now(), end))
            if left <= 1e-9:
                return began
            clock.sleep(min(TICK_S, left))
        if tick is not None:
            tick(min(self._halted_at, end))
        raise _Aborted(began)

    def _aborted(self, verb: Verb, **flags) -> Result:
        """Wait for the tip to be withdrawn, then report; a park reports "parked" itself."""
        reason = self._abort_reason or "aborted"
        wait = self._z.t[-1] - clock.now()
        if wait > 0:
            clock.sleep(wait)
        if not reason.startswith("park"):
            self._say("idle", "stopped, tip withdrawn")
        if "verified" in flags:
            flags["verified"] = self._withdrawn()
        return Result(verb, "aborted", reason, **flags)

    def _withdraw_first(self) -> None:
        """Never move XY with the Z drive above withdrawn (or still on its way down)."""
        if not self._withdrawn() or self._z.t[-1] > clock.now():
            self._step("withdrawing", "withdrawing the tip", None, self._lay_withdraw)

    def _go_to(self, x: float, y: float, speed: float, cite=None, text="") -> None:
        self._withdraw_first()
        self._step("moving", text or f"moving to ({x:.3f}, {y:.3f}) um", cite,
                   lambda t0: self._lay_move(t0, x, y, speed))

    # --- interface: commands ----------------------------------------------------------

    def move(self, command: MoveTo) -> Result:
        self._begin()
        try:
            self._go_to(command.x_um, command.y_um, command.speed_um_s, command.cite)
        except _Aborted:
            return self._aborted("move")
        return Result("move", "done")

    def engage(self, command: EngageAt) -> Result:
        self._begin()
        try:
            self._step("engaging", "engaging the tip", command.cite,
                       lambda t0: self._lay_engage(t0, command.mode, command.setpoint))
        except _Aborted:
            return self._aborted("engage")
        self._say("engaged", "engaged", command.cite)
        return Result("engage", "done")

    def withdraw(self, command: WithdrawTip) -> Result:
        self._begin()
        try:
            self._step("withdrawing", "withdrawing the tip", command.cite, self._lay_withdraw)
        except _Aborted:
            return self._aborted("withdraw", verified=False)
        self._say("idle", "withdrawn", command.cite)
        return Result("withdraw", "done", verified=self._withdrawn())

    def read(self, command: ReadChannels) -> Result:
        self._begin()
        bad = [c for c in command.channels if c not in SUPPORTED]
        if bad:
            return Result("read", "refused", f"the twin cannot read {', '.join(bad)}")
        self._say("reading", "reading " + ", ".join(command.channels), command.cite)
        t = clock.now()
        with self._lock:
            x, y = (np.array([v]) for v in self.position_at(t))
            values = [float(self._value(c, x, y, t)[0]) for c in command.channels]
        if any(math.isnan(v) for v in values):
            return Result("read", "refused", "the image has no measured height at the tip")
        return Result("read", "done", readings=tuple(map(Reading, command.channels, values)))

    def park(self, command: ParkTip) -> Result:
        end = self._halt(f"park: {command.reason}", "parking", f"parking: {command.reason}")
        # A loop, not one sleep: before Python 3.13, time.monotonic on Windows ticks every
        # ~15.6 ms, so a single sleep can wake with the clock still short of `end`.
        while end > clock.now():
            clock.sleep(end - clock.now())
        self._say("parked", "parked", command.cite)
        verified = self._withdrawn()
        return Result("park", "done", verified=verified, critical=not verified)

    def write(self, command: WriteObject) -> Result:
        self._begin()
        bad = [c for c in command.record if c not in SUPPORTED]
        if bad:
            return Result("write", "refused", f"the twin cannot record {', '.join(bad)}")
        if command.transit_speed_um_s is None:
            return Result("write", "refused", NO_TRANSIT_SPEED)
        if self._mode == "ac" and not self._withdrawn():
            return Result("write", "refused", "a write needs contact mode, the tip is in AC")
        if self._stop_signalled("write"):
            return Result("write", "aborted", STOP_BEFORE_START)
        self._chained = command.chained
        x, y, steps = command.x_um, command.y_um, command.steps
        hover = command.source is not None and command.source == self._last_source
        self._last_source = command.source
        # The run's steps are Move, [Settle, Read], Engage, LithoObject, Withdraw: the last
        # three are always engage, write and withdraw; the Read is the one before them.
        move, engage, fire, back_off = (steps[0], *steps[-3:]) if len(steps) >= 4 else (None,) * 4
        read = steps[-4] if len(steps) > 4 else None
        times = self._path_times(x, y, command.speed_um_s)
        readings, recorded, landed = (), (), None
        try:
            self._go_to(float(x[0]), float(y[0]), command.transit_speed_um_s, move,
                        f"{'hovering' if hover else 'moving'} to the start of {command.name}")
            self._note("move", how="hover" if hover else "transit", name=command.name,
                       to_um=(float(x[0]), float(y[0])))
            if not hover:
                self._say("reading", "reading free air", read)
                readings = (Reading(DEFLECTION, self._defl.value(clock.now())),
                            Reading(Z_SENSOR, self._z.value(clock.now())))
                self._note("read", name=command.name, free_air_v=readings[0].volts,
                           z_drive_v=readings[1].volts)
            self._step("engaging", "engaging the tip", engage,
                       lambda t0: self._lay_engage(t0, "contact", command.setpoint))
            self._note("engage", name=command.name)
            try:
                began = self._step("writing", f"writing object {command.name}", fire,
                                   lambda t0: self._lay_path(t0, times, x, y))
            except _Aborted as cut:
                self._draw_wire(command, cut.began, times, self._halted_at)
                recorded, landed = self._prefix(command, cut.began, times)
                self._note("pass", name=command.name, aborted=True)
                raise
            self._note("pass", name=command.name, aborted=False)
            self._step("withdrawing", "withdrawing the tip", back_off, self._lay_withdraw)
            self._note("withdraw", name=command.name)
        except _Aborted:
            return self._aborted("write", recorded=recorded, landed=landed)
        recorded = tuple(Recorded(c, "write", self._value(c, x, y, began)) for c in command.record)
        self._draw_wire(command, began, times, None)
        self._chained = False
        self._say("idle", f"finished {command.name}")
        return Result("write", "done", readings=readings, recorded=recorded)

    def _prefix(self, command: WriteObject, began, times):
        """What an aborted pass had reached: the recorded values of those points and the
        fraction of the pass (None when the pass had not begun)."""
        if began is None:
            return (), None
        elapsed = self._halted_at - began
        n = int(np.searchsorted(times, elapsed, side="right"))
        fraction = min(1.0, max(0.0, elapsed / times[-1])) if times[-1] > 0 else 1.0
        if n == 0:
            return (), fraction
        x, y = command.x_um[:n], command.y_um[:n]
        return (tuple(Recorded(c, "write", self._value(c, x, y, began))
                      for c in command.record), fraction)

    def _draw_wire(self, command: WriteObject, began, times, stopped_at) -> None:
        """Draw the wire; after an abort only the points the tip reached by `stopped_at`."""
        if began is None:
            return
        n = len(times) if stopped_at is None else int(
            np.searchsorted(times, stopped_at - began, side="right"))
        self.sample.draw(command.x_um[:n], command.y_um[:n], command.tip_v[:n])

    def scan(self, command: ScanRaster) -> Result:
        self._begin()
        frame = command.frame
        bad = [c for c in command.record if c != HEIGHT]
        if bad:
            return Result("scan", "refused", f"the twin scans Height only, not {', '.join(bad)}")
        if command.transit_speed_um_s is None:
            return Result("scan", "refused", NO_TRANSIT_SPEED)
        if not command.engage and self._withdrawn():
            return Result("scan", "refused", "the tip is not engaged and the scan does not engage")
        if self._stop_signalled("scan"):
            return Result("scan", "aborted", STOP_BEFORE_START)
        gx, gy = _raster_points(frame)
        trace = self.sample.height_nm(gx, gy) * HEIGHT_V_PER_NM
        retrace = self.sample.height_nm(gx, gy, retrace=True) * HEIGHT_V_PER_NM
        # The hardware plays `pixels` points per line at pixels*line_rate_hz, and doubles the
        # points per line for trace and retrace (hw_scan), so a trace+retrace line takes 2/f.
        line_s = self._seconds((2.0 if frame.trace_and_retrace else 1.0) / frame.line_rate_hz)
        sent = [0]
        raster_began = [0.0]

        def lay_raster(t0: float) -> float:
            raster_began[0] = t0
            return self._lay_raster(t0, gx, gy, line_s, frame.trace_and_retrace)

        def lines_done(now: float) -> None:
            """Hand over each line whose time has passed (all of them when time is instant)."""
            due = frame.lines if line_s <= 0 else min(
                frame.lines, int((now - raster_began[0]) / line_s + 1e-9))
            while sent[0] < due:
                self._send_line(sent[0], trace, retrace, frame.trace_and_retrace)
                sent[0] += 1

        try:
            self._go_to(float(gx[0, 0]), float(gy[0, 0]), command.transit_speed_um_s,
                        command.steps[0] if command.steps else None)
            self._step("engaging", "engaging the tip", None,       # also re-engages after transit
                       lambda t0: self._lay_engage(t0, command.mode, command.setpoint))
            self._step("scanning", f"scanning {frame.lines} lines", None, lay_raster,
                       tick=lines_done)
        except _Aborted:
            return self._aborted("scan")
        while sent[0] < frame.lines:             # a finished scan has handed over every line
            self._send_line(sent[0], trace, retrace, frame.trace_and_retrace)
            sent[0] += 1
        self._say("engaged", "scan finished")
        # Like the hardware path, the retrace is delivered flipped back to align with the trace.
        recorded = [Recorded(HEIGHT, "trace", trace)]
        if frame.trace_and_retrace:
            recorded.append(Recorded(HEIGHT, "retrace", retrace))
        return Result("scan", "done", recorded=tuple(recorded))

    def _send_line(self, index: int, trace: np.ndarray, retrace: np.ndarray,
                   with_retrace: bool) -> None:
        if self.on_line is None:
            return
        try:
            self.on_line(index, trace[index] / HEIGHT_V_PER_NM,
                         retrace[index] / HEIGHT_V_PER_NM if with_retrace else None)
        except Exception:    # noqa: BLE001, S110 - a listener's fault never stops a scan
            pass

    # --- channel values ---------------------------------------------------------------

    def _value(self, channel: str, x, y, t: float) -> np.ndarray:
        """`channel` in volts under the points (x, y), with the Z loop as it is at time t."""
        if channel == HEIGHT:
            return self.sample.height_nm(x, y) * HEIGHT_V_PER_NM
        value = {DETECTOR_SUM: SUM_V, DEFLECTION: self._defl.value(t),
                 AMPLITUDE: self._amp.value(t), Z_SENSOR: self._z.value(t)}[channel]
        return np.full(len(np.atleast_1d(x)), float(value))


def _raster_points(frame) -> tuple[np.ndarray, np.ndarray]:
    """Lab X and Y (lines x pixels) of the frame's pixel centres, rotated by its angle about
    the frame centre; direction "down" scans the lines from the top."""
    u = (np.arange(frame.pixels) + 0.5) * frame.size_um / frame.pixels - frame.size_um / 2
    v = (np.arange(frame.lines) + 0.5) * frame.size_um / frame.lines - frame.size_um / 2
    if frame.direction == "down":
        v = v[::-1]
    uu, vv = np.meshgrid(u, v)
    x, y = rotate(uu, vv, frame.angle_deg)
    return frame.offset_um[0] + x, frame.offset_um[1] + y


raster_points = _raster_points     # the service lays its scan rows out the same way


def _target(setpoint: Setpoint, free_air: float) -> float:
    """A relative setpoint is measured from the free-air value, an absolute one is the value."""
    return setpoint.volts if setpoint.reference == "absolute" else free_air + setpoint.volts


def open_twin(choice, **options) -> TwinInstrument:
    """The registry factory for the twin: `PluginChoice("twin")` plays the Cypher."""
    return TwinInstrument(platform=choice.platform, **options)
