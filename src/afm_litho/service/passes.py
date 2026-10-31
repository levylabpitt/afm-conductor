"""Imaging passes executed against the software ARC.

Two execution paths, both honest about what they are:

- Point passes (approach, force curve, tune) run through the per-tick ARC:
  the same servo loops and wave banks the capture-replay test exercises.
- Raster lines use a vectorized line renderer through the same World/Sample
  (scanner lag + z-servo lag + parachuting emulated with filters), because a
  per-tick Python loop cannot hold 50 kHz across a full frame. The renderer
  produces height, deflection/amplitude (error), and z-sensor channels with
  the classic trace/retrace artifacts.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter

from ..hardware import channels
from ..litho.frame import rotate
from ..twin.arc import CLOCK_HZ, DT, SoftwareARC
from ..twin.physics import World
from .twin_player import WAIT_SLICE_S

# Sheet conductance of fully written material.
G_SAT_S = 2.0e-6

# Choreography timing, in ARC ticks: the controller runs at CLOCK_HZ, so a dwell
# of `int(seconds * CLOCK_HZ)` ticks lasts `seconds`. These are the settle /
# clear / relax pauses between the discrete steps of a point pass or a litho
# stroke; naming them tied to CLOCK_HZ means the one place that sets the sample
# clock also sets what they mean, instead of bare integers scattered inline.
APPROACH_POLL_TICKS = int(0.02 * CLOCK_HZ)   # 20 ms between threshold checks
WITHDRAW_TICKS = int(0.40 * CLOCK_HZ)        # ramp the tip clear, then settle
LIFT_CLEAR_TICKS = int(0.06 * CLOCK_HZ)      # let z lift before a transit moves
JUMP_SETTLE_TICKS = int(0.30 * CLOCK_HZ)     # settle after an unphysical jump
TIP_DOWN_TICKS = int(0.20 * CLOCK_HZ)        # land before arming a stroke
RELAX_TICKS = int(0.10 * CLOCK_HZ)           # bias off, let the stroke relax
STREAM_CHUNK_TICKS = 2000                    # one trace / surface sample per 40 ms
CAPTURE_DECIMATION = 50                      # capture at CLOCK_HZ/50 = 1 kHz
# Controller ticks really stepped where a write comes to rest (tip down, end of a
# stroke, end of a transit, an abort), so the servo model itself leaves every signal
# the rest of the service reads. 2 ms of controller time: the Z loop closes in under
# 1 ms, and the block costs about a millisecond to compute.
REST_TICKS = int(0.002 * CLOCK_HZ)

# Positions per second of WALL time the twin reports while it moves the tip, at any
# `twin_speed`: the tip stream's own tick (service/twin_tip_stream.py) and a write's
# reports. A transit report is one `tip_pos`; a stroke report is one `litho_progress`
# (carrying everything passed since the last one as `trail_um`) followed by the
# `tip_pos` for the same point. The instrument reports the tip from its telemetry
# poll at 4-10 Hz.
TWIN_FRAME_HZ = 10.0
REPORT_S = 1.0 / TWIN_FRAME_HZ
I_NA_PER_FRAME = 40                          # live-current values in one report


def arm_z_servo(arc, setpoint: float, mode: str = "contact") -> None:
    """Configure PIS loop 2 as the Z servo at `setpoint`. The gains and output
    bounds are the instrument's, fixed; only the setpoint and (now) the feedback
    channel vary between approach, withdraw, transit and write.

    mode selects the feedback channel in the TWIN's own vocabulary, which differs
    from the hardware's (SoftwareARC.step resolves loop inputs by name):
      - contact/current -> "Fast%Input" (deflection); byte-identical to before.
      - ac              -> "Lockin.0.R" (amplitude).
    SAFETY note on the gain sign: unlike the hardware Z loop (hw_approach, where AC
    NEGATES the gain because amplitude falls on approach), the twin encodes that
    inversion in the ERROR DIRECTION instead -- its amplitude branch computes
    `err = amp - target` while deflection computes `err = target - defl` -- so the
    twin keeps a POSITIVE gain in both modes and still drives toward the setpoint.
    The channel name is therefore what makes the twin loop mode-correct; the gain
    stays 1200. Default "contact" so every current caller (approach, withdraw,
    transit, and the always-contact litho run) is unchanged."""
    z_in = "Lockin.0.R" if str(mode).lower() == "ac" else "Fast%Input"
    arc.ir_SetPISLoop(2, "Always,Never", z_in, setpoint,
                      0.0, 1200.0, 0.0, "Z%Output", -10.0, 150.0)


@dataclass
class ScanConfig:
    size_um: float = 30.0
    pixels: int = 256              # points per line (fast axis)
    lines: int = 0                 # lines (slow axis); 0 = same as pixels
    line_rate_hz: float = 1.0
    x_offset_um: float = 0.0
    y_offset_um: float = 0.0
    # "contact" (z on deflection), "ac" (z on amplitude), or "current"
    # (z on tip current). The last is not an exotic addition: XCalculated.ipf
    # maps a Current trigger channel to STM mode driving the Height loop, so
    # current feedback is a first-class imaging mode in the vendor software,
    # and it is exactly the c-AFM regime.
    mode: str = "contact"          # "contact" | "ac" | "current"
    bias_v: float = 1.0            # tip bias while imaging on current
    setpoint: float = 0.2          # V (defl for contact, amplitude for ac)
    angle_deg: float = 0.0         # fast-axis angle, arbitrary (deg CCW from +x)
    direction: str = "up"          # frame builds "up" or "down"
    # scan each line both ways (forward trace + reverse retrace), captured separately.
    # The /api/scan/start endpoint turns this ON by default for real scans; the raw
    # dataclass default is OFF so internal/test single-sweep rasters are unchanged.
    trace_retrace: bool = False
    # engage the Z servo at `setpoint` for real imaging. Default OFF: the tip stays
    # retracted (no Z into the surface) until the safety kernel exists.
    engage: bool = False
    # Every hardware scan is callback-chained (Igor advances lines on the capture-fill
    # callback). Kept only so the echoed config still reads chained=True; an explicit
    # False is refused in scan_cmds.build_scan_config.
    chained: bool = True
    # Optional Z-servo gain override for THIS scan, hardware backend only, and only
    # meaningful when `engage` is true (see commands.scan_cmds.start_scan, which
    # rejects them -32602 without engage -- same rule as `setpoint`). PANEL-gain
    # convention: the same units `setZGain`/`startApproach` take (scaled to
    # controller units by hw_approach._PSCALE/_ISCALE, signed by mode via
    # hw_approach.mode_loop_params), so a value that tunes the live loop through
    # setZGain tunes the same loop here. None (default) leaves the scan's normal
    # default Z gains untouched. The twin has no servo to tune -- like `setpoint`,
    # it accepts and ignores these (echoed back in the config, never acted on).
    pgain: float | None = None
    igain: float | None = None

    def __post_init__(self):
        if not self.lines:
            self.lines = self.pixels


@dataclass
class ScanState:
    config: ScanConfig
    row: int = 0
    channels: dict[str, np.ndarray] = field(default_factory=dict)
    running: bool = False
    aborted: bool = False
    # Thermal drift (m) of the sample under the tip, captured at start_scan from
    # World.drift_v (V/s) x LVDT sens x elapsed time and held CONSTANT for this
    # frame -- so two consecutive scans of the same region differ by a pure
    # translation the drift estimator can recover. Zero when drift_v is (0, 0).
    drift_m: tuple[float, float] = (0.0, 0.0)


class TwinInstrument:
    """High-level vocabulary over one SoftwareARC. The service publishes this."""

    def __init__(self, arc: SoftwareARC):
        self.arc = arc
        self.world: World = arc.world
        self.scan: ScanState | None = None
        self.rng = np.random.default_rng(11)

    # ------------------------------------------------------------ approach --

    def approach(self, target_defl_v: float = 0.2, speed_v_per_s: float = 4.0):
        """Z ramp with threshold trip, through the per-tick ARC servo path."""
        arc = self.arc
        arm_z_servo(arc, target_defl_v)
        # run until deflection reaches setpoint or timeout (simulated 4 s)
        for _ in range(200):
            arc.step(APPROACH_POLL_TICKS)
            if abs(arc.sig[channels.INPUT_FAST] - target_defl_v) < 0.02:
                break
        return arc.snapshot()

    def withdraw(self):
        arc = self.arc
        arm_z_servo(arc, -10.0)
        arc.step(WITHDRAW_TICKS)
        return arc.snapshot()

    def read_free_air_deflection(self) -> float:
        """Off-contact deflection baseline (V), the twin's stand-in for the
        vendor free-air read (td_ReadValue(`DeflectionValue`), ScanMaster.ipf:
        4665). Call after a withdraw+settle; drifts over the world clock so the
        relative-setpoint loop has a moving baseline to track."""
        return self.world.free_air_deflection()

    def read_z_drive(self) -> float:
        """The HV Z-drive voltage (V, -10..+150 V domain), the twin's stand-in for
        the vendor Z-output readback (td_ReadValue(`Output.Z`), HardWare.ipf:
        9595-9596) that the rail-dwell safety guard polls during a continuous scan.
        Walks/pins to a rail via world.z_drive_v / z_drive_drift_v so both guards
        are exercisable off-instrument."""
        return self.world.z_drive_voltage()

    def read_sum(self) -> float:
        """The photodiode SUM voltage (V), the twin's stand-in for the vendor
        td_ReadValue(`DetectorSum`) (CONFIRMED on the Cypher and the MFP-3D,
        2026-09-15; lowercase "sum" reads 0 V on the Cypher) the host-side
        fail-safe reads each re-reference for the 0.9*SUM clamp and the SUM-loss
        interlock. Drives via world.detector_sum_v / detector_sum_drift_v so both
        monitors are exercisable off-instrument."""
        return self.world.sum_voltage()

    # --------------------------------------------------------------- scan --

    def start_scan(self, config: ScanConfig) -> ScanState:
        n = config.pixels
        n_lines = config.lines
        # Freeze this frame's drift: the sample has walked drift_v (V/s) x sens
        # x elapsed-time since t=0. Held constant across the frame's lines, so a
        # later scan of the same region is a pure translation of this one.
        w = self.world
        drift_m = (w.drift_v[0] * w.x_lvdt_sens * w.t,
                   w.drift_v[1] * w.y_lvdt_sens * w.t)
        self.scan = ScanState(config=config, drift_m=drift_m, channels={
            channels.HEIGHT: np.full((n_lines, n), np.nan, dtype=np.float32),
            {"contact": channels.DEFLECTION, "ac": channels.AMPLITUDE,
             "current": "Current"}.get(config.mode, channels.DEFLECTION):
                np.full((n_lines, n), np.nan, dtype=np.float32),
            channels.Z_SENSOR: np.full((n_lines, n), np.nan, dtype=np.float32),
            "X_um": np.full((n_lines, n), np.nan, dtype=np.float32),
            "Y_um": np.full((n_lines, n), np.nan, dtype=np.float32),
        })
        self.scan.running = True
        return self.scan

    def scan_line(self) -> dict | None:
        """Render the next line; returns per-channel rows or None when done."""
        s = self.scan
        if s is None or not s.running:
            return None
        cfg = s.config
        n = cfg.pixels
        n_lines = cfg.lines
        row = s.row
        if row >= n_lines:
            s.running = False
            return None

        size_m = cfg.size_um * 1e-6
        # slow-axis index: frame builds up (bottom row first) or down
        slow_idx = row if cfg.direction == "up" else (n_lines - 1 - row)
        slow = -size_m / 2 + size_m * slow_idx / max(n_lines - 1, 1)

        # Arbitrary fast-axis angle, through the SAME rotation lithography
        # uses (litho/frame.py). The vendor's sign is not the intuitive one:
        # at a positive scan angle the fast axis is (cos A, -sin A). Keeping
        # imaging and writing on one rotation is what makes a pattern drawn
        # on a rotated image land on the feature it was drawn over.
        ux, uy = rotate(1.0, 0.0, cfg.angle_deg)      # fast unit vector
        vx, vy = rotate(0.0, 1.0, cfg.angle_deg)      # slow unit vector

        # oversample the line 4x along the fast axis
        m = n * 4
        fast = np.linspace(-size_m / 2, size_m / 2, m)
        cx = cfg.x_offset_um * 1e-6
        cy = cfg.y_offset_um * 1e-6
        xs = cx + ux * fast + vx * slow
        ys = cy + uy * fast + vy * slow
        # Thermal drift: the scanner reports xs/ys (undrifted -- that is where
        # the piezo is), but the sample material under those coordinates has
        # walked by drift_m, so a feature at sample-coord s now reads at scanner
        # xs = s + drift_m. Sample the surface at the drifted coordinates; keep
        # xs/ys as the reported scanner positions (X_um/Y_um below).
        dxm, dym = s.drift_m
        surface = self.world.sample.height(xs - dxm, ys - dym)

        # scanner + z-servo dynamics along the line, trace and retrace
        line_time = 1.0 / cfg.line_rate_hz / 2.0     # trace half-period
        dt = line_time / m
        tau_z = 1.2e-3 / max(cfg.line_rate_hz, 0.1)  # slower servo at high rates
        a = math.exp(-dt / tau_z)
        z = lfilter([1 - a], [1, -a], surface, zi=[surface[0] * a])[0]
        z_re = lfilter([1 - a], [1, -a], surface[::-1], zi=[surface[-1] * a])[0][::-1]
        err = surface - z

        noise = self.rng.normal(0, self.world.noise_z, m)
        height_px = _decimate(z + noise, n)
        retrace_px = _decimate(z_re + noise[::-1], n)
        z_sensor_px = _decimate(z, n) + self.rng.normal(0, 0.15e-9, n)

        if cfg.mode == "current":
            # Current-feedback imaging. The servo holds the tip current at
            # the setpoint, so the recorded height is a contour of constant
            # conductance rather than of constant force: a written wire
            # stands out even where the topography is flat. The channel
            # itself is the current, in nanoamps, signed as ORCA signs it.
            w = self.world
            gx = xs - s.drift_m[0]           # conductance drifts with the sample
            gy = ys - s.drift_m[1]
            g = np.array([w.sample.conductance(float(a), float(b))
                          for a, b in zip(gx, gy)])
            # Sample.conductance returns a saturating 0..1 occupancy, not
            # siemens. Scaling it by the sheet conductance keeps the units
            # real; without it a written wire read 4.7e8 nA, half an amp.
            i_a = g * G_SAT_S * cfg.bias_v
            inst = getattr(w, "instrument", None)
            if inst is not None and getattr(inst, "is_orca", False):
                i_a = i_a * inst.orca_sign
            err_px = _decimate(i_a, n) * 1e9 + self.rng.normal(0, 0.002, n)
            err_name = "Current"
            # the height a current servo would report: it lifts over
            # conducting material to hold the setpoint
            height_px = height_px + _decimate(g, n) * 2e-9
        elif cfg.mode == "contact":
            err_px = _decimate(err / self.world.lever.invols, n) \
                + self.rng.normal(0, self.world.noise_defl_v, n)
            err_name = channels.DEFLECTION
        else:
            free = self.world.lever.free_amplitude_v
            err_px = np.clip(free - _decimate(err, n) / self.world.lever.invols * 0.5,
                             0, free) + self.rng.normal(0, 0.004, n)
            err_name = channels.AMPLITUDE

        s.channels[channels.HEIGHT][slow_idx, :] = height_px
        s.channels[err_name][slow_idx, :] = err_px
        s.channels[channels.Z_SENSOR][slow_idx, :] = z_sensor_px
        s.channels["X_um"][slow_idx, :] = _decimate(xs, n) * 1e6
        s.channels["Y_um"][slow_idx, :] = _decimate(ys, n) * 1e6
        # Keep the world's notion of tip position honest for telemetry and
        # the minimap. A reported line is a trace AND a retrace, so the tip
        # ends the cycle back at the start of the line, not at its far end:
        # the only motion between lines is one slow-axis step. Leaving it at
        # xs[-1] made the tip appear to fly the full width every line.
        w = self.world
        w.x_v = (xs[0] / w.x_lvdt_sens) + w.x_lvdt_offset
        w.y_v = (ys[0] / w.y_lvdt_sens) + w.y_lvdt_offset
        w.z_v = -(z[0] / w.z_lvdt_sens) + w.z_lvdt_offset
        # Advance the world clock by one full line period (trace + retrace) so
        # drift accumulates across the frame and BETWEEN successive scans; with
        # drift_v == (0, 0) this is inert (drift_m stays zero).
        w.t += 1.0 / max(cfg.line_rate_hz, 1e-6)
        self.arc.sig.update({channels.X_SENSOR: w.x_v, channels.Y_SENSOR: w.y_v})
        if cfg.mode == "ac":
            self.arc.sig["Lockin.0.R"] = float(np.mean(err_px))
            self.arc.sig[channels.INPUT_FAST] = 0.0
        s.row += 1
        if s.row >= n_lines:
            s.running = False
        return {
            "row": row,
            "slow_idx": slow_idx,
            channels.HEIGHT: height_px,
            "Retrace": retrace_px,
            err_name: err_px,
            channels.Z_SENSOR: z_sensor_px,
            "slow_um": slow * 1e6,
        }

    def abort_scan(self):
        if self.scan:
            self.scan.running = False
            self.scan.aborted = True
        self.arc.td_Stop()

    # ------------------------------------------------------------- curves --

    def force_curve(self, z_range_nm: float = 300.0, points: int = 400):
        """Approach/retract deflection vs z, analytic through the lever model."""
        w = self.world
        x_m, y_m = w.positions_m()
        surf = float(w.sample.height(np.array([x_m]), np.array([y_m]))[0])
        z = np.linspace(surf + z_range_nm * 1e-9, surf - 40e-9, points)
        indent = np.maximum(surf - z, 0)
        defl_app = indent / w.lever.invols
        # retract with a small adhesion hysteresis
        defl_ret = np.where(indent > 0, defl_app, -0.12 * np.exp(-(z - surf) / 25e-9))
        defl_ret = np.where(z - surf > 120e-9, 0.0, defl_ret)
        noise = self.rng.normal(0, w.noise_defl_v, points)
        return {
            "z_nm": ((z - surf) * 1e9).tolist(),
            "approach_v": (defl_app + noise).tolist(),
            "retract_v": (defl_ret + noise[::-1]).tolist(),
        }

    # --------------------------------------------------------------- tune --

    def thermal_tune(self, n_points: int = 2 ** 15):
        data = self.arc.thermal(n_points)
        rate = 5e6
        freqs = np.fft.rfftfreq(n_points, 1 / rate)
        psd = np.abs(np.fft.rfft(data)) ** 2 / n_points
        # smooth for display
        k = 15
        psd_s = np.convolve(psd, np.ones(k) / k, mode="same")
        band = (freqs > 50e3) & (freqs < 800e3)
        lever = self.world.lever
        return {
            "f_khz": (freqs[band][::8] / 1e3).tolist(),
            "psd": psd_s[band][::8].tolist(),
            "fit": {"f0_khz": lever.f0 / 1e3, "Q": lever.Q, "k_nm": lever.k},
        }

    def frequency_sweep(self, f_start: float = 200e3, f_stop: float = 400e3, n: int = 600):
        f, amp, phase = self.arc.sweep(f_start, f_stop, n)
        return {
            "f_khz": (f / 1e3).tolist(),
            "amp_v": amp.tolist(),
            "phase_deg": phase.tolist(),
        }


def _decimate(a: np.ndarray, n: int) -> np.ndarray:
    return a.reshape(n, -1).mean(axis=1)


# ---------------------------------------------------------------- lithography


class LithoRun:
    """Plays a compiled pattern on the twin, the real per-segment cycle:
    V=0 -> move to start -> tip down -> arm XY+V out-waves and current capture
    on one event -> fire -> play -> V=0 -> relax. Progress, tip positions and the
    live current go into `queue`, in order, for the service to broadcast.

    The pattern is compiled ONCE into the instrument's wire format (one long
    NaN-separated wave per axis, plus the matching bias wave) and then walked
    back out of it with the same index arithmetic the vendor executor uses.
    That is deliberate: the twin runs what the hardware driver will hand the
    machine, not a parallel representation that happens to agree today. If
    the wire format is wrong, the twin writes the wrong thing.

    The motion is PLAYED, not simulated: `_play` advances along the armed points on
    the wall clock scaled by `twin_speed` (service/twin_player.py), placing the twin
    on the commanded path and marking the sample for the points passed. The waves
    are still armed and fired through the td_ surface, so the command sequence is
    the instrument's; the controller model is stepped only for REST_TICKS where the
    tip comes to rest.
    """

    def __init__(self, inst: TwinInstrument, pattern, queue, policy=None,
                 surface=None, frame=None, amp_gain: float = 1.0,
                 cmd_capture=None, tip_frame=None, twin_speed=None,
                 clock=time.monotonic, sleep=time.sleep):
        from ..litho.frame import LithoFrame
        from ..litho.transit import TransitPolicy
        from .twin_player import Pacer, speed_from_env
        self.inst = inst
        self.pattern = pattern
        self.queue = queue
        self.abort = False
        self.policy = policy or TransitPolicy()
        self.surface = surface
        # The external tip-bias amplifier gain the write is armed with. The DAC
        # drives V_tip / amp_gain, so the COMMANDED Output.A at each written point
        # is that point's tip voltage over the gain -- the same `v_lit` the
        # hardware LithographyRun loads. Recorded at full resolution (13.1 item 16).
        self.amp_gain = float(amp_gain) if amp_gain else 1.0
        # The capture accumulator this run appends its commanded Output.A into
        # (`hub.hw_litho_capture`), under key `output_a_cmd_v`, in write order across
        # objects. The twin path records no measured hardware capture, so this array
        # is what makes writes/npz/<stem>.npz non-empty for a twin write.
        self._cmd_capture = cmd_capture
        # Display hook: tip_frame(x_um, y_um) returns the `tip_pos` message for a
        # position, or None when none may be shown. The message goes into `queue`
        # like every other frame of the run, so the page receives positions and
        # progress in the order this thread produced them.
        self._tip_frame = tip_frame
        # None takes the station's setting (env AFM_LITHO_TWIN_SPEED, else real time).
        self.pacer = Pacer(speed_from_env() if twin_speed is None else twin_speed,
                           lambda: self.abort, clock, sleep)
        self._next_report = 0.0
        # The litho frame. Its angle is the IMAGING scan angle (the vendor
        # reads RVW[%ScanAngle] for it), so a pattern drawn on a rotated
        # image is written in that same rotated frame. Sizes and offsets are
        # the litho panel's own.
        self.frame = frame or LithoFrame()

    # ------------------------------------------------------------- frames --

    def _tip(self, x_um: float, y_um: float) -> None:
        if self._tip_frame is None:
            return
        try:
            msg = self._tip_frame(float(x_um), float(y_um))
        except Exception:                                # noqa: BLE001
            msg = None                                   # display only
        if msg is not None:
            self.queue.put(msg)

    def _put_progress(self, name, frac, i_na, x_um, y_um, trail) -> None:
        self.queue.put({
            "type": "litho_progress",
            "name": name,
            "frac": min(1.0, frac),
            "i_na": i_na,
            "x_um": x_um,
            "y_um": y_um,
            "trail_um": trail,
        })

    # -------------------------------------------------------- twin state --

    def _here_um(self) -> tuple[float, float]:
        arc, w = self.inst.arc, self.inst.world
        return ((arc.sig[channels.X_SENSOR] - w.x_lvdt_offset) * w.x_lvdt_sens * 1e6,
                (arc.sig[channels.Y_SENSOR] - w.y_lvdt_offset) * w.y_lvdt_sens * 1e6)

    def _servo_z(self, surf: float) -> float:
        """Z (volts) where the Z loop, as it is armed now, holds the tip over a
        surface at height `surf`."""
        arc, w = self.inst.arc, self.inst.world
        lp = arc.loops[2]
        if not lp.running or not lp.input.startswith("Fast"):
            return w.z_v
        setpoint = arc.params.get("PIDSLoop.2.Setpoint", 0.0)
        if setpoint <= 0:
            return lp.out_min
        z = w.z_lvdt_offset - (surf - setpoint * w.lever.invols) / w.z_lvdt_sens
        return min(max(z, lp.out_min), lp.out_max)

    def _tip_down(self) -> bool:
        """Whether the tip is on the surface where the twin is now: within 5 nm of
        it, the contact condition `SoftwareARC.step` writes under."""
        arc, w = self.inst.arc, self.inst.world
        with arc._lock:
            surf = w.sample.height_scalar(*w.positions_m())
            tip_h = -(self._servo_z(surf) - w.z_lvdt_offset) * w.z_lvdt_sens
        return surf - tip_h > -5e-9

    def _place(self, x_v: float, y_v: float, bias=None) -> None:
        """Put the twin at commanded scanner volts (x_v, y_v): the world position,
        the XY setpoints, Z where the servo holds it there, and every signal the
        controller model publishes after a block. `bias` also sets Output.A and the
        tip current it reads. Takes the arc lock for the assignment only.

        Z follows the Z loop as it is armed: on the surface at a positive deflection
        setpoint, at the loop's lower bound when armed to lift (-10 V), and left
        where it is when the loop is off or is not a deflection loop."""
        arc, w = self.inst.arc, self.inst.world
        with arc._lock:
            w.x_v, w.y_v = float(x_v), float(y_v)
            # The XY loops track their setpoints directly, so a setpoint left behind
            # would pull the scanner back on the next block anyone steps.
            arc.params["PIDSLoop.0.Setpoint"] = w.x_v
            arc.params["PIDSLoop.1.Setpoint"] = w.y_v
            x_m, y_m = w.positions_m()
            lp = arc.loops[2]
            invols = w.lever.invols
            surf = w.sample.height_scalar(x_m, y_m)
            z = self._servo_z(surf)
            if lp.running and lp.input.startswith("Fast"):
                # the loop's own output, or the next block stepped would drive Z
                # back to wherever its integrator was left
                lp.integrator = z
            tip_h = -(z - w.z_lvdt_offset) * w.z_lvdt_sens
            indent = surf - tip_h
            defl = indent / invols if indent > 0 else 0.0
            free_v = w.lever.free_amplitude_v
            free_m = free_v * invols
            a = min(max(-indent / free_m, 0.0), 1.0) if indent < 0 else 0.0
            w.z_v = z
            w.engaged = bool(defl > 0.02 or (0 <= -indent < 5 * free_m))
            arc.sig.update({
                channels.X_SENSOR: w.x_v, channels.Y_SENSOR: w.y_v, channels.Z_SENSOR: z,
                channels.INPUT_FAST: defl, "Lockin.0.R": free_v * (a * (2 - a)),
                channels.OUTPUT_X: w.x_v, channels.OUTPUT_Y: w.y_v, "Z%Output": z,
            })
            if arc.is_cypher:
                arc.sig[channels.HEIGHT] = z
            if bias is not None:
                arc.sig[channels.OUTPUT_A] = float(bias)
                arc.sig[channels.INPUT_A] = w.sample.conductance(x_m, y_m) * float(bias) * 0.5

    def _rest(self) -> None:
        """The tip has come to rest: let the controller model itself settle it where
        it was placed, so its own signals (noise, engaged, the Z drive) are the state
        the next scan, park or withdraw starts from."""
        w = self.inst.world
        self._place(w.x_v, w.y_v)
        self.inst.arc.step(REST_TICKS)

    def _fire(self) -> None:
        """Fire Event.0 with the armed waves out of the controller model's hands:
        `_play` moves the tip, and a bank the event activated would be replayed from
        its first point by any other caller's `arc.step`. The banks are detached
        for the length of the fire, under the arc lock, so the event starts none of
        them and no other thread steps in between."""
        arc = self.inst.arc
        with arc._lock:
            out, captures = dict(arc.out_banks), dict(arc.in_banks)
            arc.out_banks.clear()
            arc.in_banks.clear()
            try:
                arc.td_WriteString("Event.0", "Once")
            finally:
                arc.out_banks.update(out)
                arc.in_banks.update(captures)

    # ------------------------------------------------------------ player --

    def _play(self, lo: int, hi: int, dt_s: float, t0: float, advance, report) -> int:
        """Advance from point `lo` to point `hi` of a path whose point i is reached
        `i * dt_s` simulated seconds after wall time `t0`. Returns the point reached:
        `hi`, or less when a stop landed.

        `advance(i, j)` moves the twin from point i to point j; `report(i)` sends
        point i to the page, once per REPORT_S of wall time. Measuring from `t0`
        keeps a late wake-up from accumulating. Between looks at the clock the
        thread sleeps at most WAIT_SLICE_S, and it tests the stop flag before every
        advance, so a stop is seen within one slice at any speed."""
        pacer = self.pacer
        idx = lo
        t_end = t0 + pacer.wall(hi * dt_s)
        while not self.abort:
            now = pacer.now()
            reached = pacer.sim(now - t0) / dt_s + 1e-9
            target = hi if reached >= hi else max(idx, int(reached))
            if target > idx:
                advance(idx, target)
                idx = target
            if idx >= hi:
                break
            if now >= self._next_report:
                report(idx)
                self._next_report = max(self._next_report + REPORT_S,
                                        now + REPORT_S / 2)
            pacer.wait_until(min(self._next_report, t_end, now + WAIT_SLICE_S))
        return idx

    def _transit_to(self, target_um, setpoint: float = 0.2):
        """Move the tip from where it is to target_um under the policy.

        The move is armed through the same out-bank machinery as a write, with the
        bias pinned at zero, and played along the planned bounded-speed trajectory.
        """
        from ..litho.transit import plan_transit
        arc, w = self.inst.arc, self.inst.world
        plan = plan_transit(self._here_um(), target_um, self.policy, self.surface)
        if not plan["n"]:
            return plan

        arc.td_WriteValue(channels.OUTPUT_A, 0.0)          # nothing is written in transit
        lifted = plan["mode"] == "lift"
        if lifted:
            arm_z_servo(arc, -10.0)
            self.pacer.dwell(LIFT_CLEAR_TICKS / CLOCK_HZ)
            self._rest()

        path = plan["path_um"]
        x_v = path[:, 0] * 1e-6 / w.x_lvdt_sens + w.x_lvdt_offset
        y_v = path[:, 1] * 1e-6 / w.y_lvdt_sens + w.y_lvdt_offset
        if plan["mode"] == "none":
            arc.td_WriteValue("PIDSLoop.0.Setpoint", float(x_v[-1]))
            arc.td_WriteValue("PIDSLoop.1.Setpoint", float(y_v[-1]))
            self._place(x_v[-1], y_v[-1], bias=0.0)
            self.pacer.dwell(JUMP_SETTLE_TICKS / CLOCK_HZ)
            self._tip(float(path[-1, 0]), float(path[-1, 1]))
        else:
            ticks = max(1, int(plan["duration_s"] * CLOCK_HZ) // max(len(x_v) - 1, 1))
            arc.td_xSetOutWavePair(0, "0", "PIDSLoop.0.Setpoint", x_v,
                                   "PIDSLoop.1.Setpoint", y_v, -ticks)
            # Arm the bias explicitly at zero for the length of the move.
            # Event.0 re-fires every bank registered against it, so leaving
            # the previous stroke's voltage wave armed would replay that
            # stroke's bias along the transit and write a line between the
            # two objects. Writing zeros is the only way to be sure.
            arc.td_xSetOutWave(1, "0", channels.OUTPUT_A, np.zeros(len(x_v)), -ticks)
            self._fire()
            dt_s = ticks * DT

            def advance(i, j):
                self._place(x_v[j], y_v[j], bias=0.0)
                arc.t += (j - i) * dt_s

            t0 = self.pacer.now()
            self._next_report = t0 + REPORT_S
            reached = self._play(
                0, len(x_v) - 1, dt_s, t0, advance,
                lambda i: self._tip(float(path[i, 0]), float(path[i, 1])))
            # Where the move stopped (its end, or an abort) is always reported.
            if reached > 0:
                self._tip(float(path[reached, 0]), float(path[reached, 1]))
            arc.td_Stop()

        if lifted:
            arm_z_servo(arc, setpoint)
            self.pacer.dwell(self.policy.settle_s)       # land and settle
        self._rest()
        self.queue.put({
            "type": "litho_transit",
            "mode": plan["mode"],
            "distance_um": round(plan["distance_um"], 3),
            "seconds": round(plan["duration_s"], 3),
            "peak_um_s": (None if math.isinf(plan["peak_um_s"])
                          else round(plan["peak_um_s"], 1)),
            "path_um": [[round(float(x), 3), round(float(y), 3)]
                        for x, y in path[::2]],
        })
        return plan

    def _accum_cmd_output_a(self, v_full) -> None:
        """Append the FULL-RESOLUTION commanded Output.A (`v_tip / amp_gain`) for a
        written stroke to the capture accumulator, in write order (13.1 item 16;
        spec sec 5). No-op when no accumulator was supplied. The decimated
        `v_written` on the wire stays UI-only; this is the per-point record."""
        if self._cmd_capture is None:
            return
        vals = np.asarray(v_full, dtype=float) / self.amp_gain
        self._cmd_capture.setdefault("output_a_cmd_v", []).extend(
            float(x) for x in vals)

    def _play_stroke(self, comp, seg) -> int:
        """Play one stroke, tip down throughout; returns the last point reached
        (`len - 1` for a whole stroke, 0 when a stop landed before any motion).

        A stroke longer than the DSP's per-out-wave cap cannot be armed in one go,
        so it is armed as a CHAIN of chunks cut at the same indices the hardware
        driver uses (litho.chunker.chunk_boundaries). Between two chunks there is NO
        arm_z_servo, NO transit and NO `Output.A = 0` -- the servo holds, the bias
        holds at the last point of the previous chunk, and the next chunk resumes
        from the same index. Mirrors Igor's AFM_LithoAdvance, which zeroes the bias
        only at the object's TRUE end.
        """
        from ..litho.chunker import _penup_mask, chunk_boundaries
        from ..litho.compiler import capture_points
        from .twin_player import conductance_at, stamp_path, substeps
        arc, w = self.inst.arc, self.inst.world
        sample = w.sample
        name, tpp = comp["name"], comp["ticks_per_point"]
        xv, yv, v = seg["x"], seg["y"], np.asarray(seg["v"], dtype=float)
        n_pts = len(xv)
        # The points that are path. A compiled stroke may end in samples that only
        # hold the last position at 0 V; they are played but never traced.
        n_path = min(int(comp.get("path_points", n_pts)), n_pts)
        x_m = (np.asarray(xv, dtype=float) - w.x_lvdt_offset) * w.x_lvdt_sens
        y_m = (np.asarray(yv, dtype=float) - w.y_lvdt_offset) * w.y_lvdt_sens
        x_um, y_um = x_m * 1e6, y_m * 1e6
        dt_s = tpp * DT
        parts = substeps(sample, x_m, y_m)
        # A trace position at least every 40 ms of controller time, whatever the speed.
        stride = max(1, STREAM_CHUNK_TICKS // tpp)
        sent = 0                         # last point a frame has reported

        def advance(i, j):
            # The mark follows the COMMANDED path, for exactly the points passed,
            # and only with the tip on the surface: a tip lifted mid-stroke by
            # anything else writes nothing more.
            if self._tip_down():
                stamp_path(sample, x_m[i:j + 1], y_m[i:j + 1], v[i:j + 1], dt_s, parts)
            self._place(xv[j], yv[j], bias=v[j])
            arc.t += (j - i) * dt_s

        def report(i):
            nonlocal sent
            at = min(i, n_path - 1)
            passed = np.arange(min(sent, at) + 1, at + 1)
            trail = passed[(passed % stride == 0) | (passed == at)]
            picks = passed[::max(1, len(passed) // I_NA_PER_FRAME)]
            # Display only: the conductance under the tip times the bias, on the
            # scale the Input.A capture used to be shown at.
            i_na = conductance_at(sample, x_m[picks], y_m[picks]) * v[picks] * 0.5 * 10.0
            self._put_progress(
                name, i / max(n_pts - 1, 1), [round(float(c), 3) for c in i_na],
                float(x_um[at]), float(y_um[at]),
                [[round(float(x_um[k]), 3), round(float(y_um[k]), 3)] for k in trail])
            self._tip(float(x_um[at]), float(y_um[at]))
            sent = i

        bounds = chunk_boundaries(n_pts, _penup_mask(v, xv, yv))
        dec = CAPTURE_DECIMATION
        t0 = self.pacer.now()
        self._next_report = t0 + REPORT_S
        reached = 0
        for k in range(len(bounds) - 1):
            if self.abort:
                break
            lo, hi = bounds[k], bounds[k + 1]
            arc.td_xSetOutWavePair(0, "0", "PIDSLoop.0.Setpoint", xv[lo:hi],
                                   "PIDSLoop.1.Setpoint", yv[lo:hi], -tpp)
            arc.td_xSetOutWave(1, "0", channels.OUTPUT_A, v[lo:hi], -tpp)
            # in-wave lengths are a multiple of 32 on the instrument
            # (td_xSetInWavePair); capture_points pads UP, so an already-aligned
            # length stays put. Same rule the hardware driver applies.
            n_cap = capture_points(max(8, ((hi - lo) - 1) * tpp // dec))
            arc.td_xSetInWavePair(2, "0", channels.INPUT_A, channels.X_SENSOR, n_cap, dec)
            self._fire()
            reached = self._play(reached, hi - 1, dt_s, t0, advance, report)
            # td_Stop frees the banks for the next chunk. It stops the WAVE
            # PLAYBACK only: the Z servo stays armed and Output.A holds the last
            # value the wave drove it to, so the seam is a dwell, not a lift.
            arc.td_Stop()
        # The stroke lasts the compiled `duration_s`, which may run past its last
        # sample (the controller holds it until the capture fills).
        if reached >= n_pts - 1:
            self.pacer.wait_until(t0 + self.pacer.wall(float(comp["duration_s"])))
        # The stroke ended (`frac` 1.0) or a stop landed: what has not been reported
        # goes out now, unless nothing moved at all.
        if reached > sent or (not self.abort and reached >= n_pts - 1):
            report(reached)
        return reached

    def run(self):
        from ..litho.compiler import compile_pattern, split_segments, trace_indices
        arc, w = self.inst.arc, self.inst.world
        # XY loops track their setpoints directly (gain-0, as in the capture).
        # The X/Y piezo drives are members of the Output group, so they are
        # `Output.X` and `Output.Y` -- the same dotted form the captures use
        # for both loops. Only the Z servo output is "Z%Output" (a different
        # path); "Y%Output" was that Z-style name misapplied to Y, undocumented
        # in the td reference and absent from the capture. A golden comparison
        # of a one-object write against igor-commands-commented.txt caught it.
        arc.ir_SetPISLoop(0, "Always,Never", channels.X_SENSOR, arc.sig[channels.X_SENSOR],
                          0.0, 0.0, 0.0, channels.OUTPUT_X, -10.0, 150.0)
        arc.ir_SetPISLoop(1, "Always,Never", channels.Y_SENSOR, arc.sig[channels.Y_SENSOR],
                          0.0, 0.0, 0.0, channels.OUTPUT_Y, -10.0, 150.0)

        waves = compile_pattern(self.pattern, w.x_lvdt_sens, w.y_lvdt_sens,
                                w.x_lvdt_offset, w.y_lvdt_offset,
                                frame=self.frame)
        self.queue.put({
            "type": "litho_pattern_armed",
            "segments": waves["segment_count"],
            "nan_count": waves["nan_count"],
            "points": int(len(waves["x"])),
            "duration_s": round(waves["duration_s"], 2),
            "frame": waves["frame"],
        })
        # walk the wire back out, exactly as the instrument's index loop does
        parsed = split_segments(waves["x"], waves["y"], waves["v"])
        # Each piece's span of its DESIGN object, as fractions of the
        # object's total path, so the panel can animate the written part
        # of the object growing while the piece streams: frac within the
        # piece maps to object_frac0..object_frac1 of the whole wire.
        totals: dict[str, float] = {}
        for comp in waves["segments"]:
            totals[comp["object"]] = (totals.get(comp["object"], 0.0)
                                      + comp["length_um"])
        prefix: dict[str, float] = {}
        for seg, comp in zip(parsed, waves["segments"]):
            if self.abort:
                break
            obj_name = comp["object"]
            f0 = prefix.get(obj_name, 0.0) / max(totals[obj_name], 1e-12)
            prefix[obj_name] = prefix.get(obj_name, 0.0) + comp["length_um"]
            f1 = prefix[obj_name] / max(totals[obj_name], 1e-12)
            self.queue.put({"type": "litho_object_start", "name": comp["name"],
                            "object": comp["object"],
                            "voltage": comp["voltage"],
                            "duration_s": round(comp["duration_s"], 2),
                            "length_um": round(comp["length_um"], 2),
                            "object_frac0": round(f0, 4),
                            "object_frac1": round(f1, 4)})

            # 1) bias to zero and transit to the start of the stroke under
            #    the policy: no setpoint jumps, no writing on the way
            start_um = (float(comp["path_um"][0][0]), float(comp["path_um"][0][1]))
            self._transit_to(start_um)
            if self.abort:
                # aborted in transit, before any point of this object was
                # written -- report nothing done for it, matching the
                # hardware worker's break after write_object(aborted=True)
                break

            # 2) tip down (a no-op if the transit never lifted it)
            arm_z_servo(arc, 0.2)
            self.pacer.dwell(TIP_DOWN_TICKS / CLOCK_HZ)
            self._rest()

            # 3+4) arm waves + synchronized capture, fire Event.0, play. The
            #    arrays are the ones parsed out of the NaN wave, not the ones
            #    that went into it.
            n_pts = len(seg["x"])
            reached = self._play_stroke(comp, seg)

            # 5) cleanup: bias to zero and let the stroke relax. Whether the
            #    tip comes up is the transit policy's call, not this loop's:
            #    a slew keeps it down, a lift raises it on the way out.
            arc.td_WriteValue(channels.OUTPUT_A, 0.0)
            self.pacer.dwell(RELAX_TICKS / CLOCK_HZ)
            self._rest()

            if self.abort:
                # cut mid-stroke: cleanup above still ran (bias to zero), but
                # this object was NOT fully written -- withhold the completion
                # claim exactly as the hardware worker does (it breaks between
                # write_object and its litho_object_done put on res.aborted /
                # ab.is_set()). Reporting done here over-marks
                # hub.written_objects/written_accum and makes select_objects
                # silently skip undosed geometry.
                #
                # The operator still needs to SEE and later query what did
                # land before the cut, so report it on a side channel: the
                # prefix of the wire arrays up to the point the player reached
                # (the same index `litho_progress`'s `frac` uses) -- NOT the
                # planned full wave, which would claim points that were never
                # written. A stop that lands before the first advance, like one
                # in the transit, is zero points written and emits nothing.
                n_wire = 0 if reached <= 0 else min(n_pts, reached + 1)
                if n_wire > 0:
                    # the commanded Output.A for exactly the prefix that played,
                    # full-resolution (13.1 item 16), before the mid-stroke break
                    self._accum_cmd_output_a(seg["v"][:n_wire])
                    # display trace: path points only, never the 0 V end sample
                    part = trace_indices(min(n_wire, int(comp["path_points"])))
                    frac_piece = min(1.0, reached / max(n_pts - 1, 1))
                    obj_index = next((i for i, od in
                                      enumerate(self.pattern.objects)
                                      if od.name == comp["object"]), None)
                    x_um = ((seg["x"][:n_wire] - w.x_lvdt_offset)
                            * w.x_lvdt_sens * 1e6)
                    y_um = ((seg["y"][:n_wire] - w.y_lvdt_offset)
                            * w.y_lvdt_sens * 1e6)
                    self.queue.put({
                        "type": "litho_object_partial",
                        "name": comp["name"],
                        "object": comp["object"],
                        "object_index": obj_index,
                        "voltage": comp["voltage"],
                        "frac0": round(f0, 4),
                        "frac1": round(f0 + (f1 - f0) * frac_piece, 4),
                        "path_um": [[round(float(x), 3), round(float(y), 3)]
                                    for x, y in zip(x_um[part], y_um[part])],
                        "v_written": [round(float(v), 3) for v in seg["v"][part]],
                    })
                break

            # the analytic profile of the design object, when it has one,
            # so the panel's written trace never falls back to a sampled
            # polyline that would alias the coils
            prof = next((getattr(od, "profile", None)
                         for od in self.pattern.objects
                         if od.name == comp["object"]), None)
            # the commanded Output.A for the whole stroke, full-resolution
            # (13.1 item 16) -- the decimated v_written below stays UI-only
            self._accum_cmd_output_a(seg["v"])
            # display trace: path points only, never the 0 V end sample
            trace = trace_indices(int(comp["path_points"]))
            self.queue.put({
                "type": "litho_object_done", "name": comp["name"],
                "object": comp["object"],
                "object_frac0": round(f0, 4),
                "object_frac1": round(f1, 4),
                "profile": prof,
                "voltage": comp["voltage"],
                "path_um": [[round(float(x), 3), round(float(y), 3)]
                            for x, y in np.asarray(comp["path_um"])[trace]],
                # what was actually written, point for point, so the canvas
                # can colour the as-written trace by bias
                "v_written": [round(float(v), 3) for v in seg["v"][trace]],
                "surface_samples": self._surface_samples(comp, seg),
            })
        self.queue.put({"type": "litho_done", "aborted": self.abort})

    def _surface_samples(self, comp, seg) -> dict:
        """The stroke as a height profile (the pass IS a scan): the commanded path,
        one sample per 40 ms of controller time, at the height the tip rides the
        sample's height map with the servo at its setpoint."""
        w = self.inst.world
        n_pts = min(int(comp.get("path_points", len(seg["x"]))), len(seg["x"]))
        stride = max(1, STREAM_CHUNK_TICKS // comp["ticks_per_point"])
        idx = np.unique(np.append(np.arange(stride, n_pts, stride), n_pts - 1))
        x_m = (np.asarray(seg["x"], dtype=float)[idx] - w.x_lvdt_offset) * w.x_lvdt_sens
        y_m = (np.asarray(seg["y"], dtype=float)[idx] - w.y_lvdt_offset) * w.y_lvdt_sens
        setpoint = self.inst.arc.params.get("PIDSLoop.2.Setpoint", 0.0)
        z_m = w.sample.height(x_m, y_m) - setpoint * w.lever.invols
        return {
            "x_um": [round(float(x) * 1e6, 4) for x in x_m],
            "y_um": [round(float(y) * 1e6, 4) for y in y_m],
            "z_nm": [round(float(z) * 1e9, 2) for z in z_m],
        }
