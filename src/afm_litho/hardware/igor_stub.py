"""A test-double for the thin Igor bridge (see bridge_protocol.py).

Seeded with instrument values from a `.ibw` note (all 25 startup variables are
present in the note), it answers STARTUP/GETVALUE from that store, RECORDS
everything the driver sends -- the SETVALUE/SERVOLOOP choreography, the
CONVERTED wave payloads, and the ARMFIRE parameters -- and fires DONE after
ARMFIRE (asynchronously, so ABORT can arrive mid-write). That lets the Python
driver be exercised end to end with zero hardware, and its conversions and
per-object cycle asserted against real calibration.
"""

from __future__ import annotations

import re
import socket
import threading
import time
from dataclasses import dataclass
from typing import NamedTuple

from .. import clock
from ..goldens.wire_tap import WireTap
from . import bridge_protocol as P
from .igor_stub_v3 import V3_VERBS, V3Wire

# the two channels a chained scan streams per line, matching AFM_ScanAdvance in
# afm_litho_run.ipf (ZSensor -> topography, Deflection -> feedback error).
_HEIGHT = "ZSensor"
_ERR = "Deflection"

# GV/td_ names the driver may GETVALUE live; default 0 (a test can override).
_LIVE_PARAMS = ("XSensor", "YSensor", "Height", "Deflection", "Amplitude",
                "DetectorSum", "Output.A", "ARSysScanBusy")

# config the .ibw note carries -- the GV names are the note keys verbatim.
_CONFIG_KEYS = ("ScanSize", "ScanRate", "ScanPoints", "ScanLines", "ScanAngle",
                "DeflectionSetpointVolts", "AmplitudeSetpointVolts", "MaxScanSize",
                "InputRange", "IntegralGain", "ProportionalGain", "ThermalFrequency",
                "ThermalQ", "SpringConstant", "InvOLS", "AmpInvOLS", "DriveFrequency",
                "DriveAmplitude", "XLVDTSens", "YLVDTSens", "ZLVDTSens",
                "XLVDTOffset", "YLVDTOffset", "ZLVDTOffset")


# Hardware out-wave limit, 87,380 points (DECISIONS, second round: "technically 87,380").
# Production stays under it: litho/chunker.py MAX_OUTWAVE_POINTS = 87_000 (vendor cMaxOutWave
# in Temp1.ipf:13) and CHAIN_CHUNK_POINTS = 86_976.
MAX_OUT_WAVE_POINTS = 87_380
MAX_BANK = 2
IN_WAVE_MULTIPLE = 32
_XY_SENSORS = ("XSensor", "YSensor")

# ---- safety-simulation constants (SLICES 0.5b); assumptions: slices/0.5b-stub-physics.md
CYPHER_ID = 4                  # cMicroscopeCypher, HardWare.ipf:10
TICK_S = 1 / 50_000            # an out-wave point lasts |interp| controller ticks
SURFACE_Z_V = 1.0              # Z drive at or above this: tip on the surface (park is 0 V)
RETRACT_SETPOINT_V = -5.0      # a Deflection setpoint this low is unreachable in free air
Z_LOOP_SLEW_V_S = 100.0
# Simulated time between AFM_SafePark's loop re-arm and the next command: Igor runs the loop
# on its ~50 ms dispatch tick before the host's next line arrives. An assumption, not a
# measurement; the rig only shows the park's end result (CYPHER_CHECKLIST 2026-09-25/27).
PARK_GAP_S = 0.05
# AFM_SafePark's own confirm (afm_litho_run.ipf kAbortBackupTicks, kRetractedHeightV).
ABORT_BACKUP_S = 5.0
RETRACTED_HEIGHT_V = 1.0
XY_SLEW_V_S = 100.0
_XY_AXIS = {"PIDSLoop.0.Setpoint": "XSensor", "PIDSLoop.1.Setpoint": "YSensor"}
_LOOP_AXIS = {0: "XSensor", 1: "YSensor"}
_Z_DRIVES = ("Height", "Output.Z")


def _pid_loop_param(name: str, field: str) -> int | None:
    """n for "PIDSLoop.<n>.<field>" (or "Cypher.PIDSLoop..."), else None."""
    m = re.fullmatch(r"(?:Cypher\.)?PIDSLoop\.(\d)\." + field, name)
    return int(m.group(1)) if m else None


def _toward(here: float, goal: float, max_step: float) -> float:
    if abs(goal - here) <= max_step:
        return goal
    return here + max_step if goal > here else here - max_step


@dataclass
class _Play:
    """An out-wave playing on one channel: sample k is output at t0 + k * step_s."""
    samples: list
    step_s: float
    t0: float


@dataclass
class _Ramp:
    """IR_SetRamp: move `channel` to `dest` at `rate` per second, for at most `left` s."""
    channel: str
    dest: float
    rate: float
    left: float


class SafetySimulation:
    """Time, bias, XY and Z for the fake controller, in the bridge's `values` store.
    Time is simulated: `advance(dt)` moves it, or `real_time=True` follows the wall clock
    at every command and read. Each rule cites the `.ipf` or audit C."""

    def __init__(self, values: dict, real_time: bool = False, park_gap_s: float = PARK_GAP_S,
                 height_ramp_after_stoploop: bool = False):
        self.values = values
        self.real_time = real_time
        self.park_gap_s = park_gap_s
        # Rig 2026-10-27: a Height ramp with loop 2 stopped does not move Z, so the default is
        # False (the re-armed park loop does the retract). True models the older hypothesis
        # that a stopped loop frees Height; it is kept for the safe-state rows that vary it.
        self.height_ramp_after_stoploop = height_ramp_after_stoploop
        self._park_gap_due = False                  # a park re-arm awaits its gap (both platforms)
        self.now = 0.0
        self._last_wall = clock.now()
        self._lock = threading.RLock()
        self._plays: dict[str, _Play] = {}          # channel -> out-wave playing on it
        self._ramps: dict[str, _Ramp] = {}
        self._xy_setpoint: dict[str, float] = {}    # "XSensor"/"YSensor" -> setpoint, V
        self.loop_state: dict[int, dict] = {}            # armed PIDS loops by number
        self.bias_log: list[tuple[float, str, float]] = []    # (time, what set it, volts)
        self.ignored_ramps: list[tuple] = []        # RAMPs the controller does nothing with
        for chan in ("Output.A", "Height", "Output.Z", "ZSensor"):
            self.values.setdefault(chan, 0.0)

    # ---- what tests observe ----
    @property
    def is_cypher(self) -> bool:
        return self.values.get("MicroscopeID") == CYPHER_ID

    @property
    def z_drive_channel(self) -> str:
        """The stored Z DAC. The program drives Z by the Height alias on both platforms
        (afm_litho_run.ipf AFM_SafePark); the MFP-3D resolves it to Output.Z, which the stub
        keeps as the one Z value there (ir_readAlias check, 2026-10-27)."""
        return "Height" if self.is_cypher else "Output.Z"

    def _z_name(self, channel: str) -> str:
        """Height names the stored Z DAC: Output.Z on the MFP-3D."""
        return "Output.Z" if channel == "Height" and not self.is_cypher else channel

    @property
    def z_drive(self) -> float:
        return self.values[self.z_drive_channel]

    @property
    def tip_off_surface(self) -> bool:
        return self.z_drive < SURFACE_Z_V

    @property
    def bias_volts(self) -> float:
        return self.values["Output.A"]

    @property
    def xy_moving(self) -> bool:
        """An XY out-wave or ramp is running, or a sensor is short of its setpoint."""
        with self._lock:
            if set(_XY_AXIS) & (set(self._plays) | set(self._ramps)):
                return True
            return any(abs(self.values.get(axis, 0.0) - sp) > 1e-9
                       for axis, sp in self._xy_setpoint.items())

    # ---- time ----
    def sync(self) -> None:
        """Called before every command, read and telemetry tick: the Z readings follow
        the Z drive, and in real-time mode the model catches up to the wall clock."""
        with self._lock:
            if self.real_time:
                self.advance(max(0.0, clock.now() - self._last_wall))
            elif self._park_gap_due:
                self._park_gap_due = False
                self.advance(self.park_gap_s)       # the wall clock supplies this in real time
            self._follow_z_drive()

    def _follow_z_drive(self) -> None:
        """ZSensor and the Height alias read the Z drive (on the Cypher Height is it)."""
        self.values["ZSensor"] = self.values["Height"] = self.z_drive

    def advance(self, dt: float) -> None:
        with self._lock:
            self._last_wall = clock.now()
            self.now += dt
            self._run_ramps(dt)
            self._run_z_loop(dt)
            self._slew_xy(dt)
            self._play_waves()          # last: a playing out-wave overrides ramp and loop
            self._follow_z_drive()

    # ---- commands, as the .ipf runs them ----
    def write_value(self, name: str, value: float) -> None:
        """td_WriteValue via AFM_SetValue."""
        with self._lock:
            if name == "Output.A":
                self._set_bias(value, "write")
            elif name in _XY_AXIS:
                self._xy_setpoint[_XY_AXIS[name]] = value
            elif name == "PIDSLoop.2.Setpoint" and 2 in self.loop_state:
                self.loop_state[2]["setpoint"] = value        # MFP-3D set-down and lift

    def arm_loop(self, n: int, in_chan: str, setpoint: float, out_chan: str,
                 out_min: float, out_max: float) -> None:
        """ir_SetPISLoop with StartEvent "Always": it runs the moment it is armed."""
        with self._lock:
            out_chan = self._z_name(out_chan)
            self.loop_state[n] = {"in": in_chan, "setpoint": setpoint, "out": out_chan,
                             "out_min": out_min, "out_max": out_max}
            if n in _LOOP_AXIS:
                self._xy_setpoint[_LOOP_AXIS[n]] = setpoint

    def undefine_loops(self) -> None:
        """td_Stop undefines every loop (Cypher rig: PIDSLoop.2.Status 1 -> -1)."""
        with self._lock:
            self.loop_state.clear()

    def stop_loop(self, n: int) -> None:
        """ir_StopPISLoop freezes the output where it is (afm_litho_run.ipf:573-575)."""
        with self._lock:
            self.loop_state.pop(n, None)

    def ramp(self, channel: str, dest: float, rate: float, max_s: float) -> None:
        """A ramp moves a raw DAC, but Height is the Z loop's output, not a DAC. Rig 2026-10-27
        (Cypher): with loop 2 stopped a Height ramp does NOT move Z; while loop 2 runs it owns
        Height. So a Height ramp is ignored unless the `height_ramp_after_stoploop` hypothesis
        is set (default off; see __init__). Only the Cypher was rig-tested; making the MFP-3D
        Height ramp inert too is an ASSUMPTION (there Height is the raw Output.Z DAC and a ramp
        might move it). A ramp of `Output.Z` moves only the MFP-3D, where
        it is the Z DAC itself; on the Cypher it is not Z and is ignored."""
        with self._lock:
            loop = self.loop_state.get(2)
            height_free = (channel == "Height" and self.height_ramp_after_stoploop
                           and not (loop and loop["out"] in _Z_DRIVES))
            z_dac = channel == "Output.Z" and not self.is_cypher
            if channel in _XY_AXIS or z_dac or height_free:
                if channel == "Height" and not self.is_cypher:
                    channel = "Output.Z"
                self._ramps[channel] = _Ramp(channel, dest, abs(rate), max_s)
            else:
                self.ignored_ramps.append((channel, dest, rate, max_s))

    def play(self, out_waves: list) -> None:
        """An event fired. `out_waves`: (channels, wave_data, interp) per bound out-bank."""
        with self._lock:
            for channels, data, interp in out_waves:
                step = max(1.0, abs(interp or 1.0)) * TICK_S
                for chan, samples in zip(channels, data or ()):
                    if samples and (chan == "Output.A" or chan in _XY_AXIS):
                        self._plays[chan] = _Play(samples, step, self.now)
            self._play_waves()

    def td_stop(self) -> None:
        """td_Stop: out-waves stop where they are, and the PIDS loops are undefined (rig
        2026-10-27, Cypher.PIDSLoop.2.Status 1 -> -1), so Z holds where it is. Ramps
        surviving td_Stop is an assumption: the .ipf does not say."""
        with self._lock:
            self._plays.clear()
            self.loop_state.clear()
            self._xy_setpoint = {a: self.values.get(a, 0.0) for a in self._xy_setpoint}

    def abort(self, stop: bool) -> None:
        """AFM_SafePark: td_Stop only when `stop` (Igor's flag: once per abort), bias to 0,
        then loop 2 re-armed at -10 V with OutputMin 0, which drives Height to 0 V."""
        with self._lock:
            if stop:
                self.td_stop()
            self._set_bias(0.0, "abort")
        self.arm_loop(2, "Deflection", -10, "Height", 0, 150)
        self._park_gap_due = True

    def vendor_withdraw(self) -> None:
        """Asylum's Withdraw button: kills the loops (2026-10-27) and retracts Z."""
        with self._lock:
            self.loop_state.clear()
            self._ramps.pop(self.z_drive_channel, None)
            self.values[self.z_drive_channel] = 0.0
            self._follow_z_drive()

    def zero_bias(self, why: str) -> None:
        """The .ipf's own zeroing: AFM_Done, AFM_LithoDone, AFM_LithoAdvance at the true
        end of an object, AFM_LithoRunEnd."""
        with self._lock:
            self._set_bias(0.0, why)

    # ---- one step of the model ----
    def _set_bias(self, volts: float, why: str) -> None:
        self._plays.pop("Output.A", None)           # a write ends a playing bias wave
        self.values["Output.A"] = volts
        self.bias_log.append((self.now, why, volts))

    def _run_ramps(self, dt: float) -> None:
        for chan, r in list(self._ramps.items()):
            here = self._setpoint_of(chan)
            there = _toward(here, r.dest, r.rate * min(dt, r.left))
            r.left -= dt
            if chan in _XY_AXIS:
                self._xy_setpoint[_XY_AXIS[chan]] = there
            else:
                self.values[chan] = there
            if there == r.dest or r.left <= 0:
                del self._ramps[chan]

    def _setpoint_of(self, chan: str) -> float:
        if chan in _XY_AXIS:
            axis = _XY_AXIS[chan]
            return self._xy_setpoint.get(axis, self.values.get(axis, 0.0))
        return self.values[chan]

    def _run_z_loop(self, dt: float) -> None:
        """Loop 2 holds Z, unless its setpoint is unreachable in free air (the retract
        setpoint): then it drives Z down to its output minimum, no lower."""
        loop = self.loop_state.get(2)
        if not loop or loop["out"] not in _Z_DRIVES or loop["setpoint"] > RETRACT_SETPOINT_V:
            return
        out = loop["out"]
        self.values[out] = _toward(self.values[out], loop["out_min"], Z_LOOP_SLEW_V_S * dt)

    def _slew_xy(self, dt: float) -> None:
        for axis, setpoint in self._xy_setpoint.items():
            self.values[axis] = _toward(self.values.get(axis, 0.0), setpoint, XY_SLEW_V_S * dt)

    def _play_waves(self) -> None:
        """A playing out-wave sets its channel to the current sample, then holds the
        last one when it ends."""
        for chan, p in list(self._plays.items()):
            k = int((self.now - p.t0) / p.step_s)
            ended = k >= len(p.samples) - 1
            v = p.samples[-1] if ended else p.samples[max(k, 0)]
            if chan == "Output.A":
                self.values[chan] = v
            else:
                axis = _XY_AXIS[chan]
                self._xy_setpoint[axis] = self.values[axis] = v
            if ended:
                del self._plays[chan]


class Violation(NamedTuple):
    """One call the real controller would reject. `length` is the wave length, else None."""
    rule: str
    verb: str
    bank: int | None
    length: int | None


def values_from_note(note: dict) -> dict:
    """Build the stub's value store from a `.ibw` note. Config keys come from
    the note; live readings default to 0; ImagingMode becomes an index (as the
    live GV returns), not the note's mode string."""
    def f(k):
        try:
            return float(note.get(k, 0.0))
        except (TypeError, ValueError):
            return 0.0
    v = {k: f(k) for k in _CONFIG_KEYS}
    mode = str(note.get("ImagingMode", "")).strip().lower()
    v["ImagingMode"] = 1.0 if mode.startswith("ac") else 0.0
    if "MicroscopeID" in note:
        v["MicroscopeID"] = f("MicroscopeID")       # the platform the safety simulation runs as
    for k in _LIVE_PARAMS:
        v.setdefault(k, 0.0)
    return v


def startup_string(values: dict) -> str:
    """The AFM_MacroStartup key=value string, built from the value store. Mirrors
    afm_igor.ipf's mapping so the driver sees exactly what real Igor would."""
    def g(k):
        return values.get(k, 0.0)
    mode_idx = int(g("ImagingMode"))
    setpoint = g("AmplitudeSetpointVolts") if mode_idx == 1 else g("DeflectionSetpointVolts")
    sp_mode = "ac" if mode_idx == 1 else "contact"
    fields = [
        ("size_m", g("ScanSize")), ("line_rate_hz", g("ScanRate")),
        ("pixels", g("ScanPoints")), ("lines", g("ScanLines")),
        ("angle_deg", g("ScanAngle")), ("mode", mode_idx), ("setpoint", setpoint),
        ("max_scan_m", g("MaxScanSize")), ("input_range", g("InputRange")),
        ("igain", g("IntegralGain")), ("pgain", g("ProportionalGain")),
        ("f0", g("ThermalFrequency")), ("q", g("ThermalQ")), ("k", g("SpringConstant")),
        ("invols", g("InvOLS")), ("amp_invols", g("AmpInvOLS")),
        ("drive_hz", g("DriveFrequency")), ("drive_v", g("DriveAmplitude")),
        ("x_lvdt_sens", g("XLVDTSens")), ("y_lvdt_sens", g("YLVDTSens")),
        ("z_lvdt_sens", g("ZLVDTSens")),
        ("x_lvdt_offset", g("XLVDTOffset")), ("y_lvdt_offset", g("YLVDTOffset")),
        ("z_lvdt_offset", g("ZLVDTOffset")),
        ("x", g("XSensor")), ("y", g("YSensor")), ("z", g("Height")),
        ("defl", g("Deflection")), ("amp", g("Amplitude")), ("sum", g("DetectorSum")),
        ("bias", g("Output.A")), ("scanning", g("ARSysScanBusy")),
    ]
    s = "".join(f"{k}={float(v):.12g};" for k, v in fields)
    return s + f"setpoint_mode={sp_mode};"


class FakeIgorBridge:
    """Fake Igor, speaking bridge_protocol over one TCP connection."""

    def __init__(self, values: dict, done_delay: float = 0.0, capture=None,
                 telem_hz: float = 0.0, sensor_lag_s: float = 0.0,
                 fill_scale: float | None = None, safety_sim: bool = False,
                 safety_sim_real_time: bool = False,
                 park_gap_s: float = PARK_GAP_S,
                 height_ramp_after_stoploop: bool = False,
                 v3_proto: int = 3, v3_ipf: int = 3, v3_sha: str = "0123456789abcdef"):
        self.values = dict(values)
        self.speaks_v3 = True
        self.v3 = V3Wire(self, proto=v3_proto, ipf=v3_ipf, sha=v3_sha)
        # None: a stroke completes after `done_delay` alone. A number: it completes
        # when its capture fills, as the controller does (capLen x ticks / 50 kHz),
        # that time multiplied by this scale, plus `done_delay`.
        self.fill_scale = fill_scale
        # XSensor/YSensor specifically default to 0.0, not GETVALUE's usual "nan
        # for an unseeded key": xy_transit's single-out-wave-pair move (XY_SLEW_
        # SPEC.md, 2026-09-16) reads them as the tip's LIVE current position for
        # every guarded XY move (not just litho/scan tests that seed a note), and
        # a real controller never reports NaN there -- only a bare `FakeIgorBridge
        # ({}, ...)` did, for a key nothing happened to set. `values_from_note`
        # already defaults every `_LIVE_PARAMS` key including these two the same
        # way; this just makes the SAME default hold for a caller that skips it.
        self.values.setdefault("XSensor", 0.0)
        self.values.setdefault("YSensor", 0.0)
        # DeflectionValue defaults to 0.0: the free-air deflection LithographyRun reads
        # before every object's approach (PER_OBJECT_FREE_AIR_SPEC); a real controller
        # reports a number, never NaN.
        self.values.setdefault("DeflectionValue", 0.0)
        # Output.Z and Height default to 0.0 (withdrawn) for the Z-drive check that read
        # must pass first; a test leaves one high to model a tip not withdrawn.
        self.values.setdefault("Output.Z", 0.0)
        self.values.setdefault("Height", 0.0)
        self.done_delay = done_delay
        # Opt-in tip/bias/XY/Z model (SafetySimulation). Off: readings are static, and ABORT
        # zeroes them by fiat, as before.
        self.safety_sim = SafetySimulation(
            self.values, safety_sim_real_time, park_gap_s,
            height_ramp_after_stoploop) if safety_sim else None
        # Sensor-lag injection (the Cypher "bare FIRE -> immediate DONE" defect, see
        # xy_transit.py): >0 defers the OUTPAIR-time XSensor/YSensor landing by
        # clock seconds (real, or virtual on the testing clock). A bare FIRE's DONE
        # stays synchronous (AFM_FireEvent); a FIRE that arms
        # XSensor/YSensor waits at least this long for DONE (`sensor_lag_gates_done`).
        self.sensor_lag_s = float(sensor_lag_s)
        # False: an armed FIRE's DONE ignores sensor_lag_s, to prove xy_transit's own
        # `ramp_settle_s` guard alone rescues that case.
        self.sensor_lag_gates_done = True
        # last OUTPAIR-bound XY setpoint pair (LVDT volts): the trace a later XSensor/
        # YSensor capture reports, in place of the generic ramp.
        self._xy_transit_wave: dict[str, list[float]] = {}
        # Always-on TELEM poll (real Igor's AFMtip runs from bridge startup). telem_hz > 0
        # emits TELEM frames from the live value store, so a test can drive Output.Z to a
        # rail mid-run; 0 = off. A scalar poll: no in-wave banks.
        self.telem_hz = float(telem_hz)
        self._telem_thread: threading.Thread | None = None
        # what the bridge "measured" and streams back as CAPTURE frames on
        # completion: a dict {channel: samples}, a callable (name, cap_len) ->
        # such a dict, or None for a default single Output.Z ramp of cap_len.
        self.capture = capture
        # Whether litho in-bank 2 armed the UserIn2 + Lateral PAIR. False models the
        # .ipf fail-safe ladder's fallback (pair refused, UserIn2 armed alone): no
        # Lateral frame is streamed on a litho write or chain.
        self.lateral_armed = True
        # Loop 2 armed on Deflection at a reachable setpoint settles at once: Deflection
        # reads the setpoint (LithographyRun confirms every set-down by it); a retract
        # arm puts back the reading from before the set-down. False models a tip that
        # never engages: Deflection stays as seeded.
        self.deflection_follows_loop = True
        self._deflection_before_set_down: float | None = None
        # everything the driver sent, for test assertions:
        self.sets: list[tuple[str, float]] = []
        self.loops: list[tuple] = []
        # (out_min, out_max) per SERVOLOOP, in the SAME order/length as `loops` --
        # `zip(stub.loops, stub.loop_limits)` pairs each arm with its limits. The
        # trailing outMin/outMax tokens are OPTIONAL on the wire (servoloop-limits
        # fix, 2026-09-16): default -10/150 when a line omits them, matching
        # AFM_Dispatch's SERVOLOOP case in afm_litho_run.ipf. Kept as a separate
        # list (not appended into `loops`) so the many existing 6-element
        # unpacks/equality checks against `stub.loops` stay unchanged.
        self.loop_limits: list[tuple[float, float]] = []
        self.stopped_loops: list[int] = []      # STOPLOOP <loop>
        self.ramps: list[tuple] = []            # RAMP (channel, dest, rate, max_s)
        self.waves: dict[str, list[float]] = {}
        self.arms: list[tuple[str, float, int, int]] = []
        # Igor-side bias zeroings, in order: one (name, chunk_index_at_zero) per time
        # the .ipf would run `AFM_SetValue("Output.A", 0)` at the end of a pass. It is
        # Igor, not the driver, that zeroes the bias (AFM_LithoDone; AFM_LithoAdvance
        # ONLY at the TRUE end of a chain, never at a chunk seam), so the "bias never
        # reaches 0 mid-stroke" invariant is not observable in `sets` -- it is modelled
        # here. Exactly one entry per written object is the invariant.
        self.bias_zeros: list[tuple[str, int]] = []
        # EVERY verb the driver sent, in order (WAVE, SERVOLOOP, RAMP, ARMFIRE, ...).
        # The per-verb lists above each record one KIND of command and lose the
        # interleaving; the no-lift invariant is precisely a statement about the
        # interleaving ("no SERVOLOOP/RAMP/SETVALUE between the ARMFIRE and the
        # terminal"), so it needs the ordered stream.
        self.transcript: list[str] = []
        # the same stream with arguments, UIDs normalised, waves summarised.
        self.tap = WireTap()
        self.getgvs: list[str] = []             # names read over GETGV, in order
        self.pid_loops: dict[int, str] = {}     # running PIDS loop number -> its OutputChannel
        # general td_-mirroring surface, recorded for assertions:
        #   out_binds: one tuple per bound OUTPUT channel
        #              (bank, event, channel, wave_name, interp). OUTPAIR appends
        #              TWO of these, one per channel (chA then chB).
        #   in_channels: one tuple per armed INPUT channel
        #              (bank, event, channel, cap_len, decim). INPAIR appends TWO
        #              (chA then chB). Full history is kept across FIREs.
        #   fires: one (name, event) per FIRE.
        self.out_binds: list[tuple] = []
        self.in_channels: list[tuple] = []
        self.fires: list[tuple[str, str]] = []
        # one (xname, yname, ppl, lines, interp, decim) per SCANCHAIN, for asserts.
        self.scanchains: list[tuple] = []
        # error-capture channel of the most recent SCANCHAIN (Deflection|Amplitude).
        self._last_scanchain_err: str = _ERR
        # one (name, xname, yname, vname, boundsName, interp, decim) per LITHOCHAIN
        # (an oversize object fired as a chained continuation), for asserts.
        self.lithochains: list[tuple] = []
        # channels armed since the previous FIRE; the next FIRE emits a CAPTURE
        # for exactly these, then resets. (in_channels keeps the full history.)
        self._armed_in: list[tuple] = []
        self.aborts = 0
        # ABORT == AFM_SafePark on the wire (stop and retract). Knobs: `safe_parks` counts
        # ABORTs; `safe_park_zeros` False = the readback never reaches 0; `safe_park_zero_after`
        # N = only the (N+1)th ABORT zeros it; `safe_park_ramp_s` = the zero lands after
        # that many seconds (async ramp); default 0.0 lands synchronously.
        self.safe_parks = 0
        # STALLED-BRIDGE injection: when True, GETVALUE is SILENTLY DROPPED -- the
        # request is received and no VALUE ever comes back, which is what a wedged
        # Igor looks like from the host. The stand-in for "the read blocks", used to
        # prove every fail-safe / verify read is BOUNDED (an unbounded one would sit
        # on HardwareARC.timeout, 30 s, and make an emergency stop look like a hang).
        self.stall_getvalue = False
        self.safe_park_zeros = True
        self.safe_park_zero_after = 0
        self.safe_park_ramp_s = 0.0
        # AFM_SafePark's globals (afm_litho_run.ipf): `abort_stopped` is root:AFM_abortStopped
        # (td_Stop already ran for this abort; cleared by ABORTCONFIRM or the backup), and
        # `abort_fault` is root:AFM_abortFault (the 5 s backup ran the vendor Withdraw).
        self.abort_stopped = False
        self.abort_fault = False
        # root:AFM_passArmed: a pass armed out-waves or fired since the last td_Stop. While it
        # is set, an abort runs td_Stop even with a stale abort_stopped (AFM_PassMayPlay).
        self.pass_armed = False
        self.td_stops = 0                       # td_Stop calls made by AFM_SafePark
        self.abort_confirms = 0                 # ABORTCONFIRM lines received
        self.fault_clears = 0                   # CLEARFAULT lines received
        self.getstr_passes_error_text = False   # an older .ipf: GETSTR answers "*ERROR*" as a value
        self.backup_runs: list[str] = []        # each 5 s backup outcome: "confirmed" | "fault"
        self.vendor_withdraws = 0               # DoScanFunc("StopEngageButton") from the backup
        self._backup = None                     # the pending 5 s backup clock.timer, if any
        # SCANCHAIN fault injection: when set, a SCANCHAIN replies with ERR <uid> <msg>
        # instead of running the chain -- the deterministic stand-in for a router /
        # bridge fault mid-scan (the chained driver surfaces it as summary["error"]).
        self.scanchain_err: str | None = None
        self.showtips: list[bool] = []          # one True/False per SHOWTIP ON/OFF
        # one vendor mode token per SETMODE (the MainPopupFunc arg: "Contact"/"AC"),
        # so a test can assert the imaging-mode switch reached the bridge (or did NOT,
        # when the endpoint refuses because the tip is engaged).
        self.modes: list[str] = []
        # the crosspoint the READXPT read-back reports (the mode discriminator the litho-
        # AC guard reads). {} -> reply an empty XPT payload, which the driver reads as
        # "unknown" (guard falls back). Set e.g. {"InFast": "FilterOut"} (contact) or
        # {"InFast": "ACDefl"} (AC) to exercise the authoritative read path.
        self.crosspoint: dict[str, str] = {}
        self._cur = ""                          # current pass NAME (for the ABORTED terminal)
        self._cur_uid = ""                      # current pass UID (echoed in ABORTED)
        self._stop = threading.Event()
        self._sendlock = threading.RLock()     # also the lock of wire v3 (igor_stub_v3.py)
        self._timer = None              # the pending completion clock.timer, if any
        # a running SCANCHAIN: the per-line callback runs in its own thread so an
        # ABORT (or stop()) can interrupt it mid-chain, like Igor's real callback.
        self._sc_abort = threading.Event()
        self._sc_thread: threading.Thread | None = None
        # ---- controller limits + event-bound bank model ----
        # Real Igor discards the td_x* return code of every bind and sends no ERR
        # (afm_litho_run.ipf:309-320, 610-649), so a rejected call here is silent too:
        # recorded in `violations`, not bound, no reply. A failed in-bind still has its
        # channel recorded, so with no working in-bank the FIRE never gets a DONE.
        self.violations: list[Violation] = []
        self._ok_in_binds = 0                 # in-binds accepted since the last FIRE
        # out_banks / in_banks: bank -> {"event", "channels", "waves"/"cap_len", "epoch"}.
        # A binding persists until released; whether the controller drops it once an event
        # has played is unknown (F9), so the model keeps it -- the pessimistic case.
        self.out_banks: dict[int, dict] = {}
        self.in_banks: dict[int, dict] = {}
        # one record per fired event: the banks it played, and which out-banks were bound
        # by an earlier pass (stale). `stale_plays` lists them.
        self.bank_fires: list[dict] = []
        self._epoch = 0                       # pass counter; a binding carries its pass
        self._fired = False
        self._bank_lock = threading.RLock()

    @classmethod
    def from_note(cls, note: dict, **kw) -> "FakeIgorBridge":
        return cls(values_from_note(note), **kw)

    @classmethod
    def from_ibw(cls, path, **kw) -> "FakeIgorBridge":
        from ..io.ibw import read_ibw
        return cls.from_note(read_ibw(path).note_dict, **kw)

    # ---- server ----
    def serve_in_thread(self, host: str = "127.0.0.1", port: int = 0):
        """Listen and handle the driver's connection in a daemon thread.
        Returns the bound port. Call stop() to end."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        bound = srv.getsockname()[1]
        threading.Thread(target=self._accept_loop, args=(srv,), daemon=True).start()
        return bound

    def connect_in_thread(self, host: str, port: int):
        """Client role: connect OUT to a listening Python driver and handle the
        link, mirroring real Igor's AFM_LithoRunStart (Igor is the SOCKIT
        client). The complement of serve_in_thread; used to test the hardware
        topology where LithographyRun.serve() accepts and the bridge dials in."""
        link = P.BridgeLink.connect(host, port)
        threading.Thread(target=self._handle, args=(link,), daemon=True).start()

    def advance(self, seconds: float) -> None:
        """Step the simulated clock (safety-simulation mode)."""
        if self.safety_sim is None:
            raise RuntimeError("safety simulation is off; pass safety_sim=True")
        self.safety_sim.advance(seconds)

    def v3_advance(self, seconds: float) -> None:
        """Move the testing clock (and the safety simulation, unless it follows the clock)."""
        clock.advance(seconds)
        sim = self.safety_sim
        if sim is not None and not sim.real_time:
            sim.advance(seconds)

    def set_height(self, volts: float) -> None:
        """Set the Z drive and the readings that follow it (Height, Output.Z, ZSensor)."""
        for k in ("Height", "Output.Z", "ZSensor"):
            self.values[k] = float(volts)

    def drop_host_socket(self) -> None:
        """Close the connection from the instrument side, as a dead Python service does."""
        link = getattr(self, "_v3_link", None) or getattr(self, "link", None)
        if link is not None:
            link.close()
            self.v3.on_close(link)

    def stop(self) -> None:
        self.v3.close()
        if self.safety_sim:
            self.safety_sim.zero_bias("run end")       # AFM_LithoRunEnd
        self._stop.set()
        self._sc_abort.set()
        if self._timer:
            self._timer.cancel()
        self._cancel_backup()

    # ---- AFM_SafePark: the one abort function, its flag and its 5 s backup ----
    def _abort_function(self) -> None:
        """AFM_SafePark's controller side: td_Stop unless a repeat abort finds nothing armed
        since the last one (the flag and AFM_PassMayPlay), bias 0, loop 2 re-armed at -10 V,
        and the 5 s backup re-armed. The terminal frame is the caller's."""
        stop = not self.abort_stopped or self.pass_armed
        if stop:
            self.td_stops += 1
            self.abort_stopped = True               # AFM_StopRan
            self.pass_armed = False
        if self.safety_sim:
            self.safety_sim.abort(stop)
        self._arm_backup()

    def _arm_backup(self) -> None:
        """CtrlNamedBackground AFMabortBackup: stop, then start 5 s from now."""
        self._cancel_backup()
        self._backup = clock.timer(ABORT_BACKUP_S, self._run_backup)
        self._backup.daemon = True
        self._backup.start()

    def _cancel_backup(self) -> None:
        if self._backup is not None:
            self._backup.cancel()
            self._backup = None

    def height(self) -> float:
        """What td_ReadValue("Height") returns: the Z drive (the meter's "Z Voltage")."""
        if self.safety_sim:
            self.safety_sim.sync()
            return float(self.safety_sim.z_drive)
        return float(self.values.get("Height", 0.0))

    def _run_backup(self) -> None:
        """AFM_AbortBackup: nothing if the host confirmed; Height under 1 V clears the flag;
        otherwise the vendor Withdraw once and Igor's fault flag (the abort flag stays)."""
        self._backup = None
        if not self.abort_stopped:
            return
        h = self.height()
        if h < RETRACTED_HEIGHT_V:
            self.abort_stopped = False
            self.backup_runs.append("confirmed")
            self.v3.on_backup(h, "confirmed")
            return
        self.vendor_withdraws += 1
        self.vendor_withdraw()
        self.abort_fault = True
        self.backup_runs.append("fault")
        self.v3.on_backup(h, "fault")

    def _deflection_follows(self, setpoint: float) -> None:
        """Deflection after a loop-2 arm on Deflection (see `deflection_follows_loop`)."""
        if not self.deflection_follows_loop:
            return
        if setpoint > RETRACT_SETPOINT_V:
            if self._deflection_before_set_down is None:
                self._deflection_before_set_down = self.values.get("Deflection", 0.0)
            self.values["Deflection"] = setpoint
        elif self._deflection_before_set_down is not None:
            self.values["Deflection"] = self._deflection_before_set_down
            self._deflection_before_set_down = None

    def vendor_withdraw(self) -> None:
        """DoScanFunc("StopEngageButton"): the vendor Withdraw retracts Z."""
        if self.safety_sim:
            self.safety_sim.vendor_withdraw()
        else:
            for k in ("height", "Height", "Output.Z", "ZSensor", "Deflection"):
                self.values[k] = 0.0

    def _fill_delay(self, points: int, ticks: int) -> float:
        """Seconds until a capture of `points` samples at `ticks` per sample fills."""
        if self.fill_scale is None:
            return self.done_delay
        return self.done_delay + points * abs(ticks) / 50_000.0 * self.fill_scale

    def _accept_loop(self, srv: socket.socket) -> None:
        srv.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._handle(P.BridgeLink(conn))
        srv.close()

    def _handle(self, link: "P.BridgeLink") -> None:
        if self.telem_hz > 0 and self._telem_thread is None:
            # Model real Igor's ALWAYS-ON AFMtip poll: start the TELEM emitter the
            # moment the link is up (independent of SHOWTIP), like AFM_LithoRunStart.
            self._telem_thread = threading.Thread(
                target=self._telem_loop, args=(link,), daemon=True)
            self._telem_thread.start()
        self._v3_link = link
        while not self._stop.is_set():
            try:
                line = link.read_line(timeout=0.5)
            except socket.timeout:
                continue                    # nothing yet: keep waiting
            except OSError:
                line = None
            if line is None:
                self.v3.on_close(link)      # driver closed the connection
                break
            self.v3.intake(line, link, self._dispatch)

    def _telem_loop(self, link) -> None:
        """Emit `TELEM x y zdrive defl sum amp` from the CURRENT value store, at telem_hz,
        until stop(). Mirrors AFM_TipTick: a SCALAR poll of XSensor/YSensor/Output.Z/
        Deflection/DetectorSum/amplitude -- NO in-wave capture. The SUM channel is keyed
        "DetectorSum" (CONFIRMED on the Cypher and the MFP-3D, 2026-09-15 -- matching
        AFM_TipTick's td_ReadValue("DetectorSum") and the host's SUM_CHANNEL read; lowercase
        "sum" is NOT served, see values_from_note / _LIVE_PARAMS). AMPLITUDE stays keyed
        LOWERCASE "amplitude" (the confirmed rig scalar name, untouched by the SUM fix).
        Per-channel NaN-safe (a missing key serializes to "nan" via msg_telem)."""
        nan = float("nan")
        period = 1.0 / self.telem_hz
        while not self._stop.is_set():
            if self.safety_sim:
                self.safety_sim.sync()
            v = self.values
            reads = (v.get("XSensor", nan), v.get("YSensor", nan),
                     v.get("Height" if self.safety_sim else "Output.Z", nan),
                     v.get("Deflection", nan), v.get("DetectorSum", nan), v.get("amplitude", nan))
            frame = (self.v3.tick_frame(*reads) if self.v3.is_v3_link(link)
                     else P.msg_telem(*reads))
            try:
                self._send(link, frame)
            except OSError:
                return                      # link closed: stop emitting
            self.tap.record_pseudo("@TIPTICK")
            time.sleep(period)

    def _loop_status(self, name: str) -> float:
        """PIDSLoop.<n>.Status as the Cypher rig answers it (2026-10-27): 1 for an armed loop,
        -1 otherwise, and -1 always for the bare name. Other names: nan."""
        n = _pid_loop_param(name, "Status")
        if n is None:
            return float("nan")
        if self.values.get("MicroscopeID") == CYPHER_ID and not name.startswith("Cypher."):
            return -1.0
        return 1.0 if n in self.pid_loops else -1.0

    def _send(self, link, line: str) -> None:
        with self._sendlock:
            link.send(line)

    def _capture_for(self, name: str, cap_len: int) -> dict:
        """Resolve the capture config into {channel: samples} for this object.
        Default (None) is NO capture frames -- capture is opt-in per fixture, so
        the plain DONE-only behaviour other tests rely on is unchanged."""
        c = self.capture
        if callable(c):
            return c(name, cap_len)
        return c or {}

    def _complete(self, link, uid: str, name: str, cap_len: int) -> None:
        """Stream the measured capture (one CAPTURE frame per channel), then DONE --
        all tagged with the pass `uid` the driver correlated the fire under."""
        # Each "@FILL" marks where Igor's in-bank fill callback runs; the expander needs it.
        self.tap.record_pseudo("@FILL")
        for chan, vals in self._capture_for(name, cap_len).items():
            if chan == "Lateral" and not self.lateral_armed:
                continue
            self._send(link, P.msg_capture(uid, name, chan, vals))
        self.bias_zeros.append((name, 0))   # AFM_LithoDone: AFM_SetValue("Output.A", 0)
        self._safety_sim_zero_bias("litho done")
        self._send(link, P.msg_done(uid, name))

    def _capture_channel(self, name: str, channel: str, cap_len: int) -> list[float]:
        """Samples for one armed input channel of a general FIRE. Consults
        self.capture: a dict -> self.capture.get(channel); a callable ->
        self.capture(name, cap_len) then [channel] if present. UNLIKE ARMFIRE's
        _capture_for (capture is opt-in there, default = no frames), the general
        FIRE path DEFAULTS to a deterministic ramp so downstream tests always see
        a capture for every armed channel: [i/(n-1) if n>1 else 0.0 ...] of
        cap_len. Config None, or a config that lacks this channel, falls back to
        that ramp -- UNLESS `channel` is XSensor/YSensor and the most recent
        OUTPAIR bound the XY setpoint pair (`_xy_transit_wave`, set in
        `_dispatch`'s OUTPAIR case): then the fallback is THAT wave (padded by
        repeating its last sample, or truncated, to cap_len) -- a realistic
        "sensor tracked the commanded setpoint" trace for xy_transit's own
        capture-armed move, not an unrelated 0..1 ramp."""
        c = self.capture
        vals = None
        if callable(c):
            d = c(name, cap_len)
            if d and channel in d:
                vals = d[channel]
        elif isinstance(c, dict):
            vals = c.get(channel)
        if vals is not None:
            return list(vals)
        wave = self._xy_transit_wave.get(channel)
        if wave:
            n = int(cap_len)
            if len(wave) >= n:
                return list(wave[:n])
            return list(wave) + [wave[-1]] * (n - len(wave))
        n = int(cap_len)
        return [i / (n - 1) if n > 1 else 0.0 for i in range(n)]

    def _complete_fire(self, link, uid: str, name: str, armed: list[tuple]) -> None:
        """General FIRE completion: one CAPTURE per armed input channel (in the
        order they were armed), then DONE -- all tagged with the pass `uid`."""
        self.tap.record_pseudo("@FILL")
        self._safety_sim_zero_bias("done")     # AFM_Done
        for (_bank, _event, channel, cap_len, _decim) in armed:
            self._send(link, P.msg_capture(
                uid, name, channel, self._capture_channel(name, channel, cap_len)))
        self._send(link, P.msg_done(uid, name))

    def _safety_sim_zero_bias(self, why: str) -> None:
        if self.safety_sim:
            self.safety_sim.zero_bias(why)

    def _complete_lithochain(self, link, uid: str, name: str, bounds: list[int],
                             bad_chunks=frozenset(), xyv=None, interp=None) -> None:
        """Mirror AFM_LithoArmChunk/AFM_LithoAdvance: for each chunk
        `[bounds[k]:bounds[k+1]]` stream a CAPTURE for each of the SIX channels the
        .ipf sends -- UserIn2 (measured bias), Lateral (when the bank-2 pair armed),
        ZSensor, Deflection, XSensor, YSensor, in that order -- ALL under the one
        object `name`, at that chunk's padded capture length -- then a single terminal DONE. The driver ACCUMULATES the
        same-channel frames across chunks into one array per channel, exactly as it
        will off real Igor.

        Every channel the .ipf streams, because the volume is the point -- six
        ~87k-point captures is ~5.8 MB of "%.9g;" text formatted INSIDE the
        capture-fill callback with the tip stationary at full bias."""
        for k in range(len(bounds) - 1):
            m = bounds[k + 1] - bounds[k]
            cap = m + (32 - m % 32) % 32              # AFM_LithoCapPoints / capture_points
            if k:
                self._bank_arm_litho(name, cap, out_ok=(k not in bad_chunks,) * 2,
                                     interp=interp, xyv=self._slice3(xyv, bounds[k], bounds[k + 1]))
            self.tap.record_pseudo("@FILL")
            for chan in ("UserIn2", "Lateral", "ZSensor", "Deflection", "XSensor", "YSensor"):
                if chan == "Lateral" and not self.lateral_armed:
                    continue
                self._send(link, P.msg_capture(
                    uid, name, chan, self._capture_channel(name, chan, cap)))
            # AFM_LithoAdvance zeroes Output.A ONLY when the chain is finished -- the
            # seam itself keeps the bias live and the tip down. Model that exactly.
            if k == len(bounds) - 2:
                self.bias_zeros.append((name, k))
                self._safety_sim_zero_bias("litho chain end")
        self._send(link, P.msg_done(uid, name))

    def _run_scanchain(self, link, uid: str, xname: str, yname: str, ppl: int,
                       nlines: int, err_chan: str = _ERR, out_ok: bool = True,
                       interp=None) -> None:
        """Mirror AFM_ScanChainStart/AFM_ScanAdvance: for each line emit a ZSensor
        and an ERROR CAPTURE named "L<line>", advancing on the (simulated) capture-fill
        callback, then a terminal SCANDONE. `err_chan` is the mode-aware second channel
        of the bank-0 pair -- "Deflection" (contact) or "Amplitude" (AC) -- exactly the
        name the driver armed via SCANCHAIN, so the host demux maps it to the error grid.
        `done_delay` paces the lines so an ABORT can land mid-chain (as it would on real
        Igor). Abort -> a single SCANDONE -1, matching AFM_Abort's terminal on a live
        chain."""
        for ln in range(int(nlines)):
            if self._sc_abort.is_set() or self._stop.is_set():
                self._send(link, P.msg_scandone(uid, -1))
                return
            lo, hi = ln * ppl, (ln + 1) * ppl
            self._bank_arm_scan_line(ppl, err_chan, out_ok=out_ok, interp=interp,
                                     xy=self._slice3((self.waves.get(xname), self.waves.get(yname)),
                                                     lo, hi))
            self.tap.record_pseudo("@FILL")
            z = self._capture_channel(f"L{ln}", _HEIGHT, ppl)
            d = self._capture_channel(f"L{ln}", err_chan, ppl)
            self._send(link, P.msg_capture(uid, f"L{ln}", _HEIGHT, z))
            self._send(link, P.msg_capture(uid, f"L{ln}", err_chan, d))
            if self.done_delay:
                clock.sleep(self.done_delay)
        self._send(link, P.msg_scandone(uid, int(nlines)))

    # ---- controller limits + bank model ----
    @property
    def stale_plays(self) -> list[tuple[str, int, str]]:
        """(event, out_bank, waves) for every out-bank that played on a fire while bound
        by an earlier pass."""
        return [(f["event"], b, w) for f in self.bank_fires for (b, w) in f["stale_out"]]

    def _touch(self) -> None:
        """A bind after a fire starts a new pass: its bindings carry a new epoch."""
        with self._bank_lock:
            if self._fired:
                self._epoch += 1
                self._fired = False

    def _violate(self, rule: str, verb: str, bank=None, length=None) -> bool:
        self.violations.append(Violation(rule, verb, bank, length))
        return True

    def _check_bank(self, verb: str, bank: int, pair: bool) -> bool:
        if bank < 0:
            return self._violate("bank_below_0", verb, bank)
        if bank > MAX_BANK:
            return self._violate("bank_above_2", verb, bank)
        if pair and bank == 2:
            return self._violate("pair_on_bank_2", verb, bank)
        return False

    def _check_out_len(self, verb: str, bank: int, n: int) -> bool:
        if n > MAX_OUT_WAVE_POINTS:
            return self._violate("out_wave_over_cap", verb, bank, n)
        return False

    def _check_in_len(self, verb: str, bank: int, n: int) -> bool:
        """The controller needs every in-wave a multiple of 32, on every in-bind."""
        if n % IN_WAVE_MULTIPLE == 0:
            return False
        self.violations.append(Violation("in_wave_not_multiple_of_32", verb, bank, n))
        return True

    def _check_xy_split(self, verb: str, bank: int, channels) -> bool:
        """XSensor and YSensor must share one in-bank within a pass."""
        self._touch()
        with self._bank_lock:
            for c in channels:
                if c not in _XY_SENSORS:
                    continue
                partner = _XY_SENSORS[1 - _XY_SENSORS.index(c)]
                if partner in channels:
                    continue
                for b, rec in self.in_banks.items():
                    if (b != bank and rec["epoch"] == self._epoch
                            and partner in rec["channels"]):
                        return self._violate("xy_sensors_split", verb, bank)
        return False

    def _release_out(self, *banks: int) -> None:
        with self._bank_lock:
            for b in banks:
                self.out_banks.pop(b, None)

    def _release_in(self, *banks: int) -> None:
        with self._bank_lock:
            for b in banks:
                self.in_banks.pop(b, None)

    def _bind_out(self, bank, event, channels, waves, interp=None, data=None) -> None:
        """`data`: the wave samples the safety simulation plays; `interp`: ticks per point."""
        self._touch()
        with self._bank_lock:
            self.out_banks[bank] = {"event": str(event), "channels": list(channels),
                                    "waves": list(waves), "epoch": self._epoch,
                                    "interp": interp, "data": data}

    def _slice3(self, waves, lo: int, hi: int):
        """The [lo:hi] part of each wave in `waves` (None if no safety simulation, or no waves)."""
        if not self.safety_sim or waves is None:
            return None
        return tuple(list(w[lo:hi]) if w is not None else [] for w in waves)

    def _bind_in(self, bank, event, channels, cap_len) -> None:
        self._touch()
        with self._bank_lock:
            self.in_banks[bank] = {"event": str(event), "channels": list(channels),
                                   "cap_len": int(cap_len), "epoch": self._epoch}

    def _play(self, name: str, event: str) -> None:
        """Firing Event.N plays every bank still bound to N."""
        self._fired = True
        self.pass_armed = True                      # root:AFM_passArmed, set at every fire site
        with self._bank_lock:
            ev = str(event)
            out = [(b, r) for b, r in sorted(self.out_banks.items()) if r["event"] == ev]
            inn = [(b, r) for b, r in sorted(self.in_banks.items()) if r["event"] == ev]
            self.bank_fires.append({
                "name": name, "event": ev, "epoch": self._epoch,
                "out": [(b, list(r["channels"]), list(r["waves"])) for b, r in out],
                "in": [(b, list(r["channels"])) for b, r in inn],
                "stale_out": [(b, ",".join(r["waves"])) for b, r in out
                              if r["epoch"] != self._epoch],
            })
            if self.safety_sim:
                self.safety_sim.play([(r["channels"], r["data"], r["interp"]) for _, r in out])

    def _bank_arm_litho(self, name: str, cap: int, event: str = "14",
                        out_ok: tuple = (True, True), in_ok: bool = True,
                        interp=None, xyv=None) -> None:
        """AFM_ArmAndFireLitho / AFM_LithoArmChunk: release the banks, bind, fire.
        `xyv`: the (X, Y, V) samples being bound, for the safety simulation."""
        x, y, v = xyv or (None, None, None)
        with self._bank_lock:
            self._release_out(0, 2)
            self._release_in(0, 1, 2)
            if out_ok[0]:
                self._bind_out(0, event, ("PIDSLoop.0.Setpoint", "PIDSLoop.1.Setpoint"),
                               ("XLitho", "YLitho"), interp, (x, y))
            if out_ok[1]:
                self._bind_out(2, event, ("Output.A",), ("VLitho",), interp, (v,))
            if in_ok:
                self._bind_in(0, event, ("ZSensor", "Deflection"), cap)
                self._bind_in(1, event, ("XSensor", "YSensor"), cap)
                self._bind_in(2, event, ("UserIn2", "Lateral") if self.lateral_armed
                              else ("UserIn2",), cap)
            self._play(name, event)

    def _bank_arm_scan_line(self, cap: int, err_chan: str, event: str = "14",
                            out_ok: bool = True, interp=None, xy=None) -> None:
        """AFM_ScanArmLine: releases out 0 and in 0/1 only (never out 2), binds, fires."""
        with self._bank_lock:
            self._release_out(0)
            self._release_in(0, 1)
            if out_ok:
                self._bind_out(0, event, ("PIDSLoop.0.Setpoint", "PIDSLoop.1.Setpoint"),
                               ("AFM_scLX", "AFM_scLY"), interp, xy)
            self._bind_in(0, event, (_HEIGHT, err_chan), cap)
            self._bind_in(1, event, ("UserIn2",), cap)
            self._play("scan", event)

    def _record(self, line: str, verb: str) -> None:
        """Record one received line. A hook, so a harness can route recording."""
        self.transcript.append(verb)
        self.tap.record(line)

    def _dispatch(self, line: str, link) -> None:
        parts = line.split(" ")
        verb = parts[0]
        if verb in V3_VERBS and len(parts) > 1:       # v2 CLEARFAULT has no seq                # wire v3: igor_stub_v3.py translates to the v2 lines
            return self.v3.handle(parts, link)
        self._record(line, verb)
        if self.safety_sim:
            self.safety_sim.sync()
        match verb:
            case P.STARTUP:
                # STARTUP <uid> <version>: echo the uid + our proto for the handshake.
                uid = parts[1] if len(parts) > 1 else ""
                self._send(link, P.msg_startup(uid, startup_string(self.values)))
            case P.GETVALUE:
                # GETVALUE <uid> <name>
                if self.stall_getvalue:
                    return                       # wedged Igor: no reply, ever
                uid = parts[1]
                v = self.values.get(parts[2], self._loop_status(parts[2]))
                self._send(link, P.msg_value(uid, float(v)))
            case P.GETSTR:
                # GETSTR <uid> <name>: only PIDSLoop.<n>.OutputChannel of a running loop is
                # known. td_ReadString answers "*ERROR*" for anything else; the .ipf turns
                # that into its ERR (AFM_DispatchOther). `getstr_passes_error_text` models an
                # older .ipf that passed the "*ERROR*" text on as the value.
                uid, name = parts[1], parts[2]
                loop = _pid_loop_param(name, "OutputChannel")
                if loop in self.pid_loops:
                    self._send(link, P.msg_xpt(uid, {name: self.pid_loops[loop]}))
                elif self.getstr_passes_error_text:
                    self._send(link, P.msg_xpt(uid, {name: "*ERROR*"}))
                else:
                    self._send(link, f"{P.ERR} {uid} exception: td_ReadString({name}) failed")
            case P.GETGV:
                # GETGV <uid> <name>: the GV table is the same value store; an
                # unknown global answers nan, as AFM_GetGVTok does on a GV error.
                self.getgvs.append(parts[2])
                if self.stall_getvalue:
                    return                       # wedged Igor answers no read at all
                v = self.values.get(parts[2], float("nan"))
                self._send(link, P.msg_value(parts[1], float(v)))
            case P.SETVALUE:
                self.sets.append((parts[1], float(parts[2])))
                if self.safety_sim:
                    self.safety_sim.write_value(parts[1], float(parts[2]))
            case P.SERVOLOOP:
                self.loops.append((int(parts[1]), parts[2], float(parts[3]),
                                   float(parts[4]), float(parts[5]), parts[6]))
                self.pid_loops[int(parts[1])] = parts[6]
                if int(parts[1]) == 2 and parts[2] == "Deflection":
                    self._deflection_follows(float(parts[3]))
                # trailing outMin/outMax are OPTIONAL -- default -10/150 when the
                # line omits them (backward compat, mirrors AFM_Dispatch).
                if len(parts) >= 9:
                    self.loop_limits.append((float(parts[7]), float(parts[8])))
                else:
                    self.loop_limits.append((-10.0, 150.0))
                if self.safety_sim:
                    self.safety_sim.arm_loop(self.loops[-1][0], parts[2], float(parts[3]),
                                          parts[6], *self.loop_limits[-1])
            case P.STOPLOOP:
                self.stopped_loops.append(int(parts[1]))
                self.pid_loops.pop(int(parts[1]), None)
                if self.safety_sim:
                    self.safety_sim.stop_loop(int(parts[1]))
            case P.RAMP:
                # RAMP <channel> <dest> <rate> <max_seconds>
                self.ramps.append((parts[1], float(parts[2]), float(parts[3]),
                                   float(parts[4])))
                self.pass_armed = True              # AFM_RampOutput sets root:AFM_passArmed
                if self.safety_sim:
                    self.safety_sim.ramp(*self.ramps[-1])
            case P.WAVE:
                self.waves[parts[1]] = [float(x) for x in parts[2].split(";") if x != ""]
            case P.WAVEAPP:                          # chunked big-wave load: append
                self.waves.setdefault(parts[1], []).extend(
                    float(x) for x in parts[2].split(";") if x != "")
            case P.ARMFIRE:
                # ARMFIRE <uid> <name> <interp> <capLen> <decim>
                uid = parts[1]
                name = parts[2]
                self._cur = name
                self._cur_uid = uid
                cap_len = int(parts[4])
                in_bad = self._check_in_len(verb, 0, cap_len)
                xy_bad = sum(self._check_out_len(verb, 0, len(self.waves.get(wn, ())))
                             for wn in ("XLitho", "YLitho")) > 0
                v_bad = self._check_out_len(verb, 2, len(self.waves.get("VLitho", ())))
                self._bank_arm_litho(
                    name, cap_len, out_ok=(not xy_bad, not v_bad), in_ok=not in_bad,
                    interp=float(parts[3]),
                    xyv=tuple(self.waves.get(w) for w in ("XLitho", "YLitho", "VLitho")))
                self.arms.append((name, float(parts[3]), cap_len, int(parts[5])))
                # fire completion asynchronously so an ABORT can still arrive
                # mid-write. Completion streams CAPTURE frames, then DONE.
                if in_bad:
                    return          # bank 0 callback never registered: no capture, no DONE
                self._timer = clock.timer(
                    self._fill_delay(cap_len, int(parts[5])),
                    lambda: self._complete(link, uid, name, cap_len))
                self._timer.start()
            case P.OUTWAVE:
                # OUTWAVE <bank> <event> <channel> <name> <interp>
                bank = int(parts[1])
                self.out_binds.append(
                    (bank, parts[2], parts[3], parts[4], float(parts[5])))
                bad = any([self._check_bank(verb, bank, False),
                           self._check_out_len(verb, bank, len(self.waves.get(parts[4], ())))])
                if not bad:
                    self._bind_out(bank, parts[2], [parts[3]], [parts[4]],
                                   float(parts[5]), (self.waves.get(parts[4]),))
            case P.OUTPAIR:
                # OUTPAIR <bank> <event> <chA> <nameA> <chB> <nameB> <interp>
                bank, event = int(parts[1]), parts[2]
                interp = float(parts[7])
                self.out_binds.append((bank, event, parts[3], parts[4], interp))
                self.out_binds.append((bank, event, parts[5], parts[6], interp))
                bad = any([self._check_bank(verb, bank, True)]
                          + [self._check_out_len(verb, bank, len(self.waves.get(wn, ())))
                             for wn in (parts[4], parts[6])])
                if bad:
                    return          # silent, as Igor: the pair is not bound
                self._bind_out(bank, event, [parts[3], parts[5]], [parts[4], parts[6]],
                               interp, (self.waves.get(parts[4]), self.waves.get(parts[6])))
                # xy_transit's one out-wave-pair move (XY_SLEW_SPEC.md) verifies arrival by an
                # XSensor/YSensor read. `values` is static, so land the readback on the wave's
                # last point (a perfectly-tracking servo), at once or after sensor_lag_s.
                # Litho strokes go through ARMFIRE/LITHOCHAIN and never reach this.
                if parts[3] == "PIDSLoop.0.Setpoint" and parts[5] == "PIDSLoop.1.Setpoint":
                    xs, ys = self.waves.get(parts[4]), self.waves.get(parts[6])
                    self._xy_transit_wave = {"XSensor": xs or [], "YSensor": ys or []}
                    if self.safety_sim:
                        return      # the pair plays at FIRE and the sensors follow it
                    if self.sensor_lag_s > 0:
                        # sensor-lag injection: land the readback only after
                        # sensor_lag_s clock seconds (real, or virtual on the testing
                        # clock), not synchronously with the OUTPAIR bind (see sensor_lag_s in __init__).
                        def _land(xs=xs, ys=ys):
                            if xs:
                                self.values["XSensor"] = xs[-1]
                            if ys:
                                self.values["YSensor"] = ys[-1]
                        t = clock.timer(self.sensor_lag_s, _land)
                        t.daemon = True
                        t.start()
                    else:
                        if xs:
                            self.values["XSensor"] = xs[-1]
                        if ys:
                            self.values["YSensor"] = ys[-1]
            case P.INWAVE:
                # INWAVE <bank> <event> <channel> <capLen> <decim>
                rec = (int(parts[1]), parts[2], parts[3], int(parts[4]), int(parts[5]))
                self.in_channels.append(rec)
                self._armed_in.append(rec)
                bad = any([self._check_bank(verb, rec[0], False),
                           self._check_in_len(verb, rec[0], rec[3]),
                           self._check_xy_split(verb, rec[0], [rec[2]])])
                if not bad:
                    self._bind_in(rec[0], rec[1], [rec[2]], rec[3])
                    self._ok_in_binds += 1
            case P.INPAIR:
                # INPAIR <bank> <event> <chA> <chB> <capLen> <decim>
                bank, event = int(parts[1]), parts[2]
                cap_len, decim = int(parts[5]), int(parts[6])
                for chan in (parts[3], parts[4]):
                    rec = (bank, event, chan, cap_len, decim)
                    self.in_channels.append(rec)
                    self._armed_in.append(rec)
                bad = any([self._check_bank(verb, bank, True),
                           self._check_in_len(verb, bank, cap_len),
                           self._check_xy_split(verb, bank, [parts[3], parts[4]])])
                if not bad:
                    self._bind_in(bank, event, [parts[3], parts[4]], cap_len)
                    self._ok_in_binds += 1
            case P.FIRE:
                # FIRE <uid> <name> <event>.
                uid = parts[1]
                name = parts[2]
                self._cur = name
                self._cur_uid = uid
                self._play(name, parts[3])
                self.fires.append((name, parts[3]))
                armed = self._armed_in            # snapshot the channels armed so far
                self._armed_in = []               # reset for the next pass
                ok_in, self._ok_in_binds = self._ok_in_binds, 0
                if armed and not ok_in:
                    return          # every in-bind failed: no callback, so no DONE (silent)
                if not armed:
                    # AFM_FireEvent (afm_litho_run.ipf:491-506): with no input bank armed it
                    # sends the terminal itself, synchronously.
                    self._send(link, P.msg_done(uid, name))
                else:
                    # async so ABORT can cancel; DONE waits for done_delay, or sensor_lag_s
                    # when XSensor/YSensor are armed.
                    delay = self.done_delay
                    if (self.sensor_lag_gates_done and self.sensor_lag_s > 0
                            and any(c[2] in ("XSensor", "YSensor") for c in armed)):
                        delay = max(delay, self.sensor_lag_s)
                    # reuse the same _timer slot as ARMFIRE so ABORT cancels it too
                    self._timer = clock.timer(
                        delay, lambda: self._complete_fire(link, uid, name, armed))
                    self._timer.start()
            case P.SCANCHAIN:
                # SCANCHAIN <uid> <xname> <yname> <ppl> <lines> <interp> <decim> [<errChan>].
                # Run the per-line callback in a thread so ABORT can interrupt it (Igor's
                # callback is likewise asynchronous to the dispatch loop). The optional
                # trailing <errChan> is the mode-aware bank-0 error channel (Deflection|
                # Amplitude); absent -> Deflection (contact), for a pre-field caller.
                uid = parts[1]
                xname, yname = parts[2], parts[3]
                ppl, nlines = int(parts[4]), int(parts[5])
                err_chan = parts[8] if len(parts) > 8 else _ERR
                self._last_scanchain_err = err_chan
                self._cur = ""                    # scanchain's terminal is SCANDONE, not ABORTED
                self._cur_uid = uid
                self.scanchains.append((xname, yname, ppl, nlines,
                                        float(parts[6]), int(parts[7])))
                if self.scanchain_err is not None:
                    # deterministic router/bridge fault: reply ERR instead of running
                    # the chain (the chained driver surfaces it as summary["error"]).
                    self._send(link, P.msg_err(uid, self.scanchain_err))
                elif self._check_in_len(verb, 0, ppl):
                    pass            # bank 0 callback never registered: nothing advances
                else:
                    out_ok = not self._check_out_len(verb, 0, ppl)
                    self._sc_abort.clear()
                    self._sc_thread = threading.Thread(
                        target=self._run_scanchain,
                        args=(link, uid, xname, yname, ppl, nlines, err_chan, out_ok,
                              float(parts[6])),
                        daemon=True)
                    self._sc_thread.start()
            case P.LITHOCHAIN:
                # LITHOCHAIN <uid> <name> <xname> <yname> <vname> <boundsName> <interp> <decim>
                # An oversize object fired as a chained continuation. Stream per-chunk
                # captures (accumulated by the driver under the one name) then DONE,
                # async so an ABORT can still cancel it mid-chain.
                uid = parts[1]
                name = parts[2]
                self._cur = name
                self._cur_uid = uid
                bname = parts[6]
                self.lithochains.append((name, parts[3], parts[4], parts[5], bname,
                                         float(parts[7]), int(parts[8])))
                bounds = [int(round(b)) for b in self.waves.get(bname, [])]
                bad_chunks = {k for k in range(len(bounds) - 1)
                              if self._check_out_len(verb, 0, bounds[k + 1] - bounds[k])}
                xyv = tuple(self.waves.get(w) for w in parts[3:6])
                interp = float(parts[7])
                if len(bounds) > 1:
                    m = bounds[1] - bounds[0]
                    self._bank_arm_litho(name, m + (32 - m % 32) % 32,
                                         out_ok=(0 not in bad_chunks,) * 2, interp=interp,
                                         xyv=self._slice3(xyv, bounds[0], bounds[1]))
                self._timer = clock.timer(
                    self._fill_delay(bounds[-1] if bounds else 0, int(parts[8])),
                    lambda: self._complete_lithochain(link, uid, name, bounds, bad_chunks,
                                                      xyv, interp))
                self._timer.start()
            case P.SHOWTIP:
                # DISPLAY hint only. Fire-and-forget (no reply), like real Igor's
                # AFM_ShowTip -- which NO LONGER starts/stops the always-on telemetry
                # poll (that runs from bridge startup); it only tells the host whether
                # to draw the tip marker. Recorded for assertions. The TELEM stream is
                # driven autonomously by _telem_loop (telem_hz), NOT by this toggle.
                self.showtips.append(len(parts) > 1 and parts[1].upper() == "ON")
            case P.SETMODE:
                # SETMODE <Contact|AC>: model the vendor MainPopupFunc mode switch.
                # Fire-and-forget (no reply), like SHOWTIP -- just record the vendor
                # mode token the driver sent so a test can assert the mapping.
                self.modes.append(parts[1] if len(parts) > 1 else "")
            case P.READXPT:
                # READXPT <uid> <group>: read-only crosspoint read-back. Reply the
                # configured crosspoint as an XPT <uid> entry=value; payload (empty when
                # unset -> the driver reads {} -> "unknown"). Read-only on the wire.
                uid = parts[1] if len(parts) > 1 else ""
                self._send(link, P.msg_xpt(uid, self.crosspoint))
            case P.ABORT:
                # ABORT runs AFM_SafePark, the one abort function: td_Stop once per abort,
                # bias 0, loop 2 re-armed at -10 V, the 5 s backup armed; then exactly ONE
                # terminal for the in-flight pass, tagged with THAT pass's uid so the router
                # routes it to the right waiter: a running SCANCHAIN ends itself with
                # SCANDONE <uid> -1 (its own thread), so here we only signal it; any other
                # in-flight pass (single fire / litho chain) gets one ABORTED <uid> <name>.
                # AFM_Abort bars the scan chain FIRST (root:AFM_scAbort), before anything
                # a held chain could see: it wakes on `aborts`.
                sc_running = (self._sc_thread is not None
                              and self._sc_thread.is_alive())
                self._sc_abort.set()          # a live chain ends at its next line
                self.aborts += 1
                self.safe_parks += 1
                self._abort_function()
                if not self.safety_sim and self.safe_park_zeros and (
                        self.safe_parks > self.safe_park_zero_after):
                    def _land():
                        # Off the safety simulation the readbacks are static: the re-arm's
                        # retract lands them at 0 (Height, ZSensor, Deflection), at once or
                        # after `safe_park_ramp_s` (a too-early host read still sees them up).
                        self.values["height"] = 0.0
                        self.values["Height"] = 0.0
                        self.values["Output.Z"] = 0.0
                        self.values["ZSensor"] = 0.0
                        self.values["Deflection"] = 0.0
                    if self.safe_park_ramp_s > 0:
                        t = clock.timer(self.safe_park_ramp_s, _land)
                        t.daemon = True
                        t.start()
                    else:
                        _land()
                if self._timer:
                    self._timer.cancel()
                if not sc_running:
                    self._send(link, P.msg_aborted(self._cur_uid, self._cur))
            case P.ABORTCONFIRM:
                # AFM_AbortConfirmed: the host read Height < 1 V; the backup stands down.
                self.abort_confirms += 1
                self.abort_stopped = False
                self._cancel_backup()
            case P.ABORTFAULT:
                # ABORTFAULT <uid>: Igor's fault flag, read only (AFM_ReadAbortFault).
                uid = parts[1] if len(parts) > 1 else ""
                self._send(link, P.msg_value(uid, 1.0 if self.abort_fault else 0.0))
            case P.CLEARFAULT:
                # AFM_ClearFault: the operator cleared the fault; both flags back to 0.
                self.fault_clears += 1
                self.abort_stopped = False
                self.abort_fault = False
                self._cancel_backup()
            case _:
                self._send(link, P.msg_err("", f"unknown verb: {verb}"))
