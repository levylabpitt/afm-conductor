"""The software ARC: Asylum controller emulated at the td_ command level.

Interface fidelity is the contract: the same call names, the same arming
semantics (waves bound to events, fired together, running autonomously),
the same parameter names the annotated captures use. Physical fidelity is
provided by twin.physics behind the boundary.

The clock is the controller's 50 kHz sample clock. `step(n)` advances n
samples; a service loop calls it continuously (faster than real time if
asked). All armed out-waves write, all armed in-waves record, servo loops
run, every sample, exactly once.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass

import numpy as np

from ..litho.chunker import check_outwave_len
from .physics import World

CLOCK_HZ = 50_000
DT = 1.0 / CLOCK_HZ

# Vendor MicroscopeID for a Cypher on the hardware side is read from STARTUP
# (hardware.hardware_arc.CYPHER_MICROSCOPE_ID); the twin has no bridge to ask,
# so its platform is selected by this env var instead (Sec 6/10). Default
# mfp3d keeps every twin test that predates this platform concept unchanged.
_PLATFORM_ENV = "AFM_LITHO_TWIN_PLATFORM"
_PLATFORMS = ("cypher", "mfp3d")

# The twin's one physical Z-drive line, moved to the withdrawn/free-air value
# by `vendor_withdraw` -- same rail-out value mfp3d's ambient default already
# used (`out_min=-10.0` everywhere loop 2 is armed).
_WITHDRAW_V = -10.0


@dataclass
class _Faults:
    """Fault injection for twin tests (Sec 10). `ramp_noop=True` makes every
    `SoftwareARC.ramp()` and `abort_retract()` leave Z where it is (still logged) --
    the twin's stand-in for a retract that did not happen, so `safe_park`'s
    vendor-fallback path has something real to fall back FROM."""

    ramp_noop: bool = False
    # Fourth-slice interlock fault injection (docs/INSTRUMENT_MODEL_SPEC.md 12.1).
    # Each drives a distinct runtime interlock or the free-air read, so a twin test
    # can prove the guard trips + safe_parks without touching real hardware:
    free_air_jitter: bool = False   # first three free-air reads disagree, next three agree
    z_rail_stuck: bool = False      # read('Output.Z'/'Height') reports 148 V regardless of ramps
    sum_drop: bool = False          # DetectorSum reads 0.1 V (SUM collapse)
    defl_over_sum: bool = False     # |Deflection| drops above SUM (deflection integrity)


@dataclass
class _OutChannel:
    target: str
    wave: np.ndarray


@dataclass
class _OutBank:
    channels: list[_OutChannel]
    ticks_per_point: int
    event: str
    active: bool = False
    tick: int = 0

    def value(self, ch: int) -> float:
        w = self.channels[ch].wave
        pos = self.tick / self.ticks_per_point
        i = int(pos)
        if i >= len(w) - 1:
            return float(w[-1])
        frac = pos - i
        return float(w[i] * (1 - frac) + w[i + 1] * frac)

    @property
    def done(self) -> bool:
        # done only after the final point has been written (tick has moved past it)
        return self.tick > (len(self.channels[0].wave) - 1) * self.ticks_per_point


@dataclass
class _InBank:
    channels: list[str]
    buffers: list[np.ndarray]
    ticks_per_sample: int
    event: str
    active: bool = False
    tick: int = 0
    pos: int = 0

    @property
    def done(self) -> bool:
        return self.pos >= len(self.buffers[0])


@dataclass
class _PISLoop:
    input: str = ""
    output: str = ""
    pgain: float = 0.0
    igain: float = 0.0
    sgain: float = 0.0
    out_min: float = -10.0
    out_max: float = 150.0
    running: bool = False
    integrator: float = 0.0


class SoftwareARC:
    """One instance = one controller. Not thread-safe; drive from one loop."""

    def __init__(self, world: World, seed: int = 7):
        self.world = world
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.params: dict[str, float] = {}
        self.strings: dict[str, str] = {}
        # Platform id (Sec 6/10), env-selected, mirrors HardwareARC.is_cypher /
        # .z_out_channel naming so a caller (pass_executor, safe_park) can treat
        # both backends the same way via `getattr(arc, "z_out_channel", ...)`.
        platform = str(os.environ.get(_PLATFORM_ENV, "mfp3d")).strip().lower()
        # A non-empty, unrecognised platform is an OPERATOR ERROR, not a reason to
        # silently drive the wrong Z channel -- raise. An empty/unset value keeps
        # the historical mfp3d default (every twin test predating platforms).
        if platform and platform not in _PLATFORMS:
            raise ValueError(
                f"{_PLATFORM_ENV}={platform!r} is not a known platform "
                f"{_PLATFORMS}")
        self.platform: str = platform if platform in _PLATFORMS else "mfp3d"
        # SoftwareARC is single-threaded, but on the twin backend the safety
        # watchdog/park path (safe_park -> ramp/vendor_withdraw/read) and the
        # always-on interlock feed run from the event loop / a foreign thread while
        # a scan or litho worker thread is inside step(). Serialise every physical
        # mutation/read through ONE reentrant lock so a park never tears a step().
        self._lock = threading.RLock()
        self.command_log: list[tuple[str, float]] = []
        self.faults = _Faults()
        self.gv: dict[str, float | str] = {
            "XLVDTSens": world.x_lvdt_sens,
            "YLVDTSens": world.y_lvdt_sens,
            "ZLVDTSens": world.z_lvdt_sens,
            "XLVDTOffset": world.x_lvdt_offset,
            "YLVDTOffset": world.y_lvdt_offset,
            "ZLVDTOffset": world.z_lvdt_offset,
            "ImagingMode": 1.0,  # index; ImageModeList maps it
            "Amplitude": world.lever.free_amplitude_v,
        }
        self.loops: dict[int, _PISLoop] = {i: _PISLoop() for i in range(6)}
        self.out_banks: dict[int, _OutBank] = {}
        self.in_banks: dict[int, _InBank] = {}
        self.completed: list[tuple[str, int]] = []   # ("out"|"in", bank)
        self.control_actions: list[tuple] = []       # GUI procs fired over the bridge
        # live signal cache (updated every block, last-sample values)
        self.sig: dict[str, float] = {
            "XSensor": 0.0, "YSensor": 0.0, "ZSensor": 0.0,
            "Input.Fast": 0.0, "Input.A": 0.0, "Lockin.0.R": world.lever.free_amplitude_v,
            "Output.A": 0.0, "Output.X": 0.0, "Output.Y": 0.0, "Z%Output": -10.0,
        }
        if self.is_cypher:
            # The Cypher's real Z-drive line, "Height" -- kept at the resting
            # ZSensor value (0.0) so nothing ambiently pulls Z before any ramp
            # or armed loop 2 commands it (unlike mfp3d's legacy "Z%Output",
            # which has always defaulted to -10.0 and keeps doing so).
            self.sig["Height"] = 0.0
        self.params.setdefault("Output.A", 0.0)
        for i in range(6):
            self.params.setdefault(f"PIDSLoop.{i}.Setpoint", 0.0)

    # --------------------------------------------------------- platform id --

    @property
    def is_cypher(self) -> bool:
        """Same semantics as `HardwareARC.is_cypher`, env-selected here instead
        of read off a STARTUP payload."""
        return self.platform == "cypher"

    @property
    def z_out_channel(self) -> str:
        """The Z (loop 2) OUTPUT channel: "Height" on every platform, as
        `HardwareARC.z_out_channel` (hardware/hardware_arc.py)."""
        return "Height"

    @property
    def _z_ambient_key(self) -> str:
        """Which `self.sig` key holds the twin's one physical Z-drive signal.
        mfp3d keeps the original "Z%Output" key (so its ambient default and
        every existing mfp3d twin test are unchanged); a Cypher's physical
        line is "Height" instead, decoupling it from "Z%Output" entirely so a
        ramp aimed at the wrong channel is genuinely inert, not just unlogged."""
        return "Height" if self.is_cypher else "Z%Output"

    # ------------------------------------------------------------- td_ API --

    def GV(self, name: str):
        return self.gv.get(name, 0.0)

    def PV(self, name: str, value: float) -> None:
        """Write a MasterVariablesWave global (the panel parameters GV reads).
        The real PV also refreshes dependent UI; the twin only needs the value,
        which the scan/litho code reads back via GV when it runs."""
        self.gv[name] = float(value)

    def fire_control(self, control: str, proc: str = "",
                     var_num: float = 0.0, var_str: str = "") -> None:
        """A GUI control action arrived over the Igor bridge (a button press or
        a popup selection). The twin records it so callers and tests can see
        what was driven, and applies the actions that change state the hardware
        model actually tracks -- imaging mode being the important one."""
        self.control_actions.append((proc, control, var_num, var_str))
        if control == "ImagingModePopup" and var_str:
            idx = {"Contact": 0.0, "AC Mode": 1.0}.get(var_str)
            if idx is not None:
                self.gv["ImagingMode"] = idx

    def td_ReadValue(self, name: str) -> float:
        if name in self.sig:
            return self.sig[name]
        for i, lp in self.loops.items():
            if name == f"PIDSLoop.{i}.Setpoint":
                return self.params.get(name, 0.0)
            for g in ("PGain", "IGain", "SGain"):
                if name == f"PIDSLoop.{i}.{g}":
                    return getattr(lp, g.lower())
        return self.params.get(name, math.nan)

    def td_WriteValue(self, name: str, value: float) -> int:
        self.params[name] = float(value)
        if name == "Output.A":
            self.sig["Output.A"] = float(value)
        return 0

    def td_ReadString(self, name: str) -> str:
        return self.strings.get(name, "")

    def td_WriteString(self, name: str, value: str) -> int:
        self.strings[name] = value
        if name.startswith("Event.") and value.lower() in ("once", "set"):
            self._fire_event(name)
        return 0

    def ir_SetPISLoop(self, index: int, events: str, input_: str, setpoint: float,
                      pgain: float, igain: float, sgain: float, output: str,
                      out_min: float, out_max: float) -> int:
        lp = self.loops[int(index)]
        lp.input, lp.output = input_, output
        lp.pgain, lp.igain, lp.sgain = float(pgain), float(igain), float(sgain)
        lp.out_min, lp.out_max = float(out_min), float(out_max)
        lp.running = "always" in events.lower()
        if not math.isnan(setpoint):
            self.params[f"PIDSLoop.{index}.Setpoint"] = float(setpoint)
        return 0

    def ir_StopPISLoop(self, index: int) -> int:
        self.loops[int(index)].running = False
        return 0

    def td_xSetOutWave(self, bank: int, event: str, channel: str,
                       wave: np.ndarray, interpolation: int) -> int:
        return self._set_out(bank, event, [(channel, wave)], interpolation)

    def td_xSetOutWavePair(self, bank: int, event: str, ch_a: str, wave_a: np.ndarray,
                           ch_b: str, wave_b: np.ndarray, interpolation: int) -> int:
        return self._set_out(bank, event, [(ch_a, wave_a), (ch_b, wave_b)], interpolation)

    def _set_out(self, bank, event, pairs, interpolation) -> int:
        ticks = max(1, abs(int(interpolation)))
        # THE SAME PER-ARM CAP THE HARDWARE ENFORCES (litho/chunker.MAX_OUTWAVE_POINTS,
        # 87,000 -- the v19 vendor cMaxOutWave, not the v16 reference's 87,380). The twin
        # would happily play a wave of any length; the instrument silently TRUNCATES
        # one. Without this, a twin test passes where hardware raises -- which is the
        # one failure mode a digital twin must not have. Every pair is validated
        # BEFORE any bank is replaced, so a refusal leaves the bank as it was.
        checked = [(t, check_outwave_len(t, w)) for t, w in pairs]
        self.out_banks[int(bank)] = _OutBank(
            channels=[_OutChannel(t, w) for t, w in checked],
            ticks_per_point=ticks,
            event=_event_name(event),
        )
        return 0

    def td_xSetInWave(self, bank: int, event: str, channel: str,
                      n_points: int, decimation: int) -> int:
        return self._set_in(bank, event, [channel], n_points, decimation)

    def td_xSetInWavePair(self, bank: int, event: str, ch_a: str, ch_b: str,
                          n_points: int, decimation: int) -> int:
        return self._set_in(bank, event, [ch_a, ch_b], n_points, decimation)

    def _set_in(self, bank, event, channels, n_points, decimation) -> int:
        self.in_banks[int(bank)] = _InBank(
            channels=list(channels),
            buffers=[np.zeros(int(n_points), dtype=np.float32) for _ in channels],
            ticks_per_sample=max(1, abs(int(decimation))),
            event=_event_name(event),
        )
        return 0

    def td_StopOutWaveBank(self, bank: int) -> int:
        if bank == -1:
            for b in self.out_banks.values():
                b.active = False
        elif bank in self.out_banks:
            self.out_banks[bank].active = False
        return 0

    def td_StopInWaveBank(self, bank: int) -> int:
        if bank == -1:
            for b in self.in_banks.values():
                b.active = False
        elif bank in self.in_banks:
            self.in_banks[bank].active = False
        return 0

    def td_Stop(self) -> int:
        for b in self.out_banks.values():
            b.active = False
        for b in self.in_banks.values():
            b.active = False
        return 0

    # ------------------------------------------------------------ mechanics --

    def _fire_event(self, event: str):
        for b in self.out_banks.values():
            if b.event == event and not b.active:
                b.active = True
                b.tick = 0
        for b in self.in_banks.values():
            if b.event == event and not b.active:
                b.active = True
                b.tick = 0
                b.pos = 0

    def fire(self, event: str, ticks: int) -> dict[str, np.ndarray]:
        """Arm-and-fire seam: fire `event`, advance `ticks` samples, and return
        the filled input-wave capture as {channel: samples}. This is the one
        method a pass executor calls to run and read a pass -- SoftwareARC does
        it by stepping the sim; a bridge HardwareARC does it by FIRE + awaiting
        CAPTURE/DONE, ignoring `ticks` (the controller runs autonomously)."""
        self.td_WriteString(f"Event.{event}", "Once")
        if ticks > 0:
            self.step(int(ticks))
        out: dict[str, np.ndarray] = {}
        for b in self.in_banks.values():
            for chan, buf in zip(b.channels, b.buffers):
                out[chan] = np.asarray(buf, dtype=np.float64).copy()
        return out

    def ramp(self, channel: str, target_v: float) -> None:
        """Fire-and-forget Z ramp -- the twin's IR_SetRamp (Sec 10). Every call
        is logged (even a faulted or off-channel one, so `safe_park`'s
        first-failed-verify branch and any reader of `command_log` sees what
        was actually commanded), but only a ramp aimed at THIS platform's
        `z_out_channel` has any physical effect, and `faults.ramp_noop`
        suppresses that effect entirely (channel names it does not own are
        already inert by construction, not because of the fault)."""
        with self._lock:
            self.command_log.append((channel, float(target_v)))
            if self.faults.ramp_noop:
                return
            if channel == self.z_out_channel:
                self.sig[self._z_ambient_key] = float(target_v)
                # A successful Z retract also settles the INDEPENDENT verify
                # sensors safe_park reads: ZSensor tracks the commanded Z, and a
                # retracted tip's deflection returns to its free-air baseline, so a
                # sensor-verified park succeeds on the twin (z_rail_stuck lies only
                # about the Output.Z readback, not the true retract).
                self.sig["ZSensor"] = float(target_v)
                self.sig["Input.Fast"] = float(self.world.free_air_deflection())

    def read(self, channel: str) -> float:
        """Read one named sensor/output channel (Sec 10). "Deflection" aliases
        the twin's internal "Input.Fast"; "Height" and "Output.Z" both report
        the twin's one physical Z-drive signal.

        `faults.z_rail_stuck` (12.1) makes the Z OUTPUT readback report 148 V
        regardless of ramps -- the loop can still press on the true Z (ZSensor
        retracts), so a sensor-based verify still passes while an Output.Z-
        TRUSTING verifier is fooled: exactly the failure the twin must reproduce."""
        with self._lock:
            if channel in ("Height", "Output.Z") and self.faults.z_rail_stuck:
                return 148.0
            if channel == "Deflection":
                return float(self.sig["Input.Fast"])
            if channel in ("Height", "Output.Z"):
                return float(self.sig.get(self._z_ambient_key, 0.0))
            if channel in self.sig:
                return float(self.sig[channel])
            return float(self.td_ReadValue(channel))

    def abort_retract(self) -> None:
        """Igor's one abort function as the twin models it (afm_litho_run.ipf
        AFM_SafePark): stop the banks, bias to 0, and loop 2's -10 V re-arm drives Z to
        its 0 V minimum. The twin's loop has no Deflection input, so the retract is set
        directly. Logged as `("abort", 0.0)`; `faults.ramp_noop` leaves Z where it is."""
        with self._lock:
            self.td_Stop()
            self.td_WriteValue("Output.A", 0.0)
            self.loops[2].running = False
            self.command_log.append(("abort", 0.0))
            if self.faults.ramp_noop:
                return
            self.sig[self._z_ambient_key] = 0.0
            self.sig["ZSensor"] = 0.0
            self.sig["Input.Fast"] = float(self.world.free_air_deflection())

    def confirm_retract(self) -> None:
        """ABORTCONFIRM. The twin keeps no abort flag, so there is nothing to clear."""

    def vendor_withdraw(self) -> None:
        """Full retract via the vendor fallback path (Sec 10) -- the twin
        analogue of the hardware ARC's out-of-band retract. ALWAYS moves Z to
        the withdrawn value and stops loop 2, ignoring `faults.ramp_noop`:
        the fault only breaks the twin's OWN `ramp()`, and the entire point of
        a vendor fallback is that it still works when that primary path is
        broken. Recorded in `command_log` as `("vendor_withdraw", 0.0)` so a
        reader sees the fallback fired, distinct from an ordinary ramp."""
        with self._lock:
            self.sig[self._z_ambient_key] = _WITHDRAW_V
            if self.is_cypher:
                self.sig["Height"] = _WITHDRAW_V
            else:
                self.sig["Z%Output"] = _WITHDRAW_V
            # Settle the verify sensors to the retracted state too, so a vendor
            # fallback also passes a sensor verify (the tip really is off-surface).
            self.sig["ZSensor"] = float(_WITHDRAW_V)
            self.sig["Input.Fast"] = float(self.world.free_air_deflection())
            self.loops[2].running = False
            # Log the REAL target the fallback drove Z to, not a placeholder 0.0.
            self.command_log.append(("vendor_withdraw", float(_WITHDRAW_V)))

    def thermal(self, n_points: int, rate_hz: float = 5e6) -> np.ndarray:
        """5 MHz burst of Input.Fast: white detector noise + thermal peak at f0."""
        lever = self.world.lever
        freqs = np.fft.rfftfreq(n_points, 1.0 / rate_hz)
        f0, q = lever.f0, lever.Q
        resp = 1.0 / np.sqrt((1 - (freqs / f0) ** 2) ** 2 + (freqs / (f0 * q)) ** 2)
        amp = 4e-6 * resp + 1.2e-6
        phases = self.rng.uniform(0, 2 * np.pi, len(freqs))
        spec = amp * np.exp(1j * phases) * math.sqrt(n_points)
        spec[0] = 0
        return np.fft.irfft(spec, n=n_points).astype(np.float32)

    def sweep(self, f_start: float, f_stop: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Driven tune: amplitude (V) and phase (deg) vs frequency."""
        lever = self.world.lever
        f = np.linspace(f_start, f_stop, n)
        x = f / lever.f0
        denom = np.sqrt((1 - x ** 2) ** 2 + (x / lever.Q) ** 2)
        amp = self.params.get("Lockin.0.Amp", 0.1) * (lever.Q / 60.0) / np.maximum(denom, 1e-9) * 0.06
        phase = np.degrees(np.arctan2(x / lever.Q, 1 - x ** 2))
        return f, amp.astype(np.float64), phase

    def step(self, n: int):
        """Advance n controller samples: banks write, loops servo, banks record.

        Guarded by `self._lock` so the safety park/interlock paths cannot read or
        ramp mid-block from another thread (see __init__)."""
        with self._lock:
            self._step_locked(n)

    def _step_locked(self, n: int):
        # Per-sample loop with scalar math: honest sequencing, adequate speed.
        w = self.world
        rng = self.rng
        loops = self.loops
        sp = self.params

        # signals we integrate through the block
        x_out = self.sig["Output.X"]
        y_out = self.sig["Output.Y"]
        z_out = self.sig[self._z_ambient_key]
        defl = self.sig["Input.Fast"]
        amp = self.sig["Lockin.0.R"]

        ax = math.exp(-DT / w.scanner_tau)
        az = math.exp(-DT / w.z_tau)
        x_v, y_v, z_v = w.x_v, w.y_v, w.z_v
        invols = w.lever.invols
        free_v = w.lever.free_amplitude_v
        noise_defl = w.noise_defl_v
        half_noise_batch = rng.normal(0.0, 1.0, n)

        for i in range(n):
            # 1) armed out-banks write their targets this tick
            for b in self.out_banks.values():
                if not b.active:
                    continue
                for c, ch in enumerate(b.channels):
                    sp[ch.target] = b.value(c)
                    if ch.target == "Output.A":
                        self.sig["Output.A"] = sp[ch.target]
                b.tick += 1
                if b.done:
                    b.active = False
                    self.completed.append(("out", id(b)))
                    cb = self.strings.get("ARC.OutWave0StatusCallback", "")
                    if cb:
                        self.strings["_pending_callback"] = cb

            # 2) servo loops compute their outputs
            for idx, lp in loops.items():
                if not lp.running:
                    continue
                target = sp.get(f"PIDSLoop.{idx}.Setpoint", 0.0)
                match lp.input:
                    case "XSensor":
                        err = target - x_v
                    case "YSensor":
                        err = target - y_v
                    case s if s.startswith("Fast") or s == "Input.Fast":
                        err = target - defl
                    case s if s.startswith("Lockin"):
                        err = amp - target   # amplitude feedback: too little damping -> extend
                    case _:
                        err = 0.0
                lp.integrator += lp.igain * err * DT
                lp.integrator = min(max(lp.integrator, lp.out_min), lp.out_max)
                out = lp.integrator + lp.pgain * err
                out = min(max(out, lp.out_min), lp.out_max)
                match lp.output:
                    case "Output.X" | "X%Output":
                        x_out = out
                    case "Y%Output" | "Output.Y":
                        y_out = out
                    case s if s in ("Z%Output", self.z_out_channel):
                        # Loop 2 drives Z only when its OUTPUT is this platform's
                        # real Z channel (Output.Z on mfp3d, Height on cypher). A
                        # loop 2 armed on the wrong channel drives nothing -- rig
                        # behaviour (12.1 review). "Z%Output" stays accepted as the
                        # mfp3d legacy alias every existing twin test arms with.
                        z_out = out

            # XY loops with zero gains: the capture defines them gain-0 and the
            # real instrument tracks setpoints; the twin does the honest
            # equivalent: setpoint feeds the scanner command directly.
            lp0, lp1 = loops[0], loops[1]
            if lp0.running and lp0.igain == 0 and lp0.pgain == 0:
                x_out = sp.get("PIDSLoop.0.Setpoint", 0.0)
            if lp1.running and lp1.igain == 0 and lp1.pgain == 0:
                y_out = sp.get("PIDSLoop.1.Setpoint", 0.0)

            # 3) world: scanner dynamics + surface interaction (single sample)
            x_v = ax * x_v + (1 - ax) * x_out
            y_v = ax * y_v + (1 - ax) * y_out
            z_v = az * z_v + (1 - az) * z_out
            x_m = (x_v - w.x_lvdt_offset) * w.x_lvdt_sens
            y_m = (y_v - w.y_lvdt_offset) * w.y_lvdt_sens
            tip_h = -(z_v - w.z_lvdt_offset) * w.z_lvdt_sens
            surf = w.sample.height_scalar(x_m, y_m)
            indent = surf - tip_h
            defl = (indent / invols if indent > 0 else 0.0) + noise_defl * half_noise_batch[i]
            gap = tip_h - surf
            free_m = free_v * invols
            a = min(max(gap / free_m, 0.0), 1.0) if gap > 0 else 0.0
            amp = free_v * (a * (2 - a))

            # c-AFM writing: in contact with bias, deposit dose; tip current
            # reads the local written conductance (pen-is-probe)
            bias = self.sig["Output.A"]
            if indent > -5e-9 and abs(bias) >= 1.0:
                w.sample.deposit(x_m, y_m, bias, DT)
            g_loc = w.sample.conductance(x_m, y_m)
            self.sig["Input.A"] = g_loc * bias * 0.5 + 0.002 * half_noise_batch[i]

            # 4) armed in-banks record this tick
            for b in self.in_banks.values():
                if not b.active:
                    continue
                if b.tick % b.ticks_per_sample == 0 and b.pos < len(b.buffers[0]):
                    for c, ch in enumerate(b.channels):
                        b.buffers[c][b.pos] = self._signal_value(
                            ch, x_v, y_v, z_v, defl, amp, z_out)
                    b.pos += 1
                b.tick += 1
                if b.done and b.active:
                    b.active = False
                    self.completed.append(("in", id(b)))

        w.x_v, w.y_v, w.z_v = x_v, y_v, z_v
        w.engaged = bool(defl > 0.02 or (0 <= (tip_h - surf) < 5 * free_m))
        self.t += n * DT
        self.sig.update({
            "XSensor": x_v, "YSensor": y_v, "ZSensor": z_v,
            "Input.Fast": defl, "Lockin.0.R": amp,
            "Output.X": x_out, "Output.Y": y_out, "Z%Output": z_out,
        })
        if self.is_cypher:
            self.sig["Height"] = z_out

    def _signal_value(self, name, x_v, y_v, z_v, defl, amp, z_out) -> float:
        match name:
            case "XSensor":
                return x_v
            case "YSensor":
                return y_v
            case "ZSensor":
                return z_v
            case "Input.Fast" | "Fast%Input":
                return defl
            case s if s.startswith("Lockin"):
                return amp
            case "Z%Output" | "Output.Z":
                return z_out
            case _:
                return self.sig.get(name, self.params.get(name, 0.0))

    # convenience for the service/meters
    def snapshot(self) -> dict:
        w = self.world
        x_m, y_m = w.positions_m()
        return {
            "t": self.t,
            "x_um": x_m * 1e6,
            "y_um": y_m * 1e6,
            "z_um": -(w.z_v - w.z_lvdt_offset) * w.z_lvdt_sens * 1e6,
            "defl_v": self.sig["Input.Fast"],
            "amp_v": self.sig["Lockin.0.R"],
            "bias_v": self.sig["Output.A"],
            "engaged": w.engaged,
        }


def _event_name(event: str) -> str:
    e = str(event).strip()
    return e if e.startswith("Event.") else f"Event.{e}"
