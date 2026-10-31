"""Safe-state matrix harness: the golden rig on the fake-Igor safety simulation, plus a judge.

Safe means three things (DECISIONS, 2026-10-15): the tip is off the surface, the sample (XY) is
not moving, and the tip voltage is zero. The judge reads them as the program can, with GETVALUE
through the program's own bridge, never from the stub's internals.

The stub is the golden stub with the safety simulation on (docs/migration/slices/0.5b-stub-physics.md) and one
test-side addition: a servo loop 2 armed on a reachable setpoint drives the Z drive down onto the
surface (`SURFACE_ENGAGED_Z_V`), so an engaged tip is really down and "tip off the surface" can
fail. The stub's own model only holds Z.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from golden.harness import GoldenRig, GoldenStub, load_fixture

from afm_litho.hardware.hardware_arc import CYPHER_MICROSCOPE_ID
from afm_litho.hardware.igor_stub import (
    RETRACT_SETPOINT_V,
    Z_LOOP_SLEW_V_S,
    SafetySimulation,
    _toward,
)

# The stub puts the surface at a Z drive of 1.0 V (an assumption stated in the 0.5b report: a
# park goes to 0 V, an engaged tip sits above). Stated here as numbers, not imported.
SURFACE_V = 1.0
SURFACE_ENGAGED_Z_V = 70.0      # an engaged tip's Z drive: the park has to retract it
BIAS_TOL_V = 1e-6
XY_STEP_TOL_V = 1e-6
XY_LOOK_S = 0.25                 # simulated seconds between the two XY reads
POLL_S = 0.01
BIAS_ON_V = 0.01                 # Output.A above this counts as a voltage on the tip
XY_MOVE_LOOK_S = 0.15            # wall seconds between the two reads of "is XY moving"; an
                                 # out-wave sample lasts up to 62 ms, so 50 ms could miss a step
LAST_COMMAND_LANDS_S = 0.2       # after the program goes quiet, for the stub's last command
SEAM_HOLD_MAX_S = 15.0           # longest the stub holds a chain at its seam
QUIET_TIMEOUT_S = 30.0
ENGAGED_REACHES_SURFACE_S = 5.0
DEADMAN_S = 120                  # the control token's dead-man, within its 5 to 300 s clamp
WORKER_STOP_TIMEOUT_S = 15.0
_Z_DRIVES = ("Height", "Output.Z")
_DONE_STATES = ("idle", "parked", "engaged", "fault")


class SurfaceSafetySim(SafetySimulation):
    """SafetySimulation where an armed loop 2 on a reachable setpoint pushes Z onto the surface.
    `z_stuck` is a fault injection: the Z drive refuses to move, so a retract does not take."""

    z_stuck = False

    def _follow_z_drive(self) -> None:
        """The withdraw snapshot reads the lower-case vendor alias `height`
        (hw_approach.snapshot); on the instrument it moves with the Z drive."""
        super()._follow_z_drive()
        self.values["height"] = self.z_drive

    def advance(self, dt: float) -> None:
        stuck_at = self.z_drive if self.z_stuck else None
        super().advance(dt)
        if stuck_at is not None:
            self.values[self.z_drive_channel] = stuck_at
            self._follow_z_drive()

    def _run_z_loop(self, dt: float) -> None:
        loop = self.loop_state.get(2)
        if loop and loop["out"] in _Z_DRIVES and loop["setpoint"] > RETRACT_SETPOINT_V:
            out = loop["out"]
            self.values[out] = _toward(self.values[out], SURFACE_ENGAGED_Z_V,
                                       Z_LOOP_SLEW_V_S * dt)
            return
        super()._run_z_loop(dt)


class SafeStub(GoldenStub):
    """The golden stub on SurfaceSafetySim, with a hold at a chain seam.

    The real controller advances a chain by itself, so the host has no way to land a command
    exactly between two chunks. `hold_at_seam()` makes the stub stop there until the host's
    ABORT arrives; after that the chain is dead, as it is on the controller (td_Stop)."""

    def __init__(self, values: dict, height_ramp_after_stoploop: bool = False, **kw):
        super().__init__(values, safety_sim=True, safety_sim_real_time=True, **kw)
        self.safety_sim = SurfaceSafetySim(
            self.values, real_time=True,
            height_ramp_after_stoploop=height_ramp_after_stoploop)
        self.seam_reached = threading.Event()
        self._hold_seam = False
        self._released = False
        self._arms_in_chain = 0
        self._dead_chain: threading.Thread | None = None

    def hold_at_seam(self) -> None:
        self._hold_seam = True

    def let_one_scan_line_through(self) -> None:
        """A held scan delivers one more line, which the chain driver then tests for a stop."""
        self._release.set()

    def release_seam(self) -> None:
        """Let a held chain go (it then ends dead)."""
        self._released = True

    def _chain_is_dead(self) -> bool:
        return self._dead_chain is threading.current_thread()

    def _send(self, link, line: str) -> None:
        if not self._chain_is_dead():
            super()._send(link, line)

    def _safety_sim_zero_bias(self, why: str) -> None:
        if not self._chain_is_dead():
            super()._safety_sim_zero_bias(why)

    def _bank_arm_litho(self, name, cap, *a, **kw):
        self._arms_in_chain += 1
        if self._hold_seam and self._arms_in_chain >= 2:
            self.seam_reached.set()
            aborts = self.aborts
            end = time.monotonic() + SEAM_HOLD_MAX_S
            while self.aborts == aborts and not self._released and time.monotonic() < end:
                time.sleep(POLL_S)
            self._dead_chain = threading.current_thread()
            return
        if not self._chain_is_dead():
            super()._bank_arm_litho(name, cap, *a, **kw)


class SafeRig(GoldenRig):
    """GoldenRig on the fake-Igor safety simulation, as a Cypher (default) or an MFP-3D."""

    def __init__(self, hub, scenario: str, *, microscope_id: int = CYPHER_MICROSCOPE_ID,
                 client=None, hold_scan: bool = False, fill_scale: float | None = None,
                 telem_hz: float = 0.0, tip_at_volts: tuple[float, float] | None = None,
                 height_ramp_after_stoploop: bool = False):
        super().__init__(hub, scenario, telem_hz=telem_hz)
        self.client = client
        self.microscope_id = int(microscope_id)
        self.values = dict(load_fixture()["values"])
        if self.microscope_id == CYPHER_MICROSCOPE_ID:
            self.values["MicroscopeID"] = CYPHER_MICROSCOPE_ID
        if tip_at_volts is not None:
            # Start where the row's first move ends, so the program's pre-move transit is
            # a few seconds shorter and the row spends its time on what it is about.
            self.values["XSensor"], self.values["YSensor"] = tip_at_volts
        self.stub = SafeStub(self.values, microscope_id=self.microscope_id,
                             fill_scale=fill_scale, telem_hz=telem_hz,
                             height_ramp_after_stoploop=height_ramp_after_stoploop)
        if hold_scan:
            # The scan stays in flight after its first line until an abort, so a stop
            # always lands mid-scan (the stub no longer paces lines by itself).
            self.stub.scan_hold_line = 0
        self.stub.crosspoint = dict(load_fixture()["crosspoint"])
        self.tap = self.stub.tap

    @property
    def safety_sim(self):
        return self.stub.safety_sim

    def read(self, name: str) -> float:
        """GETVALUE through the program's own bridge."""
        return float(self.arc.td_ReadValue(name))

    def judge(self) -> Safe:
        return judge(self.read, self.stub.advance)

    def tip_is_down(self) -> bool:
        return self.read("Height") >= SURFACE_V

    def bias_is_on(self) -> bool:
        return abs(self.read("Output.A")) > BIAS_ON_V

    def xy_is_moving(self) -> bool:
        x0, y0 = self.read("XSensor"), self.read("YSensor")
        time.sleep(XY_MOVE_LOOK_S)
        return (abs(self.read("XSensor") - x0) > XY_STEP_TOL_V
                or abs(self.read("YSensor") - y0) > XY_STEP_TOL_V)

    def wait_for_actuation(self, verb: str, arg: str | None = None, timeout: float = 20.0) -> bool:
        """Block until the stub has received `verb` (with `arg` among its arguments)."""
        return wait_for(lambda: any(
            e["verb"] == verb and (arg is None or arg in e["args"] or arg in str(e["args"]))
            for e in list(self.tap.actuation)), timeout)

    def _start_from_a_fresh_hub(self) -> None:
        """A row that failed part-way can leave a run winding down or a park in force on the
        process-global hub; the golden rig refuses to start on that, so clear it first."""
        hub = self.hub
        wait_for(lambda: not self._worker_alive(), WORKER_STOP_TIMEOUT_S)
        hub.host._safe_park["active"] = False
        hub.fault = None

    def __enter__(self):
        self._start_from_a_fresh_hub()
        return super().__enter__()

    def __exit__(self, *exc) -> None:
        """Stop whatever a failed row left running, so it cannot touch the next rig's hub."""
        self.stub.release_seam()
        self.hub.host._hw_abort.set()
        self.hub.host._hw_litho_abort.set()
        wait_for(lambda: not self._worker_alive(), WORKER_STOP_TIMEOUT_S)
        super().__exit__(*exc)
        self.hub.host._hw_abort.clear()
        self.hub.host._hw_litho_abort.clear()

    def wait_quiet(self, timeout: float = QUIET_TIMEOUT_S) -> None:
        """Until no run is in flight and the state has settled, then a moment for the last
        stub command to land. A run that never winds down fails here, not silently later."""
        assert wait_for(self._quiet, timeout), (
            f"the program never went quiet within {timeout} s: state "
            f"{self.door.call('getState')}, worker alive {self._worker_alive()}")
        time.sleep(LAST_COMMAND_LANDS_S)

    def _worker_alive(self) -> bool:
        t = getattr(self.hub.host, "litho_thread", None)
        return bool(t is not None and t.is_alive())

    def _quiet(self) -> bool:
        return self.door.call("getState") in _DONE_STATES and not self._worker_alive()


def acquire(door) -> str:
    return door.call("acquireControl", {"client": "safe-state", "deadman_s": DEADMAN_S})["token"]


def engage(rig: SafeRig) -> str:
    """A real engage through the door; returns the control token. The tip ends down on the
    surface (loop 2 on a reachable setpoint), which the safe-state paths exist to undo."""
    tok = acquire(rig.door)
    rig.door.call("startApproach", {"token": tok, "setpoint": 0.2, "pgain": 0.0, "igain": 10.0,
                                    "settle_s": 0.2, "mode": "contact"})
    rig.door.wait_state("engaged")
    assert wait_for(rig.tip_is_down, ENGAGED_REACHES_SURFACE_S), "the engaged tip never reached the surface"
    return tok


@dataclass(frozen=True)
class Safe:
    tip_off: bool
    xy_at_rest: bool
    bias_zero: bool
    z_v: float
    bias_v: float

    @property
    def all(self) -> bool:
        return self.tip_off and self.xy_at_rest and self.bias_zero

    def __str__(self) -> str:
        return (f"tip_off={self.tip_off} (Height {self.z_v:.3f} V) "
                f"xy_at_rest={self.xy_at_rest} bias_zero={self.bias_zero} "
                f"(Output.A {self.bias_v:.3f} V)")


def judge(read, advance) -> Safe:
    """The three conditions from reads alone. `read(name)` is a GETVALUE; `advance(s)` lets
    time pass. The tip is off when Height and ZSensor are both below the surface level; XY is
    at rest when XSensor and YSensor are unchanged either side of the wait."""
    height, zsensor = read("Height"), read("ZSensor")
    x0, y0 = read("XSensor"), read("YSensor")
    advance(XY_LOOK_S)
    x1, y1 = read("XSensor"), read("YSensor")
    bias = read("Output.A")
    return Safe(
        tip_off=height < SURFACE_V and zsensor < SURFACE_V,
        xy_at_rest=abs(x1 - x0) <= XY_STEP_TOL_V and abs(y1 - y0) <= XY_STEP_TOL_V,
        bias_zero=abs(bias) <= BIAS_TOL_V,
        z_v=height,
        bias_v=bias,
    )


def wait_for(cond, timeout: float = 10.0, step: float = 0.02) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(step)
    return bool(cond())
