"""Drive a lithography run against the thin Igor bridge (bridge_protocol.py).

Igor only arms, fires and reports; calibration, unit conversion, timing and the
per-object cycle (transit, set down, load, arm+fire, await DONE, lift) live here, using
litho.compiler's validated math. The one addition is the external tip-amp gain:
V_dac = V_tip / amp_gain. Works identically against igor_stub.FakeIgorBridge.
"""

from __future__ import annotations

import math
import queue
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .. import clock

# Shared with the twin so both backends cut an over-cap stroke at the SAME indices.
from ..litho.chunker import (
    CHAIN_CHUNK_POINTS,
    MAX_OUTWAVE_POINTS,
    _penup_mask,
    chunk_boundaries,
    is_chained,
)
from ..litho.compiler import capture_points, time_segment
from . import bridge_protocol as P
from . import channels, engage_check, wire
from .bridge_protocol import BridgeLink
from .bridge_router import BridgeRouter, send_abort
from .free_air import FREE_AIR_ABS_BAND_V, agree_within
from .hardware_arc import CYPHER_MICROSCOPE_ID
from .output_a_check import stop_loops_driving_output_a

# RAMP_SETTLE_MARGIN_S is re-exported (litho_cmds and a test import it from here).
from .xy_transit import (
    _X_IN,
    _X_LOOP,
    _X_OUT,
    _X_SETPOINT,
    _Y_IN,
    _Y_LOOP,
    _Y_OUT,
    _Y_SETPOINT,
    DEFAULT_WITHDRAW_MAX_S,
    DEFAULT_WITHDRAW_SETTLE_S,
    MOVE_SPEED_UM_S,
    RAMP_SETTLE_MARGIN_S,  # noqa: F401 -- re-exported, see the comment above
    _plan_ramp,
    TransitAborted,
    xy_transit,
)

# Z feedback is loop 2 (X/Y loops 0/1 live in xy_transit). The mode-agnostic aliases
# Fast%Input/Z%Output are REJECTED on the MFP-3D crosspoint (ae_BadParm), so use the
# concrete contact channels (AC would need Amplitude). _Z_OUT is the Height alias.
_Z_LOOP = 2
_Z_IN = channels.DEFLECTION
_Z_OUT = channels.Z_OUT

# STARTUP keys that are the LVDT calibration the conversion needs.
_CAL_KEYS = ("x_lvdt_sens", "x_lvdt_offset", "y_lvdt_sens", "y_lvdt_offset",
             "z_lvdt_sens", "z_lvdt_offset")

# SETTLE_MOVE_S: piezo/feedback settle after any physical-setting change (Asylum's
# litho waits cLithoSleepTics = 500 ms, 1 s in AC). 1.0 s: on the Cypher at 0.30 s the
# integral-only loop 2 was still bringing the tip down when bias and XY started
# (2026-10-28). CMD_GAP_S: transport recovery gap after EVERY command. Both overridable.
SETTLE_MOVE_S = 1.0
CMD_GAP_S = 0.02

# Dead alias of the chunker's out-wave cap (only a test reads it): docs/TECH_DEBT.md.
OUTWAVE_MAX_POINTS = MAX_OUTWAVE_POINTS

# Points per WAVE/WAVEAPP bridge line (a line-size bound, not a wave limit).
WAVE_CHUNK_POINTS = 8192


# Settle between the first object's withdraw+move and its free-air read (2026-09-29,
# docs/PER_OBJECT_FREE_AIR_SPEC.md). Tests pass 0.
FIRST_OBJECT_SETTLE_S = 3.0

# A free-air read counts only if the Z piezo drive (the run's z_out, read by alias) is
# BELOW this: a retract can leave it at -10 V, which passes (2026-09-29).
WITHDRAWN_Z_DRIVE_MAX_V = 1.0

# The deflection setpoint of the back-off after a pass: a withdraw that holds the loop
# closed at -10 V, not the verified withdraw (which parks Z at 0 V).
RETRACT_SETPOINT_V = -10.0

# The raw controller integral gain of the write's loop 2 (`_arm_z`), positive: on a contact
# approach the deflection rises. Its sign is the sign the run states for every engage.
Z_IGAIN = 1200.0

# Free-air deflection reads that must agree before a setpoint is taken from them.
FREE_AIR_READS = 3

# Controller master clock (Hz): one tick of an out-wave's interpolation.
CLOCK_HZ = 50_000

# DONE wait beyond the capture fill of the fired stroke: Igor streams the captures
# before it sends DONE (AFM_LithoDone).
DONE_MARGIN_S = 15.0


def _done_wait_s(timeout_s: float, fill_s: float) -> float:
    """Seconds to wait for a stroke's DONE: its capture fill (points fired x tick) plus
    the margin, never less than the run's command timeout."""
    return max(float(timeout_s), float(fill_s) + DONE_MARGIN_S)


class WriteAborted(RuntimeError):
    """The run's abort event was set before the object's approach: nothing was set down
    or fired for this object (the abort path owns the retract)."""


class FreeAirReadFailed(RuntimeError):
    """A free-air deflection or Z-drive read came back dead (NaN/Inf, or a read error):
    the write stops before the approach, relative or absolute (2026-09-29)."""


class FreeAirRefused(RuntimeError):
    """A relative write's first object has no usable free-air read (reads disagree, or
    the Z drive is not below 1 V): no earlier setpoint exists, so the write is refused
    before any approach."""


@dataclass(frozen=True)
class WriteSetpoint:
    """The setpoint as the write was requested. `relative=True`: `value_v` is added to
    each object's own free-air read (a door's `setpoint_reference` "free_air", or the cockpit's `setpoint_relative`).
    `relative=False`: `value_v` is the absolute deflection setpoint, used unchanged
    (`setpoint_reference` "absolute", the default)."""
    value_v: float
    relative: bool


def as_write_setpoint(setpoint: float | WriteSetpoint) -> WriteSetpoint:
    """A bare float is an absolute setpoint; a `WriteSetpoint` is taken as it is."""
    if isinstance(setpoint, WriteSetpoint):
        return setpoint
    return WriteSetpoint(float(setpoint), relative=False)


def is_hover_move(first: bool, source: str | None, last_source: str | None) -> bool:
    """The move to a stroke is a hover (the next stroke of the same source object) rather
    than a transit with a verified withdraw. The first stroke of a write never hovers."""
    return not first and source is not None and source == last_source


def fired_point_count(n: int, last_v: float) -> int:
    """Points fired for a stroke of `n` points whose last bias is `last_v`. Compiler output,
    a multiple of 32 ending at 0 V, is fired as it is; anything else is padded with 0 V up
    to capture_points(n + 1) so the controller does not hold a bias after the stroke."""
    if n == 0 or (n % 32 == 0 and last_v == 0.0):
        return n
    return capture_points(n + 1)


def stroke_name(name) -> str:
    """The name a stroke goes over the wire under. The bridge is space-delimited: a space
    splits ARMFIRE and the DONE match never happens (the write hangs); an empty name shifts
    every argument by one."""
    return "_".join(str(name).split()) or "obj"


def pad_fired(x_um, y_um, v_tip) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The arrays as they are fired: compiler output passes through; anything else keeps
    its points and gets the last XY repeated at 0 V up to `fired_point_count`."""
    x = np.asarray(x_um, dtype=np.float64)
    y = np.asarray(y_um, dtype=np.float64)
    v = np.asarray(v_tip, dtype=np.float64)
    n = len(x)
    extra = fired_point_count(n, v[-1] if n else 0.0) - n
    if extra == 0:
        return x, y, v
    return (np.concatenate([x, np.full(extra, x[-1])]),
            np.concatenate([y, np.full(extra, y[-1])]),
            np.concatenate([v, np.zeros(extra)]))


def lvdt_volts(position_um, sens: float, offset: float):
    """Position in um to LVDT volts (compile_object / Asylum XYMetersToVoltage):
    Volts = Position[m] / Sensitivity + Offset."""
    return position_um * 1e-6 / sens + offset


def dac_volts(v_tip, amp_gain: float):
    """The DAC gets the tip voltage divided by the external tip-amp gain (a driver
    config, not the Asylum TipGain)."""
    return v_tip / amp_gain


def chunk_bounds(n: int, v, x, y) -> list[int]:
    """Chunk boundaries [0, c1, ..., n] of an n-point stroke: [0, n] unless it chains.
    A pen-up (NaN) in any array marks a cut candidate, so `v`, `x`, `y` may be in um or
    volts; the NaNs fall in the same places."""
    if not is_chained(n, CHAIN_CHUNK_POINTS):
        return [0, n]
    return chunk_boundaries(n, _penup_mask(v, x, y), CHAIN_CHUNK_POINTS)


def _finite(v, default: float = 0.0) -> float:
    """Coerce to a finite float; NaN / Inf / None / non-numeric -> default."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


@dataclass
class WriteResult:
    """What one object's write did, for the caller and the tests."""
    name: str
    points: int
    interp: float          # signed ticks/point (negative == linear)
    cap_len: int           # capture length, padded to a multiple of 32
    duration_s: float
    length_um: float
    aborted: bool = False
    # Per-channel input-wave capture streamed back by the bridge.
    capture: dict[str, list[float]] = field(default_factory=dict)


class LithographyRun:
    """One lithography session over a connected bridge link."""

    def __init__(self, link: BridgeLink, *, amp_gain: float = 1.0,
                 retract: float = RETRACT_SETPOINT_V, z_igain: float = Z_IGAIN,
                 decimation: int = 50, timeout: float = 30.0,
                 move_speed_um_s: float = MOVE_SPEED_UM_S,
                 settle_move_s: float = SETTLE_MOVE_S,
                 cmd_gap_s: float = CMD_GAP_S, z_out: str = _Z_OUT,
                 xy_withdraw_max_s: float = DEFAULT_WITHDRAW_MAX_S,
                 xy_withdraw_settle_s: float = DEFAULT_WITHDRAW_SETTLE_S,
                 first_object_settle_s: float = FIRST_OBJECT_SETTLE_S,
                 free_air_abs_band_v: float | None = None,
                 router: BridgeRouter | None = None, com_igor=None, log=None,
                 setpoint_log=None, abort_event=None):
        self.link = link
        # Loop-2 OUTPUT channel from the caller (hub.hardware_arc.z_out_channel): Height.
        # A wrong output means the lift/servo drives nothing: a tip crash, not just a
        # failed retract. Input is Deflection.
        self.z_out = str(z_out)
        # Single-reader router, shared with the HardwareARC scan path on the instrument so
        # a scan's reads and a write's completion never race on the one socket.
        self._log = log or (lambda msg: None)
        self._owns_router = router is None
        self.router = router
        if link is not None and self.router is None and wire.WIRE_VERSION == 3:
            from .wire_v3 import open_v3
            self.link, self.router = open_v3(link, log=self._log)
        elif link is not None and self.router is None:
            self.router = BridgeRouter(link, log=self._log).start()
        # Out-of-band abort escape (COM first, TCP fallback); None -> TCP only.
        self.com_igor = com_igor
        self.amp_gain = float(amp_gain)
        self.retract = float(retract)
        self.z_igain = float(z_igain)
        self.decimation = int(decimation)
        self.timeout = float(timeout)
        # Repositioning-move speed (um/s), separate from the per-object WRITE speed.
        self.move_speed_um_s = float(move_speed_um_s)
        # Previous stroke's LAST XY point (um): the ramp START for `_hover_move`, which
        # has no live-position read. Load-bearing, not informational.
        self._last_xy_um: tuple[float, float] | None = None
        # Source object of the last write_object call (XY_SLEW_SPEC rule 5): a match
        # means a fill row of the SAME shape (hover move), else the verified withdraw.
        self._last_source: str | None = None
        self.settle_move_s = float(settle_move_s)
        self.cmd_gap_s = float(cmd_gap_s)
        # Verified-withdraw ramp bound + settle before its verify (xy_transit ->
        # hw_withdraw); defaults match hw_withdraw's, tests pass 0.
        self.xy_withdraw_max_s = float(xy_withdraw_max_s)
        self.xy_withdraw_settle_s = float(xy_withdraw_settle_s)
        # Tip engaged (set by set-down, cleared ONLY by xy_transit's verified withdraw):
        # xy_transit's `engaged` input for the next move.
        self._engaged: bool = False
        # Per-object free-air read (docs/PER_OBJECT_FREE_AIR_SPEC.md): one 3-read agreement
        # (free_air.agree_within, the door's band), no dwell or second triple.
        self.first_object_settle_s = float(first_object_settle_s)
        if free_air_abs_band_v is None:
            free_air_abs_band_v = FREE_AIR_ABS_BAND_V
        self.free_air_abs_band_v = float(free_air_abs_band_v)
        # The one line per object (free air read, setpoint used); separate from `log` so
        # the worker can surface it without also surfacing xy_transit's lines.
        self._setpoint_log = setpoint_log or self._log
        # Absolute setpoint of the current source object; None until the first object's
        # free-air read, i.e. no earlier setpoint to fall back on.
        self._object_setpoint: float | None = None
        # The current source object's agreed free-air read (engage_check measures from it).
        self._object_free_air: float | None = None
        # One record per free-air re-read (object, free air, setpoint used, note), i.e.
        # per object, not per written segment: same-source hover rows add none. For the
        # write record: the per-object setpoints differ from the door-time one.
        self.object_setpoints: list[dict] = []
        # The worker's abort flag (threading.Event). Abort stops everything: the settle
        # and reads wait on it, and nothing is approached once it is set.
        self._abort_event = abort_event
        self.startup: dict = {}
        self.cal: dict = {}
        # Called as on_outcome(kind, make) after each step of an object (move, settle, read,
        # engage, pass, withdraw); `make()` returns what the step found. The service sets it to
        # journal the outcome; it must not block, and a failure in it never reaches the write.
        self.on_outcome: Callable[..., None] | None = None

    def _outcome(self, kind: str, make: Callable[[], dict]) -> None:
        """Report a step's outcome. `make` runs inside the guard, so building the report
        cannot stop a write either."""
        if self.on_outcome is not None:
            try:
                self.on_outcome(kind, make)
            except Exception:  # noqa: BLE001,S110 - a note must never stop a write
                pass

    @property
    def on_telem(self):
        """Optional telemetry hook, forwarded to the router: called with TELEM frames as
        (x, y, zdrive, defl, sum, amp). None skips TELEM."""
        return self.router.on_telem if self.router is not None else None

    @on_telem.setter
    def on_telem(self, cb):
        if self.router is not None:
            self.router.on_telem = cb

    @classmethod
    def connect(cls, host: str, port: int, **cfg) -> "LithographyRun":
        """Client role: Python connects OUT to the bridge (used by the tests,
        where the stub is the server)."""
        return cls(BridgeLink.connect(host, port), **cfg)

    @classmethod
    def serve(cls, port: int, *, host: str = "0.0.0.0",
              accept_timeout: float | None = None, **cfg) -> "LithographyRun":
        """Server role (the HARDWARE topology: Igor's SOCKIT connects out from
        AFM_LithoRunStart). Blocks until Igor connects or `accept_timeout` elapses."""
        srv, _ = BridgeLink.listen(host, port)
        try:
            link = BridgeLink.accept_on(srv, timeout=accept_timeout)
        finally:
            srv.close()
        return cls(link, **cfg)

    # ------------------------------------------------------------- setup ----

    def start(self) -> dict:
        """Handshake: ask the bridge for every startup value, verify the protocol
        version, and parse the calibration. A version-mismatched peer fails LOUDLY
        here rather than mid-run."""
        import queue as _queue
        w = self.router.open_waiter()
        try:
            self.link.send(P.cmd_startup(w.uid))
            verb, rest = w.get(self.timeout)
        except _queue.Empty:
            raise RuntimeError(
                "bridge handshake failed: no STARTUP reply within "
                f"{self.timeout:g}s (is the Igor .ipf the UID-aware "
                f"protocol v{P.PROTOCOL_VERSION} build?)") from None
        finally:
            self.router.release(w.uid)
        if verb != P.STARTUP:
            raise RuntimeError(f"expected STARTUP reply, got {verb} {rest!r}")
        self.startup = _parse_kv(rest)
        if not getattr(self.link, "checks_version", False):   # a v3 link gates by itself
            _check_proto(self.startup)
        self.cal = {k: self.startup[k] for k in _CAL_KEYS if k in self.startup}
        return self.cal

    # ---------------------------------------------------------- one object --

    def write_object(self, name: str, x_um, y_um, v_tip,
                     setpoint: float | WriteSetpoint,
                     speed: float, *, source: str | None = None) -> WriteResult:
        """Compile and fire one object, then wait for its completion report.

        x_um/y_um/v_tip are the final per-point waveforms (already sampled): the
        driver converts them with the validated conventions and runs the cycle
        transit -> set down -> load waves -> arm+fire -> await -> lift.

        `source` (XY_SLEW_SPEC rule 5): the parent object name (`meta["object"]`) before
        a fill's `expand()`. Matching the previous call's -> hover move; differing or
        None -> full `xy_transit` (verified withdraw when engaged; always on the first
        object), then a free-air read before the set-down (PER_OBJECT_FREE_AIR_SPEC).

        `setpoint`: a `WriteSetpoint`, or a bare float meaning an absolute setpoint.
        """
        name = stroke_name(name)
        x, y, v = self._fired_arrays(name, x_um, y_um, v_tip)
        n = len(x)

        # LVDT convention (compile_object / Asylum XYMetersToVoltage):
        #   Volts = Position[m] / Sensitivity + Offset
        x_lit = lvdt_volts(x, self.cal["x_lvdt_sens"], self.cal["x_lvdt_offset"])
        y_lit = lvdt_volts(y, self.cal["y_lvdt_sens"], self.cal["y_lvdt_offset"])
        # compile_object emits the TIP voltage; the DAC gets it divided by the
        # external tip-amp gain (a driver config, NOT the Asylum TipGain).
        v_lit = dac_volts(v, self.amp_gain)

        # Interp is SIGNED: negative == linear, |interp| ticks per point.
        path = np.column_stack([x, y])
        t = time_segment(path, float(speed), round_up=True)   # a stroke: tick rounds up
        interp = -float(t["ticks_per_point"])
        # Decimation must track speed (one sample per interpolated point): a FIXED
        # decimation fills the capture bank early and lifts the tip mid-write.
        decim = max(1, abs(int(round(interp))))
        # Igor reports DONE when the capture fills (AFM_LithoDone): n points, one per
        # tick, n being the capture's own length.
        fill_s = n * float(t["ticks_per_point"]) / CLOCK_HZ
        wait_s = _done_wait_s(self.timeout, fill_s)

        # Chain: one object name, captures accumulated per chunk. n and the chunk size
        # are multiples of 32, so every chunk is one too.
        chained = is_chained(n, CHAIN_CHUNK_POINTS)
        bounds = chunk_bounds(n, v_lit, x_lit, y_lit)
        if chained:
            # total captured length is the sum of each chunk's padded capture length
            cap_len = sum(capture_points(bounds[k + 1] - bounds[k])
                          for k in range(len(bounds) - 1))
        else:
            # capture length padded UP to a multiple of 32 (capture_points).
            cap_len = capture_points(n)

        # --- the per-object cycle, in order --------------------------------
        # Transit (XY_SLEW_SPEC rule 5): same source -> `_hover_move`; else xy_transit
        # (VERIFIED withdraw if engaged, then the ramp). Re-approach is the set-down
        # below (_arm_z, raw z_igain), never xy_transit's (different gain convention).
        x_pgain = _finite(self.startup.get("x_pgain"))
        x_igain = _finite(self.startup.get("x_igain"))
        y_pgain = _finite(self.startup.get("y_pgain"))
        y_igain = _finite(self.startup.get("y_igain"))
        first = self._object_setpoint is None
        if first:
            stop_loops_driving_output_a(
                self, self._log, self.startup.get("microscope_id") == CYPHER_MICROSCOPE_ID,
                self._abort_requested)
        if is_hover_move(first, source, self._last_source):
            self._hover_move(float(x_um[0]), float(y_um[0]),
                             float(x_lit[0]), float(y_lit[0]),
                             x_pgain, x_igain, y_pgain, y_igain)
            # `_engaged` untouched: the hover lift is not a withdraw.
            self._outcome("move", lambda: {
                "how": "hover", "name": name, "retract_setpoint_v": self.retract,
                "from_um": self._last_xy_um, "to_um": (float(x_um[0]), float(y_um[0]))})
        else:
            # First object: withdraw even if believed withdrawn; the run cannot see
            # hub.engaged, and a write started engaged would otherwise move XY in contact.
            try:
                moved = xy_transit(self, self.cal, float(x_um[0]), float(y_um[0]),
                                   engaged=self._engaged or first, reapproach=False,
                                   x_pgain=x_pgain, x_igain=x_igain, y_pgain=y_pgain,
                                   y_igain=y_igain, move_speed_um_s=self.move_speed_um_s,
                                   ramp_settle_s=self.settle_move_s,
                                   withdraw_max_s=self.xy_withdraw_max_s,
                                   withdraw_settle_s=self.xy_withdraw_settle_s,
                                   should_abort=self._abort_requested, log=self._log)
            except TransitAborted:
                raise WriteAborted(f"write {name}: aborted in the move") from None
            self._engaged = False
            self._outcome("move", lambda: {
                "how": "transit", "name": name, "to_um": (float(x_um[0]), float(y_um[0])),
                **moved})
            if first and self.first_object_settle_s:
                self._wait_or_abort(self.first_object_settle_s, name)
                self._outcome("settle", lambda: {"seconds": self.first_object_settle_s})
            self._object_setpoint = self._free_air_setpoint(name, setpoint)
        self._last_source = source
        self._raise_if_aborted(name)
        # Set down at the litho (contact) setpoint (platform paths: _arm_z).
        self._arm_z(self._object_setpoint)
        self._settle_move()                    # let the tip engage
        self._engaged = True
        found = self._confirm_engage(name) if engage_check.ENABLED else {}
        self._outcome("engage", lambda: {
            "name": name, "setpoint_v": self._object_setpoint, "settle_s": self.settle_move_s,
            **found})
        # An abort has already sent SafePark: from here on no fire and no _arm_z (on
        # the Cypher a re-arm of loop 2 would undo SafePark's retract).
        self._raise_if_aborted(name, "before the fire")
        # Load the waves; WAVE+WAVEAPP chunks keep each bridge line small.
        self._load_wave("XLitho", x_lit)
        self._load_wave("YLitho", y_lit)
        self._load_wave("VLitho", v_lit)
        if chained:
            self._load_wave("LithoBounds", bounds)
        self._raise_if_aborted(name, "before the fire")
        # Arm + fire: oversize -> chained LITHOCHAIN over the bounds wave, else one ARMFIRE.
        try:
            if chained:
                verb, _, capture = self._fire_and_wait(
                    lambda uid: P.cmd_lithochain(uid, name, interp, decim, "LithoBounds"),
                    name, timeout=wait_s)
            else:
                verb, _, capture = self._fire_and_wait(
                    lambda uid: P.cmd_armfire(uid, name, interp, cap_len, decim),
                    name, timeout=wait_s)
        except queue.Empty:
            raise TimeoutError(
                f"write {name}: no DONE within {wait_s:.1f} s (capture fill "
                f"{fill_s:.1f} s: {n} points at {decim} ticks)") from None
        aborted = (verb == P.ABORTED)
        self._outcome("pass", lambda: {
            "name": name, "fire": P.LITHOCHAIN if chained else P.ARMFIRE, "answer": verb,
            "aborted": aborted, "points": n, "chunks": len(bounds) - 1 if chained else 1,
            "captured_points": max((len(v) for v in (capture or {}).values()), default=0)})
        # Lift by re-arming loop 2 (_arm_z); not after an abort, whose SafePark has
        # already retracted.
        if not aborted and not self._abort_requested():
            self._arm_z(self.retract)
            self._settle_move()
            self._outcome("withdraw", lambda: {
                "name": name, "setpoint_v": self.retract, "settle_s": self.settle_move_s})
        else:
            self._outcome("withdraw", lambda: {"name": name, "skipped": True})
        # `self._engaged` deliberately stays True: this lift is a mild hover (Z loop still
        # CLOSED), NOT a withdraw, so the NEXT xy_transit must see engaged=True and run
        # its VERIFIED withdraw. Only that withdraw clears the flag.
        # Ramp START for the next `_hover_move` (that path has no live-position read);
        # removing this would slew the next same-source move from (0, 0).
        self._last_xy_um = (float(x_um[-1]), float(y_um[-1]))

        return WriteResult(name=name, points=n, interp=interp, cap_len=cap_len,
                           duration_s=float(t["duration_s"]),
                           length_um=float(t["length_um"]), aborted=aborted,
                           capture=capture)

    def _fired_arrays(self, name: str, x_um, y_um, v_tip
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The arrays as they are fired. Arrays from the compiler are a multiple of 32
        points ending at 0 V and pass through. Anything else keeps its points and gets
        the last XY repeated at 0 V up to capture_points(n + 1), with one log line:
        the controller holds the last out-wave sample until the capture fills, so a
        stroke that does not end at 0 V is held at its last bias."""
        x, y, v = pad_fired(x_um, y_um, v_tip)
        if len(x) != len(x_um):
            self._log(f"write {name}: {len(x_um)} points not a multiple of 32 ending at 0 V; "
                      f"{len(x) - len(x_um)} points at 0 V appended at the last position")
        return x, y, v

    def run(self, objects) -> list[WriteResult]:
        """Write a sequence of (name, x_um, y_um, v_tip, setpoint, speed)."""
        results = []
        for o in objects:
            results.append(self.write_object(*o))
        return results

    # ---------------------------------------------------------- controls ----

    def abort(self) -> str:
        """Stop now AND retract, OUT-OF-BAND (COM first, TCP ABORT fallback): `send_abort`
        fires `AFM_SafePark()`, a superset of `AFM_Abort()` (docs/ABORT_SEQUENCE.md).
        Send-only, never waits behind the router. Returns the transport used."""
        return send_abort(self.link, self.com_igor, log=self._log)

    def get_value(self, param: str, *, timeout: float | None = None) -> float:
        """One GETVALUE; `timeout` bounds the reply wait (default: the run's timeout)."""
        import queue as _queue
        wait_s = self.timeout if timeout is None else float(timeout)
        w = self.router.open_waiter()
        try:
            self.link.send(P.cmd_getvalue(w.uid, param))
            verb, rest = w.get(wait_s)
        except _queue.Empty:
            raise RuntimeError(
                f"expected VALUE reply for {param!r}, none within "
                f"{wait_s:g}s") from None
        finally:
            self.router.release(w.uid)
        if verb == P.VALUE:
            return float(rest.strip())
        if verb == P.ERR:
            raise RuntimeError(f"bridge error: {rest.strip()}")
        raise RuntimeError(f"expected VALUE reply for {param!r}, got {verb} {rest!r}")

    def close(self) -> None:
        if self._owns_router and self.router is not None:
            self.router.stop()
        self.link.close()

    # ------------------------------------------------------------ internals --

    def _send(self, cmd: str) -> None:
        """Send one command, then the small recovery gap (cmd_gap_s)."""
        self.link.send(cmd)
        if self.cmd_gap_s:
            clock.sleep(self.cmd_gap_s)

    def _load_wave(self, name: str, values) -> None:
        """Transfer one named wave to Igor: WAVE with the first chunk, WAVEAPP for the
        rest (same framing as HardwareARC._load_wave)."""
        vals = np.asarray(values, dtype=np.float64)
        if hasattr(self.link, "load_wave"):          # v3: paced by Igor's replies
            self.link.load_wave(name, vals)
            return
        c = WAVE_CHUNK_POINTS
        self._send(P.cmd_wave(name, vals[:c]))       # create (or whole, if small)
        for i in range(c, len(vals), c):
            self._send(P.cmd_waveapp(name, vals[i:i + c]))

    def _settle_move(self) -> None:
        """Relax time after a physical move (piezo settle)."""
        if self.settle_move_s:
            clock.sleep(self.settle_move_s)

    def _arm_z(self, setpoint: float) -> None:
        """Apply a new loop-2 setpoint for a set-down (contact) or lift (retract) by
        RE-ARMING loop 2 (ir_SetPISLoop, Deflection -> self.z_out, integral only), the
        primitive hw_scan/hw_approach engage with. A PIDSLoop.2.Setpoint poke is IGNORED
        by the running Height loop (the vendor uses HeightLoop.Setpoint), so it would stay
        at the -10 lift setpoint, ride the output-min rail and write nothing."""
        self._send(P.cmd_servoloop(_Z_LOOP, _Z_IN, float(setpoint),
                                   0, self.z_igain, self.z_out))

    def _abort_requested(self) -> bool:
        return self._abort_event is not None and self._abort_event.is_set()

    def _stop_hover_if_aborted(self) -> None:
        if self._abort_requested():
            raise WriteAborted("write aborted in the hover move")

    def _raise_if_aborted(self, name: str, where: str = "before the approach") -> None:
        if self._abort_requested():
            raise WriteAborted(f"write {name}: aborted {where}")

    def _wait_or_abort(self, seconds: float, name: str,
                       where: str = "before the approach") -> None:
        """Sleep `seconds`, returning early (and raising) the moment abort is set."""
        if self._abort_event is None:
            clock.sleep(seconds)
        elif clock.wait(self._abort_event, seconds):
            raise WriteAborted(f"write {name}: aborted {where}")

    def _confirm_engage(self, name: str) -> dict:
        """engage_check's Deflection poll for this object (switch on only): reads, no
        commands. An abort raises WriteAborted, a timeout EngageNotConfirmed."""
        where = "while waiting for the engage"
        defl = engage_check.confirm_engage(
            name, float(self._object_setpoint), float(self._object_free_air),
            read=self._read_deflection,
            check_abort=lambda: self._raise_if_aborted(name, where),
            wait=lambda s: self._wait_or_abort(s, name, where))
        return {"deflection_v": defl, "free_air_v": self._object_free_air}

    def _read_deflection(self) -> float:
        """One Deflection read, bounded like the withdraw-path reads; NaN on any failure."""
        from .hw_approach import VERIFY_READ_TIMEOUT_S
        try:
            v = float(self.get_value(_Z_IN, timeout=VERIFY_READ_TIMEOUT_S))
        except Exception:                      # noqa: BLE001 -- a failed read never counts
            return math.nan
        return v if math.isfinite(v) else math.nan

    def _free_air_setpoint(self, name: str, setpoint: float | WriteSetpoint) -> float:
        """Read free air (tip withdrawn, at the object's first point) and return this
        object's absolute setpoint: free air + the value for a relative request, the
        value unchanged for an absolute one. Reads that disagree on a relative write: the
        previous object's setpoint, or `FreeAirRefused` on the first object. An absolute
        setpoint is never refused or changed over free air (the script owns it); a
        disagreement only logs. The read counts only with the Z drive below
        WITHDRAWN_Z_DRIVE_MAX_V (else handled as a disagreement); a dead Z-drive or
        deflection read raises `FreeAirReadFailed` on any write.
        Logs and records one line either way."""
        from .hw_approach import hw_read_free_air  # looked up at call time: tests patch it
        req = as_write_setpoint(setpoint)
        prev = self._object_setpoint
        self._object_free_air = None
        self._raise_if_aborted(name)
        try:
            z_v = float(self.td_ReadValue(self.z_out))
        except Exception:                      # noqa: BLE001 -- a failed read is dead
            z_v = math.nan
        if not math.isfinite(z_v):
            msg = f"Z drive ({self.z_out}) read failed on {name}; write stopped"
            self._record(name, None, None, msg, z_drive_v=z_v)
            raise FreeAirReadFailed(msg)
        if z_v >= WITHDRAWN_Z_DRIVE_MAX_V:
            why = (f"Z drive {self.z_out} {z_v:.3f} V >= {WITHDRAWN_Z_DRIVE_MAX_V:g} V, "
                   "tip not confirmed withdrawn; free-air read not used")
            return self._unusable_read(name, req, prev, why, z_drive_v=z_v)
        samples = []
        for _ in range(FREE_AIR_READS):
            self._raise_if_aborted(name)
            v = float(hw_read_free_air(self))
            if not math.isfinite(v):
                msg = f"free-air deflection read failed on {name}; write stopped"
                self._record(name, None, None, msg, z_drive_v=z_v, reads=samples)
                raise FreeAirReadFailed(msg)
            samples.append(v)
        if agree_within(samples, frac=0.10, abs_band_v=self.free_air_abs_band_v):
            fa = sum(samples) / float(FREE_AIR_READS)
            sp = fa + req.value_v if req.relative else req.value_v
            refusal = engage_check.step_refusal(name, sp, fa) if engage_check.ENABLED else None
            if refusal:
                self._record(name, fa, None, refusal, z_drive_v=z_v, reads=samples)
                raise FreeAirRefused(refusal)
            self._object_free_air = fa
            kind = f"relative {req.value_v:+.4f} V" if req.relative else "absolute"
            msg = f"write {name}: free air {fa:.4f} V, setpoint {sp:.4f} V ({kind})"
            self._record(name, fa, sp, msg, z_drive_v=z_v, reads=samples)
            return sp
        reads = ", ".join(f"{v:.4f}" for v in samples)
        return self._unusable_read(name, req, prev, f"free-air reads disagree ({reads} V)",
                                   z_drive_v=z_v, reads=samples)

    def _unusable_read(self, name: str, req: WriteSetpoint, prev: float | None,
                       why: str, **found) -> float:
        """No usable free air for this object: absolute -> as sent; relative -> the
        previous object's setpoint, or `FreeAirRefused` on the first object. With
        engage_check on, always `FreeAirRefused`: the engage is measured from free air."""
        if engage_check.ENABLED:
            msg = (f"write {name}: {why}; no free-air reference for the engage check, "
                   "write refused before the set-down")
            self._record(name, None, None, msg, **found)
            raise FreeAirRefused(msg)
        if not req.relative:
            msg = f"write {name}: {why}; setpoint {req.value_v:.4f} V (absolute, as sent)"
            self._record(name, None, req.value_v, msg, **found)
            return req.value_v
        if prev is None:
            msg = (f"write {name}: {why} on the first object, no earlier setpoint; "
                   "write refused before any approach")
            self._record(name, None, None, msg, **found)
            raise FreeAirRefused(msg)
        msg = f"write {name}: {why}; using the previous object's setpoint {prev:.4f} V"
        self._record(name, None, prev, msg, **found)
        return prev

    def _record(self, name: str, free_air_v: float | None, setpoint_v: float | None,
                msg: str, **found) -> None:
        """Keep one free-air record, log it, and report it (`found`: the Z drive and the
        reads behind it)."""
        self.object_setpoints.append({"object": name, "free_air_v": free_air_v,
                                      "setpoint_v": setpoint_v, "note": msg})
        self._setpoint_log(msg)
        self._outcome("read", lambda: {"name": name, "free_air_v": free_air_v,
                                       "setpoint_v": setpoint_v, "note": msg, **found})

    def _hover_move(self, x_um0: float, y_um0: float, x_lit0: float, y_lit0: float,
                    x_pgain: float, x_igain: float, y_pgain: float, y_igain: float) -> None:
        """Move between strokes of the SAME source object (XY_SLEW_SPEC rule 5), e.g. fill
        rows; deliberately not `xy_transit`, which would open/close Z on every row.

        Lift to the retract deflection setpoint (loop stays CLOSED: NOT a verified
        withdraw), arm X/Y at `self._last_xy_um` (no live read on this path), then a
        rate-limited ramp to this stroke's first point via `_plan_ramp`. An abort stops it
        before each send: no XY after an abort, and its park has already lifted the tip."""
        self._stop_hover_if_aborted()
        self._send(P.cmd_servoloop(_Z_LOOP, _Z_IN, self.retract,
                                   0, self.z_igain, self.z_out))
        self._settle_move()                    # let the tip retract
        self._stop_hover_if_aborted()
        start_x_um, start_y_um = self._last_xy_um if self._last_xy_um else (0.0, 0.0)
        start_x = lvdt_volts(start_x_um, self.cal["x_lvdt_sens"], self.cal["x_lvdt_offset"])
        start_y = lvdt_volts(start_y_um, self.cal["y_lvdt_sens"], self.cal["y_lvdt_offset"])
        self._send(P.cmd_servoloop(_X_LOOP, _X_IN, float(start_x),
                                   x_pgain, x_igain, _X_OUT))
        self._send(P.cmd_servoloop(_Y_LOOP, _Y_IN, float(start_y),
                                   y_pgain, y_igain, _Y_OUT))
        dx_um = abs(x_um0 - start_x_um)
        dy_um = abs(y_um0 - start_y_um)
        rate_x, rate_y, maxsec = _plan_ramp(dx_um, dy_um, self.cal, self.move_speed_um_s)
        self._stop_hover_if_aborted()
        self._send(P.cmd_ramp(_X_SETPOINT, float(x_lit0), rate_x, maxsec))
        self._send(P.cmd_ramp(_Y_SETPOINT, float(y_lit0), rate_y, maxsec))
        if self.settle_move_s:
            clock.sleep(float(maxsec) + self.settle_move_s)

    # -------------------------------- minimal ARCBackend-ish surface (xy_transit) --
    # Thin wrappers sending the SAME wire commands as HardwareARC over this run's link,
    # so xy_transit (and hw_withdraw) drive `self` like a HardwareARC. xy_transit falls
    # back to `self.z_out` (no z_out_channel property).

    def ir_SetPISLoop(self, index: int, events: str, input_: str, setpoint: float,
                      pgain: float, igain: float, sgain: float, output: str,
                      out_min: float, out_max: float) -> int:
        # out_min/out_max must reach Igor: hw_withdraw's Cypher retract relies on 0/150.
        self._send(P.cmd_servoloop(int(index), input_, float(setpoint),
                                   float(pgain), float(igain), output,
                                   float(out_min), float(out_max)))
        return 0

    def ir_StopPISLoop(self, index: int) -> int:
        self._send(P.cmd_stoploop(int(index)))
        return 0

    def ramp_output(self, channel: str, dest: float, rate: float = 80.0,
                    max_seconds: float = 2.0) -> int:
        self._send(P.cmd_ramp(channel, float(dest), float(rate), float(max_seconds)))
        return 0

    def td_ReadValue(self, name: str) -> float:
        return self.get_value(name)

    def td_xSetOutWavePair(self, bank: int, event: str, ch_a: str, wave_a,
                           ch_b: str, wave_b, interpolation: float) -> int:
        """OUTPAIR for xy_transit's single out-wave-pair move. Fixed wave names are safe:
        xy_transit's arm -> fire -> await is synchronous and WAVE overwrites."""
        self._load_wave("XYTransitA", wave_a)
        self._load_wave("XYTransitB", wave_b)
        self._send(P.cmd_outpair(int(bank), str(event), ch_a, "XYTransitA",
                                 ch_b, "XYTransitB", float(interpolation)))
        return 0

    def td_xSetInWavePair(self, bank: int, event: str, ch_a: str, ch_b: str,
                          n_points: int, decimation: int) -> int:
        """INPAIR: arm capture so the next `fire()` blocks on a real capture fill; a bare
        FIRE with no input bank returns DONE immediately (see xy_transit.py)."""
        self._send(P.cmd_inpair(int(bank), str(event), ch_a, ch_b,
                                int(n_points), int(decimation)))
        return 0

    def fire(self, event: str, ticks: int = 0, *, timeout: float | None = None) -> dict:
        """Fire `event` and wait for DONE/ABORTED (mirrors HardwareARC.fire), returning the
        capture. `ticks` is ignored. The fixed pass name is safe: synchronous, and
        `lr_pass()` refuses a second concurrent lr-class pass."""
        name = "XYTransit"
        _verb, _got, capture = self._fire_and_wait(
            lambda uid: P.cmd_fire(uid, name, str(event)), name, timeout=timeout)
        return capture

    def _fire_and_wait(self, cmd_fn, name: str, *, timeout: float | None = None
                       ) -> tuple[str, str, dict]:
        """The ONE register/send/await/release cycle for every lr-class pass on this link.

        The UID is registered BEFORE sending so every CAPTURE + the terminal route to
        THIS waiter. `lr_pass()` guards Igor's single root:AFM_lrUID: a second concurrent
        pass fails loudly instead of clobbering the UID (afm_litho_run.ipf)."""
        with self.router.lr_pass():
            w = self.router.open_waiter()
            try:
                self._send(cmd_fn(w.uid))
                return self._await_completion(
                    w, name, timeout=self.timeout if timeout is None else timeout)
            finally:
                self.router.release(w.uid)

    def _await_completion(self, waiter, name: str | None,
                          timeout: float | None = None) -> tuple[str, str, dict]:
        """Collect CAPTURE frames into {channel: [samples]} until DONE/ABORTED. The router
        has already filtered by UID and diverted TELEM. `timeout` bounds each frame wait
        (write_object passes the stroke duration)."""
        read_timeout = self.timeout if timeout is None else timeout
        capture: dict[str, list[float]] = {}
        while True:
            verb, rest = waiter.get(read_timeout)     # BridgeClosed / queue.Empty propagate
            if verb == P.CAPTURE:
                # CAPTURE <name> <channel> v;v;...
                cname, _, tail = rest.partition(" ")
                chan, _, body = tail.partition(" ")
                if name is None or cname == name:
                    vals = [float(v) for v in body.split(";") if v != ""]
                    # ACCUMULATE: a chained write streams one CAPTURE per channel PER
                    # CHUNK; appending concatenates them in fire order.
                    capture.setdefault(chan, []).extend(vals)
                continue
            got = rest.strip()
            if verb == P.ERR:
                raise RuntimeError(f"bridge error: {got}")
            if verb in (P.DONE, P.ABORTED):
                return verb, got, capture
            # anything else (shouldn't happen mid-write) is ignored


def _check_proto(startup: dict) -> None:
    """Fail LOUDLY on a protocol-version mismatch (mirror of hardware_arc._check_proto):
    the bridge echoes `proto=<v>` in its STARTUP reply; absent or wrong means the two
    ends speak different dialects, so refuse at connect."""
    proto = startup.get("proto")
    try:
        got = int(float(proto))
    except (TypeError, ValueError):
        got = None
    if got != P.PROTOCOL_VERSION:
        raise RuntimeError(
            f"bridge protocol mismatch: driver v{P.PROTOCOL_VERSION}, bridge "
            f"v{proto!r}. Reinstall afm_litho_run.ipf to the matching version.")


def _parse_kv(body: str) -> dict:
    """Parse a ';'-delimited k=v STARTUP body; numbers become floats."""
    out: dict = {}
    for item in body.split(";"):
        item = item.strip()
        if not item or "=" not in item:
            continue
        k, val = item.split("=", 1)
        k = k.strip()
        val = val.strip()
        try:
            out[k] = float(val)
        except ValueError:
            out[k] = val
    return out
