"""The hardware ARC: the `ARCBackend` seam driven over the thin Igor bridge.

`twin.arc.SoftwareARC` satisfies the `ARCBackend` Protocol by simulating each
`td_`/`ir_` call against the physics model. `HardwareARC` satisfies the SAME
Protocol by translating each call into `bridge_protocol` commands to real Igor
(or the test stub, `igor_stub.FakeIgorBridge`), which arms the waves and fires
them autonomously, then streams the capture back. Because both conform to the
seam, a single pass executor drives either backend unchanged.

The mapping is the general td_-mirroring surface of the bridge:

  * scalars      -> GETVALUE / SETVALUE
  * servo loops  -> SERVOLOOP (the bridge carries outMin/outMax, optional trailing
                    tokens -- Igor defaults -10/150 when absent; only sgain is fixed)
  * output waves -> WAVE (load by a unique name) + OUTWAVE / OUTPAIR (bind)
  * input waves  -> INWAVE / INPAIR (arm capture; the armed channels are tracked
                    locally so `fire` knows what will come back)
  * fire         -> FIRE, then await CAPTURE frames + the terminal DONE

The wire protocol is identical to `lithography_run.LithographyRun`'s -- this is
the same "arm/fire/await" pattern, but exposing the general per-channel seam the
executor depends on rather than the fixed litho channel set.
"""

from __future__ import annotations

import numpy as np

from ..litho.chunker import check_outwave_len
from . import bridge_protocol as P

from . import channels, retract, wire
from .bridge_protocol import BridgeLink
from .bridge_router import BridgeReadTimeout, BridgeRouter, send_abort

# Load a wave in pieces of at most this many points: one WAVE for the first chunk
# then WAVEAPP for the rest, so no single socket line is a multi-MB transfer. 8192
# is the size a 64x64 trace/retrace frame sent as one line, which loads cleanly; a
# 256x256 frame (131,072 points) becomes 16 lines instead of one 1.3 MB line.
WAVE_CHUNK_POINTS = 8192

# Bounded read (s) for the `read()` park-verify wrapper: the fail-safe verify must
# not inherit the 30 s default (an emergency stop that appears to hang). Matches the
# safe_park verify bound.
_PARK_READ_TIMEOUT_S = 2.0

# Vendor MicroscopeID for a Cypher (cMicroscopeCypher, HardWare.ipf:10). The instrument
# reports its id in the STARTUP payload (AFM_MacroStartup -> "microscope_id"); the Cypher
# differs from the MFP-3D in the crosspoint mode read-back (see is_cypher / check_mode).
CYPHER_MICROSCOPE_ID = 4


def _event(event) -> str:
    """Normalize an event arg to the bare bank string the bridge wants.

    Accepts an int, "0", or "Event.0" and returns "0" -- the bridge's event
    banks are addressed by the bare string, and a leading "Event." (as the twin
    uses internally) is stripped."""
    e = str(event).strip()
    if e.startswith("Event."):
        e = e[len("Event."):]
    return e


class HardwareARC:
    """The `ARCBackend` seam over a connected bridge link.

    Drop-in for `SoftwareARC`: same `td_`/`ir_` surface, same arm-and-fire
    semantics, but every call goes to real Igor over `bridge_protocol`.
    """

    def __init__(self, link: BridgeLink, *, timeout: float = 30.0,
                 router: BridgeRouter | None = None, com_igor=None, log=None):
        self.link = link
        self.timeout = float(timeout)
        self.startup: dict = {}
        # The single-reader ROUTER owns the socket receive side: EVERY reply is read
        # here and only here, then delivered to the requesting waiter by UID (see
        # bridge_router.py). No method reads `link` directly -- that was the race.
        # A None link (the Protocol-only smoke test) skips the router.
        self._log = log or (lambda msg: None)
        self._owns_router = router is None
        self.router = router
        if link is not None and self.router is None and wire.WIRE_VERSION == 3:
            from .wire_v3 import open_v3
            self.link, self.router = open_v3(link, log=self._log)
        elif link is not None and self.router is None:
            self.router = BridgeRouter(link, log=self._log).start()
        # Out-of-band abort escape: an optional COM Igor handle (anything exposing
        # .execute). When set, abort() calls Igor's stop over COM FIRST (preempts,
        # independent of the socket); otherwise it falls back to the TCP ABORT.
        # None off-instrument -> TCP only. NEEDS RIG CONFIRMATION (see send_abort).
        self.com_igor = com_igor
        # Out-of-band retract HOOK, wired by the service (hub.bind_hardware ->
        # FailsafeWatchdog.park_now). When set, `abort()` hands the retract to the
        # dedicated fail-safe thread instead of touching COM here. WHY: `abort()` is
        # called from the chained-scan reader's worker thread, and `LazyIgor` attaches
        # (and CACHES) its COM proxy on whichever thread uses it FIRST -- so an abort
        # from a pool thread could create the proxy in an un-CoInitialize'd apartment
        # that later dies, poisoning the handle the fail-safe itself depends on.
        # park_now() is NON-BLOCKING (a queue put + an Event set), so abort keeps its
        # send-only, never-queues-behind-the-router property. None (off-instrument,
        # the stub, a unit test) -> the direct COM/TCP path below, unchanged.
        self.failsafe_park = None
        # td_ReadString/WriteString are a local cache (the bridge has no string
        # get/set; firing is done via fire(), NOT by writing an Event string).
        self._strings: dict[str, str] = {}
        # channels armed since the previous fire; fire() expects a CAPTURE for
        # each, then clears them. Each entry is (channel, n_points).
        self._armed_in: list[tuple[str, int]] = []
        self._wave_seq = 0        # unique WAVE name counter (ow0, ow1, ...)
        self._fire_seq = 0        # unique pass name counter (pass0, pass1, ...)

    @property
    def on_telem(self):
        """The always-on telemetry hook. TELEM frames are routed by the router (the
        single reader); this forwards to it so `arc.on_telem = cb` keeps working. None
        (default) = TELEM quietly skipped. cb(x, y, zdrive, defl, sum, amp)."""
        return self.router.on_telem if self.router is not None else None

    @on_telem.setter
    def on_telem(self, cb):
        if self.router is not None:
            self.router.on_telem = cb

    @property
    def is_cypher(self) -> bool:
        """True when the connected instrument is a Cypher (MicroscopeID ==
        cMicroscopeCypher == 4, HardWare.ipf:10). Read from the STARTUP payload
        (`microscope_id`, emitted by AFM_MacroStartup). Defaults False when the field
        is absent (a pre-field bridge, the stub, or a link-less smoke arc) -- i.e. the
        MFP-3D behaviour, so nothing changes off a real Cypher. RIG-UNVERIFIED: the
        Cypher-specific channel choices this gates must be confirmed on the rig."""
        try:
            return int(float(self.startup.get("microscope_id", 0))) == CYPHER_MICROSCOPE_ID
        except (TypeError, ValueError):
            return False

    @property
    def z_out_channel(self) -> str:
        """The Z loop (loop 2) OUTPUT channel: the Height alias on both platforms. The
        Cypher's HeightLoop drives Height (the live PIDS panel); on the MFP-3D the alias
        resolves to Output.Z (2026-10-27). Only "loop 2 is the MFP-3D's Z loop" is
        still to confirm (RIG_CHECKLIST section 8). The INPUT stays Deflection on both.
        Every site that arms or ramps Z loop 2 must use this, or the servo drives nothing."""
        return channels.Z_OUT

    @classmethod
    def connect(cls, host: str, port: int, **cfg) -> "HardwareARC":
        """Client role: Python connects OUT to the bridge (used by the tests,
        where the stub is the server)."""
        return cls(BridgeLink.connect(host, port), **cfg)

    @classmethod
    def serve(cls, port: int, *, host: str = "0.0.0.0",
              accept_timeout: float | None = None, **cfg) -> "HardwareARC":
        """Server role (the HARDWARE topology): listen and accept the bridge's
        inbound connection -- real Igor is the SOCKIT client and dials out
        (AFM_LithoRunStart), so Python listens. Blocks until Igor connects (or
        `accept_timeout` seconds elapse). Mirrors LithographyRun.serve."""
        srv, _ = BridgeLink.listen(host, port)
        try:
            link = BridgeLink.accept_on(srv, timeout=accept_timeout)
        finally:
            srv.close()
        return cls(link, **cfg)

    # ------------------------------------------------------------- setup ----

    def start(self) -> dict:
        """Handshake: ask the bridge for every startup value, verify the protocol
        version, parse and return it. A UID-aware driver and a non-UID (or
        version-mismatched) bridge fail LOUDLY here, at connect, not mid-run."""
        verb, rest = self._handshake()
        if verb != P.STARTUP:
            raise RuntimeError(f"expected STARTUP reply, got {verb} {rest!r}")
        self.startup = _parse_kv(rest)
        if not getattr(self.link, "checks_version", False):   # a v3 link gates by itself
            _check_proto(self.startup)
        return self.startup

    def _handshake(self) -> tuple[str, str]:
        """Send STARTUP with a UID + our protocol version and await the reply via
        the router. A missing reply (a legacy bridge whose STARTUP echoes no UID,
        so the router can't match it) surfaces as a loud connect-time error rather
        than a silent timeout."""
        import queue as _queue
        w = self.router.open_waiter()
        try:
            self.link.send(P.cmd_startup(w.uid))
            return w.get(self.timeout)
        except _queue.Empty:
            raise RuntimeError(
                f"bridge handshake failed: no STARTUP reply within "
                f"{self.timeout:g}s. The Igor .ipf must be the UID-aware "
                f"(protocol v{P.PROTOCOL_VERSION}) build -- reinstall "
                f"afm_litho_run.ipf.") from None
        finally:
            self.router.release(w.uid)

    # ------------------------------------------------------------ scalars ----

    def td_ReadValue(self, name: str, *, timeout: float | None = None) -> float:
        """Read one controller value. The router correlates the VALUE reply by UID,
        so a concurrent scan's CAPTUREs / the always-on TELEM stream can no longer
        be misread as this read's reply -- the waiter receives ONLY this request's
        VALUE (or ERR). No tolerant frame-skipping loop is needed any more.

        `timeout` (seconds) BOUNDS this one read; None inherits `self.timeout`
        (30 s by default). FAIL-SAFE and VERIFY reads MUST pass an explicit short
        bound: inheriting the 30 s default is what made a worst-case safe_park
        ~65 s and a withdraw ~62.5 s (docs/FLEX_INTEGRATION_PLAN.md Sec F.2), i.e.
        the difference between an emergency stop that RETURNS (reporting itself
        unverified) and one that appears to hang. A bounded read that gives up
        raises `BridgeReadTimeout`, which the caller distinguishes from a bridge
        ERR reply."""
        return self._read_scalar(P.cmd_getvalue, name, timeout)

    def get_gv(self, name: str, *, timeout: float | None = None) -> float:
        """Read one Asylum global (GV, e.g. "XLVDTSens") over GETGV. td_ReadValue
        reaches controller channels only, not these globals. NaN when Igor's GV
        read errored; a timeout raises `BridgeReadTimeout`. A pre-GETGV .ipf answers
        with a uid-less ERR the router cannot deliver, so every call costs the full
        timeout (5 s per refresh); service/xy_cal.probe_getgv detects this once at
        connect and disables the refresh."""
        return self._read_scalar(P.cmd_getgv, name, timeout)

    def _read_scalar(self, cmd, name: str, timeout: float | None) -> float:
        """Send one uid-correlated read and return its VALUE reply as a float."""
        import queue as _queue
        bound = self.timeout if timeout is None else float(timeout)
        w = self.router.open_waiter()
        try:
            self.link.send(cmd(w.uid, name))
            verb, rest = w.get(bound)
        except _queue.Empty:
            raise BridgeReadTimeout(
                f"expected VALUE reply for {name!r}, none within "
                f"{bound:g}s") from None
        finally:
            self.router.release(w.uid)
        if verb == P.VALUE:
            return float(rest.strip())
        if verb == P.ERR:
            raise RuntimeError(f"bridge error: {rest.strip()}")
        raise RuntimeError(f"expected VALUE reply for {name!r}, got {verb} {rest!r}")

    def td_WriteValue(self, name: str, value: float) -> int:
        self.link.send(P.cmd_setvalue(name, float(value)))
        return 0

    def read_crosspoint(self, group: str = "Crosspoint") -> dict[str, str]:
        """Read the live instrument crosspoint (an AUTHORITATIVE mode read-back) and
        return {entry: value} of routed input names, e.g. {"InFast": "FilterOut"} in
        contact / {"InFast": "ACDefl"} in AC. Synchronous via the router, exactly like
        td_ReadValue: register a UID waiter, send READXPT, block for the correlated XPT
        reply. Read-only on the Igor side (td_ReadGroup), so it is safe mid-scan.

        SAFETY: NEVER raises. Any failure -- ERR, timeout, closed link, no router (the
        Protocol-only smoke path), a malformed reply -- returns {} so the caller's
        check_mode resolves to "unknown" and the guard falls back conservatively; a
        read failure must never be mistaken for "contact" and downgrade the guard. A
        SHORT timeout (crosspoint reads are cheap) bounds the guard's latency even when
        the bridge is unresponsive."""
        if self.router is None:
            return {}
        timeout = min(self.timeout, 5.0)
        w = self.router.open_waiter()
        try:
            self.link.send(P.cmd_readxpt(w.uid, group))
            verb, rest = w.get(timeout)
        except Exception:                                   # noqa: BLE001
            # queue.Empty (timeout), BridgeClosed, a send/socket error -- all -> {}
            return {}
        finally:
            self.router.release(w.uid)
        if verb == P.XPT:
            try:
                return P.parse_xpt(rest)
            except Exception:                               # noqa: BLE001
                return {}
        return {}                                           # ERR or an unexpected verb

    def check_mode(self) -> str:
        """Infer the instrument imaging mode from the live crosspoint read-back:
        "contact" | "ac" | "unknown". Keyed on the InFast discriminator (rig-validated):
        FilterOut -> contact (contact + current/c-AFM), ACDefl -> ac; anything else
        (read failed / custom / unexpected) -> "unknown", which NEVER downgrades safety
        (the guard falls back to its engaged/not-engaged logic)."""
        infast = str(self.read_crosspoint().get("InFast", "")).strip().lower()
        if infast == "acdefl":
            return "ac"
        if infast == "filterout":
            return "contact"
        return "unknown"

    def td_ReadString(self, name: str) -> str:
        return self._strings.get(name, "")

    def td_WriteString(self, name: str, value: str) -> int:
        # Local cache only. Firing is done via fire(), NOT by writing an Event
        # string, so this never triggers a pass.
        self._strings[name] = value
        return 0

    # ---------------------------------------------------------- servo loops --

    def ir_SetPISLoop(self, index: int, events: str, input_: str, setpoint: float,
                      pgain: float, igain: float, sgain: float, output: str,
                      out_min: float, out_max: float) -> int:
        # The bridge SERVOLOOP carries no sgain -- the Igor side applies a fixed
        # sgain -- but out_min/out_max DO cross the bridge (servoloop-limits fix,
        # 2026-09-16): AFM_SetupServoLoop used to hardcode -10/150 regardless of
        # what the caller asked for, which parked a Cypher withdraw's Height loop
        # at -10 V instead of 0 (see hw_approach.hw_withdraw).
        self.link.send(P.cmd_servoloop(int(index), input_, float(setpoint),
                                       float(pgain), float(igain), output,
                                       float(out_min), float(out_max)))
        return 0

    def ir_StopPISLoop(self, index: int) -> int:
        self.link.send(P.cmd_stoploop(int(index)))
        return 0

    def set_imaging_mode(self, mode: str) -> int:
        """Switch the instrument's imaging mode (contact <-> AC) via the vendor's own
        mode-profile call (SETMODE -> MainPopupFunc; see cmd_setmode). Fire-and-forget,
        send-only, like Show Tip -- the BridgeLink send lock keeps it from interleaving
        with a worker's mid-arm burst. SAFETY-CRITICAL: it reconfigures the Z feedback
        channel + gain sign and does NOT withdraw, so the caller must refuse it while
        the tip is engaged (POST /api/mode's engaged gate)."""
        self.link.send(P.cmd_setmode(mode))
        return 0

    def ramp_output(self, channel: str, dest: float, rate: float = 80.0,
                    max_seconds: float = 2.0) -> int:
        """Rate-limited ramp of one output channel to `dest` (IR_SetRamp). Not used by the
        withdraw, which re-arms loop 2. Fire-and-forget."""
        self.link.send(P.cmd_ramp(channel, float(dest), float(rate),
                                  float(max_seconds)))
        return 0

    # ---- the uniform park surface (docs/INSTRUMENT_MODEL_SPEC.md Sec 10) --------
    # `safe_park` reads, confirms and falls back through ONE object at
    # hub.hardware_arc on BOTH backends; the twin's SoftwareARC exposes the same names.

    def clear_fault(self) -> None:
        """CLEARFAULT: the operator cleared the fault, so Igor's stopped flag and fault flag go
        back to 0 and the next abort calls td_Stop again. Send-only."""
        self.link.send(P.cmd_clearfault())

    def confirm_retract(self) -> None:
        """ABORTCONFIRM: Height read under 1 V, so Igor clears its abort flag and its next
        abort calls td_Stop again. Send-only."""
        self.link.send(P.cmd_abortconfirm())

    def read_abort_fault(self) -> bool:
        """Igor's own fault flag, set when its 5 s backup found Height not retracted and ran
        the vendor Withdraw. Read only; the caller latches `hub.fault`, and the operator's
        clear resets it (`clear_fault`). Bounded like the park reads; raises on a timeout
        or an ERR."""
        return bool(self._read_scalar(lambda uid, _name: P.cmd_abortfault(uid),
                                      "abort fault", _PARK_READ_TIMEOUT_S))

    def read(self, channel: str) -> float:
        """One bounded channel read (Sec 10). BOUNDED (not the 30 s default): this is
        on the fail-safe verify path, where a read that gives up beats one that
        hangs. RIG NOTE (Thursday checklist): the exact readback names for the Z
        output (Height/Output.Z vs the vendor "height" alias) and for
        ZSensor/Deflection are confirmed at the instrument; this reads the name
        it is given."""
        return float(self.td_ReadValue(channel, timeout=_PARK_READ_TIMEOUT_S))

    def vendor_withdraw(self) -> dict:
        """The vendor retract exactly as the Withdraw button does it:
        `DoScanFunc("StopEngageButton")` over COM `Execute2`. There is no `Withdraw()`
        function in the Asylum experiment (rig, 2026-09-25). Raises if no COM handle
        is attached, so `safe_park`'s re-verify then reports the park unverified rather
        than assuming a retract that never happened."""
        retract.vendor_withdraw_over_com(self.com_igor)
        return {"ok": True}

    # ----------------------------------------------------- output waves ------

    def _load_wave(self, values: np.ndarray) -> str:
        """Load a wave under a fresh unique name and return that name. A wave longer
        than WAVE_CHUNK_POINTS is sent as WAVE (first chunk) + WAVEAPP (the rest) so
        no single line is a multi-MB transfer; Igor still ends up with one wave."""
        name = f"ow{self._wave_seq}"
        self._wave_seq += 1
        vals = np.asarray(values, dtype=np.float64)
        if hasattr(self.link, "load_wave"):          # v3: paced by Igor's replies
            self.link.load_wave(name, vals)
            return name
        n = len(vals)
        c = WAVE_CHUNK_POINTS
        self.link.send(P.cmd_wave(name, vals[:c]))       # create (or whole, if small)
        for i in range(c, n, c):
            self.link.send(P.cmd_waveapp(name, vals[i:i + c]))
        return name

    def td_xSetOutWave(self, bank: int, event: str, channel: str,
                       wave: np.ndarray, interpolation: int) -> int:
        name = self._load_wave(check_outwave_len(channel, wave))
        self.link.send(P.cmd_outwave(int(bank), _event(event), channel, name,
                                     interpolation))
        return 0

    def td_xSetOutWavePair(self, bank: int, event: str, ch_a: str, wave_a: np.ndarray,
                           ch_b: str, wave_b: np.ndarray, interpolation: int) -> int:
        # BOTH validated before EITHER is loaded. Checking B after loading A would
        # leave A transferred to Igor as an orphan `ow{N}` wave that nothing ever
        # frees -- a leak on the instrument every time the caller got the pair wrong.
        vals_a = check_outwave_len(ch_a, wave_a)
        vals_b = check_outwave_len(ch_b, wave_b)
        name_a = self._load_wave(vals_a)
        name_b = self._load_wave(vals_b)
        self.link.send(P.cmd_outpair(int(bank), _event(event), ch_a, name_a,
                                     ch_b, name_b, interpolation))
        return 0

    # ------------------------------------------------------ input waves ------

    def td_xSetInWave(self, bank: int, event: str, channel: str,
                      n_points: int, decimation: int) -> int:
        self.link.send(P.cmd_inwave(int(bank), _event(event), channel,
                                    int(n_points), int(decimation)))
        self._armed_in.append((channel, int(n_points)))
        return 0

    def td_xSetInWavePair(self, bank: int, event: str, ch_a: str, ch_b: str,
                          n_points: int, decimation: int) -> int:
        self.link.send(P.cmd_inpair(int(bank), _event(event), ch_a, ch_b,
                                    int(n_points), int(decimation)))
        self._armed_in.append((ch_a, int(n_points)))
        self._armed_in.append((ch_b, int(n_points)))
        return 0

    # -------------------------------------------------------- arm-and-fire ---

    def fire(self, event: str, ticks: int) -> dict[str, np.ndarray]:
        """Fire `event` and return the filled input-wave capture as
        {channel: samples (float64 ndarray)}.

        `ticks` is IGNORED: the controller runs the pass autonomously. The
        completion arrives as zero or more CAPTURE frames (one per armed input
        channel) followed by the terminal DONE.
        """
        name = f"pass{self._fire_seq}"
        self._fire_seq += 1
        # Register the pass under a fresh UID BEFORE firing, so every CAPTURE and
        # the terminal DONE (all tagged with this UID by the bridge) route to this
        # waiter -- a concurrent td_ReadValue can no longer steal the frames.
        # lr_pass() guards the single shared Igor root:AFM_lrUID: FIRE is lr-class,
        # so a second concurrent litho/fire pass fails loudly here rather than
        # clobbering the UID (afm_litho_run.ipf single-global assumption).
        with self.router.lr_pass():
            w = self.router.open_waiter()
            try:
                self.link.send(P.cmd_fire(w.uid, name, _event(event)))
                capture = self._await_completion(w, name)
            finally:
                self.router.release(w.uid)
        self._armed_in = []            # reset for the next pass
        return capture

    # -------------------------------------------------------------- stop -----

    def td_StopOutWaveBank(self, bank: int) -> int:
        return 0

    def td_StopInWaveBank(self, bank: int) -> int:
        return 0

    def abort(self) -> str:
        """Abort the instrument OUT-OF-BAND **and RETRACT**: COM first, TCP ABORT
        fallback (spec Sec 5). Send-only -- it never reads or waits behind the
        router. The router routes the resulting async ABORTED to the in-flight pass
        waiter (which then returns aborted); abort itself does not depend on that.
        Returns the transport used ("com" / "tcp"), for observability.

        Since 2026-09-10 this is stop-AND-verified-withdraw, not stop-and-hold: the
        Igor function it fires is `AFM_SafePark()` (which calls `AFM_Abort()` first,
        so the sequencing is identical) rather than the bare `AFM_Abort()`. See
        `send_abort` and docs/ABORT_SEQUENCE.md.

        When the service has wired `failsafe_park` (hub.bind_hardware), the retract is
        HANDED to the dedicated fail-safe thread rather than fired from whatever
        thread called us -- see `failsafe_park` in __init__ for why that matters. The
        hand-off is non-blocking, so this still returns immediately. Falls back to the
        direct COM/TCP path when no hook is wired (off-instrument, tests, the stub)."""
        hook = self.failsafe_park
        if hook is not None:
            try:
                hook("arc.abort")
                return "watchdog"
            except Exception as exc:                      # noqa: BLE001
                # NEVER let a hand-off failure swallow an abort: fall through to the
                # direct out-of-band path, which is what this method did before.
                self._log(f"abort: watchdog hand-off failed ({exc}); firing directly")
        return send_abort(self.link, self.com_igor, log=self._log)

    def td_Stop(self) -> int:
        """Abort now (out-of-band) -- stop AND retract, see `abort`. No read: the
        ABORTED is routed to the in-flight pass waiter by the router, not drained
        here."""
        self.abort()
        return 0

    def close(self) -> None:
        if self._owns_router and self.router is not None:
            self.router.stop()
        self.link.close()

    # ------------------------------------------------------------ internals --

    def _await_completion(self, waiter, name: str) -> dict[str, np.ndarray]:
        """Consume this pass's frames from `waiter` until DONE/ABORTED, collecting
        the CAPTURE frames that precede the terminal into {channel: float64 ndarray}.
        The router has already correlated every frame to this pass by UID and routed
        TELEM elsewhere, so the loop only sees CAPTURE + the terminal. Raise on ERR
        or a closed link.

        ACCUMULATES: a large capture is chunked by the bridge (AFM_SendCapture,
        kCapChunkPoints) into multiple same-(name,channel) CAPTURE frames; each
        channel's frames are appended in ARRIVAL ORDER, so N frames reassemble into
        one array. A small capture sends its channel once, so this is identical to a
        single assignment there."""
        capture: dict[str, list[float]] = {}
        while True:
            verb, rest = waiter.get(self.timeout)     # BridgeClosed / queue.Empty propagate
            if verb == P.CAPTURE:
                # CAPTURE <name> <channel> v;v;...  (one of possibly several chunks)
                cname, _, tail = rest.partition(" ")
                chan, _, body = tail.partition(" ")
                if cname == name:
                    capture.setdefault(chan, []).extend(
                        float(v) for v in body.split(";") if v != "")
                continue
            got = rest.strip()
            if verb == P.ERR:
                raise RuntimeError(f"bridge error: {got}")
            if verb in (P.DONE, P.ABORTED):
                # The router only delivers this pass's UID, so a terminal is ours
                # (got matches name; an ABORTED after an abort may carry the name too).
                return {c: np.array(v, dtype=np.float64)
                        for c, v in capture.items()}
            # anything else mid-pass is ignored


def _check_proto(startup: dict) -> None:
    """Fail LOUDLY on a protocol-version mismatch. The bridge echoes `proto=<v>` in
    its STARTUP reply; a value that is absent or != our PROTOCOL_VERSION means the
    two ends speak different dialects (the exact silent-desync class this refactor
    removes), so refuse at connect rather than corrupt a run."""
    proto = startup.get("proto")
    try:
        got = int(float(proto))
    except (TypeError, ValueError):
        got = None
    if got != P.PROTOCOL_VERSION:
        raise RuntimeError(
            f"bridge protocol mismatch: driver v{P.PROTOCOL_VERSION}, bridge "
            f"v{proto!r}. The Igor .ipf (afm_litho_run.ipf) and the Python driver "
            "must be the SAME versioned pair -- reinstall the .ipf.")


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
