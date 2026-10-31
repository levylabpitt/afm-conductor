"""Golden-trace harness: a fixed Cypher (or MFP-3D) on the Igor stub, driven through the FLEX door.

`GoldenRig` connects a `FakeIgorBridge` to the process hub the way `app.py`'s hardware
connect does, wraps every COM handle in `TappedCom` on the stub's tap, and hands a scenario a
`FlexDoor`, and when it asks for them an `HttpDoor` (what FLEX has no verb for) and an `Inst`
(the instrument side). A scenario never touches service internals, so the same scenario judges
a later architecture through the same doors.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Self

from hub_fields import owner

from afm_litho import clock
from afm_litho.goldens import expand
from afm_litho.goldens import golden_format as G
from afm_litho.goldens.wire_tap import TappedCom
from afm_litho.hardware.hardware_arc import CYPHER_MICROSCOPE_ID, HardwareARC
from afm_litho.hardware.igor import IgorResult
from afm_litho.hardware import bridge_protocol as P
from afm_litho.hardware.igor_stub import FakeIgorBridge

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixture"
VALUES_FILE = FIXTURE / "igor_values.json"
MODEL_FILE = FIXTURE / "instrument.toml"

# Hub attributes a scenario or the connect can change; restored after every rig.
_HUB_STATE = ("hardware_arc", "hw_cal", "instrument_model", "pattern", "written_objects",
              "engaged", "imaging_mode", "z_loop", "free_air", "control", "fault",
              "_safe_park", "last_park", "litho_frame", "transit", "scan_params", "_scan_row")

# The state a freshly started service has when Igor connects. Earlier tests in the same process
# leave hub state behind (a twin scan's angle sets the litho frame, a tracked imaging mode skips
# the crosspoint read), so every recording starts from this.
_FRESH = {"pattern": None, "written_objects": set(), "engaged": False, "imaging_mode": None,
          "z_loop": {}, "free_air": None, "control": None, "fault": None, "last_park": None,
          "litho_frame": None, "transit": None, "scan_params": {}, "_scan_row": None}


# The service's wall-clock promises (a verb's published bound, the cal-refresh debounce) are
# not what a golden judges: on a busy machine they refuse or skip work that a quiet one does,
# and the recording then differs. A rig widens them for its own lifetime (F39).
_BOUND_SLACK = 20.0
# The host's read/DONE wait. app.py's 30 s is generous for Igor, not for a stub starved of CPU
# while it formats 5 MB of capture text. A scenario recording widens it; the safe-state matrix
# keeps app.py's 30 s.
_APP_LINK_TIMEOUT_S = 30.0
_RECORDING_LINK_TIMEOUT_S = 120.0


# Named exception to "no patching product timing": no door reaches these literals (F39 report).
@contextlib.contextmanager
def _unhurried_service():
    """Widen every verb's published bound (and the loop-marshal ceiling) and turn the
    cal-refresh debounce off, so a loaded machine cannot refuse a verb or skip a GETGV."""
    from afm_litho.service import flex_dispatch, flex_provider, xy_cal
    registry = flex_dispatch.REGISTRY
    saved = (dict(registry), flex_provider.LOOP_CALL_MAX_S, xy_cal.DEBOUNCE_S)
    for name, m in saved[0].items():
        registry[name] = dataclasses.replace(m, bound_s=m.bound_s * _BOUND_SLACK)
    flex_provider.LOOP_CALL_MAX_S = 10.0 * _BOUND_SLACK
    xy_cal.DEBOUNCE_S = 0.0
    try:
        yield
    finally:
        registry.update(saved[0])
        flex_provider.LOOP_CALL_MAX_S, xy_cal.DEBOUNCE_S = saved[1], saved[2]


# The chain driver tests the abort flag right after it queues a row, a few instructions before
# the row can be published. Neither a verb, the state stream nor the stub's record shows that
# test (F41 looked), so a row is given this long to pass it: a bound, not a signal.
_ROW_CHECK_S = 0.25


_PARK_DEFAULT = {"count": 0, "active": False, "last_reason": None, "verified": None,
                 "critical": False, "transport": None, "z_v": None, "at": None}


def _clear_leftovers(hub) -> None:
    """Drop what a scenario that failed mid-run leaves on the shared hub (a park in force, a
    run still open, the repeat switch, a pending fail-safe trigger), so one failure cannot
    fail every later scenario on the worker."""
    hub.host._safe_park = dict(_PARK_DEFAULT)
    hub.host._safe_park_reason = None
    hub.host._safe_park_now.clear()
    hub.control = None
    hub.host._approach_in_flight = 0
    if isinstance(hub.host.scan_repeat, dict):
        hub.host.scan_repeat["enabled"] = False
    rec = hub.runs.in_flight()
    if rec is not None:
        hub.runs.close_if_open(rec.run_id, error="left open by an earlier recording")


def _shallow(v):
    """A copy of a dict or set (the hub mutates some in place), else the object itself."""
    return v.copy() if isinstance(v, (dict, set)) else v


def fixture_hashes() -> dict[str, str]:
    return {p.name: G.file_sha256(p) for p in (MODEL_FILE, VALUES_FILE)}


def platform_for(microscope_id: int) -> str:
    """The model platform for a vendor MicroscopeID: the fixture states the platform once."""
    return "cypher" if int(microscope_id) == CYPHER_MICROSCOPE_ID else "mfp3d"


def load_fixture() -> dict:
    return json.loads(VALUES_FILE.read_text(encoding="utf-8"))


class GoldenStub(FakeIgorBridge):
    """The stub, plus what real Igor has and the stub lacks: `microscope_id` in STARTUP
    (AFM_MacroStartup, afm_igor.ipf:174) and a COM door (`StubCom`)."""

    def __init__(self, values: dict, *, microscope_id: int, **kw):
        super().__init__(values, **kw)
        self.microscope_id = int(microscope_id)
        self.link = None
        self._quiet = threading.local()
        # Holds: a scenario parks the stub at a chosen point so an abort lands exactly there.
        # `held` is set when the stub reaches the hold.
        self.held = threading.Event()
        self._hold_verb: str | None = None
        self._hold_nth = 0
        self.scan_hold_line: int | None = None
        self.chain_hold_chunk: int | None = None
        self._release = threading.Event()
        # OUTPAIR binds that must not move the XSensor/YSensor readback (a transit that lands
        # short of its target).
        self.skip_landings = 0
        self.com_down = False
        # Exceptions that killed a stub worker thread; the rig raises them.
        self.thread_errors: list[BaseException] = []


    def _handle(self, link) -> None:
        self.link = link
        super()._handle(link)

    def _send(self, link, line: str) -> None:
        if line.startswith("STARTUP "):
            line = f"{line}microscope_id={self.microscope_id};"
        super()._send(link, line)

    def _record(self, line: str, verb: str) -> None:
        if not getattr(self._quiet, "on", False):
            super()._record(line, verb)

    def hold_next(self, verb: str, nth: int = 1) -> None:
        """Do not complete the `nth` next `verb` (ARMFIRE, an armed FIRE, LITHOCHAIN) until an
        abort; `held` is set once it has been received."""
        self.held.clear()
        self._hold_verb, self._hold_nth = verb, int(nth)

    def _dispatch(self, line: str, link) -> None:
        verb = line.split(" ")[0]
        if verb == "OUTPAIR" and self.skip_landings > 0:
            self.skip_landings -= 1
            before = {k: self.values.get(k) for k in ("XSensor", "YSensor")}
            super()._dispatch(line, link)
            self.values.update(before)
            return
        if verb == self._hold_verb and not getattr(self._quiet, "on", False):
            self._hold_nth -= 1
            if self._hold_nth <= 0:
                self._hold_verb = None
                saved, self.done_delay = self.done_delay, 3600.0
                try:
                    super()._dispatch(line, link)
                finally:
                    self.done_delay = saved
                if self._timer:
                    self._timer.cancel()
                self.held.set()
                return
        super()._dispatch(line, link)

    def _run_scanchain(self, link, uid, xname, yname, ppl, nlines, err_chan="Deflection",
                       out_ok=True, interp=None):
        """The stub's chain loop, plus a hold after line `scan_hold_line` until an abort."""
        try:
            self._scanchain_loop(link, uid, xname, yname, ppl, nlines, err_chan, out_ok,
                                 interp)
        except BaseException as exc:
            self.thread_errors.append(exc)
            raise

    def _scanchain_loop(self, link, uid, xname, yname, ppl, nlines, err_chan, out_ok,
                        interp):
        a0 = self.aborts
        for ln in range(int(nlines)):
            if self._sc_abort.is_set() or self._stop.is_set():
                self._send(link, P.msg_scandone(uid, -1))
                return
            lo, hi = ln * ppl, (ln + 1) * ppl
            self._bank_arm_scan_line(
                ppl, err_chan, out_ok=out_ok, interp=interp,
                xy=self._slice3((self.waves.get(xname), self.waves.get(yname)), lo, hi))
            self.tap.record_pseudo("@FILL")
            z = self._capture_channel(f"L{ln}", "ZSensor", ppl)
            d = self._capture_channel(f"L{ln}", err_chan, ppl)
            self._send(link, P.msg_capture(uid, f"L{ln}", "ZSensor", z))
            self._send(link, P.msg_capture(uid, f"L{ln}", err_chan, d))
            if self.scan_hold_line is not None and ln == self.scan_hold_line:
                self.held.set()
                while (self.aborts == a0 and not self._release.is_set()
                       and not self._stop.is_set()):
                    time.sleep(0.01)
                if self._release.is_set():
                    # a release lets exactly one more line through, then holds again
                    self._release.clear()
                    self.scan_hold_line = ln + 1
        self._send(link, P.msg_scandone(uid, int(nlines)))

    def _complete_lithochain(self, link, uid, name, bounds, bad_chunks=frozenset(),
                             xyv=None, interp=None):
        """The stub's chunk loop, plus a hold after chunk `chain_hold_chunk` (a seam)."""
        try:
            self._lithochain_loop(link, uid, name, bounds, bad_chunks, xyv, interp)
        except BaseException as exc:
            self.thread_errors.append(exc)
            raise

    def _lithochain_loop(self, link, uid, name, bounds, bad_chunks, xyv, interp):
        a0 = self.aborts
        for k in range(len(bounds) - 1):
            m = bounds[k + 1] - bounds[k]
            cap = m + (32 - m % 32) % 32
            if k:
                self._bank_arm_litho(
                    name, cap, out_ok=(k not in bad_chunks,) * 2, interp=interp,
                    xyv=self._slice3(xyv, bounds[k], bounds[k + 1]))
            self.tap.record_pseudo("@FILL")
            for chan in ("Output.A", "ZSensor", "Deflection", "XSensor", "YSensor"):
                self._send(link, P.msg_capture(
                    uid, name, chan, self._capture_channel(name, chan, cap)))
            if self.chain_hold_chunk is not None and k == self.chain_hold_chunk:
                self.held.set()
                while self.aborts == a0 and not self._stop.is_set():
                    time.sleep(0.01)
                return
            if k == len(bounds) - 2:
                self.bias_zeros.append((name, k))
        self._send(link, P.msg_done(uid, name))

    def run_unrecorded(self, line: str) -> None:
        """Run `line` as Igor would, without a wire record (the caller records it)."""
        self._quiet.on = True
        try:
            self._dispatch(line, self.link)
        finally:
            self._quiet.on = False


# A channel read over COM, as retract.read_height_over_com sends it.
_COM_READ = re.compile(r'fprintf 0, "%\.15g", \(td_ReadValue\("(\w+)"\)\)')


class StubCom:
    """COM `Execute2` on the stub. `AFM_SafePark()` runs what the wire ABORT runs (both call
    AFM_SafePark, afm_litho_run.ipf:826); `DoScanFunc("StopEngageButton")` retracts (the
    vendor Withdraw button); a `td_ReadValue` printed by fprintf answers the stub's value;
    other `DoScanFunc` and `Print` calls succeed with no stub
    effect; anything else raises, so a caller falls back as it would on a rejected command.
    With `stub.com_down` every call raises, as with no COM server."""

    def __init__(self, stub: GoldenStub):
        self.stub = stub

    def execute(self, command: str, *, check: bool = True) -> IgorResult:
        cmd = str(command).strip()
        if self.stub.com_down:
            raise RuntimeError("StubCom: COM is down")
        if cmd == "AFM_SafePark()":
            self.stub.run_unrecorded("ABORT")
        elif cmd == 'DoScanFunc("StopEngageButton")':
            self.stub.vendor_withdraw()
        elif m := _COM_READ.fullmatch(cmd):
            # A channel the stub does not hold reads NaN, never a retracted 0.
            value = float(self.stub.values.get(m.group(1), float("nan")))
            return IgorResult(0, "", "", f"{value:.15g}")
        elif not cmd.startswith(("DoScanFunc(", "Print ")):
            raise RuntimeError(f"StubCom: {cmd!r} is not modelled")
        return IgorResult(0, "", "", "")


class Inst:
    """What a scenario may do to the instrument side: set a value Igor would report, or park
    the stub at a point so an abort lands exactly there. Never the service."""

    def __init__(self, stub: GoldenStub):
        self._stub = stub

    def set_value(self, name: str, value: float) -> None:
        self._stub.values[name] = float(value)

    def set_crosspoint(self, crosspoint: dict) -> None:
        """What the READXPT read-back reports (the mode discriminator): {"InFast": "ACDefl"}
        for an instrument already in AC."""
        self._stub.crosspoint = dict(crosspoint)

    def hold_next(self, verb: str, nth: int = 1) -> None:
        self._stub.hold_next(verb, nth)

    def hold_scan_after_line(self, line: int) -> None:
        self._stub.held.clear()
        self._stub._release.clear()
        self._stub.scan_hold_line = int(line)

    def release_hold(self) -> None:
        """Let a held scan deliver exactly one more line, then hold again."""
        self._stub._release.set()

    def is_held(self) -> bool:
        return self._stub.held.is_set()

    def lateral_refused(self) -> None:
        """Litho in-bank 2 refuses the UserIn2 + Lateral pair: the write arms UserIn2 alone."""
        self._stub.lateral_armed = False

    def com_down(self) -> None:
        """No COM server: every COM call raises (the park falls back to the wire ABORT)."""
        self._stub.com_down = True

    def park_leaves_tip_down(self) -> None:
        """The wire ABORT / COM `AFM_SafePark()` does not bring the readbacks to 0 (the re-arm
        failed to retract); only the vendor Withdraw button does."""
        self._stub.safe_park_zeros = False

    def land_short(self, moves: int = 1) -> None:
        """The next `moves` XY moves leave the XSensor/YSensor readback where it was."""
        self._stub.skip_landings = int(moves)

    def hold_chain_after_chunk(self, chunk: int) -> None:
        self._stub.held.clear()
        self._stub.chain_hold_chunk = int(chunk)

    def wait_held(self, timeout: float = 120.0) -> None:
        if not self._stub.held.wait(timeout):
            if self._stub.thread_errors:
                raise AssertionError(
                    f"a stub worker thread crashed: {self._stub.thread_errors[0]!r}")
            raise AssertionError("the stub never reached the hold")
        self._stub.held.clear()

    def wait_for_verb(self, verb: str, count: int = 1, timeout: float = 30.0) -> None:
        """Block until the stub has received `verb` `count` times in total."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if sum(1 for v in self._stub.transcript if v == verb) >= count:
                return
            time.sleep(0.01)
        raise AssertionError(f"the stub never received {verb} x{count}")


class FlexDoor:
    """The FLEX command door, in process: the same `handle_message` the ZMQ socket calls."""

    def __init__(self, hub):
        self.hub = hub
        self._n = 0
        self.calls: list[str] = []

    def call(self, method: str, params: dict | None = None):
        from afm_litho.service.flex_provider import handle_message
        self._n += 1
        self.calls.append(method)
        raw = json.dumps({"jsonrpc": "2.0", "id": str(self._n), "method": method,
                          "params": params or {}})
        rep = json.loads(handle_message(self.hub, raw))
        if "error" in rep:
            raise AssertionError(f"{method} refused: {rep['error']}")
        return rep["result"]

    def call_refused(self, method: str, params: dict | None = None) -> dict:
        """A call that must be refused: returns the JSON-RPC error object."""
        from afm_litho.service.flex_provider import handle_message
        self._n += 1
        self.calls.append(method)
        raw = json.dumps({"jsonrpc": "2.0", "id": str(self._n), "method": method,
                          "params": params or {}})
        rep = json.loads(handle_message(self.hub, raw))
        if "error" not in rep:
            raise AssertionError(f"{method} was expected to be refused: {rep}")
        return rep["error"]

    def wait_state(self, *done: str, timeout: float = 60.0) -> str:
        end = time.monotonic() + timeout
        state = None
        while time.monotonic() < end:
            state = self.call("getState")
            if state in done:
                return state
            time.sleep(0.05)
        raise AssertionError(f"state stayed {state!r}, wanted one of {done}")

    def wait_scan_row(self, row: int, timeout: float = 60.0) -> None:
        """Block until `getStatus` reports the scan has taken in raster row `row` (its detail
        reads `line <row>`, with `/<rows>` when the scan config is known), then give the chain driver's own abort check on that row
        time to run. The stub's hold says only that the row was SENT; an abort that lands
        before the driver has read it is counted as arriving mid-row, a different run."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            detail = str(self.call("getStatus").get("detail") or "")
            if detail.startswith("line ") and detail[5:].split("/")[0] == str(int(row)):
                time.sleep(_ROW_CHECK_S)
                return
            time.sleep(0.02)
        raise AssertionError(f"the scan never reported raster row {row}")


class HttpDoor:
    """The cockpit's HTTP door, for what FLEX does not carry (Show Tip, scan stop, the relative
    setpoint). A scenario using it is stamped `door: flex+http`."""

    def __init__(self, client):
        self.client = client
        self.calls: list[str] = []

    def post(self, path: str, body: dict | None = None) -> dict:
        self.calls.append(path)
        r = self.client.post(path, json=body or {})
        if r.status_code >= 400:
            raise AssertionError(f"POST {path} -> {r.status_code}: {r.text[:300]}")
        return r.json()

    def get(self, path: str) -> dict:
        self.calls.append(path)
        r = self.client.get(path)
        if r.status_code >= 400:
            raise AssertionError(f"GET {path} -> {r.status_code}: {r.text[:300]}")
        return r.json()


class GoldenRig:
    """One recording. `with GoldenRig(hub) as rig: scenario(rig.door)`, then `rig.golden()`."""

    def __init__(self, hub, scenario: str, *, telem_hz: float = 0.0,
                 microscope_id: int | None = None, link_timeout_s: float = _APP_LINK_TIMEOUT_S):
        self.hub = hub
        self.link_timeout_s = float(link_timeout_s)
        self.scenario = scenario
        fx = load_fixture()
        self.microscope_id = int(fx["microscope_id"] if microscope_id is None
                                 else microscope_id)
        self.values = dict(fx["values"])
        self.stub = GoldenStub(self.values, microscope_id=self.microscope_id,
                               done_delay=0.0, telem_hz=telem_hz)
        self.stub.crosspoint = dict(fx["crosspoint"])
        self.tap = self.stub.tap
        self.door = FlexDoor(hub)
        self.http: HttpDoor | None = None
        self._pump0 = None
        self.inst = Inst(self.stub)
        self.arc: HardwareARC | None = None
        self._saved: dict = {}
        self._wd_com = None
        self._wd = None
        self._inst_scan = None
        self.expansion = None
        self._unhurried = contextlib.ExitStack()

    def __enter__(self) -> Self:
        hub = self.hub
        self._saved = {k: _shallow(getattr(owner(hub, k), k, None)) for k in _HUB_STATE}
        if hub.hw_bridge() is not None or hub.host.bridge_closed.is_set():
            raise AssertionError("golden rig needs a hub with no bridge attached")
        self._inst_scan = hub.host.inst.scan
        self._pump0 = getattr(hub, "_litho_pump_task", None)
        for k, v in _FRESH.items():
            setattr(owner(hub, k), k, _shallow(v))
        _clear_leftovers(hub)
        assert not (hub.host._safe_park or {}).get("active"), \
            "golden rig needs a hub with no park in force"
        hub.host.inst.scan = None
        from afm_litho.service.instrument_model import load_model
        hub.host.instrument_model = load_model(platform=platform_for(self.microscope_id),
                                          startup_cal=None,
                                          config_path=MODEL_FILE, overrides_log=None)
        try:
            self._unhurried.enter_context(_unhurried_service())
            port = self.stub.serve_in_thread()
            self._connect(port)
            self._assert_stub_backend()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _connect(self, port: int) -> None:
        """app.py `_connect_hardware`, step for step, with the COM handles tapped."""
        from afm_litho.service.host import telemetry
        from afm_litho.service.host.safe_park import watchdog_for
        from afm_litho.service.xy_cal import note_xy_cal, probe_getgv
        hub = self.hub
        arc = HardwareARC.connect("127.0.0.1", port, timeout=self.link_timeout_s)
        self.arc = arc
        wd = watchdog_for(hub)
        self._wd, self._wd_com = wd, wd.com_igor
        wd.com_igor = TappedCom(StubCom(self.stub), self.tap, "failsafe")
        arc.router.on_closed = hub.mark_bridge_closed
        arc.com_igor = TappedCom(StubCom(self.stub), self.tap, "com")
        hub.host.hw_cal = arc.start()
        note_xy_cal(hub, "connect")
        probe_getgv(hub, arc)
        telemetry.install_hw_telemetry(arc)
        hub.bind_hardware(arc)

    def _assert_stub_backend(self) -> None:
        hub, arc = self.hub, self.arc
        assert hub.hw_bridge() is arc and isinstance(arc, HardwareARC)
        assert hub.backend_live() == "hardware"
        assert arc.link.sock.getpeername()[1] == self._stub_port(), "arc is not on the stub"
        assert arc.is_cypher == (self.microscope_id == CYPHER_MICROSCOPE_ID)
        assert hub.host.instrument_model.platform.value == platform_for(self.microscope_id)
        assert any(e["verb"] == "STARTUP" for e in self.tap.reads), "no STARTUP reached the stub"
        assert self.door.call("getStatus")["backend"] == "hardware"

    def _stub_port(self) -> int:
        return self.stub.link.sock.getsockname()[1]

    def quiesce(self, settle_s: float = 0.4, timeout: float = 30.0) -> None:
        """Wait until the hub is idle and no actuation has arrived for `settle_s`, then
        make one read round trip so everything sent before it has been dispatched."""
        self.door.wait_state("idle", "parked", "engaged", timeout=timeout)
        end = time.monotonic() + timeout
        n = -1
        while time.monotonic() < end:
            cur = len(self.tap.actuation)
            if cur == n:
                break
            n = cur
            time.sleep(settle_s)
        self.check_stub_threads()
        self.drain_tasks()
        self.arc.td_ReadValue("XSensor")

    def drain_tasks(self, timeout: float = 30.0) -> None:
        """Wait for the run's pump and supervisor tasks to finish their tails (the write pump
        clears `hub.litho_worker` last): a tail that outlives the recording would clear the
        next recording's worker handle. A write that started leaves a new pump task on the
        hub; a refused one leaves none."""
        hub = self.hub
        tasks = [hub.host._scan_task, hub.host._scan_repeat_task]
        pump = getattr(hub, "_litho_pump_task", None)
        if pump is not self._pump0:
            tasks.append(pump)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if all(t is None or t.done() for t in tasks):
                return
            time.sleep(0.02)
        raise AssertionError("a pump or supervisor task did not finish")

    def check_stub_threads(self) -> None:
        """A stub worker thread that crashed fails the recording here, not as a later hang."""
        if self.stub.thread_errors:
            raise AssertionError(
                f"a stub worker thread crashed: {self.stub.thread_errors[0]!r}"
            ) from self.stub.thread_errors[0]

    def __exit__(self, *exc) -> None:
        hub = self.hub
        drain_error = None
        try:
            self.drain_tasks(10.0)
        except Exception as err:                  # noqa: BLE001  (cleanup must still run)
            drain_error = err
        with contextlib.suppress(Exception):
            if hub.control is not None:
                self.door.call("releaseControl", {"token": hub.control.token})
        if self.arc is not None:
            self.arc.router.on_closed = None
            with contextlib.suppress(Exception):
                self.arc.close()
        if self._wd is not None:
            self._wd.com_igor = self._wd_com
        self.stub.stop()
        self._unhurried.close()
        for k, v in self._saved.items():
            with contextlib.suppress(Exception):
                setattr(owner(hub, k), k, v)
        hub.host.inst.scan = self._inst_scan
        _clear_leftovers(hub)
        if exc[0] is None:
            if drain_error is not None:
                raise drain_error
            self.check_stub_threads()

    def golden(self, door: str = "flex") -> dict:
        from .scenarios import KNOWN_DEFECTS
        self.expansion = expand(self.tap.expander_input(), microscope_id=self.microscope_id,
                                gv=self.values,
                                lateral_rc=0 if self.stub.lateral_armed else 100)
        return G.build(self.scenario, self.tap, self.expansion,
                       microscope_id=self.microscope_id, door=door,
                       fixture=fixture_hashes(),
                       known_defects=KNOWN_DEFECTS.get(self.scenario, []))

    def golden_wire(self, door: str = "flex") -> dict:
        """The wire level alone, for the MFP-3D (the end-of-line level is Cypher-first)."""
        from .scenarios import KNOWN_DEFECTS
        return {"format": G.GOLDEN_FORMAT, "scenario": self.scenario, "door": door,
                "microscope_id": self.microscope_id, "ipf_sha256": G.ipf_hashes(),
                "fixture": fixture_hashes(),
                "known_defects": KNOWN_DEFECTS.get(f"{self.scenario}@mfp3d", []),
                "wire": G.wire_level(self.tap)}


MFP3D_MICROSCOPE_ID = 1


def compare_wire(expected: dict, actual: dict) -> list[str]:
    """`golden_format.compare` for a wire-only golden: same keys, same actuation rule, and
    every frozen read still present."""
    problems = []
    for key in ("format", "scenario", "door", "microscope_id", "ipf_sha256", "fixture",
                "known_defects"):
        if expected.get(key) != actual.get(key):
            problems.append(f"{key}: frozen {expected.get(key)!r} != now {actual.get(key)!r}")
    if set(expected) != set(actual):
        problems.append(f"keys: frozen {sorted(expected)} != now {sorted(actual)}")
    exp, act = expected["wire"], actual["wire"]
    if exp["actuation"] != act["actuation"]:
        problems.append("wire.actuation differs:\n" + G._first_diff(
            exp["actuation"], act["actuation"]))
    missing = sorted(set(exp["reads"]) - set(act["reads"]))
    if missing:
        problems.append(f"wire.reads lost {len(missing)}: {missing[:5]}")
    return problems


@contextlib.contextmanager
def _environ(env: dict[str, str]):
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def scenario_args(fn) -> tuple[str, ...]:
    """The doors a scenario asks for, by parameter name: door, inst, http."""
    return tuple(inspect.signature(fn).parameters)


def door_label(fn) -> str:
    return "flex+http" if "http" in scenario_args(fn) else "flex"


def record(hub, scenario: str, *, telem_hz: float | None = None,
           microscope_id: int | None = None, client=None) -> dict:
    """Run one scenario on a fresh rig and return its golden (the wire level alone when
    `microscope_id` is not the fixture's). `client` is the FastAPI TestClient, needed only by
    a scenario that takes the HTTP door."""
    return record_with_waits(hub, scenario, telem_hz=telem_hz, microscope_id=microscope_id,
                             client=client)[0]


def record_with_waits(hub, scenario: str, *, telem_hz: float | None = None,
                      microscope_id: int | None = None, client=None
                      ) -> tuple[dict, list[clock.Wait]]:
    """`record`, plus the waits the product commanded during the run.

    The run is on the testing clock (waits recorded and skipped) unless the scenario's
    OPTIONS name `clock: "testing-time-sensitive"`: an abort, dead-man or interlock landing
    at a chosen moment needs real time, because its setter is another real thread."""
    from .scenarios import OPTIONS, SCENARIOS
    fn = SCENARIOS[scenario]
    opts = OPTIONS.get(scenario, {})
    hz = opts.get("telem_hz", 0.0) if telem_hz is None else telem_hz
    with clock.using(opts.get("clock", clock.TESTING)), _environ(opts.get("env", {})), GoldenRig(
            hub, scenario, telem_hz=hz, microscope_id=microscope_id,
            link_timeout_s=_RECORDING_LINK_TIMEOUT_S) as rig:
        doors = {"door": rig.door, "inst": rig.inst}
        if "http" in scenario_args(fn):
            if client is None:
                raise AssertionError(f"{scenario} needs the HTTP door: pass client=")
            rig.http = doors["http"] = HttpDoor(client)
        fn(**{k: doors[k] for k in scenario_args(fn)})
        rig.quiesce()
        waits = clock.waits()
    label = door_label(fn)
    golden = rig.golden(label) if microscope_id is None else rig.golden_wire(label)
    return golden, waits


def commanded(waits: list[clock.Wait], seconds: float, where: str) -> list[clock.Wait]:
    """The recorded waits of `seconds` asked for by code in `where` (a file name such as
    "lithography_run.py"); a test asserts this is non-empty to show the product asked."""
    return [w for w in waits if w.seconds is not None and abs(w.seconds - seconds) < 1e-9
            and w.caller.split(":")[0].endswith(where)]
