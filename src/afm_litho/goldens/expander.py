"""Expand a v2 wire transcript into the vendor calls Igor executes.

The expander re-implements what `igor/afm_litho_run.ipf` (with its includes `afm_igor.ipf` and
`afm_sockit.ipf`) does with each dispatched line, down to the `td_*` / `ir_*` / `IR_*` / `GV` /
`MainPopupFunc` calls and their arguments. Every emitted call carries the `.ipf` line it comes
from (`Call.site`); `ipf_table.py` holds the per-function table these methods follow.

Input is an ordered transcript of `(verb, args)` where `args` are the wire tokens after the verb.
Pseudo-verbs (leading `@`) stand for what the wire does not carry:

  @FILL [event]  the in-banks bound to `event` (default: the last fired) fill; callbacks run
  @SCANTICK      one AFMscan background tick (only needed with auto_tick=False)
  @LITHOTICK     one AFMlitho background tick (only needed with auto_tick=False)
  @TIPTICK       one AFMtip poll tick (reads only)
  @SAFEPARK      COM Execute2("AFM_SafePark()")
  @DOSCANFUNC    COM Execute2('DoScanFunc("<arg>")'), one opaque vendor call
  @RUNSTART      AFM_LithoRunStart(host, port), run by hand at bridge start
  @RUNEND        AFM_LithoRunEnd(), run by hand at bridge end
  @DRYRUN        AFM_LithoRunDryRun(<0|1>)
  @PRINT         COM `Print ...`: no vendor call
  @COM           any other COM command: unmodelled, always refused
  @COMFAIL       a COM command Igor rejected mid-run: how far it got is unknown, always refused

Unknown verbs, pseudo-verbs and entry shapes raise `ExpanderError`; `strict=False` models Igor's
own reply to an unknown wire verb (an ERR frame) instead.

The expansion starts in the state AFM_LithoRunStart leaves (link up, dispatch and tip poll
running). Waves are held as float32: Igor's `Make` without `/D` is single precision.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

RUN_IPF = "igor/afm_litho_run.ipf"
IGOR_IPF = "igor/afm_igor.ipf"

K_EVENT = "14"                       # kEvent, afm_litho_run.ipf:81
BIAS_IN = "UserIn2"                  # kBiasInChannel, :73: the measured tip bias
LATERAL_IN = "Lateral"               # kLateralInChannel, :77: its litho bank-2 partner
DEFAULT_OUT_MIN = -10.0              # SERVOLOOP without limit tokens, :295
DEFAULT_OUT_MAX = 150.0
SERVO_EVENTS = "Always,Never"        # :584
SERVO_SGAIN = 0                      # :584

WIRE_VERBS = frozenset({
    "STARTUP", "GETVALUE", "GETGV", "SETVALUE", "SERVOLOOP", "STOPLOOP", "RAMP", "WAVE",
    "WAVEAPP", "ARMFIRE", "OUTWAVE", "OUTPAIR", "INWAVE", "INPAIR", "FIRE", "SCANCHAIN",
    "LITHOCHAIN", "SHOWTIP", "SETMODE", "READXPT", "ABORT", "ABORTCONFIRM", "ABORTFAULT", "CLEARFAULT",
    "GETSTR"})
DRY_RUN_VERBS = frozenset({
    "STARTUP", "GETVALUE", "GETGV", "READXPT", "FIRE", "ARMFIRE", "ABORT", "SCANCHAIN",
    "LITHOCHAIN"})
PSEUDO_VERBS = frozenset({
    "@FILL", "@SCANTICK", "@LITHOTICK", "@TIPTICK", "@SAFEPARK", "@DOSCANFUNC", "@RUNSTART",
    "@RUNEND", "@DRYRUN", "@PRINT", "@COM", "@COMFAIL"})


class ExpanderError(Exception):
    """The transcript asks for something the modelled Igor state cannot do."""


def _site(line: int, path: str = RUN_IPF) -> str:
    return f"{path}:{line}"


def _num(v: float) -> str:
    v = float(v)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


def _f32(v: float | None) -> str:
    if v is None:
        return "none"
    if math.isnan(v) or math.isinf(v):
        return _num(v)
    return f"{v:.9g}"


@dataclass(frozen=True)
class Wave:
    """An out-wave argument, summarised: never inlined."""
    name: str
    n: int
    first: float | None
    last: float | None
    sha: str
    missing: bool = False

    def __str__(self) -> str:
        if self.missing:
            return f"wave({self.name},MISSING)"
        return (f"wave({self.name},n={self.n},first={_f32(self.first)},"
                f"last={_f32(self.last)},sha={self.sha})")


@dataclass(frozen=True)
class CapWave:
    """A capture (in-wave) buffer: only its name and length reach the vendor call."""
    name: str
    n: int

    def __str__(self) -> str:
        return f"cap({self.name},n={self.n})"


def _fmt(a) -> str:
    if isinstance(a, str):
        return f'"{a}"'
    if isinstance(a, Wave | CapWave):
        return str(a)
    return _num(a)


@dataclass(frozen=True)
class Call:
    """One vendor call. `site` is the `.ipf` line it runs from; it is not part of equality."""
    fn: str
    args: tuple = ()
    site: str = field(default="", compare=False)

    def __str__(self) -> str:
        return f"{self.fn}({','.join(_fmt(a) for a in self.args)})"


@dataclass(frozen=True)
class Event:
    step: int
    verb: str
    kind: str              # "act" | "read"
    call: Call
    tag: str = ""          # reads: wire | startup | tip | safepark | xpt


@dataclass
class Expansion:
    timeline: list[Event] = field(default_factory=list)
    frames: list[tuple] = field(default_factory=list)

    @property
    def actuation(self) -> list[Call]:
        return [e.call for e in self.timeline if e.kind == "act"]

    @property
    def reads(self) -> list[Call]:
        return [e.call for e in self.timeline if e.kind == "read"]

    def lines(self, kind: str = "act") -> list[str]:
        return [str(e.call) for e in self.timeline if e.kind == kind]


# ------------------------------------------------------------ Igor string semantics --

_NUM_RE = re.compile(
    r"\s*([+-]?(?:inf|nan|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?))", re.IGNORECASE)


def str2num(s: str) -> float:
    """Igor `str2num`: the leading number of `s`, NaN when there is none."""
    m = _NUM_RE.match(s)
    return float(m.group(1)) if m else math.nan


def _items(s: str, sep: str) -> list[str]:
    """Igor list items: a trailing separator does not start an extra (empty) item."""
    if s == "":
        return []
    parts = s.split(sep)
    if parts[-1] == "":
        parts.pop()
    return parts


def _tok(line: str, i: int) -> str:
    """Igor `StringFromList(i, line, " ")`."""
    parts = line.split(" ")
    return parts[i] if i < len(parts) else ""


def _npnts(v: float) -> int:
    """`Make/N=(v)` point count; integer counts assumed (fractions truncate)."""
    if not math.isfinite(v) or v < 0:
        return 0
    return int(v)


def _igor_mod(a: float, b: float) -> float:
    return math.fmod(a, b)


def cap_points(n: float) -> int:
    """AFM_LithoCapPoints (afm_litho_run.ipf:1035-1041): pad up to a multiple of 32."""
    n = max(n, 1)
    return int(n + _igor_mod(32 - _igor_mod(n, 32), 32))


def summarise(name: str, values: np.ndarray | None) -> Wave:
    if values is None:
        return Wave(name, 0, None, None, "", missing=True)
    v = np.asarray(values, dtype="<f4")
    sha = hashlib.sha256(v.tobytes()).hexdigest()[:16]
    first = float(v[0]) if len(v) else None
    last = float(v[-1]) if len(v) else None
    return Wave(name, len(v), first, last, sha)


def parse_line(line: str) -> tuple[str, tuple[str, ...]]:
    """Split one wire line the way AFM_Dispatch does (single-space separated)."""
    parts = line.rstrip("\n").split(" ")
    return parts[0], tuple(parts[1:])


def _entry(i: int, item) -> tuple[str, tuple]:
    """One transcript entry: a raw wire line, or `(verb, args)` with scalar args."""
    if isinstance(item, str):
        return parse_line(item)
    if isinstance(item, (tuple, list)) and 1 <= len(item) <= 2 and isinstance(item[0], str):
        args = item[1] if len(item) == 2 else ()
        if isinstance(args, (str, bytes)) or not isinstance(args, Iterable):
            raise ExpanderError(f"entry {i}: args must be a sequence of tokens, got {args!r}")
        args = tuple(args)
        if all(isinstance(a, (str, int, float)) for a in args):
            return item[0], args
    raise ExpanderError(
        f"entry {i}: expected a wire line or (verb, args) of scalar tokens, got "
        f"{type(item).__name__} {str(item)[:80]}. Tap summaries cannot be expanded; "
        f"feed tap.expander_input()")


def _as_token(a) -> str:
    if isinstance(a, str):
        return a
    if isinstance(a, int):
        return str(a)
    return f"{float(a):.12g}"


# ------------------------------------------------------------------- the expander --

class IgorExpander:
    """Replays a wire transcript against a model of the bridge `.ipf` state.

    State inputs the wire does not decide:
      microscope_id  GV("MicroscopeID") (Cypher = 4)
      vcap_rc        return code of the UserIn2 measured-bias td_xSetInWave arm made alone
                     (the vErr branches after a refused litho pair, and the scan bank);
                     an int, or a callable
                     (site, index) -> int with site in {"ARMFIRE", "SCANCHAIN", "LITHOCHAIN"}
                     and index the line or chunk
      lateral_rc     return code of the litho bank-2 td_xSetInWavePair (lErr): nonzero refuses
                     the pair and UserIn2 is armed by itself; same int-or-callable shape
      gv             other Asylum globals read by STARTUP / GETGV (default 0)
      read_values    td_ReadValue results (default 0)
      crosspoint     td_ReadGroup result for READXPT as [(label, value)]; None = no data
      auto_tick      run the scan/litho background tick right after each @FILL
      strict         raise on an unknown wire verb or a vendor call on an unloaded wave
                     (default); False models Igor (an ERR reply, a MISSING wave reference)
    """

    def __init__(self, *, microscope_id: int | None = None,
                 vcap_rc: int | Callable[[str, int], int] = 0,
                 lateral_rc: int | Callable[[str, int], int] = 0,
                 gv: dict[str, float] | None = None,
                 read_values: dict[str, float] | None = None,
                 crosspoint: list[tuple[str, str]] | None = None,
                 auto_tick: bool = True, strict: bool = True) -> None:
        self.microscope_id = microscope_id
        self.strict = strict
        self._vcap_rc = vcap_rc
        self._lateral_rc = lateral_rc
        self._gv = dict(gv or {})
        self._reads = dict(read_values or {})
        self._xpt = crosspoint
        self.auto_tick = auto_tick
        self.out = Expansion()
        self._step = -1
        self._verb = ""
        self._waves: dict[str, tuple[str, np.ndarray]] = {}
        self._in_banks: dict[int, tuple[str, str]] = {}
        self._fired: str | None = None
        self.g: dict[str, object] = {}
        self.scan_task = False
        self.litho_task = False
        self._run_start()

    # -------------------------------------------------------------- emission --

    def _act(self, fn: str, args: tuple, site: str) -> None:
        self.out.timeline.append(Event(self._step, self._verb, "act", Call(fn, args, site)))
        self._bank_model(fn, args)

    def _read(self, fn: str, args: tuple, site: str, tag: str) -> float:
        self.out.timeline.append(
            Event(self._step, self._verb, "read", Call(fn, args, site), tag))
        if fn == "GV":
            return self._gv_value(args[0])
        if fn == "td_ReadValue":
            return float(self._reads.get(args[0], 0.0))
        return 0.0

    def _frame(self, *f) -> None:
        self.out.frames.append(tuple(f))

    def _bank_model(self, fn: str, args: tuple) -> None:
        """Which in-bank callbacks are registered on which event (for @FILL)."""
        if fn in ("td_xSetInWave", "td_xSetInWavePair"):
            bank = args[0]
            if isinstance(bank, float) and not math.isfinite(bank):
                return
            self._in_banks[int(bank)] = (str(args[1]), args[-2])
        elif fn == "ir_StopInWaveBank":
            self._in_banks.pop(int(args[0]), None)
        elif fn == "td_Stop":
            # ASSUMPTION (unverified; 0.8b): td_Stop drops every in-bank callback registration.
            self._in_banks.clear()
        elif fn == "td_WriteString" and args[1] == "Once" and args[0].startswith("Event."):
            self._fired = args[0][len("Event."):]
            # root:AFM_passArmed, set at every fire site (out-waves armed or event fired)
            self.g["passArmed"] = 1

    def _gv_value(self, name: str) -> float:
        if name == "MicroscopeID":
            return math.nan if self.microscope_id is None else float(self.microscope_id)
        return float(self._gv.get(name, 0.0))

    def _vcap(self, site: str, index: int) -> int:
        rc = self._vcap_rc
        return int(rc(site, index)) if callable(rc) else int(rc)

    def _lateral(self, site: str, index: int) -> int:
        rc = self._lateral_rc
        return int(rc(site, index)) if callable(rc) else int(rc)

    def _arm_bank2_pair(self, site: str, index: int, v: CapWave, lat: CapWave, decim: float,
                        lines: tuple[int, int, int, int]) -> tuple[int, int]:
        """The litho bank-2 fail-safe ladder: the UserIn2 + Lateral pair, else UserIn2 alone,
        else nothing. `lines` are the .ipf lines of the pair, the first bank release, the
        single arm and the second release. Returns (UserIn2 armed, Lateral armed)."""
        pair_line, stop1, single, stop2 = lines
        pair_rc = self._lateral(site, index)      # lErr is the pair call's rc alone
        self._act("td_xSetInWavePair", (2, K_EVENT, BIAS_IN, v, LATERAL_IN, lat, "", decim),
                  _site(pair_line))
        if pair_rc == 0:
            return 1, 1
        self._act("ir_StopInWaveBank", (2,), _site(stop1))
        rc = self._vcap(site, index)
        self._act("td_xSetInWave", (2, K_EVENT, BIAS_IN, v, "", decim), _site(single))
        if rc != 0:
            self._act("ir_StopInWaveBank", (2,), _site(stop2))
        return int(rc == 0), 0

    # ----------------------------------------------------------------- waves --

    def _store_wave(self, name: str, body: str) -> int:
        if body.startswith("wave("):
            raise ExpanderError(f"WAVE {name}: a summarised wave cannot be expanded; feed the "
                                f"raw lines (tap.expander_input())")
        vals = np.array([str2num(t) for t in _items(body, ";")], dtype=np.float32)
        self._waves[name.casefold()] = (name, vals)
        return len(vals)

    def _append_wave(self, name: str, body: str) -> int:
        if name.casefold() not in self._waves:
            return self._store_wave(name, body)
        self._store_wave("AFM_appTmp", body)
        tmp = self._waves["afm_apptmp"][1]
        nm, w = self._waves[name.casefold()]
        self._waves[name.casefold()] = (nm, np.concatenate([w, tmp]))
        return len(w) + len(tmp)

    def _wave(self, name: str) -> np.ndarray | None:
        hit = self._waves.get(name.casefold())
        return None if hit is None else hit[1]

    def _ref(self, name: str) -> Wave:
        w = self._wave(name)
        if w is None and self.strict:
            raise ExpanderError(f"a vendor call references wave {name!r}, which no WAVE loaded")
        return summarise(name, w)

    def _slice(self, src: np.ndarray, a: float, n: int) -> np.ndarray:
        """`dst = src[a + p]` over n points; out-of-range indices clip to the ends."""
        if len(src) == 0:
            return np.full(n, np.nan, dtype=np.float32)
        idx = np.clip(np.arange(n) + int(a), 0, len(src) - 1)
        return src[idx].astype(np.float32)

    def _make_slice(self, name: str, src: np.ndarray | None, a: float, n: int) -> Wave:
        vals = self._slice(np.zeros(0, np.float32) if src is None else src, a, n)
        self._waves[name.casefold()] = (name, vals)
        return summarise(name, vals)

    # ------------------------------------------------------------- transcript --

    def feed(self, verb: str, args: Iterable = ()) -> None:
        self._step += 1
        self._verb = verb
        toks = tuple(_as_token(a) for a in args)
        if verb.startswith("@"):
            self._pseudo(verb, toks)
            return
        if not self.g["dispatch_running"]:
            raise ExpanderError(f"{verb}: the AFMdispatch task is stopped (after @RUNEND)")
        self._dispatch(" ".join((verb, *toks)))

    def run(self, transcript: Iterable) -> Expansion:
        for i, item in enumerate(transcript):
            self.feed(*_entry(i, item))
        return self.out

    def _pseudo(self, verb: str, toks: tuple[str, ...]) -> None:
        if verb == "@FILL":
            self._fill(toks[0] if toks else None)
        elif verb == "@SCANTICK":
            self._scan_tick()
        elif verb == "@LITHOTICK":
            self._litho_tick()
        elif verb == "@TIPTICK":
            if self.g["tip_running"]:
                self._tip_tick()
        elif verb == "@SAFEPARK":
            self._safe_park()
        elif verb == "@DOSCANFUNC":
            name = toks[0] if toks else "StopEngageButton"
            self._act("DoScanFunc", (name,), "vendor DoScanFunc via COM (below the line)")
        elif verb == "@RUNSTART":
            self._run_start()
        elif verb == "@RUNEND":
            self._run_end()
        elif verb == "@DRYRUN":
            self.g["dry"] = bool(toks and str2num(toks[0]) != 0)
        elif verb == "@PRINT":
            pass                        # Igor prints to its history; no vendor call
        elif verb == "@COM":
            raise ExpanderError(f"unmodelled COM command: {' '.join(toks)}")
        elif verb == "@COMFAIL":
            raise ExpanderError(f"a COM command failed in Igor, after an unknown part of it "
                                f"ran: {' '.join(toks)}")
        else:
            raise ExpanderError(f"unknown pseudo-verb {verb}")

    def _fill(self, event: str | None = None) -> None:
        ev = self._fired if event is None else event
        banks = sorted(b for b, (e, _) in self._in_banks.items() if e == ev)
        if ev is None or not banks:
            raise ExpanderError(f"@FILL: no in-bank is armed on event {ev}")
        callbacks = [self._in_banks.pop(b)[1] for b in banks]
        table = {"AFM_Done()": self._afm_done, "AFM_LithoDone()": self._litho_done,
                 "AFM_ScanAdvance()": self._scan_advance,
                 "AFM_LithoAdvance()": self._litho_advance}
        for cb in callbacks:
            if cb:
                if cb not in table:
                    raise ExpanderError(f"@FILL: unmodelled callback {cb}")
                table[cb]()
        if self.auto_tick:
            self._scan_tick()
            self._litho_tick()

    # --------------------------------------------------------- run lifecycle --

    def _run_start(self) -> None:
        """AFM_LithoRunStart (:103-139), assuming the SOCKIT connect succeeds."""
        self.g.update(sock=True, lrName="", lrUID="", scUID="", capChans=[], capWaves=[],
                      doneSent=0, dry=False, dispatch_running=True, tip_running=True)

    def _run_end(self) -> None:
        """AFM_LithoRunEnd (:141-150). AFMscan / AFMlitho are not stopped here."""
        self.g["dispatch_running"] = False
        self.g["tip_running"] = False
        self._set_value("Output.A", 0, _site(554), via=144)

    # ---------------------------------------------------------------- dispatch --

    def _dispatch(self, line: str) -> None:
        """AFM_Dispatch (:210-379)."""
        verb = _tok(line, 0)
        t = lambda i: _tok(line, i)
        n = lambda i: str2num(_tok(line, i))
        if self.g["dry"]:
            self._dispatch_dry(verb, line)
            return
        if verb == "STARTUP":
            self._macro_startup()
            self._frame("STARTUP", t(1))
        elif verb == "GETVALUE":
            self._read("td_ReadValue", (t(2),), _site(560), "wire")
            self._frame("VALUE", t(1))
        elif verb == "GETGV":
            self._read("GV", (t(2),), _site(568), "wire")
            self._frame("VALUE", t(1))
        elif verb == "SETVALUE":
            self._set_value(t(1), n(2), _site(554), via=284)
        elif verb == "SERVOLOOP":
            lo, hi = DEFAULT_OUT_MIN, DEFAULT_OUT_MAX
            if len(_items(line, " ")) >= 9:
                lo, hi = n(7), n(8)
            self._act("ir_SetPISLoop", (n(1), SERVO_EVENTS, t(2), n(3), n(4), n(5),
                                        SERVO_SGAIN, t(6), lo, hi), _site(584))
        elif verb == "STOPLOOP":
            self._act("ir_StopPISLoop", (n(1),), _site(594))
        elif verb == "RAMP":
            self._ramp(t(1), n(2), n(3), n(4))
        elif verb == "WAVE":
            self._store_wave(t(1), t(2))
        elif verb == "WAVEAPP":
            self._append_wave(t(1), t(2))
        elif verb == "ARMFIRE":
            self.g["lrUID"] = t(1)
            self._armfire(t(2), n(3), n(4), n(5))
        elif verb == "OUTWAVE":
            self._act("td_xSetOutWave", (n(1), t(2), t(3), self._ref(t(4)), n(5)), _site(446))
        elif verb == "OUTPAIR":
            self._act("td_xSetOutWavePair", (n(1), t(2), t(3), self._ref(t(4)), t(5),
                                             self._ref(t(6)), n(7)), _site(455))
        elif verb == "INWAVE":
            self._arm_in_wave(n(1), t(2), t(3), n(4), n(5))
        elif verb == "INPAIR":
            self._arm_in_pair(n(1), t(2), t(3), t(4), n(5), n(6))
        elif verb == "FIRE":
            self.g["lrUID"] = t(1)
            self._fire_event(t(2), t(3))
        elif verb == "SCANCHAIN":
            self.g["scUID"] = t(1)
            err = t(8) or "Deflection"
            self._scan_chain_start(t(2), t(3), n(4), n(5), n(6), n(7), err)
        elif verb == "LITHOCHAIN":
            self.g["lrUID"] = t(1)
            self._litho_chain_start(t(2), t(3), t(4), t(5), t(6), n(7), n(8))
        elif verb == "SHOWTIP":
            if t(1) == "ON" and self.g["sock"]:
                self.g["tip_running"] = True
        elif verb == "SETMODE":
            if t(1) == "AC":
                self._act("MainPopupFunc", ("ImagingModePopup_0", 2, "AC Mode"), _site(1301))
            else:
                self._act("MainPopupFunc", ("ImagingModePopup_0", 1, "Contact"), _site(1303))
        elif verb == "READXPT":
            self._read_xpt(t(1), t(2))
        elif verb == "ABORT":
            self._safe_park()
        elif verb == "ABORTCONFIRM":
            self._abort_confirmed()
        elif verb == "ABORTFAULT":
            self._frame("VALUE", t(1))                  # AFM_ReadAbortFault (:811-814), read only
        elif verb == "CLEARFAULT":
            self._clear_fault()
        elif verb == "GETSTR":                 # reached through default: -> AFM_DispatchOther
            self._read("td_ReadString", (t(2),), _site(1490), "wire")
            self._frame("XPT", t(1))
        else:
            self._unknown(verb)

    def _dispatch_dry(self, verb: str, line: str) -> None:
        """The dry-run branch (:233-267): reads answer, nothing is commanded."""
        uid = _tok(line, 1)
        if verb == "STARTUP":
            self._macro_startup()
            self._frame("STARTUP", uid)
        elif verb == "GETVALUE":
            self._read("td_ReadValue", (_tok(line, 2),), _site(560), "wire")
            self._frame("VALUE", uid)
        elif verb == "GETGV":
            self._read("GV", (_tok(line, 2),), _site(568), "wire")
            self._frame("VALUE", uid)
        elif verb == "READXPT":
            self._read_xpt(uid, _tok(line, 2))
        elif verb in ("FIRE", "ARMFIRE", "LITHOCHAIN"):
            self._frame("DONE", uid, _tok(line, 2))
        elif verb == "ABORT":
            self._frame("ABORTED", "", "")
        elif verb == "SCANCHAIN":
            self._frame("SCANDONE", uid, 0)
        elif verb not in WIRE_VERBS:
            self._unknown(verb)

    def _unknown(self, verb: str) -> None:
        if self.strict:
            raise ExpanderError(f"unknown wire verb {verb!r}")
        self._frame("ERR", "-", f"unknown verb: {verb}")

    # ------------------------------------------------------------- primitives --

    def _set_value(self, param: str, value: float, site: str, via: int | None = None) -> None:
        if via is not None:
            site = f"{site} (via :{via})"
        self._act("td_WriteValue", (param, value), site)

    def _ramp(self, chan: str, dest: float, rate: float, maxsec: float) -> None:
        """AFM_RampOutput (:599-605): note the argument order IR_SetRamp takes."""
        self.g["passArmed"] = 1
        self._act("IR_SetRamp", (maxsec, chan, rate, dest, "", 0, 0, "", 0, 0, ""), _site(603))

    def _cap_name(self, channel: str) -> str:
        return ("cap_" + channel).replace(".", "_").replace("%", "_")

    def _arm_in_wave(self, bank, event, ch, cap_len, decim) -> None:
        wn = self._cap_name(ch)
        self.g["capChans"].append(ch)
        self.g["capWaves"].append(wn)
        (cw,) = self._make_caps((wn,), _npnts(cap_len))
        self._act("td_xSetInWave", (bank, event, ch, cw, "AFM_Done()", decim), _site(487))

    def _arm_in_pair(self, bank, event, ch_a, ch_b, cap_len, decim) -> None:
        wa, wb = self._cap_name(ch_a), self._cap_name(ch_b)
        self.g["capChans"] += [ch_a, ch_b]
        self.g["capWaves"] += [wa, wb]
        ca, cb = self._make_caps((wa, wb), _npnts(cap_len))
        self._act("td_xSetInWavePair", (bank, event, ch_a, ca, ch_b, cb, "AFM_Done()", decim),
                  _site(500))

    def _fire_event(self, name: str, event: str) -> None:
        """AFM_FireEvent (:505-520): with no capture recorded, AFM_Done runs at once."""
        self.g["lrName"] = name
        self.g["doneSent"] = 0
        self._act("td_WriteString", ("Event." + event, "Once"), _site(509))
        if not self.g["capChans"]:
            self._afm_done()

    def _afm_done(self) -> None:
        """AFM_Done (:526-546), guarded so only the first filling bank does the work."""
        if self.g["doneSent"]:
            return
        self.g["doneSent"] = 1
        self._set_value("Output.A", 0, _site(554), via=537)
        for ch, wn in zip(self.g["capChans"], self.g["capWaves"], strict=True):
            w = self._wave(wn)
            self._frame("CAPTURE", self.g["lrUID"], self.g["lrName"], ch,
                        0 if w is None else len(w))
        self._frame("DONE", self.g["lrUID"], self.g["lrName"])
        self.g["capChans"] = []
        self.g["capWaves"] = []

    def _make_caps(self, names: tuple[str, ...], n: int) -> list[CapWave]:
        for nm in names:
            self._waves[nm.casefold()] = (nm, np.zeros(n, dtype=np.float32))
        return [CapWave(nm, n) for nm in names]

    # ------------------------------------------------------------ litho single --

    def _armfire(self, name: str, interp: float, cap_len: float, decim: float) -> None:
        """AFM_ArmAndFireLitho (:626-674). Drives fixed waves XLitho/YLitho/VLitho."""
        zc, dc, xc, yc, vc, lc = self._make_caps(
            ("ZcapLitho", "DcapLitho", "XcapLitho", "YcapLitho", "VcapLitho", "LcapLitho"),
            _npnts(cap_len))
        self._act("ir_StopOutWaveBank", (0,), _site(634))
        self._act("ir_StopOutWaveBank", (2,), _site(635))
        self._act("ir_StopInWaveBank", (0,), _site(636))
        self._act("ir_StopInWaveBank", (1,), _site(637))
        self._act("ir_StopInWaveBank", (2,), _site(638))
        self._act("td_xSetOutWavePair", (0, K_EVENT, "PIDSLoop.0.Setpoint", self._ref("XLitho"),
                                         "PIDSLoop.1.Setpoint", self._ref("YLitho"), interp),
                  _site(639))
        self._act("td_xSetOutWave", (2, K_EVENT, "Output.A", self._ref("VLitho"), interp),
                  _site(641))
        self._act("td_xSetInWavePair", (0, K_EVENT, "ZSensor", zc, "Deflection", dc,
                                        "AFM_LithoDone()", decim), _site(643))
        self._act("td_xSetInWavePair", (1, K_EVENT, "XSensor", xc, "YSensor", yc, "", decim),
                  _site(645))
        self.g["VcapArmed"], self.g["LcapArmed"] = self._arm_bank2_pair(
            "ARMFIRE", 0, vc, lc, decim, (655, 659, 661, 664))
        self.g["lrName"] = name
        self._act("td_WriteString", ("Event." + K_EVENT, "Once"), _site(671))

    def _send_capture(self, uid: str, name: str, channel: str, wave: str) -> None:
        w = self._wave(wave)
        if w is not None:
            self._frame("CAPTURE", uid, name, channel, len(w))

    def _litho_done(self) -> None:
        """AFM_LithoDone (:721-740)."""
        uid, name = self.g["lrUID"], self.g["lrName"]
        self._set_value("Output.A", 0, _site(554), via=725)
        for ch, wn in (("ZSensor", "ZcapLitho"), ("Deflection", "DcapLitho"),
                       ("XSensor", "XcapLitho"), ("YSensor", "YcapLitho")):
            self._send_capture(uid, name, ch, wn)
        if self.g.get("VcapArmed"):
            self._send_capture(uid, name, BIAS_IN, "VcapLitho")
        if self.g.get("LcapArmed"):
            self._send_capture(uid, name, LATERAL_IN, "LcapLitho")
        self._frame("DONE", uid, name)

    # ----------------------------------------------------------------- abort --

    def _abort(self) -> None:
        """AFM_Abort (:747-786): td_Stop while root:AFM_abortStopped is clear or a pass armed
        since the last td_Stop (AFM_PassMayPlay); AFM_StopRan then sets the flag."""
        self.g.update(scAbort=1, scPending=0, ltAbort=1, ltPending=0)
        if not self.g.get("abortStopped") or self.g.get("passArmed"):
            self._act("td_Stop", (), _site(756))
            self.g.update(abortStopped=1, passArmed=0)
        self._set_value("Output.A", 0, _site(554), via=759)
        if not self.g["sock"]:
            return
        if self.g.get("scActive"):
            self.g["scActive"] = 0
            self._frame("SCANDONE", self.g["scUID"], -1)
            return
        if self.g.get("ltActive"):
            self.g["ltActive"] = 0
        self._frame("ABORTED", self.g["lrUID"], self.g["lrName"])

    def _safe_park(self) -> None:
        """AFM_SafePark (:836-843): AFM_Abort, the same re-arm of loop 2 on every platform,
        then the 5 s AFMabortBackup task armed (AFM_ArmAbortBackup, :1446-1451)."""
        self._abort()
        self._act("ir_SetPISLoop", (2, SERVO_EVENTS, "Deflection", -10, 0, 1000, 0,
                                    "Height", 0, 150), _site(839))
        self.g["abortBackup"] = True

    def _abort_confirmed(self) -> None:
        """AFM_AbortConfirmed (:803-808): clear the flag, stop the backup task."""
        self.g["abortStopped"] = 0
        self.g["abortBackup"] = False

    def _clear_fault(self) -> None:
        """AFM_ClearFault (:817-823): both abort flags to 0, stop the backup task."""
        self.g.update(abortStopped=0, abortFault=0, abortBackup=False)

    # ------------------------------------------------------------ chained scan --

    def _scan_chain_start(self, xname, yname, ppl, nlines, interp, decim, err) -> None:
        """AFM_ScanChainStart (:856-894)."""
        if self._wave(xname) is None or self._wave(yname) is None:
            self._frame("ERR", self.g["scUID"] or "-", "SCANCHAIN missing wave")
            return
        self.g.update(scErrChan=err, scXName=xname, scYName=yname, scPPL=ppl,
                      scLines=nlines, scInterp=interp, scDecim=decim, scLine=0, scAbort=0,
                      scActive=1, scPending=0)
        self.scan_task = True
        self._scan_arm_line(0)

    def _scan_arm_line(self, ln: int) -> None:
        """AFM_ScanArmLine (:898-948). Capture length is ppl (not padded to 32)."""
        g = self.g
        ppl, interp, decim = g["scPPL"], g["scInterp"], g["scDecim"]
        n = _npnts(ppl)
        a = ln * ppl
        lx = self._make_slice("AFM_scLX", self._wave(g["scXName"]), a, n)
        ly = self._make_slice("AFM_scLY", self._wave(g["scYName"]), a, n)
        cz, cd, cv = self._make_caps(("AFM_scCZ", "AFM_scCD", "AFM_scCV"), n)
        self._act("ir_StopOutWaveBank", (0,), _site(921))
        self._act("ir_StopInWaveBank", (0,), _site(922))
        self._act("ir_StopInWaveBank", (1,), _site(923))
        self._act("td_xSetOutWavePair", (0, K_EVENT, "PIDSLoop.0.Setpoint", lx,
                                         "PIDSLoop.1.Setpoint", ly, interp), _site(924))
        self._act("td_xSetInWavePair", (0, K_EVENT, "ZSensor", cz, g["scErrChan"], cd,
                                        "AFM_ScanAdvance()", decim), _site(930))
        rc = self._vcap("SCANCHAIN", ln)
        self._act("td_xSetInWave", (1, K_EVENT, BIAS_IN, cv, "", decim), _site(938))
        g["scVArmed"] = int(rc == 0)
        if rc != 0:
            self._act("ir_StopInWaveBank", (1,), _site(941))
        self._act("td_WriteString", ("Event." + K_EVENT, "Once"), _site(946))

    def _scan_advance(self) -> None:
        """AFM_ScanAdvance (:954-985): stream the line, then finish or request the next."""
        g = self.g
        label = f"L{g['scLine']}"
        if g.get("scVArmed"):
            self._send_capture(g["scUID"], label, BIAS_IN, "AFM_scCV")
        self._send_capture(g["scUID"], label, "ZSensor", "AFM_scCZ")
        self._send_capture(g["scUID"], label, g["scErrChan"], "AFM_scCD")
        g["scLine"] += 1
        if g["scAbort"] or g["scLine"] >= g["scLines"]:
            if g.get("scActive"):
                g["scActive"] = 0
                self._frame("SCANDONE", g["scUID"], g["scLine"])
            g["scPending"] = 0
        else:
            g["scPending"] = 1

    def _scan_tick(self) -> None:
        """AFM_ScanTick (:991-1012)."""
        if not self.scan_task:
            return
        g = self.g
        if "scLine" not in g:
            self.scan_task = False
            return
        if not g["scActive"] or g.get("scAbort"):
            g["scPending"] = 0
            self.scan_task = False
            return
        if not g["scPending"]:
            return
        g["scPending"] = 0
        self._scan_arm_line(g["scLine"])

    # ------------------------------------------------------------- litho chain --

    def _litho_chain_start(self, name, xname, yname, vname, bname, interp, decim) -> None:
        """AFM_LithoChainStart (:1044-1079)."""
        if any(self._wave(w) is None for w in (xname, yname, vname, bname)):
            self._frame("ERR", self.g["lrUID"] or "-", "LITHOCHAIN missing wave")
            return
        self.g.update(ltName=name, lrName=name, ltXName=xname, ltYName=yname, ltVName=vname,
                      ltBName=bname, ltInterp=interp, ltDecim=decim,
                      ltNChunks=len(self._wave(bname)) - 1, ltChunk=0, ltAbort=0,
                      ltActive=1, ltPending=0)
        self.litho_task = True
        self._litho_arm_chunk(0)

    def _litho_arm_chunk(self, k: int) -> None:
        """AFM_LithoArmChunk (:1088-1161): slice [bounds[k], bounds[k+1]) of all three."""
        g = self.g
        bw = self._wave(g["ltBName"])
        a = float(bw[min(k, len(bw) - 1)])
        b = float(bw[min(k + 1, len(bw) - 1)])
        m = b - a
        cap = cap_points(m)
        n = _npnts(m)
        lx = self._make_slice("AFM_ltLX", self._wave(g["ltXName"]), a, n)
        ly = self._make_slice("AFM_ltLY", self._wave(g["ltYName"]), a, n)
        lv = self._make_slice("AFM_ltLV", self._wave(g["ltVName"]), a, n)
        zc, dc, xc, yc, vc, lc = self._make_caps(
            ("AFM_ltZcap", "AFM_ltDcap", "AFM_ltXcap", "AFM_ltYcap", "AFM_ltVcap",
             "AFM_ltLcap"), cap)
        interp, decim = g["ltInterp"], g["ltDecim"]
        self._act("ir_StopOutWaveBank", (0,), _site(1120))
        self._act("ir_StopOutWaveBank", (2,), _site(1121))
        self._act("ir_StopInWaveBank", (0,), _site(1122))
        self._act("ir_StopInWaveBank", (1,), _site(1123))
        self._act("ir_StopInWaveBank", (2,), _site(1124))
        self._act("td_xSetOutWavePair", (0, K_EVENT, "PIDSLoop.0.Setpoint", lx,
                                         "PIDSLoop.1.Setpoint", ly, interp), _site(1125))
        self._act("td_xSetOutWave", (2, K_EVENT, "Output.A", lv, interp), _site(1127))
        self._act("td_xSetInWavePair", (0, K_EVENT, "ZSensor", zc, "Deflection", dc,
                                        "AFM_LithoAdvance()", decim), _site(1129))
        self._act("td_xSetInWavePair", (1, K_EVENT, "XSensor", xc, "YSensor", yc, "", decim),
                  _site(1131))
        g["ltVArmed"], g["ltLArmed"] = self._arm_bank2_pair(
            "LITHOCHAIN", k, vc, lc, decim, (1139, 1143, 1147, 1150))
        self._act("td_WriteString", ("Event." + K_EVENT, "Once"), _site(1158))

    def _litho_advance(self) -> None:
        """AFM_LithoAdvance (:1168-1203): Output.A is zeroed only at the true end."""
        g = self.g
        uid, name = g["lrUID"], g["ltName"]
        if g.get("ltVArmed"):
            self._send_capture(uid, name, BIAS_IN, "AFM_ltVcap")
        if g.get("ltLArmed"):
            self._send_capture(uid, name, LATERAL_IN, "AFM_ltLcap")
        for ch, wn in (("ZSensor", "AFM_ltZcap"), ("Deflection", "AFM_ltDcap"),
                       ("XSensor", "AFM_ltXcap"), ("YSensor", "AFM_ltYcap")):
            self._send_capture(uid, name, ch, wn)
        g["ltChunk"] += 1
        if g["ltAbort"] or g["ltChunk"] >= g["ltNChunks"]:
            self._set_value("Output.A", 0, _site(554), via=1193)
            if g.get("ltActive"):
                g["ltActive"] = 0
                self._frame("DONE", uid, name)
            g["ltPending"] = 0
        else:
            g["ltPending"] = 1

    def _litho_tick(self) -> None:
        """AFM_LithoTick (:1209-1230)."""
        if not self.litho_task:
            return
        g = self.g
        if "ltChunk" not in g:
            self.litho_task = False
            return
        if not g["ltActive"] or g.get("ltAbort"):
            g["ltPending"] = 0
            self.litho_task = False
            return
        if not g["ltPending"]:
            return
        g["ltPending"] = 0
        self._litho_arm_chunk(g["ltChunk"])

    # ------------------------------------------------------------------ reads --

    def _read_xpt(self, uid: str, group: str) -> None:
        """AFM_ReadXPT (:1320-1357)."""
        group = group or "Crosspoint"
        self._read("td_ReadGroup", (group, CapWave("AFM_xptTmp", 0)), _site(1328), "xpt")
        if not self._xpt:
            self._frame("ERR", uid, f"td_ReadGroup({group}) returned no data")
        else:
            self._frame("XPT", uid)

    def _macro_startup(self) -> None:
        """AFM_MacroStartup (afm_igor.ipf:161-176), every read in source order."""
        def gv(name: str, line: int) -> float:
            return self._read("GV", (name,), _site(line, IGOR_IPF), "startup")

        for name, line in (("ScanSize", 41), ("ScanRate", 42), ("ScanPoints", 43),
                           ("ScanLines", 44), ("ScanAngle", 45), ("ImagingMode", 46)):
            gv(name, line)
        if gv("ImagingMode", 55) == 1:
            gv("AmplitudeSetpointVolts", 57)
        else:
            gv("DeflectionSetpointVolts", 60)
        for name, line in (("MaxScanSize", 69), ("InputRange", 70), ("IntegralGain", 77),
                           ("ProportionalGain", 78), ("ThermalFrequency", 85),
                           ("ThermalQ", 86), ("SpringConstant", 87), ("InvOLS", 88),
                           ("AmpInvOLS", 89), ("DriveFrequency", 90),
                           ("DriveAmplitude", 91), ("XLVDTSens", 101), ("YLVDTSens", 103),
                           ("ZLVDTSens", 105), ("XLVDTOffset", 107), ("YLVDTOffset", 109),
                           ("ZLVDTOffset", 111), ("XPiezoSens", 130), ("YPiezoSens", 130),
                           ("XPGain", 131), ("XIGain", 131), ("YPGain", 132),
                           ("YIGain", 132)):
            gv(name, line)
        for chan, line in (("XSensor", 24), ("YSensor", 25), ("Height", 26),
                           ("Deflection", 27), ("Amplitude", 28), ("DetectorSum", 29),
                           ("Output.A", 30)):
            self._read("td_ReadValue", (chan,), _site(line, IGOR_IPF), "startup")
        gv("ARSysScanBusy", 31)
        gv("MicroscopeID", 174)

    def _tip_tick(self) -> None:
        """AFM_TipTick (:1383-1428): six reads, one TELEM frame."""
        for chan, line in (("XSensor", 1411), ("YSensor", 1412), ("Height", 1416),
                           ("Deflection", 1417), ("DetectorSum", 1421), ("amplitude", 1422)):
            self._read("td_ReadValue", (chan,), _site(line), "tip")
        self._frame("TELEM")


def expand(transcript: Iterable, **state) -> Expansion:
    """Expand `transcript` (wire lines or `(verb, args)` pairs) with the given state inputs."""
    return IgorExpander(**state).run(transcript)
