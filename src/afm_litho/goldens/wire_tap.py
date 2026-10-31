"""Ordered, normalised transcript of every driver line the Igor stub receives.

Two forms are kept side by side (the 0.2a contract, docs/migration/slices/0.2a-golden-format.md):

* `raw`: every line in arrival order, verbatim, plus the out-of-socket and stub-side entries
  spelled as expander pseudo-verbs (`@SAFEPARK`, `@DOSCANFUNC <name>`, `@FILL`, `@TIPTICK`, ...).
  `expander_input()` returns it; the expander consumes it directly. It holds full wave payloads,
  so it is kept in memory only and never serialised.
* the summaries: `actuation`, `reads` and `combined`, entries `{"seq", "kind", "verb", "args"}`.
  UIDs are replaced by per-list labels ("a1", ... / "r1", ...) because the counter is shared
  with the safety poll; waves are summarised with the expander's own `summarise`, so the wire
  and the end-of-line goldens carry one hash (sha256 of the float32 bytes Igor holds).
"""

from __future__ import annotations

import json
import re
import threading

import numpy as np

from ..hardware import bridge_protocol as P
from .expander import _f32, _items, str2num, summarise

FORMAT = 2

# Every driver -> bridge verb and its class. A verb added to bridge_protocol without a class
# here fails tests/test_golden_contract.py, so a new read cannot land in the exact list.
VERB_CLASS = {
    P.STARTUP: "read", P.GETVALUE: "read", P.GETGV: "read", P.GETSTR: "read", P.READXPT: "read",
    P.SETVALUE: "act", P.SERVOLOOP: "act", P.STOPLOOP: "act", P.RAMP: "act", P.WAVE: "act",
    P.WAVEAPP: "act", P.ARMFIRE: "act", P.SHOWTIP: "act", P.SETMODE: "act", P.ABORT: "act",
    P.OUTWAVE: "act", P.OUTPAIR: "act", P.INWAVE: "act", P.INPAIR: "act", P.FIRE: "act",
    P.SCANCHAIN: "act", P.LITHOCHAIN: "act", P.ABORTCONFIRM: "act", P.ABORTFAULT: "read",
    P.CLEARFAULT: "act",
}
READ_VERBS = frozenset(v for v, c in VERB_CLASS.items() if c == "read")

# verbs whose first argument is a request UID (docs/IGOR_PYTHON_COMMS.md section 6).
UID_VERBS = frozenset({P.STARTUP, P.GETVALUE, P.GETGV, P.READXPT, P.ARMFIRE, P.FIRE,
                       P.SCANCHAIN, P.LITHOCHAIN, P.ABORTFAULT, P.GETSTR})

# Stub-side pseudo-entries that are reads (the AFMtip poll runs on Igor's own clock).
PSEUDO_READS = frozenset({"@TIPTICK"})

# Each pattern must match the WHOLE string (fullmatch); a string with ";" is two commands.
_SAFEPARK = re.compile(r"AFM_SafePark\(\s*\)")
_DOSCANFUNC = re.compile(r'DoScanFunc\(\s*"([^"\s]*)"\s*\)')
_RUNSTART = re.compile(r"AFM_LithoRunStart\([^()]*\)")
_RUNEND = re.compile(r"AFM_LithoRunEnd\(\s*\)")
_DRYRUN = re.compile(r"AFM_LithoRunDryRun\(\s*([^()\s]*)\s*\)")
_PRINT = re.compile(r'Print\s+"(?:[^"\\]|\\.)*"')


def com_entry(text: str) -> tuple[str, list[str]]:
    """The one spelling of an out-of-socket (COM `Execute2`) command, as an expander pseudo-verb.

    Anything not listed becomes `@COM <text>`, which the expander refuses as unmodelled.
    """
    t = str(text).strip()
    if ";" in t or "\n" in t or "\r" in t:
        return "@COM", [t]
    if _SAFEPARK.fullmatch(t):
        return "@SAFEPARK", []
    if m := _DOSCANFUNC.fullmatch(t):
        return "@DOSCANFUNC", [m.group(1)]
    if _RUNSTART.fullmatch(t):
        return "@RUNSTART", []
    if _RUNEND.fullmatch(t):
        return "@RUNEND", []
    if m := _DRYRUN.fullmatch(t):
        return "@DRYRUN", [m.group(1)]
    if _PRINT.fullmatch(t):
        return "@PRINT", []                 # COM Print: no vendor call
    return "@COM", [t]


class _Wave:
    """Running summary of one named wave across WAVE / WAVEAPP, parsed as Igor parses it."""
    __slots__ = ("chunks", "name")

    def __init__(self, name: str) -> None:
        self.name = name
        self.chunks: list[np.ndarray] = []

    def feed(self, payload: str) -> None:
        self.chunks.append(np.array([str2num(t) for t in _items(payload, ";")],
                                    dtype=np.float32))

    def summary(self, name: str) -> list:
        w = summarise(name, np.concatenate(self.chunks) if self.chunks
                      else np.zeros(0, np.float32))
        return [name, w.n, _f32(w.first), _f32(w.last), w.sha]


class WireTap:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._waves: dict[str, _Wave] = {}
        self._labels: dict[str, dict[str, str]] = {"a": {}, "r": {}}
        self.actuation: list[dict] = []
        self.reads: list[dict] = []
        self.combined: list[dict] = []
        self.raw: list[str] = []

    def _label(self, side: str, uid: str) -> str:
        m = self._labels[side]
        if uid not in m:
            m[uid] = f"{side}{len(m) + 1}"
        return m[uid]

    def record(self, line: str) -> None:
        """Record one received wire line."""
        parts = line.split(" ")
        verb = parts[0]
        side = "r" if verb in READ_VERBS else "a"
        with self._lock:
            self.raw.append(line)
            args: list = list(parts[1:])
            if verb in UID_VERBS and args:
                args[0] = self._label(side, args[0])
            if verb in (P.WAVE, P.WAVEAPP) and args:
                # Igor keys waves case-insensitively; WAVE replaces, WAVEAPP appends.
                key = parts[1].casefold()
                if verb == P.WAVE or key not in self._waves:
                    self._waves[key] = _Wave(parts[1])
                w = self._waves[key]
                w.feed(parts[2] if len(parts) > 2 else "")
                args = [w.summary(parts[1])]
            self._add(verb, args, side, "read" if side == "r" else "act")

    def record_pseudo(self, verb: str, *args: str) -> None:
        """Record a stub-side event the wire does not carry (`@FILL`, `@TIPTICK`)."""
        side = "r" if verb in PSEUDO_READS else "a"
        with self._lock:
            self.raw.append(" ".join((verb, *args)))
            self._add(verb, list(args), side, "read" if side == "r" else "pseudo")

    def record_com(self, source: str, text: str) -> None:
        """Record a command that reached Igor over COM, spelled by `com_entry`."""
        verb, args = com_entry(text)
        with self._lock:
            self.raw.append(" ".join((verb, *args)))
            self._add(verb, args, "a", "ext", source=source)

    def record_com_failure(self, source: str, text: str, code: int) -> None:
        """Record a COM command Igor ran and rejected (nonzero error code): vendor calls before
        the error may have run, so it is kept, as `@COMFAIL <cmd> <code>`."""
        t = str(text).strip()
        with self._lock:
            self.raw.append(f"@COMFAIL {t} {int(code)}")
            self._add("@COMFAIL", [t, str(int(code))], "a", "ext", source=source)

    def expander_input(self) -> list[str]:
        """The arrival-order transcript the expander consumes (`expand(tap.expander_input())`)."""
        with self._lock:
            return list(self.raw)

    def _add(self, verb: str, args: list, side: str, kind: str, source: str = "") -> None:
        lst = self.reads if side == "r" else self.actuation
        e = {"seq": len(lst), "kind": kind, "verb": verb, "args": args}
        if source:
            e["source"] = source
        lst.append(e)
        self.combined.append({**e, "seq": len(self.combined)})

    def snapshot(self) -> dict:
        with self._lock:
            return {"format": FORMAT, "actuation": list(self.actuation),
                    "reads": list(self.reads), "combined": list(self.combined)}

    def to_json(self, *, combined: bool = True) -> bytes:
        """Sorted-key, compact JSON. `combined=False` drops the debug list."""
        d = self.snapshot()
        if not combined:
            del d["combined"]
        return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def actuation_json(self) -> bytes:
        """The deterministic part: just the actuation list."""
        with self._lock:
            d = {"format": FORMAT, "actuation": list(self.actuation)}
        return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")


class TappedComRefused(RuntimeError):
    """A COM entry point that would bypass the tap was called on a `TappedCom`."""


class TappedCom:
    """Wrap a COM-like object (`execute(cmd, check=...)`) so each call that reaches Igor is
    recorded on `tap` as an out-of-socket entry. A success is spelled by `com_entry`; a call
    Igor rejected with a nonzero error code (returned, or raised as `IgorError`) may have run
    vendor calls before the error, so it is recorded as `@COMFAIL`. A call that raises
    anything else never reached Igor (no COM, no connection) and is not recorded.

    `arc_call`, `value` and `string` are refused: each runs vendor calls through `execute`
    on the inner object, so they would reach Igor unrecorded.
    """

    def __init__(self, inner, tap: WireTap, source: str = "com") -> None:
        self._inner, self._tap, self._source = inner, tap, source

    def execute(self, cmd, *a, **kw):
        from ..hardware.igor import IgorError
        try:
            r = self._inner.execute(cmd, *a, **kw)
        except IgorError as exc:
            self._tap.record_com_failure(self._source, str(cmd), exc.code)
            raise
        code = int(getattr(r, "error_code", 0) or 0)
        if code == 0:
            self._tap.record_com(self._source, str(cmd))
        else:
            self._tap.record_com_failure(self._source, str(cmd), code)
        return r

    def _refuse(self, name: str):
        raise TappedComRefused(f"TappedCom.{name} is not recorded by the golden tap; "
                               f"route the call through execute()")

    def arc_call(self, *a, **kw):
        self._refuse("arc_call")

    def value(self, *a, **kw):
        self._refuse("value")

    def string(self, *a, **kw):
        self._refuse("string")

    def __getattr__(self, name):
        return getattr(self._inner, name)
