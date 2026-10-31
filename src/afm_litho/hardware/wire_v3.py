"""The v3 wire to Igor: docs/migration/plan/phase-5-wire-v3-contract.md (the contract).

The pass drivers keep sending today's requests (`bridge_protocol` lines) and keep reading
today's reply frames from router waiters. Under `wire.WIRE_VERSION = 3` the link and router
built here carry them as v3 instead:

  * `V3Link.send(v2 line)` sends the matching v3 line (contract §3 is the table): reads
    become READ, setters become SET (and wait for Igor's echo), the transit's
    OUTPAIR + INPAIR + FIRE become one PLAY move, ARMFIRE / LITHOCHAIN / SCANCHAIN become
    PLAY litho1 / lithochain / scanchain, ABORT becomes MAKESAFE.
  * `V3Link.load_wave` loads a whole wave as LOAD + LOADAPP pieces, each sent only after
    Igor has stored the one before (§2d).
  * `V3Router` reads REPLY / REPORT / CAP / TICK and hands each driver the frame it
    already expects (VALUE, XPT, CAPTURE, DONE, SCANDONE, ABORTED, ERR, TELEM). It keeps
    the up-message counter (`useq`), the make-safe mark and Igor's fault stamp.

Each v3 line carries the same arguments, in the same order, as the v2 line it replaces,
so the `igor` level of the goldens is meant to stay the same (§3a). That is untested until
the v3 fake Igor (W3) and the conformance suite (W4) run every scenario under v3. An .ipf
that does not know HELLO is spoken to in v2 unchanged (§7).
"""

from __future__ import annotations

import hashlib
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import __version__, clock
from . import bridge_protocol as P
from . import channels
from .bridge_router import BridgeRouter

PROTO = 3

# Points per LOAD / LOADAPP line (a line-size bound, not a controller limit; §4 rule 7).
LOAD_PIECE_POINTS = 8192
# A LOAD, LOADAPP or SET reply missing for this long fails the run (§2d).
PIECE_TIMEOUT_S = 10.0
# The fixed-pause fallback of §2d: off unless the rig shows Igor still freezing.
LOAD_PAUSE_ON = False
LOAD_PAUSE_S = 0.25
LOAD_PAUSE_ABOVE_POINTS = 4 * LOAD_PIECE_POINTS

HELLO_TIMEOUT_S = 10.0

# The transit is the only user of OUTPAIR + INPAIR + FIRE (§3); PLAY move fixes its channels.
_MOVE_OUT = ("PIDSLoop.0.Setpoint", "PIDSLoop.1.Setpoint")
_MOVE_IN = (channels.X_SENSOR, channels.Y_SENSOR)
_MOVE_EVENT = "0"

# The raw make-safe over COM, for when `AFM_SafePark()` fails over COM: our .ipf missing,
# not compiled, or its dispatcher blocked (contract §10 item 3, §12; v3 only). The lines
# AFM_SafePark runs. Igor accepts td_Stop() even when our procedures fail to compile (rig).
RAW_STOP = "td_Stop()"
RAW_MAKE_SAFE_LINES = (
    RAW_STOP,
    f'td_WriteValue("{channels.OUTPUT_A}",0)',
    f'ir_SetPISLoop(2,"Always,Never","{channels.DEFLECTION}",-10,0,1000,0,'
    f'"{channels.Z_OUT}",0,150)',
)

_IPF_PATH = Path(__file__).resolve().parents[3] / "igor" / "afm_litho_run.ipf"

BANNER_FIX = ("Park the tip, then run `python scripts/install_afm_ipf.py` on the instrument "
              "PC and recompile in Igor.")


class WireTimeout(RuntimeError):
    """Igor did not answer a LOAD, SET or HELLO in time: the run fails (§2d)."""


class WireRefused(RuntimeError):
    """Play or set refused: the Igor procedures do not match this program (§7), or Igor
    refused the line (`err busy|mark|fault|host`)."""


# ------------------------------------------------------------------ codec --

def expected_ipf_sha(path: Path = _IPF_PATH) -> str | None:
    """`kIpfSha` of the .ipf shipped with this program (contract §11.7), or None if the
    file is not beside the package."""
    try:
        text = path.read_bytes()
    except OSError:
        return None
    lines = text.split(b"\n")
    blanked = [b"" if b"StrConstant kIpfSha" in ln else ln for ln in lines]
    return hashlib.sha256(b"\n".join(blanked)).hexdigest()[:16]


def parse_kv(body: str) -> dict[str, str]:
    """`k=v;k=v;` into {k: v} strings; malformed items are skipped."""
    out: dict[str, str] = {}
    for item in (body or "").split(";"):
        k, eq, v = item.strip().partition("=")
        if eq and k.strip():
            out[k.strip()] = v.strip()
    return out


@dataclass
class Up:
    """One up-message. `kind` is REPLY, REPORT, CAP or TICK."""

    kind: str
    seq: str = ""            # REPLY
    ok: bool = False         # REPLY
    body: str = ""           # REPLY body; REPORT fields after the event; CAP values
    useq: int | None = None
    mark: int = 0
    fault: int | None = None  # REPORT and TICK carry it, CAP does not
    run: str = ""
    event: str = ""
    step: int = 0
    alias: str = ""
    telem: tuple = ()


def parse_up(line: str) -> Up | None:
    """Parse a v3 up-message, or None if it is not one (a v2 frame or noise)."""
    verb, _, tail = line.partition(" ")
    if verb == "REPLY":
        seq, _, rest = tail.partition(" ")
        word, _, body = rest.partition(" ")
        if word not in ("ok", "err"):
            return None
        return Up("REPLY", seq=seq, ok=word == "ok", body=body)
    if verb == "REPORT":
        t = tail.split(" ", 7)
        if len(t) < 7:
            return None
        return Up("REPORT", useq=int(t[0]), mark=int(t[2]), fault=int(t[3]), run=t[5],
                  event=t[6], body=t[7] if len(t) > 7 else "")
    if verb == "CAP":
        t = tail.split(" ", 6)
        if len(t) < 7:
            return None
        return Up("CAP", useq=int(t[0]), mark=int(t[2]), run=t[3], step=int(t[4]),
                  alias=t[5], body=t[6])
    if verb == "TICK":
        t = tail.split(" ")
        if len(t) != 11:
            return None
        return Up("TICK", useq=int(t[0]), mark=int(t[2]), fault=int(t[3]),
                  telem=tuple(float(v) for v in t[5:11]))
    return None


# ----------------------------------------------------------------- router --

@dataclass
class _Run:
    """One PLAY in flight: what the driver calls it, and what Igor said it sent."""

    kind: str                                 # move | litho1 | lithochain | scanchain
    name: str
    lines: int = 0
    step_points: dict[int, int] = field(default_factory=dict)
    got: dict[tuple[int, str], int] = field(default_factory=dict)


class V3Router(BridgeRouter):
    """The single reader (contract §8), for v3 and v2 frames on one socket.

    REPLY goes to the waiter whose uid is its `seq`; REPORT and CAP go to the waiter whose
    uid is their `run`. Anything else is a v2 frame and is routed as today.
    """

    def __init__(self, link, **kw):
        super().__init__(link, **kw)
        self._expect: dict[str, str] = {}     # seq -> what the REPLY answers
        self._runs: dict[str, _Run] = {}
        # Called once, inline on the reader thread, when Igor's fault stamp rises to 1
        # (same non-blocking contract as on_telem). The service latches hub.fault.
        self.on_fault = None
        self.igor_fault = False
        self.mark = False
        self.last_useq: int | None = None
        self.holes: list[str] = []            # captures Igor counted but did not deliver

    # -- bookkeeping the link does before it sends --------------------------------

    def expect(self, seq: str, what: str) -> None:
        with self._lock:
            self._expect[seq] = what

    def expect_run(self, run: str, kind: str, name: str, lines: int = 0) -> None:
        with self._lock:
            self._expect[run] = "play"
            self._runs[run] = _Run(kind, name, lines)

    def deliver(self, uid: str, frame: tuple[str, str], *, terminal: bool = True) -> None:
        """Hand `frame` to waiter `uid` (the link's answer to a STARTUP it translated)."""
        with self._lock:
            w = self._waiters.pop(uid, None) if terminal else self._waiters.get(uid)
        if w is not None:
            w.put(frame)

    def latch_fault(self, detail: str) -> None:
        """Raise the fault on the service side (the stamp, or Python's own vendor withdraw)."""
        if self.igor_fault:
            return
        self.igor_fault = True
        cb = self.on_fault
        if cb is not None:
            try:
                cb(detail)
            except Exception as exc:                      # noqa: BLE001
                self._log(f"wire v3: on_fault raised ({exc})")

    # -- the read loop's routing --------------------------------------------------

    def _route(self, line: str) -> None:
        verb = line.partition(" ")[0]
        if verb == P.ERR and "unknown verb: HELLO" in line:
            self._old_ipf_hello()
            return
        up = parse_up(line)
        if up is None:
            super()._route(line)
            return
        if up.useq is not None:
            self._stamp(up)
        if up.kind == "TICK":
            cb = self.on_telem
            if cb is not None:
                try:
                    cb(*up.telem)
                except Exception as exc:                  # noqa: BLE001
                    self._log(f"wire v3: on_telem raised ({exc})")
        elif up.kind == "REPLY":
            self._reply(up)
        elif up.kind == "REPORT":
            self._report(up)
        else:
            self._cap(up)

    def _stamp(self, up: Up) -> None:
        if self.last_useq is not None and up.useq != self.last_useq + 1:
            self._log(f"wire v3: useq gap, {self.last_useq} then {up.useq}")
        if self.last_useq is None or up.useq > self.last_useq:
            self.last_useq = up.useq
        self.mark = bool(up.mark)
        if up.fault is None:
            return
        if up.fault:
            self.latch_fault(f"Igor raised its fault (useq {up.useq}): the make-safe "
                             "retract was not confirmed and the vendor withdraw ran")
        else:
            self.igor_fault = False

    def _reply(self, up: Up) -> None:
        with self._lock:
            what = self._expect.pop(up.seq, None)
        if what is None:
            self._log(f"wire v3: REPLY for unknown seq {up.seq!r} -- dropped")
        elif what == "quiet":
            self._log(f"wire v3: seq {up.seq} {'ok' if up.ok else 'err'} {up.body}")
        elif what == "play":
            if not up.ok:                                 # refused: the run never started
                with self._lock:
                    self._runs.pop(up.seq, None)
                self.deliver(up.seq, (P.ERR, f"PLAY refused: {up.body}"))
            else:
                with self._lock:
                    self._expect[up.seq] = "play"         # the run's REPORTs follow
        elif what in ("value", "xpt"):
            word = (P.VALUE if what == "value" else P.XPT) if up.ok else P.ERR
            self.deliver(up.seq, (word, up.body))
        elif what == "flag":                              # 0 or 1, anything else is an error
            good = up.ok and up.body.strip() in ("0", "1")
            self.deliver(up.seq, (P.VALUE, up.body.strip()) if good
                         else (P.ERR, f"flag read: {'ok' if up.ok else 'err'} {up.body}"))
        elif what == "clearfault":
            # Cleared here, on Igor's acknowledgement, not when CLEARFAULT was sent: a TICK
            # or REPORT stamped fault=1 before Igor cleared must not re-latch hub.fault.
            self._log(f"wire v3: CLEARFAULT {'ok' if up.ok else 'err'} {up.body}")
            if up.ok:
                self.igor_fault = False
        else:                                             # raw: the link reads it itself
            self.deliver(up.seq, ("REPLY", ("ok " if up.ok else "err ") + up.body))

    def _old_ipf_hello(self) -> None:
        with self._lock:
            seq = next((s for s, w in self._expect.items() if w == "hello"), None)
            if seq is not None:
                self._expect.pop(seq)
        if seq is not None:
            self.deliver(seq, ("REPLY", "err unknown verb"))

    def _report(self, up: Up) -> None:
        if up.event in ("madesafe", "aborted"):
            self.note_igor_abort(f"REPORT {up.run} {up.event} {up.body}".rstrip())
        if up.event in ("madesafe", "confirmed", "backup", "error") and up.run == "-":
            self._log(f"wire v3: Igor {up.event} {up.body}")
            return
        with self._lock:
            run = self._runs.get(up.run)
        if run is None:
            self._log(f"wire v3: REPORT for unknown run {up.run!r} ({up.event} {up.body})")
            return
        if up.event == "step":
            k, _n, points = (int(v) for v in up.body.split()[:3])
            run.step_points[k] = points
            return
        if up.event == "armed":
            return
        if up.event == "error":
            self._end_run(up.run, (P.ERR, up.body))
        elif up.event == "aborted":
            self._end_run(up.run, (P.ABORTED, run.name))
        elif up.event == "done":
            self._note_holes(up.run, run)
            done = ((P.SCANDONE, str(run.lines)) if run.kind == "scanchain"
                    else (P.DONE, run.name))
            self._end_run(up.run, done)
        else:
            self._log(f"wire v3: run {up.run} {up.event} {up.body}")

    def _end_run(self, run_id: str, frame: tuple[str, str]) -> None:
        with self._lock:
            self._runs.pop(run_id, None)
            self._expect.pop(run_id, None)
        self.deliver(run_id, frame)

    def _note_holes(self, run_id: str, run: _Run) -> None:
        """A lost capture shows as a known hole, never as silently short data (§2c)."""
        aliases = {a for (_k, a) in run.got}
        for k, points in run.step_points.items():
            for a in aliases:
                have = run.got.get((k, a), 0)
                if have < points:
                    hole = f"run {run_id} step {k} {a}: {have} of {points} points"
                    self.holes.append(hole)
                    self._log(f"wire v3: capture hole, {hole}")

    def _cap(self, up: Up) -> None:
        with self._lock:
            run = self._runs.get(up.run)
        if run is None:
            return
        n = up.body.count(";") + 1 if up.body else 0
        run.got[(up.step, up.alias)] = run.got.get((up.step, up.alias), 0) + n
        name = f"L{up.step}" if run.kind == "scanchain" else run.name
        self.deliver(up.run, (P.CAPTURE, f"{name} {up.alias} {up.body}"), terminal=False)


# ------------------------------------------------------------------- link --

class V3Link:
    """The driver-facing link under wire v3: takes today's request lines, sends v3.

    `state` after HELLO: "v3" (match), "hand_edit" (same version, other hash: play and
    set wait for `accept_hand_edit`), "mismatch" (other version), "v2" (an .ipf without
    HELLO: every line goes out unchanged). Reads, telemetry and make-safe work in all.
    """

    def __init__(self, link, router: V3Router, *, log=None,
                 resume_from_useq: int | None = None):
        self._link = link
        self.router = router
        self._log = log or (lambda msg: None)
        self.state = "unknown"
        self.version_banner: str | None = None
        self.hello: dict[str, str] = {}
        self._resume_from = resume_from_useq
        self._hello_lock = threading.Lock()
        # A make-safe went out and no confirm or clear has come back: an abort is running.
        self._park_open = False
        self._move_out: list[str] | None = None   # OUTPAIR tokens held for PLAY move
        self._move_in: list[str] | None = None    # INPAIR tokens held for PLAY move

    @property
    def checks_version(self) -> bool:
        """True: this link owns the version check, so the drivers skip theirs."""
        return self.state != "v2"

    def close(self) -> None:
        self._link.close()

    def _raw(self, line: str) -> None:
        self._link.send(line)

    def _new_seq(self) -> str:
        w = self.router.open_waiter()
        return w.uid

    def _ask(self, line_for_seq, what: str = "raw",
             timeout: float = PIECE_TIMEOUT_S) -> tuple[bool, str]:
        """Send one line under a fresh seq and wait for its REPLY: (ok, body)."""
        w = self.router.open_waiter()
        self.router.expect(w.uid, what)
        try:
            self._raw(line_for_seq(w.uid))
            _verb, rest = w.get(timeout)
        except queue.Empty:
            raise WireTimeout(f"no reply from Igor within {timeout:g} s to "
                              f"{line_for_seq('<seq>')[:60]!r}") from None
        finally:
            self.router.release(w.uid)
        word, _, body = rest.partition(" ")
        return word == "ok", body

    # -- HELLO and the version gate (§7) ------------------------------------------

    def say_hello(self) -> dict[str, str]:
        """HELLO once per connection; sets `state` and, on a mismatch, the banner."""
        with self._hello_lock:
            if self.state != "unknown":
                return self.hello
            ok, body = self._ask(lambda s: f"HELLO {s} {PROTO} {__version__}", "hello",
                                 HELLO_TIMEOUT_S)
            if not ok and body.startswith("unknown verb"):
                self.state = "v2"
                self._log("wire v3: the Igor procedures do not know HELLO; speaking v2")
                return self.hello
            if not ok and body.startswith("mismatch"):
                body = body[len("mismatch"):].strip()
            self.hello = parse_kv(body)
            igor_proto = self.hello.get("proto", "?")
            if not ok:
                self.state = "mismatch"
                self.version_banner = (f"Igor procedures are v{igor_proto}, this program "
                                       f"needs v{PROTO}. {BANNER_FIX}")
            elif self._hand_edited():
                self.state = "hand_edit"
                self.version_banner = (
                    f"Igor procedures are v{igor_proto} but differ from this program's "
                    f"copy (sha {self.hello.get('sha')}, expected {expected_ipf_sha()}). "
                    f"Play and set wait for the operator's confirm. {BANNER_FIX}")
            else:
                self.state = "v3"
                self._log(f"wire: Igor answered HELLO, proto={igor_proto} "
                          f"ipf={self.hello.get('ipf', '?')}")
            if self.version_banner:
                self._log(self.version_banner)
            if self.hello.get("fault") == "1":
                self.router.latch_fault("Igor's fault was set at connect (HELLO fault=1)")
            self._fetch_missed()
            return self.hello

    def _hand_edited(self) -> bool:
        want, got = expected_ipf_sha(), self.hello.get("sha", "")
        return bool(want and got) and want != got

    def accept_hand_edit(self) -> None:
        """The operator confirms playing on hand-edited procedures of the right version."""
        if self.state == "hand_edit":
            self.state = "v3"

    def _fetch_missed(self) -> None:
        """After a reconnect, ask for the REPORTs sent while we were away (§2c)."""
        start = self._resume_from
        newest = self.hello.get("ring", "").partition("-")[2]
        if start is None or not newest.isdigit() or int(newest) <= start:
            return
        ok, body = self._ask(lambda s: f"FETCH {s} {start + 1}")
        if not ok:
            self._log(f"wire v3: REPORTs after useq {start} are gone from Igor's ring "
                      f"({body})")

    def _may_change(self, what: str) -> None:
        """Play and set need a matching HELLO (§7)."""
        self.say_hello()
        if self.state not in ("v3", "v2"):
            raise WireRefused(f"{what} refused: {self.version_banner}")

    # -- the requests ---------------------------------------------------------------

    def send(self, line: str) -> None:
        """Send one of today's request lines as its v3 line (contract §3)."""
        verb, _, rest = line.partition(" ")
        if verb == P.ABORT:
            self.make_safe()
            return
        if verb in (P.ABORTCONFIRM, P.CLEARFAULT):
            self._park_open = False            # the park confirmed, or the operator cleared
        if verb in (P.ABORTCONFIRM, P.ABORTFAULT, P.CLEARFAULT) and not self._speaks_v3():
            self._raw(line)                    # the v2 words of today's park (abort-redesign)
            return
        if self.state == "v2" and verb != P.STARTUP:
            self._raw(line)
            return
        t = rest.split(" ")
        if verb == P.STARTUP:
            self._startup(t[0], line)
        elif verb in (P.GETVALUE, P.GETGV, P.READXPT):
            what = {P.GETVALUE: "value", P.GETGV: "gv", P.READXPT: "xpt"}[verb]
            self.router.expect(t[0], "xpt" if verb == P.READXPT else "value")
            self._raw(f"READ {t[0]} {what} {' '.join(t[1:])}")
        elif verb == P.GETSTR:
            self.router.expect(t[0], "xpt")
            self._raw(f"READ {t[0]} str {t[1]}")
        elif verb == P.ABORTFAULT:
            self.router.expect(t[0], "flag")
            self._raw(f"READ {t[0]} flag abortFault")
        elif verb == P.ABORTCONFIRM:
            self._quiet("CONFIRMSAFE")         # Igor re-reads Height before it clears
        elif verb == P.CLEARFAULT:
            self._quiet("CLEARFAULT", what="clearfault")
        elif verb == P.SETVALUE:
            self._set(f"value {rest}")
        elif verb == P.SERVOLOOP:
            self._set(f"loop {rest}")
        elif verb == P.STOPLOOP:
            self._set(f"loopstop {rest}")
        elif verb == P.RAMP:
            self._set(f"ramp {rest}")
        elif verb == P.SETMODE:
            self._set(f"mode {rest}")
        elif verb == P.SHOWTIP:
            self._set(f"showtip {rest}")
        elif verb == P.OUTPAIR:
            self._move_out = t
        elif verb == P.INPAIR:
            self._move_in = t
        elif verb == P.FIRE:
            self._play_move(t)
        elif verb == P.ARMFIRE:
            uid, name, interp, cap_len, decim = t
            self._play(uid, "litho1", name,
                       f"{name} XLitho YLitho VLitho {interp} {cap_len} {decim}")
        elif verb == P.LITHOCHAIN:
            uid, name = t[0], t[1]
            self._play(uid, "lithochain", name, " ".join(t[1:]))
        elif verb == P.SCANCHAIN:
            uid, xw, yw, ppl, lines, interp, decim, err_chan = t
            self._play(uid, "scanchain", "", f"{xw} {yw} {ppl} {lines} {interp} {decim} "
                       f"{err_chan}", lines=int(lines))
        else:
            # WAVE/WAVEAPP go through load_wave; OUTWAVE/INWAVE have no v3 verb (§3).
            raise ValueError(f"wire v3 has no line for {verb}")

    def _startup(self, uid: str, line: str) -> None:
        """The driver's STARTUP: HELLO for the version gate, then STARTUP itself (permanent
        safe set, §7), so Igor makes today's calibration reads and the driver reads
        today's reply (F100, contract §11.13)."""
        self.say_hello()
        self._raw(line)

    def _set(self, body: str) -> None:
        self._may_change("set")
        ok, echo = self._ask(lambda s: f"SET {s} {body}")
        if not ok:
            raise WireRefused(f"SET {body.split(' ')[0]} refused by Igor: {echo}")

    def _play(self, uid: str, kind: str, name: str, fields: str, *, lines: int = 0) -> None:
        """PLAY under the driver's uid, which is both the seq and the run id (§11.4)."""
        self._may_change("play")
        self.router.expect_run(uid, kind, name, lines)
        self._raw(f"PLAY {uid} {kind} {uid} {fields}")

    def _play_move(self, t: list[str]) -> None:
        """OUTPAIR + INPAIR + FIRE of the transit -> PLAY move (§3)."""
        uid, name, event = t
        out, inp = self._move_out, self._move_in
        self._move_out = self._move_in = None
        if out is None or inp is None:
            raise ValueError("wire v3: FIRE without the transit's OUTPAIR and INPAIR")
        _b, out_ev, ch_a, xw, ch_b, yw, interp = out
        _b, in_ev, in_a, in_b, cap_len, decim = inp
        if (event, out_ev, in_ev) != (_MOVE_EVENT,) * 3 or (ch_a, ch_b) != _MOVE_OUT \
                or (in_a, in_b) != _MOVE_IN:
            raise ValueError(f"wire v3: PLAY move is the XY transit only, not {out} {inp}")
        self._play(uid, "move", name, f"{xw} {yw} {interp} {cap_len} {decim}")

    def load_wave(self, name: str, values) -> None:
        """Load one whole wave, each piece sent only after Igor stored the last (§2d).
        A missing reply raises WireTimeout: the run fails into stop and withdraw."""
        vals = np.asarray(values, dtype=np.float64)
        n, c = len(vals), LOAD_PIECE_POINTS
        self.say_hello()
        if self.state == "v2":
            self._raw(P.cmd_wave(name, vals[:c]))
            for i in range(c, n, c):
                self._raw(P.cmd_waveapp(name, vals[i:i + c]))
            return
        self._may_change("load")
        if n == 0:
            raise ValueError(f"wire v3: LOAD {name} has no points")
        pause = LOAD_PAUSE_ON and n > LOAD_PAUSE_ABOVE_POINTS
        for i in range(0, n, c):
            body = ";".join(P._n(v) for v in vals[i:i + c])
            head = f"LOAD {{}} {name} {n}" if i == 0 else f"LOADAPP {{}} {name}"
            ok, echo = self._ask(lambda s, h=head, b=body: h.format(s) + " " + b)
            if not ok:
                raise WireRefused(f"LOAD {name} refused by Igor: {echo}")
            if pause and i + c < n:
                clock.sleep(LOAD_PAUSE_S)

    # -- make-safe (§5, §6) ---------------------------------------------------------

    def _speaks_v3(self) -> bool:
        return self.state in ("v3", "hand_edit")

    def make_safe(self) -> None:
        """Make-safe on the socket, at once: no HELLO, no gate, no wait (parallel-work
        rule 5). Igor runs AFM_SafePark(). MAKESAFE only to procedures known to speak v3;
        any other .ipf gets ABORT, which is in the permanent safe set (§7)."""
        self._park_open = True
        if not self._speaks_v3():
            self._raw(P.cmd_abort())
            return
        self._quiet("MAKESAFE")

    def abort_running(self) -> bool:
        """An abort is under way: Igor's mark stamp says unconfirmed, or a make-safe went
        out with no confirm or clear since."""
        return self._park_open or self.router.mark

    def raw_com_make_safe(self, com_igor, log=None) -> bool:
        """RAW_MAKE_SAFE_LINES over COM, each as an ARC call (Execute2 alone hides ARC error
        codes). td_Stop() is skipped while an abort is running, so a retract under way is
        not undefined. True only when the bias zero and the re-arm both returned 0;
        otherwise the caller still sends the socket make-safe."""
        skip_stop = self.abort_running()
        self._park_open = True
        ok = True
        for line in RAW_MAKE_SAFE_LINES:
            if line == RAW_STOP and skip_stop:
                continue
            try:
                call = getattr(com_igor, "arc_call", None)
                if call is not None:
                    call(line)
                else:
                    com_igor.execute(line, check=True)
            except Exception as exc:                      # noqa: BLE001
                ok = ok and line == RAW_STOP
                if log:
                    log(f"safe_park: raw COM {line} failed ({exc})")
        return ok

    def _quiet(self, verb: str, what: str = "quiet") -> None:
        """A send-only v3 line under a fresh seq; the router handles its REPLY."""
        seq = self._new_seq()
        self.router.release(seq)
        self.router.expect(seq, what)
        self._raw(f"{verb} {seq}")


def open_v3(link, *, log=None, poll_timeout: float = 2.0,
            resume_from_useq: int | None = None) -> tuple[V3Link, V3Router]:
    """Start the v3 reader on a connected BridgeLink and return (link, router) for a driver."""
    router = V3Router(link, log=log, poll_timeout=poll_timeout).start()
    return V3Link(link, router, log=log, resume_from_useq=resume_from_useq), router
