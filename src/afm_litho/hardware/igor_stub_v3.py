"""Wire v3 for the fake Igor (docs/migration/plan/phase-5-wire-v3-contract.md, slice W3).

`V3Wire` answers every v3 verb beside the v2 verbs of `FakeIgorBridge`. Each state-changing v3
verb is TRANSLATED into the v2 line the existing stub already runs (WAVE, SERVOLOOP, ARMFIRE,
LITHOCHAIN, SCANCHAIN, OUTPAIR + INPAIR + FIRE, ...). So the stub's tap, banks and safety
simulation record exactly the Asylum commands a v2 pass records: the Igor-level log of a v3
pass equals the v2 one by construction. Only the replies and the up-messages are new, and the
v2 completion frames (CAPTURE, DONE, SCANDONE, ABORTED, ERR) are turned into CAP and REPORT.

Make-safe is the agreed sequence (abort-redesign.md, contract section 5). It lives here, apart from
the v2 ABORT and park code of the stub.
"""

from __future__ import annotations

import collections
import re

from .. import clock
from . import bridge_protocol as P

PROTO = 3
K_SOCK_DEAD_PARK_S = 5.0        # kSockDeadParkS, 2026-10-27
HEIGHT_SAFE_V = 1.0             # Height under this counts as retracted (2026-10-27)
PIECE_POINTS = 8192             # kCapChunkPoints, and the LOAD piece size
RING_SIZE = 256
TICKS_PER_S = 60                # Igor `ticks`

HELLO, READ, STATUS, FETCH = "HELLO", "READ", "STATUS", "FETCH"
LOAD, LOADAPP, PLAY, SET = "LOAD", "LOADAPP", "PLAY", "SET"
MAKESAFE, CONFIRMSAFE, VENDORWITHDRAW, CLEARFAULT = (
    "MAKESAFE", "CONFIRMSAFE", "VENDORWITHDRAW", "CLEARFAULT")
V3_VERBS = frozenset({HELLO, READ, STATUS, FETCH, LOAD, LOADAPP, PLAY, SET, MAKESAFE,
                      CONFIRMSAFE, VENDORWITHDRAW, CLEARFAULT})
# Queued behind a MAKESAFE, these are dropped (contract section 6).
_DROPPED_BEHIND_MAKESAFE = frozenset({PLAY, LOAD, LOADAPP, SET})
_LITHO_WAVES = ("XLitho", "YLitho", "VLitho")     # the names ARMFIRE reads in the .ipf
_LINE_NAME = re.compile(r"L(\d+)$")


def _num(v: float) -> str:
    return f"{float(v):.12g}"


def _verb_of(line: str) -> str:
    return line.split(" ", 1)[0]


class _NullLink:
    """A link that goes nowhere: the v2 lines a v3 verb runs for their effect only."""

    def send(self, line: str) -> None:
        pass


class _RunLink:
    """The `link` the translated v2 lines are dispatched with: it turns the v2 completion
    frames of one run into CAP and REPORT messages."""

    def __init__(self, wire: "V3Wire", run: str, kind: str, n_steps: int) -> None:
        self.wire, self.run, self.kind, self.n_steps = wire, run, kind, n_steps
        self.step = 0
        self.points: int | None = None      # points of the open step, None until it has data
        self.total_points = 0               # per channel, over the closed steps
        self.steps_closed = 0
        self._seen: set[str] = set()
        self.finished = False
        self._hold: list[str] | None = []   # frames that arrive before 'armed' is sent

    def release(self) -> None:
        held, self._hold = self._hold or [], None
        for line in held:
            self.send(line)

    def send(self, line: str) -> None:
        if self._hold is not None:
            self._hold.append(line)
            return
        verb = _verb_of(line)
        if verb == P.CAPTURE:
            _, _uid, name, chan, vals = line.split(" ", 4)
            self._capture(name, chan, vals)
        elif verb == P.DONE:
            self._terminal("done")
        elif verb == P.SCANDONE:
            self._terminal("aborted" if int(line.split(" ")[2]) < 0 else "done")
        elif verb == P.ABORTED:
            self._terminal("aborted")
        elif verb == P.ERR:
            self._terminal("error -1 " + line.split(" ", 2)[2])

    def _capture(self, name: str, chan: str, vals: str) -> None:
        if self.finished:
            return
        m = _LINE_NAME.match(name)
        step = int(m.group(1)) if m else self.step
        if m is None and chan in self._seen:
            step += 1                           # the chunks of one object share its name
        if step != self.step:
            self._close_step()
            self.step, self._seen = step, set()
        self._seen.add(chan)
        n = len(vals.split(";")) if vals else 0
        self.points = n
        self.wire.up_cap(self.run, self.step, chan, vals)

    def _close_step(self) -> None:
        if self.points is not None:
            self.wire.up_report(self.run, f"step {self.step} {self.n_steps} {self.points}")
            self.total_points += self.points
            self.steps_closed += 1
            self.points = None

    def _terminal(self, event: str) -> None:
        if self.finished:
            return
        self.finished = True
        self._close_step()
        if event == "done":
            event += f" {self.steps_closed if self.kind == 'scanchain' else self.total_points}"
        self.wire.up_report(self.run, event)
        self.wire.run_ended(self)


class V3Wire:
    """Wire v3 state and verbs of one fake Igor. Knobs are plain attributes a test may set."""

    def __init__(self, stub, *, proto: int = PROTO, ipf: int = 3,
                 sha: str = "0123456789abcdef") -> None:
        self.stub = stub
        self.proto, self.ipf, self.sha = int(proto), int(ipf), sha
        # LOAD acknowledgement pacing: Igor "chokes" this long on each piece before the reply.
        self.load_delay_s = 0.0
        self.drop_load_acks_from: int | None = None     # the Nth LOAD/LOADAPP and later: no reply
        self._t0 = clock.now()
        self.useq = 0
        self.ring: collections.deque = collections.deque()      # (useq, line), REPORTs only
        self._evicted_max = 0
        self.host_link = None
        self.connected = False
        self._link_state: dict[int, str] = {}           # id(link) -> "host" | "mismatch"
        self.run: _RunLink | None = None
        self.last_run: _RunLink | None = None
        # The make-safe flags are the stub's (abort_stopped = `mark`, abort_fault = `fault`).
        self.vendor_withdrawn = False                   # per make-safe; the flags are the stub's
        self.igor_calls: list[str] = []                 # the make-safe's own commands, in order
        self._park_timer = None
        self._closed = False
        self.dead_since: float | None = None
        # what the guard and STATUS read, rebuilt from the loop commands Igor ran
        self.tip_down = False
        self.loop2: tuple | None = None                 # (setpoint, min, max); None = undefined
        self._loops_seen = 0
        self._loads = 0
        self._totals: dict[str, int] = {}               # LOAD's n_total per wave
        self.dropped: list[str] = []

    @property
    def lock(self):
        """The stub's send lock: a completion holds it while it sends, so v3 state and the
        wire share one lock and cannot deadlock on each other."""
        return self.stub._sendlock

    @property
    def abort_stopped(self) -> bool:
        return self.stub.abort_stopped

    @abort_stopped.setter
    def abort_stopped(self, v: bool) -> None:
        self.stub.abort_stopped = bool(v)

    @property
    def abort_fault(self) -> bool:
        return self.stub.abort_fault

    @abort_fault.setter
    def abort_fault(self, v: bool) -> None:
        self.stub.abort_fault = bool(v)

    # ---- stamps and up-messages ----
    def tick(self) -> int:
        return int((clock.now() - self._t0) * TICKS_PER_S)

    def _mark(self) -> int:
        return int(self.abort_stopped)

    def _fault(self) -> int:
        return int(self.abort_fault)

    def ring_span(self) -> str:
        with self.lock:
            return f"{self.ring[0][0]}-{self.ring[-1][0]}" if self.ring else "0-0"

    def _to_host(self, line: str) -> None:
        link = self.host_link
        if link is None or not self.connected:
            return
        try:
            self.stub._send(link, line)
        except OSError:
            pass

    def up_report(self, run: str, event: str, route: str = "sock") -> None:
        with self.lock:
            self.useq += 1
            line = (f"REPORT {self.useq} {self.tick()} {self._mark()} {self._fault()} "
                    f"{route} {run} {event}")
            self.ring.append((self.useq, line))
            while len(self.ring) > RING_SIZE:
                self._evicted_max = self.ring.popleft()[0]
            self._to_host(line)

    def up_cap(self, run: str, step: int, alias: str, vals: str) -> None:
        pieces = vals.split(";") if vals else [""]
        for lo in range(0, len(pieces), PIECE_POINTS):
            with self.lock:
                self.useq += 1
                self._to_host(f"CAP {self.useq} {self.tick()} {self._mark()} {run} {step} "
                              f"{alias} {';'.join(pieces[lo:lo + PIECE_POINTS])}")

    def tick_frame(self, x, y, z, defl, sum_v, amp) -> str:
        with self.lock:
            self.useq += 1
            return (f"TICK {self.useq} {self.tick()} {self._mark()} {self._fault()} - "
                    f"{P._n(x)} {P._n(y)} {P._n(z)} {P._n(defl)} {P._n(sum_v)} {P._n(amp)}")

    def is_v3_link(self, link) -> bool:
        return self._link_state.get(id(link)) == "host"

    def _reply(self, link, seq: str, status: str, body: str = "") -> None:
        self.stub._send(link, f"REPLY {seq} {status}" + (f" {body}" if body else ""))

    # ---- the socket ----
    def intake(self, first: str, link, dispatch) -> None:
        """One read of the socket. A MAKESAFE anywhere in what has arrived runs first; the
        PLAY, LOAD, LOADAPP and SET lines queued with it are dropped, each with a reply."""
        batch = [first]
        while b"\n" in getattr(link, "_buf", b""):
            batch.append(link.read_line(0))
        if not any(_verb_of(line) == MAKESAFE for line in batch):
            for line in batch:
                dispatch(line, link)
            return
        for line in batch:
            if _verb_of(line) == MAKESAFE:
                dispatch(line, link)
        for line in batch:
            verb = _verb_of(line)
            if verb == MAKESAFE:
                continue
            if verb in _DROPPED_BEHIND_MAKESAFE:
                self.dropped.append(line)
                self._reply(link, line.split(" ")[1], "err", "dropped")
            else:
                dispatch(line, link)

    def on_close(self, link) -> None:
        """The host closed. A running run goes on; the guard parks a tip left down."""
        with self.lock:
            if link is not self.host_link or not self.connected or self._closed:
                return
            self.connected = False
            self.dead_since = clock.now()
            self._cancel(self._park_timer)
            self._park_timer = clock.timer(K_SOCK_DEAD_PARK_S, self._park_check)
            self._park_timer.start()

    def close(self) -> None:
        """Stop every timer (a test that leaves virtual timers pending fails)."""
        with self.lock:
            self._closed = True
            self._cancel(self._park_timer)

    @staticmethod
    def _cancel(timer) -> None:
        if timer is not None:
            timer.cancel()

    def sock_dead_s(self) -> float:
        return 0.0 if self.dead_since is None else clock.now() - self.dead_since

    def _park_check(self) -> None:
        """The dispatcher's guard (contract 5b): dead for kSockDeadParkS, no run, tip down."""
        with self.lock:
            if self.dead_since is None or self.run is not None:
                return
            if clock.now() - self.dead_since < K_SOCK_DEAD_PARK_S - 1e-6:
                return
            self._refresh_loops()
            if self.tip_down:
                self.make_safe("socket", route="-")

    def run_ended(self, run: _RunLink) -> None:
        with self.lock:
            self.last_run = run
            if self.run is run:
                self.run = None
        self._park_check()

    # ---- what Igor tracks from the loop commands it ran ----
    def _refresh_loops(self) -> None:
        loops, limits = self.stub.loops, self.stub.loop_limits
        while self._loops_seen < len(loops):
            n, _in, sp, _p, _i, _out = loops[self._loops_seen]
            lo, hi = limits[self._loops_seen]
            self._loops_seen += 1
            if n == 2:
                self.loop2 = (sp, lo, hi)
                self.tip_down = sp != -10          # a lift or withdraw is setpoint -10

    # ---- dispatch ----
    def handle(self, parts: list[str], link) -> None:
        verb, seq = parts[0], parts[1] if len(parts) > 1 else "0"
        sim = self.stub.safety_sim
        if sim:
            sim.sync()
        try:
            getattr(self, "_do_" + verb.lower())(parts, link, seq)
        except (IndexError, ValueError) as exc:
            self._reply(link, seq, "err", f"bad {type(exc).__name__}")

    def _host_error(self, link) -> str | None:
        """Why this connection may not LOAD, PLAY or SET, or None."""
        state = self._link_state.get(id(link))     # "mismatch" or None: not the host
        if state == "host" and link is self.host_link and self.connected:
            return None
        return "host"

    def _hello_body(self) -> str:
        from .igor_stub import startup_string
        s = self.stub
        mid = getattr(s, "microscope_id", s.values.get("MicroscopeID", 0))
        return (f"proto={self.proto};ipf={self.ipf};sha={self.sha};MicroscopeID={int(mid)};"
                f"clock={self.tick()};ring={self.ring_span()};mark={self._mark()};"
                f"fault={self._fault()};" + startup_string(s.values))

    def _do_hello(self, parts, link, seq) -> None:
        with self.lock:
            if int(parts[2]) != self.proto:
                self._link_state[id(link)] = "mismatch"
                self._reply(link, seq, "err", "mismatch " + self._hello_body())
                return
            self._link_state[id(link)] = "host"
            self.host_link, self.connected, self.dead_since = link, True, None
            self._cancel(self._park_timer)
            self._reply(link, seq, "ok", self._hello_body())

    def _do_read(self, parts, link, seq) -> None:
        s, what = self.stub, parts[2]
        if what == "value":
            s._record(f"GETVALUE {seq} {parts[3]}", P.GETVALUE)
            if s.stall_getvalue:
                return
            self._reply(link, seq, "ok",
                        P._n(s.values.get(parts[3], s._loop_status(parts[3]))))
        elif what == "flag":
            flags = {"abortFault": s.abort_fault}
            if parts[3] not in flags:
                self._reply(link, seq, "err", f"bad flag {parts[3]}")
                return
            self._reply(link, seq, "ok", str(int(flags[parts[3]])))
        elif what == "str":
            self._read_str(link, seq, parts[3])
        elif what == "gv":
            s._record(f"GETGV {seq} {parts[3]}", P.GETGV)
            s.getgvs.append(parts[3])
            if s.stall_getvalue:
                return
            name = parts[3]
            self._reply(link, seq, "ok", P._n(s.values.get(name, float("nan"))))
        elif what == "xpt":
            s._record(f"READXPT {seq} {parts[3] if len(parts) > 3 else 'Crosspoint'}",
                      P.READXPT)
            self._reply(link, seq, "ok", "".join(f"{k}={v};" for k, v in s.crosspoint.items()))
        else:
            self._reply(link, seq, "err", f"bad read {what}")

    def _read_str(self, link, seq: str, name: str) -> None:
        """td_ReadString of a running loop's OutputChannel; anything else fails."""
        from .igor_stub import _pid_loop_param
        loop = _pid_loop_param(name, "OutputChannel")
        if loop in self.stub.pid_loops:
            self._reply(link, seq, "ok", f"{name}={self.stub.pid_loops[loop]};")
        else:
            self._reply(link, seq, "err", f"td_ReadString({name}) failed")

    def _do_status(self, parts, link, seq) -> None:
        with self.lock:
            self._refresh_loops()
            run = self.run or self.last_run
            active = self.run is not None
            loop2 = "none" if self.loop2 is None else ",".join(_num(v) for v in self.loop2)
            body = (f"pass={run.kind if active else 'none'};"
                    f"run={run.run if active else '-'};"
                    f"step={run.step if active else 0}/{run.n_steps if active else 0};"
                    f"mark={self._mark()};fault={self._fault()};tipdown={int(self.tip_down)};"
                    f"loop2={loop2};sock_dead_s={_num(self.sock_dead_s())};"
                    f"ring={self.ring_span()};tick={self.tick()};")
        self._reply(link, seq, "ok", body)

    def _do_fetch(self, parts, link, seq) -> None:
        start = int(parts[2])
        with self.lock:
            if start <= self._evicted_max:
                oldest = self.ring[0][0] if self.ring else self.useq + 1
                self._reply(link, seq, "err", f"gone {oldest}")
                return
            for useq, line in list(self.ring):
                if useq >= start:
                    self.stub._send(link, line)
            self._reply(link, seq, "ok", self.ring_span())

    # ---- loading ----
    def _do_load(self, parts, link, seq) -> None:
        self._load(parts, link, seq, append=False)

    def _do_loadapp(self, parts, link, seq) -> None:
        self._load(parts, link, seq, append=True)

    def _load(self, parts, link, seq, append: bool) -> None:
        why = self._host_error(link)
        if why:
            self._reply(link, seq, "err", why)
            return
        name = parts[2]
        total = int(parts[3]) if not append else None
        payload = parts[4] if not append else parts[3]
        n_piece = len(payload.split(";")) if payload else 0
        if n_piece > PIECE_POINTS:
            self._reply(link, seq, "err", f"bad piece {n_piece}")
            return
        if append and name not in self.stub.waves:
            self._reply(link, seq, "err", f"bad noname {name}")
            return
        verb = P.WAVEAPP if append else P.WAVE
        self.stub._dispatch(f"{verb} {name} {payload}", link)       # stored when this returns
        self._loads += 1
        if self.load_delay_s:
            clock.sleep(self.load_delay_s)
        if self.drop_load_acks_from is not None and self._loads >= self.drop_load_acks_from:
            return                                  # a choked Igor: the piece is stored, no reply
        have = len(self.stub.waves[name])
        if total is not None:
            self._totals[name] = total
        total = self._totals.get(name, have)
        self._reply(link, seq, "ok", f"{name} {have}/{total}")

    # ---- PLAY ----
    def _do_play(self, parts, link, seq) -> None:
        why = self._host_error(link)
        with self.lock:
            if not why:
                why = ("mark" if self.abort_stopped else "fault" if self.abort_fault
                       else "busy" if self.run is not None else None)
        if why:
            self._reply(link, seq, "err", why)
            return
        kind, run = parts[2], parts[3]
        a = parts[4:]
        s = self.stub
        if kind == "move":
            xw, yw, interp, cap, decim = a
            lines = [P.cmd_outpair(0, "0", "PIDSLoop.0.Setpoint", xw,
                                   "PIDSLoop.1.Setpoint", yw, float(interp)),
                     P.cmd_inpair(0, "0", "XSensor", "YSensor", int(cap), int(decim)),
                     P.cmd_fire(run, "move", "0")]
            n_steps = 1
        elif kind == "litho1":
            name, xw, yw, vw, interp, cap, decim = a
            self._alias_waves((xw, yw, vw))
            lines = [f"{P.ARMFIRE} {run} {name} {interp} {cap} {decim}"]
            n_steps = 1
        elif kind == "lithochain":
            name, xw, yw, vw, bw, interp, decim = a
            lines = [f"{P.LITHOCHAIN} {run} {name} {xw} {yw} {vw} {bw} {interp} {decim}"]
            n_steps = max(1, len(s.waves.get(bw, [])) - 1)
        elif kind == "scanchain":
            xw, yw, ppl, nlines, interp, decim, err = a
            lines = [f"{P.SCANCHAIN} {run} {xw} {yw} {ppl} {nlines} {interp} {decim} {err}"]
            n_steps = int(nlines)
        else:
            self._reply(link, seq, "err", f"bad kind {kind}")
            return
        rl = _RunLink(self, run, kind, n_steps)
        with self.lock:
            self.run = rl
        for line in lines:
            s._dispatch(line, rl)
        self._reply(link, seq, "ok", f"armed {run}")
        self.up_report(run, "armed")
        rl.release()

    def _alias_waves(self, names) -> None:
        """ARMFIRE reads the waves XLitho, YLitho and VLitho; a v3 client may name its own."""
        for have, want in zip(names, _LITHO_WAVES):
            if have != want and have in self.stub.waves:
                self.stub._dispatch(
                    f"{P.WAVE} {want} " + ";".join(_num(v) for v in self.stub.waves[have]),
                    None)

    # ---- SET ----
    def _do_set(self, parts, link, seq) -> None:
        why = self._host_error(link)
        if why:
            self._reply(link, seq, "err", why)
            return
        s, what = self.stub, parts[2]
        if what == "value":
            s._dispatch(f"{P.SETVALUE} {parts[3]} {parts[4]}", link)
            if not s.safety_sim:
                s.values[parts[3]] = float(parts[4])
            self._reply(link, seq, "ok",
                        f"value {parts[3]} {P._n(s.values.get(parts[3], float('nan')))}")
        elif what == "loop":
            tail = parts[3:]
            limits = tail[6:8] if len(tail) >= 8 else ["-10", "150"]
            s._dispatch(" ".join([P.SERVOLOOP, *tail[:6], *(limits if len(tail) >= 8 else [])]),
                        link)
            with self.lock:
                self._refresh_loops()
            n, i, sp, p, ig, out = tail[:6]
            self._reply(link, seq, "ok", f"loop {n} {i} {sp} {p} {ig} {out} "
                        f"{limits[0]} {limits[1]} err=0")
        elif what == "ramp":
            s._dispatch(" ".join([P.RAMP, *parts[3:7]]), link)
            self._reply(link, seq, "ok", "ramp " + " ".join(parts[3:7]) + " err=0")
        elif what == "loopstop":
            s._dispatch(f"{P.STOPLOOP} {parts[3]}", link)
            if int(parts[3]) == 2:
                self.loop2 = None
            self._reply(link, seq, "ok", f"loopstop {parts[3]} err=0")
        elif what == "mode":
            s._dispatch(f"{P.SETMODE} {parts[3]}", link)
            self._reply(link, seq, "ok", f"mode {parts[3]}")
        elif what == "showtip":
            s._dispatch(f"{P.SHOWTIP} {parts[3]}", link)
            self._reply(link, seq, "ok")
        else:
            self._reply(link, seq, "err", f"bad set {what}")

    # ---- make-safe (contract section 5) ----
    def _height(self) -> float:
        return self.stub.height()

    def make_safe(self, why: str = "stop", route: str = "com") -> None:
        """AFM_SafePark(): the stub's one abort function, run through the v2 ABORT line so the
        golden tap sees what ABORT sees (contract 3: ABORT runs the same function). Called by
        MAKESAFE (route sock), the COM door (route com) and the dead-socket guard."""
        s = self.stub
        with self.lock:
            self._refresh_loops()
            run, self.run = self.run, None                  # no run is left to arm
            stops = s.td_stops
            s._dispatch("ABORT", run or _NullLink())
            if s.td_stops != stops:
                self.vendor_withdrawn = False
                self.igor_calls.append("td_Stop()")
                self.loop2 = None                           # td_Stop undefines the loops
            self.igor_calls.append('td_WriteValue("Output.A",0)')
            self.igor_calls.append(
                'ir_SetPISLoop(2,"Always,Never","Deflection",-10,0,1000,0,"Height",0,150)')
            self.loop2 = (-10.0, 0.0, 150.0)
            self.up_report(run.run if run else "-", f"madesafe {why}", route)

    def on_backup(self, height: float, outcome: str) -> None:
        """The stub's 5 s backup ran (`confirmed` or `fault`)."""
        with self.lock:
            if outcome == "confirmed":
                self.tip_down = False
                self.up_report("-", f"backup {_num(height)} cleared", "-")
            else:
                self.vendor_withdrawn = True
                self.loop2 = None
                self.up_report("-", f"backup {_num(height)} withdraw", "-")

    def _clear_mark(self) -> None:
        self.stub._dispatch("ABORTCONFIRM", _NullLink())
        self.tip_down = False

    def _vendor_withdraw(self) -> bool:
        """DoScanFunc("StopEngageButton"), once per make-safe. True if it ran now."""
        with self.lock:
            if self.vendor_withdrawn:
                return False
            s = self.stub
            s.tap.record_com("v3", 'DoScanFunc("StopEngageButton")')
            self.igor_calls.append('DoScanFunc("StopEngageButton")')
            s.vendor_withdraws += 1
            s.vendor_withdraw()
            self.vendor_withdrawn = s.abort_fault = True
            self.loop2 = None                               # it also kills the loops
            return True

    def _do_makesafe(self, parts, link, seq) -> None:
        self._reply(link, seq, "ok")
        self.make_safe("stop", route="sock")

    def _do_confirmsafe(self, parts, link, seq) -> None:
        with self.lock:
            h = self._height()
            if h < HEIGHT_SAFE_V:
                self._clear_mark()
                self._reply(link, seq, "ok", f"height={_num(h)}")
                self.up_report("-", f"confirmed {_num(h)}", "sock")
            else:
                self._reply(link, seq, "err", f"height={_num(h)}")

    def _do_vendorwithdraw(self, parts, link, seq) -> None:
        ran = self._vendor_withdraw()
        self._reply(link, seq, "ok", "ran" if ran else "already")

    def _do_clearfault(self, parts, link, seq) -> None:
        why = self._host_error(link)
        if why:
            self._reply(link, seq, "err", why)
            return
        self.stub._dispatch("CLEARFAULT", _NullLink())      # AFM_ClearFault: both flags to 0
        self.vendor_withdrawn = False
        self._reply(link, seq, "ok", "fault=0")
