"""A small model of Igor's v3 side, written from the contract alone.

It exists so the verb-level conformance tests are themselves tested: they run against this
model today and against the real fake Igor (slice W3) once that lands. It is not the fake Igor
and takes no part in the product. It speaks section 2 and section 5 over a loopback socket, on
a manual clock (`advance`), and keeps the Igor-level lines it would have run in `igor_lines`.
"""

from __future__ import annotations

import socket
import threading

from . import contract as C

IPF_VERSION = 3
IPF_SHA = "0123456789abcdef"
RUN_SECONDS = 2.0     # how long a PLAY keeps a pass active on the manual clock


class ReferenceIgor:
    speaks_v3 = True
    concurrent_connections = True

    def __init__(self):
        self.now = 0.0
        self.height = 0.0
        self.igor_lines: list[str] = []
        self.lock = threading.RLock()
        self.useq = 0
        self.ring: list[tuple[int, str]] = []
        self.mark = 0                  # abortStopped
        self.armed = 0                 # passArmed: a PLAY armed since the last td_Stop
        self.fault = 0                 # abortFault
        self.vendor_withdrawn = 0
        self.backup_at: float | None = None
        self.run: str | None = None
        self.run_ends = 0.0
        self.tip_down = False
        self.loop2 = (0.0, -10.0, 150.0)
        self.waves: dict[str, list[float]] = {}
        self.totals: dict[str, int] = {}
        self.host: _Conn | None = None
        self.dead_since: float | None = None
        self._next_tick = 0.0
        self._srv = socket.socket()
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(4)
        self.port = self._srv.getsockname()[1]
        self._closed = False
        self._conns: list = []
        threading.Thread(target=self._accept, daemon=True).start()

    # ---- handle API used by the tests ------------------------------------------------
    def advance(self, seconds: float) -> None:
        end = self.now + seconds
        while self.now < end - 1e-9:
            with self.lock:
                self.now = min(end, self.now + 0.05)
                self._timers()

    def set_height(self, v: float) -> None:
        self.height = float(v)

    def drop_host(self) -> None:
        with self.lock:
            if self.host:
                self.host.close()
                self.host, self.dead_since = None, self.now

    def com_safe_park(self) -> None:
        with self.lock:
            self._safe_park("stop", "com")

    def vendor_withdraws(self) -> int:
        return sum("StopEngageButton" in ln for ln in self.igor_lines)

    def close(self) -> None:
        self._closed = True
        self._srv.close()
        with self.lock:
            for c in list(self._conns):
                c.close()

    # ---- socket plumbing -------------------------------------------------------------
    def _accept(self):
        while not self._closed:
            try:
                s, _ = self._srv.accept()
            except OSError:
                return
            c = _Conn(self, s)
            self._conns.append(c)
            threading.Thread(target=c.serve, daemon=True).start()

    # ---- Igor behaviour ---------------------------------------------------------------
    def _stamp(self, route: str) -> str:
        return f"{self.mark} {self.fault} {route}"

    def _report(self, run: str, event: str, fields: str = "", route: str = "sock") -> None:
        self.useq += 1
        line = (f"REPORT {self.useq} {int(self.now * 60)} {self.mark} {self.fault} {route} "
                f"{run or '-'} {event} {fields}").rstrip()
        self.ring.append((self.useq, line))
        del self.ring[:-C.RING_SIZE]
        if self.host:
            self.host.send(line)

    def _ring_span(self) -> str:
        return f"{self.ring[0][0]}-{self.ring[-1][0]}" if self.ring else "0-0"

    def _safe_park(self, why: str, route: str) -> None:
        self.run = None                              # player flags cleared first (rule 9)
        if not self.mark or self.armed:
            self.igor_lines.append(C.TD_STOP)
            self.mark, self.armed = 1, 0
            self.vendor_withdrawn = 0
            self.backup_at = self.now + C.BACKUP_S
        self.igor_lines.append(C.BIAS_ZERO)
        self.igor_lines.append(C.RETRACT)
        self.tip_down = False
        self._report("-", "madesafe", why, route)

    def _vendor_withdraw(self) -> bool:
        if self.vendor_withdrawn:
            return False
        self.igor_lines.append('DoScanFunc("StopEngageButton")')
        self.vendor_withdrawn = 1
        self.fault = 1
        return True

    def _timers(self) -> None:
        if self.run and self.now >= self.run_ends:
            run, self.run = self.run, None
            self._report(run, "done")
        if self.backup_at is not None and self.now >= self.backup_at:
            self.backup_at = None
            if self.mark:
                if self.height < C.HEIGHT_SAFE_V:
                    self.mark = 0
                    self._report("-", "backup", f"{self.height:g} cleared")
                else:
                    self._vendor_withdraw()
                    self._report("-", "backup", f"{self.height:g} withdraw")
        if (self.host is None and self.dead_since is not None and self.run is None
                and self.tip_down and self.now - self.dead_since > C.SOCK_DEAD_PARK_S):
            self.dead_since = None
            self._safe_park("socket", "sock")
        if self.host and self.now >= self._next_tick:
            hz = C.TICK_HZ_PASS if self.run else C.TICK_HZ_IDLE
            self._next_tick = self.now + 1.0 / hz
            self.useq += 1
            self.host.send(f"TICK {self.useq} {int(self.now * 60)} {self.mark} {self.fault} - "
                           f"0 0 {self.height:g} 0 5 0")

    def host_lost(self, conn) -> None:
        with self.lock:
            if self.host is conn:
                self.host = None
                self.dead_since = self.now

    # ---- verbs -------------------------------------------------------------------------
    def handle(self, conn, lines: list[str]) -> None:
        with self.lock:
            urgent = [ln for ln in lines if ln.split(" ")[0] == "MAKESAFE"]
            queue = urgent + [ln for ln in lines if ln not in urgent] if urgent else lines
            for ln in queue:
                if urgent and ln not in urgent and conn.is_host_verb(ln):
                    seq = ln.split(" ")[1]
                    conn.send(f"REPLY {seq} err dropped")
                    continue
                self._one(conn, ln)

    def _hello_body(self) -> str:
        return (f"proto={C.PROTO};ipf={IPF_VERSION};sha={IPF_SHA};MicroscopeID=4;"
                f"clock={int(self.now * 60)};ring={self._ring_span()};mark={self.mark};"
                f"fault={self.fault};")

    def _one(self, conn, line: str) -> None:
        t = line.split(" ")
        verb = t[0]
        if verb in ("STARTUP", "GETVALUE", "ABORT"):
            return self._v2(conn, t)
        try:
            seq = int(t[1])
        except (IndexError, ValueError):
            return conn.send("REPLY 0 err malformed")
        r = conn.send
        if verb == "HELLO":
            if int(t[2]) != C.PROTO:
                return r(f"REPLY {seq} err mismatch {self._hello_body()}")
            conn.matched = True
            if self.host is not conn:
                if self.host:
                    self.host.close()
                self.host, self.dead_since = conn, None
            return r(f"REPLY {seq} ok {self._hello_body()}")
        if verb == "READ":
            if t[2] == "value":
                return r(f"REPLY {seq} ok {self.height:g}" if t[3] == "Height"
                         else f"REPLY {seq} ok 0")
            if t[2] == "flag":
                return (r(f"REPLY {seq} ok {self.fault}") if t[3] == "abortFault"
                        else r(f"REPLY {seq} err bad flag {t[3]}"))
            if t[2] == "gv":
                return r(f"REPLY {seq} ok {'nan' if t[3] != 'abortFault' else self.fault}")
            return r(f"REPLY {seq} ok Crosspoint=1;")
        if verb == "STATUS":
            sp, lo, hi = self.loop2
            return r(f"REPLY {seq} ok pass={'move' if self.run else 'none'};run={self.run or '-'};"
                     f"step=0/0;mark={self.mark};fault={self.fault};"
                     f"tipdown={int(self.tip_down)};loop2={sp:g},{lo:g},{hi:g};"
                     f"sock_dead_s={0 if self.host else self.now - (self.dead_since or 0):g};"
                     f"ring={self._ring_span()};tick={int(self.now * 60)};")
        if verb == "FETCH":
            frm = int(t[2])
            if self.ring and frm < self.ring[0][0] and frm != 0:
                return r(f"REPLY {seq} err gone {self.ring[0][0]}")
            for u, ln in self.ring:
                if u >= frm:
                    conn.send(ln)
            return r(f"REPLY {seq} ok {self._ring_span()}")
        if verb == "MAKESAFE":
            r(f"REPLY {seq} ok")
            return self._safe_park("stop", "sock")
        if verb == "CONFIRMSAFE":
            if self.height < C.HEIGHT_SAFE_V:
                self.mark = 0
                self.backup_at = None
                self.tip_down = False
                return r(f"REPLY {seq} ok height={self.height:g}")
            return r(f"REPLY {seq} err height={self.height:g}")
        if verb == "VENDORWITHDRAW":
            return r(f"REPLY {seq} ok ran" if self._vendor_withdraw()
                     else f"REPLY {seq} ok already")
        if not conn.matched:
            return r(f"REPLY {seq} err host")
        if verb == "CLEARFAULT":          # section 11 (W3): AFM_ClearFault zeroes both flags
            self.mark = self.fault = 0
            self.vendor_withdrawn = 0
            return r(f"REPLY {seq} ok fault=0")
        if verb in ("LOAD", "LOADAPP"):
            name = t[2]
            vals = [float(x) for x in t[-1].split(";") if x]
            have = (self.waves.get(name, []) if verb == "LOADAPP" else []) + vals
            self.waves[name] = have
            if verb == "LOAD":
                self.totals[name] = int(t[3])
            total = self.totals.get(name, len(have))
            return r(f"REPLY {seq} ok {name} {len(have)}/{total}")
        if verb == "PLAY":
            if self.run:
                return r(f"REPLY {seq} err busy")
            if self.mark:
                return r(f"REPLY {seq} err mark")
            if self.fault:
                return r(f"REPLY {seq} err fault")
            if t[2] not in C.PLAY_KINDS:
                return r(f"REPLY {seq} err kind")
            self.run, self.run_ends = t[3], self.now + RUN_SECONDS
            self.armed = 1
            r(f"REPLY {seq} ok armed {t[3]}")
            return self._report(t[3], "armed")
        if verb == "SET":
            if t[2] == "loop":
                sp = float(t[5])
                self.loop2 = (sp, *(float(x) for x in t[9:11])) if len(t) > 10 else (sp, -10, 150)
                if t[3] == "2":
                    self.tip_down = sp != -10
                return r(f"REPLY {seq} ok loop {' '.join(t[3:9])} err=0")
            if t[2] == "value":
                return r(f"REPLY {seq} ok value {t[3]} {t[4]}")
            if t[2] == "mode":
                return r(f"REPLY {seq} ok mode {t[3]}")
            if t[2] == "showtip":
                return r(f"REPLY {seq} ok")
            return r(f"REPLY {seq} err bad {t[2]}")
        return r(f"REPLY {seq} err unknown verb")

    def _v2(self, conn, t) -> None:
        if t[0] == "ABORT":
            self._safe_park("stop", "sock")
            conn.send("ABORTED")
        elif t[0] == "GETVALUE":
            conn.send(f"VALUE {t[1]} {self.height:g}")
        else:
            conn.send(f"STARTUP {t[1]} MicroscopeID=4;")


class _Conn:
    def __init__(self, igor: ReferenceIgor, sock: socket.socket):
        self.igor, self.sock, self.matched = igor, sock, False

    def is_host_verb(self, line: str) -> bool:
        return line.split(" ")[0] in C.HOST_ONLY

    def send(self, line: str) -> None:
        try:
            self.sock.sendall(line.encode("latin1") + b"\n")
        except OSError:
            pass

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()

    def serve(self) -> None:
        buf = b""
        try:
            while True:
                data = self.sock.recv(65536)
                if not data:
                    break
                buf += data
                lines = buf.split(b"\n")
                buf = lines.pop()
                self.igor.handle(self, [ln.decode("latin1").rstrip("\r") for ln in lines if ln])
        except OSError:
            pass
        finally:
            self.igor.host_lost(self)
            self.close()
