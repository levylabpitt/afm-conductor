"""What the conformance suite runs against.

A *target* is any object that speaks the v3 contract over a socket. `LineSession` is a plain
client for it (no service code, so the tests judge the contract and not W2's client). A target
hands the tests a *handle* with:

    port                    TCP port to connect to
    advance(seconds)        move the target's (simulated) clock
    set_height(volts)       what `td_ReadValue("Height")` returns
    drop_host()             close the host socket from Igor's side
    com_safe_park()         the COM route: `Execute2("AFM_SafePark()")`
    vendor_withdraws()      how many times the vendor withdraw ran
    igor_lines              the Igor-level lines it ran, in order (list of str)
    close()

Two targets exist today. `reference` is `reference_igor.ReferenceIgor`, a model written from
the contract; it keeps the verb tests themselves honest. `fake` is the stub of slice W3
(`FakeIgorBridge`, flag `speaks_v3`), found by `fake_available()`; the scenario equivalence
tests use it with the Python client of slice W2. Anything else can be named with
`AFM_LITHO_V3_TARGET=module:factory`, a no-argument callable returning a handle.
"""

from __future__ import annotations

import importlib
import os
import queue
import socket
import threading
import time

import pytest

from . import contract as C

HANDLE_METHODS = ("advance", "set_height", "drop_host", "com_safe_park")


def env_factory():
    spec = os.environ.get("AFM_LITHO_V3_TARGET", "").strip()
    if not spec:
        return None
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)


def fake_available() -> tuple[bool, str]:
    """Whether the W2 client and the W3 fake exist, for the scenario equivalence tests."""
    try:
        from afm_litho.hardware import wire, wire_v3  # noqa: F401
        from afm_litho.hardware.igor_stub import FakeIgorBridge
    except ImportError as exc:
        return False, f"v3 client not available yet ({exc})"
    if not hasattr(wire, "use_version"):
        return False, "afm_litho.hardware.wire.use_version missing"
    if not getattr(FakeIgorBridge({}), "speaks_v3", False):
        return False, "FakeIgorBridge does not speak v3 yet (slice W3)"
    return True, ""


class LineSession:
    """One raw connection to a target, speaking newline-terminated latin1 lines."""

    def __init__(self, port: int, timeout: float = 5.0):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self.timeout = timeout
        self.seq = 0
        self.lines: queue.Queue[str] = queue.Queue()
        self.seen: list[C.Up] = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        buf = b""
        try:
            while True:
                data = self.sock.recv(65536)
                if not data:
                    break
                buf += data
                parts = buf.split(b"\n")
                buf = parts.pop()
                for p in parts:
                    self.lines.put(p.decode("latin1").rstrip("\r"))
        except OSError:
            pass
        self.lines.put("")          # EOF marker

    # ---- sending ----
    def raw(self, *lines: str) -> None:
        self.sock.sendall("".join(f"{ln}\n" for ln in lines).encode("latin1"))

    def next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def send(self, verb: str, *args) -> int:
        seq = self.next_seq()
        self.raw(C.down(seq, verb, *args))
        return seq

    # ---- receiving ----
    def pump(self, wait: float = 0.15) -> list[C.Up]:
        """Everything that arrives within `wait` seconds of quiet."""
        got = []
        stop_at = time.monotonic() + max(1.0, 4 * wait)      # TICKs may never go quiet
        while time.monotonic() < stop_at:
            try:
                ln = self.lines.get(timeout=wait)
            except queue.Empty:
                return got
            if ln:
                up = C.parse_up(ln) if ln.split(" ")[0] in ("REPLY", "REPORT", "CAP", "TICK") \
                    else None
                if up:
                    got.append(up)
                    self.seen.append(up)
        return got

    def reply(self, seq: int, timeout: float | None = None) -> C.Up:
        end = time.monotonic() + (timeout or self.timeout)
        while time.monotonic() < end:
            for up in self.seen:
                if up.kind == "REPLY" and up.seq == seq:
                    return up
            self.pump(0.05)
        raise TimeoutError(f"no REPLY {seq}; saw {[u.raw for u in self.seen][-5:]}")

    def ask(self, verb: str, *args) -> C.Up:
        return self.reply(self.send(verb, *args))

    def reports(self, event: str | None = None) -> list[C.Up]:
        self.pump(0.05)
        return [u for u in self.seen if u.kind == "REPORT" and event in (None, u.event)]

    def hello(self) -> C.Up:
        return self.ask("HELLO", C.PROTO, "conformance")

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def require(handle, *names) -> None:
    """Skip the test when the target's handle lacks a method it needs."""
    missing = [n for n in names if not hasattr(handle, n)]
    if missing:
        pytest.skip(f"target handle lacks {missing}")


class FakeIgorHandle:
    """The W3 fake (`FakeIgorBridge`) as a target handle, on the testing clock.

    The fake serves one connection at a time, so it has no `concurrent_connections`; its
    Igor-level make-safe lines are `stub.v3.igor_calls`.
    """

    def __init__(self, run_seconds: float = 2.0):
        from afm_litho.hardware.igor_stub import FakeIgorBridge
        self.stub = FakeIgorBridge({"MicroscopeID": 4.0, "Height": 0.0, "XSensor": 0.0,
                                    "YSensor": 0.0}, done_delay=run_seconds, telem_hz=4.0)
        self.port = self.stub.serve_in_thread()

    @property
    def igor_lines(self):
        return self.stub.v3.igor_calls

    def advance(self, seconds: float) -> None:
        self.stub.v3_advance(seconds)
        time.sleep(0.15)                     # the clock's timer thread runs the callbacks

    def set_height(self, volts: float) -> None:
        """Pin Height at `volts`: the retract does not move it (a stuck tip, or a chosen low)."""
        self.stub.safe_park_zeros = False
        self.stub.set_height(volts)

    def drop_host(self) -> None:
        self.stub.drop_host_socket()

    def com_safe_park(self) -> None:
        self.stub.v3.make_safe("stop", route="com")

    def vendor_withdraws(self) -> int:
        """How many times DoScanFunc("StopEngageButton") ran (the verb and the backup)."""
        return self.stub.vendor_withdraws

    def close(self) -> None:
        self.stub.stop()
