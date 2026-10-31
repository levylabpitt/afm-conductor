"""The single-reader router + UID protocol (docs/IGOR_PYTHON_COMMS.md §7).

These prove the fix for the 2026-08-28 regression OFF-instrument: with one thread
owning the socket receive side and every reply correlated by UID, concurrent reads
during a streaming scan no longer steal each other's frames, a spurious/duplicate
reply is DETECTED (logged + dropped) rather than silently shifting every match, a
lost reply times out ITS waiter only, TELEM routes to the handler, abort is
out-of-band (COM first, TCP fallback), and a version mismatch fails loudly at
connect.
"""

from __future__ import annotations

import queue
import socket
import threading
import time

import pytest

from afm_litho.hardware import bridge_protocol as P
from afm_litho.hardware.bridge_router import BridgeClosed, BridgeRouter, send_abort
from afm_litho.hardware.hardware_arc import HardwareARC


class QueueLink:
    """A controllable BridgeLink stand-in for the router: `read_line` blocks on a
    queue (raising socket.timeout when idle, so the router loop stays alive like a
    real socket), `send` records and can auto-answer via `on_send`, `close` ends the
    stream. Lets a test drive the router deterministically with no real socket."""

    def __init__(self, on_send=None):
        self._q: "queue.Queue[str | None]" = queue.Queue()
        self.sent: list[str] = []
        self.on_send = on_send
        self._closed = False

    def send(self, line: str) -> None:
        self.sent.append(line)
        if self.on_send is not None:
            for reply in (self.on_send(line) or []):
                self._q.put(reply)

    def feed(self, line: str) -> None:
        self._q.put(line)

    def read_line(self, timeout=None):
        if self._closed:
            return None
        try:
            item = self._q.get(timeout=timeout)
        except queue.Empty:
            raise socket.timeout
        return item

    def close(self) -> None:
        self._closed = True
        self._q.put(None)


# --------------------------------------------------------------- TELEM routing --

def test_telem_routes_to_handler_while_capture_routes_to_waiter():
    """A TELEM broadcast (no UID) goes to on_telem; the pass's CAPTURE/DONE (its
    UID) go to the waiter -- interleaved, with neither misread as the other."""
    link = QueueLink()
    router = BridgeRouter(link).start()
    seen: list[tuple] = []
    router.on_telem = lambda x, y, z, d, s, a: seen.append((x, y, z, d, s, a))
    w = router.open_waiter()
    try:
        link.feed(P.msg_telem(0.10, 0.20, 66.0, 0.2, 5.0, 3.0))
        link.feed(P.msg_capture(w.uid, "obj", "XSensor", [1, 2, 3]))
        link.feed(P.msg_telem(0.11, 0.21, 67.0, 0.3, 4.0, 2.0))
        link.feed(P.msg_done(w.uid, "obj"))
        frames = []
        while True:
            verb, rest = w.get(2.0)
            frames.append((verb, rest))
            if verb == P.DONE:
                break
    finally:
        router.stop()
    assert seen == [(pytest.approx(0.10), pytest.approx(0.20), pytest.approx(66.0),
                     pytest.approx(0.2), pytest.approx(5.0), pytest.approx(3.0)),
                    (pytest.approx(0.11), pytest.approx(0.21), pytest.approx(67.0),
                     pytest.approx(0.3), pytest.approx(4.0), pytest.approx(2.0))]
    assert frames[0][0] == P.CAPTURE and frames[0][1] == "obj XSensor 1;2;3"
    assert frames[-1] == (P.DONE, "obj")


def test_telem_dropped_when_no_hook():
    """No on_telem set -> TELEM is quietly skipped, the pass capture still lands."""
    link = QueueLink()
    router = BridgeRouter(link).start()
    w = router.open_waiter()
    try:
        link.feed(P.msg_telem(0.9, 0.9, 0.0, 0.0, 5.0, 3.0))
        link.feed(P.msg_capture(w.uid, "obj", "XSensor", [7, 8]))
        link.feed(P.msg_done(w.uid, "obj"))
        assert w.get(2.0) == (P.CAPTURE, "obj XSensor 7;8")
        assert w.get(2.0) == (P.DONE, "obj")
    finally:
        router.stop()


# ------------------------------------------------------------ desync detection --

def test_unknown_uid_is_logged_and_dropped_real_waiter_unaffected():
    """A spurious VALUE for a bogus UID and a duplicate reply are logged as
    'unknown UID' and dropped; the real waiter still gets its correct value."""
    logs: list[str] = []
    link = QueueLink()
    router = BridgeRouter(link, log=logs.append).start()
    w = router.open_waiter()
    try:
        link.feed(P.msg_value("999", 3.14))      # spurious: no such waiter
        link.feed(P.msg_value(w.uid, 1.5))        # the real reply
        assert w.get(2.0) == (P.VALUE, "1.5")
        # a DUPLICATE reply for the (now-released) uid is also dropped
        link.feed(P.msg_value(w.uid, 9.9))
        time.sleep(0.1)
    finally:
        router.stop()
    assert any("unknown UID" in m and "999" in m for m in logs)
    assert any("unknown UID" in m and str(w.uid) in m for m in logs)


# ----------------------------------------------------------- timeout isolation --

def test_timeout_isolation_one_waiter_times_out_others_fine():
    """A reply that never arrives times out ITS waiter only; a sibling waiter still
    gets its reply, and the router keeps running afterwards."""
    link = QueueLink()
    router = BridgeRouter(link).start()
    w_missing = router.open_waiter()
    w_ok = router.open_waiter()
    try:
        link.feed(P.msg_value(w_ok.uid, 2.0))
        assert w_ok.get(2.0) == (P.VALUE, "2")
        with pytest.raises(queue.Empty):
            w_missing.get(0.2)                     # its reply never came
        # router still alive: a fresh waiter still works
        w2 = router.open_waiter()
        link.feed(P.msg_value(w2.uid, 5.0))
        assert w2.get(2.0) == (P.VALUE, "5")
    finally:
        router.stop()


# ------------------------------------------------------------- closed link wake --

def test_close_wakes_a_blocked_waiter():
    """A dropped link wakes a blocked waiter with BridgeClosed at once, rather than
    hanging it out to its full timeout."""
    link = QueueLink()
    router = BridgeRouter(link).start()
    w = router.open_waiter()

    def closer():
        time.sleep(0.1)
        link.close()

    threading.Thread(target=closer, daemon=True).start()
    try:
        with pytest.raises(BridgeClosed):
            w.get(5.0)
    finally:
        router.stop()


# -------------------------------------------------------- concurrency: no steal --

def test_concurrent_reads_during_streaming_scan_lose_no_frames():
    """N concurrent td_ReadValue calls WHILE a pass streams CAPTUREs: each read
    returns its correct value (matched by UID) AND the scan's CAPTUREs all reach the
    pass waiter -- zero stolen frames. This is the core invariant."""
    store = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}

    def respond(line: str):
        parts = line.split(" ")
        if parts[0] == P.GETVALUE:                 # GETVALUE <uid> <name>
            return [P.msg_value(parts[1], store[parts[2]])]
        return []

    link = QueueLink(on_send=respond)
    hw = HardwareARC(link, timeout=5.0)            # builds + starts the router
    # a pass waiter, opened first (uid "1"): a feeder thread streams its CAPTUREs +
    # TELEM while reads run concurrently.
    scan = hw.router.open_waiter()
    n_lines = 200

    def feeder():
        for i in range(n_lines):
            hw.link.feed(P.msg_capture(scan.uid, f"L{i}", "ZSensor",
                                       [float(i)] * 4))
            if i % 10 == 0:
                hw.link.feed(P.msg_telem(0.1, 0.2, 66.0, 0.2, 5.0, 3.0))
        hw.link.feed(P.msg_scandone(scan.uid, n_lines))

    results: dict[int, float] = {}
    errors: list[str] = []

    def reader(idx: int):
        name = "ABCD"[idx % 4]
        try:
            results[idx] = hw.td_ReadValue(name)
        except Exception as exc:                   # noqa: BLE001
            errors.append(f"{idx}: {exc}")

    ft = threading.Thread(target=feeder)
    readers = [threading.Thread(target=reader, args=(k,)) for k in range(40)]
    ft.start()
    for t in readers:
        t.start()
    for t in readers:
        t.join(5.0)
    ft.join(5.0)

    # every read returned ITS value
    assert not errors, errors
    assert len(results) == 40
    for k, v in results.items():
        assert v == pytest.approx(store["ABCD"[k % 4]])
    # the scan lost ZERO frames: all n_lines CAPTUREs + the terminal reached the
    # pass waiter, never stolen by a concurrent read.
    caps = 0
    while True:
        verb, _rest = scan.get(2.0)
        if verb == P.SCANDONE:
            break
        assert verb == P.CAPTURE
        caps += 1
    assert caps == n_lines
    hw.close()


# ---------------------------------------------------------- abort: com then tcp --

class _FakeCom:
    """A COM Igor stand-in: records execute() calls, optionally raising to force the
    TCP fallback."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[str] = []

    def execute(self, command: str, *, check: bool = True):
        self.calls.append(command)
        if self.fail:
            raise RuntimeError("COM unavailable")
        return None


class _SendOnlyLink:
    def __init__(self):
        self.sent: list[str] = []

    def send(self, line: str) -> None:
        self.sent.append(line)


def test_abort_uses_com_first_and_retracts():
    """STOP-AND-RETRACT (docs/ABORT_SEQUENCE.md): the COM abort fires AFM_SafePark(),
    not the old non-retracting AFM_Abort(). AFM_SafePark calls AFM_Abort first, so the
    stop + bias-zero + terminal-frame sequencing is unchanged -- it only ADDS the
    retract, which is what every abort path was missing."""
    link = _SendOnlyLink()
    com = _FakeCom()
    how = send_abort(link, com)
    assert how == "com"
    assert com.calls == ["AFM_SafePark()"]           # stop AND retract, over COM
    assert link.sent == []                            # no TCP ABORT needed


def test_abort_falls_back_to_tcp_when_com_fails():
    logs: list[str] = []
    link = _SendOnlyLink()
    com = _FakeCom(fail=True)
    how = send_abort(link, com, log=logs.append)
    assert how == "tcp"
    assert com.calls == ["AFM_SafePark()"]            # COM tried FIRST
    # the socket ABORT verb, which the UID-aware .ipf dispatches to AFM_SafePark too
    assert link.sent == [P.cmd_abort()]
    assert any("COM path failed" in m for m in logs)


def test_abort_tcp_only_when_no_com():
    link = _SendOnlyLink()
    assert send_abort(link, None) == "tcp"
    assert link.sent == [P.cmd_abort()]


class _FakeExecute2App:
    """A COM Igor automation-server stand-in exposing Execute2, so a REAL Igor
    wrapper drives it. Returns a fixed (errorCode, errorMsg, history, results),
    letting a test force an Igor-LEVEL error with a clean COM transport."""

    def __init__(self, code: int = 0):
        self.code = int(code)
        self.commands: list[str] = []

    def Execute2(self, allow_interrupt, binary, command):
        self.commands.append(command)
        msg = "AFM_SafePark() not found" if self.code else ""
        return (self.code, msg, "", "")


def test_abort_falls_back_when_igor_reports_error():
    """COM transport SUCCEEDS but Igor REJECTS the park (error_code != 0): the real
    Igor.execute(check=True) raises, so send_abort treats it as a COM failure and
    fires the TCP ABORT fallback -- an Igor-level rejection can no longer be
    swallowed as a successful abort (Fix 1)."""
    from afm_litho.hardware.igor import Igor

    app = _FakeExecute2App(code=1001)               # Igor reports a nonzero error
    igor = Igor(app=app)
    link = _SendOnlyLink()
    logs: list[str] = []
    how = send_abort(link, igor, log=logs.append)
    assert how == "tcp"
    assert app.commands == ["AFM_SafePark()"]       # COM tried FIRST
    assert link.sent == [P.cmd_abort()]             # then the socket ABORT fired
    assert any("COM path failed" in m for m in logs)


# ------------------------------------------------------ lr-pass single-UID guard --

def test_lr_pass_guard_blocks_second_concurrent_pass():
    """Two OVERLAPPING lr-class passes would clobber the shared Igor root:AFM_lrUID;
    the guard makes the second fail LOUDLY. A single pass, and a fresh pass after the
    first releases, are unaffected (Fix 3)."""
    router = BridgeRouter(QueueLink())              # not started: the guard is pure state
    with router.lr_pass():                          # normal single pass: fine
        with pytest.raises(RuntimeError, match="already in flight"):
            with router.lr_pass():                  # a second, overlapping: refused
                pass
    # released after the first exits: a fresh lr-class pass works again
    with router.lr_pass():
        pass


def test_lr_pass_guard_releases_on_exception():
    """An exception inside an lr-class pass still releases the guard (finally), so a
    later pass is not wedged out."""
    router = BridgeRouter(QueueLink())
    with pytest.raises(ValueError):
        with router.lr_pass():
            raise ValueError("boom")
    with router.lr_pass():                          # slot freed despite the error
        pass


# ------------------------------------------------------------------- handshake --

def test_handshake_rejects_version_mismatch():
    """A bridge that echoes a different proto in its STARTUP reply is a mismatched
    dialect -> loud connect-time failure, not a silent mid-run desync."""
    def respond(line: str):
        parts = line.split(" ")
        if parts[0] == P.STARTUP:                     # STARTUP <uid> <version>
            # echo the uid but the WRONG proto (a legacy v1 peer)
            return [f"{P.STARTUP} {parts[1]} proto=1;pixels=1;"]
        return []

    hw = HardwareARC(QueueLink(on_send=respond), timeout=2.0)
    try:
        with pytest.raises(RuntimeError, match="protocol mismatch"):
            hw.start()
    finally:
        hw.close()


def test_handshake_rejects_non_uid_peer():
    """A legacy non-UID bridge replies STARTUP with no echoed UID, so the router
    can't match it and the handshake fails loudly (rather than hanging forever)."""
    def respond(line: str):
        if line.split(" ")[0] == P.STARTUP:
            return ["STARTUP pixels=1;lines=1;"]      # untagged: no uid
        return []

    hw = HardwareARC(QueueLink(on_send=respond), timeout=0.5)
    try:
        with pytest.raises(RuntimeError, match="handshake failed"):
            hw.start()
    finally:
        hw.close()
