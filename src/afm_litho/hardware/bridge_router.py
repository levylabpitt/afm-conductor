"""Single-reader router for the bridge socket's receive side.

Invariants (docs/IGOR_PYTHON_COMMS.md §3, §8): ONE thread, this router, calls
`link.read_line()`; replies go to waiters by UID; an unmatched UID or unknown verb is logged
and dropped, never guessed; uid-less TELEM routes to `on_telem` by type; the router never
sends, so abort stays out-of-band. A waiter receives `(verb, rest)`, `rest` being the line
after the verb and the UID.
"""

from __future__ import annotations

import queue
import socket
import threading
from contextlib import contextmanager

from .. import clock
from . import bridge_protocol as P

# Verbs addressed to a waiter by UID; any other verb is dropped and logged.
_REPLY_VERBS = frozenset((
    P.VALUE, P.XPT, P.STARTUP, P.DONE, P.ABORTED, P.ERR, P.SCANDONE, P.CAPTURE))

# A terminal removes its waiter, so a duplicate/late reply logs as "unknown UID" (desync
# detection). CAPTURE is the only non-terminal reply.
_TERMINAL = frozenset((
    P.VALUE, P.XPT, P.STARTUP, P.DONE, P.ABORTED, P.ERR, P.SCANDONE))

# Expected unmatched noise: ABORT is uid-less, so its ABORTED (and a post-abort SCANDONE)
# may find no waiter.
_QUIET_UNMATCHED = frozenset((P.ABORTED, P.SCANDONE))

# Put to every waiter on close; Waiter.get raises BridgeClosed so the owner unwinds at once.
_CLOSED = ("\x00CLOSED", "")


class BridgeClosed(RuntimeError):
    """The bridge link closed while a waiter was awaiting a reply."""


class BridgeReadTimeout(RuntimeError):
    """A bounded read gave up before its reply arrived.

    Distinct from RuntimeError so a fail-safe caller reports the retract UNVERIFIED
    instead of a protocol fault. Fail-safe reads pass a short explicit timeout, not
    HardwareARC.timeout (30 s). docs/FLEX_INTEGRATION_PLAN.md §F.2.
    """


class Waiter:
    """A UID plus its own queue of `(verb, rest)` frames, filled by the router."""

    __slots__ = ("uid", "_q")

    def __init__(self, uid: str):
        self.uid = uid
        self._q: "queue.Queue[tuple[str, str]]" = queue.Queue()

    def put(self, frame: tuple[str, str]) -> None:
        self._q.put(frame)

    def get(self, timeout: float | None = None) -> tuple[str, str]:
        """Next frame. Raises queue.Empty on this waiter's own timeout, or BridgeClosed
        if the link dropped."""
        frame = clock.queue_get(self._q, timeout)          # queue.Empty propagates
        if frame is _CLOSED:
            raise BridgeClosed("bridge closed while awaiting reply")
        return frame


class BridgeRouter:
    """Owns the receive side of one BridgeLink. Callers open_waiter(), send straight on
    the (locked) link, then block on Waiter.get; a send never waits on the router.
    """

    def __init__(self, link, *, on_telem=None, poll_timeout: float = 2.0,
                 log=None):
        self._link = link
        # on_telem(x, y, zdrive, defl, sum, amp) per TELEM frame; None skips TELEM.
        # CONTRACT: runs INLINE on the reader thread, so it MUST be non-blocking and MUST NOT
        # send, open_waiter/get or read; a blocking callback stalls ALL bridge traffic. Hand
        # loop-bound work off (run_coroutine_threadsafe).
        self.on_telem = on_telem
        # Called ONCE from _mark_closed (every close path); the service points it at
        # hub.mark_bridge_closed (plan §C.1). Same inline, non-blocking contract as on_telem.
        self.on_closed = None
        # on_igor_abort(frame) for every abort Igor reports (ABORTED, SCANDONE -1; wire v3:
        # REPORT madesafe / aborted), called BEFORE the frame reaches its waiter so the
        # run's abort flag is already set when the pass wakes. Same contract as on_telem.
        self.on_igor_abort = None
        # Read timeout between stop-flag checks. A concurrent send shares the socket
        # timeout, so keep it longer than any single bounded transfer.
        self._poll_timeout = float(poll_timeout)
        self._log = log or (lambda msg: None)
        self._waiters: dict[str, Waiter] = {}
        self._lock = threading.Lock()
        self._seq = 0
        # lr_pass guard: Igor keeps the lr-class pass UID in ONE global (root:AFM_lrUID).
        self._lr_lock = threading.Lock()
        self._lr_in_flight = False
        self._closed = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --------------------------------------------------------------- lifecycle --

    @property
    def closed(self) -> bool:
        """True once `_mark_closed` has fired. `on_closed` is a one-shot EDGE, so anything
        that wires it late must also check this LEVEL once, right after (hub.bind_hardware)."""
        return self._closed

    def start(self) -> "BridgeRouter":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._loop, name="bridge-router", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._mark_closed()

    # ---------------------------------------------------------------- waiters --

    def open_waiter(self) -> Waiter:
        """Register a waiter under a fresh UID; if already closed, pre-arm it with CLOSED."""
        with self._lock:
            self._seq += 1
            uid = str(self._seq)
            w = Waiter(uid)
            if self._closed:
                w.put(_CLOSED)
            else:
                self._waiters[uid] = w
        return w

    def release(self, uid: str) -> None:
        """Unregister a waiter. Owners call this in `finally` so a timed-out request
        cannot leak a registration (the router auto-releases only on a terminal)."""
        with self._lock:
            self._waiters.pop(uid, None)

    # ----------------------------------------------------------- lr-pass guard --

    @contextmanager
    def lr_pass(self):
        """Guard the SINGLE in-flight lr-class pass (fire / armfire / lithochain).

        afm_litho_run.ipf keeps the lr-class pass UID in one global, root:AFM_lrUID,
        shared by FIRE, ARMFIRE and LITHOCHAIN; an overlapping second pass would mis-tag
        its DONE, so it fails loudly here. Scans use root:AFM_scUID (SCANCHAIN), are not
        guarded, and may overlap an lr-class pass.
        """
        with self._lr_lock:
            if self._lr_in_flight:
                raise RuntimeError(
                    "an lr-class pass (fire/armfire/lithochain) is already in "
                    "flight; a second would clobber the shared Igor root:AFM_lrUID "
                    "and mis-tag its DONE. lr-class passes must be sequential.")
            self._lr_in_flight = True
        try:
            yield
        finally:
            with self._lr_lock:
                self._lr_in_flight = False

    # ------------------------------------------------------------- read loop --

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                line = self._link.read_line(timeout=self._poll_timeout)
            except socket.timeout:
                continue                                  # nothing yet: re-check stop
            except Exception as exc:                      # noqa: BLE001
                # Broken link: exit cleanly (waking waiters below); OSError is the normal case.
                if not isinstance(exc, OSError):
                    self._log(f"bridge router: read failed ({exc!r}); reader exiting")
                break
            if line is None:
                break                                     # peer closed
            if line.partition(" ")[0] != P.TELEM:              # TELEM is high-rate, like the stream's
                P.note_wire("read", line)
            try:
                self._route(line)
            except Exception as exc:                      # noqa: BLE001
                # A malformed line must never kill the single reader: log + carry on.
                self._log(f"bridge router: dropped malformed line {line!r} ({exc})")
        self._mark_closed()

    def _route(self, line: str) -> None:
        verb, _, tail = line.partition(" ")
        if verb == P.TELEM:
            tp = P.parse_telem(line)
            cb = self.on_telem
            if tp is not None and cb is not None:
                # Inline on the reader thread (see on_telem); a raising callback is logged.
                try:
                    cb(tp[0], tp[1], tp[2], tp[3], tp[4], tp[5])
                except Exception as exc:                  # noqa: BLE001
                    self._log(f"bridge router: on_telem raised ({exc})")
            return
        if verb not in _REPLY_VERBS:
            self._log(f"bridge router: unknown verb {verb!r} (dropped)")
            return
        uid, _, rest = tail.partition(" ")
        if verb == P.ABORTED or (verb == P.SCANDONE and rest.strip().startswith("-")):
            self.note_igor_abort(line)
        with self._lock:
            w = self._waiters.get(uid)
            if w is not None and verb in _TERMINAL:
                self._waiters.pop(uid, None)              # terminal ends the waiter
        if w is None:
            if verb not in _QUIET_UNMATCHED:
                self._log(f"bridge router: reply for unknown UID {uid!r} "
                          f"({verb}) -- dropped")
            return
        w.put((verb, rest))

    def note_igor_abort(self, frame: str) -> None:
        """Hand an abort report to `on_igor_abort`; a raising hook is logged."""
        cb = self.on_igor_abort
        if cb is None:
            return
        try:
            cb(frame)
        except Exception as exc:                          # noqa: BLE001
            self._log(f"bridge router: on_igor_abort raised ({exc})")

    def _mark_closed(self) -> None:
        """Flag closed and wake every waiter with CLOSED. The single choke point for "the
        bridge is gone" (reader exit, peer close, stop()), so `on_closed` fires from one
        place; a bad hook cannot keep waiters from being woken."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            waiters = list(self._waiters.values())
            self._waiters.clear()
        for w in waiters:
            w.put(_CLOSED)
        cb = self.on_closed
        if cb is not None:
            try:
                cb()
            except Exception as exc:                      # noqa: BLE001
                self._log(f"bridge router: on_closed raised ({exc})")


def send_abort(link, com_igor=None, *, log=None) -> str:
    """Alias of send_safe_park: fires AFM_SafePark, Igor's one abort function, so every
    abort retracts. Re-firing is harmless: td_Stop runs once per abort (Igor's flag) and
    the -10 V re-arm is absolute. Returns "com" or "tcp".
    """
    return send_safe_park(link, com_igor, log=log)


def send_safe_park(link, com_igor=None, *, log=None) -> str:
    """Safe-park out-of-band, never queued behind the router (docs/IGOR_PYTHON_COMMS.md §5).

      1. COM first: `AFM_SafePark()` with check=True, so an Igor-level rejection (function
         not loaded, runtime error, wrong .ipf) raises and falls through to TCP.
      2. TCP fallback: send `ABORT` (send-only); the .ipf dispatches it to AFM_SafePark.

    Returns "com" or "tcp".
    """
    if com_igor is not None:
        try:
            # check=True: a nonzero Igor error_code RAISES, so an Igor-level
            # rejection falls back to TCP -- never swallowed as a successful park.
            com_igor.execute("AFM_SafePark()", check=True)
            return "com"
        except Exception as exc:                          # noqa: BLE001
            if log:
                log(f"safe_park: COM path failed ({exc}); falling back to TCP ABORT")
        # Wire v3 only (contract §12): the raw make-safe lines over COM, before the socket.
        raw = getattr(link, "raw_com_make_safe", None)
        if raw is not None and raw(com_igor, log):
            return "com_raw"
    link.send(P.cmd_abort())
    return "tcp"
