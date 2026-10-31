"""The FLEX provider: two ZMQ REP sockets bound inside this FastAPI process, sharing
the `hub` singleton.

Truth document: docs/FLEX_INTEGRATION_PLAN.md Sec B (native provider, the two-socket
topology, the deadlock rule, the guardrails). Wire surface: `flex_dispatch.py` and
docs/FLEX_PROVIDER.md. Rig checks: docs/FLEX_RIG_RUNBOOK.md "02".

MILESTONE 3 ADDS THE CONTROL PLANE. The command socket now serves `acquireControl` /
`releaseControl` / `heartbeat` and the token-free `abort` / `safePark`. Those last two
are the first verbs here that touch the tip, and both only RETRACT -- through the same
`safe_park` coroutine the GUI Abort button fires, marshalled onto the loop by
`on_loop` (never on a REP thread: plan Sec B.2 is explicit that a REP thread cannot
fail-safe on its own, since nothing in the repo CoInitialize's COM off the failsafe
thread). Everything else the table serves is still a pure in-memory read, and the
start verbs are still named-but-refused.

TWO SOCKETS (plan Sec B.4)
--------------------------
    29180  REP, command    every mutating verb, serialised, one at a time
    29181  REP, read-only  ACK/IDN/HELP/getState/getStatus/getTelemetry/...
    29182  PUB (future)    progress mirror, if it ever ships

A single REP socket would serialise everything, so a blocking `withdraw` (up to 62.5 s
before the plan's Sec F.2 bounds land) would stall a second observer's `getState`
behind it -- which destroys the "many observers, one commander" property and breaks
the polling loop the whole design rests on. From milestone 3 the split is load-
bearing rather than anticipatory: the control verbs are `command_only`, so 29181
refuses them with -32601 and remains a plane on which no request can move the tip --
which is what makes it safe for any number of observers to hammer it while one
commander drives 29180.

THE DEADLOCK RULE (plan Sec B.4), restated because it is easy to lose
---------------------------------------------------------------------
A REP thread must never hold a `threading.Lock` that a loop coroutine can also
acquire: the REP thread takes the lock, then blocks on
`run_coroutine_threadsafe(...).result()`, and if the coroutine it waits on takes the
same lock, both wait forever and the tip stays where it is. So:

  * there is NO dispatch lock here at all -- each socket has its own thread, and REP
    is already lockstep per socket, so nothing needs serialising across them;
  * `service/runs.py` (which both threads read and the loop writes) is deliberately
    lock-free for the same reason;
  * a handler that needs the loop uses `on_loop()`, which is a BOUNDED
    `run_coroutine_threadsafe(...).result(timeout)` and is called while holding
    nothing. The control verbs are the first handlers that use it;
    `hub._control_lock` (service/control.py) is taken only over a compare-and-swap
    with no await inside it, so the rule is kept on both sides of the call.

GUARDRAILS (plan Sec B.5): every dispatch is wrapped so an unhandled exception becomes
a JSON-RPC `-32000` and the thread survives to serve the next request; no bridge I/O
is ever initiated on these threads; and the poll uses `zmq.Poller` with a timeout, so
`stop()` unblocks it instead of leaving a thread parked in a blocking `recv`.

OPTIONAL AT RUNTIME: pyzmq is an optional extra (`pip install -e ".[flex]"`). The
import is guarded; with pyzmq absent the service logs one line and runs without the
provider.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import threading

from .flex_dispatch import (
    COMMAND_ONLY,
    INTERNAL_ERROR,
    INVALID_REQUEST,
    PARSE_ERROR,
    READ_ONLY,
    REGISTRY,
    Ctx,
    RpcError,
)
from .requests import flex_door
from .requests import stops as _stop_rows

DEFAULT_PORT = 29180
DEFAULT_BIND = "127.0.0.1"
# How long a poll blocks before re-checking the stop flag. Bounds shutdown latency
# without busy-spinning -- the same shape as the fail-safe watchdog's tick.
_POLL_MS = 200
# Floor and margin for a loop-marshalled call's timeout. The bound itself comes from
# the method's published `bound_s` (see `loop_timeout_for`), so the provider never
# waits materially longer than the client was told to. The floor covers scheduling
# jitter on a busy loop; the ceiling is a backstop for a method with no bound.
LOOP_CALL_FLOOR_S = 1.0
LOOP_CALL_MARGIN_S = 0.5
LOOP_CALL_MAX_S = 10.0


def run_on_loop(hub, coro, timeout: float | None = None, *,
                method: str | None = None):
    """Run `coro` on `hub`'s asyncio loop from another thread and wait, BOUNDED.

    The ONE implementation of the marshal (plan Sec B.4) -- `FlexProvider.on_loop` is
    a thin bound wrapper, and `flex_dispatch` falls back to this when a handler is
    driven with no provider (a unit test, an in-process probe). A second copy of this
    logic is exactly how a "hold no lock across the wait" rule stops being true in
    one of the copies.
    """
    if timeout is None:
        timeout = loop_timeout_for(method)
    loop = getattr(hub, "_loop", None)
    if loop is None or not loop.is_running():
        coro.close()                       # never leave an un-awaited coroutine
        raise RpcError(INTERNAL_ERROR, "the service loop is not running")
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return fut.result(timeout)
    except TimeoutError as exc:
        fut.cancel()
        raise RpcError(INTERNAL_ERROR,
                       f"the service loop did not answer within {timeout:g}s") \
            from exc


def loop_timeout_for(method: str | None) -> float:
    """How long a REP thread may wait on the loop for `method`: its published bound
    plus a margin, never below the floor, never above the backstop.

    THE BACKSTOP IS A HARD CEILING, deliberately. An earlier version lifted it for any
    method whose own bound was higher, so that a 35 s `setMode` would not be abandoned
    mid-call -- but the command socket is SERIALISED, so a verb that holds it for 35 s
    holds `abort` and `safePark` behind it for 35 s, and no bound is worth that. The
    rule runs the other way instead: every command-socket bound stays under
    LOOP_CALL_MAX_S (see the scan-control block in flex_dispatch), and anything that
    genuinely cannot becomes a start+poll verb rather than a longer wait.
    """
    m = REGISTRY.get(method or "")
    if m is None:
        return LOOP_CALL_MAX_S
    return min(LOOP_CALL_MAX_S, max(LOOP_CALL_FLOOR_S, m.bound_s + LOOP_CALL_MARGIN_S))


def flex_enabled() -> bool:
    """`AFM_LITHO_FLEX=0` disables the provider; anything else (default "1") enables
    it. Explicit, because a station may want the ports free."""
    return str(os.environ.get("AFM_LITHO_FLEX", "1")).strip().lower() not in (
        "0", "false", "no", "off")


def flex_bind() -> str:
    return str(os.environ.get("AFM_LITHO_FLEX_BIND", DEFAULT_BIND) or DEFAULT_BIND)


def flex_port() -> int:
    try:
        return int(os.environ.get("AFM_LITHO_FLEX_PORT", str(DEFAULT_PORT)))
    except (TypeError, ValueError):
        return DEFAULT_PORT


# ------------------------------------------------------------- the wire codec --

def _jsonable(value):
    """Recursively replace anything JSON cannot carry with null.

    Python's `json` happily emits bare `NaN` / `Infinity`, which are NOT valid JSON:
    a strict parser on the other end (LabVIEW's unflatten, Go, many JS validators)
    rejects the whole frame, so one NaN in a stats dict would look to the client like
    a corrupt reply rather than a missing number. Instrument data is exactly where
    NaNs come from -- an unscanned row, a failed read -- so this is a matter of when,
    not if. Encoding uses `allow_nan=False` so the invariant is enforced, and this
    pass is what makes that safe.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _dumps(payload: dict) -> str:
    """Encode one reply. Sanitises first, then encodes with `allow_nan=False` so a
    non-finite number that slipped past the sanitiser raises HERE -- where the caller
    turns it into a -32000 that still gets sent -- rather than putting an unparseable
    frame on the wire."""
    return json.dumps(_jsonable(payload), allow_nan=False)


def error_reply(msg_id, code: int, message: str, data=None) -> str:
    err = {"code": int(code), "message": str(message)}
    if data is not None:
        err["data"] = data
    try:
        return _dumps({"jsonrpc": "2.0", "id": msg_id, "error": err})
    except (ValueError, TypeError):
        # An un-encodable `data` must never cost the client its reply.
        return _dumps({"jsonrpc": "2.0", "id": msg_id,
                       "error": {"code": int(code), "message": str(message)}})


def result_reply(msg_id, result) -> str:
    return _dumps({"jsonrpc": "2.0", "id": msg_id, "result": result})


# The FLEX verbs of the stop rows. Derived, so a new stop row cannot be left out.
_STOP_VERBS = frozenset(r.flex.verb for r in _stop_rows.ROWS
                        if r.control == "stop" and r.flex is not None)


def handle_message(hub, raw: str, *, role: str = "command", on_loop=None) -> str:
    """Decode one request, run it, and encode exactly one reply.

    Pure and synchronous, so the tests can drive it with no sockets. Every failure
    mode still produces a reply -- a REQ peer that gets nothing back is stuck
    forever, so "malformed input" must never mean "silence" (plan Sec B.5).

    `role` is which socket this arrived on: the read socket refuses the command-only
    verbs with -32601, which is what keeps 29181 a plane that can never move the tip.
    `on_loop` is how a handler reaches the loop; it defaults to `run_on_loop` against
    `hub._loop`.
    """
    msg_id = None
    try:
        req = json.loads(raw)
    except Exception:                                    # noqa: BLE001
        return error_reply(None, PARSE_ERROR, "parse error: not valid JSON")
    if not isinstance(req, dict):
        return error_reply(None, INVALID_REQUEST,
                           "invalid request: expected a JSON object")
    # The id is echoed VERBATIM -- whatever type arrived. The LevyLab dialect sends
    # strings (their LabVIEW JSON unflatten is strict about it, see lockin.py), but a
    # provider that silently retyped an int id would break correlation for any client
    # that does not.
    msg_id = req.get("id")
    method = req.get("method")
    if not isinstance(method, str) or not method:
        return error_reply(msg_id, INVALID_REQUEST,
                           "invalid request: `method` must be a non-empty string")
    params = req.get("params")
    if params is not None and not isinstance(params, (dict, list)):
        if method not in _STOP_VERBS:
            return error_reply(msg_id, INVALID_REQUEST,
                               "invalid request: `params` must be an object or an array")
        # A stop is never refused for its params (DECISIONS 2026-10-19, "stops never
        # refused"): the verb names the stop, the garbled extra is dropped.
        params = None
    try:
        ctx = Ctx(role=role, on_loop=on_loop)
        return result_reply(msg_id, flex_door.handle(hub, method, params, role=role, ctx=ctx))
    except RpcError as exc:
        return error_reply(msg_id, exc.code, exc.message, exc.data)
    except Exception as exc:                             # noqa: BLE001
        # Plan Sec B.4/B.5: EVERY handler exception becomes -32000 with the
        # exception text, and the thread lives on. This also catches an ENCODER
        # failure (a non-finite number or an unserialisable object in a result), so
        # a reply always goes out even when the intended one could not be built.
        return error_reply(msg_id, INTERNAL_ERROR,
                           f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------- the provider --

class FlexProvider:
    """Two REP sockets on two daemon threads. Start once at app startup; `stop()` is
    idempotent and unblocks the polls rather than killing threads."""

    def __init__(self, hub, *, bind: str | None = None, port: int | None = None,
                 log=None):
        self.hub = hub
        self.bind = bind or flex_bind()
        self.port = int(port if port is not None else flex_port())
        self.read_port = self.port + 1
        self._log = log or (lambda msg: print(msg, flush=True))
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._ctx = None
        self._started = threading.Event()
        self.requests = 0                     # observability, not load-bearing
        self.errors = 0
        self.rebuilds = 0                     # wedged sockets recovered (see _rebuild)
        # Set once a socket has actually bound, so a caller (and a test) can tell
        # "the provider is listening" from "the thread started".
        self.bound: dict[str, str] = {}

    # ------------------------------------------------------------- lifecycle --

    def start(self) -> FlexProvider:
        """Bind both sockets and start their threads. Raises if a bind fails -- the
        caller (app.py) catches and logs, because a provider that cannot bind must
        not take the service down."""
        if self._threads:
            return self
        import zmq

        # WARM THE LAZY IMPORTS ON THIS THREAD, before anything can be served.
        # `get_state` imports routers on first use, which drags in numpy
        # and friends -- close to a second, once. Paying that inside a handler would
        # blow a published bound (`getState` promises 50 ms) or, on a control verb,
        # the loop-marshal timeout derived from it: the very first `heartbeat` would
        # answer "the service loop did not answer within 1s" on a loop that was
        # perfectly healthy and merely importing.
        try:
            from .flex_dispatch import get_state
            get_state(self.hub)
        except Exception:                                # noqa: BLE001,S110
            pass

        self._ctx = zmq.Context()
        sock = None
        try:
            for role, port in (("command", self.port), ("read", self.read_port)):
                sock = self._ctx.socket(zmq.REP)
                sock.setsockopt(zmq.LINGER, 0)
                endpoint = f"tcp://{self.bind}:{port}"
                sock.bind(endpoint)                  # may raise: the port is held
                self.bound[role] = endpoint
                t = threading.Thread(target=self._serve, args=(role, sock),
                                     name=f"afm-flex-{role}", daemon=True)
                sock = None                          # the thread owns it now
                self._threads.append(t)
                t.start()
        except Exception:
            # CLOSE THE SOCKET THAT FAILED TO BIND before unwinding. `Context.term()`
            # blocks until every socket in the context is closed, so a leaked
            # half-built socket here would hang `stop()` -- and therefore the caller
            # -- FOREVER, turning "the port is busy" into a service that never
            # finishes starting.
            if sock is not None:
                try:
                    sock.close(linger=0)
                except Exception:                    # noqa: BLE001, S110
                    pass
            self.stop()
            raise
        self._started.set()
        return self

    def stop(self, *, join: float = 2.0) -> None:
        """Ask both threads to exit and wait briefly. The poll timeout is what makes
        this work: the threads are never parked in a blocking `recv`."""
        self._stop.set()
        for t in self._threads:
            if t.is_alive():
                t.join(join)
        self._threads = []
        ctx = self._ctx
        self._ctx = None
        if ctx is not None:
            try:
                ctx.term()
            except Exception:                            # noqa: BLE001, S110
                pass
        self.bound = {}

    @property
    def alive(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    # ---------------------------------------------------------- thread bodies --

    def _serve(self, role: str, sock) -> None:
        """One socket's whole life. Poll with a timeout (never a blocking recv), then
        reply to exactly one request per poll.

        THE REP STATE MACHINE IS THE WHOLE HAZARD HERE. Once `recv` returns, the
        socket is in must-send state and accepts NOTHING until a reply goes out. Any
        path that takes a message and then skips the send wedges this socket
        PERMANENTLY -- for every client, while the thread still looks alive. On the
        command socket that is the socket milestone 3 puts `abort` and `safePark` on,
        so "the reply raised" must never become "the emergency stop is unreachable".

        Two holes this closes:

        * **Decoding.** `recv_string()` does the ZMQ recv (the socket now owes a
          reply) and THEN decodes -- so a single non-UTF-8 frame raised
          `UnicodeDecodeError` past the point of no return and the socket never
          answered again. Now: `recv()` bytes, decode with `errors="replace"`, and
          let the JSON parse fail into an ordinary -32700.
        * **Sending.** A failed `send` leaves the socket owing a reply it can never
          deliver, and there is no API to reset it. So the socket is REBUILT (closed
          and re-bound) and service resumes.
        """
        import zmq

        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        try:
            while not self._stop.is_set():
                try:
                    ready = dict(poller.poll(_POLL_MS))
                except Exception:                        # noqa: BLE001
                    if self._stop.is_set():
                        break
                    continue                             # a hiccup is not a death
                if sock not in ready:
                    continue
                try:
                    frame = sock.recv()                  # BYTES: cannot raise on content
                except Exception:                        # noqa: BLE001
                    if self._stop.is_set():
                        break
                    continue                             # nothing received, nothing owed
                # PAST THIS POINT A REPLY IS OWED. Everything below is wrapped.
                self.requests += 1
                try:
                    # errors="replace", not a raise: a binary or mis-encoded frame
                    # becomes text with U+FFFD, which is not valid JSON, so it lands
                    # in the ordinary -32700 path with an id of null.
                    raw = frame.decode("utf-8", errors="replace")
                    reply = handle_message(self.hub, raw, role=role,
                                           on_loop=self.on_loop)
                except Exception as exc:                 # noqa: BLE001
                    # handle_message already catches everything; this is the last
                    # line of defence for a failure INSIDE the decoder or encoder.
                    # The thread must not die, and the reply must still go out
                    # (plan Sec B.5).
                    self.errors += 1
                    reply = error_reply(None, INTERNAL_ERROR,
                                        f"request could not be handled: {exc}")
                try:
                    self._send(sock, reply)
                except Exception as exc:                 # noqa: BLE001
                    self.errors += 1
                    self._log(f"[flex] {role}: reply NOT sent ({exc}) -- the socket "
                              f"still owes a reply it can never deliver; rebuilding.")
                    sock = self._rebuild(role, sock, poller)
                    if sock is None:
                        return                           # cannot re-bind: exit loudly
        finally:
            try:
                if sock is not None:
                    sock.close(linger=0)
            except Exception:                            # noqa: BLE001, S110
                pass

    def _send(self, sock, reply: str) -> None:
        """The ONE place a reply leaves this process. A method rather than an inline
        call so the send-failure recovery path has a seam a test can drive -- the
        alternative is patching pyzmq's Socket class, which the test's own client
        shares."""
        sock.send_string(reply)

    def _rebuild(self, role: str, sock, poller):
        """Replace a wedged REP socket with a fresh one on the same endpoint.

        A REP socket that failed to send owes a reply forever and there is no API to
        reset it, so closing and re-binding is the only recovery -- strictly better
        than leaving a socket that is alive, pollable and permanently mute. Returns
        the new socket, or None when the re-bind failed (the caller then exits the
        thread rather than spin)."""
        import zmq

        endpoint = self.bound.get(role)
        try:
            poller.unregister(sock)
        except Exception:                                # noqa: BLE001, S110
            pass
        try:
            sock.close(linger=0)
        except Exception:                                # noqa: BLE001, S110
            pass
        if self._stop.is_set() or self._ctx is None or endpoint is None:
            return None
        try:
            fresh = self._ctx.socket(zmq.REP)
            fresh.setsockopt(zmq.LINGER, 0)
            fresh.bind(endpoint)
        except Exception as exc:                         # noqa: BLE001
            self._log(f"[flex] {role}: could not re-bind {endpoint} ({exc}); this "
                      f"socket is now DOWN. Restart the service to restore it.")
            return None
        poller.register(fresh, zmq.POLLIN)
        self.rebuilds += 1
        self._log(f"[flex] {role}: socket rebuilt on {endpoint}.")
        return fresh

    # ------------------------------------------------------------- loop calls --

    def on_loop(self, coro, timeout: float | None = None, *, method: str | None = None):
        """Run a coroutine on the service's asyncio loop from a REP thread and wait,
        BOUNDED (plan Sec B.4).

        The bound comes from the METHOD'S OWN published `bound_s` (plus a small
        margin), not from a fixed constant. `getCapabilities.method_bounds` is what
        the FLEX driver derives its per-call timeout from, so a provider that waited
        ten seconds on a method it published as 50 ms would be answering after the
        client had already given up and destroyed its REQ socket -- the reply would
        be discarded and the socket wedged. Tying both ends to the same number keeps
        that from ever happening.

        The control verbs are the first handlers to use it. It is the ONE sanctioned
        way a REP thread reaches the loop, and the rule that goes with it (hold no
        lock across this call) is easier to keep when there is exactly one place to
        look. The body lives in `run_on_loop` so a handler driven without a provider
        marshals through the identical code.
        """
        return run_on_loop(self.hub, coro, timeout, method=method)


# ------------------------------------------------------------ app integration --

_PROVIDER: FlexProvider | None = None


def current_provider() -> FlexProvider | None:
    return _PROVIDER


def start_provider(hub, *, log=None) -> FlexProvider | None:
    """Start the provider for `hub`, or return None with one logged line.

    Called from app startup AFTER the hub exists (twin or hardware). Every failure --
    pyzmq missing, a port already held, the feature switched off -- is a LOG LINE and
    a None, never an exception: the provider is an observability surface, and it must
    not be able to stop the service that drives the tip from starting.
    """
    global _PROVIDER
    say = log or (lambda msg: print(msg, flush=True))
    if _PROVIDER is not None:
        return _PROVIDER
    if not flex_enabled():
        say("[flex] provider disabled (AFM_LITHO_FLEX=0).")
        return None
    try:
        import zmq  # noqa: F401
    except Exception:                                    # noqa: BLE001
        say("[flex] pyzmq is not installed -- the FLEX provider is OFF and the "
            "service runs without it. Install it with: "
            'pip install -e ".[flex]"')
        return None
    provider = FlexProvider(hub, log=say)
    try:
        provider.start()
    except Exception as exc:                             # noqa: BLE001
        say(f"[flex] provider FAILED to bind ({type(exc).__name__}: {exc}). The "
            f"service runs without it; likely another process holds "
            f"{provider.bind}:{provider.port}/{provider.read_port}.")
        try:
            provider.stop()
        except Exception:                                # noqa: BLE001, S110
            pass
        return None
    _PROVIDER = provider
    say(f"[flex] provider UP: command {provider.bound.get('command')} "
        f"({len(COMMAND_ONLY)} control verbs, incl. token-free abort/safePark), "
        f"read {provider.bound.get('read')} ({len(READ_ONLY)} read methods). "
        f"Probe it with: python scripts/flex_probe.py --port {provider.port}")
    return provider


def stop_provider() -> None:
    global _PROVIDER
    provider = _PROVIDER
    _PROVIDER = None
    if provider is not None:
        provider.stop()


# The wall-clock a probe should expect for the slowest read, published in the runbook
# so a rig check has a number to compare against.
SLOWEST_BOUND_S = max((m.bound_s for m in REGISTRY.values()), default=0.2)
