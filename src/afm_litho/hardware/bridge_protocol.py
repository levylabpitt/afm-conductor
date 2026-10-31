"""The driver <-> Igor-bridge wire protocol (protocol v2).

Newline-framed latin1 text over TCP; wave payloads are ';'-delimited numbers. Real Igor (the
SOCKIT dispatch in igor/afm_litho_run.ipf) and the test stub (igor_stub.py) speak the same
vocabulary, so the driver cannot tell them apart. Verb/reply table: docs/IGOR_PYTHON_COMMS.md §6.

A request that expects a reply carries a UID as the first token after the verb and every reply
echoes it; bridge_router.py delivers by UID. Setters, SHOWTIP, SETMODE and ABORT are
fire-and-forget and uid-less; TELEM is a uid-less async broadcast routed by type. A pass streams
one CAPTURE per armed input channel before its terminal DONE/SCANDONE.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import hashlib
import socket
import sys
import threading

from . import channels

# A peer echoing a different (or absent) proto in its STARTUP reply is rejected at connect.
PROTOCOL_VERSION = 2

# driver -> bridge verbs
STARTUP = "STARTUP"
GETVALUE = "GETVALUE"
GETGV = "GETGV"              # read one Asylum global (GV); replies VALUE, "nan" on error
GETSTR = "GETSTR"            # read a string parameter (td_ReadString); -> XPT <uid> <name>=<text>;
SETVALUE = "SETVALUE"
SERVOLOOP = "SERVOLOOP"
STOPLOOP = "STOPLOOP"        # stop a running PIDS loop (ir_StopPISLoop)
RAMP = "RAMP"                # rate-limited ramp of one output channel to a value
WAVE = "WAVE"
WAVEAPP = "WAVEAPP"          # append to an existing wave (chunked transfer of a big wave)
ARMFIRE = "ARMFIRE"          # litho convenience: fixed channel set (kept)
SHOWTIP = "SHOWTIP"          # display-only hint; the TELEM poll always runs
SETMODE = "SETMODE"          # switch the instrument imaging mode (contact <-> AC)
READXPT = "READXPT"          # read-only crosspoint read-back (mode discriminator); -> XPT
ABORT = "ABORT"              # Igor's one abort function, AFM_SafePark (stop once, bias 0, retract)
ABORTCONFIRM = "ABORTCONFIRM"  # the host read Height < 1 V: Igor clears its abort flag
ABORTFAULT = "ABORTFAULT"      # read Igor's 5 s backup fault (read only) -> VALUE <uid> 0|1
CLEARFAULT = "CLEARFAULT"      # the operator cleared the fault: Igor's abort flag and fault to 0
# general td_-mirroring surface (drives any channel set; what HardwareARC uses)
OUTWAVE = "OUTWAVE"          # bind a loaded wave to one output channel
OUTPAIR = "OUTPAIR"          # bind two loaded waves to a bank's output pair
INWAVE = "INWAVE"            # arm capture of one input channel
INPAIR = "INPAIR"            # arm capture of an input pair
FIRE = "FIRE"               # fire the event; completion -> CAPTURE(all in) + DONE
SCANCHAIN = "SCANCHAIN"     # chained per-line raster: Igor advances lines via callback
LITHOCHAIN = "LITHOCHAIN"   # oversize object as chained chunks; bias zeroed only at TRUE end

# bridge -> driver verbs
VALUE = "VALUE"
XPT = "XPT"                 # READXPT reply: "entry=value;" STRINGS (parse_xpt)
CAPTURE = "CAPTURE"
DONE = "DONE"
ABORTED = "ABORTED"
ERR = "ERR"
SCANDONE = "SCANDONE"       # terminal for a SCANCHAIN (per-line CAPTUREs precede it)
TELEM = "TELEM"             # async, uid-less telemetry+safety frame (parse_telem -> on_telem)


def _n(v: float) -> str:
    return f"{float(v):.12g}"


# ------------------------------------------------- driver command builders --

def cmd_startup(uid: str, version: int = PROTOCOL_VERSION) -> str:
    """Handshake; the bridge echoes `version` so a dialect mismatch fails loudly."""
    return f"{STARTUP} {uid} {int(version)}"


def cmd_getvalue(uid: str, param: str) -> str:
    return f"{GETVALUE} {uid} {param}"


def cmd_getgv(uid: str, name: str) -> str:
    return f"{GETGV} {uid} {name}"


def cmd_getstr(uid: str, name: str) -> str:
    """Read a string parameter such as PIDSLoop.2.OutputChannel. The reply reuses the XPT frame."""
    return f"{GETSTR} {uid} {name}"


def cmd_setvalue(param: str, value: float) -> str:
    return f"{SETVALUE} {param} {_n(value)}"


def cmd_servoloop(loop: int, in_chan: str, setpoint: float, pgain: float,
                  igain: float, out_chan: str, out_min: float | None = None,
                  out_max: float | None = None) -> str:
    """`out_min`/`out_max` are appended only when BOTH are given; absent, the Igor
    handler defaults the output limits to -10/150."""
    line = (f"{SERVOLOOP} {int(loop)} {in_chan} {_n(setpoint)} "
            f"{_n(pgain)} {_n(igain)} {out_chan}")
    if out_min is not None and out_max is not None:
        line += f" {_n(out_min)} {_n(out_max)}"
    return line


def cmd_stoploop(loop: int) -> str:
    """ir_StopPISLoop. The loop output freezes where it is; a following RAMP parks it."""
    return f"{STOPLOOP} {int(loop)}"


def cmd_ramp(channel: str, dest: float, rate: float, max_seconds: float) -> str:
    """IR_SetRamp: ramp one output to `dest` at <= `rate`/s, within `max_seconds`."""
    return (f"{RAMP} {channel} {_n(dest)} {_n(rate)} {_n(max_seconds)}")


def cmd_scanchain(uid: str, xname: str, yname: str, points_per_line: int,
                  lines: int, interp: int, decimation: int,
                  err_chan: str = channels.DEFLECTION) -> str:
    """Callback-chained raster over the full xname/yname waves, one line per fire.
    Replies CAPTURE <uid> L<n> per channel per line, then SCANDONE <uid> <lines>.

    `err_chan` is the second bank-0 in-wave channel: Deflection (contact) or
    Amplitude (AC); the first is ZSensor with the AFM_ScanAdvance callback. Appended
    last so an older bridge still parses tokens 1-7 positionally."""
    return (f"{SCANCHAIN} {uid} {xname} {yname} {int(points_per_line)} "
            f"{int(lines)} {int(interp)} {int(decimation)} {err_chan}")


def cmd_wave(name: str, values) -> str:
    return f"{WAVE} {name} " + ";".join(_n(v) for v in values)


def cmd_waveapp(name: str, values) -> str:
    """Append to an existing wave: a big wave loads as WAVE + WAVEAPP chunks, never
    one multi-MB line, and still ends up as one Igor wave."""
    return f"{WAVEAPP} {name} " + ";".join(_n(v) for v in values)


def cmd_armfire(uid: str, name: str, interp: float, cap_len: int,
                decimation: int) -> str:
    return (f"{ARMFIRE} {uid} {name} {_n(interp)} {int(cap_len)} "
            f"{int(decimation)}")


def cmd_lithochain(uid: str, name: str, interp: float, decimation: int,
                   bounds_wave_name: str) -> str:
    """Fire one OVERSIZE object as chained chunks (docs/LITHO_CHUNK_CHAIN_PLAN.md §4-6).

    The driver first loads XLitho/YLitho/VLitho in full plus `bounds_wave_name`, the chunk
    boundaries [0, c1, ..., n] (each chunk <= chunker.CHAIN_CHUNK_POINTS). Igor slices
    every chunk off all three waves at the same index and advances on the capture-fill
    callback, zeroing the bias only at the object's TRUE end. Per-chunk CAPTUREs all
    arrive under `name` (the driver accumulates them), then one DONE <uid> <name>."""
    return (f"{LITHOCHAIN} {uid} {name} XLitho YLitho VLitho {bounds_wave_name} "
            f"{_n(interp)} {int(decimation)}")


def cmd_abort() -> str:
    return ABORT


def cmd_abortconfirm() -> str:
    """The retract is confirmed (Height < 1 V), so Igor's next abort calls td_Stop again."""
    return ABORTCONFIRM


def cmd_abortfault(uid: str) -> str:
    """Read Igor's fault flag (set by its 5 s backup); replies VALUE 0 or 1. Read only: only
    CLEARFAULT resets it, so a reply lost to a timeout loses nothing."""
    return f"{ABORTFAULT} {uid}"


def cmd_clearfault() -> str:
    """The operator cleared the fault: Igor's stopped flag and fault flag go back to 0, so
    the next abort calls td_Stop again (2026-10-27)."""
    return CLEARFAULT


def cmd_showtip(on: bool) -> str:
    """Tip-marker DISPLAY hint (AFM_ShowTip). The TELEM telemetry+safety poll always
    runs, so this never stops the safety channels."""
    return f"{SHOWTIP} {'ON' if on else 'OFF'}"


def cmd_setmode(mode: str) -> str:
    """Switch imaging mode (contact <-> AC); fire-and-forget, no UID. The wire token is
    the bare word ("Contact"/"AC"); the .ipf calls MainPopupFunc("ImagingModePopup_0",
    1|2, "Contact"|"AC Mode") (MainPanel.ipf:2562; ScanMaster.ipf:8369).

    SAFETY-CRITICAL: this reconfigures the Z feedback input channel and gain sign and
    does NOT withdraw. The caller (POST /api/mode) MUST refuse it while the tip is
    engaged: a mid-engage channel/gain-sign flip is a tip crash."""
    return f"{SETMODE} {'AC' if str(mode).strip().lower() == 'ac' else 'Contact'}"


def cmd_readxpt(uid: str, group: str = "Crosspoint") -> str:
    """Read the crosspoint group -> `XPT <uid> <entry=value;...>`. Read-only
    (td_ReadGroup), so safe mid-scan; carries the mode discriminator `InFast`."""
    return f"{READXPT} {uid} {group}"


def parse_xpt(rest: str) -> dict[str, str]:
    """Parse an XPT payload (after verb and UID) into {entry: value} strings. Tolerant:
    malformed items are skipped, so an odd reply reads as "unknown" instead of raising."""
    out: dict[str, str] = {}
    for item in (rest or "").split(";"):
        item = item.strip()
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        if k:
            out[k] = v.strip()
    return out


def parse_telem(
        line: str) -> tuple[float, float, float, float, float, float] | None:
    """Return the six floats of `TELEM <x> <y> <zdrive> <defl> <sum> <amp>` (volts), or
    None if the line is not a well-formed TELEM frame. A channel sent as "nan" parses to
    NaN, so one bad channel never drops the frame. Amplitude is meaningful only in AC."""
    parts = line.split()
    if len(parts) != 7 or parts[0] != TELEM:
        return None
    try:
        return (float(parts[1]), float(parts[2]), float(parts[3]),
                float(parts[4]), float(parts[5]), float(parts[6]))
    except ValueError:
        return None


# --- general td_-mirroring commands (drive any channel set) -----------------
# OUTWAVE/OUTPAIR bind loaded waves to outputs (td_xSetOutWave/Pair); INWAVE/INPAIR arm
# capture (td_xSetInWave/Pair). `interp` is signed (negative == linear).

def cmd_outwave(bank: int, event: str, channel: str, wave_name: str,
                interp: float) -> str:
    return f"{OUTWAVE} {int(bank)} {event} {channel} {wave_name} {_n(interp)}"


def cmd_outpair(bank: int, event: str, ch_a: str, name_a: str,
                ch_b: str, name_b: str, interp: float) -> str:
    return (f"{OUTPAIR} {int(bank)} {event} {ch_a} {name_a} "
            f"{ch_b} {name_b} {_n(interp)}")


def cmd_inwave(bank: int, event: str, channel: str, cap_len: int,
               decimation: int) -> str:
    return f"{INWAVE} {int(bank)} {event} {channel} {int(cap_len)} {int(decimation)}"


def cmd_inpair(bank: int, event: str, ch_a: str, ch_b: str, cap_len: int,
               decimation: int) -> str:
    return (f"{INPAIR} {int(bank)} {event} {ch_a} {ch_b} "
            f"{int(cap_len)} {int(decimation)}")


def cmd_fire(uid: str, name: str, event: str) -> str:
    return f"{FIRE} {uid} {name} {event}"


# --------------------------------------------- bridge -> driver builders ----

def msg_capture(uid: str, name: str, channel: str, values) -> str:
    """One captured input channel for object `name`, tagged with the pass `uid`."""
    return (f"{CAPTURE} {uid} {name} {channel} "
            + ";".join(_n(v) for v in values))


def msg_value(uid: str, value) -> str:
    """Reply to GETVALUE <uid> (bridge -> driver)."""
    return f"{VALUE} {uid} {_n(value)}"


def msg_xpt(uid: str, entries: dict[str, str]) -> str:
    """Reply to READXPT (stub mirror of Igor's AFM_ReadXPT). Empty dict -> "unknown"."""
    body = "".join(f"{k}={v};" for k, v in entries.items())
    return f"{XPT} {uid} {body}"


def msg_startup(uid: str, kv: str, version: int = PROTOCOL_VERSION) -> str:
    """Reply to STARTUP: proto=<version>, then the AFM_MacroStartup key=value payload."""
    return f"{STARTUP} {uid} proto={int(version)};{kv}"


def msg_done(uid: str, name: str) -> str:
    return f"{DONE} {uid} {name}"


def msg_aborted(uid: str, name: str) -> str:
    """Async abort terminal; `uid` is the interrupted pass, or empty if none was in flight."""
    return f"{ABORTED} {uid} {name}"


def msg_scandone(uid: str, lines) -> str:
    return f"{SCANDONE} {uid} {int(lines)}"


def msg_err(uid: str, message: str) -> str:
    return f"{ERR} {uid} {message}"


def msg_telem(xv: float, yv: float, zdrive: float, defl: float,
              sum_v: float, amp: float) -> str:
    """One TELEM frame (stub mirror of Igor's AFM_TipTick); NaN serializes as "nan"."""
    return (f"{TELEM} {_n(xv)} {_n(yv)} {_n(zdrive)} "
            f"{_n(defl)} {_n(sum_v)} {_n(amp)}")


# ------------------------------------------------------------- link (I/O) --

_WIRE_TEXT_MAX = 200


def _wave_text_summary(payload: str) -> dict:
    """Count, first, last and a hash of the ';' text itself. No float parsing, and sha256 on
    bytes releases the GIL, so an 87,380-point line costs well under 1 ms on the wire thread."""
    first, _, _ = payload.partition(";")
    last = payload[payload.rfind(";") + 1:]
    return {"chars": len(payload), "points": payload.count(";") + 1, "first": first,
            "last": last, "sha256": hashlib.sha256(payload.encode("latin1")).hexdigest()[:16]}


# The door request whose handler is running, set by `requests/dispatch.py` around the handler.
# note_wire stamps it on a wire entry, so the journal can say which request caused a send.
# Plain threads a handler starts (run workers, the fail-safe thread) do not inherit it, but
# asyncio tasks and `to_thread` copy it. A task that outlives its request must not carry that
# request's trace onto later commands (a park after someone else's Stop), so it is started
# with `spawn_untraced`.
wire_trace: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "wire_trace", default=None)

# (execution id, step n) of the run step whose commands are being sent, set by the run's step
# notes (service/step_notes.py) in the thread or task that sends them. note_wire stamps it on
# the wire entry as `run` and `step`. Reads on the router's own thread carry none.
wire_step: contextvars.ContextVar[tuple[str, int] | None] = contextvars.ContextVar(
    "wire_step", default=None)


@contextlib.contextmanager
def untraced():
    """Run the block with no request trace, so wire entries made in it (or in a thread or task
    it starts) carry none."""
    token = wire_trace.set(None)
    try:
        yield
    finally:
        wire_trace.reset(token)


def spawn_untraced(coro):
    """`asyncio.create_task` for work that outlives the request that started it."""
    context = contextvars.copy_context()
    context.run(wire_trace.set, None)
    return asyncio.create_task(coro, context=context)


def note_wire(what: str, line: str, **fields) -> None:
    """Journal one wire line (src "wire", with the sending thread's name).

    Called only after the send or read it records, and returns at once when no journal runs.
    A ';' wave payload is summarised from its text (see `_wave_text_summary`). It never raises
    and does no I/O or waiting; its cost is a dict build, a hash and a queue put (under 3 ms
    for an 87,380-point line, tests/test_wire_tap_journal.py).
    """
    try:
        # Never import here: an import lock held elsewhere must not delay an abort. No module
        # means no journal was started, so there is nothing to note.
        journal = sys.modules.get("afm_litho.service.journal")
        if journal is None or getattr(journal, "_running", None) is None:
            return
        body = {"text": line, **fields}
        head, _, payload = line.rpartition(" ")
        if len(line) > _WIRE_TEXT_MAX and ";" in payload:
            body = {"text": head, "wave": _wave_text_summary(payload), **fields}
        trace = wire_trace.get()
        if trace is not None:
            body["trace"] = trace
        step = wire_step.get()
        if step is not None:
            body["run"], body["step"] = step
        journal.running().note("wire", what, thread=threading.current_thread().name, **body)
    except Exception:  # noqa: BLE001,S110 - the journal must not stop the wire
        pass


class BridgeLink:
    """Line-framed I/O over one socket, used by both the driver and the bridge/stub."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self._buf = b""
        # Serialize sends: any thread may send (abort, workers); the router owns recv.
        self._send_lock = threading.Lock()

    @classmethod
    def connect(cls, host: str, port: int, timeout: float = 5.0) -> "BridgeLink":
        s = socket.create_connection((host, port), timeout=timeout)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return cls(s)

    @staticmethod
    def listen(host: str, port: int) -> tuple[socket.socket, int]:
        """Bind a listener; return (server_socket, bound_port). Port 0 lets the OS pick."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        return srv, srv.getsockname()[1]

    @classmethod
    def accept_on(cls, srv: socket.socket, timeout: float | None = None) -> "BridgeLink":
        """Accept one inbound connection (Igor dials out via SOCKIT); None blocks."""
        srv.settimeout(timeout)
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return cls(conn)

    def send(self, line: str) -> None:
        data = (line + "\n").encode("latin1")
        with self._send_lock:
            self.sock.sendall(data)
        note_wire("send", line)                      # after the send, never before

    def read_line(self, timeout: float | None = None) -> str | None:
        """Return one line, or None if the peer CLOSED. A timeout raises socket.timeout,
        so "nothing yet" is distinct from "closed"."""
        self.sock.settimeout(timeout)
        while b"\n" not in self._buf:
            data = self.sock.recv(1 << 16)          # socket.timeout propagates
            if not data:
                return None                          # peer closed
            self._buf += data
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode("latin1")

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
