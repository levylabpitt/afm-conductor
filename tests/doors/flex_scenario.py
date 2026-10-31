"""The FLEX transcript scenario: every verb on both sockets, and every refusal, in order.

It drives a real twin service (own process, own runtime folder, own ports) over real ZMQ
sockets, so the transcript is the door as a script sees it. The service is fresh, so every
reply starts from the same state. Each step is recorded as {socket, request, reply}; the
`group` it belongs to names the fixture file it lands in. Polling that waits for a state is not
recorded.
"""

from __future__ import annotations

import json
import time

import zmq
from masks import Masker

READ_VERBS = (
    "ACK", "IDN", "HELP", "getState", "getStatus", "getCapabilities", "getInstrument",
    "getLastPark", "getEstimate", "getTelemetry", "getPattern", "getWritten",
    "getScanResult", "getWriteResult", "getMode",
)
# Both sockets serve these; the rest of the registry is command-only.
COMMAND_VERBS = (
    "acquireControl", "releaseControl", "heartbeat", "abort", "safePark", "setMode",
    "startApproach", "withdraw", "setZGain", "startScan", "setContinuous", "loadPattern",
    "startWrite", "abortWrite", "setObjects",
)

SMALL_SCAN = {"size_um": 2.0, "pixels": 16, "lines": 8, "line_rate_hz": 40.0,
              "x_offset_um": 0.0, "y_offset_um": 0.0}
RECT = {"kind": "region", "shape": "rect", "x_um": 1.0, "y_um": 1.0, "w_um": 0.4,
        "h_um": 0.2, "pitch_um": 0.1, "voltage": 2.0}


class Recorder:
    """Two REQ sockets, a masker, and the steps recorded so far, by group."""

    def __init__(self, svc):
        self.svc = svc
        self.ctx = zmq.Context()
        self.socks = {"command": self._sock(svc.flex_port),
                      "read": self._sock(svc.flex_port + 1)}
        self.mask = Masker(str(svc.runtime), (svc.flex_port, svc.flex_port + 1, svc.port))
        self.groups: dict[str, list[dict]] = {}
        self._n = 0
        self.token: str | None = None

    def _sock(self, port):
        s = self.ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVTIMEO, 60000)
        s.setsockopt(zmq.SNDTIMEO, 60000)
        s.connect(f"tcp://127.0.0.1:{port}")
        return s

    def close(self):
        for s in self.socks.values():
            s.close(linger=0)
        self.ctx.term()

    def send(self, group, socket, method, params=None, *, raw=None, record=True):
        """One request. `raw` sends that exact text instead of a built request."""
        # Polling is not recorded and varies in length, so it must not use up request ids.
        if record:
            self._n += 1
        if raw is None:
            req = {"jsonrpc": "2.0", "id": str(self._n) if record else "poll",
                   "method": method}
            if params is not None:
                req["params"] = params
            raw = json.dumps(req)
        self.socks[socket].send_string(raw)
        text = self.socks[socket].recv_string()
        reply = json.loads(text)
        if record:
            self.mask.learn(reply)
            try:
                request = json.loads(raw)
            except ValueError:
                request = {"raw": raw}
            self.groups.setdefault(group, []).append({
                "socket": socket, "request": self.mask.apply(request),
                "reply": self.mask.apply(reply)})
        return reply

    def ok(self, group, socket, method, params=None):
        reply = self.send(group, socket, method, params)
        assert "error" not in reply, (method, reply)
        return reply["result"]

    def wait_state(self, *want, timeout=60.0):
        end = time.monotonic() + timeout
        seen = None
        while time.monotonic() < end:
            seen = self.send("", "read", "getState", record=False)["result"]
            if seen in want:
                return seen
            time.sleep(0.05)
        raise AssertionError(f"state stayed {seen!r}, wanted one of {want}")

    def wait_quiet(self, timeout=60.0):
        """Wait until no pass is running (not approaching, scanning or writing)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.send("", "read", "getState", record=False)["result"] not in (
                    "approaching", "scanning", "writing"):
                return
            time.sleep(0.05)
        raise AssertionError("a pass never finished")

    def tok(self, **extra):
        return {"token": self.token, **extra}


def run(svc) -> dict[str, list[dict]]:
    r = Recorder(svc)
    try:
        _reads(r)
        _malformed_and_unknown(r)
        _wrong_socket(r)
        _control(r)
        _no_token_and_bad_params(r)
        _scan(r)
        _write(r)
        _stops(r)
        _revoked_and_operator(r)
    finally:
        r.close()
    return r.groups


def _reads(r):
    """Every read verb on both sockets, at rest, before anything has run."""
    for sock in ("read", "command"):
        for verb in READ_VERBS:
            r.send("reads_at_rest", sock, verb)
    from afm_litho.service import flex_dispatch as fd
    for verb in sorted(fd.REGISTRY):                 # signature, doc text and bound of each
        r.send("reads_at_rest", "read", "HELP", {"command": verb})
    r.send("reads_at_rest", "read", "HELP", ["setMode"])
    r.send("reads_at_rest", "read", "HELP", {"command": "noSuchVerb"})
    r.send("reads_at_rest", "command", "getScanResult", {"run_id": "unknown-run"})


def _malformed_and_unknown(r):
    g = "refusals_protocol"
    for sock in ("read", "command"):
        r.send(g, sock, "noSuchVerb")
        r.send(g, sock, "executePass")
        r.send(g, sock, "", raw="this is not json")
        r.send(g, sock, "", raw="[1, 2]")
        r.send(g, sock, "", raw=json.dumps({"jsonrpc": "2.0", "id": "m1"}))
        r.send(g, sock, "", raw=json.dumps({"jsonrpc": "2.0", "id": "m2", "method": "ACK",
                                            "params": 5}))
        r.send(g, sock, "", raw=json.dumps({"jsonrpc": "2.0", "id": 7, "method": "ACK"}))


def _wrong_socket(r):
    """The read socket refuses every command-only verb, token or not."""
    g = "refusals_read_socket"
    for verb in COMMAND_VERBS:
        r.send(g, "read", verb, {})
    r.send(g, "read", "setMode", {"token": "not-a-real-token", "mode": "contact"})


def _control(r):
    g = "control"
    first = r.ok(g, "command", "acquireControl", {"client": "doors-script"})
    r.token = first["token"]
    r.send(g, "command", "acquireControl", {"client": "second-script"})       # held
    r.send(g, "command", "acquireControl", {"client": ""})                    # bad params
    r.ok(g, "command", "heartbeat", {"token": r.token})
    r.send(g, "command", "heartbeat", {"token": "stale-token"})
    r.send(g, "command", "heartbeat", {})
    r.send(g, "command", "releaseControl", {"token": "stale-token"})          # never an error
    r.send(g, "command", "releaseControl", {})
    r.send(g, "read", "getStatus")                                            # commander set
    r.ok(g, "command", "releaseControl", {"token": r.token})
    again = r.ok(g, "command", "acquireControl", {"client": "doors-script",
                                                  "deadman_s": 300})
    r.token = again["token"]
    r.send(g, "command", "acquireControl", {"client": "x", "deadman_s": "soon"})


def _no_token_and_bad_params(r):
    g = "refusals_token_and_params"
    token_verbs = (("setMode", {"mode": "contact"}), ("startApproach", {}),
                   ("setZGain", {"pgain": 1.0}), ("startScan", dict(SMALL_SCAN)),
                   ("setContinuous", {"enabled": False}), ("loadPattern", {"source": "auto"}),
                   ("startWrite", {}), ("setObjects", {"updates": {}}))
    for verb, params in token_verbs:
        r.send(g, "command", verb, params)                                  # no token
        r.send(g, "command", verb, {**params, "token": "not-the-token"})    # wrong token
    r.send(g, "command", "setMode", r.tok(mode="sideways"))
    r.send(g, "command", "setMode", r.tok())
    r.send(g, "command", "startApproach", r.tok(setpoint="high"))
    r.send(g, "command", "startScan", r.tok(**{**SMALL_SCAN, "pixels": "many"}))
    r.send(g, "command", "setContinuous", r.tok())
    r.send(g, "command", "loadPattern", r.tok())
    r.send(g, "command", "loadPattern", r.tok(source="auto", objects=[RECT]))
    r.send(g, "command", "loadPattern", r.tok(objects=[{"kind": "blob"}]))
    r.send(g, "command", "startWrite", r.tok(**SP))                            # no pattern loaded
    r.send(g, "command", "setObjects", r.tok(updates={"nope": {"voltage": 1.0}}))
    r.send(g, "command", "setZGain", r.tok(pgain=1.0))                       # twin: -32010
    r.send(g, "command", "setZGain", r.tok(pgain="loud"))
    r.send(g, "command", "abort", {"scope": "everything"})
    r.send(g, "read", "getScanResult", {"run_id": 5})


def _scan(r):
    g = "scan"
    r.ok(g, "command", "setMode", r.tok(mode="contact"))
    r.ok(g, "read", "getMode")
    approach = r.ok(g, "command", "startApproach", r.tok())
    r.wait_state("engaged")
    r.send(g, "read", "getState")
    r.send(g, "command", "setMode", r.tok(mode="ac"))                       # refused: engaged
    r.send(g, "read", "getMode")
    r.send(g, "command", "setZGain", r.tok(pgain=1.0))
    r.send(g, "command", "startApproach", r.tok())                          # while engaged
    r.wait_quiet()
    scan = r.ok(g, "command", "startScan", r.tok(**SMALL_SCAN))
    r.send(g, "command", "startScan", r.tok(**SMALL_SCAN))                  # busy
    r.send(g, "command", "startApproach", r.tok())                          # busy
    r.send(g, "command", "setMode", r.tok(mode="ac"))                       # refused: scanning
    r.send(g, "command", "setContinuous", r.tok(enabled=True))              # busy
    r.wait_quiet()
    _wait_complete(r, scan["run_id"], "getScanResult")
    r.send(g, "read", "getScanResult", {"run_id": scan["run_id"]})
    r.send(g, "read", "getScanResult")
    r.send(g, "read", "getScanResult", {"run_id": approach["run_id"]})
    r.send(g, "read", "getStatus")
    r.send(g, "command", "setContinuous", r.tok(enabled=False))
    r.send(g, "command", "setContinuous", r.tok(enabled=True, max_frames=1))
    r.wait_quiet()
    r.send(g, "command", "setContinuous", r.tok(enabled=False))


def _wait_complete(r, run_id, verb, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        rep = r.send("", "read", verb, {"run_id": run_id}, record=False)
        if rep.get("result", {}).get("completed"):
            return
        time.sleep(0.05)
    raise AssertionError(f"{verb}({run_id}) never completed")


# The setpoint every write here used before it was required: 0.2 V, absolute.
SP = {"setpoint": 0.2, "setpoint_reference": "absolute"}


def _write(r):
    g = "write"
    r.ok(g, "command", "withdraw", r.tok())
    r.wait_state("idle", "parked")
    r.send(g, "command", "loadPattern", r.tok(objects=[RECT]))
    r.send(g, "read", "getPattern")
    r.send(g, "read", "getEstimate")
    r.send(g, "command", "setObjects",
           r.tok(updates={"region-1": {"voltage": 3.0}}))
    r.send(g, "command", "setObjects", r.tok(updates={"region-1": {"colour": "red"}}))
    r.send(g, "command", "startWrite", r.tok(**SP))                         # amp_gain missing
    r.send(g, "command", "startWrite", r.tok(amp_gain=25.0))                # setpoint missing
    r.ok(g, "command", "setMode", r.tok(mode="ac"))
    r.send(g, "command", "startWrite", r.tok(amp_gain=25.0, twin_speed=0, **SP))   # AC-mode write
    r.wait_quiet()
    r.send(g, "read", "getState")
    r.ok(g, "command", "setMode", r.tok(mode="contact"))
    r.send(g, "command", "startWrite", r.tok(amp_gain=25.0, objects=["nope"], **SP))
    r.send(g, "command", "startWrite", r.tok(amp_gain=25.0, twin_speed="fast", **SP))
    done = r.ok(g, "command", "startWrite", r.tok(amp_gain=25.0, twin_speed=0, **SP))
    r.wait_quiet()
    _wait_complete(r, done["run_id"], "getWriteResult")
    r.send(g, "read", "getWriteResult", {"run_id": done["run_id"]})
    r.send(g, "read", "getWritten")
    r.send(g, "read", "getStatus")
    # A slow write holds the claim: busy, and the pattern is locked, until abortWrite.
    slow = r.ok(g, "command", "startWrite", r.tok(amp_gain=25.0, twin_speed=0.001, **SP))
    r.wait_state("writing")
    r.send(g, "command", "startWrite", r.tok(amp_gain=25.0, twin_speed=0.001, **SP))
    r.send(g, "command", "setObjects", r.tok(updates={"region-1": {"voltage": 1.0}}))
    r.send(g, "command", "startApproach", r.tok())
    r.send(g, "command", "setMode", r.tok(mode="ac"))                       # refused: writing
    r.send(g, "read", "getState")
    r.ok(g, "command", "abortWrite", {})
    r.wait_state("idle", "parked", "engaged")
    r.send(g, "read", "getWriteResult", {"run_id": slow["run_id"]})
    r.send(g, "read", "getWritten")


def _stops(r):
    g = "stops"
    r.send(g, "command", "withdraw", {})                                    # token-free
    r.send(g, "command", "abort", {})
    r.send(g, "command", "safePark", {})
    r.send(g, "command", "safePark", {"reason": "doors transcript"})
    r.send(g, "command", "abortWrite", {})
    r.send(g, "read", "getLastPark")
    r.send(g, "read", "getState")


def _revoked_and_operator(r):
    g = "access"
    old = r.token
    status, _ = r.svc.post("/api/control/take", {"client": "doors-operator"})
    assert status == 200
    r.send(g, "command", "setMode", {"token": old, "mode": "contact"})     # revoked token
    r.send(g, "command", "heartbeat", {"token": old})
    r.send(g, "command", "acquireControl", {"client": "doors-script"})      # operator holds
    r.send(g, "command", "withdraw", {"token": old})                        # stop: still served
    r.send(g, "read", "getStatus")
    status, _ = r.svc.post("/api/control/release", {})
    assert status == 200
    new = r.ok(g, "command", "acquireControl", {"client": "doors-script"})
    r.token = new["token"]
    r.ok(g, "command", "releaseControl", {"token": r.token})
    r.send(g, "read", "getStatus")
