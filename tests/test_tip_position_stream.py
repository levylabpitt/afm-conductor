"""The live tip-position stream (now the always-on TELEM poll), off-instrument.

Covers the parts that run on the dev machine: the `SHOWTIP` command framing, the
`TELEM` line parser (tip XY + Output.Z + Deflection + Sum, per-channel NaN-safe),
the router routing a TELEM frame to the telemetry hook (without disturbing the
object's CAPTURE/DONE), the host volts->um conversion via a fake `hw_cal`, and the
`POST /api/tip/show` endpoint on both the hardware branch (verb sent) and the twin
no-op branch (no bridge). The Igor task itself is hardware-verified by the user (it
must stream ~10 Hz during a pass without stalling it, and the marker must track the
real tip).
"""

from __future__ import annotations

import types

import pytest

from afm_litho.hardware import bridge_protocol as P
from afm_litho.hardware.lithography_run import LithographyRun
from afm_litho.service.tip_stream import tip_pos_msg, tippos_to_um


# --------------------------------------------------------- command framing ---

def test_cmd_showtip_framing():
    assert P.cmd_showtip(True) == "SHOWTIP ON"
    assert P.cmd_showtip(False) == "SHOWTIP OFF"


# ------------------------------------------------------------- line parser ---

def test_parse_telem_valid():
    assert P.parse_telem("TELEM 0.5 -0.25 66 0.2 5.0 3.0") == (
        0.5, -0.25, 66.0, 0.2, 5.0, 3.0)
    # scientific / high-precision volts round-trip through float()
    xv, yv, zv, dv, sv, av = P.parse_telem("TELEM 1.234e-01 6.7e-02 6.6e1 2e-1 5 3e0")
    assert abs(xv - 0.1234) < 1e-9 and abs(yv - 0.067) < 1e-9
    assert abs(zv - 66.0) < 1e-9 and abs(dv - 0.2) < 1e-9 and sv == 5.0 and av == 3.0


def test_parse_telem_per_channel_nan_safe():
    import math
    # a channel the poll could not read arrives as "nan"; the frame still parses,
    # with a real NaN in that slot (per-channel NaN-safe).
    x, y, z, d, s, a = P.parse_telem("TELEM 0.5 0.25 nan 0.2 nan 3.0")
    assert (x, y, d, a) == (0.5, 0.25, 0.2, 3.0)
    assert math.isnan(z) and math.isnan(s)


def test_parse_telem_rejects_non_telem_and_malformed():
    assert P.parse_telem("CAPTURE obj XSensor 1;2;3") is None
    assert P.parse_telem("DONE obj") is None
    assert P.parse_telem("TELEM 0.5 0.5 0.5 0.5") is None      # too few fields
    assert P.parse_telem("TELEM a b c d e f") is None          # non-numeric (6 vals)
    assert P.parse_telem("TELEM 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8") is None  # too many fields


# ------------------------------------- router routes TELEM to on_telem ----
# TELEM is demuxed by the single-reader ROUTER (bridge_router.py), not the per-pass
# read loop: the interleaved always-on telemetry stream goes to `on_telem` while the
# object's UID-tagged CAPTURE/DONE go to the pass waiter. LithographyRun.on_telem
# forwards to that router, so `run.on_telem = cb` keeps working end to end.

class _QueueLink:
    """A controllable BridgeLink stand-in for the router: read_line blocks on a queue
    (socket.timeout when idle, so the router loop stays alive), send/close no-op."""

    def __init__(self):
        import queue
        self._q = queue.Queue()
        self._closed = False

    def feed(self, line):
        self._q.put(line)

    def send(self, line):
        pass

    def read_line(self, timeout=None):
        import queue
        import socket
        if self._closed:
            return None
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            raise socket.timeout

    def close(self):
        self._closed = True
        self._q.put(None)


def test_run_on_telem_forwards_to_the_router():
    """LithographyRun exposes on_telem as a forward to its router: setting it makes
    the router route interleaved TELEM frames to the hook while a pass's CAPTURE +
    DONE go to the waiter. Driven off a controllable link, no socket."""
    from afm_litho.hardware.bridge_router import BridgeRouter

    link = _QueueLink()
    router = BridgeRouter(link).start()
    run = LithographyRun(None)                             # no link -> no own router
    run.router = router                                    # share this router
    got = []
    run.on_telem = lambda x, y, z, d, s, a: got.append((x, y, z, d, s, a))  # -> router
    w = router.open_waiter()
    try:
        link.feed(P.msg_telem(0.10, 0.20, 66.0, 0.2, 5.0, 3.0))
        link.feed(P.msg_capture(w.uid, "obj", "XSensor", [1, 2, 3]))
        link.feed(P.msg_telem(0.11, 0.21, 67.0, 0.3, 4.0, 2.0))
        link.feed(P.msg_capture(w.uid, "obj", "YSensor", [4, 5, 6]))
        link.feed(P.msg_done(w.uid, "obj"))
        verb, name, capture = run._await_completion(w, "obj")
    finally:
        router.stop()

    assert verb == P.DONE and name == "obj"
    assert got == [(pytest.approx(0.10), pytest.approx(0.20), pytest.approx(66.0),
                    pytest.approx(0.2), pytest.approx(5.0), pytest.approx(3.0)),
                   (pytest.approx(0.11), pytest.approx(0.21), pytest.approx(67.0),
                    pytest.approx(0.3), pytest.approx(4.0), pytest.approx(2.0))]  # both routed
    assert capture["XSensor"] == [1.0, 2.0, 3.0]                 # capture intact
    assert capture["YSensor"] == [4.0, 5.0, 6.0]


def test_router_skips_telem_when_no_hook():
    # with no hook set, a TELEM frame is quietly dropped -- never misread as capture.
    from afm_litho.hardware.bridge_router import BridgeRouter

    link = _QueueLink()
    router = BridgeRouter(link).start()                    # on_telem is None
    run = LithographyRun(None)
    run.router = router
    w = router.open_waiter()
    try:
        link.feed(P.msg_telem(0.9, 0.9, 0.0, 0.0, 5.0, 3.0))
        link.feed(P.msg_capture(w.uid, "obj", "XSensor", [7, 8]))
        link.feed(P.msg_done(w.uid, "obj"))
        verb, name, capture = run._await_completion(w, "obj")
    finally:
        router.stop()
    assert verb == P.DONE and capture["XSensor"] == [7.0, 8.0]


# ------------------------------------------- host volts -> um conversion -----

_CAL = {"x_lvdt_sens": 7.46e-6, "x_lvdt_offset": -0.63,
        "y_lvdt_sens": 7.04e-6, "y_lvdt_offset": 0.07}


def test_tippos_to_um_matches_lvdt_convention():
    # inverse of the pass forward map volts = um*1e-6/sens + offset, so
    # um = (volts - offset)*sens*1e6. Check a round-trip from a known micron point.
    x_um_in, y_um_in = 3.0, -2.0
    xv = x_um_in * 1e-6 / _CAL["x_lvdt_sens"] + _CAL["x_lvdt_offset"]
    yv = y_um_in * 1e-6 / _CAL["y_lvdt_sens"] + _CAL["y_lvdt_offset"]
    x_um, y_um = tippos_to_um(xv, yv, _CAL)
    assert abs(x_um - x_um_in) < 1e-6
    assert abs(y_um - y_um_in) < 1e-6


def test_tippos_to_um_rig_pairs_raised_vs_engaged():
    """The vendor's own formula with the two Cypher rig pairs (2026-09-25): the
    RAISED-tip sensitivities reproduce the stale readings we showed (-15.75 / 15.53),
    the ENGAGED ones put the vendor's Y = 15 um back at 15.00. The conversion reads
    whatever cal it is handed, so a refreshed hub.hw_cal fixes telemetry."""
    raised = {"x_lvdt_sens": 5.07983e-6, "x_lvdt_offset": -0.02,
              "y_lvdt_sens": 4.75782e-6, "y_lvdt_offset": -0.38}
    engaged = dict(raised, x_lvdt_sens=4.9163e-6, y_lvdt_sens=4.59453e-6)
    xr, yr = tippos_to_um(-3.11956, 2.88476, raised)
    assert abs(xr + 15.745) < 0.01 and abs(yr - 15.53) < 0.01
    _, ye = tippos_to_um(-3.11956, 2.88476, engaged)
    assert abs(ye - 15.00) < 0.01


def test_tip_pos_msg_shape():
    xv = _CAL["x_lvdt_offset"]      # at the offset -> position 0
    yv = _CAL["y_lvdt_offset"]
    msg = tip_pos_msg(xv, yv, _CAL)
    assert msg["type"] == "tip_pos"
    assert abs(msg["x_um"]) < 1e-6 and abs(msg["y_um"]) < 1e-6


def test_tippos_to_um_missing_cal_is_safe():
    # off-instrument / empty cal: identity-ish defaults, never a throw/div-by-zero.
    x_um, y_um = tippos_to_um(0.5, 0.5, {})
    assert x_um == 0.5 * 1e6 and y_um == 0.5 * 1e6


# ---------------------------------------------- POST /api/tip/show endpoint --

def test_tip_show_twin_noop_when_no_hardware(client):
    from afm_litho.service.app import hub
    # 13.1 item 8: the twin now BINDS its SoftwareARC to hub.hardware_arc so the
    # park/interlock primitives run on the twin; the real-hardware BRANCH keys on
    # hub.hw_bridge() (None on the twin), so assert that instead of `is None`.
    assert hub.hw_bridge() is None                         # dev-machine default
    r = client.post("/api/tip/show", json={"on": True})
    body = r.json()
    assert body["ok"] is True and body["on"] is True
    assert body["backend"] == "twin"


def test_tip_show_sends_verb_on_hardware(client):
    from afm_litho.service.app import hub
    sent = []
    # a minimal fake hardware_arc: the endpoint only touches hw.link.send.
    hub.host.hardware_arc = types.SimpleNamespace(
        link=types.SimpleNamespace(send=sent.append))
    try:
        r = client.post("/api/tip/show", json={"on": True})
        assert r.json() == {"ok": True, "on": True, "backend": "hardware"}
        r2 = client.post("/api/tip/show", json={"on": False})
        assert r2.json() == {"ok": True, "on": False, "backend": "hardware"}
        assert sent == [P.cmd_showtip(True), P.cmd_showtip(False)]
    finally:
        hub.host.hardware_arc = None                             # restore simulation


def test_tip_show_send_failure_is_error_status(client):
    # E1: a failed bridge send is a hardware-action FAILURE. It must surface as an
    # HTTP error status (5xx), never HTTP 200 with {"ok": False}, so a client that
    # checks only the status code can't read a failed toggle as success.
    from afm_litho.service.app import hub

    def _boom(_msg):
        raise RuntimeError("link down")

    hub.host.hardware_arc = types.SimpleNamespace(
        link=types.SimpleNamespace(send=_boom))
    try:
        r = client.post("/api/tip/show", json={"on": True})
        assert r.status_code == 502            # not 200
        assert not r.is_success
        assert "link down" in r.json()["detail"]
    finally:
        hub.host.hardware_arc = None                             # restore simulation
