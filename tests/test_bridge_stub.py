"""The fake-Igor bridge stub: seeds from a .ibw note, records driver commands,
reports DONE. This is what lets the Python driver be tested with no hardware.

Protocol v2 (UID-tagged, docs/IGOR_PYTHON_COMMS.md): every request that expects a
reply carries a UID as the first token after the verb, and the stub echoes it in
the reply (STARTUP/VALUE/CAPTURE/DONE/ABORTED), exactly as the Igor .ipf does. The
raw-wire tests here allocate a UID per request and assert the echo, so the stub is
proven to speak the same dialect the router expects.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from afm_litho.hardware import bridge_protocol as P
from afm_litho.hardware.igor_stub import FakeIgorBridge

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "SA40656B0002.ibw"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample .ibw not present")


def _link(stub):
    port = stub.serve_in_thread()
    return P.BridgeLink.connect("127.0.0.1", port)


def _skip_no_sample():
    if not SAMPLE.is_file():
        pytest.skip("sample .ibw not present")


def test_startup_echoes_uid_proto_and_note_calibration():
    _skip_no_sample()
    stub = FakeIgorBridge.from_ibw(str(SAMPLE))
    link = _link(stub)
    try:
        link.send(P.cmd_startup("u1"))
        reply = link.read_line(timeout=5)
        # STARTUP <uid> proto=<v>;<kv;...>
        verb, uid, body = reply.split(" ", 2)
        assert verb == P.STARTUP and uid == "u1"
        kv = dict(p.split("=", 1) for p in body.split(";") if "=" in p)
        assert int(kv["proto"]) == P.PROTOCOL_VERSION         # versioned handshake
        # values straight out of the SA40656B0002 note
        assert float(kv["x_lvdt_sens"]) == pytest.approx(7.4643e-6)
        assert float(kv["x_lvdt_offset"]) == pytest.approx(-0.63)
        assert float(kv["size_m"]) == pytest.approx(3e-5)
        assert kv["setpoint_mode"] == "ac"                    # note is AC Mode
        assert float(kv["setpoint"]) == pytest.approx(0.5)     # AmplitudeSetpointVolts
        assert int(float(kv["mode"])) == 1
    finally:
        link.close()
        stub.stop()


def test_getvalue_echoes_uid_from_the_store():
    _skip_no_sample()
    stub = FakeIgorBridge.from_ibw(str(SAMPLE))
    link = _link(stub)
    try:
        link.send(P.cmd_getvalue("u7", "YLVDTSens"))
        r = link.read_line(timeout=5)
        verb, uid, val = r.split(" ", 2)
        assert verb == P.VALUE and uid == "u7"
        assert float(val) == pytest.approx(7.0428e-6)
    finally:
        link.close()
        stub.stop()


def test_records_choreography_and_reports_done_with_uid():
    _skip_no_sample()
    stub = FakeIgorBridge.from_ibw(str(SAMPLE), done_delay=0.0)
    link = _link(stub)
    try:
        link.send(P.cmd_setvalue("PIDSLoop.2.Setpoint", -10.0))   # fire-and-forget, no uid
        link.send(P.cmd_servoloop(0, "XSensor", 1.5, 0, 0, "Output.X"))
        link.send(P.cmd_wave("XLitho", [0.1, 0.2, 0.3]))
        link.send(P.cmd_armfire("u2", "obj1", -25.0, 96, 50))
        assert link.read_line(timeout=5) == P.msg_done("u2", "obj1")
        # everything the driver sent was recorded (before DONE), for assertions
        assert stub.sets == [("PIDSLoop.2.Setpoint", -10.0)]
        assert stub.loops[0][:2] == (0, "XSensor")
        # SERVOLOOP with no trailing outMin/outMax (the wire line sent above) is the
        # pre-fix, still-valid dialect: the stub defaults to -10/150 (servoloop-limits
        # fix, 2026-09-16), matching AFM_Dispatch's own default.
        assert stub.loop_limits[0] == (-10.0, 150.0)
        assert stub.waves["XLitho"] == pytest.approx([0.1, 0.2, 0.3])
        assert stub.arms[0] == ("obj1", -25.0, 96, 50)
    finally:
        link.close()
        stub.stop()


def test_cmd_servoloop_carries_explicit_output_limits():
    """servoloop-limits fix (2026-09-16): a SERVOLOOP line WITH trailing
    outMin/outMax parses to exactly those limits, not the -10/150 default."""
    stub = FakeIgorBridge({}, done_delay=0.0)
    link = _link(stub)
    try:
        link.send(P.cmd_servoloop(2, "Deflection", -10.0, 0, 1000, "Height", 0.0, 150.0))
        link.send(P.cmd_getvalue("u9", "XSensor"))   # round-trip so the stub has processed it
        link.read_line(timeout=5)
        assert stub.loops[0] == (2, "Deflection", -10.0, 0.0, 1000.0, "Height")
        assert stub.loop_limits[0] == (0.0, 150.0)
    finally:
        link.close()
        stub.stop()


def test_abort_mid_write_cancels_done_with_uid():
    _skip_no_sample()
    stub = FakeIgorBridge.from_ibw(str(SAMPLE), done_delay=2.0)   # a "slow" write
    link = _link(stub)
    try:
        link.send(P.cmd_armfire("u3", "obj2", -25.0, 96, 50))
        link.send(P.cmd_abort())                                  # before DONE fires
        assert link.read_line(timeout=5) == P.msg_aborted("u3", "obj2")
        assert stub.aborts == 1
    finally:
        link.close()
        stub.stop()


def test_capture_frames_precede_done_tagged_by_uid():
    _skip_no_sample()
    zwave = [0.0, 0.5, 1.0]
    stub = FakeIgorBridge.from_ibw(str(SAMPLE), done_delay=0.0,
                                   capture={"Output.Z": zwave, "Deflection": [0.0, 0.0, 0.0]})
    link = _link(stub)
    try:
        link.send(P.cmd_armfire("u4", "obj1", -25.0, 96, 50))
        got: dict = {}
        while True:
            verb, uid, rest = link.read_line(timeout=5).split(" ", 2)
            assert uid == "u4"                       # every frame echoes the pass uid
            if verb == P.DONE:
                assert rest == "obj1"
                break
            assert verb == P.CAPTURE                  # every frame before DONE is a capture
            cname, _, tail = rest.partition(" ")
            chan, _, body = tail.partition(" ")
            assert cname == "obj1"
            got[chan] = [float(v) for v in body.split(";") if v != ""]
        assert got["Output.Z"] == pytest.approx(zwave)
        assert got["Deflection"] == pytest.approx([0.0, 0.0, 0.0])
    finally:
        link.close()
        stub.stop()


# --- general td_-mirroring surface (OUTWAVE/OUTPAIR/INWAVE/INPAIR/FIRE) ------

def _drain_fire(link, uid, name):
    """Read CAPTURE frames until DONE <uid> <name>; return {chan: samples}."""
    got: dict = {}
    while True:
        verb, ruid, rest = link.read_line(timeout=5).split(" ", 2)
        assert ruid == uid
        if verb == P.DONE:
            assert rest == name
            return got
        assert verb == P.CAPTURE
        cname, _, tail = rest.partition(" ")
        chan, _, body = tail.partition(" ")
        assert cname == name
        got[chan] = [float(v) for v in body.split(";") if v != ""]


def test_outwave_inwave_recorded():
    stub = FakeIgorBridge({})
    link = _link(stub)
    try:
        link.send(P.cmd_outwave(0, "0", "PIDSLoop.0.Setpoint", "x", -50.0))
        link.send(P.cmd_inwave(1, "0", "Output.Z", 96, 50))
        link.send(P.cmd_getvalue("ub", "nothing"))      # round-trip barrier
        link.read_line(timeout=5)
        assert stub.out_binds == [(0, "0", "PIDSLoop.0.Setpoint", "x", -50.0)]
        assert stub.in_channels == [(1, "0", "Output.Z", 96, 50)]
    finally:
        link.close()
        stub.stop()


def test_outpair_inpair_recorded_as_two_channels_each():
    stub = FakeIgorBridge({})
    link = _link(stub)
    try:
        link.send(P.cmd_outpair(0, "0", "PIDSLoop.0.Setpoint", "x",
                                "PIDSLoop.1.Setpoint", "y", -50.0))
        link.send(P.cmd_inpair(0, "0", "Output.Z", "Deflection", 96, 50))
        link.send(P.cmd_getvalue("uc", "nothing"))
        link.read_line(timeout=5)
        assert stub.out_binds == [
            (0, "0", "PIDSLoop.0.Setpoint", "x", -50.0),
            (0, "0", "PIDSLoop.1.Setpoint", "y", -50.0),
        ]
        assert stub.in_channels == [
            (0, "0", "Output.Z", 96, 50),
            (0, "0", "Deflection", 96, 50),
        ]
    finally:
        link.close()
        stub.stop()


def test_general_pass_captures_pair_then_done():
    stub = FakeIgorBridge({}, done_delay=0.0)
    link = _link(stub)
    try:
        link.send(P.cmd_wave("x", [0.1, 0.2, 0.3]))
        link.send(P.cmd_wave("y", [0.4, 0.5, 0.6]))
        link.send(P.cmd_outpair(0, "0", "PIDSLoop.0.Setpoint", "x",
                                "PIDSLoop.1.Setpoint", "y", -50.0))
        link.send(P.cmd_inpair(0, "0", "Output.Z", "Deflection", 96, 50))
        link.send(P.cmd_fire("u5", "scan0", "0"))
        got = _drain_fire(link, "u5", "scan0")
        assert set(got) == {"Output.Z", "Deflection"}
        assert len(got["Output.Z"]) == 96
        assert len(got["Deflection"]) == 96
        assert stub.fires == [("scan0", "0")]
        assert stub.waves["x"] == pytest.approx([0.1, 0.2, 0.3])
        # default ramp: 0 .. 1 over 96 points
        assert got["Output.Z"][0] == pytest.approx(0.0)
        assert got["Output.Z"][-1] == pytest.approx(1.0)
    finally:
        link.close()
        stub.stop()


def test_fire_uses_configured_channel_and_ramp_default():
    stub = FakeIgorBridge({}, done_delay=0.0, capture={"Output.Z": [float(i) for i in range(1, 33)]})
    link = _link(stub)
    try:
        link.send(P.cmd_inpair(0, "0", "Output.Z", "Deflection", 32, 1))
        link.send(P.cmd_fire("u6", "s1", "0"))
        got = _drain_fire(link, "u6", "s1")
        assert got["Output.Z"] == pytest.approx(list(range(1, 33)))   # configured
        assert got["Deflection"] == pytest.approx([i / 31 for i in range(32)])  # ramp default
    finally:
        link.close()
        stub.stop()


def test_abort_cancels_slow_fire():
    stub = FakeIgorBridge({}, done_delay=2.0)
    link = _link(stub)
    try:
        link.send(P.cmd_inwave(0, "0", "Output.Z", 96, 50))
        link.send(P.cmd_fire("u8", "s2", "0"))
        link.send(P.cmd_abort())
        assert link.read_line(timeout=5) == P.msg_aborted("u8", "s2")
        assert stub.aborts == 1
    finally:
        link.close()
        stub.stop()
