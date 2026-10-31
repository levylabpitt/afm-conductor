"""HardwareARC must satisfy the ARCBackend seam and drive the thin Igor bridge
exactly as SoftwareARC drives the twin: arm output/input waves on an event, fire
it, and collect the streamed capture. All exercised against FakeIgorBridge, the
same test double the LithographyRun tests use, so no hardware is required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from afm_litho.hardware.arc_backend import ARCBackend
from afm_litho.hardware.hardware_arc import HardwareARC
from afm_litho.hardware.igor_stub import FakeIgorBridge


def test_td_read_value_round_trips_via_router():
    """td_ReadValue reads one value back through the single-reader router: the VALUE
    reply is correlated to this read's waiter by UID, so a concurrent scan's frames
    can never be misread as the reply (proven under load in test_bridge_router.py).
    Here just confirm the plain round-trip returns the stored value."""
    stub = FakeIgorBridge({}, done_delay=0.0)
    stub.values["DeflectionValue"] = 1.5
    port = stub.serve_in_thread()
    hw = HardwareARC.connect("127.0.0.1", port)
    try:
        assert hw.td_ReadValue("DeflectionValue") == pytest.approx(1.5)
    finally:
        hw.close()
        stub.stop()


def test_td_read_value_raises_on_closed_link():
    """A genuinely missing reply (link closed) raises rather than hanging: closing
    the router wakes the read's waiter with a BridgeClosed."""
    stub = FakeIgorBridge({}, done_delay=0.0)
    port = stub.serve_in_thread()
    hw = HardwareARC.connect("127.0.0.1", port)
    hw.close()                                       # stops the router -> link closed
    stub.stop()
    # either the send fails on the closed socket (OSError) or the waiter wakes with
    # a BridgeClosed (RuntimeError) -- either way it raises promptly, never hangs.
    with pytest.raises((RuntimeError, OSError)):
        hw.td_ReadValue("DeflectionValue")

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "SA40656B0002.ibw"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample .ibw not present")


def test_hardware_arc_satisfies_the_backend_seam():
    # No connection needed for the Protocol check; a None link is fine here.
    hw = HardwareARC(None)              # type: ignore[arg-type]
    assert isinstance(hw, ARCBackend)   # runtime_checkable Protocol


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample .ibw not present")
def test_start_parses_startup():
    stub = FakeIgorBridge.from_ibw(SAMPLE, done_delay=0.0)
    port = stub.serve_in_thread()
    hw = HardwareARC.connect("127.0.0.1", port)
    try:
        startup = hw.start()
        assert startup["pixels"] == pytest.approx(256)
        assert startup is hw.startup
    finally:
        hw.close()
        stub.stop()


def test_full_general_pass_drives_stub_and_returns_capture():
    stub = FakeIgorBridge({}, done_delay=0.0)
    port = stub.serve_in_thread()
    hw = HardwareARC.connect("127.0.0.1", port)
    try:
        hw.td_xSetOutWavePair(0, "0", "PIDSLoop.0.Setpoint",
                              np.array([0.0, 1.0, 2.0]),
                              "PIDSLoop.1.Setpoint", np.array([0.0, 0.0, 0.0]),
                              -50)
        hw.td_xSetInWavePair(0, "0", "Output.Z", "Deflection", 96, 50)
        cap = hw.fire("0", 0)

        assert set(cap) == {"Output.Z", "Deflection"}
        for chan in ("Output.Z", "Deflection"):
            assert len(cap[chan]) == 96
            assert cap[chan].dtype == np.float64

        # the stub recorded the OUTPAIR (two channels) and INPAIR (two channels)
        out_chans = [b[2] for b in stub.out_binds]
        assert out_chans == ["PIDSLoop.0.Setpoint", "PIDSLoop.1.Setpoint"]
        in_chans = [c[2] for c in stub.in_channels]
        assert in_chans == ["Output.Z", "Deflection"]
        assert len(stub.fires) == 1
        assert stub.fires[0][1] == "0"       # event normalized to bare bank
    finally:
        hw.close()
        stub.stop()


def test_read_write_value_round_trip():
    stub = FakeIgorBridge({}, done_delay=0.0)
    stub.values["XSensor"] = 1.25
    port = stub.serve_in_thread()
    hw = HardwareARC.connect("127.0.0.1", port)
    try:
        assert hw.td_WriteValue("PIDSLoop.2.Setpoint", 0.4) == 0
        # the read round-trips, so it's a barrier: the fire-and-forget SETVALUE
        # is guaranteed processed by the stub before we assert on stub.sets.
        assert hw.td_ReadValue("XSensor") == pytest.approx(1.25)
        assert ("PIDSLoop.2.Setpoint", pytest.approx(0.4)) in stub.sets
    finally:
        hw.close()
        stub.stop()


def test_capture_values_come_back():
    stub = FakeIgorBridge({}, done_delay=0.0,
                          capture={"Output.Z": [float(i) for i in range(1, 33)]})
    port = stub.serve_in_thread()
    hw = HardwareARC.connect("127.0.0.1", port)
    try:
        hw.td_xSetInWave(0, "0", "Output.Z", 32, 50)
        cap = hw.fire("0", 0)
        assert list(cap["Output.Z"]) == [float(i) for i in range(1, 33)]
    finally:
        hw.close()
        stub.stop()
