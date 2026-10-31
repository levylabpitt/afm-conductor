"""F24: binding the real instrument discards the state the twin produced.

The twin sets AC mode, a scan angle and a free-air baseline; bind_hardware must
drop them, let the next hardware approach read the mode live, and tell the panel.
"""

from __future__ import annotations

import asyncio

from _im_fixtures_guard import seed_free_air

from afm_litho.hardware.hardware_arc import HardwareARC
from afm_litho.hardware.igor_stub import FakeIgorBridge
from afm_litho.service.passes import ScanConfig


class _Ws:
    def __init__(self):
        self.sent = []

    async def send_text(self, data):
        self.sent.append(data)


def test_bind_hardware_forgets_twin_state_and_rereads_mode():
    from afm_litho.service.host.scan_cmds import ensure_software_mode
    from afm_litho.service.hub import Hub

    hub = Hub()
    stub = FakeIgorBridge({"height": 0.0, "Output.Z": 0.0, "ZSensor": 0.0,
                           "Deflection": 0.0})
    hw = HardwareARC.connect("127.0.0.1", stub.serve_in_thread())
    reads = []
    hw.check_mode = lambda: (reads.append(1), "ac")[1]
    ws = _Ws()

    async def run():
        hub._loop = asyncio.get_running_loop()
        hub.clients.add(ws)
        hub.host.imaging_mode = "ac"
        hub.host.inst.start_scan(ScanConfig(angle_deg=30.0, pixels=8, lines=8))
        hub.host.inst.scan.running = False
        seed_free_air(hub, mode="ac")
        await asyncio.to_thread(hub.bind_hardware, hw)
        await asyncio.sleep(0.1)
        assert hub.host.imaging_mode is None
        assert hub.host.free_air is None
        assert hub.host.inst.scan is None
        assert not hub.host.engaged
        assert any('"instrument_rebound"' in m for m in ws.sent), ws.sent
        assert reads == []
        assert await ensure_software_mode(hub, "ac") is False
        assert reads == [1]                     # the live READXPT path ran
        assert hub.host.imaging_mode == "ac"
        assert stub.modes == []                 # read back, no SETMODE sent

    try:
        asyncio.run(run())
    finally:
        wd = getattr(hub.host, "_failsafe", None)
        if wd is not None:
            wd.stop()
        hw.close()
        stub.stop()


def test_litho_frame_angle_defaults_to_zero_without_a_scan():
    from afm_litho.service.hub import Hub
    import afm_litho.service.host.litho_worker as litho

    hub = Hub()
    hub.host.inst.start_scan(ScanConfig(angle_deg=30.0, pixels=8, lines=8))
    hub.forget_twin_state()
    saved = litho.hub
    litho.hub = hub
    try:
        assert litho._litho_frame().angle_deg == 0.0
    finally:
        litho.hub = saved


def test_bind_keeps_engaged_so_the_first_hardware_scan_withdraws():
    """A twin approach leaves hub.engaged True; bind must not clear it, because
    host/scan_pump.py passes it to the pre-scan XY transit, which withdraws on it."""
    from afm_litho.service.hub import Hub
    from afm_litho.service.hw_scan import run_hw_scan_chained
    from test_xy_slew_guard import CAL, _connect

    hub = Hub()
    hub.host.engaged = True
    stub, hw = _connect({"XSensor": 2.0, "YSensor": 2.0})
    cfg = ScanConfig(size_um=5.0, pixels=32, lines=2, engage=True, setpoint=0.3,
                     mode="contact")

    async def broadcast(m):
        pass

    try:
        hub.forget_twin_state()
        assert hub.host.engaged is True
        asyncio.run(run_hw_scan_chained(hw, cfg, CAL, broadcast,
                                        engaged=bool(hub.host.engaged)))
    finally:
        hw.close()
        stub.stop()
    assert stub.loops[0][:3] == (2, "Deflection", -10.0)   # the verified withdraw ran


def test_bind_drops_z_loop_and_relative_setpoint_readouts():
    import afm_litho.service.host.continuous as cont
    from afm_litho.service.hub import Hub

    hub = Hub()
    hub.host.z_loop = {"mode": "contact", "setpoint": 0.2}
    hub.host.relative_setpoint["last_free_air_v"] = -0.30
    hub.host.relative_setpoint["last_effective_v"] = -0.10
    hub.forget_twin_state()
    assert hub.host.z_loop == {}
    assert hub.host.relative_setpoint["last_effective_v"] is None
    last = hub.host.relative_setpoint["last_free_air_v"]
    assert last is None

    reads = iter([-0.31, -0.31, -0.31])
    n = []

    async def fake_read():
        n.append(1)
        return next(reads)

    saved = cont._read_free_air_deflection
    cont._read_free_air_deflection = fake_read
    try:
        value, why = asyncio.run(cont._read_free_air_guarded(last))
    finally:
        cont._read_free_air_deflection = saved
    assert why is None and value == -0.31
    assert len(n) == 2                              # first-frame rule: two reads agree
