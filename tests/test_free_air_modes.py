"""Free-air baseline in every imaging mode (docs/INSTRUMENT_MODEL_SPEC.md 13.1 item 33).

The guard (`command_guard.check_approach`) demands a fresh withdrawn baseline tagged
with the engage mode. These tests pin that a mode switch, an approach and a withdraw
each produce a baseline tagged with the mode in use -- so an AC approach can pass --
and that a baseline tagged with another mode is still refused. AC baselines are the
free amplitude (the AC fraction band divides by it); other modes the deflection.

`client` comes from tests/conftest.py; `hub` is a process global, so every test
restores the fields it touches.
"""

from __future__ import annotations

import contextlib
import time

import pytest

from afm_litho.service.command_guard import FreeAir, GuardRefused, check_approach


@pytest.fixture
def hub_state(client):
    from afm_litho.service.app import hub
    prev = (hub.host.imaging_mode, hub.host.free_air, hub.host.engaged,
            dict(hub.host.scan_repeat), getattr(hub.host, "hardware_arc", None))
    hub.host.engaged = False
    hub.host.scan_repeat["enabled"] = False
    yield hub
    with contextlib.suppress(Exception):
        client.post("/api/withdraw")
    hub.host.imaging_mode, hub.host.free_air, hub.host.engaged = prev[0], prev[1], prev[2]
    hub.host.scan_repeat.clear()
    hub.host.scan_repeat.update(prev[3])
    hub.host.hardware_arc = prev[4]


def _ac_setpoint(hub, free_amp: float) -> float:
    lo, hi = hub.host.instrument_model.ac_setpoint_fraction_band.value
    return free_amp * (lo + hi) / 2.0


def _assert_fresh(fa: FreeAir, mode: str, t0: float) -> None:
    assert fa is not None
    assert fa.mode == mode, fa
    assert fa.withdrawn is True
    assert fa.ts >= t0


# ------------------------------------------------------------------ pure guard --

def test_contact_tagged_baseline_refused_by_ac_approach(hub_state):
    hub = hub_state
    now = time.time()
    fa = FreeAir(value_v=1.0, ts=now, mode="contact", withdrawn=True)
    with pytest.raises(GuardRefused) as ei:
        check_approach(hub.host.instrument_model, "ac", _ac_setpoint(hub, 1.0), fa, now=now)
    assert ei.value.reason == "free_air_stale"
    assert "'contact'" in ei.value.message


def test_ac_tagged_baseline_refused_by_contact_approach(hub_state):
    hub = hub_state
    now = time.time()
    fa = FreeAir(value_v=-0.2, ts=now, mode="ac", withdrawn=True)
    with pytest.raises(GuardRefused) as ei:
        check_approach(hub.host.instrument_model, "contact", 0.0, fa, now=now)
    assert ei.value.reason == "free_air_stale"


def test_ac_band_needs_free_amplitude_not_deflection(hub_state):
    """The AC fraction band divides by the baseline, so an AC-tagged withdrawn
    DEFLECTION (Cypher free air sits near -0.2 V) refuses every positive amplitude
    setpoint; the free amplitude passes. Why the AC baseline reads Amplitude."""
    hub = hub_state
    now = time.time()
    sp = _ac_setpoint(hub, 1.1)
    with pytest.raises(GuardRefused) as ei:
        check_approach(hub.host.instrument_model, "ac", sp,
                       FreeAir(value_v=-0.2, ts=now, mode="ac", withdrawn=True), now=now)
    assert ei.value.reason == "setpoint_out_of_band"
    check_approach(hub.host.instrument_model, "ac", sp,
                   FreeAir(value_v=1.1, ts=now, mode="ac", withdrawn=True), now=now)


# ------------------------------------------------------------------------ twin --

def test_twin_mode_ac_then_approach_accepted_with_fresh_ac_baseline(client, hub_state):
    hub = hub_state
    free_amp = float(hub.host.inst.world.lever.free_amplitude_v)
    t0 = time.time()
    r = client.post("/api/mode", json={"mode": "ac"})
    assert r.json().get("ok") is True, r.json()
    _assert_fresh(hub.host.free_air, "ac", t0)
    assert hub.host.free_air.value_v == pytest.approx(free_amp)

    r = client.post("/api/approach",
                    json={"mode": "ac", "setpoint": _ac_setpoint(hub, free_amp)})
    body = r.json()
    assert body.get("ok") is True, body
    _assert_fresh(hub.host.free_air, "ac", t0)


def test_twin_approach_establishes_ac_baseline_when_none_on_record(client, hub_state):
    hub = hub_state
    free_amp = float(hub.host.inst.world.lever.free_amplitude_v)
    r = client.post("/api/mode", json={"mode": "ac"})
    assert r.json().get("ok") is True, r.json()
    hub.host.free_air = None
    t0 = time.time()
    r = client.post("/api/approach",
                    json={"mode": "ac", "setpoint": _ac_setpoint(hub, free_amp)})
    body = r.json()
    assert body.get("ok") is True, body
    _assert_fresh(hub.host.free_air, "ac", t0)


def test_twin_mode_ac_withdraw_then_approach_accepted(client, hub_state):
    hub = hub_state
    free_amp = float(hub.host.inst.world.lever.free_amplitude_v)
    r = client.post("/api/mode", json={"mode": "ac"})
    assert r.json().get("ok") is True, r.json()
    t0 = time.time()
    client.post("/api/withdraw")
    _assert_fresh(hub.host.free_air, "ac", t0)

    r = client.post("/api/approach",
                    json={"mode": "ac", "setpoint": _ac_setpoint(hub, free_amp)})
    assert r.json().get("ok") is True, r.json()
    assert hub.host.free_air.mode == "ac"


def test_twin_mode_contact_rereads_contact_baseline(client, hub_state):
    hub = hub_state
    client.post("/api/mode", json={"mode": "ac"})
    t0 = time.time()
    r = client.post("/api/mode", json={"mode": "contact"})
    assert r.json().get("ok") is True, r.json()
    _assert_fresh(hub.host.free_air, "contact", t0)
    assert hub.host.free_air.value_v == pytest.approx(
        float(hub.host.inst.read_free_air_deflection()), abs=0.05)


def test_twin_withdraw_with_unknown_mode_tags_contact(client, hub_state):
    hub = hub_state
    hub.host.imaging_mode = None
    t0 = time.time()
    client.post("/api/withdraw")
    _assert_fresh(hub.host.free_air, "contact", t0)


# -------------------------------------------------------------------- hardware --

_FREE_AMP_V = 1.2
_FREE_DEFL_V = -0.2


@pytest.fixture
def hw_hub(hub_state, monkeypatch):
    """The hub over a fake Igor bridge; hw_withdraw's 2.5 s move wait replaced by a
    fast snapshot so the withdraw door stays quick."""
    from afm_litho.hardware.hardware_arc import HardwareARC
    from afm_litho.hardware.igor_stub import FakeIgorBridge
    from afm_litho.instrument import asylum
    from afm_litho.hardware import hw_approach

    stub = FakeIgorBridge({"Amplitude": _FREE_AMP_V, "DeflectionValue": _FREE_DEFL_V,
                           "Deflection": 1.5, "DetectorSum": 5.0, "height": 20.0},
                          done_delay=0.0)
    port = stub.serve_in_thread()
    hub = hub_state
    hub.host.hardware_arc = HardwareARC.connect("127.0.0.1", port)
    prev_zloop = getattr(hub.host, "z_loop", None)
    prev_telem = getattr(hub.host, "_hw_telemetry", None)
    fast_withdraw = lambda arc, **kw: hw_approach.snapshot(arc, engaged=False)  # noqa: E731
    monkeypatch.setattr(hw_approach, "hw_withdraw", fast_withdraw)
    monkeypatch.setattr(asylum, "hw_withdraw", fast_withdraw)    # the door withdraws via the plugin
    yield hub, stub
    with contextlib.suppress(Exception):
        hub.host.hardware_arc.close()
    hub.host.hardware_arc = None
    hub.host.engaged = False
    hub.host.z_loop = prev_zloop if prev_zloop is not None else {}
    hub.host._hw_telemetry = prev_telem
    stub.stop()


def test_hw_mode_ac_then_approach_accepted_with_free_amplitude(client, hw_hub):
    hub, _stub = hw_hub
    t0 = time.time()
    r = client.post("/api/mode", json={"mode": "ac"})
    assert r.json().get("ok") is True, r.json()
    _assert_fresh(hub.host.free_air, "ac", t0)
    assert hub.host.free_air.value_v == pytest.approx(_FREE_AMP_V)

    r = client.post("/api/approach",
                    json={"mode": "ac", "setpoint": _ac_setpoint(hub, _FREE_AMP_V),
                          "settle_s": 0.0})
    body = r.json()
    assert body.get("ok") is True, body
    assert hub.host.free_air.mode == "ac"


def test_hw_mode_ac_withdraw_then_approach_accepted(client, hw_hub):
    hub, _stub = hw_hub
    r = client.post("/api/mode", json={"mode": "ac"})
    assert r.json().get("ok") is True, r.json()
    hub.host.free_air = None
    t0 = time.time()
    snap = client.post("/api/withdraw").json()
    assert "error" not in snap, snap
    _assert_fresh(hub.host.free_air, "ac", t0)
    assert hub.host.free_air.value_v == pytest.approx(_FREE_AMP_V)

    r = client.post("/api/approach",
                    json={"mode": "ac", "setpoint": _ac_setpoint(hub, _FREE_AMP_V),
                          "settle_s": 0.0})
    assert r.json().get("ok") is True, r.json()


def test_hw_mode_contact_rereads_deflection_tagged_contact(client, hw_hub):
    hub, _stub = hw_hub
    client.post("/api/mode", json={"mode": "ac"})
    t0 = time.time()
    r = client.post("/api/mode", json={"mode": "contact"})
    assert r.json().get("ok") is True, r.json()
    _assert_fresh(hub.host.free_air, "contact", t0)
    assert hub.host.free_air.value_v == pytest.approx(_FREE_DEFL_V)


def test_twin_ac_record_carries_deflection_and_amplitude(client, hub_state):
    hub = hub_state
    r = client.post("/api/mode", json={"mode": "ac"})
    assert r.json().get("ok") is True, r.json()
    fa = hub.host.free_air
    assert fa.value_v == pytest.approx(float(hub.host.inst.world.lever.free_amplitude_v))
    assert fa.defl_v == pytest.approx(float(hub.host.inst.read_free_air_deflection()), abs=0.05)
    inst = client.get("/api/instrument").json()
    assert inst["free_air"]["defl_v"] == pytest.approx(fa.defl_v)
    assert inst["free_air"]["value_v"] == pytest.approx(fa.value_v)


def test_twin_ac_approach_then_safe_park_verifies(client, hub_state):
    from afm_litho.service.host.safe_park import safe_park

    hub = hub_state
    free_amp = float(hub.host.inst.world.lever.free_amplitude_v)
    assert client.post("/api/mode", json={"mode": "ac"}).json().get("ok") is True
    r = client.post("/api/approach",
                    json={"mode": "ac", "setpoint": _ac_setpoint(hub, free_amp)})
    assert r.json().get("ok") is True, r.json()
    assert hub.host.free_air.mode == "ac"
    res = safe_park(hub, "test: ac park verify")
    assert res.verified is True, res
    assert res.critical is False, res
    assert getattr(hub, "fault", None) is None


def test_hw_approach_switches_mode_before_the_free_air_read(client, hw_hub):
    hub, _stub = hw_hub
    arc = hub.host.hardware_arc
    events: list[str] = []
    real_set, real_read = arc.set_imaging_mode, arc.td_ReadValue

    def spy_set(mode):
        events.append(f"setmode:{mode}")
        return real_set(mode)

    def spy_read(name, **kw):
        events.append(f"read:{name}")
        return real_read(name, **kw)

    arc.set_imaging_mode = spy_set
    arc.td_ReadValue = spy_read
    hub.host.imaging_mode = "contact"
    hub.host.free_air = None
    r = client.post("/api/approach",
                    json={"mode": "ac", "setpoint": _ac_setpoint(hub, _FREE_AMP_V),
                          "settle_s": 0.0})
    assert r.json().get("ok") is True, r.json()
    assert "setmode:ac" in events, events
    assert "read:Amplitude" in events, events
    assert events.index("setmode:ac") < events.index("read:Amplitude"), events
    assert events.index("setmode:ac") < events.index("read:DeflectionValue"), events
    assert hub.host.imaging_mode == "ac"
    assert hub.host.free_air.mode == "ac"
    assert hub.host.free_air.value_v == pytest.approx(_FREE_AMP_V)
    assert hub.host.free_air.defl_v == pytest.approx(_FREE_DEFL_V)


def _spy_arc(hub, events: list) -> None:
    arc = hub.host.hardware_arc
    real_set, real_read = arc.set_imaging_mode, arc.td_ReadValue

    def spy_set(mode):
        events.append(f"setmode:{mode}")
        return real_set(mode)

    def spy_read(name, **kw):
        events.append(f"read:{name}")
        return real_read(name, **kw)

    arc.set_imaging_mode = spy_set
    arc.td_ReadValue = spy_read


def _telemetry(hub, zdrive_v):
    hub.host._hw_telemetry = {"t": time.time(), "zdrive_v": zdrive_v}


def test_hw_approach_with_unknown_mode_switches_before_the_free_air_read(client, hw_hub):
    """Tracked mode unknown, live crosspoint says contact, Z drive retracted: switch
    to AC before any free-air read."""
    hub, stub = hw_hub
    stub.crosspoint = {"InFast": "FilterOut"}
    _telemetry(hub, 0.0)
    events: list[str] = []
    _spy_arc(hub, events)
    hub.host.imaging_mode = None
    hub.host.free_air = None
    r = client.post("/api/approach",
                    json={"mode": "ac", "setpoint": _ac_setpoint(hub, _FREE_AMP_V),
                          "settle_s": 0.0})
    assert r.json().get("ok") is True, r.json()
    assert "setmode:ac" in events, events
    assert events.index("setmode:ac") < events.index("read:Amplitude"), events
    assert events.index("setmode:ac") < events.index("read:DeflectionValue"), events
    assert hub.host.imaging_mode == "ac"
    assert hub.host.free_air.value_v == pytest.approx(_FREE_AMP_V)


def test_hw_approach_switch_settles_and_clears_the_prior_baseline(client, hw_hub,
                                                                  monkeypatch):
    from afm_litho.service.host import scan_cmds

    hub, _stub = hw_hub
    events: list[str] = []
    _spy_arc(hub, events)

    def spy_settle():
        events.append("settle")
        return 0.01

    monkeypatch.setattr(scan_cmds, "mode_settle_s", spy_settle)
    hub.host.imaging_mode = "contact"
    # A fresh AC-tagged baseline from before the switch: if the switch did not clear
    # it, the approach would keep it (no re-read) and value_v would stay 9.9.
    hub.host.free_air = FreeAir(value_v=9.9, ts=time.time(), mode="ac", withdrawn=True,
                           defl_v=0.0)
    r = client.post("/api/approach",
                    json={"mode": "ac", "setpoint": _ac_setpoint(hub, _FREE_AMP_V),
                          "settle_s": 0.0})
    assert r.json().get("ok") is True, r.json()
    i_set = events.index("setmode:ac")
    i_settle = events.index("settle")
    i_amp = events.index("read:Amplitude")
    assert i_set < i_settle < i_amp, events
    assert hub.host.free_air.value_v == pytest.approx(_FREE_AMP_V)


def test_hw_refused_approach_says_the_mode_was_switched(client, hw_hub):
    hub, _stub = hw_hub
    hub.host.imaging_mode = "contact"
    hub.host.free_air = None
    r = client.post("/api/approach",
                    json={"mode": "ac", "setpoint": 0.05 * _FREE_AMP_V, "settle_s": 0.0})
    body = r.json()
    assert body.get("ok") is not True, body
    assert body.get("reason") == "setpoint_out_of_band", body
    assert "switched to 'ac'" in body.get("error", ""), body
    assert hub.host.imaging_mode == "ac"


def test_hw_unknown_mode_matching_live_mode_is_recorded_without_setmode(client, hw_hub):
    hub, stub = hw_hub
    stub.crosspoint = {"InFast": "ACDefl"}
    hub.host.imaging_mode = None
    hub.host.free_air = None
    r = client.post("/api/approach",
                    json={"mode": "ac", "setpoint": _ac_setpoint(hub, _FREE_AMP_V),
                          "settle_s": 0.0})
    assert r.json().get("ok") is True, r.json()
    assert stub.modes == []
    assert hub.host.imaging_mode == "ac"


def test_hw_unknown_live_mode_with_retracted_z_switches(client, hw_hub):
    hub, stub = hw_hub
    stub.crosspoint = {}                        # read-back unavailable -> "unknown"
    _telemetry(hub, 0.0)
    hub.host.imaging_mode = None
    hub.host.free_air = None
    r = client.post("/api/approach",
                    json={"mode": "ac", "setpoint": _ac_setpoint(hub, _FREE_AMP_V),
                          "settle_s": 0.0})
    assert r.json().get("ok") is True, r.json()
    assert _wait_modes(stub, ["AC"]), stub.modes
    assert hub.host.imaging_mode == "ac"


@pytest.mark.parametrize("zdrive_v", [70.0, None], ids=["z_down", "no_telemetry"])
def test_hw_unknown_mode_mismatch_with_tip_down_is_refused(client, hw_hub, zdrive_v):
    hub, stub = hw_hub
    stub.crosspoint = {"InFast": "FilterOut"}
    if zdrive_v is None:
        hub.host._hw_telemetry = None
    else:
        _telemetry(hub, zdrive_v)
    hub.host.imaging_mode = None
    hub.host.free_air = None
    body = client.post("/api/approach",
                       json={"mode": "ac", "setpoint": _ac_setpoint(hub, _FREE_AMP_V),
                             "settle_s": 0.0}).json()
    assert body.get("ok") is not True, body
    assert body.get("reason") == "mode_unknown_tip_engaged", body
    time.sleep(0.2)
    assert stub.modes == []
    assert hub.host.imaging_mode is None
    assert hub.host.engaged is False


def test_hw_unknown_mode_while_engaged_records_without_setmode(hw_hub):
    import asyncio

    from afm_litho.service.host.scan_cmds import ensure_software_mode

    hub, stub = hw_hub
    stub.crosspoint = {"InFast": "FilterOut"}
    _telemetry(hub, 0.0)
    hub.host.imaging_mode = None
    hub.host.engaged = True
    assert asyncio.run(ensure_software_mode(hub, "ac")) is False
    time.sleep(0.2)
    assert stub.modes == []
    assert hub.host.imaging_mode == "ac"


def _wait_modes(stub, want, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if stub.modes == want:
            return True
        time.sleep(0.02)
    return False
