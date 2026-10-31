"""The hardware scan pump drives the Asylum plugin with `service_owns_scan`: the service has
done the mode reconcile, the XY-cal refresh and the abort clear, and withdraws itself. The
plugin must do none of those, and a cancelled pump must stop the scan before it arms Z."""

import asyncio
import time

import pytest

from afm_litho.instrument import asylum as A
from afm_litho.service import hw_scan
from afm_litho.service.passes import ScanConfig
from afm_litho.service.host import scan_pump as S


class _Arc:
    is_cypher = True
    timeout = 5.0


@pytest.fixture
def pump(tmp_path, monkeypatch):
    """Run the scan pump on a stand-in arc, with the plugin's chained scan replaced."""
    from afm_litho.service.sessions import SessionStore
    hub = S.hub
    monkeypatch.setattr(hub, "store", SessionStore(tmp_path / "sessions.jsonl"))
    monkeypatch.setattr(hub.host, "hardware_arc", _Arc())
    monkeypatch.setattr(hub.host, "hw_cal", {})
    seen = {"withdraws": 0, "reconcile": 0, "plugin_xy_cal": 0, "pump_xy_cal": 0,
            "during_cal": None}

    def count(name):
        def counted(*args, **kwargs):
            seen[name] += 1
        return counted

    async def pump_xy_cal(source):
        seen["pump_xy_cal"] += 1
        if seen["during_cal"]:
            seen["during_cal"]()

    monkeypatch.setattr(A, "hw_withdraw", count("withdraws"))
    monkeypatch.setattr(A.AsylumInstrument, "_reconcile_mode", count("reconcile"))
    monkeypatch.setattr(A.AsylumInstrument, "_refresh_xy_cal", count("plugin_xy_cal"))
    monkeypatch.setattr(hub, "refresh_xy_cal", pump_xy_cal)

    def run(fake, during_cal=None):
        seen["during_cal"] = during_cal
        monkeypatch.setattr(A, "run_hw_scan_chained", fake)
        hub.host._hw_abort.clear()
        try:
            asyncio.run(S._hw_scan_pump(ScanConfig(size_um=1.0, pixels=8, lines=2)))
        finally:
            hub.host._hw_abort.clear()
        return seen

    return run


def _summary(**extra):
    return {"lines": 0, "pixels": 8, "capture": {}, "line_indices": [], "aborted": False,
            "error": None, **extra}


def test_a_chained_error_or_failure_is_not_withdrawn_by_the_plugin(pump):
    async def errored(*args, **kwargs):
        return _summary(aborted=True, error="boom")

    async def raised(*args, **kwargs):
        raise RuntimeError("boom")

    assert pump(errored)["withdraws"] == 0
    assert pump(raised)["withdraws"] == 0


def test_a_stop_set_before_the_plugin_starts_still_aborts_the_scan(pump):
    saw = []

    async def fake(arc, config, cal, sink, *, should_abort, **kwargs):
        saw.append(should_abort())
        return _summary(aborted=bool(should_abort()))

    pump(fake, during_cal=S.hub.host._hw_abort.set)      # a Stop while the XY cal is refreshed
    assert saw == [True]


def test_the_plugin_leaves_mode_and_xy_cal_to_the_service(pump):
    async def fake(*args, **kwargs):
        return _summary()

    seen = pump(fake)
    assert seen["reconcile"] == 0
    assert seen["plugin_xy_cal"] == 0
    assert seen["pump_xy_cal"] == 1


def test_a_cancelled_pump_stops_the_scan_before_z_is_armed(monkeypatch):
    monkeypatch.setattr(S.hub.host, "hardware_arc", _Arc())
    monkeypatch.setattr(S.hub.host, "hw_cal", {})
    armed = []

    async def slow_transit(*args, **kwargs):
        await asyncio.to_thread(time.sleep, 0.4)     # the blocking xy_transit

    def raster(*args, **kwargs):
        armed.append("raster")
        raise RuntimeError("the raster was built")

    monkeypatch.setattr(hw_scan, "_pre_scan_xy_transit", slow_transit)
    monkeypatch.setattr(hw_scan, "raster_pass", raster)

    async def main():
        task = asyncio.ensure_future(S._scan_on_instrument(
            ScanConfig(size_um=1.0, pixels=8, lines=2, engage=True), S.NO_NOTES))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.8)

    asyncio.run(main())
    assert armed == []
