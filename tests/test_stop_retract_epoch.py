"""Stale-withdraw RACE GUARD for the always-retract Stop (docs/STOP_RETRACT_SPEC.md
Sec 2).

Every Stop retracts now. On hardware a running pass owns the bridge link, so the
real retract DEFERS to the scan pump's finally
(`host/scan_pump._deferred_withdraw_if_current`). A deferred withdraw stamped for
scan N must NOT fire into a later scan/approach N+1 (the real bug from the rig
log: an old withdraw ramped Output.Z->0 into a freshly-started scan). The guard
compares the epoch stamped at abort time with `hub._scan_epoch`; a newer
engage/scan bumps the epoch and the stale withdraw is SKIPPED.

These drive the guard helper directly with a fake hardware_arc and a spied
hw_withdraw -- no real bridge, no hardware -- plus a twin-client check that
approach / scan-start actually bump the epoch.
"""

from __future__ import annotations

import asyncio

import afm_litho.hardware.hw_approach as hw_approach_mod
from afm_litho.service.app import hub
from afm_litho.service.host import scan_pump
from afm_litho.service.routers import scan as scan_mod


class _FakeARC:
    """Stand-in for a connected hardware ARC; the guard only passes it through to
    hw_withdraw (which we spy on), so it needs no behaviour."""


def _reset_withdraw_state():
    hub.host._hw_withdraw_after = False
    hub.host._hw_withdraw_epoch = None


def test_deferred_withdraw_fires_when_epoch_unchanged(monkeypatch):
    """Epoch stamped at abort == current epoch -> the deferred retract fires through
    the unified fail-safe primitive `safe_park` (the SAFETY backstop: verified
    Output.Z->0, not a raw hw_withdraw), which clears hub.engaged on a verified
    retract."""
    import afm_litho.service.host.safe_park as safety_mod
    parks: list[str] = []

    async def _spy(hub_arg, reason):
        parks.append(reason)
        hub_arg.host.engaged = False                           # safe_park clears it on verify
        return {"verified": True, "critical": False}

    monkeypatch.setattr(safety_mod, "safe_park_async", _spy)
    prev = hub.host.hardware_arc
    hub.host.hardware_arc = _FakeARC()
    try:
        _reset_withdraw_state()
        hub.host.engaged = True
        hub.host._scan_epoch = 5
        # abort stamped the deferred withdraw under the CURRENT epoch
        hub.host._hw_withdraw_after = True
        hub.host._hw_withdraw_epoch = hub.host._scan_epoch          # 5 == 5
        asyncio.run(scan_pump._deferred_withdraw_if_current())
        assert parks == ["scan stopped"], "the deferred retract must route through safe_park"
        assert hub.host.engaged is False, "the retract must clear the engaged flag"
        assert hub.host._hw_withdraw_after is False
        assert hub.host._hw_withdraw_epoch is None
    finally:
        hub.host.hardware_arc = prev
        _reset_withdraw_state()
        hub.host.engaged = False


def test_deferred_withdraw_skipped_when_epoch_bumped(monkeypatch):
    """A new scan/approach bumps the epoch AFTER the abort stamped its epoch ->
    the deferred withdraw is SKIPPED (no hw_withdraw, so no Output.Z->0 into the
    fresh pass) and the log says superseded."""
    calls = []
    logs: list[str] = []
    monkeypatch.setattr(hw_approach_mod, "hw_withdraw",
                        lambda hw: calls.append(hw) or {"z_v": 0.0})

    async def _spy_log(text):
        logs.append(text)

    monkeypatch.setattr(hub, "log", _spy_log)
    prev = hub.host.hardware_arc
    hub.host.hardware_arc = _FakeARC()
    try:
        _reset_withdraw_state()
        hub.host.engaged = True
        hub.host._scan_epoch = 5
        hub.host._hw_withdraw_after = True
        hub.host._hw_withdraw_epoch = 5
        # a NEW scan/approach claims the tip AFTER the abort stamped epoch 5
        hub.host._scan_epoch = 6
        asyncio.run(scan_pump._deferred_withdraw_if_current())
        assert calls == [], "a stale withdraw must be SKIPPED (no Output.Z->0)"
        assert hub.host.engaged is True, "the fresh engagement keeps the tip on the surface"
        assert hub.host._hw_withdraw_after is False
        assert hub.host._hw_withdraw_epoch is None
        assert any("superseded" in m.lower() for m in logs), logs
    finally:
        hub.host.hardware_arc = prev
        _reset_withdraw_state()
        hub.host.engaged = False


def test_deferred_withdraw_noop_when_nothing_pending(monkeypatch):
    """No deferred withdraw pending -> the helper does nothing (no hw_withdraw)."""
    calls = []
    monkeypatch.setattr(hw_approach_mod, "hw_withdraw",
                        lambda hw: calls.append(hw) or {"z_v": 0.0})
    prev = hub.host.hardware_arc
    hub.host.hardware_arc = _FakeARC()
    try:
        _reset_withdraw_state()
        asyncio.run(scan_pump._deferred_withdraw_if_current())
        assert calls == []
    finally:
        hub.host.hardware_arc = prev
        _reset_withdraw_state()


def test_approach_and_scan_start_bump_the_epoch(client):
    """The bump sites the guard depends on: a successful /api/approach engage and
    a /api/scan/start each advance hub._scan_epoch (twin backend)."""
    # B1 migration: approach now runs check_approach (needs a fresh withdrawn
    # free-air baseline). withdraw clears hub.free_air, so re-seed before the scan.
    from _im_fixtures_guard import seed_free_air
    seed_free_air(hub)
    e0 = hub.host._scan_epoch
    client.post("/api/approach")
    assert hub.host._scan_epoch > e0, "approach must bump the epoch after engaging"
    client.post("/api/withdraw")
    e1 = hub.host._scan_epoch
    client.post("/api/scan/start",
                json={"size_um": 5.0, "pixels": 16, "lines": 4,
                      "line_rate_hz": 200.0, "mode": "contact"})
    try:
        assert hub.host._scan_epoch > e1, "scan/start must bump the epoch"
    finally:
        client.post("/api/scan/abort")   # stop + retract; leave no running scan


# ---- end-to-end through the real scan_abort endpoint (audit coverage gap) ----
#
# The unit tests above drive `_deferred_withdraw_if_current` with a HAND-SET epoch.
# These drive the REAL `scan_abort` endpoint handler with a genuinely-running pass
# (a pending `_scan_task`) so the STAMP path is exercised end-to-end, and use a REAL
# `/api/approach` engage to bump the epoch (not a hand-set) for the supersede case.


def _pending_scan_task() -> asyncio.Task:
    """A never-completing task standing in for a running hardware pass (so
    `scan_abort` sees `_scan_task` not-done and takes the DEFER-and-stamp path)."""
    async def _never():
        await asyncio.Event().wait()
    return asyncio.create_task(_never())


def test_scan_abort_endpoint_stamps_epoch_then_deferred_retract_fires(monkeypatch):
    """End-to-end: a hardware Stop WHILE a pass is running defers the real retract and
    stamps it under the CURRENT epoch (`_hw_withdraw_epoch == _scan_epoch`,
    `_hw_withdraw_after` set); when the pass then ends with the epoch unchanged, the
    pump's `_deferred_withdraw_if_current` fires the retract through `safe_park`."""
    import afm_litho.service.host.safe_park as safety_mod
    parks: list[str] = []

    async def _spy(hub_arg, reason):
        parks.append(reason)
        hub_arg.host.engaged = False
        return {"verified": True, "critical": False}

    monkeypatch.setattr(safety_mod, "safe_park_async", _spy)
    prev_hw = hub.host.hardware_arc
    prev_task = hub.host._scan_task
    try:
        async def _run():
            hub.host.hardware_arc = _FakeARC()
            _reset_withdraw_state()
            hub.host.engaged = True
            hub.host._scan_epoch = 10
            hub.host._hw_abort.clear()
            hub.host.scan_repeat["enabled"] = False
            hub.host._scan_task = _pending_scan_task()         # a pass owns the link
            await asyncio.sleep(0)                         # let it start (still not done)
            # REAL endpoint: Stop defers the retract and stamps the current epoch.
            res = await scan_mod.scan_abort({"withdraw": True})
            assert res == {"ok": True, "withdraw": True}
            assert hub.host._hw_withdraw_after is True, "the retract must be deferred + pending"
            assert hub.host._hw_withdraw_epoch == hub.host._scan_epoch == 10, "stamped under cur epoch"
            assert parks == [], "the retract must NOT fire while the pass still owns the link"
            hub.host._scan_task.cancel()                        # the pass releases the link
            # pump's finally: epoch unchanged -> the deferred retract FIRES via safe_park.
            await scan_pump._deferred_withdraw_if_current()

        asyncio.run(_run())
        assert parks == ["scan stopped"], parks
        assert hub.host.engaged is False
        assert hub.host._hw_withdraw_after is False
        assert hub.host._hw_withdraw_epoch is None
    finally:
        hub.host.hardware_arc = prev_hw
        hub.host._scan_task = prev_task
        _reset_withdraw_state()
        hub.host.engaged = False


def test_scan_abort_endpoint_stamp_superseded_by_real_engage(monkeypatch):
    """End-to-end supersede: a hardware Stop stamps the deferred retract under epoch N;
    a REAL `/api/approach` engage (twin path -- a genuine bump, not hand-set) advances
    the epoch to N+1; the pump's `_deferred_withdraw_if_current` then SKIPS the retract
    as superseded (no safe_park, engaged stays True), so it can never ramp Output.Z->0
    into the fresh engagement."""
    import afm_litho.service.host.safe_park as safety_mod
    parks: list[str] = []
    logs: list[str] = []

    async def _spy(hub_arg, reason):
        parks.append(reason)
        return {"verified": True, "critical": False}

    async def _spy_log(text):
        logs.append(text)

    monkeypatch.setattr(safety_mod, "safe_park_async", _spy)
    monkeypatch.setattr(hub, "log", _spy_log)
    prev_hw = hub.host.hardware_arc
    prev_task = hub.host._scan_task
    try:
        async def _run():
            hub.host.hardware_arc = _FakeARC()
            _reset_withdraw_state()
            hub.host.engaged = True
            hub.host._scan_epoch = 20
            hub.host._hw_abort.clear()
            hub.host.scan_repeat["enabled"] = False
            hub.host._scan_task = _pending_scan_task()
            await asyncio.sleep(0)
            await scan_mod.scan_abort({"withdraw": True})   # stamps epoch 20
            assert hub.host._hw_withdraw_epoch == 20
            hub.host._scan_task.cancel()
            # a REAL new engagement claims the tip and bumps the epoch (twin approach
            # endpoint: hub._scan_epoch += 1) -- NOT a hand-set.
            hub.host.hardware_arc = None                         # force the twin approach path
            # B1 migration: approach now runs check_approach (needs fresh free-air).
            from _im_fixtures_guard import seed_free_air
            seed_free_air(hub)
            await scan_mod.approach({})
            assert hub.host._scan_epoch == 21, "the real approach must bump the epoch"
            hub.host.hardware_arc = _FakeARC()                   # restore hw for the deferred call
            await scan_pump._deferred_withdraw_if_current()  # stamp 20 != 21 -> SKIP

        asyncio.run(_run())
        assert parks == [], "a superseded deferred retract must NOT safe_park"
        assert hub.host.engaged is True, "the fresh engagement keeps the tip on the surface"
        assert hub.host._hw_withdraw_after is False
        assert any("superseded" in m.lower() for m in logs), logs
    finally:
        hub.host.hardware_arc = prev_hw
        hub.host._scan_task = prev_task
        _reset_withdraw_state()
        hub.host.engaged = False
