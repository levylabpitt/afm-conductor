"""`HostState.pass_ended`: every pass tells the base exactly once how it ended (H7.5a).

A spy stands in for the base handler. A pass gives one call with JSON-plain arguments, and
no page frame is named `pass_ended`: the handler is a base function, not the page sink.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from _im_fixtures_guard import GUARD_WRITE_KW, seed_free_air

from afm_litho.service.host import scan_cmds, scan_pump

SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg">'
    '<path d="M 0 0 L 3000 0"/><path d="M 0 1500 L 3000 1500"/></svg>'
)


class _Calls(list):
    """The handler calls; `.frames` holds the page frame types sent meanwhile."""

    frames: list


@pytest.fixture
def real_hub(client):
    from afm_litho.service.hub import hub

    yield hub
    for _ in range(100):                   # let a stray pass release its claim
        if hub._run_claim is None:
            break
        time.sleep(0.05)
    hub._run_claim = None


@pytest.fixture
def ended(real_hub, monkeypatch):
    """Calls to the base handler, in order, plus every page frame sent meanwhile."""
    calls, frames = _Calls(), []
    real_handler = real_hub.host._on_pass_ended

    def _spy(*a):
        calls.append(a)
        real_handler(*a)                   # the base still releases the claim

    monkeypatch.setattr(real_hub.host, "_on_pass_ended", _spy)
    sink, broadcast = real_hub.host._sink, real_hub.broadcast

    async def _sink(msg):
        frames.append(msg.get("type"))
        await sink(msg)

    async def _broadcast(msg):
        frames.append(msg.get("type"))
        await broadcast(msg)

    monkeypatch.setattr(real_hub.host, "_sink", _sink)
    monkeypatch.setattr(real_hub, "broadcast", _broadcast)
    calls.frames = frames
    return calls


def _wait_closed(hub, run_id, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rec = hub.runs.get(run_id)
        if rec is not None and rec.finished is not None:
            return rec
        time.sleep(0.05)
    raise AssertionError(f"{run_id} never closed")


def _only(ended, run_id):
    mine = [c for c in ended if c[0] == run_id]
    assert len(mine) == 1, mine
    run, outcome, written, files = mine[0]
    assert json.loads(json.dumps(mine[0])) == list(mine[0])
    assert "pass_ended" not in ended.frames
    return outcome, written, files


def test_a_twin_scan_ends_once_completed(client, real_hub, ended):
    r = client.post("/api/scan/start", json={"size_um": 5, "pixels": 8, "lines": 8,
                                             "line_rate_hz": 40.0}).json()
    assert r.get("ok"), r
    _wait_closed(real_hub, r["run_id"])
    assert _only(ended, r["run_id"]) == ("completed", None, [])
    assert real_hub._run_claim is None


def test_an_aborted_twin_scan_ends_once_aborted(client, real_hub, ended):
    r = client.post("/api/scan/start", json={"size_um": 5, "pixels": 64, "lines": 64,
                                             "line_rate_hz": 5.0}).json()
    assert r.get("ok"), r
    client.post("/api/scan/abort")
    _wait_closed(real_hub, r["run_id"])
    assert _only(ended, r["run_id"]) == ("aborted", None, [])


def test_a_write_ends_once_with_its_object_count(client, real_hub, ended, monkeypatch):
    monkeypatch.setattr(real_hub.host, "last_scan_config", {"size_um": 30.0})
    assert "error" not in client.post("/api/litho/load", json={"source": SVG}).json()
    real_hub.host.written_objects = set()
    seed_free_air(real_hub)
    body = {"objects": [o.name for o in real_hub.pattern.objects],
            "setpoint_relative": 0.2, **GUARD_WRITE_KW}
    r = client.post("/api/litho/write", json=body).json()
    assert r.get("ok"), r
    _wait_closed(real_hub, r["run_id"])
    assert _only(ended, r["run_id"]) == ("completed", 2, [])


def test_an_aborted_write_ends_once_aborted(client, real_hub, ended, monkeypatch):
    monkeypatch.setattr(real_hub.host, "last_scan_config", {"size_um": 30.0})
    assert "error" not in client.post("/api/litho/load", json={"source": SVG}).json()
    real_hub.host.written_objects = set()
    seed_free_air(real_hub)
    body = {"objects": [o.name for o in real_hub.pattern.objects],
            "setpoint_relative": 0.2, **GUARD_WRITE_KW}
    r = client.post("/api/litho/write", json=body).json()
    assert r.get("ok"), r
    client.post("/api/litho/abort")
    _wait_closed(real_hub, r["run_id"])
    outcome, _written, files = _only(ended, r["run_id"])
    assert outcome in ("aborted", "completed")      # an abort can lose the race to a 2 s write
    assert all(isinstance(f, str) for f in files)


def test_an_approach_ends_once(real_hub, ended, monkeypatch):
    async def _impl(hub, body):
        return {"z_um": 0.0}

    monkeypatch.setattr(scan_cmds, "_approach_impl", _impl)

    async def _go():
        run_id = real_hub.next_run_id("approach")
        claim = real_hub.claim_run("approach")
        claim.hand_over(run_id, None)
        await scan_cmds._approach_run(real_hub, {}, run_id, claim)
        return run_id

    run_id = asyncio.run(_go())
    assert _only(ended, run_id) == ("completed", None, [])
    assert real_hub._run_claim is None


def test_a_failed_approach_ends_once_failed(real_hub, ended, monkeypatch):
    async def _impl(hub, body):
        raise RuntimeError("engage exploded")

    monkeypatch.setattr(scan_cmds, "_approach_impl", _impl)

    async def _go():
        run_id = real_hub.next_run_id("approach")
        claim = real_hub.claim_run("approach")
        claim.hand_over(run_id, None)
        await scan_cmds._approach_run(real_hub, {}, run_id, claim)
        return run_id

    run_id = asyncio.run(_go())
    assert _only(ended, run_id) == ("failed", None, [])


def test_a_pump_that_raises_ends_once_failed(real_hub, ended, monkeypatch):
    async def _boom(config, rid=None, claim=None, *extra):
        raise RuntimeError("sim scan exploded")

    monkeypatch.setattr(scan_pump, "_scan_pump_body", _boom)
    run_id = real_hub.next_run_id("scan")
    asyncio.run(scan_pump._scan_pump(None, run_id))
    assert _only(ended, run_id) == ("failed", None, [])


def test_a_repeat_for_one_run_is_dropped(real_hub, ended):
    for outcome in ("completed", "failed"):
        real_hub.host.pass_ended("scan-repeat", outcome, None, [])
    assert [c[:2] for c in ended] == [("scan-repeat", "completed")]


def test_a_failing_pass_ended_handler_does_not_raise_into_the_release():
    from afm_litho.service.host_state import HostState
    host = HostState.__new__(HostState)
    host._ended_runs, host._last_heard = {}, None
    host.heard = lambda: None

    def boom(*_a):
        raise RuntimeError("handler broke")

    host._on_pass_ended = boom
    host.pass_ended("scan-9", "completed", None, [])     # must not raise
    assert "scan-9" not in host._ended_runs              # so the backstop can retry

    calls = []
    host._on_pass_ended = lambda *a: calls.append(a)
    host.pass_ended("scan-9", "failed", None, [])        # the pump's finally backstop
    assert len(calls) == 1 and "scan-9" in host._ended_runs


def test_a_repeat_pass_ended_leaves_a_newer_claim_alone(real_hub):
    from afm_litho.service.runs import on_pass_ended

    old = real_hub.claim_run("write")
    old.hand_over("write-old", None)
    on_pass_ended(real_hub, "write-old", "completed", 2, [])
    assert real_hub._run_claim is None
    new = real_hub.claim_run("write")
    new.hand_over("write-new", None)
    on_pass_ended(real_hub, "write-old", "failed", None, [])     # a late backstop
    assert real_hub._run_claim is new
    real_hub.release_run(None)                                   # no id releases nothing
    assert real_hub._run_claim is new
    on_pass_ended(real_hub, "write-new", "completed", 2, [])
    assert real_hub._run_claim is None


def test_the_base_claim_gates_new_starts_until_the_base_releases(
        client, real_hub, monkeypatch):
    """The host is done, but the base has not yet released: both starts are refused busy.

    The held claim has a live worker (as in a link with a gap); only the base's release
    by run id frees it."""
    monkeypatch.setattr(real_hub.host, "_on_pass_ended", lambda *a: None)
    r = client.post("/api/scan/start", json={"size_um": 5, "pixels": 8, "lines": 8,
                                             "line_rate_hz": 40.0}).json()
    assert r.get("ok"), r
    _wait_closed(real_hub, r["run_id"])
    claim = real_hub._run_claim
    assert claim is not None and claim.run_id == r["run_id"]

    class _Alive:
        def is_alive(self):
            return True

    claim.worker = _Alive()
    assert client.post("/api/scan/start", json={"size_um": 5}).json().get("error") == "busy"
    assert client.post("/api/litho/write", json={}).json().get("error") == "busy"
    real_hub.release_run(r["run_id"])
    assert real_hub._run_claim is None
    real_hub.release_claim(real_hub.claim_run("scan"))        # a start is accepted again
