"""Every change of the hub's fault latch reaches open pages as a `fault` ws frame
(docs/COCKPIT_BANNERS_SPEC.md: the cockpit's fault banner follows the latch).

A latch set from any thread and `POST /api/fault/clear` both broadcast
`{"type": "fault", "fault": <record or null>}`; setting the same value again does
not. GET /api/state keeps carrying the record for pages that connect later.
"""

from __future__ import annotations

import threading
import time

from afm_litho.service.hub import hub


def _ready_ws(ws) -> None:
    ws.send_text("ping")
    for _ in range(200):
        if _recv(ws).get("type") == "telemetry":
            return
    raise AssertionError("ws never confirmed registration")


def _recv(ws, timeout_s: float = 5.0) -> dict:
    """`receive_json` with a timeout: the TestClient's receive blocks forever, so a
    missing frame would otherwise hang to the suite's per-test timeout."""
    box: dict = {}
    t = threading.Thread(target=lambda: box.setdefault("m", ws.receive_json()),
                         daemon=True)
    t.start()
    t.join(timeout_s)
    if "m" not in box:
        raise AssertionError(f"no ws frame within {timeout_s} s")
    return box["m"]


def _next_fault_frame(ws, timeout_s: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        m = _recv(ws, max(0.1, deadline - time.monotonic()))
        if m.get("type") == "fault":
            return m
    raise AssertionError(f"no fault frame within {timeout_s} s")


def _record(reason: str = "withdraw_failed") -> dict:
    return {"reason": reason, "at": time.time(), "state": "fault",
            "detail": "test latch"}


def test_latch_and_clear_are_broadcast(client):
    with client.websocket_connect("/ws") as ws:
        _ready_ws(ws)
        rec = _record()
        hub.fault = rec
        m = _next_fault_frame(ws)
        assert m["fault"] == rec

        assert client.get("/api/state").json()["fault"] == rec

        r = client.post("/api/fault/clear", json={"confirm": True}).json()
        assert r.get("cleared") is True, r
        m = _next_fault_frame(ws)
        assert m["fault"] is None
        assert client.get("/api/state").json()["fault"] is None


def test_latch_from_a_worker_thread_is_broadcast(client):
    with client.websocket_connect("/ws") as ws:
        _ready_ws(ws)
        rec = _record("park_unverified")
        t = threading.Thread(target=lambda: setattr(hub, "fault", rec))
        t.start()
        t.join()
        assert _next_fault_frame(ws)["fault"] == rec
        hub.fault = None
        assert _next_fault_frame(ws)["fault"] is None


def test_an_unconfirmed_clear_leaves_the_latch_and_sends_nothing(client):
    with client.websocket_connect("/ws") as ws:
        _ready_ws(ws)
        rec = _record()
        hub.fault = rec
        assert _next_fault_frame(ws)["fault"] == rec
        r = client.post("/api/fault/clear", json={}).json()
        assert r.get("cleared") is False and r.get("fault") == rec, r
        # the same value again is not a change: the next fault frame is the real clear
        hub.fault = dict(rec)
        hub.fault = None
        assert _next_fault_frame(ws)["fault"] is None


def test_a_failing_broadcast_still_sets_the_latch(client, monkeypatch):
    import afm_litho.service.hub as hub_mod

    def boom(*_a, **_k):
        raise RuntimeError("send failed")

    monkeypatch.setattr(hub_mod.asyncio, "run_coroutine_threadsafe", boom)
    rec = _record()
    hub.fault = rec                                      # must not raise
    assert hub.fault == rec
    monkeypatch.setattr(hub, "_announce_fault", boom)
    rec2 = _record("park_unverified")
    hub.fault = rec2                                     # must not raise either
    assert hub.fault == rec2
    assert client.get("/api/state").json()["fault"] == rec2
    monkeypatch.undo()
    hub.fault = None


def test_an_operator_clear_is_logged_once_with_its_time(client):
    with client.websocket_connect("/ws") as ws:
        _ready_ws(ws)
        hub.fault = _record()
        _next_fault_frame(ws)
        client.post("/api/fault/clear", json={"confirm": True})
        logs = []
        for _ in range(400):
            m = _recv(ws)
            if m.get("type") == "log" and "fault cleared by operator" in m.get("msg", ""):
                logs.append(m)
            if m.get("type") == "fault":
                assert m["fault"] is None
                break
        for _ in range(50):                              # the log may trail the frame
            if logs:
                break
            m = _recv(ws)
            if m.get("type") == "log" and "fault cleared by operator" in m.get("msg", ""):
                logs.append(m)
        assert len(logs) == 1, logs
        assert "withdraw_failed" in logs[0]["msg"] and logs[0].get("t")


def test_concurrent_accepts_log_exactly_one_clear(client):
    with client.websocket_connect("/ws") as ws:
        _ready_ws(ws)
        hub.fault = _record()
        _next_fault_frame(ws)
        results = []
        barrier = threading.Barrier(4)

        def accept():
            barrier.wait()
            results.append(client.post("/api/fault/clear", json={"confirm": True}).json())

        threads = [threading.Thread(target=accept) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        assert all(r.get("cleared") is True for r in results) and len(results) == 4
        logs = []
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                m = _recv(ws, max(0.1, deadline - time.monotonic()))
            except AssertionError:
                break
            if m.get("type") == "log" and "fault cleared by operator" in m.get("msg", ""):
                logs.append(m)
        assert len(logs) == 1, logs
