"""The service owns the journal (slice 2.1): started at startup in the runtime folder in use,
drained at shutdown, and every FLEX request leaves an entry with its actor."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from _service import Service

from afm_litho.service import deps, journal

pytest.importorskip("zmq", reason="pyzmq is the optional [flex] extra")


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def _door_entries(folder: Path) -> list[dict]:
    files = list(folder.glob("*.jsonl"))
    return [e for e in _lines(files[0]) if e["src"] == "door"] if files else []


def _flex_entries(folder: Path) -> list[dict]:
    """Only what FLEX sent; the service's own HTTP reads (readiness probe) are journalled too."""
    return [e for e in _door_entries(folder) if str(e["actor"]).startswith("flex")]


def _wait_for_flex_entries(folder: Path, count: int, timeout_s: float = 10.0) -> None:
    """The writer thread is asynchronous and the service is killed, not drained, in this test."""
    deadline = time.time() + timeout_s
    while len(_flex_entries(folder)) < count and time.time() < deadline:
        time.sleep(0.1)


def test_startup_starts_the_journal_in_the_runtime_folder_and_shutdown_drains_it():
    from fastapi.testclient import TestClient

    from afm_litho.service.app import app
    from afm_litho.service.flex_provider import handle_message
    from afm_litho.service.hub import hub
    folder = deps.RUNTIME / "journal"
    with TestClient(app):
        running = journal.running()
        assert isinstance(running, journal.Journal)
        assert running.path.parent == folder
        assert folder.is_relative_to(Path(os.environ["AFM_LITHO_RUNTIME"]))
        handle_message(hub, json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ACK"}))
        path = running.path
    assert journal.running() is not running
    entries = _lines(path)
    assert entries[0]["what"] == "header"
    assert [e["what"] for e in entries if e["src"] == "door"] == ["request", "reply"]
    assert entries[-1]["what"] == "closed"


def test_a_live_service_journals_flex_requests_with_the_actor(tmp_path):
    import zmq
    svc = Service(runtime=tmp_path / "rt", twin_speed=0.0, flex=True).start()
    ctx = zmq.Context()
    try:
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, 10000)
        sock.connect(f"tcp://127.0.0.1:{svc.flex_port}")
        for i, method in enumerate(("ACK", "getState")):
            sock.send_string(json.dumps({"jsonrpc": "2.0", "id": str(i), "method": method}))
            assert "result" in json.loads(sock.recv_string())
        _wait_for_flex_entries(tmp_path / "rt" / "journal", 4)
    finally:
        ctx.destroy(linger=0)
        svc.stop()
    (path,) = (tmp_path / "rt" / "journal").glob("*.jsonl")
    door = _flex_entries(path.parent)
    assert [(e["what"], e["body"]["row"]) for e in door] == [
        ("request", "ACK"), ("reply", "ACK"), ("request", "getState"), ("reply", "getState")]
    assert {e["actor"] for e in door} == {"flex (unnamed)"}
    assert door[0]["trace"] == door[1]["trace"] != door[2]["trace"]


def test_an_unwritable_journal_folder_does_not_stop_the_service_or_flex(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from afm_litho.service.app import app
    from afm_litho.service.flex_provider import handle_message
    from afm_litho.service.hub import hub
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the runtime folder should be")
    monkeypatch.setattr(deps, "RUNTIME", blocker / "rt")
    with TestClient(app):
        reply = json.loads(handle_message(
            hub, json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ACK"})))
    assert reply["result"] == "ACK"
