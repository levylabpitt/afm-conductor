"""The startup Surface replay is OFF by default (test-suite audit Phase 2 slice 2).

Replaying every raw pass in the sample's raw dir on each app start walked the whole
directory in an uncancellable thread (~30 s+ per TestClient lifespan on a box with
thousands of .npz), which dominated the suite. These tests pin the new contract:

  * a default app start does NOT touch `SurfaceModel.reload`, and explains itself both
    on the WS log AND in the service log at warning level (so it survives uvicorn's
    default log level and a panel that connects later can still find out why the
    Surface model starts empty);
  * `AFM_LITHO_REBUILD_SURFACE=1` restores the startup replay (reload is scheduled);
  * a rebuild is single-flight: a second request while one is running is refused
    rather than racing the first and fusing every pass twice.

All fast: `SurfaceModel.reload` is stubbed to record (or block on) the call, and the
raw dir is redirected at a tmp dir with a known number of .npz files, so nothing walks
the real runtime dir.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import afm_litho.service.routers.surface as surface_mod
from afm_litho.service.app import app
from afm_litho.service.hub import hub
from afm_litho.service.surface import SurfaceModel


def _seed_raw(tmp_path: Path, n: int) -> Path:
    """A raw dir with `n` .npz files (plus a decoy archive dir + non-npz file)."""
    for i in range(n):
        (tmp_path / f"pass_{i}.npz").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")           # must not be counted
    (tmp_path / "archive-old").mkdir()                  # non-recursive: not counted
    (tmp_path / "archive-old" / "buried.npz").write_bytes(b"")
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_rebuild_marker():
    """The in-flight rebuild task is a module global; clear it around every test so a
    scheduled rebuild from one test never looks in-flight to the next."""
    surface_mod._rebuild_task = None
    yield
    surface_mod._rebuild_task = None


@pytest.fixture
def record_reload(monkeypatch):
    """Stub SurfaceModel.reload so nothing walks the real raw dir; record each call."""
    calls: list[bool] = []

    def _stub(self, limit=None):
        calls.append(True)
        return 0                                         # 0 -> no surface_changed broadcast

    monkeypatch.setattr(SurfaceModel, "reload", _stub)
    return calls


@pytest.fixture
def record_log(monkeypatch):
    """Record every hub.log line so we can find the startup skip message."""
    msgs: list[str] = []

    async def _log(text):
        msgs.append(text)

    monkeypatch.setattr(hub, "log", _log)
    return msgs


def _wait_for(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


# --------------------------------------------------------------- startup: OFF --

def test_default_startup_does_not_replay_and_logs_skip(
    tmp_path, monkeypatch, record_reload, record_log, caplog
):
    monkeypatch.delenv("AFM_LITHO_REBUILD_SURFACE", raising=False)
    monkeypatch.setattr(hub.surface, "raw_dir", _seed_raw(tmp_path, 3))

    with caplog.at_level(logging.INFO, logger="afm_litho.service.app"), TestClient(app):
        # Give any stray background task a chance to run; reload stays untouched.
        time.sleep(0.2)

    assert record_reload == [], "startup must not call SurfaceModel.reload by default"

    # (a) the WS log line, for panels connected now
    skip = [m for m in record_log if "startup replay skipped" in m]
    assert len(skip) == 1, record_log
    assert "3 raw passes" in skip[0]

    # (b) the service-log record at WARNING, which survives uvicorn's default log level
    logged = [r for r in caplog.records if "startup replay skipped" in r.getMessage()]
    assert len(logged) == 1, [r.getMessage() for r in caplog.records]
    assert "3 raw passes" in logged[0].getMessage()
    assert logged[0].levelno == logging.WARNING


# ---------------------------------------------------------------- startup: ON --

def test_flag_on_schedules_the_replay(
    tmp_path, monkeypatch, record_reload
):
    monkeypatch.setenv("AFM_LITHO_REBUILD_SURFACE", "1")
    monkeypatch.setattr(hub.surface, "raw_dir", _seed_raw(tmp_path, 2))

    with TestClient(app):
        assert _wait_for(lambda: len(record_reload) >= 1), \
            "AFM_LITHO_REBUILD_SURFACE=1 must schedule SurfaceModel.reload at startup"


# ------------------------------------------------------------- single flight --

def test_second_rebuild_while_one_is_running_is_not_started(tmp_path, monkeypatch):
    """Overlapping rebuilds would race SurfaceModel._replaying and double-fuse every pass,
    so `schedule_rebuild` refuses a second while one is running."""
    import asyncio

    monkeypatch.setattr(hub.surface, "raw_dir", _seed_raw(tmp_path, 6))
    gate = threading.Event()
    calls: list[bool] = []

    def _blocking_reload(self, limit=None):
        calls.append(True)
        gate.wait(timeout=5.0)                           # hold the first rebuild open
        return 0

    monkeypatch.setattr(SurfaceModel, "reload", _blocking_reload)

    async def scenario():
        assert surface_mod.schedule_rebuild() is True
        while not calls:
            await asyncio.sleep(0.01)
        assert surface_mod.rebuild_in_flight()
        assert surface_mod.schedule_rebuild() is False
        gate.set()
        await surface_mod._rebuild_task
        assert not surface_mod.rebuild_in_flight()

    try:
        asyncio.run(scenario())
    finally:
        gate.set()
    assert calls == [True], "reload must run exactly once; the second request scheduled none"
