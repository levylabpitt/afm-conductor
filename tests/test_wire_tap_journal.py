"""Slice J2: the journal's wire tap notes after each send and each upward frame, and can never
alter them.

The proof that matters: every golden is the same with the journal on and with its writer
stalled, and the scenarios finish as fast as without a journal.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import numpy as np
import pytest
from golden.harness import HERE, MFP3D_MICROSCOPE_ID, compare_wire, record
from golden.scenarios import OPTIONS, SCENARIOS
from test_journal import StalledDisk

from afm_litho.goldens import golden_format as G
from afm_litho.hardware import bridge_protocol as P
from afm_litho.hardware.bridge_router import BridgeRouter
from afm_litho.hardware.igor import Igor
from afm_litho.service import journal as journal_module
from afm_litho.service.journal import Journal

ABORTS = ["abort_mid_stroke", "abort_in_transit", "abort_at_chain_seam", "safe_park",
          "safe_park_com_down", "safe_park_vendor_withdraw", "scan_stop_deferred_withdraw",
          "deadman_during_write"]
MFP_NAMES = [n for n in SCENARIOS if OPTIONS.get(n, {}).get("mfp", True) is True]


@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


@pytest.fixture(autouse=True)
def _no_journal_left_running():
    yield
    journal_module._running = None


def _lines(j: Journal) -> list[dict]:
    j.stop()
    return [json.loads(x) for x in j.path.read_text("utf-8").splitlines()]


def _timed(hub, client, name: str) -> tuple[dict, float]:
    t0 = time.monotonic()
    g = record(hub, name, client=client)
    return g, time.monotonic() - t0


def _same_as_frozen(g: dict, name: str) -> None:
    problems = G.compare(G.load(HERE / f"{name}.golden.json"), g)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_golden_unchanged_with_journal_on_and_writer_stalled(hub, client, tmp_path, name):
    j = Journal(tmp_path / "on").start()
    on = record(hub, name, client=client)
    wire = [e for e in _lines(j) if e["src"] == "wire"]
    _same_as_frozen(on, name)
    assert wire, "the tap noted nothing"
    base_s = None
    if name in ABORTS:
        _, base_s = _timed(hub, client, name)          # today's bound: no journal at all
    disk = StalledDisk()
    stalled = Journal(tmp_path / "stalled", sink=disk).start()
    try:
        g, stalled_s = _timed(hub, client, name)
    finally:
        disk.release.set()
        stalled.stop()
    _same_as_frozen(g, name)
    if base_s is not None:
        # Stop timing unchanged: a stalled journal adds no wait to an abort scenario.
        assert stalled_s < max(base_s * 1.5, base_s + 3.0), (name, base_s, stalled_s)


@pytest.mark.parametrize("name", MFP_NAMES)
def test_mfp3d_wire_golden_unchanged_with_journal_on(hub, client, tmp_path, name):
    j = Journal(tmp_path).start()
    g = record(hub, name, client=client, microscope_id=MFP3D_MICROSCOPE_ID)
    j.stop()
    assert not compare_wire(G.load(HERE / f"{name}.mfp3d.golden.json"), g)


class _RaisingJournal:
    def note(self, *a, **k):
        raise RuntimeError("journal exploded")


def _igor_with(execute2) -> Igor:
    igor = Igor()
    igor._app = type("App", (), {"Execute2": execute2})()
    return igor


def test_a_raising_note_cannot_affect_send_read_or_com(monkeypatch):
    monkeypatch.setattr(journal_module, "_running", _RaisingJournal())
    a, b = socket.socketpair()
    seen = threading.Event()
    link = P.BridgeLink(a)
    router = BridgeRouter(link, poll_timeout=0.05, log=lambda m: seen.set()).start()
    try:
        link.send("GETVALUE u1 Height")
        assert b.recv(100) == b"GETVALUE u1 Height\n"
        b.sendall(b"VALUE u9 1.5\n")             # unknown UID: the reader logs and carries on
        assert seen.wait(2)
        seen.clear()
        b.sendall(b"VALUE u9 2.5\n")
        assert seen.wait(2)                      # the reader is still alive after a raise
        link.send("ABORT")
        assert b.recv(100) == b"ABORT\n"
    finally:
        router.stop()
        a.close()
        b.close()
    ran = []
    igor = _igor_with(lambda self, *x: ran.append(x[2]) or (0, "", "", ""))
    assert igor.execute("AFM_SafePark()").error_code == 0
    assert ran == ["AFM_SafePark()"]


class _Recorder:
    def __init__(self):
        self.notes = []

    def note(self, src, what, /, **body):
        self.notes.append((src, what, body))


def test_com_note_comes_after_execute2_returns(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(journal_module, "_running", rec)
    seen_during = []
    igor = _igor_with(lambda self, *x: seen_during.append(len(rec.notes)) or (0, "", "", ""))
    igor.execute("AFM_SafePark()")
    assert seen_during == [0]                    # nothing noted before the command was sent
    assert rec.notes == [("wire", "com", {"text": "AFM_SafePark()", "code": 0,
                                       "thread": threading.current_thread().name})]


def test_request_wire_reply_excerpt(tmp_path):
    j = Journal(tmp_path).start()
    a, b = socket.socketpair()
    seen = threading.Event()
    link = P.BridgeLink(a)
    router = BridgeRouter(link, poll_timeout=0.05, log=lambda m: seen.set()).start()
    try:
        j.note("door", "request", actor="operator", trace=7, row="writeWave")
        link.send(P.cmd_wave("w1", np.arange(500.0)))
        link.send("GETVALUE u1 Height")
        b.recv(10000)
        b.sendall(b"TELEM 1 2 3 4 5 6\nVALUE u1 1.5\n")
        assert seen.wait(2)
        j.note("door", "reply", actor="operator", trace=7, row="writeWave")
    finally:
        router.stop()
        a.close()
        b.close()
    rows = [e for e in _lines(j) if e["src"] != "journal"]
    print("\n".join(json.dumps(e)[:230] for e in rows))
    assert [(e["src"], e["what"]) for e in rows] == [
        ("door", "request"), ("wire", "send"), ("wire", "send"), ("wire", "read"),
        ("door", "reply")]
    wave = rows[1]["body"]
    assert wave["text"] == "WAVE w1" and wave["wave"]["points"] == 500      # summarised
    assert wave["wave"]["first"] == "0" and wave["wave"]["last"] == "499"
    assert "sha256" in wave["wave"] and rows[2]["body"]["text"] == "GETVALUE u1 Height"
    assert rows[3]["body"]["text"] == "VALUE u1 1.5"                        # TELEM not noted
    assert rows[1]["body"]["thread"] == threading.current_thread().name
    assert rows[3]["body"]["thread"] == "bridge-router"


def _best_cpu_ms(fn, runs: int = 15) -> float:
    """Fastest of several runs, in this thread's CPU time: a loaded machine can steal wall
    time from any one run, but cannot make the cheapest run cost more CPU than the code does."""
    times = []
    for _ in range(runs):
        t0 = time.thread_time()
        fn()
        times.append((time.thread_time() - t0) * 1000)
    return min(times)


@pytest.mark.parametrize("journal_on", [False, True])
def test_a_long_wave_line_costs_under_3_ms(tmp_path, journal_on):
    values = np.linspace(-1.0, 1.0, 87380)
    send_line = P.cmd_wave("w1", values)
    read_line = P.msg_capture("u1", "w1", "Height", values)
    if journal_on:
        Journal(tmp_path).start()
    assert _best_cpu_ms(lambda: P.note_wire("send", send_line)) < 3.0
    assert _best_cpu_ms(lambda: P.note_wire("read", read_line)) < 3.0
