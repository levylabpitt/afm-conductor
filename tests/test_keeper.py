"""The keeper: restart, the restart cap, READY and heartbeat lines, and `status`.

The supervisor is driven one step at a time on the testing clock, with scripted
children, so no test waits for real time or starts a process.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from afm_litho import clock
from afm_litho.keeper import status, supervisor
from afm_litho.keeper.pids import pid_alive
from afm_litho.keeper.protocol import Beat, Ready, heartbeat_line, parse_line, ready_line
from afm_litho.keeper.supervisor import ChildSpec, Supervisor


class FakeChild:
    """A child the test controls: it prints what it is given and exits when told."""

    def __init__(self, pid):
        self.pid = pid
        self.code = None
        self.lines: list[str] = []
        self.terminated = False

    def poll(self):
        return self.code

    def drain(self):
        out, self.lines = self.lines, []
        return out

    def terminate(self):
        self.terminated = True
        self.code = 1


class FakeLauncher:
    def __init__(self):
        self.children: list[FakeChild] = []
        self.extra_args: list[tuple] = []

    def __call__(self, spec, extra):
        child = FakeChild(1000 + len(self.children))
        self.children.append(child)
        self.extra_args.append(extra)
        return child

    @property
    def latest(self):
        return self.children[-1]


@pytest.fixture
def testing_clock():
    with clock.using(clock.TESTING):
        yield


def make(testing_clock, **kw):
    launcher = FakeLauncher()
    said, shouted = [], []
    spec = ChildSpec("service", ["python", "-m", "x"], restart_args=("--restarted",))
    keeper = Supervisor([spec], launcher, out=said.append, err=shouted.append, **kw)
    keeper.start()
    return keeper, launcher, said, shouted


# ------------------------------------------------------------ restart ----

def test_a_crashed_child_is_restarted_with_the_restart_arguments(testing_clock):
    keeper, launcher, said, _ = make(testing_clock)
    assert launcher.extra_args == [()]               # the first start has none
    launcher.latest.code = 1
    keeper.step()
    assert len(launcher.children) == 2
    assert launcher.extra_args[1] == ("--restarted",)
    assert keeper.slots[0].state == supervisor.STARTING
    assert any("restart 1 of 3" in m for m in said)
    assert not keeper.finished


def test_a_clean_exit_stops_the_keeper_without_a_restart(testing_clock):
    keeper, launcher, _, _ = make(testing_clock)
    launcher.latest.code = 0
    keeper.step()
    assert len(launcher.children) == 1
    assert keeper.finished and keeper.exit_code == 0


# --------------------------------------------------------- restart cap ----

def test_the_fourth_death_inside_two_minutes_is_a_loud_give_up(testing_clock):
    keeper, launcher, _, shouted = make(testing_clock)
    for _ in range(3):
        launcher.latest.code = 1
        keeper.step()
        clock.advance(10)
    assert len(launcher.children) == 4 and not keeper.finished   # 3 restarts allowed
    launcher.latest.code = 1
    keeper.step()
    assert len(launcher.children) == 4                           # no fourth restart
    assert keeper.finished and keeper.exit_code == supervisor.EXIT_GAVE_UP
    assert keeper.slots[0].state == supervisor.GAVE_UP
    assert "GIVING UP" in shouted[0]


def test_deaths_more_than_two_minutes_apart_never_reach_the_cap(testing_clock):
    keeper, launcher, _, _ = make(testing_clock)
    for _ in range(6):
        launcher.latest.code = 1
        keeper.step()
        clock.advance(supervisor.RESTART_WINDOW_S + 1)
    assert len(launcher.children) == 7 and not keeper.finished


def test_run_returns_the_give_up_code(testing_clock):
    """The real loop, on virtual time: every child dies the moment it starts."""
    class Dying(FakeLauncher):
        def __call__(self, spec, extra):
            child = super().__call__(spec, extra)
            child.code = 1
            return child

    shouted = []
    spec = ChildSpec("service", ["x"])
    keeper = Supervisor([spec], Dying(), out=lambda m: None, err=shouted.append)
    assert keeper.run() == supervisor.EXIT_GAVE_UP
    assert shouted and "GIVING UP" in shouted[0]


# --------------------------------------------------------------- READY ----

def test_ready_and_heartbeat_lines_parse():
    assert parse_line("READY mark=0 fault=1 pid=77") == Ready(0, 1, 77)
    assert parse_line("READY mark=? fault=0 pid=5") == Ready(None, 0, 5)
    assert parse_line("HEARTBEAT pid=77") == Beat(77)
    for junk in ("", "hello", "READY", "READY mark=x fault=0 pid=1", "READY mark=0 pid=1",
                 "HEARTBEAT"):
        assert parse_line(junk) is None


def test_the_lines_the_service_writes_are_the_lines_the_keeper_reads():
    assert parse_line(ready_line(1, 0, 9)) == Ready(1, 0, 9)
    assert parse_line(ready_line(None, 1, 9)) == Ready(None, 1, 9)
    assert parse_line(heartbeat_line(9)) == Beat(9)


def test_ready_marks_the_child_up_and_other_lines_are_echoed(testing_clock):
    keeper, launcher, said, _ = make(testing_clock)
    launcher.latest.lines = ["booting", "READY mark=0 fault=0 pid=1000"]
    keeper.step()
    slot = keeper.slots[0]
    assert slot.state == supervisor.UP and slot.ready == Ready(0, 0, 1000)
    assert "booting" in said and any("READY mark=0 fault=0" in m for m in said)


# ----------------------------------------------------------- heartbeat ----

def test_a_missed_heartbeat_is_reported_once_and_the_child_is_not_killed(testing_clock):
    keeper, launcher, said, _ = make(testing_clock)
    launcher.latest.lines = [ready_line(0, 0, 1000)]
    keeper.step()
    clock.advance(supervisor.BEAT_TIMEOUT_S - 1)
    keeper.step()
    assert keeper.slots[0].state == supervisor.UP                # still inside the limit
    clock.advance(2)
    keeper.step()
    keeper.step()
    assert keeper.slots[0].state == supervisor.QUIET
    assert sum("no heartbeat" in m for m in said) == 1           # reported once
    assert not launcher.latest.terminated and len(launcher.children) == 1

    launcher.latest.lines = [heartbeat_line(1000)]
    keeper.step()
    assert keeper.slots[0].state == supervisor.UP
    assert any("heartbeat is back" in m for m in said)


def test_a_child_that_never_says_ready_is_reported_after_the_startup_allowance(testing_clock):
    keeper, launcher, said, _ = make(testing_clock)
    clock.advance(supervisor.BEAT_TIMEOUT_S + 1)
    keeper.step()
    assert keeper.slots[0].state == supervisor.STARTING          # imports are slow
    clock.advance(supervisor.READY_TIMEOUT_S)
    keeper.step()
    assert keeper.slots[0].state == supervisor.QUIET
    assert any("no READY" in m for m in said)


# ------------------------------------------------------------ status ----

def test_status_lists_the_children_and_their_pids(testing_clock, tmp_path):
    path = tmp_path / "keeper" / "children.json"
    keeper, launcher, _, _ = make(testing_clock, state_path=path)
    launcher.latest.lines = [ready_line(None, 0, 1000), heartbeat_line(1000)]
    keeper.step()
    keeper._publish(force=True)

    state = status.read_state(path)
    assert state["keeper_pid"] == os.getpid()
    text = "\n".join(status.render(state, alive=lambda pid: pid == os.getpid() or pid == 1000))
    assert f"keeper   pid {os.getpid()}  running" in text
    assert "service  pid 1000  up" in text
    assert "READY mark=? fault=0" in text and "heartbeat" in text and "restarts 0" in text


def test_status_says_so_when_a_child_outlived_its_keeper_or_died():
    state = {"keeper_pid": 11, "children": [
        {"name": "service", "pid": 22, "state": "up", "ready": None,
         "last_beat_wall": None, "restarts_total": 2}]}
    text = "\n".join(status.render(state, alive=lambda pid: pid == 22))
    assert "keeper   pid 11  NOT RUNNING" in text
    assert "service  pid 22  up" in text
    text = "\n".join(status.render(state, alive=lambda pid: False))
    assert "service  pid 22  NOT RUNNING (last state: up)" in text


def test_status_without_a_state_file(tmp_path):
    assert status.read_state(tmp_path / "none.json") is None
    assert "no keeper has run here" in status.render(None)[0]


def test_pid_alive_tells_a_running_process_from_a_finished_one():
    assert pid_alive(os.getpid())
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    assert not pid_alive(done.pid)


# -------------------------------------------- the service's side of it ----

def test_the_service_sends_ready_then_a_heartbeat_and_stays_quiet_without_a_link(monkeypatch):
    import asyncio
    import io

    from afm_litho.service import keeper_link

    class Hub:
        fault = None

        @property
        def host(self):                # fake: host fields live on the fake itself
            return self

    async def go():
        keeper_link.start(Hub())
        await asyncio.sleep(0.05)
        keeper_link.stop()

    monkeypatch.setattr(keeper_link, "_link", None)
    asyncio.run(go())                                    # no link: nothing happens

    pipe = io.StringIO()
    monkeypatch.setattr(keeper_link, "_link", pipe)
    asyncio.run(go())
    lines = pipe.getvalue().splitlines()
    assert parse_line(lines[0]) == Ready(None, 0, os.getpid())
    assert parse_line(lines[1]) == Beat(os.getpid())


def test_an_unwritable_beat_file_does_not_break_the_service(monkeypatch, tmp_path):
    from afm_litho.keeper.beatfile import BeatFile
    from afm_litho.service import keeper_link

    blocker = tmp_path / "not-a-folder"
    blocker.write_text("x")
    link = BeatFile(blocker / "beat.txt")                # its folder cannot be made
    monkeypatch.setattr(keeper_link, "_link", link)
    keeper_link._send("HEARTBEAT pid=1")                 # must not raise
    keeper_link._outbox.put_nowait(None)
    import threading
    writer = threading.Thread(target=keeper_link._writer, args=(link,))
    writer.start()
    writer.join(timeout=2)
    assert not writer.is_alive() and keeper_link._link is link


# ------------------------------------------------- shutdown, port taken ----

def test_shutdown_waits_for_the_child_to_exit_and_never_ends_it(testing_clock):
    keeper, launcher, _, _ = make(testing_clock)
    child = launcher.latest
    polls = {"n": 0}

    def poll():
        polls["n"] += 1
        return None if polls["n"] < 500 else 0       # far longer than the old 10 s grace

    child.poll = poll
    keeper.shutdown()
    assert not child.terminated and polls["n"] >= 500


def test_a_second_ctrl_c_ends_the_child(testing_clock):
    keeper, launcher, _, _ = make(testing_clock)
    child = launcher.latest
    calls = {"n": 0}

    def poll():
        calls["n"] += 1
        if calls["n"] == 5:
            raise KeyboardInterrupt
        return child.code

    child.poll = poll
    keeper.shutdown()
    assert child.terminated


def test_a_child_that_found_its_port_taken_is_not_restarted(testing_clock):
    keeper, launcher, said, _ = make(testing_clock)
    launcher.latest.code = supervisor.PORT_TAKEN_EXIT
    keeper.step()
    assert len(launcher.children) == 1 and keeper.finished
    assert keeper.exit_code == supervisor.PORT_TAKEN_EXIT
    assert sum("port is taken" in m for m in said) == 1


def test_the_default_port_is_checked_too(monkeypatch, capsys):
    import socket

    import afm_litho.keeper.cli as cli
    from afm_litho import site

    held = socket.socket()
    held.bind(("127.0.0.1", 0))
    held.listen(1)
    port = held.getsockname()[1]
    monkeypatch.setattr(site, "tls_possible", lambda: False)
    monkeypatch.setattr(site, "choose_port", lambda tls: port)     # the default path
    monkeypatch.setattr(site, "resolve_site", lambda p: {"url": "http://x/", "reasons": []})
    launcher = FakeLauncher()
    try:
        assert cli.main(["--no-browser"], launch=launcher) == supervisor.PORT_TAKEN_EXIT
    finally:
        held.close()
    assert launcher.children == [] and "already in use" in capsys.readouterr().err


# ------------------------------------- a keeper that is not reading ----

def test_a_keeper_that_never_reads_cannot_freeze_the_service(monkeypatch):
    """The pipe fills, the writer thread blocks in write(), and the event loop (where
    Stop and every door run) keeps answering."""
    import asyncio
    import threading
    import time

    from afm_litho.service import keeper_link

    release = threading.Event()

    class NeverRead:
        def write(self, _):
            release.wait()

        def flush(self):
            pass

    class Hub:
        fault = None

        @property
        def host(self):                # fake: host fields live on the fake itself
            return self

    monkeypatch.setattr(keeper_link, "_link", NeverRead())

    async def go():
        keeper_link.start(Hub())
        t0 = time.monotonic()
        for _ in range(5000):                # far more lines than the outbox holds
            keeper_link._send("HEARTBEAT pid=1")
        done = asyncio.Event()
        asyncio.get_running_loop().call_soon(done.set)
        await asyncio.wait_for(done.wait(), 1.0)
        elapsed = time.monotonic() - t0
        keeper_link.stop()
        return elapsed

    try:
        assert asyncio.run(go()) < 1.0
    finally:
        release.set()
