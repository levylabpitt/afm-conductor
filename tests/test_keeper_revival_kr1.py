"""KR1 of the keeper revival: the heartbeat off the pipe, one keeper at a time, and the
clean-shutdown marker (docs/migration/plan/phase6-board/keeper-revival.md)."""

from __future__ import annotations

import socket
import subprocess
import sys
import time

import pytest

from afm_litho import clock
from afm_litho.keeper import cli, lock, supervisor
from afm_litho.keeper.beatfile import BeatFile, read_lines
from afm_litho.keeper.process import PopenHandle
from afm_litho.keeper.protocol import Beat, Ready, heartbeat_line, parse_line, ready_line
from afm_litho.keeper.supervisor import ChildSpec, Supervisor


# ---------------------------------------------------------------- beat file ----

def test_a_beat_file_carries_the_latest_ready_and_a_new_line_for_every_beat(tmp_path):
    path = tmp_path / "keeper" / "beat-x.txt"
    beats = BeatFile(path)
    beats.write(ready_line(1, 0, 42, "up") + "\n")
    beats.write(heartbeat_line(42, "up") + "\n")
    beats.flush()
    first = read_lines(path)
    assert parse_line(first[0]) == Ready(1, 0, 42, "up")
    assert parse_line(first[1]) == Beat(42, "up")
    beats.write(heartbeat_line(42, "up") + "\n")
    beats.flush()
    second = read_lines(path)
    assert second[0] == first[0] and second[1] != first[1]


def test_the_keeper_reads_a_child_heartbeat_without_any_pipe(tmp_path, monkeypatch):
    """The child's stdout is not the channel; the beat file is."""
    monkeypatch.setenv("AFM_LITHO_RUNTIME", str(tmp_path))
    code = ("from afm_litho.keeper.beatfile import from_env\n"
            "b = from_env()\n"
            "b.write('READY mark=0 fault=0 pid=7\\nHEARTBEAT pid=7\\n')\n"
            "b.flush()\n"
            "print('noise on stdout')\n")
    child = PopenHandle(ChildSpec("kr1", [sys.executable, "-c", code]))
    end = time.monotonic() + 30
    got: list[str] = []
    while time.monotonic() < end and len(got) < 2:
        got += child.drain()
        time.sleep(0.05)
    assert isinstance(parse_line(got[0]), Ready) and isinstance(parse_line(got[1]), Beat)
    assert child.drain() == []                       # unchanged lines are not read twice


def test_a_new_child_does_not_inherit_the_last_runs_beat(tmp_path, monkeypatch):
    monkeypatch.setenv("AFM_LITHO_RUNTIME", str(tmp_path))
    stale = tmp_path / "keeper" / "beat-kr1.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("READY mark=0 fault=0 pid=1\nHEARTBEAT pid=1 n=9\n")
    child = PopenHandle(ChildSpec("kr1", [sys.executable, "-c", "import time; time.sleep(1)"]))
    assert child.drain() == []                       # the old run's beat was removed at launch


# --------------------------------------------------------------------- lock ----

def test_the_lock_file_stops_a_second_keeper(tmp_path):
    first = lock.acquire(tmp_path)
    assert first is not None
    assert lock.acquire(tmp_path) is None
    first.release()
    again = lock.acquire(tmp_path)
    assert again is not None
    again.release()


def test_the_lock_is_released_when_the_holder_dies(tmp_path):
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, time; from pathlib import Path; from afm_litho.keeper import lock\n"
         "h = lock.acquire(Path(sys.argv[1])); print('held', flush=True); time.sleep(120)",
         str(tmp_path)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        assert lock.acquire(tmp_path) is None
    finally:
        if sys.platform == "win32":     # under a venv Popen's pid is the launcher stub
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(holder.pid)],
                           capture_output=True, check=False)
        else:
            holder.kill()
        holder.wait()
        holder.stdout.close()
    again = lock.acquire(tmp_path)
    assert again is not None
    again.release()


def test_a_second_keeper_exits_without_starting_anything(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AFM_LITHO_RUNTIME", str(tmp_path))
    held = lock.acquire(tmp_path)
    launched = []
    try:
        code = cli.main(["--no-browser"], launch=lambda *a: launched.append(a))
    finally:
        held.release()
    assert code == lock.ANOTHER_KEEPER_EXIT and launched == []
    assert "Another keeper is already running" in capsys.readouterr().err


# ------------------------------------------------------------------- marker ----

class _Child:
    pid = 1

    def __init__(self):
        self.code = None

    def poll(self):
        return self.code

    def drain(self):
        return []

    def terminate(self):
        self.code = 1


@pytest.fixture
def testing_clock():
    with clock.using(clock.TESTING):
        yield


def _keeper(runtime, first_code=None):
    children: list[_Child] = []

    def launch(spec, extra):
        children.append(_Child())
        children[-1].code = first_code if len(children) == 1 else None
        return children[-1]

    keeper = Supervisor([ChildSpec("service", ["x"])], launch, out=lambda m: None,
                        err=lambda m: None,
                        on_clean_exit=lambda: lock.write_clean_marker(runtime))
    return keeper, children


def test_a_clean_shutdown_writes_the_marker(tmp_path, testing_clock):
    keeper, _ = _keeper(tmp_path, first_code=0)       # the service quits from the cockpit
    assert keeper.run() == 0
    assert lock.clean_shutdown_marked(tmp_path)


def test_ctrl_c_writes_the_marker_before_waiting_for_the_children(tmp_path, testing_clock):
    keeper, _ = _keeper(tmp_path)
    marked_at_shutdown = []
    keeper.shutdown = lambda: marked_at_shutdown.append(lock.clean_shutdown_marked(tmp_path))

    def interrupt():
        raise KeyboardInterrupt

    keeper.step = interrupt
    keeper.run()
    assert marked_at_shutdown == [True]


def test_giving_up_leaves_no_marker(tmp_path, testing_clock):
    keeper, children = _keeper(tmp_path)
    keeper.start()
    for _ in range(supervisor.MAX_RESTARTS + 1):
        children[-1].code = 1
        keeper.step()
    assert keeper.exit_code == supervisor.EXIT_GAVE_UP
    assert keeper.finished
    assert not lock.clean_shutdown_marked(tmp_path)


def test_a_starting_keeper_clears_the_old_marker_and_lets_go_of_the_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("AFM_LITHO_RUNTIME", str(tmp_path))
    lock.write_clean_marker(tmp_path)
    with socket.socket() as s:                        # a taken port ends the run at once
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        assert cli.main(["--no-browser", "--port", str(port)]) == 3
    assert not lock.clean_shutdown_marked(tmp_path)
    again = lock.acquire(tmp_path)
    assert again is not None
    again.release()


def test_a_failed_read_of_the_beat_file_does_not_replay_old_lines(tmp_path, monkeypatch):
    monkeypatch.setenv("AFM_LITHO_RUNTIME", str(tmp_path))
    code = ("from afm_litho.keeper.beatfile import from_env; import time; b = from_env(); "
            "b.write('READY mark=0 fault=0 pid=7'); b.write('HEARTBEAT pid=7'); b.flush(); "
            "time.sleep(2)")
    child = PopenHandle(ChildSpec("kr1", [sys.executable, "-c", code]))
    end = time.monotonic() + 30
    got: list[str] = []
    while time.monotonic() < end and len(got) < 2:
        got += child.drain()
        time.sleep(0.05)
    assert len(got) == 2
    beats = tmp_path / "keeper" / "beat-kr1.txt"
    saved = beats.read_text()
    beats.unlink()                                   # unreadable for one tick
    assert child.drain() == []
    beats.write_text(saved)                          # readable again, unchanged
    assert child.drain() == []


def test_the_bridge_word_of_a_child_that_dies_at_once_reaches_the_restart(testing_clock):
    seen_extra = []
    children: list[_Child] = []

    class Dying(_Child):
        """Its first drain is empty. Then it writes bridge=up and exits, so only the drain
        after the exit can carry the word."""

        def __init__(self, first):
            super().__init__()
            self.first, self.drained = first, False

        def poll(self):
            return 1 if self.first and self.drained else None

        def drain(self):
            if self.first and self.drained:
                return ["HEARTBEAT pid=9 bridge=up"]
            self.drained = True
            return []

    def launch(spec, extra):
        seen_extra.append(extra)
        children.append(Dying(len(children) == 0))
        return children[-1]

    spec = ChildSpec("service", ["x"], restart_args=("--after-death",))
    keeper = Supervisor([spec], launch, out=lambda m: None, err=lambda m: None)
    keeper.start()
    keeper.step()
    assert seen_extra[1] == ("--after-death", "--last-bridge", "up")


def test_a_failed_wait_counts_as_exited(monkeypatch):
    if sys.platform != "win32":
        pytest.skip("process handles are Windows-only")
    import ctypes

    from afm_litho.keeper.pids import ProcessHandle
    handle = ProcessHandle(1)
    handle._handle = 12345                           # a pretend handle
    kernel32 = ctypes.windll.kernel32
    monkeypatch.setattr(kernel32, "WaitForSingleObject", lambda *a: 0xFFFFFFFF)  # WAIT_FAILED
    assert handle.exited()
    monkeypatch.setattr(kernel32, "WaitForSingleObject", lambda *a: 258)         # WAIT_TIMEOUT
    assert not handle.exited()
    handle._handle = None
