"""KR2 of the keeper revival: the service revives a dead keeper, which adopts the running
children (docs/migration/plan/phase6-board/keeper-revival.md).

Real processes, locks and sockets, real time: every test is time-sensitive.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import ClassVar
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from afm_litho.keeper import adopt, lock, revive, status
from afm_litho.keeper.beatfile import BeatFile
from afm_litho.keeper.pids import pid_alive
from afm_litho.keeper.process import AdoptedHandle
from afm_litho.keeper.supervisor import EXITED, ChildSpec, Supervisor
from afm_litho.keeper.protocol import heartbeat_line, ready_line
from afm_litho.service import keeper_link

pytestmark = [pytest.mark.time_sensitive, pytest.mark.xdist_group("stop-port")]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_until(cond, timeout_s: float = 30.0) -> None:
    end = time.monotonic() + timeout_s
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.05)


def sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])


def kill_tree(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":     # under a venv Popen's pid is the launcher stub
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, check=False)
    else:
        proc.kill()
    proc.wait()


# ------------------------------------------------------------- the reviver ----

class FakeSpawn:
    """Stands in for the new keeper: after `delay_s` it takes the keeper lock, as a real
    one does, and holds it until `release()`."""

    def __init__(self, runtime, delay_s: float = 0.2):
        self.runtime, self.delay_s = runtime, delay_s
        self.count = 0
        self.held: list = []
        self._guard = threading.Lock()

    def __call__(self, runtime, by_pid):
        with self._guard:
            self.count += 1

        def become_keeper():
            time.sleep(self.delay_s)
            for _ in range(50):         # the reviver's own probe may hold the lock for a moment
                h = lock.acquire(self.runtime)
                if h is not None:
                    self.held.append(h)
                    return
                time.sleep(0.1)
        threading.Thread(target=become_keeper, daemon=True).start()
        return type("P", (), {"pid": 1, "poll": staticmethod(lambda: None)})()

    def release(self) -> None:
        while self.held:
            self.held.pop().release()


def test_a_dead_keeper_is_revived_once(tmp_path):
    spawn = FakeSpawn(tmp_path)
    outcome, _ = revive.Reviver(tmp_path, spawn).revive()
    spawn.release()
    assert outcome == revive.REVIVED and spawn.count == 1


def test_a_clean_shutdown_is_not_revived(tmp_path):
    lock.write_clean_marker(tmp_path)
    spawn = FakeSpawn(tmp_path)
    outcome, _ = revive.Reviver(tmp_path, spawn).revive()
    assert outcome == revive.CLEAN and spawn.count == 0


def test_a_live_keeper_is_never_revived(tmp_path):
    keeper = lock.acquire(tmp_path)
    spawn = FakeSpawn(tmp_path)
    try:
        outcome, _ = revive.Reviver(tmp_path, spawn).revive()
    finally:
        keeper.release()
    assert outcome == revive.ALIVE and spawn.count == 0


def test_two_revivers_at_once_give_exactly_one_keeper(tmp_path):
    spawn = FakeSpawn(tmp_path, delay_s=0.5)
    results: list[str] = []

    def go():
        results.append(revive.Reviver(tmp_path, spawn).revive()[0])

    threads = [threading.Thread(target=go) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    spawn.release()
    assert spawn.count == 1 and len(spawn.held) == 0
    assert sorted(results) in (sorted([revive.REVIVED, revive.BUSY]),
                               sorted([revive.REVIVED, revive.ALIVE]))


def test_a_reviver_that_comes_late_finds_the_new_keeper_alive(tmp_path):
    spawn = FakeSpawn(tmp_path)
    assert revive.Reviver(tmp_path, spawn).revive()[0] == revive.REVIVED
    assert revive.Reviver(tmp_path, spawn).revive()[0] == revive.ALIVE
    spawn.release()
    assert spawn.count == 1


def test_the_revival_limit(tmp_path):
    spawn = FakeSpawn(tmp_path, delay_s=0.0)
    reviver = revive.Reviver(tmp_path, spawn)
    for _ in range(revive.MAX_REVIVALS):
        assert reviver.revive()[0] == revive.REVIVED
        spawn.release()                     # the new keeper dies again
    outcome, words = reviver.revive()
    assert outcome == revive.LIMIT and "NOT reviving" in words
    assert spawn.count == revive.MAX_REVIVALS


def test_a_new_keeper_that_exits_at_once_is_a_failure(tmp_path):
    class Dead:
        pid = 1

        @staticmethod
        def poll():
            return 5

    outcome, _ = revive.Reviver(tmp_path, lambda r, by: Dead()).revive()
    assert outcome == revive.FAILED
    assert lock.acquire(tmp_path) is not None       # the revive lock was let go as well


# ----------------------------------------- the whole chain, with real keepers ----

class ServiceStub(BaseHTTPRequestHandler):
    """The service's contact route: what the real route does with the keeper's pid."""

    contacts: ClassVar[list[int]] = []

    def do_POST(self):
        from afm_litho.service.routers import instrument
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        answer = instrument.post_keeper_contact(body)
        if answer.get("ok"):
            ServiceStub.contacts.append(body["pid"])
        data = json.dumps(answer).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_killing_the_keeper_gives_one_new_keeper_that_adopts_the_children(
        tmp_path, monkeypatch, capfd):
    monkeypatch.setenv("AFM_LITHO_RUNTIME", str(tmp_path))
    keeper_port, host_port, service_port = free_port(), free_port(), free_port()
    monkeypatch.setenv("AFM_LITHO_KEEPER_PORT", str(keeper_port))
    monkeypatch.setenv("AFM_LITHO_STOP_PORT", str(host_port))      # nothing listens: unreachable
    procs: list[subprocess.Popen] = []
    old_keeper = sleeper()
    service, recorder = sleeper(), sleeper()
    procs += [old_keeper, service, recorder]
    ServiceStub.contacts = []
    server = HTTPServer(("127.0.0.1", service_port), ServiceStub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(keeper_link, "_link", object())         # a service a keeper started
    monkeypatch.setattr(keeper_link, "_reviver",
                        revive.Reviver(tmp_path, lambda rt, by: spawn_without_a_console(rt, by)))
    adopted: list[subprocess.Popen] = []

    def spawn_without_a_console(runtime, by_pid):
        proc = subprocess.Popen([sys.executable, "-m", "afm_litho.keeper", revive.ADOPT_ARG,
                                 revive.REVIVED_BY_ARG, str(by_pid)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                stdin=subprocess.DEVNULL)
        adopted.append(proc)
        return proc

    try:
        # What the old keeper left behind: children.json, beat files, and the lock gone with it.
        env = {"AFM_LITHO_RECORDER_PORT": "1", "AFM_LITHO_KEEPER_PORT": str(keeper_port)}
        children = []
        for name, proc, argv in (
                ("recorder", recorder, ["rec"]),
                ("service", service, ["svc", "--port", str(service_port), "--host", "127.0.0.1"])):
            beats = BeatFile(tmp_path / "keeper" / f"beat-{name}.txt")
            beats.write(ready_line(0, 0, proc.pid, "up" if name == "service" else None) + "\n" + heartbeat_line(proc.pid) + "\n")
            beats.flush()
            # the service's own pid is the test process, the one that revives
            launcher = os.getpid() if name == "service" else proc.pid
            children.append({"name": name, "pid": launcher, "state": "up", "restarts_total": 0,
                             "bridge": "up" if name == "service" else None,
                             "ready": {"mark": 0, "fault": 0, "pid": proc.pid},
                             "launch": {"argv": argv, "env": env, "essential": name == "service",
                                        "restart_args": []}})
        status.write_state(status.state_path(), {"children": children})
        state = status.read_state(status.state_path())
        state["keeper_pid"] = old_keeper.pid
        status.state_path().write_text(json.dumps(state))

        keeper_link.watch_keeper(old_keeper.pid)
        assert keeper_link.keeper_lost() is None
        old_keeper.kill()
        old_keeper.wait()
        assert keeper_link.keeper_lost() is not None                 # writes refused
        wait_until(lambda: (status.read_state(status.state_path()) or {}).get("keeper_pid")
                   not in (None, old_keeper.pid))
        wait_until(lambda: keeper_link.keeper_lost() is None)        # contact: writes allowed
        new_state = status.read_state(status.state_path())
        assert len(adopted) == 1 and ServiceStub.contacts == [new_state["keeper_pid"]]
        assert pid_alive(new_state["keeper_pid"])
        assert lock.acquire(tmp_path) is None                         # the new keeper holds it
        assert [(c["name"], c["pid"], c["restarts_total"]) for c in new_state["children"]] == [
            ("recorder", recorder.pid, 0), ("service", service.pid, 0)]
        assert new_state["children"][1]["bridge"] == "up"        # a restart still makes safe
        assert service.poll() is None and recorder.poll() is None    # nothing was restarted

        # The revived keeper serves Stop on the same port (the host is down: it says so).
        with socket.create_connection(("127.0.0.1", keeper_port), timeout=5) as s:
            s.sendall(b"STOP kr2-test after revival\n")
            s.settimeout(10)
            reply = s.makefile("r").readline().strip()
        assert reply.startswith("NOT confirmed")
    finally:
        server.shutdown()
        for proc in adopted + procs:
            kill_tree(proc)
        monkeypatch.undo()
        keeper_link.forget_keeper()         # the watch was left on the revived keeper just killed
        capfd.readouterr()


def test_a_hung_keeper_is_not_revived(tmp_path, monkeypatch):
    """Alive but silent: the OS handle says it is there, so the service starts nothing."""
    monkeypatch.setenv("AFM_LITHO_RUNTIME", str(tmp_path))
    monkeypatch.setattr(keeper_link, "_link", object())
    started: list = []
    monkeypatch.setattr(keeper_link, "_revive", lambda hub: started.append(1))
    hung = sleeper()
    try:
        keeper_link.watch_keeper(hung.pid)
        time.sleep(0.3)
        assert keeper_link.keeper_lost() is None and started == []
    finally:
        kill_tree(hung)
        keeper_link.forget_keeper()


def test_a_dead_pid_the_state_file_does_not_name_is_not_revived(tmp_path, monkeypatch):
    monkeypatch.setenv("AFM_LITHO_RUNTIME", str(tmp_path))
    monkeypatch.setattr(keeper_link, "_link", object())
    started: list = []
    monkeypatch.setattr(keeper_link, "_revive", lambda hub: started.append(1))
    stranger = sleeper()
    keeper_link.watch_keeper(stranger.pid)
    kill_tree(stranger)
    try:
        assert keeper_link.keeper_lost() is not None
        time.sleep(0.2)
        assert started == []
    finally:
        keeper_link.forget_keeper()


def test_the_revive_path_never_talks_to_igor():
    folder = Path(revive.__file__).parent
    for name in ("revive.py", "adopt.py"):
        text = (folder / name).read_text(encoding="utf-8")
        assert "win32com" not in text and "td_" not in text and "pythoncom" not in text


# ------------------------------------------------------ the review's fixes ----

class RestartedChild:
    """A stand-in for the child a restart launches."""

    pid = 1

    @staticmethod
    def poll():
        return None

    @staticmethod
    def drain():
        return []


def _adopted_supervisor(tmp_path, monkeypatch, code: int, bridge=None):
    monkeypatch.setenv("AFM_LITHO_RUNTIME", str(tmp_path))
    child = subprocess.Popen([sys.executable, "-c", f"raise SystemExit({code})"])
    child.wait()
    launched: list = []
    keeper = Supervisor([ChildSpec("service", ["x"], restart_args=("--after-death",))],
                        lambda spec, extra: launched.append(extra) or RestartedChild(),
                        out=lambda m: None, err=lambda m: None)
    keeper.adopt({"service": AdoptedHandle("service", child.pid)}, {"service": bridge})
    return keeper, launched


def test_an_adopted_child_that_quit_cleanly_is_not_restarted(tmp_path, monkeypatch):
    keeper, launched = _adopted_supervisor(tmp_path, monkeypatch, 0)
    keeper.step()
    assert launched == [] and keeper.slots[0].state == EXITED and keeper.finished


def test_an_adopted_child_that_crashed_is_restarted_with_its_last_bridge(tmp_path, monkeypatch):
    keeper, launched = _adopted_supervisor(tmp_path, monkeypatch, 7, bridge="up")
    keeper.step()
    assert launched == [("--after-death", "--last-bridge", "up")]


def test_a_revived_keeper_started_by_a_stranger_does_not_adopt(tmp_path):
    state = {"children": [{"name": "service", "pid": 4242, "ready": {"pid": 4243}}]}
    assert adopt.reviver_is_a_child(state, 4243) and adopt.reviver_is_a_child(state, 4242)
    assert not adopt.reviver_is_a_child(state, os.getpid())
    status.write_state(tmp_path / "keeper" / "children.json", {"children": [
        {"name": "service", "pid": 4242, "state": "up", "ready": None,
         "launch": {"argv": ["x"], "env": {}, "essential": True, "restart_args": []}}]})
    env = {**os.environ, "AFM_LITHO_RUNTIME": str(tmp_path)}
    done = subprocess.run([sys.executable, "-m", "afm_litho.keeper", "--adopt", "--revived-by",
                           str(os.getpid())], env=env, capture_output=True, text=True,
                          timeout=60)
    assert done.returncode == 1 and "Not adopting" in done.stderr
    assert lock.acquire(tmp_path) is not None


def test_a_keeper_that_cannot_reach_the_service_says_so_loudly(monkeypatch):
    monkeypatch.setattr(adopt, "CONTACT_ALERT_AFTER", 2)
    said, reported = [], []
    stop = threading.Event()
    adopt.keep_contact(free_port(), stop, said.append, reported.append)
    wait_until(lambda: reported, 30)
    stop.set()
    assert "SAFETY" in said[0] and "writes stay refused" in reported[0]


def test_a_revival_is_reported_only_when_contact_comes(monkeypatch):
    said: list[str] = []
    monkeypatch.setattr(keeper_link, "CONTACT_WAIT_S", 1.0)
    monkeypatch.setattr(keeper_link, "_report", lambda hub, msg: said.append(msg))
    monkeypatch.setattr(keeper_link, "_keeper_lost", "lost")
    keeper_link._await_contact(None, "a new keeper holds the lock")
    assert said[-1].startswith("SAFETY") and "no contact" in said[-1]
    monkeypatch.setattr(keeper_link, "_keeper_lost", None)
    keeper_link._await_contact(None, "a new keeper holds the lock")
    assert "in contact" in said[-1] and not said[-1].startswith("SAFETY")
