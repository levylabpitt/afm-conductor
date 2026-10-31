"""K4: the keeper's Stop listener, its notes, and the doors that also send their Stop to it.

The keeper passes `STOP <who> <why>` to the host's stop port and relays `sent` and the
outcome; only then it writes an fsynced note. The cockpit Abort and FLEX abort/safePark send
to the keeper from a thread, beside their own route, which runs as before.

Real sockets and real time (these tests check timing and order), so every test is
time-sensitive.
"""

from __future__ import annotations

import itertools
import json
import socket
import threading
import time

import pytest
from golden.harness import HERE, GoldenRig

from afm_litho.keeper import stop_listener
from afm_litho.keeper.notes import Notes, RecorderFeed
from afm_litho.keeper.stop_listener import CONFIRMED, SENT, StopListener
from afm_litho.service import keeper_stop

pytestmark = [pytest.mark.time_sensitive, pytest.mark.xdist_group("stop-port")]


# ---------------------------------------------------------------- helpers --

# Event order across threads (time.monotonic is too coarse on Windows to order them).
_ORDER = itertools.count()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_until(cond, timeout_s: float = 10.0) -> None:
    end = time.monotonic() + timeout_s
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.02)


class FakeHost:
    """A host stop port: records each line it gets, then answers as `mode` says.
    alive: `sent`, then `confirmed`. slow: accepts and reads, never answers."""

    def __init__(self, mode: str = "alive", outcome: str = CONFIRMED):
        self.mode, self.outcome = mode, outcome
        self.lines: list[tuple[float, str]] = []
        self._srv = socket.create_server(("127.0.0.1", 0))
        self.port = self._srv.getsockname()[1]
        self._conns: list[socket.socket] = []
        self._closed = threading.Event()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self._closed.is_set():
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            self._conns.append(conn)
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(256)
            if not chunk:
                return
            buf += chunk
        self.lines.append((next(_ORDER), buf.split(b"\n")[0].decode()))
        if self.mode == "alive":
            conn.sendall(b"sent\n")
            conn.sendall(self.outcome.encode() + b"\n")
            conn.close()

    def close(self):
        self._closed.set()
        self._srv.close()
        for c in self._conns:
            c.close()


class RecordingNotes:
    """Stands in for Notes: records each note and when it was written."""

    def __init__(self):
        self.notes: list[tuple[float, dict]] = []

    def write_stop(self, **fields):
        from afm_litho.keeper.notes import stop_note
        self.notes.append((next(_ORDER), stop_note(**fields)))


def stop(port: int, line: str = "STOP test why\n", timeout: float = 20.0) -> list[str]:
    """Send one line to a Stop port and read replies until the port closes."""
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
        s.sendall(line.encode())
        buf = b""
        while True:
            chunk = s.recv(256)
            if not chunk:
                break
            buf += chunk
    return buf.decode().splitlines()


@pytest.fixture
def keeper_for():
    """A StopListener on an OS-chosen port in front of the given host port."""
    made = []

    def make(host_port: int, notes=None):
        notes = notes if notes is not None else RecordingNotes()
        sl = StopListener(notes, port=0, host_port=host_port).start()
        made.append(sl)
        return sl, notes

    yield make
    for sl in made:
        sl.close()


# --------------------------------------------- the listener, host alive/slow/dead --

def test_host_alive_the_replies_are_relayed_and_the_note_says_confirmed(keeper_for):
    host = FakeHost("alive")
    try:
        sl, notes = keeper_for(host.port)
        assert stop(sl.port, "STOP key operator pressed S\n") == [SENT, CONFIRMED]
        wait_until(lambda: notes.notes)
    finally:
        host.close()
    assert host.lines[0][1] == "STOP key operator pressed S (via keeper)"
    body = notes.notes[0][1]["body"]
    assert body["who"] == "key" and body["why"] == "operator pressed S"
    assert body["replies"] == [SENT, CONFIRMED] and body["outcome"] == CONFIRMED
    assert body["confirmed"] is True and body["connected"] is True
    assert body["sent_to"] == f"127.0.0.1:{host.port}"


def test_the_hosts_not_confirmed_is_relayed_as_it_came(keeper_for):
    host = FakeHost("alive", outcome="NOT confirmed")
    try:
        sl, notes = keeper_for(host.port)
        assert stop(sl.port) == [SENT, "NOT confirmed"]
        wait_until(lambda: notes.notes)
    finally:
        host.close()
    assert notes.notes[0][1]["body"]["confirmed"] is False


def test_host_slow_the_keeper_says_so_within_a_second_and_still_writes_the_note(keeper_for):
    host = FakeHost("slow")
    try:
        sl, notes = keeper_for(host.port)
        t0 = time.monotonic()
        replies = stop(sl.port)
        took = time.monotonic() - t0
        wait_until(lambda: notes.notes)
    finally:
        host.close()
    assert replies == ["NOT confirmed (sent, no answer from the host)"]
    assert took < stop_listener.SENT_WAIT_S + 1.0
    assert host.lines, "the line reached the host"
    body = notes.notes[0][1]["body"]
    assert body["connected"] is True and body["replies"] == [] and not body["confirmed"]


def test_host_dead_the_keeper_answers_not_confirmed_and_still_writes_the_note(keeper_for):
    sl, notes = keeper_for(free_port())             # nothing listens there
    replies = stop(sl.port)
    wait_until(lambda: notes.notes)
    assert len(replies) == 1 and replies[0].startswith("NOT confirmed (host unreachable")
    body = notes.notes[0][1]["body"]
    assert body["connected"] is False and body["forwarded_t"] is None
    assert body["outcome"] == replies[0]


def test_the_note_is_written_after_the_stop_reached_the_host(keeper_for):
    host = FakeHost("alive")
    try:
        sl, notes = keeper_for(host.port)
        stop(sl.port)
        wait_until(lambda: notes.notes)
    finally:
        host.close()
    assert host.lines[0][0] < notes.notes[0][0]
    assert notes.notes[0][1]["body"]["outcome"] == CONFIRMED   # the answer is in it


@pytest.mark.parametrize("line", ["\n", "STOP\n", "STOP key\n", "HALT key why\n",
                                  "STOP key " + "x" * 600 + "\n"])
def test_a_malformed_line_is_refused_and_nothing_is_passed_on(keeper_for, line):
    host = FakeHost("alive")
    try:
        sl, notes = keeper_for(host.port)
        replies = stop(sl.port, line)
        time.sleep(0.2)
    finally:
        host.close()
    assert len(replies) == 1 and replies[0].startswith("error ")
    assert host.lines == [] and notes.notes == []


def test_a_long_why_is_cut_to_fit_the_hosts_line(keeper_for):
    host = FakeHost("alive")
    try:
        sl, _ = keeper_for(host.port)
        stop(sl.port, "STOP flex " + "y" * 490 + "\n")
    finally:
        host.close()
    sent = host.lines[0][1]
    assert len(sent) <= 512 and sent.endswith("(via keeper)")


def test_it_binds_to_loopback_only(keeper_for):
    sl, _ = keeper_for(free_port())
    assert sl._srv.getsockname()[0] == "127.0.0.1"


def test_a_taken_port_is_reported_and_the_keeper_goes_on():
    holder = StopListener(RecordingNotes(), port=0, host_port=1).start()
    said = []
    try:
        assert stop_listener.start(RecordingNotes(), port=holder.port, say=said.append) is None
        assert "NOT listening" in said[0]
    finally:
        holder.close()


def test_the_ports_come_from_the_env_with_fixed_defaults(monkeypatch):
    monkeypatch.delenv(stop_listener.PORT_ENV, raising=False)
    monkeypatch.delenv(stop_listener.HOST_PORT_ENV, raising=False)
    assert stop_listener.port_from_env() == 9211
    assert stop_listener.host_port_from_env() == 9210
    from afm_litho.service.host.stop_port import DEFAULT_PORT
    assert stop_listener.DEFAULT_HOST_PORT == DEFAULT_PORT


# ---------------------------------------------------------------- the notes file --

def test_notes_are_appended_one_json_line_each(tmp_path):
    from afm_litho.keeper.stop_listener import Relay
    notes = Notes(tmp_path / "keeper" / "notes.jsonl")
    for who in ("key", "cockpit"):
        r = Relay()
        r.outcome = "NOT confirmed (host unreachable: ConnectionRefusedError)"
        notes.write_stop(who=who, why="w", received_wall=time.time(), mono=1.0,
                         host_port=9210, relay=r)
    lines = [json.loads(x) for x in notes.path.read_text("utf-8").splitlines()]
    assert [n["body"]["who"] for n in lines] == ["key", "cockpit"]
    for n in lines:
        assert n["keeper_notes"] == 1 and n["src"] == "keeper" and n["what"] == "stop"
        assert set(n) == {"keeper_notes", "src", "what", "t", "mono", "body"}
        assert set(n["body"]) == {"who", "why", "sent_to", "connected", "forwarded_t",
                                  "replies", "outcome", "confirmed"}


def test_the_recorder_gets_each_note_as_a_late_entry_once_it_is_up(tmp_path):
    from afm_litho.recorder_main import Recorder
    from afm_litho.keeper.stop_listener import Relay
    port = free_port()
    target = tmp_path / "journal" / "keeper" / "notes.jsonl"
    feed = RecorderFeed(port, target)
    notes = Notes(tmp_path / "notes.jsonl", feed)
    r = Relay()
    r.outcome = CONFIRMED
    notes.write_stop(who="key", why="before", received_wall=time.time(), mono=1.0,
                     host_port=9210, relay=r)
    time.sleep(0.3)
    assert not target.exists() or target.read_text("utf-8") == ""   # recorder still down
    rec = Recorder(port, tmp_path / "journal")
    try:
        notes.write_stop(who="key", why="after", received_wall=time.time(), mono=2.0,
                         host_port=9210, relay=r)
        wait_until(lambda: feed.delivered == 2, 10.0)
    finally:
        notes.close()
        rec.stop()
    got = [json.loads(x) for x in target.read_text("utf-8").splitlines()]
    assert [g["body"]["why"] for g in got] == ["before", "after"]
    assert all(g["late"] is True and g["src"] == "keeper" for g in got)


# ------------------------------------------------- the doors: fire and forget --

def test_with_no_keeper_port_a_door_sends_nothing(monkeypatch):
    for value in (None, "", "0", "x"):
        if value is None:
            monkeypatch.delenv(keeper_stop.PORT_ENV, raising=False)
        else:
            monkeypatch.setenv(keeper_stop.PORT_ENV, value)
        assert keeper_stop.tell_keeper("cockpit", "operator abort") is None


def test_an_absent_keeper_fails_quickly_and_silently(monkeypatch):
    monkeypatch.setenv(keeper_stop.PORT_ENV, str(free_port()))
    t0 = time.monotonic()
    t = keeper_stop.tell_keeper("flex", "abort")
    assert time.monotonic() - t0 < 0.05            # the caller does not connect
    t.join(2.0)
    assert not t.is_alive()


def test_the_door_returns_while_the_keeper_connect_is_still_hanging(monkeypatch):
    release = threading.Event()

    def hanging(*_a, **_k):
        release.wait(10)
        raise OSError("gave up")

    monkeypatch.setenv(keeper_stop.PORT_ENV, "9")
    monkeypatch.setattr(keeper_stop.socket, "create_connection", hanging)
    t0 = time.monotonic()
    t = keeper_stop.tell_keeper("flex", "abort")
    assert time.monotonic() - t0 < 0.05 and t.is_alive()
    release.set()
    t.join(2.0)


def test_the_stop_line_is_printable_ascii():
    assert keeper_stop.stop_line("cockpit", "operator abort") == b"STOP cockpit operator abort\n"
    assert keeper_stop.stop_line("flex", "é\x00  x") == b"STOP flex ?? x\n"
    assert keeper_stop.stop_line("flex", "") == b"STOP flex stop\n"
    assert keeper_stop.stop_line("flex", "a #b", "0123456789abcdef") == \
        b"STOP flex a ?b #0123456789abcdef\n"


def _no_id(lines: list[str]) -> list[str]:
    """The lines with their `#<id>` cut off, after checking each carries one."""
    out = []
    for line in lines:
        head, _, sid = line.rpartition(" #")
        assert len(sid) == 16 and int(sid, 16) >= 0, line
        out.append(head)
    return out


class FakeKeeper:
    """A keeper port that records each line and, with `hold`, never answers or closes."""

    def __init__(self):
        self._srv = socket.create_server(("127.0.0.1", 0))
        self.port = self._srv.getsockname()[1]
        self.lines: list[str] = []
        self._conns = []
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            self._conns.append(conn)
            threading.Thread(target=self._read, args=(conn,), daemon=True).start()

    def _read(self, conn):
        buf = b""
        while b"\n" not in buf:
            try:
                chunk = conn.recv(256)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
        self.lines.append(buf.split(b"\n")[0].decode())

    def close(self):
        self._srv.close()
        for c in self._conns:
            c.close()


# ------------------------------ the doors on the rig: route unchanged, keeper told --

@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


def _frozen(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("keeper", ["absent", "unreachable", "listening"])
def test_flex_safepark_commands_exactly_the_park_golden_whatever_the_keeper(
        hub, monkeypatch, keeper):
    fake = FakeKeeper() if keeper == "listening" else None
    if keeper == "absent":
        monkeypatch.delenv(keeper_stop.PORT_ENV, raising=False)
    else:
        monkeypatch.setenv(keeper_stop.PORT_ENV, str(fake.port if fake else free_port()))
    try:
        with GoldenRig(hub, "safe_park") as rig:
            res = rig.door.call("safePark", {"reason": "golden safe-park"})
            assert res["transport"] == "com" and res["verified"] is True, res
            rig.quiesce()
        frozen = _frozen("safe_park.golden.json")
        now = rig.golden("flex")
        assert now["igor"] == frozen["igor"]
        assert now["wire"] == frozen["wire"]
        if fake is not None:
            wait_until(lambda: fake.lines)
            assert _no_id(fake.lines) == ["STOP flex golden safe-park"]
    finally:
        if fake is not None:
            fake.close()


def test_flex_abort_and_the_cockpit_abort_tell_the_keeper(hub, client, monkeypatch):
    fake = FakeKeeper()
    monkeypatch.setenv(keeper_stop.PORT_ENV, str(fake.port))
    try:
        with GoldenRig(hub, "safe_park") as rig:
            assert rig.door.call("abort")["safe_park"]["verified"] is True
            wait_until(lambda: len(fake.lines) == 1)
            r = client.post("/api/safe_park", json={"reason": "operator abort"})
            assert r.status_code == 200 and r.json()["safe_park"]["verified"] is True
            wait_until(lambda: len(fake.lines) == 2)
            rig.quiesce()
    finally:
        fake.close()
    assert _no_id(fake.lines) == ["STOP flex abort", "STOP cockpit operator abort"]


def test_a_keeper_that_never_answers_does_not_delay_the_cockpit_abort(hub, client,
                                                                      monkeypatch):
    """The keeper accepts and never reads past the line: the abort answers as it does with
    no keeper at all."""
    fake = FakeKeeper()
    timings = {}
    try:
        with GoldenRig(hub, "safe_park") as rig:
            for name, port in (("absent", None), ("present", fake.port)):
                if port is None:
                    monkeypatch.delenv(keeper_stop.PORT_ENV, raising=False)
                else:
                    monkeypatch.setenv(keeper_stop.PORT_ENV, str(port))
                t0 = time.monotonic()
                r = client.post("/api/safe_park", json={"reason": "operator abort"})
                timings[name] = time.monotonic() - t0
                assert r.json()["safe_park"]["verified"] is True
                rig.quiesce()
    finally:
        fake.close()
    assert timings["present"] < timings["absent"] + 0.5, timings


# ---------------------------------------- the whole loop: door + keeper relay, twice --

def test_a_stop_through_the_keeper_parks_the_rig_and_the_note_says_confirmed(hub, tmp_path):
    from afm_litho.service.host.stop_port import StopPort
    with GoldenRig(hub, "safe_park") as rig:
        sp = StopPort(hub, 0).start()
        notes = Notes(tmp_path / "notes.jsonl")
        sl = StopListener(notes, port=0, host_port=sp.port).start()
        try:
            assert stop(sl.port, "STOP key operator pressed S\n") == [SENT, CONFIRMED]
            wait_until(lambda: notes.path.exists())
        finally:
            sl.close()
            sp.close()
        rig.quiesce()
        assert hub.host.last_park.reason == "STOP from key: operator pressed S (via keeper)"
    assert rig.stub.td_stops == 1 and rig.stub.abort_confirms == 1
    (note,) = [json.loads(x) for x in notes.path.read_text("utf-8").splitlines()]
    assert note["body"]["outcome"] == CONFIRMED and note["body"]["confirmed"] is True


def test_a_stop_through_the_keeper_confirms_while_the_event_loop_is_blocked(hub, tmp_path):
    import asyncio

    from afm_litho.service.host.stop_port import StopPort
    with GoldenRig(hub, "safe_park") as rig:
        sp = StopPort(hub, 0).start()
        sl = StopListener(RecordingNotes(), port=0, host_port=sp.port).start()
        blocked, release = threading.Event(), threading.Event()

        async def _block():
            blocked.set()
            release.wait(60)

        asyncio.run_coroutine_threadsafe(_block(), hub._loop)
        try:
            assert blocked.wait(5)
            assert stop(sl.port, "STOP key loop blocked\n") == [SENT, CONFIRMED]
            assert not release.is_set()
        finally:
            release.set()
            sl.close()
            sp.close()
        rig.quiesce()


class RelayingKeeper:
    """A keeper that takes the door's line and relays it to the host's stop port with the
    keeper's own `relay`, once `go` is set."""

    def __init__(self, host_port: int):
        self.go = threading.Event()
        self.replies: list[str] = []
        self.done = threading.Event()
        self._fake = FakeKeeper()
        self.port = self._fake.port
        self._host_port = host_port
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        wait_until(lambda: self._fake.lines, 30.0)
        self.go.wait(30)
        who, why = stop_listener.parse_stop(self._fake.lines[0])
        stop_listener.relay(self._host_port, who, why, self.replies.append)
        self.done.set()

    def close(self):
        self._fake.close()


def _gate_abort(stub):
    entered, gate = threading.Event(), threading.Event()
    real = stub.run_unrecorded

    def held(line):
        if line == "ABORT":
            entered.set()
            gate.wait(30)
        real(line)

    stub.run_unrecorded = held
    return entered, gate


def _one_park(rig, fault, height):
    s = rig.stub
    assert s.safe_parks == 1 and s.td_stops == 1 and s.abort_confirms == 1
    assert fault is None and height < 1.0


def test_door_first_then_the_keepers_copy_shares_its_park(hub, monkeypatch):
    """The door's park is inside Igor when the keeper's copy reaches the stop port: the copy
    parks nothing more and answers with the door's outcome."""
    from afm_litho.service.host.stop_port import StopPort
    with GoldenRig(hub, "safe_park") as rig:
        sp = StopPort(hub, 0).start()
        keeper = RelayingKeeper(sp.port)
        monkeypatch.setenv(keeper_stop.PORT_ENV, str(keeper.port))
        entered, gate = _gate_abort(rig.stub)
        out = {}
        door = threading.Thread(target=lambda: out.update(
            r=rig.door.call("safePark", {"reason": "door first"})))
        try:
            door.start()
            assert entered.wait(10)              # the door's park is running
            keeper.go.set()
            wait_until(lambda: keeper.replies, 10.0)
            assert keeper.replies == [SENT]
            gate.set()
            door.join(30)
            assert keeper.done.wait(20)
        finally:
            gate.set()
            keeper.go.set()
            keeper.close()
            sp.close()
        rig.quiesce()
        height, fault = rig.arc.td_ReadValue("Height"), hub.fault
    assert out["r"]["verified"] is True and out["r"]["critical"] is False
    assert keeper.replies == [SENT, CONFIRMED]
    _one_park(rig, fault, height)


def test_keeper_copy_first_then_the_door_shares_its_park(hub, monkeypatch):
    """The event loop is held, so the FLEX door's hop waits while the keeper's copy reaches
    the stop port and starts the park. The door then attaches to that park: one park, and
    the door's answer is that park's confirmed outcome, not a timeout."""
    import asyncio

    from afm_litho.service.host.stop_port import StopPort
    with GoldenRig(hub, "safe_park") as rig:
        sp = StopPort(hub, 0).start()
        keeper = RelayingKeeper(sp.port)
        keeper.go.set()
        monkeypatch.setenv(keeper_stop.PORT_ENV, str(keeper.port))
        entered, gate = _gate_abort(rig.stub)
        blocked, release = threading.Event(), threading.Event()

        async def _block():
            blocked.set()
            release.wait(60)

        asyncio.run_coroutine_threadsafe(_block(), hub._loop)
        out = {}
        door = threading.Thread(target=lambda: out.update(
            r=rig.door.call("safePark", {"reason": "keeper first"})))
        try:
            assert blocked.wait(5)
            door.start()
            assert entered.wait(10)              # the keeper's copy started the park
            wait_until(lambda: keeper.replies, 10.0)
            assert keeper.replies == [SENT]
            release.set()                        # now the door's route runs
            time.sleep(0.3)
            gate.set()
            door.join(30)
            assert keeper.done.wait(20)
        finally:
            release.set()
            gate.set()
            keeper.close()
            sp.close()
        rig.quiesce()
        height, fault = rig.arc.td_ReadValue("Height"), hub.fault
    assert out["r"]["verified"] is True and out["r"]["critical"] is False
    assert keeper.replies == [SENT, CONFIRMED]
    _one_park(rig, fault, height)


def test_the_same_stop_twice_door_and_keeper_relay_parks_once(hub, monkeypatch, tmp_path):
    """FLEX safePark with the real keeper listener relaying to the host: one park, and the
    note says confirmed."""
    from afm_litho.service.host.stop_port import StopPort
    with GoldenRig(hub, "safe_park") as rig:
        sp = StopPort(hub, 0).start()
        notes = Notes(tmp_path / "notes.jsonl")
        sl = StopListener(notes, port=0, host_port=sp.port).start()
        monkeypatch.setenv(keeper_stop.PORT_ENV, str(sl.port))
        try:
            res = rig.door.call("safePark", {"reason": "door and keeper"})
            assert res["verified"] is True, res
            wait_until(lambda: notes.path.exists(), 20.0)
        finally:
            sl.close()
            sp.close()
        rig.quiesce()
        height, fault = rig.arc.td_ReadValue("Height"), hub.fault
    (note,) = [json.loads(x) for x in notes.path.read_text("utf-8").splitlines()]
    assert note["body"]["who"] == "flex" and note["body"]["outcome"] == CONFIRMED
    _one_park(rig, fault, height)


@pytest.mark.parametrize("mfp", [False, True], ids=["cypher", "mfp3d"])
def test_with_a_real_relaying_keeper_igor_runs_exactly_the_park_golden(hub, monkeypatch,
                                                                       tmp_path, mfp):
    from golden.harness import MFP3D_MICROSCOPE_ID

    from afm_litho.service.host.stop_port import StopPort
    with GoldenRig(hub, "safe_park",
                   microscope_id=MFP3D_MICROSCOPE_ID if mfp else None) as rig:
        sp = StopPort(hub, 0).start()
        notes = Notes(tmp_path / "notes.jsonl")
        sl = StopListener(notes, port=0, host_port=sp.port).start()
        monkeypatch.setenv(keeper_stop.PORT_ENV, str(sl.port))
        try:
            res = rig.door.call("safePark", {"reason": "golden safe-park"})
            assert res["transport"] == "com" and res["verified"] is True, res
            wait_until(lambda: notes.path.exists(), 20.0)
        finally:
            sl.close()
            sp.close()
        rig.quiesce()
    if mfp:
        frozen = _frozen("safe_park.mfp3d.golden.json")
        assert rig.golden_wire("flex")["wire"] == frozen["wire"]
    else:
        frozen = _frozen("safe_park.golden.json")
        now = rig.golden("flex")
        assert now["igor"] == frozen["igor"]
        assert now["wire"] == frozen["wire"]


def test_a_repeat_make_safe_before_the_confirm_sees_igors_flag(hub):
    """The fake Igor's AFM_SafePark: a second make-safe that reaches Igor before the host's
    confirm finds `abortStopped` set and skips td_Stop."""
    with GoldenRig(hub, "safe_park") as rig:
        s = rig.stub
        s.run_unrecorded("ABORT")
        assert s.td_stops == 1 and s.abort_stopped
        s.run_unrecorded("ABORT")
        assert s.td_stops == 1 and s.abort_stopped
        res = rig.door.call("safePark", {"reason": "after two"})
        assert res["verified"] is True
        rig.quiesce()


def test_the_cockpit_send_goes_out_before_the_route_awaits_the_park(hub, client,
                                                                     monkeypatch):
    """Order: the keeper's line leaves while Igor is still held inside the door's park."""
    fake = FakeKeeper()
    monkeypatch.setenv(keeper_stop.PORT_ENV, str(fake.port))
    try:
        with GoldenRig(hub, "safe_park") as rig:
            entered, gate = threading.Event(), threading.Event()
            real = rig.stub.run_unrecorded

            def held(line):
                if line == "ABORT":
                    entered.set()
                    gate.wait(30)
                real(line)

            rig.stub.run_unrecorded = held
            out = {}
            t = threading.Thread(target=lambda: out.update(
                r=client.post("/api/safe_park", json={"reason": "operator abort"})))
            t.start()
            try:
                assert entered.wait(10)
                wait_until(lambda: fake.lines, 5.0)    # told while the park is held
            finally:
                gate.set()
                t.join(30)
            assert out["r"].json()["safe_park"]["verified"] is True
            rig.quiesce()
    finally:
        fake.close()
    assert _no_id(fake.lines) == ["STOP cockpit operator abort"]
