"""K5: the S key in the keeper's console, and `keeper stop` (Stop AFM.cmd)."""

from __future__ import annotations

import socket
import threading

from afm_litho.keeper import console_key, stop_client


def _serve(replies: list[str], got: list[str]):
    """A one-shot fake stop port; records the line it receives, answers, closes."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def run():
        conn, _ = srv.accept()
        with conn:
            got.append(conn.recv(512).decode().strip())
            for r in replies:
                conn.sendall(r.encode() + b"\n")
        srv.close()
    threading.Thread(target=run, daemon=True).start()
    return srv.getsockname()[1]


def _dead_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_each_s_press_sends_exactly_one_stop_and_other_keys_none():
    keys = iter(["a", "S", None, "x", "s", "\x03", None])
    stop = threading.Event()
    sent = []

    def poll():
        try:
            return next(keys)
        except StopIteration:
            stop.set()
            return None
    console_key.run(poll, lambda: sent.append(1), stop, tick_s=0)
    assert len(sent) == 2


def test_the_key_thread_sends_the_console_stop_and_prints_the_outcome():
    got, said = [], []
    port = _serve(["sent", "confirmed"], got)
    keys = iter(["s"])
    stop = console_key.start(keeper_port=port, host_port=_dead_port(),
                             poll=lambda: next(keys, None), say=said.append)
    for _ in range(100):
        if any("confirmed" in m for m in said):
            break
        threading.Event().wait(0.05)
    stop.set()
    assert got == ["STOP keeper-console key"]
    assert any(m.endswith("sent") for m in said) and any(m.endswith("confirmed") for m in said)


def test_no_console_means_no_key_thread(monkeypatch):
    monkeypatch.setattr(console_key, "_console_poll", lambda: None)
    assert console_key.start() is None


def test_stop_goes_to_the_keeper_first():
    got, host_got, said = [], [], []
    kp = _serve(["sent", "confirmed"], got)
    hp = _serve(["sent", "confirmed"], host_got)
    out = stop_client.send_stop("stop-cmd", "double-click", said.append,
                                keeper_port=kp, host_port=hp)
    assert out == "confirmed" and got == ["STOP stop-cmd double-click"] and host_got == []


def test_stop_falls_back_to_the_host_port_when_the_keeper_is_silent():
    host_got, said = [], []
    hp = _serve(["sent", "confirmed"], host_got)
    out = stop_client.send_stop("stop-cmd", "double-click", said.append,
                                keeper_port=_dead_port(), host_port=hp)
    assert out == "confirmed" and host_got == ["STOP stop-cmd double-click"]
    assert any("not answering" in m for m in said)


def test_stop_says_not_confirmed_when_nothing_answers():
    out = stop_client.send_stop("stop-cmd", "double-click", lambda m: None,
                                keeper_port=_dead_port(), host_port=_dead_port())
    assert out.startswith("NOT confirmed")


def test_a_keeper_that_accepts_but_stays_silent_falls_back_after_the_sent_wait(monkeypatch):
    monkeypatch.setattr(stop_client, "SENT_WAIT_S", 0.3)
    silent = socket.socket()
    silent.bind(("127.0.0.1", 0))
    silent.listen(1)                      # accepts in the backlog, never answers
    host_got = []
    hp = _serve(["sent", "confirmed"], host_got)
    out = stop_client.send_stop("stop-cmd", "double-click", lambda m: None,
                                keeper_port=silent.getsockname()[1], host_port=hp)
    silent.close()
    assert out == "confirmed" and host_got == ["STOP stop-cmd double-click"]
