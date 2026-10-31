"""`python -m afm_litho [--port N] [--host H] [--no-browser]`, and `... keeper status|stop`.

Chooses the port, starts the service as a child (afm_litho.service_main), opens the
cockpit once it answers, and keeps the service running. The flags are the ones
`python -m afm_litho` always had.
"""

from __future__ import annotations

import argparse
import importlib.util
import socket
import sys
import threading
import webbrowser
from typing import Callable

from afm_litho import clock, wire_version
from afm_litho.keeper import adopt, console_key, lock, status, stop_client, stop_listener
from afm_litho.keeper.notes import Notes, RecorderFeed, contact_note
from afm_litho.keeper.protocol import AFTER_DEATH_ARG, PORT_TAKEN_EXIT
from afm_litho.keeper.revive import ADOPT_ARG, REVIVED_BY_ARG
from afm_litho.keeper.supervisor import ChildSpec, Launch, Supervisor

DEFAULT_PORT = 7461
ADOPT_LOCK_TRIES = 50
ADOPT_LOCK_WAIT_S = 0.1
OPEN_BROWSER_FOR_S = 20.0      # how long to wait for the port before giving up on the tab


def _free_port(preferred: int) -> int:
    """Use the preferred port if it is free, otherwise the next one that is."""
    for port in range(preferred, preferred + 20):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return preferred


class BrowserOpener:
    """Opens the cockpit once, as soon as the service's port accepts a connection."""

    def __init__(self, port: int, url: str, open_url: Callable[[str], object]):
        self._port, self._url, self._open = port, url, open_url
        self._next = 0.0
        self._until = clock.now() + OPEN_BROWSER_FOR_S
        self.done = False

    def poll(self) -> None:
        now = clock.now()
        if self.done or now < self._next:
            return
        self._next = now + 0.25
        with socket.socket() as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", self._port)) == 0:
                self._open(self._url)
                self.done = True
        if now > self._until:
            self.done = True


def _loopback_port() -> int:
    """A free port on 127.0.0.1 for the recorder; the keeper hands it to both children.
    Another program can take it between this check and the recorder's bind. Then the recorder
    exits with PORT_TAKEN_EXIT and is not restarted, and the service spills to its files
    until the keeper is started again."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def child_specs(port: int, host: str, *, recorder_port: int | None = None,
                keeper_port: int | None = None) -> list[ChildSpec]:
    """The children, in start order. A child with `essential=False` is not a startup
    gate; the others end the keeper when they are finished for good. The recorder starts
    first but the service never waits for it: it spills until the recorder answers.
    `keeper_port` is the keeper's Stop port; the service's doors send their Stops there too."""
    recorder_port = _loopback_port() if recorder_port is None else recorder_port
    # The service child (and every restart of it) gets the wire version the keeper checked.
    env = {"AFM_LITHO_RECORDER_PORT": str(recorder_port),
           wire_version.ENV: str(wire_version.from_env())}
    if keeper_port is not None:
        env[stop_listener.PORT_ENV] = str(keeper_port)
    recorder = ChildSpec("recorder", [sys.executable, "-m", "afm_litho.recorder_main"],
                         env=env, essential=False)
    service = ChildSpec(
        "service",
        [sys.executable, "-m", "afm_litho.service_main", "--port", str(port), "--host", host],
        env=env,
        # A restarted service makes the tip safe first if it had Igor (service_main.py).
        restart_args=(AFTER_DEATH_ARG,))
    return [recorder, service]


def _missing_dependency() -> int | None:
    """The keeper cannot import uvicorn (stdlib only), so it asks whether it is there."""
    if importlib.util.find_spec("uvicorn") is None:
        print("Missing dependencies. Install them with:\n"
              "    python -m pip install -e .\n"
              "(run that from the afm-litho folder), or just use run.ps1",
              file=sys.stderr)
        return 1
    # The panel is useless without WebSockets: every live update (telemetry,
    # scan lines, write progress, the log) arrives over /ws.
    if not (importlib.util.find_spec("websockets") or importlib.util.find_spec("wsproto")):
        print("\n  This build of uvicorn has no WebSocket support, so the\n"
              "  panel would load but never update. Install it with:\n\n"
              f"      {sys.executable} -m pip install websockets\n\n"
              "  (or re-run run.ps1, which now installs it for you)\n",
              file=sys.stderr)
        return 2
    return None


def main(argv: list[str] | None = None, *, launch: Launch | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["status"]:
        return status.print_status()
    if argv[:1] == ["stop"]:
        return stop_client.main_stop(argv[1:])

    ap = argparse.ArgumentParser(prog="afm-conductor", description="Run the AFM twin")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help="a port given here is used as-is; the default lets "
                         "site.py pick (443 when AFM_LITHO_PORT=443 and TLS "
                         "is available)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument(REVIVED_BY_ARG, type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument(ADOPT_ARG, action="store_true",
                    help="a revived keeper: take over the running children (started by the "
                         "service when the keeper died, not by hand)")
    args = ap.parse_args(argv)
    if args.adopt:
        return _main_adopt(launch, args.revived_by)

    problem = _missing_dependency()
    if problem is not None:
        return problem
    try:
        wire_version.from_env()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    runtime = status.runtime_dir()
    held = lock.acquire(runtime)
    if held is None:
        print("\n  Another keeper is already running (it holds runtime/keeper/keeper.lock). "
              "Use that one, or stop it first.\n", file=sys.stderr)
        return lock.ANOTHER_KEEPER_EXIT
    try:
        lock.clear_clean_marker(runtime)
        lock.mark_on_console_close(runtime)
        return _run(args, launch, runtime)
    finally:
        held.release()


def _main_adopt(launch: Launch | None, revived_by: int | None) -> int:
    runtime = status.runtime_dir()
    held = None
    for _ in range(ADOPT_LOCK_TRIES):       # the reviver let go of the lock just before this
        held = lock.acquire(runtime)
        if held is not None:
            break
        clock.sleep(ADOPT_LOCK_WAIT_S)
    if held is None:
        print("\n  Another keeper is already running; not adopting.\n", file=sys.stderr)
        return lock.ANOTHER_KEEPER_EXIT
    try:
        lock.clear_clean_marker(runtime)
        lock.mark_on_console_close(runtime)
        return _run_adopted(launch, runtime, revived_by)
    finally:
        held.release()


def _run_adopted(launch: Launch | None, runtime, revived_by: int | None) -> int:
    state = status.read_state(status.state_path()) or {}
    specs, handles = adopt.specs_and_handles(state)
    if not specs:
        print("\n  Nothing to adopt: no running children are recorded in children.json.\n",
              file=sys.stderr)
        return 1
    if revived_by is None or not adopt.reviver_is_a_child(state, revived_by):
        print(f"\n  Not adopting: pid {revived_by} is not one of the children recorded in "
              f"children.json.\n", file=sys.stderr)
        return 1
    if launch is None:
        from afm_litho.keeper.process import launch
    adopt.apply_ports(specs)
    port = adopt.recorder_port(specs)
    print("\n  afm-litho keeper (revived): taking over the running children\n"
          "  (Ctrl+C ends this keeper; the children keep running)\n")
    notes = Notes(feed=None if port is None else RecorderFeed(port))
    listener = stop_listener.start(notes)
    keeper = Supervisor(specs, launch, state_path=status.state_path(),
                        on_clean_exit=lambda: lock.write_clean_marker(runtime))
    keeper.adopt(handles, adopt.bridges(state))     # also writes children.json: names this keeper
    key = None if listener is None else console_key.start(keeper_port=listener.port,
                                                          host_port=listener.host_port)
    done = threading.Event()
    service = adopt.service_port(specs)
    if service is not None and listener is not None:    # no Stop port, no contact
        adopt.keep_contact(service, done, lambda m: print(m, flush=True),
                           report=lambda m: notes.write(contact_note(m)))
    try:
        return keeper.run(adopted=True)
    finally:
        done.set()
        for handle in handles.values():
            handle.close()
        if key is not None:
            key.set()
        if listener is not None:
            listener.close()
        notes.close()


def _run(args, launch: Launch | None, runtime) -> int:
    from afm_litho import site

    # The port depends on whether we are about to serve under our own name
    # over TLS, because 443 is what removes the port from the address.
    if args.host not in ("127.0.0.1", "0.0.0.0", "localhost"):
        port = _free_port(args.port)                    # an explicit host: as asked
        url = f"http://{args.host}:{port}"
    else:
        if args.port != DEFAULT_PORT:
            # Asked for by name: bind exactly that, and say so if it is taken
            # rather than sliding onto the next one. Sliding is how five
            # stale servers ended up on 443 and 7461-7464.
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", args.port)) == 0:
                    print(f"\n  Port {args.port} is already in use. Stop what is "
                          f"on it, or leave --port off to pick one.\n",
                          file=sys.stderr)
                    return 3
            port = args.port
        else:
            port = site.choose_port(site.tls_possible())
        resolved = site.resolve_site(port)
        url = resolved["url"]
        # Why it fell back, rather than leaving somebody guessing at DNS.
        for why in resolved["reasons"]:
            print(f"  [site] staying on 127.0.0.1: {why}", file=sys.stderr)

    # Whatever path chose the port, a listener already on it is reported, not slid past.
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            print(f"\n  Port {port} is already in use. Stop what is on it, or "
                  f"leave --port off to pick one.\n", file=sys.stderr)
            return PORT_TAKEN_EXIT

    cockpit = site.cockpit_url(url)
    watchers = []
    if not args.no_browser:
        watchers.append(BrowserOpener(port, cockpit, webbrowser.open).poll)

    if launch is None:
        from afm_litho.keeper.process import launch
    print(f"\n  afm-litho twin -> {cockpit}\n"
          f"  (old main panel: {url})\n  (Ctrl+C to stop)\n")
    recorder_port = _loopback_port()
    notes = Notes(feed=RecorderFeed(recorder_port))
    listener = stop_listener.start(notes)
    specs = child_specs(port, args.host, recorder_port=recorder_port,
                        keeper_port=None if listener is None else listener.port)
    keeper = Supervisor(specs, launch, watchers=watchers, state_path=status.state_path(),
                        on_clean_exit=lambda: lock.write_clean_marker(runtime))
    key = None if listener is None else console_key.start(keeper_port=listener.port,
                                                          host_port=listener.host_port)
    try:
        return keeper.run()
    finally:
        if key is not None:
            key.set()
        if listener is not None:
            listener.close()
        notes.close()
