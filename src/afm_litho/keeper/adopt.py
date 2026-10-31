"""A revived keeper: it takes over the running children instead of starting new ones.

It reads children.json (written by the dead keeper, with how each child was started),
opens a handle on each running child, binds the same Stop port again, and tells the service
it is in contact, which lifts the write lock (answer 8). It never restarts a child
that is alive and never talks to Igor.
"""

from __future__ import annotations

import http.client
import json
import os
import ssl
import threading
import urllib.error
import urllib.request

from afm_litho import clock
from afm_litho.keeper import stop_listener
from afm_litho.keeper.process import AdoptedHandle
from afm_litho.keeper.supervisor import EXITED, GAVE_UP, ChildSpec

CONTACT_PATH = "/api/keeper/contact"
CONTACT_RETRY_S = 1.0
CONTACT_TIMEOUT_S = 3.0
CONTACT_ALERT_AFTER = 10


def specs_and_handles(state: dict) -> tuple[list[ChildSpec], dict[str, AdoptedHandle]]:
    """The children to supervise, and a handle for each one that was recorded with a pid.
    A child the old keeper had finished with (exited, gave up) is left out."""
    specs: list[ChildSpec] = []
    handles: dict[str, AdoptedHandle] = {}
    for c in state.get("children", []):
        how = c.get("launch")
        if not how or c.get("state") in (GAVE_UP, EXITED):
            continue
        specs.append(ChildSpec(c["name"], list(how["argv"]), env=dict(how.get("env", {})),
                               restart_args=tuple(how.get("restart_args", ())),
                               essential=bool(how.get("essential", True))))
        # Under a venv on Windows the pid the keeper launched is the launcher stub; the pid
        # the child reported in READY is the real process.
        pid = (c.get("ready") or {}).get("pid") or c.get("pid")
        if pid:
            handles[c["name"]] = AdoptedHandle(c["name"], int(pid))
    return specs, handles


def apply_ports(specs: list[ChildSpec]) -> None:
    """Bind the Stop port the old keeper had (the children's doors send Stops there)."""
    for spec in specs:
        port = spec.env.get(stop_listener.PORT_ENV)
        if port:
            os.environ[stop_listener.PORT_ENV] = port
            return


def recorder_port(specs: list[ChildSpec]) -> int | None:
    for spec in specs:
        if "AFM_LITHO_RECORDER_PORT" in spec.env:
            return int(spec.env["AFM_LITHO_RECORDER_PORT"])
    return None


def service_port(specs: list[ChildSpec]) -> int | None:
    for spec in specs:
        if spec.name == "service" and "--port" in spec.argv:
            try:
                return int(spec.argv[spec.argv.index("--port") + 1])
            except (IndexError, ValueError):
                return None
    return None


def bridges(state: dict) -> dict[str, str | None]:
    """The last bridge state each child reported to the dead keeper. A restart of an adopted
    service passes it on (--last-bridge), so a service that had Igor still makes safe."""
    return {c["name"]: c.get("bridge") for c in state.get("children", [])}


def reviver_is_a_child(state: dict, pid: int) -> bool:
    """Whether `pid` (the process that revived this keeper) is one of the recorded children."""
    for c in state.get("children", []):
        if pid in (c.get("pid"), (c.get("ready") or {}).get("pid")):
            return True
    return False


def _loopback_tls():
    """The site's own certificate on 127.0.0.1 is not checked: it names the site, not the
    loopback address, and nothing leaves this machine."""
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def post_contact(port: int, pid: int) -> bool:
    """Tell the service at 127.0.0.1:port that keeper `pid` is in contact: over http, and over
    https when the site serves TLS on that port."""
    body = json.dumps({"pid": pid}).encode()
    for scheme, context in (("http", None), ("https", _loopback_tls())):
        req = urllib.request.Request(
            f"{scheme}://127.0.0.1:{port}{CONTACT_PATH}", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=CONTACT_TIMEOUT_S, context=context) as resp:
                if json.loads(resp.read()).get("ok"):
                    return True
        except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException):
            continue
    return False


def keep_contact(port: int, stop: threading.Event, say, report=None) -> threading.Thread:
    """Post the contact until the service accepts it (it may still be busy), on a thread so
    the supervisor loop is never held up by the service. Call after the state file names
    this keeper: the service checks it there. After CONTACT_ALERT_AFTER failed tries it
    says so loudly (and calls `report`, which writes the journal note), then keeps trying."""
    pid = os.getpid()

    def run() -> None:
        tries = 0
        while not stop.is_set():
            if post_contact(port, pid):
                say(f"[keeper] in contact with the service on port {port}; writes are allowed")
                return
            tries += 1
            if tries == CONTACT_ALERT_AFTER:
                msg = (f"SAFETY: the revived keeper (pid {pid}) cannot make contact with the "
                       f"service on port {port} after {tries} tries; writes stay refused. "
                       f"Stop, park and withdraw still work.")
                say(f"[keeper] {msg}")
                if report is not None:
                    report(msg)
            clock.wait(stop, CONTACT_RETRY_S)

    thread = threading.Thread(target=run, name="keeper-contact", daemon=True)
    thread.start()
    return thread
