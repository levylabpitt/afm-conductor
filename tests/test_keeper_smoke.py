"""`python -m afm_litho --no-browser --port N` still serves the twin, now through the
keeper and a service child. A real subprocess; real time (a start takes seconds)."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request

from afm_litho.keeper import status


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _kill_tree(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
    else:
        os.kill(pid, 9)


def test_the_launcher_still_serves_the_twin_and_status_lists_the_pids(tmp_path):
    port = _free_port()
    env = {**os.environ, "AFM_LITHO_RUNTIME": str(tmp_path), "AFM_LITHO_FLEX": "0"}
    log = open(tmp_path / "keeper.log", "w")  # noqa: SIM115 - closed in the finally below
    keeper = subprocess.Popen(
        [sys.executable, "-m", "afm_litho", "--no-browser", "--port", str(port)],
        env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    try:
        body = None
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and body is None:
            assert keeper.poll() is None, "the keeper exited before the twin answered"
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/edit-cockpit",
                                            timeout=2) as r:
                    body = r.status
            except OSError:
                time.sleep(0.5)
        assert body == 200

        listing = ""
        for _ in range(40):                 # the keeper sees READY within a tick
            listing = subprocess.run([sys.executable, "-m", "afm_litho.keeper", "status"],
                                     env=env, capture_output=True, text=True).stdout
            if listing.count("READY") >= 2:       # the recorder and the service
                break
            time.sleep(0.25)
        state = status.read_state(tmp_path / "keeper" / "children.json")
        service_pid = next(c for c in state["children"] if c["name"] == "service")["ready"]["pid"]
        assert f"keeper   pid {state['keeper_pid']}  running" in listing
        assert f"service  pid {service_pid}" in listing
        assert "recorder pid" in listing
        assert "READY mark=0 fault=0" in listing        # the twin has no Igor stop mark
    finally:
        _kill_tree(keeper.pid)
        keeper.wait(timeout=30)
        log.close()
