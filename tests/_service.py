"""A twin service child process for the browser and restart tests.

One helper for `tests/usertwin/` and `tests/test_design_autosave.py`. Each
instance gets its own runtime folder and OS-chosen ports, so no two modules or
xdist workers share a folder or a port.

uvicorn cannot hand back a port it chose itself, so the port is picked by
binding port 0 and releasing it. Another process can take it before the child
binds it; `start()` therefore retries on fresh ports when the child exits at
once, whatever the cause (a lost bind is the usual one, but any immediate exit
retries, and the last try raises with the child's log).
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import closing
from pathlib import Path
from typing import Self

START_TRIES = 5
FIXTURE_RUNTIME = Path(__file__).parent / "fixtures" / "runtime"


def free_port() -> int:
    """A port the OS has just said is free. Only a hint; see the module docstring."""
    with closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def free_port_pair() -> int:
    """A port p with p and p + 1 both free right now (FLEX reads on command_port + 1)."""
    for _ in range(50):
        with closing(socket.socket()) as a, closing(socket.socket()) as b:
            try:
                a.bind(("127.0.0.1", 0))
                port = a.getsockname()[1]
                b.bind(("127.0.0.1", port + 1))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port pair")


class Service:
    """The twin's own service instance, on its own ports and runtime folder.

    `flex=True` also enables the FLEX provider on a port pair of its own (the
    in-process `client` fixture turns FLEX off so it cannot claim the real
    29180/29181). `launch_code` is an optional `python -c` body that starts the
    service itself, with `{port}` substituted, for a test that needs hub state
    no HTTP door can set.
    """

    launch_code: str | None = None

    def __init__(self, runtime: Path | None = None, twin_speed: float | None = None,
                 flex: bool = False):
        self._own_runtime = runtime is None     # a folder made here is removed by stop()
        self.runtime = Path(runtime) if runtime else Path(
            tempfile.mkdtemp(prefix="afm-litho-test-service-"))
        self.twin_speed = twin_speed
        self.flex = flex
        self.port = 0
        self.flex_port = 0
        self.url = ""
        self.proc: subprocess.Popen[bytes] | None = None
        self.log = self.runtime / "service.log"

    def start(self, timeout_s: float = 60.0) -> Self:
        self.runtime.mkdir(parents=True, exist_ok=True)
        # The same checked-in model every other test runs on (tests/conftest.py);
        # without it a test that reads the file passes or fails by whether an
        # earlier test in its module happened to write it first.
        seed = FIXTURE_RUNTIME / "instrument.toml"
        if not (self.runtime / "instrument.toml").exists():
            shutil.copy2(seed, self.runtime / "instrument.toml")
        for attempt in range(START_TRIES):
            self.port = free_port()
            self.flex_port = free_port_pair() if self.flex else 0
            self.url = f"http://127.0.0.1:{self.port}"
            self._spawn()
            if self._wait_until_answering(timeout_s, last_try=attempt == START_TRIES - 1):
                return self
            self._terminate()       # the runtime stays for the next try
        raise RuntimeError("unreachable")

    def _spawn(self) -> None:
        # app.py reads RUNTIME at import time, so this has to be a child process.
        env = dict(os.environ)
        env["AFM_LITHO_RUNTIME"] = str(self.runtime)
        for name in ("AFM_LITHO_PORT", "AFM_LITHO_TLS_CERT", "AFM_LITHO_TLS_KEY"):
            env.pop(name, None)
        env["PYTHONUTF8"] = "1"
        if self.twin_speed is not None:
            env["AFM_LITHO_TWIN_SPEED"] = repr(float(self.twin_speed))
        if self.flex:
            env["AFM_LITHO_FLEX"] = "1"
            env["AFM_LITHO_FLEX_PORT"] = str(self.flex_port)
        argv = [sys.executable, "-m", "uvicorn", "afm_litho.service.app:app",
                "--host", "127.0.0.1", "--port", str(self.port), "--log-level", "warning"]
        if self.launch_code:
            argv = [sys.executable, "-c", self.launch_code.replace("{port}", str(self.port))]
        self._logfh = self.log.open("wb")
        self.proc = subprocess.Popen(
            argv, env=env, stdout=self._logfh, stderr=subprocess.STDOUT)

    def _wait_until_answering(self, timeout_s: float, last_try: bool) -> bool:
        """True once the child answers. False if it exited at once (a lost port
        race) and another try is left; otherwise raises."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                if not last_try:
                    return False
                raise RuntimeError(
                    f"service exited with {self.proc.returncode}\n"
                    + self.log.read_text(errors="replace")[-3000:])
            try:
                with urllib.request.urlopen(self.url + "/api/state", timeout=1.5):
                    # an answer from a stranger on the port is not ours
                    if self.proc.poll() is None:
                        return True
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.25)
        self.stop()
        raise TimeoutError(f"service did not answer on {self.url} in {timeout_s} s")

    def stop(self) -> None:
        """Stop the child. A runtime folder this object created itself is removed;
        one the caller passed in is left alone."""
        self._terminate()
        if self._own_runtime:
            shutil.rmtree(self.runtime, ignore_errors=True)

    def _terminate(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        fh = getattr(self, "_logfh", None)
        if fh and not fh.closed:
            fh.close()

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def get(self, path: str) -> tuple[int, dict | None]:
        return self._request(urllib.request.Request(self.url + path))

    def post(self, path: str, body: dict | None = None) -> tuple[int, dict | None]:
        req = urllib.request.Request(
            self.url + path, data=json.dumps(body or {}).encode("utf-8"), method="POST",
            headers={"content-type": "application/json"})
        return self._request(req)

    @staticmethod
    def _request(req: urllib.request.Request) -> tuple[int, dict | None]:
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read().decode("utf-8"))
            except Exception:      # noqa: BLE001 -- best-effort JSON decode of an error body
                return exc.code, None
