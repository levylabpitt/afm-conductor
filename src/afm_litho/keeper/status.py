"""The keeper's state file and `python -m afm_litho.keeper status`.

The keeper writes runtime/keeper/children.json; `status` reads it and asks the
operating system whether each pid is still running, so it is truthful when the
keeper itself has died and left its children running.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from afm_litho import clock
from afm_litho.keeper.pids import pid_alive


REPLACE_TRIES = 10
REPLACE_WAIT_S = 0.02


def runtime_dir() -> Path:
    """The same folder service/deps.py calls RUNTIME (the keeper may not import it)."""
    override = os.environ.get("AFM_LITHO_RUNTIME")
    if override:
        return Path(override)
    if getattr(sys, "frozen", False):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
        return Path(base) / "afm-litho"
    return Path(__file__).resolve().parents[3] / "runtime"


def state_path() -> Path:
    return runtime_dir() / "keeper" / "children.json"


def write_state(path: Path, snapshot: dict) -> None:
    """Atomic: `status` never reads half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"keeper_pid": os.getpid(), "written_wall": clock.wall(), **snapshot}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, indent=1), encoding="utf-8")
    for attempt in range(REPLACE_TRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:         # Windows: a reader has the file open right now
            if attempt == REPLACE_TRIES - 1:
                raise
            clock.sleep(REPLACE_WAIT_S)


def read_state(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def render(state: dict | None, now_wall: float | None = None,
           alive=pid_alive) -> list[str]:
    """The lines `status` prints."""
    if state is None:
        return ["no keeper has run here (no state file)"]
    now = clock.wall() if now_wall is None else now_wall
    kpid = state.get("keeper_pid")
    lines = [f"keeper   pid {kpid}  " + ("running" if alive(kpid) else
             "NOT RUNNING (children, if any, keep running on their own)")]
    for c in state.get("children", []):
        ready = c.get("ready")
        # The pid the child reports is the real service. Under a venv on Windows the
        # pid the keeper launched is python.exe's launcher stub, not the service.
        pid = ready["pid"] if ready else c.get("pid")
        running = pid is not None and alive(pid)
        what = c["state"] if running else f"NOT RUNNING (last state: {c['state']})"
        parts = [f"{c['name']:<8} pid {pid}  {what}"]
        if ready:
            mark = "?" if ready["mark"] is None else ready["mark"]
            parts.append(f"READY mark={mark} fault={ready['fault']}")
        if c.get("last_beat_wall") is not None:
            parts.append(f"heartbeat {max(0.0, now - c['last_beat_wall']):.1f} s ago")
        parts.append(f"restarts {c['restarts_total']}")
        lines.append("  ".join(parts))
    return lines


def print_status() -> int:
    for line in render(read_state(state_path())):
        print(line)
    return 0
