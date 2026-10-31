"""The service's side of the keeper: the READY line and the heartbeat, and the watch on the
keeper itself.

Only a service started by the keeper (AFM_LITHO_KEEPER_CHILD=1) has a link; under
the test client and in `python -m afm_litho.service_main` run by hand there is none
and every call here does nothing.

Keeper death (decision 8, phase6-board/plan.md): the service reports it, lets the
object being written finish, then refuses every further write, the next queued object of
the same run included, until a keeper is in contact again (`restore_contact`). Stop, park
and withdraw never ask this module anything.
"""

from __future__ import annotations

import asyncio
import os
import queue
import sys
import threading

from afm_litho import clock
from afm_litho.keeper import beatfile, revive, status
from afm_litho.keeper.pids import ProcessHandle
from afm_litho.keeper.protocol import (
    BRIDGE_DOWN,
    BRIDGE_UP,
    HEARTBEAT_EVERY_S,
    KEEPER_PID_ENV,
    heartbeat_line,
    ready_line,
)

_link = None                       # the beat file the keeper reads (keeper/beatfile.py)
_beat_task: asyncio.Task | None = None
_outbox: queue.Queue = queue.Queue(maxsize=8)
_writer_thread: threading.Thread | None = None

# The keeper being watched, and why contact with it was lost (None: in contact, or no
# keeper at all). Only the operating system says a keeper is gone, never a missed beat.
_watch_lock = threading.Lock()
_keeper_pid: int | None = None
_keeper_handle: ProcessHandle | None = None   # held open: a reused pid is not this keeper
_keeper_lost: str | None = None
CONTACT_WAIT_S = 30.0       # how long a revived keeper has to make contact
_reviver: revive.Reviver | None = None     # one per service, so its revival limit is kept


def claim_link() -> None:
    """Open the beat file the keeper gave this service (AFM_LITHO_BEAT_FILE) and watch
    the keeper's pid. The service's stdout stays the keeper's console."""
    global _link
    if os.environ.get("AFM_LITHO_KEEPER_CHILD") != "1":
        return
    _link = beatfile.from_env()
    pid = os.environ.get(KEEPER_PID_ENV, "").strip()
    if pid.isdigit():
        watch_keeper(int(pid))


def _send(line: str) -> None:
    """Queue a line for the writer thread, so a slow disk never freezes the event loop."""
    if _link is None:
        return
    try:
        _outbox.put_nowait(line)
    except queue.Full:
        pass                       # the writer is behind; a newer heartbeat follows


def _writer(link) -> None:
    while True:
        line = clock.queue_get(_outbox, None)
        if line is None:
            return
        link.write(line + "\n")
        link.flush()


# ------------------------------------------------------------------ READY --

def _igor_mark(arc) -> int | None:
    """Igor's stop mark (root:AFM_abortStopped) as the wire last told it: the newest v3
    stamp, else HELLO's `mark=`. None when Igor has not told it (procedures older than
    wire v3 have no mark on the wire)."""
    router = getattr(arc, "router", None)
    if getattr(router, "last_useq", None) is not None:
        return 1 if router.mark else 0
    mark = (getattr(getattr(arc, "link", None), "hello", None) or {}).get("mark")
    return int(mark) if mark in ("0", "1") else None


def read_marks(hub) -> tuple[int | None, int]:
    """(Igor stop mark, fault latched) for the READY line. The twin has no Igor, so no
    stop mark can be set (0). A hardware station reports the mark only once Igor has
    dialled in and told it."""
    fault = 1 if getattr(hub, "fault", None) else 0
    backend = getattr(hub.host, "backend", None)
    if backend is None:
        return None, fault
    if backend != "hardware":
        return 0, fault
    from .hub import real_bridge
    arc = real_bridge(hub)
    return (None if arc is None else _igor_mark(arc)), fault


def bridge_state(hub) -> str | None:
    """"up" while a live bridge to Igor is bound, else "down"; None for a hub that has
    no backend. The keeper passes the last one to a restart after a death."""
    live = getattr(hub, "backend_live", None)
    if not callable(live):
        return None
    return BRIDGE_UP if live() == "hardware" else BRIDGE_DOWN


def send_ready(hub) -> None:
    """READY with the mark and fault as they are now. Sent at start, and again once
    Igor's link is bound (app.py), when the mark can first be read."""
    mark, fault = read_marks(hub)
    _send(ready_line(mark, fault, os.getpid(), bridge_state(hub)))


def start(hub) -> None:
    """Send READY, then start the heartbeat. It runs on the event loop, not a
    thread, so a blocked loop shows up at the keeper as a missed heartbeat. Each
    beat also looks for a dead keeper, so its death is reported within a second."""
    global _beat_task, _writer_thread
    if _link is None:
        return
    _writer_thread = threading.Thread(target=_writer, args=(_link,), daemon=True)
    _writer_thread.start()
    send_ready(hub)

    async def beat() -> None:
        while True:
            _send(heartbeat_line(os.getpid(), bridge_state(hub)))
            keeper_lost(hub)
            await clock.async_sleep(HEARTBEAT_EVERY_S)

    _beat_task = asyncio.get_running_loop().create_task(beat())


def stop() -> None:
    global _beat_task
    if _beat_task is not None:
        _beat_task.cancel()
        _beat_task = None
    global _writer_thread
    if _writer_thread is None:
        return
    while True:                    # drop what is unsent, so the end marker fits
        try:
            _outbox.get_nowait()
        except queue.Empty:
            break
    _outbox.put_nowait(None)       # ends the writer thread
    _writer_thread.join(timeout=0.2)    # not forever: the writer may be stuck in write()
    _writer_thread = None


# ------------------------------------------------------- the keeper watch --

def watch_keeper(pid: int) -> None:
    """Watch keeper `pid`: the first keeper (from its env), or a restarted one that has
    made contact. Contact is restored, so writes are allowed again."""
    global _keeper_pid, _keeper_lost, _keeper_handle
    with _watch_lock:
        if _keeper_handle is not None:
            _keeper_handle.close()
        _keeper_pid = int(pid)
        _keeper_handle = ProcessHandle(_keeper_pid)     # not opened: judged lost
        _keeper_lost = None


def forget_keeper() -> None:
    """Watch no keeper (a service with none). Writes are not refused."""
    global _keeper_pid, _keeper_lost, _keeper_handle
    with _watch_lock:
        if _keeper_handle is not None:
            _keeper_handle.close()
        _keeper_pid, _keeper_lost, _keeper_handle = None, None, None


def keeper_lost(hub=None) -> str | None:
    """Why contact with the keeper is lost, or None while it is in contact (or there is
    none). Checked now against the operating system: is the keeper's process gone. The
    first time it is seen it is reported on the console and in the cockpit log."""
    global _keeper_lost
    pid = _keeper_pid
    if pid is None or _keeper_lost is not None:
        return _keeper_lost
    with _watch_lock:       # watch_keeper closes the handle under this lock
        handle = _keeper_handle
        if handle is not None and not handle.exited():
            return None
    msg = (f"the keeper (pid {pid}) is gone (its process has exited). The object being written finishes; "
           f"no further write starts until a keeper is in contact again. Stop, park and "
           f"withdraw still work.")
    with _watch_lock:
        if _keeper_pid != pid or _keeper_lost is not None:
            return _keeper_lost
        _keeper_lost = msg
    _report(hub, f"SAFETY: {msg}")
    _revive_in_background(pid, hub)
    return msg


def _revive_in_background(pid: int, hub) -> None:
    """Bring the dead keeper back (keeper/revive.py). Only a service a keeper started does:
    one watching a pid by hand (tests, a hand-run service) has nothing to revive. The dead
    keeper must be the one the state file names, so no other pid starts a revival."""
    if _link is None:
        return
    state = status.read_state(status.state_path()) or {}
    if state.get("keeper_pid") != pid:
        return
    threading.Thread(target=_revive, args=(hub,), name="keeper-revive", daemon=True).start()


def _revive(hub) -> None:
    global _reviver
    if _reviver is None:
        _reviver = revive.Reviver(status.runtime_dir())
    outcome, words = _reviver.revive()
    if outcome == revive.REVIVED:
        _await_contact(hub, words)
        return
    loud = outcome in (revive.LIMIT, revive.FAILED)
    _report(hub, f"{'SAFETY: ' if loud else ''}keeper revival: {words}")


def _await_contact(hub, words: str) -> None:
    """A revival counts only once the new keeper has made contact (restore_contact)."""
    end = clock.now() + CONTACT_WAIT_S
    while clock.now() < end:
        if _keeper_lost is None:
            _report(hub, f"keeper revival: {words}; it is in contact, writes are allowed")
            return
        clock.sleep(0.5)
    _report(hub, f"SAFETY: keeper revival: {words}, but it has made no contact in "
                 f"{CONTACT_WAIT_S:.0f} s; writes stay refused. Stop, park and withdraw still work.")


def restore_contact(pid: int, hub=None) -> None:
    """A keeper (`pid`, alive) is in contact again: watch it and allow writes."""
    was = _keeper_lost
    watch_keeper(pid)
    if was is not None:
        _report(hub, f"keeper pid {pid} is in contact; writes are allowed again")


def _report(hub, msg: str) -> None:
    log = getattr(hub, "_log_sync", None)
    if callable(log):
        log(msg)
    else:
        print(f"[keeper link] {msg}", file=sys.stderr, flush=True)
