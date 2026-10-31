"""Start the children, restart a dead one, and report one that goes quiet.

A child that exits cleanly (or by Ctrl+C) was asked to stop, so the keeper ends
with it. Any other exit is a crash: restart it, up to MAX_RESTARTS within
RESTART_WINDOW_S, then give up loudly. A child whose heartbeat stops is only
reported, never killed: a killed socket does not stop a pass that Igor is playing
(2026-10-28).

All waits go through afm_litho.clock, so tests drive the loop with virtual time.
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Protocol

from afm_litho import clock
from afm_litho.keeper import status
from afm_litho.keeper.protocol import (
    AFTER_DEATH_ARG,
    BRIDGE_DOWN,
    LAST_BRIDGE_ARG,
    PORT_TAKEN_EXIT,
    Beat,
    Ready,
    parse_line,
)

MAX_RESTARTS = 3
RESTART_WINDOW_S = 120.0
BEAT_TIMEOUT_S = 5.0       # heartbeat is 1 s; five missed in a row is "quiet"
READY_TIMEOUT_S = 60.0     # the service imports numpy, scipy and fastapi before READY

# Exit codes of a child that was asked to stop: clean, Ctrl+C on Windows
# (STATUS_CONTROL_C_EXIT, signed and unsigned), SIGINT on POSIX.
QUIT_CODES = frozenset({0, 0xC000013A, -1073741510, -2})

EXIT_GAVE_UP = 4

STARTING, UP, QUIET, GAVE_UP, EXITED = "starting", "up", "quiet", "gave_up", "exited"


@dataclass
class ChildSpec:
    name: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    # Extra arguments passed to a restarted child, and not to its first start.
    restart_args: tuple[str, ...] = ()
    # When an essential child is finished for good, the keeper ends.
    essential: bool = True


class Handle(Protocol):
    pid: int

    def poll(self) -> int | None: ...
    def drain(self) -> list[str]: ...
    def terminate(self) -> None: ...


Launch = Callable[[ChildSpec, tuple[str, ...]], Handle]


@dataclass
class Slot:
    spec: ChildSpec
    handle: Handle | None = None
    state: str = STARTING
    started_at: float = 0.0
    started_wall: float = 0.0
    last_beat: float = 0.0
    last_beat_wall: float | None = None
    ready: Ready | None = None
    bridge: str | None = None       # the child's last word on its bridge to Igor
    restarts: deque = field(default_factory=deque)     # clock times, inside the window
    restarts_total: int = 0


class Supervisor:
    def __init__(self, specs: list[ChildSpec], launch: Launch, *,
                 out: Callable[[str], None] | None = None,
                 err: Callable[[str], None] | None = None,
                 watchers: list[Callable[[], None]] | None = None,
                 state_path=None,
                 on_clean_exit: Callable[[], None] | None = None):
        self.slots = [Slot(s) for s in specs]
        self._launch = launch
        self._out = out or (lambda m: print(m, flush=True))
        self._err = err or (lambda m: print(m, file=sys.stderr, flush=True))
        self._watchers = watchers or []
        self._state_path = state_path
        self._on_clean_exit = on_clean_exit or (lambda: None)
        self._last_published: tuple | None = None
        self._last_publish_at = -1e9
        self.exit_code = 0

    # -- what happened, in words ------------------------------------------
    def _say(self, msg: str) -> None:
        self._out(f"[keeper] {msg}")

    # -- starting ------------------------------------------------------------
    def start(self) -> None:
        for slot in self.slots:
            self._start(slot, ())
        self._publish()

    def adopt(self, handles: dict[str, Handle],
              bridges: dict[str, str | None] | None = None) -> None:
        """Take over children a dead keeper left running, instead of starting new ones. A
        child with no handle here is started as usual."""
        for slot in self.slots:
            handle = handles.get(slot.spec.name)
            if handle is None:
                self._start(slot, ())
                continue
            slot.handle = handle
            slot.bridge = (bridges or {}).get(slot.spec.name)
            slot.started_at, slot.started_wall = clock.now(), clock.wall()
            slot.last_beat, slot.last_beat_wall = slot.started_at, None
            self._say(f"{slot.spec.name}: adopted, pid {handle.pid}")
        self._publish(force=True)

    def _start(self, slot: Slot, extra: tuple[str, ...]) -> None:
        slot.started_at = clock.now()
        slot.started_wall = clock.wall()
        slot.last_beat, slot.last_beat_wall = slot.started_at, None
        slot.ready, slot.state, slot.bridge = None, STARTING, None
        try:
            slot.handle = self._launch(slot.spec, extra)
        except OSError as exc:
            slot.handle = None
            self._say(f"{slot.spec.name}: could not start ({exc})")
            self._died(slot, code=None)
            return
        self._say(f"{slot.spec.name}: started, pid {slot.handle.pid}")

    # -- one pass of supervision ---------------------------------------------
    def step(self) -> None:
        for watch in self._watchers:
            watch()
        for slot in self.slots:
            if slot.state in (GAVE_UP, EXITED) or slot.handle is None:
                continue
            lines = slot.handle.drain()
            code = slot.handle.poll()
            if code is not None:
                lines += slot.handle.drain()
            self._read_lines(slot, lines)
            if code is not None:
                self._died(slot, code)
            else:
                self._check_beat(slot)
        self._publish()

    def _read_lines(self, slot: Slot, lines: list[str]) -> None:
        for line in lines:
            msg = parse_line(line)
            if isinstance(msg, (Ready, Beat)) and msg.bridge is not None:
                slot.bridge = msg.bridge
            if isinstance(msg, Ready):
                slot.ready, slot.state = msg, UP
                self._beat(slot)
                mark = "?" if msg.mark is None else msg.mark
                self._say(f"{slot.spec.name}: READY mark={mark} fault={msg.fault} "
                          f"pid={msg.pid}")
            elif isinstance(msg, Beat):
                if slot.state == QUIET:
                    self._say(f"{slot.spec.name}: heartbeat is back")
                    slot.state = UP if slot.ready else STARTING
                self._beat(slot)
            elif line:
                self._out(line)

    @staticmethod
    def _beat(slot: Slot) -> None:
        slot.last_beat, slot.last_beat_wall = clock.now(), clock.wall()

    def _check_beat(self, slot: Slot) -> None:
        limit = BEAT_TIMEOUT_S if slot.ready else READY_TIMEOUT_S
        silent_s = clock.now() - slot.last_beat
        if silent_s > limit and slot.state != QUIET:
            slot.state = QUIET
            what = "heartbeat" if slot.ready else "READY"
            self._say(f"{slot.spec.name}: no {what} for {silent_s:.0f} s. It may be hung; "
                      f"the keeper does not kill it.")

    # -- death and restart -----------------------------------------------------
    def _died(self, slot: Slot, code: int | None) -> None:
        name = slot.spec.name
        if code == PORT_TAKEN_EXIT:
            slot.state = EXITED
            self.exit_code = PORT_TAKEN_EXIT
            self._say(f"{name}: its port is taken; not restarting")
            return
        if code in QUIT_CODES:
            slot.state = EXITED
            self._say(f"{name}: exited (code {code}); stopping")
            return
        now = clock.now()
        while slot.restarts and now - slot.restarts[0] > RESTART_WINDOW_S:
            slot.restarts.popleft()
        if len(slot.restarts) >= MAX_RESTARTS:
            slot.state = GAVE_UP
            self.exit_code = EXIT_GAVE_UP
            self._err(f"\n[keeper] {name} died again (code {code}) after {MAX_RESTARTS} "
                      f"restarts in {RESTART_WINDOW_S / 60:.0f} minutes. GIVING UP: "
                      f"{name} is NOT running. Read the messages above, fix the cause, "
                      f"and start afm-litho again.\n")
            return
        slot.restarts.append(now)
        slot.restarts_total += 1
        self._say(f"{name}: died (code {code}); restart {len(slot.restarts)} of "
                  f"{MAX_RESTARTS}")
        self._start(slot, self._restart_args(slot))

    @staticmethod
    def _restart_args(slot: Slot) -> tuple[str, ...]:
        """The spec's restart arguments; with --after-death, also the dead child's last
        bridge state (down when it never said), so only a service that had Igor makes safe."""
        args = slot.spec.restart_args
        if AFTER_DEATH_ARG in args:
            args = (*args, LAST_BRIDGE_ARG, slot.bridge or BRIDGE_DOWN)
        return args

    @property
    def finished(self) -> bool:
        return any(s.spec.essential and s.state in (GAVE_UP, EXITED) for s in self.slots)

    # -- the loop ----------------------------------------------------------------
    def run(self, tick: float = 0.25, *, adopted: bool = False) -> int:
        """Supervise until finished. `adopted`: the children were taken over (`adopt`)
        before this call, not to be started."""
        try:
            if not adopted:
                self.start()
            while not self.finished:
                self.step()
                if not self.finished:
                    clock.sleep(tick)
        except KeyboardInterrupt:
            self._on_clean_exit()           # before anything else: a normal exit is not revived
            self.shutdown()
        if self.exit_code == 0:
            self._on_clean_exit()
        self._publish(force=True)
        return self.exit_code

    def shutdown(self) -> None:
        """Ctrl+C at the keeper's console. The children share the console and got it
        too; wait for them to exit in their own time, as the single process always did.
        A second Ctrl+C ends any that remain."""
        self._say("stopping; waiting for the children (Ctrl+C again to force)")
        try:
            while any(self._alive(s) and getattr(s.handle, "owned", True) for s in self.slots):
                clock.sleep(0.25)
        except KeyboardInterrupt:
            for slot in self.slots:
                if self._alive(slot) and getattr(slot.handle, "owned", True):
                    self._say(f"{slot.spec.name}: forced to end")
                    slot.handle.terminate()
        for slot in self.slots:
            if slot.state not in (GAVE_UP, EXITED):
                slot.state = EXITED

    @staticmethod
    def _alive(slot: Slot) -> bool:
        return slot.handle is not None and slot.handle.poll() is None

    # -- the state file `keeper status` reads ------------------------------------
    def snapshot(self) -> dict:
        return {
            "children": [{
                "name": s.spec.name,
                "pid": s.handle.pid if s.handle else None,
                "state": s.state,
                "started_wall": s.started_wall,
                "last_beat_wall": s.last_beat_wall,
                "restarts_in_window": len(s.restarts),
                "restarts_total": s.restarts_total,
                "ready": ({"mark": s.ready.mark, "fault": s.ready.fault, "pid": s.ready.pid}
                          if s.ready else None),
                "bridge": s.bridge,
                # how to start it again; a revived keeper builds its specs from these
                "launch": {"argv": s.spec.argv, "env": s.spec.env, "essential": s.spec.essential,
                           "restart_args": list(s.spec.restart_args)},
            } for s in self.slots]}

    def _publish(self, force: bool = False) -> None:
        if self._state_path is None:
            return
        snap = self.snapshot()
        key = [(c["pid"], c["state"], c["restarts_total"], c["ready"]) for c in snap["children"]]
        now = clock.now()
        if not force and key == self._last_published and now - self._last_publish_at < 1.0:
            return
        try:
            status.write_state(self._state_path, snap)
        except OSError:
            return              # a reader holds the file; the next step writes it again
        self._last_published, self._last_publish_at = key, now
