"""The two lines a child tells the keeper, kept in its beat file (keeper/beatfile.py).

    READY mark=<0|1|?> fault=<0|1> pid=<n> [bridge=up|down]
                                                when the child is up, and again when
                                                its instrument link comes up
    HEARTBEAT pid=<n> [bridge=up|down]          every second after the first READY

`mark` is Igor's stop mark (root:AFM_abortStopped); `?` means the child has not read it
(a hardware service before Igor dials in, or procedures older than wire v3). `fault` is
the service's latched fault, which includes Igor's (root:AFM_abortFault) once read.
`bridge` is whether the service has a live bridge to Igor. The keeper keeps the last one
and passes it to a restart (`--last-bridge`): only a service that had Igor makes safe after
its death (2026-10-28: before the bridge is up, nothing is sent to Igor).
"""

from __future__ import annotations

from dataclasses import dataclass

HEARTBEAT_EVERY_S = 1.0

# Exit code of a service that found its port in use. Restarting cannot fix it.
PORT_TAKEN_EXIT = 3

# Passed to a restarted service, with LAST_BRIDGE_ARG=<up|down>: the service that died
# had (or had not) a live bridge to Igor.
AFTER_DEATH_ARG = "--after-death"
LAST_BRIDGE_ARG = "--last-bridge"
BRIDGE_UP, BRIDGE_DOWN = "up", "down"

# The keeper's own pid, given to each child, which watches it (service/keeper_link.py).
KEEPER_PID_ENV = "AFM_LITHO_KEEPER_PID"


@dataclass(frozen=True)
class Ready:
    mark: int | None        # None: not read by the child
    fault: int
    pid: int
    bridge: str | None = None   # "up" | "down"; None: the child does not say


@dataclass(frozen=True)
class Beat:
    pid: int
    bridge: str | None = None


def _bridge_word(bridge: str | None) -> str:
    return "" if bridge is None else f" bridge={bridge}"


def ready_line(mark: int | None, fault: int, pid: int, bridge: str | None = None) -> str:
    return (f"READY mark={'?' if mark is None else mark} fault={fault} pid={pid}"
            f"{_bridge_word(bridge)}")


def heartbeat_line(pid: int, bridge: str | None = None) -> str:
    return f"HEARTBEAT pid={pid}{_bridge_word(bridge)}"


def parse_line(line: str) -> Ready | Beat | None:
    """A Ready or Beat, or None for any other (or malformed) line."""
    words = line.split()
    if not words:
        return None
    try:
        kv = dict(w.split("=", 1) for w in words[1:] if "=" in w)
        if words[0] == "READY":
            mark = kv["mark"]
            return Ready(None if mark == "?" else int(mark), int(kv["fault"]), int(kv["pid"]),
                         _bridge(kv))
        if words[0] == "HEARTBEAT":
            return Beat(int(kv["pid"]), _bridge(kv))
    except (KeyError, ValueError):
        return None
    return None


def _bridge(kv: dict) -> str | None:
    b = kv.get("bridge")
    return b if b in (BRIDGE_UP, BRIDGE_DOWN) else None
