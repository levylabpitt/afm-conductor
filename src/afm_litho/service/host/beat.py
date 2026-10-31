"""The host's own heartbeat: it stamps the host alive, so a silent host reads stale.

In-process it stands for the host's periodic snapshot frames (0.25 s at H8). The base
never stamps the host; only host frames and this beat do (`HostState.heard`).
"""
from __future__ import annotations

import asyncio

HOST_BEAT_S = 0.5
_beat: asyncio.Task | None = None


async def _beat_loop(host) -> None:
    """Separate from the control tick, which can await a ~60 s withdraw; only a frozen
    event loop stops this. Real time on every clock mode: it measures real loop
    liveness, so it must not advance virtual time."""
    while True:
        host.heard()
        await asyncio.sleep(HOST_BEAT_S)


def start_host_heartbeat(host) -> None:
    """Stamp now (silence is measured from startup) and keep stamping. Idempotent."""
    global _beat
    host.heard()
    if _beat is None or _beat.done():
        _beat = asyncio.get_running_loop().create_task(_beat_loop(host))


def stop_host_heartbeat() -> None:
    global _beat
    if _beat is not None and not _beat.done():
        _beat.cancel()
    _beat = None
