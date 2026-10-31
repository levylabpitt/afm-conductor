"""The Surface model's start-up replay: bring it back from the raw passes on disk.

The main panel no longer has a Surface view and this module has no routes. The model
(`hub.surface`) still takes in every scan and write pass, and `AFM_LITHO_REBUILD_SURFACE=1`
replays the raw passes at start-up through `schedule_rebuild`.
"""

from __future__ import annotations

import asyncio

from ..hub import hub

# The single in-flight Surface rebuild for this process. `SurfaceModel.reload`
# replays every raw pass while `_replaying` is set (surface.py), a plain bool
# cleared in a `finally`. Two overlapping rebuilds would race on that
# flag: the first to finish clears it, the other's mid-replay ingest
# resumes writing npz, and every pass fuses twice. So a rebuild is scheduled
# through `schedule_rebuild` only, which refuses to start a second while one is
# still running.
_rebuild_task: asyncio.Task | None = None


def rebuild_in_flight() -> bool:
    """True while a scheduled Surface rebuild has not finished."""
    return _rebuild_task is not None and not _rebuild_task.done()


def schedule_rebuild() -> bool:
    """Schedule `_rebuild_surface` unless one is already running.

    Returns True if a rebuild was scheduled, False if one was already in flight
    (the caller must not start a second -- see the module note above).
    """
    global _rebuild_task
    if rebuild_in_flight():
        return False
    _rebuild_task = asyncio.create_task(_rebuild_surface())
    return True


async def _rebuild_surface():
    """Fuse the raw passes already on disk (runtime/raw/*.npz) back into the Surface model.

    Runs in the background: replaying a full raw directory takes tens of seconds.
    """
    surf = getattr(hub, "surface", None)
    if surf is None or not hasattr(surf, "reload"):
        return
    try:
        n = await asyncio.to_thread(surf.reload)
    except Exception as exc:  # noqa: BLE001 - logged; a rebuild failure must not crash the handler
        await hub.log(f"Surface: could not rebuild from raw passes ({exc})")
        return
    if n:
        await hub.log(f"Surface: {n} earlier passes restored from disk.")
        await hub.broadcast({"type": "surface_changed"})
