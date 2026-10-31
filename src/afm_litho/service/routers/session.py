"""Session archive: list recorded scans/writes and recall one onto the canvas."""

from __future__ import annotations

import asyncio

from ..hub import hub


async def sessions_list():
    local = hub.store.list()
    remote = []
    if hub.llab.available():
        try:
            remote = await asyncio.to_thread(hub.llab.recent_windows)
        except Exception:  # noqa: BLE001 - remote history is optional
            remote = []
    return {"local": local, "remote": remote,
            "remote_available": hub.llab.available()}


async def sessions_recall(id: str):
    if id.startswith("llab-"):
        paths = await asyncio.to_thread(hub.llab.window_paths, id[5:])
        return {"id": id, "kind": "llab_079", "written": paths}
    rec = hub.store.get(id)
    return rec or {"error": "unknown session"}
