"""Meta / root endpoints: the panel page and the twin's live state."""

from __future__ import annotations

import re

import numpy as np

from ..deps import WEB
from ..hub import hub


def _is_ac() -> bool:
    """Amplitude is meaningful only in AC mode. Shared with the /ws heartbeat."""
    from ..host.telemetry import current_scan_is_ac
    return current_scan_is_ac()


def _cache_busted_html(filename: str) -> str:
    """Read a page from WEB and stamp every /static asset it references with ?v=<mtime>.

    Assets are discovered from the markup (flat /static/<name>.js|css only), so a
    new file is cache-busted automatically. ES sub-imports are not in the markup;
    they rely on the /static mount's Cache-Control: no-cache.
    """
    html = (WEB / filename).read_text("utf-8")
    for name in sorted(set(re.findall(r'/static/([A-Za-z0-9_.-]+\.(?:js|css))', html))):
        f = WEB / name
        if f.exists():
            html = html.replace(f"/static/{name}",
                                f"/static/{name}?v={int(f.stat().st_mtime)}")
    return html


async def index():
    """`/` goes to the cockpit; the old main window is no longer served."""
    from fastapi.responses import RedirectResponse
    from ...site import COCKPIT_PATH
    return RedirectResponse(COCKPIT_PATH, status_code=307)


async def edit_cockpit():
    """The edit cockpit, through the same cache-busting helper as index."""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(_cache_busted_html("edit-cockpit.html"))


async def state():
    """GET /api/state: the service's live state, read from the board."""
    from .. import board
    return board.snapshot(hub)


async def set_speed(factor: float):
    hub.host.speed = max(0.25, min(64.0, factor))
    return {"speed": hub.host.speed}


async def optics():
    """Coarse basemap for the simulated CCD view: full scanner range."""
    n = 160
    half = hub.host.instrument.xy_range_um * 1e-6 / 2
    ax = np.linspace(-half, half, n)
    gx, gy = np.meshgrid(ax, ax, indexing="xy")
    h = hub.host.world.sample.height(gx.ravel(), gy.ravel()).reshape(n, n)
    h = h - h.mean()
    lo, hi = np.percentile(h, 2), np.percentile(h, 98)
    img = np.clip((h - lo) / (hi - lo + 1e-12), 0, 1)
    return {"n": n, "range_um": hub.host.instrument.xy_range_um,
            "gray": (img * 255).astype(int).ravel().tolist()}
