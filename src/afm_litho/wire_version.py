"""Which wire the service speaks: AFM_LITHO_WIRE_VERSION, "2" (default) or "3".

Standard library only, so the keeper can check it before starting the service.
"""

from __future__ import annotations

import os

ENV = "AFM_LITHO_WIRE_VERSION"


def from_env(environ=None) -> int:
    """2 when unset or empty; 3 for "3"; anything else is refused with a clear message."""
    raw = (os.environ if environ is None else environ).get(ENV, "").strip()
    if raw == "":
        return 2
    if raw in ("2", "3"):
        return int(raw)
    raise ValueError(f'{ENV} must be "2" or "3", not {raw!r}')
