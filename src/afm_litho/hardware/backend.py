"""Backend selection: build the app's ARC backend.

One app with a swappable `ARCBackend` seam: `SoftwareARC` over the physics
`world` off the instrument, `HardwareARC` over a connected bridge `link` on it.

Env precedence: AFM_LITHO_ARC_BACKEND > AFM_LITHO_BACKEND > "sim". Unrecognized values fall back to sim with one warning, never
an error. Stays free of service imports; `service/backends.py` re-exports it.
"""

from __future__ import annotations

import logging
import os

from .arc_backend import ARCBackend

_log = logging.getLogger(__name__)

SIM = "sim"
HARDWARE = "hardware"
_HARDWARE_ALIASES = {HARDWARE, "hw", "real", "igor"}
# "twin": the flex-afm integration tests set AFM_LITHO_BACKEND=twin.
_SIM_ALIASES = {SIM, "twin"}
_KNOWN_BACKENDS = _HARDWARE_ALIASES | _SIM_ALIASES

_BASE_VAR = "AFM_LITHO_BACKEND"
_ARC_VAR = "AFM_LITHO_ARC_BACKEND"

# (source_var, value) pairs already warned about; keeps each to one warning ever.
_warned: set[tuple[str, str]] = set()


def is_hardware(value: str | None) -> bool:
    """True if `value` is a hardware alias (hardware/hw/real/igor), case- and
    whitespace-insensitive. None, "" and "twin" are not hardware."""
    return (value or "").strip().lower() in _HARDWARE_ALIASES


def backend_mode(default: str = SIM) -> str:
    """Legacy: env AFM_LITHO_BACKEND only, lowercased. New code uses
    `arc_backend()`."""
    return os.environ.get(_BASE_VAR, default).strip().lower()


def _resolve(var: str, default: str) -> str:
    """`var`, else AFM_LITHO_BACKEND, else `default`, lowercased.

    Hardware aliases normalize to "hardware" because the downstream gates
    (app.py bridge bring-up, `Hub.require_live_backend`, `Hub.backend_label`)
    compare `== "hardware"` literally (plan Sec C.2 silent-twin hazard).
    Anything else becomes "sim"; an unrecognized value warns once, blamed on the
    variable it actually came from (a bad BASE value reached via ARC is
    BASE's fault)."""
    base_raw = os.environ.get(_BASE_VAR)
    base = (base_raw if base_raw is not None else default).strip().lower()
    raw = os.environ.get(var)
    value = (raw if raw is not None else base).strip().lower()
    if value in _HARDWARE_ALIASES:
        return HARDWARE
    if value not in _SIM_ALIASES:
        source_var = var if raw is not None else _BASE_VAR
        key = (source_var, value)
        if key not in _warned:
            _warned.add(key)
            _log.warning(
                "%s=%r is not a recognized backend (expected one of %s); "
                "treating it as the simulation backend",
                source_var, value, sorted(_KNOWN_BACKENDS),
            )
    return SIM


def arc_backend(default: str = SIM) -> str:
    """AFM_LITHO_ARC_BACKEND, else AFM_LITHO_BACKEND, else `default`.
    Selects the Igor/ARC driver (`make_arc`, bridge bring-up, `Hub.backend`)."""
    return _resolve(_ARC_VAR, default)


def make_arc(mode: str | None = None, *, world=None, link=None) -> ARCBackend:
    """Build the ARC backend: the simulation needs `world`, the real backend a
    connected bridge `link`."""
    mode = (mode or arc_backend()).lower()
    if is_hardware(mode):
        if link is None:
            raise ValueError("the real backend needs a connected bridge `link`")
        from .hardware_arc import HardwareARC
        return HardwareARC(link)
    if world is None:
        raise ValueError("the simulation backend needs a physics `world`")
    from ..twin.arc import SoftwareARC
    return SoftwareARC(world)
