"""Which wire the service speaks to Igor: v2 (today's) or v3 (phase-5-wire-v3-contract.md).

The operator picks it on the rig with the env var AFM_LITHO_WIRE_VERSION (2 or 3). Under 3 the pass drivers send the same requests, and
`wire_v3` carries them as v3 lines, so Igor's final td_/ir_ commands do not change.
HardwareARC reads this when it is built.
"""

from __future__ import annotations

import contextlib

from ..wire_version import from_env

# Read once at import from AFM_LITHO_WIRE_VERSION; a bad value stops the service here.
WIRE_VERSION = from_env()


@contextlib.contextmanager
def use_version(version: int):
    """Set WIRE_VERSION for the block (tests and the golden harness), then restore it."""
    global WIRE_VERSION
    if version not in (2, 3):
        raise ValueError(f"wire version must be 2 or 3, not {version!r}")
    before = WIRE_VERSION
    WIRE_VERSION = version
    try:
        yield
    finally:
        WIRE_VERSION = before
