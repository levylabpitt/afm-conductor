"""The service layer's one front door onto backend-mode config.

`arc_backend()` reads `AFM_LITHO_ARC_BACKEND` first, falling back to
`AFM_LITHO_BACKEND`, then to "sim". Vocabulary: "sim"/"twin" are both the
simulation backend; "hardware"/"hw"/"real"/"igor" are the real one; anything
else unrecognized is ALSO treated as sim (one WARNING logged, never an error)
-- see `hardware.backend`'s module docstring for why that leniency matters.

The actual env resolution lives in `hardware.backend` (imported here, not
duplicated) because `hardware.backend.make_arc` needs it internally and the
hardware layer stays free of a service import (see `Hub.bind_hardware`'s note
on that boundary). Service-side consumers import it from here.
"""

from __future__ import annotations

from ..hardware.backend import (
    HARDWARE,
    SIM,
    arc_backend,
    is_hardware,
)

__all__ = [
    "HARDWARE",
    "SIM",
    "arc_backend",
    "is_hardware",
]
