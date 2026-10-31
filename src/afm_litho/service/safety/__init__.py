"""Safety kernel: the pure guard logic, with no hub and no instrument
(docs/SAFETY_KERNEL_PLAN.md is the architecture).

  * `RailDwellGuard` / `RailDwellLimits` (rail_dwell.py) -- the Z-drive rail-dwell
    telemetry watchdog (hazard #23), the kernel's first concrete entry.
  * `WriteInterlocks` (write_interlocks.py) -- the rail-dwell, SUM-collapse and
    deflection-integrity guards armed during every hardware engage.
  * stop_pairs.py -- pairs a door's Stop with the keeper's copy of it.

The code that acts on the tip (safe_park, engage, the stop port, the after-death
make-safe) is in service/host/.
"""

from __future__ import annotations

from .rail_dwell import RailDwellGuard, RailDwellLimits

__all__ = ["RailDwellGuard", "RailDwellLimits"]
