"""The ARC backend seam.

A pass executor drives a *controller* through the `td_`/`ir_` surface below. Two
things satisfy this Protocol:

  * `twin.arc.SoftwareARC` -- simulates each call against the physics model, and
    advances the simulation with `step(n)`.
  * a bridge-backed `HardwareARC` (to build) -- translates each call into
    `bridge_protocol` commands to real Igor, which runs the armed waves
    autonomously and streams the capture back.

The seam is deliberately **arm-and-fire, not per-tick**: it has no `step()`. The
executor arms output waves and an input-wave capture on a shared event, fires the
event, then reads the captured samples back. On the twin, "fire + await" is
`step(n)` internally; on hardware, it's `ARMFIRE` + awaiting `DONE`/`CAPTURE`.
Because `step()` is the one method with no hardware counterpart, keeping it out of
this Protocol is what lets a single executor drive either backend unchanged.

This module is the interface contract only. `SoftwareARC` already conforms (see
tests/test_arc_backend.py). The concrete `HardwareARC` and the unified executor
that drives this seam are the next pieces (see docs/history/2026-08-19-PASS_MODEL_BUILD_LOG.md).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ARCBackend(Protocol):
    """The controller surface a pass executor depends on. No `step()`."""

    # scalars ---------------------------------------------------------------
    def td_ReadValue(self, name: str) -> float: ...
    def td_WriteValue(self, name: str, value: float) -> int: ...
    def td_ReadString(self, name: str) -> str: ...
    def td_WriteString(self, name: str, value: str) -> int: ...

    # servo loops -----------------------------------------------------------
    def ir_SetPISLoop(self, index: int, events: str, input_: str, setpoint: float,
                      pgain: float, igain: float, sgain: float, output: str,
                      out_min: float, out_max: float) -> int: ...
    def ir_StopPISLoop(self, index: int) -> int: ...

    # output waves (drive) --------------------------------------------------
    def td_xSetOutWave(self, bank: int, event: str, channel: str,
                       wave: np.ndarray, interpolation: int) -> int: ...
    def td_xSetOutWavePair(self, bank: int, event: str, ch_a: str, wave_a: np.ndarray,
                           ch_b: str, wave_b: np.ndarray, interpolation: int) -> int: ...

    # input waves (capture) -------------------------------------------------
    def td_xSetInWave(self, bank: int, event: str, channel: str,
                      n_points: int, decimation: int) -> int: ...
    def td_xSetInWavePair(self, bank: int, event: str, ch_a: str, ch_b: str,
                          n_points: int, decimation: int) -> int: ...

    # arm-and-fire ----------------------------------------------------------
    def fire(self, event: str, ticks: int) -> dict:
        """Fire `event` and return the filled input-wave capture as
        {channel: samples}. The twin steps `ticks` samples internally; a bridge
        backend fires and awaits CAPTURE/DONE, ignoring `ticks`."""
        ...

    # stop / abort ----------------------------------------------------------
    def td_StopOutWaveBank(self, bank: int) -> int: ...
    def td_StopInWaveBank(self, bank: int) -> int: ...
    def td_Stop(self) -> int: ...
