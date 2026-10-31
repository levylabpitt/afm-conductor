"""Each recipe that lifts the tip, as its own named function. Not merged: the callers
differ on purpose. The out-of-band trigger stays `bridge_router.send_safe_park`.

Every STOP (abort, park) goes through Igor's one abort function, AFM_SafePark
(afm_litho_run.ipf): td_Stop once per abort, bias 0, loop 2 re-armed at -10 V with
OutputMin 0. On the Cypher only that re-arm retracts; a Height ramp does not move Z (rig
2026-10-27, docs/guides/INSTRUMENT_CONFIG.md section 6). Its flag stays set until the host
confirms the retract (Height < 1 V), so a repeat abort skips td_Stop and cannot kill the
retract loop. Routine withdraws (transit, end of run, plain Withdraw) are the bare loop-2
re-arm, `withdraw_rearm`, with no td_Stop.
"""

from __future__ import annotations

import logging
import math

from .channels import HEIGHT

_LOG = logging.getLogger(__name__)

# Height under this counts as retracted (2026-10-27). Height is the commanded Z the
# meter shows as "Z Voltage"; the withdraw parks it at its 0 V OutputMin.
RETRACTED_HEIGHT_V = 1.0


def twin_abort(arc) -> None:
    """The twin's model of Igor's abort function (`SoftwareARC.abort_retract`). On hardware
    the out-of-band trigger runs the real one; the park sends nothing of its own."""
    arc.abort_retract()


def withdraw_rearm(arc, z_out: str, *, loop: int, defl_channel: str, setpoint: float,
                   igain: float, out_min: float, out_max: float) -> None:
    """The plain Withdraw, on both platforms: re-arm loop 2 at -10 with OutputMin 0, so
    Height parks at 0 V. Stopping the loop and ramping does not retract a Cypher, and the
    same order was chosen for the MFP-3D (2026-10-27)."""
    arc.ir_SetPISLoop(loop, "Always,Never", defl_channel, setpoint,
                      0.0, igain, 0.0, z_out, out_min, out_max)


def is_retracted(height_v: float | None) -> bool:
    """Height read under 1 V. A failed read (None, NaN) is never a retract."""
    return (isinstance(height_v, (int, float)) and not math.isnan(height_v)
            and height_v < RETRACTED_HEIGHT_V)


def confirm_retract(arc) -> None:
    """Tell Igor the retract is confirmed (ABORTCONFIRM), so its next abort calls td_Stop
    again and its 5 s backup stands down. Best effort: a lost confirm leaves the backup
    to clear the flag itself."""
    try:
        arc.confirm_retract()
    except Exception as exc:                              # noqa: BLE001
        _LOG.warning("abort confirm failed: %s", exc)


def vendor_withdraw(arc) -> None:
    """The vendor Withdraw button. A failure is logged, so the re-verify reports it."""
    try:
        arc.vendor_withdraw()
    except Exception as exc:                              # noqa: BLE001
        _LOG.warning("safe_park vendor_withdraw failed: %s", exc)


def read_height_over_com(com_igor) -> float | None:
    """Height read through COM Execute2 (`td_ReadValue(HEIGHT)` printed by fprintf), for a
    service with no bridge socket. None on any failure: a failed read is never a retract."""
    try:
        r = com_igor.execute(f'fprintf 0, "%.15g", (td_ReadValue("{HEIGHT}"))', check=True)
        v = float(str(r.results).strip())
    except Exception as exc:                              # noqa: BLE001
        _LOG.warning("Height read over COM failed: %s", exc)
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def vendor_withdraw_over_com(com_igor) -> None:
    """`DoScanFunc("StopEngageButton")`. Raises without a COM handle, never a silent no-op."""
    if com_igor is None:
        raise RuntimeError(
            "vendor_withdraw: no COM Igor handle attached; cannot fire "
            "the vendor withdraw out-of-band")
    com_igor.execute('DoScanFunc("StopEngageButton")', check=True)
