"""The digital twin's tip-position stream: `tip_pos` frames from the twin's own position.

On hardware the Igor TELEM poll is the only producer of `tip_pos`
(host/telemetry._hw_telemetry_frame). The twin has no poll, so this loop task
broadcasts the same message (built by `tip_stream.tip_pos_msg`) from
`hub.arc.snapshot()`: about TWIN_FRAME_HZ while something is moving the tip, about
REST_HZ at rest so the cockpit marker is always present. Once the twin plugin has run a
write or scan, its telemetry (service/twin_plugin.py) is the tip display and this loop
stands down.

DISPLAY ONLY. It reads the twin's position and writes nothing to it, so telemetry,
state, records, captures and the stroke physics are untouched.

TWIN ONLY. `twin_only(hub)` is false on a hardware-configured station (bridge or no
bridge) and whenever a real bridge object is bound, live or closed. The task is not
created on a hardware station, ends itself the first time the check fails, and
re-checks immediately before each send.

The service has no display switch for it: on the twin it always streams, and each
cockpit page shows or hides the marker itself ("show tip").
"""

from __future__ import annotations

import asyncio
import contextlib
import time

from .passes import TWIN_FRAME_HZ
from .tip_stream import tip_pos_msg
from .twin_plugin import UM_CAL as _UM_CAL
from .twin_plugin import owns_tip_display, twin_only

REST_HZ = 1.5

# The twin's position is already lab microns; `tip_pos_msg` takes sensor volts and an
# LVDT calibration. This calibration makes that conversion the identity, so the frame
# goes through the same builder (and rounding) as the hardware frame.


def position_um(hub) -> tuple[float, float]:
    """The old twin's own position (`hub.arc`), which approach and withdraw still move."""
    snap = hub.host.arc.snapshot()
    return float(snap["x_um"]), float(snap["y_um"])


def frame(hub, now: float | None = None, at_um=None) -> dict | None:
    """The `tip_pos` message to send now, or None when nothing may be sent (a hardware
    station or a bound bridge). `at_um` is a position already read by the caller;
    without it the display position is read."""
    if not twin_only(hub):
        return None
    x_um, y_um = position_um(hub) if at_um is None else at_um
    return tip_pos_msg(x_um, y_um, _UM_CAL)


async def send(hub, now: float | None = None, at_um=None) -> bool:
    """Broadcast one frame if allowed. Returns whether a frame went out."""
    msg = frame(hub, now, at_um)
    if msg is None or not hub.clients:
        return False
    hub._twin_tip_sent_at = time.monotonic()
    await hub.broadcast(msg)
    return True


def _operation_running(hub) -> bool:
    if hub.state_label() == "scanning":
        return True
    return bool(getattr(hub.host, "_approach_in_flight", 0))


async def run(hub) -> None:
    """The stream loop. Ticks at TWIN_FRAME_HZ; sends every tick while a scan or an
    approach runs or the twin's position (x, y or z) changed since the last tick, which covers
    withdraw and park, and otherwise once per 1/REST_HZ. A tick is skipped when
    another sender (a scan line) sent within the last half period, so those add to
    the stream's rate only where they are faster than it.

    No tick is sent while a write is in flight: the write worker is then the one
    source of positions, in order with its progress frames, and a tick read from
    the twin here could reach the page ahead of an older frame still queued."""
    period = 1.0 / TWIN_FRAME_HZ
    rest_period = 1.0 / REST_HZ
    next_t = time.monotonic()
    last_pose = None
    while True:
        # Deadline scheduling: Windows timers round a sleep up by as much as 15 ms and
        # a busy write worker delays the wake-up further; a plain sleep(period) would
        # accumulate both into a slower rate. More than one period late, start over
        # rather than send a burst.
        next_t += period
        now = time.monotonic()
        if next_t < now - period:
            next_t = now
        await asyncio.sleep(max(0.0, next_t - now))
        try:
            # Ends for good rather than idling: a bridge was bound to this process,
            # and nothing afterwards should be able to bring a simulated marker back.
            if not twin_only(hub):
                return
            now = time.monotonic()
            snap = hub.host.arc.snapshot()
            pose = (snap["x_um"], snap["y_um"], snap["z_um"])
            moved = last_pose is not None and pose != last_pose
            last_pose = pose
            since = now - getattr(hub, "_twin_tip_sent_at", 0.0)
            if owns_tip_display(hub):
                continue
            busy = moved or _operation_running(hub)
            if (busy and since >= period / 2) or since >= rest_period:
                await send(hub, now)
        except asyncio.CancelledError:
            raise
        except Exception:                                # noqa: BLE001,S110
            pass        # a display stream must never take the loop down with it


def start(hub) -> asyncio.Task | None:
    """Start the stream on the running loop. No task is created on a hardware station
    or with a real bridge bound."""
    stop(hub)
    if not twin_only(hub):
        return None
    hub._twin_tip_task = asyncio.get_running_loop().create_task(
        run(hub), name="twin-tip-stream")
    return hub._twin_tip_task


def stop(hub) -> None:
    task = getattr(hub, "_twin_tip_task", None)
    hub._twin_tip_task = None
    if task is not None and not task.done():
        # the task may belong to a loop that has already closed
        with contextlib.suppress(Exception):
            task.cancel()
