"""The service's hold on the twin plugin (instrument/twin.py): one instance per sample.

The twin write and the twin scan run on this instance, so a wire written stays in the
next scan. The plugin's telemetry is the cockpit's tip display on the twin: each frame goes
out as a `tip_pos` message. A new sample (the cockpit's sample menu) is converted to the
plugin's `.npz` and starts a new instance.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from pathlib import Path

import numpy as np

from .. import clock
from ..instrument.twin import TwinInstrument
from .hub import real_bridge
from .tip_stream import tip_pos_msg

# A twin position is already lab micrometres; this calibration makes `tip_pos_msg`'s
# volts-to-um conversion the identity, so the twin's frame has the hardware frame's shape.
UM_CAL = {"x_lvdt_sens": 1e-6, "x_lvdt_offset": 0.0,
           "y_lvdt_sens": 1e-6, "y_lvdt_offset": 0.0}


def sample_to_npz(sample, dest: Path) -> Path:
    """Write the twin's sample file from a loaded IBW sample.

    `height` is rows (Y) by pixels (X) in nm, `size_um` the image width, centred on the
    origin at angle 0 (the sample sits at the centre of the scanner range).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    height_nm = np.asarray(sample.height_map, dtype=float).T * 1e9    # the map is indexed [x, y]
    extra = {}
    retrace_map = getattr(sample, "retrace_map", None)
    if retrace_map is not None:
        extra["retrace"] = np.asarray(retrace_map, dtype=float).T * 1e9
    for key, grid in (getattr(sample, "telemetry_maps", None) or {}).items():
        extra[key] = np.asarray(grid, dtype=float).T
    np.savez(dest, height=height_nm, size_um=float(sample.extent) * 1e6, x_offset_um=0.0,
             y_offset_um=0.0, angle_deg=0.0, **extra)
    return dest


def twin_only(hub) -> bool:
    """True only on a twin station with no real bridge object bound.

    Stricter than `hub.backend_live() == "twin"`, which is also "twin" for a hardware
    station whose bridge has not connected or has closed: there the real tip position
    is unknown and a simulated marker would be a false reading."""
    return getattr(hub.host, "backend", None) != "hardware" and real_bridge(hub) is None


def _stop_signalled(hub):
    def stop_signalled(verb: str) -> bool:
        event = hub.host._hw_litho_abort if verb == "write" else hub.host._hw_abort
        return event.is_set()
    return stop_signalled


def _send_tip(hub):
    def on_telemetry(frame) -> None:
        loop = getattr(hub, "_loop", None)
        if loop is None or loop.is_closed() or not hub.clients or not twin_only(hub):
            return
        if frame.x_um != frame.x_um or frame.y_um != frame.y_um:      # NaN: not read
            return
        message = tip_pos_msg(frame.x_um, frame.y_um, UM_CAL)
        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(hub.broadcast(message), loop)
    return on_telemetry


_PLUGIN_LOCK = threading.Lock()    # the scan pump and a caller can both call plugin() at once


def plugin(hub) -> TwinInstrument:
    """The twin plugin for the loaded sample, built (and listening) on first use. A plugin
    made for another sample object (a sample swapped in without the sample menu) is dropped
    and rebuilt, so a scan never reads a stale image."""
    sample = hub.host.world.sample
    with _PLUGIN_LOCK:
        if hub.host._twin_plugin is not None and getattr(hub.host, "_twin_plugin_sample", None) is not sample:
            forget(hub)
        if hub.host._twin_plugin is None:
            path = sample_to_npz(sample, hub.store.path.parent / "twin" / f"{hub.sample_name}.npz")
            twin = TwinInstrument(sample=path, should_stop=_stop_signalled(hub))
            twin.listen(_send_tip(hub), lambda status: None)
            hub.host._twin_plugin, hub.host._twin_plugin_sample = twin, sample
    return hub.host._twin_plugin


def twin_snapshot(hub, snap: dict) -> dict:
    """The old twin's snapshot with the plugin's truth laid over it once the plugin has run
    a write or scan: where the tip is, and whether it is down. (The old twin still describes
    its own Z, deflection and amplitude.)"""
    twin = getattr(hub.host, "_twin_plugin", None)
    if twin is not None and twin_only(hub):
        snap["x_um"], snap["y_um"] = twin.position_at(clock.now())
        snap["engaged"] = not twin.withdrawn()
    return snap


def forget(hub) -> None:
    """Drop the plugin (a new sample, or a hardware bridge bound): the next use rebuilds it."""
    twin, hub.host._twin_plugin = hub.host._twin_plugin, None
    hub.host._twin_plugin_sample = None
    hub.host.twin_scan_angle_deg = None
    if twin is not None:
        # A scan or write still running on the dropped twin would otherwise play out its
        # whole simulated length where no Stop can reach it.
        twin.abort("twin plugin dropped")
        twin.close()


def abort(hub, reason: str) -> None:
    """Stop whatever the plugin is doing and withdraw its tip. Sends and returns; never raises."""
    twin = getattr(hub.host, "_twin_plugin", None)
    if twin is not None:
        with contextlib.suppress(Exception):
            twin.abort(reason)


def owns_tip_display(hub) -> bool:
    """True once the plugin has been used: its telemetry is then the only tip source."""
    return getattr(hub.host, "_twin_plugin", None) is not None
