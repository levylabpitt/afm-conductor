"""The litho domain: load a design, frame it, preview the toolpath, write it.

This is the twin's reason for existing: turn a design into tip motion that
doses the surface, and report the write as it happens. `commands.litho_cmds`
starts a worker thread that runs the pattern and posts progress into a queue;
the async `_litho_pump` drains that queue onto the WebSocket, fuses each
finished pass into the Surface.

WHAT IS AND IS NOT IN THIS FILE. The three write ENDPOINTS (`/api/litho/load`,
`/api/litho/write`, `/api/litho/abort`) are thin adapters over
`service/commands/litho_cmds.py` -- the same coroutines the wire's
`loadPattern` / `startWrite` / `abortWrite` call (plan Sec C). The streaming
machinery those coroutines spawn (the two workers, the pump, and the persistence the
pump does at the end of a pass) lives in `service/host/litho_worker.py`.

The transit endpoints live here too: transit is how the tip gets from one
object to the next, so its policy is part of the write.
"""

from __future__ import annotations

import asyncio

import numpy as np

from ...hardware import channels
from ..host import litho_cmds
from ..host.api import HostApi
from ..commands.errors import CommandError
from ..host.litho_worker import _litho_frame, _transit_policy
from ..hub import hub



# ------------------------------------------------------------------ adapters --
# `/api/litho/load`, `/api/litho/write` and `/api/litho/abort` are THIN ADAPTERS over
# `service/commands/litho_cmds.py` -- the same coroutines the wire's `loadPattern`,
# `startWrite` and `abortWrite` call, so the cockpit and a FLEX script meet one
# contact-only guard, one silent-twin refusal and one single-writer claim (plan Sec C).
#
# An adapter's only job is the HTTP dialect. This service answers HTTP 200 with
# `{"error": ..., "reason": ...}` bodies rather than status codes, and a large part of
# the suite asserts those bodies verbatim -- they are the regression guard that the
# extraction did not change what the cockpit sees. `CommandError.http_body()` carries
# the exact legacy shape from the raise site.


async def load_design(source, allow_paths: bool):
    """The cockpit's load, refusals in the legacy HTTP-200 `{"error": ...}` shape."""
    try:
        res = await litho_cmds.load_pattern(hub, source, allow_paths=allow_paths)
    except CommandError as exc:
        return exc.http_body()
    return res["design"]


async def transit_get():
    """The transit policy, and what it would cost on the current pattern.

    The cost matters: `lift` protects the surface but pays a z cycle per
    gap, `slew` is quick but drags the tip. Showing seconds and peak speed
    next to the choice makes that a decision rather than a guess.
    """
    from ...litho.compiler import compile_object
    from ...litho.transit import TRANSIT_MODES, TransitPolicy, plan_pattern_transits, transit_report
    pol = _transit_policy()
    report = {}
    pat = getattr(hub, "pattern", None)
    if pat and pat.objects:
        w = hub.host.world
        paths = []
        for o in pat.objects:
            if not o.enabled:
                continue
            c = compile_object(o, w.x_lvdt_sens, w.y_lvdt_sens,
                               w.x_lvdt_offset, w.y_lvdt_offset,
                               frame=_litho_frame())
            paths.append((o.name, c["path_um"]))
        here = (hub.host.arc.sig[channels.X_SENSOR] * w.x_lvdt_sens * 1e6,
                hub.host.arc.sig[channels.Y_SENSOR] * w.y_lvdt_sens * 1e6)
        for mode in TRANSIT_MODES:
            p = TransitPolicy(**{**pol.to_dict(), "mode": mode})
            report[mode] = transit_report(plan_pattern_transits(paths, here, p))
    return {"policy": pol.to_dict(), "modes": list(TRANSIT_MODES), "cost": report}


async def transit_set(body: dict):
    from ...litho.transit import TRANSIT_MODES, TransitPolicy
    cur = _transit_policy().to_dict()
    for k in ("mode", "speed_um_s", "max_speed_um_s", "lift_nm", "settle_s"):
        if k in body and body[k] is not None:
            cur[k] = body[k] if k == "mode" else float(body[k])
    if cur["mode"] not in TRANSIT_MODES:
        return {"error": f"mode must be one of {TRANSIT_MODES}"}
    hub.transit = TransitPolicy(**cur)
    await hub.log(f"Transit policy: {cur['mode']} at {hub.transit.speed:g} um/s"
                  + (f", {cur['lift_nm']:g} nm clearance" if cur["mode"] == "lift" else ""))
    return {"policy": hub.transit.to_dict()}


async def litho_frame_get():
    f = _litho_frame()
    pinned = (getattr(hub, "litho_frame", None) or {}).get("angle_deg")
    return {"frame": f.to_dict(), "angle_follows_scan": pinned is None}


async def litho_frame_set(body: dict):
    """Set the litho frame. `angle_deg: null` hands the angle back to the scan."""
    cur = dict(getattr(hub, "litho_frame", None) or {})
    for k in ("fast_size_um", "slow_size_um", "fast_offset_um",
              "slow_offset_um"):
        if k in body and body[k] is not None:
            cur[k] = float(body[k])
    if "angle_deg" in body:
        cur["angle_deg"] = None if body["angle_deg"] is None \
            else float(body["angle_deg"])
    hub.litho_frame = cur
    f = _litho_frame()
    await hub.log(f"Litho frame: {f.fast_size_um:g} x {f.slow_size_um:g} um, "
                  f"offset ({f.fast_offset_um:g}, {f.slow_offset_um:g}), "
                  f"{f.angle_deg:g} deg"
                  + ("" if cur.get("angle_deg") is not None else " (from scan)"))
    return {"frame": f.to_dict(),
            "angle_follows_scan": cur.get("angle_deg") is None}


def _rdp_keep(pts, tol_um: float):
    """The boolean keep-mask for `_rdp`. Exposed so a parallel per-point array
    (e.g. the bias along a modulated wire) can be subsampled at the SAME kept
    indices, staying registered to the geometry it decorates.
    """
    import numpy as np
    p = np.asarray(pts, dtype=float)
    n = len(p)
    if n < 3:
        return np.ones(n, dtype=bool)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = p[a:b + 1]
        d = seg[-1] - seg[0]
        L = float(np.hypot(d[0], d[1]))
        if L < 1e-12:                              # closed loop back to start
            dist = np.hypot(seg[:, 0] - seg[0, 0], seg[:, 1] - seg[0, 1])
        else:                                      # perpendicular to the chord
            dist = np.abs(d[0] * (seg[:, 1] - seg[0, 1])
                          - d[1] * (seg[:, 0] - seg[0, 0])) / L
        i = int(np.argmax(dist))
        if dist[i] > tol_um:
            keep[a + i] = True
            stack.append((a, a + i))
            stack.append((a + i, b))
    return keep


def _rdp(pts, tol_um: float):
    """Douglas-Peucker line simplification: drop points that sit within tol of
    the chord, keep the ones that carry the shape. A zig-zag row of a thousand
    collinear samples collapses to its two ends; a curved sweep keeps only the
    handful its curvature needs.
    """
    import numpy as np
    p = np.asarray(pts, dtype=float)
    if len(p) < 3:
        return p
    return p[_rdp_keep(p, tol_um)]


async def litho_toolpath():
    """The compiled toolpath of the current design, for on-screen preview.

    This is the ACTUAL path the tip runs -- every fill pass: the zig-zag rows
    of a rectangle, the sweeps of a funnel, the interpolated points of a wire --
    not the authored outline the canvas draws by default. The expensive part
    (expand + fill + interpolate) is the same code the writer uses, so what you
    see previewed is exactly the shape that would be written.

    Compiled with the IDENTITY frame, so path_um comes back in the design's own
    microns (like points_um), and the canvas maps it with the same umToPx as the
    outline -- the preview then follows reframing without a refetch.

    The points are simplified for display (Douglas-Peucker at 12.5 nm): the
    WRITTEN path keeps every interpolated point, but drawing all of them is tens
    of thousands of line segments for a shape the eye reads from a few hundred.
    The tolerance is kept fine deliberately -- about double the points a coarser
    pass would leave -- so a user zoomed right in still sees smooth curves rather
    than the chords of an under-sampled polyline.
    """
    from ...litho.compiler import compile_pattern
    pat = getattr(hub, "pattern", None)
    if not pat or not pat.objects:
        return {"segments": []}
    w = hub.host.world
    try:
        waves = await asyncio.to_thread(
            compile_pattern, pat, w.x_lvdt_sens, w.y_lvdt_sens,
            w.x_lvdt_offset, w.y_lvdt_offset, frame=None)
    except Exception as exc:  # noqa: BLE001 - a bad design is reported to the client
        return {"error": f"could not compile toolpath: {exc}"}
    import numpy as np
    segs = []
    for s in waves["segments"]:
        path = s["path_um"]
        if len(path) < 2:
            continue
        p = np.asarray(path, dtype=float)
        keep = _rdp_keep(p, 0.0125)                # display only, not the write
        pts = p[keep]
        seg = {"object": s["object"], "name": s["name"],
               "voltage": float(s["voltage"]),
               "path_um": [[float(x), float(y)] for x, y in pts]}
        # A modulated pass (e.g. a chiral wire) carries a per-point bias that a
        # single `voltage` cannot show. Forward it, subsampled at the SAME kept
        # points so it stays registered to the geometry -- but only when it
        # actually varies, so flat passes stay lean.
        v = s.get("v")
        if v is not None and len(v) == len(p):
            vv = np.asarray(v, dtype=float)[keep]
            if np.isfinite(vv).all() and float(vv.max() - vv.min()) > 1e-9:
                seg["volts"] = [round(float(x), 3) for x in vv]
        segs.append(seg)
    return {"segments": segs}


async def litho_demo():
    """The REAL compiled toolpath of a debug demo pattern, for the litho panel.

    Compiles a fixed demo of one of each object kind -- a straight `path`, a
    funnel `taper`, a `rect` fill, and a filled curved `shape` -- through the
    ACTUAL `compile_pattern`, so the panel draws the exact toolpath the writer
    would run. The geometry (funnel serpentine, rect boustrophedon, shape fill)
    is the compiler's, never a client-side re-derivation. Identity frame, so
    `path_um` comes back in the design's own microns, mapped client-side with the
    same umToPx as the scan overlay. Read-only: does not touch hub.pattern or any
    run state. Points are Douglas-Peucker simplified for display only.
    """
    from ...litho.compiler import LithoObject, Pattern, compile_pattern
    th = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    ring = [(1.8 * np.cos(t) + 4.5, 1.1 * np.sin(t) + 4.0) for t in th]
    pat = Pattern(name="litho-debug-demo", objects=[
        LithoObject("wire", "path", [(-6.0, 0.0), (6.0, 0.0)], voltage=8.0),
        LithoObject("funnel", "taper", [(-5.0, -4.0), (2.0, -4.0)],
                    width0_um=2.0, width1_um=0.3, voltage=6.0),
        LithoObject("pad", "rect", [(3.5, -5.5), (6.0, -2.5)], voltage=4.0),
        LithoObject("ring", "shape", ring, voltage=-6.0),
    ])
    w = hub.host.world
    try:
        waves = await asyncio.to_thread(
            compile_pattern, pat, w.x_lvdt_sens, w.y_lvdt_sens,
            w.x_lvdt_offset, w.y_lvdt_offset, frame=None)
    except Exception as exc:  # noqa: BLE001 - a bad demo is reported to the client
        return {"error": f"could not compile demo: {exc}"}
    segs = []
    for s in waves["segments"]:
        if len(s["path_um"]) < 2:
            continue
        pts = _rdp(s["path_um"], 0.0125)           # display only, not the write
        segs.append({"object": s["object"], "name": s["name"],
                     "voltage": float(s["voltage"]),
                     "path_um": [[float(x), float(y)] for x, y in pts]})
    return {"name": pat.name, "segments": segs}


def check_mode() -> str:
    """`commands.litho_cmds.check_mode` against the process hub -- the AUTHORITATIVE
    instrument imaging mode read off the live crosspoint. Re-exported under the name
    this module's callers and tests have always imported."""
    return litho_cmds.check_mode(hub)


def _ac_mode_active() -> bool:
    """`commands.litho_cmds.ac_mode_active` against the process hub.

    THE RULE ITSELF MOVED, VERBATIM AND WITH ITS PRIORITY ORDER INTACT (plan Sec C,
    Sec J risk 11): a failed crosspoint read must fall through to the engaged-mode and
    conservative branches and must NEVER be read as "contact". This wrapper exists so
    the mode-switch gate and the guard's tests keep importing one name."""
    return litho_cmds.ac_mode_active(hub)


async def litho_write(body: dict | None = None):
    """Write what has not been written yet (`commands.litho_cmds.litho_write`).

    Blocking only in the sense that it ARMS: the pass itself streams over the
    WebSocket. The FLEX door (`startWrite`) wraps the same coroutine and answers
    `{run_id, est_s, objects}` instead (plan Sec F.1).
    """
    try:
        return await litho_cmds.litho_write(hub, body)
    except CommandError as exc:
        return exc.http_body()


async def litho_abort():
    """Stop the write AND retract (the `abort_write` verb) -- the unified safe_park, not a
    bespoke stop. The wire's `abortWrite` goes through the same verb."""
    res = await HostApi(hub).abort_write()
    return {"ok": True, "backend": res["backend"]}


async def litho_written(size_um: float = 30.0, x_off: float = 0.0,
                        y_off: float = 0.0, n: int = 256):
    """Conductance (written-state) map over the requested frame."""
    s = hub.host.world.sample
    half = size_um / 2
    ax = (np.linspace(x_off - half, x_off + half, n)) * 1e-6
    ay = (np.linspace(y_off - half, y_off + half, n)) * 1e-6
    gi = np.clip(((ax / s.dose_range + 0.5) * (s.dose_n - 1) + 0.5).astype(int), 0, s.dose_n - 1)
    gj = np.clip(((ay / s.dose_range + 0.5) * (s.dose_n - 1) + 0.5).astype(int), 0, s.dose_n - 1)
    d = s.dose[np.ix_(gi, gj)].astype(np.float64)
    g = d * d / (1.0 + d * d)
    return {"n": n, "g": np.round(g.T.ravel(), 3).tolist()}
