"""The design surface: sketching, path/object operations, shapes and SVG I/O.

Everything here edits the working design (`hub.pattern`): strokes become
lithography objects, Inkscape-style operations reshape them, the node editor
drags shapes and generators, and SVG comes in and goes back out.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import numpy as np

from ...litho.design_file import unique_name as _unique_name
from ..commands import CommandError, design_cmds
from ..hub import hub


class _VoltageWindowRefused(Exception):
    """set_props tried to write a voltage/voltage_end/v_profile that cannot
    legitimately reach the tip (adversarial review finding M1,
    docs/INSTRUMENT_MODEL_SPEC.md section 10): the same window
    `command_guard.check_write` enforces at write time, enforced here too so a
    bad value never even enters the saved design. Caught in `design_op`'s
    existing exception chain, which already rolls `hub.pattern.objects` back to
    the pre-op snapshot on any exception."""


def _check_voltage_window(objs) -> None:
    """Thin wrapper over `command_guard.check_design_voltages` (spec 13.1 item
    19): the window + DAC-rail check is now ONE function shared by this REST
    set_props door and the FLEX `setObjects`/`loadPattern`/SVG doors, so the
    three cannot drift. The `GuardRefused` it raises is re-wrapped as
    `_VoltageWindowRefused` so `design_op`'s existing exception chain rolls the
    op back and returns this door's established `{"error", "reason", "design"}`
    shape unchanged; the guard's `.reason` (`tip_voltage_out_of_window` or, on a
    window wider than the amplifier can drive, `dac_rail_exceeded`) is carried
    through."""
    from ..command_guard import GuardRefused, check_design_voltages
    model = getattr(hub.host, "instrument_model", None)
    if model is None:
        return                                            # no model yet -- nothing to check against
    try:
        check_design_voltages(model, objs)
    except GuardRefused as exc:
        refused = _VoltageWindowRefused(str(exc))
        refused.reason = exc.reason
        raise refused from exc


async def pens():
    from ...litho.pens import CHIRAL_PRESETS, palette
    return {"pens": palette(),
            "presets": {"chiral": CHIRAL_PRESETS},
            "modes": [
                {"key": "literal", "label": "Literal",
                 "hint": "exactly as drawn"},
                {"key": "clean", "label": "Clean",
                 "hint": "smoothed, kinks removed"},
                {"key": "ideal", "label": "Idealize",
                 "hint": "straight strokes become lines, curves become arcs"},
            ]}


async def sketch(body: dict):
    """A stroke drawn on the canvas becomes lithography objects.

    Touch, stylus or mouse: the client sends the polyline in sample microns
    plus the pen and its parameters; the objects are appended to the working
    design so Write executes them like any other pattern.
    """
    from ...litho.compiler import Pattern
    from ...litho.pens import render_stroke

    pts = body.get("points_um") or []
    if len(pts) < 2:
        return {"error": "stroke too short"}
    pen = str(body.get("pen", "write"))
    mode = str(body.get("mode", "clean"))
    params = body.get("params") or {}
    if not getattr(hub, "pattern", None):
        hub.pattern = Pattern(name="sketch", objects=[])
    _design_snapshot()
    idx = len(hub.pattern.objects)
    objs = await asyncio.to_thread(
        render_stroke, pts, pen, params, mode, f"{pen}{idx}")
    hub.pattern.objects.extend(objs)
    from ...litho.pens import path_report
    # In a thread: a region's report resolves its fill, and a 10 nm pitch
    # over a few microns is millions of points. On the event loop that is
    # the whole panel stopping -- telemetry, scan lines and all -- while a
    # sketch is being reported on.
    r = await asyncio.to_thread(path_report, objs)
    # For a chirality experiment the amplitude IS the physics, so a clamp on
    # tight curvature is said out loud, not left to be inferred from a
    # strange picture. It is read OFF THE OBJECTS THIS CALL BUILT (the chiral pen
    # attaches it), never off a side channel on the builder: a concurrent sketch or a
    # FLEX `loadPattern` on another thread would otherwise have replaced it between
    # the render and this line, and a pen with no chiral in it would replay the last
    # chiral stroke's clamp.
    clamp = next((c for o in objs if (c := getattr(o, "chiral_clamp", None))), None)
    if clamp and clamp["fraction"] > 0:
        await hub.log(
            f"Chiral amplitude clamped on {clamp['fraction'] * 100:.0f}% of "
            f"the wire: requested {clamp['requested_um'] * 1000:.0f} nm, "
            f"floor {clamp['min_amp_um'] * 1000:.0f} nm at a curvature "
            f"radius of {clamp['tightest_radius_um'] * 1000:.0f} nm. The "
            f"written amplitude follows the curve, not the request.")
    else:
        clamp = None
    # in the high-amplitude regime the tip travels many times the length of
    # the wire it draws, so report both and the cost in time
    extra = (f" ({r['ratio']:.1f}x the {r['axial_um']:.2f} um it spans)"
             if r["ratio"] > 1.5 else "")
    await hub.log(f"Sketch: {pen} ({mode}) -> {len(objs)} object(s), "
                  f"{r['path_um']:.1f} um of tip travel{extra}, "
                  f"~{r['seconds']:.0f} s to write")
    return {"ok": True, "added": len(objs), "design": hub.pattern.to_dict(),
            "report": r, "clamp": clamp}


async def sketch_undo(body: dict | None = None):
    """Step back through the design's history.

    This used to delete the last N objects, which was right while the only
    thing that could change a design was appending a stroke to it. It stops
    being right the moment an operation removes two objects and adds one:
    undoing a union by deleting the union leaves the two inputs gone. So
    undo is a history of the whole object list now, and everything that
    edits it takes a snapshot first.
    """
    n = max(1, int((body or {}).get("count", 1)))
    if not getattr(hub, "pattern", None):
        return {"ok": True, "design": None}
    undone = 0
    for _ in range(n):
        if not _design_history:
            break
        hub.pattern.objects = _design_history.pop()
        undone += 1
    if not undone:
        await hub.log("Nothing to undo.")
        return {"ok": True, "design": hub.pattern.to_dict()}
    await hub.log(f"Undo: back {undone} step(s), "
                  f"{len(hub.pattern.objects)} object(s) on the canvas")
    return {"ok": True, "design": hub.pattern.to_dict(),
            "steps_left": len(_design_history)}


async def sketch_clear():
    if getattr(hub, "pattern", None):
        _design_snapshot()
        hub.pattern.objects.clear()
        getattr(hub.host, "written_objects", set()).clear()
    await hub.log("Sketch cleared")
    return {"ok": True}


# ------------------------------------------------------- the design surface --
# Inkscape's object and path operations, on the working design. See
# docs/INKSCAPE.md: these are not graphics luxuries here. `outset` is how a
# drawn line gets the width that makes it a lead, and `union` is how two
# overlapping regions become one region with one fill and no seam down the
# middle where the tip lifted.

_DESIGN_HISTORY_MAX = 60
_design_history: list = []
_design_evicted: list = []


def _norm_binding(raw):
    """Validate an endpoint->electrode binding from the client (EDIT_TOOLS §4).

    Returns {"node": "start"|"end", "port_id": <str>} for a well-formed binding,
    or None (which CLEARS the binding) for an explicit null / anything malformed.
    Kept deliberately dumb: it only records the reference; resolving the endpoint
    through the port is the later connections/drift deliverable.
    """
    if not isinstance(raw, dict):
        return None
    node = raw.get("node")
    pid = raw.get("port_id")
    if node not in ("start", "end") or not isinstance(pid, str) or not pid:
        return None
    return {"node": node, "port_id": pid}


def _copy_name(source: str, existing) -> str:
    """`<stem>-copy`, then `<stem>-copy-2`, ... where the stem is `source` without a
    trailing `-copy[-k]` or `-k`. A copy never takes a `<stem>-k` id: that is the
    name /add gives the next drawn object, and scripts address objects by it."""
    stem = re.sub(r"-copy(-\d+)?$", "", source)
    stem = re.sub(r"-\d+$", "", stem) or source
    return _unique_name(f"{stem}-copy", existing)


def _same(a, b) -> bool:
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


def _design_snapshot() -> None:
    """Remember the object list before something changes it."""
    import copy
    pat = getattr(hub, "pattern", None)
    if pat is None:
        return
    _design_history.append(copy.deepcopy(pat.objects))
    _design_evicted[:] = _design_history[:-_DESIGN_HISTORY_MAX]
    del _design_history[:-_DESIGN_HISTORY_MAX]


def _design_unsnapshot() -> list | None:
    """Take back the snapshot the current op just took, leaving the history exactly
    as it was before it (the step a full history evicted for it goes back)."""
    if not _design_history:
        return None
    snap = _design_history.pop()
    _design_history[:0] = _design_evicted
    _design_evicted.clear()
    return snap


#: Every operation the design surface offers, and what it needs. The front
#: end builds its menus from this and the test holds the two in step, so an
#: operation cannot exist in one and not the other.
DESIGN_OPS = {
    "union":        {"label": "Union", "keys": "Ctrl +", "min": 2, "area": True},
    "difference":   {"label": "Difference", "keys": "Ctrl -", "min": 2, "area": True},
    "intersection": {"label": "Intersection", "keys": "Ctrl *", "min": 2, "area": True},
    "exclusion":    {"label": "Exclusion", "keys": "Ctrl ^", "min": 2, "area": True},
    "outset":       {"label": "Outset", "keys": "Ctrl )", "min": 1, "area": False},
    "inset":        {"label": "Inset", "keys": "Ctrl (", "min": 1, "area": False},
    "simplify":     {"label": "Simplify", "keys": "Ctrl L", "min": 1, "area": False},
    "duplicate":    {"label": "Duplicate", "keys": "Ctrl D", "min": 1, "area": False},
    "delete":       {"label": "Delete", "keys": "Del", "min": 1, "area": False},
    "transform":    {"label": "Move / scale / flip", "keys": "arrows, h, v",
                     "min": 1, "area": False},
    "set_props":    {"label": "Set properties", "keys": "properties panel",
                     "min": 1, "area": False},
    "raise":        {"label": "Raise", "keys": "Page Up", "min": 1, "area": False},
    "lower":        {"label": "Lower", "keys": "Page Down", "min": 1, "area": False},
    "top":          {"label": "Raise to top", "keys": "Home", "min": 1, "area": False},
    "bottom":       {"label": "Lower to bottom", "keys": "End", "min": 1, "area": False},
}


async def design_get():
    """The current design (hub.pattern) as a design dict — same shape as the
    /api/design/svg import response's `design`. Lets a client (the litho
    cockpit) hydrate from the backend on load instead of showing placeholder
    data. Returns an empty objects list when nothing is loaded."""
    pat = getattr(hub, "pattern", None)
    if not pat:
        return {"name": None, "objects": []}
    return pat.to_dict()


async def design_capabilities():
    """What the design surface can do, and honestly whether it can do it.

    The palette asks before it draws itself: an operation whose geometry
    library is missing is shown disabled with the reason in its tooltip,
    which is the difference between a button that explains itself and a
    button that fails when pressed.
    """
    from ...litho import pathops
    try:
        pathops._db()
        geometry, why = True, ""
    except pathops.PathOpError as exc:
        geometry, why = False, str(exc)
    return {"ops": DESIGN_OPS, "geometry": geometry, "geometry_error": why,
            "undo_steps": len(_design_history)}


def _selected(indices) -> list:
    pat = getattr(hub, "pattern", None)
    objs = list(pat.objects) if pat else []
    out = []
    for i in indices or []:
        i = int(i)
        if not (0 <= i < len(objs)):
            raise ValueError(f"no object at index {i}")
        out.append(objs[i])
    return out


async def design_reorder(body: dict):
    """Reorder the design's objects -- the list order IS the write order, so
    dragging an object to the top makes it write first. `order` is a full
    permutation of the current object indices (0..n-1)."""
    from ...litho.compiler import Pattern
    if not getattr(hub, "pattern", None):
        hub.pattern = Pattern(name="sketch", objects=[])
    pat = hub.pattern
    n = len(pat.objects)
    order = [int(i) for i in (body.get("order") or [])]
    if sorted(order) != list(range(n)):
        return {"error": f"order must be a permutation of 0..{n - 1}",
                "design": pat.to_dict()}
    _design_snapshot()
    pat.objects = [pat.objects[i] for i in order]
    await hub.log(f"Reordered {n} object(s)")
    await hub.notify_design_changed()
    return {"design": pat.to_dict()}


async def design_add(body: dict):
    """Append a NEW object drawn on the canvas (rectangle or pen path).

    The edit cockpit's shape/pen tools author geometry in microns and post it
    here; it becomes a LithoObject appended to the working design and returned
    so the client re-adopts it. Snapshots for undo like every other edit, and
    names the object uniquely so the list, set_props (by index) and reorder
    keep working on it. `points_um` is the polyline (a path) or the closed
    outline (a shape/electrode); physics defaults match a hand-drawn stroke.
    """
    from ...litho.compiler import LithoObject, Pattern

    kind = str(body.get("kind", "path"))
    if kind not in ("path", "shape", "electrode"):
        return {"error": f"cannot add a {kind} here"}
    # A pen path can carry its Bezier (nodes + in/out handles + kinds) so it lands
    # node-editable exactly as drawn, not flattened to dozens of tiny segments. When
    # present, the SAME flatten the /api/design/bezier handler uses becomes
    # points_um, so the object still compiles and writes; the bezier is stored on it
    # and /api/design/bezier/{index} hands it straight back to the node tool.
    bez_dict = None
    bez_pts = None
    raw_bez = body.get("bezier")
    if raw_bez:
        from ...litho.bezier import BezierPath
        try:
            bp = BezierPath.from_dict(raw_bez)
            bez_pts = bp.flatten(0.004)
            bez_dict = bp.to_dict()
        except Exception as exc:  # noqa: BLE001 - bad geometry is reported to the client
            return {"error": f"bad bezier: {exc}"}
        if len(bez_pts) < 2:
            return {"error": "bezier flattens to fewer than two points"}
        # Drawn === imported: a closed outline drawn with the pen (kind "shape")
        # whose Bezier is a FUNNEL -- a flat base and two curved sides to an apex
        # -- becomes an `electrode`, exactly as the SVG import chooses (the
        # `as_funnel` branch above), so the compiler fills between its two side
        # curves and the cockpit labels it a funnel. Any other closed outline
        # (a rectangle's 4-node bezier, a generic blob) stays a `shape`.
        if kind == "shape" and bp.closed and bp.as_funnel() is not None:
            kind = "electrode"
    if bez_pts is not None:
        pts = [(float(x), float(y)) for x, y in bez_pts]
    else:
        pts = body.get("points_um") or []
        try:
            pts = [(float(x), float(y)) for x, y in pts]
        except (TypeError, ValueError):
            return {"error": "points_um must be [[x, y], ...] microns"}
    if len(pts) < 2:
        return {"error": "need at least two points"}
    if not getattr(hub, "pattern", None):
        hub.pattern = Pattern(name="sketch", objects=[])
    pat = hub.pattern
    _design_snapshot()
    name = _unique_name(str(body.get("name") or kind), (o.name for o in pat.objects))
    o = LithoObject(name, kind, pts,
                    voltage=float(body.get("voltage", 8.0)),
                    speed_um_s=float(body.get("speed_um_s", 2.0)))
    if bez_dict is not None:
        o.bezier = bez_dict
    if kind in ("shape", "electrode"):
        fs = body.get("fill_spacing_um")
        if fs is not None:
            o.fill_spacing_um = float(fs)
    # An object can arrive already bound (a wire whose end was snapped to a port
    # before commit). Optional -- absent leaves it unbound (EDIT_TOOLS §4).
    if "end_binding" in body:
        o.end_binding = _norm_binding(body["end_binding"])
    pat.objects.append(o)
    await hub.log(f"Added {kind} {name}: {len(pts)} point(s)")
    await hub.notify_design_changed()
    return {"ok": True, "design": pat.to_dict(),
            "index": len(pat.objects) - 1, "name": name}


async def design_points(body: dict):
    """Replace an object's polyline vertices after node editing.

    The node tool drags a path's vertices on the canvas and posts the whole
    new polyline (microns). Geometry only -- physics/kind untouched. Snapshots
    for undo. Hand-editing the vertices redefines the path, so any modulation
    base (the chiral axis / profile / bias) is dropped: it no longer describes
    what is on the canvas.
    """
    pat = getattr(hub, "pattern", None)
    idx = int(body.get("index", -1))
    if not pat or not (0 <= idx < len(pat.objects)):
        return {"error": "no such object"}
    pts = body.get("points_um") or []
    try:
        pts = [(float(x), float(y)) for x, y in pts]
    except (TypeError, ValueError):
        return {"error": "points_um must be [[x, y], ...] microns"}
    if len(pts) < 2:
        return {"error": "need at least two points"}
    o = pat.objects[idx]
    _design_snapshot()
    o.points_um = pts
    o.axis_um = None
    o.profile = None
    o.v_profile = None
    # A wholesale vertex replace can also carry an endpoint->port binding update
    # (set or clear). Only touched when the key is present (EDIT_TOOLS §4).
    if "end_binding" in body:
        o.end_binding = _norm_binding(body["end_binding"])
    await hub.log(f"Edited {o.name}: {len(pts)} vertices")
    await hub.notify_design_changed()
    return {"ok": True, "design": pat.to_dict()}


def _polyline_as_bezier(pts) -> dict | None:
    """Any polyline as a path of cusp nodes (no handles).

    The open-path companion to _polygon_as_bezier below: an open wire keeps all
    its vertices and stays open; a polyline whose ends coincide closes and drops
    the duplicate. Cusp nodes with no handles flatten back to exactly these
    points, so opening a plain path in the node editor and closing it again
    cannot move anything. This is what makes a hand-drawn wire (which never came
    from control points) editable in the Inkscape-style node tool.
    """
    pts = [(float(x), float(y)) for x, y in (pts or [])]
    if len(pts) < 2:
        return None
    closed = (len(pts) >= 4
              and abs(pts[0][0] - pts[-1][0]) < 1e-9
              and abs(pts[0][1] - pts[-1][1]) < 1e-9)
    if closed:
        pts = pts[:-1]
    return {"nodes": [{"x": round(x, 5), "y": round(y, 5),
                       "h_in": None, "h_out": None, "kind": "cusp"}
                      for x, y in pts],
            "closed": closed}


def _reapply_chiral(o) -> bool:
    """Re-bake a chiral wire from its (edited) spine; False if it cannot be, so the
    caller can fall back to the plain base rather than leave a stale toolpath."""
    from ...litho.pens import rebake_chiral
    try:
        return rebake_chiral(o) is not None
    except ValueError:
        return False


async def design_bezier_get(index: int):
    """The editable Bezier of an object -- what the node tool edits.

    Collapse-to-base: when the object carries a modulation (a captured
    `axis_um`, or a profile), the editable path is the BASE axis, NOT the baked
    modulated toolpath, so node-editing reshapes the underlying wire and the
    modulation re-applies on top. An object with real control points (an
    imported curve/funnel) returns them verbatim; a plain polyline is
    synthesized into cusp nodes so it becomes editable. `modulated` tells the
    client the returned curve is a base a modulation sits on.
    """
    from ...litho.bezier import BezierPath
    pat = getattr(hub, "pattern", None)
    if not pat or not (0 <= index < len(pat.objects)):
        return {"error": "no such object"}
    o = pat.objects[index]
    modulated = getattr(o, "axis_um", None) is not None or bool(o.profile)
    if modulated:
        base = getattr(o, "axis_um", None) or (o.profile or {}).get("axis_um")
        bez = _polyline_as_bezier(base)
    elif o.bezier:
        bez = o.bezier
    else:
        bez = _polygon_as_bezier(o) or _polyline_as_bezier(o.points_um)
    if not bez:
        return {"error": "object has no editable geometry"}
    # normalize the node kinds the way the SVG parser labels them, so imported
    # curves open with smooth/symmetric handles behaving as Inkscape showed them
    try:
        BezierPath.from_dict(bez)      # validate shape; raises on a bad dict
    except Exception as exc:  # noqa: BLE001 - bad geometry is reported to the client
        return {"error": f"un-editable geometry: {exc}"}
    return {"index": index, "name": o.name, "kind": o.kind,
            "bezier": bez, "modulated": modulated}


async def design_bezier(body: dict):
    """Store an object's edited Bezier control points and re-flatten.

    The node tool edits the exact curve (not the flat polyline); this keeps
    BOTH the updated `bezier` and the re-flattened `points_um`, so the object
    still compiles and writes. Collapse-to-base: when the object carries a
    modulation, the edited Bezier IS the base axis -- it becomes `axis_um` and
    the modulation regenerates from it (the sine/chiral re-applies on top of the
    reshaped wire). Snapshots for undo and broadcasts like every other edit.
    """
    from ...litho.bezier import BezierPath
    pat = getattr(hub, "pattern", None)
    idx = int(body.get("index", -1))
    if not pat or not (0 <= idx < len(pat.objects)):
        return {"error": "no such object"}
    try:
        bp = BezierPath.from_dict(body.get("bezier") or {})
        pts = bp.flatten(0.004)
    except Exception as exc:  # noqa: BLE001 - bad geometry is reported to the client
        return {"error": f"bad bezier: {exc}"}
    if len(pts) < 2:
        return {"error": "need at least two points"}
    o = pat.objects[idx]
    _design_snapshot()
    base = [(float(x), float(y)) for x, y in pts]
    o.bezier = bp.to_dict()
    # The KIND follows the curve's closedness (mirrors /api/design/add): joining a
    # wire's two ends closes the Bezier, which should make it a FILLED region, and
    # splitting a closed shape open should turn it back into a traced path. A closed
    # curve is an `electrode` when it reads as a funnel (flat base + two curved
    # sides), else a generic filled `shape`; an open curve is a `path`. Not gated on
    # `modulated`: a chiral wire's base axis is always open, so it stays a path.
    o.kind = ("electrode" if bp.as_funnel() is not None else "shape") \
        if bp.closed else "path"
    # Snap-to-bind: the node tool sends the endpoint->port binding alongside the
    # edited curve when a wire end snaps to (or leaves) an electrode. Only touched
    # when the key is present, so an ordinary node drag never changes it (EDIT_TOOLS §4).
    if "end_binding" in body:
        o.end_binding = _norm_binding(body["end_binding"])
    modulated = getattr(o, "axis_um", None) is not None or bool(o.profile)
    reapplied = False
    if modulated:
        o.axis_um = base                          # the edited curve is the base
        reapplied = _reapply_chiral(o)
    if reapplied:
        msg = (f"Edited base of {o.name}: {len(bp.nodes)} node(s), "
               f"modulation re-applied")
    else:
        o.points_um = base
        if modulated:
            # a modulation we can't regenerate (unknown kind / missing params):
            # the hand edit redefines the path, so drop the stale modulation
            # rather than leave a toolpath that no longer matches the curve
            o.axis_um = None
            o.profile = None
            o.v_profile = None
            msg = (f"Edited {o.name}: {len(bp.nodes)} node(s) "
                   f"(modulation dropped -- could not regenerate)")
        else:
            msg = f"Edited {o.name}: {len(bp.nodes)} node(s)"
    await hub.log(msg)
    await hub.notify_design_changed()
    return {"ok": True, "design": pat.to_dict(), "reapplied": reapplied,
            "message": msg}


async def design_op(body: dict):
    """Run one operation on the selected objects.

    One route rather than fourteen because every one of them does the same
    three things -- resolve the selection, snapshot for undo, replace part of
    the object list -- and those are the invariants that must not drift apart.
    """
    from ...litho import pathops
    from ...litho.compiler import Pattern
    from ...litho.pens import is_chiral, rebake_chiral

    op = str(body.get("op", ""))
    spec = DESIGN_OPS.get(op)
    if spec is None:
        return {"error": f"no such operation: {op}"}
    if not getattr(hub, "pattern", None):
        hub.pattern = Pattern(name="sketch", objects=[])
    pat = hub.pattern
    idx = sorted({int(i) for i in (body.get("indices") or [])})
    try:
        objs = _selected(idx)
    except ValueError as exc:
        return {"error": str(exc)}
    if len(objs) < spec["min"]:
        return {"error": f"{spec['label']} needs {spec['min']} object(s) "
                         f"selected, and {len(objs)} are"}

    def replaced(new_objs):
        """Put `new_objs` where the selection was, keeping z-order sane.

        `idx` is sorted, so its first entry is the lowest object in the
        selection and everything removed sits at or above it: the surviving
        objects below it keep their positions and the result lands there.
        """
        at = idx[0]
        keep = [o for i, o in enumerate(pat.objects) if i not in set(idx)]
        pat.objects = keep[:at] + list(new_objs) + keep[at:]

    new: list = []
    unchanged = False
    try:
        _design_snapshot()
        match op:
            case _ if op in pathops.BOOLEAN_OPS:  # union/difference/intersection/exclusion
                new = pathops.boolean(objs, op, name=f"{op}{len(pat.objects)}")
                replaced(new)
                msg = (f"{spec['label']}: {len(objs)} objects -> "
                       f"{len(new)} region(s)")
            case "outset" | "inset":
                d = abs(float(body.get("delta_um", 0.25)))
                if op == "inset":
                    d = -d
                new = []
                for o in objs:
                    new += pathops.offset(o, d, name=f"{o.name}{'+' if d > 0 else '-'}")
                replaced(new)
                msg = (f"{spec['label']} {abs(d):g} um: {len(objs)} object(s) -> "
                       f"{len(new)}")
            case "simplify":
                tol = float(body.get("tol_um", 0.02))
                before_n = sum(len(o.points_um) for o in objs)
                new = [pathops.simplify(o, tol) for o in objs]
                replaced(new)
                after_n = sum(len(o.points_um) for o in new)
                msg = (f"Simplify at {tol:g} um: {before_n} -> {after_n} vertices")
            case "duplicate":
                import copy
                dx = float(body.get("dx_um", 0.0))
                dy = float(body.get("dy_um", 0.0))
                new = []
                for o in objs:
                    c = copy.deepcopy(o)
                    c.name = _copy_name(o.name, [x.name for x in [*pat.objects, *new]])
                    # A port takes one wire; the copy is offset off the port anyway.
                    c.end_binding = None
                    if dx or dy:
                        pathops.transform(c, dx_um=dx, dy_um=dy)
                    new.append(c)
                pat.objects = list(pat.objects) + new
                msg = f"Duplicated {len(new)} object(s)"
            case "delete":
                names = ", ".join(o.name for o in objs[:4])
                replaced([])
                msg = (f"Deleted {len(objs)} object(s): {names}"
                       + ("…" if len(objs) > 4 else ""))
            case "transform":
                about = body.get("about_um")
                if about is None:
                    x0, y0, x1, y1 = pathops.bbox_of(objs)
                    about = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
                for o in objs:
                    pathops.transform(
                        o, dx_um=float(body.get("dx_um", 0.0)),
                        dy_um=float(body.get("dy_um", 0.0)),
                        sx=float(body.get("sx", 1.0)), sy=float(body.get("sy", 1.0)),
                        rot_deg=float(body.get("rot_deg", 0.0)),
                        about=(float(about[0]), float(about[1])))
                msg = body.get("say") or f"Moved {len(objs)} object(s)"
                unchanged = bool(_design_history) and _same(
                    pat.objects, _design_history[-1])
            case "set_props":
                # Edit the electrical/kind properties of the selected object(s)
                # in place -- the "Fill & Stroke" of a litho object. Geometry is
                # untouched: only the fields present in the body are written, so
                # a panel that sends just `voltage` cannot clobber `speed_um_s`.
                # `pen` is accepted as an alias for `kind` (the palette calls the
                # concept a pen; the object stores it as kind).
                changed = []
                for o in objs:
                    chiral = is_chiral(o)
                    for key in ("voltage_end", "v_profile"):
                        if chiral and key in body and not (
                                key == "voltage_end" and body[key] is None):
                            raise pathops.ChiralRefused(
                                "chiral_bias_fixed",
                                f"object {o.name!r} is a chiral wire: its bias is "
                                f"defined by its voltage and chiral settings, so "
                                f"{key} cannot be set on it. Change its voltage or "
                                f"chiral settings, or turn chiral off first.")
                    if "voltage" in body and body["voltage"] is not None:
                        o.voltage = float(body["voltage"])
                        changed.append("voltage")
                        if chiral:
                            try:
                                built = rebake_chiral(o)
                            except ValueError as exc:
                                raise pathops.ChiralRefused(
                                    "chiral_build_refused",
                                    f"object {o.name!r} is a chiral wire and "
                                    f"cannot be re-baked: {exc}") from exc
                            if built is None:
                                raise pathops.ChiralRefused(
                                    "chiral_unrebakeable",
                                    f"object {o.name!r} is a chiral wire with no "
                                    f"spine or settings to re-bake from, so its "
                                    f"voltage cannot be changed. Re-apply its chiral "
                                    f"settings.")
                    if "voltage_end" in body:
                        ve = body["voltage_end"]
                        o.voltage_end = None if ve is None else float(ve)
                        changed.append("voltage_end")
                    if "speed_um_s" in body and body["speed_um_s"] is not None:
                        o.speed_um_s = float(body["speed_um_s"])
                        changed.append("speed_um_s")
                    kind = body.get("kind", body.get("pen"))
                    if kind is not None:
                        o.kind = str(kind)
                        changed.append("kind")
                    if "v_profile" in body:
                        vp = body["v_profile"]
                        o.v_profile = (None if vp is None
                                       else [float(x) for x in vp])
                        changed.append("v_profile")
                    if "enabled" in body and body["enabled"] is not None:
                        o.enabled = bool(body["enabled"])
                        changed.append("enabled")
                    fs = body.get("fill_spacing_um")
                    if fs is not None and float(fs) > 0:
                        o.fill_spacing_um = float(fs)
                        changed.append("fill_spacing_um")
                # M1: refuse a voltage/voltage_end/v_profile that could never
                # legitimately reach the tip BEFORE it is ever saved into the
                # design -- raises _VoltageWindowRefused, caught below, which
                # rolls this whole op back via the existing history snapshot.
                if {"voltage", "voltage_end", "v_profile"} & set(changed):
                    _check_voltage_window(objs)
                fields = ", ".join(dict.fromkeys(changed)) or "nothing"
                msg = (f"Set {fields} on {len(objs)} object(s): "
                       + ", ".join(o.name for o in objs[:4])
                       + ("…" if len(objs) > 4 else ""))
            case _:                                   # z-order: top/bottom/raise/lower
                sel = [pat.objects[i] for i in idx]
                rest = [o for i, o in enumerate(pat.objects) if i not in set(idx)]
                match op:
                    case "top":
                        pat.objects = rest + sel
                    case "bottom":
                        pat.objects = sel + rest
                    case "raise":
                        pat.objects = list(pat.objects)
                        for i in reversed(idx):
                            if i + 1 < len(pat.objects):
                                pat.objects[i], pat.objects[i + 1] = \
                                    pat.objects[i + 1], pat.objects[i]
                    case _:                           # "lower"
                        pat.objects = list(pat.objects)
                        for i in idx:
                            if i > 0:
                                pat.objects[i], pat.objects[i - 1] = \
                                    pat.objects[i - 1], pat.objects[i]
                msg = f"{spec['label']}: {len(sel)} object(s)"
    except pathops.PathOpError as exc:
        if _design_history:
            hub.pattern.objects = _design_unsnapshot()   # it did not happen
        await hub.log(f"{spec['label']}: {exc}")
        out = {"error": str(exc), "design": pat.to_dict()}
        if getattr(exc, "reason", None):
            out["reason"] = exc.reason
        return out
    except _VoltageWindowRefused as exc:
        if _design_history:
            hub.pattern.objects = _design_unsnapshot()   # it did not happen
        await hub.log(f"{spec['label']} REFUSED: {exc}")
        # M1/N5 note: this door's existing convention is a bare {"error": str}
        # (unlike the write doors' reason-carrying bodies) -- carry BOTH, so a
        # caller that already parses the write doors' {"reason": ...} shape
        # works here too, and one that only greps the error string still finds
        # "tip_voltage_out_of_window" in it.
        return {"error": str(exc),
               "reason": getattr(exc, "reason", "tip_voltage_out_of_window"),
               "design": pat.to_dict()}
    # any op failure is reported, design restored
    except Exception as exc:  # noqa: BLE001  # pragma: no cover
        if _design_history:
            hub.pattern.objects = _design_unsnapshot()
        return {"error": f"{spec['label']} failed: {exc}", "design": pat.to_dict()}

    if unchanged:
        _design_unsnapshot()
        return {"ok": True, "design": pat.to_dict(), "selection": idx,
                "message": "No change", "undo_steps": len(_design_history)}
    await hub.log(msg)
    # Which objects the caller should now have selected. After a union that
    # is the union, not the two things that no longer exist, and the panel
    # cannot work it out for itself without knowing what the operation did.
    if op == "delete":
        survivors = set()
    elif new:
        survivors = {o.name for o in new}
    else:
        survivors = {o.name for o in objs}
    selection = [i for i, o in enumerate(pat.objects) if o.name in survivors]
    # Notify every other client (the cockpit's `design_changed` re-adopt, the FLEX
    # door, the write-estimate refresh) that the design changed -- exactly as
    # /add, /points, /reorder and /chiral already do. set_props (enabled/voltage/
    # speed) and the other ops mutate hub.pattern the same way, so an operator on a
    # second panel and the #writeEstimate total must see it without a reload
    # (docs/INSTRUMENT_MODEL_SPEC.md sec 4). This door was the one design-mutating
    # route missing the broadcast.
    await hub.notify_design_changed()
    return {"ok": True, "design": pat.to_dict(), "selection": selection,
            "message": msg, "undo_steps": len(_design_history)}


async def design_chiral(body: dict):
    """Make a path chiral, re-parametrize it, or revert it to a plain wire.

    Chiral is a per-PATH modulation, not a DESIGN_OPS entry: it has its own
    body shape and its own apply/revert semantics, so it gets a dedicated
    route like /api/shapes/{index}. The critical invariant is axis
    preservation -- the plain axis a path was drawn/imported as is captured
    once on the first enable into `axis_um`, and every regenerate runs
    chiral_profile against THAT axis, never against the already-modulated
    points_um, so re-editing amplitude/phase/lambda/v_k cannot compound.
    Disabling restores points_um from the stored axis (falling back to the
    profile's decimated axis only if the field is somehow missing).
    """
    from ...litho.pens import bake_chiral
    pat = getattr(hub, "pattern", None)
    if not pat:
        return {"error": "no design loaded"}
    raw_idx = body.get("indices") or []
    if not isinstance(raw_idx, list) or any(
            isinstance(i, bool) or not isinstance(i, int) for i in raw_idx):
        return {"error": f"chiral: indices must be a list of object indices, got "
                         f"{raw_idx!r}; nothing was changed",
                "reason": "chiral_bad_setting", "field": "indices",
                "design": pat.to_dict()}
    idx = sorted(set(raw_idx))
    try:
        objs = _selected(idx)
    except ValueError as exc:
        return {"error": str(exc)}
    # chiral is only meaningful on a plain path; reject anything else and
    # leave the objects untouched (the cockpit also hides the control off-path)
    bad = [o.name for o in objs if o.kind != "path"]
    if bad:
        return {"error": f"chiral is a path-only setting; not a path: "
                         f"{', '.join(bad)}", "design": pat.to_dict()}
    enabled = bool(body.get("enabled"))
    clamp = None
    if enabled:
        import copy
        import math

        def refuse(field, text, reason="chiral_bad_setting"):
            return {"error": text, "reason": reason, "field": field,
                    "design": pat.to_dict()}

        vals = {}
        for key in ("lambda_um", "y_amp_um", "v_k", "phase_deg", "v0"):
            raw = body.get(key)
            if raw is None and key == "v0":
                continue
            try:
                good = (not isinstance(raw, bool) and isinstance(raw, (int, float))
                        and math.isfinite(float(raw)))
            except OverflowError:
                good = False
            if not good:
                shown = repr(raw) if len(repr(raw)) <= 40 else repr(raw)[:37] + "..."
                return refuse(key, f"chiral: {key} must be a finite number, got "
                                   f"{shown}; nothing was changed")
            vals[key] = float(raw)
        if vals["lambda_um"] <= 0:
            return refuse("lambda_um", f"chiral: the pitch (lambda_um) must be greater "
                                       f"than 0, got {vals['lambda_um']:g}; nothing "
                                       f"was changed")
        hand = -1.0 if str(body.get("handedness", "R")).upper()\
                          .startswith("L") else 1.0
        staged = []
        for o in objs:
            c = copy.deepcopy(o)
            if "v0" in vals:
                c.voltage = vals["v0"]
            try:
                build = bake_chiral(c, vals["lambda_um"], vals["y_amp_um"],
                                    vals["v_k"], vals["phase_deg"], hand)
            except ValueError as exc:
                text = str(exc)
                field = ("y_amp_um" if "relative to the pitch" in text
                         else "points_um" if "no length" in text
                         else "settings" if "not finite" in text else "lambda_um")
                return refuse(field, f"object {o.name!r} cannot be made chiral: "
                                     f"{text}; nothing was changed",
                              "chiral_build_refused")
            staged.append(c)
            # THIS call's clamp, not whatever a concurrent build last left behind.
            clamp = build.clamp or clamp
        _design_snapshot()
        for o, c in zip(objs, staged):
            o.__dict__.update(c.__dict__)
    else:
        _design_snapshot()
        for o in objs:
            base = (getattr(o, "axis_um", None)
                    or (o.profile or {}).get("axis_um"))
            if base:
                o.points_um = [(float(a), float(b)) for a, b in base]
            o.axis_um = None
            o.profile = None
            o.v_profile = None
    await hub.log(f"Chiral {'on' if enabled else 'off'}: {len(objs)} path(s)")
    await hub.notify_design_changed()
    return {"ok": True, "design": pat.to_dict(),
            "clamp": clamp if (clamp and clamp["fraction"] > 0) else None}


async def shapes():
    """The editable Bezier shapes in the current design.

    """
    pat = getattr(hub, "pattern", None)
    out = []
    for i, o in enumerate(pat.objects if pat else []):
        if o.kind != "shape":
            continue
        bez = o.bezier or _polygon_as_bezier(o)
        if not bez:
            continue
        out.append({"index": i, "name": o.name,
                    "bezier": bez,
                    "voltage": o.voltage,
                    "fill_spacing_um": o.fill_spacing_um})
    return {"shapes": out}


def _polygon_as_bezier(o) -> dict | None:
    """A plain polygon outline, as a path of cusp nodes.

    Every shape here is a path, and Inkscape's node tool edits any path. A
    region drawn with the shape tools, a boolean result, an outset lead: none
    of them came from control points, and until this existed the node tool
    had nothing to offer for any of them -- press `n`, and the editor said
    there were no editable shapes while four of them were on screen.

    Cusp nodes with no handles flatten back to exactly these vertices, so
    opening a polygon in the node editor and closing it again cannot move
    anything.
    """
    pts = list(o.points_um or [])
    if len(pts) < 3:
        return None
    closed = (abs(pts[0][0] - pts[-1][0]) < 1e-9
              and abs(pts[0][1] - pts[-1][1]) < 1e-9)
    if closed:
        pts = pts[:-1]
    if len(pts) < 3:
        return None
    return {"nodes": [{"x": round(float(x), 5), "y": round(float(y), 5),
                       "h_in": None, "h_out": None, "kind": "cusp"}
                      for x, y in pts],
            "closed": True}


async def shape_update(index: int, body: dict):
    """Replace a shape's Bezier outline after node editing.

    The outline is re-flattened here rather than trusting the client's
    polygon, so the geometry that gets written is always the geometry the
    control points describe. Editing a lead drops its centre line, because
    once the nodes have been moved by hand the fill can no longer follow a
    path that no longer bounds it.
    """
    from ...litho.bezier import BezierPath
    pat = getattr(hub, "pattern", None)
    if not pat or not (0 <= index < len(pat.objects)):
        return {"error": "no such object"}
    o = pat.objects[index]
    if o.kind != "shape":
        return {"error": f"{o.name} is not an editable shape"}
    try:
        bp = BezierPath.from_dict(body.get("bezier") or {})
        pts = bp.flatten(0.004)
    except Exception as exc:  # noqa: BLE001 - bad geometry is reported to the client
        return {"error": f"bad outline: {exc}"}
    if len(pts) < 3:
        return {"error": "outline needs at least three points"}
    _design_snapshot()
    o.bezier = bp.to_dict()
    o.points_um = [(float(x), float(y)) for x, y in pts]
    await hub.log(f"Shape {o.name}: {len(bp.nodes)} nodes edited")
    await hub.notify_design_changed()   # M1: shape outline edited
    return {"ok": True, "design": pat.to_dict()}


async def shape_toolpath(index: int):
    """The tip path that filling this shape will actually write.

    The canvas shows the outline; the tip writes the fill. Serving the
    expanded strokes lets the editor draw the toolpath alongside the
    outline, so what will be written is visible while the shape is being
    aimed, not discovered when the write starts.
    """
    from ...litho.compiler import expand
    pat = getattr(hub, "pattern", None)
    if not pat or not (0 <= index < len(pat.objects)):
        return {"error": "no such object"}
    o = pat.objects[index]
    parts = expand(o)
    paths = []
    total = 0
    for p in parts:
        pts = np.asarray(p.points_um, dtype=np.float64)
        if len(pts) < 2:
            continue
        total += len(pts)
        stride = max(1, len(pts) // 1500)
        keep = pts[::stride]
        if stride > 1 and not np.array_equal(keep[-1], pts[-1]):
            keep = np.vstack([keep, pts[-1:]])
        paths.append([[round(float(x), 4), round(float(y), 4)]
                      for x, y in keep])
    return {"paths": paths, "strokes": len(paths), "points": total}


async def import_svg(text: str, name: str, voltage: float, speed_um_s: float,
                     fill_spacing_um: float):
    """Import an SVG as a design: closed paths become filled shapes.

    This is the archive route. A drawing made in Inkscape years ago opens
    with the geometry it was drawn with, including the Bezier handles, and
    each closed outline becomes a shape the tip fills. Open paths become
    plain wires.

    Scale is fixed: this lab's SVGs are ALWAYS drawn 1 unit = 1 nm, so the
    import is 1e-3 um per unit, full stop -- no per-file scale to get wrong.
    """
    UM_PER_UNIT = 1e-3                             # 1 SVG unit = 1 nm, always
    from fastapi.responses import JSONResponse

    from ...litho.bezier import load_svg
    from ...litho.compiler import LithoObject, Pattern
    from ...litho.frame import bezier_paths_to_lab
    from ..design_import import (
        DesignSizeError,
        frame_bbox_check,
        resolve_import_size_um,
    )

    # Every upload REPLACES the source, a failed one included: cleared up front
    # so a parse failure / no-geometry / oversized-design early return below
    # cannot leave a stale `design_source` for the background-swap route to
    # silently act on against the PREVIOUS design (docs/IMAGE_SAVING_NOTES.md;
    # review finding on the svg-background branch). Re-set to the real value
    # only once the import actually succeeds, below.
    hub.design_source = None
    if "<svg" not in text.lower():
        return {"error": "not an SVG"}
    # WHERE the design sits: the page top-left is the scan top-left (the
    # design-origin rule), so the import needs the scan size it was drawn over.
    # Resolved BEFORE parsing so a design with no size to place it against is
    # refused up front rather than parsed and then dropped.
    try:
        size_um, size_source = resolve_import_size_um(text, hub)
    except DesignSizeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    try:
        paths = await asyncio.to_thread(load_svg, text, UM_PER_UNIT,
                                        preserve_origin=True)
    except Exception as exc:  # noqa: BLE001 - a malformed file is reported to the client
        return {"error": f"could not read {name}: {exc}"}
    if not paths:
        return {"error": f"no drawable geometry in {name}"}
    # Page-origin microns -> lab: translate + flip y about the scan centre, once.
    bezier_paths_to_lab(paths, size_um)

    # Guard the flatten below. At 1 nm/unit a normal design is tens of um; a file
    # that comes out metres wide is malformed (or not drawn to the 1-unit-is-1-nm
    # convention), and flattening a 40 mm curve to a 4 nm tolerance is millions of
    # points -- tens of GB and a killed process. No scanner writes past a few
    # hundred um, so bound the design cheaply from its control points (the curve
    # lives inside their hull) and refuse an absurd one before flattening it.
    import numpy as _np
    lo = _np.array([_np.inf, _np.inf]); hi = -lo
    for bp in paths:
        for seg in bp.segments():
            for pt in seg:
                q = _np.asarray(pt, dtype=float)
                lo = _np.minimum(lo, q); hi = _np.maximum(hi, q)
    span_um = float(_np.max(hi - lo)) if _np.isfinite(lo[0]) else 0.0
    MAX_DESIGN_UM = 2000.0
    if span_um > MAX_DESIGN_UM:
        return {"error":
                f"design spans {span_um:.0f} um -- far larger than any scanner. "
                f"The file is malformed or not drawn 1 unit = 1 nm. Not importing."}

    objs = []
    for i, bp in enumerate(paths):
        pts = bp.flatten(0.004)
        if len(pts) < 2:
            continue
        label = f"{Path(name).stem}-{i + 1}"
        if bp.closed and len(pts) >= 3:
            # A closed outline is one of the two shapes the archive is built
            # from. If it is a FUNNEL -- a flat base and two curved sides
            # meeting at an apex, the shape most of the lab's designs are -- import
            # it as an electrode so the compiler fills between its two side
            # curves (fill_funnel_from_curves) and the cockpit labels it a
            # funnel, not a rectangle. The outline itself is the generator: we
            # keep the Bezier verbatim and fill between its sides, NOT from a
            # centre line + widths (that is the hand-drawn electrode pen). Any
            # other closed outline stays a generic filled shape (a rectangle
            # or region).
            kind = "electrode" if bp.as_funnel() is not None else "shape"
            objs.append(LithoObject(
                label, kind, [(float(x), float(y)) for x, y in pts],
                bezier=bp.to_dict(), voltage=voltage, speed_um_s=speed_um_s,
                fill_spacing_um=fill_spacing_um))
        else:
            objs.append(LithoObject(
                label, "path", [(float(x), float(y)) for x, y in pts],
                voltage=voltage, speed_um_s=speed_um_s))
    if not objs:
        return {"error": f"no usable paths in {name}"}
    # Voltage window + DAC rail, BEFORE the design is installed (spec 13.1 item
    # 19): a caller can pass any `voltage` to this import, so refuse one outside
    # the sample's window (or over the rail) here rather than letting it enter
    # the saved pattern and trip only at Write. Same choke point the FLEX doors
    # and set_props use; the pattern is untouched on refusal.
    try:
        _check_voltage_window(objs)
    except _VoltageWindowRefused as exc:
        await hub.log(f"Imported {name} REFUSED: {exc}")
        return {"error": str(exc),
                "reason": getattr(exc, "reason", "tip_voltage_out_of_window")}
    hub.pattern = Pattern(name=Path(name).stem, objects=objs)
    # Keep the ORIGINAL markup too, not just the parsed LithoObjects -- the
    # background-swap route (service/svg_background.py, /api/design/background)
    # edits the real SVG (its layers, existing background image, Inkscape
    # metadata), which the flattened `pattern` above has already thrown away.
    # Only reached once the import has actually succeeded (see the None reset
    # at the top of this handler for the failure paths).
    hub.design_source = {"name": name, "text": text}
    filled = sum(1 for o in objs if o.kind in ("shape", "electrode"))
    funnels = sum(1 for o in objs if o.kind == "electrode")
    tail = f" ({funnels} funnel(s))" if funnels else ""
    await hub.log(f"Imported {name}: {len(objs)} object(s), "
                  f"{filled} filled shape(s){tail} "
                  f"[placed over {size_um:g} um from {size_source}]")
    # SANITY (not a refusal): a design mapped partly outside the scan it was drawn
    # over is placed anyway, but flagged so the operator can see it before writing.
    outside, bbox = frame_bbox_check(hub.pattern, size_um)
    if outside:
        await hub.log(f"WARNING: {name} extends outside the {size_um:g} um frame "
                      f"(lab bbox {bbox} um) -- part of it is off the scan.")
    await hub.notify_design_changed()   # M1: SVG import replaces the whole design
    return {"ok": True, "design": hub.pattern.to_dict(),
            "origin": {"size_um": size_um, "source": size_source,
                       "outside_frame": outside}}


async def design_svg_export():
    """Export the current design's shapes back out as SVG."""
    from fastapi.responses import Response

    from ...litho.bezier import BezierPath, to_svg
    pat = getattr(hub, "pattern", None)
    paths = []
    for o in (pat.objects if pat else []):
        if o.bezier:
            paths.append(BezierPath.from_dict(o.bezier))
    if not paths:
        return {"error": "no Bezier shapes in the current design"}
    return Response(to_svg(paths), media_type="image/svg+xml")


# ------------------------------------------------- design background swap --
# The Scan tab's "Update design background" button: copy the loaded design
# SVG with its background `<image>` swapped for the newest scan PNG, at the
# scan's physical size (docs/IMAGE_SAVING_NOTES.md 2026-09-17). Deliberately
# independent of Slice A's writer internals -- this only globs the CONTRACT
# paths (`scans/png/<stem>_<channel>.png`, sidecar `scans/json/<stem>.json`)
# under the runtime dir, so it works with whatever wrote them. `hub.store`'s
# parent is the runtime dir throughout this service (see host/scan_pump.py); a
# test retargets it exactly the way test_scan_last.py does, by swapping
# `hub.store` for a fresh SessionStore over a tmp_path.


def _runtime_dir() -> Path:
    return hub.store.path.parent


def _latest_scan_stem(runtime_dir: Path) -> str | None:
    """The newest scan stem with ANY channel PNG, across the whole `png/`
    directory -- used only to report "what IS available" in an error/listing,
    never to pick the file a swap actually uses (see `_channel_png_path`,
    which globs the requested channel directly, per-channel newest)."""
    png_dir = runtime_dir / "scans" / "png"
    if not png_dir.is_dir():
        return None
    stems = {p.stem.rsplit("_", 1)[0] for p in png_dir.glob("*_*.png")}
    return max(stems) if stems else None


def _channels_for_stem(runtime_dir: Path, stem: str) -> list[str]:
    png_dir = runtime_dir / "scans" / "png"
    if not png_dir.is_dir():
        return []
    prefix = f"{stem}_"
    return sorted(p.stem[len(prefix):]
                  for p in png_dir.glob(f"{prefix}*.png"))


def _channel_png_path(runtime_dir: Path, channel: str) -> Path | None:
    """The newest PNG for `channel`, however many scans/channels exist.
    Stems sort lexicographically = chronologically (`YYYYMMDD_HHMMSS`, a
    same-second collision breaking the tie with a zero-padded `_NN`), so the
    last name sorted is the newest -- the rule the task contract specifies."""
    png_dir = runtime_dir / "scans" / "png"
    if not png_dir.is_dir():
        return None
    matches = sorted(png_dir.glob(f"*_{channel}.png"))
    return matches[-1] if matches else None


def _safe_design_name(name: str | None) -> str:
    stem = Path(name or "design").stem or "design"
    return "".join(c if (c.isalnum() or c in "-._") else "_" for c in stem)


async def design_background_channels():
    """Channels available for the background swap: those with a PNG under the
    newest scan's stem. Powers the Scan tab's channel <select> without the
    cockpit reaching into `scans/png` itself."""
    runtime_dir = _runtime_dir()
    stem = _latest_scan_stem(runtime_dir)
    if stem is None:
        return {"channels": [], "stem": None}
    return {"channels": _channels_for_stem(runtime_dir, stem), "stem": stem}


async def design_background(body: dict | None = None):
    """Write a copy of the loaded design SVG with its background `<image>`
    swapped for the newest scan PNG of `channel` (default "height"), at the
    scan's physical size.

    400s, in order, on the three ways this cannot proceed (docs/
    IMAGE_SAVING_NOTES.md 2026-09-17): no design has been loaded (no SVG
    upload this session), no scan PNG exists yet for the requested channel
    (message lists the channels the newest scan DOES have), or the scan's
    sidecar JSON -- the only place `size_um` lives -- is missing.
    """
    from fastapi import HTTPException

    src = getattr(hub, "design_source", None)
    if not src:
        raise HTTPException(
            status_code=400,
            detail="no design loaded -- import an SVG first (Load SVG)")

    channel = str((body or {}).get("channel") or "height").strip().lower()
    runtime_dir = _runtime_dir()
    png_path = await asyncio.to_thread(_channel_png_path, runtime_dir, channel)
    if png_path is None:
        stem = await asyncio.to_thread(_latest_scan_stem, runtime_dir)
        if stem is None:
            detail = "no scan taken yet -- there is no PNG to use as a background"
        else:
            avail = _channels_for_stem(runtime_dir, stem)
            detail = (f"no '{channel}' scan PNG yet -- the latest scan ({stem}) "
                      f"has: {', '.join(avail) if avail else 'nothing'}")
        raise HTTPException(status_code=400, detail=detail)

    stem = png_path.stem.rsplit("_", 1)[0]
    sidecar_path = runtime_dir / "scans" / "json" / f"{stem}.json"
    if not sidecar_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"no sidecar for scan {stem} -- its physical size is unknown")
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        size_um = float(sidecar["size_um"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"sidecar for scan {stem} is unreadable ({exc}) -- "
                   f"its physical size is unknown") from exc

    from ..svg_background import swap_background
    png_href = f"../png/{png_path.name}"
    new_svg = await asyncio.to_thread(
        swap_background, src["text"], png_href, size_um)

    design_name = _safe_design_name(src.get("name"))
    svg_dir = runtime_dir / "scans" / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    out_path = svg_dir / f"{stem}_{channel}_{design_name}.svg"
    out_path.write_text(new_svg, encoding="utf-8")

    await hub.log(f"Design background updated: {out_path.name} "
                  f"<- {png_path.name} ({size_um:g} um)")
    return {"ok": True, "path": out_path.resolve().as_posix(), "channel": channel}


# ------------------------------------------------------------ design files --

async def design_files():
    return {"ok": True, **await design_cmds.list_designs(hub)}


async def design_save(body: dict | None = None):
    try:
        return {"ok": True, **await design_cmds.save_design(hub, (body or {}).get("name"))}
    except CommandError as exc:
        return exc.http_body()


async def design_open(body: dict | None = None):
    body = body or {}
    try:
        return {"ok": True, **await design_cmds.open_design(
            hub, body.get("name"), body.get("mode"))}
    except CommandError as exc:
        return exc.http_body()
