"""The FLEX wire surface: the method table, the handlers, and the JSON-RPC error
vocabulary. Transport lives next door in `flex_provider.py`.

Truth document: docs/FLEX_INTEGRATION_PLAN.md Sec A (method table), A.1 (results),
A.3 (telemetry), C.1 (`disconnected`), C.2 (`require_backend` / -32010), F.2 (bounds),
F.3 (state machine). Reader-facing summary: docs/FLEX_PROVIDER.md.

WHAT THE VERBS CAN DO TO THE TIP
--------------------------------
`abort` and `safePark` only RETRACT: they route through `service/host/safe_park.safe_park`,
the coroutine the cockpit's Abort button fires. `acquireControl`, `releaseControl` and
`heartbeat` arbitrate who may command (plan Sec E).

`setMode`, `startApproach`, `withdraw`, `setZGain`, `startScan`, `setContinuous`,
`loadPattern`, `startWrite` and `abortWrite` go through `service/commands/`, the same
coroutines the cockpit's endpoints call, so they inherit the engaged gate, the silent-twin
refusal, the free-air guards and the single-writer claim. There is no second copy of a
guard on this wire (plan Sec C). `startWrite` puts VOLTAGE on a tip on the surface and is
contact-only: the AC guard reads the live crosspoint and refuses, and an unreadable
crosspoint refuses rather than downgrading to "contact" (plan Sec J risk 11).

`executePass` is RESERVED: it answers `-32601 not implemented in this build`, so a script
that sends it is told the truth rather than a "method not found" that reads like a typo.

Every read handler is a pure read of a hub attribute, a cached dict or a
terminal record, and none of them issues a bridge read (plan Sec B.5 forbids
instrument I/O on the REP threads outright). Every mutating verb is `command_only`,
so the read socket refuses it: 29181 stays a plane that cannot move the tip at all.
`getMode` is served on both sockets, because it is a memory read and a
script should not have to queue behind a withdraw to learn whether a mode switch
would even be accepted.

THE getState DERIVATION TABLE (plan Sec F.3)
--------------------------------------------
`getState` answers with exactly one of
`disconnected | idle | approaching | engaged | scanning | writing | measuring |
parked | fault`. First matching row wins, top to bottom -- and it is a pure in-memory
read, so it answers in well under 50 ms at all times, including mid-run:

    #  condition (on `hub`)                                        -> state
    1  fault is not None                                            fault
       (a latched terminal fault: the bridge dropped, or a verify failed)
    2  backend == "hardware" and backend_live() != "hardware"       disconnected
       (CONFIGURED hardware with no live bridge -- pre-connect or post-drop.
        NEVER `idle`: a polling script must see the truth on its first call.
        Terminal for the process; `serve()` has no re-accept path, plan C.1.)
    3  a WRITE run record is in flight (or litho_thread is alive)   writing
       (the record, not the thread: the state must not leave `writing`
        while the pump is still persisting -- plan A.1)
    4  _approach_in_flight                                          approaching
    5  _scan_task in flight, or the twin scan is running,            scanning
       or scan_repeat["enabled"] (a continuous run between frames)
    7  _safe_park["active"]                                         parked
    8  engaged                                                      engaged
    9  -                                                            idle

Row 1 before row 2 on purpose: a bridge that dropped latches BOTH `fault` and the
non-live backend, and `fault` is the more specific, more actionable answer (plan Sec
E.3 maps a bridge failure to `fault`). A station that was configured for hardware and
never saw Igor dial in has no fault, so it reads `disconnected`.

Row 4 before row 5 so a continuous loop's between-frame re-reference (which withdraws,
reads free-air and re-approaches while `scan_repeat["enabled"]` is still true) reports
`approaching` -- what the tip is actually doing -- rather than `scanning`.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from dataclasses import dataclass
from typing import Any

from .. import __version__

# the ONE per-out-wave DSP cap (87,000 -- the v19 vendor `cMaxOutWave`, LOWER than
# the v16 reference's 87,380; see litho/chunker). Published in getCapabilities.
from ..litho.chunker import MAX_OUTWAVE_POINTS
from . import access as _access
from . import control as _ctl
from .host import deadman as _deadman
from .commands.errors import (
    BACKEND_MISMATCH,
    BUSY,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,  # noqa: F401 -- re-exported; flex_provider reads it off here
    METHOD_NOT_FOUND,
    PARSE_ERROR,  # noqa: F401 -- re-exported, as above
    REFUSED,
)

# The Height band the plain `withdraw` reports `verified` against: |Height| <= 0.5 V
# (rig-confirmed; engaged Height sits at tens of volts, a withdrawn one at 0). Shared with
# instrument/asylum.py's withdraw check. The park itself confirms on Height < 1 V instead.
from .host.safe_park import _WITHDRAWN_HEIGHT_TOL_V as _Z_PARK_TOL_V

# ------------------------------------------------------------- error vocabulary --
# The JSON-RPC 2.0 standard codes plus this provider's own, imported above:
#
#   -32010 BACKEND_MISMATCH  plan Sec C.2 -- the mutating verbs' silent-twin gate
#   -32020 BUSY              plan Sec F.1 -- a start verb arrived while busy
#   -32021 REFUSED           a STATE refusal: a mode switch with the tip on the
#                            surface. Distinct from busy, because retrying does not
#                            fix it -- a withdraw does.
#
# `lockin.py` (ZmqJsonRpc) is the client half of the same dialect and unwraps exactly
# this shape. The numbers are DEFINED IN `commands/errors.py`, beside the refusals
# that raise them, and re-exported here under the names the wire and its tests have
# always used: a second copy of these integers is how a -32010 becomes a -32011 in
# one file and nowhere else.
# The ARBITRATION codes (plan Sec E.2) are defined in service/control.py, beside the
# state they describe, and imported here so the wire and the hub cannot drift apart.
CONTROL_REVOKED = _ctl.CONTROL_REVOKED     # -32011: you HAD it; a human took it, or
#                                            you went silent and the dead-man fired
CONTROL_REQUIRED = _ctl.CONTROL_REQUIRED   # -32030: no token, or one we never issued
CONTROL_HELD = _ctl.CONTROL_HELD           # -32040: someone else is the commander

# Rendered into each error message, so the reply reads as a NAME and not just a
# number -- FLEX has no schema anywhere, and a driver author reading a log needs the
# vocabulary in the string.
_CODE_NAMES = {CONTROL_REVOKED: "control_revoked",
               CONTROL_REQUIRED: "control_required",
               CONTROL_HELD: "control_held",
               BUSY: "busy",
               BACKEND_MISMATCH: "backend_mismatch",
               REFUSED: "refused"}

# Published in getCapabilities and consumed by the FLEX driver to derive each call's
# timeout (plan Sec F.2: the driver must not hardcode numbers that will drift).
IN_BANKS = 3
INLINE_CAP_BYTES = 2 * 1024 * 1024

# Twin-only / no-op surfaces a script must not assume are real (plan Sec J risk 9).
TWIN_ONLY = ["force_curve", "tune_thermal", "tune_sweep", "td_ReadString",
             "td_WriteString", "td_StopOutWaveBank", "td_StopInWaveBank"]


class RpcError(Exception):
    """A JSON-RPC error object with a code, raised by a handler and rendered by the
    provider. Anything else a handler raises becomes -32000 (plan Sec B.4)."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = int(code)
        self.message = str(message)
        self.data = data


@dataclass(frozen=True)
class Method:
    """One wire method. `bound_s` is the HONEST worst case (plan Sec F.2), published
    in `getCapabilities.method_bounds`.

    `command_only` marks a verb the READ socket must refuse. The read socket exists
    to never block and never queue (plan Sec B.4), and a verb that can retract the
    tip belongs on the serialised command plane -- so 29181 answers -32601 for it,
    which a test asserts from the wire rather than trusting.
    """

    name: str
    params: str
    result: str
    bound_s: float
    doc: str
    command_only: bool = False


@dataclass(frozen=True)
class Ctx:
    """One dispatch's context: which socket it arrived on, and how to reach the loop.

    `on_loop` is `FlexProvider.on_loop` -- a BOUNDED `run_coroutine_threadsafe`
    whose timeout comes from the method's own published bound, so the provider never
    waits materially longer than the client was told to (plan Sec F.2).
    """

    role: str = "command"
    on_loop: Any = None


# ============================================================ built-ins / status ==

def _real_bridge(hub):
    """The REAL hardware ARC bridge attached to `hub`, or None on the twin.
    Local + lazy (imports only SoftwareARC, never the hub singleton) so
    importing this module stays cheap -- see hub.real_bridge for the rule."""
    from ..twin.arc import SoftwareARC
    arc = getattr(hub.host, "hardware_arc", None)
    if arc is None or isinstance(arc, SoftwareARC):
        return None
    return arc


def _h_ack(hub, params) -> str:
    """Reads no hub state at all, so it answers while a write runs."""
    return "ACK"


def _h_idn(hub, params) -> dict:
    return {
        "Manufacturer": "Levy Lab",
        "Model": "afm-litho",
        "Serial Number": _microscope_id(hub) or "twin",
        "Firmware": __version__,
        # The anti-fallback field (plan Sec C): hardware | twin | disconnected,
        # derived live from the bridge, never from a config read.
        "Backend": hub.backend_label(),
    }


def _h_help(hub, params) -> Any:
    """No argument: the method list. `{"command": name}`: that method's card."""
    name = _opt_str(params, "command", 0)
    if name is None:
        return sorted(_verbs())
    m = _verbs().get(name)
    if m is None:
        raise RpcError(INVALID_PARAMS, f"no such method: {name}",
                       {"methods": sorted(_verbs())})
    return {"method": m.name, "params": m.params, "result": m.result,
            "bound_s": m.bound_s, "doc": m.doc}


def get_state(hub) -> str:
    """The state machine (plan Sec F.3), per the derivation table in the module docstring.
    Its one home is the host's `deadman.pass_state`, which the dead-man reads."""
    return _deadman.pass_state(hub)


def _write_in_flight(hub) -> bool:
    """Is a write still going, persistence included (host `deadman.write_in_flight`)."""
    return _deadman.write_in_flight(hub)


def _h_get_state(hub, params) -> str:
    from . import board
    return board.state(hub)


def _h_get_status(hub, params) -> dict:
    from . import board
    return board.status(hub)


def _h_get_capabilities(hub, params) -> dict:
    arc = _real_bridge(hub)
    return {
        "microscope_id": _microscope_id(hub),
        "is_cypher": bool(getattr(arc, "is_cypher", False)) if arc is not None else False,
        "backend": hub.backend_label(),
        # Same split as getStatus (docs/FLEX_INTEGRATION_PLAN.md Sec J risk 2):
        # the raw configured ARC backend.
        "arc_backend": getattr(hub.host, "backend", None),
        "modes": ["contact", "ac"],
        "max_outwave_points": MAX_OUTWAVE_POINTS,
        "in_banks": IN_BANKS,
        "inline_cap_bytes": INLINE_CAP_BYTES,
        "method_bounds": {m.name: m.bound_s for m in _verbs().values()},
        "twin_only": list(TWIN_ONLY),
        "ports": {"command": _port(), "read": _port() + 1},
    }


def _h_get_instrument(hub, params) -> dict:
    """The instrument model, read-socket side (docs/INSTRUMENT_MODEL_SPEC.md
    section 10: "FLEX gets getInstrument (read) only"). Delegates to the SAME
    function `GET /api/instrument` calls, so the two doors can never drift apart
    on this."""
    from .routers.instrument import get_instrument
    return get_instrument()


def _h_get_last_park(hub, params) -> dict:
    """The most recent park outcome, read-socket side (E1). Same dict as
    `GET /api/park/last`: the `ParkResult` fields (verified, method, z_out_v,
    zsensor_v, defl_v, critical, reason, ts), or `{}` when no park has run this
    process. Read-only; served on both sockets."""
    lp = getattr(hub.host, "last_park", None)
    return lp.to_dict() if lp is not None else {}


def _h_get_telemetry(hub, params) -> dict:
    from . import board
    return board.telemetry(hub)


# ==================================================== pattern / written objects ==

def _h_get_pattern(hub, params) -> dict:
    """The loaded design, as a summary a script can index into: one entry per object,
    with the point COUNT rather than the points themselves (a pattern is tens of
    thousands of points; `loadPattern` in milestone 4 returns the compile report).

    `strokes` is how many CONTINUOUS TIP PATHS the object is -- one transit, one
    completion and one `getWritten` entry each. It is >1 only for a filled shape that
    expands into several disconnected regions; a wire is one stroke however long it
    is (since 2026-09-11 the compiler no longer cuts a stroke at 8,192 points).

    `segments` is how many OUT-WAVES are actually ARMED. It equals `strokes` unless a
    stroke is longer than the DSP's per-out-wave cap, in which case it is fired as a
    seamless chain of chunks -- more arms, but the tip stays down and the bias stays
    live across every boundary, so a chain seam is NOT a lift.

    Both come from `loadPattern`'s cached compile report when that report belongs to
    the pattern loaded now; a pattern set some other way (a sketch endpoint, a design
    edit) falls back to a floor from the DESIGN point count, which underestimates a
    filled shape exactly as `startWrite`'s geometric `est_s` fallback does.
    """
    from .host.litho_cmds import _chunks_for, _object_view
    pattern = getattr(hub, "pattern", None)
    report = getattr(hub, "_pattern_report", None) or {}
    fresh = report.get("pattern_id") == id(pattern)
    by_object = (report.get("segments") or {}) if fresh else {}
    strokes_by_object = (report.get("strokes") or {}) if fresh else {}
    objects = []
    for i, obj in enumerate(getattr(pattern, "objects", None) or []):
        pts = getattr(obj, "points_um", None) or []
        name = getattr(obj, "name", None)
        entry = {
            "index": i,
            "name": name,
            "kind": getattr(obj, "kind", None),
            # `voltage`/`speed`/`enabled`/`written` are ALSO in `properties` now, but
            # kept flat here too: this shape predates the shared listing and callers
            # (the cockpit, tests) read them off the top level.
            "voltage": _finite(getattr(obj, "voltage", None)),
            "speed": _finite(getattr(obj, "speed_um_s", None)),
            "points": len(pts),
            "segments": int(by_object.get(name, _chunks_for(len(pts)))),
            "strokes": int(strokes_by_object.get(name, 1)),
            "enabled": bool(getattr(obj, "enabled", True)),
            "written": id(obj) in (getattr(hub.host, "written_objects", None) or set()),
        }
        # `id`/`vertices_um`/`properties` -- the SAME block loadPattern and setObjects
        # return, from the one helper, so the listing is identical across the verbs.
        entry.update(_object_view(obj))
        objects.append(entry)
    return {"name": getattr(pattern, "name", None), "objects": objects,
            "frame": _frame_dict(hub)}


def _h_get_written(hub, params) -> dict:
    """What this session has actually written (plan Sec A: `{objects:[{index, t_start,
    t_end, aborted}]}`).

    ONE ENTRY PER STROKE, NOT PER OBJECT, and they are deliberately not aggregated: a
    filled shape arrives as several entries whose `name` is `"<object>#k"` (the
    compiler's own label for an expanded part) and whose `object` field is the design
    object they belong to. That is the truth of what the instrument did -- each entry
    is one continuous tip path, with a lift and a set-down either side of it -- and a
    script that wants per-object totals should group by `object`.

    A CHAIN CHUNK IS NOT AN ENTRY. An over-cap stroke is armed as several out-waves,
    but the tip never comes up between them, so it is one stroke and one entry; the
    extra arms show up in `segments`, never here. (Before 2026-09-11 an entry was one
    fired SEGMENT and a long wire produced one per 8,192 points, because the compiler
    cut it there and each cut really was a lift -- that cut is gone.)

    `segments` on each entry is how many out-waves that object arms in all, so a
    script can tell a whole object from a partial one without a `getPattern` round
    trip.

    DELIBERATE DEVIATION from "mirror /api/litho/written": that endpoint returns a
    conductance RASTER sampled off the simulated dose field (`{n, g:[...]}`), not a
    list of written objects -- it answers "what does the surface look like", which is
    not what a script polling a write needs. The written-object list is
    `hub.written_accum`, accumulated by the litho pump per completed object, plus the
    terminal record's aborted flag.
    """
    run = hub.runs.last("write", completed_only=False)
    report = getattr(hub, "_pattern_report", None) or {}
    by_object = report.get("segments") or {}
    objects = []
    for i, entry in enumerate(getattr(hub.host, "written_accum", None) or []):
        t_end = entry.get("t")
        dur = entry.get("duration_s")
        parent = entry.get("object")
        objects.append({
            "index": i,
            "name": entry.get("name"),
            "object": parent,
            "segments": int(by_object.get(parent, 1)),
            "voltage": _finite(entry.get("voltage")),
            "length_um": _finite(entry.get("length_um")),
            "points": len(entry.get("path_um") or []),
            "t_start": (t_end - dur) if (t_end is not None and dur is not None)
            else None,
            "t_end": t_end,
        })
    return {"objects": objects,
            "run_id": run.run_id if run is not None else None,
            "aborted": bool(run.aborted) if run is not None else False,
            "backend": hub.backend_label()}


# =============================================================== bulk results ====

def _h_get_scan_result(hub, params) -> dict:
    return _result(hub, "scan", params)


def _h_get_write_result(hub, params) -> dict:
    """As `getScanResult`, plus the WRITTEN-PATH TRACE when `inline` is asked for.

    Plan Sec A calls conductance-along-path "the payload goal 2 cares about", and the
    raw `.npz` capture that `data` carries is a HARDWARE artefact -- on the twin there
    is none, and even on hardware it is indexed by sample, not by object. `trace` is
    the per-object path the pump recorded (x/y in lab microns, with the tip voltage
    that was applied along it), so a script can line a lock-in series up against tip
    position without opening a file. It is bounded when it is remembered, not here
    (`litho_cmds.remember_trace`), and it is a pure memory read -- so an ABORTED run
    still hands back exactly the objects that completed.
    """
    out = _result(hub, "write", params)
    if bool(_opt(params, "inline", 1) or False):
        from .host.litho_cmds import trace_for
        trace = trace_for(hub, out.get("run_id"))
        if trace is not None:
            out["trace"] = trace
    return out


def _result(hub, kind: str, params) -> dict:
    """Plan Sec A.1. `run_id` omitted = the last COMPLETED run of this kind."""
    run_id = _opt_str(params, "run_id", 0)
    inline = bool(_opt(params, "inline", 1) or False)
    decimate = _opt(params, "decimate", 2)
    if run_id:
        rec = hub.runs.get(run_id)
        if rec is None:
            raise RpcError(INVALID_PARAMS, f"unknown run_id: {run_id}",
                           {"known": [r.run_id for r in hub.runs.all()]})
        if rec.kind != kind:
            raise RpcError(INVALID_PARAMS,
                           f"run {run_id} is a {rec.kind} run, not a {kind} run")
    else:
        rec = hub.runs.last(kind, completed_only=True)
        if rec is None:
            raise RpcError(INVALID_PARAMS, f"no completed {kind} run to return",
                           {"kind": kind})
    out = {"run_id": rec.run_id, "kind": rec.kind, "backend": rec.backend,
           "completed": bool(rec.completed), "aborted": bool(rec.aborted),
           "partial": bool(rec.partial), "shape": rec.shape,
           "channels": list(rec.channels), "files": dict(rec.files),
           "stats": dict(rec.stats), "record_id": rec.record_id,
           "started": rec.started, "finished": rec.finished, "error": rec.error,
           "run": rec.run_ref}
    if inline or decimate is not None:
        step = 1 if decimate is None else max(1, int(decimate))
        arrays = _load_arrays(rec)            # dict[str, np.ndarray] -- NOT `.tolist()`'d
        total_points = sum(int(arr.size) for arr in arrays.values())
        bpp, overhead = _sample_bytes_per_point(arrays)
        # PROJECT first, from a SAMPLE, before touching the real (possibly
        # multi-megapoint) array at all -- review round 2: the previous fix (exact
        # encoding) was correct about what to trust, but encoded the FULL array to
        # get there, which is the one thing a REP-thread handler with a 0.2 s bound
        # must never do (measured 22 s on a 4096^2 x4 capture). A projection can be
        # wrong in either direction, so it only ever decides whether to attempt the
        # real (bounded) encode below -- it never itself becomes `bytes` in a
        # successful reply.
        projected = _projected_bytes(total_points, step, bpp, overhead)
        if projected > INLINE_CAP_BYTES:
            need = _required_decimate(arrays, step, total_points, bpp, overhead)
            raise RpcError(
                INVALID_PARAMS,
                f"result is an estimated {int(projected)} bytes at decimate={step}, "
                f"over the {INLINE_CAP_BYTES}-byte cap; re-request with "
                f"decimate={need}",
                {"required_decimate": need, "bytes": int(projected), "decimate": step,
                 "inline_cap_bytes": INLINE_CAP_BYTES})
        # The projection says this candidate fits. Confirm it for REAL before it can
        # go on the wire (plan Sec A.1: an estimate must never be the last word on
        # what actually gets sent) -- bounded cost, because a candidate the
        # projection places under a megabyte-scale cap cannot itself be
        # multi-megapoint.
        sent, nbytes = _encode_candidate(arrays, step)
        if nbytes > INLINE_CAP_BYTES:
            # The sample under-projected (rare: e.g. an unusually compressible
            # sample next to a less compressible remainder). Re-derive from the
            # same verify-and-bump path, which checks the REAL encoding at each
            # candidate step rather than trusting arithmetic twice in a row.
            need = _required_decimate(arrays, step, total_points, bpp, overhead)
            raise RpcError(
                INVALID_PARAMS,
                f"result is {nbytes} bytes at decimate={step}, over the "
                f"{INLINE_CAP_BYTES}-byte cap; re-request with decimate={need}",
                {"required_decimate": need, "bytes": nbytes, "decimate": step,
                 "inline_cap_bytes": INLINE_CAP_BYTES})
        # `inline` wins when both are given: `data` and `preview` would otherwise
        # carry the IDENTICAL decimated array, doubling the reply for nothing --
        # review round 2 found this made an "under the cap" reply actually ~2x the
        # cap on exactly the path `required_decimate` tells a client to retry on.
        if inline:
            out["data"] = sent
            if step > 1:
                out["decimate"] = step        # say so: `data` is NOT the full array
        else:
            out["preview"] = sent
            out["decimate"] = step
    return out


def _wire_bytes(payload: dict) -> int:
    """The exact encoded size of an already-plain-Python (post-`.tolist()`) channel
    dict, in the same JSON shape it will actually ride the wire in.

    A per-float BYTE ESTIMATE is not safe in either direction: JSON floats are not
    fixed-width (measured ~40.5 B/pt for full-precision values like
    `-3.141592653589793`, not the 20 B/pt this used to assume), so an "under the
    cap" estimate could in fact be roughly double the real cap. This function is
    exact, but it must only ever be called on a candidate whose SIZE is already
    bounded -- a sample, or a decimated slice -- never the full array; see
    `_sample_bytes_per_point` and `_encode_candidate`."""
    return len(json.dumps(payload).encode())


_SAMPLE_POINTS = 1024


def _sample_bytes_per_point(arrays: dict) -> tuple[float, int]:
    """Estimate `(bytes_per_point, fixed_overhead)` from up to `_SAMPLE_POINTS`
    points of each channel, EVENLY SPACED across it -- never the full array, so
    this costs the same whether the capture is 512^2 or 4096^2 x4. Evenly spaced
    rather than the first N: a channel that ramps (a topography line, or -- worst
    case in testing -- `np.arange`) has a value range, and so a JSON width, that
    grows across the array; sampling only the head under-represents the tail
    encoding of a candidate later stridden through the whole thing. JSON floats
    are not fixed-width in general, so `bytes_per_point` is a blended average
    across channels, not an exact figure; `fixed_overhead` is the cost of the
    dict/key/bracket scaffolding with no elements at all, which does not grow with
    the array. Both feed a PROJECTION -- `_result` still confirms the real
    candidate by encoding it before it goes on the wire, and `_required_decimate`
    corrects for a bad projection by measuring the actual candidate, not by
    trusting this estimate twice."""
    overhead = len(json.dumps({k: [] for k in arrays}).encode())
    sample = {}
    for k, v in arrays.items():
        n = v.size
        stride = max(1, n // _SAMPLE_POINTS) if n > _SAMPLE_POINTS else 1
        sample[k] = v[::stride][:_SAMPLE_POINTS].tolist()
    sample_points = sum(len(v) for v in sample.values())
    if sample_points == 0:
        return 0.0, overhead
    sample_bytes = len(json.dumps(sample).encode())
    return max(0.0, (sample_bytes - overhead) / sample_points), overhead


def _projected_bytes(total_points: int, step: int, bpp: float, overhead: int) -> float:
    """The estimated encoded size of the full array decimated by `step`, from the
    sampled bytes-per-point -- no array touched, let alone encoded."""
    kept = total_points / step if step else 0.0
    return overhead + kept * bpp


def _encode_candidate(arrays: dict, step: int) -> tuple[dict, int]:
    """Slice with NUMPY first (a view, not a copy -- O(1)) and only THEN convert to
    plain Python floats/ints with `.tolist()`. Calling `.tolist()` on the FULL array
    before slicing is exactly the O(n) cost this fix exists to avoid; this function
    must only ever be called on a `step` the caller already believes keeps the
    result near or under the cap, so the slice `.tolist()` touches is bounded."""
    sliced = {k: (v[::step] if step > 1 else v) for k, v in arrays.items()}
    jsonable = {k: v.tolist() for k, v in sliced.items()}
    return jsonable, _wire_bytes(jsonable)


def _required_decimate(arrays: dict, step: int, total_points: int,
                       bpp: float, overhead: int) -> int:
    """The smallest total decimation (relative to the FULL array) whose ACTUAL
    encoding fits under the cap.

    Projects a starting guess from the sampled bytes-per-point, then verifies by
    encoding only the DECIMATED candidate at each guess -- never the full array --
    and corrects by the REAL measured overshoot ratio, not a `+= 1` bump. A sample
    can be systematically wrong (a monotonic channel's value range, and so its
    JSON width, grows across the array; the evenly-spaced sample in
    `_sample_bytes_per_point` narrows this but does not eliminate it), and a
    `+= 1` correction step takes as many tries as the estimate was wrong by --
    measured at ~140 tries, ~0.05 s each, on a synthetic worst case, which is
    exactly the unbounded-latency failure mode this function exists to avoid.
    Scaling by the measured ratio converges in a handful of tries regardless of
    how far off the projection was. A candidate placed near the cap is, by
    construction, at most a few hundred thousand points regardless of how large
    the source array is, so every verify step here is bounded cost."""
    denom = max(1.0, INLINE_CAP_BYTES - overhead)
    need = max(step, math.ceil(total_points * bpp / denom) if bpp > 0 else step, 1)
    for _ in range(64):
        _, nbytes = _encode_candidate(arrays, need)
        if nbytes <= INLINE_CAP_BYTES:
            return need
        scaled = math.ceil(need * (nbytes / INLINE_CAP_BYTES))
        need = scaled if scaled > need else need + 1
    # Unreachable outside a cap smaller than a single point's own encoding.
    raise RpcError(INTERNAL_ERROR, "result cannot be decimated under the inline cap")


def _load_arrays(rec) -> dict:
    """Read a terminal record's `.npz` back as numpy arrays -- NOT plain Python
    lists. `.tolist()` on a multi-megapoint array is the O(n) cost review round 2
    found (67M-element capture, materialised whether or not it was ever going to
    be sent): callers slice with numpy first, and convert to plain Python only the
    bounded slice they actually intend to encode (`_sample_bytes_per_point`,
    `_encode_candidate`).

    Off the REP thread's fast path by construction -- the provider marshals this
    handler like any other -- and it only ever touches files this process wrote."""
    path = rec.files.get("npz") or rec.files.get("frame")
    if path is None:
        for name, p in sorted(rec.files.items()):
            if name.startswith("npz"):
                path = p
                break
    if path is None:
        return {}
    try:
        import numpy as np
        with np.load(path) as z:
            return {k: np.asarray(z[k]).ravel() for k in z.files}
    except Exception as exc:
        raise RpcError(INTERNAL_ERROR,
                       f"could not read {path}: {exc}") from exc


# ======================================== control / arbitration (plan Sec E) ====
#
# The FIRST tip-moving verbs on the wire, and BOTH of them only RETRACT. They live on
# the command socket alone (`command_only=True`) and every one of them reaches the
# instrument through the SAME coroutine the GUI's Abort button uses --
# `service/host/safe_park.safe_park` -- marshalled onto the loop by `ctx.on_loop`.
#
# NEVER `arc.abort()`: that routes to a COM `AFM_Abort()` which stops WITHOUT
# retracting (plan Sec J risk 1). A FLEX abort is a cooperative stop *plus* a
# verified retract, which is the whole reason it is safe to hand a script an
# emergency stop at all.


def _on_loop(hub, coro, ctx, method: str):
    """Run `coro` on the service loop and wait, BOUNDED. The ONE way a handler here
    reaches the loop, and it is called holding no lock (plan Sec B.4).

    `ctx.on_loop` is the provider's bound marshal; the fallback covers a handler
    driven directly (a unit test, `scripts/flex_probe.py` in-process) and uses the
    identical implementation rather than a second copy of it."""
    if ctx is not None and ctx.on_loop is not None:
        return ctx.on_loop(coro, method=method)
    from .flex_provider import run_on_loop
    return run_on_loop(hub, coro, method=method)


def _h_acquire_control(hub, params, ctx):
    client = _opt_str(params, "client", 0)
    if not client:
        raise RpcError(INVALID_PARAMS, "acquireControl needs a `client` name -- it "
                                       "is what the operator sees in the cockpit "
                                       "chip and in the log")
    # VALIDATE, then clamp. The two are different failures and only one of them is
    # safe to paper over: a NUMBER outside [5, 300] is a request the provider can
    # honour approximately (clamping fails safe), but `deadman_s: "abc"` is a typo,
    # and silently substituting the 30 s default would hand a script a dead-man it
    # never asked for and does not know about.
    deadman = _opt(params, "deadman_s", 1)
    if deadman is not None and (isinstance(deadman, bool)
                                or not isinstance(deadman, (int, float))):
        raise RpcError(INVALID_PARAMS,
                       f"deadman_s must be a number of seconds (clamped to "
                       f"[{_ctl.DEADMAN_MIN_S:g}, {_ctl.DEADMAN_MAX_S:g}]), got "
                       f"{type(deadman).__name__}")
    # `withdraw_on_deadman` decides whether a silent script leaves the tip ON the
    # surface, so it must be an ACTUAL bool: `bool("false")`, `bool("no")` and
    # `bool("0")` are all True, which would turn "do not withdraw" into a withdraw
    # -- or, worse for a caller who meant it, read as set when it was a typo.
    withdraw = _opt(params, "withdraw_on_deadman", 2)
    if withdraw is not None and not isinstance(withdraw, bool):
        raise RpcError(INVALID_PARAMS,
                       f"withdraw_on_deadman must be true or false, got "
                       f"{type(withdraw).__name__} -- it decides whether a silent "
                       f"script leaves the tip on the surface, so it is never "
                       f"inferred from a string")
    withdraw = bool(withdraw)

    async def _go():
        rec = _access.script_requests(hub, client, deadman, withdraw)
        await hub.log(f"Control: {rec.client} acquired the instrument "
                      f"(dead-man {rec.deadman_s:g}s, withdraw_on_deadman="
                      f"{bool(rec.withdraw_on_deadman)})")
        await hub.broadcast_control()
        return {"token": rec.token, "expires_s": rec.expires_s(),
                "deadman_s": rec.deadman_s,
                "withdraw_on_deadman": bool(rec.withdraw_on_deadman)}

    return _on_loop(hub, _go(), ctx, "acquireControl")


def _h_release_control(hub, params, ctx):
    """NEVER an error, whatever the token -- and that has to include "the loop is
    busy".

    Plan Sec H puts this call in a `finally`, so it must not raise. Marshalling it to
    the loop like its siblings would make it raise -32000 whenever the loop did not
    answer within the published bound: the ONE call a script makes while cleaning up
    after something has already gone wrong is the one most likely to meet a busy
    loop. So the release itself runs HERE, on the REP thread -- `release_control` is
    a compare-and-swap under `hub._control_lock`, safe from any thread by
    construction -- and only the log and the broadcast (which need the loop) are
    handed over, fire-and-forget. The token is gone the moment this returns, whether
    or not anyone ever hears about it.
    """
    token = _opt(params, "token", 0)
    out = _access.release(hub, token)
    if not out["stale"]:
        _notify(hub, _released(hub))
    return out


async def _released(hub) -> None:
    await hub.log("Control: released")
    await hub.broadcast_control()


def _notify(hub, coro) -> None:
    """Fire a coroutine on the loop and DO NOT WAIT. For the reporting half of an
    action whose state change has already happened on this thread: a log line that
    cannot be delivered must never turn into an error the caller sees."""
    loop = getattr(hub, "_loop", None)
    if loop is None or not loop.is_running():
        coro.close()
        return
    try:
        asyncio.run_coroutine_threadsafe(coro, loop)
    except Exception:                            # noqa: BLE001
        coro.close()


def _h_heartbeat(hub, params, ctx):
    """Refresh the dead-man. A revoked or expired token raises -32011 rather than
    answering `valid: false` (see docs/FLEX_PROVIDER.md, "Deviation from plan E.1"):
    one refusal shape for every token-bearing verb is easier to get right in a
    driver than two. The error `data` still carries `valid: false`, so a client that
    wants the soft answer has it without parsing the message.

    An ACCEPTED heartbeat also re-broadcasts the `control` ws event (the same one
    `acquireControl`/`releaseControl` fire) so the cockpit countdown re-seeds from
    the fresh `expires_s` instead of running down from the acquire-time snapshot. A
    REJECTED heartbeat (bad/absent token, no holder) never gets this far --
    `require_token` raises first -- so it broadcasts nothing.

    The broadcast is SPAWNED, not awaited: `heartbeat` has a 1 s loop budget, and
    `hub.broadcast` is a sequential, untimed fan-out to every websocket client (see
    `ControlMixin.spawn_control_broadcast`). Awaiting it here would let one stalled
    client make the heartbeat reply late -- the exact failure being fixed. The reply
    below returns as soon as the dead-man is stamped.
    """
    token = _opt(params, "token", 0)

    async def _go():
        rec = _access.check(hub, token)          # -32011 / -32030
        _access.refresh(hub, rec.token)
        hub.spawn_control_broadcast()
        cur = hub.control
        return {"state": get_state(hub), "valid": True,
                "expires_s": cur.expires_s() if cur is not None else None}

    return _on_loop(hub, _go(), ctx, "heartbeat")


def _h_abort(hub, params, ctx, stop_id=None):
    """TOKEN-FREE (plan Sec A/E.2): an emergency stop must never be blocked by
    arbitration, from either side. Cooperative stop of whatever is in flight PLUS a
    verified retract, through the same `safe_park` the GUI Abort button fires.

    `scope` is accepted and recorded in the reason so a caller can say what it meant
    to stop; there is only one stop, and it stops everything."""
    token = _opt(params, "token", 0)
    scope = _stop_str(params, "scope", 1)

    async def _go():
        from .host.api import HostApi
        if isinstance(token, str) and token:
            hub.touch_control(token)            # courtesy: an abort is contact
        reason = f"flex abort ({scope})" if scope else "flex abort"
        res = await HostApi(hub).park(reason, stop_id=stop_id)
        return {"ok": True, "state": get_state(hub), "reason": reason,
                "safe_park": {k: res.get(k) for k in
                              ("reason", "backend", "transport", "com_error",
                               "verified", "z_v", "critical", "verify_timeout")}}

    return _on_loop(hub, _go(), ctx, "abort")


def _h_safe_park(hub, params, ctx, stop_id=None):
    """TOKEN-FREE, same route as `abort`. The difference is intent, not mechanism:
    `abort` says "stop what I started", `safePark` says "get the tip somewhere
    safe" -- and both are one verified retract."""
    reason = _stop_str(params, "reason", 0) or "flex safePark"

    async def _go():
        from .host.api import HostApi
        res = await HostApi(hub).park(reason, stop_id=stop_id)
        return {"ok": True, "state": get_state(hub),
                **{k: res.get(k) for k in
                   ("reason", "backend", "transport", "com_error", "verified",
                    "z_v", "critical", "verify_timeout", "retried")}}

    return _on_loop(hub, _go(), ctx, "safePark")


def _commander(hub) -> dict | None:
    """The `getStatus.commander` field. Token-free by construction."""
    try:
        cur = getattr(hub, "control", None)
    except Exception:                            # noqa: BLE001
        return None
    if cur is None:
        return None
    return {"holder": cur.holder, "client": cur.client, "kind": cur.kind,
            "expires_s": cur.expires_s(),
            "withdraw_on_deadman": bool(cur.withdraw_on_deadman),
            # Repeated inside `commander` as well as beside it: a script reading its
            # own commander record is exactly who needs to know the timer that
            # governs it is alive.
            "deadman_armed": _ctl.deadman_armed(hub)}


# ================================== instrument operations (plan Sec A) ==========
#
# THE FIRST WIRE VERBS THAT CAN DRIVE THE TIP TOWARD A SURFACE. Every one of them
# goes through `service/commands/` -- the same coroutines `POST /api/approach`,
# `/api/mode`, `/api/withdraw`, `/api/zgain`, `/api/scan/start` and
# `/api/scan/continuous` call -- so a script meets the engaged gate, the silent-twin
# refusal, the free-air guards and the single-writer claim that the cockpit meets,
# because they are literally the same code (plan Sec C, Sec D).
#
# Each handler below does three dull things and no instrument reasoning at all:
#
#   1. TOKEN     -- required on every verb here except `withdraw` (plan Sec A: a
#                   retract must never be blocked by arbitration, from either side);
#   2. BACKEND   -- `require_backend`, the -32010 silent-twin gate (plan Sec C.2).
#                   `withdraw` is exempt for the same reason it needs no token, and
#                   `setContinuous` applies it one layer down, where DISABLING a
#                   running loop stays possible on a station whose bridge just died;
#   3. MARSHAL   -- `_on_loop`, bounded by the method's OWN published bound, holding
#                   no lock (plan Sec B.4).
#
# Everything else -- what a setpoint means, when an engage is unsafe, which run id a
# poll will find -- lives one layer down, where the HTTP door meets it too.


def _body(params, method: str) -> dict:
    """Params as an object, for the verbs that need NAMED fields.

    A bare JSON array is accepted elsewhere for single-argument methods (plan Sec A),
    but not here: every verb using this carries a `token` alongside its arguments, and
    a positional array has nowhere to put one. The stops do not use it (they accept any
    shape). Saying so is kinder
    than a `control_required` a driver author would read as an arbitration bug.
    """
    if params is None:
        return {}
    if isinstance(params, dict):
        return dict(params)
    raise RpcError(INVALID_PARAMS,
                   f"{method} takes a params OBJECT with named fields (it carries a "
                   f"`token`), got {type(params).__name__}",
                   {"method": method})


def _need_token(hub, params: dict, method: str) -> None:
    """The single-commander gate (plan Sec E.2). `hub.require_token` raises
    `control_revoked` (-32011) when the token WAS ours and a human took it or the
    dead-man fired, and `control_required` (-32030) when we never issued it --
    `dispatch` renders both. A token that IS current also refreshes the dead-man,
    which `dispatch` has already done for any command-socket request carrying one."""
    _access.check(hub, params.get("token"))


def _h_set_mode(hub, params, ctx):
    """Contact <-> AC. REFUSED (-32021) while the tip is engaged, scanning or
    writing: the vendor switch changes the Z feedback channel and the gain sign and
    does NOT withdraw, so doing it with the tip down is a crash. Withdraw first."""
    body = _body(params, "setMode")
    _need_token(hub, body, "setMode")
    require_backend(hub)
    from .host.api import HostApi
    return _on_loop(hub, HostApi(hub).set_mode(body), ctx, "setMode")


def _h_start_approach(hub, params, ctx):
    """START+POLL. Returns `{run_id, est_s}` in well under a second; `getState` says
    `approaching` until the run record closes, then `engaged` -- or `parked`, because
    a failed scripted engage fires safe_park (plan Sec F.3)."""
    body = _body(params, "startApproach")
    _need_token(hub, body, "startApproach")
    require_backend(hub)
    from .host.api import HostApi
    return _on_loop(hub, HostApi(hub).start_approach(body), ctx, "startApproach")


def _h_withdraw(hub, params, ctx):
    """Retract. BLOCKING, and deliberately the least gated verb here: no token is
    read and the backend gate is skipped, for the same reason `abort` and
    `safePark` are token-free -- getting the tip OFF the surface must never be
    refused by bookkeeping. (It is still command-socket only: 29181 cannot move the
    tip at all.)

    `verified` is derived the same way the fail-safe derives it: a parked Z within
    the ±0.5 V tolerance, and no read that timed out. An unverified retract that says
    so is strictly better than a verified one that never answers (plan Sec F.2)."""
    body = params if isinstance(params, dict) else {}   # a stop never refuses its body
    from .host.api import HostApi
    snap = _on_loop(hub, HostApi(hub).withdraw(body), ctx, "withdraw")
    snap = snap if isinstance(snap, dict) else {}
    z = _finite(snap.get("z_v"))
    timed_out = bool(snap.get("verify_timeout"))
    verified = (not timed_out) and z is not None and abs(z) <= _Z_PARK_TOL_V
    if snap.get("not_connected"):
        # Before Igor dials in nothing was sent (2026-10-28).
        return {"ok": True, "z_v": None, "verified": False, "critical": False,
                "not_connected": True, "detail": snap.get("detail"),
                "verify_timeout": False, "backend": hub.backend_label(),
                "state": get_state(hub), "snapshot": snap}
    return {"ok": True, "z_v": z, "verified": verified,
            "verify_timeout": timed_out, "backend": hub.backend_label(),
            "state": get_state(hub), "snapshot": snap}


def _h_set_zgain(hub, params, ctx):
    """Re-arm the running Z loop with new P/I gains, without withdrawing. HARDWARE
    ONLY: on the twin there is no servo to re-arm, so it answers -32010 rather than
    reporting a success it did not have (plan Sec A)."""
    body = _body(params, "setZGain")
    _need_token(hub, body, "setZGain")
    require_backend(hub)
    from .host.api import HostApi
    snap = _on_loop(hub, HostApi(hub).set_zgain(body), ctx, "setZGain")
    return {"ok": True, "backend": hub.backend_label(),
            "snapshot": snap if isinstance(snap, dict) else None}


def _h_start_scan(hub, params, ctx):
    """START+POLL: `{run_id, est_s}`, then poll `getState` and fetch
    `getScanResult(run_id)` once it leaves `scanning`. The result's `completed` is
    gated on PERSISTENCE, so the paths in `files` are whole (plan Sec A.1).

    On hardware a `setpoint`, `pgain` or `igain` without `engage: true` is REJECTED
    (-32602) rather than accepted and ignored -- see `commands.scan_cmds.start_scan`.
    `pgain`/`igain` are PANEL gains, the SAME convention `setZGain` takes (scaled to
    controller units and signed by mode; see `ScanConfig.pgain`/`igain`), and only
    replace this scan's default Z gains when given."""
    body = _body(params, "startScan")
    _need_token(hub, body, "startScan")
    require_backend(hub)
    from .host.api import HostApi
    return _on_loop(hub, HostApi(hub).start_scan(body), ctx, "startScan")


def _h_set_continuous(hub, params, ctx):
    """Arm or stop the auto-repeat imaging loop. The state stays `scanning` for as
    long as it runs (plan Sec F.1), and every frame gets its own `run_id`.

    NO `require_backend` HERE, on purpose: the guard belongs to ENABLING, and it is
    applied in the service layer. Disabling must keep working on a station whose
    bridge has just died -- that is exactly when someone wants the loop to stop."""
    body = _body(params, "setContinuous")
    _need_token(hub, body, "setContinuous")
    from .host.api import HostApi
    return _on_loop(hub, HostApi(hub).set_continuous(body), ctx, "setContinuous")


# ======================================== write control (plan Sec I stage 4) ====
#
# THE VERBS THAT PUT VOLTAGE ON A TIP THAT IS ON THE SURFACE. Same construction as
# scan control above: every one goes through `service/commands/litho_cmds.py` -- the
# coroutines `POST /api/litho/load`, `/api/litho/write` and `/api/litho/abort` call --
# so a script meets the contact-only AC guard, the silent-twin refusal and the
# single-writer claim that the cockpit meets, because it is the same code.
#
# `abortWrite` is TOKEN-FREE, like `abort` and `safePark` and for the same reason.


def _h_load_pattern(hub, params, ctx):
    """Compile and validate a design. ARMS NOTHING -- no claim, no thread, no tip.

    TWO DOORS, and EXACTLY ONE per call:

      * `source` -- GEOMETRY: `"auto"`, a design name in `data/designs`, a FILE PATH,
        or an inline SVG document. The wire door passes `allow_paths=True` (plan Sec
        H's script loads `D:/designs/hall_bar_v7.oas`); the cockpit's door stays
        confined to `data/designs`.
      * `objects` -- PARAMETERS: `[{"kind": "chiral"|"region", ...}]`. A design file
        carries geometry and no tip voltage, so the experiments that ARE a voltage
        (a chiral wire's phase relation, an erase rectangle's negative bias) could be
        drawn by a script but never ordered by one. See `litho_cmds._objects_pattern`
        for the spec and `docs/FLEX_PROVIDER.md` for the worked examples.

    Neither, or both, is `-32602`: silently preferring one would make a script that
    sent both write something it never read back.
    """
    body = _body(params, "loadPattern")
    _need_token(hub, body, "loadPattern")
    require_backend(hub)
    source = body.get("source")
    objects = body.get("objects")
    if source is not None and not isinstance(source, str):
        raise RpcError(INVALID_PARAMS,
                       f"loadPattern: source must be a string (a design name, a file "
                       f"path, or an inline SVG), got {type(source).__name__}",
                       {"param": "source"})
    if source is None and objects is None:
        raise RpcError(INVALID_PARAMS,
                       "loadPattern: pass exactly one of `source` (a design name, a "
                       "file path, or an inline SVG) or `objects` (a list of object "
                       "specs); got neither",
                       {"param": "source"})
    if source is not None and objects is not None:
        raise RpcError(INVALID_PARAMS,
                       "loadPattern: pass exactly one of `source` (geometry) or "
                       "`objects` (parameters), not both",
                       {"param": "objects"})
    from .host.litho_cmds import load_pattern
    res = _on_loop(hub, load_pattern(hub, source, objects=objects,
                                     allow_paths=True), ctx, "loadPattern")
    # `design` (the full geometry) is deliberately NOT on the wire: it is tens of
    # thousands of points, and `getPattern` is the verb for looking at the objects.
    return {k: v for k, v in res.items() if k != "design"}


def _h_set_objects(hub, params, ctx):
    """Edit the loaded pattern in place: `{token, updates: {id: {property: value}}}`.

    ARMS NOTHING and moves no tip -- it merges the updates onto the loaded pattern and
    recompiles, so it is token-gated but not a tip-mutating verb. REFUSED `-32020 busy`
    while a write is in flight, because editing the pattern a running write is reading
    would change what it is writing under it. A bad id/property/type, or a property
    invalid for an object's kind, is `-32602` naming the id and field, and the WHOLE
    call is refused with the previous pattern kept."""
    body = _body(params, "setObjects")
    _need_token(hub, body, "setObjects")
    if _write_in_flight(hub):
        msg = ("setObjects: a write is in flight; editing the loaded pattern now would "
               "change what it is writing. Wait for it to finish.")
        raise RpcError(BUSY, msg,
                       {"reason": "write_in_flight", "run_id": hub._in_flight_run_id()})
    from .host.litho_cmds import set_objects
    return _on_loop(hub, set_objects(hub, body.get("updates")), ctx, "setObjects")


def _h_start_write(hub, params, ctx):
    """START+POLL: `{run_id, est_s, objects}`, then poll `getState` until it leaves
    `writing` and fetch `getWriteResult(run_id)`.

    CONTACT-ONLY. On hardware the AC-guard reads the live crosspoint and refuses
    `-32021` if the instrument is in AC/tapping -- the Z loop must never servo on
    amplitude with a bias on the tip. `objects` defaults to `"all"`."""
    body = _body(params, "startWrite")
    _need_token(hub, body, "startWrite")
    require_backend(hub)
    from .host.api import HostApi
    return _on_loop(hub, HostApi(hub).start_write(body), ctx, "startWrite")


def _h_abort_write(hub, params, ctx):
    """TOKEN-FREE stop of a write PLUS a verified retract, through the same
    `safe_park` the GUI Abort button fires. The partial result stays retrievable:
    `getWriteResult` reports `aborted: true` with whatever the run got through."""
    async def _go():
        from .host.api import HostApi
        res = await HostApi(hub).abort_write()
        park = res.get("safe_park") or {}
        return {"ok": True, "state": get_state(hub),
                "backend": res.get("backend"),
                "safe_park": {k: park.get(k) for k in
                              ("reason", "backend", "transport", "com_error",
                               "verified", "z_v", "critical", "verify_timeout")}}

    return _on_loop(hub, _go(), ctx, "abortWrite")
# ============================================================ the verb list =====
# The verbs, their sockets and their HELP cards are the request table's (`requests/table.py`,
# `requests/cards.py`). `REGISTRY`, `READ_ONLY` and `COMMAND_ONLY` are built from it on first
# use, so this module can be imported by the rows that the table gathers.

def __getattr__(name: str):
    if name == "REGISTRY":
        return _verbs()
    if name in ("READ_ONLY", "COMMAND_ONLY"):
        from .requests import table
        return getattr(table, name.lower())()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _verbs() -> dict[str, Method]:
    from .requests.table import registry
    return registry()


# Verbs named in the plan that this build does not serve: told "not implemented", not "not found".
RESERVED = frozenset({
    "executePass",
})

NOT_IMPLEMENTED_MESSAGE = "not implemented in this build"


# ============================================================ backend guard ======

def require_backend(hub) -> None:
    """Plan Sec C.2. Raise `-32010 backend_mismatch` when a MUTATING verb arrives at
    a station configured as hardware whose live backend is not hardware.

    Implemented now, exercised by tests now, called by the mutating verbs when they
    land: the hazard it closes is an overnight script "writing" a whole chip against
    simulated physics while every call returns ordinary success. Reuses the same
    `hub.require_live_backend()` the HTTP endpoints are gated on, so there is one
    rule, not two.
    """
    why = hub.require_live_backend()
    if why:
        raise RpcError(BACKEND_MISMATCH, f"backend_mismatch: {why}",
                       {"backend": hub.backend_label(),
                        "required": getattr(hub.host, "backend", None)})


# ================================================================== helpers ======

def _port() -> int:
    try:
        return int(os.environ.get("AFM_LITHO_FLEX_PORT", "29180"))
    except (TypeError, ValueError):
        return 29180


def _microscope_id(hub):
    cal = getattr(hub.host, "hw_cal", None) or {}
    mid = cal.get("microscope_id")
    if mid in (None, ""):
        return None
    try:
        return int(float(mid))
    except (TypeError, ValueError):
        return str(mid)


def _finite(v):
    """None for anything that cannot survive JSON (NaN/inf are not valid JSON)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _frame_dict(hub):
    try:
        from .host.litho_worker import _litho_frame
        return _litho_frame(hub).to_dict()
    except Exception:                            # noqa: BLE001
        return None


def _opt(params, key: str, index: int):
    """Read one optional argument. `params` may be a dict (keyed), a list (positional
    -- some FLEX drivers send positional, plan Sec A), or absent."""
    if isinstance(params, dict):
        return params.get(key)
    if isinstance(params, (list, tuple)):
        return params[index] if index < len(params) else None
    return None


def _opt_str(params, key: str, index: int):
    v = _opt(params, key, index)
    if v is None:
        return None
    if not isinstance(v, str):
        raise RpcError(INVALID_PARAMS, f"{key} must be a string, got "
                                       f"{type(v).__name__}")
    return v or None


def _stop_str(params, key: str, index: int):
    """`_opt_str` for the stops: a field that is not a string reads as absent, never an error."""
    v = _opt(params, key, index)
    return v if isinstance(v, str) and v else None


def refuse_unknown(method: str) -> None:
    """A verb with no row: -32601, naming it "not implemented" if it is one of the reserved
    verbs and "not found" otherwise."""
    if method in RESERVED:
        raise RpcError(METHOD_NOT_FOUND, NOT_IMPLEMENTED_MESSAGE,
                       {"method": method, "milestone": "control plane"})
    raise RpcError(METHOD_NOT_FOUND, f"method not found: {method}",
                   {"methods": sorted(_verbs())})