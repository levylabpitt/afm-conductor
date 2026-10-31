"""Design files: save, open and list. One function per verb, for the cockpit's
routes and (later) the FLEX handlers alike. Every refusal is a `CommandError`
whose `data` carries a `reason`.

An open never compiles the design and puts no limit on its points. It reads and
builds the file on its own worker thread, under a wall clock.
"""

from __future__ import annotations

import asyncio
import threading

from ...litho import design_file
from .. import design_store
from .errors import BUSY, INTERNAL_ERROR, INVALID_PARAMS, REFUSED, CommandError

MODES = ("add", "replace")
# Measured on the dev machine: 300 wires of 5 um, or 60 of 90 um, build in 4 to 7 s.
BUILD_BOUND_S = 20.0
_INSTALL_ATTEMPTS = 2
_FRAME_TOL = 1e-9

_build_lock = threading.Lock()
_build_worker = None
_building = None


def _refuse(code: int, reason: str, message: str, **data) -> CommandError:
    return CommandError(code, message, {"reason": reason, **data})


def _no_name() -> CommandError:
    return _refuse(INVALID_PARAMS, "bad_name", "a design needs a name", param="name")


def _save_name(raw) -> str:
    name = design_store.file_name(raw) if isinstance(raw, str) else ""
    if not name:
        raise _no_name()
    if design_store.is_reserved(name):
        raise _refuse(INVALID_PARAMS, "reserved_name",
                      f"{name} is a name the service keeps for itself; choose another",
                      param="name")
    return name


def _write_in_flight(hub) -> bool:
    from ..flex_dispatch import _write_in_flight as in_flight
    return in_flight(hub)


def _refuse_during_write(hub) -> None:
    if _write_in_flight(hub):
        raise _refuse(BUSY, "write_in_flight",
                      "a write is in flight; open the design when it has finished",
                      run_id=hub._in_flight_run_id())


async def _write_snapshot(snapshot, name: str) -> dict:
    """Write a snapshot under `name`; returns its document header. Raises OSError
    or ValueError."""
    def work():
        doc = snapshot.document(name)
        design_store.write_document(snapshot.runtime_dir, name, doc)
        return {"saved_at": doc["saved_at"], "object_count": doc["object_count"]}
    return await asyncio.to_thread(work)


def _save_refusal(snapshot, reason: str, lead: str, exc: Exception) -> CommandError:
    why, where = snapshot.describe(exc)
    return _refuse(INTERNAL_ERROR, reason, f"{lead}: {why}", **where)


async def save_design(hub, name) -> dict:
    """Write the current design as `<name>.design.json`. Allowed during a write."""
    name = _save_name(name)
    snapshot = design_store.Snapshot(hub)
    if not snapshot.objects:
        raise _refuse(REFUSED, "no_design",
                      "there is no design to save: nothing has been drawn, imported "
                      "or loaded")
    overwrote = design_store.design_path(snapshot.runtime_dir, name).is_file()
    try:
        head = await _write_snapshot(snapshot, name)
    except (OSError, ValueError) as exc:
        raise _save_refusal(snapshot, "save_failed",
                            f"the design could not be saved as {name}", exc) from exc
    await hub.log(f"Design saved as {name} ({head['object_count']} objects)"
                  + (", replacing the earlier file" if overwrote else ""))
    return {"name": name, "overwrote": overwrote, **head}


async def list_designs(hub) -> dict:
    runtime_dir = hub.store.path.parent
    designs = await asyncio.to_thread(design_store.list_designs, runtime_dir)
    return {"designs": designs}


def _build(path, model, lenient: bool, cancelled):
    """Read the file, build its objects and check them. Runs on the build worker.
    `lenient` (a set-aside autosave) leaves out a record that cannot be built."""
    from ..command_guard import check_design_voltages
    doc = design_store.read_document(path)
    skipped = [] if lenient else None
    try:
        objects = design_file.document_to_objects(doc, skipped=skipped,
                                                  cancelled=cancelled)
    except design_file.BuildCancelled:
        objects = None
    if objects is None:
        # Raised afresh, outside the handler: the first one's traceback holds
        # every object built so far, and the abandoned future keeps its exception.
        raise design_file.BuildCancelled
    check_design_voltages(model, objects)
    left_out = [{"name": exc.object_name, "reason": str(exc)} for exc in skipped or []]
    return doc, objects, left_out


def _submit_build(*args):
    """Start a build on the one build worker, a daemon thread apart from the
    default pool. Refused while an earlier build, abandoned at its wall clock, is
    still running."""
    global _build_worker, _building
    from ..host.litho_cmds import _DaemonSingleWorker
    with _build_lock:
        if _building is not None and not _building.done():
            raise _refuse(BUSY, "build_in_progress",
                          "an earlier design open is still being built; wait for it "
                          "to finish, then open again")
        if _build_worker is None:
            _build_worker = _DaemonSingleWorker(thread_name_prefix="design-build")
        _building = _build_worker.submit(_build, *args)
        return _building


async def _read_and_build(hub, path, name: str):
    cancelled = threading.Event()
    lenient = name in design_store.SET_ASIDE
    building = asyncio.wrap_future(_submit_build(
        path, getattr(hub.host, "instrument_model", None), lenient, cancelled))
    try:
        # real time: set by a real thread (the build worker)
        return await asyncio.wait_for(building, BUILD_BOUND_S)
    except TimeoutError as exc:
        cancelled.set()
        raise _refuse(REFUSED, "build_timeout",
                      f"{name} was not read and built within {BUILD_BOUND_S:g} s, so "
                      f"it was not opened; the design is unchanged") from exc
    except FileNotFoundError as exc:
        raise _refuse(INVALID_PARAMS, "not_found",
                      f"there is no saved design named {name}", param="name") from exc
    except design_file.DesignFileError as exc:
        raise _refuse(INVALID_PARAMS, exc.reason, f"{name}: {exc}",
                      object=exc.object_name, field=exc.field) from exc
    except OSError as exc:
        raise _refuse(INTERNAL_ERROR, "read_failed",
                      f"{name} could not be read: {exc}") from exc
    except (ArithmeticError, RecursionError, ValueError) as exc:
        raise _refuse(INVALID_PARAMS, "not_a_design_file",
                      f"{name} is not a readable design file "
                      f"({type(exc).__name__})") from exc


def _frames_differ(saved, now) -> bool:
    if not isinstance(saved, dict):
        return False
    for part in ("scan", "litho"):
        a, b = saved.get(part), (now or {}).get(part)
        if a is None:
            continue
        if not isinstance(a, dict) or not isinstance(b, dict):
            return True
        for key, value in a.items():
            try:
                if abs(float(value) - float(b.get(key))) > _FRAME_TOL:
                    return True
            except (TypeError, ValueError, OverflowError):
                return True
    return False


def _missing_ports(hub, objects) -> list[dict]:
    known = {p.get("id") for p in (getattr(hub, "ports", None) or [])}
    return [{"name": o.name, "port_id": o.end_binding.get("port_id")}
            for o in objects
            if isinstance(o.end_binding, dict)
            and o.end_binding.get("port_id") not in known]


async def _keep_previous(hub) -> tuple[bool, dict]:
    """Save the design about to be replaced as `_previous`, and return once the
    design in memory is still the one that was saved."""
    for _ in range(_INSTALL_ATTEMPTS):
        snapshot = design_store.Snapshot(hub)
        if snapshot.objects:
            try:
                await _write_snapshot(snapshot, design_store.PREVIOUS)
            except (OSError, ValueError) as exc:
                raise _save_refusal(
                    snapshot, "previous_save_failed",
                    f"the design was not replaced, because the current one could "
                    f"not be kept as {design_store.PREVIOUS}", exc) from exc
        _refuse_during_write(hub)
        if snapshot.matches(hub):
            return bool(snapshot.objects), snapshot.frame
    raise _refuse(BUSY, "design_changed",
                  "the design was being edited while the file was opened; nothing "
                  "was changed, try again")


async def open_design(hub, name, mode="add") -> dict:
    """Open a saved design into the current one (`add`) or in its place (`replace`).

    Either the whole file is opened or the design is left as it was."""
    from ...litho.compiler import Pattern

    if mode is None:
        mode = "add"
    if mode not in MODES:
        raise _refuse(INVALID_PARAMS, "bad_mode",
                      "mode must be \"add\" or \"replace\"", param="mode")
    path = None
    if isinstance(name, str):
        path = await asyncio.to_thread(design_store.open_path, hub.store.path.parent,
                                       name)
    if path is None:
        raise _no_name()
    name = design_store.opened_name(path)
    _refuse_during_write(hub)
    doc, incoming, left_out = await _read_and_build(hub, path, name)
    if not incoming and left_out:
        raise _refuse(INVALID_PARAMS, "no_buildable_record",
                      f"{name} holds no object that can be rebuilt: "
                      f"{left_out[0]['reason']}", left_out=left_out)
    if not incoming:
        raise _refuse(INVALID_PARAMS, "empty_file", f"{name} holds no objects")
    _refuse_during_write(hub)
    saved_names = [o.name for o in incoming]
    pattern_name = str(doc.get("pattern_name") or name)

    # In each branch there is no await between the last check and the install.
    previous_saved = False
    if mode == "replace":
        previous_saved, frame_now = await _keep_previous(hub)
        hub.pattern = Pattern(name=pattern_name, objects=incoming)
        hub.host.written_objects = set()
    else:
        if not getattr(hub, "pattern", None):
            hub.pattern = Pattern(name=pattern_name, objects=[])
        names = design_file.unique_names(saved_names,
                                         [o.name for o in hub.pattern.objects])
        for obj, new_name in zip(incoming, names):
            obj.name = new_name
        hub.pattern.objects.extend(incoming)
        written = getattr(hub.host, "written_objects", None)
        if written:
            written.difference_update({id(o) for o in incoming})
        frame_now = design_store.current_frame(hub)

    frame_saved = doc.get("frame")
    objects = [{"saved_name": saved, "name": obj.name, "renamed": obj.name != saved}
               for saved, obj in zip(saved_names, incoming)]
    renamed = sum(1 for entry in objects if entry["renamed"])
    await hub.log(
        f"Design {name} opened ({mode}): {len(incoming)} objects"
        + (f", {renamed} renamed" if renamed else "")
        + (f", {len(left_out)} left out" if left_out else "")
        + (f"; the earlier design is kept as {design_store.PREVIOUS}"
           if previous_saved else ""))
    await hub.notify_design_changed()
    return {"name": name, "mode": mode, "objects": objects,
            "object_count": len(incoming),
            "total_objects": len(hub.pattern.objects),
            "previous_saved": previous_saved, "left_out": left_out,
            "frame_saved": frame_saved, "frame_now": frame_now,
            "frame_differs": _frames_differ(frame_saved, frame_now),
            "missing_ports": _missing_ports(hub, incoming)}
