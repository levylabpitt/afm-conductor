"""The `designs` folder under the runtime directory: named design files, the
autosave that mirrors the design in memory, and the restore at start.

Nothing here raises into the service: `request_autosave`, `flush_autosave` and
`restore_autosave` catch, log and report.

When the autosave file exists at start but cannot be read (the read itself
fails), it is left where it is and the autosave is held: an empty design is not
written over it, and before the first design with objects is written the unread
file is moved to `_autosave.unread.json`. The hold ends when that write has
succeeded. A file that reads but is not a design is set aside as
`_autosave.bad.json`; one that restores only in part is copied there, and if that
copy cannot be made the autosave is held in the same way. Both set-aside files
open by the names `_autosave.bad` and `_autosave.unread`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
from pathlib import Path

from ..litho import design_file

log = logging.getLogger(__name__)

SUFFIX = ".design.json"
AUTOSAVE = "_autosave"
PREVIOUS = "_previous"
RESERVED = (AUTOSAVE, PREVIOUS)
BAD_AUTOSAVE = "_autosave.bad.json"
UNREAD_AUTOSAVE = "_autosave.unread.json"
SET_ASIDE = {"_autosave.bad": BAD_AUTOSAVE, "_autosave.unread": UNREAD_AUTOSAVE}
ENV_SWITCH = "AFM_LITHO_DESIGN_AUTOSAVE"
# Measured: a wire of 2,442,265 points saved with its points and bias is 173 MB.
# A file of spines and definitions is kilobytes; this stops only an absurd one.
MAX_FILE_BYTES = 360_000_000
_KEEP_EARLIER = 3

_RETRY_FIRST_S = 2.0
_RETRY_MAX_S = 60.0
_RETRY_DOUBLINGS = 10

_write_lock = threading.Lock()
_HEAD_BYTES = 4096
_HEAD_SAVED_AT = re.compile(r'"saved_at"\s*:\s*"([^"]*)"')
_HEAD_COUNT = re.compile(r'"object_count"\s*:\s*(\d+)')


def file_name(raw: str) -> str:
    """The name a design is filed under: `raw` without the file suffix, with every
    character a file name cannot safely hold (path separators included) replaced.
    A dot inside the name is kept, so `dev.1` and `dev.2` are two designs."""
    raw = raw.strip()
    if raw.lower().endswith(SUFFIX):
        raw = raw[:-len(SUFFIX)]
    if not raw.strip(". "):
        return ""
    return "".join(c if (c.isalnum() or c in "-._") else "_" for c in raw)


def is_reserved(name: str) -> bool:
    return name.lower() in RESERVED or name.lower() in SET_ASIDE


def designs_dir(runtime_dir) -> Path:
    return Path(runtime_dir) / "designs"


def design_path(runtime_dir, name: str) -> Path:
    return designs_dir(runtime_dir) / f"{name}{SUFFIX}"


def open_path(runtime_dir, raw: str) -> Path | None:
    """The file a name refers to when it is opened: a set-aside autosave, a file
    in the folder under exactly that name (however it was named by hand), or the
    name as it would be filed. None when `raw` is no name at all."""
    folder = designs_dir(runtime_dir)
    typed = raw.strip()
    if typed.lower() in SET_ASIDE:
        return folder / SET_ASIDE[typed.lower()]
    try:
        present = {p.name for p in folder.iterdir()}
    except OSError:
        present = set()
    for stem in (raw, typed):
        exact = stem if stem.lower().endswith(SUFFIX) else stem + SUFFIX
        if exact in present:
            return folder / exact
    name = file_name(raw)
    return design_path(runtime_dir, name) if name else None


def opened_name(path: Path) -> str:
    for name, file in SET_ASIDE.items():
        if path.name == file:
            return name
    return path.name[:-len(SUFFIX)]


def autosave_enabled() -> bool:
    return os.environ.get(ENV_SWITCH, "1").strip() != "0"


def now_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ----------------------------------------------------------------- snapshot --

rebuild_cache = design_file.RebuildCache()


def current_frame(hub) -> dict:
    """The scan frame and the litho frame the design is drawn in, as saved."""
    scan = getattr(getattr(hub.host, "inst", None), "scan", None)
    cfg = getattr(scan, "config", None) if scan is not None else None
    if cfg is not None:
        src = {k: getattr(cfg, k, None)
               for k in ("size_um", "x_offset_um", "y_offset_um", "angle_deg")}
    else:
        last = getattr(hub.host, "last_scan_config", None) or {}
        src = {k: last.get(k)
               for k in ("size_um", "x_offset_um", "y_offset_um", "angle_deg")}
    scan_frame = None
    if src.get("size_um") is not None:
        try:
            scan_frame = {k: float(v or 0.0) for k, v in src.items()}
        except (TypeError, ValueError):
            scan_frame = None
    try:
        from .host.litho_worker import _litho_frame
        litho = _litho_frame(hub).to_dict()
    except Exception:                                      # noqa: BLE001
        litho = None
    return {"scan": scan_frame, "litho": litho}


def _shallow(obj):
    twin = object.__new__(type(obj))
    twin.__dict__.update(obj.__dict__)
    return twin


class Snapshot:
    """The design as it is now, taken on the event loop. Objects are shallow
    copies: a route replaces an object's lists, it does not edit them in place."""

    def __init__(self, hub):
        pat = getattr(hub, "pattern", None)
        self.live = list(pat.objects) if pat else []
        self.objects = [_shallow(o) for o in self.live]
        self.pattern_name = getattr(pat, "name", None) if pat else None
        self.frame = current_frame(hub)
        self.runtime_dir = hub.store.path.parent

    def matches(self, hub) -> bool:
        """True while the design still holds exactly what was snapshotted."""
        pat = getattr(hub, "pattern", None)
        now = list(pat.objects) if pat else []
        if len(now) != len(self.live):
            return False
        try:
            return all(obj is live and vars(obj) == vars(shadow)
                       for obj, live, shadow in zip(now, self.live, self.objects))
        except Exception:                                  # noqa: BLE001
            return False

    def document(self, name: str) -> dict:
        """The document to write. Flattens each Bezier object to decide what to
        store, so this belongs on a worker thread."""
        return design_file.pattern_to_document(
            self.objects, name=name, saved_at=now_stamp(), frame=self.frame,
            pattern_name=self.pattern_name, cache=rebuild_cache)

    def describe(self, exc: Exception) -> tuple[str, dict]:
        """A failed write in words, naming the object when a number that is not
        finite caused it."""
        if isinstance(exc, ValueError):
            where = design_file.find_non_finite(self.objects)
            if where:
                message = (f"object {where[0]!r}: {where[1]} holds a number that "
                           f"is not finite")
                return message, {"object": where[0], "field": where[1]}
        return (str(exc) or type(exc).__name__), {}


# --------------------------------------------------------------------- files --

def write_document(runtime_dir, name: str, doc: dict) -> Path:
    """Write `doc` as `<name>.design.json`, atomically. Raises OSError, or
    ValueError when the design holds a number that is not finite."""
    path = design_path(runtime_dir, name)
    tmp = path.with_name(path.name + ".tmp")
    with _write_lock:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                fh.writelines(design_file.iter_json(doc))
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise
    return path


def parse_document(data: bytes, label: str) -> dict:
    """The document in `data`. Raises `DesignFileError` when it is not JSON or not
    a design file."""
    try:
        doc = json.loads(data.decode("utf-8"))
    except (ValueError, RecursionError) as exc:
        why = str(exc) if isinstance(exc, ValueError) else "it is nested too deeply"
        raise design_file.DesignFileError(
            "not_a_design_file", f"{label} is not readable JSON ({why})") from exc
    design_file.check_document(doc)
    return doc


def read_document(path: Path) -> dict:
    """The parsed document. Raises FileNotFoundError or OSError when the file
    cannot be read, `DesignFileError` when it is too large to be a design or what
    was read is not a design file."""
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise design_file.DesignFileError(
            "file_too_large",
            f"{path.name} is {size / 1e6:,.0f} MB; a design file of more than "
            f"{MAX_FILE_BYTES / 1e6:,.0f} MB is not opened")
    return parse_document(path.read_bytes(), path.name)


def _make_room(target: Path) -> None:
    """Before a set-aside file is replaced, keep the earlier one under a dated
    name, and only the newest few of those."""
    if not target.is_file():
        return
    stem = target.name[:-len(".json")]
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(target.stat().st_mtime))
    dated = target.with_name(f"{stem}.{stamp}.json")
    n = 1
    while dated.exists():
        n += 1
        dated = target.with_name(f"{stem}.{stamp}-{n}.json")
    os.replace(target, dated)
    for old in sorted(target.parent.glob(f"{stem}.*-*.json"))[:-_KEEP_EARLIER]:
        old.unlink(missing_ok=True)


def _entry(path: Path, name: str | None = None) -> dict | None:
    name = name or path.name[:-len(SUFFIX)]
    try:
        with open(path, "rb") as fh:
            head = fh.read(_HEAD_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return None
    saved = _HEAD_SAVED_AT.search(head)
    count = _HEAD_COUNT.search(head)
    return {"name": name, "saved_at": saved.group(1) if saved else None,
            "object_count": int(count.group(1)) if count else None}


def list_designs(runtime_dir) -> list[dict]:
    """`[{name, saved_at, object_count}]`, newest first, from the first bytes of
    each file: `saved_at` and `object_count` are None when the header does not
    give them. The autosave is not listed; `_previous` is. A set-aside autosave
    comes after the designs, marked `recovered`."""
    folder = designs_dir(runtime_dir)
    if not folder.is_dir():
        return []
    out = []
    for path in folder.glob(f"*{SUFFIX}"):
        if path.name[:-len(SUFFIX)].lower() == AUTOSAVE:
            continue
        entry = _entry(path)
        if entry is not None:
            out.append(entry)
    out.sort(key=lambda e: (e["saved_at"] or "", e["name"]), reverse=True)
    for name, file in SET_ASIDE.items():
        entry = _entry(folder / file, name) if (folder / file).is_file() else None
        if entry is not None:
            out.append({**entry, "recovered": True})
    return out


# ------------------------------------------------------------------ autosave --

class Autosaver:
    """Writes the latest requested snapshot on one worker thread. A request that
    arrives while a write is running replaces any request still waiting. A failed
    write is tried again, at growing intervals, until one succeeds."""

    def __init__(self):
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._pending: Snapshot | None = None
        self._thread: threading.Thread | None = None
        self._failures = 0
        self._retry: Snapshot | None = None
        self.hold = False
        self.status = {"ok": True, "error": None, "saved_at": None}

    def request(self, snapshot: Snapshot) -> None:
        with self._lock:
            self._pending = snapshot
            self._idle.clear()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="design-autosave", daemon=True)
                self._thread.start()
        self._wake.set()

    def flush(self, timeout: float = 10.0, *, retry_now: bool = False) -> bool:
        """Wait until nothing is waiting or being written. A write that failed and
        is waiting for its next try does not count, unless `retry_now` asks for
        that try at once."""
        if retry_now:
            with self._lock:
                if self._retry is not None and self._pending is None:
                    self._pending = self._retry
                    self._idle.clear()
                    self._wake.set()
        # real time: set by a real thread (the autosave worker)
        return self._idle.wait(timeout)

    def _retry_delay(self) -> float:
        doublings = min(self._failures - 1, _RETRY_DOUBLINGS)
        return min(_RETRY_FIRST_S * 2 ** doublings, _RETRY_MAX_S)

    def _run(self) -> None:
        while True:
            self._wake.wait(self._retry_delay() if self._retry is not None else None)
            with self._lock:
                snapshot, self._pending = self._pending, None
                self._wake.clear()
            snapshot = snapshot or self._retry
            if snapshot is not None:
                self._retry = None if self._write(snapshot) else snapshot
            with self._lock:
                if self._pending is None:
                    self._idle.set()

    def _write(self, snapshot: Snapshot) -> bool:
        if self.hold and not snapshot.objects:
            return True
        try:
            if self.hold:
                unread = design_path(snapshot.runtime_dir, AUTOSAVE)
                if unread.is_file():
                    _make_room(unread.with_name(UNREAD_AUTOSAVE))
                    os.replace(unread, unread.with_name(UNREAD_AUTOSAVE))
            doc = snapshot.document(AUTOSAVE)
            write_document(snapshot.runtime_dir, AUTOSAVE, doc)
        except Exception as exc:                           # noqa: BLE001
            message = f"{type(exc).__name__}: {snapshot.describe(exc)[0]}"
            if not self._failures:
                log.warning("design autosave failed (%s); the design in memory is "
                            "untouched and the write is tried again", message)
            self._failures += 1
            self.status = {"ok": False, "error": message,
                           "saved_at": self.status.get("saved_at")}
            return False
        if self._failures:
            log.warning("design autosave is working again")
        self._failures = 0
        self.hold = False
        self.status = {"ok": True, "error": None, "saved_at": doc["saved_at"]}
        return True


_autosaver = Autosaver()


def request_autosave(hub) -> None:
    """Ask for the current design to be written to the autosave. Never raises."""
    try:
        if not autosave_enabled():
            return
        _autosaver.request(Snapshot(hub))
    except Exception as exc:                               # noqa: BLE001
        log.warning("design autosave could not be requested: %s", exc)


def flush_autosave(timeout: float = 10.0, *, retry_now: bool = False) -> bool:
    """Wait, at most `timeout` seconds, for a requested autosave to be on disk.
    Never raises."""
    try:
        return _autosaver.flush(timeout, retry_now=retry_now)
    except Exception as exc:                               # noqa: BLE001
        log.warning("design autosave could not be flushed: %s", exc)
        return False


def autosave_status() -> dict:
    return dict(_autosaver.status)


# ------------------------------------------------------------------- restore --

def _remove_stale_temp_files(folder: Path) -> None:
    for tmp in folder.glob(f"*{SUFFIX}.tmp"):
        try:
            tmp.unlink()
        except OSError as exc:
            log.warning("could not remove the unfinished design file %s: %s",
                        tmp.name, exc)


def _set_aside(path: Path, *, copy: bool = False) -> bool:
    target = path.with_name(BAD_AUTOSAVE)
    try:
        _make_room(target)
        if copy:
            shutil.copyfile(path, target)
        else:
            os.replace(path, target)
        return True
    except OSError as exc:
        log.warning("could not set the unreadable design autosave aside: %s", exc)
        return False


def restore_autosave(hub) -> str | None:
    """Bring the autosaved design back into an empty design. Returns the note for
    the operator, or None when there is nothing to say. Never raises, never
    refuses, and never compiles a design."""
    try:
        if not autosave_enabled():
            return None
        note = _restore(hub)
        folder = designs_dir(hub.store.path.parent)
        kept = [name for name, file in SET_ASIDE.items() if (folder / file).is_file()]
        if kept:
            note = ((note + "; " if note else "Design: ")
                    + "an earlier autosave is kept; what can be rebuilt of it opens "
                    + "by the name " + " or ".join(kept))
        return note
    except Exception as exc:                               # noqa: BLE001
        log.warning("design restore failed: %s", exc)
        return None


def _restore(hub) -> str | None:
    runtime_dir = hub.store.path.parent
    folder = designs_dir(runtime_dir)
    if folder.is_dir():
        _remove_stale_temp_files(folder)
    pat = getattr(hub, "pattern", None)
    if pat and pat.objects:
        return None
    path = design_path(runtime_dir, AUTOSAVE)
    if not path.is_file():
        return None
    try:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            log.warning("the design autosave is %.0f MB, more than the %.0f MB "
                        "a named design file may be; restoring it anyway",
                        size / 1e6, MAX_FILE_BYTES / 1e6)
        data = path.read_bytes()
    except OSError as exc:
        _autosaver.hold = True
        log.warning("the design autosave could not be read (%s); it is left in "
                    "place and the design starts empty", exc)
        return (f"Design: the autosave could not be read ({exc}); it is left in "
                f"place and is not written over until there is a design")
    warnings: list[str] = []
    skipped: list = []
    try:
        doc = parse_document(data, path.name)
        pattern = design_file.document_to_pattern(
            doc, warnings=warnings, skipped=skipped)
    except Exception as exc:                           # noqa: BLE001
        if _set_aside(path):
            where = f"set aside as {BAD_AUTOSAVE}"
        else:
            _autosaver.hold = True
            where = "left in place and is not written over until there is a design"
        log.warning("the design autosave is not a readable design (%s); it is "
                    "%s and the design starts empty", exc, where)
        return (f"Design: the autosave is not a readable design and was {where}; "
                f"the design starts empty")
    kept = ""
    if skipped:
        for line in skipped:
            log.warning("restored design: left out, %s", line)
        n_out = len(skipped)
        kept = (f"; {n_out} object{'' if n_out == 1 else 's'} could not be "
                f"rebuilt and {'was' if n_out == 1 else 'were'} left out")
        if _set_aside(path, copy=True):
            kept += f"; the autosave as it was is kept as {BAD_AUTOSAVE}"
        else:
            _autosaver.hold = True
            kept += ("; a copy of the autosave could not be made, so it is left "
                     "as it is and is not written over until there is a design")
    if not pattern.objects:
        return f"Design: nothing restored{kept}" if skipped else None
    for warning in warnings:
        log.warning("restored design: %s; it is restored as saved and an open "
                    "of the same file would refuse it", warning)
    try:
        from .command_guard import GuardRefused, check_design_voltages
        try:
            check_design_voltages(getattr(hub.host, "instrument_model", None),
                                  pattern.objects)
        except GuardRefused as exc:
            log.warning("restored design: %s (it is restored as saved; a write "
                        "is still checked)", exc)
    except Exception as exc:                           # noqa: BLE001
        log.warning("restored design: the voltage check could not run: %s", exc)
    hub.pattern = pattern
    hub.host.written_objects = set()
    n = len(pattern.objects)
    note = (f"Design: {n} object{'' if n == 1 else 's'} restored from the "
            f"autosave of {doc.get('saved_at') or 'an unknown time'}{kept}")
    log.warning(note)
    return note
