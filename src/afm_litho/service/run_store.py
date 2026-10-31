"""Where the runs of writes and scans are kept, and how a write or scan emits its run.

One folder of plain files, beside the journal, named by execution id:
  <id>.run.json      the run (canonical JSON, see afm_litho.run)
  <id>.design.json   the design file's settings the write was compiled from
  <id>.compile.json  the compile settings (frame, LVDT calibration, clock)
The commanded X/Y files of a write sit next to its capture (writes/xy/), not here.

Emitting a run must never change what the instrument does. The functions a write or scan calls
as it starts (`emit_write_run`, `store_scan_run`) return "no run: <reason>" instead of raising.
The ones called after the pass is over (`finish_write_run`, `link_captures`) may raise; their
callers catch and log. Nothing here runs on an abort, stop or park path.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import io
import json
import logging
import os
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from ..run import Execution, Run, next_execution_id, run_from_json, run_to_json
from ..run.ids import id_prefix
from ..run.model import Design, LithoObject, Placement, StepFile

log = logging.getLogger(__name__)

# The one setting: where runs are stored. Default: a folder beside the journal.
RUN_STORE_ENV = "AFM_LITHO_RUN_STORE"
RUN_STORE_FOLDER = "runs"
COMMANDED_XY_KIND = "xy"  # writes/xy/<stem>.step<NN>.npz


def store_dir(runtime) -> Path:
    """The run store folder for a runtime directory."""
    override = os.environ.get(RUN_STORE_ENV)
    return Path(override) if override else Path(runtime) / RUN_STORE_FOLDER


def link_text(path: Path, runtime) -> str:
    """A saved file's path as the run links it: relative to the runtime folder when it is
    inside it, else absolute (the same rule as `captures.save_capture`)."""
    try:
        return path.relative_to(Path(runtime)).as_posix()
    except ValueError:
        return path.as_posix()


def no_run(reason: str) -> str:
    return f"no run: {reason}"


def _atomic_write(path: Path, data: bytes) -> None:
    """Temp file then rename, so a reader never sees half a file; the temp name starts with a
    dot and does not end in .json or .npz, so no listing picks it up."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class RunStore:
    """The folder of stored runs."""

    def __init__(self, folder):
        self.folder = Path(folder)

    def path_for(self, execution_id: str) -> Path:
        return self.folder / f"{execution_id}.run.json"

    def taken_ids(self, kind: str, started: datetime) -> set[str]:
        """Stored execution ids of `kind` started in the same minute as `started`, so ids stay
        unique across restarts."""
        if not self.folder.is_dir():
            return set()
        prefix = id_prefix(kind, started)
        return {p.name.removesuffix(".run.json") for p in self.folder.glob(f"{prefix}*.run.json")}

    def save(self, run: Run) -> Path:
        """Write the run as <execution id>.run.json (replacing an earlier save of the same
        run, which is how links added later reach the file)."""
        path = self.path_for(run.execution.execution_id)
        _atomic_write(path, run_to_json(run).encode("ascii"))
        return path

    def load(self, execution_id: str) -> Run:
        return run_from_json(self.path_for(execution_id).read_text(encoding="ascii"))

    def save_json(self, name: str, doc: dict) -> Path:
        """A companion file (design or compile settings), canonical so the same settings
        always give the same bytes."""
        path = self.folder / name
        _atomic_write(path, _canonical(doc).encode("ascii"))
        return path


def _canonical(doc) -> str:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def fingerprint(doc) -> str:
    return hashlib.sha256(_canonical(doc).encode("ascii")).hexdigest()


def run_reference(run: Run, path: Path, runtime) -> dict:
    """What a sidecar or result reply says about the stored run."""
    return {"execution_id": run.execution.execution_id, "path": link_text(path, runtime)}


def platform_name(arc) -> str:
    return "cypher" if getattr(arc, "is_cypher", False) else "mfp3d"


def _ascii(text: str) -> str:
    """The run's text fields are printable ASCII with no double quote."""
    return "".join(c if " " <= c <= "~" and c != '"' else "?" for c in str(text))


@functools.cache
def compiler_version() -> str:
    """Which code compiled the design: package version and git commit. Cached because asking
    git takes a moment (the service asks once at start so no write waits for it)."""
    from .. import __version__
    from .journal import git_commit
    return f"afm-litho {__version__} {git_commit()}"


def compile_settings(frame, x_sens, y_sens, x_offset, y_offset) -> dict:
    """The arguments `compile_pattern` was called with, with its other defaults filled in."""
    from ..litho.compiler import compile_pattern
    defaults = {n: p.default for n, p in inspect.signature(compile_pattern).parameters.items()
                if n not in ("pattern", "x_sens", "y_sens", "x_offset", "y_offset", "frame")
                and p.default is not inspect.Parameter.empty}
    return {"frame": frame.to_dict(), "x_sens": x_sens, "y_sens": y_sens,
            "x_offset": x_offset, "y_offset": y_offset, **defaults}


@dataclass
class StoredWriteRun:
    """A write's stored run and what is still needed when the write finishes: each litho pass's
    step number, compiled stroke name and fired X/Y points (float32), saved beside the capture
    once its name is known."""

    run: Run
    path: Path
    ref: dict
    xy: list[tuple[int, str, np.ndarray, np.ndarray]] = field(default_factory=list)


def emit_write_run(store: RunStore, runtime, *, design_doc: dict, compile_doc: dict,
                   design_name: str, objects, segments, setpoint, amp_gain: float,
                   platform: str, placement: Placement, bias_window_v,
                   started: datetime | None = None) -> StoredWriteRun | str:
    """Build the write's run from the compiled strokes, store it with its design and
    compile-settings files, and return it; or "no run: <reason>" if anything fails."""
    from ..run.build_write import build_write_run, fired_points
    try:
        started = started or datetime.now().astimezone()
        snapshot = fingerprint({k: v for k, v in design_doc.items() if k != "saved_at"})
        design = Design(_ascii(design_name), 0, len(objects), snapshot,
                        fingerprint(compile_doc), compiler_version())
        run = build_write_run(
            segments=segments, objects=objects, setpoint=setpoint, amp_gain=amp_gain,
            platform=platform, placement=placement, design=design,
            bias_window_v=bias_window_v, started=started,
            taken_ids=store.taken_ids("write", started))
        ident = run.execution.execution_id
        design_path = store.save_json(f"{ident}.design.json", design_doc)
        compile_path = store.save_json(f"{ident}.compile.json", compile_doc)
        run = replace(run, execution=replace(
            run.execution, design_snapshot=link_text(design_path, runtime),
            compile_settings=link_text(compile_path, runtime)))
        path = store.save(run)
        xy = []
        for n, entry in enumerate(run.recipe.body):
            if isinstance(entry, LithoObject):
                meta = _segment_of(segments, objects, entry)
                x, y, _v = fired_points(meta)
                xy.append((n, meta["name"], x.astype("<f4"), y.astype("<f4")))
        return StoredWriteRun(run, path, run_reference(run, path, runtime), xy)
    except Exception as exc:  # noqa: BLE001 - a run that cannot be stored must not stop a write
        log.warning("write run not stored: %s", exc)
        return no_run(f"{type(exc).__name__}: {exc}")


def _segment_of(segments, objects, litho: LithoObject) -> dict:
    """The compiled stroke a litho pass was built from (one stroke per object)."""
    source = objects[litho.index].name
    (meta,) = [m for m in segments if m["object"] == source]
    return meta


def finish_write_run(store: RunStore, runtime, stem: str, stored: StoredWriteRun, *,
                     written: set[str], partial: tuple[str, float | None] | None,
                     captures: tuple[str, ...]) -> Run:
    """After the write: save the commanded X/Y of every pass that was fired, link the capture
    files, and save the run again.

    `written` are the compiled stroke names that finished; `partial` is the stroke an abort cut
    short and the fraction of it estimated to have played (None if unknown). Files are
    writes/xy/<stem>.step<NN>.npz (NN the pass's step in the run) with x_um and y_um in float32;
    a cut pass is saved whole as <stem>.step<NN>.partial.npz with `fired_fraction`.
    """
    from . import stems
    folder = stems.dir_for(runtime, "litho", COMMANDED_XY_KIND)
    files = []
    for n, name, x, y in stored.xy:
        arrays = {"x_um": x, "y_um": y}
        suffix = ""
        if name not in written:
            if partial is None or partial[0] != name:
                continue
            suffix = ".partial"
            arrays["fired_fraction"] = np.float32(-1.0 if partial[1] is None else partial[1])
        path = folder / f"{stem}.step{n:02d}{suffix}.npz"
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        _atomic_write(path, buf.getvalue())
        files.append(StepFile(n, link_text(path, runtime)))
    run = replace(stored.run, execution=replace(
        stored.run.execution, commanded_xy=tuple(files), captures=captures))
    store.save(run)
    return run


def capture_links(runtime, *paths) -> tuple[str, ...]:
    """The links of those capture files (paths or runtime-relative text) that exist."""
    out = []
    for p in paths:
        if not p:
            continue
        path = Path(p) if Path(p).is_absolute() else Path(runtime) / p
        if path.is_file():
            out.append(link_text(path, runtime))
    return tuple(out)


def link_captures(store: RunStore, execution_id: str, captures: tuple[str, ...]) -> None:
    """Add capture links to a stored run (a scan, whose files exist only once it ends)."""
    run = store.load(execution_id)
    store.save(replace(run, execution=replace(run.execution, captures=captures)))


def store_scan_run(store: RunStore, runtime, config, *, platform: str, imaging_mode,
                   engaged: bool, mode_gate_closed: bool, drift_offset: bool,
                   started: datetime | None = None,
                   ends_with_withdraw: bool = True) -> tuple[Run, dict] | str:
    """Build the scan's run and store it; return it with its reference, or "no run: <reason>"."""
    from ..run.build_scan import build_scan_run
    try:
        started = started or datetime.now().astimezone()
        ident = next_execution_id("scan", started, store.taken_ids("scan", started))
        run = build_scan_run(
            config, platform=platform, execution=Execution(ident, started.isoformat()),
            imaging_mode=imaging_mode, engaged=engaged, mode_gate_closed=mode_gate_closed,
            drift_offset=drift_offset, ends_with_withdraw=ends_with_withdraw)
        return run, run_reference(run, store.save(run), runtime)
    except Exception as exc:  # noqa: BLE001 - a run that cannot be stored must not stop a scan
        log.warning("scan run not stored: %s", exc)
        return no_run(f"{type(exc).__name__}: {exc}")
