"""Per-sample electrode-PORT store for the Connect tab (Phase 1).

The operator MARKS the sample's existing electrodes on the scan as connection
PORTS -- a point + a direction the connecting wire comes in from + a pad width +
a label. A later phase couples design wires to these ports by id (so a wire
follows the electrode through drift), which is why the id must be STABLE and
NEVER reused: deleting a port must not free its id for the next one.

Mirrors `captures.py`/`scan_cache.py`: one JSON file per sample under the runtime
store, an atomic write (temp file + os.replace) so an interrupted save never
leaves a half-file, and a defensive load that returns `[]` for a
missing/corrupt/empty file rather than raising (same fix scan_cache carries).

On-disk shape:  {"ports": [port, ...], "seq": N}
  `seq` is a monotonically increasing counter persisted alongside the list, so
  `next_port_id` hands out `e{seq}` and bumps it; it is NEVER decremented on
  delete, and thus an id is never reused even across deletes.

A port dict is exactly:
  {"id": str, "label": str, "position_um": [x, y],
   "direction_deg": float, "width_um": float}
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_DIR = "electrodes"


def _safe_sample(name: str) -> str:
    # Same sanitising the rest of the service uses (deps._safe_sample /
    # raw_dir_for): a sample name is not a filename, so map anything that is not
    # alnum/-._ to underscore. Kept local so this module imports nothing from the
    # service package (it sits below the router that uses it).
    return "".join(c if (c.isalnum() or c in "-._") else "_" for c in str(name))


def _electrodes_dir(runtime_dir) -> Path:
    d = Path(runtime_dir) / _DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _store_path(runtime_dir, sample) -> Path:
    return _electrodes_dir(runtime_dir) / f"{_safe_sample(sample)}.json"


def coerce_port(raw: dict) -> dict:
    """Validate/coerce one port dict into the canonical shape.

    Raises ValueError on anything that cannot be a port (missing/!=2 position,
    non-numeric fields), so a bad payload becomes a `{"error": ...}` at the
    router rather than a stored garbage record. Missing optional fields default
    (direction 0.0, width 1.0, label = the id or "").
    """
    if not isinstance(raw, dict):
        raise ValueError("port must be an object")
    pos = raw.get("position_um")
    if not isinstance(pos, (list, tuple)) or len(pos) != 2:
        raise ValueError("position_um must be [x, y] microns")
    try:
        x, y = float(pos[0]), float(pos[1])
    except (TypeError, ValueError):
        raise ValueError("position_um must be two numbers")
    try:
        direction = float(raw.get("direction_deg", 0.0) or 0.0)
    except (TypeError, ValueError):
        raise ValueError("direction_deg must be a number")
    try:
        width = float(raw.get("width_um", 1.0))
    except (TypeError, ValueError):
        raise ValueError("width_um must be a number")
    pid = str(raw.get("id", ""))
    label = raw.get("label")
    label = pid if label is None else str(label)
    return {
        "id": pid,
        "label": label,
        "position_um": [x, y],
        "direction_deg": direction,
        "width_um": width,
    }


def load_store(runtime_dir, sample) -> tuple[list[dict], int]:
    """Return (ports, seq) for a sample, robust to a missing/corrupt/empty file.

    A missing file, an empty file, invalid JSON, or a wrong-typed body all read
    as ([], 0): the store never raises so the Connect endpoints never 500 on a
    file that was truncated by an interrupted save. `seq` is floored so it is at
    least as large as any `e<N>` id already on disk -- a hand-edited or partially
    written file can never hand back an id that is already in use.
    """
    p = _store_path(runtime_dir, sample)
    if not p.is_file():
        return [], 0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                  # noqa: BLE001 -- corrupt
        return [], 0
    if not isinstance(data, dict):
        return [], 0
    raw_ports = data.get("ports")
    ports: list[dict] = []
    if isinstance(raw_ports, list):
        for r in raw_ports:
            try:
                ports.append(coerce_port(r))
            except (ValueError, TypeError):
                continue                               # skip one bad record
    try:
        seq = int(data.get("seq", 0))
    except (TypeError, ValueError):
        seq = 0
    # Never let seq fall below an id already present, or a fresh id would collide.
    for port in ports:
        pid = port["id"]
        if pid.startswith("e") and pid[1:].isdigit():
            seq = max(seq, int(pid[1:]) + 1)
    return ports, max(seq, 0)


def load_ports(runtime_dir, sample) -> list[dict]:
    """The sample's ports, or `[]` if none/corrupt. Never raises."""
    return load_store(runtime_dir, sample)[0]


def save_ports(runtime_dir, sample, ports, seq) -> None:
    """Atomically write {"ports": ports, "seq": seq} for a sample.

    Temp file + os.replace (like scan_cache.save_scan_frame) so a save the
    service is killed midway through never leaves a partial JSON for the loader
    to choke on. Ports are coerced on the way out too, so only canonical records
    reach the disk.
    """
    clean = [coerce_port(p) for p in (ports or [])]
    path = _store_path(runtime_dir, sample)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"ports": clean, "seq": int(seq)}, fh, indent=1)
    os.replace(tmp, path)


def next_port_id(seq: int) -> tuple[str, int]:
    """Hand out a fresh id `e{seq}` and the bumped counter (seq + 1).

    The counter is persisted and never decremented on delete, so this id is
    never handed out twice -- a wire that references it keeps referencing the
    same port for the life of the sample.
    """
    seq = int(seq)
    return f"e{seq}", seq + 1
