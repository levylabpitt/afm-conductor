"""Atomic small-JSON writer -- used for the scan sidecar (docs/
IMAGE_SAVING_NOTES.md, `scans/json/<stem>.json`): one JSON file per scan run,
written next to its PNGs, carrying everything a design-swap or a notebook
needs to place the height PNG at the right physical size and know which
channel PNGs exist, without re-deriving it from the npz. Same temp +
`os.replace` discipline as every other writer in this package (`captures.py`,
`scan_cache.py`, `io/png.py`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def write_json_atomic(path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    return path
