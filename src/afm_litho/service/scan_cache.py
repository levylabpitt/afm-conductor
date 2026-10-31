"""A tiny on-disk cache of completed scan frames for the litho cockpit.

Design: every finished scan drops its height frame into `scans/npz/` on
disk; the cockpit asks for the NEWEST one and paints it under the design overlay.
So this module is the disk side of that -- mirrors `captures.py`: one compressed
`.height.npz` per frame, named by a caller-supplied lexicographically-sortable
stamp (the run's date-time stem, service/stems.py), plus a loader that returns
the newest as the wire dict the cockpit expects (height flat, row-major
slow_idx*pixels+i, in NANOMETRES).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import stems

# The frame cache is the "height" npz alongside a scan's other formats under
# `scans/npz/` (service/stems.py) -- suffixed `.height.npz` (not bare `.npz`)
# so it coexists in the same directory as the RAW multi-channel capture
# (`captures.save_capture`, `<stem>.npz`) without colliding, and so the
# newest-by-name readers below can glob for it specifically.
_HEIGHT_SUFFIX = ".height.npz"


def save_scan_frame(runtime_dir, frame: dict) -> str:
    """Write one scan frame to scans/npz/<stamp>.height.npz and return its
    reference, relative to `runtime_dir`.

    The caller supplies `frame["stamp"]` (wall-clock APIs are unavailable here)
    -- the run's stem (service/stems.make_stem), used verbatim as the
    filename; stems sort lexicographically by age so the newest file wins.
    `frame["height"]` is a 2D array (rows, pixels) in nm; the geometry scalars
    are stored as 0-d arrays alongside it.
    """
    import os

    stamp = str(frame["stamp"])
    height = np.asarray(frame["height"], dtype=np.float64)
    d = stems.dir_for(runtime_dir, "scan", "npz")
    path = d / f"{stamp}{_HEIGHT_SUFFIX}"
    # Write to a temp file then atomically replace, so a save interrupted midway
    # (service killed) never leaves a 0-byte/partial .npz for the loader to choke
    # on. Pass a file handle (not a str) so numpy doesn't re-append ".npz".
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            height=height,
            size_um=np.asarray(float(frame["size_um"])),
            x_offset_um=np.asarray(float(frame["x_offset_um"])),
            y_offset_um=np.asarray(float(frame["y_offset_um"])),
            angle_deg=np.asarray(float(frame["angle_deg"])),
            pixels=np.asarray(int(frame["pixels"])),
            rows=np.asarray(int(frame["rows"])),
        )
    os.replace(tmp, path)
    try:
        return str(path.relative_to(Path(runtime_dir))).replace("\\", "/")
    except ValueError:
        return str(path)


def load_latest_scan_frame(runtime_dir) -> dict | None:
    """Return the newest cached frame as the cockpit wire dict, or None.

    Newest = the last filename in sort order (stems are lexicographically
    sortable, service/stems.py); st_mtime breaks any tie / covers oddly-named
    files. Returns None when `scans/npz/` is missing or holds no
    `*.height.npz`. Reads the path directly (rather than through
    `stems.dir_for`) so a check for "nothing cached yet" never creates the
    directory as a side effect.
    """
    d = Path(runtime_dir) / stems.kind_dir("scan") / "npz"
    if not d.is_dir():
        return None
    files = [p for p in d.glob(f"*{_HEIGHT_SUFFIX}") if p.is_file()]
    if not files:
        return None
    # Newest first; skip any corrupt/empty/truncated file and fall back to the
    # next. An interrupted save (service killed mid-write) can leave a 0-byte .npz,
    # which np.load raises EOFError on -- that must not 500 /api/scan/last.
    for newest in sorted(files, key=lambda p: (p.name, p.stat().st_mtime),
                         reverse=True):
        try:
            with np.load(newest) as z:
                height = np.asarray(z["height"], dtype=np.float64)
                return {
                    "pixels": int(z["pixels"]),
                    "rows": int(z["rows"]),
                    "size_um": float(z["size_um"]),
                    "x_offset_um": float(z["x_offset_um"]),
                    "y_offset_um": float(z["y_offset_um"]),
                    "angle_deg": float(z["angle_deg"]),
                    # flat, row-major (index = slow_idx*pixels + i), in nanometres
                    "height": height.reshape(-1).tolist(),
                }
        except Exception:  # noqa: BLE001, S112 - corrupt file, try older
            continue
    return None
