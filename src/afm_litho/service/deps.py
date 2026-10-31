"""Shared paths, constants and small helpers for the service layer.

Split out of app.py so the domain routers can import these without dragging in
the whole app. This module imports nothing from the service package -- it is the
base of the service import graph (app -> hub -> deps).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..twin.physics import Instrument

# Where the read-only assets are, and where the writable state goes. These
# are the same directory in a source checkout and deliberately different in
# an installed build.
#
# REPO points at bundled, read-only content: sample .ibw files, demo designs.
# In the packaged app that is inside the PyInstaller bundle, which the
# installer REPLACES on every update, so nothing that has to survive an
# update may live there.
#
# RUNTIME is per-user writable state: session archives and raw passes. In an
# installed build it goes to
# %LOCALAPPDATA%\afm-litho, which is where per-user data belongs on Windows,
# survives updates, and is not shared between two people on one bench.
def _repo_root() -> Path:
    if getattr(sys, "frozen", False):
        # PyInstaller one-folder: _MEIPASS is the _internal directory that
        # the spec's datas were unpacked into.
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[3]


def _runtime_root() -> Path:
    override = os.environ.get("AFM_LITHO_RUNTIME")
    if override:
        return Path(override)
    if getattr(sys, "frozen", False):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
        return Path(base) / "afm-litho"
    return Path(__file__).resolve().parents[3] / "runtime"


REPO = _repo_root()
RUNTIME = _runtime_root()
JOURNAL_DIR = RUNTIME / "journal"
WEB = Path(__file__).resolve().parent.parent / "web"
SAMPLES = REPO / "data" / "samples"
DEFAULT_SAMPLE = "SA40656B0000.ibw"


def _safe_sample(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-._") else "_" for c in str(name))


def raw_dir_for(name: str) -> Path:
    """Where this sample's raw passes live.

    User request: "when i load a new sample all the old stuff should be erased by
    default. previous topographic scans, litho patterns. i will upload those
    and configure them."

    They were all in one directory. Loading a new sample built a fresh
    SurfaceModel over the SAME `RUNTIME/raw`, so the new sample's Surface
    filled in with the previous sample's topography -- measurements of a
    different piece of oxide, drawn as though they were of this one. On an
    instrument that is not untidiness, it is a wrong measurement on the
    screen, and the panel had no way to say so.

    Raw passes belong to the sample they were taken on, so they are filed
    under it. Nothing is deleted: switching back brings a sample's own
    history with it.
    """
    return RUNTIME / "raw" / _safe_sample(name)


def _instrument_dict(i: Instrument) -> dict:
    return {
        "model": i.model, "id": i.microscope_id,
        "range_um": i.xy_range_um,
        "f0_khz": i.thermal_f0 / 1e3, "Q": i.thermal_q,
        "k_nm": i.spring_constant, "invols_nm_v": i.invols * 1e9,
        "x_sensor": i.x_sensor, "dual_lockin": i.dual_lockin,
        "igain": i.integral_gain,
    }
