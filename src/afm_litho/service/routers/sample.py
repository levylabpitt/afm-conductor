"""The sample domain: which scan the twin is looking at, and its contacts.

An .ibw scan is more than a picture: its note carries the microscope model,
scanner range, LVDT and piezo calibrations and lever parameters, so loading a
scan reconfigures the WHOLE twin -- an MFP-3D scan gives you the MFP-3D twin,
a Cypher scan the Cypher twin. Contacts belong to the sample, not the session:
they are written under `RUNTIME/contacts` per sample and read back whenever
that sample is loaded (which is why `restore_contacts` is also called by the
start-up primer still in app.py, for the sample the Hub opens with).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import numpy as np

from ...instrument.twin_capture import load_capture
from ...io import read_ibw
from ...twin.arc import SoftwareARC
from ...twin.physics import Instrument, Sample, World
from ..deps import RUNTIME, SAMPLES, _instrument_dict, _safe_sample, raw_dir_for
from ..hub import hub
from ..passes import TwinInstrument
from ..surface import SurfaceModel
from ..twin_plugin import forget as forget_twin_plugin


def _sample_dirs() -> list[Path]:
    """Where .ibw scans are read from: the bundled samples dir, plus an optional
    AFM_LITHO_IBW_DIR when it actually exists. One rule in one place, so listing
    and loading can never disagree about where samples live (they used to: the
    loader searched a literal "/nonexistent" sentinel path instead of this)."""
    dirs = [SAMPLES]
    extra = os.environ.get("AFM_LITHO_IBW_DIR")
    if extra and Path(extra).is_dir():
        dirs.append(Path(extra))
    return dirs


def _from_capture(path: Path) -> tuple[Instrument, Sample]:
    """A hardware scan capture (.npz) as a Cypher sample carrying its real retrace.
    The twin scans Height only, so the capture's Amplitude is not used."""
    seed = load_capture(path)
    mean = float(np.nanmean(seed["height"]))

    def to_map(nm):                                   # the map is indexed [x, y], in metres
        return (nm - mean).T * 1e-9
    instrument = Instrument.from_note({"MicroscopeModel": "Cypher S",
                                       "MaxScanSize": seed["size_um"] * 1e-6})
    sample = Sample(height_map=to_map(seed["height"]), extent=seed["size_um"] * 1e-6,
                    name=path.stem, retrace_map=to_map(seed["retrace"]))
    return instrument, sample


async def samples():
    files = []
    for d in _sample_dirs():
        files += sorted(p.name for p in d.glob("*.ibw")) + sorted(p.name for p in d.glob("*.npz"))
    return {"current": hub.sample_name, "files": files}


async def sample_load(body: dict):
    """Load a scan: the file's own metadata reconfigures the whole twin.

    An .ibw carries MicroscopeModel, MaxScanSize, LVDT and piezo
    calibrations, lever InvOLS / k / f0 / Q, gains and channel names, so
    loading an MFP-3D scan gives you the MFP-3D twin and loading a Cypher
    scan gives you the Cypher twin.
    """
    name = str(body.get("file", ""))
    for d in _sample_dirs():
        p = d / name
        if not p.exists():
            continue
        if p.suffix.lower() == ".npz":
            instrument, sample = await asyncio.to_thread(_from_capture, p)
        else:
            wave = await asyncio.to_thread(read_ibw, p)
            instrument = Instrument.from_note(wave.note_dict)
            sample = await asyncio.to_thread(Sample.from_ibw, p)
        hub.host.instrument = instrument
        hub.host.world = World(sample=sample, instrument=instrument)
        hub.host.arc = SoftwareARC(hub.host.world)
        hub.host.inst = TwinInstrument(hub.host.arc)
        forget_twin_plugin(hub)        # the next write or scan converts this sample
        # A new sample starts empty. Its Surface reads only ITS raw
        # passes, its design is cleared, and nothing already written is
        # remembered as written.
        hub.surface = SurfaceModel(raw_dir=raw_dir_for(sample.name),
                                   range_um=instrument.xy_range_um)
        hub.host.written_objects = set()
        hub.host.written_accum = []
        hub.sample_name = sample.name
        hub.pattern = None
        from ..design_store import request_autosave
        request_autosave(hub)
        # Electrode ports belong to the sample: loading a scan back brings its
        # ports with it, and a new sample starts with its own (empty) list.
        from .. import electrodes
        hub.ports, hub.ports_seq = electrodes.load_store(RUNTIME, sample.name)
        # Marks belong to the sample, not to the session: loading a scan back
        # brings its contacts with it.
        n_marked = restore_contacts(sample, sample.name)
        if n_marked:
            await hub.log(f"Contacts: {n_marked} restored from your marks")
        elif n_marked == 0:
            await hub.log("Contacts: none, because you erased them. "
                          "Mark contacts to put some back.")
        await hub.log(f"Instrument: {instrument.describe()}")
        await hub.log(
            f"Sample loaded: {sample.name} "
            f"({sample.extent * 1e6:.1f} um scan tiled over the "
            f"{instrument.xy_range_um:g} um scanner, "
            f"{len(sample.contact_centroids_um)} contacts)")
        await hub.broadcast({"type": "sample_changed", "name": sample.name,
                             "instrument": _instrument_dict(instrument)})
        return {"ok": True, "name": sample.name,
                "instrument": _instrument_dict(instrument)}
    return {"error": f"not found: {name}"}


async def sample_upload(data: bytes, name: str = "scan.ibw"):
    """Take an .ibw straight from the browser and load it.

    Dropping a scan on the window is the whole procedure: no copying files
    into a directory the panel happens to watch, no restart. The file lands
    in data/samples so it is there next time too.

    The body is the raw file rather than a multipart form, which keeps
    python-multipart out of the dependency list: one less thing to install
    on a machine that just wants to run the twin.
    """
    name = Path(name).name
    if not name.lower().endswith(".ibw"):
        return {"error": "expected an Igor binary wave (.ibw)"}
    if not data:
        return {"error": "empty file"}
    SAMPLES.mkdir(parents=True, exist_ok=True)
    dest = SAMPLES / name
    if dest.exists() and dest.read_bytes() != data:
        stem = dest.stem
        n = 2
        while (SAMPLES / f"{stem}-{n}.ibw").exists():
            n += 1
        dest = SAMPLES / f"{stem}-{n}.ibw"
    await asyncio.to_thread(dest.write_bytes, data)
    await hub.log(f"Received {dest.name} ({len(data) / 1e6:.1f} MB)")
    return await sample_load({"file": dest.name})


CONTACTS_DIR = RUNTIME / "contacts"


def _contacts_path(name: str) -> Path:
    return CONTACTS_DIR / f"{_safe_sample(name)}.json"


def restore_contacts(sample, name: str) -> int | None:
    """Put back the contacts marked for this sample. None if none were saved.

    The tri-state matters and returning 0 for both cases is why it is here.
    An empty file means "I erased them all", which the detector must not
    undo; no file means "nobody has said", which is when the detector's
    guess is the best answer available. Reporting both as 0 made an erasure
    indistinguishable from silence.
    """
    p = _contacts_path(name)
    if not p.is_file():
        return None
    try:
        pills = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - corrupt contacts file is treated as absent
        return None
    sample.set_contacts(pills)
    return len(pills)


async def contacts_get():
    """The contacts as SHAPES, not points.

    User request: "I need to define the contacts. maybe you can give me a tool that
    puts the contacts in various places, that I can define (for now). later I
    will give the klayout that created these and it will be automatic."

    So the wire format is the pill -- centre, size, angle -- because that is
    what a person marks and what a KLayout cell will hand over. The automatic
    path fills exactly this and nothing downstream learns a second shape.
    """
    s = hub.host.world.sample
    pills = getattr(s, "user_contacts", None)
    return {"sample": hub.sample_name, "contacts": pills or [],
            "defined": pills is not None,
            "detected": len(getattr(s, "contact_centroids_um", []) or [])}


async def contacts_set(body: dict):
    """Replace the contacts. An empty list means there are none, and stays
    empty: the detector's guess does not creep back in."""
    pills = body.get("contacts")
    pills = [] if pills is None else list(pills)
    s = hub.host.world.sample
    s.set_contacts(pills)
    CONTACTS_DIR.mkdir(parents=True, exist_ok=True)
    _contacts_path(hub.sample_name).write_text(
        json.dumps(pills, indent=1), encoding="utf-8")
    await hub.log(f"Contacts: {len(pills)} defined by hand" if pills
                  else "Contacts: erased; none defined")
    return {"ok": True, "n": len(pills), "contacts": pills}
