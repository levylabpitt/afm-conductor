"""Two twins, one code path: each instrument configures itself from its scans.

Every Asylum scan carries its machine's calibration in the wave note, so
loading an MFP-3D file gives the MFP-3D twin and a Cypher file gives the
Cypher twin, with no hardcoded instrument tables.
"""

from pathlib import Path

import numpy as np
import pytest

from afm_litho.io import read_ibw
from afm_litho.litho.compiler import LithoObject, compile_object
from afm_litho.twin.physics import Instrument, Sample, World

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
MFP = SAMPLES / "SA40656B0000.ibw"
CYPHER = SAMPLES / "C_2d0044.ibw"


@pytest.mark.skipif(not MFP.exists() or not CYPHER.exists(), reason="samples absent")
def test_instruments_self_configure():
    mfp = Instrument.from_note(read_ibw(MFP).note_dict)
    cyp = Instrument.from_note(read_ibw(CYPHER).note_dict)

    assert mfp.model == "MFP3D" and cyp.model == "Cypher"
    # scanner ranges come from MaxScanSize in the files themselves
    assert mfp.xy_range_um == pytest.approx(90.0, abs=0.5)
    assert cyp.xy_range_um == pytest.approx(30.0, abs=0.5)
    # controller channel naming differs between the two machines
    assert mfp.x_sensor == "Input.X"
    assert cyp.x_sensor.startswith("Cypher.")
    assert cyp.dual_lockin and not mfp.dual_lockin
    # per-machine calibration actually differs
    assert mfp.x_lvdt_sens != cyp.x_lvdt_sens
    assert mfp.thermal_f0 != cyp.thermal_f0


@pytest.mark.skipif(not CYPHER.exists(), reason="sample absent")
def test_world_adopts_instrument_range():
    inst = Instrument.from_note(read_ibw(CYPHER).note_dict)
    world = World(sample=Sample.from_ibw(CYPHER), instrument=inst)
    assert world.x_lvdt_sens == inst.x_lvdt_sens
    assert world.lever.f0 == inst.thermal_f0
    # the written-state grid follows the scanner, not a hardcoded 90 um
    assert world.sample.dose_range == pytest.approx(inst.xy_range_um * 1e-6)


@pytest.mark.skipif(not MFP.exists(), reason="sample absent")
def test_topography_is_continuous_across_the_whole_scanner():
    """No fabricated terrain outside the loaded scan.

    A scan smaller than the scanner range is mirror-tiled, which guarantees
    (a) continuity across the edge of the loaded map, and (b) the same height
    statistics everywhere the tip can reach. The old model invented a tilted
    plane out there, which sat hundreds of nm off and wrecked the contrast.
    """
    s = Sample.from_ibw(MFP)
    edge = s.extent / 2.0

    # (a) continuity: mirroring makes h(edge + d) == h(edge - d)
    d = np.linspace(1e-9, 300e-9, 50)
    ys = np.zeros_like(d)
    assert np.allclose(s.height(edge + d, ys), s.height(edge - d, ys), atol=2e-9)

    # (b) same statistics inside the map and two tiles away (2-D patches)
    def patch(cx, cy, span=12e-6, n=64):
        ax = np.linspace(cx - span / 2, cx + span / 2, n)
        ay = np.linspace(cy - span / 2, cy + span / 2, n)
        gx, gy = np.meshgrid(ax, ay, indexing="ij")
        return s.height(gx.ravel(), gy.ravel())

    inner, outer = patch(0.0, 0.0), patch(36e-6, 30e-6)
    assert abs(np.median(outer) - np.median(inner)) < 150e-9
    assert 0.5 < outer.std() / inner.std() < 2.0


def test_compiler_applies_lvdt_offset():
    """Volts = Position/Sensitivity + Offset (the driver's own convention)."""
    obj = LithoObject("l", "path", [(0.0, 0.0), (10.0, 0.0)])
    sens, off = 7.0e-6, -0.63
    c = compile_object(obj, sens, sens, off, off)
    assert c["x_v"][0] == pytest.approx(off)
    assert c["x_v"][-1] == pytest.approx(10e-6 / sens + off)
