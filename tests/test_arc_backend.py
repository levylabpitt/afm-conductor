"""SoftwareARC must satisfy the ARCBackend seam, so the pass executor written
against the Protocol drives it -- and later a bridge-backed HardwareARC -- with
no change. And the seam must NOT include step() (the twin's sim engine has no
hardware counterpart)."""

import numpy as np

from afm_litho.hardware.arc_backend import ARCBackend
from afm_litho.twin.arc import SoftwareARC
from afm_litho.twin.physics import Sample, World


def test_software_arc_satisfies_the_backend_seam():
    arc = SoftwareARC(World(sample=Sample(height_map=np.zeros((8, 8)), extent=1e-5)))
    assert isinstance(arc, ARCBackend)          # runtime_checkable Protocol


def test_seam_is_arm_and_fire_not_step():
    methods = {n for n in dir(ARCBackend) if not n.startswith("_")}
    assert "step" not in methods                # no per-tick engine in the seam
    # the arm-and-fire surface the executor needs IS in it
    for m in ("ir_SetPISLoop", "td_xSetOutWavePair", "td_xSetInWavePair",
              "td_WriteString", "td_ReadValue", "td_Stop"):
        assert m in methods
