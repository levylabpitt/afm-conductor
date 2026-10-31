"""Backend selection: one app, swappable backend (simulation vs real), both
satisfying the ARCBackend seam."""

import numpy as np

from afm_litho.hardware.arc_backend import ARCBackend
from afm_litho.hardware.backend import SIM, backend_mode, make_arc
from afm_litho.hardware.hardware_arc import HardwareARC
from afm_litho.twin.arc import SoftwareARC
from afm_litho.twin.physics import Sample, World


def _world():
    return World(sample=Sample(height_map=np.zeros((8, 8)), extent=1e-5))


def test_sim_backend_is_softwarearc():
    arc = make_arc(SIM, world=_world())
    assert isinstance(arc, SoftwareARC) and isinstance(arc, ARCBackend)


def test_real_backend_is_hardwarearc():
    arc = make_arc("hardware", link=object())      # link unused until a command
    assert isinstance(arc, HardwareARC) and isinstance(arc, ARCBackend)


def test_aliases_and_missing_args():
    import pytest
    assert isinstance(make_arc("real", link=object()), HardwareARC)
    with pytest.raises(ValueError):
        make_arc("hardware", link=None)            # real needs a link
    with pytest.raises(ValueError):
        make_arc("sim", world=None)                # sim needs a world


def test_env_default(monkeypatch):
    monkeypatch.delenv("AFM_LITHO_BACKEND", raising=False)
    assert backend_mode() == "sim"
    monkeypatch.setenv("AFM_LITHO_BACKEND", "Hardware")
    assert backend_mode() == "hardware"            # lowercased
    arc = make_arc(link=object())                  # mode from env
    assert isinstance(arc, HardwareARC)
