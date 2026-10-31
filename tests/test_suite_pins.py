"""The suite's own settings (H5.pre): the wire version under the goldens, and xdist grouping."""

from __future__ import annotations

import os

import pytest

from afm_litho import wire_version
from afm_litho.hardware import wire


def test_the_suite_runs_on_the_v2_wire_whatever_the_shell_says():
    assert os.environ[wire_version.ENV] == "2"
    assert wire.WIRE_VERSION == 2


def test_xdist_groups_are_honoured_when_xdist_is_loaded(request):
    if not request.config.pluginmanager.hasplugin("xdist"):
        pytest.skip("xdist is not loaded")
    assert request.config.option.dist == "loadgroup"
