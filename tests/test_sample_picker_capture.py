"""The sample picker lists a hardware scan capture and loading it seeds the twin's real retrace."""

import numpy as np
import pytest
from hub_fields import owner

CAPTURE = "20261022_193212.npz"


@pytest.fixture
def restore_twin():
    from afm_litho.service.app import hub
    from afm_litho.service.twin_plugin import forget
    names = ("instrument", "world", "arc", "inst", "surface", "written_objects",
             "written_accum", "sample_name", "pattern", "ports", "ports_seq")
    saved = {n: getattr(owner(hub, n), n) for n in names}
    yield hub
    forget(hub)
    for n, v in saved.items():
        setattr(owner(hub, n), n, v)


def test_picker_lists_and_loads_the_capture_with_its_real_retrace(client, restore_twin):
    from afm_litho.service.twin_plugin import plugin
    assert CAPTURE in client.get("/api/samples").json()["files"]
    reply = client.post("/api/sample/load", json={"file": CAPTURE}).json()
    assert reply.get("ok") is True and reply["name"] == "20261022_193212"
    twin = plugin(restore_twin)
    assert twin.sample.retrace is not None
    corr = np.corrcoef(twin.sample.height.ravel(), twin.sample.retrace.ravel())[0, 1]
    assert corr < 0.999
