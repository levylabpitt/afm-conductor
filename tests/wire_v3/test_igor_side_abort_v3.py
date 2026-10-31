"""Wire v3: Igor's own abort (the AFM Abort panel, AFM_SafePark over Igor's COM route) reports
`REPORT <run> aborted` and `REPORT <run|-> madesafe`; the service stops the whole run on it,
as on v2 (tests/test_igor_side_abort.py)."""

from __future__ import annotations

import pytest
from golden.harness import GoldenRig
from golden.scenarios import WIRE, WIRE_B, _acquire, _release, _start_write
from test_igor_side_abort import _count, _engages, _parked_by_igor, _wait, _write_aborted

from afm_litho import clock
from afm_litho.hardware import wire

from .target import fake_available
from .test_golden_equivalence import v3_rig  # noqa: F401 -- the fixture

AVAILABLE, WHY = fake_available()
pytestmark = pytest.mark.skipif(not AVAILABLE, reason=WHY or "v3 target available")


@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


def press_igor_abort(rig) -> None:
    """AFM_SafePark run in Igor: the run ends aborted, then REPORT madesafe stamped com."""
    rig.stub.v3.make_safe("stop", route="com")


def test_v3_igor_abort_during_the_transit_ends_the_run(hub, client, v3_rig):  # noqa: F811
    with clock.using(clock.TESTING), wire.use_version(3), \
            GoldenRig(hub, "igor_abort_transit_v3") as rig:
        door, inst = rig.door, rig.inst
        tok = _acquire(door)
        inst.land_short(1)
        inst.hold_next("FIRE", 1)
        run = _start_write(door, tok, [WIRE, WIRE_B])
        inst.wait_held()
        press_igor_abort(rig)
        _write_aborted(rig, run)
        _parked_by_igor(hub)
        rig.quiesce()
        assert v3_rig, "no v3 verb reached the stub"
        assert _count(rig, "FIRE") == 1, "the transit was retried"
        assert _engages(rig) == [] and _count(rig, "ARMFIRE") == 0
        _release(door, tok)


def test_v3_madesafe_with_no_run_between_objects_stops_the_run(hub, client, v3_rig,  # noqa: F811
                                                              monkeypatch):
    """Between objects no PLAY is in flight: only `REPORT - madesafe` arrives."""
    from afm_litho.hardware.lithography_run import LithographyRun
    real_write = LithographyRun.write_object
    pressed: list = []

    def write_then_press(self, *a, **kw):
        out = real_write(self, *a, **kw)
        if not pressed:
            pressed.append(True)
            press_igor_abort(rig)
            _wait(hub.host._hw_litho_abort.is_set)
        return out

    monkeypatch.setattr(LithographyRun, "write_object", write_then_press)
    with clock.using(clock.TESTING), wire.use_version(3), \
            GoldenRig(hub, "igor_abort_between_v3") as rig:
        door = rig.door
        tok = _acquire(door)
        run = _start_write(door, tok, [WIRE, WIRE_B])
        _write_aborted(rig, run)
        _parked_by_igor(hub)
        rig.quiesce()
        assert _count(rig, "ARMFIRE") == 1 and _count(rig, "FIRE") == 1
        assert len(_engages(rig)) == 1
        _release(door, tok)
