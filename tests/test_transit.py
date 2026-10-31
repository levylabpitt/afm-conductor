"""The tip must never jump, and must never write on its way anywhere."""

import numpy as np
import pytest

from afm_litho.litho.compiler import LithoObject, Pattern
from afm_litho.litho.transit import (TRANSIT_MODES, TransitPolicy, plan_transit,
                                     plan_pattern_transits, transit_report)
from afm_litho.service.passes import LithoRun, TwinInstrument
from afm_litho.twin.arc import SoftwareARC
from afm_litho.twin.physics import Sample, World


def _inst():
    sample = Sample(height_map=np.zeros((64, 64)), extent=30e-6)
    return TwinInstrument(SoftwareARC(World(sample=sample)))


@pytest.mark.parametrize("mode", ["lift", "slew", "contour"])
def test_speed_never_exceeds_the_ceiling(mode):
    pol = TransitPolicy(mode=mode, speed_um_s=500.0, max_speed_um_s=40.0)
    plan = plan_transit((0, 0), (25.0, 0.0), pol)
    p = plan["path_um"]
    dt = plan["duration_s"] / max(len(p) - 1, 1)
    v = np.hypot(*np.diff(p, axis=0).T) / dt
    assert v.max() <= pol.max_speed_um_s * 1.02
    assert plan["duration_s"] >= 25.0 / pol.max_speed_um_s


def test_trapezoid_starts_and_ends_at_rest():
    plan = plan_transit((0, 0), (20.0, 0.0), TransitPolicy(mode="slew"))
    p = plan["path_um"]
    step = np.hypot(*np.diff(p, axis=0).T)
    # a constant-speed move would have equal steps everywhere; the ramps
    # make the first and last steps markedly shorter
    assert step[0] < 0.5 * step[len(step) // 2]
    assert step[-1] < 0.5 * step[len(step) // 2]


def test_lift_clears_the_surface_and_lands():
    pol = TransitPolicy(mode="lift", lift_nm=200.0)
    plan = plan_transit((0, 0), (10.0, 0.0), pol)
    z = plan["z_nm"]
    assert z.max() == pytest.approx(200.0)
    assert z[0] == pytest.approx(0.0)
    assert z[-1] == pytest.approx(0.0)          # settled back in contact
    # airborne for most of the move, not just an instant at the midpoint
    assert (z > 0.9 * 200.0).mean() > 0.6


def test_slew_stays_in_contact():
    plan = plan_transit((0, 0), (10.0, 0.0), TransitPolicy(mode="slew"))
    assert np.all(plan["z_nm"] == 0.0)


def test_none_is_reported_as_the_unphysical_jump_it_is():
    plan = plan_transit((0, 0), (10.0, 0.0), TransitPolicy(mode="none"))
    assert plan["n"] == 2
    assert plan["peak_um_s"] == float("inf")
    # ...but the report must stay JSON-serialisable, so infinity becomes None
    rep = transit_report([plan])
    assert rep["peak_um_s"] is None
    import json
    json.dumps(rep)


def test_coincident_points_need_no_move():
    assert plan_transit((3.0, 3.0), (3.0, 3.0), TransitPolicy())["n"] == 0


def test_report_covers_every_gap_including_the_first():
    paths = [("a", np.array([[0.0, 0.0], [5.0, 0.0]])),
             ("b", np.array([[5.0, 8.0], [0.0, 8.0]]))]
    plans = plan_pattern_transits(paths, (0.0, 0.0), TransitPolicy(mode="slew"))
    assert [p["to"] for p in plans] == ["a", "b"]
    assert transit_report(plans)["moves"] == 1        # the move to "a" is zero-length
    assert transit_report(plans)["distance_um"] == pytest.approx(8.0)


@pytest.mark.parametrize("mode", TRANSIT_MODES)
def test_nothing_is_written_between_strokes(mode):
    """The whole point: dose appears on the strokes and nowhere else.

    Two parallel wires 6 um apart. Whatever route the tip takes between
    them, the corridor in between must come back clean.
    """
    inst = _inst()
    pat = Pattern(name="t", objects=[
        LithoObject("w1", "path", [(-4.0, -3.0), (4.0, -3.0)], voltage=9.0,
                    speed_um_s=8.0),
        LithoObject("w2", "path", [(-4.0, 3.0), (4.0, 3.0)], voltage=9.0,
                    speed_um_s=8.0),
    ])
    import queue
    LithoRun(inst, pat, queue.Queue(), TransitPolicy(mode=mode)).run()

    s = inst.world.sample
    ax = np.linspace(-4e-6, 4e-6, 40)

    def dose_along(y_um):
        g = np.array([s.conductance(x, y_um * 1e-6) for x in ax])
        return float(g.max())

    assert dose_along(-3.0) > 0.2, "wire 1 should be written"
    assert dose_along(3.0) > 0.2, "wire 2 should be written"
    assert dose_along(0.0) < 0.02, f"{mode} transit wrote in the gap"


def test_scan_line_ends_where_the_next_one_starts():
    """A reported line is trace + retrace, so the tip comes home.

    Leaving it at the far end made the tip appear to fly the full scan
    width between every pair of lines.
    """
    from afm_litho.service.passes import ScanConfig
    inst = _inst()
    w = inst.world
    inst.start_scan(ScanConfig(size_um=30.0, pixels=32, lines=8, line_rate_hz=2.0))
    inst.scan_line()
    x0 = (w.x_v - w.x_lvdt_offset) * w.x_lvdt_sens * 1e6
    inst.scan_line()
    x1 = (w.x_v - w.x_lvdt_offset) * w.x_lvdt_sens * 1e6
    assert abs(x1 - x0) < 0.1                       # only a slow-axis step
    assert abs(x0 + 15.0) < 0.1                     # parked at the line start
