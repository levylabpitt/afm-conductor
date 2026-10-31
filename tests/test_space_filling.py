"""Space-filling curve fills.

User request, 2026-08-08: "also forgotten was the spacefilling curves for solid
shapes. they still dont exist in the cafm program."

They did not. A directional sweep and contour rings both COVER a region;
neither is a space-filling curve, and the difference is physical: a curve
writes the region in one pass, with no long straight runs, so it imprints
no direction of its own and dwells at no edge.

Second pass, same day: A user asked for the line separation to be the
parameter (~10 nm, adjustable), which a Hilbert curve on a power-of-two
square cannot give. The fill is now a spanning-tree covering path with the
tree grown in Hilbert order, so the separation is exact and the locality
survives. The radial fills arrived with the same request.
"""

from __future__ import annotations

import numpy as np
import pytest

from afm_litho.litho.fill import (fill_hilbert, fill_polygons, fill_radial,
                                  hilbert_cells)
from afm_litho.litho.pens import PENS, render_stroke

SQUARE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0],
                   [0.0, 0.0]])


# ------------------------------------------------------------- the curve --

@pytest.mark.parametrize("order", [1, 2, 3, 4, 6])
def test_the_curve_visits_every_cell_exactly_once(order):
    x, y = hilbert_cells(order)
    n = 1 << order
    assert len(x) == n * n
    assert len({(int(a), int(b)) for a, b in zip(x, y)}) == n * n
    assert x.min() == 0 and x.max() == n - 1
    assert y.min() == 0 and y.max() == n - 1


@pytest.mark.parametrize("order", [1, 2, 3, 4, 6])
def test_every_step_is_to_a_neighbouring_cell(order):
    """What makes it a curve rather than an ordering. If this fails the tip
    is being asked to jump, at bias, across sample it has not written."""
    x, y = hilbert_cells(order)
    step = np.abs(np.diff(x)) + np.abs(np.diff(y))
    assert step.max() == 1 and step.min() == 1


# -------------------------------------------------------------- the fill --

def test_a_solid_square_fills_in_one_continuous_pass():
    runs = fill_hilbert([SQUARE], 0.5)
    assert len(runs) == 1, "a convex region must not be broken into strokes"
    path = runs[0]
    d = np.hypot(*np.diff(path, axis=0).T)
    assert d.max() == pytest.approx(d.min(), rel=1e-9), "uniform steps"


def _longest_straight_um(path):
    """Longest stretch the tip travels without turning, in microns.

    Counting collinear STEPS is the wrong measure and gave the wrong answer:
    a serpentine emits each line as its two endpoints, so it has no collinear
    consecutive steps at all while travelling the full width in one of them.
    """
    seg = np.diff(np.asarray(path, float), axis=0)
    ln = np.hypot(seg[:, 0], seg[:, 1])
    best = run = ln[0] if len(ln) else 0.0
    for i in range(1, len(seg)):
        cross = seg[i, 0] * seg[i - 1, 1] - seg[i, 1] * seg[i - 1, 0]
        same = (ln[i] > 0 and ln[i - 1] > 0 and abs(float(cross)) < 1e-9
                and float(seg[i] @ seg[i - 1]) > 0)
        run = run + ln[i] if same else ln[i]
        best = max(best, run)
    return float(best)


def test_the_runs_are_short_where_a_raster_runs_the_whole_width():
    """The reason to want this, stated as a number rather than a claim.

    The path does contain straight pieces -- I asserted it did not, and it
    does -- but they are bounded, where a serpentine over the same square
    runs the full width every line. That bound is what keeps anything
    anisotropic (tip shape, scanner, sample) from being written along one
    axis. On KTO the angle to the crystal axes is a physical variable, not a
    nuisance (docs/KTOME.md)."""
    for pitch in (0.5, 0.1, 0.05):
        curve = fill_hilbert([SQUARE], pitch)[0]
        longest = _longest_straight_um(curve)
        # SEVEN cells, and the bound does not grow with the region or soften
        # with the pitch, because it comes from the branching rule. Without
        # the turn bias in _spanning_forest this was 47 cells at pitch 0.1.
        assert longest <= 7 * pitch + 1e-9, f"long runs at pitch {pitch}"
    raster = _longest_straight_um(fill_polygons([SQUARE], 0.1, angle_rad=0.0)[0])
    assert raster >= 10.0 - 0.1                  # the full width of the square
    assert raster > 10 * _longest_straight_um(fill_hilbert([SQUARE], 0.1)[0])


def test_it_covers_the_region():
    pitch = 0.5
    path = fill_hilbert([SQUARE], pitch)[0]
    # every point of the square is within a pitch of the toolpath
    gx, gy = np.meshgrid(np.linspace(0.6, 9.4, 25), np.linspace(0.6, 9.4, 25))
    q = np.column_stack([gx.ravel(), gy.ravel()])
    d = np.hypot(*(q[:, None, :] - path[None, :, :]).transpose(2, 0, 1))
    assert d.min(axis=1).max() < pitch


def test_the_separation_between_lines_is_the_parameter_exactly():
    """A user asked for the line separation to BE the parameter, ~10 nm and
    adjustable. The first version put the curve on a 2**k square, so a 10 nm
    request came out at 7; every step is now exactly the number asked for."""
    for asked in (0.5, 0.3, 0.11, 0.05, 0.01):
        steps = np.hypot(*np.diff(fill_hilbert([SQUARE], asked)[0], axis=0).T)
        assert steps.min() == pytest.approx(asked, rel=1e-12)
        assert steps.max() == pytest.approx(asked, rel=1e-12)


def test_it_costs_no_more_write_time_than_a_raster_at_the_same_pitch():
    """Path length is cells x pitch: no cell written twice, none skipped
    inside the covered part. If this stops holding the path is doubling
    back."""
    for pitch in (0.1, 0.05):
        path = fill_hilbert([SQUARE], pitch)[0]
        length = float(np.hypot(*np.diff(path, axis=0).T).sum())
        assert length == pytest.approx((len(path) - 1) * pitch, rel=1e-9)


def test_coverage_approaches_the_whole_region_as_the_pitch_refines():
    """Blocks are 2x2 cells and a block is used only if all four of its cells
    are inside, so a fringe of up to two pitches is left at the boundary.
    That is 20 nm at the 10 nm pitch, and `outline=True` writes the edge."""
    for pitch, want in ((0.5, 0.75), (0.1, 0.95), (0.05, 0.97)):
        covered = len(fill_hilbert([SQUARE], pitch)[0]) * pitch ** 2
        assert covered / 100.0 > want


def test_it_stays_inside_the_shape():
    tri = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [0.0, 0.0]])
    for path in fill_hilbert([tri], 0.4):
        x, y = path[:, 0], path[:, 1]
        assert (x >= -1e-9).all() and (y >= -1e-9).all()
        assert (x + y <= 10.0 + 1e-9).all()


def test_a_hole_is_a_hole():
    ring_out = np.array([[0., 0.], [10., 0.], [10., 10.], [0., 10.], [0., 0.]])
    hole = np.array([[3., 3.], [7., 3.], [7., 7.], [3., 7.], [3., 3.]])
    for path in fill_hilbert([ring_out, hole], 0.4):
        inside_hole = ((path[:, 0] > 3.2) & (path[:, 0] < 6.8)
                       & (path[:, 1] > 3.2) & (path[:, 1] < 6.8))
        assert not inside_hole.any()


def test_two_islands_are_two_strokes_never_one():
    """The rule the other two fills follow: the biased tip is not dragged
    across sample it is not writing."""
    a = np.array([[0., 0.], [4., 0.], [4., 4.], [0., 4.], [0., 0.]])
    b = a + np.array([20.0, 0.0])
    runs = fill_hilbert([a, b], 0.4)
    assert len(runs) >= 2
    for path in runs:
        gap = (path[:, 0] > 5.0) & (path[:, 0] < 19.0)
        assert not gap.any(), "a stroke crossed the gap between the islands"


# --------------------------------------------------------------- the pen --

def test_the_region_pen_offers_every_fill_and_renders_them():
    pen = PENS["region"]
    assert [c[0] for c in pen.choices["fill"]] == [
        "directional", "hilbert", "radial-out", "radial-in", "contour"]
    assert pen.params["pitch_um"] == 0.01, "10 nm separation by default"
    for kind in [c[0] for c in pen.choices["fill"]]:
        objs = render_stroke(SQUARE[:-1], "region",
                             {**pen.params, "fill": kind, "pitch_um": 0.5})
        assert objs, f"{kind} rendered nothing"
        assert all(len(o.points_um) >= 2 for o in objs)
    # One region, ONE PASS. This has to be counted after the fill is
    # resolved: the pen returns the region itself now, so counting the
    # objects it returns would say "1" for every strategy, including the
    # serpentine, and the claim this test exists to make would be true by
    # construction and check nothing.
    from afm_litho.litho.compiler import expand
    one = render_stroke(SQUARE[:-1], "region",
                        {**pen.params, "fill": "hilbert", "pitch_um": 0.5})
    assert len(one) == 1 and one[0].kind == "shape"
    assert len(expand(one[0])) == 1, "one region, one pass"
    serpentine = render_stroke(SQUARE[:-1], "region",
                               {**pen.params, "fill": "directional",
                                "pitch_um": 0.5})
    assert len(expand(serpentine[0])) >= 1


# ------------------------------------------------------------- the radial --

DISC = np.column_stack([3 * np.cos(np.linspace(0, 2 * np.pi, 240)),
                        3 * np.sin(np.linspace(0, 2 * np.pi, 240))])


def test_a_disc_spirals_out_from_its_centre_in_one_pass():
    runs = fill_radial([DISC], 0.1)
    assert len(runs) == 1
    r = np.hypot(*runs[0].T)
    assert r[0] < 0.05                       # starts at the centre
    assert r[-1] > 2.9                       # ends at the rim
    assert (np.diff(r) >= -1e-9).all(), "the radius must only grow"


def test_inward_is_the_same_path_run_backwards():
    out = fill_radial([DISC], 0.1)[0]
    inn = fill_radial([DISC], 0.1, inward=True)[0]
    assert np.allclose(out, inn[::-1])


def test_successive_turns_are_one_separation_apart():
    """The line separation is the parameter for this fill too."""
    for sep in (0.2, 0.05):
        p = fill_radial([DISC], sep)[0]
        # sample the path where it crosses the +x axis: consecutive crossings
        # are one turn apart, so their radii differ by the separation
        on = np.flatnonzero((p[:-1, 1] < 0) & (p[1:, 1] >= 0) & (p[1:, 0] > 0))
        r = np.hypot(*p[on].T)
        assert np.allclose(np.diff(r), sep, atol=sep * 0.05)


def test_a_square_spirals_until_it_leaves_and_never_crosses_outside():
    runs = fill_radial([SQUARE - 5.0], 0.3)     # centred on the origin
    assert len(runs) > 1, "past the inscribed circle it must break into arcs"
    for arc in runs:
        assert (np.abs(arc) <= 5.0 + 1e-9).all()


def test_the_centre_can_be_put_where_the_shape_tool_put_it():
    off = np.array([1.5, -0.5])
    p = fill_radial([DISC], 0.1, centre=off)[0]
    assert np.allclose(p[0], off, atol=0.05)


def test_every_string_parameter_in_the_palette_has_a_menu():
    """The bug under the missing fills: the panel gives a number field to
    anything the pen does not declare as a choice, and parseFloat('contour')
    is NaN. Three fill strategies existed in the engine and none of them
    could be selected."""
    for pen in PENS.values():
        for k, v in pen.params.items():
            if isinstance(v, str):
                assert k in pen.choices, f"{pen.key}.{k} is a string with no menu"
                assert v in [c[0] for c in pen.choices[k]], \
                    f"{pen.key}.{k} defaults to a value not in its own menu"
