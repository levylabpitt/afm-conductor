"""Where the twin thinks the contacts are.

User request, 2026-08-08: "i dont think you found the interface contacts. I wrote
between two of them and no jump."

He was right, and the cause was cruder than the known gold-versus-pills
correction. seed_contacts labelled the MIRROR-TILED mask, so a feature
touching the edge of the loaded scan merged with its own reflection and the
merged component's centroid landed on the seam by symmetry. Measured on
data/samples/SA40656B0000.ibw: 21 of the 40 contacts sat exactly on the
+-15.06 um mirror line and NOT ONE was inside the 30 um scan. There was
nothing in the frame to write between, so no wire drawn there could ever
have moved the conductance matrix.
"""

from __future__ import annotations

import numpy as np
import pytest

from afm_litho.twin.physics import Sample, _mirror_images


def _sample_with_two_pads(n=256, extent=30e-6):
    """A flat surface with two square mesas well inside the frame."""
    h = np.zeros((n, n))
    h[40:70, 40:70] = 80e-9                      # around (-11, -11) um
    h[180:220, 150:200] = 80e-9                  # around (+8, +5) um
    return Sample(height_map=h, extent=extent)


def test_a_contact_lands_on_the_feature_that_produced_it():
    s = _sample_with_two_pads()
    half = s.extent * 1e6 / 2
    inside = [c for c in s.contact_centroids_um
              if abs(c[0]) <= half and abs(c[1]) <= half]
    assert len(inside) == 2, f"expected the two mesas, got {len(inside)}"
    for x, y, _area in inside:
        z = float(s.height(np.array(x * 1e-6), np.array(y * 1e-6)))
        assert z > 40e-9, f"contact at ({x:.1f}, {y:.1f}) is not on a mesa"


def test_no_contact_sits_on_the_mirror_seam():
    """The seam is an artifact of continuing the scan across the scanner
    range. Nothing is there."""
    s = _sample_with_two_pads()
    half = s.extent * 1e6 / 2
    for x, y, _ in s.contact_centroids_um:
        on = min(abs(abs(x) - half), abs(abs(y) - half))
        assert on > 0.2, f"contact at ({x:.1f}, {y:.1f}) is on the seam"


@pytest.mark.parametrize("ibw", ["SA40656B0000.ibw", "SA40656B0002.ibw"])
def test_real_scans_have_contacts_inside_the_frame_you_are_looking_at(ibw):
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / "samples" / ibw
    if not path.is_file():
        pytest.skip(f"{ibw} not present")
    s = Sample.from_ibw(str(path))
    half = s.extent * 1e6 / 2
    inside = [c for c in s.contact_centroids_um
              if abs(c[0]) <= half and abs(c[1]) <= half]
    assert inside, "not one contact inside the loaded scan"
    seam = [c for c in s.contact_centroids_um
            if min(abs(abs(c[0]) - half), abs(abs(c[1]) - half)) < 0.2]
    assert not seam, f"{len(seam)} contacts on the mirror seam"


def test_the_contact_is_the_most_interior_point_not_the_centre_of_mass():
    """A C-shaped feature has its centre of mass on bare surface, and an
    electrode attached there is attached to nothing. Three of the eight found
    in SA40656B0000 were like that."""
    n = 256
    h = np.zeros((n, n))
    h[60:200, 60:200] = 80e-9
    h[90:170, 90:260] = 0.0                      # carve it into a C
    s = Sample(height_map=h, extent=30e-6)
    half = s.extent * 1e6 / 2
    inside = [c for c in s.contact_centroids_um
              if abs(c[0]) <= half and abs(c[1]) <= half]
    assert inside
    for x, y, _ in inside:
        z = float(s.height(np.array(x * 1e-6), np.array(y * 1e-6)))
        assert z > 40e-9, f"contact at ({x:.1f}, {y:.1f}) is in the notch"


# --------------------------------------------------------- the tiling rule --

def test_the_images_are_the_inverse_of_the_tiling():
    """_mirror_images must land exactly where _map_coords folds back to, or
    the contacts are near the features rather than on them."""
    s = Sample(height_map=np.zeros((64, 64)), extent=30e-6)
    p = s.extent * 1e6
    for v0 in (0.0, 3.0, -11.5, 14.9):
        want = s._map_coords(np.array(v0 * 1e-6), np.array(0.0))[0]
        for v in _mirror_images(v0, p, 45.0):
            got = s._map_coords(np.array(v * 1e-6), np.array(0.0))[0]
            assert float(got) == pytest.approx(float(want), abs=1e-6), (
                f"image {v:.2f} of {v0:.2f} folds somewhere else")


def test_every_feature_repeats_across_the_scanner_range():
    """The images are the point: the real chip's pad array continues past the
    loaded scan, and the tip can reach it."""
    s = _sample_with_two_pads()
    half_range = s.dose_range * 1e6 / 2
    xs = [c[0] for c in s.contact_centroids_um]
    assert max(xs) > s.extent * 1e6 / 2, "no contact outside the loaded scan"
    assert all(abs(x) <= half_range + 1e-6 for x in xs)


# ------------------------------------------------ marking them by hand --

def _web_path(name: str):
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / "src" / "afm_litho" / "web" / name


_needs_old_window = pytest.mark.skipif(
    not (_web_path("app.js").exists() and _web_path("app-contacts.js").exists()),
    reason="the old main window's scripts are not in this tree")


def _web(name: str) -> str:
    from pathlib import Path
    return (Path(__file__).resolve().parents[1] / "src" / "afm_litho" / "web"
            / name).read_text(encoding="utf-8")


def _app_js() -> str:
    return _web("app.js")


# The ES-module split (WEB_REFACTOR_PLAN.md Phase 5) moved the contact stadium
# geometry + hit-test into app-contacts.js; the marking/removal handler that
# rebuilds state.contacts stayed in app.js. These are repointed structural
# checks (plan §5): contactAt only reads state, but its import graph pulls in
# document, so it is not cleanly node-importable for a behavioural test.
@_needs_old_window
def test_one_contact_can_be_removed_on_its_own():
    """Erasing all of them and re-marking to lose a single mistake is not a
    tool. A click while marking removes the one under it."""
    assert "function contactAt(" in _web("app-contacts.js")
    assert "state.contacts.filter((_, k) => k !== i)" in _app_js()


@_needs_old_window
def test_the_click_test_is_the_same_stadium_the_physics_uses():
    """What you can click has to be exactly what is conductive, or removing a
    contact becomes a guessing game."""
    src = _web("app-contacts.js")
    for piece in ("Math.max(Math.abs(u) - (hw - r), 0)",
                  "Math.max(Math.abs(v) - (hh - r), 0)",
                  "Math.hypot(qx, qy) <= r"):
        assert piece in src, f"the hit test no longer matches: {piece}"


@_needs_old_window
def test_the_geometry_is_what_travels_so_klayout_can_fill_it():
    """User request: "later I will give the klayout that created these and it will
    be automatic." Points would not survive that; shapes do."""
    src = _app_js()
    for k in ("x_um:", "y_um:", "w_um:", "h_um:", "angle_deg:"):
        assert k in src, f"the marking tool no longer sends {k}"
