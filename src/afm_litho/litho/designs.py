"""Design generation: auto-routed devices that terminate on real contacts.

The loaded scan's mesas are the interface contacts (Sample.seed_contacts
finds them). auto_device() picks pads and builds a SketchFET whose leads
are tapered fills running from pad edges to the device, the geometry every
hand-drawn Inkscape design in the lab's archive uses.

CORRECTION PENDING (2026-08-08). The contact detection below is
looking at the wrong features.

In a real scan of these devices the BRIGHT regions are the top gold that runs
out to the wire bond pads. They are not the electrical contacts. The contacts
to the interface are PILL-SHAPED and sit in contact with that gold, and the
writable canvas -- the region a device can actually be drawn in -- is much
smaller than the scan frame and smaller than this module assumes.

Detecting the gold mesas and routing between them produces leads that run
across the whole image between the wrong things, when Write is pressed with
no sketch loaded and the auto device is substituted.

Until this is corrected, treat auto_device() on a real scan as a placeholder.

NOT the green triangle, though I said it was. That was the conductance
overlay drawing the pad seeding back at him, and it was fixed
(2026-08-08). The two faults share a
cause -- pads detected on the gold -- which is what made the wrong one look
like the culprit. The correction above is still owed.
"""

from __future__ import annotations

import numpy as np

from ..twin.physics import Sample
from .compiler import LithoObject, Pattern


def _toward(p, q, dist):
    """Point at `dist` from p along p->q."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    v = q - p
    n = np.hypot(*v)
    return tuple(p + v / n * min(dist, n)) if n > 0 else tuple(p)


def _pad_edge(sample: Sample, pad, target, overlap_um: float = 0.8):
    """Start point for a lead: just inside the pad's true boundary facing the
    target. The contacts are elongated (fingernail-shaped), so the boundary is
    found by walking the metal mask along the approach direction rather than
    assuming a circular pad; only `overlap_um` of written taper stays on gold.
    """
    cx, cy, _area = pad
    p = np.array([cx, cy], float)
    q = np.array(target, float)
    v = q - p
    n = float(np.hypot(*v))
    if n <= 0:
        return (cx, cy)
    u = v / n
    edge = 0.0
    step = 0.15
    d = 0.0
    while d < n:
        pt = p + u * d
        if not sample.is_metal_um(pt[0], pt[1]):
            break
        edge = d
        d += step
    start = p + u * max(0.0, edge - overlap_um)
    return (float(start[0]), float(start[1]))


def auto_device(sample: Sample, write_v: float = 8.0,
                speed: float = 2.0) -> Pattern:
    pads = getattr(sample, "contact_centroids_um", [])
    if len(pads) < 4:
        # featureless sample: fall back to a floating demo at the center
        from .compiler import demo_pattern
        return demo_pattern()

    # pads are (x, y, area_um2); choose source/drain as the most x-opposed pair
    left = min(pads, key=lambda p: p[0])
    right = max(pads, key=lambda p: p[0])
    center = np.array([(left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0])
    # gates: the two topmost remaining pads, matched to gate side by x
    rest = [p for p in pads if p is not left and p is not right]
    rest.sort(key=lambda p: -p[1])
    top_two = rest[:2] if len(rest) > 1 else [rest[0], rest[0]]
    top_two = sorted(top_two, key=lambda p: p[0])   # no crossed leads
    g1, g2 = top_two[0], top_two[1]

    # wire endpoints: pull in from the pads toward center
    left_xy = np.array(left[:2])
    right_xy = np.array(right[:2])
    w_a = _toward(left_xy, center, float(np.hypot(*(center - left_xy))) - 4.0)
    w_b = _toward(right_xy, center, float(np.hypot(*(center - right_xy))) - 4.0)
    def on_wire(x):
        """Point on the wire segment at horizontal position x."""
        t = (x - w_a[0]) / (w_b[0] - w_a[0])
        t = min(max(t, 0.0), 1.0)
        return (w_a[0] + t * (w_b[0] - w_a[0]), w_a[1] + t * (w_b[1] - w_a[1]))

    # four-terminal device: two voltage probes land ON the wire, one on each
    # side of the barrier junction
    p1_foot = on_wire(center[0] - 2.5)
    p2_foot = on_wire(center[0] + 2.5)
    p1_knee = (p1_foot[0], p1_foot[1] + 3.2)
    p2_knee = (p2_foot[0], p2_foot[1] + 3.2)
    barrier_mid = on_wire(center[0])

    def lead(name, pad, target):
        # start just inside the true (walked) pad boundary facing the target
        return LithoObject(name, "taper",
                           [_pad_edge(sample, pad, target), target],
                           voltage=write_v, speed_um_s=speed * 2,
                           width0_um=3.0, width1_um=0.3, fill_spacing_um=0.25,
                           interpolation_um=0.05)

    objs = [
        lead("lead-source", left, w_a),
        lead("lead-drain", right, w_b),
        LithoObject("wire", "path", [w_a, w_b],
                    voltage=write_v, speed_um_s=speed),
        lead("lead-probe-1", g1, p1_knee),
        LithoObject("probe-1", "path", [p1_knee, p1_foot],
                    voltage=write_v, speed_um_s=speed),
        lead("lead-probe-2", g2, p2_knee),
        LithoObject("probe-2", "path", [p2_knee, p2_foot],
                    voltage=write_v, speed_um_s=speed),
        LithoObject("erase-barrier", "path",
                    [(barrier_mid[0], barrier_mid[1] + 1.2),
                     (barrier_mid[0], barrier_mid[1] - 1.2)],
                    voltage=-write_v, speed_um_s=speed * 0.75),
    ]
    return Pattern(name="auto-device", objects=objs)
