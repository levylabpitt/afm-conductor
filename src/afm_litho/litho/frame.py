"""The frame a pattern is drawn in, and how it maps to scanner coordinates.

Litho runs in the scan's ROTATED frame, offset by the litho panel's own fast/slow size and
offset, at the IMAGING angle (`RVW[%ScanAngle]`). Lithography.ipf builds every drive wave as

    AngleWave = RotateWave(DriveY - LithoSlowScanSize/2 - LithoSlowScanOffset,
                           DriveX - LithoFastScanSize/2 - LithoFastScanOffset,
                           scanAngle)

RotateWave rotates by MINUS A (fast axis -> (cos A, -sin A)); intentional, verified at
Lithography.ipf:3559, ScanMaster.ipf:2619 and 4409. Imaging and litho share `rotate()`. See
docs/asylum-v16-findings.md, "Scan angle rotates the litho coordinates".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def rotate(fast, slow, angle_deg: float):
    """Frame (fast, slow) -> lab (x, y), the vendor's RotateWave. Rigid, so arc lengths,
    step sizes and timing are the same before or after interpolation."""
    if not angle_deg:
        return fast, slow
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    return fast * ca + slow * sa, -fast * sa + slow * ca


def unrotate(x, y, angle_deg: float):
    """Lab (x, y) -> frame (fast, slow). The inverse of `rotate`."""
    return rotate(x, y, -angle_deg)


def svg_to_lab(points, size_um: float):
    """Map SVG page-origin microns (right/down from the top-left) to centred lab microns:
    x - size_um/2, size_um/2 - y, no scale (docs/DESIGN_ORIGIN_SPEC.md). `size_um` is the
    scan the design was drawn over. Imaging offsets are added later by the servo."""
    h = size_um / 2.0
    return [(x - h, h - y) for x, y in points]


def bezier_paths_to_lab(paths, size_um: float) -> None:
    """Apply `svg_to_lab` in place to every node point and both Bezier handles.
    Duck-typed over BezierPath/Node to avoid an import cycle with the bezier module."""
    h = size_um / 2.0
    for p in paths:
        for nd in p.nodes:
            nd.x, nd.y = nd.x - h, h - nd.y
            if nd.h_in is not None:
                nd.h_in = (nd.h_in[0] - h, h - nd.h_in[1])
            if nd.h_out is not None:
                nd.h_out = (nd.h_out[0] - h, h - nd.h_out[1])


@dataclass
class LithoFrame:
    """Where a pattern's coordinates live. Fields mirror the vendor's LithoFastScanSize,
    LithoSlowScanSize, LithoFastScanOffset, LithoSlowScanOffset and the imaging ScanAngle.

    The default is the identity: twin designs are centred um, so no offset. `for_image()`
    builds the vendor's other convention, a pattern in image coordinates 0..size.

    The imaging scan offset is NOT here: the XY loops add it under the drive wave
    (SetpointOffset = XOffset/abs(XLVDTSens) + xLVDTOffset); adding it here doubles it.
    """

    fast_size_um: float = 0.0
    slow_size_um: float = 0.0
    fast_offset_um: float = 0.0
    slow_offset_um: float = 0.0
    angle_deg: float = 0.0

    @classmethod
    def for_image(cls, size_um: float, angle_deg: float = 0.0,
                  fast_offset_um: float = 0.0, slow_offset_um: float = 0.0):
        """A frame for a pattern drawn in image coordinates (0..size)."""
        return cls(fast_size_um=size_um, slow_size_um=size_um,
                   fast_offset_um=fast_offset_um,
                   slow_offset_um=slow_offset_um, angle_deg=angle_deg)

    @property
    def is_identity(self) -> bool:
        return not (self.fast_size_um or self.slow_size_um
                    or self.fast_offset_um or self.slow_offset_um
                    or self.angle_deg)

    def to_lab(self, fast, slow):
        """Pattern coordinates -> scanner coordinates, both in um."""
        f = fast - self.fast_size_um / 2 - self.fast_offset_um
        s = slow - self.slow_size_um / 2 - self.slow_offset_um
        return rotate(f, s, self.angle_deg)

    def from_lab(self, x, y):
        """Scanner coordinates -> pattern coordinates. Exact inverse."""
        f, s = unrotate(x, y, self.angle_deg)
        return (f + self.fast_size_um / 2 + self.fast_offset_um,
                s + self.slow_size_um / 2 + self.slow_offset_um)

    def path_to_lab(self, path: np.ndarray) -> np.ndarray:
        """Apply to an (N, 2) path of (fast, slow) pairs."""
        if self.is_identity:
            return path
        x, y = self.to_lab(path[:, 0], path[:, 1])
        return np.column_stack([x, y])

    def to_dict(self) -> dict:
        return {"fast_size_um": self.fast_size_um,
                "slow_size_um": self.slow_size_um,
                "fast_offset_um": self.fast_offset_um,
                "slow_offset_um": self.slow_offset_um,
                "angle_deg": self.angle_deg}

    @classmethod
    def from_dict(cls, d: dict | None):
        if not d:
            return cls()
        return cls(fast_size_um=float(d.get("fast_size_um", 0.0)),
                   slow_size_um=float(d.get("slow_size_um", 0.0)),
                   fast_offset_um=float(d.get("fast_offset_um", 0.0)),
                   slow_offset_um=float(d.get("slow_offset_um", 0.0)),
                   angle_deg=float(d.get("angle_deg", 0.0)))
