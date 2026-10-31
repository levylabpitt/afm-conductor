"""Where an imported SVG design is PLACED on the sample.

The design-origin rule (docs/DESIGN_ORIGIN_SPEC.md) says the top-left of the
SVG page lands on the top-left of the scan, and the mapping from page-origin
microns to lab coordinates (`litho.frame.svg_to_lab`) needs one number: the
scan SIZE the design was drawn over. This module is the single place both
import doors -- `POST /api/design/svg` and FLEX `loadPattern` -- resolve that
size, so the two cannot drift on it.

Resolution order (2026-09-18 review):
  1. the CURRENT SCAN FRAME -- the live scan config, else `last_scan_config`. This
     is AUTHORITATIVE: the rule is "upper-left of the current scan in the UI =
     upper-left of the SVG", and real design files carry STALE background pictures
     (Mock_Device.svg is a 40 um page around a 20 um picture), so a live/last scan
     the operator is actually looking at wins over a possibly-stale embedded image;
  2. else the design's own background `<image>` (the `topo_bgd` element the
     cockpit's "Update design background" button writes, width = size_um * 1000) --
     used only when no scan frame exists at all;
  3. else REFUSE, rather than guess a size and silently place the design wrong.
"""

from __future__ import annotations

from .svg_background import background_size_um


class DesignSizeError(Exception):
    """No scan frame and no background image -- the design's size is unknown."""


def resolve_import_size_um(svg_text: str, hub) -> tuple[float, str]:
    """`(size_um, source)` for an SVG import, or raise `DesignSizeError`.

    `source` is `"frame"` (from the current/last scan, tried FIRST because it is
    what the operator is looking at) or `"image"` (from the design's background
    `<image>`, used only when no scan frame exists). Never guesses: a design with
    neither is refused so it is not placed against a size nobody chose.
    """
    scan = getattr(getattr(hub.host, "inst", None), "scan", None)
    cfg = getattr(scan, "config", None) if scan is not None else None
    if cfg is not None and getattr(cfg, "size_um", 0):
        return float(cfg.size_um), "frame"
    last = getattr(hub.host, "last_scan_config", None)
    if last and last.get("size_um"):
        return float(last["size_um"]), "frame"
    size = background_size_um(svg_text)
    if size and size > 0:
        return float(size), "image"
    raise DesignSizeError(
        "cannot place the design: no scan frame is set and it carries no "
        "background scan image, so the scan size it was drawn over is unknown. "
        "Run a scan first, or embed the scan background in the SVG (the cockpit's "
        "Update design background button does this).")


def frame_bbox_check(pattern, size_um: float):
    """`(outside_frame, bbox_um)` for a placed design.

    `bbox_um` is `[x0, y0, x1, y1]` in lab microns over every object's mapped
    geometry (`points_um` and any `holes_um`), or None for an empty pattern.
    `outside_frame` is True when that box runs past +/- size_um/2 on either axis:
    the design has been placed partly OUTSIDE the scan it was drawn over. This is
    a sanity flag, not a refusal -- a design may legitimately extend past the
    frame -- so the doors log a WARNING and report it, and write nothing off.
    """
    half = size_um / 2.0
    lo_x = lo_y = float("inf")
    hi_x = hi_y = float("-inf")
    for o in getattr(pattern, "objects", []):
        rings = [getattr(o, "points_um", None) or []]
        rings.extend(getattr(o, "holes_um", None) or [])
        for ring in rings:
            for x, y in ring:
                lo_x, hi_x = min(lo_x, x), max(hi_x, x)
                lo_y, hi_y = min(lo_y, y), max(hi_y, y)
    if lo_x == float("inf"):
        return False, None
    bbox = [round(lo_x, 4), round(lo_y, 4), round(hi_x, 4), round(hi_y, 4)]
    eps = 1e-6
    outside = (lo_x < -half - eps or hi_x > half + eps
               or lo_y < -half - eps or hi_y > half + eps)
    return outside, bbox
