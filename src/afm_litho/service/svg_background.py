"""Swap a design SVG's background `<image>` for a scan PNG, in place.

The one thing this must not do is what Inkscape files punish hardest: round
one through `xml.etree.ElementTree` unprepared and every `xlink:`/`inkscape:`/
`sodipodi:` attribute comes back out as `ns0:...`, because ET invents a prefix
for any namespace URI it has not been told a name for. `ET.register_namespace`
fixes that -- it is a small piece of module-global state consulted at
*serialization* time, so registering it before `swap_background` returns is
what matters; doing it up front here (before `fromstring`, per the plan) keeps
the parse and the fix visibly paired rather than relying on that ordering
detail.

Strict and narrow by design (docs/IMAGE_SAVING_NOTES.md 2026-09-17): this is
not a general SVG editor. It finds (or makes) ONE background `<image>`, points
it at the new PNG by reference, and touches nothing else -- position,
transform, every other element, untouched.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
# The REAL Inkscape sodipodi URI ends .dtd, not .0 -- a plausible-looking typo
# that silently never matches a real file's namespace (ET registers a prefix
# by exact URI), so registration was a no-op and the writer fell back to
# inventing ns2:namedview etc. Caught on a real Inkscape file after this
# branch merged.
SODIPODI_NS = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
# Inkscape's default template also carries Creative Commons + RDF metadata
# (<cc:...>/<rdf:RDF> inside <metadata>) using these two namespaces; `dc` is
# already known to ElementTree by default, so only these two need registering.
CC_NS = "http://creativecommons.org/ns#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

_IMAGE_TAG = f"{{{SVG_NS}}}image"
_XLINK_HREF = f"{{{XLINK_NS}}}href"

#: The size floor (Inkscape/user units) that tells a real background apart
#: from a small icon/logo dropped into the same design.
_MIN_BG_SIZE = 100.0

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _num(value: str | None) -> float | None:
    """The leading number in an SVG length ("120", "120.5", "120px"), or None
    for anything absent/unparsable. Only the number is needed here -- the unit
    suffix (if any) is a display convention this lab's SVGs do not use."""
    if not value:
        return None
    m = _NUM_RE.match(value.strip())
    return float(m.group()) if m else None


def _fmt_len(value: float) -> str:
    """A plain SVG length: integral sizes print without a trailing ".0"."""
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def background_size_um(svg_text: str) -> float | None:
    """The scan size a design was drawn over, read off its background `<image>`.

    Selection mirrors `swap_background` exactly (docs/IMAGE_SAVING_NOTES.md), so
    the image the cockpit's "Update design background" button writes is the one
    read back: prefer the element with `id="topo_bgd"`, else the FIRST `<image>`
    in document order (= bottom-most = the background); either way `width` AND
    `height` must both be > 100 user units. The width is in the same 1-unit-is-1-nm
    convention the writer uses (`width = size_um * 1000`), so `size_um = width / 1000`.
    Returns None when the design carries no such background image.
    """
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return None
    tagged = None
    fallback = None
    for el in root.iter(_IMAGE_TAG):
        w, h = _num(el.get("width")), _num(el.get("height"))
        if w is None or h is None:
            continue
        # The `topo_bgd` id branch holds to the same > 100-unit floor as the
        # bottom-most fallback: a real scan background is tens of thousands of
        # units, so the floor rejects a tiny/decorative image that happens to
        # carry the id rather than reading it back as a scan size.
        if el.get("id") == "topo_bgd" and w > _MIN_BG_SIZE and h > _MIN_BG_SIZE:
            tagged = w
            break
        if fallback is None and w > _MIN_BG_SIZE and h > _MIN_BG_SIZE:
            fallback = w
    width = tagged if tagged is not None else fallback
    return None if width is None else width / 1000.0


def swap_background(svg_text: str, png_href: str, size_um: float) -> str:
    """Return `svg_text` with its background `<image>` pointed at `png_href`.

    Selection (docs/IMAGE_SAVING_NOTES.md): the FIRST `<image>` in document
    order, at any nesting depth (inside a layer `<g>`, for instance), whose
    `width` AND `height` are both > 100 user units -- document order is paint
    order, so the first one is the bottom-most, i.e. the background. A design
    with no such element gets a brand new `<image>` inserted as the root's
    first child at x=0, y=0 (painted first = bottom-most).

    Either way the element ends up with `id="topo_bgd"`, `width=height=
    size_um*1000` (1 SVG unit = 1 nm, matching the design-import convention),
    and an `xlink:href` of `png_href` -- LINKED, never embedded as data. A
    plain SVG2 `href` is set too when the element already carries one (Inkscape
    writes both), so whichever attribute a renderer honours points at the same
    file. Position (`x`/`y`) and any `transform` on an EXISTING element are
    left exactly as found.
    """
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    ET.register_namespace("inkscape", INKSCAPE_NS)
    ET.register_namespace("sodipodi", SODIPODI_NS)
    ET.register_namespace("cc", CC_NS)
    ET.register_namespace("rdf", RDF_NS)

    root = ET.fromstring(svg_text)

    target = None
    for el in root.iter(_IMAGE_TAG):
        w, h = _num(el.get("width")), _num(el.get("height"))
        if w is not None and h is not None and w > _MIN_BG_SIZE and h > _MIN_BG_SIZE:
            target = el
            break

    if target is None:
        target = ET.Element(_IMAGE_TAG)
        target.set("x", "0")
        target.set("y", "0")
        root.insert(0, target)

    side = _fmt_len(size_um * 1000.0)
    target.set("id", "topo_bgd")
    target.set("width", side)
    target.set("height", side)
    target.set(_XLINK_HREF, png_href)
    if target.get("href") is not None:
        target.set("href", png_href)

    return ET.tostring(root, encoding="unicode")
