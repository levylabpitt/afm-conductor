"""OASIS boundary codec for litho patterns.

OASIS is the canonical design format: geometry carries pass parameters as
named per-shape properties (litho.voltage, litho.speed, litho.name, ...).
Uses the headless klayout module; GDSII comes free (same reader/writer,
though GDS drops properties on very old tools; OASIS keeps them natively).

Layers: 10/0 write shapes, 20/0 erase shapes (by convention; the properties
are authoritative, layers are for human eyes in any CAD viewer).
"""

from __future__ import annotations

from pathlib import Path

import klayout.db as db

from .compiler import LithoObject, Pattern

DBU = 0.001  # um per database unit


def pattern_to_oasis(pattern: Pattern, path: str | Path) -> None:
    layout = db.Layout()
    layout.dbu = DBU
    cell = layout.create_cell("DESIGN")
    l_write = layout.layer(10, 0)
    l_erase = layout.layer(20, 0)

    for order, obj in enumerate(pattern.objects):
        pts = [db.DPoint(x, y) for x, y in obj.points_um]
        if obj.kind == "rect":
            shape = db.DBox(pts[0], pts[1])
        else:
            shape = db.DPath(pts, 0.05)
        layer = l_write if obj.voltage >= 0 else l_erase
        s = cell.shapes(layer).insert(shape)
        s.set_property("litho.order", str(order))
        s.set_property("litho.name", obj.name)
        s.set_property("litho.kind", obj.kind)
        s.set_property("litho.voltage", str(obj.voltage))
        if obj.voltage_end is not None:
            s.set_property("litho.voltage_end", str(obj.voltage_end))
        s.set_property("litho.speed_um_s", str(obj.speed_um_s))
    layout.write(str(path))


def oasis_to_pattern(path: str | Path) -> Pattern:
    layout = db.Layout()
    layout.read(str(path))
    cell = layout.top_cell()
    objects: list[LithoObject] = []
    orders: list[int] = []
    for li in layout.layer_indexes():
        for s in cell.shapes(li).each():
            name = s.property("litho.name") or f"shape{len(objects)}"
            kind = s.property("litho.kind") or ("path" if s.is_path() else "rect")
            voltage = float(s.property("litho.voltage") or 8.0)
            v_end = s.property("litho.voltage_end")
            speed = float(s.property("litho.speed_um_s") or 2.0)
            if s.is_path():
                p = s.dpath
                pts = [(pt.x, pt.y) for pt in p.each_point()]
            elif s.is_box():
                b = s.dbox
                pts = [(b.left, b.bottom), (b.right, b.top)]
            else:
                continue
            objects.append(LithoObject(
                name=str(name), kind=str(kind), points_um=pts,
                voltage=voltage,
                voltage_end=float(v_end) if v_end is not None else None,
                speed_um_s=speed,
            ))
            orders.append(int(s.property("litho.order") or len(orders)))
    objects = [o for _, o in sorted(zip(orders, objects), key=lambda t: t[0])]
    return Pattern(name=Path(path).stem, objects=objects)
