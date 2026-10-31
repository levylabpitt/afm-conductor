"""The Connect tab: the sample's electrode PORTS (Phase 1).

The operator marks each existing electrode on the scan as a connection PORT --
a point (`position_um`) the connecting wire targets, a `direction_deg` the wire
comes in from (its tangent at the pad, so no sharp turn), a pad `width_um`, and
an editable `label`. Ports are keyed to `hub.sample_name` and persisted per
sample by `electrodes`, so they survive a reload -- that is the whole point.

Ids are STABLE and NEVER reused (a later phase couples a design wire to a port
by id so the wire follows the electrode through drift). The store's persisted
`seq` counter guarantees that: it only ever increases, even across deletes.

Every handler validates its input and returns `{"error": "..."}` -- never a 500
-- on bad input, mirroring the design router. The list of ports is loaded from
disk on every request (keyed to the current sample) and mirrored onto `hub.ports`
so a sample switch cannot serve the previous sample's ports.
"""

from __future__ import annotations

from fastapi import APIRouter

from .. import electrodes
from ..deps import RUNTIME
from ..hub import hub

router = APIRouter()


def _load() -> tuple[list[dict], int]:
    """The current sample's (ports, seq) from disk, mirrored onto the hub."""
    ports, seq = electrodes.load_store(RUNTIME, hub.sample_name)
    hub.ports = ports
    hub.ports_seq = seq
    return ports, seq


def _save(ports: list[dict], seq: int) -> None:
    electrodes.save_ports(RUNTIME, hub.sample_name, ports, seq)
    hub.ports = ports
    hub.ports_seq = seq


def _default_label(ports: list[dict], seq: int) -> str:
    """Next integer label as a string: max existing integer label + 1.

    Falls back to the seq counter when no existing label is a plain integer, so
    a fresh sample's first port is labelled from the id space rather than "1"
    colliding with a hand-typed label later.
    """
    best = 0
    seen = False
    for p in ports:
        lab = str(p.get("label", "")).strip()
        if lab.isdigit():
            best = max(best, int(lab))
            seen = True
    return str(best + 1) if seen else str(seq + 1)


async def connect_ports():
    """The current sample's ports (keyed to hub.sample_name)."""
    ports, _seq = _load()
    return {"ports": ports, "sample": hub.sample_name}


async def connect_ports_add(body: dict):
    """Drop a new port at a scan-µm point.

    Assigns a fresh never-reused id (`e{seq}`), a default label (next integer),
    default `direction_deg` = 0.0 (the frontend computes toward-centre; the
    backend only needs a valid default) and `width_um` = 1.0. Appends, saves,
    returns the new list and the new id.
    """
    if not isinstance(body, dict):
        return {"error": "expected a JSON object"}
    pos = body.get("position_um")
    if not isinstance(pos, (list, tuple)) or len(pos) != 2:
        return {"error": "position_um must be [x, y] microns"}
    try:
        x, y = float(pos[0]), float(pos[1])
    except (TypeError, ValueError):
        return {"error": "position_um must be two numbers"}
    try:
        direction = float(body.get("direction_deg", 0.0) or 0.0)
    except (TypeError, ValueError):
        return {"error": "direction_deg must be a number"}
    try:
        width = float(body.get("width_um", 1.0))
    except (TypeError, ValueError):
        return {"error": "width_um must be a number"}

    ports, seq = _load()
    pid, new_seq = electrodes.next_port_id(seq)
    label = body.get("label")
    label = _default_label(ports, seq) if label is None else str(label)
    port = {
        "id": pid,
        "label": label,
        "position_um": [x, y],
        "direction_deg": direction,
        "width_um": width,
    }
    ports = ports + [port]
    _save(ports, new_seq)
    await hub.log(f"Electrode port {label} added at "
                  f"({x:.2f}, {y:.2f}) um")
    return {"ports": ports, "id": pid}


async def connect_ports_update(body: dict):
    """Patch only the provided fields on one port. Unknown id -> {error}."""
    if not isinstance(body, dict):
        return {"error": "expected a JSON object"}
    pid = body.get("id")
    if not pid:
        return {"error": "id is required"}
    pid = str(pid)
    ports, seq = _load()
    target = next((p for p in ports if p["id"] == pid), None)
    if target is None:
        return {"error": f"no such port: {pid}"}

    if "label" in body and body["label"] is not None:
        target["label"] = str(body["label"])
    if "position_um" in body and body["position_um"] is not None:
        pos = body["position_um"]
        if not isinstance(pos, (list, tuple)) or len(pos) != 2:
            return {"error": "position_um must be [x, y] microns"}
        try:
            target["position_um"] = [float(pos[0]), float(pos[1])]
        except (TypeError, ValueError):
            return {"error": "position_um must be two numbers"}
    if "direction_deg" in body and body["direction_deg"] is not None:
        try:
            target["direction_deg"] = float(body["direction_deg"])
        except (TypeError, ValueError):
            return {"error": "direction_deg must be a number"}
    if "width_um" in body and body["width_um"] is not None:
        try:
            target["width_um"] = float(body["width_um"])
        except (TypeError, ValueError):
            return {"error": "width_um must be a number"}

    _save(ports, seq)
    return {"ports": ports}


async def connect_ports_delete(body: dict):
    """Remove one port. The seq counter is NOT decremented (ids stay unique)."""
    if not isinstance(body, dict):
        return {"error": "expected a JSON object"}
    pid = body.get("id")
    if not pid:
        return {"error": "id is required"}
    pid = str(pid)
    ports, seq = _load()
    kept = [p for p in ports if p["id"] != pid]
    if len(kept) == len(ports):
        return {"error": f"no such port: {pid}"}
    _save(kept, seq)                       # seq unchanged: an id is never reused
    await hub.log(f"Electrode port {pid} deleted")
    return {"ports": kept}


async def connect_ports_replace(body: dict):
    """Bulk-set the port list (a future import). Validates every port; keeps the
    seq counter monotonic so ids stay unique against everything ever handed out.
    """
    if not isinstance(body, dict):
        return {"error": "expected a JSON object"}
    raw = body.get("ports")
    if not isinstance(raw, list):
        return {"error": "ports must be a list"}
    _ports, seq = _load()
    clean: list[dict] = []
    for r in raw:
        try:
            clean.append(electrodes.coerce_port(r))
        except (ValueError, TypeError) as exc:
            return {"error": f"bad port: {exc}"}
    # keep the counter ahead of any incoming e<N> id, so a later add cannot collide
    for p in clean:
        pid = p["id"]
        if pid.startswith("e") and pid[1:].isdigit():
            seq = max(seq, int(pid[1:]) + 1)
    _save(clean, seq)
    return {"ports": clean}
