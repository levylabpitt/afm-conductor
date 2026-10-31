"""The Connect tab's per-sample electrode-PORT store and its CRUD endpoints.

A port is a marked electrode: a point + the direction the connecting wire comes
in from + a pad width + a label. Two things must hold for the later phases that
couple design wires to ports by id:

  * an id is STABLE and NEVER reused -- deleting a port must not free its id for
    the next add, or a wire would silently re-point at a different electrode;
  * ports are keyed to the SAMPLE and survive a reload -- marking electrodes is
    pointless if they vanish when the page reloads.

Store-level tests exercise `electrodes` directly against a tmp runtime; the HTTP
tests drive the router with the store redirected at a tmp dir so the suite never
touches the real runtime.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from afm_litho.service import electrodes


# --------------------------------------------------------------- store layer --

def test_add_list_update_delete_roundtrip(tmp_path: Path):
    ports, seq = electrodes.load_store(tmp_path, "S")
    assert ports == [] and seq == 0                    # nothing yet

    pid, seq = electrodes.next_port_id(seq)
    ports.append({"id": pid, "label": "1", "position_um": [1.0, 2.0],
                  "direction_deg": 0.0, "width_um": 1.0})
    electrodes.save_ports(tmp_path, "S", ports, seq)

    got = electrodes.load_ports(tmp_path, "S")
    assert len(got) == 1
    assert got[0]["id"] == pid
    assert got[0]["position_um"] == [1.0, 2.0]

    # update
    got[0]["width_um"] = 3.5
    electrodes.save_ports(tmp_path, "S", got, seq)
    assert electrodes.load_ports(tmp_path, "S")[0]["width_um"] == 3.5

    # delete
    electrodes.save_ports(tmp_path, "S", [], seq)      # seq NOT decremented
    assert electrodes.load_ports(tmp_path, "S") == []


def test_id_is_never_reused_across_deletes(tmp_path: Path):
    """add -> delete -> add: the second id must differ from the first."""
    ports, seq = electrodes.load_store(tmp_path, "S")
    id1, seq = electrodes.next_port_id(seq)
    electrodes.save_ports(tmp_path, "S",
                          [{"id": id1, "label": "1", "position_um": [0, 0],
                            "direction_deg": 0.0, "width_um": 1.0}], seq)

    # delete everything, but persist the (undecremented) seq
    _ports, seq_after = electrodes.load_store(tmp_path, "S")
    electrodes.save_ports(tmp_path, "S", [], seq_after)

    ports, seq = electrodes.load_store(tmp_path, "S")
    id2, seq = electrodes.next_port_id(seq)
    assert id2 != id1, "an id was reused after a delete"


def test_persistence_survives_reload(tmp_path: Path):
    electrodes.save_ports(
        tmp_path, "S",
        [{"id": "e0", "label": "1", "position_um": [4.0, 5.0],
          "direction_deg": 90.0, "width_um": 2.0}], 1)
    ports, seq = electrodes.load_store(tmp_path, "S")
    assert seq == 1
    assert ports == [{"id": "e0", "label": "1", "position_um": [4.0, 5.0],
                      "direction_deg": 90.0, "width_um": 2.0}]


def test_missing_file_reads_as_empty(tmp_path: Path):
    assert electrodes.load_ports(tmp_path, "never-saved") == []
    assert electrodes.load_store(tmp_path, "never-saved") == ([], 0)


def test_corrupt_and_empty_file_do_not_raise(tmp_path: Path):
    d = tmp_path / "electrodes"
    d.mkdir()
    (d / "S.json").write_text("", encoding="utf-8")           # empty
    assert electrodes.load_ports(tmp_path, "S") == []
    (d / "S.json").write_text("{ this is not json", encoding="utf-8")
    assert electrodes.load_ports(tmp_path, "S") == []
    (d / "S.json").write_text("[]", encoding="utf-8")         # wrong top type
    assert electrodes.load_ports(tmp_path, "S") == []


def test_seq_floors_above_any_existing_id(tmp_path: Path):
    """A file whose seq is behind its ids can never hand out a colliding id."""
    d = tmp_path / "electrodes"
    d.mkdir()
    (d / "S.json").write_text(
        '{"seq": 0, "ports": [{"id": "e7", "label": "1", '
        '"position_um": [0, 0], "direction_deg": 0, "width_um": 1}]}',
        encoding="utf-8")
    _ports, seq = electrodes.load_store(tmp_path, "S")
    new_id, _ = electrodes.next_port_id(seq)
    assert new_id == "e8"


def test_two_samples_keep_separate_lists(tmp_path: Path):
    electrodes.save_ports(tmp_path, "A",
                          [{"id": "e0", "label": "1", "position_um": [1, 1],
                            "direction_deg": 0.0, "width_um": 1.0}], 1)
    electrodes.save_ports(tmp_path, "B",
                          [{"id": "e0", "label": "1", "position_um": [2, 2],
                            "direction_deg": 0.0, "width_um": 1.0},
                           {"id": "e1", "label": "2", "position_um": [3, 3],
                            "direction_deg": 0.0, "width_um": 1.0}], 2)
    assert len(electrodes.load_ports(tmp_path, "A")) == 1
    assert len(electrodes.load_ports(tmp_path, "B")) == 2
    assert electrodes.load_ports(tmp_path, "A")[0]["position_um"] == [1.0, 1.0]


def test_bad_port_is_skipped_on_load_not_raised(tmp_path: Path):
    d = tmp_path / "electrodes"
    d.mkdir()
    (d / "S.json").write_text(
        '{"seq": 2, "ports": [{"id": "e0"}, {"id": "e1", "label": "x", '
        '"position_um": [1, 2], "direction_deg": 0, "width_um": 1}]}',
        encoding="utf-8")
    ports = electrodes.load_ports(tmp_path, "S")
    assert len(ports) == 1 and ports[0]["id"] == "e1"


# ---------------------------------------------------------------- HTTP layer --

@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A TestClient whose connect router persists into a tmp runtime dir."""
    from afm_litho.service.routers import connect as connectmod

    monkeypatch.setattr(connectmod, "RUNTIME", tmp_path)
    from afm_litho.service.app import app
    with TestClient(app) as c:
        yield c


def _add(c, **body):
    return c.post("/api/connect/ports/add", json=body).json()


def test_http_add_list_update_delete(client):
    r = _add(client, position_um=[1.0, 2.0])
    assert "id" in r and len(r["ports"]) >= 1
    pid = r["id"]
    port = next(p for p in r["ports"] if p["id"] == pid)
    assert port["direction_deg"] == 0.0            # default
    assert port["width_um"] == 1.0                 # default
    assert port["label"]                           # auto-labelled

    listed = client.get("/api/connect/ports").json()
    assert listed["sample"]
    assert any(p["id"] == pid for p in listed["ports"])

    up = client.post("/api/connect/ports/update",
                     json={"id": pid, "width_um": 4.0, "label": "gate"}).json()
    patched = next(p for p in up["ports"] if p["id"] == pid)
    assert patched["width_um"] == 4.0 and patched["label"] == "gate"

    dl = client.post("/api/connect/ports/delete", json={"id": pid}).json()
    assert all(p["id"] != pid for p in dl["ports"])


def test_http_id_never_reused_after_delete(client):
    id1 = _add(client, position_um=[0, 0])["id"]
    client.post("/api/connect/ports/delete", json={"id": id1})
    id2 = _add(client, position_um=[0, 0])["id"]
    assert id2 != id1


def test_http_persists_and_reloads(client):
    id1 = _add(client, position_um=[7.0, 8.0], direction_deg=45.0,
               width_um=2.5, label="src")["id"]
    # a fresh GET reads from disk, not memory
    listed = client.get("/api/connect/ports").json()["ports"]
    p = next(p for p in listed if p["id"] == id1)
    assert p["position_um"] == [7.0, 8.0]
    assert p["direction_deg"] == 45.0
    assert p["width_um"] == 2.5
    assert p["label"] == "src"


def test_http_bad_input_returns_error_not_500(client):
    r = client.post("/api/connect/ports/add", json={"position_um": [1]})
    assert r.status_code == 200 and "error" in r.json()

    r = client.post("/api/connect/ports/add", json={"position_um": "nope"})
    assert r.status_code == 200 and "error" in r.json()

    r = client.post("/api/connect/ports/update", json={"id": "e999"})
    assert r.status_code == 200 and "error" in r.json()

    r = client.post("/api/connect/ports/delete", json={"id": "e999"})
    assert r.status_code == 200 and "error" in r.json()


def test_http_per_sample_isolation(client):
    from afm_litho.service.hub import hub

    prev = hub.sample_name
    try:
        hub.sample_name = "sampleA"
        a_id = _add(client, position_um=[1, 1])["id"]

        hub.sample_name = "sampleB"
        assert client.get("/api/connect/ports").json()["ports"] == []
        _add(client, position_um=[2, 2])            # B's own list

        # A still has exactly its one port at its own position -- B's add did
        # not leak in. (ids are unique PER sample, so both samples' first port
        # is e0; isolation is that the lists and positions stay separate.)
        hub.sample_name = "sampleA"
        a_ports = client.get("/api/connect/ports").json()["ports"]
        assert [p["id"] for p in a_ports] == [a_id]
        assert a_ports[0]["position_um"] == [1.0, 1.0]
    finally:
        hub.sample_name = prev


def test_http_replace_bulk_sets(client):
    _add(client, position_um=[1, 1])
    r = client.post("/api/connect/ports/replace", json={"ports": [
        {"id": "e0", "label": "1", "position_um": [9, 9],
         "direction_deg": 0.0, "width_um": 1.0}]}).json()
    assert len(r["ports"]) == 1 and r["ports"][0]["position_um"] == [9.0, 9.0]
    # a subsequent add must not collide with the replaced e0
    new_id = _add(client, position_um=[0, 0])["id"]
    assert new_id != "e0"
