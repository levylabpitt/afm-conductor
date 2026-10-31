"""The autosave brings the design back after the service is killed.

The kill and restart tests run a child service on a private runtime; the rest
call the restore directly against a temp runtime.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from _service import Service
from afm_litho.litho.compiler import LithoObject, Pattern
from afm_litho.service import design_store


def _call(service, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(service.url + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _objects_on_disk(runtime, name):
    path = runtime / "designs" / f"{name}.design.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))["objects"]
    except (OSError, ValueError, KeyError):
        return None


def _build_a_design(service):
    assert _call(service, "/api/design/add", {
        "kind": "path", "name": "wire", "voltage": 6.5, "speed_um_s": 0.8,
        "points_um": [[-2, 0], [0, 0.4], [2, 0]]})["ok"]
    assert _call(service, "/api/design/add", {
        "kind": "path", "name": "curve", "bezier": {"closed": False, "nodes": [
            {"x": -1, "y": 1, "h_in": None, "h_out": [-0.5, 1.6], "kind": "smooth"},
            {"x": 1, "y": 1, "h_in": [0.5, 1.6], "h_out": None, "kind": "smooth"}]}})["ok"]
    assert _call(service, "/api/design/add", {
        "kind": "path", "name": "helix", "points_um": [[-2, -1], [2, -1]]})["ok"]
    assert _call(service, "/api/design/chiral", {
        "indices": [2], "enabled": True, "lambda_um": 0.05, "y_amp_um": 0.05,
        "v_k": 1.0, "phase_deg": 90, "handedness": "L"})["ok"]
    assert _call(service, "/api/design/add", {
        "kind": "shape", "name": "pad", "fill_spacing_um": 0.2,
        "points_um": [[3, 3], [4, 3], [4, 4], [3, 4], [3, 3]]})["ok"]
    assert _call(service, "/api/design/op", {
        "op": "set_props", "indices": [3], "voltage": 9.0, "voltage_end": 7.0,
        "speed_um_s": 1.25, "enabled": False}).get("ok")


@pytest.fixture
def autosave_on(monkeypatch):
    monkeypatch.setenv("AFM_LITHO_DESIGN_AUTOSAVE", "1")


def test_a_killed_service_comes_back_with_the_same_design(tmp_path, autosave_on):
    first = Service(runtime=tmp_path).start()
    try:
        _build_a_design(first)
        assert _call(first, "/api/design/save", {"name": "before"})["ok"]
        before = _objects_on_disk(tmp_path, "before")
        assert [o["name"] for o in before] == ["wire", "curve", "helix", "pad"]
        deadline = time.time() + 20
        while _objects_on_disk(tmp_path, "_autosave") != before:
            assert time.time() < deadline, "the autosave never caught up"
            time.sleep(0.1)
        first.proc.kill()
        first.proc.wait(timeout=10)
    finally:
        first.stop()

    second = Service(runtime=tmp_path).start()
    try:
        shown = _call(second, "/api/design")
        assert [o["name"] for o in shown["objects"]] == ["wire", "curve", "helix", "pad"]
        assert _call(second, "/api/design/save", {"name": "after"})["ok"]
        assert _objects_on_disk(tmp_path, "after") == before
        state = _call(second, "/api/state")
        assert state["design_autosave"]["ok"] is True
    finally:
        second.stop()


def test_a_corrupt_autosave_starts_empty_and_is_set_aside(tmp_path, autosave_on):
    folder = tmp_path / "designs"
    folder.mkdir()
    (folder / "_autosave.design.json").write_text(
        '{"format": "afm-litho-design", "version": 1, "objects": [{"na',
        encoding="utf-8")
    with Service(runtime=tmp_path) as service:
        assert _call(service, "/api/design") == {"name": None, "objects": []}
        assert (folder / "_autosave.bad.json").is_file()
        assert not (folder / "_autosave.design.json").exists()
        assert _call(service, "/api/design/add", {
            "kind": "path", "points_um": [[0, 0], [1, 1]]})["ok"]


# ------------------------------------------------- the restore, in process --

@pytest.fixture
def hub(client, tmp_path, autosave_on):
    from afm_litho.service.app import hub
    from afm_litho.service.sessions import SessionStore
    saved = (hub.store, hub.pattern, hub.host.written_objects)
    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    hub.pattern = None
    hub.host.written_objects = set()
    yield hub
    design_store.flush_autosave()
    design_store._autosaver.hold = False
    hub.store, hub.pattern, hub.host.written_objects = saved


def _autosave_path(tmp_path):
    return tmp_path / "designs" / "_autosave.design.json"


def _autosaved_names(tmp_path):
    doc = json.loads(_autosave_path(tmp_path).read_text(encoding="utf-8"))
    return [o["name"] for o in doc["objects"]]


def test_a_setting_an_open_refuses_is_restored_as_saved_with_a_warning(
        hub, tmp_path, monkeypatch, caplog):
    from afm_litho.litho import compiler
    from afm_litho.service.host import litho_cmds

    def never(*a, **kw):
        raise AssertionError("the restore must not compile")
    for module, name in ((litho_cmds, "_compile_report"), (compiler, "compile_pattern"),
                         (compiler, "compile_object"), (compiler, "expand")):
        monkeypatch.setattr(module, name, never)
    pad = LithoObject("pad", "shape", [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)],
                      fill_spacing_um=0.0)
    slow = LithoObject("slow", "path", [(0.0, 0.0), (1.0, 0.0)], speed_um_s=-1.0,
                       interpolation_um=0.0)
    _autosave_of(hub, [pad, slow])
    with caplog.at_level("WARNING"):
        assert "2 objects restored" in design_store.restore_autosave(hub)
    assert hub.pattern.objects == [pad, slow]
    said = " | ".join(r.getMessage() for r in caplog.records)
    for part in ("'pad': fill.spacing_um is 0", "'slow': speed_um_s is -1",
                 "'slow': interpolation_um is 0"):
        assert part in said


def test_an_unfinished_file_from_the_last_run_is_removed_at_start(hub, tmp_path):
    folder = tmp_path / "designs"
    folder.mkdir()
    for name in ("_autosave.design.json.tmp", "dev.design.json.tmp"):
        (folder / name).write_text("{", encoding="utf-8")
    (folder / "notes.tmp").write_text("mine", encoding="utf-8")
    assert design_store.restore_autosave(hub) is None
    assert sorted(p.name for p in folder.iterdir()) == ["notes.tmp"]


def _locked(monkeypatch, path):
    real = Path.read_bytes

    def read_bytes(self):
        if self == path:
            raise PermissionError(13, "The process cannot access the file")
        return real(self)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)


def test_an_autosave_that_cannot_be_read_at_start_is_not_written_over(
        hub, tmp_path, monkeypatch):
    _autosave_of(hub, [_wire("good")])
    with monkeypatch.context() as patch:
        _locked(patch, _autosave_path(tmp_path))
        note = design_store.restore_autosave(hub)
    assert "could not be read" in note and "left in place" in note
    assert "set aside" not in note
    assert hub.pattern is None
    assert not (tmp_path / "designs" / "_autosave.bad.json").exists()

    hub.pattern = Pattern("sketch", [])
    design_store.request_autosave(hub)
    hub.pattern = None
    design_store.request_autosave(hub)
    assert design_store.flush_autosave()
    assert _autosaved_names(tmp_path) == ["good"]

    unread = tmp_path / "designs" / "_autosave.unread.json"
    assert not unread.exists() and design_store._autosaver.hold

    monkeypatch.setattr(design_store, "_RETRY_FIRST_S", 30.0)
    hub.pattern = Pattern("sketch", [_wire("drawn")])
    with monkeypatch.context() as patch:
        def full(*a):
            raise OSError(28, "No space left on device")
        patch.setattr(design_store, "write_document", full)
        design_store.request_autosave(hub)
        assert design_store.flush_autosave()
        assert design_store.autosave_status()["ok"] is False
        assert design_store._autosaver.hold
        assert json.loads(unread.read_text(encoding="utf-8"))["objects"][0]["name"] == "good"
    assert design_store.flush_autosave(retry_now=True)
    assert design_store.autosave_status()["ok"] is True
    assert not design_store._autosaver.hold
    assert _autosaved_names(tmp_path) == ["drawn"]
    assert json.loads(unread.read_text(encoding="utf-8"))["objects"][0]["name"] == "good"
    hub.pattern = Pattern("sketch", [])
    design_store.request_autosave(hub)
    assert design_store.flush_autosave()
    assert _autosaved_names(tmp_path) == []


def test_the_note_is_true_when_a_bad_autosave_cannot_be_set_aside(hub, tmp_path,
                                                                  monkeypatch):
    folder = tmp_path / "designs"
    folder.mkdir()
    _autosave_path(tmp_path).write_text("not json", encoding="utf-8")

    def locked(src, dst):
        raise PermissionError(13, "in use")
    monkeypatch.setattr(design_store.os, "replace", locked)
    note = design_store.restore_autosave(hub)
    assert "left in place" in note and "set aside" not in note
    assert _autosave_path(tmp_path).read_text(encoding="utf-8") == "not json"
    assert hub.pattern is None


def test_a_failed_autosave_is_tried_again_without_another_edit(hub, tmp_path,
                                                              monkeypatch):
    monkeypatch.setattr(design_store, "_RETRY_FIRST_S", 0.05)
    real = design_store.os.replace
    tries = []

    def full_twice(src, dst):
        tries.append(time.monotonic())
        if len(tries) <= 2:
            raise OSError(28, "No space left on device")
        return real(src, dst)
    monkeypatch.setattr(design_store.os, "replace", full_twice)
    hub.pattern = Pattern("sketch", [_wire("a")])
    design_store.request_autosave(hub)
    assert design_store.flush_autosave()
    assert design_store.autosave_status()["ok"] is False
    deadline = time.time() + 5
    while not design_store.autosave_status()["ok"] and time.time() < deadline:
        time.sleep(0.02)
    assert design_store.autosave_status()["ok"] is True
    assert len(tries) == 3 and _autosaved_names(tmp_path) == ["a"]
    assert tries[1] - tries[0] >= 0.045 and tries[2] - tries[1] >= 0.09
    time.sleep(0.3)
    assert len(tries) == 3


def test_the_retry_delay_stays_finite_however_long_the_failure_lasts(monkeypatch):
    saver = design_store.Autosaver()
    delays = []
    for failures in (1, 2, 3, 6, 11, 1024, 1025, 5000, 10 ** 6):
        saver._failures = failures
        delays.append(saver._retry_delay())
    assert delays[:4] == [2.0, 4.0, 8.0, 60.0]
    assert all(d == 60.0 for d in delays[3:])


def test_the_retry_worker_outlives_a_very_long_failure(hub, tmp_path, monkeypatch):
    monkeypatch.setattr(design_store, "_RETRY_FIRST_S", 0.01)
    monkeypatch.setattr(design_store, "_RETRY_MAX_S", 0.02)
    real = design_store.write_document
    tries = []

    def full(*a):
        tries.append(1)
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(design_store, "write_document", full)
    hub.pattern = Pattern("sketch", [_wire("a")])
    design_store.request_autosave(hub)
    assert design_store.flush_autosave()
    design_store._autosaver._failures = 2000
    count = len(tries)
    deadline = time.time() + 5
    while len(tries) < count + 3 and time.time() < deadline:
        time.sleep(0.01)
    assert len(tries) >= count + 3 and design_store._autosaver._thread.is_alive()
    monkeypatch.setattr(design_store, "write_document", real)
    assert design_store.flush_autosave(retry_now=True)
    assert design_store.autosave_status()["ok"] is True
    assert _autosaved_names(tmp_path) == ["a"]


def test_a_clean_stop_makes_a_last_try_for_a_write_waiting_on_its_retry(
        hub, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from afm_litho.service.app import app
    monkeypatch.setattr(design_store, "_RETRY_FIRST_S", 30.0)
    real = design_store.write_document
    with TestClient(app) as second:
        def full(*a):
            raise OSError(28, "No space left on device")
        monkeypatch.setattr(design_store, "write_document", full)
        hub.pattern = None
        assert second.post("/api/design/add", json={
            "kind": "path", "name": "last", "points_um": [[0, 0], [1, 1]]}).json()["ok"]
        assert design_store.flush_autosave()
        assert design_store.autosave_status()["ok"] is False
        monkeypatch.setattr(design_store, "write_document", real)
    assert _autosaved_names(tmp_path) == ["last"]


def test_a_clean_stop_waits_for_the_autosave(hub, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from afm_litho.service.app import app
    real = design_store.write_document

    def slow(runtime_dir, name, doc):
        time.sleep(0.4)
        return real(runtime_dir, name, doc)
    monkeypatch.setattr(design_store, "write_document", slow)
    with TestClient(app) as second:
        hub.pattern = None
        assert second.post("/api/design/add", json={
            "kind": "path", "name": "last", "points_um": [[0, 0], [1, 1]]}).json()["ok"]
        assert not _autosave_path(tmp_path).exists()
    assert _autosaved_names(tmp_path) == ["last"]


def _autosave_of(hub, objects):
    hub.pattern = Pattern("drawn", objects)
    design_store.request_autosave(hub)
    assert design_store.flush_autosave()
    hub.pattern = None


def _wire(name, voltage=8.0):
    return LithoObject(name, "path", [(0.0, 0.0), (1.0, 0.5)], voltage=voltage)


def test_restore_fills_an_empty_design(hub):
    saved = [_wire("a"), _wire("b")]
    _autosave_of(hub, saved)
    hub.host.written_objects = {1234}
    note = design_store.restore_autosave(hub)
    assert "2 objects restored" in note
    assert hub.pattern.objects == saved and hub.pattern.name == "drawn"
    assert hub.host.written_objects == set()


def test_restore_is_skipped_when_a_design_exists(hub):
    _autosave_of(hub, [_wire("a"), _wire("b")])
    mine = Pattern("mine", [_wire("keep")])
    hub.pattern = mine
    assert design_store.restore_autosave(hub) is None
    assert hub.pattern is mine and [o.name for o in mine.objects] == ["keep"]


def test_restore_fills_a_design_that_exists_but_holds_nothing(hub):
    _autosave_of(hub, [_wire("a")])
    hub.pattern = Pattern("sketch", [])
    assert design_store.restore_autosave(hub)
    assert [o.name for o in hub.pattern.objects] == ["a"]


def test_restore_keeps_a_voltage_outside_the_window_and_warns(hub, caplog):
    top = float(hub.host.instrument_model.tip_voltage_window_v.value[1])
    _autosave_of(hub, [_wire("hot", voltage=top + 50.0)])
    with caplog.at_level("WARNING"):
        assert design_store.restore_autosave(hub)
    assert hub.pattern.objects[0].voltage == top + 50.0
    assert any("hot" in r.getMessage() for r in caplog.records)


def test_a_restore_leaves_out_what_it_cannot_build_and_keeps_the_file(
        hub, tmp_path, monkeypatch, caplog):
    from afm_litho.litho import pens
    records = [
        {"name": "good", "kind": "path", "points_um": [[0, 0], [1, 1]]},
        {"name": "no-kind"},
        {"name": "huge", "kind": "path", "chiral": {
            "lambda_um": 0.0097, "y_amp_um": 0, "v_k": 1.0, "phase_deg": 0.0,
            "axis_um": [[-1000, 0], [1000, 0]]}},
        {"name": "far", "kind": "path", "bezier": {"closed": False, "nodes": [
            {"x": 0.0, "y": 0.0, "h_in": None, "h_out": [1e9, 1e9]},
            {"x": 1.0, "y": 0.0, "h_in": [-1e9, 5.0], "h_out": None}]}},
        {"name": "curve", "kind": "path", "bezier": {"closed": False, "nodes": [
            {"x": 0.0, "y": 0.0, "h_in": None, "h_out": [0.5, 0.4]},
            {"x": 2.0, "y": 0.0, "h_in": [1.5, 0.4], "h_out": None}]}}]
    text = json.dumps({"format": "afm-litho-design", "version": 1, "objects": records})
    folder = tmp_path / "designs"
    folder.mkdir()
    _autosave_path(tmp_path).write_text(text, encoding="utf-8")
    monkeypatch.setattr(pens, "chiral_profile", None)
    began = time.monotonic()
    with caplog.at_level("WARNING"):
        note = design_store.restore_autosave(hub)
    assert time.monotonic() - began < 1.0
    assert [o.name for o in hub.pattern.objects] == ["good", "curve"]
    assert "2 objects restored" in note and "3 objects could not be rebuilt" in note
    assert "_autosave.bad.json" in note
    said = " | ".join(r.getMessage() for r in caplog.records)
    for name in ("'no-kind'", "'huge'", "'far'"):
        assert name in said
    assert (folder / "_autosave.bad.json").read_text(encoding="utf-8") == text
    assert _autosave_path(tmp_path).read_text(encoding="utf-8") == text

    hub.pattern.objects.append(_wire("drawn"))
    design_store.request_autosave(hub)
    assert design_store.flush_autosave()
    assert _autosaved_names(tmp_path) == ["good", "curve", "drawn"]
    assert (folder / "_autosave.bad.json").read_text(encoding="utf-8") == text


def test_an_autosave_with_nothing_that_can_be_built_starts_empty_and_is_kept(
        hub, tmp_path):
    text = '{"format": "afm-litho-design", "version": 1, "objects": [{"name": "a"}]}'
    (tmp_path / "designs").mkdir()
    _autosave_path(tmp_path).write_text(text, encoding="utf-8")
    note = design_store.restore_autosave(hub)
    assert "nothing restored" in note and "1 object could not be rebuilt" in note
    assert hub.pattern is None
    assert (tmp_path / "designs" / "_autosave.bad.json").read_text(
        encoding="utf-8") == text


@pytest.mark.parametrize("text", [
    "", "not json", '{"format": "afm-litho-design", "version": 9, "objects": []}',
    '{"format": "afm-litho-design", "version": 1, "builder": 9, "objects": []}'])
def test_an_unreadable_autosave_never_raises(hub, tmp_path, text):
    folder = tmp_path / "designs"
    folder.mkdir()
    (folder / "_autosave.design.json").write_text(text, encoding="utf-8")
    (folder / "_autosave.bad.json").write_text("older", encoding="utf-8")
    note = design_store.restore_autosave(hub)
    assert "set aside" in note
    assert hub.pattern is None
    assert (folder / "_autosave.bad.json").read_text(encoding="utf-8") == text
    assert not (folder / "_autosave.design.json").exists()


def test_an_emptied_design_is_mirrored_and_restores_nothing(hub, tmp_path):
    _autosave_of(hub, [_wire("a")])
    hub.pattern = Pattern("sketch", [])
    design_store.request_autosave(hub)
    assert design_store.flush_autosave()
    doc = json.loads((tmp_path / "designs" / "_autosave.design.json")
                     .read_text(encoding="utf-8"))
    assert doc["objects"] == [] and doc["object_count"] == 0
    hub.pattern = None
    assert design_store.restore_autosave(hub) is None
    assert hub.pattern is None


def test_requests_are_coalesced_and_the_last_design_wins(hub, tmp_path, monkeypatch):
    import threading
    gate = threading.Event()
    written = []
    real = design_store.write_document

    def slow(runtime_dir, name, doc):
        written.append([o["name"] for o in doc["objects"]])
        gate.wait(10)
        return real(runtime_dir, name, doc)
    monkeypatch.setattr(design_store, "write_document", slow)
    hub.pattern = Pattern("sketch", [])
    for i in range(6):
        hub.pattern.objects.append(_wire(f"w{i}"))
        design_store.request_autosave(hub)
        if i == 0:
            deadline = time.time() + 5
            while not written and time.time() < deadline:
                time.sleep(0.01)
    gate.set()
    assert design_store.flush_autosave()
    assert written == [["w0"], [f"w{i}" for i in range(6)]]
    doc = json.loads((tmp_path / "designs" / "_autosave.design.json")
                     .read_text(encoding="utf-8"))
    assert [o["name"] for o in doc["objects"]] == [f"w{i}" for i in range(6)]


def test_the_sample_switch_asks_for_an_autosave(client, hub, tmp_path):
    from afm_litho.service.deps import DEFAULT_SAMPLE
    hub.pattern = Pattern("sketch", [_wire("a")])
    assert client.post("/api/sample/load", json={"file": DEFAULT_SAMPLE}).json()["ok"]
    assert design_store.flush_autosave()
    assert hub.pattern is None
    doc = json.loads((tmp_path / "designs" / "_autosave.design.json")
                     .read_text(encoding="utf-8"))
    assert doc["objects"] == []


# ------------------------------------------- every object the service saved comes back --

def _chiral_wires(client, count, length, amp):
    for i in range(count):
        assert client.post("/api/design/add", json={
            "kind": "path", "name": "w",
            "points_um": [[-length / 2, i * 0.5], [length / 2, i * 0.5]]}).json()["ok"]
    assert client.post("/api/design/chiral", json={
        "indices": list(range(count)), "enabled": True, "lambda_um": 0.025,
        "y_amp_um": amp, "v_k": 1.0, "phase_deg": 90, "handedness": "R"}).json()["ok"]


@pytest.mark.parametrize("count, length, amp", [(60, 90.0, 0.0), (300, 5.0, 0.055)],
                         ids=["60 x 90 um, no amplitude", "300 x 5 um, pen default"])
def test_every_chiral_wire_the_service_saved_is_restored(client, hub, tmp_path, count,
                                                         length, amp):
    _chiral_wires(client, count, length, amp)
    assert design_store.flush_autosave(60)
    before = hub.pattern.objects
    assert _autosave_path(tmp_path).stat().st_size < 200_000
    hub.pattern = None
    note = design_store.restore_autosave(hub)
    assert f"{count} objects restored" in note and "left out" not in note
    assert not (tmp_path / "designs" / "_autosave.bad.json").exists()
    assert hub.pattern.objects == before

    if count != 300:
        return
    assert client.post("/api/design/save", json={"name": "wires"}).json()["ok"]
    for mode in ("replace", "add"):
        reply = client.post("/api/design/open", json={"name": "wires", "mode": mode})
        assert reply.json().get("ok"), reply.text
    assert hub.pattern.objects[:count] == before
    assert len(hub.pattern.objects) == 2 * count


def test_the_autosave_is_plain_text_with_only_documented_keys(client, hub, tmp_path):
    from afm_litho.litho import design_file
    _chiral_wires(client, 2, 2.0, 0.055)
    assert client.post("/api/design/add", json={
        "kind": "shape", "name": "pad", "fill_spacing_um": 0.2,
        "points_um": [[3, 3], [4, 3], [4, 4], [3, 4], [3, 3]]}).json()["ok"]
    assert design_store.flush_autosave()
    text = _autosave_path(tmp_path).read_bytes().decode("utf-8")
    doc = json.loads(text)
    assert set(doc) <= set(design_file.DOCUMENT_KEYS)
    for rec in doc["objects"]:
        assert set(rec) <= set(design_file.RECORD_KEYS)
        assert set(rec.get("chiral", {})) <= {"lambda_um", "y_amp_um", "v_k",
                                              "phase_deg", "axis_um"}
        assert set(rec.get("fill", {})) <= set(design_file.FILL_KEYS)
    assert [("points_um" in r) for r in doc["objects"]] == [False, False, True]
    assert set(doc["objects"][0]) == {"name", "kind", "voltage", "speed_um_s",
                                      "enabled", "chiral"}
    assert doc["objects"][2]["fill"] == {"spacing_um": 0.2, "kind": "directional"}
    assert text.count("\n") > 40


# --------------------------------------------------------- what is set aside is kept --

def test_a_partial_restore_that_cannot_keep_a_copy_holds_the_autosave(
        hub, tmp_path, monkeypatch):
    records = [{"name": "good", "kind": "path", "points_um": [[0, 0], [1, 1]]},
               {"name": "no-kind"}]
    text = json.dumps({"format": "afm-litho-design", "version": 1, "objects": records})
    (tmp_path / "designs").mkdir()
    _autosave_path(tmp_path).write_text(text, encoding="utf-8")

    def no_copy(src, dst):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(design_store.shutil, "copyfile", no_copy)
    note = design_store.restore_autosave(hub)
    assert "1 objects restored" in note.replace("object ", "objects ")
    assert "copy of the autosave could not be made" in note
    assert design_store._autosaver.hold
    assert [o.name for o in hub.pattern.objects] == ["good"]

    hub.pattern.objects.append(_wire("drawn"))
    design_store.request_autosave(hub)
    assert design_store.flush_autosave()
    assert _autosaved_names(tmp_path) == ["good", "drawn"]
    unread = tmp_path / "designs" / "_autosave.unread.json"
    assert unread.read_text(encoding="utf-8") == text


def test_an_earlier_set_aside_file_is_kept_under_a_dated_name(hub, tmp_path):
    folder = tmp_path / "designs"
    folder.mkdir()
    bad = folder / "_autosave.bad.json"
    for i in range(6):
        _autosave_path(tmp_path).write_text(f"not json {i}", encoding="utf-8")
        assert "set aside" in design_store.restore_autosave(hub)
        os.utime(bad, (1_700_000_000 + 100 * i, 1_700_000_000 + 100 * i))
    assert bad.read_text(encoding="utf-8") == "not json 5"
    dated = sorted(p for p in folder.iterdir() if p.name != bad.name)
    assert [p.read_text(encoding="utf-8") for p in dated] == [
        "not json 2", "not json 3", "not json 4"]
    assert all(p.name.startswith("_autosave.bad.2023") for p in dated)


def test_the_note_says_a_set_aside_file_exists_and_it_opens_by_name(client, hub,
                                                                    tmp_path):
    folder = tmp_path / "designs"
    folder.mkdir()
    kept = json.dumps({"format": "afm-litho-design", "version": 1, "objects": [
        {"name": "lost", "kind": "path", "points_um": [[0, 0], [1, 1]]}]})
    (folder / "_autosave.bad.json").write_text(kept, encoding="utf-8")
    (folder / "_autosave.unread.json").write_text(kept, encoding="utf-8")
    note = design_store.restore_autosave(hub)
    assert "_autosave.bad" in note and "_autosave.unread" in note and "opens by" in note
    for name in ("_autosave.bad", "_autosave.unread"):
        hub.pattern = None
        reply = client.post("/api/design/open", json={"name": name}).json()
        assert reply.get("ok") and reply["name"] == name, reply
        assert [o.name for o in hub.pattern.objects] == ["lost"]
        refused = client.post("/api/design/save", json={"name": name}).json()
        assert refused["reason"] == "reserved_name"
    listed = client.get("/api/design/files").json()["designs"]
    assert [(e["name"], e["recovered"]) for e in listed] == [
        ("_autosave.bad", True), ("_autosave.unread", True)]


def test_an_autosave_over_the_size_limit_is_still_restored(hub, tmp_path, monkeypatch,
                                                           caplog):
    _autosave_of(hub, [_wire("a")])
    monkeypatch.setattr(design_store, "MAX_FILE_BYTES", 10)
    with caplog.at_level("WARNING"):
        assert "1 object restored" in design_store.restore_autosave(hub)
    assert any("MB" in r.getMessage() for r in caplog.records)


def test_two_set_aside_files_in_the_same_second_are_both_kept(hub, tmp_path):
    folder = tmp_path / "designs"
    folder.mkdir()
    bad = folder / "_autosave.bad.json"
    for i in range(3):
        _autosave_path(tmp_path).write_text(f"not json {i}", encoding="utf-8")
        assert "set aside" in design_store.restore_autosave(hub)
        os.utime(bad, (1_700_000_000, 1_700_000_000))
    kept = sorted(p.read_text(encoding="utf-8") for p in folder.iterdir())
    assert kept == ["not json 0", "not json 1", "not json 2"]


def test_an_autosave_from_a_newer_version_that_cannot_be_set_aside_is_not_written_over(
        hub, tmp_path, monkeypatch):
    text = json.dumps({"format": "afm-litho-design", "version": 9, "objects": [
        {"name": "theirs", "kind": "path", "points_um": [[0, 0], [1, 1]]}]})
    (tmp_path / "designs").mkdir()
    _autosave_path(tmp_path).write_text(text, encoding="utf-8")
    with monkeypatch.context() as patch:
        def locked(src, dst):
            raise PermissionError(13, "in use")
        patch.setattr(design_store.os, "replace", locked)
        note = design_store.restore_autosave(hub)
    assert "left in place" in note and "not written over" in note
    assert design_store._autosaver.hold and hub.pattern is None

    hub.pattern = Pattern("sketch", [])
    design_store.request_autosave(hub)
    assert design_store.flush_autosave()
    assert _autosave_path(tmp_path).read_text(encoding="utf-8") == text

    hub.pattern = Pattern("sketch", [_wire("drawn")])
    design_store.request_autosave(hub)
    assert design_store.flush_autosave()
    assert _autosaved_names(tmp_path) == ["drawn"]
    unread = tmp_path / "designs" / "_autosave.unread.json"
    assert unread.read_text(encoding="utf-8") == text


def test_the_design_is_restored_before_the_flex_provider_starts(hub, tmp_path,
                                                               monkeypatch):
    from fastapi.testclient import TestClient

    from afm_litho.service import flex_provider
    from afm_litho.service.app import app
    _autosave_of(hub, [_wire("a"), _wire("b")])
    seen = []
    monkeypatch.setattr(
        flex_provider, "start_provider",
        lambda h: seen.append([o.name for o in h.pattern.objects] if h.pattern else None))
    with TestClient(app):
        pass
    assert seen == [["a", "b"]]


def test_a_design_change_still_asks_for_an_autosave(hub, tmp_path):
    import asyncio
    hub.pattern = Pattern("sketch", [_wire("a")])
    asyncio.run(hub.notify_design_changed())
    assert design_store.flush_autosave()
    assert _autosaved_names(tmp_path) == ["a"]
