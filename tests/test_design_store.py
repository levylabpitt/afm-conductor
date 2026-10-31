"""Save, open and list through the cockpit's routes, against a temp runtime."""

from __future__ import annotations

import json
import types

import pytest

from afm_litho.litho.compiler import LithoObject, Pattern
from afm_litho.service import design_store


@pytest.fixture
def hub(client, tmp_path):
    from afm_litho.service.app import hub
    from afm_litho.service.sessions import SessionStore
    saved = (hub.store, hub.pattern, hub.host.written_objects, hub.ports)
    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    hub.pattern = None
    hub.host.written_objects = set()
    yield hub
    design_store.flush_autosave()
    hub.store, hub.pattern, hub.host.written_objects, hub.ports = saved


def _wire(name, y=0.0, **kw):
    return LithoObject(name, "path", [(0.0, y), (1.0, y), (2.0, y + 0.5)], **kw)


def _design(hub, *names):
    hub.pattern = Pattern("sketch", [_wire(n, float(i)) for i, n in enumerate(names)])
    return list(hub.pattern.objects)


def _save(client, name):
    return client.post("/api/design/save", json={"name": name}).json()


def _open(client, name, mode=None):
    body = {"name": name} if mode is None else {"name": name, "mode": mode}
    return client.post("/api/design/open", json=body).json()


def _file(tmp_path, name):
    return tmp_path / "designs" / f"{name}.design.json"


def test_save_writes_a_file_and_says_when_it_replaced_one(client, hub, tmp_path):
    _design(hub, "a", "b")
    first = _save(client, "device A")
    assert first["ok"] and first["name"] == "device_A"
    assert _save(client, "dev.1.design.json")["name"] == "dev.1"
    for typed, filed in [("../../dev 2", ".._.._dev_2"), ("a/b", "a_b"),
                         ("a\\b", "a_b"), ("a:b", "a_b"), ("C:evil", "C_evil")]:
        assert _save(client, typed)["name"] == filed
    for stray in ("dev.1", ".._.._dev_2", "a_b", "C_evil"):
        _file(tmp_path, stray).unlink()
    assert sorted(p.name for p in tmp_path.rglob("*.design.json")) == [
        "device_A.design.json"]
    assert first["overwrote"] is False and first["object_count"] == 2
    doc = json.loads(_file(tmp_path, "device_A").read_text(encoding="utf-8"))
    assert doc["format"] == "afm-litho-design" and doc["version"] == 1
    assert [o["name"] for o in doc["objects"]] == ["a", "b"]
    assert set(doc["frame"]) == {"scan", "litho"}
    assert _save(client, "device A")["overwrote"] is True
    listed = client.get("/api/design/files").json()["designs"]
    doc = json.loads(_file(tmp_path, "device_A").read_text(encoding="utf-8"))
    assert listed == [{"name": "device_A", "saved_at": doc["saved_at"],
                       "object_count": 2}]
    assert not list((tmp_path / "designs").glob("*.tmp"))


def test_an_empty_design_is_not_saved(client, hub, tmp_path):
    reply = _save(client, "nothing")
    assert reply["reason"] == "no_design" and "error" in reply
    hub.pattern = Pattern("sketch", [])
    assert _save(client, "nothing")["reason"] == "no_design"
    assert not _file(tmp_path, "nothing").exists()


@pytest.mark.parametrize("name, reason", [
    ("_autosave", "reserved_name"), ("_previous", "reserved_name"),
    ("_AutoSave", "reserved_name"), ("_previous.design.json", "reserved_name"),
    ("", "bad_name"), ("   ", "bad_name"), (None, "bad_name"), (7, "bad_name"),
    (".design.json", "bad_name"), ("..", "bad_name")])
def test_reserved_and_missing_names_are_refused(client, hub, tmp_path, name, reason):
    _design(hub, "a")
    assert _save(client, name)["reason"] == reason
    assert not (tmp_path / "designs").exists()


def test_open_adds_by_default_and_reports_every_name(client, hub):
    _design(hub, "a", "b", "c")
    _save(client, "dev")
    existing = _design(hub, "a", "c", "c-2")
    reply = _open(client, "dev")
    assert reply["ok"] and reply["mode"] == "add" and reply["name"] == "dev"
    assert reply["objects"] == [
        {"saved_name": "a", "name": "a-2", "renamed": True},
        {"saved_name": "b", "name": "b", "renamed": False},
        {"saved_name": "c", "name": "c-3", "renamed": True}]
    assert reply["object_count"] == 3 and reply["total_objects"] == 6
    assert reply["frame_differs"] is False and reply["missing_ports"] == []
    assert set(reply) >= {"frame_saved", "frame_now"}
    assert [o.name for o in hub.pattern.objects] == ["a", "c", "c-2", "a-2", "b", "c-3"]
    assert all(now is before for now, before in zip(hub.pattern.objects, existing))


def test_an_add_keeps_written_marks_and_gives_the_new_objects_none(client, hub):
    _design(hub, "a", "b")
    _save(client, "dev")
    live = hub.pattern.objects
    existing = list(live)
    hub.host.written_objects = {id(existing[0])}
    _open(client, "dev", "add")
    assert hub.pattern.objects is live
    assert all(now is before for now, before in zip(live, existing))
    assert id(existing[0]) in hub.host.written_objects
    assert not any(id(o) in hub.host.written_objects for o in hub.pattern.objects[2:])


def test_an_add_into_an_empty_design_keeps_the_saved_names(client, hub):
    _design(hub, "a", "b")
    _save(client, "dev")
    hub.pattern = None
    reply = _open(client, "dev")
    assert [e["renamed"] for e in reply["objects"]] == [False, False]
    assert [o.name for o in hub.pattern.objects] == ["a", "b"]


def test_replace_keeps_the_earlier_design_as_previous(client, hub, tmp_path):
    _design(hub, "a", "b")
    _save(client, "dev")
    earlier = _design(hub, "a", "x", "y")
    hub.host.written_objects = {id(earlier[0])}
    reply = _open(client, "dev", "replace")
    assert reply["ok"] and reply["mode"] == "replace"
    assert [e["name"] for e in reply["objects"]] == ["a", "b"]
    assert not any(e["renamed"] for e in reply["objects"])
    assert [o.name for o in hub.pattern.objects] == ["a", "b"]
    assert hub.host.written_objects == set()
    assert _file(tmp_path, "_previous").is_file()
    names = [d["name"] for d in client.get("/api/design/files").json()["designs"]]
    assert sorted(names) == ["_previous", "dev"]
    back = _open(client, "_previous", "replace")
    assert back["ok"]
    assert [o.name for o in hub.pattern.objects] == ["a", "x", "y"]
    assert hub.pattern.objects == earlier


def test_replace_is_refused_when_previous_cannot_be_written(client, hub, monkeypatch):
    _design(hub, "a")
    _save(client, "dev")
    earlier = _design(hub, "keep-me")
    real = design_store.write_document

    def failing(runtime_dir, name, doc):
        if name == design_store.PREVIOUS:
            raise OSError(28, "No space left on device")
        return real(runtime_dir, name, doc)
    monkeypatch.setattr(design_store, "write_document", failing)
    reply = _open(client, "dev", "replace")
    assert reply["reason"] == "previous_save_failed"
    assert "No space left" in reply["error"]
    assert hub.pattern.objects == earlier and hub.pattern.objects[0] is earlier[0]


def test_replacing_an_empty_design_writes_no_previous(client, hub, tmp_path):
    _design(hub, "a")
    _save(client, "dev")
    hub.pattern = None
    assert _open(client, "dev", "replace")["ok"]
    assert not _file(tmp_path, "_previous").exists()


def _hand_written(tmp_path, name, objects, **top):
    folder = tmp_path / "designs"
    folder.mkdir(exist_ok=True)
    doc = {"format": "afm-litho-design", "version": 1, "name": name,
           "saved_at": "2026-10-05T10:00:00", "objects": objects, **top}
    _file(tmp_path, name).write_text(json.dumps(doc), encoding="utf-8")


def test_a_voltage_outside_the_window_leaves_the_design_untouched(client, hub, tmp_path):
    window = hub.host.instrument_model.tip_voltage_window_v.value
    _hand_written(tmp_path, "hot", [
        {"name": "fine", "kind": "path", "points_um": [[0, 0], [1, 0]], "voltage": 5.0},
        {"name": "too-hot", "kind": "path", "points_um": [[0, 1], [1, 1]],
         "voltage": float(window[1]) + 100.0}])
    earlier = _design(hub, "a")
    for mode in ("add", "replace"):
        reply = _open(client, "hot", mode)
        assert reply["reason"] == "tip_voltage_out_of_window", reply
        assert reply["object_id"] == "too-hot"
        assert hub.pattern.objects == earlier and hub.pattern.objects[0] is earlier[0]
    assert not _file(tmp_path, "_previous").exists()


def _writing():
    return types.SimpleNamespace(is_alive=lambda: True)


def test_open_is_refused_during_a_write_before_the_file_is_read(client, hub,
                                                                monkeypatch):
    from afm_litho.service.commands import design_cmds
    earlier = _design(hub, "a")
    _save(client, "dev")
    monkeypatch.setattr(hub.host, "litho_thread", _writing())
    read = []
    monkeypatch.setattr(design_cmds, "_submit_build", lambda *a: read.append(a))
    for mode in ("add", "replace"):
        reply = _open(client, "dev", mode)
        assert reply["reason"] == "write_in_flight", reply
    assert read == [] and hub.pattern.objects == earlier
    assert _save(client, "during-write")["ok"]


@pytest.mark.parametrize("mode", ["add", "replace"])
def test_a_write_that_starts_while_the_file_is_built_refuses_the_open(
        client, hub, monkeypatch, mode):
    from afm_litho.service.commands import design_cmds
    earlier = _design(hub, "a")
    _save(client, "dev")
    real = design_cmds._read_and_build

    async def write_starts(h, path, name):
        built = await real(h, path, name)
        monkeypatch.setattr(h.host, "litho_thread", _writing())
        return built
    monkeypatch.setattr(design_cmds, "_read_and_build", write_starts)
    reply = _open(client, "dev", mode)
    assert reply["reason"] == "write_in_flight", reply
    assert hub.pattern.objects == earlier and hub.pattern.objects[0] is earlier[0]


def test_a_write_that_starts_while_previous_is_saved_refuses_the_replace(
        client, hub, monkeypatch):
    from afm_litho.service.commands import design_cmds
    earlier = _design(hub, "a")
    _save(client, "dev")
    real = design_cmds._write_snapshot

    async def write_starts(snapshot, name):
        head = await real(snapshot, name)
        monkeypatch.setattr(hub.host, "litho_thread", _writing())
        return head
    monkeypatch.setattr(design_cmds, "_write_snapshot", write_starts)
    reply = _open(client, "dev", "replace")
    assert reply["reason"] == "write_in_flight", reply
    assert hub.pattern.objects == earlier and hub.pattern.objects[0] is earlier[0]


@pytest.mark.parametrize("mode", ["add", "replace"])
def test_an_opened_object_never_inherits_a_stale_written_mark(client, hub, monkeypatch,
                                                              mode):
    from afm_litho.service.commands import design_cmds
    existing = _design(hub, "a")
    incoming = [_wire("new")]

    async def built(h, path, name):
        return {"objects": []}, incoming, []
    monkeypatch.setattr(design_cmds, "_read_and_build", built)
    hub.host.written_objects = {id(existing[0]), id(incoming[0])}
    assert _open(client, "dev", mode)["ok"]
    assert hub.pattern.objects[-1] is incoming[0]
    assert id(incoming[0]) not in hub.host.written_objects
    assert (id(existing[0]) in hub.host.written_objects) == (mode == "add")


def test_the_autosave_is_not_listed_and_previous_is(client, hub, tmp_path):
    _design(hub, "a")
    _save(client, "dev")
    for name in ("_autosave", "_previous"):
        _hand_written(tmp_path, name, [])
    names = {d["name"] for d in client.get("/api/design/files").json()["designs"]}
    assert names == {"dev", "_previous"}


def test_the_list_reads_only_the_head_of_each_file(client, hub, tmp_path, monkeypatch):
    _design(hub, "a")
    _save(client, "dev")
    _hand_written(tmp_path, "by-hand", [{"name": "a", "kind": "path",
                                         "points_um": [[0, 0], [1, 1]]}])
    _file(tmp_path, "torn").write_bytes(b"\xff\xfe not a design")
    monkeypatch.setattr(design_store, "parse_document", None)
    listed = {d["name"]: d for d in client.get("/api/design/files").json()["designs"]}
    assert listed["dev"]["object_count"] == 1 and listed["dev"]["saved_at"]
    assert listed["by-hand"] == {"name": "by-hand", "saved_at": "2026-10-05T10:00:00",
                                 "object_count": None}
    assert listed["torn"] == {"name": "torn", "saved_at": None, "object_count": None}


def test_the_reply_says_whether_the_earlier_design_was_kept(client, hub, tmp_path):
    _design(hub, "a")
    _save(client, "dev")
    assert _open(client, "dev", "add")["previous_saved"] is False
    assert _open(client, "dev", "replace")["previous_saved"] is True
    hub.pattern = None
    _file(tmp_path, "_previous").unlink()
    assert _open(client, "dev", "replace")["previous_saved"] is False
    assert not _file(tmp_path, "_previous").exists()


# ------------------------------------------------------------ size and build cost --

_SQUARE = [[0, 0], [5, 0], [5, 5], [0, 5]]


@pytest.fixture
def never_compiles(monkeypatch):
    from afm_litho.litho import compiler
    from afm_litho.service.host import litho_cmds

    def never(*a, **kw):
        raise AssertionError("an open must not compile")
    for module, name in ((litho_cmds, "_compile_report"), (compiler, "compile_pattern"),
                         (compiler, "compile_object"), (compiler, "expand")):
        monkeypatch.setattr(module, name, never)


@pytest.mark.parametrize("record, field", [
    ({"name": "s", "kind": "shape", "points_um": _SQUARE, "fill": {"spacing_um": 0.0}},
     "fill.spacing_um"),
    ({"name": "s", "kind": "shape", "points_um": _SQUARE, "fill": {"spacing_um": -1.0}},
     "fill.spacing_um"),
    ({"name": "s", "kind": "path", "points_um": _SQUARE, "speed_um_s": -2.0},
     "speed_um_s"),
    ({"name": "s", "kind": "path", "points_um": _SQUARE, "interpolation_um": 0.0},
     "interpolation_um"),
])
def test_a_setting_that_cannot_be_compiled_is_refused_at_once(client, hub, tmp_path,
                                                              record, field):
    import time
    _hand_written(tmp_path, "bad", [record])
    earlier = _design(hub, "a")
    began = time.monotonic()
    reply = _open(client, "bad")
    assert time.monotonic() - began < 2.0
    assert (reply["reason"], reply["object"], reply["field"]) == ("bad_record", "s", field)
    assert hub.pattern.objects == earlier


def test_a_file_too_large_to_be_a_design_is_refused_before_it_is_read(
        client, hub, tmp_path, monkeypatch):
    from pathlib import Path
    _design(hub, "a")
    _save(client, "dev")
    size = _file(tmp_path, "dev").stat().st_size
    monkeypatch.setattr(design_store, "MAX_FILE_BYTES", size - 1)
    monkeypatch.setattr(Path, "read_bytes", None)
    reply = _open(client, "dev")
    assert reply["reason"] == "file_too_large", reply
    assert "MB" in reply["error"]
    assert {d["name"] for d in client.get("/api/design/files").json()["designs"]} == {
        "dev"}


@pytest.mark.parametrize("stem", ["sp ace  two", "it's", "a;b&c", "dev.v2", "  padded",
                                  "Ünïcode", "_previous"])
def test_a_file_named_by_hand_that_is_listed_also_opens(client, hub, tmp_path, stem):
    _hand_written(tmp_path, stem, [{"name": "w", "kind": "path",
                                    "points_um": [[0, 0], [1, 1]]}])
    (listed,) = client.get("/api/design/files").json()["designs"]
    assert listed["name"] == stem
    reply = _open(client, listed["name"])
    assert reply.get("ok") and reply["name"] == stem, reply
    assert _open(client, stem + ".design.json").get("ok")


def test_a_name_cannot_reach_outside_the_designs_folder(client, hub, tmp_path):
    outside = tmp_path / "secret.design.json"
    outside.write_text(json.dumps({
        "format": "afm-litho-design", "version": 1,
        "objects": [{"name": "w", "kind": "path", "points_um": [[0, 0], [1, 1]]}]}),
        encoding="utf-8")
    (tmp_path / "designs").mkdir()
    for name in ("../secret", "..\\secret", str(outside), str(outside)[:-12],
                 "designs/../secret"):
        assert _open(client, name)["reason"] == "not_found", name
    assert hub.pattern is None


def test_a_saved_file_is_plain_readable_text(client, hub, tmp_path):
    from afm_litho.litho import design_file
    assert client.post("/api/design/add", json={
        "kind": "path", "name": "wire", "points_um": [[0, 0], [1, 0.5]]}).json()["ok"]
    assert client.post("/api/design/add", json={
        "kind": "path", "name": "helix", "points_um": [[0, 1], [2, 1]]}).json()["ok"]
    assert client.post("/api/design/chiral", json={
        "indices": [1], "enabled": True, "lambda_um": 0.05, "y_amp_um": 0.05, "v_k": 1.0,
        "phase_deg": 90, "handedness": "R"}).json()["ok"]
    assert _save(client, "dev")["ok"]
    text = _file(tmp_path, "dev").read_bytes().decode("utf-8")
    doc = json.loads(text)
    assert set(doc) <= set(design_file.DOCUMENT_KEYS)
    for rec in doc["objects"]:
        assert set(rec) <= set(design_file.RECORD_KEYS)
        assert set(rec.get("chiral", {})) <= {"lambda_um", "y_amp_um", "v_k",
                                              "phase_deg", "axis_um"}
    assert "points_um" not in doc["objects"][1]
    lines = text.splitlines()
    assert len(lines) > 30 and '      "name": "helix",' in lines
    assert "          [0.0, 1.0]," in lines and text.endswith("}\n")


def test_a_region_the_cockpit_made_reopens_without_compiling(client, hub, tmp_path,
                                                             monkeypatch):
    """A contour region compiles for seconds; saving and reopening it must not."""
    import time
    sketched = client.post("/api/sketch", json={
        "pen": "region", "mode": "literal", "params": {"pitch_um": 0.05, "fill": "contour"},
        "points_um": [[0, 0], [6, 0], [6, 6], [0, 6], [0, 0]]}).json()
    assert sketched.get("ok"), sketched
    drawn = list(hub.pattern.objects)
    assert drawn and drawn[0].fill_kind == "contour"
    assert _save(client, "region")["ok"]
    from afm_litho.litho import compiler
    from afm_litho.service.host import litho_cmds

    def never(*a, **kw):
        raise AssertionError("an open must not compile")
    for module, name in ((litho_cmds, "_compile_report"), (compiler, "compile_pattern"),
                         (compiler, "compile_object"), (compiler, "expand")):
        monkeypatch.setattr(module, name, never)
    began = time.monotonic()
    for mode in ("replace", "add", "replace"):
        reply = _open(client, "region", mode)
        assert reply.get("ok"), reply
    assert time.monotonic() - began < 2.0
    assert hub.pattern.objects == drawn


def test_an_object_in_the_design_that_cannot_be_compiled_does_not_stop_an_open(
        client, hub, never_compiles):
    _design(hub, "a")
    _save(client, "dev")
    hub.pattern.objects.append(LithoObject(
        "pad", "shape", [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)], fill_spacing_um=0.0))
    assert _open(client, "dev", "add")["ok"]
    assert [o.name for o in hub.pattern.objects] == ["a", "pad", "a-2"]


def _square(half, c=0.0):
    return [(c - half, c - half), (c + half, c - half), (c + half, c + half),
            (c - half, c + half)]


@pytest.mark.parametrize("record", [
    {"name": "s", "kind": "shape", "points_um": _SQUARE, "fill": {"spacing_um": 1e-5}},
    {"name": "s", "kind": "rect", "points_um": [[0, 0], [50, 50]],
     "fill": {"spacing_um": 0.001}},
    {"name": "s", "kind": "shape", "points_um": _square(25), "holes_um": [_square(24)],
     "fill": {"spacing_um": 0.05, "kind": "contour"}},
    {"name": "s", "kind": "path", "chiral": {
        "lambda_um": 0.025, "y_amp_um": 0.055, "v_k": 1.0, "phase_deg": 90.0,
        "axis_um": [[-45, 0], [45, 0]]}},
], ids=["fine fill", "large fine rect", "contour frame", "90 um chiral wire"])
def test_open_puts_no_limit_on_points(client, hub, tmp_path, never_compiles, record):
    _hand_written(tmp_path, "large", [record])
    reply = _open(client, "large")
    assert reply.get("ok"), reply


def test_a_build_that_runs_long_is_abandoned(client, hub, monkeypatch):
    import time

    from afm_litho.service.commands import design_cmds
    earlier = _design(hub, "a")
    _save(client, "dev")
    real = design_cmds._build
    monkeypatch.setattr(design_cmds, "BUILD_BOUND_S", 0.1)
    monkeypatch.setattr(design_cmds, "_build",
                        lambda *a: time.sleep(0.6) or real(*a))
    began = time.monotonic()
    reply = _open(client, "dev", "replace")
    assert time.monotonic() - began < 0.5
    assert reply["reason"] == "build_timeout", reply
    assert hub.pattern.objects == earlier and hub.pattern.objects[0] is earlier[0]
    again = _open(client, "dev")
    assert again["reason"] == "build_in_progress" and "loadPattern" not in again["error"]
    deadline = time.monotonic() + 5
    while not design_cmds._building.done() and time.monotonic() < deadline:
        time.sleep(0.02)
    monkeypatch.undo()
    assert _open(client, "dev")["ok"]


def test_the_build_runs_on_its_own_thread_not_the_default_pool(client, hub, monkeypatch):
    import threading

    from afm_litho.service.commands import design_cmds
    _design(hub, "a")
    _save(client, "dev")
    real = design_cmds._build
    seen = []
    monkeypatch.setattr(design_cmds, "_build",
                        lambda *a: seen.append(threading.current_thread()) or real(*a))
    assert _open(client, "dev")["ok"]
    assert seen[0].name.startswith("design-build") and seen[0].daemon


def _longest_stall(coro_fn):
    """Run a coroutine and return the longest time the event loop went without
    turning while it ran."""
    import asyncio
    import time

    async def main():
        gaps, done = [], False

        async def ticker():
            last = time.perf_counter()
            while not done:
                await asyncio.sleep(0.001)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now
        task = asyncio.create_task(ticker())
        await asyncio.sleep(0.05)
        gaps.clear()
        result = await coro_fn()
        done = True
        await task
        return result, max(gaps)
    return asyncio.run(main())


@pytest.mark.parametrize("existing, incoming, same_name", [
    (1, 20000, True), (5000, 5000, False)])
def test_opening_many_objects_does_not_stall_the_event_loop(
        client, hub, tmp_path, existing, incoming, same_name):
    from afm_litho.service.commands import design_cmds
    name = (lambda i: "w") if same_name else (lambda i: f"n{i}")
    _hand_written(tmp_path, "many", [
        {"name": name(i), "kind": "path", "points_um": [[0, i / 1000], [1, i / 1000]]}
        for i in range(incoming)])
    hub.pattern = Pattern("sketch", [_wire(name(i)) for i in range(existing)])
    reply, stall = _longest_stall(lambda: design_cmds.open_design(hub, "many", "add"))
    assert reply["object_count"] == incoming
    assert len({o.name for o in hub.pattern.objects}) == existing + incoming
    assert reply["objects"][-1]["name"] == (
        f"w-{incoming + 1}" if same_name else f"n{incoming - 1}-2")
    assert stall < 0.5, f"the event loop stalled for {stall:.2f} s"


# ------------------------------------------------------- files that are not designs --

@pytest.mark.parametrize("content, reason", [
    (b"\xff\xfe\x00not utf-8\x80", "not_a_design_file"),
    (b"[" * 100_000, "not_a_design_file"),
    (b'{"format": "afm-litho-design", "version": 1, "objects": [{"name": "a", '
     b'"kind": "path", "points_um": [[0, 0], [1, ' + b"9" * 400 + b"]]}]}", "bad_record"),
    (b'{"format": "afm-litho-design", "version": 1, "objects": [{"name": "a", '
     b'"kind": "path", "points_um": [[0, 0], [1, 1]], "voltage": ' + b"9" * 400 + b"}]}",
     "bad_record"),
    (b'{"format": "afm-litho-design", "version": ' + b"9" * 400 + b', "objects": []}',
     "unsupported_version"),
], ids=["not utf-8", "nested too deeply", "huge number in points",
        "huge number in a setting", "huge version"])
def test_a_file_that_is_not_a_design_is_refused_and_does_not_break_the_list(
        client, hub, tmp_path, content, reason):
    earlier = _design(hub, "a")
    _save(client, "good")
    _file(tmp_path, "broken").write_bytes(content)
    reply = client.post("/api/design/open", json={"name": "broken"})
    assert reply.status_code == 200 and reply.json()["reason"] == reason, reply.text
    assert hub.pattern.objects == earlier
    listed = client.get("/api/design/files")
    assert listed.status_code == 200
    assert "good" in {d["name"] for d in listed.json()["designs"]}


def test_a_number_that_is_not_finite_is_named_when_the_save_fails(client, hub):
    _design(hub, "a", "b")
    hub.pattern.objects[1].voltage = float("nan")
    reply = _save(client, "dev")
    assert (reply["reason"], reply["object"], reply["field"]) == (
        "save_failed", "b", "voltage")
    assert "'b'" in reply["error"] and "voltage" in reply["error"]


@pytest.mark.parametrize("objects, top, reason", [
    ([], {"version": 2}, "unsupported_version"),
    ([], {"format": "svg"}, "not_a_design_file"),
    ([], {}, "empty_file"),
    ([{"name": "a", "kind": "path", "points_um": [[0, 0], [1, "x"]]}], {}, "bad_record"),
])
def test_a_file_that_cannot_be_opened_says_why(client, hub, tmp_path, objects, top,
                                               reason):
    _hand_written(tmp_path, "odd", objects, **top)
    earlier = _design(hub, "a")
    reply = _open(client, "odd")
    assert reply["reason"] == reason, reply
    assert hub.pattern.objects == earlier


def test_open_refusals_that_need_no_file(client, hub, tmp_path):
    _design(hub, "a")
    assert _open(client, "nowhere")["reason"] == "not_found"
    assert _open(client, "nowhere", "merge")["reason"] == "bad_mode"
    assert _open(client, "")["reason"] == "bad_name"
    (tmp_path / "designs").mkdir()
    _file(tmp_path, "torn").write_text('{"format": "afm-litho-des', encoding="utf-8")
    assert _open(client, "torn")["reason"] == "not_a_design_file"
    assert client.get("/api/design/files").json() == {"ok": True, "designs": [
        {"name": "torn", "saved_at": None, "object_count": None}]}


def test_open_reports_a_different_frame_and_a_missing_port(client, hub, tmp_path):
    _hand_written(tmp_path, "framed", [
        {"name": "w", "kind": "path", "points_um": [[0, 0], [1, 0]], "voltage": 5.0,
         "end_binding": {"node": "end", "port_id": "e-gone"}},
        {"name": "v", "kind": "path", "points_um": [[0, 1], [1, 1]], "voltage": 5.0,
         "end_binding": {"node": "start", "port_id": "e1"}}],
        frame={"scan": {"size_um": 12345.0, "x_offset_um": 0.0, "y_offset_um": 0.0,
                        "angle_deg": 0.0}, "litho": None})
    hub.ports = [{"id": "e1"}]
    reply = _open(client, "framed")
    assert reply["ok"] and reply["frame_differs"] is True
    assert reply["frame_saved"]["scan"]["size_um"] == 12345.0
    assert reply["missing_ports"] == [{"name": "w", "port_id": "e-gone"}]
    assert hub.pattern.objects[0].points_um == [(0.0, 0.0), (1.0, 0.0)]


def test_an_edit_made_while_previous_is_saved_is_not_lost(client, hub, monkeypatch):
    """The design changes under every attempt, so the replace gives up and
    changes nothing."""
    from afm_litho.service.commands import design_cmds
    _design(hub, "a")
    _save(client, "dev")
    real = design_cmds._write_snapshot

    async def edit_meanwhile(snapshot, name):
        head = await real(snapshot, name)
        hub.pattern.objects.append(_wire(f"late-{len(hub.pattern.objects)}", 9.0))
        return head
    monkeypatch.setattr(design_cmds, "_write_snapshot", edit_meanwhile)
    reply = _open(client, "dev", "replace")
    assert reply["reason"] == "design_changed"
    assert [o.name for o in hub.pattern.objects] == ["a", "late-1", "late-2"]


def test_a_full_disk_does_not_break_an_edit(client, hub, tmp_path, monkeypatch):
    monkeypatch.setenv("AFM_LITHO_DESIGN_AUTOSAVE", "1")

    def full(src, dst):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(design_store.os, "replace", full)

    added = client.post("/api/design/add", json={
        "kind": "path", "points_um": [[0, 0], [1, 1]], "name": "w"}).json()
    assert added.get("ok") is True and added["name"] == "w"
    assert design_store.flush_autosave()
    status = client.get("/api/state").json()["design_autosave"]
    assert status["ok"] is False and "No space left" in status["error"]
    assert not list((tmp_path / "designs").glob("*"))

    reply = _save(client, "dev")
    assert reply["reason"] == "save_failed" and "No space left" in reply["error"]
    assert [o.name for o in hub.pattern.objects] == ["w"]

    monkeypatch.undo()
    monkeypatch.setenv("AFM_LITHO_DESIGN_AUTOSAVE", "1")
    assert client.post("/api/design/add", json={
        "kind": "path", "points_um": [[0, 1], [1, 2]], "name": "w"}).json()["ok"]
    assert design_store.flush_autosave()
    status = client.get("/api/state").json()["design_autosave"]
    assert status["ok"] is True and status["error"] is None and status["saved_at"]
    doc = json.loads(_file(tmp_path, "_autosave").read_text(encoding="utf-8"))
    assert [o["name"] for o in doc["objects"]] == ["w", "w-2"]


def test_notify_design_changed_survives_a_store_that_throws(client, hub, monkeypatch):
    import asyncio
    monkeypatch.setenv("AFM_LITHO_DESIGN_AUTOSAVE", "1")

    def boom(_hub):
        raise RuntimeError("snapshot failed")
    monkeypatch.setattr(design_store, "Snapshot", boom)
    asyncio.run(hub.notify_design_changed())


def test_with_the_switch_off_nothing_is_written(client, hub, tmp_path):
    assert not design_store.autosave_enabled()
    assert client.post("/api/design/add", json={
        "kind": "path", "points_um": [[0, 0], [1, 1]]}).json()["ok"]
    assert design_store.flush_autosave()
    assert not (tmp_path / "designs").exists()
    assert design_store.restore_autosave(hub) is None


@pytest.mark.parametrize("records, named", [
    ([{"name": f"c{i}", "kind": "path", "chiral": {
        "lambda_um": 0.009, "y_amp_um": 0, "v_k": 1.0, "phase_deg": 0.0,
        "axis_um": [[-1000, 0], [1000, 0]]}} for i in range(3)], "c0"),
    ([{"name": "knot", "kind": "path", "bezier": {"closed": False, "nodes": [
        {"x": 1999.0 * (1 if i % 2 else -1), "y": 1999.0 * (1 if i % 4 < 2 else -1),
         "h_in": [-1999.0 * (1 if i % 2 else -1), 1999.0],
         "h_out": [1999.0 * (1 if i % 2 else -1), -1999.0]} for i in range(3000)]}}],
     "knot"),
], ids=["three 2000 um wires", "3000-node knot"])
def test_a_small_file_with_an_absurd_object_is_refused_at_once(
        client, hub, tmp_path, monkeypatch, records, named):
    import time

    from afm_litho.litho import design_file, pens
    monkeypatch.setattr(pens, "chiral_profile", None)
    monkeypatch.setattr(design_file, "flatten_bezier", None)
    _hand_written(tmp_path, "hostile", records)
    earlier = _design(hub, "a")
    began = time.monotonic()
    reply = _open(client, "hostile")
    assert time.monotonic() - began < 1.0
    assert (reply["reason"], reply["object"]) == ("object_too_large", named), reply
    assert hub.pattern.objects == earlier


def test_a_1e9_um_curve_is_refused_at_once(client, hub, tmp_path):
    _hand_written(tmp_path, "far", [{"name": "far", "kind": "path", "bezier": {
        "closed": False, "nodes": [
            {"x": 0.0, "y": 0.0, "h_in": None, "h_out": [1e9, 1e9]},
            {"x": 1.0, "y": 0.0, "h_in": [-1e9, 5.0], "h_out": None}]}}])
    reply = _open(client, "far")
    assert (reply["reason"], reply["object"], reply["field"]) == (
        "bad_record", "far", "bezier")


# ------------------------------------------------------------- chiral, whatever its kind --

_CHIRAL_ON = {"enabled": True, "lambda_um": 0.025, "y_amp_um": 0.055, "v_k": 1.0,
              "phase_deg": 90, "handedness": "R"}


def test_no_chiral_record_ever_holds_points_bias_or_profile(client, hub, tmp_path):
    """Through the routes that turn a chiral wire into another kind: the node tool
    closing its spine, and the properties panel setting its kind."""
    assert client.post("/api/design/add", json={
        "kind": "path", "name": "closed", "points_um": [[0, 0], [5, 0], [5, 5], [0, 5]],
    }).json()["ok"]
    assert client.post("/api/design/chiral", json={
        "indices": [0], **_CHIRAL_ON}).json()["ok"]
    bezier = client.get("/api/design/bezier/0").json()["bezier"]
    bezier["closed"] = True
    assert client.post("/api/design/bezier", json={
        "index": 0, "bezier": bezier}).json().get("ok")
    assert client.post("/api/design/add", json={
        "kind": "path", "name": "retyped", "points_um": [[-5, -3], [5, -3]]}).json()["ok"]
    assert client.post("/api/design/chiral", json={
        "indices": [1], **_CHIRAL_ON}).json()["ok"]
    client.post("/api/design/op", json={"op": "set_props", "indices": [1],
                                        "kind": "shape"})
    before = list(hub.pattern.objects)
    assert [o.kind for o in before] == ["shape", "shape"]
    assert all(len(o.points_um) > 30_000 for o in before)
    assert _save(client, "flipped")["ok"]
    assert _file(tmp_path, "flipped").stat().st_size < 3000
    doc = json.loads(_file(tmp_path, "flipped").read_text(encoding="utf-8"))
    for rec in doc["objects"]:
        assert rec["kind"] == "shape"
        assert not {"points_um", "v_profile", "extra"} & set(rec)
        assert set(rec["chiral"]) <= {"lambda_um", "y_amp_um", "v_k", "phase_deg",
                                      "axis_um"}
    assert _open(client, "flipped", "replace")["ok"]
    assert hub.pattern.objects == before


def test_a_wire_the_chiral_door_accepts_saves_reopens_and_restores(
        client, hub, tmp_path, monkeypatch):
    assert client.post("/api/design/add", json={
        "kind": "path", "name": "fine", "points_um": [[-45, 0], [45, 0]]}).json()["ok"]
    reply = client.post("/api/design/chiral", json={
        "indices": [0], **_CHIRAL_ON, "lambda_um": 0.003, "y_amp_um": 0.0066}).json()
    assert reply.get("ok"), reply
    before = list(hub.pattern.objects)
    assert len(before[0].points_um) == 2_610_002
    assert _save(client, "fine")["ok"]
    assert _open(client, "fine", "replace").get("ok")
    assert hub.pattern.objects == before

    monkeypatch.setenv("AFM_LITHO_DESIGN_AUTOSAVE", "1")
    design_store.request_autosave(hub)
    assert design_store.flush_autosave(60)
    hub.pattern = None
    assert "1 object restored" in design_store.restore_autosave(hub)
    assert hub.pattern.objects == before


# ------------------------------------------------------------ an abandoned build stops --

def test_a_build_abandoned_at_the_clock_stops_within_one_record(client, hub, tmp_path,
                                                               monkeypatch):
    import time

    from afm_litho.litho import pens
    from afm_litho.service.commands import design_cmds
    _hand_written(tmp_path, "forty", [
        {"name": f"w{i}", "kind": "path", "chiral": {
            "lambda_um": 0.025, "y_amp_um": 0.055, "v_k": 1.0, "phase_deg": 0.0,
            "axis_um": [[-45, i], [45, i]]}} for i in range(40)])
    _hand_written(tmp_path, "small", [{"name": "a", "kind": "path",
                                       "points_um": [[0, 0], [1, 1]]}])
    built = []
    real = pens.chiral_profile
    monkeypatch.setattr(pens, "chiral_profile",
                        lambda *a, **kw: built.append(1) or real(*a, **kw))
    monkeypatch.setattr(design_cmds, "BUILD_BOUND_S", 0.6)
    assert _open(client, "forty")["reason"] == "build_timeout"
    at_refusal = len(built)
    deadline = time.monotonic() + 5
    while not design_cmds._building.done() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert design_cmds._building.done()
    assert len(built) <= at_refusal + 1 < 40
    assert design_cmds._building.exception().__context__ is None
    assert _open(client, "small").get("ok")


# ---------------------------------------------------------- the set-aside autosaves --

def _set_aside(tmp_path, file, records):
    folder = tmp_path / "designs"
    folder.mkdir(exist_ok=True)
    (folder / file).write_text(json.dumps({
        "format": "afm-litho-design", "version": 1, "saved_at": "2026-10-05T10:00:00",
        "object_count": len(records), "objects": records}), encoding="utf-8")


_GOOD = {"name": "good", "kind": "path", "points_um": [[0, 0], [1, 1]]}
_BAD = {"name": "broken", "kind": "path", "points_um": [[0, 0]]}


@pytest.mark.parametrize("name, file", [("_autosave.bad", "_autosave.bad.json"),
                                        ("_autosave.unread", "_autosave.unread.json")])
def test_a_set_aside_autosave_opens_with_what_can_be_rebuilt(client, hub, tmp_path,
                                                             name, file):
    _set_aside(tmp_path, file, [_GOOD, _BAD, {"name": "no-kind"},
                                dict(_GOOD, name="also-good")])
    _design(hub, "a")
    reply = _open(client, name)
    assert reply.get("ok") and reply["name"] == name, reply
    assert [e["name"] for e in reply["objects"]] == ["good", "also-good"]
    assert [e["name"] for e in reply["left_out"]] == ["broken", "no-kind"]
    assert "needs at least 2 points" in reply["left_out"][0]["reason"]
    assert set(reply["left_out"][0]) == {"name", "reason"}
    assert [o.name for o in hub.pattern.objects] == ["a", "good", "also-good"]
    assert _open(client, name, "replace")["left_out"] == reply["left_out"]
    assert [o.name for o in hub.pattern.objects] == ["good", "also-good"]


def test_a_set_aside_autosave_with_nothing_to_rebuild_is_refused(client, hub, tmp_path):
    _set_aside(tmp_path, "_autosave.bad.json", [_BAD])
    earlier = _design(hub, "a")
    reply = _open(client, "_autosave.bad")
    assert reply["reason"] == "no_buildable_record", reply
    assert reply["left_out"][0]["name"] == "broken" and "broken" in reply["error"]
    assert hub.pattern.objects == earlier


def test_an_ordinary_file_is_still_refused_whole_on_one_bad_record(client, hub,
                                                                  tmp_path):
    _hand_written(tmp_path, "mixed", [_GOOD, _BAD])
    earlier = _design(hub, "a")
    reply = _open(client, "mixed")
    assert (reply["reason"], reply["object"]) == ("bad_record", "broken")
    assert hub.pattern.objects == earlier
    _hand_written(tmp_path, "whole", [_GOOD])
    assert _open(client, "whole")["left_out"] == []


def test_the_set_aside_autosaves_are_listed_after_the_designs(client, hub, tmp_path):
    _design(hub, "a")
    _save(client, "dev")
    _set_aside(tmp_path, "_autosave.bad.json", [_GOOD, _BAD])
    (tmp_path / "designs" / "_autosave.unread.json").write_bytes(b"\xff not text")
    for stray in ("_autosave.bad.20261005-101500.json", "_autosave.design.json"):
        (tmp_path / "designs" / stray).write_text("{}", encoding="utf-8")
    listed = client.get("/api/design/files").json()["designs"]
    assert listed[0]["name"] == "dev" and "recovered" not in listed[0]
    assert listed[1:] == [
        {"name": "_autosave.bad", "saved_at": "2026-10-05T10:00:00", "object_count": 2,
         "recovered": True},
        {"name": "_autosave.unread", "saved_at": None, "object_count": None,
         "recovered": True}]
    for entry in listed[1:]:
        assert _save(client, entry["name"])["reason"] == "reserved_name"
    assert _open(client, "_autosave.bad").get("ok")


# ------------------------------------ a chiral wire edited in memory reloads as it is --

_RING = [[3 * __import__("math").cos(6.283185307179586 * i / 60),
          3 * __import__("math").sin(6.283185307179586 * i / 60)] for i in range(61)]


@pytest.mark.parametrize("points, edit", [
    ([[-2, 0], [2, 0]], {"op": "set_props", "voltage": 6.0}),
    ([[-2, 0], [2, 0]], {"op": "transform", "sx": 1.5, "sy": 1.5}),
    ([[-2, 0], [2, 0]], {"op": "transform", "rot_deg": 33.3}),
    ([[-2, 0], [2, 0]], {"op": "transform", "dx_um": 1.2345678, "dy_um": -0.7654321}),
    (_RING, None),
], ids=["voltage edited", "scaled", "rotated", "moved", "closed ring"])
def test_a_chiral_wire_reloads_equal_to_memory_after_an_edit(client, hub, tmp_path,
                                                              points, edit):
    assert client.post("/api/design/add", json={
        "kind": "path", "name": "w", "points_um": points}).json()["ok"]
    assert client.post("/api/design/chiral", json={
        "indices": [0], **_CHIRAL_ON, "phase_deg": 37.3, "handedness": "L"}).json()["ok"]
    if edit:
        reply = client.post("/api/design/op", json={"indices": [0], **edit}).json()
        assert "error" not in reply, reply
    before = list(hub.pattern.objects)
    assert _save(client, "wire")["ok"]
    (rec,) = json.loads(_file(tmp_path, "wire").read_text(encoding="utf-8"))["objects"]
    assert "points_um" not in rec and rec["chiral"]["lambda_um"] == 0.025
    assert _open(client, "wire", "replace")["ok"]
    (loaded,), (was,) = hub.pattern.objects, before
    assert loaded is not was
    if edit and "dx_um" in edit:
        # A pure move shifts the baked points in memory instead of re-baking them.
        import numpy as np
        assert len(loaded.points_um) == len(was.points_um)
        assert np.abs(np.asarray(loaded.points_um) - np.asarray(was.points_um)).max() < 1e-12
        assert np.abs(np.asarray(loaded.v_profile) - np.asarray(was.v_profile)).max() < 1e-9
        assert loaded.axis_um == was.axis_um
    else:
        assert loaded == was


def test_the_write_estimates_detached_copy_and_the_autosave_snapshot_keep_apart(
        client, hub, tmp_path, monkeypatch):
    from afm_litho.service.host import litho_cmds
    monkeypatch.setenv("AFM_LITHO_DESIGN_AUTOSAVE", "1")
    assert client.post("/api/design/add", json={
        "kind": "path", "name": "w", "points_um": [[-2, 0], [2, 0]]}).json()["ok"]
    assert client.post("/api/design/chiral", json={
        "indices": [0], **_CHIRAL_ON}).json()["ok"]
    live = hub.pattern.objects[0]
    snapshot = design_store.Snapshot(hub)
    copy = litho_cmds._detached(live)
    assert copy == live and copy.points_um is not live.points_um
    assert snapshot.objects[0].points_um is live.points_um
    copy.points_um.clear()
    copy.profile["v0"] = 99.0
    assert snapshot.matches(hub)
    design_store.request_autosave(hub)
    assert design_store.flush_autosave()
    hub.pattern = None
    assert design_store.restore_autosave(hub)
    assert hub.pattern.objects == [live]
