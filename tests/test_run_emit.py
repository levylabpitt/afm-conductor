"""Slice 3.3: each hardware write and scan builds its run, stores it, and links it.

Driven on the golden rig (the fake Igor), through the same FLEX door the goldens use. The
failure tests check the rule that matters most: a run that cannot be built or stored changes
nothing the instrument is told.
"""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone

import numpy as np
import pytest
from golden.harness import HERE, GoldenRig, record
from golden.scenarios import SCENARIOS
from test_build_write_run import compile_scenario

from afm_litho.hardware.lithography_run import WriteSetpoint
from afm_litho.run import Placement, run_from_json, run_to_json
from afm_litho.run.build_write import fired_points, points_fingerprint
from afm_litho.run.model import LithoObject, Scan
from afm_litho.service import run_store
from afm_litho.service.run_store import RunStore


@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


def runtime_of(hub):
    return hub.store.path.parent


def stored_ids(hub) -> set[str]:
    folder = run_store.store_dir(runtime_of(hub))
    return {p.name.removesuffix(".run.json") for p in folder.glob("*.run.json")}


def play(hub, name: str, *, result: str):
    """Run a golden scenario on the rig; return (new execution ids, the last result reply)."""
    before = stored_ids(hub)
    with GoldenRig(hub, name) as rig:
        doors = {"door": rig.door, "inst": rig.inst}
        fn = SCENARIOS[name]
        fn(**{k: doors[k] for k in inspect.signature(fn).parameters})
        rig.quiesce()
        reply = rig.door.call(result, {})
        pattern = _compile_like_the_worker(hub) if name.startswith("litho") else None
    return sorted(stored_ids(hub) - before), reply, pattern


def _compile_like_the_worker(hub):
    """The compiled strokes of the loaded design, with the worker's own arguments."""
    from afm_litho.litho.compiler import compile_pattern
    from afm_litho.service.routers.litho import _litho_frame
    w = hub.host.world
    return compile_pattern(hub.pattern, w.x_lvdt_sens, w.y_lvdt_sens, w.x_lvdt_offset,
                           w.y_lvdt_offset, frame=_litho_frame())["segments"]


def load(hub, execution_id: str):
    return RunStore(run_store.store_dir(runtime_of(hub))).load(execution_id)


def resolve(hub, link: str):
    from pathlib import Path
    path = Path(link)
    return path if path.is_absolute() else runtime_of(hub) / path


def test_a_write_stores_one_run_with_its_links(hub, client):
    (ident,), reply, segments = play(hub, "litho_single", result="getWriteResult")
    run = load(hub, ident)
    assert run.recipe.kind == "write" and ident.startswith("w-")
    assert reply["run"]["execution_id"] == ident
    assert resolve(hub, reply["run"]["path"]).is_file()
    # The sidecar names the run.
    sidecar = json.loads(resolve(hub, reply["files"]["npz"]).parents[1].joinpath(
        "json", resolve(hub, reply["files"]["npz"]).stem + ".json").read_text())
    assert sidecar["run"] == reply["run"]
    # The design and compile settings are saved once, and linked.
    ex = run.execution
    design = json.loads(resolve(hub, ex.design_snapshot).read_text())
    assert design["object_count"] == 1
    assert "frame" in json.loads(resolve(hub, ex.compile_settings).read_text())
    assert run.recipe.design.compiler_version.startswith("afm-litho ")
    # The JSON round-trips.
    assert run_from_json(run_to_json(run)) == run
    # The commanded X/Y file sits beside the capture, pairs by stem, and matches the run.
    (litho,) = [e for e in run.recipe.body if isinstance(e, LithoObject)]
    (link,) = ex.commanded_xy
    assert link.step == run.recipe.body.index(litho)
    path = resolve(hub, link.path)
    capture = resolve(hub, reply["files"]["npz"])
    assert path.name == f"{capture.stem}.step{link.step:02d}.npz"
    with np.load(path) as f:
        assert f["x_um"].dtype == np.float32 and f["y_um"].dtype == np.float32
        x, y, v = fired_points(*segments)
        assert len(f["x_um"]) == litho.points == len(x)
        assert np.array_equal(f["x_um"], x.astype(np.float32))
        assert np.array_equal(f["y_um"], y.astype(np.float32))
    assert points_fingerprint(x, y, v) == litho.points_fingerprint


def test_an_aborted_write_has_commanded_files_only_for_the_passes_it_fired(hub, client):
    """abort_in_transit: the first wire is written, the abort lands in the move to the second."""
    (ident,), reply, _ = play(hub, "abort_in_transit", result="getWriteResult")
    assert reply["aborted"] is True
    run = load(hub, ident)
    passes = [n for n, e in enumerate(run.recipe.body) if isinstance(e, LithoObject)]
    assert len(passes) == 2
    (link,) = run.execution.commanded_xy
    assert link.step == passes[0] and resolve(hub, link.path).is_file()
    assert not any(f"step{passes[1]:02d}" in f.path for f in run.execution.commanded_xy)
    own = resolve(hub, link.path)                 # "<this run's stem>.stepNN.npz"
    stem = own.name.split(".step")[0]
    assert not list(own.parent.glob(f"{stem}.step{passes[1]:02d}*"))


def test_a_pass_cut_short_is_saved_whole_and_marked_partial(tmp_path):
    from golden.scenarios import WIRE
    objs, segments = compile_scenario([WIRE])
    store = RunStore(tmp_path / "runs")
    stored = run_store.emit_write_run(
        store, tmp_path, design_doc={"objects": []}, compile_doc={}, design_name="w",
        objects=objs, segments=segments, setpoint=WriteSetpoint(0.0, relative=False),
        amp_gain=25.0, platform="cypher", placement=Placement("test", 30, 0, (0, 0)),
        bias_window_v=(-8, 30))
    name = segments[0]["name"]
    run = run_store.finish_write_run(store, tmp_path, "20261024_180200", stored,
                                     written=set(), partial=(name, 0.4), captures=())
    (link,) = run.execution.commanded_xy
    assert link.path.endswith(".partial.npz")
    with np.load(tmp_path / link.path) as f:
        assert float(f["fired_fraction"]) == pytest.approx(0.4)
        assert len(f["x_um"]) == run.recipe.body[link.step].points
    none = run_store.finish_write_run(store, tmp_path, "20261024_180201", stored,
                                      written=set(), partial=None, captures=())
    assert none.execution.commanded_xy == ()


def test_the_capture_files_are_linked_from_the_run(hub, client):
    (ident,), reply, _ = play(hub, "litho_single", result="getWriteResult")
    links = load(hub, ident).execution.captures
    assert links and links[0].endswith(".npz")
    assert all(resolve(hub, link).is_file() for link in links)
    (sident,), sreply, _ = play(hub, "scan_chained_contact", result="getScanResult")
    assert load(hub, sident).execution.captures
    assert sreply["run"]["execution_id"] == sident


def test_a_store_that_cannot_be_written_changes_nothing_for_a_write(hub, client, monkeypatch):
    def broken(_runtime):
        raise OSError("disk full")
    monkeypatch.setattr(run_store, "store_dir", broken)
    recorded = record(hub, "litho_single", client=client)
    assert recorded["wire"]["actuation"] == frozen_wire("litho_single")


def test_a_store_that_cannot_be_written_changes_nothing_for_a_scan(hub, client, monkeypatch):
    def broken(_runtime):
        raise OSError("disk full")
    monkeypatch.setattr(run_store, "store_dir", broken)
    recorded = record(hub, "scan_chained_contact", client=client)
    assert recorded["wire"]["actuation"] == frozen_wire("scan_chained_contact")
    with GoldenRig(hub, "scan_chained_contact") as rig:
        SCENARIOS["scan_chained_contact"](rig.door)
        rig.quiesce()
        reply = rig.door.call("getScanResult", {})
    assert reply["run"] == "no run: OSError: disk full"


def test_a_scan_stores_one_run_and_no_commanded_file(hub, client):
    (ident,), reply, _ = play(hub, "scan_chained_contact", result="getScanResult")
    run = load(hub, ident)
    assert run.recipe.kind == "scan" and ident.startswith("s-")
    assert reply["run"]["execution_id"] == ident
    assert any(isinstance(e, Scan) for e in run.recipe.body)
    assert run.execution.commanded_xy == ()
    assert run_from_json(run_to_json(run)) == run
    sidecar = json.loads(resolve(hub, reply["files"]["json"]).read_text())
    assert sidecar["run"] == reply["run"]


def test_ids_stay_unique_when_runs_are_already_stored(hub, client):
    first, _, _ = play(hub, "scan_chained_contact", result="getScanResult")
    second, _, _ = play(hub, "scan_chained_contact", result="getScanResult")
    assert len(first) == len(second) == 1 and first != second


def frozen_wire(name: str):
    return json.loads((HERE / f"{name}.golden.json").read_text())["wire"]["actuation"]


def test_a_write_whose_run_cannot_be_built_sends_the_same_commands(hub, client, monkeypatch):
    def broken(**_kw):
        raise RuntimeError("forced")
    monkeypatch.setattr("afm_litho.run.build_write.build_write_run", broken)
    before = stored_ids(hub)
    recorded = record(hub, "litho_single", client=client)
    assert recorded["wire"]["actuation"] == frozen_wire("litho_single")
    assert stored_ids(hub) == before


def test_a_write_whose_run_cannot_be_built_says_so_in_its_sidecar(hub, client, monkeypatch):
    def broken(**_kw):
        raise RuntimeError("forced")
    monkeypatch.setattr("afm_litho.run.build_write.build_write_run", broken)
    with GoldenRig(hub, "litho_single") as rig:
        SCENARIOS["litho_single"](rig.door)
        rig.quiesce()
        reply = rig.door.call("getWriteResult", {})
    assert reply["run"] == "no run: RuntimeError: forced"
    npz = resolve(hub, reply["files"]["npz"])
    sidecar = json.loads((npz.parents[1] / "json" / f"{npz.stem}.json").read_text())
    assert sidecar["run"] == "no run: RuntimeError: forced"


def test_a_scan_whose_run_cannot_be_built_sends_the_same_commands(hub, client, monkeypatch):
    def broken(*_a, **_kw):
        raise RuntimeError("forced")
    monkeypatch.setattr("afm_litho.run.build_scan.build_scan_run", broken)
    before = stored_ids(hub)
    recorded = record(hub, "scan_chained_contact", client=client)
    assert recorded["wire"]["actuation"] == frozen_wire("scan_chained_contact")
    assert stored_ids(hub) == before


def test_a_filled_region_is_recorded_as_no_run(tmp_path):
    from golden.scenarios import PAD
    objs, segments = compile_scenario([PAD])
    out = run_store.emit_write_run(
        RunStore(tmp_path / "runs"), tmp_path, design_doc={"objects": []}, compile_doc={},
        design_name="pad", objects=objs, segments=segments,
        setpoint=WriteSetpoint(0.0, relative=False), amp_gain=25.0, platform="cypher",
        placement=Placement("test", 30, 0, (0, 0)), bias_window_v=(-8, 30))
    assert isinstance(out, str) and out.startswith("no run: UnsupportedWrite")
    assert not (tmp_path / "runs").exists() or not list((tmp_path / "runs").glob("*.run.json"))


def test_the_store_folder_follows_the_setting(tmp_path, monkeypatch):
    assert run_store.store_dir(tmp_path) == tmp_path / "runs"
    monkeypatch.setenv(run_store.RUN_STORE_ENV, str(tmp_path / "elsewhere"))
    assert run_store.store_dir(tmp_path) == tmp_path / "elsewhere"


def test_stored_ids_are_taken_across_restarts(tmp_path):
    from golden.scenarios import WIRE
    objs, segments = compile_scenario([WIRE])
    when = datetime(2026, 10, 24, 18, 2, tzinfo=timezone.utc)
    store = RunStore(tmp_path / "runs")
    ids = []
    for _ in range(2):  # a new RunStore each time: nothing is remembered in memory
        out = run_store.emit_write_run(
            RunStore(store.folder), tmp_path, design_doc={"objects": []}, compile_doc={},
            design_name="w", objects=objs, segments=segments,
            setpoint=WriteSetpoint(0.0, relative=False), amp_gain=25.0, platform="cypher",
            placement=Placement("test", 30, 0, (0, 0)), bias_window_v=(-8, 30), started=when)
        ids.append(out.run.execution.execution_id)
    assert ids == ["w-2026-10-24-1802-01", "w-2026-10-24-1802-02"]
    assert store.taken_ids("write", when) == set(ids)
    assert store.taken_ids("scan", when) == set()
