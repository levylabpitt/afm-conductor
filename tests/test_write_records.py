"""RED (wave-1 tests-only) coverage for docs/INSTRUMENT_MODEL_SPEC.md sec 5 + 10:
every litho write record -- twin or hardware, completed or aborted -- persists
`amp_gain, setpoint, free_air_v, free_air_ts, model_hash, aborted` plus an
`output_a_cmd_v` array of the commanded tip bias per captured point.

None of this exists yet: the twin write path (`_litho_write_body`'s simulation
branch in `service/commands/litho_cmds.py`) never reads `setpoint`/`amp_gain`
from the body and never accumulates a capture (`hw_litho_capture` stays empty,
so `host/litho_worker.py`'s persistence block -- gated on `if cap:` -- writes no
npz/gwy/ibw at all and no json sidecar exists for litho, ever, on any backend).
`hub.instrument_model` (the section-10 `InstrumentModel`, corrected name per
the coordinator's 2026-09-22 note -- `hub.instrument` is the PRE-EXISTING twin
physics calibration object, not this) does not exist either, nor does
`hub.free_air` (the section-10 `FreeAir` baseline) or `/api/instrument/override`.

Coordinator corrections folded in:
  * (12a0593, sec 4+10) the write door's setpoint is RELATIVE to free air:
    the body key is `setpoint_relative`, never a bare `setpoint`.
  * every write body here is the full explicit contract
    `{objects, setpoint_relative, amp_gain}` -- never a bare/default
    selection -- so a test failure cannot be blamed on which objects were
    implicitly picked.
  * `hub.free_air` is seeded with a known, fresh, withdrawn, contact-mode
    baseline BEFORE every write, so `free_air_v`/`free_air_ts` in the record
    can be checked for EQUALITY, not just presence.
  * `output_a_cmd_v` is checked for CONTENT: two objects at two different
    voltages (`_VOLT_A`, `_VOLT_B`) make a trivial constant/placeholder array
    fail, and the expected value at each is reconstructed independently as
    `voltage / amp_gain` (`web/edit-cockpit.html`'s own description of what
    `amp_gain` means: "the DAC drives V_tip / gain so the tip sees V_tip
    after the external amplifier" -- section 10's `check_write` computes the
    same "object voltage" quantity for a plain, non-chiral object).

Written against `client` (tests/conftest.py's in-process TestClient) the same
way tests/test_behavior_litho.py drives a write, so these run as plain unit
tests -- no browser, no subprocess service. Being in-process also means each
test has a live `hub` handle, which is used to seed `hub.free_air` directly
(there is no HTTP door for that in section 10 -- a free-air baseline is a
side effect of a withdrawn telemetry read, not a settable field) and to
mutate `LithoObject.voltage` directly for the "changing the voltage changes
the record" check.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import pytest

# `client` comes from tests/conftest.py.


@pytest.fixture(autouse=True)
def _isolate_instrument_model(tmp_path, monkeypatch):
    """Self-containment for the process-global instrument model and its on-disk
    config. `client` is module-scoped and `hub` is a process global, and an accepted
    override PERSISTS to `<runtime>/instrument.toml`
    (docs/INSTRUMENT_MODEL_SPEC.md sec 1). Without this, the override test would leak
    `tip_amp_gain = 17.5` into every later test AND across runs. Redirect the config
    at a per-test temp path (monkeypatch reverts the env afterwards) and reset
    `hub.instrument_model` to the fresh twin defaults so each test starts from the
    documented gain."""
    from afm_litho.service.app import hub
    from afm_litho.service.instrument_model import load_model, overrides_log_path
    cfg = tmp_path / "instrument.toml"
    monkeypatch.setenv("AFM_LITHO_INSTRUMENT_CONFIG", str(cfg))
    hub.host.instrument_model = load_model(platform="twin", startup_cal=None,
                                      config_path=cfg,
                                      overrides_log=overrides_log_path(cfg))
    yield


_VOLT_A = 6.0
_VOLT_B = 10.0
_SPEED_UM_S = 2.0        # slow enough that a mid-run abort is easy to land
_SETPOINT_REL = 0.42
# The twin instrument model's tip_amp_gain (service/instrument_model.py:
# `_TWIN_TIP_AMP_GAIN`). The command guard REFUSES any write whose amp_gain does
# not equal the model's (INSTRUMENT_MODEL_SPEC.md sec 2, "amp_gain must be present
# and equal to model.tip_amp_gain"; sec 13, "The guard's scope is not loosened for
# the twin or for tests"), so a twin write must present exactly this value or be
# refused `amp_gain_mismatch` before any record is written. The record's INTENT --
# amp_gain persisted, output_a reconstructing to voltage/amp_gain -- is unchanged:
# 6/25 and 10/25 are still two distinct, independently-checkable constants.
_AMP_GAIN = 25.0

_RECORD_KEYS = {"amp_gain", "setpoint", "free_air_v", "free_air_ts",
                "model_hash", "aborted"}


def _tmp_store(tmp_path):
    from afm_litho.litho.compiler import Pattern
    from afm_litho.service.app import hub
    from afm_litho.service.sessions import SessionStore
    hub.store = SessionStore(tmp_path / "s.jsonl")
    # Self-containment: `client` is module-scoped but `hub` is a process global
    # (tests/conftest.py: "individual tests must still be self-contained"), so the
    # design added by a previous test -- in this file or an earlier one -- would
    # otherwise leak into this test's write selection and /api/design/add naming.
    hub.pattern = Pattern(name="sketch", objects=[])
    hub.host.written_objects = set()
    return hub


@dataclass(frozen=True)
class _FreeAirStub:
    """Stand-in for section 10's `service/command_guard.py:FreeAir`, which
    does not exist yet. Same field names, so a test written against this
    keeps working unchanged once the real dataclass lands (only the import
    below needs to stop falling back)."""
    value_v: float
    ts: float
    mode: str
    withdrawn: bool


_FREE_AIR_V = 0.015


def _seed_free_air(hub, value_v: float = _FREE_AIR_V, mode: str = "contact",
                   withdrawn: bool = True, age_s: float = 0.0):
    """A known, fresh, withdrawn, contact-mode free-air baseline, seeded
    directly on `hub` -- section 10 gives no HTTP door for this (it is the
    side effect of a withdrawn telemetry read), and this file's tests are
    in-process so `hub` is directly reachable."""
    try:
        from afm_litho.service.command_guard import FreeAir
        fa = FreeAir(value_v=value_v, ts=time.time() - age_s, mode=mode,
                     withdrawn=withdrawn)
    except ImportError:
        fa = _FreeAirStub(value_v=value_v, ts=time.time() - age_s, mode=mode,
                          withdrawn=withdrawn)
    hub.host.free_air = fa
    return fa


def _two_voltage_objects(client) -> list[str]:
    """Two straight, single-voltage paths -- no fill, no scan-frame placement
    needed (`/api/design/add` does not call `design_import.
    resolve_import_size_um`) -- so `output_a_cmd_v` has exactly two expected
    constants to find, `voltage / amp_gain` each, and changing one object's
    voltage has one, unambiguous, independently-checkable effect."""
    r1 = client.post("/api/design/add", json={
        "kind": "path", "name": "wireA", "voltage": _VOLT_A,
        "speed_um_s": _SPEED_UM_S, "points_um": [[2.0, 2.0], [22.0, 2.0]]})
    assert r1.json().get("ok"), r1.json()
    r2 = client.post("/api/design/add", json={
        "kind": "path", "name": "wireB", "voltage": _VOLT_B,
        "speed_um_s": _SPEED_UM_S, "points_um": [[2.0, 6.0], [22.0, 6.0]]})
    assert r2.json().get("ok"), r2.json()
    return ["wireA", "wireB"]


def _wait(cond, timeout=60.0, interval=0.1):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return False


def _n_litho(hub):
    return len([s for s in hub.store.list() if s["kind"] == "litho"])


def _last_litho_record(hub):
    # `SessionStore.list()` returns records NEWEST-FIRST (service/sessions.py:
    # `out[::-1]`), so the most recent litho record is `recs[0]`, not `recs[-1]`.
    # `recs[-1]` is the OLDEST -- harmless for a single-write test (one record), but
    # it silently returns the FIRST write's record in the two multi-write tests
    # (changing-voltage, model-hash-after-override), which is exactly what those
    # assertions are trying to distinguish from the second write.
    recs = [s for s in hub.store.list() if s["kind"] == "litho"]
    assert recs, "no litho session record was created"
    return recs[0]


def _stem_for(tmp_path, hub, rec_id):
    """The stem the write's files were saved under, via the run_id -> stem
    index `service/stems.py` owns (registered under the SESSION record id,
    `host/litho_worker.py`'s `stems.register(runtime, rec["id"], "litho", stem)`)."""
    from afm_litho.service import stems
    info = stems.lookup(tmp_path, rec_id)
    assert info is not None, (
        f"no stem registered for litho session {rec_id!r} in "
        f"{tmp_path / 'stem_index.jsonl'}")
    return info["stem"]


def _sidecar_path(tmp_path, stem):
    # Same convention as the scan sidecar (service/stems.py docstring:
    # "npz, gwy, ibw, png, json sidecar" all share one stem under
    # <kind_dir>/<fmt>/<stem>.<fmt>), extended to litho -- spec sec 5 asks for
    # a litho sidecar that does not exist today ("No PNG/sidecar for a litho
    # pass", host/litho_worker.py comment).
    return tmp_path / "writes" / "json" / f"{stem}.json"


def _npz_path(tmp_path, stem):
    return tmp_path / "writes" / "npz" / f"{stem}.npz"


def _load_sidecar(tmp_path, hub, rec_id):
    stem = _stem_for(tmp_path, hub, rec_id)
    npz_path = _npz_path(tmp_path, stem)
    sidecar_path = _sidecar_path(tmp_path, stem)
    assert npz_path.is_file(), f"no npz saved for the write at {npz_path}"
    assert sidecar_path.is_file(), f"no litho sidecar at {sidecar_path}"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    missing = _RECORD_KEYS - set(sidecar)
    assert not missing, f"sidecar {sidecar_path} is missing keys {missing}: {sidecar}"
    npz = np.load(npz_path)
    assert "output_a_cmd_v" in npz.files, (
        f"npz {npz_path} has no output_a_cmd_v array (has {npz.files})")
    return sidecar, npz


def test_a_completed_twin_write_produces_npz_and_sidecar_with_gain_and_freeair_keys(
        client, tmp_path):
    """Spec sec 5: "Every write record (aborted or not) persists amp_gain,
    setpoint, free_air_v, free_air_ts, the model snapshot hash, and the
    commanded Output.A per point." This must hold on the TWIN too -- sec 6
    says the twin is where everything is tested before Thursday's hardware
    run -- but `_litho_write_body`'s simulation branch does not even parse
    `setpoint`/`amp_gain` out of the body today, and never populates
    `hw_litho_capture` (hardware-only), so today's twin write persists no
    npz/sidecar for litho at all.

    `free_air_v`/`free_air_ts` are checked for EQUALITY against the seeded
    baseline, not merely present -- a record that always writes `None` or
    some placeholder there must fail.
    """
    hub = _tmp_store(tmp_path)
    fa = _seed_free_air(hub)
    ids = _two_voltage_objects(client)
    hub.host.written_objects = set()
    n0 = _n_litho(hub)

    r = client.post("/api/litho/write", json={
        "objects": ids, "setpoint_relative": _SETPOINT_REL, "amp_gain": _AMP_GAIN})
    body = r.json()
    assert body.get("ok"), body
    assert _wait(lambda: _n_litho(hub) > n0), "write never completed"

    rec = _last_litho_record(hub)
    sidecar, npz = _load_sidecar(tmp_path, hub, rec["id"])

    assert sidecar["amp_gain"] == pytest.approx(_AMP_GAIN)
    assert sidecar["aborted"] is False
    assert sidecar["free_air_v"] == pytest.approx(fa.value_v)
    assert sidecar["free_air_ts"] == pytest.approx(fa.ts)
    # "setpoint" (sec 5's persisted key, unchanged by the sec-4/10 relative-box
    # correction) is the EFFECTIVE/absolute setpoint the write ran at -- the
    # same free_air + relative convention `hub.relative_setpoint` already uses
    # elsewhere in this codebase (service/hub.py: "effective per-frame
    # setpoint = free_air + the CURRENT scan setpoint"). Flagged as an
    # inference: section 10 does not spell out this record key's exact value.
    assert sidecar["setpoint"] == pytest.approx(fa.value_v + _SETPOINT_REL)

    arr = np.asarray(npz["output_a_cmd_v"], dtype=float)
    assert arr.size > 0, "output_a_cmd_v is empty on a completed write"
    assert np.all(np.isfinite(arr)), "output_a_cmd_v contains non-finite values"
    uniq = sorted({round(float(v), 9) for v in arr})
    assert len(uniq) >= 2, (
        f"output_a_cmd_v is constant ({uniq}) across two differently-voltaged "
        f"objects -- looks like a placeholder, not a real per-point capture")
    expected_a = _VOLT_A / _AMP_GAIN
    expected_b = _VOLT_B / _AMP_GAIN
    assert any(abs(v - expected_a) < 1e-9 for v in uniq), (
        f"no value in output_a_cmd_v matches wireA's expected "
        f"{_VOLT_A}/{_AMP_GAIN} = {expected_a}: {uniq}")
    assert any(abs(v - expected_b) < 1e-9 for v in uniq), (
        f"no value in output_a_cmd_v matches wireB's expected "
        f"{_VOLT_B}/{_AMP_GAIN} = {expected_b}: {uniq}")

    # Section 10: `model_hash` equals `hub.instrument_model.snapshot_hash()` --
    # NOT `hub.instrument`, which is the pre-existing twin physics/calibration
    # object (service/hub.py: `self.instrument = Instrument.from_note(...)`,
    # `twin/physics.py:Instrument`). The coordinator's 2026-09-22 correction
    # renamed the new field to `instrument_model` for exactly this collision.
    assert hasattr(hub.host, "instrument_model"), (
        "hub.instrument_model (the section-10 InstrumentModel) does not exist; "
        "hub.instrument is the pre-existing twin physics calibration object "
        "and must not be reused for this")
    assert sidecar["model_hash"] == hub.host.instrument_model.snapshot_hash()


def test_changing_an_object_voltage_changes_its_commanded_output_a(client, tmp_path):
    """A second, independent write of the SAME object at a DIFFERENT voltage
    must land a different `output_a_cmd_v` -- a hash or a placeholder that
    happens to look non-constant with two objects but is actually a fixed
    per-object index rather than a real voltage/gain reconstruction must
    fail this."""
    hub = _tmp_store(tmp_path)
    _seed_free_air(hub)
    _two_voltage_objects(client)
    hub.host.written_objects = set()
    n0 = _n_litho(hub)

    r = client.post("/api/litho/write", json={
        "objects": ["wireA"], "setpoint_relative": _SETPOINT_REL, "amp_gain": _AMP_GAIN})
    assert r.json().get("ok"), r.json()
    assert _wait(lambda: _n_litho(hub) > n0), "first write never completed"
    rec1 = _last_litho_record(hub)
    _sidecar1, npz1 = _load_sidecar(tmp_path, hub, rec1["id"])
    arr1 = np.asarray(npz1["output_a_cmd_v"], dtype=float)
    assert arr1.size > 0

    # Mutate the object's voltage directly (no props-setter door is named in
    # section 10; this is in-process test setup, the same pattern as seeding
    # `hub.written_objects` elsewhere in this file).
    wire_a = next(o for o in hub.pattern.objects if o.name == "wireA")
    new_voltage = _VOLT_A + 4.0
    wire_a.voltage = new_voltage

    hub.host.written_objects = set()
    n1 = _n_litho(hub)
    r = client.post("/api/litho/write", json={
        "objects": ["wireA"], "setpoint_relative": _SETPOINT_REL, "amp_gain": _AMP_GAIN})
    assert r.json().get("ok"), r.json()
    assert _wait(lambda: _n_litho(hub) > n1), "second write never completed"
    rec2 = _last_litho_record(hub)
    _sidecar2, npz2 = _load_sidecar(tmp_path, hub, rec2["id"])
    arr2 = np.asarray(npz2["output_a_cmd_v"], dtype=float)
    assert arr2.size > 0

    expected1 = _VOLT_A / _AMP_GAIN
    expected2 = new_voltage / _AMP_GAIN
    # one stroke each: every sample at the object's bias, then the 0 V end sample
    assert arr1[-1] == 0.0 and arr2[-1] == 0.0
    assert np.allclose(arr1[:-1], expected1, atol=1e-9), (
        f"first write's output_a_cmd_v does not reconstruct to {expected1}: "
        f"{sorted(set(np.round(arr1, 9)))}")
    assert np.allclose(arr2[:-1], expected2, atol=1e-9), (
        f"second write's output_a_cmd_v does not reconstruct to {expected2}: "
        f"{sorted(set(np.round(arr2, 9)))}")
    assert not np.allclose(arr1, arr2), (
        "changing wireA's voltage did not change its commanded output_a_cmd_v")


def test_model_hash_in_sidecar_changes_after_an_instrument_override(client, tmp_path):
    """Section 10: `POST /api/instrument/override {field, value, reason,
    confirm}` mutates the live `InstrumentModel`; a record's `model_hash`
    that stays constant across an accepted override would mean the hash is
    not actually a snapshot of the model, which defeats its purpose (section
    1: overrides are meant to be visible and auditable)."""
    hub = _tmp_store(tmp_path)
    _seed_free_air(hub)
    ids = _two_voltage_objects(client)
    hub.host.written_objects = set()
    n0 = _n_litho(hub)

    r = client.post("/api/litho/write", json={
        "objects": ids, "setpoint_relative": _SETPOINT_REL, "amp_gain": _AMP_GAIN})
    assert r.json().get("ok"), r.json()
    assert _wait(lambda: _n_litho(hub) > n0), "first write never completed"
    rec1 = _last_litho_record(hub)
    sidecar1, _ = _load_sidecar(tmp_path, hub, rec1["id"])
    hash1 = sidecar1["model_hash"]

    ov = client.post("/api/instrument/override", json={
        "field": "tip_amp_gain", "value": 17.5, "reason": "test", "confirm": True})
    assert ov.status_code == 200 and ov.json().get("applied"), (
        f"POST /api/instrument/override did not apply: "
        f"{ov.status_code} {ov.text}")

    hub.host.written_objects = set()
    n1 = _n_litho(hub)
    # The override changed the model's tip_amp_gain to 17.5, so the second write
    # must present 17.5 to clear the guard (see _AMP_GAIN above): a mismatch would
    # be refused before any record is written. The assertion under test is that the
    # model_hash CHANGED across the override, which it does regardless of gain value.
    r = client.post("/api/litho/write", json={
        "objects": ids, "setpoint_relative": _SETPOINT_REL, "amp_gain": 17.5})
    assert r.json().get("ok"), r.json()
    assert _wait(lambda: _n_litho(hub) > n1), "second write never completed"
    rec2 = _last_litho_record(hub)
    sidecar2, _ = _load_sidecar(tmp_path, hub, rec2["id"])
    hash2 = sidecar2["model_hash"]

    assert hash2 != hash1, (
        "model_hash did not change after an accepted instrument override -- "
        "it is not a real snapshot of the model")


def test_an_aborted_twin_write_still_records_aborted_true_with_partial_capture(
        client, tmp_path):
    """Spec sec 5: "On abort/exception the worker reads back the in-waves
    captured so far and writes the record with aborted: true before
    withdrawing." An abort landed mid-run must not skip the record, and the
    record must say so plus keep whatever gain/setpoint/free-air context the
    write was armed with.

    The abort is timed deterministically rather than polled off an internal
    hub attribute: no live per-point progress counter is named anywhere in
    section 10, and `hub._litho_partial`/`written_accum` are only populated
    AFTER an abort lands (host/litho_worker.py), so they cannot be polled
    beforehand. `wireA` is 20 um at `_SPEED_UM_S` = 2 um/s -> a 10 s stroke;
    waiting a third of that guarantees the abort lands mid-stroke, not before
    the first point or after the last.
    """
    hub = _tmp_store(tmp_path)
    fa = _seed_free_air(hub)
    ids = _two_voltage_objects(client)
    hub.host.written_objects = set()
    n0 = _n_litho(hub)

    # played in real time (`twin_speed` 1; the suite's default is unlimited), so
    # the sleep below is a third of the stroke on any machine
    r = client.post("/api/litho/write", json={
        "objects": ids, "setpoint_relative": _SETPOINT_REL, "amp_gain": _AMP_GAIN,
        "twin_speed": 1.0})
    assert r.json().get("ok"), r.json()
    # the twin first moves to the wire and engages; abort once the tip is writing
    assert _wait(lambda: hub.host._twin_plugin is not None
                 and hub.host._twin_plugin._activity == "writing")
    time.sleep((22.0 - 2.0) / _SPEED_UM_S / 3.0)     # ~a third of wireA's stroke
    client.post("/api/litho/abort")
    assert _wait(lambda: _n_litho(hub) > n0), "aborted write never closed its record"

    rec = _last_litho_record(hub)
    sidecar, npz = _load_sidecar(tmp_path, hub, rec["id"])
    assert sidecar["aborted"] is True
    assert sidecar["amp_gain"] == pytest.approx(_AMP_GAIN)
    assert sidecar["free_air_v"] == pytest.approx(fa.value_v)
    assert sidecar["free_air_ts"] == pytest.approx(fa.ts)

    arr = np.asarray(npz["output_a_cmd_v"], dtype=float)
    # "whatever capture was read back" -- an abort landed mid-run, so the
    # partial capture should be shorter than a full run's, but it must not be
    # silently empty (that would mean the abort path dropped it on the floor).
    assert arr.size > 0, (
        "an aborted write recorded zero commanded points -- the partial "
        "capture was dropped")
    assert np.all(np.isfinite(arr))


def test_hub_last_write_params_setpoint_relative_is_updated_after_a_write(
        client, tmp_path):
    """Section 10 (as corrected, instrument-model 12a0593 sec 4+10):
    `hub.last_write_params` carries `setpoint_relative` (the write door's
    body key), not a bare `setpoint` -- the write setpoint is RELATIVE to
    free air. No dedicated read endpoint for this is named in section 10
    (only `GET /api/instrument` for the model itself) -- asserting on hub
    state directly per the brief's fallback instruction.
    """
    hub = _tmp_store(tmp_path)
    _seed_free_air(hub)
    ids = _two_voltage_objects(client)
    hub.host.written_objects = set()
    n0 = _n_litho(hub)

    r = client.post("/api/litho/write", json={
        "objects": ids, "setpoint_relative": _SETPOINT_REL, "amp_gain": _AMP_GAIN})
    assert r.json().get("ok"), r.json()
    assert _wait(lambda: _n_litho(hub) > n0), "write never completed"

    assert hasattr(hub.host, "last_write_params"), (
        "hub.last_write_params does not exist (spec sec 10)")
    assert hub.host.last_write_params.get("setpoint_relative") == pytest.approx(_SETPOINT_REL)
