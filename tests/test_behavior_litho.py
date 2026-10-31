"""End-to-end lithography BEHAVIOUR through the API: not "the endpoint responds"
but "the system does the right thing" -- writes deduplicate, material lands where
it was drawn, and the transit policy round-trips as a real choice.
"""

from __future__ import annotations

import time

from _im_fixtures_guard import GUARD_WRITE_KW, seed_free_air
from _scan_frame import run_tiny_scan

# `client` comes from tests/conftest.py.

_RECT = ('<svg xmlns="http://www.w3.org/2000/svg" width="40000" height="40000" '
         'viewBox="0 0 40000 40000">'
         '<rect x="16000" y="16000" width="8000" height="8000" fill="black"/></svg>')


def _tmp_store(tmp_path):
    from afm_litho.service.app import hub
    from afm_litho.service.sessions import SessionStore
    hub.store = SessionStore(tmp_path / "s.jsonl")
    return hub


def _load_design(client, svg=_RECT, spacing=2.0):
    run_tiny_scan(client)          # a design is placed against a scan frame
    r = client.post(f"/api/design/svg?name=b.svg&fill_spacing_um={spacing}",
                    content=svg, headers={"content-type": "image/svg+xml"})
    assert r.json().get("ok")


def _wait(cond, timeout=30.0, interval=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return False


def _n_litho(hub):
    return len([s for s in hub.store.list() if s["kind"] == "litho"])


def _enabled_ids(hub):
    """Object ids of every enabled object in the loaded design (the explicit
    REST shape that repeats a write regardless of the written set)."""
    return [o.name for o in hub.pattern.objects if getattr(o, "enabled", True)]


def _write_to_completion(client, hub, body=None):
    """Fire a REST write and wait for it to record. Seeds the command guard's
    free-air baseline and carries `amp_gain` (INSTRUMENT_MODEL_SPEC secs 2/10,
    13); with no `body` the default write covers everything unwritten."""
    n0 = _n_litho(hub)
    seed_free_air(hub)
    j = client.post("/api/litho/write",
                    json={"setpoint_relative": 0.2, **GUARD_WRITE_KW, **(body or {})}
                    ).json()
    assert _wait(lambda: _n_litho(hub) > n0), "write never completed"
    return j


def test_a_write_deduplicates_already_written_objects(client, tmp_path):
    hub = _tmp_store(tmp_path)
    _load_design(client)
    hub.host.written_objects = set()
    first = _write_to_completion(client, hub)
    assert first["ok"] and first["writing"] >= 1
    # everything is now written; a second write finds nothing new to do
    second = client.post("/api/litho/write").json()
    assert second.get("reason") == "already_written"
    assert "error" in second


def test_write_all_deliberately_repeats_everything(client, tmp_path):
    # The old "Write all" cockpit shape was `{"all": true}`; under the command
    # guard the REST door takes an explicit id list instead (spec sec 4/10,
    # migration sec 13). Naming the objects re-writes them "whether or not they
    # have been written before" (select_objects), which is exactly the deliberate
    # repeat this test asserts -- the behaviour survives, the wire shape changes.
    hub = _tmp_store(tmp_path)
    _load_design(client)
    hub.host.written_objects = set()
    _write_to_completion(client, hub)
    seed_free_air(hub)
    forced = client.post(
        "/api/litho/write",
        json={"objects": _enabled_ids(hub), "setpoint_relative": 0.2,
              **GUARD_WRITE_KW}).json()
    assert forced.get("ok") and forced["writing"] >= 1   # re-writes despite the record


def test_a_write_draws_the_wire_into_the_twins_sample(client, tmp_path):
    """Writing is not just a state change: the wire is drawn into the twin's sample, where
    it stays for later scans (2026-10-27: wires show as a height overlay, no dose)."""
    from afm_litho.service.twin_plugin import plugin
    hub = _tmp_store(tmp_path)
    _load_design(client)
    hub.host.written_objects = set()
    plugin(hub).sample.overlay[:] = False
    _write_to_completion(client, hub)
    assert plugin(hub).sample.overlay.any(), "the write drew nothing into the sample"


def test_transit_policy_is_a_real_choice_that_round_trips(client):
    _load_design(client)                          # a pattern to plan transits over
    for mode in ("lift", "slew", "contour"):
        client.post("/api/transit", json={"mode": mode, "speed_um_s": 12.0})
        got = client.get("/api/transit").json()
        assert got["policy"]["mode"] == mode
        assert got["policy"]["speed_um_s"] == 12.0
        assert mode in got["cost"]                # a cost is planned per mode
    client.post("/api/transit", json={"mode": "lift", "speed_um_s": 20.0})   # restore
