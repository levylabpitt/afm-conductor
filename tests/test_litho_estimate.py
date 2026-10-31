"""RED (wave-1 tests-only) coverage for the coordinator's instrument-model
ce30dfa addition (sec 4+10): `GET /api/litho/estimate` ->
`{"total_s", "objects": [{id, seconds, length_um, speed_um_s, enabled}]}`,
computed by the service from ITS OWN enabled set (not whatever a client
last told it).

Does not exist yet: no `/api/litho/estimate` route anywhere in
`service/routers/litho.py`, and no `GET`/`POST` handler in `routers/design.py`
either.

Written against `client` (tests/conftest.py's in-process TestClient), same
style as tests/test_write_records.py: two straight, known-length,
known-speed paths so `length_um / speed_um_s` is independently computable,
plus a generous overhead allowance for whatever transit/settle time the
service's estimate may add on top of pure travel time.
"""

from __future__ import annotations

import pytest

# `client` comes from tests/conftest.py.

_LEN_A_UM = 20.0
_SPEED_A_UM_S = 2.0
_LEN_B_UM = 10.0
_SPEED_B_UM_S = 5.0
# Generous: the estimate may reasonably add transit/settle overhead on top of
# pure length/speed travel time; this only bounds it from blowing up, it does
# not pretend to know the service's exact transit policy.
_OVERHEAD_TOL_S = 5.0


def _tmp_store(tmp_path):
    from afm_litho.litho.compiler import Pattern
    from afm_litho.service.app import hub
    from afm_litho.service.sessions import SessionStore
    hub.store = SessionStore(tmp_path / "s.jsonl")
    # Self-containment: `client` is module-scoped but `hub` is a process global
    # (tests/conftest.py: "individual tests must still be self-contained"), so the
    # design added by a previous test -- in this file or an earlier one -- would
    # otherwise leak into this one's enabled set and its /api/design/add naming
    # (wireA -> wireA-2). Reset the working design so each test's estimate is over
    # exactly the two objects it adds.
    hub.pattern = Pattern(name="sketch", objects=[])
    hub.host.written_objects = set()
    return hub


def _two_objects(client) -> tuple[int, int]:
    r1 = client.post("/api/design/add", json={
        "kind": "path", "name": "wireA", "voltage": 6.0,
        "speed_um_s": _SPEED_A_UM_S,
        "points_um": [[2.0, 2.0], [2.0 + _LEN_A_UM, 2.0]]})
    assert r1.json().get("ok"), r1.json()
    r2 = client.post("/api/design/add", json={
        "kind": "path", "name": "wireB", "voltage": 10.0,
        "speed_um_s": _SPEED_B_UM_S,
        "points_um": [[2.0, 6.0], [2.0 + _LEN_B_UM, 6.0]]})
    assert r2.json().get("ok"), r2.json()
    return r1.json()["index"], r2.json()["index"]


def _by_id(objects: list[dict], name: str, idx: int) -> dict | None:
    """`id` isn't specified further than "id" in the coordinator's message --
    accept either the object's name or its list index, whichever the service
    ends up using."""
    for o in objects:
        if str(o.get("id")) in (name, str(idx)):
            return o
    return None


def test_estimate_matches_length_over_speed_and_sums_enabled_objects(client, tmp_path):
    _tmp_store(tmp_path)
    idx_a, idx_b = _two_objects(client)

    r = client.get("/api/litho/estimate")
    assert r.status_code == 200, r.text
    est = r.json()
    assert "total_s" in est and "objects" in est, est

    oa = _by_id(est["objects"], "wireA", idx_a)
    ob = _by_id(est["objects"], "wireB", idx_b)
    assert oa is not None, f"wireA not found in {est['objects']}"
    assert ob is not None, f"wireB not found in {est['objects']}"

    assert oa["length_um"] == pytest.approx(_LEN_A_UM, rel=0.05)
    assert oa["speed_um_s"] == pytest.approx(_SPEED_A_UM_S)
    assert ob["length_um"] == pytest.approx(_LEN_B_UM, rel=0.05)
    assert ob["speed_um_s"] == pytest.approx(_SPEED_B_UM_S)

    expected_a = _LEN_A_UM / _SPEED_A_UM_S
    expected_b = _LEN_B_UM / _SPEED_B_UM_S
    assert expected_a <= oa["seconds"] <= expected_a + _OVERHEAD_TOL_S, (
        f"wireA seconds {oa['seconds']} not in "
        f"[{expected_a}, {expected_a + _OVERHEAD_TOL_S}]")
    assert expected_b <= ob["seconds"] <= expected_b + _OVERHEAD_TOL_S, (
        f"wireB seconds {ob['seconds']} not in "
        f"[{expected_b}, {expected_b + _OVERHEAD_TOL_S}]")

    assert oa["enabled"] is True
    assert ob["enabled"] is True
    # 13.1 item 18: the per-object `seconds` is the pure TRAJECTORY time, and the
    # sum of the enabled objects' trajectory is reported as `trajectory_s`. `total_s`
    # is what the service will actually spend -- trajectory PLUS per-object
    # arm/withdraw/stroke/chain overhead PLUS a whole-run margin -- so it is STRICTLY
    # GREATER than the trajectory sum. The old assertion `total_s == sum(seconds)`
    # encoded the B2 contradiction the reviewer rejected: the published total omitted
    # the overhead the write-start estimate (`write_est_s`) charged, so the two
    # disagreed for the same design. They must now agree.
    assert "trajectory_s" in est, est
    # each number is rounded to 1 ms on its own, so the sum can differ by a rounding step
    assert est["trajectory_s"] == pytest.approx(oa["seconds"] + ob["seconds"],
                                                abs=2e-3), (
        "trajectory_s is not the sum of the two (both enabled) objects' seconds")
    assert est["total_s"] > est["trajectory_s"], (
        "total_s must include the per-object overhead the write-start estimate "
        f"charges, so it exceeds the pure trajectory sum: {est}")


def test_disabling_an_object_over_rest_drops_it_from_total_s(client, tmp_path):
    _tmp_store(tmp_path)
    idx_a, idx_b = _two_objects(client)

    r0 = client.get("/api/litho/estimate")
    assert r0.status_code == 200, r0.text
    est0 = r0.json()

    r = client.post("/api/design/op", json={
        "op": "set_props", "indices": [idx_a], "enabled": False})
    assert r.status_code == 200 and "error" not in r.json(), r.json()

    r1 = client.get("/api/litho/estimate")
    assert r1.status_code == 200, r1.text
    est1 = r1.json()

    oa1 = _by_id(est1["objects"], "wireA", idx_a)
    ob1 = _by_id(est1["objects"], "wireB", idx_b)
    assert oa1 is not None and oa1["enabled"] is False, oa1
    assert ob1 is not None and ob1["enabled"] is True, ob1

    assert est1["total_s"] < est0["total_s"], (
        f"total_s did not drop after disabling wireA: {est0['total_s']} -> "
        f"{est1['total_s']}")
    # 13.1 item 18: with wireA disabled, the pure trajectory sum is wireB's seconds
    # alone (disabled objects excluded from both totals). `total_s` adds wireB's
    # overhead + margin on top, so assert on `trajectory_s` here -- the old
    # `total_s == ob seconds` encoded the pre-fix contradiction (the published total
    # left out the overhead the write-start estimate charged).
    assert est1["trajectory_s"] == pytest.approx(ob1["seconds"]), (
        "trajectory_s with wireA disabled should equal wireB's seconds alone "
        f"(disabled objects must not still be summed): {est1}")
    assert est1["total_s"] > est1["trajectory_s"], est1


def test_estimate_reflects_enabled_set_live_not_a_stale_cached_one(client, tmp_path):
    """"computed by the service from ITS enabled set" -- calling the
    endpoint twice in a row, with a mutation in between, must not return the
    same total_s twice (a naive cache-at-load implementation would)."""
    _tmp_store(tmp_path)
    _idx_a, idx_b = _two_objects(client)

    r0 = client.get("/api/litho/estimate")
    assert r0.status_code == 200
    total0 = r0.json()["total_s"]

    client.post("/api/design/op", json={
        "op": "set_props", "indices": [idx_b], "enabled": False})
    r1 = client.get("/api/litho/estimate")
    assert r1.status_code == 200
    total1 = r1.json()["total_s"]

    assert total1 != total0, (
        "GET /api/litho/estimate returned the same total_s before and after "
        f"disabling an object: {total0}")


def test_estimate_total_agrees_with_the_write_start_estimator(client, tmp_path):
    """13.1 item 18: ONE estimator. `GET /api/litho/estimate`'s `total_s` must equal
    `write_est_s` (the number the write-start path sleeps on / logs) over the SAME
    enabled set. The B2 REJECT was exactly that these two disagreed -- the published
    total omitted the per-object arm/withdraw/stroke/chain overhead and the whole-run
    margin the write-start estimate charged."""
    from afm_litho.service.app import hub
    from afm_litho.service.host.litho_cmds import write_est_s

    _tmp_store(tmp_path)
    _two_objects(client)

    r = client.get("/api/litho/estimate")
    assert r.status_code == 200, r.text
    est = r.json()

    todo = [o for o in hub.pattern.objects if getattr(o, "enabled", True)]
    assert est["total_s"] == pytest.approx(write_est_s(hub, todo)), (
        f"GET total_s {est['total_s']} disagrees with write_est_s "
        f"{write_est_s(hub, todo)} over the same enabled set")
