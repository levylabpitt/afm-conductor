"""HTTP-layer integration tests for the instrument-control surface.

Companion to test_api.py: those cover design/svg, litho write+scan streaming and
persistence; these drive the currently-untested control endpoints -- speed,
approach/withdraw, scan/abort, the force/tune measurements, litho load/frame/
written/abort, the transit policy.

`client` comes from tests/conftest.py (a TestClient). The
underlying `hub` is a process global, so every test here is written to be
self-contained and order-independent: anything that starts a scan aborts it, and
nothing depends on which sample or pattern a previous test left behind.
"""

from __future__ import annotations


# `client` comes from tests/conftest.py (a TestClient).

# A tiny scan config: 16 px x 4 lines at 200 Hz finishes almost immediately, so
# a test can start a scan, exercise the running-scan path and abort it without
# leaving the background pump churning.
_TINY_SCAN = {"size_um": 5.0, "pixels": 16, "lines": 4,
              "line_rate_hz": 200.0, "mode": "contact"}


# --------------------------------------------------------------- speed --

def test_speed_sets_and_reflects_the_factor(client):
    assert client.post("/api/speed/4").json() == {"speed": 4.0}
    # clamped to the 0.25..64 window at both ends
    assert client.post("/api/speed/1000").json()["speed"] == 64.0
    assert client.post("/api/speed/0.01").json()["speed"] == 0.25
    client.post("/api/speed/1")     # leave it at the default for other tests


# ----------------------------------------------------- approach / withdraw --

def test_approach_engages_and_returns_a_snapshot(client):
    # B1 migration (docs/INSTRUMENT_MODEL_SPEC.md secs 2/10/13): approach now runs
    # check_approach, which needs a fresh withdrawn free-air baseline.
    from _im_fixtures_guard import seed_free_air

    from afm_litho.service.hub import hub
    seed_free_air(hub)
    j = client.post("/api/approach").json()
    # snapshot contract from ARC.snapshot(): position, deflection, engaged flag
    for k in ("z_um", "defl_v", "bias_v", "engaged", "x_um", "y_um"):
        assert k in j, (k, j)
    assert j["engaged"] is True
    client.post("/api/withdraw")    # park the tip again


def test_withdraw_returns_a_snapshot(client):
    j = client.post("/api/withdraw").json()
    for k in ("z_um", "defl_v", "bias_v", "engaged"):
        assert k in j, (k, j)


# --------------------------------------------------------------- scan/abort --

def test_scan_abort_always_retracts(client):
    # Every Stop retracts now (docs/STOP_RETRACT_SPEC.md): the `withdraw` request
    # field is vestigial, so the response reports withdraw True regardless of the
    # body -- there is no stop-and-hold path any more.
    assert client.post("/api/scan/abort", json={"withdraw": False}).json() \
        == {"ok": True, "withdraw": True}
    assert client.post("/api/scan/abort", json={"withdraw": True}).json() \
        == {"ok": True, "withdraw": True}
    assert client.post("/api/scan/abort").json() == {"ok": True, "withdraw": True}


def test_scan_abort_stops_a_running_scan(client):
    started = client.post("/api/scan/start", json=_TINY_SCAN).json()
    # start may race the previous test's scan; either way we then abort cleanly
    if not started.get("ok"):
        assert "already" in started.get("error", "")
    j = client.post("/api/scan/abort", json={"withdraw": False}).json()
    # always-retract: the response reports withdraw True even for a false body.
    assert j["ok"] is True and j["withdraw"] is True


# ----------------------------------------------------------- curves / tune --

def test_force_curve_tune_and_sweep_are_not_available_on_the_twin(client):
    """The twin plugin has none of them (2026-10-27); they come back once the
    architecture is transferred (docs/TECH_DEBT.md)."""
    for path, what in (("/api/force_curve", "a force curve"),
                       ("/api/tune/thermal", "a thermal tune"),
                       ("/api/tune/sweep", "a frequency sweep")):
        assert client.post(path).json() == {"error": f"{what} is not available on the twin"}


# --------------------------------------------------------------- litho/load --

def test_litho_load_auto_routes_a_pattern(client):
    j = client.post("/api/litho/load", json={"source": "auto"}).json()
    assert "name" in j and isinstance(j["objects"], list)
    assert len(j["objects"]) >= 1                 # the router wires the contacts
    o = j["objects"][0]
    assert "name" in o and "points_um" in o and "voltage" in o


def test_litho_load_missing_oasis_file_returns_a_clean_error(client):
    # A missing design name is a user error, not a server crash: it must come
    # back as a JSON error, not a 500.
    r = client.post("/api/litho/load", json={"source": "does-not-exist.oas"})
    assert r.status_code == 200
    assert "no such design" in r.json().get("error", "").lower()


def test_litho_load_rejects_a_path_traversal_name(client):
    # A name that climbs out of data/designs must never read a file there.
    r = client.post("/api/litho/load", json={"source": "../../pyproject.toml"})
    assert "error" in r.json()


# ------------------------------------------------------------ litho/written --

def test_litho_written_returns_a_conductance_map(client):
    j = client.get("/api/litho/written",
                   params={"size_um": 20.0, "x_off": 0.0, "y_off": 0.0,
                           "n": 32}).json()
    assert j["n"] == 32
    assert isinstance(j["g"], list) and len(j["g"]) == 32 * 32
    assert all(0.0 <= v <= 1.0 for v in j["g"])   # g = d^2/(1+d^2) is a 0..1 occupancy


# -------------------------------------------------------------- litho/frame --

def test_litho_frame_round_trips_a_pinned_angle(client):
    body = {"fast_size_um": 12.0, "slow_size_um": 8.0,
            "fast_offset_um": 1.5, "slow_offset_um": -2.0, "angle_deg": 30.0}
    set_j = client.post("/api/litho/frame", json=body).json()
    assert set_j["angle_follows_scan"] is False
    f = set_j["frame"]
    assert f["fast_size_um"] == 12.0 and f["slow_size_um"] == 8.0
    assert f["fast_offset_um"] == 1.5 and f["slow_offset_um"] == -2.0
    assert f["angle_deg"] == 30.0
    # GET reflects the same pinned frame
    get_j = client.get("/api/litho/frame").json()
    assert get_j["angle_follows_scan"] is False
    assert get_j["frame"]["angle_deg"] == 30.0
    assert get_j["frame"]["fast_size_um"] == 12.0


def test_litho_frame_null_angle_follows_the_scan(client):
    # Hand the angle back to the scan. With no scan running the effective angle
    # falls back to 0, and the flag reports it now follows the scan.
    client.post("/api/scan/abort", json={"withdraw": False})   # ensure no scan
    j = client.post("/api/litho/frame", json={"angle_deg": None}).json()
    assert j["angle_follows_scan"] is True
    assert j["frame"]["angle_deg"] == 0.0
    assert client.get("/api/litho/frame").json()["angle_follows_scan"] is True


# -------------------------------------------------------------- litho/abort --

def test_litho_abort_is_ok(client):
    # Fix 4: the litho ABORT now routes through the unified safe_park (stop +
    # retract), so it is backend-tagged like the operator Abort. Twin backend here.
    assert client.post("/api/litho/abort").json() == {"ok": True, "backend": "twin"}


# ----------------------------------------------------------------- transit --

def test_transit_policy_round_trips(client):
    body = {"mode": "slew", "speed_um_s": 15.0, "max_speed_um_s": 30.0,
            "lift_nm": 200.0, "settle_s": 0.05}
    set_j = client.post("/api/transit", json=body).json()
    p = set_j["policy"]
    assert p["mode"] == "slew" and p["speed_um_s"] == 15.0
    assert p["max_speed_um_s"] == 30.0 and p["lift_nm"] == 200.0
    assert p["settle_s"] == 0.05
    # GET returns the same policy plus the mode list and (optional) cost report
    get_j = client.get("/api/transit").json()
    assert get_j["policy"]["mode"] == "slew"
    assert get_j["policy"]["speed_um_s"] == 15.0
    assert "modes" in get_j and "slew" in get_j["modes"]
    # restore the default so a later transit test is not surprised by "slew"
    client.post("/api/transit", json={"mode": "lift"})


def test_transit_rejects_an_invalid_mode(client):
    j = client.post("/api/transit", json={"mode": "teleport"}).json()
    assert "error" in j and "mode must be one of" in j["error"]
