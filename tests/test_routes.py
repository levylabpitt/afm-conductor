"""The route table, pinned. This is the safety net for splitting app.py into
domain routers: moving handlers between modules must not lose, rename, or
re-path a single endpoint. If this test fails after a refactor, a route moved.

FastAPI's auto-generated docs routes (/docs, /redoc, /openapi.json) are excluded
because they are the framework's, not ours.
"""

from __future__ import annotations


from starlette.routing import Mount, WebSocketRoute

EXPECTED = {
    ("/", "GET"),
    ("/api/approach", "POST"),
    ("/api/connect/ports", "GET"),
    ("/api/connect/ports/add", "POST"),
    ("/api/connect/ports/delete", "POST"),
    ("/api/connect/ports/replace", "POST"),
    ("/api/connect/ports/update", "POST"),
    ("/api/contacts", "GET"),
    ("/api/contacts", "POST"),
    ("/api/design", "GET"),
    ("/api/design/add", "POST"),
    ("/api/design/background", "POST"),
    ("/api/design/background/channels", "GET"),
    ("/api/design/bezier", "POST"),
    ("/api/design/bezier/{index}", "GET"),
    ("/api/design/capabilities", "GET"),
    ("/api/design/chiral", "POST"),
    ("/api/design/files", "GET"),
    ("/api/design/open", "POST"),
    ("/api/design/save", "POST"),
    ("/api/design/op", "POST"),
    ("/api/design/points", "POST"),
    ("/api/design/reorder", "POST"),
    ("/api/design/svg", "GET"),
    ("/api/design/svg", "POST"),
    ("/api/drift", "GET"),
    ("/api/drift/enable", "POST"),
    ("/edit-cockpit", "GET"),
    ("/api/drift/reference", "POST"),
    ("/api/fault/clear", "POST"),
    ("/api/force_curve", "POST"),
    ("/api/instrument", "GET"),
    ("/api/keeper/contact", "POST"),
    ("/api/instrument/open_config", "POST"),
    ("/api/instrument/override", "POST"),
    ("/api/instrument/reload", "POST"),
    ("/api/litho/abort", "POST"),
    ("/api/litho/demo", "GET"),
    ("/api/litho/estimate", "GET"),
    ("/api/litho/frame", "GET"),
    ("/api/litho/frame", "POST"),
    ("/api/litho/load", "POST"),
    ("/api/litho/toolpath", "POST"),
    ("/api/litho/write", "POST"),
    ("/api/litho/written", "GET"),
    ("/api/mode", "GET"),
    ("/api/mode", "POST"),
    ("/api/optics", "GET"),
    ("/api/park/last", "GET"),
    ("/api/pens", "GET"),
    ("/api/safe_park", "POST"),
    ("/api/sample/load", "POST"),
    ("/api/sample/upload", "POST"),
    ("/api/samples", "GET"),
    ("/api/scan/abort", "POST"),
    ("/api/scan/continuous", "GET"),
    ("/api/scan/continuous", "POST"),
    ("/api/scan/last", "GET"),
    ("/api/scan/params", "POST"),
    ("/api/scan/relative", "POST"),
    ("/api/scan/start", "POST"),
    ("/api/sessions", "GET"),
    ("/api/sessions/recall", "GET"),
    ("/api/shapes", "GET"),
    ("/api/shapes/{index}", "POST"),
    ("/api/shapes/{index}/toolpath", "GET"),
    ("/api/sketch", "POST"),
    ("/api/sketch/clear", "POST"),
    ("/api/sketch/undo", "POST"),
    ("/api/speed/{factor}", "POST"),
    ("/api/state", "GET"),
    ("/api/tip/show", "POST"),
    ("/api/transit", "GET"),
    ("/api/transit", "POST"),
    # single-writer arbitration (docs/FLEX_INTEGRATION_PLAN.md Sec E.2)
    ("/api/control", "GET"),
    ("/api/control/release", "POST"),
    ("/api/control/take", "POST"),
    ("/api/tune/sweep", "POST"),
    ("/api/tune/thermal", "POST"),
    ("/api/withdraw", "POST"),
    ("/api/zgain", "POST"),
    ("/ws", "WS"),
    ("/static", "MOUNT"),
}

_FRAMEWORK = {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}


def _actual_routes():
    from afm_litho.service.app import app
    out = set()
    # HTTP routes come from the OpenAPI schema, which is nesting-independent:
    # this FastAPI version hides include_router() routes behind an opaque
    # _IncludedRouter in app.routes, so iterating app.routes would miss every
    # endpoint that has been moved into a domain router. The schema sees them
    # all (verified: a moved route serves 200 and appears here).
    for path, ops in app.openapi()["paths"].items():
        for method in ops:
            out.add((path, method.upper()))
    # the websocket and the static mount are not HTTP paths, so read them off
    # the route table directly.
    for r in app.routes:
        if isinstance(r, WebSocketRoute):
            out.add((r.path, "WS"))
        elif isinstance(r, Mount):
            out.add((r.path, "MOUNT"))
    return {(p, m) for p, m in out if p not in _FRAMEWORK}


def test_route_table_is_exactly_as_expected():
    """No endpoint added, removed, renamed, or re-pathed. The master check for
    the app.py -> routers split."""
    actual = _actual_routes()
    missing = EXPECTED - actual
    added = actual - EXPECTED
    assert not missing, f"routes disappeared: {sorted(missing)}"
    assert not added, f"unexpected routes appeared: {sorted(added)}"


def test_the_app_boots_and_serves_state():
    """Startup runs and a basic read works -- catches an import/order break from
    moving code into routers (e.g. app created before its routers, or a helper that
    didn't come along)."""
    from fastapi.testclient import TestClient

    from afm_litho.service.app import app
    with TestClient(app) as c:          # triggers startup handlers
        assert c.get("/api/state").status_code == 200
        assert c.get("/").status_code == 200


def test_park_last_door_before_and_after_a_twin_park():
    """E1: `GET /api/park/last` is `{}` before any park and the stored ParkResult
    dict afterwards, and `GET /api/instrument`'s `last_park` key mirrors it. The park
    is a real twin park through the model-driven sync primitive (the same one
    safe_park drives), so this covers the whole read path, not a hand-set field."""
    try:
        from fastapi.testclient import TestClient

        from afm_litho.service.app import app, hub
        from afm_litho.service.host.safe_park import safe_park

        prev_last_park = getattr(hub.host, "last_park", None)
        with TestClient(app) as c:
            hub.host.last_park = None
            # BEFORE: no park has run this process -> empty body, 200 (not an error).
            r0 = c.get("/api/park/last")
            assert r0.status_code == 200 and r0.json() == {}
            assert c.get("/api/instrument").json().get("last_park") is None

            # AFTER: a real twin park. The sync primitive stores hub.last_park.
            result = safe_park(hub, "test: routes twin park")
            assert hub.host.last_park is not None

            r1 = c.get("/api/park/last")
            assert r1.status_code == 200
            body = r1.json()
            assert set(body) == {"verified", "method", "z_out_v", "zsensor_v",
                                 "defl_v", "critical", "reason", "ts"}
            assert body["reason"] == "test: routes twin park"
            assert body["ts"] is not None
            assert body == result.to_dict()
            # one call, same dict: GET /api/instrument carries it too.
            assert c.get("/api/instrument").json()["last_park"] == body
    finally:
        hub.host.last_park = prev_last_park


def test_operator_safe_park_door_populates_last_park_on_the_twin():
    """E2: driving the OPERATOR abort door `POST /api/safe_park` on the twin now
    populates `hub.last_park` and `GET /api/park/last`. Before E2 the twin door went
    through the native `inst.withdraw` and never stored a ParkResult, so the read door
    returned `{}` after a twin abort while hardware was populated -- the twin-vs-Cypher
    mismatch this fixes. The door reports a verified retract; the stored ParkResult is
    the same dict the read door returns."""
    try:
        from fastapi.testclient import TestClient

        from afm_litho.service.app import app, hub

        prev_last_park = getattr(hub.host, "last_park", None)
        with TestClient(app) as c:
            hub.host.last_park = None
            r = c.post("/api/safe_park", json={"reason": "test: operator twin abort"})
            assert r.status_code == 200
            sp = r.json()["safe_park"]
            assert sp["verified"] is True and sp["critical"] is False
            assert sp["backend"] == "twin"

            # E2: the door stored the ParkResult, so the read door is populated.
            assert hub.host.last_park is not None
            body = c.get("/api/park/last").json()
            assert body["verified"] is True
            assert body["method"] in ("abort", "vendor_withdraw")
            assert body["reason"] == "test: operator twin abort"
            assert body == hub.host.last_park.to_dict()
            # GET /api/instrument carries the same dict.
            assert c.get("/api/instrument").json()["last_park"] == body
    finally:
        hub.host.last_park = prev_last_park
