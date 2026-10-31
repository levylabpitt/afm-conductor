"""The two doors, listed (slice 2.0b).

Pins every HTTP route and every FLEX verb, so neither door can lose or gain an
entry without this test failing, and checks the list against two independent
sources: the edit-cockpit's own `ec-*.js` call sites, and the recorded HTTP
transcripts. The route table is also pinned by tests/test_routes.py; this file
adds the class of each route (what it is for) and the timing bound of each verb.

Classes of HTTP route:
  EC          called by an `ec-*.js` file (the edit-cockpit), plus its page.
  EC wrapper  has a named wrapper in ec-api.js but no caller yet.
  PAIR        the HTTP half of a FLEX verb that the edit-cockpit does not call.
  MAIN        the main window only; left untouched by the doors work.
  PROCESS     called by another afm-litho process (the keeper), not by a page.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[2] / "src" / "afm_litho" / "web"
TRANSCRIPTS = Path(__file__).parent / "http" / "transcripts"

# ----------------------------------------------------------------- HTTP routes --
# (method, path) of every route the edit-cockpit's ec-*.js files call.
EC_ROUTES = {
    ("POST", "/api/approach"),
    ("GET", "/api/connect/ports"),
    ("POST", "/api/connect/ports/add"),
    ("POST", "/api/connect/ports/delete"),
    ("POST", "/api/connect/ports/update"),
    ("GET", "/api/control"),
    ("POST", "/api/control/release"),
    ("POST", "/api/control/take"),
    ("GET", "/api/design"),
    ("POST", "/api/design/add"),
    ("POST", "/api/design/background"),
    ("GET", "/api/design/background/channels"),
    ("POST", "/api/design/bezier"),
    ("GET", "/api/design/bezier/{index}"),
    ("POST", "/api/design/chiral"),
    ("GET", "/api/design/files"),
    ("POST", "/api/design/op"),
    ("POST", "/api/design/open"),
    ("POST", "/api/design/points"),
    ("POST", "/api/design/reorder"),
    ("POST", "/api/design/save"),
    ("POST", "/api/design/svg"),
    ("GET", "/api/drift"),
    ("POST", "/api/drift/enable"),
    ("POST", "/api/drift/reference"),
    ("POST", "/api/fault/clear"),
    ("GET", "/api/instrument"),
    ("POST", "/api/instrument/open_config"),
    ("POST", "/api/instrument/override"),
    ("POST", "/api/instrument/reload"),
    ("GET", "/api/litho/estimate"),
    ("POST", "/api/litho/toolpath"),
    ("POST", "/api/litho/write"),
    ("GET", "/api/mode"),
    ("POST", "/api/mode"),
    ("POST", "/api/safe_park"),
    ("POST", "/api/scan/abort"),
    ("GET", "/api/scan/continuous"),
    ("POST", "/api/scan/continuous"),
    ("GET", "/api/scan/last"),
    ("POST", "/api/scan/relative"),
    ("POST", "/api/scan/start"),
    ("GET", "/api/state"),
    ("POST", "/api/tip/show"),
    ("POST", "/api/withdraw"),
    ("POST", "/api/zgain"),
}
EC_PAGE = ("GET", "/edit-cockpit")
# HTTP halves of FLEX verbs that the edit-cockpit does not call.
PAIR_ONLY = {
    ("POST", "/api/litho/load"),
    ("POST", "/api/litho/abort"),
    ("GET", "/api/litho/written"),
    ("GET", "/api/park/last"),
}
# Left untouched by the doors work (the main window).
MAIN_ONLY = {
    ("GET", "/"),
    ("POST", "/api/connect/ports/replace"),
    ("GET", "/api/contacts"),
    ("POST", "/api/contacts"),
    ("GET", "/api/design/capabilities"),
    ("GET", "/api/design/svg"),
    ("POST", "/api/force_curve"),
    ("GET", "/api/litho/demo"),
    ("GET", "/api/litho/frame"),
    ("POST", "/api/litho/frame"),
    ("GET", "/api/optics"),
    ("GET", "/api/pens"),
    ("POST", "/api/sample/load"),
    ("POST", "/api/sample/upload"),
    ("GET", "/api/samples"),
    ("POST", "/api/scan/params"),
    ("GET", "/api/sessions"),
    ("GET", "/api/sessions/recall"),
    ("GET", "/api/shapes"),
    ("POST", "/api/shapes/{index}"),
    ("GET", "/api/shapes/{index}/toolpath"),
    ("POST", "/api/sketch"),
    ("POST", "/api/sketch/clear"),
    ("POST", "/api/sketch/undo"),
    ("POST", "/api/speed/{factor}"),
    ("GET", "/api/transit"),
    ("POST", "/api/transit"),
    ("POST", "/api/tune/sweep"),
    ("POST", "/api/tune/thermal"),
}
# Called by another afm-litho process, not a page: a restarted keeper makes contact.
PROCESS_ONLY = {
    ("POST", "/api/keeper/contact"),
}
NOT_HTTP = {("/ws", "WS"), ("/static", "MOUNT")}
FRAMEWORK = {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}

# ------------------------------------------------------------------ FLEX verbs --
# verb -> (sockets it answers on, bound_s). "command" verbs are refused on the read
# socket; "both" answer on either. bound_s is the published worst case, which the
# provider's own loop wait and the driver's per-call timeout are derived from.
VERBS = {
    "ACK": ("both", 0.005),
    "IDN": ("both", 0.02),
    "HELP": ("both", 0.02),
    "getState": ("both", 0.05),
    "getStatus": ("both", 0.05),
    "getCapabilities": ("both", 0.02),
    "getInstrument": ("both", 0.05),
    "getLastPark": ("both", 0.05),
    "getEstimate": ("both", 0.1),
    "getTelemetry": ("both", 0.02),
    "getPattern": ("both", 0.1),
    "getWritten": ("both", 0.1),
    "getScanResult": ("both", 0.2),
    "getWriteResult": ("both", 0.2),
    "acquireControl": ("command", 0.05),
    "releaseControl": ("command", 0.05),
    "heartbeat": ("command", 0.05),
    "abort": ("command", 10.0),
    "safePark": ("command", 10.0),
    "getMode": ("both", 0.05),
    "setMode": ("command", 2.0),
    "startApproach": ("command", 1.0),
    "withdraw": ("command", 6.5),
    "setZGain": ("command", 1.0),
    "startScan": ("command", 1.0),
    "setContinuous": ("command", 5.0),
    "loadPattern": ("command", 2.0),
    "startWrite": ("command", 1.0),
    "abortWrite": ("command", 10.0),
    "setObjects": ("command", 2.0),
}

# FLEX verb -> the HTTP routes that reach the same coroutine (phase-2-doors.md section 1).
PAIRS = {
    "startApproach": [("POST", "/api/approach")],
    "setMode": [("POST", "/api/mode")],
    "getMode": [("GET", "/api/mode")],
    "setZGain": [("POST", "/api/zgain")],
    "startScan": [("POST", "/api/scan/start")],
    "setContinuous": [("POST", "/api/scan/continuous")],
    "startWrite": [("POST", "/api/litho/write")],
    "loadPattern": [("POST", "/api/litho/load")],
    "withdraw": [("POST", "/api/withdraw")],
    "abort": [("POST", "/api/safe_park")],
    "safePark": [("POST", "/api/safe_park")],
    "abortWrite": [("POST", "/api/litho/abort")],
    "getInstrument": [("GET", "/api/instrument")],
    "getLastPark": [("GET", "/api/park/last")],
    "getEstimate": [("GET", "/api/litho/estimate")],
    "getState": [("GET", "/api/state")],
    "getWritten": [("GET", "/api/litho/written")],
    "getScanResult": [("GET", "/api/scan/last")],
    "acquireControl": [("POST", "/api/control/take")],
    "releaseControl": [("POST", "/api/control/release")],
    "heartbeat": [("GET", "/api/control")],
}
# Verbs with no HTTP half. getTelemetry is the /ws `telemetry` frame; setObjects has
# no route at all; executePass is reserved and not a verb yet.
FLEX_ONLY = {"ACK", "IDN", "HELP", "getStatus", "getCapabilities", "getTelemetry",
             "getPattern", "getWriteResult", "setObjects"}


# --------------------------------------------------------------------- helpers --
def _actual_http_routes() -> set[tuple[str, str]]:
    from afm_litho.service.app import app
    return {(m.upper(), p) for p, ops in app.openapi()["paths"].items()
            if p not in FRAMEWORK for m in ops}


def _actual_non_http() -> set[tuple[str, str]]:
    from starlette.routing import Mount, WebSocketRoute

    from afm_litho.service.app import app
    out = set()
    for r in app.routes:
        if isinstance(r, WebSocketRoute):
            out.add((r.path, "WS"))
        elif isinstance(r, Mount):
            out.add((r.path, "MOUNT"))
    return out


def _no_line_comments(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("//"))


def _template(path: str) -> str:
    return re.sub(r"\$\{[^}]*\}", "{index}", path)


def _ec_js_sources() -> dict[str, str]:
    return {f.name: _no_line_comments(f.read_text(encoding="utf-8"))
            for f in sorted(WEB.glob("ec-*.js"))}


def _ec_api_wrappers(api_js: str) -> dict[str, tuple[str, str]]:
    """The named wrappers in ec-api.js: name -> (method, route)."""
    pat = re.compile(r"export const (\w+)\s*=\s*\([^)]*\)\s*=>\s*request\(\s*\"(GET|POST|PUT)\""
                     r"\s*,\s*[`\"']([^`\"']+)[`\"']")
    return {m.group(1): (m.group(2), "/api/" + _template(m.group(3)))
            for m in pat.finditer(api_js)}


def ec_js_call_sites() -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """(routes some ec-*.js file calls, routes ec-api.js names but nobody calls)."""
    sources = _ec_js_sources()
    wrappers = _ec_api_wrappers(sources["ec-api.js"])
    others = {n: t for n, t in sources.items() if n != "ec-api.js"}
    called: set[tuple[str, str]] = set()
    for name, route in wrappers.items():
        if any(re.search(r"\b" + name + r"\(", t) for t in others.values()):
            called.add(route)
    named_only = {route for name, route in wrappers.items()
                  if not any(re.search(r"\b" + name + r"\(", t) for t in others.values())}
    for text in others.values():
        for m in re.finditer(r"\bapi\(\s*[`\"']([^`\"']+)[`\"']", text):
            called.add(("POST", "/api/" + _template(m.group(1))))
        for m in re.finditer(r"\brequest\(\s*\"(GET|POST|PUT)\"\s*,\s*[`\"']([^`\"']+)[`\"']", text):
            called.add((m.group(1), "/api/" + _template(m.group(2))))
        for m in re.finditer(r"[\"'](/api/[^\"'?]+)[\"']", text):
            after = text[m.end(): m.end() + 140]
            method = "POST" if re.search(r"method:\s*\"POST\"", after) else "GET"
            called.add((method, _template(m.group(1))))
    return called, named_only - called


def _recorded_steps() -> list[tuple[str, tuple[str, str]]]:
    """(group, (method, path)) for every HTTP step in the transcripts."""
    steps = []
    for f in sorted(TRANSCRIPTS.glob("*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        for e in doc["entries"]:
            req = e.get("request")
            if req:
                steps.append((doc["group"], (req["method"], req["path"])))
    return steps


def _template_regex(path: str) -> re.Pattern:
    return re.compile("^" + re.sub(r"\{[^}]+\}", "[^/]+", re.escape(path).replace(r"\{", "{")
                                   .replace(r"\}", "}")) + "$")


def _route_of(method: str, path: str, routes: set[tuple[str, str]]):
    for m, template in routes:
        if m == method and _template_regex(template).match(path):
            return (m, template)
    return None


# ----------------------------------------------------------------------- tests --
def test_every_http_route_is_in_exactly_one_class():
    classes = {
        "EC": EC_ROUTES | {EC_PAGE},
        "PAIR": PAIR_ONLY,
        "MAIN": MAIN_ONLY,
        "PROCESS": PROCESS_ONLY,
    }
    names = list(classes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = classes[a] & classes[b]
            assert not both, f"{sorted(both)} sit in both {a} and {b}"
    listed = set().union(*classes.values())
    actual = _actual_http_routes()
    assert not actual - listed, f"routes the inventory does not know: {sorted(actual - listed)}"
    assert not listed - actual, f"routes the inventory lists that are gone: {sorted(listed - actual)}"
    # 83 after auth, minus the three Surface routes removed in slice 7a, plus keeper contact (K3)
    assert len(actual) == 81


def test_the_websocket_and_the_static_mount_are_still_served():
    assert _actual_non_http() == NOT_HTTP


def test_the_ec_route_list_matches_the_ec_js_call_sites():
    called, named_only = ec_js_call_sites()
    assert called == EC_ROUTES, (
        f"in ec-*.js but not in the list: {sorted(called - EC_ROUTES)}; "
        f"in the list but no longer called: {sorted(EC_ROUTES - called)}")
    assert not named_only, f"wrappers nobody calls: {sorted(named_only)}"


def test_the_edit_cockpit_page_is_linked_to_ec_main():
    html = (WEB / "edit-cockpit.html").read_text(encoding="utf-8")
    assert "ec-main.js" in html


def test_every_flex_verb_is_pinned_with_its_sockets_and_bound():
    from afm_litho.service.flex_dispatch import REGISTRY
    actual = {name: ("command" if m.command_only else "both", m.bound_s)
              for name, m in REGISTRY.items()}
    assert actual == VERBS
    assert len(actual) == 30


def test_every_verb_is_paired_with_http_or_flex_only():
    assert set(PAIRS) | FLEX_ONLY == set(VERBS)
    assert not set(PAIRS) & FLEX_ONLY
    routes = _actual_http_routes()
    for verb, halves in PAIRS.items():
        for half in halves:
            assert half in routes, f"{verb} pairs with {half}, which is not a route"
            assert half in EC_ROUTES | PAIR_ONLY, f"{verb}'s half {half} is classed MAIN"


@pytest.mark.parametrize("pair_only", sorted(PAIR_ONLY))
def test_a_pair_only_route_belongs_to_a_verb(pair_only):
    assert any(pair_only in halves for halves in PAIRS.values())


def test_the_transcripts_cover_every_ec_and_pair_route():
    covered = {r for _g, (m, p) in _recorded_steps()
               if (r := _route_of(m, p, _actual_http_routes()))}
    must = EC_ROUTES | {EC_PAGE} | PAIR_ONLY
    assert not must - covered, f"no transcript step for: {sorted(must - covered)}"


def test_every_transcript_step_hits_a_real_route():
    routes = _actual_http_routes()
    unknown = [(g, s) for g, s in _recorded_steps() if _route_of(s[0], s[1], routes) is None]
    assert not unknown, f"steps that match no route: {unknown}"


def test_every_ec_and_error_status_is_recorded_somewhere():
    """The shapes the plan names: 200-with-error bodies and real status codes."""
    statuses = set()
    error_bodies = 0
    for f in TRANSCRIPTS.glob("*.json"):
        for e in json.loads(f.read_text(encoding="utf-8"))["entries"]:
            if "status" in e:
                statuses.add(e["status"])
                if e["status"] == 200 and isinstance(e["response"], dict) and "error" in e["response"]:
                    error_bodies += 1
    assert {200, 400, 404, 500} <= statuses
    assert error_bodies == 48


def test_there_is_a_transcript_file_per_group_and_no_stray_file():
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "http"))
    from http_door_cases import GROUPS
    files = {f.stem for f in TRANSCRIPTS.glob("*.json")}
    assert files == set(GROUPS)
