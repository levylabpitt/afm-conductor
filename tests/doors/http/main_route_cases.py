"""The scripted session for the 26 MAIN-only routes that had no transcript (H5.pre).

Same recorder as `http_door_cases`, kept apart so the existing transcripts and the
inventory's counts over them stay untouched. One fresh twin service, one ordered list
of requests; nothing here asserts behaviour, the recorded JSON under
`transcripts_main/` does.
"""

from __future__ import annotations

from http_door_cases import TINY_SCAN, get, post
from http_door_recorder import Session, Step


def main_routes(s: Session) -> None:
    # reads that answer on an idle twin
    get(s, "/api/optics")
    get(s, "/api/pens")
    get(s, "/api/samples", masked=("files",))
    get(s, "/api/sessions")
    get(s, "/api/sessions/recall", query={"id": "nope"}, name="GET /api/sessions/recall (unknown id)")
    get(s, "/api/contacts")
    get(s, "/api/transit", name="GET /api/transit (no design)")
    get(s, "/api/litho/frame")
    get(s, "/api/litho/demo")
    get(s, "/api/shapes", name="GET /api/shapes (none)")
    get(s, "/api/shapes/0/toolpath", name="GET /api/shapes/0/toolpath (no shapes)")

    # scan panel records and the three twin refusals
    post(s, "/api/scan/params", {"size_um": 7.0, "mode": "ac"})
    post(s, "/api/scan/params", None, name="POST /api/scan/params (no body)")
    post(s, "/api/force_curve")
    post(s, "/api/tune/thermal")
    post(s, "/api/tune/sweep")
    post(s, "/api/speed/2.0")
    post(s, "/api/speed/1000", name="POST /api/speed/1000 (clamped)")
    post(s, "/api/speed/1.0", name="POST /api/speed/1.0 (back)")

    # litho frame and transit policy
    post(s, "/api/litho/frame", {"fast_size_um": 4.0, "slow_size_um": 3.0, "angle_deg": 10.0})
    get(s, "/api/litho/frame", name="GET /api/litho/frame (after set)")
    post(s, "/api/litho/frame", {"angle_deg": None}, name="POST /api/litho/frame (angle back to the scan)")
    post(s, "/api/transit", {"mode": "lift", "lift_nm": 80.0})
    post(s, "/api/transit", {"mode": "banana"}, name="POST /api/transit (bad mode)")
    get(s, "/api/transit", name="GET /api/transit (after set)")

    # sketch, shapes, undo, clear (a stroke builds a design without a scan)
    post(s, "/api/sketch", {"points_um": [[0, 0]]}, name="POST /api/sketch (too short)")
    post(s, "/api/sketch", {"points_um": [[0, 0], [1, 0.2], [2, 0.1], [3, 0.4]], "pen": "write",
                            "mode": "literal"}, name="POST /api/sketch (a stroke)")
    get(s, "/api/shapes", name="GET /api/shapes (one stroke)")
    get(s, "/api/shapes/0/toolpath", name="GET /api/shapes/0/toolpath")
    get(s, "/api/shapes/9/toolpath", name="GET /api/shapes/9/toolpath (no such shape)")
    post(s, "/api/shapes/0", {"voltage": 5.0}, name="POST /api/shapes/0 (voltage)")
    post(s, "/api/shapes/9", {"voltage": 5.0}, name="POST /api/shapes/9 (no such shape)")
    post(s, "/api/sketch/undo", None, name="POST /api/sketch/undo")
    post(s, "/api/sketch/undo", {"count": 5}, name="POST /api/sketch/undo (past the history)")
    post(s, "/api/sketch", {"points_um": [[0, 0], [2, 1]], "pen": "write", "mode": "literal"},
         name="POST /api/sketch (second stroke)")
    post(s, "/api/sketch/clear")

    # sample and contacts: refusals only, so no file is written under data/samples
    post(s, "/api/sample/load", {"file": "no-such-scan.ibw"}, name="POST /api/sample/load (not found)")
    s.run(Step("POST /api/sample/upload (not an ibw)", "POST", "/api/sample/upload", raw=b"x",
               content_type="application/octet-stream", query={"name": "x.txt"}))
    s.run(Step("POST /api/sample/upload (empty)", "POST", "/api/sample/upload", raw=b"",
               content_type="application/octet-stream", query={"name": "x.ibw"}))
    post(s, "/api/contacts", {"contacts": []}, name="POST /api/contacts (erase all)")
    get(s, "/api/contacts", name="GET /api/contacts (after erase)")

    # ports
    post(s, "/api/connect/ports/replace", {"ports": "no"}, name="POST /api/connect/ports/replace (not a list)")
    post(s, "/api/connect/ports/replace", {"ports": [{"bad": 1}]},
         name="POST /api/connect/ports/replace (bad port)")
    post(s, "/api/connect/ports/replace", {"ports": []}, name="POST /api/connect/ports/replace (empty)")

    # a scan leaves a session record
    post(s, "/api/withdraw", name="POST /api/withdraw (baseline)")
    post(s, "/api/scan/start", TINY_SCAN, name="POST /api/scan/start (leaves a session record)",
         quiet_s=0.8, max_wait_s=15.0)
    get(s, "/api/sessions", name="GET /api/sessions (after a scan)", masked=("local",))
