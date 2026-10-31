"""The scripted sessions whose HTTP traffic is recorded (slice 2.0b).

Each group is one fresh twin service and an ordered list of requests, so a step
may rely on what the steps before it left behind. A step records the request, the
status code, the body and the WS events it caused; nothing here asserts behaviour,
it only drives the door and lets `http_door_recorder` write the result down.

A refusal is recorded as the door gives it today: mostly status 200 with an
`{"error", "reason"}` body, with a few real status codes (404, 400, 502).
"""

from __future__ import annotations

from http_door_recorder import Session, Step

# A scan small enough to finish at once at twin speed 0.
TINY_SCAN = {"size_um": 5.0, "pixels": 16, "lines": 4, "line_rate_hz": 200.0,
             "mode": "contact"}

# Long enough that a second request meets it still running, even at twin speed 0.
BIG_SCAN = {"size_um": 5.0, "pixels": 256, "lines": 256, "line_rate_hz": 200.0,
            "mode": "contact"}

# Closed rectangle, 1 SVG unit = 1 nm; 2000 nm = 2 um on a side.
SVG_RECT = (b'<svg xmlns="http://www.w3.org/2000/svg" width="5000" height="5000" '
            b'viewBox="0 0 5000 5000"><path d="M1000 1000 L3000 1000 L3000 3000 '
            b'L1000 3000 Z"/><path d="M500 4000 L4500 4200"/></svg>')

DESIGN_SVG_QUERY = {"name": "door-test.svg"}


def get(s: Session, path: str, name: str | None = None, **kw):
    return s.run(Step(name or f"GET {path}", "GET", path, **kw))


# The stops log from two threads and broadcast `safe_park` from a third, so the order
# of their frames within one request is not fixed; the frames are kept, sorted.
STOPS = frozenset({"/api/safe_park", "/api/scan/abort", "/api/litho/abort"})


def post(s: Session, path: str, body=None, name: str | None = None, **kw):
    if path in STOPS:
        kw.setdefault("unordered", True)
    return s.run(Step(name or f"POST {path}", "POST", path, body=body, **kw))


# ---------------------------------------------------------------- reads group --
def reads_idle(s: Session) -> None:
    """Every read an idle twin answers, and the reads that answer 'nothing yet'."""
    get(s, "/api/state")
    get(s, "/api/mode")
    get(s, "/api/control")
    get(s, "/api/instrument")
    get(s, "/api/park/last", name="GET /api/park/last (no park yet)")
    get(s, "/api/litho/estimate", name="GET /api/litho/estimate (no design)")
    get(s, "/api/litho/written")
    get(s, "/api/scan/last", name="GET /api/scan/last (404: no scan yet)")
    get(s, "/api/scan/continuous")
    get(s, "/api/drift")
    get(s, "/api/connect/ports")
    get(s, "/api/design", name="GET /api/design (nothing loaded)")
    get(s, "/api/design/files")
    get(s, "/api/design/capabilities")
    get(s, "/api/design/svg", name="GET /api/design/svg (no Bezier shapes)")
    get(s, "/api/design/background/channels", name="GET /api/design/background/channels (no scan)")
    get(s, "/api/design/bezier/0", name="GET /api/design/bezier/0 (no design)")
    s.run(Step("GET / (page)", "GET", "/", head_only=True))
    s.run(Step("GET /edit-cockpit (page)", "GET", "/edit-cockpit", head_only=True))


# ------------------------------------------------------------------ tip group --
def tip_and_stops(s: Session) -> None:
    """Withdraw, approach, mode, z-gain, scan, continuous, abort, park, litho."""
    post(s, "/api/approach", name="POST /api/approach (no body)")
    post(s, "/api/withdraw", name="POST /api/withdraw (tip already off)")
    post(s, "/api/approach", {"setpoint": 0.2, "igain": 12.0, "mode": "contact"},
         name="POST /api/approach (engages)")
    get(s, "/api/state", name="GET /api/state (engaged)")
    post(s, "/api/mode", {"mode": "ac"}, name="POST /api/mode (refused while engaged)")
    post(s, "/api/zgain", {"setpoint": 0.2, "igain": 3.0, "mode": "contact"},
         name="POST /api/zgain (twin refuses)")
    post(s, "/api/scan/abort", {"withdraw": True}, name="POST /api/scan/abort (nothing scanning)")
    get(s, "/api/state", name="GET /api/state (after scan/abort)")
    post(s, "/api/mode", {"mode": "ac"}, name="POST /api/mode (ac, tip off)")
    get(s, "/api/mode", name="GET /api/mode (after switch)")
    post(s, "/api/mode", {"mode": "banana"}, name="POST /api/mode (bad mode)")
    post(s, "/api/mode", {"mode": "contact"}, name="POST /api/mode (back to contact)")
    post(s, "/api/scan/start", {**TINY_SCAN, "size_um": -1.0},
         name="POST /api/scan/start (negative size is accepted)", quiet_s=0.8, max_wait_s=15.0)
    post(s, "/api/scan/start", BIG_SCAN, name="POST /api/scan/start (long scan)", quiet_s=0.2, max_wait_s=2.0,
         drop=("line",))
    post(s, "/api/scan/start", TINY_SCAN, name="POST /api/scan/start (refused: one already running)",
         quiet_s=0.2, max_wait_s=2.0, drop=("line",))
    post(s, "/api/mode", {"mode": "ac"}, name="POST /api/mode (refused while scanning)",
         quiet_s=0.2, max_wait_s=2.0, drop=("line",))
    post(s, "/api/scan/abort", {"withdraw": True}, name="POST /api/scan/abort (mid-scan)",
         quiet_s=0.5, max_wait_s=15.0, drop=("line",), unordered=True, until="scan_done")
    post(s, "/api/scan/start", TINY_SCAN, name="POST /api/scan/start (tiny scan, completes)",
         quiet_s=0.8, max_wait_s=15.0)
    get(s, "/api/scan/last", name="GET /api/scan/last (after scans)")
    post(s, "/api/scan/abort", {"withdraw": True}, name="POST /api/scan/abort (after scan)")
    get(s, "/api/design/background/channels", name="GET /api/design/background/channels (after scan)")
    post(s, "/api/safe_park", {"reason": "door transcript"}, name="POST /api/safe_park")
    post(s, "/api/safe_park", None, name="POST /api/safe_park (no body)")
    post(s, "/api/safe_park", {"reason": "operator vendor withdraw"},
         name="POST /api/safe_park (the vendor-withdraw button's reason)")
    get(s, "/api/park/last", name="GET /api/park/last (after park)")
    post(s, "/api/tip/show", {"on": True}, name="POST /api/tip/show (on)")
    post(s, "/api/tip/show", {"on": False}, name="POST /api/tip/show (off)")
    post(s, "/api/tip/show", None, name="POST /api/tip/show (no body)")
    post(s, "/api/withdraw", name="POST /api/withdraw (final)")


def continuous_scan(s: Session) -> None:
    """The continuous-imaging door: read, arm, refused arm, relative setpoint, stop."""
    get(s, "/api/scan/continuous", name="GET /api/scan/continuous (off)")
    post(s, "/api/scan/continuous", {"enabled": False}, name="POST /api/scan/continuous (off, already off)")
    post(s, "/api/scan/continuous", {"enabled": True}, name="POST /api/scan/continuous (on, no config)")
    post(s, "/api/scan/relative", {"enabled": True, "settle_s": 0.5},
         name="POST /api/scan/relative (on)")
    post(s, "/api/scan/relative", {"enabled": False}, name="POST /api/scan/relative (off)")
    post(s, "/api/scan/relative", None, name="POST /api/scan/relative (no body)")
    post(s, "/api/withdraw", name="POST /api/withdraw (baseline)")
    post(s, "/api/scan/continuous",
         {"enabled": True, "config": TINY_SCAN, "max_frames": 2, "interval_s": 0.0},
         name="POST /api/scan/continuous (arm, 2 frames)", quiet_s=1.0, max_wait_s=60.0,
         types_only=True, until="safe_park")
    get(s, "/api/scan/continuous", name="GET /api/scan/continuous (after run)")
    post(s, "/api/scan/continuous", {"enabled": False}, name="POST /api/scan/continuous (stop)")
    post(s, "/api/withdraw", name="POST /api/withdraw (final)")


# ---------------------------------------------------------------- control group --
def control(s: Session) -> None:
    """Operator take/release and the quirks a script's token meets (flex=True)."""
    get(s, "/api/control", name="GET /api/control (free)")
    post(s, "/api/control/release", None, name="POST /api/control/release (nobody holds)")
    post(s, "/api/control/take", {"client": "operator", "reason": "Take control button"},
         name="POST /api/control/take")
    get(s, "/api/control", name="GET /api/control (operator holds)")
    post(s, "/api/control/take", None, name="POST /api/control/take (again, no body)")
    post(s, "/api/control/release", {"token": "not-a-token"}, name="POST /api/control/release (wrong token)")
    get(s, "/api/control", name="GET /api/control (still held after wrong token)")
    post(s, "/api/control/release", {}, name="POST /api/control/release (operator, {} as the cockpit sends it)")

    # C4: a script holds control; the HTTP release needs no token and releases it.
    script = s.flex_note("acquireControl", {"client": "script-1", "deadman_s": 60})
    token = script["result"]["token"]
    get(s, "/api/control", name="GET /api/control (script holds; no token in the body)")
    post(s, "/api/control/release", None, name="POST /api/control/release (C4: releases the script, no token)")
    s.flex_note("heartbeat", {"token": token}, label="heartbeat after the HTTP release")

    # A1: an operator tip action revokes a script's token before the handler runs.
    # The holder is left empty; no operator record is made.
    token = s.flex_note("acquireControl", {"client": "script-2", "deadman_s": 60})["result"]["token"]
    post(s, "/api/withdraw", name="POST /api/withdraw (C7: takes control from script-2)")
    s.flex_note("heartbeat", {"token": token}, label="heartbeat after the operator's withdraw")
    get(s, "/api/control", name="GET /api/control (script revoked, nobody holds)")

    token = s.flex_note("acquireControl", {"client": "script-3", "deadman_s": 60})["result"]["token"]
    post(s, "/api/approach", {"setpoint": 0.2, "igain": 12.0, "mode": "contact"},
         name="POST /api/approach (engages, script-3 loses control)")
    s.flex_note("heartbeat", {"token": token}, label="heartbeat after the operator's approach")

    # A2: the take fires even when the request is then refused.
    token = s.flex_note("acquireControl", {"client": "script-4", "deadman_s": 60})["result"]["token"]
    post(s, "/api/mode", {"mode": "ac"},
         name="POST /api/mode (A2: refused while engaged, yet script-4 loses control)")
    s.flex_note("heartbeat", {"token": token}, label="heartbeat after the refused mode switch")
    get(s, "/api/control", name="GET /api/control (nobody holds after the refused mode switch)")
    token = s.flex_note("acquireControl", {"client": "script-6", "deadman_s": 60})["result"]["token"]
    post(s, "/api/scan/continuous", {"enabled": True},
         name="POST /api/scan/continuous (A2: refused, no config, yet script-6 loses control)")
    s.flex_note("heartbeat", {"token": token}, label="heartbeat after the refused continuous arm")
    token = s.flex_note("acquireControl", {"client": "script-7", "deadman_s": 60})["result"]["token"]
    post(s, "/api/litho/write", {"setpoint_relative": 0.1},
         name="POST /api/litho/write (A2: refused, no design, yet script-7 loses control)")
    s.flex_note("heartbeat", {"token": token}, label="heartbeat after the refused write")

    # An operator who holds control keeps it against a script.
    post(s, "/api/control/take", {"client": "alice"}, name="POST /api/control/take (alice)")
    s.flex_note("acquireControl", {"client": "script-8", "deadman_s": 60},
                label="acquireControl while the operator holds (refused)")
    post(s, "/api/mode", {"mode": "ac"}, name="POST /api/mode (refused; alice keeps control)")
    get(s, "/api/control", name="GET /api/control (alice still holds)")
    post(s, "/api/control/release", None, name="POST /api/control/release (alice)")

    # C2: z-gain adjustments neither take control nor preempt the script.
    token = s.flex_note("acquireControl", {"client": "script-5", "deadman_s": 60})["result"]["token"]
    post(s, "/api/zgain", {"setpoint": 0.2, "igain": 3.0, "mode": "contact"},
         name="POST /api/zgain (C2: script-5 keeps control)")
    get(s, "/api/control", name="GET /api/control (script-5 still holds)")
    # Stops never ask for control.
    post(s, "/api/safe_park", {"reason": "door transcript"}, name="POST /api/safe_park (script-5 holds)")
    post(s, "/api/scan/abort", {}, name="POST /api/scan/abort (script-5 holds)")
    post(s, "/api/litho/abort", None, name="POST /api/litho/abort (script-5 holds)")
    get(s, "/api/control", name="GET /api/control (stops left script-5 in place)")
    s.flex_note("releaseControl", {"token": token}, label="script-5 releases")
    post(s, "/api/withdraw", name="POST /api/withdraw (final)")


def design_edit(s: Session) -> None:
    """Import, draw, edit, reorder, chiral, save and reopen a design; the toolpath,
    the estimate and the pattern load that read it."""
    post(s, "/api/design/svg", None, name="POST /api/design/svg (not an svg)",
         raw=b"hello", content_type="image/svg+xml", query=DESIGN_SVG_QUERY)
    s.run(Step("POST /api/design/svg (400: no scan frame to place it on)", "POST", "/api/design/svg",
               raw=SVG_RECT, content_type="image/svg+xml", query=DESIGN_SVG_QUERY))
    post(s, "/api/design/background", {"channel": "height"},
         name="POST /api/design/background (400: no design loaded)")
    post(s, "/api/withdraw", name="POST /api/withdraw (baseline)")
    post(s, "/api/scan/start", TINY_SCAN, name="POST /api/scan/start (gives the design a frame)",
         quiet_s=0.8, max_wait_s=15.0)
    s.run(Step("POST /api/design/svg (rectangle and wire)", "POST", "/api/design/svg",
               raw=SVG_RECT, content_type="image/svg+xml", query=DESIGN_SVG_QUERY))
    get(s, "/api/design", name="GET /api/design (after import)")
    get(s, "/api/design/bezier/0", name="GET /api/design/bezier/0 (the rectangle)")
    get(s, "/api/design/bezier/1", name="GET /api/design/bezier/1 (the wire)")
    get(s, "/api/design/bezier/99", name="GET /api/design/bezier/99 (no such object)")
    post(s, "/api/design/add", {"kind": "path", "name": "wire", "points_um": [[0, 0], [1, 1], [2, 0.5]],
                                "voltage": 6.0, "speed_um_s": 2.0}, name="POST /api/design/add (path)")
    post(s, "/api/design/add", {"kind": "blob"}, name="POST /api/design/add (bad kind)")
    post(s, "/api/design/add", {"kind": "path", "points_um": [[0, 0]]},
         name="POST /api/design/add (one point)")
    post(s, "/api/design/add", {"kind": "path", "points_um": "no"},
         name="POST /api/design/add (points not numbers)")
    post(s, "/api/design/points", {"index": 2, "points_um": [[0, 0], [1, 2], [3, 1], [4, 0]]},
         name="POST /api/design/points")
    post(s, "/api/design/points", {"index": 99, "points_um": [[0, 0], [1, 1]]},
         name="POST /api/design/points (no such object)")
    post(s, "/api/design/points", {"index": 0, "points_um": [[0, 0]]},
         name="POST /api/design/points (one point)")
    get(s, "/api/design/bezier/2", name="GET /api/design/bezier/2 (a plain polyline)")
    post(s, "/api/design/bezier", {"index": 2, "bezier": {"nodes": [
        {"x": 0, "y": 0, "h_in": None, "h_out": [0.5, 0.5], "kind": "smooth"},
        {"x": 2, "y": 1, "h_in": [1.5, 0.5], "h_out": None, "kind": "smooth"}], "closed": False}},
         name="POST /api/design/bezier")
    post(s, "/api/design/bezier", {"index": 2, "bezier": {"nodes": []}},
         name="POST /api/design/bezier (no nodes)")
    post(s, "/api/design/bezier", {"index": 99, "bezier": {"nodes": []}},
         name="POST /api/design/bezier (no such object)")
    post(s, "/api/design/op", {"op": "duplicate", "indices": [2], "dx_um": 1.0, "dy_um": 1.0},
         name="POST /api/design/op duplicate")
    post(s, "/api/design/op", {"op": "set_props", "indices": [2], "voltage": 7.5, "speed_um_s": 3.0},
         name="POST /api/design/op set_props")
    post(s, "/api/design/op", {"op": "transform", "indices": [3], "dx_um": 0.5, "rot_deg": 10.0},
         name="POST /api/design/op transform")
    post(s, "/api/design/op", {"op": "nonsense", "indices": [0]}, name="POST /api/design/op (bad op)")
    post(s, "/api/design/op", {"op": "delete", "indices": []}, name="POST /api/design/op (empty selection)")
    post(s, "/api/design/op", {"op": "delete", "indices": [99]}, name="POST /api/design/op (no such object)")
    post(s, "/api/design/reorder", {"order": [0, 0]}, name="POST /api/design/reorder (not a permutation)")
    post(s, "/api/design/reorder", {"order": [3, 2, 1, 0]}, name="POST /api/design/reorder")
    post(s, "/api/design/chiral", {"indices": [2], "enabled": True, "lambda_um": 1.0, "y_amp_um": 0.2,
                                   "v_k": 1.0, "phase_deg": 0.0, "v0": 6.0},
         name="POST /api/design/chiral (enable)")
    post(s, "/api/design/chiral", {"indices": [3], "enabled": True, "lambda_um": 1.0, "y_amp_um": 0.2,
                                   "v_k": 1.0, "phase_deg": 0.0},
         name="POST /api/design/chiral (not a path)")
    post(s, "/api/design/chiral", {"indices": "x", "enabled": True},
         name="POST /api/design/chiral (indices not a list)")
    post(s, "/api/design/chiral", {"indices": [2], "enabled": True, "lambda_um": "wide"},
         name="POST /api/design/chiral (bad number)")
    post(s, "/api/design/chiral", {"indices": [2], "enabled": False}, name="POST /api/design/chiral (disable)")

    get(s, "/api/litho/estimate", name="GET /api/litho/estimate (with design)")
    post(s, "/api/litho/toolpath", None, name="POST /api/litho/toolpath")
    post(s, "/api/design/save", {"name": "door-test"}, name="POST /api/design/save")
    post(s, "/api/design/save", {"name": "door-test"}, name="POST /api/design/save (overwrite)")
    post(s, "/api/design/save", {"name": ""}, name="POST /api/design/save (empty name)")
    get(s, "/api/design/files", name="GET /api/design/files (one saved)")
    post(s, "/api/design/open", {"name": "door-test", "mode": "bogus"}, name="POST /api/design/open (bad mode)")
    post(s, "/api/design/open", {"name": "no-such"}, name="POST /api/design/open (unknown name)")
    post(s, "/api/design/open", {"name": "door-test", "mode": "replace"}, name="POST /api/design/open (replace)")
    post(s, "/api/design/open", {"name": "door-test"}, name="POST /api/design/open (add)")
    get(s, "/api/design/background/channels", name="GET /api/design/background/channels (after a scan)")
    post(s, "/api/design/background", {"channel": "height"}, name="POST /api/design/background")
    post(s, "/api/design/background", {"channel": "phase"}, name="POST /api/design/background (400: no such channel)")
    post(s, "/api/design/background", None, name="POST /api/design/background (no body)")
    post(s, "/api/design/op", {"op": "delete", "indices": [0, 1, 2, 3, 4, 5, 6, 7]},
         name="POST /api/design/op delete (several)")
    post(s, "/api/litho/load", {"source": "no-such-design"}, name="POST /api/litho/load (unknown source)")
    post(s, "/api/litho/load", None, name="POST /api/litho/load (auto, no body)")
    post(s, "/api/withdraw", name="POST /api/withdraw (final)")


# ---------------------------------------------------------------- litho group --
def litho_write(s: Session) -> None:
    """The write door: its refusals, then writes of the twin's default design."""
    get(s, "/api/design", name="GET /api/design (the twin starts with a demo design)")
    get(s, "/api/litho/estimate", name="GET /api/litho/estimate (demo design)")
    post(s, "/api/litho/write", {"setpoint_relative": 0.1},
         name="POST /api/litho/write (amp_gain missing)")
    post(s, "/api/litho/write", {"amp_gain": 25.0},
         name="POST /api/litho/write (a write with no setpoint is refused)")
    post(s, "/api/litho/write", {"amp_gain": 25.0, "setpoint_relative": 0.1, "objects": [99]},
         name="POST /api/litho/write (no such object)")
    post(s, "/api/litho/write", {"amp_gain": 25.0, "setpoint_relative": 0.1},
         name="POST /api/litho/write (tip off, writes every object)",
         quiet_s=1.0, max_wait_s=20.0, types_only=True)
    # The twin plugin has no dose model and no noise (2026-10-27: realism comes from
    # real scans), so the conductance map is flat and the parked tip is where the plugin
    # left it, not where the old twin's arc did.
    get(s, "/api/litho/written", name="GET /api/litho/written (after a write)",
        masked=("g",))
    post(s, "/api/litho/write", {"amp_gain": 25.0, "setpoint_relative": 0.1},
         name="POST /api/litho/write (refused: nothing new to write)")
    post(s, "/api/design/add", {"kind": "path", "name": "extra", "points_um": [[0, 0], [1, 1], [2, 0.5]],
                                "voltage": 6.0, "speed_um_s": 2.0}, name="POST /api/design/add (path)")
    post(s, "/api/litho/write", {"amp_gain": 25.0, "setpoint_relative": 0.1},
         name="POST /api/litho/write (writes only the new object)",
         quiet_s=1.0, max_wait_s=20.0, types_only=True)
    post(s, "/api/litho/abort", None, name="POST /api/litho/abort (nothing writing)")
    get(s, "/api/park/last", name="GET /api/park/last (after litho/abort)")
    post(s, "/api/litho/load", {"source": "no-such-design"}, name="POST /api/litho/load (unknown source)")
    post(s, "/api/litho/load", None, name="POST /api/litho/load (auto, no body)")
    post(s, "/api/withdraw", name="POST /api/withdraw (final)",
         masked=("x_um", "y_um", "defl_v"))


# ------------------------------------------------------------------ misc group --
def instrument_ports_drift_measure(s: Session) -> None:
    """The instrument model, the electrode ports, drift, the fault latch, the
    measure config and the measure run/abort pair."""
    # instrument model
    post(s, "/api/instrument/override", {}, name="POST /api/instrument/override (no field)")
    post(s, "/api/instrument/override", {"field": "to_dict", "value": 1},
         name="POST /api/instrument/override (not an overridable field)")
    post(s, "/api/instrument/override", {"field": "tip_amp_gain", "value": 20.0},
         name="POST /api/instrument/override (preview, nothing applied)")
    post(s, "/api/instrument/override", {"field": "xy_range_um", "value": [1.0], "confirm": True},
         name="POST /api/instrument/override (bad value)")
    post(s, "/api/instrument/override",
         {"field": "tip_amp_gain", "value": 20.0, "confirm": True, "reason": "door transcript"},
         name="POST /api/instrument/override (applied)")
    get(s, "/api/instrument", name="GET /api/instrument (after an override)")
    post(s, "/api/instrument/reload", {}, name="POST /api/instrument/reload")
    # open_config would start the host's default editor, so only its refusal is recorded.
    (s.svc.runtime / "instrument.toml").rename(s.svc.runtime / "instrument.toml.away")
    post(s, "/api/instrument/open_config", {},
         name="POST /api/instrument/open_config (file missing; the success path opens an editor)")
    (s.svc.runtime / "instrument.toml.away").rename(s.svc.runtime / "instrument.toml")

    # fault latch
    post(s, "/api/fault/clear", None, name="POST /api/fault/clear (no body)")
    post(s, "/api/fault/clear", {"confirm": True}, name="POST /api/fault/clear (confirmed, nothing latched)")

    # electrode ports
    post(s, "/api/connect/ports/add", {"position_um": [1.0, 2.0], "direction_deg": 90.0},
         name="POST /api/connect/ports/add")
    post(s, "/api/connect/ports/add", {"position_um": [1.0]}, name="POST /api/connect/ports/add (bad position)")
    post(s, "/api/connect/ports/update", {"id": "e0", "label": "source", "width_um": 2.0},
         name="POST /api/connect/ports/update")
    post(s, "/api/connect/ports/update", {"id": "nope"}, name="POST /api/connect/ports/update (no such port)")
    post(s, "/api/connect/ports/update", {}, name="POST /api/connect/ports/update (no id)")
    get(s, "/api/connect/ports", name="GET /api/connect/ports (one port)")
    post(s, "/api/connect/ports/delete", {"id": "nope"}, name="POST /api/connect/ports/delete (no such port)")
    post(s, "/api/connect/ports/delete", {"id": "e0"}, name="POST /api/connect/ports/delete")

    # drift
    post(s, "/api/drift/reference", {}, name="POST /api/drift/reference (no scan cached)")
    post(s, "/api/drift/enable", {"enabled": True}, name="POST /api/drift/enable (no scan cached)")
    post(s, "/api/drift/enable", {"enabled": False}, name="POST /api/drift/enable (off)")
    post(s, "/api/withdraw", name="POST /api/withdraw (baseline)")
    post(s, "/api/scan/start", TINY_SCAN, name="POST /api/scan/start (for the drift reference)",
         quiet_s=0.8, max_wait_s=15.0)
    post(s, "/api/drift/reference", {}, name="POST /api/drift/reference")
    post(s, "/api/drift/enable", {"enabled": True}, name="POST /api/drift/enable (on)")
    get(s, "/api/drift", name="GET /api/drift (enabled)")
    post(s, "/api/drift/enable", {"enabled": False}, name="POST /api/drift/enable (off again)")

    # Last, because it leaves the model file unreadable: a string is accepted for a number.
    post(s, "/api/instrument/override", {"field": "tip_amp_gain", "value": "loud", "confirm": True},
         name="POST /api/instrument/override (accepts a string for a number)")
    post(s, "/api/instrument/reload", {}, name="POST /api/instrument/reload (500: the file is now unreadable)")
    post(s, "/api/withdraw", name="POST /api/withdraw (final)")


GROUPS = {
    "instrument_ports_drift_measure": (False, instrument_ports_drift_measure),
    "litho_write": (False, litho_write),
    "design_edit": (False, design_edit),
    "reads_idle": (False, reads_idle),
    "tip_and_stops": (False, tip_and_stops),
    "continuous_scan": (False, continuous_scan),
    "control": (True, control),
}
