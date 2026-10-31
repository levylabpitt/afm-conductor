"""Write control on the wire (slice 09): `loadPattern`, `startWrite`, `abortWrite`,
and the `getPattern` / `getWritten` / `getWriteResult` reads beside them.

Truth document: docs/FLEX_INTEGRATION_PLAN.md Sec A (the instrument-operations rows),
A.1 (results by path), C (the `service/commands/` layer these verbs and the cockpit's
endpoints share), E.2 (token, single-writer claim taken before the first await),
F.1/F.2/F.3 (start+poll, bounds, the state machine), J risk 11 (the AC-guard
fail-open ordering). Wire reference: docs/FLEX_PROVIDER.md. Rig checks:
docs/FLEX_RIG_RUNBOOK.md "09 Scripted write".

REAL SOCKETS, REAL HUB, REAL LOOP -- the same construction `test_flex_scan.py` uses,
and for the same reason: a fake hub would test the handler's plumbing and none of the
things that have actually broken here (a claim not released, a run record nobody
closes, a pump that never exits).

ZERO HARDWARE. Every case runs on the twin. The rules that are hardware-only -- the
AC-guard refusal and the silent-twin backend mismatch -- are asserted against the
service layer with a hub stub that merely SAYS it is hardware. They are refusals, so
nothing is driven either way.
"""

from __future__ import annotations

import asyncio
import json
import math
import queue as _queue
import subprocess
import sys
import threading
import time

import pytest
from _im_fixtures_guard import make_model, seed_free_air
from conftest import start_flex_provider

from afm_litho.litho.chunker import CHAIN_CHUNK_POINTS
from afm_litho.service import flex_dispatch as fd
from afm_litho.service import flex_provider as fp
from afm_litho.service.host import litho_cmds
from afm_litho.service.commands.errors import CommandError

zmq = pytest.importorskip("zmq", reason="pyzmq is the optional [flex] extra")


def _gw(relative=0.2):
    """FLEX `startWrite` command-guard params (INSTRUMENT_MODEL_SPEC secs 2/10,
    13). Seeds a fresh withdrawn free-air baseline on the process hub and returns
    the ABSOLUTE `setpoint` (free_air + relative -- FLEX keeps the absolute form,
    unlike the REST door) plus `amp_gain`, to spread into the startWrite body. A
    +0.2 relative over the seeded -0.2 free air lands at an absolute -0.0 V, inside
    the contact band the guard checks on `setpoint - free_air` (spec sec 2)."""
    from afm_litho.service.hub import hub as real_hub
    fa = seed_free_air(real_hub)
    return {"setpoint_reference": "absolute", "setpoint": fa.value_v + relative, "amp_gain": 25.0}

# Long enough that no test's polling loop can let the dead-man fire mid-write: reads
# ride the READ socket, which deliberately does NOT refresh it.
DEADMAN_S = 300

# An SVG drawn the way this lab draws them: 1 unit = 1 nm. Two objects, so an
# `objects: [idx]` selection has something to select FROM.
# A design-origin loadPattern needs the scan size the design was drawn over. The
# cockpit writes it into the SVG as a `topo_bgd` background image (width =
# size_um * 1000); these fixtures embed one so the FLEX door resolves a size the
# same way a real drawing does. bbox assertions here are self-consistent (trace
# vs the reported bbox_um) and so are unaffected by where the design lands.
TWO_OBJECT_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg">'
    '<image id="topo_bgd" x="0" y="0" width="4000" height="4000"/>'
    '<path d="M 0 0 L 4000 0"/>'
    '<path d="M 0 2000 L 4000 2000"/>'
    "</svg>"
)


# The suite plays a twin write at unlimited speed (tests/conftest.py), so it is over
# within milliseconds of `startWrite`. A case that acts on a write IN FLIGHT plays it
# at twice real time instead: TWO_OBJECT_SVG is two 4 um strokes at 2 um/s, about two
# seconds of write for the next call to land in.
IN_FLIGHT = {"twin_speed": 2.0}


def _over_cap_svg() -> str:
    """One object whose compiled toolpath is ~100k points -- over the out-wave cap.

    A sawtooth, not a long straight line: the compiled count is arc length over the
    interpolation spacing, so 100 segments of 20 um each blows the cap while the
    design's BOUNDING BOX stays 20 um square and well inside the flatten guard. The
    `topo_bgd` image gives the 20 um scan the design was drawn over.
    """
    legs = " ".join(f"L {i * 200} {20000 if i % 2 else 0}" for i in range(1, 101))
    return ('<svg xmlns="http://www.w3.org/2000/svg">'
            '<image id="topo_bgd" x="0" y="0" width="20000" height="20000"/>'
            f'<path d="M 0 0 {legs}"/></svg>')


# ------------------------------------------------------------------ fixtures --

class _Client:
    def __init__(self, endpoint: str, timeout_s: float = 30.0):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
        self.sock.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
        self.sock.connect(endpoint)
        self._n = 0

    def send(self, method, params=None) -> dict:
        self._n += 1
        req = {"jsonrpc": "2.0", "method": method, "id": str(self._n)}
        if params is not None:
            req["params"] = params
        self.sock.send_string(json.dumps(req))
        return json.loads(self.sock.recv_string())

    def call(self, method, params=None):
        reply = self.send(method, params)
        assert "error" not in reply, reply
        return reply["result"]

    def close(self):
        self.sock.close(linger=0)
        self.ctx.term()


@pytest.fixture(scope="module")
def hub(client):
    """The PROCESS hub, with the app running (so `hub._loop` is a live loop and the
    control watcher, the run registry and the twin instrument are all real)."""
    from afm_litho.service.hub import hub as real_hub
    return real_hub


@pytest.fixture(scope="module")
def provider(hub):
    prov = start_flex_provider(fp, hub)
    yield prov
    prov.stop()


@pytest.fixture(scope="module")
def cmd(provider):
    c = _Client(provider.bound["command"])
    yield c
    c.close()


@pytest.fixture(scope="module")
def read(provider):
    c = _Client(provider.bound["read"])
    yield c
    c.close()


@pytest.fixture(autouse=True)
def _clean(hub, cmd):
    """Leave the instrument the way we found it. The hub is a process global shared
    with every other test module, so a case that left a write running, a token held or
    the tip engaged would leak into whatever runs next -- and the leak would look like
    a bug in THAT file."""
    yield
    _wait_write_done(hub)
    cmd.send("withdraw", {})
    _release(cmd, hub)
    hub.host.written_objects = set()


def _token(cmd, **kw) -> str:
    """Become the commander, taking over from a previous case's leftovers."""
    reply = cmd.send("acquireControl",
                     {"client": "test-09", "deadman_s": DEADMAN_S, **kw})
    if "error" in reply:                    # held by a token this module abandoned
        from afm_litho.service.hub import hub as real_hub
        real_hub.control = None
        reply = cmd.send("acquireControl",
                         {"client": "test-09", "deadman_s": DEADMAN_S, **kw})
    return reply["result"]["token"]


def _release(cmd, hub) -> None:
    cur = getattr(hub, "control", None)
    if cur is not None:
        cmd.send("releaseControl", {"token": cur.token})


def _wait_write_done(hub, timeout: float = 90.0) -> bool:
    """Wait until no write worker is alive AND the pump has released the claim."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        thread = getattr(hub.host, "litho_thread", None)
        if ((thread is None or not thread.is_alive())
                and getattr(hub, "_run_claim", None) is None):
            return True
        time.sleep(0.05)
    return False


def _wait_run(hub, run_id: str, timeout: float = 120.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rec = hub.runs.get(run_id)
        if rec is not None and rec.finished is not None:
            return rec
        time.sleep(0.05)
    raise AssertionError(f"{run_id} never finished")


def _wait_orphan_clear(timeout: float = 5.0) -> None:
    """Block until the `objects` door's tracked orphan (if any) has actually
    finished on its dedicated executor. A test that deliberately times out a build
    leaves that build running in the background for real (it cannot be killed --
    `litho_cmds._submit_guarded`); without this, the NEXT test to load `objects`
    -- in this file, or in another module sharing the same process-wide
    executor -- could see a spurious `-32020 build_in_progress` left over from a
    PREVIOUS test, which is exactly the kind of order-dependent flake this suite
    does not want to introduce."""
    deadline = time.monotonic() + timeout
    while litho_cmds._orphan_future is not None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert litho_cmds._orphan_future is None, "a previous test's orphan never cleared"


def _load(cmd, source) -> dict:
    return cmd.call("loadPattern", {"token": _token(cmd), "source": source})


def _spy_on_logs(hub, monkeypatch) -> list[str]:
    """Capture what the hub LOGS. The real hub broadcasts its log lines over the
    WebSocket rather than keeping them, and the chunk WARNING is the only thing that
    tells an operator a seam is coming, so it has to be asserted somewhere."""
    lines: list[str] = []
    original = hub.log

    async def spy(text: str):
        lines.append(text)
        await original(text)

    monkeypatch.setattr(hub, "log", spy)
    return lines


class _HwHub:
    """A hub that merely SAYS it is on hardware. Nothing here drives anything -- it
    exists so the two hardware-only REFUSALS can be asserted off-instrument."""

    hardware_arc = None
    backend = "hardware"
    engaged = False
    imaging_mode = "contact"
    litho_thread = None
    pattern = None

    def __init__(self, live: bool = True):
        self._live = live
        self.logs: list[str] = []
        self.z_loop: dict = {}
        self.written_objects: set = set()
        self._run_claim = None

    # The single-writer claim, with the REAL `RunClaim` object rather than a mock, so
    # a refusal here proves the claim was taken and then released on the failure path.
    def claim_run(self, kind: str):
        from afm_litho.service.control import RunClaim
        self._run_claim = RunClaim(kind, time.time())
        return self._run_claim

    def release_claim(self, claim) -> None:
        if claim is not None and self._run_claim is claim:
            self._run_claim = None

    def _in_flight_run_id(self):
        return None

    def backend_label(self):
        return "hardware" if self._live else "disconnected"

    def backend_live(self):
        return "hardware" if self._live else "twin"

    def require_live_backend(self):
        if self._live:
            return None
        return ("backend is configured as hardware but no bridge is live")

    async def log(self, text):
        self.logs.append(text)

    @property
    def host(self):                # fake: host fields live on the fake itself
        return self


# ============================================================ 1. the surface ====

def test_the_write_verbs_are_no_longer_reserved():
    """They answered `-32601 not implemented in this build` until this milestone."""
    for name in ("loadPattern", "startWrite", "abortWrite"):
        assert name in fd.REGISTRY, name
        assert name not in fd.RESERVED, name
    # ...and what is still honestly reserved stays reserved.
    assert fd.RESERVED == frozenset({"executePass"})


def test_capabilities_publishes_a_bound_for_every_write_verb(read):
    bounds = read.call("getCapabilities")["method_bounds"]
    for name in ("loadPattern", "startWrite", "abortWrite",
                 "getPattern", "getWritten", "getWriteResult"):
        assert name in bounds, name
        assert bounds[name] > 0
    # Asserted as VALUES, because a driver derives its per-call timeout from them.
    # startWrite is sub-second BY CONSTRUCTION (validate, claim, spawn, return) -- the
    # plan's "<=32 s today" was the unbounded AC-guard crosspoint read, which now
    # carries the explicit 2 s fail-safe timeout and runs off the loop.
    assert bounds["startWrite"] <= 1.0
    assert bounds["loadPattern"] == 2.0
    assert bounds["abortWrite"] == 10.0        # the same verified retract as `abort`
    assert fp.loop_timeout_for("startWrite") >= bounds["startWrite"]


def test_every_write_verb_stays_under_the_serialisation_ceiling():
    """29180 handles one request at a time, so a verb that blocks for half a minute
    blocks `abort` and `safePark` behind it for half a minute."""
    for name in ("loadPattern", "startWrite", "abortWrite"):
        assert fd.REGISTRY[name].bound_s <= fp.LOOP_CALL_MAX_S, name
        assert fp.loop_timeout_for(name) <= fp.LOOP_CALL_MAX_S, name


@pytest.mark.parametrize("method", ["loadPattern", "startWrite", "abortWrite"])
def test_the_write_verbs_are_refused_by_the_read_socket(read, method):
    """29181 must stay a plane on which no request can affect the tip -- and that has
    to be true of the SOCKET, not just of the documentation."""
    reply = read.send(method, {"token": "x"})
    assert reply["error"]["code"] == fd.METHOD_NOT_FOUND
    assert "COMMAND socket" in reply["error"]["message"]


# =============================================================== 2. the token ===

@pytest.mark.parametrize("method,params", [
    ("loadPattern", {"source": "auto"}),
    ("startWrite", {"objects": "all"}),
])
def test_the_mutating_write_verbs_need_a_token(cmd, hub, method, params):
    _release(cmd, hub)
    reply = cmd.send(method, params)
    assert reply["error"]["code"] == fd.CONTROL_REQUIRED
    assert "control_required" in reply["error"]["message"]


def test_abortwrite_is_token_free(cmd, hub):
    """Like `abort` and `safePark`: an emergency stop must never be blocked by
    arbitration, from either side. Nothing is running, so this is a bare park."""
    _release(cmd, hub)
    res = cmd.call("abortWrite")
    assert res["ok"] is True
    assert res["state"] in ("parked", "idle", "fault")
    assert "safe_park" in res


# ============================================================ 3. loadPattern ====

def test_loadpattern_from_a_file_path(cmd, read, tmp_path):
    """Plan Sec H's script loads `source="D:/designs/hall_bar_v7.oas"` -- an absolute
    path outside the service's own tree. The WIRE door allows that; the cockpit's door
    stays confined to data/designs (asserted below)."""
    design = tmp_path / "leads.svg"
    design.write_text(TWO_OBJECT_SVG, encoding="utf-8")
    res = _load(cmd, str(design))
    assert res["name"] == "leads"
    assert res["object_count"] == 2
    assert [o["name"] for o in res["objects"]] == ["leads-1", "leads-2"]
    assert res["total_points"] > 0
    assert res["segments"] == sum(o["segments"] for o in res["objects"]) >= 2
    x0, y0, x1, y1 = res["bbox_um"]
    assert x1 > x0 and y1 >= y0
    # `design` -- the full geometry -- is deliberately NOT on the wire: tens of
    # thousands of points belong to `getPattern`, which reports COUNTS.
    assert "design" not in res
    pattern = read.call("getPattern")
    assert pattern["name"] == "leads"
    assert [o["name"] for o in pattern["objects"]] == ["leads-1", "leads-2"]
    assert all(o["points"] > 0 for o in pattern["objects"])


def test_loadpattern_from_inline_svg(cmd, read):
    """A script that generates geometry should not have to write a file first."""
    res = _load(cmd, TWO_OBJECT_SVG)
    assert res["object_count"] == 2
    assert res["total_points"] > 0
    assert read.call("getPattern")["name"] == "inline"


def test_loadpattern_from_a_design_name(cmd, read):
    """The cockpit's own idiom -- a name inside data/designs -- keeps working."""
    res = _load(cmd, "demo.oas")
    assert res["object_count"] >= 1
    assert read.call("getPattern")["objects"]


def test_loadpattern_arms_nothing(cmd, read, hub):
    """"Compile and validate only; arms nothing" (plan Sec A). No claim, no thread, no
    run record, and the state machine does not move."""
    before = read.call("getState")
    _load(cmd, TWO_OBJECT_SVG)
    assert read.call("getState") == before
    assert getattr(hub, "_run_claim", None) is None
    assert getattr(hub.host, "litho_thread", None) is None or \
        not hub.host.litho_thread.is_alive()


def test_loadpattern_chunks_an_over_cap_object_instead_of_refusing_it(
        cmd, read, hub, monkeypatch):
    """NO SIZE REFUSAL (decision of 2026-09-11, superseding plan Sec A). A big
    object is not refused: it is armed as MORE OUT-WAVES -- a seamless chain of chunks
    -- while staying ONE stroke, one continuous tip path. The out-wave cap is now the
    only thing that chunks a write; the compiler's old 8,192-point cut is gone (it was
    a lift in the middle of a line), so `max_segment_points` is gone from this wire."""
    logs = _spy_on_logs(hub, monkeypatch)
    res = _load(cmd, _over_cap_svg())
    assert res["object_count"] == 1
    assert res["total_points"] > litho_cmds.MAX_OUTWAVE_POINTS
    assert "max_segment_points" not in res
    assert res["max_outwave_points"] == 87000
    # THE PUBLISHED COUNT IS THE REAL ONE: out-waves armed = chain chunks.
    entry = res["objects"][0]
    assert res["segments"] == entry["segments"]
    # against the CHUNK size (86,976 = 2718*32), which is what a chain is cut to --
    # NOT the 87,000 arm cap. The two agree for this design, so assert the one that
    # is actually load-bearing.
    assert entry["segments"] == -(-entry["points"] // CHAIN_CHUNK_POINTS) >= 2
    # ...and it is still ONE STROKE: the tip goes down once and comes up once.
    assert entry["strokes"] == 1 == res["strokes"]
    assert "chunks" not in res and "chunked_objects" not in res
    multi = res["multi_segment_objects"]
    assert len(multi) == 1
    assert multi[0]["name"] == entry["name"]            # BY NAME, not by index
    assert multi[0]["points"] > litho_cmds.MAX_OUTWAVE_POINTS
    assert multi[0]["segments"] == res["segments"]
    assert multi[0]["strokes"] == 1
    assert multi[0]["chain_boundaries"] == res["segments"] - 1
    assert multi[0]["max_outwave_points"] == 87000
    # It is LOADED -- the next startWrite acts on it.
    pattern = read.call("getPattern")
    assert pattern["objects"][0]["name"] == entry["name"]
    assert pattern["objects"][0]["segments"] == res["segments"]
    assert pattern["objects"][0]["strokes"] == 1
    # ...and the operator is WARNED, by name, that a CHAIN is coming -- and that the
    # chain has never run on hardware. NOT warned about a lift: there isn't one.
    warn = [ln for ln in logs if "CHAIN of chunks" in ln]
    assert len(warn) == 1 and entry["name"] in warn[0]
    # the warning must name the ACTUAL hazard. "no lift" is not reassurance here:
    # the tip sits still at full bias for seconds while Igor re-arms.
    assert "WARNING" in warn[0]
    assert "STATIONARY AT FULL BIAS" in warn[0] and "DOSE BLOB" in warn[0]
    assert "SECONDS" in warn[0]
    assert "never run on the instrument" in warn[0]
    assert "lift / move / set-down" not in warn[0]


def test_loadpattern_leaves_a_normal_design_unchunked_and_unwarned(
        cmd, read, hub, monkeypatch):
    """The other half of the rule: a small design is one segment per object, nothing
    listed as multi-segment, and no warning."""
    logs = _spy_on_logs(hub, monkeypatch)
    res = _load(cmd, TWO_OBJECT_SVG)
    assert res["segments"] == 2 == res["object_count"]   # one out-wave per object
    assert [o["segments"] for o in res["objects"]] == [1, 1]
    assert res["multi_segment_objects"] == []
    assert [o["segments"] for o in read.call("getPattern")["objects"]] == [1, 1]
    assert not [ln for ln in logs if "SEVERAL SEGMENTS" in ln]


def test_loadpattern_refusal_leaves_the_previous_design_in_place(cmd, read):
    """A refused load must not half-replace the loaded design: the next `startWrite`
    would then write something nobody asked for. An over-cap object is no longer a
    refusal, so the refusal under test is a design that does not exist."""
    _load(cmd, TWO_OBJECT_SVG)
    before = read.call("getPattern")
    cmd.send("loadPattern", {"token": _token(cmd), "source": "does-not-exist.oas"})
    assert read.call("getPattern") == before


def test_loadpattern_says_which_design_it_could_not_find(cmd):
    reply = cmd.send("loadPattern", {"token": _token(cmd),
                                     "source": "does-not-exist.oas"})
    assert reply["error"]["code"] == fd.INVALID_PARAMS
    assert "no such design" in reply["error"]["message"]


def test_the_cockpit_door_stays_confined_to_data_designs(client, tmp_path):
    """The wire door takes a path; the HTTP door never did and must not start. This is
    the ONE thing the two doors resolve differently, and it is deliberate."""
    design = tmp_path / "outside.svg"
    design.write_text(TWO_OBJECT_SVG, encoding="utf-8")
    body = client.post("/api/litho/load", json={"source": str(design)}).json()
    assert "no such design" in body.get("error", "")


# =========================================================== 4. startWrite ======

def test_startwrite_runs_and_the_result_is_retrievable(cmd, read, hub):
    """THE HAPPY PATH, end to end: start+poll, then the terminal record (plan Sec A.1,
    F.1). `completed` is gated on persistence, so what `files` lists is whole."""
    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": TWO_OBJECT_SVG})
    res = cmd.call("startWrite", {"token": token, "objects": "all", **_gw()})
    assert res["run_id"].startswith("write-")
    assert res["objects"] == 2
    assert res["est_s"] > 0
    assert res["state"] == "writing"
    # The state machine says `writing` from the moment the start verb returns.
    assert read.call("getState") == "writing"
    rec = _wait_run(hub, res["run_id"])
    assert rec.completed and not rec.aborted, rec.error
    out = read.call("getWriteResult", {"run_id": res["run_id"], "inline": True})
    assert out["kind"] == "write"
    assert out["completed"] is True and out["aborted"] is False
    assert out["stats"]["objects"] == 2
    assert out["stats"]["objects_submitted"] == 2
    assert isinstance(out["files"], dict)
    # THE TRACE goal 2 cares about: the written path with the tip voltage along it,
    # inline and bounded (plan Sec A: "conductance-along-path is the payload").
    trace = out["trace"]
    assert len(trace) == 2
    for obj in trace:
        assert obj["path_um"] and obj["v_tip"]
        assert len(obj["path_um"]) == len(obj["v_tip"])
        assert len(obj["path_um"]) <= litho_cmds.TRACE_MAX_POINTS_PER_OBJECT


def test_startwrite_writes_an_over_cap_object_whole_on_the_twin(cmd, read, hub):
    """The other half of the 2026-09-11 decision: loading an over-cap object is not
    enough, it has to be WRITTEN, whole, with no geometry dropped at a seam.

    The write is driven at an unphysical 2000 um/s so the twin's simulated clock (the
    real speed would be an eight-minute pass) finishes in seconds; nothing about the
    chunking depends on speed -- the point count is arc length over the interpolation
    spacing, not over time.

    `getWritten` is per STROKE, and an over-cap object is ONE stroke however many
    out-waves it takes, so a single entry is the assertion: the chain's chunks are
    arms, not lifts, and must not show up as separate written pieces.
    """
    token = _token(cmd)
    res = cmd.call("loadPattern", {"token": token, "source": _over_cap_svg()})
    assert res["segments"] >= 2
    name = res["multi_segment_objects"][0]["name"]
    x0, y0, x1, y1 = res["bbox_um"]
    hub.pattern.objects[0].speed_um_s = 2000.0         # simulated-clock relief only
    started = cmd.call("startWrite", {"token": token, "objects": "all", **_gw()})
    assert started["objects"] == 1
    assert started["segments"] == res["segments"]      # the seams are PUBLISHED
    rec = _wait_run(hub, started["run_id"])
    assert rec.completed and not rec.aborted, rec.error

    out = read.call("getWriteResult", {"run_id": started["run_id"], "inline": True})
    assert out["completed"] is True and out["aborted"] is False
    # THE WHOLE PATH: the union of the trace spans the design's own bounding box, so
    # no chunk was dropped at either end and no seam swallowed a middle piece.
    xs = [p[0] for obj in out["trace"] for p in obj["path_um"]]
    ys = [p[1] for obj in out["trace"] for p in obj["path_um"]]
    assert min(xs) == pytest.approx(x0, abs=0.5) and max(xs) == pytest.approx(x1, abs=0.5)
    assert min(ys) == pytest.approx(y0, abs=0.5) and max(ys) == pytest.approx(y1, abs=0.5)

    written = read.call("getWritten")["objects"]
    # ONE ENTRY PER STROKE -- and this object is one stroke, chained or not.
    assert res["segments"] >= 2                         # it really did chain
    assert len(written) == res["strokes"] == 1
    assert {e["object"] for e in written} == {name}
    assert {e["segments"] for e in written} == {res["segments"]}
    assert {e["name"].split("[")[0] for e in written} == {name}
    # ...and the object itself is marked written, once.
    marked = [o for o in read.call("getPattern")["objects"] if o["written"]]
    assert [o["name"] for o in marked] == [name]


def test_the_published_segments_match_the_report_and_the_twin_plays_each_stroke_whole(
        cmd, read, hub, monkeypatch):
    """`segments` is the number of out-waves the INSTRUMENT arms for the write; the Asylum
    plugin chunks an over-cap stroke (its goldens pin the seams), the twin does not
    (2026-10-27: the generic twin plays a whole object in one go). So the twin gets
    one `write` per stroke, however many segments the stroke would chain into."""
    from afm_litho.service.twin_plugin import plugin
    token = _token(cmd)
    res = cmd.call("loadPattern", {"token": token, "source": _over_cap_svg()})
    hub.pattern.objects[0].speed_um_s = 2000.0          # simulated-clock relief only
    twin = plugin(hub)
    plays = []
    real_write = twin.write

    def spy(command):
        plays.append(len(command.x_um))
        return real_write(command)

    monkeypatch.setattr(twin, "write", spy)
    started = cmd.call("startWrite", {"token": token, "objects": "all", **_gw()})
    rec = _wait_run(hub, started["run_id"])
    assert rec.completed and not rec.aborted, rec.error
    assert started["segments"] == res["segments"] >= 2
    assert plays == [res["objects"][0]["points"]] and res["strokes"] == 1


def test_startwrite_writes_only_the_objects_it_was_given(cmd, read, hub):
    """`objects: [idx]`. A script naming one object must get one object written --
    and `getWritten` must agree, because that is what a rig check compares."""
    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": TWO_OBJECT_SVG})
    names = [o["name"] for o in read.call("getPattern")["objects"]]
    res = cmd.call("startWrite", {"token": token, "objects": [1], **_gw()})
    assert res["objects"] == 1
    rec = _wait_run(hub, res["run_id"])
    assert rec.completed and not rec.aborted, rec.error
    written = read.call("getWritten")
    assert [o["name"] for o in written["objects"]] == [names[1]]
    assert written["run_id"] == res["run_id"]
    assert written["aborted"] is False


def test_getwriteresult_without_a_run_id_returns_the_last_completed_write(cmd, read,
                                                                          hub):
    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": TWO_OBJECT_SVG})
    res = cmd.call("startWrite", {"token": token, "objects": "all", **_gw()})
    _wait_run(hub, res["run_id"])
    assert read.call("getWriteResult")["run_id"] == res["run_id"]


# ============================================================= 5. refusals ======

def test_a_second_write_is_refused_busy_with_the_run_id(cmd, hub):
    """A start verb arriving while busy errors -32020 and NEVER queues (plan Sec F.1).
    The single-writer claim is taken before the first await, so the second caller is
    refused HERE rather than discovering the collision on the instrument."""
    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": TWO_OBJECT_SVG})
    first = cmd.call("startWrite", {"token": token, "objects": "all", **_gw(),
                                    **IN_FLIGHT})
    try:
        reply = cmd.send("startWrite", {"token": token, "objects": "all", **_gw()})
        assert reply["error"]["code"] == fd.BUSY
        assert "busy" in reply["error"]["message"]
        assert reply["error"]["data"]["run_id"] == first["run_id"]
        assert reply["error"]["data"]["kind"] == "write"
    finally:
        _wait_run(hub, first["run_id"])


def test_startwrite_refuses_when_no_pattern_is_loaded(cmd, hub):
    """The cockpit substitutes the demo design when nothing is loaded (a human can see
    what appeared); the WIRE door never does. A script that meant to load a design and
    did not must be told, not handed a demo pattern to write on a real sample."""
    token = _token(cmd)
    saved = getattr(hub, "pattern", None)
    hub.pattern = None
    try:
        reply = cmd.send("startWrite", {"token": token, "objects": "all", **_gw()})
        assert reply["error"]["code"] == fd.REFUSED
        assert reply["error"]["data"]["reason"] == "no_pattern"
        assert "loadPattern" in reply["error"]["message"]
    finally:
        hub.pattern = saved


def test_startwrite_names_an_out_of_range_object(cmd):
    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": TWO_OBJECT_SVG})
    reply = cmd.send("startWrite", {"token": token, "objects": [9], **_gw()})
    assert reply["error"]["code"] == fd.INVALID_PARAMS
    assert "out of range" in reply["error"]["message"]
    assert reply["error"]["data"]["index"] == 9


def test_startwrite_deadman_s_rearms_the_commander_clamped(cmd, hub):
    """A write is minutes long, and re-acquiring control to lengthen the window would
    mean dropping the token on the one path where dropping it is worst. Clamped by
    the same [5, 300] rule `acquireControl` uses -- never disabled."""
    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": TWO_OBJECT_SVG})
    res = cmd.call("startWrite", {"token": token, "objects": "all",
                                  "deadman_s": 9000, **_gw()})
    try:
        assert hub.control.deadman_s == 300.0      # clamped, not honoured, not refused
    finally:
        _wait_run(hub, res["run_id"])


def test_startwrite_refuses_a_non_numeric_deadman(cmd):
    """A NUMBER out of range is clamped (clamping fails safe); a typo is -32602.
    Substituting a default would hand a script a dead-man it never asked for."""
    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": TWO_OBJECT_SVG})
    reply = cmd.send("startWrite", {"token": token, "objects": "all",
                                    "deadman_s": "a while", **_gw()})
    assert reply["error"]["code"] == fd.INVALID_PARAMS
    assert reply["error"]["data"]["param"] == "deadman_s"


def test_startwrite_refuses_a_meaningless_objects_value(cmd):
    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": TWO_OBJECT_SVG})
    for bad in ("some", 3, {"all": True}):
        reply = cmd.send("startWrite", {"token": token, "objects": bad})
        assert reply["error"]["code"] == fd.INVALID_PARAMS, bad


def test_both_doors_refuse_an_out_of_range_object_identically(cmd, client):
    """IDENTICAL VALIDATION, PROVEN (plan Sec C/D). The cockpit's endpoint and the
    wire verb call ONE `select_objects`, so the refusal is the same refusal -- the
    same sentence, not two implementations that happen to agree today."""
    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": TWO_OBJECT_SVG})
    wire = cmd.send("startWrite", {"token": token, "objects": [9], **_gw()})["error"]
    http = client.post("/api/litho/write", json={"objects": [9]}).json()
    assert http["error"] == wire["message"]
    assert http["reason"] == "bad_params"


def test_a_refused_startwrite_does_not_clear_the_written_marks(cmd, hub):
    """`objects: "all"` clears the cockpit's per-session "already written" set -- but
    only when a write really STARTS.

    Two places got this wrong. `startWrite` sized the run before it claimed anything,
    and that sizing cleared the set, so a BUSY refusal wiped the marks. And the
    hardware AC guard ran AFTER the selection, so the one refusal an operator actually
    meets on the rig -- write, get told to engage in contact -- wiped them too, and
    the panel's next default write silently re-dosed the whole design."""
    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": TWO_OBJECT_SVG})
    first = cmd.call("startWrite", {"token": token, "objects": "all", **_gw(),
                                    **IN_FLIGHT})
    try:
        hub.host.written_objects.add("sentinel")       # stand-in for an earlier write
        reply = cmd.send("startWrite", {"token": token, "objects": "all", **_gw()})
        assert reply["error"]["code"] == fd.BUSY
        assert "sentinel" in hub.host.written_objects
    finally:
        _wait_run(hub, first["run_id"])


@pytest.mark.parametrize("kind", ["ac_mode", "backend_mismatch"])
def test_a_hardware_refusal_leaves_the_written_marks_intact(hub, kind):
    """The same property for the two refusals that only exist on hardware, asserted
    against the service layer with a hub that merely SAYS it is hardware."""
    class _Arc:
        def check_mode(self):
            return "ac"

    stub = _HwHub(live=(kind == "ac_mode"))
    if kind == "ac_mode":
        stub.hardware_arc = _Arc()
    stub.pattern = hub.pattern
    stub.written_objects = {"sentinel"}
    # Command guard (INSTRUMENT_MODEL_SPEC secs 2/10, 13): give the stub a complete
    # model + a fresh withdrawn free-air baseline and carry amp_gain, so the guard
    # PASSES and the hardware-only refusal under test (ac_mode / backend_mismatch,
    # both raised past the guard) is the one that fires. backend_mismatch refuses at
    # the live-backend gate before the guard, so these params are inert for it.
    stub.instrument_model = make_model()
    seed_free_air(stub)
    with pytest.raises(CommandError):
        asyncio.run(litho_cmds.litho_write(
            stub, {"objects": "all", "setpoint_relative": 0.2, "amp_gain": 25.0}))
    assert stub.written_objects == {"sentinel"}


def test_objects_are_marked_written_at_COMPLETION_not_at_start(cmd, read, hub):
    """A write used to mark every submitted object the moment it STARTED, so a run
    aborted after one object left the cockpit believing the whole design was written
    -- and the operator's next default write skipped geometry that was never dosed.

    Marks now land per object, as each one completes."""
    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": TWO_OBJECT_SVG})
    hub.host.written_objects = set()
    res = cmd.call("startWrite", {"token": token, "objects": "all", **_gw()})
    assert not hub.host.written_objects, "nothing is written yet at start"
    rec = _wait_run(hub, res["run_id"])
    assert rec.completed and not rec.aborted, rec.error
    # Both objects completed, so both are marked -- and `getPattern` says so.
    assert len(hub.host.written_objects) == 2
    assert all(o["written"] for o in read.call("getPattern")["objects"])


def test_startwrite_refuses_a_backend_mismatch(hub):
    """THE SILENT-TWIN GATE (plan Sec C.2). A station CONFIGURED as hardware whose
    bridge is not live must never "write" against simulated physics and report
    ordinary success. Asserted on the service layer with a hub that only SAYS it is
    hardware -- it is a refusal, so nothing is driven."""
    stub = _HwHub(live=False)
    stub.pattern = hub.pattern
    with pytest.raises(CommandError) as exc:
        asyncio.run(litho_cmds.litho_write(stub, {"all": True}))
    assert exc.value.code == fd.BACKEND_MISMATCH
    assert exc.value.http_body()["reason"] == "backend_not_live"
    assert any("REFUSED" in line for line in stub.logs)


def test_startwrite_refuses_ac_mode(monkeypatch, hub):
    """LITHO IS CONTACT-ONLY. The Z loop must never servo on amplitude with a bias on
    the tip. Asserted against the service layer with a stubbed guard, because the
    crosspoint the real guard reads only exists on the instrument."""
    class _Arc:
        def check_mode(self):
            return "ac"

    stub = _HwHub(live=True)
    stub.hardware_arc = _Arc()
    stub.pattern = hub.pattern
    stub.written_objects = set()
    # Guard params so the write reaches the AC refusal, which is raised past the
    # command guard (INSTRUMENT_MODEL_SPEC secs 2/10, 13).
    stub.instrument_model = make_model()
    seed_free_air(stub)
    with pytest.raises(CommandError) as exc:
        asyncio.run(litho_cmds.litho_write(
            stub, {"objects": "all", "setpoint_relative": 0.2, "amp_gain": 25.0}))
    assert exc.value.code == fd.REFUSED
    assert exc.value.data["reason"] == "ac_mode_active"
    assert "contact-only" in exc.value.message
    assert any("AC/tapping" in line for line in stub.logs)
    # ...and it released the claim it took: a refusal must leave nothing wedged.
    assert getattr(stub, "_run_claim", None) is None


def test_an_unreadable_crosspoint_never_downgrades_to_contact():
    """THE FAIL-OPEN ORDERING, PINNED (plan Sec J risk 11).

    "unknown" is not "contact". A crosspoint read that FAILS must fall through to the
    engaged-servo branch and then to the conservative one -- never be taken as proof
    the instrument is in contact. This is the difference between "cannot prove we are
    in AC, so refuse" and "cannot prove anything, so proceed", and it moved verbatim
    when the guard was extracted; this case is what stops a later refactor reordering
    it quietly.
    """
    class _Raises:
        def check_mode(self):
            raise OSError("bridge is wedged")

    class _Unknown:
        def check_mode(self):
            return "unknown"

    class _Hub:
        def __init__(self, arc, **kw):
            self.hardware_arc = arc
            self.engaged = False
            self.z_loop = {}
            self.imaging_mode = None
            self.__dict__.update(kw)

        @property
        def host(self):                # fake: host fields live on the fake itself
            return self

    # A read that RAISES is "unknown", and an AC engage underneath it still refuses.
    assert litho_cmds.check_mode(_Hub(_Raises())) == "unknown"
    assert litho_cmds.ac_mode_active(
        _Hub(_Raises(), engaged=True, z_loop={"mode": "ac"})) is True
    # Unknown + not engaged + the last COMMANDED mode was AC -> conservative refusal.
    assert litho_cmds.ac_mode_active(
        _Hub(_Unknown(), imaging_mode="ac")) is True
    # And the authoritative branch still WORKS: a contact crosspoint allows, over a
    # stale AC engage. (The read is the truth when there is one.)
    class _Contact:
        def check_mode(self):
            return "contact"

    assert litho_cmds.ac_mode_active(
        _Hub(_Contact(), engaged=True, z_loop={"mode": "ac"})) is False


# ============================================================== 6. abortWrite ===

def test_abortwrite_mid_write_parks_and_keeps_the_partial_result(cmd, read, hub):
    """A mid-write abort must get the tip OFF the surface AND leave the run
    retrievable (plan Sec E.3): `aborted: true`, the objects that did complete listed,
    and the partial trace still inline. A stop that loses the partial result makes a
    script re-write geometry it has already dosed.

    The twin plays a stroke in the clock's time, so racing `abortWrite` against it from the
    client side is not reliable: it can land before EITHER of these two short lines has
    finished, giving 0 completed objects, which would not exercise the object-boundary
    contract this test is about. So the worker is gated deterministically instead: the twin
    plugin's 2nd `write` call is the instant the second object is about to start. The
    worker is parked there, the REAL `abortWrite` RPC runs (exercising the actual control
    path -- claim, retract, run record), and only then is the worker released: the plugin
    sees the stop that was signalled while it was starting and the object is withheld.
    """
    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": TWO_OBJECT_SVG})

    from afm_litho.service.twin_plugin import plugin
    gate = threading.Event()
    reached_second = threading.Event()
    twin = plugin(hub)
    real_write = twin.write
    calls = {"n": 0}

    def gated_write(command):
        calls["n"] += 1
        if calls["n"] == 2:
            reached_second.set()
            gate.wait(timeout=10.0)
        return real_write(command)

    twin.write = gated_write
    try:
        res = cmd.call("startWrite", {"token": token, "objects": "all", **_gw()})
        assert read.call("getState") == "writing"
        assert reached_second.wait(timeout=10.0), \
            "the second object never reached the plugin"
        park = cmd.call("abortWrite", {"token": token})
        assert park["ok"] is True
    finally:
        gate.set()                                  # let the worker thread proceed
        del twin.write
    rec = _wait_run(hub, res["run_id"])
    assert rec.aborted is True
    assert read.call("getState") in ("parked", "fault")
    out = read.call("getWriteResult", {"run_id": res["run_id"], "inline": True})
    assert out["aborted"] is True
    assert out["run_id"] == res["run_id"]
    # `getWritten` lists the objects that COMPLETED -- never the one the abort cut.
    # Before the fix, the off-by-one in `LithoRun.run` reported the cut object as
    # done too, making this 2.
    written = read.call("getWritten")
    assert written["aborted"] is True
    assert len(written["objects"]) == 1
    assert written["objects"][0]["name"] == hub.pattern.objects[0].name
    assert len(out.get("trace") or []) == len(written["objects"])
    # the completed object is marked written; the one the abort cut is NOT -- an
    # under-marked object safely gets re-covered by the next default write, which
    # is the point of docs/FLEX_INTEGRATION_PLAN.md's `_marks` completion contract.
    assert id(hub.pattern.objects[0]) in hub.host.written_objects
    assert id(hub.pattern.objects[1]) not in hub.host.written_objects
    assert getattr(hub, "_run_claim", None) is None      # the claim was released


# =================================== 7. the pump's executor hold (slice 07 review) ==

def test_a_pump_whose_worker_dies_closes_the_run_instead_of_hanging(hub):
    """THE UNTIMED EXECUTOR HOLD, as a regression guard.

    `_litho_pump_body` used to drain its worker queue with
    `await loop.run_in_executor(None, q.get)` -- one shared default-pool thread parked
    on a blocking get for the WHOLE write. On the instrument a pump whose worker dies
    mid-write parks that thread forever, and enough of those starve every
    `asyncio.to_thread` in the service, INCLUDING `hw_withdraw`: the call that gets the
    tip off the surface would queue behind a bookkeeping leak.

    So: a worker that exits without ever queuing `litho_done`. The pump must notice,
    close the record as aborted with a reason, release the claim, and RETURN -- within
    a couple of seconds, not never.
    """
    from afm_litho.service.host.litho_worker import _litho_pump

    q = _queue.Queue()
    worker = threading.Thread(target=lambda: None, daemon=True)   # dies immediately
    claim = hub.claim_run("write")
    run_id = hub.next_run_id("write")
    hub.runs.annotate(run_id, stats={"objects_submitted": 1})
    claim.hand_over(run_id, worker)
    worker.start()
    worker.join(timeout=5)

    started = time.monotonic()
    asyncio.run(asyncio.wait_for(_litho_pump(q, run_id, claim), timeout=20))
    elapsed = time.monotonic() - started

    assert elapsed < 10.0, f"the pump took {elapsed:.1f}s to notice a dead worker"
    rec = hub.runs.get(run_id)
    assert rec is not None and rec.finished is not None, "the record never closed"
    assert rec.aborted is True
    assert "without signalling completion" in (rec.error or "")
    assert hub._run_claim is None, "the claim leaked -- every later write is refused"


def test_the_pump_never_parks_a_default_executor_thread(hub):
    """The same property, stated as the INVARIANT rather than the symptom: while the
    pump is draining, `asyncio.to_thread` must still be serviced promptly. A pump
    holding the pool hostage is exactly what starves the fail-safe."""
    from afm_litho.service.host.litho_worker import _litho_pump

    q = _queue.Queue()
    worker = threading.Thread(target=lambda: None, daemon=True)
    claim = hub.claim_run("write")
    run_id = hub.next_run_id("write")
    claim.hand_over(run_id, worker)
    worker.start()
    worker.join(timeout=5)

    async def _go():
        pump = asyncio.create_task(_litho_pump(q, run_id, claim))
        await asyncio.sleep(0.2)                 # the pump is mid-drain here
        t0 = time.monotonic()
        await asyncio.to_thread(lambda: None)    # must not queue behind the pump
        cost = time.monotonic() - t0
        # BOUNDED: a bare `await` here would make a regression HANG the suite instead
        # of failing it, which is the exact failure mode this file exists to catch.
        await asyncio.wait_for(pump, timeout=20)
        return cost

    assert asyncio.run(_go()) < 1.0


# ============================================ 8. the reviewer's findings, pinned ==

def test_withdraw_stops_a_running_write(cmd, read, hub):
    """WITHDRAW IS STOP-AND-RETRACT, and it has to stop the WRITE too.

    It used to set only the scan aborts, so a withdraw during lithography retracted
    the tip and left the write running -- and on hardware that is the dangerous
    direction: `LithographyRun.write_object` SELF-ENGAGES, so the pass's next object
    drives the tip straight back down onto the surface the operator just withdrew
    from. Asserted on the twin through the write's abort flag and the plugin, which a
    withdraw must both stop.
    """
    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": TWO_OBJECT_SVG})
    res = cmd.call("startWrite", {"token": token, "objects": "all", **_gw(),
                                  **IN_FLIGHT})
    try:
        assert read.call("getState") == "writing"
        assert hub.host._twin_plugin is not None and not hub.host._hw_litho_abort.is_set()
        cmd.call("withdraw", {"token": token})
        assert hub.host._hw_litho_abort.is_set(), "withdraw left the write running"
        rec = _wait_run(hub, res["run_id"])
        assert rec.aborted is True
    finally:
        _wait_write_done(hub)


def test_the_out_of_band_park_stops_a_twin_write_too():
    """The park primitive STEP 1 sets every cooperative abort and tells the twin plugin,
    which has no event to watch. Without that a dead-man park or a bridge-drop park stopped
    a HARDWARE write and let a TWIN write run to completion behind a `parked` state.
    Asserted on the thread-safe primitives themselves, because the watchdog thread is what
    calls them.
    """
    from types import SimpleNamespace

    from afm_litho.service.host.safe_park import _signal_stop, signal_run_abort

    class _Hub:
        def __init__(self):
            self.scan_repeat: dict = {}
            self._twin_plugin = SimpleNamespace(aborted=False)
            self._twin_plugin.abort = lambda reason: setattr(self._twin_plugin, "aborted", True)

        @property
        def host(self):                # fake: host fields live on the fake itself
            return self

    for fn in (signal_run_abort, _signal_stop):
        stub = _Hub()
        fn(stub)
        assert stub._twin_plugin.aborted is True, fn.__name__


def test_withdraw_does_not_stop_the_continuous_loop():
    """The other half of that fix, and the reason withdraw signals `signal_run_abort`
    rather than the full stop: the continuous loop between-frame re-reference CALLS
    withdraw, so disabling `scan_repeat` there would make a continuous run stop itself
    after its first frame.
    """
    from afm_litho.service.host.safe_park import signal_run_abort

    class _Hub:
        def __init__(self):
            self.scan_repeat: dict = {"enabled": True}

        @property
        def host(self):                # fake: host fields live on the fake itself
            return self

    stub = _Hub()
    signal_run_abort(stub)
    assert stub.scan_repeat["enabled"] is True


def test_the_state_never_leaves_writing_before_the_result_is_fetchable(cmd, read, hub):
    """PLAN Sec A.1, AS AN INVARIANT SAMPLED ON EVERY POLL.

    `getState` derived `writing` from the worker THREAD, so the state left `writing`
    the instant the worker exited -- while the pump was still draining and writing the
    .npz / .gwy / .ibw. A script doing the documented thing (poll until it leaves
    `writing`, then fetch) got `completed: false`, and on hardware raced the file
    writer for the whole persistence block.
    """
    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": TWO_OBJECT_SVG})
    res = cmd.call("startWrite", {"token": token, "objects": "all", **_gw()})
    run_id = res["run_id"]
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        if read.call("getState") != "writing":
            out = read.call("getWriteResult", {"run_id": run_id})
            assert out["completed"] or out["aborted"], (
                "the state left `writing` while the run was still open -- a script "
                "polling as documented would fetch an unfinished result")
            return
        time.sleep(0.02)
    raise AssertionError("the write never left `writing`")


def test_the_pump_task_is_referenced_so_it_cannot_be_collected(cmd, hub):
    """The `create_task` result was dropped on the floor. The loop keeps only a WEAK
    reference to a running task, so a mid-flight pump was collectable -- and a
    collected pump is a run record that never closes and a claim that never frees.
    """
    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": TWO_OBJECT_SVG})
    res = cmd.call("startWrite", {"token": token, "objects": "all", **_gw()})
    try:
        assert hub._litho_pump_task is not None
    finally:
        _wait_run(hub, res["run_id"])


def test_the_claim_is_held_across_the_start_path(cmd, hub):
    """THE CLAIM IS PINNED. Every other case in this file passes with the claim
    disabled, because they observe its EFFECT (a busy refusal) rather than the claim
    itself. This asserts the claim exists, names the run, and is handed to a live
    worker -- which is what makes -32020 mean anything.
    """
    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": TWO_OBJECT_SVG})
    res = cmd.call("startWrite", {"token": token, "objects": "all", **_gw()})
    try:
        claim = hub._run_claim
        assert claim is not None, "the start path ran without a single-writer claim"
        assert claim.kind == "write"
        assert claim.run_id == res["run_id"]
        assert claim.worker is not None and claim.worker_alive()
    finally:
        _wait_run(hub, res["run_id"])
    _wait_write_done(hub)
    assert hub._run_claim is None, "the claim outlived the run"


def test_the_inline_trace_is_bounded_for_a_big_design(hub):
    """The per-object cap is not a bound. A 200-object design was ~100k points per run
    and ~2.5 MB on a reply whose published bound is 200 ms.
    """
    written = [{"name": f"o{i}", "voltage": -8.0, "length_um": 1.0, "t": 0.0,
                "path_um": [[float(j), 0.0] for j in range(4000)],
                "v_written": [-8.0] * 4000}
               for i in range(200)]
    litho_cmds.remember_trace(hub, "write-bound-probe", written)
    trace = litho_cmds.trace_for(hub, "write-bound-probe")
    total = sum(len(o["path_um"]) for o in trace)
    # Pin the constant AND assert a literal ceiling separately: asserting only
    # against `litho_cmds.TRACE_MAX_POINTS_PER_RUN` itself means raising the
    # constant passes this test unchanged, silently reopening the 2.5 MB /
    # 200 ms reply the cap exists to prevent.
    assert litho_cmds.TRACE_MAX_POINTS_PER_RUN == 20_000
    assert total <= 20_000, total
    assert total <= litho_cmds.TRACE_MAX_POINTS_PER_RUN, total
    for obj in trace:
        assert len(obj["path_um"]) == len(obj["v_tip"])
        assert obj["decimate"] >= 1           # say so, as everywhere else
    assert len(hub._write_traces) <= litho_cmds.TRACE_KEEP_RUNS


def test_remember_trace_tags_a_partial_entry_and_leaves_others_alone(hub):
    """`_litho_finish` splices the one abort-cut object into the list it hands
    `remember_trace`, tagged `partial: True` -- and only that one. A completed
    object's entry never carries the key at all (not even `False`), so a client
    checking `"partial" in obj` and `obj.get("partial")` agree."""
    written = [
        {"name": "wire", "voltage": 8.0, "length_um": 6.0, "t": 0.0,
         "path_um": [[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]],
         "v_written": [8.0, 8.0, 8.0]},
        {"name": "cut", "voltage": -8.0, "t": 0.0,
         "path_um": [[0.0, 1.0], [0.0, 0.6]],
         "v_written": [-8.0, -8.0], "partial": True},
    ]
    litho_cmds.remember_trace(hub, "write-partial-probe", written)
    trace = litho_cmds.trace_for(hub, "write-partial-probe")
    by_name = {o["name"]: o for o in trace}
    assert "partial" not in by_name["wire"]
    assert by_name["cut"]["partial"] is True


def test_abort_mid_stroke_partial_rides_the_trace_not_the_written_marks(hub):
    """The pump/finish half of the fix (`_litho_pump_body`'s `litho_object_partial`
    case + `_litho_finish`'s splice), exercised directly against the real hub the
    way `test_a_pump_whose_worker_dies_closes_the_run_instead_of_hanging` drives
    `_litho_pump` -- the twin's own tick arithmetic that BUILDS the partial message
    is `test_litho.py`'s job; this is only the routing.

    `litho_object_partial` must land in `getWriteResult(inline=True).trace`
    (tagged `partial: true`) and nowhere else: not `written_accum`
    (`getWritten`), not the persisted `written`/`stats.objects`, not
    `written_objects` (even when a mark IS registered for it -- only
    `litho_object_done` ever sets one).
    """
    from afm_litho.service.host.litho_worker import _litho_pump

    q = _queue.Queue()
    worker = threading.Thread(target=lambda: None, daemon=True)
    claim = hub.claim_run("write")
    run_id = hub.next_run_id("write")
    claim.hand_over(run_id, worker)
    worker.start()
    worker.join(timeout=5)

    hub.host.written_objects = set()
    marks = {"wire": {"id": "wire-id", "expect": 1},
             "cut": {"id": "cut-id", "expect": 1}}
    q.put({"type": "litho_object_done", "name": "wire", "object": "wire",
           "object_frac0": 0.0, "object_frac1": 1.0, "profile": None,
           "voltage": 8.0, "duration_s": 1.0, "length_um": 6.0,
           "path_um": [[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]],
           "v_written": [8.0, 8.0, 8.0]})
    q.put({"type": "litho_object_partial", "name": "cut", "object": "cut",
           "object_index": 1, "voltage": -8.0, "frac0": 0.0, "frac1": 0.4,
           "path_um": [[0.0, 1.0], [0.0, 0.6]], "v_written": [-8.0, -8.0]})
    q.put({"type": "litho_done", "aborted": True})

    try:
        asyncio.run(asyncio.wait_for(_litho_pump(q, run_id, claim, marks),
                                     timeout=20))

        rec = hub.runs.get(run_id)
        assert rec is not None and rec.aborted is True
        assert rec.stats["objects"] == 1          # "cut" never counted as written
        assert hub.host.written_objects == {"wire-id"}   # marked; "cut-id" is NOT

        trace = litho_cmds.trace_for(hub, run_id)
        by_name = {o["name"]: o for o in trace}
        assert set(by_name) == {"wire", "cut"}
        assert "partial" not in by_name["wire"]
        assert by_name["cut"]["partial"] is True
        assert len(by_name["cut"]["path_um"]) == 2
    finally:
        hub.host.written_objects = set()


def test_an_uncompilable_design_is_a_refusal_on_both_doors(cmd, client, monkeypatch):
    """The compile moved INTO loading, so a file that reads cleanly but will not
    compile (degenerate geometry, a fill that cannot be generated) would otherwise
    have become an HTTP 500 on the cockpit own Load button.
    """
    def _boom(*a, **kw):
        raise ValueError("degenerate fill")

    monkeypatch.setattr(litho_cmds, "_compile_report", _boom)
    reply = cmd.send("loadPattern", {"token": _token(cmd), "source": "auto"})
    assert reply["error"]["code"] == fd.INVALID_PARAMS
    assert "could not compile" in reply["error"]["message"]
    body = client.post("/api/litho/load", json={"source": "auto"}).json()
    assert "could not compile" in body.get("error", "")     # HTTP 200, not a 500


def test_a_failed_end_of_run_withdraw_escalates_to_the_failsafe(hub, monkeypatch):
    """CLAUDE.md rule: any failure during a hardware pass is a stop AND a verified
    withdraw. The hardware worker logged a failed end-of-run withdraw and closed the
    run merely `aborted` -- no safe_park, no fault latch, `hub.engaged` left True,
    with the tip on the surface at bias and the Z loop closed.

    Driven through the worker own error path (`hardware_arc` is a bare object, so
    `LithographyRun` raises before anything is sent anywhere) with a `hw_withdraw`
    that also raises.
    """
    from afm_litho.hardware import hw_approach
    from afm_litho.service.host import litho_worker

    def _raise(*a, **kw):
        raise OSError("bridge gone")

    monkeypatch.setattr(hw_approach, "hw_withdraw", _raise)
    monkeypatch.setattr(hub.host, "hardware_arc", object(), raising=False)
    saved_fault = hub.fault
    hub.fault = None
    hub.host._safe_park_now.clear()
    q = _queue.Queue()
    try:
        litho_worker._hw_litho_worker(q, hub.pattern, 0.2, 1.0)
        assert hub.host._safe_park_now.is_set(), "the fail-safe was never triggered"
        assert (hub.fault or {}).get("reason") == "withdraw_failed"
        msgs = []
        while not q.empty():
            msgs.append(q.get_nowait())
        done = [m for m in msgs if m["type"] == "litho_done"]
        assert done and done[0]["aborted"] is True
        assert "withdraw failed" in (done[0].get("error") or "")
    finally:
        hub.fault = saved_fault
        hub.host._safe_park_now.clear()
        hub.host.engaged = False


# ============================ 9. loadPattern by PARAMETERS (`objects`) ===========
#
# THE CAPSTONE. `source` carries geometry and nothing else -- SVG, OASIS and GDS have
# nowhere to put a tip voltage -- so the two experiments this service exists for could
# be DRAWN by a script but not ORDERED by one:
#
#   * a CHIRAL nanowire: a line with a sinusoidal spatial deviation, written with the
#     tip voltage modulated at a chosen PHASE relative to that deviation. phi = +/-90
#     are the two enantiomers, phi = 0 the achiral control -- so a phase SWEEP is the
#     experiment, and the phase has to be a parameter on the wire.
#   * an ERASE rectangle: a small filled region written at a NEGATIVE bias.
#
# What these cases pin, in order of what would actually break:
#   1. the FLEX door builds through the SAME builders the HTTP door uses
#      (`chiral_profile`, the region pen), so the phase convention cannot fork;
#   2. `v_written` on the twin really is `v0 + v_k sin(theta + phi)` -- asserted
#      against the GEOMETRY, not against a number this test also computed, so a
#      builder that dropped the phase would fail here;
#   3. an erase rectangle is written at the negative voltage, every point of it;
#   4. a refused spec leaves the previously loaded pattern alone;
#   5. an over-cap chiral CHAINS (the rule: chunk, don't refuse).

CHIRAL_SPEC = {
    "kind": "chiral", "name": "helix",
    # A STRAIGHT base axis, and a whole number of periods along it (2 um / 0.1 um =
    # 20). Both matter for the assertions below: on a straight axis the signed
    # deviation from the chord IS `y_amp_um * sin(theta)` exactly, and with the axis
    # unclosed and the period whole the wire starts and ends on the axis, so the
    # chord and the axis are the same line whatever rigid frame the write is in.
    "points_um": [[0.0, 0.0], [2.0, 0.0]],
    "lambda_um": 0.1, "y_amp_um": 0.05,
    "v0": 10.0, "v_k": 5.0, "phase_deg": 0.0,
    # Unphysical, and only so the twin's simulated clock finishes in well under a
    # second. Nothing being asserted depends on it: the compiled point count is arc
    # length over the interpolation spacing, never over time.
    "speed_um_s": 400.0,
}

ERASE_SPEC = {
    "kind": "region", "name": "erase-pad", "shape": "rect",
    "x_um": 6.0, "y_um": 0.0, "w_um": 0.6, "h_um": 0.4,
    "pitch_um": 0.05, "voltage": -10.0, "speed_um_s": 400.0,
}


def _chiral(**over) -> dict:
    return {**CHIRAL_SPEC, **over}


def _load_objects(cmd, specs) -> dict:
    return cmd.call("loadPattern", {"token": _token(cmd), "objects": specs})


def _deviation(path):
    """Signed perpendicular deviation of `path` from its own chord, and the arc-length
    coordinate along it.

    Chord-relative, not `y`, so the assertions survive a non-zero litho frame: the
    write runs in the imaging scan's rotated frame, and this module shares one process
    hub with every other test file, so the frame is not ours to assume.
    """
    import numpy as np
    p = np.asarray(path, dtype=float)
    chord = p[-1] - p[0]
    t = chord / max(float(np.hypot(*chord)), 1e-12)
    n = np.array([-t[1], t[0]])
    rel = p - p[0]
    return rel @ n, rel @ t


# ---------------------------------------------------------------- the door ----

def test_loadpattern_takes_exactly_one_of_source_and_objects(cmd):
    """Neither is a script that meant to load something; both is a script that is
    about to write whichever one the service happened to prefer."""
    token = _token(cmd)
    neither = cmd.send("loadPattern", {"token": token})["error"]
    assert neither["code"] == fd.INVALID_PARAMS
    assert "exactly one" in neither["message"] and "got neither" in neither["message"]
    both = cmd.send("loadPattern", {"token": token, "source": TWO_OBJECT_SVG,
                                    "objects": [_chiral()]})["error"]
    assert both["code"] == fd.INVALID_PARAMS
    assert "not both" in both["message"]


def test_loadpattern_objects_reports_kind_and_the_built_chiral_profile(cmd, read,
                                                                       hub):
    """The result keeps its shape and gains `kind` per object plus, for a chiral one,
    the profile ACTUALLY BUILT -- the echo a phase sweep reads its own parameter back
    from."""
    # ERASE_SPEC is a -10 V erase, outside the default [-8, 30] window the loadPattern
    # door now enforces (spec 13.1 item 19); widen the sample window for the load, as
    # a real -10 V erase would carry.
    with _model_override(hub, tip_voltage_window_v=(-12.0, 30.0)):
        res = _load_objects(cmd, [_chiral(phase_deg=45.0), ERASE_SPEC])
    assert res["object_count"] == 2
    assert res["total_points"] > 0 and res["duration_s"] > 0
    assert res["segments"] == res["strokes"] >= 2   # the fill is several strokes
    assert res["multi_segment_objects"] == []
    assert res["max_outwave_points"] == 87000
    helix, pad = res["objects"]
    assert helix["name"] == "helix" and helix["kind"] == "chiral"
    assert pad["name"] == "erase-pad" and pad["kind"] == "region"
    assert "profile" not in pad                     # only a chiral one carries it
    prof = helix["profile"]
    assert prof == {"lambda_um": pytest.approx(0.1), "y_amp_um": pytest.approx(0.05),
                    "v0": 10.0, "v_k": 5.0, "phase_deg": 45.0, "hand": 1.0,
                    "curvature_clamped": False,
                    # unclamped: the built amplitude IS the requested one
                    "y_amp_built_um": pytest.approx(0.05),
                    "curvature_clamped_fraction": 0.0}
    # It is LOADED, and the next startWrite acts on it.
    assert [o["name"] for o in read.call("getPattern")["objects"]] == ["helix",
                                                                      "erase-pad"]


def test_a_design_file_still_reports_the_compilers_own_kind(cmd):
    """`kind` is a superset, not a replacement: an object that came from a design FILE
    has no spec kind, so it reports the word `getPattern` has always used for it."""
    res = _load_objects(cmd, [_chiral()])
    assert res["objects"][0]["kind"] == "chiral"
    svg = cmd.call("loadPattern", {"token": _token(cmd), "source": TWO_OBJECT_SVG})
    assert [o["kind"] for o in svg["objects"]] == ["path", "path"]


def test_the_flex_door_builds_through_the_http_doors_own_chiral_builder(cmd, hub):
    """THE ANTI-FORK CASE. If this file ever grows its own sinusoid, the FLEX phase
    and the cockpit phase drift apart and every sweep written by a script is written
    at a phase the panel would not reproduce. So: what landed in the pattern must be
    what `chiral_profile` -- the exact call `POST /api/design/chiral` makes -- returns
    for the same parameters, point for point and volt for volt.
    """
    import numpy as np

    from afm_litho.litho.pens import chiral_profile
    _load_objects(cmd, [_chiral(phase_deg=-45.0, hand=-1)])
    o = hub.pattern.objects[0]
    build = chiral_profile(
        np.asarray(CHIRAL_SPEC["points_um"], dtype=float),
        CHIRAL_SPEC["lambda_um"], CHIRAL_SPEC["y_amp_um"],
        CHIRAL_SPEC["v0"], CHIRAL_SPEC["v_k"], -45.0, -1.0)
    assert np.allclose(np.asarray(o.points_um), build.points)
    assert np.allclose(np.asarray(o.v_profile), build.volts)
    # ...and the object carries the analytic profile + its untouched base axis, so it
    # is still re-parametrizable exactly as one made through the panel is.
    assert (o.profile or {}).get("kind") == "chiral"
    assert o.axis_um == [(0.0, 0.0), (2.0, 0.0)]
    # hand is FOLDED INTO the stored phase, which is the convention `_reapply_chiral`
    # reads back: phi = radians(phase_deg) * hand.
    assert o.profile["phase_rad"] == pytest.approx(math.radians(-45.0) * -1.0)


# ------------------------------------------------- the phase, as WRITTEN ------

@pytest.mark.parametrize("phase_deg", [-90.0, 0.0, 90.0])
def test_v_written_carries_the_phase_the_script_asked_for(cmd, read, hub, phase_deg):
    """THE CAPSTONE ASSERTION: what the twin actually WRITES is
    `v0 + v_k sin(theta + phi)`, with phi the phase the script ordered.

    Asserted against the GEOMETRY of the same trace, not against a profile this test
    recomputed: on a straight axis the signed deviation from the chord is
    `y_amp_um * sin(theta)`, so

        u  = deviation / y_amp_um        = sin(theta)
        vn = (v_written - v0) / v_k      = sin(theta + phi)

    and the phase is then a pure identity between the two columns:

        phi = 0    ->  vn == u                       (bias in phase with the wiggle)
        phi = +-90 ->  vn = +-cos(theta), so vn^2 + u^2 == 1, and the SIGN of vn
                       follows +-d(deviation)/ds -- which is what separates the two
                       enantiomers, and is exactly what a builder that dropped the
                       phase, or folded `hand` in the wrong direction, would get wrong.

    A builder that ignored `phase_deg` entirely would pass none of the three.
    """
    import numpy as np

    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "objects": [_chiral(phase_deg=phase_deg)]})
    res = cmd.call("startWrite", {"token": token, "objects": "all", **_gw()})
    rec = _wait_run(hub, res["run_id"])
    assert rec.completed and not rec.aborted, rec.error
    trace = read.call("getWriteResult", {"run_id": res["run_id"],
                                         "inline": True})["trace"]
    assert len(trace) == 1                       # one stroke, however long
    entry = trace[0]
    assert len(entry["path_um"]) == len(entry["v_tip"])

    dev, s = _deviation(entry["path_um"])
    u = dev / CHIRAL_SPEC["y_amp_um"]
    vn = (np.asarray(entry["v_tip"]) - CHIRAL_SPEC["v0"]) / CHIRAL_SPEC["v_k"]
    # The wire really is modulated -- otherwise every identity below is 0 == 0.
    assert np.abs(u).max() == pytest.approx(1.0, abs=0.1)
    assert np.abs(vn).max() == pytest.approx(1.0, abs=0.1)

    if phase_deg == 0.0:
        assert np.abs(vn - u).max() < 0.15
    else:
        # in quadrature: the (u, vn) locus is the unit circle, not a line
        assert np.abs(vn ** 2 + u ** 2 - 1.0).max() < 0.15
        assert np.abs(vn - u).max() > 1.0        # and emphatically NOT in phase
        # ...and the two enantiomers differ by the SIGN of vn against d(dev)/ds.
        slope = np.gradient(u, s)
        steep = np.abs(slope) > 0.5 * np.abs(slope).max()
        agree = (np.sign(vn[steep]) == np.sign(slope[steep])).mean()
        assert agree > 0.95 if phase_deg > 0 else agree < 0.05


def test_v_written_is_the_built_profile_point_for_point(cmd, read, hub):
    """The looser half of the same claim, made exact: `v_written` is the compiled bias
    the builder produced, sample for sample, not merely something with the right
    shape. The compiler spreads `v_profile` uniformly over the object's arc length
    (over its path points: the reported count less the 0 V end sample) and the worker
    takes every 4th path point and the last, so the expected samples are
    reconstructible here."""
    import numpy as np

    token = _token(cmd)
    res = cmd.call("loadPattern", {"token": token,
                                   "objects": [_chiral(phase_deg=-90.0)]})
    n = res["objects"][0]["points"]
    prof = np.asarray(hub.pattern.objects[0].v_profile, dtype=float)
    started = cmd.call("startWrite", {"token": token, "objects": "all", **_gw()})
    rec = _wait_run(hub, started["run_id"])
    assert rec.completed and not rec.aborted, rec.error
    entry = read.call("getWriteResult", {"run_id": started["run_id"],
                                         "inline": True})["trace"][0]
    from afm_litho.litho.compiler import trace_indices
    expected = np.interp(np.linspace(0.0, 1.0, n - 1),
                         np.linspace(0.0, 1.0, len(prof)), prof)
    expected = expected[trace_indices(n - 1)][::entry["decimate"]]
    got = np.asarray(entry["v_tip"], dtype=float)
    assert len(got) == len(expected)
    assert np.abs(got - expected).max() < 2e-3        # the worker's round(v, 3)


# --------------------------------------------------------------- the erase ----

def test_an_erase_rectangle_is_written_at_the_negative_voltage(cmd, read, hub):
    """A region's bias is passed STRAIGHT THROUGH, signed. The `erase` PEN coerces
    `-abs(voltage)`, which would make a positive-bias region unorderable and a sign
    typo invisible; this door does not, so an erase is simply a region at -10 V --
    and every written point of it has to be at -10 V, not just its nominal
    `voltage` field."""
    import dataclasses

    import numpy as np

    from afm_litho.service.instrument_model import Field, FieldSource

    token = _token(cmd)
    # The erase writes at -10 V, below the default sample window [-8, 30], so BOTH
    # the loadPattern door (spec 13.1 item 19) and the write guard
    # (INSTRUMENT_MODEL_SPEC secs 2/10, 13) refuse it as `tip_voltage_out_of_window`.
    # This case is about SIGNED passthrough, not the window, so widen only the window
    # on the live model for the duration -- a real -10 V erase would carry a sample
    # window that admits it. Widened BEFORE the load, since the design door now
    # enforces the window too. Guard scope unchanged.
    saved_model = hub.host.instrument_model
    hub.host.instrument_model = dataclasses.replace(
        saved_model,
        tip_voltage_window_v=Field(value=(-12.0, 30.0),
                                   source=FieldSource.override))
    try:
        cmd.call("loadPattern", {"token": token, "objects": [ERASE_SPEC]})
        assert hub.pattern.objects[0].voltage == -10.0
        res = cmd.call("startWrite", {"token": token, "objects": "all", **_gw()})
        rec = _wait_run(hub, res["run_id"])
        assert rec.completed and not rec.aborted, rec.error
        trace = read.call("getWriteResult", {"run_id": res["run_id"],
                                             "inline": True})["trace"]
    finally:
        hub.host.instrument_model = saved_model
    assert trace, "the fill wrote nothing"
    for obj in trace:
        assert obj["voltage"] == -10.0
        assert np.allclose(np.asarray(obj["v_tip"], dtype=float), -10.0)
    # ...and it landed where it was asked to: a 0.6 x 0.4 um pad at (6, 0).
    xs = [p[0] for obj in trace for p in obj["path_um"]]
    ys = [p[1] for obj in trace for p in obj["path_um"]]
    assert max(xs) - min(xs) == pytest.approx(0.6, abs=0.1)
    assert max(ys) - min(ys) == pytest.approx(0.4, abs=0.1)


def test_a_region_takes_any_fill_the_pen_accepts_and_refuses_the_rest(cmd, hub):
    from afm_litho.litho.pens import PENS
    # ERASE_SPEC is a -10 V erase, outside the default [-8, 30] window the loadPattern
    # door now enforces (spec 13.1 item 19); widen the sample window for these loads.
    with _model_override(hub, tip_voltage_window_v=(-12.0, 30.0)):
        for fill, _label in PENS["region"].choices["fill"]:
            _load_objects(cmd, [{**ERASE_SPEC, "fill": fill}])
            assert hub.pattern.objects[0].fill_kind == fill
        reply = cmd.send("loadPattern", {"token": _token(cmd),
                                         "objects": [{**ERASE_SPEC, "fill": "spiral"}]})
    assert reply["error"]["code"] == fd.INVALID_PARAMS
    assert "objects[0].fill" in reply["error"]["message"]


# ------------------------------------------------------------- the refusals ---

@pytest.mark.parametrize("spec,field", [
    ({"kind": "helical"}, "kind"),
    ({"kind": "chiral"}, "points_um"),
    (_chiral(points_um=[[0.0, 0.0]]), "points_um"),
    (_chiral(points_um=[[0.0, 0.0], [1.0, "x"]]), "points_um"),
    (_chiral(points_um=[[0.0, 0.0], [0.0, 0.0]]), "points_um"),
    (_chiral(lambda_um=0.0), "lambda_um"),
    (_chiral(lambda_um=-1.0), "lambda_um"),
    (_chiral(lambda_um="fine"), "lambda_um"),
    (_chiral(y_amp_um=-0.01), "y_amp_um"),
    (_chiral(y_amp_um=float("nan")), "y_amp_um"),
    (_chiral(v0=float("inf")), "v0"),
    (_chiral(v_k=None), "v_k"),
    (_chiral(phase_deg=None), "phase_deg"),
    (_chiral(hand=0), "hand"),
    (_chiral(hand=2), "hand"),
    (_chiral(speed_um_s=0.0), "speed_um_s"),
    (_chiral(phase=90.0), "phase"),                 # the typo that must not be ignored
    ({**ERASE_SPEC, "shape": "circle"}, "shape"),
    ({**ERASE_SPEC, "w_um": 0.0}, "w_um"),
    ({**ERASE_SPEC, "h_um": -1.0}, "h_um"),
    ({**ERASE_SPEC, "pitch_um": 0.0}, "pitch_um"),
    ({**ERASE_SPEC, "voltage": float("nan")}, "voltage"),
    ({**ERASE_SPEC, "x_um": None}, "x_um"),
    ({**ERASE_SPEC, "amp_um": 1.0}, "amp_um"),      # unknown key on a region
])
def test_a_bad_object_spec_is_refused_by_name(cmd, spec, field):
    """Every refusal NAMES THE FIELD, in the message and in `data`: a script sending a
    phase sweep of a dozen specs cannot otherwise tell which one it got wrong."""
    reply = cmd.send("loadPattern", {"token": _token(cmd), "objects": [spec]})
    err = reply["error"]
    assert err["code"] == fd.INVALID_PARAMS, reply
    assert f"objects[0].{field}" in err["message"], err["message"]
    assert err["data"]["param"] == "objects"
    assert err["data"]["index"] == 0 and err["data"]["field"] == field


@pytest.mark.parametrize("objects", [[], "chiral", {"kind": "chiral"}, [None], [7]])
def test_objects_must_be_a_non_empty_list_of_specs(cmd, objects):
    reply = cmd.send("loadPattern", {"token": _token(cmd), "objects": objects})
    assert reply["error"]["code"] == fd.INVALID_PARAMS, reply
    assert reply["error"]["data"]["param"] == "objects"


def test_two_objects_may_not_share_a_name(cmd, hub):
    """The compile report is keyed BY NAME, so a duplicate would be reported -- and
    its points summed -- as a single object."""
    reply = cmd.send("loadPattern", {"token": _token(cmd),
                                     "objects": [_chiral(), _chiral()]})
    assert reply["error"]["code"] == fd.INVALID_PARAMS
    assert "duplicate name" in reply["error"]["message"]
    # ...and unnamed specs get distinct names automatically, so the common case works.
    # ERASE_SPEC is a -10 V erase, outside the default [-8, 30] window the loadPattern
    # door now enforces (spec 13.1 item 19); widen the sample window for the load.
    with _model_override(hub, tip_voltage_window_v=(-12.0, 30.0)):
        res = _load_objects(cmd, [{k: v for k, v in _chiral().items() if k != "name"},
                                  {k: v for k, v in ERASE_SPEC.items() if k != "name"}])
    assert [o["name"] for o in res["objects"]] == ["chiral-1", "region-2"]


def test_a_refused_object_spec_leaves_the_previous_pattern_in_place(cmd, read, hub):
    """The existing rule, on the new door: a half-replaced pattern is something the
    next `startWrite` would go and write."""
    # ERASE_SPEC is a -10 V erase, outside the default [-8, 30] window the loadPattern
    # door now enforces (spec 13.1 item 19); widen the sample window for the setup load.
    with _model_override(hub, tip_voltage_window_v=(-12.0, 30.0)):
        _load_objects(cmd, [_chiral(), ERASE_SPEC])
    before = read.call("getPattern")
    cmd.send("loadPattern", {"token": _token(cmd),
                             "objects": [_chiral(), _chiral(name="bad",
                                                            lambda_um=-1.0)]})
    assert read.call("getPattern") == before


def test_a_runaway_chiral_is_refused_rather_than_allocated(cmd, read):
    """`chiral_profile` refuses an amplitude-to-pitch ratio whose sample count would
    be a multi-gigabyte allocation. That refusal has to arrive as a `-32602` naming
    the offending parameter, not as a `-32000` from a killed process."""
    before = read.call("getPattern")
    reply = cmd.send("loadPattern", {"token": _token(cmd),
                                     "objects": [_chiral(lambda_um=1e-4,
                                                         y_amp_um=50.0)]})
    assert reply["error"]["code"] == fd.INVALID_PARAMS
    assert "objects[0].y_amp_um" in reply["error"]["message"]
    assert "relative to the pitch" in reply["error"]["message"]
    assert read.call("getPattern") == before


# ---------------------------------------------------------------- the cap -----

def test_an_over_cap_chiral_chains_rather_than_being_refused(cmd, read, hub,
                                                             monkeypatch):
    """The rule, on the new door: CHUNK, DON'T REFUSE. A chiral wire is ONE stroke
    however long it is, so an over-cap one is armed as a seamless chain of chunks --
    tip down and bias live across every boundary -- and is reported, not rejected.

    A 250 um axis at a 0.1 um pitch is ~90k compiled points: over the 86,976-point
    chunk size, and nowhere near the 2000 um design guard.
    """
    logs = _spy_on_logs(hub, monkeypatch)
    res = _load_objects(cmd, [_chiral(points_um=[[0.0, 0.0], [250.0, 0.0]])])
    entry = res["objects"][0]
    assert entry["points"] > CHAIN_CHUNK_POINTS
    assert entry["kind"] == "chiral" and entry["profile"]["phase_deg"] == 0.0
    assert entry["segments"] == -(-entry["points"] // CHAIN_CHUNK_POINTS) >= 2
    assert entry["strokes"] == 1 == res["strokes"]     # ONE continuous tip path
    multi = res["multi_segment_objects"]
    assert len(multi) == 1 and multi[0]["name"] == "helix"
    assert multi[0]["chain_boundaries"] == res["segments"] - 1
    # ...and the operator is warned by name, exactly as for an over-cap design file.
    warn = [ln for ln in logs if "CHAIN of chunks" in ln]
    assert len(warn) == 1 and "helix" in warn[0] and "DOSE BLOB" in warn[0]
    # It is LOADED, not refused: the next startWrite would write it.
    assert read.call("getPattern")["objects"][0]["segments"] == res["segments"]


# ==================== 9b. the SIZE POLICY on the `objects` door =================
#
# `source` has always been bounded -- `_svg_pattern` refuses a design whose extent is
# over MAX_DESIGN_UM before it flattens anything. `objects` bypassed that entirely,
# and the adversarial review of 2026-09-12 measured what that cost:
#
#   * a 400 x 400 um rect at a 0.5 um pitch was ACCEPTED -- 16 M points, ~185 chain
#     chunks, an 89-hour write -- because nothing on this door looked at the size;
#   * a 3000 x 3000 um rect took the COMPILE past ten minutes on a verb published at
#     2 s, so the caller timed out while the service kept burning a thread;
#   * `pitch_um: 10` on a 0.4 um rect compiled to an outline plus one centre line: an
#     erase that does not erase, and it looked like it had worked.
#
# All three are a units slip (nm typed as um is exactly 1000x), which is why the
# refusals below say so in the message. The rule these tests hold to is that a refusal
# is ARITHMETIC and arrives in MILLISECONDS: the guard is worthless if finding out
# costs the same compile it exists to prevent.

DOC_EXAMPLE_OBJECTS = [
    # Verbatim from docs/FLEX_PROVIDER.md, the `objects` section's worked example.
    {"kind": "chiral", "name": "helix-R", "points_um": [[0, 0], [2, 0]],
     "lambda_um": 0.025, "y_amp_um": 0.055,
     "v0": 10.0, "v_k": 5.0, "phase_deg": 90.0, "hand": 1, "speed_um_s": 1.5},
    {"kind": "region", "name": "erase-pad", "shape": "rect",
     "x_um": 6.0, "y_um": 0.0, "w_um": 0.6, "h_um": 0.4,
     "pitch_um": 0.01, "voltage": -10.0, "fill": "directional", "speed_um_s": 1.0},
]


def test_the_documented_example_still_compiles_to_the_documented_numbers(cmd, hub):
    """docs/FLEX_PROVIDER.md prints a worked `objects` request and the exact result it
    returns. A doc that has drifted from the service is worse than no doc -- a script
    is written against the printed numbers -- so the example is asserted, not admired.
    """
    # The documented erase-pad is at -10 V, outside the default [-8, 30] window the
    # loadPattern door now enforces (spec 13.1 item 19); widen the sample window for
    # the load so the documented compile numbers can still be asserted.
    with _model_override(hub, tip_voltage_window_v=(-12.0, 30.0)):
        res = _load_objects(cmd, DOC_EXAMPLE_OBJECTS)
    # FIRED counts: every stroke is a multiple of 32 points, its last one at 0 V.
    # The helix is 2 um at 25 nm wavelength and 55 nm amplitude; the chiral spacing keeps
    # every sine peak within 0.5 nm (pens.chiral_spacing_um, 1.07 nm here), which is 16608
    # points. The pad's two strokes are 1344. Total 16608 + 1344.
    assert res["total_points"] == 17952
    helix, pad = res["objects"]
    assert helix["points"] == 16608 and helix["segments"] == 1 and helix["strokes"] == 1
    assert pad["points"] == 1344 and pad["segments"] == 2 and pad["strokes"] == 2
    assert res["segments"] == 3 and res["strokes"] == 3
    assert res["multi_segment_objects"] == []


def test_an_absurd_region_is_refused_by_ARITHMETIC_not_by_compiling_it():
    """THE 400 x 400 CASE. 16 million points and 89 hours, accepted before this.

    Timed against the BUILDER rather than the wire, because the wire's own latency is
    not what is being measured: the claim is that the refusal is extent/pitch
    arithmetic and never touches the fill. 100 ms is three orders of magnitude over
    the ~0.05 ms it actually takes, and four orders under the multi-minute compile.
    """
    spec = {**DOC_EXAMPLE_OBJECTS[1], "w_um": 400.0, "h_um": 400.0, "pitch_um": 0.5}
    t0 = time.perf_counter()
    with pytest.raises(CommandError) as caught:
        litho_cmds._objects_pattern([spec])
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.1, f"refusal took {elapsed * 1000:.0f} ms -- it compiled it"
    err = caught.value
    assert err.code == fd.INVALID_PARAMS
    assert "objects[0].pitch_um" in err.message
    assert "16,080,001 points" in err.message      # the estimate, said out loud
    assert f"{litho_cmds.MAX_OBJECT_POINTS:,}" in err.message
    assert err.data["field"] == "pitch_um"


@pytest.mark.parametrize("spec,field", [
    # A rectangle wider than any scanner, on the door that had no extent check at all.
    ({"kind": "region", "shape": "rect", "x_um": 0.0, "y_um": 0.0,
      "w_um": 3000.0, "h_um": 10.0, "pitch_um": 0.5, "voltage": -10.0}, "w_um"),
    ({"kind": "region", "shape": "rect", "x_um": 0.0, "y_um": 0.0,
      "w_um": 10.0, "h_um": 3000.0, "pitch_um": 0.5, "voltage": -10.0}, "h_um"),
    # ...and the same bound on a chiral wire's axis, which is the same units slip.
    ({"kind": "chiral", "points_um": [[0, 0], [5000, 0]], "lambda_um": 0.025,
      "y_amp_um": 0.055, "v0": 10.0, "v_k": 5.0, "phase_deg": 0.0}, "points_um"),
])
def test_an_object_larger_than_any_scanner_is_refused_on_both_kinds(cmd, read,
                                                                    spec, field):
    """MAX_DESIGN_UM is 2000 um -- twenty times the largest scanner here -- and it
    bounded a design FILE but not an object spec. Both kinds carry it now, and both
    say the same thing, because it is always the same mistake: nanometres typed into
    a field that is microns."""
    before = read.call("getPattern")
    reply = cmd.send("loadPattern", {"token": _token(cmd), "objects": [spec]})
    err = reply["error"]
    assert err["code"] == fd.INVALID_PARAMS, reply
    assert f"objects[0].{field}" in err["message"]
    assert "MICRONS" in err["message"] and "1000x" in err["message"]
    assert read.call("getPattern") == before       # nothing half-replaced


def test_a_pitch_wider_than_the_rectangle_is_refused(cmd, hub):
    """AN ERASE THAT DOES NOT ERASE. `pitch_um` is the separation between fill lines,
    so a pitch wider than the shape leaves the outline and at most one line down the
    middle -- measured: a 0.4 um rect at `pitch_um: 10` compiled to exactly that, and
    reported success. Half the short side is the loosest rule that still guarantees
    more than one interior line."""
    spec = {**DOC_EXAMPLE_OBJECTS[1], "w_um": 0.4, "h_um": 0.4, "pitch_um": 10.0}
    err = cmd.send("loadPattern", {"token": _token(cmd), "objects": [spec]})["error"]
    assert err["code"] == fd.INVALID_PARAMS
    assert "objects[0].pitch_um" in err["message"]
    assert "does not erase" in err["message"]
    assert err["data"]["field"] == "pitch_um"
    # ...and exactly at the boundary it is allowed, so the rule is a limit and not a
    # margin someone has to guess at. The spec is a -10 V erase, outside the default
    # [-8, 30] window the loadPattern door now enforces (spec 13.1 item 19); widen
    # the sample window for the boundary-ok load.
    with _model_override(hub, tip_voltage_window_v=(-12.0, 30.0)):
        ok = _load_objects(cmd, [{**spec, "pitch_um": 0.2}])
    assert ok["object_count"] == 1


def test_the_pattern_as_a_whole_has_a_point_budget(cmd, read):
    """Every object inside MAX_OBJECT_POINTS and the PATTERN still unwritable: enough
    legal regions is enough legal regions' worth of compile, on a verb bounded at 2 s.
    Refused at the object that crosses the line, by index."""
    before = read.call("getPattern")
    # ~1.008 M points each: comfortably inside MAX_OBJECT_POINTS, four of them over
    # MAX_TOTAL_POINTS. So nothing here is individually refusable and the pattern is
    # still too big to be a pattern.
    big = {**DOC_EXAMPLE_OBJECTS[1], "w_um": 20.0, "h_um": 20.0, "pitch_um": 0.02}
    assert litho_cmds._region_points(20.0, 20.0, 0.02) < litho_cmds.MAX_OBJECT_POINTS
    specs = [{**big, "name": f"pad-{i}"} for i in range(4)]
    reply = cmd.send("loadPattern", {"token": _token(cmd), "objects": specs})
    err = reply["error"]
    assert err["code"] == fd.INVALID_PARAMS, reply
    assert f"{litho_cmds.MAX_TOTAL_POINTS:,}" in err["message"]
    assert err["data"]["index"] >= 1                 # not the first one
    assert read.call("getPattern") == before


def test_angle_deg_is_accepted_for_a_directional_fill_and_refused_for_the_others(cmd,
                                                                                 hub):
    """The region pen sweeps `directional` at a chosen angle and every other strategy
    ignores it (a Hilbert curve HAS no preferred direction -- that is the reason to
    pick one). Accepted-and-ignored is the failure mode `_spec_keys` exists to
    prevent, so it is a refusal by name."""
    base = {**DOC_EXAMPLE_OBJECTS[1], "pitch_um": 0.05}
    # `base` is a -10 V erase, outside the default [-8, 30] window the loadPattern
    # door now enforces (spec 13.1 item 19); widen the sample window for the accepted
    # directional load (the hilbert case below refuses on angle_deg before the guard).
    with _model_override(hub, tip_voltage_window_v=(-12.0, 30.0)):
        res = _load_objects(cmd, [{**base, "fill": "directional", "angle_deg": 30.0}])
    assert res["objects"][0]["kind"] == "region"
    err = cmd.send("loadPattern", {
        "token": _token(cmd),
        "objects": [{**base, "fill": "hilbert", "angle_deg": 30.0}]})["error"]
    assert err["code"] == fd.INVALID_PARAMS
    assert "objects[0].angle_deg" in err["message"]
    assert err["data"]["field"] == "angle_deg"


def test_a_degenerate_axis_says_what_is_wrong_in_plain_words(cmd):
    """The spline says "`x` must contain at least 2 elements". That is not what the
    caller did wrong, and it is not a sentence anyone can act on. Name the mistake."""
    spec = {**DOC_EXAMPLE_OBJECTS[0], "points_um": [[1.0, 1.0]] * 3}
    err = cmd.send("loadPattern", {"token": _token(cmd), "objects": [spec]})["error"]
    assert "objects[0].points_um" in err["message"]
    assert "same place" in err["message"]
    assert "two DIFFERENT endpoints" in err["message"]


def test_a_long_chiral_blames_the_length_not_the_amplitude(cmd):
    """The runaway refusal has TWO causes and used to report only one. A 1500 um axis
    at the pen's own default 55 nm / 25 nm was refused as "amplitude very large
    relative to the pitch (ratio 2:1)" -- the recommended regime, and nothing the
    caller could act on. n = L/lambda * per_period: name the factor responsible."""
    spec = {**DOC_EXAMPLE_OBJECTS[0], "points_um": [[0, 0], [1500, 0]]}
    err = cmd.send("loadPattern", {"token": _token(cmd), "objects": [spec]})["error"]
    assert err["code"] == fd.INVALID_PARAMS
    assert "objects[0].points_um" in err["message"]
    assert "it is the LENGTH that does not fit" in err["message"]
    assert "relative to the pitch" not in err["message"]


# ------------------------------------------ what was BUILT, not what was asked ---

# A tight ring: at CURVATURE_HEADROOM = 0.6 a 60 nm radius admits at most 36 nm of
# amplitude, so the pen's 55 nm is clamped over the whole loop.
TIGHT_RING = [[0.06 * math.cos(i * 2 * math.pi / 64),
               0.06 * math.sin(i * 2 * math.pi / 64)] for i in range(65)]


def test_a_clamped_chiral_echoes_the_AMPLITUDE_it_built_not_just_a_boolean(cmd):
    """`curvature_clamped: true` spans everything from one tight corner to a wire
    written at two thirds of the amplitude that was ordered -- and in a chirality
    experiment the amplitude IS the measurement, so the boolean alone is not a report.
    The magnitude has to be on the wire."""
    spec = {"kind": "chiral", "name": "ring", "points_um": TIGHT_RING,
            "lambda_um": 0.025, "y_amp_um": 0.055,
            "v0": 10.0, "v_k": 5.0, "phase_deg": 90.0, "speed_um_s": 400.0}
    prof = _load_objects(cmd, [spec])["objects"][0]["profile"]
    assert prof["curvature_clamped"] is True
    assert prof["curvature_clamped_fraction"] > 0.5
    assert prof["y_amp_um"] == pytest.approx(0.055)          # what was ASKED for
    assert prof["y_amp_built_um"] < 0.055                    # what was WRITTEN
    assert prof["y_amp_built_um"] == pytest.approx(0.6 * 0.06, rel=0.15)
    # ...and a wire that was not clamped reports the two as equal, so a script can
    # compare them without special-casing the happy path.
    clean = _load_objects(cmd, [_chiral()])["objects"][0]["profile"]
    assert clean["curvature_clamped"] is False
    assert clean["curvature_clamped_fraction"] == 0.0
    assert clean["y_amp_built_um"] == pytest.approx(clean["y_amp_um"])


def test_a_floored_speed_is_echoed_back(cmd):
    """The tick interpolation factor is an integer, so the speed field is a CEILING
    the machine quantises down from (`compiler.time_segment`). On a chiral wire, whose
    interpolation spacing is a fraction of the pitch, it can land well under the
    request -- and a dose sweep is a sweep over speed, so the real number has to come
    back. It is emitted only when it is news."""
    entry = _load_objects(cmd, [_chiral(speed_um_s=400.0)])["objects"][0]
    got = entry.get("speed_achieved_um_s")
    assert got is not None, "a 400 um/s request cannot survive the tick quantisation"
    assert 0 < got < 400.0


# --------------------------------------------- the profile survives an edit -----

def test_a_chiral_loaded_by_parameters_rebuilds_from_its_own_profile(cmd, hub):
    """THE REASON THE PROFILE MATTERS. It is not a receipt -- `routers/design`
    `_reapply_chiral` REBUILDS the geometry and the bias from it after the node tool
    moves the base axis. So the profile a `loadPattern` object carries has to be the
    profile of THAT wire: re-applying it with the axis untouched must reproduce the
    wire point for point. (Before `ChiralBuild` the profile came off a function
    attribute, and a concurrent build could hand this object another wire's.)"""
    import numpy as np

    from afm_litho.service.routers.design import _reapply_chiral

    _load_objects(cmd, [_chiral(phase_deg=45.0, hand=-1)])
    o = hub.pattern.objects[0]
    before_pts = np.asarray(o.points_um, dtype=float)
    before_v = np.asarray(o.v_profile, dtype=float)
    assert _reapply_chiral(o) is True
    assert np.allclose(np.asarray(o.points_um, dtype=float), before_pts)
    assert np.allclose(np.asarray(o.v_profile, dtype=float), before_v)


def test_a_build_that_runs_past_the_verbs_bound_is_refused_not_waited_on(cmd, read,
                                                                         monkeypatch):
    """THE BACKSTOP. The arithmetic guards above refuse the absurd request in
    microseconds, but arithmetic only catches what it was taught to look for; a
    3000 x 3000 um rect once held the compile for TEN MINUTES on a verb published at
    2 s, with the caller long gone. So build+compile run under a wall clock set to
    the verb's OWN published bound -- the number `method_bounds` advertises and the
    FLEX driver derives its client timeout from -- and over it the answer is
    `-32021 refused`, not silence.

    Not `-32602`: nothing about the request was found to be invalid, and retrying it
    unchanged will not help either, which is what separates this from `busy`.
    """
    before = read.call("getPattern")
    real = litho_cmds._objects_pattern

    def slow(specs, deadline=None):
        time.sleep(0.5)
        return real(specs, deadline)

    monkeypatch.setattr(litho_cmds, "_objects_bound_s", lambda: 0.05)
    monkeypatch.setattr(litho_cmds, "_objects_pattern", slow)
    reply = cmd.send("loadPattern", {"token": _token(cmd), "objects": [_chiral()]})
    err = reply["error"]
    assert err["code"] == fd.REFUSED, reply
    assert err["data"]["reason"] == "build_timeout"
    assert "unchanged" in err["message"]
    # The pattern that was loaded is STILL loaded: a half-replaced pattern is what the
    # next startWrite would go and write.
    assert read.call("getPattern") == before
    # The orphan (`slow` is still sleeping in the background) must not leak into
    # whatever test runs next -- see `_wait_orphan_clear`.
    _wait_orphan_clear()


def test_a_timed_out_orphan_is_tracked_isolated_and_discarded(cmd, read, monkeypatch):
    """THE OTHER HALF OF THE BACKSTOP (adversarial review of 7bd011d, measured
    2026-09-12). `_within`'s wall clock answers the CALLER promptly, but the worker
    thread it abandoned cannot be killed and keeps running -- and it used to run on
    the process-wide DEFAULT `ThreadPoolExecutor`, the same pool `withdraw` /
    `approach` / the free-air and SUM reads use. A script issuing repeated slow
    `loadPattern {objects}` calls could pile up one orphan per call until the
    pool's fixed slots were exhausted and a hardware-safety read queued behind
    them (measured: 2.8 s of queueing in a 3 s-timeout / 20-slot reproduction).

    Three things this proves, together:

      1. a SECOND `objects` load arriving while the first orphan is still running
         is refused immediately (`-32020 busy`, `build_in_progress`) rather than
         silently queued behind it;
      2. a plain `asyncio.to_thread` call -- standing in for a hardware-safety
         read -- is serviced promptly THE WHOLE TIME, because it never shares an
         executor with the orphan;
      3. once the orphan actually finishes, its result is discarded (never
         installed as `hub.pattern`) and the door reopens for an ordinary load.
    """
    before = read.call("getPattern")
    real = litho_cmds._objects_pattern

    def slow(specs, deadline=None):
        time.sleep(1.0)
        return real(specs, deadline)

    monkeypatch.setattr(litho_cmds, "_objects_bound_s", lambda: 0.05)
    monkeypatch.setattr(litho_cmds, "_objects_pattern", slow)

    reply = cmd.send("loadPattern", {"token": _token(cmd), "objects": [_chiral()]})
    err = reply["error"]
    assert err["code"] == fd.REFUSED, reply
    assert err["data"]["reason"] == "build_timeout"

    # (1) THE ORPHAN IS STILL RUNNING (it sleeps 1.0 s; the verb gave up at 0.05 s),
    # so a second `objects` load must be refused AT ONCE, not queued behind it.
    t0 = time.monotonic()
    reply2 = cmd.send("loadPattern", {"token": _token(cmd), "objects": [_chiral()]})
    elapsed2 = time.monotonic() - t0
    assert elapsed2 < 0.5, f"the second call queued behind the orphan ({elapsed2:.2f}s)"
    err2 = reply2["error"]
    assert err2["code"] == fd.BUSY, reply2
    assert err2["data"]["reason"] == "build_in_progress"
    assert err2["data"]["age_s"] >= 0.0

    # (2) THE DEFAULT POOL IS UNAFFECTED -- a hardware-safety `to_thread` read (here,
    # a stand-in no-op) must complete promptly while the orphan is still running,
    # because it is not competing for the objects door's single dedicated worker.
    async def _default_pool_read():
        t1 = time.monotonic()
        await asyncio.to_thread(lambda: None)
        return time.monotonic() - t1

    cost = asyncio.run(_default_pool_read())
    assert cost < 0.05, f"a default-pool to_thread call took {cost * 1000:.0f} ms"

    # (3) WAIT FOR THE ORPHAN TO ACTUALLY FINISH, then check what it left behind.
    _wait_orphan_clear()
    assert read.call("getPattern") == before, \
        "the orphan's late result was installed as the loaded pattern"

    # THE DOOR REOPENED: restore the real build path and an ordinary load succeeds.
    monkeypatch.undo()
    res = _load_objects(cmd, [_chiral()])
    assert res["total_points"] > 0


def test_objects_executor_is_a_daemon_thread():
    """THE MUST-FIX ITSELF, checked directly (re-review of 450bebb, 2026-09-12): the
    objects door's dedicated worker must be a daemon thread, not a
    `concurrent.futures.ThreadPoolExecutor`'s -- see
    `test_shutdown_does_not_delay_interpreter_exit` for why that distinction is not
    cosmetic (a non-daemon worker is joined unconditionally by Python's own
    `atexit`, which is exactly what let an orphan delay process exit before this
    fix)."""
    fut = litho_cmds._submit_guarded(time.sleep, 0.01)
    executor = litho_cmds._OBJECTS_EXECUTOR
    assert isinstance(executor, litho_cmds._DaemonSingleWorker)
    assert executor._thread.daemon, "the objects-door worker must be a daemon thread"
    fut.result(timeout=2.0)
    _wait_orphan_clear()


def test_shutdown_does_not_delay_interpreter_exit():
    """THE MUST-FIX, empirically (re-review of 450bebb, measured 2026-09-12). The
    executor this branch shipped for the `objects` door was a
    `concurrent.futures.ThreadPoolExecutor(max_workers=1)`. That was still a bug:
    `ThreadPoolExecutor` workers are NOT daemon threads -- CPython registers every
    one in `concurrent.futures.thread._threads_queues`, and the interpreter's own
    `atexit` hook joins them UNCONDITIONALLY, no matter what
    `shutdown(wait=False, cancel_futures=True)` asked for. So an orphaned build
    (already unkillable -- Python cannot interrupt a running thread; see
    `_submit_guarded`) also held up PROCESS EXIT, which nothing about this door
    should ever do: measured, a 3 s orphaned task still made the process take 3.1 s
    to exit despite `wait=False`.

    `litho_cmds._DaemonSingleWorker` replaces it with a worker built on
    `threading.Thread(daemon=True)`, so this must now be false: exercised in a
    SEPARATE PROCESS on purpose, since interpreter teardown is not something the
    test process itself can undergo without exiting.
    """
    script = (
        "import time\n"
        "from afm_litho.service.host import litho_cmds\n"
        "litho_cmds._submit_guarded(time.sleep, 3.0)\n"
        "litho_cmds.shutdown_objects_executor()\n"
    )
    t0 = time.monotonic()
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0, (
        f"process exit took {elapsed:.2f}s with a 3 s orphan still running -- "
        f"the objects-door worker delayed interpreter exit")


def test_submit_after_shutdown_raises_like_stdlib():
    """`_DaemonSingleWorker.submit` after `shutdown()` must raise, matching
    `concurrent.futures.Executor.submit` -- not silently enqueue behind a worker
    thread that has already been told to stop, which would leave the returned
    Future unresolved forever (nothing is left to ever call `set_result` /
    `set_exception` on it once the worker sees the sentinel and exits)."""
    worker = litho_cmds._DaemonSingleWorker(thread_name_prefix="test-shutdown-submit")
    worker.shutdown(wait=True)
    with pytest.raises(RuntimeError, match="shut down"):
        worker.submit(time.sleep, 0.0)


def test_shutdown_cancel_futures_actually_cancels_a_queued_job():
    """`shutdown(cancel_futures=True)` must cancel a job that is QUEUED but has not
    started -- not merely accept the kwarg for parity. With the worker already busy
    on a 1 s job, a second submission sits in the queue behind it; shutting down
    with `cancel_futures=True` must drain that queued job and cancel its Future
    before the worker ever gets to it, rather than letting it run because the
    sentinel was enqueued after it (which is what a no-op `cancel_futures` would
    do)."""
    worker = litho_cmds._DaemonSingleWorker(thread_name_prefix="test-shutdown-cancel")
    started = threading.Event()

    def busy():
        started.set()
        time.sleep(1.0)
        return "busy-done"

    ran = threading.Event()

    def queued():
        ran.set()
        return "queued-done"

    busy_future = worker.submit(busy)
    assert started.wait(timeout=2.0), "the busy job never started"
    queued_future = worker.submit(queued)

    worker.shutdown(wait=False, cancel_futures=True)

    assert queued_future.cancelled(), "the queued-but-not-started job was not cancelled"
    # It must never actually run -- give it a beat past the busy job's own runtime.
    assert not ran.wait(timeout=1.5), "the cancelled job ran anyway"

    # The job that was already running when shutdown() was called is unaffected --
    # it cannot be interrupted and finishes normally.
    assert busy_future.result(timeout=2.0) == "busy-done"


# ================= 10. SVG ids, the shared listing, and setObjects ==============
#
# `loadPattern {objects}` orders geometry BY PARAMETERS; this section is the OTHER
# half of the same need -- editing geometry a script (or a drawing program) already
# loaded. Three things are pinned:
#
#   1. an SVG element's `id` NAMES the object, verbatim, so a script addresses it by
#      the name it drew with -- and the per-object listing carries that id plus the
#      editable `properties` beside it, the SAME shape on loadPattern, getPattern and
#      setObjects;
#   2. `setObjects` MERGES a `{id: {property: value}}` map onto the loaded pattern --
#      a bias retune, a disable, a plain path turned chiral -- through the same
#      `chiral_profile` builder the cockpit uses, so the phase convention cannot fork;
#   3. `startWrite` selects by id as well as by index, in the order given.

# Four <path> elements, drawn 1 unit = 1 nm, each with its own id. `path4` is a
# straight 3 um wire so the enantiomer assertion (a straight axis, a whole number of
# periods) applies to it after it is turned chiral.
SVG_WITH_IDS = (
    '<svg xmlns="http://www.w3.org/2000/svg">'
    '<image id="topo_bgd" x="0" y="0" width="3000" height="3000"/>'
    '<path id="path1" d="M 0 0 L 2000 0"/>'
    '<path id="path2" d="M 0 500 L 2000 500"/>'
    '<path id="path4" d="M 0 1000 L 3000 1000"/>'
    '<path id="path4-1" d="M 0 1500 L 2000 1500"/>'
    "</svg>"
)

SVG_DUP_ID = (
    '<svg xmlns="http://www.w3.org/2000/svg">'
    '<image id="topo_bgd" x="0" y="0" width="2000" height="2000"/>'
    '<path id="dup" d="M 0 0 L 2000 0"/>'
    '<path id="dup" d="M 0 500 L 2000 500"/>'
    "</svg>"
)


def _load_ids(cmd) -> dict:
    return cmd.call("loadPattern", {"token": _token(cmd), "source": SVG_WITH_IDS})


def _set_objects(cmd, updates) -> dict:
    return cmd.call("setObjects", {"token": _token(cmd), "updates": updates})


def test_svg_ids_name_the_objects_and_the_listing_carries_properties(cmd, read):
    """An element's `id` becomes the object's name verbatim, and every listing entry
    gains `id`, `vertices_um` and the editable `properties` block."""
    res = _load_ids(cmd)
    assert [o["name"] for o in res["objects"]] == ["path1", "path2", "path4",
                                                   "path4-1"]
    assert [o["id"] for o in res["objects"]] == ["path1", "path2", "path4",
                                                 "path4-1"]
    assert {o["kind"] for o in res["objects"]} == {"path"}
    for o in res["objects"]:
        props = o["properties"]
        assert set(props) == {"voltage", "speed_um_s", "enabled", "chiral"}
        assert props["enabled"] is True
        assert props["chiral"] is None          # a plain path, not yet modulated
        assert "fill" not in props              # fill is a region's property only
        assert len(o["vertices_um"]) >= 2       # the source polyline
    # getPattern reports the SAME id/properties block.
    pat = read.call("getPattern")
    assert [o["id"] for o in pat["objects"]] == ["path1", "path2", "path4", "path4-1"]
    assert pat["objects"][0]["properties"]["chiral"] is None


def test_two_svg_elements_sharing_an_id_are_refused(cmd, read):
    """Two elements with one id is an SVG error (ids are unique), not a subpath split,
    so it is refused by name and the previous pattern is left in place."""
    _load_ids(cmd)
    before = read.call("getPattern")
    reply = cmd.send("loadPattern", {"token": _token(cmd), "source": SVG_DUP_ID})
    assert reply["error"]["code"] == fd.INVALID_PARAMS
    assert "dup" in reply["error"]["message"]
    assert read.call("getPattern") == before


def test_setobjects_merges_voltage_and_speed(cmd, read):
    """A `{id: {property: value}}` merge changes only the properties it names, and the
    listing reads them back."""
    _load_ids(cmd)
    res = _set_objects(cmd, {"path1": {"voltage": -3.5},
                             "path2": {"speed_um_s": 1.5}})
    by_id = {o["id"]: o for o in res["objects"]}
    assert by_id["path1"]["properties"]["voltage"] == -3.5
    assert by_id["path2"]["properties"]["speed_um_s"] == 1.5
    # untouched objects keep what they had
    assert by_id["path4"]["properties"]["voltage"] == 8.0
    # ...and it is really loaded: getPattern agrees.
    pat = {o["id"]: o for o in read.call("getPattern")["objects"]}
    assert pat["path1"]["voltage"] == -3.5
    assert pat["path2"]["speed"] == 1.5


def test_setobjects_turns_a_path_chiral_and_the_write_carries_the_phase(cmd, read,
                                                                        hub):
    """THE CAPSTONE for this door: turning a plain SVG path chiral through `setObjects`
    writes `v0 + v_k sin(theta + phi)` -- the SAME identity §9 asserts for the
    parameter door, so the two builders cannot have forked. phi = -90 is one
    enantiomer: (u, vn) traces the unit circle and the sign of vn follows -d(dev)/ds.
    """
    import numpy as np

    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": SVG_WITH_IDS})
    listing = cmd.call("setObjects", {"token": token, "updates": {
        "path4": {"voltage": 10.0, "speed_um_s": 400.0,
                  "chiral": {"lambda_um": 0.1, "y_amp_um": 0.05, "v_k": 5.0,
                             "phase_deg": -90.0, "hand": 1.0}}}})
    ch = {o["id"]: o for o in listing["objects"]}["path4"]["properties"]["chiral"]
    assert ch == {"lambda_um": pytest.approx(0.1), "lambda_built_um": pytest.approx(0.1),
                  "y_amp_um": pytest.approx(0.05),
                  "v_k": 5.0, "phase_deg": pytest.approx(-90.0), "hand": 1.0}
    res = cmd.call("startWrite", {"token": token, "objects": ["path4"], **_gw()})
    rec = _wait_run(hub, res["run_id"])
    assert rec.completed and not rec.aborted, rec.error
    trace = read.call("getWriteResult", {"run_id": res["run_id"],
                                         "inline": True})["trace"]
    assert len(trace) == 1                       # one stroke, however long
    entry = trace[0]
    dev, s = _deviation(entry["path_um"])
    u = dev / 0.05
    vn = (np.asarray(entry["v_tip"]) - 10.0) / 5.0
    assert np.abs(u).max() == pytest.approx(1.0, abs=0.1)
    assert np.abs(vn).max() == pytest.approx(1.0, abs=0.1)
    assert np.abs(vn ** 2 + u ** 2 - 1.0).max() < 0.15      # quadrature, not in phase
    slope = np.gradient(u, s)
    steep = np.abs(slope) > 0.5 * np.abs(slope).max()
    agree = (np.sign(vn[steep]) == np.sign(slope[steep])).mean()
    assert agree < 0.05                          # phi = -90: the sign is inverted


def test_a_second_setobjects_changes_only_the_phase(cmd, read, hub):
    """Merging `{phase_deg}` alone keeps lambda/amp/v_k/hand and moves ONLY the
    written phase -- so a phase SWEEP is one small edit per point, not a rebuild."""
    import numpy as np

    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": SVG_WITH_IDS})
    cmd.call("setObjects", {"token": token, "updates": {
        "path4": {"voltage": 10.0, "speed_um_s": 400.0,
                  "chiral": {"lambda_um": 0.1, "y_amp_um": 0.05, "v_k": 5.0,
                             "phase_deg": -90.0, "hand": 1.0}}}})
    # change ONLY the phase, to the other enantiomer
    res = cmd.call("setObjects", {"token": token,
                                  "updates": {"path4": {"chiral": {"phase_deg": 90.0}}}})
    ch = {o["id"]: o for o in res["objects"]}["path4"]["properties"]["chiral"]
    assert ch == {"lambda_um": pytest.approx(0.1), "lambda_built_um": pytest.approx(0.1),
                  "y_amp_um": pytest.approx(0.05),
                  "v_k": 5.0, "phase_deg": pytest.approx(90.0), "hand": 1.0}
    started = cmd.call("startWrite", {"token": token, "objects": ["path4"], **_gw()})
    rec = _wait_run(hub, started["run_id"])
    assert rec.completed and not rec.aborted, rec.error
    entry = read.call("getWriteResult", {"run_id": started["run_id"],
                                         "inline": True})["trace"][0]
    dev, s = _deviation(entry["path_um"])
    u = dev / 0.05
    vn = (np.asarray(entry["v_tip"]) - 10.0) / 5.0
    slope = np.gradient(u, s)
    steep = np.abs(slope) > 0.5 * np.abs(slope).max()
    agree = (np.sign(vn[steep]) == np.sign(slope[steep])).mean()
    assert agree > 0.95                          # phi = +90: the OTHER enantiomer


def test_setobjects_chiral_null_restores_the_plain_path(cmd, read):
    """`chiral: null` collapses the wire back to its base axis -- the straight path's
    original point count, not the tens-of-points modulated toolpath."""
    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": SVG_WITH_IDS})
    plain = {o["id"]: o for o in read.call("getPattern")["objects"]}["path4"]["points"]
    cmd.call("setObjects", {"token": token, "updates": {
        "path4": {"chiral": {"lambda_um": 0.1, "y_amp_um": 0.05, "v_k": 5.0,
                             "phase_deg": 0.0, "hand": 1.0}}}})
    modulated = {o["id"]: o
                 for o in read.call("getPattern")["objects"]}["path4"]["points"]
    assert modulated > plain                     # the wiggle added points
    res = cmd.call("setObjects", {"token": token,
                                  "updates": {"path4": {"chiral": None}}})
    # `getPattern`'s `points` is the DESIGN polyline count (the setObjects listing's is
    # the far larger COMPILED count), so compare like for like against `plain`.
    restored = {o["id"]: o
                for o in read.call("getPattern")["objects"]}["path4"]["points"]
    assert restored == plain
    assert {o["id"]: o for o in res["objects"]}["path4"]["properties"]["chiral"] is None


def test_setobjects_disable_makes_all_skip_the_object(cmd, read, hub):
    """A disabled object is not written by `objects: "all"` -- and it still shows in
    the listing, with `enabled: false`, so a script can see it and re-enable it."""
    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": SVG_WITH_IDS})
    listing = cmd.call("setObjects", {"token": token,
                                      "updates": {"path4-1": {"enabled": False}}})
    by_id = {o["id"]: o for o in listing["objects"]}
    assert by_id["path4-1"]["properties"]["enabled"] is False    # still listed
    res = cmd.call("startWrite", {"token": token, "objects": "all", **_gw()})
    rec = _wait_run(hub, res["run_id"])
    assert rec.completed and not rec.aborted, rec.error
    written = read.call("getWriteResult", {"run_id": res["run_id"],
                                           "inline": True})["trace"]
    # a plain path's trace `name` is the object name (a filled shape would suffix
    # "#k"); strip the suffix so the comparison is by design object.
    wrote = {t["name"].split("#")[0] for t in written}
    assert "path4-1" not in wrote
    assert {"path1", "path2", "path4"} <= wrote


def test_startwrite_accepts_a_list_of_ids_in_order(cmd, read, hub):
    """`objects` may be a list of ids, written in the order given -- proven by giving
    each a distinct bias and reading the written trace back in that order."""
    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": SVG_WITH_IDS})
    cmd.call("setObjects", {"token": token,
                            "updates": {"path1": {"voltage": 3.0},
                                        "path2": {"voltage": 6.0}}})
    res = cmd.call("startWrite", {"token": token, "objects": ["path2", "path1"], **_gw()})
    rec = _wait_run(hub, res["run_id"])
    assert rec.completed and not rec.aborted, rec.error
    trace = read.call("getWriteResult", {"run_id": res["run_id"],
                                         "inline": True})["trace"]
    assert [t["voltage"] for t in trace] == [6.0, 3.0]


def test_startwrite_refuses_an_unknown_id(cmd):
    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": SVG_WITH_IDS})
    reply = cmd.send("startWrite", {"token": token, "objects": ["nope"], **_gw()})
    assert reply["error"]["code"] == fd.INVALID_PARAMS
    assert "nope" in reply["error"]["message"]


@pytest.mark.parametrize("updates,field", [
    ({"nope": {"voltage": 1.0}}, "id"),                    # unknown id
    ({"path1": {"phase_deg": 1.0}}, "phase_deg"),          # unknown property
    ({"path1": {"voltage": "loud"}}, "voltage"),           # wrong type
    ({"path1": {"enabled": 1}}, "enabled"),                # not a real bool
    ({"path1": {"fill": {"kind": "contour"}}}, "fill"),    # fill on a path
])
def test_setobjects_refuses_bad_updates_with_the_pattern_intact(cmd, read, updates,
                                                                field):
    """Every refusal NAMES the id and the field, and the WHOLE call is refused: the
    previous pattern is left exactly as it was."""
    cmd.call("loadPattern", {"token": _token(cmd), "source": SVG_WITH_IDS})
    before = read.call("getPattern")
    reply = cmd.send("setObjects", {"token": _token(cmd), "updates": updates})
    err = reply["error"]
    assert err["code"] == fd.INVALID_PARAMS, reply
    assert err["data"]["param"] == "updates"
    assert err["data"]["field"] == field
    assert read.call("getPattern") == before


def test_setobjects_refuses_chiral_on_a_region(cmd, hub):
    """The WRONG-KIND refusal from the other side: a region has no chirality."""
    token = _token(cmd)
    # ERASE_SPEC is a -10 V erase, outside the default [-8, 30] window the loadPattern
    # door now enforces (spec 13.1 item 19); widen the sample window for the setup load.
    with _model_override(hub, tip_voltage_window_v=(-12.0, 30.0)):
        cmd.call("loadPattern", {"token": token, "objects": [ERASE_SPEC]})
    reply = cmd.send("setObjects", {"token": token,
                                    "updates": {"erase-pad": {"chiral": None}}})
    assert reply["error"]["code"] == fd.INVALID_PARAMS
    assert reply["error"]["data"]["field"] == "chiral"


def test_setobjects_is_refused_busy_while_a_write_runs(cmd, hub):
    """Editing the pattern a running write is reading would change what it writes, so
    setObjects is -32020 busy for as long as the write is in flight."""
    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": SVG_WITH_IDS})
    started = cmd.call("startWrite", {"token": token, "objects": "all", **_gw()})
    try:
        reply = cmd.send("setObjects", {"token": token,
                                        "updates": {"path1": {"voltage": 1.0}}})
        assert reply["error"]["code"] == fd.BUSY
    finally:
        _wait_run(hub, started["run_id"])


def test_the_flex_provider_setobjects_worked_example_runs(cmd, read, hub):
    """The worked example documented in docs/FLEX_PROVIDER.md, end to end: load the
    SVG, set voltages, turn path4 chiral and disable path4-1, write two objects by id,
    then change ONLY the phase and write path4."""
    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "source": SVG_WITH_IDS})
    cmd.call("setObjects", {"token": token, "updates": {
        "path1": {"voltage": -6.0},
        "path2": {"voltage": 7.0},
        "path4": {"voltage": 10.0, "speed_um_s": 400.0,
                  "chiral": {"lambda_um": 0.1, "y_amp_um": 0.05, "v_k": 5.0,
                             "phase_deg": -90.0, "hand": 1.0}},
        "path4-1": {"enabled": False}}})
    first = cmd.call("startWrite", {"token": token, "objects": ["path1", "path2"], **_gw()})
    assert _wait_run(hub, first["run_id"]).completed
    cmd.call("setObjects", {"token": token,
                            "updates": {"path4": {"chiral": {"phase_deg": 90.0}}}})
    second = cmd.call("startWrite", {"token": token, "objects": ["path4"], **_gw()})
    rec = _wait_run(hub, second["run_id"])
    assert rec.completed and not rec.aborted, rec.error
    pat = {o["id"]: o for o in read.call("getPattern")["objects"]}
    assert pat["path4"]["properties"]["chiral"]["phase_deg"] == pytest.approx(90.0)
    assert pat["path4-1"]["properties"]["enabled"] is False


# ========== 11. FLEX doors run the design voltage-window + DAC-rail guard =========
#
# Spec 13.1 item 19 (MFP-3D 2026-09-23): `setObjects` and `loadPattern` used to let a
# voltage outside `tip_voltage_window_v` / over the DAC rail into the design; only REST
# `set_props` and the write doors refused, so a script could stage an out-of-window
# object and the cockpit's Write would be the one to trip. Both FLEX doors now run the
# SAME `command_guard.check_design_voltages` the set_props door uses, BEFORE installing
# the design, refusing over the wire as -32050 (`GuardRefused`) with `.reason` and
# leaving the previous design untouched.

import contextlib as _contextlib
import dataclasses as _dataclasses

from afm_litho.service.command_guard import GUARD_CODE
from afm_litho.service.instrument_model import Field as _Field
from afm_litho.service.instrument_model import FieldSource as _FieldSource


@_contextlib.contextmanager
def _model_override(hub, **fields):
    """Temporarily swap `InstrumentModel` fields on the live process hub (window,
    amp gain, rail) for a case that needs a different sample envelope, restoring the
    real model afterwards. Mirrors `test_an_erase_rectangle_...`'s dataclasses.replace
    dance; guard scope is unchanged, only the model values under test move."""
    saved = hub.host.instrument_model
    repl = {k: _Field(value=v, source=_FieldSource.override) for k, v in fields.items()}
    hub.host.instrument_model = _dataclasses.replace(saved, **repl)
    try:
        yield
    finally:
        hub.host.instrument_model = saved


def test_setobjects_over_window_voltage_refused_and_object_unchanged(cmd, read, hub):
    """setObjects a voltage outside the sample window is refused as -32050
    tip_voltage_out_of_window, and the object keeps its old voltage (8.0)."""
    _load_ids(cmd)                                       # path1..path4-1, each at 8.0 V
    with _model_override(hub, tip_voltage_window_v=(-10.0, 10.0)):
        reply = cmd.send("setObjects", {"token": _token(cmd),
                                        "updates": {"path1": {"voltage": 30.0}}})
    err = reply["error"]
    assert err["code"] == GUARD_CODE, reply
    assert err["data"]["reason"] == "tip_voltage_out_of_window"
    assert err["data"]["object_id"] == "path1"
    # the live design is untouched: path1 is still at its loaded 8.0 V
    pat = {o["id"]: o for o in read.call("getPattern")["objects"]}
    assert pat["path1"]["properties"]["voltage"] == 8.0


def test_setobjects_voltage_alone_rebakes_a_chiral_wire(cmd, read, hub):
    """`{voltage}` with no `chiral` block re-applies the wire's chiral settings at the
    new v0: the object, `getPattern` and the write all agree."""
    import numpy as np

    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "objects": [_chiral(phase_deg=90.0)]})
    before = {o["id"]: o for o in read.call("getPattern")["objects"]}["helix"]
    listing = cmd.call("setObjects", {"token": token,
                                      "updates": {"helix": {"voltage": 6.0}}})
    props = {o["id"]: o for o in listing["objects"]}["helix"]["properties"]
    assert props["voltage"] == 6.0
    assert props["chiral"] == before["properties"]["chiral"]

    obj = next(o for o in hub.pattern.objects if o.name == "helix")
    assert obj.voltage == 6.0 and obj.profile["v0"] == 6.0
    assert min(obj.v_profile) == pytest.approx(1.0, abs=5e-3)
    assert max(obj.v_profile) == pytest.approx(11.0, abs=5e-3)

    after = {o["id"]: o for o in read.call("getPattern")["objects"]}["helix"]
    assert after["properties"]["voltage"] == 6.0 and after["voltage"] == 6.0
    assert after["properties"]["chiral"] == before["properties"]["chiral"]
    assert after["vertices_um"] == before["vertices_um"]
    assert after["points"] == before["points"]

    res = cmd.call("startWrite", {"token": token, "objects": ["helix"], **_gw()})
    rec = _wait_run(hub, res["run_id"])
    assert rec.completed and not rec.aborted, rec.error
    entry = read.call("getWriteResult", {"run_id": res["run_id"],
                                         "inline": True})["trace"][0]
    v = np.asarray(entry["v_tip"], dtype=float)
    assert v.min() == pytest.approx(1.0, abs=0.1)
    assert v.max() == pytest.approx(11.0, abs=0.1)
    assert v.mean() == pytest.approx(6.0, abs=0.1)


def test_setobjects_voltage_on_a_chiral_wire_is_judged_on_the_rebuilt_profile(cmd, read,
                                                                              hub):
    """28 V is inside [-8, 30] but 28 + v_k is not: refused, naming the object, and
    the wire is left as loaded."""
    token = _token(cmd)
    cmd.call("loadPattern", {"token": token, "objects": [_chiral()]})
    reply = cmd.send("setObjects", {"token": token,
                                    "updates": {"helix": {"voltage": 28.0}}})
    err = reply["error"]
    assert err["code"] == GUARD_CODE, reply
    assert err["data"]["reason"] == "tip_voltage_out_of_window"
    assert err["data"]["object_id"] == "helix"
    obj = next(o for o in hub.pattern.objects if o.name == "helix")
    assert obj.voltage == 10.0 and obj.profile["v0"] == 10.0
    assert max(obj.v_profile) == pytest.approx(15.0, abs=5e-3)


def test_loadpattern_over_window_object_refused_previous_pattern_kept(cmd, read, hub):
    """loadPattern an over-window object is refused as -32050 tip_voltage_out_of_window,
    and the previously loaded pattern stays loaded."""
    _load_ids(cmd)
    before = read.call("getPattern")
    with _model_override(hub, tip_voltage_window_v=(-10.0, 10.0)):
        reply = cmd.send("loadPattern",
                         {"token": _token(cmd),
                          "objects": [{**ERASE_SPEC, "voltage": 30.0}]})
    err = reply["error"]
    assert err["code"] == GUARD_CODE, reply
    assert err["data"]["reason"] == "tip_voltage_out_of_window"
    # the previous SVG pattern is exactly as it was
    assert read.call("getPattern") == before


def test_setobjects_within_rail_with_gain_one_accepted(cmd, read, hub):
    """8 V with amp_gain 1.0 and a 10 V rail is INSIDE the rail (|8/1| = 8 <= 10) and
    inside the window, so setObjects accepts it and the change loads."""
    _load_ids(cmd)
    with _model_override(hub, tip_voltage_window_v=(-40.0, 40.0),
                         tip_amp_gain=1.0, dac_rail_v=10.0):
        res = cmd.call("setObjects", {"token": _token(cmd),
                                      "updates": {"path1": {"voltage": 8.0}}})
    by_id = {o["id"]: o for o in res["objects"]}
    assert by_id["path1"]["properties"]["voltage"] == 8.0
    pat = {o["id"]: o for o in read.call("getPattern")["objects"]}
    assert pat["path1"]["voltage"] == 8.0


def test_setobjects_over_rail_with_gain_one_refused_even_with_wide_window(cmd, read,
                                                                          hub):
    """12 V with amp_gain 1.0 and a 10 V rail exceeds the DAC rail (|12/1| = 12 > 10)
    even though the window is wide open, so setObjects refuses it as -32050
    dac_rail_exceeded and the object keeps its old voltage."""
    _load_ids(cmd)
    with _model_override(hub, tip_voltage_window_v=(-40.0, 40.0),
                         tip_amp_gain=1.0, dac_rail_v=10.0):
        reply = cmd.send("setObjects", {"token": _token(cmd),
                                        "updates": {"path1": {"voltage": 12.0}}})
    err = reply["error"]
    assert err["code"] == GUARD_CODE, reply
    assert err["data"]["reason"] == "dac_rail_exceeded"
    assert err["data"]["object_id"] == "path1"
    pat = {o["id"]: o for o in read.call("getPattern")["objects"]}
    assert pat["path1"]["properties"]["voltage"] == 8.0
