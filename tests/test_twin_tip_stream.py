"""The twin tip stream (docs/TWIN_TIP_STREAM_SPEC.md): `tip_pos` frames on the twin only.

Once the twin plugin has run a write or a scan, its telemetry is the cockpit's tip display
(4 Hz, 10 Hz while writing a chain or scanning); the old stream, which reads the old
twin's position, stands down. A ws client watches a twin write (frames through the stroke
AND through the transit to it) and a twin scan (the tip stepping down the frame), each
driven over REST and over FLEX. With a FakeIgorBridge attached the twin sends nothing and
the hardware `tip_pos` frames are the TELEM poll's, unchanged.
"""

from __future__ import annotations

import contextlib
import json
import socket
import time
import types

import pytest
from _im_fixtures_guard import GUARD_WRITE_KW, seed_free_air

from afm_litho.litho.compiler import LithoObject, Pattern
from afm_litho.service import twin_tip_stream as tts
from afm_litho.service.tip_stream import tip_pos_msg

zmq = pytest.importorskip("zmq", reason="pyzmq is the optional [flex] extra")

from afm_litho.service import flex_provider as fp

# One stroke, far enough from the tip's start (the origin) that the transit to it and the
# stroke itself each last a second or two at TWIN_SPEED, so the plugin's telemetry (4 Hz while
# moving or writing) puts several frames on each.
A_START, A_END = (-8.0, -6.0), (0.0, -6.0)
STROKE_UM_S = 2.0
TWIN_SPEED = 2.0
# A frame this close (um) to an end of the transit counts as an endpoint, not travel.
ENDPOINT_TOL_UM = 0.5
DEADMAN_S = 300


def _pattern() -> Pattern:
    kw = {"voltage": 6.0, "speed_um_s": STROKE_UM_S, "interpolation_um": 0.05}
    return Pattern(name="tip-stream", objects=[LithoObject("a", "path", [A_START, A_END], **kw)])


def _plugin(hub):
    from afm_litho.service.twin_plugin import plugin
    return plugin(hub)


def _tip_at_origin(hub):
    """Put the plugin's tip at the origin, instantly, so a write's transit has a known start."""
    from afm_litho.instrument import MoveTo
    twin = _plugin(hub)
    twin.speed_factor = 0
    assert twin.move(MoveTo(0.0, 0.0, 5.0)).done
    return twin


# ------------------------------------------------------------------ fixtures --

@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


def _free_port() -> int:
    for _ in range(50):
        a, b = socket.socket(), socket.socket()
        try:
            a.bind(("127.0.0.1", 0))
            port = a.getsockname()[1]
            b.bind(("127.0.0.1", port + 1))
            return port
        except OSError:
            continue
        finally:
            a.close()
            b.close()
    raise RuntimeError("no free port pair")


class _Flex:
    def __init__(self, endpoint: str):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, 30000)
        self.sock.setsockopt(zmq.SNDTIMEO, 30000)
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
def flex(hub):
    prov = fp.FlexProvider(hub, bind="127.0.0.1", port=_free_port(),
                           log=lambda msg: None)
    prov.start()
    c = _Flex(prov.bound["command"])
    yield c
    c.close()
    prov.stop()


def _token(flex, hub) -> str:
    reply = flex.send("acquireControl", {"client": "tip-stream", "deadman_s": DEADMAN_S})
    if "error" in reply:
        hub.control = None
        reply = flex.send("acquireControl",
                          {"client": "tip-stream", "deadman_s": DEADMAN_S})
    return reply["result"]["token"]


@pytest.fixture(autouse=True)
def _clean(hub, tmp_path):
    """Each case starts from a quiet twin and leaves it so."""
    from afm_litho.service.sessions import SessionStore
    prev_store, prev_speed, prev_transit = hub.store, hub.host.speed, hub.transit
    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    # The hub is a process global: a litho frame another module left set would offset
    # and rotate these writes off the lines the cases look along.
    prev_frame = getattr(hub, "litho_frame", None)
    hub.litho_frame = None
    yield
    _wait(lambda: hub.state_label() == "idle", 60.0)
    _wait(lambda: getattr(hub, "_run_claim", None) is None, 10.0)
    cur = getattr(hub, "control", None)
    if cur is not None:
        hub.control = None
    hub.host.written_objects = set()
    hub.store, hub.host.speed, hub.transit = prev_store, prev_speed, prev_transit
    hub.litho_frame = prev_frame


def _wait(cond, timeout=20.0, interval=0.02) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return False


def _collect(ws, *, until: str | None = None, seconds: float | None = None,
             timeout: float = 60.0) -> list[dict]:
    """Frames off the socket until one of type `until` arrives, or for `seconds`.
    The /ws heartbeat sends telemetry every 0.25 s, so a receive never blocks long."""
    out: list[dict] = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < (seconds if seconds is not None else timeout):
        m = ws.receive_json()
        out.append(m)
        if until is not None and m.get("type") == until:
            return out
    assert until is None, f"no {until!r} frame within {timeout} s"
    return out


def _tips(frames) -> list[tuple[float, float]]:
    return [(m["x_um"], m["y_um"]) for m in frames if m.get("type") == "tip_pos"]


def _open(client):
    ws = client.websocket_connect("/ws")
    return ws


# --------------------------------------------------------------- unit: shape --

def test_frame_is_the_hardware_tip_pos_message(hub):
    """Same builder, same keys, same rounding as the hardware frame."""
    msg = tts.frame(hub)
    hw = tip_pos_msg(0.1, 0.2, {"x_lvdt_sens": 7.46e-6, "x_lvdt_offset": -0.63,
                                "y_lvdt_sens": 7.04e-6, "y_lvdt_offset": 0.07})
    assert set(msg) == set(hw) == {"type", "x_um", "y_um"}
    assert msg["type"] == "tip_pos"
    snap = hub.host.arc.snapshot()
    assert msg["x_um"] == pytest.approx(snap["x_um"], abs=1e-3)
    assert msg["y_um"] == pytest.approx(snap["y_um"], abs=1e-3)
    assert msg["x_um"] == round(msg["x_um"], 3)


# ------------------------------------------------------------------- at rest --

def test_the_plugins_telemetry_is_the_tip_at_rest(client, hub):
    """Nothing running: the marker is still refreshed, at the plugin's 4 Hz."""
    from afm_litho import clock
    twin = _tip_at_origin(hub)
    assert _wait(lambda: hub.state_label() == "idle")
    time.sleep(0.3)                       # let any motion from a previous case settle
    with _open(client) as ws:
        frames = _collect(ws, seconds=3.0)
    tips = _tips(frames)
    assert 6 <= len(tips) <= 20, f"{len(tips)} tip_pos frames in 3 s at rest"
    x, y = twin.position_at(clock.now())
    assert tips[-1] == (pytest.approx(x, abs=2e-3), pytest.approx(y, abs=2e-3))


def test_the_service_has_no_switch_for_the_twin_stream(client, hub):
    """Show tip on the twin is each page's own display switch. `/api/tip/show` (which
    a twin page does not call) leaves the stream running either way."""
    try:
        r = client.post("/api/tip/show", json={"on": False}).json()
        assert r == {"ok": True, "on": False, "backend": "twin"}
        with _open(client) as ws:
            assert _tips(_collect(ws, seconds=1.5))
    finally:
        hub._show_tip_display = False


def test_the_old_stream_stands_down_once_the_plugin_is_in_use(hub, monkeypatch):
    """The plugin's telemetry is then the one source of positions: the old stream, which
    reads the old twin, would put the marker back where that twin stands."""
    monkeypatch.setattr(hub.host, "_twin_plugin", None)
    assert tts.owns_tip_display(hub) is False
    monkeypatch.setattr(hub.host, "_twin_plugin", object())
    assert tts.owns_tip_display(hub) is True
    with _open_client_ws(hub) as sent:
        time.sleep(1.0)
        assert sent == []
        monkeypatch.setattr(hub.host, "_twin_plugin", None)
        assert _wait(lambda: len(sent) >= 1, 3.0)


class _open_client_ws:
    """Count the stream loop's own sends (`tts.send`) without a socket."""

    def __init__(self, hub):
        self.hub = hub
        self.sent: list = []

    def __enter__(self):
        self._real = tts.send

        async def spy(h, now=None, at_um=None):
            self.sent.append(at_um)
            return True

        tts.send = spy
        return self.sent

    def __exit__(self, *exc):
        tts.send = self._real


# --------------------------------------------------------------------- write --

def _start_write(door: str, client, hub, flex) -> None:
    hub.pattern = _pattern()
    hub.host.written_objects = set()
    fa = seed_free_air(hub)
    if door == "rest":
        r = client.post("/api/litho/write", json={
            "objects": ["a"], "setpoint_relative": 0.2, "twin_speed": TWIN_SPEED,
            **GUARD_WRITE_KW}).json()
        assert r.get("ok") is True and r["writing"] == 1, r
    else:
        tok = _token(flex, hub)
        res = flex.call("startWrite", {"token": tok, "objects": "all", "twin_speed": TWIN_SPEED,
                                       "setpoint_reference": "absolute", "setpoint": fa.value_v + 0.2, "amp_gain": 25.0})
        assert res["objects"] == 1, res


@pytest.mark.parametrize("door", ["rest", "flex"])
def test_write_streams_tip_pos_through_the_transit_and_the_stroke(client, hub, flex, door):
    _tip_at_origin(hub)
    with _open(client) as ws:
        _start_write(door, client, hub, flex)
        frames = _collect(ws, until="litho_done", timeout=90.0)
        after = _collect(ws, seconds=1.5)
    assert not frames[-1].get("aborted"), frames[-1]
    tips = _tips(frames)

    # through the transit: positions strictly between the origin and the stroke's start, on
    # the straight line between them (y = 0.75 x), in travel order
    def on_transit(p):
        x, y = p
        return (A_START[0] + ENDPOINT_TOL_UM < x < -ENDPOINT_TOL_UM
                and abs(y - 0.75 * x) < 0.1)
    transit = [p for p in tips if on_transit(p)]
    assert len(transit) >= 2, f"the transit showed {transit} of {tips}"
    assert [p[0] for p in transit] == sorted((p[0] for p in transit), reverse=True), transit

    # through the stroke: positions strictly inside it, on its line
    stroke = [p for p in tips if abs(p[1] - A_START[1]) < 0.05
              and A_START[0] + ENDPOINT_TOL_UM < p[0] < A_END[0] - ENDPOINT_TOL_UM]
    assert len(stroke) >= 2, f"the stroke showed {stroke} of {tips}"
    assert [p[0] for p in stroke] == sorted(p[0] for p in stroke), stroke

    # the object's progress frame still comes, one per stroke, at the stroke's end
    progress = [m for m in frames if m.get("type") == "litho_progress"]
    assert [(m["x_um"], m["y_um"]) for m in progress] == [A_END]

    # the stream goes on after the write: the marker stays where the tip stopped
    last = _tips(after)
    assert last, "no tip_pos after litho_done"
    assert last[-1] == (pytest.approx(A_END[0], abs=0.05), pytest.approx(A_END[1], abs=0.05))


def test_a_write_takes_its_speed_from_the_call_else_from_the_station(client, hub, flex):
    """`twin_speed` on the write door overrides `hub.twin_speed` for that write."""
    assert hub.host.twin_speed == 0.0                 # the suite's setting (conftest)
    hub.pattern = Pattern(name="one", objects=[LithoObject(
        "a", "path", [(0.0, 0.0), (0.2, 0.0)], voltage=6.0, speed_um_s=5.0,
        interpolation_um=0.05)])
    seed_free_air(hub)
    twin = _plugin(hub)
    for body, want in (({}, 0.0), ({"twin_speed": 40}, 40.0), ({"twin_speed": "0"}, 0.0)):
        twin.speed_factor = -1.0               # the worker sets it for the write
        hub.host.written_objects = set()
        r = client.post("/api/litho/write", json={
            "objects": ["a"], "setpoint_relative": 0.2, **GUARD_WRITE_KW, **body}).json()
        assert r.get("ok") is True, r
        assert _wait(lambda want=want: twin.speed_factor == want, 5.0), (body, twin.speed_factor)
        assert _wait(lambda: hub.state_label() == "idle", 30.0)
        assert _wait(lambda: getattr(hub, "_run_claim", None) is None, 10.0)


@pytest.mark.parametrize("bad", [-1, "fast", True])
def test_a_bad_twin_speed_refuses_the_write_on_both_doors(client, hub, flex, bad):
    hub.pattern = _pattern()
    hub.host.written_objects = set()
    fa = seed_free_air(hub)
    r = client.post("/api/litho/write", json={
        "objects": ["a"], "setpoint_relative": 0.2, **GUARD_WRITE_KW,
        "twin_speed": bad}).json()
    assert "twin_speed" in r.get("error", "") and r.get("reason") == "bad_params", r
    reply = flex.send("startWrite", {"token": _token(flex, hub), "objects": "all",
                                     "setpoint_reference": "absolute", "setpoint": fa.value_v + 0.2, "amp_gain": 25.0,
                                     "twin_speed": bad})
    assert reply["error"]["code"] == -32602 and "twin_speed" in reply["error"]["message"]
    assert hub.state_label() == "idle" and getattr(hub, "_run_claim", None) is None


# ---------------------------------------------------------------------- scan --

FAST_SCAN = {"size_um": 4.0, "pixels": 16, "lines": 8, "line_rate_hz": 200.0,
             "mode": "contact"}


def _start_scan(door: str, client, hub, flex, body: dict) -> None:
    if door == "rest":
        r = client.post("/api/scan/start", json=body).json()
        assert r.get("ok") is True, r
    else:
        tok = _token(flex, hub)
        flex.call("startScan", {"token": tok, **body})


@pytest.mark.parametrize("door", ["rest", "flex"])
def test_scan_shows_the_tip_stepping_down_the_frame(client, hub, flex, door):
    """While the raster runs the plugin's telemetry (10 Hz) puts the tip on the lines: x
    sweeps across the frame and y steps up it, line by line."""
    hub.host.speed = 1.0
    body = {"size_um": 4.0, "pixels": 16, "lines": 6, "line_rate_hz": 4.0, "mode": "contact"}
    _tip_at_origin(hub)
    with _open(client) as ws:
        _start_scan(door, client, hub, flex, body)
        frames = _collect(ws, until="scan_done", timeout=60.0)
    lines = [m for m in frames if m.get("type") == "line"]
    assert [m["row"] for m in lines] == list(range(body["lines"]))
    first = frames.index(lines[0])
    tips = _tips(frames[first:])
    half = body["size_um"] / 2
    inside = [p for p in tips if abs(p[0]) <= half and abs(p[1]) <= half]
    assert len(inside) >= body["lines"], tips
    ys = [p[1] for p in inside]
    assert ys == sorted(ys) and ys[-1] - ys[0] > 2.0, ys     # stepping up the frame
    xs = [p[0] for p in inside]
    assert max(xs) - min(xs) > 1.5, xs                       # sweeping across it


# ------------------------------------------------------------------ hardware --

CAL = {"x_lvdt_sens": 7.46e-6, "x_lvdt_offset": -0.63,
       "y_lvdt_sens": 7.04e-6, "y_lvdt_offset": 0.07,
       "z_lvdt_sens": 5e-6, "z_lvdt_offset": 0.0}
HW_VALUES = {"XSensor": 0.11, "YSensor": 0.22, "Output.Z": 0.0,
             "Deflection": 0.0, "DetectorSum": 5.0, "amplitude": 0.0}


def _restart_stream(hub) -> None:
    """Bring the twin stream back for the cases that follow (it ends itself for good
    once it has seen a real bridge)."""
    hub._loop.call_soon_threadsafe(tts.start, hub)
    assert _wait(lambda: hub._twin_tip_task is not None
                 and not hub._twin_tip_task.done(), 5.0)


def test_no_twin_frame_with_a_bridge_attached_and_hardware_frames_as_before(client, hub):
    from afm_litho.hardware.hardware_arc import HardwareARC
    from afm_litho.hardware.igor_stub import FakeIgorBridge
    from afm_litho.service.host import telemetry as C

    assert hub._twin_tip_task is not None and not hub._twin_tip_task.done()
    _plugin(hub)                      # listening, as after any twin write or scan
    hw_msg = tip_pos_msg(HW_VALUES["XSensor"], HW_VALUES["YSensor"], CAL)
    twin_msg = tts.frame(hub)
    assert twin_msg is not None and twin_msg != hw_msg

    stub = FakeIgorBridge(dict(HW_VALUES), telem_hz=20.0)
    port = stub.serve_in_thread()
    hw = HardwareARC.connect("127.0.0.1", port)
    prev = (hub.host.hardware_arc, hub.host.hw_cal, hub._show_tip_display, hub.host._hw_telemetry)
    try:
        with _open(client) as ws:
            hub.host.hw_cal = dict(CAL)
            # attached AFTER startup, with the twin stream already running
            hub.host.hardware_arc = hw
            C.install_hw_telemetry(hw)
            assert tts.twin_only(hub) is False
            assert tts.frame(hub) is None
            assert _wait(lambda: hub._twin_tip_task.done(), 2.0), \
                "the twin stream kept running with a bridge attached"
            _collect(ws, seconds=0.3)              # drain frames sent before the attach

            # show-tip OFF (the hardware default): no tip_pos from anyone
            hub._show_tip_display = False
            assert _tips(_collect(ws, seconds=1.5)) == []

            # the twin plugin is listening but its frames are shut at the entry: nothing
            # of its own goes out with a bridge attached
            assert tts.frame(hub, None, (1.0, 2.0)) is None
            assert _tips(_collect(ws, seconds=1.0)) == []

            # show-tip ON: the TELEM poll's frames, and only those
            hub._show_tip_display = True
            frames = [m for m in _collect(ws, seconds=1.5) if m.get("type") == "tip_pos"]
            assert len(frames) >= 10, len(frames)             # ~20 Hz poll
            assert all(m == hw_msg for m in frames), frames
    finally:
        with contextlib.suppress(Exception):
            hw.close()
        stub.stop()
        (hub.host.hardware_arc, hub.host.hw_cal, hub._show_tip_display, hub.host._hw_telemetry) = prev
        _restart_stream(hub)


def test_a_closed_bridge_stays_silent(hub):
    """A bridge that dropped is still a bound bridge: `backend_live()` says "twin"
    again, and the twin stream must not start drawing a simulated tip."""
    prev = hub.host.hardware_arc
    hub.host.hardware_arc = types.SimpleNamespace(link=None)
    hub.host.bridge_closed.set()
    try:
        assert hub.backend_live() == "twin"
        assert tts.twin_only(hub) is False and tts.frame(hub) is None
    finally:
        hub.host.bridge_closed.clear()
        hub.host.hardware_arc = prev


def test_a_hardware_station_never_gets_the_task(client, hub, monkeypatch):
    """Configured for hardware with no bridge yet: no task, no frame, and neither
    door a page reads says "twin" (which is what makes a page turn Show Tip on)."""
    monkeypatch.setattr(hub.host, "backend", "hardware")
    prev_arc, hub.host.hardware_arc = hub.host.hardware_arc, None
    try:
        assert hub.backend_live() == "twin"             # the label that must not leak
        assert client.get("/api/state").json()["backend"] == "disconnected"
        try:
            r = client.post("/api/tip/show", json={"on": True}).json()
            assert r == {"ok": True, "on": True, "backend": "disconnected"}
        finally:
            hub._show_tip_display = False
        with _open(client) as ws:
            assert _wait(lambda: hub._twin_tip_task is None
                         or hub._twin_tip_task.done(), 2.0)
            _collect(ws, seconds=0.2)
            assert _tips(_collect(ws, seconds=1.5)) == []
    finally:
        hub.host.hardware_arc = prev_arc
    assert tts.twin_only(hub) is False
    assert tts.frame(hub) is None
    done: list = []
    hub._loop.call_soon_threadsafe(lambda: done.append(tts.start(hub)))
    assert _wait(lambda: bool(done), 5.0)
    assert done == [None] and hub._twin_tip_task is None
    monkeypatch.undo()
    _restart_stream(hub)
