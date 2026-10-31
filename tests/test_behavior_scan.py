"""Behavioural tests for scan / sample / surface through the FastAPI service.

These are invariants and workflows, not smoke: a scan's error channel must
follow its imaging mode, a frame must arrive whole, a completed frame must fuse
into the Surface, and an .ibw's own metadata must reconfigure the twin.

`client` comes from tests/conftest.py (a TestClient). The service's
`hub` is a PROCESS GLOBAL, so every test here is self-contained and *always*
aborts and drains any scan it starts -- a scan left running makes the next
scan_start return {"error": "scan already running"}. Each test also redirects
hub.store to a per-test temp file so a completing scan cannot append to the real
runtime/sessions.jsonl.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

# `client` comes from tests/conftest.py (a TestClient).


# ----------------------------------------------------------------- helpers --

def _wait(cond, timeout=30.0, interval=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return False


def _tmp_store(tmp_path):
    """Point the session archive at a temp file so a scan that runs to
    completion records there, not in the real runtime/sessions.jsonl. Returns
    the live process-global hub for assertions."""
    from afm_litho.service.app import hub
    from afm_litho.service.sessions import SessionStore
    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    return hub


def _first_line(ws, timeout=6.0):
    """Pull the next {"type":"line"} broadcast off the socket, stepping over the
    interleaved telemetry heartbeats. Returns the message, or None if the frame
    finished (scan_done) or nothing streamed within the window."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        m = ws.receive_json()
        t = m.get("type")
        if t == "line":
            return m
        if t == "scan_done":
            return None
    return None


def _abort_and_drain(client, ws, timeout=6.0):
    """Stop the scan and consume the acknowledgement, THEN wait for the pump task
    to fully finish. The second wait is the important one: scan_start only spawns
    a new pump when the previous `hub._scan_task` is done(), so if the old task is
    still in its post-scan fusion/record when the next test starts, the new scan
    gets NO pump and streams 0 lines. Draining the ws isn't enough -- the task
    keeps running past scan_done -- so wait on the task itself."""
    from afm_litho.service.app import hub
    client.post("/api/scan/abort")
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        m = ws.receive_json()
        if m.get("type") in ("scan_done", "aborted"):
            break
    _wait(lambda: hub.host._scan_task is None or hub.host._scan_task.done(), timeout=timeout)


def _ready_ws(ws):
    """Block until the /ws handler has registered this client (past
    hub.clients.add) so a fast scan started next can't broadcast its lines before
    we're subscribed -- the flakiness these tests used to have. Sending a frame
    makes the handler's receive_text return at once and emit telemetry, which
    only happens after the add."""
    ws.send_text("ping")
    for _ in range(200):
        if ws.receive_json().get("type") == "telemetry":
            return
    raise AssertionError("ws never confirmed registration")


_TINY = {"size_um": 5.0, "pixels": 16, "lines": 4, "line_rate_hz": 200.0}


# ---------------------------------------------- scan mode -> error channel --
# _scan_pump broadcasts {"type":"line", "errName": ...}; the twin's scan_line
# (passes.py) names the error channel per mode. Contact reads cantilever
# deflection, AC reads oscillation amplitude, current reads tip current -- and
# _scan_pump's errName must agree with scan_line's channel map for all three
# (it did not for current mode; that KeyError-killed-the-pump bug is now fixed).

@pytest.mark.parametrize("mode,err_name", [
    ("contact", "Deflection"),
    ("ac", "Amplitude"),
])
def test_scan_mode_selects_error_channel(client, tmp_path, mode, err_name):
    _tmp_store(tmp_path)
    with client.websocket_connect("/ws") as ws:
        _ready_ws(ws)
        r = client.post("/api/scan/start", json={**_TINY, "mode": mode})
        assert r.json().get("ok") is True, r.text
        try:
            line = _first_line(ws)
            assert line is not None, \
                f"no 'line' message streamed for mode={mode}"
            assert line["errName"] == err_name, \
                f"mode={mode} streamed errName={line['errName']!r}"
        finally:
            _abort_and_drain(client, ws)


# --------------------------------------------------- a scan streams a frame --

def test_scan_streams_the_full_frame(client, tmp_path):
    """Every configured line arrives, each row is `pixels` wide, and the frame
    is bracketed by exactly `lines` rows before scan_done."""
    _tmp_store(tmp_path)
    pixels, lines = 16, 4
    with client.websocket_connect("/ws") as ws:
        _ready_ws(ws)
        r = client.post("/api/scan/start", json={
            "size_um": 5.0, "pixels": pixels, "lines": lines,
            "line_rate_hz": 200.0, "mode": "contact"})
        assert r.json().get("ok") is True, r.text
        rows = []
        try:
            end = time.monotonic() + 20.0
            while time.monotonic() < end:
                m = ws.receive_json()
                t = m.get("type")
                if t == "line":
                    rows.append(m)
                elif t == "scan_done":
                    break
        finally:
            _abort_and_drain(client, ws)

    assert len(rows) == lines, \
        f"expected {lines} line rows, streamed {len(rows)}"
    # each streamed row carries full-width pixel arrays for every channel
    for m in rows:
        for ch in ("height", "retrace", "error", "zsensor"):
            assert len(m[ch]) == pixels, \
                f"row {m['row']} channel {ch} had {len(m[ch])} px, want {pixels}"
    # the slow-axis rows cover the whole frame exactly once
    assert sorted(m["slow_idx"] for m in rows) == list(range(lines))


# ----------------------------------------- non-square frames honour `lines` --
# Instrument.scan_line used to terminate at `row >= cfg.pixels` instead of
# `row >= cfg.lines`. That's invisible whenever lines <= pixels (the row gate
# at the TOP of scan_line already stops the frame correctly), so it only shows
# up once lines > pixels -- a tall/narrow raster used to quit after `pixels`
# rows instead of running the configured `lines`.

@pytest.mark.parametrize("pixels,lines", [(32, 8), (8, 32)])
def test_scan_honours_lines_not_pixels(client, tmp_path, pixels, lines):
    """A non-square frame runs exactly `lines` rows of `pixels` width, whichever
    of the two is larger -- not `min(lines, pixels)`."""
    _tmp_store(tmp_path)
    with client.websocket_connect("/ws") as ws:
        _ready_ws(ws)
        r = client.post("/api/scan/start", json={
            "size_um": 5.0, "pixels": pixels, "lines": lines,
            "line_rate_hz": 400.0, "mode": "contact"})
        assert r.json().get("ok") is True, r.text
        rows = []
        try:
            end = time.monotonic() + 30.0
            while time.monotonic() < end:
                m = ws.receive_json()
                t = m.get("type")
                if t == "line":
                    rows.append(m)
                elif t == "scan_done":
                    break
        finally:
            _abort_and_drain(client, ws)

    assert len(rows) == lines, \
        f"pixels={pixels} lines={lines}: expected {lines} rows, streamed {len(rows)}"
    for m in rows:
        assert len(m["height"]) == pixels, \
            f"row {m['row']} had {len(m['height'])} px, want {pixels}"
    assert sorted(m["slow_idx"] for m in rows) == list(range(lines))

    from afm_litho.service.app import hub
    frame = hub.last_scan
    assert frame is not None
    assert frame["rows"] == lines and frame["pixels"] == pixels, \
        f"cached frame {frame['rows']}x{frame['pixels']}, want {lines}x{pixels}"


# ------------------------------------------- a completed scan fuses surface --

def test_completed_scan_fuses_into_surface(client, tmp_path):
    """A frame that runs to completion becomes a Surface contribution: the
    contributions list grows and real height appears in the fused map."""
    hub = _tmp_store(tmp_path)

    # Load a sample first: that rebuilds hub.surface empty (see sample_load),
    # giving a known starting point rather than whatever a prior test fused.
    files = client.get("/api/samples").json()["files"]
    assert files, "no sample .ibw files to scan"
    assert client.post("/api/sample/load", json={"file": files[0]}).json().get("ok")

    n_before = len(hub.surface.contributions)

    # lines must exceed 4: _scan_pump only ingests when scan.row > 4.
    # The twin reads the sample only where the image is (flat outside it), so the frame
    # must lie on the loaded sample or the levelled map is flat.
    size_um = round(hub.host.world.sample.extent * 1e6 * 0.8, 3)
    r = client.post("/api/scan/start", json={
        "size_um": size_um, "pixels": 16, "lines": 8,
        "line_rate_hz": 200.0, "mode": "contact"})
    assert r.json().get("ok") is True, r.text
    try:
        # completion is signalled by the recorded scan session
        assert _wait(lambda: any(s["kind"] == "scan" for s in hub.store.list())), \
            "scan never recorded a session (pump did not run to completion)"
    finally:
        client.post("/api/scan/abort")   # no-op if already done; prevents leak

    contributions = hub.surface.contributions
    assert len(contributions) == n_before + 1,         "a completed frame did not add a Surface contribution"
    newest = contributions[-1]
    assert newest.cid.startswith("scan-") and newest.n_samples > 0
    # the fused map now holds real measured height, not an empty plane
    fused = hub.surface.snapshot(32, -size_um / 2, -size_um / 2, size_um)   # the scanned frame
    assert np.isfinite(fused).any(), "no pixels marked measured after a completed scan"
    assert np.nanmax(np.abs(fused)) > 0.0,         "surface fused a contribution but the height map stayed flat"


# ------------------------------------------ loading a sample reconfigures ---

def test_sample_load_reconfigures_the_twin(client):
    """Each .ibw's metadata drives the twin: loading a file sets the active
    sample name and a well-formed instrument, and two files that are different
    microscope models yield different model/scanner-range instruments.
    """
    files = client.get("/api/samples").json()["files"]
    assert len(files) >= 1, "no sample .ibw files present"

    required = {"model", "id", "range_um", "f0_khz", "invols_nm_v", "x_sensor"}
    loaded = {}
    for f in files:
        j = client.post("/api/sample/load", json={"file": f}).json()
        assert j.get("ok") is True, (f, j)
        stem = f.rsplit(".", 1)[0]
        # the returned name matches the file that was loaded...
        assert j["name"] == stem, (f, j["name"])
        instr = j["instrument"]
        assert required <= set(instr), (f, sorted(instr))
        assert isinstance(instr["range_um"], (int, float)) and instr["range_um"] > 0
        # ...and /api/state now reports the same active sample + instrument
        st = client.get("/api/state").json()
        assert st["sample"] == stem, (f, st["sample"])
        assert st["instrument"]["model"] == instr["model"]
        loaded[f] = instr

    models = {f: i["model"] for f, i in loaded.items()}
    ranges = {f: i["range_um"] for f, i in loaded.items()}
    distinct_models = set(models.values())
    if len(distinct_models) >= 2:
        # BEHAVIOUR: the .ibw's own metadata selects the microscope. At least
        # one pair of samples must disagree on model AND on scanner range.
        assert len(set(ranges.values())) >= 2, \
            f"different models but identical scanner ranges: {ranges}"
        # pin a concrete disagreeing pair so a regression that flattens every
        # sample to one model is caught, not just counted
        by_model = {}
        for f, m in models.items():
            by_model.setdefault(m, f)
        picked = list(by_model.values())
        a, b = loaded[picked[0]], loaded[picked[1]]
        assert a["model"] != b["model"]
        assert a["range_um"] != b["range_um"], \
            f"{picked[0]} and {picked[1]} share range {a['range_um']}"
    else:
        # All samples are the same microscope model. We still proved the active
        # sample NAME changes per load and every instrument is well-formed
        # above; there is simply no model/range contrast to assert here.
        assert len(loaded) >= 1


# ------------------------------------------- hardware backend: scan as a pass --
# On the REAL backend the scan endpoint runs the chained scan over the bridge and
# projects the capture onto the SAME {"type":"line"} stream (no HeightImage0
# harvest). Injected here with a HardwareARC over the fake Igor -- no instrument.

def test_hardware_backend_scan_streams_projected_lines(client, tmp_path):
    from afm_litho.hardware.hardware_arc import HardwareARC
    from afm_litho.hardware.igor_stub import FakeIgorBridge

    hub = _tmp_store(tmp_path)
    stub = FakeIgorBridge({}, done_delay=0.0)
    # Live crosspoint reads contact, so an unknown tracked mode resolves to the
    # scan's contact mode without a switch.
    stub.crosspoint = {"InFast": "FilterOut"}
    port = stub.serve_in_thread()
    hub.host.hardware_arc = HardwareARC.connect("127.0.0.1", port)
    hub.host.hw_cal = {"x_lvdt_sens": 7.46e-6, "x_lvdt_offset": -0.63,
                  "y_lvdt_sens": 7.04e-6, "y_lvdt_offset": 0.07,
                  "z_lvdt_sens": 5e-6, "z_lvdt_offset": 0.0}
    try:
        with client.websocket_connect("/ws") as ws:
            _ready_ws(ws)
            r = client.post("/api/scan/start", json={
                "size_um": 5.0, "pixels": 16, "lines": 4, "line_rate_hz": 200.0})
            assert r.json().get("backend") == "hardware", r.text
            rows = 0
            end = time.monotonic() + 8.0
            while time.monotonic() < end:
                m = ws.receive_json()
                if m.get("type") == "line":
                    rows += 1
                    assert len(m["height"]) == 16     # projected onto the raster
                elif m.get("type") == "scan_done":
                    break
            assert rows == 4                          # every line streamed
            # the RAW measured capture is persisted for the recorded session,
            # under its date-time STEM (service/stems.py), in scans/npz/ next
            # to (but not confused with) the display frame cache
            # (<stem>.height.npz) -- async, just after scan_done, co-located
            # with the temp store.
            import numpy as np
            raw_dir = tmp_path / "scans" / "npz"

            def _raw_files():
                if not raw_dir.is_dir():
                    return []
                return [p for p in raw_dir.glob("*.npz")
                        if not p.name.endswith(".height.npz")]

            assert _wait(_raw_files)
            with np.load(_raw_files()[0]) as z:
                cap = {k: z[k] for k in z.files}
            # trace_retrace defaults on, so each line captures BOTH sweeps
            # (forward + reverse) -> 2 * pixels per line in the raw capture,
            # even though the streamed rows are split back to `pixels` wide.
            assert "ZSensor" in cap and len(cap["ZSensor"]) == 2 * 16 * 4
    finally:
        try:
            hub.host.hardware_arc.close()
        except Exception:  # noqa: S110 - teardown must not mask the test result
            pass
        hub.host.hardware_arc = None                       # restore simulation
        stub.stop()
