"""The hardware-free scan streamer: it must produce the EXACT frames Igor's
AFM_SendRow sends, so the real receiver parser and the browser renderer accept
them unchanged. These tests lock the framing and, by driving the streamer's
output through the actual receiver, give sockit_receiver its first coverage."""

from __future__ import annotations

import asyncio
import socket
import threading
from pathlib import Path

import numpy as np
import pytest

from afm_litho.hardware import sockit_receiver
from afm_litho.hardware.scan_streamer import (
    frame_row, iter_frames, stream_array, stream_ibw_file,
)

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "SA40656B0002.ibw"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample .ibw not present")


class _RecordingHub:
    """Stand-in for service.Hub: records what the receiver broadcasts."""

    def __init__(self, scan_params=None):
        self.scan_params = scan_params or {}
        self.messages: list[dict] = []

    async def broadcast(self, msg):
        self.messages.append(msg)

    async def log(self, msg):
        pass


def test_frame_row_matches_igor_framing():
    # "L <row> <npix> v;v;...;\n" with %.9g values and the trailing ';'
    assert frame_row(5, [1e-9, 2.5e-9]) == "L 5 2 1e-09;2.5e-09;\n"


def test_iter_frames_covers_every_row():
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    frames = list(iter_frames(data))
    assert len(frames) == 3
    assert frames[0].startswith("L 0 2 ")
    assert frames[2].startswith("L 2 2 ")


def test_frames_parse_in_the_real_receiver():
    """Producer/consumer parity: the streamer's frames, fed through the actual
    receiver parser, yield a scan_reset then one line per row (meters -> nm)."""
    hub = _RecordingHub({"lines": 2, "size_um": 30.0})
    rows = [[1e-9, 2e-9, 3e-9], [4e-9, 5e-9, 6e-9]]

    async def go():
        st = {"first": True, "rows": None}
        for r, vals in enumerate(rows):
            raw = frame_row(r, vals).rstrip("\n").encode("latin1")
            await sockit_receiver._broadcast_line(hub, raw, st)

    asyncio.run(go())

    assert hub.messages[0]["type"] == "scan_reset"
    assert hub.messages[0]["pixels"] == 3
    lines = [m for m in hub.messages if m["type"] == "line"]
    assert len(lines) == 2
    assert lines[0]["row"] == 0 and lines[0]["pixels"] == 3
    assert lines[0]["height"] == [1.0, 2.0, 3.0]        # 1e-9 m -> 1.0 nm
    assert lines[1]["height"] == [4.0, 5.0, 6.0]


def _collect_on_socket() -> tuple[int, threading.Thread, dict]:
    """A one-shot TCP server that accepts one connection and reads all bytes."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    out: dict = {}

    def run():
        conn, _ = srv.accept()
        buf = b""
        while True:
            d = conn.recv(4096)
            if not d:
                break
            buf += d
        out["buf"] = buf
        conn.close()
        srv.close()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return port, t, out


def test_stream_array_writes_the_frames_over_tcp():
    port, t, out = _collect_on_socket()
    data = np.array([[1e-9, 2e-9], [3e-9, 4e-9]])
    n = stream_array(data, host="127.0.0.1", port=port, rate=0.0)
    t.join(timeout=5)

    assert n == 2
    frames = [f for f in out["buf"].decode("latin1").split("\n") if f]
    assert frames == ["L 0 2 1e-09;2e-09;", "L 1 2 3e-09;4e-09;"]


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample .ibw not present")
def test_stream_ibw_file_streams_a_real_capture():
    port, t, out = _collect_on_socket()
    n = stream_ibw_file(SAMPLE, host="127.0.0.1", port=port, rate=0.0)
    t.join(timeout=5)

    frames = [f for f in out["buf"].decode("latin1").split("\n") if f]
    assert n == len(frames) > 1                      # a real multi-row frame
    first = frames[0].split(" ", 3)
    assert first[0] == "L" and first[1] == "0"       # row 0 leads
    npix = int(first[2])
    assert len([v for v in first[3].split(";") if v]) == npix


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample .ibw not present")
def test_real_ibw_flows_through_the_actual_receiver():
    """End to end: stream a real .ibw through sockit_receiver's OWN server
    (_handle -> _broadcast_line), exercising the real line-buffering. Every
    streamed row must surface as a broadcast 'line', preceded by scan_reset --
    the same path real Igor pixels take, minus Igor."""
    hub = _RecordingHub({"size_um": 30.0})

    async def scenario():
        server = await asyncio.start_server(
            lambda r, w: sockit_receiver._handle(hub, r, w), "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        loop = asyncio.get_event_loop()
        n = await loop.run_in_executor(
            None, lambda: stream_ibw_file(SAMPLE, host="127.0.0.1", port=port))
        await asyncio.sleep(0.25)            # let the handler drain the buffer
        server.close()
        await server.wait_closed()
        return n

    n = asyncio.run(scenario())

    assert hub.messages[0]["type"] == "scan_reset"
    lines = [m for m in hub.messages if m["type"] == "line"]
    assert len(lines) == n > 1               # every streamed row broadcast
    assert lines[0]["row"] == 0
    assert len(lines[0]["height"]) == lines[0]["pixels"]
