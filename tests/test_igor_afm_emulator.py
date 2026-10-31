"""Test the Igor scan-stream PASS-THROUGH realistically -- the LineCount polling,
direction handling, and frame boundaries -- not just the receiver.

IgorAFMEmulator plays Asylum's scan side (image wave + LineCount + ScanLines +
ScanDown + ScanBusy). AFMStreamTick mirrors AFM_StreamTick, consuming ONLY those
real-Igor-available signals and pushing S/L/D frames. Driving them against the
REAL sockit_receiver exercises the whole Igor-side logic end to end.
"""

from __future__ import annotations

import asyncio

import numpy as np

from afm_litho.hardware import sockit_receiver
from afm_litho.hardware.igor_afm_emulator import AFMStreamTick, IgorAFMEmulator


class _RecordingHub:
    def __init__(self, scan_params=None):
        self.scan_params = scan_params or {}
        self.messages: list[dict] = []

    async def broadcast(self, msg):
        self.messages.append(msg)

    async def log(self, msg):
        pass


def _ramp(rows, cols):
    # row r, col c -> value r*cols + c (nm), so height_nm round-trips to the index
    return np.arange(rows * cols, dtype=np.float64).reshape(rows, cols) * 1e-9


def _run_against_receiver(emu, **stream_kw):
    """Run emu.stream() against the real sockit_receiver server; return
    (tick, recorded_messages)."""
    hub = _RecordingHub({"size_um": 30.0})

    async def scenario():
        server = await asyncio.start_server(
            lambda r, w: sockit_receiver._handle(hub, r, w), "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        loop = asyncio.get_event_loop()
        tick = await loop.run_in_executor(
            None, lambda: emu.stream(host="127.0.0.1", port=port, **stream_kw))
        await asyncio.sleep(0.25)
        server.close()
        await server.wait_closed()
        return tick

    tick = asyncio.run(scenario())
    return tick, hub.messages


# ----------------------------------------------------------------- logic -----

def test_linecount_to_completed_count_both_directions():
    up = IgorAFMEmulator(np.zeros((4, 2)))
    up.start_frame()
    up.complete_next_line(); up.complete_next_line()
    assert up.line_count == 2                       # up counts 0 -> ScanLines
    assert AFMStreamTick(up, lambda s: None)._completed() == 2

    dn = IgorAFMEmulator(np.zeros((4, 2)), scan_down=True)
    dn.start_frame()
    assert dn.line_count == 3                        # ScanLines-1 at the top
    dn.complete_next_line(); dn.complete_next_line()
    assert dn.line_count == 1                        # down counts toward -1
    assert AFMStreamTick(dn, lambda s: None)._completed() == 2


def test_frame_marker_sequence_and_abort():
    emu = IgorAFMEmulator(np.ones((5, 3)))
    sent: list[str] = []
    tick = AFMStreamTick(emu, sent.append)

    emu.start_frame(); tick.poll()                   # S
    emu.complete_next_line(); tick.poll()            # L row 0
    emu.complete_next_line(); tick.poll()            # L row 1
    emu.abort(); tick.poll()                         # D (aborted after 2 lines)

    tags = [s.split(" ", 1)[0] for s in sent]
    assert tags == ["S", "L", "L", "D"]
    assert tick.lines_sent == 2                       # partial frame
    assert tick.frames_done == 1


# ------------------------------------------------------ end to end -----------

def test_up_scan_frame_through_the_real_receiver():
    emu = IgorAFMEmulator(_ramp(4, 3))
    tick, msgs = _run_against_receiver(emu, frames=1)

    assert msgs[0] == {"type": "scan_reset", "pixels": 3, "rows": 4,
                       "size_um": 30.0}
    lines = [m for m in msgs if m["type"] == "line"]
    assert [m["row"] for m in lines] == [0, 1, 2, 3]          # top-to-bottom
    assert lines[0]["height"] == [0.0, 1.0, 2.0]              # ramp round-trips
    assert lines[3]["height"] == [9.0, 10.0, 11.0]
    assert msgs[-1] == {"type": "scan_done"}
    assert (tick.frames_started, tick.frames_done, tick.lines_sent) == (1, 1, 4)


def test_continuous_two_frames_reset_each_time():
    emu = IgorAFMEmulator(_ramp(3, 2))
    tick, msgs = _run_against_receiver(emu, frames=2)

    resets = [m for m in msgs if m["type"] == "scan_reset"]
    dones = [m for m in msgs if m["type"] == "scan_done"]
    lines = [m for m in msgs if m["type"] == "line"]
    assert len(resets) == 2                          # a fresh reset per frame
    assert len(dones) == 2
    assert len(lines) == 6                           # 3 rows x 2 frames
    assert (tick.frames_started, tick.frames_done) == (2, 2)


def test_down_scan_streams_top_row_last():
    emu = IgorAFMEmulator(_ramp(4, 2), scan_down=True)
    tick, msgs = _run_against_receiver(emu, frames=1)

    lines = [m for m in msgs if m["type"] == "line"]
    assert [m["row"] for m in lines] == [3, 2, 1, 0]         # down: high-to-low
    # row 3 of the ramp is [6,7]; it must arrive FIRST and carry its own data
    assert lines[0]["row"] == 3 and lines[0]["height"] == [6.0, 7.0]
    assert lines[-1]["row"] == 0 and lines[-1]["height"] == [0.0, 1.0]
