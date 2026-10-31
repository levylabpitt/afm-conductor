"""A realistic emulation of the Igor + AFM scan-stream pass-through, so it can be
tested with no instrument -- and, unlike scan_streamer.py, without bypassing the
Igor logic.

Two pieces, mirroring the two halves of a real scan:

  * IgorAFMEmulator -- the AFM as Igor sees it. A 2D image wave that fills one
    line at a time, exposing exactly the globals ScanMaster.ipf does: LineCount,
    ScanLines, ScanDown, and ARSysScanBusy. LineCount resets per frame and moves
    UP for an up-scan (0 -> ScanLines) or DOWN for a down-scan (ScanLines-1 ->
    -1), per docs/SCAN_STREAMING.md.

  * AFMStreamTick -- a faithful mirror of igor/afm_sockit.ipf's AFM_StreamTick.
    On each tick it reads ONLY what real Igor has (LineCount + ScanLines +
    ScanDown + ScanBusy) and the image wave, works out which lines are newly
    complete, and pushes them as the exact wire frames AFM_SendRow sends. It also
    emits frame markers so continuous mode works: `S <rows> <cols>` at each frame
    start (on a LineCount reset) and `D` when a frame ends.

Because the tick consumes ONLY the real-Igor-available signals and produces the
real wire bytes, driving it against the real sockit_receiver tests the
pass-through LOGIC end to end (poll -> detect new lines -> frame boundary ->
finish/abort), not just the receiver. It is the executable reference the real
.ipf must match.
"""

from __future__ import annotations

import socket
from typing import Callable

import numpy as np

from .. import clock
from .sockit_receiver import DEFAULT_PORT


class IgorAFMEmulator:
    """The AFM scan side as Igor sees it: a 2D image wave filling line by line,
    with the same LineCount / ScanLines / ScanDown / ScanBusy semantics as
    ScanMaster.ipf. Drive it with complete_next_line(), or run the whole
    scan+stream against a receiver with stream()."""

    def __init__(self, frame, scan_down: bool = False):
        self.frame = np.asarray(frame, dtype=np.float64)
        if self.frame.ndim != 2:
            raise ValueError(f"frame must be 2D, got {self.frame.ndim}D")
        self.scan_lines, self.pixels = self.frame.shape
        self.scan_down = bool(scan_down)
        self.image = np.zeros_like(self.frame)    # the live <Channel>Image0 wave
        self.scan_busy = 0
        self.completed = 0                         # lines finished this frame
        self.line_count = self._reset_value()

    def _reset_value(self) -> int:
        # ResetLineCount(): ScanLines-1 for a down-scan, else 0 (ScanMaster:2029)
        return (self.scan_lines - 1) if self.scan_down else 0

    def start_frame(self) -> None:
        """Begin a fresh frame: clear the live wave and reset LineCount."""
        self.image[:] = 0.0
        self.completed = 0
        self.scan_busy = 1
        self.line_count = self._reset_value()

    def complete_next_line(self):
        """Finish one more scan line: copy it into the live image wave and move
        LineCount exactly as Asylum would. Returns the row index just completed,
        or None if the frame is already full."""
        if self.completed >= self.scan_lines:
            return None
        k = self.completed
        # up-scan acquires rows 0,1,..; down-scan acquires ScanLines-1,..,0
        row = (self.scan_lines - 1 - k) if self.scan_down else k
        self.image[row] = self.frame[row]
        self.completed += 1
        # up: LineCount counts 0->ScanLines; down: (ScanLines-1)->-1
        self.line_count = (self._reset_value() - k - 1) if self.scan_down else (k + 1)
        if self.completed >= self.scan_lines:
            self.scan_busy = 0                     # frame finished
        return row

    def abort(self) -> None:
        """Stop the scan partway (operator abort): not busy, frame unfinished."""
        self.scan_busy = 0

    def stream(self, *, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
               line_rate: float = 0.0, frames: int = 1,
               log: Callable[[str], None] | None = None) -> "AFMStreamTick":
        """Run a realistic scan: connect to the receiver (as real Igor does),
        then for each frame fill the wave line by line while an AFMStreamTick
        polls and pushes. `line_rate` seconds per line paces it like a live scan.
        Returns the tick (with its frame/line counters) for inspection."""
        sock = socket.create_connection((host, port), timeout=5.0)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def send(line: str) -> None:
            sock.sendall((line + "\n").encode("latin1"))

        tick = AFMStreamTick(self, send)
        try:
            for f in range(frames):
                self.start_frame()
                tick.poll()                        # emits S for this frame
                while self.completed < self.scan_lines:
                    self.complete_next_line()
                    if line_rate > 0:
                        clock.sleep(line_rate)
                    tick.poll()                    # push new lines (+ D at the end)
                if log:
                    log(f"frame {f + 1}/{frames}: {self.scan_lines} lines")
        finally:
            sock.close()
        return tick


class AFMStreamTick:
    """Mirror of AFM_StreamTick. Consumes ONLY the real-Igor-available signals
    (LineCount, ScanLines, ScanDown, ScanBusy) + the image wave, and pushes the
    exact S/L/D frames to the receiver. Direction-aware, and treats a LineCount
    reset as a new-frame boundary (docs/SCAN_STREAMING.md)."""

    def __init__(self, afm: IgorAFMEmulator, send: Callable[[str], None]):
        self.afm = afm
        self._send = send
        self._active = False
        self._sent = 0            # completed-count already streamed this frame
        self.frames_started = 0
        self.frames_done = 0
        self.lines_sent = 0

    def _completed(self) -> int:
        """Turn the raw LineCount into a direction-independent completed-count --
        the one conversion real Igor must also do (LineCount + ScanDown)."""
        a = self.afm
        return a.line_count if not a.scan_down else (a.scan_lines - 1 - a.line_count)

    def poll(self) -> None:
        """One poll of the background task: emit a frame start on reset, every
        newly-completed line, and a frame end on finish or abort."""
        a = self.afm
        completed = self._completed()

        # new frame: the AFM is busy and the completed-count has reset below what
        # we already streamed (LineCount snapped back to its per-direction start)
        if a.scan_busy and (not self._active or completed < self._sent):
            self._send(f"S {a.scan_lines} {a.pixels}")
            self._active = True
            self._sent = 0
            self.frames_started += 1

        # stream every newly-completed line, in acquisition order
        while self._active and self._sent < completed:
            k = self._sent
            row = (a.scan_lines - 1 - k) if a.scan_down else k
            vals = a.image[row]
            body = "".join(f"{v:.9g};" for v in vals)
            self._send(f"L {row} {len(vals)} {body}")
            self._sent += 1
            self.lines_sent += 1

        # frame ended: finished (all lines) or aborted (busy dropped early)
        if self._active and (completed >= a.scan_lines or not a.scan_busy):
            self._send("D")
            self._active = False
            self.frames_done += 1
