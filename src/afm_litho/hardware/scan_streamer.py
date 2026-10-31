"""Stream a real scan capture out to the service's SOCKIT receiver -- with NO
Igor and NO instrument.

The service already listens for Igor's scan push (hardware/sockit_receiver.py):
Igor connects as a TCP client and sends one text-framed line per completed row,

    "L <row> <npix> v;v;...;v\\n"     (v = height values, in meters)

and the receiver fans each line to the browser's renderer. `scripts/stream_ibw.py`
produces those frames the REAL way -- by driving Igor over COM to LoadWave +
AFM_StreamStart -- which needs a live Igor.

This module produces the exact same frames straight from Python: it reads a real
`.ibw` capture with io.read_ibw and pushes it to the receiver over a plain TCP
socket. Same wire bytes, same `%.9g;` framing as Igor's AFM_SendRow, so the
receiver and the browser can't tell it from real Igor. That makes it a
hardware-free bring-up tool for the receive + render path (and gives that path
its first real test coverage).
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .. import clock
from ..io import read_ibw
from .sockit_receiver import DEFAULT_PORT


def frame_row(row: int, values) -> str:
    """One scan row as the exact frame Igor's AFM_SendRow sends: the "L" tag, the
    row index, the pixel count, then each value `%.9g` with a TRAILING ';', and a
    newline. The trailing ';' is why the receiver filters empty split tokens."""
    body = "".join(f"{float(v):.9g};" for v in values)
    return f"L {row} {len(values)} {body}\n"


def iter_frames(data: np.ndarray) -> Iterator[str]:
    """Yield one frame per row of a 2D image (rows x cols), top row first."""
    if data.ndim != 2:
        raise ValueError(f"expected a 2D image, got {data.ndim}D")
    for row in range(data.shape[0]):
        yield frame_row(row, data[row])


def _to_2d(wave, channel: str | None) -> tuple[np.ndarray, str]:
    """Pick a 2D image out of an IBWave. A scan is often stacked channels
    (rows x cols x nchan); select by name substring (case-insensitive) or numeric
    index, defaulting to Height if present, else the first channel."""
    d = np.asarray(wave.data, dtype=np.float64)
    if d.ndim == 2:
        return d, wave.name
    if d.ndim == 3:
        names = list(getattr(wave, "channel_names", []) or [])
        idx = 0
        if channel:
            hits = [i for i, n in enumerate(names) if channel.lower() in n.lower()]
            if hits:
                idx = hits[0]
            elif channel.isdigit():
                idx = int(channel)
        elif names:
            hits = [i for i, n in enumerate(names) if "height" in n.lower()]
            if hits:
                idx = hits[0]
        label = names[idx] if idx < len(names) else f"ch{idx}"
        return d[:, :, idx], label
    raise ValueError(f"expected a 2D or 3D image wave, got {d.ndim}D")


def stream_array(data: np.ndarray, *, host: str = "127.0.0.1",
                 port: int = DEFAULT_PORT, rate: float = 0.0,
                 log: Callable[[str], None] | None = None) -> int:
    """Push a 2D image (rows x cols, meters) to the receiver as `L` frames, one
    per row. `rate` seconds between rows paces it like a live scan (0 = as fast
    as possible). Returns the number of rows sent."""
    frames = list(iter_frames(data))
    with socket.create_connection((host, port), timeout=5.0) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for i, frame in enumerate(frames):
            sock.sendall(frame.encode("latin1"))
            if log:
                log(f"row {i + 1}/{len(frames)}")
            if rate > 0 and i + 1 < len(frames):
                clock.sleep(rate)
    return len(frames)


def stream_ibw_file(path: str | Path, *, channel: str | None = None,
                    host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                    rate: float = 0.0,
                    log: Callable[[str], None] | None = None) -> int:
    """Read a real `.ibw` capture and stream one of its channels to the receiver.
    Returns the number of rows sent."""
    wave = read_ibw(path)
    data, label = _to_2d(wave, channel)
    if log:
        log(f"{Path(path).name}: channel {label!r}, {data.shape[0]}x{data.shape[1]}")
    return stream_array(data, host=host, port=port, rate=rate, log=log)
