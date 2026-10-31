"""Receive SOCKIT-pushed scan lines from Igor and broadcast them to the browser.

The FastAPI service is the TCP SERVER; Igor (afm_sockit.ipf, AFM_StreamStart) is
the client and pushes one text-framed line per completed scan row:

    "L <row> <npix> v;v;...;v\\n"     (v = HeightImage0 values, in meters)

Each line becomes the exact {"type":"line", ...} message the browser's existing
renderer already consumes (app.js), so real Igor pixels light up the same canvas
the twin uses. Metadata the stream doesn't carry (rows, size_um, angle) comes
from hub.scan_params -- the latest values the panel pushed.

Runs as an asyncio task on the service's own event loop, so it can await
hub.broadcast directly (no thread hand-off). Started from app startup when
AFM_LITHO_IGOR_STREAM is set; harmless/absent otherwise.
"""

from __future__ import annotations

import asyncio
import contextlib

from . import channels

DEFAULT_PORT = 9193


async def _broadcast_line(hub, raw: bytes, st: dict) -> None:
    """Broadcast one pushed frame line. Understands three tags:
      S <rows> <cols>   frame start -> scan_reset (clears the previous frame)
      L <row> <npix> v; a scan line -> a 'line' render message
      D                 frame end   -> scan_done
    `st` is per-connection state: {"first": bool, "rows": int|None}. The first L
    of a stream also resets (back-compat with the plain streamer that sends no
    S), and the S-supplied row count is remembered for the frame's L lines."""
    parts = raw.decode("latin1").split(" ", 3)
    tag = parts[0] if parts else ""
    cfg = hub.scan_params or {}
    size_um = float(cfg.get("size_um", 30.0))
    angle = float(cfg.get("angle_deg", 0.0))

    # explicit frame start: reset the display and remember this frame's rows
    if tag == "S" and len(parts) >= 3:
        try:
            rows, npix = int(parts[1]), int(parts[2])
        except ValueError:
            return
        st["rows"] = rows
        st["first"] = False
        await hub.broadcast({"type": "scan_reset", "pixels": npix,
                             "rows": rows, "size_um": size_um})
        return

    # explicit frame end
    if tag == "D":
        await hub.broadcast({"type": "scan_done"})
        return

    if tag != "L" or len(parts) != 4:
        return
    try:
        row, npix = int(parts[1]), int(parts[2])
        vals = [float(x) for x in parts[3].split(";") if x != ""]
    except ValueError:
        return
    if len(vals) != npix or npix == 0:
        return

    rows = st.get("rows")
    if rows is None:
        rows = int(cfg.get("lines") or cfg.get("pixels") or npix)

    if st.get("first", True):
        st["first"] = False
        await hub.broadcast({"type": "scan_reset", "pixels": npix,
                             "rows": rows, "size_um": size_um})

    # HeightImage0 is in meters; the browser wants nanometers. We only stream one
    # channel today, so error/zsensor mirror height until we push more channels.
    height_nm = [round(v * 1e9, 3) for v in vals]
    await hub.broadcast({
        "type": "line",
        "row": row,
        "slow_idx": row,
        "rows": rows,
        "pixels": npix,
        "size_um": size_um,
        "direction": "up",
        "angle_deg": angle,
        "errName": channels.DEFLECTION,
        "height": height_nm,
        "retrace": height_nm,
        "error": [0.0] * npix,
        "zsensor": height_nm,
        "slow_um": (row / max(1, rows)) * size_um,
    })


async def _handle(hub, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    buf = b""
    st = {"first": True, "rows": None}   # per-connection: reset-on-first, frame rows
    try:
        await hub.log("Igor scan stream connected")
        while True:
            data = await reader.read(65536)
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                await _broadcast_line(hub, line, st)
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        with contextlib.suppress(Exception):
            writer.close()


async def run_receiver(hub, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
    """Serve forever, handing each Igor connection to _handle. Cancel the task to
    stop (app shutdown)."""
    server = await asyncio.start_server(
        lambda r, w: _handle(hub, r, w), host, port)
    await hub.log(f"Igor scan-stream receiver listening on {host}:{port}")
    async with server:
        await server.serve_forever()
