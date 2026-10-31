"""The state stream: the one place events reach the browser and the journal.

Events go to every open /ws page and, except the high-rate ones, to the journal (src "stream").
Delivery is unchanged on purpose: each send is awaited with no per-client timeout, and a
client whose send raises is dropped. FLEX does not subscribe; it polls.
"""

from __future__ import annotations

import json

from .. import journal

TELEMETRY_PERIOD_S = 0.25   # how long /ws waits for a client message before the next frame

# Position and progress at 4-10 Hz; telemetry, not events.
NOT_JOURNALLED = frozenset({"telemetry", "tip_pos", "litho_progress"})


def _journal_event(msg: dict) -> None:
    """Never raises and never delays the send that follows."""
    try:
        kind = msg.get("type")
        if kind not in NOT_JOURNALLED:
            journal.running().note("stream", str(kind), msg=msg)
    except Exception:  # noqa: BLE001,S110 - the journal must not stop an event reaching pages
        pass


async def publish(hub, msg: dict) -> None:
    """Send one event to every client, then forget the clients whose send failed."""
    _journal_event(msg)
    dead = []
    data = json.dumps(msg)
    for ws in hub.clients:
        try:
            await ws.send_text(data)
        except Exception:  # noqa: BLE001 - a dead socket is dropped, not fatal
            dead.append(ws)
    for ws in dead:
        hub.clients.discard(ws)


def subscribe(hub, websocket) -> None:
    hub.clients.add(websocket)


def unsubscribe(hub, websocket) -> None:
    hub.clients.discard(websocket)


def telemetry_frame(hub) -> dict:
    """One telemetry frame, built per connection per tick. Not journalled."""
    from .. import board
    return board.ws_frame(hub)
