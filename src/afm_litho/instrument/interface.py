"""The instrument: the one interface the program drives, whichever plugin is behind it."""

from __future__ import annotations

from typing import Protocol

from ..run.model import Platform
from .commands import (EngageAt, MoveTo, ParkTip, ReadChannels, Result, ScanRaster,
                       WithdrawTip, WriteObject)
from .reports import StatusListener, TelemetryListener


class Instrument(Protocol):
    """An AFM behind messages: commands in, results back; telemetry and status pushed out.

    Commands are sent one at a time; each call returns its Result when the instrument has
    finished, refused or aborted it. Two exceptions: `park` may be called while another
    command is running, and `abort` may be called from any thread at any time. Telemetry
    is pushed on the plugin's own thread, status on the thread running the command (or the
    one calling abort or park); listeners must return quickly. A plugin keeps no
    reference to the program, so it can later run in its own process without the program
    changing.
    """

    platform: Platform

    def listen(self, on_telemetry: TelemetryListener, on_status: StatusListener) -> None:
        """Set the one subscriber; telemetry is pushed from then on until close()."""
        ...

    def abort(self, reason: str) -> None:
        """Stop now: callable from any thread, never waits, harmless to call twice.

        The command in flight returns `aborted` with the reason. The Asylum plugin calls
        today's abort and park code unchanged.
        """
        ...

    def move(self, command: MoveTo) -> Result: ...

    def engage(self, command: EngageAt) -> Result: ...

    def withdraw(self, command: WithdrawTip) -> Result: ...

    def read(self, command: ReadChannels) -> Result: ...

    def park(self, command: ParkTip) -> Result: ...

    def write(self, command: WriteObject) -> Result: ...

    def scan(self, command: ScanRaster) -> Result: ...

    def close(self) -> None:
        """Stop pushing and let go of the instrument; it is left as it is."""
        ...
