"""The brand-agnostic instrument interface: commands, results, telemetry, status, plugin choice.

Spec: docs/migration/plan/phase-4-interface.md. The Asylum plugin and the twin plugin both
implement `Instrument`; everything above it runs one path for both.
"""

from .commands import (ALIAS, VERBS, Citation, Command, EngageAt, MoveTo, Outcome, ParkTip,
                       ReadChannels, Reading, Recorded, Result, ScanRaster, Verb, WithdrawTip,
                       WriteObject)
from .interface import Instrument
from .plugins import PLUGINS, PluginChoice, PluginFactory, PluginName, PluginRegistry
from .reports import (ACTIVITIES, TELEMETRY_HZ_CHAINED, TELEMETRY_HZ_IDLE, Activity, Status,
                      StatusListener, TelemetryFrame, TelemetryListener, telemetry_hz)

__all__ = [
    "ACTIVITIES", "ALIAS", "PLUGINS", "TELEMETRY_HZ_CHAINED", "TELEMETRY_HZ_IDLE", "VERBS",
    "Activity", "Citation", "Command", "EngageAt", "Instrument", "MoveTo", "Outcome",
    "ParkTip", "PluginChoice", "PluginFactory", "PluginName", "PluginRegistry",
    "ReadChannels", "Reading", "Recorded", "Result", "ScanRaster", "Status", "StatusListener",
    "TelemetryFrame", "TelemetryListener", "Verb", "WithdrawTip", "WriteObject",
    "telemetry_hz",
]
