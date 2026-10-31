"""The host half of the service: the code that drives or guards the tip.

The scan pumps, the litho workers, the continuous supervisor and the telemetry tick, and
the safety code that acts on the tip: safe_park (the one park, the fail-safe watchdog,
the fault latch), engage, the stop port and the after-death make-safe. The
routers, FLEX and the WebSocket are the doors; they reach this package through
`commands/` (and a few module-level imports).
"""
