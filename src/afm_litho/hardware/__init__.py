"""Real-instrument I/O (Phase 4). Distinct from twin/, which simulates.

Nothing here is imported off-instrument by default: the modules connect to a
live Igor only when asked, so the twin-based dev flow and the test suite never
need pywin32 or a registered Igor.
"""
