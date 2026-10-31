"""The keeper: the small process that starts the service and keeps it running.

`python -m afm_litho` runs it. It uses only the standard library (plus the clock
and the site chooser, which are too), so it stays up when the service's own
dependencies are what broke, and it never talks to Igor: only the service does.
tests/test_keeper_stdlib_only.py holds it to that.
"""
