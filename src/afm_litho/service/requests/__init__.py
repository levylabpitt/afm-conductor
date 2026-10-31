"""One front door for every request: the request table, `dispatch`, and a thin adapter per
carrier. Nothing in this package knows which carrier a request came from; a carrier
difference lives in a row's intake (docs/migration/plan/phase-2-doors.md)."""
