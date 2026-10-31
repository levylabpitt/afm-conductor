"""HostApi v1: the one list of what the base may ask the host to do.

Every verb is a thin call to the function that already ran it. The reply is that function's
reply and the errors are its `CommandError` codes. The base builds a `HostApi(hub)` and calls
a verb by name; nothing here adds a check, a lock or a wait.

Words used for the tip (the instrument's terms):
  pass      one run of the instrument: a scan, a write, an approach.
  set-down  `Engage`: the tip goes down to the surface and the Z feedback takes hold.
  lift      `Withdraw` back-off: the first step up off the surface.
  withdraw  `Withdraw` verified: the tip is lifted and Output.Z is read back near 0 V.
  park      `AFM_SafePark`: the full make-safe (tip off, XY still, tip voltage zero).
`stop_scan` and `withdraw` are not the emergency stop; `park` is.

The stop verbs (`park`, `abort_write`, `withdraw`, `stop_scan`) are plain methods that return
the awaitable today's function returns; the caller awaits it as it awaited that function, so
a stop goes through no extra coroutine, lock or wait on its way to the instrument.
`commander_lapsed` is a plain method too, for the same reason: it reads the tip's state in
the call and returns the awaitable that acts. `snapshot` is named in `VERBS` so the surface
is complete; it is not a method yet.
"""
from __future__ import annotations

HOST_API_VERSION = 1

# (verb, class). Any change to this list bumps HOST_API_VERSION.
VERBS: tuple[tuple[str, str], ...] = (
    ("start_scan", "command"),
    ("start_write", "command"),
    ("set_continuous", "command"),
    ("set_mode", "command"),
    ("set_zgain", "command"),
    ("approach", "command"),
    ("start_approach", "command"),
    ("clear_fault", "command"),
    ("set_override", "command"),
    ("park", "stop"),
    ("abort_write", "stop"),
    ("withdraw", "stop"),
    ("stop_scan", "stop"),
    ("commander_lapsed", "fact"),
    ("snapshot", "read"),
)

COMMAND_VERBS = tuple(name for name, kind in VERBS if kind == "command")
STOP_VERBS = tuple(name for name, kind in VERBS if kind == "stop")


class HostApi:
    """The command and stop verbs, in-process. `hub` is the process hub the host runs on."""

    def __init__(self, hub) -> None:
        self._hub = hub

    async def start_scan(self, cfg: dict):
        """Arm a raster scan and hand it to a pump. Reply: start dict with `run_id`."""
        from . import scan_cmds
        return await scan_cmds.start_scan(self._hub, cfg)

    async def start_write(self, body: dict):
        """Start a litho write. Reply: dict with `run_id`, `est_s`, `objects`."""
        from . import litho_cmds
        return await litho_cmds.start_write(self._hub, body)

    async def set_continuous(self, body: dict):
        """Arm or stop the auto-repeat imaging loop."""
        from . import continuous_cmds
        return await continuous_cmds.set_continuous_wire(self._hub, body)

    async def set_mode(self, body: dict):
        """Switch contact <-> AC. Refused while the tip is engaged, scanning or writing."""
        from . import scan_cmds
        return await scan_cmds.set_mode(self._hub, body)

    async def set_zgain(self, body: dict):
        """Re-arm the running Z loop with new P/I gains, without withdrawing."""
        from . import scan_cmds
        return await scan_cmds.set_zgain(self._hub, body)

    async def approach(self, body: dict):
        """Fine set-down, blocking. A failed set-down parks."""
        from . import scan_cmds
        return await scan_cmds.approach(self._hub, body)

    async def start_approach(self, body: dict):
        """Start a set-down in the background; reply `{run_id, est_s}` at once."""
        from . import scan_cmds
        return await scan_cmds.start_approach(self._hub, body)

    def clear_fault(self, body: dict | None = None):
        """Clear the latched fault when `confirm` is true. Plain function: callers run it in
        a worker thread."""
        from . import instrument_cmds
        return instrument_cmds.clear_fault(self._hub, body)

    def set_override(self, body: dict | None = None):
        """Override one instrument-model field. Plain function: callers run it in a worker
        thread, as the route always did."""
        from . import instrument_cmds
        return instrument_cmds.set_override(self._hub, body)

    # The fact verb.

    def commander_lapsed(self, holder: str, withdraw_on_deadman: bool):
        """The commander went silent and the base has released its token. Reads the tip's
        state now and returns the awaitable that acts on plan Sec E.1's table (safe park
        while a pass runs; withdraw or warn while engaged; else nothing). Reply:
        `{action, state, holder}`, plus `safe_park` or `not_connected`."""
        from . import deadman
        return deadman.commander_lapsed(self._hub, holder, withdraw_on_deadman)

    # The stop verbs. Never refused; each returns today's awaitable without awaiting it.

    def park(self, why: str, stop_id: str | None = None):
        """The full park (`request_park`, style PARK_FULL). `stop_id` pairs a door's park
        with the keeper's copy of the same Stop. Reply: today's park dict; never raises."""
        from . import safe_park
        return safe_park.request_park(self._hub, why, stop_id=stop_id)

    def abort_write(self):
        """Stop the write, then the full park. Reply: `{ok, backend, safe_park}`."""
        from . import litho_cmds
        return litho_cmds.abort_write(self._hub)

    def withdraw(self, body: dict | None = None):
        """The plain withdraw, from any state. Reply: today's snapshot (`z_v`,
        `verify_timeout`, ...). Through `request_park`, the one home for every park
        trigger; PARK_WITHDRAW returns `scan_cmds.withdraw(hub, body)` un-awaited."""
        from . import safe_park
        return safe_park.request_park(self._hub, "withdraw", safe_park.PARK_WITHDRAW,
                                      body=body)

    def stop_scan(self, body: dict | None = None):
        """The operator Stop: end the scan and retract (deferred to the pass on hardware).
        A running pump reads the process hub, so callers build this HostApi on it. Reply:
        `{ok, withdraw}`."""
        from . import scan_cmds
        return scan_cmds.stop_scan(self._hub, body)
