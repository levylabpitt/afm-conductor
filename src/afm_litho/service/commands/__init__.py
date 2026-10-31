"""`service/commands/` -- THE service layer (docs/FLEX_INTEGRATION_PLAN.md Sec C).

One module per verb family, one coroutine (or plain function, where the existing code
was synchronous) per verb. Each takes a hub and plain dicts, raises a typed
`CommandError(code, message, data)`, and returns plain dicts. Two doors sit on top of
it and neither owns a guard of its own:

    FLEX script  -- ZMQ 29180 -->  service/flex_dispatch.py  --\\
                                                                 >-- service/commands/
    cockpit      -- HTTP POST -->  service/routers/*.py      --/

The plan's reasoning, compressed: dispatching FLEX calls through the routers would
mean a JSON-RPC caller inherits HTTP-shaped `{"error": ...}` bodies with no code, and
dispatching through internal HTTP would make the asyncio loop a hard dependency of
every wire call. Extracting the middle instead gives the cockpit and a script ONE
validation path, ONE guard set and ONE state machine -- which is the whole of goal 3,
delivered without changing the GUI's transport (plan Sec D).

Extraction is per verb, interleaved into the stage that needs it, never a big-bang
refactor. Today: the scan-control verbs (slice 07) and `setContinuous`. Next: the
write verbs (`loadPattern`, `startWrite`), whose AC-guard fail-open ordering must be
preserved VERBATIM when it moves (plan Sec C, Sec J risk 11).
"""

from __future__ import annotations

from .errors import (
    BACKEND_MISMATCH,
    BUSY,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    REFUSED,
    CommandError,
)

__all__ = ["BACKEND_MISMATCH", "BUSY", "INTERNAL_ERROR", "INVALID_PARAMS",
           "METHOD_NOT_FOUND", "REFUSED", "CommandError"]
