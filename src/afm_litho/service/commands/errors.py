"""The error vocabulary the service layer raises, and the ONE place the JSON-RPC
codes are written down.

Truth document: docs/FLEX_INTEGRATION_PLAN.md Sec C (extract `service/commands/`,
one coroutine per verb, "each raising a typed `CommandError(code, message, data)`
and returning plain dicts; each router body becomes a thin adapter over the same
coroutine").

WHY THE CODES LIVE DOWN HERE
----------------------------
The layer that REFUSES a command is the layer that knows what kind of refusal it is,
so the vocabulary is defined beside the refusals and imported by the wire
(`flex_dispatch`) rather than the other way round. A second copy of these integers in
the dispatcher is exactly how a `-32010` turns into a `-32011` in one file and
nowhere else.

WHY `CommandError` CARRIES AN HTTP BODY
---------------------------------------
The cockpit's endpoints answer HTTP 200 with `{"error": ..., "reason": ...}` bodies,
and a hundred-odd tests assert those bodies verbatim -- they are the regression guard
that the extraction did not change what the panel sees. So a refusal carries BOTH
shapes: `code`/`message`/`data` for the wire, and `http` for the router adapter, set
at the raise site where the legacy body is known. A refusal that omits `http` gets a
faithful default (`{"error": message}` plus whatever `data` carried), so a NEW
refusal is never silently shapeless.
"""

from __future__ import annotations

from typing import Any

# ---- JSON-RPC 2.0 standard codes -------------------------------------------------
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32000

# ---- this provider's own codes ---------------------------------------------------
BACKEND_MISMATCH = -32010     # plan Sec C.2: configured hardware, backend not live
BUSY = -32020                 # plan Sec F.1: a start verb arrived while busy
# The STATE refusal. Distinct from `busy` on purpose: `busy` means "something is
# running, poll and retry", while this means "the instrument is in a state in which
# this verb is unsafe, and no amount of waiting changes that by itself" -- a mode
# switch with the tip on the surface is the case it exists for, and the fix is a
# withdraw, not a retry. A script that retried a `busy` forever would spin here.
REFUSED = -32021


class CommandError(Exception):
    """A typed refusal from the service layer.

    `code` is the JSON-RPC code the wire renders; `http` is the legacy HTTP-200 body
    the router adapter returns, so the GUI sees exactly what it always saw.
    """

    def __init__(self, code: int, message: str, data: Any = None,
                 *, http: dict | None = None):
        super().__init__(message)
        self.code = int(code)
        self.message = str(message)
        self.data = data
        self.http = dict(http) if isinstance(http, dict) else None

    def http_body(self) -> dict:
        """The body the HTTP adapter returns. Explicit when the raise site set one;
        otherwise `{"error": message}` plus the JSON-safe half of `data`."""
        if self.http is not None:
            return dict(self.http)
        body = {"error": self.message}
        if isinstance(self.data, dict):
            for key, value in self.data.items():
                body.setdefault(key, value)
        return body

    def __repr__(self) -> str:              # pragma: no cover - debugging aid
        return f"CommandError({self.code}, {self.message!r})"
