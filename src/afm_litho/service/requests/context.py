"""The hub a request runs against, handed to row handlers without widening their signature.

A row handler takes the request only. FLEX tests run a provider on their own hub, so the
door sets the hub for the duration of one request; the browser door sets the process hub.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

_hub: ContextVar = ContextVar("request_hub")
_ctx: ContextVar = ContextVar("request_ctx", default=None)


@contextmanager
def using_hub(hub, ctx=None):
    """`ctx` is the FLEX provider's per-request context (how a handler reaches the service
    loop); the browser has none."""
    token = _hub.set(hub)
    ctx_token = _ctx.set(ctx)
    try:
        yield
    finally:
        _ctx.reset(ctx_token)
        _hub.reset(token)


def current_hub():
    return _hub.get()


def current_ctx():
    return _ctx.get()
