"""Fixtures for the wire v3 conformance suite."""

from __future__ import annotations

import pytest

from . import target as T
from .reference_igor import ReferenceIgor


def _targets():
    ids = ["reference"]
    if T.fake_available()[0]:
        ids.append("fake")
    if T.env_factory() is not None:
        ids.append("env")
    return ids


@pytest.fixture(params=_targets())
def igor(request):
    """A fresh target handle. Skips when the target lacks a capability the test needs."""
    from afm_litho import clock
    with clock.using(clock.TESTING):
        if request.param == "reference":
            handle = ReferenceIgor()
        elif request.param == "fake":
            handle = T.FakeIgorHandle()
        else:
            handle = T.env_factory()()
        try:
            yield handle
        finally:
            if hasattr(handle, "close"):
                handle.close()


@pytest.fixture
def host(igor):
    """A raw session that has said HELLO (so it is the host)."""
    s = T.LineSession(igor.port)
    reply = s.hello()
    assert reply.ok, reply.raw
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def session(igor):
    """A raw session that has not said HELLO."""
    s = T.LineSession(igor.port)
    try:
        yield s
    finally:
        s.close()
