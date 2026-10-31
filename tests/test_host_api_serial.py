"""H7.5e: the serializing HostApi works, covers every verb, and its exemptions only shrink."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import host_api_serial as serial
import pytest

from afm_litho.service.host.api import VERBS, HostApi
from afm_litho.service.host_state import HostState

ROOT = Path(__file__).resolve().parents[1]

# Tests that turn the round trip off because they push a stub object through a verb to check
# its identity, not wire data. Shrink-only.
EXEMPT_TESTS = {
    "tests/test_host_api.py::test_a_stop_verb_is_a_plain_method_returning_todays_awaitable",
}

# Verbs named in VERBS that have no method yet.
NO_METHOD_YET = {"snapshot"}


def test_every_verb_and_pass_ended_was_exercised_under_serialization():
    """The frozen shape log lists every verb that crossed the round trip in a full run."""
    seen = {r["verb"] for r in json.loads(serial.GOLDEN.read_text("utf-8"))}
    wanted = {name for name, _ in VERBS if name not in NO_METHOD_YET} | {"pass_ended"}
    assert not wanted - seen, f"never called under the JSON round trip: {sorted(wanted - seen)}"
    assert not any(hasattr(HostApi, v) for v in NO_METHOD_YET), "snapshot has a method now"


def test_the_wrapper_is_on_the_classes_in_this_run(request):
    if request.config.getoption("--no-serial-host-api"):
        pytest.skip("run with the serializing API off")
    assert hasattr(HostApi.start_scan, "__wrapped__")
    assert hasattr(HostState.pass_ended, "__wrapped__")


def test_the_exemption_list_only_shrinks():
    marked = set()
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        lines = path.read_text("utf-8").splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("@pytest.mark.host_api_unserialized"):
                j = i
                while not lines[j].startswith(("def ", "async def ")):
                    j += 1
                name = lines[j].split("def ", 1)[1].split("(")[0]
                marked.add(f"{path.relative_to(ROOT).as_posix()}::{name}")
    assert marked <= EXEMPT_TESTS, f"new exemption: {sorted(marked - EXEMPT_TESTS)}"
    assert EXEMPT_TESTS <= marked, f"stale exemption: {sorted(EXEMPT_TESTS - marked)}"


# --- the wrapper's self-test: a fake verb that does not survive a socket fails -----------------

class _Fake:
    def returns_tuple(self):
        return (1, 2)

    def returns_plain(self, body):
        return {"a": [1, 2.5, None, "x"], "n": float("nan")}

    async def awaits_tuple(self):
        return (1,)

    def stop_like(self):
        async def act():
            return {"ok": True}
        return act()

    def takes_object(self, x):
        return {"ok": True}


def test_a_tuple_reply_fails_the_round_trip():
    with pytest.raises(serial.NotJsonPlain, match="tuple -> list"):
        serial.wrap("fake", _Fake.returns_tuple)(_Fake())


def test_a_tuple_in_an_awaited_reply_fails():
    with pytest.raises(serial.NotJsonPlain):
        asyncio.run(serial.wrap("fake", _Fake.awaits_tuple)(_Fake()))


def test_a_non_json_argument_fails():
    with pytest.raises(serial.NotJsonPlain, match="not JSON-encodable"):
        serial.wrap("fake", _Fake.takes_object)(_Fake(), object())


def test_an_int_key_is_not_a_plain_key():
    with pytest.raises(serial.NotJsonPlain):
        serial.wrap("fake", _Fake.takes_object)(_Fake(), {1: "a"})


def test_a_plain_reply_passes_and_a_copy_comes_back():
    out = serial.wrap("fake", _Fake.returns_plain)(_Fake(), {"k": [1]})
    assert out["a"] == [1, 2.5, None, "x"]


def test_a_plain_verb_returning_an_awaitable_still_returns_it_at_once():
    out = serial.wrap("fake", _Fake.stop_like)(_Fake())
    assert asyncio.iscoroutine(out)
    assert asyncio.run(out) == {"ok": True}
