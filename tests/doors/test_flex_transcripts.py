"""FLEX door transcripts (slice 2.0a): the request and reply of every verb on both sockets,
plus every refusal, recorded once and compared byte for byte after masking (`masks.py`).

The fixtures in `tests/doors/flex/` are the door as it behaves today, quirks included. A later
slice that moves the door must leave them unchanged. To re-record after an intended change:
`AFM_LITHO_DOORS_REGEN=1 pytest tests/doors`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from _service import Service
from flex_scenario import COMMAND_VERBS, READ_VERBS, run
from masks import canonical

pytest.importorskip("zmq", reason="pyzmq is the optional [flex] extra")

FIXTURES = Path(__file__).resolve().parent / "flex"
REGEN = os.environ.get("AFM_LITHO_DOORS_REGEN") == "1"
GROUPS = ("reads_at_rest", "refusals_protocol", "refusals_read_socket", "control",
          "refusals_token_and_params", "scan", "write", "stops", "access")


START_TRIES = 5


def _flex_answers(svc) -> bool:
    """True when this service's own provider answers ACK on both sockets. The port pair is a
    hint (`tests/_service.py`): if another process takes it first, the service logs the failed
    bind and runs without FLEX, and a probe could reach the stranger instead."""
    import zmq
    if "provider FAILED to bind" in svc.log.read_text(errors="replace"):
        return False
    ctx = zmq.Context()
    try:
        for port in (svc.flex_port, svc.flex_port + 1):
            s = ctx.socket(zmq.REQ)
            s.setsockopt(zmq.LINGER, 0)
            s.setsockopt(zmq.RCVTIMEO, 10000)
            s.connect(f"tcp://127.0.0.1:{port}")
            s.send_string('{"jsonrpc": "2.0", "id": "up", "method": "ACK"}')
            try:
                s.recv_string()
            except zmq.Again:
                return False
            finally:
                s.close(linger=0)
        return True
    finally:
        ctx.term()


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    """The whole scenario, run once on a fresh twin service of this module's own."""
    for _ in range(START_TRIES):
        svc = Service(runtime=tmp_path_factory.mktemp("doors-runtime"), twin_speed=0.0,
                      flex=True)
        try:
            svc.start()
        except BaseException:
            svc.stop()
            raise
        if _flex_answers(svc):
            break
        print(f"[doors] FLEX provider not answering on port {svc.flex_port}; retrying",
              flush=True)
        svc.stop()
    else:
        raise RuntimeError("no service with a FLEX provider after several tries")
    try:
        yield run(svc)
    finally:
        svc.stop()


@pytest.mark.parametrize("group", GROUPS)
def test_transcript_matches_fixture(live, group):
    text = canonical(live[group])
    path = FIXTURES / f"{group}.json"
    if REGEN:
        FIXTURES.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    assert path.read_text(encoding="utf-8") == text


def test_every_group_is_a_test_and_a_fixture(live):
    assert set(live) == set(GROUPS)
    assert {p.stem for p in FIXTURES.glob("*.json")} == set(GROUPS)


def _verbs_seen(live, socket):
    return {s["request"].get("method") for steps in live.values() for s in steps
            if s["socket"] == socket and "method" in s["request"]}


def test_every_verb_is_recorded_on_both_sockets(live):
    from afm_litho.service import flex_dispatch as fd
    assert set(READ_VERBS) | set(COMMAND_VERBS) == set(fd.REGISTRY)
    assert set(fd.REGISTRY) <= _verbs_seen(live, "command")
    assert set(fd.READ_ONLY) <= _verbs_seen(live, "read")
    assert set(fd.COMMAND_ONLY) <= _verbs_seen(live, "read")      # each one refused there


def test_the_transcript_masks_every_token_and_run_id(live):
    import re
    text = canonical(live)
    assert not re.search(r"(scan|approach|write|litho)-\d+", text)
    minted = re.findall(r'"token": "([^"<]+)"', text)
    assert set(minted) <= {"not-the-token", "stale-token", "not-a-real-token"}, minted
