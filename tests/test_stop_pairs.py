"""A door's Stop and the keeper's copy of it share one park (service/safety/stop_pairs.py)."""

from __future__ import annotations

import pytest

from afm_litho import clock
from afm_litho.hardware.failsafe_thread import ParkRequest
from afm_litho.service.safety import stop_pairs


@pytest.fixture(autouse=True)
def fresh():
    stop_pairs.clear()
    with clock.using(clock.TESTING):
        yield
    stop_pairs.clear()


def _starter(made: list):
    def start():
        req = ParkRequest("r")
        made.append(req)
        return req
    return start


def _finish(req):
    req.result = {"verified": True}
    req.future.set_result(req.result)
    req.done.set()


def test_the_second_with_the_same_id_shares_the_first_park():
    made = []
    a, first_a = stop_pairs.start_once("0" * 16, _starter(made))
    b, first_b = stop_pairs.start_once("0" * 16, _starter(made))
    assert (first_a, first_b) == (True, False) and a is b and len(made) == 1


def test_no_id_always_parks():
    made = []
    for _ in range(3):
        _, first = stop_pairs.start_once(None, _starter(made))
        assert first
    assert len(made) == 3


def test_different_ids_park_each():
    made = []
    stop_pairs.start_once("a" * 16, _starter(made))
    stop_pairs.start_once("b" * 16, _starter(made))
    assert len(made) == 2


def test_an_id_is_forgotten_a_while_after_its_outcome_then_a_late_copy_parks():
    made = []
    req, _ = stop_pairs.start_once("c" * 16, _starter(made))
    clock.advance(60.0)                          # no outcome yet: still held
    assert stop_pairs.start_once("c" * 16, _starter(made))[1] is False
    _finish(req)
    clock.advance(stop_pairs.EXPIRE_AFTER_S - 1.0)
    assert stop_pairs.start_once("c" * 16, _starter(made))[1] is False
    clock.advance(2.0)
    assert stop_pairs.start_once("c" * 16, _starter(made))[1] is True
    assert len(made) == 2


def test_split_id():
    sid = "0123456789abcdef"
    assert stop_pairs.split_id(f"operator abort #{sid} (via keeper)") == (
        "operator abort (via keeper)", sid)
    assert stop_pairs.split_id(f"abort #{sid}") == ("abort", sid)
    assert stop_pairs.split_id("pressed S (via keeper)") == ("pressed S (via keeper)", None)
    assert stop_pairs.split_id("x #nothex0123456789") == ("x #nothex0123456789", None)
