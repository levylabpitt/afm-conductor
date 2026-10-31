"""The access holder (slice 2.6a): each rule, and the journal note it writes."""
# ruff: noqa: F811

from __future__ import annotations

import json
import time

import pytest

from afm_litho.service import access
from afm_litho.service import control as ctl
from afm_litho.service.host import deadman
from afm_litho.service.journal import Journal
from test_flex_control import cmd, hub, provider  # noqa: F401


@pytest.fixture
def jr(tmp_path):
    j = Journal(tmp_path).start()
    yield j
    j.stop()


def access_notes(j):
    j.stop()
    lines = [json.loads(x) for x in j.path.read_text("utf-8").splitlines()]
    return [x for x in lines if x.get("src") == "access"]


def test_script_requests_grants_when_free_and_clamps(hub):
    rec = access.script_requests(hub, "s1", 1, True)
    assert rec.deadman_s == ctl.DEADMAN_MIN_S and rec.withdraw_on_deadman
    assert hub.control is rec


def test_script_refused_while_held_but_not_when_expired(hub):
    first = access.script_requests(hub, "s1")
    with pytest.raises(ctl.ControlHeld) as e:
        access.script_requests(hub, "s2")
    assert e.value.code == ctl.CONTROL_HELD and e.value.data["holder"] == "s1"
    second = access.script_requests(hub, "s2", now=first.expires_at + 1)
    assert hub.control is second
    assert first.token in hub._revoked_tokens


def test_operator_takes_revokes_script_and_never_expires(hub):
    script = access.script_requests(hub, "s1")
    prev, rec = access.operator_takes(hub)
    assert prev is script and rec.kind == "operator" and rec.expires_at is None
    with pytest.raises(ctl.ControlRevoked):
        access.check(hub, script.token)


def test_operator_tip_action_only_revokes_a_script(hub):
    assert access.operator_tip_action(hub, "/api/approach") is None
    access.operator_takes(hub)
    assert access.operator_tip_action(hub, "/api/approach") is None
    assert hub.control.kind == "operator"
    access.release(hub, hub.control.token)
    script = access.script_requests(hub, "s2")
    assert access.operator_tip_action(hub, "/api/withdraw") is script
    assert hub.control is None


def test_release_matching_token_only_and_never_raises(hub):
    rec = access.script_requests(hub, "s1")
    assert access.release(hub, "wrong") == {"ok": True, "stale": True}
    assert access.release(hub, None) == {"ok": True, "stale": True}
    assert access.release(hub, rec.token) == {"ok": True, "stale": False}
    assert access.release(hub, rec.token) == {"ok": True, "stale": True}


def test_check_codes(hub):
    rec = access.script_requests(hub, "s1")
    assert access.check(hub, rec.token) is rec
    with pytest.raises(ctl.ControlRevoked) as e:
        access.check(hub, rec.token, now=rec.expires_at + 1)
    assert e.value.code == ctl.CONTROL_REVOKED
    with pytest.raises(ctl.ControlRequired) as e:
        access.check(hub, "never issued")
    assert e.value.code == ctl.CONTROL_REQUIRED
    access.release(hub, rec.token)
    with pytest.raises(ctl.ControlRevoked):
        access.check(hub, rec.token)


def test_refresh_extends_but_never_resurrects_an_expired_record(hub):
    rec = access.script_requests(hub, "s1", 10)
    assert access.refresh(hub, rec.token, now=rec.acquired_at + 5)
    assert hub.control.expires_at == rec.acquired_at + 15
    assert not access.refresh(hub, "other")
    assert not access.refresh(hub, rec.token, now=rec.acquired_at + 100)


@pytest.mark.parametrize("asked", [False, True])
def test_lapsed_releases_first_and_names_the_holder(hub, monkeypatch, asked):
    """The base half only: the tip's state and the action are the host's
    (host/deadman.py, tests/test_commander_lapsed.py)."""
    def no_state(h):
        raise AssertionError("access read the tip's state")

    monkeypatch.setattr(deadman, "pass_state", no_state)
    rec = access.script_requests(hub, "s1", 5, asked)
    assert access.lapsed(hub, rec.acquired_at + 1) is None and hub.control is rec
    lapse = access.lapsed(hub, rec.expires_at + 1)
    assert (lapse.holder, lapse.withdraw_on_deadman) == ("s1", asked)
    assert hub.control is None and rec.token in hub._revoked_tokens


def test_operator_never_lapses(hub):
    access.operator_takes(hub)
    assert access.lapsed(hub, time.time() + 1e9) is None


def test_every_change_is_journalled_with_actor_and_no_token(hub, jr):
    rec = access.script_requests(hub, "s1")
    with pytest.raises(ctl.ControlHeld):
        access.script_requests(hub, "s2")
    access.release(hub, rec.token)
    script = access.script_requests(hub, "s3")
    access.operator_tip_action(hub, "/api/approach")
    access.operator_takes(hub)
    got = [(n["what"], n["actor"]) for n in access_notes(jr)]
    assert got == [("request", "s1"), ("grant", "s1"), ("request", "s2"),
                   ("refused", "s2"), ("release", "s1"), ("request", "s3"),
                   ("grant", "s3"), ("auto_take", "operator"), ("take", "operator")]
    assert script.token not in jr.path.read_text("utf-8")


def test_release_is_journalled_with_the_given_actor(hub, jr):
    rec = access.script_requests(hub, "s1")
    access.release(hub, rec.token, actor="operator")
    assert [n["actor"] for n in access_notes(jr) if n["what"] == "release"] == ["operator"]


def test_stop_verbs_reach_access_only_through_the_abort_refresh(cmd, hub, monkeypatch):
    """No stop verb asks access for anything, with one exception that is today's
    behaviour: FLEX `abort` carrying a token refreshes the dead-man (`_h_abort` ->
    `hub.touch_control` -> `access.refresh`). Token-free stops reach access not at all."""
    reached = []

    def spy(name):
        real = getattr(access, name)

        def wrapper(*args, **kw):
            reached.append(name)
            return real(*args, **kw)
        monkeypatch.setattr(access, name, wrapper)

    for name in ("script_requests", "operator_takes", "operator_tip_action", "release",
                 "check", "refresh", "lapsed"):
        spy(name)
    for verb in ("abort", "safePark", "withdraw", "abortWrite"):
        cmd.send(verb)
    assert reached == []
    token = cmd.call("acquireControl", {"client": "s"})["token"]
    reached.clear()
    cmd.send("abort", {"token": token})
    assert set(reached) == {"refresh"}
