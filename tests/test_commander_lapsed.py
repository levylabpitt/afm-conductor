"""The dead-man and engaged-idle verdict table: every input to today's timers and what
each one does (action record, park calls, log text, page frames, journal note).

Written on today's code before H7.5d moved the verdict into the host, and kept
byte-identical after it (docs/migration/plan/host-lift-board/H7.5-spec.md, H7.5d).
Everything is driven through `control.control_tick(hub, now=...)` with an injected clock.

The park is a spy on `request_park` (both the host module and the scan router's import of
it), so no case needs the twin to retract. The spy ends the engagement as a real retract
does, which is what makes the in-tick order (dead-man first, then the cap) visible.
"""
# ruff: noqa: F811

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from afm_litho.service import control as ctl
from afm_litho.service.host import safe_park
from afm_litho.service.journal import Journal
from test_flex_control import _FakeScan, _run, hub  # noqa: F401

CLIENT = "flex-exp"
T0 = 1_000_000.0          # injected clock origin; no case reads the wall clock
CAP_S = 10.0              # AFM_LITHO_ENGAGED_IDLE_MAX_S for every case


# ------------------------------------------------------------------- the spies --

@pytest.fixture
def parks(monkeypatch):
    """Every `request_park` call as (reason, how, body). A full park returns a marker
    dict; a withdraw returns `parks.withdraw_reply` (default `{}`)."""
    from afm_litho.service.routers import scan as scan_router

    box = SimpleNamespace(calls=[], reply={"withdraw": {}})

    def spy(hub, reason, how=safe_park.PARK_FULL, **kw):
        box.calls.append((reason, how, kw.get("body")))
        reply = ({"spy_park": reason} if how == safe_park.PARK_FULL
                 else dict(box.reply["withdraw"]))

        async def _done():
            if not reply.get("not_connected"):
                hub.host.engaged = False         # a retract ends the engagement
            return reply
        return _done()

    monkeypatch.setattr(safe_park, "request_park", spy)
    monkeypatch.setattr(scan_router, "request_park", spy)
    return box


@pytest.fixture
def jr(tmp_path):
    j = Journal(tmp_path).start()
    yield j
    j.stop()


def _lapse_notes(j):
    j.stop()
    lines = [json.loads(x) for x in j.path.read_text("utf-8").splitlines()]
    return [(x["actor"], x["body"]["state"], x["body"]["action"]) for x in lines
            if x.get("src") == "access" and x.get("what") == "lapse"]


@pytest.fixture(autouse=True)
def _cap(monkeypatch):
    monkeypatch.setenv("AFM_LITHO_ENGAGED_IDLE_MAX_S", str(CAP_S))


# ------------------------------------------------------------------ the states --

class _AliveThread:
    def is_alive(self):
        return True


def _approaching(h):
    h.host._approach_in_flight = 1


def _scanning(h):
    h.host._scan_task = _FakeScan(h)


def _continuous(h):
    h.host.scan_repeat["enabled"] = True


def _writing_thread(h):
    h.host.litho_thread = _AliveThread()


def _writing_run(h):
    h.runs.allocate("write")


def _engaged(h):
    h.host.engaged = True


def _parked(h):
    h.host._safe_park["active"] = True


def _fault(h):
    h.host.fault = {"reason": "test fault"}


def _disconnected(h):
    h.host.backend = "hardware"          # the fake's backend_live() is always "twin"


# name -> (setups, the state word today's derivation gives). Combined rows pin the
# derivation's first-match order.
STATES = {
    "approaching": ((_approaching,), "approaching"),
    "scanning": ((_scanning,), "scanning"),
    "continuous": ((_continuous,), "scanning"),
    "writing_thread": ((_writing_thread,), "writing"),
    "writing_run": ((_writing_run,), "writing"),
    "engaged": ((_engaged,), "engaged"),
    "idle": ((), "idle"),
    "parked": ((_parked,), "parked"),
    "fault": ((_fault,), "fault"),
    "disconnected": ((_disconnected,), "disconnected"),
    "fault_over_scanning": ((_fault, _scanning), "fault"),
    "disconnected_over_writing": ((_disconnected, _writing_thread), "disconnected"),
    "writing_over_approaching": ((_writing_thread, _approaching), "writing"),
    "approaching_over_scanning": ((_approaching, _scanning), "approaching"),
    "scanning_over_parked": ((_scanning, _parked), "scanning"),
    "parked_over_engaged": ((_parked, _engaged), "parked"),
    "engaged_scan_running": ((_engaged, _scanning), "scanning"),
}


def _set_up(h, name):
    for fn in STATES[name][0]:
        fn(h)
    return STATES[name][1]


def _holder(h, holder, wod, now):
    """Install the commander. `script` is acquired so it has expired by `now`;
    `script_live` has not."""
    if holder == "vacant":
        return None
    if holder == "operator":
        return h.acquire_control(CLIENT, "operator", now=now - 100)
    start = now - 5.001 if holder == "script" else now - 1
    return h.acquire_control(CLIENT, "script", deadman_s=5,
                             withdraw_on_deadman=wod, now=start)


# --------------------------------------------------------------- the verdicts --

def _expected_lapse(word, wod):
    if word in ("approaching", "scanning", "writing", "measuring"):
        return "safe_park"
    if word == "engaged":
        return "withdraw" if wod else "warn"
    return "release"


def _expected_log(action, word):
    return {
        "safe_park": f"DEAD-MAN: {CLIENT} went silent while {word} -- aborting and "
                     f"safe-parking (verified retract)",
        "withdraw": f"DEAD-MAN: {CLIENT} went silent while engaged -- withdrawing "
                    f"(withdraw_on_deadman was set)",
        "warn": f"WARNING DEAD-MAN: {CLIENT} went silent while ENGAGED. The token is "
                f"released and the tip is LEFT ON THE SURFACE (withdraw_on_deadman was "
                f"not set). The engaged-idle limit ({CAP_S:g}s) still applies.",
        "release": f"DEAD-MAN: {CLIENT} went silent while {word} -- token released, no "
                   f"instrument action",
    }[action]


CAP_LOG = (f"ENGAGED-IDLE LIMIT: the tip has been engaged with nothing running for over "
           f"{CAP_S:g}s -- withdrawing. This limit is not optional and applies whether "
           f"or not anyone holds control.")


def _expected_record(action, word):
    rec = {"action": action, "state": word, "holder": CLIENT}
    if action == "safe_park":
        rec["safe_park"] = {"spy_park": "deadman"}
    return rec


def _expected_parks(action):
    if action == "safe_park":
        return [("deadman", safe_park.PARK_FULL, None)]
    if action == "withdraw":
        return [("withdraw", safe_park.PARK_WITHDRAW, {})]
    return []


# --------------------------------------------------------- 1. the dead-man table --

@pytest.mark.parametrize("wod", [False, True], ids=["plain", "withdraw_on_deadman"])
@pytest.mark.parametrize("holder", ["script", "script_live", "operator", "vacant"])
@pytest.mark.parametrize("state", list(STATES))
def test_deadman_verdict(hub, parks, jr, state, holder, wod):
    now = T0
    word = _set_up(hub, state)
    rec = _holder(hub, holder, wod, now)
    acts = _run(hub, ctl.control_tick(hub, now=now))
    deadman = [a for a in acts if a["action"] != "engaged_idle_withdraw"]
    notes = _lapse_notes(jr)
    if holder != "script":
        # No lapse: the token (if any) is untouched and nothing is parked, logged
        # or journalled by the dead-man.
        assert deadman == []
        assert parks.calls == []
        assert not [x for x in hub.logs if "DEAD-MAN" in x]
        assert notes == []
        assert hub.control is rec
        return
    action = _expected_lapse(word, wod)
    assert deadman == [_expected_record(action, word)]
    assert parks.calls == _expected_parks(action)
    assert [x for x in hub.logs if "DEAD-MAN" in x] == [_expected_log(action, word)]
    assert notes == [(CLIENT, word, action)]
    # Release first: the token is gone and revoked, and the pages saw it go.
    assert hub.control is None and rec.token in hub._revoked_tokens
    assert [f["type"] for f in hub.frames][:1] == ["control"]


@pytest.mark.parametrize("state", ["engaged", "idle", "scanning"])
def test_a_withdraw_that_sent_nothing_is_reported_not_connected(hub, parks, state):
    parks.reply["withdraw"] = {"not_connected": True}
    word = _set_up(hub, state)
    _holder(hub, "script", True, T0)
    acts = _run(hub, ctl.control_tick(hub, now=T0))
    if word == "engaged":
        assert acts == [{"action": "withdraw", "state": "engaged", "holder": CLIENT,
                         "not_connected": True}]
        assert hub.host.engaged is True
    else:
        assert acts[0]["action"] == _expected_lapse(word, True)


# ------------------------------------------------------ 2. the engaged-idle table --

@pytest.mark.parametrize("holder", ["script_live", "operator", "vacant"])
@pytest.mark.parametrize("state", list(STATES))
def test_engaged_idle_verdict(hub, parks, state, holder):
    word = _set_up(hub, state)
    rec = _holder(hub, holder, False, T0 + CAP_S + 1)
    assert _run(hub, ctl.control_tick(hub, now=T0)) == []          # starts the clock
    assert _run(hub, ctl.control_tick(hub, now=T0 + CAP_S - 1)) == []
    acts = _run(hub, ctl.control_tick(hub, now=T0 + CAP_S + 1))
    caps = [f for f in hub.frames if f.get("type") == "engaged_idle_withdraw"]
    if word != "engaged":
        assert acts == [] and parks.calls == [] and caps == []
        assert not hub.logged("ENGAGED-IDLE")
    else:
        assert acts == [{"action": "engaged_idle_withdraw", "limit_s": CAP_S}]
        assert parks.calls == [("withdraw", safe_park.PARK_WITHDRAW, {})]
        assert [x for x in hub.logs if "ENGAGED-IDLE" in x] == [CAP_LOG]
        assert caps == [{"type": "engaged_idle_withdraw", "t": T0 + CAP_S + 1,
                         "limit_s": CAP_S}]
    assert hub.control is rec


def test_the_cap_that_sent_nothing_announces_no_withdraw(hub, parks):
    parks.reply["withdraw"] = {"not_connected": True}
    _engaged(hub)
    _run(hub, ctl.control_tick(hub, now=T0))
    acts = _run(hub, ctl.control_tick(hub, now=T0 + CAP_S + 1))
    assert acts == [{"action": "engaged_idle_withdraw", "limit_s": CAP_S,
                     "not_connected": True}]
    assert not [f for f in hub.frames if f.get("type") == "engaged_idle_withdraw"]
    assert hub.logged("ENGAGED-IDLE LIMIT")


def test_the_cap_restarts_after_anything_but_engaged(hub, parks):
    _engaged(hub)
    _run(hub, ctl.control_tick(hub, now=T0))
    hub.host._scan_task = _FakeScan(hub)                  # a scan runs for a while
    assert _run(hub, ctl.control_tick(hub, now=T0 + CAP_S + 1)) == []
    hub.host._scan_task = None                            # engaged and idle again
    assert _run(hub, ctl.control_tick(hub, now=T0 + CAP_S + 2)) == []   # clock restarts
    assert _run(hub, ctl.control_tick(hub, now=T0 + 2 * CAP_S + 1)) == []
    acts = _run(hub, ctl.control_tick(hub, now=T0 + 2 * CAP_S + 3))
    assert [a["action"] for a in acts] == ["engaged_idle_withdraw"]


# ------------------------------------------- 3. both timers in one tick: the order --

def test_a_deadman_withdraw_ends_the_engagement_before_the_cap_looks(hub, parks):
    _engaged(hub)
    _run(hub, ctl.control_tick(hub, now=T0))              # the cap starts counting
    _holder(hub, "script", True, T0 + CAP_S + 1)
    acts = _run(hub, ctl.control_tick(hub, now=T0 + CAP_S + 1))
    assert [a["action"] for a in acts] == ["withdraw"]
    assert parks.calls == [("withdraw", safe_park.PARK_WITHDRAW, {})]


def test_a_deadman_warn_leaves_the_cap_to_withdraw_in_the_same_tick(hub, parks):
    _engaged(hub)
    _run(hub, ctl.control_tick(hub, now=T0))
    _holder(hub, "script", False, T0 + CAP_S + 1)
    acts = _run(hub, ctl.control_tick(hub, now=T0 + CAP_S + 1))
    assert [a["action"] for a in acts] == ["warn", "engaged_idle_withdraw"]
    assert parks.calls == [("withdraw", safe_park.PARK_WITHDRAW, {})]


# ------------------------------------------------- 4. start, then lapse 0.25 s on --

@pytest.mark.parametrize("start,word", [(_writing_run, "writing"),
                                        (_writing_thread, "writing"),
                                        (_scanning, "scanning")])
def test_a_lapse_0_25_s_after_a_start_safe_parks_once(hub, parks, start, word):
    rec = hub.acquire_control(CLIENT, "script", deadman_s=5, now=T0 - 4.75)
    start(hub)                                            # started at T0
    assert _run(hub, ctl.control_tick(hub, now=T0)) == []
    acts = _run(hub, ctl.control_tick(hub, now=rec.expires_at))   # T0 + 0.25
    assert [(a["action"], a["state"]) for a in acts] == [("safe_park", word)]
    assert parks.calls == [("deadman", safe_park.PARK_FULL, None)]
    assert hub.control is None
