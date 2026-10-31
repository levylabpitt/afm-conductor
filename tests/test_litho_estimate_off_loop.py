"""The write estimate runs off the service loop (docs/TWIN_FRAME_RATE_SPEC.md, fix 2).

`start_write`'s estimate and `GET /api/litho/estimate` compile every object, which is
seconds for a dense fill. They compute in a thread, from copies of the design, so the
loop keeps answering (Abort, Withdraw, `getState`), an edit that arrives meanwhile
neither breaks the compile nor is broken by it, and an Abort, Withdraw or scan Stop
that arrives while `start_write` is estimating stops the write from starting.

The compiles are bounded and never in the default executor, which Withdraw and the
twin park run in: `start_write` has one (a second start while it is alive is refused),
the endpoint has one running and one waiting (latest wins).

The slow compile is a stub that blocks until the test releases it, so "while the
estimate is computing" is a state the test holds, not a race against the machine.
"""

from __future__ import annotations

import asyncio
import importlib
import threading
import time
import types

import pytest
from _im_fixtures_guard import seed_free_air

from afm_litho.litho.compiler import LithoObject, Pattern
from afm_litho.service.host import litho_cmds
from afm_litho.service.commands.errors import BUSY, REFUSED, CommandError

# `client` comes from tests/conftest.py.
WAIT_S = 30.0


@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


def _wait(cond, timeout=WAIT_S, interval=0.02) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return False


def _pattern() -> Pattern:
    kw = {"voltage": 6.0, "speed_um_s": 5.0, "interpolation_um": 0.05}
    return Pattern(name="estimate", objects=[
        LithoObject("a", "path", [(0.0, 0.0), (1.0, 0.0)], **kw),
        LithoObject("b", "path", [(0.0, 1.0), (2.0, 1.0)], **kw)])


@pytest.fixture(autouse=True)
def _design(hub, tmp_path):
    from afm_litho.service.sessions import SessionStore
    prev_store, prev_pattern = hub.store, hub.pattern
    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    hub.pattern = _pattern()
    hub.host.written_objects = set()
    yield
    _wait(lambda: hub.state_label() == "idle", 60.0)
    _wait(lambda: getattr(hub, "_run_claim", None) is None, 10.0)
    hub.store, hub.pattern = prev_store, prev_pattern
    hub.host.written_objects = set()


@pytest.fixture
def slow(monkeypatch):
    """The per-object compile blocks until `release` is set. `entered` says a compile
    is in progress; `threads` lists the thread of each call."""
    real = litho_cmds._object_trajectory
    st = types.SimpleNamespace(entered=threading.Event(), release=threading.Event(),
                               threads=[], running=0, peak=0, lock=threading.Lock(),
                               kinds=set(), seen=[])

    def stub(obj, *cal):
        with st.lock:
            st.threads.append(threading.get_ident())
            st.kinds.add((threading.current_thread().name,
                          threading.current_thread().daemon))
            st.seen.append(obj)
            st.running += 1
            st.peak = max(st.peak, st.running)
        st.entered.set()
        try:
            assert st.release.wait(WAIT_S), "the test never released the compile"
            return real(obj, *cal)
        finally:
            with st.lock:
                st.running -= 1

    monkeypatch.setattr(litho_cmds, "_object_trajectory", stub)
    yield st
    st.release.set()
    # no compile outlives its test: the next one must find the start estimate free
    from afm_litho.service.hub import hub as real_hub
    _wait(lambda: st.running == 0)
    _wait(lambda: real_hub.host._start_estimate is None or real_hub.host._start_estimate[0].done())


def _on_loop(hub, coro):
    return asyncio.run_coroutine_threadsafe(coro, hub._loop)


def _loop_thread(hub) -> int:
    async def ident():
        return threading.get_ident()
    return _on_loop(hub, ident()).result(5.0)


def _loop_answers(hub, client) -> None:
    """The loop takes a turn and serves a request, within a bound that a loop blocked
    on the held compile could never meet."""
    _on_loop(hub, asyncio.sleep(0)).result(5.0)
    assert client.get("/api/state").status_code == 200


def _default_pool_is_free(hub) -> None:
    """A job handed to the default executor (where Withdraw's off-loop work runs)
    starts at once: no estimate compile is occupying or queueing in it."""
    assert _on_loop(hub, asyncio.to_thread(threading.get_ident)).result(5.0)


def _write_params(hub) -> dict:
    fa = seed_free_air(hub)
    return {"objects": "all", "setpoint": fa.value_v + 0.2, "amp_gain": 25.0}


# --------------------------------------------------------------- the estimate --

def test_estimate_endpoint_compiles_off_the_loop(client, hub, slow):
    out: list = []
    t = threading.Thread(
        target=lambda: out.append(client.get("/api/litho/estimate").json()))
    t.start()
    try:
        assert slow.entered.wait(WAIT_S), "the estimate never started compiling"
        _loop_answers(hub, client)
        assert not out, "the estimate answered before its compile was released"
    finally:
        slow.release.set()
    t.join(WAIT_S)
    assert slow.threads and _loop_thread(hub) not in slow.threads
    assert out == [litho_cmds.write_estimate(hub)]
    assert [o["id"] for o in out[0]["objects"]] == ["a", "b"]
    assert out[0]["total_s"] > out[0]["trajectory_s"] > 0


def test_async_estimates_equal_the_sync_ones(hub):
    todo = list(hub.pattern.objects)
    assert (_on_loop(hub, litho_cmds.write_estimate_async(hub)).result(WAIT_S)
            == litho_cmds.write_estimate(hub))
    assert (_on_loop(hub, litho_cmds.write_est_s_async(hub, todo)).result(WAIT_S)
            == litho_cmds.write_est_s(hub, todo))
    assert (litho_cmds.write_estimate(hub)["total_s"]
            == litho_cmds.write_est_s(hub, todo))


def test_an_edit_during_the_estimate_neither_breaks_it_nor_is_undone(hub, slow):
    """The compile works on copies taken when it starts."""
    slow.release.set()
    expected = litho_cmds.write_estimate(hub)
    slow.release.clear()
    slow.entered.clear()

    fut = _on_loop(hub, litho_cmds.write_estimate_async(hub))
    try:
        assert slow.entered.wait(WAIT_S)
        # an edit lands while object "a" is compiling: "b" loses its points and its
        # speed changes, "a" is removed, a third object appears
        objs = hub.pattern.objects
        b = objs[1]
        b.points_um.clear()
        b.speed_um_s = 99.0
        objs.pop(0)
        extra = LithoObject("c", "path", [(0.0, 2.0), (3.0, 2.0)], voltage=6.0)
        objs.append(extra)
    finally:
        slow.release.set()
    assert fut.result(WAIT_S) == expected               # the design as it was
    # ...and the design is exactly what the edit made it
    assert hub.pattern.objects == [b, extra]
    assert b.points_um == [] and b.speed_um_s == 99.0


def test_detached_copy_shares_no_container_with_the_design():
    obj = LithoObject("r", "shape", [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                      holes_um=[[(0.2, 0.2), (0.4, 0.2), (0.3, 0.4)]],
                      bezier={"nodes": [{"p": [0.0, 0.0]}]},
                      v_profile=[1.0, 2.0, 3.0])
    obj.chiral_clamp = {"fraction": 0.5}                # an attribute set after init
    c = litho_cmds._detached(obj)
    assert c == obj and c is not obj
    assert c.chiral_clamp == {"fraction": 0.5} and c.chiral_clamp is not obj.chiral_clamp
    obj.points_um[0][0] = 9.0
    obj.points_um.append([5.0, 5.0])
    obj.holes_um[0].clear()
    deep = LithoObject("h", "shape", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
                       holes_um=[[[0.2, 0.2], [0.4, 0.2], [0.3, 0.4]]])
    d = litho_cmds._detached(deep)
    deep.holes_um[0][0][0] = 9.0                         # a point pair inside a hole
    assert d.holes_um == [[[0.2, 0.2], [0.4, 0.2], [0.3, 0.4]]]
    obj.bezier["nodes"][0]["p"][0] = 9.0
    obj.v_profile.append(4.0)
    assert c.points_um == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    assert c.holes_um == [[(0.2, 0.2), (0.4, 0.2), (0.3, 0.4)]]
    assert c.bezier == {"nodes": [{"p": [0.0, 0.0]}]}
    assert c.v_profile == [1.0, 2.0, 3.0]


def test_endpoint_estimates_are_latest_wins(client, hub, slow):
    """A burst of requests costs two compiles: the one running, and one more that
    every request made meanwhile shares and that reads the design when it starts."""
    slow.release.set()
    before = litho_cmds.write_estimate(hub)
    slow.release.clear()
    slow.entered.clear()
    del slow.threads[:]
    slow.kinds.clear()

    first = _on_loop(hub, litho_cmds.write_estimate_async(hub))
    try:
        assert slow.entered.wait(WAIT_S)
        later = [_on_loop(hub, litho_cmds.write_estimate_async(hub)) for _ in range(6)]
        for _ in range(20):                              # let every request get going
            _on_loop(hub, asyncio.sleep(0)).result(5.0)
        assert slow.peak == 1 and len(slow.threads) == 1          # one compile alive
        _default_pool_is_free(hub)
        assert client.post("/api/withdraw").status_code == 200    # and Withdraw answers
        # an edit lands after those requests were made, before their compile starts
        hub.pattern.objects.pop()
    finally:
        slow.release.set()
    assert first.result(WAIT_S) == before
    got = [f.result(WAIT_S) for f in later]
    n_compiled, kinds = len(slow.threads), set(slow.kinds)
    after = litho_cmds.write_estimate(hub)               # the design as edited
    assert got == [after] * 6
    assert len(after["objects"]) == 1 and after != before
    assert n_compiled == 2 + 1                 # 2 objects, then 1: two compiles in all
    assert kinds == {("litho-estimate", True)}           # own daemon threads


def test_an_endpoint_request_that_goes_away_does_not_cancel_the_others(hub, slow):
    a = _on_loop(hub, litho_cmds.write_estimate_async(hub))
    assert slow.entered.wait(WAIT_S)
    b = _on_loop(hub, litho_cmds.write_estimate_async(hub))
    c = _on_loop(hub, litho_cmds.write_estimate_async(hub))
    _on_loop(hub, asyncio.sleep(0)).result(5.0)
    a.cancel()
    b.cancel()
    slow.release.set()
    assert c.result(WAIT_S) == litho_cmds.write_estimate(hub)
    # the queue is free again afterwards
    assert (_on_loop(hub, litho_cmds.write_estimate_async(hub)).result(WAIT_S)
            == litho_cmds.write_estimate(hub))
    assert hub.host._estimate_queue.running is None and hub.host._estimate_queue.waiting is None


def test_start_estimate_works_on_copies_of_the_objects(hub, slow):
    """An edit to the very objects being estimated, in place, does not reach the
    compile."""
    todo = list(hub.pattern.objects)
    slow.release.set()
    expected = litho_cmds.write_est_s(hub, todo)
    slow.release.clear()
    slow.entered.clear()
    del slow.seen[:]

    fut = _on_loop(hub, litho_cmds.write_est_s_async(hub, todo))
    try:
        assert slow.entered.wait(WAIT_S)
        for obj in todo:                       # every stroke ten times as long
            obj.points_um[1] = (obj.points_um[1][0] * 10.0, obj.points_um[1][1])
            obj.speed_um_s = 0.01
    finally:
        slow.release.set()
    assert fut.result(WAIT_S) == expected
    assert all(o is not t for o in slow.seen for t in todo)
    assert litho_cmds.write_est_s(hub, todo) > 10 * expected       # the edit is real


# ----------------------------------------------------------------- start_write --

def test_start_write_estimates_off_the_loop_and_its_reply_is_unchanged(client, hub, slow):
    params = _write_params(hub)
    todo = list(hub.pattern.objects)
    slow.release.set()
    expected_est = litho_cmds.write_est_s(hub, todo)
    slow.release.clear()
    slow.entered.clear()
    del slow.threads[:]

    fut = _on_loop(hub, litho_cmds.start_write(hub, params))
    try:
        assert slow.entered.wait(WAIT_S), "start_write never started its estimate"
        _loop_answers(hub, client)
        assert not fut.done()
        assert hub.state_label() != "writing"            # nothing started yet
    finally:
        slow.release.set()
    res = fut.result(WAIT_S)
    assert _loop_thread(hub) not in slow.threads
    assert set(res) == {"run_id", "est_s", "objects", "segments", "state", "backend"}
    assert res["est_s"] == expected_est
    assert res["objects"] == 2 and res["state"] == "writing" and res["run_id"]
    assert _wait(lambda: hub.state_label() != "writing", 120.0)


@pytest.mark.parametrize("stop", ["/api/litho/abort", "/api/withdraw", "/api/scan/abort"])
def test_a_stop_during_the_estimate_is_served_and_the_write_does_not_start(
        client, hub, slow, stop):
    params = _write_params(hub)
    runs_before = hub._in_flight_run_id()
    thread_before = getattr(hub.host, "litho_thread", None)

    fut = _on_loop(hub, litho_cmds.start_write(hub, params))
    try:
        assert slow.entered.wait(WAIT_S), "start_write never started its estimate"
        # the stop is answered while the estimate is still held
        r = client.post(stop)
        assert r.status_code == 200, r.text
        assert not fut.done()
    finally:
        slow.release.set()
    with pytest.raises(CommandError) as exc:
        fut.result(WAIT_S)
    assert exc.value.code == REFUSED
    assert exc.value.data == {"reason": "aborted_during_start"}
    # no write behind the stop: no worker, no claim, no run
    assert getattr(hub.host, "litho_thread", None) is thread_before
    assert getattr(hub, "_run_claim", None) is None
    assert hub._in_flight_run_id() == runs_before
    assert hub.state_label() != "writing"
    assert hub.host.written_objects == set()


def test_a_stop_before_the_estimate_does_not_refuse_the_write(client, hub):
    """Only a stop that arrives DURING the start refuses it: the abort flags stay set
    after every park, and a write after a park is the normal case."""
    assert client.post("/api/litho/abort").status_code == 200
    res = _on_loop(hub, litho_cmds.start_write(hub, _write_params(hub))).result(WAIT_S)
    assert res["state"] == "writing" and res["objects"] == 2
    assert _wait(lambda: hub.state_label() != "writing", 120.0)


# ------------------------------------------- start_write's compile is bounded --

def test_an_abandoned_start_estimate_is_the_only_one_and_delays_no_withdraw(
        client, hub, slow):
    """The FLEX reply bound cancels `start_write` while its compile runs on. Repeated
    starts must not add compiles: each later one is refused, the default executor
    stays free, and Withdraw and Abort answer."""
    from afm_litho.service import flex_provider as fp
    params = _write_params(hub)
    with pytest.raises(fp.RpcError):                     # the bound, as FLEX applies it
        fp.run_on_loop(hub, litho_cmds.start_write(hub, dict(params)), timeout=0.2)
    assert slow.entered.wait(WAIT_S)
    for _ in range(22):
        with pytest.raises(CommandError) as exc:
            fp.run_on_loop(hub, litho_cmds.start_write(hub, dict(params)), timeout=5.0)
        assert exc.value.code == BUSY
        assert exc.value.data["reason"] == "estimate_in_progress"
    assert slow.peak == 1 and len(slow.threads) == 1
    assert slow.kinds == {("litho-estimate", True)}
    _default_pool_is_free(hub)
    assert client.post("/api/withdraw").status_code == 200
    assert client.post("/api/litho/abort").status_code == 200
    assert hub.state_label() != "writing" and getattr(hub, "_run_claim", None) is None

    # released, the abandoned compile stops at its next object instead of finishing
    slow.release.set()
    assert _wait(lambda: hub.host._start_estimate[0].done())
    assert len(slow.threads) == 1, "the abandoned estimate went on to the next object"
    # nothing started behind the cancelled call, and the next start is accepted
    assert hub.state_label() != "writing" and getattr(hub, "_run_claim", None) is None
    res = fp.run_on_loop(hub, litho_cmds.start_write(hub, _write_params(hub)),
                         timeout=WAIT_S)
    assert res["state"] == "writing" and res["objects"] == 2
    assert _wait(lambda: hub.state_label() != "writing", 120.0)


# ------------------------------------------------ where the stop counters change --
#
# `start_write` refuses when a counter changed across its estimate. Each counter
# changes only once its stop has been signalled (2026-10-06): whoever sees the
# count move finds the abort flags already set, and the count moves before the caller
# goes on to the Igor command.

class _OrderHub:
    """A hub stand-in whose abort counter records the flags at the moment it is set."""

    def __init__(self, fail: bool = False):
        self._hw_abort = threading.Event()
        self._hw_litho_abort = threading.Event()
        self._twin_plugin = types.SimpleNamespace(aborted=False, abort=self._abort_twin)
        self.scan_repeat = {"enabled": True, "stopped_reason": None}
        self._scan_repeat_stop = threading.Event()
        self._seq = 0
        self._fail = fail
        self.flags_at_count: list[tuple] = []

    def _abort_twin(self, reason: str) -> None:
        self._twin_plugin.aborted = True

    @property
    def _run_abort_seq(self) -> int:
        return self._seq

    @_run_abort_seq.setter
    def _run_abort_seq(self, value: int) -> None:
        self.flags_at_count.append((self._hw_abort.is_set(),
                                    self._hw_litho_abort.is_set(),
                                    self._twin_plugin.aborted))
        if self._fail:
            raise RuntimeError("counter unavailable")
        self._seq = value

    @property
    def host(self):                # fake: host fields live on the fake itself
        return self


@pytest.mark.parametrize("entry", ["signal_run_abort", "_signal_stop"])
def test_the_abort_counter_changes_after_every_abort_flag_is_set(entry):
    """Both entries: `withdraw` calls `signal_run_abort`; every park (the async door,
    the watchdog's work, the sync `safe_park`) calls `_signal_stop`, which calls it."""
    sp = importlib.import_module("afm_litho.service.host.safe_park")
    h = _OrderHub()
    getattr(sp, entry)(h)
    assert h.flags_at_count == [(True, True, True)]
    assert h._run_abort_seq == 1
    getattr(sp, entry)(h)
    assert h._run_abort_seq == 2 and h.flags_at_count[-1] == (True, True, True)


def test_a_failing_abort_counter_neither_raises_nor_costs_a_flag():
    sp = importlib.import_module("afm_litho.service.host.safe_park")
    h = _OrderHub(fail=True)
    sp.signal_run_abort(h)                               # must not raise
    assert h._hw_abort.is_set() and h._hw_litho_abort.is_set() and h._twin_plugin.aborted
    sp._signal_stop(h)                                   # nor through a park's step 1
    assert h._scan_repeat_stop.is_set() and h.scan_repeat["enabled"] is False


def test_every_park_entry_counts_through_the_one_function(hub):
    """Each way into a park or withdraw bumps the hub's abort counter, which is how
    `start_write` learns an abort arrived while it was estimating."""
    sp = importlib.import_module("afm_litho.service.host.safe_park")
    from afm_litho.service.host import scan_cmds

    def watchdog_park(h, reason):
        wd = sp.watchdog_for(h)
        try:
            wd.request(reason).wait()
        finally:
            wd.stop()
            h.host._failsafe = None

    entries = {
        "safe_park_async": lambda h: asyncio.run(sp.safe_park_async(h, "test: counter")),
        "safe_park": lambda h: sp.safe_park(h, "test: counter"),
        "watchdog": lambda h: watchdog_park(h, "test: counter"),
        "withdraw": lambda h: asyncio.run(scan_cmds.withdraw(h)),
    }
    for name, enter in entries.items():
        before = hub.host._run_abort_seq
        enter(hub)
        assert hub.host._run_abort_seq > before, f"{name} did not count an abort"


def test_the_scan_stop_counter_changes_after_the_stop_is_signalled(client, hub,
                                                                   monkeypatch):
    from afm_litho.service.twin_plugin import plugin
    seen: list[tuple] = []
    calls = {"abort_scan": 0}
    twin = plugin(hub)
    real_abort = twin.abort

    def abort(reason):
        calls["abort_scan"] += 1
        return real_abort(reason)

    monkeypatch.setattr(twin, "abort", abort)
    box = {"v": hub.host._scan_stop_seq}

    def setter(self, value):
        seen.append((calls["abort_scan"], self.scan_repeat.get("enabled"),
                     self._scan_repeat_stop.is_set()))
        box["v"] = value

    # a data descriptor on the class takes precedence over the instance attribute
    monkeypatch.setattr(type(hub.host), "_scan_stop_seq",
                        property(lambda self: box["v"], setter), raising=False)
    prev_repeat = dict(hub.host.scan_repeat)
    hub.host.scan_repeat["enabled"] = True                    # a continuous run to stop
    hub.host._scan_repeat_stop.clear()
    try:
        before = box["v"]
        assert client.post("/api/scan/abort").status_code == 200
        assert box["v"] == before + 1
        # at the moment it changed: the scan was told to stop, the repeat loop was
        # disabled and its stop event set
        assert seen == [(1, False, True)]
    finally:
        hub.host.scan_repeat.clear()
        hub.host.scan_repeat.update(prev_repeat)
        hub.host.scan_repeat["enabled"] = False
        hub.host._scan_repeat_stop.clear()
