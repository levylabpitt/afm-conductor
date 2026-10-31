"""Shared fixtures for the HTTP-layer integration tests.

The `client` fixture stands up a FastAPI TestClient.
Module-scoped so each test file gets a fresh startup; the underlying `hub` is a
process global, so individual tests must still be self-contained.

The FLEX provider is switched OFF for the same reason (AFM_LITHO_FLEX=0): app
startup would otherwise bind the real 29180/29181 on every module's TestClient, and
those ports may well be held by a service the developer is actually running.
test_flex_provider.py starts its own provider on ephemeral ports instead.

The twin plays a write at unlimited speed for the whole suite
(AFM_LITHO_TWIN_SPEED=0, service/twin_player.py), so no test's run time is the
simulated duration of what it writes. It is set at import, before any test module
can construct the process-global hub that reads it. A test that needs a write to
last sets its own speed (`hub.twin_speed`, a write's `twin_speed`, or a `LithoRun`
argument).
"""

from __future__ import annotations

import atexit
import contextlib
import os
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path

import pytest
from hub_fields import owner

os.environ["AFM_LITHO_TWIN_SPEED"] = "0"
# The goldens are v2 wire. hardware/wire.py reads this once at import, and a shell that
# runs the rig (v3) would otherwise compare the wrong wire. Spawned services inherit it.
# v3 is exercised by reload in test_wire_switch and tests/wire_v3, not by the shell env.
os.environ["AFM_LITHO_WIRE_VERSION"] = "2"

# Runtime isolation. `service/deps.py` fixes RUNTIME at import, so the process-wide
# folder must exist before any afm_litho import: a private copy of the checked-in
# model per process (each xdist worker imports this file and gets its own).
_FIXTURE_RUNTIME = Path(__file__).parent / "fixtures" / "runtime"
_USER_RUNTIME = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "afm-litho"


def _seed_runtime(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_FIXTURE_RUNTIME / "instrument.toml", dest / "instrument.toml")
    return dest


_SESSION_RUNTIME = _seed_runtime(Path(tempfile.mkdtemp(prefix="afm-litho-rt-")))
os.environ["AFM_LITHO_RUNTIME"] = str(_SESSION_RUNTIME)
atexit.register(shutil.rmtree, _SESSION_RUNTIME, ignore_errors=True)


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    if root.is_dir():
        for p in root.rglob("*"):
            try:
                st = p.stat()
            except OSError:
                continue
            out[str(p)] = (st.st_mtime_ns, st.st_size)
    return out


@pytest.fixture(autouse=True, scope="session")
def _user_runtime_untouched():
    """Fails the session if anything wrote under the per-user runtime folder."""
    before = _snapshot(_USER_RUNTIME)
    yield
    after = _snapshot(_USER_RUNTIME)
    assert after == before, "the suite touched the per-user runtime folder"


@pytest.fixture(autouse=True)
def _no_keeper_watch_carried_over():
    """The service watches one keeper pid in a module global. A test that leaves it on a
    keeper it then killed makes every later write on that worker refuse `keeper_lost`
    (a revived keeper's pid was left behind by test_keeper_revival_kr2)."""
    link = sys.modules.get("afm_litho.service.keeper_link")
    if link is not None:
        link.forget_keeper()
    yield


@pytest.fixture(autouse=True)
def _isolated_runtime(tmp_path_factory, monkeypatch):
    """Each test gets its own runtime folder seeded from the fixture model.

    The env var covers code that resolves the folder at call time. Modules that
    bound `RUNTIME` at import are re-pointed too; objects the process-global hub built
    at startup keep the session folder."""
    dest = _seed_runtime(tmp_path_factory.mktemp("runtime"))
    monkeypatch.setenv("AFM_LITHO_RUNTIME", str(dest))
    monkeypatch.delenv("AFM_LITHO_INSTRUMENT_CONFIG", raising=False)
    for name, mod in list(sys.modules.items()):
        if not name.startswith("afm_litho"):
            continue
        for attr in ("RUNTIME", "CONTACTS_DIR"):
            cur = getattr(mod, attr, None)
            if isinstance(cur, Path) and cur.parent == _SESSION_RUNTIME:
                monkeypatch.setattr(mod, attr, dest / cur.name)
            elif attr == "RUNTIME" and cur == _SESSION_RUNTIME:
                monkeypatch.setattr(mod, attr, dest)
    yield dest
    assert _USER_RUNTIME not in dest.parents


def start_flex_provider(fp, hub, tries: int = 25, **kwargs):
    """Start a FlexProvider on a port pair that is free at bind time.

    The provider binds `port` and `port + 1`; probing a pair and releasing it leaves a
    window in which another process can take either. The bind itself is the only
    reliable test, so a failed start is retried on a fresh pair."""
    import zmq

    last: Exception | None = None
    for _ in range(tries):
        prov = fp.FlexProvider(hub, bind="127.0.0.1", port=free_port_pair(),
                               log=lambda msg: None, **kwargs)
        try:
            return prov.start()
        except zmq.ZMQError as exc:
            last = exc
            with contextlib.suppress(Exception):
                prov.stop()
    raise RuntimeError(f"no free FLEX port pair after {tries} tries") from last


def free_port_pair() -> int:
    """A port p with p and p + 1 both free right now. Only a hint: start through
    `start_flex_provider` so a lost race retries."""
    for _ in range(50):
        a = socket.socket()
        b = socket.socket()
        try:
            a.bind(("127.0.0.1", 0))
            port = a.getsockname()[1]
            b.bind(("127.0.0.1", port + 1))
            return port
        except OSError:
            continue
        finally:
            a.close()
            b.close()
    raise RuntimeError("no free port pair")


@pytest.fixture(autouse=True)
def _clear_latched_fault():
    """Test isolation for the process-global hub's fault latch.

    A HARDWARE safe_park can now legitimately latch `hub.fault` (spec 13.1 item 7: an
    unverified INDEPENDENT-sensor retract is CRITICAL and must refuse every subsequent
    engage until an operator clears it). Before this fold the twin never latched, so no
    test ever left a fault behind; now a test that trips a hardware park (e.g. against a
    fake/sentinel backend that cannot retract) can, and the hub is a process global. A
    latched fault leaking into the next test would refuse its approach/scan/write with
    `fault_latched`. Clear it before each test -- every test that needs a fault sets its
    own. Only touches the hub if the module is already imported (pure unit tests that
    never stand up the app are untouched)."""
    import sys
    mod = sys.modules.get("afm_litho.service.hub")
    if mod is not None:
        try:
            mod.hub.fault = None
        except Exception:                                # noqa: BLE001,S110
            pass
    yield


@pytest.fixture(autouse=True)
def _reset_approach_counter(request):
    """hub._approach_in_flight > 0 makes getState answer "approaching", so a test that ends
    with its approach unfinished would stall every later waiter on the worker. Record the
    leaking test in <tempdir>/afm-approach-leaks.txt, then reset the counter."""
    yield
    import sys
    mod = sys.modules.get("afm_litho.service.hub")
    if mod is None:
        return
    n = getattr(mod.hub.host, "_approach_in_flight", 0)
    if n > 0:
        leaks = Path(tempfile.gettempdir()) / "afm-approach-leaks.txt"
        with open(leaks, "a", encoding="utf-8") as f:
            f.write(f"{request.node.nodeid} left _approach_in_flight={n}\n")
        mod.hub.host._approach_in_flight = 0


@pytest.fixture(autouse=True)
def _forget_saved_scans():
    """The hub's store is built once per session, so a scan frame a test saves lives in the
    session folder; drop it and the drift reference so the next test sees no scan cached."""
    yield
    import shutil
    import sys
    mod = sys.modules.get("afm_litho.service.hub")
    if mod is not None:
        scans = mod.hub.store.path.parent / "scans"
        if _SESSION_RUNTIME in scans.parents:
            shutil.rmtree(scans, ignore_errors=True)
        from afm_litho.service.drift import new_state
        mod.hub.drift = new_state()


@pytest.fixture(autouse=True, scope="module")
def _end_the_park_episode_between_modules():
    """A park left in force on the shared hub (an interlock trip, a stop) is ended when a
    module finishes, as a new engagement ends it (scan_cmds). The characterisation and golden
    rigs refuse to run with a park in force (tests/golden/harness.py), so a park leaked by
    whichever module ran before them on the same xdist worker would error their setup. No
    door clears a park alone: engaging moves the instrument."""
    yield
    import sys
    mod = sys.modules.get("afm_litho.service.hub")
    if mod is not None and isinstance(mod.hub.host._safe_park, dict):
        mod.hub.host._safe_park["active"] = False


@pytest.fixture(autouse=True)
def _drain_litho_pump():
    """Test isolation for the process-global hub's in-flight litho pump.

    A hardware/twin litho write starts a worker THREAD (`hub.litho_thread`) plus an
    async PUMP task (`hub._litho_pump_task`) that only AFTER the worker's
    `litho_done` records the session and persists the capture (host/litho_worker.py
    `_litho_finish`: forensic npz/gwy first, then the session record). The
    hw-litho route tests wait for the worker thread to die but NOT for the pump to
    finish -- so a test could return while its pump is still mid-`_litho_finish`.
    The next test on the same module-scoped `client` then starts a fresh write
    that resets the very accumulators the previous pump is still reading/writing
    (`hw_litho_capture`, `_litho_forensics`, `written_accum`), and the target
    `test_hardware_litho_pass_saves_loadable_capture` intermittently sees its
    capture never persisted ("capture not saved"). Draining the pump to completion
    after each test -- then clearing the litho accumulators -- makes every litho
    test start from a quiescent hub. Only touches the hub if its module is already
    imported (pure unit tests that never stand up the app are untouched)."""
    import sys

    yield
    mod = sys.modules.get("afm_litho.service.hub")
    if mod is None:
        return
    hub = mod.hub
    deadline = time.monotonic() + 10.0
    t = getattr(hub.host, "litho_thread", None)
    while t is not None and t.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    # The pump task runs on the TestClient's event-loop thread; poll `.done()`
    # (a threadsafe read) rather than awaiting it from this sync fixture.
    task = getattr(hub, "_litho_pump_task", None)
    while task is not None and not task.done() and time.monotonic() < deadline:
        time.sleep(0.02)
    try:
        hub.host.hw_litho_capture = {}
        hub._litho_forensics = None
        hub.host.written_accum = []
        hub.host._litho_partial = None
    except Exception:                                    # noqa: BLE001,S110
        pass


@pytest.fixture(autouse=True, scope="session")
def _flex_provider_off():
    """No TestClient startup may claim the real FLEX ports (see the module docstring).
    Session-scoped and autouse so it is set before the first app import."""
    prev = os.environ.get("AFM_LITHO_FLEX")
    os.environ["AFM_LITHO_FLEX"] = "0"
    yield
    if prev is None:
        os.environ.pop("AFM_LITHO_FLEX", None)
    else:
        os.environ["AFM_LITHO_FLEX"] = prev


@pytest.fixture(autouse=True, scope="session")
def _stop_port_ephemeral():
    """Every app a test starts puts its stop port on a port the OS picks, so xdist workers
    and a service running on this machine never fight over the fixed default."""
    prev = os.environ.get("AFM_LITHO_STOP_PORT")
    os.environ["AFM_LITHO_STOP_PORT"] = "0"
    yield
    if prev is None:
        os.environ.pop("AFM_LITHO_STOP_PORT", None)
    else:
        os.environ["AFM_LITHO_STOP_PORT"] = prev


@pytest.fixture(autouse=True, scope="session")
def _keeper_port_ephemeral():
    """A keeper a test starts takes a port the OS picks, and an in-process service has no
    keeper to tell (port 0 sends nothing), so no test reaches a keeper running here."""
    prev = os.environ.get("AFM_LITHO_KEEPER_PORT")
    os.environ["AFM_LITHO_KEEPER_PORT"] = "0"
    yield
    if prev is None:
        os.environ.pop("AFM_LITHO_KEEPER_PORT", None)
    else:
        os.environ["AFM_LITHO_KEEPER_PORT"] = prev


@pytest.fixture(autouse=True, scope="session")
def _design_autosave_off():
    """No test reads or writes `runtime/designs` unless it turns the autosave on
    against its own runtime directory. Session-scoped so it is set before the first
    app start, and inherited by any child service a test launches."""
    prev = os.environ.get("AFM_LITHO_DESIGN_AUTOSAVE")
    os.environ["AFM_LITHO_DESIGN_AUTOSAVE"] = "0"
    yield
    if prev is None:
        os.environ.pop("AFM_LITHO_DESIGN_AUTOSAVE", None)
    else:
        os.environ["AFM_LITHO_DESIGN_AUTOSAVE"] = prev


@pytest.fixture(scope="module")
def client(_flex_provider_off):
    from fastapi.testclient import TestClient

    from afm_litho.service.app import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _time_sensitive_clock(request):
    """A test marked `time_sensitive` checks timing itself (an abort, a stop or a park
    answered by another real thread during a wait): it keeps real waits, now recorded."""
    if not request.node.get_closest_marker("time_sensitive"):
        yield
        return
    from afm_litho import clock
    with clock.using(clock.TESTING_TIME_SENSITIVE):
        yield


@pytest.fixture(autouse=True)
def _instrument_model_restored():
    """An accepted override swaps `hub.instrument_model` for a new object (tip_amp_gain
    17.5, ...). Put the previous object back so no test sees another test's override,
    whichever xdist worker ran it first. One reference swap per test."""
    from afm_litho.service.hub import hub
    before = hub.host.instrument_model
    yield
    hub.host.instrument_model = before


@pytest.fixture(autouse=True)
def _scan_config_restored():
    """The AC guard reads the last scan's mode when it cannot read the instrument, so an AC
    scan left behind by one test refuses the next test's litho write on the same xdist
    worker. Put the scan config back after every test."""
    from afm_litho.service.hub import hub
    last, repeat = hub.host.last_scan_config, hub.host.scan_repeat.get("config")
    yield
    hub.host.last_scan_config = last
    hub.host.scan_repeat["config"] = repeat


def _twin_run_finished(hub) -> bool:
    """No write worker alive, no run claim held, no scan pump pending."""
    thread, task = hub.host.litho_thread, hub.host._scan_task
    return ((thread is None or not thread.is_alive()) and hub._run_claim is None
            and (task is None or task.done()))


@pytest.fixture(autouse=True)
def _interlocks_disarmed():
    """A test that ends with a run's interlocks armed (a scan or write it did not wait out)
    would make the next test on the same xdist worker see them armed. Disarm them once the
    test is over and nothing is running."""
    yield
    from afm_litho.service.hub import hub
    interlocks = getattr(hub.host, "write_interlocks", None)
    if interlocks is not None and interlocks.armed and _twin_run_finished(hub):
        from afm_litho.service.host.engage import disarm_interlocks
        disarm_interlocks(hub)


@pytest.fixture(autouse=True)
def _twin_plugin_dropped():
    """The hub is one object for the whole process, and its twin plugin outlives a test.
    The plugin withdraws on its own clock (2.5 s), so a test that aborts a twin write and
    moves on leaves the worker finishing, the interlocks armed and the claim held for the
    next test on the same xdist worker; a finished scan leaves the tip down and
    `hub.engaged` set, as it does for the operator. After every test: wait for the run to
    end, then drop the plugin so the next test starts with the tip off the surface."""
    yield
    from afm_litho.service.hub import hub
    from afm_litho.service.twin_plugin import forget
    twin = hub.host._twin_plugin
    if twin is None:
        return
    end = time.monotonic() + 30.0
    while not _twin_run_finished(hub) and time.monotonic() < end:
        time.sleep(0.02)
    if not _twin_run_finished(hub):
        import warnings
        warnings.warn("a twin run was still going 30 s after its test ended; dropping the "
                      "plugin anyway", stacklevel=1)
    if not twin.withdrawn():
        hub.host.engaged = False
    forget(hub)


@pytest.fixture
def litho_ready_hub(client):
    """Puts the shared hub's scan and engage fields in the idle state a litho write
    expects. Without it the AC guard reads whatever scan config an earlier test on the
    same xdist worker left behind. Afterwards the tip is left idle (not engaged, no
    engaged Z loop) and the scan fields are put back. Only these few fields are touched;
    the app and hub are never rebuilt."""
    from afm_litho.service.app import hub
    saved = {n: getattr(owner(hub, n), n) for n in ("last_scan_config", "imaging_mode")}
    saved_repeat_cfg = hub.host.scan_repeat.get("config")
    hub.host.last_scan_config = None
    hub.host.scan_repeat["config"] = None
    hub.host.engaged = False
    hub.host.imaging_mode = None
    hub.host.z_loop = {}                       # exists only after an approach
    yield hub
    for n, v in saved.items():
        setattr(owner(hub, n), n, v)
    hub.host.scan_repeat["config"] = saved_repeat_cfg
    hub.host.engaged = False
    hub.host.z_loop = {}


# The heavy groups, marked `slow` so `pytest -m "not slow"` is the quick run. Plain `pytest`
# still runs everything. Each file re-plays real product waits (settles, abort holds).
_SLOW_FILES = {
    "test_golden_scenarios.py", "test_golden_garbled_stop.py", "test_wire_tap_journal.py",
    "test_run_wire_check.py", "test_safe_state_matrix.py", "test_instrument_asylum_goldens.py",
}
_SLOW_DIRS = ("usertwin",)               # browser tests (Playwright)
_HW_LITHO_ROUTE_10S = (                  # test_hw_litho_route.py: the ~10 s writes
    "test_write_takes_hardware_branch_and_emits_contract",
    "test_hardware_litho_pass_saves_loadable_capture",
    "test_litho_write_allowed_in_contact_mode_on_hardware",
    "test_rest_relative_setpoint_is_resolved_against_the_runs_free_air",
    "test_flex_absolute_setpoint_is_used_as_sent",
    "test_engaged_write_with_an_old_baseline_is_accepted_and_reads_fresh_free_air",
    "test_write_record_carries_each_objects_free_air_and_setpoint",
)
_SLOW_TESTS = (                          # single tests over ~20 s: path under tests/ + "::" + name
    "test_design_autosave.py::test_every_chiral_wire_the_service_saved_is_restored"
    "[300 x 5 um, pen default]",
    "test_design_autosave.py::test_every_chiral_wire_the_service_saved_is_restored"
    "[60 x 90 um, no amplitude]",
    "journal/test_journal_questions.py::test_q3_script_steps_and_the_operators_withdraw",
    "doors/http/test_http_transcripts.py::test_http_door_transcript[design_edit]",
    "doors/http/test_http_transcripts.py::test_http_door_transcript[control]",
    "doors/http/test_http_transcripts.py::test_http_door_transcript[continuous_scan]",
    *(f"test_hw_litho_route.py::{n}" for n in _HW_LITHO_ROUTE_10S),
)

# The safety core stays in the fast run: one recording each of the Cypher abort, park,
# dead-man and interlock-trip goldens (no determinism repeat, no MFP-3D), the rail-stuck
# park, and the second-stop journal case. Marked `safety_core`, never `slow`.
_CORE_SCENARIOS = (
    "abort_in_transit", "abort_mid_stroke", "abort_at_chain_seam", "safe_park",
    "safe_park_com_down", "safe_park_vendor_withdraw", "deadman_during_write",
    "deadman_engaged_withdraw", "interlock_trip",
)
_SAFETY_CORE = (
    *(f"test_golden_scenarios.py::test_golden_matches_frozen_file[{n}]" for n in _CORE_SCENARIOS),
    "test_write_interlocks.py::test_twin_scan_with_z_rail_stuck_is_parked_with_rail_dwell",
    "journal/test_journal_questions_more.py::test_second_run_and_second_stop",
)


def _tests_relative(item) -> str:
    """The node id below the `tests/` folder, whatever the rootdir or separator."""
    node = item.nodeid.replace("\\", "/")
    return node.split("tests/", 1)[-1]


# klayout is the optional `geometry` extra; these tests run its boolean and sizing ops.
_NEEDS_KLAYOUT = (
    "test_api_design.py::test_design_op_union_merges_overlapping_regions",
    "test_api_design.py::test_design_op_difference_subtracts_one_region",
    "test_api_design.py::test_design_op_outset_and_inset_change_the_geometry",
    "test_behavior_design.py::test_outset_strictly_increases_area",
    "test_behavior_design.py::test_inset_strictly_decreases_area",
    "test_behavior_design.py::test_union_of_overlapping_rects_is_one_region_within_area_bounds",
    "test_behavior_design.py::test_difference_strictly_shrinks_the_first_shape",
    "test_flex_litho.py::test_loadpattern_from_a_design_name",
    "test_http_transcripts.py::test_http_door_transcript[design_edit]",
)


def _klayout_missing() -> bool:
    try:
        import klayout.db  # noqa: F401
    except ImportError:
        return True
    return False


def pytest_collection_modifyitems(config, items):
    skip_klayout = pytest.mark.skip(
        reason="klayout not installed (pip install -e .[geometry])")
    no_klayout = _klayout_missing()
    for item in items:
        rel = _tests_relative(item)
        if no_klayout and any(rel.endswith("/" + t) or rel == t for t in _NEEDS_KLAYOUT):
            item.add_marker(skip_klayout)
        path = Path(str(item.fspath))
        if any(rel.startswith(t) for t in _SAFETY_CORE):
            item.add_marker(pytest.mark.safety_core)
        elif (path.name in _SLOW_FILES or path.parent.name in _SLOW_DIRS
                or any(rel.startswith(t) for t in _SLOW_TESTS)):
            item.add_marker(pytest.mark.slow)


def pytest_addoption(parser):
    parser.addoption("--no-serial-host-api", action="store_true", default=False,
                     help="run without the JSON round trip on every HostApi call")
    parser.addoption("--freeze-host-api-log", action="store_true", default=False,
                     help="merge this run's HostApi call shapes into tests/golden/"
                          "host_api_calls.json")


def pytest_sessionfinish(session, exitstatus):
    log_dir = os.environ.get("AFM_HOST_API_LOG_DIR")
    if (session.config.getoption("--freeze-host-api-log")
            and not hasattr(session.config, "workerinput") and log_dir):
        import host_api_serial
        print(f"{host_api_serial.merge_log_dir(Path(log_dir))} HostApi call shapes frozen")


@pytest.fixture(autouse=True, scope="session")
def _serializing_host_api(request):
    """Every HostApi call and pass_ended crosses a JSON round trip (H7.5e). Test-only."""
    if request.config.getoption("--no-serial-host-api"):
        yield
        return
    import host_api_serial
    undo = host_api_serial.install()
    yield
    host_api_serial.uninstall(undo)


@pytest.fixture(autouse=True)
def _host_api_unserialized_marker(request):
    """A test marked host_api_unserialized hands the HostApi a stub object, not wire data."""
    if request.node.get_closest_marker("host_api_unserialized") is None:
        yield
        return
    import host_api_serial
    host_api_serial.ENABLED = False
    yield
    host_api_serial.ENABLED = True


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "host_api_unserialized: this test passes a stub object through the HostApi "
        "(an identity check), so the JSON round trip is off for it; list is in "
        "test_host_api_serial.EXEMPT_TESTS")
    if not hasattr(config, "workerinput") and not config.getoption("--no-serial-host-api"):
        if config.getoption("--freeze-host-api-log"):
            os.environ["AFM_HOST_API_FREEZING"] = "1"
            os.environ["AFM_HOST_API_LOG_DIR"] = tempfile.mkdtemp(prefix="afm-hostapi-log-")
    # Tests marked xdist_group (stop port, after-death, keeper) must share one worker under
    # -n. Set only when xdist is loaded: its --dist option does not exist without it, so an
    # addopts line would stop every run on a minimal venv.
    if config.pluginmanager.hasplugin("xdist") and config.option.dist in ("no", "load"):
        config.option.dist = "loadgroup"
    config.addinivalue_line(
        "markers",
        "expects_in_wave_violation: the test deliberately breaks the multiple-of-32 "
        "in-wave rule on the fake Igor")


@pytest.fixture(autouse=True)
def _fake_igor_in_wave_rule_holds(request, monkeypatch):
    """Fails any test whose fake Igor recorded a multiple-of-32 violation it did not
    declare. Real Igor drops such a bind silently, so the scan would stall for the arc
    timeout and an assert-light test would still pass."""
    from afm_litho.hardware import igor_stub

    made = []
    real_init = igor_stub.FakeIgorBridge.__init__

    def tracking_init(self, *a, **kw):
        real_init(self, *a, **kw)
        made.append(self)

    monkeypatch.setattr(igor_stub.FakeIgorBridge, "__init__", tracking_init)
    yield
    if request.node.get_closest_marker("expects_in_wave_violation"):
        return
    bad = [v for stub in made for v in stub.violations
           if v.rule == "in_wave_not_multiple_of_32"]
    assert not bad, f"fake Igor saw in-wave lengths that are not a multiple of 32: {bad}"
