"""The litho contact-only guard: the AUTHORITATIVE crosspoint mode read-back.

SAFETY-CRITICAL (a tip-crash guard). See docs/history/2026-08-31-LITHO_AC_GUARD_FIX.md. The observed
bug: abort an AC scan -> contact approach (engaged on Deflection) -> litho was
REFUSED `ac_mode_active`, because the guard consulted the STALE AC scan config even
though the instrument was genuinely in contact. The fix reads the real hardware
crosspoint (`READXPT` -> `XPT InFast=<value>`): FilterOut = contact, ACDefl = AC.

Every reply/mock here stands in for `td_ReadGroup("Crosspoint", w)`, which cannot be
exercised off the instrument (no XOP on the dev box -- that half is RIG-PENDING); the
Python logic is fully covered with mocks and the fake Igor bridge.
"""

from __future__ import annotations

import contextlib
import time

from _im_fixtures_guard import GUARD_WRITE_KW, seed_free_air
from hub_fields import owner

from afm_litho.hardware import bridge_protocol as P

# `client` comes from tests/conftest.py (a TestClient).


def _guard_write_body(hub, **extra):
    """The REST `/api/litho/write` body under the command guard
    (INSTRUMENT_MODEL_SPEC secs 2/10, 13): a fresh withdrawn free-air baseline
    seeded on the hub, the enabled objects named explicitly (replacing `all:
    true`), plus `setpoint_relative` and `amp_gain`. The AC crosspoint refusal
    under test is raised PAST the guard, so the guard must pass first."""
    seed_free_air(hub)
    ids = [o.name for o in hub.pattern.objects if getattr(o, "enabled", True)]
    return {"objects": ids, "setpoint_relative": 0.2, **GUARD_WRITE_KW, **extra}


# ============================================================ protocol ========
# READXPT build + XPT parse round-trip: the string-returning read-back verb.

def test_readxpt_command_build():
    assert P.cmd_readxpt("7") == "READXPT 7 Crosspoint"
    assert P.cmd_readxpt("7", "Crosspoint") == "READXPT 7 Crosspoint"


def test_xpt_build_parse_roundtrip():
    # bridge -> driver: msg_xpt serializes, parse_xpt (on the router's post-uid rest)
    # reads it back to the same dict.
    wire = P.msg_xpt("12", {"InFast": "FilterOut"})
    assert wire == "XPT 12 InFast=FilterOut;"
    verb, _, tail = wire.partition(" ")
    uid, _, rest = tail.partition(" ")
    assert verb == "XPT" and uid == "12"
    assert P.parse_xpt(rest) == {"InFast": "FilterOut"}


def test_parse_xpt_multi_and_tolerant():
    assert P.parse_xpt("InFast=ACDefl;InA=FilterOut;") == {
        "InFast": "ACDefl", "InA": "FilterOut"}
    assert P.parse_xpt("") == {}                       # empty payload -> {}
    assert P.parse_xpt("   ") == {}
    assert P.parse_xpt("garbage;InFast=FilterOut;") == {"InFast": "FilterOut"}


# ==================================================== check_mode mapping ======
# HardwareARC.check_mode keys on InFast: FilterOut -> contact, ACDefl -> ac, else
# unknown. Exercised end to end over the fake bridge (real read_crosspoint path).

def _arc_over_stub(crosspoint):
    from afm_litho.hardware.hardware_arc import HardwareARC
    from afm_litho.hardware.igor_stub import FakeIgorBridge
    stub = FakeIgorBridge({}, done_delay=0.0)
    stub.crosspoint = dict(crosspoint)
    port = stub.serve_in_thread()
    arc = HardwareARC.connect("127.0.0.1", port)
    return arc, stub


def test_read_crosspoint_and_check_mode_over_stub():
    for xpt, mode in (
            ({"InFast": "FilterOut"}, "contact"),
            ({"InFast": "ACDefl"}, "ac"),
            ({"InFast": "Something"}, "unknown"),      # unexpected route -> unknown
            ({}, "unknown"),                            # empty crosspoint -> unknown
    ):
        arc, stub = _arc_over_stub(xpt)
        try:
            assert arc.read_crosspoint() == xpt
            assert arc.check_mode() == mode
        finally:
            arc.close()
            stub.stop()


def test_check_mode_case_insensitive():
    arc, stub = _arc_over_stub({"InFast": "filterout"})
    try:
        assert arc.check_mode() == "contact"
    finally:
        arc.close()
        stub.stop()


def test_read_crosspoint_failure_safe_returns_empty():
    # No router (the Protocol-only path) -> {} without raising; a closed link -> {}.
    from afm_litho.hardware.hardware_arc import HardwareARC
    arc = HardwareARC(link=None)                       # router is None
    assert arc.read_crosspoint() == {}
    assert arc.check_mode() == "unknown"

    arc2, stub = _arc_over_stub({"InFast": "FilterOut"})
    stub.stop()
    arc2.close()
    # link closed -> read must still not raise, and must not fabricate a mode
    assert arc2.read_crosspoint() == {}
    assert arc2.check_mode() == "unknown"


# ==================================================== the guard logic =========
# _ac_mode_active priority: (1) hardware crosspoint read; (2) engaged z_loop mode;
# (3) not-engaged conservative fallback (imaging_mode / current_scan_is_ac).

class _FakeARC:
    """Stands in for hub.hardware_arc: check_mode returns a fixed verdict so the
    guard's priority order can be exercised without a live bridge."""

    def __init__(self, mode="unknown"):
        self._mode = mode

    def check_mode(self):
        return self._mode

    def read_crosspoint(self, group="Crosspoint"):
        return {}


@contextlib.contextmanager
def _hub_state(**attrs):
    """Snapshot and restore the hub attributes the guard reads, so these process-
    global mutations never leak into other tests."""
    from afm_litho.service.app import hub
    keys = ("hardware_arc", "engaged", "imaging_mode", "z_loop", "last_scan_config")
    saved = {k: getattr(owner(hub, k), k, None) for k in keys}
    saved_cfg = hub.host.scan_repeat.get("config")
    # a clean slate: no continuous run, no stale single scan
    hub.host.scan_repeat["config"] = None
    hub.host.last_scan_config = None
    hub.host.engaged = False
    hub.host.imaging_mode = None
    hub.host.z_loop = {}
    hub.host.hardware_arc = None
    for k, v in attrs.items():
        setattr(owner(hub, k), k, v)
    try:
        yield hub
    finally:
        for k, v in saved.items():
            setattr(owner(hub, k), k, v)
        hub.host.scan_repeat["config"] = saved_cfg


def _guard():
    from afm_litho.service.routers.litho import _ac_mode_active
    return _ac_mode_active()


def test_guard_hardware_ac_refuses():
    # (1) an AC crosspoint refuses, even if nothing else says AC.
    with _hub_state(hardware_arc=_FakeARC("ac")):
        assert _guard() is True


def test_guard_hardware_contact_allows_over_stale_ac():
    # (1) a contact crosspoint ALLOWS -- and OVERRIDES a stale AC z_loop AND a stale AC
    # scan config (the authoritative read wins). This is the exact bug scenario.
    from afm_litho.service.app import hub
    with _hub_state(hardware_arc=_FakeARC("contact"), engaged=True,
                    z_loop={"mode": "ac"}):
        hub.host.last_scan_config = {"mode": "ac"}
        assert _guard() is False


def test_guard_bug_regression_contact_engage_after_ac_scan():
    # (2) crosspoint UNKNOWN (read failed) + a contact engage active + a STALE AC scan
    # config -> ALLOWED. The engaged contact mode clears the guard; the stale AC scan
    # config no longer overrides it (the fix). This is the reported rig failure.
    from afm_litho.service.app import hub
    with _hub_state(hardware_arc=_FakeARC("unknown"), engaged=True,
                    z_loop={"mode": "contact"}):
        hub.host.last_scan_config = {"mode": "ac"}          # the earlier aborted AC scan
        assert _guard() is False


def test_guard_engaged_ac_refuses_when_read_unknown():
    # (2) crosspoint unknown but an AC engage is active -> refuse.
    with _hub_state(hardware_arc=_FakeARC("unknown"), engaged=True,
                    z_loop={"mode": "ac"}):
        assert _guard() is True


def test_guard_not_engaged_imaging_mode_ac_refuses():
    # (3) not engaged, read unknown, commanded imaging mode AC -> conservative refuse.
    with _hub_state(hardware_arc=_FakeARC("unknown"), engaged=False,
                    imaging_mode="ac"):
        assert _guard() is True


def test_guard_not_engaged_stale_ac_scan_refuses():
    # (3) not engaged, read unknown, last scan config AC -> conservative refuse (there
    # is no engage to trust, so we cannot rule out an AC crosspoint).
    from afm_litho.service.app import hub
    with _hub_state(hardware_arc=_FakeARC("unknown"), engaged=False):
        hub.host.last_scan_config = {"mode": "ac"}
        assert _guard() is True


def test_guard_all_contact_allows():
    with _hub_state(hardware_arc=_FakeARC("unknown"), engaged=False,
                    imaging_mode="contact"):
        assert _guard() is False


def test_guard_twin_backend_uses_fallbacks():
    # Twin: no hardware_arc -> check_mode "unknown" -> fall to the fallbacks. Contact
    # engage allows; an AC engage refuses; a stale AC scan config while not engaged
    # refuses. (No crosspoint exists on the twin.)
    from afm_litho.service.app import hub
    from afm_litho.service.routers.litho import check_mode
    with _hub_state(hardware_arc=None, engaged=True, z_loop={"mode": "contact"}):
        assert check_mode() == "unknown"
        hub.host.last_scan_config = {"mode": "ac"}
        assert _guard() is False                        # contact engage wins
    with _hub_state(hardware_arc=None, engaged=True, z_loop={"mode": "ac"}):
        assert _guard() is True


# ============================================= end-to-end regression ==========
# The exact bug through the real /api/litho/write stack over the fake bridge: a
# contact crosspoint + a STALE AC scan config -> the write PROCEEDS (was refused).

def _tiny_pattern():
    from afm_litho.litho.compiler import LithoObject, Pattern
    return Pattern(name="ac-guard-test", objects=[
        LithoObject("wire", "path", [(-1.0, 0.0), (1.0, 0.0)], voltage=6.0,
                    speed_um_s=8.0, interpolation_um=0.1)])


def _wait(cond, timeout=20.0, interval=0.02):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return False


def test_write_proceeds_contact_crosspoint_despite_stale_ac_scan(client, tmp_path):
    from afm_litho.hardware.hardware_arc import HardwareARC
    from afm_litho.hardware.igor_stub import FakeIgorBridge
    from afm_litho.service.app import hub
    from afm_litho.service.sessions import SessionStore

    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    stub = FakeIgorBridge({}, done_delay=0.0)
    stub.crosspoint = {"InFast": "FilterOut"}          # AUTHORITATIVE: contact
    port = stub.serve_in_thread()
    hub.host.hardware_arc = HardwareARC.connect("127.0.0.1", port)
    hub.host.hw_cal = {"x_lvdt_sens": 7.46e-6, "x_lvdt_offset": -0.63,
                  "y_lvdt_sens": 7.04e-6, "y_lvdt_offset": 0.07,
                  "z_lvdt_sens": 5e-6, "z_lvdt_offset": 0.0}
    hub.pattern = _tiny_pattern()
    hub.host.written_objects = set()
    hub.host.engaged = True
    hub.host.z_loop = {"mode": "contact"}                   # a contact engage is active
    hub.host.last_scan_config = {"mode": "ac"}              # ... after an aborted AC scan
    try:
        with client.websocket_connect("/ws") as ws:
            ws.send_text("ping")
            for _ in range(200):
                if ws.receive_json().get("type") == "telemetry":
                    break
            r = client.post("/api/litho/write", json=_guard_write_body(hub))
            body = r.json()
            assert body.get("backend") == "hardware", r.text
            assert body["ok"] is True                   # NOT refused by ac_mode_active
            end = time.monotonic() + 20.0
            while time.monotonic() < end:
                if ws.receive_json().get("type") == "litho_done":
                    break
    finally:
        with contextlib.suppress(Exception):
            hub.host.hardware_arc.close()
        hub.host.hardware_arc = None
        hub.host.hw_cal = {}
        hub.host.engaged = False
        hub.host.z_loop = {}
        hub.host.last_scan_config = None
        stub.stop()
        _wait(lambda: not (hub.host.litho_thread and hub.host.litho_thread.is_alive()))


def test_write_refused_ac_crosspoint(client, tmp_path):
    from afm_litho.hardware.hardware_arc import HardwareARC
    from afm_litho.hardware.igor_stub import FakeIgorBridge
    from afm_litho.service.app import hub
    from afm_litho.service.sessions import SessionStore

    hub.store = SessionStore(tmp_path / "sessions.jsonl")
    stub = FakeIgorBridge({}, done_delay=0.0)
    stub.crosspoint = {"InFast": "ACDefl"}             # AUTHORITATIVE: AC
    port = stub.serve_in_thread()
    hub.host.hardware_arc = HardwareARC.connect("127.0.0.1", port)
    hub.pattern = _tiny_pattern()
    hub.host.written_objects = set()
    # even a CONTACT engage cannot override an AC crosspoint read (authoritative first)
    hub.host.engaged = True
    hub.host.z_loop = {"mode": "contact"}
    try:
        r = client.post("/api/litho/write", json=_guard_write_body(hub))
        body = r.json()
        assert body.get("reason") == "ac_mode_active", r.text
        assert not (getattr(hub.host, "litho_thread", None) and hub.host.litho_thread.is_alive())
        assert hub.host.written_objects == set()
    finally:
        with contextlib.suppress(Exception):
            hub.host.hardware_arc.close()
        hub.host.hardware_arc = None
        hub.host.engaged = False
        hub.host.z_loop = {}
        stub.stop()
