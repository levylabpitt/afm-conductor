"""AFM_LITHO_ARC_BACKEND (service/backends.py).

Covers the resolution matrix, the lenient fallback for an unrecognized value (the
pre-split AFM_LITHO_BACKEND never validated -- see hardware/backend.py's module
docstring) and that the getStatus / getCapabilities / IDN / /api/state fields land
without disturbing the existing ones.
"""

from __future__ import annotations

import logging

import pytest

from afm_litho.hardware import backend as hwbackend
from afm_litho.service import backends as svc_backends
from afm_litho.service import flex_dispatch as fd
from afm_litho.service.hub import Hub  # used below -- see the note on why it's imported here

# `Hub` is imported here, at module (collection) time, DELIBERATELY -- not only
# where it's used below. `afm_litho.service.hub` builds its process-wide
# singleton (`hub = Hub()`) as a side effect of being imported for the first
# time ANYWHERE in the session; a test that does that import for the first time
# from INSIDE a monkeypatched-env test body would permanently bake that env into
# the singleton every later test (including the `client` fixture's FastAPI app)
# reads. Importing at collection time, before any monkeypatching can run,
# builds the singleton under a clean environment instead.

_ENV_VARS = ("AFM_LITHO_BACKEND", "AFM_LITHO_ARC_BACKEND")


@pytest.fixture(autouse=True)
def _clean_backend_env(monkeypatch):
    """Every test starts with neither var set, regardless of what the
    outer shell exported -- the resolution logic is the thing under test. Also
    clears the module-level warned-once set (process-lifetime by design in
    production, but that would make an earlier test's warning silently suppress
    this test's -- see `hardware.backend._warned`)."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    hwbackend._warned.clear()


# ============================================================ resolution matrix ==

def test_unset_resolves_to_sim():
    assert svc_backends.arc_backend() == "sim"


def test_backend_alone_drives_arc(monkeypatch):
    monkeypatch.setenv("AFM_LITHO_BACKEND", "hardware")
    assert svc_backends.arc_backend() == "hardware"


def test_arc_backend_var_wins_over_base(monkeypatch):
    monkeypatch.setenv("AFM_LITHO_BACKEND", "sim")
    monkeypatch.setenv("AFM_LITHO_ARC_BACKEND", "hardware")
    assert svc_backends.arc_backend() == "hardware"


@pytest.mark.parametrize("alias", ["hardware", "HW", " Real ", "igor"])
def test_hardware_aliases_are_recognized(monkeypatch, alias):
    monkeypatch.setenv("AFM_LITHO_ARC_BACKEND", alias)
    resolved = svc_backends.arc_backend()
    assert svc_backends.is_hardware(resolved)
    # Normalized to the canonical string, not the alias verbatim: every downstream
    # gate (app.py's bridge bring-up, Hub.require_live_backend, Hub.backend_label)
    # is a literal `== "hardware"` comparison, not a membership test -- an alias
    # passed through unnormalized silently satisfied none of them (the bug this
    # guards against; see test_arc_hw_alias_actually_arms_the_hardware_gates below).
    assert resolved == "hardware"


def test_arc_hw_alias_actually_arms_the_hardware_gates(monkeypatch):
    """Integration check for the normalization above: a freshly constructed Hub
    under AFM_LITHO_ARC_BACKEND=hw (an alias, not the canonical spelling) must be
    genuinely hardware-CONFIGURED -- `backend_label()` reports the disconnected
    (not twin) label, and `require_live_backend()` refuses pre-connect -- exactly
    as AFM_LITHO_ARC_BACKEND=hardware would. Before the `_resolve` normalization
    fix, `hub.backend` held the literal 'hw', which satisfied neither gate: the
    bridge bring-up never started and the silent-twin refusal never fired."""
    monkeypatch.setenv("AFM_LITHO_ARC_BACKEND", "hw")
    hub = Hub()
    try:
        assert hub.host.backend == "hardware"
        assert hub.host.hardware_arc is None                  # never bound in this test
        assert hub.backend_live() == "twin"               # truthfully not live yet
        assert hub.backend_label() == "disconnected"       # NOT "twin" -- configured
        why = hub.require_live_backend()
        assert why is not None and "has not connected" in why
    finally:
        wd = getattr(hub.host, "_failsafe", None)
        if wd is not None:
            wd.stop()


def test_twin_is_a_recognized_sim_alias_and_does_not_warn(monkeypatch, caplog):
    """The flex-afm integration tests start the service with
    AFM_LITHO_BACKEND=twin -- that must keep working exactly as it did before
    the split, silently, not as a tolerated-but-noisy fallback."""
    monkeypatch.setenv("AFM_LITHO_BACKEND", "twin")
    with caplog.at_level(logging.WARNING, logger="afm_litho.hardware.backend"):
        assert svc_backends.arc_backend() == "sim"
        assert caplog.records == []


def test_unrecognized_value_falls_back_to_sim_with_one_warning(monkeypatch, caplog):
    """Not an error (the pre-split code never validated this variable either) --
    a clear one-line warning naming the bad value, then sim."""
    monkeypatch.setenv("AFM_LITHO_ARC_BACKEND", "bogus")
    with caplog.at_level(logging.WARNING, logger="afm_litho.hardware.backend"):
        resolved = svc_backends.arc_backend()
    assert resolved == "sim"
    assert not svc_backends.is_hardware(resolved)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "bogus" in warnings[0].message
    assert "AFM_LITHO_ARC_BACKEND" in warnings[0].message


def test_unrecognized_base_value_is_blamed_on_the_base_var_not_the_seam(monkeypatch, caplog):
    """A typo in the BASE variable, read via the ARC/MEASURE fallback, must be
    attributed to AFM_LITHO_BACKEND -- the variable actually set -- not to
    AFM_LITHO_ARC_BACKEND, which he never touched. And it must warn exactly ONCE
    even when arc_backend() is resolved twice."""
    monkeypatch.setenv("AFM_LITHO_BACKEND", "twn")     # typo for "twin"
    with caplog.at_level(logging.WARNING, logger="afm_litho.hardware.backend"):
        assert svc_backends.arc_backend() == "sim"
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, [r.message for r in warnings]
    assert "AFM_LITHO_BACKEND" in warnings[0].message
    assert "AFM_LITHO_ARC_BACKEND" not in warnings[0].message
    assert "twn" in warnings[0].message


def test_is_hardware_helper():
    assert svc_backends.is_hardware("hardware")
    assert svc_backends.is_hardware("HW")
    assert svc_backends.is_hardware(" real ")
    assert not svc_backends.is_hardware("sim")
    assert not svc_backends.is_hardware("twin")
    assert not svc_backends.is_hardware(None)
    assert not svc_backends.is_hardware("")


def test_backend_mode_unchanged_reads_only_the_base_variable(monkeypatch):
    """backend_mode() (hardware/backend.py) is kept verbatim for existing callers
    that read AFM_LITHO_BACKEND directly -- it must NOT fall through to the
    ARC var (it is one-directional: they fall back to BACKEND, not
    the reverse)."""
    monkeypatch.setenv("AFM_LITHO_ARC_BACKEND", "hardware")
    assert hwbackend.backend_mode() == "sim"             # AFM_LITHO_BACKEND unset


# ================================================= provider fields (getStatus etc) ==

# The exact key sets before this change -- a regression guard: if either of these
# shrinks (a key renamed or dropped) or the two new keys land somewhere unexpected,
# this fails loudly rather than a caller silently losing a field.
_STATUS_KEYS_BEFORE = {
    "state", "backend", "run_id", "run_kind", "progress", "detail",
    "safe_park", "commander", "deadman_armed", "bridge",
}
_CAPS_KEYS_BEFORE = {
    "microscope_id", "is_cypher", "backend", "modes", "max_outwave_points",
    "in_banks", "inline_cap_bytes", "method_bounds", "twin_only", "ports",
}
_IDN_KEYS = {"Manufacturer", "Model", "Serial Number", "Firmware", "Backend"}


def test_get_status_gains_exactly_the_arc_backend_field(client):
    from afm_litho.service.hub import hub as real_hub

    st = fd._h_get_status(real_hub, None)
    assert set(st) == _STATUS_KEYS_BEFORE | {"arc_backend"}
    assert st["arc_backend"] == real_hub.host.backend
    # Compatibility: the pre-existing "backend" field is untouched -- still the
    # ARC's LIVE label, not the raw configured value.
    assert st["backend"] == real_hub.backend_label()


def test_get_capabilities_gains_exactly_the_arc_backend_field(client):
    from afm_litho.service.hub import hub as real_hub

    caps = fd._h_get_capabilities(real_hub, None)
    assert set(caps) == _CAPS_KEYS_BEFORE | {"arc_backend"}
    assert caps["arc_backend"] == real_hub.host.backend
    assert caps["backend"] == real_hub.backend_label()


def test_idn_is_byte_identical_arc_backend_only(client):
    """IDN.Backend is the ARC backend (IDN is the instrument's identity)."""
    from afm_litho.service.hub import hub as real_hub

    idn = fd._h_idn(real_hub, None)
    assert set(idn) == _IDN_KEYS
    assert idn["Backend"] == real_hub.backend_label()


def test_api_state_reports_the_arc_backend(client):
    r = client.get("/api/state")
    assert r.status_code == 200
    body = r.json()
    assert "arc_backend" in body
    from afm_litho.service.hub import hub as real_hub
    assert body["arc_backend"] == real_hub.host.backend
    # Compatibility: the pre-existing "backend" field is untouched.
    assert body["backend"] == real_hub.backend_label()
