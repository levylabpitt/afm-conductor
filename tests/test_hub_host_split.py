"""Every Hub field has exactly one owner: the door (Hub) or the instrument (HostState)."""

from __future__ import annotations

import pytest

from afm_litho.service.host_state import HOST_FIELD_NAMES
from afm_litho.service.hub import Hub

# Door, run-store and design state that stays on Hub, plus the control mixin's fields.
DOOR_FIELDS = {
    "host", "clients", "sample_name", "store", "llab", "surface", "pattern",
    "design_source", "ports", "ports_seq", "drift", "_show_tip_display",
    "_twin_tip_sent_at", "_twin_tip_task", "_loop", "transit",
    "litho_frame", "startup_note", "scan_params", "runs",
    "_control_lock", "control", "_revoked_tokens", "_run_claim",
    "control_watch", "_control_bg_tasks",
}
HOST_ONLY = [name for name in HOST_FIELD_NAMES if name != "fault"]


def test_each_field_has_one_owner():
    hub = Hub()
    door = set(vars(hub))
    host = set(vars(hub.host))
    assert door == DOOR_FIELDS
    assert host == set(HOST_FIELD_NAMES)
    assert len(HOST_FIELD_NAMES) == len(host) == 72     # none listed twice
    assert not door & host


def test_only_fault_has_a_hub_property():
    assert isinstance(Hub.fault, property)
    assert [name for name in HOST_ONLY if hasattr(Hub, name)] == []


@pytest.mark.parametrize("name", HOST_ONLY)
def test_a_host_field_round_trips_on_the_host_and_never_on_the_hub(name):
    hub = Hub()
    marker = object()
    hub.host.__dict__[name] = marker                    # bypass any HostState logic
    assert getattr(hub.host, name) is marker
    assert not hasattr(hub, name)                       # not readable on the hub
    with pytest.raises(AttributeError, match=f"hub.host.{name}"):
        setattr(hub, name, object())                    # nor writable there
    assert name not in vars(hub)
    assert getattr(hub.host, name) is marker


def test_fault_round_trips_both_ways_through_its_property():
    hub = Hub()
    hub.fault = {"reason": "t"}
    assert hub.host.fault == {"reason": "t"}
    hub.host.fault = {"reason": "u"}
    assert hub.fault == {"reason": "u"}
    hub.fault = None
    assert hub.host.fault is None
    assert "fault" not in vars(hub)


def test_door_fields_are_still_set_on_the_hub():
    hub = Hub()
    hub.pattern = "p"
    assert vars(hub)["pattern"] == "p"
