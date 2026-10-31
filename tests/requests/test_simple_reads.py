"""The simple paired reads (slice 2.2): rows, both doors, journal entries, and that the HELP
cards and the registry order the rows were taken from are untouched."""

from __future__ import annotations

import pytest

from afm_litho.service import flex_dispatch as fd
from afm_litho.service import journal
from afm_litho.service.hub import hub
from afm_litho.service.requests import flex_door, table

PAIRS = [("getInstrument", "/api/instrument"), ("getLastPark", "/api/park/last"),
         ("getEstimate", "/api/litho/estimate"), ("getMode", "/api/mode")]
HTTP_ONLY = {"takeControl", "getControl", "stateSnapshot", "lastScan", "drift", "connectPorts",
             "scanContinuous", "design", "designCapabilities", "designSvg",
             "designBackgroundChannels", "designFiles", "designReorder", "designAdd",
             "designPoints", "designBezierGet", "designBezier", "designOp", "designChiral",
             "designSvgImport", "designBackground", "designSave", "designOpen", "abortScan",
             "instrumentOverride", "instrumentReload", "instrumentOpenConfig", "faultClear",
             "keeperContact", "connectPortsAdd", "connectPortsUpdate", "connectPortsDelete",
             "connectPortsReplace", "driftReference", "driftEnable", "scanRelative",
             "lithoToolpath", "tipShow", "indexPage", "editCockpitPage", "optics", "simSpeed",
             "pens", "sketch", "sketchUndo", "sketchClear", "shapes", "shapeUpdate",
             "shapeToolpath", "transitGet", "transitSet", "lithoFrameGet", "lithoFrameSet",
             "lithoDemo", "lithoWritten", "scanParams", "forceCurve", "tuneThermal",
             "tuneSweep", "samples", "sampleLoad", "sampleUpload", "contactsGet",
             "contactsSet", "sessions", "sessionRecall"}
FLEX_ONLY = ["ACK", "IDN", "getCapabilities"]


class Recorder:
    def __init__(self):
        self.events = []

    def note(self, src, what, /, **fields):
        self.events.append((what, fields))


@pytest.fixture
def events(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(journal, "_running", rec)
    return rec.events


@pytest.mark.parametrize("verb", FLEX_ONLY + [v for v, _ in PAIRS])
def test_flex_read_goes_through_its_row_and_is_journalled(verb, events):
    flex_door.handle(hub, verb, None, role="read", ctx=None)
    assert [w for w, _ in events][:2] == ["request", "reply"]
    assert events[0][1]["row"] == table.flex_row(verb).name


@pytest.mark.parametrize("verb,path", PAIRS)
def test_http_read_is_served_by_the_row_and_journalled_as_operator(verb, path, client, events):
    reply = client.get(path)
    assert reply.status_code == 200
    ours = [f for w, f in events if w == "request" and f["row"] == table.flex_row(verb).name]
    assert ours and ours[0]["actor"] == "operator"
    same = flex_door.handle(hub, verb, None, role="read", ctx=None)
    assert reply.json() == same or verb == "getEstimate"


def test_rows_keep_the_registry_bound_and_card():
    for row in table.ROWS:
        if row.flex is None:
            continue
        m = fd.REGISTRY[row.flex.verb]
        assert row.flex.bound_s == m.bound_s and row.help == m.doc
    assert {r.name for r in table.ROWS if r.flex is None} == HTTP_ONLY


def test_registry_order_and_help_list_are_unchanged():
    names = list(fd.REGISTRY)
    assert names[:2] == ["ACK", "IDN"] and names.index("getCapabilities") < names.index(
        "getInstrument") < names.index("getLastPark") < names.index("getEstimate")
    assert flex_door.handle(hub, "HELP", None, role="read", ctx=None) == sorted(names)


def test_a_script_polling_a_moved_read_with_its_token_keeps_its_deadman_fed(monkeypatch):
    touched = []
    monkeypatch.setattr(hub, "touch_control", lambda tok: touched.append(tok))
    flex_door.handle(hub, "getMode", {"token": "t1"}, role="command", ctx=None)
    assert touched == ["t1"]


def test_the_read_socket_never_feeds_the_deadman(monkeypatch):
    touched = []
    monkeypatch.setattr(hub, "touch_control", lambda tok: touched.append(tok))
    flex_door.handle(hub, "getMode", {"token": "t1"}, role="read", ctx=None)
    assert touched == []


def test_a_token_holder_with_a_malformed_request_still_feeds_the_deadman(monkeypatch):
    from afm_litho.service.requests.table import Flex, Row
    touched = []
    monkeypatch.setattr(hub, "touch_control", lambda tok: touched.append(tok))

    def bad_intake(_p):
        raise ValueError("malformed")

    row = Row(name="probe", kind="read", control="none", handler=lambda r: r,
              flex=Flex(verb="probe", socket="read", bound_s=1.0, intake=bad_intake, gate="none"))
    monkeypatch.setattr(table, "ROWS", [row])
    with pytest.raises(ValueError):
        flex_door.handle(hub, "probe", {"token": "t9"}, role="command", ctx=None)
    assert touched == ["t9"]
