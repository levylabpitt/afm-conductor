"""The hardware write worker when the plugin reports a stopped write: the order it keeps.

Journal abort note, "end" step label, withdraw, then the forensic record (read with the park
reason as it was before the withdraw); the record is flushed before `litho_done`. The wire
side of the order (one withdraw, stamped with the end step) is proved by the golden
scenarios and tests/journal/test_step_outcomes.py.
"""

import importlib
import queue

import numpy as np
import pytest

from afm_litho.instrument import Recorded, Result
from afm_litho.instrument.asylum import AsylumInstrument
from afm_litho.litho.compiler import demo_pattern
from afm_litho.hardware import hw_approach
from afm_litho.service.host import litho_worker

safe_park = importlib.import_module("afm_litho.service.host.safe_park")

# The plugin write as shipped, kept before the fixture swaps it for a cut one.
_REAL_PLUGIN_WRITE = AsylumInstrument.write


@pytest.fixture
def worker(client, monkeypatch):
    """Runs the worker with a plugin whose write is cut, and records what happened in order."""
    from afm_litho.service.app import hub
    events = []
    monkeypatch.setattr(hub, "pattern", demo_pattern())
    monkeypatch.setattr(hub.host, "hardware_arc", object(), raising=False)
    monkeypatch.setattr(hub.host, "_safe_park_reason", "park: operator", raising=False)
    monkeypatch.setattr(hub, "fault", None)
    hub.host._hw_litho_abort.clear()       # a park in an earlier test leaves it set
    monkeypatch.setattr(hub.host, "hw_litho_capture", {})
    monkeypatch.setattr(litho_worker, "_emit_write_run", lambda *a, **k: None)
    monkeypatch.setattr(
        litho_worker, "_flush_forensics_sync",
        lambda runtime, cap, ctx, *, aborted, abort_reason:
            events.append(("flush", abort_reason, sorted(cap))) or ("stem", None))
    notes = litho_worker.StepNotes
    monkeypatch.setattr(notes, "abort", lambda self, reason: events.append(("abort", reason)))
    monkeypatch.setattr(notes, "stamp_step", lambda self, kind: events.append(("stamp", kind)))
    cut = Result("write", "aborted", "the pass answered ABORTED", landed=0.5,
                 recorded=(Recorded("Deflection", "write", np.ones(4)),))
    monkeypatch.setattr(AsylumInstrument, "write", lambda self, command: cut)

    def run(q=None):
        q = q or queue.Queue()
        try:
            litho_worker._hw_litho_worker(q, hub.pattern, 0.2, 1.0)
        finally:
            hub.host.engaged = False
        messages = []
        while not q.empty():
            messages.append(q.get_nowait())
        return messages

    return hub, events, run


def test_a_stopped_write_notes_labels_withdraws_then_flushes_the_original_reason(
        worker, monkeypatch):
    hub, events, run = worker
    monkeypatch.setattr(hw_approach, "hw_withdraw",
                        lambda arc: events.append("withdraw") or {"z_v": 0.0})
    messages = run()
    assert events == [("abort", "the pass answered ABORTED"), ("stamp", "end"), "withdraw",
                      ("flush", "park: operator", ["Deflection", "output_a_cmd_v"])]
    done = [m for m in messages if m["type"] == "litho_done"]
    assert messages[-1] is done[0] and done[0]["aborted"] is True and done[0]["error"] is None


def test_a_failed_withdraw_keeps_the_original_reason_and_todays_texts(worker, monkeypatch):
    hub, events, run = worker
    parks = []

    def failing_withdraw(arc):
        events.append("withdraw")
        raise OSError("bridge gone")

    def park(hub_, reason, how):
        parks.append(reason)
        hub_.host._safe_park_reason = reason       # what the real request_park does

    monkeypatch.setattr(hw_approach, "hw_withdraw", failing_withdraw)
    monkeypatch.setattr(safe_park, "request_park", park)
    messages = run()
    text = "end-of-run withdraw failed: bridge gone"
    assert parks == ["litho " + text]
    assert hub.fault["detail"] == "litho " + text
    assert events[-1] == ("flush", "park: operator", ["Deflection", "output_a_cmd_v"])
    done = next(m for m in messages if m["type"] == "litho_done")
    assert done["aborted"] is True and done["error"] == text


class StopsOnObjectStart(queue.Queue):
    """A park lands after the worker's own abort check, as the first object starts."""

    def __init__(self, hub):
        super().__init__()
        self.hub = hub

    def put(self, item, *args, **kwargs):
        if item.get("type") == "litho_object_start":
            self.hub.host._hw_litho_abort.set()
        super().put(item, *args, **kwargs)


def test_a_stop_landing_as_a_stroke_starts_keeps_the_park_reason(worker, monkeypatch):
    hub, events, run = worker
    refused = Result("write", "refused", "an abort or park was signalled while the write "
                     "was starting")
    monkeypatch.setattr(AsylumInstrument, "write", lambda self, command: refused)
    monkeypatch.setattr(hub.host, "_safe_park_reason", "deadman expired", raising=False)
    monkeypatch.setattr(hw_approach, "hw_withdraw", lambda arc: {"z_v": 0.0})
    try:
        messages = run(StopsOnObjectStart(hub))
    finally:
        hub.host._hw_litho_abort.clear()
    assert ("flush", "deadman expired", []) in events
    done = next(m for m in messages if m["type"] == "litho_done")
    assert done["aborted"] is True and done["error"] is None


def test_an_unconfirmed_engage_stops_the_write_and_the_worker_withdraws(worker, monkeypatch):
    """EngageNotConfirmed raised by the run goes through the real plugin write
    (`_write_stopped`; the service owns the withdraw) to the worker's one verified
    end-of-run withdraw, and the reason reaches litho_done."""
    from types import SimpleNamespace

    import afm_litho.hardware.lithography_run as LR
    from afm_litho.hardware.engage_check import EngageNotConfirmed
    hub, events, run = worker
    monkeypatch.setattr(AsylumInstrument, "write", _REAL_PLUGIN_WRITE)
    monkeypatch.setattr(hub.host, "hardware_arc", SimpleNamespace(
        link=None, router=None, com_igor=None, z_out_channel="Height"), raising=False)
    monkeypatch.setattr(hub.host, "hw_cal", {}, raising=False)
    fired = []

    def no_engage(self, name, *a, **k):
        fired.append(name)
        raise EngageNotConfirmed(f"write {name}: tip did not engage within 6 s")

    monkeypatch.setattr(LR.LithographyRun, "write_object", no_engage)
    monkeypatch.setattr(hw_approach, "hw_withdraw",
                        lambda arc: events.append("withdraw") or {"z_v": 0.0})
    messages = run()
    assert len(fired) == 1 and events.count("withdraw") == 1
    done = next(m for m in messages if m["type"] == "litho_done")
    assert done["aborted"] is True and "did not engage" in done["error"]
    assert "EngageNotConfirmed" in done["error"]
