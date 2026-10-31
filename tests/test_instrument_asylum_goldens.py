"""The Asylum plugin on the golden rig: it must send Igor exactly what the service sends today.

Each test drives the plugin through the same stub as the golden scenario of the same name,
with the same objects and frame, and judges the recorded wire against the frozen golden file
the way test_golden_scenarios.py does (Cypher: both levels; MFP-3D: the wire level).
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from golden.harness import (
    _RECORDING_LINK_TIMEOUT_S,
    HERE,
    MFP3D_MICROSCOPE_ID,
    GoldenRig,
    compare_wire,
)
from golden.recordings import golden_group
from golden.scenarios import AC_CROSSPOINT, PAD, WIRE, WIRE_B, WIRE_LONG, _acquire, _release

from afm_litho import clock
from afm_litho.goldens import golden_format as G
from afm_litho.instrument import (Citation, ReadChannels, ScanRaster, WithdrawTip,
                                  WriteObject)
from afm_litho.instrument.asylum import AsylumInstrument
from afm_litho.run.model import ScanFrame, Setpoint

RUN = "w-2026-10-26-1200-00"
AMP_GAIN = 25.0
RECORD = ("ZSensor", "Deflection", "UserIn2")


@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


def guard_read(asylum):
    """The service's SUM-ceiling guard reads DetectorSum before it starts a write; the plugin
    leaves the guards to the caller, so a scenario that must carry that read sends it."""
    assert asylum.read(ReadChannels(("DetectorSum",))).done


ABSOLUTE_ZERO = Setpoint("absolute", 0.0)


def write_commands(hub, door, objects, setpoint=ABSOLUTE_ZERO, steps=None):
    """The WriteObject for each stroke of `objects`, compiled as the service's write worker
    compiles them (host/litho_worker.py _hw_litho_worker)."""
    from afm_litho.litho.compiler import compile_pattern
    from afm_litho.service.routers.litho import _litho_frame
    token = _acquire(door)
    door.call("loadPattern", {"token": token, "objects": [dict(o) for o in objects]})
    _release(door, token)
    world = hub.host.world
    waves = compile_pattern(hub.pattern, world.x_lvdt_sens, world.y_lvdt_sens,
                            world.x_lvdt_offset, world.y_lvdt_offset, frame=_litho_frame())
    speeds = {o.name: float(o.speed_um_s) for o in hub.pattern.objects}
    commands = []
    for n, meta in enumerate(waves["segments"]):
        path = np.asarray(meta["path_um"], dtype=float)
        commands.append(WriteObject(
            meta["name"], path[:, 0], path[:, 1], np.asarray(meta["v"], dtype=float),
            speeds.get(meta["object"], 2.0), setpoint, RECORD, meta["object"],
            steps=() if steps is None else steps(n)))
    return commands


def scan_frame():
    return ScanFrame(source="golden", size_um=1.0, lines=2, pixels=32, angle_deg=0.0,
                     offset_um=(0.0, 0.0), drift_offset=False, direction="up",
                     line_rate_hz=50.0, trace_and_retrace=True)


def drive(hub, scenario, work, *, microscope_id=None, telem_hz=0.0):
    """Run `work(rig, plugin)` on a fresh golden rig on the testing clock; return the golden."""
    with clock.using(clock.TESTING), GoldenRig(
            hub, scenario, telem_hz=telem_hz, microscope_id=microscope_id,
            link_timeout_s=_RECORDING_LINK_TIMEOUT_S) as rig:
        asylum = AsylumInstrument(hub, amp_gain=AMP_GAIN)
        try:
            work(rig, asylum)
        finally:
            asylum.close()
        rig.quiesce()
        return rig.golden() if microscope_id is None else rig.golden_wire()


def judge(scenario, golden, mfp=False):
    """The same comparison test_golden_scenarios.py makes against the frozen file."""
    path = HERE / (f"{scenario}.mfp3d.golden.json" if mfp else f"{scenario}.golden.json")
    frozen = G.load(path)
    problems = compare_wire(frozen, golden) if mfp else G.compare(frozen, golden)
    assert not problems, "\n".join(problems)


def write_objects(objects, scenario):
    """The plugin writes each stroke, then the service's end-of-run withdraw."""
    def work(rig, asylum):
        guard_read(asylum)
        for command in write_commands(rig.hub, rig.door, objects):
            result = asylum.write(command)
            assert result.done, result.reason
        assert asylum.withdraw(WithdrawTip()).verified is not None
    return work


def scan_work(mode, record=("ZSensor", "Deflection")):
    def work(rig, asylum):
        guard_read(asylum)
        if mode == "ac":
            rig.inst.set_crosspoint(AC_CROSSPOINT)
        result = asylum.scan(ScanRaster(scan_frame(), mode, Setpoint("absolute", 0.2), False,
                                        record))
        assert result.done, result.reason
        shape = (2, 32)
        assert [r.part for r in result.recorded] == ["trace", "retrace"] * len(record)
        assert all(r.volts.shape == shape for r in result.recorded)
        # The service ends a scan as it ends a write run: the routine withdraw.
        assert asylum.withdraw(WithdrawTip()).verified is not None
    return work


WRITES = {"litho_single": [WIRE], "litho_multi_object": [PAD, WIRE],
          "litho_chain_two_chunks": [WIRE_LONG]}
SCANS = {"scan_chained_contact": "contact", "scan_chained_ac": "ac"}


@pytest.mark.parametrize("name", [pytest.param(n, marks=golden_group(n)) for n in WRITES])
def test_writes_send_the_golden_wire_on_a_cypher(hub, client, name):
    judge(name, drive(hub, name, write_objects(WRITES[name], name)))


@pytest.mark.parametrize("name", [pytest.param(n, marks=golden_group(n))
                                  for n in ("litho_single", "litho_multi_object")])
def test_writes_send_the_golden_wire_on_an_mfp3d(hub, client, name):
    golden = drive(hub, name, write_objects(WRITES[name], name),
                   microscope_id=MFP3D_MICROSCOPE_ID)
    judge(name, golden, mfp=True)


@pytest.mark.parametrize("name", [pytest.param(n, marks=golden_group(n)) for n in SCANS])
def test_scans_send_the_golden_wire(hub, client, name):
    record = ("ZSensor", "Amplitude") if SCANS[name] == "ac" else ("ZSensor", "Deflection")
    judge(name, drive(hub, name, scan_work(SCANS[name], record)))


@pytest.mark.parametrize("name", [pytest.param(n, marks=golden_group(n)) for n in SCANS])
def test_scans_send_the_golden_wire_on_an_mfp3d(hub, client, name):
    record = ("ZSensor", "Amplitude") if SCANS[name] == "ac" else ("ZSensor", "Deflection")
    golden = drive(hub, name, scan_work(SCANS[name], record), microscope_id=MFP3D_MICROSCOPE_ID)
    judge(name, golden, mfp=True)


@pytest.mark.parametrize("name,hold", [("abort_mid_stroke", ("ARMFIRE", 1)),
                                       ("abort_in_transit", ("FIRE", 2))])
def test_an_abort_mid_write_sends_the_golden_wire(hub, client, name, hold):
    objects = [WIRE] if name == "abort_mid_stroke" else [WIRE, WIRE_B]
    outcome = {}

    def work(rig, asylum):
        commands = write_commands(rig.hub, rig.door, objects)
        guard_read(asylum)
        rig.inst.hold_next(*hold)

        def writer():
            for command in commands:
                outcome["result"] = asylum.write(command)
                if not outcome["result"].done:
                    return

        thread = threading.Thread(target=writer)
        thread.start()
        rig.inst.wait_held()
        asylum.abort("golden abort")
        thread.join(60)
        assert not thread.is_alive()

    judge(name, drive(hub, name, work))
    assert outcome["result"].outcome == "aborted"
    assert "abort" in outcome["result"].reason


def test_a_two_object_write_reports_each_step_with_its_citation(hub, client):
    statuses, frames = [], []
    # object 1: move, settle, read, engage, pass, withdraw; object 2 (a new source): the same
    # without the settle
    plans = [(0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10)]

    def work(rig, asylum):
        asylum.listen(frames.append, statuses.append)
        commands = write_commands(
            rig.hub, rig.door, [WIRE, WIRE_B],
            steps=lambda n: tuple(Citation(RUN, s) for s in plans[n]))
        for command in commands:
            assert asylum.write(command).done

    drive(hub, "litho_multi_object", work)
    cited = [(s.activity, s.cite.step) for s in statuses if s.cite is not None]
    assert cited == [("moving", 0), ("moving", 1), ("reading", 2), ("engaged", 3),
                     ("writing", 4), ("withdrawing", 5),
                     ("moving", 6), ("reading", 7), ("engaged", 8), ("writing", 9),
                     ("withdrawing", 10)]
    assert [s.text for s in statuses if s.text.startswith("writing object")] == [
        "writing object wire", "writing object wire_b"]


def test_a_hover_stroke_reports_four_steps(hub, client):
    statuses = []

    def work(rig, asylum):
        asylum.listen(lambda f: None, statuses.append)
        commands = write_commands(rig.hub, rig.door, [PAD])
        plans = [(0, 1, 2, 3, 4, 5), (6, 7, 8, 9)]
        for command, plan in zip(commands[:2], plans):
            command = WriteObject(command.name, command.x_um, command.y_um, command.tip_v,
                                  command.speed_um_s, command.setpoint, command.record,
                                  command.source, tuple(Citation(RUN, s) for s in plan))
            assert asylum.write(command).done

    drive(hub, "litho_multi_object", work)
    hover = [(s.activity, s.cite.step, s.text) for s in statuses
             if s.cite is not None and s.cite.step >= 6]
    assert [h[:2] for h in hover] == [("moving", 6), ("engaged", 7), ("writing", 8),
                                      ("withdrawing", 9)]
    assert "hover" in hover[0][2]


def test_a_write_returns_the_free_air_read_and_the_captured_channels(hub, client):
    out = {}

    def work(rig, asylum):
        (command,) = write_commands(rig.hub, rig.door, [WIRE])
        rig.stub.capture = {name: [0.5 + 0.01 * i] * 32 for i, name in enumerate(RECORD)}
        out["result"] = asylum.write(command)

    drive(hub, "litho_single", work)
    result = out["result"]
    assert result.done and {r.channel for r in result.readings} == {"Deflection", "Height"}
    assert [r.channel for r in result.recorded] == list(RECORD), result.reason
    assert all(r.part == "write" and r.volts.shape == (32,) for r in result.recorded)
    assert result.recorded[1].volts[0] == pytest.approx(0.51)


def test_telemetry_frames_arrive_in_micrometres_and_the_service_still_gets_them(hub, client):
    frames, snapshots = [], []

    def work(rig, asylum):
        asylum.listen(frames.append, lambda status: None)
        end = time.monotonic() + 10
        while len(frames) < 3 and time.monotonic() < end:
            time.sleep(0.02)
        snapshots.append(dict(hub.host._hw_telemetry))
        cal = dict(hub.host.hw_cal)
        values = rig.values
        snapshots.append({
            "x_um": (values["XSensor"] - cal["x_lvdt_offset"]) * cal["x_lvdt_sens"] * 1e6,
            "y_um": (values["YSensor"] - cal["y_lvdt_offset"]) * cal["y_lvdt_sens"] * 1e6})

    drive(hub, "show_tip", work, telem_hz=50.0)
    assert len(frames) >= 3
    expected = snapshots[1]
    assert frames[-1].x_um == pytest.approx(expected["x_um"], abs=1e-6)
    assert frames[-1].y_um == pytest.approx(expected["y_um"], abs=1e-6)
    assert snapshots[0]["x_um"] == pytest.approx(expected["x_um"], abs=1e-3)
