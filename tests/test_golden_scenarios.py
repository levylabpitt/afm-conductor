"""Golden traces: each scenario, driven through the FLEX door (plus the cockpit's HTTP door where
FLEX has no verb) on the Igor stub, judged against its frozen file.

Every scenario is judged as a Cypher at two levels (wire and end of line). Every scenario that
can run on an MFP-3D is also judged as one at the wire level (`<name>.mfp3d.golden.json`).

Regenerate with AFM_LITHO_REGEN_GOLDENS=1, in a commit of its own (tests/golden/README.md).
AFM_LITHO_GOLDEN_REPEAT sets how many recordings the determinism test compares (default 2;
the slice acceptance ran 20, and 5 for the recordings that take over 8 s).
"""

from __future__ import annotations

import difflib
import os
import re

import pytest
from golden.harness import (
    HERE,
    MFP3D_MICROSCOPE_ID,
    GoldenRig,
    commanded,
    compare_wire,
    record,
)
from golden.recordings import golden_group, recorded
from golden.scenarios import OPTIONS, SCENARIOS

from afm_litho.goldens import golden_format as G
from afm_litho.hardware.xy_transit import DEFAULT_WITHDRAW_MAX_S, DEFAULT_WITHDRAW_SETTLE_S
from afm_litho.hardware.lithography_run import (
    FIRST_OBJECT_SETTLE_S,
    SETTLE_MOVE_S,
    LithographyRun,
)

NAMES = list(SCENARIOS)
MFP_NAMES = [n for n in NAMES if OPTIONS.get(n, {}).get("mfp", True) is True]


def _cases(names):
    return [pytest.param(n, marks=golden_group(n)) for n in names]


@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


def _path(name: str, mfp: bool = False):
    return HERE / (f"{name}.mfp3d.golden.json" if mfp else f"{name}.golden.json")


def _record(hub, client, name: str, mfp: bool = False, **kw) -> dict:
    return record(hub, name, client=client,
                  microscope_id=MFP3D_MICROSCOPE_ID if mfp else None, **kw)


def _recorded(hub, client, name: str, mfp: bool = False) -> dict:
    return recorded(hub, client, name, MFP3D_MICROSCOPE_ID if mfp else None).golden


@pytest.mark.parametrize("name", _cases(NAMES))
def test_golden_matches_frozen_file(hub, client, name):
    g = _recorded(hub, client, name)
    if G.regen_requested():
        _path(name).write_bytes(G.dumps(g))
        pytest.skip(f"regenerated {_path(name).name}")
    assert _path(name).exists(), f"no frozen golden; regenerate with {G.REGEN_ENV}=1"
    frozen = G.load(_path(name))
    assert _path(name).read_bytes().replace(b"\r\n", b"\n") == G.dumps(frozen), \
        "frozen file is not canonical; regenerate it, never hand-edit"
    problems = G.compare(frozen, g)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("name", _cases(MFP_NAMES))
def test_mfp3d_wire_golden_matches_frozen_file(hub, client, name):
    g = _recorded(hub, client, name, mfp=True)
    if G.regen_requested():
        _path(name, True).write_bytes(G.dumps(g))
        pytest.skip(f"regenerated {_path(name, True).name}")
    assert _path(name, True).exists(), f"no frozen golden; regenerate with {G.REGEN_ENV}=1"
    frozen = G.load(_path(name, True))
    assert _path(name, True).read_bytes().replace(b"\r\n", b"\n") == G.dumps(frozen), \
        "frozen file is not canonical; regenerate it, never hand-edit"
    assert frozen["microscope_id"] == MFP3D_MICROSCOPE_ID and "igor" not in frozen
    problems = compare_wire(frozen, g)
    assert not problems, "\n".join(problems)


_REPEAT = int(os.environ.get("AFM_LITHO_GOLDEN_REPEAT", "2"))


def _repeat_case(name: str, mfp: bool):
    # a litho recording takes 10 to 25 s, mostly the run's own settle sleeps
    return pytest.param(name, mfp, id=f"{name}{'-mfp3d' if mfp else ''}",
                        marks=[pytest.mark.timeout(90 + 40 * _REPEAT), golden_group(name)])


def _diff(a: bytes, b: bytes) -> str:
    lines = difflib.unified_diff(a.decode().splitlines(), b.decode().splitlines(),
                                 "first", "again", n=1, lineterm="")
    return "\n".join(list(lines)[:40])


@pytest.mark.parametrize("name,mfp", [_repeat_case(n, False) for n in NAMES]
                         + [_repeat_case(n, True) for n in MFP_NAMES])
def test_recording_is_byte_identical(hub, client, name, mfp):
    n = _REPEAT
    first = G.dumps(_recorded(hub, client, name, mfp))
    for i in range(1, n):
        again = G.dumps(_record(hub, client, name, mfp))
        assert again == first, (
            f"{name}: recording {i + 1} of {n} differs:\n" + _diff(first, again))


# ---- the waits the product commanded (the testing clock records them, skips them) ---------

# scenario -> (0.3 s piezo settles, withdraw waits) the product commands in that recording
_COMMANDED = {"litho_single": (2, 2), "litho_multi_object": (7, 3)}


@pytest.mark.parametrize("name", list(_COMMANDED))
def test_a_write_commands_its_settles_and_withdraws(hub, client, name):
    """The write asks for the 3 s first-object settle once, the 0.3 s piezo settle after each
    move and the withdraw's bound plus settle margin before each read; the golden harness
    skips them but they are what the instrument would wait."""
    settles, withdraws = _COMMANDED[name]
    waits = recorded(hub, client, name).waits
    assert len(commanded(waits, FIRST_OBJECT_SETTLE_S, "lithography_run.py")) == 1
    assert len(commanded(waits, SETTLE_MOVE_S, "lithography_run.py")) == settles
    assert len(commanded(waits, DEFAULT_WITHDRAW_MAX_S + DEFAULT_WITHDRAW_SETTLE_S,
                         "hw_approach.py")) == withdraws


def test_a_rig_starts_clean_after_an_earlier_scenario_left_a_park_in_force(hub, client):
    """One scenario failing mid-run must not fail every later one on the worker."""
    hub.host._safe_park = {**hub.host._safe_park, "active": True, "count": 3}
    hub.host._safe_park_now.set()
    with GoldenRig(hub, "isolation") as rig:
        assert rig.door.call("getState") == "idle"
        hub.host._safe_park["active"] = True          # as a late park from the scenario would
    assert not hub.host._safe_park["active"] and not hub.host._safe_park_now.is_set()


@pytest.mark.parametrize("name", ["scan_chained_contact", "litho_single"])
def test_safety_poller_running_keeps_the_golden(hub, client, name):
    """With Igor's always-on tip poll streaming TELEM (and the host consuming it), actuation
    is unchanged at both levels and the frozen reads still occur."""
    if G.regen_requested():
        pytest.skip("regenerating")
    g = _record(hub, client, name, telem_hz=50.0)
    assert "@TIPTICK" in g["wire"]["reads"], "the poller did not run"
    assert 'td_ReadValue("amplitude")' in g["igor"]["reads"]
    problems = G.compare(G.load(_path(name)), g)
    assert not problems, "\n".join(problems)


def test_a_changed_gain_constant_fails_the_golden(hub, client, monkeypatch):
    if G.regen_requested():
        pytest.skip("regenerating")
    kw = dict(LithographyRun.__init__.__kwdefaults__)
    kw["z_igain"] = 1300.0
    monkeypatch.setattr(LithographyRun.__init__, "__kwdefaults__", kw)
    problems = G.compare(G.load(_path("litho_single")), _record(hub, client, "litho_single"))
    assert any(p.startswith("wire.actuation") for p in problems), problems
    assert any(p.startswith("igor.actuation") for p in problems), problems


def test_a_changed_withdraw_path_fails_the_golden(hub, client, monkeypatch):
    """The withdraw's retract setpoint (-10 V) moved to -9 V: the engage scenario's withdraw
    line changes at both levels, and in the MFP-3D wire golden, which re-arms the same way."""
    if G.regen_requested():
        pytest.skip("regenerating")
    from afm_litho.hardware import hw_approach
    monkeypatch.setattr(hw_approach, "_Z_RETRACT_SETPOINT", -9.0)
    name = "engage_contact_tune_withdraw"
    problems = G.compare(G.load(_path(name)), _record(hub, client, name))
    assert any(p.startswith("wire.actuation") for p in problems), problems
    assert any(p.startswith("igor.actuation") for p in problems), problems
    assert compare_wire(G.load(_path(name, True)), _record(hub, client, name, mfp=True))


def test_the_rig_refuses_a_backend_that_is_not_the_stub(hub, monkeypatch):
    monkeypatch.setattr(type(hub), "backend_live", lambda self: "twin")
    with pytest.raises(AssertionError), GoldenRig(hub, "safe_park"):
        pass
    assert hub.hw_bridge() is None


# ---- the frozen files and the README stay consistent with each other ----------------------

def _frozen(name: str, mfp: bool = False) -> dict:
    return G.load(_path(name, mfp))


@pytest.mark.parametrize("name", NAMES)
def test_defect_1_is_listed_in_no_golden(name):
    """defect-1 (the park's overlapping loop-2 commands) is fixed by slice AB: no Cypher or
    MFP-3D golden lists it, and no park golden holds a host STOPLOOP or RAMP of Z."""
    if G.regen_requested():
        pytest.skip("regenerating")
    for mfp in (False, True):
        if mfp and name not in MFP_NAMES:
            continue
        g = _frozen(name, mfp)
        assert "defect-1" not in [d["id"] for d in g["known_defects"]], (name, mfp)
        act = g["wire"]["actuation"]
        if any(x in act for x in ("@SAFEPARK", "ABORT")):
            assert not any(str(x).startswith(("STOPLOOP 2", "RAMP Height", "RAMP Output.Z"))
                           for x in act), (name, mfp)


_README = (HERE / "README.md").read_text(encoding="utf-8")


def test_every_scenario_is_in_the_readme_catalogue():
    section = _README.split("## Scenario catalogue", 1)[1].split("\n## ", 1)[0]
    listed = set(re.findall(r"^\| `([a-z0-9_]+)` \|", section, flags=re.M))
    assert listed == set(NAMES), sorted(listed ^ set(NAMES))


def test_the_entry_points_table_names_real_scenarios():
    section = _README.split("## Igor entry points", 1)[1].split("\n## ", 1)[0]
    rows = [r for r in section.splitlines() if r.startswith("| ") and "---" not in r][1:]
    assert len(rows) >= 6, rows
    for row in rows:
        cols = [c.strip() for c in row.strip().strip("|").split("|")]
        status = cols[-2]
        assert status in ("goldened", "below the line", "gap"), row
        named = re.findall(r"`([a-z0-9_]+)`", cols[-1])
        if status == "goldened":
            assert named, f"a goldened entry point names its scenarios: {row}"
        for n in named:
            assert n in SCENARIOS, f"{n} is not a scenario ({row})"


@pytest.mark.xfail(strict=True, reason=(
    "SLICES F30(2), a real safety race, not fixed here: the first write's pump clears "
    "hub.litho_worker in its finally without checking the handle is its own, so a park during "
    "a second write started in the first one's persistence tail skips abort+join and sends Z "
    "commands while that worker is still sending"))
def test_second_write_started_in_the_first_ones_tail_is_joined_before_any_z_command(
        hub, client, monkeypatch):
    """Write, a second write started while the first persists, then Stop. The park must join
    the second write's worker, so the worker's end-of-run withdraw (SERVOLOOP) comes before the
    park's STOPLOOP. Today the handle is gone and STOPLOOP comes first."""
    import time

    from golden.scenarios import WIRE, WIRE_B, _acquire, _start_write

    from afm_litho.service import captures
    real = captures.save_capture

    def slow(*a, **kw):
        time.sleep(1.5)                        # stretch the first write's persistence tail
        return real(*a, **kw)

    monkeypatch.setattr(captures, "save_capture", slow)
    with GoldenRig(hub, "pump_race") as rig:
        door = rig.door
        tok = _acquire(door)
        _start_write(door, tok, [WIRE])
        while hub.host.litho_thread.is_alive():
            time.sleep(0.01)
        rig.inst.hold_next("ARMFIRE", 1)
        end = time.monotonic() + 10
        while True:
            try:
                _start_write(door, tok, [WIRE_B])
                break
            except AssertionError:             # the claim is still held: try again
                assert time.monotonic() < end
                time.sleep(0.01)
        rig.inst.wait_held()
        door.call("abortWrite", {})
        door.wait_state("idle", "engaged", "parked")
        acts = [e["verb"] for e in rig.tap.snapshot()["actuation"]]
        after = acts[max(i for i, v in enumerate(acts) if v == "@SAFEPARK") + 1:]
        assert after[0] == "SERVOLOOP", after
