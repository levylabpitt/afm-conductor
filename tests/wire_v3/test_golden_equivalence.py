"""Every golden scenario, played over wire v3, ends in the Igor commands the contract names.

Contract section 3a and the opening paragraph: the `igor` level of each frozen golden is the
oracle. A scenario recorded with `wire.use_version(3)` must give the same `igor.actuation`,
same order, byte for byte, with two sanctioned differences:

* the make-safe: no STOPLOOP and no Height ramp (section 3a "Park and abort", section 5a,
  the abort redesign), applied by `contract.contract_igor`; once slice AB re-freezes those
  goldens the transform changes nothing;
* the MFP-3D, whose frozen goldens hold the wire level only until slice (a) adds the igor
  level (M2); until then its case is skipped.

Reads: every frozen read must still occur; new reads are allowed (the SET read-back, section 2b).

Skips until slices W2 (client) and W3 (fake Igor) exist: see `target.fake_available`.
"""

from __future__ import annotations

import re

import pytest
from golden.harness import MFP3D_MICROSCOPE_ID, HERE, record
from golden.recordings import golden_group
from golden.scenarios import OPTIONS, SCENARIOS

from afm_litho.goldens import golden_format as G

from . import contract as C
from .target import fake_available

AVAILABLE, WHY = fake_available()
NAMES = list(SCENARIOS)
MFP_NAMES = [n for n in NAMES if OPTIONS.get(n, {}).get("mfp", True) is True]

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not AVAILABLE, reason=WHY or "v3 target available")]


def _cases(names):
    return [pytest.param(n, marks=golden_group(n)) for n in names]


@pytest.fixture(scope="module")
def hub(client):
    from afm_litho.service.hub import hub as real_hub
    return real_hub


@pytest.fixture
def v3_rig(monkeypatch):
    """Fit the golden rig to v3 without editing it: the fake announces the hash of the .ipf this
    program ships (else HELLO is a hand-edit and play waits for the operator), and the rig's
    stub-backend check reaches the socket through the v3 link."""
    import golden.harness as H

    from afm_litho.hardware import wire_v3

    sha = wire_v3.expected_ipf_sha()
    base = H.GoldenStub.__init__

    def init(self, values, **kw):
        kw.setdefault("v3_sha", sha)
        base(self, values, **kw)

    monkeypatch.setattr(H.GoldenStub, "__init__", init)

    # Real Igor's HELLO carries AFM_MacroStartup's `microscope_id=` (contract 2a); the golden
    # stub adds it to the v2 STARTUP line only.
    from afm_litho.hardware.igor_stub_v3 import V3Wire
    hello = V3Wire._hello_body
    monkeypatch.setattr(
        V3Wire, "_hello_body",
        lambda self: hello(self) + f"microscope_id={getattr(self.stub, 'microscope_id', 0)};")
    def assert_backend(rig):
        # GoldenRig._assert_stub_backend, with HELLO where it looks for STARTUP.
        hub, arc = rig.hub, rig.arc
        assert hub.hw_bridge() is arc and hub.backend_live() == "hardware"
        assert arc.link.sock.getpeername()[1] == rig._stub_port(), "arc is not on the stub"
        assert arc.is_cypher == (rig.microscope_id == H.CYPHER_MICROSCOPE_ID)
        assert rig.stub.v3.host_link is not None, "no v3 HELLO reached the stub"
        assert rig.door.call("getStatus")["backend"] == "hardware"

    monkeypatch.setattr(H.GoldenRig, "_assert_stub_backend", assert_backend)

    # The golden tap shows what the fake handed on to its v2 core, so the proof that v3 was on
    # is the verbs the fake's v3 front end received.
    seen: list[str] = []
    handle = V3Wire.handle

    def spy(self, parts, link):
        seen.append(parts[0])
        return handle(self, parts, link)

    monkeypatch.setattr(V3Wire, "handle", spy)
    monkeypatch.setattr(wire_v3.V3Link, "sock", property(lambda self: self._link.sock),
                        raising=False)
    return seen


_CACHE: dict = {}


def _record_v3(hub, client, name, microscope_id=None) -> dict:
    """One v3 recording per scenario per process; the actuation and the reads test share it."""
    key = (name, microscope_id)
    if key not in _CACHE:
        from afm_litho.hardware import wire
        with wire.use_version(3):           # the arc is built inside the block (W2)
            _CACHE[key] = record(hub, name, client=client, microscope_id=microscope_id)
    return _CACHE[key]


def _fold(lines):
    return [G.fold_fn(x) for x in lines]


def _actuation(frozen_igor: dict, new_igor: dict, name: str) -> None:
    want = _fold(C.contract_igor(frozen_igor["actuation"]))
    got = _fold(new_igor["actuation"])
    assert want == got, f"{name}: igor.actuation differs over v3:\n" + G._first_diff(want, got)


# The reads the v3 connect does not make (F100, a W2 gap): the GETGV calibration reads, and the
# scalar probes td_ReadValue of Output.A/Amplitude/Height/YSensor/DetectorSum/Deflection and
# td_ReadString of PIDSLoop.n.OutputChannel. Narrow it as W2 fixes it.
F100_LOST = re.compile(
    r'^(gv\(".*"\)|td_readvalue\("(Output\.A|Amplitude|Height|YSensor|DetectorSum|Deflection)"\)'
    r'|td_readstring\("PIDSLoop\.\d\.OutputChannel"\))$')


def _lost_reads(frozen_igor: dict, new_igor: dict) -> list[str]:
    return sorted(set(_fold(frozen_igor["reads"])) - set(_fold(new_igor["reads"])))


@pytest.mark.parametrize("name", _cases(NAMES))
def test_cypher_igor_reads_that_survive_over_v3(hub, client, v3_rig, name):
    """Every frozen read still happens, except the F100 set: the gv(...) calibration reads and
    the scalar probes of the v2 connect. Anything else lost is a new regression."""
    frozen = G.load(HERE / f"{name}.golden.json")
    lost = _lost_reads(frozen["igor"], _record_v3(hub, client, name)["igor"])
    new = [r for r in lost if not F100_LOST.match(r)]
    assert not new, f"{name}: v3 lost reads outside the known F100 set: {new[:5]}"


@pytest.mark.parametrize("name", _cases(["connect_reads"]))
def test_cypher_igor_gv_reads_survive_over_v3(hub, client, v3_rig, name):
    """F100 closed by W2 (8d0c7815, STARTUP after HELLO): the connect makes every frozen read."""
    frozen = G.load(HERE / f"{name}.golden.json")
    assert not _lost_reads(frozen["igor"], _record_v3(hub, client, name)["igor"])


@pytest.mark.parametrize("name", _cases(MFP_NAMES))
def test_mfp3d_igor_actuation_is_the_same_over_v3(hub, client, v3_rig, name):
    frozen = G.load(HERE / f"{name}.mfp3d.golden.json")
    if "igor" not in frozen:
        pytest.skip("MFP-3D goldens hold the wire level only until slice (a) adds the igor level")
    got = _record_v3(hub, client, name, MFP3D_MICROSCOPE_ID)
    if "igor" not in got:
        pytest.skip("the harness records the MFP-3D at the wire level only")
    _actuation(frozen["igor"], got["igor"], name)
