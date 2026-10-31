"""The callback-CHAINED hardware scan path (SCANCHAIN): the driver sends the full
XY setpoint waves once and Igor advances line-to-line on the capture-fill callback,
streaming each line back as a CAPTURE("L<n>") pair, terminated by SCANDONE. Exercised
off-instrument against FakeIgorBridge, which mirrors AFM_ScanChainStart/ScanAdvance."""

import asyncio

import numpy as np
import pytest

from afm_litho.service.hw_scan import run_hw_scan_chained
from afm_litho.service.passes import ScanConfig

CAL = {"x_lvdt_sens": 7.46e-6, "x_lvdt_offset": -0.63,
       "y_lvdt_sens": 7.04e-6, "y_lvdt_offset": 0.07,
       "z_lvdt_sens": 5e-6, "z_lvdt_offset": 0.0}


def _run(cfg, cal, **kw):
    """Wire a FakeIgorBridge to a connected HardwareARC, run one chained scan,
    and return (broadcast messages, summary, stub)."""
    from afm_litho.hardware.hardware_arc import HardwareARC
    from afm_litho.hardware.igor_stub import FakeIgorBridge

    # XSensor/YSensor default to 0.0, not the stub's usual NaN-for-unseeded-key:
    # xy_transit's pre-scan move (docs/XY_SLEW_SPEC.md) reads them as the CURRENT
    # position before planning its out-wave pair, and a real controller never
    # reports NaN there -- only this stub does, for a key nothing seeded.
    stub = FakeIgorBridge({"XSensor": 0.0, "YSensor": 0.0},
                          done_delay=kw.pop("done_delay", 0.0))
    port = stub.serve_in_thread()
    hw = HardwareARC.connect("127.0.0.1", port)
    msgs: list[dict] = kw.pop("msgs", [])

    async def broadcast(m):
        msgs.append(m)

    try:
        summary = asyncio.run(run_hw_scan_chained(hw, cfg, cal, broadcast, **kw))
    finally:
        hw.close()
        stub.stop()
    return msgs, summary, stub


def test_chained_scan_streams_a_line_per_advance():
    cfg = ScanConfig(size_um=5.0, pixels=32, lines=4)
    msgs, summary, _stub = _run(cfg, CAL)

    lines = [m for m in msgs if m["type"] == "line"]
    assert len(lines) == 4 and summary["lines"] == 4
    assert [m["row"] for m in lines] == [0, 1, 2, 3]      # rows advance in order
    assert all(len(m["height"]) == 32 for m in lines)      # ppl == pixels (single sweep)
    assert msgs[-1] == {"type": "scan_done", "aborted": False}


def test_chained_scan_sends_full_waves_once_then_one_scanchain():
    """The whole point of chaining: the XY setpoint waves cross the wire ONCE
    (not per line), followed by a single SCANCHAIN carrying the geometry."""
    cfg = ScanConfig(size_um=5.0, pixels=32, lines=4)
    _msgs, _summary, stub = _run(cfg, CAL)

    assert len(stub.scanchains) == 1                      # exactly one SCANCHAIN
    xname, yname, ppl, nlines, _interp, _decim = stub.scanchains[0]
    assert ppl == 32 and nlines == 4                       # points/line and lines
    # both named waves were loaded, each the full frame (pixels*lines points).
    assert xname in stub.waves and yname in stub.waves
    assert len(stub.waves[xname]) == 32 * 4
    assert len(stub.waves[yname]) == 32 * 4


def test_chained_scan_arms_xy_loops_with_calibration_gains():
    """The chained path arms the XY loops with the same calibration-derived P/I
    gains as the per-line path -- a zero-gain closed loop does not track."""
    cfg = ScanConfig(size_um=5.0, pixels=32, lines=4)
    cal = dict(CAL, x_pgain=0.0, x_igain=2818.4, y_pgain=0.0, y_igain=2818.4)
    _msgs, _summary, stub = _run(cfg, cal)

    by_index = {lp[0]: lp for lp in stub.loops}
    assert by_index[0][1] == "XSensor" and by_index[1][1] == "YSensor"
    assert by_index[0][4] == 2818.4 and by_index[1][4] == 2818.4   # tuned I gain


def test_chained_scan_warns_when_an_xy_i_gain_is_absent():
    cal = {k: v for k, v in CAL.items() if k not in ("x_igain", "y_igain")}
    msgs, _summary, _stub = _run(ScanConfig(size_um=5.0, pixels=32, lines=4), cal)
    logs = [m["msg"] for m in msgs if m["type"] == "log"]
    assert any("WARNING: an XY loop I-gain is 0/absent" in t for t in logs)


def test_chained_scan_engage_contact_arms_z_on_deflection_positive():
    """SAFETY-CRITICAL: a contact engage arms the Z loop (index 2) on Deflection ->
    Height with a POSITIVE gain -- reconciled off the rig-rejected Fast%Input /
    Z%Output aliases to the rig-verified concrete channels."""
    cfg = ScanConfig(size_um=5.0, pixels=32, lines=4, mode="contact",
                     engage=True, setpoint=0.30)
    _msgs, _summary, stub = _run(cfg, CAL)

    z = {lp[0]: lp for lp in stub.loops}[2]
    _, inp, sp, pg, ig, outp = z
    assert inp == "Deflection" and outp == "Height"
    assert sp == 0.30 and pg == 0.0 and ig == 1200.0     # positive (Pass default z_igain)


def test_chained_scan_engage_ac_arms_z_on_amplitude_negated():
    """SAFETY-CRITICAL: an AC engage arms the Z loop on Amplitude -> Height with a
    NEGATED integral gain (amplitude FALLS on approach, so the loop error inverts and
    the gain sign must invert with it, or the piezo drives the wrong way = crash)."""
    cfg = ScanConfig(size_um=5.0, pixels=32, lines=4, mode="ac",
                     engage=True, setpoint=0.65)
    _msgs, _summary, stub = _run(cfg, CAL)

    z = {lp[0]: lp for lp in stub.loops}[2]
    _, inp, sp, pg, ig, outp = z
    assert inp == "Amplitude" and outp == "Height"     # AC feedback channel
    assert sp == 0.65 and pg == 0.0 and ig == -1200.0    # NEGATED integral gain


def test_chained_scan_gain_override_scales_and_signs_like_setzgain_contact():
    """startScan `pgain`/`igain` override THIS scan's default Z gains -- the SAME
    PANEL-gain convention `setZGain` takes: scaled to controller units by
    hw_approach._PSCALE/_ISCALE, POSITIVE sign in contact mode."""
    cfg = ScanConfig(size_um=5.0, pixels=32, lines=4, mode="contact",
                     engage=True, setpoint=0.30, pgain=2.0, igain=15.0)
    _msgs, _summary, stub = _run(cfg, CAL)

    z = {lp[0]: lp for lp in stub.loops}[2]
    _, inp, sp, pg, ig, outp = z
    assert inp == "Deflection" and outp == "Height"
    assert sp == 0.30
    assert pg == pytest.approx(2.0 * 0.1)       # _PSCALE, positive (contact)
    assert ig == pytest.approx(15.0 * 100.0)    # _ISCALE, positive (contact)


def test_chained_scan_gain_override_scales_and_signs_like_setzgain_ac():
    """Same override, AC mode: the sign NEGATES (amplitude falls on approach), same
    as the existing default-gain AC test above -- pgain follows igain's sign now
    that both come from the Z arm's baked-in mode sign."""
    cfg = ScanConfig(size_um=5.0, pixels=32, lines=4, mode="ac",
                     engage=True, setpoint=0.65, pgain=2.0, igain=15.0)
    _msgs, _summary, stub = _run(cfg, CAL)

    z = {lp[0]: lp for lp in stub.loops}[2]
    _, inp, sp, pg, ig, outp = z
    assert inp == "Amplitude" and outp == "Height"
    assert sp == 0.65
    assert pg == pytest.approx(-2.0 * 0.1)      # NEGATED
    assert ig == pytest.approx(-15.0 * 100.0)   # NEGATED


def test_chained_scan_no_gain_override_keeps_the_default_z_gains():
    """Omitted `pgain`/`igain`: byte-identical to before this feature -- same
    default P/I this file's existing engage tests already assert (0.0 / ±1200.0)."""
    cfg = ScanConfig(size_um=5.0, pixels=32, lines=4, mode="contact",
                     engage=True, setpoint=0.30)
    _msgs, _summary, stub = _run(cfg, CAL)

    z = {lp[0]: lp for lp in stub.loops}[2]
    _, _inp, _sp, pg, ig, _outp = z
    assert pg == 0.0 and ig == 1200.0


def test_chained_scan_no_engage_does_not_arm_the_z_loop():
    """Retracted (no engage): the Z loop is never armed, so no Z is driven into the
    surface -- unchanged from before the mode-aware change."""
    cfg = ScanConfig(size_um=5.0, pixels=32, lines=4, mode="ac")   # engage default False
    _msgs, _summary, stub = _run(cfg, CAL)
    assert 2 not in {lp[0] for lp in stub.loops}         # only XY loops armed


def test_chained_scan_returns_the_raw_measured_capture():
    """The chained path returns the raw ZSensor/Deflection
    frame so the session recorder (A4) persists it, not an empty capture."""
    cfg = ScanConfig(size_um=5.0, pixels=32, lines=4)
    _msgs, summary, _stub = _run(cfg, CAL)

    cap = summary["capture"]
    assert set(cap) == {"ZSensor", "Deflection"}
    assert len(cap["ZSensor"]) == 32 * 4                   # ppl * lines, row-major
    assert len(cap["Deflection"]) == 32 * 4


def test_chained_scan_ac_captures_amplitude_as_the_error_channel():
    """In AC/tapping mode the meaningful error signal is AMPLITUDE, not Deflection.
    The chained scan must arm the bank-0 error channel as Amplitude, the stub must
    stream it back under that name, and the demux must map it to the error grid: the
    per-line messages label errName "Amplitude" and the persisted capture is keyed
    "Amplitude" (ZSensor topography is unchanged). Contact stays Deflection."""
    cfg = ScanConfig(size_um=5.0, pixels=32, lines=4, mode="ac")
    msgs, summary, _stub = _run(cfg, CAL)

    lines = [m for m in msgs if m["type"] == "line"]
    assert len(lines) == 4 and summary["lines"] == 4       # every line demuxed
    assert all(m["errName"] == "Amplitude" for m in lines)
    cap = summary["capture"]
    assert set(cap) == {"ZSensor", "Amplitude"}            # error keyed under the AC name
    assert len(cap["Amplitude"]) == 32 * 4


def test_chained_scan_passes_the_error_channel_on_scanchain():
    """The mode reaches the .ipf arm THROUGH the SCANCHAIN command: the driver appends
    the error-channel name so Igor binds it as the second channel of the bank-0 pair."""
    from afm_litho.hardware import bridge_protocol as P

    for mode, expect in (("contact", "Deflection"), ("ac", "Amplitude")):
        cfg = ScanConfig(size_um=5.0, pixels=32, lines=4, mode=mode)
        _msgs, _summary, stub = _run(cfg, CAL)
        assert stub._last_scanchain_err == expect
    # protocol carries the channel as the trailing token, default Deflection.
    assert P.cmd_scanchain("u", "x", "y", 6, 4, -50, 50).split()[-1] == "Deflection"
    assert P.cmd_scanchain("u", "x", "y", 6, 4, -50, 50, "Amplitude").split()[-1] == "Amplitude"


def test_chained_scan_aborts_mid_chain():
    """should_abort() flips once the first line is out: the driver sends ABORT, Igor ends
    the chain with a single terminal, and the scan reports aborted with fewer lines."""
    cfg = ScanConfig(size_um=5.0, pixels=32, lines=6)
    seen: list[dict] = []

    def should_abort():
        return any(m["type"] == "line" for m in seen)    # abort after the first line

    msgs, summary, stub = _run(cfg, CAL, done_delay=0.02, should_abort=should_abort,
                               msgs=seen)

    lines = [m for m in msgs if m["type"] == "line"]
    assert 0 < len(lines) < 6                             # stopped before the full frame
    assert summary["aborted"] is True
    assert msgs[-1] == {"type": "scan_done", "aborted": True}
    assert stub.aborts == 1                               # the driver sent one ABORT


def test_load_wave_chunks_a_large_frame_into_one_wave():
    """A wave longer than WAVE_CHUNK_POINTS is transferred as WAVE + WAVEAPP pieces,
    but Igor (and the stub) reconstruct ONE wave a single SCANCHAIN then walks. This
    is the 256x256 fix: a 131k-point frame becomes 16 bounded lines, not one 1.3 MB
    line that chokes the socket -- with no change to the chained scan's behaviour."""
    from afm_litho.hardware.hardware_arc import WAVE_CHUNK_POINTS, HardwareARC
    from afm_litho.hardware.igor_stub import FakeIgorBridge

    stub = FakeIgorBridge({}, done_delay=0.0)
    port = stub.serve_in_thread()
    hw = HardwareARC.connect("127.0.0.1", port)
    big = np.arange(WAVE_CHUNK_POINTS * 3 + 17, dtype=np.float64)   # spans 4 chunks
    try:
        name = hw._load_wave(big)
        hw.td_ReadValue("sync")            # round-trip: stub has processed all chunks
    finally:
        hw.close()
        stub.stop()
    got = np.asarray(stub.waves[name])
    assert len(got) == len(big) and np.allclose(got, big)   # reassembled exactly


def test_small_wave_still_loads_as_a_single_wave():
    """A wave within the chunk size sends exactly one WAVE (no WAVEAPP), so the common
    small-scan path is unchanged."""
    from afm_litho.hardware.hardware_arc import WAVE_CHUNK_POINTS, HardwareARC
    from afm_litho.hardware.igor_stub import FakeIgorBridge

    stub = FakeIgorBridge({}, done_delay=0.0)
    port = stub.serve_in_thread()
    hw = HardwareARC.connect("127.0.0.1", port)
    small = np.arange(WAVE_CHUNK_POINTS, dtype=np.float64)   # exactly one chunk
    try:
        name = hw._load_wave(small)
        hw.td_ReadValue("sync")
    finally:
        hw.close()
        stub.stop()
    assert len(stub.waves[name]) == WAVE_CHUNK_POINTS


# ============== chained scan: a chunked per-line capture reassembles ===========
# A per-line capture (ppl points) is normally far under kCapChunkPoints and arrives as
# one frame per channel. But the reassembly must survive a channel that DOES arrive in
# multiple CAPTURE("L<n>") frames: the driver appends chunks in order and completes the
# line only once BOTH driving channels reach ppl -- never on a partial chunk (which would
# emit a truncated line). A scripted fake ARC feeds the exact chunked frame sequence.

class _FakeRouter:
    """A router stand-in for the scripted driver: `open_waiter` hands back a waiter
    pre-loaded with the scripted (untagged) frames as (verb, rest) tuples -- exactly
    what the real router delivers after it strips the verb + UID."""

    def __init__(self, lines):
        self._lines = list(lines)
        self.on_telem = None

    def open_waiter(self):
        from afm_litho.hardware.bridge_router import Waiter
        w = Waiter("s")
        for line in self._lines:
            verb, _, rest = line.partition(" ")
            w.put((verb, rest))
        return w

    def release(self, uid):
        pass


class _ScriptedARC:
    """A minimal HardwareARC stand-in for run_hw_scan_chained's `driver()`: its
    router hands out a scripted frame sequence, and the setup calls are no-ops.

    Also the ARCBackend-ish surface `xy_transit` (docs/XY_SLEW_SPEC.md) needs for
    the pre-scan move: `td_ReadValue` (current XY, starting at 0 V, i.e. the LVDT
    offset in um), `ramp_output` (a no-op, like the other setup calls), and
    `td_xSetOutWavePair`/`td_xSetInWavePair`/`fire` for the single out-wave-pair
    move itself (2026-09-16: the move now arms an XSensor/YSensor capture pair
    before firing -- see xy_transit.py's module docstring -- so this needs the
    same no-op INPAIR arm `FakeIgorBridge`'s real bridge handling provides).
    Unlike a real bridge (or `FakeIgorBridge`, which now does the same thing --
    see igor_stub.py's OUTPAIR handler), `td_ReadValue`'s XSensor/YSensor answers
    are NOT static: `td_xSetOutWavePair` lands them on the wave's own last
    commanded point, so xy_transit's post-move arrival verify sees the move land
    on target instead of raising over a fake that never actually "moves"."""

    def __init__(self, lines):
        self.sent: list[str] = []
        self.link = self          # driver uses arc.link.send
        self.router = _FakeRouter(lines)
        self.on_telem = None
        self.timeout = 5.0
        self.is_cypher = False
        self._xy_v = {"XSensor": 0.0, "YSensor": 0.0}

    def ir_SetPISLoop(self, *a, **k):
        pass

    def ramp_output(self, *a, **k):
        pass

    def td_ReadValue(self, name, **k):
        return self._xy_v.get(name, 0.0)

    def td_xSetOutWavePair(self, bank, event, ch_a, wave_a, ch_b, wave_b, interp):
        if ch_a == "PIDSLoop.0.Setpoint":
            self._xy_v["XSensor"] = float(wave_a[-1])
        if ch_b == "PIDSLoop.1.Setpoint":
            self._xy_v["YSensor"] = float(wave_b[-1])
        return 0

    def td_xSetInWavePair(self, bank, event, ch_a, ch_b, n_points, decimation):
        return 0

    def fire(self, event, ticks=0):
        return {}

    def _load_wave(self, values):
        return "w"

    def send(self, line):                 # arc.link.send
        self.sent.append(line)

    def abort(self):                      # out-of-band abort (unused in these scripts)
        self.sent.append("ABORT")
        return "tcp"


def _run_scripted(cfg, cal, lines, **kw):
    arc = _ScriptedARC(lines)
    msgs: list[dict] = []

    async def broadcast(m):
        msgs.append(m)

    summary = asyncio.run(run_hw_scan_chained(arc, cfg, cal, broadcast, **kw))
    return msgs, summary, arc


def test_chained_scan_reassembles_a_chunked_line():
    # pixels=4 -> ppl=4 (single sweep). Line 0's three channels each arrive in 2-point
    # chunks; line 1 arrives single-frame. The chunked line must complete only after
    # BOTH ZSensor and Deflection reach ppl, and the raw capture must concatenate the
    # chunks in order.
    cfg = ScanConfig(size_um=5.0, pixels=4, lines=2)
    script = [
        "CAPTURE L0 UserIn2 0.1;0.2",     # bias, streamed first, in 2 chunks
        "CAPTURE L0 UserIn2 0.3;0.4",
        "CAPTURE L0 ZSensor 1;2",          # height in 2 chunks
        "CAPTURE L0 ZSensor 3;4",
        "CAPTURE L0 Deflection 10;20",     # error in 2 chunks -- 1st is partial (len 2 < 4)
        "CAPTURE L0 Deflection 30;40",     # completes L0 only here (len 4 == ppl)
        "CAPTURE L1 UserIn2 0.5;0.6;0.7;0.8",
        "CAPTURE L1 ZSensor 5;6;7;8",
        "CAPTURE L1 Deflection 50;60;70;80",
        "SCANDONE 2",
    ]
    msgs, summary, _arc = _run_scripted(cfg, CAL, script)

    lines = [m for m in msgs if m["type"] == "line"]
    assert len(lines) == 2                              # exactly two lines, none truncated
    assert [m["row"] for m in lines] == [0, 1]
    cap = summary["capture"]
    assert list(cap["ZSensor"]) == [1, 2, 3, 4, 5, 6, 7, 8]        # chunks in order
    assert list(cap["Deflection"]) == [10, 20, 30, 40, 50, 60, 70, 80]
    # UserIn2 carried on every line -> a full row-major frame in the capture
    assert "UserIn2" in cap
    assert list(cap["UserIn2"]) == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    assert msgs[-1] == {"type": "scan_done", "aborted": False}


def test_chained_scan_line_carries_the_tip_bias_as_tipv():
    cfg = ScanConfig(size_um=5.0, pixels=4, lines=1)
    script = ["CAPTURE L0 UserIn2 0.1;0.2;0.3;0.4", "CAPTURE L0 ZSensor 1;2;3;4",
              "CAPTURE L0 Deflection 5;6;7;8", "SCANDONE 1"]
    msgs, _summary, _arc = _run_scripted(cfg, CAL, script)
    line = next(m for m in msgs if m["type"] == "line")
    assert line["tipv"] == [0.1, 0.2, 0.3, 0.4]
