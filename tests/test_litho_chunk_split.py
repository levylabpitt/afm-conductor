"""Off-instrument tests for the CHUNKED litho write (seamless chained continuation
past the out-wave cap). See docs/LITHO_CHUNK_CHAIN_PLAN.md.

Three things are unit-testable without hardware and are covered here:
  1. the §4 split-point selection (pen-up-preferring; each chunk <= the cap;
     consecutive boundaries; the SAME indices slice x, y and v),
  2. routing: an object at/under the cap fires the single-ARMFIRE path, an oversize
     object fires LITHOCHAIN with an integer bounds wave,
  3. capture reassembly: per-chunk CAPTURE frames (incl. UserIn2) accumulate IN
     ORDER into one array per channel across the object.

The seam continuity itself (no lift, bias held, no stall, no dose blob on a real
oversize write) is HARDWARE-only -- verify on the Asylum PC with the afm_cbtest bench.
"""

from __future__ import annotations

import numpy as np
import pytest

from afm_litho import clock
from afm_litho.hardware import bridge_protocol as P
from afm_litho.hardware import lithography_run
from afm_litho.hardware.igor_stub import FakeIgorBridge, values_from_note
from afm_litho.hardware.lithography_run import (
    OUTWAVE_MAX_POINTS,
    LithographyRun,
    _penup_mask,
    chunk_boundaries,
)
from afm_litho.litho.chunker import CHAIN_CHUNK_POINTS
from afm_litho.litho.compiler import capture_points

CAP = OUTWAVE_MAX_POINTS          # 87,000 -- the cap on ONE ARMED wave
CHUNK = CHAIN_CHUNK_POINTS        # 86,976 -- what a chain is CUT to (32-aligned)


# ============================ §4 split-point selection ======================

def _check_consecutive(bounds, n):
    assert bounds[0] == 0
    assert bounds[-1] == n
    assert all(bounds[k] < bounds[k + 1] for k in range(len(bounds) - 1))  # no gap/overlap, strictly increasing


def test_at_or_under_the_chunk_size_is_a_single_chunk():
    for n in (1, 100, CHUNK - 1, CHUNK):
        b = chunk_boundaries(n, np.zeros(n, dtype=bool))
        assert b == [0, n]                       # no chaining at/under the chunk size


def test_continuous_stroke_cuts_at_exactly_the_chunk_size():
    # one continuous stroke (no pen-up) longer than the cap: mid-stroke cuts land
    # at exactly start+cap, and every chunk is within the cap.
    n = CHUNK * 2 + 500
    b = chunk_boundaries(n, np.zeros(n, dtype=bool))
    _check_consecutive(b, n)
    assert b == [0, CHUNK, 2 * CHUNK, n]
    assert all(b[k + 1] - b[k] <= CHUNK for k in range(len(b) - 1))


def test_prefers_last_penup_within_the_window():
    # a pen-up sits inside the first cap-window; the cut must land THERE (the tip is
    # already lifted, bias already off -> zero dose artifact), not at start+cap.
    n = CHUNK + 3000
    penup = np.zeros(n, dtype=bool)
    penup[CHUNK - 1000] = True                     # last pen-up before the cap
    penup[CHUNK - 5000] = True                     # an earlier one (should be ignored)
    b = chunk_boundaries(n, penup)
    _check_consecutive(b, n)
    assert b == [0, CHUNK - 1000, n]               # cut at the LAST pen-up in the window


def test_penup_outside_window_falls_back_to_midstroke():
    # the only pen-up is beyond start+cap: it cannot be used, so the first cut is
    # mid-stroke at start+cap; the pen-up then governs the following window.
    n = CHUNK * 2
    penup = np.zeros(n, dtype=bool)
    penup[CHUNK + 10] = True
    b = chunk_boundaries(n, penup)
    _check_consecutive(b, n)
    assert b[1] == CHUNK                           # first cut mid-stroke (pen-up was out of reach)
    assert all(b[k + 1] - b[k] <= CHUNK for k in range(len(b) - 1))


def test_same_indices_slice_all_three_arrays():
    # the mask is built from x, y AND v together (a NaN in ANY marks a pen-up), so the
    # ONE boundary list registers position and voltage across every seam.
    n = CHUNK + 10
    x = np.ones(n)
    y = np.ones(n)
    v = np.ones(n)
    v[CHUNK - 7] = np.nan                          # a bias-only pen-up marker
    mask = _penup_mask(v, x, y)
    assert mask[CHUNK - 7] and mask.sum() == 1
    b = chunk_boundaries(n, mask)
    assert b == [0, CHUNK - 7, n]                  # cut at the NaN, same index for x/y/v


def test_every_chunk_within_cap_random():
    rng = np.random.default_rng(0)
    n = CHUNK * 3 + 1234
    penup = rng.random(n) < 1e-4                 # sparse pen-ups
    b = chunk_boundaries(n, penup)
    _check_consecutive(b, n)
    assert all(0 < b[k + 1] - b[k] <= CHUNK for k in range(len(b) - 1))


# ================================ routing ===================================

def _bridge(**kw):
    note = {"XLVDTSens": 7.0e-6, "XLVDTOffset": -0.6, "YLVDTSens": 7.0e-6,
            "YLVDTOffset": 0.07, "ScanRate": 1.0}
    return FakeIgorBridge(values_from_note(note), **kw)


def _run(port, **cfg):
    # zero the settle waits (incl. the verified-withdraw's, docs/XY_SLEW_SPEC.md)
    # so a test with more than one object/stroke doesn't actually sleep.
    return LithographyRun.connect("127.0.0.1", port, settle_move_s=0.0,
                                  cmd_gap_s=0.0, xy_withdraw_max_s=0.0,
                                  xy_withdraw_settle_s=0.0, **cfg)


def test_small_object_uses_armfire_not_chain():
    stub = _bridge(done_delay=0.0)
    port = stub.serve_in_thread()
    try:
        run = _run(port)
        run.start()
        res = run.write_object("small", [0.0, 1.0, 2.0], [0.0, 0.0, 0.0],
                               [3.0, 3.0, 3.0], setpoint=0.4, speed=5.0)
        run.get_value("XSensor")                 # barrier: stub processed everything
    finally:
        stub.stop()
    assert stub.arms and stub.arms[0][0] == "small"   # the single-fire ARMFIRE path
    assert stub.lithochains == []                      # NOT chained
    assert "LithoBounds" not in stub.waves
    assert not res.aborted
    run.close()


def test_oversize_object_chains_and_accumulates(monkeypatch):
    # Shrink the CHUNK SIZE so a tiny object exercises the real routing (no 87k-point
    # socket round-trip needed). The driver must send LITHOCHAIN + an integer bounds
    # wave, and reassemble the per-chunk captures into one array per channel.
    # 32, not an arbitrary 40: the real CHAIN_CHUNK_POINTS is 32-aligned so a chunk's
    # padded capture length equals the chunk, and this stand-in keeps that property.
    monkeypatch.setattr(lithography_run, "CHAIN_CHUNK_POINTS", 32)
    n = 100
    x = list(np.linspace(0.0, 9.0, n))
    y = [0.0] * n
    v = [3.0] * n
    stub = _bridge(done_delay=0.0)
    port = stub.serve_in_thread()
    try:
        run = _run(port)
        run.start()
        res = run.write_object("big", x, y, v, setpoint=0.4, speed=5.0)
        run.get_value("XSensor")
    finally:
        stub.stop()

    # routed to the chain, single ARMFIRE not used
    assert stub.arms == []
    assert len(stub.lithochains) == 1
    lc = stub.lithochains[0]
    assert lc[0] == "big"
    assert lc[1:5] == ("XLitho", "YLitho", "VLitho", "LithoBounds")

    # the integer bounds wave: consecutive, each chunk within the (shrunk) cap
    bounds = [int(round(b)) for b in stub.waves["LithoBounds"]]
    # 100 points that did not come from the compiler: fired as 128, the last
    # position repeated at 0 V (capture_points(100 + 1))
    assert bounds == [0, 32, 64, 96, 128]
    assert all(0 < bounds[k + 1] - bounds[k] <= 32 for k in range(len(bounds) - 1))
    # EVERY chunk is 32-aligned, the last one too, so Igor's pad is a no-op on all
    for k in range(len(bounds) - 1):
        m = bounds[k + 1] - bounds[k]
        assert capture_points(m) == m
    assert stub.waves["VLitho"][99] != 0.0 and stub.waves["VLitho"][100:] == [0.0] * 28

    # captures accumulate in fire order: length = sum of per-chunk padded capture lens
    expect_len = sum(capture_points(bounds[k + 1] - bounds[k])
                     for k in range(len(bounds) - 1))
    for chan in ("UserIn2", "Lateral", "ZSensor", "Deflection", "XSensor", "YSensor"):
        assert len(res.capture[chan]) == expect_len
    assert res.cap_len == expect_len
    assert not res.aborted
    run.close()


def test_chain_without_the_lateral_pair_still_captures_the_bias(monkeypatch):
    # The .ipf fail-safe ladder: in-bank 2 refuses the UserIn2 + Lateral pair, so
    # UserIn2 is armed alone. The chain still writes and the bias capture still
    # accumulates; only Lateral is missing.
    monkeypatch.setattr(lithography_run, "CHAIN_CHUNK_POINTS", 32)
    n = 100
    stub = _bridge(done_delay=0.0)
    stub.lateral_armed = False
    port = stub.serve_in_thread()
    try:
        run = _run(port)
        run.start()
        res = run.write_object("big", list(np.linspace(0.0, 9.0, n)), [0.0] * n,
                               [3.0] * n, setpoint=0.4, speed=5.0)
        run.get_value("XSensor")
    finally:
        stub.stop()
    assert "Lateral" not in res.capture
    assert set(res.capture) == {"UserIn2", "ZSensor", "Deflection",
                                "XSensor", "YSensor"}
    assert len(res.capture["UserIn2"]) == res.cap_len
    assert not res.aborted
    run.close()


# ==================== capture reassembly (accumulate order) =================

def _waiter(lines):
    """A router Waiter pre-loaded with the given (untagged) frames as (verb, rest)
    tuples -- exactly what the router delivers after stripping the verb + UID. Lets
    the accumulation collectors be exercised with no socket / router thread."""
    from afm_litho.hardware.bridge_router import Waiter
    w = Waiter("t")
    for line in lines:
        verb, _, rest = line.partition(" ")
        w.put((verb, rest))
    return w


def test_await_completion_accumulates_multichunk_captures():
    # two chunks: each streams XSensor + YSensor + UserIn2 under the one name, then a
    # single DONE. The frames must concatenate in order into one array per channel.
    w = _waiter([
        "CAPTURE obj UserIn2 0.1;0.2",
        "CAPTURE obj XSensor 1;2",
        "CAPTURE obj YSensor 3;4",
        "CAPTURE obj UserIn2 0.3;0.4",
        "CAPTURE obj XSensor 5;6",
        "CAPTURE obj YSensor 7;8",
        "DONE obj",
    ])
    run = LithographyRun(None, settle_move_s=0.0, cmd_gap_s=0.0)   # no link -> no router
    verb, got, capture = run._await_completion(w, "obj")
    assert verb == P.DONE and got == "obj"
    assert capture["XSensor"] == [1.0, 2.0, 5.0, 6.0]         # chunk 0 then chunk 1
    assert capture["YSensor"] == [3.0, 4.0, 7.0, 8.0]
    assert capture["UserIn2"] == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_await_completion_single_chunk_is_unchanged():
    # the <=cap path streams each channel once: accumulation must equal assignment.
    w = _waiter([
        "CAPTURE solo XSensor 1;2;3",
        "CAPTURE solo YSensor 4;5;6",
        "DONE solo",
    ])
    run = LithographyRun(None, settle_move_s=0.0, cmd_gap_s=0.0)
    verb, got, capture = run._await_completion(w, "solo")
    assert verb == P.DONE
    assert capture["XSensor"] == [1.0, 2.0, 3.0]
    assert capture["YSensor"] == [4.0, 5.0, 6.0]


# ============ HardwareARC path: chunked capture reassembles in order ==========
# A scan fire awaits completion via HardwareARC, whose big
# per-fire captures (a whole-frame scan, an oversize ARMFIRE-class capture) the bridge
# now CHUNKS (AFM_SendCapture, kCapChunkPoints) into several same-(name,channel) frames.
# HardwareARC._await_completion must APPEND those in order (it used to overwrite), and a
# single-frame small capture must still behave exactly as before.

def test_hardware_arc_await_accumulates_chunked_capture():
    from afm_litho.hardware.hardware_arc import HardwareARC
    w = _waiter([
        "CAPTURE p0 ZSensor 1;2",          # ZSensor arrives in three chunks
        "CAPTURE p0 ZSensor 3;4",
        "CAPTURE p0 ZSensor 5;6",
        "CAPTURE p0 Deflection 7;8",       # Deflection in two chunks
        "CAPTURE p0 Deflection 9;10",
        "DONE p0",
    ])
    hw = HardwareARC(None)                 # no link -> no router
    cap = hw._await_completion(w, "p0")
    assert list(cap["ZSensor"]) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]   # in fire order
    assert list(cap["Deflection"]) == [7.0, 8.0, 9.0, 10.0]


def test_hardware_arc_await_single_frame_unchanged():
    from afm_litho.hardware.hardware_arc import HardwareARC
    w = _waiter([
        "CAPTURE p0 ZSensor 1;2;3",
        "CAPTURE p0 Deflection 4;5;6",
        "DONE p0",
    ])
    hw = HardwareARC(None)
    cap = hw._await_completion(w, "p0")
    assert list(cap["ZSensor"]) == [1.0, 2.0, 3.0]
    assert list(cap["Deflection"]) == [4.0, 5.0, 6.0]


# ====================== the cap binds at the OUT-WAVE ARM ===================
#
# The 87,000-point cap is on ONE ARMED OUT-WAVE, not on one transferred Igor wave
# and not on an object. `_load_wave` deliberately builds far longer waves (WAVE +
# WAVEAPP) because SCANCHAIN and LITHOCHAIN slice them Igor-side a chunk at a time;
# the place the cap actually binds is `td_xSetOutWave`/`Pair`, where a bank is armed.
# Over-cap there is a SILENT truncation on the instrument, so it has to be loud here.

class _RecordingLink:
    """A BridgeLink that records nothing but WAVE/WAVEAPP, reassembling each named
    wave exactly as Igor would. No socket, no router, no replies."""

    def __init__(self):
        self.waves: dict[str, list[float]] = {}
        self.lines: list[str] = []

    def send(self, line: str) -> None:
        self.lines.append(line)
        verb, _, rest = line.partition(" ")
        if verb not in (P.WAVE, P.WAVEAPP):
            return
        name, _, payload = rest.partition(" ")
        vals = [float(v) for v in payload.split(";") if v != ""]
        if verb == P.WAVE:
            self.waves[name] = vals
        else:
            self.waves.setdefault(name, []).extend(vals)


def _hw_with_recording_link():
    from afm_litho.hardware.hardware_arc import HardwareARC
    link = _RecordingLink()
    # router=object() so the constructor does not start a real BridgeRouter on a
    # link that can never reply.
    return HardwareARC(link, router=object()), link


def test_arming_an_out_wave_at_the_cap_is_allowed():
    hw, link = _hw_with_recording_link()
    wave = np.arange(CAP, dtype=np.float64)
    hw.td_xSetOutWave(1, "0", "Output.A", wave, -1)
    assert len(link.waves) == 1
    (got,) = link.waves.values()
    assert len(got) == CAP and np.allclose(got, wave)   # reassembled whole


def test_arming_an_out_wave_over_the_cap_is_refused_loudly():
    """One point past the cap. The controller would TRUNCATE it and report success --
    a partially written object nobody is told about -- so the driver refuses."""
    hw, link = _hw_with_recording_link()
    wave = np.zeros(CAP + 1, dtype=np.float64)
    with pytest.raises(ValueError, match="87,000"):
        hw.td_xSetOutWave(1, "0", "Output.A", wave, -1)
    with pytest.raises(ValueError, match="Output.A"):
        hw.td_xSetOutWave(1, "0", "Output.A", wave, -1)
    # refused BEFORE any transfer: nothing went down the wire
    assert link.lines == []


def test_an_over_cap_pair_partner_leaves_no_orphan_wave_on_igor():
    """THE PAIR PATH VALIDATES BOTH BEFORE LOADING EITHER. Checking B only after
    loading A would transfer A to Igor and then raise -- leaving an `ow{N}` wave
    nothing ever frees, once per mistake, for the life of the Igor session."""
    hw, link = _hw_with_recording_link()
    ok = np.zeros(10, dtype=np.float64)
    over = np.zeros(CAP + 1, dtype=np.float64)
    with pytest.raises(ValueError, match="PIDSLoop.1.Setpoint"):
        hw.td_xSetOutWavePair(0, "0", "PIDSLoop.0.Setpoint", ok,
                              "PIDSLoop.1.Setpoint", over, -1)
    assert link.lines == []          # A was NOT transferred: no orphan
    assert link.waves == {}
    # the same, with the over-cap wave FIRST, so neither order leaks
    with pytest.raises(ValueError, match="PIDSLoop.0.Setpoint"):
        hw.td_xSetOutWavePair(0, "0", "PIDSLoop.0.Setpoint", over,
                              "PIDSLoop.1.Setpoint", ok, -1)
    assert link.lines == []


def test_every_out_wave_the_chunker_plans_is_armable():
    """The contract between `litho/chunker.py` and the arm path: a trajectory of any
    length, tiled by `chunk_indices`, arms chunk by chunk with no refusal and no wave
    over the cap. This is what `the arm path does for any long trajectory."""
    from afm_litho.litho.chunker import chunk_indices

    n = 3 * CAP + 777
    x = np.arange(n, dtype=np.float64)
    y = -x
    hw, link = _hw_with_recording_link()
    slices = chunk_indices(n)
    assert len(slices) >= 4                       # genuinely chunked
    for a, b in slices:
        hw.td_xSetOutWavePair(0, "0", "PIDSLoop.0.Setpoint", x[a:b],
                              "PIDSLoop.1.Setpoint", y[a:b], -1)
    assert len(link.waves) == 2 * len(slices)     # one wave per axis per chunk
    assert all(len(v) <= CAP for v in link.waves.values())
    # ...and the chunks tile the trajectory exactly: every point armed once.
    assert sum(len(v) for v in link.waves.values()) == 2 * n


# ============ the published `segments` IS the number of ARMFIREs ============

def test_published_segments_equal_the_armfires_the_hardware_path_sends():
    """The hardware half of the twin parity check, and the one that pins the claim in
    runbook 09. `service/host/litho_worker.py`'s worker iterates
    `compile_pattern(...)["segments"]` and calls `write_object` once per STROKE, and
    `write_object` sends ONE ARMFIRE for a stroke inside the cap -- so a 30,000-point
    line is one arm, not four, and there is no LITHOCHAIN either.
    """
    from afm_litho.litho.compiler import LithoObject, Pattern, compile_pattern

    # A sawtooth: one stroke, far over the old 8,192-point cut, well under the cap.
    pts = [(i * 0.05, 5.0 if i % 2 else 0.0) for i in range(40)]
    pattern = Pattern(name="saw", objects=[
        LithoObject("saw", "path", pts, voltage=8.0, speed_um_s=200.0)])
    metas = compile_pattern(pattern, 7.0e-6, 7.0e-6, 0.0, 0.0)["segments"]
    published = len(metas)                  # exactly what `_compile_report` publishes
    assert published == 1, f"one stroke, one segment; got {published}"
    assert len(metas[0]["path_um"]) > 8192, "the stroke must beat the OLD 8,192 cut"

    stub = _bridge(done_delay=0.0)
    port = stub.serve_in_thread()
    try:
        run = _run(port)
        run.start()
        for meta in metas:
            path = np.asarray(meta["path_um"], dtype=float)
            run.write_object(meta["name"], path[:, 0], path[:, 1],
                             np.asarray(meta["v"], dtype=float),
                             setpoint=0.2, speed=200.0)
        run.get_value("XSensor")            # barrier: the stub processed everything
    finally:
        stub.stop()
    run.close()

    assert len(stub.arms) == published       # one ARMFIRE per published segment
    assert stub.lithochains == []            # ...and never a chunk chain
    assert "LithoBounds" not in stub.waves
    # the WHOLE stroke reached Igor, in one wave, within the cap
    for name in ("XLitho", "YLitho", "VLitho"):
        assert len(stub.waves[name]) == len(metas[0]["path_um"])
        assert len(stub.waves[name]) <= CAP
    # it went out CHUNKED on the wire -- WAVE then WAVEAPP -- not as one giant line
    assert stub.transcript.count("WAVEAPP") >= 3
    # and Igor zeroed the bias exactly ONCE, at the end of the stroke
    assert stub.bias_zeros == [(metas[0]["name"], 0)]


def test_no_lift_inside_a_stroke_over_8192_points():
    """THE REGRESSION TEST FOR THE BUG. Between the ARMFIRE that starts a stroke and
    the terminal that ends it the driver must send NOTHING that moves the tip: no
    SERVOLOOP (lift / set-down / XY arm), no RAMP (reposition), no SETVALUE (the
    MFP-3D Z-setpoint poke). Before 2026-09-11 a 30,000-point line was cut into four
    segments and every seam ran that whole cycle with the bias at zero.

    `_arm_z` re-arms loop 2 (SERVOLOOP) on both platforms; the end-of-stroke lift is
    the ONLY tip-moving verb after the fire.
    """
    from afm_litho.litho.compiler import LithoObject, Pattern, compile_pattern

    pts = [(i * 0.05, 5.0 if i % 2 else 0.0) for i in range(40)]
    pattern = Pattern(name="saw", objects=[
        LithoObject("saw", "path", pts, voltage=8.0, speed_um_s=200.0)])
    meta = compile_pattern(pattern, 7.0e-6, 7.0e-6, 0.0, 0.0)["segments"][0]
    n = len(meta["path_um"])
    assert n > 8192

    stub = _bridge(done_delay=0.0)
    port = stub.serve_in_thread()
    try:
        run = _run(port)
        run.start()
        path = np.asarray(meta["path_um"], dtype=float)
        run.write_object("saw", path[:, 0], path[:, 1],
                         np.asarray(meta["v"], dtype=float),
                         setpoint=0.2, speed=200.0)
        run.get_value("XSensor")
    finally:
        stub.stop()
    run.close()

    t = stub.transcript
    assert t.count("ARMFIRE") == 1 and t.count("LITHOCHAIN") == 0
    fire = t.index("ARMFIRE")
    # the driver's trailing lift is the FIRST tip-moving verb after the fire, and it
    # comes after the completion the driver waited for -- so nothing in between.
    after = t[fire + 1:]
    moved = [v for v in after if v in ("SERVOLOOP", "RAMP", "SETVALUE")]
    assert moved == ["SERVOLOOP"], moved          # exactly the end-of-object lift
    loop, chan, setpoint, _p, _i, _out = stub.loops[-1]
    assert (loop, chan, setpoint) == (2, "Deflection", -10.0)
    # Igor zeroed the bias once, at the end -- never mid-stroke.
    assert stub.bias_zeros == [("saw", 0)]


def test_a_chain_boundary_moves_nothing_and_zeroes_the_bias_once():
    """An over-cap stroke goes out as ONE LITHOCHAIN of >=2 chunks. The whole stroke
    is uploaded once (chunked on the wire with WAVEAPP, but one wave in Igor), the
    per-chunk captures accumulate into one array per channel, and between chunks the
    driver sends nothing at all -- the advance is Igor's, with the tip down and the
    bias live (AFM_LithoAdvance zeroes Output.A only at the true end).
    """
    n = 100_000
    x = np.linspace(0.0, 30.0, n)
    y = np.zeros(n)
    v = np.full(n, 8.0)

    stub = _bridge(done_delay=0.0)
    port = stub.serve_in_thread()
    try:
        run = _run(port)
        run.start()
        res = run.write_object("big", x, y, v, setpoint=0.2, speed=200.0)
        run.get_value("XSensor")
    finally:
        stub.stop()
    run.close()

    t = stub.transcript
    assert t.count("LITHOCHAIN") == 1 and t.count("ARMFIRE") == 0
    name, xn, yn, vn, bn, _interp, _dec = stub.lithochains[0]
    assert (name, xn, yn, vn) == ("big", "XLitho", "YLitho", "VLitho")
    bounds = [int(b) for b in stub.waves[bn]]
    # CUT AT 86,976, NOT AT THE 87,000 ARM CAP. Igor pads each chunk's capture UP to a
    # multiple of 32 (AFM_LithoCapPoints), so an 87,000-point chunk would arm FIVE
    # 87,008-point in-waves -- 8 over the cap, on a live biased tip, with no evidence
    # either way about whether the DSP enforces it on in-waves. 86,976 = 2718*32 makes
    # capLen == m exactly.
    # 100,000 points that did not come from the compiler: fired as 100,032, the last
    # position repeated at 0 V (capture_points(n + 1))
    n_fired = capture_points(n + 1)
    assert n_fired == 100_032
    assert bounds == [0, CHAIN_CHUNK_POINTS, n_fired]
    for k in range(len(bounds) - 1):
        m = bounds[k + 1] - bounds[k]
        assert m <= CHAIN_CHUNK_POINTS <= CAP
        assert capture_points(m) == m, "the padded in-wave must not grow the chunk"
        assert capture_points(m) <= CAP
    # ONE wave per channel in Igor, the whole stroke, framed WAVE + WAVEAPP
    for wname in ("XLitho", "YLitho", "VLitho"):
        assert len(stub.waves[wname]) == n_fired
    assert stub.transcript.count("WAVEAPP") >= 3 * (n // 8192)
    # nothing that moves the tip between the chain start and the terminal
    fire = t.index("LITHOCHAIN")
    moved = [verb for verb in t[fire + 1:] if verb in ("SERVOLOOP", "RAMP", "SETVALUE")]
    assert moved == ["SERVOLOOP"], moved            # only the end-of-object lift
    # ONE bias zeroing for the whole object, at the LAST chunk
    assert stub.bias_zeros == [("big", len(bounds) - 2)]
    # FIVE channels stream per chunk, as AFM_LithoAdvance does -- the volume is what
    # makes a chain seam a multi-second stationary biased dwell, so the stub must not
    # understate it.
    assert set(res.capture) == {"UserIn2", "Lateral", "ZSensor", "Deflection",
                                "XSensor", "YSensor"}
    # capture ACCUMULATED across chunks: the whole stroke, not one chunk of it
    assert res.capture, "no capture came back"
    for chan, vals in res.capture.items():
        assert len(vals) == res.cap_len, chan
    assert res.cap_len >= n


# ===== the published per-stroke / per-object allowances ARE the driver's sleeps ===

@pytest.fixture
def testing_clock():
    """The testing clock records each commanded wait and skips it."""
    with clock.using(clock.TESTING) as c:
        yield c


def _sleeps_since(testing_clock, start: int) -> list[float]:
    return [w.seconds for w in testing_clock.waits[start:] if w.kind == "sleep"]


def test_per_stroke_transit_s_equals_what_the_hover_move_actually_sleeps(testing_clock):
    """`est_s` charges PER_STROKE_TRANSIT_S for every gap BETWEEN two strokes OF
    THE SAME SOURCE OBJECT (docs/XY_SLEW_SPEC.md rule 5 -- a fill row's gap inside
    one shape, `_hover_move`, NOT a verified withdraw). Derive it: write TWO
    zero-hop strokes of ONE source (`source="a"` both calls) with the real settle
    constants, add up every sleep the SECOND stroke's move performs (not the
    first: the first is a fresh run, tip never engaged, so its move takes the
    xy_transit path instead, which is a DIFFERENT cost -- see
    test_per_object_withdraw_s_equals_what_the_verified_withdraw_actually_sleeps),
    and compare against the published constant.

    It caught a real error once (pre-2026-09-15 history). ZERO HOP, deliberately:
    the second stroke starts where the tip already is (`_last_xy_um`, the
    software-tracked previous point `_hover_move` ramps from), so `_plan_ramp`
    floors `maxsec` at RAMP_SETTLE_MARGIN_S and the XY travel time drops out. A
    real hop at MOVE_SPEED_UM_S adds to this, which is exactly why the constant is
    documented as a FLOOR rather than an estimate.
    """
    from afm_litho.hardware.lithography_run import CMD_GAP_S, RAMP_SETTLE_MARGIN_S, SETTLE_MOVE_S
    from afm_litho.service.host.litho_cmds import PER_STROKE_TRANSIT_S

    stub = _bridge(done_delay=0.0)
    port = stub.serve_in_thread()
    try:
        # the REAL settle constants, not the tests' zeroed ones
        run = LithographyRun.connect("127.0.0.1", port,
                                     settle_move_s=SETTLE_MOVE_S,
                                     cmd_gap_s=CMD_GAP_S)
        run.start()
        # first stroke: fresh run, tip never engaged -> xy_transit path (no withdraw
        # to pay, but a DIFFERENT cost than the hover move -- excluded below).
        run.write_object("a#1", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [8.0, 8.0, 8.0],
                         setpoint=0.2, speed=5.0, source="a")
        mark = len(testing_clock.waits)  # only the SECOND stroke's move counts now
        # second stroke of the SAME source object -> _hover_move, zero hop.
        run.write_object("a#2", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [8.0, 8.0, 8.0],
                         setpoint=0.2, speed=5.0, source="a")
    finally:
        stub.stop()
    run.close()

    # everything that is not a per-command transport gap is tip-settling time
    settles = [d for d in _sleeps_since(testing_clock, mark) if d != CMD_GAP_S]
    assert sum(settles) == pytest.approx(PER_STROKE_TRANSIT_S)
    assert PER_STROKE_TRANSIT_S == pytest.approx(4 * SETTLE_MOVE_S + RAMP_SETTLE_MARGIN_S)


def test_per_object_withdraw_s_equals_what_the_verified_withdraw_actually_sleeps(testing_clock):
    """`write_est_s` charges PER_OBJECT_WITHDRAW_S once for the move TO EACH
    DISTINCT source object (docs/XY_SLEW_SPEC.md rule 5 -- `xy_transit`'s verified
    withdraw, not the mild `_hover_move`). Derive it the same way: write two
    zero-hop, DISTINCT-source objects with the real settle + withdraw constants,
    add up every sleep the SECOND object's move performs (the tip is engaged from
    the FIRST object's set-down, so this move's xy_transit call withdraws first),
    and compare against the published constant."""
    from afm_litho.hardware.lithography_run import CMD_GAP_S, SETTLE_MOVE_S
    from afm_litho.hardware.xy_transit import DEFAULT_WITHDRAW_MAX_S, DEFAULT_WITHDRAW_SETTLE_S
    from afm_litho.service.host.litho_cmds import PER_OBJECT_WITHDRAW_S

    stub = _bridge(done_delay=0.0)
    stub.values["XSensor"], stub.values["YSensor"] = -0.6, 0.07  # == um (0, 0) here
    port = stub.serve_in_thread()
    try:
        # the REAL settle + withdraw constants, not the tests' zeroed ones
        run = LithographyRun.connect("127.0.0.1", port,
                                     settle_move_s=SETTLE_MOVE_S,
                                     cmd_gap_s=CMD_GAP_S,
                                     xy_withdraw_max_s=DEFAULT_WITHDRAW_MAX_S,
                                     xy_withdraw_settle_s=DEFAULT_WITHDRAW_SETTLE_S)
        run.start()
        # first object: fresh run, tip never engaged -> its move pays no withdraw.
        run.write_object("a", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [8.0, 8.0, 8.0],
                         setpoint=0.2, speed=5.0, source="a")
        mark = len(testing_clock.waits)  # only the SECOND object's move counts now
        # second, DISTINCT object -> the tip is engaged from "a"'s set-down, so this
        # move's xy_transit call withdraws first.
        run.write_object("b", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [8.0, 8.0, 8.0],
                         setpoint=0.2, speed=5.0, source="b")
    finally:
        stub.stop()
    run.close()

    # everything that is not a per-command transport gap is tip-settling time.
    # 2026-09-16: no more `+ RAMP_SETTLE_MARGIN_S` term here -- xy_transit's move
    # no longer waits it out with a real `time.sleep`; it blocks on the bridge's
    # own DONE report instead (see litho_cmds.PER_OBJECT_WITHDRAW_S's comment).
    settles = [d for d in _sleeps_since(testing_clock, mark) if d != CMD_GAP_S]
    assert sum(settles) == pytest.approx(PER_OBJECT_WITHDRAW_S)
    assert PER_OBJECT_WITHDRAW_S == pytest.approx(
        DEFAULT_WITHDRAW_MAX_S + DEFAULT_WITHDRAW_SETTLE_S + 2 * SETTLE_MOVE_S)
