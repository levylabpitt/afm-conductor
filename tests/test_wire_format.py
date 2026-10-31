"""The three rules that only matter at first contact with the instrument.

None of these changes what the twin draws. All of them change what the
hardware driver has to say to the machine, which is why they are pinned
here: the NaN-separated wave format, the 8192-point segment ceiling, and
the rotated litho frame.

Taken from the shipped Asylum v16 source; see docs/asylum-v16-findings.md
for the file and line of each.
"""

import math

import numpy as np
import pytest

from afm_litho.litho.compiler import (LithoObject, Pattern, compile_object,
                                      compile_pattern, nan_waves,
                                      split_long_segments, split_segments)
from afm_litho.litho.frame import LithoFrame, rotate


def _pattern(*objs):
    return Pattern(name="t", objects=list(objs))


def _nan_runs(x):
    """Lengths of each consecutive run of NaNs, in order."""
    runs, n = [], 0
    for v in x:
        if np.isnan(v):
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    return runs


# ---------------------------------------------------- the NaN wire format --


def test_segments_are_separated_by_nan_in_one_long_wave():
    """XLitho/YLitho are ONE wave for the whole pattern, NaN marking pen-up.

    Not a serialisation choice: the executor addresses segments by searching
    for the next NaN, so this is the instrument's own indexing scheme.
    """
    pat = _pattern(
        LithoObject("a", "path", [(0, 0), (2, 0)]),
        LithoObject("b", "path", [(0, 3), (2, 3)]),
        LithoObject("c", "path", [(0, 6), (2, 6)]),
    )
    waves = compile_pattern(pat, 6e-6, 6e-6)
    assert waves["segment_count"] == 3
    # WaveStats V_numNaNs, which the panel calls LithoTotal
    assert waves["nan_count"] == 3           # group-terminated: one per segment
    assert len(waves["x"]) == len(waves["y"]) == len(waves["v"])
    # The volts wave carries NaN in exactly the same places as XY. The vendor
    # gets that for free by writing LithoVolts = setpoint*YLitho/YLitho and
    # letting the division propagate them; we have to do it on purpose.
    assert np.array_equal(np.isnan(waves["x"]), np.isnan(waves["y"]))
    assert np.array_equal(np.isnan(waves["x"]), np.isnan(waves["v"]))


def test_the_wire_format_round_trips():
    """Assemble it, then walk it back with the executor's index arithmetic."""
    pat = _pattern(
        LithoObject("a", "path", [(0, 0), (2, 0)], voltage=8.0),
        LithoObject("b", "path", [(0, 3), (2, 3), (2, 5)], voltage=-6.0),
    )
    waves = compile_pattern(pat, 6e-6, 6e-6)
    back = split_segments(waves["x"], waves["y"], waves["v"])
    assert len(back) == waves["segment_count"] == 2
    for got, want in zip(back, waves["segments"]):
        assert np.allclose(got["x"], want["x_v"])
        assert np.allclose(got["y"], want["y_v"])
        assert np.allclose(got["v"], want["v"])
        assert not np.isnan(got["x"]).any()      # no separator leaks into one
    assert back[0]["v"][0] == pytest.approx(8.0)
    assert back[1]["v"][0] == pytest.approx(-6.0)


def test_a_double_nan_is_a_group_boundary_and_advances_the_index_by_three():
    """`if YLitho[Index+2] is NaN then LithoIndex = Index+3 else Index+2`.

    The doubled separator is how a group survives in the wave. Getting the
    advance wrong swallows the segment after every group, silently.
    """
    segs = [np.array([[0.0, 0.0], [1.0, 0.0]]),
            np.array([[0.0, 1.0], [1.0, 1.0]]),
            np.array([[0.0, 2.0], [1.0, 2.0]])]
    w = nan_waves(segs, groups={0})              # group break after the first
    assert _nan_runs(w["x"]) == [2, 1, 1]        # double, single, terminator
    back = split_segments(w["x"], w["y"], w["v"])
    assert len(back) == 3
    assert back[0]["group_break"] is True
    assert back[1]["group_break"] is False
    for got, want in zip(back, segs):
        assert np.allclose(got["x"], want[:, 0])
        assert np.allclose(got["y"], want[:, 1])


def test_group_boundaries_survive_a_whole_pattern():
    pat = _pattern(LithoObject("a", "path", [(0, 0), (2, 0)]),
                   LithoObject("b", "path", [(0, 3), (2, 3)]))
    w = compile_pattern(pat, 6e-6, 6e-6, group_objects=True)
    assert _nan_runs(w["x"]) == [2, 1]
    assert w["nan_count"] == 3            # and so exceeds the segment count,
    assert w["segment_count"] == 2        # which is why LithoTotal only ever
    back = split_segments(w["x"], w["y"], w["v"])   # counts for the display
    assert len(back) == 2
    assert back[0]["group_break"] is True


def test_ungrouped_patterns_can_leave_off_the_terminator():
    """Without the trailing NaN the count is N-1. The vendor's own count is
    only right about a pattern that was closed as a group."""
    segs = [np.zeros((3, 2)), np.ones((3, 2))]
    assert nan_waves(segs, terminate=False)["nan_count"] == 1
    assert nan_waves(segs, terminate=True)["nan_count"] == 2


def test_a_stray_separator_does_not_invent_an_empty_segment():
    x = np.array([0.0, 1.0, np.nan, np.nan, np.nan, 2.0, 3.0])
    y = x.copy()
    back = split_segments(x, y)
    assert [len(b["x"]) for b in back] == [2, 2]


# ------------------------------------------------ the 8192-point ceiling --


def test_long_segments_are_split_at_8192_points():
    """CheckLithoWaves chops any segment longer than 8192 points, without
    duplicating one, so the pieces rejoin seamlessly."""
    seg = np.column_stack([np.arange(20000.0), np.arange(20000.0) * 2])
    pieces = split_long_segments([seg])
    assert [len(p) for p in pieces] == [8192, 8192, 3616]
    assert np.array_equal(np.concatenate(pieces), seg)      # seamless


def test_a_long_stroke_reaches_the_machine_as_ONE_segment():
    """END TO END, AND THE POINT OF THE FIX: one stroke as the user drew it is one
    segment to the machine. `compile_pattern` no longer applies CheckLithoWaves'
    8192-point cut, because on this stack every segment boundary is a full lift /
    move / set-down -- a dose gap in the middle of a line (2026-09-11)."""
    obj = LithoObject("long", "path", [(0, 0), (19.999, 0)],
                      interpolation_um=0.001, speed_um_s=2.0)
    solo = compile_object(obj, 6e-6, 6e-6)
    # 20,000 points from the geometry -> fired as 20,032 (a multiple of 32): 20,031
    # on the path, then the last position at 0 V
    assert len(solo["x_v"]) == 20032            # one continuous stroke to us
    waves = compile_pattern(_pattern(obj), 6e-6, 6e-6)
    assert waves["segment_count"] == 1          # ...and one to the instrument
    assert waves["nan_count"] == 1              # just the terminator
    back = split_segments(waves["x"], waves["y"], waves["v"])
    assert [len(b["x"]) for b in back] == [20032]
    assert np.allclose(back[0]["x"], solo["x_v"])
    assert np.allclose(back[0]["y"], solo["y_v"])
    assert np.allclose(back[0]["v"], solo["v"])
    assert not any(m["split"] for m in waves["segments"])
    assert waves["segments"][0]["name"] == "long"


def test_the_vendor_split_is_still_available_opt_in():
    """The 8,192-point cut did not go away; it stopped being the default. Ask for it
    by name and the stroke's PATH points are cut the vendor's way (none lost, none
    repeated); each piece is then a stroke of its own, a multiple of 32 points ending
    at 0 V at its last position."""
    obj = LithoObject("long", "path", [(0, 0), (19.999, 0)],
                      interpolation_um=0.001, speed_um_s=2.0)
    solo = compile_object(obj, 6e-6, 6e-6)
    waves = compile_pattern(_pattern(obj), 6e-6, 6e-6, max_segment_points=8192)
    assert waves["segment_count"] == 3
    assert waves["nan_count"] == 3
    back = split_segments(waves["x"], waves["y"], waves["v"])
    # 20,031 path points = 8191 + 8191 + 3649; each piece adds its 0 V end
    # (8192, 8192, and 3649 -> 3680, the next multiple of 32)
    assert [len(b["x"]) for b in back] == [8192, 8192, 3680]
    kept = [m["path_points"] for m in waves["segments"]]
    assert kept == [8191, 8191, 3649]
    # rejoined, the path points are the stroke that went in: none lost, none repeated
    n_path = solo["path_points"]
    for key in ("x", "y", "v"):
        joined = np.concatenate([b[key][:k] for b, k in zip(back, kept)])
        assert np.allclose(joined, solo[key if key == "v" else key + "_v"][:n_path])
    for b, k in zip(back, kept):
        assert np.all(b["v"][k:] == 0.0) and np.all(b["x"][k:] == b["x"][k - 1])
    assert all(m["split"] for m in waves["segments"])
    assert waves["segments"][0]["name"] == "long[1/3]"


def test_a_stroke_of_exactly_the_limit_is_not_split():
    seg = np.zeros((8192, 2))
    assert len(split_long_segments([seg])) == 1
    assert len(split_long_segments([np.zeros((8193, 2))])) == 2


def test_every_piece_carries_its_own_timing():
    """The vendor splits first and computes the interpolation factor per
    segment afterwards. With arc-length-uniform interpolation the pieces
    agree, but that is a RESULT of timing each one, not an assumption: the
    duration of a piece is its own length over its own rate."""
    obj = LithoObject("long", "path", [(0, 0), (19.999, 0)],
                      interpolation_um=0.001, speed_um_s=2.0)
    waves = compile_pattern(_pattern(obj), 6e-6, 6e-6)
    for m in waves["segments"]:
        assert m["ticks_per_point"] >= 1
        assert m["duration_s"] == pytest.approx(
            (len(m["x_v"]) - 1) / m["points_per_second"])
    total = sum(m["duration_s"] for m in waves["segments"])
    assert waves["duration_s"] == pytest.approx(total)


# --------------------------------------------------- the rotated litho frame


def test_the_litho_frame_is_the_identity_by_default():
    """Every existing centred design has to compile exactly as it did."""
    f = LithoFrame()
    assert f.is_identity
    obj = LithoObject("w", "path", [(-8, 0), (8, 0)])
    a = compile_object(obj, 6e-6, 6e-6)
    b = compile_object(obj, 6e-6, 6e-6, frame=f)
    assert np.allclose(a["path_um"], b["path_um"])


def test_rotation_matches_the_vendor_formula():
    """RotateWave unwound: x = f*cos + s*sin, y = -f*sin + s*cos.

    The sign is the surprising one and it is the vendor's: at +90 degrees
    the fast axis points at -y, not +y.
    """
    for ang in (0.0, 17.0, 90.0, -33.0, 210.0):
        a = math.radians(ang)
        for f, s in ((1.0, 0.0), (0.0, 1.0), (2.0, -3.0)):
            x, y = rotate(f, s, ang)
            assert x == pytest.approx(f * math.cos(a) + s * math.sin(a))
            assert y == pytest.approx(-f * math.sin(a) + s * math.cos(a))
    assert rotate(1.0, 0.0, 90.0)[1] == pytest.approx(-1.0)


def test_the_frame_centres_offsets_and_rotates_in_that_order():
    """to_lab = rotate(fast - size/2 - offset, slow - size/2 - offset)."""
    f = LithoFrame(fast_size_um=10.0, slow_size_um=8.0, fast_offset_um=1.0,
                   slow_offset_um=-2.0, angle_deg=25.0)
    x, y = f.to_lab(7.0, 3.0)
    want = rotate(7.0 - 5.0 - 1.0, 3.0 - 4.0 + 2.0, 25.0)
    assert x == pytest.approx(want[0])
    assert y == pytest.approx(want[1])
    back = f.from_lab(x, y)                     # and it inverts exactly
    assert back[0] == pytest.approx(7.0)
    assert back[1] == pytest.approx(3.0)


def test_an_image_frame_puts_the_centre_pixel_at_the_origin():
    f = LithoFrame.for_image(30.0)
    assert f.to_lab(15.0, 15.0) == pytest.approx((0.0, 0.0))


def test_the_frame_is_rigid_so_it_never_changes_the_timing():
    """Rotation and translation cannot change an arc length, so the speed
    quantisation has to come out bit for bit the same at any angle."""
    obj = LithoObject("w", "path", [(0, 0), (4, 1), (4, 5)], speed_um_s=3.0)
    flat = compile_object(obj, 6e-6, 6e-6)
    for ang in (0.0, 31.0, 90.0, 180.0):
        turned = compile_object(obj, 6e-6, 6e-6,
                                frame=LithoFrame(angle_deg=ang))
        assert turned["ticks_per_point"] == flat["ticks_per_point"]
        assert turned["length_um"] == pytest.approx(flat["length_um"])
        assert turned["duration_s"] == pytest.approx(flat["duration_s"])


def test_the_frame_actually_moves_the_pattern():
    """A guard against the frame being plumbed in but never applied."""
    obj = LithoObject("w", "path", [(1.0, 0.0), (2.0, 0.0)])
    turned = compile_object(obj, 6e-6, 6e-6,
                            frame=LithoFrame(angle_deg=90.0))
    assert turned["path_um"][0] == pytest.approx((0.0, -1.0), abs=1e-9)
    assert turned["path_um"][-1] == pytest.approx((0.0, -2.0), abs=1e-9)


@pytest.mark.parametrize("ang", [0.0, 30.0, 90.0])
def test_writing_and_imaging_agree_about_where_a_rotated_frame_points(ang):
    """The reason this rule matters at all.

    A pattern is drawn on an image. If lithography rotates one way and the
    scanner rotates the other, the pattern lands mirrored across the fast
    axis and writes over the wrong feature, at every angle except 0 and 180
    where the bug hides. So: take the lab position the SCAN gives a pixel,
    and the lab position the LITHO FRAME gives that same frame coordinate,
    and require them to be one point.
    """
    from afm_litho.service.passes import ScanConfig, TwinInstrument
    from afm_litho.twin.arc import SoftwareARC
    from afm_litho.twin.physics import Sample, World

    sample = Sample(height_map=np.zeros((64, 64)), extent=30e-6)
    inst = TwinInstrument(SoftwareARC(World(sample=sample)))
    cfg = ScanConfig(size_um=10.0, pixels=8, lines=8, angle_deg=ang,
                     line_rate_hz=20.0)
    inst.start_scan(cfg)
    while inst.scan_line() is not None:
        pass

    frame = LithoFrame(angle_deg=ang)
    xs = inst.scan.channels["X_um"]
    ys = inst.scan.channels["Y_um"]
    n = cfg.pixels
    m = n * 4                                  # the renderer's oversample
    for row in (0, n // 2, n - 1):
        slow = -cfg.size_um / 2 + cfg.size_um * row / (cfg.lines - 1)
        for col in (0, n // 2, n - 1):
            # a pixel is the mean of four oversampled points, so its fast
            # coordinate is the mean of theirs
            fast = np.mean([-cfg.size_um / 2 + cfg.size_um * k / (m - 1)
                            for k in range(col * 4, col * 4 + 4)])
            want_x, want_y = frame.to_lab(fast, slow)
            assert xs[row, col] == pytest.approx(want_x, abs=1e-3)
            assert ys[row, col] == pytest.approx(want_y, abs=1e-3)
