"""The design file's records: every kind of object survives a save and a load
with every field equal, and a bad file is refused whole."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

from afm_litho.litho import design_file
from afm_litho.litho.bezier import BezierPath, Node
from afm_litho.litho.compiler import LithoObject
from afm_litho.litho.design_file import DesignFileError
from afm_litho.litho.pens import chiral_spacing_um


def _through_a_file(objects, **kw) -> list[LithoObject]:
    doc = design_file.pattern_to_document(
        objects, name="t", saved_at="2026-10-05T00:00:00", **kw)
    text = "".join(design_file.iter_json(doc))
    return design_file.document_to_objects(json.loads(text))


def _assert_same(loaded: LithoObject, original: LithoObject) -> None:
    for f in dataclasses.fields(LithoObject):
        assert getattr(loaded, f.name) == getattr(original, f.name), f.name
    assert loaded == original


def _plain_path():
    return LithoObject("wire", "path", [(0.0, 0.0), (1.25, 0.1), (2.5, -0.3333333333333333)],
                       voltage=6.5, speed_um_s=0.75, interpolation_um=0.01,
                       enabled=False, voltage_end=9.0,
                       end_binding={"node": "end", "port_id": "e3"})


def _bezier_path():
    bez = BezierPath([Node(0.0, 0.0, None, (0.5, 0.4)),
                      Node(2.0, 0.0, (1.5, 0.4), None)], closed=False)
    pts = [(float(x), float(y)) for x, y in bez.flatten(0.004)]
    return LithoObject("curve", "path", pts, bezier=bez.to_dict(), voltage=7.0)


def _chiral_path():
    rec = {"name": "helix", "kind": "path", "voltage": 8.0, "speed_um_s": 1.5,
           "chiral": {"lambda_um": 0.025, "y_amp_um": 0.055, "v_k": 1.5,
                      "phase_deg": 90.0, "axis_um": [[0, 0], [1.0, 0.2], [2.0, 0.0]]}}
    return design_file.record_to_object(rec)


def _shape():
    outline = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
    hole = [(0.5, 0.25), (1.0, 0.25), (1.0, 0.75), (0.5, 0.25)]
    return LithoObject("pad", "shape", outline, holes_um=[hole],
                       fill_kind="radial-out", fill_angle_deg=37.5,
                       fill_centre_um=(1.0, 0.5), fill_spacing_um=0.04,
                       inset_um=0.02, outline_pass=True, voltage=10.0)


def _funnel():
    bez = BezierPath([Node(-1.0, 0.0, None, (-0.6, 0.8)),
                      Node(1.0, 0.0, (0.6, 0.8), None),
                      Node(0.0, 2.0, (0.3, 1.2), (-0.3, 1.2))], closed=True)
    pts = [(float(x), float(y)) for x, y in bez.flatten(0.004)]
    return LithoObject(
        "funnel", "electrode", pts, bezier=bez.to_dict(), fill_spacing_um=0.08)


def _erase_rect():
    return LithoObject("erase", "rect", [(-1.0, -1.0), (1.0, 1.0)], voltage=-8.0,
                       fill_spacing_um=0.1, fill_angle_deg=90.0)


KINDS = {"plain path": _plain_path, "bezier path": _bezier_path,
         "chiral path": _chiral_path, "shape": _shape, "funnel": _funnel,
         "erase rect": _erase_rect}


@pytest.mark.parametrize("build", KINDS.values(), ids=KINDS.keys())
def test_every_field_survives_a_save_and_a_load(build):
    original = build()
    (loaded,) = _through_a_file([original])
    _assert_same(loaded, original)


def test_a_whole_design_keeps_its_order():
    originals = [build() for build in KINDS.values()]
    loaded = _through_a_file(originals)
    assert [o.name for o in loaded] == [o.name for o in originals]
    for got, want in zip(loaded, originals):
        _assert_same(got, want)


def test_a_field_outside_the_named_keys_goes_through_extra():
    obj = LithoObject("taper", "taper", [(0.0, 0.0), (3.0, 0.0)], width0_um=1.5,
                      width1_um=0.3)
    rec = design_file.object_to_record(obj)
    assert rec["extra"] == {"width0_um": 1.5, "width1_um": 0.3}
    _assert_same(_through_a_file([obj])[0], obj)


def test_a_path_with_fill_settings_keeps_them():
    obj = LithoObject("odd", "path", [(0.0, 0.0), (1.0, 0.0)], fill_kind="contour",
                      inset_um=0.5)
    rec = design_file.object_to_record(obj)
    assert "fill" not in rec
    _assert_same(_through_a_file([obj])[0], obj)


def test_points_given_as_lists_load_as_tuples_and_then_stay_put():
    obj = LithoObject("wire", "path", [[0, 0], [1, 2]], holes_um=[[[0, 0], [1, 1]]])
    (once,) = _through_a_file([obj])
    assert once.points_um == [(0.0, 0.0), (1.0, 2.0)]
    assert once.holes_um == [[(0.0, 0.0), (1.0, 1.0)]]
    _assert_same(_through_a_file([once])[0], once)


def test_the_chiral_block_carries_the_parameters_a_script_reads():
    obj = _chiral_path()
    rec = design_file.object_to_record(obj)
    chiral = rec["chiral"]
    assert chiral["lambda_um"] == pytest.approx(0.025)
    assert chiral["y_amp_um"] == pytest.approx(0.055)
    assert chiral["v_k"] == pytest.approx(1.5)
    assert chiral["phase_deg"] == pytest.approx(90.0)
    assert chiral["axis_um"] == obj.axis_um
    assert "axis_um" not in rec.get("extra", {})


def test_a_short_chiral_record_is_built_as_the_cockpit_builds_it():
    obj = _chiral_path()
    wire = _baked([(0.0, 0.0), (1.0, 0.2), (2.0, 0.0)], 0.025, 0.055, 8.0, 1.5, 90.0,
                  name="helix")
    wire.speed_um_s = 1.5
    assert obj == wire and obj.profile["lambda_set_um"] == 0.025
    assert obj.axis_um == [(0.0, 0.0), (1.0, 0.2), (2.0, 0.0)]
    assert obj.interpolation_um == pytest.approx(chiral_spacing_um(0.025, 0.055))
    assert not hasattr(obj, "chiral_clamp")


def test_a_short_bezier_record_is_flattened_as_the_cockpit_flattens_it():
    bez = BezierPath([Node(0.0, 0.0, None, (0.5, 0.4)),
                      Node(2.0, 0.0, (1.5, 0.4), None)], closed=False)
    obj = design_file.record_to_object(
        {"name": "curve", "kind": "path", "bezier": bez.to_dict()})
    assert obj.points_um == [(float(x), float(y)) for x, y in bez.flatten(0.004)]
    assert obj.bezier == bez.to_dict()


def test_a_record_that_carries_points_is_not_rebuilt():
    chiral = _chiral_path()
    rec = _record(chiral)
    rec["points_um"] = [[0.0, 0.0], [5.0, 5.0]]
    rec["chiral"]["profile"] = {"kind": "chiral", "lambda_um": 1.0}
    obj = design_file.record_to_object(rec)
    assert obj.points_um == [(0.0, 0.0), (5.0, 5.0)]
    assert obj.profile == {"kind": "chiral", "lambda_um": 1.0}
    assert obj.axis_um == chiral.axis_um and obj.v_profile is None


# ------------------------------------- what the file stores, and what it drops --

def _record(obj) -> dict:
    return json.loads("".join(design_file.iter_json(design_file.object_to_record(obj))))


def test_a_bezier_object_is_stored_without_its_points():
    for build in (_bezier_path, _funnel):
        rec = _record(build())
        assert "points_um" not in rec and rec["bezier"]["nodes"]


def _baked(axis, lam, amp, v0, v_k, phase_deg, hand=1.0, name="w") -> LithoObject:
    """A chiral wire as `/api/design/chiral` makes it: through `pens.bake_chiral`."""
    from afm_litho.litho.pens import bake_chiral
    wire = LithoObject(name, "path", [tuple(p) for p in axis], voltage=v0)
    bake_chiral(wire, lam, amp, v_k, phase_deg, hand)
    return wire


def _as_the_route_builds(axis, lam, amp, v0, v_k, phase_deg):
    """The fields `/api/design/chiral` sets for this spine and these parameters."""
    wire = _baked(axis, lam, amp, v0, v_k, phase_deg)
    return {"points_um": wire.points_um, "v_profile": wire.v_profile,
            "profile": wire.profile}


def _assert_rebuilt(loaded: LithoObject, original: LithoObject, **params) -> None:
    """`loaded` is what the chiral route builds on the original's spine with its
    parameters and its current voltage, and keeps every other setting."""
    want = _as_the_route_builds(loaded.axis_um, v0=original.voltage, **params)
    for name, value in want.items():
        assert getattr(loaded, name) == value, name
    for f in dataclasses.fields(LithoObject):
        if f.name not in want and f.name != "axis_um":
            assert getattr(loaded, f.name) == getattr(original, f.name), f.name


_HELIX = {"lam": 0.025, "amp": 0.055, "v_k": 1.5, "phase_deg": 90.0}


def test_a_chiral_path_is_stored_as_its_spine_and_parameters():
    obj = _chiral_path()
    rec = _record(obj)
    assert set(rec) == {"name", "kind", "voltage", "speed_um_s", "enabled", "chiral"}
    assert set(rec["chiral"]) == {"lambda_um", "y_amp_um", "v_k", "phase_deg", "axis_um"}
    assert rec["chiral"]["axis_um"] == [[0.0, 0.0], [1.0, 0.2], [2.0, 0.0]]
    assert rec["voltage"] == 8.0 and obj.interpolation_um == chiral_spacing_um(0.025, 0.055)
    assert len(json.dumps(rec)) < 600 < len(obj.points_um)
    (loaded,) = _through_a_file([obj])
    _assert_same(loaded, obj)
    _assert_rebuilt(loaded, obj, **_HELIX)


def _moved(obj, dx, dy):
    """As `pathops.transform` moves a chiral path: its points and its spine."""
    obj.points_um = [(x + dx, y + dy) for x, y in obj.points_um]
    obj.axis_um = [(x + dx, y + dy) for x, y in obj.axis_um]
    return obj


def _edited(change):
    obj = _chiral_path()
    if change == "voltage":
        obj.voltage = 6.0
    elif change == "moved":
        _moved(obj, 1.2345678, -0.7654321)
    elif change == "scaled":
        obj.points_um = [(1.5 * x, 1.5 * y) for x, y in obj.points_um]
        obj.axis_um = [(1.5 * x, 1.5 * y) for x, y in obj.axis_um]
    elif change == "profile":
        obj.profile = dict(obj.profile, normal=[[0.0, 1.0]] * len(obj.profile["normal"]))
    elif change == "points":
        obj.points_um = [(x, y + 0.5) for x, y in obj.points_um]
        obj.v_profile = [v + 1.0 for v in obj.v_profile]
    return obj


@pytest.mark.parametrize("change", ["voltage", "moved", "scaled", "profile", "points"])
def test_a_chiral_path_never_stores_its_points_whatever_was_done_to_it(change):
    obj = _edited(change)
    rec = _record(obj)
    assert not {"points_um", "v_profile"} & set(rec)
    assert "profile" not in rec["chiral"] and "extra" not in rec
    (loaded,) = _through_a_file([obj])
    assert loaded.axis_um == obj.axis_um and loaded.voltage == obj.voltage
    _assert_rebuilt(loaded, obj, **_HELIX)


def test_a_chiral_path_edited_in_voltage_is_rebuilt_about_the_new_voltage():
    obj = _edited("voltage")
    (loaded,) = _through_a_file([obj])
    assert sum(obj.v_profile) / len(obj.v_profile) == pytest.approx(8.0, abs=0.01)
    assert sum(loaded.v_profile) / len(loaded.v_profile) == pytest.approx(6.0, abs=0.01)
    assert loaded.points_um == obj.points_um and loaded.profile["v0"] == 6.0


def test_a_closed_chiral_ring_is_rebuilt_from_its_spine():
    ring = [(3 * math.cos(2 * math.pi * i / 60), 3 * math.sin(2 * math.pi * i / 60))
            for i in range(61)]
    obj = _baked(ring, 0.025, 0.055, 8.0, 1.5, 90.0, name="ring")
    assert obj.profile["closed"] and obj.profile["lambda_um"] != 0.025
    rec = _record(obj)
    assert "points_um" not in rec
    assert rec["chiral"]["lambda_um"] == 0.025 == obj.profile["lambda_set_um"]
    (loaded,) = _through_a_file([obj])
    _assert_same(loaded, obj)
    _assert_same(_through_a_file([loaded])[0], obj)


def test_a_ring_saved_before_the_asked_pitch_was_kept_reloads_at_its_built_pitch():
    ring = [(3 * math.cos(2 * math.pi * i / 60), 3 * math.sin(2 * math.pi * i / 60))
            for i in range(61)]
    obj = _baked(ring, 0.025, 0.055, 8.0, 1.5, 90.0, name="ring")
    del obj.profile["lambda_set_um"]
    assert _record(obj)["chiral"]["lambda_um"] == obj.profile["lambda_um"]
    (loaded,) = _through_a_file([obj])
    assert abs(len(loaded.points_um) - len(obj.points_um)) <= 4
    assert loaded.profile["periods"] == obj.profile["periods"]

def test_a_plain_polyline_and_a_rect_keep_their_points():
    assert _record(_plain_path())["points_um"] == [[0.0, 0.0], [1.25, 0.1],
                                                   [2.5, -0.3333333333333333]]
    assert _record(_erase_rect())["points_um"] == [[-1.0, -1.0], [1.0, 1.0]]


def _assert_close(loaded: LithoObject, original: LithoObject) -> None:
    """Equal in every field, but for the flattened points, which are within the
    file's tolerance and of equal length."""
    for f in dataclasses.fields(LithoObject):
        got, want = getattr(loaded, f.name), getattr(original, f.name)
        if f.name == "points_um":
            assert len(got) == len(want)
            gap = np.abs(np.asarray(got) - np.asarray(want)).max()
            assert gap <= design_file.COORD_TOL_UM
        else:
            assert got == want, f.name


def test_points_within_the_tolerance_of_the_bezier_are_dropped():
    obj = _bezier_path()
    obj.points_um = [(x + 9e-6, y - 9e-6) for x, y in obj.points_um]
    assert "points_um" not in _record(obj)
    (loaded,) = _through_a_file([obj])
    assert loaded != obj
    _assert_close(loaded, obj)


@pytest.mark.parametrize("change", [
    lambda pts: [(x, y + 2e-5) for x, y in pts],
    lambda pts: pts[:3] + [((pts[2][0] + pts[3][0]) / 2, (pts[2][1] + pts[3][1]) / 2)]
    + pts[3:],
    lambda pts: pts[:-1]])
def test_points_that_the_bezier_does_not_rebuild_are_kept(change):
    obj = _bezier_path()
    obj.points_um = change(obj.points_um)
    rec = _record(obj)
    assert len(rec["points_um"]) == len(obj.points_um)
    _assert_same(_through_a_file([obj])[0], obj)


def _left_handed(phase_deg):
    """A chiral path as the cockpit's control makes one with handedness L."""
    return _baked([(0.0, 0.0), (1.0, 0.2), (2.0, 0.0)], 0.025, 0.055, 7.0, 1.5,
                  phase_deg, hand=-1.0, name="left")


@pytest.mark.parametrize("phase_deg", [90.0, 37.3, 123.456, 0.0, 180.0, 270.0])
def test_a_left_handed_chiral_path_rebuilds_exactly(phase_deg):
    obj = _left_handed(phase_deg)
    assert "points_um" not in _record(obj)
    _assert_same(_through_a_file([obj])[0], obj)

def test_a_chiral_path_on_a_bezier_spine_stores_the_bezier_alone_when_it_is_exact():
    bez = BezierPath([Node(0.0, 0.0, None, (0.5, 0.4)),
                      Node(2.0, 0.0, (1.5, 0.4), None)], closed=False)
    axis = [(float(x), float(y)) for x, y in bez.flatten(0.004)]
    obj = design_file.record_to_object({
        "name": "bent", "kind": "path", "bezier": bez.to_dict(), "voltage": 9.0,
        "chiral": {"lambda_um": 0.05, "y_amp_um": 0.04, "v_k": 1.0, "phase_deg": -90.0}})
    assert obj.axis_um == axis and obj.bezier == bez.to_dict()
    rec = _record(obj)
    assert "points_um" not in rec and "axis_um" not in rec["chiral"]
    _assert_same(_through_a_file([obj])[0], obj)


def test_a_spine_the_bezier_only_nearly_gives_is_stored_as_points():
    """A pen-drawn curve: the spine was flattened before the Bezier was rounded."""
    bez = BezierPath([Node(0.0, 0.0, None, (0.5, 0.4)),
                      Node(2.0, 0.0, (1.5, 0.4), None)], closed=False)
    drawn = [(x, y * (1 + 4e-6)) for x, y in
             ((float(x), float(y)) for x, y in bez.flatten(0.004))]
    obj = design_file.record_to_object({
        "name": "bent", "kind": "path", "voltage": 9.0,
        "chiral": {"lambda_um": 0.025, "y_amp_um": 0.055, "v_k": 1.5,
                   "phase_deg": 37.3, "axis_um": drawn}})
    obj.bezier = bez.to_dict()
    rec = _record(obj)
    assert "points_um" not in rec and len(rec["chiral"]["axis_um"]) == len(drawn)
    assert rec["bezier"] == bez.to_dict()
    _assert_same(_through_a_file([obj])[0], obj)


def test_a_chiral_path_with_only_a_bezier_takes_it_as_the_spine():
    obj = design_file.record_to_object({
        "name": "bent", "kind": "path", "voltage": 9.0, "bezier": _bezier_path().bezier,
        "chiral": {"lambda_um": 0.05, "y_amp_um": 0.04, "v_k": 1.0, "phase_deg": 0.0}})
    spine, obj.axis_um = obj.axis_um, None
    rec = _record(obj)
    assert "points_um" not in rec and "axis_um" not in rec["chiral"]
    assert _through_a_file([obj])[0].axis_um == spine

def test_a_chiral_path_with_no_spine_keeps_its_points():
    """The old panel's chiral pen: nothing but the points defines the wire."""
    obj = _chiral_path()
    obj.axis_um = None
    rec = _record(obj)
    assert len(rec["points_um"]) == len(obj.points_um)
    assert rec["chiral"]["profile"]["kind"] == "chiral"
    _assert_same(_through_a_file([obj])[0], obj)

@pytest.mark.parametrize("change", [
    {"interpolation_um": 0.5}, {"speed_um_s": 0.1}, {"enabled": False},
    {"end_binding": {"node": "start", "port_id": "e1"}}])
def test_a_chiral_path_comes_back_with_the_settings_it_had(change):
    obj = dataclasses.replace(_chiral_path(), **change)
    rec = _record(obj)
    assert "points_um" not in rec
    assert ("interpolation_um" in rec) == ("interpolation_um" in change)
    _assert_same(_through_a_file([obj])[0], obj)

def test_a_file_from_the_electrode_pen_still_opens(monkeypatch):
    """Files saved before the electrode pen was removed carry centerline_um,
    half_width_um and generator; they are dropped and a recognised funnel
    Bezier fills as ONE stroke through fill_funnel_from_curves."""
    from afm_litho.litho import fill
    from afm_litho.litho.bezier import parse_svg_path
    from afm_litho.litho.compiler import expand
    bez = parse_svg_path("M 0,0 L 4,0 C 3.5,3 2.5,5 2,6 C 1.5,5 0.5,3 0,0 Z")[0]
    assert bez.as_funnel() is not None
    pts = [(float(x), float(y)) for x, y in bez.flatten(0.004)]
    obj = LithoObject("funnel", "electrode", pts, bezier=bez.to_dict(),
                      fill_spacing_um=0.08)
    rec = _record(obj)
    rec["extra"] = {
        "centerline_um": [[0.0, 0.0], [0.0, 2.0]],
        "half_width_um": [1.0, 0.0],
        "generator": {"kind": "electrode", "width_pad_um": 2.0,
                      "bezier": {"nodes": [], "closed": False}}}
    doc = design_file.pattern_to_document([obj], name="t", saved_at="2026-10-05T00:00:00")
    doc["objects"] = [rec]
    loaded = design_file.document_to_objects(json.loads(json.dumps(doc)))
    _assert_same(loaded[0], obj)
    calls = []
    real = fill.fill_funnel_from_curves
    monkeypatch.setattr(fill, "fill_funnel_from_curves",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    assert len(expand(loaded[0])) == 1
    assert calls == [1]


def test_the_flatten_is_remembered_until_the_object_changes(monkeypatch):
    calls = []
    real = design_file._rebuilds
    monkeypatch.setattr(design_file, "_rebuilds",
                        lambda obj: calls.append(obj.name) or real(obj))
    cache = design_file.RebuildCache()
    objs = [_chiral_path(), _bezier_path(), _plain_path()]
    for _ in range(4):
        design_file.pattern_to_document(objs, name="t", saved_at="now", cache=cache)
    assert calls == ["helix", "curve", "wire"]
    objs[1].points_um = [(x, y + 1.0) for x, y in objs[1].points_um]
    doc = design_file.pattern_to_document(objs, name="t", saved_at="now", cache=cache)
    assert calls == ["helix", "curve", "wire", "curve"]
    assert "points_um" in doc["objects"][1]
    objs[0].voltage = 9.5
    design_file.pattern_to_document(objs, name="t", saved_at="now", cache=cache)
    assert calls[-1] == "helix" and len(calls) == 5


def test_the_memo_forgets_what_no_document_uses_any_more():
    cache = design_file.RebuildCache()
    old, new = _bezier_path(), _bezier_path()
    design_file.pattern_to_document([old], name="t", saved_at="now", cache=cache)
    for _ in range(4):
        design_file.pattern_to_document([new], name="t", saved_at="now", cache=cache)
    held = [entry[0] for entry in cache._entries.values()]
    assert len(held) == 1 and held[0][0] is new.points_um


def test_the_memo_survives_two_threads_writing_at_once(monkeypatch):
    import threading
    calls = []
    real = design_file._rebuilds
    monkeypatch.setattr(design_file, "_rebuilds",
                        lambda obj: calls.append(obj.name) or real(obj))
    cache = design_file.RebuildCache()
    objs = [dataclasses.replace(_bezier_path(), name=f"c{i}") for i in range(40)]

    def write():
        for _ in range(5):
            design_file.pattern_to_document(objs, name="t", saved_at="now", cache=cache)
    threads = [threading.Thread(target=write) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    first = len(calls)
    design_file.pattern_to_document(objs, name="t", saved_at="now", cache=cache)
    assert len(calls) == first <= 4 * len(objs)

def test_negative_zero_and_numpy_values_are_written_safely():
    obj = LithoObject("odd", "path", [(0.0, -0.0), (1.0, 1.0)], inset_um=-0.0,
                      v_profile=np.array([1.0, 2.0, 3.0]))
    rec = design_file.object_to_record(obj)
    assert math.copysign(1.0, rec["extra"]["inset_um"]) == -1.0
    (loaded,) = _through_a_file([obj])
    assert math.copysign(1.0, loaded.inset_um) == -1.0
    assert math.copysign(1.0, loaded.points_um[0][1]) == -1.0
    assert loaded.v_profile == [1.0, 2.0, 3.0]


def test_the_object_holding_a_bad_number_can_be_named():
    good = _plain_path()
    bad = LithoObject("pad", "shape", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
                      fill_spacing_um=math.inf)
    assert design_file.find_non_finite([good]) is None
    assert design_file.find_non_finite([good, bad]) == ("pad", "fill.spacing_um")
    bad = LithoObject("wire-2", "path", [(0.0, 0.0), (math.nan, 0.0)])
    assert design_file.find_non_finite([good, bad]) == ("wire-2", "points_um")


# ------------------------------------------------------ settings an open refuses --

_SQUARE = [[0, 0], [5, 0], [5, 5], [0, 5]]


@pytest.mark.parametrize("rec, field", [
    ({"name": "s", "kind": "shape", "points_um": _SQUARE, "fill": {"spacing_um": 0.0}},
     "fill.spacing_um"),
    ({"name": "s", "kind": "shape", "points_um": _SQUARE, "fill": {"spacing_um": -0.1}},
     "fill.spacing_um"),
    ({"name": "s", "kind": "path", "points_um": _SQUARE, "speed_um_s": -1.0},
     "speed_um_s"),
    ({"name": "s", "kind": "path", "points_um": _SQUARE, "speed_um_s": 0}, "speed_um_s"),
    ({"name": "s", "kind": "path", "points_um": _SQUARE, "interpolation_um": 0.0},
     "interpolation_um"),
    ({"name": "s", "kind": "path", "points_um": _SQUARE,
      "extra": {"fill_spacing_um": 0.0}}, "fill.spacing_um"),
])
def test_a_spacing_speed_or_step_that_is_not_positive_is_refused(rec, field):
    with pytest.raises(DesignFileError) as err:
        design_file.record_to_object(rec)
    assert (err.value.reason, err.value.object_name, err.value.field) == (
        "bad_record", "s", field)
    warnings: list = []
    obj = design_file.record_to_object(rec, warnings=warnings)
    assert len(warnings) == 1 and "'s'" in warnings[0] and field in warnings[0]
    assert obj.points_um == [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]


@pytest.mark.parametrize("field, value", [
    ("points_um", [["1", "2"], ["3", "4"]]),
    ("points_um", [[True, False], [1, 2]]),
    ("points_um", [[0, 0], [1, 2, 3]]),
    ("points_um", [[0, 0], 5]),
    ("points_um", "0,0 1,1"),
    ("points_um", [[0, 0], [1, 10 ** 400]]),
    ("v_profile", ["1", "2"]),
    ("v_profile", [True, 1.0]),
    ("v_profile", [[1.0], [2.0]]),
    ("voltage", "8"),
    ("voltage", True),
    ("voltage", 10 ** 400),
    ("holes_um", [[["a", "b"]]]),
    ("bezier", {"nodes": [{"x": 0.0}], "closed": False}),
    ("bezier", {"nodes": [{"x": 0.0, "y": 0.0}], "closed": False}),
    ("bezier", {"nodes": [{"x": 0.0, "y": 0.0}, {"x": "1", "y": []}]}),
])
def test_a_value_of_the_wrong_type_is_refused(field, value):
    rec = {"name": "a", "kind": "path", "points_um": [[0, 0], [1, 1]], field: value}
    with pytest.raises(DesignFileError) as err:
        design_file.record_to_object(rec)
    assert (err.value.reason, err.value.field) == ("bad_record", field)


def test_a_bezier_too_large_to_be_a_design_is_refused_before_it_is_flattened():
    rec = {"name": "far", "kind": "path", "bezier": {"closed": False, "nodes": [
        {"x": 0.0, "y": 0.0, "h_in": None, "h_out": [1e9, 1e12]},
        {"x": 1.0, "y": 0.0, "h_in": [-1e12, 5.0], "h_out": None}]}}
    with pytest.raises(DesignFileError) as err:
        design_file.record_to_object(rec)
    assert (err.value.field, err.value.object_name) == ("bezier", "far")


def test_a_chiral_record_far_past_the_builders_own_limit_is_refused():
    rec = {"name": "long", "kind": "path",
           "chiral": {"lambda_um": 0.001, "y_amp_um": 0.5, "v_k": 1.0, "phase_deg": 0.0,
                      "axis_um": [[0, 0], [1500, 0]]}}
    with pytest.raises(DesignFileError) as err:
        design_file.record_to_object(rec)
    assert (err.value.reason, err.value.field) == ("object_too_large", "chiral")


def test_the_written_text_is_the_same_json_however_it_is_chunked(monkeypatch):
    doc = design_file.pattern_to_document(
        [build() for build in KINDS.values()], name="t", saved_at="now",
        frame={"scan": {"size_um": 20.0}, "litho": None})
    whole = json.loads("".join(design_file.iter_json(doc)))
    monkeypatch.setattr(design_file, "_JSON_CHUNK", 3)
    assert json.loads("".join(design_file.iter_json(doc))) == whole
    assert whole["object_count"] == len(KINDS)
    assert whole["format"] == "afm-litho-design" and whole["version"] == 1


def _doc(**changes):
    doc = design_file.pattern_to_document([_plain_path()], name="t", saved_at="now")
    doc = json.loads("".join(design_file.iter_json(doc)))
    doc.update(changes)
    return doc


def test_a_newer_version_is_refused():
    with pytest.raises(DesignFileError) as err:
        design_file.document_to_objects(_doc(version=2))
    assert err.value.reason == "unsupported_version"


@pytest.mark.parametrize("doc", [
    _doc(format="something-else"), {"version": 1, "objects": []}, [], "text",
    _doc(version=None), _doc(objects=None)])
def test_what_is_not_a_design_file_is_refused(doc):
    with pytest.raises(DesignFileError) as err:
        design_file.document_to_objects(doc)
    assert err.value.reason == "not_a_design_file"


@pytest.mark.parametrize("field, value", [
    ("points_um", [[0.0, 0.0], [1.0, float("nan")]]),
    ("voltage", float("inf")),
    ("speed_um_s", float("nan")),
    ("v_profile", [1.0, float("-inf")]),
    ("holes_um", [[[0.0, 0.0], [float("nan"), 1.0]]]),
    ("bezier", {"nodes": [{"x": float("nan"), "y": 0.0}], "closed": False}),
])
def test_a_number_that_is_not_finite_refuses_the_whole_open(field, value):
    doc = _doc()
    good = dict(doc["objects"][0], name="good")
    bad = dict(doc["objects"][0], name="bad-one")
    bad[field] = value
    doc["objects"] = [good, bad]
    with pytest.raises(DesignFileError) as err:
        design_file.document_to_objects(doc)
    assert err.value.reason == "bad_record"
    assert err.value.object_name == "bad-one"
    assert err.value.field == field
    assert "bad-one" in str(err.value) and field in str(err.value)


def test_a_number_that_is_not_finite_cannot_be_written():
    obj = LithoObject("wire", "path", [(0.0, 0.0), (1.0, math.nan)])
    doc = design_file.pattern_to_document([obj], name="t", saved_at="now")
    with pytest.raises(ValueError):
        "".join(design_file.iter_json(doc))


@pytest.mark.parametrize("rec, field", [
    ({"name": "", "kind": "path", "points_um": [[0, 0], [1, 1]]}, "name"),
    ({"name": "a", "kind": "blob", "points_um": [[0, 0], [1, 1]]}, "kind"),
    ({"name": "a", "kind": "path"}, "points_um"),
    ({"name": "a", "kind": "path", "points_um": [[0, 0]]}, "points_um"),
    ({"name": "a", "kind": "path", "points_um": [[0, 0], [1, 1]],
      "extra": {"no_such_field": 1}}, "extra.no_such_field"),
    ({"name": "a", "kind": "shape", "points_um": [[0, 0], [1, 1]],
      "fill": {"pitch": 1}}, "fill.pitch"),
    ({"name": "a", "kind": "path", "points_um": [[0, 0], [1, 1]],
      "enabled": "yes"}, "enabled"),
    ({"name": "a", "kind": "path",
      "chiral": {"axis_um": [[0, 0], [1, 0]], "lambda_um": 0.025}}, "chiral.y_amp_um"),
])
def test_a_bad_record_names_its_field(rec, field):
    with pytest.raises(DesignFileError) as err:
        design_file.record_to_object(rec)
    assert err.value.reason == "bad_record"
    assert err.value.field == field


def test_unique_name_is_the_cockpits_rule():
    from afm_litho.service.routers.design import _unique_name
    assert _unique_name is design_file.unique_name
    assert design_file.unique_name("a", []) == "a"
    assert design_file.unique_name("a", ["a"]) == "a-2"
    assert design_file.unique_name("a", ["a", "a-2", "a-4"]) == "a-3"


# ------------------------------------------------------ names, cost and builder --

def test_unique_names_gives_what_unique_name_gives_one_at_a_time():
    import random
    rng = random.Random(7)
    for _ in range(50):
        pool = ["a", "b", "a-2", "a-3", "w", "w-2-2", "b-2"]
        existing = rng.sample(pool, rng.randint(0, len(pool)))
        bases = [rng.choice(["a", "b", "w", "a-2", "w-2"]) for _ in range(40)]
        taken, want = list(existing), []
        for base in bases:
            want.append(design_file.unique_name(base, taken))
            taken.append(want[-1])
        assert design_file.unique_names(bases, existing) == want


def test_naming_many_objects_is_linear():
    import time
    began = time.perf_counter()
    names = design_file.unique_names(["w"] * 20000, ["w"])
    assert time.perf_counter() - began < 0.5
    assert names[0] == "w-2" and names[-1] == "w-20001" and len(set(names)) == 20000


_GOLDEN_BEZIER = [
    (0.0, 0.0), (0.099365234375, 0.0703125), (0.208984375, 0.13125),
    (0.327392578125, 0.18281250000000002), (0.453125, 0.225),
    (0.584716796875, 0.2578125), (0.720703125, 0.28125000000000006),
    (0.859619140625, 0.29531250000000003), (1.0, 0.30000000000000004),
    (1.140380859375, 0.29531250000000003), (1.279296875, 0.28125000000000006),
    (1.415283203125, 0.2578125), (1.546875, 0.225),
    (1.672607421875, 0.18281250000000002), (1.791015625, 0.13125),
    (1.900634765625, 0.0703125), (2.0, 0.0)]
_GOLDEN_CHIRAL = {
    0: ((0.0, 0.0), 6.0),
    7: ((0.014285714285714287, 0.01949855824363647), 4.632496873057757),
    25: ((0.05102040816326531, 0.0025575432336901226), 5.9917900138232465),
    49: ((0.1, 0.0), 6.0)}


def test_the_flatten_a_file_relies_on_has_not_changed():
    """If this fails the rebuild changed: raise `design_file.BUILDER`, so files
    written before can be told apart, and replace these values."""
    obj = design_file.record_to_object({
        "name": "b", "kind": "path", "bezier": {"closed": False, "nodes": [
            {"x": 0.0, "y": 0.0, "h_in": None, "h_out": [0.5, 0.4], "kind": "smooth"},
            {"x": 2.0, "y": 0.0, "h_in": [1.5, 0.4], "h_out": None, "kind": "smooth"}]}})
    assert len(obj.points_um) == len(_GOLDEN_BEZIER)
    for got, want in zip(obj.points_um, _GOLDEN_BEZIER):
        assert got == pytest.approx(want, abs=1e-12)


def test_the_chiral_build_a_file_relies_on_has_not_changed():
    """If this fails the rebuild changed: raise `design_file.BUILDER`, so files
    written before can be told apart, and replace these values."""
    obj = design_file.record_to_object({
        "name": "g", "kind": "path", "voltage": 5.0,
        "chiral": {"lambda_um": 0.05, "y_amp_um": 0.02, "v_k": 1.0, "phase_deg": 90.0,
                   "axis_um": [[0.0, 0.0], [0.1, 0.0]]}})
    assert len(obj.points_um) == len(obj.v_profile) == 50
    assert obj.interpolation_um == pytest.approx(chiral_spacing_um(0.05, 0.02), abs=1e-15)
    for i, (point, volts) in _GOLDEN_CHIRAL.items():
        assert obj.points_um[i] == pytest.approx(point, abs=1e-12)
        assert obj.v_profile[i] == pytest.approx(volts, abs=1e-12)
    assert obj.profile["periods"] == pytest.approx(2.0, abs=1e-12)
    assert obj.profile["phase_rad"] == pytest.approx(math.pi / 2, abs=1e-15)


def test_the_file_says_which_builder_wrote_it_and_a_newer_one_is_refused():
    assert _doc()["builder"] == design_file.BUILDER == 1
    older = _doc()
    del older["builder"]
    assert len(design_file.document_to_objects(older)) == 1
    with pytest.raises(DesignFileError) as err:
        design_file.document_to_objects(_doc(builder=design_file.BUILDER + 1))
    assert err.value.reason == "unsupported_version"
    with pytest.raises(DesignFileError) as err:
        design_file.document_to_objects(_doc(builder="new"))
    assert err.value.reason == "not_a_design_file"


_HUGE_CHIRAL = {"lambda_um": 0.009, "y_amp_um": 0, "v_k": 1.0, "phase_deg": 0.0,
                "axis_um": [[-1000, 0], [1000, 0]]}


def _doc_of(records):
    return {"format": "afm-litho-design", "version": 1, "objects": records}




def test_many_ordinary_chiral_wires_open_whatever_they_add_up_to():
    records = [{"name": f"w{i}", "kind": "path", "chiral": {
        "lambda_um": 0.025, "y_amp_um": 0.055, "v_k": 1.0, "phase_deg": 90.0,
        "axis_um": [[-0.5, i], [0.5, i]]}} for i in range(40)]
    objects = design_file.document_to_objects(_doc_of(records))
    assert len(objects) == 40 and all(len(o.points_um) > 3000 for o in objects)


def test_a_chiral_axis_too_far_out_to_be_a_design_is_refused():
    rec = {"name": "far", "kind": "path",
           "chiral": {"lambda_um": 100.0, "y_amp_um": 0.0, "v_k": 1.0, "phase_deg": 0.0,
                      "axis_um": [[0, 0], [2500, 0]]}}
    with pytest.raises(DesignFileError) as err:
        design_file.record_to_object(rec)
    assert (err.value.reason, err.value.field) == ("bad_record", "chiral.axis_um")
    rec["chiral"]["axis_um"] = [[0, 0], [1999, 0]]
    assert design_file.record_to_object(rec).axis_um == [(0.0, 0.0), (1999.0, 0.0)]


def test_points_too_far_out_to_be_a_design_are_refused_on_open_only():
    rec = {"name": "far", "kind": "path", "points_um": [[0, 0], [2000.5, 0]]}
    with pytest.raises(DesignFileError) as err:
        design_file.record_to_object(rec)
    assert (err.value.reason, err.value.field) == ("bad_record", "points_um")
    assert design_file.record_to_object(rec, warnings=[]).points_um[1] == (2000.5, 0.0)


def test_a_restore_leaves_out_the_record_it_cannot_build_and_keeps_the_rest():
    records = [
        {"name": "good", "kind": "path", "points_um": [[0, 0], [1, 1]]},
        {"name": "huge", "kind": "path", "chiral": dict(_HUGE_CHIRAL, lambda_um=0.0001)},
        {"name": "far", "kind": "path", "bezier": {"closed": False, "nodes": [
            {"x": 0.0, "y": 0.0, "h_in": None, "h_out": [1e9, 1e9]},
            {"x": 1.0, "y": 0.0, "h_in": [-1e9, 5.0], "h_out": None}]}},
        {"name": "also-good", "kind": "path", "points_um": [[0, 1], [1, 2]]}]
    skipped: list = []
    objects = design_file.document_to_objects(_doc_of(records), warnings=[],
                                              skipped=skipped)
    assert [o.name for o in objects] == ["good", "also-good"]
    assert [e.object_name for e in skipped] == ["huge", "far"]


# ------------------------------------------------------------ the text of a file --

def _no_undocumented_keys(doc):
    assert set(doc) <= set(design_file.DOCUMENT_KEYS)
    for rec in doc["objects"]:
        assert set(rec) <= set(design_file.RECORD_KEYS), rec["name"]
        assert set(rec.get("chiral", {})) <= set(design_file.CHIRAL_KEYS)
        assert set(rec.get("fill", {})) <= set(design_file.FILL_KEYS)
        assert set(rec.get("extra", {})) <= {f.name for f in
                                             dataclasses.fields(LithoObject)}


def test_a_file_is_indented_text_with_only_documented_keys():
    objs = [build() for build in KINDS.values()]
    doc = design_file.pattern_to_document(objs, name="t", saved_at="now",
                                          frame={"scan": {"size_um": 20.0}, "litho": None})
    text = "".join(design_file.iter_json(doc))
    assert text.encode("utf-8").decode("utf-8") == text and text.isascii()
    parsed = json.loads(text)
    _no_undocumented_keys(parsed)
    lines = text.split("\n")
    assert lines[0] == "{" and lines[1] == '  "format": "afm-litho-design",'
    assert '      "name": "wire",' in lines and '      "kind": "path",' in lines
    assert "        [1.25, 0.1]," in lines
    assert max(map(len, lines)) < 200
    assert sum(1 for line in lines if line == "    {") == len(objs)


def test_no_chiral_record_ever_holds_points_bias_or_profile():
    objs = [_chiral_path(), _left_handed(37.3)] + [
        _edited(c) for c in ("voltage", "moved", "scaled", "profile", "points")]
    doc = json.loads("".join(design_file.iter_json(
        design_file.pattern_to_document(objs, name="t", saved_at="now"))))
    for rec in doc["objects"]:
        assert set(rec["chiral"]) == {"lambda_um", "y_amp_um", "v_k", "phase_deg",
                                      "axis_um"}
        assert not {"points_um", "v_profile", "extra"} & set(rec)
    _no_undocumented_keys(doc)


# ------------------------------------------------- only what was set is written --

def test_the_unset_parts_of_a_fill_are_left_out():
    plain = LithoObject("pad", "shape", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
    rec = _record(plain)
    assert rec["fill"] == {"spacing_um": 0.15, "kind": "directional"}
    assert rec["interpolation_um"] == 0.02 and "extra" not in rec
    _assert_same(_through_a_file([plain])[0], plain)
    rec = _record(_shape())
    assert rec["fill"] == {"spacing_um": 0.04, "kind": "radial-out", "angle_deg": 37.5,
                           "centre_um": [1.0, 0.5], "inset_um": 0.02,
                           "outline_pass": True}
    assert _record(_erase_rect())["fill"] == {"spacing_um": 0.1, "kind": "directional",
                                              "angle_deg": 90.0}


def test_a_fill_spacing_at_todays_default_is_still_written_and_read_back(monkeypatch):
    """The default is expected to change; a file must not follow it."""
    pad = LithoObject("pad", "shape", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
    doc = design_file.pattern_to_document([pad, _plain_path()], name="t",
                                          saved_at="now")
    text = "".join(design_file.iter_json(doc))
    monkeypatch.setitem(design_file._FIELDS, "fill_spacing_um",
                        dataclasses.field(default=0.1))
    (loaded, wire) = design_file.document_to_objects(json.loads(text))
    assert loaded.fill_spacing_um == 0.15 and wire.interpolation_um == 0.01


def test_a_record_written_before_fields_were_left_out_still_loads():
    old = {"name": "helix", "kind": "path", "voltage": 8.0, "speed_um_s": 1.5,
           "enabled": True, "interpolation_um": chiral_spacing_um(0.025, 0.055),
           "chiral": {"lambda_um": 0.025, "y_amp_um": 0.055, "v_k": 1.5,
                      "phase_deg": 90.0, "axis_um": [[0, 0], [1.0, 0.2], [2.0, 0.0]]}}
    assert design_file.record_to_object(old) == _chiral_path()
    pad = {"name": "pad", "kind": "shape", "points_um": [[0, 0], [1, 0], [1, 1]],
           "voltage": 8.0, "speed_um_s": 2.0, "enabled": True, "interpolation_um": 0.02,
           "fill": {"spacing_um": 0.15, "kind": "directional", "angle_deg": None,
                    "centre_um": None, "inset_um": 0.0, "outline_pass": False}}
    assert design_file.record_to_object(pad) == LithoObject(
        "pad", "shape", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])


# ------------------------------------------------- one absurd object is refused --

_PEN = {"lambda_um": 0.025, "y_amp_um": 0.055, "v_k": 1.0, "phase_deg": 90.0}
_EDGE = [[-45, -45], [45, -45], [45, 45], [-45, 45], [-45, -45]]
_ORDINARY_WIRES = {
    "5 um": ([[-2.5, 0], [2.5, 0]], _PEN, 17_402),
    "90 um, no amplitude": ([[-45, 0], [45, 0]], dict(_PEN, y_amp_um=0), 86_402),
    "scan diagonal": ([[-45, -45], [45, 45]], _PEN, 442_933),
    "diagonal at 10 nm, no amplitude": (
        [[-45, -45], [45, 45]], dict(_PEN, lambda_um=0.010, y_amp_um=0), 305_472),
    "loop round a 90 um square": (_EDGE, _PEN, 1_252_802)}


@pytest.mark.parametrize("axis, params, samples", _ORDINARY_WIRES.values(),
                         ids=_ORDINARY_WIRES.keys())
def test_ordinary_wires_are_far_inside_the_chiral_guard(axis, params, samples):
    from afm_litho.litho.pens import _CHIRAL_MAX_POINTS
    assert design_file.chiral_samples(axis, params) == samples
    assert 3 * samples < _CHIRAL_MAX_POINTS == 5_000_000


@pytest.mark.parametrize("label", ["diagonal at 10 nm, no amplitude",
                                   "loop round a 90 um square"])
def test_the_largest_ordinary_wires_open(label):
    axis, params, samples = _ORDINARY_WIRES[label]
    obj = design_file.record_to_object({"name": "w", "kind": "path", "chiral": {
        **params, "axis_um": axis}})
    assert len(obj.points_um) == samples


def _curve_of_200_nodes(handles_anywhere: bool):
    import random
    rng = random.Random(3)
    nodes = []
    for i in range(200):
        x, y = -45 + 90 * i / 199, rng.uniform(-45, 45)
        if handles_anywhere:
            h_in, h_out = ([rng.uniform(-45, 45), rng.uniform(-45, 45)] for _ in range(2))
        else:
            h_in = [x - 0.225, y + rng.uniform(-5, 5)]
            h_out = [x + 0.225, y + rng.uniform(-5, 5)]
        nodes.append({"x": x, "y": y, "h_in": h_in, "h_out": h_out, "kind": "smooth"})
    return {"closed": False, "nodes": nodes}


_WIDE_FUNNEL = {"closed": True, "nodes": [
    {"x": -45.0, "y": -45.0, "h_in": None, "h_out": [-20.0, 10.0], "kind": "cusp"},
    {"x": 0.0, "y": 45.0, "h_in": [-5.0, 0.0], "h_out": [5.0, 0.0], "kind": "cusp"},
    {"x": 45.0, "y": -45.0, "h_in": [20.0, 10.0], "h_out": None, "kind": "cusp"}]}


@pytest.mark.parametrize("kind, bezier, most", [
    ("path", _curve_of_200_nodes(False), 6_000),
    ("path", _curve_of_200_nodes(True), 60_000),
    ("electrode", _WIDE_FUNNEL, 300)],
    ids=["pen curve of 200 nodes", "200 nodes, handles anywhere", "90 um funnel"])
def test_ordinary_curves_are_far_inside_the_bezier_guard_and_open(kind, bezier, most):
    estimate = design_file.flatten_points(bezier)
    assert estimate <= most and 4 * estimate < design_file.MAX_BEZIER_POINTS
    obj = design_file.record_to_object({"name": "c", "kind": kind, "bezier": bezier})
    assert 2 <= len(obj.points_um) <= estimate


_KNOT = {"closed": False, "nodes": [
    {"x": 1999.0 * (1 if i % 2 else -1), "y": 1999.0 * (1 if i % 4 < 2 else -1),
     "h_in": [-1999.0 * (1 if i % 2 else -1), 1999.0],
     "h_out": [1999.0 * (1 if i % 2 else -1), -1999.0]} for i in range(3000)]}


def test_one_absurd_wire_is_refused_before_it_is_built(monkeypatch):
    import time

    from afm_litho.litho import pens
    monkeypatch.setattr(pens, "chiral_profile", None)
    began = time.perf_counter()
    with pytest.raises(DesignFileError) as err:
        design_file.document_to_objects(_doc_of([
            {"name": "fine", "kind": "path", "points_um": [[0, 0], [1, 1]]},
            {"name": "c0", "kind": "path", "chiral": _HUGE_CHIRAL}]))
    assert time.perf_counter() - began < 0.5
    assert (err.value.reason, err.value.object_name, err.value.field) == (
        "object_too_large", "c0", "chiral")
    said = str(err.value)
    assert "5,333,335" in said and "5,000,000" in said and "instrument" not in said


def test_one_absurd_curve_is_refused_before_it_is_flattened(monkeypatch):
    import time
    monkeypatch.setattr(design_file, "flatten_bezier", None)
    began = time.perf_counter()
    with pytest.raises(DesignFileError) as err:
        design_file.record_to_object({"name": "knot", "kind": "path", "bezier": _KNOT})
    assert time.perf_counter() - began < 1.0
    assert (err.value.reason, err.value.object_name, err.value.field) == (
        "object_too_large", "knot", "bezier")
    assert "6,141,953" in str(err.value) and "250,000" in str(err.value)


def test_the_guards_are_per_object_with_no_total():
    wire = {"name": "w", "kind": "path", "chiral": {**_PEN, "axis_um": [[-0.5, 0], [0.5, 0]]}}
    bezier = _curve_of_200_nodes(True)
    many = design_file.MAX_BEZIER_POINTS // design_file.flatten_points(bezier) + 2
    objects = design_file.document_to_objects(_doc_of(
        [dict(wire, name=f"w{i}") for i in range(5)]
        + [{"name": f"c{i}", "kind": "path", "bezier": bezier} for i in range(many)]))
    assert len(objects) == 5 + many


def test_a_restore_leaves_out_only_the_absurd_object():
    skipped: list = []
    objects = design_file.document_to_objects(_doc_of([
        {"name": "good", "kind": "path", "points_um": [[0, 0], [1, 1]]},
        {"name": "c0", "kind": "path", "chiral": _HUGE_CHIRAL},
        {"name": "knot", "kind": "path", "bezier": _KNOT},
        {"name": "curve", "kind": "path", "bezier": _bezier_path().bezier}]),
        warnings=[], skipped=skipped)
    assert [o.name for o in objects] == ["good", "curve"]
    assert [(e.object_name, e.reason) for e in skipped] == [
        ("c0", "object_too_large"), ("knot", "object_too_large")]


def test_a_wire_the_chiral_door_accepts_is_one_the_file_accepts():
    """90 um at 3 nm pitch and 6.6 nm amplitude: 2,610,002 samples, under the
    builder's own limit."""
    wire = {"lambda_um": 0.003, "y_amp_um": 0.0066, "v_k": 1.0, "phase_deg": 90.0,
            "axis_um": [[-45, 0], [45, 0]]}
    assert design_file.chiral_samples(wire["axis_um"], wire) == 2_610_002
    obj = design_file.record_to_object({"name": "fine", "kind": "path", "chiral": wire})
    assert len(obj.points_um) == 2_610_002


def test_a_curve_past_the_guard_that_the_service_holds_is_saved_with_its_points():
    obj = LithoObject("knot", "path", [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)],
                      bezier={"closed": False, "nodes": _KNOT["nodes"][:200]})
    assert design_file.flatten_points(obj.bezier) > design_file.MAX_BEZIER_POINTS
    rec = _record(obj)
    assert rec["points_um"] == [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]] and "bezier" in rec
    _assert_same(_through_a_file([obj])[0], obj)


@pytest.mark.parametrize("kind", ["shape", "electrode", "rect", "taper"])
def test_a_chiral_wire_of_any_kind_is_stored_as_its_spine(kind):
    obj = dataclasses.replace(_chiral_path(), kind=kind)
    rec = _record(obj)
    assert rec["kind"] == kind and not {"points_um", "v_profile", "extra"} & set(rec)
    assert set(rec["chiral"]) == {"lambda_um", "y_amp_um", "v_k", "phase_deg", "axis_um"}
    _assert_same(_through_a_file([obj])[0], obj)


def test_a_build_stops_between_records_when_it_is_cancelled():
    import threading
    cancelled = threading.Event()
    records = [{"name": f"w{i}", "kind": "path", "points_um": [[0, 0], [1, 1]]}
               for i in range(5)]
    assert len(design_file.document_to_objects(_doc_of(records),
                                               cancelled=cancelled)) == 5
    built = []
    real = design_file.record_to_object

    def build(rec, i, **kw):
        built.append(i)
        if i == 1:
            cancelled.set()
        return real(rec, i, **kw)
    design_file.record_to_object = build
    try:
        with pytest.raises(design_file.BuildCancelled):
            design_file.document_to_objects(_doc_of(records), cancelled=cancelled)
    finally:
        design_file.record_to_object = real
    assert built == [0, 1]
