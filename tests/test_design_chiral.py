"""Chiral is a per-PATH modulation with apply / re-parametrize / revert.

`POST /api/design/chiral` makes a plain path chiral, re-parametrizes it, or
reverts it. The load-bearing invariant is AXIS PRESERVATION: the plain axis the
path was drawn/imported as is captured once into `axis_um` on the first enable,
and every regenerate runs against THAT axis, never against the already-modulated
`points_um`. Without it, re-editing the amplitude would modulate an
already-modulated wire and compound every edit (test 4), and disabling would
have nothing to revert to (test 2). Geometry is checked against the
full-resolution `hub.pattern` object, not the coarse to_dict polyline.
"""


from fastapi.testclient import TestClient


def _client():
    from afm_litho.service.app import app
    return TestClient(app)


# A straight axis along x: perpendicular deviation is just |y|, and a straight
# path has ~zero curvature so the amplitude is written in full (no clamp).
AXIS = [(0.0, 0.0), (3.0, 0.0)]


def _seed_path():
    from afm_litho.litho.compiler import LithoObject, Pattern
    from afm_litho.service.hub import hub
    o = LithoObject("wire", "path", list(AXIS), voltage=8.0, speed_um_s=4.0)
    hub.pattern = Pattern(name="seed", objects=[o])
    return hub


def _seed_shape():
    from afm_litho.litho.compiler import LithoObject, Pattern
    from afm_litho.service.hub import hub
    o = LithoObject("region", "shape",
                    [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0), (0.0, 0.0)],
                    voltage=8.0, speed_um_s=4.0)
    hub.pattern = Pattern(name="seed", objects=[o])
    return hub


def _obj():
    """The live, full-resolution object on the hub (not the coarse to_dict)."""
    from afm_litho.service.hub import hub
    return hub.pattern.objects[0]


def _max_dev(points):
    """Max perpendicular deviation from the straight x-axis: max |y|."""
    return max(abs(float(y)) for _x, y in points)


def test_apply_makes_the_path_chiral():
    with _client() as c:
        _seed_path()
        r = c.post("/api/design/chiral",
                   json={"indices": [0], "enabled": True,
                         "lambda_um": 0.05, "y_amp_um": 0.06,
                         "phase_deg": 90.0, "v_k": 5.0})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("ok"), body
        o0 = body["design"]["objects"][0]
        assert o0.get("profile") and o0["profile"]["kind"] == "chiral", o0
        assert abs(o0["profile"]["lambda_um"] - 0.05) < 1e-6, o0["profile"]
        assert isinstance(o0["v_profile"], list) and o0["v_profile"], o0
        # the wire really deviates from its straight axis, ~ the amplitude
        dev = _max_dev(_obj().points_um)
        assert dev > 0.0, dev
        assert abs(dev - 0.06) < 0.01, dev
        # the axis was captured for later revert / re-parametrize
        assert _obj().axis_um is not None
        assert len(_obj().axis_um) == len(AXIS)


def test_disable_reverts_to_the_plain_axis():
    with _client() as c:
        _seed_path()
        c.post("/api/design/chiral",
               json={"indices": [0], "enabled": True,
                     "lambda_um": 0.05, "y_amp_um": 0.06,
                     "phase_deg": 90.0, "v_k": 5.0})
        r = c.post("/api/design/chiral", json={"indices": [0], "enabled": False})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("ok"), body
        o0 = body["design"]["objects"][0]
        assert o0.get("profile") is None, o0
        assert o0.get("v_profile") is None, o0
        # points_um back to the original straight axis, within float tolerance
        o = _obj()
        assert o.axis_um is None
        assert o.profile is None and o.v_profile is None
        for (x, y), (ax, ay) in zip(o.points_um, AXIS):
            assert abs(x - ax) < 1e-9 and abs(y - ay) < 1e-9, o.points_um


def test_non_path_is_rejected_and_unchanged():
    with _client() as c:
        _seed_shape()
        r = c.post("/api/design/chiral",
                   json={"indices": [0], "enabled": True,
                         "lambda_um": 0.05, "y_amp_um": 0.06,
                         "phase_deg": 90.0, "v_k": 5.0})
        assert r.status_code == 200, r.text
        body = r.json()
        assert "error" in body and "path" in body["error"].lower(), body
        # the object was not touched: no profile, no captured axis
        o = _obj()
        assert o.profile is None, o
        assert getattr(o, "axis_um", None) is None, o


def test_reparametrize_runs_against_the_preserved_axis():
    """Apply at A1, then at A2 on the same object. The second deviation must
    match A2, NOT A1+A2 -- proving the regenerate runs against the untouched
    axis, not the already-modulated points_um. This is the axis-preservation
    regression guard."""
    A1, A2 = 0.03, 0.07
    with _client() as c:
        _seed_path()
        c.post("/api/design/chiral",
               json={"indices": [0], "enabled": True,
                     "lambda_um": 0.05, "y_amp_um": A1,
                     "phase_deg": 90.0, "v_k": 5.0})
        dev1 = _max_dev(_obj().points_um)
        assert abs(dev1 - A1) < 0.008, dev1
        # re-apply at a larger amplitude
        c.post("/api/design/chiral",
               json={"indices": [0], "enabled": True,
                     "lambda_um": 0.05, "y_amp_um": A2,
                     "phase_deg": 90.0, "v_k": 5.0})
        dev2 = _max_dev(_obj().points_um)
        # matches A2, and is nowhere near A1 + A2 (which compounding would give)
        assert abs(dev2 - A2) < 0.008, dev2
        assert dev2 < (A1 + A2) - 0.02, dev2
        # the preserved axis is still the original straight one
        assert _max_dev(_obj().axis_um) < 1e-9, _obj().axis_um


def test_undo_removes_the_chiral_modulation():
    with _client() as c:
        _seed_path()
        c.post("/api/design/chiral",
               json={"indices": [0], "enabled": True,
                     "lambda_um": 0.05, "y_amp_um": 0.06,
                     "phase_deg": 90.0, "v_k": 5.0})
        assert _obj().profile is not None
        r = c.post("/api/sketch/undo", json={"count": 1})
        assert r.status_code == 200, r.text
        assert r.json().get("ok"), r.text
        # back to a plain path: profile gone, points equal the axis again
        o = _obj()
        assert o.profile is None, o
        assert getattr(o, "axis_um", None) is None, o
        for (x, y), (ax, ay) in zip(o.points_um, AXIS):
            assert abs(x - ax) < 1e-9 and abs(y - ay) < 1e-9, o.points_um
