"""The raster pass: the tip path and its conversion to LVDT setpoint volts."""

import pytest


# ---- raster as a pass ----

def test_raster_path_covers_the_frame():
    from afm_litho.pass_executor import raster_path_um
    px, ln, size = 16, 8, 30.0
    x, y = raster_path_um(px, ln, size)
    assert len(x) == px * ln
    assert x.min() == pytest.approx(-size / 2) and x.max() == pytest.approx(size / 2)
    assert y.min() == pytest.approx(-size / 2) and y.max() == pytest.approx(size / 2)


def test_raster_snake_reverses_alternate_lines():
    from afm_litho.pass_executor import raster_path_um
    px, ln = 5, 3
    x, _ = raster_path_um(px, ln, 10.0)
    assert x[:px][0] < x[:px][-1]              # line 0 left-to-right
    assert x[px:2 * px][0] > x[px:2 * px][-1]  # line 1 reversed (snake)


def test_raster_pass_converts_to_volts():
    from afm_litho.pass_executor import raster_pass
    cal = {"x_lvdt_sens": 7.4643e-6, "x_lvdt_offset": -0.63,
           "y_lvdt_sens": 7.0428e-6, "y_lvdt_offset": 0.07}
    px, ln = 12, 4
    p = raster_pass(px, ln, 5.0, cal, interp=-1, capture=["XSensor"])
    assert p.line_len == px and len(p.x_v) == px * ln
    # LVDT conversion: centre pixel maps near the offset volt
    assert p.x_v.mean() == pytest.approx(cal["x_lvdt_offset"], abs=1e-3)

