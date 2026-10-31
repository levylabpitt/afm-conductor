"""A sample keeps its own marks, and does not inherit anybody else's.

Two complaints from the lab, and they are the same bug seen from two sides.

"i can't erase contacts": /api/contacts erased them and wrote the empty list
to disk, and within that session they stayed gone. But `restore_contacts` was
only ever called from the sample-SWITCHING endpoint, never for the sample the
Hub loads at import, so the default sample never read its file back. Erase all
forty, restart, and the detector's forty are there again.

"when i load a new sample all the old stuff should be erased by default":
every sample shared one `RUNTIME/raw`, so a new sample's Surface filled in
with the previous sample's topography. On an instrument that is not
untidiness. It is a measurement of a different piece of oxide drawn as though
it were this one.
"""

from __future__ import annotations

import json
from pathlib import Path


from afm_litho.service import app as appmod  # noqa: F401  (unused name; import kept as it was)
from afm_litho.service import deps as depsmod
from afm_litho.service.routers import sample as samplemod


def test_erasing_every_contact_survives_a_restart(tmp_path: Path, monkeypatch):
    """The tri-state is the point: erased is not the same as never said."""
    monkeypatch.setattr(samplemod, "CONTACTS_DIR", tmp_path)
    monkeypatch.setattr(samplemod, "_contacts_path",
                        lambda name: tmp_path / f"{name}.json")

    class FakeSample:
        def __init__(self):
            self.pills = None
        def set_contacts(self, pills):
            self.pills = list(pills)

    # Nobody has said anything yet: the detector's guess is the best answer.
    assert samplemod.restore_contacts(FakeSample(), "S") is None

    # Erased, and written down as erased.
    (tmp_path / "S.json").write_text(json.dumps([]), encoding="utf-8")
    s = FakeSample()
    assert samplemod.restore_contacts(s, "S") == 0, "an erasure read as silence"
    assert s.pills == [], "the erasure was not applied to the sample"

    # And marks come back as marks.
    (tmp_path / "S.json").write_text(
        json.dumps([{"x_um": 1, "y_um": 2, "w_um": 3, "h_um": 1, "angle_deg": 0}]),
        encoding="utf-8")
    s2 = FakeSample()
    assert samplemod.restore_contacts(s2, "S") == 1
    assert len(s2.pills) == 1


def test_two_samples_do_not_share_their_raw_passes():
    a = depsmod.raw_dir_for("SA40656B0000")
    b = depsmod.raw_dir_for("some other sample")
    assert a != b, "a new sample would rebuild its Surface from the old one's scans"
    assert a.parent == b.parent == depsmod.RUNTIME / "raw"


def test_a_sample_name_that_is_not_a_filename_still_gets_a_directory():
    """`lead-source[1/2]` in a filename has cost this project a run before."""
    d = depsmod.raw_dir_for("SA/40656B:0000 *odd*")
    assert "/" not in d.name and ":" not in d.name and "*" not in d.name
    assert d.name
