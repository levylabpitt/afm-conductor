"""Igor procedures read channels by alias, never by a platform-gated literal.

Spec 13.1 item 23 (rig, Cypher, 2026-09-25): the telemetry tick chose its Z channel with
``SelectString(isCypher, "Height", "Output.Z")``; Igor's SelectString returns the SECOND
argument when the test is false, so the Cypher read the dead ``Output.Z`` and the rail-dwell
guard saw zdrive 0 while Height sat at 150 V. The fix reads the ``Height`` alias
unconditionally.

The tick's reads are checked through the expander (behaviour). The SelectString scan stays as
text: the expander replays a model of the procedures, so it cannot see a branch that someone
writes back into the ``.ipf``.
"""

from pathlib import Path

from afm_litho.goldens.expander import expand

IGOR = Path(__file__).resolve().parents[1] / "igor"
FILES = ("afm_litho_run.ipf", "afm_igor.ipf")
CYPHER = 4


def test_cypher_tip_tick_reads_the_height_alias_not_output_z():
    reads = [str(c) for c in expand([("@TIPTICK", ())], microscope_id=CYPHER).reads]
    assert 'td_ReadValue("Height")' in reads, reads
    assert not any("Output.Z" in r for r in reads), reads


def test_no_platform_conditioned_selectstring():
    hits = []
    for name in FILES:
        for n, line in enumerate((IGOR / name).read_text(errors="replace").splitlines(), 1):
            if "SelectString(" in line and (
                "cMicroscopeCypher" in line or "MicroscopeID" in line
            ):
                hits.append(f"{name}:{n}: {line.strip()}")
    assert not hits, "platform-gated SelectString (use the alias):\n" + "\n".join(hits)
