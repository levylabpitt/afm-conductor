"""The run as text: header lines, then one numbered line per step or pass.

This is the form the operator reads (docs/migration/plan/phase-3-run.md has the agreed mockups). It is
for people only; the JSON form is the stored one. Every name and unit is typeable ASCII.
Gains are not shown: Python still sends them, unchanged, but they are not part of the run.
"""

from __future__ import annotations

import math
import textwrap

from .json_form import recipe_fingerprint
from .model import (Check, End, Engage, Entry, LithoObject, Move, Pass, Read, Recipe, Run,
                    Scan, SetMode, Settle, Setpoint, Withdraw)

WIDTH = 96
HEAD = 11  # header label column
STEP = 14  # "NN  label     " before a step's text
SHORT = 8  # hex characters shown of a fingerprint

CHECK_LABELS = {
    "sum-below-ceiling": "SUM below ceiling",
    "xy-calibration-refreshed": "XY calibration refreshed",
    "no-latched-fault": "no latched fault",
    "frame-inside-xy-range": "frame inside the XY range",
}

WITHDRAW_FIRST = {
    "none": "",
    "verified": ", verified withdraw first",
    "verified-if-engaged": ", verified withdraw first if engaged",
}


def num(x: float) -> str:
    """A number as typed: 150 not 150.0, 0.05 not 5e-02."""
    return str(int(x)) if float(x).is_integer() else repr(float(x))


def signed(x: float) -> str:
    """A relative setpoint always shows its sign, so +0.05 is not read as absolute 0.05."""
    return ("+" if x >= 0 else "-") + num(abs(x))


def _settle(s: float | None) -> str:
    """Settle times are code constants today; they are shown so the run hides no wait."""
    return "" if s is None else f" | settle {num(s)} s"


def _aside(text: str) -> str:
    return f" ({text})" if text else ""


def setpoint_flags(sp: Setpoint) -> str:
    """The operator's command-line form, e.g. `--relative --setpoint=+0.05`."""
    value = signed(sp.volts) if sp.reference == "relative" else num(sp.volts)
    return f"--{sp.reference} --setpoint={value}"


def setpoint_words(sp: Setpoint) -> str:
    if sp.reference == "relative":
        return f"relative {signed(sp.volts)} V above free air"
    return f"absolute {num(sp.volts)} V"


# --- one line per body entry ---------------------------------------------------


def check_text(e: Check) -> str:
    return " | ".join(CHECK_LABELS.get(i, i.replace("-", " ")) for i in e.items)


def read_text(e: Read) -> str:
    """`[run time]` marks a value only the execution knows; it is journalled, not stored here."""
    text = "free air: " + ", ".join(e.channels)
    if e.agreeing is not None:
        text += f", {e.agreeing} agreeing reads"
    if e.z_drive_max_v is not None:
        text += f", Z drive < {num(e.z_drive_max_v)} V"
    return text + "  [run time]"


def move_text(e: Move) -> str:
    text = f"{e.how} to {e.to}"
    if e.speed_um_s is not None:
        text += f" at {num(e.speed_um_s)} um/s"
    text += WITHDRAW_FIRST[e.withdraw_first]
    if e.hold_v is not None:
        text += f" (holding at {num(e.hold_v)} V)"
    return text + _settle(e.settle_s)


def settle_text(e: Settle) -> str:
    return f"{num(e.seconds)} s{_aside(e.why)}"


def setmode_text(e: SetMode) -> str:
    return f"{e.mode}{_aside(e.when)}{_settle(e.settle_s)}"


def engage_text(e: Engage) -> str:
    text = f"{setpoint_flags(e.setpoint)} | loop {e.loop} on {e.on}"
    if e.sign is not None:
        text += f", sign {e.sign}"
    lo, hi = e.limits_v
    text += f", {e.out} out, limits {num(lo)} to {num(hi)}"
    return text + _settle(e.settle_s) + _aside(e.when)


def withdraw_text(e: Withdraw) -> str:
    if e.how == "verified":
        text = f"verified, holding at {num(e.hold_v)} V"
    else:
        text = f"holding at {num(e.hold_v)} V"
        if e.setpoint is not None:
            sp = e.setpoint
            value = num(sp.volts) if sp.reference == "absolute" else f"relative {signed(sp.volts)}"
            text = f"setpoint {value} V, " + text
    return text + _aside(e.why) + _settle(e.settle_s)


def end_text(e: End) -> str:
    return "stays engaged (no withdraw)" if e.engaged else "withdrawn"


def litho_text(e: LithoObject) -> str:
    """One object's pass. A chained write keeps the tip in contact at full bias across every
    seam (measured about 1 ms), so the text says so rather than implying a lift."""
    text = f'litho object {e.index} "{e.name}": {e.shape}'
    text += "".join(f", {p.name} {num(p.value)} {p.unit}" for p in e.params)
    text += f", {e.points:,} points"
    if e.speed_um_s is not None:
        text += f", {num(e.speed_um_s)} um/s"
    if e.chunks == 1:
        text += f", 1 chunk ({e.fire})"
    else:
        cuts = " and ".join(f"{c:,}" for c in e.chunk_cuts)
        text += (f", {e.chunks} chunks ({e.fire}, cut at {cuts}),"
                 " bias on across the seam")
    return (text + f", ends at {num(e.ends_at_v)} V, waves loaded in"
            f" {e.wave_line_points:,}-point lines, points {e.points_fingerprint[:SHORT]}")


def scan_text(e: Scan) -> str:
    """Each X/Y out-wave holds the whole raster and is loaded in `wave_line_points` lines."""
    per_wave = math.ceil(e.lines * e.points_per_line / e.wave_line_points)
    lo, hi = e.limits_v
    return (f"scan: {e.lines} lines x {e.points_per_line} points, interpolation"
            f" {e.interpolation}, decimation {e.decimation}, error channel {e.error_channel},"
            f" XY loops {e.xy_loops[0]}/{e.xy_loops[1]} on"
            f" {e.xy_outputs[0]}/{e.xy_outputs[1]}, limits {num(lo)} to {num(hi)}, waves loaded"
            f" in {e.wave_line_points:,}-point lines ({per_wave} per wave), one {e.fire} command")


ENTRY_TEXT = {
    Check: check_text, Read: read_text, Move: move_text, Settle: settle_text,
    SetMode: setmode_text, Engage: engage_text, Withdraw: withdraw_text, End: end_text,
    LithoObject: litho_text, Scan: scan_text,
}


def entry_line(n: int, e: Entry) -> list[str]:
    """`NN  label     text`, wrapped under the text; passes are labelled PASS to stand out."""
    label = "PASS" if isinstance(e, Pass) else e.KIND
    return textwrap.wrap(ENTRY_TEXT[type(e)](e), WIDTH, initial_indent=f"{n:>2}  {label:<10}",
                         subsequent_indent=" " * STEP, break_on_hyphens=False)


# --- the header ------------------------------------------------------------------


def _head(label: str, *parts: str) -> list[str]:
    """A header line; each later part starts its own continuation line."""
    lines = textwrap.wrap(parts[0], WIDTH, initial_indent=label.ljust(HEAD),
                          subsequent_indent=" " * HEAD, break_on_hyphens=False)
    for part in parts[1:]:
        lines += textwrap.wrap(part, WIDTH, initial_indent=" " * HEAD,
                               subsequent_indent=" " * HEAD, break_on_hyphens=False)
    return lines


def capture_parts(r: Recipe) -> list[str]:
    """The channel list, then the pair's bank and fallback ladder on its own line. The ladder
    is shown because a missing Lateral otherwise looks like a lost capture."""
    c = r.capture
    names = list(c.channels) + ([" + ".join(c.pair)] if c.pair else [])
    parts = ["every pass: " + ", ".join(names)]
    if c.pair:
        ladder = [" + ".join(f) + (" alone" if len(f) == 1 else "") if f else "none"
                  for f in c.pair_fallbacks]
        note = f"pair on in-bank {c.pair_bank}"
        if ladder:
            note += "; falls back to " + ", then ".join(ladder)
        parts.append(f"({note})")
    return parts


def header_lines(run: Run) -> list[str]:
    """Labels in an 11-column gutter; write-only lines (design, compile, placement, bias) and
    the scan-only frame line appear only when the recipe has them."""
    r, x = run.recipe, run.execution
    lines = _head("run", f"{r.kind} | recipe {recipe_fingerprint(r)[:SHORT]}"
                         f" | execution {x.execution_id}")
    if r.design:
        d = r.design
        lines += _head("design", f'"{d.name}" rev {d.rev} | {d.objects} objects'
                                 f" | snapshot {d.snapshot_fingerprint[:SHORT]}")
        lines += _head("compile", f"settings {d.compile_settings_fingerprint[:SHORT]}"
                                  f" | compiler {d.compiler_version}")
    lines += _head("platform", f"{r.platform} | {r.mode} mode")
    if r.placement:
        p = r.placement
        x0, y0 = p.centre_um
        lines += _head("placement", f"from {p.source} ({num(p.size_um)} um,"
                                    f" {num(p.angle_deg)} deg, centre {num(x0)}, {num(y0)} um)")
    if r.frame:
        f = r.frame
        text = (f"from {f.source}: {num(f.size_um)} um, {f.lines} x {f.pixels},"
                f" {num(f.angle_deg)} deg, offset {num(f.offset_um[0])}, {num(f.offset_um[1])} um"
                + (" (+ drift offset)" if f.drift_offset else "")
                + f", {f.direction}, {num(f.line_rate_hz)} line/s"
                + (", trace and retrace" if f.trace_and_retrace else ", trace only"))
        lines += _head("frame", text)
    lines += _head("setpoint", setpoint_words(r.setpoint))
    if r.bias:
        lo, hi = r.bias.window_v
        lines += _head("bias", f"amp gain {num(r.bias.amp_gain)} | tip-voltage window"
                               f" {num(lo)} to {num(hi)} V")
    lines += _head("capture", *capture_parts(r))
    lines.append(f"if script goes silent   {r.script_gone}{_aside(r.script_gone_note)}")
    return lines


def run_text(run: Run) -> str:
    lines = header_lines(run)
    for n, entry in enumerate(run.recipe.body):
        lines += entry_line(n, entry)
    return "\n".join(lines) + "\n"
