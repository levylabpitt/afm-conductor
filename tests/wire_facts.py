"""Reading a golden's recorded wire: the one parser the run tests share.

`golden_actuation`, `wire_steps`, `wave_lines`, `loop2_lines` and `scanchain_args` serve the
builder tests (test_build_write_run, test_run_build_scan); `wire_facts` turns the same lines
into the fact list that `afm_litho.run.expected_wire` projects from a run (test_run_wire_check).
"""

from __future__ import annotations

import json
import re

from golden.harness import HERE

from afm_litho.run.expected_wire import BOUNDS_WAVE, Fact, collapse_repeats

# The Igor handler's output limits for a SERVOLOOP line that carries none
# (igor/afm_litho_run.ipf, SERVOLOOP case): engage and back-off lines omit them.
DEFAULT_LIMITS_V = (-10.0, 150.0)


def golden_actuation(name, *, mfp3d=False):
    suffix = ".mfp3d.golden.json" if mfp3d else ".golden.json"
    return json.loads((HERE / f"{name}{suffix}").read_text())["wire"]["actuation"]


# The wire lines that mark each step, in the order a write sends them.
TOKENS = (
    (r"^SERVOLOOP 2 Deflection -10 0 1000 Height 0 150$", "W"),   # verified withdraw
    (r"^FIRE \w+ XYTransit", "T"),                                # transit move
    (r"^RAMP PIDSLoop\.0\.Setpoint", "R"),                        # hover move
    (r"^SERVOLOOP 2 Deflection -?[\d.]+ 0 1200 Height$", "E"),    # engage or back-off
    (r"^(ARMFIRE|LITHOCHAIN) \w+ (\w+)", "F"),                    # fire
)


def wire_steps_of(lines):
    """The steps of `lines` as words: withdraw, transit, hover, engage or back-off (a setpoint
    on loop 2 before or after a fire), and the verb of each fire."""
    steps, fired = [], False
    for line in lines:
        for pattern, letter in TOKENS:
            m = re.match(pattern, line)
            if not m:
                continue
            if letter == "F":
                steps.append(m.group(1))
                fired = True
            elif letter == "E":
                steps.append("back-off" if fired else "engage")
                fired = False
            else:
                steps.append({"W": "withdraw", "T": "transit", "R": "hover"}[letter])
            break
    return steps


def wire_steps(name):
    return wire_steps_of(golden_actuation(name))


def wave_lines(name, wave):
    return [re.search(r"wave\((\w+),n=(\d+)", line) for line in golden_actuation(name)
            if re.match(rf"^WAVE(APP)? wave\({wave},", line)]


_SERVOLOOP = re.compile(r"^SERVOLOOP (\d) (\S+) (\S+) (\S+) (\S+) (\S+)(?: (\S+) (\S+))?$")


def parse_servoloop(line):
    """(loop, input, setpoint, igain, output, out_min, out_max) of a SERVOLOOP line, with the
    Igor default limits when the line has none; None for any other line."""
    m = _SERVOLOOP.match(line)
    if not m:
        return None
    loop, chan, sp, _p, igain, out, lo, hi = m.groups()
    limits = (float(lo), float(hi)) if lo is not None else DEFAULT_LIMITS_V
    return int(loop), chan, float(sp), float(igain), out, *limits


def loop2_lines_of(lines):
    """Every loop-2 SERVOLOOP of `lines` as (role, setpoint, out_min, out_max): `engage`
    before a fire, `back-off` after it, `withdraw` when it carries the 0 / 150 limits."""
    out, fired = [], False
    for line in lines:
        if re.match(r"^(ARMFIRE|LITHOCHAIN) ", line):
            fired = True
        m = re.match(r"^SERVOLOOP 2 Deflection (\S+) 0 (\d+) (\w+)(?: (\S+) (\S+))?$", line)
        if not m:
            continue
        sp, _igain, _out, lo, hi = m.groups()
        if lo is not None:
            out.append(("withdraw", float(sp), float(lo), float(hi)))
        else:
            out.append(("back-off" if fired else "engage", float(sp), None, None))
            fired = False
    return out


def loop2_lines(name):
    return loop2_lines_of(golden_actuation(name))


def scanchain_args(line):
    """SCANCHAIN <uid> <xwave> <ywave> <ppl> <lines> <interp> <decim> <error channel>, as
    (xwave, ywave, ppl, lines, interp, decim, error channel)."""
    _, _uid, xw, yw, ppl, lines, interp, decim, err = line.split()
    return xw, yw, int(ppl), int(lines), int(interp), int(decim), err


_WAVE = re.compile(r"^WAVE(?:APP)? wave\((\w+),n=(\d+),.*sha=(\w+)\)$")


def is_park(line):
    """The first line a park or abort sends; nothing after it is the run's."""
    return line == "@SAFEPARK" or line.startswith("ABORT")


def wire_facts(lines) -> list[Fact]:
    """The facts of `lines` up to the first park, in the vocabulary of
    `afm_litho.run.expected_wire`, with repeated transits and withdraws collapsed."""
    facts: list[Fact] = []
    xy_loops: dict[int, tuple] = {}
    waves: dict[str, list] = {}          # name -> [wave lines since the last fire, n, sha]
    fired = False
    for line in lines:
        if is_park(line):
            break
        verb = line.split(" ")[0]
        if (loop := parse_servoloop(line)) is not None:
            number, chan, setpoint, igain, out, lo, hi = loop
            if number != 2:
                xy_loops[number] = (number, out, lo, hi)
            elif (lo, hi) != DEFAULT_LIMITS_V:
                facts.append(("withdraw", setpoint, lo, hi))
            elif fired:
                facts.append(("back-off", setpoint, lo))
            else:
                sign = (igain > 0) - (igain < 0)
                facts.append(("engage", number, chan, out, setpoint, lo, hi, sign))
            fired = fired and number != 2
        elif verb == "FIRE":
            facts.append(("move", "transit"))
            waves.clear()
        elif line.startswith("RAMP PIDSLoop.0.Setpoint"):
            facts.append(("move", "hover"))
        elif m := _WAVE.match(line):
            name, n, sha = m.group(1), int(m.group(2)), m.group(3)
            waves[name] = [waves.get(name, [0])[0] + 1, n, sha]
        elif verb in ("ARMFIRE", "LITHOCHAIN", "SCANCHAIN"):
            facts.append(_fire_fact(line.split(" "), waves, xy_loops))
            waves.clear()
            fired = verb != "SCANCHAIN"
    return collapse_repeats(facts)


def _fire_fact(tok, waves, xy_loops) -> Fact:
    verb = tok[0]
    if verb == "SCANCHAIN":
        xw, _yw, ppl, lines, interp, decim, err = scanchain_args(" ".join(tok))
        count, n, _sha = waves[xw]
        return ("scan", verb, ppl, lines, interp, decim, err, n, count,
                tuple(xy_loops[k] for k in sorted(xy_loops)))
    count, n, _sha = waves["XLitho"]
    if verb == "ARMFIRE":                      # ARMFIRE <uid> <name> <interp> <cap_len> <decim>
        return ("litho", verb, tok[2], n, count, int(tok[4]), None)
    _count, bounds_n, bounds_sha = waves[tok[6]]   # LITHOCHAIN <uid> <name> X Y V <bounds> ..
    assert tok[6] == BOUNDS_WAVE, tok
    return ("litho", verb, tok[2], n, count, None, (bounds_n, bounds_sha))
