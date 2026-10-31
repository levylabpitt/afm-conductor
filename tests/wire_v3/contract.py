"""Wire v3, as the contract writes it (docs/migration/plan/phase-5-wire-v3-contract.md).

Pure Python, no Igor and no service: the constants, the line codec for the up-messages, the
builders for the down-messages, and the one place where a v3 pass is allowed to end in
different Igor commands than the frozen v2 goldens (the make-safe, section 3a and 5a). Every
conformance test imports its numbers from here, so a contract change is one edit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PROTO = 3                    # section 2a
RING_SIZE = 256              # section 2c: the last 256 REPORTs
PIECE_POINTS = 8192          # section 2b/2d: LOAD, LOADAPP and CAP piece size
CONFIRM_WINDOW_S = 2.5       # section 5a: Python polls Height this long before VENDORWITHDRAW
BACKUP_S = 5.0               # section 5a: Igor's own backup task
SOCK_DEAD_PARK_S = 5.0       # section 10 answer 1: kSockDeadParkS
HEIGHT_SAFE_V = 1.0          # section 5a: below this Height the tip counts as retracted
PIECE_TIMEOUT_S = 10.0       # section 2d: the client's piece timeout, to start with
TICK_HZ_PASS, TICK_HZ_IDLE = 10.0, 4.0     # section 2c

STAMPED = ("tick", "useq", "mark", "fault", "route")   # section 1, on every up-message
SAFE_SET = ("STARTUP", "GETVALUE", "ABORT")            # section 7, plus AFM_SafePark over COM
HOST_ONLY = ("LOAD", "LOADAPP", "PLAY", "SET")         # section 1, rights
PLAY_KINDS = ("move", "litho1", "lithochain", "scanchain")


def num(v: float) -> str:
    """Numbers on the wire are %.12g (section 1)."""
    return f"{float(v):.12g}"


def wave_body(values) -> str:
    return ";".join(num(v) for v in values)


def down(seq: int, verb: str, *args) -> str:
    """One down-message line: `VERB seq args...`."""
    return " ".join([verb, str(seq), *(str(a) for a in args)])


def parse_kv(body: str) -> dict[str, str]:
    """`k=v;k=v;` to a dict (HELLO, STATUS and the startup reply)."""
    out = {}
    for part in body.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


@dataclass(frozen=True)
class Up:
    kind: str                  # REPLY, REPORT, CAP, TICK
    raw: str
    seq: int | None = None     # REPLY
    ok: bool | None = None     # REPLY
    body: str = ""             # REPLY body, REPORT fields, CAP data, TICK reads
    useq: int | None = None
    tick: int | None = None
    mark: int | None = None
    fault: int | None = None
    route: str | None = None
    run: str | None = None
    event: str | None = None   # REPORT
    step: str | None = None    # CAP
    alias: str | None = None   # CAP


def parse_up(line: str) -> Up:
    """Parse one up-message per section 2c. Raises ValueError on anything else."""
    t = line.rstrip("\r\n").split(" ")
    kind = t[0]
    try:
        if kind == "REPLY":
            return Up(kind, line, seq=int(t[1]), ok=(t[2] == "ok"), body=" ".join(t[3:]))
        if kind == "REPORT":
            # REPORT <useq> <tick> <mark> <fault> <route> <run> <event> <fields>
            return Up(kind, line, useq=int(t[1]), tick=int(t[2]), mark=int(t[3]),
                      fault=int(t[4]), route=t[5], run=t[6], event=t[7],
                      body=" ".join(t[8:]))
        if kind == "CAP":
            # CAP <useq> <tick> <mark> <run> <step> <alias> <v;v;...>
            return Up(kind, line, useq=int(t[1]), tick=int(t[2]), mark=int(t[3]), run=t[4],
                      step=t[5], alias=t[6], body=" ".join(t[7:]))
        if kind == "TICK":
            # TICK <useq> <tick> <mark> <fault> - <x> <y> <z> <defl> <sum> <amp>
            return Up(kind, line, useq=int(t[1]), tick=int(t[2]), mark=int(t[3]),
                      fault=int(t[4]), route=t[5], body=" ".join(t[6:]))
    except (IndexError, ValueError) as exc:
        raise ValueError(f"malformed {kind}: {line!r}") from exc
    raise ValueError(f"not a v3 up-message: {line!r}")


# ---- make-safe (section 3a "Park and abort", section 5a) -------------------------------

TD_STOP = "td_Stop()"
BIAS_ZERO = 'td_WriteValue("Output.A",0)'
RETRACT = 'ir_SetPISLoop(2,"Always,Never","Deflection",-10,0,1000,0,"Height",0,150)'
MAKE_SAFE_LINES = (TD_STOP, BIAS_ZERO, RETRACT)   # td_Stop only when abortStopped was 0

# What the contract deletes from the park (defect 1, section 3a): Python's STOPLOOP and its
# Height ramp. The v2 goldens frozen before slice AB still hold them.
_RETIRED_PARK = (re.compile(r"^ir_StopPISLoop\(2\)$", re.I),
                 re.compile(r'^IR_SetRamp\(2,"Height"', re.I))


def retired_park_lines(actuation: list[str]) -> list[int]:
    return [i for i, ln in enumerate(actuation) if any(r.match(ln) for r in _RETIRED_PARK)]


def contract_igor(frozen_actuation: list[str]) -> list[str]:
    """The `igor.actuation` a v3 pass must end in, derived from the frozen v2 one.

    Identical except that the contract's make-safe has no STOPLOOP and no Height ramp
    (section 3a, the abort redesign). Once slice AB re-freezes the park goldens they hold no
    such lines and this returns its input unchanged, so it is idempotent on a new golden.
    """
    drop = set(retired_park_lines(frozen_actuation))
    return [ln for i, ln in enumerate(frozen_actuation) if i not in drop]
