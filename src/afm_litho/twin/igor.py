"""Igor command-string front end for the software ARC.

The real system crosses the LabVIEW->Igor boundary as Execute2("...") command
strings. This module accepts those same strings, so the annotated capture
(docs/reference/igor-commands-commented.txt) replays against the twin
verbatim. Supported grammar is exactly what the captures use: td_*/ir_*
calls, GV(), fprintf readbacks, Make, LoadWave, wave assignment, display.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .arc import SoftwareARC

_NUM = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"

# Command grammars, compiled once. `execute` matches these in order after the
# keyword-led commands; `_eval` reuses _CALL_RE for its GV()/td_Read* readbacks.
_CALL_RE = re.compile(r"(\w+)\s*\((.*)\)\s*$")            # name(args)
_LITERAL_RE = re.compile(r"(\w+)\s*=\s*\{(.*)\}\s*$")     # name = {v, v, ...}
_COPY_RE = re.compile(r"(\w+)\s*=\s*(\w+)\s*$")           # name = other_name


@dataclass
class IgorSession:
    arc: SoftwareARC
    wave_resolver: Callable[[str], np.ndarray] | None = None
    waves: dict[str, np.ndarray] = field(default_factory=dict)
    printed: list[str] = field(default_factory=list)
    executed: int = 0

    # ------------------------------------------------------------------ API --

    def execute(self, cmd: str) -> None:
        cmd = cmd.strip()
        if not cmd or cmd.startswith("//"):
            return
        self.executed += 1

        # keyword-led commands dispatch on their leading word (checked first, so a
        # Make/LoadWave/etc. is never mistaken for a function call).
        for prefix, handler in self._KEYWORD_CMDS:
            if cmd.startswith(prefix):
                handler(self, cmd)
                return
        # pattern-led commands: the first grammar that matches wins.
        for pattern, action in self._PATTERN_CMDS:
            m = pattern.match(cmd)
            if m is not None:
                action(self, m)
                return
        raise ValueError(f"igor twin: unhandled command: {cmd!r}")

    def execute_script(self, text: str) -> None:
        for line in text.splitlines():
            self.execute(line)

    # ------------------------------------------------------------- internals --

    def _fprintf(self, cmd: str) -> None:
        m = re.match(r'fprintf\s+0\s*,\s*"([^"]*)"\s*,\s*(.+)$', cmd)
        if not m:
            raise ValueError(f"igor twin: bad fprintf: {cmd!r}")
        fmt, expr = m.groups()
        val = self._eval(expr.strip())
        try:
            self.printed.append(fmt % val)
        except TypeError:
            self.printed.append(str(val))

    def _eval(self, expr: str):
        m = _CALL_RE.match(expr)
        if not m:
            return float(expr)
        name, argstr = m.groups()
        args = self._args(argstr)
        match name:
            case "GV":
                return self.arc.GV(args[0])
            case "td_ReadValue" | "td_RV":
                return self.arc.td_ReadValue(args[0])
            case "td_ReadString" | "td_RS":
                return self.arc.td_ReadString(args[0])
            case "ImageModeList":
                return "Contact;AC Mode;"
            case _:
                raise ValueError(f"igor twin: unknown function in expr: {expr!r}")

    def _call(self, name: str, argstr: str) -> None:
        args = self._args(argstr)
        arc = self.arc
        match name:
            case "td_WriteValue" | "td_WV":
                arc.td_WriteValue(args[0], float(args[1]))
            case "PV":                           # MasterVariablesWave global write
                arc.PV(args[0], float(args[1]))
            case "td_WriteString" | "td_WS":
                arc.td_WriteString(args[0], str(args[1]))
            case "ir_SetPISLoop":
                arc.ir_SetPISLoop(int(args[0]), str(args[1]), str(args[2]), float(args[3]),
                                  float(args[4]), float(args[5]), float(args[6]),
                                  str(args[7]), float(args[8]), float(args[9]))
            case "ir_StopPISLoop":
                arc.ir_StopPISLoop(int(args[0]))
            case "td_xSetOutWavePair":
                arc.td_xSetOutWavePair(int(args[0]), str(args[1]), str(args[2]),
                                       self._wave(args[3]), str(args[4]),
                                       self._wave(args[5]), int(float(args[6])))
            case "td_xSetOutWave":
                arc.td_xSetOutWave(int(args[0]), str(args[1]), str(args[2]),
                                   self._wave(args[3]), int(float(args[4])))
            case "td_Stop" | "td_SoftReset":
                arc.td_Stop()
            case "ARExecuteControl" | "ARCheckFunc":
                pass  # GUI panel toggles with no twin state: no-op
            case "ARExecuteControl2":
                # (ControlName, Proc, Graph, InfoStr, TypeNum, VarNum, VarStr)
                arc.fire_control(args[0], proc=args[1],
                                 var_num=float(args[5]) if len(args) > 5 else 0.0,
                                 var_str=args[6] if len(args) > 6 else "")
            case ("DoScanFunc" | "SimpleEngageMe" | "DoForceFunc"
                  | "CantTuneFunc" | "MainPopupFunc"):
                arc.fire_control(args[0], proc=name)  # a control's action proc, called directly
            case "td_StopOutWaveBank":
                arc.td_StopOutWaveBank(int(args[0]))
            case "td_StopInWaveBank":
                arc.td_StopInWaveBank(int(args[0]))
            case _:
                raise ValueError(f"igor twin: unknown call: {name}({argstr})")

    # literal wave: XMove = {0.0, -1.93}
    def _assign_literal(self, m: "re.Match") -> None:
        vals = [float(v) for v in re.findall(_NUM, m.group(2))]
        self.waves[m.group(1).lower()] = np.array(vals, dtype=np.float64)

    # wave copy: Xmove = Xmove_temp (Igor names are case-insensitive)
    def _assign_copy(self, m: "re.Match") -> None:
        src = m.group(2).lower()
        if src not in self.waves:
            raise KeyError(f"igor twin: wave copy from unknown wave: {m.group(2)}")
        self.waves[m.group(1).lower()] = self.waves[src].copy()

    def _wave(self, token) -> np.ndarray:
        name = str(token).lower()
        if name in self.waves:
            return self.waves[name]
        raise KeyError(f"igor twin: wave not found: {name}")

    def _make(self, cmd: str) -> None:
        m = re.match(r"Make(?:/\w+)*?/N=(\d+)\s+(.+)$", cmd)
        if not m:
            raise ValueError(f"igor twin: bad Make: {cmd!r}")
        n = int(m.group(1))
        names = m.group(2).split()
        for name in names:
            self.waves[name.lower()] = np.zeros(n, dtype=np.float64)
        self._last_make = names[-1].lower()

    def _loadwave(self, cmd: str) -> None:
        m = re.search(r'"([^"]+)"', cmd)
        if not m:
            raise ValueError(f"igor twin: bad LoadWave: {cmd!r}")
        path = m.group(1)
        if self.wave_resolver is None:
            raise RuntimeError("igor twin: LoadWave used without a wave_resolver")
        data = np.asarray(self.wave_resolver(path), dtype=np.float64)
        # Igor auto-names from the file header, which LabVIEW writes to match
        # the wave just Made (Xmove -> file header Xmove_temp). Mirror that.
        target = getattr(self, "_last_make", None)
        name = f"{target}_temp" if target else "loaded_temp"
        self.waves[name] = data

    @staticmethod
    def _args(argstr: str) -> list:
        if argstr.strip() == "":
            return []
        out, cur, depth, in_str = [], "", 0, False
        for ch in argstr:
            if ch == '"':
                in_str = not in_str
                continue
            if not in_str:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                elif ch == "," and depth == 0:
                    out.append(cur.strip())
                    cur = ""
                    continue
            cur += ch
        out.append(cur.strip())
        return out

    # ------------------------------------------------------- dispatch tables --
    # Data-driven command routing for execute(): a leading keyword picks a handler,
    # else the first matching grammar wins. Adding a command is a new table row, not
    # another branch in a chain. Handlers are the unbound functions above, called as
    # handler(self, ...); defined here (after the methods) so the names resolve.
    _KEYWORD_CMDS = (
        ("fprintf", _fprintf),
        ("Make", _make),
        ("LoadWave", _loadwave),
        ("display", lambda self, cmd: None),      # accepted, but the twin has no GUI
    )
    _PATTERN_CMDS = (
        (_CALL_RE, lambda self, m: self._call(m.group(1), m.group(2))),
        (_LITERAL_RE, _assign_literal),
        (_COPY_RE, _assign_copy),
    )
