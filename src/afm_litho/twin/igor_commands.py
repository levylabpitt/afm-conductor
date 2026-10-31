"""Igor command-string EMITTER: the twin-side map from an operation to the
`Execute2(...)` command string(s) that drive the real Asylum software.

This is the counterpart to `igor.py` (which RECEIVES these strings and replays
them against `SoftwareARC`). The transport is Igor Pro's COM/ActiveX automation
server: `IgorPro.Application.Execute2(cmd, ...)` runs an arbitrary Igor command
and returns error code + captured history/results. So the whole bridge is
"build the right command string, hand it to Execute2." This mirrors the proven
LabVIEW->Igor sequence captured in docs/reference/igor-commands-commented.txt.

Three command shapes, and only three:

  * parameter  -> PV("<Global>", value)        set a MasterVariablesWave global
  * action     -> <Proc>("<ControlName>")      call a control's action proc
  * readback   -> fprintf 0, "<fmt>", <expr>   value comes back in Execute2 history

The catalogue of names lives in docs/reference/controls/operational-controls.md.

Why call the proc directly instead of ARExecuteControl: Execute2 runs any
command, and the survey already tells us the proc a control fires, so we skip
the control-lookup layer (and its panel/tab/`_0`-suffix dependency). The one
subtlety is Igor's two proc calling conventions:

  * OLD-style procs take a control-name STRING -> callable in one command line.
    Confirmed old-style here: DoScanFunc (ScanMaster.ipf:2208 calls it exactly
    this way) and SimpleEngageMe (HardWare.ipf:1273, `func(ctrlName,[Mode])`).
  * NEW-style procs take a WMxxxAction STRUCT -> NOT buildable from a bare
    command line. For those (mode popup; and any tune/force proc that turns out
    struct-style) go through `execute_control2`, which the vendor wrapper
    `ARExecuteControl2` fills the struct for, OR set the underlying global.

MODE is the important non-obvious case: setting PV("ImagingMode",n) changes the
global but does NOT re-route the Z-feedback input / lockin / channels. That
reconfiguration lives in the popup proc, so mode MUST fire the proc, not just
write the global.
"""

from __future__ import annotations

# ImageModeList() on the instrument -> "Contact;AC Mode;..."; index is 0-based.
# The twin returns exactly "Contact;AC Mode;". "current" is deliberately absent:
# c-AFM/current feedback needs the ORCA path the bench does not have, so it is
# unsupported for bring-up (raise rather than emit a command that would fail).
_MODE_INDEX = {"contact": 0, "ac": 1}
_MODE_STR = {"contact": "Contact", "ac": "AC Mode"}

# Setpoint is a different global per feedback mode; FMapSetVarFunc routes the
# GUI control to whichever of these is live, so we pick it here explicitly.
_SETPOINT_GLOBAL = {"contact": "DeflectionSetpointVolts", "ac": "AmplitudeSetpointVolts"}


def _num(v: float) -> str:
    """Igor-friendly number literal (plain, no trailing noise)."""
    return f"{v:.9g}"


# --------------------------------------------------------------- primitives --

def set_var(global_name: str, value: float) -> str:
    """A MasterVariablesWave global write: PV("<name>", <value>)."""
    return f'PV("{global_name}",{_num(value)})'


def call_proc(proc: str, control_name: str) -> str:
    """An old-style control-proc call: <Proc>("<ControlName>")."""
    return f'{proc}("{control_name}")'


def execute_control2(control_name: str, proc: str, ctrl_type: int,
                     var_num: float = 0, var_str: str = "",
                     panel: str = "MasterPanel") -> str:
    """Robust fire-a-control that works for BOTH proc calling conventions.

    ARExecuteControl2 builds the WMxxxAction struct for new-style procs and
    calls old-style ones with the name; it does not need the control/panel to
    be on screen. ctrl_type: Button=1, CheckBox=2, PopupMenu=3, SetVariable=5,
    TabControl=8. VarStr carries a popup selection; VarNum a numeric state.
    """
    return (f'ARExecuteControl2("{control_name}","{proc}","{panel}","",'
            f'{ctrl_type},{_num(var_num)},"{var_str}")')


def read(expr: str, fmt: str = "%.12f") -> str:
    """A readback: fprintf 0 routes the value into Execute2's history buffer."""
    return f'fprintf 0, "{fmt}", {expr}'


def read_global(name: str, fmt: str = "%.12f") -> str:
    return read(f'GV("{name}")', fmt)


def read_hw(param: str, fmt: str = "%.12f") -> str:
    return read(f'td_ReadValue("{param}")', fmt)


# ---------------------------------------------------------- parameter setters --

def set_scan_size(meters: float) -> str:      return set_var("ScanSize", meters)
def set_scan_rate(hz: float) -> str:          return set_var("ScanRate", hz)   # "Linerate" in the GUI
def set_scan_angle(degrees: float) -> str:    return set_var("ScanAngle", degrees)
def set_scan_points(n: int) -> str:           return set_var("ScanPoints", int(n))
def set_scan_lines(n: int) -> str:            return set_var("ScanLines", int(n))


def set_setpoint(volts: float, mode: str) -> str:
    """Setpoint global depends on the feedback mode ('contact' or 'ac')."""
    mode = mode.lower()
    if mode not in _SETPOINT_GLOBAL:
        raise ValueError(f"setpoint: unsupported mode {mode!r} (contact|ac)")
    return set_var(_SETPOINT_GLOBAL[mode], volts)


def set_mode(mode: str) -> str:
    """Switch imaging mode. MUST fire the popup proc (re-routes Z feedback /
    lockin / channels), so this is an execute_control2, not a bare global."""
    mode = mode.lower()
    if mode not in _MODE_INDEX:
        raise ValueError(f"mode: {mode!r} unsupported for bring-up "
                         f"(contact|ac; 'current' needs the ORCA path)")
    return execute_control2("ImagingModePopup", "MainPopupFunc", 3,
                            var_num=0, var_str=_MODE_STR[mode])


# ------------------------------------------------------------------- actions --
# CONFIRMED old-style (direct call) --------------------------------------------

def approach() -> str:      return call_proc("SimpleEngageMe", "SimpleEngageButton_0")
def withdraw() -> str:      return call_proc("DoScanFunc", "Withdraw")
def frame_up() -> str:      return call_proc("DoScanFunc", "UpScan_0")
def frame_down() -> str:    return call_proc("DoScanFunc", "DownScan_0")
def stop() -> str:          return call_proc("DoScanFunc", "StopScan_0")


def stop_and_withdraw() -> list[str]:
    """The GUI 'Stop + Withdraw': halt the scan, then retract."""
    return [stop(), withdraw()]


# Proc calling-convention NOT yet confirmed -> go through the robust wrapper.
# If these turn out old-style, swap to call_proc for one fewer indirection.

def tune() -> str:
    return execute_control2("DoTuneAuto", "CantTuneFunc", 1)


def force_curve() -> str:
    return execute_control2("SingleForce", "DoForceFunc", 1)


# ---------------------------------------------------------------- composites --

def configure_scan(*, size_m: float, rate_hz: float, points: int, lines: int,
                   angle_deg: float = 0.0, setpoint_v: float | None = None,
                   mode: str = "contact") -> list[str]:
    """The full parameter block a scan needs, in order. Set mode first so the
    setpoint lands on the right global."""
    cmds = [set_mode(mode),
            set_scan_size(size_m), set_scan_rate(rate_hz),
            set_scan_points(points), set_scan_lines(lines),
            set_scan_angle(angle_deg)]
    if setpoint_v is not None:
        cmds.append(set_setpoint(setpoint_v, mode))
    return cmds


# Map twin API endpoints -> the command(s) they emit. `params` endpoints take
# the request body; action endpoints take none. Values are callables so the
# service layer can do: `ENDPOINTS["/api/scan/start"](direction="up")`.
ENDPOINTS = {
    "/api/approach":     lambda **_: [approach()],
    "/api/withdraw":     lambda **_: [withdraw()],
    "/api/scan/start":   lambda direction="up", **_: [frame_up() if direction == "up" else frame_down()],
    "/api/scan/abort":   lambda **_: [stop()],
    "/api/tune/sweep":   lambda **_: [tune()],
    "/api/force_curve":  lambda **_: [force_curve()],
}
