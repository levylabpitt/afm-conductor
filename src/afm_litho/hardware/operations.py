"""Drive the operational controls on a real Igor: approach / withdraw / scan /
tune / force, plus the scan parameters and imaging mode.

It runs the twin-side emitter `twin.igor_commands` over the
`hardware.Igor` COM bridge, so the exact strings proven by the round-trip test
(tests/test_igor_commands) are what reach the instrument.

Nothing here imports pywin32; the Igor object does that lazily on connect(), so
importing this module off-instrument costs nothing.
"""

from __future__ import annotations

from ..twin import igor_commands as ig


def _run(igor, cmds) -> None:
    for c in ([cmds] if isinstance(cmds, str) else cmds):
        igor.execute(c)


# ------------------------------------------------------------- parameters --

def set_parameters(igor, *, mode=None, size_m=None, rate_hz=None, points=None,
                   lines=None, angle_deg=None, setpoint_v=None) -> int:
    """Write whatever parameters are supplied. Mode goes first because it
    re-routes the Z feedback, and the setpoint global depends on it."""
    cmds = []
    if mode is not None:
        cmds.append(ig.set_mode(mode))
    if size_m is not None:
        cmds.append(ig.set_scan_size(size_m))
    if rate_hz is not None:
        cmds.append(ig.set_scan_rate(rate_hz))
    if points is not None:
        cmds.append(ig.set_scan_points(points))
    if lines is not None:
        cmds.append(ig.set_scan_lines(lines))
    if angle_deg is not None:
        cmds.append(ig.set_scan_angle(angle_deg))
    if setpoint_v is not None:
        if mode is None:
            raise ValueError("setpoint needs mode (contact|ac) to pick its global")
        cmds.append(ig.set_setpoint(setpoint_v, mode))
    _run(igor, cmds)
    return len(cmds)


def read_parameters(igor) -> dict:
    """Read the scan parameters back off Igor -- proves the writes landed.
    Pure software globals, so this works on trial Igor with no controller."""
    names = ("ScanSize", "ScanRate", "ScanPoints", "ScanLines",
             "ScanAngle", "ImagingMode")
    return {n: igor.value(f'GV("{n}")') for n in names}


# ---------------------------------------------------------------- actions --

def approach(igor) -> None:           _run(igor, ig.approach())
def withdraw(igor) -> None:           _run(igor, ig.withdraw())
def frame_up(igor) -> None:           _run(igor, ig.frame_up())
def frame_down(igor) -> None:         _run(igor, ig.frame_down())
def stop(igor) -> None:               _run(igor, ig.stop())
def stop_and_withdraw(igor) -> None:  _run(igor, ig.stop_and_withdraw())
def tune(igor) -> None:               _run(igor, ig.tune())
def force_curve(igor) -> None:        _run(igor, ig.force_curve())


# --------------------------------------------------------------- preflight --

# The action procs the operational buttons fire. A wrong name from the control
# survey hides HERE, so exists() is the check that matters -- and it needs no
# instrument, only the Asylum software loaded.
_PROCS = ("DoScanFunc", "MainTabButtonFunc", "SimpleEngageMe",
          "DoForceFunc", "CantTuneFunc", "MainPopupFunc")


def verify_names(igor) -> dict:
    """Confirm every action proc we drive exists in the loaded Asylum software.

    Igor's exists() returns 6 for a user function. Runs on trial Igor with no
    controller. Returns {proc_name: bool}; a False means that survey name does
    not resolve and must be fixed before the instrument session.
    """
    return {p: int(igor.value(f'exists("{p}")')) == 6 for p in _PROCS}


# --------------------------------------------------------------- selftest --

def selftest(igor) -> float:
    """No-instrument check: set ScanSize and read it back. ScanSize is a pure
    software global, so this exercises the whole COM round-trip (Execute2 +
    fprintf readback) on trial Igor with the Asylum procedures loaded and no
    controller attached. Returns the value read back (expect 5e-06)."""
    igor.execute(ig.set_scan_size(5e-6))
    return igor.value('GV("ScanSize")')


def _main() -> None:
    """`python -m afm_litho.hardware.operations` -- connect to a running Igor
    and run the no-instrument selftest."""
    from .igor import Igor
    print("Connecting to Igor (IgorPro.Application)...")
    igor = Igor().connect()
    print("Connected.")
    got = selftest(igor)
    ok = abs(got - 5e-6) < 1e-12
    print(f"selftest: set ScanSize=5e-6, read back {got:g}  ->  {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit("selftest mismatch: check the Asylum procedures are loaded")


if __name__ == "__main__":
    _main()
