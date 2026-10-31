"""Bring up the HardwareARC bridge in Igor over COM -- no manual Igor steps.

The bridge listener (afm_litho_run.ipf) calls GV/td_ (Asylum) and AFM_SendStr
(afm_sockit) / AFM_MacroStartup (afm_igor), so it can't auto-compile at Igor
startup. This does what "Programming -> Load User Func" does, over COM: it
INSERTINCLUDEs afm_litho_run (which now `#include`s afm_sockit + afm_igor, so all
three compile together) and COMPILEPROCEDURES, WITH the software loaded, then --
unless --no-connect -- dials the running afm-litho service so Igor starts
answering the bridge protocol.

SEQUENCE for a hardware session:
  1. Start the service in hardware mode FIRST (it listens for Igor to dial in):
        set AFM_LITHO_BACKEND=hardware   &&   .venv\\Scripts\\python.exe -m afm_litho
     (optional: set AFM_LITHO_BRIDGE_PORT, default 9200)
  2. With the Asylum MFP3D software loaded in Igor, run THIS:
        .venv\\Scripts\\python.exe scripts\\start_bridge.py
     It compiles the bridge and calls AFM_LithoRunStart("127.0.0.1", 9200); the
     service accepts, reads the STARTUP calibration, and the bridge is live.

SAFETY: the safety kernel is not built yet (docs/SAFETY_KERNEL_PLAN.md). This
script only opens the link + reads calibration (no motion). Keep the tip
retracted / outputs disconnected for a first round-trip.

Prereqs: afm_litho_run.ipf + afm_sockit.ipf + afm_igor.ipf in Igor's User
Procedures (install_afm_ipf.py --folder "<...v19...\\Igor Pro Folder>"), the
SOCKIT XOP loaded, and the [hardware] extra (pywin32) installed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afm_litho.hardware.igor import Igor

READY_FUNC = "AFM_LithoRunStart"          # exists once afm_litho_run compiled


def _exists(igor, name: str) -> int:
    """Igor Exists(name): 0 = not in use; >=2 for a function."""
    try:
        return int(igor.value(f'Exists("{name}")'))
    except Exception:
        return 0


def load_bridge(igor, *, timeout: float = 20.0) -> bool:
    """INSERTINCLUDE afm_litho_run + recompile (pulls afm_sockit + afm_igor via its
    own #includes), then wait for AFM_LithoRunStart to appear."""
    igor.execute(r'Execute/P "INSERTINCLUDE \"afm_litho_run\""')
    igor.execute(r'Execute/P "COMPILEPROCEDURES "')
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _exists(igor, READY_FUNC) >= 2:
            return True
        time.sleep(0.3)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="host the running afm-litho service listens on")
    ap.add_argument("--port", type=int, default=9200,
                    help="bridge port (AFM_LITHO_BRIDGE_PORT; default 9200)")
    ap.add_argument("--no-connect", action="store_true",
                    help="compile the bridge only; don't dial the service")
    args = ap.parse_args()

    # Attach to the operator's already-open Igor; do NOT launch a blank one. The
    # bridge needs Igor with the Asylum software loaded, so launching a fresh Igor
    # only to bail on the missing software would leave that Igor orphaned.
    try:
        igor = Igor().connect(launch=False)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("Connected to Igor.")

    # GV is the load-order canary: if it's missing, the MFP3D software isn't
    # loaded, so afm_litho_run (and its afm_sockit/afm_igor includes) won't compile.
    if _exists(igor, "GV") < 2:
        print("The MFP3D (Asylum) software isn't loaded -- GV is undefined. Load "
              "the RealTime software, then re-run.", file=sys.stderr)
        return 1

    print("Compiling the bridge (INSERTINCLUDE afm_litho_run + afm_sockit + afm_igor)...")
    if not load_bridge(igor):
        print(f"{READY_FUNC} did not appear after compiling. Open Igor's procedure "
              "window and look for a compile error (a missing SOCKIT XOP, or a "
              "procedure file not in User Procedures).", file=sys.stderr)
        return 1
    print("Bridge compiled.")

    if args.no_connect:
        print("Compiled only (--no-connect). Call AFM_LithoRunStart(host, port) "
              "yourself, or re-run without --no-connect once the service listens.")
        return 0

    print(f'Dialing the service at {args.host}:{args.port} '
          '(start it FIRST with AFM_LITHO_BACKEND=hardware) ...')
    sock = igor.value(f'AFM_LithoRunStart("{args.host}", {args.port})')
    if sock < 0:
        print("AFM_LithoRunStart returned a negative socket id -- is the service "
              "running in hardware mode and listening on that port?", file=sys.stderr)
        return 1
    print("Bridge live: Igor is connected and answering the protocol. The service "
          "has read the STARTUP calibration. Call AFM_LithoRunEnd() (zeros bias, "
          "closes the link) when done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
