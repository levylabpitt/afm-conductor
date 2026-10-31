"""Install the afm-litho Igor procedure files into the right Igor folders.

Two Igor folders, two behaviours:
  * "Igor procedures"  -- AUTO-compiles at Igor startup, in a BLANK experiment,
    BEFORE the Asylum software loads. Only put files here that have NO Asylum
    (GV/PV/ir_/td_ procedure) dependency, or they fail to compile at startup and
    disable ALL user functions.
  * "User Procedures"  -- NOT auto-compiled; compiles only when an experiment
    #includes it, by which time the Asylum procedures (GV, etc.) are loaded.

So:
  afm_litho.ipf      (generic COM API, no deps)   -> Igor procedures  (auto-loads)
  afm_sockit.ipf     (SOCKIT streamer; calls GV)  -> User Procedures  (#include it)
  afm_igor.ipf       (telemetry: AFM_MacroStartup, -> User Procedures  (#include it)
                      AFM_GetCal; the bridge NEEDS it)
  afm_litho_run.ipf  (bridge; #includes the two    -> User Procedures  (#include it)
                      above -- td_/ir_/GV/SOCKIT)

afm_litho_run.ipf `#include`s afm_sockit AND afm_igor, so it will not compile
unless BOTH are also on disk in User Procedures -- hence all three are installed
by default (afm_igor is NOT optional; it defines the STARTUP calibration the
bridge sends). After running, in your ASYLUM experiment add to a procedure window
and compile:
    #include "afm_litho_run"       // bridge -- pulls in afm_sockit + afm_igor
    // (or, for the streaming path only:  #include "afm_sockit")

IMPORTANT: run this with Igor CLOSED. Igor locks procedure files while running.
This script does NOT use COM (so it won't relaunch Igor); it finds the Igor Pro
Folder by searching C:\\AsylumResearch, or pass --folder.

    # close Igor, then:
    .venv\\Scripts\\python.exe scripts\\install_afm_ipf.py
    # reopen Igor, #include the User-Procedures files in your experiment
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

IGOR_DIR = Path(__file__).resolve().parent.parent / "igor"
SEARCH_ROOTS = [Path("C:/AsylumResearch")]

# generic -> auto-compile folder; Asylum-dependent -> User Procedures (#include).
# afm_litho_run (the HardwareARC bridge) #includes BOTH afm_sockit (AFM_SendStr,
# AFM_SockConnect/Close) and afm_igor (AFM_MacroStartup, the STARTUP calibration),
# so all three must be present or the bridge fails to compile. afm_igor is a HARD
# dependency now (not optional telemetry). Deps listed before the file that needs
# them so the printed #include hint reads top-down.
AUTO_IPF = ["afm_litho.ipf"]
INCLUDE_IPF = ["afm_sockit.ipf", "afm_igor.ipf", "afm_litho_run.ipf"]


def find_igor_folders_fs() -> list[Path]:
    """Every RealTime 'Igor Pro Folder' (has 'Igor Extensions' + 'More Extensions')
    under the search roots, by filesystem search -- no COM, so it works with Igor
    closed. The AFM PC keeps several Asylum versions side by side (v19, v21, ...), each
    with a RealTime and an Offline tree; the instrument is driven from RealTime, so we
    install into every version's RealTime folder and skip Offline. That covers whichever
    version the operator launches without needing --folder. Sorted for a stable order.
    (Pass --folder explicitly to target an Offline tree or any other location.)"""
    found: list[Path] = []
    for root in SEARCH_ROOTS:
        if root.is_dir():
            for ext in root.rglob("Igor Extensions"):
                folder = ext.parent
                if (folder / "More Extensions").is_dir() and folder not in found:
                    # RealTime only: the parent of "Igor Pro Folder" is the tree name.
                    if folder.parent.name.lower() == "realtime":
                        found.append(folder)
    return sorted(found, key=lambda p: str(p).lower())


def _pick_dir(folder: Path, *names: str) -> Path:
    """Return the first existing case-variant of a subfolder, else the first."""
    return next((d for d in (folder / n for n in names) if d.is_dir()),
               folder / names[0])


SHA_LINE = b"StrConstant kIpfSha"


def ipf_sha(data: bytes) -> str:
    """kIpfSha (wire v3 contract 11.7): the first 16 hex of sha256 over the file with the
    whole line holding the constant blanked. The same as wire_v3.expected_ipf_sha."""
    lines = [b"" if SHA_LINE in ln else ln for ln in data.split(b"\n")]
    return hashlib.sha256(b"\n".join(lines)).hexdigest()[:16]


def stamp_ipf_sha(data: bytes) -> bytes:
    """The file with its kIpfSha line holding its own hash; the hash skips that line, so
    stamping does not change it. A file without the line comes back unchanged."""
    sha = ipf_sha(data).encode()
    lines = data.split(b"\n")
    for i, ln in enumerate(lines):
        if SHA_LINE in ln:
            cr = b"\r" if ln.endswith(b"\r") else b""
            lines[i] = b'StrConstant kIpfSha = "' + sha + b'"' + cr
    return b"\n".join(lines)


def _copy(ipf: Path, dest_dir: Path) -> bool:
    """Copy one file; return True on success. A lock (WinError 32 -- that version's
    Igor is open) or other OSError is reported and returns False, so the caller can
    keep installing into the OTHER versions instead of aborting the whole run.
    The copy carries its kIpfSha stamp, which the HELLO reply reports."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / ipf.name
    try:
        shutil.copy2(ipf, dest)
        dest.write_bytes(stamp_ipf_sha(ipf.read_bytes()))
        print(f"  installed {ipf.name} -> {dest}")
        return True
    except OSError as exc:
        print(f"  FAILED {ipf.name}: {exc}")
        if getattr(exc, "winerror", None) == 32:
            print("  -> the file is locked. CLOSE this version's Igor Pro, then re-run.")
        return False


def install_into(folder: Path, include_names: list[str]) -> bool:
    """Install every procedure into one Igor Pro Folder. Returns True if all copies
    (and stale removals) succeeded; False if anything was locked/failed."""
    print(f"Igor Pro Folder: {folder}")
    ok = True
    auto_dir = _pick_dir(folder, "Igor procedures", "Igor Procedures")
    user_dir = _pick_dir(folder, "User Procedures", "User procedures")

    # generic files -> auto-compile folder
    for name in AUTO_IPF:
        src = IGOR_DIR / name
        if src.is_file():
            ok = _copy(src, auto_dir) and ok

    # Asylum-dependent files -> User Procedures, and REMOVE any stale copy that a
    # previous install left in the auto-compile folder (that stale copy is what
    # breaks startup: it calls GV before Asylum is loaded).
    for name in include_names:
        src = IGOR_DIR / name
        if not src.is_file():
            continue
        stale = auto_dir / name
        if stale.is_file():
            try:
                stale.unlink()
                print(f"  removed stale {name} from {auto_dir} (it auto-compiled too early)")
            except OSError as exc:
                print(f"  could not remove stale {stale}: {exc}")
                if getattr(exc, "winerror", None) == 32:
                    print("  -> CLOSE this version's Igor Pro, then re-run.")
                ok = False
                continue
        ok = _copy(src, user_dir) and ok
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Install afm-litho Igor procedures.")
    ap.add_argument("--folder", help='one "Igor Pro Folder" (else install into ALL '
                    "versions found under C:\\AsylumResearch)")
    ap.add_argument("--telemetry", action="store_true",
                    help="deprecated no-op: afm_igor.ipf is now always installed "
                         "(the bridge #includes it)")
    args = ap.parse_args()

    # afm_igor.ipf is always installed now (it is a hard dependency of the bridge,
    # which #includes it). --telemetry is kept only so old invocations don't error.
    include_names = list(INCLUDE_IPF)

    if args.folder:
        folders = [Path(args.folder)]
        if not folders[0].is_dir():
            print(f"Not a folder: {folders[0]}")
            raise SystemExit(1)
    else:
        folders = find_igor_folders_fs()
        if not folders:
            print("Could not find any Igor Pro Folder. Pass one explicitly:")
            print('  --folder "C:\\AsylumResearch\\v19\\RealTime\\Igor Pro Folder"')
            raise SystemExit(1)
        print(f"Installing into {len(folders)} Igor version(s): "
              + ", ".join(f.parent.parent.name for f in folders))

    all_ok = True
    for folder in folders:
        all_ok = install_into(folder, include_names) and all_ok
        print()

    if all_ok:
        print("Done -- all versions updated.")
    else:
        print("Done WITH FAILURES -- some files were locked or unwritable. Close the")
        print("open Igor version(s) noted above and re-run to finish those.")
    # afm_litho_run #includes afm_sockit + afm_igor, so ONE include pulls the whole
    # bridge. (Igor dedups, so listing all three would also work -- but keep the
    # hint aligned with the docstring, start_bridge.py and the runbook.)
    print("Now, in your ASYLUM experiment, add to a procedure window and compile:")
    print('    #include "afm_litho_run"    // the bridge -- pulls in afm_sockit + afm_igor')
    print('    // (streaming path only:  #include "afm_sockit")')
    print("(afm_litho auto-compiles at startup; the User-Procedures files load via")
    print(" that #include, when GV and the rest of the Asylum software are present.)")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
