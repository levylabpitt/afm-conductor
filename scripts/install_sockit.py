"""Install the prebuilt 32-bit SOCKIT XOP (from a user-provided SOCKIT Igor-5
zip) for Igor 6.

SOCKIT.xop here is 32-bit and built against Igor 5.00, which Igor 6.38 satisfies
(XOPs are backward compatible -- an Igor-5 XOP declares a MINIMUM of 5.00 and
loads in 6). It provides TCP sockets, so Igor can PUSH data to Python -- the
higher-bandwidth alternative to chunked COM reads, and it enables the push model.

SOCKIT is third-party: its source carries no licence and its zip bundles
pthreads-win32 (LGPL), so this repository never ships the binaries. This script
downloads them from WaveMetrics or takes a zip you already have.

Files placed (into the app-folder Igor Extensions that THIS Igor scans):
  SOCKIT/win/SOCKIT.xop      -> <Igor Pro Folder>\\Igor Extensions\\SOCKIT.xop
  SOCKIT/win/pthreadVC2.dll  -> Igor Extensions AND next to Igor.exe (DLL search)

Usage (run on the Igor machine, with Igor open, from the repo root). All four
commands are typed at a PowerShell prompt:

  # Download the zip from WaveMetrics, then install -- the one-step path:
  .venv\\Scripts\\python.exe scripts\\install_sockit.py --download

  # Install from a zip you already have on disk instead of downloading:
  .venv\\Scripts\\python.exe scripts\\install_sockit.py --zip "C:\\path\\to\\SOCKITIGOR.5.00.x1.xdev.zip"

  # Same as --download but from a different URL (if the hosted path moves):
  .venv\\Scripts\\python.exe scripts\\install_sockit.py --url "https://.../SOCKIT....zip"

  # Install with Igor CLOSED (filesystem folder search, no COM) -- what the
  # install-config-hardware.ps1 uses so the whole install runs Igor-closed:
  .venv\\Scripts\\python.exe scripts\\install_sockit.py --download --no-igor

  # After installing, RESTART Igor, then verify the XOP actually loaded:
  .venv\\Scripts\\python.exe scripts\\install_sockit.py --check

--zip and --download are mutually exclusive: pass one or the other, not both.
--download pulls the public WaveMetrics-hosted dev zip. The redistribution note
above is about re-hosting or shipping the binaries in the app; fetching them for
a local install is no different from a user downloading the zip by hand.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from afm_litho.hardware.igor import Igor, IgorError
from afm_litho.hardware.igor_install import find_igor_folder, igor_exe_dir

# The public dev zip on wavemetrics.com. --download uses this; --url overrides.
SOCKIT_URL = ("https://www.wavemetrics.com/sites/default/files/"
              "documents-imported/SOCKIT-IGOR.5.00.x-1.x-dev.zip")

XOP_MEMBER = "SOCKIT/win/SOCKIT.xop"
DLL_MEMBER = "SOCKIT/win/pthreadVC2.dll"
# Any nonzero exists() means it loaded; these are SOCKIT operations.
SOCKIT_SYMBOLS = ["SOCKITopenConnection", "SOCKITsendMsg", "SOCKITsendnrecv",
                  "SOCKITcloseConnection", "SOCKITlist", "SOCKITwaveToString"]


class _NoIgor:
    """Stand-in for a live Igor used by --no-igor, so the file copy can run with
    Igor CLOSED. find_igor_folder() first asks Igor over COM (via .string) and
    then falls back to a filesystem search; making .string raise IgorError sends
    it straight to that filesystem fallback -- no COM, no running Igor needed."""

    def string(self, *_args, **_kwargs):
        raise IgorError(0, "no live Igor (--no-igor)", "")


def check(igor: Igor) -> None:
    found = {}
    for name in SOCKIT_SYMBOLS:
        try:
            found[name] = int(igor.value(f'exists("{name}")'))
        except IgorError:
            found[name] = -1
    loaded = any(v > 0 for v in found.values())
    print("SOCKIT XOP:", "loaded" if loaded else "NOT loaded")
    print("  exists() map:", ", ".join(f"{k}={v}" for k, v in found.items()))
    if not loaded:
        print("  All zero -> not loaded. If you just installed, fully restart Igor.")
        print("  If it still won't load, pthreadVC2.dll isn't on the DLL search")
        print("  path -- confirm it sits next to Igor.exe.")


def download_zip(url: str) -> Path:
    """Fetch the SOCKIT zip to a temp file and return its path. The caller owns
    the file and should delete it when done. A User-Agent is set because the
    default urllib agent gets a 403 from some CDN configs."""
    fd, tmp = tempfile.mkstemp(suffix=".zip", prefix="sockit-")
    dest = Path(tmp)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(fd, "wb") as out:
            shutil.copyfileobj(resp, out)
    except OSError:
        dest.unlink(missing_ok=True)
        raise
    print(f"  saved {dest.stat().st_size:,} bytes -> {dest}\n")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="Install the prebuilt SOCKIT XOP for Igor 6.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--zip", help="path to a local SOCKIT Igor-5 zip")
    src.add_argument("--download", action="store_true",
                     help="fetch the zip from WaveMetrics before installing")
    ap.add_argument("--url", default=SOCKIT_URL,
                    help="override the download URL (implies --download)")
    ap.add_argument("--check", action="store_true",
                    help="only check whether the XOP loaded (after a restart)")
    ap.add_argument("--no-igor", action="store_true",
                    help="do not connect to Igor (COM); find the Igor Pro Folder "
                         "by filesystem search so the install runs with Igor CLOSED")
    ap.add_argument("--folder",
                    help='the "Igor Pro Folder" (skips the search; useful when '
                         "Asylum is not under C:\\AsylumResearch)")
    args = ap.parse_args()

    if args.no_igor:
        # Filesystem mode: no COM, so this works with Igor CLOSED -- required
        # because the .xop/.dll copy fails while Igor holds the files open.
        if args.check:
            print("--check queries the running Igor for the XOP, so it needs "
                  "Igor OPEN. Re-run --check without --no-igor once Igor is up.")
            raise SystemExit(1)
        igor: object = _NoIgor()
        print("Igor: filesystem mode (no COM); Igor may stay CLOSED.\n")
    else:
        print("Connecting to Igor (IgorPro.Application)...")
        igor = Igor().connect()
        print("Connected.\n")

    if args.check:
        check(igor)
        return

    want_download = args.download or args.url != SOCKIT_URL
    if not args.zip and not want_download:
        print("Pass --zip <path to the SOCKIT zip>, or --download to fetch it.")
        raise SystemExit(1)

    downloaded: Path | None = None
    if want_download:
        downloaded = download_zip(args.url)
        zip_path = downloaded
    else:
        zip_path = Path(args.zip)
    try:
        install(igor, zip_path, folder_override=args.folder)
    finally:
        if downloaded is not None:
            downloaded.unlink(missing_ok=True)


def install(igor, zip_path: Path, folder_override: str | None = None) -> None:
    zf = zipfile.ZipFile(zip_path)
    names = set(zf.namelist())
    for member in (XOP_MEMBER, DLL_MEMBER):
        if member not in names:
            print(f"{member} not found in the zip -- is this the SOCKIT Igor-5 zip?")
            raise SystemExit(1)

    folder = Path(folder_override) if folder_override else find_igor_folder(igor)
    if not folder or not folder.is_dir():
        print("Could not locate the Igor Pro Folder (the one with 'Igor Extensions').")
        raise SystemExit(1)
    ext = folder / "Igor Extensions"
    exe_dir = igor_exe_dir(folder)
    ext.mkdir(parents=True, exist_ok=True)
    print(f"Igor Pro Folder: {folder}")
    print(f"Igor.exe dir:    {exe_dir}\n")

    def extract(member: str, dest: Path) -> None:
        with zf.open(member) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
        print(f"  {Path(member).name:20} -> {dest}")

    extract(XOP_MEMBER, ext / "SOCKIT.xop")
    extract(DLL_MEMBER, ext / "pthreadVC2.dll")          # next to the XOP
    if exe_dir != ext:
        extract(DLL_MEMBER, exe_dir / "pthreadVC2.dll")  # and on the loader path

    print("\nInstalled. Now RESTART Igor, then verify:")
    print("  .venv\\Scripts\\python.exe scripts\\install_sockit.py --check")


if __name__ == "__main__":
    main()
