"""Staying current without anybody thinking about it.

The installed application checks a small JSON manifest, compares versions,
and offers the new one. It does not install anything on its own.

That last point is not timidity, it is the instrument. This program drives a
piezo against a sample with a biased tip. Software that decides on its own to
restart itself is software that can do that in the middle of somebody's
write, and no amount of update hygiene is worth a ruined device. So: check
quietly in the background, say so in the panel, and install when a human
says now. The panel refuses to offer it at all while a write is running.

The manifest is deliberately dumb -- a URL returning

    {"version": "0.2.0",
     "url": "https://.../afm-litho-setup-0.2.0.exe",
     "sha256": "…",
     "notes": "what changed"}

-- because the hosting question ("where do updates for a private repo come
from?") should be answerable by changing a URL rather than by rewriting this.
Point AFM_LITHO_UPDATE_URL at whatever ends up serving it.

The download is verified against the manifest's SHA-256 before anything is
executed. A manifest fetched over HTTPS from a host we trust is one thing; an
installer binary is another, and it gets checked.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# No default feed: update checks are off until AFM_LITHO_UPDATE_URL is set.
DEFAULT_FEED = ""
USER_AGENT = "afm-litho-updater"


def feed_url() -> str:
    return os.environ.get("AFM_LITHO_UPDATE_URL", DEFAULT_FEED).strip()


def _auth_headers() -> dict:
    """Credentials for a feed host that sits behind a login.

    The updater runs unattended on a lab PC at start-up. It cannot complete
    an interactive sign-in, so it carries a credential instead:

      AFM_LITHO_UPDATE_TOKEN  -> Authorization: Bearer <token>
      AFM_LITHO_UPDATE_BASIC  -> Authorization: Basic <base64 user:pass>

    If the host only accepts a browser session cookie, neither of these
    will work and the honest options are to expose the two update files on a
    path that does not require the login, or to add a token endpoint. The
    manifest and the installer are not the secret; the source is.
    """
    token = os.environ.get("AFM_LITHO_UPDATE_TOKEN", "").strip()
    if token:
        return {"Authorization": f"Bearer {token}"}
    basic = os.environ.get("AFM_LITHO_UPDATE_BASIC", "").strip()
    if basic:
        import base64
        enc = base64.b64encode(basic.encode("utf-8")).decode("ascii")
        return {"Authorization": f"Basic {enc}"}
    return {}


def _request(url: str) -> urllib.request.Request:
    headers = {"User-Agent": USER_AGENT}
    headers.update(_auth_headers())
    return urllib.request.Request(url, headers=headers)


def current_version() -> str:
    try:
        import importlib.metadata as md
        return md.version("afm-litho")
    except Exception:  # noqa: BLE001 - unknown version falls back to 0.0.0
        return "0.0.0"


def _parts(v: str) -> tuple:
    """Compare 1.10.0 above 1.9.0, which a string compare gets wrong."""
    out = []
    for chunk in str(v).split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out or [0])


def is_newer(candidate: str, installed: str) -> bool:
    return _parts(candidate) > _parts(installed)


def fetch_manifest(url: str | None = None, timeout: float = 8.0) -> dict:
    with urllib.request.urlopen(_request(url or feed_url()), timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def check(url: str | None = None) -> dict:
    """What is available, and whether it beats what is installed.

    Never raises. An update check that breaks the program it is checking is
    worse than no update check.
    """
    installed = current_version()
    if not (url or feed_url()):
        return {"installed": installed, "available": None, "update": False,
                "skipped": "no update feed configured (AFM_LITHO_UPDATE_URL)"}
    try:
        m = fetch_manifest(url)
    except Exception as exc:  # noqa: BLE001 - a failed check is reported, never fatal
        return {"installed": installed, "available": None,
                "update": False, "error": str(exc)}
    latest = str(m.get("version") or "")
    return {
        "installed": installed,
        "available": latest or None,
        "update": bool(latest) and is_newer(latest, installed),
        "url": m.get("url") or "",
        "sha256": (m.get("sha256") or "").lower(),
        "notes": m.get("notes") or "",
    }


def download(url: str, sha256: str, dest: Path | None = None,
             timeout: float = 120.0) -> Path:
    """Fetch the installer and verify it. Raises if the hash does not match."""
    if not sha256:
        raise ValueError("refusing to download an installer with no checksum")
    dest = Path(dest or (Path(tempfile.gettempdir()) / Path(url).name))
    h = hashlib.sha256()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(_request(url), timeout=timeout) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            h.update(chunk)
            f.write(chunk)
    got = h.hexdigest()
    if got != sha256.lower():
        tmp.unlink(missing_ok=True)
        raise ValueError(f"checksum mismatch: expected {sha256}, got {got}")
    tmp.replace(dest)                       # only a verified file gets the name
    return dest


def frozen() -> bool:
    """True when running from the packaged build rather than a source tree."""
    return bool(getattr(sys, "frozen", False))


def apply(installer: Path) -> None:
    """Run the installer and quit, so it can replace files we are holding.

    Per-user, silent, and it relaunches the app itself: /SILENT rather than
    /VERYSILENT so there is a progress window, because a program that
    vanishes for twenty seconds looks like a program that crashed.
    """
    subprocess.Popen([str(installer), "/SILENT", "/NOCANCEL",
                      "/RESTARTAPPLICATIONS"], close_fds=True)
    os._exit(0)


def check_and_notify() -> dict:
    """Background check. Records the result for the panel to pick up.

    Only meaningful in the packaged build: a source checkout updates with
    git, and telling a developer their working tree is out of date relative
    to a release is noise.
    """
    if not frozen():
        return {"update": False, "skipped": "running from source"}
    result = check()
    try:
        state_dir = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
        (state_dir / "afm-litho").mkdir(parents=True, exist_ok=True)
        (state_dir / "afm-litho" / "update.json").write_text(
            json.dumps(result), "utf-8")
    except OSError:
        pass
    return result
