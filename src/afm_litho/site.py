"""Serving the panel at a friendly name while it stays on the loopback.

A real name instead of `http://127.0.0.1:7461` is optional and off by default.
Set AFM_LITHO_HOSTNAME to a name whose public DNS A record points at
`127.0.0.1` (the variables are listed in docs/guides/DEPLOY.md). Every machine
still runs its own copy, nothing is reachable from the network, and the panel
gets a stable URL. This is an old
trick -- `localtest.me` and `lvh.me` do the same thing -- and it costs one
DNS record.

A browser treats plain http as a secure context only for the literal name
`localhost`, so the name needs a certificate. Since it resolves to loopback, the usual
HTTP-01 challenge cannot reach it from outside and the certificate has to be
issued over DNS-01, which works because the domain is ours.

Two guards, because the failure modes are quiet ones.

`resolves_to_loopback()` is checked before the hostname is ever used. If the
record is missing, or someone repoints it at a real machine, the app falls
back to 127.0.0.1 rather than cheerfully sending the browser somewhere else.
A wrong answer here would mean the panel silently talking to a different
computer's instrument.

The certificate is only used when both files are actually present and
readable. A half-configured TLS setup drops back to http on loopback, which
still works, rather than refusing to start.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

# No default name: the panel stays on http://127.0.0.1 until AFM_LITHO_HOSTNAME
# names a host whose DNS record points at the loopback.
DEFAULT_HOSTNAME = ""


def hostname() -> str:
    """The friendly name, or "" to stay on 127.0.0.1."""
    return os.environ.get("AFM_LITHO_HOSTNAME", DEFAULT_HOSTNAME).strip()


def cert_dir() -> Path:
    """Where tools/make_local_cert.py puts the certificate. One definition."""
    base = os.environ.get("PROGRAMDATA") or os.environ.get("LOCALAPPDATA") or "."
    return Path(base) / "afm-litho" / "tls"


def tls_paths() -> tuple[Path | None, Path | None]:
    """The certificate and key to serve with, or (None, None).

    The environment wins, and if it says nothing we look where
    make_local_cert.py actually writes. That fallback is not a convenience:
    without it, TLS depended on two environment variables that nothing sets
    for you, so a machine with a perfectly good certificate on disk served
    plain HTTP -- and the panel then quietly came up on 127.0.0.1 with a port,
    which is not the address anybody was told to type.
    """
    cert = os.environ.get("AFM_LITHO_TLS_CERT", "").strip()
    key = os.environ.get("AFM_LITHO_TLS_KEY", "").strip()
    if cert and key:
        c, k = Path(cert), Path(key)
        return (c, k) if (c.is_file() and k.is_file()) else (None, None)

    host = hostname()
    if not host:
        return None, None
    d = cert_dir()
    c, k = d / f"{host}.crt", d / f"{host}.key"
    return (c, k) if (c.is_file() and k.is_file()) else (None, None)


def resolves_to_loopback(host: str, timeout: float = 2.0) -> bool:
    """True only if every address for `host` is a loopback address.

    Every, not any: a name with both 127.0.0.1 and a routable address would
    otherwise pass here and then resolve to the routable one in the browser.
    """
    if not host:
        return False
    try:
        socket.setdefaulttimeout(timeout)
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, OSError):
        return False
    finally:
        socket.setdefaulttimeout(None)
    if not infos:
        return False
    for info in infos:
        addr = info[4][0]
        if not (addr.startswith("127.") or addr in ("::1", "0:0:0:0:0:0:0:1")):
            return False
    return True


def choose_port(tls: bool, preferred: int = 7461) -> int:
    """Which port to listen on.

    A browser assumes 443 for https, so `https://<name>` means
    `:443` and nothing else. Serving there is what removes the port from the
    URL, and on Windows binding it needs no elevation -- unlike Unix, where
    everything below 1024 is privileged. So when the panel is going to be
    served under its own name over TLS, 443 is tried first and the address
    becomes something a person can actually say out loud.

    If 443 is taken -- IIS, Docker Desktop, a VPN client, another copy of
    this -- it falls back to the usual range rather than refusing to start.
    The URL then carries a port again, which is a cosmetic loss, not a
    functional one.
    """
    # 443 is opt-in (AFM_LITHO_PORT=443): it changes the origin to the bare
    # https://<name>, so the default stays the port already in use.
    if tls and os.environ.get("AFM_LITHO_PORT", "").strip() == "443":
        import socket as _s
        with _s.socket() as sock:
            try:
                sock.bind(("127.0.0.1", 443))
                return 443
            except OSError:
                pass
    for p in range(preferred, preferred + 40):
        import socket as _s
        with _s.socket() as sock:
            try:
                sock.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return preferred


def tls_possible() -> bool:
    """Whether the hostname and a certificate are both in place, without
    committing to a port yet -- the port choice depends on this answer."""
    host = hostname()
    cert, _ = tls_paths()
    return bool(host) and cert is not None and resolves_to_loopback(host)


def serve(port: int, site: dict | None = None) -> None:
    """Run the service. THE one place uvicorn is started.

    Both entry points call this, and that is the whole point of it being
    here. They did not: `python -m afm_litho` (and therefore run.ps1, the
    documented way to run the twin) called uvicorn itself with no TLS and
    never imported this module, while afm_litho.desktop went through
    choose_port and resolve_site. So the documented command could not serve
    the documented address, and the browser got nothing at all.

    Absolute import, not relative: PyInstaller runs desktop.py as __main__,
    so a relative import fails in the frozen build -- inside a daemon
    thread, where it kills the server and leaves the window waiting on a
    service that will never answer.
    """
    import uvicorn

    from afm_litho.service.app import app

    kw = {}
    if site and site.get("tls"):
        kw = {"ssl_certfile": site["certfile"], "ssl_keyfile": site["keyfile"]}
    # Loopback either way. The friendly name is a DNS alias FOR 127.0.0.1,
    # not a reason to listen on the network.
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", **kw)


# The page both entry points open at start: the cache-busted cockpit route
# (service/routers/meta.py), not /static/edit-cockpit.html, which can serve stale
# modules after an update.
COCKPIT_PATH = "/edit-cockpit"


def cockpit_url(url: str) -> str:
    """The edit cockpit's address under the service root `url`. `resolve_site`
    returns the root with a trailing slash; an explicit --host root has none."""
    return url.rstrip("/") + COCKPIT_PATH


def resolve_site(port: int) -> dict:
    """Decide what URL to open and whether to serve TLS.

    Returns everything the caller needs and, importantly, WHY -- so a panel
    that came up on a bare IP address can say which check failed instead of
    leaving someone guessing at their DNS.
    """
    host = hostname()
    cert, key = tls_paths()
    reasons = []

    # No hostname is the unconfigured default, not a failure: plain loopback
    # is the intended result and gets no reason line.
    if host and not resolves_to_loopback(host):
        reasons.append(
            f"{host} does not resolve to 127.0.0.1 (missing DNS record, or it "
            f"points somewhere else)")
    if host and cert is None:
        reasons.append(
            "no readable TLS certificate (AFM_LITHO_TLS_CERT / _TLS_KEY); "
            "a name other than localhost needs https")

    ok = host and not reasons
    # 443 is implied by https, so the origin string leaves it out.
    suffix = "" if port == 443 else f":{port}"
    return {
        "url": (f"https://{host}{suffix}/" if ok else f"http://127.0.0.1:{port}/"),
        "origin": (f"https://{host}{suffix}" if ok else f"http://127.0.0.1:{port}"),
        "hostname": host if ok else "127.0.0.1",
        "tls": bool(ok),
        "certfile": str(cert) if ok else None,
        "keyfile": str(key) if ok else None,
        "reasons": reasons,
    }
