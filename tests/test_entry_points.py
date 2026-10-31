"""The two ways to start the twin must be the same server.

A user went to https://panel.example.org and got nothing. The cause was not
his pull: `python -m afm_litho` -- and so run.ps1, the documented way to run
the twin -- called uvicorn itself with no certificate and never imported
site.py, while afm_litho.desktop went through choose_port and resolve_site.
The documented command could not serve the documented address, and nothing
said so, because each entry point worked perfectly on its own terms.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "afm_litho"
LAUNCHER = (SRC / "keeper" / "cli.py").read_text(encoding="utf-8")
SERVICE = (SRC / "service_main.py").read_text(encoding="utf-8")
_DESKTOP_FILE = SRC / "desktop.py"      # not part of the public tree
DESKTOP = _DESKTOP_FILE.read_text(encoding="utf-8") if _DESKTOP_FILE.exists() else ""


def test_only_one_place_starts_uvicorn():
    starters = sorted(p.relative_to(SRC).as_posix()
                      for p in SRC.rglob("*.py")
                      if "uvicorn.run(" in p.read_text(encoding="utf-8"))
    assert starters == ["site.py"], (
        f"uvicorn is started in {starters}; every extra one is a second "
        "server that can drift away from the first")


@pytest.mark.parametrize("src,name", [
    (SERVICE, "service_main.py"),
    pytest.param(DESKTOP, "desktop.py", marks=pytest.mark.skipif(
        not _DESKTOP_FILE.exists(), reason="desktop.py is not in this tree")),
])
def test_both_entry_points_go_through_site(src, name):
    assert re.search(r"from afm_litho\.site import [^\n]*\bserve\b", src), \
        f"{name} does not use site.serve"


def test_the_launcher_asks_site_which_port_and_whether_to_use_tls():
    for fn in ("choose_port", "resolve_site", "tls_possible"):
        assert fn in LAUNCHER, f"keeper/cli.py no longer consults site.{fn}"


def test_an_explicit_port_that_is_taken_is_refused_not_slid_past():
    """Sliding onto the next free port is how five stale servers ended up on
    443 and 7461-7464, and why restarting and re-checking kept disagreeing
    with the code."""
    assert "is already in use" in LAUNCHER


# ------------------------------------------------------- the certificate --

def test_the_certificate_is_found_where_it_is_actually_written(tmp_path,
                                                               monkeypatch):
    """TLS used to require two environment variables that nothing sets for
    you. A machine with a perfectly good certificate on disk served plain
    HTTP, and the panel came up on 127.0.0.1 with a port -- not the address
    anyone had been told to type."""
    monkeypatch.delenv("AFM_LITHO_TLS_CERT", raising=False)
    monkeypatch.delenv("AFM_LITHO_TLS_KEY", raising=False)
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path))
    monkeypatch.setenv("AFM_LITHO_HOSTNAME", "panel.example.org")
    from afm_litho import site

    host = site.hostname()
    d = tmp_path / "afm-litho" / "tls"
    d.mkdir(parents=True)
    assert site.tls_paths() == (None, None)       # nothing there yet
    (d / f"{host}.crt").write_text("x")
    (d / f"{host}.key").write_text("x")
    cert, key = site.tls_paths()
    assert cert == d / f"{host}.crt" and key == d / f"{host}.key"


def test_the_environment_still_wins_over_the_default_location(tmp_path,
                                                              monkeypatch):
    from afm_litho import site

    monkeypatch.setenv("PROGRAMDATA", str(tmp_path))
    d = tmp_path / "afm-litho" / "tls"
    d.mkdir(parents=True)
    host = site.hostname()
    (d / f"{host}.crt").write_text("x")
    (d / f"{host}.key").write_text("x")

    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "c.pem").write_text("x")
    (mine / "k.pem").write_text("x")
    monkeypatch.setenv("AFM_LITHO_TLS_CERT", str(mine / "c.pem"))
    monkeypatch.setenv("AFM_LITHO_TLS_KEY", str(mine / "k.pem"))
    assert site.tls_paths() == (mine / "c.pem", mine / "k.pem")


def test_a_named_certificate_that_is_not_there_does_not_fall_back():
    """If someone points at a certificate and it is missing, that is a
    mistake to report, not a reason to quietly serve a different one."""
    from afm_litho import site

    old = dict(os.environ)
    try:
        os.environ["AFM_LITHO_TLS_CERT"] = "Z:/nope.crt"
        os.environ["AFM_LITHO_TLS_KEY"] = "Z:/nope.key"
        assert site.tls_paths() == (None, None)
    finally:
        os.environ.clear()
        os.environ.update(old)


# -------------------------------------------------------------- the port --

def test_443_is_opt_in_and_only_with_tls(monkeypatch):
    from afm_litho import site

    monkeypatch.setenv("AFM_LITHO_PORT", "443")
    assert site.choose_port(tls=False) != 443, "no certificate, no 443"
    monkeypatch.delenv("AFM_LITHO_PORT", raising=False)
    assert site.choose_port(tls=True) != 443, "443 must stay opt-in"


def test_the_bare_origin_carries_no_port(monkeypatch, tmp_path):
    """Google matches origins exactly, so :443 in the string is a mismatch."""
    import socket

    from afm_litho import site

    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **k: [(None, None, None, "", ("127.0.0.1", 0))])
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    cert.write_text("c"); key.write_text("k")
    monkeypatch.setenv("AFM_LITHO_TLS_CERT", str(cert))
    monkeypatch.setenv("AFM_LITHO_TLS_KEY", str(key))
    monkeypatch.setenv("AFM_LITHO_HOSTNAME", "panel.example.org")

    got = site.resolve_site(443)
    assert got["tls"] and got["origin"] == "https://panel.example.org"
    assert ":443" not in got["origin"] and ":443" not in got["url"]
