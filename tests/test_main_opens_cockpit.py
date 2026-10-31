"""`python -m afm_litho` and the desktop app open the edit cockpit, not the old main
panel (docs/TWIN_TIP_STREAM_SPEC.md, "the service opens the edit cockpit at start")."""

from __future__ import annotations

import importlib.util
import socket

import pytest

import afm_litho.keeper.cli as entry
from afm_litho import clock, site


@pytest.mark.parametrize("root,want", [
    ("http://127.0.0.1:7461/", "http://127.0.0.1:7461/edit-cockpit"),
    ("https://panel.example.org/", "https://panel.example.org/edit-cockpit"),
    ("https://panel.example.org:8443/", "https://panel.example.org:8443/edit-cockpit"),
    ("http://10.0.0.5:8000", "http://10.0.0.5:8000/edit-cockpit"),
])
def test_cockpit_url_joins_with_exactly_one_slash(root, want):
    assert site.cockpit_url(root) == want


def test_the_cockpit_route_exists_and_root_redirects_to_it(client):
    assert client.get(site.COCKPIT_PATH).status_code == 200
    root = client.get("/", follow_redirects=False)
    assert root.status_code == 307
    assert root.headers["location"] == site.COCKPIT_PATH


class _QuitsAfter:
    """A launcher whose service child polls as running a few times, then exits cleanly
    (the keeper ends with it). Stands in for the server."""

    def __init__(self, polls=8):
        self.polls = polls
        self.launched = []
        self.port = None
        self.listener = None

    def __call__(self, spec, extra):
        self.launched.append(list(spec.argv))
        self.listener = socket.socket()          # the service's port answers once it is up
        self.listener.bind(("127.0.0.1", self.port))
        self.listener.listen(8)
        child = self

        class Handle:
            pid = 4242

            def poll(self):
                child.polls -= 1
                return None if child.polls > 0 else 0

            def drain(self):
                return []

            def terminate(self):
                pass

        return Handle()


def _run_main(monkeypatch, argv, site_url, launcher=None):
    """Run the keeper's `main(argv)` with the service replaced by a fake child and a
    listening socket standing in for its port, so the opener sees the port up."""
    opened: list[str] = []
    monkeypatch.setattr(entry.webbrowser, "open", lambda u, *a, **k: opened.append(u))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setattr(site, "tls_possible", lambda: False)
    monkeypatch.setattr(site, "choose_port", lambda tls: port)
    monkeypatch.setattr(site, "resolve_site", lambda p: {
        "url": site_url.format(port=p), "reasons": []})
    # an explicit --host slides to a free port from --port: hand it the listener's
    monkeypatch.setattr(entry, "_free_port", lambda preferred: port)
    launcher = launcher or _QuitsAfter()
    launcher.port = port
    try:
        with clock.using(clock.TESTING):
            assert entry.main(argv, launch=launcher) == 0
    finally:
        if launcher.listener:
            launcher.listener.close()
    return opened, port


def test_main_opens_the_cockpit_route(monkeypatch, capsys):
    opened, port = _run_main(monkeypatch, [], "http://127.0.0.1:{port}/")
    assert opened == [f"http://127.0.0.1:{port}/edit-cockpit"]     # once, however long it ran
    out = capsys.readouterr().out
    assert f"http://127.0.0.1:{port}/edit-cockpit" in out      # the address to open


def test_main_opens_the_cockpit_under_a_site_name(monkeypatch):
    opened, _ = _run_main(monkeypatch, [], "https://panel.example.org/")
    assert opened == ["https://panel.example.org/edit-cockpit"]


def test_main_opens_the_cockpit_on_an_explicit_host(monkeypatch):
    opened, port = _run_main(monkeypatch, ["--host", "10.1.2.3"], "unused")
    assert opened == [f"http://10.1.2.3:{port}/edit-cockpit"]


def test_no_browser_opens_nothing(monkeypatch):
    opened, _ = _run_main(monkeypatch, ["--no-browser"], "http://127.0.0.1:{port}/")
    assert opened == []


def test_the_port_and_host_are_passed_on_to_the_service(monkeypatch):
    launcher = _QuitsAfter()
    _, port = _run_main(monkeypatch, ["--no-browser"], "http://127.0.0.1:{port}/", launcher)
    argv = next(a for a in launcher.launched if "afm_litho.service_main" in a)  # not the recorder
    assert argv[1:3] == ["-m", "afm_litho.service_main"]
    assert argv[argv.index("--port") + 1] == str(port)
    assert argv[argv.index("--host") + 1] == "127.0.0.1"


# ------------------------------------------------------------- desktop app --

def _run_desktop(monkeypatch, argv):
    if not importlib.util.find_spec("afm_litho.desktop"):
        pytest.skip("desktop.py is not in this tree")
    """Run `desktop.main(argv)` with no server and no window; the browser fallbacks
    idle in `time.sleep(3600)`, which is turned into the Ctrl+C that ends them."""
    import sys
    import types
    import webbrowser

    from afm_litho import desktop

    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda u, *a, **k: opened.append(u))
    monkeypatch.setattr(desktop, "_serve", lambda port, s: None)
    monkeypatch.setattr(desktop, "_wait_until_up", lambda port, tls=False: True)
    monkeypatch.setattr(site, "tls_possible", lambda: False)
    monkeypatch.setattr(site, "choose_port", lambda tls: 7999)
    monkeypatch.setattr(site, "resolve_site", lambda p: {
        "url": "https://panel.example.org/", "reasons": [], "tls": False})

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(desktop.time, "sleep", interrupt)
    windows: list[str] = []
    fake = types.SimpleNamespace(
        create_window=lambda title, url, **kw: windows.append(url),
        start=lambda **kw: None)
    return desktop, opened, windows, fake, sys


def test_desktop_browser_mode_opens_the_cockpit(monkeypatch):
    desktop, opened, _, _, _ = _run_desktop(monkeypatch, None)
    assert desktop.main(["--browser", "--no-update-check"]) == 0
    assert opened == ["https://panel.example.org/edit-cockpit"]


def test_desktop_window_shows_the_cockpit(monkeypatch):
    desktop, opened, windows, fake, sys = _run_desktop(monkeypatch, None)
    monkeypatch.setitem(sys.modules, "webview", fake)
    assert desktop.main(["--no-update-check"]) == 0
    assert windows == ["https://panel.example.org/edit-cockpit"] and opened == []


def test_desktop_without_pywebview_falls_back_to_the_cockpit(monkeypatch):
    desktop, opened, _, _, sys = _run_desktop(monkeypatch, None)
    monkeypatch.setitem(sys.modules, "webview", None)       # import raises ImportError
    assert desktop.main(["--no-update-check"]) == 0
    assert opened == ["https://panel.example.org/edit-cockpit"]
