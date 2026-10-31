"""The desktop application: a window, not a terminal.

User request: "can we get rid of having to use powershell to launch? I hate the
terminal window inserting itself into the mix."

So this is the entry point that the installed shortcut runs. It starts the
service in-process on a free loopback port and puts the panel in a native
window (WebView2 on Windows, WebKit elsewhere), with no console, no browser
chrome, no address bar and no port to remember. Closing the window stops the
service.

Three things it deliberately does NOT do.

It does not pick a fixed port. Two copies open at once, or anything else
already on 7461, would otherwise fail in a way that reads as "the program is
broken". It takes the first free port and tells the window about it.

It does not show a splash screen while the service warms up. It waits for the
service to actually answer before the window appears, so the first thing seen
is the panel rather than an error page that needs reloading.

It does not silently swallow a failed start. If the service cannot come up
the reason goes in a message box, because a windowed app with no console has
nowhere else to put it, and "nothing happened when I double-clicked it" is
the worst possible failure.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
import urllib.error
import urllib.request

APP_NAME = "afm-litho"


def free_port(preferred: int = 7461) -> int:
    """The preferred port if it is free, else any free one the OS gives us."""
    for port in range(preferred, preferred + 40):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket() as s:              # give up on tidiness, not on starting
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(port: int, site: dict | None = None) -> None:
    """Kept as a name; the body lives in site.serve, which __main__ shares."""
    from afm_litho.site import serve

    serve(port, site)


def _wait_until_up(port: int, tls: bool = False, timeout_s: float = 40.0) -> bool:
    """Poll until the service answers, so the window never opens on an error."""
    import ssl
    # real time: waits on a real server, a launcher loop
    deadline = time.time() + timeout_s
    scheme = "https" if tls else "http"
    url = f"{scheme}://127.0.0.1:{port}/api/state"
    # Loopback with a certificate issued for the friendly name: the name will
    # not match 127.0.0.1, and this probe is only asking "is it listening
    # yet", so it does not verify. The browser still verifies properly,
    # because it connects by name.
    ctx = ssl._create_unverified_context() if tls else None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2, context=ctx):
                return True
        except urllib.error.HTTPError:
            return True                     # answering at all is the question
        except (urllib.error.URLError, OSError):
            time.sleep(0.25)
    return False


def _die(message: str) -> int:
    """A windowed app has no console, so a failure has to be shown."""
    sys.stderr.write(message + "\n")
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            None, message, f"{APP_NAME} could not start", 0x10)
    except Exception:  # noqa: BLE001, S110 - best-effort error dialog on the way out
        pass
    return 1


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="afm-litho-app")
    ap.add_argument("--port", type=int, default=0, help="0 picks a free port")
    ap.add_argument("--browser", action="store_true",
                    help="use the default browser instead of an app window")
    ap.add_argument("--no-update-check", action="store_true")
    args = ap.parse_args(argv)

    # The port depends on whether we are about to serve under our own name
    # over TLS: if so, 443 is worth having, because it is what lets the
    # address be `https://<name>` with nothing after it.
    from afm_litho.site import choose_port, cockpit_url, resolve_site, tls_possible
    port = args.port or choose_port(tls_possible())

    # The configured name if its DNS really points at the loopback and a
    # certificate is in place; plain 127.0.0.1 otherwise, with the reason
    # recorded rather than swallowed.
    site = resolve_site(port)
    for why in site["reasons"]:
        sys.stderr.write(f"[site] staying on 127.0.0.1: {why}\n")

    server = threading.Thread(target=_serve, args=(port, site), daemon=True)
    server.start()

    if not _wait_until_up(port, tls=site["tls"]):
        return _die("The instrument service did not start.\n\n"
                    "If this keeps happening, run afm-litho from a terminal "
                    "to see why:\n"
                    f"    python -m afm_litho --port {port}")

    url = cockpit_url(site["url"])       # the edit cockpit; / is the old main panel

    # The update check runs after the window is up and never blocks it. A
    # program that will not open because it cannot reach a server is worse
    # than a program that is one version behind.
    if not args.no_update_check:
        threading.Thread(target=_background_update_check, daemon=True).start()

    if args.browser:
        import webbrowser
        webbrowser.open(url)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return 0

    try:
        import webview
    except ImportError:
        import webbrowser
        webbrowser.open(url)                # still usable, just not a window
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return 0

    webview.create_window(
        f"{APP_NAME} · twin", url,
        width=1600, height=980, min_size=(1024, 640),
        background_color="#0b0e12",         # matches the panel: no white flash
        text_select=False,
    )
    # gui=None lets pywebview choose: WebView2 on Windows, WebKit elsewhere.
    webview.start(private_mode=False)       # keep localStorage between runs
    return 0


def _background_update_check() -> None:
    try:
        from afm_litho.update import check_and_notify
        check_and_notify()
    except Exception:  # noqa: BLE001, S110 - an update check never affects the app
        pass


if __name__ == "__main__":
    raise SystemExit(main())
