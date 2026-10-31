"""The service process:  python -m afm_litho.service_main --port N [--host H] [--after-death]

Started by the keeper, which has already chosen the port and opens the browser.
Serves through site.serve, the one place uvicorn is started. The keeper adds
--after-death when it restarts a service that died: the tip is made safe first
(service/host/after_death.py).
"""

from __future__ import annotations

import argparse
import socket
import sys

from afm_litho.keeper.protocol import (
    AFTER_DEATH_ARG,
    BRIDGE_DOWN,
    BRIDGE_UP,
    LAST_BRIDGE_ARG,
    PORT_TAKEN_EXIT,
)
from afm_litho.service import keeper_link


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="afm-litho-service", description="Run the AFM service")
    ap.add_argument("--port", type=int, default=None,
                    help="used as-is; left off, site.py picks")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument(AFTER_DEATH_ARG, dest="after_death", action="store_true",
                    help="the service died; make the tip safe before anything else")
    ap.add_argument(LAST_BRIDGE_ARG, dest="last_bridge", choices=(BRIDGE_UP, BRIDGE_DOWN),
                    default=None, help="whether the service that died had a bridge to Igor")
    args = ap.parse_args(argv)

    keeper_link.claim_link()

    from afm_litho.hardware import wire
    print(f"wire: v{wire.WIRE_VERSION}", file=sys.stderr, flush=True)

    from afm_litho.site import choose_port, resolve_site, serve, tls_possible

    port = args.port if args.port is not None else choose_port(tls_possible())
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            print(f"Port {port} is already in use.", file=sys.stderr)
            return PORT_TAKEN_EXIT
    if args.after_death:
        # Before the app (and anything that can reach Igor) is even imported. After the
        # port check: a port in use may be a live service, whose pass must not be stopped.
        from afm_litho.service.hub import hub
        from afm_litho.service.host import after_death
        after_death.boot(hub, args.last_bridge)
    # An explicit host is served as asked, without a site name or certificate.
    site = resolve_site(port) if args.host in ("127.0.0.1", "0.0.0.0", "localhost") else None
    _quiet_windows_disconnects()
    serve(port, site)
    return 0


def _quiet_windows_disconnects() -> None:
    """Stop a browser refresh from printing a traceback.

    On Windows the proactor event loop raises ConnectionResetError from
    inside its own transport cleanup when a client goes away mid-connection.
    It is harmless and entirely outside our code, but it prints a scary
    multi-line traceback every time a tab is closed or reloaded.
    """
    import asyncio

    if sys.platform != "win32":
        return

    base = asyncio.proactor_events._ProactorBasePipeTransport  # type: ignore[attr-defined]
    original = base._call_connection_lost

    def patched(self, exc):
        try:
            original(self, exc)
        except (ConnectionResetError, ConnectionAbortedError):
            pass

    base._call_connection_lost = patched  # type: ignore[assignment]


if __name__ == "__main__":
    raise SystemExit(main())
