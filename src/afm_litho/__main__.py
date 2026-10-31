"""Run the twin:  python -m afm_litho

This starts the keeper (afm_litho.keeper), which chooses the port, starts the
service as a child process (afm_litho.service_main), opens the edit cockpit in the
default browser and restarts the service if it dies. No uvicorn command line to
remember, no arguments required.

This and afm_litho.desktop serve the SAME service the same way: both ask
site.py which port and whether to use TLS, and both start it through
site.serve. They did not, and the split was invisible until it mattered:
`run.ps1` -- the documented way to run the twin -- served plain HTTP on 7461 and
could never answer at the configured https name.
"""

from afm_litho.keeper.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
