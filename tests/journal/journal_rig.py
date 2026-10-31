"""A twin write rig with the journal on, for the three journal questions.

The golden rig connects the Igor stub; this adds the one thing it lacks for a journal test,
a COM door that goes through `Igor.execute`, so `AFM_SafePark()` reaches the journal the way
it does on the instrument PC. Tests read only the journal file (`journal_read`), never the hub.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from types import SimpleNamespace

from golden.harness import GoldenRig, StubCom

from afm_litho.hardware.igor import Igor
from afm_litho.service import journal as journal_module
from afm_litho.service import journal_read as jr
from afm_litho.service.journal import Journal

WAIT_S = 60.0


def igor_over_stub(stub) -> Igor:
    """An `Igor` whose Execute2 runs the stub's COM model, so `execute` notes `com` entries."""
    com = StubCom(stub)

    def execute2(_a, _b, command):
        com.execute(command)
        return (0, "", "", "")

    igor = Igor()
    igor._app = SimpleNamespace(Execute2=execute2)
    return igor


@contextlib.contextmanager
def journalled_rig(hub, folder):
    """`with journalled_rig(hub, tmp) as (rig, journal):` a rig on the stub with a journal."""
    journal = Journal(folder).start()
    try:
        with GoldenRig(hub, "journal-questions") as rig:
            for holder in (rig.arc, rig._wd):          # the park runs on the watchdog's COM
                holder.com_igor = igor_over_stub(rig.stub)
            yield rig, journal
            rig.quiesce()
    finally:
        journal.stop()
        journal_module._running = None


def wait_in_file(journal: Journal, seen: Callable[[list[dict]], bool]) -> list[dict]:
    """Entries once the file shows `seen`: the test waits on the journal, not on the hub."""
    end = time.monotonic() + WAIT_S
    while time.monotonic() < end:
        entries = jr.load(journal.path)
        if seen(entries):
            return entries
        time.sleep(0.05)
    raise AssertionError("the journal file never showed what the test waits for")


def chunk_captured(name: str, chunks: int) -> Callable[[list[dict]], bool]:
    """True once the file shows `chunks` whole chunks of object `name` read from the instrument."""
    return lambda entries: jr.chunks_delivered(entries, name, 10**9) >= chunks


def command_sent(verb: str) -> Callable[[list[dict]], bool]:
    return lambda entries: any(jr.verb_of(e) == verb for e in jr.commands(entries))
