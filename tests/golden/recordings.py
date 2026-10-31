"""One recording per scenario per pytest process.

A golden scenario takes seconds to tens of seconds (the product's own settle waits), and
several test files judge the same recording. `recorded` plays a scenario once, with the
default settings, and hands the same result to every file that asks. Each xdist worker is
its own process and keeps its own cache; `golden_group` puts every test about one scenario
on one worker so the cache is hit.

A recording that needs a different setting (journal on, telemetry poller, a monkeypatched
constant) calls `golden.harness.record` itself and is never cached.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from afm_litho.service import run_store
from afm_litho.service.run_store import RunStore

from .harness import record_with_waits


@dataclass
class Recording:
    golden: dict
    runs: list = field(default_factory=list)       # stored runs, in the order they began
    waits: list = field(default_factory=list)      # waits the product commanded, in order


_CACHE: dict[tuple[str, int | None], Recording] = {}


def recorded(hub, client, name: str, microscope_id: int | None = None) -> Recording:
    key = (name, microscope_id)
    if key not in _CACHE:
        folder = run_store.store_dir(hub.store.path.parent)
        before = {p.name for p in folder.glob("*.run.json")} if folder.is_dir() else set()
        golden, waits = record_with_waits(hub, name, client=client, microscope_id=microscope_id)
        runs = []
        if folder.is_dir():
            store = RunStore(folder)
            ids = sorted(p.name.removesuffix(".run.json") for p in folder.glob("*.run.json")
                         if p.name not in before)
            runs = sorted((store.load(i) for i in ids), key=lambda r: r.execution.started)
        _CACHE[key] = Recording(golden, runs, waits)
    return _CACHE[key]


def golden_group(name: str):
    """xdist mark (needs `--dist loadgroup`): all tests about one scenario share a worker."""
    return pytest.mark.xdist_group(f"golden-{name}")
