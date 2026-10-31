"""No new direct waits: product code waits through afm_litho.clock.

Every direct wait and every direct clock read that exists today is listed below as
(file, kind) -> count. The list may only shrink: a slice that routes a wait through
the clock lowers its count here in the same commit. A wait outside the list fails.

time.monotonic / time.time / time.perf_counter reads are their own kinds so that a
converted deadline loop cannot keep reading the real clock unnoticed.
"""
import ast
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "afm_litho"

# module -> names that wait or read a clock (attribute use and from-import both count).
_WATCHED = {
    "time": {"sleep", "monotonic", "time", "perf_counter"},
    "asyncio": {"sleep", "wait_for"},
    "threading": {"Timer"},
}
_LOOP_TIMERS = {"call_later", "call_at"}


def _aliases(tree: ast.AST) -> dict[str, str]:
    """Local name -> real module, for `import time as t` and `import asyncio as aio`."""
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names[a.asname or a.name] = a.name
    return names


def _clock_names(tree: ast.AST) -> set[str]:
    """Local names bound to the afm_litho clock module (not any variable named `clock`)."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in (None, "", "afm_litho"):
            names |= {a.asname or a.name for a in node.names if a.name == "clock"}
        elif isinstance(node, ast.Import):
            names |= {a.asname for a in node.names if a.name == "afm_litho.clock" and a.asname}
    return names


def _receiver(node: ast.expr) -> str:
    """Last name of the receiver chain (self._abort_event -> _abort_event)."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _wait_method_kind(call: ast.Call, modules: dict[str, str],
                      clocks: frozenset = frozenset()) -> str | None:
    """Kind for a method call that waits, or None."""
    func = call.func
    attr, recv = func.attr, func.value
    real = modules.get(recv.id, recv.id) if isinstance(recv, ast.Name) else ""
    kwargs = {k.arg for k in call.keywords}
    if isinstance(recv, ast.Name) and recv.id in clocks:
        return None                      # clock.wait and clock.wait_for are the routed forms
    if attr == "wait":
        if real == "asyncio":
            return "asyncio.wait"
        if "proc" in _receiver(recv).lower():
            return "Process.wait"
        return "Event.wait"
    if attr == "wait_for" and real != "asyncio":
        return "Condition.wait_for"
    if attr in _LOOP_TIMERS:
        return f"loop.{attr}"
    if attr == "get" and ("timeout" in kwargs
                          or (len(call.args) >= 2 and isinstance(call.args[0], ast.Constant)
                              and call.args[0].value is True)):
        return "queue.get(timeout)"
    if attr == "acquire" and ("timeout" in kwargs or len(call.args) >= 2):
        return "lock.acquire(timeout)"
    if attr == "result" and (call.args or "timeout" in kwargs):
        return "future.result(timeout)"
    return None


def _direct_waits(path: Path) -> list[str]:
    """Kinds of direct wait or clock read in one file, one entry per occurrence."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = _aliases(tree)
    clocks = frozenset(_clock_names(tree))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            module = modules.get(node.value.id, node.value.id)
            if node.attr in _WATCHED.get(module, ()):
                found.append(f"{module}.{node.attr}")
        elif isinstance(node, ast.ImportFrom) and node.module in _WATCHED:
            found += [f"{node.module}.{a.name}" for a in node.names
                      if a.name in _WATCHED[node.module]]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            kind = _wait_method_kind(node, modules, clocks)
            if kind:
                found.append(kind)
    return found


def _today() -> Counter:
    counts: Counter = Counter()
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "clock.py":
            continue
        rel = path.relative_to(SRC).as_posix()
        for kind in _direct_waits(path):
            counts[(rel, kind)] += 1
    return counts


ALLOWED: dict[tuple[str, str], int] = {
    ('desktop.py', 'time.sleep'): 3,
    ('desktop.py', 'time.time'): 2,
    ('hardware/failsafe_thread.py', 'Event.wait'): 3,
    ('hardware/failsafe_thread.py', 'time.time'): 1,
    ('hardware/igor_stub.py', 'time.sleep'): 1,
    # The twin's telemetry thread is a background loop and stays on real time (clock rule a).
    ('instrument/twin.py', 'Event.wait'): 1,
    # The Asylum plugin waits for real threads: a park for the write in flight, and for the
    # fail-safe thread's outcome (clock rule b).
    ('instrument/asylum.py', 'Event.wait'): 2,
    ('service/access.py', 'time.time'): 2,
    ('service/app.py', 'asyncio.wait_for'): 1,
    ('service/host/continuous_cmds.py', 'time.time'): 1,
    ('service/commands/design_cmds.py', 'asyncio.wait_for'): 1,
    ('service/host/litho_cmds.py', 'asyncio.wait_for'): 1,
    ('service/host/litho_cmds.py', 'time.monotonic'): 8,
    ('service/host/litho_cmds.py', 'time.time'): 2,
    ('service/host/scan_cmds.py', 'time.time'): 2,
    ('service/control.py', 'asyncio.sleep'): 1,
    ('service/control.py', 'time.time'): 6,
    ('service/design_store.py', 'Event.wait'): 2,
    ('service/drift.py', 'time.time'): 2,
    ('service/flex_provider.py', 'future.result(timeout)'): 1,
    ('service/host/beat.py', 'asyncio.sleep'): 1,
    ('service/host/continuous.py', 'asyncio.sleep'): 1,
    ('service/host/continuous.py', 'time.time'): 4,
    ('service/host/litho_worker.py', 'asyncio.sleep'): 1,
    ('service/host/litho_worker.py', 'time.monotonic'): 1,
    ('service/host/litho_worker.py', 'time.time'): 6,
    # real time: set by the scan's own worker thread, bounded by the link timeout
    ('service/host/scan_pump.py', 'future.result(timeout)'): 1,
    ('service/host/scan_pump.py', 'time.time'): 1,
    ('service/host/telemetry.py', 'time.time'): 1,
    ('service/hub.py', 'time.time'): 3,
    ('service/hw_scan.py', 'asyncio.sleep'): 1,
    ('service/hw_scan.py', 'time.time'): 6,
    ('service/journal.py', 'queue.get(timeout)'): 1,
    ('service/journal.py', 'time.monotonic'): 3,
    ('service/passes.py', 'time.monotonic'): 1,
    ('service/passes.py', 'time.sleep'): 1,
    ('service/routers/drift.py', 'time.time'): 1,
    ('service/runs.py', 'time.time'): 2,
    ('service/host/engage.py', 'time.monotonic'): 2,
    ('service/host/engage.py', 'time.sleep'): 2,
    ('service/host/engage.py', 'time.time'): 2,
    ('service/host/safe_park.py', 'asyncio.wait_for'): 1,
    ('service/host/safe_park.py', 'time.monotonic'): 2,
    ('service/host/safe_park.py', 'time.sleep'): 1,
    ('service/host/safe_park.py', 'time.time'): 4,
    ('service/sessions.py', 'time.time'): 1,
    ('service/stems.py', 'time.time'): 2,
    ('service/surface.py', 'time.time'): 4,
    ('service/twin_player.py', 'time.monotonic'): 1,
    ('service/twin_player.py', 'time.sleep'): 1,
    ('service/twin_tip_stream.py', 'asyncio.sleep'): 1,
    ('service/twin_tip_stream.py', 'time.monotonic'): 4,
}


def test_no_new_direct_waits():
    today = _today()
    new = {k: n - ALLOWED.get(k, 0) for k, n in today.items() if n > ALLOWED.get(k, 0)}
    assert not new, (
        "direct waits or clock reads outside afm_litho/clock.py (route them through "
        f"the clock): {new}")


def test_allow_list_has_no_stale_entries():
    today = _today()
    stale = {k: n for k, n in ALLOWED.items() if today.get(k, 0) < n}
    assert not stale, f"already routed through the clock; lower these counts: {stale}"


def test_guard_sees_aliases_and_less_obvious_waits(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import time as t\nimport asyncio as aio\nfrom time import sleep\nfrom afm_litho import clock\n"
        "def f(q, lock, fut, loop, cond, ev, proc):\n"
        "    t.sleep(1); aio.sleep(1); aio.wait_for(x, 1); aio.wait([x])\n"
        "    q.get(True, 1); lock.acquire(timeout=1); fut.result(2)\n"
        "    loop.call_later(1, f); cond.wait_for(p, 1); ev.wait(1); proc.wait()\n"
        "    clock.wait(ev, 1); clock.wait_for(p, 1)\n"
        "    return t.monotonic()\n", encoding="utf-8")
    assert Counter(_direct_waits(sample)) == Counter({
        "time.sleep": 2, "asyncio.sleep": 1, "asyncio.wait_for": 1, "asyncio.wait": 1,
        "queue.get(timeout)": 1, "lock.acquire(timeout)": 1, "future.result(timeout)": 1,
        "loop.call_later": 1, "Condition.wait_for": 1, "Event.wait": 1,
        "Process.wait": 1, "time.monotonic": 1})
