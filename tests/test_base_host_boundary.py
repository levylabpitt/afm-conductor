"""The base may not reach into the host: import and word rules, with a shrinking allowlist.

Two sides of the H8 process split (docs/migration/plan/host-lift-board/plan.md):
  host  = service/host/, service/safety/, service/host_state.py. It owns the tip, the
          controller and the safety state, and will become its own process.
  base  = every other file under service/ (doors, requests, routers, hub, board, ...).

Rules for a base file (checked on every import, including lazy ones inside functions):
  1. It may not import `afm_litho.hardware` or `afm_litho.twin`.
  2. It may import nothing from the host side except `service.host.api`, and only
     `service/requests/` and `service/flex_dispatch.py` may import that.
  3. It may not use a name in `HOST_FIELD_NAMES` on the hub (`hub.engaged`,
     `self.hub.fault`, `getattr(hub, "arc")`), nor name `HOST_FIELD_NAMES` itself.
     The base reads host state through the HostApi (`snapshot`) instead.

The allowlist below is today's violations, as (file, what) -> count, each file with its
reason. It may only shrink: a new violation or a higher count fails, and so does a count
that is now too high, so whoever fixes a file lowers or deletes its entry in the same commit.
Not encoded: the plan's ban on Igor sockets/COM in the base (board.md, "Igor"), since
plan.md's H7a row does not list it.

What rule 3 does not see (limits of the detector):
  - `hub.host.F`. Since H7b every host field is read and written on `hub.host`, so the
    detector now counts only `fault` (whose Hub property stays) and `HOST_FIELD_NAMES`.
    A base file that reaches host state through `hub.host` is not counted here.
  - receivers other than `hub` or a chain ending in `.hub` (`h`, `self`, `_hub`, a
    local alias), and names held in a variable (`getattr(hub, name)`).
  - `vars(hub)`, `hub.__dict__`, and fields reached through another object.
"""
import ast
from collections import Counter
from pathlib import Path

from afm_litho.service.host_state import HOST_FIELD_NAMES

SRC = Path(__file__).resolve().parents[1] / "src"
PKG = SRC / "afm_litho"
SERVICE = PKG / "service"
ROOT = "afm_litho"

HOST_SIDE = ("afm_litho.service.host", "afm_litho.service.safety", "afm_litho.service.host_state")
HOST_API = "afm_litho.service.host.api"
API_ALLOWED_FOR = ("service/requests/", "service/flex_dispatch.py")
FIELDS = frozenset(HOST_FIELD_NAMES)


# --- which files are base -------------------------------------------------------------------

def _is_host_file(rel: str) -> bool:
    """`rel` is relative to afm_litho/, with forward slashes."""
    return rel.startswith(("service/host/", "service/safety/")) or rel == "service/host_state.py"


def _base_files() -> list[Path]:
    return [p for p in sorted(SERVICE.rglob("*.py"))
            if not _is_host_file(p.relative_to(PKG).as_posix())]


# --- reading one file -----------------------------------------------------------------------

def _module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _is_module(dotted: str, root: Path) -> bool:
    base = root.joinpath(*dotted.split("."))
    return base.with_suffix(".py").exists() or base.is_dir()


def _imported_modules(tree: ast.AST, this: str, is_package: bool, root: Path) -> list[str]:
    """Absolute dotted module for every import in the file (lazy ones included)."""
    package = this if is_package else this.rpartition(".")[0]
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - (node.level - 1)]
                                + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            for a in node.names:
                sub = f"{base}.{a.name}"
                found.append(sub if _is_module(sub, root) else base)
    return found


def _hub_receiver(node: ast.expr) -> bool:
    """`hub`, or anything ending in `.hub` (self.hub, deps.hub)."""
    return (isinstance(node, ast.Name) and node.id == "hub") or (
        isinstance(node, ast.Attribute) and node.attr == "hub")


def _violations(path: Path, root: Path = SRC) -> list[str]:
    """One entry per violation: `import:hardware`, `import:twin`, `import:host.<mod>`,
    `import:host.api`, `import:safety...`, `field:<name>`, `field:HOST_FIELD_NAMES`."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    rel = path.relative_to(root / ROOT).as_posix()
    found = []
    for mod in _imported_modules(tree, _module_name(path, root), path.name == "__init__.py", root):
        parts = mod.split(".")
        if parts[0] == ROOT and len(parts) > 1 and parts[1] in ("hardware", "twin"):
            found.append(f"import:{parts[1]}")
        elif mod == HOST_API or mod.startswith(HOST_API + "."):
            if not rel.startswith(API_ALLOWED_FOR):
                found.append("import:host.api")
        elif any(mod == h or mod.startswith(h + ".") for h in HOST_SIDE):
            found.append("import:" + mod.removeprefix("afm_litho.service."))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in FIELDS and _hub_receiver(node.value):
            found.append(f"field:{node.attr}")
        elif isinstance(node, ast.Name) and node.id == "HOST_FIELD_NAMES":
            found.append("field:HOST_FIELD_NAMES")
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id in ("getattr", "setattr", "hasattr", "delattr")
              and len(node.args) >= 2 and _hub_receiver(node.args[0])
              and isinstance(node.args[1], ast.Constant) and node.args[1].value in FIELDS):
            found.append(f"field:{node.args[1].value}")
    return found


def _today() -> Counter:
    counts: Counter = Counter()
    for path in _base_files():
        rel = path.relative_to(PKG).as_posix()
        for what in _violations(path):
            counts[(rel, what)] += 1
    return counts


# --- today's violations ---------------------------------------------------------------------
# REASONS gives each file's reason; ALLOWED gives the counts. Delete both when a file is clean.

REASONS: dict[str, str] = {
    # On the plan's own list (H7a): leftovers it names
    'service/hub.py':
        'the process hub: builds the twin and hardware objects, refuses HOST_FIELD_NAMES in __setattr__, imports host_state and safe_park (plan H7: hub.py is on the list)',
    'service/board.py':
        'the cortex reads host state by contract until H8 (plan H7 list: board.py; fed by snapshot() at H8)',
    # Boot, backend, drivers and twin playback: move with the host at H8
    'service/app.py':
        'start-up builds the backend and starts the stop port, after-death and telemetry in-process; moves when the host becomes its own process (H8b)',
    'service/backends.py':
        'backend selection imports the hardware backend resolver',
    'service/hw_scan.py':
        'hardware scan driver: talks to the ARC and calls safe_park; belongs with the host scan pump',
    'service/instrument_model.py':
        'reads channel aliases and the free-air band from hardware; the model reaches the base through the snapshot later',
    'service/passes.py':
        'pass runner imports the twin and hardware directly',
    'service/deps.py':
        'builds the twin Instrument for the service dependencies',
    'service/twin_player.py':
        'twin playback imports the twin directly',
    # Dead-man and keeper
    'service/control.py':
        'the dead-man watcher hands a lapse to HostApi.commander_lapsed, and drives the host engaged-idle cap from the same tick to keep the dead-man-then-cap order (H7.5d; the host.deadman import goes when the cap gets its own host task)',
    'service/keeper_link.py':
        'keeper-lost handling reads backend and fault on the hub; moves with the dead-man (H7.5d)',
    # FLEX door, commands and design
    'service/flex_dispatch.py':
        'FLEX door still imports host modules lazily and reads hub fields that the HostApi verbs and snapshot() replace; get_state is a call to host deadman.pass_state until snapshot() carries the state; its host.api imports are allowed and not listed',
    'service/commands/design_cmds.py':
        'design commands read the write ledger and instrument model on the hub and call litho_cmds',
    'service/design_store.py':
        'design store reads hub fields and imports litho_worker',
    # requests/ rows
    'service/requests/cards.py':
        'card row calls scan_cmds directly instead of HostApi',
    'service/requests/dispatch.py':
        'dispatch imports wire_trace from the hardware bridge protocol',
    'service/requests/reads.py':
        'read rows call litho_cmds and scan_cmds directly instead of HostApi',
    'service/requests/stops.py':
        'the stop row imports safety.stop_pairs for the keeper copy; it needs a sanctioned stop door (board, F5)',
    # routers/
    'service/routers/continuous.py':
        'continuous router imports host modules and reads hub fields',
    'service/routers/instrument.py':
        'instrument router reads hub fields and imports instrument_cmds',
    'service/routers/litho.py':
        'litho router reaches into litho_cmds, litho_worker, the twin world and hardware; its host.api import is outside requests/ and flex_dispatch',
    'service/routers/meta.py':
        'meta router reads twin fields on the hub and imports the telemetry module',
    'service/routers/sample.py':
        'sample router builds and reads the twin world',
    'service/routers/scan.py':
        'scan router calls safe_park and scan_cmds and reads scan_repeat (footnote 7: the cockpit-only scan_repeat clear stays here on purpose); its host.api import is outside requests/ and flex_dispatch',
    'service/routers/tip.py':
        'tip router imports the hardware bridge protocol',
    # Small helpers importing hardware
    'service/step_notes.py':
        'step notes import wire_step from the hardware bridge protocol',
    'service/timeline.py':
        'timeline imports hardware channels',
}

ALLOWED: dict[tuple[str, str], int] = {
    # On the plan's own list (H7a): leftovers it names
    ('service/hub.py', 'field:HOST_FIELD_NAMES'): 1,
    ('service/hub.py', 'import:hardware'): 3,
    ('service/hub.py', 'import:host.safe_park'): 7,
    ('service/hub.py', 'import:host_state'): 2,
    ('service/hub.py', 'import:twin'): 4,
    ('service/board.py', 'field:fault'): 2,
    ('service/board.py', 'import:host.telemetry'): 1,
    # Boot, backend, drivers and twin playback: move with the host at H8
    ('service/app.py', 'import:hardware'): 3,
    ('service/app.py', 'import:host.after_death'): 1,
    ('service/app.py', 'import:host.litho_cmds'): 1,
    ('service/app.py', 'import:host.safe_park'): 1,
    ('service/app.py', 'import:host.stop_port'): 1,
    ('service/app.py', 'import:host.telemetry'): 1,
    ('service/backends.py', 'import:hardware'): 4,
    ('service/hw_scan.py', 'import:hardware'): 9,
    ('service/hw_scan.py', 'import:host.safe_park'): 2,
    ('service/instrument_model.py', 'import:hardware'): 3,
    ('service/passes.py', 'import:hardware'): 1,
    ('service/passes.py', 'import:twin'): 4,
    ('service/deps.py', 'import:twin'): 1,
    ('service/twin_player.py', 'import:twin'): 2,
    # Dead-man and keeper
    ('service/control.py', 'import:host.api'): 1,
    ('service/control.py', 'import:host.deadman'): 1,
    ('service/keeper_link.py', 'field:fault'): 1,
    # FLEX door, commands and design
    ('service/flex_dispatch.py', 'import:host.deadman'): 1,
    ('service/flex_dispatch.py', 'import:host.litho_cmds'): 5,
    ('service/flex_dispatch.py', 'import:host.litho_worker'): 1,
    ('service/flex_dispatch.py', 'import:host.safe_park'): 1,
    ('service/flex_dispatch.py', 'import:twin'): 1,
    ('service/commands/design_cmds.py', 'import:host.litho_cmds'): 1,
    ('service/design_store.py', 'import:host.litho_worker'): 1,
    # requests/ rows
    ('service/requests/cards.py', 'import:host.scan_cmds'): 1,
    ('service/requests/dispatch.py', 'import:hardware'): 1,
    ('service/requests/reads.py', 'import:host.litho_cmds'): 1,
    ('service/requests/reads.py', 'import:host.scan_cmds'): 1,
    ('service/requests/stops.py', 'import:safety.stop_pairs'): 1,
    # routers/
    ('service/routers/continuous.py', 'import:host.continuous'): 1,
    ('service/routers/continuous.py', 'import:host.continuous_cmds'): 1,
    ('service/routers/instrument.py', 'import:host.instrument_cmds'): 1,
    ('service/routers/litho.py', 'import:hardware'): 1,
    ('service/routers/litho.py', 'import:host.api'): 1,
    ('service/routers/litho.py', 'import:host.litho_cmds'): 1,
    ('service/routers/litho.py', 'import:host.litho_worker'): 2,
    ('service/routers/meta.py', 'import:host.telemetry'): 1,
    ('service/routers/sample.py', 'import:twin'): 4,
    ('service/routers/scan.py', 'import:host.api'): 1,
    ('service/routers/scan.py', 'import:host.safe_park'): 4,
    ('service/routers/scan.py', 'import:host.scan_cmds'): 1,
    ('service/routers/tip.py', 'import:hardware'): 1,
    # Small helpers importing hardware
    ('service/step_notes.py', 'import:hardware'): 1,
    ('service/timeline.py', 'import:hardware'): 1,
}


def test_no_new_boundary_violations():
    today = _today()
    new = {k: n - ALLOWED.get(k, 0) for k, n in today.items() if n > ALLOWED.get(k, 0)}
    assert not new, f"the base reaches into the host (see the rules at the top of the file): {new}"


def test_allow_list_has_no_stale_entries():
    today = _today()
    stale = {k: n for k, n in ALLOWED.items() if today.get(k, 0) < n}
    assert not stale, f"already fixed; lower or delete these counts: {stale}"


def test_every_allowed_file_has_a_reason():
    files = {f for f, _ in ALLOWED}
    assert files == set(REASONS), (files ^ set(REASONS))
    assert all(r.strip() for r in REASONS.values())


def test_allowed_files_are_base_files():
    for f, _ in ALLOWED:
        assert (PKG / f).exists() and not _is_host_file(f), f


# --- detector self-test ---------------------------------------------------------------------

def _fake_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "src"
    for name, text in files.items():
        p = root / "afm_litho" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    for pkg in ("", "service", "service/host", "service/safety", "service/requests",
                "service/routers", "hardware", "twin"):
        d = root / "afm_litho" / pkg
        d.mkdir(parents=True, exist_ok=True)
        (d / "__init__.py").write_text("", encoding="utf-8")
    return root


def test_detector_catches_each_rule(tmp_path):
    root = _fake_tree(tmp_path, {
        "service/host/api.py": "", "service/host/safe_park.py": "",
        "service/host_state.py": "", "hardware/arc.py": "", "twin/world.py": "",
        "service/bad.py": (
            "import afm_litho.hardware.arc\n"                    # import:hardware
            "from afm_litho import twin\n"                       # import:twin
            "from .. import hardware\n"                          # import:hardware (relative)
            "from .host.safe_park import request_park\n"         # import:host.safe_park
            "from .host import safe_park\n"                      # import:host.safe_park
            "from .host.api import HostApi\n"                    # import:host.api, not allowed here
            "from .safety import RailDwellGuard\n"               # import:safety
            "from .host_state import HostState\n"                # import:host_state
            "def f(hub, self, deps):\n"
            "    import afm_litho.twin.world\n"                  # import:twin (lazy)
            "    hub.engaged = True\n"                           # field:engaged
            "    self.hub.fault\n"                               # field:fault
            "    getattr(hub, 'arc')\n"                          # field:arc
            "    hub.run_store\n"                                # not a host field: clean
            "    deps.engaged\n"                                 # not the hub: clean
            "    return HOST_FIELD_NAMES\n"),                    # field:HOST_FIELD_NAMES
    })
    got = Counter(_violations(root / "afm_litho/service/bad.py", root))
    assert got == Counter({
        "import:hardware": 2, "import:twin": 2, "import:host.safe_park": 2,
        "import:host.api": 1, "import:safety": 1, "import:host_state": 1,
        "field:engaged": 1, "field:fault": 1, "field:arc": 1, "field:HOST_FIELD_NAMES": 1})


def test_detector_lets_the_api_doors_import_host_api(tmp_path):
    root = _fake_tree(tmp_path, {
        "service/host/api.py": "",
        "service/requests/tips.py": "from ..host.api import HostApi\n",
        "service/flex_dispatch.py": "def f():\n    from .host.api import HostApi\n",
        "service/routers/scan.py": "from ..host.api import HostApi\n",
    })
    assert _violations(root / "afm_litho/service/requests/tips.py", root) == []
    assert _violations(root / "afm_litho/service/flex_dispatch.py", root) == []
    assert _violations(root / "afm_litho/service/routers/scan.py", root) == ["import:host.api"]


def test_host_files_are_not_checked():
    assert _is_host_file("service/host/scan_cmds.py")
    assert _is_host_file("service/safety/rail_dwell.py")
    assert _is_host_file("service/host_state.py")
    assert not _is_host_file("service/hub.py")
    assert all("/host/" not in p.as_posix() for p in _base_files())


# --- H7.5d: the dead-man's verdict lives in the host -----------------------------------------

def test_the_host_deadman_imports_no_base_arbitration_or_board():
    """host/deadman.py reads host state only; it never reaches the base's commander code."""
    path = SERVICE / "host" / "deadman.py"
    mods = _imported_modules(ast.parse(path.read_text(encoding="utf-8")),
                             "afm_litho.service.host.deadman", False, SRC)
    banned = {f"afm_litho.service.{m}" for m in ("board", "flex_dispatch", "access", "control")}
    assert not [m for m in mods if m in banned or any(m.startswith(b + ".") for b in banned)]


def test_access_takes_no_instrument_action():
    """access.py releases the token; the park or withdraw is the host's."""
    tree = ast.parse((SERVICE / "access.py").read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not names & {"request_park", "_withdraw", "do_withdraw", "get_state", "pass_state"}


# --- H7.5e: what the host still calls on the base hub ----------------------------------------
# Host code that calls a method on the process hub reaches across the future link. This is
# every such call today, counted by method name. It only shrinks: a new base call or a higher
# count fails, and so does a count that is now too high (lower or delete it in the same
# change). The claim and run-id calls stay until F3 / J3 (where the claim is taken
# and who mints run ids at H8). The 4 `release_claim` calls are the A rows of H7.5b.

HOST_CALLS_BASE: dict[str, int] = {
    "claim_run": 2,
    "release_claim": 4,
    "next_run_id": 5,
    "finish_run": 7,
    "busy_body": 2,
    "_in_flight_run_id": 6,
    "backend_live": 10,
    "backend_label": 8,
    "require_live_backend": 3,
    "refresh_xy_cal": 3,
    "notify_design_changed": 2,
    "rearm_deadman": 1,
    "broadcast": 28,
    "log": 124,
    "_log_sync": 5,
}


def _host_calls_on_hub() -> Counter:
    host_files = [p for p in sorted(SERVICE.rglob("*.py"))
                  if _is_host_file(p.relative_to(PKG).as_posix())]
    counts: Counter = Counter()
    for path in host_files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and (_hub_receiver(node.func.value) or (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id in ("_hub", "h")))):
                counts[node.func.attr] += 1
    return counts


def test_host_calls_on_the_base_hub_only_shrink():
    today = _host_calls_on_hub()
    grown = {k: n for k, n in today.items() if n > HOST_CALLS_BASE.get(k, 0)}
    assert not grown, f"the host calls a new base method (or more often): {grown}"
    stale = {k: n for k, n in HOST_CALLS_BASE.items() if today.get(k, 0) < n}
    assert not stale, f"already fewer; lower or delete these counts: {stale}"
