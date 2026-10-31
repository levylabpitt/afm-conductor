"""Records the HTTP door as canonical JSON transcripts (slice 2.0b).

One twin service child process per transcript group, driven over real HTTP and a
real /ws socket, so each group starts from a clean hub and nothing leaks between
groups or xdist workers. Each step is one request: its status code, its body, and
the WS events it caused. The transcript is compared after masking the volatile
fields below; regenerate only with AFM_LITHO_REGEN_HTTP_TRANSCRIPTS=1.

Test-only: nothing here changes a product constant or timing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from _service import Service

TRANSCRIPT_DIR = Path(__file__).parent / "transcripts"
REGEN_ENV = "AFM_LITHO_REGEN_HTTP_TRANSCRIPTS"
MASK = "<masked>"

# ------------------------------------------------------------------ mask list --
# The whole list. Anything not named here is compared byte for byte.
#
# 1. A key whose value is a clock reading or a one-off secret. The value is
#    replaced; the key stays, so a field that vanishes still fails the test.
VOLATILE_KEYS = frozenset({
    "t", "ts", "t_start", "t_end", "age_s", "expires_s", "token", "started_at",
    "finished_at", "mtime", "saved_at", "created", "elapsed_s", "uptime_s", "acquired_at",
    "last_seen",
    # which of these two is set depends on whether klayout is installed
    "geometry", "geometry_error",
})
# 2. A string that embeds a one-off id. Matched anywhere inside the string.
VOLATILE_TEXT = (
    (re.compile(r"\b(scan|write|run|measure|force|tune)-[0-9a-f]{6,}\b"), r"\1-<id>"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), "<uuid>"),
    (re.compile(r"(?<!\d)20\d{6}_\d{6}(?!\d)"), "<scan-stamp>"),
    # pixels fused so far when a scan is stopped mid-way: it depends on how far the scan got
    (re.compile(r"frame fused \(\d+ px\)"), "frame fused (<n> px)"),
    # age of the free-air baseline, read off the wall clock when the write starts
    (re.compile(r"\bage \d+\.\d+ s\b"), "age <n> s"),
    (re.compile(r"\b20\d\d-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?(Z|[+-]\d\d:?\d\d)?\b"), "<iso-time>"),
)
# 3. The temp runtime folder, which differs per run (replaced by a fixed name).
RUNTIME_TOKEN = "<RUNTIME>"
# 4. A list of more than 16 bare numbers (an image, a toolpath) is kept as its length,
#    first, last and a digest (sha256 over little-endian float32, 16 hex, the golden
#    tap's rule), not value by value.
LONG_NUMBER_LIST = 16
#    Exception: the pixel values of a scan (`height` in scan/last) come from the twin's
#    unseeded noise and differ from run to run, so only their length is kept.
NOISY_LIST_KEYS = frozenset({"height"})
# 5. WS frames that stream on their own clock and carry no request effect.
# (`rail_guard` is one frame per poll, so its count follows the wall time of a scan.)
SKIP_EVENT_TYPES = frozenset({"telemetry", "tip_pos", "rail_guard"})
# 6. WS frames that stream per line or per point: a run of one type is kept as its
#    count and key names (`line` also its first and last row), so a changed number of
#    frames fails. Only the two cases below drop or reduce them, each because the count
#    depends on how far a scan got when it was stopped, which no door controls:
#      - `Step.drop`: the `line` frames of a scan left running in the background; the
#        step that stops it waits for `scan_done` and records the rest.
BULK_EVENT_TYPES = frozenset({"line", "litho_progress"})


def _mask_text(s: str, runtime: str) -> str:
    if runtime:
        s = s.replace(runtime, RUNTIME_TOKEN).replace(runtime.replace("\\", "/"), RUNTIME_TOKEN)
    for pat, repl in VOLATILE_TEXT:
        s = pat.sub(repl, s)
    return s


def number_list_summary(values: list) -> dict:
    """Length, first, last and a digest of a long numeric list (None counts as NaN)."""
    f32 = struct.pack(f"<{len(values)}f", *[float("nan") if v is None else float(v) for v in values])
    return {"number_list_len": len(values), "first": values[0], "last": values[-1],
            "sha256_f32le_16": hashlib.sha256(f32).hexdigest()[:16]}


def mask(value: Any, runtime: str, extra: frozenset = frozenset()) -> Any:
    """`extra`: keys masked for this step only (see `Step.masked`)."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if (k in VOLATILE_KEYS or k in extra) and v is not None:
                out[k] = MASK
            elif k in NOISY_LIST_KEYS and isinstance(v, list):
                out[k] = {"number_list_len": len(v), "values": MASK}
            else:
                out[k] = mask(v, runtime, extra)
        return out
    if isinstance(value, list):
        if len(value) > LONG_NUMBER_LIST and all(
                isinstance(v, (int, float)) or v is None for v in value):
            return number_list_summary(value)
        return [mask(v, runtime, extra) for v in value]
    if isinstance(value, str):
        return _mask_text(value, runtime)
    return value


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


# ------------------------------------------------------------------ WS tap ----
class WsTap:
    """Collects every non-telemetry /ws frame, in arrival order."""

    def __init__(self, port: int):
        from websockets.sync.client import connect
        self._connection = connect(f"ws://127.0.0.1:{port}/ws", open_timeout=10)
        self._ws = self._connection.__enter__()
        self._lock = threading.Lock()
        self._frames: list[dict] = []
        self._arrivals: list[float] = []
        self._stop = False
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        while not self._stop:
            try:
                raw = self._ws.recv(timeout=0.5)
            except TimeoutError:
                continue
            except Exception:                       # noqa: BLE001 -- closed socket ends the tap
                return
            try:
                frame = json.loads(raw)
            except ValueError:
                frame = {"type": "<non-json>", "text": str(raw)[:80]}
            if frame.get("type") in SKIP_EVENT_TYPES:
                continue
            with self._lock:
                self._frames.append(frame)
                self._arrivals.append(time.monotonic())

    def mark(self) -> int:
        with self._lock:
            return len(self._frames)

    def since(self, mark: int, quiet_s: float, max_wait_s: float,
              until: str | None = None, ignore: tuple = ()) -> list[dict]:
        """Frames after `mark`, once none has arrived for `quiet_s` (or `max_wait_s`).
        Frames of an `ignore` type do not count as activity. With `until`, the wait is
        not over before a frame of that type has arrived."""
        start = time.monotonic()
        deadline = start + max_wait_s
        while time.monotonic() < deadline:
            with self._lock:
                recent = [t for t, f in zip(self._arrivals[mark:], self._frames[mark:], strict=True)
                          if f.get("type") not in ignore]
                idle = time.monotonic() - max([start, *recent])
                seen = until is None or any(f.get("type") == until for f in self._frames[mark:])
            if seen and idle >= quiet_s:
                break
            time.sleep(0.05)
        with self._lock:
            return list(self._frames[mark:])

    def close(self) -> None:
        self._stop = True
        try:
            self._connection.__exit__(None, None, None)
        except Exception:                           # noqa: BLE001,S110
            pass


def summarise_events(frames: list[dict], runtime: str, unordered: bool = False,
                     extra: frozenset = frozenset()) -> list[dict]:
    """Frames in order. A run of bulk frames of one type collapses to its count and key set."""
    out: list[dict] = []
    for f in frames:
        t = f.get("type")
        if t in BULK_EVENT_TYPES:
            if out and out[-1].get("bulk") == t:
                out[-1]["count"] += 1
                if "row" in f:
                    out[-1]["last_row"] = f["row"]
                continue
            entry = {"type": t, "bulk": t, "count": 1, "keys": sorted(f)}
            if "row" in f:
                entry["first_row"] = entry["last_row"] = f["row"]
            out.append(entry)
            continue
        out.append(mask(f, runtime, extra))
    if unordered:
        out.sort(key=canonical)
    return out


def type_counts(frames: list[dict]) -> list[dict]:
    """A whole run's events as each type with how many arrived (order is not kept)."""
    counts: dict[str, int] = {}
    for f in frames:
        counts[f.get("type")] = counts.get(f.get("type"), 0) + 1
    return [{"type": t, "count": counts[t]} for t in sorted(counts)]


# ------------------------------------------------------------------ the step --
@dataclass
class Step:
    """One request. `body` is JSON; `raw` + `content_type` send bytes; `query` is the
    query string. `quiet_s` is how long the WS must stay silent before the step's
    events are taken (a slower effect, such as a scan, needs a longer wait)."""
    name: str
    method: str
    path: str
    body: Any = None
    raw: bytes | None = None
    content_type: str | None = None
    query: dict | None = None
    quiet_s: float = 0.4
    max_wait_s: float = 6.0
    note: str = ""
    head_only: bool = False      # a page whose bytes change with the web files: keep status + type
    drop: tuple = ()             # event types left out: a pass streaming in the background
    unordered: bool = False      # events from racing tasks: kept as a sorted list
    types_only: bool = False     # a whole run's worth of events: each type with its count
    until: str | None = None     # wait for a frame of this type (a run's terminal event)
    masked: tuple = ()           # keys whose values are left out of this step


@dataclass
class Group:
    name: str
    flex: bool = False
    steps: list = field(default_factory=list)       # Step | callable(session) -> Step | None


class Session:
    """A live service plus its taps. Steps run in order and share the service state."""

    def __init__(self, svc: Service, tap: WsTap):
        self.svc = svc
        self.tap = tap
        self.http = httpx.Client(base_url=svc.url, timeout=60.0)
        self.runtime = str(svc.runtime)
        self.entries: list[dict] = []
        self.flex_cmd = None

    def run(self, step: Step) -> dict:
        mark = self.tap.mark()
        kw: dict[str, Any] = {}
        if step.query:
            kw["params"] = step.query
        if step.raw is not None:
            kw["content"] = step.raw
            kw["headers"] = {"Content-Type": step.content_type or "application/octet-stream"}
        elif step.body is not None:
            kw["json"] = step.body
        r = self.http.request(step.method, step.path, **kw)
        ctype = r.headers.get("content-type", "").split(";")[0]
        if ctype == "application/json":
            body: Any = r.json()
        elif step.head_only:
            body = {"text_head": r.text[:15]}
        elif ctype.startswith("text/") or ctype in ("image/svg+xml",):
            body = {"text_sha256": hashlib.sha256(r.content).hexdigest()[:16],
                    "text_len": len(r.content), "text_head": r.text[:60]}
        else:
            body = {"bytes_sha256": hashlib.sha256(r.content).hexdigest()[:16],
                    "bytes_len": len(r.content)}
        frames = [f for f in self.tap.since(mark, step.quiet_s, step.max_wait_s, step.until, step.drop)
                  if f.get("type") not in step.drop]
        req: dict[str, Any] = {"method": step.method, "path": step.path}
        if step.query:
            req["query"] = step.query
        if step.raw is not None:
            req["raw"] = {"content_type": step.content_type, "len": len(step.raw)}
        elif step.body is not None:
            req["body"] = step.body
        entry = {
            "step": step.name,
            "request": req,
            "status": r.status_code,
            "content_type": ctype,
            "response": mask(body, self.runtime, frozenset(step.masked)),
            "events": (type_counts(frames) if step.types_only
                       else summarise_events(frames, self.runtime, step.unordered,
                                             frozenset(step.masked))),
        }
        if step.note:
            entry["note"] = step.note
        self.entries.append(entry)
        return entry

    # -- FLEX side door, used only to put a script in control or to show what the
    #    script sees afterwards. The FLEX door itself is recorded by slice 2.0a. ----
    def flex_note(self, method: str, params: dict | None = None, label: str | None = None,
                  quiet_s: float = 0.3) -> dict:
        """Send one command-socket request; record request, reply and WS events."""
        import zmq
        if self.flex_cmd is None:
            self._zctx = zmq.Context()
            sock = self._zctx.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.RCVTIMEO, 60000)
            sock.setsockopt(zmq.SNDTIMEO, 15000)
            sock.connect(f"tcp://127.0.0.1:{self.svc.flex_port}")
            self.flex_cmd = sock
        mark = self.tap.mark()
        req: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "id": "1"}
        if params is not None:
            req["params"] = params
        self.flex_cmd.send_string(json.dumps(req))
        reply = json.loads(self.flex_cmd.recv_string())
        frames = self.tap.since(mark, quiet_s, 6.0)
        self.entries.append({
            "step": label or f"FLEX {method}",
            "flex_request": {"method": method, "params": mask(params, self.runtime)},
            "flex_reply": mask(reply, self.runtime),
            "events": summarise_events(frames, self.runtime),
        })
        return reply

    def close(self) -> None:
        self.http.close()
        self.tap.close()
        if self.flex_cmd is not None:
            self.flex_cmd.close(linger=0)
            self._zctx.term()


def run_group(group: Group, steps_fn) -> list[dict]:
    """Start a fresh service, run `steps_fn(session)` (which calls session.run), stop."""
    os.environ["AFM_LITHO_SCAN_SPEED"] = "8.0"       # the transcripts' scans finish inside a step
    svc = Service(flex=group.flex, twin_speed=0.0).start()
    try:
        tap = WsTap(svc.port)
        sess = Session(svc, tap)
        try:
            steps_fn(sess)
        finally:
            sess.close()
        return sess.entries
    finally:
        svc.stop()
        os.environ.pop("AFM_LITHO_SCAN_SPEED", None)


# ------------------------------------------------------------ compare / regen --
def transcript_path(name: str) -> Path:
    return TRANSCRIPT_DIR / f"{name}.json"


def check_or_regen(name: str, entries: list[dict]) -> None:
    path = transcript_path(name)
    text = canonical({"group": name, "entries": entries})
    if os.environ.get(REGEN_ENV) == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        return
    assert path.exists(), f"{path} is missing; regenerate with {REGEN_ENV}=1"
    want = json.loads(path.read_text(encoding="utf-8"))
    got = json.loads(text)
    diff = _first_diff(want["entries"], got["entries"])
    assert not diff, diff


def _first_diff(want: list[dict], got: list[dict]) -> str:
    if want == got:
        return ""
    for i in range(max(len(want), len(got))):
        w = want[i] if i < len(want) else None
        g = got[i] if i < len(got) else None
        if w != g:
            step = (w or g or {}).get("step")
            return (f"step {i} ({step}) differs\n--- recorded\n{canonical(w)}"
                    f"--- now\n{canonical(g)}")
    return "entries differ"
