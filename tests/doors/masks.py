"""The one reviewed list of values a FLEX transcript masks, and the function that applies it.

A transcript is compared byte for byte, so anything that differs between two runs of the same
scenario on the same code is masked here, and nowhere else. Every entry says why. Nothing that
carries a result of the door (a code, a message, a state, a refusal reason, a shape) is masked.
"""

from __future__ import annotations

import json
import re

# Keys whose value is a clock reading or a time left on a clock. The value is replaced by
# "<TIME>"; the key stays, so a reply that loses the field still differs.
TIME_KEYS = frozenset({
    "t",             # getTelemetry: wall-clock stamp of the cached frame
    "age_s",         # getTelemetry: how old the cached frame is
    "ts",            # getLastPark: when the park happened
    "expires_s",     # acquireControl/heartbeat/refusals: dead-man seconds left
    "closed_at",     # getStatus.bridge: when the bridge closed (null until it does)
    "t_start",       # getWritten: stroke start, seconds since the write began
    "t_end",         # getWritten: stroke end
    "started", "finished", "created", "elapsed_s", "duration_wall_s",
})

# Keys whose value is a live reading from the twin's simulated tip; the twin moves it with
# wall time. Masked to "<LIVE>" (null stays null: whether a channel exists is a result).
LIVE_KEYS = frozenset({
    "x_um", "y_um", "zdrive_v", "defl_v", "sum_v", "amp_v", "row",   # getTelemetry only
    "progress",      # getStatus: fraction of the pass done at the instant of the poll
    "last_row",      # getStatus.rail: the raster row the ~10 Hz rail poll last saw
})

# getCapabilities.ports: the ephemeral FLEX port pair of this run, as numbers.
PORT_PARENT_KEYS = frozenset({"ports"})

# Keys that name an id the service mints from a counter or the clock. Replaced by an ordinal
# "<RUN:n>" in order of first appearance in the transcript, so two different runs stay
# different and the same run stays the same.
RUN_KEYS = frozenset({"run_id"})

# Keys that hold a secret the service mints per run (the control token): "<TOKEN:n>".
TOKEN_KEYS = frozenset({"token"})

# Machine-local text: the runtime folder, the ephemeral FLEX ports. Replaced inside any string.
def _text_masks(runtime: str, ports: tuple[int, ...]) -> list[tuple[re.Pattern[str], str]]:
    out = []
    for form in {runtime, runtime.replace("\\", "/"), runtime.replace("\\", "\\\\")}:
        if form:
            out.append((re.compile(re.escape(form), re.IGNORECASE), "<RUNTIME>"))
    # The save time the service puts in a scan's file names (YYYYMMDD_HHMMSS), and the _NN a second save in the same second gets.
    out.append((re.compile(r"\d{8}_\d{6}(?:_\d{2})?"), "<STAMP>"))
    # A saved record's id is its kind and the wall clock in milliseconds.
    out.append((re.compile(r"(?:scan|litho|write)-\d{10,}"), "<RECORD>"))
    # A stored run's execution id (<w|s>-YYYY-MM-DD-HHMM-NN, the start minute in UTC); it also
    # names the run's file in "path".
    out.append((re.compile(r"[ws]-\d{4}-\d{2}-\d{2}-\d{4}-\d{2}"), "<EXEC>"))
    for p in ports:
        out.append((re.compile(rf"(?<![0-9]){p}(?![0-9])"), "<PORT>"))
    return out


class Masker:
    """Applies the lists above to one transcript, keeping its ordinals across the whole run."""

    def __init__(self, runtime: str, ports: tuple[int, ...]):
        self._texts = _text_masks(runtime, ports)
        self._ports = set(ports)
        self._runs: dict[str, str] = {}
        self._tokens: dict[str, str] = {}

    def _ordinal(self, table: dict[str, str], tag: str, value: str) -> str:
        if value not in table:
            table[value] = f"<{tag}:{len(table) + 1}>"
        return table[value]

    def learn(self, node) -> None:
        """Register the run ids and tokens in a reply before it is masked, so a later request
        or a message that quotes one is masked the same way."""
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, str) and k in RUN_KEYS:
                    self._ordinal(self._runs, "RUN", v)
                elif isinstance(v, str) and k in TOKEN_KEYS:
                    self._ordinal(self._tokens, "TOKEN", v)
                else:
                    self.learn(v)
        elif isinstance(node, list):
            for v in node:
                self.learn(v)

    def _string(self, s: str) -> str:
        for value, mask in list(self._tokens.items()) + list(self._runs.items()):
            s = s.replace(value, mask)
        for rx, mask in self._texts:
            s = rx.sub(mask, s)
        return s

    def apply(self, node):
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if k in PORT_PARENT_KEYS and isinstance(v, dict):
                    out[k] = {pk: "<PORT>" if pv in self._ports else self.apply(pv)
                              for pk, pv in v.items()}
                elif k in TIME_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[k] = "<TIME>"
                elif k in LIVE_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[k] = "<LIVE>"
                else:
                    out[k] = self.apply(v)
            return out
        if isinstance(node, list):
            return [self.apply(v) for v in node]
        if isinstance(node, str):
            return self._string(node)
        return node


def canonical(obj) -> str:
    """The fixture text: sorted keys, two-space indent, one trailing newline, ASCII."""
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
