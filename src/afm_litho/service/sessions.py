"""Session archive: every scan and write is recorded and recallable.

Local store: runtime/sessions.jsonl (append-only). Remote: the lab's
Postgres (llab_079 tip trajectories) through an optional adapter, so past
*real* writing sessions can be recalled onto the canvas the same way as twin
sessions. It is off unless AFM_LITHO_SESSIONS_DSN holds a libpq connection
string (e.g. "host=... dbname=... user=... connect_timeout=3"); with no DSN
it never touches the network.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


class SessionStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, kind: str, payload: dict) -> dict:
        rec = {
            "id": f"{kind}-{int(time.time() * 1000)}",
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "kind": kind,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def list(self, limit: int = 20) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.append({k: r.get(k) for k in ("id", "ts", "kind", "label")})
        return out[::-1][:limit]

    def get(self, session_id: str) -> dict | None:
        if not self.path.exists():
            return None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("id") == session_id:
                return r
        return None


class Llab079Adapter:
    """Recall real writing sessions from the lab database (when reachable).

    llab_079 columns: d000 = tip x (m), d001 = tip y (m), d002 = speed,
    d003 = tip voltage (V), v002 = path name. Rows with only x/y are
    position samples; a row carrying voltage/name closes a segment.
    """

    def __init__(self, dsn: str | None = None):
        self.dsn = (os.environ.get("AFM_LITHO_SESSIONS_DSN", "")
                    if dsn is None else dsn).strip()
        self._ok: bool | None = None

    def available(self) -> bool:
        if not self.dsn:
            return False
        if self._ok is None:
            try:
                import psycopg2
                conn = psycopg2.connect(self.dsn)
                conn.close()
                self._ok = True
            except Exception:  # noqa: BLE001 - unreachable database means unavailable
                self._ok = False
        return self._ok

    def recent_windows(self, limit: int = 10) -> list[dict]:
        import psycopg2
        conn = psycopg2.connect(self.dsn)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT date_trunc('hour', time) AS h, count(*)
                    FROM llab_079 GROUP BY h ORDER BY h DESC LIMIT %s
                """, (limit,))
                return [{"id": f"llab-{h.isoformat()}", "ts": str(h),
                         "kind": "llab_079", "label": f"{n} samples"}
                        for h, n in cur.fetchall()]
        finally:
            conn.close()

    def window_paths(self, hour_iso: str) -> list[dict]:
        import psycopg2
        conn = psycopg2.connect(self.dsn)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT time, d000, d001, d002, d003, v002 FROM llab_079
                    WHERE time >= %s::timestamptz
                      AND time < %s::timestamptz + interval '1 hour'
                    ORDER BY time ASC
                """, (hour_iso, hour_iso))
                rows = cur.fetchall()
        finally:
            conn.close()
        paths, cur_pts = [], []
        for _, x, y, speed, volt, name in rows:
            if x is not None and y is not None:
                cur_pts.append([x * 1e6, y * 1e6])
            if name:
                if cur_pts:
                    paths.append({"name": name, "voltage": volt or 0.0,
                                  "path_um": cur_pts[::-1]})
                cur_pts = []
        return paths
