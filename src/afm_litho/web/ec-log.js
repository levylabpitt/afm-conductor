// Command log for the edit cockpit: one place every request/response and ws
// frame is recorded (the full-traffic-log requirement). It touches the DOM, but
// looks the log element up lazily -- there is no DOM access at import time -- so
// this module still imports under node, where the pure formatters (stamp /
// escapeHtml / num) are unit-tested.
//
// Moved verbatim from edit-cockpit.html's inline script: same timestamp format,
// same 3000-row cap, same autoscroll gate.

export function stamp() {
  const d = new Date();
  const p = (n, w = 2) => String(n).padStart(w, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}.${p(d.getMilliseconds(), 3)}`;
}

export function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

export function num(v) {
  if (typeof v !== "number") return String(v);
  if (v === 0) return "0";
  const a = Math.abs(v);
  return v.toFixed(a < 1 ? 4 : 3).replace(/\.?0+$/, "");
}

export function logLine(text, cls = "info") {
  const logEl = document.getElementById("log");
  const row = document.createElement("div");
  row.className = "row";
  row.innerHTML = `<span class="t">${stamp()}</span> <span class="${cls}">${escapeHtml(text)}</span>`;
  logEl.appendChild(row);
  // the folded log shows only its latest line
  const latest = document.getElementById("logLatest");
  if (latest) latest.innerHTML = row.innerHTML;
  while (logEl.childElementCount > 3000) logEl.removeChild(logEl.firstChild);
  const auto = document.getElementById("autoscroll");
  if (!auto || auto.checked) logEl.scrollTop = logEl.scrollHeight;
}

// A logged reply body is capped so one big reply cannot flood the panel.
export function capBody(text, max = 300) {
  return text.length > max ? `${text.slice(0, max)}… (${text.length} chars)` : text;
}

// The periodic GET /api/state poll is telemetry: logged only when the
// "log telemetry" box is checked (a failed poll is always logged).
export function isQuietPoll(method, path) {
  if (method !== "GET" || path !== "state") return false;
  const box = document.getElementById("logTele");
  return !(box && box.checked);
}
