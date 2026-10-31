// ec-control.js -- who holds the instrument, and the operator's way to take it back
// (docs/FLEX_INTEGRATION_PLAN.md Sec E.2; service/control.py). Kept MINIMAL and
// MODULAR per docs/COCKPIT_MODULES.md: net-new JS lives in this ES module, NOT
// appended to edit-cockpit.html's inline script, and it does no window.* bridging
// into the monolith -- the same shape as ec-safety.js, which this sits beside and
// will fold into an ec-status module with.
//
// TWO THINGS, and nothing else:
//   * a STATUS CHIP -- who the commander is, and how long the dead-man has left. It
//     is driven by the `control` websocket event (broadcast on every arbitration
//     change) and seeded once from GET /api/control, because that event only fires
//     on a CHANGE and a page loading into an already-claimed instrument would
//     otherwise start from a lie.
//   * ONE BUTTON that is Take control or Release control depending on who holds
//     the instrument. Take control (POST /api/control/take) is the plan's normal
//     preemption path: a human deliberately taking the instrument back, and SEEING
//     that they did. (Reaching for Approach / Scan / Write / Withdraw / Mode /
//     Continuous preempts too, server-side; the button is for when you want the
//     instrument WITHOUT moving the tip first.)
//
//     RELEASE CONTROL IS NOT OPTIONAL POLISH. Operator control has no dead-man by
//     design -- a human is present -- so it never expires. With only a Take control
//     button, the operator state was a DEAD END: every script got -32040 forever and
//     the only way out was someone curling /api/control/release. The button that
//     took the instrument has to be the button that gives it back.
//
// The countdown ticks locally off `expires_s` rather than polling: the dead-man is
// the server's, and a chip that re-asked every second would refresh the very timer
// it is displaying (any request bearing the token refreshes it -- but these do not
// carry one, so this is about noise, not correctness).

const CHIP_ID = "ecControlChip";

const el = document.createElement("div");
el.id = CHIP_ID;
el.style.cssText = [
  // BOTTOM-LEFT: ec-safety.js owns top-centre, and the top-right corner holds the
  // header's status box and Abort, which nothing may cover.
  "position:fixed", "left:10px", "bottom:10px", "z-index:9998",
  "display:none", "align-items:center", "gap:8px",
  "padding:5px 9px",
  "font:12px/1.3 var(--ui)",
  "background:var(--panel)", "color:var(--fg)",
  "border:1px solid var(--edge)",
].join(";");

// The chip's tone: "ok" (the operator holds control) in the accent, "warn" (a script
// holds it) in amber, "err" (dead-man down or nearly out) in red. The label's words
// say the same thing.
const TONES = {
  ok: ["var(--sel-bg)", "var(--accent)"],
  warn: ["var(--warn-bg)", "var(--warn)"],
  err: ["var(--err-bg)", "var(--err)"],
};
function setTone(tone) {
  const [bg, edge] = TONES[tone];
  el.style.background = bg;
  el.style.borderColor = edge;
}

const label = document.createElement("span");
const btn = document.createElement("button");
btn.type = "button";
btn.textContent = "Take control";
btn.style.cssText = [
  "font:12px/1 var(--ui)", "padding:4px 8px",
  "border:1px solid var(--edge)", "background:var(--panel)", "color:var(--fg)",
  "cursor:pointer",
].join(";");
el.appendChild(label);
el.appendChild(btn);
if (document.body) document.body.appendChild(el);

// The last state the server told us, plus when we heard it -- the countdown is
// derived from those two, so a stalled websocket shows the timer running out
// instead of freezing on a stale number.
let state = null;
let heardAt = 0;

function remaining() {
  if (!state || typeof state.expires_s !== "number") return null;
  return Math.max(0, state.expires_s - (Date.now() - heardAt) / 1000);
}

function render() {
  if (!state || !state.holder) {
    // Unclaimed: no chip at all. A GUI-only session is the common case and must not
    // grow a permanent badge telling it nothing.
    el.style.display = "none";
    return;
  }
  const operator = state.kind === "operator";
  const left = remaining();
  const clock = (left === null) ? "" : ` · ${left.toFixed(0)}s`;
  const armed = (state.deadman_armed !== false);
  label.textContent = operator
    ? "You have control (operator)"
    : `Script in control: ${state.holder}${clock}`;
  // The dead-man is what makes it safe to leave a script driving, so a watcher that
  // is down is operator-visible, not an internal detail.
  if (!armed) label.textContent += " · DEAD-MAN DOWN";
  setTone(operator ? "ok" : "warn");
  if (!armed) setTone("err");
  // ONE button, both directions: operator control never expires, so without a
  // Release the operator state is a dead end for every script.
  btn.style.display = "";
  btn.textContent = operator ? "Release control" : "Take control";
  btn.title = operator
    ? "Hand the instrument back, so a script can acquire it again."
    : "Revoke this script's token and take the instrument (operator control has " +
      "no dead-man).";
  // The dead-man is nearly out: say so in the colour, so an operator watching the
  // panel sees the takeover coming rather than discovering it in the log.
  if (!operator && left !== null && left <= 5) setTone("err");
  el.style.display = "inline-flex";
}

function apply(c) {
  state = c || null;
  heardAt = Date.now();
  render();
}

btn.addEventListener("click", async () => {
  const releasing = !!(state && state.kind === "operator");
  btn.disabled = true;
  try {
    const r = await fetch(releasing ? "/api/control/release" : "/api/control/take", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(releasing ? {}
        : { client: "operator", reason: "Take control button" }),
    });
    const j = await r.json();
    if (j && j.control) apply(j.control);
  } catch (e) {
    /* the websocket event will correct us; a failed click must not wedge the chip */
  } finally {
    btn.disabled = false;
  }
});

// seed once: `control` only fires on a CHANGE.
fetch("/api/control").then((r) => r.json()).then((j) => {
  if (j && j.control) apply({ ...j.control, deadman_armed: j.deadman_armed });
}).catch(() => {});

// tick the countdown locally (see the header note on why this does not poll).
setInterval(() => { if (state && state.holder && state.kind !== "operator") render(); },
  1000);

function connect() {
  let ws;
  try {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
  } catch (e) { return; }
  ws.onmessage = (ev) => {
    let m;
    // the service serializes with allow_nan=True, so tolerate bare NaN/Infinity
    // tokens exactly as the monolith's connect() does.
    try { m = JSON.parse(ev.data); }
    catch (e) {
      try {
        m = JSON.parse(ev.data.replace(/\bNaN\b/g, "null")
          .replace(/-?\bInfinity\b/g, "null"));
      } catch (e2) { return; }
    }
    if (m && m.type === "control") apply(m);
  };
  // reconnect on drop so an overnight run keeps reporting.
  ws.onclose = () => setTimeout(connect, 3000);
  ws.onerror = () => { try { ws.close(); } catch (e) {} };
}

connect();
