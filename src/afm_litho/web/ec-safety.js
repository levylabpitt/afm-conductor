// ec-safety.js -- host-side fail-safe readout (safety kernel entry, the safe_park
// primitive). Kept MINIMAL and MODULAR per docs/COCKPIT_MODULES.md: net-new safety
// JS lives in this ES module, NOT appended to edit-cockpit.html's inline script,
// and it does no window.* bridging into the monolith. It opens its OWN read-only
// websocket listener for `safe_park` broadcasts (the backend already emits them
// from service/host/safe_park.py) and shows the last fail-safe retract -- its
// reason, which out-of-band transport fired, and whether Output.Z was VERIFIED to
// ~0 -- as a small banner. A CRITICAL park (Output.Z did NOT reach 0: the tip may
// still be engaged) and the service's fault latch show red until the latch is cleared.
//
// This is deliberately a one-purpose readout, not a bootstrap of the cockpit's
// module split (§6): the abort button rewire + this listener are the whole
// frontend surface of the fail-safe. When the cockpit is genuinely modularized,
// this folds into ec-tab-scan / an ec-status module.

import { registerBanner } from "./ec-banners.js";

const el = document.createElement("div");
el.id = "ecSafetyBanner";
el.style.display = "none";

// message text lives in its own span so appending the vendor-withdraw button (for a
// park-failed notice) does not get wiped by a textContent assignment. Its line
// breaks show once the banner is opened with "more" (edit-cockpit.css).
const msg = document.createElement("span");
el.appendChild(msg);

// The park-failed notice's "vendor withdraw" control (sec 3 step 4 / sec 4): the
// vendor full withdraw. No dedicated vendor-withdraw route exists yet (B1's slice),
// so this drives the step-4 fallback via safe_park with a distinct reason; swap to
// POST /api/withdraw?vendor=1 (or B1's route) when it lands.
const vendorBtn = document.createElement("button");
vendorBtn.textContent = "vendor withdraw";
vendorBtn.title = "Vendor full withdraw (piezo + stepper) via the out-of-band safe_park path.";
vendorBtn.addEventListener("click", async (e) => {
  e.stopPropagation();
  vendorBtn.disabled = true;
  try {
    await fetch("/api/safe_park", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reason: "operator vendor withdraw" }),
    });
  } catch (err) { /* best-effort; the notice stays up */ }
  vendorBtn.disabled = false;
});
el.appendChild(vendorBtn);
if (document.body) document.body.appendChild(el);

// While the service's fault latch (hub.fault) is set or a critical park has been
// reported, this is the FAULT banner: it shows on every open page (seeded from GET
// /api/state on each ws (re)connect, then the `fault` ws frame), and its X is the
// operator accepting the fault: it clears the latch (docs/COCKPIT_BANNERS_SPEC.md,
// Decision). A latched fault refuses Approach, Scan and Write, so a close that only
// hid it would leave the operator blocked with nothing saying why. Otherwise it is an
// EVENT notice for a non-critical park result, closed like any other.
let fault = null;          // the latched record from the service, or null
let criticalPark = null;   // the critical safe_park frame of the current latch
let lastPark = null;       // the latest non-critical safe_park frame
let clearError = null;     // why the last accept failed, shown until the next try
let autoHide = null;
const banner = registerBanner(el, {
  kind: "event", severity: "err", first: true,
  interceptClose: () => { if (!latched()) return false; acceptFault(); return true; },
  closeTitle: () => (latched() ? "accept and clear the fault" : null),
});

function latched() { return !!fault || !!criticalPark; }

// A latched fault or a critical park opens with its whole message shown;
// any other notice starts as one line with "more".
function showBanner() { banner.show(); banner.setOpen(latched()); }

function parkLine(m) {
  const z = (typeof m.z_v === "number") ? ` Output.Z=${m.z_v.toFixed(3)} V` : "";
  const via = m.transport ? ` [${m.transport}]` : "";
  const vtxt = m.verified === true ? "retracted (verified)"
    : m.verified === false ? "retract UNVERIFIED" : "retracted";
  return `safe_park: ${m.reason} — ${vtxt}${z}${via}`;
}

function render() {
  const lines = [];
  if (criticalPark) {
    const m = criticalPark;
    const z = (typeof m.z_v === "number") ? ` Output.Z=${m.z_v.toFixed(3)} V` : "";
    lines.push(`CRITICAL safe_park: ${m.reason}${z}`,
      "Output.Z did NOT reach 0 -- tip may still be engaged. OPERATOR MUST INTERVENE.");
  }
  if (fault) {
    const at = (typeof fault.at === "number")
      ? ` at ${new Date(fault.at * 1000).toLocaleTimeString()}` : "";
    const detail = fault.detail ? ` — ${fault.detail}` : "";
    lines.push(`FAULT latched: ${fault.reason}${at}${detail}`,
      "Approach, Scan and Write are refused until it is cleared (× accepts and clears it).");
  }
  if (latched()) {
    if (lastPark) lines.push(parkLine(lastPark));
    if (clearError) lines.push(`clear failed: ${clearError}`);
    // a latched fault: red outline on the fault background
    el.style.background = "var(--err-bg)";
    el.style.borderColor = "var(--err)";
    // a failed or unverified park offers the vendor full withdraw
    vendorBtn.style.display = (criticalPark || (fault && fault.critical)) ? "" : "none";
  } else if (lastPark) {
    lines.push(parkLine(lastPark));
    // a park result alone is a warning: amber
    el.style.background = "var(--warn-bg)";
    el.style.borderColor = "var(--warn)";
    vendorBtn.style.display = "none";
  }
  msg.textContent = lines.join("\n");
  // the header status box reads FAULT for exactly as long as the latch holds
  const seg = document.getElementById("faultStatus");
  if (seg) seg.style.display = latched() ? "" : "none";
}

function clearAutoHide() { if (autoHide) { clearTimeout(autoHide); autoHide = null; } }

function onPark(m) {
  clearAutoHide();
  const was = latched();
  if (m.critical) {
    criticalPark = m;
    lastPark = null;
  } else {
    lastPark = m;
  }
  render();
  if (latched() !== was) notifyLatch();
  showBanner();                       // every park frame is news: show it in full
  if (!latched()) {
    autoHide = setTimeout(() => { if (!latched()) { lastPark = null; banner.hide(); } }, 8000);
  }
}

// Whether the fault latch (or a critical park) holds, published on the page as
// <html data-fault-latched> and an "ec-latch" window event, so the render can drop
// its scanning marker. Not an export: the page loads this file as its own entry
// (with a ?v= stamp), and importing it elsewhere would run a second copy, with a
// second banner and websocket.
function notifyLatch() {
  document.documentElement.dataset.faultLatched = latched() ? "1" : "";
  window.dispatchEvent(new Event("ec-latch"));
}

function onFault(rec) {
  if (rec) {
    clearAutoHide();
    fault = rec;
    render();
    showBanner();                     // a latched fault is never hidden
    notifyLatch();
  } else {
    const was = latched();
    fault = null;
    criticalPark = null;
    clearError = null;
    if (!was) return;                  // nothing latched: leave a park notice alone
    lastPark = null;
    clearAutoHide();
    render();
    banner.hide();
    notifyLatch();
  }
}

// The X on a latched banner. The service logs the clear (and when) and broadcasts
// `fault: null`, which removes the banner from every open page; this page also drops
// it on success in case nothing was latched server-side (no change, so no frame).
// Counts `fault` frames: a reply (state read or clear) that was requested before a
// later frame is stale and must not overwrite what that frame said.
let faultFrames = 0;

async function acceptFault() {
  const seenAtStart = faultFrames;
  let why = null;
  try {
    const r = await fetch("/api/fault/clear", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm: true }),
    });
    let body = null;
    try { body = await r.json(); } catch (e) { /* non-JSON error body */ }
    if (!r.ok) why = (body && (body.error || body.detail)) || `HTTP ${r.status}`;
    else if (!body || body.cleared !== true) {
      why = (body && (body.error || body.reason)) || "the service did not clear the fault";
    }
  } catch (e) {
    why = `request failed: ${e && e.message ? e.message : e}`;
  }
  if (why) {
    clearError = String(why);
    render();
    showBanner();
    return;
  }
  // a fault latched after this clear (its frame arrived meanwhile) stays shown
  if (faultFrames === seenAtStart) onFault(null);
}

// A state read taken before a latch can land after that latch's `fault` frame; it is
// stale then, and applying it would hide a latched fault until the next reconnect.
async function seedFault() {
  const seenAtStart = faultFrames;
  try {
    const r = await fetch("/api/state");
    if (!r.ok) return;
    const st = await r.json();
    if (faultFrames !== seenAtStart) return;
    if (st && "fault" in st) onFault(st.fault || null);
  } catch (e) { /* the ws frames still drive it */ }
}

function connect() {
  let ws;
  try {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
  } catch (e) { return; }
  ws.onopen = () => { seedFault(); };
  ws.onmessage = (ev) => {
    let m;
    // the service serializes with allow_nan=True, so tolerate bare NaN/Infinity
    // tokens exactly as the monolith's connect() does.
    try { m = JSON.parse(ev.data); }
    catch (e) {
      try { m = JSON.parse(ev.data.replace(/\bNaN\b/g, "null")
        .replace(/-?\bInfinity\b/g, "null")); }
      catch (e2) { return; }
    }
    if (m && m.type === "safe_park") onPark(m);
    else if (m && m.type === "fault") { faultFrames += 1; onFault(m.fault || null); }
  };
  // reconnect on drop so an overnight run keeps reporting.
  ws.onclose = () => setTimeout(connect, 3000);
  ws.onerror = () => { try { ws.close(); } catch (e) {} };
}

connect();
