// The cockpit's network + websocket surface: ONE request helper (uniform tx/rx
// logging, an r.ok check, a uniform result shape), the endpoint wrappers built
// on it, and the /ws connection (reconnect + NaN-safe parse + a handler registry
// so modules register per message type instead of growing a central switch).
//
// DOM-free at import time -- everything that touches fetch / WebSocket / location
// does so inside a function body -- so this module imports under node, where the
// pure `parseWsData` is unit-tested. All logging goes through ec-log.

import { logLine, capBody, isQuietPoll } from "./ec-log.js";

// ---- api(): POST JSON to /api/<path> -------------------------------------
// Moved verbatim from the inline script. Logs both directions so a call that
// returns {error:...} or a non-2xx is visible in the command log rather than
// silent. Returns the parsed body (or undefined on a network error).
// A single REFUSAL SINK (docs/INSTRUMENT_MODEL_SPEC.md sec 4; B3 review M2). The
// command guard refuses synchronously with {error, reason}; logging it to the
// command panel is not enough (a 30 V write at gain 1 was refused invisibly). Every
// response that carries a stable `reason` is offered to a registered hook, which
// surfaces the guard reasons as a visible docked notice. Kept HERE so both the
// api() and request() doors feed the one sink -- a caller cannot forget it.
let refusalHook = null;
export function onRefusal(fn) { refusalHook = fn; }
function _offerRefusal(body) {
  try { if (refusalHook && body && (body.reason || body.error)) refusalHook(body); }
  catch (e) { /* a notice must never break the call that triggered it */ }
}

export async function api(path, body){
  logLine(`→ POST /api/${path}${body?" "+JSON.stringify(body):""}`,"tx");
  try{const r=await fetch(`/api/${path}`,{method:"POST",headers:{"Content-Type":"application/json"},body:body?JSON.stringify(body):undefined});
      const d=await r.json().catch(()=>({})); logLine(`← ${r.status} ${capBody(JSON.stringify(d))}`, r.ok?"rx":"warn"); _offerRefusal(d); return d;
  }catch(e){logLine(`← ${path} failed: ${e.message}`,"warn");}
}

// ---- latestOnly(): one run in flight and at most one more waiting.
// For a refresh whose service side can take seconds (the write estimate, the
// toolpath compile) and that is asked for again on every design change. A call made
// while a run is in flight starts no request of its own: it asks for ONE more run
// after the current one, and every caller gets the promise that settles when the
// last run has. So a burst of edits leaves two requests, not one per edit, and the
// browser's few connections per host stay free for Abort and Withdraw.
export function latestOnly(fn) {
  let running = null, again = false;
  const run = async () => { do { again = false; await fn(); } while (again); };
  return () => {
    if (running) { again = true; return running; }
    running = run().finally(() => { running = null; });
    return running;
  };
}

// ---- request(): the generalized helper every hand-rolled fetch migrates onto.
// Uniform tx/rx traffic logging, an r.ok check (a 404 logs as an error, not
// "200"), and a uniform { ok, status, data } result. `body` is sent as JSON;
// `raw` + `contentType` send bytes as-is (the SVG import); `query` builds the
// query string. On a network error: { ok:false, status:0, data:{}, error }.
export async function request(method, path, opts = {}) {
  const { body, raw, contentType, query } = opts;
  const qs = query ? "?" + new URLSearchParams(query).toString() : "";
  const tail = body !== undefined ? " " + JSON.stringify(body) : "";
  const quiet = isQuietPoll(method, path);
  if (!quiet) logLine(`→ ${method} /api/${path}${qs}${tail}`, "tx");
  try {
    const init = { method };
    if (raw !== undefined) {
      init.headers = { "Content-Type": contentType || "application/octet-stream" };
      init.body = raw;
    } else if (body !== undefined) {
      init.headers = { "Content-Type": "application/json" };
      init.body = JSON.stringify(body);
    }
    const r = await fetch(`/api/${path}${qs}`, init);
    const data = await r.json().catch(() => ({}));
    if (!quiet || !r.ok) {
      if (quiet) logLine(`→ ${method} /api/${path}${qs}`, "tx");
      logLine(`← ${r.status} ${capBody(JSON.stringify(data))}`, r.ok ? "rx" : "warn");
    }
    _offerRefusal(data);
    return { ok: r.ok, status: r.status, data };
  } catch (e) {
    logLine(`← ${method} /api/${path} failed: ${e.message}`, "warn");
    return { ok: false, status: 0, data: {}, error: e.message };
  }
}

// ---- named endpoint wrappers ---------------------------------------------
// Thin names over request() for every endpoint the cockpit calls, so a call
// site reads as `getDesign()` not a fetch literal, and each returns the uniform
// { ok, status, data }. The semantic response handling (adoptDesign, state
// updates, UI) stays at the call site; these only carry the traffic + r.ok.
export const getDesign = () => request("GET", "design");
export const getDesignBezier = (i) => request("GET", `design/bezier/${i}`);
export const postDesignSvg = (name, file) =>
  request("POST", "design/svg", { query: { name }, raw: file, contentType: "image/svg+xml" });
export const postDesignAdd = (body) => request("POST", "design/add", { body });
export const postDesignOp = (body) => request("POST", "design/op", { body });
export const postDesignReorder = (order) => request("POST", "design/reorder", { body: { order } });
export const postDesignChiral = (body) => request("POST", "design/chiral", { body });
export const postDesignBezier = (body) => request("POST", "design/bezier", { body });
export const postDesignPoints = (body) => request("POST", "design/points", { body });
export const getDesignFiles = () => request("GET", "design/files");
export const postDesignSave = (name) => request("POST", "design/save", { body: { name } });
export const postDesignOpen = (name, mode) =>
  request("POST", "design/open", { body: { name, mode } });
export const postLithoToolpath = () => request("POST", "litho/toolpath");
export const getScanLast = () => request("GET", "scan/last");
export const getScanContinuous = () => request("GET", "scan/continuous");
export const getMode = () => request("GET", "mode");
// The service's live state snapshot (backend, engaged, scanning, writing, ...).
export const getState = () => request("GET", "state");

// ---- Instrument model (docs/INSTRUMENT_MODEL_SPEC.md sec 1/4/10) ----------
// The service-owned model + its last-used write/scan params + the override tail.
// The cockpit shows every model field read-only; open_config/reload act ON THE
// SERVICE HOST. `GET /api/litho/estimate` is the service's own total-seconds
// estimate over ITS enabled set (the service side of estimate + last_write_params
// is owned by another slice; the cockpit tolerates a 404/absent value until it
// lands).
export const getInstrument = () => request("GET", "instrument");
export const postInstrumentOverride = (body) => request("POST", "instrument/override", { body });
export const postInstrumentOpenConfig = () => request("POST", "instrument/open_config", { body: {} });
export const postInstrumentReload = () => request("POST", "instrument/reload", { body: {} });
export const getLithoEstimate = () => request("GET", "litho/estimate");
export const getPorts = () => request("GET", "connect/ports");
export const postPortAdd = (body) => request("POST", "connect/ports/add", { body });
export const postPortUpdate = (body) => request("POST", "connect/ports/update", { body });
export const postPortDelete = (body) => request("POST", "connect/ports/delete", { body });
export const getDrift = () => request("GET", "drift");
export const postDriftEnable = (enabled) => request("POST", "drift/enable", { body: { enabled } });
export const postDriftReference = () => request("POST", "drift/reference", { body: {} });

// ---- Design background swap (deprecated) ------------------------------------
// Copies the loaded design SVG with its background <image> swapped for the
// newest scan PNG of `channel`, at the scan's physical size. Its Scan-tab controls
// are hidden; these exports stay until the feature is removed.
export const getDesignBackgroundChannels = () => request("GET", "design/background/channels");
export const postDesignBackground = (channel) =>
  request("POST", "design/background", { body: { channel } });


// ---- websocket: reconnect + NaN-safe parse + handler registry ------------

// The service serializes with Python json (allow_nan=True), so empty/absent
// capture arrives as bare NaN / Infinity tokens, not valid JSON. Try a plain
// parse first, then a retry with those tokens nulled before giving up. Pure, so
// it is unit-tested under node. Returns { ok:true, value } or { ok:false }.
export function parseWsData(data) {
  try { return { ok: true, value: JSON.parse(data) }; }
  catch {
    try {
      return { ok: true, value: JSON.parse(String(data)
        .replace(/\bNaN\b/g, "null")
        .replace(/-?\bInfinity\b/g, "null")) };
    } catch { return { ok: false }; }
  }
}

const wsHandlers = new Map();   // m.type -> handler
let wsDefault = null;           // fallback for a type with no registered handler
// Register the handler for one ws message type. Modules register instead of
// growing a central switch; exactly one handler per type (last registration
// wins). Adding a feature is an additive onWs() call, not a switch edit.
export function onWs(type, fn) { wsHandlers.set(type, fn); }
// Register the fallback for any frame whose type has no handler.
export function onWsDefault(fn) { wsDefault = fn; }
// Route a parsed frame to its registered handler, or the default.
export function dispatchWs(m) {
  const h = wsHandlers.get(m.type);
  if (h) h(m);
  else if (wsDefault) wsDefault(m);
}

let ws = null, reconnectTimer = null;
// Open (and keep reopening) the /ws feed. The caller supplies lifecycle hooks
// (onConnecting / onOpen / onClose / onError), onFrame -- fired for EVERY frame
// received BEFORE the parse, so a frame counter counts unparseable frames too --
// and onParseFail. Parsed frames go to the registry. Reconnects itself 1s after
// a close, exactly as the original connect() did.
export function connectWs(handlers = {}) {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  handlers.onConnecting?.();
  ws = new WebSocket(`${scheme}//${location.host}/ws`);
  ws.onopen = () => handlers.onOpen?.();
  ws.onclose = () => {
    handlers.onClose?.();
    clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(() => connectWs(handlers), 1000);
  };
  ws.onerror = () => handlers.onError?.();
  ws.onmessage = (ev) => {
    handlers.onFrame?.(ev.data);
    const parsed = parseWsData(ev.data);
    if (!parsed.ok) { handlers.onParseFail?.(ev.data); return; }
    dispatchWs(parsed.value);
  };
}
