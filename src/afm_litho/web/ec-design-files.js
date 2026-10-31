// ec-design-files.js -- the Save design / Open design dialog and the notice an open
// leaves behind. The service owns the files (POST /api/design/save, /open, GET
// /api/design/files); nothing here refuses or confirms, a refusal is shown with the
// service's reason.
//
// No DOM access at import time, so the pure helpers below are tested under node.

import { getDesignFiles, postDesignSave, postDesignOpen } from "./ec-api.js";
import { logLine } from "./ec-log.js";
import { registerBanner } from "./ec-banners.js";
import { state, subscribe } from "./ec-state.js";

const SUFFIX = ".design.json";
const LAST_NAME_KEY = "ec.designFiles.lastName";
const FRAME_TOL = 1e-9;
const MAX_WIDTH_PX = 640;
const MARGIN_PX = 12;
const ROOMY_PX = 120;
// Above the service's 20 s bound on building an opened file.
const CALL_TIMEOUT_MS = 60000;
const REFETCH_TRIES = 3;
const REFETCH_GAP_MS = 2000;

export const PLACEHOLDER_TEXT =
  "the objects on screen are placeholders, not a design: draw, import or open one, then save";

// The name the service files `raw` under (design_store.file_name), so the dialog can
// tell that a typed name is already taken.
export function designFileName(raw) {
  let s = String(raw == null ? "" : raw).trim();
  if (s.toLowerCase().endsWith(SUFFIX)) s = s.slice(0, -SUFFIX.length);
  if (!s.replace(/^[. ]+|[. ]+$/g, "")) return "";
  return s.replace(/[^\p{L}\p{N}\-._]/gu, "_");
}

export function nameExists(raw, designs) {
  const name = designFileName(raw).toLowerCase();
  return !!name && (designs || []).some((d) => String(d.name).toLowerCase() === name);
}

const count = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

// A value from a reply as text, whatever its type turned out to be.
export function shown(v) {
  if (v === null || v === undefined) return "none";
  if (typeof v === "string") return v;
  if (typeof v !== "object") return String(v);
  const text = JSON.stringify(v);
  return text.length > 80 ? text.slice(0, 80) + "…" : text;
}

function frameDiffs(saved, now) {
  const out = [];
  for (const part of ["scan", "litho"]) {
    const a = saved ? saved[part] : null;
    if (!a || typeof a !== "object") continue;
    const b = now ? now[part] : null;
    if (!b || typeof b !== "object") { out.push(`${part}: none now`); continue; }
    for (const key of Object.keys(a)) {
      const same = typeof a[key] === "number" && typeof b[key] === "number"
        ? Math.abs(a[key] - b[key]) <= FRAME_TOL : a[key] === b[key];
      if (!same) out.push(`${part} ${key} ${shown(a[key])} → ${shown(b[key])}`);
    }
  }
  return out;
}

export function openNoticeLines(reply) {
  const n = reply.object_count;
  const lines = [reply.mode === "replace"
    ? `${reply.name} opened: ${count(n, "object")} in place of the earlier design`
    : `${reply.name} opened: ${count(n, "object")} added`
      + (reply.total_objects != null ? ` (${reply.total_objects} in the design now)` : "")];
  if (reply.mode === "replace" && reply.previous_saved === true)
    lines.push("the earlier design is kept as _previous");
  if (reply.mode === "replace" && reply.previous_saved === false)
    lines.push("nothing was kept as _previous: the design was empty");
  for (const o of reply.objects || [])
    if (o.renamed) lines.push(`renamed: ${o.saved_name} → ${o.name}`);
  if (reply.frame_differs) {
    const diffs = frameDiffs(reply.frame_saved, reply.frame_now);
    lines.push("saved in a different frame" + (diffs.length ? ` (${diffs.join("; ")})` : "")
      + "; coordinates are kept as saved");
  }
  for (const m of reply.missing_ports || [])
    lines.push(m.port_id == null
      ? `${m.name} is bound to an electrode port that this sample does not have`
      : `${m.name} is bound to electrode port ${shown(m.port_id)}, which this sample does not have`);
  for (const o of reply.left_out || [])
    lines.push(`left out: ${o.name == null ? "an object" : shown(o.name)}: ${shown(o.reason)}`);
  return lines;
}

export function saveNoticeLines(reply) {
  return [`design saved as ${reply.name} (${count(reply.object_count, "object")})`
    + (reply.overwrote ? ", replacing the earlier file" : "")];
}

// A listed file whose header gave no count or time has those as null. `recovered`
// marks an autosave the service set aside.
export function listRowMeta(d) {
  const parts = [];
  if (d.recovered === true) parts.push("recovered autosave copy");
  if (isCount(d.object_count)) parts.push(count(d.object_count, "object"));
  if (isText(d.saved_at)) parts.push(d.saved_at.replace("T", " "));
  return parts.length ? parts.join(" · ") : "no details in the file";
}

// The rows of a listing that can be shown: an object with a name.
export function listRows(designs) {
  return (designs || []).filter((d) => d && typeof d === "object" && isText(d.name));
}

// What the list area says in place of rows; "" when there are rows to show.
export function listStateText(listState, error, n) {
  if (listState === "loading") return "loading the saved designs…";
  if (listState === "failed") return `the saved designs could not be listed: ${error}`;
  return n ? "" : "no saved designs yet";
}

// What a save or open that did not succeed says. `placeholders`: the page shows only
// its placeholder objects.
export function failureText(res, what, placeholders) {
  const d = res.data && typeof res.data === "object" ? res.data : {};
  if (d.error)
    return d.reason === "no_design" && placeholders ? PLACEHOLDER_TEXT : shown(d.error);
  if (!res.status) return `${what} design failed: the service did not answer`;
  if (!res.ok) return `${what} design failed (HTTP ${res.status})`;
  return `${what} design failed: the service's reply was not understood `
    + `(HTTP ${res.status}): ${JSON.stringify(d).slice(0, 120)}`;
}

function isCount(n) { return Number.isInteger(n) && n >= 0; }
function isText(t) { return typeof t === "string" && t !== ""; }
const absentOr = (v, test) => v === undefined || v === null || test(v);

// A reply counts as a success only with `ok: true`, no `error`, and every field its
// notice prints present and of the right type.
export function succeeded(res, what) {
  const d = res.data;
  if (!res.ok || !d || typeof d !== "object" || d.ok !== true || d.error) return false;
  if (!isText(d.name) || !isCount(d.object_count)) return false;
  if (what === "save") return absentOr(d.overwrote, (v) => typeof v === "boolean");
  return (d.mode === "add" || d.mode === "replace")
    && absentOr(d.total_objects, isCount)
    && Array.isArray(d.objects)
    && d.objects.every((o) => o && isText(o.saved_name) && isText(o.name)
      && typeof o.renamed === "boolean")
    && absentOr(d.missing_ports, (m) => Array.isArray(m) && m.every((e) => e && isText(e.name)))
    && absentOr(d.left_out, (m) => Array.isArray(m) && m.every((e) => e && typeof e === "object"));
}

// Settles with the call's result, or after `ms` as a call the service did not answer.
export function withTimeout(promise, ms) {
  let timer;
  const late = new Promise((resolve) => {
    timer = setTimeout(() => resolve(
      { ok: false, status: 0, data: {}, timedOut: Math.round(ms / 1000) }), ms);
  });
  return Promise.race([promise, late]).finally(() => clearTimeout(timer));
}

// --------------------------------------------------------------- dialog --

let deps = {};
let notice = null;
let dialogMode = "save";
let designs = [];
let listState = "loading";
let listError = "";
let selectedName = null;
let lastName = "";
let listSeq = 0;
let dialogSeq = 0;        // changes each time the dialog is opened or closed
let inFlight = null;      // { what, name } while a save or open is on its way
let refusalMark = null;
let awaiting = null;      // { name, tries } after an open, until its design is on screen
let staleMark = null;
let retryTimer = null;

const $ = (id) => document.getElementById(id);
const isOpen = () => $("designDialog").style.display !== "none";

function recallName() {
  try { return localStorage.getItem(LAST_NAME_KEY) || ""; } catch { return ""; }
}
function rememberName(name) {
  lastName = name;
  try { localStorage.setItem(LAST_NAME_KEY, name); } catch { /* kept for this page only */ }
}

function setMsg(text, progress = false) {
  const el = $("designDialogMsg");
  el.textContent = text || "";
  el.classList.toggle("progress", progress);
  el.style.display = text ? "" : "none";
  placeDialog();
}

function showNotice(lines) {
  const box = $("designNoticeText");
  box.replaceChildren(...lines.map((line) => {
    const div = document.createElement("div");
    div.textContent = line;
    return div;
  }));
  notice.show();
  for (const line of lines) logLine(line, "info");
}

// The panel is kept inside the part of the scan image area that is on screen, at any
// window size: it never reaches the header (Abort) or the tab panels (Withdraw, Stop,
// Approach, frame up/down). Its CSS makes the middle scroll when this box is small.
function placeDialog() {
  const dlg = $("designDialog");
  const panel = document.querySelector(".scanpanel");
  if (!panel || !isOpen()) return;
  const r = panel.getBoundingClientRect();
  const left = Math.max(r.left, 0);
  const top = Math.max(r.top, 0);
  const availW = Math.max(0, Math.min(r.right, window.innerWidth) - left);
  const availH = Math.max(0, Math.min(r.bottom, window.innerHeight) - top);
  const mx = availW > 2 * ROOMY_PX ? MARGIN_PX : 0;
  const my = availH > 2 * ROOMY_PX ? MARGIN_PX : 0;
  const w = Math.min(MAX_WIDTH_PX, availW - 2 * mx);
  dlg.style.width = `${w}px`;
  dlg.style.maxHeight = `${availH - 2 * my}px`;
  dlg.style.left = `${left + (availW - w) / 2}px`;
  dlg.style.top = `${top + Math.max(my, (availH - dlg.offsetHeight) / 2)}px`;
}

function renderExists() {
  const el = $("designSaveExists");
  const typed = $("designSaveName").value;
  let text = "";
  if (listState === "failed")
    text = `the saved designs could not be listed (${listError}), `
      + "so it is not known whether this name is already saved";
  else if (nameExists(typed, designs))
    text = `${designFileName(typed)} is already saved; saving replaces it`;
  el.textContent = text;
  el.dataset.state = listState;
  el.style.display = text ? "" : "none";
  placeDialog();
}

function markSelection() {
  const list = $("designFileList");
  let active = "";
  for (const row of list.children) {
    const sel = row.dataset.name === selectedName;
    row.classList.toggle("sel", sel);
    row.setAttribute("aria-selected", sel ? "true" : "false");
    if (sel) { active = row.id; row.scrollIntoView({ block: "nearest" }); }
  }
  if (active) list.setAttribute("aria-activedescendant", active);
  else list.removeAttribute("aria-activedescendant");
}

function renderList() {
  const list = $("designFileList");
  if (!designs.some((d) => d.name === selectedName))
    selectedName = designs.length ? designs[0].name : null;
  list.replaceChildren(...designs.map((d, i) => {
    const row = document.createElement("div");
    row.className = "designfile";
    row.id = `designFile-${i}`;
    row.dataset.name = d.name;
    row.setAttribute("role", "option");
    const name = document.createElement("span");
    name.className = "dfname";
    name.textContent = d.name;
    const meta = document.createElement("span");
    meta.className = "dfmeta";
    meta.textContent = listRowMeta(d);
    row.append(name, meta);
    row.onclick = () => { selectedName = d.name; markSelection(); };
    row.ondblclick = () => { selectedName = d.name; markSelection(); openSelected(); };
    return row;
  }));
  list.style.display = designs.length ? "" : "none";
  const text = listStateText(listState, listError, designs.length);
  const stateEl = $("designListState");
  stateEl.textContent = text;
  stateEl.dataset.state = listState;
  stateEl.style.display = text ? "" : "none";
  markSelection();
  placeDialog();
}

function moveSelection(step) {
  if (!designs.length) return;
  const i = designs.findIndex((d) => d.name === selectedName);
  const j = Math.max(0, Math.min(designs.length - 1, (i < 0 ? 0 : i) + step));
  selectedName = designs[j].name;
  markSelection();
}

async function loadList() {
  const seq = ++listSeq;
  listState = "loading";
  designs = [];
  if (dialogMode === "open") renderList();
  else renderExists();
  let res;
  try { res = await getDesignFiles(); }
  catch (e) { res = { ok: false, status: 0, data: {} }; }
  if (seq !== listSeq) return;
  const d = res.data && typeof res.data === "object" ? res.data : {};
  if (res.ok && Array.isArray(d.designs)) {
    designs = listRows(d.designs);
    listState = "ok";
  } else {
    listState = "failed";
    listError = d.error ? shown(d.error)
      : !res.status ? "the service did not answer"
        : res.ok ? "the reply was not understood" : `HTTP ${res.status}`;
  }
  if (dialogMode === "open") {
    renderList();
    if (designs.length && document.activeElement === $("designDialog"))
      $("designFileList").focus();
  } else {
    renderExists();
  }
}

function openDialog(mode) {
  dialogSeq++;
  dialogMode = mode;
  const saving = mode === "save";
  $("designDialogTitle").textContent = saving ? "save design" : "open design";
  $("designSaveSection").style.display = saving ? "" : "none";
  $("designSaveConfirm").style.display = saving ? "" : "none";
  $("designOpenSection").style.display = saving ? "none" : "";
  $("designOpenConfirm").style.display = saving ? "none" : "";
  if (saving) $("designSaveName").value = lastName;
  else $("designOpenModeAdd").checked = true;
  $("designDialog").style.display = "";
  setMsg("");
  loadList();
  if (saving) {
    $("designSaveName").focus();
    $("designSaveName").select();
  } else {
    $("designDialog").focus();      // the list is hidden until it has rows
  }
}

function closeDialog() {
  dialogSeq++;
  setMsg("");
  $("designDialog").style.display = "none";
}

// One save or open at a time. The guard is held until the service's reply to that
// call arrives, or for CALL_TIMEOUT_MS. A reply that comes back after the operator
// has closed the dialog, or reopened it for something else, goes to the notices and
// leaves the dialog on screen alone.
async function call(what, name, send, onOk) {
  if (inFlight) {
    const earlier = inFlight.what === "open"
      ? `an open of ${inFlight.name}` : `a save as ${inFlight.name}`;
    setMsg(`${earlier} is still in progress; try again when it has finished`, true);
    return;
  }
  const mine = dialogSeq;
  inFlight = { what, name };
  setMsg(what === "open" ? `opening ${name}…` : `saving ${name}…`, true);
  const sent = Promise.resolve().then(send).catch(() => ({ ok: false, status: 0, data: {} }));
  const res = await withTimeout(sent, CALL_TIMEOUT_MS);
  inFlight = null;
  if (!res.timedOut) { finish(res, what, onOk, mine === dialogSeq); return; }

  // No answer yet is not a failure: the call may still complete, and its real
  // result replaces this text when it does.
  const text = `${what === "open" ? "opening" : "saving"} ${name}: no answer after `
    + `${res.timedOut} s; it may still complete`;
  deps.showFailure(text);
  const mark = deps.guardNoticeMark();
  if (mine === dialogSeq) setMsg(text);
  else clearProgress();
  const real = await sent;
  deps.clearGuardNotice(mark);
  if (isOpen() && $("designDialogMsg").textContent === text) setMsg("");
  finish(real, what, onOk, false);
}

function clearProgress() {
  if (!inFlight && isOpen() && $("designDialogMsg").classList.contains("progress")) setMsg("");
}

// `here`: the dialog the call was made from is still the one on screen.
function finish(res, what, onOk, here) {
  if (!here) clearProgress();
  if (succeeded(res, what)) {
    if (refusalMark !== null) deps.clearGuardNotice(refusalMark);
    refusalMark = null;
    if (here) closeDialog();
    else if (isOpen()) loadList();
    onOk(res.data);
    return;
  }
  const text = failureText(res, what, !state.designImported);
  if (res.data && res.data.error) deps.showRefusal(res.data, text);
  else deps.showFailure(text);
  refusalMark = deps.guardNoticeMark();
  if (here) setMsg(text);
}

function saveTyped() {
  const typed = $("designSaveName").value;
  return call("save", typed.trim() || "(no name)", () => postDesignSave(typed), (d) => {
    rememberName(d.name);
    showNotice(saveNoticeLines(d));
  });
}

function openSelected() {
  if (!selectedName) {
    setMsg(listState === "ok" && designs.length
      ? "no saved design is selected"
      : `no saved design is selected: ${listStateText(listState, listError, 0)}`);
    return undefined;
  }
  const name = selectedName;
  const mode = $("designOpenModeReplace").checked ? "replace" : "add";
  return call("open", name, () => postDesignOpen(name, mode), (d) => {
    if (!d.name.startsWith("_")) rememberName(d.name);
    showNotice(openNoticeLines(d));
    stopRetrying();
    awaiting = { name: d.name, tries: 0 };
    deps.refetchDesign();
  });
}

// ec-main emits "designRefetch" with true when it has put the service's design on
// screen and false when that fetch failed. After an open, a failure is said in a
// notice and the fetch is tried again a few times; any later success clears it.
// The notice is rewritten only while it is still the one on show.
function onRefetch(ok) {
  if (ok) {
    if (staleMark !== null) deps.clearGuardNotice(staleMark);
    stopRetrying();
    return;
  }
  if (!awaiting) return;
  const again = awaiting.tries < REFETCH_TRIES;
  if (staleMark === null || staleMark === deps.guardNoticeMark()) {
    deps.showFailure(`${awaiting.name} was opened, but the design on screen could not be `
      + "refreshed from the service: it still shows the earlier design. "
      + (again ? "Trying again." : "Reload the page to see it."));
    staleMark = deps.guardNoticeMark();
  }
  if (!again) { awaiting = null; return; }
  awaiting.tries++;
  clearTimeout(retryTimer);
  retryTimer = setTimeout(() => { if (awaiting) deps.refetchDesign(); }, REFETCH_GAP_MS);
}

function stopRetrying() {
  clearTimeout(retryTimer);
  retryTimer = null;
  awaiting = null;
  staleMark = null;
}

// Keys while the panel is open. Listening on `document` in the bubble phase, and
// stopping there, keeps every key from the cockpit's own shortcuts (delete, paste,
// finish a pen path, space-pan: all listeners on `window`), which would edit the
// design behind the panel. The key's target has already had it, and its default
// action is untouched, so a key on a focused Withdraw, Abort, Stop or Approach still
// activates that control. Escape closes the panel from anywhere. Every other key is
// acted on only when its target is inside the panel: Enter in the name field saves;
// Enter, Up and Down on the list (or the panel itself) drive the list. A held Enter
// does nothing, also on the panel's own buttons.
function onKeydown(ev) {
  if (!isOpen()) return;
  ev.stopPropagation();
  if (ev.key === "Escape") { closeDialog(); return; }
  const dlg = $("designDialog");
  const t = ev.target;
  if (!dlg.contains(t)) return;
  if (ev.key === "Enter" && ev.repeat) { ev.preventDefault(); return; }
  if (ev.key === "Enter" && t === $("designSaveName")) { saveTyped(); return; }
  if (dialogMode !== "open" || (t !== dlg && t !== $("designFileList"))) return;
  if (ev.key === "Enter") { ev.preventDefault(); openSelected(); }
  else if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
    ev.preventDefault();
    moveSelection(ev.key === "ArrowDown" ? 1 : -1);
  }
}

// deps: refetchDesign() fetches GET /api/design and adopts it; showRefusal(body, text) and
// showFailure(text) put a refusal or a failure in the guard notice;
// guardNoticeMark() / clearGuardNotice(mark) take that one down again.
export function initDesignFiles(injected) {
  deps = injected;
  lastName = recallName();
  notice = registerBanner($("designNotice"),
    { kind: "event", closeBtn: $("designNoticeDismiss") });
  $("saveDesign").onclick = () => openDialog("save");
  $("openDesign").onclick = () => openDialog("open");
  $("designDialogClose").onclick = () => closeDialog();
  $("designSaveConfirm").onclick = () => { saveTyped(); };
  $("designOpenConfirm").onclick = () => { openSelected(); };
  $("designSaveName").oninput = () => renderExists();
  document.addEventListener("keydown", onKeydown);
  subscribe("designRefetch", onRefetch);
  window.addEventListener("pagehide", stopRetrying);
  document.addEventListener("visibilitychange", () => { if (document.hidden) stopRetrying(); });
  window.addEventListener("resize", placeDialog);
  const panel = document.querySelector(".scanpanel");
  if (panel && typeof ResizeObserver === "function") new ResizeObserver(placeDialog).observe(panel);
}
