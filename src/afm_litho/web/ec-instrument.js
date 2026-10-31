// ec-instrument.js -- the cockpit's instrument-model surface
// (docs/INSTRUMENT_MODEL_SPEC.md sec 1/4/10). A Layer-2/3 module in the sense of
// docs/COCKPIT_MODULES.md: net-new instrument-model JS lives here, imported by
// ec-main.js, not appended to the monolith and with no window.* bridging. It owns
//
//   * the READ-ONLY instrument panel (amp gain + every model field with its source),
//   * the "open config file" / "reload config" buttons (act ON THE SERVICE HOST),
//   * hydrating #lithoSetpoint from hub.last_write_params.setpoint_relative and the
//     scan #setpoint from hub.last_scan_params per mode -- server returns truth,
//     the client re-adopts it (never an HTML literal),
//   * the #writeEstimate total (GET /api/litho/estimate) + per-object seconds,
//   * the Write gate (a write in progress, an empty setpoint box).
//
// The 2026-09-22 incident this closes: a reload reset an EDITABLE amp-gain box to
// its HTML value="1" and a 30 V write went out at gain 1. The amp gain is now
// service-owned and read-only here; the Write request takes it from THIS model.

import {
  getInstrument, getLithoEstimate, getState, latestOnly,
  postInstrumentOpenConfig, postInstrumentReload, postInstrumentOverride, onRefusal,
} from "./ec-api.js";
import { logLine } from "./ec-log.js";
import { registerBanner } from "./ec-banners.js";

// Banner visibility only; closing it never touches the refusal.
const guardBanner = registerBanner(document.getElementById("guardNotice"),
  { kind: "event", severity: "err",
    closeBtn: document.getElementById("guardNoticeDismiss") });

// The command guard's stable refusal reasons (docs/INSTRUMENT_MODEL_SPEC.md sec 2/12;
// B3 review M2). A response carrying one of these is a synchronous guard REFUSAL --
// the tip did not move -- and must be shown to the operator, not merely logged.
const GUARD_REASONS = new Set([
  "model_incomplete", "amp_gain_missing", "amp_gain_mismatch",
  "tip_voltage_out_of_window", "dac_rail_exceeded", "free_air_missing",
  "free_air_stale", "setpoint_out_of_band", "setpoint_exceeds_sum", "sum_missing",
  "unknown_object", "object_disabled", "scan_out_of_range", "fault_latched",
  "invalid_params",
  "chiral_bias_fixed", "chiral_unrebakeable", "chiral_build_refused",
  "chiral_bad_setting",
]);

// Surface a guard refusal in the docked #guardNotice (never log-only). The body's
// `error` text already carries the offending values (e.g. "amp_gain 1.0 does not
// match the model's 25.0"), so it is shown verbatim next to the reason code.
export function showRefusal(body) {
  if (!body || !GUARD_REASONS.has(body.reason)) return;
  showInGuardNotice(body.reason, body.error || "");
}

// A refusal from outside the command guard (a design file), shown in the same notice.
// `detail` replaces the service's text when given. A guard reason is skipped: the
// refusal sink has already shown it.
export function showOtherRefusal(body, detail) {
  if (!body || GUARD_REASONS.has(body.reason)) return;
  showInGuardNotice(body.reason || "error", detail || body.error || "");
}

// A call that failed without a refusal body (the service did not answer).
export function showFailure(text) {
  const txt = document.getElementById("guardNoticeText");
  if (!txt) return;
  txt.textContent = text;
  guardBanner.show();
  noticeMark++;
  logLine(text, "warn");
}

// guardNoticeMark() identifies what the notice shows now; clearGuardNotice(mark)
// closes the notice only if it still shows that, so a caller can take down its own
// refusal without closing a later one from another door.
let noticeMark = 0;
export function guardNoticeMark() { return noticeMark; }
export function clearGuardNotice(mark) {
  if (mark === noticeMark) guardBanner.hide();
}

function showInGuardNotice(reason, detail) {
  const txt = document.getElementById("guardNoticeText");
  if (!txt) return;
  txt.textContent = `refused (${reason})${detail ? ": " + detail : ""}`;
  guardBanner.show();
  noticeMark++;
  logLine(`REFUSED ${reason}: ${detail}`, "warn");
}

// The last model dict from GET /api/instrument. amp gain for the Write request is
// read from HERE (getAmpGain), never a typed box -- a stale screen can never send
// the wrong gain because there is no editable gain on the page at all.
let model = null;

// -------------------------------------------------------------- helpers --
function fieldValue(d, key) {
  const f = d && d[key];
  if (f == null) return undefined;
  // to_dict() emits every field as {value, source}; tolerate a bare value too.
  return (typeof f === "object" && "value" in f) ? f.value : f;
}
function fieldSource(d, key) {
  const f = d && d[key];
  return (f && typeof f === "object" && "source" in f) ? f.source : null;
}
function fmtValue(v) {
  if (v == null) return "—";
  if (Array.isArray(v)) return "[" + v.map(fmtValue).join(", ") + "]";
  if (typeof v === "number") return String(v);
  return String(v);
}

// The amp gain the Write request must send (docs/INSTRUMENT_MODEL_SPEC.md sec 2:
// the guard refuses a mismatch, and a MISSING value is amp_gain_missing, never a
// default). Returns the model's tip_amp_gain, or null when the model has none --
// the Write handler sends what it gets, so the service refuses honestly rather
// than a silent 1.0.
export function getAmpGain() {
  if (model == null) return null;
  const g = fieldValue(model, "tip_amp_gain");
  return (typeof g === "number") ? g : (g != null ? Number(g) : null);
}

// One numeric model field, or null before the model is loaded or when it has none.
export function getModelNumber(key) {
  const v = fieldValue(model, key);
  const n = v == null ? NaN : Number(v);
  return Number.isFinite(n) ? n : null;
}

// The instrument's XY scan range [x, y] in um, centred on the lab origin (the
// service's check_scan refuses a frame past +-range/2), or null before the model
// is loaded. A scalar range is square.
export function getXyRangeUm() {
  const v = fieldValue(model, "xy_range_um");
  const r = Array.isArray(v) ? [Number(v[0]), Number(v.length > 1 ? v[1] : v[0])]
    : [Number(v), Number(v)];
  return v != null && r.every((n) => Number.isFinite(n) && n > 0) ? r : null;
}

// The magnitude of the widest allowed tip voltage (max(|v_min|, |v_max|) of the
// sample's tip_voltage_window_v). The object-list swatch + overlay colour scale is
// FLOORED at this so a voltage's colour reflects where it sits in the physically
// allowed window, not merely relative to the current design's max -- otherwise the
// single highest-voltage object always renders the same pure colour whatever its
// value, and a voltage edit on it is invisible. 0 when the model is not loaded (the
// colour scale then falls back to pure pattern-relative, unchanged).
export function getVoltageWindowMax() {
  if (model == null) return 0;
  const w = fieldValue(model, "tip_voltage_window_v");
  if (!Array.isArray(w) || w.length < 2) return 0;
  const m = Math.max(Math.abs(Number(w[0]) || 0), Math.abs(Number(w[1]) || 0));
  return isFinite(m) ? m : 0;
}

// Every InstrumentModel Field the Config dialog surfaces, in display + save
// order (mirrors the service's OVERRIDABLE_FIELDS, section 14 item 3: changed
// fields are posted "in order"). amp gain also has its own #ampGainRO readout on
// the Write tab; it is one row here like every other field.
const CONFIG_FIELDS = [
  "platform", "z_out_channel", "z_verify_channels", "z_park_v", "z_park_tol_v",
  "dac_rail_v", "tip_amp_gain", "tip_voltage_window_v", "contact_setpoint_band_v",
  "ac_setpoint_fraction_band", "free_air_max_age_s", "xy_range_um",
  "free_air_abs_band_v", "sum_floor_v", "sum_ceiling_frac", "z_rail_v",
  "interlock_s", "write_setpoint_relative_v",
];

// The live backend flag ("twin" / "hardware"), shown read-only (greyed) in the
// dialog (section 14 items 3/4: the hardware-backend flag is never editable).
// Seeded from GET /api/state at init and refreshed when the dialog opens.
let backend = null;

// Config-dialog edit state (section 14 item 3). `editing` swaps every value cell
// for an input; `preEdit` is the snapshot Cancel restores to (and the baseline
// Save diffs against, so only CHANGED fields are posted).
let editing = false;
let preEdit = null;

// HTML-escape a value for safe innerHTML insertion (field values are
// service-owned, but a config file is hand-edited on the host).
function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderModel() {
  const gainEl = document.getElementById("ampGainRO");
  if (gainEl) {
    const g = getAmpGain();
    gainEl.textContent = (g == null) ? "—" : fmtValue(g);
    const src = fieldSource(model, "tip_amp_gain");
    if (src) gainEl.title = `tip_amp_gain = ${fmtValue(g)} (source: ${src})`;
  }
  renderConfigTable();
}

// The config-file path + backend header of the dialog.
function renderConfigMeta() {
  const pathEl = document.getElementById("configPath");
  if (pathEl) pathEl.textContent = (model && model.config_path) ? model.config_path : "—";
  const beEl = document.getElementById("configBackend");
  if (beEl) beEl.textContent = backend || "—";
}

// The editable text for a value: an array renders as comma-separated, a scalar
// as itself. Parsed back by parseInput against the field's current type.
function valueToInput(v) {
  if (Array.isArray(v)) return v.map((x) => String(x)).join(", ");
  return v == null ? "" : String(v);
}
// Parse an edited input back to the field's type, keyed off the OLD value: an
// array field splits on commas (numeric elements become numbers, else strings);
// a numeric field becomes a Number; everything else stays a string. The service
// guard has final say -- this only shapes the JSON so the override door gets the
// list/number/string it expects.
function parseInput(raw, oldValue) {
  if (Array.isArray(oldValue)) {
    const parts = String(raw).split(",").map((s) => s.trim()).filter((s) => s !== "");
    return parts.map((s) => {
      const n = Number(s);
      return (s !== "" && Number.isFinite(n)) ? n : s;
    });
  }
  if (typeof oldValue === "number") return Number(raw);
  return String(raw);
}
// Deep-ish equality for the scalar / small-array field values the model holds.
function valuesEqual(a, b) { return JSON.stringify(a) === JSON.stringify(b); }

// Render every model field as a row (two columns: parameter, value) with its
// source shown faintly. When `editing`, the value cell is an input; otherwise
// read-only text (section 14 items 2/3).
function renderConfigTable() {
  renderConfigMeta();
  const box = document.getElementById("configRows");
  if (!box || model == null) return;
  const rows = [];
  for (const key of CONFIG_FIELDS) {
    if (!(key in model)) continue;
    const v = fieldValue(model, key);
    const src = fieldSource(model, key);
    const valCell = editing
      ? `<input class="configinput" data-field-input="${key}" value="${esc(valueToInput(v))}">`
        + `<span class="configsrc">${src ? esc(src) : ""}</span>`
        + `<span class="configrowerr" data-field-err="${key}"></span>`
      : `<span class="configval">${esc(fmtValue(v))}</span>`
        + `<span class="configsrc">${src ? esc(src) : ""}</span>`;
    rows.push(
      `<tr data-field="${key}"><td class="configkey">${esc(key)}</td>`
      + `<td class="configvalcell">${valCell}</td></tr>`);
  }
  box.innerHTML = rows.join("");
}

// Toggle the Edit / Save+Cancel buttons for the current `editing` state.
function setEditButtons() {
  const edit = document.getElementById("configEdit");
  const save = document.getElementById("configSave");
  const cancel = document.getElementById("configCancel");
  if (edit) edit.style.display = editing ? "none" : "";
  if (save) save.style.display = editing ? "" : "none";
  if (cancel) cancel.style.display = editing ? "" : "none";
}
function setConfigMsg(text, isErr = true) {
  const el = document.getElementById("configMsg");
  if (!el) return;
  el.textContent = text || "";
  el.style.color = isErr ? "" : "var(--dim)";
  el.style.display = text ? "" : "none";
}

// Enter edit mode: snapshot the current values (Cancel/diff baseline) and
// re-render with inputs.
function startEdit() {
  if (model == null) return;
  preEdit = {};
  for (const key of CONFIG_FIELDS) {
    if (key in model) preEdit[key] = fieldValue(model, key);
  }
  editing = true;
  setConfigMsg("");
  renderConfigTable();
  setEditButtons();
}
// Cancel: restore the pre-edit values and send NOTHING (section 14 item 3).
function cancelEdit() {
  editing = false;
  preEdit = null;
  setConfigMsg("");
  renderConfigTable();
  setEditButtons();
}

// The fields whose input differs from the pre-edit snapshot, in CONFIG_FIELDS
// order (section 14 item 3: changed fields only, in order).
function changedFields() {
  const out = [];
  for (const key of CONFIG_FIELDS) {
    if (!preEdit || !(key in preEdit)) continue;
    const inp = document.querySelector(`[data-field-input="${key}"]`);
    if (!inp) continue;
    const parsed = parseInput(inp.value, preEdit[key]);
    if (!valuesEqual(parsed, preEdit[key])) out.push({ field: key, value: parsed });
  }
  return out;
}

// Save: post an override for each CHANGED field, in order, then re-read the
// model and close. A refused field (GuardRefused / invalid_params) STOPS the
// loop, shows its reason inline on that row, and keeps edit mode -- fields
// already applied stay applied (each override is durable on disk, section 14
// item 3). Amp gain gets a second explicit confirm before its post.
async function saveEdit() {
  const changed = changedFields();
  if (changed.length === 0) { cancelEdit(); return; }
  for (const err of document.querySelectorAll("[data-field-err]")) err.textContent = "";
  setConfigMsg("");
  for (const { field, value } of changed) {
    if (field === "tip_amp_gain"
        && !window.confirm(`Save tip amplifier gain = ${value}?`)) {
      // Declined the second confirm: leave the gain, keep editing.
      const errEl = document.querySelector('[data-field-err="tip_amp_gain"]');
      if (errEl) errEl.textContent = "not saved (confirm declined)";
      return;
    }
    const res = await postInstrumentOverride({ field, value, confirm: true });
    const body = res && res.data;
    const refused = !body || body.error || body.reason || body.applied !== true;
    if (refused) {
      const reason = (body && (body.reason || body.error)) || `HTTP ${res && res.status}`;
      const errEl = document.querySelector(`[data-field-err="${field}"]`);
      if (errEl) errEl.textContent = `refused: ${reason}`;
      setConfigMsg(`override refused on ${field}: ${reason}`);
      // Refresh the model so applied fields (before this one) show their new
      // source, but STAY in edit mode with the offending row flagged.
      await hydrateInstrument();
      preEdit = {};
      for (const key of CONFIG_FIELDS) {
        if (key in model) preEdit[key] = fieldValue(model, key);
      }
      renderConfigTable();
      setEditButtons();
      const errEl2 = document.querySelector(`[data-field-err="${field}"]`);
      if (errEl2) errEl2.textContent = `refused: ${reason}`;
      return;
    }
  }
  // All changed fields applied: re-read the model and close (section 14 item 3).
  editing = false;
  preEdit = null;
  await hydrateInstrument();
  renderConfigTable();
  setEditButtons();
  closeConfigDialog();
}

// -------------------------------------------------- hydrate from service --
// GET /api/instrument -> populate the read-only model panel, then hydrate the two
// setpoint boxes from the service's last-used params (server returns truth). No
// HTML literal survives for either box, so a reload never resets them.
// The value #lithoSetpoint hydrates to (section 15 item 2): this service process's
// last write setpoint_relative if present, else the model's write_setpoint_relative_v
// (code default 0.1). NEVER 0 and NEVER empty -- a zeroed/absent last-write falls
// through to the model default, which is a real non-zero force above free air. The
// guard's relative band still governs whatever the operator then types.
function chooseSetpointRelative() {
  if (model == null) return null;
  const lw = model.last_write_params || {};
  const lwRel = lw.setpoint_relative;
  if (typeof lwRel === "number" && isFinite(lwRel) && lwRel !== 0) return lwRel;
  const mv = fieldValue(model, "write_setpoint_relative_v");
  const mvn = (typeof mv === "number") ? mv : Number(mv);
  return (isFinite(mvn) && mvn !== 0) ? mvn : null;
}

export async function hydrateInstrument() {
  const res = await getInstrument();
  if (!res.ok || !res.data || res.data.error) {
    logLine(`instrument model unavailable: ${res.data && res.data.error ? res.data.error : res.status}`, "warn");
    return;
  }
  model = res.data;
  renderModel();

  // #lithoSetpoint <- the RELATIVE box (section 15 item 2). Order: this service
  // process's last write setpoint_relative if present, else the model's
  // write_setpoint_relative_v (code default 0.1); NEVER 0, NEVER empty. Never
  // clobber a value the operator is actively typing.
  const spEl = document.getElementById("lithoSetpoint");
  const rel = chooseSetpointRelative();
  if (spEl && rel != null && document.activeElement !== spEl) {
    spEl.value = rel;
    refreshWriteGate();   // a hydrated value clears the empty-field Write gate (N2)
  }

  // scan #setpoint <- hub.last_scan_params[mode].setpoint for the current mode.
  hydrateScanSetpoint();
}

// Hydrate the scan tab's #setpoint from hub.last_scan_params for the CURRENT mode
// (section 10: "hub.last_scan_params per mode"). Called on load and when the mode
// dropdown changes. Never clobbers a value the operator is actively typing.
export function hydrateScanSetpoint() {
  if (model == null) return;
  const lsp = model.last_scan_params || {};
  const modeEl = document.getElementById("mode");
  const mode = modeEl ? modeEl.value : "contact";
  // tolerate either {mode: {setpoint}} or a flat {setpoint} shape.
  const perMode = (lsp && typeof lsp[mode] === "object") ? lsp[mode] : lsp;
  const sp = perMode && perMode.setpoint != null ? perMode.setpoint : null;
  const el = document.getElementById("setpoint");
  if (el && sp != null && document.activeElement !== el) el.value = sp;
}

// ------------------------------------------------------- write estimate --
// #writeEstimate <- GET /api/litho/estimate total_s, and per-object seconds are
// stamped onto the object rows by name. The number comes from the SERVICE's own
// enabled set, so a stale on-screen view shows up as a mismatch. Tolerates a 404
// (the estimate endpoint's service side is owned by another slice) by showing a
// dash rather than throwing. Coalesced (latestOnly): the estimate of a dense fill
// takes the service seconds, and a fetch per design change would stack up.
export const refreshWriteEstimate = latestOnly(fetchWriteEstimate);
async function fetchWriteEstimate() {
  const el = document.getElementById("writeEstimate");
  const res = await getLithoEstimate();
  if (!res.ok || !res.data || res.data.error || typeof res.data.total_s !== "number") {
    if (el) el.textContent = "—";
    return;
  }
  const est = res.data;
  if (el) el.textContent = `${est.total_s.toFixed(1)} s`;
  // per-object seconds onto the matching rows (additive; the row's .nm/.dis/.swatch
  // structure the object-list renderer owns is left untouched).
  const perObj = {};
  for (const o of (est.objects || [])) perObj[o.id] = o;
  for (const row of document.querySelectorAll("#objList .objrow")) {
    const name = row.querySelector(".nm") ? row.querySelector(".nm").textContent : null;
    const o = name != null ? perObj[name] : null;
    let secs = row.querySelector(".secs");
    if (!secs) {
      secs = document.createElement("span");
      secs.className = "secs";
      row.appendChild(secs);
    }
    secs.textContent = (o && typeof o.seconds === "number" && o.enabled !== false)
      ? `${o.seconds.toFixed(1)} s` : "";
  }
}

// ----------------------------------------------------------- Write gate --
// Write is not gated on engagement: the run reads free air before every object
// (docs/COCKPIT_WRITE_GATE_SPEC.md), so an engaged tip is a normal pre-write state.
function applyGate() {
  // N2: an EMPTY setpoint must NOT silently coerce to 0 (a write at free air). Disable
  // Write and say why until a value is typed. The box has no HTML value= (it hydrates
  // from the service), so "" is the genuine no-value-yet state.
  const spEl = document.getElementById("lithoSetpoint");
  const spEmpty = !!(spEl && spEl.value.trim() === "");
  const hint = document.getElementById("lithoSetpointHint");
  if (hint) hint.style.display = spEmpty ? "" : "none";
  const perform = document.getElementById("perform");
  if (perform) {
    perform.disabled = perform.dataset.writing === "1" || spEmpty;
    perform.title = spEmpty
      ? "Set the write setpoint (V above free air) before writing."
      : "";
  }
}
// Re-evaluate the Write gate (call on #lithoSetpoint input so the empty-field guard
// tracks typing).
export function refreshWriteGate() { applyGate(); }
// Whether the write setpoint box currently has no value (N2). The Write handler uses
// it to refuse rather than coerce "" to 0.
export function writeSetpointEmpty() {
  const spEl = document.getElementById("lithoSetpoint");
  return !!(spEl && spEl.value.trim() === "");
}
// Called by the Write handler: while a write runs #perform stays disabled.
export function markWriting(on) {
  const perform = document.getElementById("perform");
  if (perform) perform.dataset.writing = on ? "1" : "0";
  applyGate();
}

// ------------------------------------------------------- config dialog --
// The modal opened from the header's Config button (section 14). Shows
// every model field read-only; item 3 adds Edit/Save/Cancel. Open re-fetches the
// model + backend so the dialog is always current (e.g. after an override).
async function openConfigDialog() {
  const dlg = document.getElementById("configDialog");
  if (!dlg) return;
  editing = false;               // always open read-only, never mid-edit
  preEdit = null;
  setConfigMsg("");
  const st = await getState();
  if (st.ok && st.data && st.data.backend) backend = st.data.backend;
  await hydrateInstrument();     // refetch model + re-render the table
  renderConfigTable();
  setEditButtons();
  dlg.style.display = "";
}
function closeConfigDialog() {
  const dlg = document.getElementById("configDialog");
  if (dlg) dlg.style.display = "none";
}

// -------------------------------------------------------------- wiring --
// Wire the panel buttons and seed the backend flag from the live state. Called once
// from ec-main boot.
export function initInstrument() {
  const configBtn = document.getElementById("configBtn");
  if (configBtn) configBtn.onclick = () => { openConfigDialog(); };
  const configClose = document.getElementById("configClose");
  if (configClose) configClose.onclick = () => closeConfigDialog();
  const configEdit = document.getElementById("configEdit");
  if (configEdit) configEdit.onclick = () => startEdit();
  const configSave = document.getElementById("configSave");
  if (configSave) configSave.onclick = () => { saveEdit(); };
  const configCancel = document.getElementById("configCancel");
  if (configCancel) configCancel.onclick = () => cancelEdit();
  const open = document.getElementById("openInstrumentConfig");
  if (open) open.onclick = async () => {
    const res = await postInstrumentOpenConfig();
    if (res.ok && res.data && res.data.path)
      logLine(`opened instrument config on the service host: ${res.data.path}`, "rx");
  };
  const reload = document.getElementById("reloadInstrumentConfig");
  if (reload) reload.onclick = async () => {
    const res = await postInstrumentReload();
    if (res.ok && res.data && !res.data.error) {
      logLine(`reloaded instrument config; diff: ${JSON.stringify(res.data.diff || {})}`, "rx");
      await hydrateInstrument();
    }
  };
  // M2: every guard refusal from any door (api() and request()) surfaces here.
  onRefusal(showRefusal);
  // seed the backend flag from the live state (the dialog shows it read-only).
  getState().then((res) => {
    if (res.ok && res.data) {
      if (res.data.backend) backend = res.data.backend;
      renderConfigMeta();
    }
  });
}
