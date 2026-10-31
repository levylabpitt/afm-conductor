// ec-banners.js -- one dock for every cockpit banner / notice (docs/COCKPIT_BANNERS_SPEC.md).
//
// Banners sit in the message row (#ecMessageRow) under the header, a full-width row of
// its own, so no banner can cover Abort, the tab strip or the scan image. A banner shows
// on one line with an ellipsis; when its text does not fit, a "more" control wraps it
// and the row grows, pushing the panes down. Every banner has a close control:
//
//   * a STATE banner (shown while a condition holds) collapses to a chip in the row; the
//     chip re-expands it, it stays collapsed while the condition persists, the chip
//     goes when the condition clears, and the next occurrence shows the full banner.
//   * an EVENT notice (a refusal, a park result) closes entirely; a new event shows it.
//
// Closing only changes visibility. The guard behind a banner (the Write gate, the
// fault latch, a refusal) is owned by its module and never reads banner state.
//
// A banner's own element keeps `style.display` as its visible flag ("" shown, "none"
// hidden), so existing callers and tests that read or set it keep working.

let row = null;
let dock = null;
let chipRow = null;
let pending = false;

function ensureDock() {
  if (dock) return dock;
  row = document.getElementById("ecMessageRow");
  if (!row) {
    row = document.createElement("div");
    row.id = "ecMessageRow";
    row.className = "msgrow";
    const header = document.querySelector("header");
    if (header) header.after(row);
    else document.body.prepend(row);
  }
  dock = document.getElementById("ecBannerDock");
  if (!dock) {
    dock = document.createElement("div");
    dock.id = "ecBannerDock";
    dock.className = "bannerdock";
    row.appendChild(dock);
  }
  chipRow = document.getElementById("ecBannerChips");
  if (!chipRow) {
    chipRow = document.createElement("span");
    chipRow.id = "ecBannerChips";
    chipRow.className = "bannerchips";
    row.appendChild(chipRow);
  }
  // Callers change a banner's text, and tests its style.display, without going through
  // this module, so the row re-checks itself on any change inside it.
  new MutationObserver(scheduleRefresh).observe(row, {
    subtree: true, childList: true, characterData: true,
    attributes: true, attributeFilter: ["style"],
  });
  window.addEventListener("resize", scheduleRefresh);
  if (typeof ResizeObserver === "function") new ResizeObserver(scheduleRefresh).observe(row);
  return dock;
}

// Batched to the next frame: a refresh changes the row's size, and doing it inside a
// ResizeObserver callback would loop.
function scheduleRefresh() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => { pending = false; refreshRow(); });
}

function shown(el) { return el.style.display !== "none"; }

// The row reads "none" when no banner and no chip shows, and each banner offers "more"
// only while its one-line text is cut off (or "less" while it is wrapped open).
function refreshRow() {
  if (!row) return;
  let any = false;
  for (const el of dock.children) {
    if (!shown(el)) continue;
    any = true;
    const text = el.querySelector(":scope > .banner-text");
    const more = el.querySelector(":scope > .banner-more");
    if (!text || !more) continue;
    const open = el.classList.contains("open");
    const cut = text.scrollWidth > text.clientWidth + 1;
    const want = open || cut ? "" : "none";
    if (more.style.display !== want) more.style.display = want;
    const label = open ? "less" : "more";
    if (more.textContent !== label) more.textContent = label;
  }
  for (const chip of chipRow.children) if (shown(chip)) any = true;
  row.classList.toggle("has-msg", any);
}

// Register a banner element. Options:
//   kind      "state" | "event"
//   label     short chip text (state banners)
//   severity  "warn" | "err" (chip colour; the banner keeps its own colours)
//   closeBtn  an existing button to use as the close control (else one is added)
//   first     dock this banner before the others (safety-critical ones stay in view)
//   interceptClose  called when the operator clicks the close control; returning true
//             means it handled the click and the banner is left as it is
//   closeTitle  function giving the close control's tooltip, or null for the default
// Returns { show, hide, setActive, close, expand, setSeverity, setKind, setOpen,
//   isCollapsed }.
export function registerBanner(el, opts = {}) {
  if (!el) {
    const noop = () => {};
    return { show: noop, hide: noop, setActive: noop, close: noop, expand: noop,
      setSeverity: noop, setKind: noop, setOpen: noop, isCollapsed: () => false };
  }
  ensureDock();
  let kind = opts.kind === "state" ? "state" : "event";
  let severity = opts.severity || "warn";
  let active = el.style.display !== "none";
  let collapsed = false;

  el.classList.add("dock-notice");
  if (opts.first) dock.prepend(el);
  else dock.appendChild(el);

  // The banner's first span is its message; it is the part cut to one line.
  const text = el.querySelector(":scope > span");
  if (text) text.classList.add("banner-text");

  let closeBtn = opts.closeBtn || null;
  if (!closeBtn) {
    closeBtn = document.createElement("button");
    el.appendChild(closeBtn);
  }
  closeBtn.type = "button";
  closeBtn.classList.add("banner-close");
  closeBtn.textContent = "×";
  closeBtn.setAttribute("aria-label", "close");

  const more = document.createElement("button");
  more.type = "button";
  more.className = "banner-more";
  more.textContent = "more";
  more.title = "show the whole message";
  more.style.display = "none";
  el.insertBefore(more, closeBtn);
  // Opened shows the whole message wrapped; closed is one line with an ellipsis.
  function setOpen(on) {
    el.classList.toggle("open", !!on);
    more.title = on ? "back to one line" : "show the whole message";
    scheduleRefresh();
  }
  more.addEventListener("click", (e) => {
    e.stopPropagation();
    setOpen(!el.classList.contains("open"));
  });

  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = "chip banner-chip";
  chip.dataset.banner = el.id || "";
  chip.textContent = opts.label || "notice";
  chip.title = "show the full banner";
  chip.style.display = "none";
  chipRow.appendChild(chip);

  function render() {
    const custom = typeof opts.closeTitle === "function" ? opts.closeTitle() : null;
    closeBtn.title = custom || (kind === "state"
      ? "close: collapse to a chip in the message row (the condition still applies)"
      : "close");
    el.style.display = active && !collapsed ? "" : "none";
    if (el.style.display === "none") el.classList.remove("open");
    chip.style.display = active && collapsed ? "" : "none";
    chip.classList.toggle("busy", severity !== "err");
    chip.classList.toggle("bad", severity === "err");
    scheduleRefresh();
  }
  function close() {
    if (kind === "state" && active) collapsed = true;
    else active = false;
    render();
  }
  function closeByOperator() {
    if (typeof opts.interceptClose === "function" && opts.interceptClose()) return;
    close();
  }
  function expand() { collapsed = false; render(); }
  // A new event (or a state turning on) shows the full banner.
  function show() { active = true; collapsed = false; render(); }
  function hide() { active = false; collapsed = false; render(); }
  // Idempotent for state banners: a condition that persists keeps a collapsed banner
  // collapsed; only an off -> on transition expands it.
  function setActive(on) {
    if (on && !active) show();
    else if (!on && active) hide();
    else render();
  }
  function setSeverity(s) { severity = s; render(); }
  function setKind(k) { kind = k === "state" ? "state" : "event"; render(); }

  closeBtn.addEventListener("click", (e) => { e.stopPropagation(); closeByOperator(); });
  chip.addEventListener("click", (e) => { e.stopPropagation(); expand(); });
  render();
  return { show, hide, setActive, close, expand, setSeverity, setKind, setOpen,
    isCollapsed: () => active && collapsed };
}
