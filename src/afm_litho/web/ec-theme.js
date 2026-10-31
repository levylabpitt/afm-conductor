// Theme colours for canvas drawing. The palette lives as CSS custom properties on
// :root in edit-cockpit.css; canvas code reads them here so the page and the
// canvases share one source. Reads happen at draw time (no DOM access at import).

// The value of a custom property, e.g. cssVar("--accent") -> "#4fc1e0".
export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// The same colour with an alpha, for translucent strokes and fills. The palette
// tokens are #rrggbb; anything else is returned unchanged (opaque).
export function cssVarA(name, alpha) {
  const hex = cssVar(name);
  const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  if (!m) return hex;
  const [r, g, b] = m.slice(1).map((h) => parseInt(h, 16));
  return `rgba(${r},${g},${b},${alpha})`;
}
