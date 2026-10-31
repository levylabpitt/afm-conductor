// Zoom and pan of the shared canvas view, driven through real DOM events on a jsdom
// canvas (ec-viewnav.js). The canvas is 400x400 backing pixels, laid out 1:1.
import { describe, it, expect, beforeAll, beforeEach } from "vitest";
import { state } from "../../src/afm_litho/web/ec-state.js";

let nav;
let cv;
let range = null;        // the instrument's xy_range_um as the page has read it

const wheel = (init) => {
  const ev = new WheelEvent("wheel", { bubbles: true, cancelable: true, ...init });
  cv.dispatchEvent(ev);
  return ev;
};
const ptr = (type, init) => {
  const ev = new MouseEvent(type, { bubbles: true, cancelable: true, ...init });
  ev.pointerId = 1;
  cv.dispatchEvent(ev);
  return ev;
};

beforeAll(async () => {
  document.body.innerHTML = `<div id="host"><canvas id="ov" width="400" height="400"></canvas></div>
    <span id="viewCap"></span>`;
  cv = document.getElementById("ov");
  cv.getBoundingClientRect = () => ({ left: 0, top: 0, right: 400, bottom: 400, width: 400, height: 400 });
  window.requestAnimationFrame = (fn) => { fn(); return 0; };
  nav = await import("../../src/afm_litho/web/ec-viewnav.js");
  let redraws = 0;
  nav.initViewNav({ ovCv: cv, redraw: () => { redraws++; }, resetToScan: () => { state.view = null; },
                    applyToolCursor: () => {}, xyRangeUm: () => range });
});

beforeEach(() => {
  state.SCAN_FRAME = { x0: 0, y0: 0, w: 10, h: 10 };
  state.view = null;
  state.objects = [];
  state.currentTab = "edit";
  range = [90, 90];
});

// The navigable area is the scan frame, the design and the instrument's scan range
// (centred on the origin) together, plus 10 % of its larger side on each side.
describe("viewMaxUm", () => {
  it("is the scan range plus its margin when the range holds the frame and design", () => {
    // [-45, 45] holds the 10 um frame: 90 + 2 * 9
    expect(nav.viewMaxUm()).toBeCloseTo(108, 9);
  });

  it("uses the widest scanner range before the instrument model is read", () => {
    range = null;
    expect(nav.viewMaxUm()).toBeCloseTo(1000 * 1.2, 9);
  });

  it("grows with the design when the design reaches past the scan range", () => {
    range = [20, 20];
    state.objects = [{ points_um: [[0, 0], [30, 5]] }];
    // design bounds pad 30*0.08+0.5 = 2.9 each side: x -2.9..32.9; range x -10..10
    // -> union x -10..32.9 = 42.9, plus 10 % each side
    expect(nav.viewMaxUm()).toBeCloseTo(42.9 * 1.2, 9);
  });
});

describe("wheel pan", () => {
  it("two-finger scroll pans like a document: down shows lower micron y, right shows larger x", () => {
    // 400 px / 10 um = 40 px per um
    wheel({ deltaY: 40, clientX: 200, clientY: 200 });
    expect(state.view).toEqual({ x0: 0, y0: -1, w: 10, h: 10 });
    wheel({ deltaX: 80, clientX: 200, clientY: 200 });
    expect(state.view).toEqual({ x0: 2, y0: -1, w: 10, h: 10 });
  });

  it("line-mode deltas count 16 px per line", () => {
    wheel({ deltaY: 5, deltaMode: 1, clientX: 200, clientY: 200 });
    expect(state.view.y0).toBeCloseTo(-5 * 16 / 40, 12);
  });

  it("does not scale or move the view size", () => {
    wheel({ deltaX: 13, deltaY: 7, clientX: 200, clientY: 200 });
    expect(state.view.w).toBe(10);
    expect(state.view.h).toBe(10);
  });

  it("prevents the page scroll", () => {
    expect(wheel({ deltaY: 10, clientX: 5, clientY: 5 }).defaultPrevented).toBe(true);
  });
});

describe("ctrl+wheel zoom", () => {
  it("zooms about the pointer: the micron under it keeps its pixel", () => {
    // pointer at pixel (100, 300): micron x = 100/40 = 2.5, y = (400-300)/40 = 2.5
    wheel({ ctrlKey: true, deltaY: -100, clientX: 100, clientY: 300 });
    const f = Math.exp(-100 * 0.002);
    expect(state.view.w).toBeCloseTo(10 * f, 9);
    expect(state.view.x0).toBeCloseTo(2.5 - 2.5 * f, 9);
    expect(state.view.y0).toBeCloseTo(2.5 - 2.5 * f, 9);
  });

  it("zooming out past the maximum stops at the navigable area", () => {
    wheel({ ctrlKey: true, deltaY: 100000, clientX: 200, clientY: 200 });
    expect(state.view.w).toBeCloseTo(108, 9);
    expect(state.view.h).toBeCloseTo(108, 9);
  });

  it("a pan stops at the navigable area's edge", () => {
    wheel({ deltaX: 400000, deltaY: -400000, clientX: 200, clientY: 200 });
    // area -54..54: a 10 um view ends at its right and top edges
    expect(state.view.x0).toBeCloseTo(44, 9);
    expect(state.view.y0).toBeCloseTo(44, 9);
  });

  it("zooming in past the minimum stops at 0.1 um", () => {
    wheel({ ctrlKey: true, deltaY: -100000, clientX: 200, clientY: 200 });
    expect(state.view.w).toBeCloseTo(0.1, 9);
  });

  it("at the limit a further zoom changes nothing", () => {
    wheel({ ctrlKey: true, deltaY: 100000, clientX: 200, clientY: 200 });
    const at = { ...state.view };
    wheel({ ctrlKey: true, deltaY: 500, clientX: 50, clientY: 50 });
    expect(state.view).toEqual(at);
  });

  it("the view caption reports size and centre", () => {
    wheel({ deltaY: 40, clientX: 200, clientY: 200 });
    expect(document.getElementById("viewCap").textContent).toBe("view: 10×10 µm @ (5, 4)");
  });
});

describe("drag pan", () => {
  const drag = (button, from, to) => {
    ptr("pointerdown", { button, clientX: from[0], clientY: from[1] });
    ptr("pointermove", { clientX: to[0], clientY: to[1] });
    ptr("pointerup", { clientX: to[0], clientY: to[1] });
  };

  it("right-drag pans on any tab; the content follows the pointer", () => {
    drag(2, [100, 100], [140, 60]);       // 40 px right, 40 px up = 1 um each
    expect(state.view.x0).toBeCloseTo(-1, 12);
    expect(state.view.y0).toBeCloseTo(-1, 12);
  });

  it("left-drag pans on the Scan tab only", () => {
    state.currentTab = "scan";
    drag(0, [100, 100], [180, 100]);
    expect(state.view.x0).toBeCloseTo(-2, 12);
    state.view = null;
    state.currentTab = "edit";
    drag(0, [100, 100], [180, 100]);
    expect(state.view).toBeNull();
  });

  it("a drag is measured from where it started, not accumulated step to step", () => {
    ptr("pointerdown", { button: 2, clientX: 200, clientY: 200 });
    ptr("pointermove", { clientX: 240, clientY: 200 });
    ptr("pointermove", { clientX: 280, clientY: 200 });
    ptr("pointermove", { clientX: 240, clientY: 200 });
    ptr("pointerup", { clientX: 240, clientY: 200 });
    expect(state.view.x0).toBeCloseTo(-1, 12);
  });

  it("moves after pointerup do not pan", () => {
    drag(2, [100, 100], [140, 100]);
    const at = { ...state.view };
    ptr("pointermove", { clientX: 300, clientY: 300 });
    expect(state.view).toEqual(at);
  });
});
