import { describe, it, expect } from "vitest";
import {
  fitTransform, invTransform, pxPerUm, pointInPoly, distToSeg, distToPolyline, objBounds,
  boundsOfObjects, snapRadiusUm, portSnapHit, percentileRange, zoomViewAbout,
  clampZoomFactor, wheelZoomFactor,
} from "../../src/afm_litho/web/ec-geom.js";

const close = (got, want, digits = 9) => {
  expect(got).toHaveLength(want.length);
  want.forEach((w, i) => expect(got[i]).toBeCloseTo(w, digits));
};

describe("fitTransform / invTransform", () => {
  it("maps a wide view into a square canvas, centred, y flipped", () => {
    // 10x5 um into 200x200 px: scale 20 px/um (width-limited), 50 px margin top and bottom.
    const f = fitTransform({ x0: 0, y0: 0, w: 10, h: 5 }, 200, 200);
    close(f(0, 0), [0, 150]);
    close(f(10, 5), [200, 50]);
    close(f(5, 2.5), [100, 100]);
  });

  it("honours the view origin", () => {
    const f = fitTransform({ x0: 100, y0: -50, w: 10, h: 10 }, 100, 100);
    close(f(100, -50), [0, 100]);
    close(f(110, -40), [100, 0]);
  });

  it("is inverted exactly by invTransform, in a non-square canvas", () => {
    const view = { x0: -3.5, y0: 12, w: 7, h: 2 };
    const f = fitTransform(view, 640, 480);
    const g = invTransform(view, 640, 480);
    for (const [x, y] of [[-3.5, 12], [0, 13], [3.5, 14], [1.234, 12.987]]) {
      close(g(...f(x, y)), [x, y], 9);
    }
  });

  it("pxPerUm is the smaller of the two axis scales", () => {
    expect(pxPerUm({ x0: 0, y0: 0, w: 10, h: 5 }, 200, 200)).toBe(20);
    expect(pxPerUm({ x0: 0, y0: 0, w: 2, h: 8 }, 200, 200)).toBe(25);
  });
});

describe("zoom maths", () => {
  const view = { x0: 0, y0: 0, w: 10, h: 10 };

  it("zoomViewAbout scales the box about the pointer micron", () => {
    expect(zoomViewAbout(view, 2, 4, 0.5)).toEqual({ x0: 1, y0: 2, w: 5, h: 5 });
    expect(zoomViewAbout(view, 2, 4, 2)).toEqual({ x0: -2, y0: -4, w: 20, h: 20 });
  });

  it("keeps the point under the pointer on the same pixel", () => {
    const before = fitTransform(view, 100, 100)(2, 4);
    const after = fitTransform(zoomViewAbout(view, 2, 4, 0.37), 100, 100)(2, 4);
    close(after, before, 9);
  });

  it("clampZoomFactor stops at the minimum and maximum view size", () => {
    expect(clampZoomFactor(view, 0.001, 0.1, 20)).toBeCloseTo(0.01, 12); // 10 * 0.01 = 0.1 um
    expect(clampZoomFactor(view, 100, 0.1, 20)).toBe(2);                 // 10 * 2 = 20 um
    expect(clampZoomFactor(view, 0.5, 0.1, 20)).toBe(0.5);               // inside the range
  });

  it("lets an out-of-range view move toward the range but never further out", () => {
    const big = { x0: 0, y0: 0, w: 50, h: 40 };
    expect(clampZoomFactor(big, 0.5, 0.1, 20)).toBe(0.5);
    expect(clampZoomFactor(big, 2, 0.1, 20)).toBe(1);
  });

  it("returns 1 for a degenerate view or factor", () => {
    expect(clampZoomFactor({ x0: 0, y0: 0, w: 0, h: 0 }, 2, 0.1, 20)).toBe(1);
    expect(clampZoomFactor(view, 0, 0.1, 20)).toBe(1);
    expect(clampZoomFactor(view, -1, 0.1, 20)).toBe(1);
    expect(clampZoomFactor(view, NaN, 0.1, 20)).toBe(1);
  });

  it("wheelZoomFactor is exponential in pixels and converts line/page deltas", () => {
    expect(wheelZoomFactor(0, 0, 0.002)).toBe(1);
    expect(wheelZoomFactor(-100, 0, 0.002)).toBeCloseTo(Math.exp(-0.2), 12);
    expect(wheelZoomFactor(-5, 1, 0.002)).toBeCloseTo(Math.exp(-5 * 16 * 0.002), 12);
    expect(wheelZoomFactor(1, 2, 0.001)).toBeCloseTo(Math.exp(0.8), 12);
    // many small steps compose to one big step
    expect(wheelZoomFactor(10, 0, 0.002) ** 10).toBeCloseTo(wheelZoomFactor(100, 0, 0.002), 12);
  });
});

describe("hit-test cores", () => {
  const square = [[0, 0], [4, 0], [4, 4], [0, 4]];

  it("pointInPoly: inside, outside, and a concave notch", () => {
    expect(pointInPoly(square, 2, 2)).toBe(true);
    expect(pointInPoly(square, 5, 2)).toBe(false);
    expect(pointInPoly(square, -0.1, 2)).toBe(false);
    const u = [[0, 0], [6, 0], [6, 6], [4, 6], [4, 2], [2, 2], [2, 6], [0, 6]];
    expect(pointInPoly(u, 3, 4)).toBe(false);   // in the notch
    expect(pointInPoly(u, 1, 4)).toBe(true);    // in the left arm
    expect(pointInPoly(u, 5, 4)).toBe(true);    // in the right arm
  });

  it("distToSeg: perpendicular, beyond an end, degenerate segment", () => {
    expect(distToSeg(5, 3, 0, 0, 10, 0)).toBe(3);
    expect(distToSeg(13, 4, 0, 0, 10, 0)).toBe(5);      // nearest point is the end (10,0)
    expect(distToSeg(-3, 4, 0, 0, 10, 0)).toBe(5);
    expect(distToSeg(3, 4, 0, 0, 0, 0)).toBe(5);        // zero-length segment
  });

  it("distToPolyline takes the nearest segment; one point has no segment", () => {
    const pl = [[0, 0], [10, 0], [10, 10]];
    expect(distToPolyline(pl, 12, 5)).toBe(2);
    expect(distToPolyline(pl, 5, -1)).toBe(1);
    expect(distToPolyline([[1, 1]], 1, 1)).toBe(Infinity);
  });

  it("objBounds is the axis-aligned box, or null with no extent", () => {
    expect(objBounds({ points_um: [[1, 5], [3, -2], [2, 9]] }))
      .toEqual({ minx: 1, miny: -2, maxx: 3, maxy: 9 });
    expect(objBounds({ points_um: [[2, 2]] })).toBeNull();
    expect(objBounds({ points_um: [[2, 2], [2, 2]] })).toBeNull();
    expect(objBounds({})).toBeNull();
    // a vertical line has extent in y only and still counts
    expect(objBounds({ points_um: [[2, 0], [2, 5]] })).not.toBeNull();
  });

  it("boundsOfObjects pads by 8 % plus 0.5 um; empty falls back to 10x10", () => {
    const b = boundsOfObjects([{ points_um: [[0, 0], [10, 20]] }]);
    expect(b.x0).toBeCloseTo(-1.3, 12);
    expect(b.y0).toBeCloseTo(-2.1, 12);
    expect(b.w).toBeCloseTo(12.6, 12);
    expect(b.h).toBeCloseTo(24.2, 12);
    expect(boundsOfObjects([])).toEqual({ x0: 0, y0: 0, w: 10, h: 10 });
  });

  it("snap radius is half the pad width with a 0.5 um floor; hit is inclusive", () => {
    expect(snapRadiusUm(4)).toBe(2);
    expect(snapRadiusUm(0.2)).toBe(0.5);
    expect(snapRadiusUm(undefined)).toBe(0.5);
    const port = { position_um: [10, 10], width_um: 4 };
    expect(portSnapHit(port, 12, 10)).toBe(true);          // exactly on the radius
    expect(portSnapHit(port, 12.01, 10)).toBe(false);
    expect(portSnapHit(port, 11, 10, 0.5)).toBe(false);    // explicit radius wins
    expect(portSnapHit({ width_um: 4 }, 0, 0)).toBe(false);
    expect(portSnapHit(null, 0, 0)).toBe(false);
  });
});

describe("percentileRange", () => {
  it("ignores NaN, null and infinities; nearest-rank percentiles", () => {
    const v = [NaN, 5, 1, null, 3, Infinity, 2, -Infinity, 4];
    expect(percentileRange(v, 0, 100)).toEqual([1, 5]);
    expect(percentileRange(v, 25, 75)).toEqual([2, 4]);
  });

  it("returns null with no finite values", () => {
    expect(percentileRange([NaN, null, Infinity], 2, 98)).toBeNull();
    expect(percentileRange([], 2, 98)).toBeNull();
  });

  it("widens a flat frame so the range is never zero", () => {
    const [lo, hi] = percentileRange([3, 3, 3], 2, 98);
    expect(lo).toBeCloseTo(3 - 3e-6, 12);
    expect(hi).toBeCloseTo(3 + 3e-6, 12);
    const [z0, z1] = percentileRange([0, 0], 2, 98);
    expect(z0).toBe(-1e-9);
    expect(z1).toBe(1e-9);
  });
});
