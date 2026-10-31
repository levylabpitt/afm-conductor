import { describe, it, expect } from "vitest";
import {
  CUSP, SMOOTH, SYMMETRIC, bezMoveNode, bezSetHandle, bezSegments, cubicAt, bezFlatten,
  normalizeBez, synthBezier, penToPolyline, penToBezier, rectBezier, setNodeType, deleteNode,
  insertNode, joinNodes, splitNode, isRectBezier, bezCurveDrag,
} from "../../src/afm_litho/web/ec-bezier.js";

const n = (x, y, h_in = null, h_out = null, kind = CUSP) => ({ x, y, h_in, h_out, kind });
const close = (got, want, digits = 9) => {
  expect(got).toHaveLength(want.length);
  want.forEach((w, i) => expect(got[i]).toBeCloseTo(w, digits));
};

describe("cubicAt", () => {
  it("hits the endpoints and the analytic midpoint", () => {
    const p0 = [0, 0], c1 = [0, 8], c2 = [8, 8], p3 = [8, 0];
    close(cubicAt(p0, c1, c2, p3, 0), [0, 0]);
    close(cubicAt(p0, c1, c2, p3, 1), [8, 0]);
    close(cubicAt(p0, c1, c2, p3, 0.5), [4, 6]);
  });
});

describe("bezSetHandle: continuity rules", () => {
  it("cusp leaves the opposite handle alone", () => {
    const node = n(0, 0, [-1, 0], [2, 0], CUSP);
    bezSetHandle(node, "out", 0, 3);
    expect(node.h_out).toEqual([0, 3]);
    expect(node.h_in).toEqual([-1, 0]);
  });

  it("smooth swings the opposite handle collinear, keeping its length", () => {
    const node = n(0, 0, [-1, 0], [2, 0], SMOOTH);
    bezSetHandle(node, "out", 0, 3);
    close(node.h_in, [0, -1]);
  });

  it("symmetric mirrors the opposite handle, length included", () => {
    const node = n(0, 0, [-1, 0], [2, 0], SYMMETRIC);
    bezSetHandle(node, "out", 0, 3);
    close(node.h_in, [0, -3]);
  });

  it("smooth with no opposite handle takes the dragged handle's length", () => {
    const node = n(1, 1, null, null, SMOOTH);
    bezSetHandle(node, "in", 4, 5);        // vector (3,4), length 5
    close(node.h_out, [-2, -3]);
  });

  it("a handle dragged onto the anchor leaves the opposite alone (no NaN)", () => {
    const node = n(1, 1, [0, 1], [2, 1], SMOOTH);
    bezSetHandle(node, "out", 1, 1);
    expect(node.h_in).toEqual([0, 1]);
  });
});

describe("bezMoveNode", () => {
  it("moves the anchor and both handles by the same delta; null handles stay null", () => {
    const node = n(1, 2, [0.5, 2.5], [3, 2], SMOOTH);
    bezMoveNode(node, 4, -1);
    expect(node).toEqual(n(4, -1, [3.5, -0.5], [6, -1], SMOOTH));
    const bare = n(0, 0);
    bezMoveNode(bare, 5, 5);
    expect(bare.h_in).toBeNull();
    expect(bare.h_out).toBeNull();
  });
});

describe("bezSegments / bezFlatten", () => {
  it("open path of 3 nodes has 2 segments; closed has 3", () => {
    const nodes = [n(0, 0), n(1, 0), n(1, 1)];
    expect(bezSegments({ nodes, closed: false })).toHaveLength(2);
    expect(bezSegments({ nodes, closed: true })).toHaveLength(3);
    expect(bezSegments({ nodes: [n(0, 0)], closed: false })).toEqual([]);
  });

  it("a handle-less path flattens to its own vertices", () => {
    const pts = bezFlatten({ nodes: [n(0, 0), n(3, 4), n(6, 0)], closed: false }, 0.004);
    expect(pts).toEqual([[0, 0], [3, 4], [6, 0]]);
  });

  it("a curved path stays within tolerance of the true curve and ends on the anchors", () => {
    const bez = { nodes: [n(0, 0, null, [0, 8]), n(8, 0, [8, 8], null)], closed: false };
    const tol = 0.01;
    const pts = bezFlatten(bez, tol);
    expect(pts[0]).toEqual([0, 0]);
    expect(pts[pts.length - 1]).toEqual([8, 0]);
    expect(pts.length).toBeGreaterThan(20);
    // every polyline vertex lies on the curve (it is a subdivision point)
    const [p0, c1, c2, p3] = [[0, 0], [0, 8], [8, 8], [8, 0]];
    for (const [x, y] of pts) {
      let best = Infinity;
      for (let i = 0; i <= 4000; i++) {
        const [cx, cy] = cubicAt(p0, c1, c2, p3, i / 4000);
        best = Math.min(best, Math.hypot(cx - x, cy - y));
      }
      expect(best).toBeLessThan(2e-3);
    }
  });
});

describe("insertNode (de Casteljau split)", () => {
  it("splits a symmetric arch at t = 0.5 with the analytic controls", () => {
    const bez = { nodes: [n(0, 0, null, [0, 8]), n(8, 0, [8, 8], null)], closed: false };
    const idx = insertNode(bez, 0, 0.5);
    expect(idx).toBe(1);
    expect(bez.nodes).toHaveLength(3);
    const [a, mid, b] = bez.nodes;
    close(a.h_out, [0, 4]);
    close(b.h_in, [8, 4]);
    close([mid.x, mid.y], [4, 6]);
    close(mid.h_in, [2, 6]);
    close(mid.h_out, [6, 6]);
    expect(mid.kind).toBe(SMOOTH);
  });

  it("does not change the curve: before and after flatten to the same shape", () => {
    const mk = () => ({ nodes: [n(0, 0, null, [1, 6]), n(9, 2, [7, 7], null)], closed: false });
    const before = mk();
    const after = mk();
    insertNode(after, 0, 0.3);
    const ref = bezFlatten(before, 0.0005);
    // every point of the split curve is within 1e-3 of the original polyline
    for (const [x, y] of bezFlatten(after, 0.0005)) {
      let best = Infinity;
      for (let i = 1; i < ref.length; i++) {
        const [ax, ay] = ref[i - 1], [bx, by] = ref[i];
        const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
        const t = l2 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2)) : 0;
        best = Math.min(best, Math.hypot(x - ax - t * dx, y - ay - t * dy));
      }
      expect(best).toBeLessThan(1e-3);
    }
  });

  it("a straight segment stays on its line, at the cubic's own parameter", () => {
    // handle-less ends mean control points p0,p0,p3,p3, so t=0.25 sits at 3t^2-2t^3 = 0.15625
    const bez = { nodes: [n(0, 0), n(10, 0)], closed: false };
    insertNode(bez, 0, 0.25);
    const [a, mid, b] = bez.nodes;
    close([mid.x, mid.y], [1.5625, 0]);
    close(mid.h_in, [0.625, 0]);
    close(mid.h_out, [4.375, 0]);
    expect(a.h_out).toBeNull();      // the first sub-curve's start control collapses onto p0
    expect(b.h_in).toBeNull();
  });

  it("a collapsed handle is stored as null, not as a zero-length handle", () => {
    const bez = { nodes: [n(0, 0, null, [0, 8]), n(8, 0, [8, 8], null)], closed: false };
    insertNode(bez, 0, 1e-12);
    expect(bez.nodes[0].h_out).toBeNull();
  });

  it("wraps on the closing segment of a closed path", () => {
    const bez = { nodes: [n(0, 0), n(4, 0), n(4, 4)], closed: true };
    expect(insertNode(bez, 2, 0.5)).toBe(3);
    expect(bez.nodes[3]).toMatchObject({ x: 2, y: 2 });
  });
});

describe("deleteNode / joinNodes / splitNode", () => {
  it("deleteNode drops a node but never below two", () => {
    const bez = { nodes: [n(0, 0), n(1, 0), n(2, 0)], closed: false };
    deleteNode(bez, 1);
    expect(bez.nodes.map((q) => q.x)).toEqual([0, 2]);
    deleteNode(bez, 0);
    expect(bez.nodes).toHaveLength(2);
    deleteNode(bez, 99);
    expect(bez.nodes).toHaveLength(2);
  });

  it("joinNodes averages the ends onto the lower index, keeps the other's h_in, closes", () => {
    const bez = {
      nodes: [n(0, 0, null, [1, 1]), n(5, 5), n(2, 2, [1, 3], null)], closed: false,
    };
    joinNodes(bez, 2, 0);
    expect(bez.closed).toBe(true);
    expect(bez.nodes).toHaveLength(2);
    expect(bez.nodes[0]).toMatchObject({ x: 1, y: 1, h_in: [1, 3], h_out: [1, 1] });
  });

  it("splitNode rotates the ring to start at i and duplicates the seam node", () => {
    const bez = {
      nodes: [n(0, 0), n(4, 0, [3, 0], [5, 0]), n(4, 4), n(0, 4)], closed: true,
    };
    splitNode(bez, 1);
    expect(bez.closed).toBe(false);
    expect(bez.nodes.map((q) => [q.x, q.y])).toEqual([[4, 0], [4, 4], [0, 4], [0, 0], [4, 0]]);
    expect(bez.nodes[0].h_in).toBeNull();          // nothing arrives at the new start
    expect(bez.nodes[0].h_out).toEqual([5, 0]);
    expect(bez.nodes[4].h_in).toEqual([3, 0]);     // the closing end keeps its arrival
    expect(bez.nodes[4].h_out).toBeNull();         // nothing leaves it
  });

  it("splitNode leaves an open path or a bad index alone", () => {
    const open = { nodes: [n(0, 0), n(1, 1)], closed: false };
    expect(splitNode(open, 0).nodes).toHaveLength(2);
    const ring = { nodes: [n(0, 0), n(1, 0), n(1, 1)], closed: true };
    expect(splitNode(ring, 7).closed).toBe(true);
  });

  it("split then join on the seam restores a closed path of the same nodes", () => {
    const bez = { nodes: [n(0, 0), n(4, 0), n(4, 4), n(0, 4)], closed: true };
    splitNode(bez, 2);
    joinNodes(bez, 0, bez.nodes.length - 1);
    expect(bez.closed).toBe(true);
    expect(bez.nodes.map((q) => [q.x, q.y])).toEqual([[4, 4], [0, 4], [0, 0], [4, 0]]);
  });
});

describe("setNodeType", () => {
  it("smooth re-aligns h_in off h_out; cusp only records the kind", () => {
    const node = n(0, 0, [-3, 1], [0, 2], CUSP);
    setNodeType(node, CUSP);
    expect(node.h_in).toEqual([-3, 1]);
    setNodeType(node, SMOOTH);
    close(node.h_in, [0, -Math.hypot(-3, 1)]);
    setNodeType(node, SYMMETRIC);
    close(node.h_in, [0, -2]);
  });
});

describe("isRectBezier / rectBezier", () => {
  it("recognises a drawn rectangle", () => {
    expect(isRectBezier(rectBezier(0, 0, 4, 3))).toBe(true);
    expect(rectBezier(0, 0, 4, 3).nodes.map((q) => [q.x, q.y]))
      .toEqual([[0, 0], [4, 0], [4, 3], [0, 3]]);
  });

  it("rejects a parallelogram, a bulged side, an open path and a triangle", () => {
    const para = { closed: true, nodes: [n(0, 0), n(4, 0), n(5, 3), n(1, 3)] };
    expect(isRectBezier(para)).toBe(false);
    const bulge = rectBezier(0, 0, 4, 3);
    bulge.nodes[0].h_out = [2, 2];
    expect(isRectBezier(bulge)).toBe(false);
    expect(isRectBezier({ ...rectBezier(0, 0, 4, 3), closed: false })).toBe(false);
    expect(isRectBezier({ closed: true, nodes: [n(0, 0), n(4, 0), n(4, 3)] })).toBe(false);
    expect(isRectBezier(null)).toBe(false);
  });
});

describe("pen and import helpers", () => {
  it("penToPolyline: straight segments are vertices; a curve has an adaptive count", () => {
    const straight = [{ x: 0, y: 0 }, { x: 10, y: 0 }];
    expect(penToPolyline(straight, 50)).toEqual([[0, 0], [10, 0]]);
    const curved = [{ x: 0, y: 0, hOut: { x: 0, y: 5 } }, { x: 10, y: 0 }];
    expect(penToPolyline(curved, 6)).toHaveLength(1 + 10);     // ceil(10*6/6) = 10 steps
    expect(penToPolyline(curved, 0.1)).toHaveLength(1 + 8);    // floor of 8 steps
    expect(penToPolyline(curved, 1000)).toHaveLength(1 + 64);  // ceiling of 64 steps
    const last = penToPolyline(curved, 6).at(-1);
    close(last, [10, 0]);
  });

  it("penToBezier: dragged nodes become symmetric, clicked nodes stay cusps", () => {
    const b = penToBezier([
      { x: 0, y: 0 },
      { x: 5, y: 0, hIn: [4, 1], hOut: { x: 6, y: -1 } },
    ], true);
    expect(b.closed).toBe(true);
    expect(b.nodes[0]).toEqual(n(0, 0, null, null, CUSP));
    expect(b.nodes[1]).toEqual(n(5, 0, [4, 1], [6, -1], SYMMETRIC));
  });

  it("synthBezier drops the repeated closing point of a closed polyline", () => {
    const b = synthBezier([[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]], false);
    expect(b.closed).toBe(true);
    expect(b.nodes).toHaveLength(4);
    expect(synthBezier([[0, 0], [1, 1]], false).closed).toBe(false);
  });

  it("normalizeBez coerces strings to numbers and defaults the kind", () => {
    const b = normalizeBez({ closed: 1, nodes: [{ x: "1", y: "2", h_out: ["3", "4"] }] });
    expect(b).toEqual({ closed: true, nodes: [n(1, 2, null, [3, 4], CUSP)] });
    expect(normalizeBez(null)).toEqual({ closed: false, nodes: [] });
  });
});

describe("bezCurveDrag", () => {
  it("seeds 1/3 handles on a straight segment and pulls the curve toward the pointer", () => {
    const bez = { nodes: [n(0, 0), n(9, 0)], closed: false };
    bezCurveDrag(bez, { ni0: 0, ni1: 1, t: 0.5 }, 4.5, 3);
    // the curve at t=0.5 now passes through the dragged point
    const [a, b] = bez.nodes;
    close(cubicAt([a.x, a.y], a.h_out, b.h_in, [b.x, b.y], 0.5), [4.5, 3], 6);
  });
});
