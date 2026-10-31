// JS duplicates of Python functions, checked against outputs the Python originals
// produced (fixtures/python_fixtures.json, regenerate with fixtures/generate_fixtures.py).
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { bezFlatten, bezSetHandle, bezMoveNode } from "../../src/afm_litho/web/ec-bezier.js";
import { designFileName } from "../../src/afm_litho/web/ec-design-files.js";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const fx = JSON.parse(readFileSync(here("./fixtures/python_fixtures.json"), "utf8"));

describe("bezier.py <-> ec-bezier.js", () => {
  for (const c of fx.flatten) {
    it(`flatten "${c.name}" reproduces the Python polyline`, () => {
      const js = bezFlatten(JSON.parse(JSON.stringify(c.bez)), c.tol);
      // Python drops consecutive repeats (touching ends); the JS keeps them. Compare
      // after the same drop.
      const dedup = js.filter((p, i) => i === 0 || Math.hypot(p[0] - js[i - 1][0], p[1] - js[i - 1][1]) > 1e-12);
      expect(dedup).toHaveLength(c.points.length);
      dedup.forEach((p, i) => {
        expect(p[0]).toBeCloseTo(c.points[i][0], 9);
        expect(p[1]).toBeCloseTo(c.points[i][1], 9);
      });
    });
  }

  for (const c of fx.set_handle) {
    it(`set_handle ${c.kind}/${c.which} matches Node.set_handle`, () => {
      const node = JSON.parse(JSON.stringify(c.start));
      bezSetHandle(node, c.which, c.target[0], c.target[1]);
      // Python's to_dict rounds to 5 places
      for (const key of ["h_in", "h_out"]) {
        node[key].forEach((v, i) => expect(v).toBeCloseTo(c.result[key][i], 4));
      }
      expect(node.kind).toBe(c.result.kind);
    });
  }

  it("move_to matches Node.move_to", () => {
    const node = JSON.parse(JSON.stringify(fx.move_to.start));
    bezMoveNode(node, ...fx.move_to.to);
    expect([node.x, node.y]).toEqual([fx.move_to.result.x, fx.move_to.result.y]);
    expect(node.h_in).toEqual(fx.move_to.result.h_in);
    expect(node.h_out).toEqual(fx.move_to.result.h_out);
  });
});

describe("design_store.file_name <-> designFileName", () => {
  for (const c of fx.file_name) {
    it(`file name for ${JSON.stringify(c.raw)}`, () => {
      expect(designFileName(c.raw)).toBe(c.name);
    });
  }
});
