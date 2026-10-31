// JS half of the JS<->Python bezier contract (tests/test_ec_bezier_contract.py).
// pytest flattens a set of BezierPaths with litho/bezier.py and writes each
// {name, bez, tol, expected polyline} to the JSON file named in argv[2]; this
// runner flattens the SAME dicts with ec-bezier.js `bezFlatten` and asserts the
// polylines match within tolerance. Without this the on-screen preview can
// silently diverge from what the instrument writes.

import assert from "node:assert";
import fs from "node:fs";
import { bezFlatten } from "../src/afm_litho/web/ec-bezier.js";

const cases = JSON.parse(fs.readFileSync(process.argv[2], "utf8")).cases;
let n = 0;
for (const c of cases) {
  const got = bezFlatten(c.bez, c.tol);
  const exp = c.expected;
  assert.strictEqual(got.length, exp.length,
    `case "${c.name}": JS produced ${got.length} points, Python ${exp.length}`);
  for (let i = 0; i < exp.length; i++) {
    const dx = Math.abs(got[i][0] - exp[i][0]);
    const dy = Math.abs(got[i][1] - exp[i][1]);
    assert.ok(dx <= 1e-6 && dy <= 1e-6,
      `case "${c.name}" point ${i}: JS (${got[i]}) vs Python (${exp[i]})`);
  }
  n++;
}
console.log(`bezier contract: ${n} cases matched litho/bezier.py flatten`);
