// Node tests for the edit cockpit's Layer-1 / import-safe modules, mirroring
// tests/pure_layer.mjs: the modules are real ES exports with no DOM access at
// import time, so this imports them DIRECTLY under node -- no window/document
// fakes, no concatenation harness.
//
// Covered: ec-log's pure formatters (stamp/escapeHtml/num) and ec-api's
// NaN-safe ws parse + the onWs handler registry. The DOM/fetch/WebSocket parts
// (logLine, api, request, connectWs) are exercised in the browser pass, not here.

import assert from "node:assert";
import { stamp, escapeHtml, num } from "../src/afm_litho/web/ec-log.js";
import { parseWsData, onWs, onWsDefault, dispatchWs } from "../src/afm_litho/web/ec-api.js";
import {
  CUSP, SMOOTH, SYMMETRIC, bezMoveNode, bezSetHandle, bezSegments, cubicAt,
  bezFlatten, normalizeBez, synthBezier, penToPolyline, penToBezier, rectBezier,
  setNodeType, deleteNode, insertNode, joinNodes, splitNode, isRectBezier,
} from "../src/afm_litho/web/ec-bezier.js";
import {
  fitTransform, invTransform, pxPerUm, pointInPoly, distToSeg, distToPolyline,
  objBounds, boundsOfObjects, snapRadiusUm, portSnapHit, percentileRange,
  zoomViewAbout, clampZoomFactor, wheelZoomFactor, skewAbout, unionBox, clampViewToBounds,
  boxFieldOk, FALLBACK_XY_RANGE_UM,
} from "../src/afm_litho/web/ec-geom.js";
import { state, subscribe, emit } from "../src/afm_litho/web/ec-state.js";
import {
  designFileName, nameExists, openNoticeLines, saveNoticeLines, listStateText, listRowMeta,
  failureText, succeeded, listRows, shown,
} from "../src/afm_litho/web/ec-design-files.js";
import {
  clampRightWidth, defaultRightWidth, DEFAULT_RIGHT_PX, HALF_WINDOW_PX, isHalfWindow,
  MIN_RENDER_PX, MIN_RIGHT_PX,
} from "../src/afm_litho/web/ec-panes.js";
import {
  polylineLength, bezLength, chiralSampler, chiralLength, objectLength, isClosedKind,
} from "../src/afm_litho/web/ec-measure.js";
import {
  TELE, telemetryLayout, scopeXMap, rangeOver, zScaleLabels, visibleChannels, channelValues,
  TELEMETRY_CHANNELS, barFraction, pushHistory, valueText, HALF, halfImageSide, dragZRange,
  telemetryStaleness, TELEMETRY_STALE_S,
} from "../src/afm_litho/web/ec-render-layout.js";
import { addStroke, writeMeta } from "../src/afm_litho/web/ec-written.js";

let checks = 0;
const ok = (msg) => { checks++; console.log("ok  " + msg); };

// --- 1. ec-log: stamp is HH:MM:SS.mmm --------------------------------------
{
  assert.match(stamp(), /^\d{2}:\d{2}:\d{2}\.\d{3}$/, "stamp shape");
  ok("stamp() is zero-padded HH:MM:SS.mmm");
}

// --- 2. ec-log: escapeHtml escapes & < > only ------------------------------
{
  assert.strictEqual(escapeHtml('<a>&"'), '&lt;a&gt;&amp;"', "escape & < >");
  assert.strictEqual(escapeHtml(42), "42", "non-string coerced");
  ok("escapeHtml escapes &,<,> (leaves quotes) and coerces to string");
}

// --- 3. ec-log: num -- constant with the inline formula --------------------
{
  assert.strictEqual(num(0), "0", "zero");
  assert.strictEqual(num(3), "3", "integer >=1 strips .000");
  assert.strictEqual(num(1.23456), "1.235", "abs>=1 -> 3 dp");
  assert.strictEqual(num(0.5), "0.5", "abs<1 -> 4 dp, trailing zeros stripped");
  assert.strictEqual(num(0.00012), "0.0001", "abs<1 -> 4 dp");
  assert.strictEqual(num("x"), "x", "non-number passes through as String()");
  ok("num: zero, integer strip, >=1 three-dp, <1 four-dp, non-number passthrough");
}

// --- 4. ec-api: parseWsData -- plain JSON, then NaN/Infinity retry ---------
{
  assert.deepStrictEqual(parseWsData('{"a":1}'), { ok: true, value: { a: 1 } },
                         "plain json");
  // Python json (allow_nan=True) emits bare NaN/Infinity; the retry nulls them.
  assert.deepStrictEqual(parseWsData('{"a":NaN}'), { ok: true, value: { a: null } },
                         "NaN -> null");
  assert.deepStrictEqual(parseWsData('{"a":Infinity,"b":-Infinity}'),
                         { ok: true, value: { a: null, b: null } },
                         "Infinity and -Infinity -> null");
  assert.deepStrictEqual(parseWsData("not json {"), { ok: false },
                         "unparseable -> {ok:false}");
  ok("parseWsData: plain, NaN/Infinity retry, and the unparseable failure shape");
}

// --- 5. ec-api: onWs / onWsDefault / dispatchWs registry -------------------
{
  const seen = [];
  onWs("line", (m) => seen.push(["line", m.n]));
  onWs("drift", (m) => seen.push(["drift", m.n]));
  onWsDefault((m) => seen.push(["default", m.type]));

  dispatchWs({ type: "line", n: 1 });
  dispatchWs({ type: "drift", n: 2 });
  dispatchWs({ type: "mystery", n: 3 });   // no handler -> default

  assert.deepStrictEqual(seen, [["line", 1], ["drift", 2], ["default", "mystery"]],
                         "each type routes to its handler; unknown -> default");

  // last registration wins (one handler per type)
  const seen2 = [];
  onWs("line", (m) => seen2.push(["A", m.n]));
  onWs("line", (m) => seen2.push(["B", m.n]));
  dispatchWs({ type: "line", n: 9 });
  assert.deepStrictEqual(seen2, [["B", 9]], "re-registering a type replaces it");
  ok("onWs registry routes by type, falls back to onWsDefault, last registration wins");
}

// --- 6. rectBezier contract: the shape the rect tool attaches on commit -----
// Now imported DIRECTLY from ec-bezier.js (Layer 1, import-safe under node), not
// a reference mirror. It pins the exact dict shape BezierPath.as_rect() requires
// -- closed, exactly 4 CUSP nodes with null handles, corners axis-aligned. The
// cross-language recognition (this dict -> as_rect -> fill_zigzag) is guarded in
// tests/test_rect_fill.py.
{
  const b = rectBezier(0, 0, 8, 6);
  assert.strictEqual(b.closed, true, "closed rectangle bezier");
  assert.strictEqual(b.nodes.length, 4, "exactly 4 nodes (as_rect requires 4)");
  assert.deepStrictEqual(b.nodes.map((n) => [n.x, n.y]),
                         [[0, 0], [8, 0], [8, 6], [0, 6]], "axis-aligned corner order");
  for (const n of b.nodes) {
    assert.strictEqual(n.kind, CUSP, "corner is a cusp");
    assert.strictEqual(n.h_in, null, "no in handle");
    assert.strictEqual(n.h_out, null, "no out handle");
  }
  ok("rectBezier: closed 4-node CUSP rect, null handles, axis-aligned corners");
}

// --- 7. penToBezier contract: the pen emits closed=true only on close --------
// Now imported DIRECTLY from ec-bezier.js. Pins the behaviour Slice 2 changed:
// penToBezier(nodes) finishes OPEN (closed:false), penToBezier(nodes, true)
// closes (closed:true). The cross-language recognition of a closed funnel/region
// is guarded in tests/test_pen_close_fill.py.
{
  const nodes = [
    { x: 0, y: 0, hIn: null, hOut: null },
    { x: 10, y: 0, hIn: null, hOut: { x: 11, y: 5 } },
    { x: 5, y: 15, hIn: { x: 8, y: 12 }, hOut: { x: 2, y: 12 } },
  ];
  // finish OPEN: default and explicit-false both stay open
  assert.strictEqual(penToBezier(nodes).closed, false, "finish-open -> closed:false");
  assert.strictEqual(penToBezier(nodes, false).closed, false, "explicit false stays open");
  // close: closed:true, and NO duplicate trailing point (node count unchanged)
  const closed = penToBezier(nodes, true);
  assert.strictEqual(closed.closed, true, "close -> closed:true");
  assert.strictEqual(closed.nodes.length, 3, "close adds no duplicate node");
  // handles carry through as [x,y] arrays (the BezierPath.to_dict shape)
  assert.deepStrictEqual(closed.nodes[1].h_out, [11, 5], "hOut -> [x,y]");
  assert.strictEqual(closed.nodes[0].h_out, null, "plain cusp stays handle-less");
  assert.strictEqual(closed.nodes[1].kind, SYMMETRIC, "a handled node is symmetric");
  assert.strictEqual(closed.nodes[0].kind, CUSP, "a handle-less node is a cusp");
  ok("penToBezier: closed:false on finish-open, closed:true on close, no dup point");
}

// --- 8. ec-bezier node model: move_to carries the handles ------------------
{
  const n = { x: 1, y: 1, h_in: [0, 1], h_out: [2, 1], kind: SYMMETRIC };
  bezMoveNode(n, 4, 5);              // dx=+3 dy=+4
  assert.deepStrictEqual([n.x, n.y], [4, 5], "anchor moved");
  assert.deepStrictEqual(n.h_in, [3, 5], "h_in travelled with the anchor");
  assert.deepStrictEqual(n.h_out, [5, 5], "h_out travelled with the anchor");
  const bare = { x: 0, y: 0, h_in: null, h_out: null, kind: CUSP };
  bezMoveNode(bare, 2, 3);
  assert.strictEqual(bare.h_in, null, "null handle stays null on move");
  ok("bezMoveNode: anchor + present handles translate, null handles stay null");
}

// --- 9. ec-bezier continuity rules: cusp / smooth / symmetric --------------
{
  // cusp: opposite handle is untouched
  const c = { x: 0, y: 0, h_in: [-1, 0], h_out: [1, 0], kind: CUSP };
  bezSetHandle(c, "out", 0, 2);
  assert.deepStrictEqual(c.h_out, [0, 2], "cusp out set");
  assert.deepStrictEqual(c.h_in, [-1, 0], "cusp leaves the opposite handle put");
  // symmetric: opposite handle mirrors (equal length, anti-parallel)
  const s = { x: 0, y: 0, h_in: [-1, 0], h_out: [1, 0], kind: SYMMETRIC };
  bezSetHandle(s, "out", 0, 2);     // out now points +y, len 2
  assert.deepStrictEqual(s.h_out, [0, 2], "symmetric out set");
  assert.ok(Math.hypot(s.h_in[0] - 0, s.h_in[1] - (-2)) < 1e-9,
            "symmetric in mirrors to [0,-2]");
  // smooth: opposite handle stays collinear but KEEPS its own length
  const m = { x: 0, y: 0, h_in: [-3, 0], h_out: [1, 0], kind: SMOOTH };
  bezSetHandle(m, "out", 0, 2);     // out points +y; in keeps length 3, points -y
  assert.ok(Math.hypot(m.h_in[0] - 0, m.h_in[1] - (-3)) < 1e-9,
            "smooth in stays length 3, collinear -> [0,-3]");
  ok("bezSetHandle: cusp independent, symmetric mirrors, smooth keeps length");
}

// --- 10. ec-bezier flatten: a straight segment stays its endpoints ---------
{
  // two cusp nodes, no handles -> the flattened polyline is just the two anchors
  const line = { closed: false, nodes: [
    { x: 0, y: 0, h_in: null, h_out: null, kind: CUSP },
    { x: 10, y: 0, h_in: null, h_out: null, kind: CUSP },
  ] };
  assert.deepStrictEqual(bezFlatten(line, 0.004), [[0, 0], [10, 0]],
                         "a straight segment flattens to its two endpoints");
  // segments() of an open 2-node path yields exactly one segment
  const segs = bezSegments(line);
  assert.strictEqual(segs.length, 1, "open 2-node path -> 1 segment");
  // cubicAt endpoints
  assert.deepStrictEqual(cubicAt([0, 0], [1, 0], [2, 0], [3, 0], 0), [0, 0], "cubicAt t=0");
  assert.deepStrictEqual(cubicAt([0, 0], [1, 0], [2, 0], [3, 0], 1), [3, 0], "cubicAt t=1");
  // a curve refines: a symmetric arc flattens to more than 2 points, all within
  // the tolerance band of the true cubic's convex hull (endpoints exact)
  const arc = { closed: false, nodes: [
    { x: 0, y: 0, h_in: null, h_out: [0, 4], kind: SMOOTH },
    { x: 8, y: 0, h_in: [8, 4], h_out: null, kind: SMOOTH },
  ] };
  const flat = bezFlatten(arc, 0.01);
  assert.ok(flat.length > 2, "a curved segment refines beyond its endpoints");
  assert.deepStrictEqual(flat[0], [0, 0], "first flattened point is the start anchor");
  assert.deepStrictEqual(flat[flat.length - 1], [8, 0], "last flattened point is the end anchor");
  ok("bezFlatten/bezSegments/cubicAt: straight stays, curve refines, ends exact");
}

// --- 11. ec-bezier normalizeBez / synthBezier -------------------------------
{
  // normalize copies handles to arrays and defaults the kind to cusp
  const nb = normalizeBez({ closed: 1, nodes: [
    { x: "1", y: "2", h_out: ["3", "4"] }, { x: 5, y: 6 },
  ] });
  assert.strictEqual(nb.closed, true, "closed coerced to boolean");
  assert.deepStrictEqual(nb.nodes[0], { x: 1, y: 2, h_in: null, h_out: [3, 4], kind: CUSP },
                         "numbers coerced, missing handle null, kind default cusp");
  // synthBezier from an OPEN polyline: cusp nodes, not closed
  const sb = synthBezier([[0, 0], [1, 1], [2, 0]], false);
  assert.strictEqual(sb.closed, false, "open polyline stays open");
  assert.strictEqual(sb.nodes.length, 3, "one node per point");
  assert.ok(sb.nodes.every((n) => n.kind === CUSP && n.h_in === null && n.h_out === null),
            "synthesized nodes are handle-less cusps");
  // a closed polyline whose last==first drops the duplicate trailing point
  const sc = synthBezier([[0, 0], [2, 0], [2, 2], [0, 0]], true);
  assert.strictEqual(sc.closed, true, "closed flag honoured");
  assert.strictEqual(sc.nodes.length, 3, "duplicate closing point dropped");
  ok("normalizeBez coerces + defaults; synthBezier makes cusp nodes, drops closing dup");
}

// --- 12. ec-bezier penToPolyline: pxPerUm is an explicit argument -----------
// It was DOM-free apart from the adaptive segment count; that now comes in as an
// argument (p2u = pixels-per-µm) so the module imports under node.
{
  const nodes = [{ x: 0, y: 0, hIn: null, hOut: null }, { x: 10, y: 0, hIn: null, hOut: null }];
  // straight (no handles) -> the two anchors, regardless of p2u
  assert.deepStrictEqual(penToPolyline(nodes, 20), [[0, 0], [10, 0]],
                         "a handle-less pen segment stays its two anchors");
  // a curved segment's point count scales with p2u (denser zoom -> more points)
  const curved = [
    { x: 0, y: 0, hIn: null, hOut: { x: 0, y: 6 } },
    { x: 12, y: 0, hIn: { x: 12, y: 6 }, hOut: null },
  ];
  const coarse = penToPolyline(curved, 1).length;
  const fine = penToPolyline(curved, 40).length;
  assert.ok(fine > coarse, "a higher pxPerUm yields a finer pen polyline");
  ok("penToPolyline: straight stays two anchors; curve density follows pxPerUm arg");
}

// --- 13. ec-geom view transform: invTransform o fitTransform == identity -----
// The whole "everything lands in register" guarantee rests on invTransform being
// the exact inverse of fitTransform for the same view + canvas size.
{
  const view = { x0: -3, y0: 2, w: 15, h: 12 };
  const px = 800, py = 600;                 // a non-square canvas (aspect matters)
  const tf = fitTransform(view, px, py);
  const inv = invTransform(view, px, py);
  for (const p of [[-3, 2], [0, 0], [4.5, 7.25], [12, 14], [-1.2, 3.8]]) {
    const back = inv(...tf(p[0], p[1]));
    assert.ok(Math.hypot(back[0] - p[0], back[1] - p[1]) < 1e-9,
      `round-trip ${p} -> ${back}`);
  }
  // y flips: a higher µm-y maps to a SMALLER pixel-y (screen-up == micron-up)
  const [, ylo] = tf(0, 0), [, yhi] = tf(0, 10);
  assert.ok(yhi < ylo, "higher micron-y is nearer the top of the canvas");
  // pxPerUm is the fit scale and the min over the two axes (aspect preserved)
  assert.strictEqual(pxPerUm(view, px, py), Math.min(px / view.w, py / view.h),
    "pxPerUm is the min per-axis scale");
  ok("ec-geom fit/inv: exact inverse, y flips up, pxPerUm is the fit scale");
}

// --- 14. ec-geom hit-test cores: pointInPoly / distToSeg / distToPolyline -----
{
  const sq = [[0, 0], [10, 0], [10, 10], [0, 10]];
  assert.strictEqual(pointInPoly(sq, 5, 5), true, "centre is inside");
  assert.strictEqual(pointInPoly(sq, -1, 5), false, "left of the square is outside");
  assert.strictEqual(pointInPoly(sq, 15, 5), false, "right of the square is outside");
  // distance to a horizontal segment: perpendicular, and clamped past the ends
  assert.ok(Math.abs(distToSeg(5, 3, 0, 0, 10, 0) - 3) < 1e-12, "perp distance");
  assert.ok(Math.abs(distToSeg(-4, 0, 0, 0, 10, 0) - 4) < 1e-12, "clamped before start");
  // polyline distance is the nearest segment
  const poly = [[0, 0], [10, 0], [10, 10]];
  assert.ok(Math.abs(distToPolyline(poly, 10, 5) - 0) < 1e-12, "on the second segment");
  assert.ok(Math.abs(distToPolyline(poly, 5, 2) - 2) < 1e-12, "2 above the first segment");
  ok("ec-geom hit cores: pointInPoly in/out, distToSeg perp+clamp, distToPolyline nearest");
}

// --- 15. ec-geom objBounds / boundsOfObjects --------------------------------
{
  assert.deepStrictEqual(objBounds({ points_um: [[1, 2], [4, 2], [4, 6]] }),
                         { minx: 1, miny: 2, maxx: 4, maxy: 6 }, "tight bounds");
  assert.strictEqual(objBounds({ points_um: [] }), null, "no extent -> null");
  assert.strictEqual(objBounds({ points_um: [[3, 3]] }), null, "a single point -> null");
  // boundsOfObjects pads by 8% + 0.5 µm around the union; empty -> a 10x10 box
  const b = boundsOfObjects([{ points_um: [[0, 0], [10, 0]] },
                             { points_um: [[0, 0], [0, 4]] }]);
  assert.ok(b.x0 < 0 && b.y0 < 0 && b.w > 10 && b.h > 4, "union padded outward");
  assert.deepStrictEqual(boundsOfObjects([]), { x0: 0, y0: 0, w: 10, h: 10 },
                         "nothing to frame -> 10x10 fallback");
  ok("ec-geom objBounds tight/null, boundsOfObjects padded union + empty fallback");
}

// --- 16. ec-state: the shared-model defaults (the values the monolith declared) -
// These pin the boot state every tab reads. A regression here (a wrong default,
// or a field dropped in a carve) would silently change what the cockpit shows
// before any user action.
{
  assert.strictEqual(state.selected, -1, "no selection at boot");
  assert.strictEqual(state.view, null, "no view until framed");
  assert.strictEqual(state.scanFrame, null, "no scan frame until one loads");
  assert.strictEqual(state.scanImg, null, "no scan image until built");
  assert.strictEqual(state.lastScanCfg, null, "no last scan config at boot");
  assert.strictEqual(state.designImported, false, "mock objects until an import");
  assert.strictEqual(state.sourceLabel, "mock", "source label mirrors the #src chip");
  assert.strictEqual(state.currentTab, "scan", "scan is the default tab");
  assert.strictEqual(state.activeTool, "select", "select is the default tool");
  assert.deepStrictEqual(state.SCAN_FRAME, { x0: 0, y0: 0, w: 15, h: 12 },
                         "default scan frame is the 15x12 µm placeholder");
  assert.deepStrictEqual(state.ports, [], "no ports at boot");
  assert.strictEqual(state.selectedPort, null, "no selected port");
  assert.deepStrictEqual(state.litho,
    { writing: false, name: null, frac: 0, tip: null, live: [], written: [], tipTrail: [],
      writes: [], showWritten: true },
    "litho live-state is idle/empty at boot");
  assert.deepStrictEqual(state.driftState,
    { enabled: false, has_reference: false, total: [0, 0] }, "drift state is off/no-ref");
  // the object list boots with the 5 mock objects, real-backend-shaped
  assert.strictEqual(state.objects.length, 5, "five mock objects at boot");
  assert.deepStrictEqual(state.objects.map((o) => o.name),
                         ["wire-1", "rect-1", "funnel-1", "wire-2", "rect-2"], "mock names");
  assert.ok(state.objects.every((o) => Array.isArray(o.points_um) && o.points_um.length >= 2),
            "each mock object carries a points_um polyline");
  // properties are writable through the const binding (the whole reason for the object)
  state.selected = 3; assert.strictEqual(state.selected, 3, "state.X is reassignable");
  state.selected = -1;                                   // restore for any later use
  ok("ec-state defaults: selection/view/scan/design/tab/tool/ports/litho/drift + writable");
}

// --- 17. ec-state: the tiny event bus (subscribe / emit / unsubscribe) --------
// The bus fans out to every subscriber of a topic, forwards all emit args, and
// the subscribe() return value unsubscribes. Used only where a state change
// already triggered a redraw, so timing never changes -- only who is notified.
{
  const seen = [];
  const off1 = subscribe("redraw", (a, b) => seen.push(["one", a, b]));
  subscribe("redraw", (a) => seen.push(["two", a]));
  emit("redraw", 1, 2);
  assert.deepStrictEqual(seen, [["one", 1, 2], ["two", 1]],
                         "both subscribers fire, all args forwarded");
  // a topic with no subscribers is a harmless no-op
  emit("nobody-listening", 9);
  // unsubscribe removes just that listener
  off1();
  seen.length = 0;
  emit("redraw", 7);
  assert.deepStrictEqual(seen, [["two", 7]], "unsubscribe drops only its own handler");
  ok("ec-state bus: fan-out, arg forwarding, empty-topic no-op, unsubscribe");
}

// --- 18. ec-bezier setNodeType: re-derives handles per continuity ----------
// The Type buttons. cusp leaves the two handles independent; symmetric mirrors
// the opposite handle off h_out; smooth keeps the opposite handle's own length
// but makes it collinear. Matches bezSetHandle's rules (litho/bezier.py set_handle).
{
  const c = { x: 0, y: 0, h_in: [-3, 1], h_out: [2, 0], kind: SMOOTH };
  setNodeType(c, CUSP);
  assert.strictEqual(c.kind, CUSP, "kind set to cusp");
  assert.deepStrictEqual(c.h_in, [-3, 1], "cusp leaves handles untouched");
  // symmetric: h_in mirrors h_out (anti-parallel, equal length)
  const s = { x: 0, y: 0, h_in: [-3, 1], h_out: [2, 0], kind: CUSP };
  setNodeType(s, SYMMETRIC);
  assert.strictEqual(s.kind, SYMMETRIC, "kind set to symmetric");
  assert.ok(Math.hypot(s.h_in[0] - (-2), s.h_in[1] - 0) < 1e-9, "symmetric in mirrors h_out to [-2,0]");
  // smooth: h_in stays its own length (sqrt(10)) but goes collinear (-x)
  const m = { x: 0, y: 0, h_in: [-3, 1], h_out: [2, 0], kind: CUSP };
  setNodeType(m, SMOOTH);
  assert.ok(Math.abs(Math.hypot(m.h_in[0], m.h_in[1]) - Math.hypot(3, 1)) < 1e-9,
            "smooth in keeps its length");
  assert.ok(Math.hypot(m.h_in[0] - (-Math.hypot(3, 1)), m.h_in[1] - 0) < 1e-9,
            "smooth in is collinear (-x) with h_out");
  ok("setNodeType: cusp independent, symmetric mirrors, smooth keeps length");
}

// --- 19. ec-bezier deleteNode: dumb delete, neighbours reconnect ------------
{
  const bez = { closed: false, nodes: [
    { x: 0, y: 0, h_in: null, h_out: null, kind: CUSP },
    { x: 5, y: 5, h_in: null, h_out: null, kind: CUSP },
    { x: 10, y: 0, h_in: null, h_out: null, kind: CUSP },
  ] };
  deleteNode(bez, 1);
  assert.strictEqual(bez.nodes.length, 2, "node count -1");
  assert.deepStrictEqual(bez.nodes.map((n) => [n.x, n.y]), [[0, 0], [10, 0]],
                         "the two neighbours remain, now directly joined");
  // guard: never drop below two nodes
  const two = { closed: false, nodes: [
    { x: 0, y: 0, h_in: null, h_out: null, kind: CUSP },
    { x: 1, y: 0, h_in: null, h_out: null, kind: CUSP },
  ] };
  deleteNode(two, 0);
  assert.strictEqual(two.nodes.length, 2, "delete refused at the two-node floor");
  ok("deleteNode: dumb delete drops the node and reconnects; floors at two nodes");
}

// --- 20. ec-bezier insertNode: the point lands ON the curve, shape preserved -
// de Casteljau split at t: the new anchor equals the curve point at t, node count
// +1, and re-flattening the split path matches the original curve (exact split).
{
  const arc = { closed: false, nodes: [
    { x: 0, y: 0, h_in: null, h_out: [0, 6], kind: SMOOTH },
    { x: 12, y: 0, h_in: [12, 6], h_out: null, kind: SMOOTH },
  ] };
  const before = bezFlatten(arc, 0.005);
  const seg = bezSegments(arc)[0];                 // [p0,c1,c2,p3,i0,i1]
  const t = 0.37;
  const expected = cubicAt(seg[0], seg[1], seg[2], seg[3], t);
  const idx = insertNode(arc, 0, t);
  assert.strictEqual(idx, 1, "insert returns the new node's index");
  assert.strictEqual(arc.nodes.length, 3, "node count +1");
  assert.ok(Math.hypot(arc.nodes[1].x - expected[0], arc.nodes[1].y - expected[1]) < 1e-9,
            "the inserted anchor is the curve point at t");
  const after = bezFlatten(arc, 0.005);
  // the split reproduces the original curve exactly: every 'after' point lies on
  // the 'before' polyline (sample endpoints agree, curve unchanged)
  assert.deepStrictEqual(after[0], before[0], "start unchanged");
  assert.deepStrictEqual(after[after.length - 1], before[before.length - 1], "end unchanged");
  // each 'after' point lies on the original curve: within tolerance of the
  // 'before' POLYLINE (not merely near a 'before' sample point -- the two
  // subdivisions sample at different parameters along the same curve).
  assert.ok(after.every((p) => distToPolyline(before, p[0], p[1]) < 0.02),
            "the split curve still lies on the original curve");
  // inserting on a straight segment keeps it straight (handles collapse to null)
  const line = { closed: false, nodes: [
    { x: 0, y: 0, h_in: null, h_out: null, kind: CUSP },
    { x: 10, y: 0, h_in: null, h_out: null, kind: CUSP },
  ] };
  insertNode(line, 0, 0.5);
  assert.deepStrictEqual([line.nodes[1].x, line.nodes[1].y], [5, 0], "midpoint of a straight seg");
  assert.strictEqual(line.nodes[0].h_out, null, "straight side stays handle-less");
  ok("insertNode: anchor on the curve, count +1, split is exact, straight stays straight");
}

// --- 21. ec-bezier joinNodes: two ends -> one closed path -------------------
{
  const open = { closed: false, nodes: [
    { x: 0, y: 0, h_in: null, h_out: [1, 2], kind: CUSP },
    { x: 5, y: 5, h_in: null, h_out: null, kind: CUSP },
    { x: 10, y: 0, h_in: [9, 2], h_out: null, kind: CUSP },
  ] };
  joinNodes(open, 0, 2);
  assert.strictEqual(open.nodes.length, 2, "two ends merged -> one fewer node");
  assert.strictEqual(open.closed, true, "the path is now closed");
  assert.deepStrictEqual([open.nodes[0].x, open.nodes[0].y], [5, 0],
                         "survivor sits at the average of the two ends");
  assert.deepStrictEqual(open.nodes[0].h_in, [9, 2], "survivor takes the other end's arriving handle");
  ok("joinNodes: averages the two ends onto one node, carries h_in, closes the path");
}

// --- 22. ec-bezier splitNode: open a closed path (inverse of join) ----------
{
  const closed = { closed: true, nodes: [
    { x: 0, y: 0, h_in: null, h_out: null, kind: CUSP },
    { x: 10, y: 0, h_in: null, h_out: null, kind: CUSP },
    { x: 5, y: 8, h_in: null, h_out: null, kind: CUSP },
  ] };
  splitNode(closed, 1);
  assert.strictEqual(closed.closed, false, "the path is opened");
  assert.strictEqual(closed.nodes.length, 4, "the split node becomes two coincident ends (+1)");
  assert.deepStrictEqual([closed.nodes[0].x, closed.nodes[0].y], [10, 0], "opens starting at node 1");
  assert.deepStrictEqual([closed.nodes[3].x, closed.nodes[3].y], [10, 0], "and ending on its coincident twin");
  // already-open path is untouched
  const openp = { closed: false, nodes: [
    { x: 0, y: 0, h_in: null, h_out: null, kind: CUSP },
    { x: 1, y: 0, h_in: null, h_out: null, kind: CUSP },
  ] };
  splitNode(openp, 0);
  assert.strictEqual(openp.nodes.length, 2, "an already-open path is left alone");
  ok("splitNode: opens a closed path at the node into two coincident ends; no-op when open");
}

// --- 23. ec-bezier isRectBezier: the as_rect signature (rect vs not) --------
{
  assert.strictEqual(isRectBezier(rectBezier(0, 0, 8, 6)), true, "an axis-aligned rect is a rect");
  // a rotated rectangle (right angles, four straight sides) still counts
  const rot = { closed: true, nodes: [
    { x: 0, y: 0, h_in: null, h_out: null, kind: CUSP },
    { x: 3, y: 3, h_in: null, h_out: null, kind: CUSP },
    { x: 1, y: 5, h_in: null, h_out: null, kind: CUSP },
    { x: -2, y: 2, h_in: null, h_out: null, kind: CUSP },
  ] };
  assert.strictEqual(isRectBezier(rot), true, "a rotated rectangle is a rect");
  // an open 4-node path is not a rect; a triangle is not; a funnel is not
  assert.strictEqual(isRectBezier({ closed: false, nodes: rectBezier(0, 0, 8, 6).nodes }), false,
                     "an open outline is not a rect");
  const tri = { closed: true, nodes: [
    { x: 0, y: 0, h_in: null, h_out: null, kind: CUSP },
    { x: 8, y: 0, h_in: null, h_out: null, kind: CUSP },
    { x: 4, y: 6, h_in: null, h_out: null, kind: CUSP },
  ] };
  assert.strictEqual(isRectBezier(tri), false, "a triangle (3 nodes) is not a rect");
  // a 4-node closed shape whose corners are NOT right angles is not a rect
  const skew = { closed: true, nodes: [
    { x: 0, y: 0, h_in: null, h_out: null, kind: CUSP },
    { x: 8, y: 0, h_in: null, h_out: null, kind: CUSP },
    { x: 10, y: 6, h_in: null, h_out: null, kind: CUSP },
    { x: 0, y: 6, h_in: null, h_out: null, kind: CUSP },
  ] };
  assert.strictEqual(isRectBezier(skew), false, "a non-right-angled quad is not a rect");
  ok("isRectBezier: axis-aligned + rotated rects true; open/triangle/skew-quad false");
}

// --- 24. ec-geom snap-to-bind: PURE PROXIMITY (EDIT_TOOLS §4) -----------------
// The pure geometry behind snap-to-bind. It is now PURE PROXIMITY, matching the
// close-path node snap: within the interaction RADIUS of the port centre -> bind,
// regardless of approach angle. There is NO directional gate. The radius is passed
// IN (µm) -- the call site sizes it to ~14 screen px so snapping is reachable by
// hand at any zoom -- and portSnapHit stays DOM-free. When the radius arg is
// omitted portSnapHit falls back to snapRadiusUm(width).
{
  // snapRadiusUm is still the DOM-free default: max(width/2, 0.5 µm).
  assert.strictEqual(snapRadiusUm(0.6), 0.5, "half-width below floor -> 0.5 µm floor");
  assert.strictEqual(snapRadiusUm(3.0), 1.5, "half-width above floor -> width/2");
  assert.strictEqual(snapRadiusUm(0), 0.5, "no width -> the 0.5 µm floor");
  assert.strictEqual(snapRadiusUm(undefined), 0.5, "absent width -> the floor");

  // a port at the origin (direction_deg is irrelevant now -- pure proximity)
  const port = { id: "e1", position_um: [0, 0], direction_deg: 0, width_um: 1 };
  // within the passed radius -> snaps, from ANY direction of approach.
  assert.strictEqual(portSnapHit(port, 0.4, 0, 0.5), true,
    "inside the passed radius (+x side) -> snaps");
  assert.strictEqual(portSnapHit(port, -0.4, 0, 0.5), true,
    "inside the passed radius (-x side) -> snaps regardless of direction");
  assert.strictEqual(portSnapHit(port, 0, 0.4, 0.5), true,
    "inside the passed radius (+y side) -> snaps regardless of direction");
  assert.strictEqual(portSnapHit(port, 0.4, 0.4, 0.5), false,
    "0.57 µm diagonal is outside a 0.5 µm radius -> no snap");
  // proximity is governed by the PASSED radius: same point, big radius reaches it,
  // tiny radius does not (a 1 µm pad no longer forces a 0.5 µm reach)
  assert.strictEqual(portSnapHit(port, 0.9, 0, 1.0), true,
    "0.9 µm away with a 1.0 µm radius -> reaches");
  assert.strictEqual(portSnapHit(port, 0.9, 0, 0.3), false,
    "0.9 µm away with a 0.3 µm radius -> out of reach");
  // on-centre always snaps
  assert.strictEqual(portSnapHit(port, 0, 0, 0.5), true, "on the port centre -> snaps");
  // omitting the radius arg -> back-compat fallback to snapRadiusUm(width) = 0.5
  assert.strictEqual(portSnapHit(port, 0.4, 0), true,
    "no radius arg -> width-derived default (0.5 µm) reaches 0.4 µm");
  assert.strictEqual(portSnapHit(port, 0.6, 0), false,
    "no radius arg -> beyond the 0.5 µm default -> no snap");
  // a port with no position never snaps
  assert.strictEqual(portSnapHit({ id: "x" }, 0, 0, 0.5), false, "no position_um -> no snap");
  ok("ec-geom snap: pure proximity (any direction), passed-in radius governs reach, width fallback");
}

// --- 25. ec-geom percentileRange: robust greyscale display range -------------
// The scan-image greyscale maps this 2..98 percentile range instead of raw
// min->max, so the error channel's edge spikes / dropouts stop blowing out the
// range (docs/TECH_DEBT.md #1). Nearest-rank over the sorted finite values.
{
  // 0..100 inclusive -> 101 values, index = round(p/100 * 100) = p. So p2 -> 2,
  // p98 -> 98, exactly. Spikes outside that band are ignored by the clip.
  const ramp = Array.from({ length: 101 }, (_, i) => i);
  assert.deepStrictEqual(percentileRange(ramp, 2, 98), [2, 98],
                         "nearest-rank picks the 2nd and 98th values of a 0..100 ramp");
  // a huge spike + a deep dropout do NOT move p2/p98 (that's the whole point):
  // 101 in-band values (0..100) plus two outliers -> n=103, indices
  // round(.02*102)=2 and round(.98*102)=100 -> still the 3rd and 101st values.
  const withOutliers = [-1e6, ...ramp, 1e9];
  assert.deepStrictEqual(percentileRange(withOutliers, 2, 98), [1, 99],
                         "outliers barely shift the range -- not blown out");
  // NaN / null / Infinity are ignored (a frame has NaN gaps)
  assert.deepStrictEqual(percentileRange([0, NaN, null, Infinity, -Infinity, 100], 0, 100),
                         [0, 100], "non-finite values are dropped before ranking");
  // all-NaN / empty -> null (documented fallback: caller supplies a safe range)
  assert.strictEqual(percentileRange([NaN, null, undefined], 2, 98), null,
                     "no finite values -> null");
  assert.strictEqual(percentileRange([], 2, 98), null, "empty -> null");
  // a flat frame (p2 == p98) widens by +-epsilon so callers never divide by zero
  const [flo, fhi] = percentileRange([5, 5, 5, 5], 2, 98);
  assert.ok(flo < 5 && fhi > 5 && (fhi - flo) < 1e-3, "flat frame widens by tiny epsilon around 5");
  // a flat frame at exactly zero uses the absolute-epsilon branch, still non-degenerate
  const [zlo, zhi] = percentileRange([0, 0, 0], 2, 98);
  assert.ok(zlo < 0 && zhi > 0 && (zhi - zlo) < 1e-6, "flat zero frame widens by absolute epsilon");
  ok("percentileRange: nearest-rank p2/p98, outliers clamped, non-finite dropped, empty->null, flat->epsilon");
}

// --- ec-geom zoom math: the point under the pointer stays fixed; limits hold ----
{
  const v = { x0: -3, y0: 1, w: 20, h: 10 };
  const W = 800, H = 600;
  const [ux, uy] = invTransform(v, W, H)(123, 456);
  for (const f of [0.5, 0.9, 1.7]) {
    const z = zoomViewAbout(v, ux, uy, f);
    const [px, py] = fitTransform(z, W, H)(ux, uy);
    assert.ok(Math.abs(px - 123) < 1e-9 && Math.abs(py - 456) < 1e-9, `fixed point f=${f}`);
    assert.ok(Math.abs(z.w - 20 * f) < 1e-12 && Math.abs(z.h - 10 * f) < 1e-12, "size scales");
  }
  assert.strictEqual(clampZoomFactor(v, 0.5, 0.1, 100), 0.5, "inside the range: unchanged");
  assert.ok(Math.abs(clampZoomFactor(v, 1e-6, 0.1, 100) - 0.1 / 20) < 1e-15, "min clamps to 0.1 um");
  assert.strictEqual(clampZoomFactor(v, 10, 0.1, 100), 5, "max clamps to 100 um");
  const big = { x0: 0, y0: 0, w: 300, h: 300 };
  assert.strictEqual(clampZoomFactor(big, 1.2, 0.1, 100), 1, "past the max: no further out");
  assert.strictEqual(clampZoomFactor(big, 0.8, 0.1, 100), 0.8, "past the max: may come back");
  assert.ok(wheelZoomFactor(100, 0, 0.002) > 1 && wheelZoomFactor(-100, 0, 0.002) < 1,
            "wheel down zooms out, up zooms in");
  assert.ok(Math.abs(wheelZoomFactor(3, 1, 0.002) - wheelZoomFactor(48, 0, 0.002)) < 1e-12,
            "line deltas count 16 px each");
  ok("ec-geom zoomViewAbout/clampZoomFactor/wheelZoomFactor");
}

// --- ec-design-files: the filed name, and the notice after an open ----------
{
  assert.strictEqual(designFileName("  device A.design.json "), "device_A");
  assert.strictEqual(designFileName("dev.1"), "dev.1", "a dot inside the name is kept");
  assert.strictEqual(designFileName("a/b\\c:d"), "a_b_c_d", "separators and colons become _");
  assert.strictEqual(designFileName("a/"), "a_");
  // the same cases as design_store.file_name
  assert.strictEqual(designFileName("a/b"), "a_b");
  assert.strictEqual(designFileName("C:evil"), "C_evil");
  assert.strictEqual(designFileName("../up"), ".._up");
  assert.strictEqual(designFileName("x.DESIGN.JSON"), "x");
  assert.strictEqual(designFileName(".design.json"), "");
  assert.strictEqual(designFileName("dev-1_b"), "dev-1_b");
  assert.strictEqual(designFileName(null), "");
  // non-ASCII, each expected value taken from design_store.file_name
  assert.strictEqual(designFileName("détail é"), "détail_é");
  assert.strictEqual(designFileName("设计 1"), "设计_1");
  assert.strictEqual(designFileName("naïve/ü"), "naïve_ü");
  assert.strictEqual(designFileName("x²"), "x²", "a superscript digit is kept");
  assert.strictEqual(designFileName("١٢"), "١٢", "Arabic-Indic digits");
  assert.strictEqual(designFileName("Ⅷ"), "Ⅷ", "a Roman numeral letter-number");
  assert.strictEqual(designFileName("é"), "e_", "a combining mark is not kept");
  assert.strictEqual(designFileName("a\u{1F600}b"), "a_b", "one _ for one astral character");
  assert.ok(nameExists("DÉTAIL é", [{ name: "détail_é" }]));

  const refusal = { ok: true, status: 200, data: { error: "no", reason: "no_design" } };
  assert.strictEqual(failureText(refusal, "save", false), "no");
  assert.ok(failureText(refusal, "save", true).includes("placeholders, not a design"));
  assert.ok(failureText({ ok: false, status: 0, data: {} }, "open").endsWith("did not answer"));
  assert.ok(failureText({ ok: false, status: 502, data: {} }, "open").includes("HTTP 502"));
  const odd = failureText({ ok: true, status: 200, data: { weird: 1 } }, "open");
  assert.ok(odd.includes("not understood") && odd.includes("weird"), odd);
  assert.strictEqual(failureText({ ok: true, status: 200, data: { error: { a: 1 } } }, "open"),
                     '{"a":1}', "an error that is not text is still readable");

  const got = (data, status = 200) => ({ ok: status === 200, status, data });
  const saved = { ok: true, name: "d", object_count: 3, overwrote: false, saved_at: "t" };
  const opened = {
    ok: true, name: "d", mode: "add", object_count: 1, total_objects: 4,
    objects: [{ saved_name: "a", name: "a", renamed: false }],
    previous_saved: false, frame_differs: false, missing_ports: [],
  };
  assert.ok(succeeded(got(saved), "save") && succeeded(got(opened), "open"));
  assert.ok(!succeeded(got({}), "save"), "no ok:true is not success");
  assert.ok(!succeeded(got(saved, 500), "save"), "an HTTP error is not success");
  assert.ok(!succeeded(got({ ...saved, error: "but no" }), "save"), "an error is not success");
  assert.ok(!succeeded(got({ ok: true, name: "ghost" }), "save"), "the count is needed");
  assert.ok(!succeeded(got({ ...saved, object_count: "3" }), "save"));
  assert.ok(!succeeded(got({ ...saved, name: "" }), "save"));
  assert.ok(!succeeded(got({ ok: true, name: "one" }), "open"), "an open needs its fields");
  assert.ok(!succeeded(got({ ...opened, mode: "merge" }), "open"));
  assert.ok(!succeeded(got({ ...opened, objects: null }), "open"));
  assert.ok(!succeeded(got({ ...opened, objects: [{ saved_name: "a", renamed: true }] }), "open"),
            "a renamed object needs both names");
  assert.ok(!succeeded(got({ ...opened, missing_ports: [{ port_id: "p" }] }), "open"));
  assert.ok(succeeded(got({ ...opened, total_objects: undefined, missing_ports: undefined }),
                      "open"), "optional fields may be absent");
  for (const text of [...openNoticeLines({ ...opened, frame_differs: true,
                                           frame_saved: { scan: { size_um: 20, angle_deg: 0 } },
                                           frame_now: { scan: { size_um: null } },
                                           missing_ports: [{ name: "pad", port_id: null }] }),
                      ...saveNoticeLines(saved)])
    assert.ok(!/undefined|null/.test(text), text);
  assert.strictEqual(
    listRowMeta({ name: "_autosave.bad", saved_at: null, object_count: 2, recovered: true }),
    "recovered autosave copy · 2 objects");
  assert.strictEqual(
    listRowMeta({ name: "x", saved_at: null, object_count: null, recovered: true }),
    "recovered autosave copy");
  assert.ok(!listRowMeta({ name: "_autosave.bad", object_count: 2 }).includes("recovered"),
            "the label follows the recovered key, not the name");
  assert.strictEqual(listRowMeta({ name: "x", saved_at: { a: 1 }, object_count: "7" }),
                     "no details in the file", "fields of the wrong type are left out");
  assert.deepStrictEqual(
    listRows([null, 3, "x", {}, { name: "" }, { name: 5 }, { name: "ok" }]), [{ name: "ok" }]);
  assert.deepStrictEqual(listRows(undefined), []);

  const lenient = openNoticeLines({ ...opened, left_out: [
    { name: "bad-wire", reason: "bad-wire: voltage is not a number" },
    { name: null, reason: { odd: true } }] });
  assert.deepStrictEqual(lenient.slice(1), [
    "left out: bad-wire: bad-wire: voltage is not a number",
    'left out: an object: {"odd":true}']);
  assert.ok(succeeded(got({ ...opened, left_out: [] }), "open"));
  assert.ok(!succeeded(got({ ...opened, left_out: "none" }), "open"));
  const objecty = openNoticeLines({
    ...opened, frame_differs: true, frame_saved: { scan: { size_um: { v: 20 } } },
    frame_now: { scan: { size_um: 30 } }, missing_ports: [{ name: "pad", port_id: { id: 3 } }] });
  for (const text of objecty) assert.ok(!text.includes("[object Object]"), text);
  assert.ok(objecty.some((l) => l.includes('{"id":3}')), objecty.join(" | "));
  assert.strictEqual(shown("x".repeat(200).split("")).length, 81, "a long value is shortened");
  assert.strictEqual(designFileName(" .. "), "");
  const listing = [{ name: "Device_A" }, { name: "_previous" }];
  assert.ok(nameExists("device a", listing), "compared as the service files it");
  assert.ok(!nameExists("device-b", listing) && !nameExists("", listing));

  const reply = {
    name: "device-A", mode: "add", object_count: 2, total_objects: 5,
    objects: [{ saved_name: "wire-1", name: "wire-1_2", renamed: true },
              { saved_name: "pad", name: "pad", renamed: false }],
    frame_saved: { scan: { size_um: 20, x_offset_um: 0 }, litho: null },
    frame_now: { scan: { size_um: 30, x_offset_um: 0 }, litho: null },
    frame_differs: true,
    missing_ports: [{ name: "pad", port_id: "p3" }],
  };
  const lines = openNoticeLines(reply);
  assert.strictEqual(lines[0], "device-A opened: 2 objects added (5 in the design now)");
  assert.deepStrictEqual(lines.filter((l) => l.startsWith("renamed")),
                         ["renamed: wire-1 → wire-1_2"], "only the changed names");
  assert.ok(lines.some((l) => l.includes("scan size_um 20 → 30")
                              && !l.includes("x_offset_um")), "the frame keys that differ");
  assert.ok(lines.some((l) => l.includes("pad") && l.includes("p3")), "missing port named");
  assert.deepStrictEqual(
    openNoticeLines({ name: "d", mode: "replace", object_count: 1, objects: [],
                      frame_differs: false, missing_ports: [] }),
    ["d opened: 1 object in place of the earlier design"]);
  assert.deepStrictEqual(saveNoticeLines({ name: "d", object_count: 3, overwrote: true }),
                         ["design saved as d (3 objects), replacing the earlier file"]);
  const replaced = { name: "d", mode: "replace", object_count: 1, objects: [] };
  assert.deepStrictEqual(openNoticeLines({ ...replaced, previous_saved: true }).slice(1),
                         ["the earlier design is kept as _previous"]);
  assert.deepStrictEqual(openNoticeLines({ ...replaced, previous_saved: false }).slice(1),
                         ["nothing was kept as _previous: the design was empty"]);
  assert.deepStrictEqual(openNoticeLines({ ...replaced, mode: "add", previous_saved: true })
    .slice(1), [], "an add keeps nothing as _previous");
  assert.strictEqual(listRowMeta({ object_count: 1, saved_at: "2026-10-05T14:02:11" }),
                     "1 object · 2026-10-05 14:02:11");
  assert.strictEqual(listRowMeta({ object_count: null, saved_at: null }),
                     "no details in the file");
  assert.strictEqual(listRowMeta({ object_count: 0, saved_at: null }), "0 objects");
  assert.strictEqual(listStateText("ok", "", 2), "");
  assert.strictEqual(listStateText("ok", "", 0), "no saved designs yet");
  assert.ok(listStateText("loading", "", 0).startsWith("loading"));
  assert.ok(listStateText("failed", "HTTP 500", 0).endsWith("could not be listed: HTTP 500"));
  ok("ec-design-files designFileName/nameExists/openNoticeLines/saveNoticeLines/listStateText");
}

// --- ec-panes: the right pane's width stays inside both minimums ------------
{
  assert.strictEqual(clampRightWidth(600, 1896, 7), 600, "a width that fits is kept");
  assert.strictEqual(clampRightWidth(10, 1896, 7), MIN_RIGHT_PX, "right pane minimum");
  assert.strictEqual(clampRightWidth(5000, 1896, 7), 1896 - 7 - MIN_RENDER_PX,
                     "render pane minimum");
  assert.strictEqual(clampRightWidth(500, 600, 7), MIN_RIGHT_PX,
                     "too narrow for both: the right pane's minimum wins");
  assert.strictEqual(clampRightWidth(470.6, 1896), 471, "whole pixels");
  ok("ec-panes clampRightWidth keeps both panes at or above their minimums");
  assert.strictEqual(defaultRightWidth(1896, 7), DEFAULT_RIGHT_PX, "full screen: ~600 px");
  assert.strictEqual(defaultRightWidth(948, 8), 470, "half screen: an equal split");
  assert.strictEqual(defaultRightWidth(HALF_WINDOW_PX + 1), DEFAULT_RIGHT_PX);
  assert.strictEqual(isHalfWindow(1200), true, "exactly 1200 px is a half-screen window");
  assert.strictEqual(isHalfWindow(1201), false);
  ok("ec-panes defaultRightWidth: equal split at half screen, 600 px wider");
}

// --- ec-render-layout: telemetry step-down, scope x mapping, z bar labels ----
{
  const L = (o) => telemetryLayout({ spare: 300, height: 800, n: 6, collapsed: false, ...o });
  assert.deepStrictEqual(L({ collapsed: true }),
                         { mode: "tab", width: TELE.TAB_PX, bars: false, spark: false });
  const wide = L({});
  assert.strictEqual(wide.mode, "column");
  assert.strictEqual(wide.width, 300);
  assert.ok(wide.bars && wide.spark, "room for bars and history graphs");
  assert.strictEqual(L({ spare: 2000 }).width, TELE.COL_MAX_PX, "the column is capped");
  const narrowCol = L({ spare: 200 });
  assert.ok(narrowCol.mode === "column" && !narrowCol.spark, "no history graph when narrow");
  const short = L({ height: TELE.HEAD_PX + 6 * TELE.ROW_PX + 10 });
  assert.ok(short.mode === "column" && !short.bars, "no bars without the vertical room");
  assert.strictEqual(L({ spare: TELE.COL_MIN_PX - 1 }).mode, "grid", "too narrow: grid");
  assert.strictEqual(L({ height: 100 }).mode, "grid", "a column never needs to scroll");
  assert.strictEqual(L({ half: true }).mode, "grid", "half screen: always the grid");
  assert.strictEqual(L({ half: true, collapsed: true }).mode, "tab", "a fold still folds");

  // half screen: the square image is as wide as the column, the scope takes the rest
  assert.strictEqual(halfImageSide(450, 900), 450, "spare height goes to the scope");
  assert.strictEqual(halfImageSide(450, 400), 400 - HALF.SCOPE_MIN_PX, "the scope keeps its minimum");
  assert.strictEqual(halfImageSide(450, 100), HALF.IMAGE_MIN_PX, "the image outranks the scope");

  // a 4-pixel line over an image 100..500 px, the whole image visible
  const m = scopeXMap(100, 500, 4, 0, 600);
  assert.deepStrictEqual([m.x(0), m.x(3), m.i0, m.i1], [150, 450, 0, 3]);
  // zoomed: the image runs 0..800 but only 200..600 is visible -> pixels 1 and 2
  const z = scopeXMap(0, 800, 4, 200, 600);
  assert.deepStrictEqual([z.i0, z.i1, z.x(1), z.x(2)], [1, 2, 300, 500]);
  assert.strictEqual(scopeXMap(700, 900, 4, 0, 600), null, "panned out of view");
  assert.strictEqual(scopeXMap(0, 100, 0, 0, 100), null, "no samples");
  assert.deepStrictEqual(rangeOver([[5, 1, 9, 3], null, [0, 2, NaN, 7]], 1, 2), [1, 9]);
  assert.strictEqual(rangeOver([[NaN, null]], 0, 1), null);

  assert.deepStrictEqual(zScaleLabels(-4.1, 38.2).map((t) => t.text), ["-4.1", "+17.1", "+38.2"]);
  assert.deepStrictEqual(zScaleLabels(0, 2).map((t) => t.text), ["+0.00", "+1.00", "+2.00"]);
  assert.deepStrictEqual(zScaleLabels(null, 1), []);

  const frame = { amp_v: null, hw_telemetry: { x_um: 1, y_um: 2, defl_v: 0.1, sum_v: 3,
    amp_v: 0.2, zdrive_v: 40, row: 3, t: 0, height_nm: null } };
  assert.deepStrictEqual(visibleChannels(frame).map((c) => c.id),
                         ["xy", "defl", "amp", "zdrive", "sum", "row"],
                         "every channel on the frame; height_nm is not polled and has no row");
  assert.deepStrictEqual(channelValues(TELEMETRY_CHANNELS[2], frame), [null],
                         "amplitude reads the frame's top-level (mode-blanked) value");
  // a station whose reads fail from boot sends nulls: the safety channels still draw
  const broken = { amp_v: null, hw_telemetry: { x_um: null, y_um: null, defl_v: null,
    zdrive_v: null, sum_v: null, amp_v: null, row: null, height_nm: null } };
  const drawn = visibleChannels(broken);
  for (const id of ["defl", "zdrive", "sum"]) {
    const ch = drawn.find((c) => c.id === id);
    assert.ok(ch, `${id} is drawn from a null-only frame`);
    assert.strictEqual(valueText(ch, channelValues(ch, broken)), "—", `${id} reads a dash`);
  }
  assert.strictEqual(valueText(TELEMETRY_CHANNELS[0], [1, 2]), "1.000, 2.000");
  assert.strictEqual(valueText(TELEMETRY_CHANNELS[0], [1, null]), "—");
  assert.deepStrictEqual(visibleChannels({ hw_telemetry: { defl_v: 1 } }).map((c) => c.id),
                         ["defl"], "only what the frame carries");
  assert.strictEqual(barFraction(5, [0, 10]), 0.5);
  assert.strictEqual(barFraction(50, [0, 10]), 1);
  assert.strictEqual(barFraction(5, null), null);
  const buf = [];
  for (let i = 0; i < 5; i++) pushHistory(buf, i, 3);
  assert.deepStrictEqual(buf, [2, 3, 4]);
  ok("ec-render-layout telemetryLayout/scopeXMap/rangeOver/zScaleLabels/visibleChannels");
}

// --- ec-measure: lengths from the drawn geometry -----------------------------
{
  const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b}`);
  // open polyline, closed outline with and without a repeated first point
  near(polylineLength([[0, 0], [3, 4]], false), 5, 1e-12, "one segment");
  near(polylineLength([[0, 0], [2, 0], [2, 1]], true), 3 + Math.sqrt(5), 1e-12, "closing edge added");
  near(polylineLength([[0, 0], [2, 0], [2, 1], [0, 1], [0, 0]], true), 6, 1e-12,
       "a closed polyline ending on its start adds nothing");
  assert.strictEqual(polylineLength([[1, 1]], false), 0);
  assert.strictEqual(polylineLength(null, false), 0);
  // a rectangle Bezier: its perimeter
  near(bezLength(rectBezier(0, 0, 4, 2)), 12, 1e-9, "rect bezier perimeter");
  // straight cusp nodes: the chord
  near(bezLength({ closed: false, nodes: [
    { x: 0, y: 0, h_in: null, h_out: null, kind: CUSP },
    { x: 3, y: 4, h_in: null, h_out: null, kind: CUSP }] }), 5, 1e-12, "straight segment");
  // a quarter circle (kappa cubic, radius 1): pi/2 within the flatten tolerance
  const K = 0.5522847498;
  const arc = { closed: false, nodes: [
    { x: 1, y: 0, h_in: null, h_out: [1, K], kind: CUSP },
    { x: 0, y: 1, h_in: [K, 1], h_out: null, kind: CUSP }] };
  near(bezLength(arc), Math.PI / 2, 2e-3, "quarter circle");
  // the flattened polyline is what is measured
  near(bezLength(arc), polylineLength(bezFlatten(arc, 0.004), false), 1e-12,
       "bezLength == length of bezFlatten");
  // a closed funnel-like 3-node outline: closing segment counted once
  const tri = { closed: true, nodes: [
    { x: 0, y: 0, h_in: null, h_out: null, kind: CUSP },
    { x: 4, y: 0, h_in: null, h_out: null, kind: CUSP },
    { x: 2, y: 3, h_in: null, h_out: null, kind: CUSP }] };
  near(bezLength(tri), 4 + 2 * Math.hypot(2, 3), 1e-9, "closed triangle perimeter");
  // a straight chiral profile with zero amplitude is its axis
  const P0 = { kind: "chiral", axis_um: [[0, 0], [10, 0]], normal: [[0, 1], [0, 1]],
               amp_um: [0, 0], periods: 5, v0: 1, v_k: 0, phase_rad: 0, lambda_um: 2 };
  near(chiralLength(P0), 10, 1e-9, "flat chiral == axis length");
  const at = chiralSampler(P0);
  assert.deepStrictEqual([at(0.5).x, at(0.5).y], [5, 0]);
  assert.strictEqual(chiralSampler({ axis_um: [[0, 0]] }), null, "missing data -> null");
  // with amplitude the wire is longer than its axis
  const P1 = { ...P0, amp_um: [0.5, 0.5] };
  assert.ok(chiralLength(P1) > 10.5, "modulated chiral is longer than its axis");
  // objectLength picks the drawn geometry per kind
  assert.ok(isClosedKind({ kind: "shape" }) && isClosedKind({ kind: "electrode" })
            && !isClosedKind({ kind: "path" }));
  near(objectLength({ kind: "path", points_um: [[0, 0], [3, 4]] }), 5, 1e-12, "plain path");
  near(objectLength({ kind: "shape", points_um: [[0, 0], [2, 0], [2, 1], [0, 1]] }), 6, 1e-12,
       "shape outline closes");
  near(objectLength({ kind: "shape", bezier: rectBezier(0, 0, 4, 2),
                      points_um: [[0, 0], [1, 0]] }), 12, 1e-9, "bezier wins over points");
  near(objectLength({ kind: "path", profile: P1, bezier: arc, points_um: [[0, 0], [1, 0]] }),
       chiralLength(P1), 1e-12, "chiral wire measured from its profile, not its base bezier");
  ok("ec-measure polylineLength/bezLength/chiralLength/objectLength");
}

// --- ec-geom skewAbout: the Select options' skew affine --------------------------
{
  const near2 = (a, b) => assert.ok(Math.abs(a[0] - b[0]) < 1e-12 && Math.abs(a[1] - b[1]) < 1e-12,
                                    `${a} vs ${b}`);
  near2(skewAbout([3, 4], [3, 4], 30, 20), [3, 4]);                 // the centre stays
  near2(skewAbout([1, 1], [0, 0], 45, 0), [2, 1]);                  // x shifts with y
  near2(skewAbout([1, 1], [0, 0], 0, 45), [1, 2]);                  // y shifts with x
  near2(skewAbout([2, 0], [1, 0], 0, -45), [2, -1]);
  near2(skewAbout([5, 7], [1, 2], 0, 0), [5, 7]);                   // zero is identity
  ok("ec-geom skewAbout");
}
// --- ec-geom boxFieldOk: the Select box values worth sending ----------------------
{
  assert.ok(boxFieldOk("x", -29.9, 30) && boxFieldOk("y", 30, 30));
  assert.ok(!boxFieldOk("x", 30.1, 30) && !boxFieldOk("y", NaN, 30));
  assert.ok(boxFieldOk("w", 0.001, 30) && boxFieldOk("h", 30, 30));
  assert.ok(!boxFieldOk("w", 0, 30) && !boxFieldOk("h", -1, 30) && !boxFieldOk("w", 1e300, 30));
  assert.ok(!boxFieldOk("w", Infinity, null));
  assert.ok(boxFieldOk("w", FALLBACK_XY_RANGE_UM, null) && !boxFieldOk("w", 1e300, null),
            "no model loaded: the fallback bound applies");
  ok("ec-geom boxFieldOk");
}

// --- ec-geom unionBox / clampViewToBounds: the view's pan bounds -----------------
{
  assert.strictEqual(unionBox([], 0.1), null);
  assert.deepStrictEqual(unionBox([{ x0: 0, y0: 0, w: 10, h: 4 }, null,
                                   { x0: -5, y0: 2, w: 2, h: 6 }], 0),
                         { x0: -5, y0: 0, w: 15, h: 8 });
  assert.deepStrictEqual(unionBox([{ x0: 0, y0: 0, w: 10, h: 5 }], 0.1),
                         { x0: -1, y0: -1, w: 12, h: 7 });
  const b = { x0: 0, y0: 0, w: 10, h: 10 };
  assert.deepStrictEqual(clampViewToBounds({ x0: 2, y0: 3, w: 4, h: 4 }, b),
                         { x0: 2, y0: 3, w: 4, h: 4 }, "inside: unchanged");
  assert.deepStrictEqual(clampViewToBounds({ x0: 8, y0: -3, w: 4, h: 4 }, b),
                         { x0: 6, y0: 0, w: 4, h: 4 }, "past an edge: pulled back");
  assert.deepStrictEqual(clampViewToBounds({ x0: 40, y0: 1, w: 14, h: 4 }, b),
                         { x0: -2, y0: 1, w: 14, h: 4 }, "wider than the bounds: centred");
  ok("ec-geom unionBox / clampViewToBounds");
}

// --- ec-render-layout dragZRange: the z bar's hand-set range ---------------------
{
  const st = { lo: 0, hi: 10 };
  assert.deepStrictEqual(dragZRange(st, "hi", -0.25), { lo: 0, hi: 7.5 });
  assert.deepStrictEqual(dragZRange(st, "hi", 0.5), { lo: 0, hi: 15 });
  assert.deepStrictEqual(dragZRange(st, "lo", 0.2), { lo: 2, hi: 10 });
  assert.deepStrictEqual(dragZRange(st, "hi", -3), { lo: 0, hi: 0.1 }, "ends never cross");
  assert.deepStrictEqual(dragZRange(st, "lo", 3), { lo: 9.9, hi: 10 }, "ends never cross");
  // a flat image: the range is a near-zero span, its ends still move
  const flat = dragZRange({ lo: 5, hi: 5 + 1e-5 }, "hi", 0.5);
  assert.ok(Math.abs(flat.hi - (5 + 1e-5 + 0.5 * 0.05)) < 1e-6, JSON.stringify(flat));
  assert.ok(dragZRange({ lo: 0, hi: 1e-9 }, "lo", -1).lo < -1e-4, "flat at zero");
  ok("ec-render-layout dragZRange");
}

// --- ec-written addStroke / writeMeta: strokes grouped per object per run ---------
{
  const ws = [];
  const a = addStroke(ws, 1, { name: "box#0", object: "box", voltage: 3 }, 0);
  const b = addStroke(ws, 1, { name: "box#1", object: "box", voltage: 3 }, 5);
  assert.strictEqual(a, b, "two strokes of one object in one run: one write");
  assert.strictEqual(a.strokes, 2);
  const c = addStroke(ws, 1, { name: "wire", voltage: -1.5 }, 6);
  assert.strictEqual(c.object, "wire", "no object field: the stroke's name");
  const d = addStroke(ws, 2, { name: "box#0", object: "box", voltage: 3, partial: true }, 9);
  assert.notStrictEqual(d, a, "the next run is a write of its own");
  assert.ok(d.partial && !a.partial);
  assert.strictEqual(ws.length, 3);
  assert.match(writeMeta({ t: Date.UTC(2026, 0, 1), voltage: -1.5 }), /^\d{2}:\d{2} · -1\.5 V$/);
  assert.match(writeMeta({ t: 0, voltage: 3 }), / · \+3\.0 V$/);
  ok("ec-written addStroke / writeMeta");
}

// --- ec-render-layout telemetryLayout: stable for the same box, hysteresis at edges ---
{
  const need = TELE.HEAD_PX + 6 * (TELE.ROW_PX + TELE.BAR_PX);
  const L = (height, width, prev) =>
    telemetryLayout({ spare: width, height, n: 6, collapsed: false, prev });
  // the same box gives the same answer however often it is asked
  for (const h of [need - 1, need, need + 1, 500, 790]) {
    let f = L(h, 300, null);
    for (let k = 0; k < 20; k++) {
      const g = L(h, 300, f);
      assert.deepStrictEqual(g, f, `stable at height ${h}`);
      f = g;
    }
  }
  // bars on at the threshold, kept until HYST_PX under it
  const on = L(need, 300, null);
  assert.ok(on.bars);
  assert.ok(L(need - TELE.HYST_PX + 1, 300, on).bars, "kept just under the threshold");
  assert.ok(!L(need - TELE.HYST_PX - 1, 300, on).bars, "dropped past the hysteresis");
  assert.ok(!L(need - 1, 300, L(need - 40, 300, null)).bars, "not re-added under it");
  const sp = L(800, TELE.SPARK_MIN_PX, null);
  assert.ok(sp.spark && L(800, TELE.SPARK_MIN_PX - 5, sp).spark);
  assert.ok(!L(800, TELE.SPARK_MIN_PX - 5, null).spark);
  ok("ec-render-layout telemetryLayout stable + hysteresis");
}

// --- ec-render-layout telemetryStaleness: a stalled poll or a closed link ----------
{
  assert.strictEqual(TELEMETRY_STALE_S, 5);
  assert.deepStrictEqual(telemetryStaleness(true, 0.5), { stale: false, text: "", why: "" });
  assert.deepStrictEqual(telemetryStaleness(true, 2.5), { stale: false, text: "", why: "" },
                         "rig telemetry ages about 2 s and is still live");
  assert.deepStrictEqual(telemetryStaleness(true, 5), { stale: false, text: "", why: "" });
  const old = telemetryStaleness(true, 6.23);
  assert.ok(old.stale && old.text === "stale 6.2 s" && /no new telemetry/.test(old.why));
  const down = telemetryStaleness(false, 0.3);
  assert.ok(down.stale && down.text === "stale 0.3 s" && /closed/.test(down.why));
  assert.strictEqual(telemetryStaleness(false, null).text, "stale", "no frame yet");
  assert.strictEqual(telemetryStaleness(true, null).stale, false);
  ok("ec-render-layout telemetryStaleness");
}

console.log(`\nec modules: all ${checks} checks passed`);
