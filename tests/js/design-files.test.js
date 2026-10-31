import { describe, it, expect } from "vitest";
import {
  designFileName, nameExists, openNoticeLines, saveNoticeLines, listRowMeta, listRows,
  listStateText, failureText, succeeded, shown, PLACEHOLDER_TEXT,
} from "../../src/afm_litho/web/ec-design-files.js";

describe("designFileName (the name the service files a typed name under)", () => {
  it("strips the suffix in any case and the surrounding blanks", () => {
    expect(designFileName("dev1")).toBe("dev1");
    expect(designFileName("dev1.design.json")).toBe("dev1");
    expect(designFileName("DEV1.Design.JSON")).toBe("DEV1");
    expect(designFileName("  my design  ")).toBe("my_design");
  });

  it("replaces path separators and unsafe characters", () => {
    expect(designFileName("a/b\\c")).toBe("a_b_c");
    expect(designFileName("x:y*z?")).toBe("x_y_z_");
  });

  it("keeps dots inside the name, so dev.1 and dev.2 stay different", () => {
    expect(designFileName("dev.1")).not.toBe(designFileName("dev.2"));
    expect(designFileName("dev.1")).toBe("dev.1");
  });

  it("a name that is only dots, blanks or the suffix has no filing name", () => {
    expect(designFileName("")).toBe("");
    expect(designFileName("   ")).toBe("");
    expect(designFileName("..")).toBe("");
    expect(designFileName(". .")).toBe("");
    expect(designFileName(".design.json")).toBe("");
    expect(designFileName(null)).toBe("");
  });

  it("nameExists compares the filing name, case-insensitively", () => {
    const designs = [{ name: "Dev1" }, { name: "other" }];
    expect(nameExists("dev1.design.json", designs)).toBe(true);
    expect(nameExists(" DEV1 ", designs)).toBe(true);
    expect(nameExists("dev2", designs)).toBe(false);
    expect(nameExists("", designs)).toBe(false);
    expect(nameExists("dev1", undefined)).toBe(false);
  });
});

describe("notices", () => {
  it("save notice counts objects and says when it overwrote", () => {
    expect(saveNoticeLines({ name: "a", object_count: 1 })).toEqual(["design saved as a (1 object)"]);
    expect(saveNoticeLines({ name: "a", object_count: 3, overwrote: true }))
      .toEqual(["design saved as a (3 objects), replacing the earlier file"]);
  });

  it("open in replace mode says where the earlier design went", () => {
    const base = { name: "d", mode: "replace", object_count: 2 };
    expect(openNoticeLines({ ...base, previous_saved: true })).toEqual([
      "d opened: 2 objects in place of the earlier design",
      "the earlier design is kept as _previous"]);
    expect(openNoticeLines({ ...base, previous_saved: false })[1])
      .toBe("nothing was kept as _previous: the design was empty");
  });

  it("open in add mode reports the new total, renames, missing ports and left-out objects", () => {
    const lines = openNoticeLines({
      name: "d", mode: "add", object_count: 1, total_objects: 5,
      objects: [{ saved_name: "w", name: "w-2", renamed: true }, { saved_name: "x", name: "x", renamed: false }],
      missing_ports: [{ name: "w-2", port_id: 7 }, { name: "x", port_id: null }],
      left_out: [{ name: "bad", reason: "no points" }, { reason: { code: 1 } }],
    });
    expect(lines).toEqual([
      "d opened: 1 object added (5 in the design now)",
      "renamed: w → w-2",
      "w-2 is bound to electrode port 7, which this sample does not have",
      "x is bound to an electrode port that this sample does not have",
      "left out: bad: no points",
      'left out: an object: {"code":1}',
    ]);
  });

  it("a frame difference lists only the fields that differ beyond 1e-9", () => {
    const [, line] = openNoticeLines({
      name: "d", mode: "add", object_count: 1, frame_differs: true,
      frame_saved: { scan: { size_um: 10, x_offset_um: 0 }, litho: { size_um: 5 } },
      frame_now: { scan: { size_um: 10 + 1e-12, x_offset_um: 2 }, litho: { size_um: 5 } },
    }).slice(0, 2);
    expect(line).toBe("saved in a different frame (scan x_offset_um 0 → 2); coordinates are kept as saved");
  });

  it("shown() renders any value as short text", () => {
    expect(shown(null)).toBe("none");
    expect(shown(undefined)).toBe("none");
    expect(shown("s")).toBe("s");
    expect(shown(3)).toBe("3");
    expect(shown({ a: 1 })).toBe('{"a":1}');
    expect(shown({ k: "x".repeat(200) }).length).toBe(81);
  });
});

describe("listing and failure text", () => {
  it("rows are the entries with a non-empty string name", () => {
    expect(listRows([{ name: "a" }, { name: "" }, null, { name: 3 }, "x", { name: "b" }]))
      .toEqual([{ name: "a" }, { name: "b" }]);
    expect(listRows(undefined)).toEqual([]);
  });

  it("row meta falls back to 'no details in the file'", () => {
    expect(listRowMeta({ name: "a" })).toBe("no details in the file");
    expect(listRowMeta({ name: "a", object_count: 2, saved_at: "2026-10-01T12:30:00" }))
      .toBe("2 objects · 2026-10-01 12:30:00");
    expect(listRowMeta({ name: "a", recovered: true, object_count: 0 }))
      .toBe("recovered autosave copy · 0 objects");
    expect(listRowMeta({ name: "a", object_count: -1 })).toBe("no details in the file");
  });

  it("list state text", () => {
    expect(listStateText("loading", "", 0)).toBe("loading the saved designs…");
    expect(listStateText("failed", "boom", 0)).toBe("the saved designs could not be listed: boom");
    expect(listStateText("ready", "", 0)).toBe("no saved designs yet");
    expect(listStateText("ready", "", 2)).toBe("");
  });

  it("failure text separates a service reason, no answer, HTTP error and an odd reply", () => {
    expect(failureText({ ok: true, status: 200, data: { error: "no such design" } }, "open", false))
      .toBe("no such design");
    expect(failureText({ ok: true, status: 200, data: { error: "e", reason: "no_design" } }, "save", true))
      .toBe(PLACEHOLDER_TEXT);
    expect(failureText({ ok: true, status: 200, data: { error: "e", reason: "no_design" } }, "save", false))
      .toBe("e");
    expect(failureText({ ok: false, status: 0, data: {} }, "save", false))
      .toBe("save design failed: the service did not answer");
    expect(failureText({ ok: false, status: 500, data: {} }, "open", false))
      .toBe("open design failed (HTTP 500)");
    expect(failureText({ ok: true, status: 200, data: { ok: false } }, "open", false))
      .toBe('open design failed: the service\'s reply was not understood (HTTP 200): {"ok":false}');
  });
});

describe("succeeded: only a complete, well-typed reply counts", () => {
  const save = { ok: true, name: "a", object_count: 2 };
  const open = { ok: true, name: "a", object_count: 2, mode: "add", objects: [] };
  const res = (data, ok = true) => ({ ok, status: ok ? 200 : 500, data });

  it("save", () => {
    expect(succeeded(res(save), "save")).toBe(true);
    expect(succeeded(res({ ...save, overwrote: true }), "save")).toBe(true);
    expect(succeeded(res({ ...save, overwrote: "yes" }), "save")).toBe(false);
    expect(succeeded(res({ ...save, error: "x" }), "save")).toBe(false);
    expect(succeeded(res({ ...save, ok: false }), "save")).toBe(false);
    expect(succeeded(res({ ...save, object_count: 1.5 }), "save")).toBe(false);
    expect(succeeded(res({ ...save, name: "" }), "save")).toBe(false);
    expect(succeeded(res(save, false), "save")).toBe(false);
  });

  it("open needs a mode and well-formed object entries", () => {
    expect(succeeded(res(open), "open")).toBe(true);
    expect(succeeded(res({ ...open, mode: "merge" }), "open")).toBe(false);
    expect(succeeded(res({ ...open, objects: undefined }), "open")).toBe(false);
    expect(succeeded(res({ ...open, objects: [{ saved_name: "a", name: "a" }] }), "open")).toBe(false);
    expect(succeeded(res({ ...open, objects: [{ saved_name: "a", name: "a", renamed: false }] }), "open")).toBe(true);
    expect(succeeded(res({ ...open, total_objects: -1 }), "open")).toBe(false);
    expect(succeeded(res({ ...open, missing_ports: [{}] }), "open")).toBe(false);
  });
});
