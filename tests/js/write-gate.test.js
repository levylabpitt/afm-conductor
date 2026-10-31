// The empty-setpoint gate on writes (ec-instrument.js). The service defaults a missing
// setpoint_relative to 0.0, so this gate is the only thing between an empty box and a
// write at the free-air level.
import { describe, it, expect, beforeAll, beforeEach, afterEach, vi } from "vitest";

let inst;

const MARKUP = `
  <div id="log"></div>
  <input id="lithoSetpoint" value="">
  <div id="lithoSetpointHint" style="display:none"></div>
  <button id="perform">Write</button>
  <div id="guardNotice"><span id="guardNoticeText"></span><button id="guardNoticeDismiss"></button></div>
`;

const sp = () => document.getElementById("lithoSetpoint");
const perform = () => document.getElementById("perform");
const hint = () => document.getElementById("lithoSetpointHint");
const typeSetpoint = (v) => { sp().value = v; inst.refreshWriteGate(); };

beforeAll(async () => {
  document.body.innerHTML = MARKUP;
  inst = await import("../../src/afm_litho/web/ec-instrument.js");
});

beforeEach(() => {
  sp().value = "";
  perform().dataset.writing = "0";
  inst.markWriting(false);
});

describe("empty-setpoint gate", () => {
  it("an empty box disables Write, shows the hint, and says why", () => {
    inst.refreshWriteGate();
    expect(perform().disabled).toBe(true);
    expect(hint().style.display).toBe("");
    expect(perform().title).toBe("Set the write setpoint (V above free air) before writing.");
    expect(inst.writeSetpointEmpty()).toBe(true);
  });

  it("whitespace alone counts as empty", () => {
    typeSetpoint("   ");
    expect(inst.writeSetpointEmpty()).toBe(true);
    expect(perform().disabled).toBe(true);
  });

  it("a typed value enables Write and hides the hint", () => {
    typeSetpoint("0.25");
    expect(inst.writeSetpointEmpty()).toBe(false);
    expect(perform().disabled).toBe(false);
    expect(hint().style.display).toBe("none");
    expect(perform().title).toBe("");
  });

  it("a typed zero is a value, not an empty box", () => {
    typeSetpoint("0");
    expect(inst.writeSetpointEmpty()).toBe(false);
    expect(perform().disabled).toBe(false);
  });

  it("clearing the box again re-arms the gate", () => {
    typeSetpoint("0.25");
    typeSetpoint("");
    expect(perform().disabled).toBe(true);
  });

  it("a write in progress keeps Write disabled even with a value, and releases after", () => {
    typeSetpoint("0.25");
    inst.markWriting(true);
    expect(perform().dataset.writing).toBe("1");
    expect(perform().disabled).toBe(true);
    inst.markWriting(false);
    expect(perform().dataset.writing).toBe("0");
    expect(perform().disabled).toBe(false);
  });

  it("finishing a write does not unlock Write while the box is empty", () => {
    inst.markWriting(true);
    inst.markWriting(false);
    expect(perform().disabled).toBe(true);
  });
});

describe("hydrating the setpoint from the service model", () => {
  const field = (value) => ({ value, source: "default" });
  const stubModel = (model) => vi.stubGlobal("fetch", vi.fn(async () => ({
    ok: true, status: 200, json: async () => model,
  })));
  afterEach(() => vi.unstubAllGlobals());

  it("uses the last write's relative setpoint and clears the empty gate", async () => {
    stubModel({ last_write_params: { setpoint_relative: 0.4 },
                write_setpoint_relative_v: field(0.1) });
    inst.refreshWriteGate();
    expect(perform().disabled).toBe(true);
    await inst.hydrateInstrument();
    expect(sp().value).toBe("0.4");
    expect(perform().disabled).toBe(false);
  });

  it("a zero last-write setpoint falls through to the model default, never to 0", async () => {
    stubModel({ last_write_params: { setpoint_relative: 0 },
                write_setpoint_relative_v: field(0.1) });
    await inst.hydrateInstrument();
    expect(sp().value).toBe("0.1");
  });

  it("with neither a last write nor a default, the box stays empty and Write stays gated", async () => {
    stubModel({ last_write_params: {}, write_setpoint_relative_v: field(0) });
    await inst.hydrateInstrument();
    expect(sp().value).toBe("");
    expect(perform().disabled).toBe(true);
  });

  it("does not overwrite a value the operator is typing", async () => {
    stubModel({ last_write_params: { setpoint_relative: 0.4 } });
    sp().value = "0.9";
    sp().focus();
    await inst.hydrateInstrument();
    expect(sp().value).toBe("0.9");
    sp().blur();
  });

  it("amp gain comes from the model, and is null when the model lacks it", async () => {
    stubModel({ tip_amp_gain: field(25), last_write_params: {} });
    await inst.hydrateInstrument();
    expect(inst.getAmpGain()).toBe(25);
    stubModel({ last_write_params: {} });
    await inst.hydrateInstrument();
    expect(inst.getAmpGain()).toBeNull();
  });

  it("voltage window max is the widest side of the window", async () => {
    stubModel({ tip_voltage_window_v: field([-10, 7.5]), last_write_params: {} });
    await inst.hydrateInstrument();
    expect(inst.getVoltageWindowMax()).toBe(10);
  });
});
