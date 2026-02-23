import { describe, expect, it } from "vitest";

import {
  asRecord,
  asString,
  nodeStatusLabel,
  phaseLabel,
  unwrapPayload,
} from "./payload";

describe("payload helpers", () => {
  it("unwraps nested payload fields with stable priority", () => {
    expect(unwrapPayload({ data: { foo: 1 } })).toEqual({ foo: 1 });
    expect(unwrapPayload({ snapshot: { foo: 2 } })).toEqual({ foo: 2 });
    expect(unwrapPayload({ state: { foo: 3 } })).toEqual({ foo: 3 });
    expect(unwrapPayload({ payload: { foo: 4 } })).toEqual({ foo: 4 });
    expect(unwrapPayload({ foo: 5 })).toEqual({ foo: 5 });
  });

  it("normalizes primitive values safely", () => {
    expect(asRecord(null)).toEqual({});
    expect(asRecord("x")).toEqual({});
    expect(asString("ok")).toBe("ok");
    expect(asString(12)).toBe("12");
    expect(asString(false)).toBe("false");
    expect(asString(undefined, "fallback")).toBe("fallback");
  });

  it("maps labels to localized text", () => {
    expect(phaseLabel("running")).toBe("辩论中");
    expect(nodeStatusLabel("VALIDATED")).toBe("被采纳");
  });
});
