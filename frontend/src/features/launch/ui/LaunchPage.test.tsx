import { describe, expect, it } from "vitest";

import { LaunchPage } from "./LaunchPage";

describe("LaunchPage module", () => {
  it("exports a component function", () => {
    expect(typeof LaunchPage).toBe("function");
  });
});
