import { describe, expect, it } from "vitest";
import { LivePage } from "./LivePage";

describe("LivePage module", () => {
  it("exports a component function", () => {
    expect(typeof LivePage).toBe("function");
  });
});
