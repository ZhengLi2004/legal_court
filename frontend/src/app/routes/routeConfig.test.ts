import { describe, expect, it } from "vitest";
import { APP_ROUTES, normalizeAppPath } from "./routeConfig";

describe("normalizeAppPath", () => {
  it("keeps known routes unchanged", () => {
    expect(normalizeAppPath(APP_ROUTES.launch)).toBe(APP_ROUTES.launch);
    expect(normalizeAppPath(APP_ROUTES.live)).toBe(APP_ROUTES.live);
    expect(normalizeAppPath(APP_ROUTES.team)).toBe(APP_ROUTES.team);
    expect(normalizeAppPath(APP_ROUTES.memory)).toBe(APP_ROUTES.memory);
    expect(normalizeAppPath(APP_ROUTES.judgment)).toBe(APP_ROUTES.judgment);
  });

  it("maps legacy graph path to live page", () => {
    expect(normalizeAppPath("/app/graph")).toBe(APP_ROUTES.live);
  });

  it("falls back to launch for unsupported paths", () => {
    expect(normalizeAppPath("")).toBe(APP_ROUTES.launch);
    expect(normalizeAppPath("/app/unknown")).toBe(APP_ROUTES.launch);
    expect(normalizeAppPath("/outside")).toBe(APP_ROUTES.launch);
  });

  it("normalizes trailing slash and root app path", () => {
    expect(normalizeAppPath("/app/live/")).toBe(APP_ROUTES.live);
    expect(normalizeAppPath("/app")).toBe(APP_ROUTES.launch);
  });
});
