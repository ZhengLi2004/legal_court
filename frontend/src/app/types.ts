export type AppRoute =
  | "/app/launch"
  | "/app/live"
  | "/app/judgment"
  | "/admin/debug";

export type AdapterMode = "auto" | "http" | "mock";
export type StreamStatus = "idle" | "ws" | "poll";
