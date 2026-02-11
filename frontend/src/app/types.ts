export type AppRoute =
  | "/app/launch"
  | "/app/live"
  | "/app/graph"
  | "/app/team"
  | "/app/memory"
  | "/app/judgment"
  | "/app/replay"
  | "/admin/debug";

export type AdapterMode = "auto" | "http" | "mock";
export type StreamStatus = "idle" | "ws" | "poll";
