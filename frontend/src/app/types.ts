export type AppRoute =
  | "/app/launch"
  | "/app/live"
  | "/app/team"
  | "/app/memory"
  | "/app/judgment";

export type AdapterMode = "http";
export type StreamStatus = "idle" | "ws" | "poll";
