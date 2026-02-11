export type AppRoute =
  | "/app/launch"
  | "/app/live"
  | "/app/graph"
  | "/app/team"
  | "/app/memory"
  | "/app/judgment"
  | "/app/replay"
  | "/app/playbook"
  | "/admin/debug";

export type AdapterMode = "auto" | "http" | "mock";
export type StreamStatus = "idle" | "ws" | "poll";
