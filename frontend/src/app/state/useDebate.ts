import { useContext } from "react";

import { DebateContext } from "./debateContextObject";

export function useDebate() {
  const value = useContext(DebateContext);

  if (!value) {
    throw new Error("useDebate must be used inside DebateProvider");
  }

  return value;
}
