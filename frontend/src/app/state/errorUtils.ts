import { warnFrontend } from "../utils/warning";

export function toErrorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export function warnDebateContext(scope: string, err: unknown): void {
  warnFrontend("DebateContext", `${scope} failed`, toErrorMessage(err));
}
