export function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

export function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

export function asNumber(value: unknown, fallback = 0): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === "string") {
    const parsed = Number(value);

    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  return fallback;
}

export function unwrapPayload(raw: unknown): Record<string, unknown> {
  const outer = asRecord(raw);
  const nested = outer.data ?? outer.snapshot ?? outer.state ?? outer.payload;
  return nested !== undefined ? asRecord(nested) : outer;
}

export function phaseLabel(phase: string): string {
  const normalized = phase.toLowerCase();

  if (normalized === "idle") {
    return "初始化中";
  }

  if (normalized === "running") {
    return "辩论中";
  }

  if (normalized === "ready_for_adjudication") {
    return "待裁决";
  }

  if (normalized === "finished") {
    return "已裁决";
  }

  if (normalized === "error") {
    return "异常";
  }

  return phase || "未知";
}

export function nodeStatusLabel(status: string): string {
  const normalized = status.toUpperCase();

  if (normalized === "VALIDATED") {
    return "被采纳";
  }

  if (normalized === "DEFEATED") {
    return "被驳回";
  }

  if (normalized === "HYPOTHETICAL") {
    return "待验证";
  }

  return status || "未知";
}
