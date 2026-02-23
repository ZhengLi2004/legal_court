const FRONTEND_WARNING_FLAG = "1";

function canEmitWarnings(): boolean {
  return (
    import.meta.env.DEV ||
    import.meta.env.VITE_ENABLE_FRONTEND_WARN === FRONTEND_WARNING_FLAG
  );
}

export function warnFrontend(
  scope: string,
  message: string,
  detail?: unknown,
): void {
  if (!canEmitWarnings()) {
    return;
  }

  const prefix = `[frontend] ${scope}: ${message}`;

  if (detail === undefined) {
    console.warn(prefix);
    return;
  }

  console.warn(prefix, detail);
}
