export const APP_ROUTES = {
  launch: "/app/launch",
  live: "/app/live",
  team: "/app/team",
  memory: "/app/memory",
  judgment: "/app/judgment",
} as const;

export type AppRoute = (typeof APP_ROUTES)[keyof typeof APP_ROUTES];

const APP_ROUTE_LIST: readonly AppRoute[] = [
  APP_ROUTES.launch,
  APP_ROUTES.live,
  APP_ROUTES.team,
  APP_ROUTES.memory,
  APP_ROUTES.judgment,
];

const APP_ROUTE_SET = new Set<string>(APP_ROUTE_LIST);

function trimTrailingSlash(pathname: string): string {
  const value = String(pathname || "").trim();

  if (!value) {
    return "";
  }

  const withLeadingSlash = value.startsWith("/") ? value : `/${value}`;
  const normalized = withLeadingSlash.replace(/\/+$/, "");
  return normalized || "/";
}

export function normalizeAppPath(pathname: string): AppRoute {
  const normalized = trimTrailingSlash(pathname);

  if (normalized === "/app/graph") {
    return APP_ROUTES.live;
  }

  if (normalized === "/app") {
    return APP_ROUTES.launch;
  }

  if (APP_ROUTE_SET.has(normalized)) {
    return normalized as AppRoute;
  }

  return APP_ROUTES.launch;
}
