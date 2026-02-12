import { useCallback, useEffect, useState } from "react";
import { MainShell } from "./app/components/MainShell";
import { JudgmentPage } from "./app/pages/JudgmentPage";
import { LaunchPage } from "./app/pages/LaunchPage";
import { LivePage } from "./app/pages/LivePage";
import { MemoryPage } from "./app/pages/MemoryPage";
import { TeamFlowPage } from "./app/pages/TeamFlowPage";
import { DebateProvider } from "./app/state/DebateContext";
import type { AppRoute } from "./app/types";
import "./app/app.css";

function normalizeRoute(pathname: string): AppRoute {
  if (pathname === "/app/live") {
    return "/app/live";
  }

  if (pathname === "/app/graph") {
    return "/app/live";
  }

  if (pathname === "/app/team") {
    return "/app/team";
  }

  if (pathname === "/app/memory") {
    return "/app/memory";
  }

  if (pathname === "/app/judgment") {
    return "/app/judgment";
  }

  return "/app/launch";
}

function App() {
  const [route, setRoute] = useState<AppRoute>(() =>
    normalizeRoute(window.location.pathname),
  );

  useEffect(() => {
    const normalized = normalizeRoute(window.location.pathname);

    if (window.location.pathname !== normalized) {
      window.history.replaceState({}, "", normalized);
    }
  }, []);

  useEffect(() => {
    const onPopState = (): void => {
      setRoute(normalizeRoute(window.location.pathname));
    };

    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const navigate = useCallback((to: AppRoute): void => {
    const normalized = normalizeRoute(to);

    if (window.location.pathname !== normalized) {
      window.history.pushState({}, "", normalized);
    }

    setRoute(normalized);
  }, []);

  return (
    <DebateProvider>
      <MainShell onNavigate={navigate} route={route}>
        {route === "/app/launch" ? (
          <LaunchPage
            onGoLive={() => {
              navigate("/app/live");
            }}
          />
        ) : null}

        {route === "/app/live" ? <LivePage /> : null}
        {route === "/app/team" ? <TeamFlowPage /> : null}
        {route === "/app/memory" ? <MemoryPage /> : null}
        {route === "/app/judgment" ? <JudgmentPage /> : null}
      </MainShell>
    </DebateProvider>
  );
}

export default App;
