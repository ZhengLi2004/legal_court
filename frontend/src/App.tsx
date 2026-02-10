import { useCallback, useEffect, useState } from "react";
import "@xyflow/react/dist/style.css";
import AdminDebugPage from "./AdminDebugPage";
import { MainShell } from "./app/components/MainShell";
import { JudgmentPage } from "./app/pages/JudgmentPage";
import { LaunchPage } from "./app/pages/LaunchPage";
import { LivePage } from "./app/pages/LivePage";
import { DebateProvider } from "./app/state/DebateContext";
import type { AppRoute } from "./app/types";
import "./app/app.css";

function normalizeRoute(pathname: string): AppRoute {
  if (pathname === "/admin/debug") {
    return "/admin/debug";
  }

  if (pathname === "/app/live") {
    return "/app/live";
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

  if (route === "/admin/debug") {
    return (
      <main className="ux-admin-shell">
        <section className="ux-admin-bar">
          <div>
            <strong>后台调试模式</strong>
            <p className="ux-muted">这里保留完整流程调试能力。</p>
          </div>

          <button
            onClick={() => {
              navigate("/app/launch");
            }}
            type="button"
          >
            返回用户主站
          </button>
        </section>

        <AdminDebugPage />
      </main>
    );
  }

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
        {route === "/app/judgment" ? <JudgmentPage /> : null}
      </MainShell>
    </DebateProvider>
  );
}

export default App;
