import { lazy, Suspense, useCallback } from "react";

import {
  BrowserRouter,
  Navigate,
  Outlet,
  Route,
  Routes,
  useLocation,
  useNavigate,
} from "react-router-dom";

import { MainShell } from "../components/MainShell";
import { APP_ROUTES, normalizeAppPath, type AppRoute } from "./routeConfig";

const LaunchPage = lazy(() =>
  import("../../features/launch/ui/LaunchPage").then((module) => ({
    default: module.LaunchPage,
  })),
);

const LivePage = lazy(() =>
  import("../../features/live/ui/LivePage").then((module) => ({
    default: module.LivePage,
  })),
);

const TeamFlowPage = lazy(() =>
  import("../pages/TeamFlowPage").then((module) => ({
    default: module.TeamFlowPage,
  })),
);

const MemoryPage = lazy(() =>
  import("../pages/MemoryPage").then((module) => ({
    default: module.MemoryPage,
  })),
);

const JudgmentPage = lazy(() =>
  import("../pages/JudgmentPage").then((module) => ({
    default: module.JudgmentPage,
  })),
);

function LoadingPage() {
  return (
    <article className="ux-card">
      <h2>页面加载中</h2>
      <p className="ux-empty">正在加载页面内容，请稍候。</p>
    </article>
  );
}

function RouteShell() {
  const location = useLocation();
  const navigate = useNavigate();
  const route = normalizeAppPath(location.pathname);

  const onNavigate = useCallback(
    (to: AppRoute): void => {
      if (to !== route) {
        navigate(to);
      }
    },
    [navigate, route],
  );

  return (
    <MainShell onNavigate={onNavigate} route={route}>
      <Suspense fallback={<LoadingPage />}>
        <Outlet />
      </Suspense>
    </MainShell>
  );
}

function LaunchRoutePage() {
  const navigate = useNavigate();

  return <LaunchPage onGoLive={() => navigate(APP_ROUTES.live)} />;
}

export function AppRoutes() {
  return (
    <BrowserRouter>
      <Routes>
        <Route
          element={<Navigate replace to={APP_ROUTES.live} />}
          path="/app/graph"
        />

        <Route element={<RouteShell />} path="/app">
          <Route element={<Navigate replace to={APP_ROUTES.launch} />} index />
          <Route element={<LaunchRoutePage />} path="launch" />
          <Route element={<LivePage />} path="live" />
          <Route element={<TeamFlowPage />} path="team" />
          <Route element={<MemoryPage />} path="memory" />
          <Route element={<JudgmentPage />} path="judgment" />

          <Route
            element={<Navigate replace to={APP_ROUTES.launch} />}
            path="*"
          />
        </Route>

        <Route element={<Navigate replace to={APP_ROUTES.launch} />} path="/" />
        <Route element={<Navigate replace to={APP_ROUTES.launch} />} path="*" />
      </Routes>
    </BrowserRouter>
  );
}
