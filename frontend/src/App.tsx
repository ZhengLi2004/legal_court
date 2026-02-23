import { AppRoutes } from "./app/routes";
import { DebateProvider } from "./app/state/DebateContext";
import "./app/app.css";

function App() {
  return (
    <DebateProvider>
      <AppRoutes />
    </DebateProvider>
  );
}

export default App;
