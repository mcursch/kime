import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import HistoryPage from "./pages/HistoryPage";
import ResultsPage from "./pages/ResultsPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Default route: redirect root to history */}
        <Route path="/" element={<Navigate to="/history" replace />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/:attemptId" element={<ResultsPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
