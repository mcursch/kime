import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import UploadPage from './pages/UploadPage';
import ResultsPage from './pages/ResultsPage';
import HistoryPage from './pages/HistoryPage';

export default function App() {
  return (
    <BrowserRouter>
      <nav>
        <NavLink to="/">Upload</NavLink>
        {' | '}
        <NavLink to="/history">History</NavLink>
      </nav>
      <Routes>
        <Route path="/" element={<UploadPage />} />
        <Route path="/:attemptId" element={<ResultsPage />} />
        <Route path="/history" element={<HistoryPage />} />
      </Routes>
    </BrowserRouter>
  );
}
