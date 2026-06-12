import { BrowserRouter, Routes, Route } from 'react-router-dom'
import UploadPage from './pages/UploadPage'
import ResultsPage from './pages/ResultsPage'
import HistoryPage from './pages/HistoryPage'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<UploadPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/:attemptId" element={<ResultsPage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
