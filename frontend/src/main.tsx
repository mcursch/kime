import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import ResultsPage from './pages/ResultsPage';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/results/:id" element={<ResultsPage />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
);
