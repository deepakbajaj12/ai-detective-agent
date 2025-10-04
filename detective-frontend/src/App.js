import React, { useState, useMemo, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import ErrorBoundary from './components/ErrorBoundary';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Container from '@mui/material/Container';
import './App.css';
import Header from './components/Header';
import SuspectList from './pages/SuspectList';
import DocumentIngest from './pages/DocumentIngest';
import MetricsDashboard from './pages/MetricsDashboard';
import SuspectProfile from './pages/SuspectProfile';
import CluesPage from './pages/CluesPage';
import QAPage from './pages/QAPage';
import TimelinePage from './pages/TimelinePage';
import GraphPage from './pages/GraphPage';
import { buildTheme } from './theme';

function App() {
  const [mode, setMode] = useState('light');
  const activeTheme = useMemo(()=> buildTheme(mode), [mode]);
  const toggleMode = () => setMode(m => m === 'light' ? 'dark' : 'light');
  useEffect(()=> {
    if(mode === 'dark') document.body.classList.add('dark-mode');
    else document.body.classList.remove('dark-mode');
  }, [mode]);
  return (
    <ThemeProvider theme={activeTheme}>
      <CssBaseline />
      <BrowserRouter>
        <Header mode={mode} onToggleMode={toggleMode} />
        <ErrorBoundary>
          <Container maxWidth="lg" sx={{ py: 3 }}>
            <Routes>
              <Route path="/" element={<Navigate to="/suspects" replace />} />
              <Route path="/suspects" element={<SuspectList />} />
              <Route path="/suspects/:sid" element={<SuspectProfile />} />
              <Route path="/clues" element={<CluesPage />} />
              <Route path="/ingest" element={<DocumentIngest />} />
              <Route path="/metrics" element={<MetricsDashboard />} />
              <Route path="/qa" element={<QAPage />} />
              <Route path="/timeline" element={<TimelinePage />} />
              <Route path="/graph" element={<GraphPage />} />
            </Routes>
          </Container>
        </ErrorBoundary>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
