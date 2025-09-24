import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Container from '@mui/material/Container';
import './App.css';
import Header from './components/Header';
import SuspectList from './pages/SuspectList';
import SuspectProfile from './pages/SuspectProfile';
import CluesPage from './pages/CluesPage';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#5e35b1' },
    secondary: { main: '#26a69a' },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <Header />
        <Container maxWidth="lg" sx={{ py: 3 }}>
          <Routes>
            <Route path="/" element={<Navigate to="/suspects" replace />} />
            <Route path="/suspects" element={<SuspectList />} />
            <Route path="/suspects/:sid" element={<SuspectProfile />} />
            <Route path="/clues" element={<CluesPage />} />
          </Routes>
        </Container>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
