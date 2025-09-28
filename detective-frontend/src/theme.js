import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#3e4c3a' },
    secondary: { main: '#bfa46f' },
    background: { default: '#f5ecd7', paper: 'rgba(245,236,215,0.95)' },
    error: { main: '#8b2f2f' },
    warning: { main: '#b87333' },
    success: { main: '#2f6f4f' }
  },
  typography: {
    fontFamily: 'Playfair Display, Merriweather, serif',
    h1: { fontWeight: 700, letterSpacing: '0.05em' },
    h4: { fontWeight: 600 },
    body1: { fontFamily: 'Merriweather, serif' },
    body2: { fontFamily: 'Merriweather, serif' }
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          border: '2px solid #bfa46f',
          borderRadius: 12,
          boxShadow: '0 4px 14px rgba(62,76,58,0.18)',
          backgroundImage: 'repeating-linear-gradient(0deg, rgba(255,255,255,0.04), rgba(255,255,255,0.04) 2px, transparent 2px, transparent 4px)'
        }
      }
    },
    MuiAppBar: {
      styleOverrides: {
        colorPrimary: {
          background: 'linear-gradient(135deg,#2e382b 0%, #3e4c3a 60%)',
          boxShadow: '0 4px 12px rgba(0,0,0,0.4)'
        }
      }
    }
  }
});

export default theme;
