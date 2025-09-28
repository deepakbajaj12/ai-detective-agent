import { createTheme } from '@mui/material/styles';

export const buildTheme = (mode = 'light') => {
  const isDark = mode === 'dark';
  const primary = { main: '#3e4c3a' };
  const secondary = { main: '#bfa46f' };
  return createTheme({
    palette: {
      mode,
      primary,
      secondary,
      background: isDark
        ? { default: '#121512', paper: '#1c2019' }
        : { default: '#f5ecd7', paper: 'rgba(245,236,215,0.95)' },
      error: { main: '#8b2f2f' },
      warning: { main: '#b87333' },
      success: { main: '#2f6f4f' },
      divider: isDark ? 'rgba(191,164,111,0.25)' : 'rgba(62,76,58,0.3)',
      // Always provide a text object (MUI's CssBaseline expects palette.text.primary)
      text: isDark
        ? { primary: '#efe6d2', secondary: '#c9bda4', disabled: 'rgba(239,230,210,0.38)' }
        : { primary: '#2e2b26', secondary: '#4d483f', disabled: 'rgba(46,43,38,0.38)' }
    },
    typography: {
      fontFamily: 'Playfair Display, Merriweather, serif',
      h1: { fontWeight: 700, letterSpacing: '0.05em' },
      h4: { fontWeight: 600 },
      body1: { fontFamily: 'Merriweather, serif' },
      body2: { fontFamily: 'Merriweather, serif' },
      victorian: {
        fontFamily: 'Playfair Display, serif',
        letterSpacing: '0.08em',
        fontWeight: 600,
        textTransform: 'uppercase'
      }
    },
    shadows: [
      'none',
      '0 2px 4px rgba(0,0,0,0.15)',
      '0 3px 8px rgba(0,0,0,0.18)',
      '0 4px 12px rgba(0,0,0,0.22)',
      ...Array(21).fill('0 4px 14px rgba(0,0,0,0.25)')
    ],
    components: {
      MuiCard: {
        styleOverrides: {
          root: {
            position: 'relative',
            border: `1.5px solid ${secondary.main}`,
            borderRadius: 14,
            overflow: 'hidden',
            background: isDark
              ? 'linear-gradient(135deg, rgba(46,56,43,0.65), rgba(26,31,25,0.85))'
              : 'linear-gradient(135deg, rgba(255,255,255,0.55), rgba(255,255,255,0.22))',
            backdropFilter: 'blur(3px)',
            WebkitBackdropFilter: 'blur(3px)',
            boxShadow: isDark
              ? '0 6px 20px rgba(0,0,0,0.55)'
              : '0 6px 18px rgba(62,76,58,0.25)',
            transition: 'transform .25s ease, box-shadow .3s ease'
          }
        }
      },
      MuiAppBar: {
        styleOverrides: {
          colorPrimary: {
            background: isDark
              ? 'linear-gradient(135deg,#10140f 0%, #243024 70%)'
              : 'linear-gradient(135deg,#2e382b 0%, #3e4c3a 60%)',
            boxShadow: isDark
              ? '0 4px 16px rgba(0,0,0,0.6)'
              : '0 4px 12px rgba(0,0,0,0.4)'
          }
        }
      },
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: 30,
            letterSpacing: '0.06em',
            fontWeight: 600
          },
          containedPrimary: {
            background: 'linear-gradient(120deg,#3e4c3a,#55684d)',
            '&:hover': { background: 'linear-gradient(120deg,#465941,#637157)' }
          }
        }
      },
      MuiChip: {
        styleOverrides: {
          root: {
            fontWeight: 600,
            letterSpacing: '0.04em'
          }
        }
      },
      MuiTooltip: {
        styleOverrides: {
          tooltip: {
            backgroundColor: isDark ? '#2e382b' : '#3e4c3a',
            border: `1px solid ${secondary.main}`,
            fontFamily: 'Merriweather, serif'
          }
        }
      }
    }
  });
};

const theme = buildTheme('light');
export default theme;
