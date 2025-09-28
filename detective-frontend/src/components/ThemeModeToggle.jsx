import React from 'react';
import IconButton from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import LightModeIcon from '@mui/icons-material/LightMode';

export default function ThemeModeToggle({ mode, onToggle }) {
  const isDark = mode === 'dark';
  return (
    <Tooltip title={isDark ? 'Switch to Light Mode' : 'Switch to Night Investigation Mode'}>
      <IconButton color="inherit" onClick={onToggle} size="large" aria-label="toggle theme mode">
        {isDark ? <LightModeIcon /> : <DarkModeIcon />}
      </IconButton>
    </Tooltip>
  );
}
