import React from 'react';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Stack from '@mui/material/Stack';
import Box from '@mui/material/Box';
import { Link as RouterLink } from 'react-router-dom';
import SearchIcon from '@mui/icons-material/Search';

const headerStyle = {
  fontFamily: 'Playfair Display, serif',
  fontWeight: 600,
  letterSpacing: '0.06em',
  display: 'flex',
  alignItems: 'center'
};

export default function Header() {
  return (
    <AppBar position="sticky" color="primary">
      <Toolbar>
        <Box sx={{ mr:2 }}>
          <img src="/holmes-silhouette.svg" alt="Holmes" style={{ height:42, filter:'invert(85%) sepia(20%) hue-rotate(15deg)' }} onError={(e)=>{ e.target.style.display='none'; }} />
        </Box>
        <Typography variant="h4" sx={{ ...headerStyle, flexGrow:1, textShadow: '2px 2px #2e382b' }}>
          AI Detective
        </Typography>
        <Stack direction="row" spacing={1} alignItems="center">
          <Button color="inherit" component={RouterLink} to="/suspects">Suspects</Button>
          <Button color="inherit" component={RouterLink} to="/clues">Clues</Button>
          <Button color="inherit" startIcon={<SearchIcon />} component={RouterLink} to="/">Search</Button>
        </Stack>
      </Toolbar>
    </AppBar>
  );
}

