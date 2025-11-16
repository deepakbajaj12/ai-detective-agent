import React from 'react';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Button from '@mui/material/Button';
import Stack from '@mui/material/Stack';
import Box from '@mui/material/Box';
import { Link as RouterLink } from 'react-router-dom';
import SearchIcon from '@mui/icons-material/Search';
import LogoTitle from './LogoTitle';
import ThemeModeToggle from './ThemeModeToggle';
import BuildCircleOutlinedIcon from '@mui/icons-material/BuildCircleOutlined';
import JobsPanel from './JobsPanel';
import SummarizeIcon from '@mui/icons-material/Summarize';

// Legacy headerStyle removed (LogoTitle supplies styling)
export default function Header({ mode, onToggleMode }) {
  const [jobsOpen, setJobsOpen] = React.useState(false);
  return (
    <AppBar position="sticky" color="primary">
      <Toolbar>
        <Box sx={{ mr:2 }}>
          <img src="/holmes-silhouette.svg" alt="Holmes" style={{ height:42, filter:'invert(85%) sepia(20%) hue-rotate(15deg)' }} onError={(e)=>{ e.target.style.display='none'; }} />
        </Box>
        <Box sx={{ flexGrow:1, display:'flex', alignItems:'flex-start' }}>
          <LogoTitle text="AI Detective" tagline="ANALYTICAL INTELLIGENCE" />
        </Box>
        <Stack direction="row" spacing={1} alignItems="center">
          <Button color="inherit" component={RouterLink} to="/suspects">Suspects</Button>
          <Button color="inherit" component={RouterLink} to="/clues">Clues</Button>
          <Button color="inherit" startIcon={<SearchIcon />} component={RouterLink} to="/search">Search</Button>
          <Button color="inherit" component={RouterLink} to="/ingest">Ingest PDF</Button>
          <Button color="inherit" component={RouterLink} to="/metrics">Metrics</Button>
          <Button color="inherit" component={RouterLink} to="/qa">QA</Button>
          <Button color="inherit" startIcon={<SummarizeIcon />} component={RouterLink} to="/analysis">Analysis</Button>
          <Button color="inherit" component={RouterLink} to="/timeline">Timeline</Button>
          <Button color="inherit" component={RouterLink} to="/graph">Graph</Button>
          <Button color="inherit" startIcon={<BuildCircleOutlinedIcon />} onClick={()=>setJobsOpen(true)}>Jobs</Button>
          <ThemeModeToggle mode={mode} onToggle={onToggleMode} />
        </Stack>
      </Toolbar>
      <JobsPanel open={jobsOpen} onClose={()=>setJobsOpen(false)} />
    </AppBar>
  );
}

