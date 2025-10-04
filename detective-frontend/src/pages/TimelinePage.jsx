import React, { useEffect, useState } from 'react';
import Typography from '@mui/material/Typography';
import LinearProgress from '@mui/material/LinearProgress';
import Stack from '@mui/material/Stack';
import Paper from '@mui/material/Paper';
import { apiFetch } from '../apiBase';

export default function TimelinePage(){
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  useEffect(()=> {
    apiFetch('/api/timeline')
      .then(setData)
      .catch(()=> setData({ error:'Failed'}))
      .finally(()=> setLoading(false));
  }, []);
  if(loading) return <LinearProgress />;
  if(!data) return <Typography color="error">Failed.</Typography>;
  return (
    <Stack spacing={2}>
      <Typography variant="h4" sx={{ fontFamily:'Playfair Display, serif' }}>Timeline</Typography>
      <Stack spacing={2} sx={{ position:'relative', pl:3 }}>
        {(data.events||[]).map(e => (
          <Paper key={e.id} variant="outlined" sx={{ p:1.2, position:'relative' }}>
            <Typography variant="caption" color="text.secondary" sx={{ position:'absolute', left:-16, top:12, fontSize:'0.65rem' }}>●</Typography>
            <Typography variant="caption" color="text.secondary">{e.norm_timestamp || e.event_time || 'Unknown date'}</Typography>
            <Typography variant="body2">{e.event_text}</Typography>
          </Paper>
        ))}
        {(data.events||[]).length===0 && <Typography variant="body2" color="text.secondary">No events detected.</Typography>}
      </Stack>
    </Stack>
  );
}
