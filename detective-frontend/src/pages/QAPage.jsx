import React, { useState } from 'react';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import LinearProgress from '@mui/material/LinearProgress';
import { apiFetch } from '../apiBase';

export default function QAPage(){
  const [q, setQ] = useState('');
  const [resp, setResp] = useState(null);
  const [loading, setLoading] = useState(false);
  const ask = () => {
    if(!q.trim()) return;
    setLoading(true);
    apiFetch(`/api/qa?q=${encodeURIComponent(q)}`)
      .then(setResp)
      .catch(()=> setResp({ error: 'Failed'}))
      .finally(()=> setLoading(false));
  };
  return (
    <Stack spacing={2}>
      <Typography variant="h4" sx={{ fontFamily:'Playfair Display, serif' }}>Case Q&A</Typography>
      <Stack direction={{ xs:'column', sm:'row' }} spacing={2}>
        <TextField fullWidth label="Question" value={q} onChange={e=> setQ(e.target.value)} />
        <Button variant="contained" onClick={ask} disabled={loading}>Ask</Button>
      </Stack>
      {loading && <LinearProgress />}
      {resp && !resp.error && (
        <Paper variant="outlined" sx={{ p:2 }}>
          <Typography variant="subtitle1" sx={{ mb:1 }}>Answer</Typography>
          <Typography variant="body2" sx={{ mb:2 }}>{resp.answer}</Typography>
          <Typography variant="subtitle2">Citations</Typography>
          <Typography variant="caption" color="text.secondary">Chunks</Typography>
          <ul style={{ marginTop:4 }}>
            {(resp.chunks||[]).map(c => <li key={c.chunk_id}><span style={{ fontSize:'.8rem' }}>{c.text}</span></li>)}
          </ul>
          <Typography variant="caption" color="text.secondary">Clues</Typography>
          <ul style={{ marginTop:4 }}>
            {(resp.clues||[]).map(c => <li key={c.clue_id}><span style={{ fontSize:'.8rem' }}>{c.text}</span></li>)}
          </ul>
        </Paper>
      )}
      {resp && resp.error && <Typography color="error" variant="body2">{resp.error}</Typography>}
    </Stack>
  );
}
