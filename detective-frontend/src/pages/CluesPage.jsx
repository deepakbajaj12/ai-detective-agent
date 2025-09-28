import React, { useEffect, useState, useCallback } from 'react';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import TextField from '@mui/material/TextField';
import IconButton from '@mui/material/IconButton';
import DeleteIcon from '@mui/icons-material/Delete';
import RefreshIcon from '@mui/icons-material/Refresh';
import Paper from '@mui/material/Paper';
import Chip from '@mui/material/Chip';
import Alert from '@mui/material/Alert';
import LinearProgress from '@mui/material/LinearProgress';
import Button from '@mui/material/Button';
import { apiFetch } from '../apiBase';
import Box from '@mui/material/Box';

export default function CluesPage() {
  const [clues, setClues] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState('');
  const [suspectFilter, setSuspectFilter] = useState('');

  const load = useCallback(() => {
    setLoading(true);
    apiFetch('/api/clues')
      .then(data => setClues(data))
      .catch(e => setError(e.message))
      .finally(()=> setLoading(false));
  }, []);

  useEffect(()=> { load(); }, [load]);

  useEffect(()=> {
    const h = () => load();
    window.addEventListener('clueAdded', h);
    return ()=> window.removeEventListener('clueAdded', h);
  }, [load]);

  const handleDelete = (id) => {
    apiFetch(`/api/clues/${id}`, { method: 'DELETE' })
      .then(()=> load())
      .catch(()=> {/* ignore for now */});
  };

  const filtered = clues.filter(c => {
    if(search && !c.text.toLowerCase().includes(search.toLowerCase())) return false;
    if(suspectFilter && c.suspect_id !== suspectFilter) return false;
    return true;
  });

  const suspectsFromClues = Array.from(new Set(clues.map(c => c.suspect_id).filter(Boolean)));

  return (
    <Stack spacing={2}>
      <Stack direction={{xs:'column', sm:'row'}} spacing={2} alignItems={{xs:'stretch', sm:'center'}}>
        <Typography variant="h5" sx={{ flexGrow:1 }}>Clues</Typography>
        <Button startIcon={<RefreshIcon />} onClick={load}>Refresh</Button>
      </Stack>
      <Stack direction={{xs:'column', sm:'row'}} spacing={2}>
        <TextField label="Search" value={search} onChange={e=>setSearch(e.target.value)} fullWidth />
        <TextField
          select
          label="Suspect"
          value={suspectFilter}
          onChange={e=>setSuspectFilter(e.target.value)}
          helperText="Filter by suspect"
          sx={{ minWidth: 160 }}
        >
          <option value="">All</option>
          {suspectsFromClues.map(s => <option key={s} value={s}>{s}</option>)}
        </TextField>
      </Stack>
      {loading && <LinearProgress />}
      {error && <Alert severity='error'>{error}</Alert>}
      <Stack spacing={1}>
        {filtered.map(c => (
          <Paper key={c.id} className="clue-paper" sx={{ p:1, display:'flex', alignItems:'center', gap:1 }}>
            <Box sx={{ flexGrow:1 }}>
              <Typography variant='body2'>{c.text}</Typography>
              {c.suspect_id && <Chip size='small' label={c.suspect_id} sx={{ mt:0.5 }} />}
            </Box>
            <IconButton onClick={()=>handleDelete(c.id)} size='small' color='error'>
              <DeleteIcon fontSize='small' />
            </IconButton>
          </Paper>
        ))}
        {(!loading && filtered.length === 0) && <Typography variant='body2' color='text.secondary'>No clues match filters.</Typography>}
      </Stack>
    </Stack>
  );
}
