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
import FormControlLabel from '@mui/material/FormControlLabel';
import Switch from '@mui/material/Switch';
import Slider from '@mui/material/Slider';
import MenuItem from '@mui/material/MenuItem';
import Tooltip from '@mui/material/Tooltip';
import Select from '@mui/material/Select';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import DoneIcon from '@mui/icons-material/Done';
import HelpIcon from '@mui/icons-material/Help';
import Box from '@mui/material/Box';

export default function CluesPage() {
  const [clues, setClues] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState('');
  const [suspectFilter, setSuspectFilter] = useState('');
  const [hideDuplicates, setHideDuplicates] = useState(true);
  const [minQuality, setMinQuality] = useState(0);
  const [annotationFilter, setAnnotationFilter] = useState('');
  const [recomputing, setRecomputing] = useState(false);

  const qualityLabel = (v) => `${(v*100).toFixed(0)}%`;

  const load = useCallback(() => {
    setLoading(true);
    const params = new URLSearchParams();
    if (hideDuplicates) params.set('hide_duplicates','1');
    if (minQuality>0) params.set('min_quality', String(minQuality.toFixed(2)));
    if (annotationFilter) params.set('annotation_label', annotationFilter);
    apiFetch(`/api/clues?${params.toString()}`)
      .then(data => setClues(data))
      .catch(e => setError(e.message))
      .finally(()=> setLoading(false));
  }, [hideDuplicates, minQuality, annotationFilter]);

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

  const handleAnnotate = (id, label) => {
    apiFetch(`/api/clues/${id}/annotate`, { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ label }) })
      .then(()=> load())
      .catch(()=> {/* silent */});
  };

  const triggerRecomputeDuplicates = () => {
    setRecomputing(true);
    apiFetch('/api/clues/recompute_duplicates', { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({}) })
      .then(()=> load())
      .finally(()=> setRecomputing(false));
  };

  const triggerRecomputeQuality = () => {
    setRecomputing(true);
    apiFetch('/api/clues/recompute_quality', { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({}) })
      .then(()=> load())
      .finally(()=> setRecomputing(false));
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
      <Stack spacing={2}>
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
          <FormControlLabel control={<Switch checked={hideDuplicates} onChange={e=>setHideDuplicates(e.target.checked)} />} label="Hide duplicates" />
        </Stack>
        <Stack direction={{xs:'column', sm:'row'}} spacing={2} alignItems={{xs:'stretch', sm:'center'}}>
          <Box sx={{ flexGrow:1 }}>
            <Typography variant='caption'>Min Quality: {qualityLabel(minQuality)}</Typography>
            <Slider size='small' value={minQuality} min={0} max={1} step={0.05} onChange={(_,v)=>setMinQuality(v)} />
          </Box>
          <FormControl sx={{ minWidth:180 }} size='small'>
            <InputLabel id='annotation-filter-label'>Annotation</InputLabel>
            <Select labelId='annotation-filter-label' label='Annotation' value={annotationFilter} onChange={e=>setAnnotationFilter(e.target.value)}>
              <MenuItem value=''><em>All</em></MenuItem>
              <MenuItem value='relevant'>Relevant</MenuItem>
              <MenuItem value='irrelevant'>Irrelevant</MenuItem>
              <MenuItem value='ambiguous'>Ambiguous</MenuItem>
            </Select>
          </FormControl>
          <Button size='small' variant='outlined' onClick={triggerRecomputeDuplicates} disabled={recomputing}>{recomputing? 'Recomputing...' : 'Recompute Duplicates'}</Button>
          <Button size='small' variant='outlined' onClick={triggerRecomputeQuality} disabled={recomputing}>{recomputing? 'Recomputing...' : 'Recompute Quality'}</Button>
        </Stack>
      </Stack>
      {loading && <LinearProgress />}
      {error && <Alert severity='error'>{error}</Alert>}
      <Stack spacing={1}>
        {filtered.map(c => {
          const dup = c.duplicate_of_id;
          const quality = c.clue_quality;
          return (
            <Paper key={c.id} className="clue-paper" sx={{ p:1, display:'flex', flexDirection:'column', gap:0.5 }}>
              <Stack direction='row' spacing={1} alignItems='flex-start'>
                <Box sx={{ flexGrow:1 }}>
                  <Typography variant='body2' sx={{ whiteSpace:'pre-wrap' }}>{c.text}</Typography>
                  <Stack direction='row' spacing={1} sx={{ mt:0.5, flexWrap:'wrap' }}>
                    {c.suspect_id && <Chip size='small' label={c.suspect_id} />}
                    {dup && <Tooltip title={`Duplicate of #${dup} (sim=${c.similarity})`}><Chip size='small' color='warning' label='Duplicate' /></Tooltip>}
                    {quality != null && <Chip size='small' label={`Q ${(quality*100).toFixed(0)}%`} />}
                    {c.annotation_label && <Chip size='small' color={c.annotation_label==='relevant'?'success': c.annotation_label==='irrelevant'?'default':'info'} label={c.annotation_label} />}
                    {c.source_type && <Chip size='small' variant='outlined' label={c.source_type} />}
                  </Stack>
                </Box>
                <Stack direction='row' spacing={0.5}>
                  <Tooltip title='Mark Relevant'><IconButton onClick={()=>handleAnnotate(c.id,'relevant')} size='small' color='success'><DoneIcon fontSize='small' /></IconButton></Tooltip>
                  <Tooltip title='Mark Irrelevant'><IconButton onClick={()=>handleAnnotate(c.id,'irrelevant')} size='small'><DeleteIcon fontSize='small' /></IconButton></Tooltip>
                  <Tooltip title='Mark Ambiguous'><IconButton onClick={()=>handleAnnotate(c.id,'ambiguous')} size='small'><HelpIcon fontSize='small' /></IconButton></Tooltip>
                  <IconButton onClick={()=>handleDelete(c.id)} size='small' color='error'><DeleteIcon fontSize='small' /></IconButton>
                </Stack>
              </Stack>
            </Paper>
          );
        })}
        {(!loading && filtered.length === 0) && <Typography variant='body2' color='text.secondary'>No clues match filters.</Typography>}
      </Stack>
    </Stack>
  );
}
