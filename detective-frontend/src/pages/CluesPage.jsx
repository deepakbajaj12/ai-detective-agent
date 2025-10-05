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
import Collapse from '@mui/material/Collapse';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import Drawer from '@mui/material/Drawer';
import Checkbox from '@mui/material/Checkbox';
import FormGroup from '@mui/material/FormGroup';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import Divider from '@mui/material/Divider';
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
  const [expandedDuplicates, setExpandedDuplicates] = useState({}); // clueId -> {loading, list}
  const [onlyCanonical, setOnlyCanonical] = useState(false);
  const [duplicatePanelOpen, setDuplicatePanelOpen] = useState(false);
  const [bulkSelection, setBulkSelection] = useState({}); // id -> true
  const [bulkCanonical, setBulkCanonical] = useState(null); // chosen canonical id
  const [bulkActionBusy, setBulkActionBusy] = useState(false);

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

  const toggleDuplicates = (clue) => {
    if (expandedDuplicates[clue.id]) {
      // collapse
      setExpandedDuplicates(prev => {
        const cp = { ...prev };
        delete cp[clue.id];
        return cp;
      });
      return;
    }
    // load duplicates for canonical clue (only if it has duplicates pointing to it)
    const isCanonical = !clue.duplicate_of_id;
    if (!isCanonical) return; // only canonical expandable
    setExpandedDuplicates(prev => ({ ...prev, [clue.id]: { loading: true, list: [] }}));
    apiFetch(`/api/clues/${clue.id}/duplicates`)
      .then(data => setExpandedDuplicates(prev => ({ ...prev, [clue.id]: { loading:false, list: data.duplicates || [] }})))
      .catch(()=> setExpandedDuplicates(prev => ({ ...prev, [clue.id]: { loading:false, list: [] }})));
  };

  const filtered = clues.filter(c => {
    if(search && !c.text.toLowerCase().includes(search.toLowerCase())) return false;
    if(suspectFilter && c.suspect_id !== suspectFilter) return false;
    if(onlyCanonical && c.duplicate_of_id) return false;
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
          <FormControlLabel control={<Switch checked={onlyCanonical} onChange={e=>setOnlyCanonical(e.target.checked)} />} label="Only canonical" />
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
          <Button size='small' variant='contained' color='secondary' onClick={()=>setDuplicatePanelOpen(true)}>Duplicate Panel</Button>
        </Stack>
      </Stack>
      {loading && <LinearProgress />}
      {error && <Alert severity='error'>{error}</Alert>}
      <Stack spacing={1}>
        {filtered.map(c => {
          const dup = c.duplicate_of_id;
          const quality = c.clue_quality;
          const isCanonical = !dup;
          const expanded = expandedDuplicates[c.id];
          const subduedStyle = dup ? { opacity:0.55, backgroundColor:(theme)=> theme.palette.mode==='dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.03)' } : {};
          const checked = !!bulkSelection[c.id];
          return (
            <Paper key={c.id} className="clue-paper" sx={{ p:1, display:'flex', flexDirection:'column', gap:0.5, position:'relative', ...subduedStyle }}>
              <Stack direction='row' spacing={1} alignItems='flex-start'>
                <Box sx={{ flexGrow:1 }}>
                  <Stack direction='row' spacing={0.5} alignItems='center'>
                    {isCanonical && <IconButton size='small' onClick={()=>toggleDuplicates(c)} sx={{ transform: expanded? 'rotate(90deg)':'none', transition:'0.2s' }}>
                      {expanded ? <ExpandLessIcon fontSize='inherit' /> : <ExpandMoreIcon fontSize='inherit' />}
                    </IconButton>}
                    <Checkbox size='small' checked={checked} onChange={(e)=>{
                      const val = e.target.checked;
                      setBulkSelection(prev => {
                        const next = { ...prev };
                        if(val) next[c.id]=true; else delete next[c.id];
                        return next;
                      });
                    }} />
                    <Typography variant='body2' sx={{ whiteSpace:'pre-wrap', flexGrow:1 }}>{c.text}</Typography>
                  </Stack>
                  <Stack direction='row' spacing={1} sx={{ mt:0.5, flexWrap:'wrap' }}>
                    {c.suspect_id && <Chip size='small' label={c.suspect_id} />}
                    {dup && <Tooltip title={`Duplicate of #${dup} (sim=${c.similarity})`}><Chip size='small' color='warning' label='Duplicate' /></Tooltip>}
                    {quality != null && <Chip size='small' label={`Q ${(quality*100).toFixed(0)}%`} />}
                    {c.annotation_label && <Chip size='small' color={c.annotation_label==='relevant'?'success': c.annotation_label==='irrelevant'?'default':'info'} label={c.annotation_label} />}
                    {c.source_type && <Chip size='small' variant='outlined' label={c.source_type} />}
                    {isCanonical && c.duplicate_count > 0 && <Chip size='small' color='primary' variant='outlined' label={`Dupes ${c.duplicate_count}`} />}
                  </Stack>
                </Box>
                <Stack direction='row' spacing={0.5}>
                  <Tooltip title='Mark Relevant'><IconButton onClick={()=>handleAnnotate(c.id,'relevant')} size='small' color='success'><DoneIcon fontSize='small' /></IconButton></Tooltip>
                  <Tooltip title='Mark Irrelevant'><IconButton onClick={()=>handleAnnotate(c.id,'irrelevant')} size='small'><DeleteIcon fontSize='small' /></IconButton></Tooltip>
                  <Tooltip title='Mark Ambiguous'><IconButton onClick={()=>handleAnnotate(c.id,'ambiguous')} size='small'><HelpIcon fontSize='small' /></IconButton></Tooltip>
                  <IconButton onClick={()=>handleDelete(c.id)} size='small' color='error'><DeleteIcon fontSize='small' /></IconButton>
                </Stack>
              </Stack>
              {isCanonical && expanded && (
                <Collapse in={true} timeout='auto' unmountOnExit>
                  <Stack spacing={0.5} sx={{ pl:4, pt:1 }}>
                    {expanded.loading && <Typography variant='caption' color='text.secondary'>Loading duplicates...</Typography>}
                    {!expanded.loading && expanded.list.length === 0 && <Typography variant='caption' color='text.secondary'>No duplicates recorded.</Typography>}
                    {expanded.list.map(d => (
                      <Paper key={d.id} variant='outlined' sx={{ p:0.5, opacity:0.65 }}>
                        <Typography variant='caption' sx={{ display:'block' }}>#{d.id} sim={d.similarity ?? '?'} Q {d.clue_quality != null ? (d.clue_quality*100).toFixed(0)+'%' : 'n/a'}</Typography>
                        <Typography variant='body2' sx={{ whiteSpace:'pre-wrap' }}>{d.text}</Typography>
                      </Paper>
                    ))}
                  </Stack>
                </Collapse>
              )}
            </Paper>
          );
        })}
        {(!loading && filtered.length === 0) && <Typography variant='body2' color='text.secondary'>No clues match filters.</Typography>}
      </Stack>
      <Drawer anchor='right' open={duplicatePanelOpen} onClose={()=>setDuplicatePanelOpen(false)} sx={{ '& .MuiDrawer-paper': { width: 360, p:2 }}}>
        <Stack spacing={2} sx={{ height:'100%' }}>
          <Typography variant='h6'>Duplicate Resolution</Typography>
          <Typography variant='body2' color='text.secondary'>Select a canonical clue (radio via Set Canonical) then merge or delete others. Merge deletes selected duplicates that point to the chosen canonical.</Typography>
          <Divider />
          <Box sx={{ flexGrow:1, overflow:'auto' }}>
            <List dense>
              {filtered.filter(c=> !c.duplicate_of_id).map(c => {
                const isCanonChoice = bulkCanonical === c.id;
                return (
                  <ListItem key={c.id} alignItems='flex-start' disableGutters secondaryAction={
                    <Button size='small' variant={isCanonChoice? 'contained':'outlined'} onClick={()=>setBulkCanonical(c.id)}>Set Canonical</Button>
                  }>
                    <ListItemText primary={`#${c.id} ${(c.text||'').slice(0,50)}`} secondary={`Dupes: ${c.duplicate_count || 0}`} />
                  </ListItem>
                );
              })}
            </List>
          </Box>
          <FormGroup>
            <Typography variant='caption'>Selected: {Object.keys(bulkSelection).length}  Canonical: {bulkCanonical || 'None'}</Typography>
          </FormGroup>
          <Stack direction='row' spacing={1}>
            <Button disabled={!bulkCanonical || Object.keys(bulkSelection).length === 0 || bulkActionBusy} size='small' variant='contained' onClick={()=>{
              setBulkActionBusy(true);
              const duplicate_ids = Object.keys(bulkSelection).map(Number).filter(id => id !== bulkCanonical);
              apiFetch('/api/clues/duplicates/merge', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ canonical_id: bulkCanonical, duplicate_ids }) })
                .then(()=> { setBulkSelection({}); load(); })
                .finally(()=> setBulkActionBusy(false));
            }}>Merge (Delete Duplicates)</Button>
            <Button disabled={Object.keys(bulkSelection).length === 0 || bulkActionBusy} size='small' color='error' variant='outlined' onClick={()=>{
              setBulkActionBusy(true);
              const ids = Object.keys(bulkSelection).map(Number).filter(id => id !== bulkCanonical);
              apiFetch('/api/clues/duplicates/delete', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ ids }) })
                .then(()=> { setBulkSelection({}); load(); })
                .finally(()=> setBulkActionBusy(false));
            }}>Delete Selected</Button>
            <Button size='small' variant='text' onClick={()=>{ setBulkSelection({}); setBulkCanonical(null); }}>Clear</Button>
          </Stack>
        </Stack>
      </Drawer>
    </Stack>
  );
}
