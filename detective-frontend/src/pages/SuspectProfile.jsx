import React, { useEffect, useState } from 'react';
import { useParams, Link as RouterLink } from 'react-router-dom';
import Typography from '@mui/material/Typography';
import Avatar from '@mui/material/Avatar';
import Stack from '@mui/material/Stack';
import Chip from '@mui/material/Chip';
import Alert from '@mui/material/Alert';
import LinearProgress from '@mui/material/LinearProgress';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Button from '@mui/material/Button';
import Slider from '@mui/material/Slider';
import IconButton from '@mui/material/IconButton';
import SaveIcon from '@mui/icons-material/Save';
import RefreshIcon from '@mui/icons-material/Refresh';
import InsightsIcon from '@mui/icons-material/Insights';
import { apiFetch } from '../apiBase';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import TextField from '@mui/material/TextField';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import DeleteIcon from '@mui/icons-material/Delete';
import AddIcon from '@mui/icons-material/Add';

export default function SuspectProfile() {
  const { sid } = useParams();
  const [suspect, setSuspect] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [explaining, setExplaining] = useState(false);
  const [addOpen, setAddOpen] = useState(false);
  const [newOffense, setNewOffense] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [newSeverity, setNewSeverity] = useState('medium');

  useEffect(() => {
    setLoading(true);
    apiFetch(`/api/suspects/${sid}`)
      .then(setSuspect)
      .catch(e => setError(e.message))
      .finally(()=> setLoading(false));
  }, [sid]);

  const updateEvidence = (ev) => {
    apiFetch(`/api/evidence/${ev.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ weight: ev.weight })
    }).catch(()=>{/* ignore */});
  };

  const handleRescore = () => {
    apiFetch('/api/rescore', { method: 'POST' })
      .finally(()=> {
        setLoading(true);
        apiFetch(`/api/suspects/${sid}`)
          .then(setSuspect)
          .catch(e => setError(e.message))
          .finally(()=> setLoading(false));
      });
  };

  const refresh = () => {
    setLoading(true);
    apiFetch(`/api/suspects/${sid}`)
      .then(setSuspect)
      .catch(e => setError(e.message))
      .finally(()=> setLoading(false));
  };

  if (loading) return <LinearProgress />;
  if (error) return <Alert severity="error">{error}</Alert>;
  if (!suspect) return <Alert severity="warning">Not found</Alert>;

  return (
    <Stack spacing={2}>
      <Stack direction="row" spacing={2} alignItems="center">
        <Avatar src={suspect.avatar} alt={suspect.name} sx={{ width: 64, height: 64 }} />
        <Typography variant="h4">{suspect.name}</Typography>
        <Stack direction="row" spacing={1}>
          {(suspect.tags || []).map((t) => (<Chip key={t} label={t} />))}
        </Stack>
      </Stack>

      <Typography variant="body1">{suspect.bio}</Typography>
      <Card>
        <CardContent>
          <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb:1 }}>
            <Typography variant="h6">Allegations / Suspected Offenses</Typography>
            <Button size="small" startIcon={<AddIcon />} onClick={()=> setAddOpen(true)}>Add</Button>
          </Stack>
          {suspect.allegations && suspect.allegations.length > 0 ? (
            <Stack spacing={1}>
              {suspect.allegations.map(a => (
                <Stack key={a.id} direction="row" spacing={1} alignItems="center" sx={{ border:'1px solid #bfa46f55', p:1, borderRadius:1 }}>
                  <Chip size="small" label={a.offense} color={a.severity === 'high' ? 'error' : a.severity === 'medium' ? 'warning' : 'default'} />
                  <Typography variant="body2" sx={{ flexGrow:1 }}>{a.description}</Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ textTransform:'capitalize' }}>{a.severity}</Typography>
                  <IconButton size="small" onClick={(e)=>{ e.stopPropagation(); apiFetch(`/api/allegations/${a.id}`, { method:'DELETE'}).then(()=> refresh()); }}><DeleteIcon fontSize="small" /></IconButton>
                </Stack>
              ))}
            </Stack>
          ) : (
            <Typography variant="body2" color="text.secondary">No allegations recorded.</Typography>
          )}
          <Typography variant="caption" color="text.secondary" sx={{ mt:1, display:'block' }}>Severity impacts composite via additive offense boost. Add high-severity offenses judiciously.</Typography>
        </CardContent>
      </Card>
      <Stack direction="row" spacing={1} alignItems="center" sx={{ flexWrap:'wrap' }}>
  {suspect.composite_score !== undefined && <Chip size="small" label={`Composite ${(suspect.composite_score*100).toFixed(1)}%`} color="primary" />}
  {suspect.offense_boost !== undefined && suspect.offense_boost > 0 && <Chip size="small" variant="outlined" label={`Offense +${(suspect.offense_boost*100).toFixed(1)}%`} color="error" />}
        {suspect.score !== undefined && <Chip size="small" label={`ML ${(suspect.score*100).toFixed(1)}%`} />}
        {suspect.evidence_score !== undefined && <Chip size="small" label={`Evidence ${(suspect.evidence_score*100).toFixed(1)}%`} color="secondary" />}
        {suspect.risk_level && <Chip size="small" label={suspect.risk_level} color={suspect.risk_level==='High'?'error':suspect.risk_level==='Medium'?'warning':'success'} />}
        {suspect.last_scored_at && <Typography variant="caption" color="text.secondary">@ {suspect.last_scored_at}</Typography>}
        <Button size="small" startIcon={<InsightsIcon />} disabled={explaining} onClick={()=>{
          setExplaining(true);
          apiFetch(`/api/suspects/${sid}/explain`)
            .then(setExplanation)
            .catch(()=> setExplanation({ error: 'Failed to load explanation'}))
            .finally(()=> setExplaining(false));
        }}>Explain</Button>
      </Stack>

      <Card>
        <CardContent>
          <Typography variant="h6">Related Clues</Typography>
          {suspect.relatedClues && suspect.relatedClues.length > 0 ? (
            <ul>
              {suspect.relatedClues.map((c, idx) => (<li key={idx}>{c}</li>))}
            </ul>
          ) : (
            <Typography variant="body2" color="text.secondary">No directly related clues.</Typography>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Stack direction="row" spacing={2} alignItems="center" justifyContent="space-between">
            <Typography variant="h6">Evidence</Typography>
            <IconButton size="small" onClick={handleRescore} title="Rescore suspects"><RefreshIcon fontSize="small" /></IconButton>
          </Stack>
          {suspect.evidence && suspect.evidence.length > 0 ? (
            <Stack spacing={2} sx={{ mt:1 }}>
              {suspect.evidence.map((e, idx) => (
                <Card key={e.id || idx} variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2">{e.type || 'evidence'}</Typography>
                    <Typography variant="body2" sx={{ mb:1 }}>{e.summary}</Typography>
                    <Stack direction="row" spacing={2} alignItems="center">
                      <Slider
                        value={typeof e.weight === 'number' ? e.weight : 0}
                        step={0.05}
                        min={0}
                        max={1}
                        marks={[{value:0,label:'0'},{value:1,label:'1'}]}
                        onChange={(_, val)=> {
                          const nw = Array.from(suspect.evidence);
                          nw[idx] = { ...e, weight: val };
                          setSuspect({...suspect, evidence: nw});
                        }}
                        sx={{ flexGrow:1 }}
                      />
                      <IconButton size="small" color="primary" onClick={()=>updateEvidence(e)} title="Save weight"><SaveIcon fontSize="small" /></IconButton>
                      <Typography variant="caption" sx={{ width:38, textAlign:'right' }}>{(e.weight||0).toFixed(2)}</Typography>
                    </Stack>
                  </CardContent>
                </Card>
              ))}
            </Stack>
          ) : (
            <Typography variant="body2" color="text.secondary">No evidence listed.</Typography>
          )}
        </CardContent>
      </Card>

      {explanation && (
        <Card>
          <CardContent>
            <Typography variant="h6">Explanation</Typography>
            {explanation.error && <Typography color='error' variant='body2'>{explanation.error}</Typography>}
            {explanation.top_tokens && explanation.top_tokens.length > 0 ? (
              <Stack spacing={1} sx={{ mt:1 }}>
                {explanation.top_tokens.map(t => (
                  <Stack key={t.token} direction='row' spacing={1} alignItems='center'>
                    <Chip size='small' label={t.token} />
                    <LinearProgress variant='determinate' value={Math.min(100, (t.weight / explanation.top_tokens[0].weight) * 100)} sx={{ flexGrow:1, height:6, borderRadius:3 }} />
                    <Typography variant='caption'>{t.weight.toFixed(3)}</Typography>
                  </Stack>
                ))}
              </Stack>
            ) : (!explanation.error && <Typography variant='body2' color='text.secondary'>No positive contributing tokens.</Typography>)}
          </CardContent>
        </Card>
      )}
      <Button variant="outlined" component={RouterLink} to="/suspects">Back to list</Button>

      <Dialog open={addOpen} onClose={()=> setAddOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add Allegation</DialogTitle>
        <DialogContent sx={{ pt:1 }}>
          <Stack spacing={2} sx={{ mt:1 }}>
            <TextField label="Offense" value={newOffense} onChange={e=> setNewOffense(e.target.value)} fullWidth size="small" />
            <TextField label="Description" value={newDesc} onChange={e=> setNewDesc(e.target.value)} fullWidth size="small" multiline minRows={2} />
            <FormControl size="small">
              <InputLabel id="severity-label">Severity</InputLabel>
              <Select labelId="severity-label" label="Severity" value={newSeverity} onChange={e=> setNewSeverity(e.target.value)}>
                <MenuItem value="low">Low</MenuItem>
                <MenuItem value="medium">Medium</MenuItem>
                <MenuItem value="high">High</MenuItem>
              </Select>
            </FormControl>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={()=> setAddOpen(false)}>Cancel</Button>
            <Button variant="contained" onClick={()=> {
              if(!newOffense.trim()) return;
              apiFetch(`/api/suspects/${sid}/allegations`, {
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body: JSON.stringify({ offense: newOffense.trim(), description: newDesc.trim(), severity: newSeverity })
              }).then(()=> {
                setAddOpen(false);
                setNewOffense(''); setNewDesc(''); setNewSeverity('medium');
                refresh();
              });
            }}>Save</Button>
        </DialogActions>
      </Dialog>
    </Stack>
  );
}
