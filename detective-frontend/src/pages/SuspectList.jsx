import React, { useEffect, useState, useCallback } from 'react';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardHeader from '@mui/material/CardHeader';
import Avatar from '@mui/material/Avatar';
import LinearProgress from '@mui/material/LinearProgress';
import Typography from '@mui/material/Typography';
import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
import Alert from '@mui/material/Alert';
import Fab from '@mui/material/Fab';
import AddIcon from '@mui/icons-material/Add';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import TextField from '@mui/material/TextField';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import Table from '@mui/material/Table';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import Divider from '@mui/material/Divider';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import Tooltip from '@mui/material/Tooltip';
import { Link as RouterLink } from 'react-router-dom';
import AddClueDialog from '../components/AddClueDialog';
import AttributionPanel from '../components/AttributionPanel';
import SimulationPanel from '../components/SimulationPanel';
import { apiFetch } from '../apiBase';
import FeedbackBar from '../components/FeedbackBar';

function scoreTier(score){
  if(score >= 0.6) return {label:'High', color:'error'};
  if(score >= 0.4) return {label:'Medium', color:'warning'};
  return {label:'Low', color:'success'};
}

export default function SuspectList() {
  const [suspects, setSuspects] = useState([]);
  const [meta, setMeta] = useState({ alpha: 0.7, offense_beta: 0.1 });
  const [sortMode, setSortMode] = useState('composite'); // 'composite' | 'severity'
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(() => {
    setLoading(true);
    apiFetch('/api/suspects')
      .then(data => {
        if(Array.isArray(data)) { // backward compatibility if backend not updated
          setSuspects(data);
        } else if(data && Array.isArray(data.suspects)) {
          setSuspects(data.suspects);
          if(data.meta) setMeta(data.meta);
        }
      })
      .catch(e => setError(e.message))
      .finally(()=> setLoading(false));
  }, []);

  useEffect(() => { load(); }, [load]);

  useEffect(() => {
    const handler = () => load();
    window.addEventListener('clueAdded', handler);
    return () => window.removeEventListener('clueAdded', handler);
  }, [load]);

  const [dialogOpen, setDialogOpen] = useState(false);
  const [latestClues, setLatestClues] = useState([]);
  const [fbStats, setFbStats] = useState(null);
  const [showAttribution, setShowAttribution] = useState(false);
  const [attributionData, setAttributionData] = useState(null);
  const [showSim, setShowSim] = useState(false);
  // Snapshot state
  const [snapshots, setSnapshots] = useState([]);
  const [snapshotA, setSnapshotA] = useState('');
  const [snapshotB, setSnapshotB] = useState('');
  const [snapshotDiff, setSnapshotDiff] = useState(null);
  const [snapshotDialogOpen, setSnapshotDialogOpen] = useState(false);
  const [snapshotLabel, setSnapshotLabel] = useState('');
  const [creatingSnapshot, setCreatingSnapshot] = useState(false);
  // Waterfall expanded per suspect
  const [waterfall, setWaterfall] = useState({}); // id -> { loading, steps }
  useEffect(()=> {
    apiFetch('/api/clues?limit=5')
      .then(data => setLatestClues(Array.isArray(data)? data: []))
      .catch(()=> setLatestClues([]));
  }, [suspects]);

  useEffect(()=> {
    apiFetch('/api/feedback/stats')
      .then(d => setFbStats(d))
      .catch(()=> setFbStats(null));
  }, [suspects]);

  // Load snapshots list
  const loadSnapshots = useCallback(() => {
    apiFetch('/api/snapshots?limit=100')
      .then(data => Array.isArray(data) && setSnapshots(data))
      .catch(()=> setSnapshots([]));
  }, []);

  useEffect(()=> { loadSnapshots(); }, [loadSnapshots]);

  // Load diff when both selected
  useEffect(()=> {
    if(snapshotA && snapshotB && snapshotA !== snapshotB){
      apiFetch(`/api/snapshots/compare?a=${snapshotA}&b=${snapshotB}`)
        .then(setSnapshotDiff)
        .catch(()=> setSnapshotDiff(null));
    } else {
      setSnapshotDiff(null);
    }
  }, [snapshotA, snapshotB]);

  // Load attribution lazily when panel opened
  useEffect(()=> {
    if(showAttribution){
      apiFetch('/api/suspects?attribution=1') // refresh & store attribution in DB
        .then(()=> {
          // pick top suspect for default detail fetch to warm attribution (optional)
          if(suspects[0]){
            apiFetch(`/api/suspects/${suspects[0].id}/attribution`)
              .then(setAttributionData)
              .catch(()=> setAttributionData({ attribution: [] }));
          }
        });
    }
  }, [showAttribution, suspects]);

  const openAttributionFor = (sid) => {
    setShowAttribution(true);
    apiFetch(`/api/suspects/${sid}/attribution`)
      .then(setAttributionData)
      .catch(()=> setAttributionData({ attribution: [] }));
  };

  const createSnapshot = () => {
    setCreatingSnapshot(true);
    apiFetch('/api/snapshots', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ label: snapshotLabel || null }) })
      .then(()=> { setSnapshotDialogOpen(false); setSnapshotLabel(''); loadSnapshots(); })
      .finally(()=> setCreatingSnapshot(false));
  };

  const toggleWaterfall = (sid) => {
    setWaterfall(prev => {
      const current = prev[sid];
      // collapse if already loaded
      if(current && !current.loading && current.open){
        return { ...prev, [sid]: { ...current, open:false } };
      }
      // if not loaded, set loading and fetch
      if(!current || (!current.steps && !current.loading)){
        setTimeout(()=> {
          apiFetch(`/api/suspects/${sid}/waterfall`)
            .then(data => {
              setWaterfall(p => ({ ...p, [sid]: { ...p[sid], loading:false, steps:data.steps || [], open:true } }));
            })
            .catch(()=> setWaterfall(p => ({ ...p, [sid]: { ...p[sid], loading:false, steps:[], open:true } })));
        },0);
        return { ...prev, [sid]: { loading:true, steps:null, open:true } };
      }
      // already have steps but was closed
      return { ...prev, [sid]: { ...current, open:true } };
    });
  };

  if (loading) return <LinearProgress />;
  if (error) return <Alert severity="error">{error}</Alert>;

  const isEmpty = !loading && !error && suspects.length === 0;

  // derived sorted list
  const displayed = [...suspects];
  if(sortMode === 'severity') {
    const rank = { high: 3, medium: 2, low: 1 };
    displayed.sort((a,b)=> (rank[b.primary_offense_severity?.toLowerCase()]||0) - (rank[a.primary_offense_severity?.toLowerCase()]||0));
  } else {
    displayed.sort((a,b)=> (b.composite_score||0) - (a.composite_score||0));
  }

  return (
    <Box sx={{ position:'relative' }}>
      <Box className="hero-band" sx={{ mb:4 }}>
        <Typography variant="h4" sx={{ mb:1, fontFamily:'Playfair Display, serif', letterSpacing:'0.07em' }}>Active Suspects</Typography>
        {isEmpty ? (
          <Typography variant="body2" sx={{ maxWidth:760 }} color="text.secondary">
            No suspects loaded yet. Ensure the backend API is running at /api and add a suspect or clues to populate rankings.
          </Typography>
        ) : (
          <Stack spacing={1} sx={{ maxWidth:760 }}>
            <Typography variant="body2" sx={{ lineHeight:1.4 }}>
              A consolidated dossier of current persons of interest. Composite scores blend ML inference (alpha {meta.alpha}) with evidence weights and an offense severity boost (β {meta.offense_beta}).
            </Typography>
            <Stack direction={"row"} spacing={1} alignItems="center" sx={{ flexWrap:'wrap' }}>
              <Chip size="small" label="Sort: Composite" color={sortMode==='composite'?'primary':'default'} onClick={()=> setSortMode('composite')} />
              <Chip size="small" label="Sort: Severity" color={sortMode==='severity'?'primary':'default'} onClick={()=> setSortMode('severity')} />
              <Chip size="small" variant="outlined" label="Severity Legend" />
              <Chip size="small" label="Attribution" color={showAttribution?'secondary':'default'} onClick={()=> setShowAttribution(s => !s)} />
              <Chip size="small" label="Simulate" color={showSim?'secondary':'default'} onClick={()=> setShowSim(s => !s)} />
              <Chip size="small" label="Create Snapshot" color="info" onClick={()=> setSnapshotDialogOpen(true)} />
              <Stack direction="row" spacing={1} alignItems="center">
                <Chip size="small" label="High" color="error" />
                <Chip size="small" label="Medium" color="warning" />
                <Chip size="small" label="Low" />
              </Stack>
              <Chip size="small" label={meta.ml_backend==='transformer' ? 'ML: Transformer' : 'ML: Logistic'} color={meta.ml_backend==='transformer' ? 'success':'default'} />
              {fbStats && fbStats.total > 0 && (
                <Chip size="small" variant="outlined" label={`Feedback: ${fbStats.total} (P@1 ${fbStats.precision_at_1_proxy!==null && fbStats.precision_at_1_proxy!==undefined ? (fbStats.precision_at_1_proxy*100).toFixed(0)+'%' : '—'})`} />
              )}
            </Stack>
            {/* Snapshot comparison controls */}
            {snapshots.length > 0 && (
              <Stack direction={{ xs:'column', sm:'row' }} spacing={1} alignItems={{ xs:'stretch', sm:'center' }}>
                <FormControl size="small" sx={{ minWidth:160 }}>
                  <InputLabel id="snap-a-label">Snapshot A</InputLabel>
                  <Select labelId="snap-a-label" label="Snapshot A" value={snapshotA} onChange={e=> setSnapshotA(e.target.value)}>
                    <MenuItem value=""><em>None</em></MenuItem>
                    {snapshots.map(s => <MenuItem key={s.id} value={s.id}>{s.label || `#${s.id}`} </MenuItem>)}
                  </Select>
                </FormControl>
                <FormControl size="small" sx={{ minWidth:160 }}>
                  <InputLabel id="snap-b-label">Snapshot B</InputLabel>
                  <Select labelId="snap-b-label" label="Snapshot B" value={snapshotB} onChange={e=> setSnapshotB(e.target.value)}>
                    <MenuItem value=""><em>None</em></MenuItem>
                    {snapshots.map(s => <MenuItem key={s.id} value={s.id}>{s.label || `#${s.id}`} </MenuItem>)}
                  </Select>
                </FormControl>
                {snapshotDiff && <Chip size="small" color="secondary" variant="outlined" label={`Δ suspects: ${snapshotDiff.diffs.length}`} />}
              </Stack>
            )}
          </Stack>
        )}
      </Box>
      <Grid container spacing={2}>
        {displayed.map((s, idx) => {
          const tier = scoreTier(s.composite_score ?? s.score ?? 0);
          const mlPct = ((s.score||0)*100).toFixed(1);
          const evPct = ((s.evidence_score||0)*100).toFixed(1);
          const compPct = ((s.composite_score||s.score||0)*100).toFixed(1);
          const boostPct = ((s.offense_boost||0)*100).toFixed(1);
          return (
            <Grid key={s.id} item xs={12} sm={6} md={4}>
              <Card component={RouterLink} to={`/suspects/${s.id}`} className="case-card" sx={{ textDecoration: 'none', position:'relative', background: 'linear-gradient(135deg, rgba(255,255,255,0.4), rgba(255,255,255,0.15))' }}>
                <div className={`risk-ribbon ${tier.label.toLowerCase()}`}>{tier.label}</div>
                <CardHeader
                  avatar={<Avatar src={s.avatar} alt={s.name} />}
                  title={<Stack direction="row" spacing={1} alignItems="center">
                    <span>{s.name}</span>
                    <Chip size="small" label={tier.label} color={tier.color} />
                    {s.primary_offense && (
                      <Tooltip
                        placement="top"
                        arrow
                        title={(() => {
                          if(!s.allegations || s.allegations.length === 0) return s.primary_offense;
                          const lines = s.allegations.map(a => `${a.offense} (${a.severity})${a.description?': '+a.description:''}`);
                          return <Box sx={{ whiteSpace:'pre-line' }}>{lines.join('\n')}</Box>;
                        })()}
                      >
                        <Chip
                          size="small"
                          variant="outlined"
                          label={`${s.primary_offense}${s.allegation_count>1?` (+${s.allegation_count-1} more)`:''}`}
                          color={s.primary_offense_severity === 'high' ? 'error' : s.primary_offense_severity === 'medium' ? 'warning' : 'default'}
                          onClick={(e)=> { e.preventDefault(); /* allow tooltip only */ }}
                        />
                      </Tooltip>
                    )}
                    {showAttribution && <Chip size="small" label="View Attr" variant="outlined" onClick={(e)=> { e.preventDefault(); openAttributionFor(s.id); }} />}
                    <Chip size="small" label={waterfall[s.id]?.open ? 'Hide Steps' : 'Waterfall'} variant="outlined" onClick={(e)=> { e.preventDefault(); toggleWaterfall(s.id); }} />
                  </Stack>}
                  subheader={`Composite: ${compPct}% (ML ${mlPct}%, EV ${evPct}%, Offense +${boostPct}%)`}
                />
                <CardContent>
                  <LinearProgress variant="determinate" value={parseFloat(compPct)} color={tier.color} sx={{ mb:1, height:8, borderRadius:4 }} />
                  <Typography variant="body2" color="text.secondary">{s.bio}</Typography>
                  <Stack direction="row" spacing={1} sx={{ mt: 1, flexWrap: 'wrap' }}>
                    {(s.tags || []).map((t) => (<Chip key={t} size="small" label={t} />))}
                  </Stack>
                  <FeedbackBar suspect={s} rank={idx} />
                  {waterfall[s.id]?.open && (
                    <Box sx={{ mt:1.5, p:1, border:'1px dashed rgba(191,164,111,0.4)', borderRadius:1 }}>
                      <Typography variant="caption" color="text.secondary" sx={{ display:'block', mb:0.5 }}>Score Waterfall</Typography>
                      {waterfall[s.id].loading && <Typography variant="caption">Loading breakdown...</Typography>}
                      {!waterfall[s.id].loading && waterfall[s.id].steps && waterfall[s.id].steps.length > 0 && (
                        <Stack spacing={0.5}>
                          {(() => {
                            const steps = waterfall[s.id].steps;
                            const maxVal = Math.max(...steps.map(st => Math.abs(st.value)) , 0.0001);
                            return steps.map(st => (
                              <Stack key={st.label} direction="row" spacing={1} alignItems="center">
                                <Typography variant="caption" sx={{ width:90 }}>{st.label}</Typography>
                                <Box sx={{ flexGrow:1, position:'relative', height:10, background:(theme)=> theme.palette.mode==='dark' ? '#222' : '#eee', borderRadius:4 }}>
                                  <Box sx={{ position:'absolute', left:0, top:0, bottom:0, width:`${(Math.abs(st.value)/maxVal)*100}%`, background: st.type==='increment' ? 'linear-gradient(90deg,#66bb6a,#43a047)' : st.type==='base' ? 'linear-gradient(90deg,#42a5f5,#1e88e5)' : st.type==='result' ? 'linear-gradient(90deg,#7e57c2,#5e35b1)' : st.type==='total' ? 'linear-gradient(90deg,#ef5350,#d32f2f)' : '#bfa46f', borderRadius:4 }} />
                                </Box>
                                <Typography variant="caption" sx={{ width:48, textAlign:'right' }}>{(st.value*100).toFixed(1)}%</Typography>
                              </Stack>
                            ));
                          })()}
                        </Stack>
                      )}
                      {!waterfall[s.id].loading && (!waterfall[s.id].steps || waterfall[s.id].steps.length===0) && <Typography variant="caption" color="text.secondary">No breakdown available.</Typography>}
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>
      {/* Snapshot diff table */}
      {snapshotDiff && snapshotDiff.diffs && (
        <Box sx={{ mt:6 }}>
          <Typography variant="h6" sx={{ mb:1 }}>Snapshot Comparison</Typography>
          <Table size="small" sx={{ maxWidth:860 }}>
            <TableHead>
              <TableRow>
                <TableCell>Suspect ID</TableCell>
                <TableCell>Name</TableCell>
                <TableCell align="right">A</TableCell>
                <TableCell align="right">B</TableCell>
                <TableCell align="right">Δ</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {snapshotDiff.diffs.map(d => {
                const delta = d.delta ?? 0;
                const color = delta > 0 ? 'success.main' : delta < 0 ? 'error.main' : 'text.secondary';
                return (
                  <TableRow key={d.id} hover>
                    <TableCell>{d.id}</TableCell>
                    <TableCell>{d.name}</TableCell>
                    <TableCell align="right">{d.a_composite != null ? (d.a_composite*100).toFixed(1)+'%' : '—'}</TableCell>
                    <TableCell align="right">{d.b_composite != null ? (d.b_composite*100).toFixed(1)+'%' : '—'}</TableCell>
                    <TableCell align="right" sx={{ color }}>{d.delta != null ? (delta*100).toFixed(1)+'%' : '—'}</TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
          <Typography variant="caption" color="text.secondary" sx={{ display:'block', mt:0.5 }}>Δ column = B - A (increase positive).</Typography>
        </Box>
      )}
      {!isEmpty && (
        <>
          <Divider sx={{ my:4 }} />
          <Typography variant="h6" sx={{ mb:1 }}>Latest Clues</Typography>
          <List dense>
            {latestClues.map(c => (
              <ListItem key={c.id} disableGutters secondaryAction={c.suspect_id && <Chip size='small' label={c.suspect_id} /> }>
                <ListItemText primary={c.text} />
              </ListItem>
            ))}
            {latestClues.length === 0 && <Typography variant='body2' color='text.secondary'>No clues yet.</Typography>}
          </List>
        </>
      )}
      <Tooltip title="Add Clue">
        <Fab color="primary" sx={{ position:'fixed', bottom: 32, right: 32 }} onClick={() => setDialogOpen(true)}>
          <AddIcon />
        </Fab>
      </Tooltip>
      <AddClueDialog open={dialogOpen} onClose={() => setDialogOpen(false)} />
      <AttributionPanel open={showAttribution} onClose={()=> setShowAttribution(false)} data={attributionData} />
      <SimulationPanel open={showSim} onClose={()=> setShowSim(false)} suspects={displayed} />
      <Dialog open={snapshotDialogOpen} onClose={()=> setSnapshotDialogOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle>Create Snapshot</DialogTitle>
        <DialogContent>
          <TextField fullWidth size="small" label="Label (optional)" value={snapshotLabel} onChange={e=> setSnapshotLabel(e.target.value)} sx={{ mt:1 }} />
          <Typography variant="caption" color="text.secondary" sx={{ display:'block', mt:1 }}>Captures current suspect composite scores for later comparison.</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={()=> setSnapshotDialogOpen(false)} disabled={creatingSnapshot}>Cancel</Button>
          <Button onClick={createSnapshot} variant="contained" disabled={creatingSnapshot}>{creatingSnapshot? 'Creating...':'Create'}</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
