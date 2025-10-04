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
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>
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
    </Box>
  );
}
