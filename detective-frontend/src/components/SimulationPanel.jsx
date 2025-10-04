import React, { useEffect, useState } from 'react';
import Drawer from '@mui/material/Drawer';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
import Stack from '@mui/material/Stack';
import Slider from '@mui/material/Slider';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';
import LinearProgress from '@mui/material/LinearProgress';
import DeleteIcon from '@mui/icons-material/Delete';
import { apiFetch } from '../apiBase';

export default function SimulationPanel({ open, onClose, suspects }){
  const [alpha, setAlpha] = useState(0.7);
  const [offenseBeta, setOffenseBeta] = useState(0.1);
  const [evidenceOverrides, setEvidenceOverrides] = useState({}); // sid -> total weight
  const [offenseRemovals, setOffenseRemovals] = useState({}); // sid -> Set(offense substr)
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState(null);
  const [snapshots, setSnapshots] = useState([]); // { id, ts, alpha, offenseBeta, data }

  useEffect(()=> {
    if(!open){
      setResults(null);
    }
  }, [open]);

  const toggleRemoval = (sid, offense) => {
    setOffenseRemovals(prev => {
      const cur = new Set(prev[sid] || []);
      if(cur.has(offense)) cur.delete(offense); else cur.add(offense);
      return { ...prev, [sid]: cur };
    });
  };

  const runSim = () => {
    setRunning(true);
    const payload = {
      overrides: {
        alpha,
        offense_beta: offenseBeta,
        evidence_weights: Object.fromEntries(Object.entries(evidenceOverrides).filter(([_,v])=> typeof v === 'number' && !isNaN(v))),
        remove_offenses: Object.fromEntries(Object.entries(offenseRemovals).map(([k,v])=> [k, Array.from(v)]))
      }
    };
    apiFetch('/api/simulate', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) })
      .then(setResults)
      .catch(()=> setResults({ error: 'simulation failed'}))
      .finally(()=> setRunning(false));
  };

  const snapshot = () => {
    if(!results || results.error) return;
    const id = Math.random().toString(36).slice(2,9);
    const snap = { id, ts: new Date().toLocaleTimeString(), alpha, offenseBeta, data: results.simulated };
    setSnapshots(prev => [snap, ...prev].slice(0,8));
  };

  return (
    <Drawer anchor="left" open={open} onClose={onClose} PaperProps={{ sx:{ width:{ xs:'100%', md:480 }, p:2 } }}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb:1 }}>
        <Typography variant="h6">What‑If Simulator</Typography>
        <IconButton size="small" onClick={onClose}><CloseIcon fontSize="small" /></IconButton>
      </Stack>
      <Typography variant="body2" color="text.secondary" sx={{ mb:2 }}>Adjust alpha, offense impact, evidence totals, and remove selected offenses to see hypothetical composite changes. Snapshots let you compare scenarios quickly.</Typography>
      <Box sx={{ mb:2 }}>
        <Typography variant="caption">Alpha (ML weight) {alpha.toFixed(2)}</Typography>
        <Slider min={0} max={1} step={0.05} value={alpha} onChange={(_,v)=> setAlpha(v)} />
        <Typography variant="caption">Offense β {offenseBeta.toFixed(2)}</Typography>
        <Slider min={0} max={0.5} step={0.05} value={offenseBeta} onChange={(_,v)=> setOffenseBeta(v)} />
      </Box>
      <Divider sx={{ my:1 }} />
      <Stack spacing={2} sx={{ maxHeight:220, overflowY:'auto', pr:1 }}>
        {suspects.map(s => (
          <Box key={s.id} sx={{ border:'1px solid rgba(191,164,111,0.3)', p:1, borderRadius:1 }}>
            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Typography variant="subtitle2">{s.name}</Typography>
              <Typography variant="caption" color="text.secondary">Current {(s.composite_score*100).toFixed(1)}%</Typography>
            </Stack>
            <Typography variant="caption" color="text.secondary">Evidence total override</Typography>
            <Slider size="small" min={0} max={5} step={0.25} value={evidenceOverrides[s.id] ?? (s.evidence_score * 5)} onChange={(_,v)=> setEvidenceOverrides(o=> ({...o, [s.id]: v}))} />
            {s.allegations && s.allegations.length>0 && (
              <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ mt:0.5 }}>
                {s.allegations.map(a => {
                  const active = offenseRemovals[s.id]?.has(a.offense);
                  return <Chip key={a.id} size="small" label={a.offense} color={active?'default':(a.severity==='high'?'error':a.severity==='medium'?'warning':'default')} variant={active?'outlined':'filled'} onClick={()=> toggleRemoval(s.id, a.offense)} />;
                })}
              </Stack>
            )}
          </Box>
        ))}
      </Stack>
      <Stack direction="row" spacing={1} sx={{ mt:2 }}>
        <Button variant="contained" size="small" disabled={running} onClick={runSim}>Run</Button>
        <Button variant="outlined" size="small" disabled={!results || results.error} onClick={snapshot}>Snapshot</Button>
        <Button variant="text" size="small" onClick={()=> { setEvidenceOverrides({}); setOffenseRemovals({}); }}>Reset</Button>
      </Stack>
      {running && <LinearProgress sx={{ mt:1 }} />}
      {results && !results.error && (
        <Box sx={{ mt:2 }}>
          <Typography variant="subtitle2">Results (α {results.alpha.toFixed(2)}, β {results.offense_beta.toFixed(2)})</Typography>
          <Stack spacing={1} sx={{ mt:1, maxHeight:180, overflowY:'auto' }}>
            {results.simulated.map(r => (
              <Stack key={r.suspect_id} direction="row" spacing={1} alignItems="center" sx={{ fontSize:'.8rem' }}>
                <Box sx={{ width:110, fontWeight:500 }}>{r.suspect_id}</Box>
                <LinearProgress variant="determinate" value={r.composite_score*100} sx={{ flexGrow:1, height:6, borderRadius:3 }} />
                <Box sx={{ width:64, textAlign:'right' }}>{(r.composite_score*100).toFixed(1)}%</Box>
                <Box sx={{ width:56, textAlign:'right', color: r.delta>0?'#1b7f1b': r.delta<0?'#a33':'inherit' }}>{r.delta!==null ? (r.delta*100).toFixed(1)+'%' : '—'}</Box>
              </Stack>
            ))}
          </Stack>
        </Box>
      )}
      {results && results.error && <Typography color="error" variant="body2" sx={{ mt:2 }}>{results.error}</Typography>}
      {snapshots.length>0 && (
        <Box sx={{ mt:3 }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Typography variant="subtitle2">Snapshots</Typography>
            <IconButton size="small" onClick={()=> setSnapshots([])} title="Clear"><DeleteIcon fontSize="small" /></IconButton>
          </Stack>
          <Stack spacing={1} sx={{ mt:1, maxHeight:140, overflowY:'auto' }}>
            {snapshots.map(s => (
              <Box key={s.id} sx={{ border:'1px solid rgba(255,255,255,0.15)', p:0.75, borderRadius:1 }}>
                <Typography variant="caption" sx={{ fontWeight:600 }}>{s.ts} α{s.alpha.toFixed(2)} β{s.offenseBeta.toFixed(2)}</Typography>
                <Stack spacing={0.25} sx={{ mt:0.5 }}>
                  {s.data.slice(0,4).map(r => (
                    <Stack key={r.suspect_id} direction="row" spacing={1} alignItems="center" sx={{ fontSize:'.7rem' }}>
                      <Box sx={{ width:90 }}>{r.suspect_id}</Box>
                      <LinearProgress variant="determinate" value={r.composite_score*100} sx={{ flexGrow:1, height:5, borderRadius:3 }} />
                      <Box sx={{ width:52, textAlign:'right' }}>{(r.composite_score*100).toFixed(1)}%</Box>
                    </Stack>
                  ))}
                </Stack>
              </Box>
            ))}
          </Stack>
        </Box>
      )}
    </Drawer>
  );
}
