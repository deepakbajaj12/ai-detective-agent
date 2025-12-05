import React, { useEffect, useState, useRef } from 'react';
import { apiFetch, API_BASE } from '../apiBase';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import Snackbar from '@mui/material/Snackbar';
import IconButton from '@mui/material/IconButton';
import PauseIcon from '@mui/icons-material/Pause';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import FormControlLabel from '@mui/material/FormControlLabel';
import Checkbox from '@mui/material/Checkbox';
import Divider from '@mui/material/Divider';
import Table from '@mui/material/Table';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import TableCell from '@mui/material/TableCell';
import TableBody from '@mui/material/TableBody';
import Chip from '@mui/material/Chip';
import TextField from '@mui/material/TextField';
import MenuItem from '@mui/material/MenuItem';
import Stack from '@mui/material/Stack';
import Alert from '@mui/material/Alert';
import Switch from '@mui/material/Switch';

function Section({ title, children, actions }) {
  return (
    <Paper elevation={3} style={{ padding: '1rem', marginBottom: '1.2rem' }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
        <Typography variant="h6">{title}</Typography>
        {actions}
      </Box>
      <Divider sx={{ mb: 2 }} />
      {children}
    </Paper>
  );
}

export default function AdminConsole() {
  const [metrics, setMetrics] = useState(null);
  const [modelVersions, setModelVersions] = useState([]);
  const [activeTag, setActiveTag] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [graph, setGraph] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [events, setEvents] = useState([]);
  const sseRef = useRef(null);
  const [role, setRole] = useState(null);
  const [eventsPaused, setEventsPaused] = useState(false);
  const [snack, setSnack] = useState({ open:false, message:'' });
  const [eventFilters, setEventFilters] = useState({ jobs:true, clues:true, feedback:true, documents:true, other:true });
  const eventFiltersRef = useRef(eventFilters);
  const eventsPausedRef = useRef(eventsPaused);

  useEffect(()=>{ eventFiltersRef.current = eventFilters; }, [eventFilters]);
  useEffect(()=>{ eventsPausedRef.current = eventsPaused; }, [eventsPaused]);
  const [regData, setRegData] = useState({ version_tag:'', model_type:'', path:'', metrics:'' });
  const [regError, setRegError] = useState(null);
  const [regLoading, setRegLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);

  useEffect(()=>{
    let cancelled = false;
    async function fetchMe(){
      try{
        const res = await apiFetch('/api/auth/me');
        if(!cancelled) setRole(res?.user?.role || null);
      }catch(e){ if(!cancelled) setRole(null); }
    }
    fetchMe();
    return ()=>{ cancelled = true; };
  },[]);

  const loadAll = async () => {
    setLoading(true);
    setError(null);
    try {
      const [m, mv, jb, gr] = await Promise.all([
        apiFetch('/api/metrics'),
        apiFetch('/api/model/versions'),
        apiFetch('/api/jobs'),
        apiFetch('/api/graph/analytics')
      ]);
      setMetrics(m);
      setModelVersions(mv);
      const act = mv.find(v=> v.role === 'active');
      setActiveTag(act ? act.version_tag : null);
      setJobs(jb.jobs || []);
      setGraph(gr);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(()=> { loadAll(); }, []);

  useEffect(()=> {
    if (!autoRefresh) return;
    const id = setInterval(loadAll, 10000);
    return ()=> clearInterval(id);
  }, [autoRefresh]);

  useEffect(()=> {
    // SSE subscription
    if (sseRef.current) return;
    const es = new EventSource(`${API_BASE}/api/events/stream`);
    es.onmessage = (ev) => {
      try {
        if (eventsPausedRef.current) return;
        const data = JSON.parse(ev.data);
        // Filter by event type
        const t = (data?.type || '').toLowerCase();
        const f = eventFiltersRef.current;
        const allow =
          (t.includes('job') && f.jobs) ||
          (t.includes('clue') && f.clues) ||
          (t.includes('feedback') && f.feedback) ||
          (t.includes('doc') && f.documents) ||
          f.other;
        if (!allow) return;
        setEvents(prev => [data, ...prev.slice(0,199)]); // cap size
      } catch {}
    };
    es.onerror = () => {
      setEvents(prev => [{ type: 'error', error: 'Event stream error' }, ...prev]);
    };
    sseRef.current = es;
    return () => { es.close(); };
  }, []);

  const triggerJob = async (jobType) => {
    try {
      let ep = null;
      if (jobType === 'transformer_train') ep = '/api/jobs/transformer_train';
      else if (jobType === 'index_refresh') ep = '/api/jobs/index_refresh';
      else if (jobType === 'embeddings_refresh') ep = '/api/jobs/embeddings_refresh';
      if (!ep) return;
      await apiFetch(ep, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) });
      loadAll();
    } catch (e) {
      setError(e.message);
    }
  };

  const promoteVersion = async (version_tag) => {
    try {
      await apiFetch('/api/model/promote', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ version_tag }) });
      await loadAll();
    } catch(e) { setError(e.message); }
  };

  const setShadow = async (version_tag) => {
    try {
      await apiFetch('/api/model/shadow', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ version_tag }) });
      await loadAll();
    } catch(e) { setError(e.message); }
  };

  const rollbackActive = async () => {
    try {
      await apiFetch('/api/model/rollback', { method:'POST' });
      await loadAll();
    } catch(e) { setError(e.message); }
  };

  const handleRegChange = (field, value) => {
    setRegData(d => ({ ...d, [field]: value }));
  };

  const submitRegistration = async () => {
    setRegError(null);
    if (!regData.version_tag || !regData.model_type || !regData.path) {
      setRegError('version_tag, model_type, and path are required');
      return;
    }
    let metricsObj = {};
    if (regData.metrics && regData.metrics.trim()) {
      try { metricsObj = JSON.parse(regData.metrics); }
      catch { setRegError('Metrics must be valid JSON'); return; }
    }
    setRegLoading(true);
    try {
      await apiFetch('/api/model/register', {
        method:'POST',
        headers:{ 'Content-Type':'application/json' },
        body: JSON.stringify({
          version_tag: regData.version_tag,
            model_type: regData.model_type,
            path: regData.path,
            metrics: metricsObj
        })
      });
      setRegData({ version_tag:'', model_type:'', path:'', metrics:'' });
      setSnack({ open:true, message:'Model registered successfully' });
      await loadAll();
    } catch(e) {
      setRegError(e.message);
    } finally {
      setRegLoading(false);
    }
  };

  return (
    <Box>
      {role !== 'admin' && <Alert severity="warning" sx={{ mb:2 }}>Forbidden: Admin role required.</Alert>}
      <Typography variant="h4" gutterBottom>Admin Console</Typography>
      {error && <Alert severity="error" sx={{ mb:2 }}>{error}</Alert>}
      <Section title="System Metrics" actions={<Stack direction="row" spacing={1} alignItems="center"><FormControlLabel control={<Switch size="small" checked={autoRefresh} onChange={e=>setAutoRefresh(e.target.checked)} />} label="Auto-refresh" /><Button size="small" onClick={loadAll} disabled={loading}>Refresh</Button></Stack>}>
        {!metrics && <Typography variant="body2">Loading...</Typography>}
        {metrics && (
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2">Entity Counts</Typography>
              <Box sx={{ mt:1 }}>
                {Object.entries(metrics.counts || {}).map(([k,v])=> (
                  <Box key={k} sx={{ mb:1 }}>
                    <Typography variant="caption">{k}: {v}</Typography>
                    <LinearProgress variant="determinate" value={Math.min(100, Number(v))} sx={{ height:6, borderRadius:1 }} />
                  </Box>
                ))}
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2">Embedding Stats</Typography>
              <Box sx={{ mt:1 }}>
                {metrics.embeddings ? Object.entries(metrics.embeddings).map(([k,v])=> (
                  <Box key={k} sx={{ mb:1 }}>
                    <Typography variant="caption">{k}: {v}</Typography>
                    <LinearProgress variant="determinate" value={Math.min(100, Number(v))} sx={{ height:6, borderRadius:1 }} />
                  </Box>
                )) : <Chip label="No embedding data" />}
              </Box>
            </Grid>
            <Grid item xs={12}>
              <Typography variant="subtitle2">Score Distribution</Typography>
              <Box sx={{ mt:1 }}>
                {metrics.scores && Object.entries(metrics.scores).map(([k,v])=> (
                  <Box key={k} sx={{ mb:1 }}>
                    <Typography variant="caption">{k}: {v}</Typography>
                    <LinearProgress variant="determinate" value={Math.min(100, Number(v)*100)} sx={{ height:6, borderRadius:1 }} />
                  </Box>
                ))}
                {!metrics.scores && <Chip label="No scores available" />}
              </Box>
            </Grid>
          </Grid>
        )}
      </Section>
      <Section title="Model Versions" actions={<Stack direction="row" spacing={1}><Button size="small" onClick={()=>loadAll()} disabled={loading}>Reload</Button><Button size="small" onClick={rollbackActive} disabled={loading || !activeTag}>Rollback</Button></Stack>}>
        <Box mb={2}>
          <Typography variant="subtitle2" gutterBottom>Register New Model</Typography>
          {regError && <Alert severity="error" sx={{ mb:1 }}>{regError}</Alert>}
          <Grid container spacing={2}>
            <Grid item xs={12} md={3}>
              <TextField label="Version Tag" size="small" fullWidth disabled={regLoading || role!=='admin'} value={regData.version_tag} onChange={e=>handleRegChange('version_tag', e.target.value)} />
            </Grid>
            <Grid item xs={12} md={3}>
              <TextField label="Model Type" size="small" select fullWidth disabled={regLoading || role!=='admin'} value={regData.model_type} onChange={e=>handleRegChange('model_type', e.target.value)}>
                <MenuItem value="transformer">transformer</MenuItem>
                <MenuItem value="embedding">embedding</MenuItem>
                <MenuItem value="other">other</MenuItem>
              </TextField>
            </Grid>
            <Grid item xs={12} md={3}>
              <TextField label="Path" size="small" fullWidth placeholder="models/model.bin" disabled={regLoading || role!=='admin'} value={regData.path} onChange={e=>handleRegChange('path', e.target.value)} />
            </Grid>
            <Grid item xs={12} md={3}>
              <Button variant="contained" size="small" fullWidth disabled={regLoading || role!=='admin'} onClick={submitRegistration}>Register</Button>
            </Grid>
            <Grid item xs={12}>
              <TextField label="Metrics JSON (optional)" size="small" fullWidth multiline minRows={2} disabled={regLoading || role!=='admin'} value={regData.metrics} onChange={e=>handleRegChange('metrics', e.target.value)} />
            </Grid>
          </Grid>
          {role!=='admin' && <Typography variant="caption" color="text.secondary">Admin role required to register models.</Typography>}
        </Box>
        <Table size="small">
          <TableHead><TableRow><TableCell>Version</TableCell><TableCell>Role</TableCell><TableCell>Type</TableCell><TableCell>Created</TableCell><TableCell align="right">Actions</TableCell></TableRow></TableHead>
          <TableBody>
            {modelVersions.map(m=> (
              <TableRow key={m.version_tag}>
                <TableCell>{m.version_tag}</TableCell>
                <TableCell>{m.role}</TableCell>
                <TableCell>{m.model_type}</TableCell>
                <TableCell>{m.created_at}</TableCell>
                <TableCell align="right">
                  <Stack direction="row" spacing={1} justifyContent="flex-end">
                    <Button size="small" disabled={m.role==='active'} onClick={()=>promoteVersion(m.version_tag)}>Promote</Button>
                    <Button size="small" disabled={m.role==='shadow'} onClick={()=>setShadow(m.version_tag)}>Shadow</Button>
                  </Stack>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        {modelVersions.length === 0 && <Typography variant="body2" sx={{ mt:1 }}>No versions registered.</Typography>}
      </Section>
      <Section title="Jobs" actions={<Stack direction="row" spacing={1}><Button size="small" onClick={()=>triggerJob('transformer_train')}>Train</Button><Button size="small" onClick={()=>triggerJob('index_refresh')}>Index Refresh</Button><Button size="small" onClick={()=>triggerJob('embeddings_refresh')}>Embeddings Refresh</Button></Stack>}>
        <Table size="small">
          <TableHead><TableRow><TableCell>ID</TableCell><TableCell>Type</TableCell><TableCell>Status</TableCell><TableCell>Started</TableCell><TableCell>Duration (s)</TableCell><TableCell align="right">Actions</TableCell></TableRow></TableHead>
          <TableBody>
            {jobs.map(j=> (
              <TableRow key={j.id}>
                <TableCell>{j.id}</TableCell>
                <TableCell>{j.job_type}</TableCell>
                <TableCell>
                  <Chip size="small" label={j.status}
                        color={j.status==='running' ? 'primary' : j.status==='completed' ? 'success' : j.status==='failed' ? 'error' : 'default'} />
                </TableCell>
                <TableCell>{j.started_at}</TableCell>
                <TableCell>{j.duration_s != null ? j.duration_s.toFixed(2) : ''}</TableCell>
                <TableCell align="right">
                  <Button size="small" disabled={j.status!=='running'} onClick={()=>apiFetch(`/api/jobs/${encodeURIComponent(j.id)}/cancel`, { method:'POST' }).then(loadAll).catch(e=>setError(e.message))}>Cancel</Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        {jobs.length === 0 && <Typography variant="body2" sx={{ mt:1 }}>No jobs found.</Typography>}
      </Section>
      <Section title="Graph Analytics" actions={<Button size="small" onClick={()=>apiFetch('/api/graph/analytics').then(setGraph).catch(e=>setError(e.message))}>Recompute</Button>}>
        {!graph && <Typography variant="body2">Loading...</Typography>}
        {graph && (
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <Typography variant="subtitle2">Node Metrics (top degree)</Typography>
              <Table size="small">
                <TableHead><TableRow><TableCell>ID</TableCell><TableCell>Deg</TableCell><TableCell>Bet</TableCell><TableCell>Anom</TableCell></TableRow></TableHead>
                <TableBody>
                  {Object.entries(graph.analytics?.node_metrics || {})
                    .sort((a,b)=> b[1].degree - a[1].degree)
                    .slice(0,10)
                    .map(([nid, m])=> <TableRow key={nid}><TableCell>{nid}</TableCell><TableCell>{m.degree}</TableCell><TableCell>{m.betweenness_centrality?.toFixed(3)}</TableCell><TableCell>{m.anomaly || ''}</TableCell></TableRow>)}
                </TableBody>
              </Table>
            </Grid>
            <Grid item xs={12} md={4}>
              <Typography variant="subtitle2">Communities</Typography>
              <Stack spacing={1} sx={{ mt:1 }}>
                {graph.analytics?.communities?.map((c,i)=> <Chip key={i} label={`C${i}: ${c.length} nodes`} />)}
                {(!graph.analytics?.communities || graph.analytics.communities.length===0) && <Chip label="None" />}
              </Stack>
            </Grid>
            <Grid item xs={12} md={4}>
              <Typography variant="subtitle2">Summary</Typography>
              <Stack spacing={1} sx={{ mt:1 }}>
                <Chip label={`Nodes: ${graph.node_count || graph.nodes?.length || 0}`} />
                <Chip label={`Edges: ${graph.edge_count || graph.edges?.length || 0}`} />
                <Chip label={`Communities: ${graph.analytics?.community_count || 0}`} />
              </Stack>
            </Grid>
          </Grid>
        )}
      </Section>
      <Section title="Realtime Events" actions={
        <IconButton size="small" onClick={()=>setEventsPaused(p=>!p)} aria-label={eventsPaused? 'Resume':'Pause'}>
          {eventsPaused ? <PlayArrowIcon /> : <PauseIcon />}
        </IconButton>
      }>
        <Stack direction="row" spacing={2} sx={{ mb:1 }}>
          <FormControlLabel control={<Checkbox size="small" checked={eventFilters.jobs} onChange={e=>setEventFilters(f=>({ ...f, jobs:e.target.checked }))} />} label="Jobs" />
          <FormControlLabel control={<Checkbox size="small" checked={eventFilters.clues} onChange={e=>setEventFilters(f=>({ ...f, clues:e.target.checked }))} />} label="Clues" />
          <FormControlLabel control={<Checkbox size="small" checked={eventFilters.feedback} onChange={e=>setEventFilters(f=>({ ...f, feedback:e.target.checked }))} />} label="Feedback" />
          <FormControlLabel control={<Checkbox size="small" checked={eventFilters.documents} onChange={e=>setEventFilters(f=>({ ...f, documents:e.target.checked }))} />} label="Documents" />
          <FormControlLabel control={<Checkbox size="small" checked={eventFilters.other} onChange={e=>setEventFilters(f=>({ ...f, other:e.target.checked }))} />} label="Other" />
        </Stack>
        <Box sx={{ maxHeight: 240, overflowY: 'auto', fontFamily: 'monospace', fontSize: 12, background:'#111', color:'#eee', p:1, borderRadius:1 }}>
          {events.map((ev,i)=> <div key={i}>{JSON.stringify(ev)}</div>)}
          {events.length===0 && <div>Waiting for events...</div>}
        </Box>
      </Section>
      <Snackbar open={snack.open} autoHideDuration={2500} onClose={()=>setSnack({ open:false, message:'' })} message={snack.message} />
    </Box>
  );
}
