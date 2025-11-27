import React, { useEffect, useState, useRef } from 'react';
import { apiFetch, API_BASE } from '../apiBase';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import Divider from '@mui/material/Divider';
import Table from '@mui/material/Table';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import TableCell from '@mui/material/TableCell';
import TableBody from '@mui/material/TableBody';
import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
import Alert from '@mui/material/Alert';

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
    // SSE subscription
    if (sseRef.current) return;
    const es = new EventSource(`${API_BASE}/api/events/stream`);
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
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

  return (
    <Box>
      <Typography variant="h4" gutterBottom>Admin Console</Typography>
      {error && <Alert severity="error" sx={{ mb:2 }}>{error}</Alert>}
      <Section title="System Metrics" actions={<Button size="small" onClick={loadAll} disabled={loading}>Refresh</Button>}>
        {!metrics && <Typography variant="body2">Loading...</Typography>}
        {metrics && (
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2">Entity Counts</Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ mt:1 }}>
                {Object.entries(metrics.counts || {}).map(([k,v])=> <Chip key={k} label={`${k}: ${v}`} />)}
              </Stack>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2">Embedding Stats</Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ mt:1 }}>
                {metrics.embeddings ? Object.entries(metrics.embeddings).map(([k,v])=> <Chip key={k} label={`${k}: ${v}`} />) : <Chip label="No embedding data" />}
              </Stack>
            </Grid>
            <Grid item xs={12}>
              <Typography variant="subtitle2">Score Distribution</Typography>
              <Stack direction="row" spacing={1} sx={{ mt:1 }}>
                {metrics.scores && Object.entries(metrics.scores).map(([k,v])=> <Chip key={k} label={`${k}: ${v}`} />)}
              </Stack>
            </Grid>
          </Grid>
        )}
      </Section>
      <Section title="Model Versions" actions={<Stack direction="row" spacing={1}><Button size="small" onClick={()=>loadAll()} disabled={loading}>Reload</Button><Button size="small" onClick={rollbackActive} disabled={loading || !activeTag}>Rollback</Button></Stack>}>
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
          <TableHead><TableRow><TableCell>ID</TableCell><TableCell>Type</TableCell><TableCell>Status</TableCell><TableCell>Started</TableCell><TableCell>Duration (s)</TableCell></TableRow></TableHead>
          <TableBody>
            {jobs.map(j=> <TableRow key={j.id}><TableCell>{j.id}</TableCell><TableCell>{j.job_type}</TableCell><TableCell>{j.status}</TableCell><TableCell>{j.started_at}</TableCell><TableCell>{j.duration_s != null ? j.duration_s.toFixed(2) : ''}</TableCell></TableRow>)}
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
      <Section title="Realtime Events" actions={null}>
        <Box sx={{ maxHeight: 240, overflowY: 'auto', fontFamily: 'monospace', fontSize: 12, background:'#111', color:'#eee', p:1, borderRadius:1 }}>
          {events.map((ev,i)=> <div key={i}>{JSON.stringify(ev)}</div>)}
          {events.length===0 && <div>Waiting for events...</div>}
        </Box>
      </Section>
    </Box>
  );
}
