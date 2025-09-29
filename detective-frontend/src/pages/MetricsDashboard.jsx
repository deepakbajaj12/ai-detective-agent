import React, { useEffect, useState } from 'react';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Chip from '@mui/material/Chip';
import LinearProgress from '@mui/material/LinearProgress';
import Stack from '@mui/material/Stack';
import Divider from '@mui/material/Divider';
import RefreshIcon from '@mui/icons-material/Refresh';
import IconButton from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';
import { apiFetch } from '../apiBase';

function StatCard({ title, children, accent }) {
  return (
    <Card sx={{ height:'100%', background: accent ? 'linear-gradient(135deg,#f5e7c5,#e5d4a5)' : undefined }}>
      <CardContent>
        <Typography variant='subtitle2' sx={{ mb:1, fontWeight:600 }}>{title}</Typography>
        {children}
      </CardContent>
    </Card>
  );
}

export default function MetricsDashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = () => {
    setLoading(true);
    apiFetch('/api/metrics')
      .then(d => { setData(d); setError(null); })
      .catch(e => setError(e.message))
      .finally(()=> setLoading(false));
  };

  useEffect(()=> { load(); }, []);

  if (loading) return <LinearProgress />;
  if (error) return <Typography color='error'>{error}</Typography>;
  if (!data) return null;

  const { counts, scores, severity, evidence, ingestion, feedback, model } = data;

  return (
    <Box>
      <Stack direction='row' justifyContent='space-between' alignItems='center' sx={{ mb:2 }}>
        <Typography variant='h4' sx={{ fontFamily:'Playfair Display, serif' }}>Metrics Dashboard</Typography>
        <Tooltip title='Refresh metrics'>
          <IconButton onClick={load}><RefreshIcon /></IconButton>
        </Tooltip>
      </Stack>
      <Grid container spacing={2}>
        <Grid item xs={12} md={4}>
          <StatCard title='Counts' accent>
            <Stack spacing={0.5}>
              {Object.entries(counts).map(([k,v]) => <Typography key={k} variant='body2'>{k}: {v}</Typography>)}
            </Stack>
          </StatCard>
        </Grid>
        <Grid item xs={12} md={4}>
          <StatCard title='Score Distribution'>
            {scores && scores.count ? (
              <Stack spacing={0.5}>
                <Typography variant='caption'>count: {scores.count}</Typography>
                {['min','p25','p50','p75','max'].map(q => (
                  <Typography key={q} variant='body2'>{q}: {(scores[q]*100).toFixed(1)}%</Typography>
                ))}
              </Stack>
            ) : <Typography variant='body2' color='text.secondary'>No scores</Typography>}
          </StatCard>
        </Grid>
        <Grid item xs={12} md={4}>
          <StatCard title='Severity Mix'>
            {severity && Object.keys(severity).length>0 ? (
              <Stack direction='row' spacing={1} flexWrap='wrap'>
                {Object.entries(severity).map(([k,v]) => <Chip key={k} size='small' label={`${k}:${v}`} color={k==='high'?'error':k==='medium'?'warning':'default'} />)}
              </Stack>
            ) : <Typography variant='body2' color='text.secondary'>No allegations</Typography>}
          </StatCard>
        </Grid>
        <Grid item xs={12} md={4}>
          <StatCard title='Evidence Weights'>
            <Typography variant='body2'>Avg: {evidence.avg_weight !== null && evidence.avg_weight !== undefined ? evidence.avg_weight.toFixed(2): '—'}</Typography>
            <Typography variant='body2'>Total: {evidence.total_weight.toFixed(2)}</Typography>
          </StatCard>
        </Grid>
        <Grid item xs={12} md={4}>
          <StatCard title='Ingestion'>
            <Typography variant='body2'>Docs: {ingestion.documents}</Typography>
            <Typography variant='body2'>Total chars: {ingestion.total_chars}</Typography>
            <Typography variant='body2'>Avg chars: {ingestion.avg_chars ? ingestion.avg_chars.toFixed(0) : '—'}</Typography>
          </StatCard>
        </Grid>
        <Grid item xs={12} md={4}>
          <StatCard title='Feedback'>
            <Typography variant='body2'>Total: {feedback.total}</Typography>
            <Typography variant='body2'>Confirm: {feedback.counts.confirm || 0}</Typography>
            <Typography variant='body2'>Reject: {feedback.counts.reject || 0}</Typography>
            <Typography variant='body2'>Uncertain: {feedback.counts.uncertain || 0}</Typography>
            <Typography variant='body2'>Conf Rate: {feedback.confirmation_rate !== null && feedback.confirmation_rate !== undefined ? (feedback.confirmation_rate*100).toFixed(1)+'%' : '—'}</Typography>
            <Typography variant='body2'>P@1 (proxy): {feedback.precision_at_1_proxy !== null && feedback.precision_at_1_proxy !== undefined ? (feedback.precision_at_1_proxy*100).toFixed(1)+'%' : '—'}</Typography>
          </StatCard>
        </Grid>
        <Grid item xs={12} md={4}>
          <StatCard title='Model'>
            <Typography variant='body2'>Backend: {model.backend}</Typography>
            <Typography variant='body2'>α default: {model.alpha_default}</Typography>
            <Typography variant='body2'>β default: {model.offense_beta_default}</Typography>
          </StatCard>
        </Grid>
        <Grid item xs={12} md={8}>
          <StatCard title='Narrative Insights'>
            <Typography variant='caption' color='text.secondary'>Ideas:</Typography>
            <ul style={{ marginTop: 4, paddingLeft: 16 }}>
              <li><Typography variant='body2'>Monitor drift: Add embedding centroid shift</Typography></li>
              <li><Typography variant='body2'>Track retrain versions & latency</Typography></li>
              <li><Typography variant='body2'>Add top false-positive suspects (needs ground truth)</Typography></li>
            </ul>
          </StatCard>
        </Grid>
      </Grid>
      <Divider sx={{ my:3 }} />
      <Typography variant='caption' color='text.secondary'>Live metrics snapshot. Extend with drift, latency, and active learning performance curves.</Typography>
    </Box>
  );
}
