import React from 'react';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import Stack from '@mui/material/Stack';
import LinearProgress from '@mui/material/LinearProgress';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import Box from '@mui/material/Box';
import Divider from '@mui/material/Divider';
import { startIndexRefresh, startTransformerTrain, getJobStatus, listJobs, cancelJob } from '../jobsApi';
import { apiFetch } from '../apiBase';

export default function JobsPanel({ open, onClose }) {
  const [caseId, setCaseId] = React.useState('default');
  const [trainingPath, setTrainingPath] = React.useState('inputs/sample_training.json');
  const [jobs, setJobs] = React.useState([]);
  const [loading, setLoading] = React.useState(false);
  const [backend, setBackend] = React.useState('');

  // Poll in-progress jobs
  React.useEffect(() => {
    if (!open) return;
    // initial load for history (if RQ enabled, backend returns durable history; fallback returns in-memory)
    (async () => {
      try {
        setLoading(true);
        const [sys, data] = await Promise.all([
          apiFetch('/api/system').catch(()=>null),
          listJobs(50).catch(()=>({ jobs: [] }))
        ]);
        if (sys && sys.job_backend) setBackend(sys.job_backend);
        if (data && data.jobs) setJobs(data.jobs);
      } catch {}
      finally { setLoading(false); }
    })();
    const interval = setInterval(async () => {
      setJobs((prev) => {
        const inFlight = prev.filter(j => j.status === 'queued' || j.status === 'running');
        if (inFlight.length === 0) return prev;
        // kick off async polls
        inFlight.forEach(async (j) => {
          try {
            const data = await getJobStatus(j.id);
            setJobs((curr) => curr.map(item => item.id === j.id ? { ...item, ...data } : item));
          } catch (e) {
            // ignore transient errors
          }
        });
        return prev;
      });
    }, 1200);
    return () => clearInterval(interval);
  }, [open]);

  const addJob = React.useCallback((id, type) => {
    setJobs((prev) => [{ id, type, status: 'queued', progress: 0, message: null }, ...prev]);
  }, []);

  const onStartIndex = async () => {
    try {
      const id = await startIndexRefresh(caseId);
      addJob(id, 'index_refresh');
    } catch (e) {
      alert(`Failed to start index refresh: ${e.message || e}`);
    }
  };

  const onStartTrain = async () => {
    try {
      const id = await startTransformerTrain(trainingPath);
      addJob(id, 'transformer_train');
    } catch (e) {
      alert(`Failed to start training: ${e.message || e}`);
    }
  };

  const onCancel = async (id) => {
    try {
      await cancelJob(id);
      // optimistic update
      setJobs((curr) => curr.map(j => j.id === id ? { ...j, status: 'canceled' } : j));
    } catch (e) {
      alert(`Cancel failed: ${e.message || e}`);
    }
  };

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle>Background Jobs</DialogTitle>
      <DialogContent>
        <Stack spacing={2} sx={{ mt: 1 }}>
          <Typography variant="subtitle2">Start a job</Typography>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack direction="row" spacing={1} alignItems="center">
              <TextField size="small" label="Case ID" value={caseId} onChange={(e)=>setCaseId(e.target.value)} />
              <Button variant="contained" onClick={onStartIndex}>Rebuild Index</Button>
            </Stack>
          </Paper>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack direction="row" spacing={1} alignItems="center" sx={{ width: '100%' }}>
              <TextField fullWidth size="small" label="Training JSON path" value={trainingPath} onChange={(e)=>setTrainingPath(e.target.value)} />
              <Button variant="outlined" onClick={onStartTrain}>Start Training</Button>
            </Stack>
            <Typography variant="caption" color="text.secondary">Note: transformer training requires optional dependencies (torch/transformers); otherwise job may fail fast.</Typography>
          </Paper>
          <Divider/>
          <Typography variant="subtitle2">Recent jobs {backend && (<span style={{ fontSize:12, color:'#777' }}>· backend: {backend}</span>)}</Typography>
          <Stack spacing={1}>
            {jobs.length === 0 && (
              <Typography color="text.secondary">{loading ? 'Loading...' : 'No jobs yet. Start one above.'}</Typography>
            )}
            {jobs.map((j) => (
              <Paper key={j.id} variant="outlined" sx={{ p: 1.5 }}>
                <Stack spacing={0.5}>
                  <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                    <Typography variant="body2"><b>{j.type || j.func || 'job'}</b> · <Box component="span" sx={{ color:'text.secondary' }}>{j.id.slice(0,8)}</Box></Typography>
                    <Stack direction="row" spacing={1} alignItems="center">
                      <Typography variant="caption" color={j.status === 'failed' ? 'error.main' : 'text.secondary'}>{j.status}</Typography>
                      {(j.status === 'queued' || j.status === 'running') && (
                        <Button size="small" variant="text" onClick={()=>onCancel(j.id)}>Cancel</Button>
                      )}
                    </Stack>
                  </Stack>
                  <LinearProgress variant="determinate" value={typeof j.progress === 'number' ? j.progress : 0} />
                  {j.message && <Typography variant="caption" color="text.secondary">{j.message}</Typography>}
                  {j.error && <Typography variant="caption" color="error.main">{j.error}</Typography>}
                </Stack>
              </Paper>
            ))}
          </Stack>
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}
