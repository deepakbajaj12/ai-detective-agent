import React from 'react';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import Button from '@mui/material/Button';
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import LinearProgress from '@mui/material/LinearProgress';
import TextField from '@mui/material/TextField';
import { apiFetch } from '../apiBase';

export default function CaseAnalysis(){
  const [style, setStyle] = React.useState('brief');
  const [caseId, setCaseId] = React.useState('default');
  const [loading, setLoading] = React.useState(false);
  const [report, setReport] = React.useState(null);
  const run = async () => {
    setLoading(true);
    setReport(null);
    try {
      const res = await apiFetch('/api/analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ case_id: caseId, style })
      });
      setReport(res);
    } catch (e) {
      setReport({ error: e.message || String(e) });
    } finally {
      setLoading(false);
    }
  };
  return (
    <Stack spacing={2}>
      <Typography variant="h4" sx={{ fontFamily:'Playfair Display, serif' }}>Case Analysis</Typography>
      <Stack direction={{ xs:'column', sm:'row' }} spacing={2} alignItems="center">
        <TextField label="Case ID" size="small" value={caseId} onChange={e=>setCaseId(e.target.value)} />
        <ToggleButtonGroup size="small" value={style} exclusive onChange={(e,v)=> v && setStyle(v)}>
          <ToggleButton value="brief">Brief</ToggleButton>
          <ToggleButton value="detailed">Detailed</ToggleButton>
        </ToggleButtonGroup>
        <Button variant="contained" onClick={run} disabled={loading}>Generate</Button>
      </Stack>
      {loading && <LinearProgress />}
      {report && report.error && <Typography color="error">{report.error}</Typography>}
      {report && !report.error && (
        <Paper variant="outlined" sx={{ p:2, whiteSpace:'pre-wrap' }}>
          <Typography variant="caption" color="text.secondary">Backend: {report.backend} · Style: {report.style}</Typography>
          <Typography variant="body2" sx={{ mt:1 }}>{report.report}</Typography>
        </Paper>
      )}
    </Stack>
  );
}
