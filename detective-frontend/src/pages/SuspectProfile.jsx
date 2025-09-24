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
import { apiFetch } from '../apiBase';

export default function SuspectProfile() {
  const { sid } = useParams();
  const [suspect, setSuspect] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

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
      {suspect.last_score !== undefined && (
        <Typography variant="caption" color="text.secondary">
          Last Score: {(suspect.last_score*100).toFixed(1)}% {suspect.last_scored_at && ` @ ${suspect.last_scored_at}`}
        </Typography>
      )}

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

      <Button variant="outlined" component={RouterLink} to="/suspects">Back to list</Button>
    </Stack>
  );
}
