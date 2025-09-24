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
import { apiFetch } from '../apiBase';

function scoreTier(score){
  if(score >= 0.6) return {label:'High', color:'error'};
  if(score >= 0.4) return {label:'Medium', color:'warning'};
  return {label:'Low', color:'success'};
}

export default function SuspectList() {
  const [suspects, setSuspects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(() => {
    setLoading(true);
    apiFetch('/api/suspects')
      .then(data => setSuspects(data))
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
  useEffect(()=> {
    apiFetch('/api/clues?limit=5')
      .then(data => setLatestClues(Array.isArray(data)? data: []))
      .catch(()=> setLatestClues([]));
  }, [suspects]);

  if (loading) return <LinearProgress />;
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box sx={{ position:'relative' }}>
      <Grid container spacing={2}>
        {suspects.map((s) => {
          const tier = scoreTier(s.score || 0);
          const pct = Math.min(100, (s.score||0)*100).toFixed(1);
          return (
            <Grid key={s.id} item xs={12} sm={6} md={4}>
              <Card component={RouterLink} to={`/suspects/${s.id}`} sx={{ textDecoration: 'none', position:'relative' }}>
                <CardHeader
                  avatar={<Avatar src={s.avatar} alt={s.name} />}
                  title={<Stack direction="row" spacing={1} alignItems="center"><span>{s.name}</span><Chip size="small" label={tier.label} color={tier.color} /></Stack>}
                  subheader={`Score: ${pct}%`}
                />
                <CardContent>
                  <LinearProgress variant="determinate" value={parseFloat(pct)} color={tier.color} sx={{ mb:1, height:8, borderRadius:4 }} />
                  <Typography variant="body2" color="text.secondary">{s.bio}</Typography>
                  <Stack direction="row" spacing={1} sx={{ mt: 1, flexWrap: 'wrap' }}>
                    {(s.tags || []).map((t) => (<Chip key={t} size="small" label={t} />))}
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>
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
      <Tooltip title="Add Clue">
        <Fab color="primary" sx={{ position:'fixed', bottom: 32, right: 32 }} onClick={() => setDialogOpen(true)}>
          <AddIcon />
        </Fab>
      </Tooltip>
      <AddClueDialog open={dialogOpen} onClose={() => setDialogOpen(false)} />
    </Box>
  );
}
