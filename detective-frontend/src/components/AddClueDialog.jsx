import React, { useEffect, useState } from 'react';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import MenuItem from '@mui/material/MenuItem';
import Stack from '@mui/material/Stack';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';
import CircularProgress from '@mui/material/CircularProgress';
import { apiFetch } from '../apiBase';

export default function AddClueDialog({ open, onClose }) {
  const [clueText, setClueText] = useState('');
  const [suspects, setSuspects] = useState([]);
  const [suspectId, setSuspectId] = useState('');
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [snack, setSnack] = useState('');

  useEffect(() => {
    if (open) {
      setLoading(true);
      apiFetch('/api/suspects')
        .then(data => setSuspects(data))
        .catch(e => setError(e.message))
        .finally(() => setLoading(false));
    } else {
      setClueText('');
      setSuspectId('');
      setError(null);
    }
  }, [open]);

  const handleSubmit = () => {
    if (!clueText.trim()) { setError('Clue text required'); return; }
    setSubmitting(true);
    apiFetch('/api/clues', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: clueText.trim(), suspect_id: suspectId || null })
    })
    .then(() => {
      setSnack('Clue added');
      window.dispatchEvent(new CustomEvent('clueAdded'));
      // Optional auto-snapshot after ingestion event
      apiFetch('/api/snapshots', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ label: 'auto: clue added' }) }).catch(()=>{});
      onClose();
    })
    .catch(e => setError(e.message))
    .finally(()=> setSubmitting(false));
  };

  return (
    <>
      <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
        <DialogTitle>Add New Clue</DialogTitle>
        <DialogContent dividers>
          <Stack spacing={2} sx={{ mt: 1 }}>
            <TextField
              label="Clue Text"
              value={clueText}
              onChange={(e) => setClueText(e.target.value)}
              multiline
              minRows={3}
              fullWidth
              autoFocus
            />
            {loading ? (
              <Stack direction="row" spacing={1} alignItems="center">
                <CircularProgress size={20} />
                <span>Loading suspects...</span>
              </Stack>
            ) : (
              <TextField
                select
                label="Suspect (optional)"
                value={suspectId}
                onChange={(e) => setSuspectId(e.target.value)}
                helperText="Associate clue with a suspect (optional)"
                fullWidth
              >
                <MenuItem value="">None</MenuItem>
                {suspects.map(s => <MenuItem key={s.id} value={s.id}>{s.name}</MenuItem>)}
              </TextField>
            )}
            {error && <Alert severity="error" onClose={() => setError(null)}>{error}</Alert>}
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={onClose} disabled={submitting}>Cancel</Button>
          <Button onClick={handleSubmit} variant="contained" disabled={submitting}>{submitting ? 'Saving...' : 'Add Clue'}</Button>
        </DialogActions>
      </Dialog>
      <Snackbar
        open={!!snack}
        autoHideDuration={3000}
        onClose={() => setSnack('')}
        message={snack}
      />
    </>
  );
}
